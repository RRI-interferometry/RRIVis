import pygdsm
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import TimeDelta
import astropy.units as au
from bokeh.plotting import figure, show
from bokeh.models import (
    ColorBar,
    LogColorMapper,
    FixedTicker,
    HoverTool,
    ColumnDataSource,
)
from bokeh.io import show
from bokeh.io import output_file, save
from bokeh.layouts import gridplot
import os
from bokeh.resources import CDN


def diffused_sky_model(
    location,
    obstime_start,
    total_seconds,
    frequency=76,
    fov_radius_deg=5,
    gleam_sources=None,
    save_simulation_data=False,
    folder_path=None,
    open_in_browser=True,
):
    """
    Generates the Global Sky Model data at a given frequency and time,
    and plots the model at each hour up to the specified total seconds.

    Parameters:
    - location (EarthLocation): Observer's location.
    - obstime_start (Time): Start time of the observation.
    - total_seconds (float): Total duration in seconds for plotting.
    - frequency (float): Frequency in MHz for the sky model.
    - fov_radius_deg (float): Radius of the field of view in degrees.
    - gleam_sources (list): List of GLEAM sources to overlay on the plot.
    - save_simulation_data (bool): If True, saves the grid of plots to the specified folder.
    - folder_path (str): Folder path to save the grid of plots if save_simulation_data is True.
    """

    # Generate the Global Sky Model at the specified frequency
    gsm_2008 = pygdsm.GlobalSkyModel(freq_unit="MHz")
    sky_map = gsm_2008.generate(frequency)

    # Convert from Galactic to Equatorial coordinates
    rotator = hp.Rotator(coord=["G", "C"])
    equatorial = rotator.rotate_map_pixel(sky_map)

    # Project the equatorial map into a 2D Cartesian grid
    projected_map = hp.cartview(
        equatorial,
        norm="hist",
        coord="C",
        flip="geo",
        title="",
        unit="Brightness",
        return_projected_map=True,
        notext=True,
    )
    plt.close()

    # Normalize the map data for visualization
    min_value = np.nanmin(projected_map)
    max_value = np.nanmax(projected_map)

    # List to collect all Bokeh plots
    plots = []

    # Declination range for HERA
    hera_dec = -30.7
    fov_radius_deg = 5  # Radius of the field of view in degrees

    # Draw static dotted declination lines across the entire map
    def add_declination_lines(p):
        p.line(
            x=[-180, 180],
            y=[hera_dec + fov_radius_deg, hera_dec + fov_radius_deg],
            line_dash="dotted",
            color="white",
            alpha=0.7,
            line_width=2,
        )
        p.line(
            x=[-180, 180],
            y=[hera_dec - fov_radius_deg, hera_dec - fov_radius_deg],
            line_dash="dotted",
            color="white",
            alpha=0.7,
            line_width=2,
        )

    # Generate the x, y grid
    n_y, n_x = projected_map.shape  # Dimensions of the projected map
    x = np.linspace(-180, 180, n_x)
    y = np.linspace(-90, 90, n_y)

    # Define the time points (one per hour)
    time_points = np.arange(0, total_seconds, 3600)  # Every hour up to total_seconds

    # Iterate through each time point
    for idx, current_time in enumerate(time_points):
        # Update observation time
        obstime = obstime_start + TimeDelta(current_time, format="sec")

        # Define the observer's zenith
        zenith = SkyCoord(
            alt=90 * au.deg,
            az=0 * au.deg,
            frame=AltAz(obstime=obstime, location=location),
        )
        zenith_radec = zenith.transform_to("icrs")
        ra_center = zenith_radec.ra.deg % 360
        if ra_center > 180:
            ra_center -= 360

        # Compute RA bounds for the field of view
        ra_range_highlight = np.linspace(
            ra_center - fov_radius_deg, ra_center + fov_radius_deg, 100
        )
        dec_upper = location.lat.deg + fov_radius_deg
        dec_lower = location.lat.deg - fov_radius_deg

        color_mapper = LogColorMapper(
            palette="Inferno256", low=min_value, high=max_value
        )

        # Create the Bokeh figure
        p = figure(
            title=f"Sky Model on {obstime.to_datetime().strftime('%Y-%m-%d %H:%M:%S')}",
            x_range=(-180, 180),
            y_range=(-90, 90),
            x_axis_label="RA (°)",
            y_axis_label="Dec (°)",
            aspect_ratio=2,
        )

        # Plot the background sky map
        image = p.image(
            image=[projected_map],
            x=-180,
            y=-90,
            dw=360,
            dh=180,
            color_mapper=color_mapper,
        )

        # Add hover tool
        hover_tool = HoverTool(
            tooltips=[
                ("RA", "$x°"),  # RA value
                ("Dec", "$y°"),  # Dec value
                ("Brightness", "@image"),  # Brightness value from the image
            ],
            formatters={
                "$x": "printf",  # RA formatter
                "$y": "printf",  # Dec formatter
            },
            mode="mouse",
            renderers=[image],
            attachment="right",
        )
        p.add_tools(hover_tool)

        # Add declination lines
        add_declination_lines(p)

        # Add the highlighted observable area
        p.patch(
            x=list(ra_range_highlight) + list(ra_range_highlight[::-1]),
            y=[dec_lower] * len(ra_range_highlight)
            + [dec_upper] * len(ra_range_highlight),
            color="red",
            fill_alpha=0.3,
            line_alpha=1.0,
        )

        # Add a colorbar
        color_bar = ColorBar(
            color_mapper=color_mapper,
            label_standoff=12,
            border_line_color=None,
            location=(0, 0),
        )
        p.add_layout(color_bar, "right")

        # Add custom ticks for RA and Dec
        major_ticks_ra = list(range(-180, 181, 30))  # RA ticks every 30 degrees
        major_ticks_dec = list(range(-90, 91, 30))  # Dec ticks every 30 degrees
        p.xaxis.ticker = FixedTicker(ticks=major_ticks_ra)
        p.yaxis.ticker = FixedTicker(ticks=major_ticks_dec)
        p.xaxis.major_label_overrides = {tick: f"{tick}°" for tick in major_ticks_ra}
        p.yaxis.major_label_overrides = {tick: f"{tick}°" for tick in major_ticks_dec}

        # Add GLEAM sources
        if gleam_sources:
            source_data = {
                "ra": [],
                "dec": [],
                "flux": [],
            }
            for source in gleam_sources:
                ra = source["coords"].ra.deg
                dec = source["coords"].dec.deg
                flux = source["flux"]

                if ra > 180:
                    ra -= 360

                source_data["ra"].append(ra)
                source_data["dec"].append(dec)
                source_data["flux"].append(flux)

            source_cds = ColumnDataSource(data=source_data)

            p.scatter(
                x="ra",
                y="dec",
                size=3,
                source=source_cds,
                color="#39FF14",
                alpha=0.8,
            )

            # Add hover tool for the sources
            hover_tool_sources = HoverTool(
                tooltips=[
                    ("RA", "@ra°"),
                    ("Dec", "@dec°"),
                    ("Flux", "@flux Jy"),
                ],
                mode="mouse",
                renderers=[p.renderers[-1]],
                attachment="left",  # Attach hover to the last renderer (circle layer)
            )
            p.add_tools(hover_tool_sources)

        # Collect the plot in the list
        plots.append(p)

    # Arrange plots in two columns
    grid = gridplot(children=plots, ncols=2)

    # Save the grid if required
    if save_simulation_data and folder_path:
        file_path = os.path.join(folder_path, "gsm_plots_grid.html")
        output_file(file_path, title="Global Sky Model Plots")
        save(grid, filename=file_path, resources=CDN, title="Global Sky Model Plots")
        print(f"GSM plot grid saved to {file_path}")

    # Show all plots in a single browser tab if requested
    if open_in_browser:
        # Set the page title for the browser tab when showing the grid
        from bokeh.io import reset_output
        import tempfile

        if folder_path:
            output_file_path = os.path.join(folder_path, "gsm_plots_grid.html")
        else:
            # Create a temporary directory for the output file
            temp_dir = tempfile.mkdtemp(prefix="rrivis_")
            output_file_path = os.path.join(temp_dir, "gsm_plots_grid.html")
        output_file(output_file_path, title="Global Sky Model Plots")
        try:
            show(grid)
        finally:
            reset_output()
