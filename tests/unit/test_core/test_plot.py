# tests/test_plot.py
import numpy as np
from astropy.time import Time
from bokeh.layouts import LayoutDOM

from rrivis.visualization.bokeh_plots import plot_heatmaps, plot_visibility


def test_plot_visibility():
    moduli_over_time = {
        (0, 1): np.array([[1], [2]]),  # 2 rows to match time_points
        (0, 2): np.array([[1], [1]]),  # 2 rows to match time_points
        (1, 2): np.array([[2], [2]]),  # 2 rows to match time_points
    }
    phases_over_time = {
        (0, 1): np.array([[0], [np.pi]]),  # 2 rows to match time_points
        (0, 2): np.array([[np.pi], [np.pi]]),  # 2 rows to match time_points
        (1, 2): np.array([[0], [0]]),  # 2 rows to match time_points
    }
    baselines = {(0, 1): [14, 0, 0], (0, 2): [28, 0, 0], (1, 2): [14, 0, 0]}
    # Create MJD time points
    mjd_time_points = Time("2023-01-01T00:00:00", scale="utc").mjd + np.array(
        [0, 1 / 24]
    )  # 0 and 1 hour
    freqs = np.array([1e8, 2e8])
    total_seconds = 3600

    fig = plot_visibility(
        moduli_over_time,
        phases_over_time,
        baselines,
        mjd_time_points,
        freqs,
        total_seconds,
    )
    assert isinstance(fig, LayoutDOM)


def test_plot_heatmaps():
    moduli_over_time = {
        (0, 1): np.array([[1], [2]]),
        (0, 2): np.array([[1], [1]]),
        (1, 2): np.array([[2], [2]]),
    }
    phases_over_time = {
        (0, 1): np.array([[0], [np.pi]]),
        (0, 2): np.array([[np.pi], [np.pi]]),
        (1, 2): np.array([[0], [0]]),
    }
    baselines = {(0, 1): [14, 0, 0], (0, 2): [28, 0, 0], (1, 2): [14, 0, 0]}
    freqs = np.array([1e8, 2e8])
    # Create MJD time points
    mjd_time_points = Time("2023-01-01T00:00:00", scale="utc").mjd + np.array(
        [0, 1 / 24]
    )  # 0 and 1 hour
    total_seconds = 3600

    fig = plot_heatmaps(
        moduli_over_time,
        phases_over_time,
        baselines,
        freqs,
        total_seconds,
        mjd_time_points,
    )
    assert isinstance(fig, LayoutDOM)
