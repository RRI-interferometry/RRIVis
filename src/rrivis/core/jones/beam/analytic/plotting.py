"""Analytic beam pattern visualization functions.

Provides functions to plot 1D beam cuts, 2D beam images, multi-config
comparisons, and feed illumination patterns. All functions return a
:class:`matplotlib.figure.Figure` and never call ``plt.show()``, making
them suitable for both interactive Jupyter use and scripted pipelines.

Examples
--------
>>> from rrivis.core.jones.beam.analytic.plotting import plot_beam_pattern
>>> fig = plot_beam_pattern(diameter=14.0, frequency=150e6, taper="gaussian")

>>> from rrivis.core.jones.beam.analytic.plotting import plot_beam_comparison
>>> fig = plot_beam_comparison(
...     14.0,
...     150e6,
...     [
...         {"taper": "uniform", "label": "Uniform"},
...         {"taper": "gaussian", "label": "Gaussian 10 dB"},
...     ],
... )
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from rrivis.core.jones.beam.analytic.composed import compute_aperture_beam

# Speed of light in vacuum (m/s)
_C = 299_792_458.0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_first_null(voltage: np.ndarray, theta_deg: np.ndarray) -> float | None:
    """Find the angle of the first null (sign change) in the voltage pattern.

    Parameters
    ----------
    voltage : np.ndarray
        Voltage pattern values (may go negative in sidelobes).
    theta_deg : np.ndarray
        Corresponding angles in degrees.

    Returns
    -------
    float or None
        Angle of first null in degrees, or None if not found.
    """
    # Look for first sign change after boresight
    for i in range(1, len(voltage)):
        if voltage[i - 1] * voltage[i] < 0:
            # Linear interpolation to find zero crossing
            v0, v1 = voltage[i - 1], voltage[i]
            t0, t1 = theta_deg[i - 1], theta_deg[i]
            return float(t0 - v0 * (t1 - t0) / (v1 - v0))
    return None


def _find_first_sidelobe(
    power_db: np.ndarray,
    theta_deg: np.ndarray,
    first_null_idx: int,
) -> tuple[float, float] | None:
    """Find the first sidelobe peak after the first null.

    Parameters
    ----------
    power_db : np.ndarray
        Power pattern in dB.
    theta_deg : np.ndarray
        Corresponding angles in degrees.
    first_null_idx : int
        Index of the first null in the arrays.

    Returns
    -------
    tuple[float, float] or None
        ``(angle_deg, level_dB)`` of the first sidelobe peak, or None.
    """
    if first_null_idx >= len(power_db) - 2:
        return None
    # Search for local maximum after the first null
    segment = power_db[first_null_idx:]
    for i in range(1, len(segment) - 1):
        if segment[i] >= segment[i - 1] and segment[i] >= segment[i + 1]:
            idx = first_null_idx + i
            return (float(theta_deg[idx]), float(power_db[idx]))
    return None


def _format_angle(angle_deg: float) -> str:
    """Format an angle as arcmin or degrees depending on magnitude.

    Parameters
    ----------
    angle_deg : float
        Angle in degrees.

    Returns
    -------
    str
        Formatted angle string.
    """
    if angle_deg < 1.0:
        return f"{angle_deg * 60:.1f} arcmin"
    return f"{angle_deg:.2f} deg"


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------


def plot_beam_pattern(
    diameter: float,
    frequency: float,
    aperture_shape: str = "circular",
    taper: str = "gaussian",
    edge_taper_dB: float = 10.0,
    feed_model: str = "none",
    feed_computation: str = "analytical",
    feed_params: dict | None = None,
    reflector_type: str = "prime_focus",
    magnification: float = 1.0,
    aperture_params: dict | None = None,
    max_angle_deg: float = 10.0,
    n_points: int = 1000,
    show_voltage: bool = False,
    show_hpbw: bool = True,
    show_first_null: bool = True,
    show_sidelobe_level: bool = True,
    db_min: float = -60.0,
    title: str | None = None,
    ax: object | None = None,
) -> Figure:
    """Plot a 1D beam pattern cut in dB vs angle.

    Generates an angular grid, computes the beam via
    :func:`~rrivis.core.jones.beam.analytic.composed.compute_aperture_beam`,
    and plots the power pattern in dB. Optionally annotates HPBW, first
    null, and first sidelobe level.

    Parameters
    ----------
    diameter : float
        Antenna diameter in metres.
    frequency : float
        Frequency in Hz.
    aperture_shape : str
        Aperture geometry (``'circular'``, ``'rectangular'``, ``'elliptical'``).
    taper : str
        Illumination taper name.
    edge_taper_dB : float
        Edge taper in dB.
    feed_model : str
        Feed pattern model name.
    feed_computation : str
        ``'analytical'`` or ``'numerical'``.
    feed_params : dict or None
        Feed-specific parameters.
    reflector_type : str
        Reflector geometry.
    magnification : float
        Cassegrain magnification.
    aperture_params : dict or None
        Aperture-specific parameters.
    max_angle_deg : float
        Maximum angle to plot in degrees.
    n_points : int
        Number of angular sample points.
    show_voltage : bool
        If True, add a secondary y-axis with linear voltage scale.
    show_hpbw : bool
        If True, annotate the HPBW at -3 dB.
    show_first_null : bool
        If True, mark the first null.
    show_sidelobe_level : bool
        If True, mark the first sidelobe peak.
    db_min : float
        Minimum dB level for the y-axis.
    title : str or None
        Plot title. Auto-generated if None.
    ax : matplotlib Axes or None
        If provided, plot on this axes. Otherwise create a new figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.
    """
    import matplotlib.pyplot as plt

    theta_deg = np.linspace(0.0, max_angle_deg, n_points)
    theta_rad = np.deg2rad(theta_deg)

    # Compute beam
    jones = compute_aperture_beam(
        theta=theta_rad,
        phi=None,
        frequency=frequency,
        diameter=diameter,
        aperture_shape=aperture_shape,
        taper=taper,
        edge_taper_dB=edge_taper_dB,
        feed_model=feed_model,
        feed_computation=feed_computation,
        feed_params=feed_params,
        reflector_type=reflector_type,
        magnification=magnification,
        aperture_params=aperture_params,
    )

    # Extract voltage from diagonal Jones matrix
    voltage = np.real(jones[:, 0, 0])
    voltage_abs = np.abs(voltage)

    # Power in dB
    with np.errstate(divide="ignore"):
        power_db = 20.0 * np.log10(np.maximum(voltage_abs, 1e-30))

    # Create axes
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    ax.plot(theta_deg, power_db, "b-", linewidth=1.5, label="Power pattern")
    ax.set_xlabel("Angle from boresight (degrees)")
    ax.set_ylabel("Power (dB)")
    ax.set_ylim(db_min, 3.0)
    ax.set_xlim(0, max_angle_deg)
    ax.grid(True, alpha=0.3)

    if title is None:
        freq_mhz = frequency / 1e6
        title = f"Beam pattern: D={diameter}m, f={freq_mhz:.1f} MHz, {taper} taper"
    ax.set_title(title)

    # HPBW annotation
    if show_hpbw:
        from rrivis.core.jones.beam.analytic.numerical_hpbw import (
            compute_hpbw_numerical,
        )
        from rrivis.core.jones.beam.analytic.taper import TAPER_FUNCTIONS

        taper_func = TAPER_FUNCTIONS.get(taper)
        if taper_func is not None:
            beam_kwargs = {}
            if taper in ("gaussian", "parabolic", "parabolic_squared"):
                beam_kwargs["edge_taper_dB"] = edge_taper_dB
            hpbw_rad = compute_hpbw_numerical(
                taper_func, frequency, diameter, **beam_kwargs
            )
            hpbw_deg = float(np.rad2deg(hpbw_rad[0]))
            half_hpbw_deg = hpbw_deg / 2.0
            ax.axhline(y=-3.0, color="r", linestyle="--", alpha=0.5)
            ax.axvline(x=half_hpbw_deg, color="r", linestyle="--", alpha=0.5)
            ax.annotate(
                f"HPBW = {_format_angle(hpbw_deg)}",
                xy=(half_hpbw_deg, -3.0),
                xytext=(half_hpbw_deg + max_angle_deg * 0.05, -3.0 - 5.0),
                fontsize=9,
                color="red",
                arrowprops={"arrowstyle": "->", "color": "red", "alpha": 0.7},
            )

    # First null annotation
    if show_first_null:
        null_deg = _find_first_null(voltage, theta_deg)
        if null_deg is not None:
            ax.axvline(x=null_deg, color="green", linestyle=":", alpha=0.5)
            ax.annotate(
                f"1st null: {_format_angle(null_deg)}",
                xy=(null_deg, db_min + 5),
                fontsize=8,
                color="green",
            )

    # First sidelobe annotation
    if show_sidelobe_level:
        # Find first null index for sidelobe search
        null_idx = None
        for i in range(1, len(voltage)):
            if voltage[i - 1] * voltage[i] < 0:
                null_idx = i
                break
        if null_idx is not None:
            sl = _find_first_sidelobe(power_db, theta_deg, null_idx)
            if sl is not None:
                sl_angle, sl_level = sl
                ax.plot(sl_angle, sl_level, "m^", markersize=8)
                ax.annotate(
                    f"1st SL: {sl_level:.1f} dB",
                    xy=(sl_angle, sl_level),
                    xytext=(sl_angle + max_angle_deg * 0.03, sl_level + 3.0),
                    fontsize=8,
                    color="purple",
                    arrowprops={"arrowstyle": "->", "color": "purple", "alpha": 0.7},
                )

    # Secondary voltage axis
    if show_voltage:
        ax2 = ax.twinx()
        ax2.plot(theta_deg, voltage_abs, "c-", alpha=0.3, linewidth=0.8)
        ax2.set_ylabel("Voltage (linear)", color="cyan")
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis="y", labelcolor="cyan")

    if created_fig:
        fig.tight_layout()

    return fig


def plot_beam_comparison(
    diameter: float,
    frequency: float,
    configs: list[dict],
    max_angle_deg: float = 10.0,
    n_points: int = 1000,
    db_min: float = -60.0,
    title: str | None = None,
) -> Figure:
    """Overlay multiple beam configurations on the same axes.

    Each entry in *configs* is a dict of beam parameters (matching the
    keyword arguments of :func:`compute_aperture_beam`) plus an optional
    ``"label"`` key for the legend.

    Parameters
    ----------
    diameter : float
        Default antenna diameter in metres (can be overridden per config).
    frequency : float
        Default frequency in Hz (can be overridden per config).
    configs : list[dict]
        List of beam configuration dicts. Each may contain keys accepted
        by :func:`compute_aperture_beam` and an optional ``"label"``.
    max_angle_deg : float
        Maximum angle to plot in degrees.
    n_points : int
        Number of angular sample points.
    db_min : float
        Minimum dB level for the y-axis.
    title : str or None
        Plot title. Auto-generated if None.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the comparison plot.
    """
    import matplotlib.pyplot as plt

    theta_deg = np.linspace(0.0, max_angle_deg, n_points)
    theta_rad = np.deg2rad(theta_deg)

    fig, ax = plt.subplots(figsize=(10, 6))

    beam_param_keys = {
        "aperture_shape",
        "taper",
        "edge_taper_dB",
        "feed_model",
        "feed_computation",
        "feed_params",
        "reflector_type",
        "magnification",
        "aperture_params",
    }

    for cfg in configs:
        label = cfg.get("label", None)
        d = cfg.get("diameter", diameter)
        f = cfg.get("frequency", frequency)

        beam_kwargs = {k: v for k, v in cfg.items() if k in beam_param_keys}

        jones = compute_aperture_beam(
            theta=theta_rad,
            phi=None,
            frequency=f,
            diameter=d,
            **beam_kwargs,
        )

        voltage_abs = np.abs(jones[:, 0, 0])
        with np.errstate(divide="ignore"):
            power_db = 20.0 * np.log10(np.maximum(voltage_abs, 1e-30))

        ax.plot(theta_deg, power_db, linewidth=1.5, label=label)

    ax.set_xlabel("Angle from boresight (degrees)")
    ax.set_ylabel("Power (dB)")
    ax.set_ylim(db_min, 3.0)
    ax.set_xlim(0, max_angle_deg)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if title is None:
        freq_mhz = frequency / 1e6
        title = f"Beam comparison: D={diameter}m, f={freq_mhz:.1f} MHz"
    ax.set_title(title)

    fig.tight_layout()
    return fig


def plot_beam_2d(
    diameter: float,
    frequency: float,
    aperture_shape: str = "circular",
    taper: str = "gaussian",
    edge_taper_dB: float = 10.0,
    feed_model: str = "none",
    feed_computation: str = "analytical",
    feed_params: dict | None = None,
    reflector_type: str = "prime_focus",
    magnification: float = 1.0,
    aperture_params: dict | None = None,
    max_angle_deg: float = 5.0,
    n_points: int = 201,
    db_min: float = -40.0,
    colormap: str = "inferno",
    title: str | None = None,
) -> Figure:
    """Plot a 2D image of the beam power pattern.

    Creates a Cartesian grid in ``(x_deg, y_deg)``, converts to
    ``(theta, phi)`` polar coordinates, evaluates the beam, and displays
    the result as a colour map in dB.

    Parameters
    ----------
    diameter : float
        Antenna diameter in metres.
    frequency : float
        Frequency in Hz.
    aperture_shape : str
        Aperture geometry.
    taper : str
        Illumination taper name.
    edge_taper_dB : float
        Edge taper in dB.
    feed_model : str
        Feed pattern model name.
    feed_computation : str
        ``'analytical'`` or ``'numerical'``.
    feed_params : dict or None
        Feed-specific parameters.
    reflector_type : str
        Reflector geometry.
    magnification : float
        Cassegrain magnification.
    aperture_params : dict or None
        Aperture-specific parameters.
    max_angle_deg : float
        Maximum angle from boresight in degrees (plot extent).
    n_points : int
        Grid resolution (n_points x n_points).
    db_min : float
        Minimum dB level for the colour scale.
    colormap : str
        Matplotlib colormap name.
    title : str or None
        Plot title. Auto-generated if None.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the 2D beam plot.
    """
    import matplotlib.pyplot as plt

    x_deg = np.linspace(-max_angle_deg, max_angle_deg, n_points)
    y_deg = np.linspace(-max_angle_deg, max_angle_deg, n_points)
    xx, yy = np.meshgrid(x_deg, y_deg)

    # Convert to polar (theta, phi)
    theta_rad = np.deg2rad(np.sqrt(xx**2 + yy**2))
    phi_rad = np.arctan2(yy, xx)

    # Flatten for compute_aperture_beam
    theta_flat = theta_rad.ravel()
    phi_flat = phi_rad.ravel()

    jones = compute_aperture_beam(
        theta=theta_flat,
        phi=phi_flat,
        frequency=frequency,
        diameter=diameter,
        aperture_shape=aperture_shape,
        taper=taper,
        edge_taper_dB=edge_taper_dB,
        feed_model=feed_model,
        feed_computation=feed_computation,
        feed_params=feed_params,
        reflector_type=reflector_type,
        magnification=magnification,
        aperture_params=aperture_params,
    )

    voltage_abs = np.abs(jones[:, 0, 0]).reshape(n_points, n_points)
    with np.errstate(divide="ignore"):
        power_db = 20.0 * np.log10(np.maximum(voltage_abs, 1e-30))

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(
        x_deg,
        y_deg,
        power_db,
        vmin=db_min,
        vmax=0.0,
        cmap=colormap,
        shading="auto",
    )
    fig.colorbar(im, ax=ax, label="Power (dB)")
    ax.set_xlabel("Angle x (degrees)")
    ax.set_ylabel("Angle y (degrees)")
    ax.set_aspect("equal")

    if title is None:
        freq_mhz = frequency / 1e6
        title = f"2D beam: D={diameter}m, f={freq_mhz:.1f} MHz, {aperture_shape}"
    ax.set_title(title)

    fig.tight_layout()
    return fig


def plot_feed_illumination(
    feed_model: str = "corrugated_horn",
    feed_params: dict | None = None,
    dish_diameter: float = 14.0,
    focal_ratio: float = 0.4,
    reflector_type: str = "prime_focus",
    magnification: float = 1.0,
    n_points: int = 500,
    title: str | None = None,
) -> Figure:
    """Plot the feed illumination pattern across the aperture.

    Uses :func:`~rrivis.core.jones.beam.analytic.feed.feed_to_taper` to
    compute the illumination weight as a function of normalised radial
    position, and marks the edge taper level.

    Parameters
    ----------
    feed_model : str
        Feed pattern model name.
    feed_params : dict or None
        Feed-specific parameters.
    dish_diameter : float
        Dish diameter in metres.
    focal_ratio : float
        Focal length / diameter ratio (F/D).
    reflector_type : str
        Reflector geometry.
    magnification : float
        Cassegrain magnification.
    n_points : int
        Number of radial sample points.
    title : str or None
        Plot title. Auto-generated if None.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the illumination plot.
    """
    import matplotlib.pyplot as plt

    from rrivis.core.jones.beam.analytic.feed import (
        compute_edge_taper_from_feed,
        feed_to_taper,
    )

    focal_length = focal_ratio * dish_diameter
    aperture_radius = dish_diameter / 2.0
    rho_over_a = np.linspace(0.0, 1.0, n_points)

    illumination = feed_to_taper(
        rho_over_a=rho_over_a,
        focal_length=focal_length,
        aperture_radius=aperture_radius,
        feed_model=feed_model,
        feed_params=feed_params,
        reflector_type=reflector_type,
        magnification=magnification,
    )

    # Normalize to peak
    peak = np.max(np.abs(illumination))
    if peak > 0:
        illumination_norm = illumination / peak
    else:
        illumination_norm = illumination

    # Convert to dB
    with np.errstate(divide="ignore"):
        illumination_db = 20.0 * np.log10(np.maximum(np.abs(illumination_norm), 1e-30))

    # Edge taper
    edge_taper = compute_edge_taper_from_feed(
        feed_model=feed_model,
        dish_diameter=dish_diameter,
        focal_length=focal_length,
        feed_params=feed_params,
        reflector_type=reflector_type,
        magnification=magnification,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Linear plot
    ax1.plot(rho_over_a, illumination_norm, "b-", linewidth=1.5)
    ax1.axhline(
        y=10.0 ** (-edge_taper / 20.0),
        color="r",
        linestyle="--",
        alpha=0.7,
        label=f"Edge taper: {edge_taper:.1f} dB",
    )
    ax1.set_xlabel("Normalised radial position (rho/a)")
    ax1.set_ylabel("Illumination (linear)")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # dB plot
    ax2.plot(rho_over_a, illumination_db, "b-", linewidth=1.5)
    ax2.axhline(
        y=-edge_taper,
        color="r",
        linestyle="--",
        alpha=0.7,
        label=f"Edge taper: {edge_taper:.1f} dB",
    )
    ax2.set_xlabel("Normalised radial position (rho/a)")
    ax2.set_ylabel("Illumination (dB)")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-30, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    if title is None:
        title = f"Feed illumination: {feed_model}, F/D={focal_ratio}"
    fig.suptitle(title)

    fig.tight_layout()
    return fig


__all__ = [
    "plot_beam_pattern",
    "plot_beam_comparison",
    "plot_beam_2d",
    "plot_feed_illumination",
]
