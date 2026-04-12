# rrivis/core/sky/plotter.py
"""Standalone plotting accessor for SkyModel.

Provides matplotlib-based visualization methods for point-source catalogs
and HEALPix diffuse maps. All public methods return a
:class:`matplotlib.figure.Figure` and never call ``plt.show()``, following
the convention established by the beam plotting module.

Usage::

    sky = SkyModel.from_catalog("gleam", precision=precision)
    fig = sky.plot.source_positions()
    fig = sky.plot.flux_histogram()
    fig = sky.plot("auto")  # dispatcher via __call__
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .model import SkyFormat

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .model import SkyModel

logger = logging.getLogger(__name__)

# Colorbar labels by quantity
_COLORBAR_LABELS = {
    "flux": "Flux density (Jy)",
    "flux_log": "log$_{10}$(Flux density / Jy)",
    "spectral_index": "Spectral index",
    "stokes_q": "Stokes Q (Jy)",
    "stokes_u": "Stokes U (Jy)",
    "stokes_v": "Stokes V (Jy)",
    "tb": "T$_b$ (K)",
    "tb_log": "log$_{10}$(T$_b$ / K)",
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _freq_label(freq_hz: float) -> str:
    """Format a frequency in Hz for display."""
    if freq_hz >= 1e9:
        return f"{freq_hz / 1e9:.1f} GHz"
    return f"{freq_hz / 1e6:.1f} MHz"


# Mapping from projection name to healpy function name
_HEALPY_PROJECTIONS = {
    "mollweide": "mollview",
    "cartesian": "cartview",
    "gnomonic": "gnomview",
    "orthographic": "orthview",
}

# Valid matplotlib sky projections for scatter plots
_MPL_PROJECTIONS = {"mollweide", "aitoff", "hammer", "cartesian"}

# Axis labels per coordinate system
_COORD_LABELS = {
    None: ("RA", "Dec"),
    "C": ("RA", "Dec"),
    "G": ("Galactic $l$", "Galactic $b$"),
    "E": ("Ecliptic lon", "Ecliptic lat"),
}


class SkyPlotter:
    """Plotting accessor for SkyModel. Access via ``sky_model.plot``."""

    def __init__(self, sky_model: SkyModel) -> None:
        self._sky = sky_model

    # -- helpers ----------------------------------------------------------

    def _auto_title(self, suffix: str, frequency: float | None = None) -> str:
        """Build a plot title from model metadata.

        Parameters
        ----------
        suffix : str
            Descriptive suffix, e.g. "Source Positions".
        frequency : float, optional
            Frequency in Hz to include in the title.
        """
        parts: list[str] = []
        if self._sky.model_name:
            parts.append(self._sky.model_name.upper().replace("_", " "))
        if frequency is not None:
            parts.append(_freq_label(frequency))
        if self._sky.healpix_nside is not None and self._sky.healpix_maps is not None:
            parts.append(f"nside={self._sky.healpix_nside}")
        meta = " — ".join(parts) if parts else "Sky Model"
        return f"{meta} — {suffix}"

    def _validate_plot_mode(self, required: str) -> None:
        """Raise ValueError if the model lacks the required data."""
        if required == "point_sources":
            if not self._sky.has_point_sources:
                raise ValueError(
                    f"Plot requires point-source data, but this SkyModel "
                    f"(model='{self._sky.model_name}') has mode='{self._sky.mode.value}'. "
                    f"Load a point-source catalog or call materialize_point_sources(..., lossy=True) first."
                )
        elif required == "healpix":
            if self._sky.healpix_maps is None:
                raise ValueError(
                    f"Plot requires HEALPix maps, but this SkyModel "
                    f"(model='{self._sky.model_name}') has mode='{self._sky.mode.value}'. "
                    f"Load a diffuse model or call materialize_healpix() first."
                )

    def _resolve_plot_frequency(self, frequency: float | None) -> float:
        """Pick the best frequency for plotting from what is available.

        Parameters
        ----------
        frequency : float or None
            Requested frequency in Hz.  If None, falls back to
            ``self._sky.reference_frequency`` then the first observation frequency.

        Returns
        -------
        float
            Resolved frequency in Hz.

        Raises
        ------
        ValueError
            If no frequency can be determined.
        """
        if frequency is not None:
            return float(frequency)
        if self._sky.reference_frequency is not None:
            return float(self._sky.reference_frequency)
        if (
            self._sky.observation_frequencies is not None
            and len(self._sky.observation_frequencies) > 0
        ):
            return float(self._sky.observation_frequencies[0])
        raise ValueError("No frequency specified and none available in the model.")

    def _get_color_data(self, color_by: str) -> tuple[np.ndarray, str]:
        """Return (values, colorbar_label) for a point-source color parameter.

        Parameters
        ----------
        color_by : str
            One of ``"flux"``, ``"spectral_index"``, ``"stokes_q"``,
            ``"stokes_u"``, ``"stokes_v"``.

        Returns
        -------
        values : np.ndarray
        label : str
        """
        mapping = {
            "flux": (self._sky.flux, _COLORBAR_LABELS["flux"]),
            "spectral_index": (
                self._sky.spectral_index,
                _COLORBAR_LABELS["spectral_index"],
            ),
            "stokes_q": (self._sky.stokes_q, _COLORBAR_LABELS["stokes_q"]),
            "stokes_u": (self._sky.stokes_u, _COLORBAR_LABELS["stokes_u"]),
            "stokes_v": (self._sky.stokes_v, _COLORBAR_LABELS["stokes_v"]),
        }
        if color_by not in mapping:
            raise ValueError(
                f"Unknown color_by='{color_by}'. Choose from: {list(mapping.keys())}"
            )
        values, label = mapping[color_by]
        if values is None:
            raise ValueError(
                f"color_by='{color_by}' requested but the data is not available "
                f"in this SkyModel."
            )
        return values, label

    def _rotate_coords(
        self, ra_rad: np.ndarray, dec_rad: np.ndarray, coord: str | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rotate point-source coordinates from Equatorial to another frame.

        Parameters
        ----------
        ra_rad, dec_rad : np.ndarray
            Source coordinates in radians (Equatorial).
        coord : str or None
            Target frame: ``"G"`` (Galactic), ``"E"`` (Ecliptic), or None
            (no rotation).

        Returns
        -------
        lon, lat : np.ndarray
            Rotated coordinates in radians, lon wrapped to [-pi, pi].
        """
        if coord is None or coord == "C":
            lon = ra_rad.copy()
            lon[lon > np.pi] -= 2 * np.pi
            return lon, dec_rad

        from healpy.rotator import Rotator

        rotator = Rotator(coord=["C", coord])
        theta = np.pi / 2 - dec_rad  # Dec -> colatitude
        theta_rot, phi_rot = rotator(theta, ra_rad)
        lat = np.pi / 2 - theta_rot  # colatitude -> latitude
        lon = phi_rot.copy()
        lon[lon > np.pi] -= 2 * np.pi
        return lon, lat

    def _healpy_view(
        self,
        map_data: np.ndarray,
        projection: str = "mollweide",
        coord: str | list[str] | None = None,
        rot: tuple[float, ...] | None = None,
        **kwargs,
    ) -> None:
        """Dispatch to the appropriate healpy view function.

        Parameters
        ----------
        map_data : np.ndarray
            HEALPix map to plot.
        projection : str
            ``"mollweide"``, ``"cartesian"``, ``"gnomonic"``, or
            ``"orthographic"``.
        coord : str, list, or None
            Coordinate frame or rotation (e.g. ``"G"``, ``["G","C"]``).
        rot : tuple or None
            ``(lon_deg, lat_deg)`` or ``(lon_deg, lat_deg, roll_deg)``.
        **kwargs
            Passed to the healpy view function (``fig``, ``sub``, ``title``,
            ``cmap``, ``min``, ``max``, ``unit``, ``notext``, etc.).
        """
        import healpy as hp

        func_name = _HEALPY_PROJECTIONS.get(projection)
        if func_name is None:
            raise ValueError(
                f"Unknown projection='{projection}'. "
                f"Choose from: {list(_HEALPY_PROJECTIONS.keys())}"
            )
        view_func = getattr(hp, func_name)
        view_func(map_data, coord=coord, rot=rot, **kwargs)

    def _get_stokes_map(
        self, stokes: str, frequency: float
    ) -> tuple[np.ndarray | None, str]:
        """Retrieve a single Stokes map and its label.

        Parameters
        ----------
        stokes : str
            ``"I"``, ``"Q"``, ``"U"``, or ``"V"``.
        frequency : float
            Frequency in Hz.

        Returns
        -------
        map_data : np.ndarray or None
        label : str
        """
        if stokes == "I":
            return self._sky.get_map_at_frequency(frequency), "Stokes I"

        maps_attr = {
            "Q": self._sky.healpix_q_maps,
            "U": self._sky.healpix_u_maps,
            "V": self._sky.healpix_v_maps,
        }
        if stokes not in maps_attr:
            raise ValueError(
                f"Unknown stokes='{stokes}'. Choose from 'I', 'Q', 'U', 'V'."
            )
        stokes_maps = maps_attr[stokes]
        if stokes_maps is None:
            return None, f"Stokes {stokes}"

        idx = self._sky.resolve_frequency_index(frequency)
        if idx < len(stokes_maps):
            return stokes_maps[idx], f"Stokes {stokes}"
        return None, f"Stokes {stokes}"

    def _get_stokes_cube(self, stokes: str) -> np.ndarray:
        """Return the full ``(n_freq, npix)`` cube for a Stokes parameter.

        Parameters
        ----------
        stokes : str
            ``"I"``, ``"Q"``, ``"U"``, or ``"V"``.

        Returns
        -------
        np.ndarray
            Multi-frequency cube, shape ``(n_freq, npix)``.

        Raises
        ------
        ValueError
            If the requested Stokes cube is not available.
        """
        if stokes == "I":
            cube = self._sky.healpix_maps
        else:
            maps_attr = {
                "Q": self._sky.healpix_q_maps,
                "U": self._sky.healpix_u_maps,
                "V": self._sky.healpix_v_maps,
            }
            if stokes not in maps_attr:
                raise ValueError(
                    f"Unknown stokes='{stokes}'. Choose from 'I', 'Q', 'U', 'V'."
                )
            cube = maps_attr[stokes]
        if cube is None:
            raise ValueError(
                f"Stokes {stokes} cube is not available for this SkyModel."
            )
        return np.asarray(cube)

    def _scatter_sky(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        values: np.ndarray,
        cbar_label: str,
        *,
        cmap: str = "inferno",
        marker_size: float = 0.1,
        alpha: float = 0.3,
        vmin: float | None = None,
        vmax: float | None = None,
        projection: str = "mollweide",
        coord: str | None = None,
        title: str = "",
        figsize: tuple[float, float] = (14, 7),
    ) -> Figure:
        """Shared scatter plot on a sky projection.

        Parameters
        ----------
        lon, lat : np.ndarray
            Source coordinates in radians (already rotated if needed).
        values : np.ndarray
            Values for color mapping.
        cbar_label : str
            Label for the colorbar.
        cmap : str
            Matplotlib colormap name.
        marker_size : float
            Scatter marker size.
        alpha : float
            Marker transparency.
        vmin, vmax : float or None
            Colorbar limits.
        projection : str
            Matplotlib projection: ``"mollweide"``, ``"aitoff"``,
            ``"hammer"``, or ``"cartesian"`` (rectilinear).
        coord : str or None
            Coordinate frame (used only for axis labels).
        title : str
            Plot title.
        figsize : tuple
            Figure size in inches.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        if projection not in _MPL_PROJECTIONS:
            raise ValueError(
                f"Unknown projection='{projection}'. "
                f"Choose from: {sorted(_MPL_PROJECTIONS)}"
            )

        if projection == "cartesian":
            fig, ax = plt.subplots(figsize=figsize)
            lon_deg = np.degrees(lon)
            lat_deg = np.degrees(lat)
            sc = ax.scatter(
                lon_deg,
                lat_deg,
                c=values,
                s=marker_size,
                alpha=alpha,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
            )
            ax.invert_xaxis()
            ax.set_aspect("equal")
        else:
            fig, ax = plt.subplots(
                figsize=figsize, subplot_kw={"projection": projection}
            )
            sc = ax.scatter(
                lon,
                lat,
                c=values,
                s=marker_size,
                alpha=alpha,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
            )
        fig.colorbar(sc, ax=ax, label=cbar_label, shrink=0.6)
        ax.grid(True, alpha=0.3)
        xlabel, ylabel = _COORD_LABELS.get(coord, ("RA", "Dec"))
        ax.set_xlabel(xlabel + " [deg]" if projection == "cartesian" else xlabel)
        ax.set_ylabel(ylabel + " [deg]" if projection == "cartesian" else ylabel)
        ax.set_title(title)
        return fig

    # =====================================================================
    # Point-source plots
    # =====================================================================

    def source_positions(
        self,
        color_by: str = "flux",
        cmap: str = "inferno",
        log_color: bool = True,
        marker_size: float = 0.1,
        alpha: float = 0.3,
        projection: str = "mollweide",
        coord: str | None = None,
        title: str | None = None,
        figsize: tuple[float, float] = (14, 7),
    ) -> Figure:
        """Scatter plot of source positions on a sky projection.

        Parameters
        ----------
        color_by : str
            Quantity for color mapping: ``"flux"``, ``"spectral_index"``,
            ``"stokes_q"``, ``"stokes_u"``, or ``"stokes_v"``.
        cmap : str
            Matplotlib colormap name.
        log_color : bool
            Apply log10 to the color values (only for positive-definite
            quantities like flux).
        marker_size : float
            Scatter marker size.
        alpha : float
            Marker transparency.
        projection : str
            Matplotlib projection: ``"mollweide"``, ``"aitoff"``,
            ``"hammer"``, or ``"cartesian"`` (rectilinear RA/Dec in degrees).
        coord : str, optional
            Target coordinate frame: ``"G"`` (Galactic), ``"E"``
            (Ecliptic), or None (Equatorial, no rotation).
        title : str, optional
            Custom title.  Auto-generated if None.
        figsize : tuple
            Figure size in inches.

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._validate_plot_mode("point_sources")

        values, cbar_label = self._get_color_data(color_by)

        if log_color and color_by == "flux":
            values = np.log10(np.clip(values, 1e-6, None))
            cbar_label = _COLORBAR_LABELS["flux_log"]

        lon, lat = self._rotate_coords(self._sky.ra_rad, self._sky.dec_rad, coord)

        return self._scatter_sky(
            lon,
            lat,
            values,
            cbar_label,
            cmap=cmap,
            marker_size=marker_size,
            alpha=alpha,
            projection=projection,
            coord=coord,
            title=title if title is not None else self._auto_title("Source Positions"),
            figsize=figsize,
        )

    def flux_histogram(
        self,
        n_bins: int = 50,
        log_scale: bool = True,
        title: str | None = None,
        figsize: tuple[float, float] = (10, 6),
    ) -> Figure:
        """Histogram of flux densities.

        Parameters
        ----------
        n_bins : int
            Number of histogram bins.
        log_scale : bool
            Use log-spaced bins and log-log axes.
        title : str, optional
            Custom title.  Auto-generated if None.
        figsize : tuple
            Figure size in inches.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        self._validate_plot_mode("point_sources")
        flux = self._sky.flux

        fig, ax = plt.subplots(figsize=figsize)

        if log_scale:
            positive = flux[flux > 0]
            if len(positive) == 0:
                raise ValueError("No positive flux values to plot on log scale.")
            bins = np.logspace(
                np.log10(positive.min()), np.log10(positive.max()), n_bins + 1
            )
            ax.hist(positive, bins=bins, edgecolor="black", linewidth=0.3)
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:
            ax.hist(flux, bins=n_bins, edgecolor="black", linewidth=0.3)

        ax.set_xlabel("Flux density (Jy)")
        ax.set_ylabel("Number of sources")

        # Annotate count and median
        median_flux = np.median(flux)
        ax.axvline(
            median_flux,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Median: {median_flux:.3g} Jy",
        )
        ax.legend()
        n_total = len(flux)
        ax.text(
            0.97,
            0.95,
            f"N = {n_total:,}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )

        ax.set_title(
            title if title is not None else self._auto_title("Flux Distribution")
        )
        return fig

    def spectral_index(
        self,
        plot_type: str = "histogram",
        cmap: str = "RdBu_r",
        n_bins: int = 50,
        projection: str = "mollweide",
        coord: str | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """Spectral index histogram or sky map.

        Parameters
        ----------
        plot_type : str
            ``"histogram"`` or ``"sky_map"``.
        cmap : str
            Colormap for sky map mode.
        n_bins : int
            Number of bins for histogram mode.
        projection : str
            Matplotlib projection for sky_map mode: ``"mollweide"``,
            ``"aitoff"``, ``"hammer"``, or ``"cartesian"`` (rectilinear
            RA/Dec in degrees).  Ignored for histogram.
        coord : str, optional
            Target coordinate frame for sky_map mode: ``"G"``
            (Galactic), ``"E"`` (Ecliptic), or None (Equatorial).
            Ignored for histogram.
        title : str, optional
            Custom title.  Auto-generated if None.
        figsize : tuple, optional
            Figure size.  Defaults to (10, 6) for histogram, (14, 7) for
            sky map.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        self._validate_plot_mode("point_sources")
        alpha_arr = self._sky.spectral_index

        if plot_type == "histogram":
            figsize = figsize or (10, 6)
            fig, ax = plt.subplots(figsize=figsize)
            ax.hist(alpha_arr, bins=n_bins, edgecolor="black", linewidth=0.3)
            median_alpha = np.median(alpha_arr)
            ax.axvline(
                median_alpha,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"Median: {median_alpha:.2f}",
            )
            ax.set_xlabel("Spectral index")
            ax.set_ylabel("Number of sources")
            ax.legend()
            ax.set_title(
                title
                if title is not None
                else self._auto_title("Spectral Index Distribution")
            )
            return fig

        elif plot_type == "sky_map":
            figsize = figsize or (14, 7)
            lon, lat = self._rotate_coords(self._sky.ra_rad, self._sky.dec_rad, coord)

            # Symmetric colorbar centered on median
            median_alpha = np.median(alpha_arr)
            max_dev = np.max(np.abs(alpha_arr - median_alpha))

            return self._scatter_sky(
                lon,
                lat,
                alpha_arr,
                _COLORBAR_LABELS["spectral_index"],
                cmap=cmap,
                vmin=median_alpha - max_dev,
                vmax=median_alpha + max_dev,
                projection=projection,
                coord=coord,
                title=title
                if title is not None
                else self._auto_title("Spectral Index Sky Map"),
                figsize=figsize,
            )

        else:
            raise ValueError(
                f"Unknown plot_type='{plot_type}'. Choose 'histogram' or 'sky_map'."
            )

    # =====================================================================
    # HEALPix plots
    # =====================================================================

    def healpix_map(
        self,
        frequency: float | None = None,
        stokes: str = "I",
        log_scale: bool = True,
        cmap: str = "inferno",
        projection: str = "mollweide",
        coord: str | list[str] | None = None,
        rot: tuple[float, ...] | None = None,
        title: str | None = None,
        figsize: tuple[float, float] = (10, 7),
    ) -> Figure:
        """Single HEALPix sky projection.

        Parameters
        ----------
        frequency : float, optional
            Frequency in Hz.  Defaults to reference frequency or first
            available.
        stokes : str
            ``"I"``, ``"Q"``, ``"U"``, or ``"V"``.
        log_scale : bool
            Apply log10 to Stokes I.  Ignored for Q/U/V (which can be
            negative).
        cmap : str
            Colormap.  Overridden to ``"RdBu_r"`` for Q/U/V if left as
            default ``"inferno"``.
        projection : str
            ``"mollweide"``, ``"cartesian"``, ``"gnomonic"``, or
            ``"orthographic"``.
        coord : str, list, or None
            Coordinate frame: ``"C"``, ``"G"``, ``"E"``, or a two-element
            list like ``["G", "C"]`` for rotation.  None = no rotation.
        rot : tuple, optional
            ``(lon_deg, lat_deg)`` or ``(lon_deg, lat_deg, roll_deg)`` to
            center/rotate the view.
        title : str, optional
            Custom title.  Auto-generated if None.
        figsize : tuple
            Figure size in inches.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        self._validate_plot_mode("healpix")
        freq = self._resolve_plot_frequency(frequency)
        map_data, stokes_label = self._get_stokes_map(stokes, freq)

        if map_data is None:
            raise ValueError(f"No {stokes_label} data available for this model.")

        # Determine colormap and scale
        is_quv = stokes in ("Q", "U", "V")
        if is_quv:
            if cmap == "inferno":
                cmap = "RdBu_r"
            max_abs = np.nanmax(np.abs(map_data))
            plot_data = map_data
            cbar_label = f"{stokes_label} (K$_{{RJ}}$)"
            hp_kwargs = {"min": -max_abs, "max": max_abs}
        elif log_scale:
            plot_data = np.log10(np.clip(map_data, 1e-6, None))
            cbar_label = _COLORBAR_LABELS["tb_log"]
            hp_kwargs = {}
        else:
            plot_data = map_data
            cbar_label = _COLORBAR_LABELS["tb"]
            hp_kwargs = {}

        auto = self._auto_title(f"{stokes_label} — {_freq_label(freq)}", frequency=None)
        plot_title = title if title is not None else auto

        fig = plt.figure(figsize=figsize)
        self._healpy_view(
            plot_data,
            projection=projection,
            coord=coord,
            rot=rot,
            fig=fig.number,
            title=plot_title,
            cmap=cmap,
            unit=cbar_label,
            notext=True,
            **hp_kwargs,
        )
        plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05)
        return fig

    def multifreq_grid(
        self,
        frequencies: list[float] | np.ndarray | None = None,
        max_panels: int = 12,
        ncols: int = 3,
        stokes: str = "I",
        log_scale: bool = True,
        cmap: str = "inferno",
        projection: str = "mollweide",
        coord: str | list[str] | None = None,
        rot: tuple[float, ...] | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """Grid of HEALPix projections at multiple frequencies.

        Parameters
        ----------
        frequencies : array-like, optional
            Subset of frequencies to plot.  If None, selects up to
            ``max_panels`` evenly spaced from available frequencies.
        max_panels : int
            Maximum number of panels when auto-selecting frequencies.
        ncols : int
            Number of columns in the grid.
        stokes : str
            ``"I"``, ``"Q"``, ``"U"``, or ``"V"``.
        log_scale : bool
            Apply log10 to Stokes I.
        cmap : str
            Colormap.
        projection : str
            ``"mollweide"``, ``"cartesian"``, ``"gnomonic"``, or
            ``"orthographic"``.
        coord : str, list, or None
            Coordinate frame or rotation.
        rot : tuple, optional
            ``(lon_deg, lat_deg)`` or ``(lon_deg, lat_deg, roll_deg)``.
        title : str, optional
            Overall figure title (suptitle).
        figsize : tuple, optional
            Auto-computed if None.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        self._validate_plot_mode("healpix")

        # Select frequencies
        if frequencies is not None:
            freqs = np.asarray(frequencies, dtype=float)
        else:
            all_freqs = self._sky.observation_frequencies
            if all_freqs is None or len(all_freqs) == 0:
                raise ValueError("No observation frequencies available.")
            if len(all_freqs) <= max_panels:
                freqs = all_freqs
            else:
                indices = np.linspace(0, len(all_freqs) - 1, max_panels, dtype=int)
                freqs = all_freqs[indices]

        n_panels = len(freqs)
        nrows = int(np.ceil(n_panels / ncols))
        if figsize is None:
            figsize = (6 * ncols, 4 * nrows)

        is_quv = stokes in ("Q", "U", "V")
        if is_quv and cmap == "inferno":
            cmap = "RdBu_r"

        # Collect all maps for consistent min/max
        maps = []
        for freq in freqs:
            map_data, _ = self._get_stokes_map(stokes, freq)
            if map_data is None:
                maps.append(None)
            else:
                maps.append(map_data)

        # Compute global range
        valid_maps = [m for m in maps if m is not None]
        if not valid_maps:
            raise ValueError(f"No Stokes {stokes} data available at any frequency.")

        if is_quv:
            global_max = max(np.nanmax(np.abs(m)) for m in valid_maps)
            global_min, global_max = -global_max, global_max
            cbar_label = f"Stokes {stokes} (K$_{{RJ}}$)"
        elif log_scale:
            all_log = [np.log10(np.clip(m, 1e-6, None)) for m in valid_maps]
            global_min = min(np.nanmin(m) for m in all_log)
            global_max = max(np.nanmax(m) for m in all_log)
            cbar_label = _COLORBAR_LABELS["tb_log"]
        else:
            global_min = min(np.nanmin(m) for m in valid_maps)
            global_max = max(np.nanmax(m) for m in valid_maps)
            cbar_label = _COLORBAR_LABELS["tb"]

        fig = plt.figure(figsize=figsize)

        for i, (freq, map_data) in enumerate(zip(freqs, maps, strict=False)):
            if map_data is None:
                continue
            if is_quv:
                plot_data = map_data
            elif log_scale:
                plot_data = np.log10(np.clip(map_data, 1e-6, None))
            else:
                plot_data = map_data

            self._healpy_view(
                plot_data,
                projection=projection,
                coord=coord,
                rot=rot,
                fig=fig.number,
                sub=(nrows, ncols, i + 1),
                title=_freq_label(freq),
                cmap=cmap,
                min=global_min,
                max=global_max,
                unit=cbar_label,
                notext=True,
            )

        if title is not None:
            fig.suptitle(title, fontsize=14, y=0.98)
        else:
            suptitle = self._auto_title(f"Stokes {stokes}")
            fig.suptitle(suptitle, fontsize=14, y=0.98)

        plt.subplots_adjust(hspace=0.3, wspace=0.1)
        return fig

    def stokes(
        self,
        frequency: float | None = None,
        log_scale_I: bool = True,
        cmap_I: str = "inferno",
        cmap_QUV: str = "RdBu_r",
        projection: str = "mollweide",
        coord: str | list[str] | None = None,
        rot: tuple[float, ...] | None = None,
        title: str | None = None,
        figsize: tuple[float, float] = (14, 10),
    ) -> Figure:
        """2x2 grid of Stokes I, Q, U, V at a single frequency.

        Parameters
        ----------
        frequency : float, optional
            Frequency in Hz.
        log_scale_I : bool
            Apply log10 to Stokes I.
        cmap_I : str
            Colormap for Stokes I.
        cmap_QUV : str
            Colormap for Stokes Q, U, V.
        projection : str
            ``"mollweide"``, ``"cartesian"``, ``"gnomonic"``, or
            ``"orthographic"``.
        coord : str, list, or None
            Coordinate frame or rotation.
        rot : tuple, optional
            ``(lon_deg, lat_deg)`` or ``(lon_deg, lat_deg, roll_deg)``.
        title : str, optional
            Overall figure title (suptitle).
        figsize : tuple
            Figure size in inches.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        self._validate_plot_mode("healpix")
        freq = self._resolve_plot_frequency(frequency)

        stokes_params = ["I", "Q", "U", "V"]
        fig = plt.figure(figsize=figsize)

        for i, stokes_name in enumerate(stokes_params):
            map_data, stokes_label = self._get_stokes_map(stokes_name, freq)

            if map_data is None:
                # Gray panel with "No data" text
                ax = fig.add_subplot(2, 2, i + 1)
                ax.set_facecolor("#e0e0e0")
                ax.text(
                    0.5,
                    0.5,
                    f"{stokes_label}\nNo data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="#666666",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            is_quv = stokes_name in ("Q", "U", "V")
            if is_quv:
                max_abs = np.nanmax(np.abs(map_data))
                plot_data = map_data
                cbar_label = f"{stokes_label} (K$_{{RJ}}$)"
                self._healpy_view(
                    plot_data,
                    projection=projection,
                    coord=coord,
                    rot=rot,
                    fig=fig.number,
                    sub=(2, 2, i + 1),
                    title=stokes_label,
                    cmap=cmap_QUV,
                    min=-max_abs,
                    max=max_abs,
                    unit=cbar_label,
                    notext=True,
                )
            else:
                if log_scale_I:
                    plot_data = np.log10(np.clip(map_data, 1e-6, None))
                    cbar_label = _COLORBAR_LABELS["tb_log"]
                else:
                    plot_data = map_data
                    cbar_label = _COLORBAR_LABELS["tb"]
                self._healpy_view(
                    plot_data,
                    projection=projection,
                    coord=coord,
                    rot=rot,
                    fig=fig.number,
                    sub=(2, 2, i + 1),
                    title=stokes_label,
                    cmap=cmap_I,
                    unit=cbar_label,
                    notext=True,
                )

        suptitle = (
            title
            if title is not None
            else self._auto_title(f"Stokes IQUV — {_freq_label(freq)}")
        )
        fig.suptitle(suptitle, fontsize=14, y=0.98)
        plt.subplots_adjust(hspace=0.3, wspace=0.1)
        return fig

    # =====================================================================
    # Sky-cube analysis methods  (HEALPix only)
    # =====================================================================

    def pixel_histogram(
        self,
        frequency: float | None = None,
        stokes: str = "I",
        bins: int = 200,
        show_gaussian_fit: bool = True,
        annotate_stats: bool = True,
        figsize: tuple[float, float] = (10, 6),
        title: str | None = None,
    ) -> Figure:
        """Histogram of HEALPix pixel values at a single frequency.

        Overlays a Gaussian fit and annotates skewness, excess kurtosis and
        the D'Agostino K² normality-test p-value — useful for validating
        Gaussian Random Field (GRF) realisations.

        Parameters
        ----------
        frequency : float, optional
            Frequency in Hz.  Defaults to the reference or first available.
        stokes : str
            Stokes parameter: ``"I"``, ``"Q"``, ``"U"``, ``"V"``.
        bins : int
            Number of histogram bins.
        show_gaussian_fit : bool
            Overlay a Gaussian with the sample mean / std.
        annotate_stats : bool
            Annotate skewness, kurtosis, and normaltest p-value.
        figsize : tuple
            Figure size in inches.
        title : str, optional
            Custom title.  Auto-generated if ``None``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from scipy import stats as _stats

        from ._analysis import gaussianity_stats

        self._validate_plot_mode("healpix")
        freq = self._resolve_plot_frequency(frequency)
        sky_map, _ = self._get_stokes_map(stokes, freq)
        if sky_map is None:
            raise ValueError(f"Stokes {stokes} data not available for this SkyModel.")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        values = np.asarray(sky_map).ravel()
        values = values[np.isfinite(values)]

        ax.hist(
            values,
            bins=bins,
            density=True,
            alpha=0.7,
            color="steelblue",
            label="Data",
        )

        if show_gaussian_fit and values.size > 1:
            mu, sigma = float(np.mean(values)), float(np.std(values))
            if sigma > 0:
                x = np.linspace(values.min(), values.max(), 500)
                ax.plot(
                    x,
                    _stats.norm.pdf(x, mu, sigma),
                    "r-",
                    lw=2,
                    label=f"Gaussian\n$\\mu$={mu:.3g}\n$\\sigma$={sigma:.3g}",
                )

        if annotate_stats and values.size >= 20:
            stats = gaussianity_stats(values)
            note = (
                f"skew = {stats['skewness']:.3f}\n"
                f"excess kurt = {stats['excess_kurtosis']:.3f}\n"
                f"D'Agostino p = {stats['normaltest_p']:.2e}"
            )
            ax.text(
                0.02,
                0.98,
                note,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )

        ax.set_xlabel(f"Stokes {stokes}")
        ax.set_ylabel("Probability density")
        ax.set_title(
            title
            if title is not None
            else self._auto_title(f"Pixel PDF (Stokes {stokes})", freq)
        )
        ax.legend(loc="upper right", fontsize=9)
        plt.subplots_adjust(left=0.1, right=0.97, top=0.90, bottom=0.12)
        return fig

    def variance_spectrum(
        self,
        stokes: str = "I",
        mad_threshold: float = 5.0,
        figsize: tuple[float, float] = (18, 4.5),
        title: str | None = None,
    ) -> Figure:
        """Mean, variance, and coefficient-of-variation vs frequency.

        Three-panel sanity plot highlighting outlier channels (where the
        CV σ/μ is more than ``mad_threshold`` × MAD from the median).

        Parameters
        ----------
        stokes : str
            Stokes parameter (``"I"``, ``"Q"``, ``"U"``, ``"V"``).
        mad_threshold : float
            Outlier threshold in units of MAD of the CV.
        figsize : tuple
            Figure size in inches.
        title : str, optional
            Suptitle.

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._validate_plot_mode("healpix")
        if (
            self._sky.observation_frequencies is None
            or len(self._sky.observation_frequencies) < 2
        ):
            raise ValueError(
                "variance_spectrum requires at least 2 observation frequencies."
            )

        import matplotlib.pyplot as plt

        freqs_mhz = np.asarray(self._sky.observation_frequencies) / 1e6

        # Build (n_freq, npix) cube for the requested Stokes parameter
        maps = self._get_stokes_cube(stokes)

        means = maps.mean(axis=1)
        stds = maps.std(axis=1)
        variance = stds**2
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.where(np.abs(means) > 0, stds / np.abs(means), np.nan)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        axes[0].plot(freqs_mhz, means, "o-", markersize=3, color="navy")
        axes[0].set_xlabel("Frequency [MHz]")
        axes[0].set_ylabel(f"Mean Stokes {stokes}")
        axes[0].set_title("Mean vs Frequency")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(freqs_mhz, variance, "o-", markersize=3, color="darkred")
        axes[1].set_xlabel("Frequency [MHz]")
        axes[1].set_ylabel("Variance")
        axes[1].set_title("Variance vs Frequency")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(freqs_mhz, cv, "o-", markersize=3, color="darkgreen")
        axes[2].set_xlabel("Frequency [MHz]")
        axes[2].set_ylabel(r"$\sigma / |\mu|$ (CV)")
        axes[2].set_title("Coefficient of Variation")
        axes[2].grid(True, alpha=0.3)

        # Flag outlier channels
        finite = np.isfinite(cv)
        if finite.any():
            cv_med = float(np.median(cv[finite]))
            cv_mad = float(np.median(np.abs(cv[finite] - cv_med)))
            if cv_mad > 0:
                outlier = np.abs(cv - cv_med) > mad_threshold * cv_mad
                if outlier.any():
                    axes[2].scatter(
                        freqs_mhz[outlier],
                        cv[outlier],
                        s=60,
                        edgecolor="red",
                        facecolor="none",
                        linewidth=1.5,
                        label="outlier",
                    )
                    axes[2].legend(fontsize=9)

        fig.suptitle(
            title
            if title is not None
            else self._auto_title(f"Variance / Mean / CV — Stokes {stokes}"),
            fontsize=14,
            y=1.02,
        )
        plt.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.14, wspace=0.3)
        return fig

    def frequency_spectra(
        self,
        stokes: str = "I",
        n_pixels: int = 8,
        pixel_indices: list[int] | np.ndarray | None = None,
        show_mean_spectrum: bool = True,
        seed: int = 42,
        figsize: tuple[float, float] = (16, 5),
        title: str | None = None,
    ) -> Figure:
        """Per-pixel frequency spectra, plus mean ± 1σ envelope.

        Parameters
        ----------
        stokes : str
            Stokes parameter.
        n_pixels : int
            Number of individual pixel spectra to draw (ignored if
            ``pixel_indices`` is given).
        pixel_indices : array-like, optional
            Specific pixel indices to plot.
        show_mean_spectrum : bool
            Show the population mean ± 1σ in a second panel.
        seed : int
            RNG seed for random pixel selection.
        figsize : tuple
            Figure size in inches.
        title : str, optional
            Suptitle.

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._validate_plot_mode("healpix")
        if (
            self._sky.observation_frequencies is None
            or len(self._sky.observation_frequencies) < 2
        ):
            raise ValueError(
                "frequency_spectra requires at least 2 observation frequencies."
            )

        import matplotlib.pyplot as plt

        freqs_mhz = np.asarray(self._sky.observation_frequencies) / 1e6
        maps = self._get_stokes_cube(stokes)

        if pixel_indices is None:
            rng = np.random.default_rng(seed)
            pixel_indices = rng.choice(maps.shape[1], n_pixels, replace=False)
        else:
            pixel_indices = np.asarray(pixel_indices)

        if show_mean_spectrum:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0] / 2, figsize[1]))
            ax2 = None

        for p in pixel_indices:
            ax1.plot(freqs_mhz, maps[:, p], alpha=0.6, label=f"pix {int(p)}")

        ax1.set_xlabel("Frequency [MHz]")
        ax1.set_ylabel(f"Stokes {stokes}")
        ax1.set_title("Individual pixel spectra")
        ax1.legend(fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.3)

        if ax2 is not None:
            mean_spec = maps.mean(axis=1)
            std_spec = maps.std(axis=1)
            ax2.plot(freqs_mhz, mean_spec, "k-", lw=2, label="Mean")
            ax2.fill_between(
                freqs_mhz,
                mean_spec - std_spec,
                mean_spec + std_spec,
                alpha=0.3,
                color="steelblue",
                label=r"$\pm 1\sigma$",
            )
            ax2.set_xlabel("Frequency [MHz]")
            ax2.set_ylabel(f"Stokes {stokes}")
            ax2.set_title("Mean spectrum (all pixels)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        fig.suptitle(
            title
            if title is not None
            else self._auto_title(f"Frequency spectra — Stokes {stokes}"),
            fontsize=14,
            y=1.02,
        )
        plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.14, wspace=0.25)
        return fig

    def frequency_waterfall(
        self,
        stokes: str = "I",
        pixel_stride: int = 1,
        detrend: bool = True,
        figsize: tuple[float, float] = (16, 5),
        title: str | None = None,
    ) -> Figure:
        """2-D (frequency × pixel) waterfall image.

        Two panels: raw brightness and per-channel mean-subtracted.
        Useful for spotting banding, channel-level discontinuities, or
        corrupted pixels.

        Parameters
        ----------
        stokes : str
            Stokes parameter.
        pixel_stride : int
            Subsample every Nth pixel to keep the figure manageable.
        detrend : bool
            If ``True``, include the mean-subtracted panel.
        figsize : tuple
            Figure size.
        title : str, optional
            Suptitle.

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._validate_plot_mode("healpix")
        if (
            self._sky.observation_frequencies is None
            or len(self._sky.observation_frequencies) < 2
        ):
            raise ValueError(
                "frequency_waterfall requires at least 2 observation frequencies."
            )

        import matplotlib.pyplot as plt

        freqs_mhz = np.asarray(self._sky.observation_frequencies) / 1e6
        maps = self._get_stokes_cube(stokes)
        if pixel_stride < 1:
            raise ValueError(f"pixel_stride must be >= 1, got {pixel_stride}")
        cube = maps[:, ::pixel_stride]
        n_sampled = cube.shape[1]

        if detrend:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0] / 2, figsize[1]))
            ax2 = None

        extent = [0, n_sampled, freqs_mhz[0], freqs_mhz[-1]]
        im1 = ax1.imshow(
            cube,
            aspect="auto",
            origin="lower",
            cmap="inferno",
            extent=extent,
        )
        fig.colorbar(im1, ax=ax1, label=f"Stokes {stokes}")
        ax1.set_xlabel(f"Pixel index (every {pixel_stride}th)")
        ax1.set_ylabel("Frequency [MHz]")
        ax1.set_title(f"Stokes {stokes} waterfall")

        if ax2 is not None:
            detrended = cube - cube.mean(axis=1, keepdims=True)
            vmax = float(np.percentile(np.abs(detrended), 99))
            im2 = ax2.imshow(
                detrended,
                aspect="auto",
                origin="lower",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                extent=extent,
            )
            fig.colorbar(im2, ax=ax2, label=f"Stokes {stokes} − mean")
            ax2.set_xlabel(f"Pixel index (every {pixel_stride}th)")
            ax2.set_ylabel("Frequency [MHz]")
            ax2.set_title("Mean-subtracted waterfall")

        fig.suptitle(
            title
            if title is not None
            else self._auto_title(f"Frequency waterfall — Stokes {stokes}"),
            fontsize=14,
            y=1.02,
        )
        plt.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.12, wspace=0.20)
        return fig

    def angular_power_spectrum(
        self,
        frequencies: list[float] | np.ndarray | None = None,
        lmax: int | None = None,
        representation: str = "both",
        remove_monopole: bool = True,
        ell_max_input: int | None = None,
        stokes: str = "I",
        max_panels: int = 6,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
    ) -> Figure:
        """Angular power spectrum ``C_ell`` (and/or ``D_ell``) at one or more frequencies.

        Parameters
        ----------
        frequencies : array-like, optional
            Frequencies (Hz) to plot.  Auto-selects up to ``max_panels``
            evenly-spaced channels if ``None``.
        lmax : int, optional
            Maximum multipole.  Defaults to ``3 * nside - 1``.
        representation : str
            ``"c_ell"`` (raw), ``"d_ell"`` (``ell(ell+1)C_ell/2π``, CMB/CAMB
            convention), or ``"both"`` (two-panel plot).
        remove_monopole : bool
            Remove the sky monopole before the transform.
        ell_max_input : int, optional
            If given, draw a vertical reference line (e.g. the GRF
            generation bandlimit).
        stokes : str
            Stokes parameter.
        max_panels : int
            Maximum number of frequency channels to auto-select.
        figsize : tuple, optional
            Figure size.  Defaults to ``(16, 5)`` for ``"both"`` else ``(9, 5)``.
        title : str, optional
            Suptitle.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from ._analysis import compute_angular_power_spectrum

        self._validate_plot_mode("healpix")

        if representation not in ("c_ell", "d_ell", "both"):
            raise ValueError(
                f"representation must be 'c_ell', 'd_ell' or 'both', "
                f"got {representation!r}"
            )

        import matplotlib.pyplot as plt

        freq_arr = self._sky.observation_frequencies
        if frequencies is None:
            if freq_arr is None or len(freq_arr) == 0:
                raise ValueError("SkyModel has no observation frequencies.")
            n = min(max_panels, len(freq_arr))
            idx = np.linspace(0, len(freq_arr) - 1, n, dtype=int)
            freqs_list = freq_arr[idx]
        else:
            freqs_list = np.asarray(frequencies)

        if figsize is None:
            figsize = (16, 5) if representation == "both" else (9, 5)

        if representation == "both":
            fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize)
            axes = (axL, axR)
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            axes = (ax,)

        for freq in freqs_list:
            sky_map, _ = self._get_stokes_map(stokes, float(freq))
            if sky_map is None:
                continue
            cl = compute_angular_power_spectrum(
                sky_map,
                lmax=lmax,
                remove_monopole=remove_monopole,
            )
            ell = np.arange(len(cl))
            label = _freq_label(float(freq))

            if representation in ("c_ell", "both"):
                axes[0].loglog(ell[1:], cl[1:], alpha=0.75, label=label)
            if representation in ("d_ell", "both"):
                ax_dl = axes[1] if representation == "both" else axes[0]
                dl = ell * (ell + 1) * cl / (2 * np.pi)
                ax_dl.loglog(ell[1:], dl[1:], alpha=0.75, label=label)

        # Axis labels + optional ell_max_input reference line
        for a in axes:
            a.set_xlabel(r"Multipole $\ell$")
            a.grid(True, alpha=0.3)
            a.legend(fontsize=8, ncol=2)
            if ell_max_input is not None:
                a.axvline(
                    ell_max_input,
                    color="k",
                    ls="--",
                    alpha=0.5,
                    label=f"input $\\ell_{{max}}$={ell_max_input}",
                )

        if representation == "both":
            axes[0].set_ylabel(r"$C_\ell$")
            axes[0].set_title(r"Angular Power Spectrum $C_\ell$")
            axes[1].set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$")
            axes[1].set_title(r"$D_\ell$")
        elif representation == "c_ell":
            axes[0].set_ylabel(r"$C_\ell$")
            axes[0].set_title(r"Angular Power Spectrum $C_\ell$")
        else:
            axes[0].set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$")
            axes[0].set_title(r"$D_\ell$")

        fig.suptitle(
            title if title is not None else self._auto_title("Angular power spectrum"),
            fontsize=14,
            y=1.02,
        )
        plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.14, wspace=0.25)
        return fig

    def cross_frequency_cell(
        self,
        ref_frequency: float | None = None,
        target_frequencies: list[float] | np.ndarray | None = None,
        separations: list[int] | None = None,
        lmax: int | None = None,
        remove_monopole: bool = True,
        stokes: str = "I",
        figsize: tuple[float, float] = (10, 5),
        title: str | None = None,
    ) -> Figure:
        """Cross-power spectrum ``C_ell(nu_ref, nu_target)`` vs target frequency.

        Parameters
        ----------
        ref_frequency : float, optional
            Reference frequency in Hz.  Defaults to the first channel.
        target_frequencies : array-like, optional
            Explicit target frequencies.  If ``None``, uses channel indices
            ``separations`` relative to the reference.
        separations : list of int, optional
            Channel offsets from the reference (ignored if
            ``target_frequencies`` is given).  Default: ``[1, 5, 10, 20, 40]``.
        lmax : int, optional
            Max multipole.
        remove_monopole : bool
            Remove monopoles before anafast.
        stokes : str
            Stokes parameter.
        figsize : tuple
            Figure size.
        title : str, optional
            Suptitle.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from ._analysis import (
            compute_angular_power_spectrum,
            compute_cross_cell,
        )

        self._validate_plot_mode("healpix")
        freq_arr = self._sky.observation_frequencies
        if freq_arr is None or len(freq_arr) < 2:
            raise ValueError(
                "cross_frequency_cell requires at least 2 observation frequencies."
            )

        import matplotlib.pyplot as plt

        ref_freq = (
            self._resolve_plot_frequency(ref_frequency)
            if ref_frequency is not None
            else float(freq_arr[0])
        )
        ref_idx = self._sky.resolve_frequency_index(ref_freq)
        ref_map, _ = self._get_stokes_map(stokes, ref_freq)
        if ref_map is None:
            raise ValueError(f"Stokes {stokes} not available for reference frequency.")

        if target_frequencies is None:
            if separations is None:
                separations = [1, 5, 10, 20, 40]
            targets = []
            for sep in separations:
                j = ref_idx + sep
                if 0 <= j < len(freq_arr):
                    targets.append(float(freq_arr[j]))
        else:
            targets = [float(f) for f in target_frequencies]

        fig, ax = plt.subplots(figsize=figsize)

        cl_auto = compute_angular_power_spectrum(
            ref_map,
            lmax=lmax,
            remove_monopole=remove_monopole,
        )
        ell = np.arange(len(cl_auto))
        ax.loglog(
            ell[2:],
            cl_auto[2:],
            "k-",
            lw=2,
            label=f"Auto: {_freq_label(ref_freq)}",
        )

        for f_target in targets:
            m_target, _ = self._get_stokes_map(stokes, f_target)
            if m_target is None:
                continue
            cl_cross = compute_cross_cell(
                ref_map,
                m_target,
                lmax=lmax,
                remove_monopole=remove_monopole,
            )
            df_mhz = (f_target - ref_freq) / 1e6
            ax.loglog(
                ell[2:],
                np.abs(cl_cross[2:]),
                alpha=0.7,
                label=f"Cross: $\\Delta\\nu$={df_mhz:+.2f} MHz",
            )

        ax.set_xlabel(r"Multipole $\ell$")
        ax.set_ylabel(r"$|C_\ell|$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title(
            title
            if title is not None
            else self._auto_title("Cross-frequency angular power spectrum")
        )
        plt.subplots_adjust(left=0.1, right=0.97, top=0.92, bottom=0.12)
        return fig

    def multipole_bands(
        self,
        frequency: float | None = None,
        bands: list[tuple[int, int]] | None = None,
        lmax: int | None = None,
        ncols: int = 4,
        cmap: str = "RdBu_r",
        stokes: str = "I",
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
    ) -> Figure:
        """Decompose a HEALPix map into ell bands and show each as a panel.

        Each panel is a Mollweide view of the map with spherical-harmonic
        coefficients outside the band set to zero.  Useful for visualising
        which angular scales carry the map's power.

        Parameters
        ----------
        frequency : float, optional
            Frequency in Hz.
        bands : list of (int, int), optional
            ``[(ell_lo, ell_hi), ...]`` pairs (inclusive).  Defaults to a
            granular set covering 2–500.
        lmax : int, optional
            Max multipole for the transform.  Defaults to ``3 * nside - 1``.
        ncols : int
            Number of panel columns.
        cmap : str
            Matplotlib colormap (symmetric around 0 recommended).
        stokes : str
            Stokes parameter.
        figsize : tuple, optional
            Figure size.
        title : str, optional
            Suptitle.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from ._analysis import filter_ell_band

        self._validate_plot_mode("healpix")
        freq = self._resolve_plot_frequency(frequency)
        sky_map, _ = self._get_stokes_map(stokes, freq)
        if sky_map is None:
            raise ValueError(f"Stokes {stokes} not available.")

        import healpy as hp
        import matplotlib.pyplot as plt

        nside = int(hp.npix2nside(len(sky_map)))
        if lmax is None:
            lmax = 3 * nside - 1

        if bands is None:
            default_bands = [
                (2, 10),
                (10, 30),
                (30, 60),
                (60, 120),
                (120, 200),
                (200, 350),
                (350, 500),
            ]
            if lmax > 500:
                default_bands.append((500, min(lmax, 767)))
            bands = default_bands

        n_panels = len(bands)
        nrows = int(np.ceil(n_panels / ncols))
        if figsize is None:
            figsize = (5.0 * ncols, 3.0 * nrows)

        fig = plt.figure(figsize=figsize)

        for i, (ell_lo, ell_hi) in enumerate(bands):
            filtered = filter_ell_band(
                sky_map,
                int(ell_lo),
                int(ell_hi),
                lmax=lmax,
            )
            ell_center = 0.5 * (ell_lo + ell_hi)
            theta_deg = 180.0 / max(ell_center, 1.0)
            vmax = float(np.percentile(np.abs(filtered), 99)) or 1.0
            self._healpy_view(
                filtered,
                projection="mollweide",
                fig=fig.number,
                sub=(nrows, ncols, i + 1),
                title=(
                    rf"$\ell = {ell_lo}$–${ell_hi}$   "
                    rf"($\theta \sim$ {theta_deg:.1f}°)"
                ),
                cmap=cmap,
                min=-vmax,
                max=vmax,
                notext=True,
            )

        fig.suptitle(
            title
            if title is not None
            else self._auto_title(
                f"Multipole-band decomposition — Stokes {stokes}", freq
            ),
            fontsize=14,
            y=1.00,
        )
        plt.subplots_adjust(hspace=0.3, wspace=0.1)
        return fig

    def frequency_correlation(
        self,
        pixel_stride: int = 1,
        center_maps: bool = True,
        show_decorrelation: bool = True,
        stokes: str = "I",
        figsize: tuple[float, float] = (14, 5),
        title: str | None = None,
    ) -> Figure:
        """Cross-frequency Pearson correlation matrix + decorrelation curve.

        Parameters
        ----------
        pixel_stride : int
            Subsample every Nth pixel to bound memory (use ``1`` for full).
        center_maps : bool
            Mean-subtract each channel before computing correlation so the
            monopole does not dominate.
        show_decorrelation : bool
            Include a second panel showing mean correlation vs frequency lag.
        stokes : str
            Stokes parameter.
        figsize : tuple
            Figure size.
        title : str, optional
            Suptitle.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from ._analysis import compute_frequency_correlation

        self._validate_plot_mode("healpix")
        if (
            self._sky.observation_frequencies is None
            or len(self._sky.observation_frequencies) < 2
        ):
            raise ValueError(
                "frequency_correlation requires at least 2 observation frequencies."
            )
        if pixel_stride < 1:
            raise ValueError(f"pixel_stride must be >= 1, got {pixel_stride}")

        import matplotlib.pyplot as plt

        freqs_mhz = np.asarray(self._sky.observation_frequencies) / 1e6
        maps = self._get_stokes_cube(stokes)
        cube = maps[:, ::pixel_stride]
        corr = compute_frequency_correlation(cube, center=center_maps)

        if show_decorrelation:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0] / 2, figsize[1]))
            ax2 = None

        im = ax1.imshow(
            corr,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            extent=[freqs_mhz[0], freqs_mhz[-1], freqs_mhz[-1], freqs_mhz[0]],
        )
        fig.colorbar(im, ax=ax1, label="Pearson r")
        ax1.set_xlabel("Frequency [MHz]")
        ax1.set_ylabel("Frequency [MHz]")
        ax1.set_title(f"Cross-frequency correlation (Stokes {stokes})")

        if ax2 is not None:
            n = len(freqs_mhz)
            dfreq = float(np.diff(freqs_mhz).mean())
            max_lag = n // 2
            lag_mhz = np.arange(max_lag) * dfreq
            corr_vs_lag = np.array(
                [float(np.nanmean(np.diag(corr, k=lag))) for lag in range(max_lag)]
            )
            ax2.plot(lag_mhz, corr_vs_lag, "o-", markersize=3)
            ax2.axhline(0, color="k", ls="--", alpha=0.3)
            ax2.axhline(
                1.0 / np.e,
                color="r",
                ls=":",
                alpha=0.6,
                label="$1/e$",
            )
            ax2.set_xlabel("Frequency separation [MHz]")
            ax2.set_ylabel("Mean correlation")
            ax2.set_title("Decorrelation with frequency lag")
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=9)

            # Annotate 1/e scale if crossed
            below = corr_vs_lag < (1.0 / np.e)
            if below.any():
                idx = int(np.argmax(below))
                if idx > 0:
                    ax2.axvline(
                        lag_mhz[idx],
                        color="r",
                        ls=":",
                        alpha=0.6,
                    )

        fig.suptitle(
            title
            if title is not None
            else self._auto_title(f"Frequency correlation — Stokes {stokes}"),
            fontsize=14,
            y=1.02,
        )
        plt.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.14, wspace=0.30)
        return fig

    def delay_spectrum(
        self,
        window: str = "blackmanharris",
        show_kparallel: bool = True,
        cosmology: Any | None = None,
        remove_mean: bool = True,
        pixel_stride: int = 1,
        stokes: str = "I",
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
    ) -> Figure:
        """Line-of-sight delay spectrum and optional ``P(k_parallel)``.

        FFTs along the frequency axis per pixel using a configurable window
        (Blackman-Harris by default, Parsons 2012 convention), averages the
        power over pixels, and converts delay to ``k_parallel`` using the
        Parsons 2012 formula.

        Parameters
        ----------
        window : str
            Window function: ``"blackmanharris"``, ``"blackman"``, ``"hann"``,
            ``"hamming"``, or ``"none"``.
        show_kparallel : bool
            Add a second panel showing ``P(k_parallel)``.
        cosmology : astropy.cosmology.FLRW, optional
            Cosmology for ``H(z)``.  Defaults to ``Planck15``.
        remove_mean : bool
            Subtract per-pixel mean before FFT (suppresses zero-delay monopole).
        pixel_stride : int
            Subsample pixels for performance.
        stokes : str
            Stokes parameter.
        figsize : tuple, optional
            Figure size.
        title : str, optional
            Suptitle.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from ._analysis import (
            F_21_HZ,
            compute_delay_spectrum,
            compute_kparallel,
        )

        self._validate_plot_mode("healpix")
        if (
            self._sky.observation_frequencies is None
            or len(self._sky.observation_frequencies) < 4
        ):
            raise ValueError(
                "delay_spectrum requires at least 4 observation frequencies."
            )

        import matplotlib.pyplot as plt

        freqs_hz = np.asarray(self._sky.observation_frequencies)
        maps = self._get_stokes_cube(stokes)
        cube = maps[:, ::pixel_stride]

        tau_s, delay_ps = compute_delay_spectrum(
            cube,
            freqs_hz,
            window=window,
            remove_mean=remove_mean,
        )
        tau_ns = tau_s * 1e9

        if figsize is None:
            figsize = (16, 5) if show_kparallel else (9, 5)

        if show_kparallel:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)
            ax2 = None

        ax1.semilogy(tau_ns, delay_ps, "b-", alpha=0.85)
        ax1.set_xlabel(r"Delay $\tau$ [ns]")
        ax1.set_ylabel("Delay power")
        ax1.set_title(f"Delay spectrum — window: {window}")
        ax1.grid(True, alpha=0.3)

        if ax2 is not None:
            z_center = F_21_HZ / freqs_hz.mean() - 1.0
            k_par = compute_kparallel(np.abs(tau_s), z_center, cosmology=cosmology)
            pos = tau_s > 0
            ax2.loglog(k_par[pos], delay_ps[pos], "b-", alpha=0.85)
            ax2.set_xlabel(r"$k_\parallel$ [Mpc$^{-1}$]")
            ax2.set_ylabel(r"$P(k_\parallel)$")
            ax2.set_title(f"Line-of-sight power spectrum (z={z_center:.2f})")
            ax2.grid(True, alpha=0.3)

        fig.suptitle(
            title
            if title is not None
            else self._auto_title(f"Delay spectrum — Stokes {stokes}"),
            fontsize=14,
            y=1.02,
        )
        plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.14, wspace=0.25)
        return fig

    # =====================================================================
    # Dispatcher
    # =====================================================================

    def __call__(
        self,
        plot_type: str = "auto",
        **kwargs,
    ) -> Figure:
        """Convenience dispatcher for common plot types.

        Parameters
        ----------
        plot_type : str
            Plot type.  ``"auto"`` dispatches based on the model's current
            mode.  Other options:

            - ``"sources"`` / ``"source_positions"`` -- Mollweide scatter
            - ``"flux"`` / ``"flux_histogram"`` -- flux histogram
            - ``"alpha"`` / ``"spectral_index"`` -- spectral index plot
            - ``"map"`` / ``"healpix"`` -- single HEALPix Mollweide
            - ``"grid"`` / ``"multifreq"`` -- multi-frequency grid
            - ``"stokes"`` -- Stokes IQUV 2x2 grid
            - ``"pdf"`` / ``"histogram"`` -- pixel PDF + Gaussianity
            - ``"cell"`` / ``"power_spectrum"`` -- angular power spectrum
            - ``"cross_cell"`` -- cross-frequency C_ell
            - ``"bands"`` / ``"multipole_bands"`` -- ell-band decomposition
            - ``"corr"`` / ``"correlation"`` -- frequency correlation matrix
            - ``"spectra"`` / ``"frequency_spectra"`` -- per-pixel spectra
            - ``"delay"`` / ``"delay_spectrum"`` -- delay / P(k||)
            - ``"variance"`` -- variance vs frequency
            - ``"waterfall"`` -- frequency × pixel waterfall

        **kwargs
            Passed through to the underlying plot method.

        Returns
        -------
        matplotlib.figure.Figure
        """
        _dispatch = {
            "sources": self.source_positions,
            "source_positions": self.source_positions,
            "flux": self.flux_histogram,
            "flux_histogram": self.flux_histogram,
            "alpha": self.spectral_index,
            "spectral_index": self.spectral_index,
            "map": self.healpix_map,
            "healpix": self.healpix_map,
            "grid": self.multifreq_grid,
            "multifreq": self.multifreq_grid,
            "stokes": self.stokes,
            # New cube-analysis methods
            "pdf": self.pixel_histogram,
            "histogram": self.pixel_histogram,
            "pixel_histogram": self.pixel_histogram,
            "cell": self.angular_power_spectrum,
            "power_spectrum": self.angular_power_spectrum,
            "angular_power_spectrum": self.angular_power_spectrum,
            "cross_cell": self.cross_frequency_cell,
            "cross_frequency_cell": self.cross_frequency_cell,
            "bands": self.multipole_bands,
            "multipole_bands": self.multipole_bands,
            "corr": self.frequency_correlation,
            "correlation": self.frequency_correlation,
            "frequency_correlation": self.frequency_correlation,
            "spectra": self.frequency_spectra,
            "frequency_spectra": self.frequency_spectra,
            "delay": self.delay_spectrum,
            "delay_spectrum": self.delay_spectrum,
            "variance": self.variance_spectrum,
            "variance_spectrum": self.variance_spectrum,
            "waterfall": self.frequency_waterfall,
            "frequency_waterfall": self.frequency_waterfall,
        }

        if plot_type == "auto":
            if self._sky.mode == SkyFormat.HEALPIX:
                return self.healpix_map(**kwargs)
            return self.source_positions(**kwargs)

        if plot_type not in _dispatch:
            raise ValueError(
                f"Unknown plot_type='{plot_type}'. "
                f"Choose from: 'auto', {', '.join(repr(k) for k in _dispatch)}"
            )
        return _dispatch[plot_type](**kwargs)
