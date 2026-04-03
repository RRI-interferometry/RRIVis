# rrivis/core/sky/_plotting.py
"""Sky model plotting mixin for SkyModel.

Provides matplotlib-based visualization methods for point-source catalogs
and HEALPix diffuse maps. All public methods return a
:class:`matplotlib.figure.Figure` and never call ``plt.show()``, following
the convention established by the beam plotting module.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

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
_MPL_PROJECTIONS = {"mollweide", "aitoff", "hammer"}

# Axis labels per coordinate system
_COORD_LABELS = {
    None: ("RA", "Dec"),
    "C": ("RA", "Dec"),
    "G": ("Galactic $l$", "Galactic $b$"),
    "E": ("Ecliptic lon", "Ecliptic lat"),
}


class _PlottingMixin:
    """Mixin providing plotting methods for SkyModel."""

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
        if self.model_name:
            parts.append(self.model_name.upper().replace("_", " "))
        if frequency is not None:
            parts.append(_freq_label(frequency))
        if self._healpix_nside is not None and self._healpix_maps is not None:
            parts.append(f"nside={self._healpix_nside}")
        meta = " — ".join(parts) if parts else "Sky Model"
        return f"{meta} — {suffix}"

    def _validate_plot_mode(self, required: str) -> None:
        """Raise ValueError if the model lacks the required data."""
        if required == "point_sources":
            if not self._has_point_sources():
                raise ValueError(
                    f"Plot requires point-source data, but this SkyModel "
                    f"(model='{self.model_name}') has mode='{self.mode}'. "
                    f"Load a point-source catalog or call to_point_sources() first."
                )
        elif required == "healpix":
            if self._healpix_maps is None:
                raise ValueError(
                    f"Plot requires HEALPix maps, but this SkyModel "
                    f"(model='{self.model_name}') has mode='{self.mode}'. "
                    f"Load a diffuse model or call to_healpix_for_observation() first."
                )

    def _resolve_plot_frequency(self, frequency: float | None) -> float:
        """Pick the best frequency for plotting from what is available.

        Parameters
        ----------
        frequency : float or None
            Requested frequency in Hz.  If None, falls back to
            ``self.frequency`` then the first key of ``_healpix_maps``.

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
        if self.frequency is not None:
            return float(self.frequency)
        if self._healpix_maps:
            return float(next(iter(self._healpix_maps)))
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
            "flux": (self._flux_ref, _COLORBAR_LABELS["flux"]),
            "spectral_index": (self._alpha, _COLORBAR_LABELS["spectral_index"]),
            "stokes_q": (self._stokes_q, _COLORBAR_LABELS["stokes_q"]),
            "stokes_u": (self._stokes_u, _COLORBAR_LABELS["stokes_u"]),
            "stokes_v": (self._stokes_v, _COLORBAR_LABELS["stokes_v"]),
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
        theta = np.pi / 2 - dec_rad  # Dec → colatitude
        theta_rot, phi_rot = rotator(theta, ra_rad)
        lat = np.pi / 2 - theta_rot  # colatitude → latitude
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

    # =====================================================================
    # Point-source plots
    # =====================================================================

    def plot_source_positions(
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
            Matplotlib projection: ``"mollweide"``, ``"aitoff"``, or
            ``"hammer"``.
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
        import matplotlib.pyplot as plt

        if projection not in _MPL_PROJECTIONS:
            raise ValueError(
                f"Unknown projection='{projection}'. "
                f"Choose from: {sorted(_MPL_PROJECTIONS)}"
            )

        self._validate_plot_mode("point_sources")

        values, cbar_label = self._get_color_data(color_by)

        if log_color and color_by == "flux":
            values = np.log10(np.clip(values, 1e-6, None))
            cbar_label = _COLORBAR_LABELS["flux_log"]

        lon, lat = self._rotate_coords(self._ra_rad, self._dec_rad, coord)

        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": projection})
        sc = ax.scatter(
            lon,
            lat,
            c=values,
            s=marker_size,
            alpha=alpha,
            cmap=cmap,
            rasterized=True,
        )
        fig.colorbar(sc, ax=ax, label=cbar_label, shrink=0.6)
        ax.grid(True, alpha=0.3)
        xlabel, ylabel = _COORD_LABELS.get(coord, ("RA", "Dec"))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(
            title if title is not None else self._auto_title("Source Positions")
        )
        return fig

    def plot_flux_histogram(
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
        flux = self._flux_ref

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

    def plot_spectral_index(
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
            ``"aitoff"``, or ``"hammer"``.  Ignored for histogram.
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
        alpha_arr = self._alpha

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
            if projection not in _MPL_PROJECTIONS:
                raise ValueError(
                    f"Unknown projection='{projection}'. "
                    f"Choose from: {sorted(_MPL_PROJECTIONS)}"
                )
            figsize = figsize or (14, 7)
            fig, ax = plt.subplots(
                figsize=figsize, subplot_kw={"projection": projection}
            )
            lon, lat = self._rotate_coords(self._ra_rad, self._dec_rad, coord)

            # Symmetric colorbar centered on median
            median_alpha = np.median(alpha_arr)
            max_dev = np.max(np.abs(alpha_arr - median_alpha))
            vmin = median_alpha - max_dev
            vmax = median_alpha + max_dev

            sc = ax.scatter(
                lon,
                lat,
                c=alpha_arr,
                s=0.1,
                alpha=0.3,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
            )
            fig.colorbar(
                sc,
                ax=ax,
                label=_COLORBAR_LABELS["spectral_index"],
                shrink=0.6,
            )
            ax.grid(True, alpha=0.3)
            xlabel, ylabel = _COORD_LABELS.get(coord, ("RA", "Dec"))
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(
                title
                if title is not None
                else self._auto_title("Spectral Index Sky Map")
            )
            return fig

        else:
            raise ValueError(
                f"Unknown plot_type='{plot_type}'. Choose 'histogram' or 'sky_map'."
            )

    # =====================================================================
    # HEALPix plots
    # =====================================================================

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
            return self.get_map_at_frequency(frequency), "Stokes I"
        maps_dict = {
            "Q": self._healpix_q_maps,
            "U": self._healpix_u_maps,
            "V": self._healpix_v_maps,
        }
        if stokes not in maps_dict:
            raise ValueError(
                f"Unknown stokes='{stokes}'. Choose from 'I', 'Q', 'U', 'V'."
            )
        stokes_maps = maps_dict[stokes]
        if stokes_maps is None:
            return None, f"Stokes {stokes}"
        freq_key = self._nearest_freq_key(stokes_maps, frequency)
        return stokes_maps[freq_key], f"Stokes {stokes}"

    @staticmethod
    def _nearest_freq_key(
        maps_dict: dict[float, np.ndarray], frequency: float
    ) -> float:
        """Find the nearest frequency key in a maps dict."""
        keys = np.array(list(maps_dict.keys()))
        idx = np.argmin(np.abs(keys - frequency))
        return float(keys[idx])

    def plot_healpix_map(
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

    def plot_multifreq_grid(
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
            all_freqs = self._observation_frequencies
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

    def plot_stokes(
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

        for i, stokes in enumerate(stokes_params):
            map_data, stokes_label = self._get_stokes_map(stokes, freq)

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

            is_quv = stokes in ("Q", "U", "V")
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
    # Dispatcher
    # =====================================================================

    def plot(
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

            - ``"sources"`` / ``"source_positions"`` — Mollweide scatter
            - ``"flux"`` / ``"flux_histogram"`` — flux histogram
            - ``"alpha"`` / ``"spectral_index"`` — spectral index plot
            - ``"map"`` / ``"healpix"`` — single HEALPix Mollweide
            - ``"grid"`` / ``"multifreq"`` — multi-frequency grid
            - ``"stokes"`` — Stokes IQUV 2x2 grid

        **kwargs
            Passed through to the underlying plot method.

        Returns
        -------
        matplotlib.figure.Figure
        """
        _dispatch = {
            "sources": self.plot_source_positions,
            "source_positions": self.plot_source_positions,
            "flux": self.plot_flux_histogram,
            "flux_histogram": self.plot_flux_histogram,
            "alpha": self.plot_spectral_index,
            "spectral_index": self.plot_spectral_index,
            "map": self.plot_healpix_map,
            "healpix": self.plot_healpix_map,
            "grid": self.plot_multifreq_grid,
            "multifreq": self.plot_multifreq_grid,
            "stokes": self.plot_stokes,
        }

        if plot_type == "auto":
            if self.mode == "healpix_multifreq":
                return self.plot_healpix_map(**kwargs)
            return self.plot_source_positions(**kwargs)

        if plot_type not in _dispatch:
            raise ValueError(
                f"Unknown plot_type='{plot_type}'. "
                f"Choose from: 'auto', {', '.join(repr(k) for k in _dispatch)}"
            )
        return _dispatch[plot_type](**kwargs)
