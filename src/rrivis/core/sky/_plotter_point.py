"""Point-source plotting methods for SkyPlotter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._plotter_common import _COLORBAR_LABELS, _SkyPlotterBase

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class _SkyPlotterPointMixin(_SkyPlotterBase):
    """Point-source plot family."""

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
        """Scatter plot of source positions on a sky projection."""
        self._validate_plot_mode("point_sources")

        values, cbar_label = self._get_color_data(color_by)

        if log_color and color_by == "flux":
            values = np.log10(np.clip(values, 1e-6, None))
            cbar_label = _COLORBAR_LABELS["flux_log"]

        point = self._sky.point
        lon, lat = self._rotate_coords(point.ra_rad, point.dec_rad, coord)

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
        """Histogram of flux densities."""
        import matplotlib.pyplot as plt

        self._validate_plot_mode("point_sources")
        flux = self._sky.point.flux

        fig, ax = plt.subplots(figsize=figsize)

        if log_scale:
            positive = flux[flux > 0]
            if len(positive) == 0:
                raise ValueError("No positive flux values to plot on log scale.")
            bins = np.logspace(
                np.log10(positive.min()),
                np.log10(positive.max()),
                n_bins + 1,
            )
            ax.hist(positive, bins=bins, edgecolor="black", linewidth=0.3)
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:
            ax.hist(flux, bins=n_bins, edgecolor="black", linewidth=0.3)

        ax.set_xlabel("Flux density (Jy)")
        ax.set_ylabel("Number of sources")

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
        """Spectral index histogram or sky map."""
        import matplotlib.pyplot as plt

        self._validate_plot_mode("point_sources")
        alpha_arr = self._sky.point.spectral_index

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

        if plot_type == "sky_map":
            figsize = figsize or (14, 7)
            point = self._sky.point
            lon, lat = self._rotate_coords(point.ra_rad, point.dec_rad, coord)

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

        raise ValueError(
            f"Unknown plot_type='{plot_type}'. Choose 'histogram' or 'sky_map'."
        )
