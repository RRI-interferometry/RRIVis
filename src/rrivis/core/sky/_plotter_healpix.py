"""HEALPix map plotting methods for SkyPlotter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._plotter_common import _COLORBAR_LABELS, _freq_label, _SkyPlotterBase

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class _SkyPlotterHealpixMixin(_SkyPlotterBase):
    """HEALPix plot family."""

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
        """Single HEALPix sky projection."""
        import matplotlib.pyplot as plt

        self._validate_plot_mode("healpix")
        freq = self._resolve_plot_frequency(frequency)
        map_data, stokes_label = self._get_stokes_map(stokes, freq)

        if map_data is None:
            raise ValueError(f"No {stokes_label} data available for this model.")

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
        """Grid of HEALPix projections at multiple frequencies."""
        import matplotlib.pyplot as plt

        self._validate_plot_mode("healpix")

        if frequencies is not None:
            freqs = np.asarray(frequencies, dtype=float)
        else:
            all_freqs = self._sky.healpix.frequencies
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

        maps = []
        for freq in freqs:
            map_data, _ = self._get_stokes_map(stokes, freq)
            if map_data is None:
                maps.append(None)
            else:
                maps.append(map_data)

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
            fig.suptitle(self._auto_title(f"Stokes {stokes}"), fontsize=14, y=0.98)

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
        """2x2 grid of Stokes I, Q, U, V at a single frequency."""
        import matplotlib.pyplot as plt

        self._validate_plot_mode("healpix")
        freq = self._resolve_plot_frequency(frequency)

        stokes_params = ["I", "Q", "U", "V"]
        fig = plt.figure(figsize=figsize)

        for i, stokes_name in enumerate(stokes_params):
            map_data, stokes_label = self._get_stokes_map(stokes_name, freq)

            if map_data is None:
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
