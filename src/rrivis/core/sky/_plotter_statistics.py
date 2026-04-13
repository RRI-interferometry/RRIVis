"""Statistical HEALPix cube plots for SkyPlotter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._plotter_common import _SkyPlotterBase

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class _SkyPlotterStatisticsMixin(_SkyPlotterBase):
    """Statistical plot family for HEALPix cubes."""

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
        """Histogram of HEALPix pixel values at a single frequency."""
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
        """Mean, variance, and coefficient-of-variation vs frequency."""
        self._validate_plot_mode("healpix")
        if self._sky.healpix is None or len(self._sky.healpix.frequencies) < 2:
            raise ValueError(
                "variance_spectrum requires at least 2 observation frequencies."
            )

        import matplotlib.pyplot as plt

        freqs_mhz = np.asarray(self._sky.healpix.frequencies) / 1e6
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
        """Per-pixel frequency spectra, plus mean ± 1σ envelope."""
        self._validate_plot_mode("healpix")
        if self._sky.healpix is None or len(self._sky.healpix.frequencies) < 2:
            raise ValueError(
                "frequency_spectra requires at least 2 observation frequencies."
            )

        import matplotlib.pyplot as plt

        freqs_mhz = np.asarray(self._sky.healpix.frequencies) / 1e6
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
        """2-D (frequency × pixel) waterfall image."""
        self._validate_plot_mode("healpix")
        if self._sky.healpix is None or len(self._sky.healpix.frequencies) < 2:
            raise ValueError(
                "frequency_waterfall requires at least 2 observation frequencies."
            )

        import matplotlib.pyplot as plt

        freqs_mhz = np.asarray(self._sky.healpix.frequencies) / 1e6
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
