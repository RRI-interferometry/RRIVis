"""Harmonic and cross-frequency HEALPix plots for SkyPlotter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._plotter_common import _freq_label, _SkyPlotterBase

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class _SkyPlotterHarmonicMixin(_SkyPlotterBase):
    """Harmonic-analysis plot family for HEALPix cubes."""

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
        """Angular power spectrum ``C_ell`` or ``D_ell`` at one or more frequencies."""
        from ._analysis import compute_angular_power_spectrum

        self._validate_plot_mode("healpix")

        if representation not in ("c_ell", "d_ell", "both"):
            raise ValueError(
                f"representation must be 'c_ell', 'd_ell' or 'both', "
                f"got {representation!r}"
            )

        import matplotlib.pyplot as plt

        freq_arr = self._sky.healpix.frequencies if self._sky.healpix else None
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
        """Cross-power spectrum ``C_ell(nu_ref, nu_target)`` vs target frequency."""
        from ._analysis import compute_angular_power_spectrum, compute_cross_cell

        self._validate_plot_mode("healpix")
        freq_arr = self._sky.healpix.frequencies if self._sky.healpix else None
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
        """Decompose a HEALPix map into ell bands and show each as a panel."""
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
        """Cross-frequency Pearson correlation matrix plus decorrelation curve."""
        from ._analysis import compute_frequency_correlation

        self._validate_plot_mode("healpix")
        if self._sky.healpix is None or len(self._sky.healpix.frequencies) < 2:
            raise ValueError(
                "frequency_correlation requires at least 2 observation frequencies."
            )
        if pixel_stride < 1:
            raise ValueError(f"pixel_stride must be >= 1, got {pixel_stride}")

        import matplotlib.pyplot as plt

        freqs_mhz = np.asarray(self._sky.healpix.frequencies) / 1e6
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
        """Line-of-sight delay spectrum and optional ``P(k_parallel)``."""
        from ._analysis import F_21_HZ, compute_delay_spectrum, compute_kparallel

        self._validate_plot_mode("healpix")
        if self._sky.healpix is None or len(self._sky.healpix.frequencies) < 4:
            raise ValueError(
                "delay_spectrum requires at least 4 observation frequencies."
            )

        import matplotlib.pyplot as plt

        freqs_hz = np.asarray(self._sky.healpix.frequencies)
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
