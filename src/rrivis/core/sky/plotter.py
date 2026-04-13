"""Standalone plotting accessor for SkyModel.

Provides matplotlib-based visualization methods for point-source catalogs
and HEALPix diffuse maps. All public methods return a
:class:`matplotlib.figure.Figure` and never call ``plt.show()``.

Usage::

    from rrivis.core.sky.loaders import load_gleam
    from rrivis.core.sky.plotter import SkyPlotter

    sky = load_gleam(precision=precision, max_rows=1000)
    plotter = SkyPlotter(sky)
    fig = plotter.source_positions()
    fig = plotter.flux_histogram()
    fig = plotter("auto")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._plotter_harmonics import _SkyPlotterHarmonicMixin
from ._plotter_healpix import _SkyPlotterHealpixMixin
from ._plotter_point import _SkyPlotterPointMixin
from ._plotter_statistics import _SkyPlotterStatisticsMixin

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class SkyPlotter(
    _SkyPlotterPointMixin,
    _SkyPlotterHealpixMixin,
    _SkyPlotterStatisticsMixin,
    _SkyPlotterHarmonicMixin,
):
    """Plotting helper for a ``SkyModel`` instance."""

    def __call__(
        self,
        plot_type: str = "auto",
        **kwargs,
    ) -> Figure:
        """Convenience dispatcher for common plot types."""
        dispatch = {
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
            if self._sky.healpix is not None:
                return self.healpix_map(**kwargs)
            return self.source_positions(**kwargs)

        if plot_type not in dispatch:
            raise ValueError(
                f"Unknown plot_type='{plot_type}'. "
                f"Choose from: 'auto', {', '.join(repr(k) for k in dispatch)}"
            )
        return dispatch[plot_type](**kwargs)


__all__ = ["SkyPlotter"]
