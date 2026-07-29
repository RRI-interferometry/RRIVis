"""Visualization modules for RadioSim.

This module provides interactive and static plotting capabilities for
canonical simulation results, antenna layouts, and sky models.

Every renderer export is lazy so that importing a typed visualization error, or
any other lightweight name, never pulls Bokeh, Plotly, Matplotlib, or healpy
into a process that only needs to reject a plot request.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from radiosim.visualization.bokeh_plots import (
        plot_antenna_layout,
        plot_antenna_layout_3d_plotly,
        plot_heatmaps,
        plot_modulus_vs_frequency,
        plot_visibility,
    )
    from radiosim.visualization.errors import (
        ResultBrowserError,
        ResultPlotContractError,
    )
    from radiosim.visualization.observability import ObservabilityBokehRenderer
    from radiosim.visualization.sky import (
        overlay_observability,
        plot_angular_power_spectrum,
        plot_cross_frequency_cell,
        plot_delay_spectrum,
        plot_flux_histogram,
        plot_frequency_correlation,
        plot_frequency_spectra,
        plot_frequency_waterfall,
        plot_healpix_map,
        plot_linear_polarization,
        plot_multifreq_grid,
        plot_multipole_bands,
        plot_pixel_histogram,
        plot_source_positions,
        plot_spectral_index,
        plot_stokes,
        plot_variance_spectrum,
    )


_BOKEH_EXPORTS = (
    "plot_antenna_layout",
    "plot_antenna_layout_3d_plotly",
    "plot_heatmaps",
    "plot_modulus_vs_frequency",
    "plot_visibility",
)

_ERROR_EXPORTS = (
    "ResultBrowserError",
    "ResultPlotContractError",
)

_SKY_EXPORTS = (
    "overlay_observability",
    "plot_angular_power_spectrum",
    "plot_cross_frequency_cell",
    "plot_delay_spectrum",
    "plot_flux_histogram",
    "plot_frequency_correlation",
    "plot_frequency_spectra",
    "plot_frequency_waterfall",
    "plot_healpix_map",
    "plot_linear_polarization",
    "plot_multifreq_grid",
    "plot_multipole_bands",
    "plot_pixel_histogram",
    "plot_source_positions",
    "plot_spectral_index",
    "plot_stokes",
    "plot_variance_spectrum",
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    **{name: ("radiosim.visualization.bokeh_plots", name) for name in _BOKEH_EXPORTS},
    **{name: ("radiosim.visualization.errors", name) for name in _ERROR_EXPORTS},
    **{name: ("radiosim.visualization.sky", name) for name in _SKY_EXPORTS},
    "ObservabilityBokehRenderer": (
        "radiosim.visualization.observability",
        "ObservabilityBokehRenderer",
    ),
}


def __getattr__(name: str) -> object:
    """Import one renderer module only when its public name is first used."""
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module 'radiosim.visualization' has no attribute {name!r}"
        ) from None
    return getattr(import_module(module_name), attribute)


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "ObservabilityBokehRenderer",
    "ResultBrowserError",
    "ResultPlotContractError",
    "overlay_observability",
    "plot_angular_power_spectrum",
    "plot_antenna_layout",
    "plot_antenna_layout_3d_plotly",
    "plot_cross_frequency_cell",
    "plot_delay_spectrum",
    "plot_flux_histogram",
    "plot_frequency_correlation",
    "plot_frequency_spectra",
    "plot_frequency_waterfall",
    "plot_healpix_map",
    "plot_heatmaps",
    "plot_linear_polarization",
    "plot_modulus_vs_frequency",
    "plot_multifreq_grid",
    "plot_multipole_bands",
    "plot_pixel_histogram",
    "plot_source_positions",
    "plot_spectral_index",
    "plot_stokes",
    "plot_variance_spectrum",
    "plot_visibility",
]
