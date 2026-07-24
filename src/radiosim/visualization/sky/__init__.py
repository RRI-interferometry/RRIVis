"""Module-level plotting functions for :class:`SkyModel`.

Each function takes the :class:`~radiosim.core.sky.SkyModel` it operates
on as its first positional argument; there is no construction step. The
previous ``SkyPlotter(sky).<group>.<method>()`` facade has been removed
in favour of this flat surface.

Sub-modules:

- :mod:`.point`      — ``plot_source_positions``, ``plot_flux_histogram``,
  ``plot_spectral_index``.
- :mod:`.healpix`    — ``plot_healpix_map``, ``plot_multifreq_grid``,
  ``plot_stokes``, ``plot_linear_polarization``.
- :mod:`.statistics` — ``plot_pixel_histogram``, ``plot_variance_spectrum``,
  ``plot_frequency_spectra``, ``plot_frequency_waterfall``.
- :mod:`.harmonics`  — ``plot_angular_power_spectrum``,
  ``plot_cross_frequency_cell``, ``plot_multipole_bands``,
  ``plot_frequency_correlation``, ``plot_delay_spectrum``.

``overlay_observability`` re-exports the same helper that was previously
attached to ``SkyPlotter``; its body lives in
:mod:`radiosim.core.observability.overlay`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from .harmonics import (
    plot_angular_power_spectrum,
    plot_cross_frequency_cell,
    plot_delay_spectrum,
    plot_frequency_correlation,
    plot_multipole_bands,
)
from .healpix import (
    plot_healpix_map,
    plot_linear_polarization,
    plot_multifreq_grid,
    plot_stokes,
)
from .point import plot_flux_histogram, plot_source_positions, plot_spectral_index
from .statistics import (
    plot_frequency_spectra,
    plot_frequency_waterfall,
    plot_pixel_histogram,
    plot_variance_spectrum,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from radiosim.core.observability import ObservabilityPlan


def overlay_observability(
    fig: Figure,
    plan: ObservabilityPlan,
    *,
    color: str = "white",
    linestyle: str = "--",
    linewidth: float = 1.5,
    alpha: float = 0.9,
    draw_footprint: bool = True,
    draw_beam: bool = True,
    beam_color: str = "yellow",
    beam_linestyle: str = "-",
    beam_linewidths: Mapping[float, float] | None = None,
    beam_alpha: float = 0.9,
    draw_tracks: bool = False,
    track_color: str = "yellow",
    track_marker_size: float = 20.0,
) -> Figure:
    """Draw an :class:`ObservabilityPlan`'s footprint on every Mollweide panel.

    Thin convenience wrapper around
    :func:`radiosim.core.observability.overlay.draw_observability_overlay`.
    Every style option is forwarded explicitly by name.
    """
    from radiosim.core.observability.overlay import draw_observability_overlay

    return draw_observability_overlay(
        fig,
        plan,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        draw_footprint=draw_footprint,
        draw_beam=draw_beam,
        beam_color=beam_color,
        beam_linestyle=beam_linestyle,
        beam_linewidths=beam_linewidths,
        beam_alpha=beam_alpha,
        draw_tracks=draw_tracks,
        track_color=track_color,
        track_marker_size=track_marker_size,
    )


__all__ = [
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
]
