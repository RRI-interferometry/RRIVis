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

from typing import TYPE_CHECKING, Any

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


def overlay_observability(fig: Figure, plan: Any, **kwargs: Any) -> Figure:
    """Draw an :class:`ObservabilityPlan`'s footprint on every Mollweide panel.

    Thin convenience wrapper around
    :func:`radiosim.core.observability.overlay.draw_observability_overlay`.
    Accepts the same keyword arguments (``color``, ``linestyle``,
    ``draw_footprint``, ``draw_beam``, ``beam_color``, ``beam_linewidths``,
    ``draw_tracks`` …).
    """
    from radiosim.core.observability.overlay import draw_observability_overlay

    return draw_observability_overlay(fig, plan, **kwargs)


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
