"""Composed plotting accessor for :class:`SkyModel`.

Provides matplotlib-based visualization grouped into four sub-objects:

* ``plotter.point``       — point-source scatter / histogram / spectral-index
* ``plotter.healpix``     — HEALPix map / multifreq grid / Stokes panels
* ``plotter.statistics``  — pixel histograms, frequency / variance spectra
* ``plotter.harmonics``   — Cℓ, multipole bands, delay spectra

Each sub-object is a thin :class:`_SkyPlotterBase` instance bound to the
input ``SkyModel``; they expose only their own family's methods so the
namespace is small and predictable.

All public plotting methods return :class:`matplotlib.figure.Figure`
without calling ``plt.show()``.

Usage::

    from radiosim.core.sky import SkyPlotter

    plotter = SkyPlotter(sky)
    fig = plotter.point.source_positions()
    fig = plotter.healpix.healpix_map()
    fig = plotter.statistics.frequency_spectra()
    fig = plotter.harmonics.angular_power_spectrum()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .common import _SkyPlotterBase
from .harmonics import HarmonicsPlotter
from .healpix import HealpixPlotter
from .point import PointPlotter
from .statistics import StatisticsPlotter

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ...containers.model import SkyModel


class SkyPlotter(_SkyPlotterBase):
    """Composed plotting facade for a :class:`SkyModel`.

    Sub-plotters are instantiated up-front in :meth:`__init__` and remain
    bound to the same SkyModel for the facade's lifetime.  Acquire a fresh
    facade after :meth:`SkyModel.replace` if you need plots reflecting the
    updated model.
    """

    def __init__(self, sky_model: SkyModel) -> None:
        super().__init__(sky_model)
        self.point = PointPlotter(sky_model)
        self.healpix = HealpixPlotter(sky_model)
        self.statistics = StatisticsPlotter(sky_model)
        self.harmonics = HarmonicsPlotter(sky_model)

    def overlay_observability(self, fig: Figure, plan: Any, **kwargs: Any) -> Figure:
        """Draw an :class:`ObservabilityPlan`'s footprint on every Mollweide panel.

        Thin convenience wrapper around
        :func:`radiosim.core.observability.overlay.draw_observability_overlay`.
        Accepts the same keyword arguments (``color``, ``linestyle``,
        ``draw_footprint``, ``draw_beam``, ``beam_color``,
        ``beam_linewidths``, ``draw_tracks`` …).
        """
        from radiosim.core.observability.overlay import draw_observability_overlay

        return draw_observability_overlay(fig, plan, **kwargs)


__all__ = [
    "HarmonicsPlotter",
    "HealpixPlotter",
    "PointPlotter",
    "SkyPlotter",
    "StatisticsPlotter",
]
