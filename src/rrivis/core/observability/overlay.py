"""Matplotlib overlay for an :class:`ObservabilityPlan`.

Renderer-neutral free function: given any matplotlib :class:`Figure` that
contains one or more healpy projection axes (Mollweide, etc.), this draws
the plan's footprint contours, beam contours, and optional zenith-track
markers onto each projection panel.  The figure is returned in-place.

This module also exposes ``za_ring_points`` and ``draw_za_rings_on_figure``
for drawing constant zenith-angle rings on Mollweide projections — useful
when overlaying beam null/sidelobe positions on sky maps.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from rrivis.utils.coordinates import split_wrapped_path

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .planner import ObservabilityPlan


def draw_observability_overlay(
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
    beam_linewidths: dict[float, float] | None = None,
    beam_alpha: float = 0.9,
    draw_tracks: bool = False,
    track_color: str = "yellow",
    track_marker_size: float = 20.0,
) -> Figure:
    """Overlay an :class:`ObservabilityPlan`'s footprint on every Mollweide panel.

    Iterates through ``fig.axes`` and draws each segment of
    ``plan.footprint_contours`` (and optionally ``plan.beam_contours``)
    onto any healpy projection axis found. Safe to call after any
    plotter method that produces one or more Mollweide subplots; other
    axes (e.g. colourbars) are skipped.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure that already contains one or more healpy projection axes.
    plan : ObservabilityPlan
        Observability plan whose contours will be drawn.
    color, linestyle, linewidth, alpha
        Styling for the observable-strip footprint contour.
    draw_footprint : bool, default True
        Draw ``plan.footprint_contours``.
    draw_beam : bool, default True
        Draw ``plan.beam_contours`` when the plan carries a beam
        projection. Each ``(-N dB)`` level is drawn as a separate curve.
    beam_color, beam_linestyle, beam_alpha
        Styling for beam contours.
    beam_linewidths : dict[float, float], optional
        Per-level linewidth override, e.g. ``{-3.0: 2.0, -10.0: 1.0}``.
        Levels not listed fall back to ``linewidth``.
    draw_tracks : bool, default False
        If True, also draw the LST zenith-track RA samples as a line of
        ``track_color`` markers at ``plan.latitude_deg``.
    track_color, track_marker_size
        Styling for the track markers (used only when
        ``draw_tracks=True``).

    Returns
    -------
    matplotlib.figure.Figure
        The same ``fig``, with the overlay drawn in-place.
    """
    import healpy as hp
    import matplotlib.pyplot as plt

    projection_axes = [
        ax for ax in fig.axes if hasattr(ax, "projplot") or "Hpx" in type(ax).__name__
    ]

    track_ra = np.asarray(plan.track_ra_deg, dtype=float) if draw_tracks else None
    per_level_lw = beam_linewidths or {}
    beam_contours = plan.beam_contours if (draw_beam and plan.beam_contours) else None

    def _draw_segments(segments, *, c, ls, lw, a):
        for verts in segments:
            verts = np.asarray(verts, dtype=float)
            if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] < 2:
                continue
            # Split any segment that jumps >180° in RA (contour wrapping
            # across the anti-meridian) so healpy draws distinct pieces
            # instead of a straight line through the wrong hemisphere.
            for sub_x, sub_y in split_wrapped_path(verts[:, 0], verts[:, 1], 180.0):
                hp.projplot(
                    sub_x,
                    sub_y,
                    color=c,
                    linestyle=ls,
                    linewidth=lw,
                    alpha=a,
                    lonlat=True,
                    coord=None,
                )

    for ax in projection_axes:
        plt.sca(ax)
        if draw_footprint:
            for segment_group in plan.footprint_contours:
                _draw_segments(
                    segment_group,
                    c=color,
                    ls=linestyle,
                    lw=linewidth,
                    a=alpha,
                )
        if beam_contours is not None:
            for segment_group, level_db in beam_contours:
                lw = float(per_level_lw.get(float(level_db), linewidth))
                _draw_segments(
                    segment_group,
                    c=beam_color,
                    ls=beam_linestyle,
                    lw=lw,
                    a=beam_alpha,
                )
        if track_ra is not None and track_ra.size > 0:
            hp.projscatter(
                track_ra,
                np.full_like(track_ra, plan.latitude_deg),
                color=track_color,
                s=track_marker_size,
                lonlat=True,
                coord=None,
            )

    return fig


def za_ring_points(
    zenith_ra_deg: float,
    zenith_dec_deg: float,
    theta_deg: float,
    n: int = 721,
) -> tuple[np.ndarray, np.ndarray]:
    """Return RA/Dec samples tracing a constant zenith-angle ring.

    The ring is the locus of points at angular distance ``theta_deg`` from
    the pointing centre ``(zenith_ra_deg, zenith_dec_deg)`` — i.e. the
    intersection of the celestial sphere with a cone of half-angle
    ``theta_deg`` centred on the zenith.

    Parameters
    ----------
    zenith_ra_deg, zenith_dec_deg
        Centre of the ring in degrees.
    theta_deg
        Zenith-angle radius of the ring (degrees).
    n
        Number of samples drawn around the full azimuth circle.

    Returns
    -------
    (ra_deg, dec_deg)
        RA in ``[-180, 180]`` and Dec in ``[-90, 90]``, both shape ``(n,)``.
    """
    az = np.linspace(0.0, 2.0 * np.pi, n)
    t = np.deg2rad(theta_deg)
    dz = np.deg2rad(zenith_dec_deg)
    rz = np.deg2rad(zenith_ra_deg)
    sin_dec = np.cos(t) * np.sin(dz) + np.sin(t) * np.cos(dz) * np.cos(az)
    y = np.sin(t) * np.sin(az)
    x = np.cos(t) * np.cos(dz) - np.sin(t) * np.sin(dz) * np.cos(az)
    dra = np.arctan2(y, x)
    ra = np.degrees(rz + dra)
    dec = np.degrees(np.arcsin(np.clip(sin_dec, -1.0, 1.0)))
    ra = ((ra + 180.0) % 360.0) - 180.0
    return ra, dec


def draw_za_rings_on_figure(
    fig: Figure,
    *,
    zenith_ra_deg: float,
    zenith_dec_deg: float,
    theta_deg_list: Sequence[float],
    colors: Sequence[str] | None = None,
    linestyle: str = ":",
    linewidth: float = 1.3,
    alpha: float = 0.9,
    n: int = 721,
) -> Figure:
    """Draw constant zenith-angle rings on every healpy projection axis.

    Iterates through ``fig.axes`` and draws each ring as a polyline on any
    Mollweide/healpy projection it finds; non-projection axes are skipped.
    The 180-degree wrap is handled by
    :func:`rrivis.utils.coordinates.split_wrapped_path` so each ring is
    rendered as one or more sub-segments rather than as a straight line
    crossing the anti-meridian.

    Parameters
    ----------
    fig
        Figure that already contains one or more healpy projection axes.
    zenith_ra_deg, zenith_dec_deg
        Pointing centre of all rings (degrees).
    theta_deg_list
        Zenith-angle radii of the rings to draw, in degrees.
    colors
        Per-ring colour cycle.  If ``None``, all rings are drawn in white.
        Shorter cycles wrap around.
    linestyle, linewidth, alpha
        Common matplotlib styling shared by every ring.
    n
        Sample count per ring.

    Returns
    -------
    matplotlib.figure.Figure
        The same ``fig``, with rings drawn in-place.
    """
    import healpy as hp
    import matplotlib.pyplot as plt

    thetas = list(theta_deg_list)
    if not thetas:
        return fig

    if colors is None:
        ring_colors = ["white"] * len(thetas)
    else:
        color_list = list(colors) or ["white"]
        ring_colors = [color_list[i % len(color_list)] for i in range(len(thetas))]

    projection_axes = [
        ax for ax in fig.axes if hasattr(ax, "projplot") or "Hpx" in type(ax).__name__
    ]

    for ax in projection_axes:
        plt.sca(ax)
        for theta_deg, color in zip(thetas, ring_colors, strict=True):
            ra, dec = za_ring_points(
                zenith_ra_deg, zenith_dec_deg, float(theta_deg), n=n
            )
            for sub_x, sub_y in split_wrapped_path(ra, dec, 180.0):
                if len(sub_x) < 2:
                    continue
                hp.projplot(
                    sub_x,
                    sub_y,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=alpha,
                    lonlat=True,
                    coord=None,
                )

    return fig
