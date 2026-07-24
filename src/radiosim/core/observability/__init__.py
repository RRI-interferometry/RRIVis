"""Observability planning: location + time + beam + sky → observability plan.

The planner and its output dataclasses are renderer-neutral; matplotlib or
Bokeh consumers import from here, not from :mod:`radiosim.visualization`.
"""

from radiosim.core.jones.beam.projection import (
    BeamContour,
    BeamSkyProjection,
    create_rgba_overlay,
)

from .errors import (
    InvalidObservabilityContextError,
    InvalidObservabilityReferenceError,
    ObservabilityBrowserError,
    ObservabilityError,
    ObservabilityOutputCollisionError,
    ObservabilityOutputError,
    ObservabilityRenderError,
    ObservabilitySkyUnavailableError,
    UnsupportedObservabilitySemanticsError,
)
from .geometry import (
    compute_beam_map_on_healpix,
    compute_beam_power_on_full_sky_grid,
)
from .lightcurves import (
    DriftScanLightcurve,
    compute_drift_scan_lightcurve,
    fractional_horizon_excess,
)
from .overlay import (
    draw_observability_overlay,
    draw_za_rings_on_figure,
    za_ring_points,
)
from .planner import (
    LSTObservabilityWindow,
    ObservabilityOptions,
    ObservabilityPlan,
    ObservabilityPlanner,
    ObservabilitySnapshot,
    ObservabilitySourceMetrics,
    ObservabilityWindow,
    UTCObservabilityWindow,
)

__all__ = [
    "ObservabilityError",
    "InvalidObservabilityReferenceError",
    "InvalidObservabilityContextError",
    "ObservabilitySkyUnavailableError",
    "UnsupportedObservabilitySemanticsError",
    "ObservabilityRenderError",
    "ObservabilityOutputError",
    "ObservabilityOutputCollisionError",
    "ObservabilityBrowserError",
    "UTCObservabilityWindow",
    "LSTObservabilityWindow",
    "ObservabilityWindow",
    "ObservabilityOptions",
    "ObservabilityPlanner",
    "ObservabilityPlan",
    "ObservabilitySnapshot",
    "ObservabilitySourceMetrics",
    "BeamSkyProjection",
    "BeamContour",
    "compute_beam_power_on_full_sky_grid",
    "compute_beam_map_on_healpix",
    "draw_observability_overlay",
    "za_ring_points",
    "draw_za_rings_on_figure",
    "DriftScanLightcurve",
    "compute_drift_scan_lightcurve",
    "fractional_horizon_excess",
    "create_rgba_overlay",
]
