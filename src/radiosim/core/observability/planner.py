"""Canonical renderer-neutral observability planning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from radiosim.core.beam import (
    BeamDisplayNormalizationError,
    BeamSystem,
    LoadedBeamState,
)
from radiosim.core.instrument import (
    AntennaId,
    ResolvedEarthLocation,
    ResolvedInstrument,
)
from radiosim.core.jones.beam.projection import (
    BeamContour,
    BeamSkyProjection,
    extract_contours,
)
from radiosim.utils.coordinates import (
    SIDEREAL_DEG_PER_SECOND,
    angular_separation_deg,
    axis_from_ra_deg,
    normalize_ra_deg,
    radec_to_za_az,
)

if TYPE_CHECKING:
    from radiosim.core.sky.containers.healpix import HealpixData
    from radiosim.core.sky.containers.model import SkyModel
    from radiosim.core.sky.containers.point import PointSourceData

from .errors import (
    InvalidObservabilityContextError,
    InvalidObservabilityReferenceError,
    ObservabilitySkyUnavailableError,
    UnsupportedObservabilitySemanticsError,
)
from .geometry import (
    _evaluate_reference_power,  # pyright: ignore[reportPrivateUsage]
    compute_beam_power_on_full_sky_grid,
)

AxisType = Literal["ra", "lst"]
BackgroundLayer = Literal["none", "diffuse"]
FootprintModel = Literal["beam_threshold", "manual_circular"]
DisplayMode = Literal["summary", "snapshots"]
BeamTimeReference = Literal["start", "midpoint", "end"]
ReferenceSelectionReason = Literal[
    "explicit",
    "homogeneous_default_minimum_number",
]


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InvalidObservabilityContextError(
            f"{field_name} must be a finite real number."
        )
    result = float(value)
    if not math.isfinite(result):
        raise InvalidObservabilityContextError(f"{field_name} must be finite.")
    return result


def _positive_float(value: object, field_name: str) -> float:
    result = _finite_float(value, field_name)
    if result <= 0.0:
        raise InvalidObservabilityContextError(f"{field_name} must be positive.")
    return result


def _strict_nonnegative_int(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise InvalidObservabilityContextError(
            f"{field_name} must be a strict nonnegative integer."
        )
    return value


def _owned_array(
    value: np.ndarray,
    *,
    field_name: str,
    dtype: np.dtype | type | None = None,
    ndim: int | None = None,
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError(f"{field_name} must be an exact ndarray")
    result = np.array(value, dtype=dtype, copy=True, order="C")
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{field_name} must be {ndim}-dimensional")
    result.setflags(write=False)
    return result


def _copy_contour_groups(
    value: tuple[tuple[np.ndarray, ...], ...],
    *,
    field_name: str,
) -> tuple[tuple[np.ndarray, ...], ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be an exact tuple")
    copied_groups: list[tuple[np.ndarray, ...]] = []
    for group_index, group in enumerate(value):
        if type(group) is not tuple:
            raise TypeError(f"{field_name}[{group_index}] must be an exact tuple")
        copied: list[np.ndarray] = []
        for segment_index, segment in enumerate(group):
            owned = _owned_array(
                segment,
                field_name=f"{field_name}[{group_index}][{segment_index}]",
                dtype=np.float64,
                ndim=2,
            )
            if owned.shape[1] != 2:
                raise ValueError(
                    f"{field_name}[{group_index}][{segment_index}] "
                    "must have shape (N, 2)"
                )
            copied.append(owned)
        copied_groups.append(tuple(copied))
    return tuple(copied_groups)


def _contour_groups(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    values: np.ndarray,
    levels: tuple[float, ...],
) -> tuple[tuple[np.ndarray, ...], ...]:
    """Extract numerical contours without constructing a renderer."""
    import contourpy

    generator = contourpy.contour_generator(
        x=x_grid,
        y=y_grid,
        z=values,
        name="serial",
        line_type="Separate",
    )
    return tuple(
        tuple(
            np.array(path, dtype=np.float64, copy=True, order="C")
            for path in generator.lines(level)
            if len(path) > 1
        )
        for level in levels
    )


@dataclass(frozen=True, slots=True)
class UTCObservabilityWindow:
    """One resolved UTC observability window."""

    kind: Literal["utc"]
    start_time_iso: str
    duration_seconds: float
    source: Literal["resolved_utc"]

    def __post_init__(self) -> None:
        if type(self.kind) is not str or self.kind != "utc":
            raise InvalidObservabilityContextError("UTC window kind must be 'utc'.")
        if type(self.start_time_iso) is not str or not self.start_time_iso.strip():
            raise InvalidObservabilityContextError(
                "start_time_iso must be a nonblank UTC timestamp."
            )
        if type(self.source) is not str or self.source != "resolved_utc":
            raise InvalidObservabilityContextError(
                "UTC window source must be 'resolved_utc'."
            )
        duration = _positive_float(self.duration_seconds, "duration_seconds")
        try:
            time_module: Any = import_module("astropy.time")
            _ = time_module.Time(
                self.start_time_iso,
                format="isot",
                scale="utc",
            )
        except (TypeError, ValueError) as exc:
            raise InvalidObservabilityContextError(
                f"start_time_iso={self.start_time_iso!r} is not a valid UTC timestamp."
            ) from exc
        object.__setattr__(self, "duration_seconds", duration)


@dataclass(frozen=True, slots=True)
class LSTObservabilityWindow:
    """One explicit LST observability window."""

    kind: Literal["lst"]
    start_hours: float
    end_hours: float
    wraps_midnight: bool
    source: Literal["explicit_lst"]
    beam_evaluation_time_mjd: float

    def __post_init__(self) -> None:
        if type(self.kind) is not str or self.kind != "lst":
            raise InvalidObservabilityContextError("LST window kind must be 'lst'.")
        if type(self.source) is not str or self.source != "explicit_lst":
            raise InvalidObservabilityContextError(
                "LST window source must be 'explicit_lst'."
            )
        start = _finite_float(self.start_hours, "start_hours")
        end = _finite_float(self.end_hours, "end_hours")
        if not 0.0 <= start < 24.0 or not 0.0 <= end < 24.0:
            raise InvalidObservabilityContextError("LST endpoints must be in [0, 24).")
        if type(self.wraps_midnight) is not bool:
            raise InvalidObservabilityContextError(
                "wraps_midnight must be an exact bool."
            )
        if self.wraps_midnight != (end < start):
            raise InvalidObservabilityContextError(
                "wraps_midnight must equal end_hours < start_hours."
            )
        evaluation_mjd = _finite_float(
            self.beam_evaluation_time_mjd,
            "beam_evaluation_time_mjd",
        )
        object.__setattr__(self, "start_hours", start)
        object.__setattr__(self, "end_hours", end)
        object.__setattr__(self, "beam_evaluation_time_mjd", evaluation_mjd)


ObservabilityWindow = UTCObservabilityWindow | LSTObservabilityWindow


@dataclass(frozen=True, slots=True)
class ObservabilityOptions:
    """Strict public observability planning options."""

    x_axis: AxisType = "ra"
    background_layer: BackgroundLayer = "none"
    footprint_model: FootprintModel = "beam_threshold"
    field_radius_deg: float | None = None
    mode: DisplayMode = "summary"
    snapshot_step_seconds: float = 3600.0
    footprint_step_seconds: float = 60.0
    beam_time_reference: BeamTimeReference = "midpoint"
    beam_contour_min_db: float = -40.0
    beam_contour_max_db: float = 0.0
    grid_resolution_deg: float = 1.0
    max_point_sources: int = 1000
    top_n_sources: int = 5
    nearby_source_count: int = 3
    nearby_buffer_deg: float = 10.0
    include_source_metrics: bool = False

    def __post_init__(self) -> None:
        if type(self.x_axis) is not str or self.x_axis not in {"ra", "lst"}:
            raise InvalidObservabilityContextError("x_axis must be 'ra' or 'lst'.")
        if type(self.background_layer) is not str or self.background_layer not in {
            "none",
            "diffuse",
        }:
            raise InvalidObservabilityContextError(
                "background_layer must be 'none' or 'diffuse'."
            )
        if type(self.footprint_model) is not str or self.footprint_model not in {
            "beam_threshold",
            "manual_circular",
        }:
            raise UnsupportedObservabilitySemanticsError(
                "Only 'beam_threshold' and 'manual_circular' footprint semantics "
                "are supported; rectangular, union, intersection, and multiple "
                "reference modes were removed."
            )
        if type(self.mode) is not str or self.mode not in {"summary", "snapshots"}:
            raise InvalidObservabilityContextError(
                "mode must be 'summary' or 'snapshots'."
            )
        if type(
            self.beam_time_reference
        ) is not str or self.beam_time_reference not in {"start", "midpoint", "end"}:
            raise InvalidObservabilityContextError(
                "beam_time_reference must be 'start', 'midpoint', or 'end'."
            )
        snapshot_step = _positive_float(
            self.snapshot_step_seconds,
            "snapshot_step_seconds",
        )
        footprint_step = _positive_float(
            self.footprint_step_seconds,
            "footprint_step_seconds",
        )
        grid_resolution = _positive_float(
            self.grid_resolution_deg,
            "grid_resolution_deg",
        )
        if grid_resolution > 10.0:
            raise InvalidObservabilityContextError(
                "grid_resolution_deg must be at most 10 degrees."
            )
        contour_min = _finite_float(
            self.beam_contour_min_db,
            "beam_contour_min_db",
        )
        contour_max = _finite_float(
            self.beam_contour_max_db,
            "beam_contour_max_db",
        )
        if not contour_min < contour_max <= 0.0:
            raise InvalidObservabilityContextError(
                "beam contour limits must satisfy min < max <= 0."
            )
        max_sources = _strict_nonnegative_int(
            self.max_point_sources,
            "max_point_sources",
        )
        top_sources = _strict_nonnegative_int(
            self.top_n_sources,
            "top_n_sources",
        )
        nearby_count = _strict_nonnegative_int(
            self.nearby_source_count,
            "nearby_source_count",
        )
        if top_sources > max_sources or nearby_count > max_sources:
            raise InvalidObservabilityContextError(
                "source ranking counts cannot exceed max_point_sources."
            )
        nearby_buffer = _finite_float(
            self.nearby_buffer_deg,
            "nearby_buffer_deg",
        )
        if not 0.0 <= nearby_buffer <= 180.0:
            raise InvalidObservabilityContextError(
                "nearby_buffer_deg must be in [0, 180]."
            )
        if type(self.include_source_metrics) is not bool:
            raise InvalidObservabilityContextError(
                "include_source_metrics must be an exact bool."
            )
        if self.footprint_model == "beam_threshold":
            if self.field_radius_deg is not None:
                raise InvalidObservabilityContextError(
                    "beam_threshold forbids field_radius_deg."
                )
            radius: float | None = None
        else:
            if self.field_radius_deg is None:
                raise InvalidObservabilityContextError(
                    "manual_circular requires field_radius_deg."
                )
            radius = _positive_float(self.field_radius_deg, "field_radius_deg")
            if radius > 90.0:
                raise InvalidObservabilityContextError(
                    "field_radius_deg must be at most 90 degrees."
                )

        object.__setattr__(self, "field_radius_deg", radius)
        object.__setattr__(self, "snapshot_step_seconds", snapshot_step)
        object.__setattr__(self, "footprint_step_seconds", footprint_step)
        object.__setattr__(self, "beam_contour_min_db", contour_min)
        object.__setattr__(self, "beam_contour_max_db", contour_max)
        object.__setattr__(self, "grid_resolution_deg", grid_resolution)
        object.__setattr__(self, "max_point_sources", max_sources)
        object.__setattr__(self, "top_n_sources", top_sources)
        object.__setattr__(self, "nearby_source_count", nearby_count)
        object.__setattr__(self, "nearby_buffer_deg", nearby_buffer)


@dataclass(frozen=True, slots=True, eq=False)
class ObservabilitySnapshot:
    """One immutable instantaneous sky-visibility snapshot."""

    label: str
    utc_iso: str | None
    lst_hours: float
    zenith_ra_deg: float
    zenith_dec_deg: float
    footprint_mask: np.ndarray
    visible_source_mask: np.ndarray | None

    def __post_init__(self) -> None:
        if type(self.label) is not str or not self.label:
            raise TypeError("label must be a nonblank exact string")
        if self.utc_iso is not None and type(self.utc_iso) is not str:
            raise TypeError("utc_iso must be None or an exact string")
        for name in ("lst_hours", "zenith_ra_deg", "zenith_dec_deg"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(f"{name} must be an exact finite float")
        footprint = _owned_array(
            self.footprint_mask,
            field_name="footprint_mask",
            dtype=np.bool_,
            ndim=2,
        )
        visible = (
            None
            if self.visible_source_mask is None
            else _owned_array(
                self.visible_source_mask,
                field_name="visible_source_mask",
                dtype=np.bool_,
                ndim=1,
            )
        )
        object.__setattr__(self, "footprint_mask", footprint)
        object.__setattr__(self, "visible_source_mask", visible)

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True, eq=False)
class ObservabilitySourceMetrics:
    """Immutable point-source positions and visibility metrics."""

    ra_deg: np.ndarray
    dec_deg: np.ndarray
    flux_jy: np.ndarray
    x_coord: np.ndarray
    source_name: np.ndarray | None
    visible_any: np.ndarray
    visible_fraction: np.ndarray
    min_separation_deg: np.ndarray
    first_visible_index: np.ndarray
    last_visible_index: np.ndarray
    top_visible_indices: np.ndarray
    nearby_indices: np.ndarray

    def __post_init__(self) -> None:
        copied: dict[str, np.ndarray] = {}
        for name, dtype in (
            ("ra_deg", np.float64),
            ("dec_deg", np.float64),
            ("flux_jy", np.float64),
            ("x_coord", np.float64),
            ("visible_any", np.bool_),
            ("visible_fraction", np.float64),
            ("min_separation_deg", np.float64),
            ("first_visible_index", np.int64),
            ("last_visible_index", np.int64),
            ("top_visible_indices", np.int64),
            ("nearby_indices", np.int64),
        ):
            copied[name] = _owned_array(
                cast(np.ndarray, getattr(self, name)),
                field_name=name,
                dtype=dtype,
                ndim=1,
            )
        source_name = (
            None
            if self.source_name is None
            else _owned_array(
                self.source_name,
                field_name="source_name",
                ndim=1,
            )
        )
        count = len(copied["ra_deg"])
        for name in (
            "dec_deg",
            "flux_jy",
            "x_coord",
            "visible_any",
            "visible_fraction",
            "min_separation_deg",
            "first_visible_index",
            "last_visible_index",
        ):
            if len(copied[name]) != count:
                raise ValueError(f"{name} must match ra_deg length")
        if source_name is not None and len(source_name) != count:
            raise ValueError("source_name must match ra_deg length")
        for name, value in copied.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "source_name", source_name)

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True, eq=False)
class ObservabilityPlan:
    """Immutable renderer-neutral sky-visibility description."""

    x_axis: AxisType
    mode: DisplayMode
    title: str
    frequency_hz: float
    channel_index: int
    field_radius_deg: float | None
    latitude_deg: float
    longitude_deg: float
    height_m: float
    observation_start_iso: str | None
    observation_end_iso: str | None
    lst_start_hours: float | None
    lst_end_hours: float | None
    window_source: Literal["resolved_utc", "explicit_lst"]
    track_labels: tuple[str, ...]
    track_time_isos: tuple[str | None, ...]
    track_lst_hours: np.ndarray
    track_ra_deg: np.ndarray
    ra_grid_deg: np.ndarray
    dec_grid_deg: np.ndarray
    background_layer: BackgroundLayer
    projected_background: np.ndarray | None
    footprint_model: FootprintModel
    footprint_provenance: Literal[
        "reference_beam_half_power",
        "manual_circular_display_approximation",
    ]
    footprint_mask: np.ndarray
    footprint_contours: tuple[tuple[np.ndarray, ...], ...]
    snapshots: tuple[ObservabilitySnapshot, ...]
    source_metrics: ObservabilitySourceMetrics | None
    beam_projection: BeamSkyProjection
    beam_contours: tuple[BeamContour, ...]
    beam_time_reference: BeamTimeReference
    beam_time_reference_lst_hours: float
    beam_time_reference_mjd: float
    beam_time_reference_ra_deg: float
    beam_state_fingerprint: str
    reference_antenna: AntennaId
    reference_handler_id: str
    reference_scientific_fingerprint: str
    reference_selection_reason: ReferenceSelectionReason
    power_convention: Literal["half_trace_unpolarized"]

    def __post_init__(self) -> None:
        array_fields: tuple[tuple[str, np.dtype | type, int], ...] = (
            ("track_lst_hours", np.float64, 1),
            ("track_ra_deg", np.float64, 1),
            ("ra_grid_deg", np.float64, 1),
            ("dec_grid_deg", np.float64, 1),
            ("footprint_mask", np.bool_, 2),
        )
        for name, dtype, ndim in array_fields:
            object.__setattr__(
                self,
                name,
                _owned_array(
                    cast(np.ndarray, getattr(self, name)),
                    field_name=name,
                    dtype=dtype,
                    ndim=ndim,
                ),
            )
        background = (
            None
            if self.projected_background is None
            else _owned_array(
                self.projected_background,
                field_name="projected_background",
                dtype=np.float64,
                ndim=2,
            )
        )
        object.__setattr__(self, "projected_background", background)
        object.__setattr__(
            self,
            "footprint_contours",
            _copy_contour_groups(
                self.footprint_contours,
                field_name="footprint_contours",
            ),
        )
        if type(self.snapshots) is not tuple or any(
            type(snapshot) is not ObservabilitySnapshot for snapshot in self.snapshots
        ):
            raise TypeError("snapshots must contain exact ObservabilitySnapshot values")
        copied_snapshots = tuple(
            ObservabilitySnapshot(
                label=snapshot.label,
                utc_iso=snapshot.utc_iso,
                lst_hours=snapshot.lst_hours,
                zenith_ra_deg=snapshot.zenith_ra_deg,
                zenith_dec_deg=snapshot.zenith_dec_deg,
                footprint_mask=snapshot.footprint_mask,
                visible_source_mask=snapshot.visible_source_mask,
            )
            for snapshot in self.snapshots
        )
        object.__setattr__(self, "snapshots", copied_snapshots)
        if self.source_metrics is not None and (
            type(self.source_metrics) is not ObservabilitySourceMetrics
        ):
            raise TypeError("source_metrics must be an exact public model")
        if type(self.beam_projection) is not BeamSkyProjection:
            raise TypeError("beam_projection must be an exact BeamSkyProjection")
        projection = BeamSkyProjection(
            ra_grid_deg=self.beam_projection.ra_grid_deg,
            dec_grid_deg=self.beam_projection.dec_grid_deg,
            power_db=self.beam_projection.power_db,
            zenith_ra_deg=self.beam_projection.zenith_ra_deg,
            zenith_dec_deg=self.beam_projection.zenith_dec_deg,
            max_za_deg=self.beam_projection.max_za_deg,
        )
        object.__setattr__(self, "beam_projection", projection)
        if type(self.beam_contours) is not tuple or any(
            type(contour) is not BeamContour for contour in self.beam_contours
        ):
            raise TypeError("beam_contours must contain exact BeamContour values")
        contours = tuple(
            BeamContour(level_db=contour.level_db, segments=contour.segments)
            for contour in self.beam_contours
        )
        object.__setattr__(self, "beam_contours", contours)
        if type(self.reference_antenna) is not AntennaId:
            raise TypeError("reference_antenna must be an exact AntennaId")
        object.__setattr__(
            self,
            "reference_antenna",
            AntennaId(
                self.reference_antenna.number,
                self.reference_antenna.name,
            ),
        )
        if type(self.track_labels) is not tuple or any(
            type(value) is not str for value in self.track_labels
        ):
            raise TypeError("track_labels must be an exact tuple of strings")
        if type(self.track_time_isos) is not tuple or any(
            value is not None and type(value) is not str
            for value in self.track_time_isos
        ):
            raise TypeError("track_time_isos must be an exact tuple")

    def provenance_snapshot(self) -> dict[str, object]:
        """Return JSON-safe scalar identity, window, beam, and power facts."""
        return {
            "x_axis": self.x_axis,
            "mode": self.mode,
            "title": self.title,
            "frequency_hz": self.frequency_hz,
            "channel_index": self.channel_index,
            "field_radius_deg": self.field_radius_deg,
            "location": {
                "latitude_deg": self.latitude_deg,
                "longitude_deg": self.longitude_deg,
                "height_m": self.height_m,
            },
            "window": {
                "source": self.window_source,
                "observation_start_iso": self.observation_start_iso,
                "observation_end_iso": self.observation_end_iso,
                "lst_start_hours": self.lst_start_hours,
                "lst_end_hours": self.lst_end_hours,
            },
            "background_layer": self.background_layer,
            "footprint_model": self.footprint_model,
            "footprint_provenance": self.footprint_provenance,
            "beam_time_reference": self.beam_time_reference,
            "beam_time_reference_lst_hours": self.beam_time_reference_lst_hours,
            "beam_time_reference_mjd": self.beam_time_reference_mjd,
            "beam_time_reference_ra_deg": self.beam_time_reference_ra_deg,
            "beam_state_fingerprint": self.beam_state_fingerprint,
            "reference_antenna": {
                "number": self.reference_antenna.number,
                "name": self.reference_antenna.name,
            },
            "reference_handler_id": self.reference_handler_id,
            "reference_scientific_fingerprint": (self.reference_scientific_fingerprint),
            "reference_selection_reason": self.reference_selection_reason,
            "power_convention": self.power_convention,
        }

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class _ObservabilityContext:
    instrument: ResolvedInstrument
    beam_state: LoadedBeamState
    beam_system: BeamSystem
    reference_antenna: AntennaId
    reference_handler_id: str
    reference_selection_reason: ReferenceSelectionReason
    location: ResolvedEarthLocation
    frequency_hz: float
    channel_index: int
    window: ObservabilityWindow
    sky_model: SkyModel | None
    options: ObservabilityOptions


@dataclass(frozen=True, slots=True)
class _TrackSamples:
    labels: tuple[str, ...]
    time_isos: tuple[str | None, ...]
    lst_hours: np.ndarray
    ra_deg: np.ndarray
    raw_ra_deg: np.ndarray
    mjd: np.ndarray


class ObservabilityPlanner:
    """Build one immutable plan from resolved instrument and BeamSystem state."""

    def __init__(
        self,
        *,
        instrument: ResolvedInstrument,
        beam_system: BeamSystem,
        reference_antenna: AntennaId,
        reference_selection_reason: ReferenceSelectionReason,
        location: ResolvedEarthLocation,
        frequency_hz: float,
        channel_index: int,
        window: ObservabilityWindow,
        sky_model: SkyModel | None,
        options: ObservabilityOptions,
    ) -> None:
        self._context = self._validate_context(
            instrument=instrument,
            beam_system=beam_system,
            reference_antenna=reference_antenna,
            reference_selection_reason=reference_selection_reason,
            location=location,
            frequency_hz=frequency_hz,
            channel_index=channel_index,
            window=window,
            sky_model=sky_model,
            options=options,
        )

    @staticmethod
    def _validate_context(
        *,
        instrument: ResolvedInstrument,
        beam_system: BeamSystem,
        reference_antenna: AntennaId,
        reference_selection_reason: ReferenceSelectionReason,
        location: ResolvedEarthLocation,
        frequency_hz: float,
        channel_index: int,
        window: ObservabilityWindow,
        sky_model: SkyModel | None,
        options: ObservabilityOptions,
    ) -> _ObservabilityContext:
        if type(instrument) is not ResolvedInstrument:
            raise TypeError("instrument must be an exact ResolvedInstrument")
        if type(beam_system) is not BeamSystem:
            raise TypeError("beam_system must be an exact BeamSystem")
        if type(reference_antenna) is not AntennaId:
            raise InvalidObservabilityReferenceError(
                "reference_antenna must be an exact canonical AntennaId."
            )
        if type(location) is not ResolvedEarthLocation:
            raise TypeError("location must be an exact ResolvedEarthLocation")
        if location != instrument.location:
            raise InvalidObservabilityContextError(
                "location must equal the resolved instrument location."
            )
        if type(channel_index) is not int or channel_index < 0:
            raise InvalidObservabilityContextError(
                "channel_index must be a strict nonnegative integer."
            )
        if type(frequency_hz) is not float or not math.isfinite(frequency_hz):
            raise InvalidObservabilityContextError(
                "frequency_hz must be an exact finite float."
            )
        if type(window) not in (UTCObservabilityWindow, LSTObservabilityWindow):
            raise InvalidObservabilityContextError(
                "window must be an exact public observability window model."
            )
        window.__post_init__()
        if type(options) is not ObservabilityOptions:
            raise InvalidObservabilityContextError(
                "options must be an exact ObservabilityOptions."
            )
        options.__post_init__()
        if sky_model is not None:
            from radiosim.core.sky.containers.model import SkyModel as SkyModelType

            if type(sky_model) is not SkyModelType:
                raise TypeError("sky_model must be None or an exact SkyModel")

        state = beam_system.state
        if state.resolved.instrument_fingerprint != (
            instrument.provenance.instrument_sha256
        ):
            raise InvalidObservabilityContextError(
                "BeamSystem instrument fingerprint does not match instrument."
            )
        handler_by_id = {handler.handler_id: handler for handler in state.handlers}
        assignment_map = dict(state.assignment_handler_ids)
        reference_handler_id = assignment_map.get(reference_antenna)
        if reference_handler_id is None:
            raise InvalidObservabilityReferenceError(
                "reference_antenna is not covered by the canonical BeamSystem."
            )
        fingerprints = {
            handler_by_id[handler_id].scientific_fingerprint
            for _antenna_id, handler_id in state.assignment_handler_ids
        }
        if (
            cast(object, reference_selection_reason)
            == "homogeneous_default_minimum_number"
        ):
            minimum = min(
                antenna_id.number
                for antenna_id, _handler_id in state.assignment_handler_ids
            )
            if len(fingerprints) != 1 or reference_antenna.number != minimum:
                raise InvalidObservabilityReferenceError(
                    "homogeneous default requires equivalent handler fingerprints "
                    "and the minimum canonical antenna number."
                )
        elif cast(object, reference_selection_reason) != "explicit":
            raise InvalidObservabilityReferenceError(
                "reference_selection_reason must be 'explicit' or "
                "'homogeneous_default_minimum_number'."
            )

        frequency_axis = tuple(
            value
            for value, _scale in (state.handlers[0].voltage_feature_scale_by_frequency)
        )
        if channel_index >= len(frequency_axis):
            raise InvalidObservabilityContextError(
                "channel_index is outside the BeamSystem observation channels."
            )
        if frequency_axis[channel_index] != frequency_hz:
            raise InvalidObservabilityContextError(
                "frequency_hz must exactly equal the selected BeamSystem channel."
            )
        if options.background_layer == "diffuse" and (
            sky_model is None or sky_model.healpix is None
        ):
            raise ObservabilitySkyUnavailableError(
                "background_layer='diffuse' requires an already prepared HEALPix sky."
            )
        if options.include_source_metrics and (
            sky_model is None or sky_model.point is None
        ):
            raise ObservabilitySkyUnavailableError(
                "include_source_metrics=True requires an already prepared "
                "point-source sky."
            )

        return _ObservabilityContext(
            instrument=instrument,
            beam_state=state,
            beam_system=beam_system,
            reference_antenna=AntennaId(
                reference_antenna.number,
                reference_antenna.name,
            ),
            reference_handler_id=reference_handler_id,
            reference_selection_reason=reference_selection_reason,
            location=location,
            frequency_hz=frequency_hz,
            channel_index=channel_index,
            window=window,
            sky_model=sky_model,
            options=options,
        )

    def build(self) -> ObservabilityPlan:
        """Build the plan without rendering, I/O, browser, solver, or sky loading."""
        context = self._context
        options = context.options
        step = options.grid_resolution_deg
        ra_grid_deg = np.arange(-180.0, 180.0 + 0.5 * step, step)
        dec_grid_deg = np.arange(-90.0, 90.0 + 0.5 * step, step)
        track = self._build_track_samples(options.footprint_step_seconds)
        snapshots_track = self._build_track_samples(options.snapshot_step_seconds)
        (
            reference_lst_hours,
            reference_mjd,
            reference_ra_deg,
        ) = self._beam_time_reference()

        projected_background = self._project_background(
            ra_grid_deg,
            dec_grid_deg,
        )
        footprint_mask = self._footprint_for_track(
            ra_grid_deg,
            dec_grid_deg,
            track,
        )
        footprint_contours = _contour_groups(
            ra_grid_deg,
            dec_grid_deg,
            footprint_mask.astype(np.float64),
            (0.5,),
        )
        source_metrics, snapshot_visibility = self._build_source_metrics(
            track,
            snapshots_track,
        )
        snapshots = self._build_snapshots(
            snapshots_track,
            snapshot_visibility,
            ra_grid_deg,
            dec_grid_deg,
        )
        beam_projection, beam_contours = self._build_beam_projection(
            ra_grid_deg,
            dec_grid_deg,
            reference_ra_deg,
            reference_mjd,
        )
        handler = {value.handler_id: value for value in context.beam_state.handlers}[
            context.reference_handler_id
        ]
        fingerprint = handler.scientific_fingerprint
        title = self._build_title(
            track,
            fingerprint=fingerprint,
        )

        if type(context.window) is UTCObservabilityWindow:
            observation_start_iso = track.time_isos[0]
            observation_end_iso = track.time_isos[-1]
        else:
            observation_start_iso = None
            observation_end_iso = None

        return ObservabilityPlan(
            x_axis=options.x_axis,
            mode=options.mode,
            title=title,
            frequency_hz=context.frequency_hz,
            channel_index=context.channel_index,
            field_radius_deg=options.field_radius_deg,
            latitude_deg=context.location.latitude_deg,
            longitude_deg=context.location.longitude_deg,
            height_m=context.location.height_m,
            observation_start_iso=observation_start_iso,
            observation_end_iso=observation_end_iso,
            lst_start_hours=float(track.lst_hours[0]),
            lst_end_hours=float(track.lst_hours[-1]),
            window_source=context.window.source,
            track_labels=track.labels,
            track_time_isos=track.time_isos,
            track_lst_hours=np.asarray(track.lst_hours),
            track_ra_deg=np.asarray(track.ra_deg),
            ra_grid_deg=ra_grid_deg,
            dec_grid_deg=dec_grid_deg,
            background_layer=options.background_layer,
            projected_background=projected_background,
            footprint_model=options.footprint_model,
            footprint_provenance=(
                "reference_beam_half_power"
                if options.footprint_model == "beam_threshold"
                else "manual_circular_display_approximation"
            ),
            footprint_mask=footprint_mask,
            footprint_contours=footprint_contours,
            snapshots=snapshots,
            source_metrics=source_metrics,
            beam_projection=beam_projection,
            beam_contours=beam_contours,
            beam_time_reference=options.beam_time_reference,
            beam_time_reference_lst_hours=reference_lst_hours,
            beam_time_reference_mjd=reference_mjd,
            beam_time_reference_ra_deg=reference_ra_deg,
            beam_state_fingerprint=context.beam_state.loaded_fingerprint,
            reference_antenna=context.reference_antenna,
            reference_handler_id=context.reference_handler_id,
            reference_scientific_fingerprint=fingerprint,
            reference_selection_reason=context.reference_selection_reason,
            power_convention="half_trace_unpolarized",
        )

    @staticmethod
    def _sample_offsets(total_seconds: float, step_seconds: float) -> np.ndarray:
        if total_seconds == 0.0:
            return np.array([0.0], dtype=np.float64)
        offsets = np.arange(
            0.0,
            total_seconds + 0.5 * step_seconds,
            step_seconds,
            dtype=np.float64,
        )
        offsets = offsets[offsets <= total_seconds]
        if offsets.size == 0 or offsets[-1] < total_seconds:
            offsets = np.append(offsets, total_seconds)
        return offsets

    def _build_track_samples(self, step_seconds: float) -> _TrackSamples:
        context = self._context
        window = context.window
        if type(window) is LSTObservabilityWindow:
            width_hours = (window.end_hours - window.start_hours) % 24.0
            total_seconds = width_hours * 15.0 / SIDEREAL_DEG_PER_SECOND
            offsets = self._sample_offsets(total_seconds, step_seconds)
            raw_ra = window.start_hours * 15.0 + (offsets * SIDEREAL_DEG_PER_SECOND)
            lst_hours = (raw_ra / 15.0) % 24.0
            ra_deg = np.asarray(normalize_ra_deg(raw_ra), dtype=np.float64)
            labels = tuple(f"LST {value:.6f}h" for value in lst_hours)
            time_isos = tuple(None for _value in lst_hours)
            mjd = np.full(
                len(lst_hours),
                window.beam_evaluation_time_mjd,
                dtype=np.float64,
            )
            return _TrackSamples(
                labels=labels,
                time_isos=time_isos,
                lst_hours=np.asarray(lst_hours, dtype=np.float64),
                ra_deg=ra_deg,
                raw_ra_deg=np.asarray(raw_ra, dtype=np.float64),
                mjd=mjd,
            )

        utc_window = cast(UTCObservabilityWindow, window)
        u: Any = import_module("astropy.units")
        coordinates_module: Any = import_module("astropy.coordinates")
        time_module: Any = import_module("astropy.time")

        earth_location = coordinates_module.EarthLocation.from_geodetic(
            context.location.longitude_deg * u.deg,
            context.location.latitude_deg * u.deg,
            context.location.height_m * u.m,
        )
        offsets = self._sample_offsets(utc_window.duration_seconds, step_seconds)
        start = time_module.Time(
            utc_window.start_time_iso,
            format="isot",
            scale="utc",
            location=earth_location,
        )
        times = start + time_module.TimeDelta(offsets, format="sec")
        lst_hours = np.asarray(
            times.sidereal_time("apparent").hour,
            dtype=np.float64,
        )
        wrapped_ra = lst_hours * 15.0
        raw_ra = np.rad2deg(np.unwrap(np.deg2rad(wrapped_ra)))
        ra_deg = np.asarray(normalize_ra_deg(wrapped_ra), dtype=np.float64)
        labels = tuple(str(value) for value in np.atleast_1d(times.isot))
        time_isos = tuple(labels)
        return _TrackSamples(
            labels=labels,
            time_isos=time_isos,
            lst_hours=lst_hours,
            ra_deg=ra_deg,
            raw_ra_deg=np.asarray(raw_ra, dtype=np.float64),
            mjd=np.asarray(times.mjd, dtype=np.float64),
        )

    def _beam_time_reference(self) -> tuple[float, float, float]:
        context = self._context
        window = context.window
        choice = context.options.beam_time_reference
        if type(window) is LSTObservabilityWindow:
            width = (window.end_hours - window.start_hours) % 24.0
            offset = {
                "start": 0.0,
                "midpoint": width / 2.0,
                "end": width,
            }[choice]
            lst_hours = (window.start_hours + offset) % 24.0
            return (
                float(lst_hours),
                window.beam_evaluation_time_mjd,
                float(normalize_ra_deg(lst_hours * 15.0)),
            )

        utc_window = cast(UTCObservabilityWindow, window)
        duration_offset = {
            "start": 0.0,
            "midpoint": utc_window.duration_seconds / 2.0,
            "end": utc_window.duration_seconds,
        }[choice]
        samples = self._build_track_samples(max(utc_window.duration_seconds, 1.0))
        if choice == "start":
            index = 0
        elif choice == "end":
            index = -1
        else:
            u: Any = import_module("astropy.units")
            coordinates_module: Any = import_module("astropy.coordinates")
            time_module: Any = import_module("astropy.time")

            earth_location = coordinates_module.EarthLocation.from_geodetic(
                context.location.longitude_deg * u.deg,
                context.location.latitude_deg * u.deg,
                context.location.height_m * u.m,
            )
            obstime = time_module.Time(
                utc_window.start_time_iso,
                format="isot",
                scale="utc",
                location=earth_location,
            )
            obstime += time_module.TimeDelta(duration_offset, format="sec")
            lst = float(obstime.sidereal_time("apparent").hour)
            return (
                lst,
                float(obstime.mjd),
                float(normalize_ra_deg(lst * 15.0)),
            )
        return (
            float(samples.lst_hours[index]),
            float(samples.mjd[index]),
            float(samples.ra_deg[index]),
        )

    def _normalized_power(
        self,
        za_rad: np.ndarray,
        az_rad: np.ndarray,
        *,
        time_mjd: float,
    ) -> np.ndarray:
        context = self._context
        power = _evaluate_reference_power(
            beam_system=context.beam_system,
            reference_antenna=context.reference_antenna,
            zenith_angle_rad=za_rad,
            azimuth_rad=az_rad,
            frequency_hz=context.frequency_hz,
            time_mjd=float(time_mjd),
        )
        peak = float(
            _evaluate_reference_power(
                beam_system=context.beam_system,
                reference_antenna=context.reference_antenna,
                zenith_angle_rad=np.array([0.0], dtype=np.float64),
                azimuth_rad=np.array([0.0], dtype=np.float64),
                frequency_hz=context.frequency_hz,
                time_mjd=float(time_mjd),
            )[0]
        )
        if not math.isfinite(peak) or peak <= 0.0:
            raise BeamDisplayNormalizationError(
                "Selected observability reference beam has no finite positive "
                "zenith normalization."
            )
        return np.asarray(power / peak, dtype=np.float64)

    def _membership(
        self,
        ra_deg: np.ndarray,
        dec_deg: np.ndarray,
        *,
        zenith_ra_deg: float,
        time_mjd: float,
    ) -> np.ndarray:
        options = self._context.options
        if options.footprint_model == "manual_circular":
            return angular_separation_deg(
                ra_deg,
                dec_deg,
                zenith_ra_deg,
                self._context.location.latitude_deg,
            ) <= cast(float, options.field_radius_deg)
        za_rad, az_rad = radec_to_za_az(
            ra_deg,
            dec_deg,
            zenith_ra_deg=zenith_ra_deg,
            zenith_dec_deg=self._context.location.latitude_deg,
        )
        return (
            self._normalized_power(
                za_rad,
                az_rad,
                time_mjd=time_mjd,
            )
            >= 0.5
        )

    def _footprint_for_track(
        self,
        ra_grid_deg: np.ndarray,
        dec_grid_deg: np.ndarray,
        track: _TrackSamples,
    ) -> np.ndarray:
        ra_mesh, dec_mesh = np.meshgrid(ra_grid_deg, dec_grid_deg)
        mask = np.zeros(ra_mesh.shape, dtype=np.bool_)
        for zenith_ra, time_mjd in zip(track.ra_deg, track.mjd, strict=True):
            mask |= self._membership(
                ra_mesh,
                dec_mesh,
                zenith_ra_deg=float(zenith_ra),
                time_mjd=float(time_mjd),
            )
        return mask

    def _project_background(
        self,
        ra_grid_deg: np.ndarray,
        dec_grid_deg: np.ndarray,
    ) -> np.ndarray | None:
        context = self._context
        if context.options.background_layer == "none":
            return None
        sky = cast("SkyModel", context.sky_model)
        healpix_payload = cast("HealpixData", sky.healpix)
        healpix = healpix_payload.require_dense("ObservabilityPlanner.background_layer")
        matches = np.flatnonzero(
            np.asarray(healpix.frequencies) == context.frequency_hz
        )
        if len(matches) != 1:
            raise InvalidObservabilityContextError(
                "Prepared HEALPix sky does not contain the exact selected channel."
            )
        sky_map = np.asarray(healpix.maps[int(matches[0])], dtype=np.float64)
        ra_mesh, dec_mesh = np.meshgrid(ra_grid_deg, dec_grid_deg)
        if healpix.coordinate_frame == "galactic":
            u: Any = import_module("astropy.units")
            coordinates_module: Any = import_module("astropy.coordinates")

            coordinates = coordinates_module.SkyCoord(
                ra=ra_mesh.ravel() * u.deg,
                dec=dec_mesh.ravel() * u.deg,
                frame="icrs",
            ).galactic
            phi = np.asarray(coordinates.l.rad).reshape(ra_mesh.shape)
            theta = (np.pi / 2.0) - np.asarray(coordinates.b.rad).reshape(
                dec_mesh.shape
            )
        elif healpix.coordinate_frame == "icrs":
            phi = np.deg2rad(ra_mesh % 360.0)
            theta = np.deg2rad(90.0 - dec_mesh)
        else:
            raise InvalidObservabilityContextError(
                "Prepared HEALPix sky must use ICRS or Galactic coordinates."
            )
        hp: Any = import_module("healpy")

        return np.asarray(
            hp.get_interp_val(
                sky_map,
                theta,
                phi,
                nest=healpix.is_nested,
            ),
            dtype=np.float64,
        )

    def _point_flux(self, point: PointSourceData) -> np.ndarray:
        context = self._context
        spectrum = point.spectrum
        if spectrum is not None:
            matches = np.flatnonzero(
                np.asarray(spectrum.frequencies) == context.frequency_hz
            )
            if len(matches) != 1:
                raise InvalidObservabilityContextError(
                    "Prepared point-source spectrum does not contain the exact "
                    "selected channel."
                )
            return np.asarray(spectrum.flux[int(matches[0])], dtype=np.float64)

        from radiosim.core.sky.containers.spectral import evaluate_point_flux_at_freq

        rotation_measure = (
            point.polarization.rotation_measure
            if point.polarization is not None
            else None
        )
        flux, _q, _u, _v = evaluate_point_flux_at_freq(
            point.flux,
            point.stokes_q,
            point.stokes_u,
            point.stokes_v,
            point.spectral_index,
            point.spectral_coeffs,
            point.ref_freq,
            rotation_measure,
            None,
            None,
            None,
            None,
            None,
            context.frequency_hz,
        )
        return np.asarray(flux, dtype=np.float64)

    def _source_visibility(
        self,
        track: _TrackSamples,
        ra_deg: np.ndarray,
        dec_deg: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        visible = np.empty((len(track.ra_deg), len(ra_deg)), dtype=np.bool_)
        separation = np.empty((len(track.ra_deg), len(ra_deg)), dtype=np.float64)
        for index, (zenith_ra, time_mjd) in enumerate(
            zip(track.ra_deg, track.mjd, strict=True)
        ):
            separation[index] = angular_separation_deg(
                ra_deg,
                dec_deg,
                float(zenith_ra),
                self._context.location.latitude_deg,
            )
            visible[index] = self._membership(
                ra_deg,
                dec_deg,
                zenith_ra_deg=float(zenith_ra),
                time_mjd=float(time_mjd),
            )
        return visible, separation

    def _build_source_metrics(
        self,
        track: _TrackSamples,
        snapshots: _TrackSamples,
    ) -> tuple[ObservabilitySourceMetrics | None, np.ndarray | None]:
        context = self._context
        if not context.options.include_source_metrics:
            return None, None
        sky = cast("SkyModel", context.sky_model)
        point = cast("PointSourceData", sky.point)
        flux = self._point_flux(point)
        order = np.argsort(-flux, kind="stable")[: context.options.max_point_sources]
        ra_deg = np.degrees(np.asarray(point.ra_rad)[order])
        dec_deg = np.degrees(np.asarray(point.dec_rad)[order])
        flux_jy = flux[order]
        names = (
            point.metadata.source_name
            if point.metadata is not None and point.metadata.source_name is not None
            else None
        )
        source_name = (
            None if names is None else np.asarray(names)[order].astype(str, copy=True)
        )
        visible, separation = self._source_visibility(track, ra_deg, dec_deg)
        snapshot_visible, _snapshot_separation = self._source_visibility(
            snapshots,
            ra_deg,
            dec_deg,
        )
        visible_any = visible.any(axis=0)
        visible_fraction = (
            visible.mean(axis=0)
            if visible.shape[0] > 0
            else np.zeros(len(ra_deg), dtype=np.float64)
        )
        min_separation = (
            separation.min(axis=0)
            if separation.shape[0] > 0
            else np.full(len(ra_deg), np.inf)
        )
        first = np.full(len(ra_deg), -1, dtype=np.int64)
        last = np.full(len(ra_deg), -1, dtype=np.int64)
        visible_columns = np.flatnonzero(visible_any)
        if len(visible_columns):
            first[visible_columns] = np.argmax(
                visible[:, visible_columns],
                axis=0,
            )
            last[visible_columns] = (
                len(track.labels)
                - 1
                - np.argmax(visible[:, visible_columns][::-1], axis=0)
            )
        visible_indices = np.flatnonzero(visible_any)
        top = visible_indices[np.argsort(-flux_jy[visible_indices], kind="stable")][
            : context.options.top_n_sources
        ]
        if context.options.footprint_model == "manual_circular":
            nearby_limit = (
                cast(float, context.options.field_radius_deg)
                + context.options.nearby_buffer_deg
            )
        else:
            nearby_limit = context.options.nearby_buffer_deg
        nearby_candidates = np.flatnonzero(
            (~visible_any) & (min_separation <= nearby_limit)
        )
        nearby = nearby_candidates[
            np.argsort(-flux_jy[nearby_candidates], kind="stable")
        ][: context.options.nearby_source_count]
        return (
            ObservabilitySourceMetrics(
                ra_deg=np.asarray(ra_deg, dtype=np.float64),
                dec_deg=np.asarray(dec_deg, dtype=np.float64),
                flux_jy=np.asarray(flux_jy, dtype=np.float64),
                x_coord=np.asarray(
                    axis_from_ra_deg(ra_deg, context.options.x_axis),
                    dtype=np.float64,
                ),
                source_name=source_name,
                visible_any=np.asarray(visible_any, dtype=np.bool_),
                visible_fraction=np.asarray(visible_fraction, dtype=np.float64),
                min_separation_deg=np.asarray(min_separation, dtype=np.float64),
                first_visible_index=first,
                last_visible_index=last,
                top_visible_indices=np.asarray(top, dtype=np.int64),
                nearby_indices=np.asarray(nearby, dtype=np.int64),
            ),
            snapshot_visible,
        )

    def _build_snapshots(
        self,
        track: _TrackSamples,
        source_visibility: np.ndarray | None,
        ra_grid_deg: np.ndarray,
        dec_grid_deg: np.ndarray,
    ) -> tuple[ObservabilitySnapshot, ...]:
        ra_mesh, dec_mesh = np.meshgrid(ra_grid_deg, dec_grid_deg)
        snapshots: list[ObservabilitySnapshot] = []
        for index, (label, utc_iso, lst_hours, zenith_ra, time_mjd) in enumerate(
            zip(
                track.labels,
                track.time_isos,
                track.lst_hours,
                track.ra_deg,
                track.mjd,
                strict=True,
            )
        ):
            snapshots.append(
                ObservabilitySnapshot(
                    label=label,
                    utc_iso=utc_iso,
                    lst_hours=float(lst_hours),
                    zenith_ra_deg=float(zenith_ra),
                    zenith_dec_deg=float(self._context.location.latitude_deg),
                    footprint_mask=self._membership(
                        ra_mesh,
                        dec_mesh,
                        zenith_ra_deg=float(zenith_ra),
                        time_mjd=float(time_mjd),
                    ),
                    visible_source_mask=(
                        None
                        if source_visibility is None
                        else np.asarray(source_visibility[index], dtype=np.bool_)
                    ),
                )
            )
        return tuple(snapshots)

    def _build_beam_projection(
        self,
        ra_grid_deg: np.ndarray,
        dec_grid_deg: np.ndarray,
        reference_ra_deg: float,
        reference_mjd: float,
    ) -> tuple[BeamSkyProjection, tuple[BeamContour, ...]]:
        def power_func(za_rad: np.ndarray, az_rad: np.ndarray) -> np.ndarray:
            return self._normalized_power(
                za_rad,
                az_rad,
                time_mjd=reference_mjd,
            )

        projection = compute_beam_power_on_full_sky_grid(
            power_func,
            reference_ra_deg,
            float(self._context.location.latitude_deg),
            ra_grid_deg,
            dec_grid_deg,
            90.0,
        )
        options = self._context.options
        levels = tuple(
            level
            for level in (-3.0, -10.0)
            if options.beam_contour_min_db <= level <= options.beam_contour_max_db
        )
        if not levels:
            levels = (
                0.5 * (options.beam_contour_min_db + options.beam_contour_max_db),
            )
        return projection, extract_contours(projection, list(levels))

    def _build_title(self, track: _TrackSamples, *, fingerprint: str) -> str:
        context = self._context
        prefix = (
            "Sky Visibility Snapshots"
            if context.options.mode == "snapshots"
            else "Sky Visibility"
        )
        if context.options.footprint_model == "manual_circular":
            prefix += " (manual circular display approximation)"
        identity = (
            f"reference number={context.reference_antenna.number}, "
            f"name={context.reference_antenna.name!r}, "
            f"beam={fingerprint[:12]}, "
            f"selection={context.reference_selection_reason}"
        )
        if track.time_isos[0] is not None:
            window = f"{track.time_isos[0]} to {track.time_isos[-1]}"
        else:
            window = f"LST {track.lst_hours[0]:.6f}h to {track.lst_hours[-1]:.6f}h"
        return f"{prefix}: {window}; {identity}"


__all__ = [
    "UTCObservabilityWindow",
    "LSTObservabilityWindow",
    "ObservabilityWindow",
    "ObservabilityOptions",
    "ObservabilityPlanner",
    "ObservabilityPlan",
    "ObservabilitySnapshot",
    "ObservabilitySourceMetrics",
]
