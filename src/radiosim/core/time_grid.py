"""Canonical UTC integration coordinates for visibility simulation."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Final, Protocol, cast

import numpy as np
from typing_extensions import override

from radiosim.core.result import InvalidTimeGridError, TimeGridLimitError

MAX_TIME_SAMPLES: Final = 10_000_000


class _TimeDeltaValue(Protocol):
    def to_value(self, unit: str) -> object: ...


class _TimeValue(Protocol):
    @property
    def utc(self) -> _TimeValue: ...

    @property
    def jd1(self) -> object: ...

    @property
    def jd2(self) -> object: ...

    @property
    def jd(self) -> object: ...

    @property
    def mjd(self) -> object: ...

    @property
    def isot(self) -> object: ...

    def __getitem__(self, key: object) -> _TimeValue: ...

    def __add__(self, other: object) -> _TimeValue: ...

    def __sub__(self, other: object) -> _TimeDeltaValue: ...


def _time_constructors() -> tuple[
    Callable[..., _TimeValue],
    Callable[..., object],
]:
    module = import_module("astropy.time")
    time_constructor = cast(
        Callable[..., _TimeValue],
        module.__dict__["Time"],
    )
    delta_constructor = cast(
        Callable[..., object],
        module.__dict__["TimeDelta"],
    )
    return time_constructor, delta_constructor


def _immutable_float64(value: object) -> np.ndarray:
    array = np.array(value, dtype=np.float64, order="C", copy=True, subok=False)
    return np.ndarray(array.shape, dtype=array.dtype, buffer=array.tobytes(order="C"))


@dataclass(frozen=True, slots=True, init=False, eq=False)
class ObservationTimeGrid:
    """An exact, immutable, half-open UTC sample-center grid."""

    schema_version: str
    interval_semantics: str
    start_time_iso: str
    duration_seconds: float
    cadence_seconds: float
    utc_jd1: np.ndarray
    utc_jd2: np.ndarray
    integration_time_seconds: np.ndarray

    def __init__(self) -> None:
        raise TypeError(
            "ObservationTimeGrid must be built by build_observation_time_grid factory"
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ObservationTimeGrid cannot be subclassed")

    def __len__(self) -> int:
        return int(self.utc_jd1.shape[0])

    @override
    def __eq__(self, other: object) -> bool:
        if type(other) is not ObservationTimeGrid:
            return False
        other_grid = other
        return bool(
            self.schema_version == other_grid.schema_version
            and self.interval_semantics == other_grid.interval_semantics
            and self.start_time_iso == other_grid.start_time_iso
            and self.duration_seconds == other_grid.duration_seconds
            and self.cadence_seconds == other_grid.cadence_seconds
            and np.array_equal(self.utc_jd1, other_grid.utc_jd1)
            and np.array_equal(self.utc_jd2, other_grid.utc_jd2)
            and np.array_equal(
                self.integration_time_seconds,
                other_grid.integration_time_seconds,
            )
        )

    @override
    def __hash__(self) -> int:
        raise TypeError("ObservationTimeGrid is unhashable")

    def as_astropy(self) -> _TimeValue:
        """Return a newly owned Astropy UTC coordinate."""
        time_constructor, _ = _time_constructors()
        return time_constructor(
            np.array(self.utc_jd1, copy=True),
            np.array(self.utc_jd2, copy=True),
            format="jd",
            scale="utc",
        )

    def to_jd(self) -> np.ndarray:
        """Return newly owned one-part UTC Julian dates."""
        return np.array(self.as_astropy().jd, dtype=np.float64, copy=True)

    def to_mjd(self) -> np.ndarray:
        """Return newly owned UTC modified Julian dates."""
        return np.array(self.as_astropy().mjd, dtype=np.float64, copy=True)


def build_mmode_observation_time_grid(
    *,
    start_time_iso: str,
    utc_jd1: np.ndarray,
    utc_jd2: np.ndarray,
    integration_time_seconds: np.ndarray,
    duration_seconds: float,
    cadence_seconds: float,
) -> ObservationTimeGrid:
    """Publish an m-mode full-sidereal cycle as the canonical UTC sample grid.

    ``docs/development/sci004_mmode_design.md`` Section 3.1 makes the exact turn
    coordinate authoritative and UTC an *output and provenance* coordinate: the
    centres, exposure boundaries and horizon cuts are all mapped from exact
    turns, and the resulting values are then converted to two-part UTC for the
    existing results and writers.  This factory publishes exactly those already
    mapped values; it does not re-derive a cadence, and it never regenerates a
    turn from ``k``, a width or an adjacent edge.

    Parameters
    ----------
    start_time_iso : str
        The canonical UTC anchor spelling.
    utc_jd1, utc_jd2 : ndarray
        The two-part UTC sample centres mapped from the retained exact turns.
    integration_time_seconds : ndarray
        Each sample's SI-second width, derived from its exact exposure edges.
    duration_seconds, cadence_seconds : float
        The retained cycle span and mean sample separation, recorded for the
        existing provenance surface only.
    """
    if type(start_time_iso) is not str or not start_time_iso.strip():
        raise InvalidTimeGridError("start_time must be a nonblank string")
    first = np.asarray(utc_jd1, dtype=np.float64)
    second = np.asarray(utc_jd2, dtype=np.float64)
    widths = np.asarray(integration_time_seconds, dtype=np.float64)
    count = int(first.shape[0])
    if (
        first.ndim != 1
        or second.shape != first.shape
        or widths.shape != first.shape
        or count == 0
    ):
        raise InvalidTimeGridError("m-mode UTC coordinates are invalid")
    if count > MAX_TIME_SAMPLES:
        raise TimeGridLimitError(requested_count=count, limit=MAX_TIME_SAMPLES)
    if (
        not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
        or not np.all(np.isfinite(widths))
        or not np.all(widths > 0.0)
    ):
        raise InvalidTimeGridError("m-mode UTC coordinates are invalid")
    duration = _positive_finite(duration_seconds, field_name="duration_seconds")
    cadence = _positive_finite(cadence_seconds, field_name="cadence_seconds")

    result = object.__new__(ObservationTimeGrid)
    object.__setattr__(result, "schema_version", "radiosim.time-grid.v1")
    object.__setattr__(result, "interval_semantics", "half_open_sample_centers")
    object.__setattr__(result, "start_time_iso", start_time_iso.strip())
    object.__setattr__(result, "duration_seconds", duration)
    object.__setattr__(result, "cadence_seconds", cadence)
    object.__setattr__(result, "utc_jd1", _immutable_float64(first))
    object.__setattr__(result, "utc_jd2", _immutable_float64(second))
    object.__setattr__(result, "integration_time_seconds", _immutable_float64(widths))
    return result


def _positive_finite(value: object, *, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise InvalidTimeGridError(f"{field_name} must be a positive finite number")
    try:
        normalized = float(cast(float | int | str, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidTimeGridError(
            f"{field_name} must be a positive finite number"
        ) from exc
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise InvalidTimeGridError(f"{field_name} must be a positive finite number")
    return normalized


def build_observation_time_grid(
    *,
    start_time: str,
    duration_seconds: float,
    cadence_seconds: float,
) -> ObservationTimeGrid:
    """Build the canonical half-open observation sample grid."""
    if type(start_time) is not str or not start_time.strip():
        raise InvalidTimeGridError("start_time must be a nonblank string")
    duration = _positive_finite(duration_seconds, field_name="duration_seconds")
    cadence = _positive_finite(cadence_seconds, field_name="cadence_seconds")
    if cadence > duration:
        raise InvalidTimeGridError("cadence_seconds must not exceed duration_seconds")

    quotient = duration / cadence
    if not math.isfinite(quotient):
        raise InvalidTimeGridError("time sample count is not finite")
    nearest = round(quotient)
    tolerance = 32.0 * np.finfo(np.float64).eps * max(1.0, abs(quotient))
    normalized_quotient = (
        float(nearest) if abs(quotient - nearest) <= tolerance else quotient
    )
    count = math.ceil(normalized_quotient)
    if count > MAX_TIME_SAMPLES:
        raise TimeGridLimitError(
            requested_count=count,
            limit=MAX_TIME_SAMPLES,
        )

    try:
        time_constructor, delta_constructor = _time_constructors()
        start = time_constructor(start_time.strip(), scale="utc")
        offsets = np.arange(count, dtype=np.float64) * cadence
        centers = start + delta_constructor(offsets, format="sec")
        centers = centers.utc
        jd1 = np.asarray(centers.jd1, dtype=np.float64)
        jd2 = np.asarray(centers.jd2, dtype=np.float64)
        integrations = np.full(count, cadence, dtype=np.float64)
        if (
            jd1.shape != (count,)
            or jd2.shape != (count,)
            or not np.all(np.isfinite(jd1))
            or not np.all(np.isfinite(jd2))
        ):
            raise ValueError("generated UTC coordinates are invalid")
        if count > 1:
            steps = np.asarray((centers[1:] - centers[:-1]).to_value("s"))
            if not np.all(np.isfinite(steps)) or not np.all(steps > 0.0):
                raise ValueError("generated UTC coordinates are not monotonic")
        start_iso = str(start.utc.isot)
    except InvalidTimeGridError:
        raise
    except Exception as exc:
        raise InvalidTimeGridError("could not construct the UTC time grid") from exc

    result = object.__new__(ObservationTimeGrid)
    object.__setattr__(result, "schema_version", "radiosim.time-grid.v1")
    object.__setattr__(
        result,
        "interval_semantics",
        "half_open_sample_centers",
    )
    object.__setattr__(result, "start_time_iso", start_iso)
    object.__setattr__(result, "duration_seconds", duration)
    object.__setattr__(result, "cadence_seconds", cadence)
    object.__setattr__(result, "utc_jd1", _immutable_float64(jd1))
    object.__setattr__(result, "utc_jd2", _immutable_float64(jd2))
    object.__setattr__(
        result,
        "integration_time_seconds",
        _immutable_float64(integrations),
    )
    return result
