"""Immutable projected standard-visibility models and shared projection."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from types import MappingProxyType
from typing import Any, Final, Literal, cast

import numpy as np
from typing_extensions import override

from radiosim.core.phase_center import PhaseCenter
from radiosim.core.polarization_basis import (
    AIPS_CODES_CANONICAL,
    AIPS_CODES_FILE_ORDER,
    CORRELATION_LABELS,
    POLARIZATION_BASES,
    PYUVDATA_FEEDS,
    PYUVDATA_POLARIZATIONS,
    PolarizationBasis,
)
from radiosim.core.polarization_basis import (
    parallel_hand_indices as _parallel_hand_indices,
)
from radiosim.core.result import SimulationResult
from radiosim.core.runtime_config import FrozenMapping, json_safe_mapping
from radiosim.io.result_errors import (
    FormatRepresentationError,
    UnsafeResultInputError,
    UnsupportedPolarizationBasisError,
)

STANDARD_SCHEMA: Final = "radiosim.standard-visibility.v1"
PROJECTED_PHASE_SCHEMA: Final = "radiosim.projected-phase-center.v1"
PROJECTION_TRANSFORMATION: Final = "astropy-zenith-icrs+pyuvdata-phase_to_time.v1"
PROJECTION_HISTORY_PREFIX: Final = "RADIOSIM_PROJECTION_JSON="

# The nominal pyuvdata feed orientation for each output basis, written uniformly
# for every antenna (Section 14.4).  These reproduce
# ``radiosim.core.receptor``'s zero-rotation ``feed_angle_rad``: an unrotated
# linear pair is ``(pi/2, 0)`` -- identical to what pyuvdata derives from the
# retired east x-orientation shorthand -- and an unrotated circular pair is
# ``(0, 0)``.  The per-antenna native basis and feed rotation are RadioSim
# provenance, never inferred by a reader from ``feed_array``.
_NOMINAL_FEED_ANGLES_RAD: Final[Mapping[PolarizationBasis, tuple[float, float]]] = (
    MappingProxyType(
        {
            "linear_xy": (math.pi / 2.0, 0.0),
            "circular_rl": (0.0, 0.0),
        }
    )
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PROJECTION_HISTORY_LIMIT = 16_000
_MAX_PROJECTION_JSON_DEPTH = 64
_PYUVDATA_HISTORY_TRAILING = re.compile(
    r"Read/written with pyuvdata version: [0-9A-Za-z][0-9A-Za-z.+-]*\.\Z"
)
_PROJECTION_RECORD_FIELDS = {
    "schema",
    "projected_phase",
    "source_scientific_sha256",
    "source_provenance_sha256",
    "input_visibility_dtype",
    "stored_visibility_dtype",
    "input_weight_dtype",
    "stored_weight_dtype",
    "polarization_basis",
    "receptor_sha256",
    "instrument",
    "beam",
    "solver",
}


def _accepted_correlation_sets() -> str:
    """Return the shared rejection text naming both accepted label tuples."""
    return " or ".join(
        ",".join(CORRELATION_LABELS[basis]) for basis in POLARIZATION_BASES
    )


def require_polarization_basis(correlations: object) -> PolarizationBasis:
    """Return the output basis named by one accepted correlation label tuple.

    Standard formats accept exactly the two Section 14.2 correlation coordinate
    sets, each only in its canonical row-major order.  A reordering is rejected:
    the correlation axis order is part of the contract.

    Parameters
    ----------
    correlations
        A sequence of correlation labels.

    Returns
    -------
    PolarizationBasis
        ``"linear_xy"`` or ``"circular_rl"``.

    Raises
    ------
    UnsupportedPolarizationBasisError
        The labels are not exactly one accepted tuple in its canonical order.
    """
    if isinstance(correlations, (str, bytes)) or not isinstance(correlations, Sequence):
        raise UnsupportedPolarizationBasisError(
            "correlations must be a sequence of correlation labels; expected "
            f"exactly {_accepted_correlation_sets()} in that order"
        )
    labels = tuple(cast(Sequence[object], correlations))
    for basis in POLARIZATION_BASES:
        if labels == CORRELATION_LABELS[basis]:
            return basis
    raise UnsupportedPolarizationBasisError(
        f"correlations={labels!r} is not an accepted correlation coordinate set; "
        f"expected exactly {_accepted_correlation_sets()} in that order"
    )


def basis_for_file_codes(codes: object) -> PolarizationBasis:
    """Return the output basis carried by one on-disk AIPS polarization axis.

    Both accepted code sets are matched as sets, because a Measurement Set
    stores the in-memory order in ``CORR_TYPE`` while UVFITS stores the
    descending Section 14.2 order (Tier 5A, Q3).

    Parameters
    ----------
    codes
        The AIPS polarization codes read from a file.

    Returns
    -------
    PolarizationBasis
        The basis whose code set the axis matches exactly.

    Raises
    ------
    FormatRepresentationError
        The axis is not four codes drawn from exactly one accepted set.
    """
    values = np.asarray(codes, dtype=np.int64)
    if values.shape == (4,):
        observed = set(values.tolist())
        for basis in POLARIZATION_BASES:
            if observed == set(AIPS_CODES_FILE_ORDER[basis]):
                return basis
    raise FormatRepresentationError(
        "standard input has an unsupported polarization layout"
    )


def require_feed_polarization_coupling(
    uvdata: Any,
    basis: PolarizationBasis,
) -> None:
    """Require the receptor feeds and the polarization axis to name one basis.

    pyuvdata 3.2.1 does not cross-validate ``Telescope.feed_array`` against
    ``UVData.polarization_array`` (Tier 5A, Q3), so RadioSim enforces the
    coupling itself: a reader that trusted mismatched metadata would
    misinterpret every visibility.  Inputs that carry no feed metadata at all
    are left to the format-specific readers.

    Parameters
    ----------
    uvdata
        A ``UVData``-shaped object, or any metadata view exposing ``telescope``.
    basis
        The basis already established from the polarization axis.

    Raises
    ------
    FormatRepresentationError
        The declared feeds do not belong to ``basis``.
    """
    telescope = getattr(uvdata, "telescope", None)
    feed_array = getattr(telescope, "feed_array", None)
    if feed_array is None:
        return
    observed = {str(feed).lower() for feed in np.asarray(feed_array).reshape(-1)}
    if observed != set(PYUVDATA_FEEDS[basis]):
        raise FormatRepresentationError(
            "standard input receptor feeds "
            f"{sorted(observed)!r} disagree with its "
            f"{','.join(CORRELATION_LABELS[basis])} polarization axis"
        )


def _exact_text(value: object, *, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be an exact built-in string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be nonblank")
    try:
        _ = normalized.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field_name} must be strict UTF-8") from exc
    if "\x00" in normalized:
        raise ValueError(f"{field_name} must not contain NUL")
    return normalized


def _exact_finite_float(value: object, *, field_name: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{field_name} must be an exact built-in float")
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


def _json_tree(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _json_tree(item)
            for key, item in cast(Mapping[object, object], value).items()
        }
    if isinstance(value, (tuple, list)):
        return [_json_tree(item) for item in cast(Sequence[object], value)]
    if isinstance(value, np.generic):
        return cast(object, value.item())
    return value


def _phase_snapshot(value: object) -> FrozenMapping:
    if type(value) is not dict:
        raise TypeError("original_phase_snapshot must be an exact built-in dict")
    snapshot = cast(dict[str, object], value)
    try:
        phase = PhaseCenter(**cast(dict[str, Any], dict(snapshot)))
    except Exception as exc:
        raise ValueError(
            "original_phase_snapshot must contain one canonical PhaseCenter"
        ) from exc
    return phase.to_snapshot()


@dataclass(frozen=True, slots=True)
class ProjectedPhaseCenter:
    """Fixed ICRS phase reference derived from a canonical zenith drift."""

    longitude_rad: float
    latitude_rad: float
    reference_utc_jd1: float
    reference_utc_jd2: float
    original_phase_snapshot: FrozenMapping | dict[str, object]
    transformation: str
    schema_version: Literal["radiosim.projected-phase-center.v1"] = (
        PROJECTED_PHASE_SCHEMA
    )
    kind: Literal["sidereal"] = "sidereal"
    frame: Literal["icrs"] = "icrs"

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ProjectedPhaseCenter cannot be subclassed")

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != PROJECTED_PHASE_SCHEMA
            or type(self.kind) is not str
            or self.kind != "sidereal"
            or type(self.frame) is not str
            or self.frame != "icrs"
        ):
            raise ValueError("projected phase identity is invalid")
        longitude = _exact_finite_float(
            self.longitude_rad,
            field_name="longitude_rad",
        )
        latitude = _exact_finite_float(
            self.latitude_rad,
            field_name="latitude_rad",
        )
        reference_jd1 = _exact_finite_float(
            self.reference_utc_jd1,
            field_name="reference_utc_jd1",
        )
        reference_jd2 = _exact_finite_float(
            self.reference_utc_jd2,
            field_name="reference_utc_jd2",
        )
        if not 0.0 <= longitude < 2.0 * math.pi:
            raise ValueError("longitude_rad must be in [0, 2*pi)")
        if not -math.pi / 2.0 <= latitude <= math.pi / 2.0:
            raise ValueError("latitude_rad must be in [-pi/2, pi/2]")
        snapshot_value = self.original_phase_snapshot
        if isinstance(snapshot_value, FrozenMapping):
            snapshot_value = dict(snapshot_value)
        snapshot = _phase_snapshot(snapshot_value)
        transformation = _exact_text(
            self.transformation,
            field_name="transformation",
        )
        if transformation != PROJECTION_TRANSFORMATION:
            raise ValueError("transformation has an unsupported identity")
        if (
            reference_jd1 != float(round(reference_jd1))
            or abs(reference_jd2) > 0.5
            or reference_jd1 + reference_jd2 <= 0.0
        ):
            raise ValueError(
                "reference UTC JD must use a coherent canonical two-part value"
            )
        object.__setattr__(self, "longitude_rad", longitude)
        object.__setattr__(self, "latitude_rad", latitude)
        object.__setattr__(self, "reference_utc_jd1", reference_jd1)
        object.__setattr__(self, "reference_utc_jd2", reference_jd2)
        object.__setattr__(self, "original_phase_snapshot", snapshot)
        object.__setattr__(self, "transformation", transformation)

    def to_snapshot(self) -> FrozenMapping:
        """Return a detached immutable JSON-safe projection record."""
        return json_safe_mapping(
            {
                "schema_version": self.schema_version,
                "kind": self.kind,
                "frame": self.frame,
                "longitude_rad": self.longitude_rad,
                "latitude_rad": self.latitude_rad,
                "reference_utc_jd1": self.reference_utc_jd1,
                "reference_utc_jd2": self.reference_utc_jd2,
                "original_phase_snapshot": self.original_phase_snapshot,
                "transformation": self.transformation,
            }
        )


@dataclass(frozen=True, slots=True)
class StandardReadLimits:
    """Pre-allocation limits for hostile standard visibility inputs."""

    max_times: int = 10_000_000
    max_baselines: int = 10_000_000
    max_frequencies: int = 1_000_000
    max_antennas: int = 1_000_000
    max_visibility_elements: int = 100_000_000
    max_data_bytes: int = 2_147_483_648

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("StandardReadLimits cannot be subclassed")

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int:
                raise TypeError(f"{field.name} must be an exact built-in integer")
            if value <= 0:
                raise ValueError(f"{field.name} must be positive")

    @override
    def __repr__(self) -> str:
        """Keep public signatures compact while retaining explicit fields."""
        if self == StandardReadLimits():
            return "StandardReadLimits()"
        values = ", ".join(
            f"{field.name}={getattr(self, field.name)!r}" for field in fields(self)
        )
        return f"StandardReadLimits({values})"


def _immutable_array(
    value: object,
    *,
    dtype: np.dtype[Any],
    field_name: str,
) -> np.ndarray:
    try:
        source = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{field_name} must be a numeric array") from exc
    if source.dtype != dtype:
        raise TypeError(f"{field_name} must use exact {dtype.name} dtype")
    if source.dtype.hasobject:
        raise TypeError(f"{field_name} must not use object dtype")
    array = np.array(source, dtype=dtype, order="C", copy=True, subok=False)
    return np.ndarray(array.shape, dtype=dtype, buffer=array.tobytes(order="C"))


def _immutable_visibility(value: object) -> np.ndarray:
    try:
        source = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("visibilities must be a numeric array") from exc
    if source.dtype not in {np.dtype("complex64"), np.dtype("complex128")}:
        raise TypeError("visibilities must use exact complex64 or complex128 dtype")
    array = np.array(
        source,
        dtype=source.dtype,
        order="C",
        copy=True,
        subok=False,
    )
    return np.ndarray(
        array.shape,
        dtype=array.dtype,
        buffer=array.tobytes(order="C"),
    )


def _history_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("history must be a sequence of strings")
    return tuple(
        _exact_text(item, field_name=f"history[{index}]")
        for index, item in enumerate(cast(Sequence[object], value))
    )


def _optional_sha(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lower-case SHA-256 or None")
    return value


def _telescope_snapshot(value: object) -> FrozenMapping:
    if type(value) is not dict:
        raise TypeError("telescope_snapshot must be an exact built-in dict")
    snapshot = cast(dict[str, object], value)
    expected = {
        "name",
        "instrument",
        "location_itrs_xyz_m",
        "antennas",
    }
    if set(snapshot) != expected:
        raise ValueError("telescope_snapshot has unexpected fields")
    name = _exact_text(snapshot["name"], field_name="telescope_snapshot.name")
    instrument = _exact_text(
        snapshot["instrument"],
        field_name="telescope_snapshot.instrument",
    )
    location = snapshot["location_itrs_xyz_m"]
    if not isinstance(location, (tuple, list)):
        raise ValueError("telescope_snapshot location must contain three values")
    location_sequence = cast(Sequence[object], location)
    if len(location_sequence) != 3:
        raise ValueError("telescope_snapshot location must contain three values")
    normalized_location: list[float] = []
    for _index, item in enumerate(location_sequence):
        if type(item) is not float or not math.isfinite(item):
            raise TypeError(
                "telescope_snapshot location values must be exact finite floats"
            )
        normalized_location.append(item)
    antennas_value = snapshot["antennas"]
    if not isinstance(antennas_value, list) or not antennas_value:
        raise ValueError("telescope_snapshot antennas must be a nonempty list")
    normalized_antennas: list[dict[str, object]] = []
    numbers: set[int] = set()
    names: set[str] = set()
    for index, item in enumerate(cast(list[object], antennas_value)):
        if type(item) is not dict:
            raise TypeError(f"telescope_snapshot.antennas[{index}] must be a dict")
        antenna = cast(dict[str, object], item)
        if set(antenna) != {
            "number",
            "name",
            "position_enu_m",
            "diameter_m",
        }:
            raise ValueError("telescope_snapshot antenna has unexpected fields")
        number = antenna["number"]
        if type(number) is not int:
            raise TypeError("telescope_snapshot antenna number must be an exact int")
        antenna_name = _exact_text(
            antenna["name"],
            field_name=f"telescope_snapshot.antennas[{index}].name",
        )
        position = antenna["position_enu_m"]
        if not isinstance(position, (tuple, list)):
            raise ValueError("telescope_snapshot antenna position must have length 3")
        position_sequence = cast(Sequence[object], position)
        if len(position_sequence) != 3:
            raise ValueError("telescope_snapshot antenna position must have length 3")
        normalized_position: list[float] = []
        for coordinate in position_sequence:
            if type(coordinate) is not float or not math.isfinite(coordinate):
                raise TypeError(
                    "telescope_snapshot antenna positions must be exact finite floats"
                )
            normalized_position.append(coordinate)
        diameter = antenna["diameter_m"]
        if (
            type(diameter) is not float
            or not math.isfinite(diameter)
            or diameter <= 0.0
        ):
            raise TypeError(
                "telescope_snapshot antenna diameters must be exact positive floats"
            )
        if number in numbers or antenna_name in names:
            raise ValueError("telescope_snapshot antenna identity must be unique")
        numbers.add(number)
        names.add(antenna_name)
        normalized_antennas.append(
            {
                "number": number,
                "name": antenna_name,
                "position_enu_m": normalized_position,
                "diameter_m": diameter,
            }
        )
    return json_safe_mapping(
        {
            "name": name,
            "instrument": instrument,
            "location_itrs_xyz_m": normalized_location,
            "antennas": normalized_antennas,
        }
    )


@dataclass(frozen=True, slots=True, init=False, eq=False)
class StandardVisibilityData:
    """Immutable canonical view returned by MS and UVFITS readers."""

    schema_version: str
    format: str
    visibilities: np.ndarray
    flags: np.ndarray
    weights: np.ndarray
    utc_jd1: np.ndarray
    utc_jd2: np.ndarray
    exposure_seconds: np.ndarray
    frequencies_hz: np.ndarray
    channel_widths_hz: np.ndarray
    correlations: tuple[str, ...]
    antenna1_numbers: np.ndarray
    antenna2_numbers: np.ndarray
    uvw_m: np.ndarray
    telescope_snapshot: FrozenMapping
    phase_center: ProjectedPhaseCenter
    history: tuple[str, ...]
    source_scientific_sha256: str | None
    source_provenance_sha256: str | None

    def __init__(self) -> None:
        raise TypeError(
            "StandardVisibilityData must be built by build_standard_visibility_data"
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("StandardVisibilityData cannot be subclassed")

    @override
    def __eq__(self, other: object) -> bool:
        if type(other) is not StandardVisibilityData:
            return False
        typed = other
        return bool(
            self.schema_version == typed.schema_version
            and self.format == typed.format
            and self.correlations == typed.correlations
            and self.telescope_snapshot == typed.telescope_snapshot
            and self.phase_center == typed.phase_center
            and self.history == typed.history
            and self.source_scientific_sha256 == typed.source_scientific_sha256
            and self.source_provenance_sha256 == typed.source_provenance_sha256
            and all(
                np.array_equal(left, right)
                for left, right in (
                    (self.visibilities, typed.visibilities),
                    (self.flags, typed.flags),
                    (self.weights, typed.weights),
                    (self.utc_jd1, typed.utc_jd1),
                    (self.utc_jd2, typed.utc_jd2),
                    (self.exposure_seconds, typed.exposure_seconds),
                    (self.frequencies_hz, typed.frequencies_hz),
                    (self.channel_widths_hz, typed.channel_widths_hz),
                    (self.antenna1_numbers, typed.antenna1_numbers),
                    (self.antenna2_numbers, typed.antenna2_numbers),
                    (self.uvw_m, typed.uvw_m),
                )
            )
        )

    @override
    def __hash__(self) -> int:
        raise TypeError("StandardVisibilityData is unhashable")


def build_standard_visibility_data(
    *,
    format: object,
    visibilities: object,
    flags: object,
    weights: object,
    utc_jd1: object,
    utc_jd2: object,
    exposure_seconds: object,
    frequencies_hz: object,
    channel_widths_hz: object,
    correlations: object,
    antenna1_numbers: object,
    antenna2_numbers: object,
    uvw_m: object,
    telescope_snapshot: object,
    phase_center: object,
    history: object,
    source_scientific_sha256: object = None,
    source_provenance_sha256: object = None,
) -> StandardVisibilityData:
    """Validate and copy-harden one projected standard-format view."""
    if type(format) is not str or format not in {"ms", "uvfits"}:
        raise ValueError("format must be exactly 'ms' or 'uvfits'")
    format_value = cast(Literal["ms", "uvfits"], format)
    if type(phase_center) is not ProjectedPhaseCenter:
        raise TypeError("phase_center must be an exact ProjectedPhaseCenter")
    phase_value = phase_center
    correlation_labels = CORRELATION_LABELS[require_polarization_basis(correlations)]

    visibility_array = _immutable_visibility(visibilities)
    flag_array = _immutable_array(
        flags,
        dtype=np.dtype("bool"),
        field_name="flags",
    )
    weight_array = _immutable_array(
        weights,
        dtype=np.dtype("float32"),
        field_name="weights",
    )
    jd1_array = _immutable_array(
        utc_jd1,
        dtype=np.dtype("float64"),
        field_name="utc_jd1",
    )
    jd2_array = _immutable_array(
        utc_jd2,
        dtype=np.dtype("float64"),
        field_name="utc_jd2",
    )
    exposure_array = _immutable_array(
        exposure_seconds,
        dtype=np.dtype("float64"),
        field_name="exposure_seconds",
    )
    frequency_array = _immutable_array(
        frequencies_hz,
        dtype=np.dtype("float64"),
        field_name="frequencies_hz",
    )
    width_array = _immutable_array(
        channel_widths_hz,
        dtype=np.dtype("float64"),
        field_name="channel_widths_hz",
    )
    antenna1_array = _immutable_array(
        antenna1_numbers,
        dtype=np.dtype("int64"),
        field_name="antenna1_numbers",
    )
    antenna2_array = _immutable_array(
        antenna2_numbers,
        dtype=np.dtype("int64"),
        field_name="antenna2_numbers",
    )
    uvw_array = _immutable_array(
        uvw_m,
        dtype=np.dtype("float64"),
        field_name="uvw_m",
    )
    snapshot = _telescope_snapshot(telescope_snapshot)
    frozen_history = _history_tuple(history)
    scientific = _optional_sha(
        source_scientific_sha256,
        field_name="source_scientific_sha256",
    )
    provenance = _optional_sha(
        source_provenance_sha256,
        field_name="source_provenance_sha256",
    )

    if visibility_array.ndim != 4 or visibility_array.shape[-1] != 4:
        raise ValueError("visibilities must have nonempty shape (T,B,F,4)")
    time_count, baseline_count, frequency_count, _ = visibility_array.shape
    if min(time_count, baseline_count, frequency_count) <= 0:
        raise ValueError("standard visibility axes must be nonempty")
    if flag_array.shape != visibility_array.shape:
        raise ValueError("flags shape must match visibility shape")
    if weight_array.shape != visibility_array.shape:
        raise ValueError("weights shape must match visibility shape")
    if (
        jd1_array.shape != (time_count,)
        or jd2_array.shape != (time_count,)
        or exposure_array.shape != (time_count,)
    ):
        raise ValueError("time coordinate shapes must match the visibility time axis")
    if frequency_array.shape != (frequency_count,) or width_array.shape != (
        frequency_count,
    ):
        raise ValueError(
            "frequency coordinate shapes must match the visibility frequency axis"
        )
    if (
        antenna1_array.shape != (baseline_count,)
        or antenna2_array.shape != (baseline_count,)
        or uvw_array.shape != (time_count, baseline_count, 3)
    ):
        raise ValueError(
            "baseline identity and UVW shapes must match the visibility axes"
        )
    for name, array in (
        ("visibilities", visibility_array),
        ("weights", weight_array),
        ("utc_jd1", jd1_array),
        ("utc_jd2", jd2_array),
        ("exposure_seconds", exposure_array),
        ("frequencies_hz", frequency_array),
        ("channel_widths_hz", width_array),
        ("uvw_m", uvw_array),
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
    if (
        not np.all(exposure_array > 0.0)
        or not np.all(frequency_array > 0.0)
        or not np.all(width_array > 0.0)
    ):
        raise ValueError("frequencies, channel widths, and exposures must be positive")
    if time_count > 1 and not np.all(np.diff(jd1_array + jd2_array) > 0.0):
        raise ValueError("time coordinates must be strictly increasing")
    if frequency_count > 1 and not np.all(np.diff(frequency_array) > 0.0):
        raise ValueError("frequency centers must be strictly increasing")
    pairs = list(zip(antenna1_array.tolist(), antenna2_array.tolist(), strict=True))
    if len(set(pairs)) != baseline_count:
        raise ValueError("baseline identity must be unique and rectangular")
    antenna_numbers = {
        cast(int, antenna["number"])
        for antenna in cast(tuple[FrozenMapping, ...], snapshot["antennas"])
    }
    if any(
        first not in antenna_numbers or second not in antenna_numbers
        for first, second in pairs
    ):
        raise ValueError("baseline identity references an unknown antenna")
    for baseline_index, (first, second) in enumerate(pairs):
        if first == second and not np.array_equal(
            uvw_array[:, baseline_index],
            np.zeros((time_count, 3), dtype=np.float64),
        ):
            raise ValueError("autocorrelation UVW coordinates must be exact zero")

    result = object.__new__(StandardVisibilityData)
    for name, value in {
        "schema_version": STANDARD_SCHEMA,
        "format": format_value,
        "visibilities": visibility_array,
        "flags": flag_array,
        "weights": weight_array,
        "utc_jd1": jd1_array,
        "utc_jd2": jd2_array,
        "exposure_seconds": exposure_array,
        "frequencies_hz": frequency_array,
        "channel_widths_hz": width_array,
        "correlations": correlation_labels,
        "antenna1_numbers": antenna1_array,
        "antenna2_numbers": antenna2_array,
        "uvw_m": uvw_array,
        "telescope_snapshot": snapshot,
        "phase_center": phase_value,
        "history": frozen_history,
        "source_scientific_sha256": scientific,
        "source_provenance_sha256": provenance,
    }.items():
        object.__setattr__(result, name, value)
    return result


@dataclass(frozen=True, slots=True)
class _ProjectedVisibility:
    uvdata: Any
    data: StandardVisibilityData


def _package_version() -> str:
    try:
        return version("radiosim")
    except PackageNotFoundError:
        return "unknown"


def _canonical_telescope_snapshot(result: SimulationResult) -> dict[str, object]:
    return {
        "name": result.instrument.name,
        "instrument": result.instrument.name,
        "location_itrs_xyz_m": [
            float(value) for value in result.instrument.location.itrs_xyz_m
        ],
        "antennas": [
            {
                "number": antenna.id.number,
                "name": antenna.id.name,
                "position_enu_m": [float(value) for value in antenna.position_enu_m],
                "diameter_m": float(antenna.diameter_m),
            }
            for antenna in result.instrument.antennas
        ],
    }


def validate_projection_result(
    result: object,
    *,
    format_name: str,
) -> SimulationResult:
    if type(result) is not SimulationResult:
        raise TypeError("result must be an exact SimulationResult")
    if type(format_name) is not str or format_name not in {"ms", "uvfits"}:
        raise ValueError("format must be exactly 'ms' or 'uvfits'")
    typed = result
    if typed.schema_version != "radiosim.result.v1":
        raise FormatRepresentationError("unsupported canonical result schema")
    _ = require_polarization_basis(typed.correlations)
    if typed.visibilities.dtype not in {
        np.dtype("complex64"),
        np.dtype("complex128"),
    }:
        raise FormatRepresentationError(
            f"{format_name} cannot represent {typed.visibilities.dtype.name}"
        )
    if (
        not np.all(np.isfinite(typed.visibilities))
        or not np.all(np.isfinite(typed.weights))
        or not np.all(np.isfinite(typed.frequencies_hz))
        or not np.all(np.isfinite(typed.channel_widths_hz))
    ):
        raise FormatRepresentationError("standard visibility inputs must be finite")
    return typed


def normalize_autocorrelations(
    result: SimulationResult,
) -> tuple[np.ndarray, int]:
    """Force the parallel-hand autocorrelations of either basis onto the reals.

    The parallel hands are ``XX``/``YY`` in a linear basis and ``RR``/``LL`` in a
    circular one, so their indices are derived from the published correlation
    labels rather than assumed to be ``(0, 3)``.
    """
    parallel_hands = _parallel_hand_indices(result.correlations)
    data = np.array(
        result.visibilities,
        dtype=result.visibilities.dtype,
        order="C",
        copy=True,
        subok=False,
    )
    if data.dtype == np.dtype("complex64"):
        epsilon = float(np.finfo(np.float32).eps)
        real_dtype = np.dtype("float32")
    else:
        epsilon = float(np.finfo(np.float64).eps)
        real_dtype = np.dtype("float64")
    normalized = 0
    for baseline_index, baseline in enumerate(result.selection.baselines):
        if baseline.ant1 != baseline.ant2:
            continue
        for correlation_index in parallel_hands:
            values = np.asarray(data[:, baseline_index, :, correlation_index])
            real_values = np.asarray(np.real(values), dtype=real_dtype)
            imaginary_values = np.asarray(np.imag(values), dtype=real_dtype)
            tolerance = (
                64.0
                * epsilon
                * np.maximum(
                    1.0,
                    np.abs(real_values),
                )
            )
            if np.any(np.abs(imaginary_values) > tolerance):
                raise FormatRepresentationError(
                    "parallel-hand autocorrelation imaginary component exceeds "
                    "the representable tolerance"
                )
            mask = imaginary_values != 0.0
            normalized += int(np.count_nonzero(mask))
            data[:, baseline_index, :, correlation_index] = real_values
    return data, normalized


def _projection_history(
    result: SimulationResult,
    phase_center: ProjectedPhaseCenter,
    *,
    format_name: str,
    stored_visibility_dtype: str,
    normalized_autos: int,
) -> tuple[tuple[str, ...], str]:
    input_dtype = result.visibilities.dtype.name
    lossy = format_name == "ms" and input_dtype == "complex128"
    history = tuple(result.history) + (
        f"radiosim_version={_package_version()}",
        f"standard_format={format_name}",
        f"input_visibility_dtype={input_dtype}",
        f"stored_visibility_dtype={stored_visibility_dtype}",
        f"lossy_visibility_conversion={'true' if lossy else 'false'}",
        f"input_weight_dtype={result.weights.dtype.name}",
        "stored_weight_dtype=float32",
        f"normalized_parallel_auto_samples={normalized_autos}",
        f"source_scientific_sha256={result.scientific_sha256}",
        f"source_provenance_sha256={result.provenance_sha256}",
        "original_phase_model=zenith_drift-altaz-time_dependent",
        "full_provenance=available-in-hdf5-or-summary-metadata",
    )
    record: dict[str, object] = {
        "schema": PROJECTED_PHASE_SCHEMA,
        "projected_phase": _json_tree(phase_center.to_snapshot()),
        "source_scientific_sha256": result.scientific_sha256,
        "source_provenance_sha256": result.provenance_sha256,
        "input_visibility_dtype": input_dtype,
        "stored_visibility_dtype": stored_visibility_dtype,
        "input_weight_dtype": result.weights.dtype.name,
        "stored_weight_dtype": "float32",
        "polarization_basis": result.polarization_basis,
        "receptor_sha256": result.receptors.provenance.receptor_sha256,
        "instrument": _json_tree(result.instrument.to_snapshot()),
        "beam": _json_tree(result.beam_state.to_snapshot()),
        "solver": _json_tree(result.solver.to_snapshot()),
    }
    encoded = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    if (
        len((PROJECTION_HISTORY_PREFIX + encoded).encode("utf-8"))
        > _PROJECTION_HISTORY_LIMIT
    ):
        record["instrument"] = {
            "name": result.instrument.name,
            "instrument_sha256": result.instrument.provenance.instrument_sha256,
        }
        record["beam"] = {
            "provenance_omitted": True,
            "full_provenance": "available in HDF5 or summary metadata",
        }
        record["solver"] = _json_tree(result.solver.to_snapshot())
        record["provenance_omitted"] = True
        encoded = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    projection_line = PROJECTION_HISTORY_PREFIX + encoded
    complete_history = "\n".join(history + (projection_line,))
    try:
        complete_encoded = complete_history.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise FormatRepresentationError(
            "standard projection HISTORY must be strict UTF-8"
        ) from exc
    if "\x00" in complete_history:
        raise FormatRepresentationError(
            "standard projection HISTORY must not contain NUL"
        )
    if len(complete_encoded) > _PROJECTION_HISTORY_LIMIT:
        raise FormatRepresentationError(
            "standard projection HISTORY exceeds 16000 UTF-8 bytes"
        )
    return history, projection_line


def project_simulation_result(
    result: SimulationResult,
    *,
    format: Literal["ms", "uvfits"],
) -> _ProjectedVisibility:
    """Build, explicitly phase, check, and snapshot one standard UVData view."""
    typed = validate_projection_result(result, format_name=format)
    basis = require_polarization_basis(typed.correlations)
    data, normalized_autos = normalize_autocorrelations(typed)
    stored_dtype = np.dtype("complex64") if format == "ms" else typed.visibilities.dtype
    data = np.array(data, dtype=stored_dtype, order="C", copy=True, subok=False)
    flags = np.array(typed.flags, dtype=np.bool_, order="C", copy=True, subok=False)
    weights = np.array(
        typed.weights,
        dtype=np.float32,
        order="C",
        copy=True,
        subok=False,
    )

    pyuvdata = import_module("pyuvdata")
    utilities = import_module("pyuvdata.utils")
    coordinates = import_module("astropy.coordinates")
    units = import_module("astropy.units")
    time_module = import_module("astropy.time")
    earth_location = coordinates.EarthLocation.from_geocentric(
        *typed.instrument.location.itrs_xyz_m,
        unit="m",
    )
    enu = np.asarray(
        [antenna.position_enu_m for antenna in typed.instrument.antennas],
        dtype=np.float64,
    )
    absolute_ecef = np.asarray(
        utilities.ECEF_from_ENU(enu, center_loc=earth_location),
        dtype=np.float64,
    )
    center_ecef = np.asarray(
        typed.instrument.location.itrs_xyz_m,
        dtype=np.float64,
    )
    relative_ecef = absolute_ecef - center_ecef
    antenna_numbers = [antenna.id.number for antenna in typed.instrument.antennas]
    telescope = pyuvdata.Telescope.new(
        name=typed.instrument.name,
        instrument=typed.instrument.name,
        location=earth_location,
        antenna_positions={
            number: relative_ecef[index] for index, number in enumerate(antenna_numbers)
        },
        antenna_names=[antenna.id.name for antenna in typed.instrument.antennas],
        antenna_numbers=antenna_numbers,
        antenna_diameters=[antenna.diameter_m for antenna in typed.instrument.antennas],
        feed_array=np.tile(
            np.asarray(PYUVDATA_FEEDS[basis], dtype="<U1"),
            (len(antenna_numbers), 1),
        ),
        feed_angle=np.tile(
            np.asarray(_NOMINAL_FEED_ANGLES_RAD[basis], dtype=np.float64),
            (len(antenna_numbers), 1),
        ),
        mount_type="fixed",
        update_from_known=False,
    )
    pairs = [
        (baseline.ant1.number, baseline.ant2.number)
        for baseline in typed.selection.baselines
    ]
    time_count, baseline_count, frequency_count, _ = data.shape
    uvdata = pyuvdata.UVData.new(
        freq_array=np.array(typed.frequencies_hz, dtype=np.float64, copy=True),
        polarization_array=list(PYUVDATA_POLARIZATIONS[basis]),
        times=typed.time_grid.to_jd(),
        antpairs=pairs,
        telescope=telescope,
        do_blt_outer=True,
        time_axis_faster_than_bls=False,
        update_telescope_from_known=False,
        integration_time=np.array(
            typed.time_grid.integration_time_seconds,
            dtype=np.float64,
            copy=True,
        ),
        channel_width=np.array(
            typed.channel_widths_hz,
            dtype=np.float64,
            copy=True,
        ),
        data_array=data.reshape(time_count * baseline_count, frequency_count, 4),
        flag_array=flags.reshape(time_count * baseline_count, frequency_count, 4),
        nsample_array=weights.reshape(
            time_count * baseline_count,
            frequency_count,
            4,
        ),
        history="",
        vis_units="Jy",
    )
    uvdata.polarization_array = np.asarray(
        uvdata.polarization_array,
        dtype=np.int64,
    )
    if not np.array_equal(
        uvdata.polarization_array,
        np.asarray(AIPS_CODES_CANONICAL[basis], dtype=np.int64),
    ):
        raise FormatRepresentationError(
            "pyuvdata did not preserve canonical polarization order"
        )
    require_feed_polarization_coupling(uvdata, basis)
    expected_unprojected_uvw = np.tile(
        np.asarray(
            [baseline.vector_enu_m for baseline in typed.selection.baselines],
            dtype=np.float64,
        ),
        (time_count, 1),
    )
    if not np.allclose(
        uvdata.uvw_array,
        expected_unprojected_uvw,
        rtol=0.0,
        atol=1e-6,
    ):
        raise FormatRepresentationError(
            "pyuvdata unprojected UVW disagrees with canonical ant2-ant1 geometry"
        )

    first_time = time_module.Time(
        float(typed.time_grid.utc_jd1[0]),
        float(typed.time_grid.utc_jd2[0]),
        format="jd",
        scale="utc",
    )
    zenith = coordinates.SkyCoord(
        az=0.0 * units.rad,
        alt=(math.pi / 2.0) * units.rad,
        frame=coordinates.AltAz(
            obstime=first_time,
            location=earth_location,
        ),
    ).transform_to(coordinates.ICRS())
    uvdata.phase_to_time(first_time)
    if len(uvdata.phase_center_catalog) != 1:
        raise FormatRepresentationError(
            "standard projection did not produce exactly one phase catalog"
        )
    catalog = next(iter(uvdata.phase_center_catalog.values()))
    if catalog.get("cat_type") != "sidereal" or catalog.get("cat_frame") != "icrs":
        raise FormatRepresentationError(
            "standard projection did not produce one ICRS sidereal catalog"
        )
    longitude = float(catalog["cat_lon"]) % (2.0 * math.pi)
    latitude = float(catalog["cat_lat"])
    if not np.allclose(
        [longitude, latitude],
        [float(zenith.ra.rad) % (2.0 * math.pi), float(zenith.dec.rad)],
        rtol=0.0,
        atol=1e-12,
    ):
        raise FormatRepresentationError(
            "pyuvdata projected catalog disagrees with the Astropy first-time zenith"
        )
    projected_phase = ProjectedPhaseCenter(
        longitude_rad=longitude,
        latitude_rad=latitude,
        reference_utc_jd1=float(typed.time_grid.utc_jd1[0]),
        reference_utc_jd2=float(typed.time_grid.utc_jd2[0]),
        original_phase_snapshot=dict(typed.phase_center.to_snapshot()),
        transformation=PROJECTION_TRANSFORMATION,
    )
    history, projection_line = _projection_history(
        typed,
        projected_phase,
        format_name=format,
        stored_visibility_dtype=stored_dtype.name,
        normalized_autos=normalized_autos,
    )
    uvdata.history = "\n".join(history + (projection_line,))
    if uvdata.check() is not True:
        raise FormatRepresentationError("UVData.check() did not return success")
    standard = build_standard_visibility_data(
        format=format,
        visibilities=np.asarray(uvdata.data_array).reshape(
            time_count,
            baseline_count,
            frequency_count,
            4,
        ),
        flags=np.asarray(uvdata.flag_array).reshape(
            time_count,
            baseline_count,
            frequency_count,
            4,
        ),
        weights=np.asarray(uvdata.nsample_array, dtype=np.float32).reshape(
            time_count,
            baseline_count,
            frequency_count,
            4,
        ),
        utc_jd1=np.array(typed.time_grid.utc_jd1, dtype=np.float64, copy=True),
        utc_jd2=np.array(typed.time_grid.utc_jd2, dtype=np.float64, copy=True),
        exposure_seconds=np.array(
            typed.time_grid.integration_time_seconds,
            dtype=np.float64,
            copy=True,
        ),
        frequencies_hz=np.array(
            typed.frequencies_hz,
            dtype=np.float64,
            copy=True,
        ),
        channel_widths_hz=np.array(
            typed.channel_widths_hz,
            dtype=np.float64,
            copy=True,
        ),
        correlations=CORRELATION_LABELS[basis],
        antenna1_numbers=np.array(
            [first for first, _second in pairs],
            dtype=np.int64,
        ),
        antenna2_numbers=np.array(
            [second for _first, second in pairs],
            dtype=np.int64,
        ),
        uvw_m=np.asarray(uvdata.uvw_array, dtype=np.float64).reshape(
            time_count,
            baseline_count,
            3,
        ),
        telescope_snapshot=_canonical_telescope_snapshot(typed),
        phase_center=projected_phase,
        history=history + (projection_line,),
        source_scientific_sha256=typed.scientific_sha256,
        source_provenance_sha256=typed.provenance_sha256,
    )
    return _ProjectedVisibility(uvdata=uvdata, data=standard)


def enforce_standard_read_limits(
    uvdata: Any,
    limits: StandardReadLimits,
) -> None:
    """Enforce standard visibility limits using metadata-only UVData state."""
    if type(limits) is not StandardReadLimits:
        raise TypeError("limits must be an exact StandardReadLimits")
    counts = {
        "max_times": int(uvdata.Ntimes),
        "max_baselines": int(uvdata.Nbls),
        "max_frequencies": int(uvdata.Nfreqs),
        "max_antennas": int(uvdata.telescope.Nants),
    }
    for field_name, count in counts.items():
        if count <= 0:
            raise UnsafeResultInputError(
                f"standard input has a nonpositive {field_name} axis"
            )
        if count > getattr(limits, field_name):
            raise UnsafeResultInputError(f"standard input exceeds {field_name}")
    if int(uvdata.Npols) != 4:
        raise FormatRepresentationError(
            "standard input must contain exactly four polarizations"
        )
    elements = (
        counts["max_times"] * counts["max_baselines"] * counts["max_frequencies"] * 4
    )
    if elements > limits.max_visibility_elements:
        raise UnsafeResultInputError("standard input exceeds max_visibility_elements")
    worst_case_bytes = elements * (
        np.dtype("complex128").itemsize
        + np.dtype("bool").itemsize
        + np.dtype("float32").itemsize
    )
    if worst_case_bytes > limits.max_data_bytes:
        raise UnsafeResultInputError("standard input exceeds max_data_bytes")


def validate_standard_metadata(uvdata: Any) -> PolarizationBasis:
    """Validate format-independent metadata before science allocation.

    Returns
    -------
    PolarizationBasis
        The output basis the input declares on its polarization axis.
    """
    if int(getattr(uvdata, "Nspws", 0)) != 1:
        raise FormatRepresentationError(
            "standard input must contain exactly one spectral window"
        )
    if int(uvdata.Nblts) != int(uvdata.Ntimes) * int(uvdata.Nbls):
        raise FormatRepresentationError(
            "standard input is not rectangular time-by-baseline data"
        )
    basis = basis_for_file_codes(uvdata.polarization_array)
    require_feed_polarization_coupling(uvdata, basis)
    if np.asarray(uvdata.freq_array).reshape(-1).shape != (
        int(uvdata.Nfreqs),
    ) or np.asarray(uvdata.channel_width).shape != (int(uvdata.Nfreqs),):
        raise FormatRepresentationError(
            "standard input has an unsupported spectral layout"
        )
    if len(uvdata.phase_center_catalog) != 1:
        raise FormatRepresentationError(
            "standard input must contain exactly one phase catalog"
        )
    catalog = next(iter(uvdata.phase_center_catalog.values()))
    if (
        catalog.get("cat_type") != "sidereal"
        or catalog.get("cat_frame") != "icrs"
        or not math.isfinite(float(catalog.get("cat_lon", math.nan)))
        or not math.isfinite(float(catalog.get("cat_lat", math.nan)))
    ):
        raise FormatRepresentationError(
            "standard input phase catalog must be finite ICRS sidereal metadata"
        )
    antenna_numbers = np.asarray(uvdata.telescope.antenna_numbers)
    antenna_names = np.asarray(uvdata.telescope.antenna_names)
    if (
        antenna_numbers.ndim != 1
        or antenna_names.ndim != 1
        or antenna_numbers.size != antenna_names.size
        or len({int(item) for item in antenna_numbers}) != antenna_numbers.size
        or len({str(item) for item in antenna_names}) != antenna_names.size
    ):
        raise FormatRepresentationError(
            "standard input antenna identity must be unique"
        )
    for name, values in (
        ("time", uvdata.time_array),
        ("frequency", uvdata.freq_array),
        ("channel width", uvdata.channel_width),
        ("exposure", uvdata.integration_time),
        ("UVW", uvdata.uvw_array),
        ("antenna position", uvdata.telescope.antenna_positions),
    ):
        if values is None or not np.all(np.isfinite(np.asarray(values))):
            raise FormatRepresentationError(
                f"standard input {name} metadata must be finite"
            )
    if (
        not np.all(np.asarray(uvdata.freq_array) > 0.0)
        or not np.all(np.asarray(uvdata.channel_width) > 0.0)
        or not np.all(np.asarray(uvdata.integration_time) > 0.0)
    ):
        raise FormatRepresentationError(
            "standard input frequencies, widths, and exposures must be positive"
        )
    return basis


class _DuplicateProjectionKey(ValueError):
    """Internal sentinel for duplicate JSON object names."""


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value}")


def _projection_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateProjectionKey(key)
        result[key] = value
    return result


def _validate_json_depth(
    value: object,
    *,
    depth: int = 1,
) -> None:
    if depth > _MAX_PROJECTION_JSON_DEPTH:
        raise UnsafeResultInputError(
            "standard input projection HISTORY exceeds maximum JSON nesting"
        )
    if type(value) is dict:
        for child in cast(dict[str, object], value).values():
            _validate_json_depth(child, depth=depth + 1)
    elif type(value) is list:
        for child in cast(list[object], value):
            _validate_json_depth(child, depth=depth + 1)


def _validate_projection_record(record: dict[str, object]) -> None:
    fields = set(record)
    if fields not in (
        _PROJECTION_RECORD_FIELDS,
        _PROJECTION_RECORD_FIELDS | {"provenance_omitted"},
    ):
        raise UnsafeResultInputError(
            "standard input projection HISTORY has unexpected record fields"
        )
    if record.get("schema") != PROJECTED_PHASE_SCHEMA:
        raise UnsafeResultInputError(
            "standard input projection HISTORY has the wrong schema"
        )
    if "provenance_omitted" in record and record["provenance_omitted"] is not True:
        raise UnsafeResultInputError(
            "standard input projection HISTORY has invalid omission metadata"
        )
    for name in ("instrument", "beam", "solver"):
        if type(record[name]) is not dict:
            raise UnsafeResultInputError(
                f"standard input projection HISTORY {name} must be an object"
            )
    for name in (
        "source_scientific_sha256",
        "source_provenance_sha256",
        "receptor_sha256",
    ):
        value = record[name]
        if type(value) is not str or _SHA256.fullmatch(value) is None:
            raise UnsafeResultInputError(
                f"standard input projection HISTORY {name} is invalid"
            )
    if record["polarization_basis"] not in POLARIZATION_BASES:
        raise UnsafeResultInputError(
            "standard input projection HISTORY polarization_basis is not one of "
            f"{POLARIZATION_BASES!r}"
        )
    input_visibility_dtype = record["input_visibility_dtype"]
    stored_visibility_dtype = record["stored_visibility_dtype"]
    if input_visibility_dtype not in {"complex64", "complex128"} or (
        stored_visibility_dtype not in {"complex64", "complex128"}
    ):
        raise UnsafeResultInputError(
            "standard input projection HISTORY has invalid visibility dtypes"
        )
    if record["input_weight_dtype"] not in {"float32", "float64"} or (
        record["stored_weight_dtype"] != "float32"
    ):
        raise UnsafeResultInputError(
            "standard input projection HISTORY has invalid weight dtypes"
        )
    projected = record["projected_phase"]
    if type(projected) is not dict:
        raise UnsafeResultInputError(
            "standard input projection HISTORY lacks projected_phase"
        )
    try:
        _ = ProjectedPhaseCenter(**cast(dict[str, Any], projected))
    except Exception as exc:
        raise UnsafeResultInputError(
            "standard input projected phase record is invalid"
        ) from exc


def projection_record_from_history(
    history: object,
) -> tuple[dict[str, object], tuple[str, ...]]:
    if type(history) is not str:
        raise FormatRepresentationError(
            "standard input must contain RadioSim projection HISTORY"
        )
    if "\x00" in history:
        raise UnsafeResultInputError("standard input projection HISTORY contains NUL")
    try:
        encoded_history = history.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise UnsafeResultInputError(
            "standard input projection HISTORY is not strict UTF-8"
        ) from exc
    if len(encoded_history) > _PROJECTION_HISTORY_LIMIT:
        raise UnsafeResultInputError(
            "standard input projection HISTORY exceeds 16000 UTF-8 bytes"
        )
    lines = tuple(line.strip() for line in history.splitlines() if line.strip())
    record_indices = [
        index
        for index, line in enumerate(lines)
        if line.startswith(PROJECTION_HISTORY_PREFIX)
    ]
    if len(record_indices) != 1:
        raise FormatRepresentationError(
            "standard input must contain exactly one RadioSim projection record"
        )
    record_index = record_indices[0]
    encoded = "".join(
        (
            lines[record_index][len(PROJECTION_HISTORY_PREFIX) :],
            *lines[record_index + 1 :],
        )
    )
    try:
        decoded, end = json.JSONDecoder(
            parse_constant=_reject_json_constant,
            object_pairs_hook=_projection_object,
        ).raw_decode(encoded)
    except _DuplicateProjectionKey as exc:
        raise UnsafeResultInputError(
            "standard input projection HISTORY contains a duplicate JSON key"
        ) from exc
    except (json.JSONDecodeError, RecursionError) as exc:
        raise UnsafeResultInputError(
            "standard input projection HISTORY is invalid JSON"
        ) from exc
    except ValueError as exc:
        message = (
            "standard input projection HISTORY contains a non-finite JSON constant"
            if "non-finite JSON constant" in str(exc)
            else "standard input projection HISTORY is invalid JSON"
        )
        raise UnsafeResultInputError(message) from exc
    trailing = encoded[end:].strip()
    if trailing and _PYUVDATA_HISTORY_TRAILING.fullmatch(trailing) is None:
        raise UnsafeResultInputError(
            "standard input projection HISTORY has unexpected trailing content"
        )
    if type(decoded) is not dict:
        raise UnsafeResultInputError(
            "standard input projection HISTORY must be a JSON object"
        )
    record = cast(dict[str, object], decoded)
    _validate_json_depth(record)
    _validate_projection_record(record)
    return record, lines


def projected_phase_from_uvdata(
    uvdata: Any,
    record: Mapping[str, object],
) -> ProjectedPhaseCenter:
    projected = record.get("projected_phase")
    if type(projected) is not dict:
        raise UnsafeResultInputError(
            "standard input projection HISTORY lacks projected_phase"
        )
    values = cast(dict[str, object], projected)
    try:
        phase = ProjectedPhaseCenter(**cast(dict[str, Any], values))
    except Exception as exc:
        raise UnsafeResultInputError(
            "standard input projected phase record is invalid"
        ) from exc
    catalog = next(iter(uvdata.phase_center_catalog.values()))
    if not np.allclose(
        [phase.longitude_rad, phase.latitude_rad],
        [float(catalog["cat_lon"]) % (2.0 * math.pi), float(catalog["cat_lat"])],
        rtol=0.0,
        atol=1e-12,
    ):
        raise UnsafeResultInputError(
            "standard input phase catalog disagrees with projection HISTORY"
        )
    return phase


def _snapshot_from_uvdata(uvdata: Any) -> dict[str, object]:
    utilities = import_module("pyuvdata.utils")
    telescope = uvdata.telescope
    numbers = np.asarray(telescope.antenna_numbers, dtype=np.int64)
    names = np.asarray(telescope.antenna_names)
    relative_ecef = np.asarray(telescope.antenna_positions, dtype=np.float64)
    center = np.asarray(
        [value.to_value("m") for value in telescope.location.geocentric],
        dtype=np.float64,
    )
    positions_enu = np.asarray(
        utilities.ENU_from_ECEF(
            relative_ecef + center,
            center_loc=telescope.location,
        ),
        dtype=np.float64,
    )
    diameters = np.asarray(telescope.antenna_diameters, dtype=np.float64)
    if (
        positions_enu.shape != (numbers.size, 3)
        or diameters.shape != (numbers.size,)
        or not np.all(np.isfinite(positions_enu))
        or not np.all(np.isfinite(diameters))
        or not np.all(diameters > 0.0)
    ):
        raise FormatRepresentationError(
            "standard input telescope metadata is incomplete or invalid"
        )
    name = str(telescope.name)
    instrument = str(telescope.instrument or telescope.name)
    return {
        "name": name,
        "instrument": instrument,
        "location_itrs_xyz_m": [float(item) for item in center],
        "antennas": [
            {
                "number": int(numbers[index]),
                "name": str(names[index]),
                "position_enu_m": [float(item) for item in positions_enu[index]],
                "diameter_m": float(diameters[index]),
            }
            for index in range(numbers.size)
        ],
    }


def standard_visibility_from_uvdata(
    uvdata: Any,
    *,
    format: Literal["ms", "uvfits"],
    expected_projection_record: Mapping[str, object] | None = None,
) -> StandardVisibilityData:
    """Canonicalize fully loaded rectangular UVData into immutable axes."""
    basis = validate_standard_metadata(uvdata)
    record, history = projection_record_from_history(uvdata.history)
    if expected_projection_record is not None and record != dict(
        expected_projection_record
    ):
        raise UnsafeResultInputError(
            "loaded projection HISTORY disagrees with the bounded preflight record"
        )
    if record["polarization_basis"] != basis:
        raise UnsafeResultInputError(
            "standard input projection HISTORY declares "
            f"polarization_basis={record['polarization_basis']!r} but its "
            f"polarization axis carries {basis!r}"
        )
    phase = projected_phase_from_uvdata(uvdata, record)
    if uvdata.data_array is None:
        raise UnsafeResultInputError("standard input science data were not loaded")
    times = np.asarray(uvdata.time_array, dtype=np.float64)
    unique_times = np.unique(times)
    pairs = list(
        zip(
            np.asarray(uvdata.ant_1_array, dtype=np.int64).tolist(),
            np.asarray(uvdata.ant_2_array, dtype=np.int64).tolist(),
            strict=True,
        )
    )
    first_time_rows = np.flatnonzero(times == unique_times[0])
    pair_order = [pairs[int(index)] for index in first_time_rows]
    if len(pair_order) != int(uvdata.Nbls) or len(set(pair_order)) != len(pair_order):
        raise FormatRepresentationError(
            "standard input first time does not contain one unique baseline set"
        )
    row_by_time_pair: dict[tuple[float, tuple[int, int]], int] = {}
    for row, (time_value, pair) in enumerate(zip(times, pairs, strict=True)):
        key = (float(time_value), pair)
        if key in row_by_time_pair:
            raise FormatRepresentationError(
                "standard input contains a duplicate time-baseline row"
            )
        row_by_time_pair[key] = row
    expected_keys = {
        (float(time_value), pair) for time_value in unique_times for pair in pair_order
    }
    if set(row_by_time_pair) != expected_keys:
        raise FormatRepresentationError(
            "standard input is not rectangular time-by-baseline data"
        )
    row_order = [
        row_by_time_pair[(float(time_value), pair)]
        for time_value in unique_times
        for pair in pair_order
    ]
    codes = np.asarray(uvdata.polarization_array, dtype=np.int64)
    canonical_indices = [
        int(np.flatnonzero(codes == code)[0]) for code in AIPS_CODES_CANONICAL[basis]
    ]
    data = np.asarray(uvdata.data_array)[row_order][:, :, canonical_indices]
    flags = np.asarray(uvdata.flag_array)[row_order][:, :, canonical_indices]
    weights = np.asarray(uvdata.nsample_array)[row_order][:, :, canonical_indices]
    time_count = unique_times.size
    baseline_count = len(pair_order)
    frequency_count = int(uvdata.Nfreqs)
    data = data.reshape(time_count, baseline_count, frequency_count, 4)
    flags = flags.reshape(time_count, baseline_count, frequency_count, 4)
    weights = np.asarray(weights, dtype=np.float32).reshape(
        time_count,
        baseline_count,
        frequency_count,
        4,
    )
    exposures_by_row = np.asarray(uvdata.integration_time, dtype=np.float64)[
        row_order
    ].reshape(time_count, baseline_count)
    if not np.all(exposures_by_row == exposures_by_row[:, :1]):
        raise FormatRepresentationError(
            "standard input exposures differ within one canonical time"
        )
    uvw = np.asarray(uvdata.uvw_array, dtype=np.float64)[row_order].reshape(
        time_count,
        baseline_count,
        3,
    )
    time_module = import_module("astropy.time")
    astropy_times = time_module.Time(unique_times, format="jd", scale="utc")
    scientific = cast(str, record["source_scientific_sha256"])
    provenance = cast(str, record["source_provenance_sha256"])
    return build_standard_visibility_data(
        format=format,
        visibilities=np.asarray(data),
        flags=np.asarray(flags, dtype=np.bool_),
        weights=weights,
        utc_jd1=np.asarray(astropy_times.jd1, dtype=np.float64),
        utc_jd2=np.asarray(astropy_times.jd2, dtype=np.float64),
        exposure_seconds=exposures_by_row[:, 0],
        frequencies_hz=np.asarray(uvdata.freq_array, dtype=np.float64).reshape(-1),
        channel_widths_hz=np.asarray(
            uvdata.channel_width,
            dtype=np.float64,
        ),
        correlations=CORRELATION_LABELS[basis],
        antenna1_numbers=np.array(
            [pair[0] for pair in pair_order],
            dtype=np.int64,
        ),
        antenna2_numbers=np.array(
            [pair[1] for pair in pair_order],
            dtype=np.int64,
        ),
        uvw_m=uvw,
        telescope_snapshot=_snapshot_from_uvdata(uvdata),
        phase_center=phase,
        history=history,
        source_scientific_sha256=scientific,
        source_provenance_sha256=provenance,
    )


__all__ = [
    "ProjectedPhaseCenter",
    "StandardReadLimits",
    "StandardVisibilityData",
]
