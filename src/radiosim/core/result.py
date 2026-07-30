"""Immutable canonical simulation-result models and fingerprints."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import DTypeLike, NDArray
from typing_extensions import override

from radiosim.core.polarization_basis import (
    CORRELATION_LABELS,
    POLARIZATION_BASES,
    PolarizationBasis,
    basis_for_correlations,
    parallel_hand_indices,
)
from radiosim.core.runtime_config import FrozenMapping, json_safe_mapping

if TYPE_CHECKING:
    from radiosim.backends.base import ArrayBackend
    from radiosim.core.beam.models import LoadedBeamState
    from radiosim.core.instrument import (
        ResolvedBaselineSelection,
        ResolvedInstrument,
    )
    from radiosim.core.phase_center import PhaseCenter
    from radiosim.core.receptor import ResolvedReceptorSet
    from radiosim.core.time_grid import ObservationTimeGrid

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
# Must equal ``radiosim.core.receptor``'s resolved receptor schema version; the
# equality is asserted by ``tests/unit/test_core/test_result.py`` so the two
# never drift silently.
_RECEPTOR_SCHEMA_VERSION = "1.0.0"
_RECEPTOR_ROW_KEYS = (
    "antenna_number",
    "antenna_name",
    "basis",
    "feed_rotation_rad",
    "feed_angle_rad",
)
_RECEPTOR_BASES = ("linear", "circular")


def _accepted_correlations_text() -> str:
    """Return the rejection text naming both accepted correlation tuples."""
    return " or ".join(repr(CORRELATION_LABELS[basis]) for basis in POLARIZATION_BASES)


class ResultError(RuntimeError):
    """Base class for canonical result failures."""


class ResultUnavailableError(ResultError):
    """A requested canonical result does not exist."""


class InvalidResultError(ResultError):
    """A result violates the canonical model contract."""


class ResultShapeError(InvalidResultError):
    """A result array shape is incoherent."""


class ResultCoordinateError(InvalidResultError):
    """A result coordinate is invalid."""


class InvalidPhaseCenterError(InvalidResultError):
    """A phase center is invalid or scientifically unsupported."""


class InvalidTimeGridError(InvalidResultError):
    """An observation time grid is invalid."""


class TimeGridLimitError(InvalidTimeGridError):
    """The requested time grid exceeds its allocation limit."""

    def __init__(self, *, requested_count: int, limit: int) -> None:
        self.requested_count = int(requested_count)
        self.limit = int(limit)
        super().__init__(
            f"requested {self.requested_count} time samples; limit is {self.limit}"
        )


def _reject_subclass(name: str) -> None:
    raise TypeError(f"{name} cannot be subclassed")


def _nonblank(value: object, *, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be nonblank")
    try:
        _ = normalized.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field_name} must contain valid UTF-8 text") from exc
    return normalized


@dataclass(frozen=True, slots=True)
class BackendResultProvenance:
    """Backend request, realization, precision, and output dtype identity."""

    requested_backend: str
    actual_backend: str
    requested_precision: FrozenMapping | Mapping[str, object]
    actual_precision: FrozenMapping | Mapping[str, object]
    result_dtype: str

    def __init_subclass__(cls, **kwargs: object) -> None:
        _reject_subclass("BackendResultProvenance")

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requested_backend",
            _nonblank(self.requested_backend, field_name="requested_backend"),
        )
        object.__setattr__(
            self,
            "actual_backend",
            _nonblank(self.actual_backend, field_name="actual_backend"),
        )
        object.__setattr__(
            self,
            "requested_precision",
            json_safe_mapping(self.requested_precision),
        )
        object.__setattr__(
            self,
            "actual_precision",
            json_safe_mapping(self.actual_precision),
        )
        try:
            dtype = np.dtype(self.result_dtype)
        except TypeError as exc:
            raise InvalidResultError("result_dtype is not a NumPy dtype") from exc
        if dtype.kind != "c" or dtype.itemsize not in {8, 16, 32}:
            raise InvalidResultError(
                "result_dtype must be complex64, complex128, or complex256"
            )
        object.__setattr__(self, "result_dtype", dtype.name)

    def to_snapshot(self) -> FrozenMapping:
        return json_safe_mapping(
            {
                "requested_backend": self.requested_backend,
                "actual_backend": self.actual_backend,
                "requested_precision": self.requested_precision,
                "actual_precision": self.actual_precision,
                "result_dtype": self.result_dtype,
            }
        )


@dataclass(frozen=True, slots=True)
class SolverResultProvenance:
    """Solver identity and scientific execution convention."""

    solver: Literal["rime"]
    sky_representation: Literal["point_sources", "healpix_map"]
    convention: Literal["radiosim.rime-zenith-drift.v1"]
    execution_path: Literal["scalar", "polarized"]

    def __init_subclass__(cls, **kwargs: object) -> None:
        _reject_subclass("SolverResultProvenance")

    def __post_init__(self) -> None:
        if self.solver != "rime":
            raise InvalidResultError("solver must be 'rime'")
        if self.sky_representation not in {"point_sources", "healpix_map"}:
            raise InvalidResultError("sky_representation is unsupported")
        if self.convention != "radiosim.rime-zenith-drift.v1":
            raise InvalidResultError("convention is unsupported")
        if self.execution_path not in {"scalar", "polarized"}:
            raise InvalidResultError("execution_path is unsupported")

    def to_snapshot(self) -> FrozenMapping:
        return json_safe_mapping(
            {
                "solver": self.solver,
                "sky_representation": self.sky_representation,
                "convention": self.convention,
                "execution_path": self.execution_path,
            }
        )


@dataclass(frozen=True, slots=True)
class ResultPerformance:
    """Finite nonnegative timings for one result construction."""

    setup_seconds: float
    solver_seconds: float
    result_construction_seconds: float
    host_transfer_seconds: float
    total_seconds: float

    def __init_subclass__(cls, **kwargs: object) -> None:
        _reject_subclass("ResultPerformance")

    def __post_init__(self) -> None:
        normalized: dict[str, float] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (bool, np.bool_)):
                raise InvalidResultError(f"{field.name} must be finite and nonnegative")
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise InvalidResultError(
                    f"{field.name} must be finite and nonnegative"
                ) from exc
            if not math.isfinite(number) or number < 0.0:
                raise InvalidResultError(f"{field.name} must be finite and nonnegative")
            normalized[field.name] = number
            object.__setattr__(self, field.name, number)
        minimum_total = (
            normalized["setup_seconds"]
            + normalized["solver_seconds"]
            + normalized["result_construction_seconds"]
            + normalized["host_transfer_seconds"]
        )
        allowance = 32.0 * np.finfo(np.float64).eps * max(1.0, minimum_total)
        if normalized["total_seconds"] + allowance < minimum_total:
            raise InvalidResultError(
                "total_seconds is not coherent with component times"
            )

    def to_snapshot(self) -> FrozenMapping:
        return json_safe_mapping(
            {field.name: getattr(self, field.name) for field in fields(self)}
        )


def _immutable_array(
    value: object,
    *,
    dtype: DTypeLike | None = None,
) -> NDArray[np.generic]:
    try:
        array = np.array(
            value,
            dtype=dtype,
            order="C",
            copy=True,
            subok=False,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidResultError("result array could not be normalized") from exc
    if array.dtype.hasobject:
        raise InvalidResultError("object arrays are not supported")
    return np.ndarray(array.shape, dtype=array.dtype, buffer=array.tobytes(order="C"))


def _coordinates(
    frequencies_hz: object,
    channel_widths_hz: object,
) -> tuple[np.ndarray, np.ndarray]:
    frequencies = cast(
        NDArray[np.float64],
        _immutable_array(frequencies_hz, dtype=np.float64),
    )
    widths = cast(
        NDArray[np.float64],
        _immutable_array(channel_widths_hz, dtype=np.float64),
    )
    if frequencies.ndim != 1 or not frequencies.size:
        raise ResultCoordinateError("frequencies_hz must be nonempty and 1-dimensional")
    if widths.shape != frequencies.shape:
        raise ResultCoordinateError("channel_widths_hz must match frequencies_hz shape")
    if (
        not np.all(np.isfinite(frequencies))
        or not np.all(frequencies > 0.0)
        or not np.all(np.diff(frequencies) > 0.0)
    ):
        raise ResultCoordinateError(
            "frequencies_hz must be finite, positive, and strictly increasing"
        )
    if not np.all(np.isfinite(widths)) or not np.all(widths > 0.0):
        raise ResultCoordinateError("channel_widths_hz must be finite and positive")
    return frequencies, widths


def _history(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise InvalidResultError("history must be a sequence of strings")
    return tuple(
        _nonblank(item, field_name=f"history[{index}]")
        for index, item in enumerate(value)
    )


def _runtime_snapshot(value: object) -> FrozenMapping:
    if not isinstance(value, Mapping):
        raise TypeError("resolved_config must be a mapping")
    mapping = cast(Mapping[str, object], value)
    return json_safe_mapping(
        {key: item for key, item in mapping.items() if key != "workflow"}
    )


def _optional_snapshot(
    value: object,
) -> FrozenMapping | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("configuration_provenance must be a mapping or None")
    mapping = cast(Mapping[str, object], value)
    snapshot = {key: item for key, item in mapping.items() if key != "workflow"}
    input_snapshot = snapshot.get("input_snapshot")
    if isinstance(input_snapshot, Mapping):
        snapshot["input_snapshot"] = {
            key: item
            for key, item in cast(Mapping[str, object], input_snapshot).items()
            if key != "workflow"
        }
    for field_name in ("override_origins", "path_resolutions"):
        provenance_values = snapshot.get(field_name)
        if isinstance(provenance_values, Mapping):
            snapshot[field_name] = {
                key: item
                for key, item in cast(
                    Mapping[str, object],
                    provenance_values,
                ).items()
                if not key.startswith("workflow.")
            }
    return json_safe_mapping(snapshot)


def _json_tree(value: object) -> object:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): _json_tree(item) for key, item in mapping.items()}
    if isinstance(value, tuple):
        sequence = cast(Sequence[object], value)
        return [_json_tree(item) for item in sequence]
    if isinstance(value, list):
        sequence = cast(Sequence[object], value)
        return [_json_tree(item) for item in sequence]
    if isinstance(value, np.generic):
        return cast(object, value.item())
    return value


def _tagged_update(digest: Any, tag: str, payload: bytes) -> None:
    tag_bytes = tag.encode("utf-8")
    digest.update(len(tag_bytes).to_bytes(8, "little"))
    digest.update(tag_bytes)
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


def _hash_json(digest: Any, tag: str, value: object) -> None:
    try:
        encoded = json.dumps(
            _json_tree(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise InvalidResultError(f"{tag} is not compact finite JSON") from exc
    _tagged_update(digest, tag, encoded)


def _hash_array(digest: Any, tag: str, value: np.ndarray) -> None:
    dtype = value.dtype.newbyteorder("<")
    canonical = np.array(value, dtype=dtype, order="C", copy=True, subok=False)
    _hash_json(
        digest,
        f"{tag}.metadata",
        {"dtype": dtype.str, "shape": list(value.shape)},
    )
    _tagged_update(digest, f"{tag}.data", canonical.tobytes(order="C"))


def _package_version() -> str:
    try:
        return version("radiosim")
    except PackageNotFoundError:
        return "unknown"


def _receptor_result_snapshot(snapshot: object) -> dict[str, object]:
    """Return the exact result-bearing projection of a receptor snapshot.

    The projection is the same set of values the HDF5 ``receptors/`` group
    stores (Section 21), so an in-memory result and a deserialized one produce
    byte-identical fingerprint input.  Configuration-only fields of
    :meth:`~radiosim.core.receptor.ResolvedReceptorSet.to_snapshot` -- the
    requested basis, the resolution rule, the override applications, and the
    per-antenna ``source`` and derived ``feed_array`` -- are excluded: they
    explain how the receptor set was chosen, not what it is.
    """
    if not isinstance(snapshot, Mapping):
        raise InvalidResultError("receptor snapshot must be a mapping")
    typed = cast(Mapping[str, object], snapshot)
    schema_version = typed.get("schema_version", _RECEPTOR_SCHEMA_VERSION)
    if schema_version != _RECEPTOR_SCHEMA_VERSION:
        raise InvalidResultError(
            f"receptor snapshot schema_version must be {_RECEPTOR_SCHEMA_VERSION!r}"
        )
    output_basis = typed.get("output_basis")
    if output_basis not in CORRELATION_LABELS:
        raise InvalidResultError(
            f"receptor snapshot output_basis must be one of {POLARIZATION_BASES!r}"
        )
    receptor_sha256 = typed.get("receptor_sha256")
    if type(receptor_sha256) is not str or _SHA256.fullmatch(receptor_sha256) is None:
        raise InvalidResultError(
            "receptor snapshot receptor_sha256 must be a lower-case SHA-256"
        )
    rows_value = typed.get("receptors")
    if isinstance(rows_value, (str, bytes)) or not isinstance(rows_value, Sequence):
        raise InvalidResultError("receptor snapshot receptors must be a sequence")
    rows: list[dict[str, object]] = []
    seen_numbers: set[int] = set()
    for index, row in enumerate(cast(Sequence[object], rows_value)):
        if not isinstance(row, Mapping):
            raise InvalidResultError(f"receptor snapshot receptors[{index}] is invalid")
        typed_row = cast(Mapping[str, object], row)
        missing = [key for key in _RECEPTOR_ROW_KEYS if key not in typed_row]
        if missing:
            raise InvalidResultError(
                f"receptor snapshot receptors[{index}] is missing {missing[0]}"
            )
        number = typed_row["antenna_number"]
        name = typed_row["antenna_name"]
        basis = typed_row["basis"]
        rotation = typed_row["feed_rotation_rad"]
        angles = typed_row["feed_angle_rad"]
        if type(number) is not int or number in seen_numbers:
            raise InvalidResultError(
                f"receptor snapshot receptors[{index}] antenna_number is invalid"
            )
        seen_numbers.add(number)
        if type(name) is not str or not name:
            raise InvalidResultError(
                f"receptor snapshot receptors[{index}] antenna_name is invalid"
            )
        if basis not in _RECEPTOR_BASES:
            raise InvalidResultError(
                f"receptor snapshot receptors[{index}] basis must be "
                f"one of {_RECEPTOR_BASES!r}"
            )
        if type(rotation) is not float or not math.isfinite(rotation):
            raise InvalidResultError(
                f"receptor snapshot receptors[{index}] feed_rotation_rad is invalid"
            )
        if isinstance(angles, (str, bytes)) or not isinstance(angles, Sequence):
            raise InvalidResultError(
                f"receptor snapshot receptors[{index}] feed_angle_rad is invalid"
            )
        angle_values = [
            float(value)
            for value in cast(Sequence[object], angles)
            if type(value) is float and math.isfinite(value)
        ]
        if len(angle_values) != 2 or len(cast(Sequence[object], angles)) != 2:
            raise InvalidResultError(
                f"receptor snapshot receptors[{index}] feed_angle_rad is invalid"
            )
        rows.append(
            {
                "antenna_number": number,
                "antenna_name": name,
                "basis": basis,
                "feed_rotation_rad": rotation,
                "feed_angle_rad": angle_values,
            }
        )
    if not rows:
        raise InvalidResultError("receptor snapshot must contain at least one antenna")
    return {
        "schema_version": _RECEPTOR_SCHEMA_VERSION,
        "output_basis": output_basis,
        "receptor_sha256": receptor_sha256,
        "receptors": rows,
    }


def _result_receptor_snapshot(
    result: SimulationResult | LoadedSimulationResult,
) -> dict[str, object]:
    receptors = result.receptors
    if isinstance(receptors, Mapping):
        return _receptor_result_snapshot(receptors)
    return _receptor_result_snapshot(receptors.to_snapshot())


def _scientific_hash(
    *,
    visibilities: np.ndarray,
    flags: np.ndarray,
    weights: np.ndarray,
    time_grid: ObservationTimeGrid,
    frequencies: np.ndarray,
    widths: np.ndarray,
    correlations: tuple[str, ...],
    polarization_basis: PolarizationBasis,
    receptor_snapshot: Mapping[str, object],
    phase_snapshot: Mapping[str, object],
    instrument_snapshot: Mapping[str, object],
    selection_snapshot: Mapping[str, object],
    beam_snapshot: Mapping[str, object],
    solver_snapshot: Mapping[str, object],
) -> str:
    digest = hashlib.sha256()
    _hash_json(digest, "schema", "radiosim.result.v1")
    for tag, array in (
        ("visibilities", visibilities),
        ("flags", flags),
        ("weights", weights),
        ("time.utc_jd1", time_grid.utc_jd1),
        ("time.utc_jd2", time_grid.utc_jd2),
        ("time.integration_time_seconds", time_grid.integration_time_seconds),
        ("frequency_hz", frequencies),
        ("channel_width_hz", widths),
    ):
        _hash_array(digest, tag, array)
    _hash_json(digest, "correlations", correlations)
    _hash_json(digest, "polarization_basis", polarization_basis)
    _hash_json(digest, "receptor", receptor_snapshot)
    _hash_json(digest, "instrument", instrument_snapshot)
    _hash_json(digest, "selection", selection_snapshot)
    _hash_json(digest, "beam", beam_snapshot)
    _hash_json(digest, "phase_center", phase_snapshot)
    _hash_json(digest, "solver", solver_snapshot)
    return digest.hexdigest()


def _provenance_hash(
    *,
    scientific_sha256: str,
    backend_snapshot: Mapping[str, object],
    resolved_config: Mapping[str, object],
    configuration_provenance: Mapping[str, object] | None,
    history: tuple[str, ...],
) -> str:
    digest = hashlib.sha256()
    _hash_json(digest, "scientific_sha256", scientific_sha256)
    _hash_json(digest, "backend", backend_snapshot)
    _hash_json(digest, "resolved_config", resolved_config)
    _hash_json(digest, "configuration_provenance", configuration_provenance)
    _hash_json(digest, "package_version", _package_version())
    _hash_json(digest, "history", history)
    return digest.hexdigest()


class _ResultMethods:
    schema_version: str
    visibilities: np.ndarray
    flags: np.ndarray
    weights: np.ndarray
    time_grid: ObservationTimeGrid
    frequencies_hz: np.ndarray
    channel_widths_hz: np.ndarray
    correlations: tuple[str, str, str, str]
    polarization_basis: PolarizationBasis
    receptors: ResolvedReceptorSet | FrozenMapping
    phase_center: PhaseCenter
    scientific_sha256: str
    provenance_sha256: str

    @override
    def __hash__(self) -> int:
        raise TypeError("canonical results are unhashable")

    def scientifically_equal(
        self,
        other: SimulationResult | LoadedSimulationResult,
        *,
        rtol: float = 0.0,
        atol: float = 0.0,
    ) -> bool:
        if type(other) not in {SimulationResult, LoadedSimulationResult}:
            return False
        if not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0 or atol < 0:
            raise ValueError("rtol and atol must be finite and nonnegative")
        if (
            self.schema_version != other.schema_version
            or self.visibilities.dtype != other.visibilities.dtype
            or self.visibilities.shape != other.visibilities.shape
            or self.flags.shape != other.flags.shape
            or self.weights.dtype != other.weights.dtype
            or self.correlations != other.correlations
            or self.polarization_basis != other.polarization_basis
            or _json_tree(self.phase_center.to_snapshot())
            != _json_tree(other.phase_center.to_snapshot())
            or _json_tree(
                _identity_snapshots(
                    cast(SimulationResult | LoadedSimulationResult, self)
                )
            )
            != _json_tree(_identity_snapshots(other))
        ):
            return False
        exact_pairs = (
            (self.flags, other.flags),
            (self.time_grid.utc_jd1, other.time_grid.utc_jd1),
            (self.time_grid.utc_jd2, other.time_grid.utc_jd2),
            (
                self.time_grid.integration_time_seconds,
                other.time_grid.integration_time_seconds,
            ),
            (self.frequencies_hz, other.frequencies_hz),
            (self.channel_widths_hz, other.channel_widths_hz),
        )
        if any(not np.array_equal(left, right) for left, right in exact_pairs):
            return False
        return bool(
            np.allclose(
                self.visibilities,
                other.visibilities,
                rtol=rtol,
                atol=atol,
                equal_nan=False,
            )
            and np.allclose(
                self.weights,
                other.weights,
                rtol=rtol,
                atol=atol,
                equal_nan=False,
            )
        )

    def stokes_i(self) -> np.ndarray:
        """Return a newly owned parallel-hand sum for the published basis.

        The two indices are derived from :attr:`correlations` through
        :func:`~radiosim.core.polarization_basis.parallel_hand_indices`, so the
        sum is ``XX + YY`` in ``linear_xy`` and ``RR + LL`` in ``circular_rl``
        without either literal appearing here.
        """
        first, second = parallel_hand_indices(self.correlations)
        if self.visibilities.shape[-1] != len(self.correlations):
            raise InvalidResultError(
                "the correlation axis does not match the correlation labels"
            )
        return np.array(
            self.visibilities[..., first] + self.visibilities[..., second],
            copy=True,
            order="C",
            subok=False,
        )

    def to_summary_snapshot(self) -> dict[str, object]:
        """Return bounded JSON-safe metadata without embedding science arrays."""
        receptor_snapshot = _result_receptor_snapshot(
            cast("SimulationResult | LoadedSimulationResult", self)
        )
        receptor_rows = cast(
            list[Mapping[str, object]],
            receptor_snapshot["receptors"],
        )
        native_counts = dict.fromkeys(_RECEPTOR_BASES, 0)
        for row in receptor_rows:
            native_counts[cast(str, row["basis"])] += 1
        return {
            "schema_version": self.schema_version,
            "shape": list(self.visibilities.shape),
            "dtype": self.visibilities.dtype.name,
            "correlations": list(self.correlations),
            "polarization_basis": self.polarization_basis,
            "receptor": {
                "output_basis": receptor_snapshot["output_basis"],
                "receptor_sha256": receptor_snapshot["receptor_sha256"],
                "native_basis_counts": native_counts,
                "antenna_count": len(receptor_rows),
            },
            "time": {
                "start_time_iso": self.time_grid.start_time_iso,
                "duration_seconds": self.time_grid.duration_seconds,
                "cadence_seconds": self.time_grid.cadence_seconds,
                "sample_count": len(self.time_grid),
            },
            "frequency": {
                "channel_count": int(self.frequencies_hz.size),
                "minimum_hz": float(self.frequencies_hz[0]),
                "maximum_hz": float(self.frequencies_hz[-1]),
            },
            "array_summaries": {
                "visibilities": {
                    "shape": list(self.visibilities.shape),
                    "dtype": self.visibilities.dtype.name,
                },
                "flags": {
                    "shape": list(self.flags.shape),
                    "dtype": self.flags.dtype.name,
                },
                "weights": {
                    "shape": list(self.weights.shape),
                    "dtype": self.weights.dtype.name,
                },
            },
            "scientific_sha256": self.scientific_sha256,
            "provenance_sha256": self.provenance_sha256,
        }


@dataclass(frozen=True, slots=True, init=False, eq=False)
class SimulationResult(_ResultMethods):
    """A canonical result retaining exact live runtime identity objects."""

    schema_version: str
    visibilities: np.ndarray
    flags: np.ndarray
    weights: np.ndarray
    time_grid: ObservationTimeGrid
    frequencies_hz: np.ndarray
    channel_widths_hz: np.ndarray
    correlations: tuple[str, str, str, str]
    polarization_basis: PolarizationBasis
    instrument: ResolvedInstrument
    selection: ResolvedBaselineSelection
    beam_state: LoadedBeamState
    receptors: ResolvedReceptorSet
    phase_center: PhaseCenter
    backend: BackendResultProvenance
    solver: SolverResultProvenance
    resolved_config: FrozenMapping
    configuration_provenance: FrozenMapping | None
    performance: ResultPerformance
    history: tuple[str, ...]
    scientific_sha256: str
    provenance_sha256: str

    def __init_subclass__(cls, **kwargs: object) -> None:
        _reject_subclass("SimulationResult")

    def __init__(self) -> None:
        raise TypeError("SimulationResult must be built by build_simulation_result")


@dataclass(frozen=True, slots=True, init=False, eq=False)
class LoadedSimulationResult(_ResultMethods):
    """A canonical deserialized result containing frozen identity snapshots."""

    schema_version: str
    visibilities: np.ndarray
    flags: np.ndarray
    weights: np.ndarray
    time_grid: ObservationTimeGrid
    frequencies_hz: np.ndarray
    channel_widths_hz: np.ndarray
    correlations: tuple[str, str, str, str]
    polarization_basis: PolarizationBasis
    phase_center: PhaseCenter
    instrument_snapshot: FrozenMapping
    selection_snapshot: FrozenMapping
    beam_snapshot: FrozenMapping
    receptors: FrozenMapping
    backend_snapshot: FrozenMapping
    solver_snapshot: FrozenMapping
    resolved_config_snapshot: FrozenMapping
    configuration_provenance_snapshot: FrozenMapping | None
    performance: ResultPerformance
    history: tuple[str, ...]
    scientific_sha256: str
    provenance_sha256: str

    def __init_subclass__(cls, **kwargs: object) -> None:
        _reject_subclass("LoadedSimulationResult")

    def __init__(self) -> None:
        raise TypeError(
            "LoadedSimulationResult must be built by build_loaded_simulation_result"
        )


def _identity_snapshots(
    result: SimulationResult | LoadedSimulationResult,
) -> tuple[object, ...]:
    receptor_snapshot = _result_receptor_snapshot(result)
    if isinstance(result, SimulationResult):
        return (
            result.instrument.to_snapshot(),
            result.selection.to_snapshot(),
            result.beam_state.to_snapshot(),
            receptor_snapshot,
            result.backend.to_snapshot(),
            result.solver.to_snapshot(),
        )
    return (
        result.instrument_snapshot,
        result.selection_snapshot,
        result.beam_snapshot,
        receptor_snapshot,
        result.backend_snapshot,
        result.solver_snapshot,
    )


def _require_exact(value: object, expected: type[Any], field_name: str) -> None:
    if type(value) is not expected:
        raise TypeError(f"{field_name} must be an exact {expected.__name__}")


def _require_backend(value: object) -> ArrayBackend:
    from radiosim.backends.base import ArrayBackend

    if not isinstance(value, ArrayBackend):
        raise TypeError("backend must be an ArrayBackend")
    return value


def _required_snapshot(value: object, *, field_name: str) -> FrozenMapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return json_safe_mapping(cast(Mapping[str, object], value))


def _snapshot_mapping(
    snapshot: Mapping[str, object],
    key: str,
    *,
    field_name: str,
) -> Mapping[str, object]:
    value = snapshot.get(key)
    if not isinstance(value, Mapping):
        raise InvalidResultError(f"{field_name}.{key} must be a mapping")
    return cast(Mapping[str, object], value)


def _snapshot_sequence(
    snapshot: Mapping[str, object],
    key: str,
    *,
    field_name: str,
) -> Sequence[object]:
    value = snapshot.get(key)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise InvalidResultError(f"{field_name}.{key} must be a sequence")
    return cast(Sequence[object], value)


def _validate_loaded_identity_snapshots(
    *,
    instrument: FrozenMapping,
    selection: FrozenMapping,
    beam: FrozenMapping,
    receptor: Mapping[str, object],
    backend: FrozenMapping,
    solver: FrozenMapping,
    visibility_dtype: np.dtype[Any],
    baseline_count: int,
) -> None:
    if instrument.get("schema_version") != "radiosim.instrument.v1":
        raise InvalidResultError("instrument_snapshot has an invalid schema")
    instrument_sha256 = instrument.get("instrument_sha256")
    if (
        type(instrument_sha256) is not str
        or _SHA256.fullmatch(instrument_sha256) is None
    ):
        raise InvalidResultError("instrument_snapshot has an invalid fingerprint")
    antennas = _snapshot_sequence(
        instrument,
        "antennas",
        field_name="instrument_snapshot",
    )
    if not antennas:
        raise InvalidResultError("instrument_snapshot.antennas must be nonempty")
    antenna_numbers: set[int] = set()
    antenna_identity: list[tuple[object, object]] = []
    for index, antenna in enumerate(antennas):
        if not isinstance(antenna, Mapping):
            raise InvalidResultError(
                f"instrument_snapshot.antennas[{index}] must be a mapping"
            )
        typed_antenna = cast(Mapping[str, object], antenna)
        number = typed_antenna.get("number")
        if type(number) is not int:
            raise InvalidResultError(
                "instrument_snapshot antenna numbers must be unique integers"
            )
        if number in antenna_numbers:
            raise InvalidResultError(
                "instrument_snapshot antenna numbers must be unique integers"
            )
        antenna_numbers.add(number)
        antenna_identity.append((number, typed_antenna.get("name")))

    receptor_rows = cast(Sequence[Mapping[str, object]], receptor["receptors"])
    if [
        (row["antenna_number"], row["antenna_name"]) for row in receptor_rows
    ] != antenna_identity:
        raise InvalidResultError(
            "the receptor snapshot does not cover instrument_snapshot in "
            "canonical antenna order"
        )

    if selection.get("schema_version") != "radiosim.baseline-selection.v1":
        raise InvalidResultError("selection_snapshot has an invalid schema")
    selected_ids = _snapshot_sequence(
        selection,
        "selected_ids",
        field_name="selection_snapshot",
    )
    if len(selected_ids) != baseline_count:
        raise ResultShapeError(
            "selection_snapshot baseline count does not match visibilities"
        )
    seen_pairs: set[tuple[int, int]] = set()
    for index, pair_value in enumerate(selected_ids):
        if isinstance(pair_value, (str, bytes)) or not isinstance(pair_value, Sequence):
            raise InvalidResultError(
                f"selection_snapshot.selected_ids[{index}] must contain two integers"
            )
        pair_items = cast(Sequence[object], pair_value)
        if len(pair_items) != 2:
            raise InvalidResultError(
                f"selection_snapshot.selected_ids[{index}] must contain two integers"
            )
        ant1, ant2 = pair_items[0], pair_items[1]
        if type(ant1) is not int or type(ant2) is not int:
            raise InvalidResultError(
                f"selection_snapshot.selected_ids[{index}] must contain two integers"
            )
        pair = (ant1, ant2)
        if ant1 not in antenna_numbers or ant2 not in antenna_numbers:
            raise InvalidResultError(
                "selection_snapshot contains an antenna outside instrument_snapshot"
            )
        if pair in seen_pairs:
            raise InvalidResultError(
                "selection_snapshot contains duplicate selected baselines"
            )
        seen_pairs.add(pair)

    resolved_beam = _snapshot_mapping(
        beam,
        "resolved",
        field_name="beam_snapshot",
    )
    if resolved_beam.get("instrument_fingerprint") != instrument_sha256:
        raise InvalidResultError("beam_snapshot does not belong to instrument_snapshot")

    backend_fields = {
        "requested_backend",
        "actual_backend",
        "requested_precision",
        "actual_precision",
        "result_dtype",
    }
    if set(backend) != backend_fields:
        raise InvalidResultError("backend_snapshot has unexpected fields")
    try:
        backend_identity = BackendResultProvenance(**dict(backend))
    except (TypeError, ValueError, InvalidResultError) as exc:
        raise InvalidResultError("backend_snapshot is invalid") from exc
    if np.dtype(backend_identity.result_dtype) != visibility_dtype:
        raise InvalidResultError(
            "backend_snapshot result dtype does not match visibilities"
        )

    solver_fields = {
        "solver",
        "sky_representation",
        "convention",
        "execution_path",
    }
    if set(solver) != solver_fields:
        raise InvalidResultError("solver_snapshot has unexpected fields")
    try:
        _ = SolverResultProvenance(**dict(solver))
    except (TypeError, ValueError, InvalidResultError) as exc:
        raise InvalidResultError("solver_snapshot is invalid") from exc


def _assign(target: object, **values: object) -> None:
    for key, value in values.items():
        object.__setattr__(target, key, value)


def _validate_common_shapes(
    *,
    visibilities: np.ndarray,
    flags: np.ndarray,
    weights: np.ndarray,
    time_grid: ObservationTimeGrid,
    frequencies: np.ndarray,
    baseline_count: int | None,
) -> None:
    if visibilities.ndim != 4 or any(size <= 0 for size in visibilities.shape):
        raise ResultShapeError("visibilities must have nonempty shape (T,B,F,4)")
    if visibilities.shape[-1] != 4:
        raise ResultShapeError("the correlation axis must contain exactly four values")
    if flags.shape != visibilities.shape or weights.shape != visibilities.shape:
        raise ResultShapeError("flags and weights must match visibility shape")
    if visibilities.shape[0] != len(time_grid):
        raise ResultShapeError("visibility time axis does not match time_grid")
    if visibilities.shape[2] != frequencies.size:
        raise ResultShapeError("visibility frequency axis does not match coordinates")
    if baseline_count is not None and visibilities.shape[1] != baseline_count:
        raise ResultShapeError("visibility baseline axis does not match selection")
    if not np.all(np.isfinite(visibilities)) or not np.all(np.isfinite(weights)):
        raise InvalidResultError("result arrays must contain only finite values")


def build_simulation_result(
    *,
    receptor_visibilities: object,
    backend: ArrayBackend,
    time_grid: ObservationTimeGrid,
    frequencies_hz: Sequence[float],
    channel_widths_hz: Sequence[float],
    instrument: ResolvedInstrument,
    selection: ResolvedBaselineSelection,
    beam_state: LoadedBeamState,
    receptors: ResolvedReceptorSet,
    phase_center: PhaseCenter,
    backend_provenance: BackendResultProvenance,
    solver_provenance: SolverResultProvenance,
    resolved_config: Mapping[str, object],
    configuration_provenance: Mapping[str, object] | None,
    performance: ResultPerformance,
    history: Sequence[str] = (),
) -> SimulationResult:
    """Validate, transfer once, flatten, harden, and fingerprint a result."""
    from radiosim.core.beam.models import LoadedBeamState
    from radiosim.core.instrument import ResolvedBaselineSelection, ResolvedInstrument
    from radiosim.core.phase_center import PhaseCenter
    from radiosim.core.receptor import ResolvedReceptorSet
    from radiosim.core.time_grid import ObservationTimeGrid

    construction_started = time.perf_counter()
    checked_backend = _require_backend(backend)
    for value, expected, field_name in (
        (time_grid, ObservationTimeGrid, "time_grid"),
        (instrument, ResolvedInstrument, "instrument"),
        (selection, ResolvedBaselineSelection, "selection"),
        (beam_state, LoadedBeamState, "beam_state"),
        (receptors, ResolvedReceptorSet, "receptors"),
        (phase_center, PhaseCenter, "phase_center"),
        (backend_provenance, BackendResultProvenance, "backend_provenance"),
        (solver_provenance, SolverResultProvenance, "solver_provenance"),
        (performance, ResultPerformance, "performance"),
    ):
        _require_exact(value, expected, field_name)
    if (
        selection.provenance.instrument_sha256
        != instrument.provenance.instrument_sha256
    ):
        raise InvalidResultError("selection does not belong to instrument")
    antenna_ids = {antenna.id for antenna in instrument.antennas}
    if any(
        baseline.ant1 not in antenna_ids or baseline.ant2 not in antenna_ids
        for baseline in selection.baselines
    ):
        raise InvalidResultError("selection contains a baseline outside instrument")
    if set(receptors.receptor_by_antenna) != antenna_ids:
        raise InvalidResultError("receptors do not belong to instrument")
    polarization_basis = receptors.output_basis
    if polarization_basis not in CORRELATION_LABELS:
        raise InvalidResultError(
            f"polarization_basis must be one of {POLARIZATION_BASES!r}"
        )
    receptor_snapshot = _receptor_result_snapshot(receptors.to_snapshot())

    frequencies, widths = _coordinates(frequencies_hz, channel_widths_hz)
    transfer_started = time.perf_counter()
    try:
        host = checked_backend.to_numpy(receptor_visibilities)
    except Exception as exc:
        raise InvalidResultError("backend host transfer failed") from exc
    host_transfer_seconds = time.perf_counter() - transfer_started
    if type(host) is not np.ndarray:
        host = np.asarray(host)
    expected_shape = (
        len(time_grid),
        len(selection.baselines),
        frequencies.size,
        2,
        2,
    )
    if host.shape != expected_shape:
        raise ResultShapeError(
            f"receptor_visibilities must have shape {expected_shape}, got {host.shape}"
        )
    dtype = np.dtype(backend_provenance.result_dtype)
    if dtype.kind != "c":
        raise InvalidResultError("result dtype must be complex")
    try:
        cast_host = np.array(host, dtype=dtype, order="C", copy=True, subok=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidResultError(
            "receptor visibilities cannot use result dtype"
        ) from exc
    if not np.all(np.isfinite(cast_host)):
        raise InvalidResultError("receptor visibilities must be finite")
    flattened = cast_host.reshape(expected_shape[:3] + (4,))
    visibilities = _immutable_array(flattened, dtype=dtype)
    flags = _immutable_array(np.zeros(visibilities.shape, dtype=np.bool_))
    weight_dtype = np.float32 if dtype.itemsize == 8 else np.float64
    weights = _immutable_array(
        np.ones(visibilities.shape, dtype=weight_dtype),
        dtype=weight_dtype,
    )
    _validate_common_shapes(
        visibilities=visibilities,
        flags=flags,
        weights=weights,
        time_grid=time_grid,
        frequencies=frequencies,
        baseline_count=len(selection.baselines),
    )
    frozen_config = _runtime_snapshot(resolved_config)
    frozen_provenance = _optional_snapshot(configuration_provenance)
    frozen_history = _history(history)
    instrument_snapshot = instrument.to_snapshot()
    selection_snapshot = selection.to_snapshot()
    beam_snapshot = beam_state.to_snapshot()
    backend_snapshot = backend_provenance.to_snapshot()
    solver_snapshot = solver_provenance.to_snapshot()
    scientific = _scientific_hash(
        visibilities=visibilities,
        flags=flags,
        weights=weights,
        time_grid=time_grid,
        frequencies=frequencies,
        widths=widths,
        correlations=CORRELATION_LABELS[polarization_basis],
        polarization_basis=polarization_basis,
        receptor_snapshot=receptor_snapshot,
        phase_snapshot=phase_center.to_snapshot(),
        instrument_snapshot=instrument_snapshot,
        selection_snapshot=selection_snapshot,
        beam_snapshot=beam_snapshot,
        solver_snapshot=solver_snapshot,
    )
    provenance_hash = _provenance_hash(
        scientific_sha256=scientific,
        backend_snapshot=backend_snapshot,
        resolved_config=frozen_config,
        configuration_provenance=frozen_provenance,
        history=frozen_history,
    )
    construction_elapsed = time.perf_counter() - construction_started
    result_construction_seconds = max(
        0.0,
        construction_elapsed - host_transfer_seconds,
    )
    measured_performance = ResultPerformance(
        setup_seconds=performance.setup_seconds,
        solver_seconds=performance.solver_seconds,
        result_construction_seconds=result_construction_seconds,
        host_transfer_seconds=host_transfer_seconds,
        total_seconds=performance.total_seconds + construction_elapsed,
    )
    result = object.__new__(SimulationResult)
    _assign(
        result,
        schema_version="radiosim.result.v1",
        visibilities=visibilities,
        flags=flags,
        weights=weights,
        time_grid=time_grid,
        frequencies_hz=frequencies,
        channel_widths_hz=widths,
        correlations=CORRELATION_LABELS[polarization_basis],
        polarization_basis=polarization_basis,
        instrument=instrument,
        selection=selection,
        beam_state=beam_state,
        receptors=receptors,
        phase_center=phase_center,
        backend=backend_provenance,
        solver=solver_provenance,
        resolved_config=frozen_config,
        configuration_provenance=frozen_provenance,
        performance=measured_performance,
        history=frozen_history,
        scientific_sha256=scientific,
        provenance_sha256=provenance_hash,
    )
    return result


def _performance_from_snapshot(snapshot: object) -> ResultPerformance:
    if not isinstance(snapshot, Mapping):
        raise TypeError("performance_snapshot must be a mapping")
    typed_snapshot = cast(Mapping[str, object], snapshot)
    expected = {field.name for field in fields(ResultPerformance)}
    if set(typed_snapshot) != expected:
        raise InvalidResultError("performance_snapshot has unexpected fields")
    return ResultPerformance(**cast(dict[str, Any], dict(typed_snapshot)))


def build_loaded_simulation_result(
    *,
    visibilities: object,
    flags: object,
    weights: object,
    time_grid: ObservationTimeGrid,
    frequencies_hz: object,
    channel_widths_hz: object,
    correlations: Sequence[str],
    phase_center: PhaseCenter,
    instrument_snapshot: Mapping[str, object],
    selection_snapshot: Mapping[str, object],
    beam_snapshot: Mapping[str, object],
    receptors_snapshot: Mapping[str, object],
    backend_snapshot: Mapping[str, object],
    solver_snapshot: Mapping[str, object],
    resolved_config_snapshot: Mapping[str, object],
    configuration_provenance_snapshot: Mapping[str, object] | None,
    performance_snapshot: Mapping[str, object],
    history: Sequence[str],
    expected_scientific_sha256: str,
    expected_provenance_sha256: str,
) -> LoadedSimulationResult:
    """Build and independently verify an immutable deserialized result."""
    from radiosim.core.phase_center import PhaseCenter
    from radiosim.core.time_grid import ObservationTimeGrid

    _require_exact(time_grid, ObservationTimeGrid, "time_grid")
    _require_exact(phase_center, PhaseCenter, "phase_center")
    if isinstance(correlations, (str, bytes)) or not isinstance(correlations, Sequence):
        raise InvalidResultError("correlations must be a sequence of labels")
    correlation_labels = tuple(
        label for label in cast(Sequence[object], correlations) if type(label) is str
    )
    try:
        polarization_basis = basis_for_correlations(correlation_labels)
    except (TypeError, ValueError) as exc:
        raise InvalidResultError(
            f"correlations must be exactly {_accepted_correlations_text()}"
        ) from exc
    receptor_snapshot = _receptor_result_snapshot(receptors_snapshot)
    if receptor_snapshot["output_basis"] != polarization_basis:
        raise InvalidResultError(
            "the receptor output basis does not match the correlation labels"
        )
    frequency_array, width_array = _coordinates(frequencies_hz, channel_widths_hz)
    visibility_array = _immutable_array(visibilities)
    if visibility_array.dtype.kind != "c" or visibility_array.dtype.itemsize not in {
        8,
        16,
        32,
    }:
        raise InvalidResultError("visibilities must use a supported complex dtype")
    try:
        flag_input = np.asarray(flags)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidResultError("flags could not be normalized") from exc
    if flag_input.dtype != np.dtype("bool"):
        raise InvalidResultError("flags must use bool dtype")
    flag_array = _immutable_array(flag_input, dtype=np.bool_)
    expected_weight_dtype = (
        np.dtype("float32")
        if visibility_array.dtype.itemsize == 8
        else np.dtype("float64")
    )
    weight_input = np.asarray(weights)
    if weight_input.dtype != expected_weight_dtype:
        raise InvalidResultError("weights dtype does not match visibility dtype")
    weight_array = _immutable_array(weight_input, dtype=expected_weight_dtype)
    _validate_common_shapes(
        visibilities=visibility_array,
        flags=flag_array,
        weights=weight_array,
        time_grid=time_grid,
        frequencies=frequency_array,
        baseline_count=None,
    )
    snapshots: list[FrozenMapping] = []
    for field_name, snapshot in (
        ("instrument_snapshot", instrument_snapshot),
        ("selection_snapshot", selection_snapshot),
        ("beam_snapshot", beam_snapshot),
        ("backend_snapshot", backend_snapshot),
        ("solver_snapshot", solver_snapshot),
    ):
        snapshots.append(_required_snapshot(snapshot, field_name=field_name))
    frozen_instrument, frozen_selection, frozen_beam, frozen_backend, frozen_solver = (
        snapshots
    )
    _validate_loaded_identity_snapshots(
        instrument=frozen_instrument,
        selection=frozen_selection,
        beam=frozen_beam,
        receptor=receptor_snapshot,
        backend=frozen_backend,
        solver=frozen_solver,
        visibility_dtype=visibility_array.dtype,
        baseline_count=visibility_array.shape[1],
    )
    frozen_config = _runtime_snapshot(resolved_config_snapshot)
    frozen_configuration_provenance = _optional_snapshot(
        configuration_provenance_snapshot
    )
    performance = _performance_from_snapshot(performance_snapshot)
    frozen_history = _history(history)
    for field_name, expected in (
        ("expected_scientific_sha256", expected_scientific_sha256),
        ("expected_provenance_sha256", expected_provenance_sha256),
    ):
        if type(expected) is not str or _SHA256.fullmatch(expected) is None:
            raise InvalidResultError(f"{field_name} must be a lower-case SHA-256")
    scientific = _scientific_hash(
        visibilities=visibility_array,
        flags=flag_array,
        weights=weight_array,
        time_grid=time_grid,
        frequencies=frequency_array,
        widths=width_array,
        correlations=correlation_labels,
        polarization_basis=polarization_basis,
        receptor_snapshot=receptor_snapshot,
        phase_snapshot=phase_center.to_snapshot(),
        instrument_snapshot=frozen_instrument,
        selection_snapshot=frozen_selection,
        beam_snapshot=frozen_beam,
        solver_snapshot=frozen_solver,
    )
    provenance_hash = _provenance_hash(
        scientific_sha256=scientific,
        backend_snapshot=frozen_backend,
        resolved_config=frozen_config,
        configuration_provenance=frozen_configuration_provenance,
        history=frozen_history,
    )
    if scientific != expected_scientific_sha256:
        raise InvalidResultError("scientific fingerprint mismatch")
    if provenance_hash != expected_provenance_sha256:
        raise InvalidResultError("provenance fingerprint mismatch")
    result = object.__new__(LoadedSimulationResult)
    _assign(
        result,
        schema_version="radiosim.result.v1",
        visibilities=visibility_array,
        flags=flag_array,
        weights=weight_array,
        time_grid=time_grid,
        frequencies_hz=frequency_array,
        channel_widths_hz=width_array,
        correlations=correlation_labels,
        polarization_basis=polarization_basis,
        phase_center=phase_center,
        instrument_snapshot=frozen_instrument,
        selection_snapshot=frozen_selection,
        beam_snapshot=frozen_beam,
        receptors=json_safe_mapping(receptor_snapshot),
        backend_snapshot=frozen_backend,
        solver_snapshot=frozen_solver,
        resolved_config_snapshot=frozen_config,
        configuration_provenance_snapshot=frozen_configuration_provenance,
        performance=performance,
        history=frozen_history,
        scientific_sha256=scientific,
        provenance_sha256=provenance_hash,
    )
    return result


__all__ = [
    "BackendResultProvenance",
    "InvalidPhaseCenterError",
    "InvalidResultError",
    "InvalidTimeGridError",
    "LoadedSimulationResult",
    "ResultCoordinateError",
    "ResultError",
    "ResultPerformance",
    "ResultShapeError",
    "ResultUnavailableError",
    "SimulationResult",
    "SolverResultProvenance",
    "TimeGridLimitError",
    "build_loaded_simulation_result",
    "build_simulation_result",
]
