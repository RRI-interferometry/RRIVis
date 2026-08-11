"""The benchmark record schema of ``Tier6HybridRuntimePlan.md`` Section 23.

A benchmark number without its measurement context is not evidence, so this
module makes the context structural rather than optional. :class:`BenchmarkRecord`
carries every field in ``Fix.md`` Section 15's mandatory list -- hardware and
accelerator, backend and version, precision, problem counts, setup versus steady
state, compilation time, host transfer time, peak memory, and the correctness
tolerance against NumPy -- and there is no way to build a partial one:
:meth:`BenchmarkRecord.create` raises :class:`BenchmarkRecordError` for a missing
field and ``__post_init__`` raises it for a ``None`` in any field the schema does
not declare nullable.

Two fields exist purely to keep the tier honest. ``accelerator`` is ``"none"``
for every record Tier 6 produces, because no accelerator was exercised, and
``unmeasured`` names what was *not* measured, so a reader never has to infer
absence from silence.

Two further record types accompany the main one, both added by Tier 6I to
discharge obligations the Tier 6H independent acceptance routed here:

- :class:`RetracingRecord` measures what happens when the compiled kernel's
  source axis changes size across a run, which the Section 22.2 timing loop
  (repeated *identical* calls) cannot surface;
- :class:`MemoryScalingRecord` measures the compiled kernel's ``(B, S, 2, 2)``
  working set against baseline and source counts, so the ``O(baselines x
  sources)`` hazard is bounded by evidence rather than by assumption.

Neither weakens Section 23: they are separate documents in the same output file,
and every :class:`BenchmarkRecord` still carries the full mandatory field set.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Self, TypeVar, cast

__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "MEMORY_SCALING_SCHEMA_VERSION",
    "PERF001_BACKEND_RESOLUTION_SCHEMA_VERSION",
    "PERF001_MEMORY_SCALING_SCHEMA_VERSION",
    "PERF001_PROVENANCE_SCHEMA_VERSION",
    "PERF001_RETRACING_SCHEMA_VERSION",
    "PERF001_SCHEMA_VERSION",
    "PERF001_SOLVER_MEMORY_SCHEMA_VERSION",
    "PERF001_TARGET_KERNEL_PAIRS",
    "PERF001_WORKLOAD_SCHEMA_VERSION",
    "RETRACING_SCHEMA_VERSION",
    "AcceleratorFacts",
    "BackendResolutionRecord",
    "BenchmarkDocument",
    "BenchmarkRecord",
    "BenchmarkRecordError",
    "ContractionSignatureObservation",
    "DeviceMemoryMeasurement",
    "MeasurementContext",
    "MemoryScalingRecord",
    "MemoryScalingRecordV2",
    "Perf001EvidenceDocument",
    "Perf001Provenance",
    "RetracingRecord",
    "RetracingRecordV2",
    "SolverMemoryRecord",
    "WorkloadBenchmarkRecordV2",
    "records_are_complete",
    "write_benchmark_document",
]

#: Identity of the Section 23 record schema. Bump this, never widen it silently.
BENCHMARK_SCHEMA_VERSION = "radiosim.benchmark.v1"

#: Identity of the retracing-measurement schema (Tier 6H acceptance obligation).
RETRACING_SCHEMA_VERSION = "radiosim.benchmark.retracing.v1"

#: Identity of the kernel working-set schema (Tier 6H acceptance obligation).
MEMORY_SCALING_SCHEMA_VERSION = "radiosim.benchmark.memory_scaling.v1"

#: The only two fields Section 23 declares nullable. Everything else is a
#: measurement, and an absent measurement is not a record.
NULLABLE_BENCHMARK_FIELDS = frozenset({"accelerator_driver", "precision_preset"})


class BenchmarkRecordError(ValueError):
    """A benchmark record is incomplete, or claims something it cannot support.

    Raised for a missing field, for a ``None`` in a field the schema does not
    declare nullable, and for the one honesty rule the schema enforces
    structurally: a record may not report an accelerator it did not describe.
    """


def _reject_missing(cls: type, values: Mapping[str, Any]) -> None:
    """Raise :class:`BenchmarkRecordError` when ``values`` is not a full record."""
    declared = {declared_field.name for declared_field in fields(cls)}
    missing = sorted(declared - set(values))
    if missing:
        raise BenchmarkRecordError(
            f"{cls.__name__} is missing mandatory field(s): {', '.join(missing)}. "
            "There is no partial record: every field in "
            "Tier6HybridRuntimePlan.md Section 23 must be measured or stated."
        )
    unknown = sorted(set(values) - declared)
    if unknown:
        raise BenchmarkRecordError(
            f"{cls.__name__} received unknown field(s): {', '.join(unknown)}."
        )


def _reject_none(instance: Any, nullable: frozenset[str]) -> None:
    """Raise :class:`BenchmarkRecordError` for any unexpected ``None``."""
    empty = sorted(
        declared_field.name
        for declared_field in fields(instance)
        if declared_field.name not in nullable
        and getattr(instance, declared_field.name) is None
    )
    if empty:
        raise BenchmarkRecordError(
            f"{type(instance).__name__} field(s) {', '.join(empty)} are None. "
            "A field that was not measured is not a record, it is a gap."
        )


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    """One reproducible measurement of one workload on one backend.

    Build these with :meth:`create` rather than by direct construction: the
    classmethod is what turns a missing field into a
    :class:`BenchmarkRecordError` instead of a ``TypeError``.
    """

    # identity
    schema_version: str
    recorded_at_utc: str
    radiosim_version: str
    git_sha: str
    # hardware and accelerator
    platform: str
    cpu_model: str
    cpu_count_logical: int
    accelerator: str
    accelerator_driver: str | None
    # backend and version
    backend_requested: str
    backend_actual: str
    backend_version: str
    device_kind: str
    compilation_used: bool
    # precision
    precision_preset: str | None
    precision_default: str
    precision_accumulation: str
    precision_output: str
    result_dtype: str
    # problem size
    workload: str
    n_antennas: int
    n_baselines: int
    n_point_sources: int
    n_healpix_pixels: int
    n_times: int
    n_frequencies: int
    sky_representation: str
    solver_workers: int
    loader_max_workers: int
    # timing
    setup_seconds: float
    compile_seconds: float
    steady_state_median_seconds: float
    steady_state_min_seconds: float
    steady_state_max_seconds: float
    steady_state_iterations: int
    host_transfer_seconds: float
    # memory
    peak_host_bytes: int
    backend_memory_info: dict[str, object]
    # correctness
    reference_backend: str
    max_absolute_deviation: float
    max_relative_deviation: float
    tolerance_rtol: float
    tolerance_atol: float
    within_tolerance: bool
    # honesty
    unmeasured: tuple[str, ...]

    def __post_init__(self) -> None:
        _reject_none(self, NULLABLE_BENCHMARK_FIELDS)
        if self.accelerator != "none" and not self.accelerator_driver:
            raise BenchmarkRecordError(
                f"accelerator={self.accelerator!r} without an accelerator_driver "
                "description. Section 23: a record claiming an accelerator "
                "without a corresponding hardware description is an acceptance "
                "failure."
            )
        if self.accelerator == "none" and "gpu" not in self.unmeasured:
            raise BenchmarkRecordError(
                "a record with accelerator='none' must list 'gpu' in unmeasured, "
                "so absence of an accelerator run is stated rather than inferred."
            )
        if self.steady_state_iterations < 5:
            raise BenchmarkRecordError(
                "Section 22.2 requires steady state to be the median of at least "
                f"5 iterations; got {self.steady_state_iterations}. One sample is "
                "not a measurement."
            )

    @classmethod
    def create(cls, **values: Any) -> BenchmarkRecord:
        """Build a record, rejecting a missing or unknown field.

        Raises
        ------
        BenchmarkRecordError
            If any Section 23 field is absent, or any field is not declared by
            the schema.
        """
        _reject_missing(cls, values)
        return cls(**values)

    def to_json_safe(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable view of this record."""
        return {
            declared_field.name: _json_safe(getattr(self, declared_field.name))
            for declared_field in fields(self)
        }


@dataclass(frozen=True, slots=True)
class RetracingRecord:
    """Per-step recompilation behavior when the kernel's source axis changes.

    Tier 6H's independent acceptance found that both solvers mask sources and
    pixels by ``above_horizon`` per time step, so the one compiled kernel's
    source axis can change size *within* a run -- which Section 13.6's
    "shape-stable within a run" wording does not cover and which Section 22.2's
    repeated-identical-call timing loop cannot observe. This record measures the
    consequence directly: how many distinct shapes a step sequence presents, what
    the first call at each shape costs against a later call at the same shape,
    and how much of the run's wall clock that difference accounts for.
    """

    schema_version: str
    recorded_at_utc: str
    backend_actual: str
    compilation_used: bool
    source_counts: tuple[int, ...]
    distinct_source_counts: int
    steps: int
    first_call_seconds_by_source_count: dict[str, float]
    repeat_call_seconds_by_source_count: dict[str, float]
    max_first_to_repeat_ratio: float
    total_seconds: float
    retrace_overhead_seconds: float
    notes: str

    def __post_init__(self) -> None:
        _reject_none(self, frozenset())

    @classmethod
    def create(cls, **values: Any) -> RetracingRecord:
        """Build a retracing record, rejecting a missing or unknown field."""
        _reject_missing(cls, values)
        return cls(**values)

    def to_json_safe(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable view of this record."""
        return {
            declared_field.name: _json_safe(getattr(self, declared_field.name))
            for declared_field in fields(self)
        }


@dataclass(frozen=True, slots=True)
class MemoryScalingRecord:
    """The compiled kernel's ``(B, S, 2, 2)`` working set against ``B`` and ``S``.

    ``core/contraction.py`` materializes two ``(B, S, 2, 2)`` antenna-Jones
    batches per ``(time, frequency)`` step, so its peak working set is
    ``O(baselines x sources)`` where the pre-Tier-6H per-baseline Python loop was
    ``O(sources)``. Every shipped configuration and every Section 13.4 workload
    stays at a handful of baselines, so no correctness test can see this. This
    record measures the slope so the hazard is bounded by evidence.
    """

    schema_version: str
    recorded_at_utc: str
    backend_actual: str
    n_baselines: int
    n_sources: int
    pair_count: int
    peak_host_bytes: int
    bytes_per_pair: float
    notes: str

    def __post_init__(self) -> None:
        _reject_none(self, frozenset())

    @classmethod
    def create(cls, **values: Any) -> MemoryScalingRecord:
        """Build a memory-scaling record, rejecting a missing or unknown field."""
        _reject_missing(cls, values)
        return cls(**values)

    def to_json_safe(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable view of this record."""
        return {
            declared_field.name: _json_safe(getattr(self, declared_field.name))
            for declared_field in fields(self)
        }


@dataclass(frozen=True, slots=True)
class BenchmarkDocument:
    """Everything one benchmark run measured, ready to be written as JSON."""

    records: tuple[BenchmarkRecord, ...]
    retracing: tuple[RetracingRecord, ...]
    memory_scaling: tuple[MemoryScalingRecord, ...]

    def to_json_safe(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable view of this document."""
        return {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "records": [record.to_json_safe() for record in self.records],
            "retracing": [record.to_json_safe() for record in self.retracing],
            "memory_scaling": [record.to_json_safe() for record in self.memory_scaling],
        }


def _json_safe(value: Any) -> Any:
    """Convert one record field into something ``json.dumps`` accepts."""
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): _json_safe(item) for key, item in mapping.items()}
    if isinstance(value, (str, bytes)):
        return value.decode() if isinstance(value, bytes) else value
    if isinstance(value, Sequence):
        sequence = cast(Sequence[object], value)
        return [_json_safe(item) for item in sequence]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return str(value)


def write_benchmark_document(
    document: BenchmarkDocument,
    *,
    directory: Path,
    filename: str,
) -> Path:
    """Write ``document`` to ``directory / filename`` and return the path.

    Section 22.1 puts benchmark output under ``output/benchmarks/`` as
    ``<UTC timestamp>-<host tag>.json``; the caller supplies both, because the
    harness -- not this module -- owns the clock and the host description.
    """
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    _ = destination.write_text(
        json.dumps(document.to_json_safe(), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return destination


def records_are_complete(records: Iterable[BenchmarkRecord]) -> bool:
    """Return ``True`` when every record in ``records`` is a full record.

    Construction already guarantees this; the function exists so a caller can
    assert the property against a collection it did not build itself.
    """
    for record in records:
        _reject_missing(BenchmarkRecord, record.to_json_safe())
    return True


# ---------------------------------------------------------------------------
# PERF-001 evidence schema
# ---------------------------------------------------------------------------
#
# These types intentionally do not extend the Tier 6 v1 types above.  Those
# schemas and their serializer are retained evidence and therefore frozen.

PERF001_SCHEMA_VERSION = "radiosim.benchmark.perf001.v1"
PERF001_PROVENANCE_SCHEMA_VERSION = "radiosim.benchmark.perf001.provenance.v1"
PERF001_WORKLOAD_SCHEMA_VERSION = "radiosim.benchmark.perf001.workload.v2"
PERF001_MEMORY_SCALING_SCHEMA_VERSION = "radiosim.benchmark.perf001.memory_scaling.v2"
PERF001_SOLVER_MEMORY_SCHEMA_VERSION = "radiosim.benchmark.perf001.solver_memory.v1"
PERF001_RETRACING_SCHEMA_VERSION = "radiosim.benchmark.perf001.retracing.v2"
PERF001_BACKEND_RESOLUTION_SCHEMA_VERSION = (
    "radiosim.benchmark.perf001.backend_resolution.v1"
)
PERF001_TARGET_KERNEL_PAIRS = 131072

_LOWER_HEX_40 = re.compile(r"[0-9a-f]{40}\Z")
_LOWER_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
_MEMORY_SCOPE = "contraction_wrapper_python_heap_including_output_assembly"
_SOLVER_MEMORY_SCOPE = (
    "direct_solver_step_python_heap_including_input_construction_and_output_assembly"
)
_UNCHUNKED_REFERENCE = "unchunked_reference"
_CHUNKED_PRODUCTION = "chunked_production"
_UNBUCKETED_REFERENCE = "unbucketed_reference"
_BUCKETED_PRODUCTION = "bucketed_production"
_IDENTITY_BUCKET_POLICY = "identity_reference_v1"
_PRODUCTION_BUCKET_POLICY = "pow2_compiled_v1"
_REQUIRED_BACKEND_RESOLUTION_OPERATIONS = frozenset(
    {
        ("get_backend_auto", "auto"),
        ("get_device_resources_default", "default"),
        ("simulator_setup_auto", "auto"),
    }
)
_REQUIRED_SOLVER_MEMORY_PATHS = frozenset({"point", "healpix"})
_REQUIRED_RETRACING_PATHS = frozenset({"synthetic_wrapper", "point", "healpix"})


def _perf001_error(field_name: str, detail: str) -> BenchmarkRecordError:
    return BenchmarkRecordError(f"PERF-001 field {field_name}: {detail}")


def _require_string(value: object, *, field_name: str) -> None:
    if type(value) is not str or not value:
        raise _perf001_error(field_name, "must be a non-empty JSON string")


def _require_bool(value: object, *, field_name: str) -> None:
    if type(value) is not bool:
        raise _perf001_error(field_name, "must be a JSON boolean")


def _require_nonnegative_int(value: object, *, field_name: str) -> None:
    if type(value) is not int or value < 0:
        raise _perf001_error(
            field_name, "must be a nonnegative JSON integer (booleans are invalid)"
        )


def _require_positive_int(value: object, *, field_name: str) -> None:
    _require_nonnegative_int(value, field_name=field_name)
    integer = cast(int, value)
    if integer == 0:
        raise _perf001_error(field_name, "must be a positive JSON integer")


def _require_nonnegative_number(value: object, *, field_name: str) -> None:
    if type(value) not in (int, float):
        raise _perf001_error(field_name, "must be a finite nonnegative JSON number")
    number = float(cast(int | float, value))
    if not math.isfinite(number) or number < 0.0:
        raise _perf001_error(field_name, "must be a finite nonnegative JSON number")


def _require_positive_number(value: object, *, field_name: str) -> None:
    _require_nonnegative_number(value, field_name=field_name)
    number = float(cast(int | float, value))
    if number == 0.0:
        raise _perf001_error(field_name, "must be a positive JSON number")


def _require_digest(value: object, *, field_name: str, git_sha: bool = False) -> None:
    _require_string(value, field_name=field_name)
    text = cast(str, value)
    pattern = _LOWER_HEX_40 if git_sha else _LOWER_HEX_64
    if pattern.fullmatch(text) is None:
        length = 40 if git_sha else 64
        raise _perf001_error(field_name, f"must be a lowercase {length}-hex digest")
    if git_sha and text == "0" * 40:
        raise _perf001_error(field_name, "must identify a real, known source commit")


def _require_string_tuple(value: object, *, field_name: str) -> None:
    if type(value) is not tuple:
        raise _perf001_error(field_name, "must be an ordered tuple/JSON array")
    tuple_value = cast(tuple[object, ...], value)
    for index, item in enumerate(tuple_value):
        _require_string(item, field_name=f"{field_name}[{index}]")


def _require_int_tuple(
    value: object, *, field_name: str, allow_empty: bool = False
) -> None:
    if type(value) is not tuple or (not value and not allow_empty):
        qualifier = "possibly empty " if allow_empty else "non-empty "
        raise _perf001_error(
            field_name, f"must be a {qualifier}ordered tuple/JSON array"
        )
    tuple_value = cast(tuple[object, ...], value)
    for index, item in enumerate(tuple_value):
        _require_nonnegative_int(item, field_name=f"{field_name}[{index}]")


def _require_number_tuple(value: object, *, field_name: str) -> None:
    if type(value) is not tuple or not value:
        raise _perf001_error(field_name, "must be a non-empty ordered tuple/JSON array")
    tuple_value = cast(tuple[object, ...], value)
    for index, item in enumerate(tuple_value):
        _require_nonnegative_number(item, field_name=f"{field_name}[{index}]")


def _require_string_mapping(value: object, *, field_name: str) -> None:
    if type(value) is not dict:
        raise _perf001_error(field_name, "must be a JSON object")
    mapping = cast(dict[object, object], value)
    for key, item in mapping.items():
        _require_string(key, field_name=f"{field_name}.key")
        _require_string(item, field_name=f"{field_name}[{key!r}]")


def _require_record(value: object, cls: type, *, field_name: str) -> None:
    if type(value) is not cls:
        raise _perf001_error(field_name, f"must be an exact {cls.__name__}")


def _require_schema(value: object, expected: str, *, field_name: str) -> None:
    if value != expected or type(value) is not str:
        raise _perf001_error(field_name, f"must equal {expected!r}")


def _require_fields_non_none(instance: Any, nullable: frozenset[str]) -> None:
    for declared_field in fields(instance):
        if (
            declared_field.name not in nullable
            and getattr(instance, declared_field.name) is None
        ):
            raise _perf001_error(
                declared_field.name,
                "is not nullable; an unmeasured value is an evidence gap",
            )


def _perf001_json_safe(value: object) -> Any:
    if isinstance(value, _Perf001Record):
        return value.to_json_safe()
    if type(value) is dict:
        mapping = cast(dict[object, object], value)
        serialized: dict[str, Any] = {}
        for key, item in mapping.items():
            if type(key) is not str or not key:
                raise BenchmarkRecordError(
                    "PERF-001 JSON object keys must be non-empty strings"
                )
            serialized[key] = _perf001_json_safe(item)
        return serialized
    if type(value) is tuple:
        sequence = cast(tuple[object, ...], value)
        return [_perf001_json_safe(item) for item in sequence]
    if type(value) is float and not math.isfinite(value):
        raise BenchmarkRecordError("PERF-001 JSON numbers must be finite")
    if type(value) in (str, bool, int, float) or value is None:
        return value
    raise BenchmarkRecordError(
        f"{type(value).__name__} is not a strict JSON value in PERF-001 evidence"
    )


class _Perf001Record:
    """Shared exact-field construction and recursive JSON serialization."""

    __slots__ = ()

    @classmethod
    def create(cls, **values: Any) -> Self:
        _reject_missing(cls, values)
        return cls(**values)

    def to_json_safe(self) -> dict[str, Any]:
        return {
            declared_field.name: _perf001_json_safe(getattr(self, declared_field.name))
            for declared_field in fields(cast(Any, self))
        }


@dataclass(frozen=True, slots=True)
class Perf001Provenance(_Perf001Record):
    """Clean-source and runtime identity shared by every PERF-001 row."""

    schema_version: str
    recorded_at_utc: str
    radiosim_version: str
    git_sha: str
    working_tree_clean: bool
    platform: str
    machine: str
    cpu_model: str
    cpu_count_logical: int
    python_version: str
    numpy_version: str
    jax_version: str
    jaxlib_version: str
    dask_version: str
    pixi_environment: str
    pixi_lock_sha256: str

    def __post_init__(self) -> None:
        _require_fields_non_none(self, frozenset())
        _require_schema(
            self.schema_version,
            PERF001_PROVENANCE_SCHEMA_VERSION,
            field_name="schema_version",
        )
        for field_name in (
            "recorded_at_utc",
            "radiosim_version",
            "platform",
            "machine",
            "cpu_model",
            "python_version",
            "numpy_version",
            "jax_version",
            "jaxlib_version",
            "dask_version",
            "pixi_environment",
        ):
            value = getattr(self, field_name)
            _require_string(value, field_name=field_name)
            if value == "unknown":
                raise _perf001_error(field_name, "must not be unknown")
        if not (
            self.recorded_at_utc.endswith("Z")
            or self.recorded_at_utc.endswith("+00:00")
        ):
            raise _perf001_error("recorded_at_utc", "must carry an explicit UTC offset")
        try:
            recorded_at = datetime.fromisoformat(
                self.recorded_at_utc.removesuffix("Z")
                + ("+00:00" if self.recorded_at_utc.endswith("Z") else "")
            )
        except ValueError as error:
            raise _perf001_error(
                "recorded_at_utc", "must be an ISO-8601 timestamp"
            ) from error
        if recorded_at.utcoffset() != timedelta(0):
            raise _perf001_error("recorded_at_utc", "must be normalized to UTC")
        _require_digest(self.git_sha, field_name="git_sha", git_sha=True)
        _require_bool(self.working_tree_clean, field_name="working_tree_clean")
        if self.working_tree_clean is not True:
            raise _perf001_error(
                "working_tree_clean", "dirty source cannot produce evidence"
            )
        _require_positive_int(self.cpu_count_logical, field_name="cpu_count_logical")
        _require_digest(self.pixi_lock_sha256, field_name="pixi_lock_sha256")


@dataclass(frozen=True, slots=True)
class MeasurementContext(_Perf001Record):
    """Backend, precision, policy, and logical-input identity for one row."""

    backend_requested: str
    backend_actual: str
    backend_version: str
    device_kind: str
    compilation_used: bool
    precision_preset: str
    precision_default: str
    precision_accumulation: str
    precision_output: str
    result_dtype: str
    policy_id: str
    input_identity_sha256: str
    measurement_limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_fields_non_none(self, frozenset())
        for field_name in (
            "backend_requested",
            "backend_actual",
            "backend_version",
            "device_kind",
            "precision_preset",
            "precision_default",
            "precision_accumulation",
            "precision_output",
            "result_dtype",
            "policy_id",
        ):
            _require_string(getattr(self, field_name), field_name=field_name)
        _require_bool(self.compilation_used, field_name="compilation_used")
        _require_digest(self.input_identity_sha256, field_name="input_identity_sha256")
        _require_string_tuple(
            self.measurement_limitations, field_name="measurement_limitations"
        )


def _validate_common_row(
    *,
    schema_version: object,
    expected_schema: str,
    provenance: object,
    context: object,
    comparison_id: object,
    implementation_state: object,
) -> None:
    _require_schema(schema_version, expected_schema, field_name="schema_version")
    _require_record(provenance, Perf001Provenance, field_name="provenance")
    _require_record(context, MeasurementContext, field_name="context")
    _require_string(comparison_id, field_name="comparison_id")
    _require_string(implementation_state, field_name="implementation_state")
    _validate_context_provenance(
        cast(Perf001Provenance, provenance), cast(MeasurementContext, context)
    )


def _validate_context_provenance(
    provenance: Perf001Provenance, context: MeasurementContext
) -> None:
    expected_version: str | None = None
    if context.backend_actual.startswith("jax-"):
        expected_version = provenance.jax_version
    elif context.backend_actual.startswith("numpy-"):
        expected_version = provenance.numpy_version
    elif context.backend_actual.startswith("dask-"):
        expected_version = provenance.dask_version
    if expected_version is not None and context.backend_version != expected_version:
        raise _perf001_error(
            "context.backend_version",
            "must match the corresponding version in provenance",
        )


@dataclass(frozen=True, slots=True)
class MemoryScalingRecordV2(_Perf001Record):
    """Matched wrapper-memory row for the P-a contraction policy."""

    schema_version: str
    provenance: Perf001Provenance
    context: MeasurementContext
    comparison_id: str
    implementation_state: str
    measurement_scope: str
    allocator: str
    includes_backend_native_allocations: bool
    inputs_preallocated: bool
    includes_solver_input_construction: bool
    includes_output_reassembly: bool
    logical_n_baselines: int
    logical_n_sources: int
    logical_pair_count: int
    kernel_n_sources: int
    target_kernel_pairs: int | None
    kernel_baseline_chunks: tuple[int, ...]
    kernel_pair_counts: tuple[int, ...]
    max_kernel_pair_count: int
    synthetic_input_bytes_excluded: int
    peak_host_bytes: int
    notes: str

    def __post_init__(self) -> None:
        _require_fields_non_none(self, frozenset({"target_kernel_pairs"}))
        _validate_common_row(
            schema_version=self.schema_version,
            expected_schema=PERF001_MEMORY_SCALING_SCHEMA_VERSION,
            provenance=self.provenance,
            context=self.context,
            comparison_id=self.comparison_id,
            implementation_state=self.implementation_state,
        )
        if self.implementation_state not in {
            _UNCHUNKED_REFERENCE,
            _CHUNKED_PRODUCTION,
        }:
            raise _perf001_error(
                "implementation_state",
                "must be unchunked_reference or chunked_production",
            )
        expected_policy_id = {
            _UNCHUNKED_REFERENCE: "unbounded_reference_v1",
            _CHUNKED_PRODUCTION: "target_kernel_pairs_131072_v1",
        }[self.implementation_state]
        if self.context.policy_id != expected_policy_id:
            raise _perf001_error(
                "context.policy_id",
                f"{self.implementation_state} requires {expected_policy_id!r}",
            )
        if self.measurement_scope != _MEMORY_SCOPE:
            raise _perf001_error("measurement_scope", f"must equal {_MEMORY_SCOPE!r}")
        _require_string(self.allocator, field_name="allocator")
        for field_name in (
            "includes_backend_native_allocations",
            "inputs_preallocated",
            "includes_solver_input_construction",
            "includes_output_reassembly",
        ):
            _require_bool(getattr(self, field_name), field_name=field_name)
        expected_scope_flags = {
            "allocator": "python_heap_tracemalloc",
            "includes_backend_native_allocations": False,
            "inputs_preallocated": True,
            "includes_solver_input_construction": False,
            "includes_output_reassembly": True,
        }
        for field_name, expected in expected_scope_flags.items():
            if getattr(self, field_name) != expected:
                raise _perf001_error(
                    field_name,
                    f"scope {_MEMORY_SCOPE!r} requires {expected!r}",
                )
        _require_nonnegative_int(
            self.logical_n_baselines, field_name="logical_n_baselines"
        )
        _require_nonnegative_int(self.logical_n_sources, field_name="logical_n_sources")
        _require_nonnegative_int(
            self.logical_pair_count, field_name="logical_pair_count"
        )
        _require_nonnegative_int(self.kernel_n_sources, field_name="kernel_n_sources")
        logical_n_baselines = self.logical_n_baselines
        logical_n_sources = self.logical_n_sources
        logical_pair_count = self.logical_pair_count
        kernel_n_sources = self.kernel_n_sources
        if logical_pair_count != logical_n_baselines * logical_n_sources:
            raise _perf001_error(
                "logical_pair_count",
                "must equal logical_n_baselines * logical_n_sources",
            )
        if kernel_n_sources != logical_n_sources:
            raise _perf001_error(
                "kernel_n_sources",
                "P-a matched rows must keep the logical source axis unchanged",
            )
        _require_int_tuple(
            self.kernel_baseline_chunks, field_name="kernel_baseline_chunks"
        )
        _require_int_tuple(self.kernel_pair_counts, field_name="kernel_pair_counts")
        chunks = self.kernel_baseline_chunks
        pair_counts = self.kernel_pair_counts
        if sum(chunks) != logical_n_baselines:
            raise _perf001_error(
                "kernel_baseline_chunks",
                "chunk sizes must sum to logical_n_baselines",
            )
        if logical_n_baselines > 0 and any(chunk == 0 for chunk in chunks):
            raise _perf001_error(
                "kernel_baseline_chunks", "nonempty inputs require positive chunks"
            )
        expected_pairs = tuple(chunk * kernel_n_sources for chunk in chunks)
        if pair_counts != expected_pairs:
            raise _perf001_error(
                "kernel_pair_counts",
                "each value must equal its baseline chunk * kernel_n_sources",
            )
        _require_nonnegative_int(
            self.max_kernel_pair_count, field_name="max_kernel_pair_count"
        )
        max_kernel_pair_count = self.max_kernel_pair_count
        if max_kernel_pair_count != max(pair_counts):
            raise _perf001_error(
                "max_kernel_pair_count", "must equal max(kernel_pair_counts)"
            )
        if self.implementation_state == _UNCHUNKED_REFERENCE:
            if self.target_kernel_pairs is not None:
                raise _perf001_error(
                    "target_kernel_pairs",
                    "must be null only for unchunked_reference",
                )
            if chunks != (logical_n_baselines,):
                raise _perf001_error(
                    "kernel_baseline_chunks",
                    "unchunked_reference must make one logical-baseline call",
                )
        else:
            if self.target_kernel_pairs != PERF001_TARGET_KERNEL_PAIRS:
                raise _perf001_error(
                    "target_kernel_pairs",
                    f"chunked_production must use {PERF001_TARGET_KERNEL_PAIRS}",
                )
            if logical_n_baselines == 0 or kernel_n_sources == 0:
                expected_chunks = (logical_n_baselines,)
            else:
                chunk_size = max(
                    1,
                    min(
                        logical_n_baselines,
                        PERF001_TARGET_KERNEL_PAIRS // kernel_n_sources,
                    ),
                )
                expected_chunks = tuple(
                    min(chunk_size, logical_n_baselines - start)
                    for start in range(0, logical_n_baselines, chunk_size)
                )
            if chunks != expected_chunks:
                raise _perf001_error(
                    "kernel_baseline_chunks",
                    "must be the exact stable production chunk sequence",
                )
            limit = max(PERF001_TARGET_KERNEL_PAIRS, kernel_n_sources)
            if max_kernel_pair_count > limit:
                raise _perf001_error(
                    "max_kernel_pair_count",
                    f"must be at most max({PERF001_TARGET_KERNEL_PAIRS}, S)",
                )
        for field_name in (
            "synthetic_input_bytes_excluded",
            "peak_host_bytes",
        ):
            _require_nonnegative_int(getattr(self, field_name), field_name=field_name)
        _require_string(self.notes, field_name="notes")


def _power_of_two_bucket(count: int) -> int:
    return 0 if count == 0 else 1 << (count - 1).bit_length()


@dataclass(frozen=True, slots=True)
class SolverMemoryRecord(_Perf001Record):
    """Matched end-to-end solver-step host-memory row for source bucketing."""

    schema_version: str
    provenance: Perf001Provenance
    context: MeasurementContext
    comparison_id: str
    implementation_state: str
    measurement_scope: str
    allocator: str
    includes_backend_native_allocations: bool
    includes_simulator_setup: bool
    includes_solver_input_construction: bool
    includes_output_assembly: bool
    solver: str
    sky_representation: str
    logical_n_baselines: int
    logical_source_counts: tuple[int, ...]
    kernel_source_counts: tuple[int, ...]
    n_times: int
    n_frequencies: int
    target_kernel_pairs: int
    bucket_policy: str
    peak_host_bytes: int
    notes: str

    def __post_init__(self) -> None:
        _require_fields_non_none(self, frozenset())
        _validate_common_row(
            schema_version=self.schema_version,
            expected_schema=PERF001_SOLVER_MEMORY_SCHEMA_VERSION,
            provenance=self.provenance,
            context=self.context,
            comparison_id=self.comparison_id,
            implementation_state=self.implementation_state,
        )
        if self.measurement_scope != _SOLVER_MEMORY_SCOPE:
            raise _perf001_error(
                "measurement_scope", f"must equal {_SOLVER_MEMORY_SCOPE!r}"
            )
        _require_string(self.allocator, field_name="allocator")
        for field_name in (
            "includes_backend_native_allocations",
            "includes_simulator_setup",
            "includes_solver_input_construction",
            "includes_output_assembly",
        ):
            _require_bool(getattr(self, field_name), field_name=field_name)
        expected_scope_flags = {
            "allocator": "python_heap_tracemalloc",
            "includes_backend_native_allocations": False,
            "includes_simulator_setup": False,
            "includes_solver_input_construction": True,
            "includes_output_assembly": True,
        }
        for field_name, expected in expected_scope_flags.items():
            if getattr(self, field_name) != expected:
                raise _perf001_error(
                    field_name,
                    f"direct solver-step scope requires {expected!r}",
                )
        _require_string(self.solver, field_name="solver")
        _require_string(self.sky_representation, field_name="sky_representation")
        expected_sky_representation = {
            "point": "point_sources",
            "healpix": "healpix",
        }.get(self.solver)
        if expected_sky_representation is None:
            raise _perf001_error(
                "solver", "must identify the point or HEALPix production solver"
            )
        if self.sky_representation != expected_sky_representation:
            raise _perf001_error(
                "sky_representation",
                f"solver={self.solver!r} requires {expected_sky_representation!r}",
            )
        if not self.context.backend_actual.startswith("jax-"):
            raise _perf001_error(
                "context.backend_actual",
                "source-bucket evidence requires a concrete JAX backend name",
            )
        if self.context.compilation_used is not True:
            raise _perf001_error(
                "context.compilation_used",
                "source-bucket evidence requires a compiling backend",
            )
        _require_nonnegative_int(
            self.logical_n_baselines, field_name="logical_n_baselines"
        )
        _require_int_tuple(
            self.logical_source_counts, field_name="logical_source_counts"
        )
        _require_int_tuple(self.kernel_source_counts, field_name="kernel_source_counts")
        logical_counts = self.logical_source_counts
        kernel_counts = self.kernel_source_counts
        if len(logical_counts) != len(kernel_counts):
            raise _perf001_error(
                "kernel_source_counts",
                "must have one entry per logical_source_counts entry",
            )
        _require_positive_int(self.n_times, field_name="n_times")
        _require_positive_int(self.n_frequencies, field_name="n_frequencies")
        n_times = self.n_times
        n_frequencies = self.n_frequencies
        if len(logical_counts) != n_times * n_frequencies:
            raise _perf001_error(
                "logical_source_counts",
                "must contain exactly n_times * n_frequencies entries",
            )
        if self.target_kernel_pairs != PERF001_TARGET_KERNEL_PAIRS:
            raise _perf001_error(
                "target_kernel_pairs",
                f"must equal the production target {PERF001_TARGET_KERNEL_PAIRS}",
            )
        expected_state = {
            _UNBUCKETED_REFERENCE: _IDENTITY_BUCKET_POLICY,
            _BUCKETED_PRODUCTION: _PRODUCTION_BUCKET_POLICY,
        }
        expected_policy = expected_state.get(self.implementation_state)
        if expected_policy is None:
            raise _perf001_error(
                "implementation_state",
                "must be unbucketed_reference or bucketed_production",
            )
        if self.bucket_policy != expected_policy:
            raise _perf001_error(
                "bucket_policy",
                f"{self.implementation_state} requires {expected_policy!r}",
            )
        if self.context.policy_id != self.bucket_policy:
            raise _perf001_error(
                "context.policy_id", "must equal the measured bucket_policy"
            )
        expected_counts = (
            logical_counts
            if self.implementation_state == _UNBUCKETED_REFERENCE
            else tuple(_power_of_two_bucket(count) for count in logical_counts)
        )
        if kernel_counts != expected_counts:
            raise _perf001_error(
                "kernel_source_counts",
                "must follow the row's identity or power-of-two bucket policy",
            )
        _require_nonnegative_int(self.peak_host_bytes, field_name="peak_host_bytes")
        _require_string(self.notes, field_name="notes")


_SIGNATURE_OPERANDS = (
    "jones_p",
    "jones_q",
    "coherency",
    "phase",
    "envelope",
    "stokes_i",
)


@dataclass(frozen=True, slots=True)
class ContractionSignatureObservation(_Perf001Record):
    """Complete six-operand compiled-leaf signature and its observed calls."""

    jones_p_shape: tuple[int, ...] | None
    jones_q_shape: tuple[int, ...] | None
    coherency_shape: tuple[int, ...] | None
    phase_shape: tuple[int, ...] | None
    envelope_shape: tuple[int, ...] | None
    stokes_i_shape: tuple[int, ...] | None
    jones_p_dtype: str | None
    jones_q_dtype: str | None
    coherency_dtype: str | None
    phase_dtype: str | None
    envelope_dtype: str | None
    stokes_i_dtype: str | None
    call_count: int
    first_call_seconds: float
    minimum_repeat_call_seconds: float

    def __post_init__(self) -> None:
        nullable = frozenset(
            f"{operand}_{suffix}"
            for operand in _SIGNATURE_OPERANDS
            for suffix in ("shape", "dtype")
        )
        _require_fields_non_none(self, nullable)
        for operand in _SIGNATURE_OPERANDS:
            shape = getattr(self, f"{operand}_shape")
            dtype = getattr(self, f"{operand}_dtype")
            if (shape is None) != (dtype is None):
                raise _perf001_error(
                    operand, "shape and dtype must be explicitly null as a pair"
                )
            if shape is not None:
                _require_int_tuple(
                    shape, field_name=f"{operand}_shape", allow_empty=True
                )
                _require_string(dtype, field_name=f"{operand}_dtype")
        for operand in ("jones_p", "jones_q", "phase"):
            if getattr(self, f"{operand}_shape") is None:
                raise _perf001_error(operand, "is a mandatory contraction-leaf operand")
        if (self.coherency_shape is None) == (self.stokes_i_shape is None):
            raise _perf001_error(
                "coherency/stokes_i",
                "exactly one signal operand must be present",
            )
        jones_p_shape = cast(tuple[int, ...], self.jones_p_shape)
        jones_q_shape = cast(tuple[int, ...], self.jones_q_shape)
        phase_shape = cast(tuple[int, ...], self.phase_shape)
        if len(jones_p_shape) != 4 or jones_p_shape[-2:] != (2, 2):
            raise _perf001_error(
                "jones_p_shape", "must have complete leaf shape (B, S, 2, 2)"
            )
        if jones_q_shape != jones_p_shape:
            raise _perf001_error("jones_q_shape", "must equal jones_p_shape")
        baseline_source_shape = jones_p_shape[:2]
        if phase_shape != baseline_source_shape:
            raise _perf001_error("phase_shape", "must equal the Jones (B, S) prefix")
        source_count = jones_p_shape[1]
        if self.coherency_shape is not None and self.coherency_shape != (
            source_count,
            2,
            2,
        ):
            raise _perf001_error("coherency_shape", "must equal (S, 2, 2)")
        if self.stokes_i_shape is not None and self.stokes_i_shape != (source_count,):
            raise _perf001_error("stokes_i_shape", "must equal (S,)")
        if self.envelope_shape not in (None, (), baseline_source_shape):
            raise _perf001_error(
                "envelope_shape", "must be absent, scalar, or have shape (B, S)"
            )
        _require_positive_int(self.call_count, field_name="call_count")
        if self.call_count < 2:
            raise _perf001_error(
                "call_count", "must include a first call and at least one repeat"
            )
        _require_nonnegative_number(
            self.first_call_seconds, field_name="first_call_seconds"
        )
        _require_positive_number(
            self.minimum_repeat_call_seconds,
            field_name="minimum_repeat_call_seconds",
        )


def _signature_key(observation: ContractionSignatureObservation) -> tuple[object, ...]:
    return tuple(
        getattr(observation, f"{operand}_{suffix}")
        for operand in _SIGNATURE_OPERANDS
        for suffix in ("shape", "dtype")
    )


@dataclass(frozen=True, slots=True)
class RetracingRecordV2(_Perf001Record):
    """Complete-leaf shape and retrace observations for one solver scope."""

    schema_version: str
    provenance: Perf001Provenance
    context: MeasurementContext
    comparison_id: str
    implementation_state: str
    measurement_scope: str
    solver: str
    sky_representation: str
    bucket_policy: str
    padding_location: str
    logical_source_counts: tuple[int, ...]
    kernel_source_counts: tuple[int, ...]
    distinct_logical_source_counts: int
    distinct_kernel_source_counts: int
    observed_signatures: tuple[ContractionSignatureObservation, ...]
    distinct_signature_count: int
    leaf_call_count: int
    scope_step_seconds: tuple[float, ...]
    scope_total_seconds: float
    max_first_to_repeat_ratio: float
    retrace_overhead_seconds: float
    notes: str

    def __post_init__(self) -> None:
        _require_fields_non_none(self, frozenset())
        _validate_common_row(
            schema_version=self.schema_version,
            expected_schema=PERF001_RETRACING_SCHEMA_VERSION,
            provenance=self.provenance,
            context=self.context,
            comparison_id=self.comparison_id,
            implementation_state=self.implementation_state,
        )
        for field_name in (
            "measurement_scope",
            "solver",
            "sky_representation",
            "bucket_policy",
            "padding_location",
            "notes",
        ):
            _require_string(getattr(self, field_name), field_name=field_name)
        expected_sky_representation = {
            "synthetic_wrapper": "synthetic_contraction",
            "point": "point_sources",
            "healpix": "healpix",
        }.get(self.solver)
        if expected_sky_representation is None:
            raise _perf001_error(
                "solver", "must identify the synthetic, point, or HEALPix path"
            )
        if self.sky_representation != expected_sky_representation:
            raise _perf001_error(
                "sky_representation",
                f"solver={self.solver!r} requires {expected_sky_representation!r}",
            )
        if not self.context.backend_actual.startswith("jax-"):
            raise _perf001_error(
                "context.backend_actual",
                "retracing evidence requires a concrete JAX backend name",
            )
        if self.context.compilation_used is not True:
            raise _perf001_error(
                "context.compilation_used",
                "retracing evidence requires a compiling backend",
            )
        expected_policy = {
            _UNBUCKETED_REFERENCE: _IDENTITY_BUCKET_POLICY,
            _BUCKETED_PRODUCTION: _PRODUCTION_BUCKET_POLICY,
        }.get(self.implementation_state)
        if expected_policy is None:
            raise _perf001_error(
                "implementation_state",
                "must be unbucketed_reference or bucketed_production",
            )
        if self.bucket_policy != expected_policy:
            raise _perf001_error(
                "bucket_policy",
                f"{self.implementation_state} requires {expected_policy!r}",
            )
        if self.context.policy_id != self.bucket_policy:
            raise _perf001_error(
                "context.policy_id", "must equal the measured bucket_policy"
            )
        expected_padding_location = (
            "none"
            if self.implementation_state == _UNBUCKETED_REFERENCE
            else "early_host"
        )
        if self.padding_location != expected_padding_location:
            raise _perf001_error(
                "padding_location",
                f"{self.implementation_state} requires {expected_padding_location!r}",
            )
        _require_int_tuple(
            self.logical_source_counts, field_name="logical_source_counts"
        )
        _require_int_tuple(self.kernel_source_counts, field_name="kernel_source_counts")
        logical_counts = self.logical_source_counts
        kernel_counts = self.kernel_source_counts
        if len(logical_counts) != len(kernel_counts):
            raise _perf001_error(
                "kernel_source_counts",
                "must have one entry per logical_source_counts entry",
            )
        expected_counts = (
            logical_counts
            if self.implementation_state == _UNBUCKETED_REFERENCE
            else tuple(_power_of_two_bucket(count) for count in logical_counts)
        )
        if kernel_counts != expected_counts:
            raise _perf001_error(
                "kernel_source_counts",
                "must follow the row's identity or power-of-two bucket policy",
            )
        _require_positive_int(
            self.distinct_logical_source_counts,
            field_name="distinct_logical_source_counts",
        )
        _require_positive_int(
            self.distinct_kernel_source_counts,
            field_name="distinct_kernel_source_counts",
        )
        distinct_logical = self.distinct_logical_source_counts
        distinct_kernel = self.distinct_kernel_source_counts
        if distinct_logical != len(set(logical_counts)):
            raise _perf001_error(
                "distinct_logical_source_counts",
                "must equal the number of distinct logical source counts",
            )
        if distinct_kernel != len(set(kernel_counts)):
            raise _perf001_error(
                "distinct_kernel_source_counts",
                "must equal the number of distinct kernel source counts",
            )
        if type(self.observed_signatures) is not tuple:
            raise _perf001_error(
                "observed_signatures", "must be an ordered tuple/JSON array"
            )
        for index, observation in enumerate(self.observed_signatures):
            _require_record(
                observation,
                ContractionSignatureObservation,
                field_name=f"observed_signatures[{index}]",
            )
        if len({_signature_key(item) for item in self.observed_signatures}) != len(
            self.observed_signatures
        ):
            raise _perf001_error(
                "observed_signatures", "must contain one row per distinct signature"
            )
        positive_kernel_counts = tuple(count for count in kernel_counts if count > 0)
        signature_source_counts = {
            cast(tuple[int, ...], observation.jones_p_shape)[1]
            for observation in self.observed_signatures
        }
        if signature_source_counts != set(positive_kernel_counts):
            raise _perf001_error(
                "observed_signatures",
                "leaf signatures must cover every positive kernel source count and "
                "must not represent zero-visible return steps",
            )
        for source_count in set(positive_kernel_counts):
            observed_calls = sum(
                observation.call_count
                for observation in self.observed_signatures
                if cast(tuple[int, ...], observation.jones_p_shape)[1] == source_count
            )
            required_calls = positive_kernel_counts.count(source_count)
            if observed_calls < required_calls:
                raise _perf001_error(
                    "observed_signatures",
                    f"S={source_count} has {required_calls} logical leaf steps but "
                    f"only {observed_calls} observed leaf calls",
                )
        if self.distinct_signature_count != len(self.observed_signatures):
            raise _perf001_error(
                "distinct_signature_count", "must equal len(observed_signatures)"
            )
        _require_nonnegative_int(
            self.distinct_signature_count, field_name="distinct_signature_count"
        )
        _require_nonnegative_int(self.leaf_call_count, field_name="leaf_call_count")
        leaf_call_count = self.leaf_call_count
        if leaf_call_count != sum(
            observation.call_count for observation in self.observed_signatures
        ):
            raise _perf001_error(
                "leaf_call_count", "must equal the sum of signature call_count values"
            )
        _require_number_tuple(self.scope_step_seconds, field_name="scope_step_seconds")
        step_seconds = self.scope_step_seconds
        if len(step_seconds) != len(logical_counts):
            raise _perf001_error(
                "scope_step_seconds",
                "must have one value per logical source-count step",
            )
        _require_nonnegative_number(
            self.scope_total_seconds, field_name="scope_total_seconds"
        )
        scope_total = self.scope_total_seconds
        if not math.isclose(
            scope_total, sum(step_seconds), rel_tol=1e-12, abs_tol=1e-15
        ):
            raise _perf001_error(
                "scope_total_seconds", "must equal sum(scope_step_seconds)"
            )
        measured_ratio = max(
            (
                observation.first_call_seconds / observation.minimum_repeat_call_seconds
                for observation in self.observed_signatures
            ),
            default=0.0,
        )
        _require_nonnegative_number(
            self.max_first_to_repeat_ratio,
            field_name="max_first_to_repeat_ratio",
        )
        maximum_ratio = self.max_first_to_repeat_ratio
        if not math.isclose(
            maximum_ratio, measured_ratio, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise _perf001_error(
                "max_first_to_repeat_ratio",
                "must be derived from the observed signatures",
            )
        _require_nonnegative_number(
            self.retrace_overhead_seconds, field_name="retrace_overhead_seconds"
        )
        overhead = self.retrace_overhead_seconds
        derived_overhead = sum(
            max(
                0.0,
                observation.first_call_seconds
                - observation.minimum_repeat_call_seconds,
            )
            for observation in self.observed_signatures
        )
        if not math.isclose(overhead, derived_overhead, rel_tol=1e-12, abs_tol=1e-15):
            raise _perf001_error(
                "retrace_overhead_seconds",
                "must equal the summed first-minus-repeat signature overhead",
            )
        if overhead > scope_total:
            raise _perf001_error(
                "retrace_overhead_seconds", "must not exceed scope_total_seconds"
            )


@dataclass(frozen=True, slots=True)
class BackendResolutionRecord(_Perf001Record):
    """Fresh-process backend-selection and import-boundary evidence."""

    schema_version: str
    provenance: Perf001Provenance
    context: MeasurementContext
    comparison_id: str
    implementation_state: str
    operation: str
    requested_backend: str
    resolved_backend: str
    discovery_policy: str
    fresh_process_samples: int
    cold_seconds: tuple[float, ...]
    minimum_seconds: float
    median_seconds: float
    maximum_seconds: float
    jax_distribution_installed: bool
    jax_in_sys_modules_before: bool
    jax_in_sys_modules_after: bool
    jaxlib_in_sys_modules_before: bool
    jaxlib_in_sys_modules_after: bool
    notes: str

    def __post_init__(self) -> None:
        _require_fields_non_none(self, frozenset())
        _validate_common_row(
            schema_version=self.schema_version,
            expected_schema=PERF001_BACKEND_RESOLUTION_SCHEMA_VERSION,
            provenance=self.provenance,
            context=self.context,
            comparison_id=self.comparison_id,
            implementation_state=self.implementation_state,
        )
        for field_name in (
            "operation",
            "requested_backend",
            "resolved_backend",
            "discovery_policy",
            "notes",
        ):
            _require_string(getattr(self, field_name), field_name=field_name)
        if self.context.backend_requested != self.requested_backend:
            raise _perf001_error(
                "context.backend_requested", "must equal requested_backend"
            )
        if self.context.backend_actual != self.resolved_backend:
            raise _perf001_error(
                "context.backend_actual", "must equal resolved_backend"
            )
        _require_positive_int(
            self.fresh_process_samples, field_name="fresh_process_samples"
        )
        _require_number_tuple(self.cold_seconds, field_name="cold_seconds")
        samples = self.fresh_process_samples
        cold_seconds = self.cold_seconds
        if len(cold_seconds) != samples:
            raise _perf001_error(
                "fresh_process_samples", "must equal len(cold_seconds)"
            )
        summaries = {
            "minimum_seconds": min(cold_seconds),
            "median_seconds": statistics.median(cold_seconds),
            "maximum_seconds": max(cold_seconds),
        }
        for field_name, expected in summaries.items():
            _require_nonnegative_number(
                getattr(self, field_name), field_name=field_name
            )
            measured = getattr(self, field_name)
            if not math.isclose(measured, expected, rel_tol=1e-12, abs_tol=1e-15):
                raise _perf001_error(
                    field_name, f"must be derived from cold_seconds ({expected})"
                )
        for field_name in (
            "jax_distribution_installed",
            "jax_in_sys_modules_before",
            "jax_in_sys_modules_after",
            "jaxlib_in_sys_modules_before",
            "jaxlib_in_sys_modules_after",
        ):
            _require_bool(getattr(self, field_name), field_name=field_name)


@dataclass(frozen=True, slots=True)
class AcceleratorFacts(_Perf001Record):
    """Exact accelerator, driver, JAX-device, and wheel provenance."""

    vendor: str
    model: str
    runtime: str
    driver_version: str
    compute_capability: str
    total_memory_bytes: int
    pci_bus_id: str
    device_uuid_sha256: str
    jax_device_id: int
    jax_device_kind: str
    visible_device_count: int
    wheel_versions: dict[str, str]
    allocator_environment: dict[str, str]

    def __post_init__(self) -> None:
        _require_fields_non_none(self, frozenset())
        for field_name in (
            "vendor",
            "model",
            "runtime",
            "driver_version",
            "compute_capability",
            "pci_bus_id",
            "jax_device_kind",
        ):
            value = getattr(self, field_name)
            _require_string(value, field_name=field_name)
            if value in {"unknown", "not-installed"}:
                raise _perf001_error(
                    field_name, "accelerator provenance must be measured"
                )
        _require_positive_int(self.total_memory_bytes, field_name="total_memory_bytes")
        _require_digest(self.device_uuid_sha256, field_name="device_uuid_sha256")
        _require_nonnegative_int(self.jax_device_id, field_name="jax_device_id")
        _require_positive_int(
            self.visible_device_count, field_name="visible_device_count"
        )
        _require_string_mapping(self.wheel_versions, field_name="wheel_versions")
        _require_string_mapping(
            self.allocator_environment, field_name="allocator_environment"
        )
        if self.vendor != "NVIDIA":
            raise _perf001_error("vendor", "the CUDA 13 path requires NVIDIA")
        if self.jax_device_kind != "gpu":
            raise _perf001_error("jax_device_kind", "must equal 'gpu'")
        if self.visible_device_count != 1:
            raise _perf001_error(
                "visible_device_count", "the strict preflight selects exactly one GPU"
            )
        expected_wheels = {
            "jax": "0.10.2",
            "jaxlib": "0.10.2",
            "jax-cuda13-plugin": "0.10.2",
            "jax-cuda13-pjrt": "0.10.2",
        }
        if self.wheel_versions != expected_wheels:
            raise _perf001_error(
                "wheel_versions",
                "must contain the exact locked CUDA 13 JAX 0.10.2 stack",
            )


def _require_json_mapping(value: object, *, field_name: str) -> None:
    if type(value) is not dict:
        raise _perf001_error(field_name, "must be a JSON object")
    mapping = cast(dict[object, object], value)
    for key, item in mapping.items():
        _require_string(key, field_name=f"{field_name}.key")
        try:
            _perf001_json_safe(item)
        except BenchmarkRecordError as error:
            raise _perf001_error(field_name, str(error)) from error


@dataclass(frozen=True, slots=True)
class DeviceMemoryMeasurement(_Perf001Record):
    """Sampled selected-device memory observations outside timed iterations."""

    method: str
    sampling_scope: str
    sample_interval_seconds: float
    sample_count: int
    total_bytes: int
    used_bytes_before: int
    free_bytes_before: int
    used_bytes_after_setup: int
    free_bytes_after_setup: int
    peak_observed_used_bytes: int
    used_bytes_after_transfer: int
    free_bytes_after_transfer: int
    raw_jax_memory_stats: dict[str, object] | None
    limitations: str

    def __post_init__(self) -> None:
        _require_fields_non_none(self, frozenset({"raw_jax_memory_stats"}))
        _require_string(self.method, field_name="method")
        _require_string(self.sampling_scope, field_name="sampling_scope")
        _require_positive_number(
            self.sample_interval_seconds, field_name="sample_interval_seconds"
        )
        _require_positive_int(self.sample_count, field_name="sample_count")
        _require_positive_int(self.total_bytes, field_name="total_bytes")
        total_bytes = self.total_bytes
        for field_name in (
            "used_bytes_before",
            "free_bytes_before",
            "used_bytes_after_setup",
            "free_bytes_after_setup",
            "peak_observed_used_bytes",
            "used_bytes_after_transfer",
            "free_bytes_after_transfer",
        ):
            _require_nonnegative_int(getattr(self, field_name), field_name=field_name)
            value = getattr(self, field_name)
            if value > total_bytes:
                raise _perf001_error(field_name, "must not exceed total_bytes")
        if self.raw_jax_memory_stats is not None:
            _require_json_mapping(
                self.raw_jax_memory_stats, field_name="raw_jax_memory_stats"
            )
        if self.peak_observed_used_bytes < max(
            self.used_bytes_before,
            self.used_bytes_after_setup,
            self.used_bytes_after_transfer,
        ):
            raise _perf001_error(
                "peak_observed_used_bytes",
                "must be at least every retained used-memory snapshot",
            )
        _require_string(self.limitations, field_name="limitations")


@dataclass(frozen=True, slots=True)
class WorkloadBenchmarkRecordV2(_Perf001Record):
    """Complete CPU or real-accelerator workload measurement."""

    schema_version: str
    provenance: Perf001Provenance
    context: MeasurementContext
    accelerator: AcceleratorFacts | None
    device_memory: DeviceMemoryMeasurement | None
    workload: str
    n_antennas: int
    n_baselines: int
    n_point_sources: int
    n_healpix_pixels: int
    n_times: int
    n_frequencies: int
    sky_representation: str
    solver_workers: int
    loader_max_workers: int
    setup_seconds: float
    compile_seconds: float
    steady_state_median_seconds: float
    steady_state_min_seconds: float
    steady_state_max_seconds: float
    steady_state_iterations: int
    host_transfer_seconds: float
    peak_host_bytes: int
    host_memory_method: str
    reference_backend: str
    max_absolute_deviation: float
    max_relative_deviation: float
    tolerance_rtol: float
    tolerance_atol: float
    within_tolerance: bool
    unmeasured: tuple[str, ...]
    notes: str

    def __post_init__(self) -> None:
        _require_fields_non_none(self, frozenset({"accelerator", "device_memory"}))
        _require_schema(
            self.schema_version,
            PERF001_WORKLOAD_SCHEMA_VERSION,
            field_name="schema_version",
        )
        _require_record(self.provenance, Perf001Provenance, field_name="provenance")
        _require_record(self.context, MeasurementContext, field_name="context")
        _validate_context_provenance(self.provenance, self.context)
        _require_string(self.workload, field_name="workload")
        for field_name in (
            "n_antennas",
            "n_baselines",
            "n_point_sources",
            "n_healpix_pixels",
            "solver_workers",
            "loader_max_workers",
        ):
            _require_nonnegative_int(getattr(self, field_name), field_name=field_name)
        _require_positive_int(self.n_times, field_name="n_times")
        _require_positive_int(self.n_frequencies, field_name="n_frequencies")
        _require_string(self.sky_representation, field_name="sky_representation")
        for field_name in (
            "setup_seconds",
            "compile_seconds",
            "steady_state_median_seconds",
            "steady_state_min_seconds",
            "steady_state_max_seconds",
            "host_transfer_seconds",
            "max_absolute_deviation",
            "max_relative_deviation",
            "tolerance_rtol",
            "tolerance_atol",
        ):
            _require_nonnegative_number(
                getattr(self, field_name), field_name=field_name
            )
        if not (
            self.steady_state_min_seconds
            <= self.steady_state_median_seconds
            <= self.steady_state_max_seconds
        ):
            raise _perf001_error(
                "steady_state_median_seconds",
                "must lie between the measured minimum and maximum",
            )
        _require_positive_int(
            self.steady_state_iterations, field_name="steady_state_iterations"
        )
        iterations = self.steady_state_iterations
        if iterations < 5:
            raise _perf001_error(
                "steady_state_iterations", "must contain at least five samples"
            )
        _require_nonnegative_int(self.peak_host_bytes, field_name="peak_host_bytes")
        _require_string(self.host_memory_method, field_name="host_memory_method")
        if self.reference_backend != "numpy":
            raise _perf001_error("reference_backend", "PERF-001 correctness uses NumPy")
        _require_bool(self.within_tolerance, field_name="within_tolerance")
        _require_string_tuple(self.unmeasured, field_name="unmeasured")
        unmeasured = self.unmeasured
        _require_string(self.notes, field_name="notes")
        if self.context.device_kind == "cpu":
            if self.accelerator is not None or self.device_memory is not None:
                raise _perf001_error(
                    "accelerator/device_memory",
                    "CPU rows require both values to be null",
                )
            if "gpu" not in unmeasured:
                raise _perf001_error(
                    "unmeasured", "CPU rows must state that GPU was unmeasured"
                )
        elif self.context.device_kind == "gpu":
            _require_record(
                self.accelerator, AcceleratorFacts, field_name="accelerator"
            )
            _require_record(
                self.device_memory,
                DeviceMemoryMeasurement,
                field_name="device_memory",
            )
            accelerator = cast(AcceleratorFacts, self.accelerator)
            device_memory = cast(DeviceMemoryMeasurement, self.device_memory)
            if accelerator.jax_device_kind != self.context.device_kind:
                raise _perf001_error(
                    "accelerator.jax_device_kind", "must equal context.device_kind"
                )
            if self.context.backend_requested != "gpu":
                raise _perf001_error(
                    "context.backend_requested",
                    "GPU rows require an explicit 'gpu' request",
                )
            if not self.context.backend_actual.startswith("jax-gpu-"):
                raise _perf001_error(
                    "context.backend_actual",
                    "GPU rows require a concrete JAX GPU runtime backend name",
                )
            if self.context.compilation_used is not True:
                raise _perf001_error(
                    "context.compilation_used", "GPU rows require compiled execution"
                )
            if self.context.backend_version != self.provenance.jax_version:
                raise _perf001_error(
                    "context.backend_version",
                    "must equal provenance.jax_version for GPU rows",
                )
            if accelerator.total_memory_bytes != device_memory.total_bytes:
                raise _perf001_error(
                    "device_memory.total_bytes",
                    "must equal accelerator.total_memory_bytes",
                )
            if "gpu" in unmeasured:
                raise _perf001_error(
                    "unmeasured", "a GPU row may not list GPU as unmeasured"
                )
        else:
            raise _perf001_error(
                "context.device_kind", "workload rows must measure CPU or GPU"
            )


_Perf001RowT = TypeVar("_Perf001RowT", bound=_Perf001Record)


def _require_row_tuple(
    value: object, cls: type[_Perf001RowT], *, field_name: str
) -> tuple[_Perf001RowT, ...]:
    if type(value) is not tuple or not value:
        raise _perf001_error(field_name, "must be a non-empty ordered tuple/JSON array")
    tuple_value = cast(tuple[object, ...], value)
    for index, row in enumerate(tuple_value):
        _require_record(row, cls, field_name=f"{field_name}[{index}]")
    return cast(tuple[_Perf001RowT, ...], value)


_PairedRow = MemoryScalingRecordV2 | SolverMemoryRecord | RetracingRecordV2


def _pair_rows(
    rows: tuple[_PairedRow, ...],
    *,
    field_name: str,
    expected_states: frozenset[str],
) -> dict[str, tuple[_PairedRow, _PairedRow]]:
    grouped: dict[str, list[_PairedRow]] = {}
    for row in rows:
        comparison_id = row.comparison_id
        grouped.setdefault(comparison_id, []).append(row)
    pairs: dict[str, tuple[_PairedRow, _PairedRow]] = {}
    for comparison_id, group in grouped.items():
        states = {row.implementation_state for row in group}
        if len(group) != 2 or states != set(expected_states):
            raise BenchmarkRecordError(
                f"{field_name} comparison_id={comparison_id!r} must form a pair "
                f"with states {sorted(expected_states)}"
            )
        by_state = {row.implementation_state: row for row in group}
        first_state, second_state = sorted(expected_states)
        pairs[comparison_id] = (by_state[first_state], by_state[second_state])
    return pairs


def _require_pair_identity(
    pair: tuple[_PairedRow, _PairedRow], *, field_name: str
) -> None:
    first, second = pair
    first_context = first.context
    second_context = second.context
    if first_context.input_identity_sha256 != second_context.input_identity_sha256:
        raise BenchmarkRecordError(
            f"{field_name} pair input_identity_sha256 values must match"
        )
    matched_context_fields = (
        "backend_requested",
        "backend_actual",
        "backend_version",
        "device_kind",
        "compilation_used",
        "precision_preset",
        "precision_default",
        "precision_accumulation",
        "precision_output",
        "result_dtype",
        "measurement_limitations",
    )
    for context_field in matched_context_fields:
        if getattr(first_context, context_field) != getattr(
            second_context, context_field
        ):
            raise BenchmarkRecordError(
                f"{field_name} pair context field {context_field} must match"
            )


@dataclass(frozen=True, slots=True)
class Perf001EvidenceDocument(_Perf001Record):
    """Strict PERF-001 evidence rows from one clean, homogeneous source."""

    schema_version: str
    workload_benchmarks: tuple[WorkloadBenchmarkRecordV2, ...]
    memory_scaling: tuple[MemoryScalingRecordV2, ...]
    solver_memory: tuple[SolverMemoryRecord, ...]
    retracing: tuple[RetracingRecordV2, ...]
    backend_resolution: tuple[BackendResolutionRecord, ...]

    def __post_init__(self) -> None:
        _require_fields_non_none(self, frozenset())
        _require_schema(
            self.schema_version, PERF001_SCHEMA_VERSION, field_name="schema_version"
        )
        workload_rows = _require_row_tuple(
            self.workload_benchmarks,
            WorkloadBenchmarkRecordV2,
            field_name="workload_benchmarks",
        )
        memory_rows = _require_row_tuple(
            self.memory_scaling,
            MemoryScalingRecordV2,
            field_name="memory_scaling",
        )
        solver_rows = _require_row_tuple(
            self.solver_memory,
            SolverMemoryRecord,
            field_name="solver_memory",
        )
        retracing_rows = _require_row_tuple(
            self.retracing, RetracingRecordV2, field_name="retracing"
        )
        backend_rows = _require_row_tuple(
            self.backend_resolution,
            BackendResolutionRecord,
            field_name="backend_resolution",
        )
        measured_operations = {
            (row.operation, row.requested_backend) for row in backend_rows
        }
        missing_operations = sorted(
            _REQUIRED_BACKEND_RESOLUTION_OPERATIONS - measured_operations
        )
        if missing_operations:
            raise BenchmarkRecordError(
                "backend_resolution is missing required operation/request rows: "
                f"{missing_operations}"
            )
        all_rows = (
            workload_rows + memory_rows + solver_rows + retracing_rows + backend_rows
        )
        provenances = [row.provenance for row in all_rows]
        if provenances and any(item != provenances[0] for item in provenances[1:]):
            raise BenchmarkRecordError(
                "PERF-001 document has heterogeneous provenance; every row must "
                "come from one clean implementation source"
            )
        memory_pairs = _pair_rows(
            memory_rows,
            field_name="memory_scaling",
            expected_states=frozenset({_UNCHUNKED_REFERENCE, _CHUNKED_PRODUCTION}),
        )
        for pair in memory_pairs.values():
            _require_pair_identity(pair, field_name="memory_scaling")
            first, second = pair
            for field_name in (
                "logical_n_baselines",
                "logical_n_sources",
                "logical_pair_count",
                "kernel_n_sources",
                "synthetic_input_bytes_excluded",
            ):
                if getattr(first, field_name) != getattr(second, field_name):
                    raise BenchmarkRecordError(
                        f"memory_scaling pair field {field_name} must match"
                    )
        solver_pairs = _pair_rows(
            solver_rows,
            field_name="solver_memory",
            expected_states=frozenset({_UNBUCKETED_REFERENCE, _BUCKETED_PRODUCTION}),
        )
        for pair in solver_pairs.values():
            _require_pair_identity(pair, field_name="solver_memory")
            first, second = pair
            for field_name in (
                "solver",
                "sky_representation",
                "logical_n_baselines",
                "logical_source_counts",
                "n_times",
                "n_frequencies",
                "target_kernel_pairs",
            ):
                if getattr(first, field_name) != getattr(second, field_name):
                    raise BenchmarkRecordError(
                        f"solver_memory pair field {field_name} must match"
                    )
        missing_solver_memory_paths = sorted(
            _REQUIRED_SOLVER_MEMORY_PATHS - {row.solver for row in solver_rows}
        )
        if missing_solver_memory_paths:
            raise BenchmarkRecordError(
                "solver_memory is missing required real solver paths: "
                f"{missing_solver_memory_paths}"
            )
        retracing_pairs = _pair_rows(
            retracing_rows,
            field_name="retracing",
            expected_states=frozenset({_UNBUCKETED_REFERENCE, _BUCKETED_PRODUCTION}),
        )
        for pair in retracing_pairs.values():
            _require_pair_identity(pair, field_name="retracing")
            first, second = pair
            for field_name in (
                "measurement_scope",
                "solver",
                "sky_representation",
                "logical_source_counts",
            ):
                if getattr(first, field_name) != getattr(second, field_name):
                    raise BenchmarkRecordError(
                        f"retracing pair field {field_name} must match"
                    )
        missing_retracing_paths = sorted(
            _REQUIRED_RETRACING_PATHS - {row.solver for row in retracing_rows}
        )
        if missing_retracing_paths:
            raise BenchmarkRecordError(
                "retracing is missing required synthetic and real solver paths: "
                f"{missing_retracing_paths}"
            )
