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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "MEMORY_SCALING_SCHEMA_VERSION",
    "RETRACING_SCHEMA_VERSION",
    "BenchmarkDocument",
    "BenchmarkRecord",
    "BenchmarkRecordError",
    "MemoryScalingRecord",
    "RetracingRecord",
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
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (str, bytes)):
        return value.decode() if isinstance(value, bytes) else value
    if isinstance(value, Sequence):
        return [_json_safe(item) for item in value]
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
    destination.write_text(
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
