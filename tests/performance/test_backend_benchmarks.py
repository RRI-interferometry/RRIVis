"""The Tier 6 benchmark run (``Tier6HybridRuntimePlan.md`` Sections 22, 23, 27).

Marked ``performance`` **and** ``slow``, so the CI invocation
(``pixi run test -- -m "not slow"``) excludes it with no CI change and no new
gating job. Run it deliberately::

    pixi run bench

What this module measures:

1. every row of the Section 13.4 workload matrix, on NumPy (the reference),
   JAX-CPU, and Dask, with the full Section 22.2 timing discipline;
2. one deliberately larger point workload, because the Section 13.4 rows are
   tiny by design -- they exist to compare *values*, and at two sources and two
   baselines every backend measures Python overhead rather than arithmetic. A
   record set that only covered them would invite exactly the misreading this
   tier exists to prevent;
3. per-step retracing under a time-varying visible-source count, on NumPy and on
   JAX-CPU -- the first Tier 6H acceptance obligation routed to 6I;
4. the compiled kernel's ``(B, S, 2, 2)`` working set against baseline and source
   counts -- the second such obligation.

What this module never does: assert a time threshold. Section 22.1 is explicit
that no benchmark number is hard-coded into an assertion. Every test here asserts
record *completeness*, record *honesty*, or *correctness against NumPy*. Speed is
recorded and published; it is never a pass condition, because a timing assertion
on shared CI-class hardware is a flake generator, not a guarantee.

``loader_max_workers`` is ``0`` in every record below, meaning "no sky loader ran
in this measurement": these benchmarks drive the solvers directly against an
already-constructed sky model, so loader concurrency is genuinely not exercised
and the record says so rather than reporting a policy that never took effect.

The Section 13.4 inputs are imported from the parity module rather than restated,
so a benchmark and its correctness test can never drift apart: they are the same
seven workloads by construction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from radiosim.backends import get_backend
from radiosim.benchmarks import (
    BenchmarkDocument,
    BenchmarkRecord,
    MemoryScalingRecord,
    RetracingRecord,
    WorkloadShape,
    benchmark_filename,
    benchmark_output_directory,
    build_record,
    compare_to_reference,
    describe_backend,
    describe_environment,
    measure_kernel_memory_scaling,
    measure_retracing,
    records_are_complete,
    time_backend_call,
    write_benchmark_document,
)
from radiosim.core.precision import PrecisionConfig
from radiosim.core.time_grid import build_observation_time_grid
from radiosim.core.visibility import calculate_visibility
from radiosim.core.visibility_healpix import calculate_visibility_healpix
from tests.unit.test_backends import test_backend_parity as parity

pytestmark = [pytest.mark.performance, pytest.mark.slow]

#: Backends measured, in order. NumPy is always first: it is the reference every
#: other record's correctness delta is taken against.
MEASURED_BACKENDS = ("numpy", "jax", "dask")

#: A larger point workload, so the record set contains at least one row where the
#: arithmetic, rather than the Python orchestration, dominates.
SCALED_SOURCES = 4096
SCALED_TIME_GRID = build_observation_time_grid(
    start_time=parity.OBSTIME.isot, duration_seconds=4.0, cadence_seconds=1.0
)

#: Baseline/source pairs for the kernel working-set measurement. Kept modest on
#: purpose: the point is the slope, and the slope is already unambiguous here.
MEMORY_SCALING_PAIRS = ((100, 100), (200, 200), (400, 400), (800, 800))

#: A visible-source count that rises and falls, the way a real observation's
#: above-horizon set does across a time axis.
RETRACING_SOURCE_COUNTS = (16, 24, 32, 24, 16, 24, 32)


def _backend_for(name: str):
    if name == "jax":
        return get_backend("jax", device="cpu")
    if name == "dask":
        return get_backend("dask", mode="cpu")
    return get_backend("numpy")


def _scaled_point_sources() -> dict[str, Any]:
    """A wider point sky on the same instrument, sources spread near the zenith."""
    rng = np.random.default_rng(20260731)
    n = SCALED_SOURCES
    return {
        "ra_rad": parity.LST_RAD + rng.uniform(-0.05, 0.05, n),
        "dec_rad": -0.536 + rng.uniform(-0.05, 0.05, n),
        "flux": rng.uniform(0.5, 5.0, n),
        "spectral_index": np.full(n, -0.7),
        "stokes_q": np.zeros(n),
        "stokes_u": np.zeros(n),
        "stokes_v": np.zeros(n),
        "ref_freq": np.full(n, 100e6),
        "rotation_measure": np.zeros(n),
        "spectral_coeffs": None,
        "per_channel_flux": None,
        "per_channel_stokes_q": None,
        "per_channel_stokes_u": None,
        "per_channel_stokes_v": None,
        "channel_frequencies": None,
        "major_arcsec": np.zeros(n),
        "minor_arcsec": np.zeros(n),
        "pa_deg": np.zeros(n),
    }


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    """One benchmark row: how to run it, and the counts its record must state."""

    name: str
    sky_representation: str
    n_point_sources: int
    n_healpix_pixels: int
    n_times: int
    n_frequencies: int
    run: Callable[[Any, Any], Any]


def _point_runner(**options: Any) -> Callable[[Any, Any], Any]:
    def run(backend: Any, components: Any) -> Any:
        instrument, beam_system, receptors = components
        return calculate_visibility(
            instrument=instrument,
            beam_system=beam_system,
            source_arrays=parity._point_sources(
                polarized=options["polarized"], gaussian=options["gaussian"]
            ),
            location=parity.LOCATION,
            time_grid=(
                parity.SINGLE_TIME_GRID if options["single_time"] else parity.TIME_GRID
            ),
            frequencies=parity.FREQUENCIES,
            backend=backend,
            receptors=receptors,
        )

    return run


def _healpix_runner(*, polarized: bool) -> Callable[[Any, Any], Any]:
    def run(backend: Any, components: Any) -> Any:
        instrument, beam_system, receptors = components
        return calculate_visibility_healpix(
            parity._healpix_model(polarized=polarized),
            instrument=instrument,
            beam_system=beam_system,
            location=parity.LOCATION,
            time_grid=parity.TIME_GRID,
            frequencies=parity.FREQUENCIES,
            backend=backend,
            receptors=receptors,
            include_polarization=polarized,
        )

    return run


def _hybrid_runner() -> Callable[[Any, Any], Any]:
    point = _point_runner(polarized=True, gaussian=False, single_time=False)
    healpix = _healpix_runner(polarized=True)

    def run(backend: Any, components: Any) -> Any:
        # Section 9.1: the hybrid result is the canonical sum of the two
        # components, taken through ``backend.add``, so the benchmark measures
        # the same arithmetic the hybrid solve performs.
        return backend.add(point(backend, components), healpix(backend, components))

    return run


def _scaled_runner() -> Callable[[Any, Any], Any]:
    sources = _scaled_point_sources()

    def run(backend: Any, components: Any) -> Any:
        instrument, beam_system, receptors = components
        return calculate_visibility(
            instrument=instrument,
            beam_system=beam_system,
            source_arrays=sources,
            location=parity.LOCATION,
            time_grid=SCALED_TIME_GRID,
            frequencies=parity.FREQUENCIES,
            backend=backend,
            receptors=receptors,
        )

    return run


WORKLOADS: tuple[WorkloadSpec, ...] = (
    WorkloadSpec(
        "point_unpolarized_1time_2freq",
        "point_sources",
        2,
        0,
        1,
        2,
        _point_runner(polarized=False, gaussian=False, single_time=True),
    ),
    WorkloadSpec(
        "point_polarized_2times",
        "point_sources",
        2,
        0,
        2,
        2,
        _point_runner(polarized=True, gaussian=False, single_time=False),
    ),
    WorkloadSpec(
        "point_gaussian_morphology",
        "point_sources",
        2,
        0,
        2,
        2,
        _point_runner(polarized=True, gaussian=True, single_time=False),
    ),
    WorkloadSpec(
        "healpix_scalar", "healpix_map", 0, 12, 2, 2, _healpix_runner(polarized=False)
    ),
    WorkloadSpec(
        "healpix_polarized", "healpix_map", 0, 12, 2, 2, _healpix_runner(polarized=True)
    ),
    WorkloadSpec("hybrid_point_plus_healpix", "hybrid", 2, 12, 2, 2, _hybrid_runner()),
    WorkloadSpec(
        "heterogeneous_receptor_bases",
        "point_sources",
        2,
        0,
        2,
        2,
        _point_runner(polarized=True, gaussian=False, single_time=False),
    ),
    WorkloadSpec(
        "point_scaled_4096_sources_4times",
        "point_sources",
        SCALED_SOURCES,
        0,
        4,
        2,
        _scaled_runner(),
    ),
)

#: The one workload that needs a non-default receptor configuration.
HETEROGENEOUS_WORKLOAD = "heterogeneous_receptor_bases"


@pytest.fixture(scope="module")
def measured(tmp_path_factory) -> BenchmarkDocument:
    """Run every benchmark once, write the document, and return it."""
    homogeneous_dir = tmp_path_factory.mktemp("bench-homogeneous")
    heterogeneous_dir = tmp_path_factory.mktemp("bench-heterogeneous")
    homogeneous = parity._solver_components(homogeneous_dir)
    heterogeneous = parity._solver_components(
        heterogeneous_dir, receptors=parity._HETEROGENEOUS_RECEPTORS
    )

    environment = describe_environment()
    precision = PrecisionConfig.standard()
    records: list[BenchmarkRecord] = []

    for workload in WORKLOADS:
        components = (
            heterogeneous if workload.name == HETEROGENEOUS_WORKLOAD else homogeneous
        )
        instrument = components[0]
        reference: np.ndarray | None = None
        for requested in MEASURED_BACKENDS:
            backend = _backend_for(requested)
            timing = time_backend_call(
                lambda backend=backend,
                workload=workload,
                components=components: workload.run(backend, components),
                backend=backend,
            )
            if reference is None:
                reference = timing.host_result
            deviation = compare_to_reference(reference, timing.host_result)
            records.append(
                build_record(
                    environment=environment,
                    backend_facts=describe_backend(backend, requested=requested),
                    shape=WorkloadShape(
                        workload=workload.name,
                        n_antennas=len(instrument.antenna_numbers),
                        n_baselines=len(instrument.selected_pairs),
                        n_point_sources=workload.n_point_sources,
                        n_healpix_pixels=workload.n_healpix_pixels,
                        n_times=workload.n_times,
                        n_frequencies=workload.n_frequencies,
                        sky_representation=workload.sky_representation,
                        solver_workers=1,
                        loader_max_workers=0,
                    ),
                    timing=timing,
                    deviation=deviation,
                    precision=precision,
                    precision_preset="standard",
                )
            )

    retracing = tuple(
        measure_retracing(
            _backend_for(requested), source_counts=RETRACING_SOURCE_COUNTS
        )
        for requested in ("numpy", "jax")
    )
    memory_scaling = tuple(
        measure_kernel_memory_scaling(
            get_backend("numpy"), n_baselines=baselines, n_sources=sources
        )
        for baselines, sources in MEMORY_SCALING_PAIRS
    )

    document = BenchmarkDocument(
        records=tuple(records),
        retracing=retracing,
        memory_scaling=memory_scaling,
    )
    write_benchmark_document(
        document,
        directory=benchmark_output_directory(),
        filename=benchmark_filename(),
    )
    return document


def test_every_record_is_complete(measured: BenchmarkDocument) -> None:
    """P1 at the document level: no partial record reaches the output file."""
    assert records_are_complete(measured.records)
    assert len(measured.records) == len(WORKLOADS) * len(MEASURED_BACKENDS)


def test_every_record_states_that_no_accelerator_was_exercised(
    measured: BenchmarkDocument,
) -> None:
    """P2: Section 4 forbids an accelerator claim, and the records say so."""
    for record in measured.records:
        assert record.accelerator == "none", record.workload
        assert record.accelerator_driver is None
        assert record.device_kind == "cpu"
        assert "gpu" in record.unmeasured
        assert "tpu" in record.unmeasured


def test_every_record_carries_the_full_timing_and_memory_profile(
    measured: BenchmarkDocument,
) -> None:
    """Section 22.2: setup, compile, steady state, transfer, and peak memory."""
    for record in measured.records:
        assert record.setup_seconds > 0.0, record.workload
        assert record.compile_seconds >= 0.0
        assert record.steady_state_iterations >= 5
        assert (
            record.steady_state_min_seconds
            <= record.steady_state_median_seconds
            <= record.steady_state_max_seconds
        )
        assert record.host_transfer_seconds >= 0.0
        assert record.peak_host_bytes > 0
        assert record.backend_memory_info


def test_only_the_jax_records_report_compilation(
    measured: BenchmarkDocument,
) -> None:
    """A backend that advertised compilation and did not compile would be a lie."""
    for record in measured.records:
        assert record.compilation_used is record.backend_actual.startswith("jax")


def test_jax_cpu_records_are_within_the_section_13_5_tolerance(
    measured: BenchmarkDocument,
) -> None:
    """B1's tolerance, restated as a property of the published records."""
    jax_records = [
        record for record in measured.records if record.backend_requested == "jax"
    ]

    assert len(jax_records) == len(WORKLOADS)
    for record in jax_records:
        assert record.reference_backend == "numpy"
        assert record.within_tolerance, (
            f"{record.workload}: max |dV| = {record.max_absolute_deviation:.3e}"
        )


def test_dask_records_are_bit_identical_to_numpy(
    measured: BenchmarkDocument,
) -> None:
    """B2: the Dask backend delegates to NumPy, so zero is the only answer."""
    dask_records = [
        record for record in measured.records if record.backend_requested == "dask"
    ]

    assert len(dask_records) == len(WORKLOADS)
    for record in dask_records:
        assert record.max_absolute_deviation == 0.0, record.workload


def test_numpy_records_are_their_own_reference(measured: BenchmarkDocument) -> None:
    """The reference row must be exactly zero, or the harness is measuring noise."""
    for record in measured.records:
        if record.backend_requested == "numpy":
            assert record.max_absolute_deviation == 0.0


def test_retracing_under_a_time_varying_source_count_is_measured(
    measured: BenchmarkDocument,
) -> None:
    """Tier 6H acceptance obligation 1, discharged as a published measurement.

    Section 13.6 calls the compiled kernel "shape-stable within a run". Both
    solvers mask by ``above_horizon`` per time step, so a run whose visible-source
    count changes presents the kernel with a changing source axis. Under JAX each
    newly seen source count costs a recompilation before it costs arithmetic;
    under NumPy nothing is compiled and a first call costs what a repeat call
    costs. Both are recorded so the difference is readable rather than assumed.
    """
    assert len(measured.retracing) == 2
    numpy_record = next(
        record for record in measured.retracing if not record.compilation_used
    )
    jax_record = next(
        record for record in measured.retracing if record.compilation_used
    )

    for record in (numpy_record, jax_record):
        assert record.steps == len(RETRACING_SOURCE_COUNTS)
        assert record.distinct_source_counts == len(set(RETRACING_SOURCE_COUNTS))
        assert set(record.first_call_seconds_by_source_count) == {
            str(count) for count in set(RETRACING_SOURCE_COUNTS)
        }
        assert record.retrace_overhead_seconds >= 0.0

    # The measurement that matters: on a compiling backend the first call at a
    # newly seen source count costs order-of-magnitude more than a repeat call at
    # the same count, and on a non-compiling backend it does not. The assertion
    # is a *shape* comparison between the two backends, never a time threshold.
    assert jax_record.max_first_to_repeat_ratio > numpy_record.max_first_to_repeat_ratio
    assert jax_record.retrace_overhead_seconds > numpy_record.retrace_overhead_seconds


def test_the_kernel_working_set_scales_with_baselines_times_sources(
    measured: BenchmarkDocument,
) -> None:
    """Tier 6H acceptance obligation 2, discharged as a published measurement.

    ``core/contraction.py`` materializes ``(B, S, 2, 2)`` batches, so peak memory
    grows with the *product*. No Section 13.4 workload and no shipped
    configuration exceeds fifteen baselines, so this is invisible to every
    correctness test; the record makes it visible.
    """
    assert len(measured.memory_scaling) == len(MEMORY_SCALING_PAIRS)
    ordered = sorted(measured.memory_scaling, key=lambda record: record.pair_count)

    for record in ordered:
        assert record.peak_host_bytes > 0
        assert record.bytes_per_pair > 0.0
    for smaller, larger in zip(ordered, ordered[1:], strict=False):
        assert larger.peak_host_bytes > smaller.peak_host_bytes

    # Linear, not constant: the largest row's per-pair cost sits within a factor
    # of two of the smallest's, which a sub-linear working set could not do.
    assert ordered[-1].bytes_per_pair / ordered[0].bytes_per_pair < 2.0
    assert ordered[0].bytes_per_pair / ordered[-1].bytes_per_pair < 2.0


def test_the_document_is_written_where_the_documentation_says_it_is(
    measured: BenchmarkDocument,
) -> None:
    """Section 22.1: ``output/benchmarks/<UTC timestamp>-<host tag>.json``."""
    directory = benchmark_output_directory()
    written = sorted(directory.glob("*.json"))

    assert written, f"no benchmark record written under {directory}"
    assert isinstance(measured.records[0], BenchmarkRecord)
    assert isinstance(measured.retracing[0], RetracingRecord)
    assert isinstance(measured.memory_scaling[0], MemoryScalingRecord)
