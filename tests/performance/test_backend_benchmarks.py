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

import importlib.metadata
import importlib.util
import os
import platform
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import radiosim.core.visibility as point_visibility
from radiosim.backends import get_backend
from radiosim.benchmarks import (
    PERF001_PROVENANCE_SCHEMA_VERSION,
    BenchmarkDocument,
    BenchmarkRecord,
    MemoryScalingRecord,
    MemoryScalingRecordV2,
    Perf001Provenance,
    RetracingRecord,
    RetracingRecordV2,
    SolverMemoryRecord,
    WorkloadShape,
    benchmark_backend_selection,
    benchmark_filename,
    benchmark_output_directory,
    build_record,
    compare_to_reference,
    describe_backend,
    describe_environment,
    measure_kernel_memory_scaling,
    measure_perf001_memory_scaling_pair,
    measure_perf001_solver_memory_pair,
    measure_perf001_solver_retracing_pair,
    measure_retracing,
    records_are_complete,
    time_backend_call,
    validate_perf001_cpu_evidence_document,
    verify_required_benchmark_accelerator,
    write_benchmark_document,
)
from radiosim.core.precision import PrecisionConfig
from radiosim.core.time_grid import build_observation_time_grid
from radiosim.core.visibility import _calculate_visibility, calculate_visibility
from radiosim.core.visibility_healpix import (
    _calculate_visibility_healpix,
    calculate_visibility_healpix,
)
from tests.unit.test_backends import test_backend_parity as parity

pytestmark = [pytest.mark.performance, pytest.mark.slow]

#: Backends measured, in order. With no opt-in environment this is byte-for-byte
#: the historical CPU matrix. ``bench-gpu`` selects ``numpy,gpu`` through the
#: same eight workload objects; NumPy stays first as every row's reference.
BACKEND_SELECTION = benchmark_backend_selection()
MEASURED_BACKENDS = BACKEND_SELECTION.backend_requests

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


def test_complete_perf001_cpu_generator_assembles_real_in_memory_document(
    tmp_path: Path,
) -> None:
    """Exercise the retained builder without invoking its publication path."""
    repository_root = Path(__file__).resolve().parents[2]
    tool_path = repository_root / "tools/wp7_perf001_cpu_evidence.py"
    spec = importlib.util.spec_from_file_location(
        "wp7_perf001_cpu_evidence_performance", tool_path
    )
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = tool
    spec.loader.exec_module(tool)
    source_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    provenance = Perf001Provenance.create(
        schema_version=PERF001_PROVENANCE_SCHEMA_VERSION,
        recorded_at_utc="2026-08-11T00:00:00+00:00",
        radiosim_version=importlib.metadata.version("radiosim"),
        git_sha=source_sha,
        working_tree_clean=True,
        platform=platform.platform(),
        machine=platform.machine(),
        cpu_model=platform.processor() or platform.machine(),
        cpu_count_logical=os.cpu_count() or 1,
        python_version=platform.python_version(),
        numpy_version=importlib.metadata.version("numpy"),
        jax_version=importlib.metadata.version("jax"),
        jaxlib_version=importlib.metadata.version("jaxlib"),
        dask_version=importlib.metadata.version("dask"),
        pixi_environment="default",
        pixi_lock_sha256=tool.PIXI_LOCK_SHA256,
    )

    document = tool._measure_document(
        provenance,
        tmp_path,
        repository_root=repository_root,
    )

    validate_perf001_cpu_evidence_document(document)
    assert tuple(
        len(getattr(document, field))
        for field in (
            "workload_benchmarks",
            "memory_scaling",
            "solver_memory",
            "retracing",
            "backend_resolution",
        )
    ) == (24, 8, 4, 6, 3)
    assert not (repository_root / "output/benchmarks/reference/perf001").exists()


def _backend_for(name: str):
    if name == "jax":
        backend = get_backend("jax", device="cpu")
    elif name == "gpu":
        backend = get_backend("gpu")
    elif name == "dask":
        backend = get_backend("dask", mode="cpu")
    else:
        backend = get_backend("numpy")
    verify_required_benchmark_accelerator(
        backend,
        requested=name,
        required_accelerator=BACKEND_SELECTION.required_accelerator,
    )
    return backend


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
        for requested in MEASURED_BACKENDS
        if requested in {"numpy", "jax", "gpu"}
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
    assert len(WORKLOADS) == 8
    assert len({id(workload) for workload in WORKLOADS}) == 8
    assert records_are_complete(measured.records)
    assert len(measured.records) == len(WORKLOADS) * len(MEASURED_BACKENDS)


def test_every_record_states_which_device_was_actually_exercised(
    measured: BenchmarkDocument,
) -> None:
    """CPU stays explicit; opt-in GPU rows require measured accelerator facts."""
    for record in measured.records:
        if record.backend_requested == "gpu":
            assert record.device_kind == "gpu"
            assert record.accelerator == "gpu"
            assert record.accelerator_driver
            assert "gpu" not in record.unmeasured
        else:
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


def test_jax_records_are_within_the_section_13_5_tolerance(
    measured: BenchmarkDocument,
) -> None:
    """B1's tolerance applies to selected JAX-CPU or real-GPU rows."""
    selected_jax_requests = {"jax", "gpu"}.intersection(MEASURED_BACKENDS)
    jax_records = [
        record
        for record in measured.records
        if record.backend_requested in selected_jax_requests
    ]

    assert len(jax_records) == len(WORKLOADS) * len(selected_jax_requests)
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

    expected_rows = len(WORKLOADS) if "dask" in MEASURED_BACKENDS else 0
    assert len(dask_records) == expected_rows
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
    assert len(measured.retracing) == len(
        {"numpy", "jax", "gpu"}.intersection(MEASURED_BACKENDS)
    )
    numpy_record = next(
        record for record in measured.retracing if not record.compilation_used
    )

    for record in measured.retracing:
        assert record.steps == len(RETRACING_SOURCE_COUNTS)
        assert record.distinct_source_counts == len(set(RETRACING_SOURCE_COUNTS))
        assert set(record.first_call_seconds_by_source_count) == {
            str(count) for count in set(RETRACING_SOURCE_COUNTS)
        }
        assert record.retrace_overhead_seconds >= 0.0

    # Preserve the historical CPU evidence check only when its exact JAX-CPU
    # row is selected. GPU timings are recorded as evidence but are never a
    # correctness or acceptance outcome.
    jax_cpu_records = tuple(
        record
        for record in measured.retracing
        if record.backend_actual.startswith("jax-cpu-")
    )
    if jax_cpu_records:
        assert len(jax_cpu_records) == 1
        jax_cpu_record = jax_cpu_records[0]
        assert (
            jax_cpu_record.max_first_to_repeat_ratio
            > numpy_record.max_first_to_repeat_ratio
        )
        assert (
            jax_cpu_record.retrace_overhead_seconds
            > numpy_record.retrace_overhead_seconds
        )


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


# =========================================================================
# PERF-001 v2 P-a/P-b evidence scaffolding
# =========================================================================


def _perf001_measurement_test_provenance() -> Perf001Provenance:
    """Schema-valid provenance for non-retained performance-test records.

    Retained generation uses ``describe_perf001_provenance`` and fails on a
    dirty checkout.  These tests intentionally exercise the measurement
    mechanics while their own source files are dirty during development, and
    never write these rows as evidence.
    """
    import importlib.metadata

    return Perf001Provenance.create(
        schema_version=PERF001_PROVENANCE_SCHEMA_VERSION,
        recorded_at_utc="2026-08-11T00:00:00+00:00",
        radiosim_version="0.3.0",
        git_sha="a" * 40,
        working_tree_clean=True,
        platform="performance-test-only",
        machine="performance-test-only",
        cpu_model="performance-test-only",
        cpu_count_logical=1,
        python_version="3.11-or-3.12",
        numpy_version=importlib.metadata.version("numpy"),
        jax_version=importlib.metadata.version("jax"),
        jaxlib_version=importlib.metadata.version("jaxlib"),
        dask_version=importlib.metadata.version("dask"),
        pixi_environment="test-only",
        pixi_lock_sha256="b" * 64,
    )


class _Perf001AllVisibleSkyCoord:
    """Point-solver coordinate stand-in with a controlled all-visible count."""

    def __init__(self, *, ra: object, **_kwargs: object) -> None:
        self._count = len(ra)  # type: ignore[arg-type]

    def transform_to(self, _frame: object) -> SimpleNamespace:
        return SimpleNamespace(
            az=SimpleNamespace(
                rad=np.linspace(0.1, 0.2, self._count, dtype=np.float64)
            ),
            alt=SimpleNamespace(
                rad=np.linspace(0.8, 0.9, self._count, dtype=np.float64)
            ),
        )


class _Perf001AllVisiblePixels:
    def __init__(self, count: int) -> None:
        self._count = count

    def __len__(self) -> int:
        return self._count

    def transform_to(self, _frame: object) -> SimpleNamespace:
        return _Perf001AllVisibleSkyCoord(ra=np.empty(self._count)).transform_to(_frame)


class _Perf001HealpixPayload:
    nside = 1
    pixel_solid_angle = 1.0

    def __init__(self, count: int) -> None:
        self.pixel_coords = _Perf001AllVisiblePixels(count)
        intensity = np.linspace(1.0, 2.0, count, dtype=np.float64)
        self.stokes = (
            intensity,
            intensity * 0.1,
            intensity * 0.05,
            intensity * 0.02,
        )

    def get_map_at_frequency(self, _frequency: float) -> np.ndarray:
        return self.stokes[0]

    def get_stokes_maps_at_frequency(
        self, _frequency: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.stokes


def _perf001_logical_arrays(
    values_by_count: dict[int, object],
) -> tuple[tuple[str, np.ndarray], ...]:
    logical_inputs: list[tuple[str, np.ndarray]] = []
    for count, values in sorted(values_by_count.items()):
        if isinstance(values, dict):
            items = values.items()
        else:
            items = enumerate(values.stokes)  # type: ignore[attr-defined]
        for name, array in items:
            if isinstance(array, np.ndarray):
                logical_inputs.append((f"sources_{count}.{name}", array))
    return tuple(logical_inputs)


def test_perf001_p_a_records_bounded_leaf_pairs_and_lower_large_wrapper_peak() -> None:
    backend = get_backend("numpy", precision="standard")
    reference, production = measure_perf001_memory_scaling_pair(
        backend,
        provenance=_perf001_measurement_test_provenance(),
        n_baselines=512,
        n_sources=512,
        comparison_id="perf001-p-a-large",
    )

    assert isinstance(reference, MemoryScalingRecordV2)
    assert isinstance(production, MemoryScalingRecordV2)
    assert reference.max_kernel_pair_count == 512 * 512
    assert production.max_kernel_pair_count <= 131072
    assert production.kernel_baseline_chunks == (256, 256)
    assert production.peak_host_bytes < reference.peak_host_bytes


@pytest.mark.parametrize("solver", ["point", "healpix"])
def test_perf001_p_b_measures_real_private_solver_memory_and_retracing(
    tmp_path,
    monkeypatch,
    solver: str,
) -> None:
    """Both P-b rows traverse a complete production point/HEALPix solver."""
    monkeypatch.setattr(point_visibility, "SkyCoord", _Perf001AllVisibleSkyCoord)
    instrument, beam_system, receptors = parity._solver_components(tmp_path)
    compiled_request = BACKEND_SELECTION.required_accelerator or "jax"
    backend = _backend_for(compiled_request)
    frequencies = parity.FREQUENCIES[:1].copy()
    logical_counts = (0, 3, 4, 5, 8, 0, 3, 4, 5, 8)
    if solver == "point":
        fixtures = {
            count: parity._point_sources(
                # The parity helper deliberately indexes a polarized signal;
                # for an empty sky the signal path is immaterial and must stay
                # empty so the complete solver returns before its leaf.
                polarized=count > 0,
                gaussian=True,
                n_sources=count,
                per_channel=False,
            )
            for count in sorted(set(logical_counts))
        }
    else:
        fixtures = {
            count: _Perf001HealpixPayload(count)
            for count in sorted(set(logical_counts))
        }

    common = {
        "instrument": instrument,
        "beam_system": beam_system,
        "location": parity.LOCATION,
        "time_grid": parity.SINGLE_TIME_GRID,
        "frequencies": frequencies,
        "backend": backend,
        "receptors": receptors,
    }

    def run_count(policy: str, count: int) -> object:
        if solver == "point":
            return _calculate_visibility(
                source_arrays=fixtures[count],
                **common,
                _source_bucket_policy=policy,
            )
        return _calculate_visibility_healpix(
            sky_model=SimpleNamespace(
                healpix=fixtures[count],
                has_polarized_healpix_maps=True,
                brightness_conversion="rayleigh-jeans",
                model_name=f"perf001-{count}-pixel",
            ),
            include_polarization=True,
            **common,
            _source_bucket_policy=policy,
        )

    manifest = {
        "schema_version": "radiosim.perf001.fixture.real_solver_retracing.v1",
        "solver": solver,
        "logical_source_counts": list(logical_counts),
        "time_mjd": [float(value) for value in parity.SINGLE_TIME_GRID.to_mjd()],
        "frequencies_hz": frequencies.tolist(),
    }
    logical_inputs = (
        ("baseline_vectors_enu_m", instrument.baseline_vectors_enu_m),
        ("frequencies_hz", frequencies),
        *_perf001_logical_arrays(fixtures),
    )
    provenance = _perf001_measurement_test_provenance()
    leaf_call_deltas: list[tuple[str, int, int, int]] = []
    retracing_reference, retracing_production = measure_perf001_solver_retracing_pair(
        backend,
        provenance=provenance,
        solver=solver,
        logical_source_counts=logical_counts,
        fixture_manifest=manifest,
        logical_inputs=logical_inputs,
        run_solver_step=lambda policy, index: run_count(policy, logical_counts[index]),
        comparison_id=f"perf001-p-b-{solver}-retrace",
        _leaf_call_delta_observer=lambda state, index, logical_count, delta: (
            leaf_call_deltas.append((state, index, logical_count, delta))
        ),
    )
    memory_reference, memory_production = measure_perf001_solver_memory_pair(
        backend,
        provenance=provenance,
        solver=solver,
        logical_n_baselines=len(instrument.selected_pairs),
        logical_source_counts=(3,),
        n_times=1,
        n_frequencies=1,
        fixture_manifest={**manifest, "logical_source_counts": [3]},
        logical_inputs=(
            ("baseline_vectors_enu_m", instrument.baseline_vectors_enu_m),
            ("frequencies_hz", frequencies),
            *_perf001_logical_arrays({3: fixtures[3]}),
        ),
        run_solver=lambda policy: run_count(policy, 3),
        comparison_id=f"perf001-p-b-{solver}-memory",
    )

    assert isinstance(retracing_reference, RetracingRecordV2)
    assert isinstance(retracing_production, RetracingRecordV2)
    assert isinstance(memory_reference, SolverMemoryRecord)
    assert isinstance(memory_production, SolverMemoryRecord)
    assert retracing_production.distinct_signature_count < (
        retracing_reference.distinct_signature_count
    )
    assert retracing_reference.context.input_identity_sha256 == (
        retracing_production.context.input_identity_sha256
    )
    assert retracing_reference.kernel_source_counts == logical_counts
    assert retracing_production.kernel_source_counts == (
        0,
        4,
        4,
        8,
        8,
        0,
        4,
        4,
        8,
        8,
    )
    assert retracing_reference.leaf_call_count == 8
    assert retracing_production.leaf_call_count == 8
    assert len(leaf_call_deltas) == 2 * len(logical_counts)
    for state in ("unbucketed_reference", "bucketed_production"):
        state_deltas = [item for item in leaf_call_deltas if item[0] == state]
        assert tuple((item[1], item[3]) for item in state_deltas if item[2] == 0) == (
            (0, 0),
            (5, 0),
        )
        assert all(item[3] > 0 for item in state_deltas if item[2] > 0)
    for record in (retracing_reference, retracing_production):
        assert all(
            observation.jones_p_shape is not None and observation.jones_p_shape[1] > 0
            for observation in record.observed_signatures
        )
    assert memory_reference.context.input_identity_sha256 == (
        memory_production.context.input_identity_sha256
    )
    assert memory_reference.kernel_source_counts == (3,)
    assert memory_production.kernel_source_counts == (4,)
