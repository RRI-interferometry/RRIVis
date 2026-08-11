# PERF-001 runtime-mitigation and accelerator-evidence design gate

**WP-7 design-gate record — 2026-08-11**

**Source reviewed:** `d836195c2b981b14f162d7c5ae01710ea3b5d2a2`, clean and
aligned with `origin/codex/post-tier8-remediation-complete`.

**Status:** this memo authorizes implementation of P-a through P-d and the
GPU-ready portion of P-e. It is not implementation evidence, accelerator
evidence, or acceptance. `PERF-001` remains **ROADMAP**. P-a through P-d may be
accepted separately and may satisfy WP-8's CPU dependency, but the register
must remain open until a real accelerator record is independently accepted or
the register is formally re-adjudicated.

## 1. Ruling

WP-7 separates five claims that the old `PERF-001` row combined:

1. source-dependent contraction-leaf working temporaries can be bounded
   without changing a visibility's source summation order, while output and
   assembly storage still grows with baselines;
2. the number of JAX source-axis shapes can be reduced with early, zero-signal
   host padding;
3. `backend="auto"` can be cheap and deterministic without pretending to
   discover an accelerator;
4. an abstract simulator must not inherit an unmeasured GPU capability; and
5. a real accelerator claim requires a real accelerator run.

P-a through P-d are implementable in the current checkout. P-e infrastructure
is also implementable, but the measurement is externally blocked: the local
Apple GPU is not supported by the locked JAX runtime, no tracked workflow
targets an accelerator or self-hosted label, and no authorized external GPU
host is part of this `PERF-001` design record. No synthetic record, CPU
fallback, or favourable reduced workload can close that gap.

NumPy remains RadioSim's CPU reference. A JAX-CPU result within the existing
correctness tolerance is valid even when it is slower. Benchmark time is
reported, never used as a fast-test threshold.

## 2. Invariants

The following constraints are normative across every WP-7 slice:

- `baseline_contraction` keeps its six array operands and keyword-only backend.
- `baseline_contraction_for(backend)` keeps its one-parameter public signature.
- RadioSim keeps exactly one `backend.compile(...)` site and one six-input
  compiled leaf.
- P-a never splits or reorders the source axis.
- Real sources retain their original order. P-b appends dummy sources only.
- NumPy and Dask production paths remain unpadded and byte-identical to their
  accepted values.
- JAX keeps the existing Section 13.5 float64 predicate; no tolerance changes.
- Neither P-a nor P-b is a bound on total solver memory.
- No timing value is a correctness gate.
- No accelerator sentence exists without a retained, independently accepted
  record.
- Existing v1 benchmark schemas and the historical reference record remain
  readable and byte-unchanged.
- No private scheduling choice enters scientific fingerprints or public
  configuration.

The historical reference is
`output/benchmarks/reference/20260731T104303Z-darwin-arm64.json`, SHA-256
`00a02edd98903254e1f5f04569e88def0fff5ff239fbff40f2f5f34c5dc8b225`.
It records approximately 208 bytes per baseline-source pair at the largest
NumPy memory row and a JAX first-to-repeat ratio of approximately 493.9. It is
valid before-evidence; its schemas must not be widened in place.

## 3. P-a — baseline-axis chunking

### 3.1 Stable policy

The production target is

```text
target_kernel_pairs = 131072  # 2**17
```

For logical baseline count `B` and kernel source count `S`:

```text
if B == 0 or S == 0:
    call the leaf once with the original inputs
else:
    chunk_B = max(1, min(B, target_kernel_pairs // S))
```

The wrapper slices only baseline-bearing inputs: `J_p`, `J_q`, `phase`, and a
non-scalar `envelope`. It reuses source-only `coherency` or `stokes_i`, invokes
the same compiled leaf in baseline order, and concatenates `(chunk_B,2,2)`
outputs along the baseline axis. An uneven tail is not padded.

`131072` is a target, not a universal hard cap. When `S > 131072`, one baseline
already exceeds it; splitting `S` would alter the accepted source reduction
order. The truthful maximum pair count for one call is therefore bounded by
`max(target_kernel_pairs, S)`.

The retained approximately 208-byte-per-pair NumPy measurement puts a full
target-sized contraction call near 26 MiB of Python-traced temporary
allocation. That is a sizing rationale, not a cross-backend byte guarantee.

### 3.2 API and test seam

The public factory remains exactly `baseline_contraction_for(backend)`. A
private `_baseline_contraction_for_policy(...)` may accept `None` for the
unbounded before-control and a positive integer for tests and evidence. It
must reject booleans, zero, and negative targets. The public factory always
selects the module-owned production target.

The returned object is a Python scheduling wrapper around the one compiled
leaf. Tests must assert the compile-call count and leaf signature rather than
depending on a JAX-specific `.lower` attribute on the outer wrapper.

### 3.3 Exact scope of the memory claim

P-a bounds only source-dependent working allocation within each contraction
leaf invocation. The scheduling wrapper must retain chunk outputs and assemble
the final output, so unavoidable output/assembly storage still grows with
`B`. Both solvers also construct full `J_p`, `J_q`, phase, and sometimes
envelope arrays with shape proportional to `B*S` before the wrapper. The
existing `MemoryScalingRecord` starts tracing after those inputs are built.
Consequently:

- `measurement_scope` is
  `contraction_wrapper_python_heap_including_output_assembly`;
- `allocator` is `python_heap_tracemalloc`;
- `max_kernel_pair_count` identifies the bounded leaf component;
- prebuilt inputs are excluded, while output chunks and assembly are included;
- JAX-native and device allocations are excluded; and
- neither the complete wrapper peak nor end-to-end solver memory is claimed to
  plateau with `B`.

`RIMESimulator.get_memory_estimate()` must stop omitting the full baseline-source
Jones and phase inputs. Its estimate must identify logical versus bucketed
source counts, include the possible less-than-two-times bucket expansion, and
separate full caller inputs and output assembly from the bounded
source-dependent leaf working set. The base
and RIME methods gain the backward-compatible keyword-only parameter
`kernel_n_sources: int | None = None`; `None` means the logical source count.
`Simulator.get_memory_estimate()` computes the power-of-two kernel count only
when its resolved backend supports compilation, otherwise passes the logical
count, and supplies both counts to the simulator estimate.

## 4. P-b — source-axis bucket scheduling

### 4.1 Bucket rule and placement

For a nonzero logical visible-source count `S`:

```text
bucket(S) = 1 << (S - 1).bit_length()
```

Thus `S <= bucket(S) < 2*S`. Bucketing applies only when
`backend.supports_compilation` is true. NumPy and Dask execute the original,
unmodified logical arrays.

Padding occurs at the earliest safe host boundary:

1. perform coordinate transforms and the horizon mask;
2. return the accepted zero block if no source is visible;
3. record the logical count;
4. append dummy source data to the power-of-two bucket; then
5. construct `DirectionBatch`, convert to backend arrays, evaluate Jones terms,
   phase, morphology, and the contraction.

Late JAX-array padding immediately before the leaf is rejected. On the reviewed
host, logical counts `17,24,31,24,17,31` all bucketed to 32. Raw leaves used
three compiled shapes and took `0.115000834` seconds. Late `jnp.pad` reduced
the leaf cache to one shape but took `0.305495292` seconds because padding
primitives still compiled per logical shape. Early host padding used one leaf
shape and took `0.035626083` seconds. These observations justify placement;
they are not timing gates.

### 4.2 Dummy-source contract

Real sources remain first and unchanged. Appended slots:

- repeat one finite visible altitude, azimuth, and direction-cosine tuple;
- copy a valid positive reference frequency and other domain-sensitive finite
  metadata;
- set Stokes I/Q/U/V, per-channel Stokes/flux, rotation measure, spectral
  coefficients, and morphology coefficients to exact zero where zero is
  neutral and valid; and
- never create a NaN, infinity, zero denominator, below-horizon direction, or
  out-of-domain beam/Jones input.

The repeated direction may be evaluated by analytic or FITS beams and every
configured direction-dependent term, but its coherency or Stokes-I weight is
exactly zero. Tests must exercise that whole route, not just a synthetic
contraction.

For HEALPix Planck conversion, convert the logical temperatures first and
append zero flux. Do not invent a dummy thermodynamic temperature. Keep the
logical visible-pixel count for logs and the kernel count for scheduling and
evidence.

### 4.3 Honest trace and memory claim

P-b bounds backend source-axis shapes to power-of-two buckets. Complete leaf
signatures still vary with dtype, polarized versus unpolarized structure,
scalar versus array envelope, and P-a's full versus tail baseline chunk.
Neither host control flow nor every possible JAX primitive is claimed to have
one trace.

Early padding can increase full solver-resident `J_p`, `J_q`, phase, and
envelope arrays by less than two times along the source axis. P-a does not
remove that input cost. The retained evidence must report both the trace
reduction and this memory tradeoff.

### 4.4 Private evidence control

The shared private source-bucket helper accepts an internal policy used only by
tests and the evidence harness. Production callers always select
`pow2_compiled_v1`; the harness may select `identity_reference_v1` to obtain
matched unbucketed point and HEALPix rows from the same implementation SHA.
The unexported core point and HEALPix solver functions each accept a
keyword-only `_source_bucket_policy` with the production default and pass it to
that helper. The evidence harness calls those same complete core solvers with
the identity value; `RIMESimulator` and public configuration expose no route to
the keyword. This avoids monkeypatching, solver duplication, and changes to
whether the backend compiles.

## 5. P-c — deterministic backend selection

### 5.1 Automatic selection

`get_backend("auto")` resolves precision and returns NumPy when NumPy can honor
it. If NumPy cannot honor the request, it raises
`BackendNotAvailableError`. It never imports, initializes, or probes JAX and
never auto-selects Dask.

There is no truthful import-free way to prove that an installed JAX build can
use ambient accelerator hardware. A physical GPU inventory is not a JAX
capability probe. Automatic selection is therefore deterministic; explicit
discovery is separate.

### 5.2 Explicit discovery and strict devices

`list_backends()` and `get_backend_info()` are explicit discovery operations
and may import JAX. GPU and TPU queries must be isolated so failure of one does
not erase truthful availability of the other.

`JAXBackend(device=None)` uses JAX's runtime-default device. Explicit `cpu`,
`gpu`, and `tpu` requests are strict. `get_backend("jax")` passes `None`;
`get_backend("gpu")`, `get_backend("tpu")`, and explicit `device=` calls never
fall back to CPU and preserve the runtime failure as the cause of a typed
`BackendNotAvailableError`.

Generic `get_device_resources()` reports physical hardware from platform and
vendor tools. It must not import JAX as a fallback. Otherwise a normal
`Simulator.setup()` with `backend="auto"` still pays the import that P-c is
intended to remove. JAX device discovery belongs only to explicit backend
discovery.

Cold-path evidence runs fresh processes and records minimum, median, and
maximum durations plus whether `jax` appeared in `sys.modules`. Fast tests gate
the resolved backend and no-import property, never elapsed time.

## 6. P-d — conservative simulator capability

`VisibilitySimulator.supports_gpu` defaults to `False`.
`RIMESimulator.supports_gpu` remains explicitly false. A future simulator may
return true only when its exact implementation is named by an independently
accepted end-to-end accelerator record. Importability, a physical GPU, or one
backend kernel is not sufficient.

Because this is a pre-v1 behavior change for third-party subclasses, it must
appear in the Unreleased changelog and migration guide.

## 7. PERF-001 evidence schema

The existing `BenchmarkDocument`, `BenchmarkRecord`, `RetracingRecord`, and
`MemoryScalingRecord` are frozen v1 types. WP-7 adds separate strict v2 types
under `Perf001EvidenceDocument`, whose top-level identity is
`radiosim.benchmark.perf001.v1`.

### 7.1 Exact common and document fields

The document has exactly `schema_version`, `workload_benchmarks`,
`memory_scaling`, `solver_memory`, `retracing`, and `backend_resolution`.
Unknown or missing fields fail validation.

The schema-version literals are exact:

- document: `radiosim.benchmark.perf001.v1`;
- provenance: `radiosim.benchmark.perf001.provenance.v1`;
- workload: `radiosim.benchmark.perf001.workload.v2`;
- memory scaling: `radiosim.benchmark.perf001.memory_scaling.v2`;
- solver memory: `radiosim.benchmark.perf001.solver_memory.v1`;
- retracing: `radiosim.benchmark.perf001.retracing.v2`; and
- backend resolution: `radiosim.benchmark.perf001.backend_resolution.v1`.

Each top-level collection is an ordered JSON array of its corresponding row
type. Memory-scaling rows pair `unchunked_reference` with
`chunked_production`; solver-memory and retracing rows pair
`unbucketed_reference` with `bucketed_production`, keyed by `comparison_id`.
Workload rows are grouped by workload and input identity. Backend-resolution
rows are grouped by operation and request.

`Perf001Provenance` has exactly:

- `schema_version`, `recorded_at_utc`, `radiosim_version`, `git_sha`, and
  `working_tree_clean`;
- `platform`, `machine`, `cpu_model`, and `cpu_count_logical`;
- `python_version`, `numpy_version`, `jax_version`, `jaxlib_version`, and
  `dask_version`; and
- `pixi_environment` and `pixi_lock_sha256`.

Versions come from distribution metadata without importing JAX. Missing
optional distributions use the literal `not-installed`. Accepted records
reject an unknown or dirty source and a wrong lock digest.

`MeasurementContext` has exactly `backend_requested`, `backend_actual`,
`backend_version`, `device_kind`, `compilation_used`, `precision_preset`,
`precision_default`, `precision_accumulation`, `precision_output`,
`result_dtype`, `policy_id`, `input_identity_sha256`, and
`measurement_limitations`. The identity is SHA-256 over a versioned canonical
JSON fixture manifest plus the ordered C-contiguous bytes, names, shapes, and
dtypes of every logical scientific input. Matched rows must carry the same
identity; independent acceptance rebuilds the digest from the retained fixture
definition. Every row embeds the same provenance value; the document rejects
heterogeneous provenance.

Counts and byte sizes are nonnegative JSON integers; durations and ratios are
finite nonnegative JSON numbers; flags are JSON booleans; digests are lowercase
64-hex strings; and ordered arrays are not sets. The only nullable values are
an absent operand's paired shape/dtype, CPU `accelerator`/`device_memory`,
optional `raw_jax_memory_stats`, and the unbounded target described below.

### 7.2 Memory records

`MemoryScalingRecordV2` contains matched `unchunked_reference` and
`chunked_production` rows. Each row has exactly:

- `schema_version`, `provenance`, `context`, `comparison_id`, and
  `implementation_state`;
- `measurement_scope`, `allocator`, `includes_backend_native_allocations`,
  `inputs_preallocated`, `includes_solver_input_construction`, and
  `includes_output_reassembly`;
- `logical_n_baselines`, `logical_n_sources`, `logical_pair_count`,
  `kernel_n_sources`, and `target_kernel_pairs`;
- `kernel_baseline_chunks`, `kernel_pair_counts`, `max_kernel_pair_count`, and
  `synthetic_input_bytes_excluded`; and
- `peak_host_bytes` and `notes`.

Validation requires comparison-identical inputs, chunks summing to the logical
baseline count, pair products matching their chunk dimensions, and scope
`contraction_wrapper_python_heap_including_output_assembly`. The pair-count
fields, not the whole wrapper peak, mechanize the leaf-working-set bound.
`target_kernel_pairs` is JSON null only for `unchunked_reference` and is the
positive integer `131072` for `chunked_production`.

`SolverMemoryRecord` has exactly:

- `schema_version`, `provenance`, `context`, `comparison_id`,
  `implementation_state`, `measurement_scope`, and `allocator`;
- `includes_backend_native_allocations`, `includes_simulator_setup`,
  `includes_solver_input_construction`, and `includes_output_assembly`;
- `solver`, `sky_representation`, `logical_n_baselines`,
  `logical_source_counts`, `kernel_source_counts`, `n_times`, and
  `n_frequencies`; and
- `target_kernel_pairs`, `bucket_policy`, `peak_host_bytes`, and `notes`.

This scope is the direct solver step including horizon selection, bucketing,
Jones construction, contraction, and output assembly, but excluding fixture
and public-API setup. `tracemalloc` still excludes native JAX/device memory.
Both solver-memory rows use the positive production target `131072`; their
`bucket_policy` values are respectively `identity_reference_v1` and
`pow2_compiled_v1`.

### 7.3 Retracing records

`ContractionSignatureObservation` has exactly `jones_p_shape`,
`jones_q_shape`, `coherency_shape`, `phase_shape`, `envelope_shape`,
`stokes_i_shape`, the corresponding six `*_dtype` fields, `call_count`,
`first_call_seconds`, and `minimum_repeat_call_seconds`. A nullable operand has
an explicitly null shape and dtype.

`RetracingRecordV2` contains matched `unbucketed_reference` and
`bucketed_production` rows for the synthetic wrapper and real point and
HEALPix solver paths. Each row has exactly:

- `schema_version`, `provenance`, `context`, `comparison_id`,
  `implementation_state`, `measurement_scope`, `solver`, and
  `sky_representation`;
- `bucket_policy`, `padding_location`, `logical_source_counts`,
  `kernel_source_counts`, `distinct_logical_source_counts`, and
  `distinct_kernel_source_counts`;
- `observed_signatures`, `distinct_signature_count`, and `leaf_call_count`;
- `scope_step_seconds`, `scope_total_seconds`,
  `max_first_to_repeat_ratio`, `retrace_overhead_seconds`, and `notes`.

A compile spy records the complete public leaf inputs; private JAX cache state
is not evidence.

### 7.4 Backend-resolution records

`BackendResolutionRecord` has exactly `schema_version`, `provenance`,
`context`, `comparison_id`, `implementation_state`, `operation`,
`requested_backend`, `resolved_backend`, `discovery_policy`,
`fresh_process_samples`, `cold_seconds`, `minimum_seconds`, `median_seconds`,
`maximum_seconds`, `jax_distribution_installed`,
`jax_in_sys_modules_before`, `jax_in_sys_modules_after`,
`jaxlib_in_sys_modules_before`, `jaxlib_in_sys_modules_after`, and `notes`.
Required operations include direct `auto`, default device-resource discovery,
and a minimal `Simulator.setup()`.

### 7.5 Workload and accelerator records

`AcceleratorFacts` has exactly `vendor`, `model`, `runtime`,
`driver_version`, `compute_capability`, `total_memory_bytes`, `pci_bus_id`,
`device_uuid_sha256`, `jax_device_id`, `jax_device_kind`,
`visible_device_count`, `wheel_versions`, and `allocator_environment`.

`DeviceMemoryMeasurement` has exactly `method`, `sampling_scope`,
`sample_interval_seconds`, `sample_count`, `total_bytes`,
`used_bytes_before`, `free_bytes_before`, `used_bytes_after_setup`,
`free_bytes_after_setup`, `peak_observed_used_bytes`,
`used_bytes_after_transfer`, `free_bytes_after_transfer`,
`raw_jax_memory_stats`, and `limitations`. The peak is a sampled observation,
not an allocator-exact value. The live PID selects samples during generation
but is not retained as reproducible provenance.

`WorkloadBenchmarkRecordV2` has exactly:

- `schema_version`, `provenance`, `context`, `accelerator`, `device_memory`,
  and `workload`;
- `n_antennas`, `n_baselines`, `n_point_sources`, `n_healpix_pixels`,
  `n_times`, `n_frequencies`, `sky_representation`, `solver_workers`, and
  `loader_max_workers`;
- `setup_seconds`, `compile_seconds`, `steady_state_median_seconds`,
  `steady_state_min_seconds`, `steady_state_max_seconds`,
  `steady_state_iterations`, `host_transfer_seconds`, and `peak_host_bytes`;
- `host_memory_method`, `reference_backend`, `max_absolute_deviation`,
  `max_relative_deviation`, `tolerance_rtol`, `tolerance_atol`,
  `within_tolerance`, `unmeasured`, and `notes`.

CPU rows require null accelerator and device-memory values. GPU rows require
both structures, real model/driver/runtime/memory provenance, and may not list
`gpu` as unmeasured.

### 7.6 Generation and retention

A dedicated standard-library validation/generation tool may build the record,
but it must fail closed on a dirty or unknown source, stale lock, wrong Pixi
environment, wrong output path, or existing target. Generation uses one clean
implementation commit. The record is added by a direct evidence-successor
commit naming that source. Tests authenticate every tracked PERF-001 reference
file rather than silently choosing the lexicographically first JSON.

New records live only at
`output/benchmarks/reference/perf001/<UTC>-<host>.json`, namespaced away from
the frozen historical v1 artifact.

No absolute timing threshold is a test. Schema completeness, clean provenance,
NumPy correctness, source-order preservation, expected shape reduction, and
the contraction-temporary bound are gates.

## 8. P-e — GPU-ready infrastructure and external gate

### 8.1 Current facts

The reviewed host is an Apple M1 Max with a 24-core Metal GPU, but locked JAX
`0.10.2` exposes CPU only. No tracked workflow targets an accelerator or
self-hosted label, and no authorized external GPU host is part of this
`PERF-001` design record. Read-only cloud checks found no configured GPU path:
the authenticated Google Cloud projects have Compute Engine disabled, while
AWS and Azure command-line configuration is absent. No service was enabled or
changed.

For `PERF-001`, the official JAX installation guide states that NVIDIA CUDA
wheels are Linux only, recommends `jax[cuda13]`, and does not support macOS GPU
execution:
<https://docs.jax.dev/en/latest/installation.html>. The CUDA 13 wheel requires
an NVIDIA GPU of compute capability 7.5 or newer and a sufficiently recent
driver (currently at least 580). These are environment requirements, not
evidence that RadioSim has run there.

### 8.2 Non-gating Pixi environment

GPU-ready infrastructure uses a Linux-only `jax-gpu` feature with the pinned
PyPI requirement `jax[cuda13]==0.10.2`. The `gpu` environment combines
`py311` and `jax-gpu`, omits `jax-cpu`, and uses a separate `gpu-py311` solve
group. It cannot share `py311`: the standard feature deliberately constrains
`jaxlib` to a `cpu*` conda build, while the official CUDA extra supplies the
PyPI plugin/PJRT stack. Pixi combines constraints within a solve group, so
sharing those mutually exclusive requirements would be false reproducibility;
see <https://pixi.prefix.dev/latest/reference/pixi_manifest/>.

Default, `py312`, and `crossval` environment package identities must be
unchanged after lock regeneration. No device-named public Python extra is
added.

The environment owns two non-gating tasks:

- a strict GPU preflight; and
- `bench-gpu`, which runs the existing complete performance workload matrix
  with `RADIOSIM_BENCHMARK_BACKENDS=numpy,gpu` and
  `RADIOSIM_REQUIRE_ACCELERATOR=gpu`.

Once that task exists, the canonical locked `gpu`-environment invocation
selects `bench-gpu`; the root CPU `bench` task is unchanged.

No GitHub Actions workflow is added. A CPU-only host fails preflight; it never
skips and never produces a record.

### 8.3 Strict preflight and future record

The preflight requires:

- exact clean source, current lock, and the `gpu` Pixi environment;
- Linux x86-64, successful `nvidia-smi`, NVIDIA compute capability at least
  7.5, and driver at least 580;
- exact `jax`, `jaxlib`, `jax-cuda13-plugin`, and `jax-cuda13-pjrt` versions
  `0.10.2`, with no `LD_LIBRARY_PATH` override;
- exactly one selected JAX-visible GPU, actual backend/device kind `gpu`, and
  never CPU fallback;
- x64 enabled plus a compiled complex128 contraction synchronized on that
  device and equal to NumPy under the existing predicate; and
- numeric total, used, and free device-memory provenance.

The future GPU artifact records hardware, driver/CUDA runtime, JAX stack,
setup/warm-up, compilation, five-or-more steady-state samples, host transfer,
Python-heap peak and its limitations, numeric device-memory peak, exact workload
dimensions, and the unchanged NumPy correctness predicate. A slower GPU result
is still evidence; it supports no acceleration claim.

Device memory uses a dedicated untimed iteration with a process and
selected-GPU sampler based on
`nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory`. During generation,
the sampler binds the current PID to the selected UUID. The record stores only
the hashed UUID, sample interval, sample count, `peak_observed_used_bytes`,
total/used/free snapshots, and raw
`JAX Device.memory_stats()` when available. This is an observed sampled peak,
not an allocator-exact peak, and it is excluded from steady timings. Missing
numeric samples fail the required run.

The minimum external resource is an already-authorized Linux NVIDIA host
compatible with the locked CUDA 13 JAX environment, with permission to realize
the Pixi lock. Numeric memory provenance and the complex128 smoke are preflight
checks; the unchanged full benchmark is the fit test, and an out-of-memory
condition is a failed run rather than an invented VRAM threshold. Until such a
host or equivalent supported accelerator is named and accessible, P-e remains
blocked, `supports_gpu` stays false, and `PERF-001` stays ROADMAP.

## 9. Red-first and acceptance matrix

### 9.1 P-a

Tests cover polarized/unpolarized, complex64/complex128, scalar/array envelope,
zero baselines/sources, one chunk, exact boundaries, multiple chunks, uneven
tails, several private target values, and the unbounded before-control.
NumPy and Dask require dtype, shape, `array_equal`, and `tobytes()` identity.
JAX keeps dtype and the existing tolerance. A compile spy proves one compile
site and that the source axis is never split or reordered.

For `S <= target`, every observed leaf has at most 131072 pairs. For larger
`S`, the record must state the one-baseline exception. At retained large
fixtures, matched evidence must show a lower production wrapper peak than the
unbounded control and must separately prove the bounded leaf pair count. The
record must also show the remaining baseline-dependent output/assembly growth;
no complete-wrapper plateau is required or claimed.

### 9.2 P-b

Bucket boundaries include `1,2,3,4,7,8,9,16,17,31,32,33`. Tests prove the
power-of-two rule and `<2*S` bound, finite dummy metadata, exact zero signal,
separate logical/kernel counts, and unchanged zero-visible handling.

Full point and HEALPix routes cover polarized and unpolarized skies, Gaussian
morphology, per-channel flux, analytic/FITS beams, configured direction-
dependent and baseline Jones terms, and multiple solver-worker counts. NumPy
and Dask remain byte-identical because they are unpadded. JAX keeps its existing
parity predicate. Same-host evidence must show fewer complete leaf shapes and
less measured retrace overhead for both synthetic and real solver scopes,
without turning the timing value into a fast-test threshold.

### 9.3 P-c

Fresh subprocesses prove that importing RadioSim, `get_backend("auto")`,
generic device-resource discovery, and minimal automatic Simulator setup leave
`jax` absent from `sys.modules`. Mock accelerator availability does not change
automatic NumPy selection. Explicit discovery may import JAX. Generic JAX uses
the runtime default; unavailable explicit GPU/TPU requests raise and never
return a CPU backend. Precision failures and broken runtimes remain typed and
truthful.

### 9.4 P-d and P-e

A minimal concrete simulator that does not override `supports_gpu` inherits
false. Every registered simulator is audited; any future true value must name a
tracked accepted record.

GPU manifest, preflight, task, environment, failure, dtype, device-memory, and
CPU-lock-invariance behavior are unit-tested without inventing a measurement.
Only real hardware may generate the GPU artifact.

## 10. Phase-separated writable authority

### Design commit

- `docs/development/perf001_runtime_mitigations.md`
- `docs/index.rst`
- the live WP-7 priority, design, Q4, and status-ledger text in
  `PostTier8RemediationPlan.md`

This commit changes no `Fix.md` issue-register status, source, test, lock, or
benchmark and makes no implementation-acceptance claim.

### Implementation and evidence-scaffold commits

- `src/radiosim/core/contraction.py`
- a private helper under `src/radiosim/core/`
- `src/radiosim/core/visibility.py`
- `src/radiosim/core/visibility_healpix.py`
- `src/radiosim/api/simulator.py`
- `src/radiosim/backends/__init__.py`
- `src/radiosim/backends/jax_backend.py`
- `src/radiosim/utils/device.py`
- `src/radiosim/simulator/base.py`
- `src/radiosim/simulator/rime.py`
- `src/radiosim/benchmarks/record.py`
- `src/radiosim/benchmarks/harness.py`
- `src/radiosim/benchmarks/__init__.py`
- focused unit, characterization, parity, performance, and documentation tests
- `README.md`, `CLAUDE.md`, backend/quickstart docs, migration guide, and the
  Unreleased changelog
- `pixi.toml`, `pixi.lock`, `tools/wp7_gpu_preflight.py`, and
  `tests/unit/test_perf001_gpu_environment.py`

No workflow, public device extra, fingerprint, tolerance, or historical
benchmark file is writable.

### Clean CPU evidence successor

- one new exact-source PERF-001 JSON under `output/benchmarks/reference/`
- its validator/digest constants and reproduction instructions

The evidence successor must directly parent the clean implementation source.

### Acceptance and closure

Independent P-a-through-P-d acceptance may update the memo and live WP-7 plan
status to `CPU ACCEPTED; P-e hardware-gated`, and unlock WP-8's E3 dependency.
It must not set `PERF-001` to DONE. Final register closure may edit `Fix.md` only
after real P-e evidence is independently accepted or a separately authorized
scope re-adjudication occurs.

## 11. Terminal conditions for the CPU slice

The CPU and readiness slice is accepted only after:

1. the authored design diff is independently accepted before production edits;
2. red-first tests fail for the intended reasons;
3. P-a through P-d and GPU-ready infrastructure pass focused tests in default
   and Python 3.12 environments;
4. NumPy/Dask identity, JAX parity, point/HEALPix routes, and shipped
   characterization fingerprints pass without pin changes;
5. the clean implementation SHA produces the strict retained PERF-001 record;
6. the direct evidence successor is independently authenticated;
7. lint, format, type, doctest, strict Sphinx, non-slow, and performance schema
   gates pass;
8. exact-SHA CI passes quality, backend parity, and all six compatibility cells;
9. a separate reviewer returns `ACCEPT`; and
10. docs and live plan say exactly `CPU ACCEPTED; P-e hardware-gated`, while
    `PERF-001`, `supports_gpu`, and acceleration claims remain open/false/absent.

That acceptance completes every unblocked WP-7 obligation. It does not replace
the missing accelerator run.
