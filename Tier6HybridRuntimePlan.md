# Tier 6 Hybrid Runtime and Backend Completion Plan

## 1. Identity, status, and governing sources

| Fact | Value |
|---|---|
| Status | Design accepted. Tier 6A (characterization, dependency contract, baseline fingerprints) independently accepted. Tier 6B (worker configuration schema and resolved runtime) independently accepted, 2026-07-30. Tier 6C (loader worker behavior and offline policy) independently accepted, 2026-07-30. Tier 6D (solver accumulation restructure) independently accepted, 2026-07-30. Tier 6E (solver worker policy and `run()` signature) independently accepted, 2026-07-30. Tier 6F (hybrid sky representation and canonical summation) independently accepted, 2026-07-31. Tier 6G (hybrid serialization, HDF5 3.0.0, summary, and standard formats) independently accepted, 2026-07-31. Tier 6H (backend registry truthfulness, parity, and compilation boundary) independently accepted, 2026-07-31. Tier 6I (benchmark harness, records, and documentation truth) is now the only authorized implementation slice. |
| Date | 2026-07-31 |
| Repository | `/Users/kartikmandar/MacProjects/RadioSim` |
| Branch | `main` |
| Baseline | `6928f59` (`docs(feeds): accept Tier 5 integration`) |
| Baseline parent | `09320d8` (`docs(feeds): add receptor migration note`) |
| Governing roadmap | `Fix.md` §4, §5 (`RUN-001`..`RUN-004`), §7.2, §7.3, §7.4, §15 |
| Prior accepted architecture | `Tier2InstrumentPlan.md`, `Tier3BeamObservabilityPlan.md`, `Tier4ResultOutputPlan.md`, `Tier5ReceptorFeedPlan.md` |
| Repository policy | `CLAUDE.md`, `AGENTS.md`, including the pre-v1 direct-replacement policy |
| Issues in scope | `RUN-001` (OPEN), `RUN-002` (OPEN), `RUN-003` (OPEN), `RUN-004` (ROADMAP) |
| Issues explicitly out of scope | `SCI-001`, `SCI-002`, `SCI-003` (Tier 7); `DOC-001`..`DOC-008` (Tier 8), except the narrow backend-truth obligation of §26 |

This document is the governing implementation specification for Tier 6. It is
not Tier 6 implementation. Every characterization statement below was taken from
the working tree at `6928f59` and every cited line number is true at that
commit. Two claims in this plan were established by execution rather than by
reading (§7 D9 and §9.3); both record the exact command and output. Where a
fact could not be established at this gate it is recorded as an open question in
§41 rather than asserted.

`CLAUDE.md`'s "Implementation Status" section predates Tiers 2-5 and is stale in
at least three respects that matter here (it says only K and E carry physics,
that `JonesChain` is the only backend-routed composition, and that
`spherical_harmonic` "passes config validation but raises `NotImplementedError`
at runtime"). This plan follows source and the `Fix.md` acceptance records, not
`CLAUDE.md`, and §26 does not authorize a `CLAUDE.md` rewrite — that is Tier 8
work except for the three lines §26.4 names.

## 2. Design-only authority

This gate changes exactly two tracked files:

- this file, `Tier6HybridRuntimePlan.md`, newly added at the repository root;
- one current-status note appended to `Fix.md`.

It adds no production behavior, no test, no fixture, no configuration value, no
dependency, no lockfile change, no generated artifact, and no CI behavior. It
does not modify the `Fix.md` §5 issue-register rows and does not modify any
prior acceptance record. Every implementation slice in §32 requires a separate
authorization and an independent acceptance after its implementation commit.

## 3. Tier 0-5 dependency and acceptance state

Tiers 0-5 are independently accepted. `REL-001`, `REL-002`, `CFG-001`..`CFG-003`,
`INS-001`..`INS-003`, `BEAM-001`..`BEAM-003`, `OBS-001`, `OBS-002`,
`OUT-001`..`OUT-006`, `POL-001`, and `POL-002` are `DONE`. Tier 6 preserves
without reopening:

- the one strict configuration resolution pipeline and its centralized
  precedence, including the settled backend precedence
  (`io/config_resolution.py:1444-1445`, `core/runtime_config.py:279-300`,
  `io/config.py:1894-1902`);
- the canonical frozen instrument, antenna, baseline-selection, and beam state;
- the canonical `ObservationTimeGrid` and `PhaseCenter`, and the single
  authoritative time grid shared by both solvers
  (`api/simulator.py:956`, `:974`);
- the immutable `SimulationResult` / `LoadedSimulationResult` lifecycle,
  ownership, equality, and `(T, B, F, 4)` public shape
  (`core/result.py:721-789`, `:1111-1118`);
- the resolved receptor set, the `C`/`H` Jones mathematics, the canonical chain
  order, and the data-driven correlation axis
  (`core/visibility.py:702-764`, `core/visibility_healpix.py:86-114`);
- failure-before-side-effect ordering for configuration, instrument, beam,
  observability, solver, result, and writer work;
- the four collision policies, staged run directory, atomic publish, and
  manifest.

Tier 6 changes **how many sky components contribute to one result**, **which
concurrency knobs exist and what they do**, and **which array operations execute
inside a backend**. It does not change the correlation axis meaning, the array
shape, the time grid, the phase convention, the receptor mathematics, or the
workflow transaction.

## 4. What Tier 6 will not claim

This section is binding on every slice, every commit message, every docstring,
and every acceptance record in the tier.

1. **No GPU performance claim.** The development host is macOS arm64 and CI is
   CPU-only across all six locked jobs (`.github/workflows/ci.yml:22-47`). No
   CUDA, ROCm, Metal, or TPU device is exercised anywhere in Tier 6. Per
   `Fix.md` §15, "GPU claims require a real accelerator run; CPU-only collection
   or skipped GPU tests are not sufficient evidence." Tier 6 therefore produces
   **zero** GPU numbers and states plainly that GPU acceleration remains
   undemonstrated.
2. **No end-to-end speedup claim for JAX.** JAX-CPU in the current architecture
   is expected to be *slower* than NumPy for small workloads. Tier 6 measures
   and publishes whatever it measures, including regressions.
3. **No claim that Numba compiles anything in the solver.** §15 decides this
   truthfully (see §7 D8 and §15).
4. **No claim of distributed execution.** Dask cluster creation is not a
   RadioSim compute path in Tier 6.
5. **No claim that a worker knob makes anything faster.** The tier's obligation
   is that each surviving knob has *observable, tested, deterministic* behavior
   and honest provenance — not that it is beneficial.
6. **No scientific-accuracy improvement claim.** Every Tier 6 change to the
   point-source path must be bit-identical to the baseline except where a
   declared breaking change (§36) says otherwise; hybrid adds a component that
   was previously dropped or lossily converted, which changes numbers for
   affected configurations by design.

## 5. Current source inventory

### 5.1 Hybrid sky inventory

| Fact | Site |
|---|---|
| `SkyModel` is a frozen dataclass whose docstring states both payloads may be populated simultaneously ("a hybrid model") | `core/sky/containers/model.py:75-81` |
| `point: PointSourceData \| None`, `healpix: HealpixData \| None` are independent optional fields | `core/sky/containers/model.py:101-102` |
| `_validate_state` requires *at least one* payload and never rejects both being present | `core/sky/containers/model.py:192-205` |
| `formats` returns a `set[SkyFormat]` adding each populated payload independently | `core/sky/containers/model.py:517-528` |
| `n_sky_elements_for(representation)` returns one count for one asked-for representation; it has no hybrid total | `core/sky/containers/model.py:537-548` |
| `as_point_source_arrays()` raises when `point is None`, hinting at lossy materialization | `core/sky/containers/model.py:746-763` |
| `has_polarized_healpix_maps` consults only the HEALPix payload | `core/sky/containers/model.py:560-563` |
| `prepare_sky_model(models, *, options=None, **overrides)` is the public combine entry point | `core/sky/combine/pipeline.py:29-34` |
| `resolve_target_representation` returns `None` — the hybrid signal — only when no explicit representation was requested and inputs span both formats | `core/sky/combine/engine.py:70`, `:79-96`; `pipeline.py:74-78` |
| With an explicit representation the pipeline forwards a concrete target and `_combine_models` never reaches `_combine_as_hybrid` | `pipeline.py:87-105`, `engine.py:385-391` |
| `_combine_as_hybrid` exists, keeps both piles, and performs no lossy conversion | `core/sky/combine/engine.py:212-239` |
| HEALPix-target combination rasterizes point contributors into the map cube via the merge path | `core/sky/combine/engine.py:110-135`, `:414-424` |
| Point-target combination raises for a HEALPix-only contributor unless `allow_lossy_point_materialization=True` | `core/sky/combine/concat.py:240-246` |
| Point-target combination silently uses only `m.point` for a *hybrid* contributor — its maps are dropped with no error and no warning | `core/sky/combine/concat.py:240`, `:253-254` |
| The pipeline returns a hybrid unchanged when the requested format is already present on the combined model | `pipeline.py:112-115`, `:126-127` |
| Single-model fast path bypasses `_combine_models` entirely | `pipeline.py:84-88` |
| `materialize_healpix_model(..., clear_other=False)` / `materialize_point_sources_model(..., clear_other=False)` can construct a hybrid deliberately | `core/sky/operations/operations.py:42-58`, `:181-195` |
| `HealpixData` sparse support, dense gate, and frequency accessors | `core/sky/containers/healpix.py:316-319`, `:413-475`, `:349-373` |
| Existing hybrid unit tests cover container/combine construction only, never visibilities | `tests/unit/test_core/test_sky_hybrid.py:134`, `:169-173` |

### 5.2 Worker and concurrency inventory

| Fact | Site |
|---|---|
| `Simulator.run(self, progress: bool = True, n_workers: int \| None = None)` | `api/simulator.py:847-851` |
| The `n_workers` docstring still advertises "Number of parallel workers (default: auto)" | `api/simulator.py:859-860` |
| Passing `n_workers` raises `NotImplementedError` naming Tier 6 as the remediation target | `api/simulator.py:875-880` |
| Sky loading hard-codes `max_workers=8` at the only call site | `api/simulator.py:773-787`, specifically `:782` |
| `load_models_parallel(loaders, max_workers: int = 8, ...)` also defaults to 8 | `core/sky/operations/parallel.py:116-122`, specifically `:118` |
| Pool size is `min(len(loaders), max_workers)` | `core/sky/operations/parallel.py:159` |
| Results are written to a pre-sized list by request index, so loader output order is deterministic regardless of completion order | `core/sky/operations/parallel.py:160`, `:183`, `:190`, `:212` |
| Executor choice is registry-driven: `"process"` when any requested loader's category is in `{diffuse, synthetic, file}`, else `"thread"` | `core/sky/operations/parallel.py:62`, `:70-97` |
| A `"process"` request silently degrades to threads with a `logger.warning` when kwargs fail a pickle probe | `core/sky/operations/parallel.py:100-113`, `:164-172` |
| Loader failures are aggregated into `SkyLoadAggregateError` under `strict=True` | `core/sky/operations/parallel.py:40-50`, `:201-217` |
| The only other `ThreadPoolExecutor` in the package is inside source subtraction, unrelated to simulation orchestration | `core/sky/operations/subtraction.py:763` |
| `NumbaBackend` accepts `n_workers` and may create a Dask `LocalCluster` + `Client`; `get_backend` forwards it from `**kwargs` | `backends/numba_backend.py:102`, `:159-179`; `backends/__init__.py:210-214` |
| No worker or concurrency field exists anywhere in the configuration schema | `io/config.py:1469-1483` (`ExecutionConfig` is `backend`, `precision`, `simulator`, `offline` only) |
| No worker value is recorded in any provenance snapshot | `core/result.py:116-201`, `core/runtime_config.py:279-300` |
| `execution.offline` reaches `Simulator` and is used only for the printed status banner and required-service warnings | `api/simulator.py:181`, `:667`, `:674-695` |
| `get_network_status(offline=True)` returns a forced-offline status **without** populating the module cache | `utils/network.py:267-289` |
| `is_online()` reads/writes a module-global `_cached_status` and performs a live socket probe on a cache miss | `utils/network.py:172-199` |
| Loaders enforce network availability through `require_service()`, which calls `is_online()` — not through any offline value threaded from the resolved configuration | `utils/network.py:344-399`; callers at `core/sky/loaders/vizier/core.py:219`, `vizier/racs.py:200`, `vizier/inspect.py:217`, `:307`, `loaders/diffuse.py:271`, `:615` |

### 5.3 Backend inventory

| Fact | Site |
|---|---|
| `ArrayBackend` abstract surface: `name`, `xp`, `is_available`, `asarray`, `to_numpy`, `matmul`, `conjugate_transpose`, `exp`, `sin`, `cos`, `free_memory`, `memory_info`, `get_device_info` | `backends/base.py:28`, `:143-176`, `:182-207`, `:345-415`, `:496-532` |
| Concrete helpers include `zeros_complex`, `batch_eye`, `set_at`, `sum`, `conj`, `real`, `imag`, `synchronize` | `backends/base.py:209-291`, `:300-309`, `:439-449`, `:538-544` |
| `set_at` is functional for JAX (`arr.at[index].set(value)`) and in-place for NumPy | `backends/base.py:300-309` |
| **`jit`, `vmap`, and `jit_compile` are not on the abstract surface at all** — `jit`/`vmap` exist only on `JAXBackend`, `jit_compile` only on `NumbaBackend` | `backends/base.py:28-567` (absent); `backends/jax_backend.py:336-345`, `:358-369`; `backends/numba_backend.py:422-437` |
| Nothing in `src/` calls `jit`, `vmap`, or `jit_compile` | verified by search over `src/radiosim/` |
| `JAXBackend.__init__` unconditionally enables x64 before device selection | `backends/jax_backend.py:101-104` |
| `JAXBackend.to_numpy` is the single host-transfer helper (`np.asarray`) | `backends/jax_backend.py:174-184` |
| `JAXBackend.synchronize()` blocks on a freshly created throwaway constant, not on the caller's array — it cannot be used to time a real computation | `backends/jax_backend.py:324-330` |
| `NumbaBackend` imports `jit, prange`; `prange` is never used; `jit` is used only inside `jit_compile()` to wrap a **caller-supplied** function | `backends/numba_backend.py:40`, `:422-437` |
| No `@njit`, `@jit`, `@vectorize`, or `@guvectorize` decorator exists anywhere in `src/radiosim/` | verified by search |
| Every `NumbaBackend` array operation delegates to NumPy, or to `dask.array` when Dask arrays are in use | `backends/numba_backend.py:210-224`, `:245-259`, `:274-285` |
| The class docstring claims mode `'cpu'` is "Local CPU with JIT and parallel loops" | `backends/numba_backend.py:72-97` |
| The module docstring already concedes that `mode="gpu"` only validates a device and that array operations remain NumPy/Dask | `backends/numba_backend.py:9-12` |
| `get_backend("auto")` precedence is JAX TPU → JAX GPU → Numba CUDA → Numba CPU → NumPy, with per-tier exceptions swallowed | `backends/__init__.py:157-197` |
| Explicit backend names reject an unsatisfiable precision rather than downgrading it | `backends/__init__.py:79-93`, `:199-260` |
| `list_backends()` probes availability including `jax_gpu`/`jax_tpu` | `backends/__init__.py:263-296` |
| `RIMESimulator.supports_gpu` returns unconditional `True` | `simulator/rime.py:144-146` |
| `RIMESimulator`'s class docstring still prints the pre-Tier-5 chain order `J = B @ G @ D @ P @ E @ T @ Z @ K` | `simulator/rime.py:45-53` |
| Point solver: backend-routed `asarray`/`exp`/`matmul`/`conjugate_transpose`/`sum`/`set_at`/`zeros_complex`; host-side astropy + NumPy for coordinates, horizon filtering, direction cosines, and Gaussian coefficients | `core/visibility.py:337-340`, `:347-351`, `:413-427`, `:430-478`, `:599-638` |
| Point solver writes the full output cube one `(time, baseline, freq)` cell at a time inside the innermost loop | `core/visibility.py:588-594`, `:630-638` |
| HEALPix solver: same backend routing for the contraction, plus bare `np.*` for horizon masking, direction cosines, Stokes casting, and the Planck conversion branch | `core/visibility_healpix.py:332-335`, `:356-373`, `:412-427`, `:437-447`, `:503-514`, `:466-487`, `:528-549` |
| Both solvers iterate time on the host and re-enter astropy per time step | `core/visibility.py:413-419`; `core/visibility_healpix.py:345-353` |
| The HEALPix solver rebuilds the constant `H_p @ C_p` transforms inside the time loop | `core/visibility_healpix.py:376-386` |
| Point-vs-NumPy and HEALPix-vs-NumPy Numba parity tests exist; the JAX parity tests exist but skip | `tests/unit/test_core/test_visibility_backend.py:243`, `:299-301`, `:405` |

### 5.4 Precision and backend interaction inventory

| Fact | Site |
|---|---|
| `float128`/`complex256` platform probes run once at import | `core/precision.py:76-98` |
| Requesting `float128` on `jax`/`numba` warns and falls back to `float64` | `core/precision.py:130-138`, `:171-178` |
| `validate_for_backend` aggregates every offending field per backend | `core/precision.py:757-815` |
| The `ArrayBackend.precision` setter converts those strings into real `UserWarning`s | `backends/base.py:63-73` |
| `get_backend` turns them into a hard `BackendNotAvailableError` for any explicit backend name | `backends/__init__.py:79-93` |
| `get_backend("auto")` diverts a `float128` request straight to NumPy, skipping JAX and Numba | `backends/__init__.py:157-168` |
| Config-time rejection of `float128` under explicit `jax`/`numba` already exists | `io/config.py:1894-1902` |
| Presets: `standard` all-float64; `fast` float32 with float64 islands; `precise`/`ultra` introduce `float128` | `core/precision.py:480-620` |

### 5.5 Performance, test, and CI inventory

| Fact | Site |
|---|---|
| `tests/performance/` contains only `__init__.py` — there is no benchmark | `tests/performance/` |
| `tests/integration/` contains only `__init__.py` — there is no integration test | `tests/integration/` |
| `performance`, `integration`, `slow`, and `gpu` are all registered markers | `pyproject.toml:155-160` |
| `addopts` does not include `--strict-markers` | `pyproject.toml:141-145` |
| The only test task is `test = "python -m pytest tests/"`; there is no benchmark task | `pixi.toml:9-24`, specifically `:23` |
| `numba` is a top-level conda dependency (`>=0.64,<0.67`), justified by PySM's needs, and is installed in every environment | `pixi.toml:29-32` |
| **`jax` is not declared anywhere in `pixi.toml`** and is not installed in either environment | `pixi.toml:26-61`, `:69-77` |
| Exactly six tests skip for JAX unavailability, all via `pytest.importorskip("jax")`. **Correction (2026-07-30, Tier 6A acceptance):** `pytest -rs` attributes each skip to the line where `importorskip` actually executes, not to the calling test body. `test_backend_jones.py` and `test_visibility_backend.py` route through a shared `_get_optional_backend()` helper, so both of `test_backend_jones.py`'s skips report at line 20 (not `:116`/`:132`, which are merely the helper's two call sites) and `test_visibility_backend.py`'s one skip reports at line 88 (not `:301`, its one call site). `test_sky_backend.py:123` is a literal `pytest.importorskip("jax")` inside the test body itself (distinct from that file's own unused helper-level import at line 24), so its citation was already exact. | `tests/unit/test_backends/test_jax_backend.py:15`; `tests/unit/test_jones/test_backend_jones.py:20` (2 skips; helper called from `:116`, `:132`); `tests/unit/test_core/test_sky_backend.py:123`; `tests/unit/test_core/test_visibility_backend.py:88` (1 skip; helper called from `:301`); `tests/unit/test_core/test_sky_spectral.py:556` |
| CI runs `pixi run test -- -m "not slow"` across six OS/Python jobs plus one quality job; no GPU runner and no performance job exist | `.github/workflows/ci.yml:22-47`, `:63-67`, `:69-99` |
| The characterization convention is `test_tier<N>_current_behavior.py` with per-test "Pins"/"Characterizes"/"Records" docstrings and `OWNED BY: Tier Nx` markers for pins a later slice may flip | `tests/characterization/test_tier5_current_behavior.py:1-7` |
| `README.md` is already honest about incomplete backend coverage and refuses to imply GPU execution | `README.md:59-65`, `:338-344` |

## 6. Current data-flow trace

`Simulator.setup()` → `_setup_after_instrument_state()`:

1. `sky_representation = visibility_config["sky_representation"]` — always the
   concrete literal `"point_sources"` or `"healpix_map"`
   (`api/simulator.py:582`, `io/config.py:1374-1376`).
2. Backend constructed once from the settled strategy (`api/simulator.py:615`).
3. Loader requests assembled (`api/simulator.py:730-771`).
4. `load_models_parallel(loader_requests, max_workers=8, ..., executor=recommend_executor_for_loaders(...))`
   (`api/simulator.py:773-787`).
5. `prepare_sky_model(..., representation=sky_representation, ...)`
   (`api/simulator.py:800-812`).
6. `if sky_mode == SkyFormat.HEALPIX: self._source_arrays = None` else
   `self._source_arrays = self._sky_model.as_point_source_arrays()`
   (`api/simulator.py:833-837`). **This is the point-vs-HEALPix fork on the
   setup side.**

`Simulator.run()`:

7. `n_workers` rejected (`api/simulator.py:875-880`).
8. `if _sky_mode == SkyFormat.HEALPIX and self._sky_model is not None:`
   → `calculate_visibility_healpix(...)`; `else:` →
   `solver.calculate_visibilities(...)` (`api/simulator.py:947-977`).
   **This is the point-vs-HEALPix fork on the solve side, and it is exclusive:
   exactly one solver runs per `run()`.**
9. Both branches return one backend cube of shape `(T, B, F, 2, 2)`
   (`core/visibility.py:337-340`, `core/visibility_healpix.py:332-335`).
10. `build_simulation_result(receptor_visibilities=..., ...)` transfers once,
    flattens to `(T, B, F, 4)`, hardens, and fingerprints
    (`core/result.py:1081-1118`, `:1135-1157`).
11. `SolverResultProvenance(sky_representation="healpix_map" | "point_sources", execution_path="polarized" | "scalar")`
    (`api/simulator.py:1021-1028`), whose literal set is enforced at
    `core/result.py:176`, `:186-187` and re-enforced on read at
    `core/result.py:976-987`.

Consequences that follow directly from steps 5, 6, and 8:

- A single loader that returns a hybrid model (`pyradiosky_file`,
  `skyh5_multifile`, `test_sources`, `poisson_confusion`) takes the
  single-model fast path (`pipeline.py:84-88`) and the requested-format
  early return (`pipeline.py:113-115` or `:126-127`), so the hybrid survives
  into `Simulator`. Step 6 then keeps exactly one payload and **the other
  payload contributes nothing to the visibilities, with no error and no
  warning**.
- Multiple loaders spanning both formats never produce a hybrid under the
  config path, because step 5 always passes a concrete target. Instead, either
  point contributors are rasterized into the map cube
  (`engine.py:414-424`) or a HEALPix contributor forces a hard error unless
  `allow_lossy_point_materialization` is set (`concat.py:240-246`) — the
  "lossy conversion" that `Fix.md` §7.3 rejects.
- A hybrid contributor inside a multi-model point-target combine loses its maps
  silently (`concat.py:240`, `:253-254`).

## 7. Confirmed defect matrix

| # | Defect | Evidence | Issue |
|---|---|---|---|
| D1 | The high-level API cannot express a hybrid sky; `sky_representation` admits only two literals | `io/config.py:1374-1376` | `RUN-003` |
| D2 | Exactly one solver runs per `run()`, so `V_total = V_point + V_healpix` is unreachable from the high-level API | `api/simulator.py:947-977` | `RUN-003` |
| D3 | A surviving second payload is silently discarded at setup: HEALPix mode nulls `_source_arrays`, point mode reads only the point payload | `api/simulator.py:833-837` | `RUN-003` |
| D4 | A hybrid contributor in a point-target combine loses its maps with no diagnostic | `core/sky/combine/concat.py:240`, `:253-254` | `RUN-003` |
| D5 | Mixed inputs under an explicit representation are forced through lossy conversion (point rasterization) or a hard error, never summed | `engine.py:414-424`, `concat.py:240-246` | `RUN-003` |
| D6 | Loader concurrency is hard-coded to 8 with no configuration surface and no provenance | `api/simulator.py:782`, `core/sky/operations/parallel.py:118` | `RUN-002` |
| D7 | `run(n_workers=...)` still exists as a public parameter with a misleading docstring, rejected at runtime; there is no typed solver-execution policy anywhere | `api/simulator.py:847-851`, `:859-860`, `:875-880`; `io/config.py:1469-1483` | `RUN-001` |
| D8 | The `numba` backend compiles nothing: `prange` unused, no kernel decorators, all operations delegate to NumPy/Dask, yet the class docstring advertises "JIT and parallel loops" | `backends/numba_backend.py:40`, `:72-97`, `:210-285`, `:422-437` | `RUN-004` |
| D9 | `get_backend("auto")` selects `NumbaBackend` named `"numba-cpu"` whose `xp is numpy`, so `actual_backend` provenance misreports the executing implementation | `backends/__init__.py:157-197`; verified: `pixi run python -c "from radiosim.backends import get_backend; b=get_backend('auto'); print(b.name, type(b).__name__)"` → `numba-cpu NumbaBackend`, `xp is numpy: True` | `RUN-004` |
| D10 | `RIMESimulator.supports_gpu` is unconditionally `True` | `simulator/rime.py:144-146` | `RUN-004` |
| D11 | Output accumulation is one `set_at` per `(time, baseline, frequency)`, which is a full functional copy of the whole cube on JAX — `O(T·B·F)` cube copies | `core/visibility.py:630-638`; `core/visibility_healpix.py:480-487`, `:545-549`; `backends/base.py:300-309` | `RUN-004` |
| D12 | The HEALPix solver retains bare `np.*` on sky-data arrays and recomputes the constant receptor transforms inside the time loop | `core/visibility_healpix.py:356-373`, `:412-447`, `:503-514`, `:376-386` | `RUN-004` |
| D13 | `JAXBackend.synchronize()` blocks on a throwaway constant, so no honest device timing is possible with the current surface | `backends/jax_backend.py:324-330` | `RUN-004` |
| D14 | `jit`/`vmap`/`jit_compile` are backend-private and uncallable from backend-agnostic code | `backends/base.py` (absent), `jax_backend.py:336-369`, `numba_backend.py:422-437` | `RUN-004` |
| D15 | There is no benchmark, no benchmark task, and no performance test, so no acceleration claim in the repository can be reproduced | `tests/performance/`, `pixi.toml:9-24` | `RUN-004` |
| D16 | `jax` is not an installable dependency of any pixi environment, so the mandated NumPy/JAX parity evidence cannot be produced without a dependency change | `pixi.toml:26-61`, `:69-77`; six skips in §5.5 | `RUN-004` |
| D17 | `execution.offline` never reaches loader enforcement: `require_service()` consults a module-global cache that a forced-offline status does not populate, and that a process-pool worker does not inherit | `utils/network.py:267-289`, `:172-199`, `:344-399` | `RUN-002` |
| D18 | A `"process"` executor request degrades to threads on a pickle failure with only a log warning, and neither the request nor the degradation is recorded | `core/sky/operations/parallel.py:100-113`, `:164-172` | `RUN-002` |
| D19 | `Simulator.get_memory_estimate()` counts only point sources, so it under-reports for HEALPix and would under-report for hybrid | `api/simulator.py:1508` | `RUN-003` |
| D20 | `RIMESimulator`'s docstring advertises the pre-Tier-5 chain order | `simulator/rime.py:45-53` | `RUN-004` (documentation-truth) |

D17 is a live truthfulness defect independent of workers, but §15's required
tests include "offline/network loader behavior under worker execution", so Tier 6
owns it.

## 8. Design decision 1 — hybrid representation in the high-level API

### 8.1 Decision

`visibility.sky_representation` gains a third literal:

```python
sky_representation: Literal["point_sources", "healpix_map", "hybrid"] = "point_sources"
```

`"hybrid"` resolves to `prepare_sky_model(representation=None, ...)`, which is
the existing hybrid-preserving path (`pipeline.py:74-78`, `:107-110`,
`engine.py:389-391`). No new combine machinery is written; `_combine_as_hybrid`
already reduces each pile independently with
`allow_lossy_point_materialization=False` (`engine.py:228-239`).

`SkyFormat` gains **no** `HYBRID` member. Hybrid is not a payload
representation; it is a *solve mode*. The canonical in-memory signal stays
`SkyModel.formats == {POINT_SOURCES, HEALPIX}` (`model.py:517-528`).

### 8.2 Rejection rules that make the mode truthful

Under `Fix.md` §4.2, a mode that silently degrades is not acceptable. Three new
rejections follow:

1. `sky_representation: hybrid` whose combined model carries only one payload is
   rejected. A hybrid request that runs one component is exactly the
   "validate, ignore, imply" pattern §4.2 forbids.
2. `sky_representation: point_sources` whose combined model still carries a
   HEALPix payload is rejected, closing D3/D4. The message names `hybrid` and
   the existing `allow_lossy_point_materialization` escape.
3. `sky_representation: healpix_map` whose loader set includes a point-source
   contributor is rejected unless the new
   `visibility.allow_lossy_point_rasterization: bool = False` is set, closing
   the silent half of D5. This preserves the existing rasterization capability
   behind an explicit opt-in, symmetric with
   `allow_lossy_point_materialization`.

Exact messages are in §18.3. Rule 3 is a breaking change for any configuration
that mixes a point catalog with a diffuse map under `healpix_map`; the two
shipped configurations are unaffected (`configs/config.yaml:66` and
`configs/receptor_circular_example.yaml:76` are `point_sources`;
`configs/realistic_foreground_example.yaml:66` is `healpix_map` with a diffuse
recipe source).

### 8.3 Component identity

A hybrid run has exactly two components, in this fixed order:

| Order | Component name | Solver | Payload |
|---|---|---|---|
| 1 | `point` | `RIMESimulator.calculate_visibilities` → `core.visibility.calculate_visibility` | `sky.point` |
| 2 | `healpix` | `core.visibility_healpix.calculate_visibility_healpix` | `sky.healpix` |

The order is fixed for reproducibility of the summation (§9.2) and of every
provenance record. It is not configurable.

### 8.4 What is shared, by construction

Both components must receive the *same objects*, not equivalent copies:

| Input | Shared value | Site today |
|---|---|---|
| Instrument view | one `SolverInstrumentView` | `api/simulator.py:628-630` |
| Selected baselines | that view's `selected_pairs` / `baseline_vectors_enu_m` | `core/visibility.py:334`, `core/visibility_healpix.py:322` |
| Beam system | one `BeamSystem` | `api/simulator.py:954`, `:969` |
| Receptors | one `ResolvedReceptorSet` | `api/simulator.py:961`, `:975` |
| Time grid | one `ObservationTimeGrid` | `api/simulator.py:956`, `:974` |
| Frequencies | one `self._frequencies_hz` | `api/simulator.py:957`, `:971` |
| Location | one `EarthLocation` | `api/simulator.py:955`, `:973` |
| Backend | one `ArrayBackend` | `api/simulator.py:615` |
| Phase convention | `radiosim.rime-zenith-drift.v1` | `api/simulator.py:1026` |

The hybrid orchestrator passes the identical references to both calls. A test
asserts object identity for the instrument view, beam system, receptor set, and
time grid across the two component calls (§27 H4).

## 9. Design decision 2 — component summation and one canonical result

### 9.1 Decision

Components are summed **in the backend array domain**, before
`build_simulation_result`, producing exactly one `SimulationResult` per `run()`:

```text
V_receptor = V_point + V_healpix        # backend arrays, shape (T, B, F, 2, 2)
result     = build_simulation_result(receptor_visibilities=V_receptor, ...)
```

Rationale:

- one host transfer, one dtype cast, one finiteness check, one flatten, one
  fingerprint — the entire Tier 4 hardening path is reused unchanged
  (`core/result.py:1081-1118`);
- no second `SimulationResult` is ever constructed, so there is no ambiguity
  about which result is canonical, no result-level arithmetic to define, and no
  new equality semantics;
- additivity is exact for the accumulation dtype, because both cubes are
  produced in `backend.get_complex_dtype("output")` before summation
  (`core/visibility.py:326`, `:630-633`;
  `core/visibility_healpix.py:299`, `:541-544`).

`ArrayBackend` gains one concrete helper `add(a, b)` (`self.xp.add(a, b)`)
so the summation is backend-routed rather than relying on Python `+` on an
unknown array type.

### 9.2 The additivity invariant

For any configuration whose combined model is hybrid, with everything else
identical:

```text
V_hybrid == V_point_only + V_healpix_only
```

where `V_point_only` is obtained by running the same configuration with
`sky_representation: point_sources` on a model whose HEALPix payload was
removed, and symmetrically for `V_healpix_only`. §27 H1 specifies the exact
test construction. Because the two component cubes are the same arrays in both
experiments and floating-point addition of two given values is deterministic,
the required equality is **bit-identical for the NumPy backend**, not merely
within tolerance. The looser statement in `Fix.md` §15 ("within precision
tolerance") is satisfied strictly. For JAX-CPU the tolerance of §13.4 applies to
each component; additivity itself remains exact.

### 9.3 Empty-component behavior

A component whose payload is present but contributes nothing above the horizon
returns a zero cube, not a special case: the point solver returns
`visibilities` unchanged when no source is above the horizon
(`core/visibility.py:425-427`) and pre-allocates zeros
(`:337-340`); the HEALPix solver behaves the same way
(`core/visibility_healpix.py:356-358`). Summation therefore needs no
empty-component branch. A payload that is present but *empty* (zero sources,
zero pixels) is still a valid hybrid component; only an *absent* payload
triggers the §8.2 rule 1 rejection.

### 9.4 Result-model consequences

| Change | Site |
|---|---|
| `SolverResultProvenance.sky_representation` gains `"hybrid"` | `core/result.py:176`, `:186-187` |
| `SolverResultProvenance` gains `components: tuple[str, ...]` and `component_element_counts: tuple[int, ...]` | `core/result.py:171-201` |
| The loaded-snapshot field-set check learns the two new keys | `core/result.py:976-983` |
| `ResultPerformance` gains `solver_point_seconds` and `solver_healpix_seconds` (plain floats, so the existing `fields(self)` normalization and coherence check extend naturally) | `core/result.py:204-248` |
| The coherence check additionally requires `solver_point_seconds + solver_healpix_seconds <= solver_seconds + allowance` | `core/result.py:233-243` |

Component **timings** must stay out of the scientific fingerprint because they
are nondeterministic. They are safe where placed: `_scientific_hash` consumes
`solver_snapshot` but never `performance` (`core/result.py:510-549`, `:1135-1150`),
and `_provenance_hash` consumes the backend snapshot, resolved config,
configuration provenance, and history but never `performance`
(`core/result.py:551-567`, `:1151-1157`). Component **names and counts** are
deterministic and deliberately *do* enter `scientific_sha256` through
`solver_snapshot`, so a hybrid result can never collide with a single-component
result over the same instrument and sky numbers.

For a single-component run, `components` is `("point",)` or `("healpix",)`, and
the unused timing field is `0.0`. This means every result — including
single-component ones — gets a new `scientific_sha256`. That is declared in §36.

### 9.5 Provenance additions summary

```python
SolverResultProvenance(
    solver="rime",
    sky_representation="hybrid",              # or "point_sources" / "healpix_map"
    convention="radiosim.rime-zenith-drift.v1",
    execution_path="polarized",               # see below
    components=("point", "healpix"),
    component_element_counts=(1234, 49152),
)
```

`execution_path` keeps its two-literal set. For a hybrid run it is `"polarized"`
whenever either component ran polarized; the point component is always
polarized (`api/simulator.py:964`) so a hybrid run is always `"polarized"`. The
per-component scalar/polarized detail is not added: it would duplicate what
`component_element_counts` and the HEALPix payload's own polarization state
already express, and it would grow the fingerprint surface for no scientific
gain.

## 10. Design decision 3 — double-counting policy for hybrid

### 10.1 Decision

Hybrid **reuses the existing disjointness gate unchanged**. No new physical rule
is invented in Tier 6. Specifically, `check_physical_disjointness` already runs
unconditionally inside `_combine_models` before any dispatch
(`engine.py:377-382`), and it is the same gate for the hybrid target as for the
single-format targets:

- monopole consistency always runs and is never bypassed
  (`disjointness.py:140-212`, `:264`);
- the three point-vs-diffuse pass rules (fully subtracted; subtracted above the
  catalog's completeness floor after power-law scaling; angular-scale
  separation) run unless `mixed_model_policy == "allow"` or
  `assume_disjoint=True` (`disjointness.py:40-137`, `:266-297`);
- `assume_disjoint=True` emits its `UserWarning` stating that monopole
  consistency was still enforced (`disjointness.py:266-279`).

### 10.2 Why this is the correct scope

Hybrid does not create a new double-counting risk; it *reveals* the one the
existing gate was written for. Under the current code the same physical overlap
either gets rasterized into one cube (`engine.py:414-424`) or aborts
(`concat.py:240-246`); under hybrid it becomes a literal sum of two components.
The overlap semantics are identical, so the rule set is identical. Inventing a
second, hybrid-only rule set would create two disagreeing definitions of
disjointness.

### 10.3 What Tier 6 adds

One thing only: the resolved disjointness *decision* becomes observable.
`SkyProvenance` already carries the fields the gate reads; Tier 6 records in
the summary JSON and in the resolved-config snapshot which policy was in force
(`mixed_model_policy`, `assume_disjoint`) and whether the `assume_disjoint`
escape was actually exercised. A hybrid run with `assume_disjoint: true` must
therefore be identifiable after the fact from the artifacts alone.

### 10.4 What is explicitly not added

- no footprint-overlap geometry check (`merge.py:47-71` unions coverage for
  provenance only and is not a gate — Tier 6 keeps it that way);
- no automatic flux subtraction, source masking, or catalog-hole filling;
- no per-component `mixed_model_policy`; the existing single value plus the
  per-source `RealisticForegroundSourceConfig.mixed_model_policy`
  (`io/config.py:947`) are unchanged.

## 11. Design decision 4 — worker policy

### 11.1 Decision

Two independent, typed, separately named policies replace one ambiguous
parameter, exactly as `Fix.md` §7.2 requires. Both live under `execution`
because both are execution policy, not scientific input, and `execution` is
already the section that carries backend and precision.

```python
class SkyLoadingConfig(StrictFrozenModel):
    """Loader-side concurrency policy for sky-model acquisition."""
    max_workers: int | None = None                              # None => auto
    executor: Literal["auto", "thread", "process"] = "auto"

class SolverExecutionConfig(StrictFrozenModel):
    """Solver-side concurrency policy for visibility computation."""
    workers: int = 1
    executor: Literal["thread"] = "thread"

class ExecutionConfig(StrictFrozenModel):
    backend: Literal["auto", "numpy", "jax", "dask"] = "numpy"
    precision: PrecisionInput = ...
    simulator: Literal["rime"] = "rime"
    offline: bool = False
    sky_loading: SkyLoadingConfig = Field(default_factory=SkyLoadingConfig)
    solver: SolverExecutionConfig = Field(default_factory=SolverExecutionConfig)
```

### 11.2 Loader policy semantics

- `max_workers: None` means auto and resolves to
  `min(len(loader_requests), os.cpu_count() or 1, 8)`. The cap of 8 is retained
  as an *auto ceiling* rather than a hard-coded constant, because the current
  behavior is exactly that ceiling and the tier's job is to make it
  configurable and recorded, not to change performance. The resolved integer is
  recorded; `None` never reaches the loader driver.
- `max_workers: 1` is a legal, tested value and must dispatch through the same
  code path with a pool of one, not through a special serial branch. This makes
  "worker count has no effect on results" a property of one code path.
- `max_workers: 0` or negative is rejected at schema level.
- `executor: "auto"` keeps the registry-driven choice
  (`recommend_executor_for_loaders`, `parallel.py:70-97`). `"thread"` and
  `"process"` force the choice.
- The hard-coded `max_workers=8` at `api/simulator.py:782` and the default
  `max_workers: int = 8` at `parallel.py:118` are both removed;
  `load_models_parallel` takes `max_workers: int` with **no default**, so no
  caller can silently inherit a number again.
- The silent thread degradation (D18) becomes a *recorded* degradation: the
  driver returns, alongside the models, a small frozen record
  `LoaderExecutionRecord(requested_executor, actual_executor, max_workers, degraded_reason)`.
  When `executor: "process"` was requested explicitly and the pickle probe
  fails, the run is **rejected**, not degraded — an explicit request that
  silently becomes something else is the §4.2 pattern. Under `"auto"` the
  degradation is permitted and recorded with its reason.

### 11.3 Solver policy semantics

- `workers: 1` (default) is serial and bit-identical to the baseline.
- `workers: N > 1` parallelizes **over the time axis only**. Each worker
  computes a contiguous block of time indices and returns its own
  `(T_block, B, F, 2, 2)` cube; the orchestrator concatenates blocks in time
  order. This is possible only after the §12 accumulation restructure.
- `executor` accepts `"thread"` only. `"process"` is rejected with a typed
  message: the solver closure holds a `BeamSystem` with FITS handlers and
  astropy objects, which the loader pickle probe would fail on, and a
  degradation-to-threads for the solver would repeat D18 in a worse place.
- `workers` greater than the number of time samples is clamped to the time-sample
  count, and the clamp is recorded. `workers: 0` or negative is rejected at
  schema level.
- Time blocks are contiguous and computed by an explicit deterministic
  partition function (`n_times`, `workers`) → tuple of `(start, stop)` pairs,
  which is unit-tested independently of any pool.

### 11.4 Why the time axis and not baselines or frequencies

- The time loop already re-derives everything it needs per step: astropy
  transform, horizon mask, direction cosines
  (`core/visibility.py:413-478`; `core/visibility_healpix.py:345-373`). It is
  the only axis with no cross-iteration state.
- Each time index writes a disjoint output slice
  (`core/visibility.py:634-638`), so no reduction is reordered and no partial
  sum is split. Bit-identity is a structural property, not a hope.
- Baseline and frequency parallelism would either split the source reduction or
  duplicate the per-antenna Jones cache (`core/visibility.py:571-585`), trading
  determinism or memory for nothing measurable on this host.

### 11.5 Determinism guarantee

| Axis of variation | Guarantee |
|---|---|
| Loader `max_workers` (any value) | bit-identical result; loader output order already index-preserving (`parallel.py:160`, `:183`, `:212`) |
| Loader `executor` (`thread` vs `process`) | bit-identical result |
| Solver `workers` (any value) | bit-identical result on a fixed backend, because time blocks are disjoint and no reduction is repartitioned |
| Backend (`numpy` vs `jax`) | tolerance-bounded, not bit-identical (§13.4) |
| Precision preset | not comparable; different dtypes by design |

Bit-identity is asserted through `SimulationResult.scientific_sha256`
(`core/result.py:1135-1150`), which is the strongest available statement and
already the tier-wide convention.

## 12. Design decision 5 — the fate of `run(n_workers=...)`

### 12.1 Decision

`n_workers` is **removed** from `Simulator.run()`. The new signature is:

```python
def run(self, *, progress: bool = True) -> SimulationResult:
```

`progress` becomes keyword-only. Solver concurrency is expressed only through
`execution.solver.workers`, which reaches `Simulator` through the resolved
runtime configuration like every other execution policy (§4.3 precedence:
one centralized source, recorded in provenance, never created by mutation order
in `setup()`).

### 12.2 Why removal rather than wiring it up

Keeping a Python keyword argument alongside a typed config field would create
exactly the accidental precedence `Fix.md` §4.3 forbids: two sources for one
value, resolved by argument order at a call site. Tier 1 settled that every
execution policy is resolved once, centrally, and recorded. `run()` is not a
configuration surface.

### 12.3 Migration boundary

Passing `n_workers` raises Python's own `TypeError`
(`run() got an unexpected keyword argument 'n_workers'`). This is the accepted
Tier 5 precedent for removed constructor and method keywords, and it is
unambiguous. The actionable guidance goes where a user looks it up:

- one entry in `docs/migration_guide.md` naming the replacement field;
- one test asserting the `TypeError` and its parameter name;
- the current `NotImplementedError` block (`api/simulator.py:875-880`) is
  deleted along with the parameter and its docstring lines
  (`:859-860`).

`Simulator.run()` is not called with `n_workers` anywhere in `src/`, `tests/`,
`examples/`, or `docs/` at the baseline, so removal breaks no in-tree caller.

## 13. Design decision 6 — backend completion scope

### 13.1 What "backend completion" means in Tier 6

Not "the whole simulation runs on the device". Completion here is four
verifiable properties:

1. **Registry truthfulness** — every selectable backend name describes what
   actually executes, and `actual_backend` provenance never misreports the
   implementation (closes D8, D9, D10).
2. **Path parity** — both solver paths route the same categories of operation
   through the backend, and both agree between NumPy and JAX-CPU within a
   stated tolerance (closes D12, and the HEALPix half of `RUN-004`).
3. **Transfer and copy discipline** — the accumulation pattern that makes a
   functional-array backend pathological is removed, and the reduction is
   *measured on CPU JAX* (closes D11).
4. **Documented boundaries** — every remaining host-side stage is named in one
   place, with a stated reason, so "partially integrated" stops being a vague
   disclaimer (closes the documentation half of `RUN-004`).

### 13.2 Paths that become backend-routed

| Operation | Today | Tier 6 |
|---|---|---|
| HEALPix direction cosines | `np.cos/np.sin` on host (`visibility_healpix.py:368-370`) | computed on host from astropy output, then a single `backend.asarray` per time step (unchanged in spirit, but the boundary is named) |
| HEALPix Stokes casting and RJ scaling | mixed `np.*` / backend (`visibility_healpix.py:412-459`) | backend-routed after one host cast |
| HEALPix Planck branch | `np.zeros` + masked host loop (`visibility_healpix.py:437-447`, `:503-514`) | stays host-side, explicitly named as host preprocessing (it is a masked scalar transform of sky data, not a hot array op) |
| Constant `H_p @ C_p` transforms | rebuilt inside the time loop (`visibility_healpix.py:376-386`) | hoisted above the time loop, computed once |
| Output accumulation, both solvers | `set_at` per `(t, b, f)` cell | per-time block assembly, one stack per solver (§13.3) |
| Component summation | absent | `backend.add` (§9.1) |

### 13.3 The accumulation restructure

Both solvers change shape as follows, with no change to any computed number:

```text
for each time index t:
    host preprocessing (astropy, horizon mask, direction cosines)   # named stage
    for each frequency index f:
        build the per-antenna Jones cache                            # unchanged
        assemble one (B, 2, 2) block for all baselines at (t, f)
    assemble one (F, B, 2, 2) block for time t
collect T time blocks -> one (T, B, F, 2, 2) cube in one operation
```

Consequences:

- NumPy: identical arithmetic, identical results, fewer indexing operations;
- JAX: `O(T·B·F)` whole-cube functional copies become `O(1)` assemblies, which
  is the single largest structural obstacle to any device execution and is
  measurable on CPU JAX by counting assembly operations and by wall clock;
- solver time-axis parallelism (§11.3) becomes expressible, because a worker's
  product is a self-contained time block.

`ArrayBackend` gains one concrete helper `stack(arrays, axis=0)`
(`self.xp.stack(arrays, axis=axis)`) for the assembly. `set_at` remains on the
surface; it is simply no longer used in the solver hot path.

**Correction (2026-07-30, Tier 6D implementation):** the sketch's per-time block
shape is wrong and the two statements around it cannot both hold. No single
`stack` of `T` blocks of shape `(F, B, 2, 2)` produces `(T, B, F, 2, 2)`: on
`axis=0` it produces `(T, F, B, 2, 2)`, which needs a further transpose and
returns a non-contiguous cube, contradicting "in one operation". The per-time
block is therefore assembled as `(B, F, 2, 2)` — `stack(freq_blocks, axis=1)`
over the `F` baseline blocks — so that the final `stack(time_blocks, axis=0)`
produces the canonical `(T, B, F, 2, 2)` cube exactly, contiguously, and in one
operation. Only the intermediate axis order changes; every binding property of
the sketch is preserved (one `(B, 2, 2)` block per `(t, f)`, one block per `t`,
one whole-cube assembly, no change to any computed number), so this is a
notation correction and not a decision change. A time step with no source or
pixel above the horizon contributes a pre-zeroed `(B, F, 2, 2)` block, which
keeps its slot in the single final assembly without entering the frequency loop
and without any assembly of its own. Degenerate axes (`T`, `B`, or `F` equal to
zero, or an empty source batch) return the canonical zero cube directly, because
there is nothing to assemble.

### 13.4 Backend parity matrix and tolerance

| Workload | NumPy | JAX-CPU | Dask (NumPy arrays) | Requirement |
|---|---|---|---|---|
| Point, unpolarized, 1 time, 2 freq | reference | compare | compare | §13.5 tolerance |
| Point, polarized (Q, U, V ≠ 0), 2 times | reference | compare | compare | §13.5 tolerance |
| Point with Gaussian morphology | reference | compare | compare | §13.5 tolerance |
| HEALPix scalar (I only) | reference | compare | compare | §13.5 tolerance |
| HEALPix polarized | reference | compare | compare | §13.5 tolerance |
| Hybrid (point + HEALPix) | reference | compare | compare | §13.5 tolerance, plus additivity (§9.2) |
| Heterogeneous receptor bases (linear + circular) | reference | compare | compare | §13.5 tolerance |

### 13.5 Tolerance rule

Exact bit-identity across backends is not required and must not be asserted:
XLA may fuse and reorder the source reduction, so a float64 sum over `N` terms
can differ in the last bits. The rule is:

```text
|V_backend - V_numpy| <= atol + rtol * |V_numpy|
with, for float64 accumulation:  rtol = 1e-12,  atol = 1e-12 * max(1.0, max|V_numpy|)
```

Dask-with-NumPy-arrays is required to be **bit-identical** to NumPy, because it
delegates to the same NumPy operations (`backends/numba_backend.py:245-259`);
asserting anything weaker would hide a real defect. The float32 (`fast` preset)
tolerance is `rtol = 1e-5`, `atol = 1e-5 * max(1.0, max|V_numpy|)`, and is
required only for the NumPy backend against itself across the restructure, not
across backends.

### 13.6 JAX adoption boundary

**In scope:**

- `ArrayBackend` gains two capability members with safe defaults, so
  backend-agnostic code can opt into compilation without importing JAX:

  ```python
  @property
  def supports_compilation(self) -> bool: return False       # base default
  def compile(self, func): return func                       # base default: identity
  ```

  `JAXBackend` overrides them with `True` and `jax.jit`
  (existing `jit` at `jax_backend.py:336-345` becomes the implementation).
  `NumPyBackend` and the renamed Dask backend inherit the identity default.
- Exactly **one** kernel is compiled: the per-`(time, frequency)` baseline-batched
  contraction that produces one `(B, 2, 2)` block from
  `(J_p, J_q, C, phase, envelope)`. It is pure, shape-stable within a run, and
  dtype-stable. `vmap` may be used *inside* that kernel over the baseline axis;
  it is not used anywhere else.
- The uncompiled implementation of that kernel remains the reference and is
  always used by NumPy. A test asserts compiled and uncompiled agree to §13.5
  tolerance.
- Compilation is opt-in through `execution.backend: jax` only; there is no
  separate "enable jit" switch, because a backend that reports
  `supports_compilation` and then does not compile would be another §4.2
  violation.
- Compile time is measured and reported separately from steady-state time
  (§22), and `BackendResultProvenance` records whether compilation was used.

**Explicitly out of scope:**

- compiling the Jones chain, the beam evaluation (pyuvdata/FITS interpolation is
  host-side by nature), the Planck conversion, the time loop, or the HEALPix
  Stokes assembly;
- `vmap` over time or frequency;
- `jax.device_put` placement policy, donation, or sharding;
- replacing astropy coordinate transforms with a JAX implementation. Astropy is
  the accepted source of truth for coordinates; a hand-rolled device transform
  would be a scientific change disguised as a performance change.

`JAXBackend.synchronize()` is corrected to take the array whose completion is
being awaited (`synchronize(arr=None)`; with an argument it blocks on that
array, without one it keeps the current best-effort behavior), because D13
otherwise makes every JAX timing number meaningless.

## 14. Design decision 7 — backend registry truthfulness

### 14.1 Decision

| Change | Reason |
|---|---|
| `NumbaBackend` is renamed `DaskBackend`, its `name` values become `"dask-cpu"` and `"dask-distributed"` | it is a NumPy backend with optional Dask arrays; the class never compiled anything (D8) |
| Backend name `"numba"` is removed from `ExecutionConfig.backend` and from `get_backend` | it named a capability that does not exist |
| `mode="gpu"` and the CUDA validation path are removed | it validated a device and then ran NumPy (`numba_backend.py:9-12` already concedes this); keeping it is an invitation to misread provenance |
| `jit_compile()` is removed | no production caller; `numba` stops being presented as a solver technology |
| `"auto"` precedence becomes: JAX **only when a non-CPU JAX device exists** → NumPy | closes D9; the Dask backend is never auto-selected because it requires explicit opt-in |
| `RIMESimulator.supports_gpu` returns `False` | closes D10; it will return `True` when a measured accelerator run exists, which Tier 6 does not produce |
| `RIMESimulator`'s docstring chain order is corrected to the Tier 5 canonical order | closes D20 |

`numba` remains a declared conda dependency (`pixi.toml:29-32`) because PySM
needs it. Nothing in Tier 6 claims RadioSim uses it.

### 14.2 Why not write real Numba kernels

`Fix.md` §15 item 10 offers "real compiled kernels or a less misleading name".
Real kernels would mean reimplementing the polarized RIME contraction — with the
receptor terms, the beam Jones, the Gaussian envelope, and the per-channel
spectral evaluation — as a `nopython` kernel that cannot call the existing
`JonesChain`, `BeamSystem`, or astropy. That is a second scientific
implementation of the forward model, requiring its own cross-implementation
validation under §4.4. It is Tier 7-scale work, it is not what `RUN-004` asks
for, and shipping it inside a tier whose stated purpose is *truthfulness* would
be the wrong trade. Tier 6 chooses the honest name and records the deferral.

### 14.3 Provenance additions

```python
BackendResultProvenance(
    requested_backend="auto",
    actual_backend="numpy",
    requested_precision={...},
    actual_precision={...},
    result_dtype="complex128",
    device_kind="cpu",          # new: "cpu" | "gpu" | "tpu"
    compilation_used=False,     # new
)
```

`device_kind` and `compilation_used` are execution facts, not scientific ones.
`BackendResultProvenance` feeds `_provenance_hash` and not `_scientific_hash`
(`core/result.py:551-567`, `:1151-1157`), so adding them changes
`provenance_sha256` for every result and leaves `scientific_sha256` governed by
the §9.4 solver fields. Both hashes change in Tier 6; both changes are declared
in §36.

## 15. Design decision 8 — precision interaction rules

The existing rules are correct and are preserved verbatim. Tier 6 restates them
as a contract and extends them to the renamed backend:

1. An explicit backend that cannot honor the requested precision **rejects**;
   it never downgrades (`backends/__init__.py:79-93`).
2. `"auto"` with `float128` anywhere diverts to NumPy without consulting JAX
   (`backends/__init__.py:157-168`).
3. Config-time rejection of `float128` under an explicit non-NumPy backend
   remains, with `"numba"` replaced by `"dask"` in the literal set
   (`io/config.py:1894-1902`). `"dask"` inherits the rejection because its
   arrays are NumPy but its Dask path is not float128-safe; treating it as
   NumPy-equivalent for precision would be a silent capability claim.
4. Every solver kernel keeps using `backend.get_complex_dtype("output")` and
   `backend.default_real_dtype` (`core/visibility.py:326`, `:476-478`;
   `core/visibility_healpix.py:299`, `:371-373`). The restructure of §13.3 must
   not introduce a literal dtype anywhere.
5. The compiled kernel of §13.6 must produce the same dtype as its uncompiled
   reference for every preset it is exercised with; a dtype difference is a
   failure, not a tolerance question.

## 16. Design decision 9 — offline policy under worker execution

### 16.1 Decision

The resolved `execution.offline` value becomes the single authority for loader
network behavior, and it must survive both executor kinds.

```python
# radiosim/utils/network.py
def set_offline_policy(offline: bool) -> None:
    """Install the process-wide offline policy for loader network gates."""
```

- `Simulator.setup()` calls it once, immediately after computing
  `self._network_status`, before any loader runs
  (`api/simulator.py:667` is the anchor point).
- `is_online()` consults the installed policy first and returns `False` without
  a socket probe when offline is in force. The 300 s TTL cache
  (`utils/network.py:190-199`) is untouched for the online case.
- `load_models_parallel` propagates the policy into every worker: a thread pool
  shares module state; a process pool receives the boolean as part of the
  worker entry point (`parallel.py:65-67`) and installs it before resolving the
  loader callable.
- A test asserts that with `offline: true` and a network-requiring loader, the
  loader raises `ConnectionError` from `require_service`
  (`utils/network.py:379-380`) under **both** executors, and that no socket
  probe is attempted (monkeypatched `_check_socket` fails the test if called).

### 16.2 Why this belongs in Tier 6

`Fix.md` §15's required tests include "offline/network loader behavior under
worker execution". D17 shows the behavior is currently undefined: a forced
offline status does not populate the cache it would have to populate, and a
spawned worker starts with an empty module cache regardless. Making worker
policy configurable without fixing this would ship a knob whose observable
behavior includes accidental network access.

## 17. Design decision 10 — hybrid-aware ancillary surfaces

Three surfaces branch on sky mode today and must learn the third mode. Each is
small, and each is named so no slice discovers it late.

| Surface | Today | Tier 6 |
|---|---|---|
| Setup banner and run configuration table | one `sky_label` from one mode (`api/simulator.py:840-844`, `:914-932`) | prints both component counts for hybrid |
| `get_memory_estimate()` | counts only `_source_arrays` (`api/simulator.py:1508`) | sums both components' element counts (closes D19) |
| Coarse-beam-sampling advice | gated on `sky_representation == "healpix_map"` and on `sky_model.healpix is not None` (`api/simulator.py:596-599`, `:817-831`) | the payload-based gate at `:817` already covers hybrid; the representation-based pre-check at `:596` extends to hybrid |

`plan_observability` / `plot_observability` pass `self._sky_model` through
(`api/simulator.py:1352`) and need no change: they consume the model, not the
mode.

## 18. Exact configuration schema

### 18.1 Additions

```yaml
visibility:
  sky_representation: hybrid          # new third literal
  allow_lossy_point_rasterization: false   # new; gates point -> HEALPix rasterization

execution:
  backend: numpy                      # literal set changes: numba -> dask
  sky_loading:                        # new typed block
    max_workers: null                 # null => auto (min(requests, cpu_count, 8))
    executor: auto                    # auto | thread | process
  solver:                             # new typed block
    workers: 1
    executor: thread                  # thread only
```

### 18.2 Removals

| Removed | Replacement |
|---|---|
| `Simulator.run(n_workers=...)` | `execution.solver.workers` |
| `execution.backend: numba` | `execution.backend: dask` (or `numpy`) |
| `load_models_parallel(max_workers=8)` default | required argument |
| `NumbaBackend`, `jit_compile`, `mode="gpu"` | `DaskBackend` with `mode` in `{cpu, distributed}` |

### 18.3 Exact rejection messages

These strings are asserted verbatim by tests (§27 E-rows). Each is actionable
and names its replacement.

**Schema-level (`ConfigSchemaError` via the strict Pydantic path, following the
`reject_removed_output_policy` precedent at `io/config.py:1501-1542`):**

```text
execution.backend=numba: removed before v1.0; the backend never compiled any
kernel. Use execution.backend=dask for the NumPy/Dask backend or
execution.backend=numpy.

execution.n_workers: not a field; use execution.sky_loading.max_workers for
sky-loader concurrency or execution.solver.workers for solver concurrency.

execution.sky_loading.max_workers must be a positive integer or null (null
means auto).

execution.solver.workers must be a positive integer.

execution.solver.executor=process: unsupported; the solver closure holds beam
handlers and astropy objects that cannot cross a process boundary. Use
execution.solver.executor=thread.
```

**Resolution/runtime-level (typed errors, raised before any beam load, backend
allocation, or output path is created):**

```text
visibility.sky_representation=hybrid requires a sky model with both a
point-source payload and a HEALPix payload; the resolved model carries only
{formats}. Request point_sources or healpix_map, or add a source of the missing
kind.

visibility.sky_representation=point_sources would discard the HEALPix payload
carried by the resolved sky model. Request hybrid to sum both components, or set
visibility.allow_lossy_point_materialization=true to convert the HEALPix payload
to point sources.

visibility.sky_representation=healpix_map would rasterize {n} point source(s)
into the HEALPix grid, which quantizes positions to pixel centers. Request
hybrid to sum both components, or set
visibility.allow_lossy_point_rasterization=true to opt in.

execution.sky_loading.executor=process was requested explicitly, but loader
arguments for {loader} cannot be pickled: {reason}. Use
execution.sky_loading.executor=auto to allow a thread fallback, or thread to
force it.
```

### 18.4 Resolved runtime model

```python
@dataclass(frozen=True, slots=True)
class ResolvedSkyLoadingConfig:
    max_workers: int          # already resolved; never None
    executor: Literal["auto", "thread", "process"]

@dataclass(frozen=True, slots=True)
class ResolvedSolverExecutionConfig:
    workers: int              # already clamped to <= n_times
    executor: Literal["thread"]

@dataclass(frozen=True, slots=True)
class ResolvedExecutionConfig:
    backend_strategy: Literal["auto", "numpy", "jax", "dask"]
    precision: PrecisionConfig
    simulator: Literal["rime"]
    offline: bool
    sky_loading: ResolvedSkyLoadingConfig
    solver: ResolvedSolverExecutionConfig
```

Resolution happens in `io/config_resolution.py` next to the existing execution
resolution (`:1444-1445`), with `__post_init__` validation mirroring the
existing style (`core/runtime_config.py:288-296`). `to_json_safe()`
(`core/runtime_config.py:458-463`) then carries every resolved worker value into
`resolved_config`, and therefore into `provenance_sha256`, the HDF5
`resolved_config_json`, and the summary JSON, with no further work. The
`workers` clamp needs the time-sample count, which is available on the resolved
observation config, so the clamp is applied during resolution and the
pre-clamp request is recorded in `ConfigurationProvenance`.

**Correction (2026-07-30, Tier 6B independent acceptance):** "recorded in
`ConfigurationProvenance`" names the object, not its `override_origins` field.
`override_origins` values are typed as `ValueOrigin = Literal["default",
"document", "override"]` (`core/runtime_config.py:44`) and are therefore
structurally unable to carry a pre-clamp integer; the pre-clamp request is
recorded in `ConfigurationProvenance.input_snapshot` (the validated,
pre-resolution input document), while `override_origins` continues to record
only each field's document-vs-default provenance label. No decision changes;
this only disambiguates wording that read, on a first pass, as naming the
`override_origins` field specifically.

## 19. Exact serialization changes

| Artifact | Change |
|---|---|
| HDF5 | schema `2.0.0` → `3.0.0`; `provenance/solver_json` gains `components` and `component_element_counts`; `provenance/performance_json` gains `solver_point_seconds` and `solver_healpix_seconds`; `provenance/backend_json` gains `device_kind` and `compilation_used`; `2.0.0` is rejected on read with a message naming Tier 6 |
| Summary JSON | `to_summary_snapshot()` gains a `solver` block carrying `sky_representation`, `components`, and `component_element_counts`; the `performance` block (`io/summary_json.py:326`) carries the two new timings automatically; the loader-execution record and resolved worker policy appear under the execution block |
| Measurement Set / UVFITS | no schema change. The written history line gains the component list, because a summed hybrid visibility is not reconstructible from the file otherwise |
| Reader validation | `_validate_loaded_identity_snapshots` learns the new solver and backend field sets (`core/result.py:956-987`) |

No visibility array shape, dtype, correlation order, weight, or flag semantics
change anywhere in Tier 6.

**Correction (2026-07-31, Tier 6G implementation) — the summary-JSON
`schema.version` question, settled once.** Tier 6C flagged that this section
authorizes the summary document to grow while naming no version bump, and
routed the decision here (§33, Tier 6C correction). The resolution is that the
summary schema moves `1.0.0` → `1.1.0`, and that this is deliberately *not* the
major bump the HDF5 schema takes in the same slice. The two artifacts differ in
kind, and the version field must say so:

- The HDF5 bump is **major** because it is a hard incompatibility with no
  upgrade path. `_read_root_attributes` (`io/hdf5.py:1105-1106`) rejects any
  `schema_version` that is not exactly `SCHEMA_VERSION`, and
  `_validate_loaded_identity_snapshots` (`core/result.py:1175-1184`) rejects a
  `solver_snapshot` whose field set is not exactly the current one. A `2.0.0`
  file is therefore unreadable by the `3.0.0` reader *and* a `3.0.0` file is
  unreadable by the `2.0.0` reader. That is what `3.0.0` announces.
- The summary bump is **minor** because Tier 6's changes to it are purely
  additive and no reader exists to break. The document is write-only
  (`write_result_summary_json` has no in-tree counterpart reader), and every
  `1.0.0` key survives at the same path with the same meaning and the same
  type. Tier 6 adds one top-level block (`execution`, 6C), two keys inside
  `solver`, and two inside `performance` (6F, surfaced here). Calling that
  `2.0.0` would tell a consumer its parser is broken when it is not.

The version does move, rather than staying `1.0.0`, because the document shape
did change: a version that never moves while the shape does is not a contract,
and the repository's own
`test_summary_json_is_exact_bounded_metadata_contract` asserts the exact
top-level key set precisely so no shape change passes unnoticed. `1.1.0`
records the growth honestly and leaves the major number available for the first
removal or retyping.

## 20. Error taxonomy

New typed errors, each subclassing an existing tier-appropriate base so no
caller has to catch `Exception`:

| Error | Base | Raised for |
|---|---|---|
| `HybridSkyError` | `ValueError` | §18.3 hybrid rejections (missing payload; would-drop; would-rasterize) |
| `WorkerPolicyError` | `ValueError` | explicit-process pickle failure; solver executor rejection at runtime |
| `SolverPartitionError` | `RuntimeError` | a time partition that does not cover `[0, n_times)` exactly once — an internal invariant, never user-triggerable |
| `BenchmarkRecordError` | `ValueError` | a benchmark record missing any mandatory §23 field |

`BackendNotAvailableError` (`backends/base.py:22`) is reused unchanged for
backend availability and precision rejection.

### 20.1 Mandatory failure ordering

Extending the accepted Tier 5 ordering, `setup()` must fail in this order, and
each stage must leave no side effect from a later stage:

1. configuration schema and resolution (including every §18.3 schema message);
2. instrument and baseline resolution;
3. receptor resolution;
4. beam resolution and loading;
5. backend construction and precision validation;
6. offline policy installation (§16.1);
7. sky loading (worker policy in force);
8. sky combination and disjointness gate;
9. **hybrid/representation compatibility rejection (§18.3 runtime block)**;
10. solver partition validation.

Step 9 sits after combination because the decision needs the *combined* model's
payload set. Steps 1-8 must therefore be side-effect-free with respect to
output paths, which they already are: no writer runs during `setup()`.

## 21. Scientific invariants

| # | Invariant | Where proven |
|---|---|---|
| S1 | `V_hybrid == V_point + V_healpix`, bit-identical on NumPy, on identical time/frequency/baseline coordinates | §27 H1 |
| S2 | A hybrid run's coordinates are element-wise identical to both single-component runs (`time_grid`, `frequencies_hz`, `channel_widths_hz`, `selection`) | §27 H2 |
| S3 | Component order does not affect the sum beyond float associativity, and the fixed order of §8.3 makes the result reproducible run to run | §27 H3 |
| S4 | Both components receive the identical instrument view, beam system, receptor set, and time grid objects | §27 H4 |
| S5 | Disjoint and explicitly-assumed-disjoint hybrid models do not double count: total flux equals the sum of component fluxes and nothing is counted twice | §27 H5 |
| S6 | Solver `workers = 1, 2, 3, 4` yield identical `scientific_sha256` | §27 W3 |
| S7 | Loader `max_workers` and `executor` do not affect `scientific_sha256` | §27 W1 |
| S8 | The accumulation restructure is bit-identical to the baseline for every shipped configuration, compared **within one Python environment** (`py311` against the `py311` pin, `py312` against the `py312` pin) | §27 R1 |
| S9 | NumPy and JAX-CPU agree within §13.5 tolerance on all seven §13.4 workloads | §27 B1 |
| S10 | Dask-with-NumPy-arrays is bit-identical to NumPy | §27 B2 |
| S11 | The compiled kernel agrees with its uncompiled reference within §13.5 tolerance and produces the identical dtype | §27 B3 |
| S12 | An offline run performs no socket probe and fails network-requiring loaders under both executors | §27 W5 |

## 22. Performance methodology

### 22.1 Harness location and shape

- `tests/performance/test_backend_benchmarks.py` — pytest-driven, marked
  `@pytest.mark.performance` **and** `@pytest.mark.slow`, so the existing CI
  invocation (`pixi run test -- -m "not slow"`,
  `.github/workflows/ci.yml:63-67`) excludes it with no CI change and no new
  gating job. Both markers are already registered
  (`pyproject.toml:155-160`).
- `src/radiosim/benchmarks/` — the harness itself (record dataclass, timing
  discipline, JSON writer), importable and unit-tested by fast tests so the
  harness is trustworthy even when the benchmarks do not run.
- `pixi.toml` gains one task: `bench = "python -m pytest tests/performance/ -m performance"`.
- Benchmark output goes to `output/benchmarks/<UTC timestamp>-<host tag>.json`
  and is gitignored. No benchmark number is ever hard-coded into a test
  assertion; the performance tests assert *correctness and record completeness*,
  never a time threshold. This keeps them deterministic and non-flaky.

### 22.2 Timing discipline

| Rule | Reason |
|---|---|
| Setup and steady state are timed separately; the first iteration of any backend is always reported as setup | JAX compiles on first call |
| Compile time is measured as (first call) − (median steady-state call) and reported as its own field | §15 requires it separately |
| Host transfer time is measured around `backend.to_numpy` only (`core/result.py:1081-1086` already isolates it) | it is the single transfer point |
| Device completion is awaited with the corrected `synchronize(arr)` before any timer stops | otherwise JAX timings measure dispatch, not work (D13) |
| Steady state is the median of at least 5 iterations, with min and max also recorded | one sample is not a measurement |
| Peak memory is `tracemalloc` peak for host allocation, plus `backend.memory_info()` (`backends/base.py:508-519`) where available | host-only is honest and portable |
| Every record states its correctness delta against the NumPy reference for the identical workload | §15 requires the tolerance |

### 22.3 CI posture

Performance tests never gate. The verification gate of §31 runs them locally on
the implementer's and the reviewer's machines and requires that they *pass*
(records complete, correctness within tolerance) — not that they be fast. CI
continues to run only `-m "not slow"`.

## 23. Benchmark record schema

Every record carries every field in `Fix.md` §15's mandatory list. A missing or
`None` field raises `BenchmarkRecordError`; there is no partial record.

```python
@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    # identity
    schema_version: str                 # "radiosim.benchmark.v1"
    recorded_at_utc: str
    radiosim_version: str
    git_sha: str
    # hardware and accelerator
    platform: str                       # e.g. "macOS-15-arm64"
    cpu_model: str
    cpu_count_logical: int
    accelerator: str                    # "none" for every Tier 6 record
    accelerator_driver: str | None
    # backend and version
    backend_requested: str
    backend_actual: str
    backend_version: str                # numpy/jax/dask version string
    device_kind: str                    # "cpu"
    compilation_used: bool
    # precision
    precision_preset: str | None
    precision_default: str
    precision_accumulation: str
    precision_output: str
    result_dtype: str
    # problem size
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
    reference_backend: str              # "numpy"
    max_absolute_deviation: float
    max_relative_deviation: float
    tolerance_rtol: float
    tolerance_atol: float
    within_tolerance: bool
    # honesty
    unmeasured: tuple[str, ...]         # e.g. ("gpu", "tpu", "distributed")
```

`accelerator` is `"none"` and `unmeasured` includes `"gpu"` in every record Tier 6
produces. A record claiming otherwise without a corresponding hardware
description is a §37 acceptance failure.

## 24. Public API changes

```python
# radiosim.api
Simulator.run(*, progress: bool = True) -> SimulationResult   # n_workers removed
Simulator.sky_components -> tuple[str, ...]                   # new, e.g. ("point", "healpix")

# radiosim.backends
DaskBackend                    # replaces NumbaBackend
get_backend(name: Literal["auto","numpy","cpu","jax","dask","gpu","tpu"], ...)
ArrayBackend.add(a, b)                  # new concrete helper
ArrayBackend.stack(arrays, axis=0)      # new concrete helper
ArrayBackend.supports_compilation       # new property, default False
ArrayBackend.compile(func)              # new, default identity
ArrayBackend.synchronize(arr=None)      # signature widened

# removed
NumbaBackend
NumbaBackend.jit_compile
JAXBackend.jit                          # folded into ArrayBackend.compile
backend name "numba"

# radiosim.core
SolverResultProvenance(..., components, component_element_counts)   # two new fields
ResultPerformance(..., solver_point_seconds, solver_healpix_seconds)
BackendResultProvenance(..., device_kind, compilation_used)
HybridSkyError, WorkerPolicyError, SolverPartitionError             # new

# radiosim.io
ResolvedSkyLoadingConfig, ResolvedSolverExecutionConfig             # new
SkyLoadingConfig, SolverExecutionConfig                            # new
HDF5 schema "3.0.0"                                                # "2.0.0" rejected

# radiosim.benchmarks (new module)
BenchmarkRecord, BenchmarkRecordError, run_benchmark, write_record
```

`JAXBackend.vmap` stays as a JAX-specific method (used only inside the compiled
kernel) and is not promoted to the abstract surface.

## 25. Exact implementation file inventory

### 25.1 New production files

```text
src/radiosim/benchmarks/__init__.py
src/radiosim/benchmarks/record.py
src/radiosim/benchmarks/harness.py
src/radiosim/core/contraction.py            # the one compiled kernel (6H)
src/radiosim/core/hybrid.py                 # component orchestration and summation
src/radiosim/core/solver_partition.py       # deterministic time partition
```

**Correction (2026-07-31, Tier 6H implementation):** `core/contraction.py` is
added above. §13.6 authorizes exactly one compiled kernel and both solvers use
it, so it needs one home that both can import; leaving it inside
`core/visibility.py` would make `core/visibility_healpix.py` import the point
solver for a function that belongs to neither. Its own module is also what makes
"exactly one kernel is compiled" mechanically checkable: the compilation-boundary
test asserts that `backend.compile(` occurs at exactly one path in `src/`, and
that assertion is only meaningful because that path is a file whose whole
purpose is the kernel. This is a placement decision, not a scope change.

### 25.2 Modified production files

```text
src/radiosim/api/simulator.py
src/radiosim/backends/__init__.py
src/radiosim/backends/base.py
src/radiosim/backends/jax_backend.py
src/radiosim/backends/numpy_backend.py
src/radiosim/core/precision.py
src/radiosim/core/result.py
src/radiosim/core/runtime_config.py
src/radiosim/core/sky/combine/concat.py
src/radiosim/core/sky/operations/parallel.py
src/radiosim/core/visibility.py
src/radiosim/core/visibility_healpix.py
src/radiosim/io/config.py
src/radiosim/io/config_resolution.py
src/radiosim/io/hdf5.py
src/radiosim/io/summary_json.py
src/radiosim/io/standard_visibility.py
src/radiosim/simulator/rime.py
src/radiosim/utils/network.py
```

### 25.3 Renamed production file

```text
src/radiosim/backends/numba_backend.py -> src/radiosim/backends/dask_backend.py
```

### 25.4 New test files

```text
tests/characterization/test_tier6_current_behavior.py
tests/unit/test_backends/test_array_backend_helpers.py
tests/unit/test_backends/test_backend_parity.py
tests/unit/test_backends/test_compilation_boundary.py
tests/unit/test_backends/test_dask_backend.py
tests/unit/test_core/test_benchmark_record.py
tests/unit/test_core/test_hybrid_visibility.py
tests/unit/test_core/test_solver_partition.py
tests/unit/test_core/test_visibility_accumulation.py
tests/unit/test_simulator/test_worker_policy.py
tests/unit/test_utils/test_offline_policy.py
tests/integration/test_hybrid_end_to_end.py
tests/performance/test_backend_benchmarks.py
tests/unit/test_tier6_runtime_acceptance.py
```

Every file in §25.4 is new. Files that already exist and are *modified* by a
slice appear only in that slice's §33 grant.

### 25.5 Configuration, examples, documentation, manifests

```text
configs/hybrid_sky_example.yaml            # new
pixi.toml                                  # jax-cpu feature/environment + bench task
pixi.lock                                  # regenerated
pyproject.toml                             # if a jax extra is adjusted
.github/workflows/ci.yml                   # one added job running the jax-cpu environment
docs/user_guide/configuration.rst
docs/user_guide/backends.rst               # exists at the baseline; extended
docs/api/backends.rst                      # exists at the baseline; updated names
docs/migration_guide.md
CLAUDE.md                                  # only the three stale lines of §26.4
README.md                                  # only the backend-truth paragraph
```

## 26. Documentation truth obligations

Tier 6 does **not** perform the Tier 8 documentation sweep. It owns exactly the
statements its own changes make false or newly provable:

1. `README.md:338-344` — replace "incomplete backend coverage" prose with the
   measured position: which operations are backend-routed, that NumPy and
   JAX-CPU agree within a stated tolerance, that no accelerator was exercised,
   and where the benchmark records live.
2. `docs/user_guide/backends.rst` — the backend table, the `auto` precedence,
   the renamed Dask backend, the compilation boundary, and the explicit list of
   host-side stages (astropy transforms, horizon masking, Planck conversion,
   FITS beam interpolation).
3. `docs/user_guide/configuration.rst` — the three new configuration blocks and
   the `hybrid` mode, with the §18.3 rejections shown.
4. `CLAUDE.md` — exactly three corrections: the stale Jones-implementation
   sentence, the stale `jit`/`vmap` sentence, and the stale
   `spherical_harmonic` sentence (it is rejected at config validation today,
   `io/config.py:1995-2000`, not at runtime). No other `CLAUDE.md` edit is
   authorized.
5. `docs/migration_guide.md` — one entry per §36 breaking change.

Any documentation sentence asserting a speed, a GPU capability, or a
distributed capability must cite a benchmark record file, or it does not ship.

## 27. Exact test matrix

Every row is a required test. `E` rows assert an exact §18.3 string. No row may
be satisfied by a skipped test.

| # | Test | Assertion |
|---|---|---|
| H1 | hybrid additivity | `V_hybrid` bit-identical to `V_point_only + V_healpix_only` on NumPy (S1) |
| H2 | coordinate identity | hybrid vs both single-component runs: identical `time_grid`, `frequencies_hz`, `channel_widths_hz`, baseline order, correlations (S2) |
| H3 | reproducibility | two hybrid runs of one config give identical `scientific_sha256` (S3) |
| H4 | shared inputs | both component solvers receive the identical instrument view, beam system, receptor set, time grid objects (S4) |
| H5 | no double counting | disjoint hybrid model: summed Stokes I equals the sum of component Stokes I; `assume_disjoint: true` path warns and still enforces monopole consistency (S5) |
| H6 | provenance | hybrid result records `sky_representation="hybrid"`, `components=("point","healpix")`, and true element counts |
| H7 | component timing | both `solver_*_seconds` are positive for hybrid, and their sum does not exceed `solver_seconds` |
| H8 | fingerprint separation | hybrid and point-only results over the same instrument have different `scientific_sha256` |
| H9 | serialization | HDF5 `3.0.0` round-trips a hybrid result including the new solver, performance, and backend fields; `2.0.0` is rejected |
| H10 | summary and standard formats | summary JSON reports the components; MS and UVFITS history records them |
| H11 | empty component | a hybrid model whose point payload has zero above-horizon sources still sums correctly |
| E1 | hybrid with one payload | exact §18.3 message |
| E2 | point request that would drop maps | exact §18.3 message |
| E3 | healpix request that would rasterize | exact §18.3 message |
| E4 | `backend: numba` | exact §18.3 message. **Correction (2026-07-30, Tier 6B implementation):** owned by **6H**, not 6B — the literal change ships with the rename it names (§32.2, §32.8) |
| E5 | `execution.n_workers` | exact §18.3 message |
| E6 | non-positive worker counts (both blocks) | exact §18.3 messages |
| E7 | `solver.executor: process` | exact §18.3 message |
| E8 | explicit `sky_loading.executor: process` with unpicklable kwargs | exact §18.3 message |
| E9 | `run(n_workers=1)` | `TypeError` naming `n_workers` |
| W1 | loader worker invariance | `max_workers` in `{1,2,4,8}` × `executor` in `{thread,process}` all give identical `scientific_sha256` (S7) |
| W2 | loader policy observable | resolved `max_workers` and actual executor appear in the resolved config snapshot and the summary JSON; the auto value equals `min(requests, cpu_count, 8)` |
| W3 | solver worker invariance | `workers` in `{1,2,3,4}` give identical `scientific_sha256` (S6) |
| W4 | solver policy effective | with `workers=4` and ≥4 time samples, an instrumentation hook observes more than one distinct worker thread executing time blocks — the knob is not a no-op |
| W5 | offline under workers | offline run with a network-requiring loader raises `ConnectionError` under both executors, and `_check_socket` is never called (S12) |
| W6 | partition function | the time partition covers `[0, n_times)` exactly once for every `(n_times, workers)` pair in a swept range; `workers > n_times` clamps and records |
| W7 | no hard-coded 8 | `load_models_parallel` has no `max_workers` default and `api/simulator.py` contains no literal worker count |
| R1 | restructure bit-identity | for each shipped configuration, post-restructure `scientific_sha256` equals the pinned pre-restructure value from the 6A characterization, compared within the same Python environment as the pin was recorded in (S8). **Correction (2026-07-30, Tier 6A acceptance):** 6A's fingerprints differ between `py311` and `py312` because those environments resolve different astropy releases (7.1.0 vs 8.0.1) whose `ICRS`->`AltAz` transforms disagree in the last bits (~1.4e-11 rad altitude, ~2.0e-8 rad azimuth for a fixed source/instant), which the geometric phase amplifies into every visibility; this is an environment artifact, not solver nondeterminism. R1 is therefore per-environment by construction — Section 31 already runs the gate in both environments, so 6D must compare each environment's post-restructure digest only against that same environment's 6A pin, never across environments, and a third measured environment must add its own pinned row rather than relax this assertion. |
| R2 | assembly count | the number of whole-cube assembly operations per solver call is 1, asserted through a counting backend wrapper |
| B1 | NumPy/JAX-CPU parity | all seven §13.4 workloads within §13.5 tolerance (S9) |
| B2 | Dask bit-identity | all seven §13.4 workloads bit-identical to NumPy (S10) |
| B3 | compiled kernel | agrees with the uncompiled reference within tolerance, identical dtype (S11) |
| B4 | auto precedence | with no non-CPU JAX device, `get_backend("auto")` returns the NumPy backend and `actual_backend` is a NumPy name (closes D9) |
| B5 | registry truthfulness | `"numba"` is unknown to `get_backend`; `DaskBackend` reports a `dask-*` name; `RIMESimulator.supports_gpu` is `False` |
| B6 | precision rejection | explicit `dask` or `jax` with `float128` rejects at config time and at `get_backend`; `auto` with `float128` returns NumPy |
| B7 | synchronize | `synchronize(arr)` blocks on the given array (JAX only; skipped without JAX, but B1 already requires JAX-CPU to be installed, so this must run) |
| P1 | record completeness | a record missing any mandatory field raises `BenchmarkRecordError` |
| P2 | record honesty | every Tier 6 record has `accelerator == "none"` and `"gpu" in unmeasured` |
| P3 | harness determinism | the harness produces a valid record for a tiny workload on the NumPy backend in the fast suite (unmarked, so it always runs) |

## 28. Dependency characterization requirement

D16 is the tier's hardest external constraint: `Fix.md` §15 requires
"NumPy/JAX parity for representative point and HEALPix workloads", and JAX is not
installable in any current pixi environment (`pixi.toml:26-61`, `:69-77`). Two
positions are possible and only one is acceptable:

- **Unacceptable**: keep the six `importorskip` skips and declare parity
  untested. That would leave `RUN-004`'s central claim unevidenced while the
  tier's exit criteria demand "backend documentation matches measured
  execution".
- **Adopted**: add a **CPU-only JAX** pixi feature and environment, so parity is
  actually measured, in CI, on Linux and macOS, with no accelerator claim
  attached.

Slice 6A must therefore establish, as recorded evidence before any other slice
depends on it:

1. whether a CPU-only `jax`/`jaxlib` is resolvable for `linux-64`, `osx-64`, and
   `osx-arm64` under the existing `numpy >=1.24,<2.5` pin
   (`pixi.toml:29-32`) — conda-forge or PyPI, with exact versions;
2. whether `jax.config.update("jax_enable_x64", True)`
   (`backends/jax_backend.py:101-104`) yields true float64/complex128 for the
   solver dtypes on that build;
3. what the six currently-skipping tests do when JAX is present — pass, fail, or
   reveal a defect;
4. whether the resolved JAX-CPU numbers meet §13.5 tolerance on a small
   workload, measured before any restructure.

If (1) fails on any locked platform, Q1 (§41) governs the fallback.

## 29. Tests-first implementation strategy

Every slice writes its tests before its production change, and every slice
begins from a red test that names the defect it closes:

1. write or extend the characterization pin (6A) so the baseline behavior is
   recorded and any intentional flip is visible in a diff;
2. write the new tests from §27 for that slice, red;
3. implement the minimum production change;
4. run the §31 gate;
5. commit exactly the slice's §33 file list;
6. stop for independent acceptance.

Characterization pins that a later slice is expected to flip carry the
established `OWNED BY: Tier 6x` marker
(`tests/characterization/test_tier5_current_behavior.py:1-7`).

## 30. Slice ordering rationale

- 6A first, because three later slices depend on evidence only it can produce
  (JAX-CPU availability, baseline fingerprints for R1, current worker/offline
  behavior).
- Worker configuration (6B) before worker behavior (6C, 6E), because the typed
  policy must exist before anything can consume it.
- The accumulation restructure (6D) before solver workers (6E) and before
  hybrid (6F), because both need per-time blocks. 6D must be provably
  behavior-neutral, which is only checkable against 6A's pinned fingerprints.
- Hybrid solve (6F) before hybrid serialization (6G), so the schema bump lands
  once, with the final field set known.
- Backend truthfulness and parity (6H) after the restructure, so parity is
  measured on the shape the tier ships, not on a shape it is about to discard.
- Benchmarks and documentation (6I) last among implementation slices, because
  they must report the final architecture.

## 31. Common verification gate

Run for every slice, in both Python environments:

```bash
pixi run test -- tests/unit/test_backends/
pixi run test -- tests/unit/test_core/ -k "backend or hybrid or visibility or partition"
pixi run test -- tests/unit/test_simulator/
pixi run test -- tests/characterization/
pixi run test -- tests/integration/ -m integration
pixi run test -- -m "not slow"
pixi run --environment py312 test -- -m "not slow"
pixi run lint
pixi run check-format
pixi run typecheck
```

Additionally, from 6I onward (the harness does not exist before it):

```bash
pixi run bench            # must pass: records complete, correctness within tolerance
```

Before 6I, the parity obligation is carried by `tests/unit/test_backends/` and by
6A's recorded dependency and fingerprint evidence.

`pixi run typecheck` must stay at or below the existing Pyright baseline ceiling
(`pixi.toml:20`, `tools/check_pyright_baseline.py`). Every skip in the final
counts must be independently classified; after 6H the six JAX skips must be
**gone**, not converted into a different skip.

## 32. Tier 6 implementation slices

### 32.1 Tier 6A — characterization, dependency contract, and baseline fingerprints

**Objective.** Record what the baseline does, before anything changes, and
resolve the JAX-CPU dependency question.

Work:

- pin the point-vs-HEALPix fork, the silent payload drop for a single hybrid
  loader model, and the forced lossy conversion for multi-model spans
  (D1-D5);
- pin the hard-coded `max_workers=8` and the `run(n_workers=...)` rejection
  (D6, D7);
- pin `get_backend("auto") -> numba-cpu` with `xp is numpy` (D9), the absence of
  any numba kernel (D8), `supports_gpu is True` (D10), and the per-cell
  `set_at` accumulation count (D11);
- pin the offline/`require_service` behavior including the un-populated cache
  (D17);
- record `scientific_sha256` for every shipped configuration and for the
  §13.4 workloads, as the reference values R1 will check;
- probe and record the §28 dependency facts.

Exclusions: no production file, no `pixi.toml`, no `pixi.lock`.

### 32.2 Tier 6B — worker configuration schema and resolved runtime

**Objective.** Make worker policy expressible, typed, resolved, and recorded —
without changing any behavior yet.

Work: `SkyLoadingConfig`, `SolverExecutionConfig`, the `ExecutionConfig`
extension, the `execution.n_workers` rejection, the resolved dataclasses, the
clamp, the origin tracking, and `to_json_safe` coverage. Tests E5-E7, W2
(config half).

**Correction (2026-07-30, Tier 6B implementation):** the `numba`→`dask`
`execution.backend` literal change and its E4 rejection message move to **6H**,
which is where the backend rename they describe actually happens. Two
independent reasons, both discovered while implementing 6B:

1. *Grant infeasibility.* The literal is not confined to the two config modules
   6B may write. It is also declared at `cli/main.py:38-39`
   (`_BACKEND_CHOICES = click.Choice([... "numba"])` and
   `BackendStrategy = Literal[... "numba"]`), consumed by
   `core/precision.py:131`, `:171`, `:789` for the backend/precision
   compatibility rule, and asserted from the config side by
   `tests/unit/test_cli/test_config_mode.py:54`, `:71`, `:469`,
   `tests/unit/test_core/test_precision.py:122-123`, and
   `tests/unit/test_backends/test_resolution.py:24`, `:96`, `:157`. None of
   those files is in 6B's §33 grant; all but the two CLI files are already in
   6H's.
2. *Truthfulness.* 6B's own exclusion forbids the backend rename, so accepting
   `execution.backend: dask` in 6B would create a config literal that no
   registry entry can construct (`get_backend("dask")` does not exist until 6H)
   while removing the only literal that reaches the backend that does exist.
   That is precisely the §4.2 pattern the tier exists to remove, and papering
   over it with a temporary `dask`→`NumbaBackend` alias would be backend work
   6B is excluded from.

The 6A pin `test_execution_config_has_no_worker_or_concurrency_field` therefore
splits: 6B flips its `ExecutionConfig.model_fields` half and leaves the
backend-literal half pinned in a separate test marked `OWNED BY: Tier 6H`.

Exclusions: no solver change, no loader driver change, no backend rename, no
`execution.backend` literal change.

### 32.3 Tier 6C — loader worker behavior and offline policy

**Objective.** Remove the hard-coded 8, make the loader policy effective and
observable, and make offline real under both executors.

Work: `load_models_parallel` signature change, `LoaderExecutionRecord`,
explicit-process rejection, `set_offline_policy`, worker-side policy
installation, provenance/summary surfacing. Tests W1, W2, W5, W7, E8.

Exclusions: no solver change, no hybrid, no backend rename.

### 32.4 Tier 6D — solver accumulation restructure

**Objective.** Replace per-cell `set_at` accumulation with per-time block
assembly in both solvers, with zero change to any computed number.

Work: the §13.3 restructure; `ArrayBackend.stack`; hoisting the constant
receptor transforms out of the HEALPix time loop; naming the host-preprocessing
boundary. Tests R1, R2.

Exclusions: no worker use of the new structure yet, no hybrid, no compilation.

### 32.5 Tier 6E — solver worker policy and `run()` signature

**Objective.** Make `execution.solver.workers` effective and bit-identical, and
remove `run(n_workers=...)`.

Work: `core/solver_partition.py`, thread-pool time-block execution in both
solvers, the `run()` signature change, the migration entry, provenance of the
resolved worker count. Tests W3, W4, W6, E9.

Exclusions: no process executor, no hybrid, no backend rename.

### 32.6 Tier 6F — hybrid sky representation and canonical summation

**Objective.** `V_total = V_point + V_healpix` as a first-class high-level mode.

Work: the `hybrid` literal, `allow_lossy_point_rasterization`, the three §18.3
runtime rejections, `core/hybrid.py`, `ArrayBackend.add`, the
`SolverResultProvenance` and `ResultPerformance` extensions, the §17 ancillary
surfaces, `configs/hybrid_sky_example.yaml`. Tests H1-H8, H11, E1-E3.

Exclusions: no serialization schema bump yet (6G), no backend work.

### 32.7 Tier 6G — hybrid serialization, HDF5 3.0.0, summary, and standard formats

**Objective.** Every artifact tells the truth about a hybrid result.

Work: HDF5 `3.0.0` with `2.0.0` rejection, the reader field-set updates, the
summary JSON solver block, the MS/UVFITS history line, the loaded-result
validation. Tests H9, H10.

Exclusions: no visibility-array change, no new science.

### 32.8 Tier 6H — backend registry truthfulness, parity, and compilation boundary

**Objective.** Close the rest of `RUN-004`.

Work: the `NumbaBackend`→`DaskBackend` rename and `mode="gpu"`/`jit_compile`
removal, the `auto` precedence correction, `supports_gpu`, the `rime.py`
docstring, `supports_compilation`/`compile`, the single compiled kernel, the
`synchronize` correction, the HEALPix parity routing, the jax-cpu pixi
feature/environment and the added CI job, and the removal of the six
`importorskip` skips. Tests B1-B7.

**Correction (2026-07-30, Tier 6B implementation):** this slice also owns the
`execution.backend` `numba`→`dask` literal change, its verbatim §18.3 rejection
message, and test E4, moved here from 6B for the two reasons recorded in §32.2.
The literal must change in the same commit as the rename so no released state
exposes a config name the registry cannot construct. That adds
`src/radiosim/cli/main.py` (the `click.Choice` and `BackendStrategy` literal)
and `tests/unit/test_cli/test_config_mode.py` (three config-side numba
assertions, including the `--backend [auto|numpy|jax|numba]` help string) to
this slice's §33 grant, and 6H must also flip the 6A pin
`test_execution_config_backend_literal_still_offers_numba`.

Exclusions: no new science, no GPU claim, no numba kernel.

### 32.9 Tier 6I — benchmark harness, records, and documentation truth

**Objective.** Make every acceleration statement reproducible, or delete it.

Work: `src/radiosim/benchmarks/`, `tests/performance/test_backend_benchmarks.py`,
the `bench` task, the §26 documentation obligations, and one committed set of
records under `output/benchmarks/` reproduced by the reviewer on their own
machine. Tests P1-P3.

Exclusions: no production behavior change; no threshold assertion.

### 32.10 Tier 6J — independent whole-tier acceptance

No production change. The reviewer re-derives every §37 criterion from source
and from their own runs, records the result in `Fix.md`, and flips
`RUN-001`..`RUN-004` only if every criterion passes.

## 33. Exact writable file list for every slice

### Tier 6A

```text
tests/characterization/test_tier6_current_behavior.py
```

### Tier 6B

```text
src/radiosim/core/runtime_config.py
src/radiosim/io/__init__.py
src/radiosim/io/config.py
src/radiosim/io/config_resolution.py
tests/characterization/test_tier6_current_behavior.py
tests/fixtures/configs.py
tests/unit/test_io/test_config.py
tests/unit/test_io/test_config_resolution.py
tests/unit/test_simulator/test_worker_policy.py
```

### Tier 6C

```text
src/radiosim/api/simulator.py
src/radiosim/core/sky/operations/__init__.py
src/radiosim/core/sky/operations/parallel.py
src/radiosim/io/summary_json.py
src/radiosim/utils/__init__.py
src/radiosim/utils/network.py
tests/characterization/test_tier6_current_behavior.py
tests/unit/test_core/test_sky_pipeline.py
tests/unit/test_io/test_result_summary.py
tests/unit/test_simulator/test_api.py
tests/unit/test_simulator/test_worker_policy.py
tests/unit/test_utils/test_network.py
tests/unit/test_utils/test_offline_policy.py
```

**Correction (2026-07-30, Tier 6C implementation):** this list omitted
`tests/unit/test_io/test_result_summary.py`, which the slice cannot avoid.
Section 19 requires the summary JSON to gain an `execution` block, and
`test_summary_json_is_exact_bounded_metadata_contract`
(`tests/unit/test_io/test_result_summary.py:142`) asserts the document's
top-level key set *exactly* — deliberately, so no key is ever added without a
conscious edit. The file is added to this grant for the one-line key-set update;
no other assertion in it changes. Two related notes, neither of which changes a
decision:

- The `execution` block carries the *requested* worker policy from
  `resolved_config` and the *executed* `LoaderExecutionRecord`. The record is not
  a new `SimulationResult` field: `core/result.py`, `io/writers.py`, and
  `io/readers.py` are outside 6C's grant, and the HDF5 `2.0.0`→`3.0.0` bump that
  a new provenance field would require belongs to 6G (Sections 19, 32.7). The
  record therefore travels in `SimulationResult.history` as one
  `RADIOSIM_SKY_LOADER_JSON=` line, following the established
  `PROJECTION_HISTORY_PREFIX` convention (`io/standard_visibility.py:42`), and is
  decoded by `LoaderExecutionRecord.from_history`. This also means the 6A pin
  `test_no_worker_value_is_recorded_in_provenance` keeps its assertions: it
  guards `to_summary_snapshot()`, the bounded metadata view, which is a different
  surface from the summary-JSON document and stays free of runtime worker values.
  That pin's remaining owner is 6E.
- Adding a key grows the summary document while its `schema.version` stays
  `1.0.0`. Section 19 authorizes the growth and names no version bump, and 6G
  adds a further `solver` block to the same document on the same terms, so 6C
  leaves the literal alone; whether the summary schema needs a version at all is
  a question 6G should settle once, not a question 6C should answer twice.

### Tier 6D

```text
src/radiosim/backends/base.py
src/radiosim/backends/numpy_backend.py
src/radiosim/core/visibility.py
src/radiosim/core/visibility_healpix.py
tests/characterization/test_tier6_current_behavior.py
tests/unit/test_backends/test_array_backend_helpers.py
tests/unit/test_core/test_beam_solver_integration.py
tests/unit/test_core/test_visibility_accumulation.py
tests/unit/test_core/test_visibility_backend.py
```

There is no `tests/unit/test_core/test_visibility.py` or
`test_visibility_healpix.py` at the baseline; both solvers are covered by
`tests/unit/test_core/test_visibility_backend.py` and
`tests/unit/test_core/test_beam_solver_integration.py`, and
`test_visibility_accumulation.py` and
`tests/unit/test_backends/test_array_backend_helpers.py` are new files.

### Tier 6E

```text
src/radiosim/api/simulator.py
src/radiosim/core/__init__.py
src/radiosim/core/solver_partition.py
src/radiosim/core/visibility.py
src/radiosim/core/visibility_healpix.py
src/radiosim/simulator/rime.py
docs/migration_guide.md
tests/characterization/test_tier6_current_behavior.py
tests/unit/test_core/test_solver_partition.py
tests/unit/test_simulator/test_api.py
tests/unit/test_simulator/test_worker_policy.py
tests/unit/test_tier4_result_output_acceptance.py
```

Two existing tests pin the parameter 6E removes and must be flipped in the same
commit: `tests/unit/test_tier4_result_output_acceptance.py:229-232` asserts the
exact `Simulator.run` signature string including
`n_workers: 'int | None' = None`, and `tests/unit/test_simulator/test_api.py:962-963`
asserts the `NotImplementedError`. Both are inside this grant. No other in-tree
caller passes `n_workers`.

### Tier 6F

```text
src/radiosim/api/simulator.py
src/radiosim/backends/base.py
src/radiosim/core/__init__.py
src/radiosim/core/hybrid.py
src/radiosim/core/result.py
src/radiosim/core/sky/combine/concat.py
src/radiosim/io/config.py
src/radiosim/io/config_resolution.py
configs/hybrid_sky_example.yaml
tests/characterization/test_tier6_current_behavior.py
tests/fixtures/configs.py
tests/integration/test_hybrid_end_to_end.py
tests/unit/test_core/test_hybrid_visibility.py
tests/unit/test_core/test_result.py
tests/unit/test_core/test_sky_combine.py
tests/unit/test_io/test_config.py
tests/unit/test_simulator/test_api.py
tests/unit/test_simulator/test_result_integration.py
```

**Correction (2026-07-31, Tier 6F implementation):** this list omitted two test
files that construct `SolverResultProvenance` and `ResultPerformance` directly
and that the slice therefore cannot avoid:

```text
tests/unit/test_io/test_hdf5_result.py
tests/unit/test_io/test_standard_visibility.py
```

`C10` already declares the blast radius of the two dataclass extensions as
"every construction site". In the tree there are exactly four such sites: one in
production (`api/simulator.py`, granted) and three in tests
(`tests/unit/test_core/test_result.py`, granted, plus the two above). Every
other apparently-affected test file — `tests/unit/test_io/test_output_atomicity.py`,
`tests/unit/test_io/test_result_summary.py`, `tests/unit/test_io/test_uvfits.py`,
`tests/unit/test_io/test_measurement_set.py`,
`tests/unit/test_simulator/test_instrument_integration.py` — reaches the
dataclasses through `tests/unit/test_core/test_result.py::_parts` and needs no
edit. Both added files also appear in 6G's grant; 6F touches only the two
`SolverResultProvenance(...)` / `ResultPerformance(...)` literals in each and
changes no assertion, leaving 6G's serialization work untouched.

The alternative — giving `components` and `component_element_counts` default
values so the two files keep compiling — was rejected: a defaulted
`component_element_counts` would let a construction site record a zero element
count that criterion 7 requires to be true, and `C10`'s wording shows the plan
already intended required fields.

Two further notes from the same implementation, neither of which changes a
decision:

- **Where the §18.3 runtime rejections are evaluated.** §20.1 places the
  representation-compatibility rejection at step 9, after combination, "because
  the decision needs the *combined* model's payload set". That holds for the
  `hybrid` rule and for the surviving-payload half of the `point_sources` rule.
  It cannot hold for the `healpix_map` rule, whose message quotes `{n}` point
  sources that combination has already folded into the map, nor for the
  *dropped*-payload half of the `point_sources` rule (§8.2 rule 2 says it closes
  `D3` **and** `D4`, but a `D4` combination leaves no HEALPix payload on the
  resolved model to detect). The gate therefore runs at step 9 as specified and
  is handed *both* the contributed model list and the resolved model. Nothing in
  the ordering moves; the check simply reads two inputs instead of one.
- **The `allow_lossy_point_materialization` escape named in the `point_sources`
  message.** The message is emitted verbatim, but the flag does not gate the
  rejection. `materialize_point_sources_model` returns a model that already
  carries points unchanged (`core/sky/operations/operations.py:204-209`), so the
  flag cannot convert the HEALPix payload of a hybrid model; honoring it here
  would silently re-open `D3`. The flag keeps its real and tested function — a
  HEALPix-*only* contributor under a point request — which the gate does not
  touch. The message's other escape, `hybrid`, is always correct.

### Tier 6G

```text
src/radiosim/core/result.py
src/radiosim/io/hdf5.py
src/radiosim/io/readers.py
src/radiosim/io/result_errors.py
src/radiosim/io/standard_visibility.py
src/radiosim/io/summary_json.py
docs/api/io.rst
docs/migration_guide.md
tests/integration/test_hybrid_end_to_end.py
tests/unit/test_io/test_hdf5_result.py
tests/unit/test_io/test_result_summary.py
tests/unit/test_io/test_standard_visibility.py
tests/unit/test_io/test_uvfits.py
tests/unit/test_io/test_measurement_set.py
tests/unit/test_simulator/test_result_integration.py
tests/unit/test_tier1h_documentation.py
tests/unit/test_tier4_result_output_acceptance.py
```

The summary-JSON tests live in `tests/unit/test_io/test_result_summary.py` at the
baseline; there is no `test_summary_json.py`.

**Correction (2026-07-31, Tier 6G implementation).** Four files are added above
because the `2.0.0`→`3.0.0` bump this slice is defined by cannot be expressed
without them. None adds behavior beyond §19; each is the single site that states
the schema version the slice changes.

1. `src/radiosim/io/result_errors.py` — §32.7 requires `2.0.0` to be "rejected on
   read with a message naming Tier 6", but the rejection message is not written
   in `io/hdf5.py`. `UnsupportedSchemaVersionError` composes it from a class
   constant `GUIDANCE` (`io/result_errors.py:74-78`) whose text names Tier 5 and
   tells the reader to "re-run the simulation to write a 2.0.0 file". The
   raising site cannot override it: `__init__` takes only the offending version
   string. Without this file the slice can bump the constant but cannot make the
   rejection truthful, which is the half of the requirement that matters.
2. `docs/api/io.rst` — states the schema version in prose three times
   (`:76`, `:103`, `:141`), including the sentence "Re-run the simulation to
   obtain a ``2.0.0`` file". §26 lists the documentation Tier 6 owns and omits
   this file; that omission is an oversight, not a decision, because §26's own
   rule is that a tier owns "exactly the statements its own changes make false",
   and a `3.0.0` writer makes all three false the moment it lands. The grant is
   for those three statements and the `1.0.0`/`2.0.0` rejection paragraph only;
   the rest of `io.rst` is untouched and stays outside every Tier 6 grant.
3. `tests/unit/test_tier1h_documentation.py` — pins the `io.rst` text that
   item 2 corrects (`:324` requires the literal `"2.0.0"` in `io.rst`; `:785`
   requires "schema version ``2.0.0``"). One-line-per-assertion update; no other
   assertion in the file changes. The file is also in 6I's grant, for the
   unrelated §26 items 1-4; the two grants do not overlap in content.
4. `tests/unit/test_tier4_result_output_acceptance.py` — pins
   `hdf5_module.SCHEMA_VERSION == "2.0.0"` at `:222`. This is a Tier 4
   acceptance pin, not a Tier 6 characterization pin, so it carries no
   `OWNED BY` line and no slice was granted it; it must move with the constant
   or the suite cannot pass. Only that one assertion changes.

No test in `tests/characterization/test_tier6_current_behavior.py` pins any
serialization surface, so Tier 6G flips no `OWNED BY: Tier 6G` pin — there are
none. That is consistent with §32.1, which lists the 6A characterization
subjects and does not include the HDF5 schema version.

### Tier 6H

```text
src/radiosim/backends/__init__.py
src/radiosim/backends/base.py
src/radiosim/backends/dask_backend.py
src/radiosim/backends/jax_backend.py
src/radiosim/backends/numpy_backend.py
src/radiosim/cli/main.py
src/radiosim/core/precision.py
src/radiosim/core/result.py
src/radiosim/core/visibility.py
src/radiosim/core/visibility_healpix.py
src/radiosim/io/config.py
src/radiosim/io/config_resolution.py
src/radiosim/simulator/rime.py
src/radiosim/utils/device.py
pixi.toml
pixi.lock
pyproject.toml
.github/workflows/ci.yml
tests/characterization/test_tier6_current_behavior.py
tests/unit/test_backends/test_backend_parity.py
tests/unit/test_backends/test_backends.py
tests/unit/test_backends/test_compilation_boundary.py
tests/unit/test_backends/test_dask_backend.py
tests/unit/test_backends/test_jax_backend.py
tests/unit/test_backends/test_resolution.py
tests/unit/test_cli/test_config_mode.py
tests/unit/test_core/test_precision.py
tests/unit/test_core/test_sky_backend.py
tests/unit/test_core/test_sky_spectral.py
tests/unit/test_core/test_visibility_backend.py
tests/unit/test_jones/test_backend_jones.py
tests/unit/test_release_metadata.py
tests/unit/test_utils/test_device.py
```

Note: `src/radiosim/backends/numba_backend.py` is deleted by the rename in this
slice; the deletion is part of the grant.

**Correction (2026-07-30, Tier 6B implementation):** `src/radiosim/cli/main.py`
and `tests/unit/test_cli/test_config_mode.py` are added above because the
`execution.backend` literal change moved into this slice (§32.2, §32.8) and the
literal is declared a second time in the CLI (`cli/main.py:38-39`) and asserted
from the config side in three places in `test_config_mode.py` (`:54`, `:71`,
`:469`). No other slice was ever granted `cli/main.py`, so without this addition
the CLI would keep offering a `--backend numba` choice the schema rejects.

**Correction (2026-07-31, Tier 6H implementation).** Eleven further files are
added to the grant above. Every one of them is a site the slice's *own* declared
work makes stale, found by grepping the whole tree for the removed identifiers
rather than by re-deriving the list from §25; none of them is new scope, and no
other slice is granted any of them.

```text
src/radiosim/api/simulator.py
src/radiosim/core/contraction.py
src/radiosim/core/runtime_config.py
src/radiosim/simulator/__init__.py
examples/scripts/simple_simulation.py
docs/api/backends.rst
tests/unit/test_core/test_beam_runtime.py
tests/unit/test_core/test_visibility_accumulation.py
tests/unit/test_io/test_config.py
tests/unit/test_tier1h_documentation.py
tests/unit/test_tier4_result_output_acceptance.py
```

Why each:

- `api/simulator.py` — §14.3 adds `device_kind` and `compilation_used` to
  `BackendResultProvenance`, and this is that dataclass's only construction site
  in `src/`. `core/result.py` alone cannot populate a field.
- `core/contraction.py` — the §25.1 addition recorded above.
- `core/runtime_config.py` — `ResolvedExecutionConfig.backend_strategy` declares
  the backend literal a *third* time (`:316`, `:324`, `:338`), alongside
  `io/config.py` and `cli/main.py`. §18.4 specifies it as
  `Literal["auto", "numpy", "jax", "dask"]`; without this file the resolver
  would reject the very literal the schema now requires. This is the same
  omission the 6B correction found in the CLI, one layer further in.
- `simulator/__init__.py` — its `get_simulator` docstring example prints
  `sim.supports_gpu` as `True`. §14.1 makes that `False`; leaving the example
  would ship a documented falsehood 6H itself created.
- `examples/scripts/simple_simulation.py` — offers `--backend numba` as an
  `argparse` choice, which the schema now rejects with §18.3's message.
- `docs/api/backends.rst` — `automodule:: radiosim.backends.numba_backend`
  names a module the rename deletes, so `make -C docs html` (a CI quality step)
  fails without it. §33 grants this file to 6I, which is a *later* slice: 6I
  cannot repair a build 6H breaks. 6I's own grant is unchanged and still covers
  the file for its §26.2 documentation work.
- `tests/unit/test_core/test_beam_runtime.py` — imports
  `radiosim.backends.numba_backend` directly (`:1396`).
- `tests/unit/test_io/test_config.py` — asserts
  `ExecutionConfig(backend="numba")` (`:836`); it is also the natural home for
  §27 row E4, next to the E5 `n_workers` test that established the pattern.
- `tests/unit/test_tier1h_documentation.py` — imports `numba_backend` and
  asserts against `NumbaBackend.__doc__` (`:18`, `:629-630`). Same
  later-slice-cannot-repair-an-earlier-break reasoning as `docs/api/backends.rst`.
- `tests/unit/test_tier4_result_output_acceptance.py` — pins the exact pixi
  environment feature lists (`:343`), which C18 changes by construction.
- `tests/unit/test_core/test_visibility_accumulation.py` — see the §13.3
  correction below.

**Correction (2026-07-31, Tier 6H implementation) — §13.3's per-`(t, f)`
assembly.** §13.3 has the solver "assemble one `(B, 2, 2)` block for all
baselines at `(t, f)`", which 6D implemented as a `backend.stack` over `B`
separately computed `(2, 2)` matrices, because the contraction then ran one
baseline at a time. §13.6's kernel is *baseline-batched* and returns that whole
`(B, 2, 2)` block from a single call, so at that level there is nothing left to
assemble: the `T*F` per-`(t, f)` assemblies disappear and only the `T` per-time
blocks and the one whole-cube assembly remain. Every binding property of §13.3
holds unchanged and is still asserted — one `(B, F, 2, 2)` block per time,
exactly one whole-cube assembly per call (R2), zero `set_at` calls. This is
strictly fewer assemblies, which is the direction §13.3 exists to push, so it is
a consequence of §13.6 rather than a relaxation of §13.3; the two 6D counting
tests that spell out the old count are narrowed accordingly, in place, with the
reason recorded in the test.

The kernel's *input* batching (two `(B, S, 2, 2)` antenna-Jones batches per
step) deliberately goes through `backend.xp.stack` and not through
`ArrayBackend.stack`. §13.3 defines `stack` as the solvers' one *accumulation*
primitive; routing kernel inputs through the same method would make the
accumulation counts uncountable without adding any capability. Both reach the
same backend namespace.

**Correction (2026-07-31, Tier 6H implementation) — §25.5, §28 and C18: a
jax-cpu *feature*, carried by both existing environments, not a third
environment.** §25.5 and C18 describe jax-cpu as a separate pixi *environment*
with "one added job running the jax-cpu environment". That is incompatible with
§31, which requires that "after 6H the six JAX skips must be **gone**, not
converted into a different skip" — the counts §31 governs come from
`pixi run test -- -m "not slow"` and `pixi run --environment py312 test -- -m
"not slow"`, and a JAX confined to a third environment would leave all six tests
skipping in both of them. The two statements cannot both hold.

§31's requirement is the load-bearing one: it is what makes the parity evidence
part of the standard gate rather than an optional side channel. So `jax-cpu` is
declared as a pixi **feature** and included in both `default` and `py312`,
giving the six env × platform combinations Q1 measured, and the six
`importorskip("jax")` guards are deleted outright so a missing JAX fails loudly
instead of silently reporting a green run that measured nothing. The added CI
job remains, as a `backend-parity` job that asserts the locked JAX is importable
and runs the parity and compilation-boundary suites directly — a cheap, explicit
guard that fails loudly if JAX ever drops out of the lock, which the six matrix
jobs would not do as legibly. §28's adopted position ("add a CPU-only JAX pixi
feature and environment, so parity is actually measured, in CI, on Linux and
macOS, with no accelerator claim attached") is satisfied in full; only the
placement of the feature changes.

### Tier 6I

```text
src/radiosim/benchmarks/__init__.py
src/radiosim/benchmarks/harness.py
src/radiosim/benchmarks/record.py
pixi.toml
.gitignore
README.md
CLAUDE.md
docs/user_guide/backends.rst
docs/user_guide/configuration.rst
docs/migration_guide.md
docs/api/backends.rst
tests/performance/test_backend_benchmarks.py
tests/unit/test_core/test_benchmark_record.py
tests/unit/test_tier1h_documentation.py
tests/unit/test_tier6_runtime_acceptance.py
```

**Correction (2026-07-31, Tier 6I implementation).** Two paths are added to the
grant above. Neither is new scope; each is a file this slice's own declared work
cannot be delivered without, found the same way every earlier slice's additions
were — by running the slice and seeing what it breaks.

```text
tests/characterization/test_tier6_current_behavior.py
output/benchmarks/reference/*.json
```

1. `tests/characterization/test_tier6_current_behavior.py` — 6A's pin
   `test_there_is_no_benchmark_harness_task_or_performance_test` carries
   `OWNED BY: Tier 6I` in its own docstring and asserts, verbatim, that
   `tests/performance/` holds only `__init__.py`, that
   `src/radiosim/benchmarks/` does not exist, and that `pixi.toml` contains no
   `bench` task. Those are exactly the three things §32.9 requires 6I to create,
   so the suite cannot pass unless 6I flips this pin. Every other slice's grant
   lists this file; 6I's omission is the same oversight §32.2, §32.3, §32.6, and
   §32.8 each corrected in turn, and the grant is for that one test only.
2. `output/benchmarks/reference/*.json` — §32.9 requires "one committed set of
   records under `output/benchmarks/`", and §33 grants no path under `output/`
   at all. §22.1 additionally says benchmark output "is gitignored", which is
   directly incompatible with committing it. Both statements are kept, with the
   two roles separated: every `pixi run bench` run writes the
   `<UTC timestamp>-<host tag>.json` §22.1 describes and that file stays ignored,
   so running a benchmark never dirties the tree; the curated reference set §32.9
   requires is copied into `output/benchmarks/reference/`, which `.gitignore`
   (already in 6I's grant) un-ignores for `*.json` only. Nothing else under
   `output/` becomes committable.

**Correction (2026-07-31, Tier 6I implementation) — §23 gains one field and two
sibling record types.** All three are additive; no §23 field is removed,
widened, or made optional.

- `BenchmarkRecord` gains `workload: str`. §23's field list states every
  *dimension* of a workload but never its identity, so two rows of the §13.4
  matrix with equal counts produce indistinguishable records — which defeats
  "reproducible", the property §32.9 exists to establish. The field names the
  §13.4 row (or the added scaled row) the record measured.
- `RetracingRecord` and `MemoryScalingRecord` are added alongside
  `BenchmarkRecord` in the same output document, because the two obligations the
  Tier 6H independent acceptance routed to 6I (§39's added rows) are not
  workload timings and cannot be expressed in §23's schema: one measures cost
  *per distinct kernel input shape* across a step sequence, the other measures
  peak working set against `(baselines, sources)` with no solver call at all.
  Forcing either into `BenchmarkRecord` would require inventing values for
  fields it does not measure, which is precisely what §23's no-partial-record
  rule forbids. Both are validated by the same missing-field and `None` rules.

**Correction (2026-07-31, Tier 6I implementation) — §26 item 4's "exactly three
corrections" to `CLAUDE.md` is an undercount created by Tier 6H.** §26 was
written before 6H renamed `NumbaBackend` to `DaskBackend` and deleted
`backends/numba_backend.py`. `CLAUDE.md` names that module by path, offers
`get_backend("auto" | "numpy" | "jax" | "numba")`, and calls the JAX/Numba
backends "scaffolded" — statements 6H made false and that §26's own governing
rule ("a tier owns exactly the statements its own changes make false") therefore
assigns to Tier 6. No later slice exists to fix them: 6J changes no production or
documentation file. The `CLAUDE.md` grant is read as the three sentences §26
names **plus** every sentence Tier 6H's rename falsified, **plus** the two
additions 6I's own new package requires: one `### Benchmarks (benchmarks/)`
subsection, because the Architecture section enumerates every `src/radiosim/`
subpackage and 6I adds one, and one `pixi run bench` line in the command block,
because that block is where a reader looks for how to run something. Nothing
else: the Jones inventory, the sky-model sections, and the RIME equation section
are untouched.

**Correction (2026-07-31, Tier 6I implementation) — three further documentation
files name a removed identifier.** Tier 6H expanded its own §33 grant by eleven
files found "by grepping the whole tree for the removed identifiers rather than
by re-deriving the list from §25". That grep covered `src/` and `tests/` and
`docs/api/backends.rst`; it did not cover the rest of `docs/`. Repeating it for
`numba` finds three surviving sites, each of which now instructs a reader to do
something the tree rejects:

```text
docs/installation.rst
docs/quickstart.rst
docs/user_guide/configuration_support.rst
```

- `docs/installation.rst` publishes `pip install radiosim[numba]`. That extra no
  longer exists — `pyproject.toml` renamed it to `dask` in 6H — so the command
  fails outright.
- `docs/quickstart.rst` and `docs/user_guide/configuration_support.rst` present
  `Numba` as a selectable backend. `execution.backend: numba` is now rejected by
  the schema with §18.3's message, and `get_backend("numba")` raises.

The grant is for those `numba` occurrences only. Three further files
(`docs/index.rst`, `docs/user_guide/sky_models.rst`, `docs/changelog.rst`) also
carry GPU-related prose but are deliberately **excluded**: the first two state
disclaimers that Tier 6 did not falsify, and `docs/changelog.rst`'s "Universal
GPU acceleration via JAX and Numba backends" was already false before Tier 6
began — it is `RUN-004` itself, not a statement Tier 6's changes made false, and
§26's rule therefore leaves it to the Tier 8 sweep. It is recorded here so that
it is a known, routed gap rather than an oversight.

### Tier 6J

```text
Fix.md
Tier6HybridRuntimePlan.md
tests/unit/test_tier6_runtime_acceptance.py
```

## 34. Independent acceptance gate after every slice

Every slice acceptance is performed by a reviewer who did not write the slice
and who works from source, not from the implementation summary. Each acceptance
must independently:

1. confirm the commit contains only its exact §33 file list;
2. re-derive the slice's claims from source, not from the implementation notes —
   including at least one numerical claim recomputed by hand or by an
   independent script;
3. author at least one rejected input by hand and confirm the exact §18.3
   message byte for byte;
4. run the §31 gate in both Python environments and classify every skip;
5. confirm no later-tier behavior entered the slice;
6. for any performance statement, reproduce it on the reviewer's own machine and
   record the hardware;
7. record the result in `Fix.md` as a dated acceptance note without rewriting
   any prior record.

## 35. Stop boundary after every slice

After each commit the implementer stops. The next slice is unauthorized until
its predecessor is independently accepted. No slice may be split across two
commits, and no two slices may share a commit. Any deviation from this plan
requires a plan amendment, committed and accepted, before the deviating code.

## 36. Breaking-change ledger

| # | Change | Slice | Blast radius |
|---|---|---|---|
| C1 | `execution` gains `sky_loading` and `solver` blocks | 6B | additive; defaults reproduce baseline behavior |
| C2 | `execution.backend: numba` removed in favor of `dask` | 6B, 6H | any config naming `numba`; typed message |
| C3 | `load_models_parallel(max_workers=...)` becomes required | 6C | one in-tree caller |
| C4 | Offline policy becomes authoritative for loaders | 6C | a previously-network-touching offline run now fails fast |
| C5 | Solver accumulation restructured | 6D | none numerically (R1 proves bit-identity) |
| C6 | `Simulator.run(n_workers=...)` removed | 6E | `TypeError`; no in-tree caller |
| C7 | `visibility.sky_representation` gains `hybrid` | 6F | additive |
| C8 | `point_sources` with a surviving HEALPix payload now rejects | 6F | previously silent under-count; now an error |
| C9 | `healpix_map` with point contributors requires `allow_lossy_point_rasterization` | 6F | previously silent rasterization; now opt-in |
| C10 | `SolverResultProvenance` and `ResultPerformance` gain fields | 6F | every construction site |
| C11 | `scientific_sha256` changes for every result | 6F | every recorded fingerprint, including single-component ones |
| C12 | `provenance_sha256` changes for every result | 6F, 6H | every recorded fingerprint |
| C13 | HDF5 schema `2.0.0` → `3.0.0`, `2.0.0` rejected | 6G | every previously written file; no upgrade path by design |
| C14 | `NumbaBackend` → `DaskBackend`; `jit_compile` and `mode="gpu"` removed | 6H | public backend surface |
| C15 | `get_backend("auto")` no longer selects the NumPy-delegating backend | 6H | `actual_backend` provenance values change |
| C16 | `RIMESimulator.supports_gpu` becomes `False` | 6H | anything branching on it (nothing in tree) |
| C17 | `ArrayBackend` gains `add`, `stack`, `supports_compilation`, `compile`; `synchronize` widened | 6D, 6F, 6H | third-party backend implementers (none) |
| C18 | jax-cpu becomes a declared pixi environment; six skips disappear | 6H | lockfile, CI job count |

**Note on C11 and `RUN-005` (added at RUN-005 standalone acceptance,
2026-07-30).** `RUN-005` (`Fix.md` §5, `scientific_sha256` path-dependence)
was fixed standalone between Tier 6E and Tier 6F and already changed every
recorded `scientific_sha256`, including the two Tier 6 R1 shipped-config
pins in `tests/characterization/test_tier6_current_behavior.py`, which were
re-pinned in the same standalone change. This is deliberately **not** folded
into C11: the Tier 6D acceptance review already ruled that the path-dependence
defect and C11's hybrid-summation churn are unrelated causes of the same
symptom and must not be conflated (`Fix.md`, 6D acceptance record, "a new
register row, not a Section 21/§27 C11/C12 ledger note"). This note only
flags a practical consequence for whoever re-pins C11: the *pre-RUN-005*
`scientific_sha256` values recorded in any earlier acceptance record are
**not** the correct "before" baseline for 6F's re-pin diff, because a
standalone, unrelated fix already moved them once. Diff against the
current (post-RUN-005) pinned values, not the historical pre-fix ones (both
are recorded in the `_SHIPPED_CONFIG_FINGERPRINTS` code comment and in
`Fix.md`'s RUN-005 standalone acceptance note).

**Candidate test addition for 6F (from RUN-005 standalone acceptance).** The
RUN-005 fix's beam projection (`core/result.py::_scientific_beam_projection`)
was verified checkout-path-independent for both analytic beams (via the
committed `test_scientific_fingerprint_is_independent_of_source_checkout_location`,
which only exercises the antenna-layout path since the three shipped configs
use analytic beams) and, independently by the accepting reviewer using an
uncommitted scratch script, for a FITS (`shared_fits`) beam. No FITS-beam
checkout-independence regression test is committed. `tests/unit/test_core/
test_result.py` and `tests/fixtures/configs.py` are already in 6F's Section 33
grant, so adding one there requires no grant change; `tests/fixtures/
beamfits.py` (already exists, not in 6F's grant) can be imported unmodified.
**Gotcha for whoever writes it:** `write_scalar_efield_beamfits()` is not
byte-reproducible across separate calls -- pyuvdata's beamfits writer embeds
a write-time timestamp in a FITS HISTORY card (confirmed: two independent
generations of the same fixture differ at byte offset 4519, the embedded
`"YYYY-MM-DD HH:MM:SS.fff using pyuvdata version ..."` string) -- so the test
must generate the fixture **once** and copy the identical bytes into two
checkout directories, never regenerate independently, or the test will
conflate write-time non-reproducibility with checkout-path dependence. This
is a candidate addition, not a blocking gap: routed to 6F rather than treated
as a defect in the standalone fix.

## 37. Final whole-tier acceptance criteria

Tier 6J accepts Tier 6 only when all criteria pass as one indivisible gate.

1. The implementation range is linear, every slice was independently accepted,
   and every commit contains only its exact §33 file list.
2. All ten design decisions in §8-§17 are implemented as specified, or the plan
   was amended and re-accepted before the deviation.
3. `sky_representation: hybrid` runs both components on one shared time grid,
   one shared frequency axis, one shared baseline selection, one shared receptor
   set, and one shared backend, proven by object identity (S4).
4. `V_hybrid` is bit-identical to `V_point + V_healpix` on the NumPy backend
   (S1), and hybrid coordinates are element-wise identical to both
   single-component runs (S2).
5. Hybrid results are one canonical `SimulationResult`; no second result object
   exists, and the summation happens in the backend array domain before the
   single host transfer.
6. Disjoint and explicitly-assumed-disjoint hybrid models do not double count,
   and the monopole gate is still enforced under `assume_disjoint` (S5).
7. Hybrid provenance records the representation, the component list, the true
   per-component element counts, and per-component timings; component names and
   counts are inside `scientific_sha256`; timings are outside both hashes.
8. HDF5 `3.0.0`, the summary JSON, MS, and UVFITS all round-trip and report a
   hybrid result; `2.0.0` is rejected with a message naming Tier 6.
9. Loader and solver worker policies are separate typed configuration, both
   resolved centrally, both recorded in provenance, and neither reachable
   through a function keyword argument.
10. No hard-coded worker count remains anywhere in `src/`; `load_models_parallel`
    has no `max_workers` default (W7).
11. Every worker setting has observable tested behavior: loader worker count and
    executor are provably in force (W2) and result-invariant (W1); solver worker
    count is provably in force (W4) and result-invariant (W3).
12. `Simulator.run()` has no `n_workers` parameter, and the migration boundary is
    documented and tested (E9).
13. An offline run performs no socket probe and fails network-requiring loaders
    under both executors (S12).
14. Both solvers assemble their output cube once per call (R2), and the
    restructure is bit-identical to the 6A-pinned baseline for every shipped
    configuration (R1).
15. NumPy and JAX-CPU agree within the §13.5 tolerance on all seven §13.4
    workloads, with JAX actually installed and the six baseline skips gone
    (S9).
16. Dask-with-NumPy-arrays is bit-identical to NumPy (S10).
17. Exactly one kernel is compiled, its uncompiled reference agrees within
    tolerance and dtype (S11), and `compilation_used` is recorded truthfully.
18. `get_backend("auto")` no longer returns a NumPy-delegating backend under a
    non-NumPy name, `"numba"` is not a selectable backend, `DaskBackend` reports
    a `dask-*` name, and `supports_gpu` is `False` (B4, B5).
19. Precision precedence is unchanged in substance and extended to `dask`, with
    explicit-backend rejection, `auto` diversion, and config-time rejection all
    tested (B6).
20. A reproducible benchmark harness exists, every record carries every §23
    field, an incomplete record raises, and at least one full record set was
    reproduced independently by the reviewer on their own hardware (P1, P3).
21. Every Tier 6 benchmark record states `accelerator == "none"` and lists
    `"gpu"` among its unmeasured items; no GPU, TPU, or distributed number
    appears anywhere in the tier (P2).
22. Every documentation statement about backend capability, speed, or device
    support either cites a committed record file or has been deleted; the §26.4
    `CLAUDE.md` edits are exactly the three named lines.
23. Dual-Python focused and full non-slow suites pass with only independently
    classified skips and established warnings.
24. Ruff, formatting, Pyright under the unchanged ceiling, lock metadata, YAML
    validation, offline example, clean-copy Sphinx, whitespace, fresh imports,
    and generated-artifact checks pass.
25. CI succeeds for the quality job, all six locked OS/Python jobs, and the added
    jax-cpu job, on the exact acceptance SHA.
26. No Tier 7 or Tier 8 implementation enters the range: no new Jones term, no
    m-mode solver, no numba kernel, no repository-wide documentation rewrite.

Any failed criterion keeps `RUN-001`..`RUN-004` open.

## 38. Evidence required to close RUN-001 through RUN-004

| Issue | Tier 6J evidence |
|---|---|
| `RUN-001` | criteria 9, 11, 12 — `run(n_workers=...)` is gone, a typed solver worker policy exists, is centrally resolved, is recorded in provenance, is provably in force, and is result-invariant |
| `RUN-002` | criteria 9, 10, 11, 13 — no hard-coded worker count survives, loader concurrency is typed and configurable, the resolved value is recorded, executor selection and its degradation are explicit, and offline behavior under both executors is defined and tested |
| `RUN-003` | criteria 3, 4, 5, 6, 7, 8 — hybrid is a first-class high-level mode with no lossy conversion, exact additivity, coordinate identity, a single canonical result, an unchanged disjointness gate, and full serialization |
| `RUN-004` | criteria 14, 15, 16, 17, 18, 19, 20, 21, 22 — the accumulation pathology is removed and measured, NumPy/JAX-CPU parity is actually measured with JAX installed, the registry names match what executes, the compilation boundary is explicit and verified, and every capability statement cites a reproducible record |

`RUN-004` is closed as **"backend correctness parity complete; accelerator
performance undemonstrated"** — its `ROADMAP` status becomes `DONE` only for
the scope this plan defines in §13.1. Any GPU work remains a separate,
un-opened item; §41 Q4 records how it should be filed.

## 39. Risk register

| Risk | Control and acceptance evidence |
|---|---|
| A CPU-only JAX is not installable on a locked platform under the NumPy pin | 6A must resolve this before 6H depends on it; Q1 governs the fallback, and no other slice's boundary moves |
| The accumulation restructure changes numbers | R1 compares against 6A-pinned `scientific_sha256` for every shipped configuration; a single differing digit fails the slice |
| Thread-parallel solver execution is not actually deterministic | time blocks are disjoint and no reduction is repartitioned; W3 asserts identical fingerprints for four worker counts, and W6 proves the partition is a bijection onto `[0, n_times)` |
| Thread parallelism deadlocks or contends on a shared backend or beam handler | 6E must exercise shared FITS beam handlers under `workers=4`; if any handler is stateful, the slice makes the per-worker copy explicit rather than sharing it |
| Hybrid double counts a real physical overlap | the existing gate is reused unchanged (§10.1); H5 proves the arithmetic and the `assume_disjoint` warning path |
| C8/C9 break a user configuration that silently worked | pre-v1 policy; both messages name `hybrid` and the explicit opt-in; the migration guide records both |
| The `scientific_sha256` change strands recorded Tier 4/5 evidence | C11 is declared; prior acceptance records are historical and are not rewritten |
| HDF5 `3.0.0` strands previously written files | pre-v1 policy, following the accepted `2.0.0` precedent; the rejection message names Tier 6 and there is no upgrade path by design |
| The single compiled kernel diverges from its reference under some preset | B3 asserts value and dtype agreement for every preset the kernel is exercised with; the uncompiled path stays the reference and is never removed |
| A benchmark number is quoted without its record | criterion 22 makes an uncited capability statement an acceptance failure |
| The `numba`→`dask` rename is read as a capability gain | the class docstring, the backend documentation, and the migration entry all state that no compilation ever occurred and none is added |
| Scope creep into Tier 7 Jones work or an m-mode solver | §40 exclusions; slice file lists exclude every Jones term file and every simulator strategy file except `rime.py` |
| `JAXBackend.name` reads `"jax-cpu-cpu"` on a CPU device (confirmed today in `backends/jax_backend.py:10`'s own doctest and its `f"jax-{platform}-{backend_name}"` construction at `:146`, where both `device.platform` and `jax.default_backend()` return `"cpu"`) | pre-existing, not introduced by Tier 6; 6H's B4/B5 registry-truthfulness tests must assert this exact (if inelegant) string rather than a cleaner name invented for the occasion — the name is truthful, only repetitive, and no slice may silently change the format without recording it as its own decision |

**Three rows added at Tier 6H independent acceptance (2026-07-31), no decision
change.** All three are properties of the §13.6 compiled kernel and the §14.1
`auto` precedence that no slice measured before now; none contradicts a binding
6H contract, none reproduces at the §13.4 workload/shipped-config scale the
gate actually exercises, and none blocks acceptance. Each is routed to a named
later obligation rather than left as an unrecorded gap.

| The compiled kernel's `(B, S, 2, 2)` working set (`core/contraction.py`) is `O(baselines × sources)` per `(time, frequency)` step, not `O(sources)` like the pre-6H per-baseline Python loop it replaced | measured directly at acceptance: NumPy peak traced memory scales linearly at ≈208 bytes per `(baseline, source)` pair (B=S ∈ {100, 1000, 5000} all agree to within rounding). Every §13.4 workload and both shipped configs stay at ≤15 baselines, so R1/B1/B2 exercise correctness, never this scaling — a realistic array (tens of thousands of baselines) against a populated catalog (10⁴-10⁵ sources) would need tens to hundreds of GB for one step. Filed as a 6I obligation: the benchmark harness (§22) must add a memory-vs-`(B, S)` scaling record, so the hazard is either bounded by evidence or its one-line mitigation (chunk the baseline axis inside `baseline_contraction_for`) becomes a tracked task rather than a silent gap |
| `get_backend("auto")`'s device probe (`_has_non_cpu_jax_device()`, `backends/__init__.py`) performs a real `import jax` whenever `JAX_AVAILABLE`, which C18 makes true in every declared environment, even though `auto` only ever resolves to NumPy on the locked CPU-only build | measured directly at acceptance: ~450-950ms added to the first `get_backend("auto")` call on this host — the same "roughly a second of XLA start-up" `jax_backend.py`'s own docstring says the lazy-import design exists to avoid, reintroduced through the auto-precedence path rather than the eager-import path that design closed. Reaches `execution.backend: auto`, the `radiosim simulate` CLI subcommand's default (`cli/main.py:208`); the config-file path's schema default is `numpy` (`io/config.py:1543`) and is unaffected. Not a truthfulness defect — the returned backend and its provenance are correct — and not a violation of any §13.6/§14.1 binding text, so it does not block 6H; recorded so a later slice can choose to gate or cache the device probe instead of leaving the regression unrecorded |
| §13.6 states the compiled kernel is "shape-stable within a run"; both solvers mask sources/pixels by `above_horizon` per time step (`core/visibility.py`, `core/visibility_healpix.py`), so the kernel's source axis can change size step-to-step within one run, not only between runs | confirmed by reading both solvers' time loops. Under `execution.backend: jax` a shape change forces `jax.jit` to retrace, which the 6I benchmark methodology as specified (§22.2: first call is setup, steady state is the median of repeated *identical* calls) does not measure, because its timing loop repeats one fixed workload rather than a multi-step run with a changing visible-source count. Not a correctness defect — B1-B3 prove the compiled and uncompiled forms agree at every exercised shape — and every §13.4 workload's short duration/fixed horizon avoids exercising it. Filed as an explicit 6I obligation: measure and report per-step retracing behavior for a workload whose visible-source count changes across the time axis, rather than leaving the stability claim unverified at realistic observation lengths |

## 40. Explicit exclusions and Tier 7+ boundary

Tier 6 does not implement:

- any Jones term — `Z`, `T`, `P`, `D`, `G`, `B`, `F`, `W`, `Ee`/`a`/`dE`,
  `Kd`/`Rc`/`ff`, `X`/`Kx`/`DF`, and baseline `M`/`Q` remain identity stubs
  owned by Tier 7 (`SCI-001`);
- the spherical-harmonic / m-mode solver, which stays rejected at config
  validation (`io/config.py:1995-2000`) until Tier 7 (`SCI-002`);
- any Numba-compiled kernel, any second implementation of the forward model, or
  any GPU kernel;
- distributed execution, Dask cluster orchestration as a compute path, or any
  scheduler policy;
- process-based solver parallelism;
- baseline- or frequency-axis parallelism;
- replacing astropy coordinate transforms, or moving beam interpolation off the
  host;
- gridding, FFT, or w-projection solvers;
- a repository-wide documentation rewrite, CI redesign, or release work beyond
  the one added jax-cpu job (Tier 8);
- physical GPU or TPU validation, live network validation, registry validation,
  deployment, tagging, or release.

## 41. Open questions

Each question names the slice that must resolve it, the evidence required, and
what happens if the evidence contradicts this plan. No slice may proceed past a
question that blocks it by assuming an answer.

**Q1 — Is a CPU-only JAX installable on all three locked platforms under the
existing NumPy pin? (blocks 6H; must be answered in 6A.)** §28 adopts a jax-cpu
pixi environment as the only acceptable way to satisfy the mandated NumPy/JAX
parity evidence. 6A must record exact resolvable versions of `jax` and `jaxlib`
for `linux-64`, `osx-64`, and `osx-arm64` under `numpy >=1.24,<2.5`
(`pixi.toml:29-32`), from conda-forge or PyPI, and must confirm that
`jax_enable_x64` yields true float64/complex128 for the solver dtypes on that
build. If a platform cannot resolve, the fallback is a **separate optional
environment covering only the platforms that can**, with the parity job running
there and the acceptance record stating exactly which platforms were measured
and which were not — never a silent return to `importorskip`. If no platform can
resolve, §13.4, §27 B1/B7, and criterion 15 must be amended and re-accepted
before 6H, and `RUN-004` cannot close.

**Q2 — Are the FITS beam handlers and the `BeamSystem` safe to share across
solver threads? (blocks 6E; must be answered in 6A or early 6E.)** The per-antenna
Jones cache is rebuilt per `(time, frequency)` (`core/visibility.py:571-585`) and
the HEALPix path caches by handler id inside the frequency loop
(`core/visibility_healpix.py:149-186`), so the *solver* holds no cross-time
state. **Correction (2026-07-30, Tier 6A acceptance):** the class named below,
`BeamFITSHandler`, no longer exists anywhere in `src/radiosim`; the current shape
is `core/beam/fits.py`'s `_LoadedFITSHandler` (wrapping pyuvdata `UVBeam.interp`),
reached through `core/beam/runtime.py`'s `BeamSystem.evaluate_jones`. `BeamManager`
was likewise already removed (Tier 3); the pre-Tier-6 flat/`BeamManager` inputs
are rejected outright (`io/config.py:2192`). Whether `_LoadedFITSHandler` /
pyuvdata `UVBeam` interpolation is thread-safe is not established at this gate.
Evidence required: a concurrent evaluation of one shared FITS handler from four
threads, compared against serial evaluation for bit-identity. If it is not
thread-safe, 6E must give each worker its own handler instance (a construction
change, not a numerical one), and the slice's file list gains `core/beam/fits.py`
and/or `core/beam/runtime.py` (the modules that construct and evaluate FITS beam
handlers today) — a bounded correction, not a design change.

**Q3 — Does the per-time restructure change peak memory materially? (blocks
6D's acceptance framing, not its implementation.)** Holding `T` blocks of
`(F, B, 2, 2)` before one assembly has the same asymptotic footprint as the
current pre-allocated cube, but it transiently holds both. Evidence required:
`tracemalloc` peak for the largest shipped configuration, before and after, in
6D's record. If the transient doubles peak host memory for a realistic size, 6D
must assemble incrementally into a pre-allocated host cube for the NumPy
backend while keeping the block structure for functional backends — a documented
per-backend assembly strategy, decided on 6D's measured evidence rather than
guessed here.

**Q4 — Should the remaining accelerator work be a new issue or a reopened
`RUN-004`? (blocks nothing; must be decided in 6J.)** §38 closes `RUN-004` only
for the scope of §13.1. The unfinished part — device-resident orchestration,
device coordinate transforms, and a measured accelerator run — is a distinct
piece of work with its own hardware prerequisite. 6J must either file it as a
new `PERF-001` roadmap row or leave `RUN-004` open with an explicitly narrowed
description. It must not be silently absorbed into Tier 7's Jones workstreams.

**Q5 — Does any shipped or documented configuration rely on the silent
rasterization that C9 makes opt-in? (blocks 6F.)** At this gate the three
shipped configurations are `point_sources`, `point_sources`, and a diffuse-only
`healpix_map` (`configs/config.yaml:66`,
`configs/receptor_circular_example.yaml:76`,
`configs/realistic_foreground_example.yaml:66`), so none appears to rely on it.
6F must confirm the same for every example script, every doctest, every fixture
in `tests/fixtures/configs.py`, and every documentation snippet before
introducing the rejection. If any does, 6F sets the new flag explicitly in that
artifact rather than weakening the rule.
