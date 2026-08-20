# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

NOTE: Never think about backward compatibility in the code when doing edits, or adding features until told to do so. We are in beta right now and there are no users. So it doesn't matter if we are doing breaking changes. 

## Project Overview

**RadioSim** is a Python package for simulating radio interferometer visibilities using the RIME (Radio Interferometer Measurement Equation) with full polarization support and multiple sky models. Version 0.4.0 (Beta), Python 3.11+. Both solvers route their array work through a selectable NumPy/JAX/Dask backend and one kernel is compiled under JAX, but no accelerator has ever been measured — see Implementation Status below.

## Implementation Status (read this first)

RadioSim is mid-build, but the Jones surface is no longer part of what is missing. When working here, assume:

- **Jones terms**: every exported term implements real physics and declares `term_status: "implemented"`. There are no identity stubs left — Tier 7 deleted twenty-six classes that returned the 2×2 identity for every input rather than keeping them as scaffolding, and both evaluation contracts (`JonesTerm.compute_jones_batch`, `JonesBaselineTerm.compute_baseline_factor`) are `@abstractmethod`, so a class without physics cannot be constructed. The forward model reflects **K** (geometric phase, a per-baseline *function* `geometric_phase()`, not a class), **E** (the canonical `BeamSystem`, reached through a private solver-owned adapter), **C**/**H** (`ReceptorConfigJones` / `BasisTransformJones`, always in the chain; the default unrotated linear array has `C=P` and `H=I₂`), plus the nine terms a `jones:` config section enables — **G**, **B**, **Rc**, **Kd**, **X**, **D**, **P**, **T**, **Z** — and the two baseline-dependent Hadamard terms **M** and **Q**. Omitting `jones:` selects the empty optional-term inventory; it does not remove **E**, **C**, **H**, or **K**. SCI-006 intentionally changed polarized linear results relative to the pre-Tier-7 model; see `docs/migration_guide.md`. A `jones:` block whose resolved parameters would make its term exactly the identity is *rejected*, not silently accepted.
- **Cross-validation**: Tier-1 evidence (published closed forms evaluated independently in the test body; astropy and pyuvdata 3.2.1 as independent references) is in the standard gate. Tier-2 evidence — a comparison against the `pyuvsim 1.4.0` simulator — lives in the optional `crossval` pixi environment, never gates, and is recorded in `output/crossvalidation/`. Never write a validation claim without naming the quantity and the tolerance, and never write "validated against pyuvsim" without citing that record.
- **Backends**: `ArrayBackend` (NumPy/JAX/Dask), `get_backend()`, and `list_backends()` are complete, and **both** solvers — `core/visibility.py` and `core/visibility_healpix.py` — route their Jones chain, geometric phase, coherency construction, contraction, and accumulation through the backend. Accumulation is per-time block assembly (`backend.stack`), not per-cell `set_at`. Exactly one kernel is compiled: `core/contraction.py`'s baseline-batched per-`(time, frequency)` contraction, via `ArrayBackend.compile` (`jax.jit` on the JAX backend), and that module is the only `backend.compile` call site in `src/`. **GPU acceleration is still unrealized and unmeasured**: the time and frequency axes are host-side Python loops, astropy coordinate transforms / horizon masking / Planck conversion / pyuvdata beam interpolation are host-side by design, the standard gates use CPU-only JAX, and the isolated Linux GPU environment is readiness infrastructure rather than accepted evidence. Measured JAX-CPU is *slower* than NumPy on every benchmarked workload (records: `output/benchmarks/reference/`, methodology: `docs/user_guide/backends.rst`, register: `PERF-001`). Backend *correctness* parity is complete (Dask bit-identical to NumPy, JAX-CPU within `rtol=1e-12`); backend *performance* is a roadmap item. Never write a speed or GPU claim without citing a record file.
- **`visibility.calculation_type`**: removed before v1.0. It validated `direct_sum` and `spherical_harmonic` and nothing read either value; `spherical_harmonic` was rejected during config validation and never reached the runtime. The solver strategy is selected by `execution.simulator`, whose accepted values are exactly the keys of the simulator registry (currently only `rime`). A document that still sets the removed key is rejected with guidance naming the replacement. A spherical-harmonic or m-mode solver is a future simulator registration, not a value on a removed field.

## Development Commands

All commands use **pixi** (manages the conda environment automatically).

**IMPORTANT**: `pixi run pytest` does NOT work — `pytest` is not directly on the pixi PATH. Use `pixi run test` (the pixi task) or `pixi run python -m pytest` instead. To pass extra args to pytest through the task, use `pixi run test -- <args>`.

```bash
pixi run test                                        # Run all tests
pixi run test -- tests/unit/test_core/test_sky_model.py  # Specific file
pixi run test -- tests/unit/test_core/test_sky_model.py::TestClassName::test_name  # Single test
pixi run test -- -k "keyword"                        # Filter by keyword
pixi run test -- -v --tb=short                       # Verbose with short tracebacks
pixi run test -- -m "not slow"                       # Skip slow tests
pixi run test -- -m gpu                              # GPU-only tests
pixi run test -- -m integration                      # Integration tests
pixi run test -- --cov=radiosim --cov-report=html      # With coverage (target: 70%)

pixi run bench                                       # Reproducible backend benchmarks (marked performance+slow; never gates)

pixi run format                                      # Format (ruff format)
pixi run fix                                         # Lint + autofix (ruff check)
pixi run typecheck                                   # Type check (Pyright, against the checked-in error ceiling)
pixi run doctest                                     # Docstring doctests, scoped to src/radiosim

pixi run radiosim --config config.yaml                 # Run simulation from YAML
pixi run radiosim validate config.yaml                 # Validate config
pixi run radiosim init                                 # Generate config template
pixi run radiosim simulate --antenna-layout antenna_layout_examples/hera_5.txt --frequencies 100,150,200 --telescope-name HERA --default-diameter-m 14 --latitude -30.7 --longitude 21.4 --height 1073 --start-time 2025-01-01T00:00:00
```

Shorthand pixi tasks: `pixi run test`, `pixi run lint`, `pixi run fix`, `pixi run format`, `pixi run typecheck`, `pixi run doctest`.

**Note**: Do NOT run `pixi run typecheck` unless explicitly asked — it is slow and not part of the standard workflow.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  USER API LAYER: api/simulator.py (Simulator class)  │
├──────────────────────────────────────────────────────┤
│  ALGORITHM LAYER: simulator/rime.py (RIMESimulator)  │
├──────────────────────────────────────────────────────┤
│  PHYSICS LAYER: core/* (RIME, Jones, sky models)     │
├──────────────────────────────────────────────────────┤
│  HARDWARE LAYER: backends/* (NumPy, JAX, Dask)       │
└──────────────────────────────────────────────────────┘
```

Source lives in `src/radiosim/`. Tests live in `tests/` under `unit/`, `integration/`, `characterization/` (golden fingerprint pins), `performance/` (benchmark records; never gates), `crossvalidation/` (the optional `crossval` environment), plus the shared `fixtures/` and `support/` helper packages. `simulators/` is **not** package code: it holds 41 third-party simulator checkouts as git submodules for reference reading, excluded from the wheel and from Ruff. A plain `git clone` does not fetch them — they arrive only with `--recursive` or `git submodule update --init`, and cost roughly 3.9 GB checked out. Key entry points:

- **`api/simulator.py`** — `Simulator` class: `from_config()`, `setup()`, `run()`, `plot()`, `save()`. Recommended entry point.
- **`cli/main.py`** — Click-based CLI. Primary mode: `radiosim --config config.yaml`. Subcommands: `simulate`, `init`, `validate`.
- **`io/config.py`** — Pydantic v2 config classes. Top-level: `RadioSimConfig`. Loaded via `load_config()`.

### Core Physics (`core/`)

- **`visibility.py`** — Core RIME: `V_pq = Σ J_p @ C @ J_q^H` for point sources
- **`visibility_healpix.py`** — RIME for HEALPix diffuse maps (requires a populated `sky.healpix` payload)
- **`polarization.py`** — Stokes ↔ Coherency with 1/2 factor: `B = (1/2) × [[I+Q, U+iV], [U-iV, I-Q]]` (IAU Stokes V; the mirror-image sign was corrected in Tier 5C)
- **`polarization_basis.py`** — the single canonical correlation-coordinate table: `PolarizationBasis` (`linear_xy` / `circular_rl`), labels, AIPS codes in both canonical and file order, pyuvdata feeds, `basis_for_correlations()`, `parallel_hand_indices()`
- **`receptor.py`** — `resolve_receptors()`, `ResolvedReceptorSet`, typed receptor errors; owns the receptor vocabulary (`receptor`, `feed`, `basis`, `feed_rotation`)
- **`instrument.py` / `instrument_resolution.py`** — Canonical frozen instrument models and source-aware resolution; strict source loaders live in `io/instrument_sources.py`
- **`baseline_resolution.py`** — Generates and selects canonical frozen baselines
- **`observation.py`** — Observation context (time, location, pointing)
- **`precision.py`** — `PrecisionConfig` with presets: `.standard()`, `.fast()`, `.precise()`, `.ultra()`

### Sky Model (`core/sky/`)

`SkyModel` (frozen Pydantic dataclass in `containers/model.py`) is the central data container. It holds two optional payloads — `sky.point` (`PointSourceData`) and `sky.healpix` (`HealpixData`) — and a model may carry either or both ("hybrid"). The package is organized into subpackages (NOT the flat layout previously documented):

- `containers/` — frozen dataclasses: `model.py` (`SkyModel`), `point.py` (`PointSourceData` + `PointMorphology`/`PointPolarization`/`PointMetadata`/`PointSpectrum` sub-blocks), `healpix.py` (`HealpixData`, with first-class sparse support: `hpx_inds`, `is_sparse`, `require_dense`, `to_dense`, NEST↔RING `reordered`), `spectral.py`, `footprint.py`, `provenance.py` (`SkyProvenance`), `constants.py`. (There is no `containers/data.py` — container arrays are read-only after construction, enforced by `_shared._freeze`.)
- `loaders/` — module-level loader functions registered via `loader_registry.register_loader(...)` (there is no free `@register_loader` symbol; the internal `LoaderRegistry.register` is private).
- `registry/` — loader registry (`core.py`, `facade.py`) + catalog metadata (`catalogs.py`).
- `operations/` — mutation-free transforms (`operations.py`), `factories.py`, `convert.py`, `region.py`.
- `combine/` — `prepare_sky_model()` (public) wraps `_combine_models()` (internal); includes physical-disjointness checks and optional `assume_disjoint` escape (skips double-count rules only; monopole checks still run). There is **no** `combine_models()`.
- `diagnostics/`, `recipes/`, `support/`, `io/` — subpackages have populated `__init__.py` with selective re-exports; `support/healpy.py` provides lazy healpy access so point-only import paths do not load healpy at module load.

Use typed loader functions from `radiosim.core.sky.loaders` for programmatic work, factory functions `create_from_arrays()` / `create_test_sources()` for constructed models, and `prepare_sky_model()` for combining/materializing.

| Loader file | Registered names | Representation |
|-------------|-----------------|------|
| `loaders/vizier/point_catalogs.py` | `gleam`, `mals`, `lotss`, `vlssr`, `tgss`, `wenss`, `sumss`, `nvss`, `3c`, `vlass` | `"point_sources"` |
| `loaders/vizier/racs.py` | `racs` (CASDA TAP) | `"point_sources"` |
| `loaders/bbs.py` | `bbs` | `"point_sources"` |
| `loaders/diffuse.py` | `diffuse_sky` (aliases: `gsm`, `gsm2008`, `gsm2016`, `lfsm`, `haslam`), `pysm3` | `"healpix_map"` |
| `loaders/fits.py` | `fits_image` | `"healpix_map"` |
| `recipes/realistic_foreground.py` | `realistic_foreground` | `"healpix_map"` |
| `loaders/pyradiosky.py` | `pyradiosky_file` | either |
| `loaders/skyh5_multifile.py` | `skyh5_multifile` | either |
| `loaders/synthetic.py` | `test_sources`, `poisson_confusion` | either |

Loader registry metadata is the single source of truth for config fields, aliases, network services, source category, and source representation. Catalog parameter tables live in `registry/catalogs.py` (registry fields such as `config_section`/`use_flag` are derived from catalog entries); physical constants in `containers/constants.py`. Import from `radiosim.core.sky.registry` package root, not submodule paths. NEST ordering is fully threaded through convert/combine/subtract/regrid/region paths; diffuse loaders remain ring-native with explicit `order_in/out` on `ud_grade`.

Canonical representations (`SkyFormat` enum):
- **`SkyFormat.POINT_SOURCES = "point_sources"`** — columnar `PointSourceData` arrays: `ra_rad`, `dec_rad`, `flux`, `spectral_index`, `stokes_q`, `stokes_u`, `stokes_v`, `ref_freq`, plus optional morphology / polarization / metadata / per-channel `spectrum` sub-blocks.
- **`SkyFormat.HEALPIX = "healpix_map"`** — `HealpixData` `maps` of shape `(n_freq, npix)` (Kelvin) plus `frequencies`, `nside`, optional `q/u/v_maps`, and optional sparse `hpx_inds`.

`SkyModel` is frozen — mutate via `.replace(**changes)`. There is **no** `source_format` / `available_formats`; use `SkyModel.formats` (a `set[SkyFormat]` derived from which payloads are populated). Access payloads through `sky.point` / `sky.healpix`; do not add top-level passthrough properties.

Key API: `load_diffuse_sky()` requires `frequencies=np.ndarray` or `obs_frequency_config=` (not a single float). Use `materialize_healpix_model()` to convert point sources to HEALPix and `materialize_point_sources_model(lossy=True)` only for explicit HEALPix→point conversion. Access maps via `HealpixData.get_multifreq_maps()` / `get_map_at_frequency()`. All loaders and factory functions require `precision=PrecisionConfig(...)`.

### Jones Matrix Framework (`core/jones/`)

`core/jones/__init__.py` exports exactly **19 names**: three base classes (`JonesTerm`, `JonesChain`, `JonesBaselineTerm`), thirteen concrete terms, and three non-class exports (`DirectionBatch`, `evaluate_antenna_jones`, `geometric_phase`). **Every exported term implements real physics** (see Implementation Status); `term_status` is `"implemented"` for all of them and there is no `"planned"` value in use.

Evaluation is direction-batched: a term implements `compute_jones_batch()` returning `(n_dir, 2, 2)` for a direction-dependent effect or `(1, 2, 2)` for a direction-independent one, and both solvers call the one shared `evaluate_antenna_jones()` in `evaluate.py` over a `DirectionBatch` (`directions.py`).

**The chain terms** (canonical order `J_p = H_p G_p B_p Rc_p Kd_p X_p D_p C_p E_p P_p T_p Z_p`, leftmost nearest the correlator; K is applied separately by the solver because it is per-baseline):

| Term | File | Class or function | Type |
|------|------|-------------------|------|
| K (Geometric Phase) | `geometric.py` | `geometric_phase()` — a function, not a chain term | per-baseline |
| Z (Ionosphere) | `ionosphere.py` | `IonosphereJones` | DDE |
| T (Troposphere) | `troposphere.py` | `TroposphereJones` | DDE |
| P (Parallactic Angle) | `parallactic.py` | `ParallacticAngleJones` | DDE |
| E (Primary Beam) | `beam/` | canonical `BeamSystem` via a private solver-owned adapter | DDE |
| C (Receptor Config) | `receptor.py` | `ReceptorConfigJones` | DIE |
| D (Pol. Leakage) | `polarization_leakage.py` | `PolarizationLeakageJones` | DIE |
| X (Cross-hand) | `crosshand.py` | `CrosshandJones` | DIE |
| Kd (Instrumental Delay) | `delay.py` | `DelayJones` | DIE |
| Rc (Cable Reflection) | `delay.py` | `CableReflectionJones` | DIE |
| B (Bandpass) | `bandpass.py` | `BandpassJones` | DIE |
| G (Gains) | `gain.py` | `GainJones` | DIE |
| H (Basis Transform) | `receptor.py` | `BasisTransformJones` | DIE |

**The two terms outside the chain**: `baseline_errors.py` holds `BaselineMultiplicativeJones` (M, per-baseline closure error) and `SmearingFactorJones` (Q, time and bandwidth smearing). Both descend from `JonesBaselineTerm`, apply by **Hadamard product** to finished visibilities, and are rejected by `JonesChain.add_term` with a `TypeError`.

Beam internals (the E term): `core/beam/` owns the canonical runtime — `runtime.py` (`BeamSystem`, `load_beam_system()`, plus `ruze_power_efficiency()` / `ruze_voltage_factor()`), `fits.py` (pyuvdata UVBeam, peak normalization, shared/per-antenna assignment), `models.py`, `resolution.py`. `core/jones/beam/analytic/` is a package of composable analytic beam *formulae* — aperture shapes (`circular`/Airy, `rectangular`/sinc, `elliptical`), illumination tapers (`uniform`, `gaussian`, `parabolic`, `parabolic_squared`, `cosine`), aperture illumination patterns (`illumination.py`: corrugated horn, open waveguide, dipole over ground plane), and reflector geometries; `compute_hpbw_numerical()` (in `numerical_hpbw.py`) is a diagnostic-only HPBW finder. Tier 7I added per-antenna deterministic pointing offsets and the Ruze surface-efficiency factor to `BeamSystem`. The accepted E-Jones is **scalar** (`E = e · I2`) except under two accepted `SCI-005` subsets: Stage-2 `beams.squint` (accepted 2026-08-19, analytic mode only), where the runtime composes the generally full `E = C† D_b C` from the two oppositely displaced native-feed samples and the antenna's resolved receptor matrix; and Stage-3 full-efield UVBeam files accepted under the `uvbeam_peak_common_v1` normalization literal (accepted 2026-08-20, mutually exclusive with `beams.squint`), where the runtime composes the generally full `E = C† J_native` from the file's native-feed complex efield samples via the fixed chain-tangent conversion and the antenna's resolved receptor matrix. `docs/development/beam_physics_scope.md` gives every remaining beam-physics item a disposition, a citation, and an owning register row. The old `AntennaType` class and named beam types (`airy`, `cosine`, `exponential`, `short_dipole`) have been removed.

**Terminology (do not mix)**: aperture **illumination** belongs to the beam subsystem and uses `illumination` / `taper` / `edge_angle`; the receiving **receptor** belongs to `core/receptor.py` and uses `receptor` / `feed` / `basis` / `feed_rotation`. Do not introduce `feed`-named identifiers into the beam subsystem or `illumination`-named identifiers into the receptor subsystem.

**Architecture note**: `JonesBaselineTerm` in `baseline_errors.py` is a separate ABC for baseline-dependent effects (M, Q) that apply via Hadamard multiplication, NOT the matrix chain. They cannot be added to `JonesChain`.

**Removed modules**: `faraday.py` (F), `wterm.py` (W), and `element_beam.py` (Ee/a/dE) no longer exist. Ionospheric Faraday rotation is owned by `Z`; intrinsic source RM is applied by the sky model; the direct-sum RIME already carries `w(n-1)` exactly, so a W term would double-count. `docs/migration_guide.md` names the replacement for every removed class.

To add a new Jones term: extend `JonesTerm` (or `JonesBaselineTerm`), implement `name`, `is_direction_dependent`, `compute_jones_batch()` (or `compute_baseline_factor()`), export in `__init__.py`, add the typed config block, and add tests in `tests/unit/test_jones/` — a Fix.md §16 term needs an analytic invariant test, a backend-parity case, and an effect-changes-visibility case.

### Simulator Layer (`simulator/`)

- `base.py` — `VisibilitySimulator` ABC (Strategy pattern for swappable algorithms)
- `rime.py` — `RIMESimulator`: direct RIME summation, O(N_src × N_bl × N_freq)

### Backends (`backends/`)

- `base.py` — Abstract `ArrayBackend` interface, including `stack`, `add`, `supports_compilation`, `compile`, and `synchronize(arr)`
- `numpy_backend.py` — CPU (always available); the reference every other backend is compared against
- `jax_backend.py` — JAX/XLA `ArrayBackend`; the standard gates use the CPU-only build, while the isolated Linux `gpu` environment is readiness-only (`PERF-001`). The only backend with `supports_compilation is True`
- `dask_backend.py` — NumPy, optionally through Dask arrays. Renamed from `numba_backend.py` in Tier 6H because that class never compiled anything; `jit_compile` and `mode="gpu"` are gone
- Selection: `get_backend("auto" | "numpy" | "jax" | "dask")`; explicit discovery: `list_backends()` / `get_backend_info()`. `auto` deterministically selects NumPy without importing or probing JAX and never selects Dask. Generic `jax` uses the runtime-default device; named JAX devices and the `gpu` / `tpu` aliases are strict and never fall back to CPU (`PERF-001`)
- The `VisibilitySimulator.supports_gpu` default and `RIMESimulator.supports_gpu` are both `False`; only an independently accepted end-to-end accelerator record may support `True`. `get_backend("numba")` raises

### Benchmarks (`benchmarks/`)

- `record.py` — `BenchmarkRecord` (complete or `BenchmarkRecordError`; no partial record), plus `RetracingRecord` and `MemoryScalingRecord`
- `harness.py` — the timing discipline: setup vs steady state, compile time, `synchronize` before every clock stop, host transfer timed around `to_numpy` alone, `tracemalloc` peak in a separate untimed pass, correctness delta vs NumPy
- Run with `pixi run bench` (marked `performance` + `slow`, never gates). Output: `output/benchmarks/<UTC timestamp>-<host tag>.json`, gitignored; the committed reference set is `output/benchmarks/reference/`

### I/O (`io/`)

- `config.py` — Pydantic v2 config models, top-level `RadioSimConfig`, loaded via `load_config()`
- `config_resolution.py` — resolves a loaded document plus `SimulationOverrides` into the frozen runtime inputs
- `instrument_config.py` — strict frozen instrument and baseline-selection inputs
- `beam_config.py` — strict typed loader for the `beams:` config section
- `instrument_sources.py` — strict loaders for local layouts, datasets, and known telescopes
- `hdf5.py` / `summary_json.py` / `standard_visibility.py` / `uvfits.py` — the result writers (HDF5, JSON summary, pyuvdata `UVData`, UVFITS); `readers.py` reads them back
- `measurement_set.py` — CASA Measurement Set export (requires python-casacore)
- `atomic_paths.py` / `result_format.py` / `workflow_artifacts.py` — atomic publication paths, `ResultFormat` selection, and the staged run directory
- `jones_config.py` / `receptor_config.py` — strict typed loaders for the `jones:` and `receptors:` config sections
- `fits_utils.py` — FITS file utilities

There is no `io/writers.py`; each writer is its own module in the list above.

### Utilities (`utils/`)

- `validation.py` — Pre-flight config validator that collects all errors at once
- `logging.py` — Logging configuration

## The RIME Equation

`V_pq(ν, t) = Σ_s J_p(s, ν, t) · C_s(ν) · J_q^H(s, ν, t)`

**Critical**: The coherency matrix uses a 1/2 factor (`B = (1/2) × [[I+Q, U+iV], [U-iV, I-Q]]`, IAU Stokes V), so `Tr(B) = I` rather than `2I`. In the ideal matched unit-response case this gives `V_XX + V_YY = I` and `V_RR + V_LL = I`; arbitrary heterogeneous or non-unitary Jones chains need not preserve those sums. The canonical Jones chain order is `J_p = H_p G_p B_p Rc_p Kd_p X_p D_p C_p E_p P_p T_p Z_p` with K applied separately, leftmost nearest the correlator; `JonesChain` composes `terms[0] @ ... @ terms[-1]`, so terms are added in that same order. `P` sits **sky-side** of `C` (Tier 7F corrected the Tier 5 factorization, which is wrong for a circular receptor), and `M`/`Q` are outside this product entirely. Precision is fully controlled by PrecisionConfig — respect the user's chosen dtype everywhere, including coordinate conversions and phase calculations.

## Code Style

- **Formatter**: Black (88 char line length)
- **Linter**: Ruff (E, W, F, I, B, C4, UP rules; E501 ignored)
- **Type checker**: Pyright in strict mode, pinned `pyright ==1.1.408`, configured under `[tool.pyright]` in `pyproject.toml`. `pixi run typecheck` runs `python tools/check_pyright_baseline.py`, which fails only if the strict error total rises above the checked-in ceiling in `pyright-baseline.json`; `pixi run typecheck-report` prints the plain report and `pixi run typecheck-update` lowers the ceiling (it refuses to raise it). MyPy is not used anywhere in this repository
- **Commits**: Conventional format (`feat:`, `fix:`, `refactor:`, `test:`, `chore:`). Never include co-authored-by lines.

### Git workflow

After completing and verifying any small, coherent task that changes repository
files, create a local commit automatically before handing the work back. Do not
wait for a separate request to commit. Keep each commit narrowly scoped and use
the conventional message format above. Never push, create a pull request, or
otherwise publish commits without first asking the user and receiving explicit
approval.

## Configuration

YAML config is validated by the strict Pydantic `RadioSimConfig` model and resolved by `load_config()`. Its top-level sections are `instrument`, `beams`, `receptors`, `baseline_selection`, `sky_model`, `obs_time`, `obs_frequency`, `visibility`, `jones`, `execution`, and `workflow`. The `instrument` section selects exactly one typed source and owns location and diameter precedence; `baseline_selection` owns the canonical correlation, length, and axial-azimuth filters; `receptors` owns the per-antenna receptor basis, the static `feed_rotation_deg`, and the single array-wide `output_basis` that names the reported correlation labels. The optional `jones` section carries one block per enabled term (`G`, `B`, `Rc`, `Kd`, `X`, `D`, `P`, `Z`, `T`, `M`, `Q`); there is no `enabled: false` — delete the block instead — and a block resolving to the identity is rejected. See `configs/` for complete examples.

The pre-`v1.0` policy this file opens with is written down for contributors too, in `docs/contributing.rst` ("Pre-v1 API Evolution Policy"): refactors before `v1.0` move directly to the cleaner replacement rather than preserving backward compatibility, unless a deprecation path is explicitly requested, and every breaking change still lands in `docs/changelog.rst` or `docs/migration_guide.md`.
