# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

NOTE: Never think about backward compatibility in the code when doing edits, or adding features until told to do so. We are in beta right now and there are no users. So it doesn't matter if we are doing breaking changes. 

## Project Overview

**RadioSim** is a Python package for simulating radio interferometer visibilities using the RIME (Radio Interferometer Measurement Equation) with full polarization support and multiple sky models. Version 0.2.0 (Beta), Python 3.11+. GPU backends (JAX/Numba) are scaffolded but the RIME compute path currently runs on NumPy — see Implementation Status below.

## Implementation Status (read this first)

RadioSim is mid-build; several advertised capabilities are scaffolded but not yet wired into the compute path. When working here, assume:

- **Jones terms**: Only **K** (`GeometricPhaseJones`, geometric phase) and **E** (the beam classes) implement real physics. Every other term — Z, T, P, D, G, B, F, W, C/H, Ee/a/dE, Kd/Rc/ff, X/Kx/DF, and baseline M/Q — is a stub whose `compute_jones()` returns the 2×2 identity (`TODO: implement properly`). They can be added to a chain but multiply by identity, so the forward model currently reflects only K (fringe) + E (beam).
- **Backends**: `ArrayBackend` (NumPy/JAX/Numba), `get_backend()`, and `list_backends()` are complete, and the point-source RIME hot path in `core/visibility.py` *does* route its array ops through the backend (`backend.matmul`/`asarray`/`set_at`/`exp`/`conjugate_transpose`/`sum`); `JonesChain` composes terms with functional `backend.matmul` + `batch_eye` (no invalid in-place writes — `set_at` uses JAX's `.at[].set()`). **But GPU acceleration is still unrealized**: the per-time × per-frequency × per-baseline orchestration is host-side Python loops, coordinate transforms run on astropy/NumPy, and `jit`/`vmap`/`jit_compile` are defined but never applied — so a JAX backend bounces device↔host each iteration rather than accelerating (and `core/visibility_healpix.py` still uses bare `np.*` in places). Treat GPU acceleration as a roadmap item: the backend abstraction is wired into the matmul but is not yet performance-bearing.
- **`visibility.calculation_type`**: only `direct_sum` works; `spherical_harmonic` passes config validation but raises `NotImplementedError` at runtime.

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

pixi run format                                      # Format (ruff format)
pixi run fix                                         # Lint + autofix (ruff check)
pixi run typecheck                                   # Type check (mypy)

pixi run radiosim --config config.yaml                 # Run simulation from YAML
pixi run radiosim validate config.yaml                 # Validate config
pixi run radiosim init                                 # Generate config template
pixi run radiosim simulate --antenna-layout X --frequencies 100,150,200  # CLI simulate mode
```

Shorthand pixi tasks: `pixi run test`, `pixi run lint`, `pixi run fix`, `pixi run format`, `pixi run typecheck`.

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
│  HARDWARE LAYER: backends/* (NumPy, JAX, Numba)      │
└──────────────────────────────────────────────────────┘
```

Source lives in `src/radiosim/`. Key entry points:

- **`api/simulator.py`** — `Simulator` class: `from_config()`, `setup()`, `run()`, `plot()`, `save()`. Recommended entry point.
- **`cli/main.py`** — Click-based CLI. Primary mode: `radiosim --config config.yaml`. Subcommands: `simulate`, `init`, `validate`.
- **`io/config.py`** — Pydantic v2 config classes. Top-level: `RadioSimConfig`. Loaded via `load_config()`.

### Core Physics (`core/`)

- **`visibility.py`** — Core RIME: `V_pq = Σ J_p @ C @ J_q^H` for point sources
- **`visibility_healpix.py`** — RIME for HEALPix diffuse maps (requires a populated `sky.healpix` payload)
- **`polarization.py`** — Stokes ↔ Coherency with 1/2 factor: `C = (1/2) × [[I+Q, U-iV], [U+iV, I-Q]]`
- **`antenna.py`** — Reads antenna formats (radiosim, casa, measurement_set, uvfits, mwa, pyuvdata) via a local `format_readers` dispatch dict in `read_antenna_positions()` (`io/antenna_readers.py` is a thin re-export shim)
- **`baseline.py`** — Generates antenna pair baselines
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

46 exported classes (see `core/jones/__init__.py`), across the term files plus the `beam/` subpackage. **Only K and E implement real physics; every other term currently returns the 2×2 identity** (see Implementation Status).

**Standard 8 terms** (composed in chain order K → Z → T → E → P → D → G → B; the chain applies them in reverse, sky-side first):

| Term | File | Class | Type |
|------|------|-------|------|
| K (Geometric Phase) | `geometric.py` | `GeometricPhaseJones` | DDE |
| Z (Ionosphere) | `ionosphere.py` | `IonosphereJones` | DDE |
| T (Troposphere) | `troposphere.py` | `TroposphereJones`, `TroposphericOpacityJones` | DDE |
| E (Primary Beam) | `beam/` | `BeamJones`, `AnalyticBeamJones`, `FITSBeamJones` | DDE |
| P (Parallactic Angle) | `parallactic.py` | `ParallacticAngleJones` | DIE |
| D (Pol. Leakage) | `polarization_leakage.py` | `PolarizationLeakageJones` | DIE |
| G (Gains) | `gain.py` | `GainJones`, `ElevationGainJones` | DIE |
| B (Bandpass) | `bandpass.py` | `BandpassJones` | DIE |

Beam internals (the E term — the most developed subsystem): `beam/analytic/` is a package implementing a composable analytic beam — aperture shapes (`circular`/Airy, `rectangular`/sinc, `elliptical`), illumination tapers (`uniform`, `gaussian`, `parabolic`, `parabolic_squared`, `cosine`), feed models, and reflector geometries; `compute_hpbw_numerical()` (in `beam/analytic/numerical_hpbw.py`) is a diagnostic-only HPBW finder. `beam/fits/` holds `BeamFITSHandler` (pyuvdata UVBeam, peak normalization) and `BeamManager` (shared/per-antenna FITS beams). `AnalyticBeamJones` supports `diameter_per_antenna` for heterogeneous arrays. The old `AntennaType` class and named beam types (`airy`, `cosine`, `exponential`, `short_dipole`) have been removed.

**Extended terms**: `faraday.py` (F), `wterm.py` (W), `receptor.py` (C/H), `element_beam.py` (Ee/a/dE), `delay.py` (Kd/Rc/ff), `crosshand.py` (X/Kx/DF), `baseline_errors.py` (M/Q).

**Architecture note**: `JonesBaselineTerm` in `baseline_errors.py` is a separate ABC for baseline-dependent effects (M, Q) that apply via Hadamard multiplication, NOT the matrix chain. They cannot be added to `JonesChain`.

To add a new Jones term: extend `JonesTerm` (or `JonesBaselineTerm`), implement `name`, `is_direction_dependent`, `compute_jones()`, export in `__init__.py`, add tests in `tests/unit/test_jones/`.

### Simulator Layer (`simulator/`)

- `base.py` — `VisibilitySimulator` ABC (Strategy pattern for swappable algorithms)
- `rime.py` — `RIMESimulator`: direct RIME summation, O(N_src × N_bl × N_freq)

### Backends (`backends/`)

- `base.py` — Abstract `ArrayBackend` interface
- `numpy_backend.py` — CPU (always available)
- `jax_backend.py` — JAX (GPU/TPU) `ArrayBackend` — implemented but not currently exercised by the RIME loops (see Implementation Status)
- `numba_backend.py` — Numba/Dask `ArrayBackend` — implemented but not currently exercised by the RIME loops
- Selection: `get_backend("auto" | "numpy" | "jax" | "numba")`, discovery: `list_backends()`

### I/O (`io/`)

- `config.py` — Pydantic v2 config models, top-level `RadioSimConfig`, loaded via `load_config()`
- `antenna_readers.py` — thin re-export shim for the readers in `core/antenna.py`
- `writers.py` / `readers.py` — HDF5/YAML simulation I/O
- `measurement_set.py` — CASA Measurement Set export (requires python-casacore)
- `fits_utils.py` — FITS file utilities

### Utilities (`utils/`)

- `validation.py` — Pre-flight config validator that collects all errors at once
- `logging.py` — Logging configuration

## The RIME Equation

`V_pq(ν, t) = Σ_s J_p(s, ν, t) · C_s(ν) · J_q^H(s, ν, t)`

**Critical**: The coherency matrix uses a 1/2 factor (`C = (1/2) × [[I+Q, U-iV], [U+iV, I-Q]]`) ensuring `V_XX + V_YY = I` (not 2I). Precision is fully controlled by PrecisionConfig — respect the user's chosen dtype everywhere, including coordinate conversions and phase calculations.

## Code Style

- **Formatter**: Black (88 char line length)
- **Linter**: Ruff (E, W, F, I, B, C4, UP rules; E501 ignored)
- **Type checker**: MyPy (check_untyped_defs=true, ignore_missing_imports=true)
- **Commits**: Conventional format (`feat:`, `fix:`, `refactor:`, `test:`, `chore:`). Never include co-authored-by lines.

### Git workflow

After completing and verifying any small, coherent task that changes repository
files, create a local commit automatically before handing the work back. Do not
wait for a separate request to commit. Keep each commit narrowly scoped and use
the conventional message format above. Never push, create a pull request, or
otherwise publish commits without first asking the user and receiving explicit
approval.

## Configuration

YAML config validated by Pydantic. See `configs/` for examples. Sections checked by `RadioSimConfig.validate()` (the pre-flight collector): `telescope`, `antenna_layout`, `beams` (`beam_mode`), `location`, `obs_time`, `obs_frequency`, `sky_model`, `visibility` (`sky_representation`), `output`. **`sky_model.flux_unit`** is required (Jy/mJy/uJy) — all flux limits in config use this unit. Config classes in `io/config.py`, top-level is `RadioSimConfig`.

TODO: Add an explicit contributor note that pre-`v1.0` API/config refactors should not preserve backward compatibility by default; prefer moving directly to the cleaner replacement unless a deprecation path is explicitly requested.
