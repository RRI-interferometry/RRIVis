# MeqTrees Cattery — Exhaustive Reference

This document is an exhaustive technical reference for the **`meqtrees-cattery`** package located at `simulators/meqtrees-cattery/`. It is intended to serve as a single, self-contained guide to every subsystem, module, key class, TDL option, file format, and algorithmic convention found in the codebase. It is purposely long and densely cross-referenced so that future RadioSim development can adopt or interoperate with parts of MeqTrees without re-reading the full source.

> **Author / upstream**: Oleg Smirnov (osmirnov@gmail.com).
> **Upstream URL**: https://github.com/ska-sa/meqtrees-cattery
> **Package version (`setup.py`)**: `meqtrees_cattery 1.8.0`, Python ≥ 3.0.
> **Folder snapshot scanned**: `/Users/kartikmandar/RadioSim/simulators/meqtrees-cattery/`
> **Top-level Python LOC** (excluding tests / blank): ~34 663 lines.

---

## Table of Contents

1. [What MeqTrees and the Cattery Are](#1-what-meqtrees-and-the-cattery-are)
2. [Repository Layout](#2-repository-layout)
3. [Build, Install, CI, and Runtime Stack](#3-build-install-ci-and-runtime-stack)
4. [TDL Concepts and the `Meow` Framework](#4-tdl-concepts-and-the-meow-framework)
   - 4.1 [The `Meow` package — what each module does](#41-the-meow-package--what-each-module-does)
   - 4.2 [Sky components, directions, and patches](#42-sky-components-directions-and-patches)
   - 4.3 [`IfrArray`, `IfrSet`, observations, and UVW handling](#43-ifrarray-ifrset-observations-and-uvw-handling)
   - 4.4 [`Parameterization`, `Parm`, `ParmGroup`, `SolverControl`](#44-parameterization-parm-parmgroup-solvercontrol)
   - 4.5 [Jones helpers (`Meow.Jones`)](#45-jones-helpers-meowjones)
   - 4.6 [`MeqMaker` and `TensorMeqMaker`](#46-meqmaker-and-tensormeqmaker)
   - 4.7 [`MSUtils` — Measurement Set integration](#47-msutils--measurement-set-integration)
   - 4.8 [`StdTrees`, `Bookmarks`, `Parallelization`](#48-stdtrees-bookmarks-parallelization)
   - 4.9 [`Meow.Utils`, `Meow.LSM`, `OptionTools`](#49-meowutils-meowlsm-optiontools)
5. [Siamese — Simulation Frameworks](#5-siamese--simulation-frameworks)
   - 5.1 [Top-level scripts (turbo-sim, example-sim, batch_sim)](#51-top-level-scripts)
   - 5.2 [OMS sky models](#52-oms-sky-models)
   - 5.3 [OMS ionosphere models](#53-oms-ionosphere-models)
   - 5.4 [OMS beam models (analytic, FITS, EMSS, PAF)](#54-oms-beam-models-analytic-fits-emss-paf)
   - 5.5 [OMS Jones modules (G, P/L, D, Ncorr, R, dipole)](#55-oms-jones-modules-g-pl-d-ncorr-r-dipole)
   - 5.6 [SBY — LOFAR / CS1 beams](#56-sby--lofar--cs1-beams)
   - 5.7 [AGW — Az/El sky and ionospheric RM](#57-agw--azel-sky-and-ionospheric-rm)
6. [Calico — Calibration Framework](#6-calico--calibration-framework)
   - 6.1 [Top-level scripts](#61-top-level-scripts)
   - 6.2 [`ParmTables` and `FunkOps` — solution table tooling](#62-parmtables-and-funkops--solution-table-tooling)
   - 6.3 [Solvable Jones modules (DI / DD / pointing / refraction / leakage / IFR)](#63-solvable-jones-modules)
   - 6.4 [The `Flagger` engine](#64-the-flagger-engine)
   - 6.5 [`StefCal` — fast iterative gain solver](#65-stefcal--fast-iterative-gain-solver)
7. [LSM — Local Sky Model](#7-lsm--local-sky-model)
8. [Lions — Ionospheric Modelling Framework](#8-lions--ionospheric-modelling-framework)
9. [Auxiliary Trees: Pyxides, Scripter, qt.py](#9-auxiliary-trees-pyxides-scripter-qtpy)
10. [Conventions, Math, and Constants](#10-conventions-math-and-constants)
11. [Cross-cutting File Formats](#11-cross-cutting-file-formats)
12. [Tests](#12-tests)
13. [Glossary](#13-glossary)
14. [Pointers to RadioSim Equivalents](#14-pointers-to-radiosim-equivalents)

---

## 1. What MeqTrees and the Cattery Are

**MeqTrees** is a software package for implementing the *Measurement Equation* (ME) of radio interferometry as an explicit, evaluable expression tree. The kernel (`Timba`, in the separate `meqtrees-timba` C++/Python package) executes a tree of nodes (`Meq.*`) over multidimensional `Cells` (time × frequency × …). Trees are built in the Python-embedded **Tree Definition Language (TDL)** and compiled into a `forest_state` that the kernel then executes.

The **Cattery** (this package) is a curated set of *frameworks* built on top of the Timba kernel:

| Sub-tree         | Purpose                                                                  |
| ---------------- | ------------------------------------------------------------------------ |
| `Meow`           | Object-oriented helpers for building MEs (sky comps, directions, Jones)  |
| `Siamese`        | Simulation TDL scripts and modules (sky models, beams, ionosphere, Jones)|
| `Calico`         | Calibration TDL scripts, parm tables, flagger, and the **StefCal** solver|
| `LSM`            | Local Sky Model: source catalogues, sixpacks, projection, persistence    |
| `Lions`          | Ionospheric phase modelling (MIM/TID/Kolmogorov/KL) as Z-Jones           |
| `Pyxides`        | Pyxis-compatible pipeline scripts                                        |
| `Scripter`       | Bash-driven calibration pipelines (legacy)                               |

The Cattery is the de-facto place where physics modelling and astronomer-facing scripts live; the kernel only knows about generic numerical operations.

**Top-level files**:
- `README.md` — short intro, points to the [MeqTrees wiki](https://github.com/ska-sa/meqtrees/wiki).
- `setup.py` — distutils package; declares `meqtrees_cattery==1.8.0`, depends on `numpy>=1.16`, `astropy>=3.0.0`, `python_casacore`, `scipy`, `astro_kittens`, `purr` (Timba is pulled separately).
- `pyproject.toml` — minimal.
- `Dockerfile` — full reproducible build on `kernsuite/base:9` with `casacore`, `python_casacore`, `meqtrees-timba`, `kittens`, `purr`, `tigger`, `owlcat`, then runs the Pyxis `meqtrees-batch-test` recipe via `pynose` as a smoke test. Also prepares a Python `venv` and pins `numpy==2.2.6`.
- `.travis/` and `Jenkinsfile.sh` — CI (Travis with Docker images for Py3 build, mypy, pep8; Jenkins cron job).
- `unittests/test_interpolatedbeam.py` — single regression test that exercises `InterpolatedBeams.LMVoltageBeam` against beams generated by the external `eidos` tool.

---

## 2. Repository Layout

```
meqtrees-cattery/
├── Cattery/                  ← top-level Python package
│   ├── __init__.py           (empty)
│   ├── qt.py                 PyQt3 import guard (raises RuntimeError)
│   ├── Meow/                 OO measurement-equation framework
│   ├── Siamese/              simulation scripts + OMS / SBY / AGW modules
│   ├── Calico/               calibration scripts + OMS + StefCal
│   ├── LSM/                  Local Sky Model
│   ├── Lions/                Ionosphere (Z-Jones) framework
│   ├── Pyxides/              symlink to Calico/OMS/StefCal/Pyxides/stefcal.py
│   └── Scripter/             bash-driven calibration pipelines
├── unittests/                pytest/nose tests
├── .github/workflows/python-publish.yml
├── .travis/                  CI dockerfiles
├── Dockerfile                full kern build
├── Jenkinsfile.sh            Jenkins pipeline
├── CHANGES.md                short release notes (1.4.0 → 1.5.0)
├── LICENSE                   GPL v2-or-later
├── README.md
├── pyproject.toml
├── requirements.txt
└── setup.py
```

The package is installed as **`Cattery`** (capital C). TDL scripts written by users typically `import Meow`, `from Cattery.Calico.OMS import StefCal`, etc.

---

## 3. Build, Install, CI, and Runtime Stack

The Cattery is *pure-Python*, but it is unusable without:

1. **Timba kernel** (`meqtrees-timba`) — the C++ tree evaluator (`Meq.*` nodes, `Meq.Cells`, `Meq.Request`, etc.). Built from CMake, installed with `make install`. The Dockerfile shows the canonical recipe.
2. **`python-casacore` / `pyrap.tables`** — for Measurement Set I/O. `Meow.MSUtils` imports `pyrap.tables` and aborts if missing.
3. **`astropy`** — used by LSM and ParmTables.
4. **`astro_kittens`** — utility library (logging, MSUtils helpers).
5. **`purr`** — observing-log integration (`Purr.Pipe`).
6. **`tigger` / `owlcat`** — sky-model viewer and MS utilities, useful but optional.
7. Optional native imager: `lwimager` (or AIPS++ glish-based imager).

`setup.py` walks `Cattery/` and adds every directory with `__init__.py` as a package; everything else (e.g. `*.tdl.conf`, `*.tdl.profiles`) is shipped as `data_files`.

The Dockerfile additionally installs `casacore-dev`, `casarest`, `wcslib`, `cfitsio`, `fftw3`, `makems`, builds Timba with `-DENABLE_PYTHON_3=ON`, and ends by running the `meqtrees-batch-test` recipe (a Pyxis recipe that simulates → calibrates → images a small WSRT MS).

CI matrix (`.travis/`):
- `py3.docker` — full functional run
- `mypy.docker` — static type checking
- `pep8.docker` — style check

---

## 4. TDL Concepts and the `Meow` Framework

### TDL recap

A TDL script defines two callables that the meqserver/browser invokes:

- `_define_forest(ns)` — runs at *compile time*, builds the node tree on the `NodeScope` `ns`.
- One or more `TDLJob` callables — runtime jobs (`def _tdl_job_simulate(mqs, parent, **kw):`).

Configuration happens via `TDLOption`, `TDLMenu`, `TDLCompileOption`, `TDLRuntimeOption`, etc., declared at module level. Options are persistent through `*.tdl.conf` files (one per script) and are exposed in the meqbrowser GUI.

### 4.1 The `Meow` package — what each module does

`Meow/__init__.py` sets `DiscoverMaximumW = False` and re-exports the canonical classes; it suppresses imports when run on the kernel side (i.e. when `Timba.meqkernel` is already imported), to avoid octopython re-init issues.

| Module                         | Role                                                         |
| ------------------------------ | ------------------------------------------------------------ |
| `Context.py`                   | Globals: `array`, `observation`, `vdm`, `mssel`, `correlations`, `unit_coherency` (1 per AIPS/TMS convention; 0.5 for legacy ME papers). Provides `set()`, `get_array()`, `get_observation()`, `get_dir0()`. |
| `Parameterization.py`          | Parameter management base class. Holds `_parmdefs` and `_parmnodes`, lazily creates `Meq.Parm`/`Meq.Constant` via `_parm()`. Exposes `QualScope`-based namespace: `self.ns`. |
| `Parm.py`                      | `Meow.Parm` — a *value* (constant or polc) plus options (`tiling`, `time_deg`, `freq_deg`, etc.) and tags. Used in lieu of raw `Meq.Parm` so components can decorate/redefine.|
| `ParmGroup.py`                 | Logical group of `Meq.Parm`s with subgroups, individual / subgroup / common solvability toggles, MEP-table override, polynomial degrees, subtilings, constraint clamps, "force positive" option. Includes `Subgroup` and `Controller` inner classes. |
| `SolverControl.py`             | Wraps `Meq.Solver` configuration: debug level, collinearity, LM factor, balanced-equations flag, ε, max-iter, convergence quota, multi-thread solve. |
| `Bookmarks.py`                 | Page / Folder construction inside `Settings.forest_state.bookmarks`. `make_node_folder()` recursively builds a hierarchy (default 2×2 plots, 25-entry submenu cap). |
| `OptionTools.py`               | `ListOptionParser` for "select-by-pattern" string options. |
| `Utils.py`                     | Legacy MS-options helpers (predates `MSUtils.MSSelector`), imaging options, `solver_options()`, `run_solve_job()`. Most modern scripts use `MSUtils` + `StdTrees.SolveTree` instead. |
| `Parallelization.py`           | `mpi_enable`, `mpi_nproc`, `parallelize_by_source` options + `smart_adder()` (hierarchical Meq.Add to keep cache pressure bounded) + `add_visibilities()` (parallel-aware sum). |
| `Position.py`                  | Generic 1–4D Position with optional `xpos`/`ypos`/`zpos`/`tpos` parm slots. |
| `Direction.py`                 | RA/Dec direction; PA, AzEl, LMN, KJones (`Meq.VisPhaseShift`), smearing factor (`Meq.TFSmearFactor`), `make_phase_shift()`. |
| `LMDirection.py`               | RA/Dec via `(l,m)` offset relative to `dir0`; uses `Meq.LMRaDec`. |
| `LMApproxDirection.py`         | Approximate `(l,m)` with explicit Δl/Δm relative to other LMDirections (faster phase-centre shifts). |
| `AzElDirection.py`             | Az/El direction tied to a station xyz position (uses `Meq.AzElRaDec`). |
| `SkyComponent.py`              | Abstract base for any sky element. Holds `direction`, `attrs`, `smearing`, `using_station_decomposition`, exposes `coherency()`, `visibilities()`, `corrupt()`, `get_solvables()`. |
| `PointSource.py`               | Stokes I/Q/U/V parms, optional `spi`/`freq0`/multi-term spi, optional `RM` (Faraday rotation in source frame), Cholesky `sqrt_coherency()` for station decomposition. |
| `GaussianSource.py`            | Adds Gaussian extent (`lproj`, `mproj`, `ratio`); coherency uses `Meq.PSVTensor` with `lmn=[0,0,1]`. |
| `Shapelet.py`                  | Reads `.mod` mode files (n0, β, then n0² coefficients), uses `Meq.ShapeletVisTf` + `Meq.Compounder` for FT-based shapelet visibilities. |
| `DiskSource.py`                | Disk source variant of GaussianSource. |
| `SixpackComponent.py`          | Abstract: 6-vector (RA, Dec, I, Q, U, V) → uses `Meq.FFTBrick` + `Meq.UVDetaper` + `Meq.UVInterpolWave` to produce coherency. Options for `fft_pad_factor` (RJN suggests ~8×M/N), `interpol_method` (1 bicubic / 2 4th-order / 3 bilinear), `interpol_debug`. |
| `FITSImageComponent.py`        | Concrete `SixpackComponent` reading from `Meq.FITSImage(filename, cutoff)`. |
| `Patch.py`                     | Aggregates several `SkyComponent`s, splits "solvable" vs "non-solvable" parts and adds them with `Parallelization.add_visibilities()` for cache reuse. |
| `KnownVisComponent.py`         | Wraps externally-provided visibility nodes. |
| `CorruptComponent.py`          | A `SkyComponent` plus a list of station-Jones (or DIE Jones) matrices; `apply_jones()` builds `J(p) … J(q)†` chains. |
| `IfrArray.py` + `IfrSet.py`    | Antenna array, baseline list, baseline-spec parser, UVWs (from MS spigots, computed via `Meq.UVW`, or read from MEP table). |
| `Observation.py`               | Phase centre direction, polarisation basis (linear vs circular), reference freqs/times. |
| `MSUtils.py`                   | MS selectors (`MSSelector`), DDID/field/channel/IFR/correlation/flagset selectors, BITFLAG handling, lwimager wrapper, output flag policies. |
| `MeqMaker.py`                  | Central ME-builder: registers sky models and Jones terms, builds predict and correction trees, exports Karma annotations. |
| `TensorMeqMaker.py`            | Tensor-mode subclass that uses `Meq.PSVTensor`/`Meq.CUDAPointSourceVisibility`/`Meq.ThrustPointSourceVisibility` for all-source-at-once predict (much faster for many sources). |
| `StdTrees.py`                  | `SolveTree`/`ResidualTree`, condeq + reqseq + solver assembly, `make_sinks()`, `vis_inspector()`, `jones_inspector()`, Jones-norm/abs flaggers. |
| `LSM.py`                       | `MeowLSM` — wraps `Cattery.LSM.LSM.LSM` to be a Meow sky model module (compile/runtime options, source-list factory, beam-weighted apparent-flux sort). |
| `ReadVisHeader.py`             | Helpers reading observation context from MS/spigot headers. |
| `make_dirty_image.py`          | Convenience wrapper around lwimager. |

### 4.2 Sky components, directions, and patches

Every sky element ultimately implements the API:

```python
component.coherency(array=None, observation=None, nodes=None) -> Node*(p,q)
component.visibilities(array, observation, smear=False) -> Node*(p,q)  # phase-shifts coherency from src to phase centre
component.sqrt_visibilities(...) -> Node*(p)                            # only for station-decomposable sources
component.corrupt(jones, per_station=True, label=...)                   # returns a CorruptComponent
component.is_polarized() / is_smeared() / is_station_decomposable()
component.set_attr(k, v) / get_attr(k, default)                         # bag for tags like 'cluster', 'Iapp', 'beam_lm'
component.get_solvables() -> list[Node]
```

The **PointSource** brightness uses the standard ME convention:

- **Linear basis** (`observation.circular()==False`):
  ```
  XX = I + Q
  XY = U + iV
  YX = U - iV
  YY = I - Q
  ```
- **Circular basis**:
  ```
  RR = I + V
  RL = Q + iU
  LR = Q - iU
  LL = I - V
  ```

Multiplicative factor `Context.unit_coherency` (default 1, AIPS/TMS convention) is applied via `Context.unitCoherency()`. Setting it to 0.5 reproduces the legacy MeqTrees ≤ 1.1.1 behaviour matching ME papers I–IV.

**Spectral index** (single or multi-term):
```
norm_spectrum = (ν / ν₀)^(spi + spi_2·log(ν/ν₀) + spi_3·log²(ν/ν₀) + …)
```
Implemented via `Meq.Pow` and `Meq.Log`.

**Faraday rotation** (`RM` parm) uses a Q/U rotation:
```
λ_ref² = (c / ν₀)²
α_ref  = -2 · RM · λ_ref²
Q_ref =  cos(α_ref)·Q - sin(α_ref)·U
U_ref =  sin(α_ref)·Q + cos(α_ref)·U
α(ν)  =  +2 · RM · (c/ν)²
Q(ν)  =  cos(α)·Q_ref - sin(α)·U_ref
U(ν)  =  sin(α)·Q_ref + cos(α)·U_ref
```
The factor of 2 reflects polarization-position-angle = ½·arctan(U/Q).

**Cholesky `sqrt_coherency`** (for station decomposition): for an *unpolarised* source it is just `√I`; for a polarised one it lower-triangularises the coherency matrix:
```
C = [[c11, c12], [c21, c22]]
norm = √(½ / (I+Q))
c11 = (I+Q)·norm
c12 = 0
c21 = (U - i V)·norm
c22 = √(I² - Q² - U² - V²)·norm
```
(swap Q↔V↔U accordingly for circular).

**GaussianSource** does *not* go through Cholesky; it uses `Meq.PSVTensor` with a 1×3 LMN tensor `[0,0,1]` and a 1×3 shape tensor `[lproj, mproj, ratio]`.

A **`Patch`** encloses many components with one direction, splits children into *solvable* and *non-solvable* groups so that the kernel can cache them separately during a solve, and uses `Parallelization.add_visibilities()` to combine them.

A **`CorruptComponent`** is a wrapper that adds a Jones list (per-station callable or DIE single Jones); on `coherency()` it materialises `J(p) · v(p,q) · J(q)†` chains via `apply_jones()`.

`KJones` (the per-station phase-shift Jones for direction shifting):
```python
Kj(p) = Meq.VisPhaseShift(lmn=lmn_minus1, uvw=uvw(p))
```
where `lmn_minus1 = [l, m, n-1]`. For `Direction is dir0`, `Kj` collapses to a constant 1.

### 4.3 `IfrArray`, `IfrSet`, observations, and UVW handling

`IfrArray` represents the array as a list of station-id pairs `[(ip, p), …]`. Helpful constructors:
- `IfrArray.WSRT(ns, stations=14)` — labels `0..9 A B C D E F`.
- `IfrArray.VLA(ns, stations=27)` — labels `1..27`.

Compile-time options (`Meow.IfrArray.compile_options()`):
- `uvw_source` ∈ {`"from MS"`, `"compute (VLA convention)"`, `"compute (WSRT convention)"`}.
- `uvw_refant` — explicit reference antenna for antenna-based UVWs.

If reading from MS, `IfrArray.uvw()` builds spigots; if `prefer_baseline_uvw=True`, it spawns a full *baseline* spigot grid (`uvw_ifr`) — robust against MSes with missing baselines (e.g. UVFITS-imports with AIPS-flagging excised). Reference station gets `[0,0,0]` constant; others are computed as `±spigot(refant, p)`.

If computing UVWs (`Meq.UVW(radec, xyz_0, xyz)`), the `mirror_uvw` flag flips sign for the VLA convention. An `_include_uvw_deriv=True` mode adds the velocity components (used for smearing).

`IfrSet` (the underlying immutable object) provides:
- IFR-spec parsing: `"all"`, `<=N`/`<N`/`>=N`/`>N` (baseline length), `P-Q`/`P:Q`/`PQ`, wildcards, `-` exclusion, `&` intersection, named aliases for WSRT (`FF`, `FM`, `MM`, `S83`, `S85`).
- `taql_string()` to inject the selection into a CASA TaQL query.
- `from_ms(ms)` factory: opens ANTENNA subtable, trims longest common prefix to compute compact station names.

`Observation` keeps the polarisation basis (`circular`/`linear`) and a phase-centre `Direction`.

### 4.4 `Parameterization`, `Parm`, `ParmGroup`, `SolverControl`

`Meow.Parameterization` is the cornerstone for parm-bearing objects (sources, gains, ionosphere). Key traits:

- A `Parameterization` owns a *qualified scope* `self.ns` (a `NodeScope.QualScope` keyed by `name + quals`), plus the global `self.ns0`.
- Parameters are *declared* via `_add_parm(name, value, tags=, solvable=True)`; nodes are created lazily by `_parm(name)` through `resolve_parameter()`. Values may be numeric constants (`Meq.Constant`), pre-existing nodes (used as-is), or `Meow.Parm` objects (turned into `Meq.Parm` with the supplied tiling/order/options).
- All solvable Parms get a `"solvable"` tag, allowing the solver to find them via `Node.search(tags="solvable")`.

`Meow.Parm` is a *deferred* parameter spec:
- `value` may be a scalar or a `meq.polc(…)` polc.
- `tags` is appended to.
- `tiling` is either an int (interpreted as time tiling) or a `dmi.record(time=…, freq=…)`.
- `time_deg`/`freq_deg` set polynomial degrees → `options['shape']=[t+1, f+1]`.
- Other kw args become options on `Meq.Parm` (`use_previous`, `node_groups='Parm'`, etc.).

`ParmGroup` (in `Meow/ParmGroup.py`) wraps an entire family of Parms with a UI and policy:
- Per-parm or per-subgroup solvability toggles.
- Override initial value, time/freq polynomial degrees, subtile sizes.
- MEP table override (`use_nondefault_meptable`) and "initialize from MEP table" (`use_mep`).
- Constraint sub-menu: "force positive", `constrain_min`, `constrain_max`.
- `TDLJob` to clear all funklets in the table.

`SolverControl` wraps an underlying `Meq.Solver` node:
| Option              | Default | Meaning                                            |
| ------------------- | ------- | -------------------------------------------------- |
| `debug_level`       | 0       | Solver verbosity                                    |
| `colin_factor`      | 0       | Collinearity-handling regularisation                |
| `lm_factor`         | 0.001   | Levenberg-Marquardt damping                         |
| `balanced_equations`| False   | Pretend equation count is balanced (faster; risky)  |
| `epsilon`           | 1e-5    | Convergence threshold (also used as `epsilon_deriv`)|
| `num_iter`          | 15      | Max iterations                                      |
| `convergence_quota` | 0.9     | Subtile convergence quorum                          |
| `mt_solve`          | True    | Multithreaded solve                                 |

`make_state_record(solvables, **kw)` produces a `command_by_list` record with one solvable section then a `state(solvable=False)` reset, which is exactly what `Meq.Solver` expects.

### 4.5 Jones helpers (`Meow.Jones`)

Pre-cooked Jones-matrix constructors that wrap `Parameterization.resolve_parameter`:

| Helper                                                  | Resulting matrix                                                         |
| ------------------------------------------------------- | ------------------------------------------------------------------------ |
| `gain_ap_matrix(jones, ampl, phase, …)`                 | `diag(g_x e^{iφ_x}, g_y e^{iφ_y})` — uses `'r'/'l'` qualifiers in circular pol, `'x'/'y'` in linear |
| `rotation_matrix(jones, rot, …)`                        | `[[cos α, -sin α], [sin α, cos α]]` (single angle)                        |
| `decoupled_rotation_matrix(jones, rot, …)`              | `[[cos α_x, -sin α_x], [sin α_y, cos α_y]]`                               |
| `ellipticity_matrix(jones, ell, …)`                     | `[[cos χ, i sin χ], [i sin χ, cos χ]]`                                    |
| `decoupled_ellipticity_matrix(jones, ell, …)`           | Independent χ_x / χ_y                                                     |
| `define_rotation_matrix(angle)`                         | Returns a node *definition* (must be assigned with `<<`)                  |
| `apply_corruption(vis, vis0, jones, ifrs)`              | `vis(p,q) = J(p) · vis0(p,q) · J(q)†` (chain over a list)                  |
| `apply_correction(vis, vis0, jones, ifrs)`              | `vis(p,q) = J(p)⁻¹ · vis0(p,q) · (J(q)⁻¹)†`                               |

Conjugate-transposes are memoised by `node('conj') ** Meq.ConjTranspose(node)`.

### 4.6 `MeqMaker` and `TensorMeqMaker`

`MeqMaker` (`Cattery/Meow/MeqMaker.py`, ~1430 lines) is the central orchestrator. It collects:
- a list of **sky-model modules** (one or more `module.source_list(ns)` factories);
- ordered lists of **sky-Jones terms** (Z, E, etc., direction-dependent);
- ordered lists of **uv-Jones terms** (P, G, B, D, IG, …);
- a list of **visibility-processing modules** (VPMs).

Every "module" is a Python object/module that exposes either:
- `module.compile_options()` / `module.runtime_options()` returning lists of TDL options (or implicit module-level options); and
- a Jones factory: `module.compute_jones(Jones, sources=…, stations=…, pointing_offsets=…, tags=…, label=…, meqmaker=…, inspectors=[])` that fills in `Jones(src,p)` and/or `Jones(p)` nodes; or
- a sky-model factory: `module.source_list(ns)` returning a list of `Meow.SkyComponent`.

Public API:

```python
mm = MeqMaker(namespace='me', solvable=False,
              use_correction=False,
              use_decomposition=None,         # offer "Use source coherency decomposition" toggle
              use_jones_inspectors=None,
              use_skyjones_visualizers=True)

mm.add_sky_models(modules, export_karma=False)
mm.add_sky_jones (label, name, modules, pointing=None, use_flagger=None)
mm.add_uv_jones  (label, name, modules, pointing=None, flaggable=False)
mm.add_vis_proc_module(label, name, modules)

mm.compile_options()
mm.runtime_options(nest=True)

mm.get_source_list(ns)
mm.estimate_image_size()
mm.make_predict_tree(ns, sources=None, uvdata=None, ifrs=None) -> visibility(p,q)
mm.corrupt_uv_data(ns, uvdata, ifrs=None, label='uvdata')
mm.correct_uv_data(ns, inputs, outputs=None, sky_correct=None, inspect_ifrs=None, flag_jones=True)
mm.apply_visibility_processing(ns, vis, ifrs=None)
mm.close()  # exports Karma annotations if enabled
```

Internally:
- `JonesTerm` / `SkyJonesTerm` records carry `(label, name, modules, base_node, solvable, flaggable, pointing_modules, subset_selector, base_pe_node, pe_initialized)`.
- A "module selector" is a `TDLMenu` toggled by `<group>_enable_<modname>` boolean attributes; selection is exclusive when there are multiple modules.
- "Advanced options" submenu per Jones term offers:
  - `<L>_advanced` toggle.
  - `<L>_all_stations` — share one Jones for all stations.
  - For sky-Jones: `<L>_per_source` ∈ {"each source", "entire model", or any tag name like `"cluster"`}.
  - `<L>_skip_correct` — exclude this term from the overall sky-Jones correction.
  - Subset selector (`SourceSubsetSelector`) with a rich syntax (see below).
  - Optional flagging menu when `flaggable=True`: matrix-norm vs `|J_ij|` vs `|J_ii|`, `freqmean`, upper/lower bounds.

Source subset selector grammar (`SourceSubsetSelector`):
```
all | *                        select all
name                           by name (fnmatch wildcards allowed)
=tag                           sources that have tag `tag`
tag<value | tag>=value | …     comparison on tag value (use d/m/s suffix for deg/arcmin/arcsec)
&token                         AND
-token                         remove
```

`make_predict_tree` workflow (when `use_decomposition` is True, station-decomposable sources are split into a separate fast path using `Meq.MatrixMultiply` on per-station `√V`):
1. `array.enable_uvw_derivatives(use_smearing)`.
2. For each sky-Jones term, build `J(src, p)` via `module.compute_jones()` (with optional pointing offsets `dlm` from a pointing module; pointing modules expose `compute_pointings`).
3. For each non-decomposable source, multiply `K(p) · J_n(p) · … · J_1(p) · C · J_1(q)† · … · J_n(q)† · K(q)†` (right-to-left: source coherency first; smear factor and `Kj` outermost in the source frame).
4. `Patch` solvable-corrupted vs uncorrupted sources separately.
5. Apply uv-Jones chain: `J(p) · vis(p,q) · J(q)†`.
6. Finally call VPMs: each module's `process_visibilities()` may transform the visibilities.

`correct_uv_data` builds `Jinv(p) = (J_n · … · J_1)⁻¹` per station, optionally with Jones-norm/abs flagging via `StdTrees.make_jones_norm_flagger` / `make_jones_abs_flagger`, then applies `out(p,q) = Jinv(p) · in(p,q) · Jinv(q)†`.

Visualisation: when `use_skyjones_visualizers=True`, MeqMaker injects a synthetic sky source on a chosen grid (`SKYJONES_LM`/`SKYJONES_RADEC`/`SKYJONES_AZEL`/`SKYJONES_AZEL_FULL`) and creates a `TDLJob` that requests an evaluation over that grid for all stations — the output is plotted via the `Result Plotter`. Time/freq spans, npix, and freq range are runtime options.

`export_karma_annotations(sources, filename, label_format)` writes `.ann` files with `CROSS`/`ELLIPSE` symbols and `TEXT` labels using a printf-like template (`%N`, `%I`, `%(I).3g`, `%Rd`, `%Rs`, etc.). Useful for kvis overlays.

`TensorMeqMaker` subclasses MeqMaker and replaces per-source visibility composition with a single `Meq.PSVTensor` (or `Meq.CUDAPointSourceVisibility` / `Meq.ThrustPointSourceVisibility`) per source group, dramatically reducing tree size for very many sources. Modules that wish to participate in tensor mode must implement `compute_jones_tensor(ns, srclist, stations, lmn=lmnT, pointing_offsets=dlm, inspectors=…)`, returning per-station tensor nodes; otherwise the framework falls back to assembling tensors from individual `compute_jones()` outputs.

Tensor-mode also adds smearing options: `fix_time_smearing`, `fix_freq_smearing`, `smearing_count`. The tensor PSV node is selected via the compile option `psv_class`.

### 4.7 `MSUtils` — Measurement Set integration

`Meow.MSUtils` (~1650 lines) is the canonical wrapper around `pyrap.tables`. It implements:

- `MSContentSelector` — DDID, field, channel start/end/step, additional TaQL string. When the MS is loaded, it populates DDIDs (from `DATA_DESCRIPTION` and `SPECTRAL_WINDOW`) and field names (from `FIELD`), and validates channel ranges interactively.
- `MSReadFlagSelector` / `MSWriteFlagSelector` — bitflag/legacy-flag input and output selectors using fnmatch patterns ("foo,bar", "*", "-bar"). Manages 31 bitflag bits and a "legacy" FLAG/FLAG_ROW pair. Output policy: `FLAG_ADD` / `FLAG_REPLACE` / `FLAG_REPLACE_ALL`.
- `MSSelector` — top-level selector with: MS path, IFR subset, polarisation basis, correlation subset (`1`, `1 corr→2x2 diag`, `2`, `2x2`, `2x2 diag`, `2x2 off-diag`), input/model/output column, tile size, max tiles, Hanning tapering (`HANNING_NONE` / `PRETAPERED` / `DOTAPER`), invert-phases, plus content+flag sub-menus. `setup_observation_context(ns, antennas, prefer_baseline_uvw=False)` builds and stashes a `Meow.IfrArray` and `Meow.Observation` from the MS.
- Flagging constants: `FLAGMASK_LEGACY=1`, `FLAGMASK_INPUT=2`, `FLAGMASK_OUTPUT=4`. The Meow-side trees emit *output* flags into `FLAGMASK_OUTPUT`.
- `MS_STOKES_ENUMS` and `LINEAR_CORRS` / `CIRCULAR_CORRS` to interpret CASA's `CORR_TYPE` column.
- Imager helpers: detects `lwimager`/`tigger`/`kvis`/`ds9`/`~/Tigger/tigger` at import time; `make_dirty_image()` builds and runs an imaging command-line.
- `STD_IFR_SUBSETS` provides default WSRT subsets (`"-45 -56 -67"`, `S83`, `S85`, `FM`, `FM -9A -9B`).

### 4.8 `StdTrees`, `Bookmarks`, `Parallelization`

`StdTrees`:
- `_BaseTree` provides `set_inputs()` defaulting to `array.spigots(flag_bit=1)`.
- `ResidualTree` makes `residual(p,q) = inputs(p,q) - predict(p,q)`.
- `SolveTree` builds:
  ```
  ce(p,q)   = Meq.Condeq(inputs(p,q), predict(p,q), modulo=…)        # weights optional
  solver    = Meq.Solver(children=[ce(p,q) for p,q ∈ solve_ifrs],
                         child_poll_order=…, flush_tables=True)
  reqseq(p,q) = Meq.ReqSeq(solver, outputs(p,q), result_index=1)
  ```
  Optimal `child_poll_order` is computed to maximise antenna-parallelism.
  `define_solve_job(jobname, jobid, solvables, tile_sizes, vdm)` registers a TDL menu with tile-size + solver options + a `TDLJob` that calls `Utils.run_solve_job` underneath.
- `define_inspector(nodeseries, qlists, …)` builds `Meq.Composer(plot_label=…)` with optional `Meq.Mean(reduction_axes=["freq"])` per element.
- `inspector(outnode, nodes)` / `vis_inspector(outnode, visnodes, ifrs)` / `jones_inspector(outnode, jones, array)` are convenience wrappers that also create a one-off bookmark page.
- `make_clipping_flagger(node, minval, maxval, flagmask=1)` uses `Meq.ZeroFlagger` and `Meq.MergeFlags`.
- `make_jones_abs_flagger` and `make_jones_norm_flagger` build per-element or matrix-norm clippers; `freqmean=True` collapses the freq axis first (entire timeslot flagged).
- `make_sinks(ns, outputs, post=…, vdm=ns.VisDataMux, spigots=True, output_col='DATA')` creates the `VisDataMux` with optimal poll order, attaches sinks (and the optional `Meq.ReqMux` post-step), and registers spigots as stepchildren.

`Bookmarks` provides `Page`, `Folder`, and `make_node_folder()` (recursive folder creation, max 25 entries per submenu, `2×3` plot grid by default).

`Parallelization`:
- `mpi_enable`, `mpi_nproc`, `parallelize_by_source` compile options.
- `smart_adder(nodes, visibilities, ifrs, step=8)` builds a hierarchical `Meq.Add` tree to keep cache pressure ~O(step).
- `add_visibilities()` distributes sources across MPI processes when enabled, then re-reduces with `mt_polling=True`.

### 4.9 `Meow.Utils`, `Meow.LSM`, `OptionTools`

`Meow.Utils` predates `MSUtils.MSSelector`; it provides legacy `include_ms_options()`, `ms_options()`, imaging options, generic `solver_options()`, and `run_solve_job(mqs, solvables, solver_node, vdm_node, tiling, options)` that synchronously executes a tiled solve. Most modern scripts ignore this in favour of `StdTrees.SolveTree.define_solve_job`.

`Meow.LSM.MeowLSM` is a *thin sky-model module* that:
- Reads via the LSM's own `LSM.queryLSM(count=99999999)` and a configurable beam expression for apparent-flux sorting (`cos(min(65*fq*r,1.0881))**6` is the WSRT default).
- Applies a subset string parsed by `Meow.OptionTools.ListOptionParser` (numbers or names, ranges `M:N`, with `-` exclusion).
- Builds either `PointSource` or `GaussianSource` (`size=[sx, sy], phi=-eP`).
- Promotes selected attributes to solvable Parms (`I`, `Q`, `U`, `V`, `spi`, `RM`, `pos`, `shape`).
- Optionally re-saves the LSM in native or text (HMS/DMS) format.
- Produced sources carry `Iapp` (apparent flux) and `beam_lm` attributes.

`OptionTools.ListOptionParser` parses comma/space lists with optional negations, range expansion, and validates numeric ranges using `(minval, maxval)` bounds.

---

## 5. Siamese — Simulation Frameworks

### 5.1 Top-level scripts

Path: `Cattery/Siamese/`. Three TDL entry points:

**`turbo-sim.py`** — flagship simulator. Uses **`TensorMeqMaker`** (tensor-mode predict). Wires the canonical Jones chain `E → iP → P → G → D → Z → L → Ncorr` (top-level menu controls each), supports `sim_mode ∈ {SIM_ONLY, ADD_MS, SUB_MS}`, optional `read_ms_model`, and three noise modes:
- `noise_stddev` — fixed Jy/visibility.
- "Compute from SEFD" sub-menu: `noise_sefd`, `noise_sefd_bw_khz`, `noise_sefd_integration` → `σ = SEFD / √(2 · BW · Δt)`.
- `random_seed = "time"` or any int.

The associated `turbo-sim.tdl.conf` and `Siamese/trut.tdl.conf` ship preset configurations.

**`example-sim.py`** — non-tensor variant with a simpler Jones chain (no `iP`, no `D`) and explicit noise generation via `Meq.Matrix22(real, imag, …)`. Best read first.

**`batch_sim_example.py` / `batch_sim_example.tdl.conf`** — headless driver. Boots a meqserver with `-mt 2`, loads the `.tdl.conf`, compiles `turbo-sim.py`, then sequentially:
- runs `_tdl_job_1_simulate_MS`;
- toggles `me.enable_G=False`;
- re-runs the sim;
- cleans up in `try/finally` via `meqserver.stop_default_mqs()`.

Common compile-time scaffolding shared by these scripts:
```python
mssel  = Meow.MSUtils.MSSelector(has_input=False, has_model=False,
                                 tile_sizes=[8,16,32], flags=False,
                                 hanning=True, invert_phases=True)
meqmaker = TensorMeqMaker.TensorMeqMaker(use_decomposition=False,
                                         use_jones_inspectors=True,
                                         use_skyjones_visualizers=False)
TDLCompileOptions(*mssel.compile_options())
TDLCompileOptions(*meqmaker.compile_options())
TDLCompileOption('run_purr', "Start Purr on the MS", False)
```

### 5.2 OMS sky models

`Cattery/Siamese/OMS/` ships the following **sky-model modules** (registered via `meqmaker.add_sky_models([...])`):

#### `gridded_sky.py`
Parametric grids/crosses/stars/lines of point or Gaussian sources. Compile options: `model_func` (cross/grid/circ_grid/star8/lbar/mbar), `grid_size`, `grid_step` (arcmin), `source_flux`, `center_source_flux` ("DefaultFlux" or override), `source_type` (point/gaussian), Gaussian sub-menu (smaj/smin arcsec, PA), polarisation sub-menu (Q/I, U/I, V/I), `source_spi`, `source_spi_2`, `source_freq0`. Each source dropped only if `l²+m² ≤ 1`. `estimate_image_size()` returns `grid_size · grid_step`.

#### `fitsimage_sky.py`
Wraps a single `Meow.FITSImageComponent` reading from a FITS image. Compile options: `image_filename`, `pad_factor` (default 1.2). Padding multiplies image size to attenuate FFT edge artefacts.

#### `transient_sky.py`
Time-variable sources with Gaussian temporal profile `I(t) = Ipeak · exp(-(t - tburst)² / (2 · duration²))`. Compile options: `tburst`, `duration`, `grid_size`, `grid_step`, `source_flux`. Uses `make_source(ns, name, l, m, tburst, duration, Ipeak)` factories.

#### `tigger_lsm.py`
Bridge to **Tigger** (the external sky-model GUI / CLI). `importTigger(verbose=0)` searches `~/Tigger`, `/usr/lib/meqtrees/Tigger`, and `$PATH` for `tigger` then dynamically imports `Tigger.SiameseInterface.TiggerSkyModel`. Wrapper `TiggerSkyModel(verbose=0, **kw)` exposes that as a Meow module.

### 5.3 OMS ionosphere models

#### `oms_ionosphere.py` / `oms_ionosphere2.py`
Two generations of TEC-based Z-Jones, with two TEC distributions:

- **Sine TID model** — superposition of two TIDs (one X, one Y). Time-varying amplitudes interpolated linearly between `tid_*_ampl_0` and `tid_*_ampl_1hr`:
  ```
  TEC(x, y, t) = TEC0
    + tid_x_ampl(t) · sin(2π · [x/(2·tid_x_size_km) + Δt · tid_x_rate/3600])
    + tid_y_ampl(t) · cos(2π · [y/(2·tid_y_size_km) + Δt · tid_y_rate/3600])
  TEC(src,p) = TEC(x,y,t) / cos(zenith_angle)
  ```

- **Wedge model**:
  ```
  wedge_dist(t) = (wedge_min + (wedge_max - wedge_min) · Δt / (wedge_time · 3600)) / 1e5
  TEC(x)        = (TEC0 + x · wedge_dist(t)) / cos(zenith_angle)
  ```

Phase conversion (`compute_zeta_jones_from_tecs`): `Z = exp(j · (-25 · c · TEC) / freq)` where `c = 3·10⁸`. The `-25` factor encodes the "MeqTrees TEC unit" choice (TEC in 10¹⁶ m⁻² — TECU). `oms_ionosphere2.py` is the same model but uses `iono_geometry2.py` (cartesian piercings) and adds inspectors for piercing points.

#### `iono_geometry.py` / `iono_geometry2.py`
Compute piercing points at fixed `H = 300_000 m`. `iono_geometry.py` projects antenna positions to the equatorial plane, optionally rotates with parallactic angle (`iono_rotate=True`), and computes `pxy(src,p) = ant_xy + H · (l, m) / √(1 - l² - m²)`. `iono_geometry2.py` uses azimuth/elevation directly: `dx = H · tan(za) · sin(az)`, `dy = H · tan(za) · cos(az)`. Both expose `compute_za_cosines`, `compute_zeta_jones_from_tecs`, and inspectors over `dxy`.

### 5.4 OMS beam models (analytic, FITS, EMSS, PAF)

#### `analytic_beams.py`
- `WSRT_cos³_beam.compute(E, lm, pointing=None, p=0)` — voltage gain
  ```
  E = cos³(bf · ν · r) / cos³(bf · ν · r_clip)
  ```
  with `bf=65 GHz⁻¹` default, ellipticity along E-W/N-S, "NEWSTAR-compatible clipping" toggle.
- `circular_aperture_beam.compute(...)` — Airy disk: `E = 2·J₁(θ)/θ`, `θ = (c/ν)·r·d·π·bf`. Per-antenna `dish_sizes` supported.

#### `fits_beams0.py`
Legacy `Meq.Resampler`/`Meq.Compounder`-based beam interpolator. Filename pattern: `beam_$(xy)_$(reim).fits`. Token substitutions: `%(xy)s`/`%(corr)s` → xx/xy/yx/yy; `%(reim)s` → re/im; `%(realimag)s` → real/imag. Options: `missing_is_null`, `norm_beams`, `sky_rotation`. Recommended replacement: `pybeams_fits.py`.

#### `pybeams_fits.py` (recommended FITS beam path)
Interpolates voltage beams from FITS via a PyNode (much faster than `fits_beams0`). Tokens: `$(xy)`, `$(XY)`, `$(corr)`, `$(CORR)`, `$(reim)`, `$(REIM)`, `$(ReIm)`, `$(realimag)`, `$(REALIMAG)`, `$(RealImag)`, `$(stype)`, `$(STYPE)`. Compile options:
| Option              | Default              | Notes                                            |
| ------------------- | -------------------- | ------------------------------------------------ |
| `filename_pattern`  | `beam_$(xy)_$(reim).fits` |                                            |
| `beam_type`         | `2x2`                | `2x2` / `diagonal` / `scalar`                    |
| `missing_is_null`   | True                 |                                                  |
| `spline_order`      | 3                    | scipy spline order 1–5                           |
| `normalize_gains`   | False                |                                                  |
| `ampl_interpolation`| False                | interpolate `|E|`, then re-attach phase           |
| `l_axis`/`m_axis`   | `L`/`M`              | Allowed: `L`, `X`, `TARGETX`, with `-` prefix    |
| `l_beam_offset`     | 0°                   | static offset                                    |
| `m_beam_offset`     | 0°                   |                                                  |
| `sky_rotation`      | True                 | apply parallactic-angle rotation                 |
| `verbose_level`     | None                 |                                                  |

Heterogeneous arrays are supported via JSON config:
```json
{
  "lband": {
    "patterns": {
      "cmd::default": ["$(stype)_$(corr)_$(reim).fits"],
      "ska":          ["ska_$(corr)_$(reim).fits"]
    },
    "define-stationtypes": {
      "cmd::default": "meerkat",
      "~ska[0-9]{3}": "ska"
    }
  }
}
```
The `~regex` prefix lets a station name match a regex; `cmd::default` is the fallback.

#### `InterpolatedBeams.py`
Core PyNode building blocks: `FITSAxes` parses FITS headers (CTYPE/CRPIX/CRVAL/CDELT/CUNIT and the non-standard `GR{type}{j}` for irregular grids). `LMVoltageBeam` reads paired real/imag FITS files, optionally combines multiple files (sum in voltage domain), and interpolates voltage on a 4-D `(l,m,freq,time)` grid via scipy splines. Also exports utility functions `expand_axis`, `unite_shapes`, `unite_multiple_shapes`.

#### `CompoundInterpolatedBeams.py`
`FITSCompoundBeamInterpolatorNode` — PyNode wrapping multiple `LMVoltageBeam`s, returning an N-element compound beam (e.g. 2N FITS files, N for X, N for Y). State: `filename_real`, `filename_imag`, `spline_order`, `normalize`, `ampl_interpolation`, `l_0`, `m_0`, `verbose`, `missing_is_null`.

#### `vla_beams.py`
Analytic VLA beam (Uson & Cotton 2008). Polynomial expansion in `u = (k·ν)² · (Δl² + Δm²)` with `k = 1.496e-9 · (25/d_ant)`:
```
c = [-0.56249985, 0.21093573, -0.03954289, 0.00443319, -0.00031761, 0.00001109]
E_x = 2·(0.5 + Σ_i c[i]·u_x^{i+1})
```
Beam squint applied as a frequency-dependent offset of the beam centre by `(vla_squint/3600°)·(c/ν)` along `[-sin(feed_angle), cos(feed_angle)]`.

#### `paf_beams.py`
Phased-array-feed beam model. Filename pattern includes `$(elem)`. Features: per-element weight files (`*.bw`), runtime element-gain-error simulation (uniform amplitude in dB and phase in degrees, sinusoidal time variation with random period), beam offsets read from text file, optional renormalisation per source, and "correct for first source" mode (selfcal-on-the-fly).

#### `AnalyticBeams/ClippedSincBeam.py`
PyNode `E = sin(d)/d` with `d = scale · ν · √(l² + m²)` and `scale ≈ 265.667/1.4e9`. Clamps `E=0` for `l<0` or `m<0` (single-quadrant beam — handy for testing).

#### `emss_beams/`
EMSS (Electro-Magnetic Simulation Software) voltage beams. Loaders/interpolators for both `.pat` text format and FITS:

- `EMSSVoltageBeam.py` — parses ASCII `.pat` files (`θ φ E_θ E_φ` per line, `frequency = X MHz` and `Gain = 20·log10(|E|) + X` headers); converts θ-φ to x-y via rotation+projection.
- `FITSVoltageBeam.py` — same model but reading FITS.
- `InterpolatedVoltageBeam.py` — EMSS-aware extension of `InterpolatedBeams.LMVoltageBeam`, adds **hierarchical** interpolation (lm first, then frequency) and θ-φ coordinate mode (`COORD_THETAPHI`).
- `compound_beams.py` / `emss_polar_beams.py` — top-level Meow modules. Compile options include `pattern_labels`, `freq_labels`, `beam_symmetry` (`SYM_X`/`SYM_Y` for 90°-rotated patterns), `normalization_factor`, `rotate_xy`, `interpol_coord`.
- `emss2fits.py` — utility to re-export EMSS data as FITS for use with `pybeams_fits.py`.

### 5.5 OMS Jones modules (G, P/L, D, Ncorr, R, dipole)

#### `oms_gain_models.py`
G-Jones `compute_jones(Jones, stations, …)` builds `diag(g_x · e^{j θ_x}, g_y · e^{j θ_y})`. Gain/phase generators come from `ErrorGens.Selector` (sub-options for fixed offset, random uniform, sinusoidal time variation, list of values). Defaults: gains in `[0.5, 1.5]`, phases in `0–60°`.

#### `oms_pointing_errors.py`
`compute_pointings(nodes, stations=…)` returns per-station `(dl, dm)` offsets. `station_subset` option lets a subset have errors; others get `(0,0)`.

#### `feed_angle.py` / `rotation.py`
Generate a single rotation matrix `R(ρ) = [[cos ρ, -sin ρ], [sin ρ, cos ρ]]` (or `e^{∓iρ}` in circular). `rotation.Rotation(label, pa=True, read_ms=True, feed_angle=True)` allows mixing parallactic angle (`PA_NONE`/`PA_NORMAL`/`PA_INVERTED`) with a fixed feed angle. The IAU convention: feed angle 0° ⇒ X points North, Y points West.

#### `leakage.py`
DI leakage `D = [[1, d], [-d, 1]]` with single solvable `d`.

#### `oms_dipole_projection.py`
Sky-Jones term L: per source/station, builds `[[cos az, -sin az·sin el], [sin az, cos az·sin el]]` to project sky E-field onto an NS/EW dipole pair.

#### `oms_n_inverse.py`
Sky-Jones term Ncorr (w-term correction): `J(src, p) = exp(j · 2π · w(p) · (n - 1) · ν / c)` with `n = √(1-l²-m²)`.

#### `position_shifts.py`
Sky-Jones term R, three modes:
- LM offset (`dl`, `dm` arcsec) — pure phase shift `exp(j·uvw·Δlm_rad)`.
- Differential refraction in elevation: `Δel = (el_src - el_centre) · del_rate`.
- Field rotation: rotate l,m by `θ` then evaluate phase-slope inspector.

#### `ErrorGens.py`
Plug-in error/perturbation framework. Base `ErrorGenerator`, subclasses:
- `NoError` — pass-through nominal value.
- `FixedOffset` — constant additive offset.
- `ListOfValues` — cycle through pre-supplied list per invocation.
- `RandomError` — uniform random in `[minval, maxval]` or `±maxerr`.
- `SineError` — sinusoidal time variation (random period, amplitude).
- `Selector` — meta TDL menu that constructs a `node_maker(node, station, axis)` factory at compile time.

#### `Utils.py`
- `substitute_pattern(filename_pattern, **subs)` — `$(key)` / `$key`, longest-first replacement, `$$ → $` escape.
- `json_beamconfig_reader` — parser for the heterogeneous-station JSON config; supports regex stations (`~pattern`) and multi-block chaining.

### 5.6 SBY — LOFAR / CS1 beams

`Cattery/Siamese/SBY/`:

- `lofar_beams.py` — full LOFAR beam model. Loads an external native library `lofar_beams_lib.so` via `Meq.PrivateFunction`. Supports `array_composition ∈ {DIPOLES, STATIONS, MIX}`, dipole models for **LBA** (droopy: length L, height h, slant α=45°, X/Y orientations 45°/135°, beam scale 88, all in metres/degrees) and **HBA** (bowtie: X/Y orientations 45°/135°, scale 600). Station configuration: external `.coords` files, station orientation `phi0`, station-id list.
- `sarod_cs1_beams.py` — deprecated CS1 (LOFAR core station 1) beam model superseded by `lofar_beams.py`.

### 5.7 AGW — Az/El sky and ionospheric RM

`Cattery/Siamese/AGW/`:

- `azel_sky.py` — single-source sky model fixed at `(az, el)` (RFI / ground-fixed sources). Uses `Meow.AzElDirection`. Compile options: `source_flux`, point/gaussian, polarisation, `az_pos`, `el_pos`, spectral index.
- `iono_angle.py` — Faraday-rotation Z'-Jones using an externally-provided RM file. Builds rotation matrix per-source per-station, with linear/circular handling.
- `PYRMAngle.py` — `PyGetRMAngle` PyNode that reads ALBUS / RMextract RM-vs-time files and emits the appropriate Faraday rotation angle.

---

## 6. Calico — Calibration Framework

### 6.1 Top-level scripts

`Cattery/Calico/`:

- **`calico-stefcal.py`** — flagship StefCal-driven calibration. Uses `TensorMeqMaker` with `solvable=True`. Hard-wires three gain hierarchies (DI gain `G`, bandpass `B`, differential gain `dE`) via `GainOpts` (see §6.5), assembles a composite data tensor `DT = Composer(spigots)` and as many predict tensors `MT:all`, `MT:src_k` as needed, and instantiates a single PyNode `Meq.PyNode(class_name="StefCalNode", module_name=Calico.OMS.StefCal.StefCal.__file__, …)`. Supports `do_output ∈ {CORR_DATA, CORR_DATA_SUB, CORR_RES}`.
- **`calico-generic.py`** — non-tensor flexible calibration script using `MeqMaker`. Exposes `cal_what` (visibility, amplitude, log-amplitude, phase) and `lhs/rhs ∈ {DATA, DIFF, …}`. Supports E/Es (WSRT cos³ fixed/solvable), dE, P, B, G, IG, IC. Includes Jones-norm flagging and residual flagging menus.
- **`calico-wsrt-old.py`** — legacy WSRT-specific script preserved for backwards compatibility.
- **`calico-flagger.py`** / **`Flagger.py`** / `calico-oldflagger.py` — interactive flagging trees (see §6.4).
- **`calico-view-ms.py`** — minimal MS visualisation / spigot-only inspector.
- **`calico-parmgroomer.py`** + **`ParmTables.py`** + **`FunkOps.py`** — solution-table groomer (averaging, linear interpolation, force-rank-0, infinite domain) with regex-based qualifier aggregation (e.g. `(r,i)→ToComplex`, `(ampl,phase)→Polar`, `(xx,xy,yx,yy)→Matrix22`).
- **`calico_model_iono_RM.py`** — calibration helper that reads an external ionospheric RM file (RMextract or ALBUS) and applies Faraday-rotation correction.
- **`OMS/`** — modular Jones / sky / IFR-error library (§6.3).
- **`OMS/StefCal/`** — the StefCal solver package (§6.5).
- **`OMS/StefCal/Pyxides/stefcal.py`** — Pyxis pipeline recipe driving StefCal end-to-end.

The standard `calico-stefcal.py` data-flow:
```
spigot(p,q) ──┐
              ├──► DT (Composer dim=[0])
              │
sky model + Jones chain (via TensorMeqMaker) ──► models = [MT:all, MT:src_1, MT:src_2, …]
                                                          │
                              StefCalNode(DT, *models, …) │
                              ├── solves G, B, dE        │
                              ├── outputs corrected /     │
                              │   subtracted / residual   │
                              ├── writes solutions to    │
                              │   gain.fmep / dE.fmep    │
                              └── optional ifrgains.ma   │
                                                          │
                       Selector + Composer(2x2) ──► output(p,q) ──► MS sink
```

### 6.2 `ParmTables` and `FunkOps` — solution table tooling

`ParmTables.py`:
- `ParmTab(filename, write=False, new=False)` wraps `FastParmTable`. `merge(filename)` brings funklets from another table.
- Internal cache file: `<parmtable>/ParmTab.cache` (pickle), reused if mtime ≥ funklets file mtime. Caches `_funklet_names`, `_domain_list`, `_axis_stats`, `_name_components`, `_domain_cell_index`/`_domain_reverse_index`.
- API: `funklet_names()`, `funklet_name_components()` (sets of unique components per `:`-position), `envelope_domain()`, `envelope_cells(num_time, num_freq, …)`, `subdomain_cells()`, `axis_stats(iaxis)`.
- `FunkSet` — funklets with the same name. `array(coeff=0, fill_value=0, masked=True, collapse=True)` materialises a coefficient grid (uses on-disk array cache `array.{name}.{coeff}.cache`).
- `FunkSlice` — subset along chosen axes. `apply(op_func, slicing, outtab=None, remove=False)` applies a reducer to each slice and writes the resulting funklets back (optionally deletes inputs).

`FunkOps.py` provides the canonical reducers callable from `ParmTab.apply`:
- `average(funkslice)` — mean over slice, envelope domain.
- `linear_interpol(funkslice)` — N input funklets → N-1 funklets with linear slopes between them.
- `force_rank0(funkslice)` — clamp polynomial rank to 0 (constant).
- `make_infinite_domain(funkslice)` — sets the funklet domain to `(-1e+99, 1e+99)` on every axis.

Funklet naming convention is `:`-delimited, e.g. `G:0:r`, `B:0:1`, `dE:source1:0`.

### 6.3 Solvable Jones modules

`Cattery/Calico/OMS/`:

#### `solvable_jones.py`
- `DiagAmplPhase` — diagonal Jones with amplitude / phase parameterised separately. Two ParmGroups (`{label}_phase`, `{label}_ampl`), two SolveJobs.
- `FullRealImag` — full 2×2 with real/imag Parms. 8 Parms per station (`xx_r`, `xx_i`, `xy_r`, …), one ParmGroup `{label}_diag`, one SolveJob, subgroups for X+X, Y+Y, real, imag, per-station.

#### `solvable_sky_jones.py`
Same module set but per source × station. `independent_solve` toggle keeps each source's Jones independent.

#### `solvable_pointing_errors.py`
`compute_pointings(nodes, stations, label='pnt')` creates `(dl, dm)` Parms per antenna, returns `Composer(dl, dm)`. ParmGroup `{label}`, SolveJob `cal_{label}`.

#### `solvable_position_shifts.py`
R-Jones (per source) using `Meq.VisPhaseShift(lmn=[dl, dm, 0], uvw)`. Solvable `dl, dm`.

#### `solvable_refraction.py`
Differential refraction (elevation-dependent field compression) plus optional differential extinction (`1/sin h`). Reference direction may be specified by source name, index, or explicit direction string.

#### `polarization_jones.py`
- `DecoupledLeakage` — `R(ρ) · E(χ)` (independent rotation and ellipticity).
- `CoupledLeakage` — coupled rotation+ellipticity decomposition.

#### `ifr_based_errors.py`
Per-baseline 2×2 multiplicative gains (`IfrGains`) and additive biases (`IfrBiases`). `process_visibilities()` and `correct_visibilities()` plug into MeqMaker as VPMs.

#### `gradient_mim.py`
A simple ionospheric-gradient Z-Jones. Two modes:
- `COMMON_GRAD` — global gradient `(α, β)` scaled per station by distance.
- `LOCAL_GRAD` — per-station gradients.

`Z = Polar(1, (α l + β m) · 1e9 / ν)` (i.e. radians at 1 GHz).

#### `wsrt_beams.py` / `wsrt_cos3_beam.py`
Primary-beam modules (cos³ form, optionally with solvable Zernike coefficients).

#### `central_point_source.py` / `model_3C343.py`
Tiny built-in sky models for tests.

### 6.4 The `Flagger` engine

`Calico/Flagger.py` (≈1100 lines):
- `Flagger(msname, verbose=0, chunksize=200000)` — opens an MS and reads in chunks of rows. Reads `BITFLAG`, `BITFLAG_ROW`, `FLAG`, `FLAG_ROW`. Knows whether the MS has bitflags (`has_bitflags`).
- `add_bitflags(wait=True, purr=True)` runs `addbitflagcol` on the MS.
- Bit constants: `BITMASK_ALL=0xFFFFFFFF`, `LEGACY=1<<33`.
- Selection layers (cumulative, in this order):
  - Subset A — whole rows (`ddid`, `fieldid`, `antennas`, `baselines`, `time`, `taql`).
  - Subset B — rowflag-based selection (`flagmask`/`flagmask_all`/`flagmask_none`).
  - Subset C — channel/correlation slicing (`channels`, `corrs`).
  - Subset D — visflag-based selection (per-cell flagmasks).
  - Subset E — data clipping (`clip_above`, `clip_below`, `clip_fm_above`, `clip_fm_below`, `clip_column`).
- `flag(flag=1, **kw)`, `unflag(unflag=-1, **kw)`, `transfer(flag=1, replace=False, **kw)`, `get_stats(flag=0, legacy=False, **kw)`, `xflag(...)` — orthogonal entry points with progress callbacks and Purr logging.
- `set_legacy_flags(flags)` / `clear_legacy_flags()` — sync `BITFLAG` with `FLAG`/`FLAG_ROW`.
- `AutoFlagger` wrapper — drives `aoflagger`/`flagdata`-style automated runs (`settimemed`, `setfreqmed`, `setnewtimemed`, `setsprej`, `setuvbin`, `setselect`, `run`, `save`, `load`).

`calico-flagger.py` builds in-tree flaggers: matrix-norm (`||J|| = √tr(J·J†)`), per-element absolute amplitudes, time-mean amplitudes, frequency-mean amplitudes. Uses `Meq.MatrixMultiply` + `Meq.ConjTranspose` + `Meq.Sqrt` + `Meq.Abs` to compute norms, then `Meq.ZeroFlagger` and `Meq.MergeFlags`.

`calico-oldflagger.py` keeps the older `abs_clip()`/`rms_clip()` API: `rms_clip` flags `|x| - mean(|x|) ≥ threshold·σ` per baseline (uses `Meq.StdDev`), with options `flag_xx_yy`, `flag_all_corrs`, `avg_freq`, `flag_absmax`, `flag_absmin`, `flag_rms`. Default flag_bit is 4.

### 6.5 `StefCal` — fast iterative gain solver

`Cattery/Calico/OMS/StefCal/` — implementation of the Stef(ano) calibration algorithm: a fast iterative least-squares solver for antenna-based 2×2 Jones gains.

Modules:
- `StefCal.py` — main entry point. Two PyNodes: `StefCalVisualizer` (plots gain solutions, optionally normalising off-diagonals by diagonals and flagging unity solutions) and `StefCalNode` (the solver itself). Holds gain "objects":
  ```
  self.gain   = GainOpts("","gain",  "G")     # DI gain
  self.bgain  = GainOpts("","gain1", "B")     # bandpass
  self.gainopts = [gain, bgain]               # active DI gains
  self.dgopts  = [GainOpts("","diffgain","dE") per dE source]
  ```
- `GainOpts.py` — TDLOption factory. Modes: `MODE_SOLVE_SAVE`, `MODE_SOLVE_NOSAVE`, `MODE_SOLVE_APPLY`. Median types: `TIMEMED`/`FREQMED`/`TOTMED`. Standard knobs:

  | Option                | Meaning                                                                   |
  | --------------------- | ------------------------------------------------------------------------- |
  | `enabled`             | Enable this gain term                                                     |
  | `use_float`           | Single-precision arithmetic                                               |
  | `timeint`/`freqint`   | Solution interval (0 = full domain)                                       |
  | `timesmooth`/`freqsmooth` | Post-solve smoothing kernel size                                       |
  | `flag_nonconv`        | Flag bins where the solver fails to converge                              |
  | `flag_chisq_threshold`/`flag_chisq_loop{0,1,2}` | χ² flagging threshold (`N · median`) per major loop |
  | `flag_ampl_low/high`  | Amplitude clipping thresholds                                             |
  | `implementation`      | `GainDiag`, `Gain2x2`, `Gain2x2a`, `GainDiagCommon`, `GainDiagPhase`      |
  | `mode`                | `solve-save` / `solve-nosave` / `apply`                                   |
  | `visualize`           | Wire visualizers                                                          |
  | `real_only`           | Restrict gains to real values                                             |
  | `nmajor_start`        | Start solving at major loop N                                              |
  | `weigh`               | Weight by per-baseline noise                                              |
  | `niter`/`epsilon`/`quota`/`delta`/`max_diverge`/`omega`/`average`/`ff` | Solver tuning      |
  | `table` / `intermediate_table` | FMEP table filenames                                              |

- `MatrixOps.py` — flat 4-vector representation `[m11, m12, m21, m22]`. Provides `matrix_multiply`, `_conj`, `_transpose`, `_add`, `_scale`, `_sub`, `_invert(reg=…)`, `_sqrt(via eigendecomposition)`, `_qrd`, `_eigenval`, `is_null`, `make_matrix`, `NULL_MATRIX`, plus `array_to_vells` and `mask_to_flags` helpers.
- `DataTiler.py` — generic tile manager. Given `datashape`, `subtiling`, computes `subshape`, `tiled_shape = [k0, m0, k1, m1, …]`, `tiling_slice` (tile expansion via `np.newaxis`), `subtiled_axes`, `total_slots`, `real_slots` (excluding padding).
  Methods: `tile_data(x)`, `untile_data(x)`, `tile_subshape(x)`, `reduce_tiles(x, method='sum')`, `expand_subshape(x, datashape, data_subset)`.
- `GainDiag.py` — diagonal (2-parameter per antenna) solver. Two-step iteration (G⁽⁰⁾ from previous estimate → solve gain0; gain0 → solve gain1) with the StefCal update
  ```
  for each (p, i):
    mh    = Σ_q,j  M_{pq,ij} · G_q
    Gnew  = Σ_q,j  D_{pq,ij} · mh* / |mh|²
  ```
  Convergence target: `round(real_slots · convergence_quota)` (default 99 %).
- `Gain2x2.py` — full 2×2 (4-parameter per antenna). Two-step iteration over antennas: for each `p`, accumulate `Σ_q D†·M` and `Σ_q M†·M`, then `G_p = Σ_q (M†·M)⁻¹ · Σ_q D†·M`, with optional averaging `G_new = ω · G_update + (1-ω)·G_old`.
- `GainDiagCommon.py` — diagonal gains with common phase / amplitude term.
- `GainDiagPhase.py` — phase-only diagonal gains.
- `Gain2x2a.py` — alternate 2×2 implementation (slightly different update ordering).
- `Pyxides/stefcal.py` — high-level Pyxis recipe wrapping the above.

`StefCalNode` core state (only the most relevant fields shown):

| Field                      | Meaning                                                                |
| -------------------------- | ---------------------------------------------------------------------- |
| `ifrs`, `solve_ifrs`       | Lists of `"p:q"` ifr labels                                            |
| `baselines`                | Baseline lengths (used for weighting/inspection)                       |
| `corr_names`               | `["x", "y"]`                                                           |
| `diffgain_labels`          | dE source labels                                                       |
| `num_major_loops`          | Major-loop count (default 10)                                          |
| `noise_per_chan`           | Per-channel noise estimate                                             |
| `critical_flag_threshold`  | Stop if flagged % > this                                               |
| `apply_ifr_gains`/`solve_ifr_gains`/`reset_ifr_gains`/`save_ifr_gains` | Multiplicative IFR gains (per-baseline 2×2 errors) |
| `ifr_gain_table`           | `ifrgains.ma` by default                                               |
| `per_chan_ifr_gains`/`diag_ifr_gains` | Per-channel and diagonal-only IFR-gain modes                |
| `residuals`/`subtract_dgsrc` | Output mode flags                                                    |
| `output_flag_bit`          | `MSUtils.FLAGMASK_OUTPUT`                                              |
| `verbose`                  | 0–3                                                                    |
| `use_polarizations_for_noise` | Use cross-hands for noise estimation                                |
| `chisq_rollback`           | Roll back if final χ² > initial                                        |
| `downsample_output`/`downsample_subtiling` | Output downsampling                                     |
| `regularization_factor`    | Matrix-inversion regularisation (default `1e-6`)                       |
| `rescale`                  | `no` / `scalar` / `per slot`                                           |

The major-loop algorithm (per call to `get_result`):
1. Read `DT` and `models` children. For each gain object, instantiate a tiler + solver.
2. **Minor loop**: compute residuals `R = DT - G · MT · G†`, solve normal equations via the chosen `Gain*` class, check `‖ΔG‖/‖G‖ < ε` and `|Δχ²|/χ² < δ`.
3. If `num_converged < convergence_target`, continue major loops.
4. Apply the converged gains to the output and return:
   - `CORR_DATA`: `G⁻¹ · V_in · G⁻†`
   - `CORR_RES`:  `G⁻¹ · (V_in - Model) · G⁻†`
   - `CORR_DATA_SUB`: `G⁻¹ · (V_in - dE_sources) · G⁻†`

dE sources are grouped by their `cluster` attribute (assigned by `tigger-convert`); ungrouped dE sources become singletons. Groups are sorted by descending apparent flux (`Iapp` → `I`). The resulting predict tensor list is `MT:all`, `MT:src1`, `MT:src2`, … fed to StefCal.

Solution storage:
- Each gain object writes funklets to a FMEP directory (`gain.fmep`, `bandpass.fmep`, `dE.fmep`, …). The funklet naming convention follows the Calico §6.2 rule (`G:0:0`, `dE:source1:0`, …).
- Subtiling shape is `[time_tiles, freq_tiles]`. One funklet per tile.
- Optional intermediate-values table (`intermediate-gain.fmep`).
- IFR-gains live in `ifrgains.ma`, optionally per-channel and diagonal-only.

Visualisation / Purr:
- `Meq.PyNode("StefCalVisualizer", label, freq_average, flag_unity, norm_offdiag, vells_label)` for gain diagnostics. Pages: `"StefCal G plotter"`, `"StefCal G inspector"`. Stores per-loop snapshots in `global_gains[label]`.
- `Purr.Pipe(mssel.msname).title("Calibrating").comment("Running StefCal…")` writes progress to the Purr observing log.

---

## 7. LSM — Local Sky Model

`Cattery/LSM/` is a self-contained sky-model library predating Tigger. It manages sources, aggregations of sources ("p-units" / patches), parameter tables, and IO with a number of catalogue formats. Total ~2700 lines.

Data structures:
- `Source` (`LSM_inner.py`) — name, treeType (`POINT_TYPE=0`, `PATCH_TYPE=1`, `GAUSS_TYPE=2`, `IMAGE_TYPE=3`), Gaussian extents `eX, eY, eP`.
- `SpH` (Sixpack Helper) — six vellsets `(sI, sQ, sU, sV, RA, Dec)` plus static fallbacks. Helpers: `getValue(type, freq_index, time_index)`, `getValueSize`, `updateValues(pname)`, `clone()`. Vellsets are `[time, freq, l, m]` for extended sources, `[time, freq]` for points.
- `PUnit` (`LSM.py`) — wraps one or more Source objects. Holds `cat`, `app_brightness`, `sp` (SpH), `_patch_name`, `__sixpack` (root Meq node), `_lm` (direction vector for the punit). Methods: `getRADec`, `getIQUV`, `getEssentialParms` → `(RA, Dec, I, Q, U, V, SI, f0, RM)`, `getExtParms`, `getLimits`, `change_location`, `clone()`.
- `LSM` — central container with `s_table` (name→Source), `p_table` (name→PUnit, sorted by descending brightness), `m_table` (default `'thislsm.mep'`), default patch centre method, current `NodeScope` and `mqs`/cells references. The `__barr` array caches PUnit names sorted by brightness for `queryLSM(count=N)`.

Public API (heavy-hitting):
```python
lsm = LSM()
lsm.add_source(Source, brightness=…, sixpack=…, ra=…, dec=…, lm=…)
lsm.queryLSM(all=1 | name=… | names=[…] | count=N | cat=C)
lsm.getSources(); lsm.getPUnits(); lsm.getPUnit(pname)
lsm.getBounds()                       # {min_RA, max_RA, min_Dec, max_Dec}
lsm.getMaxBrightness(type='A', f, t); lsm.getMinBrightness(); lsm.getBrightnessLims()

lsm.createPatch(slist, resolve_forest=True, sync_kernel=True)
   → [patch_name, x_min, y_min, x_max, y_max]
   uses Meq.PatchComposer; phase centre selected via 'G' (geometric) or 'C' (centroid-weighted)
lsm.createPatchesFromGrid(x_array, y_array, min_bright, max_bright, min_sources)

lsm.save(filename)                    # pickle (vellsets discarded; node tree serialised)
lsm.load(filename, ns=None)           # rebuilds the node forest
lsm.merge(filename, ns=None)          # union by source name

lsm.build_from_catalog(infile_name, ns)        # NEWSTAR/NVSS-style
lsm.build_from_orgsm(infile_name, ns)          # OR-GSM (decimal degrees)
lsm.build_from_ska(infile_name, ns)            # SKA model
lsm.build_from_newstar(infile_name, ns,
                       verbose=1, ignore_pol=False,
                       only_cleancomp=False, no_cleancomp=False)  # binary .MDL

lsm.linear_transform(A, b)                     # affine in (x,y) on all sources
lsm.move_punit(pname, new_ra, new_dec)
```

`Sixpack` (`LSM_Sixpack.py`) is the source-as-tree representation: optional Stokes IQUV node-stubs, RA/Dec, SI, RM, f0, plus `sixpack(ns)` / `iquv(ns)` / `radec(ns)` accessors that compose subtrees on demand. `decompose()` is the inverse. `newstar_source(ns, predefine=False, flux_att=1.0, slave=False, simul=False, **pp)` is the canonical constructor.

Coordinate utilities (`transform.py`, `common_utils.py`):

- `Projector(ra0, dec0, rot)` — SIN projection with rotation. Methods `sp_to_rt(ra, dec)`, `rt_to_sp(l, m)`, `give_limits(min_ra, max_ra, min_dec, max_dec)`, `On()`, `Off()`, `info()`. SIN projection formulas:
  ```
  L = cos(Dec)·sin(RA-RA₀)
  M = sin(Dec)·cos(Dec₀) - cos(Dec)·sin(Dec₀)·cos(RA-RA₀)
  l = L·cos P + M·sin P
  m = -L·sin P + M·cos P
  ```
  Inverse:
  ```
  Dec = asin(M·cos Dec₀ + sin Dec₀ · √(1 - L² - M²))
  RA  = RA₀ + atan2(L,  cos Dec₀ · √(1 - L² - M²) - M·sin Dec₀)
  ```
- `radec_to_lm` (NCP), `radec_to_lm_SIN`, `lm_to_radec` — alternative implementations used by the legacy NEWSTAR pipeline.
- `bin_search`, `radToRA`, `radToDec`, `stdForm` (SI suffixes k/M/G/T).
- Forest serialisation: `traverse(root, node_dict, subscope)` recursively walks the Meq tree, extracts name/classname/initrec/children, and `rec_parse(myrec)` flattens initrec values (polcs, records, `Timba.array`s, scalars). `create_node_stub`/`reconstruct(my_dict, ns)` re-instantiate the forest in a fresh `NodeScope`.

Catalogue formats consumed by `build_from_*`:
- **NEWSTAR/NVSS text**: 13 columns per line — catalog name, source name, RA hms, RA-error, Dec dms, Dec-error, frequency MHz, flux Jy, flux-error, equinox letter (`J`). `LSM.py` lines 1280–1306 hold the regex with named groups.
- **OR-GSM**: decimal degrees `ASSOC FLAG RA(deg) eRA DEC(deg) eDEC FLUX eFLUX`.
- **SKA model**: `SRC_ID RA(deg) DEC(deg) FLUX TYPE`.
- **NEWSTAR `.MDL`**: 512-byte header (4-char file type, header length, version, creation/revision dates and times, revision count, 80-byte node name) + binary model data.

The `LSM/test/` folder ships catalogue fragments (`3C343.txt`, `3C343_all.txt` 449 entries, NVSS / WENSS / NVSSC subsets) and `MG_LSM_test.py` (a TDL test that exercises the loaders).

`Meow.LSM.MeowLSM` (described in §4.1) is the "Meow side" of this system — it loads an LSM, applies a beam-weighted apparent-flux sort, parses a subset string, and emits a list of `Meow.PointSource`/`Meow.GaussianSource` for `MeqMaker.add_sky_models`.

Constants from `LSM/common_utils.py`:
```
POINT_TYPE = 0, PATCH_TYPE = 1, GAUSS_TYPE = 2, IMAGE_TYPE = 3
POINT_SOURCE_RTTI = 1001
PATCH_RECTANGLE_RTTI = 1002
PATCH_IMAGE_RTTI = 1003
PCOL_NAME=0, PCOL_TYPE=1, PCOL_SLIST=2, PCOL_CAT=3, PCOL_BRIGHT=4,
PCOL_FOV=5, PCOL_I=6, PCOL_Q=7, PCOL_U=8, PCOL_V=9, PCOL_RA=10, PCOL_DEC=11
```

---

## 8. Lions — Ionospheric Modelling Framework

`Cattery/Lions/` provides a family of physically motivated ionospheric Z-Jones modules, chosen at compile time via `ZJones.compile_options()`.

Architecture:
```
ZJones.py                          # Meow plug-in: dispatches to selected MIM
MIM_model.py                       # abstract base (TEC → phase chain)
PiercePoints/PiercePoints.py       # concrete base with pierce-point geometry
PiercePoints/modules/
   Poly_MIM.py                     # polynomial TEC(x,y) or TEC(lon,lat)
   TID_MIM.py                      # two travelling waves
   Kolmogorov_MIM.py               # advecting Kolmogorov phase screen
   VLSS_MIM.py                     # Kolmogorov pre-configured for VLSS night/day
   PhaseScreen.py                  # FFT-based screen generator
   KolmogorovNode.py               # PyNode interpolating screen at PP+v·t
   KL/KL_MIM.py / KL/KLNode.py     # Karhunen-Loève basis (for solving)
SimCa.py                           # MeqMaker integration (calibrate / simulate)
xyzComponent.py                    # ITRF position with ECEF-to-ENU rotation
PrintPyNode.py                     # PyNode logger
gridded_sky.py                     # alternate sky model exposed through SimCa
.tdl.conf                          # default TDL settings
```

`ZJones`:
- Top-level compile options: `make_log` (enable per-source/per-station log files), `ref_station` (rotate piercing points to the reference station's ENU frame), and an exclusive `mainmenu` selecting one of `TID_MIM`, `Kolmogorov_MIM`, `VLSS_MIM`, `Poly_MIM`, `KL_MIM`.
- `compute_jones(jones, sources, stations, …)` instantiates the chosen MIM (passing `Context.array.stations()`, optional reference rotation), then dispatches to the MIM's `compute_jones()` to build a 2×2 diagonal phase Jones.

`MIM_model`:
- Abstract `make_tec()` returns `ns['tec'](src, station)`.
- Default `make_phase_error()` is `phase = (75e8 / ν) · TEC` (the 75e8 prefactor matches the `-25 c` scaling used in `compute_zeta_jones_from_tecs`).
- Helpers: `make_freq()` (creates `Meq.Freq` if needed), `make_azel()` (per source-station `Meq.AzEl` + selectors).
- `compute_jones(Jones, …)` outputs `Jones(src,p) = [[exp(j φ), 0], [0, exp(j φ)]]`.

`PiercePoints`:
- Constructor accepts `height` (default 300 km, modelled as a solvable `Meow.Parm` so it can be fit), `make_log`.
- `make_pp(ref_station=None)` produces `ns['pp'](src,p) = Composer([x, y, z])` — the intersection of the source ray with a sphere at altitude `h` above earth radius `R_e`. The geometry uses `α' = asin(cos(el)·R_e/(R_e + h))` to compute the secant from elevation; the resulting PP is then optionally rotated into the ENU frame of `ref_station` via `make_rot_matrix(ref_station)`.
- `make_xy_pp` / `make_xyz_pp` / `make_longlat_pp` / `make_longlat_vector_pp` — coordinate variants used by individual MIMs.
- `create_log_nodes(xy=False)` injects a `Lions.PrintPyNode` writing `phases_<station>_<source>.dat`.
- `inspectors()` returns a list of MeqBrowser inspectors.

MIM modules:

#### `Poly_MIM.py`
Polynomial TEC parameterisation. Compile options: `N_long`, `N_lat` (each 1/2/3), `height`, `use_lonlat`. Implements
```
TEC(x, y) = Σ_{n_x, n_y} c_{n_x,n_y} · x^{n_x} · y^{n_y}
```
via Horner's method, with parameters `N:n_x:n_y` (Meow.Parm).

#### `TID_MIM.py`
Two superposed travelling waves:
```
TEC(x, y, t) = TEC0 + sec · {
   A1·TEC0·cos((2π/λ1)·[cos θ1·x - V1·t])
 + A1·TEC0·cos((2π/λ1)·[sin θ1·y - V1·t])
 + A2·TEC0·cos((2π/λ2)·[cos θ2·x - V2·t])
 + A2·TEC0·cos((2π/λ2)·[sin θ2·y - V2·t])
}
```
Compile options expose `TEC0`, `height`, `Wavelength_1/2`, `Speed_1/2`, `Theta_1/2`, `Amp_1/2`, `use_lonlat`. When `use_lonlat`, conversions: `λ_rad = λ_km/(R_earth+h)`, `V_rad/s = (V_km/h/3600)/(R_earth+h)`. `make_time()` uses `Meq.Time()`.

#### `Kolmogorov_MIM.py` + `PhaseScreen.py` + `KolmogorovNode.py`
Synthetic Kolmogorov phase screen (default `β = 5/3`, `N = 1024 px`). Compile options: `beta`, `N`, `speedx/speedy`, `scale`, `amp_scale`, `TEC0`, `height`, `seed_nr`, `use_longlat`. Screen generation in `PhaseScreen.init_phasescreen(N, beta, seed_nr)`:
```
Q ∝ |k|^{-(2+β)/2}
W = white_noise_complex_field(N×N)
S = W · Q
screen = ifft2(fftshift(S))
```
Normalised to `[-1, 1]`, then scaled by `TEC0 · amp_scale`. Wrapped in a PyNode (`KolmogorovNode`) that interpolates the screen at `pp + v·(t - t0)` with bilinear interpolation; advection wraps periodically.

#### `VLSS_MIM.py`
Two preset profiles for VLSS-era observations:
- **Night**: β=5/3, N=1024, speedx=400 km/h, speedy=50 km/h, scale=0.5 km, amp_scale=0.5, TEC0=5 TECU, h=400 km, seed=310573, use_lonlat=True.
- **Day**: same except amp_scale=1.0, h=300 km, seed=3418.

Earth radius hardcoded to `6365 km`.

#### `KL/KL_MIM.py` + `KL/KLNode.py`
Karhunen-Loève basis expansion (intended for solver use):
```
Phase(src, p) = Σ_{i=1..rank} c_i · φ_i(lon_pp, lat_pp)
```
where `φ_i` are eigenvectors of the autocorrelation matrix for a Kolmogorov-spectrum field and `c_i` are solvable parameters (`KLParm:i`, tiling `time=1`). `KL_MIM` composes all parameters into a single vector and all PPs into a `(N_src·N_stat) × 3` matrix and feeds them into `KLNode`, which evaluates the basis expansion. Compile options: `rank` (1/5/10), `height`.

`SimCa.py` is the MeqMaker integration script: registers `[lsm, fitsimage_sky, gridded_sky]` as sky models and `ZJones.ZJones()` as the Z-term, then exposes `run_option ∈ {simulate, calibrate}` and the appropriate solve jobs.

`xyzComponent` packages an ITRF position node and `make_rot_matrix()` (ECEF→ENU) plus `make_longlat(use_w=1)` (elliptical earth) / `use_w=0` (spherical).

---

## 9. Auxiliary Trees: Pyxides, Scripter, qt.py

- `Cattery/qt.py` — single-file safety net. If a TDL script tries `import qt` (the PyQt3 module), this file is found first, prints `"Caught Qt3 import"`, and raises a `RuntimeError` describing why. PyQt3 used to segfault when mixed with the (PyQt4-era) MeqTrees GUI.
- `Cattery/Pyxides/__init__.py` — empty. The directory contains a single `stefcal.py` which is a *symlink* to `Cattery/Calico/OMS/StefCal/Pyxides/stefcal.py`. Pyxis users `import Pyxides.stefcal` to drive end-to-end calibration.
- `Cattery/Scripter/` — bash-driven calibration pipelines (legacy):
  - `scripter.sh` — main runner.
  - `scripter.example.conf` — sample config: per-DDID channel ranges, IFR exclusions (`-5* -46 -67 -68`), default script (`calico-wsrt-tens.py`), default per-MS steps (`reset_ms cal_g cal_de`), TDL config file, output dir, `MT=--mt=7`, file-name pattern, merged-MS path, function-to-section map for cal steps, choice of external commands (`merge-ms.py`, `run-imager.sh`, `plot-ms.py`, `plot-parms.py`, `flag-ms.py`, `downweigh-redundant-baselines.py`).
  - `scripter.10calibration.funcs` — bash function library for the calibration steps.

---

## 10. Conventions, Math, and Constants

### Coordinates and units
- **RA, Dec**: radians, J2000.0.
- **`(l, m, n)`**: direction cosines via SIN projection, `n = √(1 - l² - m²)`. `Meow.Direction.radec_to_lmn` matches the IAU/CASA convention.
- **`(lon, lat)`**: geodetic (radians).
- **ECEF / ITRF**: Cartesian xyz, metres.
- **ENU**: tangent-plane east/north/up.
- **Az / El**: radians; Az measured clockwise from North.
- **Flux**: Jy (`1 Jy = 10⁻²⁶ W m⁻² Hz⁻¹`).
- **Frequency**: Hz (Hz internally; MHz in many compile options).
- **TEC**: TECU (`1 TECU = 10¹⁶ m⁻²`); ionosphere height: 300 km nominal, 200–400 km typical.
- **Phase**: radians, modulo 2π.

### Coherency conventions
`Context.unit_coherency` defaults to **1** (AIPS/TMS / MeqTrees ≥ 1.2). Set to `0.5` to match ME papers I–IV. The convention only affects the multiplicative factor between Stokes IQUV and the coherency matrix; downstream Jones calculations are unaffected.

Linear basis:
```
[ XX, XY, YX, YY ] = unitCoherency · [ I+Q, U+iV, U-iV, I-Q ]
```
Circular basis:
```
[ RR, RL, LR, LL ] = unitCoherency · [ I+V, Q+iU, Q-iU, I-V ]
```

### Smearing
`Direction.smear_factor(array, dir0)` returns `Meq.TFSmearFactor(Kjarg(p), Kjarg(q))` where `Kjarg(p) = uvw(p) · (lmn-1)`. Multiplied into the source coherency before phase-shifting.

### Faraday rotation
Source-frame RM uses a 2× factor: `α = 2 · RM · λ²`. See §4.2 for the full equations.

### Ionospheric phase
`Z = exp(j · -25 · c · TEC / ν)` (TEC in TECU, ν in Hz, c=3e8). Equivalent: `Z = exp(j · 75e8 / ν · TEC)` (used in `MIM_model.make_phase_error`).

### Solver poll order
`StdTrees.SolveTree` and `StdTrees.make_sinks` both compute an *optimal poll order*: pair condeqs/sinks so that disjoint antennas are polled first, maximising parallelism in the kernel. Falls back to the default order once half the stations have been seen.

---

## 11. Cross-cutting File Formats

| Format                 | Producer / consumer                                                                                                  |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Measurement Set**    | Read/written via `Meow.MSUtils` (pyrap/casacore). `BITFLAG` / `BITFLAG_ROW` columns optional, added by `addbitflagcol`. |
| **MEP (`.mep`) tables**| Older parameter storage (Meq-Parm-Polc). Used by ParmGroup with `nondefault_meptable` option.                          |
| **FMEP tables**        | "Fast" Meq-Parm-Polc tables — per-funklet directory. Used by StefCal: `gain.fmep`, `bandpass.fmep`, `dE.fmep`, `intermediate-gain.fmep`, etc. Funklet names follow `<term>:<qual1>:<qual2>:…`. |
| **`*.fmep/ParmTab.cache`** | Pickle cache produced by `Calico.ParmTables.ParmTab` (funklet names, domain list, axis stats, name components). |
| **Tigger LSM (`.lsm`)** | Native Tigger format, loaded via `Cattery.LSM.LSM.LSM.load` and `Meow.LSM.MeowLSM`.                                 |
| **NEWSTAR/NVSS text**   | 13-column ASCII catalogue (see §7).                                                                                |
| **OR-GSM text**         | Decimal-degree catalogue.                                                                                            |
| **SKA model text**      | Minimal `SRC_ID RA DEC FLUX TYPE`.                                                                                   |
| **NEWSTAR `.MDL`**      | Binary model file (512-byte header + payload).                                                                       |
| **Karma `.ann`**        | Output of `MeqMaker.export_karma_annotations` for kvis overlays.                                                     |
| **FITS images**         | `Meq.FITSImage` for sky cubes; `Meq.FITSReader`/`pybeams_fits` for beams. CTYPE-driven axis interpretation (incl. non-standard `GR{type}{j}` for irregular grids). |
| **EMSS `.pat` text**    | `θ φ E_θ E_φ`, with `frequency = X MHz` and `Gain = 20·log10(|E|) + X` headers.                                       |
| **`.bw`** weight files  | Initial PAF beam weights (`paf_beams.weight_filename_x/y`).                                                          |
| **`.iono` / RMextract / ALBUS** | Ionospheric RM tables consumed by `Siamese.AGW.PYRMAngle.PyGetRMAngle` and `calico_model_iono_RM`.        |
| **TDL conf (`*.tdl.conf`)** | Persistent TDL option values. Same syntax as INI.                                                                |
| **Karma `.ann`**        | `COORD W` / `PA STANDARD` header; `CROSS`, `ELLIPSE`, `TEXT` directives.                                              |

---

## 12. Tests

`unittests/test_interpolatedbeam.py` is the sole maintained unittest:
1. Generates a beam cube using the external `eidos` simulator (`eidos -d 0.5 -r 0.015625 -f 950 1050 20 -P /test/test_beam -o8`).
2. Loads `<prefix>_xy_re.fits` and `<prefix>_xy_im.fits` via `Cattery.Siamese.OMS.InterpolatedBeams.LMVoltageBeam`.
3. Interpolates at `(l, m) = (0.05°, 0.05°)`, freq=1 GHz.
4. Asserts the result equals a known-good complex value (`-0.00091718 + 0.00020009j`).

Functional tests live in the external Pyxis `meqtrees-batch-test` recipe (referenced in the Dockerfile), which runs WSRT-style simulate→calibrate→image pipelines.

---

## 13. Glossary

| Term            | Meaning                                                                                          |
| --------------- | ------------------------------------------------------------------------------------------------ |
| **TDL**         | Tree Definition Language — Python-embedded DSL for declaring MeqTree node forests.               |
| **`Meq.*`**     | Built-in node classes provided by Timba (`Meq.Constant`, `Meq.Parm`, `Meq.Solver`, …).           |
| **NodeScope `ns`** | Namespace for declaring nodes; `ns.foo(qual1, qual2)` creates qualified nodes.               |
| **Spigot**      | Source node reading from an MS column; used per-baseline.                                        |
| **Sink**        | Output node writing to an MS column.                                                              |
| **Sixpack**     | 6-vector `(I, Q, U, V, RA, Dec)` representation of a sky source.                                  |
| **PUnit**       | Patchwork unit — LSM's organisational unit (point or patch).                                      |
| **PSV / PSVTensor** | Point Source Visibility tensor node — vectorised predict for many sources at once.            |
| **Funklet**     | A single Meq.Parm value over one (sub)domain.                                                     |
| **Polc**        | Polynomial-coefficient funklet (`meq.polc`).                                                      |
| **Subtile / tiling** | Sub-domain partition for solving (e.g., one Parm per N timeslots × M channels).             |
| **Condeq**      | Condition equation node (`Meq.Condeq`) — input to the solver.                                     |
| **MIM**         | Minimum Ionospheric Model — phenomenological ionospheric parameterisation.                       |
| **TID**         | Travelling Ionospheric Disturbance.                                                              |
| **TECU**        | TEC Unit (10¹⁶ m⁻²).                                                                            |
| **DDID**        | Data Description ID in an MS (combines spectral window + polarisation setup).                    |
| **DI / DD**     | Direction-Independent / Direction-Dependent.                                                      |
| **DIE / DDE**   | Direction-Independent / Direction-Dependent Effect (Jones term).                                  |
| **VPM**         | Visibility-Processing Module (a MeqMaker plug-in that mutates visibilities).                      |
| **ParmGroup**   | Named collection of Meq.Parms with a UI (Meow.ParmGroup).                                         |
| **MEP/FMEP**    | (Fast) Measurement Equation Parameter table — funklet storage on disk.                            |
| **Purr**        | Observing-log/notes service used during execution; integrates with TDL.                          |
| **Tigger**      | External GUI for sky-model editing, used by `tigger_lsm.py`.                                     |
| **kvis**        | Karma image viewer; the Cattery exports `.ann` overlay files for it.                             |
| **StefCal**     | Stefano Salvini's iterative calibration algorithm.                                                |

---

## 14. Pointers to RadioSim Equivalents

For convenience while integrating concepts from MeqTrees into RadioSim, here is a cross-reference between Cattery concepts and their nearest RadioSim equivalents (paths relative to `src/radiosim/`):

| Cattery concept                                  | RadioSim analogue                                                       |
| ------------------------------------------------ | ---------------------------------------------------------------------- |
| `Meow.Context.array` / `IfrArray`                | `core/antenna.py`, `core/baseline.py`, `core/observation.py`           |
| `Meow.PointSource` / `GaussianSource` / RM       | `core/sky/model.py` (`SkyModel`), `core/sky/spectral.py` (PointSpectrum, RM) |
| `MeqMaker` Jones chain (E/G/Z/T/F/W/B/D/G/B)     | `core/jones/*.py` chain (K/Z/T/E/P/D/G/B), see `CLAUDE.md` for layout |
| `Meow.MSUtils` MS reader/writer                  | `io/measurement_set.py`, `io/readers.py`/`writers.py`                  |
| `Meow.LSM` / `Cattery.LSM.LSM`                   | `core/sky/loaders/`, `core/sky/_loaders_*.py`, `core/sky/operations.py`|
| `Meow.MSUtils.MSSelector` / `BITFLAG`            | (No direct equivalent yet — RadioSim writes to a fresh MS via `python-casacore`.) |
| `Cattery.Calico.OMS.StefCal`                     | (No equivalent — RadioSim is simulation-only at present.)                |
| `Lions` MIMs (Poly/TID/Kolmogorov/KL)            | `core/jones/ionosphere.py` (`IonosphereJones`)                        |
| `Siamese.OMS.pybeams_fits`                       | `core/jones/beam/fits.py` (`FITSBeamJones`, `BeamFITSHandler`)         |
| `Siamese.OMS.analytic_beams.WSRT_cos³_beam`      | `core/jones/beam/analytic.py` (`AnalyticBeamJones`, Gaussian only — see `CLAUDE.md`) |
| `Meow.OptionTools.ListOptionParser`              | (RadioSim uses Pydantic v2 + `io/config.py`.)                            |
| `Meq.PSVTensor` tensor-mode predict              | `simulator/rime.py` (`RIMESimulator`) using NumPy/JAX/Numba backends   |
| `purr` observing-log integration                 | `utils/logging.py` + standard `radiosim simulate` CLI                    |
| `Karma .ann` annotation export                   | (No equivalent — RadioSim uses `utils/diagnostics/*` for plots.)         |

The Cattery is fundamentally a *symbolic* ME builder: trees are constructed in TDL and evaluated lazily by Timba over multidimensional cells. RadioSim instead computes visibilities directly with numpy/jax/numba arrays. Many of the Jones models, beam models, ionosphere models, and sky-loaders described above are conceptually identical to the corresponding RadioSim classes; this document is intended to make those mappings explicit and to preserve the exact mathematical and structural details from the original codebase.

