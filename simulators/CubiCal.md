# CubiCal — Exhaustive Technical Reference

> A fast complex-Jacobian / Wirtinger-calculus radio-interferometric calibration suite.

This document is an exhaustive, source-grounded reference for the **CubiCal** package as vendored at
`simulators/CubiCal/` inside the RRIVis repository. All claims are cited to specific files in that
checkout. No external documentation has been consulted while authoring this reference.

---

## 1. Overview, Purpose, License, Version

| Field            | Value                                                                              | Source |
| ---------------- | ---------------------------------------------------------------------------------- | ------ |
| Project name     | CubiCal                                                                            | `simulators/CubiCal/cubical/__init__.py:1` |
| Tagline          | "a radio interferometric calibration suite"                                        | `simulators/CubiCal/cubical/__init__.py:1` |
| One-liner (PyPI) | "Fast calibration implementation exploiting complex optimisation."                 | `simulators/CubiCal/setup.py` |
| Version (vendored) | `1.6.5`                                                                          | `simulators/CubiCal/cubical/__init__.py` (`VERSION = "1.6.5"`) |
| Latest checked-in tag | `v1.6.4` (HEAD is on `5686a1d "Noxcal prepare release"` past v1.6.4)          | `git -C simulators/CubiCal tag` |
| License          | GNU GPL v2 (header) — `setup.py` classifier says "GNU General Public License v3 (GPLv3)" | `simulators/CubiCal/LICENSE.md`, `simulators/CubiCal/setup.py` |
| Copyright        | (c) 2017 Rhodes University & Jonathan S. Kenyon; (c) 2017 SKA South Africa         | `simulators/CubiCal/cubical/__init__.py`, `simulators/CubiCal/setup.py` |
| Primary author   | Jonathan Kenyon `<jonosken@gmail.com>`                                             | `simulators/CubiCal/setup.py` |
| Upstream URL     | https://github.com/ratt-ru/CubiCal                                                 | `simulators/CubiCal/setup.py`, `simulators/CubiCal/cubical/__init__.py` |
| Console entry    | `gocubical → cubical.main:main`                                                    | `simulators/CubiCal/setup.py` (`entry_points`) |
| Auxiliary scripts | `print-cubical-stats`, `plot-leakage-solutions`, `plot-gain-solutions`            | `simulators/CubiCal/setup.py` (`scripts=...`), `simulators/CubiCal/cubical/bin/` |
| Languages        | Pure Python (Python 3.6+, runtime hot-loops compiled with **Numba** `@jit`)        | `simulators/CubiCal/setup.py` (`python_requires=">=3.6"`); `simulators/CubiCal/cubical/kernels/full_complex.py:33` (`from numba import jit, prange`) |
| Build system     | `setuptools` declarative entry from `pyproject.toml` (no Cython, no compiled C)    | `simulators/CubiCal/pyproject.toml`, `simulators/CubiCal/setup.py` |

CubiCal is the production calibration tool used by the **MeerKAT / SKA-SA** community
for direction-independent (DIE) and direction-dependent (DDE) gain calibration of radio
interferometric measurement sets (CASA Measurement Set v2 format). It exploits the
**Wirtinger / complex-Jacobian** formulation of the Gauss–Newton update equation so that
the per-iteration linear algebra is performed directly on complex 2×2 Jones matrices,
without doubling the parameter dimension. Its "Cubi" naming alludes to the cubic
data tensor `(time, freq, baseline)` that each chunk represents.

The driver script `gocubical` (`cubical.main:main`) reads an INI-style **parset** of options
(`cubical/DefaultParset.cfg`) plus optional `--section-option value` command-line overrides,
opens a CASA Measurement Set via `python-casacore`, breaks the data into time/frequency
**tiles** and **chunks**, and farms the chunks out to worker processes that run a
Gauss–Newton / Levenberg–Marquardt loop on per-chunk **gain machines**. Solutions are written
to a "parmdb" pickled database, and optionally exported as CASA caltables. Output
visibility products (corrected data, residuals, model column) are written back to the MS.

Despite predating most modern dataflow frameworks, CubiCal pioneered three patterns that
later tools adopted:

1. **Chunked / tiled MS access** with shared-memory hand-off between an I/O process and a
   pool of compute workers (`cubical/workers.py`, `cubical/tools/shared_dict.py`,
   `cubical/tools/NpShared.py`).
2. **Pluggable "gain machines"** (`cubical/machines/`): each subclass of `MasterMachine`
   represents a particular Jones term parameterisation (full 2×2, diagonal, phase-only,
   delay/rate slopes, robust Student-t, ionospheric TEC, polarisation leakage, …) and is
   composed by `JonesChain` to form arbitrary Jones chains.
3. **Numba kernels** with explicit memory layout reordering (`allocate_reordered_array`)
   to keep the inner `JHJ`/`JHr` accumulations cache-local
   (`cubical/kernels/full_complex.py:42-60`).

---

## 2. Repository Layout

Tree at `simulators/CubiCal/` (Python packages only — vendored test MS subtree omitted):

```
CubiCal/
├── cubical/                                # main package
│   ├── __init__.py                         # exports VERSION = "1.6.5"
│   ├── main.py                             # gocubical entry point: 680 LOC driver
│   ├── solver.py                           # 960 LOC: solver loop + dispatch
│   ├── workers.py                          # 493 LOC: multiprocess pool + I/O worker
│   ├── flagging.py                         # 344 LOC: BITFLAG bit definitions, post-cal flagging
│   ├── statistics.py                       # 360 LOC: SolverStats container
│   ├── param_db.py                         # 81 LOC: thin parmdb open/create wrappers
│   ├── DefaultParset.cfg                   # 556 LOC: ground-truth option schema
│   ├── data_handler/                       # CASA MS reader, model predictor, beams
│   │   ├── ms_data_handler.py              # 1578 LOC MSDataHandler
│   │   ├── ms_tile.py                      # 1493 LOC MSTile / RowChunk
│   │   ├── MBTiggerSim.py                  # 284 LOC Montblanc bridge
│   │   ├── TiggerSourceProvider.py         # 294 LOC LSM (Tigger) source provider
│   │   └── wisdom.py                       # 96 LOC memory-budget estimator
│   ├── database/                           # solution table back-end
│   │   ├── iface_database.py               # 135 LOC abstract interface
│   │   ├── parameter.py                    # 796 LOC Parameter container w/ interpolation
│   │   ├── pickled_db.py                   # 303 LOC native CubiCal "parmdb"
│   │   └── casa_db_adaptor.py              # 490 LOC CASA caltable export adapter
│   ├── kernels/                            # Numba JIT inner loops
│   │   ├── __init__.py                     # allocate_reordered_array, import_kernel
│   │   ├── generics.py                     # 2x2 inverse, chi^2
│   │   ├── full_complex.py                 # full 2x2 complex gain kernel (residual/JHr/JHJ)
│   │   ├── full_W_complex.py               # weighted variant for robust solver
│   │   ├── diag_complex.py                 # diagonal-only optimisation
│   │   ├── diagdiag_complex.py             # diagonal data + diagonal gain
│   │   ├── diag_phase_only.py              # phase-only diagonal
│   │   ├── diag_robust.py                  # robust (Student-t) diagonal kernel
│   │   ├── phase_only.py                   # phase-only 2x2
│   │   ├── f_slope.py                      # frequency-slope (delay) parameterisation
│   │   ├── t_slope.py                      # time-slope (rate)
│   │   ├── tf_plane.py                     # joint delay-rate plane
│   │   ├── ff2_slope.py                    # delay+TEC composite
│   │   ├── chain.py                        # chain-rule helpers for JonesChain
│   │   ├── madmax.py                       # MAD/MMAD residual flagger
│   │   └── rebinning.py                    # on-the-fly time/freq binning
│   ├── machines/                           # Jones-term solution machines
│   │   ├── abstract_machine.py             # MasterMachine + Factory + provenance
│   │   ├── interval_gain_machine.py        # PerIntervalGains base
│   │   ├── complex_2x2_machine.py          # default 'complex-2x2'
│   │   ├── complex_W_2x2_machine.py        # 'robust-2x2' (Student-t)
│   │   ├── phase_diag_machine.py           # 'phase-diag'
│   │   ├── pol_gain_machine.py             # 'complex-pol' polarisation gains
│   │   ├── slope_machine.py                # f-slope / t-slope / tf-plane / TEC
│   │   ├── parameterised_machine.py        # base for slope/parametric machines
│   │   ├── jones_chain_machine.py          # chain of any of the above
│   │   ├── jones_chain_robust_machine.py   # robust chain variant
│   │   ├── parallactic_machine.py          # P-Jones (alt-az feed rotation)
│   │   ├── ifr_gain_machine.py             # baseline-based corrections (BBC)
│   │   └── machine_types.py                # type-string → class registry
│   ├── madmax/                             # "Mad Max" MAD-residual flagger
│   │   ├── flagger.py                      # 344 LOC wrapper around kernels/madmax.py
│   │   └── plots.py                        # 260 LOC diagnostic plots
│   ├── plots/                              # post-run summary plots
│   │   ├── __init__.py                     # make_summary_plots dispatcher
│   │   ├── gainsols.py                     # 659 LOC waterfall plotter
│   │   ├── ifrgains.py                     # 282 LOC BBC plot
│   │   ├── leakages.py                     # 183 LOC leakage diagnostics
│   │   └── stats.py                        # 106 LOC χ² / SNR stats
│   ├── degridder/                          # optional DDFacet predict
│   │   ├── DDFacetSim.py                   # 680 LOC dico-model degridder bridge
│   │   ├── DicoSourceProvider.py           # 229 LOC dico source iterator
│   │   ├── FITSBeamInterpolator.py         # 98 LOC FITS-beam wrapper
│   │   └── geometry.py                     # 576 LOC facet/coord geometry helpers
│   ├── stimela/                            # Stimela cab schema generator
│   │   └── generate_schema.py              # 70 LOC parset → stimela schema
│   ├── tools/                              # generic utilities
│   │   ├── parsets.py                      # parset reader (with type/options/metavar)
│   │   ├── dynoptparse.py                  # parset → optparse adapter
│   │   ├── logger.py                       # multiprocess-safe logging
│   │   ├── ModColor.py                     # ANSI colour helper
│   │   ├── ClassPrint.py                   # pretty option dump
│   │   ├── shared_dict.py                  # /dev/shm dict for IPC
│   │   ├── NpShared.py                     # numpy-on-/dev/shm
│   │   ├── shm_utils.py                    # SHM cleanup
│   │   └── dtype_checks.py                 # complex64/128 helpers
│   └── bin/                                # console scripts (installed via setup.py)
│       ├── gocubical                       # → cubical.main:main
│       ├── plot-gain-solutions
│       ├── plot-leakage-solutions
│       └── print-cubical-stats
├── docs/                                   # ReadTheDocs Sphinx source
│   ├── index.rst, introduction.rst, installation.rst, usage.rst,
│   ├── parset.rst, performance.rst, examples.rst, licence.rst, cubical.rst
│   └── conf.py, Makefile, _static/
├── examples/                               # Jupyter notebooks
│   ├── Reading parameter tables and interpolation.ipynb
│   └── Waterfall plots prototype.ipynb
├── test/                                   # integration test driver (real MS)
│   ├── d147_test.py                        # gocubical-on-D147 regression script
│   ├── d147-test.parset                    # baseline parset for the test
│   ├── 3C147-dE-apparent.lsm.html          # Tigger LSM for dE test case
│   ├── SUBSET-D147.MS/                     # vendored CASA MS (ANTENNA, FIELD, …)
│   └── SUBSET-D147-output.MS.tgz           # tarball of expected output
├── HEADER                                  # GPL header inserted into source files
├── README.md                               # 15-line stub pointing at ReadTheDocs
├── LICENSE.md                              # full GPL v2 text
├── MANIFEST.in                             # source dist manifest
├── setup.py                                # install entry, deps, extras, scripts
├── pyproject.toml                          # PEP-517: setuptools + wheel + six + numpy
├── rtd_requirements.txt                    # ReadTheDocs build deps
├── Jenkinsfile.sh                          # CI launcher
└── .jenkins/                               # Jenkins pipeline files
```

### File-count summary (Python only)

| Subpackage           | Files | LOC (sum of `wc -l`) |
| -------------------- | ----- | -------------------- |
| Top-level (`main.py`, `solver.py`, `workers.py`, `flagging.py`, `statistics.py`, `param_db.py`, `__init__.py`) | 7 | 3 481 |
| `data_handler/`      | 5     | 3 715 |
| `machines/`          | 13    | 5 661 |
| `kernels/`           | 16    | ~3 800 |
| `database/`          | 4     | 1 724 |
| `madmax/`            | 3     | 604 |
| `plots/`             | 5     | 1 270 |
| `degridder/`         | 5     | 1 583 |
| `stimela/`           | 2     | 103 |
| `tools/`             | 9     | 1 759 |

Plus `DefaultParset.cfg` (556 LOC of option schema). Total ≈ 24 k LOC of Python.

---

## 3. Installation, System Dependencies, Build System

### 3.1 Build system

CubiCal is a **pure-Python `setuptools` project**. There is no Cython, no C/C++ extension,
no separate build step.

`simulators/CubiCal/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools", "wheel", "six", "numpy"]
```

Notice that `setup.py` is **legacy-imperative**: it imports `cubical` itself to read
`cubical.VERSION`, and re-checks for `six` and `numpy` at install time
(`setup.py` lines around `import cubical` and `try: import six / numpy`). This is why
both packages are listed in the PEP-517 build requires. There is **no `[project]`
table** in `pyproject.toml`; the metadata is owned by `setup.py`.

### 3.2 Runtime install requirements

From `simulators/CubiCal/setup.py` (the `requirements = [...]` block, used when
`READTHEDOCS` is **not** set):

| Package              | Reason                                                                  |
| -------------------- | ----------------------------------------------------------------------- |
| `future`             | Py2/3 compatibility shim (legacy)                                       |
| `numpy`              | Core array library                                                      |
| `numba`              | JIT compiler for the hot kernels in `cubical/kernels/`                   |
| `python-casacore`    | CASA Measurement Set, table, image access                                |
| `sharedarray >= 3.2.1` | Shared-memory NumPy arrays via `/dev/shm` (used by `tools/NpShared.py`) |
| `matplotlib`         | Summary, gain, BBC, leakage, MadMax plots                                |
| `scipy`              | Linear algebra / interpolation                                           |
| `astro-tigger-lsm`   | Read Tigger sky models (`*.lsm.html`)                                   |
| `six`                | Py2/3 compatibility helpers (`add_metaclass`, `string_types`, …)         |
| `astropy>=3.0`       | FITS, time, units                                                        |
| `psutil`             | Memory/CPU introspection                                                 |

`python_requires=">=3.6"` is set in `setup.py`, but `simulators/CubiCal/cubical/main.py`
still uses `from __future__ import print_function` and `from six import string_types`,
hinting at the legacy Py2 support that has now been dropped.

### 3.3 Optional `extras_require`

Defined in `setup.py`:

| Extra              | Pulls in                                                               | Purpose                                                            |
| ------------------ | ---------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `lsm-support`      | `montblanc >= 0.6.4`                                                   | GPU/CPU model-visibility prediction from a Tigger LSM via Montblanc |
| `degridder-support`| `ddfacet >= 0.6.1`, `regions < 0.5`, `meqtrees-cattery >= 1.7.7`       | Predict from a DDFacet **DicoModel** (faceted image model)         |

`regions < 0.5` is pinned because of "bug in new DS9 parser" (cf. `setup.py` comment).

### 3.4 System packages

Per `simulators/CubiCal/docs/installation.rst`, on Ubuntu 20.04 the following are
required (provided by the **KERN-8** PPA):

```
casacore-dev, casacore-data, build-essential, python3-pip,
libboost-all-dev, wcslib-dev, git, libcfitsio-dev
```

Notes from `installation.rst`:

* The doc warns about a **casacore data-corruption bug on large reads** and
  recommends building casacore from source for those workloads.
* Recommended install: `pip3 install git+https://github.com/ratt-ru/CubiCal.git@1.4.0`
  (with `cubical[lsm-support]` for Montblanc support).
* Editable installs supported via `pip3 install -e path/to/repo/`.

### 3.5 Console scripts

Two layers of CLI hand-off:

1. **PEP-517 entry point** in `setup.py`:
   ```
   entry_points={'console_scripts': ['gocubical = cubical.main:main']}
   ```
2. **Plain-text scripts** registered with `scripts=[...]` in `setup.py`:
   * `cubical/bin/gocubical` — duplicate of the entry point, runs `cubical.main.main()`.
   * `cubical/bin/print-cubical-stats` — reads a `*.stats.pickle` from a previous run
     and prints summary tables.
   * `cubical/bin/plot-gain-solutions` — opens a parmdb and produces gain waterfall
     plots (delegates to `cubical.plots.gainsols`).
   * `cubical/bin/plot-leakage-solutions` — leakage-specific plots
     (`cubical.plots.leakages`).

### 3.6 Docker / Jenkins

`simulators/CubiCal/Jenkinsfile.sh` exists at the repo root and the `.jenkins/`
directory contains the multi-stage pipeline. There is **no Dockerfile** in this
checkout; CubiCal is exercised inside the upstream RATT containerized CI rather than
shipping its own image.

### 3.7 ReadTheDocs build

The `READTHEDOCS` environment variable triggers a slimmed-down requirement list in
`setup.py` (`numpy`, `matplotlib`, `scipy` only) so the docs build does not need
casacore. `simulators/CubiCal/rtd_requirements.txt` provides the additional Sphinx
deps. See `simulators/CubiCal/docs/conf.py` for the Sphinx config.

---

## 4. End-to-End Architecture

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ CLI / DRIVER                                                                  │
│   gocubical (cubical.main:main)                                               │
│     - Parses parset (cubical/DefaultParset.cfg + user parset + CLI overrides) │
│     - Builds GD: dict-of-dict of every option                                 │
│     - Initialises logger, matplotlib backend, SHM cleanup                     │
│     - Picks SOLVER class from solver.SOLVERS based on --out-mode              │
│     - Picks gain machine class from machine_types.GAIN_MACHINE_TYPES          │
│     - Constructs MSDataHandler (open MS, init flags, init models)             │
│     - Builds gm_factory (or JonesChain factory) and ifrgain_machine           │
│     - Calls ms.define_chunk(...) → list of MSTile, each with RowChunks        │
│     - workers.setup_parallelism(...) → decides ncpu/nworker/nthread/affinity  │
│     - workers.run_process_loop(...) → executes the full MS                    │
│     - On completion: SolverStats.save, post-mortem flagging,                  │
│       plots.make_summary_plots, plots.ifrgains.make_ifrgain_plots             │
└────────────────────────────────────┬──────────────────────────────────────────┘
                                     │
            ┌────────────────────────┴───────────────────────────┐
            │                                                    │
┌───────────▼──────────────┐                       ┌─────────────▼────────────┐
│ I/O WORKER (cf. PoolEx)  │                       │ COMPUTE WORKERS (N+1)    │
│   _io_handler()          │                       │   solver.run_solver()    │
│     - tile.load(load_model)                      │     - For each chunk in │
│     - tile.save(only_save=...)                   │       Tile, GN/LM solve │
│     - solver.gm_factory.save_solutions           │     - Build VisDataMgr  │
│     - solver.ifrgain_machine.accumulate          │     - GainMachine ←     │
│     - tile.release(final=...)                    │       gm_factory        │
│   Holds the casacore table handle                │     - SOLVERS[mode].run │
│   /dev/shm hand-off via SharedArray              │     - corr_vis written  │
└──────────────────────────┘                       │       back to tile      │
                                                   └──────────┬──────────────┘
                                                              │
              ┌───────────────────────────────────────────────┴──────────────────┐
              │ DATA / MODEL LAYER (cubical/data_handler/)                       │
              │   MSDataHandler (ms_data_handler.py)                             │
              │     ms (casacore.tables.table)                                   │
              │     ANTENNA / FIELD / SPECTRAL_WINDOW / POLARIZATION /           │
              │     DATA_DESCRIPTION / OBSERVATION / FEED subtable readers      │
              │     Per-tile RowChunk lists, define_chunk(), define_flags()      │
              │     init_models([…]) — recognises columns, LSMs (@dE tag),      │
              │       DDFacet DicoModels, Montblanc predictors                   │
              │   MSTile / RowChunk (ms_tile.py)                                 │
              │     load(): cube up rows × chans → (m,t,f,a,a,c,c)               │
              │     save(): writes corrected/model/weight/bitflag back           │
              │   MBTiggerSim / TiggerSourceProvider — Montblanc bridge         │
              │   degridder/DDFacetSim — DDFacet bridge                          │
              └───────────────────────┬──────────────────────────────────────────┘
                                      │
              ┌───────────────────────▼──────────────────────────────────────────┐
              │ SOLVER LAYER (cubical/solver.py)                                 │
              │   SOLVERS = { 'so': SolveOnly,                                   │
              │               'sc': SolveAndCorrect,                             │
              │               'sr': SolveAndCorrectResiduals,                    │
              │               'ss': SolveAndSubtract,                            │
              │               'ac': CorrectOnly,                                 │
              │               'ar': CorrectResiduals,                            │
              │               'as': SubtractOnly }                               │
              │   _solve_gains(gm, …): the main GN/LM loop                       │
              │     - Iterates: gm.compute_residual / compute_jh / compute_jhr   │
              │       / compute_jhj / compute_update                             │
              │     - Convergence on δgain & δχ²                                 │
              │     - Hands off to madmax.flagger.Flagger for in-loop MAD flagging│
              │   _VisDataManager: weighted vs unweighted obs/model arrays      │
              └───────────────────────┬──────────────────────────────────────────┘
                                      │
              ┌───────────────────────▼──────────────────────────────────────────┐
              │ GAIN MACHINES (cubical/machines/)                                │
              │   MasterMachine (abstract_machine.py)                            │
              │     ↑                                                            │
              │   PerIntervalGains (interval_gain_machine.py)                    │
              │     ↑                                                            │
              │   Complex2x2Gains, ComplexW2x2Gains, PhaseDiagGains,             │
              │   PolarizationGains, ParameterisedGains→PhaseSlopeGains,         │
              │   parallactic_machine, IfrGainMachine                            │
              │   JonesChain / JonesChain (robust) — composition of the above    │
              └───────────────────────┬──────────────────────────────────────────┘
                                      │
              ┌───────────────────────▼──────────────────────────────────────────┐
              │ NUMBA KERNELS (cubical/kernels/)                                 │
              │   compute_residual / compute_jh / compute_jhr / compute_jhj /    │
              │   compute_update / compute_corrected / apply_gains /             │
              │   right_multiply_gains  — in {full,diag,diagdiag,phase}_complex, │
              │   {full,diag}_W_complex (weighted), {f,t,tf,ff2}_slope,          │
              │   chain (chain rule), madmax (MAD flagger), rebinning            │
              │   All decorated with @numba.jit(nopython=True, fastmath=True,    │
              │   parallel=use_parallel, nogil=True, cache=use_cache)            │
              │   Memory layouts reordered via allocate_reordered_array          │
              │   to keep antenna pair / corr indices innermost                  │
              └───────────────────────┬──────────────────────────────────────────┘
                                      │
              ┌───────────────────────▼──────────────────────────────────────────┐
              │ DATABASE / OUTPUT (cubical/database/, cubical/plots/)            │
              │   PickledDatabase — native CubiCal ".parmdb" via pickle          │
              │   casa_db_adaptor — exports same tables to CASA caltables       │
              │   Parameter — n-D record with axes, interpolation_axes, grid     │
              │   plots/gainsols.py, plots/ifrgains.py, plots/leakages.py,      │
              │   plots/stats.py, madmax/plots.py                                │
              └──────────────────────────────────────────────────────────────────┘
```

The boxes correspond to top-level sub-packages of `cubical/`. Every arrow is a
direct call (no message passing) except for the I/O ↔ compute hand-off, which goes
through Python `multiprocessing` pickle channels and `/dev/shm` shared memory.

---

## 5. Public CLI: `gocubical`

### 5.1 How invocations are interpreted

From `cubical/main.py` lines 169-201:

* If the **first non-flag argument** does not start with `-`, it is treated as a
  **parset filename**. The parset is parsed via `cubical.tools.parsets.Parset`,
  and its values overlay the defaults from `cubical/DefaultParset.cfg`.
* Any `--section-option value` flags on the command line then overlay the parset.
  This is implemented in `cubical.tools.dynoptparse.DynamicOptionParser`, which
  builds an `optparse` parser **dynamically** from the parset schema (one option
  per `[section] key = value` entry, with type/options/metavar enforced).
* The fully-resolved option dictionary is `GD` (a "global defaults" dict-of-dicts),
  and is also pickled to `cubical.last` in the cwd so that
  `cubical.main.debug()` can resume from it.
* The full dataset of options used is also written back as a parset to
  `<basename>.parset` for provenance.

`gocubical -h` prints the auto-generated help (one option per parset entry).

### 5.2 Output directory & overwrite logic (main.py:200-265)

* `--out-dir OUTDIR` sets a directory whose name has `.cc-out` implicitly appended
  unless it already ends with it or with `/`.
* `--out-name OUTNAME` sets the base filename; if it contains a `/`, OUTDIR is
  ignored.
* If `<basename>.parset` already exists, refuse to proceed unless
  `--out-overwrite` is set.
* `--out-backup` (default 1) renames an existing `.cc-out` directory to
  `<dir>.0`, `.1`, …

### 5.3 Mode dispatch (main.py:347-356)

`--out-mode` selects a `SolverMachine` subclass from `solver.SOLVERS`:

| `--out-mode` | Class                         | Solves? | Writes corrected? | Writes residuals? | Model required? |
| ------------ | ----------------------------- | ------- | ----------------- | ----------------- | --------------- |
| `so`         | `SolveOnly`                   | Yes     | No                | No                | Yes             |
| `sc`         | `SolveAndCorrect`             | Yes     | Yes               | No                | Yes             |
| `sr`         | `SolveAndCorrectResiduals`    | Yes     | Yes (residuals)   | Yes               | Yes             |
| `ss`         | `SolveAndSubtract`            | Yes     | No                | Yes (uncorrected) | Yes             |
| `ac`         | `CorrectOnly`                 | No      | Yes               | No                | No (unless madmax) |
| `ar`         | `CorrectResiduals`            | No      | Yes (residuals)   | Yes               | Yes             |
| `as`         | `SubtractOnly`                | No      | No                | Yes (uncorrected) | Yes             |

The `is_apply_only` and `is_model_required` class attributes on each subclass are
inspected by `main.py` to decide whether to demand `--model-list` and whether the
gain machine should be created in apply-only mode (`solver.py:585-852`).

### 5.4 Jones-term resolution (main.py:328-503)

`--sol-jones` is a **comma-separated list of section names** (default `G`). Each
name `J` is looked up as section `[j]` (lower-cased) in the parset. Each Jones
section is a clone of the `[JONES-TEMPLATE]` template (`DefaultParset.cfg:386-490`),
which defines all per-Jones options (`type`, `time-int`, `freq-int`,
`update-type`, `dd-term`, `solvable`, `load-from`, `save-to`, …). The two
default-instantiated sections are `[g]` (DIE complex-2x2) and `[de]`
(direction-dependent).

If exactly one term is enabled, the corresponding `GAIN_MACHINE_TYPES[type]`
class is used directly. With more than one term, the driver assembles a
`JonesChain` (or `jones_chain_robust_machine.JonesChain` if any term has a `robust*`
type) — see `main.py:423-436`.

`--sol-term-iters N1,N2,...` defines an iteration *recipe* over the chain — for a
two-term chain `G,B`, `--sol-term-iters 10,20,10` means: 10 iters on G, 20 on B,
then 10 more on G.

### 5.5 Full per-section reference

The complete option schema lives in `cubical/DefaultParset.cfg` (556 lines).
The tables below summarise it — **every option below is enforceable on the
command line as `--<section>-<option>`**.

#### `[data]` — visibility data options
| Option       | Default        | Description (paraphrased from DefaultParset.cfg) |
| ------------ | -------------- | ------------------------------------------------ |
| `ms`         | (required)     | Path to the CASA Measurement Set                 |
| `column`     | `DATA`         | MS column to read for observed visibilities      |
| `time-chunk` | `32`           | Chunk size in timeslots (or `'300s'`)            |
| `freq-chunk` | `32`           | Chunk size in channels (or `'128MHz'`)           |
| `rebin-time` | `1`            | On-the-fly time averaging                        |
| `rebin-freq` | `1`            | On-the-fly frequency averaging                   |
| `chunk-by`   | `SCAN_NUMBER`  | Break chunks at jumps in named column(s)         |
| `chunk-by-jump` | `1`         | Jump threshold for `chunk-by`                    |
| `single-chunk`  |             | Process only this chunk ID (debug)               |
| `single-tile`   | `-1`        | Process only this tile index (debug)             |
| `normalize`     | `0`         | Normalise data amplitude to unity                |

#### `[sel]` — data selection
| Option   | Default | Description |
| -------- | ------- | ----------- |
| `field`  | `0`     | FIELD_ID    |
| `ddid`   | `None`  | DATA_DESC_ID(s); supports `5`, `5,6,7`, `5~7`, `5:8`, `5:` |
| `taql`   |         | Extra TaQL string AND-ed with all other selection |
| `chan`   |         | Channel selection (`5`, `10~20`, `10:21`, `:10:2`) |
| `diag`   | `0`     | Use parallel-hand correlations only |

#### `[out]` — output products
| Option              | Default          | Description |
| ------------------- | ---------------- | ----------- |
| `dir`               | `cubical`        | Output dir; `.cc-out` suffix appended |
| `name`              | `cc`             | Base filename |
| `overwrite`         | `0`              | Allow overwrite |
| `backup`            | `1`              | Auto-backup existing `.cc-out` dirs |
| `mode`              | `sc`             | One of `so/sc/sr/ss/ac/ar/as` (see §5.3) |
| `apply-solver-flags`| `1`              | Write solver-raised flags to MS |
| `column`            | `CORRECTED_DATA` | Output visibility column |
| `derotate`          | `None`           | Force enable/disable post-cal de-rotation |
| `model-column`      |                  | Optional: write model visibilities to this column |
| `weight-column`     |                  | Optional: write robust-solver weights to this column |
| `reinit-column`     | `0`              | Re-create the output column from scratch |
| `subtract-model`    | `0`              | Which model index to subtract for residuals |
| `subtract-dirs`     | `:`              | Directions to subtract (`:`, `N`, `N:M`, `N,M,K`) |
| `correct-dir`       | `-1`             | Direction to correct for in DDE mode (-1 = DIE only) |
| `plots`             | `1`              | Generate summary plots; `show` for interactive |
| `casa-gaintables`   | `1`              | Also export solutions as CASA caltables |

#### `[model]` — calibration model
| Option         | Default | Description |
| -------------- | ------- | ----------- |
| `list`         |         | Comma-separated list of MS columns and/or LSMs (with `@dE` cluster tags) |
| `ddes`         | `auto`  | `never`/`auto`/`always` — gate DDE prediction on DDE Jones terms |
| `beam-pattern` | `None`  | FITS beam filename pattern (e.g. `beam_$(corr)_$(reim).fits`) |
| `beam-l-axis`  | `None`  | Override L axis of beam FITS |
| `beam-m-axis`  | `None`  | Override M axis of beam FITS |
| `feed-rotate`  | `0`     | Apply feed-angle rotation; `auto` reads FEED subtable |
| `pa-rotate`    | `0`     | Apply parallactic angle rotation to model |
| `null-v`       | `0`     | Force Stokes V = 0 in model (special polcal mode) |

#### `[montblanc]` — Montblanc predictor
| Option        | Default   | Description |
| ------------- | --------- | ----------- |
| `device-type` | `CPU`     | `CPU` or `GPU` |
| `dtype`       | `double`  | `float` / `double` |
| `mem-budget`  | `1024`    | Megabytes |
| `verbosity`   | `WARNING` | `DEBUG`/`INFO`/`WARNING`/`ERROR` |
| `threads`     | `0`       | OMP threads (0 = default) |
| `pa-rotate`   | `None`    | Override `--model-pa-rotate` for Montblanc only |

#### `[weight]`
| Option         | Default            | Description |
| -------------- | ------------------ | ----------- |
| `column`       | `WEIGHT_SPECTRUM`  | Empty disables |
| `fill-offdiag` | `0`                | Use geometric mean of diagonals for missing off-diags |
| `legacy-v1-2`  | `0`                | Replicate buggy CubiCal ≤1.2.1 weight handling |

#### `[flags]`
| Option            | Default    | Description |
| ----------------- | ---------- | ----------- |
| `apply`           | `-cubical` | Bitflag set(s) to apply pre-cal; `-` prefix excludes |
| `auto-init`       | `legacy`   | Insert BITFLAG and seed it from FLAG/FLAG_ROW |
| `save`            | `cubical`  | Bitflag set name written by CubiCal; `0`/`none` disables |
| `save-legacy`     | `auto`     | Whether to also write FLAG/FLAG_ROW (`0/1/auto/apply`) |
| `reinit-bitflags` | `0`        | Wipe existing BITFLAG before run |
| `warn-thr`        | `0.3`      | Warn if flagged fraction exceeds |
| `see-no-evil`     | `0`        | Override BITFLAG corruption checks |

#### `[degridding]` — DDFacet predict (cf. §13)
Many low-level options including `OverS`, `Support`, `Nw`, `wmax`, `Padding`,
`NDegridBand`, `MaxFacetSize`, `MinNFacetPerAxis`, `NProcess`, `BeamModel`,
`NBand`, `FITSFile`, `FITSFeed`, `FITSFeedSwap`, `DtBeamMin`,
`FITSParAngleIncDeg`, `FITSLAxis`, `FITSMAxis`, `FITSVerbosity`, `FITSFrame`
(`altaz`/`altazgeo`/`equatorial`/`zenith`), `FeedAngle`, `ApplyPJones`,
`FlipVisibilityHands`, `PointingCenterAt`. The `FITSFile` option supports an
elaborate **station-typed JSON config** for heterogeneous arrays
(see DefaultParset.cfg:175-211).

#### `[postmortem]` — chi²-based post-cal flagging
| Option            | Default | Description |
| ----------------- | ------- | ----------- |
| `enable`          | `0`     | Enable post-mortem flagging round |
| `tf-chisq-median` | `1.2`   | Multiplier on median χ² to flag |
| `tf-np-median`    | `0.5`   | Multiplier on median number of valid points |
| `time-density`    | `0.5`   | Flag whole timeslot if fraction flagged > |
| `chan-density`    | `0.5`   | Flag whole channel if fraction flagged > |
| `ddid-density`    | `0.5`   | Flag whole DDID if fraction flagged > |

#### `[madmax]` — MAD-residual flagger
| Option              | Default | Description |
| ------------------- | ------- | ----------- |
| `enable`            | `0`     | `0/1/pretend/trial` |
| `residuals`         | `0`     | Extra round on final residuals |
| `estimate`          | `corr`  | `corr/all/diag/offdiag` |
| `diag`              | `1`     | Flag on parallel-hand residuals |
| `offdiag`           | `1`     | Flag on cross-hand residuals |
| `threshold`         | `10`    | List of σ-thresholds applied at successive iterations |
| `global-threshold`  | `12`    | Same but on per-baseline-MMAD |
| `plot`              | `1`     | `0/1/show` |
| `plot-frac-above`   | `0.01`  | Plot only if flagged fraction exceeds |
| `plot-bl`           |         | Always plot these baselines (comma-separated IDs) |
| `flag-ant`          | `0`     | Flag whole antennas with bad residuals |
| `flag-ant-thr`      | `5`     | σ-threshold for whole-antenna flagging |

#### `[sol]` — solver-level options
| Option            | Default  | Description |
| ----------------- | -------- | ----------- |
| `jones`           | `G`      | Comma-separated list of Jones term names |
| `precision`       | `32`     | `32` (complex64) or `64` (complex128) |
| `delta-g`         | `1e-6`   | Deprecated: per-Jones `epsilon` is preferred |
| `delta-chi`       | `1e-6`   | Deprecated: per-Jones `delta-chi` is preferred |
| `chi-int`         | `5`      | Iterations between χ² checks |
| `last-rites`      | `1`      | Re-estimate χ² and noise at end of cycle |
| `stall-quorum`    | `0.99`   | Fraction of intervals that must stall before exit |
| `term-iters`      | `50`     | Per-term iteration count, or recipe |
| `flag-divergence` | `0`      | Flag intervals that immediately diverge |
| `min-bl`          | `0`      | Min baseline length (m) |
| `max-bl`          | `0`      | Max baseline length (m); 0 disables |
| `subset`          |          | Extra TaQL applied only during solving |

#### `[bbc]` — baseline-based corrections / IFR gains
| Option        | Default | Description |
| ------------- | ------- | ----------- |
| `load-from`   |         | Load BBCs from a previous parmdb |
| `compute-2x2` | `0`     | Compute full 2×2 BBCs vs diag-only |
| `apply-2x2`   | `0`     | Apply full 2×2 BBCs vs diag-only |
| `save-to`     | `{out[name]}-BBC-field_{sel[field]}-ddid_{sel[ddid]}.parmdb` | Save BBC parmdb |
| `per-chan`    | `1`     | Per-channel BBC (else single across band) |
| `plot`        | `1`     | Make BBC plots |

#### `[dist]` — parallelisation
| Option       | Default | Description |
| ------------ | ------- | ----------- |
| `ncpu`       | `0`     | Cap on cores |
| `nworker`    | `0`     | Worker subprocess count (auto if 0) |
| `nthread`    | `0`     | OMP threads per worker (auto if 0) |
| `max-chunks` | `0`     | Max chunks resident in RAM |
| `min-chunks` | `0`     | Min chunks resident in RAM |
| `pin`        | `0`     | Affinity start core, or `N:K` step |
| `pin-io`     | `0`     | Pin I/O process to its own core |
| `pin-main`   | `io`    | Pin main process; `io` shares with I/O |
| `safe`       | `1`     | Memory-safety multiplier (≤1: hard cap on RAM) |

#### `[log]`
| Option         | Default      | Description |
| -------------- | ------------ | ----------- |
| `memory`       | `1`          | Log memory usage (`0/1/2`) |
| `stats`        | `chi2:.3f`   | Format spec for summary stats |
| `stats-warn`   | `chi2:10`    | Highlight when stat > threshold |
| `boring`       | `0`          | Disable progress bars |
| `append`       | `0`          | Append to log instead of overwriting |
| `verbose`      | `0`          | Console verbosity |
| `file-verbose` | `None`       | Logfile verbosity (None = follow console) |

#### `[debug]`
| Option              | Default | Description |
| ------------------- | ------- | ----------- |
| `pdb`               | `0`     | Drop into pdb on exception |
| `panic-amplitude`   | `0`     | Raise if any output amp > N |
| `stop-before-solver`| `0`     | pdb breakpoint before solver |
| `escalate-warnings` | `0`     | Turn warnings into exceptions |

#### `[misc]`
| Option           | Default | Description |
| ---------------- | ------- | ----------- |
| `random-seed`    | `None`  | RNG seed for reproducibility |
| `parset-version` | `0.1`   | Schema version (cmd-line locked) |

#### `[JONES-TEMPLATE]` (and `[g]`, `[de]`)

Template instantiated once per name in `--sol-jones`. Cf. `DefaultParset.cfg:386-490`.

| Option                | Default        | Description |
| --------------------- | -------------- | ----------- |
| `label`               | `{LABEL}`      | Auto-substituted Jones label |
| `solvable`            | `1`            | If 0, term is loaded from disk |
| `type`                | `complex-2x2`  | `complex-2x2/complex-diag/phase-diag/robust-2x2/f-slope/t-slope/tf-plane` (see Machines table) |
| `delay-estimate-pad-factor` | `8`      | FFT padding factor for `f-slope` initial estimate |
| `load-from`           |                | Load solutions from db (must match grid) |
| `xfer-from`           |                | Same, but interpolated onto current grid |
| `save-to`             | `{out[name]}-{JONES}-field_{sel[field]}-ddid_{sel[ddid]}.parmdb` | Output db path |
| `dd-term`             | `0`            | Direction-dependent? |
| `fix-dirs`            |                | Pin these directions as non-solvable |
| `update-type`         | `full`         | Restrict update subspace: `full/diag/phase-diag/amp-diag/amp-scalar/phase-scalar/pzd-diag/leakage/pzd-leakage/rel-leakage/pzd-rel-leakage` |
| `estimate-pzd`        | `0`            | Initialize diagonal phases from PZD estimate |
| `time-int`            | `1`            | Time solution interval |
| `freq-int`            | `1`            | Freq solution interval |
| `max-prior-error`     | `.1`           | Flag intervals with prior error above |
| `max-post-error`      | `.1`           | Flag intervals with posterior variance above |
| `low-snr-warn`        | `75`           | SNR warning threshold |
| `high-gain-var-warn`  | `30`           | Posterior variance warning threshold |
| `clip-low`            | `.1`           | Flag if any diag amp below this |
| `clip-high`           | `10`           | Flag if any amp above this; 0 disables |
| `clip-after`          | `5`            | Iter after which clipping engages |
| `max-iter`            | `20`           | Max iterations on this term |
| `pin-slope-iters`     | `0`            | Iters to hold delay constant in slope solvers |
| `epsilon`             | `1e-6`         | Convergence threshold on gain change |
| `delta-chi`           | `1e-6`         | Stagnation threshold on χ² improvement |
| `conv-quorum`         | `0.99`         | Fraction of intervals that must converge |
| `ref-ant`             | `None`         | Reference antenna (zero phase) |
| `prop-flags`          | `default`      | Flag propagation policy: `never/always/default` |
| `diag-only`           | `0`            | Solve using parallel-hand data only |
| `offdiag-only`        | `0`            | Solve using off-diagonals only (leakage) |
| `robust-cov`          | `compute`      | `compute/identity/hybrid` covariance for robust-2x2 |
| `robust-scale`        | `0`            | Robust covariance down-scaling factor |
| `robust-npol`         | `2`            | 2 or 4 correlations |
| `robust-int`          | `1`            | Iterations between v-param + cov re-fits |
| `robust-flag-weights` | `0`            | Flag from robust weights (dummy iter) |
| `robust-cov-thresh`   | `1`            | Cov threshold treated as RFI |
| `robust-sigma-thresh` | `3`            | σ for robust weight flagging |
| `robust-save-weights` | `0`            | Persist robust weights to MS |
| `estimate-delays`     | `0`            | Initial delay estimate from FFT in f-slope |

The `[g]` and `[de]` sections override a few defaults — `[g]` is DIE
(`dd-term=0`), `[de]` is DDE (`dd-term=1`) with relaxed clipping & error tolerances.

## 6. The Solver Loop in Detail

Source: `simulators/CubiCal/cubical/solver.py`.

### 6.1 Module-level singletons

```python
GD               = None  # global options dict
metadata         = None  # MS metadata
gm_factory       = None  # MasterMachine.Factory subclass instance
ifrgain_machine  = None  # IfrGainMachine
legacy_version12_weights = False
```

These are populated by `cubical.main:main()` before any worker is spawned, and are
inherited by forked workers via `os.fork()` semantics.

### 6.2 `_VisDataManager` (`solver.py:441-580`)

A small helper that holds, per chunk:

* `obser_arr`        — `(n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor)` complex visibilities
* `model_arr`        — `(n_dir, n_mod, n_tim, n_fre, n_ant, n_ant, n_cor, n_cor)` per-direction model
* `flags_arr`        — `(n_tim, n_fre, n_ant, n_ant)` `uint16` bitflags
* `weight_arr`       — same shape as `obser_arr` (or `None`)
* `freq_slice`       — slice into the global frequency grid this chunk represents
* `weighted_obser`, `weighted_model`, `corrupt_weighted_model` — derived views used by the GN/LM loop
* `gm`               — pointer to the gain machine for this chunk

It also exposes `corrupt_residual(model_index, dirs)` which feeds the `SolveAndSubtract`
family of solvers.

### 6.3 The Gauss–Newton / Levenberg–Marquardt loop (`_solve_gains`, `solver.py:53-440`)

The exact pseudocode (paraphrased from source):

```
estimate noise → stats.chunk.noise, inv_var_(antchan|ant|chan)
if not gm.dd_term and model has multiple directions: collapse model directions
gm.precompute_attributes(data, model, flags, inv_var_chan)
madmax = Flagger(GD, label, metadata, stats)        # if --madmax-enable
loop:
    iter += 1
    # 1) Inner update: gm.compute_update(model, data) does the work
    flag_count = gm.compute_update(model_arr, obser_arr)
        # Internally:  jhr, jhjinv, fc = gm.compute_js(obser_arr, model_arr)
        #              gm.implement_update(jhr, jhjinv)
    # 2) gm.flag_solutions(flags, final=0) to propagate gain flags into data flags
    # 3) Every chi-int iters: compute residual, then chi^2 via kernels.generics.compute_chisq
    #    Compare to previous chi^2 → decide convergence/stagnation
    # 4) if gm.has_converged or gm.has_stalled: break
    # 5) madmax.flag(...) every chi-int iters
end loop
gm.flag_solutions(flags, final=1)  # last_rites
```

* `chi-int` (default 5) bounds how often the (relatively expensive) χ² is recomputed.
* `epsilon` (per-Jones), `delta-chi`, `conv-quorum`, and `stall-quorum` together
  define the termination policy.
* `flag-divergence` aborts a chunk early if the very first χ² update is positive.
* The MAD-Max flagger is invoked on the residuals at the same cadence (`chi-int`).

When CubiCal is solving a **JonesChain**, `gm.compute_update` is overridden to
iterate one Jones term at a time, advancing through the `term-iters` recipe in
`main.py:402-421`. `JonesChain.next_iteration` returns a hint indicating whether
a "major" step (term switch) was just taken — `_solve_gains` uses that to decide
whether to re-evaluate χ² immediately rather than waiting for the regular cadence.

### 6.4 `SolverMachine` family (`solver.py:585-852`)

`SolverMachine` is the higher-level **mode dispatcher**, not to be confused with the
**gain machines** that live in `cubical/machines/`. It holds the `_VisDataManager`,
`gm`, `stats`, `madmax`, and `sol_opts`, and exposes a single `run()` method whose
output is the corrected-vis or residual array that gets written back to the tile.

| Class                       | `is_apply_only` | `is_model_required` | What `run()` does                                     |
| --------------------------- | --------------- | ------------------- | ----------------------------------------------------- |
| `SolveOnly`                 | False           | True                | Just `_solve_gains(...)`                              |
| `SolveAndCorrect`           | False           | True                | Solve, then `gm.apply_inv_gains(obs)` → corrected     |
| `SolveAndSubtract`          | False           | True                | Solve, then `vdm.corrupt_residual(...)` → residuals   |
| `SolveAndCorrectResiduals`  | False           | True                | Solve, residuals, then `apply_inv_gains` → corr resid |
| `CorrectOnly`               | True            | bool(madmax)        | Apply prior solutions to obs                          |
| `SubtractOnly`              | True            | True                | Apply prior solutions, generate residuals             |
| `CorrectResiduals`          | True            | True                | Apply prior solutions, residuals, then apply inv      |

The `SOLVERS` dict at `solver.py:845-852` maps `--out-mode` strings to these classes.

### 6.5 `run_solver` (`solver.py:855-960`)

The per-chunk worker entry point. From `main.py`'s `workers.run_process_loop`,
each chunk is dispatched to `cubical.solver.run_solver(solver_type, itile, chunk_key, sol_opts, debug_opts)`,
which:

1. Calls `_init_worker()` to set OMP/affinity/log labels.
2. Looks up the chunk in `tile_list[itile]`, fetches `obser_arr`, `model_arr`,
   `flags_arr`, `weight_arr` via `tile.get_chunk_cubes(...)` (using the
   gain-machine-specific allocators).
3. Applies any pre-existing IFR gains via `ifrgain_machine.apply(obs, freq_slice)`.
4. Creates a `_VisDataManager` and a `GainMachine` via `gm_factory.create_machine`.
5. Instantiates the chosen `SolverMachine` subclass.
6. Calls `solver_machine.run()`.
7. `gm_factory.export_solutions(gm, soldict)` ships the solutions through the
   shared dict back to the I/O process.
8. `tile.set_chunk_cubes(corr_vis, flags?, weights?, chunk_key)` writes the result
   back into the tile (still in shared memory).
9. Returns `solver_machine.stats` (a `SolverStats` instance).

A full traceback is logged on any exception, and the exception re-raised so
`workers._run_multi_process_loop` notices the failure.

---

## 7. Gain Machines (`cubical/machines/`)

### 7.1 Class hierarchy

```
MasterMachine                          (abstract — abstract_machine.py)
├── PerIntervalGains                   (interval_gain_machine.py)
│   ├── Complex2x2Gains                ('complex-2x2', 'complex-diag')
│   ├── ComplexW2x2Gains               ('robust-2x2', 'robust-diag')   ← Student-t
│   ├── PhaseDiagGains                 ('phase-diag')
│   └── PolarizationGains              ('complex-pol')
├── ParameterisedGains                 (parameterised_machine.py)
│   └── PhaseSlopeGains                ('f-slope', 't-slope', 'tf-plane',
│                                       'if-slope', 'if2-slope', 'fif-slope',
│                                       and aliases delay/rate/tec/tec2/
│                                       delay-rate/rate-delay/delay-tec)
├── JonesChain                         (jones_chain_machine.py)
└── JonesChain (robust)                (jones_chain_robust_machine.py)
```

`parallactic_machine.parallactic_machine` and
`ifr_gain_machine.IfrGainMachine` are *not* subclasses of `MasterMachine` — the
former applies a non-solvable feed/PA rotation to the model, and the latter
solves a **baseline-based correction (BBC)** outside the chain.

### 7.2 `MasterMachine` interface (`abstract_machine.py:22-755`)

The official machine interface every concrete machine must implement:

| Method / property                         | Purpose                                                                  |
| ----------------------------------------- | ------------------------------------------------------------------------ |
| `__init__(jones_label, data_arr, ndir, nmod, times, freqs, chunk_label, options, diagonal=None)` | Stash dims, dtypes, options |
| `compute_js(obser_arr, model_arr)`        | Build J^H R and (J^H J)^{-1}; abstract                                   |
| `implement_update(jhr, jhjinv)`           | Apply the parameter update; abstract                                     |
| `compute_update(model_arr, obser_arr)`    | Default = `compute_js` then `implement_update`                            |
| `compute_residual(obs, model, resid, require_full=True)` | Full-resolution residual; abstract                       |
| `apply_inv_gains(obs, corr_vis=None, full2x2=True, direction=None)` | abstract                                       |
| `apply_gains(model_arr, full2x2=True, dd_only=False)` | abstract                                                     |
| `precompute_attributes(data, model, flags, inv_var_chan)` | Equation counts, conditioning. Default returns ``unflagged`` mask |
| `update_model(model_arr)`                 | Hook called when a new model arrives                                     |
| `check_convergence(min_delta_g)`          | abstract                                                                 |
| `flag_solutions(flag_arr, final=0)`       | Propagate gain flags into data flags; abstract                           |
| `num_gain_flags(mask=None, final=False)`  | abstract                                                                 |
| `next_iteration()`                        | Default just bumps `_iters`                                              |
| `restrict_solution(gains)`                | E.g. ref-ant phase fix; abstract                                         |
| `has_converged`, `has_stalled`            | abstract properties                                                      |
| `conditioning_status_string`, `current_convergence_status_string`, `final_convergence_status_string` | abstract / default |
| `exportable_solutions()` (static)         | `{label: (empty_value, axes_list)}` declaring DB schema                  |
| `importable_solutions(grid0)`             | Inverse: what the machine can read back                                  |
| `export_solutions()`                      | Returns dict of arrays to write into parmdb; abstract                    |
| `import_solutions(solutions_dict)`        | Inverse; abstract                                                        |
| `determine_allocators(options)`           | Class method: which kernel-specific allocators to use                    |
| `determine_diagonality(options)`          | Class method: True if this term is diagonal                              |

### 7.3 Inner `MasterMachine.Factory` (`abstract_machine.py:756-1023`)

A **per-Jones-term singleton** that owns the global solution grid and any
parmdb load/save handles. It is created (per term) in `cubical.main`:

```python
solver.gm_factory = jones_class.create_factory(grid=grid, apply_only=...,
                                               double_precision=double_precision,
                                               global_options=GD,
                                               jones_options=jones_opts)
```

Responsibilities:

* `init_solutions()` opens any `load-from`/`xfer-from` parmdbs and creates the
  `save-to` parmdb, defining each parameter via
  `casa_db_adaptor → PickledDatabase.define_param`.
* `make_filename(template, jones_label=...)` expands `{out[name]}-{JONES}-...`
  using `cubical.main.expand_templated_name` (which has access to `GD` plus
  `_runtime_templates` like `{DATE}`, `{TIME}`, `{USER}`, `{HOST}`, `{ENV}`).
* `create_machine(data_arr, n_dir, n_mod, chunk_ts, chunk_fs, label)` allocates a
  fresh per-chunk `MasterMachine` instance and pre-loads its solutions if the
  factory has any.
* `export_solutions(gm, subdict)` pickles the chunk's solutions into the shared
  dict for the I/O worker to flush to disk.
* `save_solutions(subdict)` is the I/O-side counterpart that calls
  `parmdb.add_chunk(...)`.
* `determine_allocators()` returns the kernel-appropriate allocator triple
  `(allocate_vis_array, allocate_flag_array, allocate_gain_array)` — these are
  used by `tile.get_chunk_cubes` to lay out the visibility/flag arrays in the
  exact memory order expected by the JIT kernels for that machine type.
* `close()` finalises all open parmdbs.

### 7.4 `PerIntervalGains` (`interval_gain_machine.py`)

The base class for all "constant per (time,freq) interval" machines. Implements:

* Solution-interval bookkeeping (`time-int`, `freq-int`).
* Gain array allocation (DD vs DI).
* Reference antenna application (`restrict_solution` zeros the phase of the
  reference antenna at every iteration).
* Per-interval clipping (`clip-low`, `clip-high`, `clip-after`).
* Prior/posterior error estimation (`max-prior-error`, `max-post-error`,
  `low-snr-warn`, `high-gain-var-warn`) and corresponding
  `FL.LOWSNR` / `FL.GVAR` flag bits.
* Convergence and stagnation detection (`has_converged`, `has_stalled`).
* Default `compute_residual` / `apply_gains` / `apply_inv_gains` implementations
  that delegate to the kernel functions in `cubical.kernels.full_complex` (or
  `diag_complex` for diagonal terms).

### 7.5 Concrete machines

| `type=` string                | Class                                  | Backing kernel(s)                                  |
| ----------------------------- | -------------------------------------- | -------------------------------------------------- |
| `complex-2x2`                 | `Complex2x2Gains`                      | `kernels/full_complex.py` (also `diag_complex.py` if diag) |
| `complex-diag`                | `Complex2x2Gains` (diag mode)          | `kernels/diag_complex.py`, `diagdiag_complex.py`   |
| `complex-pol`                 | `PolarizationGains`                    | `kernels/full_complex.py` (off-diag only)          |
| `phase-diag`                  | `PhaseDiagGains`                       | `kernels/diag_phase_only.py`, `phase_only.py`      |
| `robust-2x2`                  | `ComplexW2x2Gains`                     | `kernels/full_W_complex.py`                        |
| `robust-diag`                 | `ComplexW2x2Gains` (diag)              | `kernels/diag_robust.py`                           |
| `f-slope`, `delay`            | `PhaseSlopeGains`                      | `kernels/f_slope.py`                               |
| `t-slope`, `rate`             | `PhaseSlopeGains`                      | `kernels/t_slope.py`                               |
| `tf-plane`, `delay-rate`, `rate-delay` | `PhaseSlopeGains`             | `kernels/tf_plane.py`                              |
| `if-slope`, `tec`             | `PhaseSlopeGains` (1/ν dep)            | `kernels/f_slope.py`                               |
| `if2-slope`, `tec2`           | `PhaseSlopeGains` (1/ν² dep)           | `kernels/f_slope.py`                               |
| `fif-slope`, `delay-tec`, `tec-delay` | `PhaseSlopeGains` (joint)      | `kernels/ff2_slope.py`                             |

The dispatch happens in `machines/machine_types.py`:

```python
GAIN_MACHINE_TYPES = {
    'complex-2x2': Complex2x2Gains,
    'complex-diag': Complex2x2Gains,
    'complex-pol': PolarizationGains,
    'phase-diag': PhaseDiagGains,
    'robust-2x2': ComplexW2x2Gains,
    'robust-diag': ComplexW2x2Gains,
    # plus all SLOPE_TYPES → PhaseSlopeGains
}
```

`SLOPE_TYPES` (in `slope_machine.py`) declares each slope type as a tuple of
**dependent variables** (`DepVar.TIME`, `DepVar.FREQ`, `DepVar.IFREQ` for `1/ν`,
`DepVar.IFREQ2` for `1/ν²`, `DepVar.PHASE0` for the per-antenna phase offset)
and the kernel module name. This declarative mapping is what allows a single
`PhaseSlopeGains` class to support seven different parameterisations.

### 7.6 `JonesChain` (`jones_chain_machine.py`, `jones_chain_robust_machine.py`)

When `--sol-jones` lists more than one term, a `JonesChain` (633 LOC) is built.
It composes a list of child `MasterMachine` instances and:

* Iterates over them according to the `--sol-term-iters N1,N2,N3,...` recipe.
* Maintains a chain-rule helper from `cubical.kernels.chain` to apply
  `J = J_n J_{n-1} … J_1` when computing residuals, and the proper
  Wirtinger chain rule for J^H J / J^H r when updating each term.
* Handles direction-dependence: only the DD terms see the `(n_dir, ...)`
  axis of the model, while DIE terms collapse over directions.
* Forwards `flag_solutions` to every child term, with the
  `prop-flags = never|always|default` policy.
* Aggregates parmdb load/save: each term has its own `save-to` /
  `load-from` filename template.

`jones_chain_robust_machine.JonesChain` (745 LOC) is the analogous chain that
embeds a `ComplexW2x2Gains` Student-t solver inside the chain, with the same
v-parameter / covariance machinery (see §10.5).

### 7.7 `IfrGainMachine` (`ifr_gain_machine.py`)

This is **not** a Jones term in the chain. It implements per-baseline,
per-channel "interferometer gains" (a.k.a. "Bandpass-Baseline Calibration",
BBC). It is created always, but only "computes" if `load_model` is True
(`main.py:509`).

Per-chunk update:

```python
ifrgain_machine.update(weighted_obser, corrupt_weighted_model, flags_arr, freq_slice, soldict)
```

It accumulates `Σ_t Σ_chunks D_pq · M_pq* / |M_pq|²` (and Hermitian conjugate),
producing a multiplicative per-baseline correction. After all tiles are
processed:

* `accumulate(soldict)` is called by the I/O worker on every saved chunk.
* `save()` writes a dedicated parmdb (`save-to` template includes
  `BBC-field_X-ddid_Y.parmdb`).

In `apply()` (called at the start of every `run_solver` iteration), it multiplies
`obser_arr` by the loaded BBC for the chunk's freq slice. The `--bbc-compute-2x2`
and `--bbc-apply-2x2` switches choose between diag-only and full 2×2 BBCs.

A diagnostic plot is produced by `cubical.plots.ifrgains.make_ifrgain_plots`
(`main.py:641`).

### 7.8 Parallactic-angle "machine" (`parallactic_machine.py`)

A 325-LOC, *non-solvable* helper that sits between Montblanc/MS column models
and the solver. Given `--model-pa-rotate 1`, it builds per-time-per-antenna
2×2 rotation matrices using `astropy.coordinates` (alt-az feed assumed) and
applies them to the model visibilities.

When `--out-derotate` is set, it also applies the inverse rotation to the
output corrected-data column so that downstream tools can still treat the data
in a sky-fixed frame.

---

## 8. Numba Kernel Layer (`cubical/kernels/`)

### 8.1 Why Numba and not Cython?

Despite the "Cython kernels" wording in older docstrings (`full_complex.py:6`),
all hot kernels are implemented with **`numba.jit(nopython=True, fastmath=True,
parallel=use_parallel, nogil=True, cache=use_cache)`**. The `use_parallel` flag
is set from `cubical.kernels.num_omp_threads > 1` (i.e. only when
`--dist-nthread > 1`), which makes numba's `prange` parallelise across
antenna pairs. The `cache=` flag is read from `cubical.kernels.use_cache`,
which is normally False; setting it to True caches compiled kernels on disk.

### 8.2 Memory-layout contract

Every kernel module declares an explicit memory-layout permutation, e.g. in
`full_complex.py`:

```python
_model_axis_layout = [4,5,1,2,3,0,6,7]    # AAMTFD CC
_gain_axis_layout  = [3,1,2,0,4,5]        # ATFD CC
_flag_axis_layout  = [2,3,0,1]            # AATF
```

`cubical.kernels.allocate_reordered_array(shape, dtype, order, zeros=False)`
builds an array of a given logical shape, but stores it with the dimensions
listed in `order` as the slowest-varying, transposed back to logical layout for
external code. This ensures that the inner antenna-pair loops in `JHJ`/`JHr`
are over the *fastest-varying* axes, maximising cache locality.

`Complex2x2Gains.determine_allocators` returns the right allocator triple
to `tile.get_chunk_cubes`, so that the data is loaded *directly* into the
preferred layout — no copies.

### 8.3 Kernel module catalogue

| Module                | Public functions                                                                                       | Notes |
| --------------------- | ------------------------------------------------------------------------------------------------------ | ----- |
| `generics.py`         | `compute_2x2_inverse`, `compute_diag_inverse`, `compute_chisq`, `compute_chisq_diag`, `compute_chisq_offdiag` | Used by all machines |
| `full_complex.py`     | `compute_residual`, `compute_jh`, `compute_jhr`, `compute_jhj`, `compute_update`, `compute_corrected`, `apply_gains`, `right_multiply_gains` + allocators | Default 2×2 complex-gain kernel |
| `full_W_complex.py`   | Same set + weight inputs                                                                              | Robust 2×2 |
| `diag_complex.py`     | `compute_residual`, `compute_jh`, `compute_update`, `compute_corrected`, `apply_gains`, `right_multiply_gains` | Diagonal-only optimisation |
| `diagdiag_complex.py` | Same set, both data and gains diagonal                                                                | Tightest inner loop |
| `diag_phase_only.py`  | Phase-only diagonal                                                                                   | `phase-diag` |
| `phase_only.py`       | Phase-only 2×2                                                                                        | Used by parameterised |
| `diag_robust.py`      | Diagonal + Student-t                                                                                  | `robust-diag` |
| `f_slope.py`          | Phase = m·ν + φ₀ (delay)                                                                              | Per-antenna |
| `t_slope.py`          | Phase = m·t + φ₀ (rate)                                                                               |       |
| `tf_plane.py`         | Phase = a·t + b·ν + φ₀ (delay+rate)                                                                  |       |
| `ff2_slope.py`        | Phase = a·ν + b/ν + φ₀ (delay+TEC)                                                                   |       |
| `chain.py`            | `multiply_jones_chain`, `apply_inv_chain`, etc.                                                       | Used by JonesChain |
| `madmax.py`           | `compute_mad`, `compute_mad_internals`, `compute_mad_per_corr`, `threshold_mad`                       | MAD-residual flagger |
| `rebinning.py`        | `rebin_index_columns`, `rebin_vis`, `rebin_model`                                                     | On-the-fly time/freq averaging via `--data-rebin-*` |

Inside a typical kernel (`full_complex.compute_jhr`):

```python
@jit(nopython=True, fastmath=True, parallel=use_parallel, cache=use_cache, nogil=True)
def compute_jhr(jh, r, jhr, t_int, f_int):
    n_dir, n_mod, n_tim, n_fre, n_ant, _, _, _ = jh.shape
    for d in range(n_dir):
        for m in range(n_mod):
            for ti in prange(n_tim_int):
                for fi in range(n_fre_int):
                    for aa in range(n_ant):
                        for ab in range(n_ant):
                            for c in range(2):
                                for cc in range(2):
                                    jhr[d, ti, fi, aa, c, cc] += ...
```

The use of `prange` over the time-interval axis is what lets the compute
worker exploit `--dist-nthread`. With one OMP thread per worker, the kernel
falls back to a plain serial `range`.

### 8.4 The Wirtinger / complex-Jacobian update

For a single direction-independent diagonal Jones term `g_p` per antenna `p`:

```
V_pq(model)        = g_p · M_pq · g_q*
∂V_pq / ∂g_p       =  M_pq · g_q*           (Wirtinger derivative)
∂V_pq / ∂g_p*      =  0                     (analytic in g_p)
```

The Gauss–Newton normal equations in the **complex** Wirtinger formulation are

```
  J^H · J · Δg = J^H · r
```

with `r = D − V(model)`. CubiCal builds these as 2×2 matrices in correlation
space (`compute_jh`, `compute_jhr`, `compute_jhj`), then inverts the
diagonal-block J^H J using `generics.compute_2x2_inverse` (or
`compute_diag_inverse`), and applies the update. The tradeoff vs. the
"real-augmented" formulation is that all arithmetic stays complex — half the
floating-point ops, half the memory.

For chained `J_n … J_1`, `cubical.kernels.chain` performs the chain-rule
transpose:

```
∂V/∂J_k = (J_{k+1} ... J_n)^H · M_pq · (J_1 ... J_{k-1})^H g_q*
```

so that `JonesChain.compute_js` only ever has to assemble the J^H R / J^H J
contributions for the term currently being iterated.

### 8.5 The robust (Student-t) update

`ComplexW2x2Gains` (`complex_W_2x2_machine.py`, 642 LOC) generalises the GN
update to a maximum-likelihood Student-t model:

* Per-visibility weights `w_pq` are computed from the residuals as
  `w_pq = (ν + 2) / (ν + |r_pq|² Σ^{-1}_pq)` where `ν` is the Student-t
  degrees-of-freedom and `Σ` the residual covariance.
* `--JONES-robust-cov compute|identity|hybrid` chooses between full 2×2 covariance,
  identity, or hybrid (cov but capped at 1).
* `--JONES-robust-int N` sets the cadence at which `ν` and `Σ` are re-fit
  (default 1, i.e. every iteration).
* `--JONES-robust-flag-weights` runs a dummy iteration to pre-flag visibilities
  whose weights fall below a `robust-sigma-thresh` σ-cut.
* `--JONES-robust-save-weights` persists the final `w_pq` matrix to the MS
  weight column (only meaningful if `--out-weight-column` is set).

The kernel is `kernels/full_W_complex.py`; everything else (chain composition,
flag policy, parmdb interaction) is identical to the standard solver.

---

## 9. Data Handler (`cubical/data_handler/`)

### 9.1 `MSDataHandler` (`ms_data_handler.py`, 1578 LOC)

Constructor (`ms_data_handler.py:153-389`) opens the MS via
`pt.table(self.ms_name, readonly=False, ack=False)` and immediately reads the
sub-tables `ANTENNA`, `FIELD`, `SPECTRAL_WINDOW`, `POLARIZATION`,
`DATA_DESCRIPTION`, `OBSERVATION`, `FEED`. Key attributes set up at this stage:

* `ms` — top-level table handle
* `metadata` — a `Metadata` record with antenna names, antenna positions,
  observation epochs, etc., consumed by the gain machines and the casa caltable
  exporter
* `ctype = np.complex64` (MS data type), `wtype = np.float32`
* `nmscorrs` (1, 2, or 4); `ncorr` (1, 2, or 4 — set to 2 if `--sel-diag` and
  `nmscorrs == 4`); `_corr_slice` (`(0,3)` for diag-of-4)
* `ncorr == 1` triggers a deprecation warning (`ms_data_handler.py:268-271`).
* `phadir` — phase centre direction (`astropy.SkyCoord`)
* `chunk_freq`, `rebin_freq` — channel selection and on-the-fly rebinning

The `define_chunk(time_chunk, rebin_time, freq_chunk, chunk_by, chunk_by_jump,
chunks_per_tile, max_chunks_per_tile)` method
(`ms_data_handler.py:978-1224`) is the heart of the chunking scheme. It:

1. Sorts visibilities by `(TIME, ANTENNA1, ANTENNA2)` via TaQL.
2. Walks through the rows building **`RowChunk`** objects, each describing a
   contiguous range of MS rows that all share `(DDID, time_chunk_index)`.
3. Groups consecutive `RowChunk`s into **`MSTile`** objects; a new tile is
   started either when `chunks_per_tile` is reached or when a `chunk-by` jump
   is detected.
4. Returns `(chunks_per_tile, tile_list)` to `main`.

`define_flags(tile_list, flagopts)` (`ms_data_handler.py:1225-1383`) is a
*one-pass scan over the entire MS* that:

* Adds the `BITFLAG` column if `--flags-auto-init`.
* If `--flags-reinit-bitflags`, wipes BITFLAG.
* Populates the `Flagsets` registry from the `BITFLAG` column keywords,
  matching against the `--flags-apply` request.
* Reads the FLAG/FLAG_ROW columns and any requested bitflag sets.
* Prints a tally of pre-cal flagged fractions.

### 9.2 `MSTile` and `RowChunk` (`ms_tile.py`, 1493 LOC)

`RowChunk` (line 41) is a tiny dataclass: `(ddid, tchunk, timeslice, rows, rows0)`.

`MSTile` (line 70) is the unit of I/O. It owns:

* A list of `RowChunk`s (the contiguous MS rows it covers)
* A `SharedDict` (`tools.shared_dict`) carved out of `/dev/shm`
* Per-chunk cubes (allocated lazily in `load()` using the gain-machine
  allocators)

Methods of interest:

| Method                                | Purpose |
| ------------------------------------- | ------- |
| `load(load_model=True)`               | Reads DATA, FLAG, BITFLAG, optional MODEL, optional WEIGHT_SPECTRUM, predicts via Montblanc/DDFacet if needed, materialises the `(d,m,t,f,a,a,c,c)` cubes for every chunk |
| `save(final=False, only_save=...)`    | Writes corrected/model/weight/bitflag back to MS via `putcolslice` |
| `get_chunk_cubes(key, ctype, allocator, flag_allocator)` | Returns `(obser, model, flags, weights)` for one chunk |
| `set_chunk_cubes(cube, flag_cube, weight_cube, key, column='covis')` | Replaces solver outputs into the tile's shared dict |
| `create_solutions_chunk_dict(key)`    | Sub-dict for the gain machine's solutions |
| `iterate_solution_chunks()`           | Yields each chunk's solutions for parmdb writing |
| `release(final=False)`                | Frees the shared-memory tile |

The `_column_to_cube` and `_cube_to_column` helpers (line 354 / 462) are the
core MS↔cube transpose operations. They flatten `(t,f,row)` rows back into the
`(t,f,a,a,c,c)` cube layout, accounting for `(ant1,ant2)` permutations and
hermitian-conjugation.

### 9.3 Model prediction back-ends

CubiCal can predict model visibilities from any of:

1. **An MS column** (`MODEL_DATA`, `CORRECTED_DATA`, …) — handled directly in
   `_column_to_cube`.
2. **A unity model `1`** — i.e. `V_pq = δ_pq · I`, used for phase-only solutions
   on a flat-spectrum source (`init_models` line 658-660).
3. **A Tigger LSM** (`*.lsm.html`) — predicted via **Montblanc** (`MBTiggerSim.py`,
   `TiggerSourceProvider.py`). The LSM may carry direction tags
   (`@dE` notation) that get clustered into separate model directions
   (`cluster_sources` in `TiggerSourceProvider.py:228`).
4. **A DDFacet `DicoModel`** (a faceted image-domain model) — degridded via
   `cubical/degridder/DDFacetSim.py` (uses DDFacet's gridder), with
   beam application via `FITSBeamInterpolator.py`.

Each can be combined with `+` and `+-` (subtract) operators in `--model-list`
(`ms_data_handler.py:651-720`). Multiple semicolon-separated entries in the
list create *multiple* models that the solver can use as alternative model
columns.

### 9.4 `MBTiggerSim.py` and `TiggerSourceProvider.py`

* `MSSourceProvider` and `ColumnSinkProvider` are Montblanc `SourceProvider` /
  `SinkProvider` adapters: they feed Montblanc the MS UVWs / channel
  frequencies / phase centre, and absorb the resulting visibility cube.
* `simulate(src_provs, snk_provs, polarisation_type, opts)` is the main entry,
  driving Montblanc's `RIME solver`.
* `_shutdown_mb_slvr()` cleans up the GPU/CPU solver between tiles.
* `TiggerSourceProvider.py` reads `astro-tigger-lsm` LSMs, materialises Stokes
  IQUV / position / shape / spectral-index / RM arrays, and exposes them to
  Montblanc through the source-provider API. The `cluster_sources(sm, dde_tag)`
  function does the `@dE`-tag clustering.

### 9.5 DDFacet bridge (`cubical/degridder/`)

| File                      | Role |
| ------------------------- | ---- |
| `DDFacetSim.py` (680 LOC) | Wraps DDFacet's `ClassImage` / `ClassFFTW` to compute model visibilities from a `DicoModel` (faceted image model) |
| `DicoSourceProvider.py`   | Reads `*.DicoModel` files and exposes a per-cluster source iterator |
| `FITSBeamInterpolator.py` | Wraps DDFacet's `ClassFITSBeam` for E-Jones application |
| `geometry.py` (576 LOC)   | Faceting geometry, l/m grids, w-projection helpers |

This back-end is selected automatically when a model component ends in
`.DicoModel` (`ms_data_handler.py:663`). It bypasses Montblanc and uses
DDFacet's own gridders, controlled by the `[degridding]` parset section.

### 9.6 `wisdom.py`

96 LOC. Implements `estimate_mem(ms, tile_list, data_opts, dist_opts)`, a
heuristic memory-budget estimator. Compares estimated tile-resident memory
against `psutil.virtual_memory()` and aborts (or warns) if the
`--dist-safe` fraction is exceeded.

---

## 10. Flagging

### 10.1 Flag bits (`cubical/flagging.py:23-51`)

CubiCal manages a `uint16` BITFLAG column with sixteen pre-defined bits:

| Bit | Mask     | Symbol     | Meaning |
| --- | -------- | ---------- | ------- |
| 0   | `0x0001` | `PRIOR`    | Prior flag (from MS FLAG/FLAG_ROW or another flagset) |
| 1   | `0x0002` | `MISSING`  | Missing data (no row in MS for this baseline at this time/freq) |
| 2   | `0x0004` | `INVALID`  | Invalid data (zero, inf, nan in DATA) |
| 3   | `0x0008` | `ILLCOND`  | Solution ill-conditioned — bad inverse |
| 4   | `0x0010` | `DIVERGE`  | Solution diverged |
| 5   | `0x0020` | `NOSOL`    | Missing solution (in `load-from`/`xfer-from`) |
| 6   | `0x0040` | `GOOB`     | Gain solution out of bounds |
| 7   | `0x0080` | `BOOM`     | Gain solution exploded (inf/nan) |
| 8   | `0x0100` | `GNULL`    | Gain solution went to zero |
| 9   | `0x0200` | `LOWSNR`   | Prior SNR too low for gain solution |
| 10  | `0x0400` | `GVAR`     | Posterior variance too high |
| 11  | `0x0800` | `INVMODEL` | Invalid model (zero, inf, nan) |
| 12  | `0x1000` | `INVWGHT`  | Invalid weight (inf or nan) |
| 13  | `0x2000` | `NULLWGHT` | Null weight |
| 14  | `0x4000` | `MAD`      | Residual exceeds MAD-based threshold |
| 15  | `0x8000` | `SKIPSOL`  | Omit from solver (transient mark, not a true flag) |

The `Flagsets` class (`flagging.py:53-213`) is the on-disk-name registry: each
named "flagset" (`legacy`, `cubical`, `aoflagger`, …) corresponds to one
bit in BITFLAG, recorded as `FLAGSET_<name>` keywords on the BITFLAG column.
`--flags-apply` and `--flags-save` operate on these names.

### 10.2 MAD-Max in-loop flagger (`cubical/madmax/flagger.py`, `kernels/madmax.py`)

Algorithm (per chunk, every `chi-int` solver iterations and once at the end):

1. Compute `|residual|` per (t,f,a,a,c,c).
2. Estimate per-baseline MAD: median absolute residual, with normalisation
   factor 1/1.4826 (so MAD ≈ σ for Gaussian residuals).
3. Optionally compute the **MMAD** (median of per-baseline MADs) for a
   global second-pass cut.
4. Flag points where `|r| > S · MAD/1.4826`. `S` is taken from the
   `--madmax-threshold` list — first invocation uses the first value,
   then the next, etc., reusing the last value once exhausted.
5. Optionally produce diagnostic plots (`madmax/plots.py`) of the worst-flagged
   baseline and of any user-requested baselines (`--madmax-plot-bl`).

Modes (`--madmax-enable`):

| Value     | Effect |
| --------- | ------ |
| `0`       | Disabled |
| `1`       | Active flagging — flags written to BITFLAG with bit `FL.MAD` |
| `pretend` | Compute and report, but do not actually flag (`flagbit = 0`) |
| `trial`   | Apply during solving, but discard before writing back |

Antenna-level flagging (`--madmax-flag-ant 1`) further looks at which antennas
are responsible for excess MAD; those that exceed `--madmax-flag-ant-thr`
σ are flagged whole.

The `--madmax-residuals` option enables an extra round on the *final*
residuals after the solver converges (works in any `--out-mode`).

### 10.3 Post-mortem χ²-based flagging (`cubical/flagging.py:214-344`)

Triggered by `--postmortem-enable 1` after the run (in `main.py:617`). Operates
on `SolverStats.timechan.chi2` accumulated across all chunks:

* Median χ² → `chi_median_thresh × median` flags timeslot/channel slots
  exceeding it.
* Median count per slot → flags slots with `nvalid < np_median_thresh × median`.
* `--postmortem-time-density` / `--chan-density` / `--ddid-density` flag
  whole timeslots / channels / DDIDs if more than the given fraction is
  flagged.
* If `--flags-save`, the new flags are written back to the MS (only if the
  run did not use `--data-single-chunk`).

Plots are produced inline (a 6-panel `pylab.imshow` figure) when
`--out-plots` is enabled.

---

## 11. Statistics (`cubical/statistics.py`)

The `SolverStats` (`statistics.py:20-360`) class is the per-chunk and global
statistics container. It owns four record arrays:

| Array        | Shape          | Per-element fields |
| ------------ | -------------- | ------------------ |
| `chanant`    | `(n_fre, n_ant)` | `dv2, dv2n, dr2, dr2n, chi2, chi2n, initchi2, initchi2n` |
| `timeant`    | `(n_tim, n_ant)` | same                                          |
| `timechan`   | `(n_tim, n_fre)` | same                                          |
| `chunk`      | scalar           | `label, num_prior_flagged, num_data_points,` plus `chi2u, noise, chi2, iters, num_solutions, num_converged, num_stalled, num_sol_flagged, num_mad_flagged, frac_converged, frac_stalled, end_chi2,` plus 100 sets of these stamped at every "intermediate" step (see §6.3) |

* `dv2`, `dv2n` track the noise estimate (sum of |Δvis|² and count) computed by
  `estimate_noise` (statistics.py:110), which uses *first differences along the
  channel axis* to estimate the per-(t,a) noise level.
* `chi2`, `chi2n` accumulate the post-solve χ² and its valid-point count.
* `initchi2`, `initchi2n` likewise but from the *first* iteration, providing
  the χ² improvement metric.

`SolverStats` methods of interest:

* `_init_for_chunk(data)` — allocate all record arrays for one chunk.
* `_concatenate(stats_dict)` — stitch per-chunk SolverStats objects into one
  global object (called in `main.py:593`).
* `save(filename)` / `load(file)` — pickle just the four record arrays
  (no class needed to read).
* `estimate_noise(data, flags, residuals=False)` — returns
  `(noise, inv_var_antchan, inv_var_ant, inv_var_chan)`.
* `get_notrivial_chunk_statfields()` — returns the list of fields with
  non-zero values (used by `print-cubical-stats`).
* `format_chunk_stats(format_str, threshold=...)` — formats a per-chunk
  table for the `--log-stats` console output.
* `apply_flagcube(flag3)` — back-applies a `(n_times, n_ddids, n_chans)` flag
  cube to the per-(time,chan) statistics (post-mortem flagging).

The full `<basename>.stats.pickle` file is what `cubical/bin/print-cubical-stats`
consumes.

---

## 12. Parameter Database (`cubical/database/`, `cubical/param_db.py`)

### 12.1 The "parmdb" file format

CubiCal solutions are stored in a custom **append-only pickle stream**. The
implementation lives in `cubical/database/pickled_db.py` (`PickledDatabase`).

Layout of `<basename>-G-field_X-ddid_Y.parmdb`:

```
[HEADER]   pickle.dump(metadata: OrderedDict(mode='fragmented', time=..., field=...))
[PARAM 1]  pickle.dump(Parameter(name='G:gain', dtype=complex64, axes=['dir','ant','time','freq','corr1','corr2'], ...))
[CHUNK]    pickle.dump(_ParmSegment(name='G:gain', array=ma.array(...), grid={time:..., freq:...}))
[CHUNK]    pickle.dump(...)
...
[PARAM 2]  pickle.dump(Parameter(name='G:gflags', dtype=uint16, axes=[...]))
[CHUNK]    ...
...
```

Plus a sibling `<filename>.skel` file containing the **finalised** Parameter
skeletons (i.e. fully-resolved axis grids), so a reader can know the full
shape without scanning the entire data file.

The `_create()` method writes to `filename + ".tmp"`; on `close()` it calls
`_backup_and_rename(backup)`, which renames any existing file to
`filename + ".0"`/`.1`/... and atomically promotes the .tmp file into place.

### 12.2 `Parameter` class (`database/parameter.py`, 796 LOC)

A `Parameter` describes a single named, multidimensional solution table. Key
attributes:

* `name` — e.g. `"G:gain"`, `"G:gflags"`, `"B:bandpass"`
* `dtype` — `complex64`/`complex128`/`uint16`/`float64`
* `axis_labels`, `axis_index`, `ax` — list of axis names (`"dir"`, `"ant"`,
  `"time"`, `"freq"`, `"corr1"`, `"corr2"`, …) and a `_Record` for sugar
  access (`p.ax.time`)
* `interpolation_axes` — usually `["time", "freq"]` (must be ≤ 2 axes)
* `grid` — list of coordinate vectors per axis (None until populated)
* `grid_index` — per-axis ordered map from coordinate value → ordinal index
  (for sparse / labelled axes like antenna names)
* `shape` — numerical shape after finalisation
* `_populated`, `_state` — tracks whether the parameter is in
  `prototype` / `skeleton` / `populated` state

Public reader API once a parmdb has been loaded:

* `get_slice(ant=..., corr1=..., corr2=..., dir=..., time=..., freq=...)` — returns
  the masked array slice for the requested discrete axis values; continuous
  axes can be sliced by index.
* `reinterpolate(time=..., freq=..., method='linear'|'nearest')` — re-grids
  the parameter onto the requested time/freq vectors using `scipy.interpolate`.
* `interpolated_slice(...)` — one-shot get-slice + reinterpolate.
* `__call__(...)` — sugar over `get_slice` with grid-coordinate kwargs.

Writer API (used inside the gain-machine `Factory.export_solutions`):

* `add_chunk(name, array, grid={...})` on the `PickledDatabase` instance.
  The first `add_chunk` for a parameter triggers a pickle of the prototype
  Parameter object; subsequent calls just append `_ParmSegment` records.

Every Jones term creates **two** parameters in its parmdb: the gain values
themselves (e.g. `"G:gain"`) and the gain flags (e.g. `"G:gflags"`). For
parameterised machines (slope/TEC), the underlying *parameters* are stored
instead of the materialised gain (e.g. `"K:delay"`).

### 12.3 CASA caltable export (`database/casa_db_adaptor.py`, 490 LOC)

When `--out-casa-gaintables 1` (the default), every parmdb is mirrored into a
CASA caltable on close. The export is implemented by
`casa_caltable_factory` (`casa_db_adaptor.py:23`) which:

1. Extracts a packaged empty caltable from `blankcaltable.CASA.tgz` (alongside
   the source file).
2. Populates the `ANTENNA`, `FIELD`, `OBSERVATION`, `SPECTRAL_WINDOW` etc.
   subtables from the MS metadata that was passed in via `set_metadata(ms)`.
3. Creates one of the following CASA-style tables depending on the Jones type:
   * `create_G_table` — phase-only `Gphase` caltable
   * `create_B_table` — full bandpass `B` caltable (diag-only or full 2×2)
   * `create_D_table` — leakage `D` caltable
   * `create_K_table` — delay `K` caltable

`casa_db_adaptor.close()` (`line 483`) is the orchestration entry-point that
calls the factories on the right parameters.

> **Caveat from source (`line 44`):** "Gaintables cannot be written in Python 3
> mode due to current casacore implementation issues" — this fallback prints
> an error and skips export when running on Py3.

---

## 13. Plots (`cubical/plots/`, `cubical/madmax/plots.py`)

`cubical.plots.make_summary_plots(st, ms, GD, basename)` (`plots/__init__.py:27`)
is the dispatcher invoked by `main.py:630`. It calls into:

* `plots.gainsols.plot_bandpass_cc` and `plot_gain_cc` for waterfall/amp-phase
  plots of every Jones term in the parmdb (659 LOC). Supports both CubiCal
  parmdbs and AIPS PRTAB bandpass dumps.
* `plots.ifrgains.make_ifrgain_plots` for per-baseline BBC plots
  (282 LOC) — invoked separately in `main.py:646`.
* `plots.leakages` — D-Jones (polarisation leakage) per-antenna plots
  (183 LOC). Used by the `plot-leakage-solutions` script.
* `plots.stats` — χ² maps, SNR maps, convergence/iter heatmaps from the
  `SolverStats` object (106 LOC).

`madmax.plots` (260 LOC) is invoked by `madmax.flagger.Flagger.report_carnage`
to draw before/after waterfall plots of the worst-flagged baselines and any
user-requested baselines.

All plotting is matplotlib-based. The `--out-plots show` mode brings up
interactive figures; otherwise plots are saved as PNGs alongside the parmdb.

---

## 14. Tools (`cubical/tools/`)

### 14.1 `parsets.py` — Parset reader (285 LOC)

`Parset` class (line 133) parses the CubiCal-style `.cfg` / `.parset` files.
Each line `key = value  # comment #attr:val` is broken into:

* The `value` (parsed via `parse_as_python` if possible, else kept as a string).
* The `comment` (becomes the autogenerated `--help` string).
* Optional `#attr:value` attributes:
  * `#type:str|int|float|bool` — enforce a Python type for the value
  * `#options:A|B|C` — restrict to a fixed set
  * `#metavar:NAME` — set the optparse `metavar`
  * `#cmdline-only:1` — disallow parset-only setting
  * `#no_cmdline:1`, `#no_print:1` — visibility flags

The class also recognises the `[SECTION-TEMPLATE]` mechanism: a section whose
`_NameTemplate` references a higher-level option (e.g. `_ExpandedFrom = --sol-jones`)
is instantiated *once per value* in that option. Used by `[JONES-TEMPLATE]`
to spawn `[g]`, `[de]`, etc. dynamically.

### 14.2 `dynoptparse.py` — Parset → optparse adapter (277 LOC)

`DynamicOptionParser` builds an `optparse.OptionParser` on the fly from a
Parset, creating one `--section-option` for every entry. It also writes the
resolved option set back as a parset (`write_to_parset`) for run provenance.

### 14.3 `shared_dict.py` — `/dev/shm` dict (315 LOC)

A `SharedDict` is a dict-like object backed by files in `/dev/shm/<name>/`,
one file per top-level value. Pickled containers, `numpy` memory-mapped
arrays, sub-dicts. Used by `MSTile` to share loaded MS visibility cubes
between the I/O worker and the compute workers without copying.

### 14.4 `NpShared.py` — Numpy on `/dev/shm` (302 LOC)

A lower-level layer (older, separate from `shared_dict`) that wraps the
`SharedArray` PyPI package: `CreateShared(name, shape, dtype)`, `ToShared(name, A)`,
`GiveArray(name)`, `DelArray(name)`, etc. Also `PackListArray` / `UnPackListArray`
for storing variable-length per-baseline lists.

### 14.5 `shm_utils.py` — SHM cleanup (79 LOC)

`cleanupStaleShm()` is called twice from `main.py` (lines 286, 304) to remove
any `/dev/shm/` files left over from killed runs.

### 14.6 `logger.py` — multiprocess-safe logging (424 LOC)

A custom logger built on top of Python's `logging` that:

* Supports per-subprocess labels (`set_subprocess_label`) — so log lines
  are tagged with the worker that produced them.
* Honours per-subsystem verbosity via `--log-verbose name1=N,name2=M`.
* Optionally records memory stats per log line (`--log-memory`).
* Routes to a per-run logfile (`<basename>.log`) and the console.
* `print(..., file=log)` and `file=log(level, color)` are the public idioms
  used throughout the codebase.

### 14.7 `ModColor.py`, `ClassPrint.py`, `dtype_checks.py`

* `ModColor` — ANSI colour shortcuts (`Str("text", "red")`, etc.).
* `ClassPrint` — pretty-prints option dicts in coloured tabular form (used by
  `DynamicOptionParser.print_config`).
* `dtype_checks` — `is_complex(dtype)`, `is_complex64(dtype)`, etc.

---

## 15. Stimela integration (`cubical/stimela/`)

`generate_schema.py` (70 LOC) walks the `DefaultParset.cfg` schema and writes
out a YAML cab definition consumable by **Stimela** (the Stimela radio-pipeline
DSL). This lets external pipelines call CubiCal as a stimela "cab" rather than
hand-rolling the CLI.

---

## 16. Tests (`test/`)

CubiCal does **not** ship a `pytest`-discoverable unit test suite. Instead, it
ships a **vendored CASA Measurement Set** (`SUBSET-D147.MS`) and a regression
driver that validates outputs against a tarred reference output
(`SUBSET-D147-output.MS.tgz`).

### 16.1 `test/d147_test.py` (146 LOC)

`SolverVerification` class drives `gocubical` against the test MS for several
calibration scenarios (`d147_test_list` at line 63):

| Output column | Configuration |
| ------------- | ------------- |
| `GSOL_DATA`   | Default (G complex-2x2) |
| `GSOL_DATA`   | Default with `dist_ncpu=1` (serial-mode regression) |
| `GBSOL_DATA`  | G+B chain: G time-only, B freq-only |
| `PO_DATA`     | `g_type='phase-diag'` |
| `FS_DATA`     | `g_type='f-slope'` (delay) |
| `TS_DATA`     | `g_type='t-slope'` (rate) |
| `TFP_DATA`    | `g_type='tf-plane'` (delay+rate) |
| `DE_DATA`     | DDE: `--sol-jones G,dE` with the bundled `3C147-dE-apparent.lsm.html` LSM |

`generate_reference(...)` runs CubiCal on the reference MS to produce the
expected outputs. `verify(...)` re-runs CubiCal on the test MS and compares
the visibility difference against the reference using the per-element
median and 95th-percentile thresholds:

```
diff = abs(CORRECTED_DATA_test - refcol)
diff_db = 10*log10(diff / abs(refcol))
assert median(diff_db) < median_tolerance       # default -30 dB
assert percentile(diff_db, 95) < ninetyfifth_tolerance  # default -25 dB
```

A failure means CubiCal output drifted >1e-3 (median) or >5e-3 (95-th)
relative to the reference.

### 16.2 `test/test_geometry.py`

Smaller test for `cubical/degridder/geometry.py` (facet projection / FOV /
celestial geometry helpers).

### 16.3 `Jenkinsfile.sh` and `.jenkins/`

CI definition. The Jenkins job calls `pip install -e .` and then runs
`d147_test.py` against the bundled MS.

---

## 17. Extension Points

### 17.1 Adding a new gain machine

The procedure (paraphrased from the abstract base in `abstract_machine.py`):

1. Subclass `MasterMachine` (for a leaf machine) or `PerIntervalGains` (for
   a constant-per-interval machine). For parameterised solvers, extend
   `ParameterisedGains` (`parameterised_machine.py`).
2. Implement the abstract methods: `compute_js`, `implement_update`,
   `compute_residual`, `apply_inv_gains`, `apply_gains`, `check_convergence`,
   `flag_solutions`, `num_gain_flags`, `restrict_solution`,
   `has_converged`, `has_stalled`, `conditioning_status_string`,
   `current_convergence_status_string`, `export_solutions`,
   `import_solutions`.
3. Implement two static class methods:
   * `determine_allocators(options)` — return `(allocate_vis, allocate_flag,
     allocate_gain)` triple, typically reusing one of the kernel modules'
     allocators.
   * `determine_diagonality(options)` — `True` if the term is diagonal.
4. Implement the static `exportable_solutions()` returning a dict of
   parmdb parameter shapes.
5. Optionally write a corresponding kernel module under `cubical/kernels/`
   that mirrors the layout of `full_complex.py` (provide `compute_residual`,
   `compute_jh`, `compute_jhr`, `compute_jhj`, `compute_update`,
   `apply_gains`, `compute_corrected`, `right_multiply_gains`).
6. Register the type string by adding an entry to the `GAIN_MACHINE_TYPES`
   dict in `cubical/machines/machine_types.py`.

The new machine is now selectable via `--<jones>-type <new-name>`.

### 17.2 Adding a new model back-end

* Define a class implementing the same interface as
  `TiggerSourceProvider` / `DicoSourceProvider`: per-direction iteration,
  per-channel/per-time predict.
* Hook the parser branch in `MSDataHandler.init_models` (`ms_data_handler.py:641`)
  so a new file extension or special string maps to the new provider.

### 17.3 Adding a new flag bit

* Add a new bit to `cubical/flagging.py` (`FL` namespace).
* Register an associated flagset name if you want `--flags-apply` /
  `--flags-save` to recognise it.
* Raise it from your gain-machine `flag_solutions` method or your kernel.

### 17.4 Adding a new solver mode

* Add a subclass of `SolverMachine` (in `cubical/solver.py`) that returns the
  desired `corr_vis` array from its `run()` method.
* Register the short code in the `SOLVERS` dict at `solver.py:845`.
* Update `DefaultParset.cfg` `out.mode` `#options:` to include the new code.

---

## 18. Notable Internals

### 18.1 Chunking & tiling

* The MS is partitioned by `(DDID, time-chunk)` into **`RowChunk`s**. Each
  chunk is the unit of one gain solution.
* `RowChunk`s are bundled into **`MSTile`s**, which are the unit of I/O. A
  tile is loaded into shared memory in one shot, then handed to the compute
  workers via `solver.run_solver(itile, chunk_key)`.
* The number of chunks per tile is `max(--dist-min-chunks, nworkers, 1)`,
  capped by `--dist-max-chunks` if set.
* `--data-time-chunk` and `--data-freq-chunk` accept either an integer
  (timeslots / channels) or a unit-suffixed value (e.g. `300s`, `128MHz`).
* `--data-rebin-time` and `--data-rebin-freq` enable on-the-fly averaging
  via `kernels/rebinning.py` (rebin happens *during* `tile.load()`, so the
  solver only sees the rebinned data).

### 18.2 Multiprocessing model

`cubical/workers.py`:

* One **main process** (does config, parmdb finalisation, plots).
* One **I/O process** (`_io_handler`) that owns the casacore table handle
  and is the only process that ever opens the MS — workers see only the
  shared-memory cube. This avoids the multi-thread problems with casacore
  table file handles.
* `N` **compute workers** (`solver.run_solver`) launched via
  `concurrent.futures.ProcessPoolExecutor`.

The main process schedules each tile's I/O job to overlap with the previous
tile's solver work. From `_run_multi_process_loop` (`workers.py:248-351`):

```
io_executor.submit(_io_handler, load=0)           # load tile 0
for itile, tile in enumerate(tile_list):
    wait for tile itile to finish loading
    io_executor.submit(_io_handler, load=itile+1, save=itile-1)  # next/previous
    for key in tile.get_chunk_keys():
        executor.submit(solver.run_solver, ..., key, ...)
    wait for all chunks of tile itile to finish
io_executor.submit(_io_handler, save=-1, finalize=True)
```

`reap_children()` polls `os.wait3(WNOHANG)` to detect dead worker processes
(usually OOM kills) and aborts the run cleanly.

### 18.3 CPU affinity & thread placement (`workers.setup_parallelism`)

* `--dist-pin N` enables affinity, starting at core `N`.
* `--dist-pin N:K` starts at core `N` with stride `K` (useful for hyper-threaded
  systems where you want one OMP thread per physical core).
* `--dist-pin-io` allocates a separate core to the I/O worker (and
  Montblanc, if active), pinned via `GOMP_CPU_AFFINITY`.
* `--dist-pin-main` allocates a separate core to the main process; if set
  to `io`, shares with the I/O worker.
* When `--dist-nthread > 1`, Numba's threading layer is set to `safe`
  (TBB) if available, else falls back to `workqueue`.
* `numba.set_num_threads(nthread)` sets the per-worker OMP thread cap.

### 18.4 Memory wisdom (`data_handler/wisdom.py`)

`estimate_mem(ms, tile_list, data_opts, dist_opts)` computes a heuristic
memory budget by summing the bytes per tile (data + model + flags + weights +
gains) and multiplying by the tile concurrency. If it exceeds
`--dist-safe × psutil.virtual_memory().total`, the run aborts before
spawning workers.

### 18.5 Reference-antenna handling (`PerIntervalGains.restrict_solution`)

`--JONES-ref-ant <name>` sets a per-Jones reference antenna. After every
update, `restrict_solution(gains)` zeros the phase of the reference antenna
on each interval (or, for diagonal gains, divides every antenna by the
reference's gain). This breaks the gauge degeneracy of the GN solution
(gains can be globally rotated by a constant phase without changing the
predicted visibilities).

### 18.6 `subset` selection inside the solver (`--sol-subset`)

Distinct from `--sel-taql` (which restricts what is read from the MS),
`--sol-subset` takes a TaQL query that is applied *only* during solving:
matched rows still go into the cube, but they are masked out via
`FL.SKIPSOL` for the duration of the solver loop. This lets you e.g. solve
on calibrators only while still applying the resulting solutions to the
target field rows.

### 18.7 Single-chunk and single-tile debug modes

* `--data-single-chunk D0T0F0` processes only the named chunk (label format
  is `D<ddid>T<tchunk>F<fchunk>`).
* `--data-single-tile N` processes only the Nth tile.
* Both force serial execution (no worker pool).
* `--debug-pdb 1` and `--debug-stop-before-solver 1` arm Python's `pdb`
  to fire on exception or on entry to the solver, respectively.

---

## 19. Known limitations and TODOs

Pulled directly from in-source comments and from the parset schema:

* **Python 3 caltable export disabled.** `casa_db_adaptor.py:44-46`:
  > "Gaintables cannot be written in Python 3 mode due to current casacore
  > implementation issues" — `--out-casa-gaintables 1` is silently a no-op
  > on Py3.
* **Single-correlation MS support is "not fully tested"** —
  `ms_data_handler.py:268-271`. CubiCal was reinstated for single-corr MSes
  in PR #449 but warns at runtime.
* **`AntennaType` and most non-Gaussian beam types are gone** from the
  associated RIME-side code — only FITS-beam interpolation (via DDFacet's
  ClassFITSBeam, see PR #461 in `git log`) and the legacy LOFAR-style
  beam patterns are supported.
* **"--data-chunk-on-scans" is deprecated** (replaced by `--data-chunk-by`).
  References in `test/d147-test.parset` still use the old name.
* **Slope solver "different slope modes" experimental change reverted** — see
  commit `ea333c7 Revert "Added different slope modes for slope solvers..."`.
* **Numba TBB backend may not be installable** — see `_setup_workers_and_threads`
  fallback at `workers.py:118-130`. When TBB is missing, Numba falls back to
  `workqueue` and CubiCal forces 1 thread per worker.
* **Pre-mortem/post-mortem flagging is "experimental"** —
  `DefaultParset.cfg:228-229`: "NB: EXPERIMENTAL. USE AT OWN RISK."
* **`--bbc-apply-2x2 1` warning** — DefaultParset.cfg:319: "Only enable this if
  you really trust the polarisation information in your sky model."
* **`--sol-delta-g` and `--sol-delta-chi` are deprecated** in favour of the
  per-Jones `--JONES-epsilon` and `--JONES-delta-chi`.
* **Random `eval()` of strings**. `--out-subtract-dirs ":,1,3"` etc. is parsed
  with `eval("np.s_[{}]".format(subdirs))` (`main.py:479`). This works but is
  obviously not safe to feed untrusted parsets.
* **Old-style `print` and `from __future__ import print_function`** are still
  pervasive — leftover Py2 compatibility despite the `python_requires=">=3.6"`.
* **Direction-dependent BBCs are not supported.** `IfrGainMachine` collapses
  over directions before the per-baseline accumulation.
* **`--montblanc-device-type GPU`** requires a working TensorFlow-GPU /
  CUDA stack visible to Montblanc; CubiCal does not negotiate this for you
  beyond setting the `OMP_NUM_THREADS` and `GOMP_CPU_AFFINITY` env vars.

---

## 20. Quick-start cheat sheet

```bash
# 1. Generate model visibilities into MODEL_DATA in your MS (e.g. via WSClean
#    'wsclean -predict ...', or via Tigger LSM through cubical itself).

# 2. Write a parset, e.g. cal.parset:
cat > cal.parset <<'EOF'
[data]
ms = my.MS
column = DATA
time-chunk = 32
freq-chunk = 32

[model]
list = MODEL_DATA

[sol]
jones = G
precision = 64

[g]
time-int = 8
freq-int = 8
type = complex-2x2
ref-ant = m000

[out]
column = CORRECTED_DATA
mode = sc            # solve and write corrected data
plots = 1
EOF

# 3. Run:
gocubical cal.parset --dist-ncpu 16

# 4. Outputs in cubical.cc-out/:
#    cc.parset                              — fully-resolved parset (provenance)
#    cc.log                                  — full log
#    cc-G-field_0-ddid_None.parmdb           — gain solutions (pickled)
#    cc-G-field_0-ddid_None.parmdb.skel      — solution-grid metadata
#    cc-G-field_0-ddid_None                  — CASA caltable mirror
#    cc-BBC-field_0-ddid_None.parmdb         — IFR-gain (BBC) solutions
#    cc.stats.pickle                         — SolverStats record
#    cc.G.png, cc.bandpass.png, cc.ifrgain.png, ...  — summary plots
```

For a chained delay+gain solve with DDE on tagged sources:

```bash
gocubical cal.parset \
    --sol-jones K,G,dE \
    --k-type f-slope --k-time-int 0 --k-freq-int 0 \
    --g-type complex-2x2 --g-time-int 8 --g-freq-int 8 \
    --de-type complex-2x2 --de-time-int 60 --de-freq-int 32 --de-dd-term 1 \
    --model-list "skymodel.lsm.html@dE" \
    --model-ddes always \
    --sol-term-iters 20,50,30,20,30 \
    --out-mode sc
```

---

## 21. Provenance audit

Every claim in this document is grounded in one of the following files (all
paths are relative to `simulators/CubiCal/`):

* `__init__.py`, `setup.py`, `pyproject.toml`, `LICENSE.md`, `README.md`
* `cubical/main.py` (driver, parset/CLI logic, mode dispatch)
* `cubical/solver.py` (solver loop, SolverMachine family, SOLVERS dict)
* `cubical/workers.py` (multiprocessing & affinity)
* `cubical/DefaultParset.cfg` (every option in §5)
* `cubical/flagging.py` (FL namespace, Flagsets, post-mortem flagging)
* `cubical/statistics.py` (SolverStats schema)
* `cubical/madmax/flagger.py` (MAD-Max flagger)
* `cubical/data_handler/ms_data_handler.py` (MSDataHandler, init_models)
* `cubical/data_handler/ms_tile.py` (MSTile, RowChunk)
* `cubical/data_handler/MBTiggerSim.py`, `TiggerSourceProvider.py` (Montblanc bridge)
* `cubical/data_handler/wisdom.py` (memory estimator)
* `cubical/machines/abstract_machine.py` (MasterMachine + Factory)
* `cubical/machines/machine_types.py` (GAIN_MACHINE_TYPES)
* `cubical/machines/slope_machine.py` (SLOPE_TYPES, SLOPE_TYPE_ALIASES)
* `cubical/machines/complex_2x2_machine.py`, `complex_W_2x2_machine.py`,
  `phase_diag_machine.py`, `pol_gain_machine.py`,
  `interval_gain_machine.py`, `parameterised_machine.py`,
  `jones_chain_machine.py`, `jones_chain_robust_machine.py`,
  `parallactic_machine.py`, `ifr_gain_machine.py`
* `cubical/kernels/__init__.py`, `full_complex.py`, `full_W_complex.py`,
  `diag_complex.py`, `diagdiag_complex.py`, `phase_only.py`,
  `diag_phase_only.py`, `diag_robust.py`, `f_slope.py`, `t_slope.py`,
  `tf_plane.py`, `ff2_slope.py`, `chain.py`, `madmax.py`, `rebinning.py`,
  `generics.py`
* `cubical/database/iface_database.py`, `pickled_db.py`,
  `casa_db_adaptor.py`, `parameter.py`
* `cubical/param_db.py` (thin wrapper)
* `cubical/plots/__init__.py`, `gainsols.py`, `ifrgains.py`,
  `leakages.py`, `stats.py`
* `cubical/madmax/plots.py`
* `cubical/degridder/DDFacetSim.py`, `DicoSourceProvider.py`,
  `FITSBeamInterpolator.py`, `geometry.py`
* `cubical/stimela/generate_schema.py`
* `cubical/tools/parsets.py`, `dynoptparse.py`, `logger.py`,
  `ModColor.py`, `ClassPrint.py`, `dtype_checks.py`,
  `shared_dict.py`, `NpShared.py`, `shm_utils.py`
* `cubical/bin/gocubical`, `print-cubical-stats`, `plot-gain-solutions`,
  `plot-leakage-solutions`
* `docs/index.rst`, `introduction.rst`, `installation.rst`, `usage.rst`,
  `parset.rst`, `examples.rst`, `performance.rst`, `licence.rst`,
  `cubical.rst`
* `test/d147_test.py`, `d147-test.parset`, `3C147-dE-apparent.lsm.html`
* `Jenkinsfile.sh`, `MANIFEST.in`, `rtd_requirements.txt`
* Git output: `git -C simulators/CubiCal log --oneline -20` and
  `git -C simulators/CubiCal tag` (HEAD: `5686a1d "Noxcal prepare release"`,
  latest tag: `v1.6.4`).

