# killMS — Exhaustive Technical Reference

> Source tree audited: `simulators/killMS/` (git submodule). All claims below are
> grounded in the files cited; no behaviour has been inferred beyond what the code
> states. Where source comments label something "to be published" or unfinished,
> the same caveat is reproduced here.

## 1. Overview

**killMS** is a **direction-dependent (third-generation) radio-interferometric
calibration package** that solves for per-direction Jones matrices on a
Measurement Set, using two original complex-optimisation algorithms based on
**Wirtinger Jacobians**:

| Algorithm | Class of solver | File | Acronym in code |
|-----------|------------------|------|-----------------|
| **CohJones** | Levenberg–Marquardt with Wirtinger Jacobian | `simulators/killMS/killMS/Wirtinger/ClassSolverLM.py` | `LM` / `CohJones` |
| **KAFCA** | Extended Kalman Filter on the half-Jacobian | `simulators/killMS/killMS/Wirtinger/ClassSolverEKF.py` | `EKF` / `KAFCA` |

The name *killMS* originates from early LOFAR commissioning ("understanding the
data inside a Measurement Set was a real challenge"; quoted from
`simulators/killMS/README.md` line 22). It is the canonical companion of
**DDFacet** — DDFacet does the imaging/deconvolution and writes a
clustered/faceted sky model, killMS solves for direction-dependent Jones
matrices in those facets, and DDFacet then applies them on a second imaging
pass (`simulators/killMS/README.md` lines 84–112).

**Mathematical references** (printed verbatim in the README):

- Tasse 2014 "Applying Wirtinger derivatives to the radio interferometric
  calibration problem" — <https://arxiv.org/abs/1410.8706>
- Smirnov & Tasse 2015 "Radio interferometric gain calibration as a complex
  optimization problem" — <https://arxiv.org/abs/1502.06974>
- Extended Kalman filter using the Wirtinger half-Jacobian — *to be published*;
  see `simulators/killMS/README.md` line 36 and arXiv:1403.6308 for a similar
  approach.

### 1.1 Identity

| Field | Value | Source |
|---|---|---|
| Distribution name | `killMS` | `simulators/killMS/pyproject.toml:12` |
| Version | `3.3.0` | `simulators/killMS/pyproject.toml:13` |
| Tagline | "A Wirtinger-based direction-dependent radio interferometric calibration package" | `pyproject.toml:14` |
| Authors | Cyril Tasse (cyril.tasse@obspm.fr) | `pyproject.toml:22` |
| Maintainer | Benjamin Hugo (bhugo@sarao.ac.za) | `pyproject.toml:23` |
| Copyright | 2013–2021 Cyril Tasse, l'Observatoire de Paris, SKA South Africa, Rhodes University | `README.md:5–6` |
| License | **GPL v2 or later** | `LICENSE.md`, every source-file header (e.g. `kMS.py:6–19`) |
| Homepage | <http://github.com/saopicc/killMS> | `pyproject.toml:24` |
| Python | `>=3.8,<3.13` | `pyproject.toml:17` |
| Status | "Production/Stable" classifier | `pyproject.toml:26` |
| OS support | "POSIX :: Linux" classifier | `pyproject.toml:30` |
| Latest tag | `v3.3.0`; preceding tags `V3.2.1`, `v3.2.2`, `v3.2.0`, `v3.1.0`, `v3.0.x`, `v2.6` | `git tag` |
| Most recent merge | `350f22c "Merge pull request #104 ... prepare312"`, on master | `git log --oneline -20` |
| Notable recent work | NumPy 2.x compatibility (`ee51c3f`), Python 3.12 prep, dtype fixes (`15636c7`, `9b8bdff`, `e3bd122`) | git log |

### 1.2 Languages & build system

| Language | Where | Notes |
|---|---|---|
| Python 3 | `simulators/killMS/killMS/**/*.py` (≈ 16,000 LOC across 38 modules in the top tree) | bulk of the package |
| C99 (CPython extension) | `Predict/predict.c`, `Predict/predict.h`, `Gridder/Gridder.c`, `Gridder/Gridder.h`, `Array/Dot/dot0.c`, `Array/Dot/dotSSE.c`, `Array/Dot/dotSSE.h` | OpenMP-parallel; -ffast-math by default |
| CMake | `killMS/CMakeLists.txt`, `killMS/Predict/CMakeLists.txt`, `killMS/Gridder/CMakeLists.txt`, `killMS/Array/CMakeLists.txt`, `killMS/Array/Dot/CMakeLists.txt`, `killMS/cmake/Find*.cmake` | drives the C extension build |
| YAML | `killMS/Parset/killms_stimela_schema.yaml`, `killMS/Parset/test_recipe.yaml` | Stimela schema for the CLI |
| INI / .cfg | `killMS/Parset/DefaultParset.cfg` | parameter defaults |
| Dockerfile | `simulators/killMS/Dockerfile` | container build on top of `bhugo/ddfacet:0.9.0.0` |
| Jenkins | `simulators/killMS/Jenkinsfile.sh` | CI driver |

The build system is **scikit-build-core** (`pyproject.toml:9`) wrapping CMake.
Default flags from `pyproject.toml:60`:
`ENABLE_NATIVE_TUNING=OFF`, `ENABLE_FAST_MATH=ON`, `ENABLE_PYTHON_3=ON`,
build-type `ReleaseWithDebugSymbols`. CMake's default tune flags add
`-march=native -mtune=native` *only* if `ENABLE_NATIVE_TUNING=ON`
(`CMakeLists.txt:37–42`); release CXXFLAGS always include `-fopenmp -std=c++14
-ggdb3 -fmax-errors=1 -pedantic -W -Wall -Wconversion` and (when `ENABLE_FAST_MATH=ON`)
`-ffast-math`. CMake required modules (`killMS/cmake/Find*.cmake`):
`PythonInterp`, `PythonLibs`, `NumPy`, `OpenMP`, `RT`, optional `CasaCore`,
`CfitsIO`, `WcsLib`, `pybind11`. CMake walks subdirectories `Predict`,
`Gridder`, `Array` (`CMakeLists.txt:68–70`).

The C extension that is *actually* used at runtime is the predict shared
library `predict3x.so`, imported at `Predict/PredictGaussPoints_NumExpr5.py:46`
as `from killMS.cbuild.Predict import predict3x as predict`. The Gridder
extension is built but is mostly used by neighbouring DDFacet code paths.

## 2. Repository layout

```
simulators/killMS/
├── .dockerignore               (7 B)
├── .git                        (45 B — submodule pointer)
├── .gitignore
├── Dockerfile                  Image build atop bhugo/ddfacet:0.9.0.0
├── Jenkinsfile.sh              Jenkins CI: docker build + nosetests TestHarness
├── LICENSE.md                  GPL-2.0-or-later (full text)
├── README.md                   short user-facing intro + DD-cal example
├── pyproject.toml              scikit-build-core / wheel / sdist config
├── killMS/                     Python package "killMS"
│   ├── __init__.py             sets __version__ from DDFacet metadata; sets KILLMS_DIR env var
│   ├── __main__.py             Per-CLI entry-point shims (kms_main, smoothsols_main, ...)
│   ├── CMakeLists.txt          Top-level CMake: walks Predict/Gridder/Array
│   ├── kMS.py                  THE main calibration driver (1354 LOC)
│   ├── AQWeight.py             Imaging/cov-based weight estimator
│   ├── BLCal.py                Per-baseline calibration helper
│   ├── ClipCal.py              Residual clipping/flagging tool
│   ├── ClassImageSM2.py        (under Predict/) image-based predict from a DDF DicoModel
│   ├── ClipCal.py              Residual clipping
│   ├── dsc.py                  "DDF/SkyModel/Cluster" small util
│   ├── grepall.py              greps over many MS solution dirs
│   ├── InterpSols.py           extrapolate solutions in frequency
│   ├── MakePlotMovie.py        builds an mp4 from per-time PlotSols snapshots
│   ├── MergeSols.py            merge .sols.npz files (in frequency)
│   ├── PlotSols.py             matplotlib plot of solutions vs time/freq
│   ├── PlotSolsIm.py           imshow plot of |G|, arg(G)
│   ├── SmoothSols.py           TEC + Amp smoothing of solutions (uses ClassFitTEC, ClassFitAmp, ClassClip)
│   ├── TestSpeedNp.py          numpy timing micro-benchmark
│   │
│   ├── Array/
│   │   ├── CMakeLists.txt
│   │   ├── ModLinAlg.py        Cholesky/LU/SVD inverses, batched 2x2 ops, BatchInverse, BatchH, BatchDot, sqrtSVD
│   │   ├── NpShared.py         /dev/shm-backed shared NumPy arrays via the SharedArray module (ToShared/GiveArray/DelArray)
│   │   ├── RecArrayOps.py      structured-array helpers
│   │   └── Dot/
│   │       ├── CMakeLists.txt, Makefile
│   │       ├── dot0.c, dotSSE.c, dotSSE.h    SSE dot products
│   │       ├── NpDotSSE.py     Python wrapper for dotSSE
│   │       ├── NpCuda.py       (skeleton CUDA dot — non-default)
│   │       └── TestDotSSE.py
│   │
│   ├── cmake/                  Find modules (CasaCore, CfitsIO, NumPy, pybind11, RT, WcsLib)
│   │
│   ├── Data/
│   │   ├── ClassMS.py          MS reader/writer (1277 LOC), wraps casacore tables via pyrap
│   │   ├── ClassVisServer.py   Streams visibility chunks; calls weighting/clustering (1312 LOC)
│   │   ├── ClassWeighting.py   Briggs/Natural/Uniform weights
│   │   ├── ClassBeam.py        Mean LOFAR/FITS beam estimator (used for KAFCA priors)
│   │   ├── ClassJonesDomains.py  domain matching for pre-applied Jones (time/freq)
│   │   ├── ClassReCluster.py
│   │   └── sidereal.py         Self-contained sidereal-time math (1480 LOC; standalone)
│   │
│   ├── Gridder/
│   │   ├── Gridder.c, Gridder.h, Makefile, CMakeLists.txt
│   │     Methods: pyGridderWPol / pyDeGridderWPol / pyGridderPoints / pyTestMatrix
│   │     / pyAddArray / pyWhereMax  (Gridder.c:53–58)
│   │
│   ├── Other/                  Heterogeneous utilities
│   │   ├── ClassClip.py            outlier clipping helper
│   │   ├── ClassFitAmp.py          smoothed amplitude fit (used by SmoothSols)
│   │   ├── ClassFitTEC.py          TEC + ConstPhase model: phase = K·TEC/freq, K=8.4479745e9
│   │   ├── ClassPrint.py
│   │   ├── ClassTimeIt.py          ad-hoc profiler used everywhere
│   │   ├── Counter.py
│   │   ├── findrms.py              robust RMS via MAD-style estimator
│   │   ├── least_squares.py        Vendored copy of scipy.optimize.least_squares
│   │   ├── logo.py                 ASCII banner / version reporter
│   │   ├── ModChanEquidistant.py   detect equidistant frequency grids
│   │   ├── ModColor.py             ANSI colour helpers
│   │   ├── ModParsetType.py
│   │   ├── MyPickle.py             trivial pickle.dump/load wrapper
│   │   ├── PrintOptParse.py        pretty-print resolved options
│   │   ├── rad2hmsdms.py           coord formatting
│   │   └── reformat.py             path-canonicalisation
│   │
│   ├── Parset/
│   │   ├── DefaultParset.cfg       Single source of truth for option defaults
│   │   ├── ReadCFG.py              ConfigParser wrapper (FormatValue type-coerces)
│   │   ├── MyOptParse.py           OptionParser layer on top of the parset
│   │   ├── PrintOptParse.py
│   │   ├── ClassPrint.py
│   │   ├── killms_stimela_schema.yaml  Generated schema for Stimela pipelines
│   │   └── test_recipe.yaml
│   │
│   ├── Plot/                   Diagnostic matplotlib helpers
│   │   ├── GiveNXNYPanels.py
│   │   ├── GiveRectSubplot.py
│   │   └── Graph.py
│   │
│   ├── Predict/                Sky-to-visibility evaluation
│   │   ├── predict.c, predict.h    C ext: predictJones2_Gauss + ApplyJones (predict.c:39–43)
│   │   ├── PredictGaussPoints_NumExpr5.py  ClassPredict + ClassPredictParallel (976 LOC, current)
│   │   ├── PredictGaussPoints_NumExpr.py   Older variant kept around for SmoothSols imports
│   │   ├── ClassImageSM2.py        Predict from a DDFacet DicoModel (image-based skymodel)
│   │   └── CMakeLists.txt
│   │
│   ├── Simul/                  Synthetic data generation
│   │   ├── DoSimul.py
│   │   ├── MakeClusterCat.py
│   │   └── MakeModelImage.py
│   │
│   ├── Weights/                Reweighting machinery (used by AQWeight)
│   │   ├── W_AntFull.py  W_DiagBL.py  W_Imag.py  W_ImagCov.py  W_TimeCov.py
│   │
│   └── Wirtinger/              The actual solvers
│       ├── ClassWirtingerSolver.py     Master driver (1542 LOC) — wires VS/SM/predict/jacobian
│       ├── ClassJacobianAntenna.py     Per-antenna Jacobian / kernel cache (1372 LOC)
│       ├── ClassSolverLM.py            CohJones (Levenberg–Marquardt)
│       ├── ClassSolverEKF.py           KAFCA (Extended Kalman Filter)
│       ├── ClassEvolve.py              Time-evolution of P (process noise + Q)
│       ├── ClassAverageMachine.py      Visibility compression / station merging
│       └── ClassSolPredictMachine.py   Prior solutions for KAFCA via EvolutionSolFile
│
└── TestHarness/                pytest/nosetests wrapper for long acceptance tests
    ├── __init__.py
    └── LongAcceptanceTests/
        ├── __init__.py
        └── TestLOFAR_J1329_p4729.py    LOFAR end-to-end test using DDFacet's
                                         ClassCompareFITSImage harness
```

## 3. Installation

### 3.1 From PyPI / source (canonical)

`simulators/killMS/README.md:42–52`:

```bash
virtualenv myvenv
source myvenv/bin/activate
(myvenv)$ pip install DDFacet           # MUST match killMS upper bound
(myvenv)$ pip install <path-to-killMS>  # source build (scikit-build-core + cmake)
# or, in development mode:
(myvenv)$ pip install -e <path-to-killMS>
```

### 3.2 Hard runtime dependency: DDFacet

`simulators/killMS/pyproject.toml:18–20`:

```toml
dependencies = [
    "DDFacet[kms-support] >= 0.7.0; python_version >= '3'",
]
```

DDFacet is **not optional**. Many modules import from it:
`from DDFacet.Other import logger, ModColor` (`kMS.py:49`),
`from DDFacet.Data import ClassVisServer as ClassVisServer_DDF` (`kMS.py:339`),
`from DDFacet.Other import AsyncProcessPool` (`kMS.py:554`),
`from DDFacet.Imager import ClassDDEGridMachine` (`Predict/PredictGaussPoints_NumExpr5.py:54`),
`from DDFacet.Imager.ClassDeconvMachine import ClassImagerDeconv`
(`Predict/ClassImageSM2.py:25`), and many more. The `__version__` string of
killMS is even pulled from DDFacet's installed metadata
(`killMS/__init__.py:24–33`):

```python
__version__ = version("DDFacet")    # falls back to "dev"
```

The `[kms-support]` extra of DDFacet is the marker that the SkyModel + DDF
predict-side code paths killMS leans on are installed.

### 3.3 Build-time requirements

`pyproject.toml:2–8`:

```toml
[build-system]
requires = [
    "numpy >= 1.15.1, <= 2.3.2; python_version >= '3.11' and python_version < '3.13'",
    "numpy >= 1.15.1, <= 1.22.4; python_version >= '3.8' and python_version < '3.11'",
    "pybind11 >= 2.2.2",
    "cython<=0.29.30",
    "cmake",
    "scikit-build-core"]
build-backend = "scikit_build_core.build"
```

System packages (implied by the CMake Find modules):

- `libcasacore-dev` (`Findcasacore.cmake`)
- `libcfitsio-dev` (`FindCfitsIO.cmake`)
- `libwcs-dev` (`FindWcsLib.cmake`)
- `librt` (POSIX realtime; `FindRT.cmake`, mandatory)
- `libomp` / OpenMP-capable compiler
- `pybind11` headers
- For the runtime SHM layer: the **`SharedArray`** PyPI module (imported at
  `killMS/Array/NpShared.py:25`).

### 3.4 Docker

`simulators/killMS/Dockerfile`:

```dockerfile
FROM bhugo/ddfacet:0.9.0.0
ADD killMS /opt/killMS/killMS
ADD pyproject.toml /opt/killMS/pyproject.toml
ADD README.md /opt/killMS/README.md
ADD LICENSE.md /opt/killMS/LICENSE.md
ADD .git /opt/killMS/.git
ADD .gitignore /opt/killMS/.gitignore
WORKDIR /opt
RUN . /opt/venv/bin/activate && python3 -m pip install ./killMS
RUN DDF.py --help
RUN MakeMask.py --help
RUN MakeCatalog.py --help
RUN MakeModel.py --help
RUN MaskDicoModel.py --help
RUN ClusterCat.py --help
RUN kMS.py --help
RUN pybdsf --version
ENTRYPOINT ["kMS.py"]
CMD ["--help"]
```

Jenkins runs the docker image with **150 GB** shared memory (`--shm-size=150g`,
`Jenkinsfile.sh:25`) and `OPENBLAS_NUM_THREADS=1`, then invokes `nosetests
TestHarness`.

### 3.5 Console entry points

`pyproject.toml:35–46` registers eleven scripts, all routed through
`killMS/__main__.py`:

| Console name | Function | Module driver |
|---|---|---|
| `kMS.py` | `kms_main` | `killMS.kMS.driver` |
| `AQWeight.py` | `aqweight_main` | `killMS.AQWeight.driver` |
| `ClipCal.py` | `clipcal_main` | `killMS.ClipCal.driver` |
| `dsc.py` | `dsc_main` | `killMS.dsc.driver` |
| `grepall.py` | `grepall_main` | `killMS.grepall.driver` |
| `InterpSols.py` | `interpsols_main` | `killMS.InterpSols.driver` |
| `MakePlotMovie.py` | `makeplotmovie_main` | `killMS.MakePlotMovie.driver` |
| `MergeSols.py` | `mergesols_main` | `killMS.MergeSols.driver` |
| `PlotSols.py` | `plotsols_main` | `killMS.PlotSols.driver` |
| `PlotSolsIm.py` | `plotsolsim_main` | `killMS.PlotSolsIm.driver` |
| `SmoothSols.py` | `smoothsols_main` | `killMS.SmoothSols.driver` |

(`BLCal.py` has a `blcal_main` shim in `__main__.py:42–43` but is not exposed in
`pyproject.toml`.)

## 4. Architecture diagram

```
                 ┌──────────── kMS.py (CLI driver) ──────────────┐
                 │  read_options() →  Parset + MyOptParse        │
                 │  main()                                       │
                 └────────────┬────────────┬─────────────────────┘
                              │            │
              Predict mode    │            │  Solver pick
   (Catalog / Image / Column) │            │  (CohJones | KAFCA)
                              ▼            ▼
   ┌─────────────────────────────┐   ┌──────────────────────────────────┐
   │ Predict/ClassImageSM2       │   │ Wirtinger/ClassWirtingerSolver  │
   │   Predict from DicoModel    │   │   master per-chunk solver loop   │
   │   (DDFacet FacetMachine)    │   └──────────────┬──────────────────┘
   └────────────┬────────────────┘                  │
                │                                   │ for each Jones-chan, antenna
                │                                   ▼
                │                  ┌──────────────────────────────────────┐
                │                  │ Wirtinger/ClassJacobianAntenna       │
                │                  │   K-matrix (kernel) per-direction    │
                │                  │   Wirtinger half-Jacobian assembly   │
                │                  │   J_x, JH_z, JHJinv_x, Msq_x         │
                │                  └─────────┬──────────────────┬────────┘
                │                            │                  │
                │                  ┌─────────▼──────┐   ┌──────▼──────────┐
                │                  │ ClassSolverLM  │   │ ClassSolverEKF  │
                │                  │  (CohJones)    │   │  (KAFCA)        │
                │                  │   doLMStep     │   │   doEKFStep     │
                │                  │   Tikhonov     │   │   Pa, Q, kapa   │
                │                  └────────────────┘   └─────────────────┘
                │                                                ▲
                │                                                │
                ▼                                       ClassEvolve
   ┌──────────────────────────────────────────────────┐  (time-evolution of P)
   │ Predict/PredictGaussPoints_NumExpr5              │
   │  ClassPredict + ClassPredictParallel             │
   │  → C extension Predict/predict.c (predictJones2_Gauss, ApplyJones)
   └──────────────────────────────────────────────────┘
                ▲
                │
   ┌────────────────────────────┐    ┌────────────────────────────────┐
   │  Data/ClassMS              │    │  Array/NpShared (/dev/shm)     │
   │   pyrap.tables, casacore   │◀──▶│  shared NumPy arrays for       │
   │   read columns, flags, UVW │    │  multiprocess workers          │
   └────────────────────────────┘    └────────────────────────────────┘
                ▲
                │
   ┌────────────────────────────┐
   │  Data/ClassVisServer       │
   │  TChunk streaming, weights │
   │  Briggs / Natural / Uniform│
   └────────────────────────────┘
```

The data path is:

1. `kMS.driver()` builds `OP` (`MyOptParse`) from `DefaultParset.cfg`.
2. A `ClassMS` is opened on the requested `--MSName`; a `ClassVisServer`
   wraps it and streams TChunks (default 15 hours
   per chunk — `DefaultParset.cfg:3`).
3. The sky model is loaded in one of three modes (chosen by `kMS.py:434–442`):
    - `PredictMode="Catalog"`     `--SkyModel=...npy` consumed by `SkyModel.Sky.ClassSM`
    - `PredictMode="Image"`       `--BaseImageName=...` reads a DDFacet `*.DicoModel`
    - `PredictMode="Column"`      `--SkyModelCol=COL` uses a vis-column predict
4. A `ClassWirtingerSolver` is constructed and `InitSol()` allocates the
   shared per-time `SolsArray_*` arrays (covered below).
5. Per chunk, `doNextTimeSolve_Parallel()` (`ClassWirtingerSolver.py:828`)
   spawns `NCPU` `WorkerAntennaLM` processes (multiprocessing.Process). Each
   worker pulls (`iAnt, iChanSol, …`) jobs from a queue, builds a fresh
   `ClassSolverLM` or `ClassSolverEKF` for that antenna, calls `doLMStep`
   or `doEKFStep`, and returns the result via `result_queue`.
6. After all antennas converge, the solution for time-step `iCurrentSol`
   is appended to `SolsArray_G`.
7. After the chunk completes, the solution is written to disk as a
   `*.killMS.<SolName>.sols.npz` file (see *Outputs* below).

## 5. The CLI: `kMS.py`

All runtime configuration is stored in a single layered object:

- `simulators/killMS/killMS/Parset/DefaultParset.cfg` — single source of truth
  for default values, organised in INI sections.
- `simulators/killMS/killMS/Parset/ReadCFG.py` — `configparser`-based loader,
  with `FormatValue()` automatically coercing `True/False`, `None`, `[a,b]`
  lists, `(a,b)` tuples, ints and floats (`ReadCFG.py:42–87`).
- `simulators/killMS/killMS/Parset/MyOptParse.py` — wraps `optparse` so each
  option is mirrored to a section of the parset; tracks `parameter_types` for
  schema generation (`MyOptParse.py:73`).

Special CLI behaviour:

- `kMS.py --MSName MAKE_SCHEMA` regenerates `killms_stimela_schema.yaml`
  (`kMS.py:313–317`).
- `kMS.py nocol` silences ANSI colour, `kMS.py nox` switches matplotlib to the
  Agg backend (`kMS.py:58–63`).
- The last-used options are pickled to `last_killMS.obj` so a re-run with no
  arguments resumes the previous configuration (`kMS.py:55, 242–244, 299–300`).

### 5.1 Parset sections, defaults, and CLI flags

The exhaustive list below is the union of `DefaultParset.cfg` and the
`OP.add_option()` calls in `kMS.read_options()` (`kMS.py:83–235`). Defaults
shown are from the parset; the CLI flag is `--<Name>` (kMS uses a flat option
namespace where the section is reconstructed from the dest key).

#### `[VisData]` — input MS and columns

| Option | Default | Type | Description (verbatim where given) |
|---|---|---|---|
| `MSName` | _empty_ | str | Input MS to draw [no default]. Accepts a single MS or a `.txt` mslist. `kMS.py:93` |
| `TChunk` | `15` | float | Time chunk in hours. `kMS.py:94` |
| `InCol` | `CORRECTED_DATA_BACKUP` | str | Column to work on. `kMS.py:95` |
| `OutCol` | `CORRECTED_DATA` | str | Column to write to. `kMS.py:96` |
| `FreePredictColName` | `None` | str | Column to write the free predicted vis to. `kMS.py:98` |
| `FreePredictGainColName` | `None` | str | Column to write predicted-with-gains vis to. `kMS.py:99` |
| `Parallel` | `1` | int | Run `doNextTimeSolve_Parallel` rather than the serial code path. `kMS.py:100` |

#### `[SkyModel]` — catalog/column predict

| Option | Default | Type | Description |
|---|---|---|---|
| `SkyModel` | _empty_ | str | NPY catalog (typically `<root>.txt.npy` produced by SkyModel/MakeModel). `kMS.py:104` |
| `kills` | _empty_ | str/list | Comma-separated source names/indices to *kill* (subtract). `kMS.py:106, 388` |
| `invert` | `False` | bool | Invert the kill selection. `kMS.py:107` |
| `Decorrelation` | _empty_ | str | Smearing model — `T` and/or `F` toggles bandwidth/time decorrelation. `kMS.py:108`; consumed by `ClassPredict` via `LExp/LSinc` lookup tables (`PredictGaussPoints_NumExpr5.py:88–115`). |
| `FreeFullSub` | `0` | int | If 1, subtract the full predicted gain-corrected vis from the input. `kMS.py:109, 797` |
| `SkyModelCol` | `None` | str | Use a vis column as the sky-model. `kMS.py:110, 437` |

#### `[ImageSkyModel]` — DDFacet DicoModel predict

| Option | Default | Type | Description |
|---|---|---|---|
| `BaseImageName` | _empty_ | str | DDFacet output basename. Reads `<base>.DicoModel`. `kMS.py:114, 442–462` |
| `ImagePredictParset` | _empty_ | str | (legacy) external parset for the predict step. |
| `DicoModel` | `None` | str | Override the DicoModel path. `kMS.py:451` |
| `OverS` | `None` | int | Oversampling factor for the convolution kernels. `kMS.py:544` |
| `wmax` | `None` | float | Override max-w for the W-projection. `kMS.py:546` |
| `MaskImage` | `None` | str | Mask FITS image. |
| `NodesFile` | `None` | str | Cluster catalog NPY (e.g. `<base>.npy.ClusterCat.npy`). |
| `MaxFacetSize` | `None` | float | Max DDF facet size (deg). `kMS.py:517` |
| `MinFacetSize` | `None` | float | Min facet size. `kMS.py:522` |
| `DDFCacheDir` | _empty_ | str | DDF cache dir. `kMS.py:497` |
| `RemoveDDFCache` | `False` | bool | Reset the DDF cache. |
| `FilterNegComp` | `False` | bool | Drop negative model components. |
| `ThSolve` | `0.0` | float | Tessel solving threshold; below `ThSolve*MaxSumFlux` (max over tessels) the tessel is skipped (J=I). `kMS.py:126` |

#### `[Compression]` — visibility compression

| Option | Default | Type | Description |
|---|---|---|---|
| `CompressionMode` | `None` | str | `None` or `auto`. `kMS.py:129` |
| `CompressionDirFile` | `None` | str | Directions NPY for manual compression. `kMS.py:130` |
| `MergeStations` | `None` | str | Pattern, e.g. `[CS]` to merge LOFAR core stations. `kMS.py:131` |

When `MergeStations` is set, `ClassAverageMachine` averages the kernel matrix
over baselines that share the merged stations
(`ClassAverageMachine.py:8–60`).

#### `[DataSelection]`

| Option | Default | Type | Description |
|---|---|---|---|
| `UVMinMax` | `None` | tuple | km, e.g. `0.1,100`. `kMS.py:134, 401–404` |
| `ChanSlice` | `None` | tuple/str | `start,stop,step` channel slice. `kMS.py:135, 511–514` |
| `FlagAnts` | _empty_ | str | Antenna flag pattern. `kMS.py:136, 405–407` |
| `DistMaxToCore` | `10000.` | float | km cut on antenna distance. `kMS.py:137` |
| `FillFactor` | `1.` | float | Down-sampling factor for the data. `kMS.py:138` |
| `FieldID` | `0` | int | MS FIELD_ID. `kMS.py:139` |
| `DDID` | `0` | int | MS DATA_DESC_ID. `kMS.py:140` |

(`ClassMS` builds the TaQL `FIELD_ID==FieldID && DATA_DESC_ID==DDID`,
`ClassMS.py:65`.)

#### `[Beam]`

| Option | Default | Type | Description |
|---|---|---|---|
| `BeamModel` | `None` | str | `None`/`LOFAR`/`FITS` — model to apply. `kMS.py:143` |
| `BeamAt` | `facet` | str | `tessel` or `facet`. `kMS.py:144` |
| `LOFARBeamMode` | `AE` | str | `A` (array factor only) or `AE` (array+element). `kMS.py:145` |
| `DtBeamMin` | `5` | float | Beam re-evaluation interval [min]. `kMS.py:146` |
| `CenterNorm` | `True` | bool | Normalise beam at field centre. `kMS.py:147` |
| `NChanBeamPerMS` | `1` | int | Number of beam frequencies per MS. `kMS.py:148` |
| `FITSParAngleIncDeg` | `5` | float | PA increment for FITS-beam re-eval. `kMS.py:149` |
| `FITSFile` | `beam_$(corr)_$(reim).fits` | str | FITS-beam template. `kMS.py:150` |
| `FITSLAxis` | `-X` | str | FITS L-axis convention (`-` reverses). `kMS.py:151` |
| `FITSMAxis` | `Y` | str | FITS M-axis. `kMS.py:152` |
| `FITSFeed` | _empty_ | str | `xy`/`rl`/None. `kMS.py:153` |
| `FITSFeedSwap` | `0` | int | Swap feeds. `kMS.py:154` |
| `FITSVerbosity` | `1` | int | `kMS.py:155` |
| `ApplyPJones` | `0` | int | Derotate visibilities for parallactic angle (FITS beam only). `kMS.py:156` |
| `FlipVisibilityHands` | `0` | int | Anti-diagonal swap of the polarisation hands. `kMS.py:157` |
| `FeedAngle` | `0` | float | Offset to add to the parallactic angle. `kMS.py:158` |
| `FITSFrame` | `altaz` | str | `altaz`/`altazgeo`/`equatorial`/`zenith`. `kMS.py:159` |

#### `[PreApply]` — pre-applied solutions

| Option | Default | Type | Description |
|---|---|---|---|
| `PreApplySols` | `[]` | list | List of killMS solution names to apply *before* solving. `kMS.py:162` |
| `PreApplyMode` | `[]` | list | Per-entry: `"A"`, `"P"`, `"AP"`. `kMS.py:163` |

#### `[Weighting]`

| Option | Default | Type | Description |
|---|---|---|---|
| `Resolution` | `0.` | float | Solution resolution [arcsec]. `kMS.py:167` |
| `WeightInCol` | `None` | str | Column to read weights from. `kMS.py:168` |
| `Weighting` | `Natural` | str | `Natural`/`Briggs`/`Uniform`. `kMS.py:169` |
| `Robust` | `0.` | float | Briggs robustness. `kMS.py:170` |
| `WeightUVMinMax` | `None` | tuple | km baselines that get full weight. `kMS.py:171` |
| `WTUV` | `1` | float | Weight scaling outside that range. `kMS.py:172` |

#### `[Actions]`

| Option | Default | Type | Description |
|---|---|---|---|
| `DoPlot` | `0` | int | Plot solutions (matplotlib) for debugging. `kMS.py:175` |
| `SubOnly` | `0` | int | Subtract only, no solve. `kMS.py:176` |
| `DoBar` | `1` | str | Show progress bars. `kMS.py:177` |
| `NCPU` | (parset 1; CLI default 75% of cpu count) | int | Worker count. `kMS.py:70, 178` |
| `NThread` | `1` | int | OMP / BLAS thread count. `kMS.py:179` |
| `UpdateWeights` | `1` | str | Update IMAGING_WEIGHT after the solve. `kMS.py:180` |
| `DebugPdb` | `1` | int | Drop into pdb on error. `kMS.py:181` |

#### `[Solutions]`

| Option | Default | Type | Description |
|---|---|---|---|
| `ExtSols` | _empty_ | str | External solution name to apply (skips solving). `kMS.py:188` |
| `ApplyMode` | `AP` | str | `A`/`P`/`AP` — what part of the solution to apply. `kMS.py:189` |
| `ClipMethod` | `[ResidAnt]` | list | `Resid`/`DDEResid`/`ResidAnt` for `IMAGING_WEIGHT` clipping. `kMS.py:190` |
| `OutSolsName` | _empty_ | str | Output solution name (used in the file name). `kMS.py:191` |
| `ApplyToDir` | `-2` | int | -1 = mean over directions; -2 = off; ≥0 = direction index. `kMS.py:192` |
| `MergeBeamToAppliedSol` | `0` | int | Bake beam into output Jones. `kMS.py:194` |
| `SkipExistingSols` | `0` | int | Skip if the output `.sols.npz` already exists. `kMS.py:195` |
| `SolsDir` | `None` | str | Output directory for solutions. `kMS.py:196` |

#### `[Solvers]`

| Option | Default | Type | Description |
|---|---|---|---|
| `SolverType` | `CohJones` | str | `CohJones` or `KAFCA`. `kMS.py:200` |
| `PrecisionDot` | `D` | str | `S`/`D`. (Note: `ClassJacobianAntenna.__init__` overrides to `complex128` regardless — `ClassJacobianAntenna.py:195`.) |
| `PolMode` | `Scalar` | str | `Scalar`/`IDiag`/`IFull`. `kMS.py:202` |
| `dt` | `30` | float | Solution interval [minutes]. `kMS.py:203` |
| `NChanSols` | `1` | int | Number of frequency-domain solution slots. `kMS.py:204` |

`PolMode` controls the Jacobian shape (`ClassJacobianAntenna.py:203–217`):

| PolMode | NJacobBlocks_X × NJacobBlocks_Y | npolData |
|---|---|---|
| `Scalar` | 1 × 1 | 1 |
| `IDiag` | 2 × 1 | 2 |
| `IFull` | 2 × 2 | 4 |

#### `[CohJones]`

| Option | Default | Type | Description |
|---|---|---|---|
| `NIterLM` | `7` | int | LM iterations per time-step. `kMS.py:209` |
| `LambdaLM` | `1` | float | LM damping. `kMS.py:210` |
| `LambdaTk` | `0.0` | float | Tikhonov regularisation strength. When >0, `ClassSolverLM.PrepareJHJ_LM` adds a normalised diagonal `Linv = LambdaTk * mean(|diag(M)|)/(1+LambdaLM)` (`ClassSolverLM.py:48–73`). |

#### `[KAFCA]`

| Option | Default | Type | Description |
|---|---|---|---|
| `NIterKF` | `6` | int | KAFCA iterations per time-step. `kMS.py:215` |
| `LambdaKF` | `0.5` | float | Kalman damping factor. `kMS.py:216` |
| `InitLM` | `0` | int | Bootstrap KAFCA from an LM solve. `kMS.py:217, 384` |
| `InitLMdt` | `5` | float | LM bootstrap interval [min]. `kMS.py:218` |
| `CovP` | `0.1` | float | Initial prior covariance, fraction of `|G|`. `kMS.py:219` |
| `CovQ` | `0.1` | float | Process noise, fraction of `|G|`. `kMS.py:220` |
| `PowerSmooth` | `1.` | float | Down-weights `Q` when an antenna has missing baselines. `kMS.py:221` |
| `evPStep` | `120` | int | Re-compute the evolution matrix every N steps. `kMS.py:222` |
| `evPStepStart` | `1` | int | Start step for the `(I-KJ)` calc. `kMS.py:223` |
| `EvolutionSolFile` | _empty_ | str | Prior solution file (KAFCA initial state). `kMS.py:224, 200` |

When `SolverType="KAFCA"`, the CLI hides `[CohJones]`, and vice-versa
(`kMS.py:230–236`).

## 6. Public Python API

killMS is primarily a CLI; its `[project.scripts]` are the supported public
surface. Programmatic usage is supported but undocumented; the most stable
classes are:

### 6.1 Top-level driver (`killMS.kMS`)

```python
def driver()       # parse CLI, save options, call main(OP)
def read_options() # build MyOptParse, return the OP object
def main(OP=None, MSName=None)  # execute the solve loop; pickle resume supported
def GiveNoise(options, DicoSelectOptions, IdSharedMem, SM, PM, PM2, ConfigJacobianAntenna, GD)
                   # bootstrap solver — used internally for KAFCA InitLM
```

### 6.2 Wirtinger solver (`killMS.Wirtinger.ClassWirtingerSolver`)

```python
class ClassWirtingerSolver:
    def __init__(self, VS, SM,
                 BeamProps=None, PolMode="IFull", Lambda=1, NIter=20,
                 NCPU=6, SolverType="CohJones", IdSharedMem="",
                 evP_StepStart=0, evP_Step=1,
                 DoPlot=False, DoPBar=True, GD=None,
                 ConfigJacobianAntenna={}, TypeRMS="GlobalData",
                 VS_PredictCol=None)

    def InitSol(self, G=None, TestMode=True)
    def InitMeanBeam(self)
    def InitCovariance(self, FromG=False, sigP=0.1, sigQ=0.01)
    def InitReg(self)
    def SetRmsFromExt(self, rms)

    def setNextData(self)             # one iteration's data; returns True | "EndChunk" | "EndOfObservation" | "AllFlaggedThisTime"
    def doNextTimeSolve(self, SkipMode=False)            # serial loop
    def doNextTimeSolve_Parallel(self, OnlyOne=False, SkipMode=False, Parallel=True)
                                       # spawns NCPU WorkerAntennaLM processes

    def AppendGToSolArray(self)
    def GiveSols(self, SaveStats=False)  # → recarray with t0/t1/G/Stats
```

`ConfigJacobianAntenna` is a dict — keys consumed in `kMS.py:655–663`:

```python
{"DoSmearing":   options.Decorrelation,
 "ResolutionRad": (options.Resolution/3600)*(np.pi/180),
 "LambdaKF":     options.LambdaKF,
 "LambdaLM":     options.LambdaLM,
 "DoReg":        False,
 "gamma":        1,
 "AmpQx":        0.5,
 "PrecisionDot": options.PrecisionDot,
 "DicoMergeStations": VS.DicoMergeStations}
```

### 6.3 Per-antenna Jacobian (`killMS.Wirtinger.ClassJacobianAntenna`)

```python
class ClassJacobianAntenna:
    def __init__(self, SM, iAnt, PolMode="IFull", Precision="S", PrecisionDot="D",
                 IdSharedMem="", PM=None, PM_Compress=None, SM_Compress=None,
                 GD=None, NChanSols=1, ChanSel=None,
                 SharedDicoDescriptors=None, **kwargs)

    # Core Wirtinger ops
    def J_x(self, Gains)           # Σ_d J_d g_d
    def JH_z(self, zin)            # J^H z
    def JHJinv_x(self, Gains)      # (J^H J)^{-1} g
    def Msq_x(self, LM, Gains)     # generic L · g

    # State machinery
    def setDATA(self, DATA);  def setDATA_Shared(self)
    def CalcKernelMatrix(self, rms=None)
    def CalcJacobianAntenna(self, GainsIn)
    def SelectChannelKernelMat(self)
    def GiveData(self, DATA, iAnt, rms=None)
    def GiveDataVec(self);   def GiveSubVecGainAnt(self, GainsIn)

    # Hooks for Image-mode predict
    def PredictOrigFormat(self, GainsIn)
    def CalcMatrixEvolveCov(self, G, P, rms)
```

`ClassSolverLM` (`Wirtinger/ClassSolverLM.py:34`) and `ClassSolverEKF`
(`Wirtinger/ClassSolverEKF.py:36`) are subclasses that add `doLMStep()` and
`doEKFStep()` respectively.

### 6.4 MS access (`killMS.Data.ClassMS`)

```python
class ClassMS:
    def __init__(self, MSname, Col="DATA", zero_flag=True, ReOrder=False,
                 EqualizeFlag=False, DoPrint=True, DoReadData=True,
                 TimeChunkSize=None, GetBeam=False, RejectAutoCorr=False,
                 SelectSPW=None, DelStationList=None, Field=0, DDID=0,
                 ReadUVWDT=False, ChanSlice=None, GD=None, ToRADEC=None)
    # public methods include AddCol, GiveMainTable, ReadMSInfo, ToOrigFreqOrder,
    # SaveVis, LoadLOFAR_ANTENNA_FIELD, etc. (1277 LOC, see ClassMS.py)
```

Reads via `pyrap.tables.table` and `pyrap.measures` / `pyrap.quanta`
(`ClassMS.py:25–32`). TaQL field/DDID selection is hard-wired:
`FIELD_ID==Field && DATA_DESC_ID==DDID` (`ClassMS.py:65`).

### 6.5 Visibility streaming (`killMS.Data.ClassVisServer`)

```python
class ClassVisServer:
    def __init__(self, MSName, ColName="DATA", TChunkSize=1, TVisSizeMin=1,
                 DicoSelectOptions={}, LofarBeam=None, AddNoiseJy=None,
                 IdSharedMem="", SM=None, NCPU=None,
                 Robust=2, Weighting="Natural",
                 WeightUVMinMax=None, WTUV=1.0, GD=None, GDImag=None)
    def setSM(self, SM); def setGridProps(self, Cell, NpixPaddedFacet)
    def setFOV(self, ...); def CalcWeigths(self)
    def LoadNextVisChunk(self); def GiveNextVis()
```

### 6.6 Shared-memory layer (`killMS.Array.NpShared`)

`/dev/shm`-backed NumPy arrays via the `SharedArray` PyPI package
(`NpShared.py:25`). Public surface:

| Function | Effect |
|---|---|
| `ToShared(Name, A)` | create a SHM array of `A.shape, A.dtype`, copy into it. (`NpShared.py:36–47`) |
| `zeros(*args, **kwargs)` | thin alias for `SharedArray.create` (`NpShared.py:33`) |
| `GiveArray(Name)` | attach an existing SHM array; returns `None` on failure (`NpShared.py:70`) |
| `DelArray(Name)` | delete; swallows errors |
| `DelAll(key=None)` | delete every SHM array (or those with `key` substring) |
| `ListNames()` | list SHM array names |
| `DicoToShared(Prefix, Dico, DelInput=False)` | bulk move a dict-of-arrays to SHM |
| `SharedToDico`, `SharedObjectToDico`, `SharedDictToObject` | counterparts |

### 6.7 Linear-algebra helpers (`killMS.Array.ModLinAlg`)

```python
invertChol(A); invertLU(A); sqrtSVD(A, Rank=None); invSVD(A)
BatchInverse(A, H=False);   # batched 2x2 inverse / Hermitian
BatchH(A); BatchDot(A, B)   # batched 2x2 ops; reshapes to (N,2,2)
```

The 2×2 batched ops are the workhorses of the per-direction matrix products in
`PredictGaussPoints_NumExpr5.ApplyCal`.

### 6.8 The C extensions

`predict.c` exports two Python functions — only the second pair is included in
the methods table:

```c
static PyMethodDef module_functions[] = {
    {"predictJones2_Gauss", predictJones2_Gauss, METH_VARARGS},
    {"ApplyJones",         ApplyJones,         METH_VARARGS},
    {NULL, NULL}
};
```

`predict.h` declares (but the methods table does not export, presumably for
backwards-compat):

```c
predict, predictJones, predictJones2,
predictJones2_Gauss, ApplyJones, CorrVis, GiveMaxCorr
```

Helpers inside `predict.c`/`predict.h`: `MatInv`, `MatH`, `MatDot`, `GiveJones`,
`PrintArray`, `GiveFunc` (lookup-table for the smearing exp/sinc tables).

`Gridder.c` exports W-projection gridder/degridder methods:
```c
{"pyGridderWPol",    pyGridderWPol,    METH_VARARGS},
{"pyGridderPoints",  pyGridderPoints,  METH_VARARGS},
{"pyDeGridderWPol",  pyDeGridderWPol,  METH_VARARGS},
{"pyTestMatrix",     pyTestMatrix,     METH_VARARGS},
{"pyAddArray",       pyAddArray,       METH_VARARGS},
{"pyWhereMax",       pyWhereMax,       METH_VARARGS},
```

`Array/Dot/dotSSE.c` provides an SSE-vectorised dot-product, callable from
`NpDotSSE.dot_A_BT(A, B)` (used in `J_x`/`JH_z` only when `self.TypeDot=="SSE"`,
`ClassJacobianAntenna.py:347–391`; the default is `"Numpy"`).

## 7. Core algorithms

### 7.1 The RIME and the Wirtinger Jacobian

For one polarisation block, the model visibility is

  V_{pq}(t,ν) = Σ_d J_p(d) · K_d(p,q,t,ν) · J_q(d)^H

where `K_d(p,q,t,ν)` is the **kernel matrix** — the per-direction predicted
visibility *without* the Jones gains. In code, `K_XX` and `K_YY` are stored
shape `(NDir, n4vis_AllChan/nchan, nchan)` per antenna in
`ClassJacobianAntenna.CalcKernelMatrix` (`Wirtinger/ClassJacobianAntenna.py:600–800`).

The Wirtinger half-Jacobian for antenna *p* is built explicitly in
`CalcJacobianAntenna` (`Wirtinger/ClassJacobianAntenna.py:483–540`); for
`PolMode="Scalar"` the assembly is

```python
Gains = GainsIn.reshape((na, NDir, 1, 1))
for iDir in range(NDir):
    G = Gains[A1, iDir].conj()              # gains of the *other* antenna
    K_XX = self.K_XX[iDir]
    g0_conj = G[:, 0, 0].reshape((nr, 1))
    Jacob[:, 0, iDir, 0] = (g0_conj * K_XX).reshape((-1,))
```

For `IFull` four blocks `(0,0),(0,1),(1,0),(1,1)` are filled with the
appropriate `g_conj * K_{XX|YY}` products (`ClassJacobianAntenna.py:525–540`).
The result is the **complex Jacobian** `J = ∂(model vis) / ∂(g_p^*)` — the
"Wirtinger half" in the cited Tasse 2014 paper. It is *only* the conjugate
half because the model is bilinear in `(g_p, g_q^*)` and the partial w.r.t.
`g_p^*` depends only on `g_q^*`.

### 7.2 CohJones (Levenberg–Marquardt)

`Wirtinger/ClassSolverLM.py:75–246` implements one LM step per antenna:

```
1. CalcKernelMatrix once per chunk (cached in SHM)
2. CalcJacobianAntenna(Gains)           # build LJacob[ipol]
3. PrepareJHJ_LM:                       # invert J^H J (+ Tikhonov diag)
       M    = L_JHJ[ipol]               # = J^H W J  precomputed
       if DoTikhonov:
           Linv = diag(self.Linv[:,ipol,:].ravel())
           Linv *= LambdaTk * mean(|diag(M)|) / (1 + LambdaLM)
           M2 = M + Linv
       else:
           M2 = M
       JHJinv = ModLinAlg.invSVD(M2)
4. Jx   = J_x(Ga)                       # forward predict from current Ga
   zr   = z - Jx                         # residual
   zr[flagged] = 0
   JH_z = JH_z(zr)                       # back-project
   if DoTikhonov:
       JH_z -= LambdaTkNorm * Linv * (Gi - X0)
5. Δx   = (1/(1+LambdaLM)) * JHJinv_x(JH_z)
   xout = Ga + Δx
```

(`ClassSolverLM.py:100–246`.) `LambdaLM=1` (the default) gives the standard
half-step LM update; `LambdaLM>>1` damps toward steepest-descent; `LambdaLM=0`
becomes Gauss–Newton.

`PrepareJHJ_LM` requires `L_JHJ` to have been computed; that happens inside
`CalcJacobianAntenna` (which the cited file caches as
`self.L_JHJ[ipol] = J^H · W · J`). Tikhonov regularisation uses a per-direction
inverse-flux matrix `Linv` populated by `ClassWirtingerSolver.InitReg`
(`ClassWirtingerSolver.py:479–514`):

```python
SumIApp     = SumI * AbsMeanBeamAnt**2
Linv        = (1 / SumIApp).reshape((NDir,1,1)) ** 2
NpShared.ToShared("%sLinv"%IdSharedMem, Linv)
NpShared.ToShared("%sX0"%IdSharedMem,  X0)
```

This regularises *bright* directions toward `X0=1` (identity Jones) more
strongly than faint ones — the inverse Linv increases as the apparent flux
drops, but the multiplication is `Linv * (G - X0)` so dim directions get the
strongest pull to identity.

### 7.3 KAFCA (Extended Kalman Filter)

`Wirtinger/ClassSolverEKF.py:242–443` implements one EKF update per antenna:

```
state x:          per-antenna gain vector (shape NDir × NJacobBlocks_X × NJacobBlocks_Y)
state covar P:    block-diagonal across antennas, (NDir·npx·npy) × (NDir·npx·npy)
process noise Q:  built in InitCovariance from apparent flux per cluster
                  Q_d = sigQ² * |G_max|² * (ApparentFlux_d) on the (d,d) block
                  optional smoothing (1 + d/d0)^{-2} where d0 = 1°
                  PowerSmooth controls how much |meanW|/n_baselines down-weights Q
```

EKF step:

```
1. CalcKernelMatrix (cached) and CalcJacobianAntenna(Gains)
2. PrepareJHJ_EKF(Pa, rms):
       PaPol  = block-diagonal pol-slice of Pa
       Pinv   = invSVD(PaPol)
       JHJ    = L_JHJ[ipol] + Pinv         # (J^H R^-1 J + P^{-1})
       JHJinv = invSVD(JHJ)
3. zr  = z - J_x(Ga);   zr[flag] = 0
4. kapa = CalcKapa_i(zr, Pa, rms)
       ≈ sqrt( |trYYH - trR| / |trJPJ^H| )
       Floors at 1.  Re-scales the data residual energy vs the predicted
       residual energy (this is the noise-scale estimator that gives KAFCA
       its kappa=1 sanity check).
5. ApplyK_vec(zr, rms, Pa, DoReg=True):
       Rinv·zr → x1 = (J^H J + P^{-1})^{-1} · J^H R^-1 zr
       z1 = J·x1
       zr -= z1; zr *= R^-1
       x2 = J^H zr
       x3 = Pa · x2                           # i.e. Kalman gain  K = P·H^T·(...)
       (optional regularisation toward G0 = 1 with weight gamma)
       returns x3
6. Gnew = Ga + LambdaKF · x3
   Pnew = Pa - LambdaKF · Pa · J^H · (J^H J + P^{-1})^{-1} · J · Pa
        ≈ (I - K·J)·Pa     (the standard EKF posterior covariance)
7. Time-evolve:  Pa' = ClassEvolve.Evolve0(Pnew, kapa)
       Pa' = Pa + kapa · Q                    when iCurrentSol > evPStepStart
```

(See `ClassSolverEKF.PrepareJHJ_EKF` lines 51–69, `CalcKapa_i` 74–93,
`ApplyK_vec` 143–238, `doEKFStep` 242–443.)

`InitCovariance` constructs the prior:

```
P = sigP² · max(|G|)² · I              # initial state covariance
Q = sigQ² · max(|G|)² · Q_a            # process noise
   where Q_a uses ApparentFluxes (mean beam · NormFluxes)
   and optionally a Gaussian "directional coherence" matrix QQ
ApparentSumI = AbsMeanBeamAnt · NormFluxes   (computed in InitMeanBeam)
```

(`ClassWirtingerSolver.py:353–443, 445–477`). Beam-aware priors mean the
KAFCA covariance is anisotropic: bright directions have larger Q and
therefore evolve faster.

`ClassEvolve.ClassModelEvolution.Evolve0` (`Wirtinger/ClassEvolve.py:40–100+`)
implements a low-pass on previously-solved Gains:

```
Gm = mean over time of |G|, at this antenna/dir
dG = Gm[t-1] - Gm[t]
weight by exp(-dt/WeigthScale)        (default WeigthScale = 0.3 min)
Pnew = Pa + kapa · Q
```

`InitLM=1` runs CohJones first to seed `G` and `rms` for KAFCA
(`kMS.py:692–706`).

### 7.4 Compression / station merging

`ClassAverageMachine.AverageKernelMatrix` (`Wirtinger/ClassAverageMachine.py:26–60`)
re-phases the kernel onto a coarser sky-direction grid:

```
For each compressed direction d̃:
    K_compress = predict at d̃  (the rephasing factor)
    KOut[d, d̃, blbl] = sum over original baselines (K[d,*] · K_compress.conj())
```

`MergeStations` then merges the rephased blocks per the `DicoMergeStations`
mapping (`ClassJacobianAntenna.py:177`).

## 8. Inputs and outputs

### 8.1 Input formats

| Format | Where it's used | Reader |
|---|---|---|
| Measurement Set (CASA v2) | The data column. `--MSName` (single) or a `mslist.txt` (one per line). | `Data/ClassMS.py` (uses `pyrap.tables.table`) |
| Sky-model `.npy` (SkyModel format) | `--SkyModel`. | `SkyModel.Sky.ClassSM` (DDFacet's helper package, **not** in this repo) |
| DDFacet `.DicoModel` | `--BaseImageName` (image-mode predict). Pickled dict per DDFacet. | `DDFacet.Other.MyPickle.Load`, then `Predict/ClassImageSM2.ClassPreparePredict` |
| Cluster-cat `.npy` | `--NodesFile`. | NumPy-loaded recarray |
| Parset `.cfg` | INI sections matching `DefaultParset.cfg`. | `Parset/ReadCFG.py` |
| BBS-style sky models | NOT supported here (this differs from older killMS docs); the code path expects `.npy` from `MakeModel.py`. |
| FITS beam | `--FITSFile beam_$(corr)_$(reim).fits`. | `Data/ClassBeam.py` |

### 8.2 Output: solution file

After every chunk, `kMS.py:866–876` writes a NPZ:

```
<MSName>/killMS.<SolName>.sols.npz       (default)
<SolsDir>/<MSName>/killMS.<SolName>.sols.npz   (if --SolsDir set)

contents:
    MSName           absolute path
    MSNameTime0      MS time-zero (UTC seconds)
    Sols             recarray with fields:
                       t0     float64
                       t1     float64
                       G      complex64, shape (NChanSols, NAnt, NDir, 2, 2)
                       Stats  float32,   shape (NChanSols, NAnt, 4)
                              [0]=std(zr) [1]=max(|zr|) [2]=kapa [3]=rms
    StationNames     array of antenna names
    SkyModel         ClusterCat recarray (the cluster catalog)
    ClusterCat       same
    SourceCatSub     subtracted-source catalog (or None)
    ModelName        the SkyModel/BaseImageName used
    FreqDomains      array of frequency domains per ChanSol
    BeamTimes        array of beam re-evaluation times
```

The accompanying `<MSName>/killMS.<SolName>.sols.parset` is written by
`OP.ToParset(ParsetName)` immediately on launch (`kMS.py:419–426`).

### 8.3 Output column-mode predict

If `--FreePredictColName` or `--FreePredictGainColName` is set, kMS will write
the no-gain or with-gain predict to a new MS column (created via
`ClassMS.AddCol` if necessary) — see `kMS.py:769–795`.

## 9. Testing

The test surface is **minimal** and lives outside the package:

- `simulators/killMS/TestHarness/` is **not shipped in the wheel**
  (`pyproject.toml:120` `sdist.exclude = ["killMS/cbuild", "TestHarness"]`).
- The single test class is `TestHarness/LongAcceptanceTests/TestLOFAR_J1329_p4729.py`.
  It inherits from `DDFacet.Tests.ShortAcceptanceTests.ClassCompareFITSImage`
  (`TestLOFAR_J1329_p4729.py:38`) and runs an end-to-end
  DDFacet→MakeMask→MaskDicoModel→DDFacet pipeline and compares the resulting
  FITS images pixel-by-pixel against reference frames. Tolerances:
  `1e-6/1e-6/1e-4/1e-4/5e-3/5e-3/5e-3/5e-3` per image (`defineMaxSquaredError`),
  `1e-7…1e-5` MSE (`defMeanSquaredErrorLevel`).
- `Jenkinsfile.sh:35` invokes
  `OPENBLAS_NUM_THREADS=1 nosetests -s --with-xunit --xunit-file ... /src/killMS/TestHarness`
  inside the docker image with `--shm-size=150g` and a 100 GB memory cap.
- There is also `killMS/Array/Dot/TestDotSSE.py` (a micro-bench) and
  `killMS/TestSpeedNp.py` (numpy timing demo). Neither is wired into the test
  runner.

There are **no unit tests** of the solvers themselves — the verification is
end-to-end via image comparison.

## 10. Internals worth knowing

### 10.1 Shared memory + multiprocessing

killMS does not use threads for the solve loop; instead it forks `NCPU` worker
processes (`multiprocessing.Process`) and shares state via the **`SharedArray`**
module which mmaps `/dev/shm`. `kMS.main()` checks at startup that
`/dev/shm` is sized to ≥60% of RAM and issues a SIGBUS/file-size warning if
not (`kMS.py:251–263`). The code also probes `/sbin/sysctl vm.max_map_count`
and warns if it is below 500000 (`kMS.py:274–283`).

The `IdSharedMem` prefix is `<pid>.` (`kMS.py:349`), which means killMS
processes never collide on `/dev/shm` even if multiple instances run on the
same host. `NpShared.DelAll(IdSharedMem)` is called at startup to evict any
stale arrays from prior runs of the same pid (`kMS.py:395`).

The major SHM arrays:

| Name | Used as | Created by |
|---|---|---|
| `<id>SharedGains` | per-antenna Jones G | `InitSol` |
| `<id>SharedGains0Iter` | snapshot at iteration start | `InitSol` |
| `<id>SharedCovariance` | EKF state P | `InitCovariance` |
| `<id>SharedCovariance_Q` | EKF Q | `InitCovariance` |
| `<id>SharedEvolveCovariance` | evP (predicted P) | `InitCovariance` |
| `<id>SolsArray_G/_t0/_t1/_done/_tm/_Stats` | output ring-buffer | `InitSol` |
| `<id>KernelMat.<iAnt:02d>` | cached per-direction kernel matrices | `CalcKernelMatrix` |
| `<id>DicoData.<iAnt:02d>` | per-antenna chunk data | `setDATA_Shared` |
| `<id>SharedVis.*` | the chunk vis | `ClassVisServer.LoadNextVisChunk` |
| `<id>PredictedData`, `<id>PredictedDataGains`, `<id>IndicesData[...]` | output predicts | `PredictOrigFormat_Type` |
| `<id>Linv`, `<id>X0` | Tikhonov regularisation | `InitReg` |

### 10.2 Worker processes

Two worker classes (both `multiprocessing.Process` subclasses):

- `WorkerAntennaLM` (`Wirtinger/ClassWirtingerSolver.py:1296+`) — the
  per-antenna solve. Pulls `(iAnt, iChanSol, DoCalcEvP, ThisTime, rms, DoEvP,
  DoFullPredict, SharedDicoDescriptors)` tuples; instantiates a fresh
  `ClassSolverLM` or `ClassSolverEKF`; calls `doLMStep` or `doEKFStep`; pushes
  the resulting `[iAnt, iChanSol, x, P, rmsFromData, InfoNoise, DT,
  AntennaVisDescriptor]` back on `result_queue`.
- `WorkerPredict` (`Predict/PredictGaussPoints_NumExpr5.py:897+`) — used by
  `ClassPredictParallel.predictKernelPolCluster` and `ApplyCal` for
  catalog-mode predicts; takes `(Row0, Row1)` chunks.

Both workers use `multiprocessing.Queue`-based job dispatch and a
`multiprocessing.Event` for shutdown.

### 10.3 BLAS thread pinning

`kMS.read_options` exposes `--NThread` (default 1) and the docstring at
`kMS.py:319–326` shows the original intention was to set
`OPENBLAS_NUM_THREADS=str(options.NThread)` before importing numpy — but the
relevant lines are commented out in the current code. The Jenkins runs do set
it explicitly (`Jenkinsfile.sh:35`). Practically: **set
`OPENBLAS_NUM_THREADS=1` yourself** when running with `--NCPU > 1` to avoid
oversubscription. The lock-limit warning at `kMS.py:288–295` also urges
`ulimit -l unlimited` for performance.

### 10.4 Numerical precision

- `PrecisionDot=D` is the parset default but **the constructor of
  `ClassJacobianAntenna` overrides every call to `complex128`**:
  `self.CType = np.complex128` (`ClassJacobianAntenna.py:195`). The
  `Precision="S"` flag passed to `ClassPredict` is honoured for the
  predict-side arrays only.
- `ClassWirtingerSolver` stores `G` as `complex64` (`ClassWirtingerSolver.py:319`)
  and `Sols.G` is `complex64` on disk.
- The recent commits `15636c7`/`9b8bdff`/`e3bd122`/`ee51c3f` (April–May 2026
  prep312 branch) explicitly restored deprecated NumPy dtypes for NumPy 2.x.

### 10.5 Logging and progress

The logger is configured by DDFacet (`from DDFacet.Other import logger`), with
killMS-specific loggers (`ClassWirtingerSolver`, `ClassMS`, `ClassVisServer`,
`NpShared`, etc.). Some loggers are explicitly silenced (`ClassFitTEC`,
`NpShared`). Progress bars come from
`DDFacet.Other.progressbar.ProgressBar` (`kMS.py:609`). Tracing is via
`killMS.Other.ClassTimeIt.ClassTimeIt` (lots of `T.timeit("…")` calls — most
disabled by default in production paths).

### 10.6 Resume / replay

`MyPickle.Save(OP, "last_killMS.obj")` is called at the end of `read_options`
(`kMS.py:243`) so re-running kMS without arguments will reload the previous
options (`kMS.py:299–300`). `SmoothSols`, `InterpSols`, `MergeSols` use the
same pattern with their own `last_*.obj` files.

## 11. Worked example

Reproduced verbatim from `simulators/killMS/README.md:74–112`:

```bash
# 1. Write all your MS paths to mslist.txt
$ cat mslist.txt
/data/.../L374583_SB244_uv.dppp.pre-cal_127080C79t_121MHz.pre-cal.ms
/data/.../L374583_SB254_uv.dppp.pre-cal_127080C79t_123MHz.pre-cal.ms
...

# 2. DI image with DDFacet
$ DDF.py --Output-Name=image_DI --Data-MS mslist.txt --Data-ColName DATA \
         --Parallel-NCPU=40 --Image-Mode=Clean --Deconv-Mode SSD \
         --Image-NPix=10000 --Image-Cell 3 --Facets-NFacets=11 \
         --Mask-Auto=1 --Mask-SigTh=5.00 ...

# 3. Cluster the sky into 10 directions
$ MakeModel.py --BaseImageName image_DI --NCluster 10
# → image_DI.npy.ClusterCat.npy

# 4. Direction-dependent calibration with KAFCA, scalar Jones
$ kMS.py --MSName mslist.txt \
         --SolverType KAFCA \
         --PolMode Scalar \
         --BaseImageName image_DI \
         --dt 1 \
         --NCPU 40 \
         --OutSolsName testKAFCA \
         --NChanSols 1 \
         --InCol DATA --OutCol DATA \
         --Weighting Natural \
         --NodesFile image_DI.npy.ClusterCat.npy \
         --MaxFacetSize 1.5
# → <MS>/killMS.testKAFCA.sols.npz per MS

# 5. Re-image, applying the killMS solutions
$ DDF.py --Output-Name=image_DD --Data-MS mslist.txt ... \
         --DDESolutions-DDSols testKAFCA \
         --Predict-InitDicoModel image_DI.DicoModel \
         --Facets-DiamMax 1.5 --Facets-DiamMin 0.1
```

## 12. Companion CLI tools

| Script | Purpose | Defining file |
|---|---|---|
| `kMS.py` | Direction-dependent calibration. | `killMS/kMS.py` |
| `SmoothSols.py` | Smooth a `.sols.npz` in time and frequency using TEC + Amp models (uses `Other/ClassFitTEC.py`, `Other/ClassFitAmp.py`, `Other/ClassClip.py`). | `killMS/SmoothSols.py` |
| `MergeSols.py` | Merge multiple `.sols.npz` (sorted by frequency). | `killMS/MergeSols.py` |
| `InterpSols.py` | Frequency-extrapolate a solution to a new MS. | `killMS/InterpSols.py` |
| `PlotSols.py` | matplotlib `plot()` of `|G|` and `arg(G)` per antenna/direction. | `killMS/PlotSols.py` |
| `PlotSolsIm.py` | `imshow` per (time × freq) panel. | `killMS/PlotSolsIm.py` |
| `MakePlotMovie.py` | Stitch `PlotSols` snapshots into an mp4. | `killMS/MakePlotMovie.py` |
| `AQWeight.py` | Recompute imaging weights based on the residual (uses `Weights/W_*.py`). | `killMS/AQWeight.py` |
| `ClipCal.py` | Threshold-clip outliers in a column (or back up FLAG). | `killMS/ClipCal.py` |
| `BLCal.py` | Per-baseline self-cal helper. | `killMS/BLCal.py` |
| `dsc.py` | Tiny DDF/SkyModel/Cluster utility. | `killMS/dsc.py` |
| `grepall.py` | Grep across the per-MS solution dirs. | `killMS/grepall.py` |

## 13. Extension points

### 13.1 Adding a solver

To add a third Wirtinger-style solver, the integration points are:

1. Subclass `Wirtinger/ClassJacobianAntenna.ClassJacobianAntenna` and
   implement a single `do<X>Step(Gains, ...)` method, mirroring
   `ClassSolverLM.doLMStep` or `ClassSolverEKF.doEKFStep`.
2. Add an option value to the parset's `[Solvers]/SolverType`.
3. Branch in `kMS.main` for the new SolverType (currently lines 665–668 and
   690–706 select `NIter` and call `InitMeanBeam/InitCovariance` for KAFCA only).
4. Add the new class to the worker dispatch in `WorkerAntennaLM.run` (a single
   `if self.SolverType=="<X>": SolverClass=<X>; ...` branch — see
   `Wirtinger/ClassWirtingerSolver.py:1383–1388`).

### 13.2 Adding a sky-model loader

`kMS.main` selects between `Catalog`, `Image`, and `Column` predict modes
(`kMS.py:434–442`). To add a fourth, plumb a new `PredictMode` branch and an
`SM` object exposing the contract:

```
SM.Type             "Catalog" | "Image" | "Column" | <new>
SM.NDir
SM.Calc_LM(rac, decc)
SM.ClusterCat       recarray with at least: ra, dec, SumI, Cluster
SM.SourceCat        recarray (Catalog only)
SM.Dirs             list of cluster ids
```

### 13.3 Adding a Predict backend

`PredictGaussPoints_NumExpr5.ClassPredict` is the reference. Its `Precision`
toggle (`S`/`D`) is honoured throughout, and it is interchangeable with
`PredictGaussPoints_NumExpr.ClassPredict` (the older numexpr-only variant
still imported by `Simul/DoSimul.py` and `SmoothSols.py`).

### 13.4 Adding a weighting scheme

`Data/ClassWeighting.py` provides `Natural`, `Briggs`, `Uniform`. New schemes
can hook in there and be exposed via `[Weighting]/Weighting`.

## 14. Caveats, TODOs, and known limitations

Discovered while reading the source:

- **Linux only.** `pyproject.toml:30` declares only `POSIX :: Linux`. The use of
  `/dev/shm`, `os.statvfs`, `/sbin/sysctl` (`kMS.py:251–283`), and `librt`
  (`FindRT.cmake`) confirm this.
- **DDFacet must be installed.** Without `DDFacet[kms-support]>=0.7.0` killMS
  cannot import (`pyproject.toml:18–20`, `kMS.py:49`).
- **`PrecisionDot` is partially honoured.** `ClassJacobianAntenna.py:195`
  unconditionally sets `self.CType = np.complex128` regardless of the user's
  `--PrecisionDot=S` choice.
- **`TypeDot` is hard-coded to `"Numpy"`.** The SSE path
  (`Array/Dot/dotSSE.c`) is wired up via `NpDotSSE.dot_A_BT` and exercised by
  the `TypeDot=="SSE"` branches in `J_x`/`JH_z`, but `ClassJacobianAntenna.py:196`
  hard-codes `self.TypeDot="Numpy"` (the line `#self.TypeDot="SSE"` is
  commented out).
- **`predict.h` declares more functions than `predict.c` exports.** Only
  `predictJones2_Gauss` and `ApplyJones` are reachable from Python
  (`predict.c:39–43`); `predict`, `predictJones`, `predictJones2`, `CorrVis`,
  and `GiveMaxCorr` are dead from the Python side.
- **`Predict/PredictGaussPoints_NumExpr.py` is legacy.** It still imports
  `predict27`/`predict3x` and is only retained for `Simul/DoSimul.py` /
  `SmoothSols.py` (as `ClassPredict5` is the new name in
  `SmoothSols.py:36–37`).
- **Heavy reliance on numexpr threading.** `ClassPredict.__init__` calls
  `ne.set_num_threads(self.NCPU)` (`PredictGaussPoints_NumExpr5.py:81`); each
  child worker also creates a `ClassPredict(NCPU=1, …)` to avoid
  oversubscription (`PredictGaussPoints_NumExpr5.py:958`).
- **Stop / `stop` markers.** Several debug paths still contain bare `stop`
  identifiers (e.g. `ClassWirtingerSolver.py:725`,
  `ClassJacobianAntenna.py:484`, `ClassJacobianAntenna.py:135`). These will
  raise `NameError` if reached — they are intentional development asserts.
- **Pickle-based options resume.** `MyPickle.Save/Load` use `pickle` directly
  (`Other/MyPickle.py:27–35`); loading a `last_killMS.obj` from an untrusted
  source executes arbitrary code.
- **The KAFCA Wirtinger paper is "to be published".** The README explicitly
  notes (`README.md:36`) that the EKF approach has not been formally
  published; users are pointed at arXiv:1403.6308 for "a similar approach".
- **No units tests for the Jacobian or solver kernels.** The only tests
  compare end-to-end FITS images.

## 15. Quick-start cheat-sheet

```bash
# Inspect available options
kMS.py -h

# Bare CohJones run with a sky-model NPY
kMS.py --MSName my.MS --SkyModel sky.npy \
       --SolverType CohJones --PolMode Scalar \
       --dt 5 --NCPU 8 --OutSolsName test1

# KAFCA + DicoModel + LM bootstrap
kMS.py --MSName mslist.txt \
       --SolverType KAFCA --PolMode Scalar \
       --BaseImageName image_DI \
       --InitLM 1 --InitLMdt 5 \
       --CovP 0.1 --CovQ 0.1 \
       --dt 1 --NChanSols 1 --NCPU 40 \
       --OutSolsName testKAFCA

# Subtract sources flagged kill=1 in the catalog
kMS.py --MSName my.MS --SkyModel sky.npy --kills "src1,src2" --SubOnly 1

# Free predict to a column (no gain applied)
kMS.py --MSName my.MS --SkyModel sky.npy --FreePredictColName MODEL_DATA

# Apply a previously-computed solution (no solve)
kMS.py --MSName my.MS --SkyModel sky.npy \
       --ExtSols testKAFCA --ApplyMode AP

# Smooth a solution in TEC+Amp
SmoothSols.py --SolsFileIn killMS.testKAFCA.sols.npz \
              --SolsFileOut killMS.testKAFCA.smooth.sols.npz \
              --InterpMode TEC,Amp
```
