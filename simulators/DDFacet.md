# DDFacet — Exhaustive Technical Reference

> A facet-based, w-projecting wide-field radio interferometric imager with
> direction-dependent CLEAN, written in Python with C/C++ OpenMP-accelerated
> gridding kernels and tight integration with the killMS direction-dependent
> calibration suite.

This document is a self-contained reference for the version of DDFacet
vendored under `simulators/DDFacet/` of the RadioSim project. All citations
point at concrete files and line ranges in that directory.

---

## 1. Overview

| Item | Value | Source |
|------|-------|--------|
| Project name | `DDFacet` | `simulators/DDFacet/pyproject.toml` line 11 |
| Tagline | "Facet-based radio astronomy continuum imager" | `simulators/DDFacet/pyproject.toml` line 13 |
| Version | `0.9.1.0` | `simulators/DDFacet/pyproject.toml` line 12 |
| Latest git tag | `v0.9.1` (head: `f7b0b2aa bump version`) | `git -C simulators/DDFacet log/tag` |
| License | GNU General Public License v2 (or any later) | `simulators/DDFacet/LICENSE.md` lines 1–7; `pyproject.toml` line 14 |
| Authors | Cyril Tasse (cyril.tasse@obspm.fr) | `pyproject.toml` line 60 |
| Maintainer | Benjamin Hugo (bhugo@sarao.ac.za) | `pyproject.toml` line 60 |
| Homepage | http://github.com/saopicc/DDFacet | `pyproject.toml` line 61 |
| Languages | Python (>=3.11,<3.13), C++14, C99 (legacy) | `pyproject.toml` line 16; `Gridder/CMakeLists.txt` lines 24–27 |
| Build backend | `scikit_build_core.build` over CMake | `pyproject.toml` lines 1–8 |
| Python entry | `DDF.py` (subcommand `DDF.py = DDFacet.__main__:ddf_main`) | `pyproject.toml` lines 67–73; `DDFacet/__main__.py` |

DDFacet is a faceted wide-field radio imager that decomposes the FoV into
many co-planar facet tangent planes, grids visibilities onto each one with a
w-projection convolution kernel that also folds in direction-dependent
Jones effects (DDEs), and performs CLEAN-style deconvolution either as a
classic Hogbom CLEAN, a multi-scale/multi-frequency Hybrid Matching
Pursuit (HMP/MSMF), Wide-Scale Multi-scale (WSCMS), MORESANE-based
multi-slice deconvolution, or the Sub-Space Deconvolution (SSD/SSD2)
genetic-algorithm island-based deconvolver. It is the imaging half of the
"saopicc" radio reduction stack (DDFacet + killMS + DynSpecMS).

The package is the canonical imager driving LOFAR LoTSS surveys, MeerKAT
deep continuum reductions, and similar high-dynamic-range projects where
direction-dependent calibration matters. It is purely a CPU/OpenMP code —
there is no GPU back-end. Speed comes from (a) shared-memory mmap arrays
exchanged between forked workers, (b) per-facet OpenMP parallelism inside
the C++ gridder, (c) baseline-dependent averaging (BDA) of visibilities
during gridding/degridding, and (d) Cyril Tasse's "AsyncProcessPool"
multi-process job scheduler.

### 1.1 Companion CLI binaries

`pyproject.toml` (lines 67–86) installs the following entry-point scripts
into the venv:

| Script | Module entry | Purpose |
|--------|--------------|---------|
| `DDF.py` | `DDFacet.__main__:ddf_main` | Main imager |
| `CleanSHM.py` | `DDFacet.__main__:cleanshm_main` | Wipe stale `/dev/shm` arrays from prior runs |
| `MemMonitor.py` | `DDFacet.__main__:memmonitor_main` | Sample memory of running DDFacet |
| `Restore.py` | `DDFacet.__main__:restore_main` | Re-restore images from cached residuals + DicoModel |
| `ClusterCat.py` | `SkyModel.__main__:clustercat_main` | Build clustered catalogs for killMS |
| `dsm.py` | `SkyModel.__main__:dsm_main` | Display sky model |
| `dsreg.py` | `SkyModel.__main__:dsreg_main` | DS9 region helpers |
| `ExtractPSources.py` | `SkyModel.__main__:extractpsources_main` | Point-source extraction |
| `Gaussify.py` | `SkyModel.__main__:gaussify_main` | Convert components to Gaussians |
| `MakeCatalog.py` | `SkyModel.__main__:makecatalog_main` | PyBDSF-driven catalog from FITS |
| `MakeMask.py` | `SkyModel.__main__:makemask_main` | Generate CLEAN mask FITS |
| `MakeModel.py` | `SkyModel.__main__:makemodel_main` | Build Voronoi/clustered killMS model |
| `MaskDicoModel.py` | `SkyModel.__main__:maskdicomodel_main` | Apply mask to DicoModel |
| `MyCasapy2bbs.py` | `SkyModel.__main__:mycasapy2bbs_main` | CASA component list → BBS |

`DDFacet/__main__.py` (lines 1–17) is the indirection layer that imports
the actual `DDF.py`, `CleanSHM.py`, `MemMonitor.py`, `Restore.py` modules
on demand.

### 1.2 Where it sits in RadioSim

The package is checked in as a git submodule at
`simulators/DDFacet/`. RadioSim itself is a measurement-equation-based
visibility *simulator*; DDFacet is the *imaging* counterpart used for
gridding/deconvolving simulated visibilities back to images for validation.
RadioSim does not import DDFacet at runtime.

---

## 2. License

DDFacet is released under the **GNU General Public License v2 (or, at
your option, any later version)** — see `simulators/DDFacet/LICENSE.md`.
The header is duplicated verbatim at the top of every Python and C/C++
source file, e.g. `DDFacet/DDF.py` lines 1–20. The pyproject classifier
declares:

```
"License :: OSI Approved :: GNU General Public License v2 (GPLv2)"
```

(`pyproject.toml` line 65). Copyright is held jointly by Cyril Tasse,
l'Observatoire de Paris, SKA South Africa / SARAO and Rhodes University,
2013–2024 (`README.rst` lines 4–5).

Practically: GPLv2 propagates to anything that links DDFacet's compiled
gridder library; if you redistribute a binary derivative you must offer
the source. Pure use through the CLI (Stimela, Slurm pipelines) does not
trigger redistribution obligations.

---

## 3. Repository layout

The submodule contains both the `DDFacet` Python package and the auxiliary
`SkyModel` package (used for catalog manipulation, masking and
clustering) plus build/CI scaffolding:

```
simulators/DDFacet/
├── apt.sources.list            # Ubuntu PPA list pulled into Docker image
├── DDFacet/                    # Main Python+C++ package
├── docker.2404                 # Ubuntu 24.04 (noble) Dockerfile
├── Jenkinsfile.sh              # Public CI pipeline (legacy)
├── LICENSE.md                  # GPL v2
├── migratenumpy.sh             # Helper: migrate to NumPy 2.x APIs
├── pyproject.toml              # Build system + deps + entry points
├── README.rst                  # Install/usage instructions
└── SkyModel/                   # Companion package
```

The `DDFacet` Python package itself:

```
DDFacet/
├── __init__.py
├── __main__.py                 # console script entry-point trampolines
├── Array/                      # Shared-memory + parallel array helpers
├── CMakeLists.txt              # scikit-build-core entry, descends into Gridder/
├── CleanSHM.py                 # Wipe stale /dev/shm arrays
├── cmake/                      # FindNumPy, FindRT etc.
├── compatibility.py            # Py2/Py3 shims
├── Data/                       # MS readers, Jones loaders, beam classes
├── DDF.py                      # MAIN: parses parset and runs ClassImagerDeconv
├── DDF_parallel.py             # legacy parallel driver
├── fits2png.py                 # FITS → PNG diagnostic helper
├── Gridder/                    # C++14 OpenMP gridder + JonesServer
│   └── old_c_gridder/          # Legacy C99 gridder
├── Imager/                     # Main numerical pipeline
├── MakeMovie.py                # Diagnostic movie generator
├── MemMonitor.py               # Memory-use sampler
├── Other/                      # AsyncProcessPool, logger, caches, ModColor…
├── Parset/                     # DefaultParset.cfg + ReadCFG + OptParse glue
├── plot_clean_logs.py          # Plot RMS/peak vs major cycle
├── report_version.py           # Produces version string from git or pyproject
├── Restore.py                  # Re-restore from cached DicoModel
├── SelfCal.py                  # Self-calibration helper
├── TensorFlowServerFork.py     # TF-Serving wrapper for Montblanc DFT
├── Tests/                      # Acceptance + unit tests
└── ToolsDir/                   # FFTW, mosaic, fitting and coordinate helpers
```

### 3.1 Per-folder commentary

- **`Array/`** — `NpShared.py` is the central wrapper around
  `SharedArray` (POSIX shm) for cross-process zero-copy arrays;
  `shared_dict.py` builds a hierarchical dict-of-shared-arrays;
  `NpParallel.py` is a tiny multiprocessing pool used for embarrassingly
  parallel numpy ops; `ModSharedArray.py` is the legacy interface;
  `ModLinAlg.py` exposes a non-negative least-squares helper used by
  HMP/SSD; `lsqnonneg.py` is the canonical Lawson-Hanson implementation.

- **`cmake/`** — bundled `FindNumPy.cmake`, `FindRT.cmake`, etc., used by
  the gridder build to locate POSIX `librt`, NumPy headers and pybind11.

- **`Data/`** — visibility and metadata I/O.
  `ClassMS.py` (1818 LOC) opens a single Measurement Set via
  python-casacore, exposes ANTENNA/SPECTRAL_WINDOW tables and tracks
  rephasing; `ClassDaskMS.py` is the dask-ms / xarray alternative
  used when `--Data-Dask=1`; `ClassVisServer.py` (1231 LOC) chunks data
  by hour or row-count, prepares weights and serves visibilities into
  shared memory; `ClassFITSBeam.py` / `ClassLOFARBeam.py` /
  `ClassATCABeam.py` / `ClassGMRTBeam.py` are E-Jones beam evaluators;
  `ClassJones.py` loads killMS solutions; `ClassBeamMean.py` averages
  the beam over time/PA; `ClassSmearMapping.py` builds the BDA blocks
  used by the gridder; `ClassSmoothJones.py` smooths the killMS Jones
  solutions for use as gridding DDEs; `ClassStokes.py` performs
  correlation-product ↔ Stokes conversion.

- **`Gridder/`** — the heart of the imager. `gridder.h` /
  `degridder.h` / `GridderSmearPols.cc` (371 LOC) implement the
  pybind11-exposed C++14 gridder/degridder kernels with full BDA, OpenMP,
  and on-the-fly Jones-multiplication. `JonesServer.cc/.h` evaluates
  per-direction, per-baseline, per-time Jones products inside the inner
  loop. `Arrays.cc` / `_pyArrays` exposes a few tight numpy-coercing
  helpers (`pySetOMPNumThreads`, ...). `Semaphores.cc` provides POSIX
  semaphore wrappers used to serialise grid increments across OpenMP
  threads gridding the same uv cell. `old_c_gridder/` keeps the original
  C99 gridder around for benchmarking but is not built by default.

- **`Imager/`** — pipeline brain.
  `ClassDeconvMachine.py` (2465 LOC) is `ClassImagerDeconv`: top-level
  orchestrator (`Init`, `main`, `GiveDirty`, `MakePSF`, `GivePredict`,
  `RestoreAndShift`).
  `ClassFacetMachine.py` (1959 LOC) and
  `ClassFacetMachineTessel.py` build, distribute and combine facet images
  using Voronoi tessellations and Hanning-mixed edges.
  `ClassDDEGridMachine.py` (1190 LOC) wraps the C++ gridder for one
  facet, manages convolution-function caches and w-plane lookup tables.
  `ClassImageDeconvMachine.py` (538 LOC) is the abstract base class for
  every minor-cycle algorithm; concrete implementations live under
  `HOGBOM/`, `MSMF/`, `WSCMS/`, `SSD/`, `SSD2/`, `MultiSliceDeconv/`,
  and `SASIR/`.
  `ClassCasaImage.py` writes/reads CASA images; `ClassFrequencyMachine.py`
  performs MFS frequency fits; `ClassGainMachine.py` tracks the CLEAN
  gain through stalls/divergences; `ClassImageNoiseMachine.py` builds
  noise/auto-mask images; `ClassImToGrid.py` performs image↔grid
  transforms; `ClassMaskMachine.py` aggregates external + auto + residual
  masks; `ClassModelMachine.py` is the abstract sky-model store
  (DicoModel) base class; `ClassMontblancMachine.py` is an optional DFT
  predictor; `ClassPSFServer.py` returns per-pixel PSFs;
  `ClassScaleMachine.py` is used by WSCMS; `ClassWeighting.py` computes
  Briggs/uniform/natural weights; `ModCF.py` builds w-projection /
  spheroidal CF tables; `ModModelMachine.py` is the small factory that
  picks the right model machine for a given `--Deconv-Mode`.

- **`Other/`** — utilities. `AsyncProcessPool.py` (the "APP" import)
  defines `APP.registerJobHandlers`, `APP.runJob`,
  `APP.startWorkers/terminate/shutdown` — the package's homegrown
  multiprocessing pool with shared-memory I/O. `CacheManager.py`
  handles persistent caches keyed by the parset section dicts.
  `ClassJonesDomains.py` covers solution-interval handling for killMS
  Jones tables. `ClassPrint.py` / `ClassTimeIt.py` / `ModColor.py` /
  `progressbar.py` are diagnostics. `Multiprocessing.py` provides
  `cleanupShm`/`cleanupStaleShm` — the auxiliary that wipes
  `/dev/shm/DDF.*` between runs. `MyPickle.py` is a thin pickle wrapper
  used to persist parsed options. `logger.py` configures the logging
  hierarchy with file + console appenders. `Exceptions.py` defines
  `UserInputError` and the post-mortem-pdb hook. `logo.py` prints the
  ASCII logo on startup.

- **`Parset/`** — configuration. `DefaultParset.cfg` (496 lines) is the
  canonical option list, with embedded `#type:`, `#options:`, `#metavar:`,
  `#cmdline_only:` directives. `ReadCFG.py` parses it into a
  `Parset(value_dict, attr_dict, sections)` object; `MyOptParse.py`
  converts each section/option into an `optparse` long flag of the form
  `--Section-Option`. `generate_stimela_schema.py` exports the parset to
  `ddfacet_stimela_inputs_schema.yaml` for Stimela cab definitions.

- **`ToolsDir/`** — numerical and I/O utilities.
  `ModFFTW.py` is the threaded pyFFTW wrapper used for image FFTs;
  `ModMosaic.py` does sky-image stitching; `ModFitPSF.py` fits the
  central PSF lobe to a Gaussian; `ModFitPoly2D.py` does 2-D polynomial
  fits; `ModCoord.py` converts between sky/pixel/uv coordinates;
  `ModRotate.py` implements w-projection rotations; `ModTaper.py`
  generates spheroidal/sigmoid uv tapers; `Gaussian.py` /
  `gaussfitter2.py` are PSF/source fitters; `GiveEdges.py` returns
  facet-overlap pixel slices; `ClassMovieMachine.py` produces movies for
  diagnostics; `casapy2bbs.py` is a CASA-component-list ↔ BBS skymodel
  converter; `ClassAdaptShape.py` reshapes oddly-shaped facet arrays;
  `ClassSpectralFunctions.py` defines the spectral models used by
  WSCMS/MSMF.

- **`Imager/HOGBOM/`** — classical Hogbom CLEAN.
  `ClassImageDeconvMachineHogbom.py` does the minor cycle;
  `ClassModelMachineHogbom.py` stores the (l,m,Stokes,freqfit) deltas.

- **`Imager/MSMF/`** — Hybrid Matching Pursuit (a.k.a. multiscale-
  multifrequency CLEAN, the default `--Deconv-Mode HMP`).
  `ClassImageDeconvMachineMSMF.py`, `ClassModelMachineMSMF.py`,
  `ClassMultiScaleMachine.py` (the per-scale basis-function cache).

- **`Imager/WSCMS/`** — wide-scale multi-scale CLEAN with auto-masking.

- **`Imager/SSD/`** and **`Imager/SSD2/`** — Sub-Space Deconvolution.
  Island-based GA / metropolis solver.
  `ClassImageDeconvMachineSSD`, `ClassArrayMethodSSD`,
  `ClassConvMachine`, `ClassInitSSDModelHMP` (HMP-warm-start),
  `ClassInitSSDModelMoresane`, `ClassIslandDistanceMachine`,
  `ClassMutate`, `ClassParamMachine`. SSD2 additionally has
  `ClassInitSSDModelMultiSlice` and `ClassTaylorToPower`. The `GA/` and
  `MCMC/` subfolders carry the genetic and Metropolis-Hastings solvers.

- **`Imager/MultiSliceDeconv/`** — wraps PyMORESANE
  (`MORESANE/`) and the legacy `Orieux/` Bayesian deconvolver.

- **`Imager/SASIR/`** — sparse-redundancy / iterative-shrinkage
  experimental algorithm.

- **`Imager/GA/`** — genetic algorithm primitives shared between the
  SSD modes (`ClassEvolveGA`, `ClassArrayMethodGA`).

- **`Tests/`** — the test suite. Subdivides into:
  * `FastUnitTests/` — pytest-style unit tests of small math kernels.
  * `ShortAcceptanceTests/` — end-to-end tests that should fit in CI.
  * `VeryLongAcceptanceTests/` — large-MS reference comparisons.
  * `DebugParsets/` — known-bad parsets for regression checks.
  * `FindDiffsCache.py` — a difference-checker over cached pickles.

- **`SkyModel/`** — companion package, installed alongside DDFacet.
  Contains:
  * `Sky/ClassSM.py` — main sky-model container and de/serialisers.
  * `Sky/ClassClusterTessel.py`, `ClassClusterDEAP.py`,
    `ClassClusterKMean.py`, `ClassClusterRadial.py`,
    `ClassClusterClean.py` — clustering strategies for killMS direction
    grids.
  * `Sky/ModBBS2np.py`, `ModSMFromNp.py`, `ModTigger.py`,
    `ModRegFile.py`, `ModVoronoi*.py` — converters between BBS, NumPy,
    Tigger and DS9 region formats.
  * `PSourceExtract/` — Gaussian / point-source fitters
    (`ClassGaussFit`, `ClassPointFit`, `ClassFitIslands`,
    `ClassIncreaseIsland`, `ModConvPSF`).
  * Tools: `ClusterCat.py`, `dsm.py`, `dsreg.py`,
    `ExtractPSources.py`, `Gaussify.py`, `MakeCatalog.py`,
    `MakeMask.py`, `MakeModel.py`, `MaskDicoModel.py`,
    `MyCasapy2bbs.py` — the user-facing scripts wired up via
    `pyproject.toml` console entry points.

---

## 4. Installation

### 4.1 Recommended: Stimela / Docker

`README.rst` lines 18–80 prescribe the official user path: install
[Stimela](https://github.com/SpheMakh/Stimela) ≥ 0.2.9 in a virtualenv,
`stimela pull` + `stimela build`, then call DDFacet through the
`cab/ddfacet` recipe step. The advantage is that Stimela's container
already bundles casacore, MeqTrees, killMS and DDFacet at compatible
versions.

```python
# README.rst lines 50–62
recipe.add("cab/ddfacet", "ddfacet_test", {
    "Data-MS": ["3C147.MS/SUBMSS/D147-LO-NOIFS-NOPOL-4M5S.MS"],
    "Output-Name": "testimg",
    "Image-NPix": 2048,
    "Image-Cell": 2,
    "Cache-Reset": True,
    "Freq-NBand": 3,
    "Weight-ColName": "WEIGHT",
    "Beam-Model": "FITS",
    "Beam-FITSFile": "'beams/JVLA-L-centred_$(corr)_$(reim).fits'",
    "Data-ChunkHours": 0.5,
    "Data-Sort": True
}, input=INPUT, output=OUTPUT, shared_memory="14gb",
   label="test_image:: Make a test image using ddfacet")
```

### 4.2 Pip from PyPI

`README.rst` lines 81–103:

```bash
virtualenv ddfacet
source ddfacet/bin/activate
pip install -U pip
pip install DDFacet
```

Optional extras (declared in `pyproject.toml`):

```bash
pip install "DDFacet[dft-support,moresane-support,testing-requirements,fits-beam-support,kms-support]"
```

`pyproject.toml` line 78–79 defines `dft-support` as
`montblanc >= 0.6.1, <= 0.7.4` (DFT predictor used by
`--RIME-ForwardMode=Montblanc`).

### 4.3 Native build from source

`pyproject.toml` lines 1–8 selects scikit-build-core as the build
backend, declaring requirements:

```
numpy >= 1.15.1, <= 2.3.2
setuptools >= 82.0.0
pybind11 >= 2.2.2
cython
cmake
scikit-build-core
```

The C++ gridder is built by descending from
`DDFacet/CMakeLists.txt` into `DDFacet/Gridder/CMakeLists.txt`. Build
options:

| CMake option | Default | Effect |
|--------------|---------|--------|
| `ENABLE_NATIVE_TUNING` | `ON` | adds `-march=native -mtune=native` (non-portable binaries) |
| `ENABLE_FAST_MATH` | `ON` | adds `-ffast-math` (breaks IEEE754 callbacks) |
| `ENABLE_PYTHON_2` | `OFF` | builds `_pyArrays27.so` / `_pyGridderSmearPols27.so` |
| `ENABLE_PYTHON_3` | `ON` | builds `_pyArrays3x.so` / `_pyGridderSmearPols3x.so` |

Source: `DDFacet/Gridder/CMakeLists.txt` lines 5–9.

The compiler flags applied (lines 24–27):

```
-fopenmp -std=c++14 -Wall -fmax-errors=1 -ggdb3 -pedantic -W -Wall
-Wconversion -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
${OpenMP_CXX_FLAGS} ${VECTORIZATION_FLAGS} ${TUNING_FLAGS}
${FAST_MATH_FLAGS}
```

Required CMake `find_package`s: `PythonInterp`, `PythonLibs`, `NumPy`,
`OpenMP`, `RT`, `pybind11` (lines 42–58). The `cmake/` folder ships
`FindNumPy.cmake` and `FindRT.cmake` so the build does not depend on
the system's NumPy CMake config.

To produce a portable binary, the README (lines 105–113) instructs:

```toml
cmake.define = {ENABLE_NATIVE_TUNING = "OFF",
                ENABLE_FAST_MATH = "OFF",
                ENABLE_PYTHON_2 = "OFF",
                ENABLE_PYTHON_3 = "ON"}
```

Outputs `_pyArrays3x.so` and `_pyGridderSmearPols3x.so` are installed
into `DDFacet/cbuild/Gridder/` (CMakeLists lines 95–99); these are the
modules `DDF.py` imports at line 81:

```python
from DDFacet.cbuild.Gridder import _pyArrays3x as _pyArrays
```

### 4.4 System (apt) dependencies

The full apt list lives in `simulators/DDFacet/docker.2404` (Ubuntu
24.04 / "noble"). The non-trivial ones (lines 29–64):

```
python3-virtualenv  python3-pip       python3-numpy   python3-dev
python3-tk          python3-casacore  libfftw3-dev    libfreetype6
libfreetype-dev     libpng-dev        libboost-all-dev libcfitsio-dev
libhdf5-dev         wcslib-dev        libatlas-base-dev liblapack-dev
libreadline6-dev    liblog4cplus-dev  libncurses5-dev libtinfo6
libncurses-dev      flex bison        libbison-dev    libqdbm-dev
libgsl-dev          casacore-data     casacore-dev    casacore-tools
make                cmake             g++-11/gcc-11   gfortran-11
git wget subversion rsync
```

The Dockerfile additionally builds-from-source: SOFA 20231011 (lines
89–98), Blitz++ 1.0.2 (lines 103–114), casarest 1.8.1 (lines
119–135), makems 1.5.6 (lines 140–153), MeqTrees Timba 1.11 (lines
161–185), and a vendored LOFARBeam fork
(`bennahugo/LOFARBeam@DDF_KMS_22.04`, lines 190–199). DDFacet itself
is `pip install`-ed last from `/src/DDFacet`.

### 4.5 Shared-memory and ulimit knobs

Because DDFacet exchanges arrays through `/dev/shm`, two system-level
knobs matter (`README.rst` lines 105–119; `DDF.py` lines 198–237):

```bash
sudo mount -o remount,size=100% /run/shm   # allow shm = full RAM
echo "*  -  memlock  unlimited" > /etc/security/limits.conf
sysctl -w vm.max_map_count=1000000
```

`DDF.py` checks `os.statvfs('/dev/shm')`, `sysctl vm.max_map_count`
and `resource.RLIMIT_MEMLOCK` at startup and prints warnings when any
of these are too small (lines 191–237).

### 4.6 CI: Jenkinsfile.sh

`Jenkinsfile.sh` lines 22–36 builds the Docker image and then runs
the test suite inside it:

```bash
docker build -t ddf.2404:$BUILD_NUMBER --no-cache=true -f docker.2404 .
docker run -m 100g --cap-add sys_ptrace --memory-swap=-1 --shm-size=150g \
           ... ddf.2404:$BUILD_NUMBER \
           -c "pynose -s --with-xunit \
               --xunit-file /workspace/nosetests.xml \
               /src/DDFacet/DDFacet/Tests"
```

Note the *150 GiB* shared-memory request — a realistic indication of
production memory pressure for large MS imaging.

---

## 5. Architecture

The runtime decomposes neatly into five layers:

```
                        ┌─────────────────────────────────────────┐
USER CLI →              │  DDF.py / report_version / __main__     │
                        └──────────────────┬──────────────────────┘
                                           │ DefaultParset.cfg →
                                           │ ReadCFG.Parset → MyOptParse
                                           ▼
                        ┌─────────────────────────────────────────┐
ORCHESTRATOR →          │ Imager/ClassDeconvMachine.py            │
                        │   class ClassImagerDeconv               │
                        │     .Init() / .main() / .GiveDirty()    │
                        │     .MakePSF() / .GivePredict()         │
                        └──┬──────────────┬───────────────┬───────┘
                           │              │               │
              ┌────────────▼──┐  ┌────────▼─────────┐  ┌──▼──────────────┐
              │ Data/         │  │ Imager/          │  │ Imager/         │
              │ ClassVisServer│  │ ClassFacetMachine│  │ Class*Deconv    │
              │ (chunked MS)  │  │ + Tessel         │  │ Machine* (HMP/  │
              │ + ClassMS     │  │ + DDEGridMachine │  │ Hogbom/SSD/...) │
              └─────┬─────────┘  └────────┬─────────┘  └──┬──────────────┘
                    │                     │               │
                    └──────────┬──────────┘               │
                               ▼                          │
                    ┌─────────────────────────┐           │
                    │ Gridder/ (C++14 OMP)    │           │
                    │ GridderSmearPols.cc     │           │
                    │ JonesServer.cc          │           │
                    │ DecorrelationHelper.cc  │           │
                    │ Arrays.cc → _pyArrays   │           │
                    └────────────┬────────────┘           │
                                 │                        │
                    ┌────────────▼────────────────────────▼─────┐
                    │ Array/NpShared, shared_dict (POSIX shm)   │
                    │ Other/AsyncProcessPool (APP)              │
                    │ Other/Multiprocessing (cleanupShm)        │
                    └────────────────────────────────────────────┘
```

`DDF.py:driver()` (lines 395–514) is the entry point. Steps:

1. Read parset via `read_options()` — which builds an `OP =
   MyOptParse(...)` from `DefaultParset.cfg` and merges in any user
   parset and `--Section-Option` flags (`DDF.py` lines 120–147).
2. Initialise logging, RNG, OpenMP thread count
   (`_pyArrays.pySetOMPNumThreads(NCPU)`, line 270).
3. Call `Multiprocessing.cleanupStaleShm()` to wipe `/dev/shm/DDF.*`
   from prior crashed runs (line 256).
4. Construct
   `Imager = ClassDeconvMachine.ClassImagerDeconv(GD=DicoConfig, ...)`
   with flags telling it which sub-pipelines to enable
   (`predict_only`, `data`, `psf`, `readcol`, `deconvolve`).
5. `Imager.Init()` builds:
   - `ClassVisServer` over the (possibly globbed) MS list,
   - the right `ClassImageDeconvMachine*` for `Deconv-Mode`,
   - `ClassMaskMachine`, `ClassImageNoiseMachine`,
   - one or two `ClassFacetMachineTessel` instances (data + PSF),
   - the `AsyncProcessPool` (APP), kicked off with
     `APP.startWorkers()`.
6. Dispatch by `Output-Mode`:
   - `Clean` → `Imager.main()` (major loop, calls
     `DeconvMachine.Deconvolve()` per major cycle).
   - `Dirty` → `Imager.GiveDirty()`.
   - `PSF` → `Imager.MakePSF()`.
   - `Predict`/`Subtract` → `Imager.GivePredict()`.
   - `RestoreAndShift` → `Imager.RestoreAndShift()`.
7. On any exit, `APP.terminate()` / `.shutdown()` and
   `Multiprocessing.cleanupShm()` are called via `atexit`.

The `Imager.Init()` body in `ClassDeconvMachine.py` lines 184–336 is
the canonical view of the wiring:

```python
self.VS = ClassVisServer.ClassVisServer(mslist, ColName=...,
                                         TChunkSize=DC["Data"]["ChunkHours"],
                                         GD=self.GD)
...
if self.GD["Deconv"]["Mode"] == "HMP":
    from DDFacet.Imager.MSMF import ClassImageDeconvMachineMSMF
    self.DeconvMachine = ClassImageDeconvMachineMSMF.ClassImageDeconvMachine(
        MainCache=self.VS.maincache, **MinorCycleConfig)
elif self.GD["Deconv"]["Mode"] == "SSD":
    ...
elif self.GD["Deconv"]["Mode"] == "WSCMS":
    ...
elif self.GD["Deconv"]["Mode"] == "Hogbom":
    ...
self.DeconvMachine.setMaskMachine(self.MaskMachine)
self.CreateFacetMachines()
self.VS.setFacetMachine(self.FacetMachine or self.FacetMachinePSF)
APP.startWorkers()
self.VS.CalcWeightsBackground()
self.FacetMachine and self.FacetMachine.initCFInBackground()
```

### 5.1 Data flow per major cycle

```
ClassVisServer.LoadNextVisChunk()   ← reads next time/freq chunk into shm
        │
        ▼
ClassFacetMachine.putChunk(DATA)    ← grids each facet via ClassDDEGridMachine
        │  (uses Gridder/_pyGridderSmearPols → C++ OMP gridder)
        ▼
ClassFacetMachine.FacetsToIm()      ← inverse FFT + facet stitching → image
        │
        ▼
ClassImageDeconv*.Update*Image*     ← hand image+PSF to deconv machine
        │
        ▼
ClassImageDeconv*.Deconvolve()      ← minor cycle ⇒ updates ModelMachine
        │
        ▼
ClassImageDeconv*.GiveModelImage()  ← model image
        │
        ▼
ClassFacetMachine.GiveModelImage    ← rasterise into facets
        │
        ▼
ClassFacetMachine.predictModel(DATA)← degrid (C++) → MODEL_DATA in shm
        │
        ▼
DATA -= MODEL_DATA                  ← residuals ready for next iter
```

The shared dictionaries `DicoDirty`, `DicoImagesPSF`, `DATA` (declared
on `ClassImagerDeconv` at lines 142–144) are the boundaries
across which all per-major-cycle data flows. They are
`shared_dict.SharedDict` instances backed by POSIX shm regions,
addressable by name from any worker process.

### 5.2 Process model

DDFacet runs one mother process plus N worker processes spawned by
`AsyncProcessPool` (the singleton `APP`). The mother process owns the
deconvolution machinery and the cache manager; workers handle gridder
calls, FFTs, weight computation and CF generation. Communication is
*not* via pickling: arrays are placed in `/dev/shm` via
`SharedArray` (`Array/NpShared.py`) and workers pick them up by name.
Job dispatch uses two `multiprocessing.Queue`s and a register of
job-handler classes (`APP.registerJobHandlers(self)`).

CPU pinning is configurable via `--Parallel-Affinity` (parset section
`[Parallel]`, lines 154–164):

```
Affinity = 1 | 2 | -1 | <list> | disable | enable_ht | disable_ht
MainProcessAffinity = disable | <core> | auto
```

The "disable_ht" mode autodetects NUMA topology and avoids both
hyper-threads of the same core.

---

## 6. The parset / command-line interface

DDFacet has a single, faithful translation between its INI-style parset
file and command-line `--Section-Option` flags. Both paths feed the same
`OP.DicoConfig` Python dict consulted everywhere as `self.GD["..."]`.

### 6.1 Parser implementation

`Parset/ReadCFG.py` (lines 27–85) parses `DefaultParset.cfg` with
`configparser.RawConfigParser`. Each value cell can carry inline
`#type:`, `#options:`, `#metavar:`, `#cmdline_only:`, `#no_cmdline:`
attributes. `parse_as_python(string)` (lines 45–55) tries `eval()` to
turn `"None"` → `None`, `"[1,2,3]"` → list, etc.; if that fails it
returns the literal string.

`Parset/MyOptParse.py` then builds the optparse parser. Each parset
section becomes an `OptionGroup` (with `_Help` as the docstring), and
every option becomes `--Section-Name`. Specifying a flag from the
command line overrides the parset value. Saving the active values back
out goes through `OP.ToParset(filename)`, called automatically by
`DDF.py` line 276 after the run is configured:

```python
OP.ToParset("%s.parset" % ImageName)
```

This means every successful run writes a `.parset` next to its outputs
that fully reproduces the run.

### 6.2 Subcommand-style modes

There are no formal subcommands; the *mode* is set by
`--Output-Mode`:

| `--Output-Mode` | Behaviour | Imager method |
|------|------|------|
| `Clean` (default) | Major/minor cycle deconvolution | `Imager.main()` |
| `Dirty` | Stop after producing dirty image | `Imager.GiveDirty()` |
| `PSF` | Compute and save the PSF only | `Imager.MakePSF()` |
| `Predict` | Read DicoModel/FITS, write predicted vis to MS column | `Imager.GivePredict()` |
| `Subtract` | Like Predict, then subtract from data | `Imager.GivePredict(subtract=True)` |
| `RestoreAndShift` | Re-restore from cached residual + DicoModel, optionally shifting facets | `Imager.RestoreAndShift()` |

Source: `DDF.py` lines 285–313.

### 6.3 Parset section reference

The complete option list lives in
`simulators/DDFacet/DDFacet/Parset/DefaultParset.cfg`. Each section
below is summarised with its high-level purpose and the most
operationally important options. Defaults are exactly as in the file.

#### `[Data]`
Visibility-data inputs. (lines 3–17)
- `MS=` (required) — single MS, comma-separated list, glob, or `*.txt`
  list. Per-MS DDID/FIELD selection via `foo.MS//D0:16//F0:2`.
- `ColName=CORRECTED_DATA` — MS data column to image.
- `ChunkHours=0` / `ChunkRows=0` — IO chunk size; `0` disables.
- `Sort=0` — re-sort by baseline-time (often faster).
- `Dask=0` — switch to the dask-ms IO backend (`ClassDaskMS.py`).

#### `[Predict]`
Output of predict mode. (lines 18–24)
- `ColName=None` — MS column to write predicted vis into. Empty
  disables.
- `MaskSquare=None`, `FromImage=None`, `InitDicoModel=None`,
  `Overwrite=1`.

#### `[Selection]`
Default DDID/FIELD/TaQL/channel/UV/time/antenna selection
(lines 26–38). `UVRangeKm=[0,2000]` is the default.

#### `[Output]`
Naming and which images to save (lines 40–73).
- `Mode=Clean`, `Name=image`, `Clobber=0`.
- `Images=DdPAMRIikemz` — default code combination. Letters:
  `D` intrinsic dirty, `d` apparent dirty, `P` PSF, `A` alpha map,
  `M`/`m` model, `R`/`r` residual, `I`/`i` restored,
  `N`/`n` Norm/NormFacets, `S` flux scale, `X` mixed-scale,
  `o` intermediate models, `e` intermediate residuals, `k`/`z`
  intermediate mask/noise, `g` intermediate dirty, `F`/`f` MFS
  restored. `all` saves everything.
- `Cubes=` — same syntax for cube versions.
- `StokesResidues=I` — residual Stokes products
  (`I|IQ|IV|QU|IQUV`).

#### `[SPIMaps]`
- `AlphaThreshold=15` (lines 74–75).

#### `[Image]`
- `NPix=5000` — image side length in pixels (lines 77–86).
- `Cell=5.` — cell size in arcseconds.
- `PhaseCenterRADEC=align` — `align` rephases all MSs to the first
  one's phase centre.
- `SidelobeSearchWindow=200` — PSF window size for sidelobe search.

#### `[Facets]`
Tessellation (lines 88–100).
- `NFacets=3` — number of facets along each side (so 9 facets total
  for `NFacets=3`).
- `DiamMax=180`, `DiamMin=0`.
- `MixingWidth=10` — Hanning mix at facet edges.
- `PSFOversize=1`, `PSFFacets=0`, `Padding=1.7`.
- `FluxPaddingAppModel=None`, `FluxPaddingScale=2.` — flux-dependent
  facet padding.

#### `[Weight]`
Imaging weights (lines 102–114).
- `ColName=WEIGHT_SPECTRUM`.
- `Mode=Briggs` (`Natural|Uniform|Robust|Briggs`).
- `MFS=1`, `Robust=0`, `SuperUniform=1`.
- `OutColName=None` — optionally write computed weights back.
- Sigmoid uv-taper knobs:
  `EnableSigmoidTaper=0`, `SigmoidTaperInnerCutoff=0`,
  `SigmoidTaperOuterCutoff=0`,
  `SigmoidTaperInnerRolloffStrength=0.5`,
  `SigmoidTaperOuterRolloffStrength=0.5`.

#### `[RIME]`
RIME forward / backward modes (lines 117–125).
- `Precision=S` (single).
- `PolMode=I|IQ|IU|IV|IQU|IQUV`.
- `FFTMachine=FFTW`.
- `ForwardMode=BDA-degrid|Classic|Montblanc`.
- `BackwardMode=BDA-grid|Classic`.
- `DecorrMode=`, `DecorrLocation=Edge|Center`.

#### `[CF]`
Convolution-function settings (lines 127–133).
- `OverS=11` — oversampling factor.
- `Support=7` — CF support size.
- `Nw=100` — number of w-planes; set 1 for AIPS-style faceting.
- `wmax=0` — maximum w in metres (0 = no clamp).

#### `[Comp]`
BDA compression (lines 135–150).
- `GridDecorr=0.02`, `DegridDecorr=0.02` — max BDA decorrelation.
- `GridFoV=Facet|Full`, `DegridFoV=Facet|Full`.
- `Sparsification=0` — list of decimation factors per major cycle.
- `BDAMode=1|2`.
- `BDAJones=0|grid|both`.

#### `[Parallel]`
Process / thread layout (lines 152–165).
- `NCPU=0` — 0 = all available, 1 = serial.
- `Affinity=1` — int / list / `disable` / `enable_ht` / `disable_ht`.
- `MainProcessAffinity=disable` — `auto` / int / `disable`.
- `MotherNode=localhost`.

#### `[Cache]`
Persistent caches under `<MS>.ddfcache` (lines 167–182).
- `Reset=0`, `SmoothBeam=auto`, `Weight=auto`, `PSF=auto`,
  `Dirty=auto`, `VisData=auto`, `LastResidual=1`.
- `Dir=` — directory for caches.
- `DirWisdomFFTW=~/.fftw_wisdom`, `ResetWisdom=0`.
- `CF=1`, `HMP=0` — cache convolution functions / HMP basis.

#### `[Beam]`
E-Jones primary-beam application (lines 183–250).
- `Model=None|LOFAR|FITS|GMRT|ATCA`.
- `At=facet|tessel`.
- `LOFARBeamMode=AE|A`.
- `NBand=0`, `CenterNorm=False`.
- `Smooth=False`, `SmoothNPix=11`, `SmoothInterpMode=Linear|Log`.
- `FITSFile=beam_$(corr)_$(reim).fits` — pattern with substitutions
  `$(corr)/$(xy)`, `$(reim)`, `$(realimag)`, `$(stype)`. Stations
  can be type-specialised via JSON config (see lines 200–235 of the
  parset for the full schema).
- `FITSFeed=None|xy|XY|rl|RL`, `FITSFeedSwap=False`.
- `DtBeamMin=5`, `FITSParAngleIncDeg=5`.
- `FITSLAxis=-X`, `FITSMAxis=Y`.
- `FITSFrame=altaz|altazgeo|equatorial|zenith`.
- `FeedAngle=0`, `ApplyPJones=0`, `FlipVisibilityHands=0`.
- `PointingCentre=PhaseDir`, `RotationDirection=North2East`.

#### `[Freq]`
MFS settings (lines 252–257).
- `BandMHz=0`, `DegridBandMHz=0`, `NBand=1`, `NDegridBand=0`.

#### `[DDESolutions]`
Apply killMS solutions during gridding (lines 259–275).
- `DDSols=`, `SolsDir=None`, `GlobalNorm=None`,
  `JonesNormList=AP`, `JonesMode=Scalar|Diag|Full`,
  `DDModeGrid=AP`, `DDModeDeGrid=AP`.
- Several deprecated knobs (`ScaleAmpGrid`, `Type`, `Scale`, `gamma`,
  `RestoreSub`, `ReWeightSNR`).

#### `[PointingSolutions]` (Montblanc only) — lines 277–280.

#### `[Deconv]`
Common minor-cycle controls (lines 282–302).
- `Mode=HMP|Hogbom|SSD|WSCMS|SSD2`.
- `MaxMajorIter=20`, `MaxMinorIter=20000`.
- `AllowNegative=1`, `Gain=0.1`.
- `FluxThreshold=0`, `CycleFactor=0`, `RMSFactor=0`,
  `PeakFactor=0.15`, `PrevPeakFactor=0`.
- `NumRMSSamples=10000`.
- `ApproximatePSF=0`.
- `PSFBox=auto|sidelobe|full|<int>`.

#### `[Mask]`
- `External=None`, `Auto=False`, `SigTh=10`,
  `FluxImageType=ModelConv|Restored` (lines 304–309).

#### `[Noise]`
- `MinStats=[60,2]`, `BrutalHMP=True` (lines 311–314).

#### `[HMP]`
HMP / multiscale-multifrequency (lines 316–334).
- `Alpha=[-1.,1.,11]`, `Scales=[0]`, `Ratios=[]`,
  `NTheta=6`, `SolverMode=PI|NNLS`.
- `AllowResidIncrease=0.1`, `MajorStallThreshold=0.8`.
- `Taper=0`, `Support=0`.
- `PeakWeightImage=None`, `Kappa=0`,
  `OuterSpaceTh=2.`, `FractionRandomPeak=None`.

#### `[Hogbom]`
- `PolyFitOrder=4`, `LinearPeakfinding=Joint|Separate` (lines 336–338).

#### `[WSCMS]`
Wide-Scale Multi-scale CLEAN (lines 340–359).
- `NumFreqBasisFuncs=4`, `MultiScale=True`, `MultiScaleBias=0.55`.
- `ScaleBasis=Gauss`, `Scales=None`, `MaxScale=250`.
- `NSubMinorIter=250`, `SubMinorPeakFact=0.85`,
  `MinorStallThreshold=1e-7`, `MinorDivergenceFactor=1.3`.
- `AutoMask=True`, `AutoMaskThreshold=None`, `AutoMaskRMSFactor=3`.
- `CacheSize=3`, `Padding=1.2`.

#### `[Montblanc]`
DFT predictor (lines 361–368).
- `TensorflowServerTarget=`, `LogFile=None`, `MemoryBudget=4.0`,
  `LogLevel=WARNING`, `SolverDType=double|single`,
  `DriverVersion=tf`.

#### `[SSDClean]` / `[SSD2]`
SSD/SSD2 island-based deconvolution (lines 370–391).
- `Parallel=True`, `IslandDeconvMode=GA|Moresane|Sasir|...`.
- `SSDSolvePars=[S,Alpha]`, `SSDCostFunc=[Chi2,MinFlux]`.
- `BICFactor=0.`, `ArtifactRobust=False`,
  `ConvFFTSwitch=1000`, `NEnlargePars=0`, `NEnlargeData=2`,
  `RestoreMetroSwitch=0`.
- `MinMaxGroupDistance=[10,50]`, `MaxIslandSize=0`,
  `InitType=HMP`.
- SSD2 extras: `PolyFreqOrder=2`, `SolvePars=[Poly]`,
  `InitType=[HMP,MultiSlice:Orieux]`, `NLastCyclesDeconvAll=1`.

#### `[MultiSliceDeconv]`
- `Type=MORESANE|Orieux`, `PolyFitOrder=2` (lines 393–395).

#### `[GAClean]`
GA-based island deconv (lines 397–412).
- `NSourceKin=50`, `NMaxGen=50`, `MinSizeInit=10`.
- HMP warm-start params:
  `AlphaInitHMP=[-4.,1.,6]`, `ScalesInitHMP=[0,1,2,4,8,16,24,32]`,
  `GainInitHMP=0.1`, `RatiosInitHMP=[]`, `NThetaInitHMP=4`,
  `MaxMinorIterInitHMP=10000`, `AllowNegativeInitHMP=False`,
  `RMSFactorInitHMP=3.`, `ParallelInitHMP=True`, `NCPU=0`.

#### `[MORESANE]`
- `NMajorIter=200`, `NMinorIter=200`, `Gain=0.1`,
  `ForcePositive=True`, `SigmaCutLevel=1` (lines 415–421).

#### `[Log]`
- `Memory=0`, `Boring=0`, `Append=0` (lines 423–427).

#### `[Debug]`
- `PauseWorkers=0`, `FacetPhaseShift=[0.,0.]`,
  `PrintMinorCycleRMS=0`, `DumpCleanSolutions=0`,
  `DumpCleanPostageStamps=`, `CleanStallThreshold=0`,
  `MemoryGreedy=1`, `APPVerbose=0`, `Pdb=auto|always|never`
  (lines 429–440).

#### `[Misc]`
- `RandomSeed=None`, `ParsetVersion=0.2`,
  `ConserveMemory=0`, `IgnoreDeprecationMarking=0` (lines 442–449).

#### `[katdal]`
- `ApplyCal=default|all|<csv>` (lines 451–454, only effective when
  `--Data-Dask=1` and the input is a katdal database).

### 6.4 Parset attributes (meta)

The header of `DefaultParset.cfg` (lines 455–496) explains the
attribute mini-DSL used inside option comments:

- `#type:TYPE` forces a specific Python type (`bool`, `str`, `float`,
  `int`, etc.). Without this, `eval()` is tried, then string fallback.
- `#options:A|B|C` declares an enum.
- `#metavar:VAR` shapes the help text.
- `#cmdline_only:1` blocks parset overrides (used for `Output-Clobber`).
- `#no_cmdline:1` blocks command-line overrides
  (used for `Misc-ParsetVersion`).

---

## 7. File-by-file breakdown

### 7.1 Top-level Python drivers

#### `simulators/DDFacet/DDFacet/DDF.py` (519 LOC)
The single command-line entry point. Public functions:

- `read_options() -> MyOptParse.MyOptParse` — builds the option
  parser from `DefaultParset.cfg`, reads arguments, persists OP via
  `MyPickle.Save(OP, "last_DDFacet.obj")` (`DDF.py` lines 120–147).
- `main(OP, messages)` — once options are set, sets up logging,
  validates SHM/sysctl, builds `ClassImagerDeconv`, and dispatches by
  `Output-Mode` (lines 154–313).
- `driver()` — top-level wrapper called from
  `DDFacet.__main__:ddf_main`. Handles parset discovery via
  positional argument, parset versioning / migration via
  `Parset.update_values(TestParset, newval=False)` and
  `TestParset.migrated`, exception routing including `KeyboardInterrupt`,
  `UserInputError`, `WorkerProcessError`, generic exceptions with
  pdb-on-error, and final cleanup (lines 395–514).
- `__main__` guard at line 516; the file is *not* itself an
  entry-point script — that role is taken by `DDFacet/__main__.py`.

OpenMP and parallelism setup (lines 38, 268–272):

```python
os.environ["OMP_NUM_THREADS"] = "1"  # before any numerical import
...
ncpu = int(DicoConfig["Parallel"]["NCPU"] or psutil.cpu_count())
_pyArrays.pySetOMPNumThreads(ncpu)
NpParallel.NCPU_global = ModFFTW.NCPU_global = ncpu
numexpr.set_num_threads(ncpu)
```

#### `DDFacet/__main__.py`
Five trampoline functions exposing the runnable scripts via
`pyproject.toml` `[project.scripts]`. Each just imports and calls the
real module's `driver()`.

#### `DDFacet/CleanSHM.py`
CLI helper that `unlink()`s every `/dev/shm/DDF.*` segment left over
from a crashed prior run. Wraps `Multiprocessing.cleanupShm()`.

#### `DDFacet/MemMonitor.py`
Standalone process-memory sampler; can be attached to a running PID
and produces a CSV/PNG of RSS over time.

#### `DDFacet/Restore.py`
Re-runs only the restoration step from a cached `.DicoModel` and
residual image, useful after manually editing the model.

#### `DDFacet/SelfCal.py`
Self-calibration helper; uses killMS for solutions and DDFacet for
imaging in a closed loop.

#### `DDFacet/MakeMovie.py`, `fits2png.py`, `plot_clean_logs.py`
Diagnostic helpers — not part of the core run pipeline.

#### `DDFacet/report_version.py`
Returns the version string seen at startup (`DDF.py` line 108);
prefers `git describe` when run from a checkout, else falls back to
the value baked in at install time from
`importlib.metadata.version("DDFacet")`.

#### `DDFacet/compatibility.py`
Tiny Py2/Py3 shim — exports `range` and a few helpers consistently.
The top of every Python module imports `from DDFacet.compatibility
import range`. Although Python 2 support has been formally dropped
(see `DDF.py` lines 418–422), the shim is still imported.

#### `DDFacet/TensorFlowServerFork.py`
Forks a TensorFlow Serving subprocess for Montblanc DFT prediction
when `--RIME-ForwardMode=Montblanc` and
`--Montblanc-TensorflowServerTarget` is set.

### 7.2 `Parset/` — configuration loader

| File | Purpose |
|------|---------|
| `DefaultParset.cfg` | The single source of truth for every option |
| `ReadCFG.py` | Reads `.cfg` files, parses inline attributes, type-coerces values, and exposes `value_dict`/`attr_dict`/`sections`. Provides `Parset.update_values(other, newval=False)` for parset migration. |
| `MyOptParse.py` | Builds an `optparse.OptionParser` from a Parset, exposes `OP.DicoConfig`, `OP.Print`, `OP.ToParset(filename)`, `OP.GiveArguments`. |
| `generate_stimela_schema.py` | Emits `ddfacet_stimela_inputs_schema.yaml` for Stimela's cab definition. |
| `ddfacet_stimela_inputs_schema.yaml` | Auto-generated cab schema. |
| `ddfacet_stimela_inputs_tweaks.yaml` | Manual overlays applied on top of the auto-generated schema. |
| `test_recipe.yaml` | Sample Stimela recipe used as a smoke test. |
| `ParsetChanges` | Migration notes describing parset-version transitions. |

### 7.3 `Imager/` — pipeline brain

| File / dir | Class(es) | Role |
|------|------|------|
| `ClassDeconvMachine.py` | `ClassImagerDeconv` | Top-level orchestrator (Init / main / GiveDirty / MakePSF / GivePredict / RestoreAndShift). |
| `ClassFacetMachine.py` + `ClassFacetMachineTessel.py` | `ClassFacetMachine`, `ClassFacetMachineTessel` | Tessellate the FoV, dispatch per-facet `ClassDDEGridMachine` calls, manage facet grids in shared memory, FFT and stitch. |
| `ClassDDEGridMachine.py` | `ClassDDEGridMachine` | Wraps the C++ gridder for a single facet, includes BDA mapping and per-facet DDE Jones lookups. |
| `ClassImageDeconvMachine.py` | `ClassImageDeconvMachine` (ABC) | Base class with `setMaskMachine`, `SetDirtyPSF`, `FindPSFExtent`. Concrete subclasses live under `HOGBOM/`, `MSMF/`, `WSCMS/`, `SSD/`, `SSD2/`, `MultiSliceDeconv/`, `SASIR/`. |
| `ClassCasaImage.py` | `ClassCasaImage`, `FileToArray` | Read/write CASA images and FITS, embed history. |
| `ClassFrequencyMachine.py` | `ClassFrequencyMachine` | MFS spectral fits per pixel/island. |
| `ClassGainMachine.py` | `ClassGainMachine` | Adaptive CLEAN gain (singleton, instantiated by `ClassImagerDeconv.__init__`). |
| `ClassImageNoiseMachine.py` | `ClassImageNoiseMachine` | Per-pixel noise / brutal-restored image used for auto-mask. |
| `ClassImToGrid.py` | `ClassImToGrid` | Image ↔ grid resampling helper. |
| `ClassMaskMachine.py` | `ClassMaskMachine` | Combines external + auto + residual masks (`Mask_{i+1} = ExternalMask | ResidualMask | Mask_i`). |
| `ClassModelMachine.py` + `ModModelMachine.py` | `ClassModelMachine` (ABC) + `ClassModModelMachine` (factory) | DicoModel store; the factory in `ModModelMachine.py` returns the right concrete sub-class for `Deconv-Mode`. |
| `ClassMontblancMachine.py` | `ClassMontblancMachine` | Optional DFT predictor via Montblanc / TF. |
| `ClassPSFServer.py` | `ClassPSFServer` | Returns local PSF for a given pixel — important for direction-dependent CLEAN. |
| `ClassScaleMachine.py` | `ClassScaleMachine` | WSCMS scale-bias bookkeeping. |
| `ClassWeighting.py` | `ClassWeighting` | Computes Briggs/uniform/natural weights, with optional sigmoid uv-tapers. Calls into `_pyGridderSmearPols`. |
| `ModCF.py` | functions | Spheroidal / w-projection convolution-function generation. |
| `HOGBOM/` | `ClassImageDeconvMachineHogbom`, `ClassModelMachineHogbom` | Classical Hogbom CLEAN. |
| `MSMF/` | `ClassImageDeconvMachineMSMF`, `ClassMultiScaleMachine`, `ClassModelMachineMSMF` | Hybrid Matching Pursuit (the default `--Deconv-Mode HMP`). |
| `WSCMS/` | `ClassImageDeconvMachineWSCMS`, `ClassModelMachineWSCMS` | Wide-Scale Multi-scale CLEAN with auto-masking and basis-function caching to disk. |
| `SSD/` and `SSD2/` | `ClassImageDeconvMachineSSD`, `ClassArrayMethodSSD`, `ClassConvMachine`, `ClassInitSSDModelHMP`, `ClassInitSSDModelMoresane`, `ClassIslandDistanceMachine`, `ClassMutate`, `ClassParamMachine`, `ClassModelMachineSSD` (+ `ClassInitSSDModelMultiSlice`, `ClassTaylorToPower` for SSD2), `GA/`, `MCMC/` | Sub-Space Deconvolution: per-island GA / Metropolis fits with pluggable initialisations. |
| `MultiSliceDeconv/` | `ClassImageDeconvMachineMultiSlice`, `ClassModelMachineMultiSlice`, `MORESANE/`, `Orieux/` | Slice-by-slice deconvolution wrapping PyMORESANE or the legacy Orieux Bayesian solver. |
| `SASIR/` | `ClassSasir`, `TrySasirDeconv` | Iterative-shrinkage / sparse algorithm (experimental). |
| `GA/` | `ClassEvolveGA`, `ClassArrayMethodGA` | Generic GA primitives shared by SSD modes. |

The minor-cycle abstract base class signature
(`Imager/ClassImageDeconvMachine.py` lines 36–73):

```python
class ClassImageDeconvMachine():
    def __init__(self, Gain=0.3, MaxMinorIter=100, NCPU=6,
                 CycleFactor=2.5, GD=None):
        ...
    def setMaskMachine(self, MaskMachine): ...
    def SetDirtyPSF(self, Dirty, PSF): ...
    def FindPSFExtent(self, Method="FromBox"): ...
```

Concrete subclasses must implement at least
`Init`, `Update*`, `Deconvolve()`, `GiveModelImage()`,
`ToFile/FromFile()` for DicoModel I/O.

### 7.4 `Data/` — measurement-set + Jones loaders

| File | Class(es) | Role |
|------|------|------|
| `ClassMS.py` (1818 LOC) | `ClassMS`, `expandMSList` | Open a single MS via python-casacore; expose ANTENNA, FIELD, SPECTRAL_WINDOW, POLARIZATION, FEED tables; rephase to a common centre via `ToolsDir.ModRotate`; track flag fractions. |
| `ClassDaskMS.py` | `ClassDaskMS` | Alternative IO backend using dask-ms / xarray when `--Data-Dask=1`. Also handles katdal-format inputs. |
| `ClassVisServer.py` (1231 LOC) | `ClassVisServer` | Multi-MS, chunk-aware visibility producer. Maintains a `CacheManager` (`<MS>.ddfcache`), pre-fetches chunks in a background APP job, exposes `LoadNextVisChunk`, `collectLoadedChunk`, `FreqBandCenters`, `FreqBandChannelsDegrid`, `RefFreq`. |
| `ClassData.py` | `ClassData` | Lower-level wrapper around per-MS data dicts (legacy). |
| `ClassStokes.py` | `ClassStokes` | Stokes ↔ correlation-product matrix; used to translate `--RIME-PolMode` into the right gridder template. |
| `ClassJones.py` | `ClassJones` | Loads killMS solutions (`SolsDir/<MSName>/killMS.<SolsName>.sols.npz`), interpolates over time/frequency. |
| `ClassSmoothJones.py` | `ClassSmoothJones` | Smooth Jones-table interpolation used in gridding. |
| `ClassBeamMean.py` | `ClassBeamMean` | Time/PA-averaged beam used by `--Beam-Smooth=1`. |
| `ClassFITSBeam.py` | `ClassFITSBeam` | Generic 8-cube FITS-beam model. |
| `ClassLOFARBeam.py` | `ClassLOFARBeam` | LOFAR Element + Array beam. Requires the `LOFARBeam` C++ library. |
| `ClassATCABeam.py` | `ClassATCABeam` | Hard-coded ATCA primary beam. |
| `ClassGMRTBeam.py` | `ClassGMRTBeam` | Hard-coded uGMRT primary beam. |
| `ClassSmearMapping.py` | `ClassSmearMapping` | Builds the BDA `SmearMapping` integer block-index array consumed by the C++ gridder. Two modes: `BDAMode=1` (Cyril) and `BDAMode=2` (Oleg). |
| `PointingProvider.py` | `PointingProvider` | Pointing-offset interpolator from CSV (used only with Montblanc). |
| `sidereal.py` | functions | Pure-python sidereal-time helpers. |

### 7.5 `Gridder/` — C++14 pybind11 gridder

| File | Role |
|------|------|
| `gridder.h` (381 LOC) | Templated gridder kernel: per-correlation read/multiply-accumulate policies (`Read_4Corr`, `Read_2Corr_Pad`, `Read_1Corr_Pad`; `Mulaccum_*`), Stokes templates from `Stokes.h`, and an OpenMP-parallel inner loop that handles BDA blocks and on-the-fly Jones multiplication via `JonesServer`. |
| `degridder.h` (270 LOC) | Templated degridder kernel: same policy pattern, computes model visibilities from a facet grid plus DDEs. |
| `GridderSmearPols.cc` (371 LOC) | pybind11 module that exposes `pyGridderSmearPols`, `pyDeGridderSmearPols`, `pySetSemaphores`, `pyDeleteSemaphore`. Also exposes `pySetOMPNumThreads` (in `Arrays.cc`). |
| `JonesServer.cc/.h` (307+145 LOC) | Maintains per-direction, per-time, per-channel Jones products; supports JonesMode `Scalar|Diag|Full`, `DDModeGrid` and `DDModeDeGrid` (`A`, `P`, or `AP`). |
| `DecorrelationHelper.cc/.h` | Computes time/freq decorrelation factors over each BDA block per `--RIME-DecorrMode`. |
| `Semaphores.cc/.h` | POSIX semaphore wrappers used to serialise grid-cell increments across OpenMP threads. 3373 named semaphores are pre-allocated by `ClassFacetMachine.setup_semaphores` (line 162). |
| `Arrays.cc` | Misc numpy helpers; exposes `pySetOMPNumThreads(n)`. |
| `Stokes.h` | Templated Stokes-grid functor: I, IQ, IU, IV, IQU, IV, IQUV. |
| `CorrelationCalculator.h` | Inline Stokes-from-correlations math. |
| `common.h` | Aliases (`fcmplx = complex<float>`, `dcmplx = complex<double>`, `dcMat = array<dcmplx,4>`), constant tables. |
| `old_c_gridder/` | Legacy C99 gridder kept as a "Classic" backend (`--RIME-BackwardMode=Classic`); also produces `_pyGridderSmearPolsClassic{27,3x}.so`. Files: `Gridder.c/.h`, `GridderSmearPols.c/.h`, `JonesServer.c`, `Matrix.c`, `Constants.h`, `Tools.h`, `Stokes.h`, `Semaphores.h`. |

The pybind11 entry signature (from `gridder.h` lines 78–99):

```cpp
template<policies::ReadCorrType readcorr,
         policies::MulaccumType   mulaccum,
         policies::StokesGridType stokesgrid,
         typename accum_grid_type>
void gridder(py::array_t<std::complex<accum_grid_type>>& grid,
             const py::array_t<std::complex<float>>&     vis,
             const py::array_t<double>&                  uvw,
             const py::array_t<bool>&                    flags,
             const py::array_t<float>&                   weights,
             py::array_t<double>&                        sumwt,
             bool                                        dopsf,
             const py::list& Lcfs,  const py::list& LcfsConj,
             const py::array_t<double>& Winfos,
             const py::array_t<double>& increment,
             const py::array_t<double>& freqs,
             const py::list& Lmaps, py::list& LJones,
             const py::array_t<int32_t>& SmearMapping,
             const py::array_t<bool>&    Sparsification,
             const py::list& LOptimisation, const py::list& LSmearing,
             const py::array_t<int>& np_ChanMapping,
             const vector<string>& expstokes);
```

The number of correlations gridded (1, 2 or 4) is selected by the
`readcorr` template parameter; the Stokes products imaged are picked
by the `stokesgrid` template parameter; the inner loop is OpenMP
parallel across BDA blocks. Concurrent grid increments to the same
uv cell are serialised by hashing the cell to one of 3373
preallocated POSIX semaphores (see `Gridder/Semaphores.cc`).

### 7.6 `Other/` — utilities

| File | Role |
|------|------|
| `AsyncProcessPool.py` | The `APP` singleton: a custom multiprocessing pool with shared-memory I/O, job counters, named registered handlers, and graceful Ctrl-C/timeout handling. |
| `CacheManager.py` | Per-section cache keyed by parset dicts. Files live in `<MS>.ddfcache/<section>.<hash>`. |
| `ClassJonesDomains.py` | Solution-interval / domain bookkeeping for killMS Jones. |
| `ClassPrint.py`, `ClassTimeIt.py`, `ModColor.py`, `progressbar.py`, `MyImshow.py`, `terminal.py` | Diagnostic helpers (timers, colors, progress bars, terminal sizing). |
| `Multiprocessing.py` | `cleanupShm`, `cleanupStaleShm`, `getShmName` — wipe and name `/dev/shm/DDF.<pid>.<name>` segments. |
| `MyPickle.py`, `MyLogger.py` | Minor wrappers. |
| `logger.py` | Sets up the `DDFacet` logger hierarchy with file + console appenders, memory tracing (via `psutil.Process().memory_info()`), boring-mode silencing. |
| `Exceptions.py` | `UserInputError`, `WorkerProcessError`, `enable_pdb_on_error`, `is_pdb_enabled`. |
| `logo.py` | Prints the ASCII art at startup. |
| `PrintList.py`, `PrintOptParse.py`, `reformat.py`, `ModProbeCPU.py`, `grepall.py`, `README-APP.md` | Misc helpers and inline docs. |

### 7.7 `Array/` — shared memory

| File | Role |
|------|------|
| `NpShared.py` | Thin wrapper around `SharedArray` (POSIX shm via `/dev/shm`). Exposes `zeros(name, shape, dtype)`, `DelArray`, `SizeShm`, `attach`. Source of cross-process zero-copy arrays. |
| `shared_dict.py` | `SharedDict`: a hierarchical dict whose values can be either Python objects (pickled) or shared arrays. Used by `ClassImagerDeconv.DicoDirty`, `DicoImagesPSF`, `DATA`. |
| `NpParallel.py` | Parallel maps over numpy arrays; used by image-plane operations. |
| `ModSharedArray.py` | Legacy interface to shm. |
| `ModLinAlg.py` | Wrapper around scipy's NNLS plus pseudo-inverse. |
| `lsqnonneg.py` | Lawson-Hanson NNLS, used by HMP `SolverMode=NNLS`. |
| `PrintRecArray.py` | Pretty-printing of casacore record arrays. |

### 7.8 `ToolsDir/` — numerics

| File | Role |
|------|------|
| `ModFFTW.py` | Threaded pyFFTW wrapper, exposes `NCPU_global` and `FFTW.fft()`/`ifft()` with cached plans persisted to `~/.fftw_wisdom`. |
| `ModMosaic.py` | Image-plane stitching for facet outputs. |
| `ModFitPSF.py` | Gaussian fit of PSF main lobe → `(BMAJ, BMIN, BPA)` and sidelobe peaks. |
| `ModFitPoly2D.py` | 2-D polynomial fitting. |
| `ModCoord.py` | RA/Dec ↔ (l, m) projections, with caching. |
| `ModRotate.py` | uvw rotation for re-phasing MSs to a common direction. |
| `ModTaper.py` | Spheroidal / Hanning / sigmoid uv-taper functions. |
| `Gaussian.py`, `gaussfitter2.py` | Lower-level Gaussian fitters. |
| `GiveEdges.py` | Compute pixel slices for facet/PSF overlap. |
| `GiveMDC.py` | Helper for measurement-direction-cosines. |
| `ClassAdaptShape.py` | Reshape FITS arrays from `(NCH,NPOL,NX,NY)` to `(NPIX,NPIX)` etc. |
| `ClassMovieMachine.py` | Diagnostic movie generation. |
| `ClassSpectralFunctions.py` | Polynomial / Taylor / power spectral models for WSCMS/MSMF. |
| `casapy2bbs.py` | CASA component list ↔ BBS skymodel converter. |
| `CatToFreqs.py` | Catalogue freq-binning helpers. |
| `fft_comparison.py` | Standalone benchmark of pyFFTW vs numpy FFT. |
| `findrms.py`, `rad2hmsdms.py`, `ModParset.py`, `ModToolBox.py` (`EstimateNpix`), `GeneDist.py` | Misc. |

### 7.9 `SkyModel/` companion package

Installed alongside DDFacet (same wheel) and providing the user-facing
catalogue and clustering scripts.

| File | Role |
|------|------|
| `__main__.py` | Entry-point trampolines for `ClusterCat.py`, `dsm.py`, `dsreg.py`, `ExtractPSources.py`, `Gaussify.py`, `MakeCatalog.py`, `MakeMask.py`, `MakeModel.py`, `MaskDicoModel.py`, `MyCasapy2bbs.py`. |
| `Sky/ClassSM.py` | Central `ClassSM` sky-model container; serialises to/from BBS, NumPy `npy`, Tigger `lsm.html`. |
| `Sky/ClassClusterTessel.py`, `ClassClusterDEAP.py`, `ClassClusterKMean.py`, `ClassClusterRadial.py`, `ClassClusterClean.py` | Clustering strategies for killMS direction grids. |
| `Sky/ClassMetricDEAP.py` | DEAP fitness for the genetic clustering. |
| `Sky/ModBBS2np.py`, `ModSMFromNp.py`, `ModTigger.py`, `ModRegFile.py`, `ModVoronoi.py`, `ModVoronoiToReg.py`, `ModKMean.py`, `DeapAlgo.py` | Sky-model converters and helpers. |
| `PSourceExtract/ClassFitIslands.py`, `ClassGaussFit.py`, `ClassIncreaseIsland.py`, `ClassIslands.py`, `ClassPointFit.py`, `ClassPointFit2.py`, `ModConvPSF.py`, `findrms.py`, `Gaussian.py`, `TestGaussFit.py` | Point/Gaussian source extraction (used by `MakeCatalog.py` and the SSD pipeline). |
| `Tools/ModFFTW.py`, `PolygonTools.py` | Internal helpers (some duplicated from DDFacet to avoid the cross-dep). |
| `Other/*.py` | Local copies of `ModColor`, `ModCoord`, `MyHist`, `MyLogger`, `MyPickle`, `progressbar`, `rad2hmsdms`, `reformat`, `terminal` — keeps SkyModel scripts usable when DDFacet is not yet on the path. |
| `Array/RecArrayOps.py` | Record-array helpers for catalogue manipulation. |
| `Test/` | Unit tests for the SkyModel utilities. |

---

## 8. Algorithms

### 8.1 Faceting and w-projection

DDFacet partitions the image plane into N×N facets (`--Facets-NFacets`)
and grids visibilities onto each facet using a per-facet tangent-plane
re-projection. The combination eliminates the wide-field "w-term" by
two complementary mechanisms:

1. **w-projection** within each facet: a stack of `--CF-Nw=100`
   convolution functions tabulated against w; each visibility's w
   value is mapped to the nearest plane and convolved into the grid
   with the corresponding spheroidal-weighted CF. The CF support is
   `--CF-Support=7` cells, oversampled by `--CF-OverS=11`.
2. **Faceting** itself reduces the maximum tangential w that the
   projection has to deal with on each facet by `1/NFacets`.

Setting `--CF-Nw=1` disables w-projection and reverts to AIPS-style
facet-only treatment (`ClassFacetMachine.py` lines 101–103). The
maximum w-coordinate gridded is clamped by `--CF-wmax`; visibilities
with `|w| > wmax` are dropped.

Facet edges are blended with a Gaussian roll-off of width
`--Facets-MixingWidth=10` pixels to avoid stitching seams. Facet
shapes can be regular square or Voronoi tessellations (the
`ClassFacetMachineTessel` derived class), and can be re-sized either
by hard min/max angular size (`--Facets-DiamMin/DiamMax`) or by
flux-dependent padding (`--Facets-FluxPaddingAppModel`).

### 8.2 Imaging weights

`Imager/ClassWeighting.py` (`class ClassWeighting`) computes the per-
visibility imaging weights according to `--Weight-Mode` and writes
them either to a shared-memory array (default) or to a named MS column
(`--Weight-OutColName`). Modes:

- **Natural**: weights = data `WEIGHT[_SPECTRUM]` only.
- **Uniform**: re-normalise so that each uv cell has equal weight.
  `--Weight-SuperUniform=N` enlarges the FoV used for the count by
  factor N, smoothing the resulting beam.
- **Robust** / **Briggs** (default): Briggs-style trade-off
  controlled by `--Weight-Robust` ∈ [-2, 2]; -2 ≈ uniform,
  +2 ≈ natural.
- **MFS** (`--Weight-MFS=1`): all channels are binned onto a single
  uv grid before computing weights, ensuring a uniform PSF across
  bands.

Optional uv-tapering: `--Weight-EnableSigmoidTaper=1` plus inner /
outer cutoffs in uv-wavelengths and corresponding rolloff strengths
applies a smooth sigmoid tapering function to the weights; useful
for emphasising a target spatial scale (small-scale or extended).

The actual cell-counting and weight-application is delegated to the
C++ `_pyGridderSmearPols` module for speed.

### 8.3 BDA — baseline-dependent averaging

For long observations and large FoVs the time-frequency averaging that
preserves a given decorrelation level varies enormously across
baselines (long baselines need short steps, short baselines tolerate
huge steps). `ClassSmearMapping` builds an integer `SmearMapping`
array that groups raw visibility samples into "BDA blocks" — one
block per (uv-cell, time-window, channel-window) — and the C++
gridder/degridder operates on blocks rather than samples. Two block-
construction modes exist:

- `--Comp-BDAMode=1` — Cyril's original mode; conservative.
- `--Comp-BDAMode=2` — Oleg's faster mode (see issue #319 in upstream
  for the precision trade-off).

The maximum decorrelation per block is bounded by
`--Comp-GridDecorr=0.02` (gridding) and `--Comp-DegridDecorr=0.02`
(degridding), evaluated either across the facet (`Facet`) or full FoV
(`Full`).

`--Comp-Sparsification=N1,N2,...` further accelerates initial major
cycles by randomly dropping all but `1/N_i` of the visibilities in
major cycle `i`, which is sufficient for low-fidelity sky-model
construction in the early cycles.

### 8.4 Deconvolution algorithms

`Imager/ClassImageDeconvMachine.py` is the abstract base class.
Concrete implementations:

#### Hogbom (`--Deconv-Mode Hogbom`)
`Imager/HOGBOM/ClassImageDeconvMachineHogbom.py`. The textbook Högbom
1974 CLEAN: locate the brightest residual pixel, subtract a scaled
PSF, repeat. EVPA-preserving polarised CLEAN is supported via
`--Hogbom-LinearPeakfinding=Joint|Separate`. Multi-frequency support
is via per-pixel polynomial spectral fits of order
`--Hogbom-PolyFitOrder=4`. Header docstring (lines 28–32) explicitly
describes itself as "the minimal reference interface of how to
incorporate new deconvolution algorithms into DDFacet".

#### HMP / MSMF (`--Deconv-Mode HMP`, default)
`Imager/MSMF/ClassImageDeconvMachineMSMF.py`. Hybrid Matching Pursuit:
generalises Hogbom CLEAN by representing the model as a sum of
multiscale basis functions (`--HMP-Scales=[0]`) at multiple spectral
indices (`--HMP-Alpha=[-1.,1.,11]`) and multiple position-angle
orientations (`--HMP-NTheta=6`). At each minor iteration the
algorithm picks the (scale, alpha, theta, position) that maximises
the matched-filter residual, fitted by either pseudo-inverse
(`--HMP-SolverMode=PI`) or non-negative least squares (`NNLS`).
Auto-bailout on divergence (`--HMP-AllowResidIncrease=0.1`),
major-cycle stall detection (`--HMP-MajorStallThreshold=0.8`), and
optional Tikhonov-style regularisation (`--HMP-Kappa=0`).

#### WSCMS (`--Deconv-Mode WSCMS`)
`Imager/WSCMS/`. Wide-Scale Multi-scale CLEAN with explicit scale-
dependent auto-masking. Frequency axis is expanded into
`--WSCMS-NumFreqBasisFuncs=4` basis functions (Taylor / power
polynomial). Scale kernels are Gaussians of FWHM listed in
`--WSCMS-Scales`, capped at `--WSCMS-MaxScale=250` pixels. A sub-
minor loop of length `--WSCMS-NSubMinorIter=250` cleans within the
currently selected scale until the peak drops below
`--WSCMS-SubMinorPeakFact=0.85` of the entry peak, then the major
loop re-selects. Stall detection
(`--WSCMS-MinorStallThreshold=1e-7`) and divergence rejection
(`--WSCMS-MinorDivergenceFactor=1.3`) prevent infinite loops.
Auto-masking: `--WSCMS-AutoMask=True` plus
`--WSCMS-AutoMaskRMSFactor=3` builds a scale-dependent threshold
mask. A small in-process LRU keeps `--WSCMS-CacheSize=3` scale
basis-function tables in memory before spilling to disk.

#### SSD / SSD2 (`--Deconv-Mode SSD|SSD2`)
`Imager/SSD/`, `Imager/SSD2/`. Sub-Space Deconvolution: identifies
emission "islands" in the residual image via thresholding +
morphological dilation (`ClassIslandDistanceMachine`), then fits each
island's pixels jointly using a chosen optimiser:

- `--SSDClean-IslandDeconvMode=GA` — DEAP-based genetic algorithm
  (`Imager/GA/`, `Imager/SSD/GA/`). `--GAClean-NSourceKin=50`
  individuals, `--GAClean-NMaxGen=50` generations.
- `IslandDeconvMode=Moresane` — wraps PyMORESANE inside each island.
- `IslandDeconvMode=Sasir` — sparse iterative-shrinkage.

Initial population is seeded from one of `--SSDClean-InitType` modes
(`HMP`, `MultiSlice:Orieux`, ...). Cost functions can combine χ² and
flux-minimisation (`--SSDClean-SSDCostFunc=[Chi2,MinFlux]`).

SSD2 differs in that it always works in a polynomial spectral
parametrisation (`--SSD2-PolyFreqOrder=2`,
`--SSD2-SolvePars=[Poly]`), supports multi-init seeding (`InitType=
[HMP,MultiSlice:Orieux]`), and offers control over how many trailing
major cycles deconvolve every island
(`--SSD2-NLastCyclesDeconvAll=1`).

#### MultiSlice (`--Deconv-Mode MultiSlice`)
`Imager/MultiSliceDeconv/`. Slice-by-slice deconvolution; each
spectral slice is deconvolved independently using either
`MORESANE` or `Orieux` (`--MultiSliceDeconv-Type`). Followed by a
joint polynomial fit of order `--MultiSliceDeconv-PolyFitOrder=2`
across slices.

#### MORESANE
Bundled wrapper around PyMORESANE (`Imager/MultiSliceDeconv/MORESANE/`).
Parameters in the parset under `[MORESANE]`: `NMajorIter`,
`NMinorIter`, `Gain`, `ForcePositive`, `SigmaCutLevel`.

### 8.5 Restoration

After deconvolution the model is convolved with a fitted clean beam
(elliptic Gaussian fitted to the central PSF lobe by
`ToolsDir/ModFitPSF.py`) and added to the residual image to produce
the restored image. The intrinsic restored image divides the apparent
restored image by √(JonesNorm) — the time-averaged primary-beam image
written as `*.Norm.fits` — yielding the absolute-flux-corrected
output. The image-letter codes in `--Output-Images` (`I`/`i`/`M`/`m`)
control which versions are written; `F`/`f` toggle MFS combinations.

### 8.6 Primary beam application

`Beam-Model` selects the beam evaluator:

- `LOFAR` — `Data/ClassLOFARBeam.py`, calls into the LOFARBeam C++
  library; modes `A` (array-only) or `AE` (array × element).
- `FITS` — `Data/ClassFITSBeam.py`, an 8-cube
  (`{re,im}` × {`xx,xy,yx,yy`}) FITS-pattern model. Supports
  heterogeneous arrays via `--Beam-FITSFile <patterns.json>`
  (DefaultParset.cfg lines 200–235). Frame can be `altaz`,
  `altazgeo`, `equatorial` or `zenith`. Re-evaluation cadence is set
  by `--Beam-DtBeamMin=5` minutes plus
  `--Beam-FITSParAngleIncDeg=5` degrees of PA increment.
- `GMRT` — analytic GMRT primary beam (`Data/ClassGMRTBeam.py`).
- `ATCA` — analytic ATCA primary beam (`Data/ClassATCABeam.py`).
- `None` — no E-Jones applied.

When beams are time-averaged for restoration, the smoothed beam is
optionally written via `--Beam-Smooth=1` and read back through
`Data/ClassBeamMean.py`.

### 8.7 Direction-dependent calibration

`--DDESolutions-DDSols=<name>` enables application of killMS Jones
solutions during gridding and degridding. Files are looked up at
`<SolsDir>/<MSName>/killMS.<DDSols>.sols.npz`. `JonesMode` chooses
how much of the matrix is applied (`Scalar`, `Diag`, `Full`) and
`DDModeGrid`/`DDModeDeGrid` switch between amplitude-only (`A`),
phase-only (`P`) or both (`AP`). At gridding time the C++
`JonesServer` interpolates the solution table to each visibility and
multiplies in-line (or per BDA block if `--Comp-BDAJones=grid|both`).

### 8.8 Multi-frequency synthesis

DDFacet treats MFS at two stages:

- **Imaging cube**: `--Freq-NBand` independent grids are produced
  during gridding. `--Freq-BandMHz` may instead be used to specify a
  fixed band width.
- **Degridding cube**: independently sized via `--Freq-NDegridBand`
  / `--Freq-DegridBandMHz`. `0` means degrid each channel.

The HMP, WSCMS and SSD2 deconvolvers fit per-pixel spectral models
(α-power-law for HMP, polynomial Taylor expansion for SSD2/WSCMS),
producing a `*.alpha.fits` map that is masked at
`--SPIMaps-AlphaThreshold=15` × residual RMS.

### 8.9 Dynamic spectrum and polarisation

`--RIME-PolMode` selects the imaged Stokes products. Only `I`
deconvolution is implemented; for `IQ`/`IU`/`IV`/`IQU`/`IQUV` the
package images all requested Stokes residuals but only deconvolves
Stokes I (the parset note "the imager does not perform deconvolution
on any Stokes products other than I — it only outputs residues" makes
this explicit, lines 69–72 of `DefaultParset.cfg`).

The `--Output-StokesResidues` knob lets you keep all Stokes residuals
even when cleaning only I; the residual cube is written to
`*.cube.residual.{Q,U,V}.fits`.

Dynamic-spectrum analysis is not part of DDFacet itself; it is the
job of the sister `DynSpecMS` package in the same `saopicc`
ecosystem.

---

## 9. Inputs and outputs

### 9.1 Inputs

| Input | How specified | Notes |
|------|------|------|
| Measurement set(s) | `--Data-MS` | Glob-able, list-able, with optional `//Dx//Fy` DDID/FIELD selectors. |
| MS visibility column | `--Data-ColName=CORRECTED_DATA` | Default reads `CORRECTED_DATA`; common alternatives `DATA`, `MODEL_DATA`. |
| Weight column | `--Weight-ColName=WEIGHT_SPECTRUM` | Falls back to `WEIGHT` when `WEIGHT_SPECTRUM` is absent. |
| External CLEAN mask | `--Mask-External=<file>.fits` | Boolean FITS image. |
| FITS beam pattern | `--Beam-FITSFile=...` | Eight-cube pattern, possibly with `$(stype)` station-type patterns and a JSON sidecar. |
| killMS solutions | `--DDESolutions-DDSols=<name>` + `--DDESolutions-SolsDir=<dir>` | Loaded by `Data/ClassJones.py`. |
| Initial model | `--Predict-InitDicoModel=<model>.DicoModel` | Continue from a previous model. |
| Initial FITS image | `--Predict-FromImage=<image>.fits` | Predict from a FITS image rather than a DicoModel. |
| Pointing-offset CSV | `--PointingSolutions-PointingSolsCSV` | Only with Montblanc DFT. |
| katdal database | `--Data-MS *.h5 --Data-Dask=1` | dask-ms / katdal IO path. |

### 9.2 Output products

All outputs share the prefix `--Output-Name=image` (default `image.*`).

| File | Created when |
|------|------|
| `<Name>.parset` | always (snapshot of the run's parset) |
| `<Name>.log` | always (full log) |
| `<Name>.dirty.fits` / `<Name>.cube.dirty.fits` | `d` in `--Output-Images` / `--Output-Cubes` |
| `<Name>.dirty.corr.fits` / `<Name>.cube.dirty.corr.fits` | `D` in codes (intrinsic) |
| `<Name>.psf.fits` | `P` |
| `<Name>.model.fits` / `<Name>.cube.model.fits` | `m` / `M` |
| `<Name>.app.convmodel.fits` / `<Name>.int.convmodel.fits` | `c` / `C` |
| `<Name>.residual.fits` / `<Name>.cube.residual.fits` | `r` / `R` (`Stokes` controlled by `--Output-StokesResidues`) |
| `<Name>.app.restored.fits` / `<Name>.int.restored.fits` | `i` / `I` |
| `<Name>.MFS.app.restored.fits` / `<Name>.MFS.int.restored.fits` | `f` / `F` |
| `<Name>.alpha.fits` | `A` |
| `<Name>.Norm.fits` / `<Name>.cube.Norm.fits` | `N` / cube `N` |
| `<Name>.NormFacets.fits` | `n` |
| `<Name>.S.fits` (flux scale) | `S` |
| `<Name>.X.*.fits` (mixed-scale) | `X` |
| `<Name>.Model_<i>.fits`, `<Name>.Residual_<i>.fits` | `o` / `e` (per-major-cycle intermediates) |
| `<Name>.Mask_<i>.fits`, `<Name>.NoiseMap_<i>.fits` | `k` / `z` |
| `<Name>.SmoothNorm.fits` / `<Name>.MeanSmoothNorm.fits` | `--Beam-Smooth=1` |
| `<Name>.DicoModel` | always (deconv model) |
| `<Name>.Metro.DicoModel` | SSD with Metropolis enabled |
| `<MS>.ddfcache/` | persistent cache directory next to each MS |

`<Name>.DicoModel` is a Python pickle of the active
`ClassModelMachine` — typically loaded back via `MakeMask.py`,
`Restore.py` or another `--Predict-InitDicoModel`.

### 9.3 Cache layout

`Other/CacheManager.py` writes per-section caches under
`<MS>.ddfcache/`. Cached items include:

- `Weight.<hash>` — computed visibility weights.
- `PSF.<hash>`, `Dirty.<hash>` — last dirty/PSF arrays.
- `VisData.<hash>` — pre-rephased visibility blocks.
- `CF.<hash>` — w-projection convolution-function tables.
- `HMP.<hash>` — HMP basis tables.
- `SmoothBeam.<hash>` — averaged beam.
- `LastResidual.<hash>` — last residual at end of last minor cycle.

Each item's hash is keyed on the relevant parset sub-dict, so any
parameter change that affects the result invalidates the entry.
`--Cache-Reset=1` wipes the lot. Per-product knobs
(`--Cache-PSF=auto|reset|off|force`, `--Cache-Dirty=auto|forcedirty|forceresidual`, ...) override individual items.

A second cache lives in `--Cache-DirWisdomFFTW=~/.fftw_wisdom` for
pyFFTW plan reuse; reset with `--Cache-ResetWisdom=1`.

---

## 10. Testing

The test layout under `DDFacet/Tests/`:

```
Tests/
├── FastUnitTests/                # Pure-Python unit tests, no MS
│   ├── TestFitter.py
│   ├── TestLibraries.py
│   └── TestStokesConverter.py
├── ShortAcceptanceTests/         # End-to-end runs that fit in CI
│   ├── ClassCompareFITSImage.py  # Common base class: builds a parset,
│   │                             # runs DDFacet, and compares the named
│   │                             # FITS images against a reference set
│   │                             # within `defineMaxSquaredError`.
│   ├── TestClean.py              # Default HMP CLEAN
│   ├── TestFacetPredict.py       # Predict mode
│   ├── TestHogbomClean.py        # Hogbom CLEAN
│   ├── TestLOFAR_J1329_p4729.py  # LOFAR DDF + killMS reference
│   ├── TestMontblancPredict.py   # Montblanc DFT predict
│   ├── TestOneMinorCycleSubtract.py
│   ├── TestSupernovaStokesV.py   # Stokes V imaging
│   ├── TestUltimateDeconvRealSolsSSD.py  # SSD with real killMS sols
│   ├── TestWeighting.py
│   ├── TestWidefieldDirty.py
│   └── TestWSCMS.py
├── VeryLongAcceptanceTests/      # Reference-quality long jobs
│   ├── Test3C147.py
│   ├── TestDEEP2.py
│   ├── TestDEEP2Montblanc.py
│   ├── TestDeepClean.py
│   ├── TestHogbomPolClean.py
│   └── TestSupernova.py
├── DebugParsets/                 # Known-bad parsets for regression
│   ├── ParsetDDFacet.Imager.txt
│   ├── ParsetDDFacet.JonesDefs.txt
│   ├── ParsetDDFacet.txt
│   ├── simms.sh                  # Synthesises a simple MS for the CI
│   ├── tdlconf.profiles
│   └── testxcen-f9-ddenorm.parset
└── FindDiffsCache.py             # Standalone tool: walk two ddfcache
                                  # directories and diff entries
```

### 10.1 How tests are driven

`Jenkinsfile.sh` (lines 22–36) shows the canonical CI invocation:

```bash
docker run -m 100g --shm-size=150g ... ddf.2404:$BUILD_NUMBER \
    -c "ln -s /test_data/beams /test_output/beams && \
        pynose -s --with-xunit --xunit-file /workspace/nosetests.xml \
        /src/DDFacet/DDFacet/Tests"
```

So tests use `pynose` (a maintained nose-2 fork). They depend on
two volume mounts:

- `/test_data` — read-only fixture MS + reference FITS images.
- `/test_output` — scratch directory for run outputs.

Two environment variables, set in the Dockerfile (`docker.2404` lines
8–9), control the paths in-process:

```
ENV DDFACET_TEST_DATA_DIR /test_data
ENV DDFACET_TEST_OUTPUT_DIR /test_output
```

`Tests/ShortAcceptanceTests/ClassCompareFITSImage.py` is the common
test base class. It downloads/locates a fixture MS, runs DDFacet with
a section of overrides, and asserts that each `definedImageList` FITS
file matches the reference within `defineMaxSquaredError` thresholds.
Example from `TestClean.py` (lines 32–53):

```python
class TestSSMFClean(ClassCompareFITSImage):
    @classmethod
    def defineImageList(cls):
        return ['dirty', 'dirty.corr', 'psf', 'NormFacets', 'Norm',
                'app.residual', 'app.model',
                'app.convmodel', 'app.restored']
    @classmethod
    def defineMaxSquaredError(cls):
        return [1e-5, 1e-5, 1e-5, 1e-5, 1e-5,
                1e-3, 1e-3, ...]
```

### 10.2 Unit tests

`FastUnitTests/TestStokesConverter.py` exercises
`Data/ClassStokes.ClassStokes` against every `StokesTypes`
combination. The patterns here are the simplest entry into the test
machinery for new contributors.

`FastUnitTests/TestFitter.py` covers `ToolsDir/ModFitPSF.py` and
`SkyModel/PSourceExtract/ClassPointFit*.py` (Gaussian fitter
robustness).

`FastUnitTests/TestLibraries.py` smoke-tests the C++ extensions:
imports `_pyArrays3x` and `_pyGridderSmearPols3x`, calls
`pySetOMPNumThreads`, allocates and deallocates a small grid, asserts
that the semaphore array can be created and torn down. This is the
test that catches build-system regressions early.

### 10.3 Long-running reference jobs

`VeryLongAcceptanceTests/` are not run in CI; they are reference
imaging jobs (3C147, DEEP2, supernova fields) used by the maintainers
to validate releases against canonical sky-models. Each test script
takes hours and >100 GiB of RAM.

### 10.4 Debug parsets

`DebugParsets/` carries a handful of small parsets used to reproduce
historical bugs. `simms.sh` is a shell script that synthesises a tiny
MS via the `simms` tool — useful when you want to develop without a
real observation handy.

### 10.5 Local development workflow

The README (`README.rst` lines 132–210) recommends:

```bash
virtualenv ddfvenv
source ddfvenv/bin/activate
pip install -U pip
git clone https://github.com/cyriltasse/DDFacet
pip install -e DDFacet/
pixi run test     # not applicable — DDFacet uses pip + pynose
DDF.py mytest.parset --Output-Clobber=1
```

There is no pytest configuration; test discovery uses nose / pynose
conventions (`def test_*`, `class Test*`).

---

## 11. Extension points

### 11.1 Adding a new deconvolution algorithm

`Imager/HOGBOM/ClassImageDeconvMachineHogbom.py` lines 28–32 explicitly
nominates the Hogbom implementation as the reference for new modes.
The recipe is:

1. Create `DDFacet/Imager/<NAME>/ClassImageDeconvMachine<NAME>.py`.
   Define `class ClassImageDeconvMachine` deriving from (or matching
   the API of) `Imager/ClassImageDeconvMachine.ClassImageDeconvMachine`.
   Required methods:
   - `__init__(self, Gain, MaxMinorIter, NCPU, ..., GD=GD, **kw)`
   - `setMaskMachine(self, MaskMachine)`
   - `Init(self, **kwargs)`
   - `Update(self, DicoDirty, JonesNorm, ...)`
   - `Deconvolve(self, ch=0)` — returns `(updated, continue, ret_code)`
   - `GiveModelImage(self, freq=None)` → ndarray
   - `ToFile(filename)` / `FromFile(filename)` — DicoModel I/O.
2. Create `Class<Name>ModelMachine.py` deriving from
   `Imager/ClassModelMachine.ClassModelMachine`. Implement
   `GiveModelImage(freqs)`, `AppendIsland`, `setRefFreq`,
   `setModelImage`, `ToFile`, `FromFile`.
3. Register the algorithm in
   `Imager/ModModelMachine.ClassModModelMachine.GiveMM(Mode=...)`
   so the factory can hand back the right model machine for
   `--Deconv-Mode=<NAME>`.
4. Add the dispatch branch in
   `Imager/ClassDeconvMachine.ClassImagerDeconv.Init` (lines 270–312)
   for `--Deconv-Mode=<NAME>`.
5. Add a parset section `[<Name>]` to
   `Parset/DefaultParset.cfg` for any new options.
6. Add an extension to `Parset/DefaultParset.cfg`'s
   `--Deconv-Mode` `#options:` enum.
7. Drop a fixture-driven test under
   `Tests/ShortAcceptanceTests/Test<Name>.py` derived from
   `ClassCompareFITSImage`.

### 11.2 Adding a new beam model

1. Add `DDFacet/Data/Class<Telescope>Beam.py` exposing
   `class ClassBeam` with the same interface as
   `ClassFITSBeam` (see `Data/ClassFITSBeam.py` for the canonical
   signature: `evaluateBeam(times, freqs, ras, decs, ant_indices)`
   returning a complex Jones cube of shape
   `(n_time, n_freq, n_ant, 2, 2)`).
2. Wire it into `Imager/ClassFacetMachine` / `ClassDDEGridMachine`
   beam-construction logic (search for `Beam-Model` references).
3. Extend the `--Beam-Model` enum in `DefaultParset.cfg` line 185.
4. Optional: extend `ClassBeamMean.py` if you need time-averaging.

### 11.3 Adding a new gridder backend

The `--RIME-BackwardMode` enum exposes `BDA-grid|Classic`. To add a
new mode:

1. Implement the kernel in `DDFacet/Gridder/` — typically a new C++
   file that defines a pybind11 `PYBIND11_MODULE(_pyGridder<Name>, m)`.
2. Wire the build into `Gridder/CMakeLists.txt` (`add_library(...)`
   + `install(TARGETS ...)`).
3. Import it from `ClassFacetMachine.py` (top of file) and add a
   branch on `self.GD["RIME"]["BackwardMode"]`.
4. Mirror the change for `ForwardMode` in `ClassDDEGridMachine.py`.

### 11.4 Adding a new sky-model loader

The companion `SkyModel/Sky/ClassSM.py` already supports BBS, NumPy,
Tigger, DS9 region inputs. To add a new format, add a
`ModXxx2np.py` file with `def ToNp(filename) -> recarray` and a
`ModSMFromNp.py` round-tripper, then route it through
`Sky/ClassSM.ClassSM.read_sm_from_*`.

### 11.5 Hooking into the major loop

`Imager/ClassDeconvMachine.ClassImagerDeconv.main` is the place to
add post-major-cycle hooks (e.g. logging, on-the-fly mask updates).
Each iteration ends with a callback to
`self.DeconvMachine.Deconvolve()` and a residual recompute via
`self.FacetMachine.putChunk(...)`; either side is a natural insertion
point.

---

## 12. Notable internals

### 12.1 Shared memory: `Array/NpShared.py` + `shared_dict.py`

DDFacet does *not* use `multiprocessing.shared_memory` — it uses the
`SharedArray` PyPI package, which `mmap`s POSIX shm regions of the
form `/dev/shm/SharedArray.<name>`. The wrapper
`Array/NpShared.zeros(name, *args, **kwargs)` (lines 37–42) is the
allocation primitive:

```python
def zeros(Name, *args, **kwargs):
    try:
        return SharedArray.create(Name, *args, **kwargs)
    except:
        DelArray(Name)
        return SharedArray.create(Name, *args, **kwargs)
```

The naming convention prepends each segment with a per-PID prefix
(`Other/Multiprocessing.getShmName(name, sem=None)`) so that crashed
runs do not collide; `Multiprocessing.cleanupStaleShm()` walks
`/dev/shm` at startup and `unlink()`s any segment whose creator PID is
no longer alive (`DDF.py` line 256).

`Array/shared_dict.py` builds a hierarchical view: a `SharedDict`
behaves like a Python dict whose array values live in shm and whose
non-array values are pickled into a small "metadata" segment. The
top-level shared dicts in a run are `DicoDirty`, `DicoImagesPSF`,
`DATA` (declared on `ClassImagerDeconv` lines 142–144) and the
per-facet `_facet_grids` / `_CF` / `_model_dict` / `_norm_dict`.

### 12.2 `AsyncProcessPool` (the `APP` import)

`Other/AsyncProcessPool.py` defines a custom multiprocessing pool
with the following design choices:

- **Job-handler registration**: classes call
  `APP.registerJobHandlers(self)` at construction time; methods
  marked with `@APP.handler` (or registered manually) become
  remote-callable.
- **Job submission**: `APP.runJob("label", target=..., args=...)`
  enqueues work on one of the worker processes.
- **Job counters**: `APP.createJobCounter("name")` returns a small
  shared counter used to gate "join" semantics — workers atomically
  decrement when done, the mother process spins on
  `APP.awaitJobCounter("name")`.
- **Affinity**: workers can be CPU-pinned via taskset on Linux,
  driven by `--Parallel-Affinity`.
- **Graceful shutdown**: `APP.terminate()` SIGTERMs all workers and
  waits with a short grace period; `APP.shutdown()` joins them.
  A `WorkerProcessError` is raised in the mother if any worker exits
  with a non-zero code or unhandled exception, propagating the
  worker's traceback.
- **Pause-on-start**: with `--Debug-PauseWorkers=1`, workers `SIGSTOP`
  themselves immediately after fork so you can attach gdb.

### 12.3 OpenMP in the gridder

The C++ gridder uses `#pragma omp parallel for` over BDA blocks. The
thread count is set globally by
`_pyArrays.pySetOMPNumThreads(NCPU)` (`DDF.py` line 270). Because
each visibility may map to multiple uv cells (from the convolution
support), and concurrent OpenMP threads gridding adjacent
visibilities can collide on the same uv cell, the gridder serialises
cell increments using **POSIX semaphores**: 3373 of them, hashed by
cell index. The semaphore array is created once by
`ClassFacetMachine.setup_semaphores` (`ClassFacetMachine.py` lines
161–168) and torn down via an `atexit` hook
(`_delete_degridding_semaphores`, lines 170–176).

### 12.4 Caches keyed by parset sub-dicts

`Other/CacheManager.py` hashes the parset section dict that affects
each cache product:

```python
key = dict([("MSNames", [ms.MSName for ms in self.VS.ListMS])] +
            [(section, self.GD[section]) for section in
             ["Data", "Beam", "Selection",
              "Freq", "Image", "Comp",
              "CF", "RIME", "Facets", "Weight", "DDESolutions"]] +
           [("InitDicoModel", self.GD["Predict"]["InitDicoModel"])])
```

(`Imager/ClassDeconvMachine.py` lines 391–399). The hash is
deterministic — change any relevant parameter and the cache rebuilds
automatically.

### 12.5 SIGUSR1 graceful stop

`Imager/ClassDeconvMachine.py` lines 102–109 install a `SIGUSR1`
handler that sets a module-level `user_stopped = True`. The major
loop polls this flag and exits cleanly at the next safe point —
useful for "stop after the current major cycle" interruption of long
runs without losing the current model.

### 12.6 Memory-greedy mode

`--Debug-MemoryGreedy=1` (default) keeps the `DicoDirty`,
`DicoImagesPSF` and `DATA` shared dicts alive across major cycles so
that re-using their backing shm pages avoids repeated allocator
churn. Setting it to 0 frees them aggressively at the cost of more
mmap traffic.

### 12.7 Rephasing

`ClassMS` calls into `ToolsDir/ModRotate.py` to re-phase visibilities
to a common direction when multiple MSs differ in their phase
centre. The default `--Image-PhaseCenterRADEC=align` uses the first
MS's phase centre; an explicit `[HH:MM:SS,DD:MM:SS]` is honoured if
provided.

### 12.8 FFTW wisdom

`ToolsDir/ModFFTW.py` persists pyFFTW plans to
`--Cache-DirWisdomFFTW=~/.fftw_wisdom`, indexed by transform shape
and dtype. The first run is slow; subsequent runs reuse the wisdom.
Reset via `--Cache-ResetWisdom=1`.

### 12.9 Python 2/3 compatibility shim

`DDFacet/compatibility.py` plus `import six` everywhere is the legacy
Py2/Py3 bridging layer. With `pyproject.toml` requiring
`requires-python = ">=3.11,<3.13"` (line 16), Python 2 is no longer
buildable; the shim is preserved for source compatibility but can be
removed in a future cleanup pass (`DDF.py` lines 418–422 emit a
`DeprecationWarning` if Py2 is detected).

---

## 13. Known limitations and caveats

These are observable from the source as it currently stands; some
also appear as comments in the parset.

- **Stokes deconvolution is I-only.** The deconvolvers branch out
  for `IQ|IU|IV|IQU|IQUV` only to write Stokes residuals; deconv is
  applied to I alone (`DefaultParset.cfg` lines 69–72;
  `ClassDeconvMachine.py` lines 271–272 raise
  `NotImplementedError("Multi-polarization CLEAN is not supported in
  MSMF")` for HMP, and the same for SSD/SSD2/MultiSlice/WSCMS).
- **No GPU back-end.** All numerical work is CPU/OpenMP. The
  Montblanc DFT predictor uses TensorFlow but only for prediction,
  not gridding. CASA-style WSClean gridder, IDG, NIFTy,
  RASCIL-FFT-based modes are not bundled.
- **PSFFacets ≠ NFacets is unsupported.** `ClassDeconvMachine.py`
  lines 360–362 print "the PSFFacets version is currently not
  supported, using 0 (i.e. same facets as image)" if you try.
- **`Output-Mode=CleanMinor`** appears as a vestigial value in
  `Init` (lines 198–203) but is not exposed in
  `DefaultParset.cfg`'s `--Output-Mode` enum (`Dirty|Clean|Predict|PSF`).
  Treat it as undocumented experimental code.
- **SSD2 is not recommended for >3 major cycles**
  (`ClassDeconvMachine.py` lines 292–293 prints a warning).
- **Initial models must match the deconvolver type.** Loading a
  `*.DicoModel` of one type into a different `--Deconv-Mode` raises
  `NotImplementedError("You want to use different minor cycle and
  IniDicoModel types ...")` (lines 231–233). The legacy alias
  `MSMF` ↔ `HMP` and `GA` ↔ `SSD` are silently mapped (lines
  226–229).
- **BDAMode=2 (Oleg's mode) has known issues** — see issue #319 in
  upstream (`DefaultParset.cfg` line 147).
- **ENABLE_NATIVE_TUNING / ENABLE_FAST_MATH default to ON**.
  Binaries are non-portable across CPUs and break IEEE-754
  semantics. Disable via `cmake.define` in `pyproject.toml` for
  publishable wheels (README lines 105–113).
- **Shared-memory pressure**: large jobs need `/dev/shm` sized to
  90–100% of RAM; default Debian/Ubuntu sizing is 50% which DDFacet
  warns about at startup (`DDF.py` lines 198–207). The CI Jenkinsfile
  asks for `--shm-size=150g` (Jenkinsfile.sh line 26).
- **No formal pytest suite.** Tests use nose/pynose conventions and
  rely on environment-variable fixture paths
  (`DDFACET_TEST_DATA_DIR`, `DDFACET_TEST_OUTPUT_DIR`).
- **Several deprecated parset knobs remain** for back-compat:
  `DDESolutions.JonesNormList`, `ScaleAmpGrid`, `ScaleAmpDeGrid`,
  `CalibErr`, `Type`, `Scale`, `gamma`, `RestoreSub`, `ReWeightSNR`
  (DefaultParset.cfg lines 264–275).
- **Python 2 dead code paths** persist (`if six.PY2: ...`) but are
  no longer exercised since `requires-python >= 3.11`.
- **`apt.sources.list`** in the repo root is an Ubuntu-specific
  PPA-extension list pulled into the Docker image. It is not
  consumed by pip installs.

---

## 14. Pointers and acknowledgements

- Project page (upstream): https://github.com/saopicc/DDFacet
- Maintenance fork: https://github.com/cyriltasse/DDFacet
- Author contact: cyril.tasse@obspm.fr (Cyril Tasse, Observatoire de
  Paris).
- Maintainer: bhugo@sarao.ac.za (Benjamin Hugo, SARAO).
- The package is the imaging half of the *saopicc* radio reduction
  stack — the calibration counterpart is killMS (also by Cyril
  Tasse). DynSpecMS is the dynamic-spectrum extractor.

DDFacet's algorithmic ancestry is described in:

- Tasse, C. *et al.* 2018, A&A 611, A87 — "Faceting for
  direction-dependent spectral deconvolution" (the canonical
  algorithm paper).
- Smirnov, O. M., Tasse, C. 2015, MNRAS 449, 2668 — "Radio
  interferometric gain calibration as a complex optimization problem"
  (the killMS paper).

These references are not bundled in the repo but are the standard
citations for users of the imager.

