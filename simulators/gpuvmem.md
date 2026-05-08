# gpuvmem — Exhaustive Technical Reference

> Submodule path: `simulators/gpuvmem/`
> Upstream: <https://github.com/miguelcarcamov/gpuvmem>
> Reference paper: M. Cárcamo, P.E. Román, S. Casassus, V. Moral, F.R. Rannou,
> *"Multi-GPU maximum entropy image synthesis for radio astronomy"*,
> Astronomy and Computing 22 (2018) 16–27. <https://doi.org/10.1016/j.ascom.2017.11.003>

This document is an independent, source-derived technical reference for the
`gpuvmem` package as it appears in this RadioSim submodule. Every claim is cited
to a path of the form `simulators/gpuvmem/...`. No other `.md` files in
`simulators/` were consulted.

---

## 1. Overview

`gpuvmem` is a **GPU-accelerated, regularised maximum-likelihood / maximum-entropy
imaging code for radio interferometry**. Given a CASA Measurement Set (MS) of
calibrated visibilities and a FITS "model input" image that supplies astrometric
header information (typically a CASA `tclean` dirty image), `gpuvmem`:

1. Reads visibilities directly from the MS (column `CORRECTED_DATA` by default;
   see `simulators/gpuvmem/src/ioms.cu:10`).
2. Sets up a forward model `V_model = (FFT ∘ apply_primary_beam)(I)` on one or
   more NVIDIA GPUs (multi-GPU dispatch by frequency channel).
3. Numerically minimises a weighted sum of penalty terms
   `Φ(I) = χ²(V_obs − V_model) + Σ_k λ_k F_k(I)` with the
   non-linear conjugate gradient method (Polak–Ribière–Fletcher–Reeves, file
   `simulators/gpuvmem/src/frprmn.cu`) or with limited-memory BFGS
   (`simulators/gpuvmem/src/lbfgs.cu`).
4. Writes the final model image as FITS in `Jy/pixel`, the residual + model
   visibilities back to a (copied) MS, and optionally per-iteration FITS images
   plus error maps.

It is the reference implementation for the algorithm described in Cárcamo et al.
2018 and is used in production for ALMA/EHT-class imaging (the test suite
includes the M87 Event Horizon Telescope 2017 selfcal MSes —
`simulators/gpuvmem/tests/M87/SR1_M87_2017_101_*.ms`).

### 1.1 Languages & line counts

From `wc -l`:

| Bucket | Files | Lines |
|--------|-------|-------|
| `src/*.cu` (CUDA C++) | 39 | ~14 590 |
| `include/*.cuh` | 38 | ~3 315 |
| `include/classes/*.cuh` | 14 | ~1 880 |
| Python (`scripts/restore.py`) | 1 | 102 |
| **Total source** | ~92 files | **~17 905** |

The two largest files are `src/functions.cu` (5041 lines, host glue +
all global CUDA kernels) and `src/MSFITSIO.cu` (1254 lines, casacore/CFITSIO
I/O), followed by `src/mfs.cu` (1253 lines, the `MFS` synthesizer).

### 1.2 License

GPLv3 — `simulators/gpuvmem/LICENSE.txt:1-3`
*"GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007"*. Each `.cu` carries a
GPLv3 header (e.g. `simulators/gpuvmem/src/main.cu:1-32`). The code embeds
Numerical Recipes-derived routines (line-search, Brent, mnbrak) — users must
comply with the NR licence — and Park–Leemis multi-stream RNG
(`simulators/gpuvmem/src/rngs.cu`, `rvgs.cu`).

### 1.3 Versions & history

`git tag` (chronological end):
`v1.0, v1.1, v2.0, v2.1b, 2.1.1b, v2.2b, v2.3, v3.0, v3.1, v4.0, v5.0, v5.0.1,
6.0.0, 6.0.1, 6.1.0, 6.2.0, v7.0.0`. HEAD is `b860b94` — *"Merge pull request
#22 from miguelcarcamov/development"* (recent fixes: pragma omp loops,
gridding, `CHAN_FREQ` dtype, CMakeLists multi-GPU).

The framework has been refactored toward an extensible Strategy/Factory design
(see Section 4) — the version of code shipped here corresponds to the
"framework" generation post-v5 (`include/framework.cuh`).

### 1.4 Authors / contributors

From `simulators/gpuvmem/README.md:199-208`:

- Miguel Cárcamo (University of Manchester) — primary author
- Nicolás Muñoz, Fernando Rannou, Pablo Román (USACH)
- Simón Casassus, Axel Osses, Victor Moral (Universidad de Chile)

---

## 2. Repository layout

```
simulators/gpuvmem/
├── CMakeLists.txt              # CMake ≥ 3.18, CUDA + C++17 (default)
├── Dockerfile                  # nvidia/cuda:12.4.1-devel + casacore v3.5.0
├── Dockerfile.prod             # Slimmer production image
├── LICENSE.txt                 # GPLv3
├── README.md                   # User-facing readme + CLI table
├── _config.yml                 # Jekyll site config (gpuvmem.github.io)
├── environment.yml             # conda env (casacore, casatools)
├── environment_cudatoolkit.yml # alt conda env w/ cudatoolkit pin
├── requirements.txt            # casa* python packages for restore.py
├── .pre-commit-config.yaml
├── .github/
│   └── workflows/              # build_base_container.yml,
│                               # build_latest_container.yml,
│                               # build_tagged_container.yml,
│                               # workflow.yml
├── cmake/
│   ├── FindCasacore.cmake      # custom find module
│   └── FindCFITSIO.cmake
├── docs/
│   └── index.rst               # 1-line Jekyll URL stub
├── scripts/
│   └── restore.py              # CASA tclean-based restoration
├── tests/
│   ├── antennae/   {all_fields.ms, mod_in_0.fits, test.sh}
│   ├── co65/       {co65.ms, mod_in_0.fits, test.sh}
│   ├── FREQ78/     {FREQ78.ms, mod_in_0.fits, test.sh}
│   ├── M87/        {SR1_M87_2017_101_*.ms (3), mod_in_0.fits,
│   │                 M87_original_*freq.fits, test.sh}
│   └── selfcalband9/ {hd142_b9cont_self_tav.ms, mod_in_0.fits, test.sh}
├── include/
│   ├── classes/                # Abstract Strategy interfaces (14 files)
│   │   ├── ckernel.cuh         # CKernel: gridding/anti-alias kernel ABC
│   │   ├── error.cuh           # Error: error-image computer ABC
│   │   ├── fi.cuh              # Fi: penalty-term ABC (the "F_i")
│   │   ├── filter.cuh          # Filter ABC (e.g. Gridding)
│   │   ├── flags.cuh           # 3rd-party Flags getopt wrapper (Song Gao)
│   │   ├── image.cuh           # Image POD (device pointer + functionMap)
│   │   ├── io.cuh              # Io ABC (FITS / MS handlers)
│   │   ├── objectivefunction.cuh # ObjectiveFunction (sum of Fi)
│   │   ├── optimizer.cuh       # Optimizer ABC
│   │   ├── synthesizer.cuh     # Synthesizer ABC (top-level driver)
│   │   ├── uvtaper.cuh         # UVTaper Gaussian taper
│   │   ├── virtualimageprocessor.cuh # VIP ABC
│   │   ├── visibilities.cuh    # Visibilities (vector<MSDataset>)
│   │   └── weightingscheme.cuh # WeightingScheme ABC
│   ├── framework.cuh           # Pulls everything in + Vars + factories
│   ├── factory.cuh             # Singleton<Factory<T,V>> + createObject<>()
│   ├── chi2.cuh, entropy.cuh, gentropy.cuh,
│   │   l1norm.cuh, gl1norm.cuh, totalvariation.cuh,
│   │   totalsquaredvariation.cuh, laplacian.cuh,
│   │   quadraticpenalization.cuh           # concrete Fi
│   ├── frprmn.cuh, lbfgs.cuh, brent.cuh,
│   │   mnbrak.cuh, f1dim.cuh, linmin.cuh   # concrete optimizers
│   ├── naturalweightingscheme.cuh, uniformweightingscheme.cuh,
│   │   briggsweightingscheme.cuh, radialweightingscheme.cuh
│   ├── gaussian2D.cuh, gaussianSinc2D.cuh, sinc2D.cuh,
│   │   pillBox2D.cuh, pswf_12D.cuh         # concrete CKernels
│   ├── iofits.cuh, ioms.cuh                # concrete Io
│   ├── imageProcessor.cuh, secondderivateerror.cuh
│   ├── functions.cuh                       # Master kernel/host header
│   ├── MSFITSIO.cuh                        # MS + FITS structs/protos
│   ├── complexOps.cuh, directioncosines.cuh, gridding.cuh, mfs.cuh
│   ├── rngs.cuh, rvgs.cuh                  # Park–Leemis RNG
│   ├── fixedpoint.cuh                      # Belge-2002 λ fixed-point
│   ├── copyrightwarranty.cuh, drvrsmem.h, longnam.h, nrutil.h
└── src/
    └── (39 .cu files, mirroring the headers)
```

---

## 3. Build system

### 3.1 Required toolchain

From `simulators/gpuvmem/CMakeLists.txt:23-25,29-30,73-80`:

| Component | Minimum | Notes |
|-----------|---------|-------|
| CMake | **3.18** | `cmake_minimum_required(VERSION 3.18 FATAL_ERROR)` |
| CUDA Toolkit | **10.0** (13.0+ requires C++17) | `find_package(CUDAToolkit REQUIRED)` |
| Compute capability | **≥ 5.0** (compute_53 explicitly excluded — line 218, 261, 323) | |
| C++ standard | C++17 default; can override `-DCMAKE_CXX_STANDARD=11` for older systems | line 96-128 |
| Boost | any (uses `boost::algorithm::string`, `boost::math::bessel`, `boost::accumulators::sum_kahan`, `boost::type_index`) | `find_package(Boost REQUIRED)` |
| CFITSIO | system | `find_package(CFITSIO REQUIRED)` via `cmake/FindCFITSIO.cmake` |
| **casacore** | ≥ v3.1.2 (Docker uses v3.5.0) | components: `casa ms tables measures meas` (line 80); custom `cmake/FindCasacore.cmake` |
| OpenMP (libgomp) | required at link time | `set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")` line 458 |
| `helper_cuda.h`, `helper_functions.h` | from CUDA samples | included via `${CUDAToolkit_LIBRARY_ROOT}/samples/common/inc` (line 168). Dockerfile clones `NVIDIA/cuda-samples` v12.4 to `/usr/local/cuda/samples`. |
| Math/std libs | `m`, `stdc++`, `gomp` | `target_link_libraries` line 490 |
| CUDA libs | `cuda`, `cudart`, `cufft` | line 484 |

### 3.2 Build options

| Option | Default | Effect |
|--------|---------|--------|
| `-DCMAKE_BUILD_TYPE=Release\|Debug` | Release | line 51-56 |
| `-DCUDA_ARCH=<XX>` | auto-detected via `nvidia-smi --query-gpu=compute_cap` | line 176-183. Falls back to `nvcc --list-gpu-arch` minimum, or `75` (Turing) under CI. |
| `-DMEMORY_DEBUG=ON` | OFF | adds `-g -G -D_FORCE_INLINES -w` to nvcc (line 86, 461-464) |
| `-DUSE_FAST_MATH=ON` | OFF | adds `--use_fast_math -O3 -Xptxas -O3` (line 87, 466-471). Lower precision. |
| `-DPREFIX=/path` | `<src>/bin` | output dir for `gpuvmem` binary (line 154-158) |

### 3.3 Standard build

```bash
cd simulators/gpuvmem
mkdir build && cd build
cmake ..              # auto-detect GPU arch
make -j               # binary lands in simulators/gpuvmem/bin/gpuvmem
sudo make install     # installs to ${CMAKE_INSTALL_PREFIX}/bin (line 502)
```

Tests are registered with CTest (line 508-519):
```cmake
set(TEST_CASES antennae co65 freq78 m87 selfcalband9)
foreach(TEST_NAME ${TEST_CASES})
  add_test(NAME ${TEST_NAME}
           COMMAND bash ${TEST_DIR}/${TEST_NAME}/test.sh
                   ${BINARY_DIR}/gpuvmem ${TEST_DIR}/${TEST_NAME})
endforeach()
```
Run via `ctest` after `make`. Test MS files are pulled by `git lfs pull`
(README step 7).

### 3.4 Docker

```
docker pull ghcr.io/miguelcarcamov/gpuvmem:latest
```
The base `Dockerfile` (`simulators/gpuvmem/Dockerfile:1-69`) is built on
`nvidia/cuda:12.4.1-devel-ubuntu22.04`, installs casacore v3.5.0 from source
(`-DUSE_FFTW3=ON -DUSE_OPENMP=ON -DUSE_HDF5=ON -DUSE_THREADS=ON`), pulls CUDA
samples v12.4 for `Common/` headers, and is published to GHCR.

### 3.5 Restore-script Python deps

`scripts/restore.py` requires the `casa*` 6.6.4.34 stack listed in
`requirements.txt` (`casatasks, casatools, casaplotms, …`). The conda env in
`environment.yml` pins `python=3.10`, `numpy=1.24`, and pulls casacore from
`pkgw-forge`.

---

## 4. Runtime architecture

`gpuvmem` is built on a *Strategy + Abstract Factory* framework so users can
swap synthesizers, optimizers, regularisers and weighting schemes by name.
Concrete classes self-register at static-init time in anonymous namespaces, e.g.
`simulators/gpuvmem/src/chi2.cu:84-92`:

```cpp
namespace {
Fi* CreateChi2() { return new Chi2; }
const std::string name = "Chi2";
const bool RegisteredChi2 =
    registerCreationFunction<Fi, std::string>(name, CreateChi2);
}
```

### 4.1 Architecture diagram

```
                      ┌─────────────────────────────────┐
                      │        main.cu (driver)         │
                      │  Synthesizer = "MFS"            │
                      │  Optimizer   = "CG-FRPRMN"      │
                      │  Kernel      = PillBox2D        │
                      │  Scheme      = "Natural"        │
                      │  Fi: Chi2,Entropy,L1,TSV,Lap    │
                      └────────────────┬────────────────┘
                                       │ createObject<T,V>(id)
                       ┌───────────────▼───────────────┐
                       │   Singleton<Factory<T,V>>     │
                       └──┬──────┬──────┬──────┬──────┘
                          │      │      │      │
            ┌─────────────┘      │      │      └──────────────────┐
            ▼                    ▼      ▼                         ▼
   ┌──────────────────┐    ┌─────────────┐    ┌───────────────┐   (CKernel,
   │ Synthesizer       │    │ Optimizer   │    │ Fi (penalty)  │    Filter,
   │   = MFS           │    │  CG-FRPRMN  │    │ Chi2,Entropy, │    WeightScheme,
   │ run(), setDevice()│    │  CG-LBFGS   │    │ L1,GL1,GEntropy│   Io, Error)
   └────┬─────┬────────┘    └─────────────┘    │ TVar,TSV,Lap, │
        │     │                  uses          │ Quadratic     │
        ▼     ▼                                └───────────────┘
   ┌──────────┐ ┌───────────────────┐                  ▲
   │ Io (MS)  │ │ Visibilities      │                  │ adds to
   │ "IoMS"   │ │  vector<MSDataset>│            ┌─────┴────────────┐
   └──────────┘ │  per-GPU DVis     │            │ ObjectiveFunction│
   ┌──────────┐ └─────────┬─────────┘            │  Σ λ_k * F_k     │
   │ Io (FITS)│           │                      │  + ∇Φ aggregator │
   │ "IoFITS" │      cuFFT plan                  └──────────────────┘
   └──────────┘    per-GPU varsPerGPU
                      │
                      ▼
             ┌────────────────────────┐
             │ CUDA kernels in        │
             │ functions.cu / *.cu    │
             │  hermitianSymmetry     │
             │  apply_beam, FFT2D     │
             │  vis_mod (bilinear)    │
             │  residual, chi2Vector  │
             │  DChi2, SVector, DS,   │
             │  TVVector,  DTV ...    │
             └────────────────────────┘
```

### 4.2 Driver flow (`src/main.cu`)

1. `cudaGetDeviceCount` — abort if no GPU (`main.cu:102, 130`).
2. Build top-level objects via factory (`main.cu:147-168`):
   ```cpp
   Synthesizer*  sy   = createObject<Synthesizer, std::string>("MFS");
   Optimizer*    cg   = createObject<Optimizer,  std::string>("CG-FRPRMN");
   CKernel*      sc   = new PillBox2D();          // gridding/AA kernel
   ObjectiveFunction* of = createObject<ObjectiveFunction,std::string>("ObjectiveFunction");
   Io*           ioms = createObject<Io, std::string>("IoMS");
   Io*           iofits = createObject<Io,std::string>("IoFITS");
   WeightingScheme* scheme = createObject<WeightingScheme,std::string>("Natural");
   ```
3. Wire dependencies and call `sy->configure(argc, argv)` to parse CLI.
4. `sy->setDevice()` — read MS, allocate device buffers, build attenuation maps.
5. Build & attach `Fi*` terms (`Chi2`, `Entropy`, `L1-Norm`,
   `TotalSquaredVariation`, `Laplacian`) and `e->setPrior(0.001f)`
   (`main.cu:184-203`).
6. `sy->run()` — launches the optimizer.
7. `sy->writeImages()` (FITS), `sy->writeResiduals()` (MS),
   `sy->unSetDevice()` (free device + host memory).

The commented-out block at `main.cu:211-221` shows how the user can apply
**Belge et al. 2002** fixed-point regularisation tuning via
`fixedPointOpt(lambdas, &runGpuvmem, 1e-6, 60, sy)` defined in
`simulators/gpuvmem/src/fixedpoint.cu`.

### 4.3 Registered factory IDs

| Factory `<T,V>` | ID strings | Source |
|-----------------|-----------|--------|
| `Synthesizer, std::string` | `"MFS"` | `src/mfs.cu:1250` |
| `Optimizer, std::string` | `"CG-FRPRMN"`, `"CG-LBFGS"` | `src/frprmn.cu:202`, `src/lbfgs.cu:348` |
| `Fi, std::string` | `"Chi2"`, `"Entropy"`, `"GEntropy"`, `"L1-Norm"`, `"GL1Norm"`, `"TotalVariation"`, `"TotalSquaredVariation"`, `"Laplacian"`, `"Quadratic"` | `src/chi2.cu`, `src/entropy.cu`, `src/gentropy.cu`, `src/l1norm.cu`, `src/gl1norm.cu`, `src/totalvariation.cu`, `src/totalsquaredvariation.cu`, `src/laplacian.cu`, `src/quadraticpenalization.cu` |
| `Fi, int` | `0` → Entropy / GEntropy / GL1Norm | `src/entropy.cu:79`, `src/gentropy.cu:135`, `src/gl1norm.cu:169` |
| `WeightingScheme, std::string` | `"Natural"`, `"Uniform"`, `"Briggs"`, `"Radial"` | `src/naturalweightingscheme.cu`, `src/uniformweightingscheme.cu`, `src/briggsweightingscheme.cu`, `src/radialweightingscheme.cu` |
| `CKernel, std::string` | `"PillBox2D"`, `"Gaussian2D"`, `"GaussianSinc2D"`, `"Sinc2D"`, `"PSWF"` | corresponding `*2D.cu` / `pswf_12D.cu` |
| `Io, std::string` | `"IoMS"`, `"IoFITS"` | `src/ioms.cu:351`, `src/iofits.cu:505` |
| `Filter, std::string` | `"Gridding"` | `src/gridding.cu:46` |
| `ObjectiveFunction, std::string` | `"ObjectiveFunction"` | `include/classes/objectivefunction.cuh:109` |
| `Error, std::string` | `"SecondDerivateError"` | `src/secondderivateerror.cu:17` |

### 4.4 Factory infrastructure (`include/factory.cuh`)

```cpp
template <class T> class Singleton { public: static T& Instance(); ... };

template <class AbstractProduct, class IdentifierType,
          class ProductCreator = AbstractProduct*(*)()>
class Factory {  bool Register(id, creator);  AbstractProduct* CreateObject(id); };

template <class T, class V> T* createObject(V value);
template <class T, class V, class Creator = T*(*)()>
bool registerCreationFunction(V value, Creator function);
```
This is a textbook Loki-style factory; `boost::typeindex::type_id` is used to
print a friendly type name on lookup failure (`factory.cuh:78-83`).

`include/framework.cuh:83-242` *also* provides hand-rolled factories
(`SynthesizerFactory`, `WeightingSchemeFactory`, `FiFactory`,
`OptimizatorFactory`, `CKernelFactory`) keyed on `int` instead of
`std::string`. These coexist with the templated factory; the templated form
(`createObject<>`) is what `main.cu` uses.

---

## 5. Public CLI

The CLI is built on top of the embedded **flags.hh** (Song Gao, BSD,
`include/classes/flags.cuh:1-31`), which is a `getopt_long` wrapper. The full
table of options is registered in `simulators/gpuvmem/src/functions.cu:181-266`
inside `Vars getOptions(int argc, char** argv)`.

### 5.1 Mandatory

| Short | Long | Default | Meaning | Source line |
|-------|------|---------|---------|-------------|
| `-i` | `--input` | `NULL` | Comma-separated list of input MS files | functions.cu:185 |
| `-o` | `--output` | `NULL` | Comma-separated output MS file names (residual+model) | functions.cu:188 |
| `-m` | `--model_input` | `mod_in_0.fits` | FITS image whose header drives astrometry / pixel grid | functions.cu:194 |

### 5.2 Frequently used

| Short | Long | Default | Meaning |
|-------|------|---------|---------|
| `-O` | `--output_image` | `mod_out.fits` | Final FITS model image filename |
| `-p` | `--path` | `mem/` | Directory for FITS outputs (created if missing) |
| `-G` | `--gpus` | `0` | Comma-separated GPU device IDs (e.g. `0,1,2`) |
| `-X` | `--blockSizeX` | `-1` | Image-plane block.x; `-1` = auto |
| `-Y` | `--blockSizeY` | `-1` | Image-plane block.y; `-1` = auto |
| `-V` | `--blockSizeV` | `-1` | Visibility block size (1-D); must be power of 2 |
| `-t` | `--iterations` | `500` | Maximum optimizer iterations |
| `-z` | `--initial_values` | `NULL` | Comma-separated initial scalar values, one per image plane (e.g. `0.001,0.0` for I and α) |
| `-Z` | `--regularization_factors` | `NULL` | Comma-separated penalty factors `λ_k`, one per `Fi` term (apart from χ²) |
| `-R` | `--robust_parameter` | `2.0` | Robust/Briggs parameter; `-2.0`=uniform, `2.0`=natural, `0`=tradeoff |
| `-g` | `--gridding` | `0` | Number of CPU threads used for gridding visibilities; `0` = no gridding |
| `-e` | `--eta` | `-1` | Min image-value control for the entropy prior |
| `-T` | `--threshold` | `0` | σ-threshold for the spectral-index image (multiplied internally by 5 — `mfs.cu:107`) |

### 5.3 Optional / numerical

| Short | Long | Default | Meaning |
|-------|------|---------|---------|
| `-n` | `--noise` | `-1` | Manual visibility noise (Jy); else estimated |
| `-N` | `--noise_cut` | `10` | Multiplier on the minimum noise for the inner mask |
| `-F` | `--ref_frequency` | `-1` | ν₀ (Hz). If `-1`, set to mid of MS frequency range (`mfs.cu:329`). |
| `-r` | `--random_sampling` | `1` | Fraction of visibilities used (random subset) |
| `-f` | `--output_file` | `NULL` | Path to text file with final χ², S, λS values |
| `-U` | `--user-mask` | `NULL` | FITS mask file overriding the noise mask (forces `noise_cut=1.0`, `functions.cu:296`) |

### 5.4 Boolean flags

| Short | Long | Effect |
|-------|------|--------|
| `-v` | `--verbose` | Verbose stdout |
| `-x` | `--nopositivity` | Disable positivity for *all* image planes (default keeps positivity on plane 0 — `mfs.cu:798-810`) |
| `-a` | `--apply-noise` | Add random Gaussian noise to visibilities |
| `-P` | `--print-images` | Dump FITS image at every iteration |
| `-E` | `--print-errors` | Compute & write error map (`SecondDerivateError`) |
| `-s` | `--save_modelcolumn` | Write model visibilities back to MS `MODEL_DATA` column |
| `-M` | `--use-radius-mask` | Use a radius-based mask instead of noise-derived |
| `-W` | `--modify-weights` | Persist gpuvmem-modified weights to MS `WEIGHT` column |
| `-h/-w/-c` | `--help/--warranty/--copyright` | Print usage / GPLv3 warranty / copyright |

### 5.5 Example invocation (M87 EHT test)

`simulators/gpuvmem/tests/M87/test.sh:10`:

```bash
$1 -i  $2/SR1_M87_2017_101_hi_hops_netcal_StokesI.selfcal.LLRR.ms \
   -o  $2/residuals.ms        -O $2/mod_out.fits \
   -m  $2/mod_in_0.fits       -p $2/mem/ \
   -X  16  -Y 16  -V 256 \
   --verbose --print-images \
   -z 0.0,0.0  -Z 0.0,0.001,0.005 \
   -R -2.0  -t 500000000  --use-radius-mask
```

This runs MFS imaging with two image planes (intensity + spectral index, both
initialised to zero), zero penalty on χ², λ=0.001 on the first regulariser
(entropy), λ=0.005 on the second (L1), uniform weighting, radius mask, and an
effectively unbounded iteration limit (the optimizer exits on the
`ftol`/`gtol` criteria).

---

## 6. Input & output formats

### 6.1 Inputs

1. **Measurement Set(s)** — passed via `-i`. Read in
   `simulators/gpuvmem/src/MSFITSIO.cu` (`__host__ void readMS(...)` declared at
   `include/MSFITSIO.cuh:202-218`). Default data column: `CORRECTED_DATA`
   (`src/ioms.cu:10`). Supports multiple MSes joined with commas; each MS is
   modelled independently and contributes to the same image grid.
2. **FITS model header** — passed via `-m`. The image is *not* used as an
   initial guess (initial values come from `-z`); only its astrometry is
   inherited. Header parsing in `IoFITS::readHeader`
   (`src/iofits.cu`); fields stored in `headerValues`
   (`include/MSFITSIO.cuh:140-150`):
   `DELTAX, DELTAY, ra, dec, crpix1, crpix2, M, N, beam_bmaj, beam_bmin,
    beam_bpa, beam_noise, radesys, equinox, bitpix`.
3. **Optional user mask** — `-U` FITS file matching `M×N`.

### 6.2 Outputs

| Path | Format | Contents |
|------|--------|----------|
| `<path>/<output_image>` | FITS | Final model image, units `JY/PIXEL`, header copied from `-m` (`MFS::writeImages`, `mfs.cu:1076-1089`) |
| `<path>/alpha.fits` | FITS | Spectral-index image (only if `-P` and `image_count==2`) — `mfs.cu:1083` |
| `<path>/error_Inu_0.fits`, `<path>/error_alpha_0.fits` | FITS | Error maps if `-E` (`mfs.cu:1100-1106`) |
| `<output>` (MS) | CASA MS | Copy of input MS with `DATA` column rewritten to **residuals** and `MODEL_DATA` to model visibilities. Written by `IoMS::writeResidualsAndModel` via `MFS::writeResiduals` (`mfs.cu:1115-1155`). |
| `<path>/dataset_<d>_atten_*.fits`, `noise.fits`, `distance.fits` | FITS | Diagnostic per-iteration prints when `-P` enabled (`mfs.cu:861, 901, 904`) |
| `--output_file` | text | One-line summary: iterations, χ², reduced-χ², S, λS, wall-time (`mfs.cu:1049-1073`) |

### 6.3 Restoration (post-processing)

`simulators/gpuvmem/scripts/restore.py` re-derives the CLEAN beam by running
`tclean(niter=0)` on the residual MS in CASA, convolves the gpuvmem model with
that Gaussian, and adds residuals to produce a Jy/beam restored image. CLI:

```bash
python restore.py <model.fits> <residuals.ms[,more.ms]> <restored_basename> <weighting> <robust>
```
It returns PSNR, peak (mJy/beam), and RMS.

---

## 7. Core algorithms

### 7.1 Forward model

For a single field, frequency `ν`, Stokes `s`:

```
I_ν(x,y)  = I(x,y) * (ν / ν₀)^α(x,y)         # spectral extrapolation
B_ν(x,y)  = primary_beam(ν, antenna_diameter, x_obs, y_obs)
F[I_ν · B_ν](u,v) = V_grid(u,v)
V_model(u_k, v_k) = bilinear_interp(V_grid, u_k, v_k)
```

Implemented kernels (all in `src/functions.cu`):

| Kernel | Role |
|--------|------|
| `attenuation` (device fn, `functions.cu` near 2200) | Selects Gaussian or Airy disk primary beam |
| `total_attenuation` | Builds attenuation map per field/frequency |
| `weight_image`, `noise_image` | Build per-pixel weight and noise maps |
| `hermitianSymmetry` (functions.cu:2256) | Reflects (-u,-v) onto (u,v) so V is conjugate-symmetric |
| `linkApplyBeam2I`, `linkCalculateInu2I` | Multi-image forward model for I + α |
| `FFT2D` host wrapper around `cufftExecC2C` | FFT image → V plane (with shift) |
| `vis_mod`, `vis_mod2` (functions.cu:2557 / 2615) | Bilinear interp from V plane to scattered (u,v) — uses `__ldg` for read-only cache |
| `residual` (functions.cu:2663) | `Vr = Vo − Vm` |
| `chi2Vector` (functions.cu:2867) | `chi2_i = w_i * |Vr_i|²` followed by reduction |

The forward model is computed once per cost evaluation; gradients use
`DChi2` (functions.cu near 2900–3200) which carries the chain rule through
the FFT and primary-beam multiplication.

### 7.2 Objective function

`ObjectiveFunction` (`include/classes/objectivefunction.cuh`) holds a
`std::vector<Fi*> fis`. On each iteration:

```cpp
calcFunction(p):    Φ(p)  = Σ Fi[k]->calcFi(p);   // line 15-26
calcGradient(p,xi): zero  dphi
                    Σ Fi[k]->calcGi(p, xi); Fi[k]->addToDphi(dphi)
                    cudaMemcpy dphi → xi          // line 28-45
```

Each `Fi` exposes `calcFi(float*)` (return value), `calcGi(float* p, float*
xi)` (writes into a private `device_DS`), `restartDGi`, `addToDphi(dphi)`,
plus `setPrior`, `setEta`, `setCKernel`, `setFgScale`. The base class
allocates `device_S` and `device_DS` of size `M*N` floats per term in
`Fi::configure` (`include/classes/fi.cuh:84-88`).

### 7.3 Penalty terms (concrete `Fi`)

| `Fi` class | ID | Math (per pixel) | Kernel(s) |
|------------|-----|-----------------|-----------|
| `Chi2` | `"Chi2"` | `½ Σ_k w_k |V_obs_k − V_mod_k(I)|²` | `chi2Vector` + reduction; `dchi2` chain-rules through FFT (`src/chi2.cu`) |
| `Entropy` | `"Entropy"` | `S(I) = -Σ I log(I/η · prior)` (Gull–Skilling); η controls floor | `SEntropy` / `DEntropy` (declared `functions.cuh:195`) |
| `GEntropy` | `"GEntropy"` | Generalised entropy with full prior **map** (not just scalar) | `SGEntropy` / `DGEntropy` |
| `L1-Norm` | `"L1-Norm"` | `Σ √(I² + ε)` (smooth ‖I‖₁) | `L1Vector`, `DL1NormK` (functions.cu:2902, 2938) |
| `GL1Norm` | `"GL1Norm"` | Generalised/weighted L1 with prior, `ε_a, ε_b` | `GL1NormK`, `DGL1Norm` |
| `TotalVariation` | `"TotalVariation"` | `Σ √(|∇I|² + ε)` | `totalvariation` / `DTVariation` |
| `TotalSquaredVariation` | `"TotalSquaredVariation"` | `Σ |∇I|²` | `TotalSquaredVariation` / `DTSVariation` |
| `Laplacian` | `"Laplacian"` | `½ Σ (∇²I)²` | `laplacian` / `DLaplacian` |
| `Quadratic` | `"Quadratic"` | `½ Σ I²` (Tikhonov) | `quadraticP` / `DQuadraticP` |

Each concrete `Fi` registers its name with the factory in an anonymous
namespace at the bottom of its `.cu` file. The penalty factor `λ_k` is set
either via `Fi::setPenalizationFactor(float)` or by passing
`-Z λ_0,λ_1,...` and using `Fi::configure(penalizatorIndex=k, ...)`. A
`penalizatorIndex == -1` keeps the `Fi`'s constructor-time factor (the default
for χ² in `main.cu:190`).

### 7.4 Optimizers

#### `ConjugateGradient` (`src/frprmn.cu`, ID `"CG-FRPRMN"`)

Polak–Ribière nonlinear conjugate gradient with Fletcher–Reeves fallback
(textbook Numerical Recipes `frprmn` ported to CUDA). Skeleton
(`frprmn.cu:85-195`):

```cpp
allocate device_g, device_h, xi, temp, device_gg_vector, device_dgg_vector
fp = of->calcFunction(I);          of->calcGradient(I, xi, 0)
searchDirection<<<>>>(g, xi, h)    // g = -xi; h = g
for (i = 1..total_iterations):
    linmin(I, xi, &fret, NULL);              // 1-D minimisation
    if (2|fret-fp| <= ftol(|fret|+|fp|+EPS)) break;
    fp = of->calcFunction(I);     of->calcGradient(I, xi, i)
    CGGradCondition<<<>>>(temp, xi, I, max(fp,1), ...)
    if (deviceMaxReduce(temp) < gtol) break;
    getGGandDGG<<<>>>(...);   gg = reduce(gg_vec); dgg = reduce(dgg_vec)
    if (gg == 0) break;
    gam = max(0, dgg/gg);                    // PR-FR
    newXi<<<>>>(g, xi, h, gam, ...)          // h = g + γh; xi = h
```

`linmin` (`src/linmin.cu`) calls `mnbrak` (initial bracket) and `brent`
(golden-section / parabolic-interp 1-D minimiser) on `f1dim` —
`src/mnbrak.cu, brent.cu, f1dim.cu`. Default tolerances `ftol = gtol = 1e-12`,
`total_iterations = 500` (`include/classes/optimizer.cuh:14-30`).

#### `LBFGS` (`src/lbfgs.cu`, ID `"CG-LBFGS"`)

Limited-memory BFGS with `K=100` history pairs by default
(`include/lbfgs.cuh:30`). Extra device buffers `d_y, d_s, p_old, xi_old,
norm_vector` of size `M*N*K*image_count` (`lbfgs.cu:66-80`). Selectable via
the commented line in `main.cu:149`.

### 7.5 Multi-image (MFS) extension

When two image planes are configured, the planes represent
**(I_ν₀, α)** and the forward model uses
`I_ν = I_ν₀ · (ν/ν₀)^α`. Plane 0 is the intensity image (positivity enforced
by default — `defaultEvaluateXt`/`defaultNewP` for plane 0 in `mfs.cu:798-810`,
mapped to `particularEvaluateXt`/`particularNewP` for plane ≥1 via the
`imageMap` function pointers in `include/classes/image.cuh:4-7`).

If only one initial value is supplied via `-z`, gpuvmem auto-pads with `α=0`
and bumps `image_count` to 2 (`mfs.cu:176-180`).

### 7.6 Weighting schemes

| Class | ID | Source | Behaviour |
|-------|-----|--------|-----------|
| `NaturalWeightingScheme` | `"Natural"` | `naturalweightingscheme.cu` (64 lines) | Pass-through `weight = 1/σ²`. |
| `UniformWeightingScheme` | `"Uniform"` | `uniformweightingscheme.cu` (113 lines) | Density compensation per uv cell. |
| `BriggsWeightingScheme` | `"Briggs"` | `briggsweightingscheme.cu` (202 lines) | Robust parameter from `-R`. `configure(void* params)` interprets `*params` as `float robust_param` (see `include/briggsweightingscheme.cuh`). |
| `RadialWeightingScheme` | `"Radial"` | `radialweightingscheme.cu` (57 lines) | Radial in uv. |

All inherit `WeightingScheme` (`include/classes/weightingscheme.cuh`) which
also supports an attached `UVTaper` (Gaussian taper with bmaj/bmin/BPA from
`include/classes/uvtaper.cuh`) and a `restoreWeights` helper that copies
`backup_visibilities[].weight` back into `visibilities[].weight` post-run
(`weightingscheme.cuh:62-74`).

### 7.7 Anti-aliasing / convolution kernels (`CKernel`)

For the optional gridding path (`-g >= 1`), the visibilities are convolved
onto a regular uv grid with a separable kernel. Concrete kernels:

| Class | ID | Profile |
|-------|-----|---------|
| `PillBox2D` | `"PillBox2D"` | Boxcar (default in `main.cu:152`) |
| `Gaussian2D` | `"Gaussian2D"` | 2-D Gaussian |
| `Sinc2D` | `"Sinc2D"` | sinc/Hanning |
| `GaussianSinc2D` | `"GaussianSinc2D"` | sinc·Gaussian (the WSCLEAN-like kernel) |
| `PSWF_12D` | `"PSWF"` | Prolate spheroidal wave function (computed via `boost::math::special_functions::bessel`) |

The `CKernel` interface (`include/classes/ckernel.cuh`) builds both the kernel
itself (`buildKernel`) and a *gridding correction function* (`buildGCF`),
which is FFT'd separately and applied as a multiplicative correction in the
image plane. Each kernel exposes `setSigmas`, `setW`, `getSupportX/Y`, etc.

Gridding driver: `do_gridding(fields, data, deltau, deltav, M, N, ckernel,
threads)` declared at `include/functions.cuh:82-89`, implemented in
`src/functions.cu` with OpenMP-parallel CPU gridding (no GPU); de-gridding is
`getOriginalVisibilitiesBack` (`functions.cuh:99-103`).

---

## 8. Significant files (file-by-file)

### `src/main.cu` (229 lines)

Driver. `runGpuvmem` (line 59) implements the **Belge et al. 2002**
self-tuning regularisation (each λ_k ← (Φ_χ²/Φ_k) · log(F_k)/log(F_χ²)),
bound to be non-negative, used by `fixedPointOpt`. `optimizationOrder`
(line 88) is the default per-image optimisation order callback registered
on the synthesizer. `__host__ int main` (line 100) does GPU-count check,
factory wiring, configures objective, runs synthesizer.

### `src/mfs.cu` (1253 lines, the `MFS` synthesizer)

- **Globals** (line 4-31): `M, N, numVisibilities`,
  `device_Image, device_dphi, device_dchi2_total, device_S, device_DS,
   device_noise_image, device_weight_image, device_distance_image,
   noise_cut, MINPIX, eta, robust_param, host_I, sum_weights, penalizators,
   nMeasurementSets, datasets`. These constitute the program's de facto
   shared state (note: not encapsulated; pre-`framework.cuh` legacy).
- `MFS::configure` (line 79) — full CLI parse, MS/FITS read, initial values,
  GPU enumeration, peer-access setup, weighting+gridding setup.
- `MFS::setDevice` (line 530) — allocates per-channel/Stokes device
  visibility buffers, calls `calculateNoiseAndBeam`, builds attenuation,
  weight and noise images, computes `fg_scale` (≈ noise floor in Jy/pix),
  copies initial host image to device, sets `imageMap` per-plane, calls
  `initFFT` to create one cuFFT plan per GPU.
- `MFS::clearRun` (line 945) — resets device visibility buffers and image
  to initial state (used by `runGpuvmem` for fixed-point iteration).
- `MFS::run` (line 978) — sets χ² fg_scale, runs the optimizer (or the
  user-supplied `Order` callback), prints final stats (χ², reduced-χ², S,
  λS, CPU/wall time), optionally writes summary to `--output_file`.
- `MFS::writeImages` (line 1076) — final FITS write, error-image
  computation if `-E`.
- `MFS::writeResiduals` (line 1115) — when gridding is on, calls
  `getOriginalVisibilitiesBack` to recover ungridded `Vm/Vr`, copies to
  host with `modelToHost`, copies the input MS to the output path with
  `IoMS::copy`, then writes via `writeResidualsAndModel`.
- `MFS::unSetDevice` (line 1157) — full memory cleanup, disables peer
  access, destroys cuFFT plans.

### `src/functions.cu` (5041 lines)

Master implementation file containing nearly every CUDA `__global__` kernel
and most `__host__` glue. Organised loosely as:

- **CLI parsing**: `getOptions` (l. 181), `print_help`, `init_beam`.
- **Reductions** (`deviceReduce<T>`, `deviceMaxReduce`, `deviceMinReduce`,
  `reduceMaxKernel`, `reduceMinKernel`, `deviceReduceKernel`).
- **Image-plane utilities**: `clipping`, `clipWNoise`, `newP`,
  `newPNoPositivity`, `evaluateXt`/`evaluateXtNoPositivity`, `clip`,
  `restartDPhi`, `makePositive`, `defaultNewP`, `particularNewP`,
  `defaultEvaluateXt`, `particularEvaluateXt`, `linkChain2I`,
  `linkClipWNoise2I`, `linkClip`, `normalizeImage`.
- **Forward model**: `hermitianSymmetry` (l. 2256), `total_attenuation`,
  `weight_image`, `noise_image`, `distance_image`, `phase_rotate`, `FFT2D`
  wrapper, `vis_mod` / `vis_mod2` (bilinear interp, l. 2557 / 2615),
  `residual`, `chi2Vector`.
- **Gradient**: `DChi2` overloads (l. ~2900–3300), `searchDirection`,
  `newXi`, `getGandDGG`, `CGGradCondition`.
- **Regularisers**: `SVector`, `DS`, `QPVector`, `DQ`, `TVVector`, `DTV`,
  `L1Vector`, `DL1NormK`, `DGL1Norm`, plus host wrappers
  `SEntropy`, `DEntropy`, `SGEntropy`, `DGEntropy`, `totalvariation`,
  `DTVariation`, `TotalSquaredVariation`, `DTSVariation`, `quadraticP`,
  `DQuadraticP`, `laplacian`, `DLaplacian`, `L1Norm`, `DL1Norm`,
  `GL1NormK`, `DGL1Norm`.
- **Beams**: `__device__ AiryDiskBeam`, `__device__ GaussianBeam`,
  `__device__ attenuation` selecting between them via the `enum {AIRYDISK,
  GAUSSIAN}` flag (`include/MSFITSIO.cuh:56`).
- **Top-level host functions**: `chi2`, `dchi2`, `simulate`,
  `do_gridding`, `degridding`, `griddedTogrid`,
  `getOriginalVisibilitiesBack`, `calculateNoiseAndBeam`, `calc_sBeam`,
  `initFFT`, `calculateErrors`.

### `src/MSFITSIO.cu` (1254 lines)

Casacore + CFITSIO bridge. Implements:

- `readOpenedFITSHeader`, `readFITSHeader`, `openFITS`, `closeFITS`
  (header parsing).
- `readMS(name, antennas, fields, data, noise, W_projection,
   random_prob, gridding)` and the `data_column`-aware overload — uses
   `casacore::MeasurementSet`, `MSAntennaColumns`, `MSMainColumns`,
   `TableIter`, `TaQL` to iterate rows, group by `DATA_DESC_ID` and field,
   build `Field` / `MSAntenna` POD structs, populate
   `host_visibilities { uvw, weight, Vo, Vm, Vr, S }`.
- `MScopy(in, out)` — table-level copy of an MS so we can rewrite
  columns without touching the source.
- `modelToHost`, `writeMS` — copies device-side `Vm/Vr` back and writes
  to the output MS columns (`DATA` <- residuals, `MODEL_DATA` <- model).
- `OCopyFITS`, `OCopyFITSCufftComplex`, `createFITS`, `copyHeader` —
  helpers used by `IoFITS` to stamp out new FITS files preserving WCS.
- Inline templated readers: `open_fits<T>`, `readHeaderKeyword<T>`
  (`include/MSFITSIO.cuh:157-200`).
- Helpers: `freq_to_wavelength` (= `LIGHTSPEED/freq`), `metres_to_lambda`,
  `distance` — all `__host__ __device__`.

### `src/iofits.cu` (508 lines), `src/ioms.cu` (354 lines)

Concrete `Io` strategies. `IoMS` defaults to `CORRECTED_DATA → DATA` and
exposes `random_probability, gridding, apply_noise_input,
apply_noise_output, W_projection, store_model_vis_input,
datacolumn_input, datacolumn_output` knobs. `IoFITS` provides a host of
overloaded `printImage` / `printNotPathImage` /
`printImageIteration` / `printcuFFTComplex` methods covering every
combination of (path, normalisation, GPU-vs-host, Stokes-cube vs single).

### `src/frprmn.cu` (205) and `src/lbfgs.cu` (351)

Already covered in §7.4. Implementations rely on `linmin.cu` (line search),
`mnbrak.cu` (bracket), `brent.cu` (1-D minimiser) and `f1dim.cu` (objective
restricted to a search direction).

### Penalty implementations

| File | Class | Notable detail |
|------|-------|----------------|
| `src/chi2.cu` (92) | `Chi2` | Owns its own `result_dchi2` device buffer of size `M*N*image_count`; if `image_count==1` calls `linkAddToDPhi(dphi, result_dchi2, 0)`, otherwise overwrites all planes (line 60-69). |
| `src/entropy.cu` (80) | `Entropy` | Default `prior_value = 1.0`, `eta = -1`. Registers `int 0` alias too (line 79). |
| `src/gentropy.cu` (136) | `GEntropy` | Holds a `float* prior` map; `normalizePrior()` divides by sum. |
| `src/l1norm.cu` (60), `src/gl1norm.cu` (170) | `L1Norm`, `GL1Norm` | smooth ‖I‖₁. |
| `src/totalvariation.cu` (63) | `TVariation` | constructor sets `epsilon=1E-12`. |
| `src/totalsquaredvariation.cu` (48) | `TSquaredVariation` | non-isotropic squared-grad regulariser. |
| `src/laplacian.cu` (46) | `Laplacian` | second-derivative penalty. |
| `src/quadraticpenalization.cu` (45) | `QuadraticP` | Tikhonov. |
| `src/secondderivateerror.cu` (19) | `SecondDerivateError : Error` | computes pixel-wise diagonal of the Hessian for an error map (used by `-E`). |

### `src/fixedpoint.cu` (57)

`fixedPointOpt(initial_lambdas, runGpuvmem_func, tol, max_iter, sy)` — Belge
self-similarity λ tuning. Iteratively rebuilds gpuvmem with new
`lambda_k = lambda_k * F(args, sy)` until `|Δλ| < tol` or `max_iter`.

### `include/classes/synthesizer.cuh`

The `Synthesizer` ABC owns: `image, optimizer, ckernel, ioImageHandler,
ioVisibilitiesHandler, visibilities, error, scheme`. It exposes
`setOrder(void(*)(Optimizer*, Image*))` which lets users override the
default optimisation pass-order (e.g. alternate optimising I_ν₀ and α).

### `include/classes/image.cuh`

A trivial holder around `float* image` (device pointer of length
`M*N*image_count`) plus `float* error_image` and a per-plane
`imageMap { newP, evaluateXt }` function-pointer table — this is how
positivity vs no-positivity is selected per plane.

### `include/classes/visibilities.cuh`

Holds `vector<MSDataset> datasets`, `ndatasets`, `total_visibilities`,
`max_number_vis`, plus `applyWeightingScheme(scheme)` which delegates to
`scheme->apply(datasets)`.

### Random-number generator

`src/rngs.cu` (Park–Miller LCG, `RandomGenerate`, `Get/PutSeed`) and
`src/rvgs.cu` (variate generators: `Bernoulli, Geometric, Poisson, Normal,
Lognormal, Erlang, Exponential, …`) — used to optionally inject Gaussian
noise into visibilities (`-a/--apply-noise`). Park & Leemis attribution in
header (`src/main.cu:9-12`).

### `include/copyrightwarranty.cuh` / `src/copyrightwarranty.cu` (642)

Embedded full GPLv3 text and `print_help` stub.

---

## 9. Notable internals

### 9.1 Multi-GPU dispatch

- `mfs.cu:212-484` enumerates GPUs, prints memory clock / bandwidth / total
  global memory, then checks pairwise peer-to-peer (`cudaDeviceCanAccessPeer`)
  and Unified Virtual Addressing support before enabling
  `cudaDeviceEnablePeerAccess`. If any pair lacks UVA the program waives the
  test (`exit(EXIT_SUCCESS)`).
- The dispatch policy is **frequency-channel parallel**: one OpenMP thread per
  GPU, each thread `cudaSetDevice(firstgpu + i % num_gpus)` and processes its
  channels (`mfs.cu:820-844`). `varsPerGPU { device_chi2, device_dchi2, plan,
  device_I_nu, device_V }` (declared in `framework.cuh:49-55`) is allocated
  once per GPU.
- Per-GPU `cufftHandle plan` is created in `initFFT` (host glue in
  `functions.cu`), and the `device_V` / `device_I_nu` `cufftComplex*` buffers
  are reused across iterations.

### 9.2 Memory layout

Image cube is **plane-major** flat float array of length
`M*N*image_count` with linear index `N*M*k + N*i + j` (`mfs.cu:742`),
i.e. plane `k` is contiguous. Visibilities are stored per
`(dataset, field, channel, stokes)` quadruple as separate `DVis` device
buffers — `uvw (double3*)`, `weight (float*)`, `Vo, Vm, Vr (cufftComplex*)`,
each sized to `numVisibilitiesPerFreqPerStoke[i][s]`. There is no global
"big visibility array".

### 9.3 cuFFT use

`initFFT` creates one `cufftHandle` per GPU sized `M×N`, type `CUFFT_C2C`.
`FFT2D(out, in, plan, M, N, direction, shift)` wraps `cufftExecC2C` with
optional `fftshift` before/after via `phase_rotate`-style kernels.

### 9.4 Texture / `__ldg` optimisation

The bilinear interpolation kernels `vis_mod` and `vis_mod2`
(`functions.cu:2557, 2615`) use `__ldg(&V[...])` for read-only cached loads
of the gridded `V` plane. Recent commits (`c6cb6db`, `6791c2a`) reference
"adding texture to bilinear interpolations".

### 9.5 Precision

All host-side image storage, gradients and reductions are `float`
(single precision). UVW coordinates are `double3` to preserve baseline
precision (and casacore-native `Double`). `cufftComplex` is single precision.
`-DUSE_FAST_MATH=ON` further reduces accuracy in transcendentals.

### 9.6 Thread/block sizing

If `-X/-Y` are unset (`-1`), `mfs.cu:236-252` chooses
`threadsPerBlock = √256 = 16` per side (256 threads) and computes the grid
to cover `M×N`. Visibility kernels use a 1-D grid with `blockSizeV`
threads/block. The tests typically use `-X 16 -Y 16 -V 256`.

### 9.7 Reductions

`deviceReduce<T>` and `deviceMaxReduce`/`deviceMinReduce` use the standard
NVIDIA warp-shuffle reduction template (block-then-grid, two passes), backed
by `getNumBlocksAndThreads(...)` (`functions.cu:307-340`) which adheres to
`prop.maxGridSize[0]` and `maxThreadsPerBlock` limits.

---

## 10. Testing layout

`simulators/gpuvmem/tests/` contains five integration tests, each
self-contained:

| Test | MS | Image | Notable flags |
|------|----|----|---------------|
| `antennae` | `all_fields.ms` (mosaic) | `mod_in_0.fits` | `-Z 0.01,0.0 -g 1 -R 2.0` (natural + gridding) |
| `co65` | `co65.ms` | `mod_in_0.fits` | `-z 0.001 -Z 0.001 -g 1 -t 5e8` |
| `FREQ78` | `FREQ78.ms` | `mod_in_0.fits` | `-z 0.001 -Z 0.001,0.0 -g 2` (multi-thread gridding) |
| `M87` | three EHT 2017 selfcal MSes | `mod_in_0.fits` | `-z 0.0,0.0 -Z 0.0,0.001,0.005 -R -2.0 --use-radius-mask` (uniform weighting; α plane) |
| `selfcalband9` | `hd142_b9cont_self_tav.ms` (ALMA Band 9) | `mod_in_0.fits` | `-z 0.001,3.5 -Z 0.005,0.0 --print-errors` |

Each `test.sh` follows the pattern:

```bash
test=$($1 -i $2/<dataset>.ms -o $2/residuals.ms -O $2/mod_out.fits \
          -m $2/mod_in_0.fits -p $2/mem/ -X 16 -Y 16 -V 256 ... )
valid $test           # echoes OK / ERROR + exit 1 on failure
rm -rf $2/residuals.ms $2/mem/ $2/alpha.fits $2/mod_out.fits
```

Wired into CTest by `CMakeLists.txt:512-519`.

---

## 11. Integration & extension points

### 11.1 Adding a new regulariser

1. Create `include/myreg.cuh` deriving from `Fi`
   (`include/classes/fi.cuh`) and override `calcFi`, `calcGi`, `restartDGi`,
   `addToDphi`, plus optionally `setPrior`, `setEta`, `calculateSecondDerivate`.
2. Implement the corresponding host wrappers and `__global__` kernels (e.g.
   `MyRegVector`, `DMyReg`) in `src/myreg.cu`.
3. Register the class with the factory at the bottom of `src/myreg.cu`:
   ```cpp
   namespace {
   Fi* CreateMyReg() { return new MyReg; }
   const std::string name = "MyReg";
   const bool RegisteredMyReg =
       registerCreationFunction<Fi, std::string>(name, CreateMyReg);
   }
   ```
4. Add to `main.cu`:
   ```cpp
   Fi* mr = createObject<Fi, std::string>("MyReg");
   mr->configure(/*penalizatorIndex=*/4, 0, 0, false);
   of->addFi(mr);
   ```
5. CMake auto-globs `src/*.cu` (`CMakeLists.txt:147`), so a clean rebuild
   suffices.

### 11.2 Adding a new optimizer

Derive from `Optimizer` (`include/classes/optimizer.cuh`), implement
`allocateMemoryGpu`, `deallocateMemoryGpu`, `optimize`, register with
`registerCreationFunction<Optimizer, std::string>("MyOpt", ...)`. The
optimizer can call `of->calcFunction(I)` and `of->calcGradient(I, xi, iter)`
without knowing which `Fi` terms are present.

### 11.3 Custom optimisation order

`Synthesizer::setOrder(void(*)(Optimizer*, Image*))` lets the user run, e.g.,
"first optimise plane 0 with positivity, then both planes". `main.cu:88-98`
shows the default; commented blocks demonstrate alternating
`setFlag(0/1/2/3)`.

### 11.4 Custom convolution / gridding kernel

Derive `CKernel` (`include/classes/ckernel.cuh`) — implement `buildKernel`,
`buildGCF`, `clone`, optionally `GCF`, `getAlpha`, `getW2`. Register with
the `CKernel` factory.

### 11.5 New weighting scheme

Derive `WeightingScheme` (`include/classes/weightingscheme.cuh`) and
implement `apply(vector<MSDataset>&)` and `configure(void* params)`.

### 11.6 New input/output format

Derive `Io` (`include/classes/io.cuh`) — note the very long list of virtual
methods (~50). Practically, you copy `iofits.cu` or `ioms.cu` and replace
the relevant casacore / CFITSIO calls.

### 11.7 Self-tuning regularisation (Belge fixed-point)

Replace `sy->run()` with `fixedPointOpt(initial_lambdas, &runGpuvmem,
1e-6, 60, sy)` (uncomment `main.cu:211-221`). `runGpuvmem` (`main.cu:59-86`)
implements the fixed-point update.

---

## 12. Caveats / limitations / TODOs

1. **Global state**. `M, N, image_count, penalizators, num_gpus,
   datasets, host_I, …` are file-scope globals declared `extern` in
   headers (`include/framework.cuh:44-47, fi.cuh:8-11`) and defined in
   `mfs.cu`. This is convenient inside kernels but means only one
   `Synthesizer` can exist per process.
2. **Single precision**. All visibility arithmetic and FFTs are float; UVW
   are double. There is no double-precision build path.
3. **Single Stokes**. The synthesizer iterates over `nstokes` (`mfs.cu:564,
   822`) and sums into a single image; full polarimetric reconstruction is
   **not** implemented — the output FITS image is intensity (or I+α).
4. **CPU gridding only**. `do_gridding` is OpenMP-parallel on the host
   (CKernel-based), then transferred to GPU. There is no GPU gridding path.
5. **Hard-coded GPU memory growth**. `device_g, device_h, xi, temp` in
   `frprmn.cu:58-82` and `d_y, d_s` in `lbfgs.cu:67-75` scale linearly with
   `M*N*image_count`. For LBFGS the factor is `K=100` by default
   (`lbfgs.cuh:30`), which may exceed device memory on large grids.
6. **Help text**. `print_help` and `-w/-c` flags currently call the same
   `print_help` (`functions.cu:271-284`) — `--warranty` and `--copyright`
   don't actually print the GPLv3 warranty/copyright (note: the strings
   *exist* in `src/copyrightwarranty.cu` but are not wired in here).
7. **README install instructions are stale**. They mention CUDA 9–11 and
   casacore v3.2.1 (`README.md:69, 37`); the actual `Dockerfile` uses CUDA
   12.4.1 and casacore v3.5.0, and `CMakeLists.txt` accepts CUDA 13+.
8. **CASA 6.6.4 dependency in `restore.py`**. `requirements.txt` pins
   `casatasks==6.6.4.34` etc.; mismatched CASA versions will fail.
9. **`docs/index.rst`**. One-line stub — there is no Sphinx documentation
   shipped; the project relies on the GitHub wiki at
   <https://github.com/miguelcarcamov/gpuvmem/wiki>.
10. **`getGandDGG` vs `getGGandDGG`**. Two near-identical declarations exist
    in `include/functions.cuh:347` and elsewhere — code style debt.
11. **TODO marker** in `src/functions.cu:4255`:
    *"Here we could just use vis_mod and see what happens"* — note about
    interchangeability of `vis_mod`/`vis_mod2`.
12. **Computes `compute_53` is excluded** (`CMakeLists.txt:218, 261, 323`)
    — Tegra/Jetson Nano (Maxwell-mobile) is unsupported.
13. **No native Windows build** (only POSIX `getopt_long`, OpenMP via gomp,
    `mkdir(0700)` in `Io::createFolder`).
14. **Tests assume git-lfs**. Without `git lfs install && git lfs pull` the
    `.ms` files are LFS pointers and CTest fails silently.

---

## 13. Quick reference: kernel cheat-sheet

For users wanting to read the GPU code, a minimal map of where the key math
lives:

| What | File:line |
|------|-----------|
| Top-level driver | `src/main.cu:100` |
| MS read | `src/MSFITSIO.cu` (`readMS`) |
| FFT plan creation | `src/functions.cu` (`initFFT`) |
| Image → V (forward) | `src/functions.cu` (`FFT2D`, `total_attenuation`, `apply_beam`, `linkCalculateInu2I`) |
| Bilinear interp at (u,v) | `src/functions.cu:2557` (`vis_mod`) |
| Residual `Vo - Vm` | `src/functions.cu:2663` (`residual`) |
| χ² value | `src/functions.cu:2867` (`chi2Vector`) + `deviceReduce` |
| dχ²/dI on device | `src/functions.cu` (`DChi2`) |
| Entropy S | `src/functions.cu` (`SVector` / `SEntropy`) |
| dS/dI | `src/functions.cu` (`DS` / `DEntropy`) |
| TV | `src/functions.cu` (`TVVector` / `totalvariation`) |
| dTV/dI | `src/functions.cu` (`DTV` / `DTVariation`) |
| L1 | `src/functions.cu:2902` (`L1Vector` / `L1Norm`) |
| dL1 | `src/functions.cu:2938` (`DL1NormK` / `DL1Norm`) |
| CG-FRPRMN loop | `src/frprmn.cu:85` |
| LBFGS loop | `src/lbfgs.cu` |
| Line search | `src/linmin.cu`, `src/mnbrak.cu`, `src/brent.cu`, `src/f1dim.cu` |
| Belge λ fixed-point | `src/fixedpoint.cu` + `src/main.cu:59` |

---

*End of `simulators/gpuvmem.md`.*
