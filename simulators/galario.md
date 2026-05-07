# galario — Exhaustive Technical Reference

> **GAL**axy **A**nalysis with **R**adio **I**nterferometry **O**bservations
>
> *GPU Accelerated Library for Analysing Radio Interferometer Observations*

This document is a comprehensive technical reference for the **galario** package as it sits in the RRIVis vendor tree at `simulators/galario/`. It is built only from sources inside that directory: the CMake build files, the C++/CUDA core (`src/`), the Cython/Python bindings (`python/`), the reStructuredText documentation (`docs/`), the GitHub Actions workflow, and the git history. Every concrete claim below traces back to one of those files, with the relative path called out wherever practical.

---

## 1. Project overview

`galario` is a small, focused C++/CUDA library, dual-built for CPU (FFTW3 + OpenMP/pthreads) and GPU (cuFFT + cuBLAS), exposing two precisions (single / double) and four importable Python sub-modules. Its single purpose is to make the same calculation extremely fast:

> Given a model image (or an axisymmetric radial brightness profile) and a list of `(u, v)` baseline coordinates, return either (a) the **synthetic complex visibilities** sampled at those `(u, v)` points, or (b) the **chi-squared** between those synthetic visibilities and a vector of observed visibilities with weights.

That short loop is the kernel of every Bayesian fit of (sub-)mm interferometer data — typically protoplanetary-disk continuum data from ALMA — and the README and `docs/quickstart.rst` framing make it explicit that the library is designed to be the inner-loop accelerator for `emcee`-style MCMC fits.

The README (`README.md` lines 20-22) puts it succinctly:

> *"galario is a library that exploits the computing power of modern graphic cards (GPUs) to accelerate the comparison of model predictions to radio interferometer observations. Namely, it speeds up the computation of the synthetic visibilities given a model image (or an axisymmetric brightness profile) and their comparison to the observations."*

galario is **not** a full RIME engine: it does not handle wide-field (w-projection), Jones matrices, polarisation, multiple frequencies, primary beams, or time-dependent baselines — see `docs/tech-specs.rst` (lines 12-62, the commented-out "Assumptions" block) for an honest summary of what is deliberately out of scope. The complement of those omissions is exactly what RIMEz, RASCIL, WSClean and friends provide; galario sits in a different niche.

### 1.1 Authority and citation

| Field | Value | Source |
|------|------|--------|
| Authors | Marco Tazzari (Cambridge), Frederik Beaujean, Leonardo Testi (ESO) | `AUTHORS.rst` lines 4-6 |
| Contributions | Luca Di Mascolo, Nathanial Hendler | `AUTHORS.rst` lines 8-10 |
| Reference paper | Tazzari, Beaujean & Testi (2018) MNRAS **476** 4527 | `README.md` lines 28-46 |
| arXiv | `1709.06999` | same |
| DOI | `10.1093/mnras/sty409` (paper); `10.5281/zenodo.889991` (software) | `README.md`, `.zenodo.json` |
| ADS bibcode | `2018MNRAS.476.4527T` | `README.md` |

### 1.2 License

**GNU Lesser General Public License, version 3 (LGPLv3)** — the full 7651-byte text is in `LICENSE`. Every source file (`src/galario.cpp`, `src/galario.h`, `src/galario_defs.h`, `src/galario_py.h`, `src/galario_test.cpp`, `python/libcommon.pyx`, etc.) carries the same boilerplate header citing the license. The Zenodo metadata (`.zenodo.json`) classifies the upload as `"other-open"` software with `"access_right": "open"`. © 2017-2020 Marco Tazzari, Frederik Beaujean, Leonardo Testi.

### 1.3 Version, languages, dates

| Item | Value | Source |
|------|------|--------|
| `PACKAGE_VERSION` (string used in CMake & `__init__.py.in`) | `'1.2.2'` | `CMakeLists.txt` line 22 |
| Most recent CHANGELOG entry | `1.2.2 (2020-02-28)` | `CHANGELOG.rst` line 1 |
| First release | `1.0 (2017-09-19)` | `CHANGELOG.rst` line 44 |
| Total number of commits in this checkout | 536 | `git log --oneline | wc -l` |
| Existing tags | `0.1`, `0.3b`, `v0.2`, `v0.3`, `v1.0`, `v1.0.1`, `v1.0.2`, `v1.0.3`, `v1.1`, `v1.2`, `v1.2.1`, `v1.2.2` | `git tag` |
| Latest commit on `master` | `8949db4 drop support for Python 3.6` | `git log -1 --oneline` |
| Languages used | C++11 (with CUDA C++ via NVCC), Cython, Python ≥ 3.7, CMake, Bash | source tree |
| Supported Python versions (current) | 3.7 / 3.8 / 3.9 | `README.md` line 9, GitHub Actions matrix |
| Supported OS | Linux + macOS; Windows unsupported | `docs/install.rst` line 8 |

`src/galario.cpp` is 1664 lines — the vast majority of the library. The Python wrapper `python/libcommon.pyx` is 1082 lines. `python/utils.py` adds another 507 lines of pure-Python reference implementations and helpers used by the test suite. `python/test_galario.py` is 657 lines.

---

## 2. Repository layout

```
simulators/galario/
├── README.md                       # short marketing readme + BibTeX entry
├── AUTHORS.rst                     # author list
├── CHANGELOG.rst                   # versioned change log (releases 1.0 → 1.2.2)
├── LICENSE                         # full LGPLv3 text
├── CMakeLists.txt                  # top-level CMake (links src/, python/, docs/)
├── .gitignore                      # ignores build/, __pycache__, *.so, *.egg-info, ...
├── .codecov.yml                    # codecov coverage thresholds (range 70-100, 1% drop)
├── .zenodo.json                    # Zenodo metadata (DOI 10.5281/zenodo.889991)
├── .github/
│   └── workflows/
│       └── unit-tests.yml          # GitHub Actions: build + pytest + docs deploy
├── cmake/
│   ├── FindCython.cmake            # locate cython executable next to PYTHON_EXECUTABLE
│   ├── FindSphinx.cmake            # locate sphinx-build
│   ├── UseCython.cmake             # provides cython_add_module()
│   ├── LookUp-GreatCMakeCookOff.cmake  # downloads UCL/GreatCMakeCookOff (FFTW3/Numpy finders)
│   └── cmake_uninstall.cmake.in    # `make uninstall` driver, reads install_manifest.txt
├── src/
│   ├── CMakeLists.txt              # builds 4 library targets (single/double × cpu/cuda)
│   ├── galario.h                   # public C++ API in `namespace galario`
│   ├── galario_py.h                # void*-typed mirror API for Cython
│   ├── galario_defs.h              # `dreal`, `dcomplex` typedefs gated by DOUBLE_PRECISION
│   ├── galario.cpp                 # 1664 LoC: every kernel for both CPU and GPU
│   └── galario_test.cpp            # tiny smoke test, also used as a link sanity check
├── python/
│   ├── CMakeLists.txt              # wraps libgalario into 4 cython extension modules
│   ├── wrap_lib.cmake              # function `wrap_lib(DOUBLE? CUDA?)`
│   ├── galario_defs.pxd            # external decls of galario:: functions for cython
│   ├── galario_config.pxi.in       # configure-substituted: defines DOUBLE_PRECISION + dtypes
│   ├── __init__.py.in              # top-level galario/__init__.py template
│   ├── __init_module__.py.in       # per-flavour __init__.py (`single`, `double`, `*_cuda`)
│   ├── libcommon.pyx               # Cython implementation of the user API
│   ├── utils.py                    # py-side reference implementation (py_sampleImage, ...)
│   ├── test_galario.py             # pytest suite (657 LoC, used by CI)
│   ├── conftest.py                 # adds --gpu pytest CLI option
│   ├── speed_benchmark.py          # benchmarking driver (timeit + argparse)
│   └── speed_baseline.sh           # shell wrapper that sweeps sizes/threads
└── docs/
    ├── CMakeLists.txt              # wires `make docs` to sphinx-build
    ├── conf.py.in                  # Sphinx config template
    ├── index.rst                   # landing page (toctree)
    ├── install.rst                 # build & install (CPU + GPU)
    ├── basic_usage.rst             # 4-paragraph quickstart
    ├── quickstart.rst              # full emcee-based MCMC example
    ├── tech-specs.rst              # image origin convention, assumptions
    ├── cookbook.rst                # GPU/CPU selection, threading, meshgrid recipe
    ├── py-api.rst                  # autodoc'd Python API
    ├── C++-api.rst                 # hand-written C++ API page
    ├── C++-example.rst             # walks src/galario_test.cpp line-by-line
    ├── publications.rst            # links to ADS citation list
    ├── FAQ.rst                     # one entry: C-contiguous arrays
    ├── license.rst
    ├── uvtable.txt                 # ASCII demo dataset for quickstart
    ├── images/                     # JPGs/PNGs used by the docs
    ├── _templates/layout.html      # Sphinx layout override
    └── _static/css/custom.css      # CSS override
```

There is no `setup.py`, no `pyproject.toml`, no `requirements.txt`, no `Dockerfile`, no `MANIFEST.in`. Installation is **CMake only** — `pip` and PyPI are not used. Distribution is via `conda-forge` (binary CPU build) or a manual `cmake .. && make && make install` (which is the only way to get the GPU build, see `docs/install.rst` lines 27-29).

The `.git` entry is a git-link to the parent's `.git/modules/...`, i.e. galario is consumed as a **git submodule**.

---

## 3. License, citation, intellectual context

**LGPLv3.** Practically: you may dynamically link `libgalario.so` / `libgalario_cuda.so` from a closed-source application; modifications to the library itself must be re-released under LGPLv3. Both the per-file headers (e.g. `src/galario.cpp` lines 1-18) and the LICENSE file are explicit.

**Citation policy.** `README.md` lines 28-46 and `docs/index.rst` lines 79-101 mandate citing Tazzari et al. 2018 MNRAS 476, 4527 for any scientific use. A separate Zenodo DOI (`10.5281/zenodo.889991`, in `.zenodo.json` → `related_identifiers`) lets users cite a specific software version.

**Scientific scope.** From `docs/tech-specs.rst` (the verbatim, but commented, technical-spec block) the assumptions are:

1. *Small-field imaging* — coplanar baselines assumed; no w-projection.
2. *Primary-beam handling* — `*Image()` takes the primary-beam-corrected brightness; `*Profile()` does not apply primary-beam correction internally.
3. *Single average frequency* `ν₀` — no spectral cube; users channel-average their MS.
4. *Bandwidth smearing* and *time smearing* are user concerns.
5. *Multiple sources* — either bake them into one image, or sum visibilities from independent calls (linearity of FT).

These constraints define what users get from galario versus from a full-blown RIME engine.

---

## 4. Build system

### 4.1 Top-level `CMakeLists.txt`

`cmake_minimum_required(VERSION 3.0)`. It:

1. Defaults `CMAKE_BUILD_TYPE` to `Release` if the user did not set it.
2. **Auto-installs into the active conda env** (`CMakeLists.txt` lines 31-35): if `CONDA_PREFIX` is in the environment and the user has not overridden `CMAKE_INSTALL_PREFIX`, it sets the prefix to `$CONDA_PREFIX` and prints `"Installing inside conda env at ..."`. This was added in 1.2.1 (`CHANGELOG.rst` line 14).
3. Appends `cmake/` to `CMAKE_MODULE_PATH` and runs `include(LookUp-GreatCMakeCookOff)`. That module clones `https://github.com/UCL/GreatCMakeCookOff.git` into `${CMAKE_BINARY_DIR}/external/src/GreatCMakeCookOff` if not present (`cmake/LookUp-GreatCMakeCookOff.cmake` lines 17-49) — this is the "1.5 MB external library download" mentioned in `docs/install.rst` line 35. It supplies `FindFFTW3`, `FindNumpy`, `PythonInstall`, `AddPyTest`, etc.
4. Resolves the `GALARIO_CHECK_CUDA` switch:
   - If undefined and `CMAKE_SYSTEM_NAME == "Darwin"` → `0` (skip CUDA on macOS by default; this was introduced as the v1.0.1 change in `CHANGELOG.rst` line 42).
   - Otherwise → `1`.
   - Users override with `cmake -DGALARIO_CHECK_CUDA=0` or `=1`.
5. Calls `find_package(CUDA)` (only the legacy module form; `enable_language(CUDA)` is commented out).
6. `enable_testing()`, then forces `BUILD_SHARED_LIBS=ON` ("we can't use static libs with python", line 64), then `add_subdirectory(src)`.
7. Discovers Cython via `find_package(Cython)`. If found, looks for Python libs and NumPy headers; if all three are found, sets `BUILD_WRAPPER=TRUE` and adds the `python/` subdirectory.
8. Optional `find_package(Sphinx)`; if found, adds `docs/` and registers a `docs` target (excluded from `make all`, see `docs/CMakeLists.txt` line 57: `set_target_properties(docs PROPERTIES EXCLUDE_FROM_ALL TRUE)`).
9. Wires up `make uninstall` via `cmake/cmake_uninstall.cmake.in`.

### 4.2 `src/CMakeLists.txt` — how four libraries are built from one C++ file

Key idea: the same `galario.cpp` is **compiled four times** with different preprocessor flags and different compilers (g++ vs nvcc) to produce up to four shared libraries.

```
galario_single        ← g++,   no flags        → CPU + float
galario               ← g++,   -DDOUBLE_PRECISION → CPU + double
galario_single_cuda   ← nvcc,  no flags        → GPU + float    (only if CUDA_FOUND)
galario_cuda          ← nvcc,  -DDOUBLE_PRECISION → GPU + double (only if CUDA_FOUND)
```

Steps:

* Compiler check: gcc ≥ 4.8.1 or clang ≥ 3.3 are required for full C++11 (`src/CMakeLists.txt` lines 21-35).
* `set(cxx_std "-std=c++11")`.
* `add_compile_options(-Wall -pedantic -fstrict-aliasing -pthread)` — the `-fstrict-aliasing` is deliberate because *"all input arrays to functions must not alias for correctness"* (line 42 comment).
* `add_library(galario_single ${common_cpp})` and `add_library(galario ${common_cpp})` with `target_compile_definitions(galario PUBLIC DOUBLE_PRECISION)`.
* `find_package(OpenMP)` and `find_package(Threads)`. If both found, `OpenMP_CXX_FLAGS` are added to `CMAKE_CXX_FLAGS` and to `CMAKE_SHARED_LINKER_FLAGS`, and `FFTW_PARALLEL` is set to `THREADS`.
* `find_package(FFTW3 COMPONENTS SINGLE DOUBLE ${FFTW_PARALLEL})` — galario needs **all four** FFTW libs: `libfftw3`, `libfftw3f`, `libfftw3_threads`, `libfftw3f_threads` (`docs/install.rst` lines 226-232).
* Optional `GALARIO_TIMING` flag: `cmake -DGALARIO_TIMING=1` adds `-DGALARIO_TIMING=1` to every target, enabling the embedded `GPUTimer`/`CPUTimer` machinery.
* Both CPU libraries are linked against `${FFTW3_LIBRARIES}` and given the FFTW include dirs.
* If `CUDA_FOUND`:
  - Adds NVCC flags `--gpu-architecture=compute_30 --gpu-code=compute_30` (deliberate JIT for forward-compat with newer GPUs, see comment lines 81-84). 
  - `-D_FORCE_INLINES` to work around the cuda 7.5 + Ubuntu 16.04 issue.
  - **`--default-stream per-thread`** — this is the v1.0.2 fix that lets multiple host processes share one GPU (`CHANGELOG.rst` line 36, `src/CMakeLists.txt` lines 91-93).
  - Copies `galario.cpp` to `${CMAKE_CURRENT_BINARY_DIR}/cuda_lib.cu` — necessary because *"cmake doesn't call nvcc on it"* unless the suffix is `.cu`. The trick uses `configure_file` to keep one set of sources.
  - Builds `galario_single_cuda` (no `-DDOUBLE_PRECISION`), then appends `-DDOUBLE_PRECISION` to `CUDA_NVCC_FLAGS` and builds `galario_cuda`.
  - Each cuda library gets `cuda_add_cublas_to_target()` and `cuda_add_cufft_to_target()` linkage.
* Install: every target ends up under `lib/`, and three headers (`galario.h`, `galario_py.h`, `galario_defs.h`) under `include/`.
* Test: `add_executable(galario_test_cpp galario_test.cpp)` and `add_test(cpp_compile_test galario_test_cpp)` register a CPU-side smoke test.

### 4.3 `python/CMakeLists.txt` — generating four Cython modules

Same trick at the Python level: `wrap_lib.cmake` defines `wrap_lib(DOUBLE? CUDA?)` and is invoked four times:

```cmake
wrap_lib()                # → python/galario/single
wrap_lib(DOUBLE)          # → python/galario/double
wrap_lib(CUDA)            # → python/galario/single_cuda    (if CUDA_FOUND)
wrap_lib(DOUBLE CUDA)     # → python/galario/double_cuda    (if CUDA_FOUND)
```

Inside `wrap_lib()` (`python/wrap_lib.cmake`):

* Picks `outdir = $PYGALARIO_DIR/{single,double}[_cuda]`.
* `configure_file` writes `outdir/galario_config.pxi` from `galario_config.pxi.in`, substituting `GALARIO_DOUBLE_PRECISION` to `0` or `1`. That file then provides `ctypedef double dreal` (or `float`), `real_dtype = np.float64` (or `np.float32`), `complex_dtype = np.complex128` (or `np.complex64`), and the corresponding `complex_typenum` (`NPY_COMPLEX128` or `NPY_COMPLEX64`).
* Writes a per-flavour `__init__.py` from `__init_module__.py.in`. It is just three lines: `from .libcommon import *`, then `_init()` and `atexit.register(_cleanup)`.
* Adds the cython module via `cython_add_module(libcommon${suffix} libcommon.pyx)`. The output library is then renamed to `libcommon` (`set_target_properties OUTPUT_NAME libcommon`) so the Python import is identical regardless of the underlying flavour. CMake handles the disambiguation by placing the `.so` in different directories.
* Links the cython module against the matching C++ library (`galario`, `galario_single`, `galario_cuda`, `galario_single_cuda`).
* `target_compile_options(${libcommon} PUBLIC "-Wno-cpp")` to silence Cython/NumPy deprecation noise.

The top-level Python package `galario/__init__.py` (rendered from `__init__.py.in`, lines 20-32) does:

```python
__version__ = '1.2.2'
HAVE_CUDA = ("${CUDA_FOUND}" == "TRUE")  # baked at build time
if HAVE_CUDA:
    from . import single_cuda
    from . import double_cuda
from . import single
from . import double
from .double import arcsec, au, cgs_to_Jy, pc, deg
```

So `galario.HAVE_CUDA` is a **build-time** truth value, not a runtime check, and the constants `arcsec, au, cgs_to_Jy, pc, deg` are re-exported from `galario.double`.

### 4.4 Documentation build

`docs/CMakeLists.txt` configures Sphinx via a templated `conf.py.in`, copies `_static`/`_templates` into the build tree, and registers a `docs` target driven by `sphinx-build -b html`. `make docs` (after `cmake ..`) produces HTML in `build/docs/html`. The CI workflow then deploys that to `gh-pages` for `master` builds with Python 3.7 (`unit-tests.yml` lines 65-71).

### 4.5 Continuous integration

`.github/workflows/unit-tests.yml`:

| Stage | Action |
|------|------|
| Trigger | `on: push` |
| Runner | `ubuntu-20.04` |
| Python matrix | `[3.7, 3.8, 3.9]` |
| Env | `OMP_NUM_THREADS: 2` |
| Setup | `conda-incubator/setup-miniconda@v2` from `conda-forge`, `use-only-tar-bz2: true` |
| Build deps | `apt-get install libfftw3-dev`, conda installs `astropy cython nomkl numpy pytest scipy sphinx`, pip installs `coverage codecov pytest-cov` |
| Build | `cmake -DCMAKE_INSTALL_PREFIX=/tmp -DCMAKE_PREFIX_PATH=$CONDA_PREFIX ..` then `make && make install` |
| Tests | `python/py.test.sh -sv --cov=./ python/test_galario.py` (run from `build/`) |
| Coverage | `codecov.io/bash` uploader |
| Docs deploy | `JamesIves/github-pages-deploy-action@4.1.0` to `gh-pages`, only on `master` ∧ Python 3.7 |

Note that CI never tests the GPU path (no nvidia-smi runners on GitHub Actions). The CPU + CUDA matrix is only validated by external testers (Tazzari's lab, conda-forge feedstock).

---

## 5. CPU vs GPU branch selection

The CPU-vs-GPU split is **purely compile-time**, controlled by whether the source file is fed to nvcc (which defines the implicit macro `__CUDACC__`) or to g++/clang++. Throughout `src/galario.cpp` you see pairs of guarded blocks:

```cpp
#ifdef __CUDACC__
// CUDA implementation
#else
// CPU + OpenMP/FFTW implementation
#endif
```

The single-vs-double precision split is also compile-time, controlled by the `DOUBLE_PRECISION` preprocessor symbol set in `src/CMakeLists.txt` (`target_compile_definitions(galario PUBLIC DOUBLE_PRECISION)` line 53; CUDA equivalent line 114).

`src/galario_defs.h` then collapses both knobs into two typedefs:

```cpp
#ifdef DOUBLE_PRECISION
    typedef double dreal;
    #ifdef __CUDACC__
        typedef cufftDoubleComplex dcomplex;
    #else
        typedef std::complex<dreal> dcomplex;
    #endif
#else
    typedef float dreal;
    #ifdef __CUDACC__
        typedef cufftComplex dcomplex;
    #else
        typedef std::complex<float> dcomplex;
    #endif
#endif
```

So `dreal`/`dcomplex` is the abstract numeric type used everywhere in `galario.cpp`. On the GPU side it is the cuFFT complex struct; on the CPU side it is `std::complex<...>`. This lets one source file produce four binaries with no manual code duplication.

Inside `galario.cpp`, helper macros adapt complex arithmetic to the active backend (lines 317-353):

```cpp
#ifdef __CUDACC__
  #ifdef DOUBLE_PRECISION
    #define CUFFTEXEC cufftExecD2Z      ; #define CUFFTTYPE CUFFT_D2Z
    #define CMPLXSUB cuCsub             ; #define CMPLXADD cuCadd
    #define CMPLXMUL cuCmul             ; #define CUBLASNRM2 cublasDznrm2
    #define CMPLXABS cuCabs
    #define CMPLXARG(a) atan2(cuCimag(a), cuCreal(a))
  #else
    #define CUFFTEXEC cufftExecR2C      ; #define CUFFTTYPE CUFFT_R2C
    #define CMPLXSUB cuCsubf            ; ... (f-suffix variants)
    #define CUBLASNRM2 cublasScnrm2
  #endif
#else // CPU
  #define CMPLXSUB(a, b) ((a) - (b))    ; #define CMPLXADD(a, b) ((a) + (b))
  #define CMPLXMUL(a, b) ((a) * (b))    ; #define CMPLXCONJ conj
  #define CMPLXABS abs                  ; #define CMPLXARG arg
#endif
```

The `FFTW(name)` token-paste macro on line 358 selects `fftw_*` (double) or `fftwf_*` (single) without code duplication.

---

## 6. The C public API (`namespace galario`)

`src/galario.h` declares a flat C++ functional API in `namespace galario`. Two precision-correct typedefs (`dreal`, `dcomplex`) come from `galario_defs.h`. Every function in `galario.h` has a sibling in `galario_py.h` whose array parameters are `void*` instead of `dreal*`/`dcomplex*` — that simplification is what makes the Cython binding generic across the four flavours.

### 6.1 Main user functions

| Function | Purpose | Returns |
|----|----|----|
| `void sample_profile(int nr, const dreal* intensity, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal inc, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, const dreal* u, const dreal* v, dcomplex* vis_int)` | Build image from radial profile, FFT, sample at `(u, v)` | populates `vis_int[nd]` |
| `void sample_image(int nx, int ny, const dreal* image, const dreal v_origin, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, const dreal* u, const dreal* v, dcomplex* vis_int)` | FFT a 2-D image and sample at `(u, v)` | populates `vis_int[nd]` |
| `dreal chi2_profile(int nr, const dreal* intensity, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal inc, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, const dreal* u, const dreal* v, const dreal* vis_obs_re, const dreal* vis_obs_im, const dreal* weights)` | `sample_profile` + weighted chi² | `dreal` |
| `dreal chi2_image(int nx, int ny, const dreal* image, const dreal v_origin, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, const dreal* u, const dreal* v, const dreal* vis_obs_re, const dreal* vis_obs_im, const dreal* weights)` | `sample_image` + weighted chi² | `dreal` |
| `void sweep(int nr, const dreal* intensity, dreal Rmin, dreal dR, int nxy, dreal dxy, dreal inc, dcomplex* image)` | Build axisymmetric 2-D image from a 1-D profile, store in real-image-aligned complex buffer | populates `image` |
| `void uv_rotate(dreal PA, dreal dRA, dreal dDec, dreal* dRArot, dreal* dDecrot, int nd, const dreal* u, const dreal* v, dreal* urot, dreal* vrot)` | Rotate `(u, v)` and `(dRA, dDec)` by Position Angle PA | populates outputs |

### 6.2 Expert / individual operations

These are the ingredients used inside the high-level functions; the Python wrappers expose them too for advanced users.

| Function | Purpose |
|----|----|
| `dcomplex* copy_input(int nx, int ny, const dreal* image)` | Allocate FFTW-aligned complex buffer of size `nx*(ny/2+1)` and copy the real `image` in with the padding required by `fftw_plan_dft_r2c_2d`. Caller must `galario_free()` the result. |
| `void galario_free(void* data)` | Free a buffer from `copy_input()` (calls `fftw_free` on CPU, plain `free` on GPU). |
| `void fft2d(int nx, int ny, dcomplex* image)` | In-place 2-D real-to-complex FFT (FFTW or cuFFT). |
| `void fftshift(int nx, int ny, dcomplex* image)` | 2-quadrant swap of a real image stored in the R2C buffer (see §7.4). |
| `void fftshift_axis0(int nx, int ny, dcomplex* matrix)` | Axis-0-only fftshift of a complex matrix. Used post-FFT. |
| `void interpolate(int nrow, int ncol, const dcomplex* image, dreal v_origin, int nd, const dreal* u, const dreal* v, dreal duv, dcomplex* vis_int)` | Bilinear interpolation of the half-plane R2C output at `(u, v)`. |
| `void apply_phase_sampled(dreal dRA, dreal dDec, int nd, const dreal* u, const dreal* v, dcomplex* vis_int)` | Multiply each `vis_int[i]` by `exp(2πi (u·dRA + v·dDec))`. |
| `dreal reduce_chi2(int nd, const dreal* vis_obs_re, const dreal* vis_obs_im, const dcomplex* vis_int, const dreal* weights)` | Weighted sum of squared differences. |

### 6.3 Lifecycle

| Function | Purpose |
|----|----|
| `void init()` | On CPU: `fftw{,f}_init_threads()`. On GPU: no-op (cuBLAS handle is created lazily — see CHANGELOG 1.0.2 "Fix memory leak in GPU version"). |
| `void cleanup()` | On CPU: `fftw_cleanup_threads()` + `fftw_cleanup()`. On GPU: destroys cuBLAS handle if it was created. |
| `int threads(int num = 0)` | Get/set thread count. On CPU: maps to `omp_set_num_threads`. On GPU: sets the per-axis size of `dim3` blocks (square root of total threads/block). With `num=0`, just returns the current value. |
| `int ngpus()` | `cudaGetDeviceCount()` if CUDA, else `0`. |
| `void use_gpu(int device_id)` | `cudaSetDevice()` if CUDA, else no-op. |

### 6.4 The `_*` Cython-friendly mirror

Every function above has a `void*` twin in `src/galario_py.h`, e.g.:

```cpp
void _sample_image(int nx, int ny, void* data, dreal v_origin, dreal dRA, dreal dDec,
                   dreal duv, dreal PA, int nd, void* u, void* v, void* vis_int);
```

These trampolines just `static_cast` the pointers to the right type and call the typed sibling. Cython sees a uniform `void*` ABI.

### 6.5 Exception surface

The library throws C++ exceptions. The Cython `cdef extern from ... except +` declarations in `python/galario_defs.pxd` (lines 24-46) propagate them as Python exceptions according to the table in `docs/py-api.rst`:

| C++ exception | Trigger | Python exception |
|----|----|----|
| `std::bad_alloc` | OOM on CPU/GPU (or cuBLAS/cuFFT alloc failure) | `MemoryError` |
| `std::invalid_argument` | Image dimension < 2, odd, non-square; central-pixel ratio < 5 | `ValueError` |
| `std::runtime_error` | All other CUDA / cuBLAS / cuFFT failures, OOM | `RuntimeError` |

The CPU-side checks come from the `CHECK_INPUT`, `CHECK_INPUTXY`, and `CHECK_CENTRAL_PIXEL` macros (`src/galario.cpp` lines 101-121). The GPU-side checks come from `CCheck`, `CBlasCheck`, `CUFFTCheck` wrappers (lines 164-198) — every cuda call goes through one of these.

Concrete error messages, verbatim from the source, include:
* `"x dimension = {N} is less than 2"`
* `"x dimension = {N} is odd"`
* `"Expect a square image but got shape ({nx}, {ny})"`
* `"Expect (dxy/2-Rmin)/dR > 5, ... Try reducing dR."`
* `"Could not initialize cuda. Is a CUDA GPU available at all?"`

`python/test_galario.py::test_exception` (lines 617-628) asserts these propagate, e.g. `pytest.raises(ValueError, match="dimension.*is less than 2")`.

---

## 7. Algorithms — what each kernel does

The fundamental observation behind galario is that **the synthetic visibility for a small-field model image is just samples of the 2-D FFT of the image at the `(u, v)` baseline coordinates, possibly with phase shifts and rotations**. Once that fact is locked in, the question is how to do the FFT, interpolation, and reductions as fast as possible. Below, each kernel is described against `src/galario.cpp` line numbers.

### 7.1 `copy_input` — packing real input into the R2C buffer (lines 460-487)

FFTW's in-place R2C transform writes its `(nx, ny/2+1)` complex output into the same buffer that holds the `(nx, ny)` real input — but the real input rows must be *padded*: each row holds `2*(ny/2+1)` real values, not `ny`. `copy_input`:

* Validates `nx == ny` and even (`CHECK_INPUTXY`).
* Allocates `nx * (ny/2+1)` complex elements with `fftw_alloc_complex` (CPU, for SIMD alignment) or plain `malloc` (GPU side, via the GPU-only `copy_input_d` helper at lines 435-448 which pads using `cudaMemcpy2D`).
* Copies row-by-row in a `#pragma omp parallel for`.
* Returns the buffer; the caller owns it.

### 7.2 FFT (lines 494-547)

`fft2d` (CPU): builds an FFTW plan on every call (`fftw_plan_dft_r2c_2d`, `FFTW_ESTIMATE` flag), executes, destroys. `// TODO: find a way to store the plan` (line 523). On systems with OpenMP, `fftw_plan_with_nthreads(galario::threads())` is invoked first. Plan caching is **not implemented** — the runtime cost of `FFTW_ESTIMATE` is small, but a bigger win is left on the table for repeated identical-shape calls.

`fft_d` (GPU): `cufftPlan2d`, `CUFFTEXEC`, `cudaDeviceSynchronize`, `cufftDestroy`. Same per-call cost issue.

### 7.3 `shift_core` / `fftshift` — quadrant swap on the R2C buffer (lines 565-643)

Standard `numpy.fft.fftshift` swaps the four quadrants of a 2-D matrix. Here the matrix is real but stored inside an FFTW R2C complex buffer with padding. Lines 565-589 do the swap by re-interpreting the complex buffer as `dreal*`, computing source/target indices that account for the row stride `2*(ny/2+1)`, and exchanging two pairs (UL↔LR, UR↔LL) per `(idx_x, idx_y)` to avoid a conditional. Both halves are processed; the iteration is over `[0, nx/2) × [0, ny/2)`.

`shift_axis0_core` (lines 657-668) is the simpler 1-axis swap used after the FFT to make the half-plane R2C output centred along v.

### 7.4 `interpolate` — bilinear interpolation of the half-plane R2C output (lines 749-815)

The Numerical-Recipes-style bilinear interpolation formula

```
vis(u, v) = (1-t)(1-q) y0 + t(1-q) y1 + t·q·y2 + (1-t)·q·y3
```

is rewritten as

```
vis = t·q·(y0 - y1 + y2 - y3) + t·(y1 - y0) + q·(y3 - y0) + y0
```

The crucial detail is that the FFT output is **half-plane** (`(nrow, ny/2+1)`): only `u >= 0` is stored. For `u < 0`, the code uses Hermitian symmetry by reflecting through the v-axis, sign-flipping the imaginary part:

```cpp
dreal const indu = fabs(u)/duv;
dreal const sign_u = copysign(1., u);
indv = half_nrow + v_origin * sign_u * v / duv;
```

Then it interpolates **amplitude** and **phase** separately (the `interp_amp` and `interp_phase` blocks, lines 802-810). This separation is deliberate: `docs/py-api.rst` notes it improves accuracy near zero-crossings of the FT. The 1.2 release adds: `[core/bugfix] More robust DFT interpolation for sources that are large or hugely offset from phase center.` (`CHANGELOG.rst` line 19).

### 7.5 `apply_phase_sampled` — per-source translation by a Fourier phase (lines 877-923)

Translation in image space ⇔ multiplication by `exp(2πi(u·dRA + v·dDec))` in Fourier space. Lines 891-895 short-circuit on `(dRA, dDec) == (0, 0)`. Otherwise:

```cpp
dRA  *= 2.*M_PI;
dDec *= 2.*M_PI;
for each (u_i, v_i):
    angle = u_i*dRA + v_i*dDec;
    vis_int[i] *= dcomplex{cos(angle), sin(angle)};
```

GPU launch: `apply_phase_sampled_d<<<nd/(tpb*tpb)+1, tpb*tpb>>>(...)`.

### 7.6 `uv_rotate` — Position-Angle rotation (lines 955-1038)

Rotates `(u, v)` *and* `(dRA, dDec)` by an angle `PA` (radians, East-of-North). Short-circuits if `PA == 0`. CPU path uses `#pragma omp parallel for`; GPU path launches `uv_rotate_d` then runs `uv_rotate_core` on the host for the scalars.

Note: the rotation is applied **to the `(u, v)` points by `-PA`** (not to the image), as documented in `docs/py-api.rst` lines 32-33. This is faster than rotating the image, which would require a second FFT.

### 7.7 `sweep` — image creation from a radial intensity profile (lines 1041-1255)

For an axisymmetric profile `I(R)` sampled on a linear radial grid `(Rmin, Rmin+dR, ..., Rmin+(nr-1)dR)`, the image is built by:

1. Looping over the `(2·rmax)²` pixels around the centre (where `rmax = min(ceil((Rmin + nr·dR)/dxy), nxy/2)`).
2. For each pixel computing `r = sqrt((x/cos_inc)² + y²)` — the **stretching of the x-axis by `cos(inc)`** is how galario implements an inclined disc at zero PA with no second interpolation.
3. Finding `iR = floor((r - Rmin) / dR)` and linearly interpolating `intensity[iR]` and `intensity[iR+1]`.
4. Multiplying by `sr_to_px = dxy²` (Jy/sr → Jy/pixel).
5. Setting the **central pixel** to the average flux inside that pixel, computed as the trapezoidal-rule integral of `2π R I(R) dR` from `Rmin` to `dxy/2`, divided by the pixel area `(dxy/2)² - Rmin²`. The CPU implementation is at lines 1186-1234; the GPU version at lines 1110-1161 is identical except the radial integral is done on the host before launching `central_pixel_d<<<1,1>>>(...)`.

The check `(dxy/2 - Rmin)/dR > 5` (`CHECK_CENTRAL_PIXEL`, line 113) ensures enough radial samples fall inside the central pixel for the integral to be reliable — this is the v1.1 fix `[core/bugfix] More robust interpolation of brightness profile in the central pixel for steep f(R) profiles.` (`CHANGELOG.rst` line 28).

A pure-Python equivalent, `g_sweep_prototype`, lives in `python/utils.py` lines 209-242 and is used by the test suite for cross-checking.

### 7.8 `sample_image` / `sample_profile` — putting it together (lines 1258-1444)

The *sample-from-image* pipeline is:

```
sample_image:
    copy_input          (real → padded complex buffer)
    sample_h / sample_d:
        fftshift              (centre image before FFT)
        fft2d                 (R2C 2-D FFT)
        fftshift_axis0        (centre v)
        uv_rotate             (PA rotation of u, v, dRA, dDec)
        interpolate           (bilinear @ (urot, vrot))
        apply_phase_sampled   (translation phase)
    free buffer
```

The *sample-from-profile* pipeline is identical except that `sample_image::copy_input` is replaced by `create_image_h` / `create_image_d` (which is `sweep`) with `v_origin = 1` (i.e. `'upper'`).

The GPU variant, `sample_d` (lines 1258-1321), keeps every intermediate array on the device and does only one device→host transfer at the end.

### 7.9 `chi2_*` — sample + weighted reduction (lines 1556-1663)

```
chi2_image:
    sample_image      → vis_int[nd]
    reduce_chi2(vis_obs_re, vis_obs_im, vis_int, weights)
```

`reduce_chi2` is implemented in two ways:

* **CPU** (lines 1518-1530): an OpenMP `#pragma omp parallel for reduction(+:chi2)` loop, computing the weighted residual `chi = sqrt(w_i)·(vis_int_i − (Re_i + i·Im_i))` and accumulating `|chi|²`.
* **GPU** (lines 1481-1502): `diff_weighted_d` writes the weighted residuals back into `vis_int` (in place), then `cublasDznrm2`/`cublasScnrm2` computes the Euclidean norm; the chi² is the squared norm.

Using the cuBLAS norm function is the trick that makes the chi² reduction extremely fast on GPU — no custom reduction kernel needed.

### 7.10 Ordering: shift → FFT → axis-0 shift, not the obvious shift–FFT–shift

Note that `sample_d`/`sample_h` do `fftshift` *before* FFT and `fftshift_axis0` *after*, not a full `fftshift` after. The reason: after R2C, the matrix is `(nx, ny/2+1)`, the v-axis is sampled from `0..nx-1` (but represents `-N/2..N/2-1` modes due to the centring), and only axis 0 needs centring; axis 1 is already half-plane. The `interpolate` kernel then knows it must use Hermitian symmetry for negative u (see §7.4).

---

## 8. Python API

Four importable sub-packages, all with the same surface:

```python
import galario                         # holds HAVE_CUDA, __version__
from galario import double             # CPU, double precision
from galario import single             # CPU, single precision
from galario import double_cuda        # GPU, double precision (only if HAVE_CUDA)
from galario import single_cuda        # GPU, single precision (only if HAVE_CUDA)
```

Inside each, `from .libcommon import *` exposes the names below. The top-level `galario/__init__.py` also re-exports the constants from `galario.double`.

### 8.1 Constants (`python/libcommon.pyx` lines 43-47)

| Name | Value | Meaning |
|----|----|----|
| `arcsec` | `4.84813681109536e-06` | radians per arcsecond |
| `deg` | `0.017453292519943295` | radians per degree |
| `cgs_to_Jy` | `1e23` | 1 Jy = 1e-23 erg/(s cm² Hz) |
| `pc` | `3.0856775815e18` | cm (IAU 2015 Resolution B2) |
| `au` | `1.49597870700e13` | cm (IAU 2012 Resolution B1) |

### 8.2 Lifecycle and runtime control

| Function | Sig | Effect |
|----|----|----|
| `_init()` | `()` | Calls `cpp.init()`. Auto-called by `__init_module__.py.in`. |
| `_cleanup()` | `()` | Calls `cpp.cleanup()`. Registered with `atexit`. |
| `set_v_origin(origin)` | `'upper'`→`+1.`, `'lower'`→`-1.`, else AssertionError | Maps the matplotlib-style origin convention to the v-axis sign |
| `ngpus()` | `() -> int` | Number of CUDA GPUs |
| `use_gpu(device_id)` | `(int)` | `cudaSetDevice` |
| `threads(num=0)` | `(int) -> int` | Get/set OMP threads (CPU) or `√(threads-per-block)` (GPU) |

### 8.3 Domain-specific helpers

| Function | Behaviour |
|----|----|
| `check_obs(vis_obs_re, vis_obs_im, vis_obs_w, vis=None, u=None, v=None)` | Length consistency check |
| `check_image_size(u, v, nxy, dxy, duv, PB=0, verbose=False)` | Validates Nyquist criteria. Specifically: `Nxy*dxy/MRS > 1` and `Nxy*duv/(2·max(u,v)) > 2`; if `PB != 0`, also `Nxy*dxy/PB > 1`; and `max(|u|)/duv ≤ nxy/2 + 1`, `max(|v|)/duv ≤ nxy/2`. The MRS used is `0.6/min(uvdist)` (`libcommon.pyx` lines 257-287). |
| `get_image_size(u, v, PB=0, f_min=5., f_max=2.5, verbose=False) → (nxy, dxy)` | Suggests a square image power-of-two `nxy` and `dxy` consistent with the data. `nxy = 2**ceil(log2(max(uvdist)*2*f_max / (1/MRS/f_min)))`. If `PB != 0`, doubles `nxy` until `dxy*nxy >= PB`. |
| `get_coords_meshgrid(nrow, ncol, dxy=1., inc=0., Dx=0., Dy=0., origin='upper') → (x, y, x_m, y_m, R_m)` | Produces the `(R.A., Dec.)` mesh used to evaluate `f(R_m)` — the Cookbook recipe. The x-axis is shrunk by `cos(inc)`. |

### 8.4 Scientific API

These are the four functions described in `docs/py-api.rst`. All take `dreal[::1]` typed memoryviews — i.e. C-contiguous 1-D arrays of the matching float dtype (this is the cause of the FAQ entry on `np.ascontiguousarray`).

```python
sampleImage(image, dxy, u, v, dRA=0, dDec=0, PA=0, check=False, origin='upper')
sampleProfile(intensity, Rmin, dR, nxy, dxy, u, v,
              dRA=0, dDec=0, PA=0, inc=0, check=False)
chi2Image(image, dxy, u, v, vis_obs_re, vis_obs_im, vis_obs_w,
          dRA=0, dDec=0, PA=0, check=False, origin='upper')
chi2Profile(intensity, Rmin, dR, nxy, dxy, u, v, vis_obs_re, vis_obs_im, vis_obs_w,
            dRA=0, dDec=0, PA=0, inc=0, check=False)
```

Returned by `sample*`: a `complex_dtype` array of length `len(u)` (Jy). Returned by `chi2*`: a Python float (un-normalised chi-square).

Argument units (verbatim from `python/libcommon.pyx`):

| Argument | Units |
|----|----|
| `image` | Jy/pixel |
| `intensity` | Jy/sr |
| `dxy`, `dRA`, `dDec`, `Rmin`, `dR`, `PA`, `inc` | radians |
| `u`, `v` | observing wavelengths |
| `vis_obs_re`, `vis_obs_im` | Jy |
| `vis_obs_w` | inverse variance, weights |

The Cython `dreal[::1]` type means `image` and the `u`/`v`/`vis_obs_*` arrays must match the active flavour: `np.float64` for `galario.double[_cuda]`, `np.float32` for `galario.single[_cuda]`.

### 8.5 Lower-level helpers

| Function | Behaviour |
|----|----|
| `sweep(intensity, Rmin, dR, nxy, dxy, inc=0)` | Build the 2-D intensity image (Jy/pixel). Note: the Cython implementation creates a `(nxy, nxy/2+1)` complex buffer, fills it via `_sweep`, then returns `np.ascontiguousarray(image.view(real_dtype)[:, :-2])` — i.e. you get back the **real** image of shape `(nxy, nxy)` after dropping the FFTW R2C padding. |
| `uv_rotate(PA, dRA, dDec, u, v) → (dRArot, dDecrot, urot, vrot)` | C/python wrapper for the rotation. |
| `interpolate(r2cFT, duv, u, v, origin='upper') → vis` | Bilinear interp of an external `(nxy, nxy/2+1)` complex array. |
| `apply_phase_vis(dRA, dDec, u, v, vis) → vis_out` | Apply the translation phase. Returns a new array. |
| `reduce_chi2(vis_obs_re, vis_obs_im, vis_obs_w, vis) → chi2` | Pure reduction kernel. |
| `_fft2d(image)` | R2C 2-D FFT, returns an `ArrayWrapper`-backed ndarray. |
| `_fftshift(matrix)` | Quadrant swap. |
| `_fftshift_axis0(matrix)` | Half-plane axis-0 swap. |

### 8.6 The `ArrayWrapper` class (`python/libcommon.pyx` lines 50-94)

Wraps a raw C buffer (returned by `_copy_input` / `_fft2d`) into a NumPy array without copying. The C buffer lives in FFTW-aligned memory; when the NumPy array is garbage-collected, `__dealloc__` calls `cpp.galario_free(self.data_ptr)` to give the memory back to FFTW. Pattern adapted from <https://gist.github.com/GaelVaroquaux/1249305>.

---

## 9. The pure-Python reference (`python/utils.py`)

`utils.py` is **not installed** with the C extension — it is copied into the build directory by `python/CMakeLists.txt` lines 67 (`configure_file(utils.py "${CMAKE_CURRENT_BINARY_DIR}" COPYONLY)`) so that `test_galario.py` can import it. It contains pure-NumPy/SciPy versions of the algorithms, used as ground truth in tests:

| Function | Mirror of |
|----|----|
| `py_sampleImage(reference_image, dxy, udat, vdat, dRA=0, dDec=0, PA=0, origin='upper')` | `sampleImage` — uses `np.fft.fft2`, `RectBivariateSpline` (kx=ky=1), and an explicit Hermitian-symmetry sign flip. |
| `py_sampleProfile(intensity, Rmin, dR, nxy, dxy, ...)` | `sampleProfile` — calls `py_sampleImage` after `interp1d`-based image creation. |
| `py_chi2Image`, `py_chi2Profile` | corresponding chi² wrappers |
| `radial_profile(Rmin, delta_R, nrad, mode='Gauss'|'Cos-Gauss', ...)` | Test-only profile generator (Jy/sr) |
| `central_pixel(I, Rmin, dR, dxy)` | Trapezoidal integral matching `src/galario.cpp` exactly |
| `g_sweep_prototype(I, Rmin, dR, nrow, ncol, dxy, inc, ...)` | Pure-Python `sweep` |
| `sweep_ref(I, Rmin, dR, ..., Dx, Dy, ..., origin='upper')` | Reference sweep with arbitrary `(Dx, Dy)` and `origin` |
| `create_reference_image`, `create_sampling_points`, `generate_random_vis` | Test fixture builders (Gaussian image, uniform-disc `(u, v)` samples, random visibilities) |
| `uv_idx`, `uv_idx_r2c` | Map `(u, v)` to pixel indices for C2C and R2C |
| `int_bilin_MT(f, x, y)` | Bilinear interpolation written explicitly (used to cross-check the C kernel) |
| `matrix_size(udat, vdat, **kwargs)` | Suggests `Nuv`, `minuv`, `maxuv` from `(u, v)` — older equivalent of `get_image_size` |
| `apply_phase_array(u, v, vis_int, x0, y0)` | NumPy version of `apply_phase_vis` |
| `apply_rotation(PA, dRA, dDec, udat, vdat)` | NumPy version of `uv_rotate` |
| `unique_part(array)` | Take `[:, 0:N/2+1]` for comparing C2C vs R2C |
| `assert_allclose(x, y, rtol, atol)` | Custom variant that prints the offending elements |

---

## 10. Test suite (`python/test_galario.py`)

657 LoC, parametrised pytest. The suite is what `ctest` (and CI) actually runs. By default it uses CPU only; setting `GALARIO_TEST_GPU=1` swaps in `galario.double_cuda` / `galario.single_cuda` (lines 35-40). `python/conftest.py` adds a `--gpu` flag.

Test inventory:

| Test | What it checks | Tolerance |
|----|----|----|
| `test_intensity_sweep` | `sweep` against `sweep_ref` and `g_sweep_prototype`, 4 parameter sets | `rtol=1e-12` |
| `test_R2C_vs_C2C` | Old C2C vs current R2C implementation, 4 parameter sets | `rtol=1e-6` |
| `test_interpolate` | `interpolate()` vs `int_bilin_MT` reference | SP `2e-4`, DP `1e-16` |
| `test_FFT` | `_fft2d()` vs `np.fft.fft2` | SP `1e-5`, DP `1e-16` |
| `test_shift_axes01` | `_fftshift()` vs `np.fft.fftshift` | SP `1e-8`, DP `1e-16` |
| `test_shift_axis0` | `_fftshift_axis0()` vs `np.fft.fftshift(axes=0)` | same |
| `test_apply_phase_vis` | `apply_phase_vis()` vs `apply_phase_array` (6 parameter sets, SP+DP × par1/par2/par3) | SP `1e-7`/`1e-3` |
| `test_reduce_chi2` | `reduce_chi2()` vs explicit `np.sum(...)` | SP `1e-6`, DP `1e-15` |
| `test_image_origin` | Cross-check that `origin='upper'` and `origin='lower'` give the same visibilities (constructs an asymmetric image from 4 Gaussian rings with displacements + position angles) | SP `1e-2`, DP `1e-10` |
| `test_all` | The big one: `sampleImage`, `sampleProfile`, `chi2Image`, `chi2Profile` cross-checked against their `py_*` counterparts (4 parameter sets) | DP `1e-6` |
| `test_loss` | Step-by-step precision audit (shift / FFT / shift / phase / interpolate) | per-step |
| `test_exception` | C++ exceptions become correct Python exceptions | exact match on regex |
| `test_get_coords_meshgrid` | `get_coords_meshgrid` vs hand-rolled meshgrid | SP `1e-6`, DP `1e-15` |

The four standard parameter sets (`par1` … `par4`, lines 43-46):

```python
par1 = {'dRA': 0.,   'dDec': 0.4, 'PA':   2.,  'nxy': 1024}
par2 = {'dRA': -3.5, 'dDec': 7.2, 'PA': -23.,  'nxy': 2048}
par3 = {'dRA': 2.3,  'dDec': 3.2, 'PA':  88.,  'nxy': 4096}
par4 = {'dRA': 0.,   'dDec': 0.,  'PA': 145.,  'nxy': 1024}
```

The integration test `test_all` uses `nsamples = 1000` random `(u, v)` points generated with `np.random.seed(42)` (seeded inside `create_sampling_points`) so results are reproducible.

---

## 11. Performance benchmarking (`python/speed_benchmark.py`, `python/speed_baseline.sh`)

`speed_benchmark.py` is an `argparse + timeit` driver. Flags:

| Flag | Default | Purpose |
|----|----|----|
| `--gpu` / `--cpu` | gpu off, cpu on | which backend to time |
| `--gpu_id` | 0 | `use_gpu()` selection |
| `--cycles` | 5 | repetitions of `timeit.Timer.repeat` |
| `--size` | 4096 | square image side |
| `--nsamples` | 1e6 | number of `(u, v)` points |
| `--tpb` | `[16]` | threads-per-block sweep on GPU |
| `--ompnthreads` | `[1]` | OpenMP threads sweep on CPU |
| `--dtype` | `float64` | image/uv dtype |
| `--image` | False | choose `chi2Image` instead of `chi2Profile` |
| `--use-py` | False | use `py_chi2*` Python reference instead of galario |

Output: a tab-separated file with columns `size, nsamples, real, OMP, tpb, Ttot, Tavg, Tstd, Tmin`. The first call's timing is dropped (warmup overhead).

`speed_baseline.sh` runs the suite over `sizes="512 1024 2048 4096 8192 16384"`, `openmp_threads="1 2 4 6 8 10 12"`, `threads_per_block="8 16 32"`, and `cycles=20` for both `--image` and `--profile` modes; output names are `profile_baseline_<git-hash>_<host>.txt` and `image_baseline_...`.

The Tazzari et al. (2018) paper reports ~150× speedup of the GPU version over the multi-threaded CPU version for typical sizes (4096²); the benchmarks here are how that number is measured.

---

## 12. C++ usage example

`src/galario_test.cpp` is the canonical hand-written example, also used as a `cpp_compile_test` in CMake (`src/CMakeLists.txt` lines 138-144). The full source is 73 lines; the heart of it is:

```cpp
#include "galario.h"
using namespace galario;

int main() {
    init();
    constexpr int nx = 128, ny = nx;
    std::vector<dreal> realdata(nx*ny);     // zero-initialised
    dcomplex* res = copy_input(nx, ny, &realdata[0]);

    int n = 4;
    dreal* rp = &realdata[0];
    dcomplex* cp = res;
    dreal r = realdata[0];
    dreal dxy = 0.2;

    sweep(nx, rp, dxy/100., dxy/10.5, nx, dxy, 0.5, cp);
    uv_rotate(r, r, r, rp, rp, n, rp, rp, rp, rp);
    fft2d(nx, ny, res);
    fftshift(n, n, cp);
    fftshift_axis0(n, n, cp);
    apply_phase_sampled(r, r, n, rp, rp, cp);

    auto ncomplex = nx*(ny/2+1);
    std::vector<dcomplex> vis_int(res, res + ncomplex);
    auto chi2 = reduce_chi2(300, &realdata[0], &realdata[0], res, &realdata[0]);

    // verify reduce_chi2 didn't trash the input
    for (auto i = 0; i < ncomplex; ++i)
        assert(vis_int[i] == res[i]);

    galario_free(res);
    cleanup();
    return 0;
}
```

`docs/C++-example.rst` lines 19-23 give the build line:

```bash
g++ -I/path/to/galario/include -L/path/to/galario/lib \
    -lgalario -DDOUBLE_PRECISION galario_test.cpp -o galario_test
```

For single precision, drop `-DDOUBLE_PRECISION` and link `-lgalario_single`. For CUDA, link `-lgalario_cuda` or `-lgalario_single_cuda`.

---

## 13. End-user workflow (MCMC fit)

`docs/quickstart.rst` walks through the canonical use case in seven steps. The reduced version (assuming `u, v, Re, Im, w` come from a uvtable that has been wavelength-normalised):

```python
import numpy as np
from emcee import EnsembleSampler
from galario import deg, arcsec
from galario.double import chi2Profile, get_image_size

def GaussianProfile(f0, sigma, Rmin, dR, nR):
    R = np.linspace(Rmin, Rmin + dR*nR, nR, endpoint=False)
    return f0 * np.exp(-0.5*(R/sigma)**2)

def lnpriorfn(p, par_ranges):
    for i in range(len(p)):
        if p[i] < par_ranges[i][0] or p[i] > par_ranges[i][1]:
            return -np.inf
    return -p[0]    # log-jacobian for f0

def lnpostfn(p, p_ranges, Rmin, dR, nR, nxy, dxy, u, v, Re, Im, w):
    lnprior = lnpriorfn(p, p_ranges)
    if not np.isfinite(lnprior):
        return -np.inf
    f0, sigma, inc, PA, dRA, dDec = p
    f0 = 10.**f0
    sigma *= arcsec; Rmin *= arcsec; dR *= arcsec
    inc *= deg;     PA *= deg
    dRA *= arcsec;  dDec *= arcsec
    f = GaussianProfile(f0, sigma, Rmin, dR, nR)
    chi2 = chi2Profile(f, Rmin, dR, nxy, dxy, u, v, Re, Im, w,
                       inc=inc, PA=PA, dRA=dRA, dDec=dDec)
    return -0.5*chi2 + lnprior

# Image size from the data
nxy, dxy = get_image_size(u, v, verbose=True)

# 6 parameters, 40 walkers, 4 threads
ndim, nwalkers, nthreads = 6, 40, 4
sampler = EnsembleSampler(nwalkers, ndim, lnpostfn,
    args=[p_ranges, 1e-4, 0.01, 2000, nxy, dxy, u, v, Re, Im, w],
    threads=nthreads)

sampler.run_mcmc(pos, nsteps=3000)
```

To switch to GPU it suffices to replace `from galario.double import chi2Profile` with `from galario.double_cuda import chi2Profile`. Nothing else changes. (`docs/cookbook.rst` lines 36-47 makes this portable-import idiom explicit.)

The matching `uvplot.UVTable` and `corner.corner` calls are in the docs (`docs/quickstart.rst` lines 250-269).

---

## 14. Input and output formats

galario does not own a file format. It consumes plain NumPy arrays. The canonical demo dataset, `docs/uvtable.txt`, is a 5-column ASCII table:

```
u  [m]    v  [m]    Re  [Jy]    Im  [Jy]    w
-155.90093  234.34887   0.01810   0.13799   200.05723
9.290660    362.97853  -0.05827   0.02820   216.95405
...
```

That is then loaded with:

```python
u, v, Re, Im, w = np.require(np.loadtxt("uvtable.txt", unpack=True),
                              requirements='C')
wle = 1e-3
u /= wle; v /= wle
```

Note the `np.require(..., requirements='C')` — galario's Cython memoryview decorators require C-contiguous arrays, and `np.loadtxt(..., unpack=True)` returns transposed columns that are not contiguous (`docs/FAQ.rst` Question 1.1).

There is no I/O for FITS, MS, UVFITS, HDF5, or any other format inside galario. Pre-processing is the user's job: it is expected to come from CASA via `split` for averaging into a single channel.

---

## 15. Numerical and design internals

### 15.1 Why amplitude/phase rather than real/imag interpolation?

`interpolate_core` (`src/galario.cpp` lines 749-815) interpolates the **amplitude** (using the four corner magnitudes) and the **phase** (using the four corner full complex values) separately, then reconstructs `vis = amp·(cos φ + i sin φ)`. That is more accurate than independently interpolating real and imaginary parts whenever the function rotates rapidly in the complex plane (i.e. for sources displaced from the phase centre). The 1.2 release explicitly mentions this fix (`CHANGELOG.rst` line 19).

### 15.2 Why R2C, not C2C?

For real-valued images, the 2-D R2C transform produces a half-plane spectrum of size `(N, N/2+1)`. galario stores only this half-plane (saving ~50% memory) and uses Hermitian symmetry inside `interpolate_core` to handle negative-u points. This is the difference checked in `test_R2C_vs_C2C`.

### 15.3 No FFT plan caching

Both `fft_h` (CPU, line 514) and `fft_d` (GPU, line 495) **build a new plan on every call** and destroy it at the end. `// TODO: find a way to store the plan` (line 523) and `// TODO: find a way to store the plan (maybe homogeneously with the cuFFTPlan` (line 499) flag this as known tech debt. For repeated calls at the same `(nx, ny)` (which is the MCMC hot path) this is overhead.

### 15.4 `--default-stream per-thread` (multi-process GPU sharing)

`src/CMakeLists.txt` line 93 sets the NVCC flag `--default-stream per-thread`. Combined with the lazy `cublas_init()` (`galario.cpp` lines 207-218 with mutex-protected double-checked init), this is what allows multiple Python processes to share one GPU. The CHANGELOG entry is `[core] Allow multiple processes to use the GPU concurrently by default.` (1.0.2, line 36).

### 15.5 RAII memory handling on GPU

`CudaMemory<T>` (`galario.cpp` lines 230-273) is a small RAII wrapper around `cudaMalloc`/`cudaFree` that:

* Throws `std::bad_alloc` on `cudaErrorMemoryAllocation`.
* Frees on destruction (no `CCheck` in the destructor, deliberately, because a destructor must not throw).
* Forbids copy, allows move.
* Has a `(size_t n, const T* source)` ctor that copies host → device on construction, and a `Retrieve(T*)` method that copies device → host.

This is what the 1.1 CHANGELOG entry `[core] Memory handling on GPU: memory is now automatically freed in case of an error (allows catching errors with Exceptions).` refers to.

### 15.6 Image origin convention

`docs/tech-specs.rst` is dedicated to this. The matrix index `(i, j)` can have its `[0, 0]` either upper-left (`origin='upper'`, default, `v_origin = +1`) or lower-left (`origin='lower'`, `v_origin = -1`). The R.A. axis always *decreases* with increasing `j` (East is left); the Dec axis decreases with `i` for `'upper'` and increases for `'lower'`. The central pixel is always `[Nxy/2, Nxy/2]`. galario's `set_v_origin('upper')` returns `+1` and `set_v_origin('lower')` returns `-1`; that constant is propagated into `interpolate_core` (line 760) and into `get_coords_meshgrid` (line 788), which is how the convention takes effect numerically. The 1.2 release is the one that introduced the option (CHANGELOG line 19).

### 15.7 Coordinate normalisation conventions

Throughout, `(u, v)` are in **wavelengths** (not metres). `(dRA, dDec)` are in **radians**. The phase factor in `apply_phase_sampled` is `exp(2πi (u·dRA + v·dDec))`, so when both are in their natural units, the formula gives a translation of `(dRA, dDec)` radians on the sky. Helpful constants `arcsec` and `deg` (lines 43-44 of `libcommon.pyx`) let users write `sigma * arcsec`, etc.

---

## 16. Build-time matrix at a glance

| Setting | CPU build | GPU build |
|----|----|----|
| Compiler | `g++ ≥ 4.8.1` or `clang++ ≥ 3.3` (with C++11) | NVIDIA `nvcc` (CUDA Toolkit ≥ 8.0) |
| Threads | OpenMP (`-fopenmp`) + pthreads | per-block CUDA threads (`tpb²`) |
| FFT | FFTW3 (single+double, threads variant) | cuFFT |
| BLAS | not used directly (manual reduction) | cuBLAS for `Dznrm2`/`Scnrm2` |
| Output libraries | `libgalario.so`, `libgalario_single.so` | `libgalario_cuda.so`, `libgalario_single_cuda.so` |
| Python modules | `galario.double`, `galario.single` | `galario.double_cuda`, `galario.single_cuda` |
| `HAVE_CUDA` | `False` | `True` |
| Default install prefix | `$CONDA_PREFIX` if active, else system | same |
| MPI | not supported | not supported |
| Distributed | not supported | not supported |

---

## 17. Limitations, deliberate omissions, and TODOs

Plain reading of the source and docs gives:

* **No w-projection / wide-field correction.** All small-field. (`docs/tech-specs.rst`)
* **No primary-beam correction in `*Profile()`.** User must apply it externally for the image case if needed. (same)
* **Single frequency only.** No spectral cubes; users channel-average their MS first. (same)
* **No polarisation, no Jones matrices, no time/baseline-dependent effects.**
* **No FFT plan caching.** `// TODO: find a way to store the plan` in both CPU (`galario.cpp` line 523) and GPU (line 499).
* **No async memory copy on GPU.** `// TODO copy memory asynchronously or create streams to define dependencies` appears at lines 1268 and 1567.
* **No conda GPU build.** The conda-forge package is CPU-only because of CUDA ABI constraints (`docs/install.rst` lines 27-29).
* **No Python packaging.** No `setup.py`, no `pyproject.toml` — installation is CMake-only.
* **Square images required.** `CHECK_INPUTXY` enforces `nx == ny`, even, ≥ 2.
* **Central-pixel ratio constraint.** `(dxy/2 - Rmin)/dR` must be > 5 for `*Profile()`.
* **No automatic dtype promotion.** Passing a `float32` array to `galario.double.sampleImage` raises a Cython type error (`dreal[::1]` is `double[::1]` in that flavour).
* **Windows is unsupported** (`docs/install.rst` line 8).
* **macOS GPU build off by default** because of NVCC/host-compiler version conflicts (`CMakeLists.txt` lines 49-55, `docs/install.rst` lines 322-325).
* **Drop-of-Python-2 / Python 3.6 happened recently.** The latest commit on master is `8949db4 drop support for Python 3.6`; CHANGELOG goes only up to 1.2.2, and the README badge advertises 3.7-3.9.
* **Numerical robustness for huge-offset sources** improved in 1.2 but still relies on bilinear interpolation; users with extreme offsets (`(dRA, dDec) >> dxy`) should verify the result.

---

## 18. Integration into RRIVis (or any host project)

galario lives in this tree as a git submodule (the `.git` file in `simulators/galario/` is a one-line gitlink, not a real git directory). A consumer typically picks one of three paths:

1. **Use the conda-forge build** (`conda install -c conda-forge galario`) for the CPU-only path. This installs four importable Python sub-packages and the four `.so` files. Easiest. Recommended by the README.
2. **Build manually with CMake** to get the GPU path. From `simulators/galario/`:
   ```bash
   mkdir build && cd build
   cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
   make -jN
   make install
   ```
   This works inside an active conda env (CMake auto-detects `CONDA_PREFIX`).
3. **Vendor + symlink**. Keep the source in-tree, build it, and add the resulting `lib/` and `python/<flavour>/libcommon.so` to the project's install steps.

Programmatic integration points relevant to a wider RIME codebase like RRIVis:

* **Inputs:** `(u, v)` in wavelengths, image in Jy/pixel (or radial profile in Jy/sr), `dxy` in radians, `dRA/dDec/PA/inc` in radians.
* **Outputs:** complex visibilities (Jy) sampled at the same `(u, v)` points, *or* a chi-squared scalar.
* **Threading:** call `galario.double.threads(N)` once per process; thread-safe afterwards.
* **GPU device selection:** `galario.double_cuda.use_gpu(id)` once per process. Multi-process GPU sharing is automatic (per-thread default stream).
* **Backend swap:** the four sub-packages have identical Python signatures, so a code path that takes `acc_lib` as a parameter (as `test_galario.py` and `speed_benchmark.py` do) can flip CPU↔GPU and single↔double with no source changes.
* **Per-frequency loop:** since galario is single-frequency, a simulator wrapping it for a frequency cube just loops over channels, calling `sampleImage`/`sampleProfile` per channel. This is the obvious extension noted in `docs/FAQ.rst` (commented-out paragraph about line cubes).
* **No callback interface.** The library does not expose hooks for streaming visibilities; callers must collect the returned arrays themselves.

---

## 19. File-by-file index

### `src/`

| File | Role | LoC |
|----|----|----|
| `galario.h` | Public C++ API (`namespace galario`); 6 main + 8 expert + 5 lifecycle functions. | 59 |
| `galario_py.h` | `void*`-typed mirror used by Cython. | 48 |
| `galario_defs.h` | `dreal`, `dcomplex` typedefs gated by `DOUBLE_PRECISION` and `__CUDACC__`. | 46 |
| `galario.cpp` | Single-file implementation with `#ifdef __CUDACC__` branches: every kernel for both backends. | 1664 |
| `galario_test.cpp` | Smoke test, also wired as `cpp_compile_test` in CMake. | 72 |
| `CMakeLists.txt` | Builds 4 library targets, conditional on CUDA + DOUBLE_PRECISION. | 145 |

### `python/`

| File | Role | LoC |
|----|----|----|
| `libcommon.pyx` | Cython implementation of the user API (`sampleImage`, `chi2Profile`, …, plus `ArrayWrapper`, constants, helpers). | 1082 |
| `galario_defs.pxd` | `cdef extern from "galario_py.h"` declarations of the 17 underlying `_*` functions. | 46 |
| `galario_config.pxi.in` | Templated `.pxi` defining `dreal`, `real_dtype`, `complex_dtype`, `complex_typenum` per flavour. | 41 |
| `__init__.py.in` | Top-level `galario/__init__.py` template (sets `HAVE_CUDA`, imports submodules, re-exports constants). | 32 |
| `__init_module__.py.in` | Per-flavour `__init__.py` template (`from .libcommon import *`, `_init`, `atexit.register(_cleanup)`). | 26 |
| `wrap_lib.cmake` | CMake function to build the four cython targets. | 80 |
| `CMakeLists.txt` | Drives `wrap_lib`, install rules, and pytest registration. | 89 |
| `utils.py` | Pure-Python references for tests (not installed). | 507 |
| `test_galario.py` | The pytest suite (~13 test functions, parametrised). | 657 |
| `conftest.py` | Adds `--gpu` CLI option. | 24 |
| `speed_benchmark.py` | argparse + timeit benchmarking driver. | 200 |
| `speed_baseline.sh` | Bash sweep over sizes, threads, GPU TPB. | 65 |

### `cmake/`

| File | Role |
|----|----|
| `FindCython.cmake` | Locate `cython`/`cython3` next to `PYTHON_EXECUTABLE`. |
| `FindSphinx.cmake` | Locate `sphinx-build`. |
| `UseCython.cmake` | Provides `cython_add_module()` (from <https://github.com/thewtex/cython-cmake-example>). |
| `LookUp-GreatCMakeCookOff.cmake` | Clones <https://github.com/UCL/GreatCMakeCookOff.git> into the build dir; provides the `Find*` modules for FFTW3 and Numpy. |
| `cmake_uninstall.cmake.in` | Template driving `make uninstall` from `install_manifest.txt`; also removes the python package directory. |

### `docs/`

| File | Content |
|----|----|
| `index.rst` | Landing page, citation, toctree. |
| `install.rst` | Build & install, FFTW + CUDA + Python instructions. |
| `basic_usage.rst` | 4-paragraph quick reference. |
| `quickstart.rst` | Full `emcee` MCMC fit walkthrough (~270 lines). |
| `tech-specs.rst` | Image origin convention, coordinate axes diagram. |
| `cookbook.rst` | GPU vs CPU import, threads, meshgrid recipe. |
| `py-api.rst` | Auto-generated Python API page (uses `autofunction`). |
| `C++-api.rst` | Hand-written C++ API page. |
| `C++-example.rst` | Walks `galario_test.cpp` line-by-line. |
| `publications.rst` | ADS link to citing papers. |
| `FAQ.rst` | One question: C-contiguous arrays. |
| `license.rst` | Renders `LICENSE`. |
| `uvtable.txt` | ASCII demo dataset. |
| `images/` | JPGs/PNGs used by the docs. |
| `CMakeLists.txt`, `conf.py.in` | Sphinx wiring. |

---

## 20. Quick-look summary

* **What it is:** a tightly-scoped, 1.6 kLoC C++/CUDA library + Cython bindings for one calculation only — synthetic visibilities and chi² for small-field, single-channel (sub-)mm interferometer fits, intended as the inner loop of MCMC samplers like `emcee`.
* **Key design wins:** one C++ file producing four binaries (CPU/GPU × float/double); R2C FFT with Hermitian-symmetry-aware interpolation; cuBLAS-based chi² reduction; RAII GPU memory; per-thread default cuda streams for multi-process GPU sharing; auto conda-prefix install.
* **Key limitations:** no plan caching; no async device transfers; no wide-field; no polarisation; no spectral cube; no PyPI; no Windows; macOS GPU off by default.
* **Status:** version 1.2.2 (2020-02-28), latest commit "drop support for Python 3.6". Active enough to maintain a CI matrix on Python 3.7-3.9 but fundamentally feature-stable since 1.2.
