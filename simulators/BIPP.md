# BIPP — Bluebild Imaging++

Exhaustive reference for the BIPP source tree at `simulators/BIPP/`, version
**1.0.0** (file `VERSION`). BIPP is the high-performance C++17/CUDA/ROCm/Python
implementation of the **Bluebild** algorithm for radio-interferometric image
synthesis, developed by EPFL's Radio Astronomy / IMOS group
(`https://github.com/epfl-radio-astro/bipp`). License: **GPL v3**
(`LICENSE`, 35 KB). Documentation: `https://bipp.readthedocs.io/`.

This document describes only what is in this checkout — every API, every
public type, every CLI flag, every dependency, every file format constant,
every algorithmic choice, traced back to the corresponding source file and
line. It is not an interpretation of the published Bluebild papers; it is the
implementation manual.

## 1. What Bluebild is and what BIPP does

The Bluebild algorithm replaces the conventional `V → griddedV → FFT → image`
imaging pipeline with a *functional principal-component decomposition* of the
sky brightness operator:

1. The instrument-level coherency matrix `S` (visibilities arranged as a
   `(N_beam, N_beam)` Hermitian matrix) and the *Gram matrix* `G = W^H Φ W`
   (where `Φ_{ij} = sinc(2 |xyz_i − xyz_j| / λ)` is the antenna-domain plane-
   wave inner product and `W` are beamforming weights) are formed for every
   integration sample.
2. The generalized Hermitian eigenproblem `S v = d G v` (or the standard one
   `S v = d v` if no Gram matrix is supplied) is solved with LAPACK `?hegv`
   / `?heev`. Each `(d_k, v_k)` pair is one *energy eigenpair*.
3. Eigenvalues are clustered (typically by k-means in linear or log space)
   into a small number of *energy levels*, and each cluster becomes one
   "image" (e.g. *strong sources*, *mild sources*, *faint sources*). Each
   level is independently filtered through one of `lsq, std, sqrt, inv,
   inv_sqrt`.
4. For every selected level, BIPP forms *virtual visibilities*
   `V̂ = (V D V^H)` per integration, where `D = diag(d̃)` are the filtered
   eigenvalues.
5. Virtual visibilities are projected to the sphere by a 3-D
   non-uniform-FFT (NUFFT type-3) implemented in the external
   **neonufft** library (CPU and CUDA/ROCm), evaluated at the LMN
   coordinates of the imaging pixel grid. The real part of the resulting
   complex pixel values is summed across integrations to form the
   level image.
6. Optional normalisation by the number of integration samples and
   write-out to HDF5.

Compared to direct DFT or W-stacking imaging, the eigenvalue clustering
produces a low-rank, regularised representation that lets the user separate
flux scales (bright sources / sidelobes / faint diffuse), and the NUFFT keeps
the per-step cost manageable even for 3-D `lmn` grids on wide-field arrays
(SKA-Low, LOFAR HBA, MWA). The HPC value-add over the original Python
Bluebild prototype is: a single C++ core; OpenMP, CUDA and ROCm backends
plus optional MPI; neonufft as the NUFFT engine; HDF5-backed dataset/image
files so the four pipeline stages (`dataset`, `selection`, `image_prop`,
`synthesis`) can be invoked independently, possibly on different machines.

## 2. Repository layout

```
simulators/BIPP/
├── VERSION                       # "1.0.0"
├── README.md                     # build / pip / CLI quick-start
├── LICENSE                       # GPLv3
├── pyproject.toml                # scikit-build-core wheel build, pybind11 ≥2.13
├── CMakeLists.txt                # 325 lines — top-level build config
├── .clang-format                 # formatting (Google-derived)
├── .readthedocs.yml              # docs build
├── cmake/                        # bipp{Shared,Static}{Config,Targets}.cmake
├── docs/
│   ├── Doxyfile, Makefile, requirements.txt
│   └── source/                   # Sphinx + Breathe (doxygenfile directives)
│       ├── index.rst, context.rst, exceptions.rst,
│       ├── synthesis.rst, eigendecomposition.rst, templates/
├── examples/
│   └── simulation/
│       ├── lofar_bootes_nufft.py   # full Bootes field demo (NUFFT)
│       └── test_writer.py          # data-generation skeleton
├── include/bipp/                 # C++ public headers
│   ├── bipp.hpp, config.h.in, enums.h, errors.h
│   ├── communicator.hpp, context.hpp,
│   ├── dataset.hpp, dataset_file.hpp,
│   ├── exceptions.hpp,
│   ├── image_data.hpp, image_data_file.hpp,
│   ├── image_prop.hpp, image_prop_file.hpp,
│   ├── image_synthesis.hpp
├── python/
│   ├── CMakeLists.txt
│   ├── pybipp.cpp                # pybind11 bindings (~494 lines)
│   └── bipp/                     # pure-Python helpers + C-extension
│       ├── __init__.py           # `from .pybipp import *`
│       ├── __main__.py           # delegates to apps.cli:run_cli
│       ├── core.py               # `Block` ABC
│       ├── array.py              # `LabeledMatrix`
│       ├── numpy_compat.py
│       ├── filter.py             # eigenvalue filters
│       ├── frame.py              # UVW frame, lmn grid, uvw reshape
│       ├── gram.py               # GramMatrix, GramBlock
│       ├── source.py             # SkyEmission, TGSS catalog reader
│       ├── instrument.py         # InstrumentGeometry, LofarBlock, MwaBlock
│       ├── beamforming.py        # BeamWeights, MatchedBeamformerBlock
│       ├── statistics.py         # VisibilityMatrix, VisibilityGeneratorBlock
│       ├── parameter_estimator.py# k-means level inference
│       ├── selection.py          # JSON export of level selections
│       ├── measurement_set.py    # casacore-based MS readers (LOFAR/MWA/SKA)
│       ├── apps/                 # CLI subcommand implementations
│       │   ├── cli.py            # argparse dispatcher
│       │   ├── create_dataset.py
│       │   ├── create_image_prop.py
│       │   ├── create_selection.py
│       │   ├── image_synthesis.py
│       │   └── plot_images.py
│       ├── data/instrument/      # LOFAR.csv, MWA.csv (ITRF station coordinates)
│       └── imot_tools/           # vendored math utilities (S. Kashani)
│           ├── data/, io/, math/, phased_array/, util/, LICENSE.txt
├── src/                          # C++ implementation (LANG="CXX17")
│   ├── CMakeLists.txt
│   ├── apps/bipp_synthesis.cpp   # standalone synthesis CLI (CLI11 + json)
│   ├── communicator.cpp, context.cpp, context_internal.{cpp,hpp}
│   ├── logger.{cpp,hpp}, rt_graph.{cpp,hpp}
│   ├── eigensolver.cpp           # eigh / eigh_gram (LAPACK backed)
│   ├── nufft_synthesis.{cpp,hpp},
│   ├── nufft_interface.hpp, nufft_util.hpp
│   ├── image_synthesis.cpp,
│   ├── image_data_file.cpp, image_prop_file.cpp,
│   ├── dataset_file.cpp,
│   ├── host/                     # CPU kernels
│   │   ├── blas_api.hpp, lapack_api.hpp,
│   │   ├── domain_partition.hpp,
│   │   ├── gram_matrix.{cpp,hpp},
│   │   ├── nufft.hpp,
│   │   ├── virtual_vis.{cpp,hpp},
│   │   └── kernels/nuft_sum.{cpp,hpp}
│   ├── gpu/                      # CUDA / HIP kernels (mostly stubs / WIP)
│   │   ├── nufft.hpp, util/, kernels/{add_vector,nuft_sum}.{cu,hpp}
│   ├── io/                       # HDF5 helpers, dataset_spec versioning
│   │   ├── dataset_spec.hpp,    h5_util.hpp
│   ├── memory/                   # allocator/array/view abstractions
│   │   ├── allocator.hpp, allocator_factory.{cpp,hpp},
│   │   ├── array.hpp, view.hpp, copy.hpp,
│   │   ├── pool_allocator.hpp, umpire_allocator.hpp
│   └── mpi_util/                 # mpi_check_status / mpi_data_type / init guard
├── tests/                        # GoogleTest C++ tests
│   ├── CMakeLists.txt, run_tests.cpp,
│   ├── test_domain_partition.cpp, test_nufft_synthesis_lofar.cpp
│   └── data/                     # JSON fixtures (lofar input + expected outputs)
└── scripts/python_install_path.py
```

## 3. Build system

`pyproject.toml` (top of file):

```toml
[build-system]
requires = ["wheel", "scikit-build-core", "cmake>=3.15", "pybind11>=2.13", "make"]
build-backend = "scikit_build_core.build"

[project]
name = "bipp"
dynamic = ["version"]    # version pulled from VERSION via regex

[project.scripts]
bipp = "bipp.apps.cli:run_cli"
```

The default scikit-build configuration sets:

- `cmake.build-type = "RELEASE"`
- `wheel.packages = ["python/bipp"]`
- `BIPP_INSTALL_PYTHON_MODE = "skbuild"`, `BIPP_INSTALL_PYTHON = ON`,
  `BIPP_INSTALL_APPS = OFF`, `BIPP_INSTALL_LIB = OFF` (i.e. wheels ship the
  Python module + extension only, not the development C++ headers/libs)

Override env vars (`tool.scikit-build.overrides` blocks):

- `BIPP_GPU=CUDA` → adds `-DBIPP_GPU=CUDA` to CMake.
- `BIPP_GPU=ROCM` → adds `-DBIPP_GPU=ROCM`.
- `BIPP_CIBW_WHEEL_BUILD=1` → forces `BLAS_LIBRARIES=/usr/lib64/libopenblaso.so.0`
  for cibuildwheel manylinux jobs.

Top-level `CMakeLists.txt` (325 lines) options (canonical defaults):

| Option                          | Values                  | Default | Effect |
|---------------------------------|-------------------------|---------|--------|
| `BUILD_SHARED_LIBS`             | ON / OFF                | ON      | shared vs static `libbipp` |
| `BIPP_PYTHON`                   | ON / OFF                | ON      | build pybind11 module `pybipp` |
| `BIPP_GPU`                      | OFF / CUDA / ROCM       | OFF     | enable CUDA or HIP backend |
| `BIPP_BUILD_TESTS`              | ON / OFF                | OFF     | gtest-based C++ tests |
| `BIPP_BUILD_APPS`               | ON / OFF                | ON      | build `bipp_synthesis` C++ exe |
| `BIPP_UMPIRE`                   | ON / OFF                | OFF     | use Umpire memory pools |
| `BIPP_MPI`                      | ON / OFF                | OFF     | enable MPI distributed synthesis |
| `BIPP_INSTALL`                  | LIB / PYTHON / OFF      | LIB     | top-level install target |
| `BIPP_INSTALL_LIB`              | ON / OFF                | ON      | install C++ library |
| `BIPP_INSTALL_APPS`             | ON / OFF                | depends | install `bipp_synthesis` |
| `BIPP_INSTALL_PYTHON`           | ON / OFF                | OFF     | install python module |
| `BIPP_INSTALL_PYTHON_DEPS`      | ON / OFF                | OFF     | bundle Python deps |
| `BIPP_INSTALL_PYTHON_MODE`      | platlib / skbuild       | platlib | install layout |
| `BIPP_BUNDLED_LIBS`             | ON / OFF                | ON      | parent toggle for the four below |
| `BIPP_BUNDLED_SPDLOG`           | ON / OFF                | ON      | FetchContent spdlog v1.14.1 |
| `BIPP_BUNDLED_PYBIND11`         | ON / OFF                | ON      | use bundled pybind11 |
| `BIPP_BUNDLED_GOOGLETEST`       | ON / OFF                | ON      | FetchContent googletest 1.13.0 |
| `BIPP_BUNDLED_JSON`             | ON / OFF                | ON      | FetchContent nlohmann/json 3.12.0 |
| `BIPP_BUNDLED_CLI11`            | ON / OFF                | ON      | FetchContent CLI11 v2.4.2 |
| `BIPP_INSTALL_LIB_SUFFIX`       | string                  | lib/lib64 | install location suffix |
| `BIPP_BUILD_TYPE`               | Debug/.../OFF           | OFF     | overrides `CMAKE_BUILD_TYPE` |

Mandatory external dependencies (resolved with `find_package`):

- **C++17 compiler** with C++17 mode (`CMAKE_CXX_STANDARD 17`).
- **CMake ≥ 3.20** (`cmake_minimum_required(VERSION 3.20 FATAL_ERROR)`).
- **BLAS** and **LAPACK** with the 32-bit interface (`BLA_SIZEOF_INTEGER 4`).
  Detected via `find_package(BLAS REQUIRED)` / `find_package(LAPACK REQUIRED)`.
  Two C-API symbols are probed at config time: `cblas_zgemm` (sets
  `BIPP_BLAS_C`) and `LAPACKE_chegv` (sets `BIPP_LAPACK_C`).
- **HDF5** with C bindings: `find_package(HDF5 MODULE REQUIRED COMPONENTS C)`.
- **neonufft** (EPFL, `https://github.com/epfl-radio-astro/neonufft`) — the
  NUFFT engine. Required as `find_package(neonufft CONFIG REQUIRED)`. If
  `BIPP_GPU` is on, neonufft must itself have been built with GPU support
  (target `neonufft::neonufft_gpu`).
- **spdlog** (bundled v1.14.1 by default).
- **pybind11 ≥ 2.13** (when `BIPP_PYTHON=ON`), bundled by default.
- **CLI11 v2.4.2** (when `BIPP_BUILD_APPS=ON`), bundled by default.
- **nlohmann/json v3.12.0** (when `BIPP_BUILD_APPS` or `BIPP_BUILD_TESTS`).
- **googletest 1.13.0** (when `BIPP_BUILD_TESTS`).
- **Umpire** (optional, only when `BIPP_UMPIRE=ON`).
- **MPI CXX** (optional, only when `BIPP_MPI=ON`).
- **CUDA Toolkit 11+** (when `BIPP_GPU=CUDA`) — links `CUDA::cudart`,
  `CUDA::cublas`, `CUDA::cusolver` (cusolver detection block left
  commented out as a fallback path).
- **HIP / rocBLAS / hipCUB** (when `BIPP_GPU=ROCM`).

Configured artifacts:

- `${PROJECT_BINARY_DIR}/bipp/config.h` — generated from
  `include/bipp/config.h.in` and exports `BIPP_VERSION`, `BIPP_CUDA`,
  `BIPP_ROCM`, `BIPP_OMP`, `BIPP_MPI`, `BIPP_UMPIRE` macros plus
  `BIPP_EXPORT` visibility decorations.
- C++ `bipp` library (object library `bipp_objects` is linked into both
  shared and static targets).
- `bipp_synthesis` C++ executable (`src/apps/bipp_synthesis.cpp`).
- `pybipp` Python extension (`python/pybipp.cpp` → `bipp/pybipp.so`).
- `bipp` CLI entry point (`bipp.apps.cli:run_cli`).

`src/CMakeLists.txt` lists the **active** C++ object files:

```
context_internal.cpp, communicator.cpp, context.cpp,
logger.cpp, rt_graph.cpp,
dataset_file.cpp, eigensolver.cpp,
nufft_synthesis.cpp,
memory/allocator_factory.cpp,
host/gram_matrix.cpp,
host/virtual_vis.cpp,
host/kernels/nuft_sum.cpp,
image_synthesis.cpp, image_data_file.cpp, image_prop_file.cpp
```

Note: a large block of GPU implementations is present in the tree
(`gpu/eigensolver.cpp`, `gpu/gram_matrix.cpp`, `gpu/nufft_3d3.cpp`, …) but
**commented out** in `src/CMakeLists.txt`. Only `gpu/kernels/add_vector.cu`
and `gpu/kernels/nuft_sum.cu` are conditionally added when
`BIPP_CUDA OR BIPP_ROCM`. As of v1.0.0 the GPU host-side wiring is
under refactor; the CPU host path is the canonical, fully wired backend.
The `BippProcessingUnit` enum and pybind dispatcher still expose
`AUTO/CPU/GPU` values — choosing GPU at runtime when the GPU code path is
not compiled raises `bipp::GPUSupportError` from
`nufft_synthesis.cpp:122`.

### 3.1 Pip install paths

For x86 systems EPFL ships pre-built wheels:

- CPU only: `python -m pip install bipp`
- CUDA 12: `python -m pip install bipp-cuda12x` (CUDA libs not bundled, set
  `LD_LIBRARY_PATH` to the system CUDA install).

For source builds with custom external deps (e.g. neonufft installed
elsewhere):

```bash
BIPP_GPU=CUDA \
CMAKE_PREFIX_PATH="${path_to_neonufft};${CMAKE_PREFIX_PATH}" \
python3 -m pip install .
```

C++-only minimal build:

```bash
mkdir build && cd build
cmake .. -DBIPP_PYTHON=OFF -DCMAKE_INSTALL_PREFIX=/usr/local -DBIPP_INSTALL=LIB
make -j8 install
```

C++ + Python with custom prefix:

```bash
mkdir build && cd build
cmake .. -DBIPP_PYTHON=ON -DBIPP_INSTALL=PYTHON \
         -DCMAKE_INSTALL_PREFIX=${path_to_install_to} -DBIPP_PYBIND11_DOWNLOAD=ON
make -j8 install
export PYTHONPATH=${path_to_install_to}:$PYTHONPATH
```

## 4. The Bluebild pipeline (5 stages)

The `bipp` CLI (entry point `bipp.apps.cli.run_cli`, dispatched from
`__main__.py`) implements the user-facing pipeline as five subcommands:

```
bipp dataset    -t {SKAlow,LOFAR,MWA} -ms file.MS -o out.h5 [-r start end step] [-c chan_start chan_end] [-d DATA] [-a N_ant] [-s N_station]
bipp selection  -d dataset.h5 -s "filter,n_levels,sigma,cluster,min,max" [-s ...] [-r start end step] -o selection.json
bipp image_prop -d dataset.h5 -f FoV_deg -w width -o image_prop.h5
bipp synthesis  -d dataset.h5 -s selection.json -i image_prop.h5 -o images.h5 [-p {auto,cpu,gpu}] [-f {single,double}] [-t tol] [--uvw_part nx ny nz]
bipp plot       -i images.h5
```

Each stage has its own argparse subparser in `python/bipp/apps/cli.py:8-185`
and dispatches to a function defined in the matching `apps/<stage>.py`. The
five stages can be rerun independently because each persists its
intermediate state to HDF5 / JSON.

### 4.1 `bipp dataset` — eigen-decompose visibilities

`apps/create_dataset.py:create_dataset(args)` (110 lines):

1. Selects the appropriate `MeasurementSet` reader by telescope name:
   `skalow → SKALowMeasurementSet`, `lofar → LofarMeasurementSet(N_station,
   station_only=True)`, `mwa → MwaMeasurementSet`, `redundant →
   GenericMeasurementSet`. Defaults: SKA-low 512 antennas, LOFAR 37 stations,
   MWA 128 antennas. Anything else raises `NotImplementedError`.
2. Determines time and channel ranges from `--range` (default 0…end step 1)
   and `--channel`.
3. Opens `bipp.DatasetFile.create(out, telescope_name, N_antenna, N_station,
   ra_deg, dec_deg)` which creates a fresh HDF5 file (Dataset format
   v1.0 — see §6).
4. Iterates `ms.visibilities(channel_id, time_id, column=DATA)` (a
   generator yielding `(t, f, S)` triples). For each:
   - `wl = c / f.to_value(Hz)`,
   - `XYZ = ms.instrument(t)` (ICRS antenna positions at `t`),
   - `W = ms.beamformer(XYZ, wl)` (matched beamforming weights — for
     LOFAR/MWA/SKA-low this is identity-like single-beam-per-station),
   - `UVW = ms.instrument.baselines(t, uvw=True, field_center=…)`,
   - `uvw = frame.reshape_uvw(UVW)` (ravel to `(N_ant², 3)`, F-order),
   - call `bipp.eigh_gram(wl, S.data, W.data, XYZ.data)` to solve the
     generalised eigenproblem (LOFAR uses `S.conj()` to compensate the
     known LOFAR conjugation bug, see `wsclean.readthedocs.io/.../chgcentre.html#a-lofar-bug`).
   - write the sample to the dataset:
     `dataset.write(time, wl, scale, v, d, uvw)`.

The dataset HDF5 file therefore stores, per integration sample:
`time`, `wl`, `scale` (1/N_vis), eigenvalues `d` of length `nBeam`,
eigenvectors `v` of shape `(nAntenna, nBeam)` (real interleaved as
`2*nAntenna`), and `uvw` of shape `(N_ant^2, 3)`. Field centre RA/Dec
are file-level attributes.

### 4.2 `bipp selection` — cluster eigenvalues into levels

`apps/create_selection.py:create_selection(args)` (105 lines). One or
more `-s "filter,n_levels,sigma,cluster_func,min,max"` arguments produce a
list of clusterings of all eigenvalues across the chosen sample range. Each
selection 6-tuple is parsed:

| Field         | Type   | Description |
|---------------|--------|-------------|
| `filter`      | str    | one of `lsq, std, sqrt, inv, inv_sqrt` (see §5.4) |
| `n_levels`    | int    | number of k-means clusters |
| `sigma`       | float  | fraction of *smallest* eigenvalues used for clustering (0..1) |
| `cluster_func`| str    | `none` or `log` (cluster in linear or log-eigenvalue space) |
| `d_min`       | float  | lower bound (`-inf`/`inf` accepted as `float()`) |
| `d_max`       | float  | upper bound |

`parameter_estimator.infer_intervals` (`parameter_estimator.py:55-105`)
runs a `sklearn.cluster.KMeans(n_clusters=n_levels, random_state=42)` on
the eligible eigenvalue array (after `[d_min, d_max]` masking, removing
zeros, and trimming the largest `(1-sigma)*N` to ignore outliers). For
`cluster_func == 'log'` it requires `d_min ≥ 0` and clusters
`np.log(d_all)`; for `none` it clusters the raw values. The returned
cluster centroids are converted to half-way intervals around adjacent
sorted centroids by `centroid_to_intervals` (`parameter_estimator.py:18-52`),
producing an `(n_levels, 2)` array of `[lo, hi)` bounds.

For each level a `bipp.filter.Filter(filter_name, lo, hi)` is constructed
(§5.4) and applied to every per-sample eigenvalue array, producing a
`{sample_id: filtered_eigenvalues_list}` dict. The level's tag is
`"s{i}_{filter}_[{lo:.4E},{hi:.4E})"` with `.` replaced by `_` (HDF5 names
must not contain `.`, `/`, or spaces — enforced in `image_synthesis.cpp:38-46`).

The whole selection dict is dumped to JSON via
`selection.export_selection(s_dict, out)` using `NumpyArrayEncoder`
(`selection.py:8-31`) which converts NumPy scalars/arrays.

### 4.3 `bipp image_prop` — define the imaging grid

`apps/create_image_prop.py:33-39`:

```python
with bipp.DatasetFile.open(args.dataset) as dataset:
    field_center = SkyCoord(ra=dataset.ra_deg()*u.deg, dec=dataset.dec_deg()*u.deg, frame="icrs")
lmn_grid, xyz_grid = frame.make_grids(args.width, np.deg2rad(args.fov), field_center)
bipp.ImagePropFile.create(args.output, args.width, args.width, args.fov, lmn_grid.transpose())
```

`frame.make_grids(grid_size, FoV_rad, field_center)` (`frame.py:40-72`) builds:

- `lim = sin(FoV/2)`,
- two slightly offset axes shifted by half a pixel so point sources at
  pixel centres in WSClean/CASA align with single Bluebild pixels (the
  comment in source describes the offset rationale),
- `n = sqrt(1 - l² - m²)` (no `−1`, full sphere geometry, not flat sky),
- transforms `lmn → xyz` with the UVW basis matrix `frame.uvw_basis` (see
  §5.2).

`ImagePropFile` is an HDF5 container holding `width`, `height`, `fov_deg`
and a flat `(width*height, 3)` `lmn` array, plus arbitrary user metadata
via `set_meta(name, value)` (`image_prop.hpp` declares
`MetaType = std::variant<std::size_t, float, std::vector<float>>`).

### 4.4 `bipp synthesis` — NUFFT image synthesis

`apps/image_synthesis.py:13-28` opens dataset + image prop + selection
JSON, builds:

```python
comm = bipp.communicator.world()
ctx  = bipp.Context(args.proc, comm)
opt  = bipp.NufftSynthesisOptions()
opt.set_tolerance(args.tol)              # default 1e-3
opt.set_precision(args.float_precision)  # "single" / "double"
if args.uvw_part is not None:
    opt.set_local_uvw_partition(bipp.Partition.grid(args.uvw_part))
bipp.image_synthesis(ctx, opt, dataset, selection, image_prop, args.output)
```

The C++ entry point is `bipp::image_synthesis` defined at
`include/bipp/image_synthesis.hpp:160-163` and implemented in
`src/image_synthesis.cpp` (185 lines). The implementation:

1. Validates each selection tag (forbids `/`, `.`, space — see
   `image_synthesis.cpp:38-46`) and that all tags share the same sample id
   list, sorted, no duplicates, none exceeding `dataset.num_samples()`.
2. Splits samples across MPI ranks by `nSamplePerRank = ⌈nTotal/comm.size()⌉`.
3. Materialises a `dScaled` `HostArray<float, 3>` of shape
   `(nBeam, nLocalSamples, nImages)` from the user's filtered eigenvalues.
4. Calls `imageProp.pixel_lmn(...)` to get the `(nPixel, 3)` pixel grid.
5. Dispatches on `opt.precision` to either
   `nufft_synthesis<float>` or `nufft_synthesis<double>`
   (`src/nufft_synthesis.cpp:74-170`).
6. Per-rank NUFFT loop (see §5.5): for each sample, read uvw / eig vec,
   scale uvw by `2π/λ`, build virtual visibilities for every image, and
   accumulate the NUFFT type-3 transform via `host::NUFFT<T>::add(uvw, vis)`
   (host) or `gpu::NUFFT<T>` (when compiled).
7. After all samples are added the NUFFT batch is finalised in
   `get_image(idx, slice)`; resulting real-part pixel values are summed
   across MPI ranks via `MPI_Reduce(..., MPI_SUM, 0, comm)`.
8. On rank 0 the image is normalised by `1/nTotalSamples` if
   `opt.normalizeImage`, and written to the output `ImageDataFile`.

Image data file layout (HDF5, see `src/image_data_file.cpp:50-90`):

- root attributes `width`, `height`, `fovDeg`, `raDeg`, `decDeg`,
- group `images/<tag>` containing one 1-D `float` dataset of length
  `width*height` per selection tag.

### 4.5 `bipp plot` — render PNGs

`apps/plot_images.py:15-28` opens the image file, sorts tags
alphabetically, and for each tag draws a Matplotlib figure with
`cmap='cubehelix'` and saves `<tag>.png` at 200 dpi.

## 5. Algorithmic core

### 5.1 Eigendecomposition `eigh` / `eigh_gram`

Public C++ API (`include/bipp/bipp.hpp:32-75`):

```cpp
template <typename T>  // T = float or double
auto eigh(T wl, std::size_t nAntenna, std::size_t nBeam,
          const std::complex<T>* s, std::size_t lds,
          const std::complex<T>* w, std::size_t ldw,
          T* d, std::complex<T>* v, std::size_t ldv)
    -> std::pair<std::size_t, T>;

template <typename T>
auto eigh_gram(T wl, std::size_t nAntenna, std::size_t nBeam,
               const std::complex<T>* s, std::size_t lds,
               const std::complex<T>* w, std::size_t ldw,
               const T* xyz, std::size_t ldxyz,
               T* d, std::complex<T>* v, std::size_t ldv)
    -> std::pair<std::size_t, T>;
```

Both return `(n_computed_eigenvalues, scaling_factor)`. `eigh` solves the
plain Hermitian eigenproblem on the visibility matrix `S`; `eigh_gram`
solves the generalised eigenproblem with the antenna-domain Gram matrix
`G` formed inside the function.

Implementation in `src/eigensolver.cpp:21-225`:

1. Pre-flag rows/columns of `s` that are entirely zero
   (`|val| ≥ ε` test; `nVis` counts non-zero strict-upper plus diagonal
   entries). Eliminating dead beams keeps LAPACK out of the singular
   regime.
2. Copy the lower-triangle of the (possibly reduced) `s` into the work
   matrix and shrink `w` accordingly via
   `copy_lower_triangle_at_indices(...)`.
3. For `eigh_gram`, compute `g = w^H Φ w` with
   `host::gram_matrix<T>(alloc, w, xyz, wl, g)` — `Φ_{ij} = sinc_pi(2 |x_i − x_j| / λ)`,
   filled lower-triangular, multiplied left/right with `w` via two
   `cblas_?symm` / `cblas_?gemm` calls (`src/host/gram_matrix.cpp:23-65`).
4. Call `host::lapack::eigh_solve(layout, itype, mode, uplo, N, ...)`:
   - `eigh` → `LAPACKE_chegv`/`zhegv` with single-matrix mode (no Gram).
     Actually `eigh_solve` without `g` ends up using the standard
     Hermitian eigensolver (`?heev`) — see the no-`g` overload picked up
     in line 94.
   - `eigh_gram` → generalised symmetric definite eigenproblem with
     `itype=1` (`A x = λ B x`) on `(s, g)`.
5. After the eigenvalue solve, the eigenvectors `vArray` (in beam space)
   are *un-beamed* back to antenna space via
   `wView · vArray → vUnbeam` (a `gemm`), so that the eigenvectors
   exposed to the rest of BIPP are antenna-domain `(N_ant, N_eig)`
   arrays — exactly what the NUFFT layer expects.
6. The `scalingFactor` is `1 / nVis` (or zero if no visibilities); it is
   applied later to the virtual visibilities.

Python bindings (`python/pybipp.cpp:234-275, 462-492`) expose
`eigh(wl, s, w)` and `eigh_gram(wl, s, w, xyz)` for `float32` *and*
`float64` complex inputs, returning `(v, d, scale)`. Note: the order
`v, d, scale` is opposite of the C++ `(d, v)` style — examples in
`create_dataset.py:99-101` and `lofar_bootes_nufft.py:100` rely on this
3-tuple unpack.

### 5.2 UVW geometry (`bipp.frame`)

`uvw_basis(field_center)` (`frame.py:9-37`) constructs the `(3, 3)`
ICRS→UVW transform whose columns are:

- `u = (-sin λ, cos λ, 0)`,
- `v = (-cos λ sin δ, -sin λ sin δ, cos δ)`,
- `w = (cos λ cos δ, sin λ cos δ, sin δ)` (the unit pointing vector).

`make_grids(grid_size, FoV, field_center)` (`frame.py:40-72`) returns
`(lmn_grid, xyz_grid)` where `xyz_grid = uvw_basis(field_center) @ lmn_grid`
and `n = sqrt(1 - l² - m²)` (no slant-projection trick). The grid is
offset by `±lim/grid_size * 0.5` between axes to align with WSClean/CASA
pixel centres.

`reshape_uvw(UVW)` (`frame.py:75-89`): given an `(N_ant, N_ant, 3)` baseline
tensor, transpose `(1,0,2)` and ravel into a Fortran-order `(N_ant², 3)`
array — the layout expected by the C++ NUFFT path (`pybipp.cpp:163`
asserts this shape).

### 5.3 Beamforming (`bipp.beamforming`)

`MatchedBeamformerBlock(beam_config)` (`beamforming.py:166-236`): takes a
list of `(station_id, beam_id, focus_dir)` triples (validated by
`is_mb_beam_config`), and on `__call__(XYZ, wl)` produces complex
`(N_antenna, N_beam)` weights:

```
W_ij = exp(-i 2π (XYZ_i - XYZ_mean) · F_j / λ)
```

restricted to (antenna belongs to station of beam `j`). Internally the
DataFrames are merged on `STATION_ID` and reduced to a sparse-friendly
`(N_ant, N_beam)` BeamWeights matrix.

`SKALowMeasurementSet` exposes both `beamformer` (matched, single beam
per station, focus = field centre) and `beamformer_identity` (a
`MatchedBeamformerBlockIdentity` variant — declared but not present in
`beamforming.py`; the file is incomplete on that front, so callers using
the identity path must depend on a private subclass).

### 5.4 Eigenvalue filters

`python/bipp/filter.py:1-54` defines the entire filter library. A
`Filter(name, lo, hi)` is a callable that, given an eigenvalue array `D`,
zeros out entries outside `[lo, hi]` and applies one of:

| Name        | Operation                                          |
|-------------|----------------------------------------------------|
| `lsq`       | `D` (identity)                                     |
| `std`       | `sign(D)`                                          |
| `sqrt`      | `sign(D) * sqrt(|D|)`                              |
| `inv_sqrt`  | `1 / (sign(D) * sqrt(|D|))` (zero where `D = 0`)  |
| `inv`       | `1 / D` (zero where `D = 0`)                       |

`apply_filter` is also available standalone. `lsq` is the canonical
"least-squares dirty image"; `std` is binary-detection mode; the others
support inverse-Wiener-style weights.

### 5.5 Virtual visibilities and NUFFT type-3

`src/host/virtual_vis.cpp:19-73`: given `dMasked` (N_eig filtered eigenvalues),
`vAll` (N_ant × N_eig antenna-space eigenvectors), and `scale` (the
`1/nVis` returned by the eigensolver), this kernel:

1. Compacts to non-zero entries (skips zeros directly so `gemm` is on
   `nEig_actual` columns only).
2. Multiplies each surviving eigenvector by `scale * d`.
3. Computes `virtVis = vScaled · v^H` via a `cblas_?gemm` with
   `CblasNoTrans, CblasConjTrans, alpha=1, beta=0` and reshapes the
   output to a flat `(N_ant²,)` view (column-major). The result is the
   *virtual visibility* matrix used by the NUFFT.

`src/host/nufft.hpp:32-234` is the host NUFFT engine. Construction
allocates contiguous `valueCollection_` and `uvwCollection_` buffers
sized to `sampleBatchSize * nBaselines`, plus a `pixelXYZ_` copy
(possibly upcast to double). On every `add(uvw, values)` it copies the
new sample into the buffers and, when the batch is full, runs
`transform()`:

1. Build a `host::DomainPartition` of the UVW points using the policy
   selected in `NufftSynthesisOptions::localUVWPartition`:
   - `Partition::Grid{nx, ny, nz}` — bucket on a regular UV-grid
     (smaller 3-D NUFFT problems, less memory),
   - `Partition::Auto{}` — call `optimal_parition_size<T>(uvw, pixelMin,
     pixelMax, maxMem, neonufft::PlanT3<T,3>::grid_memory_size, ...)` which
     selects the smallest grid that fits in `2/3` of system memory,
   - `Partition::None{}` — single bucket.
2. Apply the partition permutation to `u`, `v`, `w` and to all images'
   `valueCollection_` columns.
3. For each non-empty partition group, instantiate a
   `neonufft::PlanT3<T, 3>(neoOpt, nrhs=1, uvwMin, uvwMax, pixelMin, pixelMax)`
   plan, set input/output points (xyz pixel coordinates as the type-3
   target frequencies), `add_input(values)` then `transform(imageCpx)`,
   accumulate the **real part** of `imageCpx` into `images_[image_idx]`
   and `plan.reset()` between images.

`neonufft::Options` used:
`tol = opt.tolerance`, `sort_input = false`, `sort_output = false` (we
already partitioned/sorted ourselves).

### 5.6 GPU path

The GPU pipeline currently in source:

- `src/gpu/util/runtime.hpp`, `runtime_api.hpp`, `device_guard.hpp`,
  `device_pointer.hpp`, `kernel_launch_grid.hpp`, `queue.hpp`,
  `blas_api.hpp`, `cub_api.hpp` — abstraction over CUDA / HIP runtimes
  (`#if defined(BIPP_CUDA) … #elif defined(BIPP_ROCM) …`).
- `src/gpu/nufft.hpp` — header for `gpu::NUFFT<T>` referenced from
  `nufft_synthesis.cpp:120-122`.
- `src/gpu/kernels/add_vector.{cu,hpp}` and `nuft_sum.{cu,hpp}`.

`nufft_synthesis.cpp:117-128` chooses `gpu::NUFFT<T>` when
`ctx.processing_unit() == BIPP_PU_GPU` and `BIPP_CUDA || BIPP_ROCM`,
otherwise raises `GPUSupportError`. The remaining device-side virtual_vis
/ gram_matrix / eigensolver implementations from earlier BIPP versions
are commented out in `src/CMakeLists.txt` — they exist as design guidance
but are not active in the v1.0.0 build.

### 5.7 MPI distribution

When compiled with `BIPP_MPI=ON`, `Communicator::world()`
(`include/bipp/communicator.hpp:38`) duplicates `MPI_COMM_WORLD`. Inside
`image_synthesis.cpp:75-148` the sample list is split across ranks
(`nSamplePerRank = ⌈nTotal/comm.size()⌉`) and the per-rank pixel images
are summed onto rank 0 with `MPI_Reduce(...)`. Rank 0 alone writes the
HDF5 image file; other ranks `MPI_Barrier` at the end. `Communicator::custom(MPI_Comm)`
allows embedding BIPP into existing MPI codes.

## 6. File formats

BIPP operates on three HDF5 file types and one JSON file; format versions
are constants in `src/io/dataset_spec.hpp`:

```cpp
constexpr unsigned int datasetFormatVersionMajor = 1;
constexpr unsigned int datasetFormatVersionMinor = 0;
```

Major-version mismatch raises `bipp::FileError` on open; minor-version
mismatch is tolerated so long as the file's minor ≤ code's minor.

### 6.1 Dataset HDF5 (`bipp.DatasetFile`)

Created/opened by `bipp::DatasetFile` (`include/bipp/dataset_file.hpp`,
implementation `src/dataset_file.cpp`). On `create(file, description,
nAntenna, nBeam, raDeg, decDeg)`:

- Root attributes: `formatVersionMajor`, `formatVersionMinor`, `nBeam`,
  `nAntenna`, `description`, `raDeg`, `decDeg`.
- Array types:
  - `eigVal` — `H5Tarray_create(float, 1, [nBeam])`,
  - `eigVec` — `H5Tarray_create(float, 2, [nBeam, 2*nAntenna])` (real/imag
    interleaved),
  - `uvw` — `H5Tarray_create(float, 2, [3, nAntenna*nAntenna])`,
  - `xyz` — `H5Tarray_create(float, 2, [3, nAntenna])`.
- 1-D datasets (chunk size 5): `wl`, `time`, `scale`, `eigVal`, `eigVec`, `uvw`.

`write(time, wl, scale, v, d, uvw)` appends to all six datasets in
lock-step. Input layout (Python side, see `pybipp.cpp:156-169`):
`v` is `(nAntenna, nBeam) complex64` Fortran-ordered, `d` is `(nBeam,)`
float32, `uvw` is `(nAntenna², 3)` float32 Fortran-ordered.

The Python wrapper `DatasetFile` (pybind, `pybipp.cpp:347-372`) supports
context-manager use (`with bipp.DatasetFile.create(...) as ds: ...`) and
exposes `eig_vec(idx)`, `eig_val(idx)`, `uvw(idx)`, `wl(idx)`,
`scale(idx)`, `num_samples()`, `num_beam()`, `num_antenna()`, `ra_deg()`,
`dec_deg()`.

The C++ `Dataset` interface (`include/bipp/dataset.hpp`) is the abstract
base; `DatasetFile` is the only concrete implementation.

### 6.2 Image properties HDF5 (`bipp.ImagePropFile`)

Stores the imaging grid produced by `frame.make_grids`. On
`create(file, height, width, fovDeg, lmn[width*height, 3])`:

- attributes `width`, `height`, `fovDeg`,
- dataset `lmn` of shape `(width*height, 3)` float32,
- arbitrary metadata accessible via `set_meta(name, value)` /
  `meta_data()` where `value ∈ {size_t, float, vector<float>}`.

### 6.3 Image data HDF5 (`bipp.ImageDataFile`)

Created at the end of `bipp::image_synthesis`. Layout
(`src/image_data_file.cpp:50-90`):

- root attributes `width`, `height`, `fovDeg`, `raDeg`, `decDeg`,
- group `images/`, with one 1-D `float` dataset per selection tag
  (length `width*height`, row-major).

Tag names cannot contain `/`, `.`, or space (enforced both when writing
and when synthesising).

### 6.4 Selection JSON

Written by `selection.export_selection(dict, file)`
(`python/bipp/selection.py:19-31`). Schema:

```json
{
  "lsq_level_0": {
    "0":   [<nBeam floats>],
    "1":   [<nBeam floats>],
    ...
  },
  "lsq_level_1": { ... },
  ...
}
```

Each top-level key is a *tag* (level + filter); each second-level key is
a string sample id; each value is a list of `nBeam` filtered eigenvalues
(zeros for clipped levels). Numpy types are encoded via
`NumpyArrayEncoder` (int → int, float → float, ndarray → list).

The C++ standalone executable `bipp_synthesis` (`src/apps/bipp_synthesis.cpp`)
reads the same JSON via `nlohmann::json` and converts `string` keys back
to `std::stoull(...)` sample ids before calling `bipp::image_synthesis`.

## 7. Public C++ API summary

Headers under `include/bipp/`:

- **`bipp.hpp`** — umbrella include; declares `eigh<T>` and `eigh_gram<T>`
  plus pulls in the rest.
- **`config.h.in`** — generated to `config.h`; defines `BIPP_VERSION`,
  `BIPP_EXPORT`, `BIPP_CUDA`, `BIPP_ROCM`, `BIPP_OMP`, `BIPP_MPI`,
  `BIPP_UMPIRE`, `BIPP_BLAS_C`, `BIPP_LAPACK_C`.
- **`enums.h`** —
  - `enum BippProcessingUnit { BIPP_PU_AUTO, BIPP_PU_CPU, BIPP_PU_GPU };`
  - `enum BippPrecision { BIPP_PRECISION_SINGLE, BIPP_PRECISION_DOUBLE };`
  - `enum BippLogLevel { OFF, ERROR, WARN, INFO, DEBUG };`
- **`errors.h`** — C error codes (`BippError` enum) for the C ABI.
- **`exceptions.hpp`** — exception hierarchy:
  `GenericError` → `InternalError`, `InvalidParameterError`,
  `InvalidPointerError`, `InvalidAllocatorFunctionError`,
  `EigensolverError`, `HDF5Error`, `NotImplementedError`, `FileError`,
  `GPUError` → `GPUSupportError`, `GPUBlasError`, `MPIError`.
- **`communicator.hpp`** — `Communicator::world()`, `Communicator::local()`,
  `Communicator::custom(MPI_Comm)`, `rank()`, `size()`, `mpi_handle()`.
- **`context.hpp`** — `Context(BippProcessingUnit)` and
  `Context(BippProcessingUnit, Communicator)`; movable, non-copyable.
  Internally holds a `std::shared_ptr<ContextInternal>` that owns
  allocators, OMP thread-pool config, and (when GPU) a `gpu::Queue` per
  device.
- **`dataset.hpp` / `dataset_file.hpp`** — abstract `Dataset` and concrete
  `DatasetFile` (HDF5, see §6.1). All accessors take an integer sample
  index (`std::size_t index`); raw pointers and leading dimensions follow
  the BLAS column-major convention.
- **`image_prop.hpp` / `image_prop_file.hpp`** — abstract `ImageProp` and
  `ImagePropFile` (HDF5, see §6.2). `MetaType = std::variant<std::size_t,
  float, std::vector<float>>`.
- **`image_data.hpp` / `image_data_file.hpp`** — abstract `ImageData` and
  `ImageDataFile` (HDF5, see §6.3). Accessors `tags()`, `num_tags()`,
  `get(tag, image)`, `set(tag, image)`.
- **`image_synthesis.hpp`** — `Partition` (`Auto`, `None`,
  `Grid{dimensions[3]}`), `NufftSynthesisOptions`
  (`precision`, `tolerance` 1e-3 default, `sampleBatchSize` opt,
  `localUVWPartition`, `normalizeImage` true, `apply_scaling` true), and
  the free function `image_synthesis(ctx, opt, dataset, selection,
  imageProp, imageFileName)`.

## 8. Public Python API summary

Top-level `bipp` module re-exports everything from the `pybipp` C extension
(`python/bipp/__init__.py: from .pybipp import *`). The `pybipp` module
binds the following objects (`python/pybipp.cpp`):

- `bipp.config` — `CompileConfig` with bool flags
  `cuda, rocm, umpire, omp, mpi`.
- `bipp.communicator` (note the lowercase, this is the bound class
  `Communicator`, not a submodule) — `world()`, `local()`, `size`,
  `rank`. `bipp.communicator.world()` returns a `Communicator` instance.
- `bipp.Context(pu_str)` / `bipp.Context(pu_str, comm)` —
  `pu_str ∈ {"auto", "cpu", "gpu"}`. Property `processing_unit`.
- `bipp.Partition` — class with `auto()`, `none()`, `grid([nx, ny, nz])`
  static factories.
- `bipp.NufftSynthesisOptions()` — readwrite fields `tolerance`,
  `sample_batch_size`, `local_uvw_partition`, `normalize_image`,
  `apply_scaling`; setters `set_tolerance`, `set_sample_batch_size`,
  `set_local_uvw_partition`, `set_normalize_image`, `set_apply_scaling`,
  `set_precision`; readonly `precision` (returns "single"/"double").
- `bipp.DatasetFile` — `open(name)`, `create(name, description, n_antenna,
  n_beam, ra_deg, dec_deg)`, methods `close, is_open, num_samples,
  num_beam, num_antenna, ra_deg, dec_deg, eig_vec(idx), eig_val(idx),
  uvw(idx), wl(idx), scale(idx), write(time, wl, scale, v, d, uvw)`.
  Context-manager protocol implemented (`__enter__/__exit__`).
- `bipp.ImagePropFile` — `open`, `create(name, height, width, fov_deg,
  lmn[width*height,3])`, `pixel_lmn()`, `set_meta(name, value)`,
  `meta_data()`, `width`, `height`, `fov_deg`. Context-manager.
- `bipp.ImageDataFile` — `open`, `create(name, height, width, fov_deg,
  ra_deg, dec_deg)`, `tags()`, `num_tags()`, `get(tag) -> 2-D ndarray`,
  `set(tag, image)`, `width/height/fov_deg/ra_deg/dec_deg`.
  Context-manager.
- `bipp.image_synthesis(ctx, opt, dataset, selection_dict, image_prop, output_filename)`
  — selection is `Dict[str, Dict[int, List[float]]]`.
- `bipp.eigh(wl, s, w)` — float32 / float64 overloads, returns `(v, d, scale)`.
- `bipp.eigh_gram(wl, s, w, xyz)` — same signature plus `xyz`.

Pure-Python helper modules (importable as `bipp.<module>`):

| Module | Purpose |
|--------|---------|
| `bipp.core` | `Block` ABC for callable building blocks (`Block.__call__`) |
| `bipp.array` | `LabeledMatrix(data, row_idx, col_idx)` — pandas-indexed 2-D arrays |
| `bipp.numpy_compat` | `asarray` shim (used by `gram.py` / `array.py`) |
| `bipp.filter` | `Filter`, `apply_filter` (§5.4) |
| `bipp.frame` | `uvw_basis`, `make_grids`, `reshape_uvw` (§5.2) |
| `bipp.gram` | `GramMatrix`, `GramBlock(ctx)` — Python fallback wraps `bipp.pybipp.gram_matrix` if `ctx` provided, else uses the local `4π · sinc(2|baseline|/λ)` definition (`gram.py:97-112`) |
| `bipp.source` | `SkyEmission`, `from_tgss_catalog(direction, FoV, N_src)` (caches `~/.bipp/catalog/TGSSADR1_7sigma_catalog.tsv`) |
| `bipp.instrument` | `InstrumentGeometry`, `InstrumentGeometryBlock`, `EarthBoundInstrumentGeometryBlock`, `LofarBlock(N_station, station_only)`, `MwaBlock(N_station, station_only)` |
| `bipp.beamforming` | `BeamWeights`, `BeamformerBlock`, `MatchedBeamformerBlock` |
| `bipp.statistics` | `VisibilityMatrix(data, beam_idx, check_hermitian, weight_spectrum)`, `VisibilityGeneratorBlock(sky_model, T, fs, SNR)` |
| `bipp.parameter_estimator` | `centroid_to_intervals`, `infer_intervals(N_level, sigma, cluster_func, d_min, d_max, d_all)` |
| `bipp.selection` | `NumpyArrayEncoder`, `export_selection(dict, file)` |
| `bipp.measurement_set` | `MeasurementSet` (base), `LofarMeasurementSet(file, N_station, station_only)`, `MwaMeasurementSet(file)`, `SKALowMeasurementSet(file, N_station)` (also `beamformer_identity`) |
| `bipp.imot_tools.*` | vendored S. Kashani math/IO utilities (s2image, plotting, sphere transforms, statistics, argcheck, …) |

Bundled instrument data (`python/bipp/data/instrument/`):

- `LOFAR.csv` — STATION_ID/ANTENNA_ID indexed ITRF (X,Y,Z) coordinates
  for the 62 LOFAR stations (HBA tile centres). `LofarBlock`
  optionally collapses each station to its centroid (`station_only=True`)
  or returns all individual elements.
- `MWA.csv` — 128 MWA tile centres. `MwaBlock` either uses the centroid
  (`station_only=True`) or expands to a flat 4×4 element grid per tile
  rotated to the local horizontal plane via two successive
  `pylinalg.rot` calls (`instrument.py:430-460`).

## 9. Measurement-Set readers (`bipp.measurement_set`)

The MS layer wraps **python-casacore** (`casacore.tables`) and is the
glue between observatory MS files and `bipp.DatasetFile`. The base
`MeasurementSet` (`measurement_set.py:60-342`):

- Validates that the path exists and is a directory.
- Lazily caches `field_center` (from `FIELD::REFERENCE_DIR`),
  `channels` (from `SPECTRAL_WINDOW::CHAN_FREQ`), `time` (unique MJD
  values from `MAIN::TIME`).
- `instrument` and `beamformer` are abstract — subclasses fill them.

`visibilities(channel_id, time_id, column='DATA', sort_time=True, log_level=0)`
(`measurement_set.py:209-342`) iterates the MAIN table grouped by `TIME`
and yields `(astropy.time.Time, frequency_quantity, VisibilityMatrix)`.
For each integration:

1. Read `ANTENNA1`, `ANTENNA2` columns.
2. Read `column[chan, polarization]` (only the parallel-hand pair `[0, 1]`
   = XX and YY for full-Stokes correlations; XY/YX dropped). Channels
   are read in chunks of 8 if contiguous (`chunk_channel_block`).
3. Read `FLAG[..., [0,1]]` and OR-reduce to a per-baseline bool. Flagged
   data is zeroed in place.
4. Optionally read `WEIGHT_SPECTRUM[..., [0,1]]` and use the
   per-baseline minimum (mimicking WSClean's reduction). Flag-masked
   weights are zeroed.
5. Build a `(N_ant, N_ant)` complex coherency matrix `S` where
   `S[ant2, ant1] = data[:, chan]`. Construct `vis.VisibilityMatrix(S,
   beam_idx, check_hermitian=False, weight_spectrum=WS)`. The
   `VisibilityMatrix` constructor (`statistics.py:37-67`):
   - zeros the diagonal (always flag autocorrelations),
   - prunes `weight_spectrum` to the visibility support,
   - rescales `data *= weight_spectrum / sum(weight_spectrum) * nz_vis`
     so the average is preserved.
6. Skip empty integrations (channels where everything was flagged).

`LofarMeasurementSet(file, N_station, station_only)`
(`measurement_set.py:399-513`):

- Reads `LOFAR_ANTENNA_FIELD::ANTENNA_ID, POSITION, ELEMENT_OFFSET,
  ELEMENT_FLAG`. Collapses to station centroid if `station_only`,
  filters flagged elements, slices first `N_station` stations.
- `beamformer` returns a `MatchedBeamformerBlock` with one beam per
  station pointed at `field_center`.
- The data path conjugates `S` (`create_dataset.py:99`) to compensate
  for the LOFAR convention bug discussed in
  `wsclean.readthedocs.io`.

`MwaMeasurementSet(file)` (`measurement_set.py:516-588`):

- Reads `ANTENNA::POSITION` (one row per station, station IDs implicit
  in row order).
- Beamformer is `MatchedBeamformerBlock` (treats MWA tile data as
  already-beamformed single-beam-per-station outputs).

`SKALowMeasurementSet(file, N_station=None)`
(`measurement_set.py:591-689`):

- Same `ANTENNA::POSITION` strategy as MWA.
- `beamformer` (matched) and `beamformer_identity` (uses
  `MatchedBeamformerBlockIdentity`).

`GenericMeasurementSet` is referenced by `create_dataset.py:20` for the
`-t redundant` option — its definition is not yet present in
`measurement_set.py` in this checkout, so the `redundant` branch of
`bipp dataset` is not functional in this snapshot.

To add a new telescope, subclass `MeasurementSet` and provide
`instrument` (returning an `EarthBoundInstrumentGeometryBlock` over an
`InstrumentGeometry` built from MS metadata) and `beamformer` (typically
`MatchedBeamformerBlock` with one beam per station).

## 10. Internal C++ infrastructure

### 10.1 `Context` and `ContextInternal`

`Context` (public, `include/bipp/context.hpp`) wraps a
`std::shared_ptr<ContextInternal>`. `ContextInternal`
(`src/context_internal.hpp`) holds the host allocator, the chosen
`BippProcessingUnit`, optional GPU queue/state, and the global logger
sink. `InternalContextAccessor` is a friend struct used by
`image_synthesis.cpp:32` to lift the internal pointer out of the public
handle.

### 10.2 Logger and rt_graph

`src/logger.{cpp,hpp}` wraps spdlog (`spdlog::spdlog_header_only`) and
exposes `globLogger` with `start_timing(level, name)`,
`stop_timing(level, name)`, `scoped_timing(level, name)` (RAII timer),
`log(level, fmt, args...)` and `log_matrix(level, name, view)` for
debug-level matrix dumps. Log levels mirror spdlog's: OFF, ERROR, WARN,
INFO, DEBUG (`enums.h`). `BIPP_LOG_LEVEL` env var controls runtime
verbosity.

`src/rt_graph.{cpp,hpp}` (named after the Runtime Graph abstraction from
SIRIUS/SPLA) provides hierarchical wallclock breakdowns of the timed
sections; the JSON output can be post-processed for performance studies.

### 10.3 Memory management

`src/memory/`:

- `allocator.hpp` — abstract `Allocator { allocate(size) -> void*; deallocate; ... }`.
- `allocator_factory.{cpp,hpp}` — `AllocatorFactory::simple_host()`,
  `AllocatorFactory::umpire_host()` (when `BIPP_UMPIRE=ON`),
  `AllocatorFactory::pool(...)`, GPU variants when compiled.
- `pool_allocator.hpp` — slab/pool allocator used to amortise frequent
  same-size allocations on the host.
- `umpire_allocator.hpp` — adapter to LLNL Umpire (`umpire::ResourceManager`).
- `array.hpp`, `view.hpp` — `HostArray<T,N>`, `HostView<T,N>` and
  `ConstHostView<T,N>` are strided multi-dim views. Layout is column-major
  by convention (`strides(1)` is the leading dimension); `is_contiguous()`,
  `slice_view(i)`, `sub_view(offset, shape)`, `data()`, `shape()`,
  `size()`, `zero()` are standard.
- `copy.hpp` — overloaded `copy(src, dst)` between Host/GPU views,
  including float→double upcasts.

Single-precision MS data flows through the dataset HDF5 file as
`float32`, then `nufft_synthesis<T>` upcasts to `T = double` via
`copy(...)` if `opt.precision == BIPP_PRECISION_DOUBLE` (`nufft_synthesis.cpp:46-71`).

### 10.4 BLAS / LAPACK abstractions

`src/host/blas_api.hpp` thin wrappers over CBLAS — `cblas_zgemm`,
`cblas_chemm`, `cblas_csymm`, `cblas_chemv`, `cblas_zscal`, etc. — with
template overloads that dispatch on `T = float, double, std::complex<float>,
std::complex<double>`.

`src/host/lapack_api.hpp` wraps LAPACKE — `LAPACKE_chegv`, `LAPACKE_zhegv`,
`LAPACKE_cheev`, `LAPACKE_zheev`. Both wrappers honour the cached config
checks `BIPP_BLAS_C` / `BIPP_LAPACK_C`. The `eigh_solve(...)` overload
set in `eigensolver.cpp` is the only call site.

### 10.5 NUFFT plumbing

- `src/nufft_interface.hpp` — `NUFFTInterface<T>` with `add(uvw, vis)` and
  `get_image(idx, image)`.
- `src/host/nufft.hpp` — host implementation around `neonufft::PlanT3<T,3>`.
- `src/gpu/nufft.hpp` — GPU implementation (placeholder in this build).
- `src/nufft_util.hpp` — `optimal_parition_size<T>(...)` chooses an
  `(nx, ny, nz)` UV partition that keeps neonufft FFT memory below the
  given budget. The estimator uses
  `neonufft::PlanT3<T,3>::grid_memory_size(opt, uvwMin, uvwMax, xyzMin,
  xyzMax)` per candidate grid.
- `src/host/domain_partition.hpp` — `DomainPartition::grid<T,3>(alloc,
  dims, {u_view, v_view, w_view})` builds a partition by binning each
  point and assigns groups; `apply(view)` permutes any view in-place.
  `DomainPartition::none(alloc, n)` produces a single-group identity
  partition.
- `src/host/kernels/nuft_sum.cpp` — `host::nuft_sum<T>(α, nIn, input, u,
  v, w, nOut, x, y, z, out)` — direct DFT fallback used in tests and
  small problems (`out += α · sum_i input[i] exp(i (u_i x + v_i y + w_i
  z))`). The function is generic over `float`/`double`.

## 11. Standalone C++ executable: `bipp_synthesis`

`src/apps/bipp_synthesis.cpp` (109 lines) is a CLI11-driven C++ binary
that runs the synthesis stage end-to-end without Python. Flags:

```
bipp_synthesis -d dataset.h5 [-s selection.json] -i image_prop.h5
               -o image_data.h5 [-t tol] [-p auto|cpu|gpu]
               [-f single|double] [--uvw_part nx ny nz]
```

If `-s` is omitted, every sample is included as one image with all
per-sample eigenvalues (i.e. unfiltered, single-level imaging — useful
for the LOFAR test fixture). The implementation:

1. Opens dataset and reads `selectionData` either from the JSON file
   (parsed by nlohmann::json) or by replicating each sample's raw
   `eig_val` array under tag `full_image`.
2. Validates that each eigenvalue list has length `dataset.num_beam()`
   and that `index < dataset.num_samples()`.
3. Builds `bipp::Context(pu, Communicator::world())` and a
   `NufftSynthesisOptions` with the user's `tolerance`, `precision`, and
   `localUVWPartition`.
4. Calls `bipp::image_synthesis(ctx, opt, dataset, selection, imageProp,
   imageDataFileName)`.

This is the C++-only entry point used in CI; Python tests reuse the
binary via `subprocess`.

## 12. Tests

`tests/` builds a single GoogleTest binary (`run_tests.cpp` registers
the tests and `CMakeLists.txt` links `gtest` and `nlohmann_json`).

- `test_domain_partition.cpp` — exercises the host
  `DomainPartition::grid<T,3>` permutation, validating that the same
  permutation applied to multiple aligned views keeps the data in step.
- `test_nufft_synthesis_lofar.cpp` — full numerical regression test:
  loads `tests/data/lofar_input.json` (the LOFAR-Bootes reference
  setup), runs `bipp::image_synthesis` for both single and double
  precision, and compares pixel arrays against
  `tests/data/lofar_nufft_output_single.json` and
  `lofar_nufft_output_double.json`.

`BIPP_TEST_DATA_DIR` is configured by CMake when `BIPP_BUILD_TESTS=ON`.

## 13. Examples

`examples/simulation/lofar_bootes_nufft.py` is the canonical end-to-end
demo (258 lines):

1. Builds `bipp.communicator.local()` and `bipp.Context("AUTO", comm)`
   and prints `bipp.config` flags.
2. Defines a Bootes-field observation: `ra=218°, dec=34.5°, FoV=10°,
   ν=145 MHz`, MJD start `56879.54171302732`, 24-station LOFAR
   (`instrument.LofarBlock`), 8-second integrations across 3595
   timestamps.
3. Generates a synthetic sky from `source.from_tgss_catalog(field_center,
   FoV, N_src=40)` and a `statistics.VisibilityGeneratorBlock(sky, T=8,
   fs=196_000, SNR=30)` (Wishart-distributed visibilities).
4. For each `time[::25]` builds `XYZ`, `UVW`, `W`, `S = vis(XYZ, W, λ)`,
   then `bipp.eigh_gram(λ, S.data, W.data, XYZ.data)`, and calls
   `dataset.write(...)`.
5. Estimates intervals (`bb_pe.infer_intervals(3, 1.0, 'log', 0,
   inf, eig_values)`) and builds two filter sets (`lsq`, `std`) for
   3 levels each, exporting to `selection.json`.
6. Creates `image_prop.h5` with `bipp.ImagePropFile.create(...)` and
   sets `fov` metadata.
7. Runs `bipp.image_synthesis(ctx, opt, dataset, selection,
   image_prop, image_data_file)` with `tol=1e-3, precision='single',
   Partition.auto()`.
8. Reads images back, separates into `lsq_*` and `std_*` arrays,
   wraps each as `s2image.Image(images, xyz_grid)` and renders a
   3-panel "Strong / Mild / Faint sources" figure with `cmap='cubehelix'`
   and the simulated catalog overlaid.

This script is the recommended on-ramp for new users; everything that
the `bipp` CLI does internally is shown explicitly.

`test_writer.py` is an abridged version (without final plotting) used as
a regression generator.

## 14. Documentation

Sphinx + Breathe under `docs/`:

- `docs/Doxyfile` — standard Doxygen config that scans `include/bipp`.
- `docs/source/index.rst` — toctree pulls
  `context.rst`, `exceptions.rst`, `synthesis.rst`,
  `eigendecomposition.rst`. Each `.rst` consists of a single
  `.. doxygenfile::` directive over the matching public header.
- `docs/source/templates/` — overrides for the readthedocs theme.
- `docs/requirements.txt` — sphinx + breathe + sphinx_rtd_theme pins.
- `.readthedocs.yml` — Doxygen + Sphinx build configuration.

Hosted at `https://bipp.readthedocs.io/`.

## 15. Configuration knobs that affect numerical results

| Setting                                          | Where                                              | Effect |
|--------------------------------------------------|----------------------------------------------------|--------|
| `NufftSynthesisOptions.tolerance`                | C++ / Python (`set_tolerance`)                     | NUFFT type-3 absolute tolerance, default `1e-3` |
| `NufftSynthesisOptions.precision`                | `single` / `double`                                | host-side dtype (eigenvectors are still stored as `float32` on disk; double activates copy + upcast) |
| `NufftSynthesisOptions.sample_batch_size`        | optional                                           | controls how many samples are accumulated before NUFFT transform; default 2 GB / (20 B per baseline) heuristic |
| `NufftSynthesisOptions.local_uvw_partition`      | `auto/none/grid(nx,ny,nz)`                         | partition strategy for memory-FFT trade-off |
| `NufftSynthesisOptions.normalize_image`          | bool                                               | divide image by `nTotalSamples` (default true) |
| `NufftSynthesisOptions.apply_scaling`            | bool                                               | apply per-sample `scale` from `eigh*` (default true) |
| `infer_intervals(n_levels, sigma, cluster, ...)` | Python `parameter_estimator`                       | k-means clustering trims top `(1-sigma)*N` eigenvalues; `log`/`none` chooses cluster space |
| `Filter(name, lo, hi)`                           | Python `filter`                                    | `lsq/std/sqrt/inv/inv_sqrt` per-level filter |
| `frame.make_grids` half-pixel offset             | `frame.py:62-66`                                   | matches WSClean / CASA pixel centres |
| `VisibilityMatrix` weight rescaling              | `statistics.py:62-66`                              | preserves average visibility magnitude when applying `WEIGHT_SPECTRUM` |
| LOFAR conjugation                                | `apps/create_dataset.py:99`                        | `S = S.conj()` only for `lofar` MS |

## 16. Known caveats and conventions

- **Column-major everywhere.** All Python NumPy arrays passed into the
  C extension are required to be Fortran-contiguous. `pybipp.cpp:158`
  uses `py::array::f_style | py::array::forcecast` for inputs; outputs
  are likewise constructed as F-style. Helpers like
  `frame.reshape_uvw` explicitly call `np.array(..., order='F')`.
- **Autocorrelations are flagged** unconditionally inside
  `VisibilityMatrix.__init__` (`statistics.py:54-55`). There is no flag
  to disable that.
- **Beam-domain reduction.** Beams (one per station for LOFAR/MWA/SKA-low)
  with all-zero rows in `S` are dropped before LAPACK is called
  (`eigensolver.cpp:54-118`). The returned eigenvalue/eigenvector arrays
  are zero-padded back to `nBeam` so consumers see a stable shape.
- **HDF5 name restrictions.** Image-tag and selection-tag strings cannot
  contain `/`, `.`, or space (`image_synthesis.cpp:38-46` for synthesis,
  `image_data_file.cpp:33-43` for image_data, `apps/create_selection.py:102`
  replaces `.` with `_` to satisfy this).
- **Single-precision storage.** Dataset HDF5 always stores `float32`
  (eigenvectors interleaved as `2*nAntenna` floats). Even when the
  synthesis runs in `double` precision the dataset payload is upcast on
  read (`nufft_synthesis.cpp:44-71`).
- **MPI safety.** Without `BIPP_MPI`, `Communicator::world()` returns a
  trivial size-1 `Communicator` that never invokes `MPI_Init`
  (`include/bipp/communicator.hpp:48-50`). It is therefore safe to write
  Python code that always calls `bipp.communicator.world()` even in
  non-MPI builds — the example scripts rely on this.
- **No back-compatibility for the dataset format.** Files written by
  `formatVersionMajor != 1` will be rejected on open
  (`dataset_file.cpp:86-92`). Minor-version growth is tolerated only
  forward (newer code reading older files).
- **GPU path is partial.** v1.0.0 only compiles host code +
  `gpu/kernels/{add_vector,nuft_sum}.cu`. Higher-level GPU components
  (eigensolver, gram, virtual_vis, full nufft GPU) listed under `src/gpu/`
  are commented out in `src/CMakeLists.txt`. Users who need the GPU
  must rely on `neonufft`'s own GPU NUFFT engine, which BIPP wires in
  through the `gpu::NUFFT<T>` placeholder when the corresponding
  pybind dispatch is enabled.
- **No measurement-set writer.** BIPP only *reads* MS files (via
  python-casacore TaQL); it has no path to write back to MS. Imaging
  outputs are HDF5 only — convert to FITS using
  `bipp.imot_tools.io.s2image.Image.to_fits` if needed.
- **Sky model loaders.** Catalog ingestion is intentionally minimal:
  `source.from_tgss_catalog` is the only built-in catalog reader, with
  cached download to `~/.bipp/catalog/`. Custom sky models must be
  built as `SkyEmission(source_config)` from `[(SkyCoord, intensity), …]`.
- **Polarization.** Only Stokes-I imaging is supported. The MS reader
  averages parallel-hand correlations `[XX, YY]` (or `[RR, LL]`) into a
  single intensity channel (`measurement_set.py:277, 310`). XY/YX are
  dropped.

## 17. Provenance and credits

- BIPP descends from the **imot_tools** Python package by Sepand
  Kashani (EPFL), whose copyright/author headers are preserved in
  every Python module:
  `core.py`, `array.py`, `gram.py`, `instrument.py`, `beamforming.py`,
  `statistics.py`, `source.py`, `parameter_estimator.py`,
  `measurement_set.py`. The vendored math/IO library lives under
  `python/bipp/imot_tools/` (with its own `LICENSE.txt`, GPLv3-compatible).
- The C++/HPC implementation, the CLI and the file-format machinery are
  the EPFL Radio Astronomy Lab's contribution
  (`https://github.com/epfl-radio-astro/bipp`); see CI status badge in
  `README.md` and the doc badge for ReadTheDocs.
- The example notebooks/scripts (`lofar_bootes_nufft.py`, `test_writer.py`)
  are attributed to Matthieu Simeoni (the original author of the Bluebild
  Bootes-field demonstrations).

## 18. Quick reference: Python pipeline (programmatic equivalent of the CLI)

```python
import bipp, bipp.frame as frame, bipp.parameter_estimator as bb_pe
import bipp.filter, bipp.selection as sel
import bipp.measurement_set as ms_mod
import numpy as np, scipy.constants as const, astropy.units as u

# 1. dataset
ms = ms_mod.SKALowMeasurementSet("EOS_21cm-gf_202MHz_4h1d_200.MS")
N_ant, N_sta = 512, 512
ra, dec = ms.field_center.ra.deg, ms.field_center.dec.deg
with bipp.DatasetFile.create("skalow.h5", "skalow", N_ant, N_sta, ra, dec) as ds:
    for t, f, S in ms.visibilities(channel_id=range(0, 1),
                                   time_id=slice(0, -1, 1),
                                   column="DATA"):
        wl = const.speed_of_light / f.to_value(u.Hz)
        XYZ = ms.instrument(t)
        W   = ms.beamformer(XYZ, wl)
        UVW = ms.instrument.baselines(t, uvw=True, field_center=ms.field_center)
        v, d, scale = bipp.eigh_gram(wl, S.data, W.data, XYZ.data)
        ds.write(t.value, wl, scale, v, d, frame.reshape_uvw(UVW))

# 2. selection (5 lsq levels for [0, inf), 1 lsq level for negatives)
with bipp.DatasetFile.open("skalow.h5") as ds:
    eig = [ds.eig_val(i) for i in range(ds.num_samples())]
    out = {}
    for sel_str, name in [("lsq,5,0.95,log,0,inf", "pos"),
                          ("lsq,1,1.0,none,-inf,0", "neg")]:
        f, n, sigma, c, lo, hi = sel_str.split(",")
        intervals = bb_pe.infer_intervals(int(n), float(sigma), c, float(lo),
                                          float(hi), eig)
        for k, (a, b) in enumerate(intervals):
            fi = bipp.filter.Filter(f, a, b)
            tag = f"{name}_lvl{k}"
            out[tag] = {i: fi(np.array(d)) for i, d in enumerate(eig)}
    sel.export_selection(out, "selection.json")

# 3. image_prop
import astropy.coordinates as coord
with bipp.DatasetFile.open("skalow.h5") as ds:
    fc = coord.SkyCoord(ra=ds.ra_deg()*u.deg, dec=ds.dec_deg()*u.deg, frame="icrs")
lmn, _ = frame.make_grids(1024, np.deg2rad(10.2), fc)
bipp.ImagePropFile.create("image_prop.h5", 1024, 1024, 10.2, lmn.transpose())

# 4. synthesis
import json
with bipp.DatasetFile.open("skalow.h5") as ds, \
     bipp.ImagePropFile.open("image_prop.h5") as ip, \
     open("selection.json") as f:
    selection = json.load(f)
    selection = {t: {int(k): v for k, v in s.items()} for t, s in selection.items()}
    ctx = bipp.Context("auto", bipp.communicator.world())
    opt = bipp.NufftSynthesisOptions()
    opt.set_tolerance(1e-3); opt.set_precision("single")
    bipp.image_synthesis(ctx, opt, ds, selection, ip, "images.h5")

# 5. read out
with bipp.ImageDataFile.open("images.h5") as f:
    for tag in sorted(f.tags()):
        img = f.get(tag)  # (height, width) float32
```

That is exactly what `bipp dataset / selection / image_prop / synthesis /
plot` does, modulo plotting.
