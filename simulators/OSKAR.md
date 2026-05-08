# OSKAR — Exhaustive Reference

> **A GPU-accelerated simulator for radio interferometers (especially aperture-array telescopes such as the SKA-Low and LOFAR).**
> Repo source vendored at `simulators/OSKAR/`. Upstream: <https://github.com/OxfordSKA/OSKAR>.

This document is a deep, code-grounded reference distilled from the OSKAR
source tree, ReST documentation, ChangeLog, XML settings schemas, and the
Python bindings. It is intentionally exhaustive: section ordering moves from
identity → theory → data formats → applications → runtime architecture →
Python API → packaging → references.

---

## 1. Identity, scope and design goals

- **Project name.** OSKAR (Oxford SKA Radio-telescope simulator).
- **Version of the vendored snapshot.** `OSKAR-2.12.3-dev` (top-level
  `CMakeLists.txt`: `OSKAR_VERSION_ID 0x020C03`, MAJOR=2, MINOR=12, PATCH=3,
  SUFFIX="dev"). The most recent tagged release in `ChangeLog.txt` is
  **2.12.2 (2026-02-19)**; tip-of-tree work is for 2.12.3 (HDF5 beam-pattern
  output, SKA sky-model column names, quoted/named-column robustness, FEKO HDF5
  loading).
- **License.** 3-clause BSD (`LICENSE`). Most files carry "Copyright … The
  OSKAR Developers" or "The University of Oxford".
- **Maintainers.** SKAO simulation tools team (`ska-telescope/sim/oskar` on
  GitLab CI; mirrored to GitHub `OxfordSKA/OSKAR`).
- **Documentation.** `https://ska-telescope.gitlab.io/sim/oskar/` (built by the
  GitLab CI pipeline).
- **Citation.** Zenodo DOI `10.5281/zenodo.3758491`.
- **Languages.** C and C++ (most code), CUDA (`.cu`/`.cl` compute kernels),
  OpenCL (experimental), Python (bindings + helpers), Qt 5 (GUI), CMake
  (build), reStructuredText (docs).

**Why OSKAR exists.** Most visibility simulators target dish arrays and treat
the primary beam as a static, diagonal Jones matrix. OSKAR was written instead
for **aperture-array** instruments where the station beam is *itself* formed by
beamforming many antenna elements, where stations can be hierarchical
(stations → tiles → elements), and where each station's beam differs in time
because the array factor is direction-dependent. OSKAR therefore evaluates the
full Hamaker–Bregman–Sault RIME source-by-source on GPU, with the station beam
recomputed for every (time, frequency, source) sample.

**Concrete capabilities** (1-line each):

- Simulate visibilities at arbitrary (t, ν, baseline) sampling using the RIME.
- Aperture-array, isotropic, simple Gaussian, or VLA-(PBCOR) station beams.
- Hierarchical (station + tile) beamforming with per-element gain/phase/cable
  errors, apodisation, feed Euler angles, and arbitrary element layouts.
- Polarised simulation: scalar (Stokes I only) or full 4-pol (XX/XY/YX/YY).
- Numerical or analytic element patterns; spherical-wave fits (TE/TM, Ludwig-3
  reverted to θ/φ as of 2.11.0); FEKO HDF5 element coefficients; HARP station
  beams (mutual coupling).
- Optional **Z-Jones ionospheric phase screen** (FITS TEC cube) and Faraday
  rotation; **G-Jones** complex gains; cable-length errors → frequency-dep
  phase.
- Sky models: point sources, elliptical Gaussians, FITS images (orthographic
  projection), HEALPix RING-ordered FITS maps (Galactic or Equatorial),
  random power-law / broken power-law / grid / HEALPix generators, BBS/DP3/
  WSClean **named-column** files (since 2.12.0).
- Spectral models per source: logarithmic polynomial (default, up to 8 terms),
  linear polynomial (WSClean style), Callingham-style **spectral curvature**,
  **spectral line** (Gaussian profile), and rotation-measure-driven Q/U
  derivation.
- Built-in **GPU multi-device parallelism** with overlapping
  compute/write/finalise (visibility blocks, work-unit scheduling).
- CASA Measurement Set output (via `casacore`) with `PHASED_ARRAY` table, plus
  OSKAR's compact native binary visibility format.
- Built-in **imager**: FFT (with spheroidal/pillbox gridding), 2-D and 3-D DFT,
  W-projection. Natural / Radial / Uniform weighting, optional Gaussian
  taper, off-phase-centre imaging.
- A standalone **beam-pattern simulator** that produces FITS / HDF5 / text
  cubes of any station's voltage, amplitude, phase, or auto/cross-power
  response (per-station or telescope-cross-power).
- Auxiliary CLI tools: vis-add, vis-add-noise, vis-summary, vis-to-ms,
  fits-image-to-sky-model, binary-file-query, system-info.
- A Qt 5 GUI (`oskar`) for editing INI settings files and launching apps.
- A Python package (`oskarpy` → `import oskar`) wrapping the C library.

---

## 2. Repository layout

```
OSKAR/
├── CMakeLists.txt                 # top-level (CMake ≥ 3.18)
├── ChangeLog.txt                  # 2.0.0 (Apr 2012) → 2.12.3 (May 2026)
├── README.md, LICENSE, .gitlab-ci.yml, .clang-tidy, .readthedocs.yaml
├── apps/                          # CLI binaries (one main per executable)
│   ├── oskar_sim_interferometer_main.cpp
│   ├── oskar_sim_beam_pattern_main.cpp
│   ├── oskar_imager_main.cpp
│   ├── oskar_vis_add_main.cpp     oskar_vis_add_noise_main.cpp
│   ├── oskar_vis_summary_main.cpp oskar_vis_to_ms_main.cpp
│   ├── oskar_binary_file_query_main.cpp
│   ├── oskar_fits_image_to_sky_model_main.cpp
│   ├── oskar_filter_sky_model_clusters_main.cpp
│   ├── oskar_rebin_sky_main.cpp
│   ├── oskar_convert_ecef_to_enu_main.cpp
│   ├── oskar_convert_geodetic_to_ecef_main.cpp
│   ├── oskar_system_info_main.cpp
│   └── test/                      # CLI integration tests
├── apptainer/                     # Apptainer.python3 (Singularity/SIF recipe)
├── cmake/                         # build helpers (compiler options, packaging,
│                                  #   CUDA arch detection, version, etc.)
├── docker/                        # oskar-python3, oskar-ci-cuda-12-6
├── docs/                          # ReST docs source (Sphinx)
│   ├── index.rst   install/ example/
│   ├── sky_model/  telescope_model/  pointing_file/  binary_file/
│   ├── settings/   apps/   theory/   faq/   license.rst
│   ├── python/     ipython_notebooks/  rtd-docs/
│   └── conf.py
├── extern/                        # vendored 3rd-party
│   (gtest 1.7, rapidxml 1.13, cfitsio, Random123, OpenCL/CL,
│    lapack subset, lcov_cobertura, etc.)
├── gui/                           # Qt 5 settings GUI (`oskar` binary)
├── oskar/                         # the OSKAR C/C++/CUDA library
│   ├── apps/       (settings → object adapters; XML schemas under apps/xml/)
│   ├── beam_pattern/
│   ├── binary/     (OSKAR native binary file reader/writer)
│   ├── convert/    (~30 coordinate conversion kernels)
│   ├── correlate/  (auto/cross correlation kernels)
│   ├── gains/      (HDF5 gain table application)
│   ├── harp/       (HARP station-beam coefficient evaluation)
│   ├── imager/     (FFT / DFT / W-projection imager + grid kernels)
│   ├── interferometer/ (top-level simulator: jones K, R, E and chain;
│   │                   block scheduler, multi-GPU run loop)
│   ├── log/        (logging subsystem)
│   ├── math/       (DFT, Bessel, splines, vector types, math primitives)
│   ├── mem/        (oskar_Mem: typed, host/device-aware buffer abstraction)
│   ├── ms/         (CASA Measurement Set wrapper around casacore)
│   ├── settings/   (XML-driven SettingsTree, INI handler, types)
│   ├── sky/        (oskar_Sky model + loaders / generators / filters)
│   ├── telescope/  (oskar_Telescope and oskar_Station; loaders)
│   │   └── station/ + station/element/ (element pattern eval, weights, etc.)
│   ├── utility/    (timer, error strings, device, kernel macros, version)
│   ├── vis/        (oskar_VisHeader, oskar_VisBlock)
│   ├── oskar.h     (umbrella header)
│   └── oskar_global.h
└── python/
    ├── README.md, setup.py, setup.cfg, LICENSE.txt, pylintrc
    ├── 3rdparty/                  (vendored helpers)
    ├── examples/
    │   ├── corruptor.py           (subclass Interferometer & process_block)
    │   ├── fits_diff.py
    │   ├── image_fft_test.py
    │   ├── sim_image_via_files.py     (run sim, then run imager from MS)
    │   ├── sim_image_via_memory.py    (ImagingInterferometer in-memory)
    │   ├── sim_mpi_multi_channel.py   (mpi4py + multi-channel sweeping)
    │   └── spead/                     (SPEAD-stream demo)
    └── oskar/
        ├── __init__.py             (re-exports public classes)
        ├── _version.py
        ├── barrier.py              (threading.Barrier helper)
        ├── bda.py                  (Baseline-dependent averaging utilities)
        ├── binary.py               (OSKAR binary file Python wrapper)
        ├── imager.py               (Imager class)
        ├── imaging_interferometer.py  (Interferometer subclass that
        │                                images each block on-the-fly)
        ├── interferometer.py       (Interferometer class)
        ├── measurement_set.py      (MeasurementSet wrapper)
        ├── settings_tree.py        (SettingsTree: app-aware INI editor)
        ├── sky.py                  (Sky model class)
        ├── telescope.py            (Telescope class)
        ├── utils.py
        ├── vis_block.py            (VisBlock data accessor)
        ├── vis_header.py           (VisHeader meta-data accessor)
        └── src/                    (C glue → Python C-API extensions:
                                     _interferometer_lib, _sky_lib,
                                     _telescope_lib, _imager_lib,
                                     _measurement_set_lib, _binary_lib,
                                     _settings_tree_lib, _vis_block_lib,
                                     _vis_header_lib, _bda_utils)
```

The library follows a **strict C ABI per module**: each public function lives
in its own `oskar_*.c`/`.cu`/`.cl` file, headers expose only the function
pointer surface, and private structs sit in `private_*.h` headers. CUDA
kernels are written as templated C macros and instantiated for `float`/`double`
via `#define Real float` / `Real4c float4c` blocks (see
`oskar_interferometer.cu`).

---

## 3. Build system & dependencies

### 3.1 CMake configuration

Top-level `CMakeLists.txt` is short and does the following (minimum
`cmake_minimum_required(VERSION 3.18)`):

1. Set deployment target on macOS to 10.13.
2. Project version `2.12.3-dev` (encoded as `0x020C03`).
3. Search for **CUDA Toolkit ≥ 7.0** (`-DFIND_CUDA=OFF` to skip).
4. Optionally search for **OpenCL** (`-DFIND_OPENCL=ON`); on Windows, an
   import lib is generated from `extern/CL/OpenCL.def`.
5. `find_package(OpenMP QUIET)` → enables CPU OpenMP correlate paths.
6. `find_package(HDF5 QUIET)` → required for HARP, FEKO HDF5 element data,
   external gain tables, HDF5 beam-pattern output, FITS TEC ionosphere.
7. `find_package(Threads REQUIRED)`.
8. `find_package(HARP QUIET)` → optional HARP electromagnetic beam library.
9. `find_package(ska-sdp-func QUIET)` → optional SDP processing-function
   library (gridder, FFT, etc., contributed by SKAO).
10. Compile `extern/`, `oskar/`, `apps/`, `gui/`, `docs/`, then packaging.

### 3.2 Required vs optional dependencies

| Component | Default | Purpose | Without it |
|---|---|---|---|
| CMake ≥ 3.1 | required | build | cannot build |
| C/C++ compiler | required | build | cannot build |
| CUDA Toolkit ≥ 7.0 | optional, ON | GPU kernels | CPU-only fallback |
| OpenCL | optional, OFF | GPU kernels | GPU disabled (CUDA still works if present) |
| Qt 5 | optional | `oskar` GUI | only command-line apps |
| casacore ≥ 2.0 | optional | Measurement Set I/O | only OSKAR `.vis` files |
| HDF5 ≥ 1.10 | optional | gain tables, HARP, FEKO, BP HDF5, TEC | features disabled |
| OpenMP | optional | CPU correlate parallelism | serial CPU |
| HARP beam library | optional | mutual-coupling SHD beams | use embedded element pattern |
| ska-sdp-func | optional | SDP gridder/FFT shared with rascil | use OSKAR's own |

### 3.3 Build flags (most useful)

- `-DCUDA_ARCH="<arch>"` — CUDA SM target list (default ALL ≈ 3.5..7.5).
  Multiple values: `"ALL;8.0"`. Documented values include 2.0, 2.1, 3.0, 3.2,
  3.5, 3.7, 5.0–5.2, 6.0–6.2, 7.0, 7.5, 8.0, 8.6, 8.7.
- `-DCMAKE_INSTALL_PREFIX=<path>` (default `/usr/local`).
- `-DCASACORE_LIB_DIR=<path>` and `-DCASACORE_INC_DIR=<path>`.
- `-DCMAKE_PREFIX_PATH=<path>` to point at a non-system Qt 5
  (e.g. `/usr/local/opt/qt5/` on Homebrew).
- `-DFIND_CUDA=ON|OFF`, `-DFIND_OPENCL=ON|OFF` (OpenCL is experimental).
- `-DNVCC_COMPILER_BINDIR=<path>` (force nvcc → host compiler combination,
  e.g. when XCode and CUDA disagree).
- `-DFORCE_LIBSTDC++=ON` to force libstdc++ with Clang.
- `-DCMAKE_BUILD_TYPE=Release|Debug` (defaults to Debug only if the build
  directory name contains "dbg" or "debug").
- `-DBUILD_INFO=ON` for verbose CMake diagnostics.
- `-DBUILD_TESTING=OFF` and `-DCOVERAGE_REPORT=ON` for coverage builds (uses
  `lcov_cobertura.py` + fastcov; coverage excludes `apps/`, `extern/`, `gui/`,
  `lapack_subset.c`, `to_sdp_mem.cpp`, all `test/` dirs).

### 3.4 Build commands

```bash
mkdir build && cd build
cmake [options] ../top/level/source/folder
make -j8
make install        # installs into prefix (default /usr/local)
ctest               # run unit tests after build
```

Installed layout (Linux defaults):

- Binaries: `/usr/local/bin/oskar*`
- Libraries: `/usr/local/lib/liboskar*`
- Headers: `/usr/local/include/oskar/`

### 3.5 Pre-built distributions

- **macOS DMG** (drag `OSKAR.app` to `/Applications`; double-click once to
  symlink the CLI tools into `/usr/local/bin`).
- **Windows installer** (.exe) — choose "Add OSKAR to the PATH" and install
  the optional headers/libs to enable the Python interface build.
- **Singularity/Apptainer SIF** (`OSKAR-Python3.sif`) — Linux. Run with
  `singularity exec --nv ./OSKAR-Python3.sif oskar_sim_interferometer …`.
- **Docker** image `artefact.skao.int/oskar-python3` (also a CUDA-12.6 CI
  image at `oskar-ci-cuda-12-6`).
- **Kubernetes** helper script `oskar_run_k8s` (in
  `docs/python/oskar_run_k8s`).

---

## 4. Theory of operation (RIME)

### 4.1 The Measurement Equation

OSKAR uses Hamaker, Bregman & Sault (1996) — see also Smirnov (2011) — and
implements the RIME for a baseline between stations p and q as:

```
V_pq = G_p · ( Σ_s K_{p,s} Z_{p,s} E_{p,s} ⟨B_s⟩ E_{q,s}^H Z_{q,s}^H K_{q,s}^H ) · G_q^H
```

Each Jones term is a 2×2 complex matrix:

| Symbol | Meaning | DD/DI | Source |
|---|---|---|---|
| **B** | Source brightness/coherency | per-source | sky model |
| **E** | Station beam (parallactic-rotated, including the array factor and per-element response) | direction-dependent | telescope model |
| **Z** | Ionospheric phase screen (∆TEC → phase) | direction-dependent | external FITS TEC + TID file |
| **K** | Geometric/interferometer phase | direction-dependent | u, v, w + l, m, n |
| **G** | Direction-independent station gains (per pol) | direction-independent | HDF5 `gain_model.h5` |
| **R** (parallactic angle) | Rotation from equatorial Stokes to antenna frame | per-source | telescope.lat + source ha,δ |

`R` is computed inside `oskar_evaluate_jones_R.{h,c,cu,cl}` and joined into
`E` via `oskar_jones_join` before chain evaluation. Cable-length errors per
pol are applied separately as a frequency-dependent phase factor on the joined
Jones matrix (`oskar_jones_apply_cable_length_errors`).

### 4.2 Coherency / brightness matrix

Following IAU 1974 polarisation convention (RA increases east, Q+ N-S, U+
NE-SW, V+ right-hand circular):

```
⟨B⟩ = | I+Q   U+iV |
      | U-iV  I-Q  |
```

(Note: this differs by a 1/2 factor from the convention used by some other
packages; OSKAR's choice means **V_XX + V_YY = 2I**, i.e. when converting
back to Stokes the imager applies `I = ½(XX+YY)`, etc., as documented in
`oskar_image.xml` and the imager code.)

Stokes recovery from the linear correlation matrix:

```
I = ½(XX+YY)   Q = ½(XX-YY)   U = ½(XY+YX)   V = -½ i (XY-YX)
```

### 4.3 Coordinate systems

- **Equatorial** `(x', y', z')`: x'→α=0, z'→NCP, RA grows from x' towards y'.
- **Local horizon (ENU)** `(x, y, z)`: x→East, y→North, z→Zenith. ϕ is the
  co-azimuth (E from N), θ the polar angle (zenith distance).
- **(u, v, w)**: standard radio convention, u→E, v→N, w→phase centre.
- **(l, m, n)**: direction cosines towards source, l→E, m→N, n→phase centre.

### 4.4 Parallactic angle

```
ψ_p = arctan( cos(φ) sin(H) / ( sin(φ) cos(δ) − cos(φ) sin(δ) cos(H) ) )
```

with observer latitude φ, source hour angle H, declination δ. The R-Jones
applies a 2-D rotation matrix `[[cos ψ, -sin ψ], [sin ψ, cos ψ]]` between
equatorial Stokes and the antenna frame.

### 4.5 Ionospheric Z-Jones

Values in the screen are interpreted as ∆TEC (TECU) above the array. Phase
applied to a source at frequency ν (in **Hz**, not GHz — bug fixed in 2.12.0):

```
Z = exp[ i · ( -8.44797245 × 10^9 · ∆TEC / ν ) ] · I
```

OSKAR also evaluates **ionospheric Faraday rotation** using IGRF magnetic
field at the station altitude (in km — fixed in 2.12.0).

Ionosphere settings (from `oskar_telescope_model.xml`):

- `ionosphere_screen_type`: `None` or `External`.
- `external_tec_screen.input_fits_file`: ARatmospy-style TEC FITS cube.
- `screen_height_km` (default 300).
- `screen_pixel_size_m` and `screen_time_interval_sec` (use `file` to read
  from FITS header, or specify explicitly; since 2.10.1 the time interval can
  differ from the visibility integration).
- `isoplanatic_screen`: if true, single value per time used for all sources.

There is also an **older 2-D TID model** available via the `ionosphere`
settings group (`oskar_ionosphere.xml`) for the `oskar_sim_tec_screen`
auxiliary tool: `TEC0` baseline TEC, `TID_file(s)`, and the per-station TEC
image generator.

### 4.6 Interferometer K-Jones

```
K = exp[ -2π i (ul + vm + w(n − 1)) ] · I
```

The sign convention can be flipped by `interferometer/use_casa_phase_convention`
(default true since 2.10.0 — this is the **CASA/VLA convention**). Older
OSKAR data used the opposite sign. Header tag 13 in the binary file
records this choice.

### 4.7 Station beam (E-Jones)

Two dipoles X, Y nominally aligned with the local x and y axes. The station
beam is an array-factor-weighted sum of per-element responses:

```
E = ( Σ_a [ w^X_a w^Y_a ]^T  ·  [[ g^X_X  g^X_Y ],
                                  [ g^Y_X  g^Y_Y ]]_a )  ·  R(ψ_p)
```

The geometric beamforming weight per direction (θ_b, φ_b), antenna position
(x, y, z), time t, including per-antenna systematic and time-variable gain
(G_0, G_std) and phase (φ_0, φ_std) errors:

```
W = W_geometric · (G_0 + G_err) · exp[ i (φ_0 + φ_err) ]
```

with G_err, φ_err drawn per time step from N(0, G_std), N(0, φ_std).

### 4.8 Noise

Optional uncorrelated Gaussian noise added per (baseline, time, channel,
polarisation) with zero-mean RMS specified in **Jy per polarisation** of an
unpolarised source. The standard derivations from Thompson, Moran & Swenson
and Wrobel & Walker apply:

```
σ_{p,q} = √( S_p S_q / (2 Δν τ_acc) )
       = k_B √( 2 T_p T_q / (A_p A_q η_p η_q Δν τ_acc) )
σ_im   = σ_{p,q} / √n_d        with  n_d = n_b · τ_0/τ_acc
```

ε_{p,q} = √2 · σ_{p,q} (real+imag combined). RMS values per station and per
frequency are stored in telescope-model `rms.txt` and `noise_frequencies.txt`
files (or specified by range / data file in settings).

---

## 5. Sky model

The sky model is a flat **table of point or elliptical-Gaussian sources** with
optional polarisation, spectral and rotation-measure metadata. It can be
loaded from a text file, generated programmatically, derived from a
HEALPix RING FITS map, or populated from a `casacore`-compatible orthographic
FITS image. Sky models are observation-independent — the user must set the
phase centre separately.

### 5.1 Named-column format (since OSKAR 2.12.0)

Compatible with **LOFAR/BBS/DP3/WSClean** `makesourcedb` files. A single
header line describes the columns; rows can be space- or comma-separated;
`#` introduces a comment. The format string can appear anywhere on the line
with `Format` (case-insensitive) and an `=` token, fields optionally in
brackets:

```
Format = RA, Dec, I
# format = RA Dec I
Format= (Ra, Dec, I, Q, U, V)
# (RA,Dec,I,Q,U) = format
```

Each column may take a default value via `Name='value'` (no spaces around
`=`). For example:

```
Format = RaD, DecD, I, ReferenceFrequency='143e6', MajorAxis, MinorAxis
```

Recognised columns (case-insensitive; aliases allowed):

| Name | Aliases | Unit | Description |
|---|---|---|---|
| `Ra` | — | rad (default) / deg / sex | Right Ascension |
| `Dec` | — | rad (default) / deg / sex | Declination |
| `RaD` | `ra_deg` | deg (default) / rad / sex | Right Ascension |
| `DecD` | `dec_deg` | deg (default) / rad / sex | Declination |
| `I` | `StokesI`, `i_pol` | Jy | Stokes I at ref freq |
| `Q`/`U`/`V` | `StokesQ`,…, `q_pol`,… | Jy | Stokes Q/U/V |
| `ReferenceFrequency` | `ref_freq` | Hz | reference ν₀ |
| `SpectralIndex` | `spec_idx` | — | scalar or `[α₀,α₁,…]` (≤ 8 terms) |
| `LogarithmicSI` | `log_spec_idx` | bool | true → log poly, false → linear |
| `MajorAxis` | `major_ax` | arcsec | Gaussian FWHM major |
| `MinorAxis` | `minor_ax` | arcsec | Gaussian FWHM minor |
| `Orientation` | `PositionAngle`, `pos_ang` | deg | PA of major (E of N) |
| `RotationMeasure` | `rot_meas` | rad m⁻² | Faraday rot |
| `PolarizationAngle` | `PolarisationAngle`, `pol_ang` | deg | used with RM and PolarizedFraction |
| `PolarizedFraction` | `PolarisedFraction`, `pol_frac` | — | linear pol fraction |
| `ReferenceWavelength` | — | m | optional; computed from ref freq otherwise |
| `SpectralCurvature` | — | — | Callingham (2017) curvature q |
| `LineWidth` | — | Hz | Gaussian-line σ; if > 0 source is a spectral line |

`Name` and `Type` columns from LOFAR are silently ignored (the parser
accepts them but they are unused). If `Name` contains spaces or commas, it
must be quoted (since 2.12.3).

**Source kind precedence** when multiple spectral columns are present:
`LineWidth > 0` → spectral-line; else `SpectralCurvature ≠ 0` → curvature
model (using only the first `α₀`); else log/linear polynomial.

If `ReferenceFrequency` is omitted or zero, the source flux is *constant in
frequency*; spectral index and rotation measure are silently inert.

Coordinate columns may be suffixed `rad` or `deg` to override the default
unit interpretation.

If `RotationMeasure` is set, OSKAR follows the BBS-cookbook recipe: Q and U
at the simulation frequency are derived from `PolarizationAngle`,
`PolarizedFraction` and `ReferenceWavelength`.

### 5.2 Fixed-format (legacy)

Plain whitespace/comma-separated columns in fixed order, all but the first
three optional, defaults are zero:

| # | Field | Unit | Notes |
|---|---|---|---|
| 1 | RA | deg | required (interpreted as **apparent** RA) |
| 2 | Dec | deg | required (interpreted as **apparent** Dec) |
| 3 | Stokes I | Jy | required |
| 4–6 | Q, U, V | Jy | optional |
| 7 | Reference frequency | Hz | optional |
| 8 | Spectral index | — | optional (single scalar) |
| 9 | Rotation measure | rad m⁻² | optional (added in 2.3.0) |
| 10 | Major axis FWHM | arcsec | optional |
| 11 | Minor axis FWHM | arcsec | optional |
| 12 | Position angle | deg | optional |

The parser accepts 3–9, 11, or 12 columns; 10 or 13+ is an error. With 11
columns, RM defaults to 0 (back-compat with OSKAR < 2.3.0).

### 5.3 Spectral profiles

**Logarithmic polynomial** (default — and the default if `LogarithmicSI`
is omitted):

```
S(ν) = S₀ (ν/ν₀)^( α₀ + α₁ log₁₀(ν/ν₀) + α₂ log₁₀(ν/ν₀)² + … )
```

**Linear polynomial** (WSClean-compatible):

```
S(ν) = S₀ + α₀ (ν/ν₀ − 1) + α₁ (ν/ν₀ − 1)² + α₂ (ν/ν₀ − 1)³ + …
```

**Spectral curvature** (Callingham et al. 2017 eq. 2):

```
S(ν) = S₀ (ν/ν₀)^α₀ · exp( q · ln(ν/ν₀)² )
```

**Spectral line**: Gaussian profile centred on ν₀ with σ = `LineWidth`:

```
S(ν) = S₀ · exp( −(ν − ν₀)² / (2 σ²) )
```

### 5.4 Gaussian sources

Position-angle θ E of N, FWHM_x major, FWHM_y minor → σ = FWHM/(2√(2 ln 2)).
The on-sky 2-D Gaussian f(x,y) = exp(−(ax² + 2bxy + cy²)) has a, b, c written
in `docs/sky_model/sky_model.rst`. Implementation: multiply each baseline's
amplitude by the Fourier-plane Gaussian (σ_uv = 1/(2π σ_sky)). The transform
is computed in `oskar_sky_evaluate_gaussian_source_parameters.{h,c}`. Sources
where the Gaussian fit fails (very far from phase centre) can be fed to OSKAR
either as point sources (default) or zeroed (`sky/advanced/zero_failed_gaussians`).

### 5.5 Generators (settings-driven)

- **Random uniform power-law** in flux: `sky/generator/random_power_law` —
  N sources scattered uniformly on the sphere with N(F)dF ∝ F^p between
  `flux_min` and `flux_max`.
- **Random broken power-law**: `sky/generator/random_broken_power_law` with
  two indices p1, p2 and a threshold flux.
- **Grid** at phase centre: `sky/generator/grid` — N×N grid over `fov_deg`,
  Stokes I drawn from `N(mean_flux, std_flux)`.
- **HEALPix all-sky**: `sky/generator/healpix` — uniform `Nside`-pixelated
  grid (12·Nside² points), constant amplitude per pixel.
- **OSKAR sky-model file(s)**: `sky/oskar_sky_model.file` (one or more text
  files; can be combined with generators).
- **FITS image**: `sky/fits_image.file` — converts an orthographic FITS
  image (one frequency at a time) using `oskar_fits_image_to_sky_model`-style
  logic; supports `Jy/beam`, `Jy/pixel`, `K`, `mK` units; spectral index and
  min-peak-fraction filters.
- **HEALPix FITS**: `sky/healpix_fits.file` — RING-ordered HEALPix maps
  (NEST not supported); units one of `Jy/pixel`, `K`, `mK`; coord system
  `Galactic` (default) or `Equatorial`; converts brightness temperature → Jy
  at `freq_hz`.

### 5.6 Filters and overrides

- **Per-source flux filter**: `sky/.../filter.flux_min/max` (in Jy).
- **Per-source radial filter**: `radius_inner_deg`, `radius_outer_deg` from
  the phase centre.
- **Common per-channel flux filter**: `sky/common_flux_filter` — applied
  *after* spectral scaling, so the same flux range can be enforced at every
  frequency.
- **Spectral-index override**: `sky/spectral_index.override` — replace all
  source SI values with `N(mean, std_dev)` at a given `ref_frequency_hz`.
- **Extended-source overrides**: `sky/.../extended_sources.FWHM_major,
  FWHM_minor, position_angle` — globally re-shape sources in a group.
- **Apply horizon clip**: `sky/advanced.apply_horizon_clip` — remove sources
  below the horizon at every station per time step (default true; disable for
  small fields known to be always up).

### 5.7 Output sky-model file

`sky/output_text_file` saves the *final* sky model — useful for debugging.
Options: `use_named_columns` (BBS/DP3/WSClean format), `use_degrees`
(`RaD`/`DecD` → deg columns vs. radian-suffixed), `write_name`,
`write_type` (POINT vs GAUSSIAN).

---

## 6. Telescope model

A telescope model is a **directory tree**:

```
my_telescope_model/
├── position.txt         (telescope reference long, lat, alt)
├── layout[_ecef|_wgs84].txt   (station positions)
├── station_type_map.txt       (optional)
└── stationNNN/
    ├── layout.txt             (element ENU positions per station)
    ├── element_types.txt
    ├── gain_phase.txt
    ├── cable_length_error.txt
    ├── apodisation.txt        (or apodization.txt)
    ├── feed_angle{,_x,_y}.txt
    ├── permitted_beams.txt
    ├── element_pattern_spherical_wave_*_<MHz>.txt
    ├── *HARP*.h5
    ├── gain_model.h5
    └── tileNNN/    (optional: hierarchical / two-level beamforming)
        ├── layout.txt
        └── …
```

Station directories are sorted **alphabetically**; pad with leading zeros
(`station001`, `station002`, …). If only one station folder is present, all
stations are taken to be identical to it. Otherwise either the number of
folders must equal the number of stations in `layout.txt`, or a
`station_type_map.txt` (one zero-based integer per row) maps each station
to its station-type folder.

### 6.1 Special files at the top level

- **`position.txt`** — required. WGS84 longitude (deg), latitude (deg),
  altitude (m, optional, default 0).
- **`layout.txt`** — ENU station positions (m): x→E, y→N, z→Up. Up to 6
  columns per row: x, y, z, x_err, y_err, z_err. The first three are the
  measured position; the optional errors are added to give the *true*
  position (used for actual phase calculation; the measured position drives
  beamforming).
- **`layout_ecef.txt`** — alternative top-level layout in Earth-centred
  Earth-fixed coordinates (m): x→(0,0), y→east, z→NCP. Up to 6 cols (3 +
  3 errors).
- **`layout_wgs84.txt`** — alternative top-level layout: WGS84 lon (deg),
  lat (deg), altitude (m). Note: with WGS84 layouts there was a bug in
  pre-2.11 where absolute coords in MS output were wrong (fixed in 2.11.0).
- **`station_type_map.txt`** — single column of zero-based station-type
  indices (one row per station), mapping each station in the array to a
  station-type folder.

### 6.2 Station-level files

- **`layout.txt`** — element ENU positions (m). No ECEF/WGS84 alternatives at
  the station level.
- **`element_types.txt`** — zero-based integer type index per element (used
  with multiple element-pattern files in the same station).
- **`gain_phase.txt`** — per-element gain G₀, phase φ₀ (deg), and time-variable
  std deviations G_std, φ_std (deg). Combined with geometric beamforming
  weights (see §4.7).
- **`cable_length_error.txt`** — per-element cable-length error (m), applied
  as a frequency-dependent phase factor.
- **`apodisation.txt` / `apodization.txt`** — complex multiplicative
  beamforming weights `Re Im` per element (default 1+0i).
- **`feed_angle.txt` / `feed_angle_x.txt` / `feed_angle_y.txt`** — Euler
  angles (α, β, γ in deg, all default 0) of the X/Y feeds. If only the
  un-suffixed form is present, both pols use the same data. Since 2.10.1
  a single-line file is allowed and broadcast to all elements.
- **`permitted_beams.txt`** — list of allowed (azimuth, elevation) pairs
  in deg; OSKAR snaps the computed phase centre to the nearest permitted
  direction at each time step.

### 6.3 HDF5 gain model

`gain_model.h5` may live in any directory; the dimensions of the antenna or
station axis must match the parent layout. Three datasets:

- `freq (Hz)` — 1-D array of channel frequencies in Hz.
- `gain_xpol` — 3-D `(time, channel, antenna_or_station)` complex (compound
  real+imag float) for the X polarisation.
- `gain_ypol` — same for Y (optional in scalar mode or if X = Y).

Time index aligns with snapshot index; channel is selected by nearest
frequency. Setting either dimension to size 1 means "constant in that
dimension". Applied via `oskar_jones_apply_station_gains` after Jones-chain
join.

### 6.4 HARP station-beam coefficients

When OSKAR is built with the HARP library, a station folder containing
HDF5 files whose name contains `HARP` is loaded as exported HARP coefficient
data. The trailing number in the filename is interpreted as the frequency
in MHz (`HARP_100.h5` ≡ `data_HARP_SKALA4_rand256_100MHz.h5`). Each file
must contain attributes `freq` (Hz), `num_ant`, `num_mbf`, `max_order`, and
datasets `alpha_te`, `alpha_tm`, `coeffs_polX`, `coeffs_polY`. HARP is
faster than per-element evaluation when mutual coupling matters; only the
nearest frequency file is used.

### 6.5 Spherical-wave element patterns

Files named:

```
element_pattern_spherical_wave_[x|y]_[te|tm]_[re|im]_<typeIdx>_<MHz>.txt
```

contain numerical coefficients of a spherical-wave decomposition of the
embedded element pattern. Element type index is 0 unless `element_types.txt`
introduces multiple types. Each file contains one row per θ-order ℓ
starting at ℓ=1, with 2ℓ_max+1 entries per row (only the first 2ℓ+1 are
used; trailing zeros pad rows for ℓ < ℓ_max). X/Y suffixes are optional.
Since 2.12.0 the **maximum order** can be capped via
`telescope/aperture_array/element_pattern.max_order`.
**FEKO HDF5** files (since 2.12.0) load coefficients directly when "FEKO"
appears anywhere in the filename.

### 6.6 Functional element patterns (built-in)

Set `telescope/aperture_array/element_pattern.functional_type` to:

- **`Dipole`** with `dipole_length` (default 0.5 wavelengths;
  `dipole_length_units` = Wavelengths or Metres). The polarised
  half-wave-dipole response (Ludwig-3 was reverted to θ/φ in 2.11.0).
- **`Isotropic (unpolarised)`** for Stokes-I-only sanity tests.

Optional taper applied on top:

- `taper.type = None` (default), `Cosine`, or `Gaussian`.
- For `Cosine`: `cosine_power` (default 1.0) → `cos^p θ`.
- For `Gaussian`: `gaussian_fwhm_deg` (default 45°) at the reference
  frequency.

`element_pattern.normalise` (default false) divides by the zenith amplitude.
`enable_numerical` (default true) prefers numerical files (spherical-wave or
FEKO HDF5) over functional fallbacks. `swap_xy` is a debug flag for
mis-ordered numerical files.

### 6.7 Array pattern (beamforming) settings

`telescope/aperture_array/array_pattern`:

- `enable` (default true): if false, the array factor is ignored.
- `normalise` (default false): divide each station beam by the number of
  antennas in the station.
- `element` overrides — per-pol element gain, phase, cable-length, position
  std. devs and orientation std. devs (with separate seeds for x/y position
  errors, time-variable errors, gain errors per pol, phase errors per pol,
  cable-length errors per pol, orientation errors per pol). When set, these
  override the corresponding per-station files.

### 6.8 Top-level station-type vs Gaussian/Isotropic/VLA

`telescope.station_type` determines the E-Jones model **at the station
level**:

- **`Aperture array`** — full array-factor + element-pattern computation
  (see `aperture_array` group).
- **`Gaussian beam`** — replace E with a circular or elliptical Gaussian
  centred on the per-station beam direction, parameterised at a reference
  frequency. Settings group `gaussian_beam`: `elliptical` (false →
  circular `fwhm_deg`; true → per-pol `x_fwhm_major_deg`,
  `x_fwhm_minor_deg`, `x_angle_deg` and Y-counterparts), `ref_freq_hz`. Since
  2.9.0 this can be polarised.
- **`Isotropic beam`** — disable all station-beam effects (E = I).
- **`VLA (PBCOR)`** — use the standard EVLA / VLA primary-beam correction
  polynomial (`oskar_evaluate_vla_beam_pbcor.h`).

Other top-level telescope settings:

- `normalise_beams_at_phase_centre` (default true): scale each station beam so
  that its amplitude is 1.0 at the phase centre at every snapshot — performs
  an effective amplitude calibration on a source at the phase centre.
- `allow_station_beam_duplication` (default false): use the station-type map
  to duplicate beams, saving a lot of compute when there are few distinct
  station types — but with long baselines this **breaks the source-position
  shift relative to each station's horizon**, so use carefully.
- `pol_mode`: `Full` (default; XX/XY/YX/YY) or `Scalar` (only Stokes I,
  much faster).
- `ionosphere_screen_type` and `external_tec_screen` (see §4.5).
- `isoplanatic_screen` — apply the same TEC value to all sources at a
  station-time.

### 6.9 Pointing file

Optional. Plain-text, one line per (sub-)station phase-centre override.
Columns: indices of station, sub-station, sub-sub-station, …, then a
coordinate frame string `AZEL` or `RADEC`, then longitude (deg), then
latitude (deg). Wildcard `*` in any index column matches all children.
Order matters — later lines override earlier. An entry recursively overrides
the beam direction for **all child stations**.

```
*    RADEC 45.0 60.0
3    RADEC 45.1 59.9
* *  AZEL  60.0 75.0
0 *  AZEL  60.1 75.0
2 6  AZEL  0.0  90.0
```

If absent, every station beam points at the global `observation/phase_centre`.

### 6.10 Noise files

Two text files (`#`-comments, blank lines ignored):

- `noise_frequencies.txt` (top-level only): list of frequencies (Hz) for
  which noise values are defined.
- `rms.txt` (top-level or top-level station folder): list of RMS Jy values
  per frequency.

Settings group `interferometer/noise`: `enable`, `seed`, `freq` (`Telescope
model` / `Observation settings` / `Data file` / `Range`), `rms` (`Telescope
model` / `Data file` / `Range`).

---

## 7. Application binaries

OSKAR ships **11 standard CLI applications**. All accept `--help` and
`--version`. Settings-file applications additionally accept `--get key`
(read a value), `--set key value` (write a value or unset to default),
and `-q` / `--quiet` (suppress most logs).

| Binary | Role | Settings root |
|---|---|---|
| `oskar` | Qt 5 GUI launcher — picks an app, edits its INI file, runs it | (auto) |
| `oskar_sim_interferometer` | Run an interferometer simulation | `oskar_sim_interferometer` (imports simulator + sky + observation + telescope + interferometer XML) |
| `oskar_sim_beam_pattern` | Generate beam-pattern image cubes | `oskar_sim_beam_pattern` (imports simulator + observation + telescope + beam_pattern XML) |
| `oskar_imager` | Image visibility data → FITS | `oskar_imager` (imports image XML) |
| `oskar_vis_add` | Concatenate / sum compatible OSKAR `.vis` files | — |
| `oskar_vis_add_noise` | Add noise to existing `.vis` file(s) | uses interferometer/noise group |
| `oskar_vis_summary` | Print metadata, settings, log of a `.vis` file | — |
| `oskar_vis_to_ms` | Convert one or more `.vis` → CASA MS | — |
| `oskar_binary_file_query` | Dump the chunk index of an OSKAR binary file | — |
| `oskar_fits_image_to_sky_model` | Convert orthographic FITS → sky model file | — |
| `oskar_system_info` | Show installed GPU details | — |

There are also a handful of **utility binaries** that are not always
documented in `apps.rst`:

- `oskar_filter_sky_model_clusters` — filter sky-model files by cluster.
- `oskar_rebin_sky` — rebin sky-model fluxes onto a different grid.
- `oskar_convert_ecef_to_enu` — coordinate-conversion helper.
- `oskar_convert_geodetic_to_ecef` — coordinate-conversion helper.

### 7.1 Settings GUI (`oskar`)

Qt 5 application that reads an XML schema (per app) embedded into the binary
at compile time, displays a settings tree, validates types and inter-setting
dependencies, and writes/reads INI files. The XML schema lives in
`oskar/apps/xml/` (see §8). Settings files are **per-app** — sharing one
between apps is unsupported because unknown keys are stripped.

### 7.2 `oskar_sim_interferometer` flow

`apps/oskar_sim_interferometer_main.cpp` walks through:

1. Parse argv via `OptionParser`; honour `--get`/`--set`/`-q`.
2. Load the INI file → `SettingsTree`.
3. Build the interferometer object: `oskar_settings_to_interferometer`.
4. Build the sky model: `oskar_settings_to_sky` (loads files, applies
   filters/overrides, generates random/grid sources, assembles point and
   Gaussian sources).
5. Build the telescope model: `oskar_settings_to_telescope` (loads layout,
   element/feed/cable/HARP/HDF5 gain data, applies overrides).
6. Run: `oskar_interferometer_run`.

The runner parallelises across GPUs; the simulation is split into
**visibility blocks** (one or more output rows in time × channel) and
**source chunks** (`max_sources_per_chunk`, default 16384). For N GPUs the
runner spawns N+1 threads — N compute threads and 1 finalisation thread —
synchronised by an internal barrier (Python `oskar.barrier.Barrier`). Per
`oskar_interferometer_run_block.c`:

1. Compute thread loops over (time × source-chunk × channel) tuples,
   ordered as `i_work_unit / num_times → chunk`, `… % num_times → time`.
2. Copy the next sky chunk to GPU only when it changes.
3. Apply horizon clipping (`oskar_sky_horizon_clip`) using GAST(MJD).
4. For each channel in the block:
   1. Scale source fluxes by spectral index/curvature/line/RM for the
      channel frequency (`oskar_sky_scale_flux_with_frequency`).
   2. Compute station UVW (`oskar_telescope_uvw`).
   3. Compute relative direction cosines for AZEL or read precomputed l, m, n.
   4. Resize per-source Jones matrices (`R`, `J`, `E`, `K`).
   5. Evaluate **E** (`oskar_evaluate_jones_E`) — array factor + element
      pattern + tropospheric/Z if enabled.
   6. Evaluate **R** (`oskar_evaluate_jones_R`) — parallactic; join into E.
   7. Evaluate **K** (`oskar_evaluate_jones_K`) — interferometer phase.
   8. Join chain `J = K · R · E` (`oskar_jones_join`).
   9. If gains: evaluate them and apply (`oskar_jones_apply_station_gains`).
   10. Apply per-pol cable-length errors (`oskar_jones_apply_cable_length_errors`).
   11. Auto-correlate (`oskar_auto_correlate`) and/or cross-correlate
       (`oskar_cross_correlate`) into the output `oskar_VisBlock`. The
       cross-correlation kernel handles point and Gaussian sources (with
       `(a, b, c)` ellipse parameters in `OSKAR_SKY_SCRATCH_EXT_*`),
       channel bandwidth smearing and time smearing.

3. After all work units, the block is copied to the host (`oskar_vis_block_copy`).

The finalisation thread writes blocks to disk while the next block is being
computed — overlapping I/O and compute. Both `.vis` and `.ms` writers are
supported simultaneously.

### 7.3 Visibility-block tuning

Settings (under `interferometer`):

- `channel_bandwidth_hz` (default 0): drives bandwidth smearing in the
  correlator.
- `time_average_sec` (default 0): drives time-smearing in `Tracking` mode.
- `max_time_samples_per_block` (default 8): how many time slots fit in one
  in-memory block.
- `max_channels_per_block` (default `auto`): same for frequency. Since 2.8.0
  visibility blocks are **tiled in frequency too**.
- `correlation_type`: `Cross-correlations` (default), `Auto-correlations`,
  `Both`.
- `use_casa_phase_convention` (default true since 2.10.0).
- `uv_filter_min`, `uv_filter_max`, `uv_filter_units` (`Wavelengths` or
  `Metres`): visibilities outside the filter are *not evaluated*.
- `oskar_vis_filename` (output `.vis`).
- `ms_filename` (output `.ms`).
- `force_polarised_ms`: write a 4-pol MS even in scalar mode.
- `ms_dish_diameter` (m): controls `DISH_DIAMETER` in the MS `ANTENNA` table
  (default 1 m for aperture arrays).
- `ignore_w_components`: zero out W → disables W-smearing (sanity test).

### 7.4 Simulator tuning

`simulator` group:

- `double_precision` (default true): set false for ~2× faster single-precision
  runs.
- `use_gpus` (default true).
- `cuda_device_ids`: comma-separated list (or `all`) of GPU IDs.
- `num_devices` (default `auto`): total CPU+GPU compute devices.
- `max_sources_per_chunk` (default 16384): reduce if GPU memory is tight.
  For best throughput, `num_chunks × times_per_block` should fully occupy all
  GPUs while each chunk has thousands of sources.
- `keep_log_file`, `write_status_to_log_file`.

### 7.5 Observation settings

`observation` group:

- `mode`: `Tracking` (default; phase centre fixed in RA, Dec) or
  `Drift scan` (Az, El). Drift scan is point-source only and does not apply
  time smearing. Limited drift-scan added in 2.8.0.
- `phase_centre_ra_deg`, `phase_centre_dec_deg` — `DoubleList`, may be
  multi-valued.
- `pointing_file` — see §6.9.
- `start_frequency_hz`, `num_channels`, `frequency_inc_hz`.
- `start_time_utc` — accepts MJD or string (`d-M-yyyy h:m:s.z`,
  `yyyy/M/d/h:m:s.z`, `yyyy-M-d h:m:s.z`, `yyyy-M-dTh:m:s.z`).
- `length` — observation length in s or `h:m:s.z`.
- `num_time_steps`.

Hidden / not-yet-wired: TAI-UTC offset, UT1-UTC offset, polar-motion x/y
arcsec.

### 7.6 Beam-pattern tuning

`beam_pattern` group (XML in `oskar_beam_pattern.xml`):

- `all_stations` vs explicit `station_ids` list (zero-based).
- `coordinate_frame`: `Equatorial` (RA, Dec image) or `Horizon` (whole sky,
  uses HEALPix-ish coords internally).
- `coordinate_type`: `Beam image` (tangent-plane image at phase centre) or
  `Sky model` (evaluate beam at supplied source positions only).
- `beam_image.size`, `beam_image.fov_deg`, `beam_image.cellsize_arcsec` (or
  `specify_cellsize=true`). Single value → square; `256,128` →
  rectangular.
- `output.separate_time_and_channel`, `average_time_and_channel`,
  `average_single_axis = None|Time|Channel`.
- `station_outputs.text_file/.hdf5_file/.fits_image` — toggle saving of:
  `raw_complex`, `amp`, `phase`, `auto_power`, `auto_power_phase`,
  `auto_power_real`, `auto_power_imag`. `auto_power*` is per-station total
  intensity.
- `telescope_outputs.{text,hdf5,fits_image}` — `cross_power_raw_complex`,
  `cross_power_amp`, `cross_power_phase`, `cross_power_real`,
  `cross_power_imag` — averaged from all selected stations.
- `test_source` (used when in `Full` polarisation mode and any auto/cross
  power output is requested): `stokes_i` (default true) or `custom` Stokes
  I/Q/U/V values for the test source.
- `root_path` — output filename prefix; OSKAR appends `_S<sid>_TIME_SEP_
  CHAN_SEP_<output>_<XX|XY|YX|YY>.fits` etc.

Output files for the `oskar_sim_beam_pattern` example are e.g.
`example_beam_pattern_S0000_TIME_SEP_CHAN_SEP_AMP_XX.fits` (a 4-pol
co/cross-pol cube of voltage amplitudes per (station, channel, time)).

### 7.7 Imager tuning

`image` group:

- Numerical: `double_precision`, `use_gpus`, `cuda_device_ids`, `num_devices`.
- Pixel grid: `specify_cellsize` (else FOV) → `fov_deg` or
  `cellsize_arcsec`; `size` (must be even).
- `image_type`: any of `Linear (XX,XY,YX,YY)`, `XX`, `XY`, `YX`, `YY`,
  `Stokes (I,Q,U,V)`, `I`, `Q`, `U`, `V`, `PSF`. Stokes images are uncalibrated
  (just the standard linear combinations).
- `channel_snapshots`: per-channel cube vs. frequency-synthesised single image.
- `freq_min_hz`/`freq_max_hz`, `time_min_utc`/`time_max_utc`,
  `uv_filter_min`/`uv_filter_max` (in **wavelengths**) — visibility filters.
- `algorithm`: `FFT` (default), `DFT 2D`, `DFT 3D`, `W-projection`.
- `weighting`: `Natural`, `Radial`, `Uniform`. Optional Gaussian taper:
  `weight_taper.u_wavelengths`, `weight_taper.v_wavelengths` →
  weight × `exp(log(0.3) · ((u/scale_u)² + (v/scale_v)²))`.
- `fft.kernel_type`: `Spheroidal` (default) or `Pillbox`. `support` (default
  3), `oversample` (default 100). FFT/grid can run on GPU.
- `wproj.num_w_planes` (auto if < 1), `generate_w_kernels_on_gpu` (default
  true).
- `direction`: `Observation direction` (phase centre) or `RA, Dec.` (with
  explicit `ra_deg`, `dec_deg`) — supports off-phase-centre imaging.
- Inputs: `input_vis_data` (one or more `.vis` or `.ms`).
- `scale_norm_with_num_input_files` (true if combining sky-model components
  from separate files, false if combining different observations of the
  same sky).
- `ms_column`: `DATA`, `MODEL_DATA`, or `CORRECTED_DATA`.
- `root_path` → `<root>_<image_type>.fits`. Since 2.12.0 the FITS header
  carries an `EQUINOX` keyword.

### 7.8 Timer breakdown

Upon completion, `oskar_sim_interferometer` logs per-step timing percentages:
`Copy` (host↔device), `Horizon clip`, `Jones E`, `Jones K`, `Jones join`,
`Jones correlate`, `Other`. Plus aggregate timers for `Total wall time`,
`Load`, `Compute time` (per GPU), and `Write time`.

---

## 8. Settings system (XML schema → INI files → SettingsTree)

OSKAR ships an XML-driven settings system (`oskar/settings/`). Each app has
a root XML in `oskar/apps/xml/` that imports per-feature XML fragments. The
schemas are compiled into the binaries at build time (cf.
`oskar_settings_xml_utility.cmake`).

### 8.1 Root XMLs

| App | Root XML | Imports |
|---|---|---|
| GUI dispatcher (`oskar`) | `oskar.xml` | simulator + sky_model + observation + telescope_model + interferometer + beam_pattern + image |
| `oskar_sim_interferometer` | `oskar_sim_interferometer.xml` | simulator + sky_model + observation + telescope_model + interferometer |
| `oskar_sim_beam_pattern` | `oskar_sim_beam_pattern.xml` | simulator + observation + telescope_model + beam_pattern |
| `oskar_imager` | `oskar_imager.xml` | image |
| `oskar_sim_tec_screen` | `oskar_sim_tec_screen.xml` | ionosphere + observation + telescope_model |

Schema fragments:

- `oskar_simulator.xml` — global compute: `double_precision`, `use_gpus`,
  `cuda_device_ids`, `num_devices`, `max_sources_per_chunk`, `keep_log_file`,
  `write_status_to_log_file`.
- `oskar_observation.xml` — phase centre, frequency, time grid, mode,
  pointing file.
- `oskar_telescope_model.xml` — telescope I/O, station model, polarisation
  mode, station type, ionosphere.
- `oskar_telescope_AA_array.xml` — `aperture_array.array_pattern.*`.
- `oskar_telescope_AA_element.xml` — `aperture_array.element_pattern.*`.
- `oskar_telescope_gaussian.xml` — `gaussian_beam.*`.
- `oskar_sky_model.xml` — sky-model loaders, generators, filters, overrides,
  output.
- `oskar_interferometer.xml` — visibility-block tuning, output files, MS
  options, UV filters, CASA convention.
- `oskar_interferometer_noise.xml` — noise enable/seed/freq/RMS specification.
- `oskar_beam_pattern.xml` — beam-pattern coordinate, image, output, test
  source.
- `oskar_image.xml` — imager (algorithm, weighting, FFT/wproj options,
  filters, direction, IO).
- `oskar_ionosphere.xml` — legacy 2-D TID screen, TEC image, pierce-point
  output.

### 8.2 XML element vocabulary

- `<s k="key">` defines a setting node, with child `<label>`, `<desc>`,
  `<type>` and optional `<depends>` / nested `<s>` groups.
- `<group k="alias">` defines a reusable group; `<import group="…"/>` and
  `<import filename="…"/>` paste it into another tree.
- `<type name="…" default="…">` — types include: `bool`, `int`, `uint`,
  `IntPositive`, `IntList`, `IntListExt`, `IntRange`, `IntRangeExt`, `double`,
  `UnsignedDouble`, `DoubleList`, `DoubleRange`, `DoubleRangeExt`,
  `RandomSeed`, `OptionList`, `DateTime`, `Time`, `InputFile`, `InputFileList`,
  `InputDirectory`, `OutputFile`, `String`. The `*Ext` types accept symbolic
  values like `auto`, `min`, `max`, `all`, `file`, `default`.
- `<depends k="path/to/key" v="value"/>` and the `<logic group="AND|OR">`
  wrapper express conditional visibility.
- `priority="1"` flags the most user-visible settings (hidden in
  "advanced-only" UI views).
- `required="true"` enforces that the user must set the value.

### 8.3 INI files

The GUI and CLI persist settings as INI-style key-value files
(`oskar_SettingsFileHandlerIni.h`). Keys use `/`-separated paths matching the
XML hierarchy:

```
[General]
app=oskar_sim_interferometer
version=2.12.3

[simulator]
double_precision=false
use_gpus=true

[telescope]
input_directory=telescope.tm
pol_mode=Full
station_type=Aperture array
…

[observation]
phase_centre_ra_deg=20.0
phase_centre_dec_deg=-30.0
start_frequency_hz=100e6
…
```

CLI usage:

```
oskar_sim_interferometer --set telescope/input_directory=/data/skala.tm settings.ini
oskar_sim_interferometer --get observation/phase_centre_ra_deg settings.ini
oskar_sim_interferometer settings.ini
```

If a key is set with `--set` and no value, it is reset to its default.

---

## 9. Visibility binary file format (`.vis`)

OSKAR's native format is a portable, self-describing chunked binary. Programs
read it via `oskar_binary_*` library functions; the layout is documented in
`docs/binary_file/binary_file.rst`.

### 9.1 File header (64 bytes, format version 2)

Only the first 10 bytes are used:

| Offset | Length | Meaning |
|---|---|---|
| 0 | 9 | ASCII `"OSKARBIN"` + NUL |
| 9 | 1 | binary format version (currently 2) |
| 10–63 | 54 | reserved (zeros). v1 used these for endianness, sizeofs, OSKAR version. |

### 9.2 Chunks and tags

Each *chunk* = (20-byte tag) + payload + optional 4-byte CRC-32C. The tag
contains:

| Off | Len | Field |
|---|---|---|
| 0 | 1 | `'T'` (0x54) |
| 1 | 1 | `'A'` (v1) or `'B'` (v2) |
| 2 | 1 | `'G'` (0x47) |
| 3 | 1 | element size in bytes |
| 4 | 1 | flags |
| 5 | 1 | data type code |
| 6 | 1 | group ID (or group-name length if extended) |
| 7 | 1 | tag ID (or tag-name length if extended) |
| 8 | 4 | user-specified index (LE int32) |
| 12 | 8 | block size in bytes (LE int64) |

Flags bits: 5 = big-endian payload, 6 = CRC present (Castagnoli polynomial
`0x82F63B78`), 7 = extended tag. Bits 0–4 reserved (must be 0).

Data-type bitmask: 0=char/string, 1=int (4B), 2=float (4B), 3=double (8B),
5=complex (pair of values), 6=matrix (4 values, row-major a,b,c,d). e.g.
double-complex matrix = 0b01101000 = 104.

### 9.3 Standard tag groups

- **Metadata (group 1)**: 1=creation date string, 2=OSKAR version string,
  3=username, 4=current working directory.
- **Settings (group 3)**: 1=settings file path, 2=settings file contents.
- **Run information (group 4)**: 1=run log (truncated to head+tail if > 20 kB
  since 2.9.6).
- **Visibility Header (group 11)**: tags 1–48 covering telescope path
  (1), tags-per-block (2), auto- and cross-correlation flags (3, 4), data
  type (5), coord precision (6), per-block max times (7) and total times (8),
  per-block max channels (9) and total channels (10), num stations (11),
  polarisation type code (12), CASA convention flag (13), phase-centre
  type (21) and (lon, lat) (22), start freq (23), freq inc (24), channel
  bandwidth (25), MJD start (26), time inc (27), correlator dump time (28),
  telescope reference (lon 29, lat 30, alt 31), per-station offset-ECEF
  positions (32, 33, 34), per-element ENU positions (35, 36, 37), station
  name (41), station diameter (42), feed Euler angles (43–48).
- **Visibility Block (group 12)**: 1=`[start_t, start_chan, n_t, n_chan,
  n_baselines, n_stations]` (int[6]); 2=auto-correlations (Jy, complex
  scalar/matrix); 3=cross-correlations; 4–6=baseline UU/VV/WW (m); 7–9=station
  U/V/W (m).

### 9.4 Polarisation type codes (Tag ID 12 in header)

| Code | Meaning |
|---|---|
| 0 | Full Stokes (I, Q, U, V) |
| 1 | Stokes I |
| 2 | Stokes Q |
| 3 | Stokes U |
| 4 | Stokes V |
| 10 | All linear (XX, XY, YX, YY) |
| 11 | XX |
| 12 | XY |
| 13 | YX |
| 14 | YY |

### 9.5 Phase centre type (Tag ID 21)

`0` = Tracking (RA, Dec); `1` = Drift scan (Az, El).

### 9.6 Block dimension order

Visibility data are stored with **time slowest, channel middle, baseline (or
station) fastest**, with the polarisation dimension implicit in the data-type
code (matrix → 4 pols, scalar → 1 pol). For *N* stations there are
*N(N-1)/2* cross-correlation baselines, formed in canonical order (0-1, 0-2,
…, 1-2, 1-3, …). Station U/V/W use *N/2* less storage than baseline UU/VV/WW
and are preferred — current OSKAR no longer writes baseline coordinates by
default. Multiple blocks tile (time, channel) with channel varying faster.

### 9.7 Supporting tools

- `oskar_binary_file_query <file>` prints the chunk index.
- `oskar_vis_summary <file>` prints metadata, settings, and run log.
- `oskar_vis_to_ms <file…>` writes a single CASA Measurement Set
  (concatenated if multiple inputs).
- `oskar_vis_add <file…>` sums compatible files (same telescope and
  observation).
- `oskar_vis_add_noise --settings <ini> <file…>` overwrites or copies and
  adds noise.

### 9.8 Measurement Set output

Provided by `oskar/ms/` (depends on `casacore`). Notable behaviour:

- `ANTENNA` table uses **absolute** ECEF coordinates (since 2.8.0). Since
  2.10.0, dish diameter is configurable via `ms_dish_diameter`; station
  directory names are written to the antenna name column.
- `FEED` table receives feed-angle values (since 2.10.0).
- `PHASED_ARRAY` table is written (since 2.8.0). The `COORDINATE_AXES`
  column is set, and `COORDINATE_SYSTEM` is omitted (since 2.10.1).
- Tile size for the `DATA` storage manager aims for ≥ 4 MB (since 2.10.0)
  or ≈ 1 MB (since 2.11.0); other relevant columns share the same tile shape.
- `ANTENNA1`/`ANTENNA2` indices were swapped in 2.9.6 to allow DP3 to apply
  gains correctly.
- `STATE` table fix in 2.9.0.

---

## 10. Imager (`oskar_imager` and `oskar.Imager`)

The imager is shared between the standalone CLI binary and the Python
`oskar.Imager` class, and is used internally by `oskar.ImagingInterferometer`.

### 10.1 Pipeline

1. `set` properties or load settings via `SettingsTree.to_imager()`.
2. `check_init` (called automatically by `run`).
3. Read or push input visibilities. Two modes:
   - **From file(s)**: call `run()` with no args; imager reads `.vis` and
     `.ms` files, applies time/freq/uv filters and weights.
   - **From memory**: call `update(uu, vv, ww, vis, weight=…)` repeatedly,
     then `finalise(return_images=1, return_grids=1)`.
   - **From a `VisBlock`**: call `update_from_block(header, block)`.
4. `coords_only=True` mode: imager reads only (u, v, w) the first pass to
   compute the uniform-weighting weight grid, then is reset and rerun with
   `coords_only=False` to grid the visibilities.

Off-phase-centre imaging: `set_direction(ra_deg, dec_deg)` shifts the image
centre and rotates the supplied (u, v, w) and visibilities accordingly via
`rotate_coords` and `rotate_vis`.

### 10.2 Algorithms

- **FFT** with separable spheroidal or pillbox gridding kernel
  (`oskar/imager/private_imager_update_plane_fft.h`). Convolution support
  size 3 by default, oversampling 100.
- **DFT 2D / DFT 3D** — direct evaluation of the imaging integral; expensive
  for many visibilities but exact (no gridding artefacts). 3-D version
  applies `e^{-2πi w (n−1)}` per source.
- **W-projection** — convolution kernel cube generated as `e^{-2πi w (n−1)}`
  in the image plane; can be generated on GPU
  (`generate_w_kernels_on_gpu=True`, default true). `num_w_planes` auto if
  ≤ 0.

### 10.3 Weighting

- `Natural` (default) — proportional to the visibility weights.
- `Radial` — weight ∝ 1/√(u² + v²).
- `Uniform` — bin (u, v) onto the imaging grid and divide each visibility
  weight by the per-cell weight sum (requires the `coords_only` two-pass
  trick).

Optional Gaussian taper:
`weight × exp(log(0.3) · ((u/scale_u)² + (v/scale_v)²))` with
`u_wavelengths` and `v_wavelengths` scales.

### 10.4 Output

- FITS image cubes named `<root>_<image_type>.fits` (since 2.12.0 with
  `EQUINOX` header).
- Optional return of image and grid arrays directly to Python with
  `return_images=1, return_grids=1`.

---

## 11. Beam-pattern simulator (`oskar_sim_beam_pattern`)

Uses the same E-Jones evaluation as the main simulator, but with no source
sum and no correlator. Outputs voltage / amplitude / phase / power patterns
either on a tangent-plane image grid (Equatorial frame) or over the whole
sky (Horizon frame), per (station, time, channel). Per-station outputs go to
text, HDF5 (since 2.12.3) or FITS files; `telescope_outputs.cross_power_*` are
station-pair averages over the chosen station list.

The `test_source` group (only used in `Full` polarisation mode and when the
selected outputs include power) supplies a Stokes I-only test source by
default, or a custom (I, Q, U, V) tuple; this drives the polarised
auto/cross-power patterns the user inspects.

---

## 12. Python interface (`oskarpy`)

The Python package is **a thin layer on top of the C library**. Each class
delegates almost every call into a compiled extension module
(`_interferometer_lib`, `_sky_lib`, `_telescope_lib`, `_imager_lib`,
`_measurement_set_lib`, `_binary_lib`, `_settings_tree_lib`, `_vis_block_lib`,
`_vis_header_lib`, `_bda_utils`). All processing-intensive methods release
the GIL for parallelism with other Python threads.

### 12.1 Installation

```bash
# Linux/macOS (after OSKAR is installed)
pip install --user 'git+https://github.com/OxfordSKA/OSKAR.git@master#egg=oskarpy&subdirectory=python'
```

If OSKAR is in a non-standard location, set `OSKAR_INC_DIR` and
`OSKAR_LIB_DIR` env vars or edit `python/setup.cfg`.

`pip uninstall oskarpy` removes only the Python interface (not OSKAR itself).

### 12.2 Top-level public API (`from oskar import …`)

`__init__.py` exports:

- `__version__` (string).
- `BDA`, `apply_gains`, `vis_list_to_matrix` (baseline-dependent averaging).
- `Binary` (raw binary-file accessor).
- `Imager` (visibility imager).
- `ImagingInterferometer` (Interferometer subclass that pipes blocks
  through one or more `Imager` instances on the fly).
- `Interferometer` (the simulator).
- `MeasurementSet` (simple casacore-backed read/write).
- `SettingsTree` (per-app INI editor and adapter).
- `Sky` (sky model).
- `Telescope` (telescope model).
- `oskar_version_string` (reports the loaded library version).
- `VisBlock`, `VisHeader` (visibility-data accessors).

### 12.3 `SettingsTree`

Constructor: `SettingsTree(app=None, settings_file='')` — `app` is one of the
schema names (`oskar_sim_interferometer`, `oskar_sim_beam_pattern`,
`oskar_imager`).

Methods:

- `from_dict(d)` — populate from a nested Python dict using XML-tree paths.
- `to_dict(include_defaults=False)`.
- `set_value(key, value, write=True)`, `value(key)`.
- `__getitem__`/`__setitem__` shortcuts to `set_value`/`value`.
- Convenience adapters: `to_imager()`, `to_interferometer()`, `to_sky()`,
  `to_telescope()`. These return a fully-built object whose internal state
  matches the settings.

### 12.4 `Sky`

Highlights:

- Constructors: `Sky(precision='double')`, `Sky(settings=…)`,
  `Sky.load(filename, precision='double')`,
  `Sky.from_array(array, precision='double')`,
  `Sky.from_fits_file(filename, min_peak_fraction=0.0, min_abs_val=0.0, …)`,
  `Sky.generate_grid(ra0, dec0, side, fov_deg, …)`,
  `Sky.generate_random_power_law(num, fmin, fmax, power, …)`.
- Mutation: `append(other)`, `append_sources(ra_deg, dec_deg, I, Q=…, U=…,
  V=…, ref_freq_hz=…, spectral_index=…, rotation_measure=…,
  major_axis_arcsec=…, minor_axis_arcsec=…, position_angle_deg=…)`,
  `append_file(filename)`.
- Filtering: `filter_by_flux(min_jy, max_jy)`,
  `filter_by_radius(inner_deg, outer_deg, ra0_deg, dec0_deg)`.
- I/O: `save(filename)`, `save_named_columns(filename, …)`,
  `to_array()`, `to_ds9_regions(filename, colour='green', width=1)`.
- Properties: `num_sources`, `capsule`.

The `from_array` accepts a 2-D NumPy array with up to 12 columns mapping to
the fixed-format sky-model columns.

### 12.5 `Telescope`

Highlights:

- Construction: `Telescope(precision='double')` or `Telescope(settings=…)`.
  When `settings` is supplied, the telescope is built from
  `settings.to_telescope()`.
- Loading: `load(dir_name)`. Note: call `set_pol_mode` and
  `set_enable_numerical_patterns` *before* `load`.
- Coordinates: `set_position(longitude_deg, latitude_deg, altitude_m=0)`,
  `set_station_coords_enu(...)`, `set_station_coords_ecef(...)`,
  `set_station_coords_wgs84(...)`.
- Phase centre: `set_phase_centre(ra_deg, dec_deg)`.
- Beam: `set_station_type('Array'|'Gaussian'|'Isotropic')` (only first
  letter is checked); `set_gaussian_station_beam_width(fwhm_deg, ref_hz)`.
- Smearing: `set_channel_bandwidth(hz)`, `set_time_average(sec)`.
- Noise: `set_enable_noise(value, seed=1)`,
  `set_noise_freq(start_hz, inc_hz=0, num_channels=1)`,
  `set_noise_rms(start_jy, end_jy=None)`.
- UV filter: `set_uv_filter(min, max, units='Metres'|'Wavelengths')`.
- Element overrides: `override_element_cable_length_errors(std, seed, mean,
  feed)`, `override_element_gains(mean, std, seed, feed)`,
  `override_element_phases(std_deg, seed, feed)`.
- Reflection: `num_stations`, `num_baselines`, `max_station_size`,
  `max_station_depth`.
- Behaviour flags: `set_allow_station_beam_duplication(bool)`,
  `set_enable_numerical_patterns(bool)`.

### 12.6 `Interferometer`

Constructor `Interferometer(precision='double', settings=None)`. Workflow:

```python
sim = oskar.Interferometer(settings=settings)   # or precision-only
sim.set_sky_model(sky)                          # optional override
sim.set_telescope_model(tel)                    # optional override
sim.run()                                       # multi-threaded run
```

Key methods:

- Setup: `set_observation_frequency(start_hz, inc_hz=0, num_channels=1)`,
  `set_observation_time(start_mjd, length_sec, num_time_steps)`.
- Multi-device: `set_gpus(ids)` (list, `-1` = all, `None` = none),
  `set_num_devices(n)`. For pure CPU: `set_gpus(None);
  set_num_devices(1)`.
- Tuning: `set_max_sources_per_chunk(n)`, `set_max_times_per_block(n)`,
  `set_horizon_clip(bool)`, `set_coords_only(bool)`.
- Output: `set_output_vis_file(path)`, `set_output_measurement_set(path)`,
  `set_settings_path(path)` (just stamped into the output file metadata).
- Lifecycle: `check_init()`, `reset_cache()`, `reset_work_unit_index()`,
  `run_block(block_index, device_id=0)`, `finalise_block(block_index)`,
  `finalise()`, `vis_header()`.
- Override: subclass and override `process_block(block, block_index)` to
  process visibilities on-the-fly. Default implementation calls
  `write_block(block, block_index)`.
- Properties: `coords_only`, `num_devices`, `num_gpus`, `num_vis_blocks`,
  `capsule`.

The `run()` method spawns `num_devices + 1` threads via Python `Thread`,
synchronised by an internal `Barrier`. Thread 0 finalises and processes the
block produced by all the GPUs; threads 1..N each call `run_block` for each
block index. See `oskar/python/oskar/interferometer.py:_run_blocks`.

A canonical Python script (`docs/python/example_hello_world.rst`):

```python
import oskar, numpy

params = {
    "simulator":   {"use_gpus": False},
    "observation": {"num_channels": 3, "start_frequency_hz": 100e6,
                    "frequency_inc_hz": 20e6,
                    "phase_centre_ra_deg": 20, "phase_centre_dec_deg": -30,
                    "num_time_steps": 24,
                    "start_time_utc": "01-01-2000 12:00:00.000",
                    "length": "12:00:00.000"},
    "telescope":   {"input_directory": "telescope.tm"},
    "interferometer": {"oskar_vis_filename": "example.vis",
                       "ms_filename": "",
                       "channel_bandwidth_hz": 1e6,
                       "time_average_sec": 10},
}
settings = oskar.SettingsTree("oskar_sim_interferometer")
settings.from_dict(params)
settings["simulator/double_precision"] = False

sky_data = numpy.array([
    [20.0, -30.0, 1, 0, 0, 0, 100.0e6, -0.7, 0.0, 0,   0,   0],
    [20.0, -30.5, 3, 2, 2, 0, 100.0e6, -0.7, 0.0, 600, 50,  45],
    [20.5, -30.5, 3, 0, 0, 2, 100.0e6, -0.7, 0.0, 700, 10, -10],
])
sky = oskar.Sky.from_array(sky_data, "single")

sim = oskar.Interferometer(settings=settings)
sim.set_sky_model(sky)
sim.run()

imager = oskar.Imager("single")
imager.set(fov_deg=4, image_size=512,
           input_file="example.vis", output_root="example")
output = imager.run(return_images=1)
```

A *visibility-corruption* pattern (from `examples/corruptor.py`):

```python
class Corruptor(oskar.Interferometer):
    def process_block(self, block, block_index):
        amp = block.cross_correlations()    # numpy array (T, C, B, P)
        amp *= 2.0
        self.write_block(block, block_index)

settings = oskar.SettingsTree('oskar_sim_interferometer', sys.argv[-1])
Corruptor(oskar_settings=settings).run()
```

### 12.7 `Imager`

Property-style API; `set(**kwargs)` is convenience for setting many at once.
Key properties: `algorithm`, `weighting`, `cellsize_arcsec`, `fov_deg`,
`channel_snapshots`, `coords_only`, `fft_on_gpu`, `grid_on_gpu`,
`generate_w_kernels_on_gpu`, `image_size` / `size`, `image_type`,
`input_file`, `ms_column`, `num_w_planes` / `wprojplanes`, `output_root` /
`root_path`, `plane_size`, `time_min_utc`, `time_max_utc`, `freq_min_hz`,
`freq_max_hz`, `uv_filter_min`, `uv_filter_max`,
`scale_norm_with_num_input_files`.

Pipeline methods: `check_init`, `update(uu, vv, ww, amps, weight=None,
…)`, `update_from_block(header, block)`, `finalise(return_images=0,
return_grids=0)`, `finalise_plane(plane, plane_norm)`, `reset_cache`,
`run(uu=None, vv=None, ww=None, amps=None, weight=None,
return_images=0, return_grids=0)`.

Off-phase-centre helpers: `set_default_direction()`, `set_direction(ra,
dec)`, `rotate_coords(uu, vv, ww)`, `rotate_vis(uu, vv, ww, vis)`.

### 12.8 `ImagingInterferometer`

Subclass of `Interferometer` that takes a list of `Imager` instances, and
pipes each finalised block through `update_from_block()`. Exits with
finalised images/grids. Used in `python/examples/sim_image_via_memory.py` to
produce Natural and Uniform images in a single sim pass — no `.ms`/`.vis`
file written.

### 12.9 `MeasurementSet`

`casacore`-backed wrapper for a single-spectral-window MS:

- Constructors: `MeasurementSet.create(filename, …)`, `MeasurementSet.open(filename)`.
- Reading: `read_column(name)`, `read_vis(start_row, …)`,
  `read_coords(start_row, …)`.
- Writing: `write_vis(…)`, `write_coords(…)`. Antenna indices are derived
  from the canonical baseline order.

### 12.10 `Binary`, `VisHeader`, `VisBlock`

Low-level access to the OSKAR `.vis` format:

- `Binary` — open a file and iterate over chunks.
- `VisHeader.read(binary_file)` → returns a header object exposing
  amp/coord types, channel/time max-per-block, total channels/times,
  phase-centre RA/Dec, station count, max channels per block, MJD start, time
  inc, etc.
- `VisBlock.create_from_header(header)` then `block.read(header,
  binary_handle, block_index)`. Accessors: `auto_correlations()`,
  `cross_correlations()`, `baseline_uu_metres()`, `baseline_vv_metres()`,
  `baseline_ww_metres()`, plus `num_baselines`, `num_channels`, `num_pols`,
  `num_stations`, `num_times`, `start_channel_index`, `start_time_index`.
  All accessors return NumPy views (no copy) into the C-allocated buffer.

### 12.11 `BDA`

Baseline-dependent averaging utilities:

```python
b = oskar.BDA(num_antennas, num_pols=1)
b.set_compression(max_fact, fov_deg, wavelength_m, max_avg_time_s)
b.set_delta_t(value_s)
b.set_num_times(value)
b.set_initial_coords(uu, vv, ww)
for t in times: b.add_data(t, vis, uu_next, vv_next, ww_next)
result = b.finalise()
```

Plus standalone helpers: `apply_gains(vis_amp, gains)` (with a `c8`/`c16`
overload) and `vis_list_to_matrix(vis_list, num_antennas)`.

### 12.12 SPEAD streaming

`python/examples/spead/` contains a demonstration of streaming visibilities
from `Interferometer.process_block` over the SPEAD protocol — used in SDP
end-to-end tests.

---

## 13. Containers and CI/CD

### 13.1 Apptainer / Singularity

`apptainer/Apptainer.python3` is the recipe; pre-built SIFs are published on
the GitHub releases page. Run with:

```bash
singularity exec --nv ./OSKAR-Python3.sif oskar_sim_interferometer settings.ini
singularity exec --nv ./OSKAR-Python3.sif python3 sim_script.py
```

`--nv` enables NVIDIA GPU support. Default behaviour mounts the user's home
directory; pass `--no-home` or `-H /tmp` to isolate from local installs.

### 13.2 Docker / Kubernetes

- `docker/oskar-python3` — runtime image published as
  `artefact.skao.int/oskar-python3` on the SKAO registry.
- `docker/oskar-ci-cuda-12-6` — CI build environment.
- Helper script `oskar_run_k8s` (in `docs/python/`) for one-shot Kubernetes
  Jobs.
- GitLab CI pipelines (`.gitlab-ci.yml`) build, test, package and publish
  the Docker images, Apptainer SIFs, and ReadTheDocs content.

### 13.3 Native packaging

- macOS: `oskar_packaging.cmake`, `oskar_hdiutil.sh`, `Info.plist.in`, and
  `cmake/oskar_bundle_deps.cmake.in` build a self-contained `OSKAR.app` bundle.
- Linux: standard `make install` into `/usr/local`; CPack rules also produce
  DEB/RPM packages.
- Windows: an NSIS installer with a "Add OSKAR to PATH" toggle and optional
  headers/libs for building the Python interface.

---

## 14. Recent ChangeLog highlights (2.5+ → 2.12.3)

Selected entries that affect behaviour rather than just packaging.

- **2.12.3** (in development): HDF5 beam-pattern output; SKA sky-model column
  names; quoted-field handling; per-line column-count validation; FEKO HDF5
  pattern files matched by "FEKO" anywhere in the name.
- **2.12.2** (2026-02-19): use `posix_memalign` instead of `aligned_alloc` for
  better macOS support.
- **2.12.1** (2026-02-13): faster FEKO normalised-polynomial evaluator;
  prefer `aligned_alloc` for host memory.
- **2.12.0** (2026-02-06): rewritten sky-model code (logarithmic and linear
  spectral polynomials with up to 8 terms; named-column files; spectral
  curvature; spectral lines); FEKO HDF5 spherical-wave coefficients; per-XML
  setting for max spherical-wave order; `EQUINOX` keyword in imager FITS
  headers; **fix** ionospheric Faraday rotation unit (GHz → Hz); **fix**
  IGRF altitude unit (m → km); **fix** small memory leak.
- **2.11.1** (2025-06-27): build system updates.
- **2.11.0** (2025-06-25): **revert** to (θ, φ) basis from Ludwig-3 in
  element-pattern evaluation (fixes pol leakage); **fix** WGS84 absolute
  station coords in MS; **fix** PHASED_ARRAY axes; remove unsupported
  spline element-pattern data; remove unused virtual antenna angle; MS
  storage manager tile size now ≈ 1 MB.
- **2.10.1** (2025-05-08): ionospheric Faraday rotation included in TEC eval;
  TEC screens may use a different time grid from the visibilities; cable
  length errors at telescope level; Apptainer build → Ubuntu 22.04 + CUDA 12.
- **2.10.0** (2025-03-26): **CASA phase convention default flipped** —
  baseline coord/visibility-phase signs and antenna index order now match
  CASA. `ms_dish_diameter` setting; station folder names → MS antenna names;
  feed angle → MS FEED table; single-line feed-angle file allowed; MS DATA
  tile ≈ 4 MB.
- **2.9.6** (2025-02-04): thread-safe HARP HDF5 loading; ANTENNA1/ANTENNA2
  swap (DP3 gain compatibility); truncate logs > 20 kB.
- **2.9.5** (2024-05-03): correct virtual antenna rotation; lazy HARP load.
- **2.9.0** (2023-11-12): polarised elliptical Gaussian station beams;
  experimental virtual antenna rotation; isotropic element fix; MS STATE fix;
  HARP library bumped.
- **2.8.3** (2022-05-26): HARP beam library option; MS tile-size fix > 2 GiB.
- **2.8.2** (2022-02-19): thread-safe HDF5 reference counter.
- **2.8.1** (2022-02-17): HDF5 gain tables in station/telescope models.
- **2.8.0** (2021-11-23): MS ANTENNA absolute coords; per-channel noise
  seeding; PHASED_ARRAY table; Ludwig-3 antenna basis (later reverted in
  2.11.0); per-pol element data; custom Stokes test-source for beam pattern;
  visibility blocks tile in frequency too; limited drift-scan mode.

For older entries see `ChangeLog.txt`.

---

## 15. References (cited in `docs/theory/theory.rst`)

- Hamaker, J. P., Bregman, J. D. & Sault, R. J., 1996, A&AS, 117, 137.
- Hamaker, J. P., Bregman, J. D., 1996, A&AS, 117, 161.
- IAU, 1974, Transactions of the IAU Vol. 15B (1973) 166.
- Smirnov, O. M., 2011, A&A, 527, 106.
- Thompson, A. R., Moran, J. M., & Swenson, G. W., 2001, *Interferometry and
  Synthesis in Radio Astronomy*.
- Wrobel, J. M., & Walker, R. C., 1999, *Synthesis Imaging in Radio Astronomy
  II*, p. 171.
- Callingham, J. R., et al., 2017, ApJ, 836, 174 — spectral curvature.
- LOFAR `makesourcedb` Wiki — named-column sky-model file format and
  logarithmic spectral indices.
- ARatmospy — TEC FITS format used by the external ionosphere screen.

External links from the package documentation:

- OSKAR releases: <https://github.com/OxfordSKA/OSKAR/releases>.
- Sphinx docs: <https://ska-telescope.gitlab.io/sim/oskar/>.
- LOFAR `makesourcedb`: <https://www.astron.nl/lofarwiki/doku.php?id=public:user_software:documentation:makesourcedb>.
- LOFAR Imaging Cookbook (BBS rotation measure):
  <https://support.astron.nl/LOFARImagingCookbook/bbs.html#rotation-measure>.

---

## 16. Quick mapping between OSKAR concepts and RadioSim concepts

For cross-reference when integrating with the rest of the `simulators/` tree:

| OSKAR concept | RadioSim equivalent |
|---|---|
| `oskar_Sky` (point + Gaussian, named-column file) | `radiosim.core.sky.SkyModel` (with `PointSourceData` and the `bbs`/`pyradiosky_file` loaders) |
| `oskar_Telescope` directory tree | RadioSim antenna readers (`io/antenna_readers.py`) + station/element model |
| Aperture-array `E`-Jones with array factor + element pattern | `radiosim.core.jones.beam.BeamJones` + `AnalyticBeamJones` (with optional `hpbw_per_antenna`) |
| Gaussian station beam | `AnalyticBeamJones` with Gaussian preset |
| `K`-Jones (`exp(-2π i (ul+vm+w(n−1)))`) | `GeometricPhaseJones` |
| `R`-Jones (parallactic) | `ParallacticAngleJones` |
| `Z`-Jones (TEC screen) | `IonosphereJones` |
| `G`-Jones (HDF5 gains) | `GainJones` (and `BandpassJones`) |
| Cable-length errors | `DelayJones` (Kd term) |
| HEALPix RING FITS sky | RadioSim `_loaders_diffuse.py` / `_loaders_fits.py` |
| Imager (FFT/DFT/W-projection) | external (`wsclean`, `rascil`, `fftvis` family) |
| `oskar_VisHeader` / `oskar_VisBlock` binary file | written via `io/writers.py` HDF5 + casacore MS via `io/measurement_set.py` |

This mapping is informational only; OSKAR's semantics (e.g., the
B = `[[I+Q,U+iV],[U-iV,I-Q]]` convention, no 1/2 factor) differ in places
from RadioSim (`C = ½[[I+Q,U-iV],[U+iV,I-Q]]`), so visibility scaling differs
by a factor of two between the two packages.
