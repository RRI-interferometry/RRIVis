# WODEN — Exhaustive Reference

**WODEN** is a hybrid Python / C / C++ / GPU (CUDA + HIP/ROCm) simulator that produces visibilities for low-frequency radio interferometers via direct evaluation of the Radio-Interferometer Measurement Equation (RIME). It was originally written by **Jack L. B. Line** (ICRAR, ASTRO-3D, Curtin University) to support the Murchison Widefield Array (MWA) Epoch-of-Reionisation pipeline driven by the RTS calibrator-imager, but has since grown into a general-purpose, multi-telescope, multi-beam simulator that can write `uvfits` for MWA, EDA2, LOFAR, OSKAR/SKA-LOW, and HERA-style arrays.

This document is a code-level reference for the `simulators/WODEN/` snapshot in this monorepo (project version `2.7.0` per `setup.py`, with the `wodenpy` Python package self-reporting `2.6.0-alpha` in `wodenpy/__init__.py`). The intent is that a reader can come away with the same level of detail as someone who has read the source line-by-line: every C struct, every Python class, every CLI flag, every primary beam, every sky-model column, every numeric convention, every script, and every dependency is described. Repository upstream: `https://github.com/JLBLine/WODEN`. Documentation: `https://woden.readthedocs.io/`. JOSS paper: `Line (2022)` — `https://doi.org/10.21105/joss.03676`.

---

## Table of Contents

1. [Project metadata, license, authorship](#1-project-metadata-license-authorship)
2. [What WODEN is and is not — scientific overview](#2-what-woden-is-and-is-not--scientific-overview)
3. [The RIME and how WODEN evaluates it](#3-the-rime-and-how-woden-evaluates-it)
4. [Repository layout](#4-repository-layout)
5. [Build system: CMake, dual precision, CUDA/HIP, EveryBeam, hyperbeam](#5-build-system-cmake-dual-precision-cudahip-everybeam-hyperbeam)
6. [Constants, units, and numeric conventions (`include/constants.h`)](#6-constants-units-and-numeric-conventions-includeconstantsh)
7. [The core C/CUDA struct universe (`woden_struct_defs.h`)](#7-the-core-ccuda-struct-universe-woden_struct_defsh)
8. [Top-level execution path: `run_woden` (in `src/woden.c`)](#8-top-level-execution-path-run_woden-in-srcwodenc)
9. [Visibility calculation orchestration (`calculate_visibilities_*`)](#9-visibility-calculation-orchestration-calculate_visibilities_)
10. [Source-component primitives: extrapolation, beam application, RIME kernels](#10-source-component-primitives-extrapolation-beam-application-rime-kernels)
11. [Fundamental coordinates: l,m,n, u,v,w, shapelet u,v](#11-fundamental-coordinates-lmn-uvw-shapelet-uv)
12. [Primary-beam catalogue](#12-primary-beam-catalogue)
13. [Shapelet basis functions](#13-shapelet-basis-functions)
14. [EveryBeam C++ wrapper (`src/call_everybeam.cc`)](#14-everybeam-c-wrapper-srccall_everybeamcc)
15. [Logger, hyperbeam-error, compilation-flag check](#15-logger-hyperbeam-error-compilation-flag-check)
16. [`wodenpy` — the Python package](#16-wodenpy--the-python-package)
17. [`run_woden.py` — main driver script](#17-run_wodenpy--main-driver-script)
18. [Companion scripts (`scripts/`)](#18-companion-scripts-scripts)
19. [Sky-model formats: FITS, hyperdrive YAML, native text](#19-sky-model-formats-fits-hyperdrive-yaml-native-text)
20. [UVFITS output format and frequency banding](#20-uvfits-output-format-and-frequency-banding)
21. [Polarisation conventions: on-cardinal vs off-cardinal](#21-polarisation-conventions-on-cardinal-vs-off-cardinal)
22. [Precession / nutation handling](#22-precession--nutation-handling)
23. [Chunking and lazy-loading the sky model](#23-chunking-and-lazy-loading-the-sky-model)
24. [Testing and CI](#24-testing-and-ci)
25. [Examples shipped with WODEN](#25-examples-shipped-with-woden)
26. [Docker / Singularity images](#26-docker--singularity-images)
27. [Performance, accuracy, known limitations](#27-performance-accuracy-known-limitations)
28. [Mapping WODEN concepts onto RadioSim](#28-mapping-woden-concepts-onto-radiosim)

---

## 1. Project metadata, license, authorship

| Item | Value |
|---|---|
| Package name | `wodenpy` (the Python wrapper); the compiled libraries are `libwoden_float.so`, `libwoden_double.so`, `libuse_everybeam.so` |
| C/C++/CUDA `project()` version (top-level `CMakeLists.txt`) | `WODEN_VERSION 2.7` |
| Python `setup.py` version | `2.7.0` |
| Python `wodenpy/__init__.py` fallback | `"2.6.0-alpha"` (used only if `importlib.metadata.version("wodenpy")` fails) |
| Author | Jack L. B. Line (`jack.line@curtin.edu.au`, ORCID `0000-0002-9130-5920`) |
| Affiliations | International Centre for Radio Astronomy Research (ICRAR), Curtin Institute of Radio Astronomy, ARC CoE for All-Sky Astrophysics in 3D (ASTRO-3D) |
| Licence | Mozilla Public License 2.0 (`LICENSE`) |
| Citation | `Line, J. L. B., 2022, JOSS 7(69), 3676` (`joss_paper/paper.md`, `paper.bib`) |
| Languages | C (`src/*.c`), C++ (`src/*.cc`), CUDA / HIP (`src/*.cpp`), Python ≥ 3.8 |
| GPU support | NVIDIA CUDA (production), AMD HIP/ROCm (experimental, since v2.2 — courtesy of Marcin Sokolowski / PaCER-BLINK) |
| CPU fallback | Full CPU equivalents of every GPU function (since v2.5; flag `--cpu_mode`) |
| Build | CMake ≥ 3.21; produces dual precision (float / double) shared libraries that Python loads via `ctypes` |
| Documentation | Sphinx + Doxygen + Breathe; rendered at `https://woden.readthedocs.io/` |
| Status | Beta. JLBLine has stepped away from astronomy as of mid-2024 and accepts bug fixes / community PRs but isn't actively developing new features. |

`setup.py` registers a custom `gitinfo` build command that captures `git describe --always`, the date of the latest commit, and the current branch, writing them into `wodenpy/wodenpy_gitinfo.txt` so that the runtime can report what was installed (`wodenpy.wodenpy_setup.git_helper.retrieve_gitdict`). Five executable `scripts/*.py` are installed onto the user's `PATH`: `run_woden.py`, `add_woden_uvfits.py`, `concat_woden_uvfits.py`, `woden_uv2ms.py`, `add_instrumental_effects_woden.py`.

`requirements.txt` (see file) lists the Python runtime needs; the most load-bearing are `numpy`, `astropy`, `pyuvdata`, `palpy` (via the IAU `PAL` library — used for precession), `erfa`, `python-casacore` (only when EveryBeam is in use), `psutil` (CPU detection), `binpacking` (sky-chunk scheduler), `importlib_resources`, `line_profiler`.

Optional / configurable runtime dependencies (loaded only when invoked):

- `mwa_hyperbeam` — Rust-implemented MWA FEE primary-beam library (`https://github.com/MWATelescope/mwa_hyperbeam`). Required at compile time.
- `EveryBeam ≥ 0.7.2` + `casacore ≥ 3.7.0` — for LOFAR / OSKAR / EveryBeam-MWA primary beams.
- `pygdsm`, `pyradiosky`, `pyuvdata` — used by example notebooks for diffuse sky models.

---

## 2. What WODEN is and is not — scientific overview

WODEN is, in one sentence, a *direct discrete RIME summation* engine. It does NOT do gridding, FFTs, deconvolution, calibration, imaging, or selfcal — it answers the question *"if the sky looked like this catalogue, what would my interferometer measure?"* and produces complex visibilities in a `uvfits` file.

It IS:

- a low-frequency radio-interferometer simulator (≲ ~300 MHz design point, but the maths is frequency-agnostic);
- GPU-first: every visibility-generating kernel exists on the GPU, with CPU mirrors for cluster nodes that don't have a GPU;
- discretised-sky based: every astrophysical model gets converted to a list of point sources, elliptical Gaussians, or shapelet basis-function sums;
- fully polarised IQUV with Stokes V supporting power-law / curved-power-law / polarisation-fraction / list, and linear polarisation supporting power-law / curved-power-law / polarisation-fraction / two-list / single-P-list with rotation measure;
- multi-telescope: bespoke array layouts in (East, North, Height) plus Earth lat/long/altitude;
- multi-beam: Gaussian, EDA2 analytic dipole, MWA RTS-analytic, MWA FEE coarse, MWA FEE interpolated, EveryBeam OSKAR / LOFAR / MWA, pyuvdata-UVBeam MWA, pyuvdata-UVBeam HERA, and a no-beam option (gain = 1, leakage = 0);
- output in MIRIAD / AIPS-style `uvfits` with antenna table, multiple coarse-bands, and an `IAUORDER` header keyword that flags whether the first polarisation is XX = N-S (IAU) or XX = E-W (MWA convention);
- fully precessed: rather than precess every sky source forward, WODEN precesses the *array* and the LST back to J2000 to match the source catalogue, using the same `palPrenut` matrix as the RTS;
- lazy-loading: a sky model with > 25 million components is read in chunked sets that fit on the GPU.

It is NOT:

- a w-projection or w-stacking imager (it's a measurement-equation generator);
- a Fourier-on-the-sky / m-mode method (compare: `RIMEz`, which projects onto spherical harmonics);
- ionospherically corrupted (no Z Jones term — the user must add ionospheric effects post-hoc, e.g. via `add_instrumental_effects_woden.py` or external software);
- a near-field / Fresnel simulator;
- coupled-element-aware (apart from FEE and EveryBeam taking embedded patterns; the array is otherwise treated as an independent collection of identical or per-station beams).

### 2.1 WODEN's scientific niche

The JOSS paper (`joss_paper/paper.md`) sets out the pitch directly:

> *"`WODEN` is designed to simulate the response of a class of telescope known as an interferometer ... `WODEN` works with input Stokes I, Q, U, V polarisations as a sky model, simulating telescopes with dual linear polarisations, and outputting linear Stokes polarisations. … An interferometer creates visibilities V by cross-correlating signals detected between pairs of antennas or dishes (baselines), described by coordinates u, v, w. Each visibility is sensitive to the entire sky, directions of which we describe by the direction cosines l, m, n."*

The accuracy / speed quantification reported in the paper:

| Mode | Bit precision | Worst-case fractional error | GTX-1080-Ti runtime, 207 k pt + 1.2 k Gaussian + 62 shapelet (10 400 basis), 14 t × 80 ν, MWA FEE | V100 runtime |
|---|---|---|---|---|
| `woden_float` | 32-bit (mostly) + 64-bit (selected) | ≲ 0.2 % at b ≲ 10 km | 10 min 39 s | 4 min 35 s |
| `woden_double` | 64-bit (mostly) | ≲ 2 × 10⁻⁶ % | 55 min 46 s | 5 min 55 s |

Earlier (v1.0) was fully 32-bit and showed up to a few-percent error on long baselines. From v1.1 onward both `woden_float` and `woden_double` are shipped; the precision is selected with `--precision=float|double` on `run_woden.py`. The numeric type in C is gated by the macro `user_precision_t`:

```c
// include/woden_precision_defs.h
#ifdef DOUBLE_PRECISION
  typedef double user_precision_t;
  typedef double _Complex user_precision_complex_t;
#else
  typedef float user_precision_t;
  typedef float _Complex user_precision_complex_t;
#endif
```

The two libraries `libwoden_float.so` and `libwoden_double.so` are produced from the same source by adding `-DDOUBLE_PRECISION` to the double build (see `CMakeLists.txt:264-272`).

---

## 3. The RIME and how WODEN evaluates it

Ignoring the antenna response, the RIME for a single Stokes parameter `s` is a discrete inverse Fourier transform across a discrete sky:

```
V_s(u_i, v_i, w_i) = Σ_j  S_s(l_j, m_j) · exp[-2π i (u_i l_j + v_i m_j + w_i (n_j - 1))]
```

with `n_j = sqrt(1 - l_j² - m_j²)`. Once the dual-polarisation Jones matrix `J_p(l, m) = [[g_x, D_x], [D_y, g_y]]` is included for antennas 1 and 2 in a baseline, the four cross-correlations are formed by an outer product:

```
[V_XX, V_XY, V_YX, V_YY]ᵀ
   = J_1 ⊗ J_2*  ·  M  ·  [V_I, V_Q, V_U, V_V]ᵀ
```

with `M` the Stokes-to-instrumental "mixing matrix". For 0/90° (on-cardinal) dipoles:

```
M = [[1,  1,  0,  0],
     [0,  0,  1,  i],
     [0,  0,  1, -i],
     [1, -1,  0,  0]]
```

For 45/135° (off-cardinal) dipoles, encoded in `BeamGroups.off_cardinal_beam_values` and applied at the C level when `woden_settings->off_cardinal_dipoles == 1`:

```
M = [[1,  0,  1,  0],
     [0, -1,  0,  i],
     [0, -1,  0, -i],
     [1,  0, -1,  0]]
```

Note the *sign* convention. WODEN's exponent is `-2π i (...)` for the forward computation. The JOSS paper footnote explicitly observes that, after extensive cross-checking with other simulators and imaging, the *positive* sign convention is what produces the correct outputs; see `joss_paper/paper.md` footnote 1. This is consistent with most "imaging-positive" conventions. Internally WODEN uses XX = N-S, XY, YX, YY = E-W (the IAU convention) all the way through the C code, and only re-orders to XX = E-W on uvfits output unless `--IAU_order` is passed.

### 3.1 Gaussian and shapelet visibility envelopes

Following the RTS, both Gaussians and shapelets are inserted as multiplicative *visibility envelopes* `ξ_j(u_i, v_i)` that sit inside the Σ over the sky:

```
V(u_i, v_i, w_i) = Σ_j  ξ_j(u_i, v_i) · S(l_j, m_j) · exp[-2π i (u_i l_j + v_i m_j + w_i(n_j - 1))]
```

For an elliptical Gaussian with major axis `θ_maj`, minor axis `θ_min`, and PA `φ_PA`,

```
ξ_j  = exp[ -π² / (4 ln 2) · (k_x² θ_maj² + k_y² θ_min²) ],
k_x  = cos(φ_PA) v_i + sin(φ_PA) u_i,
k_y  = -sin(φ_PA) v_i + cos(φ_PA) u_i.
```

For a shapelet with basis-function set `{(p_k, p_l)}` and coefficients `C_{p_k, p_l}`,

```
ξ_j  = Σ_{p_k+p_l < p_max}  C_{p_k, p_l} · B̃_{p_k, p_l}(k_x, k_y),
k_x  = (π / sqrt(2 ln 2)) · [cos(φ_PA) v_{i,j} + sin(φ_PA) u_{i,j}],
k_y  = (π / sqrt(2 ln 2)) · [-sin(φ_PA) v_{i,j} + cos(φ_PA) u_{i,j}].
```

Crucially, shapelet `u_{i,j}, v_{i,j}` are *per-shapelet* — they're rotated into the frame whose phase centre is the shapelet's own RA/Dec (because that's the centre that SHAMFI fits about). This is calculated by `calc_uv_shapelet_cpu` / `calc_uv_shapelet_gpu` and stored in `calc_visi_inouts->u_shapes` / `v_shapes`.

The basis functions `B̃_{p_k, p_l}` are read from a pre-tabulated 1-D look-up table `sbf` of `sbf_N=101` orders, each sampled `sbf_L=10001` times across `−500 ≤ x ≤ 500` at `Δx=0.01` (centred on `sbf_c=5000`) — `include/constants.h`. Each shapelet is then assembled by interpolating along that 1-D table for both `k_x` and `k_y`, and scaling by `β_maj`, `β_min`. The Python implementation (`wodenpy/use_libwoden/shapelets.py`) replicates this via numpy: `Hermite(n)(x) · exp(-x²/2) / sqrt(β · 2ⁿ · n!)`.

### 3.2 Spectral models

Stokes I uses *one of three* spectral models per component (`enum e_flux_type` in `include/woden_struct_defs.h`):

```c
typedef enum { POWER_LAW=0, CURVED_POWER_LAW, LIST } e_flux_type;
```

Power law:                  `S_i = S_0 · (ν_i / ν_0)^α`
Curved power law:           `S_i = S_0 · (ν_i / ν_0)^α · exp(q · ln²(ν_i / ν_0))`  (Callingham et al. 2017 Eq. 2)
List:                       linear interpolation in **log-flux vs log-frequency** space, *unless* a flux dips below zero in which case linear interpolation switches to *linear* space across the zero-crossing. This is intentional and tested against 21-cm power-spectrum behaviour.

The reference frequency for power law / curved-power-law is hard-coded to `200 MHz` (`#define REF_FREQ 200000000.0` in `constants.h`). Default spectral index when none is given is `−0.8` (`#define DEFAULT_SI -0.8`).

Stokes V supports `power_law`, `curved_power_law`, `polarisation fraction` (V = Π·I), and `list`. Linear polarisation supports the same four plus `p-list` (a list of `P(λ)` that is then split into Q and U by the rotation-measure model). The linear-polarisation expression is

```
P(λ) = Q + i U = P(λ) · exp[ 2 i (χ_0 + φ_RM λ²) ]
```

so independently `Q = P(λ) cos(2 χ_0 + 2 φ_RM λ²)` and `U = P(λ) sin(...)`. Lists for Q and U separately are supported (extrapolated independently), as is a single list for `P(λ)` with RM applied.

### 3.3 Where this all lives in code

| Concern | C/CUDA | Python |
|---|---|---|
| Overall RIME loop | `src/calculate_visibilities_common.c::calculate_visibilities` | — |
| Per-component-type orchestrator | `src/calculate_visibilities_common.c::calculate_component_visis` | — |
| u, v, w | `fundamental_coords_cpu.c::calc_uvw_cpu`, `fundamental_coords_gpu.cpp::calc_uvw_gpu` | — |
| l, m, n | `calc_lmn_cpu`, `calc_lmn_gpu`, `calc_lmn_for_components_*` | — |
| Spectral extrapolation | `extrap_power_laws_stokes{I,V,linpol}_*`, `extrap_curved_power_laws_*`, `extrap_list_fluxes_*`, `polarisation_fraction_{stokesV,linpol}_*`, `apply_rotation_measure_*` (in `source_components_{cpu,gpu}.{c,cpp}`) | mirror Python copies in `wodenpy/skymodel/...` for sanity testing |
| Beam-direction calc | `wodenpy/use_libwoden/...` and within `source_component_common` | `run_woden.py` & `wodenpy/skymodel/read_fits_skymodel.py` for UVBeam / EveryBeam |
| Mixing matrix → V_XX/XY/YX/YY | Inside `calc_visi_point_or_gauss_*`, `calc_visi_shapelets_*` | — |

---

## 4. Repository layout

```
WODEN/
├── CMakeLists.txt                      # top-level build (≈330 lines); splits into production/debug/test
├── CONTRIBUTION_GUIDE.md
├── DEVELOPERS_GUIDE.md                 # Jack's deep-dive for inheritors (~1.5 kLOC)
├── LICENSE                             # Mozilla Public License 2.0
├── README.md                           # high-level intro + change log per minor version
├── docker/                             # Dockerfile_cuda, Dockerfile_setonix, make_docker_image.sh, fetch_iers_data.py
├── docs/
│   ├── doxygen/                        # Doxyfile + graph-Doxyfile (C/C++/CUDA API extraction)
│   └── sphinx/                         # source for readthedocs
│       ├── installation/installation.rst
│       ├── operating_principles/      # visibility_calcs / skymodel / primary_beams / frequency_wording
│       ├── examples/                   # MWA EoR1, FornaxA, EDA2 Haslam, dipole_ampflags, polarisation,
│       │                               # lofar_lotss, lofar_lba_ncp, hera_sim, relocate_everybeam_array
│       ├── API_reference/              # Breathe-driven C/C++/GPU/Python reference
│       ├── code_graphs/                # call-graphs auto-generated by Doxygen
│       ├── scripts/                    # rst snippets that argparse-extension turns into argument tables
│       ├── testing/                    # cmake_testing per-module rsts + script_testing + everybeam_testing
│       └── developers_guide/           # (mirror of DEVELOPERS_GUIDE.md)
├── examples/                           # runnable MWA EoR1, FornaxA, EDA2_haslam, dipole_ampflags, LOFAR_*,
│                                       # HERA_sim, polarisation, metafits, relocate_everybeam_array, etc.
├── externals/                          # third-party submodules (Unity test framework, pulled in by ctest)
├── include/                            # ALL .h files for C/C++/CUDA headers (24 files, ~6 kLOC)
│   ├── woden.h                         # public API: int run_woden(woden_settings_t*, visibility_set_t*,
│   │                                   #                          source_catalogue_t*, array_layout_t*, sbf*)
│   ├── woden_struct_defs.h             # 448 lines: every struct used by C, CUDA, EveryBeam, hyperbeam
│   ├── woden_precision_defs.h          # user_precision_t typedef switch
│   ├── constants.h                     # angle/astro/RTS/shapelet constants
│   ├── beam_settings.h
│   ├── calculate_visibilities_common.h  # gateway header for CPU/GPU dispatch
│   ├── calculate_visibilities_cpu.h
│   ├── calculate_visibilities_gpu.h
│   ├── source_components_common.h      # CPU/GPU dispatch
│   ├── source_components_cpu.h         # 1.1 kLOC of header — CPU primitives
│   ├── source_components_gpu.h         # 1.6 kLOC — every kernel signature
│   ├── fundamental_coords_cpu.h
│   ├── fundamental_coords_gpu.h
│   ├── primary_beam_cpu.h
│   ├── primary_beam_gpu.h
│   ├── shapelet_basis.h
│   ├── visibility_set.h
│   ├── call_everybeam.h                # tiny C++ shim header
│   ├── call_everybeam_c.h              # C-callable interface used by woden.c / Python ctypes
│   ├── gpu_macros.h                    # CUDA-vs-HIP macros (gpuMalloc, gpuFree, gpuDeviceSynchronize, …)
│   ├── gpucomplex.h                    # vector / complex helpers usable from CUDA & HIP
│   ├── hyperbeam_error.h               # tiny error-reporting shim
│   ├── logger.h                        # ctypes-callable log forwarding
│   └── check_compilation_flags.h       # one-line: bool check_for_everybeam_compilation()
├── src/                                # implementation (15 files; ~9 kLOC C, 4 kLOC GPU)
│   ├── woden.c                         # 95 LOC: int run_woden(...) — outer band loop + FEE init/teardown
│   ├── calculate_visibilities_common.c # 500 LOC: core per-band orchestration
│   ├── calculate_visibilities_cpu.c    # 114 LOC: CPU memory mgmt
│   ├── calculate_visibilities_gpu.cpp  # 227 LOC: GPU memory mgmt
│   ├── fundamental_coords_cpu.c        # 116 LOC
│   ├── fundamental_coords_gpu.cpp      # 269 LOC
│   ├── primary_beam_cpu.c              # 646 LOC: Gaussian/EDA2/MWAanaly/hyperbeam-CPU/UVBeam-bridge
│   ├── primary_beam_gpu.cpp            # 784 LOC: same on GPU + hyperbeam.cu wrappers
│   ├── source_components_common.c      # 555 LOC: CPU/GPU dispatch + Stokes extrapolation glue
│   ├── source_components_cpu.c         # 1408 LOC: full RIME kernel mirrors on CPU
│   ├── source_components_gpu.cpp       # 2586 LOC: every CUDA kernel
│   ├── call_everybeam.cc               # 527 LOC: C++ EveryBeam wrappers + casacore glue
│   ├── shapelet_basis.c                # 213 LOC: builds the B(n,β=1) lookup table (legacy; Python now does it)
│   ├── visibility_set.c                # 188 LOC: malloc/free for Visi_Set + binary dump (legacy debug)
│   ├── beam_settings.c                 # 90 LOC: fill_primary_beam_settings()
│   ├── check_compilation_flags.c       # 8 LOC: returns true when -DHAVE_EVERYBEAM was set
│   ├── hyperbeam_error.c               # 22 LOC: hyperbeam → log
│   └── logger.c                        # 15 LOC
├── wodenpy/                            # Python package (≈ 12 kLOC)
│   ├── __init__.py                     # version handling
│   ├── array_layout/                   # create_array_layout.py (467 LOC) + precession.py (54 LOC)
│   ├── observational/                  # calc_obs.py — JD calc, GST0, DEGPDY, UT1−UTC, LST
│   ├── phase_rotate/                   # remove_phase_track.py — undo W phase
│   ├── primary_beam/                   # use_everybeam.py (1491 LOC), use_uvbeam.py (431 LOC)
│   ├── skymodel/                       # read_fits_skymodel.py (1253), chunk_sky_model.py (1162),
│   │                                   # woden_skymodel.py (927), read_yaml_skymodel.py (391),
│   │                                   # read_text_skymodel.py (353), read_skymodel.py (200)
│   ├── use_libwoden/                   # ctypes mirrors of every C struct
│   │   ├── skymodel_structs.py         # 1360 LOC of dynamic ctypes structs
│   │   ├── woden_settings.py           # 567 LOC
│   │   ├── visibility_set.py           # 365 LOC
│   │   ├── shapelets.py                # 121 LOC numpy-built sbf
│   │   ├── beam_settings.py            # BeamTypes + BeamGroups enums
│   │   ├── array_layout_struct.py      # ctypes Array_Layout
│   │   ├── create_woden_struct_classes.py  # one-stop class factory
│   │   └── use_libwoden.py             # ctypes.cdll loader; check-everybeam guard
│   ├── uvfits/wodenpy_uvfits.py        # 457 LOC: writes the AIPS-style uvfits
│   └── wodenpy_setup/                  # run_setup.py (1301 LOC, full argparse), woden_logger.py,
│                                       # git_helper.py
├── scripts/                            # 9 standalone CLI utilities; install onto $PATH via setup.py
├── cmake_testing/                      # CTest tests for every C/CUDA module + Python tests
├── cmake_casacore/                     # FindCasacore.cmake (lifted from WSClean)
├── check_documentation/                # verifies sphinx vs doxygen are still in sync
├── coverage_outputs/                   # static html coverage snapshot
├── joss_paper/                         # JOSS publication source: paper.md, paper.bib, figures, paper.pdf
├── templates/                          # cluster-specific install scripts (CUDA + AMD)
├── test_installation/                  # end-to-end install verification tests (run after pip install)
├── setup.py                            # custom build_py + gitinfo command
├── requirements.txt                    # Python runtime dependencies
└── README.md
```

The Python package is roughly 12 kLOC; the compiled (C + C++ + CUDA/HIP) source is roughly 13 kLOC; the documentation is on top of that. The single biggest file is `src/source_components_gpu.cpp` (~2.6 kLOC), which holds every CUDA kernel that produces visibilities.

---

## 5. Build system: CMake, dual precision, CUDA/HIP, EveryBeam, hyperbeam

The top-level `CMakeLists.txt` (`cmake_minimum_required(VERSION 3.21)`) handles three axes of variability:

1. **GPU back-end** — `USE_CUDA=ON` (default) or `USE_HIP=ON`. If `USE_HIP` is set, `USE_CUDA` is auto-disabled. CUDA is compiled with `nvcc` (`CMAKE_CUDA_STANDARD 17`, flag `-D__NVCC__`). HIP is compiled with `hipcc`, with `-D__HIP_PLATFORM_AMD__ -D__HIPCC__` defines and an optional `--offload-arch=${HIP_ARCH}`. The CUDA architecture defaults are taken from the env var `CUDAARCHS` if set (or `CMAKE_CUDA_ARCHITECTURES`), otherwise CMake's default. A commented-out wide list `60;61;70;75;80;86;89;90` is left in the file for power users.
2. **Float vs double** — *both* are always built. Two GPU libraries (`wodenGPU_float`, `wodenGPU_double`) and two C libraries (`woden_float`, `woden_double`) are produced from the same source by adding `-DDOUBLE_PRECISION` to the double-build target. After link, `libwoden_{float,double}.so` are *copied into* `wodenpy/` so that `pip install` ships them as `package_data`. (See `setup.py:120-124`: `package_data={"wodenpy" : ["libwoden_float.so", "libwoden_double.so", "libuse_everybeam.so", 'wodenpy_gitinfo.txt', 'bandpass_1kHz.txt']}`.)
3. **EveryBeam (optional)** — If `find_package(EveryBeam NO_MODULE)` finds it *and* casacore *and* the `aocommon` headers (e.g. `matrix2x2.h`) *and* the bespoke `EBEAM_MWA` source path holding `beam2016implementation.h`, a separate library `libuse_everybeam.so` is built from `src/*.cc` and from EveryBeam's `beam2016implementation.cc`, linked against EveryBeam and casacore, and `libwoden_*.so` then link against it with `-DHAVE_EVERYBEAM`. If any of these are missing, an empty `libuse_everybeam.so` text-file dummy is created so that `setup.py` doesn't choke on the missing data file. The `check_for_everybeam_compilation()` C function (in `src/check_compilation_flags.c`) reports the runtime status to Python via ctypes, so Python errors gracefully if a user tries to use a feature that wasn't compiled in.

**Important environment hint:** if `EVERYBEAM_FOUND` is true, CMake also pulls in
- `EBEAM_INSTALL` (location of the *installed* EveryBeam), so it can find `lib`/headers,
- `EBEAM_ROOT` (location of the EveryBeam *source*), so it can grab `external/aocommon/include/aocommon/` and `cpp/mwabeam/`, neither of which is installed by EveryBeam's own `make install`,
- `CASACORE_ROOT_DIR`, used by the local `cmake_casacore/FindCasacore.cmake` (forked from WSClean).

The `-march=native` flag is forced on the EveryBeam shim object specifically so that aocommon's `Matrix2x2` template uses the right SIMD class — without it you get either link or runtime failures.

The build also accepts `TARGET_GROUP=production|debug|test`. With `test`, CMake finds Unity (`UNITY_ROOT`), compiles a `Unity` static library with `-DUNITY_INCLUDE_DOUBLE -DUNITY_DOUBLE_PRECISION=1e-12`, and then descends into `cmake_testing/` to register all the CTest cases.

### 5.1 Dependencies

| Layer | Library | Version | Comment |
|---|---|---|---|
| Build | CMake | ≥ 3.21 | `find_package(EveryBeam NO_MODULE)` requires modern CMake |
| Build | rust | latest stable | Needed for `mwa_hyperbeam` cargo build |
| Build | NVIDIA CUDA | tested on 11.x and 12.x | OR AMD ROCm for HIP path |
| Compiled | mwa_hyperbeam | ≥ 0.10.x (runtime DLL `libmwa_hyperbeam.so` + header `mwa_hyperbeam.h`) | Must be built with `--features=cuda,hdf5-static` for CUDA, or `--features=hip,hdf5-static` for HIP |
| Compiled (optional) | EveryBeam | ≥ 0.7.2 (note: previous WODEN versions used a Jack-Line fork; 2.7+ uses upstream) | Must compile with `-DBUILD_WITH_PYTHON=ON` |
| Compiled (optional) | casacore | ≥ 3.7.0 | `sudo apt install libcasacore-dev` is too old |
| Compiled (optional) | HDF5 | ≥ 1.10 | Required by hyperbeam |
| Python ≥ 3.8 | numpy, astropy, pyuvdata, palpy, erfa, scipy, h5py | Latest | |
| Python | line_profiler | optional | for `--profile` |
| Python | psutil | latest | Used to detect `args.num_threads` default |
| Python | binpacking | latest | Used by sky-chunk scheduler |

`mwa_hyperbeam` is sourced from `https://github.com/MWATelescope/mwa_hyperbeam`. It needs a Rust toolchain. Build with:

```bash
git clone https://github.com/MWATelescope/mwa_hyperbeam.git
cd mwa_hyperbeam
export HYPERDRIVE_CUDA_COMPUTE=60        # your GPU's compute capability
cargo build --locked --release --features=cuda,hdf5-static
```

For the HIP/AMD path:

```bash
export HYPERBEAM_HIP_ARCH=gfx90a
cargo build --locked --release --features=hip,hdf5-static
```

The MWA FEE coefficient HDF5 file is *not* shipped — it's downloaded post-install (`http://ws.mwatelescope.org/static/mwa_full_embedded_element_pattern.h5`) and pointed at via env var `MWA_FEE_HDF5`. The interpolated file is `MWA_embedded_element_pattern_rev2_interp_167_197MHz.h5` and is referenced by `MWA_FEE_HDF5_INTERP`.

### 5.2 Compiling

Minimal happy-path:

```bash
git clone https://github.com/JLBLine/WODEN.git
cd WODEN && mkdir build && cd build
cmake ..      # may need -DHBEAM_INC=... -DHBEAM_LIB=... -DEBEAM_INSTALL=... -DEBEAM_ROOT=... -DCASACORE_ROOT_DIR=...
make -j 4
cd ..
pip install -r requirements.txt
pip install .
```

The README warns: *"Even if the code compiled, if your GPU has a compute capability < 5.1, newer versions of `nvcc` won't compile code that will work."* Set `CUDAARCHS` explicitly to be safe.

### 5.3 Docker images

`docker/make_docker_image.sh` builds the standard images. Available, *tested* tags:

| Tag | Architecture | Cluster tested |
|---|---|---|
| `jlbline/woden-2.3:cuda-60` | NVIDIA, compute 6.0 | Swinburne OzStar |
| `jlbline/woden-2.6:cuda-80` | NVIDIA, compute 8.0 | Swinburne Ngarrgu Tindebeek |
| `jlbline/woden-2.6:cuda-multi` | NVIDIA, computes 60–86 | Generic |
| `jlbline/woden-2.3:setonix` | AMD HIP | Pawsey Setonix |

These images bundle the MWA FEE coefficient files, but you must still pass `--gpus all` (Docker) or `--nv` (Singularity) and supply env-vars for `astropy` storage (`XDG_CONFIG_HOME`, `XDG_CACHE_HOME`).

A bundled helper `docker/fetch_iers_data.py` can be run on the head node of a cluster to refresh `astropy`'s IERS data (`Earth orientation parameters`) without internet access on compute nodes.

---

## 6. Constants, units, and numeric conventions (`include/constants.h`)

The numeric constants are exposed as `#define` macros so they're available in both C and CUDA without requiring a runtime lookup. The full set:

| Macro | Numeric value | Purpose |
|---|---|---|
| `DH2R` | `0.26179938779914943653855361527329190701643078328126` | Hour-angle (hours) → radians |
| `DD2R` | `0.017453292519943295769236907684886127134428718885417` | Degrees → radians |
| `DS2R` | `7.272205216643039903848711535369…e-5` | Sidereal-seconds → radians |
| `VELC` | `299792458.0` | Speed of light (m/s) |
| `SOLAR2SIDEREAL` | `1.00274` | Solar-rate → sidereal-rate conversion |
| `DEFAULT_SI` | `-0.8` | Default Stokes-I spectral index |
| `MWA_LAT` | `-26.703319405555554` | MWA latitude (deg) |
| `MWA_LAT_RAD` | `-0.4660608448386394` | MWA latitude (rad) |
| `M_PI_2_2_LN_2` | `7.11941466249375271693034` | π² / (2 ln 2) — Gaussian envelope |
| `SQRT_M_PI_2_2_LN_2` | `2.668223128318498282851579` | sqrt(π² / (2 ln 2)) — shapelet envelope |
| `FWHM_FACTOR` | `2.35482004503` | σ → FWHM |
| `sbf_N` | `101` | Shapelet basis-function orders 0…100 |
| `sbf_L` | `10001` | Samples per basis function |
| `sbf_c` | `5000` | Index where x = 0 |
| `sbf_dx` | `0.01` | Sampling resolution in basis-function table |
| `MAX_CHUNKING_SIZE` | `10_000_000_000` | Max per-chunk component count (default `--chunking_size=1e10`) |
| `NUM_DIPOLES` | `16` | MWA dipoles per tile |
| `DQ` | `435e-12 · VELC` | MWA beamformer delay quantum, in metres |
| `MAX_POLS` | `4` | XX / XY / YX / YY |
| `N_COPOL` | `2` | Two co-pols (X, Y) before correlation |
| `MWA_DIPOLE_HEIGHT` | `0.3` | MWA dipole above ground (metres). Note the docstring says "0.29" but the value is `0.3`. |
| `MWA_DIPOLE_SEP` | `1.1` | MWA dipole spacing (metres) |
| `INITIAL_NUM_COMPONENTS` | `10000` | Initial array size for component bookkeeping (doubled on overflow) |
| `INITIAL_NUM_FLUXES` | `100` | Initial size for list-flux arrays |
| `REF_FREQ` | `200_000_000.0` | Power-law reference frequency (Hz) — i.e. all `NORM_COMP_PL` columns are 200-MHz fluxes |

Python mirrors many of these in `wodenpy/use_libwoden/use_libwoden.py`, `wodenpy/array_layout/create_array_layout.py`, and `wodenpy/use_libwoden/woden_settings.py`. The Python defaults for the LOFAR / HERA centres are taken from `pyuvdata.telescopes.known_telescope_location` (see `wodenpy/wodenpy_setup/run_setup.py` line ~39).

---

## 7. The core C/CUDA struct universe (`woden_struct_defs.h`)

WODEN's data flow is dominated by five C structs that propagate through every code path. Each has an exact ctypes mirror that Python builds *dynamically* (so that `user_precision_t` can switch between `c_float` and `c_double` to match the loaded `libwoden_*.so`). The class factory is `wodenpy/use_libwoden/create_woden_struct_classes.py`.

### 7.1 `enum`s

```c
typedef enum {POINT=0, GAUSSIAN, SHAPELET}              e_component_type;
typedef enum {NO_BEAM=0, GAUSS_BEAM, FEE_BEAM, ANALY_DIPOLE,
              FEE_BEAM_INTERP, MWA_ANALY,
              EB_OSKAR, EB_LOFAR, EB_MWA,
              UVB_MWA, UVB_HERA}                        e_beamtype;
typedef enum {POWER_LAW=0, CURVED_POWER_LAW, LIST}      e_flux_type;
```

`BeamTypes` (Python) mirrors `e_beamtype`. `BeamGroups` (Python) bundles them:

| Group | Members | Meaning |
|---|---|---|
| `eb_beam_values` | EB_OSKAR, EB_LOFAR, EB_MWA | All EveryBeam variants |
| `eb_ms_beam_values` | EB_OSKAR, EB_LOFAR | Need a measurement set for the array layout |
| `azza_beam_values` | MWA_ANALY, FEE_BEAM, FEE_BEAM_INTERP, ANALY_DIPOLE, EB_MWA | Azimuth/zenith-angle is computed Python-side |
| `needs_MWA_delays` | MWA_ANALY, FEE_BEAM, FEE_BEAM_INTERP, EB_MWA | Need the 16-int delay vector |
| `needs_MWA_hdf5_path` | FEE_BEAM, FEE_BEAM_INTERP, EB_MWA | Need the FEE coefficients HDF5 |
| `hadec_beam_values` | GAUSS_BEAM, MWA_ANALY | Need HA/Dec computed |
| `python_calc_beams` | UVB_MWA, UVB_HERA | The beam is computed in Python before going to C |
| `uvbeam_beams` | UVB_MWA, UVB_HERA | (alias for the above) |
| `off_cardinal_beam_values` | (empty TODO) | Reserved for 45/135° dipole instruments (planned) |

### 7.2 `components_t`

A single COMPONENT-type bucket — one each for POINT, GAUSSIAN, SHAPELET inside a SOURCE. ~120 fields, key arrays:

- intrinsic — `ras[]`, `decs[]`
- power-law parameters — `power_ref_freqs[]`, `power_ref_stokesI[]`, `power_SIs[]`
- curved-power-law — `curve_ref_freqs[]`, `curve_ref_stokesI[]`, `curve_SIs[]`, `curve_qs[]`
- bookkeeping arrays mapping each model back to the component index — `power_comp_inds[]`, `curve_comp_inds[]`, `list_comp_inds[]`
- list-flux arrays — `list_freqs[]`, `list_stokesI/Q/U/V[]`, `num_list_values[]`, `list_start_indexes[]`, `total_num_flux_entires`
- per-output extrapolated fluxes — `extrap_stokesI/Q/U/V[]`
- shape/Gaussian — `shape_coeffs[]`, `n1s[]`, `n2s[]`, `majors[]`, `minors[]`, `pas[]`, `param_indexes[]`
- observational direction arrays (filled per time-step) — `azs[]`, `zas[]`, `para_angles[]`, `beam_has[]`, `beam_decs[]`, `num_primarybeam_values`
- beam-Jones arrays — `gxs[]`, `Dxs[]`, `Dys[]`, `gys[]` (all `user_precision_complex_t`)
- direction cosines — `ls[]`, `ms[]`, `ns[]`
- a *very* long list of polarisation arrays: `stokesV_pol_fracs`, `stokesV_power_*`, `stokesV_curve_*`, `stokesV_list_*`, `linpol_pol_fracs`, `linpol_power_*`, `linpol_curve_*`, `stokesQ_list_*`, `stokesU_list_*`, `linpol_p_list_*`, `rm_values`, `intr_pol_angle`, `linpol_angle_inds`, `n_*` counters and `do_QUV` flag

This is a *Struct of Arrays*, optimised for GPU coalesced memory access — kernels iterate components in a single dimension and read each contiguous array.

### 7.3 `source_t`

```c
struct source_t {
    char name[32];
    int n_comps, n_points, n_point_lists, n_point_powers, n_point_curves;
    int n_gauss, n_gauss_lists, n_gauss_powers, n_gauss_curves;
    int n_shapes, n_shape_lists, n_shape_powers, n_shape_curves, n_shape_coeffs;
    components_t point_components, gauss_components, shape_components;
};
```

Three `components_t` are nested per source so that point, Gaussian, and shapelet kernels can iterate independently. The component counts are split by spectral-model type because the CPU/GPU kernels for power, curve, list extrapolation are separate.

### 7.4 `source_catalogue_t`

```c
struct source_catalogue_t {
    int num_sources;
    int num_shapelets;
    source_t *sources;
};
```

Each *chunk* of the lazy-loaded sky model becomes one `source_t` inside the catalogue. WODEN iterates all of them in a serial loop in `calculate_visibilities` (there is intra-source parallelism on the GPU, but inter-source is serial).

### 7.5 `beam_settings_t`

Holds settings + initialised hyperbeam handles + EveryBeam handles:

| Field | Type | Meaning |
|---|---|---|
| `gauss_sdec`, `gauss_cdec`, `gauss_ha` | precision | Pointing of the Gaussian beam |
| `beam_FWHM_rad`, `beam_ref_freq` | precision / double | Gaussian beam parameters |
| `beamtype` | `e_beamtype` | Selected primary-beam type |
| `MWAFEE_freqs[]`, `num_MWAFEE` | precision array, int | Per-frequency hyperbeam initialised |
| `gpu_fee_beam` | `struct FEEBeamGpu *` | hyperbeam GPU handle |
| `fee_beam` | `struct FEEBeam *` | hyperbeam CPU handle |
| `hyper_error_str[100]` | char[] | Error-message buffer |
| `base_middle_freq` | double | Centre of the current coarse band |
| `hyper_delays[]` | `uint32_t *` | 16 delays in hyperbeam's expected format (×num_beams when using per-tile dipole amps) |
| `everybeam_telescope` | `struct Telescope *` | EveryBeam LOFAR/OSKAR telescope |
| `eb_mwa_tile_beam` | `struct Beam2016Implementation *` | EveryBeam MWA tile |

### 7.6 `visibility_set_t`

The output container for a single coarse band. The "all-steps" arrays are fastest-axis baseline, then frequency, then time, *flat*:

```c
struct visibility_set_t {
    user_precision_t *us_metres, *vs_metres, *ws_metres;
    double *allsteps_sha0s, *allsteps_cha0s, *allsteps_lsts;
    user_precision_t *allsteps_wavelengths;
    double *channel_frequencies;
    user_precision_t *sum_visi_XX_real, *sum_visi_XX_imag,
                     *sum_visi_XY_real, *sum_visi_XY_imag,
                     *sum_visi_YX_real, *sum_visi_YX_imag,
                     *sum_visi_YY_real, *sum_visi_YY_imag;
};
```

Auto-correlations (when `--do_autos` is enabled) are stored *after* the cross-correlations in each `sum_visi_*` array. Their `u, v, w` are zero (set by `set_auto_uvw_to_zero_*`).

### 7.7 `woden_settings_t`

The big settings bag (`include/woden_struct_defs.h`, lines ~296-358). Every field has a `:cvar:` documentation entry in `wodenpy/use_libwoden/woden_settings.py`. Notable fields:

- `lst_base`, `lst_obs_epoch_base` — LST at first time step, in radians, in J2000 and observation epoch respectively
- `ra0`, `dec0`, `sdec0`, `cdec0` — phase centre
- `num_baselines`, `num_ants`, `num_freqs`, `num_time_steps`, `num_bands`
- `frequency_resolution`, `base_low_freq`, `base_band_freq`, `coarse_band_width`
- `time_res`
- `band_nums[]` — which coarse bands to simulate
- `sky_crop_type`, `chunking_size`
- `beamtype`, `gauss_beam_FWHM`, `gauss_beam_ref_freq`, `gauss_ra_point`, `gauss_dec_point`
- `hdf5_beam_path`, `array_layout_file`, `array_layout_file_path`
- `latitude`, `latitude_obs_epoch_base`, `longitude` — Earth location
- `FEE_ideal_delays[]` — `16 × num_beams` ints
- `num_cross`, `num_autos`, `num_visis` — totals (`num_visis = num_cross + num_autos`)
- `do_precession` — bool
- `lsts[]`, `latitudes[]`, `mjds[]` — per-time arrays (different when precession is on)
- `do_autos`, `use_dipamps` — bool
- `mwa_dipole_amps[]` — `2 × num_ants × 16` doubles
- `single_everybeam_station` — bool; force every station to use the same beam
- `off_cardinal_dipoles` — bool
- `do_gpu` — bool (1 → GPU, 0 → CPU)
- `verbose`, `normalise_primary_beam`
- `beam_ms_path`, `eb_beam_ra0`, `eb_beam_dec0`

### 7.8 `array_layout_t`, `calc_visi_inouts_t`, `beam_gains_t`

`array_layout_t` carries `(X, Y, Z)` antenna positions plus `(X_diff, Y_diff, Z_diff)` baseline differences in metres, plus local east/north/height, latitude, lst_base, num_baselines, num_tiles. When `do_precession=1`, the differences are *time-dependent* (length `num_baselines × num_times`) because the array is rotated back to J2000 *per time step*.

`calc_visi_inouts_t` is a temporary working struct that gathers all the inputs and intermediate outputs for the visibility kernels. It contains pointers to `X_diff/Y_diff/Z_diff` (only used in the GPU path — the CPU path reads them from `array_layout_t` directly), `allsteps_*`, `u_metres/vs_metres/ws_metres` (in metres), `us/vs/ws` (in wavelengths), `freqs`, antenna→baseline mappings (`ant1_to_baseline_map`, `ant2_to_baseline_map`), the shapelet basis array `sbf`, and `u_shapes/v_shapes` for the per-shapelet rotated u, v.

`beam_gains_t` parallels `components_t` for the beam-only fields. When `use_twobeams=1` (currently triggered by `use_dipamps=1` for MWA, or by EveryBeam LOFAR/OSKAR with default per-station beams), each baseline can have a different beam at each end, and the antenna mapping arrays are followed when constructing the cross-correlation.

---

## 8. Top-level execution path: `run_woden` (in `src/woden.c`)

The top-level C entry-point is one function, `int run_woden(woden_settings_t*, visibility_set_t*, source_catalogue_t*, array_layout_t*, user_precision_t *sbf)`. It is *short* (~95 lines) — most work is delegated:

```c
beam_settings_t *beam_settings = fill_primary_beam_settings(woden_settings, woden_settings->lsts);

for (int band = 0; band < woden_settings->num_bands; band++) {
    int band_num = woden_settings->band_nums[band];
    double base_band_freq = ((band_num - 1) * woden_settings->coarse_band_width)
                          + woden_settings->base_low_freq;
    woden_settings->base_band_freq = base_band_freq;

    fill_timefreq_visibility_set(&visibility_sets[band], woden_settings,
                                  base_band_freq, woden_settings->lsts);

    if (FEE_BEAM || FEE_BEAM_INTERP) {
        beam_settings->base_middle_freq = base_band_freq + (woden_settings->coarse_band_width/2.0);
        new_fee_beam(woden_settings->hdf5_beam_path, &beam_settings->fee_beam);
    }

    calculate_visibilities(array_layout, cropped_sky_models, beam_settings,
                           woden_settings, &visibility_sets[band], sbf);

    if (FEE_BEAM || FEE_BEAM_INTERP) free_fee_beam(beam_settings->fee_beam);
}
return 0;
```

So `run_woden`:

1. Sets up the `beam_settings_t` once.
2. For each requested coarse band (`band_nums`):
   a. Computes the lower frequency edge (`(band_num − 1) × coarse_band_width + base_low_freq`),
   b. Populates per-time, per-frequency arrays in `visibility_set` (`sha0`, `cha0`, `lsts`, `wavelengths`, `channel_frequencies`),
   c. (FEE-only) initialises an MWA hyperbeam *CPU* handle for the current band centre,
   d. Hands off to `calculate_visibilities` (the actual work),
   e. (FEE-only) tears down the CPU hyperbeam handle.

The per-band design is the reason the user can split a simulation across multiple GPUs: launch `run_woden.py` once per band number and the outputs are independent `*_band01.uvfits`, `*_band02.uvfits`, … files which `concat_woden_uvfits.py` can later glue together along the frequency axis.

---

## 9. Visibility calculation orchestration (`calculate_visibilities_*`)

`src/calculate_visibilities_common.c::calculate_visibilities` is the function each band calls. ~500 lines, but the flow is:

1. **Decide whether each baseline gets one or two beams.** Default: `use_twobeams=0`, `num_beams=1` (all antennas share a beam). But:
   - if `woden_settings->use_dipamps == 1` (MWA per-tile dipole amplitudes), promote to `num_beams=num_ants`;
   - if EB_LOFAR or EB_OSKAR is in use *and* `single_everybeam_station != 1`, promote to `num_beams=num_ants`.
   This is recorded in `mem_beam_gains->use_twobeams`, and the antenna→baseline map arrays are filled.
2. **Allocate visibility outputs.** A single `chunk_visibility_set` is allocated to hold per-chunk results, plus a `mem_visibility_set` (which is GPU memory if `do_gpu=1`, otherwise a CPU-side copy).
3. **Allocate `calc_visi_inouts`.** Either `create_calc_visi_inouts_gpu` (GPU) or `create_calc_visi_inouts_cpu` (CPU). For GPU: copies `X_diff_metres`, `Y_diff_metres`, `Z_diff_metres`, `allsteps_*`, `channel_frequencies`, `sbf` into device memory; allocates GPU buffers for `u/v/w_metres`, `us/vs/ws`, shapelet rotated coords, and antenna mapping arrays.
4. **Initialise hyperbeam GPU.** If FEE_BEAM or FEE_BEAM_INTERP, populate `freqs_hz` and `hyper_delays` (`16 × num_beams`), call `new_gpu_fee_beam(fee_beam, freqs_hz, delays, mwa_dipole_amps, num_freqs, num_beams, num_amps, norm_to_zenith=1)`.
5. **Initialise EveryBeam.** If `HAVE_EVERYBEAM` and EB_LOFAR/EB_OSKAR/EB_MWA, call the appropriate constructor. `element_response_model` defaults to `"hamaker"` for LOFAR and `"skala40_wave"` for OSKAR. EB_MWA calls `load_everybeam_MWABeam` with the integer delays cast to double.
6. **Loop over sky chunks** (each chunk is a `source_t` inside `cropped_sky_models->sources[]`):
   a. If GPU: `copy_chunked_source_to_GPU(source)`. Else just alias `mem_chunked_source = source`.
   b. Zero out `chunk_visibility_set` and `mem_visibility_set`.
   c. Compute `u, v, w` for *all* baseline × time × freq combinations:
       - GPU: `calc_uvw_gpu(d_X_diff, d_Y_diff, d_Z_diff, d_u_metres, …, d_us, d_vs, d_ws, d_allsteps_wavelengths, d_allsteps_cha0s, d_allsteps_sha0s, woden_settings)`
       - CPU: `calc_uvw_cpu(...)`
   d. If `do_autos==1`, set the auto-correlation `u/v/w` to zero in both metres and wavelengths.
   e. **For each component type** (POINT, GAUSSIAN, SHAPELET):
       - For shapelets, also compute `u_shapes`, `v_shapes` (per-shapelet rotated u, v with that shapelet's RA/Dec as phase centre).
       - Call `calculate_component_visis(comptype, mem_calc_visi_inouts, channel_frequencies, woden_settings, beam_settings, source, mem_chunked_source, mem_visibility_set, num_beams, use_twobeams, do_gpu)`.
   f. **Copy outputs back.** GPU: `copy_gpu_visi_set_to_host`. CPU: a flat memcpy.
   g. **Accumulate into the per-band `visibility_set`** by adding to each `sum_visi_*_real/imag` slot.
7. **Tear down** EveryBeam handles, FEE GPU beam handle, GPU memory, etc.

`calculate_component_visis` itself:

1. Copies the right `components_t` slot out of the chunked source.
2. Allocates `beam_gains_t`.
3. Calls `source_component_common`:
   - Calculates `(l, m, n)` for each component.
   - Calls every flux-extrapolation function applicable to this component-type's flux models (`extrap_power_laws_stokesI/V/linpol_*`, `extrap_curved_power_laws_*`, `extrap_list_fluxes_*`, `polarisation_fraction_stokesV/linpol_*`, `apply_rotation_measure_*`).
   - Calls the right primary-beam wrapper depending on `beam_settings->beamtype`: `wrapper_calculate_gaussian_beam_*`, `wrapper_calculate_analytic_dipole_beam_*`, `wrapper_calculate_RTS_MWA_analytic_beam_*`, `wrapper_run_hyperbeam_*`, or pulls Python-computed UVBeam values out of `chunked_source->{point,gauss,shape}_components.gxs/Dxs/Dys/gys`.
4. Calls `calc_visi_point_or_gauss_*` (POINT or GAUSSIAN) or `calc_visi_shapelets_*` (SHAPELET):
   - Iterates over (baseline × frequency × time) outer; over (component or shape-coefficient) inner.
   - Reads beam gains, builds the 4×4 mixing matrix application onto Stokes IQUV → XX/XY/YX/YY, and accumulates onto the output.
5. Frees per-call beam-gain and component-extrapolation arrays.

The shapelet kernel is structured slightly differently: it parallelises over basis-function entries so a single shapelet with 100 basis functions takes 100× the GPU work of a point source. This is also why shapelet chunking is performed per-coefficient (see §23).

---

## 10. Source-component primitives: extrapolation, beam application, RIME kernels

Every "primitive" in the visibility calculation has a CPU implementation in `src/source_components_cpu.c` and a GPU implementation in `src/source_components_gpu.cpp`. Most of them follow this contract:

```
For comp ∈ [0, n_comps):
  for time ∈ [0, n_times):
    for freq ∈ [0, n_freqs):
      out[comp_freq_time_index] = f(input parameters)
```

Headers `include/source_components_cpu.h` (1.1 kLOC) and `include/source_components_gpu.h` (1.6 kLOC) declare every variant. The GPU header in particular has a strict `extern "C"` discipline so that the C side can link without name-mangling collisions.

Important kernel families:

| Family | Functions | Purpose |
|---|---|---|
| Spectral extrapolation (Stokes I) | `extrap_power_laws_stokesI_{cpu,gpu}`, `extrap_curved_power_laws_stokesI_{cpu,gpu}`, `extrap_list_fluxes_stokesI_{cpu,gpu}` | Fill `extrap_stokesI[freq×comp]` |
| Stokes V | `extrap_power_laws_stokesV_*`, `extrap_curved_power_laws_stokesV_*`, `extrap_list_fluxes_stokesV_*`, `polarisation_fraction_stokesV_*` | Fill `extrap_stokesV[]` |
| Linear pol | `extrap_power_laws_linpol_*`, `extrap_curved_power_laws_linpol_*`, `extrap_list_fluxes_linpol_*`, `extrap_p_list_fluxes_linpol_*`, `polarisation_fraction_linpol_*`, `apply_rotation_measure_*` | Fill `extrap_stokesQ/U[]` (after RM) |
| Beam dispatch | `wrapper_calculate_gaussian_beam_*`, `wrapper_calculate_analytic_dipole_beam_*`, `wrapper_calculate_RTS_MWA_analytic_beam_*`, `wrapper_run_hyperbeam_*` | Each fills `gxs/Dxs/Dys/gys` for `(time × freq × comp)` |
| Visibility kernels | `calc_visi_point_or_gauss_{cpu,gpu}`, `calc_visi_shapelets_{cpu,gpu}` | Compute V_XX, V_XY, V_YX, V_YY |
| Auto-correlation kernels | `calc_autos_{cpu,gpu}` | Same as above but with l=m=n=0, w=0 |
| Beam-gain memory | `malloc_beam_gains_gpu`, `free_beam_gains_{cpu,gpu}` | |
| Antenna→baseline mapping | `fill_ant_to_baseline_mapping_{cpu,gpu}` | (n_ants × (n_ants−1))/2 unique pairs |
| Component memory | `malloc_extrapolated_flux_arrays_gpu`, `free_extrapolated_flux_arrays_gpu`, `free_components_{cpu,gpu}` | |

The GPU implementations launch one block per baseline×time×freq tile, with threads cooperating across components/coefficients. CUDA/HIP differences are abstracted by `gpu_macros.h` (e.g. `gpuMalloc` is `#define`d to `cudaMalloc` if `__NVCC__` is set, otherwise `hipMalloc`).

---

## 11. Fundamental coordinates: l,m,n, u,v,w, shapelet u,v

The geometry primitives sit in `src/fundamental_coords_{cpu,gpu}.{c,cpp}` (~115 + 269 lines).

### 11.1 `(l, m, n)` from `(RA, Dec, RA0, Dec0)`

Following the standard imaging convention,

```
H  = LST − RA            (hour-angle of source)
H0 = LST − RA0           (hour-angle of phase centre)
ΔH = H − H0
sin(δ) → sdec, cos(δ) → cdec (component)

l = cos(δ) · sin(ΔH)
m = sin(δ) · cos(δ0) − cos(δ) · sin(δ0) · cos(ΔH)
n = sqrt(1 − l² − m²)
```

`calc_lmn_cpu(ra0, sdec0, cdec0, ras, decs, ls, ms, ns, num_components)` computes this for many components against a single phase centre. `calc_lmn_for_components_*` runs it on a populated `components_t` and stores into `components->ls, ms, ns`.

### 11.2 `(u, v, w)` from baseline difference and HA0/Dec0

```
u =  sin(H0) · X_diff        + cos(H0) · Y_diff
v = -sin(δ0) · cos(H0) · X_diff
                         + sin(δ0) · sin(H0) · Y_diff + cos(δ0) · Z_diff
w =  cos(δ0) · cos(H0) · X_diff
                         - cos(δ0) · sin(H0) · Y_diff + sin(δ0) · Z_diff
```

When precession is enabled, the `(X_diff, Y_diff, Z_diff)` arrays are *re-computed per time step* (because the array is rotated back to J2000 at each time). `calc_uvw_cpu` takes one big flat array of HA0 sin/cos for *every (time × freq × baseline)* triplet, plus a wavelength array, and produces both `(u, v, w)_metres` (one per time × baseline; the metre values are time-dependent but not freq-dependent — only repeated for indexing) and `(us, vs, ws)` in wavelengths.

The CPU and GPU outputs match bit-exactly in `double` mode and to ≲ 10⁻⁷ in `float` mode (verified by `cmake_testing/`).

### 11.3 Per-shapelet `(u, v)`

For shapelets, each component carries its own phase-centre RA/Dec (the centre that SHAMFI fitted the shapelet about). A shapelet's image-plane position relative to that centre is the model's *intrinsic* shape. So the rotated `(u_shape_j, v_shape_j)` is computed with `(RA_j, Dec_j)` standing in as the new phase centre:

```
H_j  = LST − RA_j
H0_j = LST − RA0
```

These are produced by `calc_uv_shapelet_*`, which iterates over all SHAPELET components × times × baselines.

### 11.4 Auto-correlation guard

Internally the cross-correlation block comes first, autos last. After computing the cross uv,uv,uvws, `set_auto_uvw_to_zero_*` zeros out the auto block (length `num_autos = num_freqs × num_times × num_ants`) in both metres and wavelengths.

---

## 12. Primary-beam catalogue

WODEN's primary-beam options are catalogued in `e_beamtype` and dispatched in `src/beam_settings.c::fill_primary_beam_settings`. The Python representation is `BeamTypes` / `BeamGroups`. Each option has a CPU and GPU implementation, except UVBeam which is computed entirely in Python.

### 12.1 `none` / `NO_BEAM`

Gain = 1, leakage = 0 for both polarisations and every direction. The mixing matrix degenerates to:

```
V_XX = V_I + V_Q,    V_YY = V_I − V_Q,
V_XY = V_U + i V_V,  V_YX = V_U − i V_V.
```

### 12.2 Gaussian (`GAUSS_BEAM`)

Symmetric (currently `σ_l = σ_m`), polarisation-locked at `φ_PA=0`, no leakage. Frequency scaling:

```
σ_l = σ_m = sin(φ0) / (2 sqrt(2 ln 2)) · ν0/ν
```

with `φ0 = beam_FWHM`, `ν0 = beam_ref_freq`. Direction is in `(l_beam, m_beam)` cosines computed about a *fixed* hour-angle/dec pointing (`gauss_ha`, `gauss_dec`), so the beam stays az/za-locked across the simulation. Computed by `calculate_gaussian_beam_*` and `gaussian_beam_from_lm_*`.

### 12.3 EDA2 / `ANALY_DIPOLE`

A single MWA-style dipole at height `h = 0.3 m` over an *infinite* ground screen. Real-valued, no leakage:

```
G       = 2 sin(π · 2h/λ · cos θ)
g_x     = G · arccos( sin(θ) cos(φ) )   (N–S)
g_y     = G · arccos( sin(θ) sin(φ) )   (E–W)
```

Always points zenith. Computed by `analytic_dipole_*` and `calculate_analytic_dipole_beam_*`.

### 12.4 RTS-analytic MWA (`MWA_ANALY`)

A port of the RTS analytic MWA tile-beam model, taking 16 dipole delays and producing a complex Jones with an inherent parallactic rotation. Computed by `RTS_MWA_beam_*` and `calculate_RTS_MWA_analytic_beam_*`. Delays as listed in the metafits are converted into path lengths in metres (multiplied by the delay quantum `DQ = 435e-12 · VELC`) before being fed in.

### 12.5 MWA FEE coarse (`FEE_BEAM`)

Calls `mwa_hyperbeam`'s spherical-harmonic-expansion FEE model from `mwa_full_embedded_element_pattern.h5`, intrinsic 1.28 MHz spacing. Hyperbeam takes `(az, za)` increments per-component-fastest, per-time-slowest, plus delays (16 ints), 16 or 32 dipole amps (32 lets X and Y be different per dipole), and a flag to apply parallactic rotation. WODEN sets `parallactic=1` and `iau_order=true` so the result aligns with WODEN's internal IAU convention (X = N–S).

`mwa_hyperbeam` also does an explicit reordering of the Jones matrix to harmonise az-from-North vs az-from-East and IAU vs MWA polarisations; this is documented in the `polarised_source_and_FEE_beam.ipynb` notebook in the `polarisation_tests_for_FEE` repository.

### 12.6 MWA FEE interpolated (`FEE_BEAM_INTERP`)

Same code path but reads the interpolated coefficient file `MWA_embedded_element_pattern_rev2_interp_167_197MHz.h5` (interpolated by Daniel Ung), which has 80 kHz frequency resolution between 167 and 197 MHz. WODEN does not warn the user when frequencies fall outside this range — hyperbeam silently returns the boundary response.

### 12.7 EveryBeam OSKAR (`EB_OSKAR`)

Wrapped through `src/call_everybeam.cc`. The C++ side loads the SKA-LOW-style telescope model from a measurement set, with element response `"skala40_wave"`. Each station has its own beam by default; pass `--single_everybeam_station` (or `--station_id N`) to reuse one.

### 12.8 EveryBeam LOFAR (`EB_LOFAR`)

Same path with element response defaulting to `"hamaker"`. The other LOFAR responses `"hamaker_lba"` and `"lobes"` are accepted but warned to be unstable. Default station behaviour is per-station.

### 12.9 EveryBeam MWA (`EB_MWA`)

Calls `everybeam::mwabeam::Beam2016Implementation` directly (Jack maintains a fork that exposes `MWALocal` to take `(az, za)` directly so WODEN's J2000-precessed coordinates feed in correctly). Always single beam (`single_everybeam_station=1`).

### 12.10 pyuvdata UVBeam MWA (`UVB_MWA`)

`wodenpy/primary_beam/use_uvbeam.py::setup_MWA_uvbeams` constructs `pyuvdata.UVBeam` objects using `UVBeam.from_file(hdf5_path, pixels_per_deg=5)`, with `delays` shape `(2, 16)`, `amplitudes` shape `(2, 16)` per tile (or `len(amplitudes)/32` separate tiles). Frequency interpolation is done by UVBeam internally — interpolation range is extended beyond the requested band by ±2.56 MHz.

### 12.11 pyuvdata UVBeam HERA (`UVB_HERA`)

Two flavours:

- `--cst_file_list` — a CSV of `path, frequency_Hz` pairs pointing at CST simulation outputs. Loaded by `setup_HERA_uvbeams_from_CST`.
- `--uvbeam_file_path` — a single FITS / HDF5 UVBeam file. Loaded by `setup_HERA_uvbeams_from_single_file`.

Both build per-station `UVBeam` objects which are evaluated *Python-side* (because UVBeam doesn't have a C interface). The resulting Jones values are written into `components->gxs/Dxs/Dys/gys` *before* the C library is called. UVBeam runs threaded internally, so WODEN forces `num_sky_model_threads=1` to avoid GIL conflicts.

### 12.12 Beam normalisation

Default behaviour (`--no_beam_normalisation` not set): every primary beam value is divided by the beam value at the beam centre, evaluated at the *centre frequency of the band*. This applies primarily to EveryBeam outputs. The MWA hyperbeam flag `norm_to_zenith=1` is independent — it normalises hyperbeam Jones values against zenith-pointing.

---

## 13. Shapelet basis functions

A shapelet model is a collection of `(N1, N2, COEFF)` triples plus a centre `(RA, Dec)`, scale `(β_maj, β_min)`, and PA. WODEN converts each `(N1, N2)` Hermite product into a *2-D Hermite-Gaussian* basis function, sampled in a 1-D look-up table:

```
B(n, x, β=1) = H_n(x) · exp(−x²/2) / sqrt(2ⁿ · n!)
```

The C side reads pre-stored `B(n, x, 1)` from `WODEN/src/shapelet_basis.c::create_sbf` (213 lines of literal constants — legacy). The Python side rebuilds the same table from numpy primitives in `wodenpy/use_libwoden/shapelets.py::create_sbf`:

```python
def calc_basis_func_1D_numpy(n, x, beta=1):
    norm    = np.sqrt(beta) * np.sqrt(2**n * float(math.factorial(n)))
    hermite = numpy_eval_hermite(n, x)            # uses np.polynomial.Hermite
    gauss   = np.exp(-0.5 * (x*beta)**2)
    return (hermite * gauss) / norm
```

A 2-D basis function is then assembled at runtime by *separable* interpolation along x and y in the LUT, scaled by the shapelet's per-component β values. The C implementation rolls this into the visibility kernel; Python reproduces the numerics for unit-testing.

The motivation for shapelets is documented in `Line et al. 2020 (PASA)` — shapelets are a compact representation of complicated source morphologies (Fornax A is the canonical example, with 3000 basis functions per lobe) that fit naturally into a measurement-equation framework and don't require gridded image data. SHAMFI (`https://shamfi.readthedocs.io`) is the companion fitting tool.

---

## 14. EveryBeam C++ wrapper (`src/call_everybeam.cc`)

527 lines of C++ wrapping the EveryBeam library so that WODEN's pure-C and ctypes-Python sides can call it. The contract is in `include/call_everybeam_c.h` (371 lines).

Functions exposed:

- `char* check_ms_telescope_type(const char *ms_path)` — peek at the MS to decide if it's LOFAR, OSKAR, MWA, or `"UNKNOWN"`.
- `Telescope* load_everybeam_telescope(int *status, const char *ms_path, const char *element_response_model, bool use_differential_beam, bool use_channel_frequency, const char *coeff_path)` — generic loader.
- `void destroy_everybeam_telescope(Telescope *)` — destructor.
- `Beam2016Implementation* load_everybeam_MWABeam(const char *coeff_path, double *delays, double *amps)` — MWA-specific loader, takes 16 delays + 16 amps.
- `void destroy_everybeam_MWABeam(Beam2016Implementation *)`.
- `void run_phased_array_beam(Telescope *, num_stations, station_idxs, num_dirs, ra0, dec0, ras, decs, num_times, mjd_sec_times, num_freqs, freqs, apply_beam_norms, parallactic_rotate, element_only, iau_order, jones)` — main worker for phased-array beams (LOFAR / OSKAR). Output is a flat array indexed `[station × time × freq × dir × pol]`.
- `int load_and_run_lofar_beam(...)` and `int load_and_run_oskar_beam(...)` — convenience wrappers that load + run + destroy in one shot. Currently identical implementations.
- `int load_and_run_mwa_beam(double *delays, double *amps, coeff_path, num_dirs, azs, zas, para_angles, num_freqs, freqs, num_times, parallactic_rotate, iau_order, jones)` — MWA convenience wrapper.
- `void run_mwa_beam(Beam2016Implementation *, num_dirs, azs, zas, para_angles, num_freqs, freqs, num_times, parallactic_rotate, iau_order, jones)` — pre-loaded version, faster when calling repeatedly.

Two thread-safety considerations:

1. EveryBeam itself has functions that *aren't* re-entrant. Jack picked the ones that are. Where mutex protection was needed, he annotated the original functions in his EveryBeam fork.
2. casacore + python-casacore both create global C++ state when imported. If `libuse_everybeam.so` is built against a different casacore than `python-casacore` in the same Python process, segfaults follow. WODEN works around this by:
   - Always loading `libwoden_*.so` in a child process to check `check_for_everybeam_compilation()` (`wodenpy/use_libwoden/use_libwoden.py::check_for_everybeam`) — see `worker_check_for_everybeam`;
   - Reading the number of stations from the MS in a child process (`wodenpy/primary_beam/use_everybeam.py::worker_get_num_stations`);
   - Running the visibility worker `woden_worker` in a child process via `multiprocessing.Process` in serial mode.

Jack notes in the developer guide that this is a hot-fix and the long-term plan is to replace `python-casacore` calls with direct casacore-via-`libuse_everybeam.so` calls.

The MWA path additionally takes a parallactic angle as an *input* because EveryBeam-MWA does not internally do parallactic rotation; WODEN computes it Python-side via `erfa.hd2pa` and passes it in.

---

## 15. Logger, hyperbeam-error, compilation-flag check

Three small but architecturally important shims:

### 15.1 `src/logger.c` (15 LOC) + `include/logger.h` (55 LOC)

A C-side logger that forwards all messages to a registered `void (*log_callback)(int level, const char *msg)`. `wodenpy/wodenpy_setup/woden_logger.py::get_log_callback` builds a `ctypes.CFUNCTYPE` that adapts the C callback signature to a Python `logging.Logger`. The C library calls `set_log_callback(callback)` once at startup, after which every `log_message(...)` call inside the C/CUDA code gets routed through Python's `logging` module — meaning C log lines and Python log lines interleave correctly.

### 15.2 `src/hyperbeam_error.c` (22 LOC)

`handle_hyperbeam_error(file, line, fn_name)` — sets an error message into `beam_settings->hyper_error_str` and logs through the same forwarding callback.

### 15.3 `src/check_compilation_flags.c` (8 LOC)

```c
bool check_for_everybeam_compilation() {
    #ifdef HAVE_EVERYBEAM
    return true;
    #else
    return false;
    #endif
}
```

Python uses this to gate features on whether the user's locally compiled library has EveryBeam in or not; if absent and the user requests an EveryBeam beam, `run_woden.py` aborts with a helpful error.

### 15.4 Logger Python tooling

`wodenpy/wodenpy_setup/woden_logger.py` provides:

- `simple_logger(log_level=logging.DEBUG)` — `logging.Logger` to stdout with a fixed format;
- `set_woden_logger(log_level, log_file_name)` — split logger to stdout *and* an optional file;
- `set_logger_header(logger, gitlabel)` — stamps the WODEN ASCII banner + git hash;
- `summarise_input_args(logger, args)` — prints every CLI argument so that future users can reproduce the run from a log file alone;
- `log_chosen_beamtype(logger, woden_settings_python, args)` — logs the beam choice and any per-beam settings;
- `get_log_callback(logger, level)` — builds the `ctypes.CFUNCTYPE` C-side callback as above.

---

## 16. `wodenpy` — the Python package

The Python source is structured to mirror the C struct universe and to wrap every ctypes interaction in a thin, *picklable* Python class so that `concurrent.futures.ProcessPoolExecutor` and `multiprocessing.Process` can hand jobs across processes without C objects in the way.

### 16.1 `wodenpy.use_libwoden`

This subpackage owns the ctypes mirrors of every C struct:

- `create_woden_struct_classes.Woden_Struct_Classes(precision="double")` — *one* class that builds:
  - `Components_Ctypes` (matches `components_t`, ~120 ctypes fields, dynamically typed against precision)
  - `Source_Ctypes` (matches `source_t`)
  - `Source_Catalogue` (matches `source_catalogue_t`)
  - `Woden_Settings` (matches `woden_settings_t`)
  - `Visi_Set` (matches `visibility_set_t`)
  Importantly, the *ordering* of `_fields_` matches `woden_struct_defs.h` exactly.
- `array_layout_struct.Array_Layout_Ctypes` — non-dynamic (uses doubles only).
- `woden_settings.fill_woden_settings_python(args, jd_date, lst)` — constructs a `Woden_Settings_Python` (a regular Python class), reads command-line args, sets every field including the chosen beam-type integer.
- `woden_settings.setup_lsts_and_phase_centre` — populates `lsts`, `latitudes`, `mjds` arrays for every time step, applying RTS-style precession to J2000 if `do_precession=1`.
- `woden_settings.convert_woden_settings_to_ctypes(woden_settings_python, woden_settings_ctypes)` — copies fields *and* casts python arrays to ctypes pointers.
- `visibility_set.setup_visi_set` / `setup_visi_set_array` / `load_visibility_set` — allocate output buffers; convert ctypes outputs back into a numpy `(num_time × num_baseline, 1, 1, num_freq, 4, 3)` container suitable for uvfits writing. `4` is XX/YY/XY/YX, `3` is real/imag/weight.
- `beam_settings.BeamTypes`, `BeamGroups` — Python enum + grouping.
- `shapelets.create_sbf(precision="double", sbf_N=101, sbf_c=5000, sbf_dx=0.01)` — re-creates the basis-function table.
- `use_libwoden.load_in_run_woden(woden_lib, woden_struct_classes)` — registers the `run_woden` symbol from the shared library, with restype `c_int` and argtypes matching the struct universe.
- `use_libwoden.check_for_everybeam(woden_lib_path)` — runs the EveryBeam-flag check in a child process to keep casacore versions isolated.

### 16.2 `wodenpy.array_layout`

- `create_array_layout.calc_XYZ_diffs(woden_settings_python, args, logger)` — Reads from `args.east, args.north, args.height` and `args.latitude` to produce a per-time `(X_diff, Y_diff, Z_diff)` array. Uses `palpy` for precession.
- `create_array_layout.enh2xyz(east, north, height, latitude_rad)` — local ENH → local XYZ.
- `create_array_layout.convert_ecef_to_enh(ecef_X, Y, Z, lon, lat)` and `convert_enh_to_ecef` — used by EveryBeam path when reading antenna positions out of the MS or moving the array to a new latitude.
- `create_array_layout.setup_array_layout_python(woden_settings_python, args)` — for `--dry_run`, builds an Array_Layout_Python without precession / X_diff calculations.
- `create_array_layout.convert_array_layout_to_ctypes(array_layout_python, array_layout_ctypes)` — final ctypes promotion.
- `precession.RTS_Precess_LST_Lat_to_J2000(lst_current, latitude_current, mjd)` — uses `palpy.prenut(2000, mjd)` to build a 3×3 rotation matrix (combining precession and nutation) that takes apparent → mean-J2000 coordinates, then applies it to a unit zenith vector. Returns `(lst_J2000, latitude_J2000)`.

### 16.3 `wodenpy.observational`

`calc_obs.py` provides:

- `calc_jdcal(date)` — splits the ISO-8601 date into integer-JD-midnight + fractional-JD (uvfits stores them separately).
- `get_uvfits_date_and_position_constants(latitude, longitude, height, date)` — returns `LST_deg, GST0_deg, DEGPDY, ut1utc` for the uvfits header. Uses `astropy.time.Time` and `astropy.coordinates.EarthLocation`.

### 16.4 `wodenpy.skymodel`

This is the most complex subpackage. The flow is:

1. `read_skymodel.read_radec_count_components(path)` — does a *first pass* over the sky model to count everything (no flux loading, just RA/Dec and component types). Returns a `Component_Type_Counter` (defined in `wodenpy/skymodel/woden_skymodel.py`). Routing:
   - `.fits` → `read_fits_skymodel.read_fits_radec_count_components` (FITS is the preferred format)
   - `.yaml` → `read_yaml_skymodel.read_yaml_radec_count_components` (hyperdrive yaml; deprecated, Stokes-I only)
   - `.txt`  → `read_text_skymodel.read_text_radec_count_components` (legacy WODEN format; deprecated)
   else, `sys.exit`.
2. `Component_Type_Counter.crop_below_horizon(lst, latitude, ...)` — cull components below the horizon at the first time step. By default crops by COMPONENT (any component below the horizon is dropped, regardless of its parent SOURCE). With `--sky_crop_sources`, an entire SOURCE is dropped if any of its components is below the horizon.
3. `chunk_sky_model.create_skymodel_chunk_map(comp_counter, chunking_size, num_baselines, num_freq_channels, num_time_steps, num_threads, max_chunks_per_set, max_dirs, beamtype_value)` — builds *sets* of chunk-maps (`Skymodel_Chunk_Map`). Each set is a list of `num_threads` chunk-maps (so each thread can read in parallel). The `chunking_size` cap is in components-per-chunk (default `1e10`), and `max_chunks_per_set=32` for GPU mode (so a GPU never sees more than 32 chunks at once, balancing throughput vs memory) or `max_chunks_per_set=num_threads` for CPU mode.
4. `read_fits_skymodel.read_fits_skymodel_chunks(args, main_table, shape_table, chunk_maps, …)` — for each chunk in `chunk_maps`, reads only the rows it needs from the astropy `Table`s, computes az/za and parallactic angles for each component × each time step, and (for UVBeam beams) also computes the beam values *Python-side*. Returns a list of `Source_Python` objects (defined in `wodenpy/use_libwoden/skymodel_structs.py`).
5. `read_skymodel.create_source_catalogue_from_python_sources(python_sources, woden_struct_classes, beamtype, precision)` — finally promotes the Python-only Source list into a fully populated `Source_Catalogue` ctypes struct ready to feed the C library. This conversion is delayed until just before the C call so as little ctypes state as possible has to be pickled across processes.

The format-conversion path (yaml/txt → on-the-fly FITS) is handled by `read_yaml_skymodel.read_full_yaml_into_fitstable` and `read_text_skymodel.read_full_text_into_fitstable`. Internally everything is FITS once it's been parsed.

### 16.5 `wodenpy.uvfits.wodenpy_uvfits`

Self-contained uvfits writer (~457 lines). Key pieces:

- `RTS_encode_baseline(b1, b2)` — `b1*256 + b2`, with the RTS extension `b1*2048 + b2 + 65536` once `b2 > 255` (more than 256 antennas). `RTS_decode_baseline` is the inverse.
- `make_antenna_table(XYZ_array, telescope_name, num_antennas, freq_cent, date, gst0_deg, degpdy, ut1utc, longitude, latitude, array_height, ant_names=False)` — builds the AIPS-style `AIPS AN` HDU with `ANNAME, STABXYZ, NOSTA, MNTSTA, STAXOF, POLTYA/POLAA/POLCALA, POLTYB/POLAB/POLCALB`. Antenna names default to zero-padded numeric strings.
- `create_uvfits(v_container, freq_cent, central_freq_chan, ch_width, ra_point, dec_point, output_uvfits_name, uu, vv, ww, longitude, latitude, array_height, telescope_name, baselines_array, date_array, jd_midnight, hdu_ant, gitlabel, IAU_order, comment)` — combines the visibility group data with the antenna table.
- `make_baseline_date_arrays(num_antennas, date, num_time_steps, time_res, do_autos)` — produces the BASELINE and DATE arrays.

When `IAU_order=False` (default), columns are reordered so XX = E-W and a header `IAUORDER = F` is written. With `IAU_order=True`, XX = N-S (IAU) and header `IAUORDER = T`. The header `INSTRUME` tracks `telescope_name`. The `HISTORY` block records the WODEN git label and the original command line if `comment` is given.

### 16.6 `wodenpy.phase_rotate.remove_phase_track`

Some downstream pipelines (notably the RTS) generate their own phase tracking and expect WODEN to *not* phase-rotate. This module reverses WODEN's phase tracking by multiplying every visibility by `exp(2π i · w · ν / c)`. **Note:** It does *not* update the `(u, v, w)` arrays — those still encode the original phase centre. This is fine for pipelines that recompute `(u, v, w)` from the antenna table, but consumers should be aware.

### 16.7 `wodenpy.primary_beam.use_uvbeam`

UVBeam helpers:

- `setup_MWA_uvbeams(hdf5_path, freqs, delays, amplitudes, pixels_per_deg=5)` — builds an array of `pyuvdata.UVBeam` objects, one per tile. `amplitudes` should be `32 × num_tiles` with the layout `[NS pol of tile 1 (16), EW pol of tile 1 (16), NS pol of tile 2 (16), …]`.
- `setup_HERA_uvbeams_from_CST(cst_paths, cst_freqs, logger)` — builds a single HERA UVBeam from CST simulation outputs.
- `setup_HERA_uvbeams_from_single_file(uvbeam_file_path, logger)` — loads a single UVBeam-style file.
- `calc_uvbeam_for_components(uvbeam_objs, components, freqs, lsts, latitudes, …)` — evaluates the UVBeam at each component's `(az, za)` per time step per freq, returning Jones values that get stuffed into `components.gxs/Dxs/Dys/gys`.

### 16.8 `wodenpy.primary_beam.use_everybeam`

EveryBeam ctypes wrapper (~1.5 kLOC). Highlights:

- `worker_get_num_stations(ms_path, q)` / `get_num_stations(ms_path)` — read `ANTENNA` table out of the MS in a child process to dodge casacore segfaults.
- `run_everybeam(args, lsts, latitudes, beamtype, ras, decs, freqs, …)` — full Jones-on-grid evaluation. Calls into `libuse_everybeam.so` via ctypes. Outputs are `c_double_complex` arrays with shape `(num_stations, num_times, num_freqs, num_dirs, 4)`. The `4` is `(g_xx, D_xy, D_yx, g_yy)`.
- `create_filtered_ms(...)` — when the user requests a different EveryBeam pointing (`--eb_point_to_phase`, `--eb_ra_point`, `--eb_dec_point`) or a different array location (`--move_array_to_latlon`), creates a *minimal copy* of the MS with only one row, the pointing rewritten, and (optionally) the `ANTENNA` table rotated to the new lat/long. This is then passed to EveryBeam.

### 16.9 `wodenpy.wodenpy_setup`

- `run_setup.get_parser()` — *the* argparse parser (~700 LOC of options).
- `run_setup.check_args(args)` — bumper-rail argument validation: opens metafits files, populates implicit parameters, checks file existences, errors with descriptive messages.
- `run_setup.get_code_version()` — combines `wodenpy_gitinfo.txt` with the package `__version__` to make a single string for log/uvfits headers.
- `git_helper.retrieve_gitdict()` — reads `wodenpy_gitinfo.txt` if present, else returns `False`.
- `woden_logger` — described in §15.4.

---

## 17. `run_woden.py` — main driver script

`scripts/run_woden.py` is the user-facing entry point (~1100 lines). The control flow is:

```
main(argv)
├── get_parser().parse_args(argv)
├── check_args(args)               # arg validation, metafits parsing, defaults
├── set_woden_logger(...)          # central logger
├── get_uvfits_date_and_position_constants(...)
├── calc_jdcal(args.date) → jd_midnight, fractional_jd
├── Woden_Struct_Classes(args.precision)         # build ctypes mirrors
├── fill_woden_settings_python(args, jd_date, lst_deg)   # Python-side settings
├── setup_lsts_and_phase_centre(woden_settings_python)   # LSTs, mjds, possibly precess
├── calc_XYZ_diffs(woden_settings_python, args)          # array layout, X_diff/Y_diff/Z_diff
├── log_chosen_beamtype(...)
├── read_radec_count_components(args.cat_filename)       # 1st pass over sky model
├── crop_below_horizon(lst_first, latitude_first, comp_counter)
├── create_skymodel_chunk_map(...)                       # chunked-set scheduling
└── if not args.dry_run:
    ├── load libwoden_<precision>.so via ctypes
    ├── create_sbf(precision)                            # shapelet basis func table
    ├── setup_visi_set_array(...)                        # ctypes output buffers
    ├── visi_sets_python = np.array(num_visi_threads × num_bands of Visi_Set_Python)
    ├── if uvbeam beam: setup_MWA_uvbeams or setup_HERA_uvbeams
    ├── get_skymodel_tables(args.cat_filename)           # astropy Tables
    └── run_woden_processing(num_threads, num_rounds, …) # the heavy lifting
        ├── (CPU mode)  ProcessPoolExecutor for sky-read + ProcessPoolExecutor for visi
        ├── (GPU mode)  ProcessPoolExecutor for sky-read + single visi-thread that runs run_woden(...)
        └── (serial)   each thread sequentially in same process via multiprocessing.Process
    ├── combine all visi_sets into visi_sets_python_combined
    ├── make_antenna_table(...)
    ├── make_baseline_date_arrays(...)
    └── for each band:
        ├── load_visibility_set(...)            # → (uu, vv, ww, v_container)
        ├── if args.remove_phase_tracking: remove_phase_tracking(...)
        └── create_uvfits(...)                  # writes <prepend>_band<NN>.uvfits
```

Two parallelism strategies live in `run_woden_processing`:

- **GPU mode (default).** Sky-model chunks are read in *parallel* (one process per `num_threads`) while a *single* GPU thread crunches visibilities. Each round, the previous round's GPU result is `await`ed before launching the next sky-read so memory stays bounded. UVBeam beams force `num_sky_model_threads=1` (UVBeam threads internally).
- **CPU mode (`--cpu_mode`).** `num_threads` parallel sky-model readers, `num_threads` parallel visibility computations. Output visi_sets are reduced after the loop completes.
- **Serial mode (`--num_threads=1`).** Each chunk is read in the main process, then a `multiprocessing.Process` is launched per visi call to keep casacore versions isolated. Slower but most portable.

The `woden_worker(thread_ind, all_loaded_python_sources, …)` function:

1. Imports `Woden_Struct_Classes(precision)`.
2. Loads `libwoden_<precision>.so` via `ctypes.cdll.LoadLibrary`.
3. Registers `set_log_callback` so the C side can talk to the Python logger.
4. Promotes the python sources to a ctypes `Source_Catalogue`.
5. Promotes settings + array layout to ctypes.
6. Allocates `Visi_Set` ctypes buffers.
7. Calls `run_woden(woden_settings, visibility_set, source_catalogue, array_layout, sbf)`.
8. Reads back the visibilities into numpy arrays (with `deepcopy` so the buffer memory can be freed).
9. Returns `(visi_sets_python, thread_ind, round_num)`.

The `read_skymodel_worker(thread_id, num_threads, chunked_skymodel_map_sets, …)`:

1. Picks the right slice of chunk maps based on `(thread_id // num_threads, thread_id % num_threads)`.
2. Calls `read_fits_skymodel_chunks` with that slice.
3. Returns `(python_sources, thread_num)`.

### 17.1 Argument groups

The parser groups options for clarity. Below is a summary of every group with the headline flags. See `wodenpy/wodenpy_setup/run_setup.py` lines 44-380 for the full per-flag help.

#### OBSERVATION OPTIONS
- `--ra0, --dec0` — phase centre (deg). Required unless using a metafits or MS.
- `--date` — initial UTC date `YYYY-MM-DDThh:mm:ss`. From metafits if available.
- `--no_precession` — disable J2000 precession of the array.

#### FREQUENCY OPTIONS
- `--band_nums=1,7,9` — which coarse bands to simulate (defaults to 1..24).
- `--lowest_channel_freq` — Hz, lowest fine channel of band 1.
- `--coarse_band_width` — Hz, per-band bandwidth.
- `--num_freq_channels` — fine channels per band; defaults to `coarse_band_width / freq_res`.
- `--freq_res` — fine-channel resolution (Hz).

#### TIME OPTIONS
- `--num_time_steps`, `--time_res`.

#### TELESCOPE OPTIONS
- `--latitude, --longitude, --array_height` — Earth location (defaults differ by primary beam: MWA / LOFAR / HERA).
- `--array_layout` — text file of `(east, north, height)` antenna positions (metres). Or read from metafits/MS.
- `--primary_beam` — see §12 catalogue.
- `--off_cardinal_dipoles` — switch to the 45/135° mixing matrix.
- `--telescope_name` — written to uvfits.

#### MWA PRIMARY BEAM OPTIONS
- `--hdf5_beam_path` — overrides `$MWA_FEE_HDF5`.
- `--MWA_FEE_delays=[0,0,…,0]` — 16 ints. From metafits when available.
- `--use_MWA_dipflags` — apply per-dipole flags from the metafits (FEE/UVBeam only).
- `--use_MWA_dipamps` — use bespoke per-dipole amplitudes from the metafits column `DipAmps` (FEE/UVBeam only).

#### EVERYBEAM PRIMARY BEAM OPTIONS
- `--beam_ms_path` — required for LOFAR/OSKAR.
- `--no_beam_normalisation`.
- `--station_id N` — force every station to use the same beam.
- `--eb_point_to_phase / --eb_ra_point / --eb_dec_point` — override the beam pointing.
- `--move_array_to_latlon` — relocate an MS to a new lat/long (rotates antenna positions).

#### PYUVDATA UVBEAM PRIMARY BEAM OPTIONS
- `--uvbeam_file_path` — for `--primary_beam=uvbeam_HERA`, single FITS/HDF5 UVBeam file.
- `--cst_file_list` — CSV of `(path, freq_Hz)` pairs for HERA CST sims.

#### GAUSSIAN PRIMARY BEAM OPTIONS
- `--gauss_beam_FWHM` (deg, default 20).
- `--gauss_beam_ref_freq` (Hz, default 150 MHz).
- `--gauss_ra_point, --gauss_dec_point` — initial RA/Dec to point.

#### INPUT/OUTPUT OPTIONS
- `--IAU_order` — re-order output XX/YY columns to IAU (XX=N-S).
- `--cat_filename` — sky model file (`.fits`, `.yaml`, `.txt`).
- `--metafits_filename` — MWA metafits (sets array, freq, time, delays).
- `--output_uvfits_prepend` — prepend for output uvfits (default `output`).
- `--sky_crop_components` (default true) / `--sky_crop_sources` — cropping behaviour.
- `--do_autos` — include auto-correlations.

#### SIMULATOR OPTIONS
- `--cpu_mode` — disable GPU.
- `--num_threads N` — defaults to physical-core count (`psutil.cpu_count(logical=False)`).
- `--max_sky_directions N` — chunk-size cap for EveryBeam (default 200 dirs/chunk for EveryBeam, otherwise derived from `--chunking_size`).
- `--precision=double|float` — selects which `libwoden_*.so` to load.
- `--chunking_size 1e10` — max components per chunk.
- `--dry_run` — go through all setup, then stop before invoking `run_woden`.
- `--remove_phase_tracking` — undo WODEN's phase tracking.

#### LOGGING OPTIONS
- `--version, --verbose, --save_log, --profile`.

#### Hidden args (filled by `check_args`)
`--east, --north, --height, --num_antennas, --array_layout_name, --dipamps, --dipflags, --pointed_ms_file_name, --command, --num_freq_channels, --hdf5_beam_path, --IAU_order, --ant_names`. These are stored on the `args` Namespace by `check_args` and forwarded to the rest of the pipeline.

---

## 18. Companion scripts (`scripts/`)

| Script | Purpose | Lines |
|---|---|---|
| `run_woden.py` | Main driver — see §17 | 1096 |
| `add_woden_uvfits.py` | Sum visibilities of two uvfits (or two sets-of-bands) | 126 |
| `concat_woden_uvfits.py` | Concatenate per-band uvfits into one frequency axis | 252 |
| `woden_uv2ms.py` | Convert uvfits → CASA Measurement Set via `pyuvdata.UVData.read` + `.write_ms` | 116 |
| `add_instrumental_effects_woden.py` | Inject instrumental effects (visibility noise, gain errors, leakage, cable reflections, fine-channel flags) | 759 |
| `convert_WSClean_list_to_WODEN.py` | Convert WSClean BBS source-list format to WODEN FITS format | 81 |
| `delay_spec_from_uvfits.py` | Plot delay-power-spectra from a WODEN uvfits | 395 |
| `unwrap_woden_phases.py` | Unwrap visibility phases for diagnostic plots | 456 |
| `update_init_WODEN.py` | Internal helper to rewrite `wodenpy/__init__.py` version on tag-bump | 55 |

The most useful for downstream pipelines:

### 18.1 `add_woden_uvfits.py`

Adds visibility data of two uvfits files: `output[i] = uvfits1[i] + uvfits2[i]`. Two modes:

- **Per-band**: `--uvfits_prepend1=A_ --uvfits_prepend2=B_ --output_name_prepend=combined --num_bands=24` produces `combined01.uvfits`…`combined24.uvfits`.
- **Single pair**: `--uvfits1=A.uvfits --uvfits2=B.uvfits --output_name=C.uvfits`.

The `u, v, w` and antenna table are inherited from `uvfits1`. Only `data[..., :2]` (the real / imag) are summed; weights (the third element) are kept from `uvfits1`. Useful for, e.g., adding diffuse + point-source contributions that were simulated separately.

### 18.2 `concat_woden_uvfits.py`

Concatenate frequency bands. Optional `--reverse_pols` swaps XX↔YY (E-W↔N-S) per the IAU/MWA debate, and `--half_power` halves the data while leaving weights unchanged (useful for averaging duplicate observations).

### 18.3 `woden_uv2ms.py`

Thin wrapper around pyuvdata for uvfits → MS, optionally over multiple bands.

### 18.4 `add_instrumental_effects_woden.py`

A surprisingly capable post-processor that:

- adds visibility noise via the radiometer equation (`--add_visi_noise`, `--visi_noise_int_time`, `--visi_noise_freq_reso`) using MWA-default receiver temperature and effective area (override via `--noise_set_*` flags);
- adds cable reflections (`--cable_reflection_from_metafits`, `--cable_reflection_coeff_amp_min/max`);
- adds per-antenna gain amplitude / phase errors (`--ant_gain_amp_error`, `--ant_gain_phase_error`);
- adds engineering-tolerance leakage (`--ant_leak_errs psi_err chi_err`) using the TMS Eq. A4.5 dipole-misalignment formula `Dx = ψ - i χ`, `Dy = -ψ + i χ`;
- adds fine-channel flagging (`--add_fine_channel_flags`);
- supports reproducible RNG via `--noise_numpy_seed`, `--inst_numpy_seed`.

These give pipelines like `hyperdrive` realistic-looking data without WODEN itself having to model gain calibration.

### 18.5 `convert_WSClean_list_to_WODEN.py`

Reads a WSClean BBS-format source list and writes a WODEN FITS sky model. Used in the LOFAR examples to convert LoTSS catalogue cutouts.

---

## 19. Sky-model formats: FITS, hyperdrive YAML, native text

WODEN reads three formats. Going forward, only FITS is actively developed. Yaml and text are deprecated — they are read into a FITS-equivalent astropy `Table` internally before being passed to the chunking/loading machinery.

### 19.1 FITS sky model

The preferred and only fully-featured format. The model lives in a FITS file with:

- **`MAIN` HDU** (or first HDU regardless of name) — one row per COMPONENT.
- **`SHAPELET` HDU** (optional) — one row per `(NAME, N1, N2, COEFF)` shapelet basis-function entry.
- **`V_LIST_FLUXES`, `Q_LIST_FLUXES`, `U_LIST_FLUXES`, `P_LIST_FLUXES` HDUs** (optional) — list-fluxes for polarisation when the corresponding `*_MOD_TYPE` column says `nan` or `p_nan`.

A `SOURCE` is a logical group of one or more `COMPONENT`s sharing the same `UNQ_SOURCE_ID`. The `NAME` column should be `UNQ_SOURCE_ID_C<NNN>` so that components match shapelet rows by unique name.

#### MAIN HDU columns

| Column | Unit | Meaning |
|---|---|---|
| `UNQ_SOURCE_ID` | — | (Required) source group id |
| `NAME` | — | (Required) component name `<id>_C<NNN>` |
| `RA` | deg | (Required) J2000 |
| `DEC` | deg | (Required) J2000 |
| `COMP_TYPE` | — | (Required) `P`, `G`, or `S` (point, Gaussian, shapelet) |
| `MAJOR_DC` | deg | Gauss/Shapelet major axis |
| `MINOR_DC` | deg | Gauss/Shapelet minor axis |
| `PA_DC` | deg | Gauss/Shapelet position angle |
| `MOD_TYPE` | — | (Required) `pl` / `cpl` / `nan` (power, curved-power, list) |
| `NORM_COMP_PL` | Jy | Stokes-I PL flux at 200 MHz |
| `ALPHA_PL` | — | Stokes-I PL spectral index |
| `NORM_COMP_CPL` | Jy | Stokes-I CPL flux at 200 MHz |
| `ALPHA_CPL` | — | Stokes-I CPL spectral index |
| `CURVE_CPL` | — | Stokes-I CPL curvature q |
| `INT_FLX<MHz>` | Jy | Stokes-I list-flux at the given frequency in MHz; e.g. `INT_FLX150` is at 150 MHz. Any number of these. |
| `V_MOD_TYPE` | — | `pl`/`cpl`/`pf`/`nan` — note `pf` is polarisation-fraction, `nan` is V_LIST |
| `V_POL_FRAC` | — | Stokes-V fraction (signed, can be > 1) |
| `V_NORM_COMP_PL`, `V_ALPHA_PL` | Jy, — | Stokes-V PL |
| `V_NORM_COMP_CPL`, `V_ALPHA_CPL`, `V_CURVE_CPL` | Jy, —, — | Stokes-V CPL |
| `LIN_MOD_TYPE` | — | `pl`/`cpl`/`pf`/`nan`/`p_nan` — `nan` = independent Q & U lists, `p_nan` = single P list with RM |
| `RM` | rad/m² | Linear-pol rotation measure |
| `INTR_POL_ANGLE` | rad | Intrinsic χ_0 (assumed 0 if missing) |
| `LIN_POL_FRAC` | — | Linear-pol fraction |
| `LIN_NORM_COMP_PL`, `LIN_ALPHA_PL` | Jy, — | Linear-pol PL |
| `LIN_NORM_COMP_CPL`, `LIN_ALPHA_CPL`, `LIN_CURVE_CPL` | Jy, —, — | Linear-pol CPL |

#### SHAPELET HDU columns

| Column | Meaning |
|---|---|
| `NAME` | The component name as in `MAIN.NAME` |
| `N1` | First Hermite order |
| `N2` | Second Hermite order |
| `COEFF` | Coefficient `C_{p_k, p_l}` |

Multiple rows per component are expected (one per basis function entry); the cross-reference is by `NAME`.

#### V_LIST_FLUXES HDU (optional)

| Column | Unit | Meaning |
|---|---|---|
| `NAME` | — | Component name |
| `V_INT_FLX<MHz>` | Jy | Stokes-V list flux at <MHz> |

Identical pattern for `Q_LIST_FLUXES`, `U_LIST_FLUXES`, `P_LIST_FLUXES`. With `LIN_MOD_TYPE = nan`, supply Q and U lists *separately* (extrapolated independently — Q and U decouple from each other, no RM applied). With `LIN_MOD_TYPE = p_nan`, supply a single P list and the `RM` and `INTR_POL_ANGLE` columns split it into Q/U:

```
P(λ) = interp(P_INT_FLX<MHz>);
Q    = P · cos(2 χ_0 + 2 RM λ²);
U    = P · sin(2 χ_0 + 2 RM λ²).
```

The example `WODEN/examples/polarisation/polarisation_examples.ipynb` shows how to build these HDUs from Python.

### 19.2 Hyperdrive YAML

YAML files in the format used by `mwa_hyperdrive`. Read by `read_yaml_skymodel.py`. Stokes-I-only, with point sources, Gaussians, and shapelets, but *without* IQUV, list-of-V, or rotation-measure features. Internally converted to a FITS-equivalent astropy Table via `read_full_yaml_into_fitstable`.

### 19.3 Native WODEN text

The original RTS-inherited text format. Per-source / per-component / per-frequency / per-shapelet basis-function indented blocks. Read by `read_text_skymodel.py`. Stokes-I-only. Internally converted to the FITS-equivalent.

Both deprecated formats produce `False` for `v_table`, `q_table`, `u_table`, `p_table` so polarisation routines are simply skipped.

---

## 20. UVFITS output format and frequency banding

### 20.1 The "bands" mental model

WODEN treats a simulation as a set of *coarse bands*, each a contiguous block of fine channels. This is inherited from MWA correlator data (24 coarse bands × 1.28 MHz, each split into 10/20/40 kHz fines).

```
[ Band 1            ] [ Band 2            ] … [ Band N         ]
[ low_freq + (1-1)*BW ]
[ N_fine fine chans of width freq_res ]
```

so band `k` spans `[base_low_freq + (k-1)·coarse_band_width, base_low_freq + k·coarse_band_width)` and the lowest fine channel of band `k` is `base_low_freq + (k-1)·coarse_band_width`.

CLI flags:

- `--lowest_channel_freq` → `base_low_freq` (Hz)
- `--coarse_band_width` → `coarse_band_width` (Hz)
- `--freq_res` → fine-channel width (Hz)
- `--num_freq_channels` → fines per band (defaults to `coarse_band_width/freq_res`)
- `--band_nums=1,4,6` → which `k`s to simulate

Each simulated band produces a *separate* `<prepend>_band%02d.uvfits` so the run is trivially parallelisable across GPUs.

### 20.2 uvfits content

The output is an AIPS-style uvfits group HDU + an `AIPS AN` antenna table. The data shape is `(num_time × (num_baselines + num_antennas if do_autos else num_baselines), 1, 1, num_freq_channels, 4, 3)`:

- axis 0: visibility ordering — baseline-then-time;
- axes 1, 2: dummy 1's (RA, Dec);
- axis 3: fine channel;
- axis 4: polarisation — XX, YY, XY, YX (in that order, regardless of `IAU_order`);
- axis 5: real / imag / weight.

Group parameters include `UU, VV, WW` (in seconds), `BASELINE` (encoded), `DATE` (fractional JD).

### 20.3 IAU vs MWA polarisation order

The C code labels things internally as XX = N-S, IAU-style. But MWA convention writes XX = E-W. WODEN handles this by re-ordering at write time:

- Default (`--IAU_order` *not* set): write `XX = E-W, YY = N-S, XY = E-W·N-S, YX = N-S·E-W`. Header `IAUORDER = F`.
- With `--IAU_order`: write `XX = N-S, YY = E-W, XY = N-S·E-W, YX = E-W·N-S`. Header `IAUORDER = T`.

Older WODEN (< 1.4.0) wrote `IAUORDER = T` always; missing `IAUORDER` should be assumed True.

### 20.4 Header keywords of note

- `IAUORDER` — `T` or `F` per above.
- `INSTRUME` — `--telescope_name`, default by primary beam (e.g. `MWA`, `LOFAR`, `EDA2`, `HERA`).
- `CRVAL4`, `CDELT4` — central-fine-channel frequency and `freq_res`.
- `CRVAL5`, `CDELT5` — centre RA and `step` (zero).
- `RDATE`, `GSTIA0`, `DEGPDY`, `UT1UTC` — set by `get_uvfits_date_and_position_constants`.
- `ARRAYX, ARRAYY, ARRAYZ` — geocentric centre of array.
- `XYZHAND = RIGHT`, `FRAME = ????` (legacy).
- `HISTORY` — git label and the original `run_woden.py` command.

### 20.5 Auto-correlations

When `--do_autos` is on, autos appear after crosses *per time step*. They are interleaved with crosses in the AIPS output via the BASELINE encoding (which can express auto-correlations as `(antenna, antenna)`). Auto u/v/w are zero. WODEN computes autos with the same `calc_autos_*` kernel that runs the RIME with `l = m = n = 0` and `w = 0`.

---

## 21. Polarisation conventions: on-cardinal vs off-cardinal

There are two sets of equations for taking Stokes IQUV into instrumental visibilities (`docs/sphinx/operating_principles/visibility_calcs.rst`):

### 21.1 On-cardinal dipoles (0/90°, default)

```
V_XX = (g_x g_x* + D_x D_x*) V_I + (g_x g_x* - D_x D_x*) V_Q
     + (g_x D_x* + D_x g_x*) V_U + i(g_x D_x* - D_x g_x*) V_V
V_XY = (g_x D_y* + D_x g_y*) V_I + (g_x D_y* - D_x g_y*) V_Q
     + (g_x g_y* + D_x D_y*) V_U + i(g_x g_y* - D_x D_y*) V_V
V_YX = (D_y g_x* + g_y D_x*) V_I + (D_y g_x* - g_y D_x*) V_Q
     + (D_y D_x* + g_y g_x*) V_U + i(D_y D_x* - g_y g_x*) V_V
V_YY = (D_y D_y* + g_y g_y*) V_I + (D_y D_y* - g_y g_y*) V_Q
     + (D_y g_y* + g_y D_y*) V_U + i(D_y g_y* - g_y D_y*) V_V
```

(Subscripts 1 and 2 for the two antennas in the baseline are dropped here for brevity; in the full expression each `g_x` becomes either `g_{1x}` or `g_{2x}` and the conjugates align with antenna 2.)

If gains = 1 and leakages = 0, the simplified form

```
V_XX = V_I + V_Q,
V_XY = V_U + i V_V,
V_YX = V_U − i V_V,
V_YY = V_I − V_Q
```

inverts trivially to recover Stokes `I, Q, U, V`.

### 21.2 Off-cardinal dipoles (45/135°)

Triggered by `--off_cardinal_dipoles` or by a beam type in `BeamGroups.off_cardinal_beam_values`. The mixing matrix is

```
M = [[1,  0,  1,  0],
     [0, -1,  0,  i],
     [0, -1,  0, -i],
     [1,  0, -1,  0]]
```

so that the no-beam case becomes

```
V_PP = V_I + V_U,
V_PQ = -V_Q + i V_V,
V_QP = -V_Q − i V_V,
V_QQ = V_I − V_U
```

and the inverse extraction of Stokes Q/U swaps relative to the on-cardinal form. WODEN still labels the outputs XX/XY/YX/YY internally — the difference is only in *what was multiplied with what* on the way out. The list `BeamGroups.off_cardinal_beam_values` is presently empty, with a developer-side TODO to investigate whether LOFAR LBA dipoles need this treatment.

---

## 22. Precession / nutation handling

Sky catalogues are nearly all in J2000. Source positions, however, *appear* shifted by the precession+nutation of Earth's rotation axis; this shifts by tens-of-arcseconds over years. To keep the sky model frozen in J2000 (faster, simpler), WODEN instead precesses the *array* and the *LST* back from the observation epoch to J2000:

```
# wodenpy/array_layout/precession.py
rmatpn          = palpy.prenut(2000, mjd)               # apparent ↔ mean-J2000 rotation
J2000_transform = transpose(rmatpn)
v1              = palpy.dcs2c(lst_current, latitude_current)
v2              = palpy.dmxv(J2000_transform, v1)
lst_J2000, latitude_J2000 = palpy.dcc2s(v2)
```

This is done *per time step* (because `mjd` changes), so the local (`X_diff, Y_diff, Z_diff`) baseline arrays change with time. The C code then uses `(LST_J2000(t), latitude_J2000(t))` consistently to compute `(u, v, w)`. The result is that the visibility produced by an input J2000 source has zero precession-induced displacement.

This matches what the RTS does and means RTS-calibrated images of WODEN simulations can be compared back to the input J2000 catalogue cleanly. To disable, pass `--no_precession`.

The price is ~100s of µs per time step in Python. For long observations, this is amortised by chunked sky reading.

---

## 23. Chunking and lazy-loading the sky model

Modern 21-cm-foreground models can have tens of millions of components. They cannot fit on a typical GPU all at once. WODEN's lazy-loading scheme, in `wodenpy/skymodel/chunk_sky_model.py`, decomposes the work as follows.

### 23.1 Two-pass reading

Pass 1 (`read_radec_count_components`) loads only RA / Dec / component-type / flux-model-type / line-number from the catalogue file. The output `Component_Type_Counter` records:

- one numpy array `comp_types` of length `n_total_comps` whose values are members of `CompTypes` enum (POINT_POWER, POINT_CURVE, POINT_LIST, GAUSS_POWER, …, V_POINT_LIST, …, LIN_SHAPE_P_LIST);
- one `source_indexes` array recording which SOURCE each component belongs to;
- one `file_line_nums` array (for txt/yaml only, line-offsets in the file);
- per-component RA / Dec (used to crop below horizon);
- per-component flux-list count (`num_list_fluxes`, `num_v_list_fluxes`, …, `num_p_list_fluxes`);
- per-component shapelet-coefficient count (`num_shape_coeffs`).

`Component_Type_Counter.crop_below_horizon(lst, latitude, …)` then trims the arrays to only those above the horizon at the simulation's first time step.

### 23.2 Chunk scheduling

`create_skymodel_chunk_map(comp_counter, chunking_size, num_baselines, num_freq_channels, num_time_steps, num_threads, max_chunks_per_set, max_dirs, beamtype_value)` uses `binpacking` to:

- Build an effective per-component cost (rough proxy for GPU occupancy) — point and Gaussian components are all weight 1, shapelet components are weighted by their basis-function count.
- Pack components into chunks of `≤ chunking_size` weight, no more than `max_dirs` directions per chunk (200 by default for EveryBeam — EveryBeam does fixed batches of directions).
- Group chunks into "sets", with `num_threads` chunks per set so that all sky-reader threads can run a chunk in parallel.
- Cap total chunks per set at `max_chunks_per_set` (32 GPU / `num_threads` CPU) to avoid VRAM blow-up.

Each chunk is a `Skymodel_Chunk_Map` carrying:

- per-component-type counts (`n_point_powers`, `n_point_curves`, `n_point_lists`, `n_gauss_*`, `n_shape_*`);
- per-component bookkeeping (`Components_Map` instances for `point_components`, `gauss_components`, `shape_components`);
- references back to original-row indices in the source catalogue table for re-reading;
- shapelet-basis indices for the shapelet rows;
- polarisation counts (V power/curve/list/pol-frac, Lin power/curve/list/p-list/pol-frac).

### 23.3 Pass 2 — read full data per chunk

`read_fits_skymodel_chunks(args, main_table, shape_table, chunk_maps, num_freq_channels, num_time_steps, beamtype, lsts, latitudes, v_table, q_table, u_table, p_table, precision, uvbeam_objs, logger)` reads each chunk by:

1. Slicing the astropy tables by the chunk's component-row indices to extract the full Stokes / shape parameters;
2. Computing per-time `(az, za, parallactic_angle)` if the beam needs it (`BeamGroups.azza_beam_values`);
3. Computing `(beam_HA, beam_Dec)` if needed (`BeamGroups.hadec_beam_values`);
4. For UVBeam beams: evaluating the UVBeam at each `(az, za, freq, time)` and storing into `Source_Python.point_components.gxs/Dxs/Dys/gys`;
5. Returning a list of `Source_Python` objects, one per shapelet-coefficient sub-chunk.

The chunks are then handed off to `woden_worker`, which promotes them to ctypes `Source_Catalogue` and calls `run_woden`.

### 23.4 Why shapelets multiply chunks

Inside the C code, the kernel for shapelets parallelises over basis-functions (because that's the dominant work). Instead of a single shapelet with 100 basis functions consuming one component-slot, the chunker splits it across multiple GPU launches based on `n_shape_coeffs`. Therefore a shapelet `Skymodel_Chunk_Map` may decompose into multiple `Source_Python` objects.

---

## 24. Testing and CI

WODEN has an extensive CTest-based test suite in `cmake_testing/` plus Python integration tests in the same tree. The Sphinx documentation has a per-module testing breakdown at `docs/sphinx/testing/`.

### 24.1 Compiled-code unit tests (CTest)

Built when `cmake .. -DTARGET_GROUP=test` is invoked. Each `.c` test compiles against:

- The Unity test framework (`src/Unity/...`) — checked out via submodule.
- The WODEN library (in either float or double form).

Tests are organised into per-module subdirectories under `cmake_testing/`:

- `cmake_testing/calculate_visibilities/` — end-to-end tests of the visibility calculation against analytic answers (Cosine/Sine truth tables).
- `cmake_testing/source_components/` — every flux-extrapolation function tested against numpy-computed truth.
- `cmake_testing/fundamental_coords/` — ABS-tolerance checks against analytic `(l, m, n)`, `(u, v, w)`, shapelet uv.
- `cmake_testing/primary_beam/` — Gaussian, EDA2, MWA-analytic, FEE, FEE-interp, EveryBeam (when compiled in) — tested against numpy + golden data.
- `cmake_testing/call_everybeam/` — fully optional, only run when EveryBeam is present.
- `cmake_testing/visibility_set/`, `cmake_testing/beam_settings/`, `cmake_testing/logger/` — small struct-allocation tests.

Run all C tests with `ctest --output-on-failure`. A pytest-based Python suite can be invoked via `pytest cmake_testing/wodenpy/`.

### 24.2 Coverage

Code coverage from `gcov`/`lcov` is generated for C / C++ only (CUDA is not covered by free tools). However, "every GPU function now has a CPU equivalent which is tested, so this number is indicative of the entire package" (per the README). The `coverage_outputs/` directory holds pre-rendered HTML.

### 24.3 Script tests / installation tests

`test_installation/` contains end-to-end shell scripts (`absolute_accuracy/run_the_absolute_accuracy_test.sh`, `array_layouts/EDA2_layout_255.txt`, …) used to validate a freshly compiled installation against the numbers reported in the JOSS paper. These also serve as worked examples for users.

### 24.4 GitHub CI / Codecov

The README links `codecov.io/gh/JLBLine/WODEN`. The CI is GitHub Actions (`.github/workflows`) — runs CMake tests on Ubuntu against several Python versions, and Sphinx doc build.

### 24.5 Documentation tests

`check_documentation/` has a script that diffs Doxygen output against the public API to catch undocumented C functions; failure is an error in CI.

---

## 25. Examples shipped with WODEN

`examples/` and `docs/sphinx/examples/` ship runnable shell scripts and notebooks. Storage requirements are noted per example.

| Example | Tests / demonstrates | Storage |
|---|---|---|
| `examples/FornaxA/` | Point + Gauss vs shapelet model of Fornax A; FLOAT vs DOUBLE precision comparison; metafits-driven simulation | small |
| `examples/MWA_EoR1/` | Large catalogue (>300 k components); EoR field simulation | ~5 GB |
| `examples/EDA2_haslam/` | Array layout from text file, EDA2 beam, no metafits (5-band run); 393 216 point sources from a pygdsm-generated nside=256 healpix; commented runtime ~61 min | 1.8 GB |
| `examples/dipole_ampflags/` | MWA dipole flag/amp metafits feature | small |
| `examples/polarisation/polarisation_examples.ipynb` | How to construct polarised FITS sky models, check Q/U/V outputs | small |
| `examples/LOFAR_LoTSS/` | LoTSS DR2 cutout → WODEN FITS, EveryBeam LOFAR | medium |
| `examples/LOFAR_LBA_NCP/` | LOFAR LBA simulation of NCP; warns about plotting at the pole | medium |
| `examples/HERA_sim/` | HERA, both CST and FITS UVBeams; F1 field with phase-1 layout | medium |
| `examples/relocate_everybeam_array/` | Move LOFAR MS to MWA latitude/longitude — useful for cross-array comparisons | small |
| `examples/metafits/` | All-MWA examples via metafits | small |

The EDA2 Haslam example command line (from `docs/sphinx/examples/eda2_haslam_sim.rst`):

```bash
run_woden.py \
  --ra0=74.79589467 --dec0=-27.0 \
  --time_res=10.0 --num_time_steps=10 \
  --freq_res=10e+3 --coarse_band_width=10e+4 \
  --lowest_channel_freq=100e+6 \
  --cat_filename=pygsm_woden-list_100MHz_n256.txt \
  --array_layout=../../test_installation/array_layouts/EDA2_layout_255.txt \
  --date=2020-02-01T12:27:45.900 \
  --output_uvfits_prepend=./data/EDA2_haslam \
  --primary_beam=EDA2 \
  --sky_crop_components \
  --band_nums=1,2,3,4,5
```

The README shows a minimal Fornax A run via metafits (see §1 of the README extract).

---

## 26. Docker / Singularity images

`docker/Dockerfile_cuda` — the canonical CUDA image. Built via `docker/make_docker_image.sh`. Bundles:

- Ubuntu base + CUDA toolkit;
- rust + mwa_hyperbeam built with `cuda,hdf5-static`;
- WODEN sources compiled with `--CUDAARCHS=$ARCH`;
- `mwa_full_embedded_element_pattern.h5` and `MWA_embedded_element_pattern_rev2_interp_167_197MHz.h5` pre-downloaded;
- `pip install -r requirements.txt && pip install .` for `wodenpy`;
- `astropy` IERS bootstrap (`docker/fetch_iers_data.py`).

`docker/Dockerfile_setonix` — the AMD HIP image targeting Pawsey Setonix. Built atop `quay.io/pawsey/rocm-mpich-base`.

`docker/run_docker.sh` — a thin runner that wires `--gpus all`, IERS env vars, etc.

Image tags published to `jlbline/woden-2.6:cuda-{60,61,70,75,80,86,multi}` and `jlbline/woden-2.3:setonix`. To convert to Singularity:

```bash
singularity build woden-2.5-70.sif docker://jlbline/woden-2.3:cuda-70
singularity exec --nv --home=/astro/mwaeor/jline woden-2.5-70.sif run_woden.py --help
```

The Setonix image specifically *must not* use `--rocm` — that will fail because of cluster shenanigans.

---

## 27. Performance, accuracy, known limitations

### 27.1 Performance scaling

WODEN's `run_woden` cost is approximately

```
T  ≈  N_t · N_ν · N_bl · ( N_pt · c_pt   +   N_g · c_g   +   N_s · c_s )
```

where `c_pt`, `c_g`, `c_s` are the per-component visibility costs (point, Gauss, shapelet basis). Shapelets dominate when present (per-basis-function work plus a 1-D LUT lookup). Empirically (JOSS Table 4):

| Card | Mode | 207 k pt + 1.2 k Gauss + 62 shapelets (10 400 basis) × 14 t × 80 ν, MWA FEE |
|---|---|---|
| GTX 1080 Ti | float | 10 min 39 s |
| GTX 1080 Ti | double | 55 min 46 s (×5.2 slowdown) |
| V100 | float | 4 min 35 s |
| V100 | double | 5 min 55 s (×1.3 slowdown) |

Consumer cards have much less double-precision hardware than scientific cards; on a V100 the slowdown is ~1.3, on a GTX 1080 Ti it's ~5×.

Memory: dominated by `(num_baselines × num_times × num_freqs)` for the visibility output and `(num_components × num_freqs)` for the extrapolated flux arrays. With `num_baselines = 8128` (MWA, 128 tiles), `num_times = 240`, `num_freqs = 384`, four polarisations × two complex floats: ~5.6 GB just for the cross-correlations in float, doubled in double. This is why chunking the *sky* is mandatory.

### 27.2 Accuracy

JOSS paper Figure 1: with float precision the worst-case fractional error in Re/Im of `V` is ≲ 0.2 % at ≤ 10 km baselines. Double precision is < 2 × 10⁻⁶ % across the same range. Older fully-32-bit WODEN (v1.0) showed a few-percent error on long baselines.

The accuracy test machinery is in `test_installation/absolute_accuracy/`. It uses analytically tractable RIME outputs by setting `(u, v, w) = b·(1, 1, 1)`, sampling specific `φ_simple ∈ {0, π/6, π/4, π/3, π/2, 2π/3, 3π/4, 5π/6, π, 7π/6, 5π/4}`, derives `(l, m, n)` accordingly, runs WODEN, and compares the visibilities against `cos(φ), sin(φ)`.

### 27.3 Known limitations

1. **Single phase-centre.** All output uvfits are phased to a single `(RA0, Dec0)`. To phase-rotate later, the user must do it externally.
2. **No ionosphere.** No Z-Jones term. Add ionospheric distortions externally.
3. **No diffuse model intrinsic to WODEN.** Diffuse maps are converted into discrete points (every healpixel becomes a point source) before being fed in. This is the responsibility of the user (or `pygdsm` / `pyradiosky` / `pysm3` external converters).
4. **Beam pointing is fixed for a run.** All beams are locked in az/za for the duration of an observation. WODEN does not currently support tracking beams or per-time-step delay updates.
5. **MWA dipole flagging only on FEE / UVBeam.** Analytic and EveryBeam beams ignore the `--use_MWA_dipflags` and `--use_MWA_dipamps` flags.
6. **EveryBeam-MWA always uses one shared beam.** `single_everybeam_station` is forced to 1.
7. **EveryBeam dipole-amplitude mapping copies the amps to both X and Y polarisations.** Per-pol per-dipole amps are only available via `mwa_hyperbeam` (FEE) or pyuvdata (UVBeam).
8. **`MWA_FEE_interp` only valid 167–197 MHz.** Below or above, hyperbeam returns the boundary response without warning.
9. **Off-cardinal beam list (`BeamGroups.off_cardinal_beam_values`) is currently empty** — the LOFAR LBA dipole orientation is suspected off-cardinal but not yet confirmed (TODO in code).
10. **Custom EveryBeam fork.** The MWA `MWALocal` direction-input variant lives in Jack's fork. Upstream EveryBeam (≥ 0.7.2) is supported in WODEN ≥ 2.7 but the MWA path still relies on the fork-level `MWALocal`.
11. **Stokes-V list with negative fluxes** uses linear-space interpolation through zero-crossings; the docstring claims it tested OK against 21-cm power-spectra but worth flagging.
12. **`shapelet_basis.c::create_sbf` is legacy.** Python builds the table; the C function is left in case a future numpy change breaks the Python path.
13. **WODEN does not phase-rotate `(u, v, w)` on `--remove_phase_tracking`.** The visibilities are de-rotated, but `(u, v, w)` still describe the original phase centre. Downstream consumers like the RTS recompute uvw from antennas, so this is fine for them — but be aware.
14. **Single-precision NaN guard.** The mixing-matrix kernels do not check for NaNs; extreme primary-beam values near the horizon can occasionally produce NaN visibilities. Users should mask/flag below-horizon directions before simulation.
15. **CUDA/HIP feature parity is not strict.** HIP is supported for the kernels but not extensively tested. Expect a small accuracy difference vs CUDA on AMD cards.
16. **Author has stepped away.** Per the README, "I (Jack Line) am no longer working in astronomy. I'll drop in to advise and/or fix bugs from time to time, but I can't commit to developing new features." Community PRs are welcome.

---

## 28. Mapping WODEN concepts onto RadioSim

This section is included because RadioSim is the parent project for this monorepo. The terminology and architecture are deliberately similar in places, deliberately different in others.

| RadioSim concept | WODEN equivalent |
|---|---|
| `Simulator` (`api/simulator.py`) | `run_woden.py` |
| `RIMESimulator` (`simulator/rime.py`) | `int run_woden(...)` in `src/woden.c` plus `calculate_visibilities` in `src/calculate_visibilities_common.c` |
| `SkyModel` dataclass + `SkyFormat` | `source_catalogue_t` (a flat list of `source_t`s) — each WODEN chunk is a single source containing N point/Gauss/shapelet COMPONENTs |
| `JonesChain` (K Z T E P D G B) | WODEN does not have a chain; the `J` is computed once per primary beam (E equivalent) and applied directly. K (geometric) is the exp(-2πi(...)) inside the kernel. T, P, G, B, D are not modelled — the user adds them post-hoc via `add_instrumental_effects_woden.py`. Z is not modelled at all. |
| `BeamJones` family | One of `e_beamtype` selectors (NO_BEAM, GAUSS_BEAM, FEE_BEAM, ANALY_DIPOLE, FEE_BEAM_INTERP, MWA_ANALY, EB_OSKAR, EB_LOFAR, EB_MWA, UVB_MWA, UVB_HERA) |
| `PrecisionConfig` | The `--precision=double|float` flag chooses between `libwoden_double.so` and `libwoden_float.so`; precision is *fixed* per simulation |
| Backends (`numpy_backend.py`, `jax_backend.py`, `numba_backend.py`) | One of `do_gpu={0,1}` (CPU mirror or CUDA/HIP). No JAX. |
| `PolarizationLeakageJones` / `GainJones` / `BandpassJones` / `ParallacticAngleJones` | Only the parallactic angle is modelled (inside hyperbeam or via Python erfa). Other Jones terms are added post-hoc. |
| `MeasurementSet` writer (`io/measurement_set.py`) | uvfits writer (`wodenpy/uvfits/wodenpy_uvfits.py`); convert via `scripts/woden_uv2ms.py` |
| `core/sky/` healpix support (`SkyFormat.HEALPIX`) | Indirect — diffuse maps must be converted to point sources externally (via `pygdsm`, `pyradiosky`, `pysm3`). WODEN has no native HEALPix. |
| `shape='gaussian'` / `shape='disk'` | Elliptical Gaussian (envelope ξ_j) / *Shapelet* (envelope ξ_j with Hermite-Gaussian basis-function sum). No disk/Sersic. |
| Spectral-Index models (`PointSpectrum`, hybrids) | `MOD_TYPE = pl/cpl/nan` in the FITS file maps to `e_flux_type = POWER_LAW / CURVED_POWER_LAW / LIST` |
| `BBSReader` (`_loaders_bbs.py`) | `scripts/convert_WSClean_list_to_WODEN.py` (BBS → WODEN FITS) |
| Sky-region cropping | `wodenpy/skymodel/woden_skymodel.py::crop_below_horizon` (azimuthal / horizon-only) |
| `_loaders_diffuse.py` (gsm, gsm2008, gsm2016, lfsm, haslam, pysm3) | None — the user must produce a discrete sky from these models externally |
| `simulator/rime.py` chunking strategy | `wodenpy/skymodel/chunk_sky_model.py` — uses `binpacking` and per-component-type bookkeeping |
| ULSA support | None |
| `core/sky/_combine_*.py` | None (no model-combine in WODEN itself; just feed multiple sky model files and add resulting uvfits via `add_woden_uvfits.py`) |
| `core/sky/region.py` | None (no rectangular SkyRegion masking in WODEN — only horizon cropping) |

If you're porting WODEN-style features into RadioSim, the most directly transferable pieces are:

- The shapelet basis-function lookup table (`wodenpy/use_libwoden/shapelets.py`).
- The IAU vs MWA polarisation reordering (`make_baseline_date_arrays`, `IAU_order` flag in uvfits writer).
- The RTS-style array precession (`wodenpy/array_layout/precession.py`) — the `palpy.prenut(2000, mjd)` rotation matrix is the canonical recipe.
- The BBS → FITS converter (`scripts/convert_WSClean_list_to_WODEN.py`).
- The horizon cropping logic (`crop_below_horizon`).

Conversely, things RadioSim already does that WODEN does not:

- A full Jones chain with K, Z, T, E, P, D, G, B, plus extended terms (F, W, X, etc.);
- A native HEALPix sky-model representation that is preserved through the simulation rather than discretised;
- A Pydantic-based YAML config layer with explicit validation;
- Multiple backends (NumPy, JAX, Numba) with auto-selection;
- A diffuse-sky loader registry (GSM, LFSM, Haslam, PySM3, pyradiosky_file, FITS image, BBS, Vizier point-source catalogues).

In short, WODEN is fast, focused, and proven on real MWA/LOFAR/HERA EoR pipelines; RadioSim is broader-scope and more configurable. The numerical conventions (positive-exponent forward RIME, Stokes-coherency 1/2 factor, IAU `XX = N–S` internally) are aligned between the two projects.

---

## Appendix A — `run_woden.py` argument cheat-sheet

```
# Minimal: MWA-style metafits-driven sim
run_woden.py \
    --ra0=50.67 --dec0=-37.2 \
    --cat_filename=srclist_msclean_fornaxA_phase1+2.fits \
    --metafits_filename=1202815152_metafits_ppds.fits \
    --primary_beam=MWA_FEE

# Without metafits — full manual control
run_woden.py \
    --ra0=50.67 --dec0=-37.2 \
    --cat_filename=srclist.fits \
    --primary_beam=MWA_FEE \
    --MWA_FEE_delays=[6,4,2,0,8,6,4,2,10,8,6,4,12,10,8,6] \
    --lowest_channel_freq=169.6e+6 \
    --freq_res=10e+3 \
    --num_time_steps=240 --time_res=0.5 \
    --date=2018-02-28T08:47:06 \
    --array_layout=MWA_phase2_extended.txt

# CPU-only run with EveryBeam-LOFAR
run_woden.py \
    --ra0=0 --dec0=90 \
    --cat_filename=lobes.fits \
    --primary_beam=everybeam_LOFAR \
    --beam_ms_path=/data/lofar.ms \
    --array_layout=lofar_layout.txt \
    --latitude=52.905329712 --longitude=6.867996528 --array_height=0 \
    --lowest_channel_freq=46e+6 --freq_res=12.2e+3 \
    --num_time_steps=10 --time_res=2.0 \
    --date=2024-01-01T12:00:00 \
    --output_uvfits_prepend=lofar_lba \
    --cpu_mode --num_threads=16 \
    --band_nums=1
```

## Appendix B — Output-file naming convention

Per band:

```
<output_uvfits_prepend>_band%02d.uvfits
```

e.g. `--band_nums=4,7,24` with `--output_uvfits_prepend=epic_output` produces `epic_output_band04.uvfits`, `epic_output_band07.uvfits`, `epic_output_band24.uvfits`.

The lowest fine channel of band `k` is `lowest_channel_freq + (k − 1) · coarse_band_width` Hz.

## Appendix C — Glossary of WODEN-specific terms

| Term | Meaning |
|---|---|
| **SOURCE** | A logical group of one or more COMPONENTs. Used for sky-cropping (`--sky_crop_sources`) and for grouping a multi-component model like Fornax A |
| **COMPONENT** | A single point / Gaussian / shapelet element with a position and flux model |
| **chunk** | A `Skymodel_Chunk_Map` — a slice of components packed to fit on the GPU |
| **set** (of chunks) | A list of `num_threads` chunks; `num_rounds` of these are processed end-to-end |
| **band** | A coarse frequency band; one uvfits per band |
| **fine channel** | Frequency increment within a band |
| **IAU order** | `XX = N–S, YY = E–W` (proper convention) |
| **MWA / E-W order** | `XX = E–W, YY = N–S` (default uvfits output unless `--IAU_order`) |
| **on-cardinal dipoles** | Aligned 0/90° to north (e.g. MWA, LOFAR-HBA) |
| **off-cardinal dipoles** | Aligned 45/135° to north (some LOFAR-LBA) |
| **`woden_float` / `woden_double`** | The two compiled libraries; `--precision=` selects |
| **`run_woden`** | The C entry point; called by Python via ctypes |
| **`Visi_Set`** | A populated `visibility_set_t` (one per band) |

## Appendix D — Notable file sizes

| File | LOC |
|---|---|
| `src/source_components_gpu.cpp` | 2586 |
| `src/source_components_cpu.c` | 1408 |
| `wodenpy/use_libwoden/skymodel_structs.py` | 1360 |
| `scripts/run_woden.py` | 1096 |
| `wodenpy/wodenpy_setup/run_setup.py` | 1301 |
| `wodenpy/skymodel/read_fits_skymodel.py` | 1253 |
| `wodenpy/skymodel/chunk_sky_model.py` | 1162 |
| `wodenpy/primary_beam/use_everybeam.py` | 1491 |
| `wodenpy/skymodel/woden_skymodel.py` | 927 |
| `src/primary_beam_gpu.cpp` | 784 |
| `scripts/add_instrumental_effects_woden.py` | 759 |
| `src/primary_beam_cpu.c` | 646 |
| `src/call_everybeam.cc` | 527 |
| `wodenpy/use_libwoden/woden_settings.py` | 567 |
| `src/source_components_common.c` | 555 |
| `src/calculate_visibilities_common.c` | 500 |
| `wodenpy/uvfits/wodenpy_uvfits.py` | 457 |
| `wodenpy/array_layout/create_array_layout.py` | 467 |
| `wodenpy/primary_beam/use_uvbeam.py` | 431 |
| `include/woden_struct_defs.h` | 448 |
| `wodenpy/wodenpy_setup/woden_logger.py` | 408 |
| `wodenpy/skymodel/read_yaml_skymodel.py` | 391 |
| `wodenpy/skymodel/read_text_skymodel.py` | 353 |
| `src/fundamental_coords_gpu.cpp` | 269 |
| `src/calculate_visibilities_gpu.cpp` | 227 |

Total: ~25 kLOC of source (12 kLOC Python + 13 kLOC compiled), plus ~5 kLOC of tests and ~5 kLOC of documentation.

---

*This reference reflects the snapshot at `simulators/WODEN/` in the RadioSim monorepo, which corresponds to upstream WODEN 2.7-alpha. Always cross-check against the live `https://woden.readthedocs.io/` for the latest CLI flags, especially the EveryBeam path which changed materially between 2.5 and 2.7.*
