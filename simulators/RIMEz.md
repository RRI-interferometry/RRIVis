# RIMEz — Exhaustive Reference

**Radio Interferometric Measurement Equation(s)** — a research-grade Python package for computing radio-interferometric visibilities directly from a polarised RIME formulation. Built primarily for the HERA (Hydrogen Epoch of Reionization Array) science use-case but architecturally general. Authored at UPenn (UPennEoR) by Zachary Martinot, with contributions from Paul La Plante and Steven Murray.

This document is an exhaustive code-level reference: every module, every function, every algorithm, every numeric/coordinate convention, every test, and every external dependency is described. The intent is that a reader of this single file can understand the package in the same depth as someone who has read the source line-by-line.

> Source repository (upstream): `https://github.com/UPennEoR/RIMEz`. Documentation domain stub: `rimez.readthedocs.org` (mostly unpopulated). Last `CHANGELOG.rst` released version: `0.1.1`. Local copy in this monorepo lives under `simulators/RIMEz/` and tracks an early-2020-era snapshot of the project.

---

## Table of Contents

1. [Project metadata, license, authorship](#1-project-metadata-license-authorship)
2. [What RIMEz does — scientific overview](#2-what-rimez-does--scientific-overview)
3. [Repository layout](#3-repository-layout)
4. [Build system, dependencies, and the Fortran stack](#4-build-system-dependencies-and-the-fortran-stack)
5. [Top-level package (`src/RIMEz/__init__.py`)](#5-top-level-package-srcrimezinitpy)
6. [`rime_funcs.py` — core RIME kernels](#6-rime_funcspy--core-rime-kernels)
7. [`management.py` — high-level orchestration classes](#7-managementpy--high-level-orchestration-classes)
8. [`utils.py` — coordinates, antenna utilities, UVData export](#8-utilspy--coordinates-antenna-utilities-uvdata-export)
9. [`beam_models.py` — Jones-matrix beams and basis transforms](#9-beam_modelspy--jones-matrix-beams-and-basis-transforms)
10. [`sky_models.py` — point-source & diffuse sky harmonics](#10-sky_modelspy--point-source--diffuse-sky-harmonics)
11. [`dfitpack_numba.py` & `dfitpack_wrappers/` — CFFI bridge to FITPACK](#11-dfitpack_numbapy--dfitpack_wrappers--cffi-bridge-to-fitpack)
12. [Test suite](#12-test-suite)
13. [Notebooks and docs](#13-notebooks-and-docs)
14. [End-to-end usage flow](#14-end-to-end-usage-flow)
15. [Conventions — coordinates, polarisation, sign, basis](#15-conventions--coordinates-polarisation-sign-basis)
16. [Known issues / quirks observed in this snapshot](#16-known-issues--quirks-observed-in-this-snapshot)

---

## 1. Project metadata, license, authorship

| Item | Value |
|---|---|
| Package name | `RIMEz` |
| Description | Methods and input models for computing radio-interferometric visibilities |
| License | MIT (Copyright (c) 2019 UPennEoR) |
| Author of record | Zachary Martinot (`zmarti@sas.upenn.edu`) |
| Other contributors | Paul La Plante (`plaplant@berkeley.edu`), Steven Murray (`steven.g.murray@asu.edu`) |
| Development status | Beta (4) |
| Python | `>= 3.6` (declared); CI tests `3.6, 3.7, 3.8` on Linux + macOS |
| Build backend | `setuptools >= 30.3.0` + `wheel` + `setuptools_scm` (declared in `pyproject.toml`) |
| Scaffold | PyScaffold 3.2.3 |
| Top-level entry points | None (pure-library; no console scripts) |
| Style tooling | Black (line length 88), isort, flake8 (configured in `setup.cfg`/`.flake8`/`.isort.cfg`), pre-commit (`.pre-commit-config.yaml`) |
| Doc generator | Sphinx (`docs/conf.py`); only stubs present (License/Authors/Changelog/api index) |
| Distribution build | `bdist_wheel` (universal=1 declared, but the embedded Fortran object makes pure-python wheels misleading) |

Optional extras (declared in `setup.cfg` `[options.extras_require]`):

- `gsm` → installs `pygsm` from `git+git://github.com/telegraphic/PyGSM`.
- `all` → currently equivalent to `gsm` (each dep duplicated, no “group of groups”).
- `testing` → `pytest`, `pytest-cov`.
- `docs` → `sphinx`.
- `dev` → `pytest`, `pytest-cov`, `sphinx` (catch-all developer install).

Two GitHub-only deps are commented-out in `setup.cfg` but are functionally required at import time of `RIMEz.beam_models` and `RIMEz.management`:

- `ssht_numba` (`git+https://github.com/UPennEoR/ssht_numba`) — Numba-callable SSHT (Spin-1 Spherical Harmonic Transform) routines (`mw_sample_positions`, `bad_meshgrid`, `mw_forward_sov_conv_sym`, `mwss_sample_grid`, `mwss_sample_grid`, `dl_m`, `generate_dl`, `elm2ind`, `ind2elm`, `mw_forward_sov_conv_sym_ss`, `mw_forward_sov_conv_sym_ss_real`).
- `spin1_beam_model` (`git+https://github.com/UPennEoR/spin1_beam_model`) — supplies `AntennaFarFieldResponse` (E-field FITS handler, spline approximations) and `cst_processing.ssht_power_spectrum`.

The CI workflow (`.github/workflows/test_suite.yaml`) explicitly installs both via `pip install git+...` (using `numba<0.49.0`) and pulls `healpy`, `pyuvdata`, `fftw` from `conda-forge`. The recipe also adds `CPATH`/`LIBRARY_PATH` so `ssht_numba` finds FFTW headers/libs.

---

## 2. What RIMEz does — scientific overview

The Radio Interferometer Measurement Equation (RIME) of Hamaker–Bregman–Sault decomposes a baseline visibility between two antennas `p` and `q` as

```
V_pq(ν, t) = ∑_s J_p(ŝ, ν, t) · C_s(ν) · J_q^H(ŝ, ν, t) · F_p(ŝ) F_q^*(ŝ)
```

where `J_p` is the 2×2 antenna Jones matrix, `C_s = (1/2)(I σ₀ + Q σ₁ + U σ₂ + V σ₃)` is the source coherency matrix in Pauli basis, and `F = exp(-2π i ν b·ŝ / c)` is the geometric fringe term for a baseline `b`.

RIMEz provides two evaluation strategies for this equation:

1. **Point-sample (direct DFT-style sum)** — `rime_funcs.parallel_point_source_visibilities` — explicit O(N_time × N_freq × N_baseline × N_src) summation over a discrete catalog of sources, with each antenna’s instantaneous Jones evaluated at every up-source on every loop iteration. Fully polarised (`Stokes I, Q, U, V`).
2. **m-mode (harmonic / Fourier-on-the-sky)** — `rime_funcs.parallel_mmode_unpol_visibilities` — for unpolarised (Stokes-I-only) skies represented as MW-sampled spherical harmonics `Slm`. The integration kernel `K(t, b̂, ν, ŝ)` (Jones·fringe·Jones†) is itself sampled on the MW grid and forward-transformed via `ssht_numba.mw_forward_sov_conv_sym`, then collapsed against `Slm` to yield the **m-modes** of the visibility — coefficients of the Fourier expansion of `V(ν, t)` in earth-rotation angle. Time samples are synthesised from m-modes by DFT (`visibility_dft_from_mmodes`, `parallel_visibility_dft_from_mmodes`) or by FFT-zeropad + cubic-spline interpolation (`visibility_from_mmodes`).

The high-level abstraction `management.VisibilityCalculation` selects between the two methods based on which sky payload the user provides:

- `Slm=...` → harmonics path → `compute_fourier_modes()` → `compute_time_series()` (DFT).
- `S=..., RA_icrs=..., Dec_icrs=...` → point-samples path → `compute_time_series()` directly.

Both paths produce a 5-D visibility tensor of shape `(N_time, N_freq, N_baseline, 2, 2)` (complex), polarised in the antenna’s 2-element instrumental basis. Conversion to Stokes (or to `pyuvdata.UVData`) is done by `utils.uvdata_from_sim_data`.

---

## 3. Repository layout

```
RIMEz/
├── .coveragerc                 # coverage config (omits dfitpack_wrappers)
├── .flake8                     # flake8 ignores
├── .gitignore
├── .isort.cfg                  # isort profile
├── .pre-commit-config.yaml     # pre-commit hooks (black, isort, flake8, …)
├── .github/
│   └── workflows/
│       ├── test_suite.yaml     # GH-Actions test matrix (Linux+macOS, py36/37/38)
│       └── pre_commit.yaml     # pre-commit lint job
├── AUTHORS.rst                 # Zachary Martinot, Paul La Plante, Steven Murray
├── CHANGELOG.rst               # v0.1.1 notes
├── LICENSE.txt                 # MIT (2019 UPennEoR)
├── README.rst                  # install + dev instructions
├── pyproject.toml              # build-system + Black config
├── setup.py                    # custom build that calls `make` on Fortran
├── setup.cfg                   # PyScaffold-driven metadata, deps, extras
├── ci/                         # conda install scripts (Linux/macOS)
├── docs/                       # Sphinx scaffolding (mostly empty)
│   ├── conf.py
│   ├── index.rst
│   ├── license.rst
│   ├── authors.rst
│   ├── changelog.rst
│   ├── _static/
│   └── _templates/
├── notebooks/
│   └── monopole_vs_no_monopole.ipynb
├── src/
│   └── RIMEz/
│       ├── __init__.py             # version discovery only
│       ├── rime_funcs.py           # ★ core RIME kernels (581 lines)
│       ├── management.py           # ★ VisibilityCalculation, PointSourceSpectraSet
│       ├── utils.py                # ★ coordinates, hex grids, UVData export (745 lines)
│       ├── beam_models.py          # ★ Jones beams + basis transforms (435 lines)
│       ├── sky_models.py           # ★ harmonics, GSM, gridding kernels (541 lines)
│       ├── dfitpack_numba.py       # CFFI bridge to FITPACK
│       └── dfitpack_wrappers/
│           ├── bispeu.f, fpbisp.f, fpbspl.f   # FITPACK Fortran 77 source
│           ├── dfitpack_wrappers.f90          # ISO_C_BINDING wrapper
│           ├── Makefile                       # gfortran build → .so
│           └── .gitignore
└── tests/
    ├── conftest.py                  # session-scope fixtures
    ├── test_beam_models.py
    ├── test_management.py
    ├── test_rime_funcs.py
    ├── test_sky_models.py
    ├── test_utils.py
    └── data/
        ├── __init__.py
        └── generate_test_data.py    # canonical 1-source visibility regression
```

The total Python codebase is ~3.2 kLOC; the Fortran wrapper layer is ~22 lines of f90 plus three FITPACK F77 files (bispeu, fpbisp, fpbspl) inherited verbatim from `scipy/interpolate`.

---

## 4. Build system, dependencies, and the Fortran stack

### 4.1 Hard runtime dependencies

Declared in `setup.cfg` `[options] install_requires`:

```
numpy
numba
cffi
astropy
h5py
scipy
healpy
pyuvdata
```

Implicit (must be installed separately because of `git+` dependency form, see §1):

- `ssht_numba`
- `spin1_beam_model`

Implicit native:

- `gfortran` (compile-time, for `dfitpack_wrappers.so`)
- `fftw` **shared** library (used by `ssht_numba`; the README warns: must be `--enable-shared` if compiled by hand). Path overridable via env var `FFTW_PATH`.

### 4.2 Optional dependencies

- `pygsm` — required only when calling `sky_models.diffuse_sky_model_from_GSM2008` or `sky_models.diffuse_sky_model`. Imported lazily at top of `sky_models.py` inside a `try/except ImportError` (the missing-pygsm branch raises a clear `ImportError` at call time).

### 4.3 Build sequence

`setup.py` defines `CustomBuild(distutils.command.build.build)` which (in this snapshot, only partially complete — the file is truncated at line 32) is intended to:

1. Find `make` on `PATH`.
2. `cd src/RIMEz/dfitpack_wrappers/` and run `make`, producing the shared object `dfitpack_wrappers.so`.
3. Continue with the standard `setuptools` build.

The Makefile (`src/RIMEz/dfitpack_wrappers/Makefile`):

```makefile
FC     = gfortran
FFLAGS = -O3 -fPIC -g -shared

OBJ    = fpbspl.o fpbisp.o bispeu.o
LIBOBJ = dfitpack_wrappers.o

dfitpack_wrappers.so: $(OBJ) $(LIBOBJ)
        $(FC) $(FFLAGS) $(OBJ) $(LIBOBJ) -o $@

%.o: %.f
        $(FC) $(FFLAGS) -c $*.f
dfitpack_wrappers.o: $(OBJ)
        $(FC) $(FFLAGS) -c dfitpack_wrappers.f90
```

The `.so` is then `dlopen`-ed at import time by `dfitpack_numba.py` via CFFI.

### 4.4 What the Fortran wrapper does

`dfitpack_wrappers.f90` exposes the FITPACK routine `bispeu` (un-gridded bivariate B-spline evaluation) under the C ABI:

```fortran
subroutine bispeu_wrap(tx,nx,ty,ny,c,kx,ky,x,y,z,m,wrk,lwrk,ier) bind(c)
  integer(4), intent(in)  :: nx,ny,kx,ky,m,lwrk
  real(8),    intent(in)  :: tx(nx),ty(ny),c((nx-kx-1)*(ny-ky-1)), x(m),y(m),wrk(lwrk)
  integer(4), intent(out) :: ier
  real(8),    intent(out) :: z(m)
  call bispeu(tx,nx,ty,ny,c,kx,ky,x,y,z,m,wrk,lwrk,ier)
end subroutine bispeu_wrap
```

This wrapper exists because `scipy.interpolate.RectBivariateSpline.__call__` is *not* `nb.njit`-callable, but the spline beam evaluation must run inside Numba-compiled visibility kernels. Routing the same `tx, ty, c, kx, ky` arrays through a `bind(c)` Fortran symbol via CFFI restores njit compatibility while preserving bit-for-bit equivalence with `scipy`.

---

## 5. Top-level package (`src/RIMEz/__init__.py`)

Tiny — only handles version discovery via `pkg_resources`:

```python
from pkg_resources import DistributionNotFound, get_distribution

try:
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound
```

There is **no re-export of submodule symbols**; users must explicitly do `from RIMEz import management, rime_funcs, sky_models, beam_models, utils`.

---

## 6. `rime_funcs.py` — core RIME kernels

The heart of the package. 581 lines. Every routine is `@nb.njit` or `@nb.guvectorize`-compiled. Imports:

```python
import numba as nb
import numpy as np
import ssht_numba as sshtn
from scipy import interpolate
```

### 6.1 Pauli sigma tensors

```python
@nb.njit
def make_sigma_tensor() -> complex128[2,2,4]
@nb.njit
def make_bool_sigma_tensor() -> bool[2,2,4]
```

These build the mapping from Stokes (I, Q, U, V) → coherency-matrix component. Indexing: `sigma[:, :, 0] = I_2`, `sigma[:, :, 1] = diag(1,-1)` (Q), `sigma[:, :, 2] = [[0,1],[1,0]]` (U), `sigma[:, :, 3] = [[0,-i],[i,0]]` (V). The `bool` companion is a mask of which `(b,c)` matrix entries of `sigma[:, :, g]` are non-zero — used to skip zero-flop branches inside the inner reduction.

### 6.2 Coordinate kernel

```python
@nb.njit
def fast_approx_radec2altaz(ra, dec, R) -> (s, alt, az)
```

A *flat-Earth*-style geometric transform: it builds the unit-vector array
`p = (cos ra cos dec, sin ra cos dec, sin dec)`, applies the rotation matrix `R`
(passed as `R.T` from callers), and then derives Alt/Az via `arctan` and `arctan2`.
Wrap-around to `[0, 2π)` is enforced. The `R` matrix is supplied by `utils.get_rotations_realistic_from_JDs` or `utils.get_rotations_idealized` and embodies the full ICRS→AltAz frame chain at a particular epoch.

### 6.3 The polarised visibility-matrix accumulator

```python
@nb.njit
def RIME_sum(J1, J2_conj, F1, F2_conj, S, sigma, bsigma) -> complex128[2,2]
```

Implements

```
V[a,d] = Σ_n F1_n F2_conj_n * Σ_{b,c} J1_n[a,b] (Σ_g σ[b,c,g] S_n[g]) J2_conj_n[d,c]
```

i.e. the 2×2 visibility for one (frequency, baseline, time) cell, summed over `Np` sources of polarisation vector `S[n,:] = [I,Q,U,V]`. Note the explicit transpose semantics: `J2_conj` is indexed `[d,c]` rather than `[c,d]`, because we want `J_p · C · J_q^†` and `J_q^†_{dc} = (J_q*)_{cd}`. The boolean sigma mask short-circuits the inner `g` loop on Pauli zeros.

### 6.4 Vectorised point-source visibility kernel

`vec_psv_constructor(beam_funcs, compile_target="parallel")` returns a closure-built `nb.guvectorize`-decorated function. The signature string is

```
"float64[:,:], float64[:], float64[:,:],"
"int64[:,:], int64[:],"
"float64[:,:,:], float64[:], float64[:], complex128[:,:,:,:]"
"(c,c), (f), (a,c),(b,j),(a),(f,n,s),(n),(n)->(f,b,j,j)"
```

i.e. for one rotation matrix `R_i` (3×3), a frequency axis (`f`), antenna positions (`a`, with 3-vec each), antenna pairs (`b`, with 2 ant indices), an `ant→beam_func_index` map, the polarised sky tensor `S(f, n, 4)`, RA, Dec, the kernel produces `V_i(f, b, 2, 2)`. The closure binds the user’s `beam_funcs` so that `nb.njit` can compile a non-generic call site.

Algorithm for `vec_psv` (per rotation `i`):

1. Build sigma/bsigma tensors.
2. `s, alt, phi = fast_approx_radec2altaz(RA, dec, R_i.T)`.
3. **Reconstruct** `s` from `(alt, phi)` — this preserves the ENU ordering `(E, N, U) = (sin φ cos alt, cos φ cos alt, sin alt)` (note the swapped sin/cos relative to the textbook ENU; see §15).
4. `v_inds = where(alt > 0)` — the visible-hemisphere mask; below-horizon sources are silently dropped (`V_i := 0` when no source is up).
5. Compute the per-baseline geometric phase `τ_g = -2π/c · r · ŝ_v.T` *once per rotation* (independent of frequency).
6. For each frequency `j`:
   - Multiply by `nu_j` to get phases, then `F = e^{i*phases} = cos + i sin` (note: `+i` here, because `τ_g` already carries the minus sign).
   - For each unique beam-function index `i_bf`, evaluate `J_arr[i_bf] = beam_funcs(index, nu_j, alt_v, phi_v)` (3-D Jones array of shape `(N_visible, 2, 2)`).
   - For each baseline `(p,q)`: pull `F_p, F_q^*`, the per-antenna Jones `J_p, J_q^*`, and call `RIME_sum`.

`parallel_point_source_visibilities(rotations_axis, nu_axis, r_axis, ant_pairs, beam_funcs, ant_ind2beam_func, S, RA, dec)` is the user-facing wrapper that allocates the `(N_t, N_f, N_bl, 2, 2)` complex output and invokes the guvectorised kernel along the leading time axis.

### 6.5 m-mode (harmonic) visibility kernel — unpolarised

`vec_muv_constructor(beam_funcs, compile_target="parallel")` builds a guvectorised analogue for the *m-mode* method. Signature string:

```
"float64, float64[:,:],"
"int64[:,:], int64[:],"
"complex128[:,:], float64[:,:], int64, int64, float64[:], complex128[:,:,:,:]"
"(),(a,c),(b,j),(a),(n,s), (c,c),(),(),(m)->(b,m,j,j)"
```

Inputs (per frequency `nu`): antenna positions `r_axis`, ant pairs, beam map, `Slm` of shape `(N_lm, 4)` (only `Slm[:, 0] = Ilm` is used because the inner loop hardcodes `g=0` (Stokes I) via `bsigma[i_b, i_c, 0]`), reference rotation matrix `R_0`, source bandlimit `Ls`, kernel bandlimit `Lb`, a dummy m-axis array (only present so `guvectorize` can pick up the `m` output dimension), and the output `V_nu_m` of shape `(N_bl, 2L_m-1, 2, 2)`.

Inner algorithm:

1. `Lm = min(Ls, Lb)`.
2. Sample MW grid: `beta_t, alpha_t = sshtn.mw_sample_positions(Lb)`, then `bad_meshgrid(alpha_t, beta_t)` produces the full 2-D `(α, β)` grid. Map to `(RA, Dec) = (α, π/2 - β)` and feed into `fast_approx_radec2altaz` with `R_0.T` to get the AltAz at the reference epoch.
3. Recompute ENU `s` from `(alt, az)` (same swapped convention as §6.4).
4. Compute the geometric phase for every grid point and every antenna: `phases = -2π ν/c · r · ŝ.T`, fringe `F = e^{i phases}`.
5. For every unique beam index, evaluate the Jones on the visible mask only.
6. For each baseline:
   - Build the 4-D kernel `K_CI[a,d,θ,φ] = Σ_{b,c} F_p A_p_ab σ[b,c,0] (F_q A_q_dc)^*`. This is exactly the Stokes-I projection of `J_p · σ_0 · J_q^†` with the geometric fringe folded in. Indexed as `(2,2,Lb,2*Lb-1)`.
   - For each `(a,b)` element, run `sshtn.mw_forward_sov_conv_sym(K_CI_ab, Lb, 0, Klm_ab)` — a separation-of-variables forward SHT — yielding `Klm_cI` of shape `(2,2,Lb²)`.
   - Accumulate the m-modes:
     ```
     V_nu_m[k, m+Lm-1, a, b] += Σ_{ℓ=|m|}^{Lm-1} Klm_cI[a, b, ind(ℓ, m)] · Slm[ind(ℓ, m), 0]^*
     ```
7. Result is the visibility’s Fourier-on-the-sphere coefficients in `m` (the conjugate variable to earth-rotation angle).

`parallel_mmode_unpol_visibilities(...)` is the user-facing wrapper that allocates `Vm` of shape `(N_freq, N_bl, 2*Lm-1, 2, 2)` and calls the guvectorised kernel.

A pure-`@nb.njit` (single-thread) variant `mmode_unpol_visibilities` is kept around — same algorithm, no `guvectorize`, same output. Used for verification / debugging.

### 6.6 Time-series synthesis from m-modes

```python
@nb.njit
def visiblity_dft_from_mmodes(era_axis, Vm) -> V_dft
```

Sums the truncated Fourier series `V(τ) = Σ_m Vm * e^{-i m τ}` for an array of differential earth-rotation angles `τ = era - era_ref`. Output shape `(N_t, N_freq, N_bl, 2, 2)`.

```python
def parallel_visibility_dft_from_mmodes(era_axis, Vm, delta_t)
```

Wraps an inner `@nb.njit(parallel=True, nogil=True)` routine `inner_parallel_visiblity_dft_from_mmodes`. Adds two refinements:

- **Integration smoothing** via the analytical sinc kernel `sinc(m · Δera/2)` where `Δera = ω_⊕ · Δt`, with `ω_⊕ = 7.292115e-5 rad/s` (USNO Circular 179 page 16) — i.e. closed-form integration of the visibility over the integration period rather than instantaneous sample.
- A `delta_t == 0` guard that bumps to `1e-20 s` to avoid `0/0`, plus an explicit `m=0` carve-out where `x` is replaced by `1e-20` before computing `sin(x)/x`.

The structural awkwardness of needing two functions (an outer allocator + inner parallel filler) is documented in a comment: `parallel=True` njit functions cannot allocate the output array themselves in this code path.

### 6.7 FFT-based time-series synthesis

```python
def vectorize_vis_mat(vmat_in)        # (N, 2, 2) complex → (N+1, 8) real
def devectorize_vis_vec(vvec_in)      # inverse
def visibility_from_mmodes(Vm, era_axis, up_sampling=10)
```

`visibility_from_mmodes` zero-pads `Vm` along the m-axis by an `up_sampling` factor, FFTs back to the era domain, and then uses `scipy.interpolate.splprep` (periodic cubic B-spline, `per=1`, `k=3`) to evaluate at user-specified `era_axis` samples. Each 2×2 complex matrix is flattened into 8 real degrees of freedom, splined separately, and recombined. A phase shift `ang_shift_up = π - 2π(L_fftup-1)/(2L_fftup-1)` is applied to undo the half-pixel offset introduced by the zero-pad. The output shape matches the DFT path: `(N_t, N_freq, N_bl, 2, 2)`.

---

## 7. `management.py` — high-level orchestration classes

445 lines. Two public classes: `VisibilityCalculation` and `PointSourceSpectraSet`. Plus a small helper `_get_versions()` returning a dict `{ "RIMEz": __version__, "ssht_numba": sshtn.__version__, "spin1_beam_model": s1b.__version__ }` used to stamp output HDF5 files.

### 7.1 `VisibilityCalculation`

A stateful container that bundles a parameter dictionary, sky data, beam function, and the resulting `Vm` / `V` arrays. Two ways to construct:

- **From scratch:** `VisibilityCalculation(parameters, beam_func=..., Slm=...)` *or* `(parameters, beam_func=..., S=..., RA_icrs=..., Dec_icrs=...)`.
- **Restored:** `VisibilityCalculation(restore_file_path="...h5")` — re-hydrates a previously saved calculation; sky inputs and beam function are all set to `None` because they were never written.

Required parameter keys (validated in `__init__`):

```
array_latitude, array_longitude, array_height,
initial_time_sample_jd, integration_time, frequency_samples_hz,
antenna_positions_meters, antenna_pairs_used,
antenna_beam_function_map, integral_kernel_cutoff
```

Each key is set as an instance attribute via `setattr`. Method dispatch is governed by `self.calculation_method`:

- `"harmonics"` if `Slm` was provided.
- `"point_samples"` if `S, RA_icrs, Dec_icrs` were provided.

`setup()` builds the array `EarthLocation` and pre-computes:

- For harmonics: `R_0 = utils.get_rotations_realistic_from_JDs(jd0, array_location, reference_jd=jd0)` (the reference rotation matrix at the initial time), and `self.Lss = sshtn.ind2elm(Slm.shape[1])[0]` (the source bandlimit derived from `Slm` length).
- For point samples: ICRS→CIRS frame transform (`utils.transform_icrs_to_cirs`) and the per-time rotation matrices `rotations_axis` against the reference `jd0`.

Methods:

- `compute_fourier_modes()` — calls `parallel_mmode_unpol_visibilities`, multiplies the result by `0.5` (the Pauli factor in `C = ½(I σ₀ + …)`), and stores `self.Vm`.
- `compute_time_series(time_sample_jds=None, integration_time=None)` — for the harmonics path, optionally lazily computes `Vm`, then converts ERA to differential ERA, and synthesises `V` via `parallel_visibility_dft_from_mmodes`. For the point-samples path, it goes straight to `parallel_point_source_visibilities`. (Note the `0` integration bumps to `1e-15` here, vs `1e-20` inside the kernel — historical inconsistency.)
- `write_visibility_fourier_modes(file_path, overwrite=False)` — writes every parameter + `Vm` + a per-package version stamp to HDF5.
- `write_visibility_time_series(file_path, overwrite=False)` — same but writes `V`.
- `load_visibility_calculation(file_path)` — re-reads an HDF5 file, populating `self.parameters` (excluding `V`, `Vm`) and setting attributes for everything.
- `to_uvdata(...)` — calls `utils.uvdata_from_sim_data` with sensible defaults to produce a `pyuvdata.UVData`. Default `telescope_name="probably HERA, but who knows?"` (literally — see source).
- `write_uvdata_time_series(uvdata_file_path, clobber=False, ...)` — persists via `uvd.write_uvh5`.

### 7.2 `PointSourceSpectraSet`

Manages a discrete catalog (RA/Dec/Iflux per frequency) and its conversion to spherical-harmonic coefficients. Constructor accepts either inputs or a `file_path` to an HDF5 file. Frame label `coordinates` defaults to `"GCRS"`. Method `generate_harmonics(L)` runs `sky_models.point_sources_harmonics_with_gridding`. Operator overloads:

- `__add__` / `__radd__`: concatenates two sets only if `nu_mhz`, `coordinates`, and `L` match; sums their `Ilm` arrays if both have them. Otherwise raises a clear `ValueError`.

I/O methods `save_to_file` / `load_from_file` write `nu_mhz, Iflux, RA, Dec, coordinates`, and (if computed) `Ilm, L`. The loader has a backward-compat warning: old files used the dataset name `"I"` instead of `"Iflux"`.

---

## 8. `utils.py` — coordinates, antenna utilities, UVData export

745 lines. A grab-bag, but cleanly partitioned into four sections.

### 8.1 HERA constants

```python
HERA_LAT    = np.radians(-30.72152777777791)
HERA_LON    = np.radians( 21.428305555555557)
HERA_HEIGHT = 1073.0000000093132   # meters (the trailing digits are an `astropy` round-trip artefact)
```

Used as defaults throughout the test suite and the test data generator.

### 8.2 General helpers

- `coords_to_location(lat_rad, lon_rad, height_m) -> astropy.coordinates.EarthLocation`.
- `kernel_cutoff_estimate(max_baseline_m, max_freq_hz, width_estimate=100) -> int` — returns the *m-mode bandlimit* `L = ceil(2π ν b / c) + width`, rounded up to the next even integer. The `width_estimate` is a heuristic safety margin reflecting the kernel's spectral spread beyond `ℓ_peak`. This is the canonical recipe for choosing `integral_kernel_cutoff` in `VisibilityCalculation`.

### 8.3 Antenna array utilities (HERA-style)

- `b_arc(b, precision=3)` — for a baseline 3-vector `b`, returns `‖b‖ · arctan(b_y/b_x)` rounded to `precision` decimals (a degenerate "redundancy fingerprint"). Returns `nan` for the auto-baseline `(0,0,0)`, and `π/2` if `b_x == 0`.
- `B(b, precision=3)` — `np.linalg.norm(b)` rounded.
- `get_minimal_antenna_set(r_axis, precision=3)` — finds a *redundant baseline group* representative for each unique `(B, b_arc)` cluster via two-level dictionary `u2a[B][arc]` and the inverse `a2u[(i,j)]`. Each group representative is the first `(i,j)` with `b_x ≥ 0` (and not the negative-y zero-x edge case). Returns `(minimal_ant_pairs, u2a, a2u)`. Aimed at hex lattices; explicitly cautious about general arrays.
- `generate_hex_positions(lattice_scale=14.7, u_lim=3, v_lim=3, w_lim=3)` — generates a closed-packed hex array on three lattice directions `e_u, e_v, e_w` separated by 120° each. Outputs antenna ENU positions (`z=0`) in metres. Defaults match the physical HERA spacing.

### 8.4 Coordinate / time conversions

- `JD2era(JD)` — uses ERFA `era00(jd1, jd2)` with UT1 scale; returns ERA in `[0, 2π)`.
- `JD2era_tot(JD)` — *total* (unwrapped) ERA from USNO Circular 179 Eqn 2.10:
  `theta = 2π · (0.7790572732640 + 1.00273781191135448 · D_U)`, where `D_U = JD - 2451545.0`.
- `era2JD(era, nearby_JD)` — Newton inversion of `JD2era_tot`.
- `era_tot2JD(theta)` — closed-form inverse: `JD = (theta/(2π) - a)/b + 2451545`.
- `get_rotations_realistic_from_JDs(jd_axis, array_location, reference_jd=None)` — for each JD, rotates a sky-fixed cartesian basis (in `gcrs` if `reference_jd is None`, else `cirs` at `reference_jd`) into AltAz at that JD via `astropy.coordinates`, projects onto an orthogonal matrix using `scipy.linalg.orthogonal_procrustes` (because astropy’s round-trip is not perfectly orthogonal), and returns the transposed rotation. Output shape `(N_jd, 3, 3)`.

  Two notable quirks in this snapshot: (1) the dtype is set with `np.float` (deprecated alias) and (2) the function returns `R = transpose(Rt)` — i.e. the inverse of what `orthogonal_procrustes` produces by default.
- `get_rotations_idealized(era_axis, array_location)` — analytical alternative that skips astropy entirely. Builds:
  ```
  R_hd2aa = [[-sin lat, 0, cos lat], [0, -1, 0], [cos lat, 0, sin lat]]
  R_ad2hd(θ) = [[cos θ, sin θ, 0], [sin θ, -cos θ, 0], [0,0,1]]   (note: not standard rotation; reflection-like)
  ```
  Useful for noise-free regression tests.
- `get_icrs_to_gcrs_rotation_matrix()` — same Procrustes pattern, returning the orthogonal projection of astropy’s ICRS→GCRS basis transform.
- `get_galactic_to_gcrs_rotation_matrix()` — likewise for Galactic→GCRS.
- `transform_icrs_to_cirs(RA_icrs, Dec_icrs, reference_jd)` — wraps `astropy.coordinates.SkyCoord.transform_to(CIRS)`. Note: the body refers to `RA_icrs_deg`/`Dec_icrs_deg` (typo bug) and treats the inputs as degrees rather than radians — this is one of the snapshot’s broken paths (see §16).

### 8.5 Beam-derived integrals

- `beam_func_to_Omegas_ssht(nu_hz, beam_func, L_use=200, beam_index=0)` — computes per-frequency beam solid angles via the SSHT MWSS sample grid:
  - `B(α, β) = |J_00|² + |J_01|²` (magnitude-squared of the first row, i.e. one polarisation’s beam intensity pattern),
  - `Ω = sqrt(4π) · Re(B_lm[0,0])`, and
  - `Ω'' = Σ_{lm} |B_lm|²`.
- `beam_func_to_Omegas_healpix_sum(nu_axis, beam_func, nside=128, beam_index=0)` — same quantities computed by direct healpix sum (rectangle rule). A consistency check against the SSHT version.
- `beam_func_to_kernel_power_spectrum(nu_hz, b_m, beam_func)` — angular power spectrum of the Stokes-I→Vokes-I integration kernel `K00 = M00 · F` where `M00 = ½ Tr(J J^†)` and `F = exp(-2π i ν b·ŝ/c)`. Picks `L_use = max(2 ℓ_peak, 350)` automatically. Useful for choosing `integral_kernel_cutoff` empirically.

### 8.6 UVData export

`uvdata_from_sim_data(...)` is a hand-rolled assembly of a `pyuvdata.UVData` from the simulation arrays. Key behaviours:

- Telescope location set from `(array_lat, array_lon, array_height)`.
- LSTs derived via `astropy.time.Time.sidereal_time("apparent")`.
- `integration_time = "derived"` → uses `(jd_axis[1] - jd_axis[0]).sec`.
- `channel_width = "derived"` → uses `nu_axis[1] - nu_axis[0]`.
- `antenna_numbers / antenna_names = "derived"` → uses `arange(N_ant)` and string casts thereof.
- Antenna positions: `r_axis` (assumed ENU) is converted to ECEF via `pyuvdata.utils.ECEF_from_ENU(...)`, then the telescope ECEF is subtracted to give the relative antenna positions (the format `pyuvdata` expects).
- Polarisation order is fixed to `["xx","yy","xy","yx"]`.
- `phase_type = "drift"` (driftscan).
- The polarisation-index map depends on `x_is_north` (default `True`):
  - If `x_is_north=True`: `pol_map = {(1,1):0, (0,0):1, (1,0):2, (0,1):3}` (i.e. `x ↔ North`); sets `uvd.x_orientation = "NORTH"`.
  - Else: `(0,0):0, (1,1):1, (0,1):2, (1,0):3`; sets `"EAST"`.
- Calls `uvd.set_uvws_from_antenna_positions()` to fill `uvw_array`.
- Final `uvd.check()`.

The companion `inflate_uvdata_redundancies(uvd, red_gps)` takes a redundant-baseline-grouped UVData (only one representative per group) and expands it back to the full set of baselines, replicating each representative across its group via `np.tile` of indices into the data/flag/nsample/uvw arrays.

---

## 9. `beam_models.py` — Jones-matrix beams and basis transforms

435 lines. Contains both analytic beams (`uniform`, `airy_dipole`, `gaussian_dipole`, `heraish_beam_func`) and the spline-based `spin1_beam_model` adapter. All `nb.njit`-compatible.

### 9.1 Spline beam from spin1_beam_model data

`model_data_to_spline_beam_func(full_file_name, nu_axis, L_synth=180, interp_method='sinc_rbf', indexed=False, horizon_taper=True)`:

1. Loads CST-derived spin-1 spherical-harmonic model via `AntennaFarFieldResponse(full_file_name)`.
2. Calls `AR.derive_symmetric_rotated_feed(rotation_angle_sign="positive")` — derives the second feed by 90° rotation of the first.
3. `AR.compute_spatial_spline_approximations(nu_axis_MHz, L_synth=L_synth, interp_method=interp_method)` — computes 2-D `RectBivariateSpline` approximations on the SSHT grid.
4. Extracts spline knots `(tx, ty)`, orders `(kx, ky)`, and per-frequency 5-D coefficient tensors `E_coeffs, rE_coeffs` of shape `(Nfreq, 2, 2, 2, N_coeff)`.
5. Returns `construct_spline_beam_func(...)` — a closure-built njit function `spline_beam_func(nu, alt, az)` (or `spline_beam_funcs(i, nu, alt, az)` if `indexed=True`).

`construct_spline_beam_func`’s closure:

- Resolves `i_nu = where(nu == nu_axis)[0][0]` — i.e. the input `nu` must be exactly one of the precomputed frequencies (otherwise undefined behaviour, as commented in source).
- Maps to spline coordinates `μ = cos θ = sin alt`, `phi = az_shiftflip(az)` (handedness fix; see §9.3).
- For each `(k, a) ∈ {0,1}²`, evaluates the appropriate `bispeu_nb(...)` and accumulates with `u = [-1.0, -1j]` weights into `J_aa[..., imap[*], a]` — these signs come from the convention that the alt/az basis components carry an opposite sign vs the model’s native basis.
- Applies a basis transform from the intermediate RA/Dec basis to AltAz via `basis_transform_components(alt, az, R)` where `R` is the rotation about +y by `90 + |HERA_LAT|` degrees.
- If `horizon_taper=True`, multiplies by a Tukey window over the upper hemisphere starting at 2° above the horizon (`tukey_window(alt, np.radians(2.0))`).

### 9.2 Analytic beams

- `uniform(i, nu, alt, az)` — identity 2×2 (signature kept compatible with the polymorphic dispatch convention).
- `make_airy_dipole(a)` — closure that returns an njit function `airy_dipole(i, nu, alt, az)` for an airy-disk-multiplied dipole element, where `a` is the aperture radius (m). Uses `2 J_1(ka cos alt)/(ka cos alt)` for the airy factor (with a `1e-20` safety bump to avoid division by zero). Off-diagonal Jones structure: `J[0,0] = -sin(az) sin(alt) G`, `J[0,1] = cos(az) G`, `J[1,0] = -cos(az) sin(alt) G`, `J[1,1] = -sin(az) G`. The dipole element is exactly the Hertzian dipole projected from the perpendicular polarisation onto the alt/az basis.
- `make_gaussian_dipole(a)` — same dipole projection but with a Gaussian factor `G = exp(-(π/2 - alt)² / 2 a²)` instead of the airy.
- `heraish_beam_func(i, nu, alt, az)` — concrete `make_airy_dipole(7.0)` plus the AltAz-to-RA/Dec basis transform plus the 2° Tukey horizon taper. The hard-coded aperture `a=7.0 m` is a HERA-like dish.
- `njit_J1(x)` — a CFFI route to `scipy.special.cython_special.j1` (Bessel function of the first kind, order 1) that is njit-callable. Same trick as `dfitpack_numba.py` but for a scalar special function rather than a Fortran routine.

### 9.3 Coordinate primitives (all `@nb.njit`)

- `altaz2cartENU(alt, az)` and inverse `cartENU2altaz`. ENU convention: `E = sin az · cos alt`, `N = cos az · cos alt`, `U = sin alt`.
- `alt_hat(alt, az)`, `az_hat(alt, az)` — unit vectors of the Alt/Az basis in ENU coordinates.
- `thetaphi2cartXYZ(theta, phi)`, `cartXYZ2thetaphi(X, Y, Z)`, `theta_hat(theta, phi)`, `phi_hat(theta, phi)` — analogues for spherical coordinates with theta measured from the equator (i.e. `theta = elevation`, not co-latitude).
- `rotation_matrix(axis, angle)` — Rodrigues’ formula producing a 3×3.
- `basis_transform_components(alt, az, R)` — projects the AltAz unit basis onto a rotated theta/phi basis (R about the y-axis), returning `cos χ, sin χ` where χ is the local rotation angle. Used both inside the spline beam closure and in `heraish_beam_func`.
- `apply_basis_transform(Ma, Mb)` — pointwise `J → J · U^T` over a stack of 2×2 matrices.
- `az_shiftflip(az)` — its own inverse: maps "azimuth measured east-of-north" to/from the right-handed angle measured north-of-east. Implementation: `(π/2 - az)` modulo `2π`.
- `tukey_window(alt, start_rad)` — flat-topped Tukey taper over the upper hemisphere; identically 1 above `start_rad`, smoothed `sin²(α · alt)` between 0 and `start_rad` (where `α = π/(2 start_rad)`), and zero below 0.

---

## 10. `sky_models.py` — point-source & diffuse sky harmonics

541 lines. Provides synthetic catalogue generation, two independent point→harmonic transforms, GSM diffuse-sky harmonics, and helper utilities for indexing differences between healpy and SSHT.

### 10.1 Catalog generation

- `random_power_law(S_min, S_max, alpha, size=1)` — inverse-CDF sampler for `pdf(S) ∝ S^{α-1}` on `[S_min, S_max]`. (Note: parameter `alpha` follows the *cumulative-power-law* sign, not the differential `−γ`.)
- `generate_point_source_flux(Nsrc, F_min, F_max, gamma)` — wraps the above with `α = 1 - γ` so that `dN/dS ∝ S^{-γ}`.
- `generate_point_source_catalog(Nsrc, seed, F_min=0.5, F_max=100.0, gamma=1.8)` — produces a dict `{ "RA", "dec", "Flux_150", "spectral_indices" }` with a GLEAM-ish slope and isotropic sky distribution. Spectral indices drawn from `−0.8 + 0.2·N(0,1)` (Hurley-Walker 2016 Fig. 16-ish). Uniform-in-cos(co-dec) draw guarantees uniform on the sphere.
- `sky_from_catalog(catalog, nu_axis)` → `(S, RA, dec)` where `S` has shape `(N_freq, N_src, 4)` with only Stokes-I populated. Uses `(ν / 150 MHz)^α` scaling.

### 10.2 Point sources → spherical harmonics

Three independent implementations (each progressively smarter):

#### (A) Direct sum via `ssht_numba.dl_m`

- `inner_point_source_harmonics(flux_density, ra, dec, L, ell_min, delta)` (`@nb.njit`) — sums the per-source contribution `Σ_i I_i · Y_lm^*(co-dec_i, ra_i)` per `(ell, m)` index, looping `ell ∈ [ell_min, L)`. Inside, `spin0_spherical_harmonics(ell, theta, phi, delta)` returns the `(2ell+1)` spin-0 harmonics at one direction, computed from the `delta = sshtn.generate_dl(π/2, L)` Wigner d-matrix. Output shape `(N_freq, L² - ell_min²)`.
- `old_point_sources_harmonics(flux_density, RA, dec, L, ell_min=0)` — the public wrapper. Note: in this snapshot the function uses `ra, dec` internally without ever taking `RA, dec` as defined in the signature — i.e. this function is broken at present (uses an undefined name `ra`). Documented but not referenced anywhere in tests.
- `threaded_point_sources_harmonics(flux_density, ra, dec, L, ell_min=0, n_blocks=2)` — parallel block decomposition over sources. Comment in source: empirically no speedup beyond `n_blocks=3` (likely cache/thread-contention in `sshtn.dl_m`).

#### (B) Pure recurrence-based `spherical_harmonic_sequence`

`spherical_harmonic_sequence(L, theta, phi)` (`@nb.njit(fastmath=True)`) implements the associated-Legendre recurrence of Reinecke 2011 §3.1.1 with the McEwen–Wiaux 2011 phase convention (matching SSHT). Computes the full `Y(L²)` array for a single direction. Used by `point_sources_harmonics(I, RA, dec, L)` (`@nb.njit(parallel=True)` with `nb.prange` over sources). Faster than (A) because the Legendre recurrence is in a single pass instead of `O(L)` calls into `sshtn.dl_m`.

#### (C) Grid-then-transform `point_sources_harmonics_with_gridding`

The most efficient and the one used by `PointSourceSpectraSet.generate_harmonics`. Workflow:

1. Sample the MW grid at bandlimit `L` via `sshtn.mwss_sample_grid(L)`, yielding `(theta, phi)` arrays.
2. Build `s_hat` direction unit vectors at each grid point.
3. For every grid point `g` and every source `i`, compute the dot-product `c_i = ŝ_g · ŝ_i`, then evaluate the **closed-form Legendre-polynomial sum kernel** `K_i = Σ_{ℓ=0}^{L-1} (2ℓ+1) P_ℓ(c)`. The recursion identity used:
   ```
   Σ_{ℓ=0}^{L-1} (2ℓ+1) P_ℓ(x) = (L · (P_L(x) - P_{L-1}(x)))/(x - 1)        # legendre_polynomial_sum
   ```
   with a Taylor expansion in `(x - 1)` near `x = 1` to avoid the removable singularity (`legendre_polynomial_sum_near_1`). A 1e-14 threshold switches between the two branches.
4. Accumulate `G[g, freq] += I[freq, i] · K_i` into the 2-D grid (`@nb.guvectorize(target='parallel')`).
5. Run `parallelized_harmonic_transform(G, L, Ilm)` — `@nb.njit(parallel=True)` with `nb.prange` over frequencies, calling `sshtn.mw_forward_sov_conv_sym_ss_real(G[k], L, Ilm[k])`.
6. Divide by `4π` to normalise.

The Legendre kernel `Σ_ℓ (2ℓ+1) P_ℓ(ŝ·ŝ_i)` equals `4π Σ_{ℓm} Y_lm(ŝ) Y_lm^*(ŝ_i)` (the sum-over-ℓm completeness relation), so multiplying by the source flux and gridding builds `Σ_i I_i · 4π Y_lm^*(ŝ_i)` *before* the SHT is taken, then taking the SHT recovers the harmonic sum. Big speedup vs the direct sum because the `(2ℓ+1)P_ℓ` product evaluation has a closed form via Legendre recursion identity, sidestepping the `O(N_src · L²)` inner loop.

The `njit_P` wrapper exposes `scipy.special.cython_special.eval_legendre` to numba via CFFI (same trick as the airy `J_1`).

`@nb.vectorize`-decorated routines:
- `njit_P(n, x)` — `P_n(x)`.
- `legendre_polynomial_sum(N, x) = (N+1)/(x-1) · (P_{N+1}(x) - P_N(x))`.
- `legendre_polynomial_sum_near_1(N, x)` — second-order Taylor expansion.

### 10.3 Diffuse-sky harmonics

- `hp2ssht_index(hp_flm_in, lmax=None)` — converts healpy-indexed `alm` into SSHT-indexed `flm`. The transform is more than a re-index: healpy and SSHT have opposite-handedness azimuth, so the routine first applies `R_xflip = diag(-1, 1, 1)` rotation via `hp.rotate_alm`, then maps each `(ℓ, m)`:
  - `m ≥ 0`: `flm[el, m] = e^{i m π} · hp_flm[ind(ℓ, |m|)]`,
  - `m < 0`: `flm[el, m] = (-1)^m · conj(e^{i m π} · hp_flm[ind(ℓ, |m|)])`.
- `diffuse_sky_model_from_GSM2008(nu_axis, smooth_deg=0.0, ssht_index=True)` — uses `pygsm.GlobalSkyModel(freq_unit="MHz", basemap="haslam", interpolation="cubic")` to generate a `(N_freq, N_pix)` healpix map at `nside=512` (implicit; lmax = 3·512//2). Converts T (K) → I (Jy) via `Jy_per_K = 1e26 · 2 k_B · (ν/c)²` using `k_B = 1.38064852e-23 J/K`, `c = 299792458 m/s`. Then `hp.map2alm(..., pol=False, use_pixel_weights=True)`, rotates from Galactic to GCRS via `R_g2c = utils.get_galactic_to_gcrs_rotation_matrix()` (in-place via `hp.rotate_alm`), and finally re-indexes to SSHT via `hp2ssht_index` if requested.
- `diffuse_sky_model(nu_axis, R_g2c=None, ssht_index=True, smth_deg=0.0)` — older variant using `pygsm.GlobalSkyModel2016(freq_unit="MHz", unit="MJysr", resolution="low")` (so flux is already in `MJy/sr` and only needs scaling by `1e6`); uses `nside=64`, `lmax = 3·nside - 1`. Optional Gaussian smoothing via `hp.smoothalm(fwhm=...)`.
- `diffuse_sky_model_egsm_preview(nu_axis)` — loads precomputed `Ilm(freq)` from a hard-coded local path (`/users/zmartino/zmartino/eGSM_preview/egsm_harmonics.h5`) and 5-th-order spline-interpolates each component. Useful only on a UPenn machine.

### 10.4 Healpix rotation utilities

- `rotate_sphr_coords(R, theta, phi)` — applies a 3-D rotation to spherical coordinates by going through the cartesian unit vector representation.
- `linear_interp_rotation(hmap, R)` — linear-interpolation-based scalar rotation of a healpix map: fetches the new pixel angles via `rotate_sphr_coords` and reads back via `hp.get_interp_val`.

---

## 11. `dfitpack_numba.py` & `dfitpack_wrappers/` — CFFI bridge to FITPACK

Files:

- `dfitpack_numba.py` (80 lines): builds a `cffi.FFI` instance, dlopens `dfitpack_wrappers.so` from the same directory, declares the `bispeu_wrap` C signature, and defines the njit-callable wrapper `bispeu_nb(tx, ty, c, kx, ky, x, y) -> (z, ier)`.
- `dfitpack_wrappers/dfitpack_wrappers.f90`: ISO-C-binding wrapper around `bispeu`.
- `dfitpack_wrappers/{bispeu,fpbisp,fpbspl}.f`: FITPACK F77 source. These are taken verbatim from FITPACK / scipy and are not edited.
- `Makefile`: `gfortran -O3 -fPIC -g -shared` build, output `dfitpack_wrappers.so`.

`bispeu_nb` allocates the work array `wrk` of length `kx + ky + 2`, calls `bispeu_wrap` through CFFI’s `from_buffer`, and returns the evaluated values `z`. The `(z, ier)` return signature mirrors FITPACK conventions: `ier == 0` indicates success.

This is the *only* path by which spline evaluation happens inside Numba’s `nopython` mode in this codebase — all spline beam evaluations route through it. The performance-critical alternative would be `scipy.interpolate.bisplev`, which is not njit-friendly.

---

## 12. Test suite

`pytest`-based, configured in `setup.cfg [tool:pytest]` with:

```
addopts = --cov RIMEz --cov-report term-missing --verbose
testpaths = tests
```

Files (with state observed in this snapshot):

### 12.1 `tests/conftest.py`

Two session-scope fixtures:

- `visibility_calculation_fixed_test_parameters` — promotes `data.generate_test_data.visibility_calculation_fixed_test_parameters` to a fixture, returning the canonical 1-source HERA-ish parameter tuple.
- `visibility_calculation_fixed_test_output` — reads `tests/data/visibility_calculation_test_output.h5` and returns the cached `(V_1src, Vm_1src, Vhrm_1src)` arrays.

### 12.2 `tests/data/generate_test_data.py`

Defines the canonical regression: 11 frequency channels (100–200 MHz), 11 time samples spanning one full Earth rotation (chosen so the LST sweep is a uniform `2π` cycle around the reference epoch), 3 antennas at `(0,0,0), (20,0,0), (0,20,0)` (m), 4 baselines including the auto on antenna 0, a single source at `(RA, Dec) = (LON+ERA0, LAT)` so that it transits zenith exactly at `jd0`, a flat-spectrum Stokes-I unit source (`S[:,:,0] = 1`), and a `J = sin(alt)³ · I_2` analytic beam. Computes:

- `V_1src` via `parallel_point_source_visibilities`,
- `Vm_1src` via `parallel_mmode_unpol_visibilities` at bandlimit `L = 2π·(2/3)·sqrt(2)·20 + 100`,
- `Vhrm_1src` via `parallel_visibility_dft_from_mmodes`,

and writes them all to `visibility_calculation_test_output.h5` with a `version` and `date_created` stamp. The script supports `--overwrite` and `--no-archive` flags; old test data is moved to `tests/data/old_test_data/<date>_<version>.h5` by default.

### 12.3 `tests/test_rime_funcs.py`

- `test_make_sigma_tensor()` — shape `(2,2,4)`, dtype `complex128`.
- `test_make_bool_sigma_tensor()` — shape `(2,2,4)`, dtype `bool`.
- `test_fast_approx_radec2altaz()` — at the equator and `R = I_3`, AZ should equal RA (modulo wrap).
- `test_RIME_sum()` — for two identical sources with a given Jones, `Σ_n F1 J σ_0 J^† F2 = N · J · J^†` (with `S = (1, 0, 0, 0)` per source). Test sets `N=2`, two scaled Jones matrices, and verifies the closed form `5 · J_n · J_n^†`.
- `test_visibility_calculations(...)` — full regression: runs all three RIME entry points and asserts `np.allclose` against the cached HDF5 output to `atol=5e-14`.

### 12.4 `tests/test_management.py`

- `test_get_versions()` — checks `_get_versions()` returns the expected `__version__` strings. (Note: the version of this file in this snapshot still has unresolved git-merge conflict markers `<<<<<<< HEAD` / `=======` / `>>>>>>> ...` near the bottom; this is a known broken state; see §16.)
- `TestVisibilityCalculation` — autouse fixture sets up a `VisibilityCalculation` instance using the harmonics path. Two tests: `test_compute_fourier_modes` (asserts `VC.Vm == 0.5 · Vm_recorded`, since the class multiplies by 0.5 internally) and `test_compute_time_series` (asserts the synthesised time series matches the recorded `Vhrm_1src_rec`).

### 12.5 `tests/test_sky_models.py`

- `test_random_power_law` — RNG-seed regression: the first sample for `(S_min=10, S_max=100, alpha=-2.7, seed=1)` must equal `12.20579531` to 8 d.p.
- `test_point_source_harmonics` — generates 10 random sources with random spectra and asserts `point_sources_harmonics_with_gridding` and `point_sources_harmonics` (the recurrence-based variant) agree to `atol=5e-9`.

### 12.6 `tests/test_beam_models.py`

Two micro-tests for `theta_hat` and `phi_hat`: at sample directions, the cartesian components match the closed-form spherical-basis vectors.

### 12.7 `tests/test_utils.py`

Tests for `coords_to_location` (returns an `astropy EarthLocation` matching the inputs), `kernel_cutoff_estimate` (matches the analytical `2π ν b/c + width` rounded to next even integer + 1), `b_arc` and `B` (zero-baseline NaN, axis-only π/2, generic case), `generate_hex_positions` against a hard-coded answer for `lattice_scale=10, u_lim=1, v_lim=2, w_lim=2`, `get_minimal_antenna_set` consistency between forward and inverse dictionaries, and round-trip `JD2era`/`JD2era_tot` sanity. (Note: this file also has a stray `>>>>>>> 3c2d717 (...)` merge marker at the very bottom; another broken state.)

---

## 13. Notebooks and docs

Only one notebook in this snapshot: `notebooks/monopole_vs_no_monopole.ipynb` (visualises the contribution of the spherical-harmonic monopole to a HERA-band visibility — a useful check for the `Slm[0,0]`-handling correctness in the harmonic method).

`docs/` is a Sphinx scaffold (PyScaffold-generated). The `index.rst` file is essentially the default placeholder; only the leaf RST stubs `license.rst`, `authors.rst`, `changelog.rst` are wired into the toctree. There is no API reference doc beyond what `:autodoc:` would generate. CI does *not* build the docs.

---

## 14. End-to-end usage flow

Below is the canonical pipeline (mirroring `tests/data/generate_test_data.py`).

```python
import numba as nb
import numpy as np
from RIMEz import management, sky_models, utils

# -- Array geometry & timing -----------------------------------------------
array_latitude  = utils.HERA_LAT
array_longitude = utils.HERA_LON
array_height    = utils.HERA_HEIGHT

nu_hz = np.linspace(100e6, 200e6, 11)                           # Hz, equally spaced
r_axis = np.array([[0, 0, 0], [20., 0, 0], [0, 20., 0]],        # ENU, meters
                  dtype=np.float64)
ant_pairs = np.array([[0, 0], [0, 1], [0, 2], [1, 2]], dtype=np.int64)

# -- Beam ------------------------------------------------------------------
@nb.njit
def beam_func(i, nu, alt, az):
    J = np.zeros((alt.shape[0], 2, 2))
    J[:, 0, 0] = np.sin(alt) ** 3
    J[:, 1, 1] = np.sin(alt) ** 3
    return J

ant_ind2beam_func = np.zeros(r_axis.shape[0], dtype=np.int64)

# -- Time axis (one earth rotation, evenly sampled) ------------------------
jd0 = 2458845.5
N_times = 11
delta_era = 2 * np.pi / (N_times - 1)
delta_jd  = utils.era_tot2JD(delta_era) - utils.era_tot2JD(0.)
jd_axis = jd0 + delta_jd * np.arange(-(N_times - 1) // 2, (N_times - 1) // 2 + 1)
integration_time = 0.0    # analytic instantaneous samples

# -- Sky -------------------------------------------------------------------
RA  = np.array([array_longitude + utils.JD2era(jd0)])
dec = np.array([array_latitude])
S = np.zeros((nu_hz.size, RA.size, 4)); S[:, :, 0] = 1.0       # 1 Jy Stokes-I

# Compute spherical-harmonic representation
L = utils.kernel_cutoff_estimate(20.0, 200e6, width_estimate=100)
Ilm = sky_models.point_sources_harmonics_with_gridding(S[..., 0], RA, dec, L)
Slm = Ilm.reshape(Ilm.shape + (1,))    # (Nfreq, L^2, 1) — only Stokes-I supported

# -- Run the harmonic visibility calculation ------------------------------
parameters = {
    "array_latitude": array_latitude,
    "array_longitude": array_longitude,
    "array_height": array_height,
    "initial_time_sample_jd": jd0,
    "integration_time": integration_time,
    "frequency_samples_hz": nu_hz,
    "antenna_positions_meters": r_axis,
    "antenna_pairs_used": ant_pairs,
    "antenna_beam_function_map": ant_ind2beam_func,
    "integral_kernel_cutoff": L,
}

VC = management.VisibilityCalculation(parameters, beam_func=beam_func, Slm=Slm)
VC.compute_fourier_modes()
VC.compute_time_series(time_sample_jds=jd_axis, integration_time=integration_time)

# -- Persist -------------------------------------------------------------
VC.write_visibility_time_series("/tmp/V.h5", overwrite=True)
uvd = VC.to_uvdata(telescope_name="HERA-mock")
uvd.write_uvh5("/tmp/V.uvh5", clobber=True)
```

For a point-sample (DFT) calculation, swap:

```python
VC = management.VisibilityCalculation(
    parameters, beam_func=beam_func,
    S=S, RA_icrs=RA, Dec_icrs=dec
)
VC.compute_time_series()    # populates VC.V directly
```

For ICRS catalogue input, `setup()` will internally call `utils.transform_icrs_to_cirs(RA_icrs, Dec_icrs, jd0)` to bring the coordinates into the CIRS frame at the reference time before computing rotations.

---

## 15. Conventions — coordinates, polarisation, sign, basis

A condensed cheat-sheet of every convention that matters when reading/extending RIMEz.

### 15.1 Frames

| Frame | Where used | Notes |
|---|---|---|
| **ICRS** | User input for source positions | International Celestial Reference System; RA/Dec in radians. |
| **CIRS** | Internal (point-sample path) | Celestial Intermediate Reference System at `reference_jd`. Converted via `transform_icrs_to_cirs`. |
| **GCRS** | Internal (harmonic path; default) | Geocentric Celestial Reference System; sky-fixed during a calculation. |
| **AltAz** | Internal (everywhere `beam_func` is called) | Alt = elevation above the horizon; Az = azimuth east-of-north; per-time derived via astropy. |
| **ENU** | Antenna positions | East / North / Up. **Internal sign convention:** `s = (sin az · cos alt, cos az · cos alt, sin alt)`. Note the swapped-sin/cos vs textbook ENU; this is what `rime_funcs.vec_psv` and `vec_muv` actually compute. |
| **ECEF** | UVData export only | Earth-Centred Earth-Fixed; obtained via `pyuvdata.utils.ECEF_from_ENU`. |
| **Galactic** | Optional input for diffuse sky | Converted to GCRS via `get_galactic_to_gcrs_rotation_matrix()`. |

### 15.2 Time

- All times are Julian Date (`jd`, scale `ut1`).
- The Earth Rotation Angle (ERA) used for harmonic-method time synthesis follows USNO Circular 179 Eqn 2.10 (`JD2era_tot`). For "wrapped" ERA, ERFA's `era00` is used.
- Reference rotation `R_0` is computed at `initial_time_sample_jd` and orthogonalised via `scipy.linalg.orthogonal_procrustes`.

### 15.3 Polarisation

- The sky tensor is `(N_freq, N_src, 4)` in Stokes order `[I, Q, U, V]`.
- Coherency matrix uses the Pauli decomposition `C = ½ Σ_g σ_g · S_g`. The factor of ½ is **not** applied inside `RIME_sum`; instead it is applied externally — `VisibilityCalculation.compute_fourier_modes` does `self.Vm = 0.5 * Vm` after the kernel returns.
- Output instrumental polarisation order in UVData is `["xx", "yy", "xy", "yx"]` (standard `pyuvdata` ordering).

### 15.4 Geometric phase sign

`rime_funcs` uses `phases = -2π ν b·ŝ / c` and `F = e^{i phases}`. Combined with the explicit `J_q^*` in the inner sum, this gives the standard `V = ∫ J_p · C · J_q^† · e^{-2π i ν b·ŝ /c} dΩ` form.

### 15.5 SSHT vs healpy

Two SHT conventions co-exist in the codebase:

- **SSHT (McEwen–Wiaux 2011)** — used everywhere in `rime_funcs` and in `sky_models.point_sources_harmonics*` for kernel/sky harmonics. The `mw` (and `mwss`) sample grid; `dl_m`, `mw_forward_sov_conv_sym_ss_real`, `elm2ind`, etc. live in `ssht_numba`.
- **healpy** — used only by the diffuse-sky import path (`pygsm` returns healpix maps) and by some of the consistency utilities (`beam_func_to_Omegas_healpix_sum`, `linear_interp_rotation`).

The two conventions disagree on **azimuth handedness**: healpy uses left-handed φ, SSHT right-handed. `hp2ssht_index` performs the explicit re-indexing + handedness flip via `R_xflip = diag(-1, 1, 1)`.

### 15.6 Beam evaluation contract

A `beam_func` (whether njit-compiled or analytic) takes `(i, nu, alt, az)` and returns `J` of shape `alt.shape + (2, 2)` complex. The integer `i` is a *beam-function index* — the kernel passes the index for the antenna currently under evaluation, allowing heterogeneous arrays (different antennas can use different beam models). The `ant_ind2beam_func` array is the per-antenna lookup.

The instrumental basis is `(alt, az)` *for the analytic beams* but `(RA, Dec)` for the `spline_beam_func` after its internal basis transform — the docstring of `construct_spline_beam_func` mentions this is "currently hard-coded to return E-field vector components in a basis aligned with intermediate RA/Dec".

---

## 16. Known issues / quirks observed in this snapshot

These are not bugs introduced by this monorepo but **state of the upstream snapshot** that any consumer should be aware of:

1. **Unresolved git-merge conflict markers in tests.** Both `tests/test_management.py` (around lines 91–96) and `tests/test_utils.py` (last line) contain `<<<<<<< HEAD` / `=======` / `>>>>>>> ...` markers. As-is, these files will fail to import.
2. **`utils.transform_icrs_to_cirs` is broken.** The body references `RA_icrs_deg` and `Dec_icrs_deg` (undefined), and treats the input as degrees while the docstring says radians. The `point_samples` calculation path therefore does not work as documented.
3. **`np.float` deprecation.** `utils.get_rotations_realistic_from_JDs` uses `dtype=np.float`, which is removed in NumPy ≥ 1.20.
4. **`sky_models.old_point_sources_harmonics` is broken.** Refers to `ra` instead of `RA` from the signature. Use `point_sources_harmonics_with_gridding` or `point_sources_harmonics`.
5. **`np.string_` deprecation.** `management.py` uses `np.string_(...)` for HDF5 string writes; modern code should use `np.bytes_`.
6. **`pyuvdata` API drift.** `utils.uvdata_from_sim_data` uses `uvd.telescope_location_lat_lon_alt`, `uvd.x_orientation`, `uvd.phase_type = "drift"`, `uvd.spw_array`, etc., all of which have moved or been renamed in modern `pyuvdata`. The function targets a circa-2019 API.
7. **`from astropy import _erfa` and `import erfa as _erfa`** both occur in `utils.py`. Modern astropy split out `erfa` into the standalone `pyerfa` package; only one of these imports works today. The CI snapshot pins old versions, so this was fine then but breaks on a fresh install.
8. **`pygsm` git scheme.** The optional dependency uses `git+git://...` which GitHub deprecated. Anyone installing the `gsm` extra needs to override to `git+https://...`.
9. **`setup.py` is truncated.** The local copy ends at line 32 mid-`os.path.join`. The custom build will not function as-is. (Compilation has presumably been done manually by `make` in the wrappers directory.)
10. **`numba<0.49.0` pin in CI.** `@nb.guvectorize` and the `@nb.vectorize` CFFI bindings used here are sensitive to numba’s typed-list handling that changed in 0.49.
11. **`dtype=np.bool`.** Used in `utils.uvdata_from_sim_data` and `tests/test_rime_funcs.py`; deprecated in modern NumPy.
12. **No `__all__` exports**, no top-level facade. Users must know which submodule to import.
13. **`zip(..., strict=False)`** in `utils.py` (`get_minimal_antenna_set`, `inflate_uvdata_redundancies`) is Python 3.10+ syntax — inconsistent with the package’s declared `python_requires >= 3.6`.
14. **Hard-coded paths.** `diffuse_sky_model_egsm_preview` looks for `/users/zmartino/zmartino/eGSM_preview/egsm_harmonics.h5`, only present on the original author's machine.
15. **Sphinx docs are mostly empty.** `docs/index.rst` is the default placeholder; there is no narrative API documentation. Anyone reading the source is the audience.
16. **Polarised m-mode method is not implemented.** `compute_fourier_modes` raises `NotImplementedError` for the polarised harmonic path; the m-mode formalism in `vec_muv` hard-codes `g=0` (Stokes I). Polarisation requires the point-samples path.

---

### Cross-reference — which file holds what

| Concept | File | Symbols |
|---|---|---|
| RIME equation kernels | `src/RIMEz/rime_funcs.py` | `RIME_sum`, `vec_psv_constructor`, `parallel_point_source_visibilities`, `vec_muv_constructor`, `parallel_mmode_unpol_visibilities`, `mmode_unpol_visibilities`, `visiblity_dft_from_mmodes`, `parallel_visibility_dft_from_mmodes`, `inner_parallel_visiblity_dft_from_mmodes`, `visibility_from_mmodes` |
| Pauli/sigma | `rime_funcs.py` | `make_sigma_tensor`, `make_bool_sigma_tensor` |
| Coordinate kernel (njit) | `rime_funcs.py` | `fast_approx_radec2altaz` |
| Real/imag re-vec for splines | `rime_funcs.py` | `vectorize_vis_mat`, `devectorize_vis_vec` |
| Top-level orchestration | `src/RIMEz/management.py` | `VisibilityCalculation`, `PointSourceSpectraSet`, `_get_versions` |
| Time/coordinate utilities | `src/RIMEz/utils.py` | `JD2era`, `JD2era_tot`, `era2JD`, `era_tot2JD`, `get_rotations_realistic_from_JDs`, `get_rotations_idealized`, `get_icrs_to_gcrs_rotation_matrix`, `get_galactic_to_gcrs_rotation_matrix`, `transform_icrs_to_cirs`, `coords_to_location`, `kernel_cutoff_estimate` |
| Antenna geometry | `utils.py` | `HERA_LAT`, `HERA_LON`, `HERA_HEIGHT`, `b_arc`, `B`, `get_minimal_antenna_set`, `generate_hex_positions` |
| Beam diagnostics | `utils.py` | `beam_func_to_Omegas_ssht`, `beam_func_to_Omegas_healpix_sum`, `beam_func_to_kernel_power_spectrum` |
| UVData export | `utils.py` | `uvdata_from_sim_data`, `inflate_uvdata_redundancies` |
| Analytic beams | `src/RIMEz/beam_models.py` | `uniform`, `make_airy_dipole`, `make_gaussian_dipole`, `heraish_beam_func`, `njit_J1` |
| Spline beam | `beam_models.py` | `model_data_to_spline_beam_func`, `construct_spline_beam_func` |
| Beam coordinate primitives | `beam_models.py` | `altaz2cartENU`, `cartENU2altaz`, `alt_hat`, `az_hat`, `thetaphi2cartXYZ`, `cartXYZ2thetaphi`, `theta_hat`, `phi_hat`, `rotation_matrix`, `basis_transform_components`, `apply_basis_transform`, `az_shiftflip`, `tukey_window` |
| Catalog generation | `src/RIMEz/sky_models.py` | `random_power_law`, `generate_point_source_flux`, `generate_point_source_catalog`, `sky_from_catalog` |
| Point-source harmonics | `sky_models.py` | `inner_point_source_harmonics`, `old_point_sources_harmonics`, `threaded_point_sources_harmonics`, `point_sources_harmonics`, `point_sources_harmonics_with_gridding`, `spherical_harmonic_sequence`, `spin0_spherical_harmonics`, `grid_sources`, `parallelized_harmonic_transform`, `legendre_polynomial_sum`, `legendre_polynomial_sum_near_1`, `njit_P`, `elm2ind`, `A` |
| Diffuse sky | `sky_models.py` | `hp2ssht_index`, `diffuse_sky_model_from_GSM2008`, `diffuse_sky_model`, `diffuse_sky_model_egsm_preview`, `rotate_sphr_coords`, `linear_interp_rotation` |
| FITPACK CFFI bridge | `src/RIMEz/dfitpack_numba.py` + `dfitpack_wrappers/` | `bispeu_nb`, `dfitpack_wrappers.so` (C-bound `bispeu_wrap`) |

---

*This document was generated by reading the entire source tree at `simulators/RIMEz/` (3 228 lines across 14 files) and consolidating the implementation, conventions, dependencies, and known quirks into a single reference. It reflects the snapshot present in this monorepo, which corresponds to a state of upstream RIMEz between v0.1.1 and an unreleased v0.1.2 with partially-merged tooling updates.*
