# matvis — Exhaustive Reference

A deep, top-to-bottom reference for the `matvis` simulator vendored at
`simulators/matvis/`. Written from a direct read of every source file in
`src/matvis/`, the docs (`docs/`), the test suite (`tests/`), `setup.cfg`,
`CHANGELOG.rst`, and the CUDA sources under `src/matvis/gpu_src/`.

`matvis` describes itself as a "fast matrix-based interferometric visibility
simulator with both CPU and GPU implementations." The package was extracted
from `hera_sim`, lives under the `hera-team/matvis` GitHub org, is MIT
licensed, requires Python ≥ 3.11, and the vendored copy here is at version
`1.0.1` (per `CHANGELOG.rst`; with a `Dev` section already accumulating
fixes).

---

## 1. What matvis is, in one paragraph

For a sky modelled as a discrete set of point sources / pixels with known
intensities `I_n` at fixed equatorial unit-vectors, and per-antenna primary
beams given as either `pyuvdata.UVBeam` or `pyuvdata.AnalyticBeam`
(wrapped by `pyuvdata.beam_interface.BeamInterface`), `matvis` evaluates
the full Radio Interferometer Measurement Equation (RIME) sum

```
V^{pq}_{ij}(t,ν) = Σ_n A^p_i(X_n(t),ν) · C_n(ν) · A^q_j*(X_n(t),ν)
                                      · exp(-2πi ν b_{ij}·X_n(t)/c)
```

over all sources above the local horizon, at each time `t`, for each
antenna pair `(i, j)` and each feed pair `(p, q)`. It does this without any
flat-sky / w-projection approximation: the only structural approximation
is the discretisation of the sky into pixels/sources (the *user* picks
the discretisation; HEALPix is one valid choice). The implementation is
deliberately *matrix-based* — rather than looping over baselines, it
constructs an antenna-major "Z" matrix and obtains all visibilities with
a single Hermitian matrix product `V = Z·Zᴴ`. This allows BLAS / cuBLAS
to dominate the compute-time for large arrays and makes the same
algorithm trivially portable to GPU (cuBLAS `cgemm`/`zgemm`).

---

## 2. Repository layout

```
simulators/matvis/
├── AUTHORS.rst           Aaron Parsons, Hugh Garsden, Phil Bull,
│                         Steven Murray (lead, ASU), Piyanat Kittiwisit
├── CHANGELOG.rst         Versions 0.1.0 → 1.0.1, plus an in-flight Dev section
├── LICENSE.txt           MIT
├── README.rst            User-facing summary, install instructions
├── setup.cfg             Authoritative metadata + extras (gpu, profile, dev,
│                         docs, test, all)
├── pyproject.toml        Build backend (setuptools+setuptools_scm)
├── ci/                   conda env files (notebook-env, test-env)
├── codecov.yml           Coverage config
├── .coveragerc           Excludes cli.py, ImportError-guarded blocks, etc.
├── .github/workflows/    check-build, publish-to-pypi, run_notebooks,
│                         test_suite
├── docs/                 Sphinx (Furo theme) — index, understanding_the_
│                         algorithm, tutorials/matvis_tutorial.ipynb, api,
│                         authors, changelog, license
├── gpu-analysis.nvprof   A pre-recorded NVIDIA profile of a GPU run
├── src/matvis/           ← all source code (described in §4)
├── tests/                See §10
└── uv.lock
```

The Python package itself is laid out as:

```
src/matvis/
├── __init__.py          Sets __version__, detects HAVE_GPU, exposes
│                        DATA_PATH, simulate_vis, the `cpu` and `gpu`
│                        sub-packages.
├── _utils.py            ceildiv, human_readable_size, memtrace, logdebug,
│                        get_dtypes, get_required_chunks, get_desired_chunks
├── _test_utils.py       get_standard_sim_params() — shared test-fixture
│                        helper (uses pyradiosky / pyuvsim).
├── coordinates.py       enu_to_az_za, eci_to_enu_matrix, enu_to_eci_matrix,
│                        altaz_to_enu, point_source_crd_eq,
│                        equatorial_to_eci_coords, calc_coherency_rotation,
│                        _calc_rotation_matrix, vecs2rot, axis_angle_rotation_
│                        matrix, spherical_basis_vector_rotation_matrix,
│                        _calc_average_rotation_matrix, theta_hat, phi_hat
├── wrapper.py           simulate_vis() — frequency-loop convenience layer
├── cli.py               `matvis profile` / `matvis hera-profile` Click CLI
│                        (line-profiler driven)
├── core/
│   ├── __init__.py      _validate_inputs()
│   ├── beams.py         prepare_beam_unpolarized, _wrangle_beams,
│   │                    BeamInterpolator (ABC)
│   ├── coords.py        CoordinateRotation (ABC, with subclass registry)
│   ├── getz.py          ZMatrixCalc
│   ├── matprod.py       MatProd (ABC)
│   └── tau.py           TauCalculator
├── cpu/
│   ├── __init__.py      re-exports simulate
│   ├── cpu.py           simulate() — CPU driver
│   ├── beams.py         UVBeamInterpolator
│   ├── coords.py        CoordinateRotationAstropy, CoordinateRotationERFA
│   └── matprod.py       CPUMatMul, CPUVectorDot
├── gpu/
│   ├── __init__.py      re-exports simulate (only if cupy importable)
│   ├── gpu.py           simulate() — GPU driver
│   ├── beams.py         GPUBeamInterpolator + gpu_beam_interpolation()
│   ├── coords.py        GPUCoordinateRotationERFA (RawKernel light deflection)
│   ├── matprod.py       GPUMatMul, GPUVectorDot
│   └── _cublas.py       Direct cuBLAS cgemm/zgemm wrappers (zdotz,
│                        complex_matmul)
├── gpu_src/             Vestigial CUDA Jinja templates kept for reference:
│   ├── beam_interpolation.cu  (texture-/grid-based bilinear interpolation;
│   │                          the live code is now cupyx.ndimage.map_coordinates)
│   └── measurement_equation.cu (one-thread-per-source MeasEq kernel)
└── data/
    └── NF_HERA_Dipole_small.fits   2.6 MB peak-normalised efield UVBeam
                                    used by tests and the CLI profiler.
```

The `gpu_src/` `.cu` files were the original PyCUDA implementation (Jinja
templates with `{{ DTYPE }}`, `{{ NANT }}`, `{{ NPIX }}` substitutions) but
the runtime path now uses `cupy.RawKernel` (only for ERFA light-deflection
in `gpu/coords.py`) plus `cupyx.scipy.ndimage.map_coordinates` for beam
interpolation and `cupy_backends.cuda.libs.cublas.{cgemm,zgemm}` for the
matrix product. The `.cu` files are useful reading because they describe
the kernel-level memory layout assumed throughout the codebase.

---

## 3. The math, with the algorithm steps

The full visibility integral is

```
V_ij = ∫_sky 𝒜_i 𝒞 𝒜_jᴴ exp(-2πi ν b_ij·n̂/c) d²Ω
```

where `𝒜_i` is the complex polarised beam (a 2×2 Jones matrix when
polarised, a scalar power beam otherwise), `𝒞` is the source coherency
matrix (Stokes I/Q/U/V → 2×2), and `b_ij` is the antenna baseline vector.

After choosing a discrete pixel basis with unit-vectors `X_n(t)` and
intensities `I_n`, and assuming a flat horizon, the equation becomes

```
V^{pq}_{ij}(t) = Σ_n A^p_i(X_n(t)) · A^q_j(X_n(t)) · I_n
                                  · exp(-2πi ν X_n·b_ij/c)
```

`matvis` implements this by *factoring* `b_ij = D_j - D_i` into per-antenna
contributions and grouping the per-antenna factors into a single matrix:

| Quantity | Symbol  | Shape |
|---|---|---|
| Antenna positions in ENU (m) | `D` | `(N_ant, 3)` |
| ECI unit vectors of sources | `X_eq` | `(3, N_src)` |
| Topocentric unit vectors | `X = R_t X_eq` | `(3, N_src)` |
| Beam at sources (post-interp) | `A_{(feed·beam),(ax·src)}` | `(N_feed·N_beam, N_ax·N'_src)` |
| Antenna phasor `exp(-2πi ν D·X/c)` | `τ` | `(N_ant, N_src)` |
| Square-root flux | `√I_n` | `(N_src,)` |
| Z-matrix (per-antenna voltage) | `Z = √I · A · exp(τ)` | `(N_feed·N_ant, N_ax·N_src)` |
| Visibilities | `V = Z · Zᴴ` | `(N_feed·N_ant, N_feed·N_ant)` |

The trick is step 7 (`V = Z Zᴴ`): forming all `N_ant²` baselines reduces
to one Hermitian matrix product. On GPU this is exactly the cuBLAS
`cgemm/zgemm` with op-A = conj-transpose, op-B = no-op (see §6 and
`gpu/_cublas.py`).

The algorithmic steps inside the *time loop* are (mirrored byte-for-byte
in `cpu/cpu.py::simulate` and `gpu/gpu.py::simulate`):

```
for t in range(N_times):
    coords.rotate(t)                  # build X = R_t X_eq for ALL sources
    for c in range(n_chunks):         # split source axis if memory tight
        crd_top, √I, n = coords.select_chunk(c, t)
                                      # cull below-horizon, optionally
                                      # rotate per-source coherency
                                      # to alt/az for polarised input
        A = bmfunc(crd_top[0], crd_top[1])   # interpolate beam at (l,m)
        exptau = taucalc(crd_top)            # exp((2π i ν / c) D·X)
        z = zcalc(√I, A, exptau, beam_idx)   # Z = √I · A · exptau
        matprod(z, c)                        # V_chunk = Z · Zᴴ
    matprod.sum_chunks(vis[t])               # accumulate chunked V
```

Outputs are then optionally reduced to a list of antenna pairs (`antpairs`)
or kept as the full `(N_ant·N_ant)` Cartesian product. The frequency loop
sits in `wrapper.simulate_vis()` *outside* the per-time loop — a single
`simulate` call is single-frequency.

### 3.1 Final shape contract (post v1.0.0)

The CHANGELOG explicitly flags this as a breaking change in v1.0.0:

* `cpu.simulate` / `gpu.simulate`:
  - `polarized=True` →  `(N_times, N_pairs, N_feed, N_feed)`
  - `polarized=False` → `(N_times, N_pairs)`  (the singleton feed axes
    are squeezed at `return vis[:, :, 0, 0]`)
* `wrapper.simulate_vis`: the same arrays, but with a leading frequency
  axis: `(N_freqs, N_times, N_pairs[, N_feed, N_feed])`.

`N_pairs = N_ant²` if `antpairs is None`; otherwise `len(antpairs)`. The
antenna pairs are *Cartesian* (i, j) with both `i==j` and (j, i) included
unless explicitly subsetted — this matches `pyuvsim`'s ordering, which
the matprod transpose was tuned to reproduce (see the comment in
`tests/test_matprod.py::test_matprod` about a transpose vs. pyuvsim).

---

## 4. Module-by-module reference

### 4.1 `matvis/__init__.py`

* Imports `cupy` lazily and sets `HAVE_GPU` accordingly.
* Re-exports `cpu`, `gpu` sub-packages; if `HAVE_GPU` is False the `gpu`
  sub-package's `simulate` will not exist (the wrapper checks for this
  and raises an `ImportError` at call time).
* Sets `DATA_PATH = Path(__file__).parent / "data"` — used by the CLI to
  locate `NF_HERA_Dipole_small.fits`.
* `__version__` is read from importlib metadata.

### 4.2 `_utils.py`

Internal helpers shared between CPU and GPU paths.

| Symbol | Purpose |
|---|---|
| `ArrayType` | `np.ndarray ∪ cp.ndarray` typing alias |
| `no_op(fn)` | Identity decorator (used as a fallback when profiling is off) |
| `ceildiv(a, b)` | `-(a // -b)` integer ceiling division |
| `human_readable_size(size, decimal_places, indicate_sign)` | "B / KiB / MiB / …" formatter; tested at `tests/test_utils.py` |
| `memtrace(highest_peak)` | Wraps `tracemalloc.get_traced_memory()`; logs current and peak GB and returns the new high-water mark |
| `logdebug(name, x)` | DEBUG-level prints of corner elements of any array (CPU or GPU) |
| `log_progress(start, prev, iters, niters, pr, last_mem)` | Periodic progress info logging including ETA and RSS delta |
| `get_required_chunks(freemem, nax, nfeed, nant, nsrc, nbeam, nbeampix, precision, source_buffer=0.55)` | Iteratively guesses the smallest `nchunks` that fits in `freemem` by *summing* the sizes of all the named buffers (antpos, flux, beam, crd_eq, eq2top, crd_top, crd_chunk, flux_chunk, exptau, beam_interp, zmat, vis). Caps at 100 chunks. |
| `get_desired_chunks(...)` | Wraps `get_required_chunks` with `min_chunks` floor and the `nsrc` ceiling; returns `(nchunks, nsrcs_per_chunk)`. |
| `get_dtypes(precision)` | `precision=1 → (float32, complex64)`; `precision=2 → (float64, complex128)`. |

The chunking machinery is the only mechanism by which `matvis` controls
peak RAM/VRAM: the source axis is the only one that gets split, because
all other axes (antennas, feeds, beams) appear in the final `Z·Zᴴ`
contraction and cannot be marginalised piecewise. Per-chunk visibilities
are accumulated by `MatProd.sum_chunks`.

### 4.3 `coordinates.py`

Stateless coordinate utilities. Important conventions:

* The `matvis` "ECI" frame is *aligned with the celestial pole* (so
  `(RA=0, Dec=0) → (1,0,0)`, `(RA=90°, Dec=0) → (0,1,0)`, `(0, 90°) →
  (0,0,1)`); this is *not* the Earth-pole-aligned ECI used elsewhere.
* ENU is east-north-up; the up axis is the topocentric `n̂` direction in
  the visibility integral.
* `enu_to_az_za` supports two conventions, governed by `orientation`:
  - `"astropy"`: Az is east-of-north (`Az(N)=0`, `Az(E)=π/2`).
  - `"uvbeam"`:  Az is north-of-east (`Az(E)=0`, `Az(N)=π/2`).

Both are exercised in `tests/test_coordinates.py::test_equatorial_to_enu`,
which also confirms `eci_to_enu_matrix` and `enu_to_eci_matrix` are
genuine inverses.

* `point_source_crd_eq(ra, dec) → (3, N_src)` builds the unit-vector matrix
  `[cos ra cos dec, sin ra cos dec, sin dec]` — used everywhere as the
  starting point for the topocentric rotation.
* `equatorial_to_eci_coords(ra, dec, obstime, location, …)` does the full
  ICRS → AltAz → ENU → "matvis-ECI" → (ra, dec) round-trip so that user
  RA/Dec values are pre-corrected for the time-dependent ICRS↔altaz
  bias/precession/nutation/aberration *at a single reference time*. This
  is the recommended way to feed RA/Dec into the simulator if you want
  exact agreement with `pyuvsim` — but note matvis/pyuvsim already
  agree to ≲ 0.01 % when the per-time CoordinateRotation is in effect.
* `calc_coherency_rotation(ra, dec, alt, az, time, location)` returns
  the per-source `2×2×N_src` matrix that rotates the equatorial coherency
  matrix to the alt/az frame — adopted from `pyradiosky`'s
  `coherency_calc` but vectorised. This matches `pyradiosky` to machine
  precision in `test_coordinates.py::test_coherency_calc`.
* The internal helpers `_calc_rotation_matrix`,
  `_calc_average_rotation_matrix`, `vecs2rot`, `axis_angle_rotation_matrix`
  (Rodrigues), `spherical_basis_vector_rotation_matrix`, `theta_hat`,
  `phi_hat` jointly implement the Procrustes-orthogonalised
  ICRS→altaz transform exactly as `pyradiosky` does.
* `get_array_module(*x)` mirrors `cupy.get_array_module` and falls back
  to numpy when GPU is absent — the `xp = get_array_module(ra, dec)`
  pattern is what makes the coordinate code GPU-safe.

### 4.4 `wrapper.py` — `simulate_vis()`

The "wrapper" provides the high-level user API. Its sole responsibility
is to handle the *frequency loop* and the dispatch between CPU and GPU
backends. It accepts:

| Argument | Type | Notes |
|---|---|---|
| `ants` | `dict[int, np.ndarray]` | `{antnum → (x, y, z) in metres ENU}` |
| `fluxes` | `np.ndarray (N_src, N_freq)` | Stokes I per source per frequency |
| `ra`, `dec` | `np.ndarray (N_src,)` (rad) | RA in `[0, 2π]`, Dec in `[-π/2, π/2]` |
| `freqs` | `np.ndarray (N_freq,)` | Hz |
| `times` | `astropy.time.Time` | Vector of times |
| `beams` | `list[UVBeam | AnalyticBeam | BeamInterface]` | Length 1, `N_ant`, or arbitrary if `beam_idx` supplied |
| `telescope_loc` | `astropy.coordinates.EarthLocation` | Array centre |
| `polarized` | `bool` | Defaults False — only Stokes I; sets `N_feed=N_ax=1` and forces power beams |
| `precision` | `Literal[1, 2]` | 1 = `float32/complex64`, 2 = `float64/complex128` |
| `use_feed` | `Literal["x","y"]` | Which feed to keep when `polarized=False` |
| `use_gpu` | `bool` | Dispatches to `gpu.simulate`; raises `ImportError` if cupy missing |
| `beam_spline_opts` | `dict` | Passed to `UVBeam.interp` (CPU) or `cupyx.ndimage.map_coordinates` (GPU). For the GPU map_coordinates path the only meaningful key is `order` (default 1, linear). |
| `beam_idx` | `np.ndarray (N_ant,)` | Maps each antenna to its beam in `beams` |
| `antpairs` | `np.ndarray (N_pairs, 2)` or `list[tuple[int,int]]` | Subset of pairs to compute |
| `source_buffer` | `float ≤ 1.0` | Fraction of `nsrc` to pre-allocate in coords/exptau/Z (default 1.0). The HERA-style sky has ≈ half its sources below horizon, so `0.55–0.6` is usually fine and saves 40 % of the source-axis memory. |
| `coord_method` | `Literal["CoordinateRotationAstropy","CoordinateRotationERFA","GPUCoordinateRotationERFA"]` | Selects rotation backend |
| `coord_method_params` | `dict` | E.g. `{"update_bcrs_every": 180.0}` for the ERFA classes |
| `matprod_method` | `Literal["MatMul","VectorLoop","CPUMatMul","GPUMatMul",…]` | Default `MatMul` is auto-prefixed with `CPU/GPU` based on `use_gpu` |

Internally `simulate_vis`:

1. Decides which backend (`cpu.simulate` or `gpu.simulate`),
2. Builds an `astropy.SkyCoord` from `ra, dec` in ICRS,
3. Allocates the output array (with frequency as the leading axis),
4. Loops over frequencies and calls the backend `simulate(...)` with
   `I_sky=fluxes[:, i]` for that freq.

It does *not* do the per-time loop or the per-chunk loop — those live
inside the backend.

### 4.5 `cpu/cpu.py` — `simulate()`

The single-frequency CPU driver. Important details:

* `tracemalloc` is started for INFO-level logging if not already
  tracing, which lets `memtrace()` produce running peak-memory numbers.
* Calls `_validate_inputs(precision, polarized, antpos, times, I_sky)`
  → returns `(nax, nfeed, nant, ntimes)`. `nax` and `nfeed` are both 1
  in unpolarised mode and both 2 in polarised mode (the algorithm
  treats them identically; the distinction matters only for beams).
* `get_desired_chunks(min(max_memory, psutil.virtual_memory().available),
  …, source_buffer=source_buffer)` decides how many source chunks to use.
* Builds five core helper objects, in order:
  1. `coord_method = CoordinateRotation._methods[coord_method]` → looks
     the rotation class up in the registry (filled by
     `__init_subclass__`).
  2. `coords = coord_method(flux=√(0.5·I_sky), …, chunk_size=npixc)` —
     note the **`0.5·I_sky`** factor: matvis splits each source's
     Stokes I evenly between feeds, so each feed sees `√(I/2)`. This is
     the analogue of the RadioSim 1/2 factor in `core/polarization.py`
     and ensures `V_XX + V_YY = I` rather than `2 I`. Whether
     `polarized=True` or not, this 0.5 split is applied.
  3. `bmfunc = UVBeamInterpolator(beam_list=…, beam_idx=…, polarized=…,
     nant=…, freq=freq, spline_opts=…, precision=…, nsrc=nsrc_alloc)`.
  4. `taucalc = TauCalculator(antpos, freq, precision, nsrc=nsrc_alloc)`.
  5. `mpcls = getattr(mp, matprod_method); matprod = mpcls(nchunks,
     nfeed, nant, antpairs, precision)`.
  6. `zcalc = ZMatrixCalc(nsrc, nfeed, nant, nax, ctype)`.
* Allocates the output `vis = np.zeros((ntimes, npairs, nfeed, nfeed))`
  — note feed-feed is the *last* two axes (matching the post-v1.0.0
  contract).
* Calls `setup()` on every helper (allocates buffers, warms caches,
  performs one ICRS↔altaz transform to read the IERS table once).
* Drives the time × chunk loops as described in §3, with progress logged
  every `ntimes // max_progress_reports + 1` time samples.
* Emits `vis if polarized else vis[:, :, 0, 0]`.

The CPU driver is `~260 lines` of pure orchestration: every numerical
step is delegated to one of the five helper objects.

### 4.6 `cpu/coords.py`

Two CoordinateRotation backends.

#### `CoordinateRotationAstropy`

* Subclass of `core.coords.CoordinateRotation`.
* `setup()`: warms IERS by transforming `skycoords[0] → AltAz(obstime=times[0])`.
* `rotate(t)`: builds `AltAz(obstime=times[t], location=telescope_loc)`
  and calls `self.skycoords.transform_to(frame)` to get `(alt, az)`. It
  then writes:
  ```
  all_coords_topo[0] = cos(alt) sin(az)   # east
  all_coords_topo[1] = cos(alt) cos(az)   # north
  all_coords_topo[2] = sin(alt)           # up
  ```
  using the *astropy* azimuth convention (east-of-north).

#### `CoordinateRotationERFA`

* Implements the ICRS → CIRS → topocentric chain by direct calls to the
  Python-port of ERFA primitives, skipping refraction (which Astropy
  applies even in vacuum) and avoiding repeated spherical↔Cartesian
  conversions. Pure-numpy by default (and reused on GPU via the
  `GPUCoordinateRotationERFA` subclass).
* Pre-computes `_eci = point_source_crd_eq(ra, dec)` once at `__init__`
  (so the per-time cost is only the rotation, not the ECI build).
* Maintains `_bcrs`: the bias-precession-nutation–corrected, deflected,
  aberrated (BCRS-natural) direction vector. The BCRS pipeline is the
  expensive part (~ 90 % of the rotation), but it changes very slowly
  in time, so the `update_bcrs_every` parameter (in seconds) lets the
  user reuse the cached `_bcrs` for several time samples in a row.
  Suggested values: a few minutes is enough to stay below 10 mas error.
* The actual CIRS→ENU step (`_atioq`) builds three rotation matrices —
  CIRS→HADEC (using `astrom["eral"]`, the Earth rotation angle local),
  polar motion, and HADEC→ENU — and chains them with `xp.matmul`. The
  rotated BCRS coordinates land in `all_coords_topo`.
* `_apco(observed_frame)` is a hook that calls `erfa_astrom.get().apco`
  to get the ERFA "astrometric parameters" struct for the frame.
* The substeps `_ld` (light deflection by the Sun) and `_ab`
  (annual aberration, including the Sun-distance gravitational correction
  via `ERFA_SRS = 1.97412574336e-8`) are pure-Python in this base class
  but are *overridden* by a CUDA `RawKernel` in `GPUCoordinateRotationERFA`
  (see §4.10).

`tests/test_coordrot.py::test_accuracy_against_astropy` confirms agreement
of the ERFA path with Astropy to within 10 mas at double precision and
50 mas at single precision (the test threshold is 150 arcsec at single
precision because the true difference is dominated by `float32` roundoff).
`test_coord_rot_erfa_set_bcrs` verifies that calling `_set_bcrs` *before*
`setup()` is safe (the dual-allocation logic).
`test_larger_chunksize` verifies that varying `chunk_size` does not
change the produced topocentric coordinates.

### 4.7 `cpu/beams.py` — `UVBeamInterpolator`

A single class. Inherits `core.beams.BeamInterpolator` (see §4.10). The
`interp(tx, ty, out)` implementation:

1. Convert (tx, ty) — sin-projected ENU east/north — to (az, za) using
   `enu_to_az_za(orientation="uvbeam")`.
2. For each beam in `self.beam_list` call
   `bm.compute_response(az_array, za_array, freq_array=[freq], reuse_spline=True, check_azza_domain=False, interpolation_function="az_za_map_coordinates", spline_opts=…)`.
3. If polarised: take `[:, :, 0, :].transpose(1, 0, 2)` to get
   `(nfeed, nax, nsrc)`.
4. If unpolarised: take the only polarisation, `[0, 0, 0, :]`, and
   `np.sqrt` it (since the input is a *power* beam — `√(power) =
   amplitude`, which is what `Z` needs).

The `reuse_spline=True` and `check_azza_domain=False` flags came in v0.4.3
(per CHANGELOG) and gave a significant speedup at the cost of a one-time
spline build.

### 4.8 `cpu/matprod.py`

Two MatProd backends, both inherit `core.matprod.MatProd`:

#### `CPUMatMul`

```python
v = z.conj().dot(z.T)              # (Nant·Nfeed, Nant·Nfeed)
v.shape = (nant, nfeed, nant, nfeed)
v = v.transpose((0, 2, 3, 1))      # (nant1, nant2, nfeed2, nfeed1)
out[:] = v.reshape(...)            # or v[ant1_idx, ant2_idx]
```

The transpose `(0, 2, 3, 1)` is what makes the output match
pyuvsim's `(ant1, ant2, feed1, feed2)` ordering. A comment in
`tests/test_matprod.py` flags that this should be checked properly —
but the test as written *enforces* this transpose by construction, so
deviating from it will break compatibility.

#### `CPUVectorDot`

For each `(ai, aj)` in `antpairs`, compute `out[i] = z[aj] · z[ai].conj().T`.
This is faster when `len(antpairs) ≪ nant²` (e.g. only the unique
redundant baselines are needed). In the matvis CHANGELOG/docs this is
described as a candidate alternative; you must benchmark to know if it
beats the full GEMM.

### 4.9 `gpu/gpu.py` — `simulate()` (GPU)

Mirrors the CPU driver structurally; differences worth knowing:

* `combine_docstrings(simcpu)` (from `docstring-parser`) glues the CPU
  docstring onto the GPU one so users get one canonical reference.
* Uses `cp.cuda.Device().mem_info[0]` (free memory) instead of psutil.
* Forwards `gpu=True` to `coord_method`, `ZMatrixCalc`, `TauCalculator`,
  forcing them onto the cupy backend.
* Uses `GPUBeamInterpolator` (cupy + `map_coordinates`) for the beams.
* Picks the matprod class from `gpu.matprod` (`GPUMatMul` or `GPUVectorDot`).
* Per-chunk events: builds 8-event structs `{start, upload, eq2top, tau,
  beam, meas_eq, vis, end}` per chunk, recording `cp.cuda.Event` markers
  on a per-chunk stream — useful for profiling but the streams are not
  used for actual concurrency in the production code (each chunk's
  `stream.use()` is called sequentially in the loop). The synchronisation
  point is `events[nchunks-1]["end"].synchronize()` before
  `matprod.sum_chunks(vis[t])`.
* Skips the chunk entirely when `nsrcs_up < 1` (no sources above
  horizon).
* Same final reduction: `vis if polarized else vis[:, :, 0, 0]`.

`HAVE_CUDA` is set at import time. The CHANGELOG's `Dev` section
mentions "better handling of errors when GPUs are present but currently
unavailable" — this is realised by the broad `except Exception as e`
that wraps the cupy import; if cupy *imports* but errors during sub-import
(e.g. missing CUDA driver), `HAVE_CUDA` is silently set to False and a
warning is emitted.

### 4.10 `gpu/coords.py` — `GPUCoordinateRotationERFA`

* Inherits `cpu.coords.CoordinateRotationERFA` (so the whole `_atioq`,
  `_ab`, `_bpn`, `_set_bcrs`, `rotate` machinery is reused).
* `requires_gpu = True` → the test harness skips it on non-GPU systems.
* Overrides `_ld(p, e, em, dlim)` with a custom CUDA `RawKernel`. The
  kernel source (templated on `REAL_DTYPE`) is in `_ld_kernel_code` at
  the top of the file — it computes light deflection by the Sun in
  closed form for each source point, parallelising across sources with
  one thread per source.
* Two compiled instances are kept: `ld_kernel_single` (`float`) and
  `ld_kernel_double` (`double`). The right one is dispatched based on
  `self.precision`.

This is the only place in `matvis` where a hand-written CUDA kernel
runs at runtime. Beam interpolation goes through `cupyx.ndimage`, the
matrix product through cuBLAS — neither path uses the `.cu` files in
`gpu_src/`.

### 4.11 `gpu/beams.py` — `GPUBeamInterpolator` & `gpu_beam_interpolation`

`GPUBeamInterpolator.setup()` does two things differently depending on
beam type:

* **All UVBeams**: extracts the *raw* `(naz, nza)` data via
  `prepare_for_map_coords(uvbeam)` (which calls `uvbeam._prepare_coordinate_data`
  and grabs the first frequency channel, since the beam was already
  pre-interpolated in frequency by `core.beams._wrangle_beams`). Stores
  `(beam_data, daz, dza, azmin)` on the GPU. *Mixed* AnalyticBeam +
  UVBeam lists are explicitly rejected (see
  `tests/test_matvis_gpu.py::test_mixed_beams`).
* **All AnalyticBeams**: falls back to the CPU `UVBeamInterpolator.interp`
  on a numpy buffer and then `.set()`s the result to GPU memory. So
  analytic-beam GPU runs incur a host↔device round-trip every chunk.

`gpu_beam_interpolation(beam, daz, dza, azmin, az, za, beam_at_src,
order=1)` is the workhorse:

1. Asserts `beam.dtype` is `float{32,64}` or `complex{64,128}`.
2. For each beam index, builds `coords = [za / dza[bm], (az - azmin[bm])
   / daz[bm]]`.
3. Calls `cupyx.scipy.ndimage.map_coordinates(beam[bm, ax, fd], coords,
   order=order, output=beam_at_src[bm, fd, ax])` for each `(feed, ax)`
   pair.
4. If the input beam is real (a *power* beam), `cp.sqrt`s the output and
   recasts to complex (matching the CPU `np.sqrt(...)` step).

`spline_opts={"order": 1}` (linear) is the default. v1.0.0 explicitly
removed the older `bm_pix`/`use_pixel_beams` paths — the rationale in
the CHANGELOG is that map_coordinates on the native UVBeam grid is at
least as accurate and much simpler than maintaining a parallel
(l, m)-grid representation.

A vestige is `gpu_src/beam_interpolation.cu`: a Jinja-templated PyCUDA
implementation of the same algorithm using a 3D texture
(`bm_tex`) and shared memory to cache per-azimuth lerp factors. Not used
at runtime, but a useful reference for understanding the kernel-level
data layout (`Agrid` is `(Nbeam, Nax, Nfeed, Nza, Naz)`, `Asrc` is
`(Nax, Nfeed, Nbeam, Nsrc)` in flattened form — note that this differs
from the runtime numpy ordering by being feed/ax-major).

### 4.12 `gpu/_cublas.py`

Direct cuBLAS bindings via `cupy_backends.cuda.libs.cublas`:

* `complex_matmul(a, b, alpha=1, beta=0, out=None)` calls the right
  variant of `cgemm/zgemm` based on dtype, with `op_A = CUBLAS_OP_C`
  (conjugate-transpose), `op_B = CUBLAS_OP_N` (no-op). Result lands in
  Fortran-ordered memory with `lda = a.shape[1]`.
* `zdotz(a, …)` is just `complex_matmul(a, a, …)`, i.e. `aᴴ · a`.

Note: cuBLAS uses Fortran (column-major) ordering by default. The matrix
shapes in `gpu/matprod.py` (e.g. `(nfeed, nant, nfeed, nant)` Fortran-
ordered) are explicitly chosen so that the `zdotz` output is what
`pyuvsim` expects after a `transpose((1, 3, 2, 0))` step in
`sum_chunks`.

`tests/test_cublas.py::test_zdotz` checks that `zdotz(a)` equals
`np.dot(a.conj(), a.T)` to single- and double-precision tolerance.

### 4.13 `gpu/matprod.py`

Two backends:

#### `GPUMatMul`

* `allocate_vis()` → list of cupy `(nfeed, nant, nfeed, nant)` Fortran-
  ordered arrays, one per chunk.
* `compute(z, out)` → `zdotz(z, out=out); cp.cuda.Device().synchronize()`.
* `sum_chunks(out)` → reduce chunks, copy to host, transpose
  `(1, 3, 2, 0) → (ant1, ant2, feed2, feed1)`, then either reshape to
  `(nant·nant, nfeed, nfeed)` or index with `(ant1_idx, ant2_idx)`.

#### `GPUVectorDot`

* For each `(ai, aj)` antpair, calls `complex_matmul(z[ai], z[aj], out=out[:, :, i])`.
* `sum_chunks` does a `transpose((2, 1, 0)).get()`.

`tests/test_matprod.py::test_matprod` parameterises across all four
classes (`CPU/GPU × MatMul/VectorDot`) × `nfeed in (1, 2)` × `precision
in (1, 2)` × `nchunks in (1, 2)` and verifies they all return the same
answer as `np.dot(z.conj(), z.T).reshape(...).transpose((0, 2, 3, 1))`.

### 4.14 `core/` — abstract base classes shared between CPU and GPU

These are the contracts both backends adhere to.

#### `core.__init__._validate_inputs(precision, polarized, antpos, times, I_sky)`

* Asserts `precision in {1, 2}`, `antpos.shape == (nant, 3)`, `I_sky.ndim == 1`.
* Returns `(nax, nfeed, nant, ntimes)` with `nax=nfeed=2` for polarised,
  `=1` otherwise.

#### `core.beams.prepare_beam_unpolarized(beam, use_feed="x", allow_beam_mutation=False)`

If the beam is already a single-pol power beam, returns it unchanged.
Otherwise:
1. If efield → `beam.as_power_beam(include_cross_pols=False, allow_beam_mutation=…)`.
2. If `Npols > 1` → `beam.with_feeds([use_feed])`.

This is what makes the unpolarised path use power beams (which is
essentially `|E|²`) while the polarised path needs the original efield.
Tested by `tests/test_beams.py::TestPrepareBeamUnpolarized`.

#### `core.beams._wrangle_beams(beam_idx, beam_list, polarized, nant, freq)`

* Wraps every entry in `BeamInterface(...)` (so AnalyticBeams and
  UVBeams are unified).
* Validates `beam_idx`: must be either `None` (with `nbeam ∈ {1, nant}`)
  or a length-`nant` integer array with `0 ≤ i < nbeam`.
* For each `BeamInterface` whose `_isuvbeam` is True (i.e. wraps a real
  UVBeam, not an analytic one), pre-interpolates the beam to the
  scalar `freq` and clones the BeamInterface with the resulting
  single-frequency UVBeam. This pre-interpolation is the v0.4.3
  speedup mentioned in the CHANGELOG.
* If `polarized`, asserts every beam is `efield` type.
* Returns `(beam_list, nbeam, beam_idx)`.

#### `core.beams.BeamInterpolator` (ABC)

* Stores `beam_list`, `beam_idx`, `polarized`, `nant`, `freq`,
  `spline_opts`, `nsrc`, plus precision-derived `complex_dtype`,
  `real_dtype`. `nfeed = nax = 2` (polarised) or `1` (unpolarised).
* `setup()` allocates `self.interpolated_beam = np.zeros((nbeam, nfeed,
  nax, nsrc), complex_dtype)`.
* `interp(tx, ty, out)` is the abstract method subclasses implement.
* `__call__(tx, ty, check=True)` invokes `interp` and, if `check`,
  validates that no NaN/Inf snuck through the interpolation. (v1.0.1
  added the ability to disable this check; in the live code it is
  disabled for `t > 0` — see `bmfunc(crd_top[0], crd_top[1], check=t == 0)`
  in `cpu/cpu.py` and `gpu/gpu.py`.)

#### `core.coords.CoordinateRotation` (ABC + registry)

* `_methods = {}` is a class-level dict. `__init_subclass__` adds every
  subclass automatically — this is what lets the user select a backend
  by name string (`"CoordinateRotationAstropy"`, `"CoordinateRotationERFA"`,
  `"GPUCoordinateRotationERFA"`).
* `requires_gpu: bool = False` — flag inspected by the wrapper to refuse
  GPU-only backends in CPU mode and vice versa.
* `__init__` packages flux, times, telescope_loc, skycoords,
  chunk_size, source_buffer, precision, gpu — and crucially decides
  `sky_model_dtype = ctype if iscomplex(flux) else rtype`. It also
  derives `nsrc_alloc = chunk_size · source_buffer` if `chunk_size > 1000`,
  otherwise `nsrc_alloc = chunk_size`. This is the buffer used to hold
  the *above-horizon* sub-array; if more sources than `nsrc_alloc` end
  up above the horizon at any time, `select_chunk` raises a `ValueError`
  asking the user to bump `source_buffer`.
* `setup()` allocates:
  - `all_coords_topo  (3, nsrc)`
  - `coords_above_horizon  (3, nsrc_alloc)`
  - `flux_above_horizon  (nsrc_alloc, …flux.shape[1:])`
* `select_chunk(chunk, t)`:
  - Slices `[chunk·chunk_size, (chunk+1)·chunk_size)` out of
    `all_coords_topo` and `flux`.
  - Masks where `topo[2] > 0` (above horizon) and copies into the
    pre-allocated buffers.
  - For polarised input (`flux.ndim == 4`, i.e. a coherency cube), it
    rotates each above-horizon source's coherency from equatorial to
    altaz via `_rotate_frame_coherency` (calls
    `coordinates.calc_coherency_rotation` and an einsum).
  - Returns `(coords_above_horizon, flux_above_horizon, n)`.
* `_rotate_frame_coherency(coh, ra, dec, alt, az, time)` does
  `xp.einsum("ban,nfbc,cdn->nfad", R, coherency, R)` — i.e. a 2×2
  similarity transform per source per frequency.
* `rotate(t)` is abstract.

#### `core.getz.ZMatrixCalc`

* Allocates `self.z` of shape `(nfeed·nant, nax·nsrc)`.
* `__call__(sqrt_flux, beam, exptau, beam_idx)`:
  1. `exptau *= sqrt_flux` (in-place; `sqrt_flux` broadcast over antennas)
  2. Reshape `self.z` into `(nant, nfeed, nax, nsrc)`, broadcast `exptau`
     into every (feed, ax) slot.
  3. Multiply by `beam` (or `beam[beam_idx]` if heterogeneous).
  4. Re-flatten to `(nfeed·nant, nax·nsrc)` for the GEMM.

The "Z" matrix is the per-antenna voltage response per source per feed
per E-field axis, and `V = Z · Zᴴ` recovers all baseline visibilities.

#### `core.matprod.MatProd` (ABC)

* Stores `nchunks`, `nfeed`, `nant`, `antpairs` (Cartesian if `None`),
  `npairs`, complex dtype, and split antenna-index arrays
  `ant1_idx`, `ant2_idx`.
* `allocate_vis()` allocates `(nchunks, npairs, nfeed, nfeed)`.
* `__call__(z, chunk)` calls `compute(z, out=self.vis[chunk])`.
* `sum_chunks(out)`: if `nchunks == 1`, copies; else `vis.sum(axis=0,
  out=out)`.

#### `core.tau.TauCalculator`

* Pre-multiplies `antpos` by `(2π·freq/c)·1j` at init — so
  `self.antpos = (2π i ν/c) D`, an imaginary-valued matrix.
* `setup()` allocates `exptau = zeros((nant, nsrc), ctype)` and moves
  `antpos` to GPU if needed.
* `__call__(crdtop)`:
  ```
  matmul(antpos, crdtop, out=exptau)   # τ_kl = (2π i ν/c) D_k·X_l
  exp(exptau, out=exptau)              # exp(τ)
  ```
  with optional `cp.cuda.Device().synchronize()`.

### 4.15 `cli.py`

A standalone Click-based profiler — invokable as `matvis profile …` or
`matvis hera-profile …` after `pip install matvis[profile]`. It uses
`line_profiler.LineProfiler` to instrument `simulate_vis` and reports
per-line and per-section timings (the `STEPS` dict maps line-substring
patterns to the named algorithm steps in §3). Outputs a `full-stats-…txt`
and a pickled `summary-stats-…pkl`. Not part of the runtime path.

The `hera_profile` subcommand even wires in `py21cmsense.antpos.hera`
to build a real HERA layout (`hex_num`, `outriggers`, `keep_ants`) and
auto-derives a redundancy-aware antpair list.

`get_standard_sim_params(...)` here is a *different* helper from the one
in `_test_utils.py`: this one uses uniformly random sources rather than
a pyradiosky `SkyModel`. The only shared structure is the seed pattern:
the first source is always at zenith for HERA so visibilities are
non-zero in the smallest possible test.

---

## 5. Sky-model contract

`matvis` accepts:

* For unpolarised use: a 1-D `I_sky` of shape `(N_src,)` per call into
  `cpu.simulate` / `gpu.simulate`. The wrapper accepts a `(N_src,
  N_freq)` cube and slices.
* For polarised use: the *coherency cube* — a 4-D array of shape
  `(N_src, N_freq, 2, 2)`. The CoordinateRotation detects polarisation
  by `flux.ndim == 4` and rotates each per-source 2×2 from the
  equatorial frame to the local alt/az frame on the fly. This rotation
  is identical (up to vectorisation) to `pyradiosky.SkyModel._calc_coherency_rotation`.

The 0.5 factor on `I_sky` (in `cpu/cpu.py`) splits Stokes I evenly
between feeds. RadioSim applies the analogous 0.5 factor in
`core/polarization.py`. *This means*: feeding `I_sky` directly is the
standard Stokes-I convention; the resulting `V_XX + V_YY = I` rather
than `2 I`.

The CHANGELOG (Limitations section in README) also flags:

* No support for *truly polarised sky models* in the wrapper API
  (despite the polarised coherency support in the core — the wrapper
  only takes Stokes-I `fluxes`). Polarised use requires calling
  `cpu/gpu.simulate` directly with a 4-D `flux` array.
* No exploitation of baseline redundancy to deduplicate work — every
  baseline is computed individually.
* Diffuse skies must be pre-pixelised (HEALPix is the assumed
  discretisation, but matvis is agnostic as long as each pixel
  represents an equal-area sky region).

---

## 6. Beam-handling pipeline

| Stage | Where | Effect |
|---|---|---|
| User passes `beams: list[UVBeam | AnalyticBeam | BeamInterface]` to `simulate_vis` | `wrapper.py` | None |
| In `cpu/gpu.simulate` → `BeamInterpolator.__init__` → `_wrangle_beams` | `core/beams.py` | Wraps in `BeamInterface`, normalises to power-or-efield depending on `polarized`, *pre-interpolates UVBeams to the single sim frequency* |
| `BeamInterpolator.setup()` | `core/beams.py` | Allocates `interpolated_beam (nbeam, nfeed, nax, nsrc, complex_dtype)` |
| Per-time, per-chunk `bmfunc(tx, ty, check=t==0)` | `cpu/beams.py` or `gpu/beams.py` | Interpolates beam at sin-projected ENU coords; CPU uses `UVBeam.compute_response(...interpolation_function="az_za_map_coordinates"...)`; GPU uses `cupyx.ndimage.map_coordinates` on the raw az/za grid |

Important constraints:
* **GPU UVBeams must use `pixel_coordinate_system == "az_za"`** —
  HEALPix and other systems are explicitly rejected (see
  `tests/test_beams.py::TestGPUBeamInterpolator::test_exceptions`).
* **GPU lists must be homogeneous**: all UVBeam *or* all AnalyticBeam.
  Mixed lists raise `ValueError("GPUBeamInterpolator only supports
  beam_lists with either all UVBeam or all AnalyticBeam objects.")`.
* `beam_idx` lets you assign a beam index per antenna (e.g. for arrays
  with two distinct dish types). Length = `nant`. If omitted, the
  beam list must have length 1 or `nant`.
* `beam_spline_opts` is forwarded as the `spline_opts` kwarg. For CPU
  `UVBeam.interp`, this means scipy `RectBivariateSpline` knobs (`kx`,
  `ky`). For GPU `map_coordinates`, the only meaningful key is `order`
  (default linear, 1).

---

## 7. Memory chunking

The chunking model is described in `_utils.get_required_chunks`:

```
gpusize = {
    "antpos":      nant * 3       * rsize,
    "flux":        nsrc           * rsize,
    "beam":        nbeampix*nfeed*nax*csize,
    "crd_eq":      3*nsrc         * rsize,
    "eq2top":      9              * rsize,
    "crd_top":     3*nsrc         * rsize,
    "crd_chunk":   3*nchunk       * rsize,
    "flux_chunk":  nchunk         * rsize,
    "exptau":      nant*nchunk    * csize,
    "beam_interp": nbeam*nfeed*nax*nchunk*csize,
    "zmat":        nchunk*nfeed*nant*nax*csize,
    "vis":         ch*nfeed*nant*nfeed*nant*csize,
}
```

* `freemem` is `psutil.virtual_memory().available` on CPU, or
  `cp.cuda.Device().mem_info[0]` on GPU. The user can lower it with
  `max_memory=` to force more chunks (useful for shared-GPU systems).
* `min_chunks=` lets the user *force* at least N chunks regardless of
  fit. Default 1.
* The doctest example in the source (`>>> get_required_chunks(1024, 2,
  4, 8, 16, 32, 64, 32) → 1`) is *wrong*-looking on first read because
  with a 1024-byte budget no real config fits — but the function is
  exercising the corner case where only one chunk is needed and the
  loop exits at the cap.

The polarised flux case adds `(N_src, N_freq, 2, 2)` to memory but the
wrapper passes only `flux[:, i]` per simulate call (a `(N_src, 2, 2)`
slice), so per-call the polarised footprint scales as `4× nsrc · csize`
relative to the unpolarised path.

`source_buffer` controls how much *extra* room is kept around the
above-horizon sub-array; the default 1.0 means "as many sources can be
above horizon as there are sources total". `0.55–0.6` gives roughly the
correct half-sky cull factor for HERA-style drift-scans and saves
≈ 40 % of the source-axis-dependent buffers (`exptau`, `beam_interp`,
`zmat`, the `coords_above_horizon` buffer in `CoordinateRotation`).

---

## 8. Coordinate-rotation backends compared

| Method | requires_gpu | Per-time cost | Approx. vs. Astropy | Tunable? |
|---|---|---|---|---|
| `CoordinateRotationAstropy` | False | High (full astropy AltAz transform) | Reference | No |
| `CoordinateRotationERFA` | False | Medium (90 % of cost is the BCRS step, which can be cached) | ≤ 10 mas at default settings | `update_bcrs_every` (seconds) |
| `GPUCoordinateRotationERFA` | True | Low (custom `_ld` kernel, rest reused from CPU class) | Same as ERFA | Same |

The ERFA backends skip atmospheric refraction and the spherical↔Cartesian
round-trips. For drift-scan HERA-style observations, ERFA with
`update_bcrs_every=180` (3 minutes) is the recommended default. For
nights-long simulations, set it lower (or fall back to Astropy).

---

## 9. Polarisation handling

The polarised path is fully wired through every layer:

* **Inputs**: `polarized=True` flips `nax = nfeed = 2` and forces
  `_wrangle_beams` to require efield beams.
* **Sky**: a `(N_src, N_freq, 2, 2)` coherency cube is detected by
  `iscomplexobj(flux)` *or* `flux.ndim == 4`. CoordinateRotation rotates
  the coherency per-source per-time into the local altaz frame.
* **Beam**: the interpolated beam now has shape `(nbeam, 2, 2, nsrc)` —
  the CPU path uses `[:, :, 0, :].transpose(1, 0, 2)` to pull
  `(nfeed, nax, nsrc)` out of the `(nfeed, nax, nfreq=1, nsrc)` block.
* **Z**: shape `(2·nant, 2·nsrc)`.
* **V**: `(nant, nant, nfeed=2, nfeed=2)` per chunk.
* **Output**: `(N_times, N_pairs, 2, 2)`, i.e. four cross-feed
  visibilities per baseline (`nn`, `ne`, `en`, `ee`).

The 1/2 split on `I_sky` is applied identically in unpolarised and
polarised mode — in polarised mode the `0.5 ·  I_sky` is the value of
the coherency at zero Q/U/V. Real polarised input (Q/U/V ≠ 0) requires
calling the backend `simulate` directly, not `simulate_vis`, since the
wrapper only accepts a Stokes-I `fluxes`.

`tests/test_compare_pyuvsim.py::test_compare_pyuvsim` parameterises
across `polarized × use_analytic_beam` and verifies agreement with
`pyuvsim.uvsim.run_uvdata_uvsim` to `rtol=2e-4` (analytic beam) or
`rtol=1e-2` (UVBeam). `tests/test_coordrot.py::test_polarized_flux`
checks that the per-source coherency rotation matches pyradiosky's
result element-wise.

---

## 10. Tests (`tests/`)

| File | Coverage |
|---|---|
| `conftest.py` | Two session-scoped UVBeam fixtures from `data/NF_HERA_Dipole_small.fits` (raw efield + power-collapsed `xx`) |
| `test_beam_interp_gpu.py` | Identity test (interpolation at grid nodes), simple linear test, dtype-rejection test for `gpu_beam_interpolation` |
| `test_beams.py` | `_wrangle_beams` exception paths, `prepare_beam_unpolarized` for 5 beam shapes, NaN-detection in `UVBeamInterpolator`, GPU-CPU beam interpolation match |
| `test_compare_pyuvsim.py` | Visibility-level parity vs. `pyuvsim` across `polarized × analytic_beam`; chunking & source-buffer parity |
| `test_coordinates.py` | `point_source_crd_eq` axis assignments, ENU↔ECI inverse property, `enu_to_az_za` for all 6 cardinal directions in both conventions, `equatorial_to_eci_coords` round-trip, coherency-rotation match against pyradiosky |
| `test_coordrot.py` | Repeatability, accuracy of ERFA & GPU backends to within 10 mas of Astropy at `precision=2`, BCRS pre-allocation, chunk-size invariance, polarised flux through CoordinateRotationAstropy |
| `test_cpu_vs_gpu.py` | End-to-end agreement between CPU and GPU `simulate_vis` across `polarized × analytic_beam × min_chunks × source_buffer`. *Imports `pycuda` as a skip-marker; fails to skip cleanly if pycuda is absent on systems where cupy is present.* |
| `test_cublas.py` | `zdotz` matches `np.dot(a.conj(), a.T)` to `complex64/128` precision |
| `test_matprod.py` | All four matprod backends produce identical results across `nfeed × precision × nchunks × antpairs-or-cartesian` |
| `test_matvis_cpu.py` | Smoke test of `simulate_vis` (1 frequency, 2 antennas, 20 sources, polarized + unpolarized) |
| `test_matvis_gpu.py` | Antizenith → 0 visibilities, multibeam vs. single beam, mixed-beam rejection, single-precision GPU agreement |
| `test_utils.py` | `human_readable_size` corner cases |
| `test_wrapper.py` | `matprod_method="CPUMatMul"` (with explicit prefix) passes through cleanly |

Tests assume a HERA telescope (`Telescope.from_known_telescopes("hera")`)
at lat ≈ −30.7215°, lon ≈ +21.4283°. The standard sim builds 250 sources
across the sky with one near zenith.

---

## 11. CLI surface

After `pip install matvis[profile]`, the entry point is
`matvis = matvis.cli:main`, which exposes:

* `matvis profile -a NANTS -s NSOURCE [-A/-I analytic-beam/UVBeam]
  [-f NFREQ] [-t NTIMES] [-b NBEAMS] [-g/-c gpu/cpu]
  [--matprod-method MatMul|VectorDot]
  [--coord-method CoordinateRotationAstropy|…ERFA|GPU…ERFA]
  [--double-precision/--single-precision]
  [--naz 360] [--nza 180] [--nchunks 1] [--source-buffer 1.0]
  [-v/-V verbose] [-l DEBUG|INFO|…] [-o outdir]`

* `matvis hera-profile -a HEX_NUM -s NSIDE [-k keep_ants]
  [--outriggers/--no-outriggers] …`
  builds a real HERA hex array from `py21cmsense.antpos.hera`, derives
  redundant antpairs, and runs the same profiler.

Both write `full-stats-<id>.txt` (raw line-by-line `LineProfiler` dump)
and `summary-stats-<id>.pkl` (a dict of `{step → (hits, time, time/hit,
percent, nlines)}` for the high-level steps named in `STEPS`).

---

## 12. Dependencies and extras

From `setup.cfg`:

* **Core (`install_requires`)**: `astropy`, `click`, `docstring-parser`,
  `line-profiler`, `numpy>=2.0`, `psutil`, `pyuvdata>=3.2.0`, `rich`,
  `scipy`.
* **`[gpu]`**: `cupy`, `jinja2` (jinja is left over from the PyCUDA
  template days; it is no longer used at runtime, only the `.cu`
  templates would have used it).
* **`[profile]`**: `click`, `line-profiler`, `pyuvsim>=1.2.5`.
* **`[test]`**: `astropy-healpix`, `hypothesis`, `ipython`, `matplotlib`,
  `pyradiosky`, `pytest`, `pytest-cov`, `pytest-lazy-fixtures`,
  `pyuvsim[sim]>=1.2.5`.
* **`[docs]`**: `furo`, `ipython`, `nbsphinx`, `numpydoc`, `sphinx`.
* **`[dev]`**: union of `docs` + `test`.
* **`[all]`**: union of `gpu` + `profile` + `dev`.

Python: `>=3.11`. License: MIT. Build: `setuptools` + `setuptools_scm`
(version derived from git tags via the empty `[tool.setuptools_scm]`
table in `pyproject.toml`).

`numpy>=2.0` is a hard floor — the package will not work with
`numpy<2.0` because it relies on the new dtype hierarchy and pyuvdata's
≥3.2.0 API.

---

## 13. Limitations, caveats, gotchas

* `wrapper.simulate_vis` only accepts Stokes-I `fluxes`; for full
  polarised sky input call `cpu/gpu.simulate` directly with a
  `(N_src, N_freq, 2, 2)` flux cube.
* Diffuse-sky support requires the user to pre-pixelate (HEALPix is the
  assumed equal-area discretisation but matvis is grid-agnostic).
* No baseline-redundancy speedup — a fully-redundant array of N
  antennas still costs `O(N²)` GEMM. Use `antpairs=` and
  `matprod_method="VectorLoop"` to skip the redundant pairs at the
  source-summing stage, but you're still paying for the full Z-matrix
  build.
* GPU UVBeams must be on the `az_za` grid; HEALPix-pixelated UVBeams are
  rejected at `setup()`.
* GPU lists cannot mix UVBeam and AnalyticBeam.
* The flat-horizon assumption means everything with `topo[2] ≤ 0` is
  silently zeroed — there is no terrain or refraction model.
* The Earth-rotation model is rigid-body single-axis; for very long
  integrations (many hours) the mismatch with full Astropy gives ~ 1
  arcsec drifts. Use `update_bcrs_every` to trade speed for accuracy.
* Memory accounting in `_utils.get_required_chunks` is *approximate* —
  it estimates buffer sizes but does not account for temporary
  allocations made by cuBLAS or by `map_coordinates`. Headroom of
  ~ 20 % is sensible.
* `tests/test_cpu_vs_gpu.py` imports `pycuda` as a skip marker — left
  over from the pre-1.0.0 PyCUDA backend. cupy-only systems may need to
  fake-install pycuda or skip this test manually.

---

## 14. Version history at a glance (CHANGELOG.rst)

| Version | Key changes |
|---|---|
| Dev | Better error handling when GPUs present-but-broken; pyradiosky ≥ 0.3.0 SkyModel handling in tests |
| 1.0.1 | Beam-coverage check fix on GPU; ability to skip the inf/nan beam check |
| **1.0.0** | GPU brought up to CPU API parity. **Removed** `bm_pix`, `use_pixel_beams`. **Added** GPU polarised support. **Faster** beam_list interp (when freq not in array). 10× speedup of `vis_cpu` from einsum→matmul. **Breaking**: output shape now `(Ntimes, Nfeed, Nfeed, Nant, Nant)` (post-1.0.0 actual: `(Ntimes, Npairs, Nfeed, Nfeed)`). `vis_cpu` and `vis_gpu` modules renamed to `cpu` and `gpu`. New "Understanding the Algorithm" docs page. |
| 0.4.3 | `UVBeam.interp` called with `reuse_spline=True`, `check_azza_domain=False` |
| 0.4.2 | Visibility integral einsum bug fix (outer product over feeds, sum over E-field components, integrate over sky) |
| 0.4.0 | Unique-beam dedup; only-above-horizon source culling (3× speedup) |
| 0.2.x | Polarisation in CPU only; new `wrapper` module with `simulate_vis`; new `coordinates` helpers (`eci_to_enu_matrix`, `point_source_crd_eq`, `equatorial_to_eci_coords`); animate_source_map plotting |
| 0.1.0 | Initial port out of `hera_sim` |

---

## 15. Glossary of internal symbols

| Symbol | Definition / shape |
|---|---|
| `nant` | Number of antennas |
| `nfeed` | 2 polarised, 1 unpolarised |
| `nax` | Number of E-field components per beam (= nfeed) |
| `nsrc` | Total source count (sky model size) |
| `nsrc_alloc` | Pre-allocated above-horizon buffer = `chunk_size · source_buffer` if `chunk_size > 1000` else `chunk_size` |
| `nbeam` | Number of *unique* beams in `beam_list` |
| `nbeampix` | Total pixels across all beams (used in chunk sizing) |
| `nchunks` | Number of source-axis chunks |
| `npairs` | `nant²` if `antpairs is None`, else `len(antpairs)` |
| `crd_eq` / `_eci` | `(3, nsrc)` ECI unit vectors for sources |
| `crd_top` / `all_coords_topo` | `(3, nsrc)` topocentric (ENU-projected) unit vectors |
| `coords_above_horizon` | `(3, nsrc_alloc)` masked subset where `topo[2] > 0` |
| `flux_above_horizon` | `(nsrc_alloc, …)` matched flux/coherency for that subset |
| `exptau` | `(nant, nsrc)` complex `exp(-2πi ν D·X/c)` |
| `interpolated_beam` | `(nbeam, nfeed, nax, nsrc)` complex |
| `Z` / `z` | `(nfeed·nant, nax·nsrc)` complex |
| `vis` | Per-time output `(npairs, nfeed, nfeed)` complex |
| `R_t` | 3×3 ICRS→altaz rotation matrix at time t (Procrustes-orthogonalised) |
| `astrom["bpn"]` | 3×3 bias-precession-nutation matrix for time t (ERFA) |
| `astrom["eral"]` | Earth rotation angle local at time t |
| `astrom["eh"]` | Heliocentric direction of observer (au) |
| `astrom["em"]` | Distance from Sun to observer (au) |
| `astrom["v"]` | Observer barycentric velocity / c |
| `astrom["bm1"]` | √(1 − \|v\|²): reciprocal Lorentz factor |
| `ERFA_SRS` | 1.97412574336e-8 — Schwarzschild radius of the Sun in au |
| `update_bcrs_every` | Time in seconds between BCRS recomputes (default 0 = every time) |
| `source_buffer` | Fraction ∈ (0, 1] for above-horizon allocation |
| `precision` | 1 → float32/complex64, 2 → float64/complex128 |
| `coord_method` | "CoordinateRotationAstropy" / "CoordinateRotationERFA" / "GPUCoordinateRotationERFA" |
| `matprod_method` | "(CPU\|GPU)(MatMul\|VectorLoop)" — the wrapper auto-prefixes if unspecified |

---

## 16. Worked example

This is the canonical "wrapper" example, distilled from
`docs/tutorials/matvis_tutorial.ipynb` and `tests/test_matvis_cpu.py`.

```python
import numpy as np
from astropy.time import Time
from pyuvdata.telescopes import Telescope
from pyuvdata.analytic_beam import GaussianBeam
from matvis import simulate_vis

NSRC, NTIMES, NFREQ = 20, 10, 5

ants = {0: (0., 0., 0.), 1: (14., 0., 0.), 2: (0., 14., 0.)}

ra  = np.linspace(0, 2*np.pi, NSRC)
dec = np.linspace(-np.pi/2, np.pi/2, NSRC)

freqs  = np.linspace(100e6, 120e6, NFREQ)
fluxes = np.ones(NSRC)[:, None] * (freqs[None, :] / 100e6) ** -2.7
times  = Time(np.linspace(2459863.0, 2459864.0, NTIMES), format="jd")

beam   = GaussianBeam(diameter=14.0)        # one beam shared by all ants

vis = simulate_vis(
    ants=ants, fluxes=fluxes, ra=ra, dec=dec, freqs=freqs, times=times,
    beams=[beam],                            # length 1 → applies to all ants
    polarized=False, precision=2,
    telescope_loc=Telescope.from_known_telescopes("hera").location,
    coord_method="CoordinateRotationERFA",
    coord_method_params={"update_bcrs_every": 180.0},
    matprod_method="MatMul",                 # auto-prefix to CPUMatMul
    source_buffer=0.6,
)
# vis.shape == (NFREQ, NTIMES, 9)   # 9 = 3·3 antpairs Cartesian
```

For polarised use: pass `polarized=True` and a list of efield beams; the
output gains two trailing axes `(2, 2)` with `nn / ne / en / ee` cross-
correlations as `[0,0] / [0,1] / [1,0] / [1,1]`.

For GPU: install `matvis[gpu]` (`pip install matvis[gpu]`), pass
`use_gpu=True`, and consider `coord_method="GPUCoordinateRotationERFA"`.

---

## 17. Pointers for integrating matvis as a "reference simulator" inside RadioSim

This is opinion based on a read-through; treat as a starting hypothesis
rather than a settled plan:

* The `Z = √I · A · exp(τ)` factorisation is already conceptually
  identical to the RIME chain in `radiosim.core.visibility` (both use the
  per-source product and a final Hermitian outer product). The only
  immediate difference is that matvis uses `(nfeed·nant, nax·nsrc)`
  flat layout and a single GEMM, whereas `radiosim` chains separate
  Jones terms.
* matvis's `core.coords.CoordinateRotation` registry is a clean pattern
  (`__init_subclass__` registry + string-keyed dispatch) that the
  radiosim `JonesChain` could borrow if you want pluggable rotation
  backends.
* matvis's pre-interpolation in `_wrangle_beams` (interp UVBeams to the
  scalar simulation frequency once, before the time loop) maps directly
  onto radiosim's `BeamFITSHandler` — verifying that radiosim hits the same
  fast path (`reuse_spline=True`, `check_azza_domain=False`,
  `interpolation_function="az_za_map_coordinates"`) is a cheap perf win.
* matvis's `gpu_beam_interpolation` via `cupyx.ndimage.map_coordinates`
  is the cleanest precedent for porting `BeamFITSHandler` to GPU
  without writing custom kernels.
* The 0.5 Stokes-I split in `cpu/cpu.py` (`np.sqrt(0.5 * I_sky)`) is
  exactly the convention `radiosim.core.polarization` uses — comparing
  against matvis is a useful regression check.
