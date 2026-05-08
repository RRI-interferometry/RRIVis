# TensorVis — Exhaustive Technical Reference

> A radio interferometer visibility simulator written for TensorFlow.
> Vendored as a git submodule of RadioSim at `simulators/TensorVis/`.

---

## 1. Overview

**TensorVis** is a small (~1.5 kLOC) Python package that computes radio interferometer visibilities for a catalogue of point sources, using **TensorFlow 2** as the array/compute backend. Its single purpose is to evaluate the per-baseline complex visibility tensor

`V_{pq}(t, ν) = Σ_s A_p(s,ν) A_q*(s,ν) · S_s(ν) · exp[2πi ν (τ_p(s,t) − τ_q(s,t))]`

for an array of antennas, an arbitrary list of LSTs, an arbitrary list of frequencies, and a list of point sources with power-law spectra. It uses TensorFlow's `@tf.function` graph compilation and is designed so that the same code path runs on CPU, GPU and TPU (via the standard TF device-placement mechanism), with optional XLA JIT acceleration via the `TF_XLA_FLAGS` env var.

Key facts (cited from `simulators/TensorVis/setup.py` and `README.md`):

| Field | Value |
|---|---|
| Name | `TensorVis` |
| Description (setup.py:13) | "Radio interferometer visibility simulator written for TensorFlow." |
| README (one line) | "Radio interferometer visibility simulator written for TensorFlow" |
| Version | `0.0.9` (`setup.py:6`) |
| Author | Phil Bull (`setup.py:9`, `LICENSE:3`) |
| Upstream URL (setup.py:11) | `https://github.com/philbull/TensorVis` |
| Submodule remote | `https://github.com/HydraRadio/TensorVis.git` (fetch & push) |
| License | MIT, Copyright (c) 2021 Phil Bull (`simulators/TensorVis/LICENSE`) |
| Languages | Python only (no C/C++ extensions, no `.pyx`, no compiled artifacts) |
| Status | Pre-alpha / experimental. No tags. 13 commits on `main`. |
| Submodule pointer | `gitdir: ../../.git/modules/simulators/TensorVis` (`.git` file content) |
| Latest commit (HEAD) | `15b0355 Specify beam frequency channels explicitly` |

### Embedded third-party code

Two of the seven Python files are *vendored verbatim* from **TensorFlow Graphics** under Apache-2.0:

| File | Source (per file header) |
|---|---|
| `TensorVis/tensorflow_interp.py` | `https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/interpolation/trilinear.py#L26-L93` (file docstring) |
| `TensorVis/tensorflow_shape.py` | `https://raw.githubusercontent.com/tensorflow/graphics/master/tensorflow_graphics/util/shape.py` (file header lines 14-17, "Copyright 2020 The TensorFlow Authors", Apache-2.0 boilerplate lines 1-13) |

These are bundled rather than imported because TensorFlow Graphics has heavy transitive deps that the author did not want to require.

---

## 2. Repository Layout

Listing produced by `find … -type f` on the working tree:

```
simulators/TensorVis/
├── .git                       # gitdir pointer (submodule)
├── .gitignore                 # standard Python ignore set (130 lines)
├── LICENSE                    # MIT, 2021 Phil Bull
├── README.md                  # 2 lines, single sentence
├── setup.py                   # setuptools, version 0.0.9
├── beam_test.py               # executable example: beam interpolation
├── vis_test.py                # executable example: full visibility sim
├── examples/
│   └── cat_gleam_cut.npy      # 2.0 MB,  NumPy v1.0, dtype <f8, shape (66639,4) – truncated GLEAM catalogue
└── TensorVis/                 # the actual Python package
    ├── __init__.py            #     3 lines
    ├── vis.py                 #   234 lines  – core visibility kernel
    ├── coords.py              #   243 lines  – RA/Dec ↔ az/za ↔ delay
    ├── beams.py               #   188 lines  – UVBeam → 3D grid + interpolator
    ├── utils.py               #    84 lines  – misc utilities (hex array, unit_interval)
    ├── tensorflow_interp.py   #    75 lines  – vendored TF Graphics trilinear interp
    └── tensorflow_shape.py    #   429 lines  – vendored TF Graphics shape utils
```

Source totals (from `wc -l`): 1256 lines under `TensorVis/`, 1501 lines including the test/example scripts and `setup.py`.

There is **no** `docs/`, `tests/`, `requirements.txt`, `pyproject.toml`, `MANIFEST.in`, `CITATION`, `AUTHORS`, `CHANGELOG`, CI configuration, or Sphinx config in the repository. The README is a single sentence.

---

## 3. Installation & Dependencies

### From `setup.py` (lines 17-23)

```python
install_requires=[
    'numpy>=1.19',
    'scipy',
    'pyuvdata',
    'tensorflow>=2.2',
    "future"
]
```

| Dependency | Used in | Notes |
|---|---|---|
| `numpy>=1.19` | All modules | Array ops at the boundary, e.g. `np.linspace`, `np.meshgrid`, RNG. |
| `scipy` | Listed but not actually `import`-ed in any module on `main`. Likely vestigial / placeholder. |
| `pyuvdata` (`UVBeam`) | `beams.construct_beam_grid` consumes a `UVBeam`; `vis_test.py` and `beam_test.py` call `UVBeam.read_beamfits`. |
| `tensorflow>=2.2` | The entire compute graph. Comments in `vis.py:160` mention TF≥2.3 enables `fn_output_signature=` for `tf.map_fn`. |
| `future` | Imported only by the vendored `tensorflow_shape.py` (`from __future__ import absolute_import/division/print_function`, lines 19-21) – Python 2/3 compat fossil. |
| `astropy` (`Time`, `SkyCoord`, `FK5`, `ICRS`, `units`) | Imported at the top of `coords.py` lines 4-7, but only used by the helper functions `transform_coords()` and `compare_coords()` (lines 186-242), which are **not** exported as part of the `tf.function` simulation path. |
| `six` | Required by `tensorflow_shape.py` (line 26). Not declared in `install_requires`. |

> **Caveat** — `astropy` and `six` are imported by source modules but absent from `install_requires`. They are typically installed transitively by `pyuvdata` and TensorFlow, but a clean install of just the four declared deps will fail to import `TensorVis` (which loads `coords.py` via `from .vis import *` → `from . import coords` → `import astropy …`). This is an installation bug.

### Install

```bash
pip install .
# or, editable
pip install -e .
```

There are no extras_require, no console_scripts, and no pinned data files (`include_package_data=True` but no `MANIFEST.in`, so the GLEAM catalogue at `examples/cat_gleam_cut.npy` is not packaged in a wheel — it is repository-only).

---

## 4. Architecture

### 4.1 Layered view

```
┌───────────────────────────────────────────────────────────────┐
│  USER SCRIPT                                                  │
│    vis_test.py / beam_test.py                                 │
│    – loads UVBeam from .beamfits                              │
│    – loads .npy point-source catalogue                        │
│    – builds antpos via tv.utils.build_hex_array               │
│    – calls tv.vis(...)                                        │
└─────────────────────────────┬─────────────────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────────┐
│  PUBLIC SIMULATION API   (TensorVis/vis.py, all @tf.function) │
│    vis(antpos, lsts, freqs, ra, dec, flux, α, beams, …)       │
│      └── tf.map_fn over LSTs                                  │
│           └── vis_snapshot(per-LST)                           │
│                 ├── tf.map_fn over Nblocks                    │
│                 └── vis_ptsrc_block (the actual RIME kernel)  │
└─────────────────┬───────────────────────────┬─────────────────┘
                  │                           │
   ┌──────────────▼─────────┐    ┌────────────▼─────────────────┐
   │  COORDINATES           │    │  BEAMS                       │
   │  TensorVis/coords.py   │    │  TensorVis/beams.py          │
   │  – equatorial→topo     │    │  – construct_beam_grid       │
   │  – topo→az/za          │    │  – coords_for_interp         │
   │  – az/za→geometric τ   │    │  – interpolate_beam          │
   └────────────────────────┘    └────────────┬─────────────────┘
                                              │
                            ┌─────────────────▼──────────────┐
                            │ VENDORED TF GRAPHICS           │
                            │ tensorflow_interp.interpolate3d│
                            │ tensorflow_shape.check_static  │
                            └────────────────────────────────┘
```

### 4.2 Compute model

Everything below the user script is a TensorFlow graph. The four `@tf.function` decorators in `vis.py` (`vis_ptsrc_block`, `vis_snapshot`, `vis`) and the seven in `coords.py`/`beams.py`/`utils.py` mean TensorVis traces a single `ConcreteFunction` per signature, then dispatches it to whichever device TF chose. There is no manual CUDA, no MPI, no Dask, no Cython. Graph-mode is the only execution mode — no eager-only paths exist.

**Precision** is a global module constant. `vis.py` lines 7-8:

```python
FLOAT_TYPE = tf.float64
COMPLEX_TYPE = tf.complex128
```

`coords.py` redefines its own `FLOAT_TYPE = tf.float64` (line 9). A user wishing to switch to `tf.float32` must edit both files; the public API does not accept a dtype argument (`beams.construct_beam_grid` and `beams.interpolate_beam` do, but the visibility kernel does not).

### 4.3 Memory-management strategy

The kernel forms intermediate tensors of shape `(Nants, Nfreqs, Nptsrc)`. For HERA-class arrays (Nants ~350) and large catalogues (Nptsrc ~10⁵), this exceeds GPU memory. The author's solution (`vis.py:139-164`) is the `nblocks` parameter: the point-source list is reshaped to `(nblocks, Nptsrc/nblocks)` and processed by `tf.map_fn`, then summed. A FIXME comment on line 150 notes that chunking by frequency would be preferable.

### 4.4 XLA / JIT

There is no programmatic XLA toggle. The example scripts document the env-var approach (`vis_test.py:5-12`, `beam_test.py:5-12`):

```bash
# CPU
TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" ./vis_test.py
# GPU
TF_XLA_FLAGS="--tf_xla_auto_jit=2" ./vis_test.py
```

---

## 5. Public API

`TensorVis/__init__.py` (the entire file, 3 lines):

```python
from .vis import *
from . import beams, coords, utils
```

Because `vis.py` does not define `__all__`, `from .vis import *` exposes every public name in `vis.py`: `vis`, `vis_snapshot`, `vis_ptsrc_block`, plus the constants `FLOAT_TYPE`, `COMPLEX_TYPE`, `C`, `freq_ref`, `phase_fac`, `ZERO`, `PI`, plus the (locally aliased) modules `np`, `tf`, `coords`, `tvbeams`. The `beams`, `coords`, `utils` sub-modules are also exposed.

### 5.1 `tv.vis(...)` — the top-level entry point (`vis.py:172-232`)

```python
@tf.function
def vis(antpos, lsts, freqs, ra, dec, flux, spectral_idx,
        beams, freq_range, nblocks=1) -> Tensor[complex128]
```

| Argument | Shape | Units | Meaning |
|---|---|---|---|
| `antpos` | `(Nants, 3)` | meters, topocentric Cartesian | Antenna positions (x, y, z). |
| `lsts` | `(Nlsts,)` | radians | LSTs to evaluate. |
| `freqs` | `(Nfreqs,)` | Hz | Frequencies. |
| `ra`, `dec` | `(Nptsrc,)` each | radians | Point-source equatorial coords (must already be precessed to current epoch). |
| `flux` | `(Nptsrc,)` | Jy (assumed) | Flux at 100 MHz. |
| `spectral_idx` | `(Nptsrc,)` | dimensionless | Spectral index α (S∝(ν/ν₀)^α). |
| `beams` | `(Nants_or_1, Ngrid_az, Ngrid_za, Ngrid_freq)` complex | — | 3D beam interpolation table (real+imag stored as `tf.complex` inside an extra leading axis). |
| `freq_range` | tuple/list of 2 floats | Hz | (min, max) of beam interpolation grid in frequency. |
| `nblocks` | int (default 1) | — | Split point sources into `nblocks` chunks for memory. Must divide Nptsrc evenly (raises `ValueError` otherwise, `vis.py:143-144`). |

Returns a complex tensor `vij_lst` of shape `(Nlsts, Nants, Nants, Nfreqs)` — full auto+cross spectrum cube for every LST.

### 5.2 `tv.vis_snapshot(...)` (`vis.py:98-169`)

Same arguments as `vis(...)` minus `lsts`, plus pre-computed `az`, `za` instead of `ra`, `dec`. Returns shape `(Nants, Nfreqs)` per the actual `return v` (line 169) — note the docstring describes the outer-product form `(Nants, Nants, Nfreqs)` but the code returns the *square root* per-antenna response. Outer product is taken inside `vis_ptsrc_block` (line 94) and then *not* used at the snapshot level — instead, blocks are summed in their per-antenna form, and the outer product at the LST level is also not re-applied. This is an inconsistency the user must be aware of: the *actual* return of `vis(...)` is whatever `vis_snapshot` returns wrapped over LSTs, i.e. the per-antenna voltage response `(Nlsts, Nants, Nfreqs)` *not* a baseline cube. (See `vis.py` lines 165-169 vs. the commented-out `tf.einsum('ik,jk->ijk', conj(v), v)`.)

> See §11 "Notable internals & inconsistencies" for the full discussion.

### 5.3 `tv.vis_ptsrc_block(...)` (`vis.py:18-95`)

Lowest-level kernel for one block of point sources at one snapshot. Returns `(Nants, Nants, Nfreqs)` via `tf.einsum('kij,lij->kli', conj(vis), vis)` — this *does* form the outer product. The output sums over point sources implicitly via the einsum contraction over `j` (Nptsrc).

### 5.4 `tv.beams` (`beams.py`)

| Function | Signature | Purpose |
|---|---|---|
| `construct_beam_grid(uvb, Nza, Naz, freq=None, axis=0, feed=0, spw=0, dtype=tf.float32, **uvbeam_interp_opts)` | Plain Python (calls `uvb.interp` from pyuvdata) | Sample a UVBeam on a regular `(Nza, Naz, Nfreq)` grid; return `(grid_re, grid_im)` as complex tensors with extra leading/trailing singleton axes shaped `(1, Naz, Nza, Nfreq, 1)`. |
| `coords_for_interp(za, az, freqs, freq_range, grid_shape, dtype=tf.float64)` | `@tf.function` | Build the trilinear `sample_pts` tensor in unit-cube coords for `interpolate3d`. |
| `interpolate_beam(grid_re, grid_im, za, az, freqs, freq_range, dtype=tf.float64)` | `@tf.function` | Trilinear-interpolate the gridded beam at source positions × frequencies. Returns complex tensor of shape `(Nptsrc, Nfreqs)`. |

### 5.5 `tv.coords` (`coords.py`)

| Function | Decorator | Purpose |
|---|---|---|
| `equatorial_to_topocentric(ra, dec, lst, latitude=HERA_LATITUDE)` | `@tf.function` | RA/Dec → topocentric direction cosines (m, l, n) via the standard rotation matrix. |
| `topocentric_to_az_za(l, m)` | `@tf.function` | Direction cosines → (az, za); below-horizon sources clipped (`tf.where(lsqr<1, sqrt(1-lsqr), 0)`). |
| `topocentric_to_delay(topo_cosines, antpos)` | `@tf.function` | Per-antenna geometric delay τ = (b·n̂)/c. |
| `az_za_to_delay(az, za, antpos)` | `@tf.function` | Same, starting from (az, za). |
| `eq_to_az_za(ra, dec, lst, latitude=HERA_LATITUDE)` | `@tf.function` | One-shot RA/Dec → az/za. The implementation contains the comment `# FIXME: There is a bug in this code` (line 144). The `vis(...)` path does **not** use this function — it goes through `equatorial_to_topocentric → topocentric_to_az_za` instead. |
| `transform_coords()` | plain | Demonstrative astropy precession code. References undefined names (`ra`, `dec`, `times`, `hera_location`) — never runs. |
| `compare_coords(ra, dec, times, lsts, use_cirs=False)` | plain | Comparison harness vs astropy. References undefined `hs` (HERA stack), `eq_to_altaz`, `CIRS`, `hera_location` — appears to be a broken scratchpad. |

`HERA_LATITUDE = -0.5361917991288512` rad (line 12, ≈ −30.7215°). Used as the default latitude everywhere — the package is implicitly HERA-centric.

### 5.6 `tv.utils` (`utils.py`)

| Function | Purpose |
|---|---|
| `unit_interval(x, xmin, xmax, scale_factor=1.)` | Affine rescale `[xmin, xmax] → [0, scale_factor]` (`@tf.function`). Used by `coords_for_interp`. |
| `build_hex_array(hex_spec=(3,4), ants_per_row=None, d=14.6)` | Build a hex-packed antenna position dict; default spacing 14.6 m (HERA short baseline). Returns `{int_id: (x, y, z)}`. |

### 5.7 What is *not* there

- No CLI / `console_scripts`. Running TensorVis means executing one of the example scripts.
- No `Simulator` class. The "API" is three free functions plus three sub-modules.
- No I/O for visibilities — the example writes via `np.save("test_vis_data", vis.numpy())` (`vis_test.py:120`). No UVData / MS / UVH5 export.
- No polarization. `axis=0, feed=0` is hardcoded in the example call to `construct_beam_grid` and there is no Jones-matrix structure anywhere.
- No diffuse sky / HEALPix support. Point sources only.
- No primary-beam-per-antenna heterogeneity. `vis_ptsrc_block` line 58-66 has the FIXME `"This just duplicates the same beam many times"` and uses `beams[0]` for every antenna.

---

## 6. File-by-file Breakdown

### 6.1 `TensorVis/__init__.py` (3 lines)

```python
from .vis import *
from . import beams, coords, utils
```

### 6.2 `TensorVis/vis.py` (234 lines, file-level constants `vis.py:7-15`)

Defines:
- Module dtype constants `FLOAT_TYPE = tf.float64`, `COMPLEX_TYPE = tf.complex128`.
- Physical constants `C = 299792458.`, `freq_ref = 100 MHz`, `phase_fac = 2π`, `ZERO = 0.`, `PI`.

#### `vis_ptsrc_block` (lines 18-95)

The arithmetic kernel. Step-by-step (line numbers in parentheses):

1. Compute beam values at every (az, za, ν): `tvbeams.interpolate_beam(real(beams[0]), imag(beams[0]), za, az, freqs, freq_range, dtype=…)` (60-62). Shape `(Nfreqs, Nptsrc)` after transpose.
2. Tile into `(Nants, Nfreqs, Nptsrc)` (64). FIXME at 66 notes this should be optional when only one beam pattern is used.
3. Mask sub-horizon sources by zeroing flux where `za >= π/2` (70).
4. Compute amplitude voltage `v = √S · (ν/ν₀)^(α/2)` (74). Note the half power-law because `v` is a voltage; the intensity exponent is recovered after the outer product.
5. Multiply by antenna pattern → `(Nants, Nfreqs, Nptsrc)` (75-78).
6. Compute geometric delays `τ` per antenna, shape `(Nants, Nptsrc)` (81), and angular phase `2πν` shape `(Nfreqs, 1)` (82).
7. Build phase tensor `(Nants, Nptsrc, Nfreqs)` via `tf.tensordot(τ, 2πν)` (83).
8. Multiply voltage by `exp(i·phase)` via `tf.einsum('ijk,ikj->ijk', v, exp(...))` (90). Output `(Nants, Nfreqs, Nptsrc)`.
9. Outer-product over antennas with sum over sources: `tf.einsum('kij,lij->kli', conj(vis), vis)` → returns `(Nants, Nants, Nfreqs)` (94).

#### `vis_snapshot` (lines 98-169)

Wraps `vis_ptsrc_block` with `tf.map_fn` over the `nblocks` chunks. Critical line 143-144 raises if `Nptsrc % nblocks != 0`. A FIXME at line 150 says "Chunk by frequency instead". The function reduces blocks via `tf.reduce_sum(vis_blocks, axis=0)` (164) — but **note** `vis_for_block` returns the full `(Nants, Nants, Nfreqs)` baseline cube already, so this reduce is summing baseline cubes across blocks which is mathematically correct (visibilities of disjoint source sets add). The outer-product comment block at 167-168 is dead code / vestigial.

#### `vis(...)` (lines 172-232)

Outer `tf.map_fn` over LSTs. For each LST it:
1. Calls `coords.equatorial_to_topocentric(ra, dec, lst)` to get cosines `(m, l, n)`.
2. Calls `coords.topocentric_to_az_za(topo_cosines[1], topo_cosines[0])` — i.e. passes `l` then `m`.
3. Calls `vis_snapshot(...)` and `tf.stack`s the result.

### 6.3 `TensorVis/coords.py` (243 lines)

Constants: `HERA_LATITUDE = -0.5361917991288512` rad (line 12), `C` (line 13), `PI` (line 14).

`equatorial_to_topocentric` builds the canonical rotation matrix

```
        [ -sin(LST)              cos(LST)              0           ]
R(LST,φ)=[ -sin(φ)cos(LST)       -sin(φ)sin(LST)      cos(φ)      ]
        [  cos(φ)cos(LST)        cos(φ)sin(LST)        sin(φ)      ]
```

and applies it to the equatorial direction cosines `[cos(ra)cos(dec), sin(ra)cos(dec), sin(dec)]`. The output ordering is documented as `(m, l, n)` (note: not `(l, m, n)`). The function contains two stray `print()` statements (57-58) that fire each trace.

`topocentric_to_az_za` clips `n = √(1−l²−m²)` to 0 below the horizon, producing `za = π/2 − asin(n)`, `az = −atan2(m, l)`.

`az_za_to_delay` and `topocentric_to_delay` both implement `τ_p = b_p · n̂ / c`. The first reconstructs `n̂` from (az, za), the second consumes already-computed cosines. `vis(...)` uses `az_za_to_delay` indirectly (via `vis_ptsrc_block:81`).

`eq_to_az_za` is documented as buggy by the author (line 144) and is not on the live code path.

`transform_coords` and `compare_coords` (186-242) are dead/scratch code with undefined references.

### 6.4 `TensorVis/beams.py` (188 lines)

Three pieces:

1. **`construct_beam_grid`** — pure Python, calls `uvb.interp(az.flatten(), za.flatten(), freq_array=…, **uvbeam_interp_opts)` from pyuvdata. Reshapes the returned `(Naxes_vec, Nspws, Nfeeds, Nfreqs, Naz*Nza)` array, picks `[axis, spw, feed, :, :]`, swaps axes to `(az, za, freq)` and wraps the real/imag halves in `tf.constant`s with extra dims `(1, Naz, Nza, Nfreq, 1)`. Note this reshape uses `np.swapaxes(beam, 0, -1)` with a freshly reshaped `(Nfreq, Nza, Naz)` array — the resulting axis order is `(Naz, Nza, Nfreq)`, not `(za, az, freq)` as the docstring suggests. The downstream `coords_for_interp` re-extracts shapes with `(grid_re.shape[2], grid_re.shape[1], grid_re.shape[3])` to compensate, so the convention is internally consistent but easy to misread.
2. **`coords_for_interp`** — converts source `(za, az)` and `freqs` to unit-cube coordinates inside the grid. Uses `tf.repeat` over Nfreqs and `tf.tile` over Nptsrc to produce the full Cartesian product; output is `(1, Nptsrc*Nfreqs, 3)` — the leading `1` is a batch dim required by `interpolate3d`.
3. **`interpolate_beam`** — calls `interpolate3d` separately for real and imaginary grids, combines via `tf.complex`, reshapes to `(Nptsrc, Nfreqs)`. A FIXME at line 186-187 flags that the reshape needs verification.

### 6.5 `TensorVis/tensorflow_interp.py` (75 lines)

Verbatim copy of the trilinear interpolator from TensorFlow Graphics. Public API: `interpolate3d(grid_3d, sampling_points, name="trilinear_interpolate")`. Algorithm:

1. Floor sample points → `bottom_left`; `top_right = bottom_left + 1`.
2. Build the 8 corner index tensors for each voxel.
3. Clip indices into `[0, voxel_cube_shape − 1]`.
4. Gather voxel values via `tf.gather_nd(..., batch_dims=…)`.
5. Compute trilinear weights as products `weights_x * weights_y * weights_z` over the 8 corners.
6. Sum `weights * content` via `tf.add_n(tf.split(...))`.

Apache-2.0 license carries through (header preserved on `tensorflow_shape.py`; not on `tensorflow_interp.py` though the docstring cites the source URL).

### 6.6 `TensorVis/tensorflow_shape.py` (429 lines)

Verbatim copy of `tensorflow_graphics/util/shape.py` with the original Apache-2.0 header (lines 1-13). Provides:

- `is_broadcast_compatible`, `get_broadcasted_shape`, `_broadcast_shape_helper`.
- `check_static(tensor, has_rank=…, has_rank_greater_than=…, has_dim_equals=…, …)` — used in `interpolate3d` (lines 26-37 of `tensorflow_interp.py`).
- `compare_batch_dimensions`, `compare_dimensions`.
- `is_static`, `add_batch_dimensions`.
- `__all__ = []` — these are deliberately not exported.

### 6.7 `TensorVis/utils.py` (84 lines)

`unit_interval(x, xmin, xmax, scale_factor=1.)` (line 7, `@tf.function`):
```python
return scale_factor * (x - xmin) / (xmax - xmin)
```

`build_hex_array(hex_spec=(3,4), ants_per_row=None, d=14.6)` (line 35) — generates a HERA-style hex with `ants_per_row` taken from `hex_spec` if not supplied: e.g. `(3,4)` → `[3,4,3]`. Row spacing is `d·sin(60°) = d·√3/2`. Returns `{int_id: (x, y, 0.)}` with the array centred on the origin.

### 6.8 `vis_test.py` (128 lines, executable)

End-to-end smoke test (`#!/usr/bin/env python` shebang). Defaults: `Nlsts=4, Nfreqs=8, Nptsrc=50, NBLOCKS=5`, hex `(3,4)` (10 antennas). Reads catalogue from `examples/cat_gleam_cut.npy` (a `(Nrows, 4)` array with columns `[ra_deg, dec_deg, flux, spectral_index]`). Reads beam from a hard-coded path `/home/phil/hera/hera_pspec/hera_pspec/data/HERA_NF_efield.beamfits`. Logs to `./logs/` via `tf.summary`. Writes `test_vis_data.npy`.

CLI: `./vis_test.py Nlsts Nfreqs Nptsrc [Nblocks]`.

### 6.9 `beam_test.py` (87 lines, executable)

Beam-only timing/correctness test. Generates 500 random sources, builds the beam grid (100×101 az/za points), times `interpolate_beam`. Same hard-coded HERA beam path.

### 6.10 `examples/cat_gleam_cut.npy` (2.0 MB)

NumPy v1.0 file, dtype `<f8`, **shape `(66639, 4)`** (per `file(1)` magic). Columns are `[ra_deg, dec_deg, flux, spectral_index]` based on how `vis_test.py:87` unpacks `np.load(catalogue_path)[:Nptsrc,:].T`. This is a "cut" subset of the GLEAM catalogue.

---

## 7. Core Algorithm — the RIME implemented

TensorVis implements the simplest scalar (unpolarised, per-feed) form of the radio interferometer measurement equation:

```
V_pq(t, ν) = Σ_s  A_p(s, ν) · A_q*(s, ν) · S_s(ν₀) · (ν/ν₀)^α_s · exp[i 2π ν (τ_p − τ_q)]
```

Implementation choices:

| Symbol | TensorVis variable | File:line |
|---|---|---|
| `ν₀` (reference freq) | `freq_ref = 100 MHz` | `vis.py:12` |
| `c` | `C = 299792458.` | `vis.py:11`, `coords.py:13` |
| `S_s` | `flux` (Jy) | API |
| `α_s` | `spectral_idx` | API |
| `(ν/ν₀)^(α/2)` (voltage form) | `(freqs/freq_ref)**(0.5*spectral_idx)` | `vis.py:74` |
| `A_p(s,ν)` | trilinear interp of the UVBeam grid | `beams.interpolate_beam` |
| Below-horizon mask | `tf.where(za < π/2, flux, 0.)` | `vis.py:70` |
| `τ_p` | `(antpos · n̂)/c` | `coords.az_za_to_delay` |
| `exp(i·2πντ)` | `tf.exp(tf.complex(0, phase))` | `vis.py:90` |
| Outer product over baselines + sum over sources | `tf.einsum('kij,lij->kli', conj(vis), vis)` | `vis.py:94` |

**Key approximations**:
- No parallactic-angle rotation, no polarised Jones matrices — purely scalar.
- No K-projection / w-term beyond what is implicit in the geometric delay.
- No precession internally — the docstring of `vis(...)` (line 196) tells the user to pre-precess RA/Dec to the current epoch.
- One beam shared across all antennas (hardcoded `beams[0]` in `vis_ptsrc_block`, `vis.py:58-65`).
- Single Stokes I, no Q/U/V.

---

## 8. Input & Output Formats

### 8.1 Inputs

| Asset | Format | Source |
|---|---|---|
| Antenna positions | `numpy.ndarray (Nants, 3)` (m, ENU) → `tf.Tensor`. Supplied by user; helper `tv.utils.build_hex_array` produces a dict that the user must `np.column_stack(...)` into a tensor (see `vis_test.py:64-66`). |
| Frequencies, LSTs, RA, Dec, flux, α | 1-D `tf.Tensor`s of `FLOAT_TYPE = float64`. |
| Primary beam | `pyuvdata.UVBeam` object loaded from a `*.beamfits` file via `uvb.read_beamfits(path)`, gridded with `tv.beams.construct_beam_grid`. |
| Point-source catalogue | `.npy` array, columns `[ra_deg, dec_deg, flux, spectral_index]`. |

### 8.2 Outputs

`tv.vis(...)` returns an in-memory `tf.Tensor` of complex-valued visibilities. The example saves it via:

```python
np.save("test_vis_data", vis.numpy())   # vis_test.py:120
```

No CASA MS, UVH5, UVFITS or any other interferometry-standard output is written.

### 8.3 Logging

`vis_test.py:59` opens a `tf.summary.FileWriter("./logs")` so TensorBoard can show the graph. Nothing else is logged.

---

## 9. Testing Layout

There is **no** test suite in the repository. No `tests/` directory, no `pytest.ini`, no `conftest.py`, no CI YAML. The two top-level scripts `beam_test.py` and `vis_test.py` are *example/timing scripts* despite their `_test` suffix — they are not pytest-collectable and they hard-code paths under `/home/phil/hera/...` for the input beamfits.

A user wanting to validate TensorVis would have to (a) supply their own beamfits, (b) run `./vis_test.py 4 8 50 5`, and (c) sanity-check the saved `test_vis_data.npy` themselves.

---

## 10. Integration & Extension Points (for embedding into RadioSim)

TensorVis is intentionally narrow. To embed it as a backend inside a wider simulator one would have to:

| Need | Where to extend |
|---|---|
| Per-antenna heterogeneous beams | Replace `beams[0]` in `vis_ptsrc_block` (`vis.py:58-65`) and remove the tile in line 64; the kernel already broadcasts over `Nants`. |
| Polarisation / Jones | The current scalar `v` (line 74) and outer-product einsum (line 94) would need to grow to 2×2 Jones matrices and a Hermitian product. This is a non-trivial rewrite. |
| Diffuse / HEALPix sky | No support; would need a new kernel. |
| Float32 mode | Change `FLOAT_TYPE`/`COMPLEX_TYPE` in `vis.py:7-8` and `coords.py:9`. |
| Custom array layout | Pass any `(Nants, 3)` ndarray; `build_hex_array` is just a convenience. |
| Output to MS/UVH5 | Wrap the returned tensor and let `pyuvdata`/`casacore` write it. |
| GPU placement | Standard TF mechanism: `with tf.device('/GPU:0'):` or `TF_FORCE_GPU_ALLOW_GROWTH=true`. |
| XLA acceleration | `TF_XLA_FLAGS` env var; see §4.4. |

The canonical entry point for a wrapper is `tv.vis(...)`. Because every public function is `@tf.function`, calling `tv.vis(...).numpy()` from outside TF will trigger the trace-and-execute pipeline transparently.

---

## 11. Notable Internals & Inconsistencies

These are real artefacts of the source, all citable:

1. **Return shape of `vis_snapshot` vs. `vis`**: `vis_snapshot` (`vis.py:165-169`) actually returns `(Nants, Nfreqs)` (the per-antenna voltage response), with the outer-product line commented out (167-168). But `vis_ptsrc_block` (line 94) *does* return the full `(Nants, Nants, Nfreqs)` baseline cube. So `vis_snapshot`'s `tf.reduce_sum(vis_blocks, axis=0)` (line 164) is summing baseline cubes across source-blocks, which is mathematically correct, and the function's actual output is therefore `(Nants, Nants, Nfreqs)` — *despite* the docstring stub at line 138 not specifying a return type and the variable name `v` suggesting voltage. Top-level `vis(...)` therefore returns `(Nlsts, Nants, Nants, Nfreqs)`. The mismatch between names/comments and behaviour is dangerous to anyone reading the source.

2. **Hard-coded HERA latitude**: `coords.HERA_LATITUDE = -0.5361917991288512` rad (line 12) is the default for `equatorial_to_topocentric` and `eq_to_az_za`. There is no site-config layer.

3. **Buggy `eq_to_az_za`**: Author flags `# FIXME: There is a bug in this code` at `coords.py:144`. Not on the live path; left in for legacy reasons.

4. **Stray `print()` calls in `equatorial_to_topocentric`** (`coords.py:57-58`) — these fire on every TF tracing and pollute stdout.

5. **`scipy` is declared but unused.** No `import scipy` anywhere on `main`.

6. **`astropy` and `six` are imported but undeclared** (see §3 caveat).

7. **`Nptsrc % nblocks` must be exact** (`vis.py:143-144`). For arbitrary catalogues the user must pad/trim or pick `nblocks=1`.

8. **Single beam duplicated**: FIXMEs at `vis.py:58` and `vis.py:66` admit the kernel only consumes `beams[0]` and tiles it.

9. **Frequency chunking missing**: FIXME at `vis.py:150` notes that frequency chunking would help memory more than source chunking.

10. **No `__all__`**: `from .vis import *` re-exports `numpy as np`, `tensorflow as tf`, etc., as well as `tvbeams`. Users picking up names from `import TensorVis as tv; tv.np` is real.

11. **Apache-2.0 vendored code**: `tensorflow_shape.py` carries the full TF Authors header; `tensorflow_interp.py` only has a docstring URL, not the explicit Apache-2.0 notice. This is a license-hygiene concern if the package is redistributed.

---

## 12. Known Limitations & Implicit TODOs

Compiled from FIXMEs in source plus the gaps in capability:

- Single shared primary beam across all antennas (`vis.py:58, 66`).
- Memory chunking is by source, not frequency (`vis.py:150`).
- `Nptsrc` must be divisible by `nblocks`.
- Reshape correctness in `interpolate_beam` is unverified per author (`beams.py:186-187`).
- `eq_to_az_za` is buggy (`coords.py:144`).
- No precession; user must pre-precess (`vis.py:196-197`).
- No polarisation, no Jones, no parallactic angle, no diffuse sky.
- No tests, no CI, no docs site, no CHANGELOG, no AUTHORS, no CITATION.
- Hard-coded paths in example scripts.
- `scipy` declared but unused.
- `astropy`/`six` imported but undeclared.
- No tags / releases on git; only a linear `main` branch with 13 commits.
- Stale branch `origin/updated_coords` — never merged.
- Stray `print()` debug output in `coords.equatorial_to_topocentric`.

---

## 13. Quick-start (recipe)

Combining what the example script and the API show:

```python
import numpy as np, tensorflow as tf
import TensorVis as tv
from pyuvdata import UVBeam

# Antennas
ants = tv.utils.build_hex_array(hex_spec=(3,4), d=14.6)
antpos = tf.convert_to_tensor(np.column_stack(list(ants.values())).T,
                              dtype=tf.float64)

# Times & freqs
lsts  = tf.constant(np.linspace(0., 0.01, 4),         dtype=tf.float64)
freqs = tf.constant(np.linspace(100e6, 120e6, 8),     dtype=tf.float64)

# Sources (RA/Dec in radians, flux at 100 MHz, alpha)
ra, dec, S, alpha = np.load("examples/cat_gleam_cut.npy")[:50, :].T
ra    = tf.constant(ra * np.pi/180,  dtype=tf.float64)
dec   = tf.constant(dec * np.pi/180, dtype=tf.float64)
S     = tf.constant(S,               dtype=tf.float64)
alpha = tf.constant(alpha,           dtype=tf.float64)

# Beam
uvb = UVBeam(); uvb.read_beamfits("HERA_NF_efield.beamfits")
uvb.interpolation_function = 'healpix_simple'
uvb.freq_interp_kind        = 'linear'
freq_range = (uvb.freq_array.min(), uvb.freq_array.max())
gr, gi = tv.beams.construct_beam_grid(uvb, Nza=100, Naz=101,
                                      freq=np.unique(uvb.freq_array),
                                      dtype=tf.float64)
beams = tf.expand_dims(tf.complex(gr, gi), axis=0)

# Run
V = tv.vis(antpos, lsts, freqs, ra, dec, S, alpha,
           beams=beams, freq_range=freq_range, nblocks=5)
print(V.shape)            # (4, 10, 10, 8)
np.save("vis.npy", V.numpy())
```

---

## 14. Reference: Function-call graph

```
tv.vis  (vis.py:172)
  └── tf.map_fn over LSTs
       └── vis_for_lst(lst)
            ├── coords.equatorial_to_topocentric  (coords.py:18)
            ├── coords.topocentric_to_az_za       (coords.py:64)
            └── vis_snapshot                       (vis.py:98)
                 └── tf.map_fn over nblocks
                      └── vis_ptsrc_block          (vis.py:18)
                           ├── tvbeams.interpolate_beam (beams.py:143)
                           │     ├── beams.coords_for_interp (beams.py:78)
                           │     │     └── utils.unit_interval (utils.py:7)
                           │     └── tensorflow_interp.interpolate3d (×2)
                           │           └── tensorflow_shape.check_static
                           └── coords.az_za_to_delay (coords.py:157)
```

---

## 15. Summary Box

| | |
|---|---|
| **Verdict** | Minimal proof-of-concept TF2 visibility kernel. Useful as a reference for trilinear-interpolated UVBeam evaluation and as a skeleton for a TF-based RIME, but not a complete simulator. |
| **Code size** | ~1.25 kLOC of package code (≈ 0.5 kLOC original + ~0.75 kLOC vendored TF Graphics). |
| **Install footprint** | Pure Python + TF + pyuvdata. |
| **GPU support** | Implicit through TensorFlow device placement; XLA via env var. |
| **Production-ready?** | No: no tests, scratch code in `coords.py`, single-beam limitation, scalar (no polarisation), no I/O standards. |
| **Best use inside RadioSim** | As a backend reference for how to express a RIME in pure TF graph mode; not as a drop-in physics engine. |
