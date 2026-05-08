# pyvisgen — Exhaustive Reference

A deep, top-to-bottom reference for the `pyvisgen` simulator vendored at
`simulators/pyvisgen/`. Written from a direct read of every source file in
`src/pyvisgen/`, the resource bundles under `resources/`, the default
configuration under `config/`, the documentation under `docs/`, the example
notebooks under `examples/`, the test suite under `tests/`, `pyproject.toml`,
`CHANGES.rst`, and the README/CITATION/license metadata.

`pyvisgen` describes itself as "a python implementation of the Radio
Interferometer Measurement Equation (RIME) formalism inspired by the VISGEN
tool of the MIT Array Performance Simulator (MAPS) developed at Haystack
Observatory." The package was developed at TU Dortmund University by the
radionets-project group (Kevin Schmitz, Felix Geyer, Stefan Fröse, Anno
Knierim, Tom Groß) and is published under the MIT License (Zenodo DOI
10.5281/zenodo.10091310). The current vendored release is **0.8.0
(2026-04-26)**, requires **Python ≥ 3.11**, and is distributed on PyPI as
`pyvisgen`.

The simulator's design centre is *machine-learning workflows*: every
component (image-domain RIME, GPU-friendly batched evaluation, HDF5/UVH5/
WebDataset/PyTorch writers, dataset converter, gridder plugin) is built so
that "input image → simulated visibility → gridded image" pairs can be
generated in bulk to train neural-network reconstructions in the
`radionets` and `pyvisgrid` ecosystems.

---

## 1. Project map

```
simulators/pyvisgen/
├── README.rst                  Project overview, badges, RIME equation
├── CITATION.cff                CFF metadata (Fröse, Schmitz, Knierim, Geyer, Groß)
├── LICENSE                     MIT
├── CHANGES.rst                 Towncrier-managed changelog (0.2.0 → 0.8.0)
├── pyproject.toml              Hatchling/hatch-vcs build, deps, ruff config
├── environment.yml             Conda runtime env
├── environment-dev.yml         Conda dev env
├── uv.lock                     uv lockfile (~600 kB)
├── .pre-commit-config.yaml     pre-commit hooks
├── .readthedocs.yaml           ReadTheDocs build config
├── config/
│   └── default_data_set.toml   Shipped default simulation TOML
├── resources/
│   ├── layouts/                10 array layout text files
│   │   ├── alma.txt, alma_dsharp.txt
│   │   ├── dsa2000W.txt, dsa2000_31b.txt
│   │   ├── eht.txt
│   │   ├── meerkat.txt, meerkat_test.txt
│   │   ├── vla.txt
│   │   └── vlba.txt, vlba_light.txt
│   └── noise_configs/          Per-telescope T_sys lookup tables
│       ├── meerkat.toml
│       └── vlba.toml
├── docs/                       Sphinx + MyST docs (User/Developer/API)
├── examples/                   Jupyter notebooks + test_model.h5
├── tests/                      pytest test tree (8 sub-suites)
└── src/pyvisgen/
    ├── __init__.py             rich-click theme + version
    ├── _plugin_manager.py      entry-point plugin discovery (gridding/ft)
    ├── version.py              version stub (hatch-vcs writes _version.py)
    ├── simulation/             Observation, scan, visibility, noise, array
    ├── layouts/                Stations dataclass + array readers
    ├── io/                     Pydantic config, data writers, converter
    ├── fits/                   AIPS UV FITS writer (vis/freq/ant/time HDUs)
    ├── dataset/                SimulateDataSet pipeline + sampling utils
    ├── tools/                  rich-click CLI (simulate / quickstart / convert)
    └── utils/                  data, logging, batch_size, carbon_tracking
```

The package ships layouts and noise configs through Hatch's
`shared-data` mechanism (`pyproject.toml` → `[tool.hatch.build.targets.wheel.shared-data]`) so
that `sysconfig.get_path("data")/share/resources/layouts/*.txt` and
`.../share/resources/noise_configs/*.toml` are populated on install. The
default TOML configuration is installed under `share/configs/`. Both layouts
and noise configs are read at runtime by helpers that resolve those
`sysconfig` paths (`pyvisgen.layouts.layouts.get_array_layout`,
`pyvisgen.simulation.noise._noise_config_dir`).

---

## 2. The RIME formulation in pyvisgen

The model implemented here is the *image-plane discrete RIME* with a
two-Jones chain on each side of the brightness matrix, written in terms of
direction cosines $(l, m)$:

$$
V_{pq}(u,v,w) \;=\; \sum_{l,m} \mathbf{E}_p(l,m)\,\mathbf{K}_p(l,m)\,
\mathbf{B}(l,m)\,\mathbf{K}_q^{H}(l,m)\,\mathbf{E}_q^{H}(l,m).
$$

* **B(l,m)** — 2×2 brightness matrix built from a Stokes-I image and an
  optional simulated polarization field. See §6 (`Polarization`).
* **K(l,m)** — Phase delay
  $\exp\!\left[-2\pi i (ul + vm + w(n-1))/c \cdot \nu\right]$, with
  $n=\sqrt{1-l^2-m^2}$. Implemented in `simulation/scan.py::calc_fourier`
  with separate kernels for the lower and upper edges of each spectral
  window (`spw_low`, `spw_high`), which are averaged later in
  `integrate()`.
* **E(l,m)** — Direction-dependent telescope response, modelled as a
  circular-aperture Airy/jinc beam:
  $\mathrm{jinc}(2\pi/\lambda \cdot d\cdot\theta_{lm})$ with
  $\theta_{lm}$ the angular distance to the phase centre and $d$ the dish
  diameter (taken from the layout via `torch.unique(obs.array.diam)`).
  Implemented in `simulation/scan.py::calc_beam`. Only applied when
  `corrupted=True`.
* **Feed rotation** — Parallactic-angle Jones, applied either after
  (`ft="default"`) or before (`ft="reversed"`) the K kernels. Two
  variants: linear feed (real rotation matrix) and circular feed
  (diagonal $\mathrm{diag}(e^{iq}, e^{-iq})$). Implemented in
  `simulation/scan.py::calc_feed_rotation`.

Three Fourier-transform back-ends are available, selected via
`fft.ft = "default" | "reversed" | "finufft"`:

* `default` — direct DFT kernel evaluated per-baseline via
  `torch.exp(-2πi·(ul+vm+wn)/c · ν)` with
  `(B,L,2,2)` broadcasting; the visibility is the pixel-summed product.
  Polarization (feed rotation) and beam are applied between K and the sum.
* `reversed` — same RIME but applied right-to-left in scan order: feed
  rotation and beam first, then K. The image is broadcast to all
  baselines via `torch.repeat_interleave`.
* `finufft` — GPU-only path that calls
  `radioft.finufft.CupyFinufft` (Flatiron NUFFT, cuFINUFFT under the
  hood) for the Fourier transform. Beam is applied in the image domain
  before the NUFFT; feed rotation is *not* applied in this path. Errors
  out with `_FINUFFT_ERROR` if `radioft` is missing or CUDA is
  unavailable.

The trailing factor `w · (n-1)` makes K a *full 3-D* phase term so the
simulator handles non-coplanar baselines (full-sky w-projection-free
DFT).

The `RIMEScan` class in `simulation/scan.py` is the central evaluator: it
caches `lm`, `rd`, `ant_diam`, `polarization`, `corrupted`, and the
optional `CupyFinufft` instance; `__call__(img, bas, spw_low, spw_high)`
dispatches to the chosen `ft_func` under `torch.no_grad()`.

The integrator (`integrate()` in `simulation/scan.py`) sums over (l,m)
and then takes a 0.5-weighted average of the two spectral-window edges,
producing one complex 2×2 matrix per baseline per frequency channel. Both
`integrate`, `jinc`, and `angular_distance` are decorated with
`@torch.compile` for kernel fusion.

A separate normalization step in `vis_loop` multiplies B by 0.5 if
`normalize=True` (default), making the Stokes-I image normalize to 1 in
the final visibilities (the same 1/2 factor used in standard
RIME conventions; see RadioSim CLAUDE.md note).

---

## 3. Coordinates, time, and baselines

### 3.1 Source and observation geometry

The user-facing `Observation` class (`simulation/observation.py`) takes
right-ascension/declination (degrees), a `datetime` start time, scan
durations, a number of scans, scan separations, integration times,
spectral-window centre/offsets/bandwidths, FOV (arcseconds), image size,
array layout name, `corrupted` flag, torch device, `dense` flag,
`sensitivity_cut`, `polarization`, `pol_kwargs`, and `field_kwargs`.

Internally, all scalars are promoted to `torch.float64` (set globally via
`torch.set_default_dtype(torch.float64)` at module import). Frequencies
are stored as `ref_frequency`, `frequency_offsets`, `bandwidths` tensors,
and the lower/upper window edges as

```
waves_low  = (ref_frequency + frequency_offsets) - bandwidths / 2
waves_high = (ref_frequency + frequency_offsets) + bandwidths / 2
```

Note: despite the variable name `waves_*`, these are stored in **Hertz**
(not wavelength); they later combine with `c` in the K-kernel to become
the proper exponent.

### 3.2 Scan timing

`Observation.create_scans()` builds a list of `Scan` dataclasses (start,
stop, separation, integration_time, all promoted to
`astropy.units.second` Quantities). Both `scan_duration`, `scan_separation`,
and `integration_time` accept either a scalar (broadcast across all
scans) or an array-like of length `num_scans` (or `num_scans - 1` for
`scan_separation`). The last `scan_separation` is forced to zero. For
each scan, `Scan.get_timesteps()` produces an `astropy.time.Time` array
of evenly spaced samples; the last sample is clamped to `stop` so the
final integration window may be short.

### 3.3 Baselines & UV coverage

`Array` (`simulation/array.py`) wraps a `Stations` dataclass and exposes
two `lazyproperty` cached views:

* `relative_pos` — `(delta_x, delta_y, delta_z)` reshaped to `(N_bl, 1)`,
  built via `torch.combinations` on antenna positions (so all unordered
  i<j pairs).
* `antenna_pairs` — `(st_num_pairs, els_low_pairs, els_high_pairs)`
  combinations of station IDs and elevation limits. There are
  `N_bl = N_ant·(N_ant−1)/2` baselines.

`Observation.calc_ref_elev()` computes, per scan time step, three
quantities:

1. **Greenwich Hour Angle** of the source: `time.sidereal_time("apparent",
   "greenwich") − src.ra.to(hourangle)`.
2. **Local hour angle per antenna**, by computing each antenna's
   sidereal time via `Time(time, location=loc).sidereal_time("mean")`
   and subtracting the source RA. This is needed by
   `calc_feed_rotation` (Eq. 13.1 of Meeus).
3. **Per-antenna elevation** at each time step, via
   `SkyCoord.transform_to(AltAz)` with each antenna as a separate
   `EarthLocation` (broadcasted with `torch.repeat_interleave`).

`calc_direction_cosines(ha, el_st, dx, dy, dz)` then builds (u, v, w)
from the (rotated) baseline vectors using the textbook equations
(Thompson, Moran & Swenson §4.1):

```
u =  sin(ha)·dx + cos(ha)·dy
v = -sin(δ)·cos(ha)·dx + sin(δ)·sin(ha)·dy + cos(δ)·dz
w =  cos(δ)·cos(ha)·dx - cos(δ)·sin(ha)·dy + sin(δ)·dz
```

Per time-step, baselines are flagged `valid` only when both antennas
have elevation inside their `[el_low, el_high]` brackets (default
15°–85° for most layouts).

`Observation.calc_feed_rotation(ha)` implements
$q = \mathrm{atan2}(\sin h,\, \tan\varphi\cos\delta - \sin\delta\cos h)$
(Meeus Eq. 13.1) per antenna and time step, then forms all
$N_\mathrm{bl}\times 2$ pair combinations.

The resulting per-time-step records are stored in a `Baselines`
dataclass with eleven fields (`st1, st2, u, v, w, valid, time, q1, q2,
el1, el2`). `Observation.calc_baselines()` calls
`get_baselines(scan.get_timesteps())` once per scan, concatenating into
the global `Baselines` object via `add_baseline`.

### 3.4 Valid baseline subsets and integration windows

`Baselines.get_valid_subset(num_baselines, device)` is the
**integration-window builder**: the per-time-step records are reshaped
into `(n_times, n_bl)`, and a *consecutive-pair* mask
`valid[:-1] & valid[1:]` selects baselines that are visible at both ends
of an integration interval. For each surviving pair, start/stop/midpoint
values of `u, v, w, q1, q2, el1, el2` are produced, the time is taken to
be the midpoint MJD, and an MS-style 256-encoded baseline ID
`256·(st1+1) + st2+1` is generated. Output is a
`ValidBaselineSubset` with 24 fields — three triples
(start/stop/valid) for u/v/w/q1/q2/el1/el2, plus `baseline_nums`,
`date`, `st_id_pairs`. `__getitem__` indexes through every field
simultaneously.

`ValidBaselineSubset.get_timerange(t_start, t_stop)` filters by date.

`ValidBaselineSubset.get_unique_grid(fov, ref_freq, img_size, device)`
buckets the (u_valid, v_valid) midpoints onto a uniform grid with
spacing `Δuv = c/(fov_rad · ref_freq)`, lexsorts the bucket indices, and
keeps only the **first** baseline per occupied cell. This implements the
"grid" sampling mode used to produce one visibility per Fourier-domain
pixel, which is what the radionets gridded learning targets need.

### 3.5 Dense (ideal) interferometer

`Observation.calc_dense_baselines()` (and
`mode="dense"` in `vis_loop`) produces a perfect Cartesian uv grid of
size `img_size × img_size` with spacing `c/(fov_rad · ref_freq)`. The
RA/Dec are forced to (0, 0) in this branch and `waves_low/high` collapse
to a single frequency (`ref_frequency`). Feed rotation and per-time
quantities are zeroed, so this mode disables polarization and is
strictly meant for ideal-instrument benchmarking. The dense path is
**GPU-only** (the user-facing check raises if `device == cpu`).

### 3.6 Image-plane grids

`Observation.create_rd_grid()` builds the (RA, Dec) grid of every pixel
relative to the source position by spacing pixels at `Δrd = fov_rad /
img_size` and stacking with `torch.meshgrid(..., indexing="xy")`. All
arithmetic is done in `np.float128` and cast back to `np.float64` to
avoid spacing drift on large images (the docstring in
`docs/user-guide/examples_tutorials/ideal_interferometer.md` discusses
this in detail; PyTorch can't handle float128).

`create_lm_grid()` then converts (RA, Dec) to direction cosines:
```
l = cos(dec)·sin(ra)
m = sin(dec)·cos(δ_src) − cos(dec)·sin(δ_src)·cos(ra)
```
where `δ_src` is the phase-centre declination. The result is shape
`(H, W, 2)` and lives on `obs.device`.

The `lm` and `rd` grids are masked by the **sensitivity cut**: only
pixels with `SI > sensitivity_cut` participate in the
visibility sum, so the brightness matrix `B` ends up flattened to
`(N_pix_above_cut, 2, 2)`.

---

## 4. Antenna layouts

### 4.1 The Stations dataclass

`pyvisgen.layouts.Stations` is a 9-field dataclass:
```
st_num, x, y, z, diam, el_low, el_high, sefd, altitude
```
All fields are stored as `torch.Tensor`. `Stations.__getitem__(i)`
returns a fresh `Stations` containing only the rows at index `i`.

### 4.2 Built-in arrays

`get_array_layout(name)` reads a whitespace-delimited text file from
`<sysconfig data>/share/resources/layouts/<name>.txt`. The first column
is `station_name` (dropped), followed by `X Y Z dish_dia el_low el_high
SEFD altitude`. A station-index column `st_num` is prepended on the
fly. Special cases:

* `vla` — file holds *relative* positions, so the loader adds
  `EarthLocation.of_site("VLA")` to convert to ITRF geocentric.

The shipped layouts (`get_array_names()`) are:
`alma`, `alma_dsharp`, `dsa2000W`, `dsa2000_31b`, `eht`,
`meerkat`, `meerkat_test`, `vla`, `vlba`, `vlba_light`. The `*_test` /
`*_light` variants are reduced for fast CI.

### 4.3 Custom layouts

`get_array_layout` also accepts a `pandas.DataFrame` (with the same
columns) or a `pathlib.Path` to a custom whitespace-delimited file.
Setting `writer=True` returns the raw DataFrame instead of the dataclass
(used by the FITS writer to build the AIPS AN HDU).

---

## 5. Noise model

Noise is added per-batch in `_batch_loop` after the integration step.
Two modes are supported via `noise_mode`:

### 5.1 SEFD mode (`mode="sefd"`)

Uniform per-baseline SEFD. The noise standard deviation is
$\sigma = (1/\eta)\cdot \mathrm{SEFD}/\sqrt{2 \tau_\mathrm{int}\,\Delta\nu}$
with $\eta=0.93$ hard-coded in `compute_noise_std`,
`τ_int = obs.int_time`, `Δν = obs.bandwidths[0]`.
Real and imaginary parts get independent Gaussians with std σ. Natural
weights $w = 1/\sigma^2$ are stored alongside.

### 5.2 T_sys / elevation mode (`mode="tsys"`)

Elevation-dependent system temperature read from per-telescope TOML
files in `resources/noise_configs/`. Each file declares
`dish_diameter` (metres) and one or more `[bands.<name>]` sections
containing `frequency_mhz`, `el_knots`, and three temperature lookup
tables `t_atm`, `t_spill_h`, `t_spill_v`.
`elevation_tsys_contribution(el_deg)` linearly interpolates
$T_\mathrm{atm}(el) + T_\mathrm{spill}(el)$ with a clamping interpolator
(`_interp1d`). `sefd_from_elevation(el1, el2, T_sys/η_ref)` uses
$\mathrm{SEFD} = 2 k_B T / A_\mathrm{geom} \cdot 10^{26}$ Jy
(geometric aperture, no efficiency) and **adds** the elevation-dependent
ΔT to the reference value at $55^\circ$, giving a baseline SEFD of
$\sqrt{\mathrm{SEFD}_1\cdot \mathrm{SEFD}_2}$.

The shipped configs cover MeerKAT (L-band 1284 MHz, dish 13.5 m) and
VLBA (X-band 15.363 GHz, dish 25 m). MeerKAT values are digitized from
the SKA-Africa `ESDKB` figure 6; VLBA values are derived from the
2026B Observational Status Summary with $T_\mathrm{phys}=265$ K and
$\tau_z=0.05$. `available_telescopes()` lists installed configs;
unknown names emit a `UserWarning` and then raise `ValueError`. If
`band` is `None`, the *first* band in the TOML is used.

### 5.3 Where noise enters the loop

`vis_loop` collects `noise_level`, `noise_mode`, `telescope`, `band`
from the user; `_batch_loop` then calls
`generate_noise(int_values.shape, obs, noise_level, mode, el1_deg, el2_deg, telescope, band)`
*after* computing the clean visibilities, adds the result, and stores
the per-baseline natural weights inside `Visibilities.weights`. If
`noise_level == 0`, weights default to `torch.ones(...)`.

---

## 6. Polarization

`Polarization` (`simulation/visibility.py`) computes the 2×2 brightness
matrix `B(l,m)` from a Stokes-I image and a configurable polarization
sub-model. Three modes:

* `"linear"` — Linear feed:
  $I = A_X^2 + A_Y^2$,
  $Q = A_X^2 - A_Y^2$,
  $U = 2 A_X A_Y \cos\delta_{XY}$,
  $V = -2 A_X A_Y \sin\delta_{XY}$,
  with `B = [[I+Q, U+iV], [U-iV, I-Q]]`.
* `"circular"` — Circular feed: same Stokes parametrisation (different
  trig identity), with `B = [[I+V, Q+iU], [Q-iU, I-V]]`.
* `None` (or `"none"`) — no polarization simulated:
  `Q,U,V = 0`, `B = diag(I, I)`.

When polarization is on, the input intensity is split between the two
feeds via `amp_ratio` (`A_X^2 / I`, default 0.5; if `None`, drawn from
`U(0,1)`). The phase difference $\delta_{XY}$ is the `delta` parameter
(default 45°). Q, U, V maps are then **multiplied pointwise** by a
random "polarization field" that simulates spatially varying
polarization fraction across the source.

`Polarization.rand_polarization_field(shape, order, scale, threshold,
random_state)` builds that field by drawing white noise of size `shape`,
smoothing with `scipy.ndimage.gaussian_filter` at $\sigma =
\bar{N}/(40\,\mathrm{order})$, then *uniformizing* the histogram via a
double-`argsort` rank trick that maps the smoothed field onto a linspace
between `scale[0]` and `scale[1]`. The result is a controllable
fluctuation pattern from large smooth lobes (order < 1) to fine speckle
(order > 1). `threshold` optionally clips the values.

`Polarization.dop()` computes per-pixel **degree of linear and circular
polarization**:
`lin_dop = sqrt(Q² + U²)/I`, `circ_dop = |V|/I`. These are stored on
`Visibilities.linear_dop` and `Visibilities.circular_dop` as image-shaped
tensors (one map per simulation, not per visibility).

The brightness matrix is masked by the sensitivity cut before being
flattened to `(N_px, 2, 2)`, then optionally scaled by 0.5
(`vis_loop(normalize=True)`).

---

## 7. The visibility loop

`pyvisgen.simulation.visibility.vis_loop(obs, SI, …)` is the main entry
point and returns a populated `Visibilities` dataclass. Outline:

1. Set `torch.set_num_threads(num_threads)` (default 10) and silence
   `torch._dynamo.config.suppress_errors`.
2. Build `Polarization(SI, …)` and obtain `B, mask, lin_dop, circ_dop`.
3. Index `lm`, `rd` with the sensitivity mask.
4. Optionally scale `B *= 0.5`.
5. Construct an empty `Visibilities` dataclass with one column per
   spectral window (length `len(obs.waves_low)`).
6. Build the baseline iterator by mode:
   * `"full"` → `obs.baselines.get_valid_subset(...)`
   * `"grid"` → same, then `.get_unique_grid(...)`
   * `"dense"` → `obs.calc_dense_baselines()` (GPU only)
7. Call `adaptive_batch_size(_batch_loop, batch_size, …)`. With
   `batch_size="auto"`, the initial guess equals
   `bas.baseline_nums.shape[0]` — i.e. one batch covering everything;
   on `OutOfMemoryError`, the wrapper halves the batch size, frees CUDA
   memory, and retries until `MIN_BATCH_SIZE=1`.
8. Inside `_batch_loop`, instantiate `RIMEScan(ft, mode, obs, lm, rd)`
   once and iterate over `bas` slices; for each slice, evaluate the
   RIME for each `(spw_low, spw_high)` pair via `rime(B, bas_p, …)`.
9. Stack frequencies along the channel axis, swap to
   `(n_bl, n_freq, 2, 2)`, drop rows containing NaNs.
10. If `noise_level != 0`, generate complex Gaussian noise and natural
    weights via `generate_noise(...)`; otherwise weights default to
    `ones`.
11. Append the batch result to the running `Visibilities` object via
    `Visibilities.add(...)`.

The output `Visibilities` dataclass has 14 fields:
`V_11, V_22, V_12, V_21` (each `(n_bl, n_freq)` complex),
`weights` (`(n_bl, n_freq)` real),
`num` (1-based visibility count for FITS),
`base_num` (256-encoded ID),
`u, v, w` (mid-integration metres),
`date` (mid-integration JD),
`st_id_pairs` (`(n_bl, 2)` int),
and the per-pixel `linear_dop` / `circular_dop` maps.
`Visibilities.get_values()` stacks the four polarization products into
`(n_bl, n_freq, 4)`.

---

## 8. Simulation modes & precision

* **`mode="full"`** (default) — every `valid` baseline interval is
  simulated.
* **`mode="grid"`** — one visibility per uv-grid pixel (first hit
  wins), used to build neural-network training pairs that match the
  Fourier representation of the input image directly.
* **`mode="dense"`** — full Cartesian uv grid generated from FOV/freq;
  ignores feed rotation; GPU-only.

Precision is fixed at **float64 / complex128** end-to-end via the
module-level `torch.set_default_dtype(torch.float64)` in
`observation.py`, `visibility.py`, and `scan.py`. The brightness matrix
is `torch.cdouble`. The image-domain RA/Dec arithmetic uses `np.float128`
internally to avoid catastrophic cancellation, then casts back to
`np.float64` (PyTorch can't carry 128-bit floats).

Devices: `obs.device` is the master torch device. All inputs must be
moved to it before `vis_loop`. `dense` and `finufft` modes are GPU-only.

---

## 9. Data writers

All writers live in `pyvisgen.io.datawriters`, all extend the abstract
`DataWriter` base class, all support the context-manager protocol, and
all use `test_shapes(array, name)` to enforce a `(B, 2, H, W)` shape
convention plus `get_half_image(x, y, overlap)` to crop training pairs
to half height with a small overlap (default 5 px).

* **`H5Writer`** — Writes one `samp_<dataset_type>_<idx>.h5` per
  bundle, with two datasets `x` and `y`. Disables HDF5 file locking.
* **`FITSWriter`** — Writes one AIPS-style UV FITS file per
  visibility set. Uses `pyvisgen.fits.create_hdu_list` (see §10).
* **`UVH5Writer`** — Writes one `<dataset_type>_<idx>.uvh5` per
  visibility set with grouped datasets:
  `visibilities/{V_11,V_22,V_12,V_21,weights}`,
  `uvw/{u,v,w,st_id_pairs}`,
  `lmn/{l,m,n}` (n recomputed via $\sqrt{1-l^2-m^2}$),
  `frequency_bands` (centre frequencies per IF), and
  `sky/SI` (the input image cube).
* **`PTWriter`** — Saves PyTorch `.pt` files with sparse complex tensors
  `SIM = x[:,0]+1j·x[:,1]` and dense `TRUTH = y[:,0]+1j·y[:,1]`, plus a
  `TYPE` key (`"amp_phase"` or `"real_imag"`).
* **`WDSShardWriter`** — Optional-extra (`pyvisgen[webdataset]`) WebDataset
  writer producing `.tar(.gz)` shards with `input.npy`/`target.npy` per
  sample plus a sidecar `.parquet` containing
  `total_samples_in_dataset`, `samples_in_shard`, `shard_idx`,
  `bundle_id`, `data_type`. Compression is automatic if `compress=True`
  (renames `.tar` → `.tar.gz`).

Writers are selected from the TOML config via
`DataWriterConfig.writer` (a string is mapped through case-insensitive
shorthands `h5/hdf5`, `uvh5`, `wds/webdataset`, `pt`).

### 9.1 Optional secondary FITS output

`BundleConfig.fits_out_path` enables a *secondary* UVFITS dump alongside
a primary UVH5 output (used by the test WSClean pipeline). The model
validator `Config.check_fits_out_path_writer` rejects the combination
`writer=FITSWriter` + `fits_out_path` (redundant).

### 9.2 The data converter

`pyvisgen.io.dataconverter.DataConverter` provides one-shot conversion
between the three serialization formats:

```python
DataConverter.from_h5(dir, dataset_split="all").to(out, output_format="wds", compress=True)
DataConverter.from_wds(dir).to(out, output_format="h5")
DataConverter.from_pt(dir).to(out, output_format="wds", bundle_size=100)
```

It also supports re-encoding amplitude/phase ↔ real/imag via
`convert_representation=True`, which uses `DataTypeConverter` to apply
`hypot/atan2` or `cos/sin` element-wise. Dataset splits accepted:
`train`, `valid`, `test`, or `all`. `from_pt` batches by `bundle_size`
files. Same-format conversion without `convert_representation` raises
`RuntimeError`.

---

## 10. FITS / AIPS UV writer

`pyvisgen.fits.writer.create_hdu_list(data, obs)` returns an
`astropy.io.fits.HDUList` containing four HDUs:

1. **Primary `GroupsHDU` (AIPS UV)** — built by `create_vis_hdu`. The
   data axis is stacked as
   `[real, imag, weight] × n_pol × n_freq × n_IF=1`. Coordinate axes
   carry `COMPLEX, STOKES, FREQ, IF, RA, DEC` (`naxis=7`). Stokes IDs
   follow AIPS Memo 114: `-1..-4` for `RR/LL/RL/LR`, `-5..-8` for
   `XX/YY/XY/YX` (auto-selected by `obs.polarization`). `OBJECT` defaults
   to `sim-source-0`. Group parameters are
   `UU/VV/WW` (in light-seconds, `u/c, v/c, w/c`), full Julian date
   `DATE` plus fractional `_DATE`, `BASELINE` (256-encoded), and
   `FREQSEL`.
2. **`AIPS FQ`** — frequency setup (`FRQSEL`, `IF FREQ`, `CH WIDTH`,
   `TOTAL BANDWIDTH`, `SIDEBAND`).
3. **`AIPS AN`** — antenna table from the layout: positions, polarization
   types `X/Y` with PA `-90°`, dish diameters, plus header keys
   `GSTIA0`, `DEGPDY`, IERS-derived `POLARX`/`POLARY`/`UT1UTC`,
   `ARRNAM = obs.layout`, frame `ITRF`, etc.
4. **`AIPS NX` (time)** — central time of the integration interval,
   total interval length, source/subarray/freq IDs (all 1), and
   `START VIS`/`END VIS` indices.

The writer pulls Earth-orientation parameters from
`astropy.utils.iers.IERS_B`, which performs a network read on first
invocation unless cached.

---

## 11. The dataset pipeline (`pyvisgen.dataset`)

`SimulateDataSet.from_config(cfg, ...)` is the high-level
"image folder → visibility dataset" entry point:

1. **Load config**: accepts a path, dict, or already-parsed
   `pyvisgen.io.Config`. The TOML schema is the union of `SamplingConfig`,
   `NoiseConfig`, `PolarizationConfig`, `BundleConfig`,
   `DataWriterConfig`, `GriddingConfig`, `FFTConfig`, and an optional
   `CodeCarbonEmissionTrackerConfig`.
2. **Discover input bundles**: `utils.data.load_bundles(in_path,
   dataset_type)` globs `*<type>*.h5` files (naturally sorted), and
   `open_bundles(path, key)` reads `f[key]` (default `y`). Each bundle
   is a 3- or 4-D image stack.
3. **Resolve gridder plugin**: `PluginManager.get_gridder(name)` looks
   up entry points under the `pyvisgen.gridding` group; default name
   `pyvisgrid.gridder` (from the sister project `pyvisgrid`). On
   missing plugin, falls back to importing
   `pyvisgrid.core.gridder.Gridder` directly with a logged warning.
   Same machinery serves `pyvisgen.ft` plugins via `get_ft`.
4. **Count images / set total**: optional pass that opens every bundle
   and sums `len(SIs)` (rich progress bar). Skipped when `num_images`
   is supplied. Empty datasets raise immediately.
5. **Sample random parameters**: `create_sampling_rc(N)` draws, per
   image, `src_ra` (uniform in `fov_center_ra`), `src_dec`,
   `start_time` (uniform hourly draws between `scan_start[0]` and
   `scan_start[1]`), `scan_duration`, `num_scans`, plus polarization
   params (`delta`, `amp_ratio`, `field_order`, `field_scale`).
6. **Pre-test parameters**: `test_rand_opts(i)` checks that the source
   is visible to ≥ 50% of the antennas for ≥ 50% of the observation
   time using a self-implemented sidereal-time and altitude
   calculation (Meeus quadratic-cubic GST polynomial,
   `_compute_altitude` via spherical trigonometry). Failing draws are
   redrawn until the visibility threshold is met. Done in parallel via
   `joblib.Parallel(n_jobs=multiprocess, backend='threading')`.
7. **Run** (`_run`): for each bundle, build an `Observation` per image
   and call `vis_loop(...)`. If `grid=True`, the gridder is invoked
   with `from_pyvisgen(...)` to produce real/imag visibility masks;
   otherwise the raw `Visibilities` go to the configured writer.
   Optional secondary FITS output runs alongside the UVH5 path.
8. **Carbon tracking** (optional): `pyvisgen.utils.carbon_tracking.carbontracker`
   is a context manager that wraps everything in a CodeCarbon
   `OfflineEmissionsTracker` if the extra is installed and enabled in
   config. The tracker writes emissions to
   `codecarbon.output_dir`.
9. **SLURM mode** (`slurm=True`): `_run_slurm` indexes one image via
   `slurm_job_id + slurm_n*500`, simulates the single visibility, and
   writes one record. Used by HPC dispatchers; not in CI.

`SimulateDataSet` also keeps a multi-progress display via
`pyvisgen.simulation.utils.create_progress_tracker()`, which builds a
nested rich `Group` of five progress bars (`overall`, `counting`,
`testing`, `bundles`, `current_bundle`) inside a `rich.panel.Panel`.

---

## 12. Configuration schema

`pyvisgen.io.config.Config` is the root Pydantic v2 model and reads from
TOML via `Config.from_toml(path)`. Sections:

### 12.1 `[sampling]` — `SamplingConfig`
| key | type | default | notes |
|---|---|---|---|
| `mode` | `"full"`/`"grid"`/`"dense"` | `"full"` | passed to `vis_loop` |
| `device` | str | `"cuda"` | torch device |
| `seed` | int/bool/str/None | `1337` | `"none"`/`False` → None |
| `layout` | str | `"vlba"` | validated against `get_array_names()` |
| `img_size` | int > 0 | 1024 | |
| `fov_center_ra` | list[float] | `[100, 110]` | uniform sampling range, deg |
| `fov_center_dec` | list[float] | `[30, 40]` | deg |
| `fov_size` | float > 0 | 0.24 | arcsec |
| `corr_int_time` | float > 0 | 30.0 | s |
| `scan_start` | list[str] (len 2) | dates 1995–2025 | `dd-mm-YYYY HH:MM:SS` |
| `scan_duration` | list[int] | `[20, 600]` | uniform draw range, s |
| `num_scans` | list[int] | `[6, 10]` | uniform draw range |
| `scan_separation` | float ≥ 0 | 360 | s |
| `ref_frequency` | float > 0 | 15.176 GHz | Hz |
| `frequency_offsets` | list[float] | 4 values | Hz |
| `bandwidths` | list[float] | 4 values | Hz |
| `normalize` | bool | True | `B *= 0.5` |
| `corrupted` | bool | False | apply E (beam) Jones |
| `sensitivity_cut` | float ≥ 0 | 1e-6 | image mask |

### 12.2 `[noise]` — `NoiseConfig`
| key | type | default |
|---|---|---|
| `noise_level` | float ≥ 0 | 0 |
| `noise_mode` | `"sefd"`/`"tsys"` | `"sefd"` |
| `telescope` | str | `"meerkat"` |
| `band` | str/None | None |

### 12.3 `[polarization]` — `PolarizationConfig`
| key | type | default | notes |
|---|---|---|---|
| `mode` | `"linear"`/`"circular"`/`None` | None | `"none"` → None pre-validator |
| `delta` | float | 45 | degrees |
| `amp_ratio` | float ∈ [0,1] | 0.5 | |
| `field_order` | list[float] | `[0.01, 0.01]` | |
| `field_scale` | list[float] | `[0, 1]` | |
| `field_threshold` | float | None | |

### 12.4 `[bundle]` — `BundleConfig`
| key | default | notes |
|---|---|---|
| `dataset_type` | `"train"` | one of `train/test/valid/none/""` |
| `in_path` | path | source HDF5 directory |
| `out_path` | path | written-to directory |
| `fits_out_path` | None | secondary UVFITS dir; UVH5Writer-only |
| `overlap` | 5 | half-image overlap |
| `grid_size` | 1024 | |
| `grid_fov` | 0.24 | |
| `amp_phase` | False | metadata flag |

### 12.5 `[datawriter]` — `DataWriterConfig`
| key | default | notes |
|---|---|---|
| `writer` | `H5Writer` | shorthands `h5/uvh5/wds/pt` |
| `overlap` | 5 | |
| `shard_pattern` | `"%06d.tar"` | wds only |
| `compress` | False | wds only |

### 12.6 `[gridding]` / `[fft]` / `[codecarbon]`
* `gridding.gridder` (default `"default"` → resolves to `pyvisgrid.gridder`).
* `fft.ft` (`"default"`, `"finufft"`, `"reversed"`).
* `codecarbon` accepts `False`, `True`, or a dict mapping to
  `CodeCarbonEmissionTrackerConfig(log_level, country_iso_code,
  output_dir)`. `True` instantiates with project name `pyvisgen`.

The shipped default config (`config/default_data_set.toml`) is shown in
§13.

---

## 13. CLI tools (`pyvisgen.tools`)

The single entry point is `pyvisgen` (from
`tools/cli.py`, registered as `[project.scripts]`). It is a `rich-click`
group with three sub-commands:

* **`pyvisgen quickstart <path>`** — copies the shipped TOML default
  (`config/default_data_set.toml`) to the target path; prompts on
  overwrite unless `-y/--yes`. Resolves the resource via `sysconfig`.

* **`pyvisgen simulate <config.toml>`** — runs `SimulateDataSet.from_config`.
  Options: `--mode {simulate, gridding, slurm}` (default `simulate`,
  toggles `grid=True/False` and `slurm=True/False`), `-k/--key`
  (HDF5 key, default `y`), `--slurm_job_id`, `--slurm_n`, `--num_images`,
  `-p/--multiprocess`, `-s/--stokes` (default `I`). All wrapped in the
  `carbontracker(config)` context manager so emissions tracking happens
  if enabled.

* **`pyvisgen convert <input_dir>`** — runs `DataConverter`. Options:
  `-o/--output-dir`, `--input-format {h5,wds,pt}`, `--output-format`,
  `-t/--dataset_split`, `--amp-phase`, `--shard-pattern`, `--compress`,
  `--bundle-size`. Refuses identical input/output formats unless
  `convert_representation=True`.

The legacy console scripts `pyvisgen-simulate` / `pyvisgen-quickstart`
have been removed; everything goes through the unified `pyvisgen`
group as of 0.7.0 (see CHANGES.rst).

---

## 14. Plugin system

`pyvisgen._plugin_manager.PluginManager` discovers two entry-point
groups via `importlib.metadata.entry_points`:

* **`pyvisgen.gridding`** — gridder plugins. The default-shipped
  `pyvisgrid.gridder` (from sister package `pyvisgrid`) is selected when
  `gridding.gridder == "default"`. Custom plugins register an entry
  point pointing at a callable/class that exposes
  `from_pyvisgen(vis_data, obs, img_size, fov, stokes_components,
  polarizations).grid().get_mask_real_imag()`.
* **`pyvisgen.ft`** — Fourier-transform plugins. Currently used only via
  `PluginManager.get_ft(name)` (no first-party plugins inside pyvisgen
  itself; intended for swap-in NUFFT implementations).

If no plugins are installed, `_get_plugin` raises a guided
`ValueError` ("…install pyvisgrid (uv pip install pyvisgrid)…").
Failures during entry-point loading log a warning but do not abort.

The simulation also gracefully degrades when `radioft` is missing:
`scan.py` catches the import, sets `_FINUFFT_AVAIL = False`, and
re-raises `RuntimeError(_FINUFFT_ERROR)` only if `ft="finufft"` is
selected.

---

## 15. Optional extras and runtime dependencies

`pyproject.toml` declares the core dependencies:
`astropy ≥ 7.2.0`, `click`, `h5py`, `joblib`, `natsort`, `numpy`,
`pandas`, `pydantic ≥ 2.12.3`, `pyvisgrid` (sister gridder),
`rich`, `rich-click ≥ 1.8.9`, `scipy`, `toml`, `torch`, `tqdm`,
`radioft`.

Optional extras:
* `[plot]` → `matplotlib`
* `[codecarbon]` → `codecarbon`
* `[tutorials]` → `matplotlib`, `pytorch_finufft`
* `[webdataset]` → `webdataset`, `pyarrow`
* `[all]` → all of the above

The `dev` group (uv `[dependency-groups]`) adds `ipython`, `jupyter`,
`pre-commit`, `pytorch_finufft`, plus the `docs` and `tests` groups.

`environment.yml` and `environment-dev.yml` provide conda-friendly
versions of the runtime/dev environments.

---

## 16. Testing

`tests/` mirrors the package layout:
```
tests/
├── conftest.py          shared fixtures
├── data/                test_inputs.h5, test_layout.txt
├── dataset/             test_dataset.py, test_dataset_utils.py
├── fits/                test_fits_writer.py
├── io/                  test_config.py, test_dataconverter.py, test_datawriters.py
├── layouts/             test_layouts.py
├── simulation/          test_array, test_noise, test_observation, test_scan, test_simulation_utils, test_visibility
├── tools/               test_cli, test_converter_cli, test_create_dataset_cli
├── utils/               test_batch_size, test_carbon_tracking, test_data
├── test_plugin_manager.py
└── test_conf.toml       test config (vla layout, 128-px image, 4 IFs)
```

The CI test config (`test_conf.toml`) uses the `vla` layout, an `img_size`
of 128, FOV 0.0064″, four spectral windows centred on 15.21 GHz, and CPU
only — chosen so the pipeline runs in a few seconds without GPU. Coverage
is reported via `pytest-cov`/Codecov; uncovered branches like
`_run_slurm`, `apply_finufft`, and `calc_dense_baselines` are explicitly
marked `# pragma: no cover` because they require GPU/SLURM. Coverage
report excludes `def __repr__`, `raise NotImplementedError`,
`if TYPE_CHECKING:`, the `if not _WDS_AVAIL:` block, and any
`try/except ImportError` guard (per `[tool.coverage.report].exclude_also`).

---

## 17. Examples and notebooks

`examples/` holds:

* **`01_layouts.ipynb`** — reads & visualises layouts from
  `pyvisgen.layouts`.
* **`02_times.ipynb`** — `Scan` dataclass + time-step generation.
* **`baselines.ipynb`** — uv-coverage exploration.
* **`fits_tests.ipynb`** — round-trip through `pyvisgen.fits.writer`.
* **`visibility.ipynb`** — small end-to-end RIME run.
* **`ideal_interferometer.ipynb`** — companion to the user-guide page
  walking through dense (u,v) coverage with `pytorch_finufft`.
* **`simulation_chain.ipynb`** / **`0X_simulation_chain.ipynb`** — full
  dataset-creation walk-through.
* `test_model.h5`, `celestial-03-05.fits`, `150.jpg` — sample inputs.

The user-guide `ideal_interferometer.md` (see `docs/user-guide/`)
mirrors the notebook and explains the float128 → float64 quantisation
trick used in `create_rd_grid` / `create_lm_grid`.

---

## 18. Documentation

Built with Sphinx + MyST under `docs/`:

* `index.rst` — landing page with three card links (User Guide, Dev
  Guide, API Docs).
* `user-guide/` — `getting-started.md`, `about.rst`, plus
  `examples_tutorials/ideal_interferometer.md`.
* `developer-guide/` — `getting-started.md`, `contributions.md`,
  `style-guide.md`, `maintainer-guide.md`, `index.md`.
* `api-reference/` — auto-generated API docs grouped by submodule
  (`simulation/{array, observation, scan, visibility}`,
  `dataset/{dataset, utils}`, `io/{config, datawriters, dataconverter}`,
  `fits/writer`, `layouts/`, `utils/{data, logging}`).
* `changes/` + `changelog.rst` — towncrier sources and rendered
  changelog.
* `references.bib` + `references.md` — bibliography (Smirnov 2011 RIME,
  Barnett 2019 FINUFFT, etc.).
* `glossary.md` — defines `RIME`, `NUFFT`, `FFT`.
* `_static/` — logos, theme CSS, image assets for the ideal-
  interferometer walkthrough.
* `conf.py` — Sphinx config with `pydata_sphinx_theme`,
  `sphinx_design`, `sphinxcontrib-bibtex`, `numpydoc`, `nbsphinx`,
  `sphinx-tippy`, `sphinx-togglebutton`.
* `make.bat` / `Makefile` for local builds; `.readthedocs.yaml` for RTD.

---

## 19. Style, tooling, releases

* **Linter / formatter**: `ruff` (`E, F, I, UP, B, SIM`; ignored
  `B905`, `UP038`). 88-char lines, double-quoted strings, format
  docstring code examples.
* **Pre-commit**: configured (`.pre-commit-config.yaml`).
* **Build backend**: `hatchling` + `hatch-vcs` (writes
  `src/pyvisgen/_version.py` from git tags).
* **Versioning**: dynamic via VCS; current vendored `__version__`
  inferred from `_version.py` at build time. README shows
  `0.8.0 (2026-04-26)`.
* **Changelog**: towncrier categories `feature/bugfix/api/optimization/maintenance`.
* **License**: MIT (single-file `LICENSE`).
* **CI**: GitHub Actions (`.github/workflows/`); pre-commit.ci status
  badge in README.
* **Citation**: `CITATION.cff` with five authors; Zenodo DOI
  `10.5281/zenodo.10091310`.

---

## 20. Mental model and how pieces connect

* The user describes an observation in TOML: a target field (RA/Dec),
  scan timing, frequency layout, antenna array, FoV, image size,
  sampling mode, polarization, noise model, optional Jones effects.
* Loading the TOML produces a `pyvisgen.io.Config` via Pydantic v2.
* `SimulateDataSet.from_config` orchestrates: discovers HDF5 input
  bundles → draws & validates random per-image observation parameters
  → instantiates per-image `Observation` objects → calls `vis_loop`
  with the Stokes-I image cube → optionally grids the result via
  `pyvisgrid` → writes through one of the `DataWriter` subclasses.
* For each image, the `Observation` precomputes time steps, antenna
  elevations, hour angles, parallactic angles, valid baselines (with
  start/stop midpoint integration), the (RA, Dec) and (l, m) grids, and
  the wavelength-band edges.
* `vis_loop` flattens the brightness matrix to the sensitivity-mask
  pixels, picks a baseline subset (full / grid / dense), and runs
  `RIMEScan` per batch; the scan applies K, optional feed rotation,
  optional Airy beam, and integrates over (l, m) and band edges.
* `Visibilities` collects the four polarization products, weights, and
  metadata; the writer either gives ML-ready gridded image pairs (via
  pyvisgrid) or AIPS-conformant UVFITS / UVH5 files for downstream
  imaging tools.

The result, by design, looks "interferometer-shaped" enough to feed
into `radionets` reconstruction networks while remaining a faithful
implementation of the standard image-plane RIME for didactic and
prototyping use. It is **not** a calibration-aware MS-producing
simulator like CASA's `simobserve`, OSKAR, or pyuvsim — there is no
gain corruption, bandpass, ionosphere, troposphere, w-projection, or
beam-from-FITS support; the only Jones terms are a circular-aperture
Airy E-Jones, a parallactic-angle feed rotation, the geometric K-Jones,
and an analytically-modelled Stokes mixing. Within that scope it is
fast, fully GPU-capable through PyTorch, NUFFT-accelerated via
cuFINUFFT/`radioft`, and tightly integrated with the radionets ML
ecosystem.
