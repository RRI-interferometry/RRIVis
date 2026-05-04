# pyuvsim — Exhaustive Reference

A deep, top-to-bottom reference for the `pyuvsim` simulator vendored at
`simulators/pyuvsim/`. Written from a direct read of every source file in
`src/pyuvsim/`, the documentation under `docs/`, the example configs in
`docs/example_configs/` and `src/pyuvsim/data/test_config/`, the
benchmarking harness under `benchmarking/`, the reference simulations
under `reference_simulations/`, the test layout under `tests/`,
`pyproject.toml`, and `CHANGELOG.md`.

`pyuvsim` describes itself as "a comprehensive simulation package for
radio interferometers in python," developed under the
**Radio Astronomy Software Group (RASG)** at
`github.com/RadioAstronomySoftwareGroup/pyuvsim`. It is published in JOSS
(Lanman et al. 2019, doi 10.21105/joss.01234), licensed under the
3-clause BSD License, requires Python ≥ 3.11, and the vendored copy
tracks releases through `1.4.2` (2025-10-03; `Unreleased` section in
`CHANGELOG.md` is currently empty).

Its central design principle is **accuracy over speed**: it is the
**"Interferometer Simulator of Record"** for the 21cm cosmology
community. It performs a brute-force evaluation of the full
direction-dependent, full-polarization Radio Interferometer Measurement
Equation (RIME), parallelized with MPI, and is intentionally slow and
memory-hungry compared to alternatives like `matvis`, `fftvis`, FHD,
PRISim, or CASA. It exists to *verify* those faster simulators against a
known-correct reference.

---

## 1. What pyuvsim is, in one paragraph

For a sky modelled as a `pyradiosky.SkyModel` (point sources or HEALPix
maps, with Stokes I/Q/U/V, in any of the spectral models supported by
`pyradiosky`: `flat`, `full`, `subband`, `spectral_index`), and for
per-antenna primary beams provided as `pyuvdata.UVBeam` objects (read
from `beamfits`, `mwa_hdf5`, etc.) and/or
`pyuvdata.analytic_beam.AnalyticBeam` subclasses (Airy, Gaussian,
ShortDipole, Uniform, or any user-defined importable analytic beam),
`pyuvsim` evaluates the RIME

```
V_pq(ν, t) = Σ_s J_p(s, ν, t) · C_s(ν) · J_q^H(s, ν, t)
                                ·  exp(2πi ν b_pq · ŝ / c)
```

over all sources above the local horizon, at each time `t`, for each
antenna pair `(p, q)` and each feed pair, with **per-source per-baseline
exact Jones-matrix construction** (no flat-sky, no w-projection, no
gridding). The `J` matrices are built by interpolating each antenna's
beam at floating-precision source positions in (az, za); coordinate
transforms from ICRS to local AltAz are done by `astropy` (or
`lunarsky`, for moon-based observatories), so source motion includes
higher-order corrections. Visibilities are accumulated baseline-by-
baseline, time-by-time, frequency-by-frequency, source-by-source, and
the work is split across MPI ranks. The result is written out as a
`pyuvdata.UVData` object in `uvh5`, `uvfits`, `miriad`, or CASA
measurement-set format.

---

## 2. Repository layout

```
simulators/pyuvsim/
├── README.md                 User-facing intro, install, deps, citation
├── CHANGELOG.md              History from 0.1.0 (2018-10-24) → 1.4.2 (2025-10-03)
├── CODE_OF_CONDUCT.md
├── LICENSE                   3-clause BSD
├── MANIFEST.in
├── paper.md, paper.bib       JOSS paper (Lanman et al. 2019)
├── pyproject.toml            Build, deps, optional-deps, scripts, ruff, pytest
├── setup.py                  setuptools_scm version stub
├── environment.yml           conda dev env
├── ci/                       GitHub-actions yamls
├── docs/                     Sphinx documentation
│   ├── conf.py
│   ├── usage.rst             Running a simulation; MPI & profiling notes
│   ├── parameter_files.rst   Authoritative reference for the obsparam yaml
│   ├── classes.rst           Auto-API reference
│   ├── developers.rst        Contributing, ref sims, benchmarking
│   ├── comparison.rst        pyuvsim vs PRISim vs FHD vs CASA
│   ├── example_configs/      tranquility_config.yaml, bl_lite_mixed.yaml,
│   │                         baseline_lite.csv
│   ├── Images/               Layout figures, FHD-vs-pyuvsim delay-spectrum plot
│   ├── HERA comparison memo.pdf
│   └── make_index.py
├── benchmarking/             Standalone HEALPix-shell benchmark harness
│   ├── benchmark.py          settings_setup, make_benchmark_configuration,
│   │                         make_jobscript, update_runlog
│   ├── run_benchmarking.py
│   ├── analyze_runtimes.py
│   ├── settings.yaml
│   ├── BENCHMARKS.log        Tracked git history of large-job timings
│   └── README.md
├── reference_simulations/    First- & second-generation reference sims
│   ├── README.md             How to download, run, compare
│   ├── jobscript.sh          SLURM submission template
│   ├── run_ref_sims.sh       Loop over obsparams
│   ├── first_generation/     Memos + (configs now in src/pyuvsim/data)
│   └── second_generation/    Older configs (out-of-date per README)
├── src/pyuvsim/              ← all package source (described in §3)
└── tests/                    Test layout described in §11
```

The Python package itself contains exactly ten Python modules totalling
~6,750 source lines:

```
src/pyuvsim/
├── __init__.py     (51 lines)   Version detection (setuptools_scm + branch
│                                 scheme), re-exports of Antenna, Baseline,
│                                 Telescope, BeamList, simsetup.*, uvsim.*
├── antenna.py      (174 lines)  Antenna class: name/number/ENU/beam_id +
│                                 get_beam_jones() (the only place beams
│                                 are evaluated)
├── baseline.py     (66 lines)   Baseline class: a pair of antennas, ENU
│                                 vector difference, comparison ops
├── cli.py          (942 lines)  All console_scripts entrypoints (see §10)
├── data/                        Test data, beams, layouts, configs (see §9)
├── mpi.py          (489 lines)  MPI bootstrap, shared-memory broadcast,
│                                 big_bcast / big_gather (>4 GiB), Counter
├── profiling.py    (131 lines)  line_profiler wiring through MPI
├── simsetup.py     (2,974 lines) Config parsing → UVData/SkyModel/BeamList
├── telescope.py    (440 lines)  Telescope dataclass + the BeamList frozen
│                                 dataclass with shared-memory broadcast
├── utils.py        (372 lines)  altaz↔za_az, write_uvdata, progsteps,
│                                 estimate_skymodel_memory_usage,
│                                 iter_array_split, get_avail_memory
└── uvsim.py        (1,111 lines) UVTask, UVEngine, uvdata_to_task_iter,
                                  run_uvdata_uvsim, run_uvsim
```

There is no Cython, no compiled extension, no GPU code. Everything is
pure Python on top of `numpy`, `astropy`, `pyuvdata`, `pyradiosky`, and
optional `mpi4py`.

---

## 3. Dependencies and install matrix

Required (from `pyproject.toml`, also enforced in `README.md`):

| Package        | Min version |
|----------------|-------------|
| `python`       | ≥ 3.11      |
| `astropy`      | ≥ 6.0       |
| `numpy`        | ≥ 1.23      |
| `psutil`       | (any)       |
| `pyradiosky`   | ≥ 1.1.0     |
| `pyuvdata`     | ≥ 3.2.3     |
| `pyyaml`       | ≥ 5.4.1     |
| `scipy`        | ≥ 1.9       |
| `setuptools_scm` | ≥ 8.1     |

Optional extras (selected by `pip install pyuvsim[<extra>]`):

| Extra          | Adds                                 | Purpose                                     |
|----------------|--------------------------------------|---------------------------------------------|
| `sim`          | `mpi4py>=3.1.3`, `psutil`            | Required to actually *run* simulations      |
| `casa`         | `python-casacore>=3.5.2`             | Write CASA Measurement Sets (no Windows)    |
| `healpix`      | `astropy-healpix>=1.0.2`             | HEALPix sky catalogs / beams                |
| `moon`         | `lunarsky>=0.2.5`                    | Telescopes on the Moon                      |
| `plot`         | `matplotlib>=3.6`                    | `plot_csv_antpos`, `text_to_catalog`        |
| `all`          | union of `casa`, `healpix`, `moon`, `plot`, `sim` | full feature set                |
| `test`         | `coverage`, `pooch>=1.8`, `pre-commit`, `pytest`, `pytest-cov>=5.0` | unit tests |
| `sim-test`     | `pyuvsim[sim,test]`, `mpi-pytest>=2025.7.0` | parallel tests                       |
| `doc`          | `matplotlib`, `pypandoc`, `sphinx`   | docs build                                  |
| `profiler`     | `line-profiler`                      | MPI-aware line profiling                    |
| `dev`          | `pyuvsim[all,test,sim-test,doc,profiler]` | full dev install                       |
| `windows-dev`  | `pyuvsim[test,doc,profiler,healpix,sim,sim-test]` | dev on Windows (no casa/moon)  |

Note: `casacore` and `lunarsky` are **not supported on Windows**.
Pyuvsim CI tests against Linux, macOS and Windows; against Open MPI,
MPICH (Linux/Mac) and MS-MPI (Windows). Reference simulations and
benchmarking presume a SLURM-managed Linux cluster.

The library can be installed without MPI (`pip install pyuvsim`), in
which case the `simsetup` config-building functions, the `cli` helpers,
and the analytic beam machinery still work, but `uvsim.run_uvsim` and
`uvsim.run_uvdata_uvsim` raise `ImportError` when invoked.

---

## 4. The four-layer architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  USER LAYER                                                     │
│   • obsparam.yaml + telescope_config.yaml + layout.csv          │
│   • or programmatic: a UVData object, a BeamList, a SkyModel    │
│   • or CLI: run_pyuvsim --param obsparam.yaml                   │
├─────────────────────────────────────────────────────────────────┤
│  CONFIG / SIMSETUP LAYER  (simsetup.py)                         │
│   • initialize_uvdata_from_params()                             │
│   • initialize_catalog_from_params() + SkyModelData             │
│   • parse_telescope_params() + _construct_beam_list() → BeamList│
│   • create_mock_catalog()                                       │
├─────────────────────────────────────────────────────────────────┤
│  ORCHESTRATION LAYER  (uvsim.py)                                │
│   • run_uvsim(params) → reads files, calls run_uvdata_uvsim     │
│   • run_uvdata_uvsim(input_uv, beams, sky, …)                   │
│       - _make_task_inds: split work over MPI ranks              │
│       - uvdata_to_task_iter: yield UVTask per (blt, freq)       │
│       - UVEngine.set_task → make_visibility → MPI.Win.Accumulate│
│       - _update_uvd: blt-reorder, history, set uvws             │
├─────────────────────────────────────────────────────────────────┤
│  PHYSICS / TASK LAYER  (uvsim.UVTask, uvsim.UVEngine)           │
│   • For each (Time t, Freq ν, Baseline b, SkyModel s):          │
│       coherency = SkyModel.coherency_calc()                     │
│       (J1, J2)  = Antenna.get_beam_jones(...)                   │
│       apparent  = J1 · coherency · J2^H                         │
│       fringe    = exp(2πi ν b·ŝ/c)                              │
│       V         = Σ_s apparent_s · fringe_s                     │
│       reshape into [xx], or [xx, yy, xy, yx]                    │
└─────────────────────────────────────────────────────────────────┘
```

`pyuvsim` does **not** have a "backend" in the matvis/fftvis sense. There
is exactly one numerical implementation; parallelism is through MPI
process distribution and shared memory, not through a CPU-vs-GPU code
path. `numpy`/`scipy` BLAS threading (controlled by the underlying BLAS
implementation, `OMP_NUM_THREADS`, etc.) provides additional sub-task
acceleration; the docs recommend 2–4 cpus per MPI rank.

---

## 5. The RIME implementation, line by line

### 5.1 The `Antenna` class (`src/pyuvsim/antenna.py`)

An `Antenna` carries name, integer number, ENU position (`pos_enu` in
`astropy.units.m`), and an integer `beam_id` indexing into a `BeamList`.
The asserts in `__init__` enforce that `name` is a `str`, and that
`number`/`beam_id` are Python `int`/`np.int32`/`np.int64`. Equality is
ENU position within 1 mm + same name + same beam_id.

`get_beam_jones(array, source_alt_az, frequency, …)` is the **only** place
in the package where beams are interpolated at source positions. Inputs:

- `array.beam_list[self.beam_id]` is a `pyuvdata.BeamInterface` wrapping
  either a UVBeam or an AnalyticBeam.
- `source_alt_az` is a (2, Nsrc) array of (alt, az) in radians, in the
  astropy convention (East-of-North, N=0, E=π/2).
- `frequency` is a single scalar frequency (a Python float in Hz or an
  `astropy.units.Quantity`); never an array. **Beams are interpolated at
  one frequency per UVTask.**

Steps:
1. Convert (alt, az) → (za, az') with `simutils.altaz_to_zenithangle_azimuth()`,
   which switches to the `pyuvdata.UVBeam` convention (North-of-East,
   E=0, N=π/2).
2. Verify `beam_list.beam_type == "efield"` (power beams cannot be used
   in a pyuvsim simulation; `apply_beam` does an einsum over an
   E-field Jones matrix).
3. Verify `beam_list.data_normalization in (None, "peak")`.
4. Build `interp_kwargs` including the global `spline_interp_opts` and
   `freq_interp_kind` carried by the BeamList, plus
   `check_azza_domain=beam_interp_check`. `reuse_spline=True` lets
   `UVBeam.interp` cache RectBivariateSpline objects across calls.
5. Call `BeamInterface.compute_response(**interp_kwargs)` which returns
   shape `(Naxes_vec=2, Nfeeds, 1, Ncomponents)`.
6. Repack into shape `(Nfeeds, 2, Ncomponents)` *swapping* the vector
   axis: `jones_matrix[:, 0] = interp_data[1, :, 0, :]` and
   `jones_matrix[:, 1] = interp_data[0, :, 0, :]`. The first axis is
   feed; the second axis is the (theta, phi) component on the sky in
   (az, za) — the swap is because uvbeam stores in (vec0, vec1) order
   while pyuvsim wants (theta, phi).

### 5.2 The `Baseline` class (`src/pyuvsim/baseline.py`)

A trivial dataclass-like wrapper around two `Antenna` objects:
- `enu = antenna2.pos_enu - antenna1.pos_enu`
- `uvw = enu` because pyuvsim **always works in the local
  alt/az frame**: there is no projection to a phase center, no
  w-projection, no rotation. The output `UVData.uvw_array` is rebuilt at
  the end via `uv_container.set_uvws_from_antenna_positions(update_vis=False)`
  and the phase-center catalog is forced to a single
  `{0: {"cat_name": "unprojected", "cat_type": "unprojected"}}` entry if
  the input was projected.

Comparison ops (`__gt__`, `__lt__`, …) sort by `(antenna1, antenna2)`,
which is what determines `lexsort` ordering inside
`uvdata_to_task_iter`.

### 5.3 The `Telescope` class (`src/pyuvsim/telescope.py`)

Plain class. Holds `name`, `location` (an
`astropy.coordinates.EarthLocation` or `lunarsky.MoonLocation`), and a
`BeamList`. Equality is location within 1 mm + per-element beam equality
+ name match. `Telescope` itself is not MPI-shared — the heavy lifting
is in `BeamList.share()`.

### 5.4 The `BeamList` (frozen dataclass)

```python
@dataclass(frozen=True)
class BeamList:
    beam_list: list[UVBeam | AnalyticBeam]
    beam_type: Literal["efield", "power"] | None = "efield"
    spline_interp_opts: dict[str, int] | None = None  # e.g. {"kx": 4, "ky": 4}
    freq_interp_kind: str = "cubic"
    peak_normalize: InitVar[bool] = True
```

`__post_init__` deep-copies any `UVBeam`s in the list, calls
`peak_normalize()` on each, and wraps every entry in a
`pyuvdata.BeamInterface(beam, beam_type=self.beam_type)`. Then
`_check_consistency()` enforces that all beams have the same
`beam_type`, `data_normalization`, `Nfeeds`, and `feed_array`; for
`efield` beams it also checks that the basis vectors are aligned with
azimuth/zenith-angle in each pixel (only `az_za` and `healpix` pixel
coordinate systems are supported, and only the `(1,0)`/`(0,1)` basis is
allowed). For `power` beams it also checks `Npols` and
`polarization_array`.

`check_all_azza_beams_full_sky()` inspects every UVBeam with
`pixel_coordinate_system == "az_za"` and decides whether the beam covers
the full sky to within two grid steps. This is used by
`run_uvdata_uvsim` to default `beam_interp_check=False` when all beams
are full-sky (skipping per-source domain checks for speed).

`share(root=0)` is the MPI broadcast routine. It takes a populated
BeamList on the root rank and replicates it to all other ranks **using
shared memory inside each node** so that each large `_data_array` is
materialized once per node, not once per process. The trick:
- `mpi.shared_mem_bcast(arr, root=…)` allocates a shared-memory window
  on each node via `MPI.Win.Allocate_shared`, fills it on the root
  rank, marks it read-only, and returns a numpy view into the shared
  buffer.
- `_share_uvbeam(bi, root)` walks every set `UVParameter` on the
  underlying UVBeam; for the special `_data_array` parameter it uses
  shared-memory broadcast, for everything else it uses a plain
  `comm.bcast`.
- AnalyticBeams are tiny, so they are simply pickled+broadcast.

### 5.5 The task object (`uvsim.UVTask`)

A `UVTask` packages everything needed for one visibility evaluation:

```
UVTask:
    sources       : pyradiosky.SkyModel   # one SkyModel chunk
    time          : astropy.time.Time     # converted from float-JD if needed
    freq          : astropy.units.Quantity # converted from float-Hz if needed
    baseline      : pyuvsim.Baseline
    telescope     : pyuvsim.Telescope
    freq_i        : int                   # 0 if spectral_type == "flat"
    visibility_vector : ndarray (Npols,)  # filled by UVEngine.make_visibility
    uvdata_index  : (blt_i, freq_i)       # set by uvdata_to_task_iter
```

Comparison ops sort first by baseline, then by frequency index, then by
blt index — the order in which work is committed back to the data array.

### 5.6 The compute kernel (`uvsim.UVEngine`)

`UVEngine` holds an internal cache of "is the time/freq/beam-pair the
same as last call?" Each `set_task(task)` updates three boolean flags
(`update_positions`, `update_local_coherency`, `update_beams`) so that
position transforms, coherency calculations, and beam interpolations are
**only redone when their relevant input has changed**. This is the
single biggest runtime optimization in pyuvsim: in a typical sim where
tasks are sorted by `(time, freq, baseline)`, the inner loop reuses
`alt_az` across all (freq, baseline) for the same time, reuses the
coherency across all (freq, baseline) for the same time, and reuses the
Jones matrices across all baselines that share the same (time, freq,
beam-pair).

`apply_beam(beam_interp_check=True)` builds the apparent coherency:

```python
sources.update_positions(time, location)            # if update_positions
self.local_coherency = sources.coherency_calc()     # if update_local_coherency
beam1_jones = ant1.get_beam_jones(...)              # if update_beams
beam2_jones = (np.copy(beam1_jones)                  # …or recomputed
               if beam1_id == beam2_id else
               ant2.get_beam_jones(...))
coherency = self.local_coherency[:, :, freq_i, :]
beam2_jones = np.swapaxes(beam2_jones, 0, 1).conj()  # transpose at each src
self.apparent_coherency = np.einsum(
    "abz,bcz,cdz->adz", beam1_jones, coherency, beam2_jones
)
```

`make_visibility()` finishes the RIME contribution from one source set:

```python
pos_lmn = sources.pos_lmn[..., sources.above_horizon]   # (3, Nsrc_up)
uvw_wavelength = baseline.uvw / c * freq.to("1/s")      # (3,)
fringe = exp(2j*pi * dot(uvw_wavelength, pos_lmn))      # (Nsrc_up,)
vij = self.apparent_coherency * fringe                  # (Nfeed, Nfeed, Nsrc)
vij = vij.sum(axis=2)                                   # (Nfeed, Nfeed)
return [vij[0,0]]                            if Nfeed==1 else
       [vij[0,0], vij[1,1], vij[0,1], vij[1,0]]    # XX YY XY YX
```

Output polarization order is fixed: **`[XX, YY, XY, YX]`** (i.e.
linear feeds; the package also supports a single-feed (one polarization)
mode added in 1.4.1). The 1/2 factor in the coherency definition is
inherited from `pyradiosky.SkyModel.coherency_calc()` — pyuvsim does
not multiply or scale the coherency anywhere itself.

`source.update_positions()` and `source.coherency_calc()` are
`pyradiosky.SkyModel` methods. `pyuvsim` therefore does not, itself,
implement coordinate transforms — it forwards them to `astropy` (for
Earth) or `lunarsky` (for the Moon) via `pyradiosky`. Source positions
are therefore "floating-precision" in astropy's sense.

### 5.7 Task generation (`uvsim.uvdata_to_task_iter`)

Given a `range` of flattened-meshgrid task indices (length `Ntasks_local`),
a `UVData`, a `SkyModelData` (already subselected to the rank's source
chunk), a `BeamList`, and a `beam_dict`:

1. Build `Telescope` and unique `Antenna` objects, but **only** for
   antennas that actually appear in `ant_1_array ∪ ant_2_array`. The
   `beam_dict` maps antenna *name* → beam_id (defaults to all-zero if
   `None`).
2. Decide whether to chunk the catalog into `Nsky_parts` slices (memory
   protection — see §6.2).
3. For each source chunk:
   - Build a `SkyModel` from the chunk via `SkyModelData.get_skymodel(inds)`.
   - If the spectral type is "flat" and there is no freq_array set,
     the simulation freq_array is attached.
   - Take the slice of `(time, baseline, freq)` that corresponds to the
     local task indices, then `lexsort((bls, freqs, times))` to get the
     traversal order — **this is what enables the UVEngine cache** to
     reuse alt/az, coherency, and beams across consecutive tasks.
   - For each task: build (or reuse) a `Baseline` object, instantiate a
     `UVTask`, set its `uvdata_index = (blt_i, freq_i)`, and yield it.

Time arrays are wrapped in `astropy.time.Time` (or `lunarsky.Time` if
the location is a `MoonLocation`); freq arrays in `astropy.units.Hz`.

### 5.8 The MPI distribution (`uvsim.run_uvdata_uvsim`)

Public, in-memory entry point that takes a fully-populated `UVData`, a
`BeamList`, a `beam_dict`, and a `SkyModel`/`SkyModelData`. The flow:

1. **Input check (`_run_uvdata_input_check`).** Only the root rank does
   real work here:
   - Confirms `input_uv` is a `UVData`; coerces a passed-in `SkyModel`
     into a `SkyModelData` (this is the bug-fix added in 1.4.1).
   - Confirms `beam_list.beam_type == "efield"` and that any UVBeams are
     peak-normalized.
   - Computes the expected polarization array from the beam's feeds via
     `pyuvdata.utils.pol.convert_feeds_to_pols(..., include_cross_pols=True)`
     and confirms that `input_uv.polarization_array` matches.
   - Forces `input_uv.blt_order = ("time", "baseline")` (saves the
     original order so we can restore it later).
   - For non-flat / non-full / non-point catalogs, reads the SkyModel out
     of the catalog, converts to the simulation frequencies via
     `at_frequencies`, converts HEALPix → point sources via
     `healpix_to_point` if needed, and re-wraps as a `SkyModelData`.
2. **Bootstrap MPI** with `mpi.start_mpi(block_nonroot_stdout=True)`.
   On non-root ranks, instantiate empty `SkyModelData` and `BeamList([])`
   placeholders.
3. **Broadcast everything.** `input_uv` is `comm.bcast`-pickled (the
   full UVData object is sent — that is one of the big memory costs);
   `beam_dict` is bcast; the catalog and the BeamList both go through
   their custom `share()` methods which use shared memory.
4. **Allocate the visibility window.** Only root touches the actual
   numpy buffer:

   ```python
   if rank == 0:
       uv_container = simsetup._complete_uvdata(input_uv, inplace=False)
       vis_data = mpi.MPI.Win.Create(
           uv_container._data_array.value, comm=mpi.world_comm)
   else:
       vis_data = mpi.MPI.Win.Create(None, comm=mpi.world_comm)
   ```

   Every rank can then `Lock(0)` / `Accumulate(vis, ...)` / `Unlock(0)`
   into the root's `data_array` without serializing any object — this is
   MPI-3 one-sided communication. Each visibility contribution is
   targeted by `flat_ind = ravel_multi_index((blti, freq_ind, 0),
   data_array_shape)` with `op=mpi.MPI.SUM`.
5. **Assign work.** `_make_task_inds(Nblts, Nfreqs, Nsrcs, rank, Npus)`
   decides whether to split the source axis or the (blt × freq) axis
   (see §6.1). A `mpi.Counter` is allocated to give the root a global
   progress counter (atomic MPI-RMA based).
6. **Set `Nsky_parts`.** `_set_nsky_parts(Nsrcs, cat_nfreqs, Nsky_parts)`
   estimates the per-process memory required by the SkyModel
   (`utils.estimate_skymodel_memory_usage`) times the number of MPI
   processes per node, divides into 50% of available node memory, and
   chooses an `Nsky_parts ≥ ceil(footprint / mem_max)`. If a user-passed
   `Nsky_parts` is too small a `ValueError` is raised; if it is `None`
   the calculated value is used.
7. **Run the task loop.** Each rank does:

   ```python
   engine = UVEngine()
   for task in local_task_iter:
       engine.set_task(task)
       vis = engine.make_visibility()
       blti, freq_ind = task.uvdata_index
       flat_ind = ravel_multi_index((blti, freq_ind, 0), data_array_shape)
       vis_data.Lock(0); vis_data.Accumulate(vis, 0, target=flat_ind*16,
                                              op=mpi.MPI.SUM); vis_data.Unlock(0)
       cval = count.next()
       if rank == 0: pbar.update(cval)
   ```

   While work is in flight a non-blocking `Ibarrier` is used so the root
   can keep updating the progress bar.
8. **Cleanup.** `count.free(); vis_data.Free()` in a `finally` block.
9. **Postprocess (root only).** `_update_uvd` reorders the BLT axis back
   to the input ordering; replaces any projected phase center catalog
   with `unprojected`; rebuilds `uvw_array` from the antenna positions;
   appends history including `pyuvsim`/`pyradiosky`/`pyuvdata` versions,
   the input config files (`obsparam`/`telecfg`/`layout` from
   `extra_keywords`), and `Npus`.

Returns the populated `UVData` only on rank 0. On non-root ranks the
function returns `None`.

### 5.9 The wrapper (`uvsim.run_uvsim`)

This is the entry point for parameter-file-driven runs. Workflow:

```python
mpi.start_mpi(block_nonroot_stdout=…)
if rank == 0:
    input_uv, beam_list, beam_dict = simsetup.initialize_uvdata_from_params(
        params, return_beams=True)
    skydata = simsetup.initialize_catalog_from_params(
        params, input_uv, return_catname=False)
    skydata = simsetup.SkyModelData(skydata)

uv_out = run_uvdata_uvsim(input_uv, beam_list, beam_dict=beam_dict,
                          catalog=skydata, quiet=quiet,
                          beam_interp_check=beam_interp_check)
if rank == 0:
    simutils.write_uvdata(uv_out, param_dict, dryrun=return_uv)

return uv_out  # only if return_uv
```

Initialization timing prints (in minutes, to stdout) bracket the
SkyModel setup and the run. `params` may be either a path to a yaml
file or a python dict already in obsparam form.

### 5.10 The `_check_ntasks_valid` guard

There is a hard ceiling, found experimentally:
`MAX_NTASKS_GATHER = 10_226_018`. The function exists but is not
currently invoked anywhere in the main flow — it is a vestige tied to
issue #289. If the total task count exceeds it the user is asked to
split the simulation into smaller jobs.

---

## 6. Parallelism, splitting, and memory model

### 6.1 Splitting the work axis

`_make_task_inds(Nblts, Nfreqs, Nsrcs, rank, Npus)` implements a
three-case heuristic:

| Condition                              | Outcome                                            |
|----------------------------------------|----------------------------------------------------|
| `Nbltf < Npus` and `Npus < Nsrcs`      | Split the source axis. Each rank gets a SLICE of sources, but ALL (blt, freq) tasks. |
| Otherwise                               | Split the (blt × freq) axis. Each rank gets a slice of tasks but ALL sources.        |
| `Nsrcs < Npus` and `Nbltf < Npus`      | Same as "otherwise" — fall through to (blt × freq) split. |

Where `Nbltf = Nblts * Nfreqs`. The actual per-rank slice is computed by
`utils.iter_array_split(rank, N, M)`, which mimics
`numpy.array_split(np.arange(N), M)[rank]` without materializing the
array.

The docs (`usage.rst`) note that further parallelism within each MPI
process is achieved by `numpy`/`scipy` BLAS threading; using ~2–4
cpus-per-task in SLURM is recommended.

### 6.2 The `Nsky_parts` chunking layer

Within each rank's source slice, the SkyModel is **further** chunked
into `Nsky_parts` pieces to bound peak memory. This second axis matters
because `SkyModel.coherency_calc()` materializes
`(2, 2, Nfreqs, Ncomponents)` complex arrays and `pos_lmn` adds another
`(3, Ncomponents)`. The estimator
`utils.estimate_skymodel_memory_usage(Ncomponents, Nfreqs)` sums:

- One float per component for `ra`, `dec`, `rise_lst`, `set_lst`.
- Two floats per component for `alt_az`.
- Three floats per component for `pos_lmn`.
- One bool per component for `horizon_mask`.
- One Python string per component for `name`.
- Four floats per (component × freq) for `stokes`, `coherency_radec`,
  and `coherency_local`.

The available memory is `psutil.virtual_memory().available`, OR
`SLURM_MEM_PER_NODE` if running under SLURM. Pyuvsim allows up to **50%**
of available memory for SkyModel data; everything else is reserved for
beams, the UVData buffer, and Python overhead.

### 6.3 Shared memory and the broadcast helpers

`mpi.shared_mem_bcast(arr, root)` creates one shared window per node
(via `MPI.COMM_TYPE_SHARED` + `MPI.Win.Allocate_shared`) and lets every
rank on that node see the same buffer. The shared array is marked
`WRITEABLE = False`. All windows are tracked in a module-level
`shared_window_list` and freed at exit by `atexit.register(free_shared)`.

`mpi.big_bcast(comm, objs, root, MAX_BYTES=INT_MAX)` and
`mpi.big_gather(...)` exist because `mpi4py.MPI.Bcast`/`Gatherv` use
32-bit integers for byte counts — sending more than ~2 GiB through a
single broadcast otherwise overflows. `big_*` chunk the pickled bytes
into ≤ `MAX_BYTES` slices and Bcast/Gatherv them in turn. For numpy
arrays the bytes are sent without pickling, with `shape`/`dtype` sent
ahead.

`mpi.Counter` is a parallel counter built on `MPI.Win.Allocate` +
`MPI.Win.Get_accumulate(op=MPI.SUM)` — adapted from the mpi4py
`nxtval-mpi3.py` demo. It returns the new value atomically.

`mpi.set_mpi_excepthook(comm)` registers a `sys.excepthook` that calls
`comm.Abort(1)` on any uncaught exception, ensuring that a Python
exception on any rank takes down the whole job (preventing hangs).

`mpi.get_max_node_rss(return_per_node=False)` reads the per-process RSS
via `resource.getrusage(RUSAGE_SELF)` (or `psutil.Process().memory_info()`
on platforms where `resource` is unavailable), `allreduce(SUM)` over
node_comm to get total RSS per node, then `allreduce(MAX)` over
world_comm for the global max.

### 6.4 stdout discipline

When `start_mpi(block_nonroot_stdout=True)` (the default), every non-root
rank's `sys.stdout` is reassigned to `/dev/null` (or `NUL` on Windows).
This is critical to keep MPI logs readable; if you want to debug a
non-root rank, pass `--keep_nonroot_stdout` to `run_pyuvsim`.

---

## 7. The configuration system (`simsetup.py`)

`simsetup.py` is by far the largest module (~2,974 lines). It implements
the entire human-facing configuration interface: how a YAML
`obsparam.yaml` plus a `telescope_config.yaml` plus a layout `csv` plus
a sky catalog become a `(UVData, BeamList, beam_dict, SkyModelData)`
4-tuple. It also implements the inverse: how to dump a `UVData` object
into matching configuration files. The interface is also used by
**other simulators** (notably `matvis.cli`, `WODEN`, `RIMEz` — pyuvsim's
obsparam.yaml format has become an informal community standard).

### 7.1 The `obsparam.yaml` schema

Top-level sections (every key shown below is documented in
`docs/parameter_files.rst` and parsed in `initialize_uvdata_from_params`):

```yaml
filing:                 # All optional. Output paths and naming.
  outdir:    "."        # Directory; created if missing.
  outfile_name: "name"  # Full output filename (extension auto-added).
  outfile_prefix: "sim" # Used if outfile_name not given.
  outfile_suffix: "results"
  output_format: "uvh5" # uvh5 | uvfits | miriad | ms
  clobber: false        # Otherwise the writer increments _0, _1, …

freq:                   # Specify the frequency axis. See §7.4.
  Nfreqs: 100
  channel_width: 195312.5
  start_freq: 1.0e8
  end_freq:                       # OR
  bandwidth:                      # OR
  freq_array: [1.0e8, 1.0008e8, …]

time:                   # Same combinatorics as freq. See §7.5.
  Ntimes: 1
  integration_time: 11.0          # seconds
  start_time: 2457458.1738949567  # Julian Date (must be a 64-bit float)
  end_time:
  duration_hours:                 # OR duration_days
  time_array: [...]
  time_offset: 2457458.0          # see §7.5 for precision trick

sources:                # Catalog parameters; see §7.3.
  catalog: "../gleam_50srcs.vot"  # path or "mock"
  filetype: "gleam"     # skyh5 | gleam | vot | text | fhd; auto-guessed if absent
  spectral_type: flat   # GLEAM only: flat | subband | spectral_index
  flux_columns: [int_flux_076, int_flux_092, …]   # required for non-GLEAM VOT
  table_name: single
  id_column: name
  ra_column: RAJ2000    # or lon_column / lat_column / frame
  dec_column: DEJ2000
  reference_frequency: 1.5e8
  freq_array: [...]
  spectral_index_column: alpha
  min_flux: 0.2         # Jy
  max_flux: 1.5         # Jy
  non_nan: "any"        # "any" | "all" — drop sources with NaN Stokes
  non_negative: true    # drop sources with negative Stokes I
  horizon_buffer: 0.04  # radians, padding on the rough rise/set cut
  # Mock-catalog branch:
  mock_arrangement: "zenith"    # zenith | off-zenith | triangle | cross |
                                # long-line | hera_text | random | diffuse
  Nsrcs: 1
  alt: 87.0
  min_alt: 30
  rseed: 42
  array_location: "lat,lon,alt"  # decimal degrees, meters
  time: 2460000.0
  diffuse_model: "monopole"     # mock_arrangement: "diffuse" only
  diffuse_params: {}
  map_nside: 128

telescope:              # See §7.2. Either layout-csv path or dict + names+nums.
  array_layout: "../baseline_lite.csv"
  telescope_config_name: "../mwa_config_short_dipole.yaml"
  # Or, if going barebones without a telescope_config_name:
  telescope_location: (-30.7, 21.4, 1073.0)
  telescope_name: "TEST"
  feed_array: ["x", "y"]
  feed_angle: [0, 1.5707963]
  mount_type: "alt-az"
  world: "earth"        # or "moon"
  ellipsoid: "SPHERE"   # for moon: SPHERE | GSFC | GRAIL23 | CE-1-LAM-GEO
  select:               # DEPRECATED at obsparam level (use telconfig:select)
    freq_buffer: 1.0e6

select:                 # Down-select the simulated UVData. See §7.6.
  bls: "[(1,2),(3,4)]"  # NOTE the wrapping string for tuple lists
  ant_str: "cross"
  antenna_nums: [1, 7, 9]
  redundant_threshold: 0.1
  no_autos: false

ordering:               # See §7.7.
  conjugation_convention: "ant1<ant2"   # default since 1.3.1
  blt_order: ["time", "baseline"]       # the only order pyuvsim runs in

polarization_array: [-5, -6, -7, -8]    # XX, YY, XY, YX (defaults for 2 feeds)
cat_name: "myfield"     # written into UVData.phase_center_catalog
object_name: "M31"      # synonym

# Anything else that is a valid UVData attribute is ALSO accepted at top-level.
```

### 7.2 The telescope config yaml

`telescope_config_name` is itself a yaml file that defines the
**physical telescope**: its name, its location, its primary beams, and
optional per-beam selects. The full schema:

```yaml
telescope_name: BLLITE
telescope_location: (-30.72152777777791, 21.428305555555557, 1073.0)
world: earth         # or "moon"
ellipsoid: SPHERE    # for moon

beam_paths:
  0:                  # Beam id 0
    filename: hera.beamfits          # implicitly a UVBeam
    # any kwarg legal for UVBeam.read can go here
  1: !UVBeam                          # explicit tag (preferred)
    filename: mwa_full_EE_test.h5
    pixels_per_deg: 1
    freq_range: [100.e+6, 200.e+6]
    mount_type: phased
    file_type: mwa_beam
  2: !AnalyticBeam                    # explicit tag (preferred)
    class: AiryBeam                   # any importable analytic beam class
    diameter: 16
  3: !AnalyticBeam
    class: GaussianBeam
    sigma: 0.03                       # OR diameter: 14
    sigma_type: efield                # default; or "power"
    spectral_index: -1.0              # for chromatic Gaussian
    reference_frequency: 1.5e8
  4: !AnalyticBeam
    class: pyuvdata.GaussianBeam      # full module path is accepted
    diameter: 14
  5: !AnalyticBeam
    class: ShortDipoleBeam
  6: !AnalyticBeam
    class: UniformBeam

# Global UVBeam interpolation options
spline_interp_opts:                   # passed to RectBivariateSpline
  kx: 4
  ky: 4
freq_interp_kind: "cubic"             # passed to scipy.interpolate.interp1d

# Optional per-read selects (only safe with UVBeams that DO NOT use !UVBeam tag,
# because the tag triggers UVBeam construction before this select is parsed)
select:
  freq_buffer: 1.0e6                  # MHz padding around simulated freqs
  freq_range: [1.0e8, 2.0e8]
  freq_chans: [...]
```

The four pyuvdata-provided analytic beams are:

- **`AiryBeam`** — Airy disk; chromatic; needs `diameter` (m); unpolarized.
- **`GaussianBeam`** — Gaussian; either `diameter` (m) → matched-to-Airy
  width, chromatic, OR `sigma` (rad) → achromatic by default but
  optionally chromatic via `(spectral_index, reference_frequency)`;
  `sigma_type` controls whether `sigma` describes E-field or power
  width; unpolarized.
- **`ShortDipoleBeam`** — classical short dipole; achromatic;
  intrinsically polarized.
- **`UniformBeam`** — same response in all directions; achromatic;
  unpolarized.

Beams may be specified either with the explicit `!UVBeam` /
`!AnalyticBeam` yaml tag (recommended; pyuvdata registers them as YAML
constructors), or as a plain dict with a `filename` (interpreted as a
UVBeam) or a `type` (a `dict[str, type]` lookup against the legacy
`{"airy": AiryBeam, "gaussian": GaussianBeam, "short_dipole":
ShortDipoleBeam, "uniform": UniformBeam}` map). The dict form raises a
`DeprecationWarning` and will become an error in v1.6.

`_construct_beam_list` is what reads this config and produces a
`BeamList`. It also enforces per-beam frequency selects from the
`select` block (using `UVBeam.select(freq_chans=…)` for already-loaded
beams, or `freq_range` as a UVBeam.read kwarg for not-yet-loaded ones).
Beam shape parameters at the global level (`diameter:`, `sigma:`)
raise an error — they must be per-beam.

### 7.3 The layout csv

Whitespace-separated, with a single header line. Columns may be in any
order, but standard names are `Name Number BeamID E N U`:

```
Name        Number   BeamID   E          N          U
ANT1        0        0        0.0000     0.0000     0.0000
ANT2        1        0        50.000     0.0000     0.0000
ANT3        2        0        0.0000     -50.00     0.0000
ANT4        3        0        26.000     -26.00     0.0000
```

Parsed by `_parse_layout_csv` via `np.rec.format_parser` with dtypes
`name=U10, number=i4, beamid=i4, e=f8, n=f8, u=f8`. The
`BeamID` column references entries in `beam_paths` of the telescope
config — any antennas with the same `BeamID` share the same beam object
in the BeamList, which is critical for memory.

`_write_layout_csv` is the inverse, used by `uvdata_to_telescope_config`
and `initialize_uvdata_from_keywords`.

### 7.4 Frequency parsing (`parse_frequency_params`)

The user can specify the frequency axis through any of:

- `freq_array` (explicit) + optional `channel_width`
- `Nfreqs` + `channel_width` + (`start_freq` or `end_freq`)
- `Nfreqs` + (`start_freq`, `end_freq`)
- `bandwidth` + (`start_freq` or `end_freq`) + (`Nfreqs` or
  `channel_width`)
- `start_freq` + `end_freq` + (`Nfreqs` or `channel_width`)

The logic is implemented by `_setup_coord_arrays`, which is shared by
both `parse_frequency_params` and `parse_time_params` via a
`coord_array/coord_delta/coord_start/coord_end/coord_n/coord_length`
abstraction. The coord-array is required to be evenly spaced unless the
user explicitly passed `freq_array` with matching `channel_width`. Any
inconsistency between user-provided keys raises a warning and uses the
first valid set of keys (priority is array > start+end > duration+n).

### 7.5 Time parsing (`parse_time_params`)

Mirror image of `parse_frequency_params`, with one extra wrinkle: a
`time_offset` parameter can be added to all absolute time parameters
(`start_time`, `end_time`, `time_array`) so that the absolute portion
can be written as a lower-precision float in a yaml dump and the
remainder as a smaller-magnitude high-precision difference. The wrapper
adds `time_offset` to those parameters unless the values already exceed
the offset, in which case it warns and skips. Units conversions:

- `integration_time` → days internally (`int_time_days`)
- `duration_hours` → days
- All other parameters in days.

`time_array_to_params` is the inverse — given a time_array and
integration_times, produce a params dict that round-trips
`parse_time_params(time_array_to_params(t, dt))[…]` ≈ `t`. This is what
`uvdata_to_config_file` uses to dump configs from a UVData.

### 7.6 Selection (`subselect`)

`subselect(uv_obj, param_dict)` accepts any of these keys under
`select`:

- `antenna_nums` — list of antenna numbers
- `antenna_names` — list of antenna names
- `ant_str` — e.g. `"cross"` or `"auto"` or `"1_2"`
- `frequencies` — list of frequencies in Hz
- `freq_chans` — list of channel indices
- `times` — list of times in JD
- `blt_inds` — list of indices along the baseline-time axis
- `bls` — list of `(ant1, ant2)` tuples; **the value MUST be a string in
  the yaml** (parsed with `ast.literal_eval`)
- `no_autos` — Boolean, applies `ant_str="cross"` after other selects
- `redundant_threshold` — Float, meters; applies
  `UVData.compress_by_redundancy(tol=…, use_grid_alg=True)` at the end

Polarization selection is **not allowed** here, because pyuvsim always
computes all polarizations the beams support. Selecting `bls` is done
*before* the UVData is constructed (since 1.4.0), saving memory.

### 7.7 Ordering (`set_ordering`)

`obsparam['ordering']` accepts:

- `conjugation_convention` — one of `"ant1<ant2"` (default since
  1.3.1; was `"ant2<ant1"` before), `"u<0"`, `"u>0"`, `"v<0"`, `"v>0"`.
  Anything other than `"ant1<ant2"` triggers a
  `UVData.conjugate_bls(convention=…)` call after the simulation runs.
- `blt_order` — defaults to `["time", "baseline"]`, which is the order
  pyuvsim itself runs in. Other orders cause the output to be reordered
  AFTER the simulation runs (so they don't change the simulator's
  inner loop).

If `conjugation_convention` is omitted, a `DeprecationWarning` fires
saying it should be set explicitly. This will become an error in 1.5.

### 7.8 The polarization helper (`_initialize_polarization_helper`)

Determines the polarization array from one of:

1. An explicit `polarization_array` in the obsparam.
2. `pyuvdata.utils.pol.convert_feeds_to_pols(beam_list[0].feed_array,
   include_cross_pols=True)` — derives from the first beam's feeds.
3. Default: `[-5, -6, -7, -8]` (XX, YY, XY, YX).

If the user-passed array requires feeds the beams don't have, it raises
a `ValueError`. `Npols` is set to `len(polarization_array)` if not
already.

### 7.9 The mock catalog (`create_mock_catalog`)

Used by `obsparam['sources']['catalog'] == 'mock'` and also directly
exposed. Returns `(SkyModel, mock_keywords)` where `mock_keywords` is a
dict of identifying parameters that go into the catalog name. The eight
arrangements:

| `arrangement` | Output                                                       |
|---------------|--------------------------------------------------------------|
| `zenith`      | `Nsrcs` sources stacked at the zenith. Each carries flux `1/Nsrcs` Jy so total Stokes I = 1 Jy. |
| `off-zenith`  | One source at altitude `alt` (default 85°), azimuth 90°, 1 Jy. |
| `triangle`    | Three sources at altitudes `alt` (default 87°), azimuths 0°/120°/240°, 1 Jy each. |
| `cross`       | Four sources at alts `[88, 90, 86, 82]` and azimuths `[270, 0, 90, 135]`, fluxes `[5, 4, 1, 2]`. |
| `long-line`   | Horizon-to-horizon line of `Nsrcs` 1 Jy sources spaced through altitudes `[min_alt, 90]` on both sides of azimuth 180/0. |
| `hera_text`   | Spells "HERA" using a hardcoded list of 43 azimuth/zenith-angle pairs. |
| `random`      | `Nsrcs` 1 Jy sources uniformly on the sphere above `min_alt` (defaults 30°). Uses `np.random.seed(rseed)`. |
| `diffuse`     | A HEALPix map at `map_nside` (default 32) populated by an `analytic_diffuse` model named `diffuse_model`, evaluated at every pixel. Returns a HEALPix `SkyModel` (Stokes I in K). Requires `analytic_diffuse` and `astropy_healpix`. |

For `discrete` arrangements, sources are defined in AltAz at the given
time and array_location, then transformed to ICRS for the SkyModel
(`pyradiosky.SkyModel(name=..., skycoord=icrs_coord, stokes=..., 
spectral_type="flat")`).

Lunar location handling: if `array_location` is a `MoonLocation`, the
local frame becomes `lunarsky.LunarTopo` and the SkyCoord wrapper
`lunarsky.SkyCoord` — both are looked up dynamically inside a
`contextlib.suppress(ImportError)` so the module imports cleanly without
lunarsky.

### 7.10 The `SkyModelData` shim

`SkyModelData` is pyuvsim's MPI-friendly shadow of
`pyradiosky.SkyModel`. Why: pickling a SkyModel through `MPI.bcast` is
slow and bloats memory; the columnar arrays of a SkyModel can instead
be stored in shared memory windows and mapped on every rank.
`SkyModelData` extracts each large attribute as a plain `np.ndarray` (in
SI / Jy units) and provides:

- `share(root)` — broadcasts each attribute. Arrays in the
  `put_in_shared` allowlist (`stokes_I`, `stokes_Q`, `stokes_U`,
  `stokes_V`, `polarized`, `ra`, `dec`, `reference_frequency`,
  `spectral_index`, `hpx_inds`) go via `mpi.shared_mem_bcast`. Smaller
  attributes go via plain `mpi.world_comm.bcast`.
- `subselect(inds)` — efficient slicing without copying when `inds` is a
  `range`.
- `get_skymodel(inds=None)` — reconstructs a real `pyradiosky.SkyModel`
  with appropriate `stokes`, `freq_array`, `freq_edge_array`,
  `reference_frequency`, `spectral_index`, `nside`, `hpx_inds`,
  `frame="icrs"`, `name`, `skycoord`, and `filename`.

`SkyModelData` retains polarization sparsely: `polarized` is the index
array of components that have non-zero Q/U/V, and only those are stored
in `stokes_Q`/`stokes_U`/`stokes_V` — saving memory when most sources
are unpolarized (which is essentially always).

### 7.11 `initialize_catalog_from_params`

Reads the `sources` block and produces a `SkyModel`:

1. If `catalog == "mock"`, dispatch to `create_mock_catalog` with mock
   keywords pulled from the dict.
2. Else, locate the file: if it's not an absolute path, try relative to
   `config_path`, then check the astropy cache for a download under the
   `"pyuvsim"` cache namespace (`is_url_in_cache(catalog,
   pkgname="pyuvsim")`). This is what makes the
   `download_data_files` workflow work — large beams/catalogs/healpix
   maps are stored in the astropy cache and referenced by URL inside the
   yaml.
3. Build `read_params` from the allowlist (`spectral_type`, `table_name`,
   `id_column`, `ra_column`, `dec_column`, `lon_column`, `lat_column`,
   `frame`, `flux_columns`, `reference_frequency`, `freq_array`,
   `spectral_index_column`).
4. If the file looks like GLEAM (`*.vot` containing "gleam"
   case-insensitively), default `spectral_type="subband"`.
5. Read with `SkyModel.from_file(catalog, **read_params,
   run_check_acceptability=False)` (acceptability check deferred to
   after selections to avoid spurious nan/negative warnings).
6. Apply `_sky_select_calc_rise_set`: any of `min_brightness`,
   `max_brightness`, `non_nan="any"|"all"`, `non_negative=True`, then
   `cut_nonrising(latitude)` and `calculate_rise_set_lsts(latitude,
   horizon_buffer=…)`. The horizon buffer defaults to none (i.e. the
   code-side default in pyradiosky), and the docs note that ~10 minutes
   is sufficient.
7. `sky.check()` (full acceptability) + assign filename.

### 7.12 `initialize_uvdata_from_params`

The big setup function (~210 lines). Steps:

1. Read the obsparam yaml; if a path was passed, store the directory in
   `param_dict["config_path"]` and the basename in
   `param_dict["obs_param_file"]`.
2. `parse_frequency_params` → `freq_array`, `channel_width`, `Nfreqs`.
3. `parse_telescope_params` →
   `(tele_params_dict, beam_list, beam_dict)`. Internally this:
   - Reads the telescope config yaml to get name/location/world/ellipsoid.
   - Reads the layout csv (or layout dict) to get antenna names, numbers,
     ENU positions, and beam_ids.
   - Computes ECEF antenna positions via
     `pyuvdata.utils.XYZ_from_LatLonAlt` (or its mcmf equivalent for
     moon).
   - Calls `_construct_beam_list(beam_ids_inc, telconfig, freq_array,
     freq_range)` for the beams.
   - Builds a per-antenna `mount_type`, `feed_angle` and `feed_array`
     from each beam (introduced in 1.4.1 with the new pyuvdata Telescope
     object).
4. Stash `obs_param_file`, `telecfg`, `layout` paths into
   `extra_keywords` for history.
5. `parse_time_params` → time_array, integration_time.
6. `_initialize_polarization_helper` → polarization_array.
7. Build the actual `pyuvdata.Telescope.new(...)` and
   `UVData.new(telescope=..., antpairs=..., ...)`. The `antpairs` come
   from `param_dict.get("select", {}).pop("bls", None)` so they are
   applied at construction time (since 1.4.0).
8. `subselect(uv_obj, param_dict)` for the remaining selects (no_autos,
   redundant_threshold, frequencies, etc.).
9. `set_ordering(uv_obj, param_dict, reorder_blt_kw)` to set the BLT
   order (default `time, baseline`) and the conjugation convention.
10. Return `(uv_obj, beam_list, beam_dict)` (or just `uv_obj` if
    `return_beams=False`).

### 7.13 `_complete_uvdata`

After `initialize_uvdata_from_params` returns a metadata-only UVData,
`_complete_uvdata(uv, inplace=False)` allocates the actual data arrays:

```python
uv.data_array    = np.zeros((Nblts, Nfreqs, Npols), dtype=complex)
uv.flag_array    = np.zeros((Nblts, Nfreqs, Npols), dtype=bool)
uv.nsample_array = np.ones( (Nblts, Nfreqs, Npols), dtype=float)
uv.set_lsts_from_time_array()
```

This is what the simulator accumulates into via MPI RMA; its `.value`
(the underlying numpy array) is the buffer behind `MPI.Win.Create`.

### 7.14 `initialize_uvdata_from_keywords`

A helper for users who don't want to author yaml. Accepts a long list of
keywords (`Nfreqs`, `start_freq`, `Ntimes`, `integration_time`,
`telescope_location`, `telescope_name`, `array_layout`, `polarization_array`,
…) plus arbitrary additional valid `UVData` attributes via `**kwargs`.
Internally builds a `param_dict` and (optionally) writes it to disk, then
calls `initialize_uvdata_from_params`. If `complete=True`, also invokes
`_complete_uvdata`. Useful for quick programmatic UVData construction.

### 7.15 `uvdata_to_telescope_config` and `uvdata_to_config_file`

Reverse direction: take a UVData, write out a telescope config yaml +
a layout csv (`uvdata_to_telescope_config(uv, beam_filepath, …)`) and an
obsparam yaml (`uvdata_to_config_file(uv, …)`). Together these can clone
a real-data UVData object's structure into a runnable simulation
(`uv` can be from any pyuvdata-readable file, MS, FHD, MIRIAD, uvh5,
…). The CLI counterparts are `uvdata_to_telescope_config` and
`uvdata_to_config`.

---

## 8. Profiling (`profiling.py`)

`pyuvsim.profiling.set_profiler(func_list, rank=0, outfile_prefix,
dump_raw=False)` wraps the optional `line_profiler` package to provide
**per-function, per-line** timing across the MPI job, but only on a
single rank (default rank 0). The default function list is:

```python
default_profile_funcs = [
    "get_beam_jones",
    "initialize_uvdata_from_params",
    "apply_beam",
    "make_visibility",
    "update_positions",
    "coherency_calc",
    "uvdata_to_task_iter",
    "run_uvdata_uvsim",
    "run_uvsim",
]
```

For each name in the list, the profiler walks `pyuvsim.__dict__` and
`pyradiosky.__dict__` (functions and class methods) and adds matching
items via `prof.add_function(item)`. It then registers `atexit` handlers
to write the human-readable line-by-line stats to
`{prefix}.out` and (if `dump_raw=True`) the pickled `LineStats` to
`{prefix}.lprof`. A separate `{prefix}_meta.out` file gets axis sizes
written by `run_uvdata_uvsim` (`Ntimes_loc`, `Nbls_loc`, `Nfreqs_loc`,
`Nsrcs_loc`, `prof_rank`).

The `--profile` and `--raw_profile` flags on `run_pyuvsim` are the
user-facing interface. Note that line_profiler interferes with
pytest-cov, hence many lines in `profiling.py` are marked
`# pragma: nocover` — see `line_profiler` issue #179.

---

## 9. Bundled data (`src/pyuvsim/data/`)

| File / dir                                  | Purpose                                                         |
|---------------------------------------------|-----------------------------------------------------------------|
| `28m_triangle_10time_10chan.uvfits`         | Small uvfits used by tests / docs.                              |
| `28mEWbl_*.uvfits`                          | EW baseline reference data.                                     |
| `5km_triangle_*.uvfits` / `*_layout.csv`    | Long-baseline triangle reference.                               |
| `gleam_50srcs.vot`                          | A 50-source slice of GLEAM, VOTable.                            |
| `gleam_triangle_*_*.uvh5`                   | Pre-simulated GLEAM x triangle for `flat`, `subband`, and `spectral_index` spectral types, both 1- and 2-frequency. |
| `mock_catalog_heratext_2458098.27471265.txt`| The "HERA"-text mock catalog as a flat file.                    |
| `single_source.txt`, `single_source.vot`    | Single-source catalogs for tests.                               |
| `testfile_singlesource.uvh5`                | Output of single-source sim.                                    |
| `HERA_layout.csv`, `HERA_NicCST.beamfits`   | HERA reference layout + a CST-derived beam.                     |
| `mwa_128T_layout.csv`, `mwa_nocore_layout.csv`, `mwa128_layout_longbl.csv` | MWA layouts. |
| `baseline_lite.csv`, `baseline_lite_4x.csv`, `baseline_lite_multi_beam.csv` | Toy 4-antenna layouts. |
| `mwa_config_*.yaml`                         | MWA telescope configs for each AnalyticBeam type and a UVBeam.  |
| `bl_lite_mixed.yaml`, `bl_lite_mixed_constructors.yaml` | Mixed-beam example with HERA+MWA+analytic beams. |
| `mwa128_config.yaml`, `mwa88_nocore_config*.yaml` | MWA full-array configs.                                  |
| `moon_config_uniform.yaml`                  | Lunar uniform-beam config (apollo11 site).                       |
| `tranquility_config.yaml`                   | Mare-Tranquillitatis lunar config from the docs.                 |
| `profiling_params.yaml`                     | Larger config used for profiling smoke-tests.                    |
| `data/test_catalogs/`                       | One-shot mock-catalog text files (zenith/cross/triangle/random/long-line/hera_text/off-zenith) and special test catalogs `RASG.txt`, `R.txt`, `MOON.txt`, `one_distant_point_2458178.5.txt`, `two_points_on_opposite_horizon.txt`. |
| `data/test_config/`                         | Reference simulation obsparams (`obsparam_ref_1.1` … `1.8`), plus regression configs (`28m_triangle_10time_10chan*.yaml`, `bl_single_gauss.yaml`, `obsparam_diffuse_sky*.yaml`, `obsparam_hex37_14.6m.yaml`, `obsparam_lunar_gauss.yaml`, `obsparam_mwa_nocore.yaml`, `obsparam_tranquility_hex.yaml`, `param_*.yaml`). |

The reference simulation configs are the gold-standard examples to
copy from when authoring a real sim.

### Reference simulation overview

The eight first-generation reference simulations
(`obsparam_ref_1.X_*.yaml`) systematically exercise individual pyuvsim
capabilities:

| Number | Name                | Tests…                                                                  |
|--------|---------------------|-------------------------------------------------------------------------|
| 1.1    | `baseline_number`   | Number of baselines (MWA 128T layout, RASG catalog, ShortDipoleBeam).   |
| 1.2    | `time_axis`         | Number of time integrations.                                            |
| 1.3    | `frequency_axis`    | Number of frequency channels.                                           |
| 1.4    | `source_axis`       | Number of point sources in catalog.                                     |
| 1.5    | `uvbeam`            | UVBeam (FITS-based) primary beams.                                      |
| 1.6    | `healpix`           | HEALPix (skyh5) sky model — downloaded from BDR at run time.            |
| 1.7    | `multi_beam`        | Mixed beam types within one array (different antennas, different beams).|
| 1.8    | `lunar`             | Telescope on the Moon (`world: moon`, MoonLocation, lunarsky frames).   |

Each is downloadable from the **Brown Digital Repository**
(`bdr:wte2qah8`), and the CI runs `pytest --refsim={…}` against the
historical reference outputs to catch regressions. The
`download_ref_sims` CLI fetches them; `run_ref_sim 1` runs ref sim 1.1
locally; `--savesim` writes new outputs to `new_data/` for upload back
to the BDR.

Second-generation reference simulations
(`reference_simulations/second_generation/obsparam_ref_2.*.yaml`) exist
but are out-of-date per the README and are not part of CI.

---

## 10. Console entrypoints (`cli.py`, `pyproject.toml`)

The `[project.scripts]` table registers ten executables. All are thin
wrappers around library functions:

| Script                          | Function                              | What it does                                                        |
|---------------------------------|---------------------------------------|---------------------------------------------------------------------|
| `run_pyuvsim`                   | `cli.run_pyuvsim`                     | Run a sim from `--param obsparam.yaml`, **OR** from explicit `--uvdata` + `--uvbeam` + `--skymodel` + `--outfile`. Optional `--profile`, `--quiet`, `--keep_nonroot_stdout`, `--raw_profile`. Always invoked under `mpiexec -n N` for parallelism. |
| `run_param_pyuvsim`             | `cli.run_param_pyuvsim`               | DEPRECATED. Old positional-argument form. Translates args and calls `run_pyuvsim`. Removed in 1.6. |
| `uvdata_to_config`              | `cli.uvdata_to_config`                | Read a UVData-readable file and write out an obsparam yaml + (referenced) telescope config + layout csv that would re-run the same observation as a sim. |
| `uvdata_to_telescope_config`    | `cli.uvdata_to_telescope_config`      | Subset of the above: write only the telescope_config yaml + layout csv. Requires `--beamfile`. |
| `text_to_catalog`               | `cli.text_to_catalog`                 | Generate a SkyModel skyh5 catalog whose sources spell out arbitrary text near zenith for a given site/JD. Uses `ImageMagick` (`convert`) and `matplotlib.image.imread` to convert text bitmap → source list. |
| `im_to_catalog`                 | `cli.im_to_catalog`                   | DEPRECATED alias for `text_to_catalog` (removed in 1.6).            |
| `plot_csv_antpos`               | `cli.plot_csv_antpos`                 | Render an antenna layout csv as a `matplotlib` scatter plot.        |
| `download_data_files`           | `cli.download_data_files`             | Download (via astropy cache) `gleam` (the GLEAM VOTable, via `pyradiosky.utils.download_gleam`), `mwa` (the MWA full embedded element pattern hdf5), or `healpix` (the BDR-hosted GSM 2016 nside-128 skyh5 map). Use `--clear` to wipe the cache first. Use `--row_limit` to truncate GLEAM. |
| `download_ref_sims`             | `cli.download_ref_sims`               | Download all eight first-generation reference simulation result files from the BDR via an api query (`bdr:wte2qah8`). Used by the regression test suite. |
| `run_ref_sim`                   | `cli.run_ref_sim`                     | Wrapper to run any first-generation reference sim by name (`1.1_baseline_number` → `1.8_lunar`) or by number (`1`–`8`).                                |

There are no other entrypoints; the package surface is deliberately
small.

The `text_to_catalog` toolchain is unique: it shells out to ImageMagick
(`subprocess.check_output(["convert", "--version"])`) to render the text
to a bitmap, reads the image with matplotlib, thresholds it (`thresh`
between 1 and 255 — default 140), and converts each lit pixel into a
1-Jy point source at the corresponding (alt, az) about zenith for the
given (lat, lon, jd). The output is a skyh5 file. This is how the
`hera_text` mock arrangement was originally produced.

---

## 11. Test layout (`tests/`)

```
tests/
├── conftest.py              Custom pytest options (--nompi, --refsim,
│                             --savesim), parametrization for refsim,
│                             enforces profiler tests run last.
├── __init__.py
├── test_antenna.py          Antenna init, equality, get_beam_jones
├── test_baseline.py         Baseline construction, ENU diff, comparators
├── test_cli.py              All console_scripts smoke-tests
├── test_mpi.py              big_bcast, big_gather, Counter, shared_mem_bcast
├── test_profiler.py         set_profiler under MPI (runs last)
├── test_run.py              run_uvsim end-to-end, run_uvdata_uvsim
├── test_run_ref.py          --refsim regression tests against BDR archives
├── test_simsetup.py         All of simsetup.py: parsing, catalogs, mock,
│                             uvdata_to_*, etc.
├── test_telescope.py        Telescope, BeamList post_init/share/consistency
├── test_utils.py            altaz↔za_az, write_uvdata, iter_array_split,
│                             estimate_skymodel_memory_usage, progsteps
└── test_uvsim.py            UVTask, UVEngine, _make_task_inds,
                              uvdata_to_task_iter
```

Tests are discovered with `pytest`; MPI-parallel tests use the
`pytest.mark.parallel(n)` decorator from `mpi-pytest>=2025.7.0`. They
can run in a forked mode (default) or with `mpiexec -n N pytest -m
"parallel[N]"`. `--nompi` disables them entirely.

Reference sims are integrated via `--refsim={…}` plus `--benchmark-only`
(requires `pytest-benchmark`); these download reference outputs from the
BDR and check that re-running locally produces the same UVData (within
tolerance).

---

## 12. Convention summary (the specifics other simulators get wrong)

This section summarizes the conventions baked into pyuvsim that any
downstream simulator (RRIVis included) must match to be cross-validated
against it.

1. **Coordinate system for beams.** UVBeam stores in `(theta, phi)` =
   `(za, az)` order, *but* `Antenna.get_beam_jones` swaps them so that
   the returned Jones matrix is `[feed, (az_component, za_component),
   source]`. `apply_beam` builds the apparent coherency in this
   `(az, za)` basis.

2. **Azimuth convention.** Astropy uses East-of-North (`N=0, E=π/2`).
   UVBeam uses North-of-East (`E=0, N=π/2`).
   `simutils.altaz_to_zenithangle_azimuth(alt, az)` does the conversion;
   it returns `za = π/2 - alt` and `new_az = π/2 - az` (wrapped into
   `[0, 2π)`).

3. **Polarization order in the output `data_array`.** `[XX, YY, XY, YX]`
   (i.e. pyuvdata polarization numbers `[-5, -6, -7, -8]`). Cross-pols
   are always computed; you cannot select pol on input.

4. **Coherency factor.** `coherency_calc` in pyradiosky returns
   `(1/2) × [[I+Q, U-iV], [U+iV, I-Q]]`. `pyuvsim` multiplies the beam
   responses by this coherency *without* any further scale factor,
   so `V_XX + V_YY = I` (not `2I`). This matches RRIVis's documented
   convention.

5. **Frame.** All catalogs are processed in ICRS internally. If
   `SkyModel.frame != "icrs"`, `SkyModelData.__init__` calls
   `sky_in.transform_to(ICRS)`. Mock catalogs are built in AltAz at the
   given time/location, then transformed to ICRS so they "stay put"
   relative to the catalog frame as the array rotates under them.

6. **uvw convention.** `Baseline.uvw = enu` (the local ENU vector
   between antennas) — the simulator does not project onto a fixed phase
   center. The output `UVData.uvw_array` is rebuilt at the end via
   `set_uvws_from_antenna_positions(update_vis=False)`, the
   phase_center_catalog is replaced with `{0: {"cat_name":
   "unprojected", "cat_type": "unprojected"}}`. **You cannot use
   pyuvsim to simulate phased / tracked observations directly** — you
   would phase the result yourself afterward via `pyuvdata.UVData.phase`.

7. **Fringe.** `fringe = exp(2j*pi * (u·l + v·m + w·n))` where
   `(l, m, n)` are direction cosines from `pyradiosky.SkyModel.pos_lmn`
   (the (1, m, n) basis where l=East, m=North, n=Up?). The sign is
   explicitly **`+2π`**, matching the `e^{+iωt}` convention used in
   pyuvdata.

8. **Output BLT order.** Internally always `(time, baseline)`. The
   output is reordered back to whatever `obsparam.ordering.blt_order`
   asked for, after the simulation.

9. **Time precision.** The user *must* pass 64-bit floats for absolute
   times. The yaml dumper would otherwise lose precision; pyuvsim uses
   the `time_offset` mechanism to dodge this when round-tripping through
   yaml.

10. **Beam normalization.** UVBeams are forcibly peak-normalized in
    `BeamList.__post_init__`. Only `efield` beams are accepted by the
    simulator itself; only `(az_za, healpix)` pixel coordinate systems
    are supported. Basis vectors must be `(az, za)`-aligned.

11. **Horizon.** A coarse horizon cut is done up front (rough
    rise/set LST calculation with `horizon_buffer` padding) to drop
    sources that *never* rise. The exact, per-time, per-source cut is
    done inside the engine via `sources.above_horizon`.

12. **Frequency interpolation.** Default `cubic` (since 1.1.0; was
    linear before). Configurable via the `freq_interp_kind` global in
    the telescope config.

13. **Spline interpolation.** Default cubic in each direction
    (`{"kx": 3, "ky": 3}`). Override per telescope-config with
    `spline_interp_opts`.

---

## 13. How pyuvsim compares to other simulators (per the docs)

From `docs/comparison.rst`. pyuvsim is intentionally positioned as the
"slow but right" reference, and recommends faster alternatives for
production work.

- **vs PRISim.** PRISim is faster (vectorized), well-tested, supports
  diffuse + spectral cubes + tracking, and parallelizes over baselines /
  freqs / sky model. But: limited to total-intensity Stokes I (cross-
  feed terms approximated to zero); does not support non-identical
  antenna patterns; MPI overhead grows for long time axes.
- **vs FHD.** FHD (IDL) is optimized for smooth-spectrum sources and
  uses fast convolution; supports diffuse, discrete pointings, and
  non-identical patterns. But: proprietary IDL; convolution introduces
  aliasing/ringing visible at delay-spectrum levels relevant for 21cm
  cosmology; primary beam motion limited to snapshot size; slow for
  complex spectral structure. (pyuvsim's docs include a delay-spectrum
  comparison plot showing FHD has excess high-delay power.)
- **vs CASA.** CASA uses compiled C, well documented, established;
  supports OpenMP within a node and MPI across; supports component lists
  and FITS image source models. But: limited UVBeam support; an
  uncorrected internal UVW rotation bug (helpdesk ticket 2291); default
  fastest mode grids point sources to pixels (point-source subtraction
  errors); no full direction-dependent Jones for in-software simulation;
  no non-identical beams.

For a quantitative cross-comparison of pyuvsim vs PRISim vs FHD on
identical inputs, see the HERA memo PDF at
`docs/HERA comparison memo.pdf`.

---

## 14. Versioning policy

`generation.major.minor` (e.g. 1.4.2):

- **Generation** — combines multiple new physical effects and / or
  major computational improvements. Backed by unittests, internal model
  validation, and significant external comparison. Currently on
  generation 1.
- **Major** — adds new physical effect or major computational
  improvement; small number per release. Backed by unittests, internal
  model validation, and limited external comparison. Most recent: 1.4.0
  (2024-10-31, replaced internal analytic beams with pyuvdata's, shared-
  memory BeamList, antpair selection at construction).
- **Minor** — bug fixes and small improvements. Most recent: 1.4.2
  (2025-10-03, CI fix for PyPI publishing).

Two major-generation deprecation periods are guaranteed for any
breaking API change. CHANGELOG is `Keep-A-Changelog` style.

Selected highlights from the changelog (full history in `CHANGELOG.md`):

- **1.4.1 (2025-09-30)** — Reference sims overhauled and moved to
  entrypoint-based downloading via astropy cache; raised mins to python
  ≥ 3.11, pyuvdata ≥ 3.2.3, pyradiosky ≥ 1.1.0; added `non_nan` /
  `non_negative` source filters; added single-feed (one-pol) support;
  used new pyuvdata Telescope (`feed_array`, `feed_angle`, `mount_type`).
  Fixed several major bugs around: SkyModel-vs-SkyModelData passing;
  unphased measurement set writes; antennas with no data being
  initialized; freq/time array setup from start/end/spacing; precision
  loss in obsparam yaml round-trip; large memory for spectral_index /
  subband sims; `Jy/sr` diffuse map handling; freq_buffer beam selects.
- **1.4.0 (2024-10-31)** — Replaced internal analytic beams with
  pyuvdata's; major restructure of BeamList (shared memory + immutable
  + BeamInterface wrapping); `select.bls` now applied before UVData
  construction; minimum pyuvdata ≥ 3.1.0; minimum lunarsky ≥ 0.2.5.
- **1.3.1 (2024-07-18)** — Added `ordering` section to obsparam;
  pyuvdata 3.0+ support; default conjugation convention flipped to
  `ant1<ant2`; switched to `UVData.new` constructor.
- **1.3.0 (2024-04-02)** — Lunar simulation accuracy tests; UVBeam
  reads any pyuvdata-supported beam file (not just beamfits).
- **1.2.6 (2023-07-17)** — pyradiosky ≥ 0.2.0; lunarsky ≥ 0.2.1; fixes
  for shared-memory window leaks, skycoord support, lunarsky imports.
- **1.2.4 (2022-06-01)** — `beam_interp_check` parameter added; default
  is to skip the check for full-sky beams.
- **1.2.3 (2022-05-10)** — `return_beams` and `return_catname`
  parameters added; output history includes pyuvsim/pyradiosky/pyuvdata
  versions.
- **1.2.2 (2022-02-22)** — Support for `Nblts != Nbls * Ntimes` UVData.
- **1.2.0 (2020-07-20)** — Diffuse mock catalogs (`analytic_diffuse`);
  `SkyModelData` introduced; `quiet` keyword; lunar (`world: moon`)
  support; benchmarking tools.
- **1.1.0 (2019-06-14)** — Parallelized counter; shared-memory
  broadcast; `initialize_uvdata_from_keywords`; SkyModel replaces the
  old Source class; tasks split over (time, freq, baseline).
- **1.0.0 (2019-05-10)** — First stable release; comparisons documented.

---

## 15. End-to-end example: invoking pyuvsim

### 15.1 Minimal obsparam.yaml + telescope_config.yaml + layout.csv

`my_obsparam.yaml`:

```yaml
filing:
  outdir: "./out/"
  outfile_name: "mysim"
  output_format: "uvh5"
freq:
  Nfreqs: 100
  channel_width: 195312.5
  start_freq: 1.0e8
sources:
  catalog: "../gleam_50srcs.vot"
  spectral_type: "subband"
  min_flux: 0.2
telescope:
  array_layout: "layout.csv"
  telescope_config_name: "tele.yaml"
time:
  Ntimes: 10
  integration_time: 11.0
  start_time: 2460000.0
ordering:
  conjugation_convention: ant1<ant2
  blt_order: [time, baseline]
```

`tele.yaml`:

```yaml
telescope_name: HERA
telescope_location: (-30.72153, 21.42831, 1073.0)
beam_paths:
  0: !AnalyticBeam
    class: GaussianBeam
    diameter: 14
spline_interp_opts:
  kx: 4
  ky: 4
freq_interp_kind: cubic
```

`layout.csv`:

```
Name        Number   BeamID   E          N          U
HH0         0        0        0.0000     0.0000     0.0000
HH1         1        0        14.6000    0.0000     0.0000
HH2         2        0        7.3000     12.6438    0.0000
```

Run with N MPI ranks:

```bash
mpiexec -n 4 run_pyuvsim --param my_obsparam.yaml
```

### 15.2 Direct in-process invocation

```python
from pyuvdata import UVData, AnalyticBeam
from pyradiosky import SkyModel
from pyuvsim import BeamList, run_uvdata_uvsim

uvd     = UVData.from_file("template.uvh5")
beams   = BeamList([AnalyticBeam.AiryBeam(diameter=14.0)])
skymod  = SkyModel.from_file("gleam_50srcs.vot", spectral_type="subband")

uv_out = run_uvdata_uvsim(
    input_uv=uvd,
    beam_list=beams,
    beam_dict=None,                  # all antennas use beams[0]
    catalog=skymod,
    beam_interp_check=None,          # auto: off for full-sky beams
    quiet=False,
)
uv_out.write_uvh5("sim_results.uvh5")
```

(Still must be invoked under `mpiexec` to get parallelism;
`run_uvdata_uvsim` works on a single rank too.)

### 15.3 Building a config from data

```bash
# extract config from an existing uvh5 / uvfits / measurement set
uvdata_to_telescope_config some_data.uvh5 --beamfile hera.beamfits \
    --layout_csv_name out_layout.csv --telescope_config_name tele.yaml
uvdata_to_config some_data.uvh5 -p obsparam.yaml \
    -t tele.yaml -l out_layout.csv
```

The generated obsparam will need a `sources:` block populated by hand
(or with `catalog: mock` for a quick smoke test).

### 15.4 Reference simulations

```bash
# install GLEAM, MWA beam, and the GSM healpix map into the astropy cache
download_data_files                          # downloads all
download_data_files gleam mwa                # specific subset

# fetch reference simulation results from the Brown Digital Repository
download_ref_sims 1.1_baseline_number

# run a reference simulation locally
mpiexec -n 4 run_ref_sim 1.1_baseline_number
mpiexec -n 4 run_ref_sim 1                   # equivalent
```

### 15.5 Profiling

```bash
mpiexec -n 4 run_pyuvsim --param my_obsparam.yaml \
    --profile time_profile --raw_profile
```

Generates `time_profile.out` (human-readable per-line stats),
`time_profile.lprof` (pickled `LineStats`), and `time_profile_meta.out`
(axis sizes per local rank + global runtime + MaxRSS GiB + date/time).

---

## 16. Citation

```
Lanman, A. E., Hazelton, B. J., Jacobs, D. C., Kolopanis, M. J.,
Pober, J. C., Aguirre, J. E., & Thyagarajan, N. (2019).
pyuvsim: A comprehensive simulation package for radio interferometers
in python.
Journal of Open Source Software, 4(37), 1234.
https://doi.org/10.21105/joss.01234
```

ADS: `https://ui.adsabs.harvard.edu/abs/2019JOSS....4.1234L/abstract`

GitHub: `https://github.com/RadioAstronomySoftwareGroup/pyuvsim`

Documentation: `https://pyuvsim.readthedocs.io/`

Maintainers (RASG Managers): Adam Beardsley (ASU), Bryna Hazelton
(UW eScience), Daniel Jacobs (ASU), Paul La Plante (UC Berkeley),
Jonathan Pober (Brown). Contact: `rasgmanagers@gmail.com`.

License: 3-clause BSD (`LICENSE`).
