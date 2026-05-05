# PRISim — Precision Radio Interferometry Simulator

> Exhaustive reference notes on the vendored copy of `PRISim/` in
> `simulators/PRISim`. Compiled from a complete read of the source tree
> (5 main modules, 14 scripts, all example YAMLs, the full changelog).
> File:line citations are retained throughout so each claim can be
> spot-checked against the upstream source.

---

## 1. Project Identity

| Field | Value | Source |
|-------|-------|--------|
| Name | **PRISim** — Precision Radio Interferometry Simulator | `prisim/__init__.py:4` |
| Version | `2.2.1` | `prisim/__init__.py:3` |
| Author / maintainer | Nithyanandan Thyagarajan (`nithyanandan.t@gmail.com`) | `prisim/__init__.py:5–8` |
| Upstream URL | <http://github.com/nithyanandan/prisim> | `prisim/__init__.py:9` |
| License | MIT (Copyright © 2015 Nithyanandan Thyagarajan) | `LICENSE.txt` |
| Citation | DOI **10.5281/zenodo.2548116** (BibTeX) | `README.rst:107–110` |
| Data archive | DOI **10.5281/zenodo.3892047** (also Google Drive mirror) | `README.rst:55–61` |
| Python target | **Python 2.6+ only** (Py3 not supported by upstream README) | `README.rst:9`, `setup.py:39` |
| Status classifier | "Development Status :: 4 - Beta" | `setup.py:36` |
| Active period | ~8 years; v1.0 → v2.2.1 (≈577 commits) | `changelog.txt` |

> **Important for any modernisation:** `setup.py` pins legacy versions
> (`astropy>=1.0,<3.0`, `matplotlib>=1.4.3,<3.0`, `mpi4py>=1.2.2`,
> `pyephem`, `aipy`, …). It also depends on `astroutils` (an external
> package by the same author, installed from
> `git+git://github.com/nithyanandan/AstroUtils`) — almost every PRISim
> module imports submodules of `astroutils` for DSP, geometry,
> catalogues, gridding, lookups, etc. (`setup.py:60–76`,
> `prisim/interferometry.py:8,20–26`).

---

## 2. Installation & Runtime Dependencies

### 2.1 Required Python packages (`setup.py:60–76`)

```
astropy >=1.0, <3.0         astroutils (github:nithyanandan/AstroUtils)
healpy  >=1.5.3             ipdb >=0.6.1
matplotlib >=1.4.3, <3.0    mpi4py >=1.2.2
numpy   >=1.8.1             progressbar >=2.3
psutil  >=2.2.1             pyephem >=3.7.5.3
pyyaml  >=3.11              scipy >=0.15.1
h5py    >=2.6.0             pyuvdata >=1.1
gdown                       aipy
```

### 2.2 Optional non-Python dependencies (`README.rst:11–17`)

* `openmpi` — required to actually run the simulator in parallel
  (`mpirun -n N run_prisim.py`).
* `xterm`   — optional; enables `mpirun -n N xterm -e run_prisim.py`
  so each MPI rank gets its own progress window.

### 2.3 Installation paths (`README.rst:19–66`)

```
# Conda packages (recommended for OpenMPI & friends):
conda install mpi4py progressbar psutil pyyaml h5py astropy \
              matplotlib numpy scipy scikit-image

# PRISim itself:
pip install git+https://github.com/nithyanandan/PRISim
# or, locally:
pip install .
```

Package data downloaded via the helper script:

```
setup_prisim_data.py        # Google-Drive / Zenodo download via gdown
```

(Falls back to manual download into
`<env>/lib/python2.7/site-packages/prisim/data/`.)

### 2.4 MPI smoke-test (`README.rst:68–86`, `scripts/test_mpi4py_for_prisim.py`)

```
mpirun -n 2 test_mpi4py_for_prisim.py
```

The README warns that `mpi4py` should be installed via `conda` rather
than `pip`, otherwise paths to MPI libraries get tangled.

### 2.5 Basic run (`README.rst:88–101`)

```
mpirun -n nproc run_prisim.py -i parameterfile.yaml
mpirun -n nproc xterm -e run_prisim.py -i parameterfile.yaml
```

> **Disk size rule of thumb (`README.rst:102–105`):**
> output size ∝ `n_bl × nchan × n_acc`.

### 2.6 Distributed package data (`setup.py:44–56`, `MANIFEST.in`)

The `data/` directory shipped with the package contains:

* `data/catalogs/` — `*.txt`, `*.csv`, `*.fits` foreground catalogues
  (NVSS, SUMSS, MWACS, GLEAM, custom).
* `data/beams/`    — `*.hmap`, `*.hdf5`, `*.FITS`, `*.txt` beam files.
* `data/array_layouts/`        — antenna layouts.
* `data/phasedarray_layouts/`  — phased-array tile layouts (e.g.
  `MWA_tile_dipole_locations.txt`).
* `data/bandpass/` — PFB / bandpass tables.

A git hash is captured at `setup.py` time and written to
`prisim/githash.txt`, then re-read in `prisim/__init__.py:11–12` and
embedded in every output HDF5 (key `'PRISim#'`).

---

## 3. High-Level Architecture

```
        ┌────────────────────────────────────────────────┐
        │             scripts/run_prisim.py              │  (MPI entry point)
        │      reads YAML → drives full simulation       │
        └────────────────────────────────────────────────┘
                              │
                              ▼
   ┌────────────────────┬───────────────────────┬───────────────────────┐
   │ prisim/            │ prisim/               │ prisim/               │
   │ interferometry.py  │ primary_beams.py      │ baseline_delay_       │
   │ (≈9 900 lines)     │ (≈2 800 lines)        │   horizon.py          │
   │ InterferometerArr. │ Analytic + tabulated  │ Geometric delays,     │
   │ GainInfo, ROI,     │ beams (VLA/GMRT/MWA/  │ horizon delay limits, │
   │ ApertureSynth.,    │ HERA/HIRAX/PAPER),    │ ENU/(HA,Dec)/altaz/   │
   │ InterferometerData │ phased arrays         │ direction-cosine bases│
   └────────────────────┴───────────────────────┴───────────────────────┘
                              │
                              ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │ prisim/delay_spectrum.py   (≈4 700 lines)                          │
   │   DelaySpectrum, DelayPowerSpectrum                                │
   │   Hogbom CLEAN in delay domain; subband BHW/BNW windows;           │
   │   Cosmological k_∥, k_⊥ conversions (Planck15, h=1).               │
   └────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │ prisim/bispectrum_phase.py (≈4 700 lines)                          │
   │   ClosurePhase, ClosurePhaseDelaySpectrum                          │
   │   Subsample-differencing uncertainties, cross-power spectra,       │
   │   k-binned ∆² output, NPZ⇄HDF5 converters.                         │
   └────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │ scripts/  (post-processing & utilities)                            │
   │   write_PRISim_visibilities.py, prisim_to_uvfits.py,               │
   │   make_redundant_visibilities.py, replicate_sim.py,                │
   │   update_PRISim_noise.py, prisim_grep/ls/resource_monitor.py,      │
   │   FEKO_beam_to_healpix.py, altsim_interface.py (PyUVSim bridge),   │
   │   write_PRISim_bispectrum_phase_to_npz.py, …                       │
   └────────────────────────────────────────────────────────────────────┘
```

External packages PRISim *actually leans on at runtime*:

* `astroutils` (DSP, geometry, catalog, gridding, lookups, mathops,
  writer, constants).
* `aipy`     — used historically for some catalog / coordinate ops
  (still listed in `install_requires`).
* `pyuvdata` — UVData/UVBeam objects for UVFITS, UVH5, MIRIAD, MS,
  CASA exports (optional; guarded by
  `prisim/interferometry.py:32–38`).
* `mwapy.pb.primary_beam` — MWA-Tools advanced beam, optional
  (`prisim/interferometry.py:40–44`).
* `healpy`   — HEALPix maps for beams and diffuse sky.
* `mpi4py`   — drives parallelism in `run_prisim.py`.

---

## 4. Source Tree

```
PRISim/
├── README.rst
├── LICENSE.txt                 # MIT
├── MANIFEST.in
├── changelog.txt               # ~215 KB, ~7 000 lines, 577 commits
├── setup.py                    # Py2-era setuptools; loose pins
├── prisim/
│   ├── __init__.py             # version + githash
│   ├── interferometry.py       # 9 895 lines (564 KB)
│   ├── primary_beams.py        # ≈2 828 lines (152 KB)
│   ├── baseline_delay_horizon.py  # ≈241 lines (9 KB)
│   ├── delay_spectrum.py       # ≈4 700 lines (264 KB)
│   ├── bispectrum_phase.py     # ≈4 700 lines (308 KB)
│   ├── scriptUtils/
│   │   ├── __init__.py
│   │   ├── replicatesim_util.py
│   │   └── write_PRISim_bispectrum_phase_to_npz_util.py
│   └── examples/
│       ├── simparms/   defaultparms.yaml, defaultparms_dev.yaml,
│       │               replicatesim.yaml, noise_update_parms.yaml
│       ├── dbparms/    defaultdbparms.yaml
│       ├── pbparms/    FEKO_beam_to_healpix.yaml
│       ├── ioparms/    uvfitsparms.yaml,
│       │               data_setup_parms.yaml,
│       │               model_bispectrum_phase_to_npz_parms.yaml
│       ├── schedulers/ MWA_Aug23_obs_scheduler.txt
│       └── codes/      BispectrumPhase/ (notebooks + parm files),
│                        21cmforest/      (theory notebooks)
└── scripts/
    ├── run_prisim.py                          # MPI driver (122 KB!)
    ├── altsim_interface.py                    # PyUVSim → PRISim bridge
    ├── FEKO_beam_to_healpix.py                # FEKO → HEALPix
    ├── make_redundant_visibilities.py
    ├── prisim_grep.py / prisim_ls.py / prisim_resource_monitor.py
    ├── prisim_to_uvfits.py
    ├── replicate_sim.py
    ├── setup_prisim_data.py
    ├── test_mpi4py_for_prisim.py
    ├── update_PRISim_noise.py
    ├── write_PRISim_bispectrum_phase_to_npz.py
    └── write_PRISim_visibilities.py
```

---

## 5. Module Reference — `prisim/interferometry.py`

This is the central module: ≈9 900 lines, ≈564 KB, with no formal
top-of-file docstring (file begins straight at imports,
`interferometry.py:1`).

### 5.1 Imports & global state (`interferometry.py:1–46`)

* Standard: `ast`, `copy`, `datetime as DT`, `warnings`,
  `distutils.version.LooseVersion`.
* Numerical: `numpy as NP`, `scipy.constants as FCNST`,
  `scipy.interpolate`, `scipy.ndimage`, `progressbar as PGB`,
  `psutil`.
* Astronomy: `astropy.units / coordinates / io / time`, `h5py`,
  `astroutils.{DSP_modules, catalog, constants, geometry,
  gridding_modules, lookup_operations, nonmathops}`.
* PRISim local: `baseline_delay_horizon as DLY`,
  `primary_beams as PB`.
* Optional: `pyuvdata.UVData / utils` (sets `uvdata_module_found`)
  and `mwapy.pb.primary_beam as MWAPB` (sets `mwa_tools_found`).
* `prisim_path = prisim.__path__[0] + '/'` for resolving bundled
  data files (`interferometry.py:46`).

### 5.2 Module-level functions

| # | Function (line) | Signature highlights | Purpose |
|---|------|------|---------|
| 1 | `_astropy_columns(cols, tabtype='BinTableHDU')` (50) | — | Astropy ≥/< 0.4.2 compat shim for FITS columns. |
| 2 | `thermalNoiseRMS(...)` (91) | `A_eff, df, dt, Tsys, nbl=1, nchan=1, ntimes=1, flux_unit='Jy', eff_Q=1.0` | Computes thermal noise RMS per visibility. Implements **SIRA II eqs 9-12 … 9-15**. Returns array shaped `(nbl, nchan, ntimes)`; supports Jy and K. |
| 3 | `generateNoise(...)` (238) | `noiseRMS=None, A_eff=None, df=None, dt=None, Tsys=None, nbl=1, nchan=1, ntimes=1, flux_unit='Jy', eff_Q=None` | Draws complex Gaussian noise; falls back to `thermalNoiseRMS()` if `noiseRMS` is None. Real/imag each √2 of total. |
| 4 | `read_gaintable(gainsfile, axes_order=None)` (335) | — | Reads HDF5 antenna- *and* baseline-based complex gains. Returns nested dict with `'gains'`, `'ordering'`, `'label'`, `'frequency'`, `'time'`. |
| 5 | `extract_gains(gaintable, bl_labels, freq_index=None, time_index=None, polarization=None)` (637) | — | Pulls per-baseline gains for given freq/time indices; combines antenna-based as products. |
| 6 | `hexagon_generator(spacing, n_total=None, n_side=None, orientation=None, center=None)` (859) | — | Hex array generator. |
| 7 | `rectangle_generator(spacing, n_side, orientation=None, center=None)` (995) | — | Rectangular grid (`n_side = (nx, ny)`). |
| 8 | `circular_antenna_array(antsize, minR, maxR=None)` (1109) | — | Concentric ring layout (used by the `'CIRC'` array preset). |
| 9 | `baseline_generator(antenna_locations, ant_label=None, ant_id=None, redundant=None)` (1186) | — | All baseline vectors + label tuples; can filter to redundant or non-redundant. |
| 10 | `uniq_baselines(baseline_locations, redundant=None)` (1375) | — | Returns `(uniq_bl, uniq_indices, redundancy_count, inversion_indices)`. |
| 11 | `getBaselineInfo(inpdict)` (1467) | — | High-level wrapper: takes a dict describing layout / file / selection / redundancy, returns full baseline structure (the entry point used by `run_prisim.py`). |
| 12 | `getBaselineGroupKeys(inp_labels, blgroups_reversemap)` (2019) | — | For input baseline labels, return their redundancy-group keys. |
| 13 | `getBaselinesInGroups(inp_labels, blgroups_reversemap, blgroups)` (2102) | — | All redundant siblings of given baselines. |
| 14 | `antenna_power(skymodel, telescope_info, pointing_info, freq_scale=None)` (2171) | — | Thin wrapper over `primary_beams.primary_beam_generator()`. |

### 5.3 Class `GainInfo` (`interferometry.py:2414`)

Manages **antenna-based and baseline-based complex gains** with full
time/freq dependence and (linear or spline) interpolation.

* `gaintable` — nested dict: `{'antenna-based': {...}, 'baseline-based': {...}}`,
  each carrying `'gains'`, `'ordering'`, `'label'`, `'frequency'`,
  `'time'`.
* `interpfuncs`, `splinefuncs` — caches of scipy
  `interp1d`/`interp2d`/`UnivariateSpline`/`RectBivariateSpline`
  objects.
* Methods: `__init__(init_file=None, axes_order=None)`,
  `read_gaintable`, `eval_gains`, `interpolator`, `splinator`,
  `interpolate_gains`, `spline_gains`, `nearest_gains`,
  `write_gaintable(outfile, axes_order=None, ...)`.
* I/O: HDF5 read/write; uses
  `astroutils.lookup_operations.find_1NN`
  (`interferometry.py:3669,3681`) for nearest-neighbour fallback.

### 5.4 Class `ROI_parameters` (`interferometry.py:3870`)

Tracks **regions of interest** snapshot-by-snapshot — which sources
are above the horizon / inside the primary beam, what beam value
they have, and what pointing was used.

* `skymodel` — `astroutils.catalog.SkyModel` instance.
* `freq` — channel array; `freq_scale ∈ {'Hz','MHz','GHz'}`.
* `telescope` — element dict; supported IDs include `'mwa'`,
  `'mwa_dipole'`, `'mwa_tools'`, `'vla'`, `'gmrt'`, `'ugmrt'`,
  `'hera'`, `'hirax'`, `'paper'`, `'chime'`, plus `'custom'`. Keys:
  `'shape'` (`'dipole' | 'delta' | 'dish'`), `'size'`,
  `'orientation'`, `'ocoords'` (`'altaz' | 'dircos'`),
  `'element_locs'`, `'groundplane'`,
  `'ground_modify'={'scale','max'}`, `'latitude'`, `'longitude'`,
  `'altitude'`, `'pol'` (MWA only).
* `info` — list-per-snapshot of `'radius'`, `'center'`, `'ind'`,
  `'pbeam'`.
* `pinfo` — list-per-snapshot of `'gains'`, `'delays'`,
  `'pointing_center'`, `'pointing_coords'`, `'delayerr'`.
* Methods: `append_settings(skymodel, freq, pinfo=None, lst=None,
  ...)` and `save(outfile)` (FITS).

### 5.5 Class `InterferometerArray` (`interferometry.py:4729`)

The **central simulator object**: holds all baselines, channels,
visibilities, noise, gains, redundancy maps, pointing/phase data,
and exposes `observe()`, `observing_run()`, `delay_transform()`,
`getClosurePhase()`, `save()`, `pyuvdata_write()`.

#### 5.5.1 Geometry & timing

* `labels`              — dict `{'A1': ..., 'A2': ...}` per baseline.
* `baselines`           — `(M, 3)` baseline vectors (m).
* `baseline_coords`     — `'localenu'` (default) or `'equatorial'`.
* `baseline_lengths`    — `(M,)` magnitudes.
* `projected_baselines` — `(M, 3, n_snaps)` after phasing.
* `layout`              — `{'positions','coords','labels','ids'}`
  (always populated, even if subset used).
* `groups`,
  `bl_reversemap`       — redundancy maps.
* `latitude`, `longitude`, `altitude` — site (deg, deg, m).
* `lst`                 — list of LSTs (deg).
* `timestamp`           — Julian dates per snapshot.
* `t_acc`, `t_obs`, `n_acc` — per-accumulation times and counts.

#### 5.5.2 Spectral / visibility data

* `channels`, `freq_resolution` — Hz.
* `lags`, `lag_kernel` — IFFT axis & impulse response of bandpass.
* `bp`, `bp_wts` — bandpass × per-snapshot weights, `(M,nchan,n_acc)`.
* `A_eff`, `eff_Q` — collecting area / efficiency, scalar or
  `(M[, nchan, ntimes])`.
* `Tsys`               — `(M, nchan, n_acc)` system temperatures (K).
* `Tsysinfo`           — list-per-snapshot of
  `{'Trx','Tant':{'f0','T0','spindex'},'Tnet'}`.
* `skyvis_freq`        — noiseless visibility, complex Jy or K.
* `vis_freq`           — `skyvis_freq + vis_noise_freq`.
* `vis_noise_freq`     — thermal noise component.
* `vis_rms_freq`       — thermal noise RMS.
* `skyvis_lag`,
  `vis_lag`,
  `vis_noise_lag`      — IFFT to delay domain (Jy·Hz / K·Hz).

#### 5.5.3 Pointing & phasing

* `telescope` — element specs (same dict as in `ROI_parameters`).
* `pointing_center`,    `pointing_coords` — `'radec'`/`'hadec'`/`'altaz'`.
* `phase_center`,       `phase_center_coords` — independent of pointing.
* `skycoords`           — frame for sky model coords.
* `obs_catalog_indices` — list of source indices in ROI per snapshot.
* `gradient_mode`       — `None` or `'baseline'` (perturbation mode).
* `gradient`            — dict of visibility derivatives.
* `simparms_file`       — path to YAML used for the run.
* `astroutils_githash`,
  `prisim_githash`      — version stamps (also embedded in HDF5).

#### 5.5.4 Methods (selection)

| Method | Purpose |
|--------|---------|
| `__init__(labels, baselines, channels, ...)` | Construct from explicit arrays *or* from an HDF5/FITS file. |
| `observe(skymodel, roi_info, freq_scale=None, ...)` | Single-snapshot visibility evaluation. |
| `observing_run(skymodel, ...)` | Multi-snapshot extended run; supports `'drift'` / `'track'` modes. |
| `generate_noise()`, `add_noise()` | Thermal noise generation and attachment. |
| `apply_gradients(perturbation)` | Computes baseline-perturbation visibilities (`gradient_mode='baseline'`). |
| `duplicate_measurements()` | Expand unique baselines to full redundant set. |
| `phase_centering(phase_center, phase_center_coords, ...)` | Apply phase rotation only. |
| `project_baselines(phase_center, phase_center_coords, ...)` | Apply baseline projection only. |
| `rotate_visibilities(ref_point)` | Combined phase + projection. |
| `conjugate(bl_labels_to_conjugate)` | Reverse baseline & conjugate visibility. |
| `delay_transform()` | IFFT visibilities to delay domain. |
| `concatenate(other_ia, axis='baseline'|'frequency'|'time')` | Merge runs. |
| `getThreePointCombinations(baseline_labels=None)` | Triplets for closure phases. |
| `getClosurePhase(baseline_triplets)` | Returns noiseless / noisy / noise closure phases. |
| `save(outfile, fmt='hdf5'\|'fits'\|'npz'\|'uvfits', ...)` | HDF5 (preferred), FITS, NPZ, UVFITS. |
| `pyuvdata_write(outfile, **kwargs)` | UVFITS, UVH5, MIRIAD, MS via `pyuvdata`. |

### 5.6 Class `ApertureSynthesis` (`interferometry.py:8992`)

Computes UVW coverage in **wavelengths** for imaging.

* Input: an existing `InterferometerArray` (`ia`).
* Attributes: `uvw_lambda` (m), `uvw` (λ), `blc/trc`,
  `grid_blc/trc`, `gridu/v/w`, `grid_ready`.
* Methods: `genUVW()`, `reorderUVW()` (flatten to 3 × M·nchan·nacc),
  `setUVWgrid()` (regular grid for FFT imaging).

### 5.7 Class `InterferometerData` (`interferometry.py:9261`)

Adapter to **pyuvdata.UVData** for UVFITS/MS/UVH5/MIRIAD export.

* `infodict` — built from the `InterferometerArray`, with all
  pyuvdata-required arrays (`Ntimes/Nbls/Nblts/Nfreqs/Npols/Nspws`,
  `data_array={'noiseless','noisy','noise'}`, `uvw_array`,
  `time_array`, `lst_array`, `ant_1_array`, `ant_2_array`,
  `baseline_array`, `freq_array`, `polarization_array` (FITS code
  `-5` for XX), `integration_time`, etc.).
* `createUVData()` — assembles a `pyuvdata.UVData` instance.
* `write(outfile, fmt='uvfits')` — writes UVFITS by default; format
  string is forwarded to pyuvdata, so MS / UVH5 / MIRIAD work too
  if pyuvdata supports them in the installed version.

### 5.8 Coordinate, polarisation & unit conventions

* **Antenna positions**     — local **East-North-Up** (m).
* **Baselines**             — `'localenu'` by default; can be stored
  in equatorial XYZ via `astroutils.geometry.xyz2enu`.
* **Sky positions**         — `'altaz'`, `'hadec'`, `'radec'` (J2000
  by default), or **direction cosines** `(l, m, n)` with
  `l² + m² + n² = 1`.
* **Pointing & phase center frames** — independent fields; converted
  with `astroutils.geometry.{hadec2altaz, altaz2hadec, altaz2dircos,
  dircos2altaz, spherematch, sphdist}`.
* **LST is always required**: `'lst'` key must be present in
  `pointing_info`. LSTs are stored in **degrees**.
* **Polarisation** — current implementation is essentially
  single-pol; visibilities are tagged as XX in UVFITS via FITS code
  `-5` (`interferometry.py:9406`). MWA can be told `pol='X'` or
  `'Y'` to select dipole orientation
  (`interferometry.py:3980–3982`).
* **Visibility units** — `flux_unit ∈ {'JY','K'}`. Jy↔K conversion
  uses `2 k T / A_eff …` with `astroutils.constants.Jy`. Delay-domain
  visibilities are scaled by bandwidth, hence `Jy·Hz` or `K·Hz`.
* **Geometric delays** — `baseline_delay_horizon.geometric_delay`
  computes them from baselines + sky direction (see §7).

### 5.9 HDF5 output schema (`interferometry.py:8725–8854`)

Top-level groups written by `InterferometerArray.save(fmt='hdf5')`:

| Group | Datasets / attributes |
|-------|----------------------|
| `header` | `flux_unit`, `AstroUtils#`, `PRISim#` |
| `telescope_parms` | `latitude` (deg), `longitude` (deg), `altitude` (m), `id` |
| `spectral_info` | `freq_resolution` (Hz), `freqs` (Hz), `lags` (s) |
| `simparms` *(optional)* | inlined YAML used for the run |
| `antenna_element` | `shape`, `size` (m), `ocoords`, `orientation` (deg) |
| `layout` *(optional)* | `positions` (ENU, m), `labels`, `ids` |
| `timing` | `timestamp` (JD), `t_acc` (s) |
| `skyparms` | `LST` (deg), … |
| `array` | `baselines` (m, ENU/eq-XYZ), `baseline_lengths` (m) |
| `instrument` | `effective_area` (m²), `eff_Q`, `Trx/Tant0/f0/Tnet/Tsys` (K, Hz) |
| `visibilities/freq_spectrum` | `rms`, `vis`, `skyvis`, `noise` (Jy or K) |
| `visibilities/delay_spectrum` *(optional)* | `vis`, `skyvis`, `noise` (Jy·Hz) |
| `gradients` *(optional)* | `baseline` |
| `gaininfo` *(optional)* | full `GainInfo.gaintable` mirror |
| `blgroupinfo` *(optional)* | `groups`, `reversemap` |

### 5.10 Externally-called symbols (high-signal grep)

* `astroutils.geometry`: `hadec2altaz`, `altaz2hadec`, `altaz2dircos`,
  `dircos2altaz`, `spherematch`, `sphdist`, `xyz2enu`.
* `astroutils.catalog.SkyModel` — type-checked input.
* `astroutils.constants.Jy` — Jy↔K conversion constant.
* `astroutils.DSP_modules.FT1D(..., inverse=True)` — `lag_kernel`
  (`interferometry.py:5794`).
* `astroutils.lookup_operations.find_1NN` — gain interpolation
  fallback.
* `astroutils.nonmathops.find_list_in_list[...]` — many list ops.
* `baseline_delay_horizon.geometric_delay` —
  `interferometry.py:6167,6257`.
* `primary_beams.primary_beam_generator` —
  `interferometry.py:2402,4601–4615,6254`.
* `mwapy.pb.primary_beam.MWA_Tile_advanced` — guarded by
  `mwa_tools_found`.
* `pyuvdata.UVData`, `pyuvdata.utils`, `pyuvdata.__version__` —
  guarded by `uvdata_module_found`.

---

## 6. Module Reference — `prisim/primary_beams.py`

≈2 828 lines. Imports:

```
import numpy as NP
import scipy.constants as FCNST
import scipy.special  as SPS         # Bessel J1
import h5py                          # imported, presently unused
from astroutils import geometry as GEOM
```

### 6.1 Top-level dispatcher `primary_beam_generator()` (`primary_beams.py:9–441`)

```python
primary_beam_generator(
    skypos, frequency, telescope, freq_scale='GHz',
    skyunits='degrees', east2ax1=0.0,
    pointing_info=None, pointing_center=None,
    short_dipole_approx=False, half_wave_dipole_approx=False)
```

Routes by `telescope['id']`:

| `telescope['id']` | Implementation |
|------------------|----------------|
| `'vla'` | `VLA_primary_beam_PBCOR()` (AIPS PBCOR polynomial). |
| `'gmrt'`, `'ugmrt'` | `GMRT_primary_beam(instrument=…)` (4-band 4th-order polynomial). |
| `'hera'` | Airy disk, D = 14 m. |
| `'hirax'` | Airy disk, D = 6 m. |
| `'mwa'` | Dipole (0.74 m) × isotropic 4×4 array (1.1 m spacing) — or `array_field_pattern()` if `pointing_info` supplied. |
| `'mwa_dipole'` | Dipole (0.74 m) only. |
| `'paper'` | Dipole (2.0 m) only. |
| `'chime'` | Custom (delegated to user-supplied shape/size). |
| else (`'custom'`) | Custom shape ∈ `{'delta','dipole','dish','gaussian','rect','square'}`, optionally combined with `array_field_pattern()` and `ground_plane_field_pattern()`. |

Ground-plane handling (lines 418–439): if `telescope['groundplane']`
is set, the power pattern is multiplied by
`ground_plane_field_pattern(...)²`, optionally modified by
`telescope['ground_modify']={'scale','max'}`.

### 6.2 Component beams

| Function | What it does |
|----------|---------------|
| `VLA_primary_beam_PBCOR(skypos, frequency, skyunits='degrees')` (445) | AIPS PBCOR polynomial across 8 reference bands (74 MHz … 43 GHz). Validated to stay within 1 ± 0.01. |
| `airy_disk_pattern(diameter, skypos, frequency, ...)` (517) | Uniform circular aperture: `2 J₁(x)/x` with horizon cut-off (`x ≥ π/2` ⇒ 0). Supports off-zenith pointing via `GEOM.sphdist`. |
| `gaussian_beam(diameter, skypos, frequency, ...)` (629) | Diameter is FWHM. `σ_aperture = D/(2√(2 ln 2))/λ`; `σ_dircos = 1/(2π σ_aperture)`. |
| `GMRT_primary_beam(skypos, frequency, skyunits='degrees', instrument='gmrt')` (734) | 4th-order PBCOR-style polynomial for GMRT and uGMRT bands (235 / 325 / 610 / 1420 MHz). |
| `ground_plane_field_pattern(height, skypos, ..., modifier=None, power=True)` (812) | `2 sin(k h sin α)` with optional `1/√|cos zenith|` modifier and clip. |
| `dipole_field_pattern(length, skypos, ..., short_dipole_approx=False, half_wave_dipole_approx=True, power=True)` (975) | Three forms: short dipole `sin θ`, half-wave `cos(½π cos θ)/sin θ`, exact `[cos(kh cos θ) − cos(kh)]/sin θ`. L'Hôpital fix-up at θ = 0/π. Arbitrary 2-D / 3-D dipole orientation in altaz or dircos. |
| `isotropic_radiators_array_field_pattern(nax1, nax2, sep1, sep2=None, skypos=None, wavelength=1.0, east2ax1=None, skycoords='altaz', pointing_center=None, power=True)` (1239) | Rectangular array factor — used for MWA's 4 × 4 tile. |
| `array_field_pattern(antpos, skypos, skycoords='altaz', pointing_info=None, wavelength=1.0, power=True)` (1482) | Generic phased array. Geometric delay = `−(antpos · ŝ)/c`. Supports `pointing_info = {'gains','gainerr','delays','delayerr','pointing_center','pointing_coords','nrand'}`; `nrand` averages many error realisations. |
| `generic_aperture_field_pattern(elementpos, skypos, ...)` (1758) | Same idea as above but with frequency-dependent gains/delays and surface imperfections. |
| `uniform_rectangular_aperture(sides, skypos, frequency, ..., east2ax1=None, pointing_center=None, power=True)` (2057) | `sinc(sides[0]·l/λ) · sinc(sides[1]·m/λ)`. |
| `uniform_square_aperture(side, ...)` (2274) | Wrapper for square = `[side, side]`. |
| `feed_illumination_of_aperture(aperture_locs, feedinfo, wavelength=1.0, ...)` (2344) | Computes how a feed (dipole / dish / rect / square) illuminates sampled aperture points; returns N × nchan complex illumination map. |
| `feed_aperture_combined_field_pattern(aperture_locs, feedinfo, skypos, ...)` (2572) | Feed × aperture combined pattern: feed pattern toward sky × aperture pattern with feed-illumination as gains. |

### 6.3 Beamforming / `pointing_info` semantics (`primary_beams.py:128–175`)

| Key | Meaning | Notes |
|-----|---------|-------|
| `'delays'` | Per-element applied delay (s). | Phase = `2π (c/λ) × delay`. |
| `'pointing_center'` | Sky direction (overrides delays). | 2-elem altaz (deg) or 2-/3-elem dircos. |
| `'pointing_coords'` | `'altaz'` or `'dircos'`. | |
| `'gains'` | Complex per-element voltage gain. | |
| `'gainerr'` | RMS amplitude error in **dB**. | Converted via `10**(dB/10)`. |
| `'delayerr'` | RMS delay jitter (s). | Drawn from normal distribution. |
| `'nrand'` | Number of MC realisations. | Averaged into output. |

### 6.4 Notable conventions

* `east2ax1` (deg) — angle, **anticlockwise from East**, of the
  primary aperture axis.
* All horizon cut-offs apply: pixels with `altitude ≤ 0°` are
  zeroed (`primary_beams.py:584,607,616,691,714,725`).
* No Jones matrix; output is a scalar power (or field) pattern.
* No FITS / HEALPix loader inside this file (HEALPix conversion
  lives in `scripts/FEKO_beam_to_healpix.py`).

---

## 7. Module Reference — `prisim/baseline_delay_horizon.py`

Tiny but central. Three functions only.

### 7.1 `delay_envelope(bl, dircos, units='mks')` (lines 7–96)

Returns an `N × M × 2` matrix where:

* `[:, :, 0]` = maximum delay (no zenith shift).
* `[:, :, 1]` = delay shift from non-zenith phase center.

Min delay = `−(delay[:,:,1] + delay[:,:,0])`; effective max =
`delay[:,:,0] − delay[:,:,1]`.

### 7.2 `horizon_delay_limits(bl, dircos, units='mks')` (lines 100–129)

Convenience wrapper that returns `[:,:,0] = min` and `[:,:,1] = max`
delays for each baseline / phase-center pair. Used by
`DelaySpectrum.get_horizon_delay_limits` and
`DelayPowerSpectrum.horizon_kprll_limits` to draw the foreground
wedge boundary.

### 7.3 `geometric_delay(baselines, skypos, altaz=False, dircos=False, hadec=True, units='mks', latitude=None)` (lines 133–241)

Computes geometric delays for arbitrary baselines and sky positions.
Exactly **one** of `{altaz, dircos, hadec}` must be `True`. When
`hadec=True`, `latitude` is mandatory; `astroutils.geometry` is
used internally for frame conversions.

---

## 8. Module Reference — `prisim/delay_spectrum.py`

≈4 700 lines. Imports include `astropy.cosmology`, `h5py`, `healpy`,
`yaml`, `astropy.io.fits`, plus `astroutils.{DSP_modules, constants,
geometry, lookup_operations, mathops, writer_module}` and PRISim's
own `baseline_delay_horizon`, `interferometry`, `primary_beams`. An
optional `pyuvdata.UVBeam` import allows reading external beam
files.

Module-level cosmology constants (`delay_spectrum.py:37–38`):

* `cosmoPlanck15 = astropy.cosmology.Planck15`
* `cosmo100 = Planck15` cloned with `H0 = 100 km/s/Mpc` (the **h = 1**
  default used throughout PRISim's EoR work).

### 8.1 Module-level functions

| Function | Purpose |
|----------|---------|
| `_astropy_columns(cols, tabtype='BinTableHDU')` (42) | Astropy ≥/<0.4.2 FITS shim. |
| `complex1dClean_arg_splitter(args, **kwargs)` (133) | Helper for parallel CLEAN. |
| `complex1dClean(inp, kernel, cbox=None, gain=0.1, maxiter=10000, threshold=5e-3, threshold_type='relative', verbose=False, progressbar=False, pid=None, progressbar_yloc=0)` (136–357) | **Hogbom CLEAN on a complex 1-D delay axis.** Stops on threshold (relative or absolute), `maxiter`, or when `rms_inside < rms_outside` of the clean box. Returns `{'cc','res','termination','iter','rms','inrms','outrms'}`. |
| `dkprll_deta(redshift, cosmo=cosmo100)` (359) | Jacobian `d k∥ / d η = (2π × H(z) × f_HI × E(z)/c)/(1+z)²` in **h Mpc⁻¹ s⁻¹**. |
| `beam3Dvol(beam, freqs, freq_wts=None, hemisphere=True)` (398–492) | 3-D effective beam volume `Ω_eff · Δν` in Sr · Hz, integrating `|B|²` over sky × frequency. |

### 8.2 Class `DelaySpectrum` (`delay_spectrum.py:496`)

Holds the **delay-domain visibilities** alongside CLEAN components,
subband decompositions, and the bandpass / lag kernel. Constructed
from an `InterferometerArray` (or restored from FITS).

Key attributes:

* Frequency / lag axes — `f`, `df`, `lags`.
* `bp`, `bp_wts`, `pad`, `clean_window_buffer`.
* Visibilities — `skyvis_lag`, `vis_lag`, `vis_noise_lag`.
* CLEAN products — `cc_lags`, `cc_skyvis_lag`, `cc_skyvis_res_lag`,
  `cc_vis_lag`, `cc_vis_res_lag`, `cc_skyvis_net_lag`,
  `cc_skyvis_freq`, `cc_skyvis_res_freq`.
* Lag kernel — `lag_kernel`, `cc_lag_kernel`.
* Horizon — `horizon_delay_limits[ntimestamps, nbaselines, 2]`.
* Subband dictionaries — `subband_delay_spectra` and
  `subband_delay_spectra_resampled`, both keyed by `'cc'` /
  `'sim'`. Each holds `freq_center`, `freq_wts`, `bw_eff`, `shape`,
  `lags`, `lag_kernel`, `lag_corr_length`, `skyvis_lag`, `vis_lag`,
  `vis_noise_lag`, plus `residuals` for `'cc'`.

Important methods (line numbers):

| Method | What it does |
|--------|---------------|
| `__init__(interferometer_array, init_file)` (904–1224) | Build from `InterferometerArray` or load from FITS. |
| `delay_transform(pad, freq_wts, downsample, action, verbose)` (1227–1347) | IFFT freq → delay with apodisation and zero-padding. |
| `delay_transform_allruns(vis, pad, freq_wts, ...)` (1478–1622) | Batch transform of external visibility arrays. |
| `delayClean(pad, freq_wts, clean_window_buffer, gain, maxiter, threshold, threshold_type, parallel, nproc, verbose)` (1625–1842) | Iterative Hogbom CLEAN per `(baseline, time)`; multiprocessing-parallelised. |
| `subband_delay_transform(bw_eff, freq_center, shape, fftpow, pad, bpcorrect, action, verbose)` (1845–2252) | Multi-subband delay transform — `shape ∈ {'rect','bnw','bhw'}` and `bpcorrect` flattens the post-CLEAN bandpass. |
| `subband_delay_transform_allruns(vis, bw_eff, ...)` (2255–2518) | Batch subband transform. |
| `subband_delay_transform_closure_phase(bw_eff, cpinfo, ...)` (2521–2974) | Same on closure-phase data. |
| `get_horizon_delay_limits(phase_center, ...)` (2979–3033) | Computes wedge envelope per baseline / time. |
| `set_horizon_delay_limits()` (3037–3047) | Stores the envelope on `self`. |
| `save(ds_outfile, ia_outfile, tabtype, overwrite, verbose)` (3051–3259) | Writes a FITS file `<ds_outfile>.ds.fits` plus a companion `InterferometerArray` file. |

#### 8.2.1 Window functions / tapers

`astroutils.DSP_modules.window_fftpow(...)` is invoked with
`shape ∈ {'rect','RECT','bhw','BHW','bnw','BNW'}` (rectangular,
Blackman-Harris, Blackman-Nuttall). `fftpow` raises the window's
FFT to a chosen power (default 1) — affects spectral weighting and
correlation length of lag bins. Centering is on; output is
power-normalised (`delay_spectrum.py:2169`).

#### 8.2.2 CLEAN algorithm details

* Hogbom CLEAN; **no `aipy.deconv.clean`** — PRISim ships its own
  `complex1dClean()`.
* Stopping criteria (any one of):
  * peak < threshold × initial peak (`relative`) or < threshold
    (`absolute`),
  * iterations ≥ `maxiter` (default 10 000),
  * `rms_inside_box < rms_outside_box` (noise floor reached).
* Default gain 0.1; default threshold 5 × 10⁻³ relative.
* `clean_box` is set automatically from
  `horizon_delay_limits ± clean_window_buffer / bw`
  (`delay_spectrum.py:1766–1768,1798`).
* `delayClean(parallel=True, nproc=…)` uses `multiprocessing.Pool`
  to clean each `(baseline, time)` pair independently (default
  `nproc = min(ncpu-1, nbl × nt)`).

### 8.3 Class `DelayPowerSpectrum` (`delay_spectrum.py:3263`)

Converts a `DelaySpectrum` to a **cosmological power spectrum** in
`K² (Mpc/h)³`.

Cosmology / scaling attributes:

* `cosmo` (default `cosmo100`), `z`, `f0`, `wl0`, `bw`.
* Comoving distances — `drz_los`, `rz_los`, `rz_transverse`.
* k-modes — `kprll` (h/Mpc, from `dkprll_deta × η`),
  `kperp` = `2π · |b|/(λ₀ · r_⊥)`,
  `horizon_kprll_limits[nt, nbl, 2]`.
* Conversion factors — `jacobian1 = Ω_beam × bw`,
  `jacobian2 = r_los² × dr_los/bw`,
  `Jy2K = λ² · Jy / (2 k_B)` (and `K2Jy = 1/Jy2K`).
* `dps` — dict with keys `{'skyvis','vis','noise','cc_skyvis',
  'cc_vis','cc_skyvis_res','cc_vis_res','cc_skyvis_net',
  'cc_vis_net'}`, each `(nbl, nlags, nt)` in K² (Mpc/h)³.
* `subband_delay_power_spectra[key]` and
  `subband_delay_power_spectra_resampled[key]` — analogous to the
  delay-spectrum subband dicts, with `z`, `dz`, `kprll`, `kperp`,
  `horizon_kprll_limits`, `rz_los`, `rz_transverse`, `drz_los`,
  `jacobian1`, `jacobian2`, `Jy2K`, `factor`, …

Methods:

| Method | Purpose |
|--------|---------|
| `__init__(dspec, cosmo)` (3608–3681) | Bind a `DelaySpectrum`; default cosmology overridable. |
| `comoving_los_depth(bw, redshift, action)` (3685–3717) | `dr_los = (c/H₀) · bw · (1+z)² / (f_HI · E(z))`. |
| `comoving_transverse_distance(redshift, action)` (3721–3751) | Astropy's `comoving_transverse_distance`. |
| `comoving_los_distance(redshift, action)` (3755–3785) | Astropy's `comoving_distance`. |
| `k_parallel(lags, redshift, action)` (3789–3824) | `k∥ = dkprll_deta · η`. |
| `k_perp(baseline_length, redshift, action)` (3828–3863) | `k⊥ = 2π · |b|/(λ₀ · r_⊥)`. |
| `beam3Dvol(freq_wts, nside)` (3867–3981) | `Ω_eff · Δν` from a beam file (FITS, HDF5, UVBeam) or an analytic primary beam. |
| `compute_power_spectrum()` (3985–4062) | `P_k = |V_delay|² × jacobian1 × jacobian2 × Jy2K²`. |
| `compute_power_spectrum_allruns(dspec, subband)` (4070–4199) | Batched. |
| `compute_individual_closure_phase_power_spectrum(...)` (4202–4352) | Power spectrum of closure-phase amplitudes (assumes 1 Jy baselines). |
| `compute_averaged_closure_phase_power_spectrum(...)` (4355–4509) | Auto/cross-averaged closure-phase power. |

### 8.4 FITS output schema for `DelaySpectrum.save()`

* **Primary HDU** keywords: `NCHAN`, `NLAGS`, `freq_resolution`,
  `N_ACC`, `PAD`, `DBUFFER`, `IARRAY` (path to the companion IA
  FITS), `SBDS = 1` if any subband product is written, plus a
  `<key>-SBDS-…` keyword family for each subband pool (`'cc'` /
  `'sim'`).
* **Image HDUs** include: `FREQUENCIES`, `LAGS`, `CLEAN FREQUENCIES`,
  `CLEAN LAGS`, `HORIZON LIMITS`, `BANDPASS`,
  `BANDPASS WEIGHTS`, `LAG KERNEL REAL/IMAG`,
  `CLEAN LAG KERNEL REAL/IMAG`,
  `NOISELESS DELAY SPECTRA REAL/IMAG`,
  `NOISY DELAY SPECTRA REAL/IMAG`,
  `DELAY SPECTRA NOISE REAL/IMAG`,
  `CLEAN NOISELESS DELAY SPECTRA REAL/IMAG`,
  `CLEAN NOISELESS DELAY SPECTRA RESIDUALS REAL/IMAG`,
  `CLEAN NOISY DELAY SPECTRA REAL/IMAG`,
  `CLEAN NOISY DELAY SPECTRA RESIDUALS REAL/IMAG`,
  and equivalent `CLEAN ... VISIBILITIES REAL/IMAG`.
* **Subband HDUs** are templated as `{key}-SBDS-F0`,
  `{key}-SBDS-FWTS`, `{key}-SBDS-BWEFF`, `{key}-SBDS-LAGS`,
  `{key}-SBDS-LAGKERN-REAL/IMAG`, `{key}-SBDS-LAGCORR`,
  `{key}-SBDS-SKYVISLAG-REAL/IMAG`,
  `{key}-SBDS-VISLAG-REAL/IMAG`,
  plus `SKYVISRESLAG-` (cc only) and `NOISELAG-` (sim only).
  Resampled subbands use `{key}-SBDSRS-…` instead.

---

## 9. Module Reference — `prisim/bispectrum_phase.py`

≈4 700 lines, ~308 KB. The module name reflects an earlier rename
from `prisim/closure.py` — the change is documented in
`changelog.txt:2058`.

### 9.1 Imports

`copy`, `glob`, `warnings`, `functools.reduce`, plus
`astropy.cosmology`, `h5py`, `healpy`, `numpy`, `progressbar`,
`astroutils` submodules, and PRISim's `delay_spectrum (DS)`,
`interferometry (RI)`, `primary_beams (PB)`. Optional
`pyuvdata.UVBeam` is guarded by a `try/except`. Module constants
mirror `delay_spectrum`: `cosmoPlanck15` and `cosmo100`
(`bispectrum_phase.py:36–37`).

### 9.2 Module-level functions

| Function | Purpose |
|----------|---------|
| `write_PRISim_bispectrum_phase_to_npz(infile_prefix, outfile_prefix, triads=None, bltriplet=None, hdf5file_prefix=None, infmt='npz', datakey='noisy', blltol=0.1)` (41–251) | Reads PRISim NPZ/HDF5, calls `RI.InterferometerArray.getClosurePhase()` to extract closure phases for selected triads, then writes a consolidated NPZ with arrays `closures`, `triads`, `flags`, `last`, `days`. |
| `loadnpz(npzfile, longitude=0.0, latitude=0.0, lst_format='fracday')` (254–357) | Load externally-produced NPZ closure phases (e.g. from a CASA pipeline). Returns dict with `cphase (nlst,ndays,ntriads,nchan)`, `triads`, `flags`, `lst`, `lst-day`, `days`, optional `dayavg`, `std_triads`, `std_lst`. |
| `npz2hdf5(npzfile, hdf5file, longitude=0.0, latitude=0.0, lst_format='fracday')` (361–469) | Same content, written as HDF5. |
| `save_CPhase_cross_power_spectrum(xcpdps, outfile)` (472–630) | HDF5 dump of a closure-phase cross-power-spectrum dict (see §9.4 for the schema). |
| `read_CPhase_cross_power_spectrum(infile)` (633–803) | Inverse of the above. |
| `incoherent_cross_power_spectrum_average(xcpdps, excpdps=None, diagoffsets=None)` (807–1232) | Incoherently averages cross-power spectra across realisations and along diagonal offsets (LST/day/triad combinations). |
| `incoherent_kbin_averaging(xcpdps, kbins=None, num_kbins=None, kbintype='log')` (1236+) | Bin power spectra by k; returns linear `P(k)` and dimensionless `Δ²(k)`. |

### 9.3 Class `ClosurePhase` (`bispectrum_phase.py:1498–2273`)

Container around closure-phase data plus smoothing / model
subtraction / sub-sample differencing utilities.

* `extfile` — backing HDF5 path.
* `cpinfo` — nested dict (see §9.5 for layout).
* `f`, `df` — frequencies (Hz) and resolution.

Methods:

| Method (line) | Purpose |
|----------------|---------|
| `__init__(infile, freqs, infmt='npz')` (1606–1691) | Load from NPZ or HDF5. |
| `expicp(force_action=False)` (1695–1724) | Compute `e^{i ϕ}` as a masked array. |
| `smooth_in_tbins(daybinsize=None, ndaybins=None, lstbinsize=None)` (1728–1975) | Time-bin closure phases; populate `cpinfo['processed']['prelim']` with mean/median/RMS/MAD. |
| `subtract(cphase)` (1979–2020) | Subtract a model bispectrum phase; populate `'submodel'` and `'residual'`. |
| `subsample_differencing(daybinsize=None, ndaybins=4, lstbinsize=None)` (2024–2250) | Make 4-pair-of-pairs differences for empirical noise estimation; populate `cpinfo['errinfo']`. |
| `save(outfile=None)` (2254–2271) | Persist `cpinfo` to HDF5. |

### 9.4 Class `ClosurePhaseDelaySpectrum` (`bispectrum_phase.py:2275+`)

Does **delay transform → power spectrum → uncertainty** on closure
phases.

Attributes: `cPhase` (the wrapped `ClosurePhase`), `f`, `df`,
`cPhaseDS`, `cPhaseDS_resampled`.

Methods:

| Method (line) | Purpose |
|----------------|---------|
| `__init__(cPhase)` (2328–2346) | Bind a `ClosurePhase`. |
| `FT(bw_eff, freq_center=None, shape=None, fftpow=None, pad=None, datapool='prelim', visscaleinfo=None, method='fft', resample=True, apply_flags=True)` (2350–2788) | Subband Fourier transform of closure phases (and visibility scaling); produces oversampled and resampled lag spectra. |
| `subset(selection=None)` (2789–2885) | Index helper for triad / LST / day selection. |
| `compute_power_spectrum(cpds=None, selection=None, autoinfo=None, xinfo=None, cosmo=cosmo100, units='K', beamparms=None)` (2889–3605) | **Coherent** averaging via `autoinfo` (axes 1=LST, 2=days, 3=triads with weights), **incoherent cross-power** via `xinfo` (axes, collapse strategy, pre/post weights, `avgcov` flag). Output dict structured as in §9.6. |
| `compute_power_spectrum_uncertainty(...)` (3606–4361) | Same pipeline applied to subsample-difference arrays from `cpinfo['errinfo']`. |
| `rescale_power_spectrum(cpdps, visfile, blindex, visunits='Jy')` (4362–4495) | Scale by reference visibility amplitudes. |
| `average_rescaled_power_spectrum(rcpdps, avgax, kprll_llim=None)` (4496–4638) | Average rescaled spectra along chosen axes. |
| `beam3Dvol(beamparms, freq_wts=None)` (4639+) | Effective beam volume for the K conversion (mirrors `DelayPowerSpectrum.beam3Dvol`). |

### 9.5 `ClosurePhase.cpinfo` schema (verbatim)

```text
cpinfo = {
  'raw': {
    'cphase':  (nlst, ndays, ntriads, nchan),
    'triads':  (ntriads, 3),
    'flags':   (nlst, ndays, ntriads, nchan),
    'lst':     (nlst, ndays),
    'days':    (ndays,)
  },
  'processed': {
    'native': { 'cphase': masked, 'eicp': complex masked, 'wts': masked },
    'prelim': {
      'lstbins', 'dlstbins', 'daybins', 'diff_dbins',
      'wts': masked,
      'eicp':   {'mean','median'},
      'cphase': {'mean','median','rms','mad'}
    },
    'submodel':  { 'cphase', 'eicp' },
    'residual':  {
      'cphase': {'mean','median'},
      'eicp':   {'mean','median'}
    }
  },
  'errinfo': {
    'daybins', 'diff_dbins', 'lstbins', 'dlstbins',
    'list_of_pair_of_pairs': [[i,j,k,m], ...],
    'eicp_diff': { '0':{'mean','median'}, '1':{'mean','median'} },
    'wts':       { '0', '1' }
  }
}
```

### 9.6 `xcpdps` (cross-power spectrum) schema

```text
xcpdps = {
  'triads', 'triads_ind', 'lst', 'lst_ind', 'dlst',
  'days',   'day_ind',    'dday', 'lstXoffsets',
  'oversampled' | 'resampled': {
    'z', 'kprll', 'lags', 'freq_center', 'bw_eff',
    'shape', 'freq_wts', 'lag_corr_length',
    'whole'   : { 'mean','median','diagoffsets','diagweights',
                  'axesmap','nsamples_incoh','nsamples_coh' },
    'submodel': { ... },
    'residual': { ... },
    'errinfo' : { ... }
  }
}
```

`save_CPhase_cross_power_spectrum()` mirrors this structure as
HDF5; `read_CPhase_cross_power_spectrum()` reconstructs it.

### 9.7 Cosmology integration

* `z = f_HI / f_center − 1`, `dz` from bandwidth (line 3397/3398).
* `dkprll_deta = DS.dkprll_deta(z, cosmo=cosmo)` (line 3399).
* `kprll, rz_los, drz_los` filled in the same way as in
  `DelayPowerSpectrum`.
* `units='K'` or `'Jy'` selects K² or Jy² reporting (lines 3404–3415).

### 9.8 Polarisation

The module is polarisation-agnostic at this level — closure phases
are scalar derived quantities. Underlying multi-polarisation
support is whatever the producing `InterferometerArray.getClosurePhase`
returns (the call lives at `bispectrum_phase.py:212`).

### 9.9 I/O matrix

| Input  | NPZ (`loadnpz`), HDF5 (`RI.InterferometerArray(init_file=…)`), PRISim native via `getClosurePhase`. |
| Output | NPZ (`write_PRISim_bispectrum_phase_to_npz`), HDF5 (`ClosurePhase.save`, `save_CPhase_cross_power_spectrum`). |

---

## 10. Script Reference — `scripts/`

All scripts are installed (via `setup.py:58`,
`scripts=glob.glob('scripts/*.py')`) and become available on
`PATH`.

### 10.1 `run_prisim.py` — the MPI driver

122 KB of straight-line code. Uses `argparse`:

```
run_prisim.py
    -i / --infile  <path>   # YAML parameters
                            # default: examples/simparms/defaultparms.yaml
```

Top of file (`run_prisim.py:1–101`):

* Shebang `#!/python` (sic).
* Imports MPI, YAML, HDF5, astropy, scipy, matplotlib, numpy,
  psutil, plus PRISim modules (`interferometry`, `primary_beams`,
  `baseline_delay_horizon`, etc.).
* MPI initialised right at the top — `comm`, `rank`, `nproc`.
* Sidereal-day constant; `prisim_path` resolved from
  `prisim.__path__[0]`.
* Loads YAML, optionally pre-loads a `template:` YAML and merges
  hierarchically (up to ≈3 levels of nesting).

Then, lines 103–300 extract and validate parameters block by block:

* `telescope` (`Trx/Tant_*/Tsys/A_eff/eff_aprtr/eff_Q`),
* `array` layout / file / parser,
* `antenna` (shape, size, orientation),
* `phasedarray` (file, errors, `nrand`),
* `bandpass` (`freq`, `freq_resolution`, `nchan`,
  `pfb_method ∈ {'theoretical','empirical', null}`),
* `obsparm` (`obs_date`, `obs_mode`, `t_obs`, `n_acc`, `t_acc`),
* `pointing` (`drift_init` / `track_init`),
* `gains` (HDF5 path),
* `beam` (`use_external`, `file`, `filefmt ∈ {'hdf5','fits','uvbeam'}`,
  `pol`, `chromatic`, `select_freq`, `spec_interp`),
* `processing` (`gradient_mode ∈ {None, 'baseline'}`, `memuse`,
  `memavail`, `bpass_shape ∈ {'rect','bnw','bhw'}`, …),
* `flags`, `save_formats` (HDF5/FITS, NPZ, UVFITS, UVH5),
* `diagnosis` (resource monitor, refresh interval, post-run shell).

The remainder of the file actually executes the simulation — but
its size precludes a line-by-line tour here; the **YAML schema in
§11 is the user-facing contract**.

### 10.2 Other scripts (summary)

| Script | Purpose |
|--------|---------|
| `altsim_interface.py` | In-process bridge that takes a `pyuvsim` parameter dict and rewrites a PRISim parameter dict in place (file paths, lat/lon/alt, layout, beam, bandpass, obs times, phase center, sky model). |
| `FEKO_beam_to_healpix.py` | Reads FEKO text beam files (`freq, θ, φ, gain_dB`), interpolates onto a HEALPix grid (`spline`, `nearest`, or native `healpix` interpolation), writes HDF5/FITS. Driven by `examples/pbparms/FEKO_beam_to_healpix.yaml`. |
| `make_redundant_visibilities.py` | Expands unique baselines in an HDF5 sim back to the full redundant set (calls `simobj.duplicate_measurements(blgroups=…)`). CLI: `-s/--simfile`, `-o/--outfile`, `--outfmt {HDF5/UVFITS/UVH5}`, optional `-p/--parmsfile`, `-w/--wait`. |
| `prisim_grep.py` | Search PRISim simulation directories by parameter values from a YAML query (`grepBoolean/String/ScalarRange/Value` over the rootdir × project tree). |
| `prisim_ls.py` | Tabulate parameters across a glob of sim IDs; CSV/TSV output; `--change` reports only differing keys. |
| `prisim_resource_monitor.py` | Live `psutil` polling of supplied PIDs; CLI: `-p/--pids`, `-t/--tint` (default 2 s); clears the screen and prints CPU% + RSS. |
| `prisim_to_uvfits.py` | HDF5 → UVFITS export driven by `examples/ioparms/uvfitsparms.yaml`. CLI: `-p/--parmsfile`, `-v/--verbose`. Handles phase rotation to RA/Dec. |
| `replicate_sim.py` | Driver for `prisim.scriptUtils.replicatesim_util.replicate(parms)`; reads `examples/simparms/replicatesim.yaml`. Adds independent thermal-noise realisations to a base sim. |
| `setup_prisim_data.py` | Downloads the data bundle from Google Drive via `gdown`, extracts the tarball, optional cleanup. Driven by `examples/ioparms/data_setup_parms.yaml`. |
| `test_mpi4py_for_prisim.py` | Tiny MPI smoke-test: each rank prints its rank/size/hostname. |
| `update_PRISim_noise.py` | Recomputes thermal noise on an existing HDF5 sim using fresh `Trx/Tant/A_eff` etc. CLI: `-s, -p, -o, --outfmt, -n, -w`; defaults to `examples/simparms/noise_update_parms.yaml`. |
| `write_PRISim_bispectrum_phase_to_npz.py` | Driver for `prisim.scriptUtils.write_PRISim_bispectrum_phase_to_npz_util.write(parms)`; reads `examples/ioparms/model_bispectrum_phase_to_npz_parms.yaml`. |
| `write_PRISim_visibilities.py` | Saves an HDF5 sim into HDF5/UVFITS/UVH5 (multi-format permitted). CLI: `-s, -o, --outfmt, -p, -w`. Uses `simobj.pyuvdata_write()` for UVFITS/UVH5 and `simobj.save()` for HDF5; rotates to a RA/Dec phase center. |

### 10.3 `prisim/scriptUtils/` library

* `replicatesim_util.replicate(parms)` (lines 13–127) — reads HDF5
  or UVFITS, builds thermal-noise RMS via
  `RI.thermalNoiseRMS()`, generates `n_realize` Gaussian noise
  realisations (real/imag each `1/√2`, scaled by `n_avg`), writes
  combined NPZ (noiseless / noisy / noise) or per-realisation
  UVFITS files. Forces UVFITS output when input was UVFITS
  (`replicatesim_util.py:39–40`).
* `write_PRISim_bispectrum_phase_to_npz_util.write(parms)` (lines
  11–49) — thin wrapper over
  `bispectrum_phase.write_PRISim_bispectrum_phase_to_npz()`,
  consuming the YAML in
  `examples/ioparms/model_bispectrum_phase_to_npz_parms.yaml`.

---

## 11. Configuration File Reference

PRISim's user surface is YAML. All defaults below are taken
verbatim from `prisim/examples/`.

### 11.1 `examples/simparms/defaultparms.yaml` — master sim parameters (1 062 lines)

#### `preload`
* `template: null` — optional path to a YAML to merge under the user file.

#### `dirstruct`
* `rootdir: '/data3/t_nithyanandan/'`
* `project: 'prisim_test'`
* `simid: null` — null ⇒ current GMT timestamp.

#### `telescope`
* `label_prefix: ''`
* `id: 'custom'` — one of `mwa`, `vla`, `gmrt`, `ugmrt`, `hera`,
  `mwa_dipole`, `custom`, `paper`, `mwa_tools`, `hirax`.
* `latitude: -30.7224`, `longitude: +21.4278`, `altitude: 0.0`.
* `A_eff: 154`, `eff_aprtr: 0.65`, `eff_Q: 0.96`.
* `Trx: 50.0`, `Tant_freqref: 150_000_000.0`, `Tant_spindex: -2.55`,
  `Tant_ref: 200.0`.
* `Tsys: null` (computed from `Trx + Tant` if null).

#### `array`
* `redundant: true`
* `layout: 'HERA-19'` (or `null` to use `file:`).
* `file: null`; `filepathtype: 'default' | 'custom'`.
* `parser`: `comment, delimiter, data_start (3), data_end,
  header_start (0), label, east 'East', north 'North', up 'Up'`.
* `minR: 141.0`, `maxR: 141.0` (CIRC layout only).
* `rms_tgtplane: 0.0`, `rms_elevation: 0.0`, `seed: 200`.

#### `baseline`
* `min: null`, `max: null`, `direction: null` (or `'E'/'SE'/'NE'/'N'`).

#### `antenna`
* `shape: 'dish'` (`dish`, `dipole`, `gaussian`, `delta`, `null`).
* `size: 14.0` (m).
* `orientation: [90.0, 270.0]`, `ocoords: 'altaz' | 'dircos'`.
* `phased_array: false`, `ground_plane: null`.

#### `phasedarray`
* `file: 'MWA_tile_dipole_locations.txt'`,
  `filepathtype: 'default'`.
* `delayerr: 0.0` (ns), `gainerr: 0.0` (dB), `nrand: 1`.

#### `beam`
* `use_external: false`,
  `file: 'NF_HERA_antenna_power_pattern_99-201_MHz_nside_128.uvbeam'`,
  `filepathtype: 'default'`,
  `filefmt: 'UVBeam' | 'HDF5' | 'FITS'`.
* `identifier: 'NF-128'`, `pol: 'X' | 'Y' | 'P1' | 'P2'`,
  `chromatic: true`, `select_freq: 150_000_000.0`,
  `spec_interp: 'fft'|'linear'|'bilinear'|'cubic'`.

#### `bandpass`
* `freq: 150_000_000.0`, `freq_resolution: 390_625.0` (Hz),
  `nchan: 256`.
* `pfb_method: null|'theoretical'|'empirical'`,
  `pfb_filepath: 'default'`,
  `pfb_file: 'MWA_pfb_512x8.fits'`.

#### `obsparm`
* `obs_date: '2015/11/23'`, `obs_mode: 'drift'|'track'|'dns'|'lstbin'|'custom'|null`.
* `t_obs: null`, `n_acc: 2`, `t_acc: 1080.0`.

#### `gains`
* `file: null`, `filepathtype: 'default'`.

#### `pointing`
* `file: null`, `jd_init: null`, `lst_init: 0.0`.
* `drift_init: { alt: null, az: null, ha: 0.0, dec: -30.7224 }`.
* `track_init: { ra: 0.0, dec: -30.7224, ha: 0.0, epoch: '2000' }`.

#### `phasing`
* `center: [90.0, 270.0]`,
  `coords: 'altaz'|'hadec'|'radec'|'dircos'`.

#### `snapshot`
* `avg_drifts: false`, `beam_switch: false`.
* `pick: null`, `range: null`, `all: true`.

#### `skyparm`
* `model: 'csm'` (also `dsm`, `asm`, `gsm2008`, `gsm2016`, `sumss`,
  `nvss`, `mss`, `gleam`, `custom`, `usm`, `noise`, `mwacs`,
  `skymod_file`, `HI_monopole`, `HI_cube`, `HI_fluctuations`).
* `fsky: null`, `epoch: '2000'`, `nside: null`,
  `n_mdl_freqs: 8`, `parallel: false`, `flux_unit: 'K'|'Jy'`,
  `custom_reffreq: 0.150` (GHz), `flux_min: 10.0`, `flux_max: null`,
  `fluxcut_reffreq: null`, `spindex: -0.83`, `spindex_rms: 0.0`,
  `spindex_seed: null`, `roi_radius: null`, `lidz: true`,
  `21cmfast: false`,
  `global_EoR_parms: [0.027, 150e6, 1.0]` (`T_spin`, frequency at
  `x_i = 0.5`, redshift width).

#### `catalog`
* `filepathtype: 'default'`.
* `DSM_file_prefix: 'gsmdata'`,
  `spectrum_file: '/data3/t_nithyanandan/project_abscal/newGSM.hdf5'`.
* Catalog files: `SUMSS_file: 'sumsscat.Mar-11-2008.txt'`,
  `NVSS_file: 'NVSS_catalog.fits'`,
  `MWACS_file: 'mwacs_b1_131016.csv'`,
  `GLEAM_file: 'GLEAM_EGC_v2.fits'`,
  `custom_file: 'custom_catalog.txt'`,
  `skymod_file: '/path/to/skymodel.hdf5'`.

#### `processing`
* `gradient_mode: null|'baseline'`, `memuse: null`, `memavail: null`.
* `n_bins_blo: 4`, `n_sky_sectors: 1`,
  `bpass_shape: 'bhw'|'bnw'|'rect'`,
  `ant_bpass_file: null`, `f_pad: 1.0`,
  `coarse_channel_width: 16`, `bp_correct: true`,
  `noise_bp_correct: false`, `n_pad: 0`,
  `max_abs_delay: 1.0` (μs), `delay_transform: false`,
  `memsave: false`, `store_prev_sky: true`, `cleanup: 3`.

#### `pp` (parallel processing)
* `key: 'freq'|'bl'|'src'`, `eqvol: true`,
  `method: 'pool'|'queue'`.

#### `flags`
* `flag_chan: -1`, `bp_flag_repeat: false`,
  `n_edge_flag: [0, 0]`, `flag_repeat_edge_channels: false`.

#### `save_redundant`
* `true` — expand and save the full redundant set.

#### `save_formats`
* `fmt: 'HDF5'|'FITS'`, `npz: true`, `uvfits: true`, `uvh5: true`,
  `uvfits_method: null|'uvdata'|'uvfits'`,
  `phase_center: null` (else `[ra, dec]` in deg).

#### `plots`
* `false`.

#### `diagnosis`
* `resource_monitor: false`, `refresh_interval: null` (default 2 s),
  `wait_after_run: true`.

### 11.2 `examples/simparms/defaultparms_dev.yaml` — diff against defaults

* `telescope.latitude: -26.701`, `longitude: +116.670815`
  (MWA site).
* `telescope.altitude: 0` (int vs 0.0).
* `array.layout: 'HERA-19'` (same; comment narrowed).
* `beam.pol`: comments narrowed to `'X' | 'Y'` only.
* `skyparm.nside: 256` (was `null`).
* `skyparm.n_mdl_freqs` and `skyparm.parallel` **omitted**.
* `catalog.spectrum_file` and `catalog.DSM_file_prefix` **omitted**.
* `processing.store_prev_sky` not present.
* `diagnosis.resource_monitor: true`.

### 11.3 `examples/simparms/noise_update_parms.yaml`

```
telescope:
  A_eff: 154.0
  eff_aprtr: 0.65
  eff_Q: 0.96
  Trx: 162.0          # commented alternative: 50.0
  Tant_freqref: 150_000_000.0
  Tant_spindex: -2.55
  Tant_ref: 200.0
  Tsys: null
```

### 11.4 `examples/simparms/replicatesim.yaml`

* `dirstruct`: `indir`, `infile`, `infmt: 'hdf5'|'uvfits'`,
  `outdir`, `outfile: 'simvis'`, `outfmt: 'npz'|'uvfits'`.
* `telescope`: same eight keys as above (`Trx: 162.0` default).
* `replicate`: `n_avg: 1`, `n_realize: 1`, `seed: 100` (null = random).
* `diagnosis`: `wait_before_run: false`, `wait_after_run: false`.

### 11.5 `examples/dbparms/defaultdbparms.yaml`

A **search/filter** schema: each scalar becomes `[min, max]` and
each list becomes a membership filter. Used by `prisim_grep.py` to
walk a tree of simulations and select runs whose recorded
parameters fall in the requested ranges. Sections mirror
`defaultparms.yaml` (`dirstruct`, `telescope`, `array`,
`baseline`, `antenna`, `phasedarray`, `beam`, `bandpass`,
`obsparm` (with extra `timeformat`), `pointing` (with extra
`drift_init.lst`), `phasing`, `fgparm` (legacy alias of
`skyparm`), `processing` (with `n_bl_chunks`, `bl_chunk_size`,
`n_freq_chunks`, `freq_chunk_size`), `pp`, `flags`,
`save_formats`).

### 11.6 `examples/pbparms/FEKO_beam_to_healpix.yaml`

* `io.indir / outdir`,
  `infmt: 'FEKO'`, `p1infile`, `p2infile`,
  `outfmt: 'HDF5'` (or FITS), `outfile`.
* `processing`: `is_grid: false`, `nside: 32`,
  `gainunit_in: 'dB'`, `gainunit_out: 'dB'`,
  `interp: 'spline'|'healpix'|'nearest'`, `wait: true`.
* `misc.source: 'somename'`.

### 11.7 `examples/ioparms/uvfitsparms.yaml`

```
infile:        '/path/to/prisim_file'   # extension auto-added
outfile:       '/path/to/output/uvfits_file'
overwrite:     true
uvfits_method: null                     # 'uvdata' | 'uvfits' | null (auto)
phase_center:  [0.0, -30.7224]          # [ra_deg, dec_deg]
```

### 11.8 `examples/ioparms/data_setup_parms.yaml`

* `download`: `action: true`,
  `url: 'https://drive.google.com/uc?id='`,
  `fid: '1KNBk6VhlY_rKSfgn8HmAncLkYQ1KGAOi'`, `fname: null`.
* `extract`: `action: true`, `fname: null`, `dir: null`.
* `cleanup`: `action: true`, `fname: null`.
* `verbose: true`.

### 11.9 `examples/ioparms/model_bispectrum_phase_to_npz_parms.yaml`

* `dirStruct`: `indir`, `infile_prfx: 'simvis'`,
  `infmt: 'npz'|'hdf5'`, `prisim_dir`, `simfile_prfx: 'simvis'`,
  `outdir`, `outfile_prfx`.
* `proc`:
  * `datakey: ['noisy']` (subset of `'noiseless'/'noisy'/'noise'`),
  * `triads`: explicit triplet list (default = HERA-19 29.2 m
    equilateral triads),
  * `bltriplet`: reference baseline triplet vector for triad
    selection (`[[29.2,0,0],[-14.6,-25.287942,0],
    [-14.6, 25.287942, 0]]`),
  * `blltol: 2.0` m.

### 11.10 `examples/schedulers/MWA_Aug23_obs_scheduler.txt`

CSV-ish, comma-separated, 5 columns:

```
# obsid (GPS s), beam elevation [deg], beam azimuth [deg], lst [hours], delay settings
1061306176,52.806,101.31,21.183,0;5;10;15;1;6;11;16;2;7;12;17;3;8;13;18
1061306296,52.806,101.31,21.217,0;5;10;15;1;6;11;16;2;7;12;17;3;8;13;18
…
```

`delay_settings` is a 16-int semicolon list — the MWA dipole delay
indices that select a beam pointing.

### 11.11 `examples/codes/`

* `BispectrumPhase/` — IPython notebooks (e.g.
  `understanding_closure_phases*.ipynb`,
  `multiday_EQ28*_closure_PS_analysis.ipynb`,
  `combine_pol_*`) with their YAML companions; example end-to-end
  closure-phase analyses.
* `21cmforest/` — theory notebooks
  (`stats_analysis_theory.ipynb` + parm YAML) for the 21-cm
  forest signal.

---

## 12. Telescope, Beam, and Catalog Support Matrix

### 12.1 Telescope IDs (all entries supported by
`primary_beams.primary_beam_generator` and accepted by
`run_prisim.py` via `telescope.id`)

| ID | Element model | Notes |
|----|---------------|-------|
| `vla` | AIPS PBCOR polynomial | 8 reference bands. |
| `gmrt` | 4th-order PBCOR-style polynomial | 235 / 325 / 610 / 1420 MHz. |
| `ugmrt` | Same form, new coefficients | 325 / 610 / 1420 MHz (NaN at 235). |
| `hera` | Airy disk, D = 14 m | External UVBeam files supported. |
| `hirax` | Airy disk, D = 6 m |  |
| `mwa` | Dipole 0.74 m × 4×4 isotropic array (1.1 m) | Switchable to `array_field_pattern()` if `pointing_info` provided. Optional `mwapy.pb.MWA_Tile_advanced` if installed. |
| `mwa_dipole` | Dipole 0.74 m only |  |
| `paper` | Dipole 2.0 m only | PAPER-64/128/112 layouts in `data/`. |
| `chime` | Custom shape/size delegated |  |
| `custom` | Pick from `dipole`, `dish`, `gaussian`, `rect`, `square`, `delta` | Optionally combine with phased-array element layout and/or ground-plane height. |

### 12.2 Catalog/sky models supported (`skyparm.model`)

`csm` (compact source model), `dsm` (diffuse), `asm` (all-sky),
`gsm2008`, `gsm2016`, `sumss`, `nvss`, `mss`, `gleam`, `mwacs`,
`custom`, `usm` (unresolved diffuse), `noise`, `skymod_file`,
`HI_monopole`, `HI_cube`, `HI_fluctuations`.

### 12.3 Output formats matrix

| Format | Producing class / method | Notes |
|--------|--------------------------|-------|
| HDF5 (PRISim native) | `InterferometerArray.save(fmt='hdf5')`, `GainInfo.write_gaintable`, `ClosurePhase.save`, `save_CPhase_cross_power_spectrum` | Schema in §5.9, §9.5/9.6, §8.4. |
| FITS | `InterferometerArray.save(fmt='fits')`, `ROI_parameters.save`, `DelaySpectrum.save` | DelaySpectrum FITS schema in §8.4. |
| NPZ | `InterferometerArray.save(fmt='npz')`, replicate sim, bispectrum-phase exporter | Combined arrays. |
| UVFITS | `InterferometerArray.save(fmt='uvfits')`, `InterferometerData.write(fmt='uvfits')`, `prisim_to_uvfits.py` | Polarisation `-5` (XX). |
| UVH5 | `pyuvdata_write` | Optional. |
| MIRIAD / MS | via `pyuvdata` if installed | Through `InterferometerData`. |

---

## 13. The RIME, Conventions, and Numerics

* **Visibility model** is built around per-snapshot evaluation:
  `InterferometerArray.observe()` integrates the sky model
  (already restricted to the ROI by `ROI_parameters`) against the
  primary beam at each LST and produces `skyvis_freq`, then
  `add_noise()` produces `vis_freq = skyvis_freq + vis_noise_freq`.
* **Polarisation** — single-pol XX in current outputs (FITS code
  `-5`). MWA can pick `'X'` or `'Y'`.
* **Coordinate frames**: ENU (antennas, baselines), HA-Dec, RA-Dec
  (J2000), Alt-Az, direction cosines `(l, m, n)`. Conversions via
  `astroutils.geometry`.
* **LST** is in **degrees** throughout the codebase.
* **Time** stored as Julian Date (`obsparm.timeformat='JD'`).
* **Ephemeris** moved from `pyephem` → `astropy.coordinates` /
  `astropy.time` over the v2 series (changelog ≈ lines 1492–1523).
* **Thermal noise**: `thermalNoiseRMS` follows SIRA II eqs 9-12 …
  9-15. Real and imaginary parts each carry `1/√2` of the total.
* **Delay convention**: `delay_transform()` uses **IFFT** (sign
  flipped from the original v1.0 FFT convention; see changelog
  line 6731–6734).
* **Cosmology**: `astropy.cosmology.Planck15` and the **h = 1
  variant `cosmo100`** (`H₀ = 100 km s⁻¹ Mpc⁻¹`). All `k`-modes
  reported in **h Mpc⁻¹**, comoving distances in **Mpc h⁻¹**, and
  cosmological power in **K² (Mpc/h)³**.
* **HI rest frequency**: 1420.405751768 MHz, used implicitly to
  map `f → z = f_HI/f − 1`.

---

## 14. Parallelism

* **MPI** (via `mpi4py`) — driven by `run_prisim.py`. The
  `processing.pp` block in YAML controls split key (`'freq'`,
  `'bl'`, `'src'`), equal-volume balancing, and method
  (`'pool'` vs `'queue'`).
* **Multiprocessing.Pool** — used inside `delay_spectrum.delayClean`
  for parallel CLEAN over `(baseline, time)` pairs (default
  `min(ncpu-1, nbl × nt)` workers).
* **Memory awareness** — `processing.memuse / memavail`, and
  per-process virtual-memory checks via `psutil`. `MemoryError`
  is caught and reported.
* **Live monitoring** — `prisim_resource_monitor.py` polls a list
  of PIDs (`-p`) at a configurable interval (`-t`, default 2 s)
  and prints CPU% and resident memory.

---

## 15. Typical Workflows

### 15.1 Single end-to-end simulation

```
# 1. install + data
pip install .
setup_prisim_data.py

# 2. edit a copy of defaults
cp prisim/examples/simparms/defaultparms.yaml my_sim.yaml
$EDITOR my_sim.yaml

# 3. run on N MPI ranks
mpirun -n N run_prisim.py -i my_sim.yaml
```

Outputs land under `<rootdir>/<project>/<simid>/` as PRISim HDF5
plus optional NPZ/UVFITS/UVH5 (per `save_formats`).

### 15.2 Add additional noise realisations

```
$EDITOR replicatesim.yaml
replicate_sim.py -i replicatesim.yaml
```

### 15.3 Refresh thermal noise on an existing sim

```
update_PRISim_noise.py \
    -s old_sim.hdf5 -p simparms.yaml -o new_sim.hdf5 \
    --outfmt HDF5 -n noise_update_parms.yaml
```

### 15.4 Convert to UVFITS / UVH5 for downstream tools

```
write_PRISim_visibilities.py \
    -s sim.hdf5 -o sim_export --outfmt UVFITS --outfmt UVH5 \
    -p simparms.yaml
# or:
prisim_to_uvfits.py -p uvfitsparms.yaml -v
```

### 15.5 Closure-phase / bispectrum analysis

```
write_PRISim_bispectrum_phase_to_npz.py \
    -i model_bispectrum_phase_to_npz_parms.yaml
# Then in Python:
from prisim.bispectrum_phase import (
    ClosurePhase, ClosurePhaseDelaySpectrum,
    save_CPhase_cross_power_spectrum,
    incoherent_cross_power_spectrum_average,
    incoherent_kbin_averaging,
)
cp  = ClosurePhase('closures.npz', freqs, infmt='npz')
cp.smooth_in_tbins(daybinsize=…); cp.subsample_differencing(ndaybins=4)
cpds = ClosurePhaseDelaySpectrum(cp)
cpds.FT(bw_eff=…, freq_center=…, shape='bhw')
xcpdps = cpds.compute_power_spectrum(autoinfo=…, xinfo=…, units='K')
save_CPhase_cross_power_spectrum(xcpdps, 'cpps.h5')
```

### 15.6 Derive a HEALPix beam from FEKO output

```
$EDITOR FEKO_beam_to_healpix.yaml
FEKO_beam_to_healpix.py            # picks up the YAML by default
```

### 15.7 Search a tree of simulations

```
prisim_grep.py    -i query.yaml         # filter by parameter ranges
prisim_ls.py      --project HERA --simid sim* --change --format csv
prisim_resource_monitor.py -p 1234 5678 -t 2
```

---

## 16. History & Feature Evolution (digest of `changelog.txt`)

The full change-log is ≈ 7 000 lines / 215 KB and spans the lifetime
of the package (v1.0 → v2.2.1, ≈ 577 commits, ~8 years). High-signal
events:

### v1.0 → v1.06 — foundations
* v1.0: original `Interferometer` (single baseline), `save()` (FITS),
  `band_averaged_noise_estimate()`, drift/tracking observing run.
* v1.02: bandpass kept independent of sky visibilities, `bp_wts`
  added, vectorised `delay_transform()` with padding.
* v1.03: load-from-file initialisation, baseline labels saved as
  binary table extensions.
* v1.04: **`InterferometerArray` introduced** (replacing the
  single-baseline class), `generate_noise()` / `add_noise()`,
  delay transform switched to IFFT, `phase_centering()`,
  `phase_center` / `phase_center_coords` attributes.
* v1.05–1.06: `airy_disk_pattern()` added, `'hera'` telescope ID
  introduced, `lag_psf` attribute, `baseline_generator()` accepts
  `ant_id`, `phase_center` terminology renamed to `pointing_center`.
* v1.07: `project_baselines()` and `projected_baselines` attribute.

### v2.x — feature consolidation
* `GainInfo` class with HDF5/FITS persistence, antenna- and
  baseline-based gains, time/frequency interpolation
  (linear + spline), nearest-neighbour fallback.
* New layouts and elements: `'paper'` (PAPER-64/128 → PAPER-112),
  `'hirax'`, `'chime'`, plus per-telescope element shape/size
  defaults.
* Foreground catalogue work: GLEAM updates, NVSS, custom catalogue
  with reference frequency; flux-cut thresholds.

### Closure-phase / bispectrum era
* `prisim/closure.py` introduced (initial `ClosurePhase`,
  `ClosurePhaseDelaySpectrum.FT`).
* Subsample differencing for empirical noise estimation; new output
  keys (`triads`, `triads_ind`, `lst`, `lst_ind`, `dlst`).
* **Module renamed `closure.py` → `bispectrum_phase.py`** (changelog
  line 2058).
* New helpers: `write_PRISim_bispectrum_phase_to_npz()`,
  `incoherent_kbin_averaging()`,
  `incoherent_cross_power_spectrum_average()`.
* `compute_power_spectrum` extended with `units='K'|'Jy'`,
  `compute_power_spectrum_uncertainty`,
  `rescale_power_spectrum`. Old API deprecated as
  `compute_power_spectrum_old`.

### Infrastructure
* HDF5 became the default native format; FITS retained as fallback.
* `pyuvdata` integration enables UVFITS, UVH5, MIRIAD, MS export.
* Redundant baselines: `groups` / `bl_reversemap` and
  `duplicate_measurements()`.
* External beams: UVBeam and HDF5 formats; chromatic / achromatic
  switch; `select_freq` for the achromatic case.
* MPI memory chunking with `memuse` / `memavail`; per-process
  RSS-aware splitting.

### Renames / removals
| From | To | Notes |
|------|----|-------|
| `prisim/closure.py` | `prisim/bispectrum_phase.py` | Reflects expanded scope. |
| `prisim/changelog_interferometry.txt` | `prisim/changelog.txt` |  |
| `scripts/prisim_memory_monitor.py` | `prisim_resource_monitor.py` | More general. |
| `scripts/simbeam_to_healpix.py` | `scripts/FEKO_beam_to_healpix.py` |  |
| `examples/pbparms/simbeam_to_healpix.yaml` | `…/FEKO_beam_to_healpix.yaml` |  |
| `Interferometer.write_uvfits()` | `InterferometerArray.pyuvdata_write()` | Now multi-format. |
| `compute_power_spectrum()` | `compute_power_spectrum_old()` (deprecated) → newer signature | API replaced. |
| `compute_closure_phase_power_spectrum()` | `compute_individual_closure_phase_power_spectrum()` |  |
| `rect_generator()` | `rectangle_generator()` |  |
| `fgparm` (YAML key) | `skyparm` |  |
| `visible` attribute | `hemisphere` |  |
| `pyephem` | `astropy.coordinates`/`astropy.time` | Ephemeris overhaul. |
| `MWA-128T` layout | `MWA-I-128T` |  |

### Breaking changes worth flagging
* Delay transform sign convention change (FFT → IFFT).
* `GainInfo` reorganisation (flat dict → hierarchical class with
  time/frequency dimensions).
* Strict `numpy` dtypes for baseline labels (no float/int
  promotion).
* Time format unified on JD; legacy `timeformat` key removed.
* Some FITS-only output paths replaced by HDF5; not all old FITS
  paths still work.

---

## 17. Cross-References to Other Vendored Simulators

This document complements the per-simulator references colocated in
`simulators/`:

* `DP3.md`            — Default Pre-Processing Pipeline (LOFAR).
* `OSKAR.md`          — OSKAR (SKA-aperture-array simulator).
* `meqtrees-cattery.md` — MeqTrees Cattery.
* `hera_sim.md` (per `~/.claude` memory) — HERA simulator.
* `RIMEz/`, `WODEN/`, `fftvis/`, `healvis/`, `matvis/`, `PRISim/`,
  `pyradiosky-style sky` — see their respective folders.

Where multiple simulators expose comparable functionality, PRISim's
particular niche is:

* A **single-author Python package** built around a
  spectral-domain RIME with strong emphasis on
  **delay-spectrum and closure-phase / bispectrum-phase
  analyses** for HI / EoR work.
* Tight integration with the author's own `astroutils` for DSP
  (windowing, FFTs), geometry, catalogues, and gridding.
* First-class support for **redundant arrays** (HERA, PAPER,
  HIRAX, MWA), **horizon-aware delay CLEAN**, and
  **closure-phase power-spectrum uncertainty via subsample
  differencing**.
* MPI-parallel `run_prisim.py` driver fed by a single, large but
  flat YAML.

---

*End of document.*
