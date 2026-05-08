# FHD — Fast Holographic Deconvolution

> Exhaustive technical reference for the IDL pipeline located at
> `/Users/kartikmandar/RadioSim/simulators/FHD/`.
>
> All claims below are grounded in the source files inspected inside this
> submodule. Where something is not present (e.g. there is no Python build
> system) it is called out explicitly. Documentation `.md` files inside the
> repo (`README.md`, `dictionary.md`, `outputs.md`, `inputs.md`,
> `assumptions.md`, `examples.md`, `publications.md`) were used as authoritative
> companion docs and are themselves part of the package.

---

## 1. Overview

**FHD (Fast Holographic Deconvolution)** is an open-source IDL imaging,
calibration and simulation pipeline for radio interferometers. The package
description from `README.md` states:

> "FHD is an open-source imaging algorithm for radio interferometers,
> specifically tested on MWA Phase I, MWA Phase II, PAPER, and HERA. There
> are three main use-cases for FHD: efficient image deconvolution for
> general radio astronomy, fast-mode Epoch of Reionization analysis, and
> simulation. A license for IDL 8.2 or above is required."

It implements the **holographic mapping function** (HMF) deconvolution
described in Sullivan, Morales & Hazelton (2012,
[arXiv:1209.1653](https://arxiv.org/abs/1209.1653)) and the EoR/power-spectrum
pipeline detailed in Barry et al. 2019a
([arXiv:1901.02980](https://arxiv.org/abs/1901.02980)). The same code base
provides:

1. **Imaging / calibration** — gridding visibilities, sky-model based
   calibration, and fast holographic deconvolution.
2. **EoR mode** — outputs frequency-split HEALPix cubes for power-spectrum
   pipelines (notably ε-ppsilon).
3. **Simulation** — `vis_simulate` / `array_simulator` produce model
   visibilities from point-source catalogs, diffuse models, EoR cubes or
   "in-situ" model-as-data round trips.

### Scientific role

FHD is one of the canonical analysis pipelines for the **MWA Epoch of
Reionization** experiment. `publications.md` lists ≈20 papers using FHD
(916 ADS citations as of Aug 2022). Within the broader 21 cm landscape it
sits alongside RTS, CASA-style packages and HERA/PAPER tooling, but its
distinguishing technical contribution is the holographic mapping function:
the gridding kernel **and its conjugate** are tracked so that PSF
deconvolution can be performed directly in the gridded *uv* plane without
an explicit IFFT/FFT cycle.

### License

`LICENSE.txt` — 2-clause BSD-style licence: "Copyright (c) 2014, Sullivan,
I., Morales, M., Hazelton, B. All rights reserved." It permits source/binary
redistribution provided the copyright notice is retained, and disclaims all
warranties.

### Languages and version

| Aspect | Value |
|---|---|
| Primary language | **IDL** (`.pro`, 256 source files) |
| Auxiliary | Python (`data_download.py`, one `.py` in `instrument_config/obsolete/`) |
| Site config | Jekyll (`_config.yml`: `theme: jekyll-theme-cayman`) |
| `git describe` | `v3.5-1558-g83137c2e` (HEAD at writing) |
| Latest tags | `v1.0`, `v2.0`, `v2.1`, `v3.0`, `v3.5`, `diffuse_decon_Jun2021`, `diffuse_image_Jun2021` |
| Total commits | 4,746 (as reported by `git log` count) |
| Required IDL | **8.2 or above** (per README) |
| GDL fallback | Not officially supported — the code uses many IDL-only constructs (`PTR_NEW`, `STRUCTURE_TO_TEXT`, `cgHasImageMagick`, `astrolib`, `rootdir('FHD')`, IDL `journal`). README does not mention GDL. |

There is no `setup.py`, `Makefile`, `configure`, `pixi`, or `conda` recipe in
the repository — installation is purely "drop into the IDL `!PATH`".

---

## 2. Repository layout

Tree of the top-level submodule (depth 2). Significant text/dot files are
shown alongside directories.

```
simulators/FHD/
├── README.md                         project landing page
├── LICENSE.txt                       BSD-style licence
├── _config.yml                       Jekyll Cayman theme (GitHub Pages)
├── .gitignore                        Eclipse + IDL artifacts
├── assumptions.md                    inherent FHD assumptions
├── inputs.md                         catalog/uvfits input descriptions
├── outputs.md                        every output struct field
├── examples.md                       wrapper recipes (firstpass, decon, …)
├── dictionary.md                     ~750 lines of keyword definitions
├── publications.md                   chronological bibliography
├── data_download.py                  MWA GPU-box → cotter → uvfits driver
│
├── catalog_data/                     point-source / diffuse catalog files
│   ├── shapelets/                    shapelet basis function files
│   ├── simulation/                   ready-to-use simulation catalogs
│   └── *.sav, *.fits, planck_map_read.pro, … (~30 files)
│
├── fhd_core/                         CORE algorithms (grouped by stage)
│   ├── fhd_main.pro                  TOP-LEVEL driver
│   ├── beam_modeling/                tile/PSF beam construction
│   ├── calibration/                  sky-model based gain solving
│   ├── deconvolution/                FHD CLEAN-style deconvolver
│   │   ├── fast_holographic_deconvolution.pro  *the* algorithm
│   │   └── source_detection/         component → source pipeline
│   ├── gridding/                     visibility ↔ uv-grid (HMF kernel)
│   ├── HEALPix/                      uv → HEALPix cube generators
│   ├── polarization/                 Jones / Mueller / Stokes
│   ├── setup_metadata/               obs / params / layout / save_io
│   ├── source_modeling/              source → uv DFT, diffuse, galaxy
│   ├── visibility_manipulation/      uvfits I/O, flagging, averaging
│   └── obsolete/                     deprecated implementations
│
├── fhd_output/                       fits/png/HEALPix exports
│   ├── fhd_quickview.pro             post-run image/PNG export driver
│   ├── imagefast.pro                 fast PNG/contour writer
│   ├── wr_uvfits.pro                 dump residual uvfits
│   ├── calibration_plots/            cal solution plotters
│   ├── fft_filters/                  uniform/natural/robust uv filters
│   ├── HEALPix/                      HEALPix-aware FITS / view
│   └── obsolete/                     legacy fhd_output.pro
│
├── fhd_utils/                        general-purpose IDL helpers
│   ├── mpfit.pro / mpfit2dfun.pro    Markwardt's MPFIT (vendored)
│   ├── FFT/                          source DFT and shift utilities
│   ├── format_conversion/            RTS, NumPy, MRC, GSM helpers
│   ├── IDL_tools/                    pointer copy, structure/text, etc.
│   ├── modified_astro/               astrometry tweaks (refraction, …)
│   ├── horizon_mask.pro, l_m_n.pro, weight_invert.pro, …
│   └── obsolete/
│
├── instrument_config/                per-instrument beam + bandpass data
│   ├── mwa_*.pro / mwa_*.txt / mwa_*.sav / mwa_*.fits
│   ├── hera_*.pro / HERA_beam_*.sav
│   ├── paper_*.pro / paper_x_beam_nside128.fits
│   ├── lofar_*.pro / lofar_hamaker_jones_*.h5
│   └── obsolete/                     historical bandpass txt files
│
├── obs_list/                         lists of MWA obsids for runs
│   ├── Aug23.txt, barry2019b*.txt, beardsley_thesis_list*.txt, …
│
├── Observations/                     pre-computed HEALPix index restorers
│   ├── *_healpix_inds*.idlsave
│   ├── general_obs.pro               main wrapper-style entry
│   └── observation_healpix_inds_select.pro
│
└── simulation/                       all simulation-only entry points
    ├── array_simulator.pro / array_simulator_init.pro
    ├── vis_simulate.pro              build model visibilities
    ├── eor_sim.pro / eor_bubble_sim.pro
    ├── uvfits_header_simulate.pro / uvfits_params_simulate.pro
    ├── in_situ/                      "instrument-as-itself" simulations
    │   ├── in_situ_sim_setup.pro
    │   └── vis_noise_simulation.pro
    ├── Instrument_configuration/     {hera,mwa,paper}_simulation_instr_config.pro
    └── obsolete/                     fhd_sim.pro
```

### Per-folder commentary

| Folder | Role | Notable contents |
|---|---|---|
| `fhd_core/` | All physics; called by wrappers | `fhd_main.pro` (319 LOC) is the canonical pipeline |
| `fhd_core/gridding/` | Gridding/degridding using the holographic kernel | `visibility_grid.pro` (431 LOC), `holo_mapfn_apply.pro`, `holo_mapfn_convert.pro`, `baseline_grid_locations.pro`, `interpolate_kernel.pro` |
| `fhd_core/deconvolution/` | The FHD algorithm | `fast_holographic_deconvolution.pro` (453 LOC), `fhd_init.pro`, `fhd_wrap.pro`, `fhd_multi.pro` (joint multi-snapshot), source-detection sub-pipeline |
| `fhd_core/calibration/` | Per-tile complex gain solver | `vis_calibrate.pro`, `vis_calibrate_subroutine.pro` (linear-LSQ inner loop), bandpass/polyfit/auto-fit modules |
| `fhd_core/beam_modeling/` | Per-tile UV beam (PSF) construction | `beam_setup.pro`, `fhd_struct_init_antenna.pro`, `fhd_struct_init_psf.pro`, Gaussian decomposition path (`beam_gaussian_decomp.pro`) |
| `fhd_core/HEALPix/` | snapshot ↔ HEALPix | `healpix_snapshot_cube_generate.pro`, `healpix_cnv_generate.pro`, `integrate_healpix_cubes.pro` |
| `fhd_core/polarization/` | Jones/Mueller/Stokes round-trips | `stokes_cnv.pro`, `fhd_struct_init_jones.pro`, `parallactic_angle_memo.pdf` (math memo) |
| `fhd_core/setup_metadata/` | Bookkeeping structures + save I/O | `fhd_struct_init_obs.pro`, `fhd_struct_init_layout.pro`, `fhd_save_io.pro`, `fhd_setup.pro`, `fhd_path_setup.pro` |
| `fhd_core/source_modeling/` | Source → UV DFT, galaxy & diffuse models | `source_dft_multi.pro`, `vis_source_model.pro`, `fhd_diffuse_model.pro`, `fhd_galaxy_model.pro`, `shapelet_model_uv.pro` |
| `fhd_core/visibility_manipulation/` | uvfits read/write + flagging + filtering | `uvfits_read.pro`, `vis_flag.pro`, `vis_flag_basic.pro`, `vis_average.pro`, `vis_delay_filter.pro` |
| `fhd_output/` | Image/FITS/PNG writers | `fhd_quickview.pro` (38 KB), `imagefast.pro`, `wr_uvfits.pro`, `vis_export.pro` |
| `fhd_utils/` | Generic IDL helpers, pointer ops, MPFIT | `mpfit.pro` (144 KB — vendored), `weight_invert.pro`, `meshgrid.pro`, FFT helpers |
| `instrument_config/` | Reference beam / bandpass / cable models | MWA tile beam (with delay + dipole coupling), HERA gain `.sav`s, PAPER nside-128 FITS, LOFAR Hamaker Jones HDF5 |
| `obs_list/`, `Observations/` | Recipe inputs (no code) | obsid lists; `general_obs.pro` is the canonical wrapper; `observation_healpix_inds_select.pro` |
| `simulation/` | Simulation-only drivers | `array_simulator.pro` (no real visibilities required), `vis_simulate.pro` (model-uvf cube), `eor_sim.pro`, `eor_bubble_sim.pro`, `in_situ/` |
| `catalog_data/` | Sky catalogs | `GLEAM_*.sav` (multiple variants), `MRC_*`, `mwa_calibration_source_list*.sav`, `gsm_150MHz.sav`, `lambda_haslam408_dsds.fits`, `master_sgal_cat.sav`, simulation/ subfolder with EoR power and synthetic source files |

---

## 3. Installation & dependencies

Per `README.md` §Installation. There is no automated installer.

| Dependency | Mandatory | Purpose | Source |
|---|---|---|---|
| **IDL 8.2+** | yes | runtime / language | proprietary licence required |
| FHD itself | yes | this repo | github.com/EoRImaging/FHD |
| **fhdps_utils** | yes | `spectral_window.pro` and other helpers | github.com/EoRImaging/fhdps_utils |
| **astro IDL Library** (NASA) | yes | astrometry, FITS I/O (`mrdfits`, `fxposit`, `sxpar`) | idlastro.gsfc.nasa.gov |
| **coyote library** | yes | Imagemagick interface, PNG writing | github.com/idl-coyote/coyote |
| **HEALPix IDL** | yes | HEALPix grid / FITS support (`init_healpix`, `query_disc`, `npix2nside`) | healpix.sourceforge.io |
| **eppsilon** | optional | downstream PS pipeline | github.com/EoRImaging/eppsilon |
| **Imagemagick** | recommended | PNG conversion (`cgHasImageMagick`) | imagemagick.org |
| **GDL** | not supported | – | – |

`README.md` provides four installation smoke tests:

```text
print, cgHasImageMagick()       ; → 1 if coyote+IM ok
astrolib                         ; → "ASTROLIB: ... added"
init_healpix                     ; → no error if HEALPix is on path
imagefast, randomN(5,256,256), file_path='.../testimage.png'
```

The IDL path must be set with the `+` recursive prefix on Unix, e.g.:

```idl
!PATH = Expand_Path('+/path/to/FHD/') + ':' + !PATH
```

There is no MPI, no GPU. `fhd_setup.pro` carries a commented-out `GPU_enable`
block that has been disabled. Visibility gridding is single-threaded IDL
`MATRIX_MULTIPLY` (with `TPOOL_MIN_ELTS=20000` to enable IDL's internal
parallelism) but there is no explicit cluster code. Higher-level
parallelism is "embarrassingly parallel over obsids" — typically dispatched
via `qsub`/SLURM scripts living in the separate
`pipeline_scripts` repository, not here.

`data_download.py` (1100+ LOC) is an MWA-specific GPU-box pre-processor that
shells out to `cotter`; not part of the IDL path.

---

## 4. Build & runtime architecture

FHD has no compile step. Each `.pro` file is a single IDL routine
(occasionally with helper sub-functions, e.g. `set_pol_cal_params` inside
`fhd_struct_init_layout.pro`). At runtime, IDL compiles routines
lazily as they are referenced; a wrapper script calls a top-level driver
(`general_obs`, `fhd_main`, or `array_simulator`) which in turn pulls in
the rest of the tree.

### The pipeline at a glance (imaging mode)

```
                                           ┌─────────────────┐
  uvfits files  ──►  uvfits_read           │ instrument_config/
                          │                │   beam, bandpass │
                          ▼                │   cable, dipole  │
                   fhd_struct_init_obs ◄───┴──────────────────┘
                          │
                          ▼
                    beam_setup ────────────► psf  (UV beam kernel pyramid)
                          │
                          ▼
                  fhd_struct_init_jones ──► jones (Mueller mapping)
                          │
                          ▼
                  vis_flag_basic
                          │
        ┌───────  vis_calibrate (optional, sky-model based)
        │                 │
        │                 ▼
        │         vis_source_model ──── catalogs / diffuse model
        │                 │
        │                 ▼
        ▼          visibility_grid_wrap ───────► image_uv_arr
                          │     ▲                weights_arr
                          │     └─ holographic mapping fn (map_fn_arr)
                          ▼
   ┌──────── fast_holographic_deconvolution (optional)
   │              │
   │              ▼
   │        component_array, source_array, model_uv_holo
   │              │
   ▼              ▼
fhd_quickview ────────────────────► fits + png
   │
   ▼
healpix_snapshot_cube_generate ──► HEALPix cubes  → eppsilon
```

### Layered view

| Layer | Routines |
|---|---|
| Top-level wrappers | `general_obs.pro`, `array_simulator.pro` |
| Driver | `fhd_main.pro` |
| Setup / metadata | `fhd_setup`, `fhd_path_setup`, `fhd_save_io`, `fhd_struct_init_*` |
| I/O | `uvfits_read`, `vis_export`, `wr_uvfits`, `imagefast`, `fits_write_healpix`, `calfits_read` |
| Calibration | `vis_calibrate*`, `vis_cal_bandpass`, `vis_cal_polyfit`, `vis_cal_auto_fit` |
| Beams (PSF kernels) | `beam_setup`, `fhd_struct_init_antenna`, `fhd_struct_init_psf`, instrument-specific `*_beam_setup_init/_gain.pro` |
| Source / sky model | `source_dft_*`, `vis_source_model`, `fhd_diffuse_model`, `fhd_galaxy_model` |
| Gridding | `visibility_grid`, `visibility_degrid`, `baseline_grid_locations`, `holo_mapfn_apply`, `holo_mapfn_convert` |
| Deconvolution | `fast_holographic_deconvolution`, `fhd_init`, `fhd_wrap`, `fhd_multi[_wrap]`, `fhd_source_detect`, `process_deconvolution_components` |
| HEALPix | `healpix_snapshot_cube_generate`, `healpix_cnv_generate`, `healpix_cnv_apply`, `integrate_healpix_cubes` |
| Output | `fhd_quickview`, `source_array_export`, `vis_export`, plotting helpers |
| Utility | `fhd_utils/*` (MPFIT, FFT helpers, refraction, astrometry tweaks) |

### IDL idioms used throughout

* **Pointer arrays** — visibilities, beam kernels, cubes, gain arrays, etc.
  are all kept as `Ptrarr(...)`. The pattern
  `Ptrarr(n_pol, /allocate)` followed by `*vis_arr[pol_i] = ...` is
  pervasive (see `vis_simulate.pro`, `array_simulator.pro`).
* **`compile_opt idl2, strictarrsubs`** — turns on integer constants and
  strict array subscripts at the top of most routines.
* **`_Extra=extra`** — keyword pass-through chain reaches all the way down
  from a user wrapper to nested routines (e.g. `obs_status` → `vis_flag`).
* **Lazy save/restore via `fhd_save_io`** — see §6.
* **`heap_gc`** is called liberally to free pointer heap.
* `git, 'describe', repo_path=rootdir('fhd')` is used inside
  `general_obs.pro` and `fhd_struct_init_obs.pro` to embed the `git
  describe` string in the `obs.code_version` field.

---

## 5. Public API / CLI

FHD has **no command-line entry**: there is no `fhd` shell binary; users
write IDL wrappers. Entry points actually intended to be called by users
are listed below; everything else is internal. Argument lists below show
the prominent keywords; the full list is `dictionary.md`'s job.

### 5.1 `general_obs` (`Observations/general_obs.pro`)

The recommended top-level wrapper for snapshot imaging / EoR runs over a
list of uvfits files.

```idl
PRO general_obs, cleanup=, recalculate_all=, export_images=, version=, $
                 mapfn_recalculate=, grid_recalculate=, snapshot_recalculate=, $
                 deconvolve=, image_filter_fn=, data_directory=, $
                 output_directory=, n_pol=, precess=, vis_file_list=, $
                 fhd_file_list=, healpix_path=, catalog_file_path=, $
                 complex_beam=, pad_uv_image=, update_file_list=, $
                 combine_healpix=, start_fi=, end_fi=, skip_fi=, $
                 flag_visibilities=, transfer_mapfn=, transfer_weights=, $
                 simultaneous=, flag_calibration=, $
                 calibration_catalog_file_path=, transfer_calibration=, $
                 snapshot_healpix_export=, save_visibilities=, $
                 firstpass=, return_cal_visibilities=, cmd_args=, silent=, $
                 _Extra=extra
```

Behaviour highlights observed in the source:

- Detects nested calls and bails (`scope_traceback`) — common pitfall after
  a previous run crashed.
- Embeds the FHD git hash via
  `git, 'describe', result=code_version, repo_path=rootdir('fhd'), args='--long --dirty'`.
- Defaults: `instrument='mwa'`, `n_pol=2`, `pad_uv_image=1`,
  `image_filter_fn='filter_uv_uniform'`,
  `catalog_file_path = catalog_data/MRC_full_radio_catalog.fits`,
  `calibration_catalog_file_path = catalog_data/<instrument>_calibration_source_list.sav`.
- If `firstpass` is set: forces `return_cal_visibilities=1`,
  `mapfn_recalculate=0`, `deconvolve=0`, `export_images=1`.

### 5.2 `fhd_main` (`fhd_core/fhd_main.pro`)

Per-observation driver. Keywords are documented exhaustively in
`dictionary.md`. The canonical sequence executed by `fhd_main` (line numbers
refer to `fhd_main.pro`):

| Step | Lines | Routine |
|---|---|---|
| Setup state structure | 31 | `fhd_setup` |
| Read uvfits | 44 | `uvfits_read` |
| Optional in-situ sim hook | 53 | `in_situ_sim_setup` |
| Build `obs` struct | 66 | `fhd_struct_init_obs` |
| Optional w-deprojection | 71 | `simple_deproject_w_term` |
| Beam (`psf`) | 75 | `beam_setup` |
| Jones / Mueller (`jones`) | 77 | `fhd_struct_init_jones` |
| Optional weight transfer | 81 | `transfer_weights_data` |
| Basic flagging | 93 | `vis_flag_basic` |
| Calibration source list | 101 | `generate_source_cal_list` |
| Calibration | 131 | `vis_calibrate` |
| `cal_stop` early exit branch | 146 | save & return |
| Anomalous-data flagging | 153 | `vis_flag` |
| Model visibility build | 189 | `vis_source_model` |
| Save skymodel | 206 | `fhd_save_io` |
| Auto-correlations | 242 | `vis_extract_autocorr` |
| Grid visibilities | 262 | `visibility_grid_wrap` |
| Deconvolve | 279 | `fhd_wrap` |
| Quickview / export | 290 | `fhd_quickview` |
| HEALPix cube export | 296 | `healpix_snapshot_cube_generate` |
| Optional MWA-QC DB filler | 308 | `SPAWN, fhd_database_filler.py …` |

### 5.3 `array_simulator` (`simulation/array_simulator.pro`)

Standalone simulation driver — does not require real uvfits input.

```idl
PRO array_simulator, vis_arr, vis_weights, obs, status_str, psf, params, jones, $
                     instrument=, n_pol=, $
                     eor_sim=, include_noise=, include_catalog_sources=, $
                     catalog_file_path=, source_array=, $
                     hpx_select_radius=, select_radius_multiplier=, $
                     snapshot_healpix_export=, export_images=, $
                     snapshot_recalculate=, grid_recalculate=, $
                     eor_uvf_cube_file=, _Extra=extra
```

Calls `array_simulator_init` → `fhd_struct_init_jones` →
`beam_setup` → `vis_simulate` → `visibility_grid` →
`fhd_quickview` / `healpix_snapshot_cube_generate`.

### 5.4 Other invokable routines

| Routine | Purpose |
|---|---|
| `vis_simulate` | Build a `Ptrarr(n_pol)` of model visibilities from a sky model. |
| `eor_sim` | Generate EoR uvf cube from `eor_power_1d.idlsave` (line-of-sight comoving conversions). |
| `eor_bubble_sim` | Add reionisation bubble HDF5 cubes (HEALPix orthographic projected) into `model_uvf_arr`. |
| `integrate_healpix_cubes` | Combine multi-obs HEALPix cubes for ε-ppsilon. |
| `fhd_quickview` | Standalone post-run visualisation/export (called by `fhd_main`). |
| `imagefast` | "Fast" PNG writer with annotations/grids. |

There is no `validate`/`init` subcommand; a wrapper is the canonical
configuration. The closest thing to a config schema is the keyword list in
`dictionary.md`.

---

## 6. State management and save/restore

`fhd_save_io.pro` (`fhd_core/setup_metadata/`) is FHD's persistence layer.
A `status_str` structure records, per output type, whether it has been
written. The default reset (line 41) initialises:

```idl
status_str = { hdr:0, params:0, obs:0, layout:0, psf:0, antenna:0, $
               jones:0, cal:0, skymodel:0, source_array:0, $
               vis_weights:0, auto_corr:0, $
               vis_ptr:intarr(4), vis_model_ptr:intarr(4), $
               grid_uv:intarr(4), weights_uv:intarr(4), $
               grid_uv_model:intarr(4), vis_count:0, $
               map_fn:intarr(4), fhd:0, fhd_params:0, $
               hpx_cnv:0, healpix_cube:intarr(4), $
               hpx_even:intarr(4), hpx_odd:intarr(4), complete:0 }
```

`fhd_save_io` dispatches on `var_name` (line 55+) to choose subdirectory,
filename suffix and pol-indexing:

| Variable | Suffix | Subdir |
|---|---|---|
| `status_str` | `_status` | `metadata/` |
| `hdr` | `_hdr` | `metadata/` |
| `obs` | `_obs` | `metadata/` |
| `params` | `_params` | `metadata/` |
| `layout` | `_layout` | `metadata/` |
| `psf` / `antenna` / `jones` / `cal` / `skymodel` | per-name | `beams/`, `metadata/`, `calibration/` |
| `vis_ptr` (per pol) | `_vis_<XX|YY|XY|YX>` | `vis_data/` |
| `grid_uv`, `weights_uv`, `grid_uv_model`, `map_fn` | per-pol | `grid_data/`, `mapfn/` |
| `fhd` / `fhd_params` | `_fhd` | top-level |
| HEALPix cubes (`healpix_cube`, `hpx_even`, `hpx_odd`) | per-pol | `Healpix/` |

`fhd_path_setup.pro` resolves the output directory tree —
`output_directory/fhd_<version>/<filename>` is the canonical layout, with
optional subdirectories (`Healpix`, `metadata`, etc.). When
`output_filename` is given it overrides the auto-derived basename.

`fhd_setup.pro` reads the `status_str` and decides whether earlier outputs
must be regenerated (handling `transfer_mapfn`, `transfer_weights`,
`transfer_calibration`, `recalculate_all`, etc.).

---

## 7. Data structures

The core in-memory objects are anonymous IDL `STRUCTURE`s built by the
`fhd_struct_init_*` routines and pickled to `.sav` files by
`fhd_save_io`. `outputs.md` is the field-by-field dictionary; the table
below summarises the most important entries.

### 7.1 `obs` (`fhd_struct_init_obs.pro`)

| Field | Meaning |
|---|---|
| `obsname` | basename of the input uvfits |
| `code_version` | `git describe` of FHD repo |
| `instrument` | lower-case instrument name (`mwa`, `mwa2`, `hera`, `paper`, `lofar`, …) |
| `n_pol`, `n_freq`, `n_tile`, `n_time`, `nbaselines` | counts |
| `pol_names` | `['XX','YY','XY','YX','I','Q','U','V']` |
| `dimension`, `elements` | image-plane size in pixels |
| `kpix`, `degpix` | uv-pixel size (wavelengths) and image-pixel size (degrees) |
| `freq_center`, `freq_res` | central / resolution in Hz |
| `min_baseline`, `max_baseline` | wavelengths |
| `baseline_info` | pointer to struct of per-channel/tile use flags (`fbin_i`, `tile_use`, `freq_use`, `tile_A`, `tile_B`, `bin_offset`, `freq`, `tile_names`) |
| `astr` | IDL FITS astrometry struct (used by `apply_astrometry`) |
| `obsra/obsdec`, `zenra/zendec`, `obsx/obsy` | pointing, zenith and pixel-zenith |
| `JD0`, `lat`, `lon`, `alt` | time/site |
| `alpha` | global spectral index used by gridding/deconvolution |
| `degrid_spectral_terms` / `grid_spectral_terms` | spectral-gradient coefficients |
| `n_vis`, `nf_vis` | visibility counts |
| `n_tile_flag`, `n_freq_flag` | flagging tallies |
| `dft_threshold` | DFT-vs-analytic threshold |
| `double_precision` | numerical precision flag |
| `primary_beam_area`, `primary_beam_sq_area` | pointer to per-freq beam integrals |
| `delays`, `base_gain` | tile-specific metadata when present |

### 7.2 `psf` (`fhd_struct_init_psf.pro`)

`psf` carries the per-baseline UV beam kernel (the "holographic" gridding
kernel). Key tags:

| Tag | Meaning |
|---|---|
| `dim` | per-axis kernel size in `uv` pixels |
| `resolution` | super-resolution factor (default 16; eor_wrapper_defaults uses 100) |
| `n_freq` | number of frequency bins for the beam |
| `fbin_i` | mapping channel → freq-bin index |
| `id` | per-`(pol, freq, baseline)` group ID enabling kernel reuse |
| `complex_flag` | whether the beam is complex |
| `interpolate_kernel` | whether bilinear kernel interpolation is used |
| `image_info` | pointer with `psf_image_dim`, etc. |
| `beam_ptr` | nested `Ptrarr(n_pol, n_freq_bin, n_baselines)` of `Ptrarr(2*dim, 2*dim)` slots — the actual kernel hierarchy |
| `pix_horizon`, `intermediate_res` | image-space sizing |

### 7.3 `cal` (`fhd_struct_init_cal.pro`)

Captures the gain-solving state.

| Tag | Meaning |
|---|---|
| `n_pol`, `n_freq`, `n_tile`, `n_time` | counts |
| `freq` | Hz array |
| `tile_A`, `tile_B`, `bin_offset` | baseline ↔ tile bookkeeping |
| `uu`, `vv` | seconds (light-travel) per visibility |
| `min_cal_baseline`, `max_cal_baseline` | wavelength cuts |
| `min_solns`, `max_iter`, `phase_iter`, `conv_thresh` | LSQ parameters |
| `polyfit`, `bandpass`, `mode_fit` | post-fit flags |
| `amp_degree`, `phase_degree` | fit orders |
| `gain[pol_i]` | `Ptrarr(n_pol)` → `Complexarr(n_freq, n_tile)` — the solutions |
| `gain_residual[pol_i]` | residuals after polynomial fits |
| `convergence[pol_i]` | per-`(freq,tile)` convergence metric |
| `amp_params/phase_params/mode_params[pol,tile]` | polynomial coefficients |
| `auto_params[pol]` | linear-fit between auto- and cross-cor amplitudes |
| `cal_origin` | git hash / file path |
| `skymodel` | embedded `fhd_struct_init_skymodel` result |

### 7.4 `skymodel` (`fhd_struct_init_skymodel.pro`)

```idl
{ include_calibration:0/1, n_sources:n_src, source_list:source_array, $
  catalog_name:'…', galaxy_model:0/1, galaxy_spectral_index:NaN/value, $
  diffuse_model:'…', diffuse_spectral_index:NaN/value }
```

`source_list` itself is an array of `source_comp_init` structs containing
`{ x, y, ra, dec, ston, freq, alpha, gain, flag, extend, flux, … }` —
defined in `fhd_core/deconvolution/source_detection/source_comp_init.pro`.

### 7.5 `jones` (`fhd_struct_init_jones.pro`)

Mueller-matrix mapping between instrumental polarizations and
RR/DD/RD/DR. Per `outputs.md`:

```
jones = { inds, dimension, elements, Jmat[4,4], Jinv[4,4] }
```

`*Jmat[i,j]` is a `dcomplex` array indexed by `inds`, giving the (i,j)
element of the per-pixel Mueller matrix that maps `[RR*, DD*, RD*, DR*]` to
`[xx*, yy*, xy*, yx*]`. `Jinv` is its inverse. `parallactic_angle_memo.pdf`
in the same folder is the algebraic derivation.

### 7.6 `layout` (`fhd_struct_init_layout.pro`)

The AIPS `AN` table reified into:

```
{ array_center, coordinate_frame, gst0, earth_degpd, ref_date, time_system, $
  dut1, diff_utc, nleap_sec, pol_type, n_pol_cal_params, n_antenna, $
  antenna_names, antenna_numbers, antenna_coords, antenna_diameters, $
  mount_type, polA, polB, polA_orientation, polB_orientation, $
  polA_cal_params, polB_cal_params, … }
```

### 7.7 `fhd_params` (`fhd_init.pro`)

Returned dict captures all deconvolution dials:

```
{ npol, beam_threshold, max_iter, max_deconvolution_components, check_iter, $
  gain_factor, add_threshold, over_resolution, dft_threshold, $
  independent_fit, reject_pol_sources, beam_max_threshold, $
  horizon_threshold, smooth_width, sigma_cut, local_max_radius, $
  transfer_mapfn, galaxy_subtract, sidelobe_subtract, sidelobe_return, $
  filter_background, decon_filter, decon_mode, joint_obs, $
  end_condition, n_iter, n_components, n_sources, $
  detection_threshold, convergence, info }
```

Notable defaults from `fhd_init.pro`:

| Key | Default |
|---|---|
| `gain_factor` | 0.15 |
| `beam_threshold` | 0.05 |
| `beam_max_threshold` | 1e-4 |
| `max_deconvolution_components` | 100 000 |
| `add_threshold` | 0.8 |
| `local_max_radius` | 3 px |
| `max_iter` | `Ceil(Sqrt(max_deconvolution_components)) > 10` |
| `check_iter` | `Round(1./gain_factor) < 5` |
| `smooth_width` | 32 px |
| `over_resolution` | 2 |
| `horizon_threshold` | 10° |
| `decon_filter` | `'filter_uv_uniform'` |
| `dft_deconvolution_threshold` | `1./((2π)^2 * dimension)` if input ≥ 1 |

---

## 8. Core algorithms

### 8.1 Holographic gridding (`visibility_grid.pro`)

The core "holographic" insight in FHD is that for *each* baseline the
gridding kernel is the **conjugate of the antenna voltage pattern** (the
hyperresolved UV beam), and that the *outer product* of those kernels
across all baselines defines the **mapping function** `M` that connects
the gridded data plane to itself under PSF convolution. Storing `M` (the
"holo map function") allows `fast_holographic_deconvolution` to apply the
PSF directly in the *uv* plane.

Stage by stage in `visibility_grid.pro` (≈430 LOC):

1. **Baseline indexing** —
   `baseline_grid_locations(obs, psf, params, …)` computes, for each
   baseline-frequency pair, the minimum integer `(xmin, ymin)` UV pixel,
   the hyper-resolution offset within the kernel
   (`x_offset, y_offset`), and (optionally) the four bilinear interpolation
   derivatives. Returns a histogram (`bin_n`), reverse-indices (`ri`),
   used-bins (`bin_i`) — i.e. the standard IDL "histogram with
   `reverse_indices`" pattern.
2. **Conjugation symmetry** — visibilities with `vv > 0` are conjugated
   so that gridding only happens in the lower-half UV plane. The full
   plane is reconstructed at the end via
   `image_uv = (image_uv + conjugate_mirror(image_uv))/2`.
3. **Group reuse** — `xyf_i = (x_off + y_off*psf_resolution +
   fbin*psf_resolution^2)*group_max + group_id` defines a unique key;
   identical visibilities are summed *before* gridding, reducing per-bin
   work to one matrix multiply.
4. **Box matrix construction** — for each unique (kernel, group), the
   kernel slice `*psf.beam_ptr[pol, fbin, baseline_id]*[x_off, y_off]` is
   inserted as a column of a `(psf_dim^2, vis_n)` matrix `box_matrix`.
   When `interpolate_kernel` is true,
   `interpolate_kernel(*beam_ptr, x_offset, y_offset, dx0dy0, …)` performs
   the 2-D bilinear interpolation in-place.
5. **Per-baseline beam path** — when `beam_per_baseline` is set,
   `grid_beam_per_baseline` builds the kernel on the fly using the actual
   `(uu, vv, ww)` and the `(l_mode, m_mode, n_tracked)` direction cosines
   from `l_m_n.pro`. (Interpolation and per-baseline cannot be combined —
   see WARNING in lines 28–31.)
6. **Gridding step** — `box_arr = matrix_multiply(vis_box/n_vis,
   box_matrix_dag, /atranspose, /btranspose)` applies the conjugate
   kernel to the visibility values; the resulting `psf_dim × psf_dim`
   block is added at `(xmin, ymin)`. Optional outputs include `weights`
   (kernel only), `variance` (`|kernel|²`), `uniform_filter`, the **mapping
   function** (full kernel × kernel†), and spectral-gradient `A`/`B`/`D`
   moments for `grid_spectral`.
7. **Map-function sparsity** — `map_fn[i,j]` is a *sparse* IDL pointer
   array: only `(xmin..xmin+psf_dim-1, ymin..ymin+psf_dim-1)` cells are
   allocated as needed (lines 169–179). After gridding,
   `holo_mapfn_convert` packs it into a sparse format suitable for
   `holo_mapfn_apply`.

### 8.2 Holographic mapping function

`holo_mapfn_apply(uv_plane, map_fn, /indexed)` applies the stored sparse
mapping function: for any UV-plane source model
`*model_uv_full[pol_i]`, `*model_uv_holo[pol_i] = M·model_uv_full`
gives the visibility-weighted, beam-modulated *uv* image that should be
compared with the gridded data. Crucially, `model_uv_holo` is in the
same units / kernel basis as the dirty image, so subtraction is exact in
the gridded domain.

`holo_mapfn_convert.pro` and `sprsax2.pro` (a vendored sparse-array
multiply) implement the storage/apply.

### 8.3 Fast Holographic Deconvolution
(`fhd_core/deconvolution/fast_holographic_deconvolution.pro`)

The deconvolver replaces the standard CLEAN inner loop with HMF-based
component subtraction. From the file's own header comment:

> "Deconvolution algorithm to fit multiple polarizations simultaneously
> using the Holographic Mapping function. Fits multiple components
> simultaneously."

Sections inside the routine (annotated by the author):

1. **§0 Setup** (lines 49–192):
   * Build `beam_base[pol]`, `beam_correction[pol]`, `beam_avg`,
     `beam_mask`.
   * Compute `gain_normalization` via `get_image_renormalization` so that
     residual Jy/sr → Jy/pixel.
   * FFT each `image_uv_arr[pol]` to `dirty_array[pol]` via
     `dirty_image_generate` (with optional `over_resolution` zero
     padding).
   * Stokes-rotate to `dirty_image_composite_{,Q,U,V}` via `stokes_cnv`
     and the per-pixel Mueller `jones_fit`.
2. **§1 Pre-deconvolution** (lines 195–262):
   * Optional `galaxy_model_fit` subtracts a Galactic-emission UV model
     (`fhd_galaxy_model`).
   * Optional `subtract_sidelobe_catalog` removes bright known sources
     in sidelobes (using the same DFT path).
3. **§2 Iteration loop** (lines 274–390):
   1. `model_holo_arr[pol] = dirty_image_generate(model_uv_holo[pol], …)`
      — the holographic model image.
   2. Stokes-rotate to `model_image_composite{,Q,U,V}`.
   3. `image_unfiltered = dirty_image_composite − model_image_composite`.
   4. Optional `background_subtraction` (smooth-width filter) gives
      `image_filtered`.
   5. `source_find_image = image_filtered * beam_avg * beam_mask *
      source_taper`.
   6. `fhd_source_detect(obs, fhd_params, jones, source_find_image, …)`
      finds local maxima above `add_threshold` of the brightest pixel,
      isolated by `local_max_radius`, and bright above
      `detection_threshold` — and packages them as a list of components.
   7. Components are accumulated into `component_array`.
   8. The new components' UV signature is added to `model_uv_full` via
      `source_dft_multi`, then `model_uv_holo = M·model_uv_full`.
   9. Convergence: every `check_iter` iterations,
      stop if `Stddev(image_use)` rises (lack of progress), or if
      `sigma_threshold * Stddev > Max(source_find_image)` (low SNR), or
      if `max_deconvolution_components` is reached.
4. **§3 Post-processing** (lines 401–432):
   * `process_deconvolution_components` culls false positives, condenses
     valid components into sources (Gaussian fitting where applicable),
     populating `source_array`.
   * Final `model_uv_full` is rebuilt from condensed `source_array` (or
     `component_array` if `no_condense_sources`).
   * `model_uv_holo` is recomputed.
   * Per-pol residuals: `residual_array[pol] = (image_uv − model_uv_holo)
     / beam_correction`.

End-conditions stored in `fhd_params.end_condition`: `'Source fit failure'`,
`'Max components'`, `'Convergence'`, `'Low SNR'`, `'Max iterations'`.

### 8.4 Source DFT (`source_dft_multi.pro`, `source_dft.pro`)

For each accepted component the routine performs a **direct discrete
Fourier transform** from sky to *uv* — *not* an FFT — to avoid pixelisation
artefacts at sub-pixel positions. Operations performed:

* Apply per-component spectral index: `flux *= (freq_use/freq_ref)^alpha`.
* Switch through `n_pol` to map Stokes I (and optionally polarized terms)
  into instrumental coherency via `Stokes_cnv(/inverse)`.
* Optional Gaussian source models (`gaussian_source_models`, see lines
  54–80): convert FWHM in arcsec → σ in radians, project onto
  `(gaussian_x, gaussian_y)` pixel coordinates with a flat-sky compression
  correction (`gaussian_ra_corr`, `gaussian_dec_corr` — see Cook et al.
  2022), and propagate the Gaussian amplitude correction into the flux.
* Compute `e^{2πi (xvals*x + yvals*y) / dimension}` over `uv_i_use` indices
  only (sub-pixel `x_vec`, `y_vec`). When `dft_threshold > 0`, sources are
  truncated by an analytic kernel approximation (controlled by `obs.dft_threshold`).

### 8.5 Visibility degridding (`visibility_degrid.pro`)

The reverse path. Given a model UV plane, compute model visibilities at
the actual baseline locations:

* The same `baseline_grid_locations` routine is used.
* `n_spectral` controls how many spectral-gradient terms are degridded
  (`obs.degrid_spectral_terms`), permitting frequency-dependent visibility
  models from a single UV plane plus its `α` derivative.
* When `beam_per_baseline` is set, kernels are made on the fly via
  `grid_beam_per_baseline` (no interpolation).
* `psf.complex_flag` toggles complex vs real kernels.
* `conserve_memory` switches the kernel matrix multiply to a per-bin loop
  when the working set exceeds `mem_thresh` bytes (default 1e8).

### 8.6 Calibration (`vis_calibrate.pro`, `vis_calibrate_subroutine.pro`)

Sky-model based per-tile complex gain solver. Outline:

1. **Transfer / load** path: if `transfer_calibration` is set,
   `vis_calibrate` short-circuits to `vis_calibration_apply` after
   reading from a `.sav`, `.txt`, `.npz`/`.npy`, or calfits FITS file
   (lines 19–69).
2. **Iteration** (in `vis_calibrate_subroutine.pro`): linear-LSQ solver
   per `(freq, tile, pol)` against the model visibilities. Notable
   knobs:
   * `cal.max_iter` (default 100), `cal.conv_thresh` (1e-7),
     `cal.phase_iter` (early phase-only iterations),
     `cal.adaptive_gain` (use `calculate_adaptive_gain` Kalman update),
     `cal.base_gain`, `divergence_history=3`, `divergence_factor=1.5`.
   * Per-pol, per-freq baseline cuts:
     `min_cal_baseline / max_cal_baseline` plus optional
     `calibration_weights` taper.
   * Visibilities can be time-averaged before solving (`cal.time_avg`).
3. **Post-fit** in `vis_calibrate.pro`:
   * `vis_cal_polyfit` (amp, phase polynomial degrees), `vis_cal_bandpass`
     (cable-grouped or whole-band bandpass with optional save-file
     transfer), `vis_cal_auto_fit` (auto-correlation amplitude tying).
   * `vis_calibration_flag` removes outliers; `cal.tile_use` propagates
     into `obs`.
4. Output: complex `gain[pol]` arrays plus residuals, polynomial
   parameters, mode-fit parameters, and bookkeeping metadata.

### 8.7 Beam construction (`beam_setup.pro`,
`fhd_struct_init_antenna.pro`, `fhd_struct_init_psf.pro`)

`beam_setup` is the one-stop builder for the `psf` structure.

* If `transfer_psf` is given (string or directory), a previous
  `*_beams.sav` / `*_obs.sav` pair is reused — with extensive baseline-set
  matching (`tile_a*ULONG(2*n_tile) + tile_b` bag) so a different
  observation's PSF can be remapped.
* Otherwise `fhd_struct_init_antenna` instantiates a per-instrument
  `antenna` struct via the dispatch:

  ```idl
  tile_init_fn = instrument + '_beam_setup_init'   ; e.g. mwa_beam_setup_init
  tile_gain_fn = instrument + '_beam_setup_gain'   ; e.g. mwa_beam_setup_gain
  ```

  All "mwa*" instrument strings re-route to `mwa_beam_setup_*`. For MWA
  (`mwa_beam_setup_init.pro` lines 30+) the 16-dipole layout is laid out
  in a 4×4 grid with measured spacing (1.1 m), antenna height 0.29 m,
  velocity factor 0.673, base delay unit 435 ps. Delays from `obs.delays`
  produce the per-tile pointing.
  `mwa_dipole_mutual_coupling.pro` adds Sutinjo 2015 mutual coupling.
* Frequencies are bin-averaged by `beam_nfreq_avg` (default 1; EoR
  wrappers use 16). For each `(pol, freq_bin, baseline)` the routine
  computes the antenna voltage pattern in *image* space at
  hyperresolution `psf_resolution × psf_intermediate_res`, FFTs to *uv*,
  clips by `beam_mask_threshold` (default 100×) and stores the kernel
  fragment in `*beam_ptr[pol, freq_bin, baseline]*[x, y]`.
* When `beam_gaussian_decomp` is set,
  `beam_gaussian_decomp.pro` fits a small set of 2-D Gaussians to the
  hyper-resolution image-space beam (see Barry & Chokshi 2022). Cached
  parameter files live in
  `instrument_config/mwa_decomp_params_pointing{-2,-1,0,1,2}.sav` and the
  Gaussian-FWHM-matched approximation is in `mwa_gauss_params_pointing*.sav`.
* `kernel_window` modifies the gridding kernel by applying a window
  (Hann/Hamming/Blackman/Nuttall/Tukey/Blackman-Harris/^2) to the
  primary beam — required for EoR power spectrum quality (per
  `dictionary.md` and `examples.md` §"Modified Gridding Kernel").

### 8.8 HEALPix snapshot cubes
(`healpix_snapshot_cube_generate.pro`, `healpix_cnv_generate.pro`,
`healpix_cnv_apply.pro`)

After `visibility_grid` has produced the gridded `image_uv_arr` and
`weights_arr` for each polarization, FHD optionally:

1. Re-derives a slightly different `obs_out` with EoR-friendly FoV /
   `dimension` (`ps_kbinsize`, `ps_dimension`, `ps_kspan`, `ps_degpix`).
2. Computes a HEALPix conversion table `hpx_cnv` that maps each Cartesian
   image pixel to its enclosing HEALPix index at `nside` ≥
   `2^Ceil(log2 sqrt(pix_sky/12))`.
3. For each of N frequency averages
   (`n_freq_use = floor(n_freq/n_avg)`), inverse-FFT the gridded UV plane
   into image space, apply the conversion table, and accumulate per-pol
   *cubes*: `dirty_cube`, `model_cube`, `weights_cube`, `variance_cube`,
   `beam_squared_cube`. When `split_ps_export` is set, the visibilities
   are split into "even" and "odd" halves so that ε-ppsilon can compute a
   noise-debiased power spectrum.
4. The resulting `<obsid>_cube[XX|YY]_cube.sav` (and `even`/`odd`
   variants) are dumped into `Healpix/`. As of commits
   `db8706ff` / `535daaee` (recent), HEALPix cubes can be saved in
   double precision via `obs.double_precision`.

`integrate_healpix_cubes.pro` is the multi-obsid combiner used to make
final integrated cubes for power-spectrum estimation. It can take either
a `.sav` glob with index range arguments, or a text file listing files,
and handles `even`/`odd` mixing and pol mixing with explicit warnings.

---

## 9. Inputs / outputs

### 9.1 Input files

| Type | Format | Loader |
|---|---|---|
| Visibilities | UVFITS (AIPS multi-source) | `uvfits_read.pro` (single-table parse, supports >4 spectral dims via `uvfits_spectral_dimension`) |
| Visibilities (cached) | IDL `.sav` | `restore_vis_savefile=1` branch in `uvfits_read` |
| Antenna table | AIPS `AN` extension | `fhd_struct_init_layout.pro` |
| Calibration | `.sav`, `.txt`, `.npz`/`.npy`, calfits | `vis_calibrate.pro` (case statement) |
| Bandpass | `.txt`, `.fits` | `vis_cal_bandpass.pro`, `transfer_bandpass.pro` |
| Cable / dipole metadata | `.txt` | `instrument_config/<instr>_cable_length.txt`, `<instr>_cable_reflection_coefficients.txt`, `mwa_dead_dipole_list.txt` |
| Source catalogs | `.sav` (IDL struct), `.fits`, BBS, RTS, SKYH5 (HDF5) | `load_source_catalog.pro`, `load_skyh5_source_catalog.pro`, `convert_rts_source_list.pro` |
| Diffuse model | `.sav` HEALPix, `.h5` SKYH5 | `load_diffuse_healpix_map.pro`, `load_skyh5_diffuse_healpix_map.pro` |
| EoR uvf cube | IDL `.sav` with `eor_uvf_cube`, `uv_arr`, `freq_arr` | `vis_simulate.pro` lines 75–123 |
| EoR bubble cube | HDF5 `/spectral_info/spectrum`, `/spectral_info/freq` | `eor_bubble_sim.pro` |
| MWA beam Z/J matrix | `mwa_ZMatrix.fits`, `mwa_Jmatrix.fits`, `mwa_LNA_impedance.sav` | used by `mwa_beam_setup_*` |
| HERA / PAPER / LOFAR beams | `.sav` (HERA), `.fits` (PAPER), `.h5` Hamaker (LOFAR) | per-instrument init files |

`inputs.md` is the human-facing index. It identifies preferred MWA
ingestion settings (ASVO with 2-s/80-kHz time/freq, 80-kHz edge width,
zenith pointing 1061315448 EoR0 field).

### 9.2 Output products

`outputs.md` is the canonical reference for every field. The directory
layout under `<output_directory>/fhd_<version>/` is:

```
metadata/      _hdr.sav, _obs.sav, _params.sav, _layout.sav,
               _settings.txt, _status.sav, _log.txt
beams/         _beams.sav (psf), _antenna.sav, _jones.sav
calibration/   _cal.sav, _bandpass.txt, gain plots .png
vis_data/      _vis_<XX|YY|XY|YX>.sav, _vis_model_<…>.sav, _flags.sav
grid_data/     _uv_<XX|YY|…>.sav, _uv_weights_<…>.sav, _uv_model_<…>.sav
mapfn/         _mapfn_<XX|YY|…>.sav  (sparse holographic MF)
output_data/   per-pol dirty/model/residual FITS, source catalogs
output_images/ per-pol PNG previews, calibration plots
Healpix/       healpix_cube .sav (per pol), even/odd splits for PS
```

Per `examples.md`, the typical EoR run produces, alongside the per-obs
products, the multi-obs HEALPix outputs assembled by
`integrate_healpix_cubes`.

### 9.3 Catalogs distributed with the repo

`catalog_data/` ships ready-to-use catalogs (see also `inputs.md`):

| File | Sky / scope |
|---|---|
| `GLEAM_EGC_catalog.sav`, `GLEAM_EGC_v2_181MHz.sav`, `GLEAM_EGC_catalog_KGSscale_ssextended.sav` | GLEAM extragalactic, multiple variants |
| `GLEAM_plus_rlb2017.sav`, `GLEAM_v2_plus_rlb2019.sav`, `GLEAM_v2_plus_gaussian_sources_rlb2019.sav` | GLEAM + bright A-team (some with Gaussian Fornax) |
| `GLEAMIDR4_181_consistent.sav` | GLEAM IDR4 at 181 MHz |
| `MRC_calibration_catalog.sav`, `MRC_full_radio_catalog.fits` | MRC full / cal |
| `mwa_calibration_source_list*.sav` | MWA-specific cal lists with/without Fornax |
| `mwa_commissioning_source_list*.sav` | Commissioning-era lists |
| `mwa_galactic_center_catalog.sav` | Galactic-centre catalog |
| `eor01_calibration_source_list.sav`, `eor1_calibration_source_list.sav` | EoR field cal lists |
| `lambda_haslam408_dsds.fits`, `gsm_150MHz.sav`, `components.fits`, `component_maps_408locked.fits` | diffuse models / GSM |
| `master_sgal_cat.sav`, `master_sgal_fornax_cat.sav`, `vlssr_*` | combined catalogs |
| `diffuse_map_Byrne2021.sav` | Byrne et al. 2022 diffuse map |
| `simulation/eor_power_1d.idlsave`, `flat_power_1d.idlsave` | EoR power-spectrum priors used by `eor_sim` |
| `simulation/100_source_plaw_*.sav`, `1000_source_plaw_*.sav`, `test_source_*` | synthetic sources |

---

## 10. Testing

There is **no formal test suite**. There are four documented "smoke
tests" in the README installation block (`cgHasImageMagick`, `astrolib`,
`init_healpix`, `imagefast` round-trip).

The closest thing to integration tests is the `examples.md` recipe set
(firstpass, deconvolution, drift scan, MWA Phase II, calibration-only,
Gaussian-decomposition beams, modified gridding kernel, in-situ
simulation) — each is a wrapper template intended to be run against
real data. The canonical zenith-pointed sanity input is **MWA obsid
1061315448** (EoR0 field, 23 Aug 2013) per `inputs.md`.

There is also an embedded "self-check" inside `array_simulator.pro`:

```idl
test_vis = max(abs(*vis_arr[0]))
if test_vis eq 0 then print, "Visiblities are probably identically zero,
   you should check very carefully!"
```

---

## 11. Configuration & extension

### 11.1 "Configuration" = wrappers

FHD does not parse YAML/JSON. A user wrapper is a `.pro` file that:

1. Defines a few mandatory variables (`obs_id`, `output_directory`,
   `version`, `vis_file_list`).
2. Calls `fhd_path_setup` to build the output directory.
3. Calls `eor_wrapper_defaults` (lives in the separate `pipeline_scripts`
   repo) to bundle keyword defaults into a structure `extra`.
4. Calls `general_obs, _Extra=extra`.

The full keyword grammar is in `dictionary.md` (~750 lines, alphabetical
sections: Beam / Calibration / Deconvolution / Export / Flagging /
Gridding / HEALPix / In-situ / Misc / Output / Plotting). Many keywords
override others — `dictionary.md` flags those explicitly.

### 11.2 Adding an instrument

To support a new instrument `xyz`, supply:

| File | Purpose |
|---|---|
| `instrument_config/xyz_beam_setup_init.pro` | Returns `antenna_str` with dipole layout, gains, delays |
| `instrument_config/xyz_beam_setup_gain.pro` | Returns the per-frequency complex voltage gain pattern |
| `instrument_config/xyz_calibration_source_list.sav` | Default calibration catalog |
| Optionally `simulation/Instrument_configuration/xyz_simulation_instr_config.pro` | Default header / params for `array_simulator` |
| Optionally `instrument_config/xyz_cable_length.txt`, `xyz_cable_reflection_coefficients.txt` | Bandpass cable groups |

Then pass `instrument='xyz'` (lower case) at the wrapper level.

The repo currently ships configs for `mwa`, `mwa2`, `hera`, `paper`,
`lofar` (PR #334).

### 11.3 Adding a deconvolution stage

`fhd_init.pro` is the single source of truth for deconvolution defaults —
extend the returned struct, then thread the new flag through
`fast_holographic_deconvolution.pro`.

### 11.4 Catalog format

A "source list" is an IDL `STRUCT[]` produced by `source_comp_init`. Each
entry has fields `{ id, x, y, ra, dec, ston, freq, alpha, gain, flag,
extend, flux, shape }` where `flux` is a sub-struct with fields named
after polarisations (`I, Q, U, V` or `XX, YY, XY, YX`) and `extend` is a
pointer to a child source list for extended sources. Catalogs in
`catalog_data/` are simply IDL `.sav` files containing one such array.

---

## 12. Notable internals & numerics

* **Numerical precision** — controlled by `obs.double_precision` (boolean).
  All gridded UV planes, weights, variance, and HEALPix cubes are built
  with `Dcomplexarr/Dblarr` instead of `Complexarr/Fltarr` when this is
  set. As of recent commits (`db8706ff`, `535daaee`) this propagates
  through HEALPix cube outputs as well.
* **Coherency / Stokes convention** — `stokes_cnv.pro` documents the
  convention inline (lines 55–60):

  ```
  I = xx* + yy*    Q = xx* - yy*
  U = xy* + yx*    V = i (xy* - yx*)
  ```

  Note this is *not* the (1/2)-factored convention used in some other
  packages — FHD's coherency matrix yields `V_XX + V_YY = I`.
* **Spectral indexing** — `obs.alpha` is the global spectral index used
  during gridding when `grid_spectral` / `degrid_spectral_terms` are set.
  Per-source overrides live in the source list (`source.alpha`).
* **DFT vs FFT cutoff** — per-component DFTs (`source_dft_multi`) keep
  sub-pixel accuracy; the cutoff is `dft_threshold = 1/((2π)² · dimension)`
  by default. Components with `flux < dft_threshold * peak` are
  truncated to an analytic Gaussian kernel approximation.
* **Spectral gradients in gridding** — when `grid_spectral` is set,
  `visibility_grid.pro` accumulates three additional UV planes
  (`spectral_A`, `spectral_B`, `spectral_D`) and computes
  `spectral_uv = (A − N·B·image_uv) / (D − B²)`, the per-pixel best-fit
  spectral slope.
* **Conjugation symmetry exploit** — FHD always grids only the
  `vv ≤ 0` half plane and reflects, halving the computation.
* **Kernel reuse hashing** — `xyf_i = (x_off + y_off*psf_resolution +
  fbin*psf_resolution²) · group_max + group_id` (visibility_grid line
  247) is the key to the algorithm's speed: identical hyper-resolution
  kernel evaluations are coalesced before the matrix multiply.
* **Map-fn sparsity** — the holographic mapping function is *huge* in
  principle but extremely sparse; FHD keeps it as a 2-D pointer array of
  `Dcomplexarr(2*psf_dim, 2*psf_dim)` blocks, materialised only where
  baselines actually contribute.
* **`heap_gc`** — called at the top of each major routine to reclaim
  pointer memory.
* **Parallelism** — only via `MATRIX_MULTIPLY` `TPOOL_MIN_ELTS` and
  per-obsid distribution from external schedulers.
* **Vendored MPFIT** — `fhd_utils/mpfit.pro` (144 KB) and
  `mpfit2dfun.pro` (29 KB) are Markwardt's IDL non-linear least-squares
  package, used by Gaussian source fitting and beam decomposition.

---

## 13. Simulation in detail

Three layered entry points exist:

### 13.1 `vis_simulate` (function, called from `array_simulator`)

Returns a `Ptrarr(n_pol)` of model visibilities. Operations:

1. Build catalog via `generate_source_cal_list` (if `include_catalog_sources`).
2. `source_dft_model` → `source_model_uv_arr` (per-pol UV planes for
   point-like components).
3. Optionally compute beam² image cubes via `beam_image_cube` and save
   as `<obsid>_initial_beam2_image.sav`.
4. Optionally generate or restore a 3-D `model_uvf_cube`:
   * From a passed-in `model_image_cube` (Jy/pixel via
     `(degpix·DtoR)²` scaling and FFT shift).
   * From an EoR `model_uvf_cube` saved in `eor_uvf_cube_file` (with
     consistency checks on `uv_arr`, `freq_arr`, dimension).
   * From a fresh `eor_sim` call (uses
     `catalog_data/simulation/eor_power_1d.idlsave` plus cosmology
     conversions `cosmology_measures` to draw a 3-D Gaussian random field
     consistent with the EoR power spectrum).
5. Optionally add `eor_bubble_sim` HEALPix bubble cubes (HDF5).
6. Optionally add a galaxy/diffuse model.
7. Combine the source UV array with the 3-D cube (`Rebin` or
   `rebin_complex` for complex types).
8. Loop over frequencies (or do a 2-D shortcut when the model is
   frequency-independent), calling `visibility_degrid` per pol/freq.
9. Optionally add `randomn` thermal noise with sigma 28 Jy (default) per
   visibility per pol.
10. Save model visibilities; return the pointer array.

### 13.2 `eor_sim` (`simulation/eor_sim.pro`)

Generates a synthetic EoR cube on a `(u_arr, v_arr, freq_arr)` grid:

* `redshifts = 1420.40 / freq_MHz − 1`.
* `cosmology_measures, redshifts, comoving_dist_los = comov_dist_los`
  (uses the standard ε-ppsilon cosmology helper).
* Jy ↔ mK·sr unit conversion via
  `c² · 1e3 / (2 · f² · k_B)`, then converted to mK·Mpc² using the
  comoving-distance squared.
* k-space binning: `kx = u · 2π / z_mpc_mean`, etc.
* Power loaded from `catalog_data/simulation/eor_power_1d.idlsave`
  (`k_centers`, `power`).
* When `flat_sigma` is set, returns a flat-power Gaussian random field
  scaled to `max(power)`.

### 13.3 `eor_bubble_sim`

Reads bubble cubes from an HDF5 file (`/spectral_info/spectrum` and
`/spectral_info/freq`), selects pixels within `select_radius` (default
20°) of `obs.obsra/obsdec` via `query_disc`, and projects to UV using the
slant-orthographic projection (`projection_slant_orthographic.pro`).

### 13.4 `in_situ_sim_setup`

Wires "model visibilities act as observed visibilities" so a user can run
the *full* pipeline (calibration → gridding → deconvolution → cubes) on
synthetic data (see `examples.md` §"In situ simulation"):

1. Restore previously-saved model visibilities from a directory of
   `<obsid>_vis_model_<XX|YY>.sav` files (or read them from a uvfits).
2. Optionally add EoR visibilities from `eor_vis_filepath` (with optional
   `enhance_eor` multiplicative scale).
3. Optionally add thermal noise from `vis_noise_simulation.pro` or a
   pre-saved noise structure (`sim_noise=path/to/dir`).
4. Optionally inject a `extra_vis_filepath` uvfits' visibilities into the
   stream.

### 13.5 Header / params synthesis

`uvfits_header_simulate.pro` and `uvfits_params_simulate.pro` (in
`simulation/`) build synthetic AIPS-style headers/`params` structures so
that `fhd_struct_init_obs` can construct `obs` even when no real uvfits
exists. The instrument-specific `*_simulation_instr_config.pro` files in
`simulation/Instrument_configuration/` provide MWA / HERA / PAPER
defaults.

---

## 14. Cross-cutting observations

### 14.1 What FHD assumes (per `assumptions.md`)

* **Stationary sky in a snapshot** — beam/PSF computed once per
  observation; recommendation is < 2 min snapshots phased to zenith.
* **No w-projection / w-stacking** — array must be relatively coplanar;
  long-baseline integration must be done in image space.
* **Point-source-component sky** — extended emission and diffuse maps are
  represented as point-source aggregates internally; the spectral slope
  is a single power-law (per-source `alpha` overrides exist).
* **(Cosmology assumption marker)** — the file marks "Maybe Bryna can
  illuminate" as a TODO.

### 14.2 Things that look like limitations or TODOs

* No tests, no CI configuration in this repo (Jekyll config only).
* The `production` keyword in `fhd_main.pro` (line 301) shells out to
  `fhd_database_filler.py` — this script is not in the FHD repo (lives
  in `MWA_Tools`) and is MWA-site-specific.
* `fhd_setup.pro` carries commented-out `GPU_enable` machinery — GPU
  paths are vestigial.
* `dictionary.md` headlines itself as "a work in progress; please add
  keywords as you find them"; not all keywords in source are listed.
* `fhd_core/obsolete/`, `fhd_utils/obsolete/`, `fhd_output/obsolete/`,
  `instrument_config/obsolete/`, `simulation/obsolete/` retain
  legacy implementations (e.g. `visibility_grid_GPU.pro`, `fhd_sim.pro`,
  `holographic_source_model.pro`, `fhd_output.pro`) that are kept for
  reference.
* The "Gaussian source models" path in `source_dft_multi.pro` is gated
  by `gaussian_source_models` and `tag_exist(source_array,'shape')`;
  per `inputs.md`, `GLEAM_v2_plus_gaussian_sources_rlb2019.sav` is "not
  yet been fully tested".

---

## 15. Directory size summary

```
fhd_core/                  ~104 .pro files across 10 subfolders
fhd_utils/                  ~30 .pro files
fhd_output/                 ~17 .pro files
simulation/                 ~14 .pro files
instrument_config/         ~12 .pro files + binary models
catalog_data/              ~30 .sav / .fits assets
Observations/                2 .pro + 6 .idlsave HEALPix index files
.md docs                    8 in repo root (~150 KB total)
TOTAL                       256 .pro files
```

Largest source files:

| Size | File |
|---|---|
| 144 KB | `fhd_utils/mpfit.pro` (vendored MPFIT) |
| 39 KB | `fhd_output/fhd_quickview.pro` |
| 35 KB | `fhd_output/obsolete/fhd_output.pro` |
| 30 KB | `fhd_core/HEALPix/integrate_healpix_cubes.pro` |
| 29 KB | `fhd_utils/mpfit2dfun.pro` |
| 22 KB | `fhd_core/deconvolution/fast_holographic_deconvolution.pro` |
| 20 KB | `fhd_core/fhd_main.pro` |
| 20 KB | `fhd_core/gridding/visibility_grid.pro` |
| 19 KB | `fhd_core/beam_modeling/beam_setup.pro` |
| 19 KB | `fhd_core/deconvolution/fhd_multi.pro` |
| 16 KB | `fhd_core/calibration/vis_cal_polyfit.pro` |
| 16 KB | `simulation/vis_simulate.pro` |

---

## 16. Citation guidance (from `README.md` and `publications.md`)

> "Please cite [Sullivan et al 2012](https://arxiv.org/abs/1209.1653) and
> [Barry et al 2019a](https://arxiv.org/abs/1901.02980) when publishing
> data reduction from FHD."

Maintainers note (per README §Maintainers): "FHD was built by Ian
Sullivan and the University of Washington radio astronomy team.
Maintainance is a group effort split across University of Washington
and Brown University, with contributions from University of Melbourne and
Arizona State University."

---

## 17. Quick-reference cheatsheet

```idl
; Build paths and call the canonical EoR firstpass driver
pro my_run_script
  obs_id            = '1061315448'
  output_directory  = '/data/out/'
  version           = 'jul2026'
  vis_file_list     = '/data/raw/' + obs_id + '.uvfits'

  fhd_file_list = fhd_path_setup(vis_file_list, $
                                 version=version, $
                                 output_directory=output_directory)
  healpix_path  = fhd_path_setup(output_dir=output_directory, $
                                 subdir='Healpix', $
                                 output_filename='Combined_obs', $
                                 version=version)

  eor_wrapper_defaults, extra        ; from pipeline_scripts repo
  fhd_depreciation_test, _Extra=extra

  general_obs, _Extra=extra
end
```

To run a simulation only:

```idl
array_simulator, vis, weights, obs, status, psf, params, jones, $
                 instrument='mwa', n_pol=2, $
                 include_catalog_sources=1, $
                 catalog_file_path='catalog_data/100_source_plaw_1-1000mJy.sav', $
                 eor_sim=1, include_noise=1, $
                 snapshot_healpix_export=1, export_images=1, $
                 file_path_fhd='/data/sim/fhd_test/sim01'
```

To deconvolve only:

```idl
general_obs, deconvolve=1, $
             max_deconvolution_components=100000, $
             return_decon_visibilities=1, $
             deconvolution_filter='filter_uv_uniform', $
             gain_factor=0.1, smooth_width=32, filter_background=1, $
             pad_uv_image=1, ring_radius=0, $
             vis_file_list=vis, version='decon_v1', $
             output_directory=output_directory
```

---

*End of reference.*
