# RASCIL — Exhaustive Reference

> *Radio Astronomy Simulation, Calibration and Imaging Library — the SKA-SDP reference Python+xarray pipeline for full-Stokes RIME simulation, self-calibration, and Dask-distributed continuum/spectral-line imaging.*

This document is an in-depth, source-grounded reference for the [RASCIL](https://gitlab.com/ska-telescope/external/rascil-main) package as vendored in `simulators/rascil/`. It is written from a complete reading of `rascil/` (74 Python files, ~17.6k LOC), the `apps/` CLI applications, the `workflows/rsexecute/` Dask-distributed orchestration layer, the `processing_components/` libraries, the `pyproject.toml` and `CHANGELOG.md`, the `data/` directory, and the `examples/` notebooks. It is intended as a primary technical companion for integrating RASCIL into other code (e.g. RadioSim simulator/calibration plumbing, SDP pipeline development).

---

## 1. What RASCIL is

RASCIL is a Python implementation of radio interferometry calibration and imaging that exposes the **RIME (Radio Interferometer Measurement Equation)** workflow on top of `xarray`-backed data containers. It is the SKA Square Kilometre Array's Science Data Processor reference library and provides:

* **Full data-flow primitives** — `Visibility`, `GainTable`, `PointingTable`, `Image`, `GridData`, `ConvolutionFunction`, `SkyComponent`, `SkyModel`. (As of RASCIL 2.0 these are imported from the sibling package `ska-sdp-datamodels`; RASCIL itself focuses on the algorithms.)
* **Simulation** of MID/LOW visibilities, primary beams, voltage-pattern (E-Jones) effects, antenna pointing errors (static, dynamic, wind-driven), troposphere/ionosphere atmospheric screens, dish-surface Zernike errors, polarisation leakage, RFI, and noise.
* **Calibration** — chains of Jones matrices `T → G → B → P → ...` solved by `solve_calibrate_chain`, applied by `apply_calibration_chain`. Optional DP3 backend for SKA-style large solves.
* **Imaging** — direct Fourier transform (DFT), 2D FFT, w-stacking via the **Nifty Gridder** (`context="ng"`), AW-projection via `griddata`/`kernels`, plus deconvolvers (Hogbom, multi-scale, MS-MFS / `mmclean`, RADLER).
* **Pipelines** — Continuum Imaging Pipeline (`CIP`), Iterative Calibration & Imaging (`ICAL`), Spectral Line Imaging — all expressed as Dask graphs by the `*_rsexecute_workflow` family.
* **CLI applications** under `rascil/apps/` for end-to-end use without writing Python.
* **Distributed execution** through the **`rsexecute`** singleton, a thin wrapper that lets identical code run as plain Python, Dask delayed graphs, or DALiuGE graphs.

The package is the convergence of about a decade of SKA-SDP work (formerly ARL → RASCIL). Releases up to and including **2.1.0** are vendored. The upstream README explicitly states RASCIL is **no longer maintained** beyond 2.0.0; new development continues in the `ska-sdp-datamodels` and `ska-sdp-func-python` packages that RASCIL now depends on.

License: **Apache-2.0** (`LICENSE`).

---

## 2. Package layout

```
simulators/rascil/
├── CHANGELOG.md                   # Release history 0.7.1 → 2.1.0
├── CODEOWNERS / CONTRIBUTORS / LICENSE
├── README.md                      # WARNING about end-of-maintenance
├── pyproject.toml                 # poetry 2.x; deps and extras
├── poetry.lock
├── setup.cfg / Makefile / make/   # ska-cicd-makefile submodule
├── docker/ / docs/ / examples/ / util/ / tests/
├── data/                          # Submodule with test fixtures, beams, GLEAM
│   ├── ska1low.cfg
│   ├── models/                    # GLEAM_EGC.fits, M31, S3 catalogs, beams,
│   │                                Zernike VPs, MID/LOW VP FEKO maps
│   ├── misc/                      # configurations
│   └── vis/                       # sample visibility files
└── rascil/                        # The Python package
    ├── __init__.py                # imports phyconst, processing_components, workflows
    ├── version.py                 # __version__ = "2.1.0"
    ├── phyconst.py                # c_m_s, sidereal_day_seconds
    ├── get_rascil_data.py         # downloader for RASCIL_DATA + casacore measures
    ├── apps/                      # End-user CLI applications (see §11)
    │   ├── apps_parser.py         # Reusable argparse fragments
    │   ├── common.py              # display_ms_as_image
    │   ├── rascil_imager.py       # CIP/ICAL/invert/load front-end
    │   ├── rascil_advise.py       # advise_wide_field over an MS
    │   ├── rascil_rcal.py         # Real-time calibration simulator
    │   ├── rascil_sensitivity.py  # MID sensitivity calculator
    │   ├── rascil_vis_ms.py       # uv-coverage and visibility plotting
    │   ├── rascil_image_check.py  # QA-statistic sanity gate
    │   ├── imaging_qa_main.py     # PyBDSF source-finding QA
    │   ├── imaging_qa/            # Diagnostics + index generator
    │   └── performance_analysis.py# Profile JSON plotting
    ├── processing_components/     # Algorithm primitives (see §6–§10)
    │   ├── parameters.py          # rascil_path / rascil_data_path / get_parameter
    │   ├── calibration/           # operations.py, iterators.py
    │   ├── flagging/              # operations.py (incl. AOFlagger)
    │   ├── griddata/              # convolution_functions.py, kernels.py
    │   ├── image/                 # operations.py, gradients.py
    │   ├── imaging/               # imaging_params.py
    │   ├── simulation/            # 8 modules (see §7)
    │   ├── skycomponent/          # plot_skycomponent.py
    │   ├── skymodel/              # operations.py
    │   ├── util/                  # compass, performance, uvw, install checks
    │   ├── visibility/            # base.py (UVFITS reader), visibility_fitting.py
    │   └── xarray/                # operations.py — generic xarray↔FITS
    └── workflows/                 # rsexecute-orchestrated graphs (see §12)
        └── rsexecute/
            ├── execution_support/ # rsexecute.py, get_dask_client
            ├── calibration/       # calibrate_list_rsexecute_workflow
            ├── image/             # image map / sum / gather workflows
            ├── imaging/           # predict / invert / restore / deconvolve / weight / taper
            ├── pipelines/         # ICAL, continuum, spectral-line skymodel pipelines
            ├── simulation/        # simulate, corrupt, MID/LOW standard simulations
            ├── skymodel/          # SkyModel-based predict/invert/deconvolve/restore
            └── visibility/        # MS readers, freq/time concatenation
```

`rascil/__init__.py` runs `check_data_directory(fatal=False)` on import — RASCIL silently logs a warning if the `data/` tree (or the `RASCIL_DATA` env-var override) is missing or contains git-LFS pointer stubs.

---

## 3. Versioning, Python, dependencies

`pyproject.toml` declares:

```toml
[tool.poetry]
name = "rascil"
version = "2.1.0"
[tool.poetry.dependencies]
python = "^3.10,<3.13"
astropy = "^6.1"
bdsf = "^1.10"
dask = {version = ">=2024.12, <2025.4", extras = ["diagnostics"]}
dask-memusage = "^1.1"
distributed = ">=2024.12, <2025.4"
h5py = "^3.11"
jupyter = "^1.0"
matplotlib = "^3.8"
numpy = "^1.26"
pandas = "^2.2"
python-casacore = "^3.5"
reproject = "^0.14"
scipy = "^1.13"
seqfile = "^0.2"
ska-sdp-datamodels = "^1.0"
ska-sdp-func-python = "^1.0"
xarray = "^2024.9"
tabulate = "^0.9"
[tool.poetry.extras]
astron = ["aoflagger", "dp3", "radler"]   # Linux-only optional flagging/cal/deconv
ska-sdp-func = ["ska-sdp-func"]            # Optional GPU-accelerated kernels
```

Two SKA sibling packages do most of the data-model and lower-level algorithmic work:

* **`ska-sdp-datamodels`** — provides `Visibility`, `GainTable`, `PointingTable`, `Image`, `Configuration`, `SkyComponent`, `SkyModel`, `GridData`, `ConvolutionFunction`, `PolarisationFrame`, the `create_visibility`, `create_visibility_from_ms`, `create_named_configuration`, `create_gaintable_from_visibility`, `create_pointingtable_from_visibility`, `import_image_from_fits`, `import_skycomponent_from_hdf5`, `export_*` functions, etc. RASCIL imports them directly.
* **`ska-sdp-func-python`** — `solve_gaintable`, `solve_calibrate_chain`, `apply_calibration_chain`, `apply_gaintable`, `dp3_gaincal`, `predict_visibility`, `invert_visibility`, `dft_skycomponent_visibility`, `advise_wide_field`, `create_image_from_visibility`, `image_gather_channels`, `deconvolve_cube`, `deconvolve_list`, `radler_deconvolve_list`, `restore_cube`, `find_skycomponents`, `apply_beam_to_skycomponent`, `convert_polimage_to_stokes`, `azel_to_hadec`, `hadec_to_azel`, `calculate_azel`, `calculate_visibility_hourangles`, `calculate_visibility_parallactic_angles`, `skycoord_to_lmn`, `lmn_to_skycoord`, `xyz_to_uvw`, `simulate_point_antenna`, `create_pb`, `create_mid_allsky`, `create_low_test_beam`, `convert_azelvp_to_radec`, `normalise_vp`, `find_skycomponent_matches`, `fit_skycomponent`, `fit_skycomponent_spectral_index`, `find_skycomponents_frequency_taylor_terms`, `calculate_skycomponent_list_taylor_terms`, `gather_skycomponents_from_channels`, `concatenate_visibility`, `concatenate_visibility_frequency`, `divide_visibility`, `integrate_visibility_by_channel`, `convert_visibility_to_stokesI`, `taper_visibility_gaussian`, `coordinates`, `grdsf`, `w_beam`, `apply_jones`, `pad_image`, `fft_image_to_griddata_with_wcs`, `fit_psf`, `image_gather_facets`, `image_scatter_facets`, `image_scatter_channels`, etc.

The `rascil-main` repo also bundles `ska-cicd-makefile` as a Git submodule; clone with `--recurse-submodules`.

`pyproject.toml` exposes a single console script:

```toml
[tool.poetry.scripts]
get_rascil_data = "rascil.get_rascil_data:main"
```

`get_rascil_data` (`rascil/get_rascil_data.py`) downloads `https://ska-telescope.gitlab.io/external/rascil-main/rascil_data.tgz` to `$RASCIL/rascil_data` and rsyncs the casacore geodetic measures from `casa-rsync.nrao.edu`, writing `~/.casarc` if absent.

---

## 4. Physical constants and path helpers

`rascil/phyconst.py` is intentionally minimal:

```python
c_m_s = 299792458.0
sidereal_day_seconds = 86164.090530833
```

Most physical constants RASCIL needs come from `astropy.constants` or `scipy.constants`; sensitivity uses `scipy.constants.Boltzmann`. Note that the duplicate `C_M_S` from `ska_sdp_datamodels.physical_constants` is what the UVFITS reader uses.

`rascil/processing_components/parameters.py` provides three helpers (`__all__ = ["rascil_path", "rascil_data_path", "get_parameter"]`):

* **`rascil_path(path)`** — resolve a path relative to the RASCIL repo root. Honours `RASCIL` env var (default: parent of `rascil/`).
* **`rascil_data_path(path, check=True)`** — resolve a path under `data/`. Honours `RASCIL_DATA` env var. With `check=True` (default) raises `FileNotFoundError` if the data directory does not exist. This is the function loaders such as `create_low_test_skycomponents_from_gleam` use to find `models/GLEAM_EGC.fits`, `ska1low.cfg`, etc.
* **`get_parameter(kwargs, key, default=None)`** — the canonical kwargs-fetcher used throughout RASCIL. Standard names enforced by convention include `vis`, `sc`, `gt`, `conf`, `im`, `qa`, `log`, `loop_gain`, `niter`, `eps`, `threshold`, `fractional_threshold`, `G_solution_interval`, `phaseonly`, `phasecentre`, `spectral_mode`.

---

## 5. Data containers (provided by `ska-sdp-datamodels`)

RASCIL operates on **xarray-Dataset subclasses** via accessor objects (`.visibility_acc`, `.image_acc`, `.gaintable_acc`, `.cf_acc`, `.griddata_acc`). The shapes:

| Container | Key dims / fields |
|-----------|-------------------|
| `Visibility` | `(time, baselines, frequency, polarisation)` for `vis`, `flags`, `weight`, `imaging_weight`; plus `uvw` (time, baselines, 3); `configuration`, `phasecentre`, `polarisation_frame`. Convenience: `visibility_acc.uvw_lambda`, `.u`, `.v`, `.w`, `.flagged_vis`, `.flagged_weight`, `.flagged_imaging_weight`, `.nants`, `.nchan`, `.npol`, `.nvis`, `.select_r_range`, `.select_uv_range`, `.performance_visibility`. (RASCIL has fully retired the older `BlockVisibility`/`Visibility` split; everything is `Visibility` now — see CHANGELOG for the rename in 1.0.) |
| `GainTable` | `gain[time, ant, frequency, receptor1, receptor2]`, `weight`, `residual`, `interval`, `time`, `phasecentre`, `receptor_frame`, `configuration`. Accessor: `.nants`, `.nchan`, `.nrec`. |
| `PointingTable` | `pointing[time, ant, frequency, receptor, angle (az/el)]`, `nominal[…]`, `time`, `interval`, `frequency`, `receptor_frame`, `configuration`. |
| `Image` | `pixels[chan, pol, dec, ra]` (canonical 4-axis), with WCS in `image_acc.wcs`. `image_acc.qa_image()` returns a `QualityAssessment` with `.data` dict (`max`, `min`, `maxabs`, `rms`, `sum`, `medianabs`, `medianabsdevmedian`, `shape`). |
| `Configuration` | `xyz[ant,3]`, `names`, `mount`, `diameter`, `vp_type`, `location` (astropy `EarthLocation`). |
| `GridData` / `ConvolutionFunction` | UV-plane gridded data; `.griddata_wcs` / `.cf_wcs`. |
| `SkyComponent` | `direction` (SkyCoord), `flux[chan, pol]`, `frequency`, `name`, `shape` ("Point" or "Gaussian"), `polarisation_frame`, `params`. |
| `SkyModel` | `image: Image`, `components: list[SkyComponent]`, `mask: Image`, `gaintable: GainTable`, `fixed: bool`. |

Polarisation frames RASCIL handles (`PolarisationFrame`): `stokesI`, `stokesIQ`, `stokesIV`, `stokesIQUV`, `linear`, `linearnp`, `circular`, `circularnp`. The UVFITS reader (`processing_components/visibility/base.py`) maps the AIPS `corr_type` codes `1,2,3,4 → stokesIQUV`, `1,4 → stokesIV`, `1,2 → stokesIQ`, `-1..-4 → circular`, `-1,-4 → circularnp`, `-5..-8 → linear`, `-5,-8 → linearnp`.

The canonical image axes are `[RA---SIN, DEC--SIN, STOKES, FREQ]` (FITS order), which transposes to `[chan, pol, dec, ra]` in numpy. `replicate_image` (`testing_support.py`) creates a canonical-shape image from a 2-D template. Cellsize is in **radians**: WCS conversion is `wcs.cdelt[0] = -180 * cellsize / pi`, `wcs.cdelt[1] = +180 * cellsize / pi` (negative on RA).

---

## 6. Calibration primitives (`processing_components/calibration/`)

### `operations.py`

`__all__` exports (5 functions):

* **`append_gaintable(gt, othergt) -> GainTable`** — concatenate two GainTables (asserts matching `receptor_frame`).
* **`create_gaintable_from_rows(gt, rows, makecopy=True)`** — boolean-row selection over a GainTable, returning a deep-copy by default.
* **`gaintable_plot(gt, cc="T", title="", ants=None, channels=None, label_max=0, min_amp=1e-5, cmap="rainbow", **kwargs)`** — standard 3-panel plot (residual, amplitude, phase). For `cc="B"` (bandpass), the panels become `imshow`-style 2-D arrays vs antenna×channel; otherwise scatter vs time per antenna.
* **`multiply_gaintables(gt, dgt, time_tolerance=1e-3)`** — gt ← gt · dgt (Einstein sum `...ik,...ij->...kj` for nrec=2; element-wise for nrec=1). Asserts time alignment within `time_tolerance` seconds.
* **`concatenate_gaintables(gt_list, dim="time")`** — wraps `xarray.concat` with `data_vars="minimal", coords="minimal", compat="override"`. Default concat dim is `"time"`; use `"frequency"` when joining bandpass solves.

### `iterators.py`

* **`gaintable_timeslice_iter(gt, **kwargs)`** — generator yielding boolean row masks. `timeslice="auto"` uses unique `gt.time` values with `timeslice=0.1`s; `None` returns one slice covering the whole table; numeric values use fixed-width boxes `arange(timemin, timemax, timeslice)`. Falls back to `gaintable_slices` (count) using `linspace`.

### Calibration solving and applying

The actual solvers live in `ska_sdp_func_python.calibration`:

* **`solve_gaintable(vis, modelvis=None, gain_table=None, phase_only=False, jones_type="T", tol=1e-6, ...)`** — least-squares gain solution per `timeslice`/`frequency`. `jones_type` ∈ {`"T"`, `"G"`, `"B"`} controls atmospheric (single phase per ant), gain (one per receptor per ant), or bandpass (per channel) models.
* **`solve_calibrate_chain(vis, modelvis=None, gaintables=None, calibration_context="TG", controls=None, iteration=0, tol=1e-6)`** — solves the chain in order, e.g. `"T"`, `"TG"`, `"TGB"`.
* **`apply_calibration_chain(vis, gt_dict, calibration_context, controls, iteration, inverse=True)`** — applies the chain (inverse=True for correction).
* **`create_calibration_controls()`** — returns the canonical dict keyed by `"T"`, `"G"`, `"B"`, with sub-keys `first_selfcal`, `phase_only`, `timeslice`, `shape` ("vector"/"matrix").
* **`dp3_gaincal(vis, calibration_context, global_solution, skymodel)`** — invokes [DP3](https://dp3.readthedocs.io/) GainCal; requires `dp3` extras.

The chain order and naming (`T → G → B → P → ...`) follow the OYSTER memo conventions:

| Term | Meaning | Phase-only? | Receptor-coupled? | Frequency-resolved? |
|------|---------|-------------|-------------------|---------------------|
| T | Atmospheric phase, one per antenna | yes (default) | no | no |
| G | Electronic gain, complex per receptor | no (default) | yes | no |
| B | Bandpass, complex per channel | no | yes | yes |
| P | Polarisation leakage / D-term | no | yes | no |

The solver output is a **dictionary of GainTables keyed by context letter** (e.g. `gt_dict["T"]`).

---

## 7. Simulation primitives (`processing_components/simulation/`)

This is the largest subpackage and the most relevant for synthesising observations.

### 7.1 `testing_support.py` — sky models, test images, unittest fixtures

Exports (12 functions):

* **`create_test_image(cellsize=None, frequency=None, channel_bandwidth=None, phasecentre=None, polarisation_frame=None) -> Image`** — loads `data/models/M31_canonical.model.fits` (the M31 Hα region, widely used in ALMA simulations). Replicates the 2D plane across `frequency` and `polarisation_frame.npol` channels via `replicate_image`. WCS axes are forced to `RA---SIN, DEC--SIN, STOKES, FREQ`.
* **`create_test_image_from_s3(npixel=16384, polarisation_frame=stokesI, cellsize=0.000015, frequency=[1e8], channel_bandwidth=[1e6], phasecentre=None, fov=20, flux_limit=1e-3) -> Image`** — generates a SKA Mid simulation from the **S3-SEX** (`SKA Simulation Sky`) catalogues at `data/models/S3_151MHz_*.csv` or `S3_1400MHz_*.csv`. Selects rows by `phasecentre`/`fov`/`flux_limit`, computes a power-law spectral index from 151–610 MHz or 610–1400 MHz pairs, and inserts components onto an image plane. Fov choices: `10`, `20`, `40` deg; flux-limit choices control which catalogue file is used.
* **`create_test_skycomponents_from_s3(...)`** — same data source but returns a `list[SkyComponent]` (one per row) rather than an image.
* **`create_low_test_image_from_gleam(npixel=512, polarisation_frame=stokesI, cellsize=0.000015, frequency=[1e8], channel_bandwidth=None, phasecentre=None, kind="cubic", applybeam=False, flux_limit=0.1, flux_max=inf, flux_min=-inf, radius=None, insert_method="Nearest")`** — reads `data/models/GLEAM_EGC.fits` (Hurley-Walker+ 2017, MNRAS 464, 1146; VIII/100 in Vizier), selects sources within `radius` of `phasecentre`, fits a cubic spline through the 20 GLEAM channels (76–227 MHz), and interpolates to the requested `frequency` array. `insert_method` ∈ {`"Nearest"`, `"PSWF"`, `"Lanczos"`}.
* **`create_low_test_skycomponents_from_gleam(flux_limit=0.1, polarisation_frame=stokesI, frequency=[1e8], kind="cubic", phasecentre=None, radius=1.0)`** — same but returns components. The 20 GLEAM frequencies are hard-coded: `[76, 84, 92, 99, 107, 115, 122, 130, 143, 151, 158, 166, 174, 181, 189, 197, 204, 212, 220, 227]` (MHz). Reads `peak_flux_wide` for the flux cut and `int_flux_NNN` for spectral interpolation.
* **`create_low_test_skymodel_from_gleam(...flux_threshold=1.0, applybeam=True, telescope="LOW")`** — partitions GLEAM sources at `flux_threshold`: bright ones become `SkyComponent`s in the returned `SkyModel.components`, weaker ones are inserted into `SkyModel.image`. If `applybeam=True`, applies the LOW primary beam (`create_pb`) to both before partitioning.
* **`replicate_image(im, polarisation_frame=stokesI, frequency=[1e8])`** — extend a 2-D image to canonical 4-D `(nchan, npol, ny, nx)`.
* **`simulate_gaintable(gt, phase_error=0.1, amplitude_error=0.0, smooth_channels=1, leakage=0.0, seed=180550721, **kwargs)`** — corrupt an empty GainTable with normal-distributed phase noise (rad), log-normal amplitude noise, optional moving-average smoothing across `smooth_channels` (uses `scipy.ndimage.convolve1d` mode `"wrap"`), and optional cross-receptor `leakage` (Gaussian complex). Pure-NumPy default RNG.
* **`ingest_unittest_visibility(config, frequency, channel_bandwidth, times, vis_pol, phasecentre, zerow=False, times_are_ha=True)`** — wraps `create_visibility` and zeroes the data.
* **`create_unittest_components(model, flux, applypb=False, telescope="LOW", npixel=None, scale=1.0, single=False, symmetric=False, angular_scale=1.0, offset=[0,0])`** — places a deterministic spread of point sources around the phase centre at fractional pixel positions `(0,0), (0.2,1.1)`, plus a diagonal cross at ±1.2× spacing. Used by the test suite.
* **`create_unittest_model(vis, model_pol, npixel=None, cellsize=None, nchan=1)`** — wraps `advise_wide_field(guard_band_image=2.0, delA=0.02, oversampling_synthesised_beam=4.0)` to derive sensible defaults, then `create_image_from_visibility`.
* **`insert_unittest_errors(vt, seed=1805550721, calibration_context="TG", amp_errors=None, phase_errors=None)`** — for each calibration context letter, builds a GainTable, corrupts it with `simulate_gaintable`, applies the **inverse** (so subsequent calibration recovers it). Defaults: `phase_errors={"T":1.0, "G":0.1, "B":0.01}` (radians), `amp_errors={"T":0.0, "G":0.01, "B":0.01}`.

### 7.2 `noise.py` — thermal noise

* **`calculate_noise_visibility(bandwidth, int_time, diameter, t_sys, eta) -> sigma[nrows, nchan]`** — thermal-noise standard deviation per visibility, computed via the radiometer equation:
  `σ = √2 · k_B · T_sys / (A · η · √(B·τ)) · 10²⁶` (Jy)
  where `A = π (d/2)²`, `B·τ` is bandwidth × integration time, factor 10²⁶ converts W/m²/Hz → Jy. `k_B = 1.38064852e-23`.
* **`addnoise_visibility(vis, t_sys=None, eta=None, seed=None) -> vis`** — defaults `t_sys=20`, `eta=0.78`. Adds independent Gaussian noise to real and imaginary parts. Uses `vis.configuration.diameter[0]` (i.e. assumes a homogeneous array), `numpy.random.default_rng(seed or 1805550721)`. The TODO note acknowledges per-frequency sensitivity is not yet supported.

### 7.3 `pointing.py` — antenna pointing errors

Exports (3 functions):

* **`simulate_pointingtable(pt, pointing_error, static_pointing_error=None, global_pointing_error=None, seed=None)`** — fills a pointing table with the sum of three random components: dynamic (per-time-step) Gaussian noise (`pointing_error`, rad), static-per-antenna offsets (`static_pointing_error=[az,el]`, broadcast over time), and a global offset applied to all antennas/times. Resets `pt["pointing"].data` to zeros first; output is in radians.
* **`simulate_pointingtable_from_timeseries(pt, type="wind", time_series_type="precision", pointing_directory=None, reference_pointing=False, seed=None)`** — builds a time series from a measured PSD file. Looks up `data/models/{precision|standard|degraded}/El{15,45,90}Az{0,45,90,135,180}.dat` based on the nominal pointing of the table. The file has columns `[freq, az, el, pxel, pel]`. For `type="tracking"`, resamples/extrapolates the az/el PSDs; for `type="wind"`, uses pxel/pel. The algorithm fits log-PSD with two 5th-order polynomials (below and above the PSD peak), generates random phases, performs an iFFT, scales to arcsec then radians. With `reference_pointing=True`, subtracts t=0 sample so errors begin at zero.
* **`simulate_gaintable_from_pointingtable(vis, sc, pt, vp, vis_slices=None, scale=1.0, order=3, elevation_limit=15°, jones_type="G", **kwargs)`** — for each component in `sc`, computes the AZELGEO position of each antenna's beam centre under the pointing offsets, looks up the voltage pattern `vp` (image with `ctype = ["AZELGEO long", "AZELGEO lati", ...]`) using bicubic spline interpolation (`scipy.RectBivariateSpline`), and writes the resulting complex gain into a `GainTable`. Below `elevation_limit`, gains are set to identity. Asserts the configuration mount is `"azel"`. Heterogeneous voltage patterns are supported via `vp_type` per antenna in the configuration. The internal `_get_worldloc` function uses `astropy.SkyCoord.spherical_offsets_to` for accurate az/el offset calculation.

### 7.4 `surface.py` — dish-surface errors

* **`simulate_gaintable_from_voltage_pattern(vis, sc, vp, vis_slices=None, order=3, elevation_limit=15°, jones_type="B", **kwargs)`** — like `simulate_gaintable_from_pointingtable`, but does not use a `PointingTable`. Each component's az/el is computed at every time slice; `vp` (or list of VPs indexed by `configuration.vp_type`) is sampled, **inverted** (`numpy.linalg.inv` for 2×2 polarisation, `1/g` for scalar). Heterogeneous arrays are supported by passing a list of VPs.
* **`simulate_gaintable_from_zernikes(vis, sc, vp_list, vp_coeffs, order=3, elevation_limit=15°, jones_type="B", **kwargs)`** — each antenna's gain is a weighted sum over a list of Zernike-mode VPs with per-antenna coefficients `vp_coeffs[nant, nvp]`. Used to model dish-figure errors with a Zernike basis.

### 7.5 `atmospheric_screen.py` — TEC and tropospheric phase

Per [SDP Memo 97](https://ska-aw.bentley.com/SKAProd/Framework/Display.aspx?o=13939&t=3). Exports (5 functions):

* **`find_pierce_points(station_locations, ha, dec, phasecentre, height) -> pp[nant,3]`** — geometric ray-tracing through a flat phase screen at altitude `height`. Uses local UVW frame (`xyz_to_uvw`) zero-meaned, then offsets pierce points by `height·(l, m, n+1)`.
* **`create_gaintable_from_screen(vis, sc, screen, height=None, vis_slices=None, r0=5e3, type_atmosphere="ionosphere", reference_component=None, jones_type="B", **kwargs) -> [GainTable]`** — for each component in `sc`, looks up the screen value at each antenna's pierce point (4-D screen WCS axes: `[XX, YY, TIME, FREQ]`). Reads the screen FITS lazily with `memmap=True`. For `type_atmosphere="ionosphere"` interprets the screen as differential TEC (units: dTEC) and computes phase as `−scale · 8.44797245e9 / ν · dTEC` rad (the dispersive ionospheric formula). For `"troposphere"` the screen is phase in radians at the screen-FITS reference frequency, scaled to `vis.frequency` linearly. `scale = (r0 / 5000)^(-5/3)` rescales to the requested Fried parameter. Out-of-bounds pierce points produce zero gain and are counted; asserts at least one good pierce point per call.
* **`grid_gaintable_to_screen(vis, gaintables, screen, height=3e5, gaintable_slices=None, scale=1.0, r0=5e3, type_atmosphere="ionosphere", vis_slices=None, **kwargs) -> (newscreen, weights)`** — the inverse: averages observed gain phases into pixels of an empty screen image. No phase unwrapping.
* **`calculate_sf_from_screen(screen)`** — computes a structure function from the screen via `scipy.signal.fftconvolve`.
* **`plot_gaintable_on_screen(vis, gaintables, height=3e5, gaintable_slices=None, plotfile=None)`** — scatter plot of pierce-point phases (HSV colormap).

### 7.6 `rfi.py` — terrestrial RFI

The scenario described in the docstring assumes a remote TV station emitting 50 kW over 7 MHz that sidelobes into LOW stations with ~55–60 dB attenuation. Exports (3 public + helpers):

* **`simulate_rfi_block_prop(bvis, apparent_emitter_power, apparent_emitter_coordinates, rfi_sources, rfi_frequencies, low_beam_gain=None, apply_primary_beam=True) -> Visibility`** — main entry. For each source, calls `apply_beam_gain_for_low` (multiply by `√beam_gain`) for SKA-LOW or uses raw apparent emitter power for MID. Then `match_frequencies` aligns the RFI spectrum onto bvis frequency channels. Computes per-station-pair correlation via `calculate_station_correlation_rfi` (outer product of antenna voltages, scaled by `1e26`). Phase-rotates each emitter against the phase centre using `simulate_point_antenna(k·uvw, l, m)` per station. For MID, optionally applies per-time voltage-pattern beam gain via `apply_beam_gain_for_mid` (which uses `simulate_gaintable_from_pointingtable` + `apply_gaintable` on a sub-vis). Accumulates over sources.
* **`calculate_station_correlation_rfi(rfi_at_station, baselines)`** — outer product per (time, channel), reshaped to (time, baseline, channel, pol).
* **`calculate_averaged_correlation(correlation, time_width, channel_width)`** — wraps `ska_sdp_func_python.util.average_chunks2`.
* **`match_frequencies(rfi_signal, rfi_frequencies, bvis_freq_channels, bvis_bandwidth)`** — adds zero rows for bvis channels with no RFI overlap; uses `numpy.median` for over-sampled RFI within a bvis channel.

### 7.7 `simulation_helpers.py` — plotting, configuration, MID component construction

Exports (13 functions). The plotting helpers (`plot_visibility`, `plot_visibility_pol`, `plot_uvcoverage`, `plot_uwcoverage`, `plot_vwcoverage`, `plot_configuration`, `plot_azel`, `plot_pa`, `plot_gaintable`, `plot_pointingtable`) are matplotlib wrappers; all accept a list of `Visibility`/`GainTable`/`PointingTable` and an optional `plot_file`. Notable details:

* `plot_visibility(vis_list, y="amp"|"phase", x="uvdist"|"time", chan=0, markersize=0.2)`.
* `plot_uvcoverage` separates flagged from unflagged with red/blue colours, plots `(u, v)` and `(-u, -v)`.
* `plot_pointingtable` annotates the title with arcsec rms in az/el.

Other functions:

* **`find_times_above_elevation_limit(start_times, end_times, location, phasecentre, elevation_limit) -> valid_start_times`** — filters time samples (in seconds) by computing hour angle and using `hadec_to_azel`; asserts at least one valid time. `elevation_limit` is in **degrees**.
* **`find_pb_width_null(pbtype, frequency, **kwargs)`** — empirical primary-beam HWHM and first-null estimates for `MID`, `MID_FEKO_B1/B2/Ku`, and others. Returns `(HWHM_deg, null_az_deg, null_el_deg)`. The 1.36 GHz reference frequency is hard-coded in the formulae (`HWHM = 0.596 * 1.36e9 / freq` for MID).
* **`create_mid_simulation_components(phasecentre, frequency, flux_limit, pbradius, pb_npixel, pb_cellsize, show=False, fov=10, polarisation_frame=stokesI, flux_max=10.0, pb_type="MID", apply_pb=True)`** — convenience builder: pulls components from S3-SEX, applies the MID primary beam, filters by post-beam flux limit, and returns either `(filtered_pb_components, filtered_components)` (if `apply_pb=True`) or `(filtered_components, reference_component_index)` otherwise.

### 7.8 `skycomponents.py`

* **`addnoise_skycomponent(sc, noise=1e-3, mode="flux_central", seed=None)`** — adds Gaussian noise to either component `direction` (RA/Dec in radians), the central-frequency Stokes I flux (`mode="flux_central"`), or all flux entries (`mode="flux_all"`). Default seed: `1805550721`. Returns a list (always — wraps a single component into a list).

---

## 8. Image, GridData, and Skymodel utilities

### 8.1 `processing_components/image/operations.py`

Exports (11 functions):

* **`add_image(im1, im2)`** — element-wise addition; preserves WCS and polarisation from `im1`.
* **`show_image(im, fig=None, title="", pol=0, chan=0, cm="Greys", components=None, vmin=None, vmax=None, vscale=1.0)`** — matplotlib display with WCS projection. Overlay `components` as red plus markers.
* **`show_components(im, comps, npixels=128, fig=None, vmax=None, vmin=None, title="")`** — postage-stamp inspection of components against an image (cuts an `npixels`×`npixels` window around each `direction`).
* **`smooth_image(model, width=1.0, normalise=True)`** — Gaussian smoothing, used to apply a clean-beam.
* **`average_image_over_frequency(im)`** — reduce nchan to 1.
* **`remove_continuum_image(im, degree=1, mask=None)`** — fits a polynomial of `degree` per pixel along frequency and subtracts it.
* **`create_window(template, window_type, **kwargs)`** — used by deconvolution (clean masks).
* **`polarisation_frame_from_wcs(wcs, shape) -> PolarisationFrame`** — interprets the STOKES axis from the WCS.
* **`sub_image(im, shape)`** — extract a centred subarray.
* **`create_w_term_like(template, w, phasecentre, remove_shift=False, dopol=False)`** — creates a w-term phase-screen image (`exp(-2πi w (√(1−l²−m²)−1))`) shaped like `template`.
* **`rotate_image(im, angle=0.0, order=5)`** — `scipy.ndimage.rotate` wrapper.
* **`apply_voltage_pattern_to_image(im, vp, inverse=False)`** — applies a Jones voltage pattern to an image plane.

`gradients.py`:

* **`image_gradients(im)`** — uses `numpy.gradient` along x/y to produce two derivative images. Used by direction-dependent solvers.

### 8.2 `processing_components/skymodel/operations.py`

Exports (9 functions):

* **`partition_skymodel_by_flux(sc, model, flux_threshold=-inf) -> SkyModel`** — bright components stay as components; weak ones are inserted into the image. Logs a summary.
* **`show_skymodel(sms, psf_width=1.75, cm="Greys", vmax=None, vmin=None)`** — matplotlib visualisation.
* **`initialize_skymodel_voronoi(model, comps, gt=None) -> [SkyModel]`** — Voronoi partitioning of an image around a list of components, producing one masked SkyModel per Voronoi cell. Optional gain-table cloning per cell with `phasecentre` set to the cell's component.
* **`calculate_skymodel_equivalent_image(sm)`** — sum of masked images across a list of SkyModels.
* **`update_skymodel_from_image(sm, im, damping=0.5)`** — additive update with a damping factor (analog of clean loop gain).
* **`update_skymodel_from_gaintables(sm, gt_list, calibration_context="T", damping=0.5)`** — accumulates gain phases into each SkyModel's gain table.
* **`expand_skymodel_by_skycomponents(sm)`** — split a single SkyModel into one SkyModel per component plus one for the residual image.
* **`create_skymodel_from_skycomponents_gaintables(components, gaintables)`** — pair-wise zip into a list of single-component SkyModels.
* **`extract_skycomponents_from_skymodel(sm, im=None, **kwargs)`** — find sources above `component_threshold` (Jy) using `find_skycomponents(fwhm=3, threshold=...)` and either `fit_skycomponent` (`component_method="fit"`) or direct extraction. Updates and returns a copy of the SkyModel.

### 8.3 `processing_components/skycomponent/plot_skycomponent.py`

Exports 8 plotting helpers used heavily by `imaging_qa_main.py`:

* `plot_skycomponents_positions(comps_test, comps_ref=None, img_size=1.0, plot_file=None, tol=1e-5, plot_error=True)` — scatter / quiver of (Δra, Δdec) for matched components. Uses `find_skycomponent_matches` from `ska_sdp_func_python`. Returns `[ra_error, dec_error]` arrays.
* `plot_skycomponents_position_distance` — distance-from-centre vs |position offset|.
* `plot_skycomponents_flux(comps_test, comps_ref, ...)` — log-log flux comparison.
* `plot_skycomponents_flux_ratio(comps_test, comps_ref, ...)` — flux_ratio vs flux scatter.
* `plot_skycomponents_flux_histogram(comps_test, comps_ref, ...)` — histograms.
* `plot_skycomponents_position_quiver(comps_test, comps_ref, ...)` — vector field of position errors.
* `plot_gaussian_beam_position(comps_test, comps_ref, beam_width)` — Gaussian-fitted beam centroids.
* `plot_multifreq_spectral_index(comps_test, ...)` — uses `fit_skycomponent_spectral_index` to plot per-component α distributions.

### 8.4 `processing_components/griddata/`

* `convolution_functions.py` (`__all__` = 3): `apply_bounding_box_convolutionfunction(cf, fractional_level=1e-4)`, `calculate_bounding_box_convolutionfunction(cf, fractional_level=1e-4)`, `export_convolutionfunction_to_fits(cf, fitsfile)`.
* `kernels.py` (`__all__` = 4):
  * **`create_box_convolutionfunction(im, oversampling=1, support=1, polarisation_frame=None) -> (gcf_image, cf)`** — box-car kernel; gridding-correction function is `1/sinc(nu)`.
  * **`create_pswf_convolutionfunction(im, oversampling=127, support=8, polarisation_frame=None)`** — Prolate Spheroidal Wave Function antialiasing kernel. Forces `oversampling` to next-greatest odd number. `width = support − 2`. The kernel is built from `grdsf` (Schwab's prolate wave function) on a 2-D outer product. Each `(y, x)` slice is normalised so the central row sums to 1.
  * **`create_awterm_convolutionfunction(im, make_pb=None, nw=1, wstep=1e15, oversampling=8, support=6, use_aaf=True, maxsupport=512, polarisation_frame=None)`** — full AW-projection kernel: A-term primary beam × W-term `create_w_term_like` × optional PSWF antialiasing.
  * **`create_vpterm_convolutionfunction(...)`** — voltage-pattern kernel.

### 8.5 `processing_components/imaging/imaging_params.py`

Maps for combining vis with image grids:

* `get_frequency_map(vis, im=None) -> (spectral_mode, vfrequencymap)` — spectral mode `"channel"` (one image channel per visibility channel) or `"mfs"` (one image channel pooling all visibility channels). With `im=None`, uses the visibility channels directly. Otherwise maps via `wcs.sub(["spectral"]).wcs_world2pix`.
* `get_polarisation_map(vis, im) -> (mode_str, pol_map_fn)` — where `mode_str` is e.g. `"stokesIQUV->stokesIQUV"` or `"unknown"`, and the function maps a visibility-pol index to an image-pol index.
* `get_rowmap(col, ucol=None) -> list[int]` — generic mapping from row values to indices in the unique-value array.

### 8.6 `processing_components/visibility/`

* **`base.create_visibility_from_uvfits(fitsname, channum=None, antnum=None) -> [Visibility]`** — minimal UVFITS importer. Detects spectral windows (`NAXIS5`), reads `CRVAL4`/`CDELT4` for frequencies, builds `if_freq` from the `AIPS FQ` table, decodes `corr_type` to `PolarisationFrame`, reads antenna `STABXYZ`/`MNTSTA`/`STAXOF`/`DIAMETER` from `AIPS AN`, and builds one `Visibility` per spectral window with full `time × baseline × chan × pol` data + flags + weights. Negative weights are mapped to flags.
* **`visibility_fitting.fit_visibility(vis, sc, tol=1e-6, niter=20, verbose=False, method="trust-exact", **kwargs) -> (sc, optimize_result)`** — fits `(S, l, m)` for a single Stokes-I component. Provides `J`, `Jboth` (Jacobian = `[−2Σ Re(V·exp), 4πS Σ u Im, 4πS Σ v Im]`), and an analytic Hessian; supports SciPy methods `BFGS`, `CG`, `Powell`, `Nelder-Mead`, `L-BFGS-B`, `trust-ncg`, `trust-exact` (default), `trust-krylov`. Bounds are hard-coded to `(None, None), (-0.1, -0.1), (-0.1, 0.1)` for L-BFGS-B (note the negative-only `l` bound is a known oddity). Initialises from `skycoord_to_lmn(sc.direction, vis.phasecentre)`.

### 8.7 `processing_components/util/`

* **`compass_bearing.calculate_initial_compass_bearing(pointA, pointB)`** — initial bearing in degrees between two `(lat, lon)` tuples.
* **`installation_checks.check_data_directory(verbose=False, fatal=True)`** — opens `models/S3_151MHz_10deg.csv`; warns if it's a Git-LFS pointer stub; raises `FileNotFoundError` if missing and `fatal=True`.
* **`uvw_coordinates.uvw_ha_dec(antxyz, ha, dec) -> uvw[N,3]`** — CASA-convention UVW from antenna ITRF-like XYZ and (ha, dec). The right-handed convention is W toward the source, baseline = `ant2 − ant1` for `index(ant1) < index(ant2)`. Implements the standard rotation:
  ```
  u = sin(h)·x + cos(h)·y
  v = -sin(δ)·(cos(h)·x − sin(h)·y) + cos(δ)·z
  w = +cos(δ)·(cos(h)·x − sin(h)·y) + sin(δ)·z
  ```
  The (l, m, n) convention is l → +RA, m → +Dec, n → toward source.
* **`performance.*`** — JSON profiling helpers (see §11.7):
  * `git_hash()` — `git rev-parse HEAD` or `"unknown"`.
  * `performance_environment(performance_file, indent=2, mode="a")` — writes `{git, cwd, hostname, time}`.
  * `performance_dask_configuration(performance_file, rsexec, indent=2, mode="a")`.
  * `performance_qa_image(performance_file, key, im, indent=2, mode="a")` — invokes `im.image_acc.qa_image()`.
  * `performance_store_dict(performance_file, key, s, indent=2, mode="a")`.
  * `performance_read(performance_file)` / `performance_read_memory_data(memory_file)` / `performance_merge_memory(performance, mem)`.

### 8.8 `processing_components/xarray/operations.py`

Generic xarray↔FITS bridge (uses `wcs.to_header()`):

* **`export_xarray_to_fits(xa, fitsfile)`** — handles Datasets via the `data_model` attribute (`"GridData"`, `"ConvolutionFunction"`, default WCS otherwise).
* **`import_xarray_from_fits(fitsfile) -> xarray.Dataset`**.

### 8.9 `processing_components/flagging/operations.py`

* **`flagging_visibility(bvis, baselines=None, antennas=None, channels=None, polarisations=None) -> Visibility`** — set flags=1 for the given baselines/antennas/channels/pols.
* **`flagging_aoflagger(vis, strategy_name) -> Visibility`** — runs [AOFlagger](https://aoflagger.readthedocs.io/) with either a LUA strategy file or a built-in telescope strategy: `AARTFAAC`, `ARECIBO`/`ARECIBO 305M`, `BIGHORNS`, `EVLA`/`JVLA`, `LOFAR`, `MWA`, `PARKES`/`PKS`/`ATPKSMB`, `WSRT`. Defaults to `Generic`. Iterates over baselines (which is slow but memory-friendly). Updates `vis.flags.data` in place. Silently no-ops if AOFlagger is not installed.

---

## 9. The `rsexecute` execution layer

`workflows/rsexecute/execution_support/rsexecute.py` defines the singleton `rsexecute` (instance of `_rsexecutebase`) that hides the choice between immediate Python, Dask Delayed, and DALiuGE. The class methods are:

* **`rsexecute.execute(func, *args, **kwargs)`** — returns `dask.delayed(func, *args, **kwargs)`, `dlg_delayed(...)`, or the bare `func` depending on backend.
* **`rsexecute.set_client(client=None, use_dask=True, use_dlg=False, verbose=False, optim=True, **kwargs)`** — assigns a new client. With `use_dask=True` and no client, calls `get_dask_client(**kwargs)`. Sets `dask.config.set(scheduler="distributed")` so xarray uses the right scheduler.
* **`rsexecute.compute(value, sync=False)`** — `value.compute()` if no client, otherwise `client.compute(value, sync=sync)`. With `sync=True`, waits and returns the result; otherwise returns a future.
* **`rsexecute.persist(graph, **kwargs)`**, **`rsexecute.scatter(graph, **kwargs)`**, **`rsexecute.gather(graph)`** — pass-throughs for the dask client.
* **`rsexecute.run(func, *args, **kwargs)`** — `client.run` (per-worker); used to install per-worker logging.
* **`rsexecute.optimize(*args, **kwargs)`** — `dask.optimize` if enabled.
* **`rsexecute.close()`** — close client + cluster.
* **`rsexecute.init_statistics()` / `save_statistics(name="dask")`** — record `client.profile()` + `client.get_task_stream()` to HTML; aggregate task-time per function, write to log, return the summary dict for `performance_store_dict`.
* **`rsexecute.memusage(memusage_file="memusage.csv")`** — installs the [`dask-memusage`](https://github.com/itamarst/dask-memusage) plugin.
* Properties: `.client`, `.using_dask`, `.using_dlg`, `.optimizing`, `.type()` (`"dask"`/`"daliuge"`/`"function"`).

`get_dask_client(timeout=30, n_workers=None, threads_per_worker=None, processes=True, create_cluster=False, memory_limit=None, local_dir=".", with_file=False, scheduler_file="./scheduler.json", dashboard_address=":8787")` consults two env vars:

* **`RASCIL_DASK_SCHEDULER`** — address (e.g. `tcp://127.0.0.1:8786`); takes priority.
* **`RASCIL_DASK_SCHEDULER_FILE`** — JSON scheduler file; second-priority.

If neither is set and `create_cluster=True`, builds a `LocalCluster`; otherwise a default `Client` (which auto-creates one). Prints the diagnostic dashboard URL.

The `_rsexecutebase.__init__` does an Astropy warm-up (`erfa.s2c`, `astropy.constants.c.unit`, `SkyCoord(...).to_string()`, `SkyCoord(...).skyoffset_frame()`) to avoid threading races on first-use inside Dask workers — RASCIL learnt this the hard way.

---

## 10. Workflows: `rascil/workflows/rsexecute/`

Every workflow function is a **graph builder**: it returns Dask delayed objects that you compute later with `rsexecute.compute(...)`. They all preserve the same convention `_list_rsexecute_workflow` for accepting and returning **lists** of delayed objects (one per output).

### 10.1 `simulation/simulation_rsexecute.py`

`__all__` = 10. Highlights:

* **`simulate_list_rsexecute_workflow(config="LOWBD2", phasecentre=SkyCoord(15°, −60°), frequency=None, channel_bandwidth=None, times=None, polarisation_frame=stokesI, order="frequency", format="vis", rmax=1000.0, zerow=False, skip=1)`** — the main entry. `order ∈ {"time", "frequency", "both", None}` controls whether the list scatters across time, frequency, both, or stays as a single `Visibility`. `config` is either a `Configuration` instance or a string passed to `create_named_configuration(config, rmax=rmax)` (e.g. `"LOWBD2"`, `"LOWR3"`, `"MID"`, `"MIDR5"`, `"MEERKAT+"`, `"VLAA_north"`, etc.). `skip > 1` decimates antennas via `decimate_configuration`. `times` are radians (hour angle).
* **`corrupt_list_rsexecute_workflow(vis_list, gt_list=None, jones_type="T", **kwargs)`** — apply optional `gt_list` (or random gains via `simulate_gaintable`) to a list of visibilities.
* **`create_atmospheric_errors_gaintable_rsexecute_workflow(sub_bvis_list, sub_components, r0=5e3, screen=None, height=3e5, type_atmosphere="iono", reference_component=None, jones_type="B", **kwargs) -> (nominal_gt, actual_gt)`** — graph form of `create_gaintable_from_screen`.
* **`create_pointing_errors_gaintable_rsexecute_workflow(...)`** — graph form of `simulate_pointingtable*` and `simulate_gaintable_from_pointingtable`. For `time_series=""` uses pure random pointings; for `"wind"` etc. reads PSD files via `simulate_pointingtable_from_timeseries`. Returns 4-tuple `(nominal_gt, actual_gt, nominal_pt, actual_pt)`.
* **`create_surface_errors_gaintable_rsexecute_workflow(band, sub_bvis_list, sub_components, vp_directory, elevation_sampling=5.0)`** — for SKA Mid, loads pre-computed FEKO interpolated voltage patterns from `{B1,B2,Ku}_{el}_{freq_MHz}_{real,imag}_interpolated.fits` files, picks the actual elevation snapped to `elevation_sampling`, and the nominal at 45°.
* **`create_polarisation_gaintable_rsexecute_workflow(band, sub_bvis_list, sub_components, get_vp, normalise=True)`** — symmetrises the `vp[:, 0]` and `vp[:, 3]` cross-pol channels to form an "ideal" nominal, compares with the real `vp` for each `vp_type`. `get_vp(name)` is a callable.
* **`create_voltage_pattern_gaintable_rsexecute_workflow(...)`** — nominal-only variant.
* **`create_heterogeneous_gaintable_rsexecute_workflow(band, sub_bvis_list, sub_components, get_vp, default_vp="MID")`** — for mixed-dish arrays (SKA + MeerKAT).
* **`create_standard_mid_simulation_rsexecute_workflow(band, rmax, phasecentre, time_range, time_chunk, integration_time, polarisation_frame=None, zerow=False, configuration="MID")`** — pre-canned MID (or MeerKAT+) simulator. `band ∈ {"B1LOW", "B1", "B2", "Ku"}` selects the centre frequency (350 MHz, 765 MHz, 1.36 GHz, 12.179 GHz). `time_range` in **hours**, `time_chunk` in seconds. Returns one `Visibility` per chunk.
* **`create_standard_low_simulation_rsexecute_workflow(...)`** — LOW analogue. Uses `data/ska1low.cfg` via `create_configuration_from_MIDfile` (note the misleading name — the function works for either telescope), at `(116.764°, −26.825°, 300 m)`. Filters times above 45° elevation.

### 10.2 `imaging/imaging_rsexecute.py`

`__all__` = 17. The core grid/de-grid:

* **`predict_list_rsexecute_workflow(vis_list, model_imagelist, context, **kwargs)`** — graph of `predict_visibility(vis, model, context=context, **kwargs)`. `context ∈ {"2d", "ng", "wstack", "awprojection"}` (the latter requires a `gcfcf` partial built from `create_awterm_convolutionfunction`). Zeroes the visibility first (since `predict_visibility` accumulates).
* **`invert_list_rsexecute_workflow(vis_list, template_model_imagelist, context, dopsf=False, normalise=True, **kwargs)`** — graph of `invert_visibility(vis, template_model, dopsf, normalise, context, **kwargs)`. Returns list of `(image, sumwt)` tuples.
* **`residual_list_rsexecute_workflow(vis, model_imagelist, context="2d", **kwargs)`** — predict, subtract, invert.
* **`restore_list_singlefacet_rsexecute_workflow / restore_list_rsexecute_workflow / restore_centre_rsexecute_workflow`** — Restore deconvolved + residual using `clean_beam` (passed in or fitted via `fit_psf`).
* **`deconvolve_list_singlefacet_rsexecute_workflow / deconvolve_list_rsexecute_workflow / deconvolve_list_channel_rsexecute_workflow`** — Hogbom/MS/MS-MFS deconvolution via `ska_sdp_func_python.image.deconvolution.deconvolve_cube` or `deconvolve_list`. With `use_radler=True` calls `radler_deconvolve_list` (RADLER, the SKA next-gen deconvolver — requires the `radler` extras).
* **`scatter_facets_and_transpose(model_imagelist, facets, overlap, taper) / image_gather_facets`** — facet-based parallel deconvolution.
* **`griddata_merge_weights_rsexecute(gd_list)`** — sum gridded weight images.
* **`weight_list_rsexecute_workflow(vis_list, model_imagelist, weighting="uniform", robustness=0.0, **kwargs)`** — uniform/Briggs/natural weighting; calls `grid_visibility_weight_to_griddata`, `griddata_merge_weights`, `griddata_visibility_reweight`. **Weighting modes**: `"uniform"` (uses gridded weight), `"robust"` (Briggs `robustness` ∈ [−2, 2] — −2 is uniform-like, +2 natural-like), `"natural"` (no reweight).
* **`taper_list_rsexecute_workflow(vis_list, size_required)`** — Gaussian taper in image plane via `taper_visibility_gaussian(size_required)`.
* **`zero_list_rsexecute_workflow(vis_list, copy=True)`** / **`subtract_list_rsexecute_workflow(vis_list, model_vislist)`**.
* **`sum_predict_results_rsexecute(bvis_list, split=2)` / `sum_invert_results_rsexecute(image_list)`** — tree reduction.
* **`threshold_list_rsexecute(imagelist, prefix="", **kwargs)`** — clean-loop threshold computation.

### 10.3 `pipelines/pipeline_skymodel_rsexecute.py`

`__all__` = 3:

* **`continuum_imaging_skymodel_list_rsexecute_workflow(vis_list, model_imagelist, context, **kwargs)`** — wraps `ical_skymodel_list_rsexecute_workflow` with `do_selfcal=False`, `pipeline_name="cip"`. The CIP minor-cycle minor-major loop:
  1. Compute PSF graphs from `invert_list_rsexecute_workflow(..., dopsf=True)`.
  2. (Optional) AOFlagger via `flagging_aoflagger`, with `concatenate_visibility_frequency` first.
  3. Init `skymodel_list` if not provided.
  4. For each major cycle: predict from skymodel, subtract, invert residual, deconvolve, accumulate into skymodel.
  5. Restore via `restore_skymodel_list_rsexecute_workflow` or `restore_centre_skymodel_list_rsexecute_workflow`.
* **`ical_skymodel_list_rsexecute_workflow(vis_list, model_imagelist, context, skymodel_list=None, calibration_context="TG", controls=None, do_selfcal=True, pipeline_name="ical", **kwargs)`** — like CIP but with selfcal between cycles via `calibrate_list_rsexecute_workflow`. Optional `reset_skymodel` zeros the model image (but not the components) after the first selfcal. Optional `calibrate_with_dp3` invokes DP3 GainCal externally; the skymodel is exported to a DP3-compatible `.skymodel` file via `export_skymodel_to_text`.
* **`spectral_line_imaging_skymodel_list_rsexecute_workflow(...)`** — skymodel-based spectral-line variant.
* **`convert_skycomponents_taylor_terms_list(...)`** — converts a list of per-channel components to MS-MFS Taylor-term coefficients.

### 10.4 `calibration/calibration_rsexecute.py`

* **`calibrate_list_rsexecute_workflow(vis_list, model_vislist, gt_list=None, calibration_context="TG", controls=None, global_solution=True, **kwargs) -> (cal_vis_list, gt_list)`** — graph for selfcal. With `global_solution=True` and len(vis_list) > 1: divides each by its model (`divide_visibility`), concatenates across frequency (`concatenate_visibility, dim="frequency"`), integrates (`integrate_visibility_by_channel`), solves a single chain, applies it to every input vis. With `global_solution=False`: solves and applies per vis. `kwargs` includes `iteration` (passed to `solve_calibrate_chain`), `tol`, and `calibrate_with_dp3` to switch to DP3 (no RASCIL gaintables produced in that case).

### 10.5 `skymodel/skymodel_rsexecute.py`

`__all__` = 5. Wraps `ska_sdp_func_python.sky_model`'s `skymodel_predict_calibrate` and `skymodel_calibrate_invert` (which fold a SkyModel's `image`, `components`, `gaintable`, `mask` into a single predict/invert call) — used by ICAL.

* **`predict_skymodel_list_rsexecute_workflow(obsvis, skymodel_list, **kwargs)`** — `obsvis` may be a single vis (broadcast over skymodels) or a list (paired). Calls `skymodel_predict_calibrate`.
* **`invert_skymodel_list_rsexecute_workflow(vis_list, skymodel_list, **kwargs)`** — calls `skymodel_calibrate_invert`.
* **`restore_skymodel_list_rsexecute_workflow / restore_centre_skymodel_list_rsexecute_workflow / restore_skymodel_single_list_rsexecute_workflow`** — restores using a clean-beam.
* **`deconvolve_skymodel_list_rsexecute_workflow(residual_imagelist, psf_imagelist, skymodel_list, prefix="", fit_skymodel=False, **kwargs)`** — runs deconvolution on residual+psf and accumulates into the skymodels' components (via `find_skycomponents_frequency_taylor_terms` + `calculate_skycomponent_list_taylor_terms` + `gather_skycomponents_from_channels` for MS-MFS).
* **`residual_skymodel_list_rsexecute_workflow(...)`** — predict from skymodel, subtract, invert.

### 10.6 `image/image_rsexecute.py` (`__all__` = 3)

* **`image_rsexecute_map_workflow(im, imfunction, facets=1, overlap=0, taper=None, **kwargs)`** — `image_scatter_facets → imfunction(facet, **kwargs) → image_gather_facets`.
* **`sum_images_rsexecute(image_list, split=2)`** — tree reduction.
* **`image_gather_channels_rsexecute(image_list, split=0)`** — gather channels into a cube; with `split>0` does a binary tree.

### 10.7 `visibility/visibility_rsexecute.py` (`__all__` = 3)

* **`create_visibility_from_ms_rsexecute(msname, nchan_per_vis, nout, dds, average_channels=False)`** — graph for parallel MS reads. Each `(dd, chan_block)` pair becomes a delayed call to `create_visibility_from_ms(msname, selected_dds=[dd], start_chan, end_chan, average_channels)`. Returns a flat list of length `nout × len(dds)`.
* **`concatenate_visibility_frequency_rsexecute(bvis_list, split=2)`** — tree reduction over freq axis.
* **`concatenate_visibility_time_rsexecute(bvis_list, split=2)`** — tree reduction over time.

---

## 11. Command-line applications (`rascil/apps/`)

### 11.1 `apps_parser.py` — reusable argparse fragments

Six factories, all returning the parser unchanged after adding flags:

* `apps_parser_app(parser)` — `--mode {cip|ical|invert|load}`, `--logfile`, `--performance_file`.
* `apps_parser_ingest(parser)` — `--ingest_msname`, `--ingest_dd [int...]` (default `[0]`), `--ingest_vis_nchan`, `--ingest_chan_per_vis` (default `1`), `--ingest_average_vis {True|False}` (string-typed).
* `apps_parser_imaging(parser)` — full imaging knobs: `--imaging_phasecentre`, `--imaging_pol stokesI`, `--imaging_nchan 1`, `--imaging_context ng`, `--imaging_ng_threads 4`, `--imaging_w_stacking True`, `--imaging_flat_sky False`, `--imaging_npixel`, `--imaging_cellsize`, `--imaging_weighting uniform`, `--imaging_robustness 0.0`, `--imaging_gaussian_taper`, `--imaging_dopsf False`, `--imaging_dft_kernel {cpu_looped|gpu_raw}`, `--imaging_uvmax/min`, `--imaging_rmax/min`, `--perform_flagging False`, `--flagging_strategy_name generic`.
* `apps_parser_cleaning(parser)` — `--clean_algorithm {hogbom|msclean|mmclean}` (default `mmclean`), `--clean_use_radler False`, `--clean_beam BMAJ BMIN BPA`, `--clean_scales [int...]` (default `[0]`), `--clean_nmoment 4`, `--clean_nmajor 5`, `--clean_niter 1000`, `--clean_psf_support 256`, `--clean_gain 0.1`, `--clean_threshold 1e-4`, `--clean_component_threshold`, `--clean_component_method {fit|extract}`, `--clean_fractional_threshold 0.3`, `--clean_facets 1`, `--clean_overlap 32`, `--clean_taper {none|linear|tukey}` (default tukey), `--clean_restore_*` siblings, `--clean_restored_output {list|taylor|integrated}` (default list).
* `apps_parser_calibration(parser)` — per-context selfcal scheduling: `--calibration_T_first_selfcal 1`, `--calibration_T_phase_only True`, `--calibration_T_timeslice`, same for `G` (`first_selfcal=3`, phase-only False) and `B` (`first_selfcal=4`, timeslice `1e5` if not given), `--calibration_context "T"` (chain to solve), `--calibration_global_solution True`, `--calibration_reset_skymodel True`, `--use_initial_skymodel False`, `--input_skycomponent_file`, `--num_bright_sources`, `--calibrate_with_dp3 False`, `--input_dp3_skymodel`.
* `apps_parser_dask(parser)` — `--use_dask True`, `--dask_nthreads`, `--dask_memory`, `--dask_memory_usage_file`, `--dask-nodes` (SSHCluster), `--dask_nworkers`, `--dask_scheduler`, `--dask_scheduler_file`, `--dask_tcp_timeout`, `--dask_connect_timeout`, `--dask_malloc_trim_threshold` (default `0` = aggressive). Also a separate `apps_parser_slurm(parser)` for `--use_slurm`, `--slurm_project SKA-SDP`, `--slurm_queue compute`, `--slurm_walltime 01:00:00`.

### 11.2 `rascil_imager.py` — the canonical end-to-end imager

`main()`:
1. Parse with `apps_parser_app + ingest + imaging + calibration + cleaning + dask`.
2. Initialise `performance_file` JSON via `performance_environment` and `performance_store_dict("cli_args", vars(args))`.
3. Call `imager(args)`:
   * `setup_rsexecute(args)` — `rsexecute.set_client(use_dask, ...)`. Supports `--dask_scheduler ssh` (builds an `SSHCluster` from `--dask-nodes`), `--dask_scheduler tcp://...`, `--dask_scheduler_file ...`, `existing`, or default LocalCluster via `get_dask_client(n_workers, threads_per_worker, memory_limit)`. Configures `distributed.comm.timeouts.tcp/connect` from CLI args, sets `MALLOC_TRIM_THRESHOLD_` for the nanny.
   * `get_vis_list(args)` — build a list of `Visibility` graph nodes via `create_visibility_from_ms_rsexecute`, splitting each data descriptor into `nout = ingest_vis_nchan / ingest_chan_per_vis` chunks.
   * `select_vis(args, bvis_list)` — apply `select_r_range(rmin, rmax)` and flag outside `select_uv_range(uvmin, uvmax)`.
   * `perf_save_vis_info(args, bvis_list)` — store per-vis performance info (`vis0`, `vis1`, ...).
   * `get_cellsize(args, bvis_list)` — compute via `advise_wide_field(guard_band_image=3.0)` if not given.
   * `convert_to_stokesI(args, bvis_list)` — `convert_visibility_to_stokesI` for `imaging_pol == "stokesI"`.
   * `create_model_image_list` — `create_image_from_visibility(npixel, nchan=imaging_nchan, cellsize, polarisation_frame=PolarisationFrame(imaging_pol))`.
   * `weight_vis(args, bvis_list, model_list)` — `weight_list_rsexecute_workflow` (uniform/robust/natural) → `taper_list_rsexecute_workflow`.
   * `get_clean_beam(args)` — dict from `--clean_beam BMAJ BMIN BPA` or `None`.
   * Dispatch on `args.mode`:
     * **`cip`** — `continuum_imaging_skymodel_list_rsexecute_workflow(...)`.
     * **`ical`** — `ical_skymodel_list_rsexecute_workflow(...)` with selfcal controls (`controls["T/G/B"]`).
     * **`invert`** — `invert_list_rsexecute_workflow + sum_invert_results_rsexecute`.
     * **`load`** — `rsexecute.compute(bvis_list, sync=True)`; logs each vis.
   * Save Dask profile (`save_statistics`), close client.
4. `write_results(restored_output, imagename, result, performance_file, gt_list=None)` — writes to FITS (suffixes: `_deconvolved.fits`, `_residual.fits`, `_restored.fits`, `_restored_centre.fits`, `_skymodel.hdf`, `.taylor.{n}.{deconvolved|residual|restored}.fits`), and per-context gaintables `_gaintable_T.hdf`, `_gaintable_G.hdf`, etc. Each image gets `image_acc.qa_image()` written into the performance JSON.

### 11.3 `rascil_advise.py`

Trivial wrapper around `advise_wide_field(bvis_list[0], guard_band_image, oversampling_synthesised_beam, delA, verbose=True)`. CLI flags: `--ingest_msname`, `--ingest_dd`, `--logfile`, `--guard_band_image 3.0`, `--oversampling_synthesised_beam 3`, `--dela 0.02`. Returns the advice dict.

### 11.4 `rascil_rcal.py` — real-time calibration simulator

CLI flags include `--ingest_msname`, `--cal_type {T|G}` (default T = atmospheric phase), `--phase_only_solution True`, `--solution_tolerance 1e-12`, `--use_previous_gaintable False`, `--ingest_components_file`, `--apply_beam False`, `--ingest_beam_file`, `--flag_rfi False`, `--initial_threshold 8.0`, `--rho 1.5`, `--do_plotting False`, `--plot_dir`.

Pipeline:
1. `_rfi_flagger(bvis, initial_threshold, rho)` — calls `ska_sdp_func.rfi.sum_threshold_rfi_flagger` with thresholds `initial_threshold / rho^log2(seq)` for `seq = [1,2,4,8,16,32]` (per Offringa et al. 2010).
2. Optional model components: read HDF or TXT, optionally `apply_beam_correction`.
3. `bvis_source` generator splits `bvis` along `time`, yields one-sample subslices.
4. `bvis_solver` calls `realtime_single_bvis_solver(bvis, model_components, previous_solution, phase_only, jones_type, tol, use_previous)` per sample. Internally: `dft_skycomponent_visibility(model, components)` then `solve_gaintable(bvis, modelvis, gain_table=previous_solution, phase_only, jones_type, tol)`.
5. `gt_sink(gt_gen, do_plotting, plot_name)` — concatenates with `concatenate_gaintables`, optionally plots `(amp-1, phase-phase[ant0], weight, residual)` over time via `gt_single_plot`.
6. `export_gaintable_to_hdf5` writes `<msname>_<datetime>_gaintable.hdf`.

### 11.5 `rascil_sensitivity.py` — MID/MeerKAT sensitivity calculator

Computes point-source and surface-brightness sensitivities using both Thompson-Moran-Swenson (TMS Eq. 6.62) and CASA approaches. CLI flags include `--imaging_npixel 1024`, `--imaging_cellsize`, `--imaging_oversampling 3.0`, `--imaging_weighting`, `--imaging_robustness [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]`, `--imaging_taper`, `--ra +15`, `--declination -45`, `--tsys 20`, `--efficiency 1.0`, `--diameter 15`, `--configuration MIDR5`, `--subarray <json>`, `--rmax 2e5`, `--frequency 1.36e9`, `--integration_time 600`, `--time_range -4 4` (hours), `--nchan 1`, `--channel_width 1e8`, `--results rascil_sensitivity`.

Outputs a CSV of one row per `(weighting, robustness, taper)` scenario with columns: `weighting`, `robustness`, `taper`, `cleanbeam_bmaj/bmin/bpa`, `sum_weights`, `psf_*`, `pss`, `pss_casa`, `reltonat_casa`, `sa` (clean-beam solid angle, sr), `sbs`, `sbs_casa`, `tb` (time-bandwidth product). Uses `ska_sdp_func_python.image.fit_psf` to extract the clean beam from the gridded PSF.

The TMS formula:
```
pss = √(Σ gridwt² / N) / (Σ gridwt / N) · √2 · 10²⁶ · k_B · T_sys / (A · η · √(Σ natwt))
```
The CASA point-source-sensitivity is `√(Σ gridwt²/natwt) / Σ gridwt`.

Optional MS export (`--msfile name.ms`) goes through `concatenate_visibility_frequency` + `export_visibility_to_ms`.

### 11.6 `rascil_vis_ms.py`

Bare-bones MS visualiser. Loads MS, plots `plot_configuration`, `plot_uvcoverage`, `plot_visibility(y="amp")`, `plot_visibility(y="phase")` at the central channel; also calls `display_ms_as_image` from `apps/common.py`.

### 11.7 `imaging_qa_main.py` (also runnable as `imaging_qa`)

Wraps **PyBDSF** (`bdsf.process_image`) for source-finding QA on a restored FITS image. Flags include `--ingest_fitsname_restored`, `--ingest_fitsname_residual`, `--ingest_fitsname_sensitivity`, `--ingest_fitsname_moment`, `--finder_beam_maj/min/pos_angle`, `--finder_thresh_isl 5.0`, `--finder_thresh_pix 10.0`, `--finder_multichan_option {single|average}`, `--apply_primary False`, `--use_frequency_moment False`, `--telescope_model MID`, `--check_source False`, `--plot_source False`, `--input_source_filename`, `--match_sep 1e-5`, `--flux_limit 1e-3`, `--trim_image False`, `--trim_box 3e-2`, `--quiet_bdsf False`, `--source_file`, `--rascil_source_file`, `--logfile`, `--savefits_rmsim False`, `--restart False`, plus dask flags.

Pipeline:
1. `imaging_qa_bdsf(...)` runs PyBDSF, generates Gaussian + island catalogues.
2. `create_source_to_skycomponent` converts found sources to RASCIL `SkyComponent`s.
3. `correct_primary_beam(input_image, sensitivity_image, comp, telescope)` divides by primary beam.
4. `calculate_spec_index_from_moment` extracts MS-MFS Taylor terms.
5. `check_source(orig, comp, match_sep)` cross-matches with the input catalogue using `find_skycomponent_matches`.
6. `plot_errors(...)` calls all 8 `plot_skycomponents_*` functions.
7. `imaging_qa_diagnostics` (in `apps/imaging_qa/`) generates a static HTML report; `create_index` (in `apps/imaging_qa/generate_results_index.py`) builds the index.

### 11.8 `rascil_image_check.py`

Single-image QA gate: `--image`, `--stat {max|min|maxabs|rms|sum|medianabs|medianabsdevmedian|shape}`, `--min`, `--max`. Returns `0` if `qa.data[stat]` is in `[min, max]`, else `1`. Designed for shell pipelines (`set -e`).

### 11.9 `performance_analysis.py`

Reads the JSON profile produced during a `rascil_imager` run and produces:
* Bar charts per-function statistics.
* Line plots of (e.g. `invert_ng` time) vs a parameter (`imaging_npixel`, `vis_nvis`).
* Contour plots (e.g. `invert_ng` vs `(imaging_npixel, vis_nvis)`).

---

## 12. Sample data tree (`data/`)

After `get_rascil_data` (or git-LFS pull):

| Path | Purpose |
|------|---------|
| `ska1low.cfg` | SKA1 LOW antenna positions |
| `models/M31.MOD`, `M31.model.fits`, `M31_canonical.model.fits` | M31 Hα test image used by `create_test_image` |
| `models/GLEAM_EGC.fits`, `GLEAM_filtered.txt`, `Gleam_ReadMe.txt` | GLEAM Extragalactic Catalog (Hurley-Walker+ 2017) |
| `models/S3_151MHz_{10,20,40}deg.csv`, `S3_1400MHz_{1mJy_10deg, 1mJy_18deg, 100uJy_10deg, 100uJy_18deg, 10uJy_10deg}.csv` | S3-SEX simulation catalogues |
| `models/MID_FEKO_VP_{B1_45_0365,B1_45_0765,B2_45_1360,Ku_45_12179}_{real,imag}.fits` | FEKO-modelled SKA Mid voltage patterns at elevation 45° |
| `models/MeerKAT_VP_60_1360_{real,imag}.fits` | MeerKAT VP at 1.36 GHz, el 60° |
| `models/SKA1_LOW_beam.fits` | LOW primary beam |
| `models/{precision,standard,degraded}/El{15,45,90}Az{0,45,90,135,180}.dat` | Wind/tracking PSDs for `simulate_pointingtable_from_timeseries` |
| `models/test_mpc_screen.fits` | Sample atmospheric phase screen |
| `models/spatial/`, `models/standard/`, `models/degraded/` | More PSD subdirectories |
| `models/VLA_A_hor_xyz.txt` | VLA A-config antenna positions |
| `vis/` | Sample MS / HDF5 visibility files for tests |
| `misc/` | Miscellaneous configurations |

The active data root is `$RASCIL_DATA` if set, else `$RASCIL/data` if `$RASCIL` is set, else the directory two levels up from `parameters.py`.

---

## 13. Examples and notebooks

`examples/notebooks/` contains 14 notebooks that double as integration tests:

| Notebook | Topic |
|---|---|
| `imaging.ipynb` | Basic invert/predict |
| `imaging-pipelines_rsexecute.ipynb` | CIP/ICAL via rsexecute |
| `imaging-fits_rsexecute.ipynb` | FITS I/O |
| `bandpass-calibration.ipynb` | B-Jones solving |
| `deconvolution.ipynb` | Hogbom/MS clean |
| `multi_frequency_deconvolution.ipynb` | MS-MFS / `mmclean` |
| `flag-visibilities.ipynb` | AOFlagger usage |
| `gridding.ipynb` | Convolution functions, AW projection |
| `mpc_simulation.ipynb` | Multi-pole calibration / atmospheric screens |
| `simple-dask_rsexecute.ipynb` | Dask basics with rsexecute |
| `queue.ipynb` | Job queueing |
| `demo_image_xarray.ipynb`, `demo_visibility_xarray.ipynb` | xarray accessor demos |
| `PSWF_Calculation.ipynb` | Prolate spheroidal kernel walkthrough |

`examples/scripts/imaging.py` and `examples/scripts/primary_beam_zernikes.py` are the script forms.

`examples/pipelines/` contains higher-level scripted pipelines (Ms-from-scratch SKA1-LOW/MID, MPC simulations).

`examples/cluster_tests/` contains scripts intended for HPC validation against an SLURM/SSHCluster setup.

`examples/performance/` contains profiling scripts that drive `performance_analysis.py`.

---

## 14. Distributed-execution recipe

A canonical RASCIL distributed simulate→image cycle:

```python
import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame

from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute, get_dask_client
from rascil.workflows import (
    simulate_list_rsexecute_workflow,
    corrupt_list_rsexecute_workflow,
    weight_list_rsexecute_workflow,
    invert_list_rsexecute_workflow,
    sum_invert_results_rsexecute,
    continuum_imaging_skymodel_list_rsexecute_workflow,
)
from rascil.processing_components.simulation import (
    create_low_test_skymodel_from_gleam,
)
from ska_sdp_func_python.imaging import create_image_from_visibility
from ska_sdp_func_python.imaging.predict import predict_visibility

# Distribute
client = get_dask_client(n_workers=8, memory_limit="32GB")
rsexecute.set_client(use_dask=True, client=client)
rsexecute.init_statistics()

# Simulate
phasecentre = SkyCoord(ra=15*u.deg, dec=-45*u.deg, frame="icrs", equinox="J2000")
freqs = numpy.linspace(1.0e8, 2.0e8, 11)
times = numpy.linspace(-3, 3, 41) * (numpy.pi/12.0)

bvis_graph = simulate_list_rsexecute_workflow(
    config="LOWBD2", phasecentre=phasecentre,
    frequency=freqs, channel_bandwidth=numpy.full(11, 1e7),
    times=times, polarisation_frame=PolarisationFrame("stokesI"),
    order="frequency", rmax=1500.0,
)

# Predict from a model skymodel
sm = create_low_test_skymodel_from_gleam(
    npixel=512, cellsize=4e-5, frequency=freqs,
    channel_bandwidth=numpy.full(11, 1e7), phasecentre=phasecentre,
    flux_limit=0.5, flux_threshold=2.0,
)
def add_components(v, components):
    return predict_visibility(v, sm.image, context="ng")
bvis_graph = [rsexecute.execute(add_components)(v, sm.components) for v in bvis_graph]

# Corrupt with phase noise
bvis_graph = corrupt_list_rsexecute_workflow(bvis_graph, phase_error=0.5)

# Image
model_graph = [rsexecute.execute(create_image_from_visibility)(v, npixel=1024, cellsize=4e-5)
               for v in bvis_graph]
bvis_graph = weight_list_rsexecute_workflow(bvis_graph, model_graph,
                                             weighting="robust", robustness=0.0)
result = continuum_imaging_skymodel_list_rsexecute_workflow(
    bvis_graph, model_graph, context="ng", niter=1000, nmajor=5,
    algorithm="mmclean", nmoment=3, threshold=1e-4,
)
residual, restored, skymodel = rsexecute.compute(result, sync=True)
rsexecute.save_statistics("my_run")
rsexecute.close()
```

Outside Dask, set `rsexecute.set_client(use_dask=False)` and the same code runs serially with no graph construction overhead.

---

## 15. Conventions and gotchas

* **`Visibility` is the only vis container** — RASCIL has fully retired the historical `BlockVisibility` split (CHANGELOG 1.0.0). Keep this in mind when porting older docs/examples. Function signatures still use names like `bvis` and `bvis_list` for historical reasons.
* **Times are radians** for synthetic data (hour-angle), but **seconds (UTC)** when read from MS/UVFITS via `ska-sdp-datamodels` readers. The `times_are_ha=True` kwarg in `ingest_unittest_visibility` controls which form `create_visibility` consumes.
* **Cellsize is radians** everywhere in the API. Convert with `cellsize = arcsec / 206265`.
* **Frequency is Hz** everywhere.
* **Stokes-I-only assumptions** — `fit_visibility` asserts `polarisation_frame.type == "stokesI"`. Many simulation helpers (`addnoise_visibility`, `create_low_test_skymodel_from_gleam`) explicitly construct stokesI components.
* **AOFlagger is iterated per baseline** — `flagging_aoflagger` does a Python loop over baselines, which is slow for SKA-LOW-scale arrays. The vendored Python aoflagger SDK is the bottleneck.
* **Default RNG seeds** are deterministic in `simulate_gaintable` (180550721 / 1805550721) and `addnoise_visibility` (1805550721) — pass `seed=None` to override (which still produces a fixed default), or pass an integer.
* **`vis.configuration.diameter[0]`** is used as if homogeneous in `addnoise_visibility` — for heterogeneous arrays you must recompute noise per baseline.
* **CLI string-typed booleans** — many flags accept `"True"`/`"False"` strings rather than argparse `store_true`. This is intentional so configs can be loaded from `@file.cfg` files (note `fromfile_prefix_chars="@"` in the parsers).
* **Dask scheduler env vars** — `RASCIL_DASK_SCHEDULER` (URL) and `RASCIL_DASK_SCHEDULER_FILE` (file path) override `get_dask_client` defaults. CLI `--dask_scheduler` and `--dask_scheduler_file` exist in the apps.
* **Pycasacore measures data is required** for any topocentric/UTC computations. `get_rascil_data` rsyncs it and writes `~/.casarc`.
* **Apple Silicon** — `python-casacore` and the optional `aoflagger`/`dp3`/`radler`/`ska-sdp-func` packages are upstream-broken on macOS; expect to work with the `extras=["astron", "ska-sdp-func"]` only on Linux. Plain RASCIL imaging works on macOS with the Nifty-Gridder context.
* **End-of-maintenance** — The README explicitly states RASCIL is no longer maintained beyond 2.0.0. Bug reports and MRs are not accepted upstream. New SKA-SDP work continues in `ska-sdp-datamodels` and `ska-sdp-func-python` (which vendor most RASCIL primitives) and the `ska-sdp-pipelines` integrations.

---

## 16. Mapping to RadioSim concepts

For cross-referencing while integrating RASCIL into RadioSim (`/Users/RRI-interferometry/RadioSim/src/radiosim/`):

| RadioSim concept | RASCIL equivalent | Notes |
|---|---|---|
| `core/visibility.py` (point-source RIME) | `predict_visibility(vis, model, context="2d"\|"ng")` + `dft_skycomponent_visibility` | RASCIL's gridder paths use ng (Nifty-Gridder); RadioSim's RIME is direct sum. |
| `core/visibility_healpix.py` | RASCIL has no first-class HEALPix predictor — must convert via `pyradiosky → SkyComponent` or grid the HEALPix into a FITS image first. |
| `core/sky/SkyModel` | `ska_sdp_datamodels.sky_model.SkyModel(image, components, mask, gaintable)` | Different shape: RASCIL bundles components+image+gaintable+mask; RadioSim splits component vs healpix. |
| `core/sky/loaders/_loaders_vizier.py (gleam, mals, ...)` | `create_low_test_skycomponents_from_gleam` | RASCIL hard-codes GLEAM only; lookup is from a local FITS file, not Vizier. |
| `core/sky/_loaders_diffuse.py` | None — RASCIL has no GSM/PySM3 path. |
| `core/jones/*` (8-term Jones chain) | `solve_calibrate_chain(calibration_context="TGB...")` + `simulate_gaintable_from_*` | RASCIL only models T/G/B/D explicitly; F/W/Z are folded into screens or predict-side. |
| `core/jones/beam/*` | `apply_beam_to_skycomponent` + `create_pb` (telescope-specific) | RASCIL's `BeamFITSHandler` analogue is `BeamManager` outside this package. |
| `simulator/rime.py` | `simulate_list_rsexecute_workflow` + `predict_list_rsexecute_workflow` | Pure direct-sum RIME requires `context="2d"` and PSWF antialiasing. |
| `backends/jax_backend.py` etc. | `--imaging_dft_kernel gpu_raw` (single flag) | RASCIL's GPU support is only in `ska-sdp-func` extras; default is NumPy + Nifty-Gridder C++. |
| `io/measurement_set.py` | `create_visibility_from_ms` / `export_visibility_to_ms` (from `ska-sdp-datamodels`) | Same CASA Table format. |
| `utils/diagnostics/*` | `simulation_helpers.plot_*` + `apps/imaging_qa/*` | RASCIL has more polished QA than RadioSim but no strip-plotter. |
| `Simulator.from_config()` | `rascil_imager --config @config.txt` | RASCIL's CLI uses argparse `@` files, not YAML. |

The sweet spot for RadioSim is to **use RASCIL only for**:
1. AOFlagger and DP3 wrappers (`flagging_aoflagger`, `dp3_gaincal`).
2. Sensitivity calculations (`rascil_sensitivity` in the apps).
3. Dish-surface and pointing-error simulators (`simulate_gaintable_from_zernikes`, `simulate_gaintable_from_pointingtable`, `simulate_pointingtable_from_timeseries`).
4. Atmospheric phase screens (`create_gaintable_from_screen`).
5. PyBDSF QA pipeline (`imaging_qa_main`).
6. Reading FEKO MID/MeerKAT voltage patterns from `data/models/MID_FEKO_VP_*.fits`.

Sky-model loading, RIME execution, and Jones-chain composition are better served by RadioSim's own primitives, which are precision-config-aware (RASCIL implicitly uses float64 throughout) and HEALPix-first.
