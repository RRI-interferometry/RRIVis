# PyFHD — Python Fast Holographic Deconvolution

> Exhaustive technical reference for the `PyFHD` git submodule located at
> `/Users/RRI-interferometry/RadioSim/simulators/PyFHD/`.
> All citations use repository-relative paths anchored at `simulators/PyFHD/...`.
> The text below is derived solely from in-tree sources (no other `.md` files in
> `simulators/` were consulted).

---

## Table of Contents

1. [Overview & Purpose](#1-overview--purpose)
2. [Relationship to FHD (IDL)](#2-relationship-to-fhd-idl)
3. [License, Authors, Citation](#3-license-authors-citation)
4. [Languages, Versioning, Tags](#4-languages-versioning-tags)
5. [Repository Layout](#5-repository-layout)
6. [Installation & Dependencies](#6-installation--dependencies)
7. [Build System & Packaging](#7-build-system--packaging)
8. [Runtime Architecture & Pipeline](#8-runtime-architecture--pipeline)
9. [CLI / Public API](#9-cli--public-api)
10. [Configuration (YAML)](#10-configuration-yaml)
11. [Module-by-Module Breakdown](#11-module-by-module-breakdown)
12. [Core Algorithms](#12-core-algorithms)
13. [Input & Output Formats](#13-input--output-formats)
14. [Resources Bundled with the Package](#14-resources-bundled-with-the-package)
15. [Testing Layout](#15-testing-layout)
16. [Documentation Site](#16-documentation-site)
17. [CI, Docker & Release Workflows](#17-ci-docker--release-workflows)
18. [Notable Internals & Idioms](#18-notable-internals--idioms)
19. [Known Limitations / TODOs](#19-known-limitations--todos)
20. [Quick Reference Tables](#20-quick-reference-tables)

---

## 1. Overview & Purpose

**PyFHD** (**Py**thon **F**ast **H**olographic **D**econvolution) is a Python
translation of the IDL package **FHD**, an open-source imaging algorithm for
radio interferometers. The README states three primary use-cases (citation
`simulators/PyFHD/README.md` lines 23–26):

> "FHD is an open-source imaging algorithm for radio interferometers,
> specifically tested on MWA Phase I, MWA Phase II, PAPER, and HERA. There are
> three main use-cases for FHD: efficient image deconvolution for general radio
> astronomy, fast-mode Epoch of Reionization analysis, and simulation."

PyFHD is the Python port focused (currently) on the **EoR analysis pipeline**
and **calibration / gridding / HEALPix snapshot export** path. The project is
a collaboration between **Astronomy Data and Computing Services (ADACS)** and
the **EoR Team** (`README.md`, lines 96–101).

The package is registered under PyPI name **`pyfhd`**, distribution name
**`PyFHD`**, project description in `pyproject.toml` (line 6):

> "Python Fast Holograhic Deconvolution: A Python package that does fast-mode
> Epoch of Reionization analysis."

## 2. Relationship to FHD (IDL)

`README.md` (line 26) is explicit about the lineage:

> "PyFHD is the translated library of FHD from IDL to Python, it aims to get
> close to the same results as the original FHD project. Do expect some minor
> differences compared to the original FHD project due to the many differences
> between IDL and Python. These differences are often due to the difference in
> precision between IDL and Python with IDL being single-precision (accurate
> upto 1e-8) and Python being double-precision (1e-16). Some of the IDL
> functions are double-precision but most default to single-precision."

The IDL legacy is visible throughout the codebase:

| Hint | Where | Example |
|------|-------|---------|
| `idl_argunique`, `idl_median` | `PyFHD/pyfhd_tools/pyfhd_utils.py` (lines 888, 1366) | Reproduces IDL `UNIQ` / `MEDIAN` semantics |
| Reverse-index histogram | `pyfhd_utils.histogram` line 224 | Mirrors IDL's `HISTOGRAM(reverse_indices=)` |
| `.sav` (IDL Save) ingest | `PyFHD/io/pyfhd_io.py:convert_sav_to_dict` (line 652) | Used to migrate FHD outputs |
| `recarray_to_dict` | `pyfhd_io.py` line 554 | Unpacks IDL struct objects loaded by `scipy.io.readsav` |
| Comment about IDL precision | `PyFHD/calibration/calibrate.py` lines 276–278 | "LA_LEAST_SQUARES does not use double precision by default…" |
| Column- vs row-major swaps | `gridding_utils.interpolate_kernel` line 48 | "x_offset and y_offset needed to be swapped around as IDL is column-major, while Python is row-major" |

PyFHD interoperates with FHD outputs:
* **Beam files** can be loaded from FHD `.sav` files (`PyFHD/beam_setup/beam.py`
  lines 277–308).
* **Model visibilities** can be transferred from FHD-style sav files via
  `PyFHD/source_modeling/vis_model_transfer.py:import_vis_model_from_sav`
  (line 83). UVFITS and PyFHD-native `.h5` are also supported.

## 3. License, Authors, Citation

**License**: MIT (`simulators/PyFHD/LICENSE` lines 1–3):

> "MIT License — Copyright (c) 2022 Astronomy Data and Computing Services"

**Authors** (from `pyproject.toml` lines 7–11):

| Name | Email |
|------|-------|
| Joel Dunstan (SkyWa7ch3r) | joel.g.w.dunstan@gmail.com |
| Jack Line (JLBLine) | jack.line@curtin.edu.au |
| Nichole Barry (nicholebarry) | nichole.barry@unsw.edu.au |

**CITATION.cff** (`simulators/PyFHD/CITATION.cff`) lists the Zenodo DOI
record. Cite via DOI **`10.5281/zenodo.15720184`** (README line 87). A test
data Zenodo DOI is also referenced: **`10.5281/zenodo.15687722`**
(`README.md` line 17).

`README.md` line 91 notes: "TODO: A JOSS Paper is being done and will be
submitted soon".

Acknowledgements (`README.md` lines 94–101) credit Ian Sullivan (UWashington
team) for FHD itself and Bryna Hazelton, Paul Hancock for advice; previous
maintainer Jack Line is now acknowledged. The CITATION file lists ORCIDs for
all three current authors.

## 4. Languages, Versioning, Tags

**Pure Python**, no compiled extensions in-tree. Acceleration is via:

* `numba` (`@njit` JIT) — see `PyFHD/pyfhd_tools/pyfhd_utils.py:get_bins`
  (line 15), `get_hist` (53), `get_ri` (113); also imported in
  `pyfhd_utils.py` line 10 (`from numba import njit`).
* `scipy` (FFTs, interpolation, `readsav`).
* `numpy` ≥ 2.2.5 (CPU only — no GPU, no Dask, no MPI).

There is no GPU / CUDA path in the current tree. The runtime is single-host
CPU; gridding loops over polarizations are explicitly noted as
"multi-processing if it's not fast enough" (`PyFHD/pyfhd.py` line 522, but
not yet implemented).

| Item | Value | Source |
|------|-------|--------|
| Project version | **1.0.2** | `pyproject.toml` line 3, `CITATION.cff` line 14 |
| Python required | `>=3.11` | `pyproject.toml` line 12 |
| Python in `.python-version` | (5 chars; matches dev pin) | `simulators/PyFHD/.python-version` |
| Dev status | "4 - Beta" | `pyproject.toml` line 14 |
| Git tags | `1.0.1`, `1.0.2` | `git tag` |

The `pyproject.toml` classifiers list Python 3.10–3.13 (lines 17–21) but the
`requires-python` is `>=3.11`. The README badge claims 3.10–3.13.

**Recent commits** (`git log --oneline -25`):

| Hash | Subject |
|------|---------|
| 9eca0f1 | Merge pull request #72 from EoRImaging/uvbeam_overhaul |
| 95921de | instrument location error |
| adebfe2 | update the sample yaml with `uvbeam-freq-buffer` |
| e934cd0 | update the changelog |
| 5f3f5a8 | Document the new beam options and setup |
| 8f1efab | fix n_tile calculation that I broke a few commits ago |
| 59bf1c5 | prevent deprecation warnings when writing FITS files |
| 769b02f | rename `beam-file-path` to `saved-beam-file-path` |
| 4812473 | handle analytic beams in config.h5 file |
| 8b2406c | more fixes for OVRO-LWA data |
| e879c96 | fix analytic beam handling |
| 7d8b573 | fixes to get OVRO-LWA working |
| a15b042 | use the telescope location from the uvfits antenna table |
| 9b64546 | Add support for passing UVBeam files |
| 79761f3 | Fix bug where points below the horizon could be passed to UVBeam |
| f662a23 | fix computation of beam squared area |
| 32ac34c | fix fft direction in beam_image function |
| 6269b49 | fix a bug in computing beam phase that caused NaNs in beams |
| 3be2190 | fix a major bug that broke the gridding kernel |
| 2cdef0f | only read in the beam for the relevant freq range |
| 0e83579 | overhaul uvbeam handling |
| f0a390b | beam hot fixes |
| 4c11ade | Merge pull request #71 from EoRImaging/various_fixes |
| 0266cb6 | update the changelog |
| 796bc33 | improved comment per review comments |

The `__git__.py` build hook (`pyproject.toml` lines 76–87) bakes commit hash,
date, and branch into the wheel for `pyfhd -v` to display.

## 5. Repository Layout

```
simulators/PyFHD/
├── .dockerignore
├── .git                         (git-link to submodule)
├── .github/workflows/
│   ├── black.yml
│   ├── dockerhub.yaml
│   ├── publish.yml
│   ├── test.yml
│   └── zenodo.yml
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── .readthedocs.yaml
├── CITATION.cff
├── codemeta.json
├── Dockerfile
├── LICENSE                       (MIT)
├── README.md
├── docs/                         Sphinx site
│   ├── make.bat / Makefile
│   ├── requirements.txt
│   └── source/
│       ├── index.rst
│       ├── _static/...           pyfhd_coverage_report/, sample HTML
│       ├── changelog/changelog.md
│       ├── develop/{contribution_guide.md, idl_translation.md}
│       ├── documentation/documentation.rst
│       ├── installation/installation.md
│       ├── reports/*.rst
│       └── tutorial/tutorial.rst (+ many PNGs)
├── environment.yml               conda spec
├── input/                        runtime input dir (created by user)
├── output/                       runtime output dir
├── pyproject.toml                hatchling build, deps, scripts
├── PyFHD/                        ← Python source package
│   ├── __init__.py               (empty)
│   ├── pyfhd.py                  CLI entry point (main pipeline)
│   ├── beam_setup/
│   │   ├── antenna.py            init_beam, general_jones_matrix
│   │   ├── beam.py               create_psf
│   │   └── beam_utils.py         gaussian_decomp, beam_image, beam_power, …
│   ├── calibration/
│   │   ├── calibrate.py          calibrate, calibrate_qu_mixing
│   │   ├── calibration_utils.py  bandpass / polyfit / auto-fit / flag / apply
│   │   └── vis_calibrate_subroutine.py   linear least-squares solver
│   ├── data_setup/
│   │   ├── obs.py                create_obs, read_metafits, update_obs
│   │   └── uvfits.py             extract_header, create_params, extract_visibilities, create_layout
│   ├── flagging/
│   │   └── flagging.py           vis_flag, vis_flag_basic, vis_flag_tiles
│   ├── gridding/
│   │   ├── filters.py            filter_uv_uniform
│   │   ├── gridding_utils.py     baseline_grid_locations, interpolate_kernel, dirty_image_generate, visibility_count, holo_mapfn_convert, …
│   │   ├── visibility_grid.py    visibility_grid (the gridder)
│   │   └── visibility_degrid.py  visibility_degrid (the degridder)
│   ├── healpix/
│   │   ├── export.py             healpix_snapshot_cube_generate
│   │   └── healpix_utils.py      healpix_cnv_generate / apply, beam_image_cube, vis_model_freq_split, phase_shift_uv_image
│   ├── io/
│   │   ├── pyfhd_io.py           HDF5 save/load, recarray_to_dict, convert_sav_to_dict
│   │   └── pyfhd_quickview.py    continuum FITS / PNG export (quickview, get_image_renormalization)
│   ├── plotting/
│   │   ├── calibration.py        plot_cals
│   │   ├── gridding.py           plot_gridding
│   │   └── image.py              quick_image, plot_fits_image, log_color_calc
│   ├── pyfhd_tools/
│   │   ├── pyfhd_setup.py        argparse + YAML config + logger + validation
│   │   ├── pyfhd_utils.py        IDL-flavoured numerical primitives (histogram, rebin, l_m_n, …)
│   │   ├── test_utils.py         test fixtures / data hooks
│   │   └── unit_conv.py          coordinate conversions (alt/az ↔ ra/dec, pixel ↔ ra/dec)
│   ├── source_modeling/
│   │   └── vis_model_transfer.py model visibility I/O from sav/uvfits/h5
│   ├── templates/
│   │   └── __init__.py            (empty)
│   └── resources/
│       ├── 1088285600_example/   sample MWA observation + YAML + Gaussian beam
│       ├── config/pyfhd.yaml     default options
│       ├── healpix/EoR{0,1}_{high,low}_healpix_inds*.h5
│       ├── instrument_config/    MWA cable lengths, bandpass FITS, dipole list
│       └── test_data/            (excluded from wheel)
├── requirements.txt              uv-compiled lockfile mirror
├── tests/                        pytest tree (54 test files, see §15)
└── uv.lock                       uv lockfile (~470 KB)
```

The package source totals **15,455** lines of Python (per `wc -l`). The
largest single module is `PyFHD/calibration/calibration_utils.py` at 1,754
lines, followed by `pyfhd_tools/pyfhd_utils.py` (1,610) and
`pyfhd_tools/pyfhd_setup.py` (1,590).

## 6. Installation & Dependencies

### Quick install (`README.md` lines 28–31)

```bash
pip install pyfhd
```

then

```bash
pyfhd -v
```

### Pinned runtime dependencies (`pyproject.toml` lines 25–37)

| Package | Floor |
|---------|-------|
| `astropy` | ≥ 6.1.7 |
| `colorama` | ≥ 0.4.6 |
| `configargparse` | ≥ 1.7 |
| `h5py` | ≥ 3.13.0 |
| `healpy` | ≥ 1.18.1 |
| `importlib-resources` | ≥ 6.5.2 |
| `matplotlib` | ≥ 3.10.3 |
| `numba` | ≥ 0.61.2 |
| `numpy` | ≥ 2.2.5 |
| `pyuvdata` | ≥ 3.2.1 |
| `scipy` | ≥ 1.15.3 |

### Dev group (`pyproject.toml` lines 48–64)

`black`, `ipykernel`, `myst-parser`, `pip`, `pre-commit`, `pyinstrument`,
`pytest`, `pytest-cov`, `pytest-html`, `pytest-html-merger`, `sphinx`,
`sphinx-argparse`, `sphinx-reports`, `sphinx-rtd-theme`.

### Conda alternative (`environment.yml`)

Lists conda-forge channel only; mostly mirrors PyPI floors and adds `pip`
extras (`sphinx-rtd-theme`, `sphinx-argparse`, `sphinx-reports`,
`pytest-cov`, `pytest-html`, `pytest-html-merger`, `myst-parser`,
`healpy>=1.18.1`).

### Lockfile mirror (`requirements.txt`)

Auto-generated by `uv pip compile pyproject.toml -o requirements.txt`,
pinning concrete versions (e.g. `astropy==7.1.0`, `numpy==2.2.6`,
`pyuvdata==3.2.2`, `numba==0.61.2`, `healpy==1.18.1`).

### Notable transitive

`pyuvdata` brings in `pyerfa`, `setuptools-scm`, `docstring-parser`,
`pyyaml`. `numba` brings `llvmlite`. `astropy` brings `astropy-iers-data`.

## 7. Build System & Packaging

`pyproject.toml` line 66–73:

```toml
[build-system]
requires = ["hatch-build-scripts", "hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build]
ignore-vcs = true
include = ["PyFHD/**", "PyFHD/__git__.py"]
exclude = ["mwa_full_embedded_element_pattern.h5", "*.pyc",
           "PyFHD/resources/test_data/**/*"]
```

A `hatch-build-scripts` hook at lines 76–87 generates `PyFHD/__git__.py` at
build time, populating `__git_commit__`, `__git_commit_date__`,
`__git_branch__`. Both `pyfhd_tools/pyfhd_setup.py:pyfhd_parser`
(line 95) and `pyfhd_logger` (line 1122) try-import these to embed into the
banner / log header. Failure to import (e.g. installed-from-sdist)
gracefully falls back to `commit = "Unknown"`.

CLI entry point (`pyproject.toml` line 39–40):

```toml
[project.scripts]
pyfhd = "PyFHD.pyfhd:main"
```

`pyproject.toml` line 89–92 declares the pytest marker `github_actions`
which is used by CI to skip-or-run a subset of tests.

## 8. Runtime Architecture & Pipeline

### Layered view

```
┌─────────────────────────────────────────────────────────────────┐
│                CLI ENTRY (pyfhd.py:main)                        │
│   configargparse(YAML)  →  pyfhd_setup  →  logger               │
├─────────────────────────────────────────────────────────────────┤
│ DATA INGEST   data_setup/{uvfits.py, obs.py}                    │
│   extract_header, create_params, extract_visibilities,          │
│   create_layout, create_obs, read_metafits                      │
├─────────────────────────────────────────────────────────────────┤
│ BEAM / PSF    beam_setup/{beam.py, antenna.py, beam_utils.py}   │
│   create_psf  (UVBeam / AnalyticBeam / .sav / .h5)              │
├─────────────────────────────────────────────────────────────────┤
│ FLAG + CAL    flagging/, calibration/                           │
│   vis_flag_basic → vis_weights_update → vis_model_transfer      │
│   → calibrate (vis_calibrate_subroutine, bandpass, polyfit,     │
│      auto_ratio, qu_mixing) → vis_calibration_apply             │
├─────────────────────────────────────────────────────────────────┤
│ GRID          gridding/{visibility_grid.py, gridding_utils.py}  │
│   per-pol visibility_grid → image_uv, weights_uv, variance_uv,  │
│   uniform_filter_uv, model_uv (+ crosspol_reformat for 4-pol)   │
├─────────────────────────────────────────────────────────────────┤
│ EXPORT        io/pyfhd_quickview.py   →  per-pol FITS / PNG     │
│               healpix/export.py        →  HEALPix HDF5 cubes    │
│               io/pyfhd_io.save          →  obs / cal / vis HDF5 │
└─────────────────────────────────────────────────────────────────┘
```

### Sequence inside `PyFHD/pyfhd.py:main()` (lines 97–677)

1. `pyfhd_parser()` (line 99) → `configargparse` returns `options`.
2. If `--get-sample-data` (102): copy bundled `1088285600_example` to cwd
   and exit.
3. `pyfhd_setup(options)` returns `(pyfhd_config, logger)`.
4. **Checkpoint logic** (lines 145–189): three checkpoint files —
   `<obs>_obs_checkpoint.h5`, `<obs>_calibrate_checkpoint.h5`,
   `<obs>_gridding_checkpoint.h5` — are auto-detected when their flags are
   set; missing files demote to recompute.
5. **UVFITS ingest** (lines 191–252):
   `extract_header → create_params → extract_visibilities → create_layout
   → create_obs`. Optional saves of raw `vis_arr` / `vis_weights`.
6. (Optional) `simple_deproject_w_term` (313).
7. `vis_flag_basic` then `vis_weights_update`.
8. `vis_model_transfer` imports/normalises model visibilities.
9. `calibrate(...)` (line 366) returns calibrated `vis_arr`, `cal`, updated
   `obs`, possibly mutated `pyfhd_config`. If `n_pol >= 4`,
   `calibrate_qu_mixing` adds a Q/U mixing phase to `cal`.
10. Post-cal: `vis_weights_update`, optional `vis_flag` (full),
    `vis_noise_calc`, optional save of cal checkpoint.
11. `cal_stop` short-circuit (line 445) saves obs/params/cal/vis and exits.
12. `create_psf(obs, pyfhd_config, logger)` (line 303) — note the call
    actually happens earlier (line 302) so the beam is available at gridding.
13. **Gridding** (lines 507–604): per-polarisation `visibility_grid` builds
    `image_uv`, `weights_uv`, `variance_uv`, `uniform_filter_uv`,
    `model_uv`. `uniform_flag = (pol_i == 0)`,
    `no_conjugate = (pol_i > 1)`. For `n_pol == 4` the four planes go
    through `crosspol_reformat`.
14. Continuum FITS / PNG via `quickview()` if `export_images` set.
15. HEALPix snapshot cubes via `healpix_snapshot_cube_generate` if
    `snapshot_healpix_export` set.
16. `finish_pyfhd(...)` writes the final collated YAML and the
    `pyfhd_config.h5`, closes log handlers.

The wrapping `try / except / finally` (652–673) catches any exception, logs
the traceback, sets `pyfhd_successful = False`, exits 1.

## 9. CLI / Public API

### Console entry point

`pyfhd` (registered via `[project.scripts]`) ⇒ `PyFHD.pyfhd:main`.

### Argparse via `configargparse`

`PyFHD/pyfhd_tools/pyfhd_setup.py:pyfhd_parser()` (line 38) builds a
`configargparse.ArgumentParser` with `YAMLConfigFileParser`. The first arg
`-c/--config` defaults to the bundled
`PyFHD/resources/config/pyfhd.yaml`. A custom action class
`OrderedBooleanOptionalAction` (line 17) ensures positive-form long opts
(`--foo`) precede negated forms (`--no-foo`) so YAML/CLI override is
unambiguous.

The single positional argument is **`obs_id`** (line 130) — the MWA-style
observation ID, used to find `<input_path>/<obs_id>.uvfits` etc.

### Argument groups (with key flags)

| Group | Sample flags | File |
|-------|--------------|------|
| top-level | `obs_id`, `-i/--input-path`, `-o/--output-path`, `-r/--recalculate-all`, `-s/--silent`, `-l/--log-file`, `--instrument {mwa,ovro-lwa,hera,other}`, `--dimension`, `--elements`, `--kbinsize`, `--FoV`, `--deproject_w_term`, `--conserve-memory`, `--memory-threshold`, `--min-baseline`, `--n-pol {0,2,4}` | pyfhd_setup.py 130–227 |
| Checkpoints | `--save-checkpoints`, `--obs-checkpoint`, `--calibrate-checkpoint`, `--gridding-checkpoint` | 228–253 |
| Instrument | `--override-target-phasera`, `--override-target-phasedec` | 254–267 |
| Calibration | `-cv/--calibrate-visibilities`, `--transfer-calibration`, `--cal-stop`, `--cal-convergence-threshold` (1e-7), `--cal-adaptive-calibration-gain`, `--cal-base-gain`, `--min-cal-baseline` (50.0), `--max-cal-baseline`, `--cable-bandpass-fit`, `--cal-bp-transfer`, `--calibration-polyfit`, `--cal-amp-degree-fit` (2), `--cal-phase-degree-fit` (1), `--cal-reflection-hyperresolve`, `--cal-reflection-mode-theory` (150), `--cal-reflection-mode-delay`, `--cal-reflection-mode-file`, `--calibration-auto-fit`, `--calibration-auto-initialize`, `--cal-gain-init` (1), `--vis-baseline-hist`, `--bandpass-calibrate`, `--cal-time-average`, `--auto-ratio-calibration`, `--digital-gain-jump-polyfit`, `--cal-phase-fit-iter` (4), `--max-cal-iter` (100) | 268–431 |
| Flagging | `-fm/--flag-model`, `-fv/--flag-visibilities`, `-fc/--flag-calibration`, `-fcf/--flag-calibration-frequencies`, `-fb/--flag-basic`, `-ft/--flag-tiles`, `-ff/--flag-frequencies`, `--flag-freq-start`, `--flag-freq-end`, `--time-cut` | 433–500 |
| Beam Setup | `-b/--saved-beam-file-path`, `--uvbeam-file-path`, `--uvbeam-freq-buffer`, `--analytic-beam-yaml`, `-ll/--lazy-load-beam`, `--recalculate-beam`, `--beam-nfreq-avg` (16), `--psf-dim` (54), `--psf-resolution` (100), `--beam-mask-threshold` (100), `--beam-model-version` (2), `--beam-clip-floor`, `--interpolate-kernel`, `--beam-per-baseline`, `--beam-offset-time` (56) | 502–608 |
| Gridding | `-g/--recalculate-grid`, `--image-filter` (only `filter_uv_uniform` impl.), `--mask-mirror-indices`, `--grid-weights`, `--grid-variance`, `--grid-uniform`, `--grid-spectral` | 610–664 |
| (Deconv stub) | `--dft-threshold` (others commented out) | 666–717 |
| Export | `-o/--output-path`, `--description`, `--export-images`, `--snapshot-healpix-export`, `--pad-uv-image`, `--ring-radius-multi`, `--save-obs`, `--save-params`, `--save-cal`, `--save-visibilities`, `--save-weights`, `--save-healpix-fits`, `--save-model` | 719–798 |
| Plotting | `--calibration-plots`, `--gridding-plots`, `--image-plots` | 800–818 |
| Model | `-m/--model-file-type {sav,uvfits}`, `--model-file-path`, `--allow-sidelobe-model-sources` | 820–839 |
| HEALPIX | `--ps-kbinsize` (0.5), `--ps-kspan`, `--ps-beam-threshold`, `--ps-fov`, `--ps-dimension`, `--ps-degpix`, `--ps-nfreq-avg`, `--ps-tile-flag-list`, `--n-avg` (2), `--rephase-weights`, `--restrict-healpix-inds`, `--healpix-inds`, `--split-ps-export` | 891–970 |

There is also `--get-sample-data` (line 142) which copies the bundled
example into `./input/1088285600_example/` for first-time users.

### Programmatic API

There is **no rich Python API** beyond the CLI: `pyfhd.py:main` orchestrates
the whole pipeline. Library consumers can call:

* `pyfhd_parser()` then `pyfhd_setup(options)` to get a validated config dict.
* Stage functions individually: `extract_header`, `create_params`,
  `extract_visibilities`, `create_layout`, `create_obs`, `create_psf`,
  `vis_flag_basic`, `vis_weights_update`, `vis_model_transfer`,
  `calibrate`, `visibility_grid`, `quickview`,
  `healpix_snapshot_cube_generate`.
* I/O helpers: `from PyFHD.io.pyfhd_io import save, load`.

## 10. Configuration (YAML)

Configurations are YAML files parsed by `configargparse`. Keys use **dashes**
(e.g. `cal-amp-degree-fit`) but Python config dict keys use **underscores**
(`cal_amp_degree_fit`); the reverse mapping happens in
`write_collated_yaml_config` (line 1008) at output time:

```python
yaml_key = key.replace("_", "-")
```

### Default config (`PyFHD/resources/config/pyfhd.yaml`)

Selected highlights:

```yaml
input-path: './input'
instrument: 'mwa'
dimension: 2048
elements: 2048
kbinsize: 0.5
n-pol: 2

save-checkpoints: true
uvbeam-file-path: './input'
uvbeam-freq-buffer: 2e6
beam-nfreq-avg: 16
psf-dim: 54
psf-resolution: 100

calibrate-visibilities: true
cable-bandpass-fit: true
calibration-polyfit: true
cal-amp-degree-fit: 2
cal-phase-degree-fit: 1
cal-reflection-hyperresolve: true
cal-reflection-mode-theory: 150
calibration-auto-initialize: true
cal-convergence-threshold: 1.0e-7
bandpass-calibrate: true
auto-ratio-calibration: true
max-cal-iter: 100

image-filter: 'filter_uv_uniform'
grid-weights: true
grid-variance: true

export-images: true
save-cal: true
save-healpix-fits: true
snapshot-healpix-export: true

ps-kbinsize: 0.5
ps-kspan: 600
n-avg: 2
restrict-healpix-inds: true
split-ps-export: true
```

### Sample run config (`PyFHD/resources/1088285600_example/1088285600_example.yaml`)

Same schema; uses `saved-beam-file-path:` pointing to a Gaussian beam file
`gauss_beam_pointing0_167635008Hz.h5` packaged alongside, the
`1088285600.uvfits` data, and `1088285600_model.uvfits` as the model.

### Three mutually-exclusive beam paths

`saved-beam-file-path` (FHD-format `.sav` / `.h5`),
`uvbeam-file-path` (anything `pyuvdata.UVBeam` reads — FITS embedded
element, MWA H5, etc.), and `analytic-beam-yaml` (a `pyuvdata.AnalyticBeam`
YAML serialisation). See changelog line 5 — this was a recent breaking
change (commits `9b64546`, `0e83579`, `769b02f`).

### Output of every run

After validation, PyFHD writes a *collated* YAML back into the run's
`config/` directory: an initial copy and a `-final` copy at termination
(`pyfhd.py` lines 77–79). The runtime config is also serialised to HDF5 as
`pyfhd_config.h5`.

## 11. Module-by-Module Breakdown

### `PyFHD/pyfhd.py` (677 lines) — orchestrator

Top-level functions:

* `_print_time_diff(start, end, description, logger)` — formats run-stage
  durations.
* `finish_pyfhd(pyfhd_start, logger, psf, pyfhd_config)` — closes h5,
  writes final config, logs total runtime.
* `main()` — see §8.

### `PyFHD/data_setup/uvfits.py` (657 lines)

* `extract_header(pyfhd_config, logger, model_uvfits=False)` (line 15) —
  reads HDU0 header (params), HDU1 (antenna table), produces
  `pyfhd_header` dict with `n_tile`, `n_baselines`, `n_freq`, `n_pol`,
  `freq_ref`, `freq_res`, `frequency_array`, `dateobs`, `pol_dim`,
  `freq_dim`, axis indices.
* `create_params(pyfhd_header, params_data, logger)` (line 232) — builds
  `params` dict with `uu`, `vv`, `ww`, `time`, `baseline`, etc.
* `extract_visibilities(pyfhd_header, params_data, pyfhd_config, logger)`
  (line 300) — returns `(vis_arr, vis_weights)` of shape
  `(n_pol, n_freq, n_baselines * n_time)` complex128 / float64.
* `_check_layout_valid` (357) — header-table sanity check.
* `create_layout(antenna_header, antenna_data, pyfhd_config, logger)` (389)
  — antenna-table → dict.

### `PyFHD/data_setup/obs.py` (583 lines)

* `create_obs(pyfhd_header, params, layout, pyfhd_config, logger)` (line 22)
  — central observation metadata builder. Populates the `obs` dict with
  `n_pol`, `n_tile`, `n_freq`, `n_time`, `n_baselines`, `n_vis`,
  `freq_res`, `lon/lat/alt`, `instrument`, `obsname`,
  `baseline_info` (with `bin_offset`, `freq`, `tile_a/b`, `tile_use`,
  `freq_use`, `tile_names`, `fbin_i`, …), `min_baseline`, `max_baseline`,
  `dimension`, `elements`, `kpix`, `degpix`, `obsx/y`, `zenx/y`, etc.
* `read_metafits(...)` (280) — MWA `.metafits` parsing for additional
  pointing info.
* `project_slant_orthographic(meta, obs, epoch=2000)` (452) — sets the
  slant-orthographic projection used throughout gridding.
* `update_obs(obs, dimension, kbinsize, beam_nfreq_avg=…, fov=…)` (527) —
  recomputes derived geometry quantities (used by HEALPix export with
  different `dimension_use`).

### `PyFHD/beam_setup/beam.py` (325 lines)

`create_psf(obs, pyfhd_config, logger)` is the single public function. It
selects between three beam sources:

1. **UVBeam / AnalyticBeam** (`uvbeam_file_path` or `analytic_beam_yaml`):
   builds `antenna`, `psf`, `beam` via `init_beam`, computes a hyperresolved
   `beam_arr` of shape
   `(n_pol, n_freq_bin, psf_resolution+1, psf_resolution+1, psf_dim**2)`,
   per-polarisation power beams via `beam_power`, normalises and rolls the
   superresolution kernel into `psf_single`, and saves to
   `<output>/beams/<obs_id>_beam.h5` with chunking.
2. **`.sav`**: reads the FHD `psf` IDL struct via `scipy.io.readsav`,
   collapses the per-baseline pointer (assuming a single beam), passes
   through `recarray_to_dict`, saves an HDF5 mirror.
3. **`.h5` / `.hdf5`**: directly loads via `pyfhd_io.load` (with
   optional `lazy_load`).

### `PyFHD/beam_setup/antenna.py` (461 lines)

* `init_beam(obs, pyfhd_config, logger)` (line 19) — wraps `pyuvdata`'s
  `BeamInterface` / `UVBeam` / `AnalyticBeam`. Currently assumes 2
  instrumental polarisations (linear or circular). Returns
  `(antenna, psf, beam)` where `psf` carries `dim`, `resolution`,
  `intermediate_res`, `superres_dim`, `image_dim`, `scale`, `xvals`,
  `yvals`, etc.
* `general_jones_matrix(...)` (line 307) — builds Jones / response matrices
  from instrument geometry.

### `PyFHD/beam_setup/beam_utils.py` (492 lines)

* `gaussian_decomp(x, y, p, ftransform=False, model_npix=None,
  model_res=None)` (line 11) — Analytic Gaussian-decomposition beam (or
  its Fourier transform).
* `beam_image(...)` (120) — produce an image-plane beam.
* `beam_image_hyperresolved(...)` (298).
* `beam_power(antenna, beam, ant_pol1, ant_pol2, freq_i, psf, …)` (384) —
  power beam product `J · J^H` projected onto the hyperresolved UV grid.

### `PyFHD/calibration/calibrate.py` (288 lines)

* `calibrate(obs, params, vis_arr, vis_weights, vis_model_arr,
  pyfhd_config, logger)` (line 22) — high-level driver. Pipeline:
  1. Initialise `cal` dict (`n_pol`, `conv_thresh`, `ref_antenna`).
  2. `vis_extract_autocorr` for data and model autos.
  3. Optional `vis_cal_auto_init` to seed gains.
  4. `vis_calibrate_subroutine` — least-squares solve.
  5. Optional `vis_calibration_flag`.
  6. Optional `cal_auto_ratio_divide` + `vis_cal_bandpass`.
  7. Optional `vis_cal_polyfit` (per-pol).
  8. Optional `cal_auto_ratio_remultiply`.
  9. Compute `cal_res_gain = cal_base[gain] - cal[gain]` (FHD
     `vis_cal_subtract` replacement).
  10. `vis_calibration_apply` returns calibrated `vis_arr`.
  11. Stats: `mean_gain`, `mean_gain_residual`, `mean_gain_restrict` (via
      `resistant_mean`), `stddev_gain_residual`.
  12. Optional `plot_cals`.
* `calibrate_qu_mixing(vis_arr, vis_model_arr, vis_weights, obs)` (195) —
  solves `arctan2(imag, real)` of the U→Q linear-fit slope on calibrated
  vs model `pseudo_q = YY-XX`, `pseudo_u = YX+XY`, returning
  `u_q_phase_model - u_q_phase`.

### `PyFHD/calibration/calibration_utils.py` (1,754 lines)

| Function | Line | Role |
|----------|------|------|
| `vis_extract_autocorr` | 18 | Pulls auto-correlations (where `tile_a == tile_b`). |
| `vis_cal_auto_init` | 99 | Seed gains from auto amplitudes. |
| `vis_calibration_flag` | 168 | Antenna-level flagging from gains. |
| `transfer_bandpass` | 322 | Read external bandpass file. |
| `vis_cal_bandpass` | 692 | Compute bandpass solutions (all-tile, cable-grouped, or per-pointing). |
| `vis_cal_polyfit` | 856 | Polyfit gains amp/phase + reflection-mode fitting (theory, delay, file, hyperresolve). |
| `vis_cal_auto_fit` | 1287 | Fit gains via auto correlations. |
| `vis_calibration_apply` | 1382 | Multiply gains into visibilities. |
| `vis_baseline_hist` | 1512 | Resolution-ratio histogram diagnostics. |
| `cal_auto_ratio_divide` | 1587 | Divide gains by auto ratio (per-pol). |
| `cal_auto_ratio_remultiply` | 1631 | Inverse op. |
| `calculate_adaptive_gain` | 1662 | Kalman-filter style adaptive gain selector for the LS solver. |

### `PyFHD/calibration/vis_calibrate_subroutine.py` (460 lines)

Pure-NumPy implementation of an iterative weighted linear least-squares
gain solver. Hardcoded `reference_tile = 1`, `min_cal_solutions = 5`. Honors
`min_cal_baseline`, `max_cal_baseline`, `cal_time_average`, `max_cal_iter`,
`cal_phase_fit_iter`, `cal_convergence_threshold`,
`cal_adaptive_calibration_gain`, `cal_base_gain`. Convergence test is on
the absolute fractional change in gains.

### `PyFHD/flagging/flagging.py` (365 lines)

* `vis_flag_tiles(obs, vis_weight_arr, tiles_to_flag, logger)` (7) —
  zero-out weights for named tiles.
* `vis_flag_basic(vis_weight_arr, vis_arr, obs, pyfhd_config, logger)` (64)
  — frequency / tile coarse flags from
  `flag_freq_start/end`, `flag_tiles`, etc.
* `vis_flag(vis_arr, vis_weights, obs, params)` (213) — full statistical
  flagging.

### `PyFHD/gridding/visibility_grid.py` (585 lines)

The gridder. Iterates over unflagged baselines, looks up the hyperresolved
kernel from `psf` (optionally bilinearly interpolated via
`interpolate_kernel`), accumulates onto `image_uv`, `weights`, `variance`.
Optionally builds a per-baseline beam (`grid_beam_per_baseline`). Handles
`mask_mirror_indices`, `no_conjugate`, `uniform_flag`. Returns a dict with
`image_uv`, `weights`, `variance`, `uniform_filter` (only when
`uniform_flag`), `obs` (with updated `nf_vis`), `model_return`.

### `PyFHD/gridding/visibility_degrid.py` (376 lines)

The inverse: takes a model `image_uv` plane and predicts visibilities. Key
options: `fill_model_visibilities`, `vis_input` (extras),
`spectral_model_uv_arr`, `beam_per_baseline`, `uv_grid_phase_only`,
`conserve_memory`, `memory_threshold`. (Only used when a UV-domain model is
supplied; the default pipeline imports a pre-computed model.)

### `PyFHD/gridding/gridding_utils.py` (890 lines)

Shared primitives:

| Function | Line | Notes |
|----------|------|-------|
| `interpolate_kernel` | 12 | Bilinear interpolation of hyperresolved kernel; swaps `x_offset/y_offset` for IDL→Python ordering. |
| `conjugate_mirror` | 57 | Hermitian mirror to populate `v < 0` half. |
| `baseline_grid_locations` | 84 | Per-baseline pixel binning, returns `bin_n`, `bin_i`, `ri`, `xmin`, `ymin`. |
| `dirty_image_generate` | 297 | UV → image FFT with optional uniform filter, padding, beam normalisation. |
| `grid_beam_per_baseline` | 498 | Per-baseline beam construction (used when `beam_per_baseline`). |
| `visibility_count` | 691 | Visibility-per-pixel histogram for uniform weighting. |
| `holo_mapfn_convert` | 783 | Converts a sparse holographic mapping function. |
| `crosspol_reformat` | 868 | XY/YX real-imag combination for 4-pol export. |

### `PyFHD/gridding/filters.py` (96 lines)

Currently only `filter_uv_uniform`. Other filters (hanning, natural,
radial, tapered_uniform, optimal) are commented out in the parser
(`pyfhd_setup.py` lines 622–632) — see §19.

### `PyFHD/source_modeling/vis_model_transfer.py` (542 lines)

* `vis_model_transfer(pyfhd_config, obs, params, logger)` (16) — dispatcher.
* `import_vis_model_from_sav` (83) — multi-file FHD `.sav` reader
  (`<obs>_vis_model_<pol>.sav` + `<obs>_params.sav`).
* `import_vis_model_from_uvfits` (174) — UVFITS path; the model phase
  centre must match the metafits `RA/DEC`, **not** `RAPHASE/DECPHASE`.
* `_FlaggingInfoCounter` class (220) — diagnostics during model alignment.
* `flag_model_visibilities` (278) — applies time and tile flag alignment.
* `convert_vis_model_arr_to_sav` (479) — round-trip back to FHD format.

### `PyFHD/healpix/export.py` (229 lines)

`healpix_snapshot_cube_generate(obs, psf, cal, params, vis_arr,
vis_model_arr, vis_weights, pyfhd_config, logger)` builds:

* `obs_out` via `update_obs` with HEALPix-tuned `dimension_use`,
  `kbinsize`, `beam_nfreq_avg`, `fov_use`.
* `beam_arr, beam_mask` via `beam_image_cube` (square=True).
* `hpx_cnv` via `healpix_cnv_generate` (with `nside` derived from
  `4π · (180/π)² / degpix_use²` rounded to the next power of two).
* Even/odd split via `split_vis_weights` if `split_ps_export`.
* Per polarisation, `vis_model_freq_split` separates dirty/model/residual
  + grids `weights_arr`, `variance_arr`, then `healpix_cnv_apply`
  produces `beam_squared_cube`, `weights_cube`, `variance_cube`,
  `model_cube`, `dirty_or_res_cube`.
* Saved to `<output>/healpix/<obs_id>_<cube_name>_<pol>.h5`.

### `PyFHD/healpix/healpix_utils.py` (697 lines)

| Function | Line | Notes |
|----------|------|-------|
| `healpix_cnv_apply` | 24 | Sparse-matrix-vector (replaces FHD's `sprsax2`). |
| `healpix_cnv_generate` | 60 | Builds `inds`, `i_use`, `ija`, `sa` from `query_disc` + interpolation weights. Optionally restricted to `healpix-inds` files. |
| `beam_image_cube` | 326 | Per-frequency primary-beam cube (square=True returns `B²`). |
| `phase_shift_uv_image` | 420 | Apply phase shift between phase centres. |
| `vis_model_freq_split` | 465 | Per-pol gridding of model + data into per-freq slices. |

### `PyFHD/io/pyfhd_io.py` (704 lines)

A self-contained HDF5 dictionary I/O layer.

| Function | Line | Purpose |
|----------|------|---------|
| `dtype_picker(dtype)` | 13 | Promote everything to double precision (int64/float64/complex128) on save. |
| `_is_complex / _is_string / _is_none` | 40/61/80 | Vectorised type-probes for object arrays (mirrors what scipy.io.readsav returns from IDL). |
| `_decode_byte_arr` | 99 | bytes → str. |
| `format_array` | 116 | Normalise object arrays → typed arrays. |
| `save_dataset` | 167 | Low-level chunked dataset writer. |
| `dict_to_group` | 309 | Recursive nested-dict → HDF5. |
| `save(filepath, data, root_name, logger=…, to_chunk=…)` | 344 | Public entry. |
| `load_dataset` | 425 | Inverse low-level. |
| `group_to_dict` | 469 | HDF5 → nested-dict. |
| `load(filepath, logger=…, lazy_load=False)` | 494 | Public entry — returns dict or `h5py.File` when lazy. |
| `recarray_to_dict(data)` | 554 | Walks `np.recarray` (esp. from `readsav`) into a Pythonic dict. |
| `convert_sav_to_dict(sav_path, logger, tmp_dir="temp_pyfhd")` | 652 | Save bridge to load FHD outputs. |

### `PyFHD/io/pyfhd_quickview.py` (478 lines)

* `get_image_renormalization(obs, weights, beam_base, filter_arr,
  pyfhd_config, logger)` (22) — Jy/beam → Jy/sr per polarisation.
* `quickview(obs, psf, params, cal, vis_arr, vis_weights, image_uv,
  weights_uv, variance_uv, uniform_filter_uv, model_uv, pyfhd_config,
  logger)` (78) — orchestrates per-pol continuum FITS / PNG dumps,
  computes residuals, calls `dirty_image_generate`, writes
  `<obs_id>_<kind>_<pol>.fits` and corresponding plots.

### `PyFHD/plotting/`

| File | Public function(s) | Notes |
|------|--------------------|-------|
| `image.py` | `quick_image`, `log_color_calc`, `color_range`, `plot_fits_image` | Uses `matplotlib.use("pdf")`-style headless config in `calibration.py`. |
| `calibration.py` | `plot_cals(obs, cal, pyfhd_config)` | 128-tile-per-page grids of amp/phase, residuals, raw. |
| `gridding.py` | `plot_gridding(obs, image_uv, weights_uv, variance_uv, pyfhd_config, model_uv, logger)` | Calls `quick_image` for each plane. |

### `PyFHD/pyfhd_tools/pyfhd_setup.py` (1,590 lines)

Builds the entire CLI surface; performs validation; sets up logging.

* `OrderedBooleanOptionalAction` (17) — argparse fix-up.
* `pyfhd_parser()` (38) — assembles the full parser.
* `_check_file_exists(config, key)` (975) — validation helper.
* `write_collated_yaml_config(pyfhd_config, output_dir, description="")`
  (1008) — emits the run's YAML.
* `pyfhd_logger(pyfhd_config)` (1089) — banner + StreamHandler +
  FileHandler in `<output>/<pyfhd_..._description>/<log_name>.log`.
* `pyfhd_setup(options)` — top-level validation entry (called from
  `pyfhd.main`, line 109). Returns `(pyfhd_config, logger)`.

### `PyFHD/pyfhd_tools/pyfhd_utils.py` (1,610 lines)

The IDL-flavoured numerical toolkit. Highlights:

| Function | Line | Notes |
|----------|------|-------|
| `get_bins / get_hist / get_ri` | 16 / 53 / 113 | `@njit` numba kernels for IDL-style `HISTOGRAM`. |
| `histogram(data, bin_size=…, min=…, max=…)` | 224 | Returns `(hist, bins, reverse_indices)`. |
| `l_m_n(...)` | 297 | Direction-cosine geometry for sources. |
| `rebin_columns / rebin_rows / rebin` | 383 / 430 / 482 | IDL `REBIN` with both up- and down-sampling semantics. |
| `weight_invert(arr, threshold=…)` | 646 | Safe inversion (zero where below threshold). |
| `array_match(a, b)` | 730 | IDL `WHERE`-style match. |
| `meshgrid(...)` | 805 | IDL-style meshgrid. |
| `deriv_coefficients(n, divide_factorial=False)` | 851 | Polynomial derivative coefficients. |
| `idl_argunique(arr)` | 888 | Stable IDL `UNIQ`. |
| `angle_difference(a, b)` | 915 | Wrap-aware diff. |
| `parallactic_angle(latitude, hour_angle, dec)` | 961 | Standard formula. |
| `simple_deproject_w_term(obs, params, vis_arr, direction, logger)` | 987 | UV-plane w-term deprojection. |
| `resistant_mean(arr, sigma_clip)` | 1027 | IDL `RESISTANT_MEAN`. |
| `run_command(cmd, dry_run=False)` | 1097 | subprocess wrapper. |
| `vis_weights_update(vis_weights, obs, psf, params)` | 1117 | Recompute `obs['nf_vis']` and zero-fail freqs. |
| `split_vis_weights(obs, vis_weights)` | 1228 | Even/odd time-step bisection for power spectrum. |
| `vis_noise_calc(obs, vis_arr, vis_weights, bi_use=…)` | 1310 | Visibility noise statistics. |
| `idl_median(arr, …)` | 1366 | IDL `MEDIAN` with `even` switch. |
| `reshape_and_average_in_time(...)` | 1422 | Used in `calibrate_qu_mixing`. |
| `region_grow(…)` | 1466 | Connected-region growing for masks. |
| `crosspol_split_real_imaginary` | 1576 | XY/YX → 4 real planes. |

### `PyFHD/pyfhd_tools/unit_conv.py` (203 lines)

Astropy-backed coordinate conversions — `altaz_to_radec`,
`radec_to_altaz`, `radec_to_pixel`, `pixel_to_radec` and friends. Default
location is MWA's `lat / lon / height`.

### `PyFHD/pyfhd_tools/test_utils.py` (295 lines)

Test fixtures and Zenodo download helpers (per `Test Data DOI` badge).

## 12. Core Algorithms

### Holographic gridding kernel

The PSF (point-spread / spreading function) is **hyperresolved by a factor
`psf_resolution`** (default 100) so that for any baseline (which lands at a
non-integer pixel in the UV grid) the closest hyperresolved pixel is a
sufficient lookup. Optional bilinear interpolation
(`gridding_utils.interpolate_kernel`, line 12) refines the lookup using
the four nearest hyperresolved pixels, scaled by the 2-D derivatives
`(dx0dy0, dx1dy0, dx0dy1, dx1dy1)`.

The kernel itself is the FFT of the antenna-pair power beam (`beam_power`
in `beam_utils.py:384`) on a superres grid of size
`(superres_dim, superres_dim)`, then sliced and rolled into a 4-D table
indexed by `(i_super, j_super, dim, dim)` per (pol, freq).

This is the "fast holographic" part: one FFT per (pol, freq) into a
hyperresolved table, then O(N_baselines · psf_dim²) lookups.

### Slant-orthographic projection

`PyFHD/data_setup/obs.py:project_slant_orthographic` (line 452) sets up
the projection geometry (`obsx`, `obsy`, `zenx`, `zeny`, `kpix`,
`degpix`). Visibilities map to a UV plane that, when Fourier-transformed,
yields a slant-orthographic image of the sky tangent at the phase centre.

### W-term handling

* **`simple_deproject_w_term`** (`pyfhd_utils.py:987`): a single-direction
  W-term deprojection driven by `--deproject_w_term <radians>`.
* **No explicit w-stacking / w-projection** is implemented in PyFHD as of
  v1.0.2; the option to disable W-projection is implicit (the kernel is
  FFTed without W-correction unless `simple_deproject_w_term` is used).

### Calibration

The fully featured pipeline in `calibration/calibrate.py` follows FHD:

1. **Auto-init** (optional) — gain seeded from auto-correlation amplitudes.
2. **Per-frequency LS solve** in `vis_calibrate_subroutine.py` —
   amplitude-only for the first `cal_phase_fit_iter` iterations
   (default 4), then amplitude+phase. Adaptive gain (Kalman-style) is
   available via `calculate_adaptive_gain`.
3. **Auto-ratio bandpass** (`cal_auto_ratio_divide` /
   `cal_auto_ratio_remultiply`) lets bandpass be computed on
   auto-correlations and reapplied to the cross-pol gains.
4. **Bandpass average** across tiles, optionally per-cable-group
   (`cable-bandpass-fit` + `mwa_cable_length.txt`).
5. **Polynomial fit** of amp (degree 2) and phase (degree 1) over
   frequency.
6. **Cable reflection** modes — three modes:
   `cal-reflection-mode-theory` (theoretical from cable length / velocity),
   `cal-reflection-mode-delay` (FFT residual gains pick max mode),
   `cal-reflection-mode-file` (predetermined modes/amps/phases). Optional
   hyperresolution via `--cal-reflection-hyperresolve`.
7. **Auto-fit** (`vis_cal_auto_fit`) substitutes auto-derived gains for
   diagnostic output.
8. **Apply** via `vis_calibration_apply` and zero-flag any failed
   frequencies if `flag_calibration_frequencies`.
9. **QU mixing**: with `n_pol >= 4`, `calibrate_qu_mixing` solves the
   excess Q–U rotation angle.

### HEALPix snapshot cubes for ε-ppsilon

`healpix_snapshot_cube_generate` outputs power-spectrum-ready cubes:
`beam_squared_cube`, `weights_cube`, `variance_cube`, `model_cube`, plus
either `dirty_cube` or `res_cube`. Even/odd time-split support
(`split-ps-export`) is for jackknife noise estimation. The
`restrict-healpix-inds` option uses pre-computed pixel index files in
`PyFHD/resources/healpix/` for EoR0/EoR1 high/low fields, accelerating the
gridding convolution generation.

### Deconvolution (not yet in PyFHD)

Comments in `pyfhd_setup.py` lines 666–717 and 841–890 reveal that the
deconvolution and simulation modes are **stubbed out** pending translation:

> "# Ready for deconvolution translation"
> "# Ready for simulation translation"

Only `--dft-threshold` survives (line 680) for source DFT use elsewhere.

## 13. Input & Output Formats

### Inputs

| Kind | Format | Reader |
|------|--------|--------|
| Visibility data | UVFITS (`<obs_id>.uvfits`) | `astropy.io.fits` via `data_setup/uvfits.py` |
| Metafits | MWA `.metafits` | `data_setup/obs.py:read_metafits` |
| Beam | UVBeam-compatible (FITS embedded element, MWA H5, etc.) | `pyuvdata` |
| Beam | FHD `.sav` | `scipy.io.readsav` → `recarray_to_dict` |
| Beam | PyFHD-native HDF5 | `pyfhd_io.load` (with optional lazy_load) |
| Analytic beam | YAML string | `pyuvdata.AnalyticBeam` |
| Model | UVFITS or per-pol `.sav` (or `.h5`) | `source_modeling/vis_model_transfer.py` |
| Calibration transfer | `<obs>_cal.h5` from prior PyFHD run | `pyfhd_io.load` |
| HEALPix index restriction | `EoR{0,1}_{high,low}_healpix_inds*.h5` | `pyfhd_io.load` |
| Cable lengths | `<instrument>_cable_length.txt` | `calibration_utils` parsing |
| Cable reflections | `<instrument>_cable_reflection_coefficients.txt` | per-tile model |

### Outputs (per run, under `<output_path>/pyfhd_<description?>_<timestamp>/`)

```
<run>/
├── pyfhd_<…>.log                 (text log)
├── config/
│   ├── pyfhd_<…>.yaml            initial collated config
│   ├── pyfhd_<…>-final.yaml
│   └── pyfhd_config.h5
├── beams/<obs_id>_beam.h5        (psf dictionary)
├── visibilities/
│   ├── <obs_id>_raw_vis_arr.h5
│   ├── <obs_id>_raw_vis_weights.h5
│   ├── <obs_id>_calibrated_vis_arr.h5
│   └── <obs_id>_calibrated_vis_weights.h5
├── metadata/<obs_id>_obs.h5, <obs_id>_params.h5
├── calibration/<obs_id>_cal.h5
├── model/<obs_id>_vis_model.h5
├── checkpoints/<…>_obs_checkpoint.h5, …_calibrate_checkpoint.h5,
│                  …_gridding_checkpoint.h5
├── plots/
│   ├── calibration/<obs_id>_cal_amp.{pdf,png}, _cal_phase, _residual_amp …
│   └── gridding/<obs_id>_grid_apparent_image_<pol>.png …
├── fits/<obs_id>_uniform_dirty_<pol>.fits, _residual_<pol>.fits, …
└── healpix/<obs_id>_{healpix_cube,hpx_even,hpx_odd}_<pol>.h5
```

### HDF5 conventions

`pyfhd_io.save` enforces double precision via `dtype_picker`; nested
dictionaries become groups; chunking can be controlled per-dataset via
the `to_chunk={"beam_ptr": {"shape": …, "chunk": …}}` argument (see
`beam.py` line 269 and `pyfhd_io.save_dataset`).

Lazy-loading of the beam returns an `h5py.File` instead of a dict; the
pipeline can operate either way (most consumers test
`isinstance(psf, h5py.File)`).

## 14. Resources Bundled with the Package

`PyFHD/resources/`:

* **`config/pyfhd.yaml`** — default configuration.
* **`1088285600_example/`** — full self-contained MWA example (UVFITS data
  + UVFITS model + metafits + Gaussian beam H5 + tuned YAML).
* **`healpix/`** — six pixel-index restriction files for EoR0 / EoR1
  (high & low band, plus `_3x` and `_large` variants for EoR0 high).
* **`instrument_config/`**:
  * `mwa_cable_length.txt`
  * `mwa_cable_reflection_coefficients.txt`
  * `mwa_dead_dipole_list.txt`
  * `mwa_eor0_highband_season1_cable_bandpass.fits`
  * `mwa_LNA_impedance.sav`
  * `mwa_ZMatrix.fits`
* **`test_data/`** — populated by tests; excluded from the wheel
  (`pyproject.toml` line 73). Real data lives on Zenodo
  (`10.5281/zenodo.15687722`).

The Dockerfile (line 36) downloads the MWA full embedded element pattern
from the MWA Telescope's static URL into `instrument_config/`.

## 15. Testing Layout

54 test files under `simulators/PyFHD/tests/`, organised mirroring the
package structure:

| Subdir | Sample tests |
|--------|--------------|
| `test_beam/` | `test_beam_image.py`, `test_gaussian_decomp.py` |
| `test_calibration/` | `test_cal_auto_ratio_divide.py`, `…_remultiply.py`, `test_calculate_adaptive_gain.py`, `test_calibrate_qu_mixing.py`, `test_split_vis_weights.py`, `test_vis_baseline_hist.py`, `test_vis_cal_auto_fit.py`, `test_vis_cal_auto_init.py`, `test_vis_cal_bandpass.py`, `test_vis_cal_polyfit.py`, `test_vis_calibration_apply.py`, `test_vis_calibration_flag.py`, `test_vis_calibration_subroutine.py`, `test_vis_extract_autocorr.py`, `test_vis_noise_calc.py` |
| `test_data_setup/` | `test_extract_visibilities.py`, `test_obs.py`, `test_sample_data_extraction.py`, `test_w_term.py` |
| `test_flagging/` | `test_vis_flag.py`, `test_vis_flag_basic.py` |
| `test_gridding/test_gridding_utils/` | `test_baseline_grid_locations.py`, `test_conjugate_mirror.py`, `test_dirty_image_generate.py`, `test_grid_beam_per_baseline.py`, `test_holo_mapfn_convert.py`, `test_interpolate_kernel.py`, `test_visibility_count.py` |
| `test_gridding/test_filters/` | `test_filter_uv_uniform.py` |
| `test_gridding/` | `test_visibility_grid.py`, `test_visibility_degrid.py` |
| `test_healpix/` | `test_beam_image_cube.py`, `test_healpix_cnv_apply.py`, `test_healpix_cnv_generate.py`, `test_phase_shift_uv_image.py`, `test_vis_model_freq_split.py` |
| `test_io/` | `test_configuration.py`, `test_save_and_load.py` |
| `test_pyfhd_tools/` | `test_array_match.py`, `test_deriv_coefficients.py`, `test_histogram.py`, `test_l_m_n.py`, `test_meshgrid.py`, `test_rebin.py`, `test_region_grow.py`, `test_resistant_mean.py`, `test_vis_weights_update.py`, `test_weight_invert.py` |
| `test_quickview/` | `test_get_image_renormalization.py`, `test_quickview.py` |
| `test_source_modeling/` | `test_vis_model_transfer.py` |
| `test_prep_scripts/` | `splitter.py` (helper to split sav files for fixture prep) |

`pyproject.toml` line 89–92 declares marker `github_actions` (used to scope
tests on CI). The test workflow is `.github/workflows/test.yml`. Coverage
reports are produced via `pytest-cov` and rendered into the docs at
`docs/source/_static/pyfhd_coverage_report/`. There is **no `conftest.py`**
in the tree (none returned by `find`), so fixtures live in the local
`test_*.py` files and `pyfhd_tools/test_utils.py`.

## 16. Documentation Site

Sphinx site under `docs/source/`:

* `index.rst` — landing page.
* `installation/installation.md`.
* `tutorial/tutorial.rst` — walks the bundled `1088285600_example` and
  also `1088281328`, `1091128160`. Tutorial PNGs include calibration
  amplitude/phase plots, gridded-image XX/YY, gridded-model XX/YY,
  variance, and screenshots of the MWA ASVO portal (`data_job_form.png`,
  `meta_job_form.png`, `download_ready.png`, `jobs_ready.png`,
  `h5_web.png`).
* `documentation/documentation.rst` — auto-generated API reference (uses
  `sphinx-argparse` for CLI, `sphinx-reports`, `sphinx-rtd-theme`).
* `develop/contribution_guide.md` and `develop/idl_translation.md` — IDL→Py
  porter's playbook.
* `changelog/changelog.md` — release notes.
* `reports/` — Sphinx pages embedding pytest HTML and coverage reports.
* `_static/pyfhd_coverage_report/` — HTML coverage tree for every module.

Build pinned in `docs/requirements.txt` and exposed at
`https://pyfhd.readthedocs.io/en/latest/` (per `.readthedocs.yaml`).

## 17. CI, Docker & Release Workflows

### `.github/workflows/`

| Workflow | Purpose |
|----------|---------|
| `black.yml` | Pre-commit black formatting check. |
| `dockerhub.yaml` | Build & push `skywa7ch3r/pyfhd:latest` image. |
| `publish.yml` | PyPI publish on tag. |
| `test.yml` | pytest matrix; surfaces results via GitHub Pages
  (badge link in README line 9). |
| `zenodo.yml` | Zenodo deposit on release. |

### `Dockerfile`

Two-stage build:

1. `ghcr.io/astral-sh/uv:python3.13-bookworm AS builder` — `uv sync
   --frozen --no-dev`, prefetches astropy IERS table and
   `EarthLocation.get_site_names()`, downloads the MWA FEE beam
   (`mwa_full_embedded_element_pattern.h5`) into
   `instrument_config/`, removes uv cache and `.git`.
2. `python:3.13-slim-bookworm AS runner` — copies the populated
   virtualenv from the builder, sets `PATH`, `XDG_CACHE_HOME`.

### Pre-commit

`.pre-commit-config.yaml` (480 bytes) wires `black` (and likely a
trailing-whitespace / EOL hook).

## 18. Notable Internals & Idioms

* **Shape & dtype contracts** — visibilities are
  `(n_pol, n_freq, n_baselines · n_time)` complex128; weights same shape
  float64; UV grids `(n_pol, elements, dimension)` complex128
  (`pyfhd.py` lines 509–516).
* **All HDF5 saves are double-precision** — `dtype_picker` is the
  invariant. This is why "Python is double-precision" appears in the
  README's caveat about results.
* **Banner-driven UX** — the ASCII `pyfhd` banner in `pyfhd_parser` and
  `pyfhd_logger` is rendered at every run, with version, commit, branch,
  config path, observation ID.
* **`OrderedBooleanOptionalAction`** — guarantees CLI overrides of YAML
  bool flags work even when the YAML positively asserts `true`/`false`.
* **`configargparse` + YAML** — single source of truth: the CLI parser
  *is* the schema; `--no-*` switches are auto-generated; the YAML uses
  dashes with `~` (None) as null.
* **Run-isolation directory** — every run gets its own
  `pyfhd_<description?>_<timestamp>/` so reruns never overwrite each
  other; the collated `*-final.yaml` makes runs reproducible.
* **Lazy beam loading** — `--lazy-load-beam` keeps the beam as an
  `h5py.File` so very large beam tables (tens of GB) can be streamed
  off disk during gridding.
* **Numba acceleration** is currently confined to histogramming kernels
  in `pyfhd_utils.py`; the rest of the code is pure NumPy / SciPy /
  AstroPy.
* **Checkpoints, not pickles** — every checkpoint is HDF5, so partial
  pipelines can be inspected outside Python.
* **No GPU, no MPI, no Dask** — strictly single-process. Memory-hungry
  loops can be tamed via `--conserve-memory` + `--memory-threshold`.
* **Cross-language conventions** — comments throughout the codebase
  flag column-major ↔ row-major swaps, single- vs double-precision
  divergence from FHD, and IDL-isms preserved deliberately
  (`idl_argunique`, `idl_median`, `histogram` with reverse indices).

## 19. Known Limitations / TODOs

From the source comments and the parser groups:

* **Image filters** other than `filter_uv_uniform` are listed but not
  implemented. The parser comment (lines 626–632) notes:

  > "the following are not implemented yet. So the code just uses
  > uniform but names the files to match the selection (so it lies)
  > commenting these out as options to prevent confusion."

* **Deconvolution mode** — entirely commented out in
  `pyfhd_setup.py` (lines 666–717). Only `--dft-threshold` survives.
* **Simulation mode** — also commented out (lines 841–890): no
  `run-simulation`, `in-situ-sim-input`, `eor-vis-filepath`,
  `enhance-eor`, `sim-noise`, `tile-flag-list`, `remove-sim-flags`,
  `extra-vis-filepath`. Comment (line 87 in the parser): "# Ready for
  simulation translation".
* **Per-baseline beams** — `beam.py` line 25 explicitly states "PyFHD
  was made with the assumption that the beam is the MWA beam, and
  assumes the beam does not differ on a per baseline basis. If you wish
  to use separate baselines you'll need to add that functionality
  yourself."
* **Group loop** — `beam.py` line 149 has a `# TODO: actually put in
  group loop and functionality` for the multi-beam case.
* **`vis_cal_combine`** — replaced inline by gain-product in
  `calibrate.py:120` rather than a dedicated function.
* **JOSS paper** — `README.md` line 91: "TODO: A JOSS Paper is being
  done…".
* **Multiprocessing for gridding** — `pyfhd.py` line 522 notes "Since
  it's done per polarization, we can do multi-processing if it's not
  fast enough" but no parallelism is wired in yet.
* **Plotting after checkpointing** — `pyfhd.py` line 570: "TODO: move
  this after the checkpointing so an error in plotting doesn't require
  rerunning gridding."
* **Beam offset `+1`** — `beam.py` line 55 has a TODO about IDL's 1-based
  indexing residue (`# TODO: we'll see if the +1 is necessary`).
* **Dependence on FHD-format `.sav` beams** — slow ingest; the warning at
  `beam.py:280` actually jokes about the load time:

  > "Reading in a beam sav file probably will take a long time… maybe
  > watch your favourite long movie, for example the extended edition
  > of LOTR: Return of the King…"

The unreleased changelog (`docs/source/changelog/changelog.md`) lists
recent breaking changes (UVBeam path overhaul, `beam-file-path` →
`saved-beam-file-path`) and many bug fixes (gridding-kernel centring, beam
phase NaNs, beam squared area, FFT direction, n_tile calc, OVRO-LWA
support, telescope location from antenna table, sub-horizon point
filtering for UVBeam). Version **1.0.2** primarily fixed
`vis-baseline-hist`-option usage and pointed docs at GitHub Pages.

## 20. Quick Reference Tables

### Module → key public callables

| Module | Public functions |
|--------|------------------|
| `PyFHD.pyfhd` | `main`, `finish_pyfhd` |
| `PyFHD.pyfhd_tools.pyfhd_setup` | `pyfhd_parser`, `pyfhd_setup`, `pyfhd_logger`, `write_collated_yaml_config` |
| `PyFHD.data_setup.uvfits` | `extract_header`, `create_params`, `extract_visibilities`, `create_layout` |
| `PyFHD.data_setup.obs` | `create_obs`, `read_metafits`, `update_obs`, `project_slant_orthographic` |
| `PyFHD.beam_setup.beam` | `create_psf` |
| `PyFHD.beam_setup.antenna` | `init_beam`, `general_jones_matrix` |
| `PyFHD.beam_setup.beam_utils` | `beam_image`, `beam_image_hyperresolved`, `beam_power`, `gaussian_decomp` |
| `PyFHD.flagging.flagging` | `vis_flag`, `vis_flag_basic`, `vis_flag_tiles` |
| `PyFHD.calibration.calibrate` | `calibrate`, `calibrate_qu_mixing` |
| `PyFHD.calibration.calibration_utils` | (12 fns; see §11) |
| `PyFHD.calibration.vis_calibrate_subroutine` | `vis_calibrate_subroutine` |
| `PyFHD.gridding.visibility_grid` | `visibility_grid` |
| `PyFHD.gridding.visibility_degrid` | `visibility_degrid` |
| `PyFHD.gridding.gridding_utils` | (8 fns; see §11) |
| `PyFHD.gridding.filters` | `filter_uv_uniform` |
| `PyFHD.source_modeling.vis_model_transfer` | `vis_model_transfer`, `import_vis_model_from_sav`, `import_vis_model_from_uvfits`, `flag_model_visibilities`, `convert_vis_model_arr_to_sav` |
| `PyFHD.healpix.export` | `healpix_snapshot_cube_generate` |
| `PyFHD.healpix.healpix_utils` | `healpix_cnv_apply`, `healpix_cnv_generate`, `beam_image_cube`, `phase_shift_uv_image`, `vis_model_freq_split` |
| `PyFHD.io.pyfhd_io` | `save`, `load`, `save_dataset`, `load_dataset`, `dict_to_group`, `group_to_dict`, `recarray_to_dict`, `convert_sav_to_dict`, `dtype_picker`, `format_array` |
| `PyFHD.io.pyfhd_quickview` | `quickview`, `get_image_renormalization` |
| `PyFHD.plotting.calibration` | `plot_cals` |
| `PyFHD.plotting.gridding` | `plot_gridding` |
| `PyFHD.plotting.image` | `quick_image`, `plot_fits_image`, `log_color_calc`, `color_range` |
| `PyFHD.pyfhd_tools.pyfhd_utils` | (24 fns; see §11) |
| `PyFHD.pyfhd_tools.unit_conv` | `altaz_to_radec`, `radec_to_altaz`, `radec_to_pixel`, `pixel_to_radec` |

### Pipeline data flow

| Stage | Input | Output |
|-------|-------|--------|
| Header | `<obs>.uvfits` | `pyfhd_header`, `params_data`, `antenna_data`, `antenna_header` |
| Params | `pyfhd_header`, `params_data` | `params` (uu/vv/ww/time/baseline) |
| Visibilities | `pyfhd_header`, `params_data` | `vis_arr`, `vis_weights` |
| Layout | `antenna_header`, `antenna_data` | `layout` |
| Obs | header + params + layout + cfg | `obs` |
| PSF / Beam | `obs`, cfg | `psf` (dict or h5py.File) |
| Flag basic | weights + cfg | `vis_weights` updated, `obs` updated |
| Model | cfg, obs, params | `vis_model_arr` |
| Calibrate | obs, params, vis, weights, model | `vis_arr`, `cal`, obs, cfg |
| Grid (per pol) | obs, psf, params, vis, weights, model | `image_uv`, `weights_uv`, `variance_uv`, `uniform_filter_uv`, `model_uv`, `obs.nf_vis` |
| Quickview | gridded planes + cal + vis | per-pol FITS, PNGs |
| HEALPix | obs, psf, cal, params, vis, model, weights | `<obs>_<cube>_<pol>.h5` |

### Frequently consulted defaults

| Option | Default | Source |
|--------|---------|--------|
| `dimension`, `elements` | 2048 | `pyfhd.yaml` |
| `kbinsize` | 0.5 | |
| `n_pol` | 2 | |
| `psf_dim` | 54 | |
| `psf_resolution` | 100 | |
| `beam_nfreq_avg` | 16 | |
| `beam_offset_time` | 56 (s in obs) | |
| `cal_convergence_threshold` | 1e-7 | |
| `max_cal_iter` | 100 | |
| `min_cal_baseline` | 50 λ | |
| `cal_amp_degree_fit` | 2 | |
| `cal_phase_degree_fit` | 1 | |
| `cal_phase_fit_iter` | 4 | |
| `cal_reflection_mode_theory` | 150 (m cable) | |
| `n_avg` (HEALPix) | 2 | |
| `ps_kbinsize` | 0.5 | |

---

*End of `simulators/PyFHD.md` — generated entirely from in-tree sources at
`simulators/PyFHD/...`.*
