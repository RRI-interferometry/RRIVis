# CARACal — Containerized Automated Radio Astronomy Calibration

> Exhaustive technical reference for the `caracal` git submodule located at
> `simulators/caracal/`.  Every concrete claim below is grounded in a source
> file in that submodule; file paths are quoted relative to `simulators/caracal/`.

---

## 1. Overview

**CARACal** (*Containerized Automated Radio Astronomy Calibration*) is a Python
pipeline for end-to-end reduction of radio interferometry data delivered in the
CASA Measurement Set (MS) format.  It is **not** a visibility *simulator* — it
is a calibration / flagging / imaging *orchestrator* that drives a heterogeneous
collection of third-party radio-astronomy tools (CASA, AOFlagger, Tricolour,
WSClean, CubiCal, MeqTrees, DDFacet, SoFiA, PyBDSF, Owlcat/Tigger, shadeMS,
ragavi, …) through Docker / Podman / Singularity containers using
[**Stimela**](https://github.com/ratt-ru/Stimela) as the runtime.

It is included in the RRIVis project's `simulators/` folder as a *reference
implementation* of a real-world MS-driven pipeline that can ingest the MS
output produced by RRIVis and turn it into images, cubes, calibration tables,
and diagnostic notebooks.

| Attribute | Value | Source |
|-----------|-------|--------|
| Package name | `caracal` | `pyproject.toml:2` |
| Version | `1.2.1rc1` | `pyproject.toml:3` |
| Description | "Containerized Automated Radio Astronomy Calibration" | `pyproject.toml:4` |
| Author | The CaraCAL Team (`caracal-info@googlegroups.com`) | `pyproject.toml:6` |
| License | GPL-2.0 | `pyproject.toml:10`, `LICENSE` |
| Schema version | `1.2.0` | `caracal/schema/__init__.py:1` |
| Python | `>=3.9, <3.13` | `pyproject.toml:8` |
| Build backend | `hatchling` | `pyproject.toml:71-73` |
| Console script | `caracal = caracal.main:driver` | `pyproject.toml:46` |
| Latest tags | `1.2.1rc1`, `v.1.2.0`, `v1.1.4`, `v1.1.3`, `v1.0.7`, … | `git tag` |
| Project home | `https://github.com/caracal-pipeline/caracal` | `pyproject.toml:42` |
| Documentation | `https://caracal.readthedocs.io` | `README.rst:13` |

The caracal team's *Carate-Kid*-flavoured CI/install helper is the bash script
`caratekit.sh` at the repo root (3853 lines), which automates virtualenv
creation, container backend selection, and end-to-end test runs (see §4.4).

---

## 2. Repository layout

The submodule contains 205 tracked files (excluding `.git`).  The top-level
layout is:

```
simulators/caracal/
├── README.rst                      # User-facing readme (install/run)
├── LICENSE                         # GPL v2
├── MANIFEST.in                     # Source-distribution manifest
├── pyproject.toml                  # Hatch project metadata + deps
├── ruff.toml                       # Ruff linter config (line=180, py310)
├── .pre-commit-config.yaml         # ruff-check + ruff-format hooks
├── .readthedocs.yml                # RTD build pipeline
├── .gitignore
├── .gitmodules                     # (empty — no submodules of its own)
├── stimela-master.txt              # `-e git+…/Stimela` pin for dev installs
├── caratekit.sh                    # Mr.Miyagi-themed install/test helper (3853 lines)
├── Jenkinsfile.sh                  # CI driver that calls caratekit.sh
├── .github/workflows/
│   ├── python-ci.yml               # ruff + pytest matrix py3.9–3.13
│   └── publish-package.yml         # Poetry build → PyPI on release
├── caracal/                        # The Python package
│   ├── __init__.py
│   ├── main.py                     # `caracal` CLI entry point
│   ├── exceptions.py
│   ├── tests/__init__.py           # (empty package marker, kept for path)
│   ├── utils/
│   │   ├── __init__.py             # YAML helpers
│   │   └── requires.py             # Optional-dependency decorator
│   ├── schema/                     # pykwalify schemas (one per worker)
│   │   ├── __init__.py             # SCHEMA_VERSION = "1.2.0"
│   │   ├── general_schema.yml
│   │   ├── getdata_schema.yml
│   │   ├── obsconf_schema.yml
│   │   ├── prep_schema.yml
│   │   ├── transform_schema.yml
│   │   ├── flag_schema.yml
│   │   ├── crosscal_schema.yml
│   │   ├── polcal_schema.yml
│   │   ├── inspect_schema.yml
│   │   ├── mask_schema.yml
│   │   ├── selfcal_schema.yml
│   │   ├── ddcal_schema.yml
│   │   ├── line_schema.yml
│   │   ├── polimg_schema.yml
│   │   └── mosaic_schema.yml
│   ├── sample_configurations/      # Ready-to-edit YAML templates
│   │   ├── minimalConfig.yml
│   │   ├── minitestConfig.yml
│   │   ├── meerkat-defaults.yml
│   │   ├── meerkat-continuum-defaults.yml
│   │   ├── meerkat-fullStokes-continuum-defaults.yml
│   │   ├── meerkat-polcal-strategies.yml
│   │   ├── carateConfig.yml
│   │   └── mosaic_basic_config.yml
│   ├── dispatch_crew/              # Pipeline-internal helpers
│   │   ├── __init__.py
│   │   ├── config_parser.py        # YAML→argparse↔schema validator
│   │   ├── worker_help.py          # `-wh <worker>` printer
│   │   ├── caltables.py            # Lazy-loaded calibrator DBs
│   │   ├── catalog_parser.py       # Custom calibrator-DB parser
│   │   ├── utils.py                # Field categorisation, geometry, …
│   │   ├── stream_director.py      # stdout/stderr → logger redirection
│   │   ├── interruptable_process.py# SIGINT-able multiprocessing.Process
│   │   └── noisy.py                # Theoretical noise estimator
│   ├── workers/                    # 14 stage modules + worker_administrator
│   │   ├── __init__.py
│   │   ├── worker_administrator.py # Pipeline driver class
│   │   ├── getdata_worker.py
│   │   ├── obsconf_worker.py
│   │   ├── prep_worker.py
│   │   ├── transform_worker.py
│   │   ├── flag_worker.py
│   │   ├── crosscal_worker.py
│   │   ├── polcal_worker.py
│   │   ├── inspect_worker.py
│   │   ├── mask_worker.py
│   │   ├── selfcal_worker.py
│   │   ├── ddcal_worker.py
│   │   ├── line_worker.py
│   │   ├── polimg_worker.py
│   │   ├── mosaic_worker.py
│   │   └── utils/                  # Worker-shared utilities
│   │       ├── __init__.py         # remove_output_products()
│   │       ├── callibs.py          # CASA callib-file generator
│   │       ├── manage_antennas.py  # Auto refant selection
│   │       ├── manage_fields.py    # Field-name → field-id mapping
│   │       ├── manage_flagsets.py  # Flag-version save/restore
│   │       ├── manage_caltabs.py   # (currently empty)
│   │       ├── flag_Uzeros.py      # UzeroFlagger class
│   │       └── image_contsub.py    # FITS-cube continuum subtraction
│   ├── notebooks/                  # Jinja2 templates for radiopadre reports
│   │   ├── __init__.py             # setup_default_notebooks / generate_report_notebooks
│   │   ├── std-progress-report.ipynb        + .j2 template
│   │   ├── detailed-final-report.ipynb      + .j2 template
│   │   ├── project-logs.ipynb               + .j2 template
│   │   ├── project-directory.ipynb
│   │   ├── header.j2
│   │   └── caracal-{logo,square-logo}-*.png
│   └── data/                       # Static data shipped with the pkg
│       ├── southern_calibrators.txt        # Custom Southern-calibrator DB
│       ├── casa_calibrators.txt            # Northern calibrator coords
│       ├── casa_calibrators.yml            # CASA standards mapping
│       ├── nrao_xcal.yml                   # NRAO polarisation reference flux
│       ├── taylor_legodi_2024.txt          # Taylor & Legodi 2024 pol-cal table
│       ├── meerkat_coeff_dict.npy          # Beam coeffs (legacy)
│       └── meerkat_files/                  # Copied to <input>/ at startup
│           ├── *.rfis                      # AOFlagger strategies
│           ├── *.yaml                      # Tricolour strategies
│           ├── meerkat.rfimask.npy
│           ├── meerkat_beam_coeffs_{ah,em}_zp_dct.npy
│           ├── pks{1934-638,0407-65}.lsm
│           ├── 0407-collapsed-uhf-cat.txt
│           ├── 1934-collapsed-uhf-cat.txt
│           ├── hicat_caracal.txt           # HI catalogue
│           ├── mk64.txt                    # MeerKAT 64-antenna list
│           └── fields/J*.FITS, Fornaxa_vla.FITS  # Per-field clean masks
├── tests/                          # Pytest suite
│   ├── __init__.py                 # InitTest fixture / TESTDIR
│   ├── test_runner.py              # End-to-end CLI smoke tests
│   ├── test_dispatch_crew_utils.py # Field/observation-length helpers
│   └── obsinfo/ms_summary.json     # Test fixture: a real MS summary
└── docs/
    ├── README.md                   # How to rebuild docs
    ├── make_caracal_docs.py        # Schema → rST generator
    └── sphinx/
        ├── conf.py                 # Sphinx (sphinx_rtd_theme, recommonmark)
        ├── Makefile
        ├── requirements.txt
        ├── caracalREADME.rst       # auto-generated from README.rst
        ├── caracal_logo.png
        ├── index.rst
        ├── manual/                 # Hand-written pages + auto-generated
        │   ├── index.rst
        │   ├── intro/
        │   ├── configfile/
        │   ├── products/
        │   ├── reduction/{workflow,prepare,flag,crosscal,selfcal,line,mosaic}/
        │   ├── workers/{general,getdata,obsconf,prep,transform,flag,crosscal,
        │   │            polcal,polimg,inspect,mask,selfcal,ddcal,line,mosaic}/
        │   ├── packages/
        │   └── caratekit_utility/
        └── credits/{team,credits,crediting,caracal_logos}/
```

### Per-folder commentary

| Folder | Role |
|--------|------|
| `caracal/` | The installable Python package. Contains *no* extension modules; pure Python. |
| `caracal/schema/` | One pykwalify schema per worker. The schema is the *single source of truth* for option names, types, defaults, enum values, and help text — argparse options and the readthedocs HTML pages are both generated from it. |
| `caracal/sample_configurations/` | YAML templates dumped via `caracal -gdt <template> -gd outfile.yml`. |
| `caracal/dispatch_crew/` | "Crew" that dispatches jobs/configurations.  Houses the YAML→argparse bridge, the field auto-categoriser, and the calibrator catalogue parser. |
| `caracal/workers/` | One Python module per pipeline stage. Each defines `NAME`, `LABEL`, optional `check_config()`, and the mandatory `worker(pipeline, recipe, config)` entry point that appends Stimela cabs to the recipe. |
| `caracal/workers/utils/` | Shared low-level utilities (flag-version manipulation, callib generation, automatic refant selection, the standalone `image_contsub.py` script, the `UzeroFlagger` class). |
| `caracal/notebooks/` | Jinja2 templates rendered into ipynb that radiopadre then converts to HTML reports. |
| `caracal/data/` | Static reference data: calibrator catalogues, MeerKAT beam coefficients, AOFlagger/Tricolour strategy files, FITS clean masks for known southern fields. |
| `tests/` | Pytest suite. Modest in scope — see §11. |
| `docs/` | Sphinx documentation. The `make_caracal_docs.py` step ingests the YAML schemas to auto-generate per-worker rST. |
| `caratekit.sh` | An opinionated install/test wrapper described in §4.4. |

---

## 3. Languages, build system, dependencies

### 3.1 Language inventory

CARACal is **pure Python** (3.9–3.12); shell (`caratekit.sh`, `Jenkinsfile.sh`)
is auxiliary tooling.  There are no compiled extensions.  YAML drives both the
schemas and the user-facing configuration.

### 3.2 Build / packaging

`pyproject.toml` declares **Hatch** as the build backend:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.sdist]
include = ["caracal"]
[tool.hatch.build.targets.wheel]
include = ["caracal"]
```

`MANIFEST.in` keeps non-Python assets (YAML schemas, `.rfis`, `.npy`, FITS, lsm,
ipynb templates, PNG logos) in the source distribution.

`pyproject.toml` also exposes the console script:

```toml
[project.scripts]
caracal = "caracal.main:driver"
```

A *legacy* Poetry workflow is still mentioned in `README.rst:129–143` for
developers, and the GitHub release workflow `.github/workflows/publish-package.yml`
uses Poetry 2.1.4 to build sdists/wheels and upload to PyPI.

### 3.3 Runtime dependencies (`pyproject.toml:21-39`)

| Dependency | Constraint | Role inside CARACal |
|------------|-----------|---------------------|
| `stimela` | `>=1.8.1, <2` | Container-aware recipe runner; provides `Recipe`, `dismissable`, `xrun`, cab logger. |
| `psutil` | `>=5.9.4, <6` | Memory/CPU heuristics in line/selfcal workers. |
| `pykwalify` | `>=1.8.0, <2` | YAML schema validation in `dispatch_crew/config_parser.py`. |
| `progressbar2` | `>=4.2, <5` | (legacy) progress bars in long-running steps. |
| `ruamel.yaml` + `ruamel.yaml.string` | `>=0.17`/`>=0.1` | Round-tripping YAML configs. |
| `astropy` | `==5.3.3` (py3.9-10) / `>=7.0` (py3.11-12) | FITS, WCS, SkyCoord, table I/O across many workers. |
| `scipy` | `>=1.10, <2` | Curve fitting in `catalog_parser.convert_pb_to_casaspi`, optimisation in `flag_Uzeros`. |
| `regions` | `>=0.7, <1` | Polygon regions for `ddcal_worker` (DE-source masks). |
| `astroquery` | `>=0.4, <1` | NVSS/SUMSS catalogue queries in `mask_worker`. |
| `numpy` | `>=1.23, <3` | Pervasive. |
| `nbconvert` | `>=6, <8` | Render reports. |
| `radiopadre-client` | `>=1, <2` | Renders ipynb→HTML reports inside containers. |
| `python-casacore` | `>=3, <4` | Direct MS table access (`casacore.tables.table`). |
| `matplotlib` | `>=3.7, <4` | Diagnostic plots, beam plotting. |
| `jinja2` | `>=3.1, <4` | Notebook template rendering. |

### 3.4 Optional groups (`pyproject.toml:48-63`)

```toml
[dependency-groups]
dev   = ["jupyter", "ruff>=0.12.9", "pre-commit>=4.3.0"]
tests = ["pytest>=7.1.3,<8", "flake8>=5.0.0,<6", "ruff>=0.12.9"]
docs  = ["Sphinx>=4.0.1,<5", "sphinx-copybutton>=0.5.0,<0.6", "furo>=2022.9.15,<2023"]
```

The dev install path can also pull Stimela from master via `stimela-master.txt`:

```
-e git+https://github.com/ratt-ru/Stimela#egg=stimela
```

### 3.5 External (non-Python) dependencies

CARACal does **not** install any radio-astronomy software itself.  Instead it
pulls Stimela cab images, each wrapping a third-party tool.  The grep-derived
table below enumerates every cab referenced by `recipe.add(...)` calls in the
worker modules:

| Stimela cab | Underlying tool | Used by (workers) |
|-------------|-----------------|-------------------|
| `cab/casa_listobs` | CASA `listobs` | obsconf, transform |
| `cab/casa_clearcal` | CASA `clearcal` | prep |
| `cab/casa_setjy` | CASA `setjy` | crosscal |
| `cab/casa_gaincal` | CASA `gaincal` | crosscal, polcal |
| `cab/casa_bandpass` | CASA `bandpass` | crosscal |
| `cab/casa_polcal` | CASA `polcal` | polcal |
| `cab/casa_polfromgain` | CASA `polfromgain` | polcal |
| `cab/casa_fluxscale` | CASA `fluxscale` | crosscal |
| `cab/casa_applycal` | CASA `applycal` | transform, polcal |
| `cab/casa_flagdata` | CASA `flagdata` | flag, crosscal, polcal |
| `cab/casa_flagmanager` | CASA `flagmanager` | manage_flagsets util |
| `cab/casa_fixvis` | CASA `fixvis` | prep, transform |
| `cab/casa_mstransform` | CASA `mstransform` | transform, line |
| `cab/casa_concat` | CASA `concat` | transform |
| `cab/casa_clean` | CASA `tclean` | line (legacy) |
| `cab/casa_imregrid` | CASA `imregrid` | mosaic |
| `cab/casa_importfits` / `cab/casa_exportfits` | CASA fits I/O | mask, line |
| `cab/casa_plotms` | CASA `plotms` | obsconf, inspect |
| `cab/casa_fringefit` | CASA `fringefit` | crosscal (S-step) |
| `cab/casa_script` | Run arbitrary CASA python | prep |
| `cab/msutils` | RATT `msutils` (summary, weights, columns) | obsconf, prep, transform |
| `cab/pycasacore` | python-casacore + Owlcat | manage_flagsets |
| `cab/sunblocker` | `sunblocker` | obsconf (vampirisms) |
| `cab/owlcat_plotelev` | Owlcat | obsconf |
| `cab/wsclean` | WSClean | crosscal (`I` step), selfcal, line, polimg |
| `cab/aimfast` | aimfast | selfcal (image QA) |
| `cab/breizorro` | breizorro mask maker | selfcal, mask |
| `cab/cleanmask` | cleanmask | mask, ddcal |
| `cab/sofia` | SoFiA-2 | mask, line |
| `cab/pybdsm` | PyBDSF | selfcal, mask |
| `cab/cubical` / `cab/cubical_ddf` / `cab/cubical_pgs` | CubiCal | selfcal, ddcal |
| `cab/calibrator` (MeqTrees) | MeqTrees | selfcal |
| `cab/crystalball` | crystalball predictor | crosscal |
| `cab/ddfacet` | DDFacet | ddcal |
| `cab/catdagger` | catdagger | ddcal |
| `cab/eidos` | EIDOS beam model | line, polimg |
| `cab/spimple_imconv` | spimple | mosaic |
| `cab/montage` | Montage astronomical mosaicker | mosaic |
| `cab/fitstool` | fitstool | line |
| `cab/imcontsub` | image-plane contsub | line |
| `cab/sharpener` | sharpener | line |
| `cab/rfimasker` | RFI masker | flag |
| `cab/autoflagger` | AOFlagger | flag |
| `cab/tricolour` | Tricolour | flag |
| `cab/politsiyakat_autocorr_amp` | politsiyakat | flag |
| `cab/flagstats` | flagstats | manage_antennas util |
| `cab/flagms` | Owlcat flag-ms | manage_flagsets util |
| `cab/rfinder` | rfinder | flag |
| `cab/tigger_convert` / `cab/tigger_restore` | Tigger | crosscal, polcal |
| `cab/simulator` | MeqTrees simulator | crosscal (model fill) |
| `cab/shadems` / `cab/shadems_direct` | shadems | inspect |
| `cab/ragavi` / `cab/ragavi_vis` | ragavi | inspect, crosscal (gain plots) |

A user supplies the **container backend** (`docker`, `podman`, `singularity`,
`udocker`) via `general.backend` in YAML or `-ct/--container-tech` on the CLI;
Stimela transparently materialises the cab images.

---

## 4. Installation

`README.rst:35-174` documents three installation paths.  All assume a Python
3.9–3.12 virtualenv with up-to-date `pip setuptools wheel`.

### 4.1 Manual (PyPI)

```bash
python3 -m venv "${caracal-venv}"
source "${caracal-venv}/bin/activate"
pip install -U pip setuptools wheel
pip install -U caracal              # stable release
# or, for the bleeding-edge master:
pip install -U 'caracal @ git+https://github.com/caracal-pipeline/caracal.git@master'
```

After install, pull cab images:

```bash
stimela clean -ac      # purge any older stimela images
stimela pull           # docker
stimela pull -s        # singularity (requires SINGULARITY_PULLFOLDER)
stimela pull -p        # podman
```

### 4.2 `caratekit.sh`

Single-shot installer + tester (`caratekit.sh`, 3853 lines, ~146 KB).  Examples
from `README.rst:115-127`:

```bash
caratekit.sh -ws ${workspace} -cr -di -ct ${caracal_dir} -rp install -f -kh   # docker
caratekit.sh -ws ${workspace} -cr -si -ct ${caracal_testdir} -rp install -f -kh # singularity
```

It implements its own argument loop using `[[ "$arg" == "--long" ]] || [[ "$arg" == "-short" ]]`
checks (40+ flags; selected ones below — see `caratekit.sh:87-300`):

| Short | Long | Effect |
|-------|------|--------|
| `-h` | `--help` | Help |
| `-v` | `--verbose-help` | Verbose help |
| `-i` | `--install` | Install caracal |
| `-ho` / `-hw` / `-hn` / `-hf` | home-original/workspace/no-delete/folder | HOME handling |
| `-dm` / `-da` / `-di` | docker-{minimal,alternative,installation} | docker modes |
| `-sm` / `-sa` / `-si` | singularity-{minimal,alternative,installation} | singularity modes |
| `-scw` / `-slw` / `-stw` / `-spw` | singularity cache-/localcache-/tmp-/pull-workspace | cache layout |
| `-ws` | `--workspace` | Workspace dir |
| `-ct` | (carate test dir) | Test directory name |
| `-rp` | (run profile) | install/test profile |
| `-f` | `--force` | Force rerun |
| `-us` / `-um` | use-stimela-stable / use-stimela-master | Stimela version pin |
| `-op` | `--omit-docker-prune` | Skip prune |

The Mr-Miyagi quotes (`kkfailquotes`/`kksuccessquotes` arrays at `caratekit.sh:18-43`)
are emitted on success/failure for thematic value.

### 4.3 Poetry (developer)

```bash
pip install poetry
poetry install         # with optional dependency groups available too
```

### 4.4 ILIFU / SLURM cluster

`README.rst:145-174` describes installing on the IDIA ILIFU shared cluster:

```bash
module add python/3.9.4
python3 -m venv <venv>
source <venv>/bin/activate
pip install -U pip setuptools wheel
git clone https://github.com/caracal-pipeline/caracal.git
pip install -U -e caracal
```

Pre-pulled Singularity images live at
`/idia/software/containers/STIMELA_IMAGES/` (legacy at
`/idia/software/containers/STIMELA_IMAGES_legacy/`).

---

## 5. Top-level architecture

```
                           ┌────────────────────────────────────┐
   user-edited YAML ──►    │  caracal.main.main(argv)           │  CLI
                           │   ├─ basic_parser  (argparse)      │  argparse setup
                           │   ├─ config_parser.validate_config │  pykwalify
                           │   └─ populate_parser  (per-key)    │  YAML→argparse
                           └────────────────────────────────────┘
                                          │
                                          ▼
                           ┌────────────────────────────────────┐
                           │  WorkerAdministrator (pipeline)    │  workers/worker_administrator.py
                           │   ├─ init_pipeline (mkdirs, sym-   │
                           │   │  links, copy MeerKAT inputs,   │
                           │   │  install notebooks)            │
                           │   ├─ for each worker:              │
                           │   │   ▸ import caracal.workers.X   │
                           │   │   ▸ stimela.Recipe(...)        │
                           │   │   ▸ X.worker(self, recipe, cfg)│  ← per-stage
                           │   │   ▸ recipe.run()               │
                           │   │   ▸ optional notebook render   │
                           │   └─ regenerate_reports()          │
                           └────────────────────────────────────┘
                                          │
                                          ▼
                           ┌────────────────────────────────────┐
                           │  Stimela recipe (one per worker)   │
                           │   each recipe.add("cab/...", ...)  │
                           │   schedules a containerised job    │
                           └────────────────────────────────────┘
                                          │
                                          ▼
                ┌──────────┬──────────┬──────────┬──────────┬─────────┐
                ▼          ▼          ▼          ▼          ▼         ▼
              CASA     WSClean   AOFlagger   CubiCal     SoFiA    radiopadre
             cabs       cab       cab         cab        cab        cab
                (Docker / Podman / Singularity / udocker images)
```

### 5.1 Why "containerised"?

CARACal never imports CASA / WSClean / AOFlagger / CubiCal / etc.  Each Stimela
cab is a Docker (or equivalent) image whose entry point is the third-party
tool with a YAML-mapped argument schema.  This decouples CARACal's Python
versioning from the wildly heterogeneous Python/C++ world of radio-astronomy
tools.

### 5.2 Module-import map

```
caracal.__init__  ── set up logger, version, DEFAULT_CONFIG, SAMPLE_CONFIGS
   ├── exceptions      Custom exception types
   ├── utils.__init__  YAML helpers (load_yaml/write_yaml/to_regular_dict)
   ├── schema          (data only; SCHEMA_VERSION)
   ├── notebooks       Jinja2 templating + radiopadre invocation
   ├── dispatch_crew
   │    ├── config_parser  ◄─ basic_parser, validate_config, populate_parser
   │    ├── worker_help    ◄─ -wh subprinter
   │    ├── caltables      ◄─ lazy-loaded calibrator DB
   │    ├── catalog_parser ◄─ Perley-Butler ↔ CASA SPI converter
   │    ├── utils          ◄─ Fields, geometry, find_in_*_calibrators, …
   │    ├── stream_director, interruptable_process, noisy
   │    └── …
   └── workers
        ├── worker_administrator  ◄─ Pipeline driver class (the heart)
        ├── (14 worker modules, dynamically imported by workers_directory path)
        └── utils                 ◄─ shared helpers, `manage_flagsets`, `callibs`, …
```

Workers are imported via `__import__(_worker)` after the *workers directory*
has been appended to `sys.path` — see `worker_administrator.py:73,117,380,399`.
This is intentional: it lets the pipeline "discover" workers by filename without
requiring them to be subpackages.

---

## 6. CLI reference

The console script is `caracal` (mapped to `caracal.main.driver`).  Argument
parsing happens twice — first the *basic* parser (always-available global
switches), then a *populated* parser whose `--worker-option` arguments come
from the active config's schema.

### 6.1 Basic options (`dispatch_crew/config_parser.py:28-104`)

| Short | Long | Type | Default | Purpose |
|-------|------|------|---------|---------|
| `-v` | `--version` | flag | – | Print `caracal X.Y.Z`. |
| `-c` | `--config FILE` | path | `DEFAULT_CONFIG` | Pipeline configuration YAML. **Mandatory in practice**. |
| `-b` | `--boring` | flag | False | Disable colour output. |
| `-sid` | `--singularity-image-dir DIR` | str | – | Where Stimela should look for/store Singularity SIFs. |
| `-gdt` | `--get-default-template NAME` | choice | `minimal` | Template to seed with `-gd`. Choices come from `caracal.SAMPLE_CONFIGS`: `minimal`, `meerkat`, `carate`, `meerkat_continuum`, `mosaic_basic`. |
| `-gd` | `--get-default FILE` | str | – | Save a config copy to this file. |
| `-sw` | `--start-worker NAME` | str | – | Begin pipeline at this worker. |
| `-ew` | `--end-worker NAME` | str | – | Stop after this worker. |
| `-ct` | `--container-tech` | choice | `default` | One of `default`, `docker`, `udocker`, `singularity`, `podman`. |
| `-wh` | `--worker-help WORKER` | str | – | Print all schema options for one worker, then exit. |
| `-pcs` | `--print-calibrator-standard` | flag | False | Dump the southern calibrator DB and exit. |
|  | `-report` | flag | False | (Re)generate HTML reports and exit (no pipeline run). |
|  | `-debug` | flag | False | Drop into pdb on any unhandled exception. |
| `-nr` | `--no-reports` | flag | False | Suppress all radiopadre reports. |

### 6.2 Populated options

For every leaf-level key in the active configuration's schema CARACal adds an
argparse option of the form `--<worker>-<group>-<key>`.  These are
`argparse.SUPPRESS`-help-text only (they shadow the YAML); they exist so users
can override single fields from the command line, e.g.

```bash
caracal -c myconfig.yml --selfcal-img_npix 4096 --selfcal-cal_niter 5
```

`update_config_from_args()` (`config_parser.py:193-203`) types-cast back from
strings — booleans accept `true|yes|1|false|no|0`, lists/dicts are parsed via
ruamel YAML.

### 6.3 Common end-to-end examples

```bash
# Initialise a config from a template
caracal -gdt meerkat -gd meerkat-config.yml

# Print the schema-driven help for a single worker
caracal -wh selfcal

# Run pipeline from start to end, Singularity backend
caracal -ct singularity -c meerkat-config.yml

# Run pipeline, but only the flag and crosscal stages
caracal -c myconfig.yml -sw flag -ew crosscal

# Re-render the HTML reports without re-running the pipeline
caracal -c myconfig.yml -report

# Print the southern calibrator catalogue
caracal -pcs
```

### 6.4 Driver lifecycle (`caracal/main.py:169-262`)

```
driver()
  └── main(sys.argv[1:])
        ├── basic_parser → parse_known_args → options
        ├── init_console_logging(boring, debug)
        ├── if -wh: print_worker_help(worker); return
        ├── if -gd: validate sample config, copy to FILE; return
        ├── if -pcs: dump calibrator_database(); return
        ├── (else) load YAML, validate, populate parser, reparse args
        ├── log_logo()  ◄─ ASCII-art banner with version
        └── execute_pipeline(options, config)
              └── WorkerAdministrator(...).run_workers() | regenerate_reports()
```

Errors are caught in three classes (line 137-159):
* `SystemExit` from a worker → log error, optional pdb, exit 1.
* `KeyboardInterrupt` → graceful shutdown.
* anything else → log full traceback, optional pdb, exit 1.

---

## 7. Configuration schema

### 7.1 Top-level structure

A configuration is a single YAML file whose top-level keys are *worker names*
(plus the special `schema_version`).  Each worker section is validated against
`caracal/schema/<worker>_schema.yml` using **pykwalify**.  Workers may be
re-instantiated by suffixing `__N`, e.g. `flag__2:` re-uses the `flag` schema
(see `dispatch_crew/config_parser.py:147-168`):

```python
_worker = worker.split("__")[0]
schema_fn = os.path.join(caracal.pckgdir, "schema", f"{_worker}_schema.yml")
```

### 7.2 The mandatory three workers

`worker_administrator.py:75-78` hard-codes that the first three workers are
mandatory:

```python
last_mandatory = 2          # general, getdata, obsconf
start_idx = last_mandatory
```

Even if a user requests `--start-worker selfcal`, `general`, `getdata`, and
`obsconf` always run.

### 7.3 Worker schema reference

The table below lists every schema, its top-level group keys, and the worker
that consumes it.  The "Lines" column hints at the configurability surface.

| Worker (label) | Schema file | Top-level keys (highlights) | Lines |
|----------------|-------------|------------------------------|------:|
| `general` | `general_schema.yml` | `title`, `msdir`, `rawdatadir`, `input`, `output`, `prefix`, `prep_workspace`, `init_notebooks`, `report_notebooks`, `final_report`, `backend` ∈ {docker,udocker,singularity,podman}, `cabs` (cab override list) | 83 |
| `getdata` | `getdata_schema.yml` | `dataid` (seq str), `extension`, `untar`, `report`, `ignore_missing`, `cabs` | 62 |
| `obsconf` | `obsconf_schema.yml` | `obsinfo.{listobs,summary_json,vampirisms,plotelev}`, `target/gcal/bpcal/fcal/xcal` (seq + 'all'/'longest'/'nearest'), `refant` ('auto'/name/index), `maxdist`, `minbase` | 114 |
| `prep` | `prep_schema.yml` | `label_in/out`, `field`, `tol`, `tol_diff`, `fixuvw`, `fixcalcoords`, `clearcal.addmodel`, `manage_flags.{mode,version}`, `specweights.mode` ∈ {uniform,calculate,delete} | 150 |
| `transform` | `transform_schema.yml` | `field`, `label_in/out`, `rewind_flags`, `split_field` (mstransform wrapper, with OTF callib + interpolation overrides), `changecentre`, `concat`, `obsinfo` | 232 |
| `flag` | `flag_schema.yml` | huge — `unflag`, `flag_autopowerspec`, `flag_autocorr`, `flag_quack`, `flag_elevation`, `flag_shadow`, `flag_spw`, `flag_time`, `flag_scan`, `flag_antennas`, `flag_mask`, `flag_manual`, `flag_rfi.flagger` ∈ {aoflagger,tricolour,rfimasker}, `summary` | 450 |
| `crosscal` | `crosscal_schema.yml` | `set_model.{meerkat_band,meerkat_skymodel,meerkat_crystalball_*,unity,field,...}`, `primary.{order,combine,solint,calmode,b_solnorm,b_fillgaps,plotgains,reuse_existing_gains}`, `secondary.{order,apply,...}`, `apply_cal.applyto` | 469 |
| `polcal` | `polcal_schema.yml` | `pol_calib`, `leakage_calib`, `feed_angle_rotation`, `freqsel`, `gain_solint`, `time_solint`, `extendflags`, `otfcal` | 236 |
| `inspect` | `inspect_schema.yml` | `shadems.{plots,plots_by_field,plots_by_corr}`, `standard_plotter`, `real_imag/amp_*/phase_*` (one block per plot type) | 324 |
| `mask` | `mask_schema.yml` | `centre_coord`, `mask_size`, `cell_size`, `extended_source_map`, `catalog_query.catalog` ∈ {NVSS,SUMSS}, `pbcorr`, `make_mask.mask_method` ∈ {thresh,sofia}, `merge_with_extended` | 166 |
| `selfcal` | `selfcal_schema.yml` | imaging (`img_*`), calibration loop (`cal_niter`, `gsols_*`, `bsols_*`, `gain_matrix_type`), `cal_cubical`, `cal_meqtrees`, `image.{cleanmask_thr,clean_cutoff,…}`, `transfer_apply_gains`, `transfer_model`, `aimfast`, `quality_check`, `restart_no_resume` | 788 |
| `ddcal` | `ddcal_schema.yml` | `image_dd.*`, `calibrate_dd.{de_sources_mode,de_target_manual,de_sources_manual,dd_dd_*}`, `use_pb`, `shared_mem` | 552 |
| `line` | `line_schema.yml` | `restfreq`, `subtractmodelcol`, `addmodelcol`, `mstransform.{doppler,uvlin}`, `make_cube`, `pb_cube`, `freq_to_vel`, `flag_mst_errors`, `remove_stokes_axis`, `sofia` | 1016 |
| `polimg` | `polimg_schema.yml` | `stokes` ∈ {QU,QUV,IQU,IQUV,I,Q,U,V}, `make_images`, `rmsynth` | 291 |
| `mosaic` | `mosaic_schema.yml` | `mosaic_type` ∈ {continuum,line}, `use_mfs`, `target_images`, `pb_type` ∈ {gaussian,mauchian}, `dish_diameter`, `ref_frequency`, `beam_cutoff`, `mosaic_cutoff` | 109 |

Each schema also carries a `cabs:` block at the bottom letting users override
container image versions/tags per-worker.  `WorkerAdministrator.parse_cabspec_dict`
(`worker_administrator.py:267-294`) merges these into a single Stimela
`cabspecs` dict, with "force tag for all invocations" semantics.

### 7.4 The schema → argparse mapping

`config_parser._process_subparser_tree()` walks the loaded schema dictionary
recursively, treating any key with `mapping:` as a sub-section and any leaf as
an option.  The leaf type is read from `subVars["type"]` (or
`subVars["seq"][0]["type"]` for sequences).  Defaults come from the YAML
file *if present*, else from `subVars["example"]`.  Booleans get a fixed enum
of `true yes 1 false no 0` (line 343-347).  Lists and dicts are passed as YAML
strings and round-tripped via `ruamel.yaml` (line 333-339).

### 7.5 Sample configurations (`caracal/sample_configurations/`)

| Template | Lines | Purpose | Highlights |
|----------|------:|---------|-----------|
| `minimalConfig.yml` | 103 | Smallest end-to-end recipe (transform → prep → flag → crosscal → inspect → transform__2 → prep__2 → flag__2 → mask → selfcal → line). | Shows the `__N` re-instantiation idiom. |
| `minitestConfig.yml` | 126 | CI/test variant with smaller fields/freqs. | Reduced cube size. |
| `meerkat-defaults.yml` | 245 | MeerKAT continuum + line baseline. | RFI strategies (`firstpass_QUV.rfis`), pre-set MeerKAT calibration order `KGBAKGB`. |
| `meerkat-continuum-defaults.yml` | 245 | MeerKAT continuum-only. | No `line:` block. |
| `meerkat-fullStokes-continuum-defaults.yml` | 300 | Full-Stokes continuum (incl. polcal/polimg). | Shows polcal `feed_angle_rotation: -90`. |
| `meerkat-polcal-strategies.yml` | 138 | Different polarisation calibration strategies. | xcal, leakage_cal variants. |
| `carateConfig.yml` | 293 | Most exhaustive `inspect.shadems` plot list. | Demonstrates multi-iteration crosscal `KGBAKGBK`. |
| `mosaic_basic_config.yml` | 16 | Mosaic-only invocation. | `mosaic.mosaic_type: spectral` (i.e. line cube). |

The minimum mandatory blocks are `general` (with `prefix`), `getdata` (with
`dataid`), and `obsconf` (with `refant`).

---

## 8. The pipeline driver in detail (`workers/worker_administrator.py`)

`WorkerAdministrator.__init__()` (lines 19-139) does the following:

1. **Path setup** — store `msdir`, `input`, `output`, plus derived directories
   (`obsinfo`, `reports`, `diagnostic_plots`, `cfgFiles`, `caltables`,
   `masking`, `continuum`, `crosscal_continuum`, `cubes`, `mosaic_continuum`,
   `mosaic_line`, `logs`).
2. **Time-stamped logs** — `self.timeNow = "{:%Y%m%d-%H%M%S}".format(datetime.now())`,
   logs go to `<output>/logs-<timeNow>/` with a stable `<output>/logs` symlink.
3. **Worker selection** — iterates through `config.keys()`, derives a worker
   filename (handling the `__N` suffix), and clamps to `[start_worker,
   end_worker]`.  The first three workers (general, getdata, obsconf) are
   always retained — see lines 75-109.
4. **Discover flag-name owners** — for each enabled worker, if the module
   defines `FLAG_NAMES`, that worker's flag tags are remembered in
   `self.flags`.
5. **`_full_init()`** — call `init_pipeline()` (described below) and copy the
   raw + merged config files into `<output>/cfgFiles/<basename>-<timeNow>.{orig.yml,yml}`.

`init_pipeline()` (lines 296-369):

* Creates all output directories (`os.mkdir`).
* Symlinks logs.
* If `general.prep_workspace` is true, **copies `caracal/data/meerkat_files/*`
  into `<input>/`** so AOFlagger/Tricolour strategies and beam files are visible
  to cabs.
* Switches `caracal.log_filehandler` to write to `<logs>/log-caracal.txt`.
* Calls `notebooks.setup_default_notebooks(...)` to copy/render the `init_*`
  and `report_*` notebooks listed in `general.{init,report}_notebooks`.

`run_workers()` (lines 374-466):

```
parse general.cabs → cabspecs_general
for each worker (label, module, config, cabspecs):
    if config.enable is False:  skip
    optional check_config(config, name)
    cabspecs ← merged with worker.cabs (if any)

for each active worker:
    label = module.LABEL or filename-stem
    if "__" in name: label += "__" + suffix
    recipe = stimela.Recipe(label, ms_dir=msdir, log_dir=logs,
                            cabspecs=cabspecs,
                            logfile_task=f"{logs}/log-{label}-{{task}}-{timeNow}.txt")
    recipe.JOB_TYPE = container_tech
    self.CURRENT_WORKER = name
    worker.worker(self, recipe, config)
    recipe.run()
    cleanup *.last  in output
    if config.report and generate_reports: regenerate_reports()
```

### 8.1 MS-name service

The administrator gives every worker a uniform interface for naming MS files:

| Method | Purpose |
|--------|---------|
| `init_names(dataids)` | Glob `<rawdatadir>/<id>.<ext>` to populate `msnames`, `msbasenames`, `prefix_msbases`, `nobs`. Honors `getdata.ignore_missing`. |
| `form_msname(msbase, label, field)` | `<msbase>[-<filtered field>][-<label>].<ext>` |
| `get_mslist(iobs, label, target)` | Single-MS list, or one-MS-per-target list. |
| `get_target_mss(label)` | `(unique_targets, all_mss, per_target_dict)`. |
| `get_msinfo(msname)` | Caches `<msbase>-summary.json` (an MSUtils dump) per file mtime. |

### 8.2 Calibration-library service

| Method | Purpose |
|--------|---------|
| `get_callib_name(name, ext='yml', extra_label=None)` | Build a `<caltables>/callib-<name>.<ext>` path. |
| `load_callib(name)` | YAML-load that file. |
| `save_callib(callib, name)` | YAML-dump. |
| `parse_cabspec_dict(cabspec_seq)` | Turn the schema's `cabs:` list-of-{name,version,tag} into a Stimela cabspecs dict. |

### 8.3 Flag-version service

`workers/utils/manage_flagsets.py` (281 lines) wraps `cab/casa_flagmanager` and
the Owlcat `flag-ms` script:

| Function | Purpose |
|----------|---------|
| `get_flags(pipeline, ms)` | Read `<msdir>/<ms>.flagversions/FLAG_VERSION_LIST`. |
| `add_cflags(...)` | `casa_flagmanager mode=save versionname=<name>`. |
| `restore_cflags(...)` | `casa_flagmanager mode=restore` with `merge=replace`. |
| `delete_cflags(...)` | Delete the named version *and everything saved after it*. |
| `delete_flagset / clear_flagset / update_flagset` | Owlcat bitflag set manipulation via `cab/pycasacore`. |
| `conflict(conflict_type, ...)` | Raise `RuntimeError`, but first emit a multi-paragraph user-facing message describing the four resolution options (re-name worker, `rewind_flags.mode=reset_worker`, `rewind_flags.mode=rewind_to_version`, or `overwrite_flagvers: true`). |

The pattern in every worker is:

```
flags_before_worker = f"{prefix}_{wname}_before"
flags_after_worker  = f"{prefix}_{wname}_after"
- save flags_before_worker (or rewind to it)
- … work …
- save flags_after_worker
```

### 8.4 Reference-antenna selection (`workers/utils/manage_antennas.py`)

`get_refant()` is invoked when `obsconf.refant == 'auto'`.  It runs
`cab/flagstats` against the MS, reads the resulting JSON, restricts to
antennas with `array_centre_dist <= maxdist` whose baselines all exceed
`minbase`, sorts them by flag fraction, and returns the top 1–3 names as a
comma-separated string.

---

## 9. Workers — file-by-file summary

Every worker module exposes the same external contract:

```python
NAME = "Human-readable name"     # logged
LABEL = "label"                  # used for stimela recipe label
FLAG_NAMES = [...]               # optional; consumed by WorkerAdministrator

def check_config(config, name): # optional pre-flight validation
    ...

def worker(pipeline, recipe, config):  # mandatory entry point
    ...
```

| Worker | File | Lines | Purpose |
|--------|------|------:|---------|
| `getdata` | `getdata_worker.py` | 37 | Validate dataid list; if `untar.enable`, pre-stage `.tar` archives via `tar -xvf`. Mostly just `pipeline.init_names(...)`. |
| `obsconf` | `obsconf_worker.py` | 279 | Run `cab/casa_listobs`, `cab/msutils command=summary`, optional `cab/sunblocker command=vampirisms` (sunrise/sunset), elevation plots via `cab/casa_plotms` or `cab/owlcat_plotelev`. Then auto-categorise calibrator/target fields using MS *intent strings* — see `dispatch_crew/utils.categorize_fields`. Stores per-MS lists `pipeline.{target,gcal,fcal,bpcal,xcal}_{,ra,dec,id}`, `nchans`, `firstchanfreq`, `lastchanfreq`, `chanwidth`, `specframe`, `startdate`, `enddate`. |
| `prep` | `prep_worker.py` | 247 | `getfield_coords()`-driven coordinate sanity check against the southern + CASA calibrator DBs (rephases bpcal via `cab/casa_fixvis` if drift is between `tol_diff` and `tol`); optional `cab/casa_clearcal`; `manage_flags.mode ∈ {legacy,restore}` to checkpoint a `caracal_legacy` flag version; spectral weights via `cab/msutils command=estimate_weights` or a CASA `initweights(wtmode='ones', dowtsp=True)` script. |
| `transform` | `transform_worker.py` | 454 | Splits / averages / OTF-applies calibration via `cab/casa_mstransform`. Two top-level modes: `transform_mode='split'` (single label_in) and `transform_mode='concat'` (comma-separated label_in → freq concat). `otfcal` resolves a callib via `callibs.resolve_calibration_library`, optionally chains a `cab/casa_applycal` for polcal-only gaintypes (`Xfparang`, `Df`, …) before a second mstransform that splits the corrected column. Output mode controlled by `output_pcal_ms ∈ {final, intermediate, both}`. Maintains `caracal_legacy` flag version on output MS. |
| `flag` | `flag_worker.py` | 649 | Workhorse flagging stage. Each flag step appends to a Stimela recipe:<br>• `flag_autocorr` → `cab/casa_flagdata mode=manual autocorr=true`<br>• `flag_autopowerspec` → `cab/politsiyakat_autocorr_amp`<br>• `flag_quack` / `flag_elevation` / `flag_shadow` / `flag_spw` / `flag_time` / `flag_scan` / `flag_antennas` / `flag_manual` → `cab/casa_flagdata` with appropriate mode<br>• `flag_mask` → `cab/rfimasker` (rfi-mask numpy file + uvrange selection)<br>• `flag_rfi.flagger` ∈ {`aoflagger`, `tricolour`, `rfimasker`} → `cab/autoflagger`, `cab/tricolour`, …<br>• `summary` → `cab/casa_flagdata mode=summary`.<br>Honours `rewind_flags.{mode,version}` and `overwrite_flagvers`. |
| `crosscal` | `crosscal_worker.py` | 893 | Bandpass, gain, delay, flux-scale calibration. The driving table is the `RULES` dict (lines 47-90) which maps a single character (`K`, `G`, `F`, `B`, `A`, `I`, `S`) to `{name, interp, cab, gaintype, mode, field}`. Users specify a string `order = "KGBAKGBK"` plus parallel arrays `combine`, `solint`, `calmode` (one entry per character). Implements `set_model` with three flavours (`unity`, MeerKAT lsm via MeqTrees `cab/calibrator`, MeerKAT crystalball model). Calls `solve()` (huge inner function, lines 123+), generates `<prefix>_{primary,secondary}.{K,G,B,F}<iter>` tables, then `cab/casa_applycal applyto: [gcal,bpcal]`. Builds a `callib-<prefix>-<label_cal>.{yml,json}`. |
| `polcal` | `polcal_worker.py` | 1797 | Polarisation calibration. `xcal_model_fcal_leak()` and friends solve for `Gpol1`, `Kcrs` (cross-hand delay), `Xref/Xf` (cross-hand phase), `Dref/Df` (leakage) via successive `cab/casa_polcal` invocations. Also applies `feed_angle_rotation` if requested (MeerKAT default suggests `-90`). Reuses `callibs` to bookkeep both polcal and crosscal libraries simultaneously. |
| `inspect` | `inspect_worker.py` | 629 | Diagnostic plots. Two modes:<br>• Native `real_imag`, `amp_phase`, `amp_uvwave`, `amp_chan`, `amp_scan`, `amp_ant`, `phase_chan`, `phase_uvwave` blocks → `cab/casa_plotms`, `cab/shadems`, or `cab/ragavi_vis` depending on `standard_plotter`.<br>• Free-form `shadems.plots`, `plots_by_field`, `plots_by_corr` lists processed by `_process_shadems_plot_list()` (recursive structure: per-field/per-corr substitutions, `dir`, `cnum`, `--cmap pride`, …).<br>`l2d()` is a shell-arg parser that turns a string like `"-x real -y imag -c SCAN_NUMBER"` into a dict for `cab/shadems_direct`. |
| `mask` | `mask_worker.py` | 820 | Builds a clean mask image from NVSS/SUMSS catalogue queries (via `astroquery`), thresholding (`make_mask.mask_method=thresh`) or running `cab/sofia`. Includes its own `ra2deg`/`dec2deg`/`nvss_pbcorr` helpers, plus a Gaussian primary-beam correction (FWHM = 1.02 λ/D, with D=13.5 m for MeerKAT). Optional merge with `extended_source_map` (e.g. `Fornaxa_vla.FITS`). |
| `selfcal` | `selfcal_worker.py` | 2630 | Continuum imaging + self-cal loop. Tools are selectable via `calibrate_with` ∈ {`meqtrees`, `cubical`}; CubiCal is the default. Constants `CUBICAL_OUT`, `CUBICAL_MT`, `SOL_TERMS_INDEX` map CubiCal output modes / matrix types to compact two-letter codes used in CARACal. The loop runs `cal_niter` rounds: WSClean image → PyBDSF/breizorro mask → CubiCal calibrate → repeat, with separate G/B/DD solution intervals (`gsols_timeslots`, `bsols_timeslots`, `ddsols_*`). Optional `transfer_apply_gains` / `transfer_model` interpolate to a finer-channel `transfer_to_label` MS. `aimfast` block runs aimfast to assess image quality and steer "automatic convergence". |
| `ddcal` | `ddcal_worker.py` | 491 | Direction-dependent calibration. Imports `astropy.units`, `SkyCoord`, `WCS`, and `regions.PolygonPixelRegion` lazily. Uses DDFacet (`cab/ddfacet`) for imaging and CubiCal-DDF/PGS for solving direction-dependent solutions. `de_sources_mode ∈ {auto, manual}` selects which targets to peel; `calibrate_dd.de_target_manual`/`de_sources_manual` lists are paired into a `de_dict`. Output goes to `<output>/3GC/`. |
| `line` | `line_worker.py` | 2067 | Spectral-line worker; the largest after polcal/selfcal. `subtractmodelcol` / `addmodelcol` rewrite `CORRECTED_DATA` ± `MODEL_DATA`. `mstransform` segment runs `cab/casa_mstransform regridms=true` with Doppler tracking (`telescope ∈ {askap,atca,gmrt,meerkat,vla,wsrt}`) and CASA `uvcontsub` (`uvlin`). `make_cube` → `cab/wsclean -channels-out N`, optional `pb_cube` correction, `freq_to_vel()` Python helper rewrites the FITS header to `VRAD`/`m·s⁻¹`, optional `remove_stokes_axis` collapses the 4th axis, and `sofia` block runs SoFiA-2. Also handles `flag_mst_errors` (post-mstransform NaN flagging) via `flag_Uzeros.UzeroFlagger` (a 1249-line analysis module that detects U≈0 stripes from solar interference). |
| `polimg` | `polimg_schema.yml` driven `polimg_worker.py` | 532 | Polarisation imaging. `stokes` ∈ {QU,QUV,IQU,IQUV,I,Q,U,V}; `make_images` configures a single WSClean run with `-pol IQUV` and `-join-channels`; `rmsynth` block invokes RM synthesis. Also calls `cab/eidos` to predict the Stokes beam cube. |
| `mosaic` | `mosaic_worker.py` | 406 | Combines multiple pointings.  `mosaic_type ∈ {continuum, line}` selects the input directory (`continuum/` vs `cubes/`). Constructs primary beams in two flavours: `pb_type=gaussian` (FWHM = 1.02 λ/D — see `make_gaussian_pb`) or `pb_type=mauchian` (Mauch et al. 2020 — `make_mauchian_pb`). Uses `cab/montage` for re-projection and `cab/spimple_imconv` for convolution. |

---

## 10. `dispatch_crew/` — internal utilities

### 10.1 `config_parser.py` (399 lines)

Already covered in §6 and §7.4.  Three classes/functions of note:

* `basic_parser(add_help=True) -> argparse.ArgumentParser` — declarative basic
  CLI options.
* `class config_parser:` (note: lowercase) —
  * `validate_config(config_file)` — load YAML, validate each section against
    the corresponding `{worker}_schema.yml`, return `(content, version)` or
    raise `ConfigErrors`.
  * `populate_parser(content)` — recursive `_process_subparser_tree`.
  * `update_config_from_args(content, args)` — re-parse argv, type-cast YAML
    overrides, return updated `(options, config)`.
  * `save_options(config, filename)` / `log_options(config)` — bookkeeping.
* `class ConfigErrors(RuntimeError)` — collects all validation errors with
  the section that produced them; the CLI iterates the `errors` dict in
  `main.py:204-209` to print a flat diagnostic before exiting 1.

### 10.2 `utils.py` (431 lines)

| Symbol | Purpose |
|--------|---------|
| `Fields` (dataclass) | Holds parallel lists `ids`, `names`, `dirs`; methods `index()`, `name_from_id()`, `id_from_name()`. |
| `angular_dist_pos_angle(ra1, dec1, ra2, dec2)` | Returns `(angular distance, position angle)` between two points (cribbed from ska-sa/tigger). |
| `categorize_fields(info)` | Maps MS `STATE` intent strings (`CALIBRATE_FLUX`, `CALIBRATE_AMPL`, `CALIBRATE_PHASE`, `CALIBRATE_BANDPASS`, `TARGET`, `CALIBRATE_POLARIZATION`) to the CARACal categories `fcal/gcal/bpcal/target/xcal`. |
| `get_field_id(info, field_name)` | Map field name(s) → `SOURCE_ID` indices. |
| `select_gcal(info, targets, calibrators, mode='nearest'\|'most_scans')` | Auto gcal selection. |
| `observed_longest(info, calfields)` | Field with longest total observation. |
| `field_observation_length(info, field, return_scans)` | Total time per field. |
| `closeby(radec1, radec2, tol)` | Boolean within `tol` rad on the sphere. |
| `hetfield(info, field, db, tol)` | Match a field to a calibrator in a DB by coords; return DB key. |
| `find_in_native_calibrators(info, field, mode='both'\|'sky'\|'mod'\|'crystal')` | Look up calibrator in southern DB; return `{I,a,b,c,d,ref}` model dict, or an `lsm` filename, or a `crystal` filename. |
| `find_in_casa_calibrators(info, field)` | Look up calibrator in CASA standards (Perley-Butler 2010/2013, Perley-Taylor 99, …); return the standard label string. |
| `read_taylor_legodi_row(info, field)` | Parse `data/taylor_legodi_2024.txt` and return polarisation flux model dict. |
| `meerkat_refant(obsinfo)` | Reads `info["RefAntenna"]` (only set for MeerKAT MSs downloaded by CARACal). |
| `estimate_solints(msinfo, skymodel, Tsys_eta, dish_diameter, npol, gain_tol=0.05, j=3)` | Sandeep Sirothia's `dt·dν` formula for the (time × freq) solution interval needed to reach a given gain noise floor.  Returns `(dt_dfreq, dtime, dfreq)`. |
| `imaging_params(info, spwid=0)` | Returns `(max_resolution_deg, FoV_deg)` from baselines and dish size. |
| `filter_name(string)` | Sanitise a field name for use in filenames (`+` → `_p_`, non-alphanum → `_`). |

### 10.3 `caltables.py` and `catalog_parser.py`

`caltables.py` (42 lines) provides two lazy loaders, `calibrator_database()`
and `casa_calibrator_database()`, which return `catalog_parser` instances
backed by `data/southern_calibrators.txt` and `data/casa_calibrators.txt`.
Both are module-globals cached after first read.

`catalog_parser.py` (251 lines) is a hand-rolled parser for a custom file
format with three line types:

* `name=… epoch=… ra=…h…m…s dec=…d…m…s a=… b=… c=… d=…` — Perley-Butler form
  in MHz (logS = a + b·log f + c·log²f + d·log³f).
* `alias src=A dest=B` — alias B to A.
* `lsm name=A epoch=… <filename>` and `crystal name=A epoch=… <filename>` —
  attach a Tigger LSM or a Crystalball model filename.

The parser's `convert_pb_to_casaspi(vlower, vupper, v0, a, b, c, d)`
classmethod fits the Perley-Butler polynomial onto the CASA SPI form
`S(v0)·(v/v0)^(a' + b'·log(v/v0) + c'·log²(v/v0) + d'·log³(v/v0))` over a
specified frequency range using `scipy.optimize.curve_fit`, asserting that
each coefficient's standard error is below 1e-6.  This makes the *southern*
calibrator catalogue usable as `setjy standard='manual'` parameters for CASA.

### 10.4 `worker_help.py`, `stream_director.py`, `interruptable_process.py`, `noisy.py`

* `worker_help.worker_options.print_worker()` (58 lines) — invoked when the
  user types `caracal -wh <worker>`.  Recursively visits the schema and
  appends an `argparse.add_argument()` for each leaf, then calls
  `parser.parse_args(["--help"])` to print and exit (a slightly hackish
  technique).
* `stream_director.stream_director` (61 lines) — context manager that wraps
  `sys.stdout`/`sys.stderr` with a `StringIO` subclass that forwards each
  written line to `logger.log(level, line)`, while filtering out lines that
  themselves look like log records (to avoid recursion).  Currently
  unused at runtime (`worker_administrator.py:118` shows it commented out).
* `interruptable_process.interruptable_process` (26 lines) — a
  `multiprocessing.Process` subclass with a public `interrupt()` method that
  sends `SIGINT` to its child PID.
* `noisy.PredictNoise(MS, tsyseff, diam, selectFieldName, verbose=0)` (213
  lines) — Computes the theoretical Stokes-I natural noise of an MS.  It
  reads `FLAG`, `INTERVAL`, `CHAN_WIDTH`, `CHAN_FREQ`, `POLARIZATION` via
  `pyrap.tables`, restricts to cross-correlations of XX/YY/RR/LL, and uses
  `rms = √2·k_B·Tsys/η / (A_ant·√(Δν·Δt·N_pol))` per channel, both ignoring
  flags and accounting for them.  Used by `line_worker` to compute expected
  cube noise.

---

## 11. Notebooks & reporting (`caracal/notebooks/`)

The pipeline relies on **radiopadre** to render `*.ipynb` files into HTML
inside a container (since the notebooks themselves often need
casa-cube/casa-table/`pyrap` to evaluate).  Templates ending in `.j2` are
rendered through Jinja2 with the user's full config as the rendering context.

| Notebook | Purpose |
|----------|---------|
| `std-progress-report.ipynb(.j2)` | Pipeline-level progress dashboard. |
| `detailed-final-report.ipynb(.j2)` | Detailed end-of-run report. |
| `project-logs.ipynb(.j2)` | Browse-by-log report. |
| `project-directory.ipynb` | Filesystem browser of `<output>/`. |
| `header.j2` | Shared Jinja header (logos, prefix, time). |

`setup_default_notebooks(notebooks, output_dir, prefix, config)`
(`notebooks/__init__.py:18-71`):

* Copies `caracal-logo-200px.png` and `caracal-square-logo-32px.png` into
  `<output>/reports/`.
* For each requested notebook name N, looks for `N.ipynb.j2` first (Jinja
  template); if absent, copies a static `N.ipynb`.  The destination filename
  is `<prefix>-<N>.ipynb`.
* Mtime-based: only overwrites the destination if the source is newer.

`generate_report_notebooks(...)` invokes `radiopadre-client run-radiopadre
--non-interactive --auto-init {--docker|--singularity} --nbconvert <ipynb>`
to produce HTML.  If the resulting HTML's mtime is older than the start time,
a warning is logged ("the container did not report an error but no HTML was
produced").

---

## 12. Calibration libraries (`workers/utils/callibs.py`)

A *callib* in CARACal is a YAML/JSON mapping of CASA gain-table types to
field-mapping policies.  The `_MODES` table (`callibs.py:6-20`) maps a CASA
gain-table extension to a logical CARACal mode:

| Ext | Mode |
|-----|------|
| `K`  | `delay_cal` |
| `B`  | `bp_cal` |
| `F`/`G` | `gain_cal` |
| `Gpol` | `gain_xcal` |
| `Kcrs` | `cross_delay` |
| `Xref` | `cross_phase_ref` |
| `Xf`/`Xfparang` | `cross_phase` |
| `Dref` | `leakage_ref` |
| `Df`/`Df0gen` | `leakage` |
| `Gxyamp` | `cross_gain` |

`add_callib_recipe(callib, gt, interp, fldmap, field=None, calwt=False)` adds
one entry; `resolve_calibration_library(pipeline, msprefix, cal_lib, cal_label,
output_fields, default_interpolation_types)` reads such a YAML, decides which
gain tables apply to which output fields, and writes a CASA-compatible
`callib.txt` (one line per gaintable):

```text
caltable="<absolute path>" calwt=False tinterp='linear' finterp='linear'
fldmap='nearest' field='J0408-65' spwmap=0
```

This `.txt` is then handed off to CASA `mstransform docallib=true` (in the
transform worker) or directly to `casa_applycal`.

---

## 13. Worker-shared utilities — extra detail

### 13.1 `workers/utils/flag_Uzeros.py` (1249 lines)

A self-contained `UzeroFlagger` class that detects and flags solar-RFI
"U=0 stripes" in MS visibilities.  Workflow:

1. `setDirs(output)` creates `stripeAnalysis/{logs,msdir,cubes,fft,plots}/`.
2. Builds short stripe MSes around `u≈0` baselines.
3. Images them into temporary cubes (via `stimela.recipe`).
4. FFTs each cube along the time axis to find stripe periods.
5. Fits a periodic model with `scipy.optimize`; sigma-clips outliers.
6. Writes corrected flags back into the parent MS using `casacore.tables`.

Imports include: `casacore.measures`, `casacore.tables`, `astropy.coordinates`,
`astropy.io.fits`, `astropy.wcs`, `astropy.io.ascii`, `matplotlib.gridspec`,
`scipy.stats`, `scipy.constants`.

### 13.2 `workers/utils/image_contsub.py` (346 lines)

A standalone CLI script (`#! /usr/bin/env python` shebang at line 1) plus the
main `imcontsub()` function.  Performs continuum subtraction in a FITS data
cube.  Two modes: `fitmode='median'` (channel-wise median over a sliding
window of `length` channels) or `fitmode='poly'` (per-spaxel polynomial of
order `polyorder`, optionally Savitzky-Golay-iterated `sgiters` times).  The
fitted continuum can be optionally Gaussian/tophat-convolved with a kernel of
size `kersiz`.  Outputs include `<outcubus>` (residual cube), `<fitted>`
(fitted continuum), and `<confit>` (convolved fitted continuum).  A pixel is
masked if `mask` cube is non-zero at that voxel.

---

## 14. Logging

`caracal/__init__.py:101-176` sets up a tree of loggers:

* `LOGGER_NAME = "CARACal"` → a stand-alone logger (no propagation).
* `STIMELA_LOGGER_NAME = "CARACal.Stimela"` → reuses Stimela's machinery via
  `stimela.logger(STIMELA_LOGGER_NAME, propagate=True, console=False)`.
* `DelayedFileHandler(MemoryHandler)` — buffers log records up to 100 000
  entries until `setFilename()` is called.  This lets log records emitted
  *before* the output directory exists still end up in the final
  `log-caracal.txt`.
* `init_console_logging(boring, debug)` — chooses
  `stimela.log_boring_formatter` vs `stimela.log_colourful_formatter`, adds a
  console filter that suppresses Stimela DEBUG/INFO unless they carry one of
  the magic attributes (`stimela_subprocess_output[1]=='start'`,
  `stimela_job_state`).
* The console filter also drops records flagged with `traceback_report=True`
  or `logfile_only=True` from the console (they still go to the file).

Per-worker recipes log to `<logs>/log-<label>-<task>-<timeNow>.txt` thanks to
`Recipe(..., logfile_task=...)` (worker_administrator.py:432).

---

## 15. Tests (`tests/`)

The test suite is small but exercises the full CLI:

| File | Lines | What it exercises |
|------|------:|-------------------|
| `tests/__init__.py` | 69 | `InitTest` fixture: creates random temp files/dirs in `TESTDIR`, helps update `general.{input,msdir,output}` to point inside `TESTDIR` so the tests do not pollute the working directory. |
| `tests/test_runner.py` | 61 | `test_help` → run `caracal --help` and assert output. `test_config_setup` → `caracal -gdt meerkat_continuum -gd <tmp>`, validate the produced config, instantiate `WorkerAdministrator(..., partial_init=True, end_worker='obsconf')`, then iterate every top-level worker and run `caracal -wh <worker>` to assert the help printer succeeds. |
| `tests/test_dispatch_crew_utils.py` | 61 | `test_Fields`: round-trips `Fields.{index,name_from_id,id_from_name}`. `test_fieldinfo`: checks `field_observation_length` and `observed_longest` on a real fixture. |
| `tests/obsinfo/ms_summary.json` | – | Real MSUtils summary fixture used by `test_dispatch_crew_utils.py`. |

`partial_init=True` (the test mode) skips `init_pipeline()`, so no
filesystem side-effects occur during tests.

CI configuration (`.github/workflows/python-ci.yml`) runs ruff (check + format
diff) and `pytest tests` against Python 3.9–3.13 using `uv` for env
management.  Note that the python-ci matrix nominally covers up to 3.13, even
though `pyproject.toml` caps the supported range at `<3.13`.

---

## 16. Documentation pipeline (`docs/`)

`docs/make_caracal_docs.py` (289 lines) **auto-generates** the per-worker rST
pages at `docs/sphinx/manual/workers/<worker>/index.rst` directly from the
schemas.  It:

1. Reads `caracal/schema/*_schema.yml`.
2. Skips any worker not in the curated `sortedWorkers` list (line 26-40).
3. For each worker, walks its YAML mapping and emits rST headers + an
   in-place description that includes type, `required`, default
   (from `example`), and the list of enum choices if present.
4. Writes the top-level `manual/workers/index.rst` toctree.
5. Pulls the install / readme out of the project's `README.rst` into
   `caracalREADME.rst` (lines 49-59 — strips everything from "Installation &
   Run" up to "Running").

The Sphinx project (`docs/sphinx/conf.py`) uses `recommonmark` and the
`sphinx_rtd_theme`.  The Read the Docs build (`.readthedocs.yml`) calls
`make_caracal_docs.py` first, then `make html`.

---

## 17. Notable internals & invariants

* **Schema is the single source of truth.**  Adding an option to a worker
  means adding a leaf to its `*_schema.yml`.  The CLI option name is
  automatically `--<worker>-<group>-<key>` with `-` separators; the Python
  attribute name uses `_` separators (`config_parser.py:251-252`).
* **Re-instantiation by `__N` suffix.**  `flag__2`, `transform__3`, etc.
  reuse the parent worker's schema and behaviour but get a fresh recipe and
  flag-version namespace.  `_worker = worker.split("__")[0]` is the universal
  rule.
* **First three workers are always run.**  `general`, `getdata`, `obsconf`
  are not skippable (`worker_administrator.py:75-109`).
* **Flag-version checkpoints.**  Every non-trivial worker saves
  `<prefix>_<workername>_before` *before* doing anything destructive and
  `<prefix>_<workername>_after` afterwards.  Re-running the same worker on
  the same MS will refuse to overwrite these checkpoints unless
  `overwrite_flagvers: true` or `rewind_flags.enable: true` is set.
* **MSes never live in containers.**  MS files live on the host filesystem
  in `general.msdir` (or `general.rawdatadir` for read-only inputs); Stimela
  cabs see them via volume mounts that Stimela handles.
* **Pure-Python orchestration only.**  The package contains no compiled code
  and no GPU code paths; performance is dominated by the cab subprocesses,
  which themselves may use multi-threading or GPU (e.g. CubiCal) but are
  configured through CARACal options like `ncpu`, `shared_mem`, `chan_chunk`.
* **Notebook templates are user-overridable.**  Anything that exists as
  `<output>/<prefix>-<name>.ipynb` is *not* re-rendered unless the source
  template is newer; users can hand-edit reports.
* **Calibrator coordinate offsets are detected automatically.**  `prep_worker`
  rephases the bandpass calibrator with `cab/casa_fixvis` if the MS-stored
  position differs from the catalogue position by more than `tol_diff` but
  less than `tol` arcsec — defending against vintage MeerKAT MSes with
  miscalculated UVWs.  This safety net only activates when
  `prep.fixcalcoords.enable: true`.
* **`ruff.toml` line-length is 180** (`ruff.toml:1`), much wider than PEP-8;
  the format check is enforced in CI.
* **Empty `manage_caltabs.py`.**  `workers/utils/manage_caltabs.py` is a
  zero-line placeholder — possibly a stub for forthcoming functionality.

---

## 18. Known limitations / TODOs visible in the code

* `dispatch_crew/noisy.py` imports `from matplotlib.pyplot import flag` at
  the top of the file (line 10) but never uses it; this is a lingering
  artefact and shadows the `flag` local later (a latent bug).
* `dispatch_crew/utils.py:263` (in `find_in_native_calibrators`) has a path
  through `read_taylor_legodi_row` that calls `utils.load(info)` rather than
  `utils.load_yaml(info)` — likely a typo that would crash if reached.
* `worker_administrator.py:413` has a comment `# OMS skipping this here` and
  `init_names()` is intentionally not invoked during `_full_init` — moved to
  the `getdata` worker instead.
* `selfcal_worker.py:1083`-ish: a global counter named `self_cal_iter_counter`
  is referenced (per the comment at line 22), an architectural smell.
* `crosscal_worker.py` / `polcal_worker.py`: large `solve()` functions with
  deeply nested option handling — this is the main maintenance hotspot
  according to the recent git log (the most recent commits, `6e4ac868`, etc.,
  all touch `polcal_worker.py`).
* `polimg_worker.py` has a commented-out `# import equolver.beach as beach`
  (line 13) — the equolver dependency is left dangling.
* The `inspect_worker._process_shadems_plot_list` plot grammar is very
  expressive but only loosely documented in the schema; `check_config()`
  dummy-processes the lists with empty substitutions, which catches some
  errors but not all.
* Singularity 4.x support is partial — `README.rst:32` notes that `Apptainer
  does not support all CARACal functionalities`.
* Polished but largely-unused: `dispatch_crew.stream_director` and
  `dispatch_crew.interruptable_process` are present but never invoked from
  the live code path (the equivalent functionality is now provided by
  Stimela).

---

## 19. How RRIVis can interact with CARACal

CARACal expects MS files at `<msdir>/<dataid>.<extension>` (default `.ms`)
and a `<msdir>/<dataid>-summary.json` produced by the `cab/msutils command=summary`
cab.  The `obsconf` worker will auto-categorise calibrator/target fields
from the MS `STATE` table's intent strings — so RRIVis-produced MS files
should populate scan intents (`TARGET`, `CALIBRATE_BANDPASS`, …) for fully
automatic handling.  Alternatively, the user can list field names in
`obsconf.{target,gcal,bpcal,fcal,xcal}` explicitly (or use the magic
strings `'all'`, `'longest'`, `'nearest'`).

The minimum config to imaging from a single RRIVis-output MS would be:

```yaml
schema_version: 1.2.0
general:
  prefix: rrivis-test
getdata:
  dataid: ['my_simulated_ms']   # expects msdir/my_simulated_ms.ms
  extension: ms
obsconf:
  refant: '0'
  target: ['my_target_field']
selfcal:
  enable: true
  img_npix: 2048
  img_cell: 2.0
  cal_niter: 1
  image:
    enable: true
  calibrate:
    enable: false   # bypass calibration loop on a clean simulator MS
```

For pure imaging without any flagging/calibration, set
`flag.enable: false`, `crosscal.enable: false`, and configure only
`getdata`, `obsconf`, and `selfcal`.  Note that the first three workers
always run regardless.

---

## 20. Quick reference cards

### 20.1 CLI cheat sheet

```bash
caracal --help                          # usage
caracal -v                              # version
caracal -gdt meerkat -gd config.yml     # bootstrap from a template
caracal -wh selfcal                     # all options for the selfcal worker
caracal -pcs                            # dump southern calibrator DB
caracal -c config.yml                   # run pipeline (docker)
caracal -ct singularity -c config.yml   # run pipeline (singularity)
caracal -c config.yml -sw flag -ew crosscal     # run only flag→crosscal
caracal -c config.yml -report -nr       # rebuild HTML reports only
caracal -c config.yml -debug            # drop into pdb on failure
```

### 20.2 Directory layout produced by a run

```
<output>/
├── log-caracal.txt              # symlink → logs/log-caracal.txt
├── logs                         # symlink → logs-YYYYMMDD-HHMMSS
├── logs-YYYYMMDD-HHMMSS/        # per-task log files
├── reports/                     # logos + HTML reports
├── obsinfo/                     # listobs / summary / elevation plots
├── diagnostic_plots/            # inspect-worker output
├── cfgFiles/                    # original + merged config copies
├── caltables/                   # all CASA gain tables + callibs
├── masking/                     # mask FITS files
├── continuum/                   # selfcal images, mosaics/
├── continuum/crosscal/          # crosscal-stage continuum images
├── cubes/                       # line cubes, mosaics/
└── 3GC/                         # ddcal output (if used)
```

### 20.3 Worker ↔ schema ↔ output table

| Worker | Schema (auto-generated rST: `docs/sphinx/manual/workers/<W>/index.rst`) | Primary outputs |
|--------|-----------------------------------------------------|-----------------|
| `general` | `general_schema.yml` | – (configures paths) |
| `getdata` | `getdata_schema.yml` | unpacks tarballs into `msdir` |
| `obsconf` | `obsconf_schema.yml` | `<msbase>-{obsinfo.txt,summary.json,elevation-tracks.png}` |
| `prep` | `prep_schema.yml` | (in-place) MS modifications, weight columns, legacy flag version |
| `transform` | `transform_schema.yml` | new MSes named `<msbase>[-<field>]-<label>.ms` |
| `flag` | `flag_schema.yml` | flag-version `<prefix>_flag[__N]_after` |
| `crosscal` | `crosscal_schema.yml` | `caltables/<prefix>_{primary,secondary}.{K,G,B,F}<n>` + `callib-<prefix>-<label>.{yml,json,txt}` + ragavi gain plots |
| `polcal` | `polcal_schema.yml` | `<prefix>.{Gpol1,Kcrs,Xref,Xf,Dref,Df}` |
| `inspect` | `inspect_schema.yml` | `diagnostic_plots/<dirname>/*.png` |
| `mask` | `mask_schema.yml` | `masking/<label_out>-<target>.fits` |
| `selfcal` | `selfcal_schema.yml` | `continuum/image_<n>/*.fits`, `quality_check.txt`, transferred MSes |
| `ddcal` | `ddcal_schema.yml` | `3GC/<prefix>-DD-precal.{model,residual,image}.fits`, DD gain tables |
| `line` | `line_schema.yml` | `cubes/cube_<n>/*.fits`, `*_mst.ms`, sofia catalogues |
| `polimg` | `polimg_schema.yml` | `continuum/image_<n>/*-{Q,U,V}.fits` |
| `mosaic` | `mosaic_schema.yml` | `continuum/mosaics/<name>.{fits,wt.fits}` or `cubes/mosaics/<name>.fits` |

---

*End of caracal reference.*
