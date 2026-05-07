# ism3d — Interferometric Source Modeling up to 3D

> Exhaustive technical reference for the `ism3d` Python package vendored as a
> git submodule under `simulators/ism3d/` of the RRIVis project.
>
> All paths in this document are relative to `simulators/ism3d/` unless stated
> otherwise. Citations point at the file and (where helpful) the line range
> that backs the claim.

---

## 1. Overview

`ism3d` (full title *"ISM3D: Interferometric Source Modeling up to 3D"*) is a
Python package for **simulating and modeling astronomical sources from radio
interferometric observations**. It is designed primarily as a forward-modeling
and parameter-fitting toolkit for resolved galaxy emission (line + continuum)
in visibility space, with optional support for image-domain (xy) data.

Quoting the project README (`README.rst:21`):

> *A Python package for simulating and modeling astronomical sources from
> radio interferometric observations.*

The package was originally a tool to extract galaxy morphology and kinematics
from large modern interferometer datasets (specifically VLA and ALMA), but it
also includes utility helpers wrapping CASA 6's modular Python interface for
preparing/imaging/visualising calibrated visibility data
(`README.rst:48–51`).

### 1.1 At-a-glance facts

| Item | Value | Source |
|------|-------|--------|
| Project name | `ism3d` | `setup.py:57` |
| Distribution version | `0.3.dev1` | `setup.py:63`, `setup.cfg:2`, `ism3d/__init__.py:5` |
| Development status | "2 — Pre-Alpha" | `setup.py:35` |
| License | BSD 3-Clause | `LICENSE`, `setup.py:53` |
| Copyright | © 2020, Rui Xue | `LICENSE:3` |
| Author | Rui Xue `<rx.astro@gmail.com>` | `AUTHORS.rst:8`, `setup.py:31–32` |
| Python supported | >= 3.6 (`setup.py`); `setup.cfg`: `>= 3.7, < 4`; classifiers list 3.5–3.8 | `setup.py:33,40–43`; `setup.cfg:48` |
| CLI entry point | `ism3d = ism3d.cli:main` | `setup.py:47–51` |
| GitHub | https://github.com/r-xue/ism3d | `setup.py:62` |
| Docs | https://ism3d.readthedocs.io / https://www.magclouds.org/ism3d | `README.rst:13,24` |
| PyPI | https://pypi.org/project/ism3d | `README.rst:26` |
| Released git tags in this checkout | `0.1.dev0`, `0.1.dev2`, `0.2.dev1` | `git tag` |
| Languages | 100% Python (no compiled extensions in this tree) | `find` over the source tree |
| Total Python LOC under `ism3d/` | ~11,652 lines across 47 `.py` modules | `wc -l` |

### 1.2 Headline features

Distilled from `README.rst:29–69`:

- *Efficient forward-modelling of galaxy emission*: build spatially and
  spectrally resolved galaxy emission models from analytical or physical
  prescriptions of geometry, emissivity, kinematics, and dynamics; combine
  multiple line/continuum components.
- *Simulated observation*: render the galaxy model into a wide range of data
  forms — radio interferometer visibilities, single-dish or optical IFU
  spectral cubes, multi-band photometric images, 1D spectra (some only
  partially implemented).
- *Flexible model fitting/optimisation*: a unified interface over several
  optimisation algorithms (Amoeba simulated annealing, lmfit, emcee MCMC).
- All modelling details live in one human-editable parameter file (a custom
  `.inp` config based on `configparser.ExtendedInterpolation`,
  see §6).
- Targets large heterogeneous multi-wavelength joint fits, with
  multi-threading and careful memory-footprint management.

### 1.3 Sub-package map (from `README.rst:53–71`)

| Sub-package | Purpose (README phrasing) |
|-------------|-----|
| `arts/` | Generate artificial sources up to 3D (position-position-frequency/wavelength/velocity) in a sparse array (cloudlet) or regular grid (spectral cube) |
| `simuv/` | Simulator for radio interferometric observations |
| `simxy/` | Simulator for spectroscopic imaging, photometric imaging, 1D spectroscopy (not fully implemented) |
| `modeling/` | Model fitting/optimiser framework (was the "modelling" name in README) |
| `uvhelper/` | Utility functions to help prepare and analyse "uv" data for modelling, built on CASA 6 |
| `xyhelper/` | Help prepare/analyse "xy" (image-domain) data for modelling |
| `utils/` | General utility functions used by other modules |
| `maths/` | Pseudo-random generators and other math helpers |
| `visualize/` | Plotting utilities (called *plots* in README) |

The README candidly notes:

> *While the current effort is still a work in progress: inconsistent
> documentation, many non-working function placeholders, etc., fast-paced
> changes are expected soon.* (`README.rst:74`)

---

## 2. Repository layout

```
simulators/ism3d/
├── AUTHORS.rst                 # Author metadata
├── CONTRIBUTING.rst            # Contribution guide (cookiecutter template style)
├── Dockerfile.dev              # Dev container layered on rxastro/casa6
├── HISTORY.rst                 # Release notes (0.1.dev1 → 0.3.dev1)
├── LICENSE                     # BSD-3-Clause
├── MANIFEST.in                 # sdist file inclusion rules
├── README.rst                  # Project description (also embedded in docs)
├── docs/                       # Sphinx documentation source + pre-built HTML
│   ├── index.html              # Top-level redirect
│   ├── readme.md               # Brief docs note
│   ├── source/                 # rST sources (conf.py, index.rst, usage, ...)
│   │   ├── conf.py             # Sphinx config
│   │   ├── index.rst           # Entry document (toctree, autosummary)
│   │   ├── usage.rst           # Hand-written usage doc
│   │   ├── readme.rst, history.rst, authors.rst, latex.rst
│   │   ├── theme.css           # Custom CSS
│   │   └── sp-ism3d.bib        # Bibliography (sphinx-astrorefs)
│   ├── statics/                # Pre-rendered HTML site (autosummary, _static, ...)
│   └── media/bigpine20100113.jpeg
├── ism3d/                      # The actual Python package
│   ├── __init__.py             # Package exports + FFT backend selection
│   ├── ism3d.py                # Stub "main module"
│   ├── cli.py                  # Argparse CLI (`ism3d` entry point)
│   ├── interface.py            # .inp parser + asteval interpreter
│   ├── logger.py               # Logging setup (with optional CASA logsink)
│   ├── arts/                   # Artificial source generators (geometry, kinematics, lensing)
│   ├── maths/                  # Custom RVs, geometry helpers
│   ├── modeling/               # Optimisation + likelihood + analysis
│   ├── simuv/                  # uv-domain forward modelling (NUFFT, galario, FFT)
│   ├── simxy/                  # xy-domain (image cube) forward modelling
│   ├── utils/                  # General utilities, I/O, metadata, misc helpers
│   ├── uvhelper/               # CASA-backed MS readers/writers, imager, FT helpers
│   ├── visualize/              # plt_* helpers (1D spectra, slices, mom0, mosaic)
│   ├── xyhelper/               # Image-cube helpers (sky, cube)
│   └── resource/               # Default config, .inp template, FITS header
├── tests/                      # pytest tree (mostly placeholder stubs)
│   ├── __init__.py
│   └── test_ism3d.py
├── macos_quickinstall.md       # macOS / macports / casa6 install recipe
├── requirements_casa6pip36.txt # Pin CASA 6 wheels from NRAO PyPI
├── requirements_dev.txt        # Full development pin set
├── setup.cfg                   # Bumpversion + setuptools options + install_requires
├── setup.py                    # Distutils setup
├── tox.ini                     # tox envlist py35–py38, flake8
├── .gitmodules                 # (root .gitmodules; submodule pointer used elsewhere)
├── .github/ISSUE_TEMPLATE.md
└── .travis.yml                 # CI config (Travis)
```

Per-folder commentary is given inline in the file-by-file breakdown in
§9.

## 3. Installation and dependency story

ism3d has *three* overlapping dependency declarations that must be reconciled.
Per `macos_quickinstall.md:51–61`:

> *Currently, the dependency libraries are specified in multiple places (which
> is not ideal)*

### 3.1 `setup.py` — Python version + classifiers only

`setup.py:24` declares `requirements = [ ]`, i.e. no install-requires from the
classic `setup.py`. Setup-time/test-time pins live in:

```python
# setup.py:26-28
setup_requirements = ['pytest-runner', ]
test_requirements  = ['pytest>=3', ]
```

### 3.2 `setup.cfg` — runtime install_requires

The actual install-time dependencies live in `setup.cfg:51–65`:

```ini
install_requires =
    astropy >= 4.0
    scipy
    emcee >= 3.0.0
    corner >= 2.0
    asteval >= 0.9.14
    numexpr >= 2.6.9
    hickle >= 4.0.0
    pyfftw
    fast-histogram >= 0.9
```

Note these notable runtime imports that are NOT in `install_requires`:

| Module | Imported in | Reason |
|--------|-------------|--------|
| `numpy` | every module | Implicit via astropy |
| `galario.single` | `simuv/render.py:18`, `simuv/ft.py:18`, `utils/utils.py:11`, `modeling/opt.py:34` | Listed in HISTORY as removed (`HISTORY.rst:17` "use nufft to replace galario"), but imports remain |
| `finufftpy` | `simuv/render.py:27`, `simuv/ft.py:27` | NUFFT replacement (Flatiron `finufftpy`) |
| `lenstronomy` | `arts/lens.py:9` | SIE ray-tracing |
| `galpy` | `arts/sparse.py:29`, `arts/dynamics.py:4`, `modeling/model.py:31` | Rotation curves / potentials |
| `radio_beam` | `utils/utils.py:9` | `Beams`, `one_beam()` |
| `spectral_cube` | `utils/utils.py:15` | Cube manipulation |
| `lmfit` | `modeling/opt.py:32-33` | Optimisation backend |
| `casatools`, `casatasks` | `uvhelper/ms.py:1–9`, `uvhelper/imager.py:20–22`, `logger.py:11–12` | MS I/O and tclean wrapper |
| `mkl_fft`, `mkl`, `mkl_random` | `__init__.py:70–74` (try/except fallback) | Preferred FFT backend |

### 3.3 `requirements_dev.txt`

`requirements_dev.txt` (76 lines) is the most exhaustive list and pulls in
extras such as `lmfit`, `tqdm`, `regions`, `pvextractor`,
`spectral-cube`, `radio-beam`, `reproject`, `scikit-image`, `galpy`,
`fast-histogram`, `numexpr`, `dask`, `lenstronomy`, `wurlitzer`,
`line_profiler`, `memory_profiler`, `psutil`, plus VCS dependencies:

```
# requirements_dev.txt:42-46
git+https://github.com/flatironinstitute/finufft@master#egg=finufftpy
git+https://github.com/radio-astro-tools/pvextractor@master#egg=pvextractor
git+https://github.com/radio-astro-tools/spectral-cube@master#egg=spectral-cube
git+https://github.com/radio-astro-tools/radio-beam@master#egg=radio-beam
```

### 3.4 `requirements_casa6pip36.txt` — CASA 6 wheels

```
--extra-index-url https://casa-pip.nrao.edu:443/repository/pypi-group/simple
casatools     # requires casadata & numpy
casatasks
casashell
casaplotms
casaviewer
```

### 3.5 macOS install recipe (`macos_quickinstall.md`)

Summary of the curated path:

1. Use macports to install `python36` (for CASA 6 wheels) and `python38`
   (for everything else).
2. Editable install: `pip3 install --user -e .`
3. Pull modular CASA 6 packages from `https://casa-pip.nrao.edu/repository/`:
   `casatools`, `casatasks`, `casaplotms`, `casaviewer`, `casashell` (versions
   6.1 and 6.2 covered).
4. Install `finufft` from PyPI (precompiled wheels) for the NUFFT path.

### 3.6 Docker (`Dockerfile.dev`)

```dockerfile
FROM rxastro/casa6:latest
SHELL ["/bin/bash", "-c"]
ENV APP_HOME /root
WORKDIR ${APP_HOME}

RUN apt-get update && apt-get dist-upgrade -y && \
    apt-get install --no-install-recommends -y \
        gfortran build-essential make \
        cython3 libfftw3-dev numdiff python3-pybind11
COPY . ./Downloads/ism3d
RUN pip install -r ./Downloads/ism3d/requirements_dev.txt
RUN cd ./Downloads/ism3d/ && pip install .
```

(`Dockerfile.dev:1–32`). Built atop the maintainer's `rxastro/casa6:latest`
image, with extra `gfortran`, `libfftw3-dev`, `pybind11`, etc.

### 3.7 Editable install workflow

Per `CONTRIBUTING.rst:60–72`:

```bash
git clone git@github.com:your_name_here/ism3d.git
cd ism3d/
mkvirtualenv ism3d
python setup.py develop   # editable
flake8 ism3d tests        # lint
pytest                    # test
tox                       # multi-py test
```

`tox.ini` declares envs `py35`, `py36`, `py37`, `py38`, `flake8`, with
flake8 running against `ism3d` and `tests`.

### 3.8 Console script

`setup.py:47–51` registers one entry point:

```python
entry_points={'console_scripts': ['ism3d = ism3d.cli:main']}
```

Once installed, `ism3d` is the CLI.

## 4. Build and runtime architecture

### 4.1 ASCII architecture diagram

```
                         ┌────────────────────────────────────┐
                         │              CLI / API             │
                         │   ism3d.cli.main  /  ism3d (entry) │
                         │   ism3d.interface.read_inp(.inp)   │
                         └──────────────┬─────────────────────┘
                                        │ inp_dct (dict)
                                        ▼
                ┌──────────────────────────────────────────────┐
                │  modeling/                                    │
                │   • opt.py        opt_setup, opt_iterate     │
                │     ├ chisq_iterate (Amoeba SA, lmfit)       │
                │     └ emcee_iterate (MCMC)                   │
                │   • evaluate.py   log_prior, log_likelihood, │
                │                   log_probability,           │
                │                   uv_chisq, xy_chisq         │
                │   • model.py      model_setup, model_realize,│
                │                   model_render               │
                │   • analyze.py    opt_analyze, chisq_analyze,│
                │                   emcee_analyze              │
                └──────────────┬───────────────────────────────┘
                               │ obj dicts (geometry+kinematics+SED)
              ┌────────────────┴─────────────────┐
              ▼                                  ▼
   ┌──────────────────┐                  ┌────────────────────┐
   │     arts/        │                  │   simuv/  simxy/   │
   │ (source models)  │                  │   (renderers)      │
   │                  │                  │                    │
   │ apmodel2d        │  →   plane image │ simuv/render.py    │
   │ analytic         │      / cloudlets │  uv_render(...)    │
   │ sparse (clouds_*)│      ──────────► │ simuv/ft.py        │
   │ dynamics (vrot,  │                  │  uv_sample(plane,  │
   │   vcirc, NFW,    │                  │   uvw,...) [nufft, │
   │   exp disk)      │                  │   galario, direct, │
   │ discretize       │                  │   interp2d,        │
   │ lens (SIE)       │                  │   nearest]         │
   └──────────────────┘                  │                    │
                                         │ simxy/render.py    │
                                         │  xy_render, lens,  │
                                         │  apmodel2d, sp3d   │
                                         └────────┬───────────┘
                                                  │ visibilities / cubes
                                                  ▼
                                       ┌──────────────────────┐
                                       │  uvhelper / xyhelper │
                                       │ (CASA-backed I/O)    │
                                       │   ms.read_ms,        │
                                       │   ms.write_ms,       │
                                       │   imager.invert,     │
                                       │   imager.xclean,     │
                                       │   ft.invert_ft       │
                                       └──────────────────────┘
                                                  │
                                                  ▼
                                       ┌──────────────────────┐
                                       │ visualize / utils.io │
                                       │  plt_*, h5/dct, fits │
                                       └──────────────────────┘
```

Cross-cutting concerns:

- **Maths layer (`maths/`)** provides the random-variate machinery used by
  `arts/sparse.py` to draw cloudlet positions.
- **Logger (`logger.py`)** unifies the `ism3d` Python logger with CASA's
  `casalog` sink when CASA is installed.
- **Resource layer (`resource/`)** ships an `input_def.inp` template, a
  default `.cfg` and a FITS header template used by `utils.meta.create_header`.

### 4.2 Top-level package boot — `ism3d/__init__.py`

```python
__version__   = pkg_resources.get_distribution('ism3d').version
__email__     = 'rx.astro@gmail.com'
__author__    = 'Rui Xue'
__credits__   = 'University of Iowa'
__tests__     = <pkg>/tests/
__demo__      = <pkg>/../examples/
__demodata__  = <pkg>/../examples/data/
__resource__  = <pkg>/resource/

from .logger import logger_config, logger_status
logger_config(logfile=None, loglevel='INFO', logfilelevel='INFO', reset=True)

from .utils.misc import check_config

# FFT backend negotiation (in priority order):
try:    import mkl_fft._numpy_fft as fft_use   # Intel MKL backend
except: try:   import pyfftw.interfaces.numpy_fft as fft_use
        except: import scipy.fft as fft_use    # final fallback
```

(`ism3d/__init__.py:9–96`).

The selected backend is exposed as `ism3d.fft_use` and consumed by the
forward-modelling modules via `from .. import fft_use` (e.g.
`simuv/ft.py:17`). `ism3d.fft_fastlen` accompanies it (used in cube
zero-padding).

### 4.3 Logger configuration — `ism3d/logger.py`

Single logger named `'ism3d'`. Functions:

| Function | Purpose |
|----------|---------|
| `logger_config(logfile, loglevel, logfilelevel, reset, log2term)` | Reset and configure the `ism3d` logger; optionally also redirect CASA's `casalog` to the same file via `casalogger_config` |
| `casalogger_config(logfile, loglevel, onconsole, reset)` | Configure `casalog.setlogfile / showconsole / filter` |
| `casalogsink_config(logfile, loglevel, onconsole)` | Marked obsolete |
| `logger_status()` | Pretty-print the current state of the `ism3d` logger, the CASA logger, and the root `logging.Logger.manager.loggerDict` |
| `class CustomFormatter(logging.Formatter)` | A multiline-aware formatter producing `YYYY-MM-DD HH:MM:SS :: <name>.<funcName> :: [LEVEL] :: msg` |

If `casatools` / `casatasks` are present, `logger.py:11–18` immediately deletes
the auto-created CASA log file and points `casalog` at `/dev/null` so it does
not pollute the working directory at import time.

## 5. Public API — key signatures

This section indexes every module-level callable found by
`grep "^def \|^class "` over the package source. Signatures are exact
copy-pastes from the source.

### 5.1 Top-level (`ism3d/`)

| Symbol | Signature / role | Source |
|--------|-----------------|--------|
| `ism3d.fft_use` | Selected FFT backend module (mkl_fft → pyfftw → scipy.fft) | `__init__.py:69–96` |
| `ism3d.fft_fastlen` | Companion `next_fast_len` for the active backend | `__init__.py` |
| `ism3d.__version__`, `__author__`, `__email__`, `__credits__`, `__resource__`, `__tests__`, `__demo__`, `__demodata__` | Package metadata strings | `__init__.py:17–24` |
| `logger_config(logfile=None, loglevel='INFO', logfilelevel='INFO', reset=True, log2term=True)` | Configure the `'ism3d'` logger and CASA logsink | `logger.py:20–68` |
| `casalogger_config(logfile, loglevel, onconsole, reset=False)` | CASA log routing | `logger.py:70–90` |
| `casalogsink_config(logfile, loglevel, onconsole)` | Obsolete CASA logsink helper | `logger.py:92–109` |
| `logger_status()` | Pretty-print logger state | `logger.py:112–126` |
| `class CustomFormatter(logging.Formatter)` | Multi-line-aware formatter | `logger.py:129–148` |
| `cli.main()` | Argparse entry point | `cli.py:49–140` |
| `cli.proc_inpfile(args)` | Workflow runner: dispatch fit/analyze/plot | `cli.py:142–257` |
| `interface.read_inp(parfile, log=False)` | Parse `.inp` parameter file via `configparser.ExtendedInterpolation` + `asteval` | `interface.py:45–80` |
| `interface.key_intepreter(key, value)` | Convert string `xypos` to `astropy.coordinates.SkyCoord` etc. | `interface.py:82–94` |
| `interface.eval_func(vs_func_ps, var_dict)` | Inline lambda-style expression evaluator with `asteval` | `interface.py:96–130` |
| `interface.write_inp(inp_dict, parfile='test,inp', overwrite=True)` | Write the underlying `ConfigParser` back to disk | `interface.py:132–137` |
| `interface.inp_to_mod(inp_dict)` | Resolve `import` cross-references and strip `ism3d.*` admin sections | `interface.py:139–172` |
| `ism3d.ism3d` | Stub `"""Main module."""` — placeholder | `ism3d/ism3d.py:1` |

### 5.2 `arts/` — artificial source generation

| Symbol | Signature | Notes / source |
|--------|-----------|----------------|
| `analytic.analytic_from_apmodels(obj)` | Stub returning `None` | `analytic.py:5–10` |
| `analytic.analytic_from_point(obj)` | Stub | `analytic.py:12` |
| `analytic.xy_render_point(objs, w)` | Stub | `analytic.py:14–15` |
| `analytic.uv_render_point()` | Stub (broken signature `def uv_render_point()`) | `analytic.py:17` |
| `analytic.analytic_from_obj(obj, ...)` | Stub | `analytic.py:24–35` |
| `apmodel2d.get_apmodel2d(obj, px=0, py=0, pscale=1)` | Build an `astropy.modeling.models.*` 2D model from `obj['sbProf']` (supports `airydisk2d`, `box2d`, `const2d`, `ellipse2d`, `disk2d`, `gaussian2d`, `planar2d`, `sersic2d`, `ring2d`, `rickerwavelet2d`) | `apmodel2d.py:5–80` |
| `apmodel2d.eval_apmodel2d(obj, xx, yy, out=None)` | Evaluate the apmodel on a coordinate grid | `apmodel2d.py:82–95` |
| `discretize.clouds_perchan(clouds_loc, clouds_wt, sv, return_v=False)` | Bin cloudlets by LOS velocity into channels using a CSR sparse partial sort | `discretize.py:44–110` |
| `discretize.channel_split(objs, w)` | Split per-object cloudlets into per-channel xy/wt lists; vectorise fluxscale | `discretize.py:111–187` |
| `discretize.lognsigma_lookup(objs, dname)` | Lookup log-normal σ from object metadata | `discretize.py:188–222` |
| `dynamics.model_vcirc(pot_dct)` | Build NFW + razor-thin exp-disk circular-velocity profile via `galpy.potential` | `dynamics.py:29–113` |
| `dynamics.model_vcirc_plot(rc, figname='vcirc_plt.pdf')` | Quick PDF plot of the rotation curve | `dynamics.py:115–128` |
| `dynamics.model_vrot_plot(mod_obj_disk3d, figname='vrot_plt.pdf')` | Plot the disk's projected rotation profile | `dynamics.py:130–146` |
| `dynamics.model_vrot(mod_dct)` | Build the full velocity model from model dict | `dynamics.py:148–211` |
| `dynamics.potential_fromobj(obj)` | Construct a `galpy` potential from an obj dict | `dynamics.py:212–312` |
| `dynamics.calc_vcirc(pot, rho, interp=True, logr=True)` | Compute v_circ on a grid from a potential | `dynamics.py:313–329` |
| `dynamics.pots_to_vcirc(pots, rho, pscorr=None)` | Combine multiple potentials | `dynamics.py:330–366` |
| `dynamics.vrot_from_rcProf(rcProf, rho)` | Evaluate a tabulated rotation curve at radii | `dynamics.py:367–406` |
| `lens.sie_rt(x, y, theta_e=1, xc=0, yc=0, pa=0, q=1.0, method='ls')` | Ray-tracing through a SIE potential, via `lenstronomy` (`'ls'`) or analytic `sie_grad_abs` (`'asb'`) | `lens.py:11–55` |
| `lens.sie_grad_abs(x, y, par)` | Analytic SIE gradient (Bolton 2009 implementation) | `lens.py:57–120` |
| `sparse.cr_tanh(r, r_in=None, r_out=None, theta_out=90*u.deg)` | Peng+2010 Appendix-A profile (with scaling correction) | `sparse.py:41–50` |
| `sparse.clouds_morph(sbProf, fmPhi, fmRho, geRho, bmY, sbQ, rotPhi, sbPA, vbProf, size=100000, seeds=[0,1,2], cmode=1)` | **Core cloudlet generator** — flexible Sersic / expon / norm / table / point / lambda profiles, with Fourier perturbation, axial ratio, vertical profile, etc. | `sparse.py:52–297` |
| `sparse.clouds_kin(car, rcProf=None, ...)` | Attach kinematics to cartesian cloudlets | `sparse.py:298–402` |
| `sparse.clouds_tosky(car, inc, pa, inplace=True)` | Project disk-frame cloudlets to sky-frame | `sparse.py:403–457` |
| `sparse.clouds_from_disk3d(obj, ...)` | Build cloudlets for a disk3d (line) component | `sparse.py:458–498` |
| `sparse.clouds_from_point(obj)` | Cloudlets for a point source | `sparse.py:499–510` |
| `sparse.clouds_from_obj(obj, ...)` | Dispatcher per `obj['type']` | `sparse.py:511–524` |
| `sparse.clouds_discretize_2d(cloudlet, axes=['y','x'], ...)` | Histogram cloudlets onto a 2D grid | `sparse.py:525–540` |
| `sparse.cloudlet_moms(cloudlet, ...)` | Compute moments (mass, vmean, vdisp, etc.) on cloudlet arrays | `sparse.py:541–579` |
| `sparse.clouds_realize(mod_dict, ...)` | Realise full mod_dct into cloudlet sets | `sparse.py:580–662` |
| `utils.fluxscale_from_contflux(contflux, w)` | Build `fluxscale(ν)` vector from `(F0, ν0, α)` continuum spec | `arts/utils.py:13–26` |
| `utils.pix2sky(obj, w, px=None, py=None, pz=None)` | Map pixel index → galactic-rest-frame (kpc, kpc, km/s) for line objects, (kpc, kpc, Hz) for continuum | `arts/utils.py:28–116` |
| `utils.clouds_split(obj, w)` | Build per-object channelised render data: `(fluxscale, xrange, yrange, x, y, wt)` | `arts/utils.py:118–176` |

### 5.3 `maths/`

| Symbol | Description |
|--------|-------------|
| `class sersic2d_gen(rv_continuous)` / `sersic2d` | Axisymmetric 2D Sersic radial PDF (`stats.py:55–79`) |
| `class expon2d_gen(rv_continuous)` / `expon2d` | 2D exponential radial PDF (`stats.py:81–99`) |
| `class norm2d_gen(rv_continuous)` / `norm2d` | 2D Gaussian radial PDF (`stats.py:101–131`) |
| `class sechsq_gen(rv_continuous)` | sech² distribution (`stats.py:133–162`) |
| `class laplace_gen(rv_continuous)` | Laplace distribution (`stats.py:164–185`) |
| `rng_seeded(seed=None)` | Construct a `numpy.random.Generator` seeded with PCG64/SFC64 (`stats.py:186–202`) |
| `custom_rvs(func, ...)` | Generic rvs sampler from a custom callable (`stats.py:203–351`) |
| `custom_sf(func, x, sersic_n=1)` | Custom survival function (`stats.py:352–402`) |
| `custom_pdf(func, x, sersic_n=1)` | Custom PDF evaluator (`stats.py:403–427`) |
| `custom_ppf(func, q, sersic_n=1)` | Custom percent-point function (`stats.py:428–524`) |
| `pdf2rv_nd(pdf, size=100000, ...)` | N-D PDF → rvs sampler (`stats.py:525–563`) |
| `cdf2rv(x, cdf, size=100000, seed=None)` | 1D CDF → rvs (`stats.py:564–590`) |
| `pdf2rv(x, pdf, size=100000, seed=None)` | 1D PDF → rvs (`stats.py:591–605`) |
| `geometry.triangle_area(x1, y1, x2, y2, x3, y3)` | Triangle area helper (`geometry.py:4–9`) |
| `geometry.points_in_triangle(...)` | Point-in-triangle query (`geometry.py:10–42`) |

### 5.4 `simuv/` — uv-domain renderer

| Symbol | Description |
|--------|-------------|
| `render.sample_prep(w, phasecenter, tol=0.01)` | Compute `(dRA, dDec, cell, wv)` from WCS+phasecenter (`render.py:37–76`) |
| `render.uv_render(objs, w, uvw, phasecenter, pb=None, wideband=False)` | Master uv-render: mixes line + continuum, renders cloudlets via `fast_histogram`, then calls `uv_sample` per channel; applies primary beam (`render.py:78–181`) |
| `ft.uv_sample(plane, cell, uu, vv, dRA=0., dDec=0., PA=0, origin='upper', method='nufft', ik=5, saveuvgrid=False)` | Master dispatcher choosing among `nufft`, `nearest`, `galario`, `direct`, `interp2d[-ri/-ap]` (`ft.py:254–342`) |
| `ft.uv_sample_nearest(plane, cell, uu, vv, dRA, dDec, PA, factor=10, saveuvgrid=False)` | Oversampled FFT + nearest-neighbour pickup (`ft.py:33–84`) |
| `ft.uv_sample_nufft(plane, cell, uu, vv, vis, dRA=0., dDec=0., PA=0)` | Type-2 NUFFT via `finufftpy.nufft2d2` (`ft.py:86–109`) |
| `ft.uv_sample_direct(plane, cell, uu, vv, dRA, dDec, PA, dimsum='uv', drange=10)` | Brute-force DFT over CSR-filtered pixels (`ft.py:111–171`) |
| `ft.uv_sample_interp2d(plane, cell, uu, vv, dRA, dDec, PA, origin='upper', mode='ap', ik=5, saveuvgrid=False)` | Spline interpolation in the FFT grid; supports re-im or amp-phase modes (`ft.py:173–252`) |

### 5.5 `simxy/` — xy-domain renderer

| Symbol | Description |
|--------|-------------|
| `render.render_spmodel3d_xyz(obj)` | Histogram cloudlets into a 3D `(z,y,x)` cube (`render.py:79–98`) |
| `render.render_spmodel3d(obj, w, out=None)` | Render a sparse model onto a WCS-defined cube (`render.py:100–132`) |
| `render.render_lens(obj, cube, w)` | Apply lensing ray-tracing to a model cube (`render.py:133–170`) |
| `render.render_apmodel2d(obj, w, out=None, normalize=True)` | Render an analytic 2D model onto an output plane (`render.py:171–235`) |
| `render.xy_render(objs, w, psf=None, pb=None, normalize_kernel=False)` | Image-cube forward model: cloudlet histogramming + per-channel convolution; multiplies by primary beam if given (`render.py:236–507`) |

### 5.6 `modeling/`

| Symbol | Description |
|--------|-------------|
| `model.model_setup(mod_dct, dat_dct, verbose=False)` | One-shot allocator: build per-dataset WCS / sampling header / output containers (`model.py:92–280`) |
| `model.model_realize(mod_dict, ...)` | Realise the parameter dict into concrete cloudlets/apmodels (`model.py:281–383`) |
| `model.model_render(mod_dct, dat_dct, models=None, ...)` | Run `simuv.uv_render` / `simxy.xy_render` over all components and stash results into `models` (`model.py:384–438`) |
| `evaluate.log_prior(theta, fit_dct)` | Uniform priors over `fit_dct['p_lo']` ↔ `fit_dct['p_up']` (`evaluate.py:51–61`) |
| `evaluate.log_likelihood(theta, fit_dct, inp_dct, dat_dct, ...)` | -χ²/2 log-likelihood (`evaluate.py:62–128`) |
| `evaluate.log_probability(theta, ...)` | `log_prior + log_likelihood` for emcee (`evaluate.py:129–148`) |
| `evaluate.model_eval(theta, fit_dct, inp_dct, dat_dct, ...)` | Single-point model+chi² evaluation (`evaluate.py:149–219`) |
| `evaluate.calc_lnprob2_initializer(dat_dct, models)` | mp.Pool initialiser pinning shared dat/models (`evaluate.py:220–229`) |
| `evaluate.calc_lnprob2(p, fit_dct, inp_dct)` | Per-walker likelihood (multiprocessing-friendly) (`evaluate.py:230–249`) |
| `evaluate.calc_lnprob(p, fit_dct, inp_dct)` | Plain log-prob (`evaluate.py:250–269`) |
| `evaluate.calc_chisq(p, ...)` | χ² for amoeba-SA / lmfit (`evaluate.py:270–302`) |
| `evaluate.calc_wdev(p, ...)` | Weighted residual vector for least-squares minimisers (`evaluate.py:303–367`) |
| `evaluate.uv_chisq(objs, dname, dat_dct, models)` | Visibility-domain χ² over MS data (`evaluate.py:368–492`) |
| `evaluate.xy_chisq(objs, dname, dat_dct, models, returnwdev=False)` | Image-domain χ² (`evaluate.py:493–627`) |
| `evaluate.xy_chisq0(objs, dname, dat_dct, models)` | Legacy/baseline image χ² implementation (`evaluate.py:628–708`) |
| `opt.opt_setup(inp_dct, dat_dct, initial_model=False, copydata=False)` | Pick fitter, build `fit_dct` (parameter vector, bounds, priors) + populate `models` (`opt.py:45–263`) |
| `opt.opt_iterate(fit_dct, inp_dct, dat_dct, models, resume=False)` | Dispatcher → calls `chisq_iterate` / `emcee_iterate` (`opt.py:264–279`) |
| `opt.chisq_iterate(fit_dct, inp_dct, dat_dct, models, nstep=20, resume=None)` | Amoeba-SA + lmfit minimiser loop (`opt.py:280–413`) |
| `opt.emcee_iterate(fit_dct, inp_dct, dat_dct, models, nstep=100, resume=False)` | emcee MCMC sampler with HDF5 backend (`opt.py:414–531`) |
| `opt.amoeba_sa(func, p0, scale, ...)` | Numerical-Recipes-style downhill simplex with simulated annealing (`opt.py:532–679`) |
| `opt.amotry_sa(func, p, psum, ihi, fac, y, ...)` | Amoeba helper move (`opt.py:680–705`) |
| `analyze.opt_analyze(inpfile, burnin=None, copydata=True, export=False)` | Dispatcher: open output folder, call `chisq_analyze` / `emcee_analyze` / `brute_analyze` (`analyze.py:32–119`) |
| `analyze.chisq_analyze(outfolder, burnin=None)` | Post-process amoeba-SA outputs (`analyze.py:120–265`) |
| `analyze.emcee_analyze(outfolder, ...)` | Build corner plots, posterior summaries, write `mod_dct` (`analyze.py:266–544`) |
| `analyze.brute_analyze(outfolder)` | Brute-grid analysis (`analyze.py:545–718`) |

### 5.7 `uvhelper/`

| Symbol | Description |
|--------|-------------|
| `cli.casatools_repack()` | Helper to repack `casatools` (legacy CLI) (`cli.py:18–104`) |
| `imager.ext_list()` | Return list of `tclean` output suffixes (`imager.py:24–36`) |
| `imager.invert(vis, imagename, datacolumn='data', antenna='', weighting='briggs', robust=1.0, npixels=0, cell=0.04, imsize=[128,128], phasecenter='', specmode='cube', start='', width='', nchan=-1, perchanweightdensity=True, restoringbeam='', onlydm=False, pbmask=0, pblimit=0, exclude_list=[...], **kwargs)` | Generate a quick dirty image from an MS via `casatasks.tclean` with `niter=0` (`imager.py:38–95`) |
| `imager.copyimages(imagename, ...)` | Copy/rename a tclean image set (`imager.py:96–109`) |
| `imager.exportimages(imagenames, ...)` | Export FITS via `casatasks.exportfits` (`imager.py:110–161`) |
| `imager.xclean(vis, imagename, ...)` | Multi-pass `tclean` wrapper with auto-masking and FITS export (`imager.py:162–254`) |
| `ft.invert_ft(uu, vv, wv, vis, wt, flag, uvdata, ...)` | Image visibilities directly via NUFFT (`ft.py:11–88`) |
| `ft.make_psf(**kwargs)` | Generate the PSF image (`ft.py:89–99`) |
| `ft.advise_header(uv, center, chanfreq, chanwidth, ...)` | Build a FITS header consistent with the uv coverage (`ft.py:100–161`) |
| `ft.advise_imsize(u, v, pb=0, f_min=5., f_max=2.0, even=True)` | Recommend an image size given uv extent and primary beam (`ft.py:162–220`) |
| `ms.read_ms(vis, polaverage=True, flagdata=False, saveflag=True, includedata=True, usedouble=False, dataset=None, keyrule='basename')` | Read DATA/FLAG/WEIGHT/UVW from MS into a dict; supports `polaverage` (Stokes I from XX/YY or RR/LL), bool-flag preservation; uses `casatools.table` (`ms.py:44–242`) |
| `ms.write_ms(vis, value, ...)` | Write back model column to MS (`ms.py:243–300`) |
| `ms.rmPointing(outvis, verbose=False)` | Drop POINTING subtable (`ms.py:301–320`) |
| `ms.rmColumns(vis, column='')` | Drop arbitrary column (`ms.py:321–336`) |
| `ms.checkchflag(vis)` | Per-channel flagging diagnostic (`ms.py:337–387`) |
| `ms.getfreqs(vis, frame='LSRK', spwids=[0], edge=0)` | Return chan freqs in requested frame using `msmetadata` (`ms.py:388–450`) |
| `ms.getcommonfreqs(vis_list, spw_list, edge_list=None, frame='LSRK', chanbin=2)` | Common frequency grid across multiple MSes (`ms.py:451–481`) |
| `ms.flagbywt(vis, datacolumn='data', fitspw='')` | Sigma-clip flagging based on WEIGHT (`ms.py:482–511`) |
| `ms.flagrow(vis)` | Row-level flagging (`ms.py:512–543`) |
| `ms.flagchan()` | Channel-level flagging stub (`ms.py:544–553`) |
| `proc.casa_version()` | Report CASA version (`proc.py:11–17`) |
| `proc.plotuv_freqtime_amp(vis='', spw=[''], xaxis='freq')` | Wrapper around `casaplotms` (`proc.py:18–32`) |
| `proc.rawSelect(name, correlation='RR,LL', keepflags=False, datacolumn='data')` | Quick row selection (`proc.py:33–44`) |
| `vis_utils.corrupt_ms(vis, ...)` | Apply visibility-noise corruption (`vis_utils.py:42–66`) |
| `vis_utils.cpredict_ms(vis, ...)` | Predict visibilities into MODEL_DATA via CASA (`vis_utils.py:67–113`) |
| `vis_utils.gpredict_ms(vis, fitsimage=None, inputvis=None, pb=None, pbaverage=True, antsize=None, ...)` | Predict via galario from a FITS image (`vis_utils.py:114–240`) |

### 5.8 `xyhelper/`

| Symbol | Description |
|--------|-------------|
| `cube.hextract(data, header, ss)` | Extract a sub-cube and update header CRPIX/CRVAL accordingly (`cube.py:1–45`) |
| `sky.linear_offset_coords(wcs, center)` | Return a tangent-plane WCS centred at `center` (`sky.py:5–38`) |
| `sky.calc_ppbeam(header)` | Pixels-per-beam computed from BMAJ/BMIN and pixel scale (`sky.py:39–50`) |

### 5.9 `visualize/`

| Symbol | Description |
|--------|-------------|
| `plts.plt_rc0(pots, pscorr=None, ...)` | Plot rotation-curve components from a list of `galpy` potentials (`plts.py:38–71`) |
| `plts.plt_rcProf(rcProf, ...)` | Plot a tabulated rotation curve (`plts.py:72–115`) |
| `plts.im_grid(images, header, ...)` | Mosaic grid of FITS thumbnails (`plts.py:116–191`) |
| `plts.plt_spec1d(fn, roi='icrs; circle(...)')` | Extract and plot a 1D spectrum within a DS9-region ROI (`plts.py:192–314`) |
| `plts.plt_yt3d(fn, roi='...')` | yt-based 3D rendering (`plts.py:315–355`) |
| `plts.plt_makeslice(fn, ...)` | Position-velocity slice maker (`plts.py:356–412`) |
| `plts.plt_slice(fn, i=1)` | Plot a previously made slice (`plts.py:413–511`) |
| `plts.plt_mom0xy(fn, linechan=None)` | Moment-0 sky panel (`plts.py:512–894`) |
| `plts.plt_radprof(fn)` | Radial profile plot (`plts.py:895–962`) |
| `msplot2.uvamp(uvdist, ...)` | Plot |V|(uv-distance) with binning (`msplot2.py:3–92`) |
| `msplot2.uvamp_average(uvdist, uvdata, bins=20, plot=False)` | Bin-averaged uv-amplitude (`msplot2.py:93–233`) |
| `nb.show_gif(fname)` | Inline-display a GIF in Jupyter (`nb.py:6–21`) |
| `nb.make_gif(fignames, gifname)` | Build an animation from PNGs via `imageio` (`nb.py:22–28`) |

### 5.10 `utils/`

`utils/__init__.py` only does `from .io import *`, so `from ism3d.utils import *`
re-exports the I/O surface plus everything that `io.py` itself imports
(notably `from .utils import *`).

`utils/io.py`:

| Symbol | Description |
|--------|-------------|
| `read_data(inp_dct, save_data=False, fill_mask=False, fill_error=False, polaverage=True, dataflag=False, saveflag=True)` | One-shot loader that, for every section in `inp_dct`, reads MS visibilities (`vis`) and/or FITS images (`image`/`mask`/`pb`/`error`/`sample`/`psf`) into `dat_dct` (`io.py:24–208`) |
| `dct2npy(dct, outname='dct2npy')` / `npy2dct(npyname)` | NPY round-trip (`io.py:209–216`) |
| `to_hdf5(value, outname='test.h5', checkname=False, **kwargs)` / `from_hdf5(h5name)` | hickle-based HDF5 IO (`io.py:217–266`) |
| `dct2hdf(dct, outname='dct2hdf')` / `hdf2dct(hdf)` | Bulk dict↔HDF5 (`io.py:267–304`) |
| `fits2dct(fits)` / `dct2fits(dct, outname='dct2fits')` | FITS↔dict export (`io.py:305–373`) |
| `export_model(models, outdir='./')` | Save model dict (cubes, vis, residuals) to disk (`io.py:374–550`) |

`utils/utils.py` (most of the heavy lifting; first imports include
`emcee`, `galario.single`, `pyfftw`, `spectral_cube.SpectralCube`,
`radio_beam.Beams`, etc.):

| Symbol | Description |
|--------|-------------|
| `arithmeticEval(s)` | Sandbox numeric evaluator (commented out — see source comments at `utils.py:62–87`) |
| `repr_parameter(v)` | repr-with-units for `Quantity`, `SkyCoord`, list, tuple (`utils.py:96–129`) |
| `pprint(*args, **kwargs)` | logger-routed `pprint` wrapper (`utils.py:130–141`) |
| `read_range(center=0, delta=0, mode='a')` | Compute a range from centre/delta in absolute or relative mode (`utils.py:142–156`) |
| `moments(imagename, outname='test', ...)` | Compute moment maps with CASA `immoments` (`utils.py:157–175`) |
| `imcontsub(imagename, linefile='', contfile='', ...)` | Continuum subtraction via CASA `imcontsub` (`utils.py:176–205`) |
| `gal_flat(im, ang, inc, cen=None, interp=True, ...)` | De-project a galaxy image (`utils.py:206–267`) |
| `sort_on_runtime(p)` | emcee load-balancing helper (`utils.py:268–273`) |
| `gmake_listpars(objs, showcontent=True)` | Pretty-print parameter dicts (`utils.py:274–291`) |
| `paste_slice(tup)` / `paste_array(wall, block, loc, method='replace')` | Compose a `block` ndarray onto a larger `wall` with bounds clipping; supports `replace` / `add` (`utils.py:292–338`) |
| `make_slice(expr)` | Parse a slice-string into `slice` objects (`utils.py:339–350`) |
| `read_par(inp_dct, par_name, to_value=False)` | Read a `keyword@section` parameter, supports `[i]`/`[i:j]` indexing (`utils.py:351–394`) |
| `write_par(inp_dct, par_name, par_value, verbose=False)` | Write/update a parameter in-place (`utils.py:395–458`) |
| `inp_validate(inp_dct, verbose=False)` | Validate an `inp_dct` against the resource defaults (`utils.py:459–498`) |
| `obj_defunit(obj)` | Strip Quantity → float in obj internal units (`utils.py:499–522`) |
| `gmake_pformat(fit_dct)` | Format the parameter table (`utils.py:523–590`) |
| `get_dirsize(dir)` | Recursive directory size (`utils.py:591–600`) |
| `h5ls_print(name, obj)` / `h5ls(filename, logfile=None)` | HDF5 tree printer (`utils.py:601–631`) |
| `set_threads(num=None)` | Set thread count for `pyfftw`, `galario`, `mkl`, OpenMP env vars (`utils.py:632–667`) |
| `backup(filename, move=True)` | Backup an existing file with timestamp (`utils.py:668–681`) |
| `get_autocoor_time(h5name)` | Pull autocorrelation time from emcee HDF5 backend (`utils.py:682–689`) |
| `rotate_xy(x, y, angle, xo=0, yo=0)` | Rotate (x,y) by `angle` around (xo,yo) (`utils.py:690–711`) |
| `one_beam(fitsname)` | Get a single representative beam from a FITS cube (`utils.py:712–724`) |
| `sample_grid(spacing, xrange=[-100,100], yrange=[-100,100], center=None, ...)` | Hex sampling grid generator (`utils.py:725–769`) |
| `chi2red_to_lognsigma(chi2red)` | Convert reduced χ² to a log-normal σ for likelihood scaling (`utils.py:770–774`) |

`utils/misc.py`:

| Symbol | Description |
|--------|-------------|
| `check_config()` | Print env summary (Python version, host, CPU count, memory), then `check_deps()` and `check_fftpack()` (`misc.py:23–37`) |
| `check_deps(package_name='ism3d')` | Iterate `pkg_resources` requires/installed (`misc.py:41–53`) |
| `check_fftpack()` | Report active FFT backend (`misc.py:55–60`) |
| `convert_size(size_bytes)` | Bytes → KB/MB/... (obsolete; use `human_unit/human_to_string`) (`misc.py:62–72`) |
| `unit_shortname(unit, nospace=True, options=False)` | Compact unit string (`misc.py:74–103`) |
| `human_unit(quantity, return_unit=False, base_index=0, scale_range=None)` | Auto-rescale a `Quantity` to a "human" unit (`misc.py:104–175`) |
| `human_to_string(q, format_string='{0.value:0.2f} {0.unit:shortname}', nospace=True)` | Human-readable Quantity string (`misc.py:176–212`) |
| `get_obj_size(obj, to_string=False)` | Recursive `sys.getsizeof` (`misc.py:213–240`) |
| `prepdir(filename)` | mkdir-p for the parent of `filename` (`misc.py:241–248`) |
| `render_component(out, im, scale=1, mode='iadd')` | Operator-dispatched in-place additive renderer (`misc.py:249–297`) |
| `pickplane(im, iz)` | Slice a plane from a cube, with broadcasting fallback (`misc.py:298–311`) |
| `makepsf(header, ...)` | Build a Gaussian PSF from header BMAJ/BMIN/BPA (`misc.py:312–366`) |
| `makepb(header, phasecenter=None, antsize=12*u.m)` | Build an Airy primary beam (`misc.py:367–398`) |
| `makekernel(xpixels, ypixels, beam, pa=0., cent=None, ...)` | Generate a 2D convolution kernel (`misc.py:399–461`) |

`utils/meta.py`:

| Symbol | Description |
|--------|-------------|
| `inp_config(file=None)` | Build the in-memory `ConfigParser` describing reserved section ids (`comment`, `general`, `optimizer`, `analyzer`, `dynamics`, `component`) — used by `inp_to_mod` (`meta.py:52–81`) |
| `create_header(file=None, objname=None, naxis=None, crval=None, crpix=None, cdelt=None)` | Build a 4-axis FITS header (RA/Dec/Freq/Stokes) from the embedded template or a file; auto-update CRPIX/CRVAL/CDELT (`meta.py:83–196`) |
| `db_global` | Module-level dict `{'dat_dct':{}, 'models':{}}` used as a process-wide scratchpad (`meta.py:46–49`) |

`utils/cli_new.py`: 19-line skeleton for a future click-based CLI (currently
just `def main()` returning `None`).

## 6. Input file (`.inp`) format

ism3d's primary user-facing artefact is a `.inp` parameter file. It is parsed
by `interface.read_inp` (`ism3d/interface.py:45–80`):

```python
cfg = configparser.ConfigParser(interpolation=ExtendedInterpolation())
cfg.optionxform = str         # case-sensitive keys
cfg.read(parfile)
inp_dict = deepcopy(cfg._sections)

for section in inp_dict:
    for key in inp_dict[section]:
        expr  = inp_dict[section][key]
        value = aeval(expr)              # asteval interpreter
        if len(aeval.error)>0 and value is None:
            value = [aeval(e) for e in expr.split()]
        value = key_intepreter(key, value)
        inp_dict[section][key] = value
inp_dict['ism3d.inp'] = cfg
```

Key features:

- **Section headings** use the standard `[name]` form (so the file is a
  superset of plain INI).
- **Values are evaluated as Python expressions** through `asteval.Interpreter`
  (`interface.py:27`) with these names pre-loaded into the symbol table:
  `u` (`astropy.units`), `SkyCoord`, `Angle`, `Number`, `Quantity`, `np`.
  Hence values such as `12*u.km/u.s` or `SkyCoord('00h42m30s', '+41d12m00s', frame='icrs')`
  work natively.
- **`xypos` special-casing** in `key_intepreter` (`interface.py:82–94`):
  strings become `SkyCoord(value, frame='icrs')`; tuples/lists become
  `SkyCoord(ra, dec, frame='icrs', unit='deg')`.
- **`ExtendedInterpolation`** allows `${section:key}` references between
  sections.
- **Cross-section import** is handled by `interface.inp_to_mod`
  (`interface.py:139–172`): a key named `import` whose value is a comma-list of
  section names will copy keys from those sections into the current one,
  then those source sections are deleted from the model dict (along with any
  section starting with `ism3d.`).

### 6.1 Reserved section ids

From `utils/meta.inp_config()` (`meta.py:52–81`):

| Section family | Aliases |
|----------------|---------|
| `inp.dynamics` | `gravity, potential, dynamics` |
| `inp.optimizer` | `optimize, fitter, optimizer` |
| `inp.analyzer` | `analysis, diagnostics, analyzer` |
| `inp.comment` | `comments, skip, ignore, changelog` |
| `inp.general` | `general` |
| `inp.component` | (default — any other section is treated as a model component) |

### 6.2 Parameter dictionary template (`resource/input_def.inp`)

`ism3d/resource/input_def.inp` is the reference of all keywords with their
defaults, format strings, types and units. Selected entries:

| Keyword | Default | Type / Unit | Purpose (paraphrased) |
|---------|---------|-------------|------------------------|
| `object`, `name`, `note` | `''` | str | Free-text annotations |
| `image`, `error`, `mask`, `psf`, `samp`, `vis`, `pmodel` | `''` | str (comma-list) | Data file paths |
| `bmaj`, `bmin`, `bpa` | None | Quantity (arcsec/arcsec/deg) | Override beam |
| `xypos` | `[0,0]` | SkyCoord or `[ra, dec]` deg | Component centre |
| `xypos_kin` | `[0,0]` | as above | Kinematic centre if different |
| `z` | `0.0` | Number | Redshift |
| `vsys`, `vsini` | `0.0 km/s` | Quantity | Systemic velocity |
| `sbrp`, `sbser` | `[1.0, 1.0]` | `[Quantity, Number]` | Sersic radial profile params |
| `ge_pa`, `ge_q` | `90 deg`, `2.0` | Quantity / Number | Geometry rotation / axial ratio |
| `pa`, `inc` | `0 deg` | Quantity | Disk position-angle / inclination |
| `restwave`, `restfreq` | `1000 Å`, `100 GHz` | Quantity | Line rest wavelengths |
| `type` | `'disk3d'` | str | One of `disk2d`, `disk3d`, `point`, `apmodel` |
| `sbvp` | `[None, 0]` | `[str, Number]` | Vertical profile (e.g. `('exp', hz)`, `('sech', hz)`, `('gaussian', hz)`) |

(See the full list in `ism3d/resource/input_def.inp` for kinematic, dynamics,
fitter, and analyser keywords.)

### 6.3 Resource files

- `ism3d/resource/default.cfg` — reserved-section id table (mirror of
  `inp_config()` source).
- `ism3d/resource/input_def.inp` — keyword reference template.
- `ism3d/resource/obselete.xymodel.header` — older FITS header template
  (now superseded by the embedded string in `meta.create_header`).

---

## 7. Core algorithms

### 7.1 uv-domain forward model (`simuv.uv_render` + `simuv.uv_sample`)

Given an observation WCS `w` (4-D RA/Dec/Freq/Stokes), a list of objects with
already-realised cloudlets / apmodels, an `uvw` array (rows × 3, in metres),
a phase centre, and an optional primary-beam cube `pb`:

1. `sample_prep(w, phasecenter)` derives `(dRA, dDec, cell, wv)`:
   - shifts `(dRA, dDec)` so the image centre coincides with the phase centre;
   - converts pixel scale to radians;
   - extracts wavelengths `wv[ν]` (m) for each spectral plane.
2. `channel_split(objs, w)` flattens each component into per-channel
   `(x, y, weights)` lists plus a `fluxscale(ν)` vector.
3. **Continuum branch:** for each continuum object, histogram cloudlet (x,y)
   with `fast_histogram.histogram2d` into the image plane, multiply by `pb`
   if available, and call `uv_sample` once at the *band-centre* wavelength —
   caching the result for fast scaling per channel.
4. **Line + assembly branch:** for every spectral channel `iz`,
   - render line cloudlets (those with `lineflux` keyword) into `plane`;
   - composite cached continuum images via `render_component` scaled by
     `fluxscale(iz)`;
   - multiply by `pb[iz]`;
   - run `uv_sample(plane, cell, uvw[:,0]/wv[iz], uvw[:,1]/wv[iz], dRA, dDec, ...)`
     to produce model visibilities for that channel.
5. Returns `vis` with shape `(nrows, nchan)`, `complex128`, Fortran-ordered
   (`render.py:97`).

### 7.2 `uv_sample` dispatcher

`simuv/ft.py:254-342`. The chosen `method` controls which kernel runs:

| `method` | Implementation | Notes |
|----------|----------------|-------|
| `'nearest'` | `uv_sample_nearest` | Oversampled rfft2 + nearest pickup; default `factor=20` (oversampling) |
| `'nufft'` (default) | `uv_sample_nufft` via `finufftpy.nufft2d2` | Type-2 NUFFT, `eps=1e-3`, `upsampfac=1.25` |
| `'galario'` | `galario.single.sampleImage` | Backwards-compat path |
| `'direct'` | `uv_sample_direct` | Brute DFT on CSR-thinned bright pixels |
| `'interp2d'` / `'interp2d-ri'` / `'interp2d-ap'` | `uv_sample_interp2d` | rfft2 + `RectBivariateSpline` (k=5) on real/imag or amp/phase |

The long docstring in `ft.py:255–311` enumerates trade-offs in accuracy
(linear interpolation vs. high-order spline; amplitude/phase vs. real/imag),
phase-shift handling (always done in uv-space rather than image-shifting),
and image-size advice (image size > 2× source size).

### 7.3 Cloudlet-domain galaxy model (`arts.sparse.clouds_morph`)

`sparse.py:52–297` is the heart of geometry generation. Given a face-on
surface-brightness profile keyword `sbProf` (`('sersic2d', r_e, n)`,
`('expon2d', r_s)`, `('norm2d', r_sigma)`, `('table', rho_q, sb_q)`,
`('point',)`, or a lambda string `('rho : maximum(1-rho/p1, 0)', 5*u.kpc)`),
it draws `size` random radii and azimuths via `pdf2rv` / custom rvs from
`maths.stats`. Optional manipulators applied in order:

- `fmPhi=(mode, amplitude, phi_m)` — Fourier perturbation along the
  azimuth (mean SB preserved).
- `fmRho` / `geRho` — radial Fourier mode + boxiness coefficient (Peng+2010).
- `bmY=(mode, amplitude)` — bending mode.
- `sbQ` — axis ratio b/a (deterministic stretch).
- `rotPhi` — Peng+2010 alpha-tanh / log-tanh coordinate rotation (used to
  produce spirals).
- `sbPA` — overall counter-clockwise rotation.
- `vbProf` — vertical profile (`('sech', zh)`, `('sech2', zh)`,
  `('laplace', zs)`, or `None` for razor-thin).
- `cmode` — per-cloud weighting scheme; values follow GIPSY's `galmod` and
  BBarolo conventions (see `https://www.astro.rug.nl/~gipsy/tsk/galmod.dc1`,
  `sparse.py:116–120`).

Output is a triple of cylindrical-frame coordinates plus weights; the call
chain `clouds_kin → clouds_tosky` then attaches velocities and projects to
the sky-frame (PA, inclination).

### 7.4 Kinematic model (`arts.dynamics`)

`model_vrot` (`dynamics.py:148–211`) constructs the rotation curve at radii
`rho` from a `mod_dct` entry, by combining contributions from:

- NFW dark halo (`galpy.potential.NFWPotential`) parameterised by virial
  mass `nfw_mvir`; concentration is interpolated from a Klypin+2011-style
  table at the input redshift (`dynamics.py:50–80`).
- Razor-thin exponential disk (`galpy.potential.RazorThinExponentialDiskPotential`),
  with `disk_sd`, `disk_rs`.
- Optional Miyamoto-Nagai, Kepler, generic potentials added through
  `potential_fromobj`.

The combined `vcirc(rho) = sqrt(Σ vc_i²)` then feeds `vrot_from_rcProf` into
`clouds_kin`.

### 7.5 Optimisation algorithms (`modeling.opt`)

Three optimisers share the same `fit_dct` parameter table:

1. **Amoeba simulated annealing** — `amoeba_sa` (Numerical Recipes
   `amotry_sa`). `chisq_iterate` runs it on `calc_chisq` until convergence
   or `nstep` is hit. Returns best-fit `p` and trace `chisq[]`.
2. **`lmfit.minimize`** — under the same `chisq_iterate` umbrella, it can
   use `'leastsq'`, `'least_squares'`, `'nelder'`, `'brute'` (selection via
   the input `[optimize]` section).
3. **emcee MCMC** — `emcee_iterate` runs `emcee.EnsembleSampler` with the
   `log_probability` evaluator. Backends use HDF5 (`emcee.backends.HDFBackend`)
   so runs can be resumed (`opt.py:414–531`). Multi-processing uses a
   `fork` `Pool` (`opt.py:41`).

### 7.6 Image-domain forward model (`simxy.xy_render`)

`render.py:236–507`. Mirrors `simuv.uv_render` but produces sky-plane cubes,
applying `astropy.convolution.convolve_fft` per channel with a beam kernel
built by `utils.misc.makekernel` from header BMAJ/BMIN/BPA. Continuum
components are rendered once and broadcast across channels with frequency-
dependent flux-scale.

---

## 8. Input / Output formats

| Domain | Reader | Writer |
|--------|--------|--------|
| Measurement Set (CASA) | `uvhelper.ms.read_ms` (uses `casatools.table` + `msmetadata`) | `uvhelper.ms.write_ms` |
| FITS image / cube | `astropy.io.fits.getdata`/`utils.io.read_data` | `utils.io.dct2fits`, `casatasks.exportfits` (in `imager.exportimages`) |
| HDF5 | `utils.io.from_hdf5` / `hdf2dct` (via `hickle`) | `utils.io.to_hdf5` / `dct2hdf` |
| NumPy `.npy` | `utils.io.npy2dct` | `utils.io.dct2npy` |
| Parameter file (`.inp`) | `interface.read_inp` | `interface.write_inp` |
| FITS header template | `utils.meta.create_header` | (built-in template string) |

Per-record MS shape conventions (from `utils/io.py:34–50`):

- `casatools.table.getcol`: `ncorr × nchan × nrecord` (CASA convention)
- `casacore.table.getcol`: `nrecord × nchan × ncorr` (the opposite!)
- DATA dtype: `complex64`
- Flagged values are set to `np.nan` in the dict-form data when
  `flagdata=True`.
- `polaverage=True` collapses XX/YY (or RR/LL) → I, mirroring CASA tclean's
  `stokes='I'` behaviour, including the rule that if either correlation is
  flagged, the result is flagged.

## 9. File-by-file breakdown

### 9.1 `ism3d/__init__.py` (96 LOC)

- Sets `__version__`, `__email__`, `__author__`, `__credits__`,
  `__resource__`, `__demo__`, `__demodata__`, `__tests__`.
- Calls `logger_config(...)` immediately on import.
- Registers `check_config` from `utils.misc`.
- Negotiates the FFT backend in priority order
  (`mkl_fft._numpy_fft → pyfftw.interfaces.numpy_fft → scipy.fft`).
- A large block of legacy imports is commented out (lines 33–67)
  showing the historical "gmake" structure before the rename.

### 9.2 `ism3d/cli.py` (257 LOC)

- Argparse CLI with flags `-f/--fit`, `-a/--analyze`, `-p/--plot`,
  `-d/--debug`, `-t/--test`, `-l/--logfile`, plus a positional `inpfile`.
- `main()` reads the inp, sets up the logfile under `inp_dct['general']['outdir']`,
  configures logging, then calls `proc_inpfile(args)`.
- `proc_inpfile`:
  - `--fit` → `read_data → opt_setup → opt_iterate(resume=False)`
  - `--analyze` → `opt_analyze(args.inpfile)` plus an extensive (commented-out)
    block that historically called `casa_proc.casa_task('ms2im', ...)` to image
    model and data MS files.
  - `--plot` → iterate `data_b?_bb?.fits` files in
    `outdir/p_*/` and call `plt_spec1d`, `plt_mom0xy`, `plt_makeslice`,
    `plt_slice` (i=1,2), `plt_radprof` for each. Hard-coded `bx610` settings
    at lines 226–245 (RA/Dec, PA, slice width/length, line channels).

### 9.3 `ism3d/ism3d.py` (1 LOC)

Single-line module: `"""Main module."""` — placeholder, kept to satisfy
the `tests/test_ism3d.py` import (`from ism3d import ism3d`).

### 9.4 `ism3d/interface.py` (420 LOC)

- `read_inp` (live), `write_inp`, `inp_to_mod`, `key_intepreter`,
  `eval_func` (covered above).
- ~250 lines of commented-out legacy code preserved as historical reference
  for the older line-oriented `@section` parameter format and the `inp2mod`
  recursive cross-reference resolver.

### 9.5 `ism3d/logger.py` (147 LOC)

Already detailed in §4.3. Notable:

- At import time, kills the auto-created CASA log file (lines 14–18).
- `CustomFormatter` outputs one record per source-line (multi-line aware),
  in the format
  `<datetime> :: <name>.<funcName> :: [LEVEL] :: msg`.

### 9.6 `ism3d/arts/`

| File | LOC | Purpose |
|------|-----|---------|
| `__init__.py` | 0 | empty namespace |
| `analytic.py` | 36 | Stubs for analytic-only renderers (`xy_render_point`, `uv_render_point` are skeletons) |
| `apmodel2d.py` | 95 | Build `astropy.modeling.models` 2D models from `obj['sbProf']` keyword |
| `discretize.py` | 223 | CSR-based per-channel cloudlet binning, `channel_split`, `lognsigma_lookup` |
| `dynamics.py` | 407 | Rotation curves, NFW + exp-disk potentials, `vrot_from_rcProf` |
| `lens.py` | 120 | SIE ray-tracing (`lenstronomy` and analytic Bolton 2009) |
| `sparse.py` | 662 | The cloudlet engine — `clouds_morph`, `clouds_kin`, `clouds_tosky`, `clouds_realize`, `cloudlet_moms` |
| `utils.py` | 175 | `pix2sky`, `clouds_split`, `fluxscale_from_contflux` |

### 9.7 `ism3d/maths/`

| File | LOC | Purpose |
|------|-----|---------|
| `__init__.py` | 0 | empty namespace |
| `geometry.py` | 42 | Triangle area + point-in-triangle (used by sparse) |
| `stats.py` | 605 | Custom `scipy.stats.rv_continuous` subclasses (`sersic2d`, `expon2d`, `norm2d`, `sechsq`, `laplace`), `pdf2rv`, `cdf2rv`, `pdf2rv_nd`, `custom_rvs/pdf/sf/ppf`, `rng_seeded` |

### 9.8 `ism3d/modeling/`

| File | LOC | Purpose |
|------|-----|---------|
| `__init__.py` | 0 | namespace placeholder |
| `analyze.py` | 719 | Post-fit analysis: corner plots, parameter summaries, residual cubes (`emcee_analyze`, `chisq_analyze`, `brute_analyze`) |
| `evaluate.py` | 709 | Likelihood/χ² machinery: `log_prior`, `log_likelihood`, `log_probability`, `calc_chisq`, `calc_wdev`, `uv_chisq`, `xy_chisq`, `xy_chisq0`, mp.Pool initialiser |
| `model.py` | 439 | `model_setup`, `model_realize`, `model_render` |
| `opt.py` | 705 | Optimiser dispatch and runner: `opt_setup`, `opt_iterate`, `chisq_iterate`, `emcee_iterate`, `amoeba_sa`, `amotry_sa` |

`modeling/__init__.py` is empty, so all symbols must be imported via the
fully qualified path (e.g. `from ism3d.modeling.opt import opt_setup`).

### 9.9 `ism3d/simuv/`

| File | LOC | Purpose |
|------|-----|---------|
| `__init__.py` | 0 | empty namespace |
| `ft.py` | 345 | Five `uv_sample_*` kernels + the `uv_sample` dispatcher |
| `render.py` | 181 | `uv_render(objs, w, uvw, phasecenter, pb=None, wideband=False)`, `sample_prep` |

### 9.10 `ism3d/simxy/`

| File | LOC | Purpose |
|------|-----|---------|
| `__init__.py` | 0 | namespace |
| `render.py` | 507 | `xy_render`, `render_apmodel2d`, `render_spmodel3d`, `render_spmodel3d_xyz`, `render_lens` |

### 9.11 `ism3d/utils/`

| File | LOC | Purpose |
|------|-----|---------|
| `__init__.py` | 2 | `from .io import *` |
| `cli_new.py` | 19 | `def main(): pass` — placeholder for a future CLI |
| `io.py` | 551 | `read_data`, `dct2npy/npy2dct`, `to_hdf5/from_hdf5`, `dct2hdf/hdf2dct`, `fits2dct/dct2fits`, `export_model` |
| `meta.py` | 205 | `inp_config`, `create_header`, `db_global` scratchpad |
| `misc.py` | 461 | env checks, FFT-backend report, unit/shortname formatting, `render_component`, `pickplane`, `makepsf`, `makepb`, `makekernel` |
| `utils.py` | 774 | The grab-bag: `paste_array`, `read_par`/`write_par`, `inp_validate`, `gmake_pformat`, `set_threads`, `h5ls`, `gal_flat`, `sample_grid`, `chi2red_to_lognsigma` and many more |

### 9.12 `ism3d/uvhelper/`

| File | LOC | Purpose |
|------|-----|---------|
| `__init__.py` | 14 | Re-exports `imager`, `proc`, `ft`, `ms` |
| `cli.py` | 104 | `casatools_repack` and CLI helpers |
| `ft.py` | 221 | `invert_ft` (NUFFT-based imager), `make_psf`, `advise_header`, `advise_imsize` |
| `imager.py` | 254 | CASA-tclean wrappers: `invert`, `xclean`, `exportimages`, `copyimages`, `ext_list` |
| `ms.py` | 553 | MS reader/writer, freq tools, flagging utilities |
| `plts.py` | 0 | empty placeholder |
| `proc.py` | 44 | `casa_version`, `plotuv_freqtime_amp`, `rawSelect` |
| `vis_utils.py` | 240 | `corrupt_ms`, `cpredict_ms`, `gpredict_ms` (model-prediction helpers) |

### 9.13 `ism3d/visualize/`

| File | LOC | Purpose |
|------|-----|---------|
| `__init__.py` | 0 | empty namespace |
| `msplot2.py` | 233 | `uvamp`, `uvamp_average` (visibility-amplitude diagnostics) |
| `nb.py` | 29 | `show_gif`, `make_gif` (Jupyter helpers) |
| `plts.py` | 962 | The plotting toolbox: `plt_rc0`, `plt_rcProf`, `im_grid`, `plt_spec1d`, `plt_yt3d`, `plt_makeslice`, `plt_slice`, `plt_mom0xy`, `plt_radprof` |

### 9.14 `ism3d/xyhelper/`

| File | LOC | Purpose |
|------|-----|---------|
| `__init__.py` | 0 | namespace |
| `cube.py` | 45 | `hextract` — extract sub-cube + update header |
| `sky.py` | 50 | `linear_offset_coords`, `calc_ppbeam` |

### 9.15 `ism3d/resource/`

| File | Purpose |
|------|---------|
| `default.cfg` | Mirror of `inp_config()` reserved-section ids |
| `input_def.inp` | Reference template enumerating every component keyword, default, type, units |
| `obselete.xymodel.header` | Older FITS header template (superseded by string in `meta.create_header`) |

### 9.16 `tests/`

| File | LOC | Purpose |
|------|-----|---------|
| `__init__.py` | 0 | namespace |
| `test_ism3d.py` | 25 | A single placeholder pytest from the cookiecutter template; both fixture and test are no-ops with commented-out content |

---

## 10. Testing layout

The current test surface is essentially empty:

```
tests/
├── __init__.py            # 0 bytes
└── test_ism3d.py          # cookiecutter placeholder
```

`test_ism3d.py:21–25`:

```python
def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
```

`tox.ini` provides a `flake8` env (`flake8 ism3d tests`) and `py35..py38`
envs that install `requirements_dev.txt` and run `pytest --basetemp={envtmpdir}`.
`setup.cfg:40-41` registers `[tool:pytest]` with
`collect_ignore = ['setup.py']`.

Continuous integration is set up via `.travis.yml` (Travis CI matrix for
py35–py38 per `tox.ini`'s `[travis]` section).

---

## 11. Integration & extension points

| Extension | Hook |
|-----------|------|
| Add a new analytic 2D model | Extend `arts/apmodel2d.py:get_apmodel2d` with another `if model_name == 'foo':` branch returning an `astropy.modeling.models.*` instance |
| Add a new uv sampler | Add a new `uv_sample_X(plane, cell, uu, vv, ...)` in `simuv/ft.py` and dispatch from `uv_sample(...)` (`ft.py:254-342`) |
| Add a new component type | Extend `arts/sparse.py:clouds_from_obj` (the `obj['type']` switch) and `simuv/render.py:uv_render` / `simxy/render.py:xy_render` continuum/line branches |
| Add a new optimiser | Extend `modeling/opt.py:opt_iterate` and add a new `*_iterate` runner; emit results in the same backend-agnostic format consumed by `analyze.opt_analyze` |
| Add a new IO format | Add a `read_<fmt>` / `write_<fmt>` to `utils/io.py` and register it from `read_data` (it currently iterates per `inp_dct[tag]` looking for known keys: `vis`, `image`, `mask`, `pb`, `error`, `sample`, `psf`) |
| Add a new dynamics term | Extend `arts/dynamics.py:potential_fromobj` and `model_vrot` with another `galpy.potential.*` constructor |
| New `.inp` keywords | Add to `ism3d/resource/input_def.inp`; if it is a section-level metadata field, also register an alias in `utils/meta.inp_config` |

External integrations:

- **CASA 6 modular** — through `casatools` (logsink, table, msmetadata,
  calibrater, simulator) and `casatasks` (tclean, exportfits, importfits).
  The Travis/Docker pathway uses `rxastro/casa6` as a base image.
- **galario** — kept as a fallback `uv_sample` method (still imported even
  after HISTORY claims it was replaced by NUFFT).
- **finufftpy** — the recommended modern uv sampler.
- **lenstronomy** — strong-lensing ray-tracing.
- **galpy** — gravitational potentials and rotation curves.
- **emcee + lmfit + corner** — the optimisation/MCMC stack.
- **hickle** — HDF5 serialisation of arbitrary Python objects.
- **pyfftw / mkl_fft** — pluggable FFT backends.

---

## 12. Notable internals & gotchas

1. **Module-level `aeval` interpreters.** Both `interface.py:27` and
   `modeling/model.py:14-17` create their own `asteval.Interpreter()` at
   import time, with `astropy.units` etc. pre-registered. This is *shared
   state* — calling `interface.eval_func` mutates the symbol table.
2. **Heavy module-import side effects.** Importing `ism3d` configures the
   logger, mutates `casalog`, and warms up the FFT backend. Importing
   `ism3d.arts.dynamics` calls `mpl.use('Agg')` (`dynamics.py:11`) — so this
   *must* happen before any other matplotlib backend is selected.
3. **galario references remain.** `HISTORY.rst:17` says NUFFT replaced
   galario, but several modules still `from galario.single import sampleImage`
   / `threads as galario_threads` at import time. Removing galario from the
   environment will break imports of `simuv.ft`, `simuv.render`,
   `simxy.render`, `utils.utils`, and `modeling.opt`.
4. **`finufftpy` API mismatch.** `simuv.ft.uv_sample_nufft` calls
   `nufft.nufft2d2(...)` with positional args matching `finufftpy<2.0`; the
   modern `finufft` package is API-incompatible.
5. **Bug in `simuv.ft.uv_sample_interp2d`.** The function body references a
   variable `method` (lines 244, 247) that is *never defined* in scope —
   the dispatcher actually passes `mode`. So `interp2d` paths will raise
   `NameError: name 'method' is not defined` if hit. Documented here, not
   fixed.
6. **Bug in `arts/analytic.py:uv_render_point()`.** Line 17 reads
   `def uv_render_point()` with no trailing colon equivalence — actually it
   *is* `def uv_render_point()` followed by an empty body comment — and
   `xy_render_point(objs, w,)` has a trailing comma (legal but suspicious).
   Both functions are stubs.
7. **`casalog` shadow-import.** `logger.py:11-17` performs an unconditional
   `os.system('rm -rf '+casalog.logfile())` on import if CASA is present —
   a destructive side effect by design.
8. **`mpl.use('Agg')`** is set in `arts/dynamics.py:11` and
   `visualize/plts.py:9`, so plots are non-interactive by default.
9. **`db_global` scratch dict** in `utils/meta.py:46-49` exists to share
   data + models between processes when the multiprocessing `Pool` is
   forked (avoiding pickling the heavy `dat_dct`); see also the
   `calc_lnprob2_initializer` in `modeling/evaluate.py:220–229`.
10. **PEP-8 / convention violations on purpose.** Many functions use the
    leading-multispace `if  args.fit:` style, double blank lines inside
    docstrings, and rST-flavoured comments — all consistent across the
    project.

---

## 13. Known limitations & TODOs

Drawn from inline comments, the README, HISTORY, and code state:

- **README explicit caveat** (`README.rst:74`): inconsistent docs and many
  non-working function placeholders are expected.
- **`utils/cli_new.py`** — placeholder click CLI returning `None`
  (cli.py is the live one).
- **`uvhelper/plts.py`** — empty file.
- **`arts/analytic.py`** — almost entirely stubs returning `None`.
- **`tests/`** — only the cookiecutter placeholder.
- **`ism3d/ism3d.py`** — empty `"""Main module."""` placeholder.
- **`HISTORY.rst:17`** notes NUFFT replaced galario, but the import is still
  required (see Notable Internals #3).
- **No `requirements.txt`** at the project root (only `requirements_dev.txt`
  and `requirements_casa6pip36.txt`); `tox.ini:21–23` notes a TODO to add a
  pinned `requirements.txt`.
- **`docs/source`** uses Sphinx `autosummary` over `ism3d` and lists
  tutorials under `tutorials/` and `demos/` (e.g. `demo_bx610_*`,
  `demo_hxmm01_*`) that are not present in the repo — they live in the
  upstream docs build only.
- **`docs/source/index.rst:91–95`** acknowledges:
  > *GMaKE is an evolving package. Although we make an effort to maintain
  > backwards compatibility for the parameter file syntax, the Python API
  > can rapidly change at the current alpha development stage.*

---

## 14. Version history

`HISTORY.rst` (verbatim, 26 lines):

```
0.3.dev1 (2020-06-20)
---------------------
* rename the project to "ism3d"
* merge uvrx into ism3d, code refactoring
* implement an imager based on nufft
* update docs/

0.2.dev2 (2020-03-26)
---------------------
* use nufft to replace galario
* use casa6 instead of py-casacore

0.1.dev1 (2019-08-07)
---------------------
* First developmental version
```

Recent git commit log (top of `git log --oneline`):

```
2eff7e6 Remove the docs/ "gateway" page.
b99a191 add a macos-setup note
23462fa add 'make pdf' for docs
670cdb9 cli.py: gmake->ism3d
bb4cc63 update maskmoment
20e72ce add docs/source; add more details for the installation options
1dd2991 update docs/
5dfaaaa add a submodule from the maskmoment fork
7af5b92 add ism3:dev
bf55ea1 add .github
c651e52 add Dockerfile
3c9ad97 update docs: inpfile.rst
0e5d7a2 update docs related to the parameter file syntax
fb9d726 update docs with a new lensing-related tutorial page
f961b48 correctly handle the char-cases of model-type specification; add sie_rt() in arts.lens; ...
457fa2f update docs
d013d45 add the lensing examples
75a7c3a add the "xyz" rendering function: render_spmodel3d_xyz(obj)
71d7da8 update doc to show advanced ism3d.arts capabilities: spiral, thickdisk, etc.
d11d524 move eval_func() to interface.py (as a part of the keyword interpreter); ...
```

Available tags in this checkout: `0.1.dev0`, `0.1.dev2`, `0.2.dev1`.

---

## 15. Quick-reference cheatsheet

```bash
# Install (editable + per macos guide)
pip install -e .
pip install --extra-index-url https://casa-pip.nrao.edu/repository/pypi-group/simple casatools casatasks casashell
pip install finufft   # or build finufftpy from git

# Run a fit
ism3d -f path/to/example.inp                  # equivalent: ism3d --fit ...

# Analyse a finished run
ism3d -a path/to/example.inp

# Diagnostic plotting after a fit
ism3d -p path/to/example.inp

# Direct Python entry
python -c "from ism3d.cli import main; main()"
```

```python
# Programmatic workflow
from ism3d.interface import read_inp, inp_to_mod
from ism3d.utils.io import read_data, export_model
from ism3d.modeling.opt import opt_setup, opt_iterate
from ism3d.modeling.model import model_setup, model_realize, model_render
from ism3d.simuv.render import uv_render
from ism3d.arts.sparse import clouds_morph, clouds_kin, clouds_tosky

inp_dct           = read_inp("example.inp")
mod_dct           = inp_to_mod(inp_dct)
dat_dct           = read_data(inp_dct)
fit_dct, models   = opt_setup(inp_dct, dat_dct)
opt_iterate(fit_dct, inp_dct, dat_dct, models, resume=False)
```

---

*End of reference. Generated from a fresh exploration of
`simulators/ism3d/` at the current submodule HEAD; no other `.md` files
in `simulators/` were consulted.*






