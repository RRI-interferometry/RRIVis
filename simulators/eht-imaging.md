# eht-imaging — Exhaustive Technical Reference

> Reference compiled from a direct walk of the submodule at
> `simulators/eht-imaging/` (HEAD = `8d74d8c`, version tag
> `v1.2.11-2-g8d74d8c`, package version string `1.2.11`).

This document is independent of any other reference in
`simulators/`. Every concrete claim cites the source file from which it
was derived. Where behaviour is genuinely ambiguous, that fact is stated
explicitly rather than being filled in.

---

## 1. Overview

### 1.1 Identity

| Attribute | Value | Source |
|-----------|-------|--------|
| Package import name | `ehtim` | `setup.py:8`, `ehtim/__init__.py` |
| PyPI name | `ehtim` | `setup.py:8` |
| Package version | `1.2.11` | `setup.py:10`; `__init__.py` self-prints `pkg_resources.get_distribution("ehtim").version` |
| Latest tagged release | `v1.2.11` | `git tag` |
| Repository HEAD | `8d74d8c7e859f70b64abd6b1293fd0e8e417e913` | `git rev-parse HEAD` |
| Author | Andrew Chael (`achael@outlook.com`) | `setup.py:12-13`, `CITATION.cff` |
| License | GPLv3 | `LICENSE.txt` (35 KB), `setup.py:16`, every source-file header |
| Languages | Pure Python (CPython 3.x; `from __future__` and `from builtins import` shims throughout) | inspection of every `.py` file |
| Build system | `setuptools` (`setup.py`); `setup.cfg` is essentially empty (one `metadata` block) | `setup.py`, `setup.cfg` |
| Packaging classifier | "Development Status :: 3 - Alpha"; declared "Programming Language :: Python :: 3.8" | `setup.py:53-58` |
| Primary citation | Chael+ 2018 (ApJ 857, 23); Zenodo DOI `10.5281/zenodo.7226661` | `README.rst`, `CITATION.cff` |

### 1.2 Purpose

`ehtim` is the canonical regularized-maximum-likelihood (RML) imaging,
data-handling, and end-to-end-simulation library for the **Event Horizon
Telescope** and other very-long-baseline interferometry (VLBI)
experiments. The package's own description (`setup.py`):

> *Imaging, analysis, and simulation software for radio interferometry.*

The `README.rst` enumerates the central object set:

> *The package contains several primary classes for loading, simulating,
> and manipulating VLBI data. The main classes are the `Image`,
> `Movie`, `Array`, `Obsdata`, `Imager`, and `Caltable` classes…*

Capability surface, taken from public method names and docstrings:

- Load/manipulate/observe sky models (`Image`, `Movie`, `Model`)
- Load/synthesize VLBI visibilities for arbitrary antenna arrays
  (`Array`, `Obsdata`, `obs_simulate.make_uvpoints`,
  `obs_simulate.add_jones_and_noise`)
- RML image reconstruction with closure quantities, polarisation,
  multi-frequency spectra, and Stokes V (`Imager`,
  `imaging/imager_utils.py`, `imaging/pol_imager_utils.py`,
  `imaging/multifreq_imager_utils.py`)
- CLEAN-style imagers (`imaging/clean.py`)
- Dynamical (movie) imaging including StarWarps (`imaging/dynamical_imaging.py`,
  `imaging/starwarps.py`)
- Self-, network-, polarimetric-, polgains-, and leakage calibration
  (`calibrating/`)
- VLBI scattering (Sgr A* in particular) via von Mises / boxcar /
  dipole / power-law screens (`scattering/stochastic_optics.py`)
- Geometric model fitting against visibility/closure data
  (`model.py`, `modeling/modeling_utils.py`)
- Vex schedule parsing (`vex.py`), UVFITS / OIFITS (read-only),
  text, FITS, HDF5 I/O (`io/`), and CASA-like Caltable handling
  (`caltable.py`)
- Survey / parameter-sweep harness using `paramsurvey` (`survey.py`)
- Plotting helpers and image-summary PDFs (`plotting/`)
- Ring-extraction / REx feature module (`features/rex.py`)

### 1.3 Position relative to other VLBI tooling

`ehtim` sits at the science-analysis layer. It does **not** correlate
raw baseband data; it consumes already-correlated visibilities (UVFITS
or its own text format) or generates synthetic visibilities from images
or geometric models. Comparisons to other simulators are out of scope
for this independent reference; see those simulators' own docs.

---

## 2. Repository layout

```
eht-imaging/
├── arrays/                      # Telescope array text files (EHT 2013…2025, GMVA, VLBA, VLA)
│   └── ephemeris/               # TLE entries for space VLBI antennas (ISS, TESS)
├── data/                        # Sample UVFITS and .UVP fixtures
├── docs/
│   └── source/                  # Sphinx .rst (wraps autodoc on each ehtim submodule)
├── ehtim/                       # The package (see §3)
├── examples/                    # Example pipelines (mostly read by users, not tested)
├── models/                      # Stokes I/Q/U sky model text files used by examples
├── scripts/                     # Console-callable tools registered in setup.py
├── tests/                       # Tiny pytest stubs (not a real CI suite)
├── tutorials/                   # Six Jupyter notebooks (M87, polarisation, multifreq, space VLBI)
├── .github/ISSUE_TEMPLATE/      # GitHub issue template (no CI workflows committed)
├── .gitignore, .mailmap
├── CITATION.cff
├── LICENSE.txt                  # GPLv3
├── README.rst                   # See §1
├── setup.py
└── setup.cfg
```

### 2.1 Per-folder commentary

| Folder | Contents | Notes |
|--------|----------|-------|
| `arrays/` | 19 text files describing site coordinates, SEFD, feed-rotation parameters, D-terms | Format documented in `ehtim/io/load.py:load_array_txt`; header reproduced in §6.1 |
| `arrays/ephemeris/` | Plain TLE files named after each space antenna (`ISS`, `TESS`) | Loaded by `ehtim.observing.obs_helpers.sat_skyfield_from_tle` via `Array.add_satellite_tle` |
| `data/` | `sample.uvfits`, `tutorial2_data.uvfits`, `hops_lo_3601_M87+zbl-dtcal_selfcal.uvfits`, three `.UVP` files (3C279, BLLAC) | The `.UVP` are AIPS UV-data exports |
| `docs/source/` | One `.rst` per sub-area (`array.rst`, `image.rst`, `obsdata.rst`, `imager.rst`, …) plus `index.rst` and `conf.py` | Built with the standard Sphinx Makefile; site lives at `https://achael.github.io/eht-imaging` |
| `ehtim/` | Source tree, ~48 kLOC of Python | See §3 |
| `examples/` | 11 example scripts + `example_survey.ipynb` covering imaging, modeling, polarisation, multifreq, scattering, calibration, surveys, and StarWarps | README warns: *"have not been recently validated"* |
| `models/` | Eight sky-model text files (Avery & Howes & Roman & Rowan ray-traced or analytic models of M87 and Sgr A*) consumable by `ehtim.image.load_txt` | All have the header `# SRC: …` (see §6.2) |
| `scripts/` | `calibrate.py`, `cleanup.py`, `cli_blur_comp.py`, `gendata.py`, `imaging.py`, `imgsum.py`, `pipeline.py` | `setup.py:32-37` registers six of these as installed CLIs (not `pipeline.py`) |
| `tests/` | Five files; all very thin (e.g. `test_io.py` is a single 12-line smoke-test of `load_obs_uvfits`) | Not a real CI suite; the repo has no GitHub Actions workflow files |
| `tutorials/` | Six Jupyter notebooks: `ehtim_tutorial1`, `ehtim_tutorial2`, `ehtim_tutorial_m87`, `ehtim_tutorial_multifreq`, `ehtim_tutorial_polarization`, `ehtim_tutorial_spacevlbi` | These are the recommended onboarding material per `README.rst` |

---

## 3. The `ehtim` package source tree

Total: ~48 053 LOC (`wc -l`), spread over the modules below. Sizes are
the file's line count from `wc -l`.

### 3.1 Top-level modules (in `ehtim/`)

| File | LOC | Headline class/role |
|------|-----|---------------------|
| `__init__.py` | 71 | Re-exports the user-facing surface; greets with `"Welcome to eht-imaging! v <ver>"`; provides `logo()` and `eht()` ASCII-art helpers |
| `array.py` | 390 | `Array` class + `load_txt` |
| `caltable.py` | 1040 | `Caltable` class + caltable I/O and gain plot helpers |
| `const_def.py` | 384 | Physical constants, recarray dtypes (`DTARR`, `DTPOL_STOKES`, `DTPOL_CIRC`, `DTAMP`, `DTBIS`, `DTCPHASE`, `DTCAMP`, `DTCAL`, `DTSCANS`, `DTCPHASEDIAG`, `DTLOGCAMPDIAG`), polarisation field maps, and the BH/EHT ASCII art |
| `diagnostics.py` | 89 | `sumdown_lin`, `sumdown_img`, `onedimize` (image down-sampling helpers used in MEM-like analyses) |
| `image.py` | 4422 | `Image` class — central data object |
| `imager.py` | 2174 | `Imager` class — RML driver |
| `model.py` | 2412 | Geometric-model machinery; `Model` class (point, gauss, disk, ring, m-ring, crescent, blurred variants, stretched variants) and `sample_*_uv` family |
| `movie.py` | 1935 | `Movie` class + helpers (`merge_im_list`, `export_multipanel_mp4`, HDF5/text/FITS readers) |
| `obsdata.py` | 4903 | `Obsdata` class — visibility container; the largest module |
| `parloop.py` | 156 | `Parloop` (multiprocessing helper), `Counter`, `HiddenPrints` |
| `survey.py` | 626 | `ParameterSet`, `run_pset`, `run_survey`, `create_params_fixed`, `create_survey_psets` (uses `paramsurvey`) |
| `vex.py` | 332 | `Vex` class — `.vex` schedule parser; `vexdate_to_MJD_hr` |

### 3.2 Sub-packages

```
ehtim/
├── calibrating/
│   ├── __init__.py            (15)
│   ├── cal_helpers.py         (74)
│   ├── network_cal.py         (463)
│   ├── pol_cal.py             (400)
│   ├── pol_cal_new.py         (490)
│   ├── polgains_cal.py        (312)
│   └── self_cal.py            (594)
├── features/
│   ├── __init__.py            (11)
│   └── rex.py                 (1057)  # Ring-EXtractor; Profiles class
├── imaging/
│   ├── __init__.py            (9)
│   ├── clean.py               (1241)
│   ├── dynamical_imaging.py   (2586)
│   ├── imager_utils.py        (4093)
│   ├── linearize_energy.py    (111)
│   ├── multifreq_imager_utils.py (300)
│   ├── patch_prior.py         (150)
│   ├── pol_imager_utils.py    (1550)
│   └── starwarps.py           (1713)
├── io/
│   ├── __init__.py            (11)
│   ├── load.py                (1745)
│   └── save.py                (914)
├── modeling/
│   ├── __init__.py            (9)
│   └── modeling_utils.py      (2700)
├── observing/
│   ├── __init__.py            (11)
│   ├── obs_helpers.py         (1810)
│   ├── obs_simulate.py        (1439)
│   └── pulses.py              (197)
├── plotting/
│   ├── __init__.py            (13)
│   ├── comp_plots.py          (914)
│   ├── comparisons.py         (238)
│   └── summary_plots.py       (1733)
├── scattering/
│   ├── __init__.py            (11)
│   └── stochastic_optics.py   (814)
└── statistics/
    ├── __init__.py            (12)
    ├── dataframes.py          (1044)
    └── stats.py               (335)
```

### 3.3 What `import ehtim` exposes

`ehtim/__init__.py` (71 lines) eagerly executes:

```python
from ehtim.const_def import *                # constants + dtypes + ASCII
from ehtim.modeling.modeling_utils import modeler_func
import ehtim.imaging
from ehtim.features import rex
import ehtim.features
from ehtim.plotting.summary_plots import *   # imgsum, imgsum_pol
from ehtim.plotting.comparisons import *
from ehtim.plotting.comp_plots import *
from ehtim.plotting import comparisons, comp_plots
import ehtim.plotting
from ehtim.calibrating.network_cal import network_cal as netcal
from ehtim.calibrating.self_cal    import self_cal    as selfcal
from ehtim.calibrating.pol_cal import *
from ehtim.calibrating.pol_cal_new import *
from ehtim.calibrating import pol_cal, network_cal, self_cal
import ehtim.calibrating, ehtim.parloop, ehtim.caltable, ehtim.vex
import ehtim.imager, ehtim.obsdata, ehtim.array, ehtim.movie
import ehtim.image, ehtim.model, ehtim.survey
```

So a user-script can refer to e.g. `eh.image.load_txt`, `eh.array.load_txt`,
`eh.obsdata.load_uvfits`, `eh.model.Model`, `eh.network_cal(...)` /
`eh.netcal(...)`, `eh.self_cal(...)` / `eh.selfcal(...)`, `eh.RADPERUAS`,
`eh.modeler_func`, `eh.imgsum`, `eh.Pipeline` (from `scripts/pipeline.py`
when on the path), etc. Note that `eh.imager_func` referenced in older
example scripts (`examples/example.py`, `examples/example_lA_ring.py`) is
**not actually defined** anywhere in the current source tree — those
examples are stale; the supported entry point is the `Imager` class
(see §5.4).

---

## 4. Installation, dependencies and runtime environment

### 4.1 PyPI install

Per `README.rst`:

```bash
pip install ehtim
```

For the dev branch:

```bash
pip install .
```

### 4.2 Hard runtime dependencies (`setup.py:39-50`)

```python
install_requires = [
    "numpy>=1.24",
    "scipy>=1.9.3",
    "astropy>=5.0.4",
    "matplotlib>=3.7.3",
    "skyfield",
    "h5py",
    "pandas",
    "requests",
    "future",
    "networkx",
    # "pynfft; platform_system!='Darwin' or platform_machine!='arm64'",
    "paramsurvey",
]
```

The `pynfft` line is commented out in `setup.py`; the README explains
that the user must install **`NFFT`** and its **`pyNFFT`** wrapper
manually. `pyNFFT` is supported only on Python ≤ 3.11 and NumPy ≤ 1.26.4.
The README announces that **eht-imaging 2.0** will replace `pyNFFT`
with **`finufft`** (Flatiron Institute) — this is in active dev and not
in the v1.2.11 line documented here.

For Apple Silicon (M1–M5) the README points to a forked
`rohandahale/pyNFFT` plus manual `fftw` and `nfft` installs.

### 4.3 Optional dependencies

- **`pyNFFT`** — required if `ttype="nfft"` is used in observation/imaging.
  Imports are wrapped in `try/except ImportError` in
  `imaging/imager_utils.py`, `imaging/clean.py`,
  `imaging/pol_imager_utils.py`, `imaging/multifreq_imager_utils.py`,
  `observing/obs_simulate.py`, `observing/obs_helpers.py`.
- **`networkx`** — used by `plotting/comparisons.image_agreements`;
  guarded in `plotting/comparisons.py`.
- **`scikit-image`** — mentioned in README for "a few image analysis
  functions"; not directly imported in the modules I traversed (so this
  is README-claimed rather than verifiable in code I read).
- **`requests`** — used by `imaging/dynamical_imaging.py` for
  downloading MOJAVE/CLEAN datasets.
- **`pandas`** — for `statistics/dataframes.py`; warning if missing
  (`obsdata.py:37-41`).
- **`sgp4`** + **`skyfield`** — for space-VLBI orbit propagation
  (`observing/obs_helpers.py:26-30`); a graceful warning is emitted if
  missing.
- **`python-casacore`** — *not* in `install_requires`; CASA support is
  not provided directly in this submodule (no `measurement_set.py`).
- **`paramsurvey`** + **`paramsurvey.params`** — required by
  `survey.py`.

### 4.4 Console scripts

`setup.py:32-37` installs the following Python scripts onto `$PATH`:

```
calibrate.py
cleanup.py
cli_blur_comp.py
gendata.py
imaging.py
imgsum.py
```

These are plain Python modules (each starting with `#!/usr/bin/env python`)
in `scripts/`. There is no `entry_points` block, so each shows up under
its full filename when installed. (The unregistered `scripts/pipeline.py`
defines a `Pipeline` declarative dataflow that depends on `ruamel.yaml`.)

---

## 5. Public API — class by class, function by function

The signatures below are taken verbatim from source. Where an argument
list is long it is reproduced in compact form.

### 5.1 `ehtim.const_def` — constants and recarray dtypes

Selected constants (`const_def.py:32-79`):

| Symbol | Value | Comment |
|--------|-------|---------|
| `EP` | `1.0e-10` | numerical floor used by RML regularizers |
| `C` | `299_792_458.0` | speed of light, m/s |
| `DEGREE` | `π/180` | radians per degree |
| `HOUR` | `15·DEGREE` | radians per fractional hour |
| `RADPERAS` | `DEGREE/3600` | radians per arcsec |
| `RADPERUAS` | `RADPERAS·1e-6` | radians per μas — **the most-used unit constant in user code** |
| `RA_SGRA`, `DEC_SGRA` | `17.7611…h`, `-28.99°` | Sgr A* J2000 |
| `RA_M87`, `DEC_M87` | `12.5137h`, `+12.39°` | M87 |
| `SOURCE_DEFAULT="SgrA"`, `RF_DEFAULT=230e9`, `MJD_DEFAULT=51544` |  |  |
| `PULSE_DEFAULT=trianglePulse2D` | imported from `observing/pulses` |  |
| `ELEV_LOW=10`, `ELEV_HIGH=85` | degrees | default elevation cuts |
| `TAUDEF=0.1`, `GAINPDEF=0.1`, `DTERMPDEF=0.05` | default noise sigmas |  |
| `FWHM_MAJ=1309 µas`, `FWHM_MIN=640 µas`, `POS_ANG=78°` | Bower et al. Sgr A* scattering kernel reference |  |
| `NFFT_KERSIZE_DEFAULT=20`, `GRIDDER_P_RAD_DEFAULT=2`, `GRIDDER_CONV_FUNC_DEFAULT='gaussian'`, `FFT_PAD_DEFAULT=2`, `FFT_INTERP_DEFAULT=3` | NFFT/FFT defaults |  |

The recarray dtypes are crucial because **every** observation flows
through them. From `const_def.py:82-126`:

- `DTARR` — telescope row: `('site','U32'), ('x','f8'), ('y','f8'), ('z','f8'),
  ('sefdr','f8'), ('sefdl','f8'), ('dr','c16'), ('dl','c16'),
  ('fr_par','f8'), ('fr_elev','f8'), ('fr_off','f8')`. `x=y=z=0` is the
  in-band convention for "this is a space antenna; ephemeris in
  `Array.ephem`".
- `DTPOL_STOKES` — visibility row Stokes-rep: `time, tint, t1, t2, tau1,
  tau2, u, v, vis, qvis, uvis, vvis, sigma, qsigma, usigma, vsigma`.
- `DTPOL_CIRC` — circular-rep counterpart with `rrvis, llvis, rlvis,
  lrvis, …`.
- `DTAMP`, `DTBIS`, `DTCPHASE`, `DTCAMP`, `DTCPHASEDIAG`,
  `DTLOGCAMPDIAG` — cached closure quantities on `Obsdata`.
- `DTCAL` — caltable row: `time, rscale, lscale` (complex).
- `DTSCANS` — `time, interval, startvis, endvis`.

Field-naming dictionaries `POLDICT_STOKES`, `POLDICT_CIRC`,
`vis_poldict`, `amp_poldict`, `sig_poldict` map between user-facing
labels (`'I','Q','U','V','RR','LL','RL','LR'`) and structured-array
column names. `FIELDS`, `FIELDS_AMPS`, `FIELDS_SIGS`, `FIELDS_PHASE`,
`FIELDS_SIGPHASE`, `FIELDS_SNRS`, `FIELD_LABELS` are used by the
`Obsdata.unpack`/plotting machinery.

### 5.2 `ehtim.array.Array`

```python
class Array(object):
    def __init__(self, tarr, ephem={})
    @property tarr / @tarr.setter         # rebuilds tkey on assignment
    def copy() -> Array
    def listbls() -> np.ndarray           # all unordered baselines
    def obsdata(ra, dec, rf, bw, tint, tadv, tstart, tstop,
                mjd=51544, timetype='UTC', polrep='stokes',
                elevmin=10, elevmax=85, no_elevcut_space=False,
                tau=0.1, fix_theta_GMST=False) -> Obsdata
    def make_subarray(sites)              -> Array
    def save_txt(fname)
    def plot_dterms(sites='all', ...)     -> matplotlib.axes
    def add_site(site, coords, sefd=10000, fr_par=0, fr_elev=0,
                 fr_off=0, dr=0+0j, dl=0+0j) -> Array
    def remove_site(site) -> Array
    def add_satellite_tle(tlelist, sefd=10000) -> Array
    def add_satellite_elements(satname, perigee_mjd=Time.now().mjd,
        period_days=1., eccentricity=0., inclination=0.,
        arg_perigee=0., long_ascending=0., sefd=10000) -> Array
    def plot_satellite_orbits(tstart_mjd=Time.now().mjd,
                              tstop_mjd=...+1, npoints=1000)

def load_txt(fname, ephemdir='ephemeris') -> Array     # module-level
```

`Array.tarr` is a `numpy.recarray` of `DTARR`, `Array.ephem` is a
`dict[site] -> [name, tle1, tle2]` (TLE) or `[perigee_mjd, period_days,
e, i, arg_peri, long_asc]` (Keplerian). Site `tkey[name]` indexes a row
of `tarr`. `obsdata` is the canonical *empty-Obsdata* factory: it calls
`obs_simulate.make_uvpoints` which generates u-v points for every
elevated baseline at every (`tstart` → `tstop`) sample.

### 5.3 `ehtim.image.Image`

`Image` (`image.py:56`) is the central data object for sky maps. The
constructor signature is:

```python
Image(image, psize, ra, dec, pa=0.0,
      polrep='stokes', pol_prim=None,
      rf=230e9, pulse=trianglePulse2D, source='SgrA',
      mjd=51544, time=0.)
```

**Internal representation.** Pixel intensities live in
`self._imdict[pol]` as a 1-D `imvec` of length `xdim*ydim`; the public
`@property` getters/setters expose:

| Property | Backing | Meaning |
|----------|---------|---------|
| `imvec` | `_imdict[pol_prim]` | active polarisation |
| `ivec, qvec, uvec, vvec` | Stokes channels | with circ→Stokes conversion when polrep=='circ' |
| `rrvec, llvec, rlvec, lrvec` | Circular channels |  |
| `pvec, mvec, chivec, evpavec` | Derived: `P=Q+iU`, `m=P/I`, `χ=½arg(P)` |  |
| `rhovec, phivec, psivec, evec, bvec` | Derived: linear polariz. fraction, EVPA, etc. |  |
| `specvec, curvvec, specvec_pol, curvvec_pol, rmvec, cmvec` | Stored in `_mflist` (length 6) | spectral index α, curvature β, RM, CM |

`pol_prim ∈ {'I','Q','U','V'}` for `polrep='stokes'` (default `'I'`),
`pol_prim ∈ {'RR','LL'}` for `polrep='circ'` (default `'RR'`).
`pol_prim='RL','LR'` are explicitly disallowed in the constructor.

**The full method surface** (`image.py:584-4221`, deduplicated):

| Method | What it does |
|--------|--------------|
| `copy(), copy_pol_images(old)` | deep copy / copy other-pol channels into self |
| `add_pol_image(image, pol)` | install a polarisation channel |
| `add_qu(qimage, uimage)`, `add_v(vimage)` | shortcut polarisation setters |
| `switch_polrep(polrep_out, pol_prim_out=None)` | invert circ↔stokes conversion |
| `orth_chi()` | rotate EVPAs by 90° |
| `get_image_mf(nu)` | apply spectral-index/curvature/RM machinery to obtain image at frequency `nu` |
| `imarr(pol=None)` | return 2-D array `(ydim, xdim)` for chosen pol |
| `sourcevec()` | unit vector to source from Earth centre |
| `fovx(), fovy(), total_flux()` | metadata |
| `lin_polfrac(), evpa(), circ_polfrac(), mavg(), vavg()` | image-integrated polarimetric scalars |
| `betamodes(ms=[2], r_min=0, r_max=None)` | radial Fourier ring decomposition (a.k.a. Palumbo β-modes) |
| `center(pol=None), centroid(pol=None)` | shift to image centre or flux centroid |
| `pad(fovx, fovy)` | zero-pad the image |
| `resample_square(xdim_new, ker_size=5)` | resample to square grid |
| `regrid_image(targetfov, npix, interp='linear')` | regrid to new fov/npix |
| `rotate(angle, interp='cubic')` | rotate by angle (rad) |
| `shift(shiftidx)`, `shift_fft(shift)` | integer-pixel and Fourier-domain shifts |
| `blur_gauss(beamparams, frac=1., frac_pol=0)` | Gaussian beam convolution `(maj, min, PA, …)` |
| `blur_circ(fwhm_i, fwhm_pol=0, filttype='gauss')` | circular Gaussian (or boxcar/exp/etc.) blur |
| `blur_mf(freqs, fwhm, fit_order=1, fit_order_pol=1, filttype='gauss')` | multifreq blur |
| `grad(gradtype='abs')` | image-domain spatial gradient |
| `mask(cutoff=0.05, beamparams=None, frac=0.0)` | construct boolean mask |
| `apply_mask(mask_im, fill_val=0.)`, `threshold(...)`, `add_flat(flux, pol=None)`, `add_tophat(...)`, `add_gauss(...)`, `add_crescent(...)`, `add_ring_m1(...)` | sky-component additions |
| `add_const_pol(mag, angle, cmag=0, csign=1)`, `add_random_pol(...)` | install constant/random polarisation |
| `add_const_mf(alpha, beta=0., alpha_pol=None, beta_pol=None, rm=None, cm=None)` | install spectral-index/curvature/RM/CM fields |
| `add_zblterm(obs, uv_min, zblval=None, new_fov=False, ...)` | extended-flux compensation |
| `sample_uv(uv, polrep_obs='stokes', sgrscat=False, ttype='nfft', ...)` | sample visibilities at arbitrary (u,v) points |
| `observe_same_nonoise(obs, sgrscat=False, ttype='nfft', ...)` | populate baselines of an empty `Obsdata` without noise |
| `observe_same(obs_in, sgrscat=False, add_th_noise=True, jones=False, inv_jones=False, opacitycal=True, ampcal=True, phasecal=True, frcal=True, dcal=True, rlgaincal=True, …)` | full Jones-matrix corrupt + noise |
| `observe(array, tint, tadv, tstart, tstop, bw, …)` | generate empty obs **and** sample/corrupt |
| `observe_vex(vex, source, t_int=0.0, tight_tadv=False, …)` | the same but driven by a `.vex` schedule |
| `compare_images(im_compare, …)`, `align_images(im_list, …)`, `find_shift(im_compare, …)` | image-image metrics |
| `fit_gauss(units='rad')`, `fit_gauss_empirical(paramguess=None)` | Gaussian fits to the image |
| `contour(...)`, `display(pol=None, cfun=False, …)`, `overlay_display(im_list, color_coding=…, …)` | plotting |
| `save_txt(fname)` | dump text |
| (other I/O) | `image.py` defines `save_fits` and others not enumerated above |

The `observe`/`observe_same` chain is what users actually invoke; it
routes through `observing/obs_simulate.sample_vis`, then optionally
`obs_simulate.make_jones`/`add_jones_and_noise`/`apply_jones_inverse`.

**Image factory functions** (module-level, `image.py:>4221`):

```python
def make_square(obs, npix, fov, ...) -> Image     # used in every example
def load_txt(filename, …)                          # in io/load.py, re-exported here
def load_fits(filename, …)
def load_image(...)                                # imported by features.rex
```

### 5.4 `ehtim.imager.Imager` — RML driver

Constructor:

```python
Imager(obs_in, init_im, prior_im=None, flux=None,
       data_term=DAT_DEFAULT, reg_term=REG_DEFAULT, **kwargs)
```

with `DAT_DEFAULT = {'vis': 100}`, `REG_DEFAULT = {'simple': 1}`,
`MAXIT = 200`, `STOP = 1e-6`, `NHIST = 50`, `MAXLS = 40`.

**Supported data terms** (`imager.py:47-48`):

```
DATATERMS     = ['vis', 'bs', 'amp', 'cphase', 'cphase_diag',
                 'camp', 'logcamp', 'logcamp_diag']
DATATERMS_POL = ['pvis', 'm', 'vvis']
```

**Regularizers** (`imager.py:50-65`):

```
REGULARIZERS         = ['gs','tv','tvlog','tv2','tv2log','l1','l1w','lA',
                        'patch','flux','cm','simple','compact','compact2',
                        'rgauss', 'flux_mf']
REGULARIZERS_POL     = ['msimple','hw','ptv','l1v','l2v','vtv','vtv2','vflux']
REGULARIZERS_SPECIND = ['l2_alpha','tv_alpha']
REGULARIZERS_CURV    = ['l2_beta','tv_beta']
REGULARIZERS_SPECIND_P = ['l2_alphap','tv_alphap']
REGULARIZERS_CURV_P    = ['l2_betap','tv_betap']
REGULARIZERS_RM        = ['l2_rm','tv_rm']
REGULARIZERS_CM        = ['l2_cm','tv_cm']
```

**Polarisation modes** (`imager.py:81`):

```
POLARIZATION_MODES = ['P','QU','IP','IQU','V','IV','IQUV','IPV']
```

**Default regparams** (`imager.py:75-79`):

```python
REGPARAMS_DEFAULT = {'major':50*RADPERUAS, 'minor':50*RADPERUAS,
                     'PA':0., 'alpha_A':1.0, 'epsilon_tv':0.0}
```

**Public methods** (the `_last`/`_next` suffixed pairs are pure
property accessors that store/retrieve the imager-history snapshots):

| Method | Behaviour |
|--------|-----------|
| `make_image(pol=None, grads=True, mf=False, **kw)` | the optimisation entry point — runs `scipy.optimize.minimize` with method `'L-BFGS-B'`, the `objfunc`/`objgrad` callbacks, and `optdict={'maxiter','ftol','gtol','maxcor':NHIST,'maxls':MAXLS}` |
| `make_image_I/P/IP/V/IV(grads=True, niter=1, blur_frac=1, **kw)` | thin wrappers over `make_image` with the right `pol_next` |
| `converge(niter, blur_frac, pol, grads=True, **kw)` | iteratively call `make_image` with intermediate Gaussian blur steps |
| `check_params()`, `check_limits()`, `init_imager()` | parameter sanity, bounds, and pre-iteration setup |
| `set_embed()` | build the `(mask, Amatrix)` embedding from the prior |
| `make_chisq_dict(imcur), make_chisqgrad_dict(imcur)` | per-data-term χ² and ∂χ²/∂θ |
| `make_reg_dict(imcur), make_reggrad_dict(imcur)` | per-regularizer values and gradients |
| `objfunc(imvec), objgrad(imvec)` | total RML loss and gradient |
| `plotcur(imvec, **kw)` | live-update plot during iterations |
| `format_outim(outarr, pol_prim='I')` | construct the output `Image` |

Module-level helpers (`imager.py:1844+`):

- `embed_imarr(imarr, mask, clipfloor=0., randomfloor=False)`
- `pack_imarr(imarr, which_solve)` / `unpack_imarr(vec, priorarr, which_solve)`
- `transform_imarr(imarr, transforms, which_solve)` / `transform_imarr_inverse`
- `transform_gradients(gradarr, imarr, transforms, which_solve)`
- `make_initarr(image, mask, norm_init=False, flux=1, ...)`

Image transforms supported are listed via the `transform` kwarg (default
`['log','mcv']`); `'mcv'` is the magnitude-change-of-variables for
linear pol, `'vcv'` for circular V, `'polcv'` for full-polarisation
imaging (`imager.py:712-735`).

### 5.5 `ehtim.obsdata.Obsdata` — visibility container

Constructor (`obsdata.py:103`):

```python
Obsdata(ra, dec, rf, bw, datatable, tarr, scantable=None,
        polrep='stokes', source='SgrA', mjd=51544, timetype='UTC',
        ampcal=True, phasecal=True, opacitycal=True,
        dcal=True, frcal=True, trial_speedups=False)
```

`datatable` must be a `np.recarray` of dtype `DTPOL_STOKES` or
`DTPOL_CIRC`; the constructor calls `reorder_baselines`,
`reorder_tarr_sefd`, computes `tstart`/`tstop`, and invalidates closure
caches (`amp, bispec, cphase, cphase_diag, camp, logcamp, logcamp_diag`
all start as `None`).

**Method groupings** (selected; full list above):

*Polrep & timetype housekeeping*
`switch_timetype(timetype_out='UTC')`, `switch_polrep(polrep_out='stokes',
allow_singlepol=True, singlepol_hand='R')`.

*Reordering* — the three `reorder_tarr_*` functions (by SEFD, by SNR,
randomly), `reorder_baselines(trial_speedups=False)`,
`reorder_baselines_trial_speedups()`.

*Conjugation, splits, lists* — `data_conj()`, `tlist(conj=False,
t_gather=0., scan_gather=False)`, `split_obs(t_gather=0.,
scan_gather=False)`, `getClosestScan(time, splitObs=None)`, `bllist(conj=False)`.

*Unpacking* — `unpack(fields, mode='all', ang_unit='deg', debias=False,
conj=False, timetype=False)`, `unpack_bl(site1, site2, fields, …)`,
`unpack_dat(data, fields, …)`. `fields` is a list drawn from `FIELDS`
(see §5.1).

*Closure-quantity computation* — `bispectra`, `bispectra_tri`,
`c_phases`, `c_phases_diag`, `cphase_tri`, `c_amplitudes`,
`c_log_amplitudes_diag`, `camp_quad` and the cached `add_amp`,
`add_bispec`, `add_cphase`, `add_cphase_diag`, `add_camp`, `add_logcamp`,
`add_logcamp_diag`, `add_all`.

*Imaging-helper* — `chisq(im_or_mov, dtype='vis', pol='I', ttype='nfft',
mask=[], **kw)`, `polchisq(im, dtype='pvis', …)`, `recompute_uv()`,
`avg_coherent(inttime, scan_avg=False, moving=False)`,
`avg_incoherent(inttime, scan_avg=False, debias=True,
err_type='predicted')`.

*Beam, dirty image* — `cleanbeam(npix, fov, pulse=…)`,
`fit_beam(weighting='uniform', units='rad')`,
`dirtybeam(npix, fov, pulse=…, weighting='uniform')`,
`dirtyimage(npix, fov, pulse=…, weighting='uniform')`,
`reweight(uv_radius, weightdist=1.0)`.

*Data manipulation / flagging* — `add_scans(info='self', filepath='',
dt=0.0165, …)`, `rescale_zbl(totflux, uv_max, debias=True)`,
`add_leakage_noise(Dterm_amp=0.1, …)`, `add_fractional_noise(...)`,
`find_amt_fractional_noise(im, dtype='vis', target=1.0, …)`,
`rescale_noise(noise_rescale_factor=1.0)`, `estimate_noise_rescale_factor(...)`,
`flag_elev`, `flag_large_fractional_pol`, `flag_uvdist`, `flag_sites`,
`flag_bl`, `flag_low_snr`, `flag_high_sigma`, `flag_UT_range`,
`flags_from_file`, `flag_anomalous`, `filter_subscan_dropouts`.

*Scattering* — `reverse_taper(fwhm)`, `taper(fwhm)`, `deblur()` (Sgr A*
ensemble-average kernel removal).

*Plotting* — `plotall(field1, field2, …)`, `plot_bl(site1, site2, field, …)`,
`plot_cphase(site1, site2, site3, …)`, `plot_camp(site1, site2, site3, site4, …)`.

*Saving* — `save_txt(fname)`, `save_uvfits(fname,
force_singlepol=False, polrep_out='circ')`, `make_hdulist(…)`.

**Module-level loaders** (`obsdata.py:4692-`):

```python
def merge_obs(obs_List, force_merge=False) -> Obsdata
def load_txt(fname, polrep='stokes')
def load_uvfits(fname, flipbl=False, remove_nan=False, force_singlepol=None, …)
def load_maps(arrfile, obsspec, ifile, qfile=0, ufile=0, vfile=0, …)
def load_obs(...)
```

These are thin wrappers over `ehtim.io.load`.

### 5.6 `ehtim.movie.Movie`

`Movie(frames, times, psize, ra, dec, rf, polrep='stokes', pol_prim=None,
pulse=trianglePulse2D, source='SgrA', mjd=51544, bounds_error=True,
interp='linear')`.

- Internally stores frames in `_movdict[pol]` and a SciPy
  `interp1d(self.times, frames.T, kind=interp, ...)` per pol in
  `_fundict`. Supported `interp` values are
  `['linear','nearest','quadratic','cubic','previous','next']`
  (`movie.py:39-40`).
- Exposes `frames`, `imvec_at_time(t)` (and analogous getters for each
  pol), `total_flux`, `blur_circ`, `regrid`, `display`, `save_txt`,
  `save_hdf5`, `save_fits`.
- Module-level helpers: `merge_im_list(imlist, framedur=-1, interp,
  bounds_error)`, `export_multipanel_mp4(input_list, out='movie.mp4',
  start_hr=None, stop_hr=None, nframes=100, …)`, `load_hdf5`, `load_txt`,
  `load_fits`.

### 5.7 `ehtim.model.Model` — geometric models

`Model(ra=RA_DEFAULT, dec=DEC_DEFAULT, pa=0.0, …)` with `models=[]`,
`params=[]`. The class supports adding components and sampling them in
the (u,v) and image planes.

Component types (each `add_*` method appends `model_type` and a
`params` dict):

| Method | Backing model |
|--------|---------------|
| `add_point(F0, x0, y0, pol_frac, pol_evpa, cpol_frac)` | δ-function |
| `add_circ_gauss(F0, FWHM, x0, y0, …)` | symmetric Gaussian |
| `add_gauss(F0, FWHM_maj, FWHM_min, PA, x0, y0, …)` | elliptical Gaussian |
| `add_disk(F0, d, x0, y0, …)` | uniform disk |
| `add_blurred_disk(F0, d, alpha, x0, y0, …)` | disk ⊗ Gaussian |
| `add_crescent(F0, d, fr, fo, ff, phi, x0, y0, …)` | Kamruddin–Dexter crescent |
| `add_blurred_crescent(F0, d, alpha, fr, fo, ff, phi, x0, y0, …)` |  |
| `add_ring(F0, d, x0, y0, …)` | δ-ring |
| `add_stretched_ring(F0, d, x0, y0, stretch, stretch_PA, …)` | anisotropic ring |
| `add_thick_ring(F0, d, alpha, x0, y0, …)` | ring ⊗ Gaussian |
| `add_stretched_thick_ring(...)` |  |
| `add_mring(F0, d, x0, y0, beta_list, beta_list_pol, beta_list_cpol)` | Johnson m-ring |
| `add_stretched_mring(...)` |  |
| `add_thick_mring(F0, d, alpha, x0, y0, beta_list, ...)` |  |
| `add_thick_mring_floor(..., ff=0.0, …)` | + floor |
| `add_thick_mring_Gfloor(..., ff=0.0, FWHM=…, …)` | + Gaussian floor |
| `add_stretched_thick_mring(...)`, `add_stretched_thick_mring_floor(...)` |  |

Sampling/imaging:

```python
sample_xy(x, y, psize=1*RADPERUAS, pol='I')
sample_uv(u, v, polrep_obs='Stokes', pol='I', jonesdict=None)
sample_graduv_uv(u, v, pol='I', jonesdict=None)
sample_grad_uv(u, v, pol='I', fit_pol=False, fit_cpol=False,
               fit_leakage=False, jonesdict=None)
make_image(fov, npix, polrep='stokes', pol_prim=None, pulse=…, time=0.) -> Image
image_same(im) -> Image
display(fov=100*RADPERUAS, npix=256, …)
observe_same_nonoise(obs, **kw) / observe_same(obs_in, …) / observe(array, tint, …)
save_txt(filename) / load_txt(filename)
total_flux(), blur_circ(fwhm), centroid(pol=None), default_prior(...)
```

Module-level helpers: `model_params`, `default_prior`, `stretch_xy`,
`stretch_uv`, `get_const_polfac`, `sample_1model_xy`, `sample_1model_uv`,
`sample_1model_graduv_uv`, `sample_1model_grad_leakage_uv_re`/`_im`,
`sample_1model_grad_uv`, `sample_model_xy`, `sample_model_uv`,
`sample_model_graduv_uv`, `sample_model_grad_uv`, `blur_circ_1model`.

### 5.8 `ehtim.caltable.Caltable`

`Caltable(ra, dec, rf, bw, datadict, tarr, source='SgrA', mjd=51544,
timetype='UTC')` — `datadict[site]` is a `DTCAL` recarray of
`(time, rscale, lscale)`.

Methods (selected): `applycal(obs, interp='nearest', extrapolate=True)`,
`pad_scans()`, `merge(other)`, `enforce_const(...)`, `gain_residuals(...)`,
`save_txt(obs, datadir='.', sqrt_gains=False)`, `plot(...)`, `interp(...)`.

Module-level helpers: `load_caltable(obs, datadir, sqrt_gains=False)`,
`save_caltable(caltable, obs, datadir='.', sqrt_gains=False)`,
`make_caltable(obs, gains, sites, times)`,
`relaxed_interp1d(x, y, **kw)`, `plot_tarr_dterms(...)`,
`plot_compare_gains(caltab1, caltab2, obs, …)`.

### 5.9 `ehtim.vex.Vex`

`Vex(filename)` parses a single-MODE `.vex` schedule file. Exposes:

- `self.metalist` — raw `$`-section blocks
- `self.source` — list of `{'source','ra','dec','ref_coord_frame'}`
- `self.freq`, `self.bw`, `self.array` (an `Array`), `self.sched` (per-scan
  start time, station, scan length)
- helper `get_sector(name)`, `get_variable(key, line)`, `get_all_variables(...)`
- `vexdate_to_MJD_hr(vexdate)` (module-level utility)

### 5.10 Calibration sub-package

| Function | Module | Signature highlights |
|----------|--------|---------------------|
| `self_cal` | `calibrating/self_cal.py:53` | `self_cal(obs, im, sites=[], pol='I', apply_singlepol=False, method='both', minimizer_method='BFGS', pad_amp=0., gain_tol=.2, solution_interval=0., scan_solutions=False, ttype='direct', fft_pad_factor=2, caltable=False, debias=True, apply_dterms=False, copy_closure_tables=False, processes=-1, show_solution=False, msgtype='bar', use_grad=False)` |
| `self_cal_scan` | same | per-scan worker; called by `self_cal` via `Parloop` |
| `network_cal` | `calibrating/network_cal.py:47` | `network_cal(obs, zbl, sites=[], zbl_uvdist_max=1e7, method='amp', minimizer_method='BFGS', pol='I', pad_amp=0., gain_tol=.2, solution_interval=0., scan_solutions=False, caltable=False, processes=-1, show_solution=False, debias=True, msgtype='bar')` |
| `leakage_cal` | `calibrating/pol_cal.py:43` | classic D-term solver |
| `leakage_cal_new` | `calibrating/pol_cal_new.py:52` | newer D-term solver supporting per-site tolerances and rescaling |
| `polgains_cal` | `calibrating/polgains_cal.py:45` | jointly solves for amplitude/phase ratios between R/L |
| `make_cluster_data` | `calibrating/cal_helpers.py:35` | clusters short baselines for network-cal zero-baseline constraints |
| `plot_leakage`, `plot_compare_gains`, `plot_tarr_dterms` | `calibrating/pol_cal.py`, `caltable.py` | visualisation |

Each function returns either an `Obsdata` (default) or a `Caltable`
(when `caltable=True`).

### 5.11 Imaging sub-package

`imaging/imager_utils.py` (4093 lines) contains:

- The χ² and ∇χ² entry points: `chisq(imvec, A, data, sigma, dtype,
  ttype='direct', mask=None)` and `chisqgrad(...)`.
- One pair of `chisq_<dtype>[_fft|_nfft]` / `chisqgrad_<dtype>[_fft|_nfft]`
  for every supported data term × every FT type:
  - dtypes: `vis`, `amp`, `bs`, `cphase`, `cphase_diag`, `camp`,
    `logcamp`, `logcamp_diag`, `logamp`
  - ttype variants: direct (DTFT), `_fft` (gridded FFT), `_nfft`
    (non-uniform FFT via pyNFFT)
- `regularizer(imvec, nprior, mask, flux, xdim, ydim, psize, stype, **kw)`
  and `regularizergrad(...)` dispatching to `sflux/scm/ssimple/sl1/sl1w/
  slA/sgs/spatch/stv/stvlog/stv2/stv2log/scompact/scompact2/sgauss` (and
  their gradients). Per-stype kwargs include `norm_reg, beam_size,
  alpha_A, epsilon_tv, major, minor, PA`.
- `chisqdata(Obsdata, Prior, mask, dtype, pol='I', **kw)` returns
  `(data, sigma, A)`. `chisqdata_<dtype>[_fft|_nfft]` are the
  format-specific implementations.
- Image-vector embedding helpers: `embed(imvec, mask, clipfloor=0,
  randomfloor=False)`, `apply_systematic_noise_snrcut(...)`.
- `plot_i(im, Prior, nit, chi2_dict, **kw)` — live-iteration plot.

`imaging/pol_imager_utils.py` (1550 lines) implements polarimetric RML.
Conventions documented in the file header
(`pol_imager_utils.py:60-67`):

```
P = M = RL = Q + iU = I·ρ·cos(ψ)·exp(iφ)
φ = 2χ = 2·EVPA
m = |Q+iU|/I  ;  v = V/I  ;  ρ = sqrt(Q²+U²+V²)/I
imarr = (I, ρ, φ, ψ)
```

Default solve mask `POL_SOLVE_DEFAULT = (0,1,1,0)` (solve ρ and φ);
`POL_SOLVE_DEFAULT_V = (0,0,0,1)` (solve ψ).

`imaging/multifreq_imager_utils.py` (300 lines) — helpers
`image_at_freq(mfarr, log_freqratio)`, `mf_all_grads_chain(...)`,
`regularizer_mf(...)`, `regularizergrad_mf(...)`, plus spectral
regularizers `l2_spec`, `l2_spec_grad`, `tv_spec`, `tv_spec_grad`. The
constant `DD_RHOPOL = 1` selects the multifreq polarisation-fraction
transform.

`imaging/clean.py` (1241 lines) — Direct-domain CLEAN family (these are
**not** the RML imager):

- `dd_clean_vis(Obsdata, InitIm, niter=1, clipfloor=-1, ttype='direct',
  loop_gain=1, method='min_chisq', weighting='uniform',
  fft_pad_factor=2, p_rad=20, show_updates=False)`
- `dd_clean_bispec_full(Obsdata, InitIm, niter=1, …, loop_gain=.1, …)`
- `dd_clean_bispec_imweight(Obsdata, InitIm, niter=1, ttype='direct', …)`
- `dd_clean_amp_cphase(Obsdata, InitIm, niter=1, …, loop_gain=.1,
  loop_gain_init=1, phaseweight=1, …)`

`imaging/dynamical_imaging.py` (2586 lines) — frame-list movie imaging
following Bouman+ 2017 / Johnson+ 2017. Many `Rd*` / `Rflow*`
regularizer pairs and the top-level entries
`dynamical_imaging_minimal(Obsdata_List, InitIm_List, Prior, …)`,
`dynamical_imaging(obs_input, init_ims, Prior, Flow_Init=None, …)`,
`multifreq_dynamical_imaging(...)`. Includes MOJAVE/CLEAN-archive
download helpers (`generateMOJAVEdates`, `downloadMOJAVEfiles`,
`generateCLEANdates`, `downloadCLEANfiles`,
`MOJAVEHTMLParser`, `BlazarHTMLParser`).

`imaging/starwarps.py` (1713 lines) — StarWarps (Bouman 2017) Bayesian
movie reconstruction. Exposes a `computeSuffStatistics`,
`forwardUpdates`, `backwardUpdates`, `runStarWarps`, etc.

`imaging/patch_prior.py` (150 lines) — EPLL-style Gaussian-mixture
patch prior; `patchPrior(im, beta, patchPriorFile='naturalPrior.mat',
patchSize=8)`, `cleanImage(...)`.

`imaging/linearize_energy.py` (111 lines) — bispectrum linearisation
helpers used by patch-prior dynamical imaging.

### 5.12 Observing sub-package

`observing/obs_simulate.py` is the simulation engine.

`make_uvpoints(array, ra, dec, rf, bw, tint, tadv, tstart, tstop, …)` —
walks every (i,j) pair with i<j, every time tick, calls
`obs_helpers.compute_uv_coordinates`, computes per-baseline thermal
sigmas via `obs_helpers.blnoise(sefd1, sefd2, tint, bw)`, and returns a
recarray of `DTPOL_STOKES` or `DTPOL_CIRC`.

`sample_vis(im_org, uv, sgrscat=False, polrep_obs='stokes',
ttype='nfft', cache=False, fft_pad_factor=2, zero_empty_pol=True,
verbose=True)` — three branches:
- `ttype='fast'`: zero-pad to a power-of-two, apply 2-D FFT, then
  bilinear interpolate at the (u,v) targets, with a triangle/cubic/etc.
  pulse correction (`obs_simulate.py:265-330`).
- `ttype='direct'`: build a DTFT matrix via `obs_helpers.ftmatrix`.
- `ttype='nfft'`: pyNFFT.

After sampling, optional Sgr A* scattering `obs_helpers.sgra_kernel_uv`
is multiplied in (`obs_simulate.py:368-376`).

`make_jones(obs, opacitycal=True, ampcal=True, phasecal=True,
dcal=True, frcal=True, rlgaincal=True, stabilize_scan_phase=False,
stabilize_scan_amp=False, neggains=False, taup=0.1, gainp=0.1,
gain_offset=0.1, phase_std=-1, dterm_offset=0.05, rlratio_std=0.,
rlphase_std=0., sigmat=None, phasesigmat=None, rlgsigmat=None,
rlpsigmat=None, caltable_path=None, seed=False)` — builds the
`{site -> {time -> 2x2 complex matrix}}` Jones-matrix dictionary used by
the corrupting/uncorrupting routines. Supports per-site dicts
for every parameter or a single scalar broadcast.

`make_jones_inverse(obs, opacitycal=True, dcal=True, frcal=True)` —
inverse Jones built from values currently stored on `obs.tarr` and
`obs.scans` (no random noise, just inverse of a-priori known terms).

`add_jones_and_noise(obs, add_th_noise=True, opacitycal=True, …, seed=False,
verbose=True)` — corrupts visibilities by left/right-multiplication with
the Jones matrices (formula `corr_corrupt = J1 · corr · J2†` —
`obs_simulate.py:1042-1048`), then adds independent complex thermal
noise drawn from `obs_helpers.cerror(σ)`. Re-derives sigmas from the
Array's SEFDs with `obs_helpers.blnoise`. The result is converted back
to the input polrep.

`apply_jones_inverse(obs, opacitycal=True, dcal=True, frcal=True,
verbose=True)` — applies the inverse Jones derived from a-priori
calibration data on `obs.tarr`/`obs.scans` (intended to remove gain and
opacity effects whose values are already known).

`add_noise(obs, add_th_noise=True, opacitycal=True, ampcal=True,
phasecal=True, …)` — older (non-Jones) noise application path; used
when `Image.observe(jones=False)`.

`observing/obs_helpers.py` is a 1810-line trove of utilities. Notable
groupings:

- **u-v geometry**: `compute_uv_coordinates(array, site1, site2, time,
  mjd, ra, dec, rf, timetype='UTC', elevmin=10, elevmax=85,
  no_elevcut_space=False, fix_theta_GMST=False, earthshadow_space=True)`
  (returns `(times, u, v)` with elevation/space-shadow masking),
  `earthrot(vecs, thetas)`, `earthshadow_mask(obsvecs, sourcevec)`,
  `elev(obsvecs, sourcevec)`, `elevcut(...)`, `hr_angle(gst, lon, ra)`,
  `par_angle(hr_angle, lat, dec)`, `xyz_2_latlong(obsvecs)`,
  `gmst_to_utc(gmst, mjd)`, `utc_to_gmst(utc, mjd)`.
- **Closure quantity construction**: `make_bispectrum(l1, l2, l3, vistype,
  polrep='stokes')`, `make_closure_amplitude(blue1, blue2, red1, red2, …)`,
  `tri_minimal_set(sites, tarr, tkey)`, `quad_minimal_set(...)`,
  `reduce_tri_minimal(obs, datarr)`.
- **Debiasing**: `amp_debias(amp, sigma, force_nonzero=False)`,
  `camp_debias(camp, snr3, snr4)`,
  `logcamp_debias(log_camp, snr1, snr2, snr3, snr4)`.
- **Noise & RNG**: `blnoise(sefd1, sefd2, tint, bw) = sqrt(sefd1*sefd2/(2*bw*tint))/0.88`
  (the 0.88 is the standard 2-bit correlator coefficient — see source),
  `cerror(sigma)`, `cerror_hash`, `hashrandn`, `hashrand`,
  `hashmultivariaterandn`. Hash-based RNGs ensure repeatable,
  per-baseline-deterministic noise.
- **Sgr A* scattering**: `sgra_kernel_uv(rf, u, v)`, `sgra_kernel_params(rf)`
  — reproduces the Bower kernel (FWHM_MAJ=1.309 mas, FWHM_MIN=0.64 mas,
  PA=78°).
- **FT matrices**: `ftmatrix(pdim, xdim, ydim, uvlist, pulse=…, mask=[])`,
  `ftmatrix_centered(...)`, `gauss_uv(u, v, flux, beamparams, x=0, y=0)`,
  `rbf_kernel_covariance(x, sigma)`.
- **Polarisation conversions**: `merr(sigma, qsigma, usigma, I, m)`,
  `merr2(rlsigma, rrsigma, llsigma, I, m)`.
- **Misc**: `image_centroid(im)`, `power_of_two(target)`, `paritycompare`,
  `sigtype(datatype)`, `rastring(ra)`, `decstring(dec)`, `gmtstring(gmt)`,
  `ticks(axisdim, psize, nticks=8)`.
- **Space VLBI**: `sat_skyfield_from_tle(name, line1, line2)`,
  `sat_skyfield_from_elements(name, mjd, perigee, period_days, e, i,
  arg_peri, long_asc)`, `sat_skyfield_from_ephementry(name, ephem, mjd)`,
  `orbit_skyfield(sat, fracmjds, whichout='gcrs')`. These use `skyfield`
  + `sgp4`.

`observing/pulses.py` (197 lines) defines the convolutional pulse
functions used to interpret pixelated images as continuous brightness
distributions. Each pulse provides both an image-domain (`'I'`) and
Fourier-domain (`'F'`) implementation:

| Pulse | Image domain | Fourier domain |
|-------|--------------|----------------|
| `deltaPulse2D` | δ(x)δ(y) | 1 |
| `rectPulse2D` | rect(x/Δ)·rect(y/Δ) | sinc-like |
| `trianglePulse2D` (default) | tri(x/Δ)·tri(y/Δ) | `(2sin(Δω/2)/(Δω))²` per axis |
| `GaussPulse2D` | (a/π)·exp(-a(x²+y²)) with σ=Δ/3 | `exp(-(x²+y²)/(4a))` |
| `cubicPulse2D` | piecewise-cubic Keys filter | cubic-Keys closed-form |
| `sincPulse2D` | sinc(πx/Δ)/Δ | rect-like |

(Disk pulse and cubic-spline pulse are commented out at file end.)

### 5.13 I/O sub-package

`io/load.py` (1745 lines) — `load_vex`, `load_im_txt`, `load_im_hdf5`,
`load_im_fits` (with `aipscc=False` to read CLEAN components),
`load_movie_hdf5`, `load_movie_txt`, `load_movie_fits`, `load_movie_dat`,
`load_array_txt(filename, ephemdir='ephemeris')` (parses
`#NAME X Y Z SEFDR SEFDL FR_PAR FR_ELEV FR_OFF DR_RE DR_IM DL_RE DL_IM`),
`load_obs_txt`, `load_obs_uvfits(filename, polrep='stokes', flipbl=False, …)`,
`load_obs_maps`, `load_dtype_txt(obs, filename, dtype='cphase')`.

`io/save.py` (914 lines) — `save_im_txt`, `save_im_fits`,
`save_mov_hdf5`, `save_mov_fits`, `save_mov_txt`, `save_array_txt`,
`save_obs_txt`, `save_obs_uvfits(obs, fname=None, force_singlepol=None,
polrep_out='circ')`, `save_dtype_txt(obs, fname, dtype='cphase')`.

The most recent commit (`8d74d8c…`) updated NFFT/pyNFFT install notes;
commit `768a536` "removed oifits support" deleted what was an OIFITS
loader/saver — this is the visible change in the I/O surface from
historical versions.

### 5.14 Modeling sub-package

`modeling/modeling_utils.py` (2700 lines) — `modeler_func` is the
exported entry point for visibility-domain geometric model fitting:

```python
def modeler_func(Obsdata, model_init, model_prior,
                 d1='vis', d2=False, d3=False,
                 alpha_d1=100, alpha_d2=100, alpha_d3=100,
                 minimizer_func='dynesty_dynamic',
                 minimizer_kwargs={}, …)
```

(The signature continues for many lines; the function dispatches to
`scipy.optimize.minimize`, MCMC samplers, or Dynesty depending on
`minimizer_func`.) Supported `DATATERMS` include `'vis', 'bs', 'amp',
'cphase', 'cphase_diag', 'camp', 'logcamp', 'logcamp_diag', 'logamp',
'pvis', 'm', 'rlrr', 'rlll', 'lrrr', 'lrll', 'rrll', 'llrr',
'polclosure'`. Per-parameter unit conversions are kept in
`PARAM_DETAILS` (`F0` in Jy, `FWHM/d/x0/y0` in μas converted via
`RADPERUAS`, `PA/arg/evpa/phi` in degrees, etc., `modeling_utils.py:73-76`).

Default priors:

```python
GAIN_PRIOR_DEFAULT    = {'prior_type':'lognormal','sigma':0.1,'mu':0.0,'shift':-1.0}
LEAKAGE_PRIOR_DEFAULT = {'prior_type':'flat','min':-0.5,'max':0.5}
N_POSTERIOR_SAMPLES   = 100
```

Module also exports `cdf`, `inverse_cdf`, `prior_func`, `priorgrad_func`,
the parameter-transform stack, plus chi-square / chi-square gradient
functions for every datum type. `minimizer_func` choices include
`L-BFGS-B`, `dynesty_dynamic`, `dynesty_static`, `pymc3`, `emcee`
(based on the kwargs documented near the function body).

### 5.15 Scattering sub-package

`scattering/stochastic_optics.py` (814 lines) — Johnson 2016 stochastic
optics framework. The single class is:

```python
class ScatteringModel:
    def __init__(self, model='dipole', scatt_alpha=1.38,
                 observer_screen_distance=2.82·3.086e21,  # cm
                 source_screen_distance=5.53·3.086e21,    # cm
                 theta_maj_mas_ref=1.380, theta_min_mas_ref=0.703,
                 POS_ANG=81.9,
                 wavelength_reference_cm=1.0,
                 r_in=800e5, r_out=1e20)
```

`model ∈ {'von_Mises', 'boxcar', 'dipole', 'power-law'}`.
Defaults match Sgr A* (Johnson+ 2018 / Issaoun+ 2019) with a Kolmogorov
slope `α_scatt=5/3` available by overriding `scatt_alpha`.

Methods (`stochastic_optics.py:132-670`):

| Method | Behaviour |
|--------|-----------|
| `P_phi(phi)` | angular power distribution |
| `rF(wavelength)` | Fresnel scale at λ |
| `Mag()` | screen magnification `M = D_obs/D_src` |
| `dDphi_dz`, `Dphi_exact(x, y, λ_cm)`, `Dphi_approx(...)` | phase-structure-function variants |
| `Dmaj`, `Dmin` | per-axis structure functions |
| `Q(qx, qy)` | power spectrum of phase fluctuations |
| `sqrtQ_Matrix(Reference_Image, Vx_km_per_s=50, Vy_km_per_s=0, t_hr=0)` | √Q on the reference grid (with translation) |
| `Ensemble_Average_Kernel(Reference_Image, λ_cm=None, use_approximate_form=True)` | image-domain ensemble-average kernel |
| `Ensemble_Average_Kernel_Visibility(u, v, λ_cm, …)` | (u,v)-domain |
| `Ensemble_Average_Blur(im, λ_cm=None, ker=None, …)` | apply EA blur to an image |
| `Deblur_obs(obs, use_approximate_form=True)` | divide observation by the EA kernel |
| `MakePhaseScreen(EpsilonScreen, Reference_Image, obs_frequency_Hz=0, Vx_km_per_s=50, Vy_km_per_s=0, t_hr=0, sqrtQ_init=None)` | realize one screen |
| `Scatter(Unscattered_Image, Epsilon_Screen=…, …, Linearized_Approximation=False, DisplayImage=False, Force_Positivity=False, …)` | apply screen → scattered image |
| `Scatter_Movie(Unscattered_Movie, Epsilon_Screen=…, framedur_sec=None, N_frames=None, …, processes=0)` | scatter a `Movie` |
| `Scatter2(args, kwargs)` | thin parallelisation helper |

Module-level helpers: `Wrapped_Convolve`, `Wrapped_Gradient`,
`MakeEpsilonScreenFromList`, `MakeEpsilonScreen(Nx, Ny, rngseed=0)`,
`plot_scatt(...)`.

### 5.16 Statistics sub-package

`statistics/stats.py` — circular statistics
(`circular_mean(theta, unit='deg')`, `circular_std`,
`circular_std_of_mean`), incoherent-amplitude estimators
(`mean_incoh_amp`, `mean_incoh_amp_from_vis`, `bootstrap`,
`mean_incoh_avg`), debiasing (`deb_amp`, `inc_sig`, `coh_sig`), TV-based
quality reports (`dicts_TV_report`, `compare_TV`).

`statistics/dataframes.py` — pandas-backed averaging:
`make_df`, `make_amp`, `coh_avg_vis(obs, dt=0, scan_avg=False,
return_type='rec', …)`, `coh_moving_avg_vis`, `roll_vis`, `roll_sig`,
`incoh_avg_vis`, `make_cphase_df`, `make_cphase_diag_df`, `make_camp_df`,
`make_logcamp_diag_df`, `make_bsp_df`, `average_cphases`,
`average_bispectra`, `average_camp`, `df_to_rec`, `round_time`,
`get_bins_labels`, `common_set`, `common_multiple_sets`,
`match_multiple_frames`, `add_gmst`.

### 5.17 Plotting sub-package

`plotting/comp_plots.py` — `plotall_compare`, `plot_bl_compare`,
`plot_cphase_compare`, `plot_camp_compare` and the obs-vs-image
counterparts `plotall_obs_compare`, `plotall_obs_im_compare`,
`plot_bl_obs_compare`, `plot_bl_obs_im_compare`,
`plot_cphase_obs_compare`, `plot_cphase_obs_im_compare`,
`plot_camp_obs_compare`, `plot_camp_obs_im_compare`,
`plotall_obs_im_cphases`, `prep_plot_lists`.

`plotting/comparisons.py` — image-image consistency:
`image_consistency(imarr, beamparams, metric='nxcorr',
blursmall=True, beam_max=1.0, beam_steps=5, savepath=[])`,
`get_psize_fov(imarr)`, `image_agreements(imarr, beamparams,
metric_mtx, fracsteps, cutoff=0.95)` (uses `networkx`),
`change_cut_off(...)`, `generate_consistency_plot(...)`.

`plotting/summary_plots.py` — `imgsum(im_or_mov, obs, obs_uncal,
outname, outdir='.', title='imgsum', commentstr='',
fontsize=22, cfun='afmhot', snrcut=0., maxset=False, ttype='nfft',
gainplots=True, ampplots=True, cphaseplots=True, campplots=True,
ebar=True, debias=True, cp_uv_min=False, force_extrapolate=True,
processes=4, sysnoise=0, syscnoise=0)` — produces a multi-page PDF
summary; `imgsum_pol` does the polarised version. Private helpers
`_display_img(im, beamparams=None, scale='linear', gamma=0.5, …)` and
`_display_img_pol(...)`.

### 5.18 Features (rex)

`features/rex.py` — Ring-EXtractor (`Profiles` class). Used by the EHT
collaboration for ring-radius extraction. Constants
`IMSIZE=160 µas, NPIX=160, NRAYS=360, NRS=100, RMAX=50 µas, RMIN=5 µas,
RPRIOR_MIN=15 µas, RPRIOR_MAX=50 µas, NRAYS_SEARCH=25, NRS_SEARCH=50,
THRESH=0.05, FOVP_SEARCH=0.1, NSEARCH=10`. The `Profiles` class
constructs angular and radial intensity profiles around a fit centroid.

### 5.19 Survey

`survey.py` — `ParameterSet(paramset, params_fixed={})` wraps one
parameter combination from a survey. Methods include `load_data()`,
`preimcal()`, `preimaging()`, `imaging()`, `postimaging()`,
`save_outputs()`, etc. (full pipeline of imaging-tutorial steps).
Module-level: `run_pset(pset, system_kwargs, params_fixed)`,
`run_survey(psets, params_fixed)`,
`create_params_fixed(infile, outfile_base, outpath,
ground_truth_img='None', …)`,
`create_survey_psets(zbl=[0.6], sys_noise=[0.02], avg_time=['scan'],
prior_fwhm=[40], …)`.

Actual parallel execution is handled by the `paramsurvey` library
(MPI / multiprocessing / Ray backend chosen at runtime).

### 5.20 Parloop

`parloop.py:Parloop(func)` provides a multiprocessing wrapper with a
counter-based progress meter. `run_loop(arglist, processes=-1)` returns
a list. `Counter` (with shared `Value`/`Lock`) tracks progress;
`HiddenPrints` is a context manager for silencing children. Used by the
calibration scan-level routines (`get_selfcal_scan_cal2`,
`get_network_scan_cal2`, etc.).

---

## 6. Input / output formats

### 6.1 Array text file

Source: `arrays/EHT2017.txt`.

```
#NAME X             Y             Z             SEFDR SEFDL FR_PAR_ANGLE FR_ELEV_ANGLE FR_OFFSET[d] DR_RE   DR_IM   DL_RE    DL_IM
PV    5088967.9000 -301681.6000   3825015.8000  1400  1400  1            -1            0            0       0       0        0
SMT   -1828796.200 -5054406.800   3427865.200   5000  5000  1             1            0            0       0       0        0
SMA   -5464523.400 -2493147.080   2150611.750   4900  4900  1            -1            45           0       0       0        0
LMT   -768713.9637 -5988541.7982  2063275.9472  600   600   1            -1            0            0       0       0        0
ALMA  2225061.164  -5440057.37   -2481681.15    90    90    1             0            0            0       0       0        0
SPT   0.01          0.01         -6359609.7     5000  5000  1             0            0            0       0       0        0
APEX  2225039.53   -5441197.63   -2479303.36    3500  3500  1             1            0            0       0       0        0
JCMT  -5464584.68  -2493001.17    2150653.98    6000  6000  1             0            0            0       0       0        0
```

XYZ are ITRF coordinates in metres. SEFDR/SEFDL are the right- and
left-circular SEFDs in Jy. `FR_PAR_ANGLE` and `FR_ELEV_ANGLE` are
parallactic-/elevation-angle multipliers (±1, 0); `FR_OFFSET` is a
constant phase offset in degrees. `DR_*`/`DL_*` are real and imaginary
parts of the antenna D-terms.

A **space antenna** is signalled by `X=Y=Z=0`; the loader
(`load_array_txt`) then reads a TLE from `arrays/ephemeris/<name>` (e.g.
`arrays/ephemeris/ISS`):

```
ISS
1 25544U 98067A   23109.53559294  .00019257  00000-0  34454-3 0  9994
2 25544  51.6389 259.1975 0006157 210.6524 231.6170 15.49989360392694
```

### 6.2 Image text file

Source: `models/avery_sgra_eofn.txt`.

```
# SRC: SgrA
# RA: 17 h 45 m 40.0409 s
# DEC: -28 deg 59 m 31.8820 s
# MJD: 48277.0000
# RF: 230.0000 GHz
# FOVX: 100 pix 0.000160 as
# FOVY: 100 pix 0.000160 as
# ------------------------------------
# x (as)     y (as)       I (Jy/pixel)  Q (Jy/pixel)  U (Jy/pixel)
0.0000784 0.0000784 0.00001215 0.00000519 -0.00000367
…
```

Columns 3+ are Stokes I, Q, U (V optional). `image.load_txt` is the
reader; `image.save_txt` the writer.

### 6.3 Visibility data formats

- **UVFITS** — read by `io/load.load_obs_uvfits`, written by
  `io/save.save_obs_uvfits`. Default polrep stored is `'circ'`.
- **AIPS UVP** — `data/3C279APR13.UVP` etc. are UVFITS written from
  AIPS; loaded with the same path.
- **eht-imaging text** — `Obsdata.save_txt` / `load_obs_txt`; columns
  follow `DTPOL_STOKES`.
- **HDF5** — `Movie.save_hdf5` / `load_movie_hdf5` use `h5py`.
- **`.vex` schedule** — `Vex(filename)` parses a VLBI standard schedule
  and provides per-source/per-frequency/per-station metadata.

### 6.4 Caltable

`Caltable.save_txt(obs, datadir='.', sqrt_gains=False)` writes a
per-site text file `<site>.txt` under `datadir`. The complementary
`load_caltable(obs, datadir, sqrt_gains=False)` reconstructs a
`Caltable` instance.

### 6.5 Models text file

Geometric `Model` instances are saved/loaded via
`Model.save_txt(filename)` / `Model.load_txt(filename)`; the file
format is documented within `model.py` (line-oriented, one component
per line, with named parameters).

---

## 7. Architecture

```
                 ┌───────────────────────────────────────────────┐
   user code →   │  ehtim package surface (eh.image, eh.array,   │
                 │  eh.obsdata, eh.imager.Imager, eh.model,      │
                 │  eh.modeler_func, eh.netcal, eh.selfcal, …)   │
                 └───────────────┬───────────────────────────────┘
                                 │
       ┌─────────────────────────┼───────────────────────────────┐
       │                         │                               │
       ▼                         ▼                               ▼
 ┌───────────┐         ┌───────────────────┐           ┌───────────────────┐
 │  Image    │         │     Obsdata       │           │ ScatteringModel    │
 │  Movie    │◀──────▶ │  (vis recarrays,  │ ◀────────▶│ (Sgr A* screens)   │
 │  Model    │         │   closure caches) │           └───────────────────┘
 └─────┬─────┘         └───────┬───────────┘                       │
       │                       │                                   │
       │  observe / observe_same                                   │
       ▼                       ▼                                   │
 ┌────────────────────────────────────────────────────────────┐    │
 │  observing/obs_simulate.py                                 │    │
 │   make_uvpoints  ─►  sample_vis (direct/fast/nfft) ─►      │    │
 │   make_jones / make_jones_inverse / add_jones_and_noise    │◀───┘
 │   add_noise / apply_jones_inverse                          │
 └────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
            ┌──────────────────────────────────────────┐
            │  observing/obs_helpers.py                │
            │   compute_uv_coordinates, earthrot,      │
            │   par_angle, elev, blnoise, cerror,      │
            │   sgra_kernel_uv, ftmatrix, …            │
            │   sat_skyfield_*  (skyfield + sgp4)      │
            └──────────────────────────────────────────┘

        ┌──────────────────────┐    ┌─────────────────────────┐
        │   imager.Imager      │    │  imaging/clean.py       │
        │ ─────────────────    │    │   dd_clean_*            │
        │ ⇢ make_chisq_dict    │    └─────────────────────────┘
        │ ⇢ make_reg_dict      │
        │ ⇢ scipy L-BFGS-B     │    ┌─────────────────────────┐
        └─────┬────────────────┘    │ imaging/dynamical_*.py  │
              │                     │ imaging/starwarps.py    │
              ▼                     └─────────────────────────┘
   ┌─────────────────────────────────────────┐
   │ imaging/imager_utils.py                 │
   │  chisq[grad]_<dtype>[_fft|_nfft]        │
   │  regularizer[grad]                      │
   │  chisqdata[_fft|_nfft]                  │
   │  embed                                  │
   │ imaging/pol_imager_utils.py             │
   │ imaging/multifreq_imager_utils.py       │
   └─────────────────────────────────────────┘

      ┌───────────────────────────────────────────┐
      │  calibrating/                             │
      │   self_cal, network_cal, leakage_cal,     │
      │   leakage_cal_new, polgains_cal           │
      │   each: per-scan worker + Parloop driver  │
      └───────────────────────────────────────────┘

      ┌───────────────────────────────────────────┐
      │  modeling/modeling_utils.py               │
      │   modeler_func — visibility model fitting │
      │   (scipy / dynesty / emcee back-ends)     │
      └───────────────────────────────────────────┘

      ┌───────────────────────────────────────────┐
      │  io/                                      │
      │   load.py  — UVFITS, FITS, HDF5, text     │
      │   save.py  — UVFITS, FITS, HDF5, text     │
      └───────────────────────────────────────────┘
```

---

## 8. The RML imaging algorithm

Restating the optimisation that `Imager.make_image` performs
(`imager.py:540-660`):

```
J(I) = Σ_d  α_d · χ²_d(F[I; uv]) + Σ_r β_r · S_r(I)
```

with `I` the image vector (`xdim·ydim` reals after the
transform/embedding), `F` the chosen Fourier transform back-end
(`'direct'`, `'fast'`, `'nfft'`), `α_d` the data-term weights from
`data_term`, `β_r` the regularizer weights from `reg_term`. Closure
quantities are computed as nonlinear maps of `F[I]`. Multifrequency
imaging extends the state vector to `(I, α, β, αp, βp, RM, CM)` channels
per pixel and applies `image_at_freq`, `mf_all_grads_chain` (chain rule
through the spectral law) at each frequency.

`scipy.optimize.minimize` is invoked with method `'L-BFGS-B'`,
`options={'maxiter': maxit_next, 'ftol': stop_next, 'gtol': stop_next,
'maxcor': 50, 'maxls': 40}` (`imager.py:604-606`). A callback
(`plotcur`) is hooked in for iteration display.

The Coherency / Stokes conventions follow the standard
`I = (RR+LL)/2`, `Q = (RL+LR)/2`, `U = i(LR-RL)/2`, `V = (RR-LL)/2`
mapping (encoded e.g. in `Image.qvec` getter,
`image.py:273-282`).

Visibility-domain RIME formulation actually used by
`add_jones_and_noise` (`obs_simulate.py:1042-1048`):

```
[ V_RR  V_RL ]      [ J_p ]   [ I+V   Q+iU ]   [ J_q ]†
[           ]   =  [     ] × [           ] × [     ]
[ V_LR  V_LL ]    2x2(circ)   [ Q-iU   I-V ]   2x2

V_corrupt = J1 · V · J2†   +  noise,  noise ~ CN(0, σ_th)
```

where the 2×2 Jones is the per-time per-site product of opacity, gain,
phase, R/L gain ratio, R/L phase difference, leakage, and feed-rotation
factors as constructed in `make_jones`.

---

## 9. Numerics, parallelism and performance

| Aspect | Detail |
|--------|--------|
| FT back-ends | `direct` (DTFT matrix), `fast` (zero-pad FFT + bilinear interpolation), `nfft` (pyNFFT — preferred where available). NFFT defaults: kernel size 20, gridder p_rad=2, pad factor 2, interp order 3 (`const_def.py:74-79`). |
| Linear algebra | NumPy + SciPy throughout; no GPU, no CuPy, no JAX. |
| Optimiser | SciPy `minimize` (L-BFGS-B for RML; BFGS / L-BFGS-B / TNC / etc. for calibration; `dynesty`, `emcee`, `pymc3` optional in `modeler_func`). |
| Parallelism | Python `multiprocessing.Pool` via `parloop.Parloop` (used in `self_cal`, `network_cal`, `polgains_cal`, `leakage_cal_new`). Survey-scale parallelism via `paramsurvey`. |
| Determinism | `obs_helpers.hashrandn`/`hashrand`/`hashmultivariaterandn` derive RNG state from a hashed seed string + per-baseline tag, so observations are reproducible given a `seed` argument. `seed=False` defaults to system time string. |
| MPI | Not used directly; if `paramsurvey` is configured for MPI it is delegated. |
| Caching | `Image.cached_fft` keyed by polarisation; `Obsdata` caches `amp/bispec/cphase/…` after `add_amp`/`add_bispec`/etc. |

---

## 10. Scripts, examples, tutorials

### 10.1 Installed scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `imaging.py` | Tutorial-script-style RML pipeline for a uvfits file — coherent averaging, SNR cuts, prior construction, multi-stage `Imager` runs (see `scripts/imaging.py:1-80` for parameter setup, plus `converge(major=5)` driver). |
| `imgsum.py` | `argparse`-driven CLI wrapper around `plotting/summary_plots.imgsum` to produce a PDF image-summary report from `(image.fits, uvfits.uvfits, uvfits_uncal.uvfits)`. |
| `calibrate.py` | Multi-station network-calibration loop using `eh.network_cal` repeated with progressively-tighter tolerances. |
| `gendata.py` | Synthetic-data generator (image → observation with optional Jones leakage, ad-hoc phasing, …). |
| `cleanup.py` | YAML-driven `eh.Pipeline` driver over a config file — loads UVFITS, applies a sequence of transforms, saves output. |
| `cli_blur_comp.py` | Beam-convolution comparison helper. |

### 10.2 Examples (`examples/`)

`example.py`, `example_calibration.py`, `example_im_closure.py`,
`example_lA_ring.py`, `example_modeling.py`, `example_multifreq.py`,
`example_pol.py`, `example_scattering.py`, `example_starwarps.py`,
`example_stochastic_optics.py`, `example_survey.py`,
`example_survey.ipynb`. README warns: *"have not been recently
validated."* Several reference `eh.imager_func` which is not present in
the current sources; users should treat those examples as *templates*
and replace the call site with `eh.imager.Imager(...).make_image()`.

### 10.3 Tutorials (`tutorials/`)

| Notebook | Topic |
|----------|-------|
| `ehtim_tutorial1.ipynb`, `ehtim_tutorial2.ipynb` | Generic introductory tutorials (data loading, observation simulation, basic imaging). |
| `ehtim_tutorial_m87.ipynb` | M87* RML pipeline. |
| `ehtim_tutorial_polarization.ipynb` | Linear-polarisation imaging (P, IP). |
| `ehtim_tutorial_multifreq.ipynb` | Multifrequency spectral-index imaging. |
| `ehtim_tutorial_spacevlbi.ipynb` | Adding a satellite via TLE / Keplerian elements; orbit visualisation; observation. |

### 10.4 Tests (`tests/`)

The test suite is intentionally minimal:

| File | Body |
|------|------|
| `test_io.py` | `assert load.load_obs_uvfits('./tests/../data/sample.uvfits')` — single smoke test; comment says *"TODO: verify the result"*. |
| `test_diagnostics.py`, `test_fft_chisquared.py`, `test_gradients.py`, `test_regularizers.py` | similar-scale check scripts |

There are **no** GitHub Actions workflow files committed
(`.github/` only contains `ISSUE_TEMPLATE/`). No `pytest.ini` or
`pyproject.toml` is present; tests must be invoked manually with
`pytest tests/`.

---

## 11. Integration & extension points

- **New telescope**: add a row to an `arrays/*.txt` file (or call
  `Array.add_site(...)`) — the SEFD and D-term columns flow through
  every downstream calculation.
- **New space antenna**: drop a TLE in `arrays/ephemeris/<name>` (or
  call `Array.add_satellite_tle`/`add_satellite_elements`).
- **New regularizer**: add a `s<name>` value-function and a
  `s<name>grad` gradient-function in `imaging/imager_utils.py`, register
  the string in `REGULARIZERS`, dispatch in `regularizer` /
  `regularizergrad`. Users pass it via `reg_term={'<name>': weight}`.
- **New data term**: add `chisq_<dtype>` and `chisqgrad_<dtype>` (× three
  ttype variants), `chisqdata_<dtype>` (× three ttype variants), and add
  to `DATATERMS` in both `imager.py` and `imaging/imager_utils.py`.
  Users pass it via `data_term={'<dtype>': weight}`.
- **New geometric model**: add a branch in `model_params`,
  `default_prior`, and four sampling functions
  (`sample_1model_xy`, `sample_1model_uv`, `sample_1model_graduv_uv`,
  `sample_1model_grad_uv`) in `model.py`, plus an `add_<name>` method
  on `Model`. The polarisation hooks `add_pol()` re-use existing
  scaffolding.
- **New scattering model**: subclass / extend `ScatteringModel` to
  override `P_phi`, `Q`, and `Dphi_*`.
- **New calibration step**: write `<step>_cal_scan` (per-scan worker)
  and `<step>_cal` (driver using `Parloop`); the existing four examples
  (`self_cal`, `network_cal`, `pol_cal`, `polgains_cal`) follow the
  same pattern.

---

## 12. Notable internals & gotchas

1. **`from builtins import str/range/object`** at the top of every
   module is a leftover from `future` for Python-2 compatibility.
   Despite the `Programming Language :: Python :: 3.8` classifier and
   the strong NumPy ≥ 1.24 / SciPy ≥ 1.9.3 / Astropy ≥ 5.0.4 floor,
   the code still runs cleanly on modern Pythons.
2. **`pyNFFT` Python-version ceiling**: 3.11 (and NumPy ≤ 1.26.4). For
   newer environments either the `'direct'` or `'fast'` ttype must be
   used, or the user must wait for v2.0 with `finufft`.
3. **`eh.imager_func` is not defined.** It is referenced in
   `examples/example.py:78` and `examples/example_lA_ring.py:43,48,…`
   but no longer exists in the source tree. Use `eh.imager.Imager(...)`
   instead.
4. **OIFITS support was removed** in commit `768a536`; the
   `io/save.py` and `io/load.py` files have no OIFITS path any more.
5. **Polarisation-mode primitives**: when initialising with
   `polrep='circ'`, only `pol_prim ∈ {'RR','LL'}` are allowed. The class
   exposes derived Q/U/V as read-only computations from RR/LL/RL/LR but
   prevents direct assignment (the setters raise unless `polrep='stokes'`).
6. **`Image.observe(...)` polrep semantics** — always returns an
   `Obsdata` in the requested `polrep_obs` (default = the image's own
   polrep). Internally, `add_jones_and_noise` switches to circ, applies
   2×2 Jones matrices, and switches back.
7. **Closure quantities are cached** on `Obsdata` (`amp, bispec, cphase,
   cphase_diag, camp, logcamp, logcamp_diag`); they are invalidated only
   when explicitly recomputed (e.g. `obs.add_amp(avg_time=...)`). Any
   code path that mutates `obs.data` directly should clear these.
8. **`reorder_baselines`** is called automatically in `__init__` to put
   visibilities in the order UVFITS expects on save. Custom data tables
   built outside the package should respect `t1 < t2` baseline order or
   they will be silently reordered.
9. **`hashrandn` etc.** — random-noise generators are *deterministic*
   given (`seed`, baseline-pair, time). Setting `seed=0` is documented
   as broken in the in-source docstring (`obs_simulate.py:961`):
   *"DO NOT set to 0!"*. Use `seed=False` for system-time seeding or
   any non-zero integer.
10. **Sgr A* scattering kernel** in `obs_helpers.sgra_kernel_uv` uses
    the Bower 2006-era constants (`FWHM_MAJ=1.309 mas`,
    `FWHM_MIN=0.64 mas`, `POS_ANG=78°`). The full Johnson stochastic-
    optics framework in `scattering/stochastic_optics.py` defaults to
    Sgr A* parameters from Johnson+ 2018 / Issaoun+ 2019
    (`theta_maj_mas_ref=1.380, theta_min_mas_ref=0.703, POS_ANG=81.9,
    α=1.38, r_in=800 km, r_out=10²⁰ cm`).
11. **`Parloop.run_loop`**: `processes=-1` runs serially (no pool);
    `processes=0` uses `cpu_count()`. The default in many drivers is
    `processes=-1`.
12. **`paramsurvey`** is hard-imported at the top of `survey.py`; users
    who do not need surveys can avoid `import ehtim.survey` to skip
    that dependency.

---

## 13. Known limitations (visible from source / README)

- v1.x is **frozen** with a `pyNFFT` dependency that is itself frozen
  to old Python/NumPy; v2.0 (with `finufft`) is in active development
  but not in this submodule.
- The committed test suite is thin (~5 small files); there is no CI in
  `.github/`. Validation rests on the example scripts and tutorials,
  some of which are stale (`README.rst` says so).
- `examples/example.py` and `examples/example_lA_ring.py` reference an
  `eh.imager_func` symbol that no longer exists.
- Heavy uses of global state in some imaging modules
  (`imaging/dynamical_imaging.py:55-68` keeps `A1_List`, `A2_List`,
  `A3_List`, `data1_List`, …, `sigma3_List` as module-level mutables,
  with the in-source comment *"These parameters are only global to allow
  parallelizing the chi^2 calculation without huge memory overhead. It
  would be nice to do this locally…"*).
- `setup.py` uses `setuptools` with no PEP-517 build-system declaration.
  The `setup.cfg` is essentially empty; there is no `pyproject.toml`.
- No type hints; no `mypy` configuration. Docstrings are Sphinx-style
  but uneven across modules.
- Python-2 transitional shims (`from builtins import object`,
  `__future__` imports) clutter every file but are harmless on
  Python 3.

---

## 14. License

Source: `LICENSE.txt` (full GPLv3 text, 35 KB).
Every source file carries a uniform GPLv3 header asserting the same
license and the copyright statement (mostly `(C) 2018 Andrew Chael`,
with some `(C) 2018 Katie Bouman`, `(C) 2018 Hotaka Shiokawa`,
`(C) 2018 Maciek Wielgus`, `(C) 2020 Michael Johnson` for individual
modules).

---

## 15. Recent commit history (top 20 from `git log`)

```
8d74d8c Update pyNFFT compatibility notes in README
766648d Revise NFFT and pyNFFT installation instructions (#221)
357eca6 Merge pull request #216 from achael/dev
401b313 deleted stray s
bab17aa updated docs
115ab1a updated readme
2580c6e updated readme
768a536 removed oifits support
dd89a5d updated readme
3927096 Merge branch 'main' of https://github.com/achael/eht-imaging
e42a2e9 fixed cff bug
418ee15 fixed cff bug
a8294ce fixed cff bug
4c0dd4e Merge pull request #211 from jberg5/faster-scatter
324617e really restore comment
1899403 move some things around, better naming too
ffacb66 restore comment and remove now rendendant copy() operations
33621f8 First pass at vectorizing equation 9
538d5fa added exception for missing time value in hdf5 header
371146b added exception for missing time value in hdf5 header
```

The visible recent thrust is (a) NFFT/pyNFFT install hygiene and (b)
performance/clean-up of the scattering code (PR #211, "faster-scatter"
work — the vectorisation of *Equation 9* is in
`scattering/stochastic_optics.py`).

---

## 16. Quick-start (literal, drawn from `examples/example.py`)

```python
import ehtim as eh

im  = eh.image.load_txt('models/avery_sgra_eofn.txt')
eht = eh.array.load_txt('arrays/EHT2017.txt')

obs = im.observe(eht,
                 tint=5, tadv=600, tstart=0, tstop=24, bw=4e9,
                 sgrscat=False, ampcal=True, phasecal=False)

beamparams = obs.fit_beam()       # (fwhm_maj, fwhm_min, theta) [rad]
res        = obs.res()            # 1 / longest baseline

empty   = eh.image.make_square(obs, npix=128, fov=im.fovx())
gauss   = empty.add_gauss(im.total_flux(),
                          (200*eh.RADPERUAS, 200*eh.RADPERUAS, 0, 0, 0))

imgr = eh.imager.Imager(obs, gauss, gauss,
                        flux=im.total_flux(),
                        data_term={'bs': 100},
                        reg_term ={'simple': 1, 'flux': 100, 'cm': 50},
                        maxit=100, ttype='nfft')
imgr.make_image_I()
out = imgr.out_last
out.display()
out.save_fits('out.fits')
```

(Compared to the README/examples this swaps the obsolete
`eh.imager_func` for the supported `eh.imager.Imager` flow.)

---

*End of reference.*
