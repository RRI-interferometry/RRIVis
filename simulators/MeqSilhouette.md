# MeqSilhouette — Exhaustive Technical Reference

`MeqSilhouette` is a synthetic-data generator for **mm-/sub-mm Very Long Baseline
Interferometry (VLBI)** observations, primarily targeted at the
**Event Horizon Telescope (EHT)** and similar arrays.  It applies a chain of
realistic propagation-path effects (atmosphere, pointing, instrumental
polarization, antenna gains, bandpass, thermal noise) to user-supplied sky
models and writes the result into a CASA Measurement Set (MS).

This document is an exhaustive, source-cited reference compiled by walking the
repository at `simulators/MeqSilhouette/` (a git submodule of RadioSim).  All
file references are relative to that directory unless noted otherwise.

---

## 1. Overview

| Field | Value | Source |
|-------|-------|--------|
| Name | MeqSilhouette | `setup.py` line 31 |
| Description | "Synthetic Data Generation for mm-VLBI Observations" | `setup.py` line 33 |
| Version (in code) | `3.0` | `meqsilhouette/__init__.py` line 1 |
| Latest git tag | `v3.0.0-alpha.2` | `git tag` output |
| License | GNU GPL v2 | `setup.py` line 37; `LICENSE.md` |
| Primary author | Iniyan Natarajan (`iniyannatarajan@gmail.com`) | `setup.py` lines 29-30 |
| Python | 3.8 (supported / classifier) | `setup.py` line 51, `.readthedocs.yaml` line 11 |
| Upstream URL | https://github.com/rdeane/MeqSilhouette | `README.rst`, `setup.py` line 36 |
| Documentation | https://meqsilhouette.readthedocs.io | `README.rst` line 15 |
| Reference paper (v2) | Natarajan et al. (2022) MNRAS 512, 490 | `docs/source/intro.rst` line 6 |
| Original paper (v1) | Blecher et al. (2017) MNRAS 464, 143 | `docs/source/intro.rst` line 9 |
| Container images | Docker (`Dockerfile`), Singularity (`singularity.def`) | repo root |
| Console entry point | `meqsilhouette` -> `meqsilhouette.driver.run_meqsilhouette:run_meqsilhouette` | `setup.py` line 43 |

### Purpose

Simulate **realistic interferometric visibilities** for mm-VLBI experiments by
combining:

1. A **sky model** (FITS image cube, ASCII source list, or Tigger LSM).
2. An **antenna configuration** (CASA-format ANTENNA table).
3. **Per-station weather and instrument parameters**.
4. A user-driven menu of **corruptions** applied to the visibilities through
   the Radio Interferometer Measurement Equation (RIME):

```
V_pq = G_p · ( Σ_s  E_ps K_ps B_s K_qs^H E_qs^H ) · G_q^H
```
   (`docs/source/components.rst`).

Output is a CASA Measurement Set v2 (`docs/source/outputs.rst` line 7) that can
be processed with downstream calibration/imaging tools (CASA, AIPS,
`eht-imaging`, etc.).

### Languages

* Python 3 (only language present in `meqsilhouette/`).  Two flavours:
  * Plain Python orchestration (driver, framework, utils).
  * One MeqTrees TDL script (`framework/turbo-sim.py`) executed under the
    Cattery / Timba runtime.
* Documentation in reStructuredText (`docs/`).

External native binaries it shells out to: `aatm` (atmosphere), `wsclean`
(predict step for FITS images), CASA `simulator` / `casacore` tables.

---

## 2. Repository layout

```
simulators/MeqSilhouette/
├── .gitignore                       (excludes *.pyc, log-simms.txt, output/)
├── .readthedocs.yaml                Sphinx RTD build (Ubuntu 20.04, py3.8)
├── Dockerfile                       Ubuntu 20.04 + KERN-7 + AATM + MeqSilhouette
├── singularity.def                  Singularity 3.10.2 recipe (mirrors Dockerfile)
├── LICENSE.md                       GNU GPL v2 (full text)
├── README.rst                       Two-line pointer to GitHub + RTD
├── setup.py                         Setuptools install spec (no pyproject.toml)
├── docs/                            Sphinx documentation source
│   ├── Makefile, make.bat
│   ├── requirements.txt             (sphinx-rtd-theme==2.0.0)
│   └── source/
│       ├── conf.py
│       ├── index.rst                TOC
│       ├── intro.rst                Project introduction
│       ├── requirements.rst         Install instructions (apt/KERN-7/AATM)
│       ├── usage.rst                CLI / Singularity / Docker / Jupyter
│       ├── inputs.rst               Full JSON parset schema + sky model + station
│       ├── example.rst              Example JSON file
│       ├── outputs.rst              Output MS columns
│       ├── components.rst           RIME formula
│       ├── pipelines.rst            SYMBA integration
│       ├── contributors.rst         Author list
│       ├── history.rst              Changelog (v1 → v3.0.0-alpha)
│       ├── modules.rst              API toctree
│       ├── meqsilhouette.driver.rst
│       ├── meqsilhouette.framework.rst
│       ├── meqsilhouette.utils.rst
│       └── LSM.png                  Tigger LSM column-format diagram
└── meqsilhouette/                   The Python package
    ├── __init__.py                  __version__ = "3.0"
    ├── driver/                      User-facing entry-point scripts
    │   ├── __init__.py              (empty)
    │   ├── run_meqsilhouette.py     Build new MS from scratch and corrupt
    │   └── readms_runmeqs.py        Use an existing MS and corrupt
    ├── framework/                   Core simulation engine
    │   ├── __init__.py              (empty)
    │   ├── SimCoordinator.py        ★ 1684-line workhorse class
    │   ├── create_ms.py             Empty MS creation via CASA simulator / simms
    │   ├── meqtrees_funcs.py        run_turbosim / run_wsclean / lwimager helpers
    │   ├── turbo-sim.py             MeqTrees TDL forest definition
    │   └── tdlconf.profiles         MeqTrees TDL config profile [turbo-sim]
    ├── utils/
    │   ├── __init__.py              (empty)
    │   ├── add_ant.py               Add station to a CASA ANTENNA table
    │   ├── comm_functions.py        info / warn / abort / print_simulation_summary
    │   └── regularize_ms.py         Pad MS with missing baseline rows (flagged)
    └── data/                        Bundled sample inputs
        ├── eht230.json              Sample JSON parset (EHT, 228 GHz)
        ├── eht_betterweather.antennas    Sample station_info table (8 EHT sites)
        ├── eht_bandpass.txt         Sample bandpass amplitude table
        ├── ANTENNA_EHT2017/         CASA-format ANTENNA subtable (8 stations)
        │   └── table.{dat,f0,info,lock}
        └── sky_models/              Example sky models (FITS + ASCII)
            ├── singlept.txt                       Tigger ASCII single point source
            ├── timevar_point/      t0000..t0009-model.fits
            ├── timepolvar_point/   t000{0,1,2}-{I,Q,U,V}-model.fits
            ├── freqvar_point/      t0000-{0000..0003}-model.fits
            └── old_grmhd_pol/      t0000-{I,Q,U,V}-model.fits
```

`tree -L 3` confirmed; total Python LOC is **3 373** across 13 `.py` files
(see `wc -l` output below).

---

## 3. Installation & dependencies

The `setup.py` install_requires list (lines 12-23):

```
mpltools, seaborn, astLib, astropy, termcolor, numpy, matplotlib,
simms, casatools==6.5.5.21, casadata
```

These pip-installables are only the **Python** half.  The heavy lifting
requires **system packages** that are not pip-installable and which the
container recipes fetch from Ubuntu / KERN-7:

| Dependency | Role | How obtained |
|------------|------|--------------|
| `meqtrees`, `meqtrees-timba`, `Cattery` | RIME node graph engine; runs `turbo-sim.py` | KERN-7 apt (`Dockerfile` line 41) |
| `tigger-lsm`, `python3-astro-tigger`, `python3-astro-tigger-lsm` | Tigger LSM (sky model) reader | KERN-7 apt |
| `casalite`, `python3-casacore`, `casatools`, `casadata` | Measurement Set creation, casacore tables, measures | KERN-7 apt + pip |
| `wsclean` | Visibility predictor for FITS images (`run_wsclean`) | KERN-7 apt |
| `pyxis` | Cattery's task runner; provides `Pyxis.ModSupport`, `mqt`, `im.argo`, `im.lwimager` | KERN-7 apt |
| `aatm` (v0.5) | "ATM" atmospheric absorption / dispersion models (subprocesses `absorption ...` and `dispersive ...`) | Built from source from `https://launchpad.net/aatm/trunk/0.5/+download/aatm-0.5.tar.gz` (`Dockerfile` line 26) |
| `boost` (libboost-program-options-dev / libboost-python-dev) | Required to compile AATM | apt |
| TeX Live (`texlive-latex-extra` etc.) | Paper-quality plot rendering (`rc('text', usetex=True)` in `SimCoordinator.py` line 24) | apt (optional but enabled in containers) |
| `numpy==1.21` (pin) | Avoid `np.BitGenerator` / `np.asscalar` errors in this stack | `Dockerfile` line 52 |

### 3.1 Docker (recommended)

`Dockerfile` (extract):

```dockerfile
FROM ubuntu:20.04 AS spython-base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y wget vim python3-pip gcc python3 unzip git time
RUN apt-get install -y build-essential cmake g++ rsync libboost-python-dev \
    libboost-program-options-dev texlive-latex-extra texlive-fonts-recommended dvipng cm-super
# build aatm manually
RUN cd /opt && wget -c https://launchpad.net/aatm/trunk/0.5/+download/aatm-0.5.tar.gz \
    && tar -xzf aatm-0.5.tar.gz && cd aatm-0.5 && ./configure && make && make install
# KERN-7
RUN apt-get install -y software-properties-common
RUN add-apt-repository -s ppa:kernsuite/kern-7 && apt-add-repository multiverse && apt-add-repository restricted
RUN apt-get install -y meqtrees meqtrees-timba tigger-lsm python3-astro-tigger \
    python3-astro-tigger-lsm casalite wsclean pyxis python3-casacore
RUN pip install numpy==1.21
RUN casa-config --exec update-data
RUN cd /opt && git clone --depth 1 https://github.com/rdeane/MeqSilhouette.git && cd MeqSilhouette && pip install .
ENV MEQTREES_CATTERY_PATH=/usr/lib/python3/dist-packages/Cattery
ENV PYTHONPATH=/usr/local/lib/python3.8/dist-packages:/usr/lib/python3/dist-packages
```

Build / pull (per `docs/source/requirements.rst`):
```bash
docker pull iniyannatarajan/meqsilhouette:focalpy38   # pre-built
# or
docker build -t meqsilhouette .
```

### 3.2 Singularity

`singularity.def` mirrors the Dockerfile and additionally defines
`%runscript` so the SIF can be invoked directly:

```sh
%runscript
    echo "Arguments received: $*"
    meqsilhouette "$@"
```

Build: `sudo singularity build meqsilhouette.sif singularity.def`
(`docs/source/requirements.rst` line 71).

### 3.3 Bare-metal Ubuntu 20.04 + Python 3.8

`docs/source/requirements.rst` lines 17-56 give the canonical recipe
(KERN-7 apt-get, manual AATM build, `pip install .` from a clone).

### 3.4 Known installation pitfalls (`docs/source/requirements.rst` lines 87-108)

* If MeqTrees can't see `TiggerSkyModel`, add Tigger's parent dir to `PYTHONPATH`.
* If AATM is not found, set `LD_LIBRARY_PATH=/path/to/aatm-0.5/lib:$LD_LIBRARY_PATH`.
* Older pinning issues with `scipy==0.17` for qhull, and `pyfits` `gdbm/winreg` import bugs (legacy).

---

## 4. Architecture & runtime pipeline

### 4.1 Layered architecture

```
┌───────────────────────────────────────────────────────────────┐
│  USER LAYER                                                   │
│  • CLI:  $ meqsilhouette obs.json                             │
│  • Python: from meqsilhouette.driver import run_meqsilhouette │
│  • Container:  singularity run / docker run                   │
├───────────────────────────────────────────────────────────────┤
│  DRIVER LAYER  (meqsilhouette/driver/)                        │
│  • run_meqsilhouette.py    – build MS from scratch            │
│  • readms_runmeqs.py       – reuse an existing MS             │
├───────────────────────────────────────────────────────────────┤
│  FRAMEWORK LAYER  (meqsilhouette/framework/)                  │
│  • create_ms.py        – CASA simulator OR simms wrapper      │
│  • SimCoordinator.py   – RIME corruption engine (★ core)      │
│  • meqtrees_funcs.py   – wrappers around mqt/wsclean/lwimager │
│  • turbo-sim.py        – MeqTrees TDL forest                  │
│  • tdlconf.profiles    – TDL config profile                   │
│  • process_input_config.py – JSON parset parser               │
├───────────────────────────────────────────────────────────────┤
│  EXTERNAL TOOLCHAIN                                           │
│  AATM   |  WSClean  |  CASA simulator  |  MeqTrees/Cattery    │
│  (atm)  |  (predict)|  (empty MS)      |  (predict + RIME)    │
├───────────────────────────────────────────────────────────────┤
│  STORAGE                                                      │
│  CASA Measurement Set v2  +  numpy .npy artefacts  +  PNG     │
└───────────────────────────────────────────────────────────────┘
```

### 4.2 End-to-end pipeline

The exact ordering is defined in
`meqsilhouette/driver/run_meqsilhouette.py` lines 39-308 and explicitly
echoed by `info()` at line 203:

```
Start corrupting the perfect visibilities. The corruptions
(if enabled) are applied in the following order:
  1. Pointing errors
  2. Tropospheric effects
  3. Parallactic angle and polarization leakage
  4. Receiver gains
  5. Bandpass effects
  6. Additive thermal noise
```

```
   JSON parset
       │
       ▼
┌──────────────┐  load_json_parameters_into_dictionary()
│ parameters{} │  setup_keyword_dictionary('ms_'/'im_'/'trop_', ...)
└──────┬───────┘
       │
       ▼
┌────────────────────────────────────────────────────┐
│ STAGE 1 — Empty MS scaffolding (create_msv2)       │
│   • CASA simulator: setconfig(antennas), setspw,   │
│     setfield, settimes, observe()                  │
│   • Optional CASA-vs-SYMBA start-time offset corr. │
│   • Add SIGMA_SPECTRUM and WEIGHT_SPECTRUM cols    │
└────────────────┬───────────────────────────────────┘
                 ▼
┌────────────────────────────────────────────────────┐
│ STAGE 2 — uv-sampling & elevation mask             │
│   • SimCoordinator.__init__:                       │
│     load UVW, A0, A1, TIME, FIELD, SPW             │
│     elevation_calc()  parallactic_angle_calc()     │
│     calc_ant_rise_set_times()                      │
│   • write_flag(): mask < elevation_limit           │
│   • Write MOUNT column from station_info col 19    │
└────────────────┬───────────────────────────────────┘
                 ▼
┌────────────────────────────────────────────────────┐
│ STAGE 3 — Forward model (perfect V_pq)             │
│   interferometric_sim():                           │
│     • Tigger ASCII / .lsm.html → run_turbosim()    │
│       (MeqTrees + Siamese fitsimage_sky/Tigger)    │
│     • Directory of FITS  → run_wsclean -predict    │
│       (handles I/Q/U/V, time, frequency variants)  │
│   Result: uncorrupted V → MODEL_DATA + datacolumn  │
└────────────────┬───────────────────────────────────┘
                 ▼
┌────────────────────────────────────────────────────┐
│ STAGE 4 — Corruption chain (right-to-left in RIME) │
│  4.1 Pointing  pointing_constant_offset()          │
│                apply_pointing_amp_error()          │
│  4.2 Tropos.   trop_opacity_attenuate()            │
│                trop_calc_mean_delays()             │
│                trop_generate_turbulence_phase_errs │
│                trop_calc_fixdelay_phase_offsets()  │
│                apply_phase_errors()                │
│  4.3 Pol.      add_pol_leakage_manual() (P + D)    │
│  4.4 Gains     add_gjones_manual() (G)             │
│  4.5 Bandpass  add_bjones_manual() (B)             │
│  4.6 Noise     add_noise() — sky + receiver        │
└────────────────┬───────────────────────────────────┘
                 ▼
┌────────────────────────────────────────────────────┐
│ STAGE 5 — Output products                          │
│   • <run>.MS  (DATA = corrupted, MODEL_DATA = clean│
│   • <run>.uvfits (optional, exportuvfits)          │
│   • plots/*.png (uv-coverage, elevation, ...)      │
│   • opacity / emissivity / transmission / *terms*  │
│     numpy arrays under $OUTDIR                     │
│   • atm_output/{ATMstring_antN.txt, *atm_abs.txt,  │
│     *atm_disp.txt, sky_noise_*, sefd_matrix_*}     │
│   • inputs/  (verbatim copy of JSON, antennas, …)  │
└────────────────────────────────────────────────────┘
```

The pipeline is implemented sequentially in a single thread (the only
parallelism is `mqt.MULTITHREAD = 32` for the MeqTrees solver in
`run_turbosim`, `meqtrees_funcs.py` line 46) and `wsclean`'s internal
threading.

---

## 5. Public CLI

### 5.1 `meqsilhouette` command

Defined as a console script in `setup.py` line 43:

```
entry_points={'console_scripts':
    ['meqsilhouette=meqsilhouette.driver.run_meqsilhouette:run_meqsilhouette']}
```

Invocation forms (`docs/source/usage.rst`):

```bash
meqsilhouette obs_settings.json                # CLI
singularity run meqsilhouette.sif obs.json     # via SIF runscript
docker run -v ~/data:/meqsdata meqsilhouette \
       meqsilhouette /meqsdata/obs.json        # via Docker
python -c "from meqsilhouette.driver import run_meqsilhouette; \
           run_meqsilhouette.run_meqsilhouette('obs.json')"
```

Argument handling (`run_meqsilhouette.py` lines 41-46):
* Single positional arg: path to a JSON parset.
* No flags (`-v`, `-h` etc.) — the script aborts unless `len(sys.argv)==2`.

### 5.2 `readms_runmeqs.py` — corrupt an existing MS

A second driver that bypasses the MS-creation stage and instead regularizes
an externally-supplied MS (`driver/readms_runmeqs.py`):

```python
def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("json", help="Name of input JSON parset file")
    p.add_argument("ms",   help="Input MS name")
```

CLI:
```bash
python readms_runmeqs.py obs_settings.json existing.ms
```

Programmatic:
```python
from meqsilhouette.driver import readms_runmeqs
readms_runmeqs.readms_runmeqs("obs.json", "existing.ms")
```

The MS is copied to `<outdir>/inputs/` and run through `regularize_ms()`
(`utils/regularize_ms.py`) which inserts fully-flagged rows for any
baseline missing in some timestamp.  This is critical for tropospheric
turbulence (Cholesky decomposition needs a regular grid;
`docs/source/usage.rst` note line 25).

### 5.3 `add_ant.py` — utility (not a console script)

Stand-alone script for editing CASA ANTENNA tables.  Run as:
```bash
python -m meqsilhouette.utils.add_ant <input_table> <output_table>
```
Hardcodes example dictionaries for AMT (`add_ant.py` lines 122-133),
MeerKAT, and APEX.

---

## 6. Configuration schema (JSON parset)

There is **no YAML or TOML** form — every run reads a JSON file with no
schema enforcement beyond manual checks in
`process_input_config.py`.  Keys are flat, but conceptually grouped by
prefix.  The full schema is documented in
`docs/source/inputs.rst` lines 19-238.

### 6.1 General I/O

| Key | Type | Units | Description |
|-----|------|-------|-------------|
| `outdirname` | str | — | Output directory (writable). Set on `v.OUTDIR`. |
| `input_fitsimage` | str | — | Path stem (no `.txt`/`.html`) **or** directory of FITS images. |
| `input_fitspol` | bool | — | True if directory contains polarised IQUV stack. |
| `input_changroups` | int | — | Number of frequency groups for FITS predict. |
| `output_to_logfile` | bool | — | Redirect Pyxis output to `<OUTDIR>/meqsilhouette-logfile.txt`. |
| `add_thermal_noise` | bool | — | Enable receiver thermal noise. |
| `make_image` | bool | — | Make dirty image with lwimager (uses `im_*` keys). |
| `exportuvfits` | bool | — | Export MS to UVFITS. |
| `corr_quantbits` | int | — | 1 → η=0.636, 2 → η=0.88 (`run_meqsilhouette.py` line 175). |
| `predict_oversampling` | int | — | Odd integer (e.g. 8191), passed to WSClean `-oversampling`. |
| `predict_seed` | int | — | RNG seed for non-atmospheric corruptions; -1 for non-deterministic. |
| `atm_seed` | int | — | RNG seed for tropospheric/sky-noise; -1 for non-deterministic. |
| `station_info` | str | — | Path to ASCII station-info table. |
| `bandpass_table` | str | — | Path to ASCII bandpass-amplitude table. |
| `bandpass_freq_interp_order` | int | — | Spline order, 1-5. |

### 6.2 MS group (prefix `ms_`)

| Key | Units | Notes |
|-----|-------|-------|
| `ms_antenna_table` | — | CASA ANTENNA subtable directory (e.g. `ANTENNA_EHT2017`). |
| `ms_datacolumn` | — | "DATA" / "CORRECTED_DATA" / "MODEL_DATA". Avoid MODEL_DATA so the clean copy is preserved. |
| `ms_RA`, `ms_DEC` | deg | Phase centre. |
| `ms_polproducts` | — | "RR RL LR LL" or "XX XY YX YY" (CASA simulator string). |
| `ms_nu` | GHz | Centre frequency. |
| `ms_dnu` | GHz | Total bandwidth. |
| `ms_nchan` | — | Number of channels. |
| `ms_obslength` | hours | Total duration. |
| `ms_tint` | seconds | Integration time. |
| `ms_StartTime` | — | "UTC,YYYY/MM/DD/hh:mm:ss.ss". |
| `ms_nscan` | — | Number of scans. |
| `ms_scan_lag` | hours | Deprecated, retained for backward compat. |
| `ms_makeplots` | bool | Generate uv-coverage / elevation plots. |
| `ms_correctCASAoffset` | bool | Two-pass start-time offset correction (see §10.2). |

### 6.3 Imaging group (prefix `im_`)

| Key | Notes |
|-----|-------|
| `im_cellsize` | "3e-6arcsec" — string with units. |
| `im_npix` | Image pixels. |
| `im_stokes` | "I" / "Q" / "U" / "V". |
| `im_weight` | "uniform" / "natural" / "briggs". |

### 6.4 Tropospheric group (prefix `trop_`)

| Key | Description |
|-----|-------------|
| `trop_enabled` | Master switch. |
| `trop_wetonly` | Use only the wet (PWV) component, skip dry. |
| `trop_attenuate` | Apply opacity-derived amplitude attenuation. |
| `trop_noise` | Inject sky-temperature noise (uses elevation-corrected emissivity). |
| `trop_turbulence` | Kolmogorov phase fluctuations (β=5/3). |
| `trop_mean_delay` | Insert mean dry+wet delay (time-variability via elevation). |
| `trop_fixdelays` | Insert constant per-station delays (testing fringe-fitters). |
| `trop_fixdelay_max_picosec` | Deprecated cap on the constant delays. |
| `trop_makeplots` | Generate troposphere PNGs. |
| `trop_percentage_calibration_error` | Deprecated. |

### 6.5 Pointing group (prefix `pointing_`)

| Key | Description |
|-----|-------------|
| `pointing_enabled` | Master switch. |
| `pointing_time_per_mispoint` | Minutes per pointing epoch (constant within epoch). |
| `pointing_makeplots` | Plot pointing offsets and amplitude errors. |

### 6.6 uv-Jones group

| Key | Description |
|-----|-------------|
| `uvjones_g_on` | Add direction-independent complex gains G. |
| `uvjones_d_on` | Add polarization leakage D and parallactic-angle rotation P. |
| `parang_corrected` | True ⇒ visibilities returned in sky frame (Leppanen 1995 2-θ rotation); False ⇒ antenna frame. |
| `bandpass_enabled` | Read `bandpass_table`, interpolate, apply B. |
| `bandpass_makeplots` | Bandpass amplitude plots. |
| `elevation_limit` | radians; baselines below this are flagged in FLAG. |

### 6.7 Provenance / virtual keys

`process_input_config.params_refactoring()` injects one **derived** key:
```python
_params['wavelength'] = 1e-9 * 299792458 / _params['ms_nu']   # metres
```
(`framework/process_input_config.py` line 73).

### 6.8 Bundled sample (`meqsilhouette/data/eht230.json`)

```json
{
  "outdirname":"/meqsdata/output/wsclean_timevar_point_allcorrupt",
  "input_fitsimage":"/meqsdata/sky_models/timevar_point",
  "input_fitspol":0, "input_changroups":1,
  "add_thermal_noise":1, "make_image":0, "exportuvfits":0,
  "station_info":"/meqsdata/eht_betterweather.antennas",
  "bandpass_enabled":1, "bandpass_table":"/meqsdata/eht_bandpass.txt",
  "bandpass_freq_interp_order":1, "bandpass_makeplots":1,
  "elevation_limit":0.174, "corr_quantbits":2,
  "predict_oversampling":8191, "predict_seed":42, "atm_seed":300,
  "ms_antenna_table":"/meqsdata/ANTENNA_EHT2017",
  "ms_datacolumn":"DATA",
  "ms_RA":187.70591666666667, "ms_DEC":12.391122222222222,
  "ms_polproducts":"RR RL LR LL",
  "ms_nu":228, "ms_dnu":2, "ms_nchan":64,
  "ms_obslength":4, "ms_tint":10,
  "ms_StartTime":"UTC,2017/04/11/00:32:00.00",
  "ms_nscan":1, "ms_scan_lag":0,
  "ms_makeplots":1, "ms_correctCASAoffset":1,
  "im_cellsize":"3e-6arcsec","im_npix":64,"im_stokes":"I","im_weight":"uniform",
  "trop_enabled":1, "trop_wetonly":0, "trop_attenuate":1, "trop_noise":1,
  "trop_turbulence":1, "trop_mean_delay":1,
  "trop_percentage_calibration_error":100, "trop_fixdelays":0,
  "trop_fixdelay_max_picosec":0, "trop_makeplots":0,
  "pointing_enabled":1, "pointing_time_per_mispoint":10, "pointing_makeplots":1,
  "uvjones_g_on":1, "uvjones_d_on":1, "parang_corrected":1
}
```
Pointing (M87) and timestamp (April 2017 EHT campaign) match the Sgr A*/M87
EHT papers.

---

## 7. File-by-file breakdown

### 7.1 `meqsilhouette/__init__.py`
```python
__version__ = "3.0"
```
Single-line module that `setup.py` imports during install (line 3).

### 7.2 `meqsilhouette/driver/run_meqsilhouette.py` (315 lines)

The main driver.  Function `run_meqsilhouette(config=None)`.

Key responsibilities:

1. Resolve the JSON parset path (positional CLI arg or function kwarg).
2. Load it via `load_json_parameters_into_dictionary()`.
3. Split into `ms_dict`, `im_dict`, `trop_dict` via prefix filtering
   (`process_input_config.setup_keyword_dictionary`).
4. Compose `ms_config_string` with antenna table name, sky model name,
   RA/DEC, polproducts, ν/Δν/nchan/tint/obslength — used as the MS
   filename suffix (line 55-60).
5. Create `OUTDIR/plots/` and `OUTDIR/inputs/`; copy parset, sky model,
   `station_info`, `bandpass_table`, `ms_antenna_table` into `inputs/`.
6. Load station_info table as `np.complex128`; split into 17 typed
   arrays: `T_rx, pwv, gpress, gtemp, coherence_time, pointing_rms,
   PB_FWHM230, aperture_eff, gR_mean, gR_std, gL_mean, gL_std, dR_mean,
   dR_std, dL_mean, dL_std, feed_angle` (lines 122-135).  Real parts
   extracted for non-complex columns; G/D means and stds keep imaginary
   part.  Mount type is read from column 19 as string.
7. Cross-check that `station_info` and the CASA ANTENNA `STATION` column
   list the same names in the same order; abort otherwise.
8. Map `corr_quantbits ∈ {1, 2}` to correlator efficiency η ∈ {0.636,
   0.88} per TMS3 §8.3 (line 175).
9. Call `create_msv2(MS, input_fitsimage, ms_dict)`.
10. Write the `MOUNT` column on the new MS using station_info col 19.
11. Instantiate `SimCoordinator(...)` (29 positional args), call
    `interferometric_sim()` to run the predict step.
12. Apply optional corruptions in the documented order.
13. Optional dirty image (`make_dirty_image_lwimager`) and uvfits
    export (`im.argo.icasa('exportuvfits', ...)`).
14. Print elapsed time.

Pyxis globals used: `v.FRAMEWORKDIR, v.OUTDIR, v.PLOTDIR, v.MS, v.LOG`.
The `II(...)` macro (lazy interpolation of `$VAR` strings) is used for
output paths.

### 7.3 `meqsilhouette/driver/readms_runmeqs.py` (305 lines)

Twin of `run_meqsilhouette` for an existing MS.  Differences:

* `argparse` based with two positional args (`json`, `ms`).
* Calls `regularize_ms(inms_abspath)` to insert fully-flagged rows for
  baselines missing in any timestamp.  The result becomes `v.MS`.
* Replaces `ms_dict['antenna_table']` with the regularised MS's ANTENNA
  subtable.
* Reads `sefd` directly from the station_info file (instead of `T_rx`
  in the new driver) — i.e. expects the station_info schema before the
  recent rewrite.  Note: this driver still uses `add_receiver_noise()`
  and `add_weights()` — the **older** noise path — whereas the new
  driver uses the unified `add_noise(tropnoise, thermalnoise)` method.

### 7.4 `meqsilhouette/framework/SimCoordinator.py` (1684 lines)

The single class `SimCoordinator` is the heart of MeqSilhouette.  Below
is the public method inventory (from `grep` of `def` signatures), with
purpose:

| Method | Purpose |
|--------|---------|
| `__init__` | Open MS, load UVW/A0/A1/TIME/SPW/FIELD; compute elevation, parallactic angle, baseline dict; flag below `elevation_limit`; compute `SEFD_rx`, `dish_area`, opacity, emissivity, transmission. |
| `interferometric_sim` | Run forward model (Tigger ASCII / .lsm.html via MeqTrees `turbo-sim.py`, FITS dirs via `wsclean -predict`). Handles I-only, polarised, time-variable, freq-variable cubes. |
| `copy_MS(new_name)` | `cp -r` MS to new path. |
| `save_data` | Write `self.data` back to `output_column`. |
| `compute_receiver_rms` | Per-baseline rms = (1/η) √(SEFD_p × SEFD_q / 2 Δt Δν). |
| `add_weights(extra)` | Populate `SIGMA`, `SIGMA_SPECTRUM`, `WEIGHT=1/σ²`, `WEIGHT_SPECTRUM`. |
| `add_receiver_noise(load=None)` | Older noise path (used by `readms_runmeqs.py`); generates Gaussian thermal noise, saves `receiver_noise_timestamp_*.npy`. |
| `make_baseline_dictionary` | `{(p,q): np.where(A0==p & A1==q)[0]}` indexing. |
| `parallactic_angle_calc` | Per-antenna χ_pa(t) using `pyrap.measures` HADEC and antenna latitude. |
| `elevation_calc` | Per-antenna alt(t) via spherical trig (latitude from antenna z over R⊕). |
| `calc_ant_rise_set_times` | First/last MJD where elevation is non-NaN per station (used to mask pointing during stow). |
| `calculate_baseline_min_elevation` / `_mean_elevation` | For uv-coverage colourisation plots. |
| `write_flag(elevation_limit)` | Set FLAG=True for any sample with either antenna below the limit. |
| `trop_opacity_attenuate` | Multiply visibilities by √(T_p · T_q), with T = exp(−τ/sin(elev)). |
| `trop_return_opacity_emissivity` | Shells out to `aatm absorption …` per antenna; parses `[freq, dry, wet, emissivity]` columns; obeys `trop_wetonly`; cross-checks #channels (catches an AATM channel-count bug, lines 565-573). |
| `trop_add_sky_noise(load=None)` | Older sky-noise path (used by `readms_runmeqs.py`). Builds elevation-dependent SEFD = 2k/A_eff · 10²⁶ · ε(1−e^(−τ/sinθ)) and draws Gaussian noise. |
| `trop_generate_turbulence_phase_errors` | Kolmogorov turbulence: build structure function D(t) = (t/τ_coh)^(5/3), autocorr C = ½(D_max−D), Cholesky L of covariance; multiplies a Gaussian RV; scales 1/√sin(elev) and frequency f/f₀. |
| `trop_calc_fixdelay_phase_offsets` | Time-invariant delays = ATM_dispersion / c, divided by sin(elev); converted to phase = 2πfd; the per-antenna mean over time is used. |
| `trop_ATM_dispersion` | Shells out to `aatm dispersive …` per antenna; reads `[wet_non_disp, wet_disp, dry_non_disp]` and sums per `trop_wetonly`. Saves `delay_norm_*.npy`. |
| `trop_calc_mean_delays` | Time-variable delay = ATM_dispersion/c divided by sin(elev) per timestamp; converts to phase. |
| `trop_phase_corrupt(...)` | Legacy phase corruption with optional normalisation (largely deprecated). |
| `apply_phase_errors(combined)` | Multiplies V_pq by exp(i(φ_p − φ_q)). |
| `trop_plots` | Up to ~10 PNGs: zenith τ vs ν, T_atm vs ν, transmission(t,ν), turbulent phase, mean phasedelay, etc. |
| `pointing_constant_offset(rms, timescale, PB_FWHM230)` | Discrete pointing epochs: PB_FWHM scaled to ν̄, num_epochs = obslength/timescale, draws `pointing_offsets ~ N(0, ptg_rms)`, masks epochs outside antenna rise/set, then converts to amplitude error via Gaussian PB: A = exp(−½(ρ/(FWHM/2.35))²). |
| `apply_pointing_amp_error` | Multiplies V_pq by A_p · A_q within each epoch's time window. |
| `plot_pointing_errors` | Three PNGs: ρ(t), A(t), A(ρ). |
| `add_bjones_manual` | Read `bandpass_table` (per-station amplitudes per representative frequency for R and L), interpolate via `InterpolatedUnivariateSpline(order=bandpass_freq_interp_order)`, attach random phase ∈ [−30°, +30°], build diag(B_R, B_L) per (ant, ν), apply via V → B_p · V · B_q^H per baseline per channel. |
| `make_bandpass_plots` | Two PNGs (R-pol amp, L-pol amp vs ν / channel). |
| `add_pol_leakage_manual` | Two branches selected by `parang_corrected`: <br>• False (antenna frame): build P-Jones diag(e^∓i(α+χ_pa)) (or with ±elevation correction for Nasmyth-L/R mounts), independent D-Jones diag with Gaussian-distributed off-diagonals, V → D P V P^H D^H. <br>• True (sky frame, Leppanen 1995): combined `pol_leak_mat` with off-diagonals d·e^±i·2(α+χ_pa[±elev]); V → M V M^H. |
| `make_pol_plots` | χ_pa(t) per antenna. |
| `add_gjones_manual` | Per-antenna time-varying complex G ~ N(g_R/L_mean, g_R/L_std) on the diagonal; V → G_p V G_q^H. |
| `add_noise(tropnoise, thermalnoise)` | New unified noise method. Supports thermal-only, tropos-only, or both.  Computes `skytemp_from_emissivity = ε / (1 − e^−τ)` (so the *atmospheric brightness temperature* is recovered from AATM emissivity), then SEFD = 2k(T_rx + T_sky·(1 − e^(−τ/sinθ))) / A_eff · 10²⁶, draws Gaussian thermal+sky noise, sums in chunks of 100 000 rows, and writes SIGMA / WEIGHT columns. Saves many `.npy` provenance files including `T_rx_*`, `sefd_rx_*`, `sefd_matrix_*`, `dish_area_*`. |
| `make_ms_plots` | uv-coverage with per-baseline colour legend, uv-coverage colourised by min/mean baseline elevation, amp vs uv-distance, percent vis per uv-bin, sensitivity per uv-bin, elevation vs time. |

The class hardcodes `chunksize = 100000` rows for memory-bounded
operations (line 50).

#### Sky model dispatch (`interferometric_sim`)
```
if exists(input_fitsimage + '.txt'):       # ASCII LSM
    -> run_turbosim
elif exists(input_fitsimage + '.html'):    # Tigger LSM
    -> run_turbosim
elif isdir(input_fitsimage):               # FITS cube directory
    glob and decide num_images / vis_per_image
    iterate t0000-, t0001-, ... and call:
        run_wsclean(stem, input_fitspol, input_changroups, startvis, endvis, oversampling)
```

### 7.5 `meqsilhouette/framework/create_ms.py` (309 lines)

Two parallel implementations of the empty-MS scaffolding:

* **`create_msv2(msname, input_fits, ms)`** (recommended; called from
  `run_meqsilhouette.py` line 181).  Uses `casatools.simulator` directly:
  ```python
  sm = simulator(); tb = table(); me = measures()
  sm.open(msname)
  sm.setauto(autocorrwt=0.0)
  sm.setconfig(telescopename='VLBA', x=…, y=…, z=…, dishdiameter=…,
               mount=…, antname=…, padname=…, coordsystem='global',
               referencelocation=obspos)
  sm.setlimits(shadowlimit=0.0, elevationlimit=0.0)
  sm.setspwindow(spwname=f'{int(startfreq)}GHz_BAND', freq=f'{startfreq}GHz',
                 deltafreq=f'{deltafreq}GHz', freqresolution=f'{deltafreq}GHz',
                 nchannels=nchan, stokes=stokes)
  sm.setfield(...)
  sm.settimes(integrationtime=tint, usehourangle=False,
              referencetime=me.epoch(*obs_starttime))
  for scan: sm.observe(...)   # nscan iterations
  ```
  After creation it adds `SIGMA_SPECTRUM` and `WEIGHT_SPECTRUM` array
  columns (initial value 1.0).
  > Note (line 50): `telname = "VLBA"` is **hardcoded** because CASA
  > simulator only knows a fixed list of observatories — the user-chosen
  > antennas are still injected via `setconfig`.

* **`create_ms(msname, input_fits, ms_dict)`** (legacy).  Shells out to
  the standalone `simms` CLI via `return_simms_string()` (line 229).
  Patches STATION/FIELD/SOURCE/SPW name columns and adds the spectrum
  columns.  Kept as a fallback.

* **`compute_casa_offset()`** (line 115).  CASA introduces a spurious
  start-time offset; this routine simulates a tiny throw-away MS, reads
  the actual `TIME` and `EXPOSURE`, compares against the JSON
  `ms_StartTime`, and writes a corrected start-time to
  `<OUTDIR>/CASAcorrectedStartTime.txt` and the residual to
  `<OUTDIR>/CASAtimeOffset.txt`.  Triggered by `ms_correctCASAoffset=1`.

### 7.6 `meqsilhouette/framework/meqtrees_funcs.py` (59 lines)

Thin wrappers around external tools.

* **`run_wsclean(input_fitsimage, input_fitspol, input_changroups, startvis, endvis, oversampling)`** — `subprocess.check_call(['wsclean', '-channels-out', N, '-predict', '-name', stem, '-interval', sv, ev, '-oversampling', os, '-no-small-inversion', msname])`.  When `input_fitspol=1`, adds `-pol I,Q,U,V -no-reorder`.
* **`copy_between_cols(dest, src)`** — straight `pt.table(...).putcol(dest, src_data)`.
* **`run_turbosim(input_fitsimage, output_column, taql_string)`** — sets a `Meow.MSUtils.MSSelector`-style options dict (`ms_sel.msname`, `me.sky.siamese_oms_fitsimage_sky` / `me.sky.tiggerskymodel`, etc.), then runs `mqt.run(script=$FRAMEWORKDIR/turbo-sim.py, config=$FRAMEWORKDIR/tdlconf.profiles, section='turbo-sim', job='_simulate_MS', options=options)`.  Sets `mqt.MULTITHREAD = 32`.
* **`make_dirty_image_lwimager(im_dict, ms_dict)`** — calls `im.lwimager.make_image(column=…, dirty_image='${OUTDIR>/}${MS:BASE}-dirty_map.fits', dirty=True, **im_dict)`.
* **`make_image_wsclean()`** — placeholder (`print('todo')`).

### 7.7 `meqsilhouette/framework/turbo-sim.py` (252 lines)

A **MeqTrees TDL** (Tree Description Language) script — the only file in
the repo that runs inside the Cattery / Timba runtime.  It builds a node
forest using:

* Sky models (Siamese):
  * `Siamese.OMS.tigger_lsm.TiggerSkyModel` (preferred for ASCII / Tigger LSM).
  * `Siamese.OMS.gridded_sky`, `azel_sky`, `transient_sky`, `fitsimage_sky`.
* Sky-Jones terms (lines 95-120):
  * `Ncorr` — `oms_n_inverse` n-term.
  * `Z` — ionosphere (Lions ZJones, `oms_ionosphere`, `oms_ionosphere2`).
  * `L` — parallactic-angle / dipole rotation
    (`Siamese.OMS.rotation.Rotation`, `oms_dipole_projection`).
  * `E` — beam: analytic, FITS, EMSS polar, PAF, FITS0, VLA, LOFAR; with
    `oms_pointing_errors` for pointing residuals.
* uv-Jones terms (lines 122-136):
  * `P` — feed orientation (`Siamese.OMS.feed_angle`).
  * `D` — leakage (`Siamese.OMS.leakage.Leakage('D')`).
  * `G` — gain (`oms_gain_models`).
  * `iP` — feed-angle correction.
* Optional Gaussian noise term (Meq.GaussNoise, dims=[2,2], complex).
* Compile/run-time selectable mode: `sim only` / `add to MS` / `subtract from MS`.
* `_simulate_MS` job: `mqs.execute('VisDataMux', mssel.create_io_request())`.

This TDL is invoked exclusively when the sky model is ASCII or Tigger
LSM (FITS images are predicted by WSClean, which is outside MeqTrees).

### 7.8 `meqsilhouette/framework/process_input_config.py` (95 lines)

Three small helpers:

* `read_json_files(config)` — `json.load`, drop empty strings and any
  key equal to `"#"`, force keys to `str`.
* `params_refactoring(p)` — adds `wavelength = c / (ms_nu × 1e9)` (m).
* `setup_keyword_dictionary(prefix, d)` — filters keys with the prefix
  and strips it (`'ms_RA' → 'RA'`).
* `load_json_parameters_into_dictionary(config)` — pipeline of the above.

### 7.9 `meqsilhouette/framework/tdlconf.profiles`
```ini
[turbo-sim]
img_sel.imaging_arcmin = 8.5e-06
img_sel.imaging_column = CORRECTED_DATA
```
A 3-line MeqTrees profile loaded by `mqt.run` to default the imaging
column for the `turbo-sim` job.

### 7.10 `meqsilhouette/utils/comm_functions.py`

Coloured terminal logger (`termcolor`):
* `info(s)` — green ` >>> MEQSILHOUETTE INFO <<< `
* `warn(s)` — yellow ` >> MEQSILHOUETTE WARNING <<`
* `abort(s, exception=SystemExit)` — red, raises.
* `print_simulation_summary(ms_dict, im_dict)` — pretty header.

### 7.11 `meqsilhouette/utils/regularize_ms.py` (161 lines)

Inserts dummy, fully-flagged rows for any baseline absent in some
timestamps.  Adapted from `casacore.tables.msutil.msregularize` but
augmented to handle antennas missing from the MAIN table while still
present in the ANTENNA subtable (`docstring`, lines 5-7).  Algorithm:

1. Build `combos` = list of all baselines that involve at least one
   missing antenna.
2. Materialise a unique-baseline reference table (`uniqants.ms`) and add
   the missing combos via TaQL inserts.
3. For each `(TIME, DATA_DESC_ID)` chunk, compute the missing rows and
   write them to `<msprefix>_missing.MS` with FLAG=True everywhere.
4. Concatenate + sort by `TIME, DATA_DESC_ID, ANTENNA1, ANTENNA2` to
   produce `<msprefix>_regularized.MS`.

### 7.12 `meqsilhouette/utils/add_ant.py` (161 lines)

Stand-alone helper to grow a CASA ANTENNA table.  Provides
`latlonh_2_xyz(lat, lon, h)` (WGS84) and `AddAnt(in_tab, out_tab,
new_dict)` which copies the table, adds one row, and writes per-column.
Hardcoded example dicts for AMT (Africa Millimetre Telescope), MeerKAT,
and APEX show the expected schema:
```python
{'OFFSET': [0,0,0], 'POSITION': xyz, 'TYPE': 'GROUND-BASED',
 'DISH_DIAMETER': 15, 'FLAG_ROW': 0, 'MOUNT': 'alt-az',
 'NAME': 'AMT', 'STATION': 'AMT'}
```

---

## 8. Core algorithms

### 8.1 Atmospheric absorption / opacity / emission (AATM coupling)

Per-antenna, per-channel opacity τ and emissivity ε are obtained by
forking AATM's `absorption` binary (`SimCoordinator.trop_return_opacity_emissivity`):

```
absorption --fmin {GHz} --fmax {GHz} --fstep {GHz}
           --pwv {mm} --gpress {mb} --gtemp {K}
```
The 4-column output `[freq, dry, wet, emissivity]` is parsed; opacity is
`wet` if `trop_wetonly=1` else `dry+wet`.  Outputs are persisted under
`<OUTDIR>/atm_output/{ant}atm_abs.txt` and `ATMstring_ant{ant}.txt`.
Transmission `T = exp(−τ)`; elevation-corrected attenuation factor for
visibility V_pq is `√(T_p · T_q)` with `T_a = exp(−τ_a / sin θ_a)`
(`trop_opacity_attenuate`).

Sky brightness temperature is recovered from emissivity via
`T_sky = ε / (1 − exp(−τ))` (line 1317).  This is then attenuated by
elevation when computing the sky-noise contribution to SEFD.

### 8.2 Tropospheric phase delays (mean + dispersive)

Path length from `aatm dispersive`:
```
dispersive --fmin {GHz} --fmax {GHz} --fstep {GHz}
           --pwv {mm} --gpress {mb} --gtemp {K}
```
columns `[freq, wet_non_disp, wet_disp, dry_non_disp]` (line 695-697).
With `trop_wetonly=1` the path = `wet_non_disp + wet_disp`; else add
`dry_non_disp`.  Convert to delay τ_d = L/c, then phase
ϕ(t,ν,a) = 2π · f · (τ_d(ν,a) / sin θ_a(t)).

`trop_calc_mean_delays` keeps the time variability (only sin θ varies
with t).  `trop_calc_fixdelay_phase_offsets` averages over time and
broadcasts the same delay to every timestamp — useful for fringe-fitting
tests.

### 8.3 Kolmogorov tropospheric turbulence

Per-antenna, in `trop_generate_turbulence_phase_errors` (lines 617-648):
```
β = 5/3
D(t) = (t / τ_coh)^β        # structure function
C(t) = | ½ (D(t_max) − D(t)) |  # autocorrelation, clipped
S    = C(|i − j|)             # covariance matrix from |Δt|
L    = cholesky(S)
g_a  ~ N(0, 1, size=N_t)
ϕ_a(t)    = (1/√sin θ_a(t)) · L · g_a
ϕ_a(t,ν)  = ϕ_a(t,ν₀) · (ν / ν₀)
```
Independent realisations per antenna (no inter-antenna correlation);
seeded from `atm_seed`.

### 8.4 Pointing errors

`pointing_constant_offset` (lines 905-950):
* PB FWHM scaled from 230 GHz to ν̄: `PB_FWHM(ν̄) = PB_FWHM230 · 230e9/ν̄`.
* Number of pointing epochs `N_ep = round(obslength / Δt_ptg)`.
* Per epoch: `ρ_a ~ N(0, ptg_rms_a)` (arcsec).  Epochs outside antenna
  rise/set are NaN-masked.
* Amplitude error A = exp(−½(ρ/(FWHM/2.35))²) for the Gaussian PB model
  (`cosine3` is sketched but disabled).

`apply_pointing_amp_error`: V_pq(t) ← V_pq(t) · A_p(epoch) · A_q(epoch)
across all polarizations and channels (no frequency dependence of the
PB within a channel range).

### 8.5 G / D / B / P Jones

* **G (`add_gjones_manual`)** — diag(g_R, g_L) per antenna per timestep,
  drawn independently from `N(gR_mean, gR_std)` (complex) and
  `N(gL_mean, gL_std)`.  V → G_p V G_q^H.
* **D + P (`add_pol_leakage_manual`)** — Two operating modes:
  * `parang_corrected=False` (antenna frame): explicit P-Jones rotation
    matrices `e^∓i(α+χ_pa)` for ALT-AZ, with extra ∓elev for
    Nasmyth-L/R mounts.  D-Jones is a unit-diagonal matrix with
    Gaussian-distributed off-diagonals drawn from
    `N(d{R,L}_mean, d{R,L}_std)` per channel.  V → D P V P^H D^H.
  * `parang_corrected=True` (sky frame, Leppanen 1995): single combined
    matrix with off-diagonals `d·e^±i·2(α+χ_pa)` (or with ∓elev for
    Nasmyth).  V → M V M^H.
* **B (`add_bjones_manual`)** — Per-antenna spline (`InterpolatedUnivariateSpline`,
  user-chosen `bandpass_freq_interp_order`) over the per-station
  amplitudes provided in `bandpass_table`.  Per-channel random phase
  drawn uniformly from [−30°, +30°] is attached.  Stored as diag(B_R, B_L);
  V → B_p V B_q^H per channel.

### 8.6 Noise

Per-baseline receiver rms (TMS3 ch.6):
```
σ_pq = (1/η) · √( SEFD_p · SEFD_q / (2 · Δt · Δν) )
```
where `SEFD_a = 2 k T_rx_a · 10²⁶ / A_eff_a` and `A_eff_a = ap_eff · π
(D/2)²`.

When `trop_enabled=1`, the elevation-dependent atmospheric SEFD is
added in quadrature (lines 1320-1336):
```
SEFD_a(t,ν) = 2 k (T_rx + T_sky · (1 − e^(−τ/sinθ))) / A_eff · 10²⁶
T_sky       = ε / (1 − e^(−τ))     # zenith
```
σ is realised via independent Gaussian draws for real and imaginary
parts of every visibility (per (row, channel, correlation)).
`MEMORY` errors are caught and chunked at `chunksize = 100 000` rows.

`SIGMA` and `WEIGHT = 1/σ²` columns are written; `SIGMA_SPECTRUM` and
`WEIGHT_SPECTRUM` mirror the per-channel array if those columns exist.

### 8.7 Quantisation efficiency

Hardcoded mapping (`run_meqsilhouette.py` line 175):
```
corr_quantbits = 1 → η = 0.636      # 1-bit (2-level) digitisation
corr_quantbits = 2 → η = 0.88       # 2-bit (4-level) digitisation
```
(TMS, 2017, Section 8.3.)

### 8.8 Random number generation

Two independent `np.random.default_rng` instances:
* `self.rng_predict` — seeded from `predict_seed`; used for thermal
  noise, G/D/B sampling, bandpass random phase, pointing offsets.
* `self.rng_atm` — seeded from `atm_seed`; used for sky-noise and
  turbulent phase realisations.

This split is what allows reproducible non-atmosphere realisations
under varied weather (introduced in v3.0.0-alpha; PR #22 in
`history.rst`).

---

## 9. Input & output formats

### 9.1 Input formats

| Item | Format | Example file in repo |
|------|--------|-----------------------|
| Driver config | JSON (single flat object, no schema enforcement) | `meqsilhouette/data/eht230.json` |
| Sky model — ASCII | Tigger TDL ASCII (`name ra_h ra_m ra_s dec_d dec_m dec_s i q u v emaj_s emin_s pa_d`) | `meqsilhouette/data/sky_models/singlept.txt` |
| Sky model — Tigger LSM | `*.lsm.html` (Tigger XML) | not bundled |
| Sky model — FITS dir | `txxxx-model.fits` (time), `txxxx-yyyy-model.fits` (time+freq), `txxxx-[IQUV]-model.fits` (pol), `txxxx-yyyy-[IQUV]-model.fits` (full) | `meqsilhouette/data/sky_models/{timevar_point, freqvar_point, timepolvar_point, old_grmhd_pol}` |
| Antenna config | CASA ANTENNA subtable (`table.{dat, f0, info, lock}`) | `meqsilhouette/data/ANTENNA_EHT2017/` |
| Station / weather info | ASCII whitespace-delimited, 20 columns, header row + 1 row per station (see schema below) | `meqsilhouette/data/eht_betterweather.antennas` |
| Bandpass table | ASCII, header row of frequencies in GHz, then `station   (B_R, B_L)   (B_R, B_L) …` | `meqsilhouette/data/eht_bandpass.txt` |

#### Station-info schema (header from `eht_betterweather.antennas`):
```
station  T_rx[K]  pwv[mm]  gpress[mb]  gtemp[K]  c_time[sec]
ptg_rms[arcsec]  PB_FWHM230[arcsec]  PB_model  ap_eff
gR_mean  gR_std  gL_mean  gL_std
dR_mean  dR_std  dL_mean  dL_std
feed_angle[deg]  mount
```
The first column is the station code (e.g. AA, AP, AZ, JC, LM, SM, SP,
PV); columns 1-7, 10, 19 are read as real, columns 11-18 as complex
(`run_meqsilhouette.py` lines 122-125).  `mount` is one of `ALT-AZ`,
`ALT-AZ+NASMYTH-R`, `ALT-AZ+NASMYTH-L`.  The 8 rows of the bundled file
correspond to the EHT 2017 array (ALMA, APEX, SMA→AZ, JCMT, LMT, SMT,
SPT, IRAM-30m).

#### Bandpass file
Single-line header of representative frequencies in GHz, then per-row:
`<station>  (BR1, BL1)  (BR2, BL2) …` parsed with `ast.literal_eval`.
Phases are random per channel (not in the file).

### 9.2 Output products

`docs/source/outputs.rst` and the `np.save` calls scattered through
`SimCoordinator`:

```
<outdirname>/
├── inputs/                              verbatim copies of inputs
│   ├── obs.json
│   ├── <station_info>
│   ├── <bandpass_table>
│   ├── <sky model>
│   └── <antenna_table>
├── plots/                               PNGs (when *_makeplots flags are 1)
│   ├── uv-coverage_legend.png
│   ├── uv-coverage_colorize_min_elevation.png
│   ├── uv-coverage_colorize_mean_elevation.png
│   ├── amp_uvdist.png  num_vis_perbin.png  sensitivity_perbin.png
│   ├── antenna_elevation_vs_time.png  parallactic_angle_vs_time.png
│   ├── pointing_angular_offset_vs_time.png
│   ├── pointing_amp_error_vs_time.png
│   ├── pointing_amp_error_vs_angular_offset.png
│   ├── zenith_transmission_vs_freq.png  transmission_vs_freq_<ANT>.png
│   ├── zenith_skytemp_vs_freq.png
│   ├── input_bandpasses_ampl_{R,L}pol.png
│   └── …
├── atm_output/
│   ├── ATMstring_antN.txt              The exact AATM CLI call per antenna
│   ├── Natm_abs.txt   Natm_disp.txt    Raw AATM stdout
│   ├── delay_norm_antN_timestamp_*.npy
│   ├── delay_norm_timestamp_*.npy
│   ├── phasedelay_alltimes_timestamp_*.npy
│   ├── delay_alltimes_timestamp_*.npy
│   ├── sefd_matrix_timestamp_*.npy
│   ├── sky_noise_timestamp_*.npy
│   └── sky_sigma_estimator_timestamp_*.npy
├── opacity.npy   emissivity.npy   transmission.npy
├── zenith_transmission_timestamp_*.npy   transmission_timestamp_*.npy
├── turbulent_phase_errors_timestamp_*.npy
├── gterms_timestamp_*.npy   bterms_timestamp_*.npy
├── pjones_noparangcorr_timestamp_*.npy
├── djones_noparangcorr_timestamp_*.npy
├── panddjones_parangcorr_timestamp_*.npy
├── dterms_parangcorr_timestamp_*.npy
├── T_rx_timestamp_*.npy   sefd_rx_timestamp_*.npy
├── skytemp_from_emissivity_timestamp_*.npy
├── elevation_tropshape_timestamp_*.npy   dish_area_timestamp_*.npy
├── receiver_noise_timestamp_*.npy
├── receiver_rms_timestamp_*.npy
├── CASAcorrectedStartTime.txt   CASAtimeOffset.txt   (if correctCASAoffset=1)
└── <antenna>_<sky>_RA…DEC…pol…_<nu>GHz-<dnu>MHz-<nchan>chan-<tint>s-<obs>hrs.MS/
        # Measurement Set; see below
```

#### Measurement Set columns (`docs/source/outputs.rst`)

* `DATA` (or whichever is `ms_datacolumn`) — corrupted complex visibilities.
* `MODEL_DATA` — the **uncorrupted** visibilities (preserved verbatim).
* `SIGMA` / `SIGMA_SPECTRUM` — baseline rms σ_pq used for thermal noise.
* `WEIGHT` / `WEIGHT_SPECTRUM` — `1/σ²`.
* `FLAG` — re-written by `write_flag()` to mask elevations below
  `elevation_limit` per baseline.
* `ANTENNA` subtable — copy of `ms_antenna_table` with `MOUNT` column
  written from station_info column 19.
* `FIELD`, `SPECTRAL_WINDOW`, `SOURCE` — populated by CASA simulator;
  patched to use the FITS basename as the source/field/spw NAME (legacy
  `create_ms` only).

UVFITS export (optional) via `im.argo.icasa('exportuvfits', ...)`.

---

## 10. Notable internals

### 10.1 Pyxis-style global namespace

`PYXIS_ROOT_NAMESPACE=True` (top of both driver scripts) instructs Pyxis
to expose `v.OUTDIR`, `v.MS`, `v.PLOTDIR`, `v.LOG` as module-level
names, and `II("$OUTDIR/...")` does shell-style interpolation against
those.  This is why most paths in the source look like
`II('$OUTDIR')+'/turbulent_phase_errors_timestamp_%d'%(self.timestamp)`.

### 10.2 CASA start-time offset workaround

CASA's `simulator.observe()` internally shifts the requested start time
by half an integration plus other small offsets.  For VLBI synthesis,
this is intolerable.  `compute_casa_offset` measures the offset by
running a tiny throwaway MS, then re-runs the real simulation with a
corrected start time.  The corrected time is cached in
`<OUTDIR>/CASAcorrectedStartTime.txt` so subsequent scans of the same
campaign reuse it (`create_ms.py` lines 115-158).

### 10.3 SIMMS legacy path

Two MS-creation routes exist: `create_msv2` (`casatools.simulator`,
default in `run_meqsilhouette.py` line 181) and `create_ms` (shells out
to the standalone `simms` CLI; `simms` is still pinned in
`install_requires`).  The `simms`-based path requires the legacy
`-T VLBA` workaround (line 300-306 comments).

### 10.4 AATM channel-count bug guard

After every `aatm absorption` invocation the number of returned
channels is compared against `self.chan_freq.shape[0]`; on mismatch the
pipeline aborts with a verbose error pointing the user to slightly
different `(nchan, dnu)` (`SimCoordinator.py` lines 565-573).

### 10.5 Elevation flagging vs NaN trick

After `elevation_calc()`, elevations below `elevation_limit` are set to
NaN (line 84) so subsequent trigonometric `sin θ` divisions evaluate to
NaN (and propagate harmlessly through phase exponentials).  A separate
`elevation_copy_dterms` retains the **un-clipped** elevation for use in
Nasmyth D-Jones rotation (where elevation must be physically sensible
even where data is flagged).

### 10.6 Mount-type aware D-Jones

Lines 1138-1146 and 1182-1191 handle three mount conventions:

| `mount` value | P / D rotation extra term |
|---------------|----------------------------|
| `ALT-AZ` | none (just feed_angle + parallactic_angle) |
| `ALT-AZ+NASMYTH-L` | subtract elevation |
| `ALT-AZ+NASMYTH-R` | add elevation |

These are the three mount families present in the EHT 2017 array (e.g.
ALMA = ALT-AZ, APEX = NASMYTH-R, IRAM-30m = NASMYTH-L).

### 10.7 Hardcoded chunk size

`SimCoordinator.chunksize = 100000` rows (line 50). All large-array
operations (noise application, RMS quadrature) iterate in steps of this
size and catch `MemoryError` to abort gracefully.

### 10.8 Hardcoded MeqTrees thread count

`mqt.MULTITHREAD = 32` in `run_turbosim` (`meqtrees_funcs.py` line 46).

---

## 11. Integration & extension points

* **SYMBA pipeline** — MeqSilhouette is the synthetic-data backend of
  the SYMBA pipeline (Roelofs et al. 2020, A&A 636, A5;
  `docs/source/pipelines.rst`).  SYMBA wraps MeqSilhouette and adds
  end-to-end calibration emulation.

* **Adding a new antenna** — `meqsilhouette/utils/add_ant.py` shows the
  pattern: convert lat/lon/h to ITRF XYZ via WGS84, then write a new
  row.  After regenerating the ANTENNA table, append a corresponding
  row to the station_info file with mount type and weather params.

* **Custom driver** — Documented in `docs/source/usage.rst` (lines
  108-112): "Advanced users can construct their own versions of the
  driver script by importing the `framework` module … additional
  operations such as flagging or averaging can be performed by an
  enhanced driver script tailored to the needs of the user."  Example:

  ```python
  from meqsilhouette.framework.SimCoordinator import SimCoordinator
  from meqsilhouette.framework.create_ms     import create_msv2
  from meqsilhouette.framework.process_input_config import (
      load_json_parameters_into_dictionary, setup_keyword_dictionary,
  )
  ```

* **Sky model adapters** — Adding a new on-disk format requires
  branching in `SimCoordinator.interferometric_sim` (lines 184-227) and
  optionally extending the MeqTrees TDL `models = [...]` list in
  `turbo-sim.py` line 79.

* **New corruption term** — Add a method on `SimCoordinator` that
  follows the existing pattern (load metadata in `__init__`, build the
  per-antenna Jones matrix, multiply chunk-wise into `self.data`, save
  numpy provenance file, then `self.save_data()`).  Wire it into the
  driver call chain at the appropriate place in
  `run_meqsilhouette.py` (after the comment block at line 203 fixes the
  ordering: pointing → trop → P/D → G → B → noise).

* **Switching predictor** — `interferometric_sim` chooses between
  WSClean (`run_wsclean`) and MeqTrees (`run_turbosim`) by sky-model
  filename extension.  The MeqTrees path is needed for ASCII / Tigger
  sky models because WSClean only consumes FITS.

---

## 12. Testing layout / examples

There is **no automated test suite** (`pytest`, `unittest`,
`tests/`, `examples/` directories are absent).  Verification relies on:

1. The bundled `meqsilhouette/data/eht230.json` parset, which exercises
   every corruption module end-to-end with `predict_seed=42` and
   `atm_seed=300` (deterministic).
2. Sample sky models (point sources with time/frequency/polarisation
   variability) under `meqsilhouette/data/sky_models/`.
3. Sample CASA ANTENNA table for the EHT 2017 array.
4. Sample station_info and bandpass tables for the same array.
5. Reproducible noise/atmosphere via the dual-RNG seeding scheme.

The Singularity definition includes a minimal `%test` block that only
verifies a Python `import numpy` works (`singularity.def` lines 83-87).

---

## 13. Known limitations / TODOs

Surfaced from comments and `history.rst`:

* `make_image_wsclean` is a stub (`meqtrees_funcs.py` line 58:
  `print('todo')`).  Imaging is currently restricted to lwimager dirty
  images.
* `trop_phase_corrupt` carries the comment *"REPLACE WITH A GENERATE
  TROP SIM COORDINATOR, THAT COLLECTS ALL TROP COORUPTIONS"* (line 725).
* `PB_model` is documented as supporting `'gaussian'` and `'cos3'` but
  the code is **hardcoded to `'gaussian'`** (`SimCoordinator.py` line
  939: `PB_model = ['gaussian']*self.Nant`).
* `ms_scan_lag` and `trop_percentage_calibration_error` are deprecated
  (kept only for backwards compatibility with old SYMBA configs).
* `trop_fixdelay_max_picosec` is deprecated; the actual delays are
  always computed by averaging over the spectral window.
* `numpy==1.21` is pinned in containers because this stack triggers
  `np.BitGenerator` / `np.asscalar` errors with newer NumPy.
* Tigger-LSM precision: comment on `inputs.rst` line 277 warns
  *"MeqTrees has been observed to occasionally give rise to precision
  errors of up to ~1 micro-arcsecond"*; FITS images via WSClean are
  recommended.
* ISM scattering was present in v1 but **removed in v2.0** (`history.rst`
  line 91).
* `legacy create_ms` (simms-based) is preserved next to `create_msv2`;
  only the latter is wired into the driver.

---

## 14. Recent git history (top 20)

```
b43ef9a np.loadtxt allows only for one delimiter
4fa4258 Merge pull request #47 from iniyannatarajan/master
5b2fda1 Update AATM link in docs
c6162d7 Merge pull request #46 from iniyannatarajan/master
7c581eb Revert CMB correction
d78619d Merge pull request #45 from iniyannatarajan/master
e414b03 Remove CMB contribution before computing elevation-corrected T_sys
0671c0a Merge pull request #44 from iniyannatarajan/master
9703dc9 Update Singularity def file
9fca690 Merge pull request #43 from iniyannatarajan/master
c8eb149 Derive skytemp from emissivity and compute elevation-corrected emissivity
c5b9c2a Merge pull request #42 from iniyannatarajan/master
1208738 Serialize some output arrays
c7be7cd Merge pull request #41 from iniyannatarajan/master
796fdb8 Pre-allocate sky sigma estimator
7706e96 Merge pull request #40 from iniyannatarajan/master
ab814a8 Use new noise gen function in driver script
37cbfa0 Update noise gen (WIP)
b3385fc Merge pull request #39 from iniyannatarajan/master
5cdb466 Add init to utils submodule
```

### Tags

`0.7, 2.3, v2.0-alpha, v2.1, v2.2, v2.3, v2.4, v2.5, v2.6, v2.6.1,
v2.6.2, v2.7, v2.7.1, v3.0-alpha.{3,4,5}, v3.0.0-alpha, v3.0.0-alpha.2`

The most recent tag (`v3.0.0-alpha.2`, in alpha) corresponds to the
`__version__ = "3.0"` literal in `meqsilhouette/__init__.py`.

---

## 15. Citations

* Natarajan, I. *et al.* "MeqSilhouette v2: spectral-line and
  polarimetric synthetic data generation for the Event Horizon
  Telescope," *MNRAS* 512, 490 (2022).
  https://ui.adsabs.harvard.edu/abs/2022MNRAS.512..490N
* Blecher, T. *et al.* "MeqSilhouette: a mm-VLBI observation and signal
  corruption simulator," *MNRAS* 464, 143 (2017).
  https://ui.adsabs.harvard.edu/abs/2017MNRAS.464..143B
* Smirnov, O. M. "Revisiting the radio interferometer measurement
  equation," 2011, https://arxiv.org/abs/1101.1764  (RIME formulation
  cited in `docs/source/components.rst`).
* Roelofs, F. *et al.* "SYMBA: An end-to-end VLBI synthetic data
  generation pipeline," 2020, A&A 636, A5.
* Leppanen, K. J., Zensus, J. A., Diamond, P. J. *AJ* 110, 2479 (1995)
  — basis for the `parang_corrected` 2θ rotation in
  `add_pol_leakage_manual`.

---

*Document compiled from a complete walk of `simulators/MeqSilhouette/`
on 2026-05-07.  All claims are sourced from files in that submodule;
no speculative or remembered information was added.*
