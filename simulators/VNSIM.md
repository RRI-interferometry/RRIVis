# VNSIM — VLBI Network SIMulator

> Exhaustive technical reference for the `VNSIM` package, a git submodule of the
> RadioSim project located at `simulators/VNSIM/`. Everything below is derived
> directly from the source tree at HEAD `ae55b25` (branch `main`, no tags).

---

## 1. Overview

**VNSIM** is an integrated VLBI (Very Long Baseline Interferometry) network
simulator written in pure Python 3.6+ with a Tkinter GUI front-end and a
command-line back-end. It is initially motivated by the East Asia VLBI Network
(EAVN) but is documented as expandable to other VLBI arrays and to generic
interferometers (`simulators/VNSIM/README.md`, lines 3–5).

The simulator is described in:

> Zhao, Z. *et al.*, "VLBI Network SIMulator: An Integrated Simulation Tool for
> Radio Astronomers", arXiv:1808.06726v1 (cited in `README.md`, line 11).

VNSIM ships with:

- A self-contained station/source/satellite SQLite database
- A Tk GUI with four plotting panels (UV, OBS, Imaging, Rad-plot) plus
  pop-out parameter calculator and database editor
- Six independent CLI scripts (one per analysis flow)
- ~17 ready-made source brightness models (point/Gaussian/disc + image plates)
- Two-body / J2 satellite orbit propagation for space-VLBI

| Item | Value | Source |
|---|---|---|
| Primary author | Zhen Zhao (Shanghai Astronomical Observatory) | `Func_gui.py:39`, `README.md:17` |
| GUI version string | `1.0` | `Func_gui.py:38` |
| Top-of-tree commit | `ae55b25 Update the source model file` | `git log --oneline -1` |
| Tags | (none) | `git tag` is empty |
| Remote | `https://github.com/ZhenZHAO/VNSIM.git` | `git remote -v` |
| Language | Python 3.6 (declared); SQL via `sqlite3` | `Installation/environment.yaml:24` |
| GUI toolkit | Tkinter / Tk-themed (`ttk`) + matplotlib `TkAgg` | `Func_gui.py:11–22` |
| License | GNU General Public License (35,149 bytes — full GPL text in `LICENSE`) | `LICENSE`, `README.md:19` |
| Source size | 14 Python files, **9,615 lines** total | `wc -l *.py` |

The README explicitly forbids non-research use: *"feel free to use the source
code for you own development purpose, but only for the research"*
(`README.md:19`).

### 1.1 What VNSIM does

The high-level capabilities, summarised from `Func_gui.py:42–48` and the
function-script docstrings:

1. **(u, v) coverage plotting** — single observation, all-sky survey
   (5×6 grid), all-year-round (12 monthly snapshots), multi-source
   simultaneous (`Func_uv.py`, `Func_uv_advanced.py`).
2. **Observation scheduling / station visibility** — Az/El curves,
   sky-survey "how many telescopes can see each pixel", optimal common
   observation interval (`Func_obs.py`).
3. **Imaging** — read APSYNSIM-style ASCII source models, compute dirty
   beam, dirty map, run a Hogbom-style CLEAN, fit an elliptical clean
   beam (`Func_img.py`).
4. **Radial visibility-amplitude plot** of observed FITS data
   (`Func_radplot.py`).
5. **Sensitivity / FoV / FITS-size calculator** for the EVN/VLBA/EAVN
   (`Func_cal.py`, GUI-only).
6. **End-to-end multi-source survey pipeline** that chains UV+OBS+IMG
   for a list of sources with multiprocessing (`Func_survey_all.py`).
7. **Database editor** for adding/removing stations, sources, telemetry
   stations and satellites (`Func_db.py`).

---

## 2. Repository layout

```
simulators/VNSIM/
├── README.md                          # 20-line project description
├── LICENSE                            # GPL (35 KB)
├── .gitignore                         # Python/Conda standard ignore
├── load_conf.py                       # Python "config" with constants + defaults
├── utility.py                         # Time + coord-transform library
├── model_satellite.py                 # Kepler orbit propagation
├── model_effect.py                    # Sun / Moon / Earth ephemerides + shadowing
├── model_obs_ability.py               # Visibility (geometric) judgement
│
├── Func_uv.py                         # CLI 1: basic UV coverage
├── Func_uv_advanced.py                # CLI 2: multi-src / all-sky / all-year UV
├── Func_obs.py                        # CLI 3: Az/El + sky-survey
├── Func_img.py                        # CLI 4: source-model → dirty/clean image
├── Func_radplot.py                    # CLI 5: FITS UVDATA radplot
├── Func_survey_all.py                 # CLI 6: pipelined multi-source survey
├── Func_cal.py                        # GUI: sensitivity / FoV / FITS size
├── Func_db.py                         # GUI: SQLite database editor
├── Func_gui.py                        # The integrated 2,499-line Tk GUI
│
├── CONFIG_FILE/                       # Per-analysis INI configs
│   ├── config_obs.ini                 #   used by Func_obs.py
│   ├── config_uv.ini                  #   used by Func_uv.py / Func_uv_advanced.py
│   ├── config_img.ini                 #   used by Func_img.py
│   └── config_survey.ini              #   used by Func_survey_all.py
│
├── DATABASE/
│   ├── database.db                    # SQLite (43 src, 46 vlbi, 5 sat, 1 telem)
│   ├── database.pkl                   # Pickled cache of database.db (8 dicts)
│   └── playuv.pkl                     # Pickled u, v, max for "play" widget
│
├── SOURCE_MODELS/                     # ASCII brightness-model files
│   ├── *.model                        # 16 model files (P/G/D/IMAGE syntax)
│   └── SRC_PICS/                      # PNG plates referenced by IMAGE models
│
├── OBSERVE_DATA/
│   └── 0106+013_1.fits                # Sample UVFITS used by Func_radplot
│
├── OUTPUT/                            # Output is written here at runtime
│   ├── uv_basic/         uv-notes.md
│   ├── uv_advance/       uv-advance-notes.md
│   ├── obs_ability/      obs-notes.md
│   ├── imaging/          imaging-notes.md
│   ├── data_analyze/     Radplot-NOTE.md
│   └── survey_all/       survey-notes.md
│
└── Installation/
    ├── requirements.txt               # pip pin set (Python 3.6 era)
    ├── environment.yaml               # conda env "enVNSIM"
    └── Installation_guide.html        # rendered installation walk-through
```

There is no `setup.py`, no `pyproject.toml`, no `tests/` directory, and no
`docs/` tree. The "documentation" is the README, the embedded `__about_text__`
strings in `Func_gui.py:42–66`, the per-output `*-notes.md` files inside
`OUTPUT/`, and the project Wiki (deleted from repo on 2018-09 by commit
`22f169a delete the folder of Wiki Asset` — only a remote pointer remains in
`README.md:9`).

---

## 3. Installation & dependencies

`simulators/VNSIM/Installation/Installation_guide.html` provides two flows.

### 3.1 Conda

```
cd <dir_put_VNSIM>
git clone https://github.com/ZhenZHAO/VNSIM.git
cd VNSIM/Installation
conda env create -f environment.yaml
source activate enVNSIM           # creates env named "enVNSIM"
cd ..
python Func_uv.py -h
python Func_uv.py -g
python Func_obs.py -g -s -i -f png
```

### 3.2 Pip

```
cd VNSIM/Installation
pip install -r requirements.txt
cd ..
python Func_uv.py -g
```

### 3.3 Pinned dependencies

From `Installation/requirements.txt`:

```
astropy==2.0.2
certifi==2016.2.28
cycler==0.10.0
matplotlib==2.0.2
numpy==1.13.1
py==1.4.34
pyparsing==2.2.0
pytest==3.2.1
python-dateutil==2.6.1
pytz==2017.2
scipy==0.19.1
six==1.10.0
```

`Installation/environment.yaml` adds `pyqt=5.6.0` and `qt=5.6.2` as conda
packages (env name `enVNSIM`, python `3.6.2`). Despite this, **VNSIM does not
import PyQt anywhere** — every GUI import is via `tkinter` and `tkinter.ttk`
(see `Func_gui.py:21,23`, `Func_cal.py:7,8`, `Func_db.py:6,7`); the PyQt
inclusion appears to be dead.

The HTML guide also lists the standard-library modules used at runtime: `os`,
`sys`, `pickle`, `argparse`, `configparser`, `numpy`, `scipy`, `matplotlib`,
`astropy`, `time`, `tkinter`, `sqlite3`, `multiprocessing`, `threading`,
`queue`, `logging`, `webbrowser`.

A documented trouble-shoot: on macOS, *"RuntimeError: Python is not installed
as a framework"* is fixed by the (already applied) `mpl.use("TkAgg")` line at
the top of every `Func_*.py` script (`Func_uv.py:7–8`, etc., enforced by
commit `e2c2bf4 specify the backend of matplotlib`).

---

## 4. Build & runtime architecture

VNSIM has no build step — it runs directly from the source tree. Two execution
modes:

### 4.1 GUI mode (the recommended entry point)

```
python Func_gui.py
```

This launches an `AppGUI` window (`Func_gui.py:734`) that ties together every
other module. It expects to find `DATABASE/database.db` and
`DATABASE/database.pkl` relative to the CWD (`Func_gui.py:741`,
`Func_gui.py:2495`).

### 4.2 CLI mode

Each `Func_*.py` script is independently executable and parses its own
arguments via `argparse`. The script reads its corresponding
`CONFIG_FILE/config_*.ini`, retrieves station/source/satellite data from
`DATABASE/database.pkl`, runs the calculation, optionally pops up a Tk window,
and saves figures into `OUTPUT/<subdir>/`.

Common CLI flags across scripts:

| Flag | Meaning |
|---|---|
| `-c, --config FILE` | INI config to load (default per script) |
| `-g, --show_gui` | Open a matplotlib window in addition to saving |
| `-s, --save_data` (or `--save_az_el`, `--save_uv`) | Dump raw data as `.txt` |
| `-i, --obs_info` (or `--show_para`) | Print extra textual info on stdout |
| `-f, --img_fmt {eps,png,pdf,svg,ps}` | Output figure format |
| `-n, --num_subprocess N` | (Func_uv_advanced, Func_survey_all) parallelism |

### 4.3 Module dependency graph

```
                   load_conf.py  ──┐
                       │           │
       utility.py ─────┘           │  (constants & globals)
            │                      │
    model_satellite.py             │
            │                      │
    model_effect.py                │
            │                      │
   model_obs_ability.py ───────────┤
            │                      │
   ┌────────┼─────────┐            │
   ▼        ▼         ▼            ▼
Func_obs  Func_uv  Func_radplot  Func_db
            │
            ▼
       Func_uv_advanced
            │
            ▼
    Func_img ─── Func_survey_all
            │
            ▼
       Func_gui ─── Func_cal
```

`Func_gui` is the orchestrator and consumes essentially every other module
(see `Func_gui.py:7–35`):

```python
from Func_cal import *
from Func_db import DbEditor
from Func_uv import FuncUv
from Func_img import FuncImg
from Func_obs import FuncObs
from Func_radplot import FuncRadPlot
import load_conf as lc
import model_effect as me
import utility as ut
```

---

## 5. The Tkinter GUI architecture

`Func_gui.py` is a single 2,499-line file containing eight classes
(`grep -n "^class\|^def " Func_gui.py`):

| Lines | Class / function | Role |
|---|---|---|
| 117 | `process_finish_call_back` | logs completion of a sub-task |
| 124 | `AppData` | non-GUI data layer; owns instances of every `Func*` class and exposes `update_all_with_flag()` / `get_data_*()` getters used by the panels |
| 734 | `AppGUI` | the main 1,000×700 px window; builds menus, the four-tab Notebook, and all panels |
| 2200 | `GressBar` | indeterminate `ttk.Progressbar` shown in a topmost Toplevel during long jobs |
| 2236 | `ImagePopWin(tk.Toplevel)` | pop-up showing `All Year UV` or `Multi Source UV` mosaic |
| 2324 | `MultiChoiceDialog(tk.Toplevel)` | listbox dialog used to multi-select sources/stations from the DB |
| 2385 | `TextHandler(logging.Handler)` | redirects `logging` records to a Tk `ScrolledText` |
| 2409 | `TopLevelParaCal` | wraps `Func_cal.ParaCal` in a Toplevel |
| 2426 | `TopLevelDbEditor` | wraps `Func_db.DbEditor` in a Toplevel; refreshes the main UI when closed |
| 2448 | `center_window`, `limit_window_size` | geometry helpers |

Layout (`Func_gui.py:1554–1693`):

- **Menubar** (`tk.Menu`) with three top-level cascades: `VNSIM` (About,
  Feedback, Quit), `Tool` (radio-buttons for the four panels +
  checkbuttons for the two pop-out tools), `View` (Image+Config /
  Image-Only) and `Help` (User Tips, Project Github via `webbrowser.open`).
- **Notebook** (`ttk.Notebook`) with four tabs:
  1. `tab_uv` — UV Funcs (single UV plot, "All Year UV" button, "Multi
     Source UV" button)
  2. `tab_obs` — OBS Funcs (Az/El curves and sky-survey)
  3. `tab_img` — Imaging (model image, dirty beam, dirty map, clean map,
     residual)
  4. `tab_radplot` — Rad Plot (UV scatter and visibility-amplitude vs.
     baseline)
- **Left configuration pane** with widgets bound to `tk.IntVar` /
  `tk.StringVar` for every parameter in `load_conf.py`.
- **Bottom status line** showing `Ready` / `Running` / `Done` and a
  `ScrolledText` log area driven by the `TextHandler`.

The "Tool" menu can additionally invoke:

- `TopLevelParaCal` → opens `ParaCal` (the EVN-Calc-style sensitivity tool
  from `Func_cal.py`)
- `TopLevelDbEditor` → opens `DbEditor` (`Func_db.py`); on close it calls
  `myDbEditor.update_db_pkl()` and `parent.refresh_ui_config_with_db()` so
  newly inserted stations propagate immediately.

Concurrency: long-running calculations run on a `threading.Thread` while the
`GressBar` Toplevel spins; results are signalled back via a `queue.Queue`
(`Func_gui.py:736, 869–877`). `apply_all_with_multiprocess` is the
multiprocessing variant.

---

## 6. File-by-file reference

### 6.1 `load_conf.py` — global constants & default scenario

`simulators/VNSIM/load_conf.py` (205 lines) holds the canonical numeric
constants and the *default* scenario used when no INI is supplied.

Constants (`load_conf.py:9–18`):

| Symbol | Value | Units | Meaning |
|---|---|---|---|
| `light_speed` | `299_792_458.8` | m/s | c |
| `earth_radius` | `6378.1363` | km | equatorial |
| `earth_flattening` | `1/298.257` | – | f |
| `eccentricity_square` | `f·(2-f)` | – | e² for the Earth ellipsoid |
| `GM` | `3.986_004_418e5` | km³/s² | Earth gravitational parameter |

Default observation block (`load_conf.py:24–66`):

```python
StartTimeGlobalYear = 2019; StartTimeGlobalMonth = 1; StartTimeGlobalDay = 20
StopTimeGlobal*  = 2019/1/20  10:00:00
TimeStepGlobal*  = 0d 0h 5m 0s
baseline_flag_gg, _gs, _ss = 1, 0, 0
baseline_type   = gg + gs*2 + ss*4   # bit-mask: 1=g-g, 2=g-s, 4=s-s
obs_freq        = 22e9    # 22 GHz
bandwidth       = 3.2e7   # 32 MHz
unit_flag       = 'km'    # else 'lambda'
cutoff_mode     = {'flag': 1, 'CutAngle': 10}
                # flag 0=DB elevation, 1=GUI value, 2=max, 3=min
precession_mode = 0       # 0=Two-Body, 1=J2
```

Default catalogues (also in `load_conf.py:73–123`):

- `pos_mat_src` — list of `[name, ra_str, dec_str]` (defaults: `0316+413`,
  `0202+319`)
- `pos_mat_vlbi` — list of `[name, x_km, y_km, z_km, el_deg, type]`
  (5 EAVN stations: ShangHai, Tianma, Urumqi, GIFU11, HITACHI)
- `pos_mat_telemetry` — initially `[]`; example: `Goldstone`
- `pos_mat_sat` — initially `[]`; example: `VSOP`, `RadioAstron` with
  Kepler elements `[name, a, e, i, ω, Ω, M0, Epoch_MJD]`

Imaging block (`load_conf.py:128–134`): `n_pix=512`, `source_model =
'Point-source.model'`, `clean_gain=0.9`, `clean_threshold=0`,
`clean_niter=20`, `color_map_name='viridis'`.

Rad-plot file: `0106+013_1.fits` (`load_conf.py:139`).

Section 6 (`load_conf.py:144–148`) loads `DATABASE/playuv.pkl` to expose
`just4fun_u`, `just4fun_v`, `just4fun_max` (used by an Easter-egg "play UV"
button in the GUI).

`print_setting()` (`load_conf.py:161–201`) is invoked when the file is run as
`__main__` and dumps a human-readable summary.

### 6.2 `utility.py` — time, coordinate and unit transforms

`simulators/VNSIM/utility.py` (596 lines). Three groups:

**(1) Time transforms** — `time_2_jde`, `time_2_mjd`, `mjd_2_time`,
`time_2_day`, `time_2_rad`, `time_str_2_rad`, `time_str_2_mjd`, `mjd_2_julian`,
`mjd_2_gmst` (Greenwich Mean Sidereal Time; `utility.py:153–162`),
`mjd_2_gast` (Greenwich Apparent Sidereal Time), `mjd_2_gst` (a higher-order
GST formulation), `ecliptic_obliquity`, `nutation_omega`, `longitude_nutation`,
`equinox_equation`. The GMST formula used:

```
gmst = 67310.548 + 8640184.812866 · T_julian
       + (mjd + 0.5 - int(mjd + 0.5)) · 86400        # rad after × π/43200
```

**(2) Coordinate transforms** — `trans_matrix_uv_itrf` (the **3×3 ITRF→(u,v,w)
rotation** used everywhere in `Func_uv`, see `utility.py:240–255`),
`geographic_2_itrf` / `itrf_2_geographic`, `rect_2_polar` / `polar_2_rect`,
`equatorial_2_horizontal` (Az/El of source from station given MJD),
`drotate` (axis rotation), `itrf_2_horizontal` (site Az/El of a satellite
including velocity), `equatorial_2_ecliptic`, `itrf_2_icrf` and `icrf_2_itrf`
(GAST-based conversions for satellite state vectors, used for u,v on
satellite baselines).

The ITRF→UV matrix:

```python
matrix = [[ sin(H),                  cos(H),                 0     ],
          [-sin(δ)·cos(H),           sin(δ)·sin(H),          cos(δ)],
          [ cos(δ)·cos(H),          -cos(δ)·sin(H),          sin(δ)]]
where H = GAST(mjd) - RA, mod 2π
```

**(3) Unit transforms** — `freq_2_wavelength`, `angle_str_2_rad` (e.g.
`"23d43m54s"` → rad), `angle_2_rad`, `rad_2_angle`, `sgn`, `angle_btw_vec`
(angle between two unit 3-vectors).

### 6.3 `model_satellite.py` — Kepler orbit propagation

`simulators/VNSIM/model_satellite.py` (153 lines). Implements the orbit
mechanics needed for space-VLBI:

- `semi_axis_cal(apogee, perigee)` → `(a, e)` (`model_satellite.py:10–19`)
- `apo_per_cal(a, e)` → `(apogee, perigee)`
- `satellite_orbit_period(a)` — Kepler's third law, returns period in days
- `kepler_2_cartesian(a, e, i, ω, Ω, M)` (`model_satellite.py:46–123`) —
  solves `M = E − e·sin E` by Newton iteration to 1e-7, then computes
  position `(x,y,z)` in km and velocity `(Vx,Vy,Vz)` in km/s in the ICRF
  frame. The hard-coded GM here is `3.986004418e14` m³/s².
- `get_satellite_position(a, e, i, ω, Ω, M0, MJDEpoch, MJDTime,
  precession_mode)` (`model_satellite.py:126–153`) — supports two
  precession modes:
  - `flag=0` Two-Body (`dω/dt = dΩ/dt = dM0/dt = 0`)
  - `flag=1` J2 with `J2 = 0.001082629832258`, equatorial radius
    `r0 = 6378.1363 km`. Standard secular rates are applied:

    ```
    dω/dt   =  0.75·J2·n·(r0/a)²·(5cos²i − 1)/(1−e²)²
    dΩ/dt   = −1.5 ·J2·n·(r0/a)²·cos i / (1−e²)²
    dM0/dt  =  0.75·J2·n·(r0/a)²·(3cos²i − 1)/(1−e²)^{3/2}
    ```

The function returns `[a, e, i, ω, Ω, M, x, y, z, Vx, Vy, Vz]` in **ICRF**.
Conversion to ITRF (needed for u,v) is done by `ut.icrf_2_itrf` in
`model_obs_ability.py`.

### 6.4 `model_effect.py` — Sun, Moon, Earth ephemerides

`simulators/VNSIM/model_effect.py` (358 lines). Implements low-precision
analytic ephemerides from *Astronomical Algorithms* (J. Meeus):

- `sun_ecliptic_pos(jd)` — apparent solar ecliptic longitude (rad), Meeus
  Chap. 25 (lines 11–34).
- `moon_ecliptic_pos(jd)` — Moon's apparent ecliptic longitude, latitude
  and geocentric distance (km). Uses the 60-term *La/Lb/Lc/Ld + Sl/Sr*
  series and the parallel 60-term *Ba/Bb/Bc/Bd + Sb* series,
  hard-coded as Python lists (lines 61–139).
- `moon_ra_dec_cal(start_mjd, stop_mjd, dt)` and `sun_ra_dec_cal(...)` —
  iterate over time, returning RA(hours)/Dec(deg) lists (used by
  `Func_obs.py` to overplot Sun and Moon on the sky-survey image).
- `earth_ecliptic_pos(pos_vec_sat, ε)` — direction from a satellite
  back to Earth in the ecliptic frame.
- `sun_effect_src(time_mjd, ra, dec)` — returns False if the source is
  within 50° of the Sun.
- `earth_shadow_sun(time_mjd, pos_vec_sat, amos_flag)` — geometric Earth
  occultation of the Sun seen from a satellite, optionally inflating the
  Earth's apparent radius by 5° to mimic refraction effects.
- `earth_shadow_src(time_mjd, pos_vec_sat, ra, dec, amos_flag)` — same
  but for the science source.

A constant `sar = 0.00465421` rad ≈ 0.267° is used as the Sun's apparent
angular radius.

### 6.5 `model_obs_ability.py` — visibility judgement

`simulators/VNSIM/model_obs_ability.py` (372 lines). Decides which stations
and satellites can observe a source at a given MJD, given a `cutoff_dict`
(`{'flag': 0|1|2|3, 'CutAngle': deg}`), `baseline_type` bitmask, and
`precession_mode`.

Top-level functions:

- `obs_all_active_sta(time_mjd, ra, dec, pos_mat_sat, pos_mat_telemetry,
  pos_mat_vlbi, baseline_type, cutoff_dict, precession_mode)` — main
  entry, returns `(vlbi_visibility, sat_visibility, sat_itrf_lst)` where
  `vlbi_visibility` is `[name, bool, name, bool, ...]` (lines 15–93).
- `obs_all_active_vlbi(...)` — ground-only path, used by `Func_uv._func_uv_gg`.
- `obs_all_active_sat(...)` — satellite-only path.
- `obs_judge_active_vlbi_station(ra_src, dec_src, time_mjd, lon, lat,
  horizon_array)` — converts (RA, Dec) → (Az, El) using
  `ut.equatorial_2_horizontal`, linearly interpolates the per-azimuth
  cutoff array `horizon_array[0..359]`, returns True iff El > min_el
  for that azimuth (lines 126–158).
- `obs_judge_active_satellite(time_mjd, pos_vec_sat, pos_mat_telemetry,
  ra_src, dec_src, cutoff_dict)` — first checks that **at least one
  telemetry/tracking station** can see the satellite, then that the
  satellite can see the source.
- `obs_satellite_to_source` — solar-elongation > 70° AND lunar
  elongation > 20° constraints (lines 273–334).
- `obs_telemetry_to_satellite` — local elevation cutoff at the
  telemetry/uplink station.

The cutoff modes (interpreted in `Func_obs.py`/`Func_uv.py` and the GUI):

| `cutoff_dict['flag']` | Behaviour |
|---|---|
| 0 | Use elevation stored per-station in DB column `vlbi_el` |
| 1 | Use the single `CutAngle` from the GUI/INI |
| 2 | Use `max(db_el, CutAngle)` |
| 3 | Use `min(db_el, CutAngle)` |

### 6.6 `Func_obs.py` — observation scheduling

`simulators/VNSIM/Func_obs.py` (633 lines). Two classes:

- `FuncObs` (lines 24–289). Methods include:
  - `_func_tv_az_el()` — Az/El curves for every selected ground station
    over the time window with step `time_step` (returns `(azimuth,
    elevation, hour, gs_names)`).
  - `_func_sky_survey()` — for each pixel of a `0.25h×2.5°` RA-Dec grid
    (`[0,24] × [-88.75,90]`, total 96×72), counts how many VLBI stations
    can see that pixel at the start time. Sun and Moon coordinates are
    overlaid.
  - `_func_best_obs_time_el()` — sign-change detection on
    `cut_line − el_line` to find continuous Az/El intervals above the
    cutoff; the longest interval per station gives `best_inter`, and the
    intersection across all stations gives the `optimal_inter`.
  - `_func_best_time_string()` — formats the optimal interval as a
    UTC string.
- `ObsConfigParser` (lines 291–478). Reads `CONFIG_FILE/config_obs.ini`
  with sections `[obs_time]`, `[bs_type]`, `[obs_mode]`, `[station]`;
  if missing, calls `rewrite_config()` which writes a default file.
  After parsing, `get_data_from_db()` opens
  `DATABASE/database.pkl` and pulls only the records whose names match
  the comma-separated INI lists. **The pickle file is loaded with
  exactly nine `pickle.load(fr)` calls** in this order (also enforced
  by `Func_db.write_to_pickle` at lines 814–832):

  ```python
  src_dict, sat_dict, telem_dict,
  vlbi_vlba_dict, vlbi_evn_dict, vlbi_eavn_dict,
  vlbi_lba_dict, vlbi_other_dict, vlbi_all_dict
  ```

`run_obs()` (lines 508–629) is the CLI entry; saves an Az/El plot and a
sky-survey heatmap into `OUTPUT/obs_ability/`.

### 6.7 `Func_uv.py` — basic (u, v) coverage

`simulators/VNSIM/Func_uv.py` (787 lines). Two classes:

- `FuncUv` (lines 24–460). The result API:
  - `get_result_single_uv_with_update()` → `(u_lst, v_lst, max_range)`
  - `get_result_year_uv_with_update()` — 12 monthly snapshots
  - `get_result_sky_uv_with_update()` — 5×6 = 30 sky cells (RA loop
    `2,6,10,14,18,22 h`, Dec loop `-60,-30,0,30,60°`)
  - `get_result_multi_src_with_update()` — every entry of `pos_multi_src`

  The core math (`_func_uv_gg`, lines 346–378):

  ```python
  for timestamp in arange(start_mjd, stop_mjd, time_step):
      active = mo.obs_all_active_vlbi(timestamp, ra, dec, vlbi, cutoff)
      uv_matrix = ut.trans_matrix_uv_itrf(timestamp, ra, dec)   # 3x3
      for i, j in pairs:
          if active[i] and active[j]:
              d_xyz = pos[i] - pos[j]                  # km
              u, v, w = uv_matrix @ d_xyz              # km
              if unit == 'lambda':
                  u, v, w = u·1000/λ, v·1000/λ, w·1000/λ
              lst_u.extend([u, -u]); lst_v.extend([v, -v])
  ```

  `_func_uv_gs` and `_func_uv_ss` are **stubs**: they return `(None,
  None, None, None, None)` (lines 380–398). In other words, **VNSIM's
  current public source produces `gg` baselines only**, even though the
  scaffolding for ground-space and space-space is present.

  `_calculate_beam_size()` (lines 401–451) implements the Tim-Pearson
  natural-weight estimator; the dirty-beam minor-axis is approximated as
  `1.1 / max_uv [rad]` (converted to mas) and the position angle is
  `−0.5·atan2(2·Σuv, Σu²−Σv²)`.
  `get_max_uv()` (lines 454–460) returns the largest |u|,|v|.

- `UVConfigParser` (lines 463–656). Parallel to `ObsConfigParser`, plus
  the extra `[obs_mode]` keys `obs_freq`, `bandwidth`, `unit_flag`.

`run_uv_basic()` (lines 698–784) is the CLI: writes `uv-plot:*.{pdf,png,…}`
into `OUTPUT/uv_basic/` and optionally `u-v:*.txt` raw data.

### 6.8 `Func_uv_advanced.py` — multi-source / all-sky / all-year UV

`simulators/VNSIM/Func_uv_advanced.py` (503 lines). Adds parallelism on top
of `FuncUv`. Bitmask:

```python
RUN_FUNC_UV_SRCS = 1   # multi-source mosaic
RUN_FUNC_UV_SKY  = 2   # all-sky 5x6 mosaic
RUN_FUNC_UV_YEAR = 4   # 12-month evolution mosaic
```

`FuncUvMore.run_uv_more()` (lines 115–...) creates a
`multiprocessing.Pool` and a `Manager().Queue`; each worker is the
`FuncUvMore.__call__` method dispatching on the `SUB_PROCESS_TYPE_*`
constants and pushing a result dict
`{"type":..., "name":..., "u":..., "v":..., "maxuv":...}` onto the queue.

The CLI (`run_uv_advanced`, lines 242–...) accepts `-t {1,2,4,3,5,6,7}`
to combine the three sub-functions, and `-n N` for the worker count.
`run_uv_advanced_single_process` is provided for benchmarking.

### 6.9 `Func_img.py` — source models, dirty/CLEAN imaging

`simulators/VNSIM/Func_img.py` (1,095 lines). Class `FuncImg` (lines 27–...).

**Model file syntax** (parsed in `_read_model`, lines 100–153):

```
# comments allowed
IMSIZE  <half-size_in_arcsec>      # optional override
P  <ra_off>  <dec_off>  <flux_Jy>                   # point
G  <ra_off>  <dec_off>  <flux>     <fwhm_arcsec>    # gaussian
D  <ra_off>  <dec_off>  <flux>     <radius_arcsec>  # uniform disc
IMAGE  <png_path>  <peak_flux>                      # image plate
```

(15 of the 17 shipped `.model` files use these primitives; e.g.
`Five-Gauss.model` mixes 50+ point sources, `Discs.model` uses 4 discs,
`Faceon-Galaxy.model` typically uses an `IMAGE` line into
`SOURCE_MODELS/SRC_PICS/M100.png`.)

**Pipeline** (`_prepare_model` → `_prepare_beam` → `_prepare_map` →
`do_clean`):

1. **Model image**: each P/G/D primitive is rasterized at the centroid;
   IMAGE plates are read by `matplotlib.image.imread`, averaged across
   channels, zoomed to `n_pix/4` via `scipy.ndimage.zoom`, and embedded
   into the central quarter. The model FFT is `np.fft.fft2(fftshift(...))`.

2. **Dirty beam**: each (u, v) sample is binned onto an `n_pix × n_pix`
   grid using `scale_uv = n_pix / (2·max_u·0.95·0.5)`, optionally
   robust-weighted (commented out), and the beam is
   `Re{ifftshift(ifft2(fftshift(mask)))}`, peak-normalised.

3. **Dirty map**: `np.fft.fftshift(np.fft.ifft2(model_fft *
   ifftshift(mask))).real / (beam_scale · 1.5)`.

4. **CLEAN** (`do_clean`, lines 365–401): Hogbom — find peak in residual,
   subtract `gain·peak·shifted_dirty_beam`, accumulate clean components,
   stop when `max(|res|) < clean_thresh` or `clean_niter` iterations
   reached.

5. **Clean beam fit** (`get_clean_beam`, lines 403–435): least-squares
   Gaussian fit to the main lobe (`beam > 0.6·max(beam)`):

   ```
   exp(-(dX² a + dY² b + dX·dY c))  →  fit (a, b, c)
   ```

   yielding the clean-beam major/minor axes and PA.

6. **Imaging metrics** (`update_result_para_cal`, etc.) compute
   `e_bpa`, `e_bmaj`, `e_bmin`, dynamic range, and rms noise.

`run_img()` (lines 836–1093) is the CLI: writes `img-*:*.{pdf,…}` figures
into `OUTPUT/imaging/`.

`overlap_indices(map, beam, mx, my)` (top-level, used both by
`Func_img.do_clean` and `Func_survey_all`) computes the overlap region
between the dirty beam shifted to `(mx, my)` and the dirty map.

### 6.10 `Func_radplot.py` — UVFITS radplot

`simulators/VNSIM/Func_radplot.py` (213 lines). Reads `OBSERVE_DATA/<file>`
via `astropy.io.fits as pf`. Expected columns:

```
['UU---SIN', 'VV---SIN', 'WW---SIN', 'DATE', '_DATE',
 'BASELINE', 'INTTIM', 'GATEID', 'CORR-ID', 'DATA']
```

`u, v` are extracted as `data['UU---SIN'] / PSCAL2 / 1e6` (so the
header keyword `PSCAL2` rescales the SIN-projected uvw); the visibility
is `DATA[:,0,0,0,0,0,0] + 1j·DATA[:,0,0,0,0,0,1]`. The CLI dumps
`uv-plot:*.pdf` and `rad-plot:*.pdf` into `OUTPUT/data_analyze/`.

### 6.11 `Func_cal.py` — sensitivity / FoV / FITS-size calculator

`simulators/VNSIM/Func_cal.py` (656 lines). Class `ParaCal` (lines 16–...)
is a self-contained Tk dialog that supersedes the EVN-Calc web tool. It
hard-codes:

- 65 stations sorted alphabetically (`teleLst`, lines 21–28): VLBA
  (`Hn,Fd,La,Kp,Pt,Ov,Br,Nl,Mk,Sc`), EAVN (`Tm65,My,Mh,Sh,Ku,Ky,Kt,Sv,Zc,
  Bd,VERAIR,VERAIS,VERAMZ,VERAOG,NRO45`), EVN (`Wb,Cm,Mc,Nt,Ef,On,Tr,
  Ys,Hh,Jb1,Jb2,Ur`), telescope arrays (`Y1,Y27,SKA1-mid,SKA1-low,FAST,
  ALMA,Pa,Mp`), space (`SAT1,SAT2`), and others (`Ar,Gb,Go,Pv,Pb,At,Sr,
  Cd,Ap,Ho,Ka,Ny,W1,Wz,Ro34,Ro70,Ir`).
- 14 observing bands (P92 to W3mm) with hard-coded SEFD dictionaries
  `_band_92cm`, `_band_49cm`, ..., `_band_3mm` (lines 497–...). A value
  of `-1` means "no receiver / SEFD unknown".
- Data-rate options `2048,1024,512,256,128,64,32,16,8` Mbps.
- Channel/integration/polarisation/subband selectors.
- Five baseline-length presets (e.g. `12000 km (EVN+VLBA)`).

Calculations (lines 318–402):

| Quantity | Formula |
|---|---|
| Mean SEFD | `sqrt(Σ(t_i·t_j)^{1-m}/2) / (Σ(t_i·t_j)^{-m/2}/2)` (m=2) |
| Thermal noise (mJy) | `1000·1.43·mean_SEFD / sqrt(dRate·1e6/2 · t_obs)` |
| Bandwidth smearing | `49500·N_chan / (B_max · BW_subband)`  [arcsec] |
| Time smearing | `18560·λ / (B_max · t_int)` [arcsec] |
| Correlator FITS size | `1.75 · (N²·N_par·N_cross·N_sb·N_chan / 131072) · (T_obs/3600) / t_int`  [GB] |

Errors and warnings (e.g. "no receivers in this band" or "subbands ×
polarisations exceeds 16") are surfaced via `messagebox.showinfo`.

### 6.12 `Func_db.py` — SQLite database editor

`simulators/VNSIM/Func_db.py` (936 lines). Two classes:

- `DbEditor` (lines 30–...) — Tk Notebook with one tab per table:
  Source / Satellite / Telemetry / VLBI. Each tab has an Entry-row form,
  `Import / Delete / Reset / Insert` buttons, and a `ttk.Treeview`
  showing the current contents.
- `DbModel` (line ~530...): wraps `sqlite3.connect` and offers
  `insert_*`, `delete_*`, `read_*_all`, plus the canonical
  `write_to_pickle()` (lines 814–832) that re-flattens the SQLite
  contents into nine ordered dicts and dumps them to
  `DATABASE/database.pkl`.

CLI (`Func_db.py:904–936`):

```
python Func_db.py -g            # GUI editor
python Func_db.py -p            # update database.pkl from database.db
python Func_db.py -i            # show design rationale
```

### 6.13 `Func_survey_all.py` — full pipeline driver

`simulators/VNSIM/Func_survey_all.py` (609 lines). Class `FuncSurvey`
(lines 38–...). For each source in the multi-source list it instantiates
`FuncUv`, `FuncObs`, `FuncImg` (re-using `overlap_indices` imported from
`Func_img`) and produces a single combined panel:

- (u, v) plot, dirty beam, dirty map, clean image, residual
- per-source Az/El curves and best-time interval
- a parameter table (`bpa`, `bmaj`, `bmin`, dynamic range, rms)

Long jobs are dispatched onto a `multiprocessing.Pool`; the CLI flag
`-n` controls worker count. Outputs land in `OUTPUT/survey_all/<source>/`.
Note `mpl.use('Agg')` (line 9) — `Func_survey_all` is non-interactive
by default.

### 6.14 `Func_gui.py` — see §5

---

## 7. Database schema

`simulators/VNSIM/DATABASE/database.db` is a SQLite3 file with four tables
(verified by `sqlite3 database.db .schema`):

```sql
CREATE TABLE table_src (
  src_name TEXT NOT NULL UNIQUE,
  src_ra   TEXT NOT NULL,            -- e.g. "3h19m48.160s"
  src_dec  TEXT NOT NULL,            -- e.g. "41d30m42.10s"
  PRIMARY KEY(src_name)
);

CREATE TABLE table_vlbi (
  vlbi_name TEXT NOT NULL UNIQUE,
  vlbi_x    REAL NOT NULL,           -- km, ITRF
  vlbi_y    REAL NOT NULL,
  vlbi_z    REAL NOT NULL,
  vlbi_el   REAL NOT NULL,           -- per-station elevation cutoff (deg)
  vlbi_type INTEGER NOT NULL,        -- 0=VLBA, 1=EVN, 2=EAVN, 3=LBA, 4=Other
  PRIMARY KEY(vlbi_name)
);

CREATE TABLE table_telem (
  telem_name TEXT NOT NULL UNIQUE,
  telem_x    REAL NOT NULL,
  telem_y    REAL NOT NULL,
  telem_z    REAL NOT NULL,
  telem_el   REAL NOT NULL,
  PRIMARY KEY(telem_name)
);

CREATE TABLE table_sat (
  sat_name TEXT NOT NULL UNIQUE,
  sat_apo  REAL NOT NULL,            -- apogee (km above r0)
  sat_peri REAL NOT NULL,            -- perigee
  sat_incl REAL NOT NULL,            -- inclination (deg)
  sat_o1   REAL NOT NULL,            -- argument of perigee ω (deg)
  sat_o2   REAL NOT NULL,            -- longitude of ascending node Ω (deg)
  sat_m0   REAL NOT NULL,            -- mean anomaly at epoch (deg)
  sat_t    TEXT NOT NULL,            -- epoch as "YYYYMMDDHHMMSS"
  PRIMARY KEY(sat_name)
);
```

Current contents (verified via `sqlite3 .. SELECT count(*) ...`):

| Table | Rows | Examples |
|---|---|---|
| `table_src` | **43** | `0316+413`, `0202+319`, `0529+483`, `1030+415`, `0000-199`, `0237-233`, ... |
| `table_vlbi` | **46** | type 0: `VLBABR..VLBASC` (10) + `VLBAGB`. type 1: `EVNCm..EVNYb` (16). type 2 (EAVN): `ShangHai, Tianma, Urumqi, GIFU11, HITACHI, KASHIM34, TAKAHAGI32, VERAIR/IS/MZ/OG, NOBEYA45, SEJONG, KVNTN/US/YS, Kunming, SRT`. type 4: `ARECIBO`. |
| `table_sat` | **5** | `SAT1`, `SAT2`, `SRT1`, `VSOP`, `RadioAstron` |
| `table_telem` | **1** | (single tracking station) |

The pickled mirror (`DATABASE/database.pkl`) consists of nine
`pickle.dump`-ed dictionaries in the order documented in §6.6 above.

`DATABASE/playuv.pkl` is a small fun-asset (loaded by `load_conf.py:144`)
with three pickled objects: `just4fun_u`, `just4fun_v`, `just4fun_max`.

---

## 8. Input formats

### 8.1 INI configs

All four `CONFIG_FILE/config_*.ini` files share the same skeleton; the
imaging configs add an `[imaging]` section. Concrete example
(`CONFIG_FILE/config_uv.ini`, full file):

```ini
[obs_time]
start = 2020/01/01/00/00/00
end   = 2020/01/02/00/00/00
step  = 00/00/05/00              ; d/h/m/s

[bs_type]
bs_flag_gg = 1
bs_flag_gs = 0
bs_flag_ss = 0

[obs_mode]
obs_freq        = 1.63e9
bandwidth       = 3.2e7
cutoff_angle    = 10.0
precession_mode = 0
unit_flag       = km             ; 'km' or 'lambda'

[station]
pos_source    = 0316+413, 0202+319
pos_vlbi      = ShangHai, Tianma, Urumqi, GIFU11, HITACHI, KASHIM34
pos_telemetry =
pos_satellite =
```

The names in `pos_*` must match the `name` column in the corresponding
SQLite table.

`config_img.ini` and `config_survey.ini` add:

```ini
[imaging]
n_pix          = 512
source_model   = Point-source.model
clean_gain     = 0.9
clean_threshold= 0.01
clean_niter    = 20
color_map_name = hot
```

### 8.2 Source model (`SOURCE_MODELS/*.model`)

Whitespace-delimited ASCII; `#` introduces a comment. Tokens at line
start drive the parser (`Func_img.py:121–139`):

| Token | Meaning | Args |
|---|---|---|
| `IMSIZE` | half-side of image in arcsec | `<size>` |
| `P` | point source | `<ra_off> <dec_off> <flux_Jy>` |
| `G` | Gaussian source | `<ra_off> <dec_off> <flux> <fwhm_arcsec>` |
| `D` | uniform disc | `<ra_off> <dec_off> <flux> <radius_arcsec>` |
| `IMAGE` | image plate | `<png_relative_path> <peak_flux_per_pixel>` |

Shipped models:

| File | Type |
|---|---|
| `Point-source.model`, `Point-source2.model`, `point.model` | single P |
| `Double-source.model`, `Double-source-small.model` | 2 × P |
| `Five-Gauss.model` | 50+ P (despite the name) |
| `Discs.model`, `One-Disc.model` | D primitives |
| `Gauss-and-bigdisc.model`, `Gauss-and-bigdisc-ALMA.model`, `Point-and-Gauss-ALMA.model` | mixed P/G/D |
| `Cloud.model`, `Nebula.model`, `Nebula_small.model`, `Faceon-Galaxy.model`, `RadioGalaxy.model` | typically `IMAGE` plates from `SRC_PICS/{Cyga_21cm,M100,M100-v2,Crab,Orion,Point,just4fun}.png` |

### 8.3 UVFITS for `Func_radplot`

A single example file ships at `OBSERVE_DATA/0106+013_1.fits`. The reader
expects standard UV random-group format; columns are documented at
`Func_radplot.py:22`.

---

## 9. Output products

Outputs are written into `OUTPUT/<subdir>/`:

| Subdirectory | Producer | Filenames |
|---|---|---|
| `uv_basic/` | `Func_uv.py` | `uv-plot:<asctime>.{pdf,png,…}`, optional `u-v:<asctime>.txt` |
| `uv_advance/` | `Func_uv_advanced.py` | mosaic plots for SRCS/SKY/YEAR |
| `obs_ability/` | `Func_obs.py` | `az-el:<asctime>.{pdf,…}`, `sky-survey:<asctime>.{pdf,…}`, optional `el-data:<asctime>.txt` |
| `imaging/` | `Func_img.py` | `img-model/beam/dirty/clean:<asctime>.{pdf,…}` |
| `data_analyze/` | `Func_radplot.py` | `uv-plot:<asctime>.{pdf,…}`, `rad-plot:<asctime>.{pdf,…}`, optional `u-v:*.txt`, `bl-vis:*.txt` |
| `survey_all/` | `Func_survey_all.py` | per-source combined panel |

Each subdirectory ships a `*-notes.md` placeholder (e.g.
`OUTPUT/imaging/imaging-notes.md`) describing what will be written there
and asking the user not to delete the directory itself (the directory is
a path-existence guarantee for `os.path.join` during the CLI runs).

---

## 10. The RIME / interferometry math, as implemented

VNSIM does **not** evaluate the full Radio Interferometer Measurement
Equation (no Jones matrices, no polarization). The visibilities it
"simulates" are the **point-source amplitude transfer function**, i.e.
the Fourier kernel evaluated on the (u, v) sample positions:

```
V(u, v) = Σ_components  S_c · exp(-2πj·(u·Δα_c + v·Δδ_c)) · K_c(u, v)

where K_c is:
  - 1                                      for P (point)
  - exp(-(π·θ_c·sqrt(u²+v²))² / (4·ln2))   for G (Gaussian) [implicit in image-plane raster]
  - 2·J_1(π·θ_c·ρ)/(π·θ_c·ρ)               for D (uniform disc) [implicit]
```

In practice the model is built directly in the **image plane** at
`n_pix × n_pix` resolution, FFT'd to obtain the model coherence on a
regular UV grid, and multiplied by the `mask` of actual (u, v) samples
(`Func_img.py:336–339`). The dirty map is therefore
`FFT⁻¹(model_FFT · sample_mask)` divided by `1.5·beam_scale`.

Sensitivity (`Func_cal.py:319–333`) follows the EVN-Calc convention:

```
σ_image = 1.43 · (1/sqrt(2·t_obs·dRate)) · S̄_eff
        = 1000·1.43 · S̄_eff / sqrt(dRate·1e6/2 · t_obs)   [mJy/beam]
```

with `S̄_eff` the inverse-variance-weighted geometric-mean SEFD.

---

## 11. Notable internals & idioms

- **Comments are bilingual.** Many in-source comments are Chinese
  (Simplified). The English docstrings are reliable, but the line-by-line
  algorithm comments often need translation.
- **Persistence is dual-mirrored.** Every record is canonical in
  `database.db` (SQLite) but every CLI/GUI flow loads from
  `database.pkl` for speed. The pickle is rewritten by
  `DbModel.write_to_pickle()` whenever the GUI editor closes
  (`Func_gui.py:2437`). Editing the SQLite directly with a third-party
  tool requires `python Func_db.py -p`.
- **Time uses MJD throughout.** The internal time variable is always
  `time_mjd` (modified Julian day, days since 1858-11-17 00:00 UT). UI
  uses civil time and converts via `ut.time_2_mjd`.
- **Coordinates are mixed-radix.** Source positions accept both string
  ("3h19m48.16s" / "41d30m42.1s") and pre-computed radians; both forms
  are handled at the entry of `FuncObs._func_tv_az_el` (line 67) and
  `FuncUv._ini_para` (line 95).
- **Dead code paths.** `FuncUv._func_uv_gs` and `_func_uv_ss` are
  documented as space-VLBI but currently return `None` quintuples
  (`Func_uv.py:380–398`). The README's "multiple-satellite space VLBI
  simulations" capability is therefore only partly delivered in the
  open-source tree — visibility judgement and orbit propagation work,
  but the (u, v) generator does not yet form ground-space or
  space-space baselines.
- **Multiprocessing.** `Func_uv_advanced` and `Func_survey_all` use
  `multiprocessing.Pool` + a `Manager().Queue` to fan out independent
  sub-tasks. The GUI uses a single helper `Thread` and a
  `queue.Queue` together with a `GressBar` Toplevel.
- **Logging.** `logging.info(time.asctime() + ": ...")` calls are
  scattered through `Func_gui` and routed to a Tk `ScrolledText` via
  the in-file `TextHandler` class (`Func_gui.py:2385–2406`).
- **No tests.** Despite `pytest==3.2.1` in `requirements.txt`, the
  repository contains no `tests/` directory; `pytest --collect-only`
  finds nothing. A single ad-hoc verifier `test_class_my_data()`
  exists at `Func_gui.py:2475–2480`.

---

## 12. Public API / extension points

There is no installable Python package — all entry points are scripts.
For programmatic reuse the following classes are import-stable:

```python
# Visibility & scheduling
from Func_obs            import FuncObs, ObsConfigParser
from Func_uv             import FuncUv, UVConfigParser
from Func_uv_advanced    import FuncUvMore
from Func_img            import FuncImg, overlap_indices
from Func_radplot        import FuncRadPlot
from Func_survey_all     import FuncSurvey
from Func_cal            import ParaCal              # Tk-bound
from Func_db             import DbEditor, DbModel    # Tk + sqlite

# Pure libraries
import utility            as ut
import load_conf          as lc
import model_satellite    as ms
import model_effect       as me
import model_obs_ability  as mo
```

Every `Func*` class exposes both `get_result_*_with_update()` (compute
+ return) and `update_result_*()` / `get_result_*()` (split form for
multiprocessing). To extend:

1. **Add a station / satellite** — insert into the SQLite table via the
   GUI editor or `DbModel.insert_*`, then `DbModel.write_to_pickle()`.
2. **Add a new source-model component** — extend the `if it[0] in
   ['G','D','P']:` block in `Func_img.py:129` and add a rasteriser in
   `Func_img._prepare_model`.
3. **Implement g-s / s-s baselines** — fill in
   `FuncUv._func_uv_gs` and `_func_uv_ss` (`Func_uv.py:380–398`). The
   per-time-step path is already wired up: call
   `mo.obs_all_active_sat`, propagate the satellite to ITRF with
   `ut.icrf_2_itrf`, then reuse `_get_uv_coordination` exactly as the
   `_func_uv_gg` body does.

---

## 13. Known limitations / TODOs (observed from the source)

- **Space-VLBI uv-coverage is a stub.** As above
  (`Func_uv.py:380–398`).
- **No polarisation, no Jones matrices, no instrumental effects.**
  VNSIM is geometric only; bandpass, gain, ionosphere, troposphere are
  *not* modelled. `Func_cal.py` provides analytic noise estimates only.
- **No proper unit tests.** `pytest` is pinned but unused.
- **Hard-coded SEFD tables.** Adding a station requires editing the
  `_band_*cm` dicts in `Func_cal.py` (lines 497+) by hand.
- **CWD-dependent paths.** Almost every script does
  `os.path.join(os.getcwd(), 'CONFIG_FILE', ...)` /
  `'DATABASE'` / `'OUTPUT'` (e.g. `Func_obs.py:294–295`); running from
  a directory other than the project root breaks the loaders.
- **Old pinned dependencies.** Tested against `numpy==1.13.1`,
  `astropy==2.0.2`, `matplotlib==2.0.2`. Modern versions of
  `matplotlib` removed `NavigationToolbar2TkAgg`
  (used in `Func_gui.py:15, 2317`); `astropy` 4+ broke
  `astropy.io.fits` API minutiae used in `Func_radplot.py:60–72`.
- **Wiki removed from the repo.** The user-facing documentation is
  exclusively on the GitHub Wiki (linked from `README.md:9`) — the
  in-repo HTML guide only covers installation, not usage.
- **No license-of-data clarification.** Source-position strings,
  station coordinates, and the example FITS file have no provenance
  metadata; treat them as "for development only" as the README does.

---

## 14. Quick command cheat-sheet

```
# Launch full GUI
python Func_gui.py

# Per-flow CLIs (each accepts -h)
python Func_uv.py          -c CONFIG_FILE/config_uv.ini      -g -s -i -f png
python Func_uv_advanced.py -c CONFIG_FILE/config_uv.ini      -t 7 -g -n 4
python Func_obs.py         -c CONFIG_FILE/config_obs.ini     -g -s -i -f pdf
python Func_img.py         -c CONFIG_FILE/config_img.ini     -g -i -f png
python Func_radplot.py     -f 0106+013_1.fits                -g -s -t pdf
python Func_survey_all.py  -c CONFIG_FILE/config_survey.ini  -i -s -n 4 -f png

# Database housekeeping
python Func_db.py -g          # open editor
python Func_db.py -p          # database.db -> database.pkl

# Defaults dump
python load_conf.py
```

---

*This document was produced by reading every Python module, the INI
configs, the SQLite schema, the model-file syntax, and the installation
guide directly from the submodule. No external sources were consulted,
and no other `.md` file in `simulators/` was read.*
