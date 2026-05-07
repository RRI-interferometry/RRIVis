# OmniUV — Exhaustive Technical Reference

> A self-contained reference for the **OmniUV** package vendored at
> `simulators/omniuv/` (commit `460183e`, tag `v1.04.1`). Every claim here
> cites a file/line in that tree. No cross-references to any other vendor docs.

---

## 1. Overview

**OmniUV** is *"a multi-purpose simulation toolkit for VLBI observation"* — a
Python toolkit that calculates baseline *uvw* tracks and synthesizes
visibilities/dirty-images for **ground, Earth-orbit, Moon-fixed, lunar-orbit,
Earth–Sun L2, Moon–Earth L2, and Distant-Retrograde-Orbit (DRO)** stations
— effectively a *space VLBI / orbital uv-coverage simulator*.

Source: `simulators/omniuv/README.MD` lines 6–8 ("We develop OmniUV, so as to
fulfill the requirement of simulation for both ground and space VLBI
observations") and lines 27–32 (functionality bullets).

| Field | Value |
|---|---|
| Repository | `https://github.com/liulei/omniuv.git` (verified via `git remote -v`) |
| Submodule path in this tree | `simulators/omniuv/` |
| Latest commit at vendoring | `460183e` "Update reference in readme" |
| Latest annotated tag | `v1.04.1` |
| Vendored author | Lei Liu, `liulei@shao.ac.cn` (Shanghai Astronomical Observatory) — credit line in `simulators/omniuv/src/aperture_array.py` line 3: `# Copyright 2022-2024. Lei Liu (liulei@shao.ac.cn) all rights reserved.` |
| Citation | Liu, L., Zheng, W., Fu, J., Xu, Z., 2022, *AJ*, 164, 67, [arXiv:2201.03797](https://arxiv.org/abs/2201.03797), per `README.MD` line 14 |
| License | **GPL-3.0** — `simulators/omniuv/LICENSE` line 2: "GNU GENERAL PUBLIC LICENSE / Version 3, 29 June 2007" |
| Languages | Python 3.11+ (see commit `58a1bca` "Update to be compatible for the higher version of Python (>=3.11)") and ANSI C (the IAU **SOFA** astrometric library, 245 `*.c` files in `data/sofa/`) |
| Total Python lines | **6,286** (output of `wc -l`, see breakdown in §4) |
| Distribution form | source-only — *no* `setup.py`, *no* `pyproject.toml`, *no* `setup.cfg`. Loaded by `sys.path.append(...)` and `from omniuv import *`, see `run_example.py` lines 11–14 |

### 1.1 Functional bullets (verbatim, `README.MD` lines 27–32)

- Trajectory calculation
- Baseline *uv* calculation, by taking the availability of each station into account
- Visibility simulation for the given *uv* distribution, source structure and system noise
- Image and beam reconstruction

### 1.2 Two imaging methods supported

`README.MD` lines 34–45:

- **FFT** — assignment-function gridding then `numpy.fft.fft2`. Fast; small field; *w*-term ignored; gridding artifacts.
- **DFT** — direct `Σ V·exp(2πi (u·l + v·m + w·(n−1)))` per pixel. Slow but GPU-accelerable; wide field; *w*-term natively supported; no gridding artifacts.

---

## 2. Repository layout

Output of `find … -type f -not -path '*/.git/*'` (selected):

```
simulators/omniuv/
├── LICENSE                            # GPL-3.0, 35 149 B
├── README.MD                          # 9 351 B, project doc
├── requirements.txt                   # 7 deps (no version pins)
├── pip_mirror.sh                      # Aliyun mirror helper for users in mainland China
├── run_example.py                     # 372-line reference driver
├── tool.py                            # 264-line top-level plotting helpers
├── omniuv -> src                      # symlink: `import omniuv` resolves to ./src
├── data/
│   ├── dataflow.png                   # architecture diagram (PNG)
│   ├── duvw.png  logo.png  logo_flat.png
│   ├── de421.bsp                      # 16.0 MB JPL DE421 SPK ephemeris
│   ├── moon_pa_de421_1900-2050.bpc    # Moon Principal-Axis PCK kernel
│   ├── ic_dro.txt                     # 20 DRO initial conditions (Zimovan 2017 Tab. 4.1)
│   └── sofa/                          # IAU SOFA C library, 245 *.c, 2 *.h, 1 makefile
├── dro/                               # Distant-Retrograde-Orbit demo
│   ├── gen_dro_rot.py
│   ├── plot_dros.py
│   └── fig/                           # 6 PNGs of integrated DRO trajectories
└── src/                               # Python package (imported as `omniuv`)
    ├── __init__.py                    # re-exports
    ├── base.py                        # Source, Image, Task, Station base classes
    ├── util.py                        # constants, EOP, time, SPK, frame matrices
    ├── backend.py                     # CPU/GPU direct-image kernels, multiprocessing
    ├── earth_fixed.py                 # ground station
    ├── earth_orbit.py                 # Earth-orbiter (Keplerian)
    ├── lunar_orbit.py                 # Moon-orbiter (Keplerian, lcs+crs)
    ├── earth_sun_L2.py                # passive ES-L2 halo proxy
    ├── moon_earth_L2.py               # passive ME-L2 proxy
    ├── dro.py                         # 4-body DRO via solve_ivp
    ├── aperture_array.py              # phased-aperture beam-pattern simulator
    ├── _el_sep.py                     # mixin: elevation & separation visibility checks
    ├── _sim.py                        # mixin: vis/beam/image FFT+DFT
    ├── _fits.py                       # mixin: FITS-IDI export
    └── orbital/                       # vendored copy of Frazer McLean's `orbital` 0.7.0
        ├── __init__.py    bodies.py    constants.py
        ├── elements.py    maneuver.py  plotting.py    utilities.py
```

`pip_mirror.sh` rewrites `~/.pip/pip.conf` to use Aliyun (`mirrors.aliyun.com`)
— purely an installation convenience for mainland-China users (`README.MD`
lines 122–126).

The `omniuv -> src` symlink at the repo root is what makes `from omniuv
import *` work without any package metadata: see `run_example.py` line 11
(`sys.path.append('.')`) followed by line 14 (`from omniuv import *`).

---

## 3. Installation, dependencies, build

### 3.1 Python dependencies

`simulators/omniuv/requirements.txt` (verbatim, no version pins):

```
numpy
matplotlib
astropy
jplephem
represent
scipy
sgp4
```

What each is used for (verified by import sites):

| Package | Used in | Purpose |
|---|---|---|
| `numpy` | everywhere | array math |
| `matplotlib` | `tool.py`, `_sim.py`, `dro/plot_dros.py`, `aperture_array.py` | plotting (`matplotlib.use('Agg')` forced in `_sim.py:6`) |
| `astropy` | `util.py:6`, `base.py:6`, `_sim.py:9`, all station classes | `astropy.constants.R_earth`, `astropy.constants.c`, `astropy.time.Time`, units |
| `jplephem` | `util.py:4` (`from jplephem.spk import SPK`), `util.py:55` (`load_spk()`), `util.py:62` (`load_pck()`) | reads `data/de421.bsp` and `data/moon_pa_de421_1900-2050.bpc` |
| `represent` | `src/orbital/elements.py:13` (`from represent import ReprHelperMixin`), `bodies.py:5` | pretty-print mixin |
| `scipy` | `dro.py:4` (`scipy.integrate.solve_ivp`), `orbital/utilities.py:13`, `orbital/constants.py:6` | numerical ODE + constants |
| `sgp4` | `orbital/elements.py:9–10,15` (`sgp4.io`, `sgp4.propagation`, `sgp4.earth_gravity.wgs72`) | TLE propagation in the vendored `orbital` package |

Optional native dependency:

- **GCC + Make** — required only for the SOFA library (precession/nutation
  precision). Build via `cd data/sofa && make` to produce `libsofa_c.so`
  (`README.MD` lines 138–144). At runtime `util.py` lines 124–129
  conditionally `ctypes.CDLL`-loads `data/sofa/libsofa_c.so` if it exists; if
  absent the precession/nutation matrix is forced to identity (`util.py` lines
  230–232).

- **GPU support is commented out**: `backend.py` line 4 (`#import cupy as cp`).
  `_sim.py` line 13 imports both `direct_image_np` and `direct_image_cp`, but
  only the NumPy path is actually wired in (line 323). The cupy path is dead
  code unless the user uncomments the import.

### 3.2 Install procedure (`README.MD` §Installation)

```bash
# Optional, if user is in mainland China
sh pip_mirror.sh

# Python deps
pip install -r requirements.txt

# Optional: build SOFA for full precession/nutation precision
cd data/sofa && make    # generates libsofa_c.so

# Use the library: there is no `python setup.py install`,
# the user must put the repo on PYTHONPATH or sys.path:
import sys; sys.path.append('/path/to/omniuv')
from omniuv import *
```

### 3.3 EOP precision contract (`README.MD` lines 150–157)

| EOP term | Default in `run_example.py` | Trajectory error if missing |
|---|---|---|
| `tmu` (TAI-UTC) | 35.0 | 2 cm at 35 s |
| `dut1` (UT1-UTC) | -2.211724 | 500 m per 1 s |
| Polar motion `xp`, `yp` | 0.027840, 0.370850 | 10 m (30 m / arcsec) |
| Precession/nutation | requires SOFA | 10 km (without SOFA) |

The note in `README.MD` lines 147–148 states OmniUV trajectories were validated
against CALC 9.1 with discrepancy at the **20 cm level** (figure
`data/duvw.png`).

---

## 4. Build & runtime architecture

```
              ┌─────────────────────────────────────────────┐
              │            User script (run_example.py)     │
              │                                             │
              │   t = Task()                                │
              │   t.set_srcs([Source(...), ...])            │
              │   t.set_eops(tmu, dut1, xp, yp)             │
              │   t.set_ts(t0, ts)                          │
              │   t.set_stns([EarthFixed/EarthOrbit/...])   │
              │   uvw_bls = t.calc_uvw_bl()                 │
              │   bls     = t.gen_vis_fft(src)              │
              │   bls     = t.vis_add_noise(src, bls)       │
              │   beam    = t.gen_beam(bls)                 │
              │   uv,img  = t.gen_image_fft(bls)            │
              │   t.to_fitsidi(name, bls)                   │
              └────────────────────┬────────────────────────┘
                                   │
       ┌───────────────────────────┴──────────────────────────────┐
       │                       omniuv.Task                        │
       │  composes mixins via   from ._sim   import ...           │
       │                       from ._fits  import ...            │
       │  (base.py lines 192–199)                                 │
       └───────────────────┬──────────────────────────────────────┘
              ┌────────────┴───────────┬──────────────┐
   ┌──────────▼─────────┐  ┌───────────▼─────┐  ┌─────▼─────┐
   │ Trajectory layer   │  │ Visibility layer│  │ I/O layer │
   │ (Station subclass) │  │  _sim.py        │  │ _fits.py  │
   │   *.calc_crs()     │  │  FFT + DFT      │  │ FITS-IDI  │
   │   →  p_crs (CRS)   │  │  noise model    │  │ export    │
   └──────────┬─────────┘  └───────────┬─────┘  └───────────┘
              │                        │
   ┌──────────▼──────────┐    ┌────────▼────────┐
   │ Reference frames    │    │ DFT kernel      │
   │ util.calc_t2c_R     │    │ backend.py      │
   │ util.calc_t2c_PN_W  │    │  direct_image_np│
   │ jplephem  (SPK)     │    │  (multiprocess) │
   │ ctypes  (libsofa)   │    │ + dead cupy path│
   └─────────────────────┘    └─────────────────┘
```

The architecture is intentionally **flat** — there is no plugin system, no
config file, no CLI. The entry point is the user-authored script that
instantiates a single `Task` and feeds it Station objects. The `Task` class is
*built up by side-effect imports*: `base.py` line 192 ends with
`from ._sim import gen_vis_direct, gen_vis_fft, …, set_uv_image_param` and
line 198 `from ._fits import to_fitsidi, …`. These appear inside the class
body, attaching free functions as bound methods.

---

## 5. Per-file Python breakdown

### 5.1 `simulators/omniuv/src/__init__.py` (12 lines)

Top-level re-exports (verbatim):

```python
from .base import Source, Task, Station, Image
from .earth_fixed import EarthFixed
from .earth_orbit import EarthOrbit
#from .lunar_fixed import LunarFixed
from .lunar_orbit import LunarOrbit
from .earth_sun_L2  import EarthSunL2
from .moon_earth_L2 import MoonEarthL2
from . import util
from . import backend
```

`LunarFixed` is **commented out** — `README.MD` lines 154–155 invite
collaboration (`Please contact the author (liulei@shao.ac.cn) for
collaboration`). The class is referenced in `dro/gen_dro_rot.py` line 77 but
the class file is not included in this distribution.

### 5.2 `simulators/omniuv/src/base.py` (283 lines)

Defines four classes:

| Class | Lines | Role |
|---|---|---|
| `Image` | 10–17 | Pixel-list image bound to a `Source`. Stores `ra0`, `dec0`, `ruvw0`, and a list `lmn` filled by the caller (see `run_example.py` lines 297–306 for the loop that converts each pixel `(ra,dec)` to a unit vector and projects onto the source's `ruvw` basis). |
| `Source` | 19–24 | Plain-dataclass-style container: `ra` and `dec` (radians), `name`. Note `README.MD` line 50: "A Source in OmniUV actually refers to a *phase center*". |
| `Task` | 26–199 | Orchestrator. Holds stations, sources, time grid, EOPs, frequency grid, and via late-imports the simulation/FITS methods. |
| `Station` | 202–283 | Abstract base. Subclasses implement `calc_crs()`. Mixes in elevation/separation logic from `_el_sep.py`. |

Key `Task` methods:

| Method | Lines | What it does |
|---|---|---|
| `set_eops(**kw)` | 43–45 | `setattr(self, k, v)` for each kw — typically `tmu, dut1, xp, yp`. |
| `calc_jd(t)` | 49–54 | UTC `datetime` → JD(TT) using `DT_JD + (t-T0_MJD).total_seconds()/86400 + (32.184+tmu)/86400`. |
| `set_ts(t0, ts)` | 56–60 | Stores `t0` (UTC datetime) and `ts` (offsets in seconds), pre-computes JDs for SPK lookups. |
| `calc_gain()` | 62–72 | Per-time **multiplicative gain amplitude error**: Gaussian `N(1, gain_error)`, clipped at 0; phase set to **zeros** (line 71 — explicit comment-out of a uniform-phase model). |
| `set_stns(stns)` | 75–113 | (a) checks unique names; (b) lazy-loads `de421.bsp` if any station mentions Moon/Sun via `util.load_spk()`; (c) lazy-loads PCK if any LunarFixed station; (d) builds `bl2stn` index list `(i,j)` for `i<j`; (e) populates `stn.gain_amp/gain_phase` if `task.gain_error` was set. **Number of baselines = N(N−1)/2**, line 78. |
| `set_srcs(srcs)` | 115–124 | Computes `src.ruvw` once via `util.rqu2ruvw(util.crs2rqu(ra, dec))` — the (u,v,w) basis triple in CRS for each phase center. |
| `calc_crs_sun()` / `calc_crs_moon()` | 142–149 | Vectorised SPK queries giving CRS Sun/Moon position arrays of shape `(nt, 3)` in metres. |
| `calc_crs_stn()` | 151–153 | Calls every station's `calc_crs()` to populate `stn.p_crs`. |
| `calc_uvw_stn()` | 156–159 | Per-station: project `p_crs` onto each source's `ruvw` triple, store as `stn.uvw` shaped `(nsrc, nt, 3)`. |
| `calc_uvw_per_bl(s1, s2)` | 126–139 | Intersect the two stations' availability arrays and subtract `uvw`. |
| `calc_uvw_bl()` | 161–177 | Master driver; returns array indexed `[bl][src][t,3]` and stores `self.idt_avail`, `self.uvw`. |
| `flat_uvw_bl(uvw)` | 180–190 | Concatenates per-baseline arrays into per-source `(N_total, 3)`. |

Late-imports (the *secret sauce*) at lines 192–199:

```python
from ._sim   import gen_vis_direct, gen_vis_fft, \
                    gen_image_fft, gen_image_direct, gen_beam, \
                    plot_uv, plot_image, plot_beam, plot_src, \
                    vis_add_noise, gen_image_fft, gen_weight, \
                    uv2id, set_uv_image_param
from ._fits  import to_fitsidi, gen_Primary, gen_UV, gen_SU, \
                    gen_FR, gen_AN, gen_AG, add_fits_keys
```

These appear *inside* the class body, so they become bound methods of `Task`.
This is how visibility/imaging/FITS code stays in separate files yet is
callable as `task.gen_vis_fft(...)`.

`Station` (base, lines 202–283):

| Method | Lines | Notes |
|---|---|---|
| `__init__(name)` | 204–212 | `uvw_updated=False`, `sep_min={}`. |
| `set_task(task)` | 214–215 | Backref. |
| `set_orbit(a, e, i, raan, arg_pe, M0, t_ref=None)` | 217–244 | Only valid for `type=='orbit'`. Validates pericentre `(1-e)*a >= R_body`. Uses the vendored `KeplerianElements`. |
| `set_SEFD(SEFD)` | 246–247 | System Equivalent Flux Density in Jy (used in noise model). |
| `calc_crs()` | 249–252 | Abstract — exits with error if not overridden. |
| `calc_uvw()` | 254–273 | Calls `calc_crs()`, projects onto each source's `ruvw` via `np.einsum('ik,jk->ij', self.p_crs, src.ruvw)`. Caches via `uvw_updated`. Calls mixin `calc_idt_avail()` at line 267. |

Mixed-in availability helpers (line 275–283) — see `_el_sep.py` below.

### 5.3 `simulators/omniuv/src/util.py` (414 lines) — physics & frames

Constants (lines 12–22):

| Symbol | Value | Source |
|---|---|---|
| `gme` | `3.986004418e14` m³/s² | "REF: 1" |
| `c` | `299792458.0` m/s | exact |
| `Re` | `astropy.constants.R_earth.value` | astropy |
| `Rm` | `1737.4e3` m | Moon mean radius |
| `Rs` | `6.957e8` m | Sun radius |
| `r_ES_L1` / `r_ES_L2` | `0.00997040269846281` / `0.0100371199532807` | Earth–Sun Lagrange ratios |
| `r_ME_L1` / `r_ME_L2` | `0.150909863379802` / `0.167802559989682` | Moon–Earth Lagrange ratios |
| `DT_JD` | `2400000.5` | JD↔MJD offset |
| `T0_MJD` | `datetime(1858, 11, 17, 0, 0, 0)` | MJD epoch |
| `path_home` | `'.'` | Used to locate `data/de421.bsp` etc. — implicitly assumes `cwd == repo_root`. |

`cbs` dict (lines 24–28) maps central-body name to radius:
`{'Earth': {'R': Re}, 'Moon': {'R': Rm}, 'Sun': {'R': Rs}}`.

`M_in_E` (lines 33–42): Solar-system body mass ratios relative to Earth, used
for ephemeris.

Key functions:

| Function | Lines | Description |
|---|---|---|
| `load_spk()` | 49–56 | Lazy `SPK.open('./data/de421.bsp')`, idempotent via `globals()`. |
| `load_pck()` | 58–64 | Lazy `PCK.open('./data/moon_pa_de421_1900-2050.bpc')`. |
| `intersect1d_many(idss)` | 67–74 | n-way `np.intersect1d`. |
| `utc2mjd(t)` | 77–78 | Datetime → MJD. |
| `crs2rqu(ra, dec)` | 80–85 | (ra, dec) → unit vector `(cos(dec)cos(ra), cos(dec)sin(ra), sin(dec))`. The "**s0**" of `README.MD` §Source. |
| `rqu2ruvw(rqu)` | 87–116 | Builds the `(u, v, w)` orthonormal triple for a phase-centre direction; `w = s0`, `u = z×w`, `v = w×u`. **The function body is duplicated** (lines 89–101 and 103–116 — likely an editing mistake; both branches compute the same thing). |
| `calc_uvw(rqu, b)` | 118–122 | One-shot baseline projection. Not used internally; kept for API. |
| `xys2006a(tt)` | 130–149 | `ctypes` wrapper calling `libsofa.iauXys06a()` for IAU 2006A precession/nutation X,Y,s. |
| `rotm(angle, tp)` | 151–179 | 3×3 rotation matrix about axis `tp ∈ {1,2,3}`. |
| `as2rad(v)` | 181–182 | arcsec → rad. |
| `interp(n, y, x0)` | 184–188 | Quadratic poly-fit interpolation; used by EOP. |
| `get_eop(eop, mjd)` | 190–201 | Looks up `tmu, dut1, xp, yp` from a table object — the table format is implied (`eop.EOP_time`, `eop.tai_utc`, `eop.ut1_utc`, `eop.xpole`, `eop.ypole`); **the loader for that object is not in the repo**, so `get_eop` is dormant in normal use (users supply EOPs by hand to `Task.set_eops`). |
| `calc_t2c_R(dut1, mjd)` | 203–214 | Earth-rotation-angle (ERA) matrix only; the simplest of the three TRS→CRS factors. |
| `calc_t2c_PN_W(mjd, tmu, dut1, xp, yp)` | 216–244 | Builds `W` (polar-motion+TIO) and `PN` (precession+nutation). If `libsofa is None`, `PN` falls back to identity. |
| `t2c(din)` | 246–285 | Combined TRS→CRS = `PN · R · W`. Equivalent to but separate from the per-step matrices used by `EarthFixed`. |
| `norm(v)` | 287–288 | Scalar Euclidean norm. |
| `valid_jd(jd)` | 290–295 | Range check `2414864.50 ≤ jd ≤ 2471184.50` (epoch coverage of the bundled DE421 segment). |
| `get_crs_moon(jd)` / `get_crs_sun(jd)` | 299–318 | SPK chain: `spk[3, 301].compute(jd) - spk[3, 399].compute(jd)` for Moon, three-segment chain for Sun. Returns metres. |
| `mas2rad`, `deg2rad`, `rad2mas`, `rad2min`, `rad2deg` | 368–413 | Scalar unit conversions (mas2rad is duplicated in `tool.py`). |
| `h2ae(hl, hh, R)` | 374–379 | Converts apoapsis/periapsis altitudes plus body radius to (a, e). |
| `calc_beam_param(uv, ws=None)` | 381–404 | DIFMAP-TPJ-style elliptical-Gaussian fit to the dirty beam from raw `uv` second moments. Returns `(bmaj, bmin, bpa)` in rad. |

`plot_crs(task)` and `plot_lcs(task, name)` (lines 320–366) are simple
matplotlib 3-D scatter helpers using `stn.p_crs` / `stn.p_lcs`.

### 5.4 `simulators/omniuv/src/_el_sep.py` (115 lines) — visibility gating

Mixed into `Station` via `from ._el_sep import …` at the bottom of the
`Station` class. Functions:

| Function | Lines | Description |
|---|---|---|
| `set_sep_min_deg(self, **kw)` | 4–11 | E.g. `set_sep_min_deg(Earth=5.0, Moon=1.0, Sun=10.0)` — stores radian thresholds in `self.sep_min`. |
| `_calc_sep(self, dp, ruvw, R0)` | 13–17 | Computes the angular distance between source-line-of-sight and the limb of a body of radius `R0` at distance `dp`. Formula: `π − asin(R0/r) − acos((dp/r)·w)`. |
| `calc_sep_moon` / `_earth` / `_sun` | 19–35 | Same kernel applied with body-specific origin and radius. |
| `calc_idt_sep_src(src)` | 37–71 | Returns the indices of `task.ts` for which the source is sufficiently far from each requested obstruction body. |
| `set_el_min_deg(deg)` | 73–74 | For Earth/Moon-fixed stations only. |
| `_calc_el(dp, ruvw)` | 77–79 | Elevation = `π/2 − arccos((dp/|dp|)·w)`. |
| `calc_idt_avail(self)` | 81–114 | The master availability calculator. For each source: combine elevation and separation indices via `np.intersect1d`. Stores `self.el`, `self.sep`. |

### 5.5 Station subclass files (one per orbit type)

| File | Lines | Class | type | cb (central body) | crs from … |
|---|---|---|---|---|---|
| `earth_fixed.py` | 40 | `EarthFixed` | `'fixed'` | `'Earth'` | `set_trs(p_trs)` then per-time `t2c = PN·R·W` (lines 22–31) |
| `earth_orbit.py` | 35 | `EarthOrbit` | `'orbit'` | `'Earth'` | Vendored `orbital.KeplerianElements(body=earth)`; samples `self.orbit.r` at each `t` (lines 19–34) |
| `lunar_orbit.py` | 36 | `LunarOrbit` | `'orbit'` | `'Moon'` | `KE(body=moon)` for the **lunar-centred** state vector `p_lcs`, then translated by `task.crs_moon` to give CRS (line 35) |
| `earth_sun_L2.py` | 23 | `EarthSunL2` | `'orbit'` | `''` (none) | `p_crs = -task.crs_sun * util.r_ES_L2` (line 21) — the L2 point lies along the Sun-Earth line, *outside* Earth, at fractional distance `r_ES_L2`. **Approximation: treats the station as fixed at L2, ignoring halo motion.** |
| `moon_earth_L2.py` | 23 | `MoonEarthL2` | `'orbit'` | `''` (none) | `p_crs = task.crs_moon * (1 + r_ME_L2)` (line 21) |
| `dro.py` | 253 | `DRO` | `'dro'` | `'Moon'` | Numerical integration (see §6.5) |

The `set_orbit(a, e, i, raan, arg_pe, M0, t_ref=None)` API in `Station`
(`base.py:217-244`) is shared by `EarthOrbit` and `LunarOrbit`; both call into
`KeplerianElements` from `orbital`.

### 5.6 `simulators/omniuv/src/dro.py` (253 lines) — Distant Retrograde Orbits

Class `DRO(Station)`:

- `__init__(name)` (lines 12–17) — `type='dro'`, `cb='Moon'`.
- `set_param(orbit_id, **kw)` (19–27) — loads `data/ic_dro.txt` (20 ICs, the
  Zimovan 2017 Tab. 4.1 set; column meaning per `ic_dro.txt` header: `JC
  Period x0 y0_dot`) and stores extra `solve_ivp` kwargs (e.g. `method`,
  `max_step`).
- `gen_init()` (29–70) — builds the *inertial-frame* 6-vector initial state.
  Computes Earth and Moon position/velocity from SPK at `t0`, derives the
  Earth-Moon-system Kepler velocity `v_kep = √((μ_E+μ_M)/r)`, then
  scales the dimensionless DRO IC `(x0, y0_dot)` by `(r, v_kep)`.
- `gen_init_rot()` (105–125) — builds the *rotating-frame* IC (with hard-coded
  normalisation: `v_n = 738.23 m/s / 0.720544`, `r_n = 9 × 10⁶ m / 0.023413`,
  `m₂ = 1.215059e-2`).
- `calc_rot()` (127–183) — `solve_ivp(f_rot, …)` with rotation-frame equation
  of motion: gravity from Earth (`m1 = 1 - m2`) + gravity from Moon (`m2`) +
  Coriolis + centrifugal terms. Used only for demonstration plots
  (`do_dro_rot=True`).
- `calc_crs()` (186–252) — production path. Integrates the **two-body Earth +
  Moon** problem in the inertial frame using `solve_ivp(f, t_span, y0,
  **self.kw)` where `f` looks up Earth/Moon positions from the JPL SPK at
  every step. Then `calc_p_rot()` (72–100) post-projects the inertial trajectory
  into the Moon-Earth rotating frame for plotting.

Notes:

- `dro.py` line 7 imports `earth_mu, moon_mu` from `.orbital`. They originate
  in `src/orbital/constants.py:15,46` and are propagated through `from .constants
  import *` in `orbital/__init__.py:5`.
- `utc2tt(t)` (line 102–103): `tt = JD + MJD + (32.184+tmu)/86400`.
- The `LunarFixed` station referenced in `dro/gen_dro_rot.py` line 77 is
  intentionally not shipped (`README.MD` line 154–155 directs users to email
  the author).

### 5.7 `simulators/omniuv/src/_sim.py` (780 lines) — visibility / beam / image

Mixed into `Task`. The module begins with `matplotlib.use('Agg')` (line 6),
forcing headless plotting.

| Function | Lines | What it does |
|---|---|---|
| `set_uv_image_param(self)` | 26–37 | Derives `urange = 1/cellsize_rad`, `umax`, `umin`, `du = urange/nc`. Cellsize is taken in **mas**; converted via `util.mas2rad`. |
| `uv2id(self, uv)` | 39–52 | Nearest-neighbour `(uv) → (iv, iu)` integer grid index. Returns `(-1, -1)` for samples outside the `[umin, umax]² ` range — used by all gridders. |
| `gen_vis_fft(self, src)` | 54–145 | (1) Rasters the user image onto an `(nc, nc)` array by nearest-pixel assignment using `dra·cos(dec)`, `ddec`. (2) `vis_uv = fftshift(fft2(fftshift(arr)))`. (3) For each baseline & frequency, look up the gridded `vis_uv` at the nearest `(u/λ, v/λ)`. Returns `bls`: list of dicts with keys `uvw_m, t, uvw_wav, vis`. |
| `gen_vis_direct(self, src)` | 147–188 | True DFT: `lmn = src.img.lmn - [0,0,1]`, `lmn_uvw = einsum('hk,ijk->hij', lmn1, uvw_wav)`, `fringe = exp(-2j·π·lmn_uvw)`, `vis = einsum('h,hij->ij', fluxes, fringe)`. |
| `vis_add_noise(self, src, bls)` | 190–223 | Per-baseline thermal-noise sample with σ derived from the radiometer eqn `σ = √(SEFD₁·SEFD₂ / (2·BW·t_ap)) / η`, `η = 0.88` (two-bit quantisation efficiency, set in `base.py:38`). Adds `eps_amp · exp(i·eps_phase)` to vis, then multiplies by per-time gain `√(g₁·g₂)·exp(i(φ₁−φ₂))`. |
| `gen_weight(self, uv)` | 423–465 | Computes per-vis weight: 1.0 (natural) by default; if `do_unif=True` divide by per-cell count (uniform); if `do_rad=True` multiply by `|uv|/1e6` (radial taper). |
| `gen_image_fft(self, bls, …)` | 467–582 | Adds Hermitian conjugate, drops out-of-range samples, accumulates `image_uv[iv,iu] += vis · w`, `image = real(fftshift(ifft2(fftshift(image_uv)))) · nc²`. Optional beam-correction from elliptical-Gaussian beam params. |
| `gen_image_direct(self, bls)` | 225–354 | Same gather, but constructs the `(nc²) × 3` `lmn1 = (l, m, n−1)` grid in the source `ruvw` basis, then dispatches to `direct_image_np` from `backend.py`. |
| `gen_image_fft_nowt(self, bls)` | 584–646 | Older un-weighted variant kept for reference. |
| `gen_beam(self, bls, …)` | 356–421 | Identical gather as `gen_image_fft` but accumulates only the weights into `beam_uv` then `ifft2`. Normalised by `np.max(beam)`. |
| `plot_uv` / `plot_beam` / `plot_image` / `plot_src` | 648–778 | Matplotlib helpers attached as Task methods. |

`do_unif`, `do_rad`, `do_beam_correction` are toggled on `Task` (defaults
`False`, see `base.py:39-41`). `cs_src_rad` controls `do_beam_correction`'s
flux normalisation (mentioned in `run_example.py` lines 340–347).

`NA = 0.0` (line 21) is used as a *sentinel* meaning "no visibility here"
in the FFT-gridded `vis_uv` lookups.

### 5.8 `simulators/omniuv/src/_fits.py` (446 lines) — FITS-IDI export

Six HDU generators glued together by `to_fitsidi(self, name, bls)` (lines
432–445). The output is a 6-extension FITS-IDI file ready for `AIPS fitld`:

| Function | Output extension |
|---|---|
| `gen_Primary` | `PrimaryHDU` with telescope/observer/origin keys (`TELESCOP='VLBA'`, `CORRELAT='DIFX'`) |
| `gen_AG` | `ARRAY_GEOMETRY` — station ECEF coords (only Earth-fixed stations get real `STABXYZ`; orbiters get `np.ones(3)` per line 97) |
| `gen_SU` | `SOURCE` — single source (only `srcs[0]` is exported; line 162 hard-codes `'3C288'`) |
| `gen_AN` | `ANTENNA` — polarisations forced to R/L (lines 269, 273) |
| `gen_FR` | `FREQUENCY` — `BANDFREQ`, `CH_WIDTH`, `TOTAL_BANDWIDTH`, all uniform across IFs |
| `gen_UV` | `UV_DATA` — the actual visibility records, `UU/VV/WW` in seconds (`/c_light`), `BASELINE = (s0+1)*256 + s1+1`, and the complex `FLUX` matrix flattened to interleaved float32 re/im. |

Key concerns documented in `README.MD` lines 168–175:

> "At present visibility records are not sorted in time order. This leads to a
> warning when importing data in AIPS using task fitld. To facilitate further
> data processing, the recommended procedure for AIPS is: fitld → uvsrt with
> sort='TB' (very important) → indxr."

`add_fits_keys` (lines 35–49) writes the per-extension keys
`OBSCODE, RDATE, NO_STKD=1, STK_1=-1, NO_BAND=len(freqs), NO_CHAN=1,
REF_FREQ=freqs[0], CHAN_BW=bandwidth, REF_PIXL=1.0`. The 1-channel-per-band
constant `NCHAN = 1` is hard-coded at line 9.

`arrayGMST(mjd)` (lines 11–33) reproduces the Aoki et al. GMST formula in
fractional days for the `GSTIA0` keyword.

### 5.9 `simulators/omniuv/src/backend.py` (135 lines) — DFT kernels

Two near-identical workers:

```python
direct_image_np(lmn1, uvw, vis, nc, s_mem_max)   # CPU
direct_image_cp(lmn1, uvw, vis, nc, s_mem_max)   # GPU (cupy, currently dead)
```

Both segment the `(npixel, nvis)` outer-product memory into chunks of size
`s_mem_max` (default 12 GB at the call site `_sim.py:323`). The CPU kernel:

```python
def kernel(_lmn1):
    lmn1_uvw = np.einsum('hk,jk->hj', _lmn1, uvw)            # (npix_seg, nvis)
    ph       = -2.*np.pi * (lmn1_uvw - np.floor(lmn1_uvw))   # phase wrap
    image1   = np.einsum('ij,j->i', np.cos(ph), np.real(vis))
    image2   = np.einsum('ij,j->i', np.sin(ph), np.imag(vis))
    return image1 + image2
```

The cupy version (`worker_cupy`) launches one process per GPU and consumes
segment IDs from a `multiprocessing.Queue`. Because the `import cupy as cp` at
line 4 is **commented out**, calling `direct_image_cp(...)` would raise
`NameError`. This is a known dormant optimisation path.

The phase-wrap trick (`(x - floor(x))` before `2π·`) reduces argument
magnitude before the trig calls; valid because cosine and sine are
2π-periodic.

### 5.10 `simulators/omniuv/src/aperture_array.py` (229 lines)

Phased-aperture *beam-pattern* simulator — independent of the rest of the
package. `class ApertureArray(lam, pos_tiles, tiles=[])` computes the
amplitude response on a (θ, φ) sky grid via vectorised `exp(i·ψ)` summation
(lines 68–127). Hierarchical: an array of tiles, each tile itself an
`ApertureArray`. Demo functions `create_tile()` (143–178) and `test_array()`
(180–225) build a 16×16 dipole tile then a 16×16 array of tiles. Output goes
through `plotxyproj()` (10–56) which projects θ, φ to a tangent-plane image
(±10° default) and saves a PNG.

This file is **not used** by `Task`/`Station`/`run_example.py`; it is a
stand-alone tool added in commit `475c7c5` "Support for aperture array beam
pattern simulation".

### 5.11 `simulators/omniuv/src/orbital/` — vendored 3rd-party package

`__init__.py` (line 16): `__version__ = '0.7.0'`, `__author__ = 'Frazer
McLean'`, `__license__ = 'MIT'`, `__description__ = 'High level orbital
mechanics package.'`

| File | Lines | Highlights |
|---|---|---|
| `bodies.py` | 202 | `Body` class + module-level instances `mercury, venus, earth, mars, jupiter, saturn, uranus, neptune, moon`. Each has `mass, mu, mean_radius, equatorial_radius, polar_radius, apoapsis_names, periapsis_names, plot_color`. |
| `constants.py` | 74 | IAU 2009 constants: `solar_mass_parameter = 1.32712440041e20`, `earth_mu = 3.986004415e14`, `moon_mu`, etc. **Used by `dro.py`.** |
| `elements.py` | 479 | `KeplerianElements` class — exposes `a, e, i, raan, arg_pe, M0`, `M`, `T` (period), `r`, `v`, `t` (time-of-flight) properties. `with_altitude`, `with_period`, `with_apside_altitudes`, `with_apside_radii`, `from_state_vector`, `from_tle` constructors. |
| `maneuver.py` | 820 | `Maneuver`, `Operation`, `PropagateAnomalyBy`, `PropagateAnomalyTo`. Not exercised by OmniUV's primary path. |
| `plotting.py` | 330 | Plotting helpers (Frazer McLean upstream). Not used by `_sim.py`. |
| `utilities.py` | 481 | `eccentric_anomaly_from_mean`, `true_anomaly_from_*`, `radius_from_altitude`, `elements_from_state_vector`, `uvw_from_elements` (different "uvw" — orbital perifocal frame, not the radio one). |

OmniUV uses only:

```python
from .orbital import earth, moon, KeplerianElements as KE   # earth_orbit.py, lunar_orbit.py
from .orbital import earth, KeplerianElements as KE, earth_mu, moon_mu  # dro.py
```

### 5.12 Top-level Python: `tool.py` and `run_example.py`

- **`tool.py`** (264 lines) — independent re-implementation of the same
  helpers as `src/util.py` (`mas2rad` is even defined twice in tool.py at lines
  208 and 263). Provides `plot_uv`, `plot_beam`, `plot_image`, `plot_crs`,
  `plot_lcs`, `plot_el_sep`, `upsample`, `lonlat_deg2xyz`, `h2ae`,
  `calc_beam_param`, `rad2mas/min/deg`. `run_example.py` imports it via
  `from tool import *` (line 6).

- **`run_example.py`** (372 lines) — the *only* end-to-end usage example.
  Documented in detail in §7.

### 5.13 `simulators/omniuv/dro/` — DRO standalone demo

| File | Lines | Notes |
|---|---|---|
| `gen_dro_rot.py` | 103 | Demonstrates the rotating-frame DRO calculation for all 20 ICs from `data/ic_dro.txt`; uses the not-shipped `LunarFixed` class. The path on line 9 (`/home/liulei/program/VLBI`) is the original author's machine — the file is a recipe, not a runnable script. |
| `plot_dros.py` | 55 | 2-D plot of `stn.p_rot` (rotating-frame) trajectories. |
| `tool.py` | 264 | A duplicate of the top-level `tool.py`. |
| `fig/*.png` | 6 files | Pre-rendered: `dros.png` (rotating-frame), `dro05_RK45_step10s.png`, `dro05_RK45_step60s.png`, `dro05_RK45_step2s.png`, `dro05_RK23_step10s.png`, `dro05_DOP853_step10s.png` — illustrate integrator-method/step-size effects on the inertial-frame DRO trajectory. |

---

## 6. Core algorithms

### 6.1 Reference-frame chain

For Earth-fixed stations the conversion is **TRS → CRS** at every time
sample (`earth_fixed.py:22–31`):

```
M = PN · R · W
p_crs(t) = M(t) · p_trs       # einsum('ijk,k->ij', m_R, p_trs)
```

- `W` (Wobble): polar motion + TIO locator s′ — `util.calc_t2c_PN_W` lines
  216–228. `W = R3(-s′) · R2(xp) · R1(yp)`.
- `R` (Rotation): Earth Rotation Angle — `util.calc_t2c_R` lines 203–214.
- `PN` (Precession-Nutation): IAU 2006A from SOFA, identity if SOFA missing
  — lines 230–244.

### 6.2 Source uvw basis

Per-source "ruvw triple" computed once in `Task.set_srcs`:

```
s0  = (cos δ cos α, cos δ sin α, sin δ)         # CRS unit vector toward phase centre
n   = (0, 0, 1)
w   = s0
u   = n × w  (normalised)
v   = w × u
```

(`util.crs2rqu` + `util.rqu2ruvw`, lines 80–116.)

For each station the per-time station-`uvw` is `p_crs · (u, v, w)` (`Station.calc_uvw`,
`base.py:254-273`). The per-baseline `uvw` is the difference of two stations'
station-`uvw`, restricted to the time indices where both are available
(`Task.calc_uvw_per_bl`).

### 6.3 Visibility availability

A time index `t_i` is "available" for a station/source pair if **all** of the
following are true (`_el_sep.calc_idt_avail`, lines 81–114):

1. **Elevation** > `el_min` (only for `type='fixed'`).
2. **Source separation** from each requested obstructing body
   (`Earth/Moon/Sun`) > `sep_min[body]`. The separation formula
   (`_calc_sep`, lines 13–17):

   ```
   sep = π − arcsin(R_body / |dp|) − arccos(dp̂ · ŵ)
   ```

   where `dp` is the vector from the station to the body's centre.

For a baseline, the available indices are `intersect1d(idt_avail[i], idt_avail[j])`
(`Task.calc_uvw_per_bl`, line 133).

### 6.4 Vis simulation methods

**FFT path** (`_sim.gen_vis_fft`, lines 54–145):

1. Raster image pixels onto an `nc × nc` flux grid using `dra·cos(δ)`, `dδ`
   offsets (lines 70–78).
2. `vis_grid = fftshift(fft2(fftshift(arr)))` (line 85).
3. For each baseline, convert per-time per-freq `(u, v) [m]` to wavenumbers
   via `λ = c / f` (lines 122–123), gridify via `uv2id`, look up `vis_grid`.

**DFT path** (`_sim.gen_vis_direct`, lines 147–188):

```
for each baseline:
    lmn1   = src.img.lmn - (0, 0, 1)                       # (npixel, 3)
    lmn_uvw = einsum('hk,ijk->hij', lmn1, uvw_wav)         # (npixel, nt, nfreq)
    fringe = exp(-2j π lmn_uvw)                            # (npixel, nt, nfreq)
    vis    = einsum('h,hij->ij', fluxes, fringe)           # (nt, nfreq)
```

The `(0, 0, 1)` subtraction is the standard `(l, m, n − 1)` shift that pulls
the phase tracking origin to the phase centre.

### 6.5 DRO integration

In production (`do_dro_rot=False`), the equation of motion in the
Earth-centred inertial frame is (`dro.calc_crs.f`, lines 201–224):

```
acc(r⃗, t) = -μ_E · (r⃗ - r⃗_E(t)) / |r⃗ - r⃗_E(t)|³
            -μ_M · (r⃗ - r⃗_M(t)) / |r⃗ - r⃗_M(t)|³

with r⃗_E, r⃗_M from JPL DE421 (jplephem) at the actual observation time.
```

Integrated by `scipy.integrate.solve_ivp(f, [t0, t1], y0, t_eval=ts, **kw)`,
where `kw` is whatever the user passes to `set_param` (e.g. `method='RK45',
max_step=10.0`). Initial state from `gen_init` (lines 29–70).

In demo mode (`do_dro_rot=True`, `calc_rot`, lines 127–183) the equation is
solved in the **non-dimensional Moon-Earth rotating frame**:

```
acc = grav(m1, p_E_rot, r⃗) + grav(m2, p_M_rot, r⃗) − 2 ω×v⃗ − ω×(ω×r⃗)
ω   = (0, 0, 1)             # non-dimensional rate
m2  = 1.215059e-2
m1  = 1 - m2
p_E_rot = (-m2, 0, 0)
p_M_rot = (1-m2, 0, 0)
```

This produces the closed periodic family in `dro/fig/dros.png` only because
the Moon orbit is treated as circular; the realistic SPK-driven version
(`fig/dro05_RK45_step60s.png`) shows the trajectory does *not* close in
inertial coordinates (`README.MD` lines 178–187).

### 6.6 Noise model

Per-baseline thermal-noise standard deviation (`_sim.vis_add_noise`,
lines 205–207):

```
σ = √(SEFD_i · SEFD_j / (2 · BW · t_ap)) / η,    η = 0.88 (2-bit quant.)
```

Real and imaginary parts are independently Gaussian (achieved via
`np.abs(N(0,σ)) · exp(i · U(0, 2π))` in lines 209–211 — note the abs which
makes the magnitude half-Gaussian, then random phase).

Per-time *gain* errors (`Task.calc_gain`, `base.py:62-72`): Gaussian
amplitude `N(1, gain_error)` clipped to ≥ 0; phase set to zero (the uniform
phase line is commented out at line 70).

### 6.7 Imaging weights

`gen_weight` (`_sim.py:423-465`) supports three options:

| Flag | Meaning |
|---|---|
| (default, both off) | Natural weighting (w=1) |
| `task.do_unif = True` | Uniform: w /= (count of vis in same uv cell) |
| `task.do_rad = True` | Radial: w *= |uv|/1e6, then renormalised |

### 6.8 FFT cellsize / uv-range coupling

From `README.MD` lines 16–22 and `_sim.set_uv_image_param`:

```
uv_max  = 1 / cellsize_rad
duv     = uv_max / nc
nc·cellsize = field of view (radians)
```

The README explicitly warns: too-large `cellsize` ⇒ tiny uv range ⇒ all
samples may be excluded; too-small `cellsize` ⇒ huge `duv` ⇒ all samples may
collapse into the centre cell. The user is expected to size `cellsize` as a
fraction of the angular resolution `1/uv_max_lambda`.

---

## 7. Public API & usage

The package has **no CLI**. The user-facing API consists of these classes and
their methods.

### 7.1 `Source`

```python
class Source:
    name: str
    ra: float          # radians
    dec: float         # radians
    # Set by Task.set_srcs():
    ruvw: tuple[ndarray, ndarray, ndarray]   # (u, v, w) basis in CRS
    id: int
    # Set by user:
    img: Image         # for visibility simulation
```

### 7.2 `Image`

```python
img = Image(src)              # copies src.ra/dec/ruvw into ra0, dec0, ruvw0
img.fluxes: list[float]       # Jy
img.ras:    list[float]       # rad
img.decs:   list[float]       # rad
img.npixel: int
img.lmn:    ndarray (npixel, 3)   # MUST be filled by the user; see run_example.py:301-306
```

### 7.3 `Task`

Constructor: `Task()`. Everything via attribute assignment + setters.

| Method | Signature | Purpose |
|---|---|---|
| `set_srcs(srcs)` | `srcs: list[Source]` | Register phase centres; computes `ruvw` for each. |
| `set_eops(**kw)` | `tmu, dut1, xp, yp` | Set Earth Orientation Parameters. |
| `set_ts(t0, ts)` | `t0: datetime`, `ts: ndarray[float]` (seconds since `t0`) | Time sampling. |
| `set_stns(stns)` | `stns: list[Station]` | Register stations; lazy-loads SPK/PCK as needed. |
| `calc_uvw_bl()` | → `uvw[bl][src][t,3]` | Master uvw driver. Also populates `self.idt_avail`. |
| `flat_uvw_bl(uvw)` | helper to flatten across baselines | per-source flattened arrays. |
| `gen_vis_fft(src)` / `gen_vis_direct(src)` | → `bls: list[dict]` | Visibility simulation (FFT or DFT). |
| `vis_add_noise(src, bls)` | → updated `bls` | Add thermal + gain noise. |
| `gen_image_fft(bls)` / `gen_image_direct(bls)` | → `(uv, image)` | Dirty-image reconstruction. |
| `gen_beam(bls)` | → `beam: ndarray (nc, nc)` | Dirty beam. |
| `to_fitsidi(name, bls)` | writes `name.FITS` | FITS-IDI export. |

Required attributes before vis simulation:

```python
task.freqs:     ndarray[Hz]    # one entry per IF
task.bandwidth: float (Hz)
task.t_ap:      float (s)
task.cellsize:  float (mas)
task.nc:        int             # image side
# optional:
task.gain_error: float
task.do_unif, task.do_rad, task.do_beam_correction: bool
task.cs_src_rad: float (rad)    # if do_beam_correction
```

`bls` data structure (see comments in `run_example.py` lines 313–322):

```python
bl = {
  'uvw_m':   ndarray (nt_avail, 3),               # uvw in metres
  't':       ndarray (nt_avail,),                 # seconds since t0
  'uvw_wav': ndarray (nt_avail, n_freq, 3),       # uvw in wavelengths
  'vis':     ndarray (nt_avail, n_freq) complex,  # visibility
}
```

### 7.4 `Station` and subclasses

Common methods (defined on `Station`, augmented per subclass):

| Method | Description |
|---|---|
| `set_SEFD(jy)` | System Equivalent Flux Density |
| `set_sep_min_deg(**kw)` | e.g. `set_sep_min_deg(Earth=5.0, Moon=1.0, Sun=10.0)` |
| `set_el_min_deg(deg)` | only meaningful for `type='fixed'` |
| `set_orbit(a, e, i, raan, arg_pe, M0, t_ref)` | Keplerian elements (orbit subclasses only) |

Subclass-specific:

```python
g = EarthFixed('TMRT')
g.set_trs(np.array([-2826708.82869, 4679236.99691, 3274667.48709]))   # ECEF metres
g.set_el_min_deg(15.0)
g.set_SEFD(48.0)

s = EarthOrbit('t1')
s.set_sep_min_deg(Earth=5.0)
s.set_orbit(a=…, e=…, i=…, raan=…, arg_pe=…, M0=…, t_ref=datetime(...))
s.set_SEFD(225.0)

lo = LunarOrbit('lo')
lo.set_sep_min_deg(Moon=1.0, Sun=10.0)
lo.set_orbit(a=util.Rm*3, e=0.0, i=0.0, raan=0.0, arg_pe=0.0, M0=0.0)
lo.set_SEFD(507)

l2_es = EarthSunL2('es')
l2_me = MoonEarthL2('me')

# DRO (loaded directly from src/dro.py — not re-exported in __init__.py)
from omniuv.dro import DRO
s = DRO('S0')
s.set_param(orbit_id=5, method='RK45', max_step=10.0)
```

### 7.5 Aperture-array beam (`aperture_array.ApertureArray`)

```python
arr = ApertureArray(lam=0.3, pos_tiles=xy_array, tiles=[sub_tile_or_empty])
arr.calc_ws(th0, ph0, ths, phs)        # phased weight for steering (th0,ph0)
arr.get_ws(ths, phs)                   # interpolate at finer grid
plotxyproj(arr, 'name')                # tangent-plane projection plot
```

Used only in `aperture_array.test_array()` and `aperture_array.create_tile()`
demos — not connected to the Task workflow.

---

## 8. End-to-end example

The single canonical example is `simulators/omniuv/run_example.py`. Outline
(line numbers refer to the file):

1. `task = Task()` (line 47).
2. `task.set_srcs([...])` with one phase centre at `(180°, 30°)` (lines 50–75).
3. `task.set_eops(tmu=35., dut1=-2.211724, xp=0.027840, yp=0.370850)` (78–82).
4. Choose schedule via `gen_ts1/2/3` (lines 18–42; 60-s sampling for 1 day, or
   1 hr/day at days 0,7,14,21, etc.). `task.set_ts(t0, ts)` (line 95).
5. Build station list (lines 97–186):
   - 2× `EarthOrbit` at altitudes 10 000 km × 100 000 km, ±30° inclination
   - 2× `EarthFixed` (TMRT and Effelsberg ECEF positions)
   - 1× `LunarOrbit` at `a = 3·R_moon`, circular
   - 1× `MoonEarthL2`
   - Lagrange `EarthSunL2` defined but not appended in this run.
6. `task.gain_error = 0.1`; `task.set_stns(stns)` (188–192).
7. `uvw_bls = task.calc_uvw_bl()` (line 201).
8. Set `task.freqs = [8.4 GHz]`, `t_ap = 2 s`, `bandwidth = 32 MHz`,
   `cellsize = 0.005 mas`, `nc = 128` (lines 239–267).
9. Build the source `Image` with 5 pixels and compute `lmn` for each (lines
   270–306).
10. `bls = task.gen_vis_fft(src)`, `bls = task.vis_add_noise(src, bls)`, then
    `task.to_fitsidi('EXAMPLE', bls)` (lines 323–332).
11. `beam = task.gen_beam(bls)`, `uv, image = task.gen_image_fft(bls)`,
    `plot_beam`, `plot_uv`, `plot_image` (lines 352–369).

The script writes (in cwd):

| File | Source |
|---|---|
| `dump_gen_vis_fft.png` | `_sim.gen_vis_fft` line 81–82 (debug raster) |
| `EXAMPLE.FITS` | `_fits.to_fitsidi` |
| `example_beam.png`, `example_uv.png`, `example_image.png` | `tool.py` plot helpers |

---

## 9. Input & output formats

### 9.1 Inputs

| Input | Format | Where |
|---|---|---|
| Antenna positions (Earth-fixed) | ECEF metres `(x, y, z)`, plain `np.array` | `EarthFixed.set_trs(p_trs)` |
| Antenna orbit | Keplerian elements via `Station.set_orbit(a, e, i, raan, arg_pe, M0, t_ref)` | `base.py:217-244` |
| Source list | `Source.ra/dec` in radians | `Task.set_srcs` |
| Source image | `Image` with `fluxes`, `ras`, `decs`, `lmn` arrays | `run_example.py:283-306` |
| Schedule | `(t0: datetime, ts: ndarray[s])` | `Task.set_ts` |
| Frequencies | `task.freqs: ndarray[Hz]` | direct attribute |
| EOPs | `task.set_eops(tmu=…, dut1=…, xp=…, yp=…)` | `base.py:43-45` |
| DRO ICs | `data/ic_dro.txt`, columns `JC Period x0 y0_dot` | `dro.py:25` |
| DE421 ephemeris | `data/de421.bsp` (binary SPK) | `util.load_spk` |
| Moon PA orientation | `data/moon_pa_de421_1900-2050.bpc` | `util.load_pck` |
| SOFA library | `data/sofa/libsofa_c.so` (built via `make`) | `util.py:124-129` |

### 9.2 Outputs

| Output | Format |
|---|---|
| Visibility records | In-memory `bls` dicts (per baseline) |
| Beam | NumPy `(nc, nc)` `float` array |
| Dirty image | NumPy `(nc, nc)` `float` array |
| FITS-IDI | 6-extension `.FITS` (loadable by AIPS `fitld`/`uvsrt`/`indxr`) |
| Diagnostic plots | PNGs via matplotlib (no public Plotly/HTML route) |

There is no native HDF5, Measurement-Set or UVH5 output.

---

## 10. Testing layout

There is **no formal test suite**. The vendored repo contains:

- `data/sofa/t_sofa_c` (binary, the SOFA library's own test driver — see the
  SOFA `makefile` `make test` target).
- `run_example.py` doubles as a smoke test (running it end-to-end exercises
  every major code path: trajectory, uvw, FFT vis, noise, FITS-IDI, FFT
  imaging).
- `dro/fig/*.png` are pre-rendered comparison outputs that document
  integration-method/step-size sensitivity but are not regression tests.

There are no `pytest`/`unittest` files anywhere in the tree (verified by
`find … -name 'test_*.py' -o -name '*_test.py'`, returns nothing).

---

## 11. Integration & extension points

The package is designed for **subclassing**, not for a plugin registry:

1. **New station type**: subclass `Station` and implement
   `calc_crs(self) -> ndarray[(nt, 3)]` returning CRS positions in metres.
   Set `self.type` (`'fixed'` / `'orbit'` / new), `self.cb`. The base class
   then handles `calc_uvw`, `calc_idt_avail`, etc.

2. **New aperture-array beam**: subclass `ApertureArray` and override
   `get_ws(ths, phs)` (`aperture_array.py:130-141`).

3. **Custom imaging weights**: set `task.do_unif`, `task.do_rad`, or replace
   `gen_weight` on the instance.

4. **External SPK kernel**: replace `data/de421.bsp` and ensure the JD range
   passes `util.valid_jd` (`util.py:290-295`); update the bounds if needed.

5. **Custom EOPs**: pre-load from any source and pass the four scalars to
   `task.set_eops(...)`. The dormant `util.get_eop` (lines 190–201) shows the
   intended interpolation API but expects an EOP table object whose loader is
   not in the repo.

6. **GPU imaging**: uncomment `import cupy as cp` in `backend.py:4` and
   switch `_sim.gen_image_direct:323` from `direct_image_np` to
   `direct_image_cp`. The cupy worker uses `multiprocessing.Process` per
   GPU (default `ngpu=2`, line 42).

7. **FITS-IDI variants**: edit `_fits.gen_AG/SU/AN/FR/UV` for non-VLBA
   telescopes (the strings `'VLBA'`, `'DIFX'`, `'3C288'` are hard-coded —
   `_fits.py:64,66,162`).

---

## 12. Notable internals & gotchas

- **The `omniuv` import name comes from a symlink**: the repo root contains a
  symlink `omniuv -> src` (`ls -la` of the repo). This is the only mechanism
  by which `import omniuv` works without packaging metadata. If you copy the
  tree without preserving symlinks, `from omniuv import *` will fail.

- **`util.path_home = '.'`** means SPK/PCK/SOFA paths are *relative to the
  current working directory*. Run scripts from `simulators/omniuv/` or set
  `omniuv.util.path_home = '/abs/path/to/repo'` before any `Task` setup.

- **`util.rqu2ruvw` body is duplicated** (lines 89–101 and 103–116). Both
  blocks compute the same `(u, v, w)` triple. Functionally harmless.

- **`mas2rad` defined twice** in `tool.py` (lines 208 and 263). Last wins;
  same body.

- **Phase centre is the only source allowed in FITS-IDI export.** `_fits.gen_SU`
  hard-codes `srcs[0]` and the source name string `'3C288'` (line 162) —
  multiple sources require manual editing.

- **All-zero gain phases.** `Task.calc_gain` line 71 zeros the phases —
  the uniform-distribution variant is commented out. So gain noise modulates
  amplitude only.

- **NA (sentinel = 0.0).** The FFT vis path uses `0.0` as a "no-vis" marker
  (`_sim.NA = 0.0`, line 21). Real visibilities that happen to be exactly
  zero will be silently dropped in `gen_image_*`.

- **`type=='orbit'` for L2 stations.** `EarthSunL2.type = 'orbit'` and
  `MoonEarthL2.type = 'orbit'` (their files line 15), but they don't have a
  `KeplerianElements` orbit object. Calling `set_orbit(...)` on them would
  raise `AttributeError`. They derive position purely from the Sun/Moon
  ephemeris.

- **`MoonEarthL2.cb = ''`**: the central-body string is empty, which means
  `set_stns` will not auto-load PCK or do `Moon`-based separation by
  default. Use `set_sep_min_deg(Moon=1.0)` to opt in (as in
  `run_example.py:183`).

- **DRO ephemeris-driven mode requires** `task.spk` to be set; the SPK is
  lazy-loaded by `set_stns` whenever a `Lunar*`/`Moon`-related station is
  registered (`base.py:91-95`). For `DRO`, the lookup uses `self.task.spk`
  (`dro.py:34, 200`) rather than the module-level `util.spk` — so this
  works only after `Task.set_stns` has triggered `util.load_spk()`.

- **EarthFixed bypasses `self.task.spk`** — it never needs ephemerides.

- **Sample ordering in FITS-IDI is NOT time-sorted** — see §5.8 for the
  AIPS post-processing recipe.

- **Field of view limit.** The FFT path is documented (`README.MD` line 39)
  as suitable only for small fields because the *w*-term is dropped during
  the 2-D fft2. For wide fields use the DFT path (which builds the full
  `(l, m, n−1)` projection).

- **Thread/process model.** Single-threaded NumPy throughout, *except*
  `backend.direct_image_cp` which uses `multiprocessing.Process` (one per
  GPU). The CPU `direct_image_np` is single-process but vectorised via
  `np.einsum` — segments the outer-product memory by `s_mem_max` (default
  12 GB at the call site).

---

## 13. Known limitations & TODOs (per source comments and README)

- `LunarFixed` station is not shipped (`README.MD` line 154; commented out
  in `src/__init__.py:5`).
- `cupy` GPU path is dead code (`backend.py:4`).
- `task.cellsize` is always in **mas** even for wide-field arrays — no
  unit override (`run_example.py:262-264` comment).
- L2 halo motion is **not modelled** — both L2 station classes are static at
  the linear-CR3BP L2 point.
- `gen_image_fft_nowt` is leftover scaffolding (`_sim.py:584-646`); not
  wired to any caller.
- FITS-IDI is single-source (`_fits.gen_SU` exports only `srcs[0]`).
- SPK validity range is fixed by the bundled DE421 segment (`util.valid_jd`
  bounds correspond roughly to 1899-08 to 2053-10).
- No CLI, no config files, no test suite, no Python packaging.
- `dro/gen_dro_rot.py:9` hard-codes the original author's filesystem path.
- The vendored `orbital` package is at version 0.7.0; bug-fixes upstream
  must be re-applied manually.

---

## 14. Citation (`README.MD` lines 13–16)

> Lei Liu, Weimin Zheng, Jian Fu, Zhijun Xu, *"OmniUV: A Multi-Purpose
> Simulation Toolkit for VLBI Observation"*, **2022, AJ, 164, 67**,
> [arXiv:2201.03797](https://arxiv.org/abs/2201.03797).
>
> *"We require that you cite the above reference and the repo link
> (https://github.com/liulei/omniuv) in your paper."*

License of OmniUV itself: **GNU GPL v3** (`simulators/omniuv/LICENSE`).
The vendored `orbital` package is **MIT** (declared at `src/orbital/__init__.py:17`).

