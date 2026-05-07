# uvplot — Exhaustive Technical Reference

> Independent technical reference for the **uvplot** package vendored as a git submodule under
> `simulators/uvplot/` inside the RRIVis repository. Every claim in this document is sourced from
> the files cited inline (paths are relative to `/Users/kartikmandar/RRIVis/`).

---

## 1. Overview

**uvplot** is a small Python utility for handling and plotting interferometric visibilities in the
`(u, v)` Fourier plane. It performs three concrete jobs:

1. **Read / write** "uvtables" — flat ASCII or NumPy `.npz` tables of visibility samples.
2. **Manipulate** visibilities geometrically: phase-shift, rotation, inclination deprojection,
   `uv`-distance cut.
3. **Plot** azimuthally-binned, weighted real and imaginary visibility profiles versus deprojected
   `uv`-distance — the "uvplot" panel that gives the package its name.

It additionally ships an `export_uvtable(...)` helper that runs **inside CASA** and converts a
Measurement Set (MS) into the flat ASCII format used by downstream Fourier-plane fitting tools
such as **Galario**.

| Field | Value | Source |
|-------|-------|--------|
| Package name | `uvplot` | `simulators/uvplot/setup.cfg` line 4 |
| Version | `0.2.11` | `simulators/uvplot/uvplot/__init__.py` line 1 |
| License | LGPLv3 | `simulators/uvplot/setup.cfg` line 11; `simulators/uvplot/LICENSE.txt` |
| Author | Marco Tazzari (Univ. of Cambridge) | `simulators/uvplot/AUTHORS.rst` |
| Contributors | Patrick Cronin-Coltsmann, Grant Kennedy | `simulators/uvplot/AUTHORS.rst` |
| Languages | Pure Python 3 | `simulators/uvplot/uvplot/*.py` |
| Python support | 3.6 – 3.9 (Py 2.7 dropped) | `setup.cfg`; commit `ad2eee6` "[setup] Drop support for Python 2.7" |
| Repository | https://github.com/mtazzari/uvplot | `setup.cfg` line 14 |
| DOI | 10.5281/zenodo.1003113 | `simulators/uvplot/README.md` line 50 |
| Lines of code | 1198 across 7 Python files | `wc -l` over `simulators/uvplot/uvplot/*.py` |

The README explicitly positions uvplot as a companion to **Galario** for visibility-domain fitting:
> "uvplot also makes it easy to export visibilities from MeasurementSets to uvtables, a handy
> format for fitting the data (e.g., using Galario)." — `simulators/uvplot/README.md` lines 12-15.

---

## 2. Repository Layout

```
simulators/uvplot/
├── .git                                  # gitlink (submodule pointer)
├── .github/workflows/
│   ├── tests.yml                         # CI: pytest matrix py3.6-3.9
│   ├── publish-to-pypi.yml               # release on tags
│   └── publish-to-test-pypi.yml          # TestPyPI uploads
├── .gitignore
├── .readthedocs.yaml                     # RTD build config (Python 3.7, sphinx-rtd-theme)
├── AUTHORS.rst                           # author + contributors list
├── LICENSE.txt                           # GNU LGPL v3
├── MANIFEST.in                           # 'include LICENSE.txt'
├── Makefile                              # legacy `pypi_update` target
├── README.md                             # PyPI long description
├── pyproject.toml                        # PEP 517 build-system shim
├── setup.cfg                             # all packaging metadata
├── setup.py                              # one-liner that calls setup()
├── docs/
│   ├── conf.py                           # Sphinx config (sphinx-rtd-theme + autodoc + napoleon)
│   ├── index.rst                         # documentation home
│   ├── install.rst                       # install instructions
│   ├── basic_usage.rst                   # plotting + MS export tutorial
│   ├── uvtable.rst                       # autodoc page for UVTable
│   ├── io.rst                            # autodoc page for export_uvtable
│   ├── license.rst
│   ├── requirements.txt                  # sphinx-rtd-theme, sphinx-copybutton
│   └── images/uvplot.png                 # sample figure
└── uvplot/                               # the package itself
    ├── __init__.py                       # public re-exports + Mac TkAgg shim
    ├── _set_unique_version.py            # CI helper for TestPyPI version bumps
    ├── constants.py                      # arcsec, clight numeric constants
    ├── example.py                        # standalone smoke-test script
    ├── io.py                             # export_uvtable() — CASA-only MS dumper
    ├── tests.py                          # pytest suite (init / deproject / uvcut)
    └── uvtable.py                        # the UVTable class (790 LOC, the package's core)
```

(Listing reconciled from `find simulators/uvplot -type f -not -path '*/\.git/*'`.)

There is **no `CHANGELOG.rst` or `CITATION.cff`** in the tree — the changelog is delegated to
GitHub Releases (see `simulators/uvplot/docs/index.rst` line 68) and the canonical citation is the
BibTeX block embedded in `simulators/uvplot/README.md` lines 44-52.

---

## 3. Installation & Dependencies

### 3.1 Install

`pip install uvplot` works in both a regular Python environment and inside CASA 6.x
(`simulators/uvplot/docs/install.rst` lines 14-31). Windows is not supported (line 7). CASA 5.x and
earlier are not supported because they cannot install via pip (line 33).

### 3.2 Build system

`simulators/uvplot/pyproject.toml` declares a minimal PEP 517 setup:

```toml
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"
```

All metadata is in `setup.cfg` (using `setuptools` declarative configuration). `setup.py` is a
3-line shim that simply calls `setup()`.

### 3.3 Runtime dependencies

From `simulators/uvplot/setup.cfg` lines 40-42:

```
install_requires =
    numpy>=1.9
    matplotlib
```

That is the entire runtime dependency surface. There is no `numpy<X` upper bound, no SciPy
dependency, no astropy dependency. The CASA `tb`/`split` task objects required by
`export_uvtable()` are **not** declared; they are obtained at runtime from the host CASA shell
(see Section 7).

### 3.4 Development / docs

`simulators/uvplot/docs/requirements.txt`:

```
sphinx-rtd-theme>=0.5.1
sphinx-copybutton>=0.3.1
```

CI (`simulators/uvplot/.github/workflows/tests.yml`) additionally installs `pytest sphinx` for the
test job and runs `py.test uvplot/tests.py` on Python 3.6, 3.7, 3.8, 3.9 on `ubuntu-20.04`.

---

## 4. Public API

`simulators/uvplot/uvplot/__init__.py` defines the public surface:

```python
__version__ = '0.2.11'
...
from .uvtable import UVTable, COLUMNS_V0, COLUMNS_V1, COLUMNS_V2
from .io import export_uvtable
from .constants import arcsec
```

So `from uvplot import ...` exposes exactly six names:

| Name | Kind | Defined in | Purpose |
|------|------|-----------|---------|
| `UVTable` | class | `uvplot/uvtable.py` | Visibility container, geometry ops, plotter |
| `COLUMNS_V0` | `list[str]` | `uvplot/uvtable.py` line 18 | `['u','v','Re','Im','weights']` |
| `COLUMNS_V1` | `list[str]` | `uvplot/uvtable.py` line 19 | `['u','v','Re','Im','weights','freqs','spws']` |
| `COLUMNS_V2` | `list[str]` | `uvplot/uvtable.py` line 20 | `['u','v','V','weights','freqs','spws']` (complex `V`) |
| `export_uvtable` | function | `uvplot/io.py` line 14 | CASA MS → ASCII uvtable |
| `arcsec` | float | `uvplot/constants.py` line 7 | `4.84813681109536e-06` rad/arcsec |

Note also a Mac-only side effect at import time: if matplotlib's backend is `'macosx'` it is
forced to `'TkAgg'` to avoid the "Python is not installed as a framework" runtime error
(`__init__.py` lines 16-17).

`uvtable.py` declares `__all__ = ["UVTable"]` (line 16); `io.py` declares
`__all__ = ["export_uvtable"]` (line 11).

---

## 5. Module Walkthrough

### 5.1 `uvplot/constants.py`

Two literal constants used throughout the package (`simulators/uvplot/uvplot/constants.py`):

```python
arcsec = 4.84813681109536e-06   # radians  (== pi/(180*3600))
clight = 2.99792458e+8          # m/s
```

Only `arcsec` is re-exported; `clight` is used internally by `uvtable.UVTable.{u_m, v_m}` and by
`io.export_uvtable` to compute the mean wavelength.

### 5.2 `uvplot/__init__.py`

- Sets `__version__ = '0.2.11'`.
- Conditionally swaps the matplotlib backend `macosx` → `TkAgg` (lines 16-17). This **runs at
  import time** of the package — a behaviour worth knowing if you embed uvplot inside another
  application that has its own backend policy.
- Re-exports the public API (Section 4).

### 5.3 `uvplot/_set_unique_version.py`

A CI-only helper that mutates the first line of `uvplot/__init__.py` to append the current Unix
timestamp to the semantic version (e.g. `'0.2.10'` → `'0.2.10.1234567890'`). Used by the
TestPyPI workflow because PyPI/TestPyPI refuse re-uploads of an already-published version. PEP 440
forbids hashes in version numbers, hence the timestamp choice (lines 6-12).

### 5.4 `uvplot/example.py`

Standalone smoke-test script demonstrating the canonical pipeline (verbatim from
`simulators/uvplot/uvplot/example.py`):

```python
wle = 0.88e-3                 # observing wavelength [m]
dRA = 0.3 * arcsec            # rad
dDec = 0.07 * arcsec          # rad
inc = np.radians(73.)
PA  = np.radians(59)
uvbin_size = 30e3             # in units of wavelength

uv = UVTable(filename='uvtable.txt', wle=wle)
uv.apply_phase(dRA, dDec)
uv.deproject(inc, PA)

uv_mod = UVTable(filename='uvtable_mod.txt', wle=wle)
uv_mod.apply_phase(dRA=dRA, dDec=dDec)
uv_mod.deproject(inc=inc, PA=PA)

axes = uv.plot(label='Data', uvbin_size=uvbin_size)
uv_mod.plot(label='Model', uvbin_size=uvbin_size, axes=axes,
            yerr=False, linestyle='-', color='r')
axes[0].figure.savefig("uvplot.png")
```

Note: the script omits `columns=COLUMNS_V0`, which means it relies on the format being detected
from the ASCII header line. The `basic_usage.rst` doc explicitly recommends passing `columns=`
from v0.2.6 onwards.

### 5.5 `uvplot/tests.py`

Three pytest functions, all using a 10000-sample synthetic uvtable created by
`create_sample_uvtable()` (Gaussian `(u,v,re,im)`, uniform weights `w∈[0,1e4]`). Source:
`simulators/uvplot/uvplot/tests.py`.

| Test | What it asserts |
|------|-----------------|
| `test_init_uvtable` | `UVTable` reads from file and from tuple, with and without `wle`, and `uvdist == hypot(u/wle, v/wle)` to ~1e-16. |
| `test_deproject` | `deproject(inc, inplace=False).u == u * cos(inc)` and `inplace=True` mutates in-place. |
| `test_uvcut` | Returned `UVTable` retains only baselines with `uvdist <= maxuv` and preserves the user-supplied `header` dict. |

### 5.6 `uvplot/uvtable.py`

The package's heart — 790 lines, one class. Documented in detail in Section 6.

### 5.7 `uvplot/io.py`

`export_uvtable(...)` — the CASA-side MS dumper. Documented in Section 7.

---

## 6. `UVTable` — the central class

Source: `simulators/uvplot/uvplot/uvtable.py` (the only class in the package, `__all__ = ["UVTable"]`).

### 6.1 Column formats

uvplot supports three on-disk schemas, declared at module top
(`simulators/uvplot/uvplot/uvtable.py` lines 18-26):

| Constant | Columns | Units (parallel array `UNITS_V*`) |
|----------|---------|-----------------------------------|
| `COLUMNS_V0` | `['u', 'v', 'Re', 'Im', 'weights']` | `lambda, lambda, Jy, Jy, None` |
| `COLUMNS_V1` | `['u', 'v', 'Re', 'Im', 'weights', 'freqs', 'spws']` | `lambda, lambda, Jy, Jy, None, Hz, None` |
| `COLUMNS_V2` | `['u', 'v', 'V', 'weights', 'freqs', 'spws']` (complex `V`) | `lambda, lambda, Jy, None, Hz, None` |

`COLUMNS_FORMATS = [COLUMNS_V0, COLUMNS_V1, COLUMNS_V2]` enumerates the valid set
(`uvtable.py` line 21). A `COLUMNS_FORMATS_TEXT` constant is built for human-readable error
messages (lines 33-39). Note an internal inconsistency: `save_ascii_uvtable` writes `columns=COLUMNS_V1` but `units=UNITS_V2` (line 401), and `save_binary_uvtable` writes
`columns=COLUMNS_V2` but `units=UNITS_V1` (line 418). These are header-only fields; the actual
on-disk column ordering is whatever the explicit `np.column_stack`/`np.savez` argument list
dictates.

### 6.2 Constructor

```python
UVTable(uvtable=None, filename=None, format='ascii',
        columns=None, wle=1., **kwargs)
```

(`uvtable.py` line 75.)

| Param | Type | Description |
|-------|------|-------------|
| `uvtable` | tuple/list of 1-D arrays or 2-D array | Visibility data already in memory. |
| `filename` | str | Path to ASCII or `.npz` uvtable. |
| `format` | `'ascii'` or `'binary'` | Selects the reader (case-insensitive). |
| `columns` | one of `COLUMNS_V0/V1/V2` | Column schema. If omitted with an ASCII file, uvplot tries to detect from line 1 (JSON header) then line 2 (`# Columns: ...`). |
| `wle` | float | Observing wavelength used to convert input `(u,v)` into wavelength units. Stored coordinates are always in `wle`. |
| `header` (kwarg) | dict | Free-form metadata persisted on save. |

Validation: passing both `filename` and `uvtable` raises `ValueError("Cannot provide both
filename and uvtable.")`; passing neither raises `ValueError("Provide at least a filename or a
uvtable.")` (lines 77-81).

After loading the constructor records `self.ndat = len(self.u)`, initialises `_uvdist` and
`bin_uvdist` to `None`, and — if frequencies are present — caches `_freqs_avg = freqs.mean()` and
`_freqs_wrt_avg = freqs / freqs_avg` (lines 107-113).

### 6.3 Properties

All physically meaningful arrays are exposed as numpy-`ascontiguousarray`-protected
properties (so re-assignment always normalises memory layout):

| Property | Setter | Units | Definition |
|----------|--------|-------|------------|
| `u`, `v` | yes | wavelengths (`wle`) | Coordinates as stored. Setter wraps with `np.ascontiguousarray`. |
| `u_m`, `v_m` | no | metres | `self.u / self.freqs * clight` (line 147 / 156). Requires `freqs` to be populated. |
| `re`, `im` | yes | Jy | Real/imag visibility components. |
| `weights` | yes | `1/Jy^2` | Per-sample weight (matches CASA convention). |
| `freqs` | yes | Hz | Per-sample frequency. |
| `freqs_avg` | no | Hz | `freqs.mean()` cached at init. |
| `freqs_wrt_avg` | no | dimensionless | `freqs / freqs_avg`. |
| `spws` | yes | int | Spectral-window index. |
| `V` | no | Jy | `re + 1j*im` complex view. |
| `uvdist` | no | wavelengths | `np.hypot(u, v)` recomputed on each access (line 255). |

(All defined in `uvtable.py` lines 115-257.)

### 6.4 I/O methods

#### `import_uvtable(uvtable, columns)` — lines 259-294

Internal entry point used by both the in-memory and on-disk constructors. Branches on
`columns ∈ {V0, V1, V2}`, copies into `_u/_v/_re/_im/_weights/_freqs/_spws`, then divides `u` and
`v` by `wle`. For `COLUMNS_V2` it splits the complex `V` into `.real` and `.imag` (lines 278-279).

#### `read_ascii_uvtable(filename, columns=None, **kwargs)` — lines 296-359

Uses `np.loadtxt(filename, unpack=True)` to load. If `columns` is not given:

1. Try to parse line 1 as a JSON object whose `"columns"` key holds the schema.
2. Fall back to line 2 of the form `# Columns:\t<space-separated-names>`.
3. Last resort: assume `COLUMNS_V0` and warn.

If the detected `columns` is not in `COLUMNS_FORMATS`, an `AssertionError` is raised. Number of
detected columns must match `Ncols` from `np.loadtxt`. Lines 305-357.

#### `read_binary_uvtable(filename, columns=None, **kwargs)` — lines 361-395

Uses `np.load(filename, allow_pickle=True)` and reads named arrays from the resulting NpzFile:
`u`, `v`, `Re`/`Im` or `V`, `weights`, optionally `freqs`/`spws`, and `header`. Header is
restored via `loaded['header'].item()`.

#### `save_ascii_uvtable(filename, ascii_fmt='%10.6e')` — lines 397-412

Writes a tab-separated ASCII file via `np.savetxt(... fmt=ascii_fmt, delimiter='\t',
header=json.dumps(self.header))`. The header JSON is updated in place to record
`columns=COLUMNS_V1, units=UNITS_V2` (note inconsistency mentioned earlier). Always writes the
seven columns `[u, v, V.real, V.imag, weights, freqs, spws]` regardless of the original schema —
so V0 inputs get up-promoted on save and you must have `freqs`/`spws` populated.

#### `save_binary_uvtable(filename, compressed=True)` — lines 414-428

Calls `np.savez_compressed` (or `np.savez`) with `u, v, V, weights, freqs, spws, header`. Header
written via `np.savez(... header=self.header)`.

#### `save(filename, export_fmt, **kwargs)` — lines 430-447

Thin dispatcher: `export_fmt.upper() in {'ASCII', 'BINARY'}` routes to the corresponding writer
above.

### 6.5 Geometry and binning

#### `apply_phase(dRA=0, dDec=0)` — lines 520-545

Implements the Fourier-domain shift theorem. Multiplies each visibility by `exp(2πi(u·dRA +
v·dDec))`:

```
phi = u*(2π·dRA) + v*(2π·dDec)
V  ← V * (cos(phi) + i sin(phi))
```

Both offsets are in radians; positive `dRA` translates the image East, positive `dDec` North.
Short-circuits to a no-op if both deltas are zero.

#### `rotate(x, y, theta)` (`@staticmethod`) — lines 547-577

Standard counter-clockwise rotation by `theta` (radians):

```
x_r =  x cos θ − y sin θ
y_r =  x sin θ + y cos θ
```

#### `deproject(inc, PA=0, inplace=True)` — lines 579-628

Two-step geometry transform:

1. Rotate `(u, v)` by `+PA` (the docstring on lines 603-605 explicitly notes that because
   right-ascension is a reversed x-axis, "the anti-rotation of a reversed PA Angle is the same
   as a direct rotation").
2. Multiply `u` by `cos(inc)`. The maths for an inclined disk in image-space is "divide by
   cos", but in the Fourier dual we **multiply** by `cos(inc)` (comment on lines 608-610).

When `inplace=False`, a fresh `UVTable` is returned with the original `re/im` (or `V`),
`weights`, `freqs`, `spws`, and `header`, but with the deprojected `u`/`v`. Returns `None` when
`inplace=True`.

#### `uvcut(maxuv, verbose=False)` — lines 630-664

Boolean filter `uvdist <= maxuv`, returning a new `UVTable` of the matching subset (preserves
the original `columns` schema and `header`). Verbose mode prints "Consider only baselines up to
{maxuv/1e3} klambda ({n} out of {ndat} uv-points)".

#### `uvbin(uvbin_size, **kwargs)` — lines 449-485

The package's azimuthal-averaging routine. Algorithm:

1. Number of bins: `nbins = ceil(uvdist.max() / uvbin_size)`.
2. Bin edges: `uv_bin_edges[i] = i * uvbin_size`, for `i = 0 ... nbins`.
3. For each bin `i`, collect indices with `uv_bin_edges[i] ≤ uvdist < uv_bin_edges[i+1]`:
   - `bin_count[i]` = number of samples,
   - `bin_uvdist[i]` = arithmetic mean `uvdist` inside the bin (or the bin centre when empty),
   - `bin_weights[i]` = sum of weights in the bin.
4. Calls `bin_quantity(re)` and `bin_quantity(im)` to obtain `bin_re/bin_im` with their errors.

The intervals (an array of `np.where` tuples) are kept in `self.uv_intervals` for re-use by
`bin_quantity`. The docstring's "weight_corr" comment refers to the fact that the radial weight
correction cancels out of the weighted-mean numerator-and-denominator — see line 460.

#### `bin_quantity(x, use_std=False)` — lines 487-518

Computes the inverse-variance weighted mean per bin:

```
bin_x[i]   = Σ_k x_k · w_k  /  Σ_k w_k
bin_x_err[i] = std(x in bin)            if use_std=True
             = 1 / sqrt(Σ_k w_k)        otherwise (the "natural" variance estimator
                                        for inverse-variance weights w_k = 1/σ_k²).
```

When `bin_count[i] == 0` both `bin_x[i]` and `bin_x_err[i]` are left at zero (vector pre-allocated
with `np.zeros`).

### 6.6 Plotting — `plot(...)`

(`uvtable.py` lines 666-790.) Signature:

```python
plot(fig_filename=None, color='k', linestyle='.', label='',
     fontsize=18, linewidth=2.5, alpha=1., yerr=True, caption=None,
     axes=None, uvbin_size=0, vis_filename=None, verbose=True)
```

Behaviour:

- If `axes is None`, creates a `(6,6)` figure with a 2-row `GridSpec(2, 1, height_ratios=[4, 1])`
  — top axis for `Re(V)`, bottom for `Im(V)`.
- If `uvbin_size != 0`, calls `self.uvbin(uvbin_size)` first; otherwise raises
  `AttributeError("Expected uv table with already binned data, or an input parameter
  uvbin_size != 0")` if `bin_uvdist is None`.
- Plots `bin_uvdist / 1e3` (kλ) vs `bin_re/bin_im` with `errorbar(...)`. Uses
  `mask = self.bin_count != 0` to skip empty bins (lines 727-728).
- `caption` must be a dict with keys `{'x', 'y', 'text', 'fontsize'}`; missing keys raise
  `KeyError`.
- If `vis_filename` is given, writes the binned data as `np.savetxt`-tab-separated:
  `uv-distance(klambda)\tRe(V)\te_Re(V)\tIm(V)\te_Im(V)`.
- If `fig_filename` is given, saves via `plt.savefig`; otherwise returns the `(ax_Re, ax_Im)`
  tuple so the caller can overplot models.

This dual-mode (return-axes-or-save) is what lets `example.py` overlay a model on the same panel:

```python
axes = uv.plot(label='Data', uvbin_size=...)
uv_mod.plot(label='Model', axes=axes, ...)   # second call reuses axes
```

---

## 7. `export_uvtable(...)` — MS to ASCII inside CASA

Source: `simulators/uvplot/uvplot/io.py` (235 lines). Single public function; CASA-only.

### 7.1 Signature

```python
export_uvtable(uvtable_filename, tb, vis="", split_args=None, split=None,
               channel='all', dualpol=True, fmt='%10.6e',
               datacolumn="CORRECTED_DATA", keep_tmp_ms=False, verbose=True)
```

`tb` and (optionally) `split` must be the **CASA-shell objects** of the same name —
`export_uvtable` cannot run outside a CASA Python interpreter (`simulators/uvplot/uvplot/io.py`
lines 30-34, 86-99). The only supported output format is ASCII.

### 7.2 Algorithm

Step-by-step (line numbers cited in `io.py`):

1. **Optional split** (lines 106-132): if `split_args` is provided, also `split` must be passed;
   otherwise raises `RuntimeError`. Adds `outputvis='mstable_tmp.ms'` if missing, runs
   `split(**split_args)`, then forces `datacolumn = 'DATA'` because the freshly split MS only has
   the DATA column.
2. **Open the MS** (line 142): `tb.open(MStb_name)`.
3. **Read coordinates** (lines 145-147): `uvw = tb.getcol("UVW")` → split into `u, v, w`.
4. **Read weights** (line 149): `weights_orig = tb.getcol("WEIGHT")` (shape `(npol, nrow)`).
5. **Read visibilities** (lines 152-157): `data = tb.getcol(datacolumn)` (shape `(npol, nchan,
   nrow)`). Raises `KeyError` if the requested column is absent.
6. **Spectral-window count check** (lines 159-167): if multiple spws are present and the user did
   not narrow them down via `split_args['spw']`, prints a warning that *all* will be exported.
7. **Channel selection** (lines 170-183):
   - `channel='first'` → take channel index 0 only; `nchan=1`.
   - `channel='all'`  → take all channels; `nchan = data.shape[1]`; tile `u` and `v` by `nchan`
     so the output uvtable has `nchan*nrow` rows.
   - Anything else raises `ValueError`.
8. **Polarisation handling** (lines 185-204):
   - `dualpol=True` (default): treat `data[0]` as XX and `data[1]` as YY. Compute the
     **inverse-variance-weighted Stokes-I-equivalent** visibility:

     ```
     V        = (V_XX · w_XX  +  V_YY · w_YY) / (w_XX + w_YY)
     weights  = w_XX + w_YY
     ```

     Weights are tiled across `nchan` if necessary.
   - `dualpol=False`: take only `data[0]` and the unmodified `weights_orig`.
9. **Lookup the SPECTRAL_WINDOW table** (lines 206-213): grab the keyword `SPECTRAL_WINDOW` from
   the main table to find its on-disk path, reopen `tb` on it, read `CHAN_FREQ`, close.
10. **Compute mean wavelength** (line 214): `wle = clight / freqs.mean()` (m).
11. **Write ASCII** (lines 220-227):

    ```
    np.savetxt(uvtable_filename,
               np.column_stack([u, v, V.real, V.imag, weights]),
               fmt=fmt, delimiter='\t',
               header=f"Extracted from {MStb_name}.\n"
                      f"wavelength[m] = {wle}\n"
                      f"Columns:\tu[m]\tv[m]\tRe(V)[Jy]\tIm(V)[Jy]\tweight")
    ```

    Crucially the on-disk `(u, v)` are in **metres**, not wavelengths — the consumer is expected
    to instantiate `UVTable(filename=..., wle=wle)` and let the constructor convert. The columns
    schema is `COLUMNS_V0`.
12. **Cleanup** (lines 229-235): if `split_args` was used and `keep_tmp_ms=False`, the temporary
    MS is removed via `subprocess.call("rm -rf <outputvis>", shell=True)`.

### 7.3 Limitations called out in the docstring

- All spws must have **equal** number of channels, otherwise CASA's `getcol` raises an
  `ArrayColumn` "array shapes vary" error (`io.py` lines 44-48 and 73-77 in the docstring).
- The function strictly assumes either 1 polarisation (`dualpol=False`) or exactly 2
  (`dualpol=True`). Full Stokes (`RR/LL/RL/LR` or `XX/XY/YX/YY`) is not handled.
- It does **not** handle flagged data on its own — the user is told to run a CASA `split` with
  `keepflags=False` first (`docs/basic_usage.rst` line 81; `io.py` lines 70-72).

### 7.4 Typical call signatures

From the docstring (`io.py` lines 22-26 and 84-95):

```python
# all visibilities, no split
export_uvtable('uvtable.txt', tb, vis='sample.ms', channel='all')

# select spws via split
export_uvtable('uvtable.txt', tb, channel='all', split=split,
               split_args={'vis': 'sample.ms',
                           'datacolumn': 'DATA', 'spw': '0,2'})

# non-interactive driver
casa --nologger --nogui -c \
  "from uvplot import export_uvtable; export_uvtable(...)"
```

---

## 8. Core Algorithms — Mathematical Summary

### 8.1 Coordinate units

`(u, v)` are stored *internally* in **wavelengths** (i.e. cycles per radian on the sky). The
constructor divides whatever the user supplied by `wle` (`uvtable.py` lines 293-294). Setting
`wle=1` is therefore the way to declare "my inputs are already in wavelengths".

### 8.2 Phase shift

Following the standard Fourier shift theorem, an image translation by `(dRA, dDec)` corresponds
to multiplying every visibility by `exp(2πi(u·dRA + v·dDec))`, with `(u, v)` in wavelengths and
`(dRA, dDec)` in radians. Implemented exactly in `apply_phase` (line 538-545).

### 8.3 Inclination deprojection

For an inclined disk with inclination `i` and position angle `PA` (East of North), the
azimuthally-symmetric component lives along the major axis. Rotating the `(u, v)` plane by `+PA`
aligns the major axis with the new `v`-axis; then squashing the orthogonal direction by `cos(i)`
in the **Fourier** domain (= multiplying `u_rot` by `cos i`) is equivalent to deprojecting the
image's minor axis to a circle. See `deproject()` lines 601-611.

### 8.4 Radial binning and weighted average

For a bin `B_i` on `uvdist`:

```
bin_uvdist[i]  =  ⟨|uv|⟩_{k∈B_i}                     (unweighted mean of uvdist)
bin_weights[i] =  Σ_{k∈B_i} w_k
bin_re[i]      =  Σ_{k∈B_i} Re(V)_k · w_k  /  bin_weights[i]
bin_im[i]      =  Σ_{k∈B_i} Im(V)_k · w_k  /  bin_weights[i]
σ_bin_x[i]     =  std(x_k)                           if use_std=True
                =  1/√(Σ_k w_k)                      otherwise   (assumes w_k = 1/σ_k²)
```

This is the standard inverse-variance estimator for visibility weights as defined by CASA.
Empty bins are masked out in `plot()` via `bin_count != 0`.

### 8.5 Dual-pol combination

Inside `export_uvtable` the dual-polarisation merge is also an inverse-variance weighted average
(io.py line 196):

```
V       = (V_XX · w_XX + V_YY · w_YY) / (w_XX + w_YY)
weights = w_XX + w_YY
```

For uncorrelated XX/YY this is the Stokes-I estimator with the lowest variance.

---

## 9. Input / Output Formats

### 9.1 ASCII uvtable

A tab-separated text file. uvplot recognises three header conventions for column detection
(`uvtable.py::read_ascii_uvtable`):

1. **JSON header on line 1**: `{"columns": ["u","v","Re","Im","weights"], ...}` — produced by
   `save_ascii_uvtable` itself.
2. **`# Columns:\t...` on line 2**: human-friendly form documented in `basic_usage.rst`:

   ```
   # Columns:	u v Re Im weights
   ```

3. **No header**: assumes `COLUMNS_V0`, prints a warning.

A canonical example produced by `export_uvtable` (`docs/basic_usage.rst` lines 87-98):

```
# Extracted from mstable.ms.
# wavelength[m] = 0.00132940778422
# Columns:	u[m]	v[m]	Re(V)[Jy]	Im(V)[Jy]	weight
-2.063619e+02	2.927104e+02	-1.453431e-02	-1.590934e-02	2.326950e+04
3.607948e+02	6.620900e+01	-1.680727e-02	1.124862e-02	3.624442e+04
...
```

`(u, v)` here are in **metres**; the consumer must read with `wle=0.00132940778422` so that the
`UVTable` divides them into wavelength units.

### 9.2 Binary uvtable (`.npz`)

`np.savez_compressed` with named arrays `u, v, V (or Re/Im), weights, freqs, spws, header`.
Header is a Python dict pickled by NumPy via `allow_pickle=True`. Read back via
`np.load(filename, allow_pickle=True)`.

### 9.3 CASA Measurement Set

Read-only on the export side — uvplot never *writes* a Measurement Set. The MS reader pulls
columns:

| MS column | uvplot use |
|-----------|-----------|
| `UVW` (3, nrow) | `u, v, w` baselines in metres |
| `WEIGHT` (npol, nrow) | per-polarisation per-baseline weights |
| `DATA_DESC_ID` | spectral-window index |
| `DATA` / `CORRECTED_DATA` / `MODEL_DATA` (npol, nchan, nrow) | visibilities |
| `SPECTRAL_WINDOW::CHAN_FREQ` (subtable) | per-channel frequencies (Hz) |

---

## 10. Examples & Tutorials

### 10.1 Plotting binned visibilities (from `example.py`)

The shipped script (Section 5.4) is the canonical recipe:

1. Build a `UVTable` from an ASCII file with a known wavelength,
2. Apply phase center offset,
3. Deproject by inclination + PA,
4. Plot data and an overlay model on the same axes,
5. Save the figure to PNG.

### 10.2 Exporting from a CASA shell (from `docs/basic_usage.rst`)

```python
CASA <1>: from uvplot import export_uvtable
CASA <2>: export_uvtable("uvtable.txt", tb, vis='mstable.ms')
```

For non-interactive batch use:

```bash
casa --nologger --nogui -c \
  "from uvplot import export_uvtable; export_uvtable('uvtable.txt', tb, vis='sample.ms')"
```

(Note the docstring's emphatic warning to use single-quoted strings in this context, otherwise
the outer double-quotes break.)

### 10.3 Pre-export hygiene

The docs strongly recommend `split(... keepflags=False)` before exporting to avoid carrying
flagged samples into the ASCII file (`docs/basic_usage.rst` line 81).

---

## 11. Integration with Galario and emcee

uvplot is sold as the **producer** half of the visibility-fitting workflow:

- The README (line 12-15) and `docs/index.rst` line 14-16 explicitly cite Galario as the
  intended consumer of uvtables.
- The `(u, v, weights, V)` schema in `COLUMNS_V0` is exactly what `galario.double.chi2Image` /
  `galario.double.sampleImage` expect: `u, v` in metres or wavelengths plus complex visibility
  and weight arrays.
- `apply_phase` and `deproject` are the standard pre-processing steps that must run **before**
  feeding visibilities to a Galario likelihood — Galario's internal forward model assumes the
  source is centred and face-on, so the user has to undo any RA/Dec offset and inclination in the
  data side.
- emcee (or any MCMC framework) is not imported by uvplot; the integration is *by convention*:
  bin the visibilities once, then evaluate Galario's `chi2` per MCMC step on either the
  unbinned or binned uvtable. The vendored `example.py` does not exercise emcee; it is
  documented in upstream Galario tutorials, not here.

---

## 12. Notable Internals

- **Mac-only TkAgg override** (`__init__.py` line 16): silently rewrites the matplotlib backend
  on macOS at import time. If the host application has already drawn on a `macosx` figure this
  can be surprising.
- **Header format inconsistency**: `save_ascii_uvtable` writes
  `header["columns"]=COLUMNS_V1` but `header["units"]=UNITS_V2`; the binary writer flips them
  (`uvtable.py` lines 401, 418). This is purely a metadata bug — the actual data layout is
  whatever the explicit `column_stack`/`savez` argument list pins down.
- **`save_ascii_uvtable` is not symmetric with V0 inputs**: it always emits the seven-column V1
  layout including `freqs` and `spws`. If you loaded a V0 file (no freqs/spws) and try to save
  ASCII, you will hit an `AttributeError` when uvplot tries to stack `self.freqs`/`self.spws`
  (`uvtable.py` lines 404-410).
- **`_uvdist` is recomputed on every access** (`uvtable.py` line 255): the property is *not*
  memoised even though the attribute name suggests caching — calling `uv.uvdist` 10 times runs
  10 `np.hypot` evaluations.
- **`uvbin` precomputes intervals as `np.where` tuples** (`uvtable.py` line 482): this can be
  memory-hungry on very large uvtables because each bin caches an integer index array.
- **`numpy.loadtxt` reading**: ASCII reading is via `np.loadtxt`, which is strict about
  whitespace and slow on multi-million-row files. There is no alternative C-side reader.

---

## 13. Tests

The pytest suite (`simulators/uvplot/uvplot/tests.py`, 86 lines) only covers three behaviours:
constructor parity between file and tuple input, `deproject`, and `uvcut`. There is **no test**
for `apply_phase`, `uvbin`, `bin_quantity`, ASCII / binary I/O round-trip, or `export_uvtable`
(the last is hard to test outside CASA). CI runs `py.test uvplot/tests.py` on Ubuntu 20.04 with
Python 3.6 / 3.7 / 3.8 / 3.9 (`simulators/uvplot/.github/workflows/tests.yml`).

---

## 14. Versioning, Releases, Tags

`git tag` (run inside `simulators/uvplot/`) lists 16 historical tags:

```
v0.1.0, v0.1.1, v0.2.0, v0.2.1, 0.2.2, 0.2.3, v0.2.4, v0.2.5, v0.2.6,
v0.2.7, 0.2.7, v0.2.8, v0.2.9, v0.2.10, 0.2.10, 0.2.11
```

(some early tags use bare `0.2.x` without the `v` prefix). The current vendored HEAD is at
commit `9a03579` — `[CI] Add build and publish to PyPI workflow for master branch` — which
corresponds to version `0.2.11`. Recent commit history (top of `git log --oneline`):

```
9a03579 [CI] Add build and publish to PyPI workflow for master branch
84938a3 [CI] Minor renaming
9147483 [CI] Fix running publish-to-test-pypi.yml on all branches but master
8cf614e [CI] use ubuntu 20.04
d439b9b [CI] use v2 actions
40a277d [CI] Minor updates to publish-to-test-pypi.yml
6ddba7c [CI] Run publish-to-test-pypi.yml on all branches but master
...
ad2eee6 [setup] Drop support for Python 2.7
ec3d64c [setup] Store version in main __init__.py to avoid pip install errors
a6f6600 [setup] Move from setup.py to setup.cfg
3e81271 [version] Bump version to 0.2.11
```

There is **no maintained changelog file** in the repo; release notes live on GitHub Releases
(documented in `docs/index.rst` line 68).

---

## 15. Known Limitations / TODOs

- **Equal-channel-count assumption** in `export_uvtable` (Section 7.3) — multi-spw MSes with
  heterogeneous channel counts must be split or channel-averaged first.
- **Dual-pol only**: full-Stokes correlation products are silently ignored.
- **No flag handling**: relies on upstream `split(keepflags=False)`.
- **Implicit Mac backend swap** at import time (Section 5.2).
- **No round-trip support for V0 files via ASCII writer** (Section 12).
- **No CASA-free MS reader**: uvplot cannot consume an MS without a live CASA `tb` object
  (no `python-casacore` fallback path).
- **No frequency-dependent deprojection**: `deproject()` operates on the stored `(u, v)` in
  wavelengths uniformly across all channels — appropriate when the array of `u`/`v` is already
  stored per-channel (as `export_uvtable` does when `channel='all'`), but the user must be aware
  that deprojection commutes with frequency only because each row carries its own frequency.
- **Windows is unsupported** (`docs/install.rst` line 7).
- **Python 2.7 support dropped** at commit `ad2eee6`; no plans to restore.
- **Sphinx docs root in `docs/conf.py`** still has a stale `sys.path.insert(0, '../../python')`
  comment (`docs/conf.py` line 22) — a leftover from the Galario template the file was copied
  from.

---

## 16. Quick-Reference Cheat-Sheet

```python
from uvplot import UVTable, COLUMNS_V0, arcsec

# 1. Load
uv = UVTable(filename='vis.txt', wle=0.00088, columns=COLUMNS_V0)   # ASCII V0, wle=880 µm
uv = UVTable(filename='vis.npz', format='binary')                    # NPZ
uv = UVTable(uvtable=(u, v, re, im, w), columns=COLUMNS_V0, wle=1.0) # in-memory, already in λ

# 2. Properties
uv.u, uv.v          # wavelengths
uv.u_m, uv.v_m      # metres (needs freqs)
uv.V                # complex visibility
uv.uvdist           # √(u²+v²)
uv.weights, uv.freqs, uv.spws

# 3. Geometry
uv.apply_phase(dRA=0.3*arcsec, dDec=0.07*arcsec)
uv.deproject(inc=np.radians(73), PA=np.radians(59), inplace=True)
uv2 = uv.uvcut(maxuv=1e6)

# 4. Binning
uv.uvbin(uvbin_size=30e3)        # populates bin_uvdist, bin_re, bin_im, bin_weights, bin_count
bin_x, bin_x_err = uv.bin_quantity(some_array, use_std=False)

# 5. Plot
axes = uv.plot(label='Data', uvbin_size=30e3)
uv_model.plot(label='Model', axes=axes, yerr=False, color='r', linestyle='-')
axes[0].figure.savefig('uvplot.png')

# 6. Save
uv.save('out.txt', export_fmt='ascii')
uv.save('out.npz', export_fmt='binary')

# 7. CASA only
from uvplot import export_uvtable
export_uvtable('uvtable.txt', tb, vis='ms.ms', channel='all',
               datacolumn='CORRECTED_DATA', dualpol=True)
```

---

## 17. Citation

From `simulators/uvplot/README.md` lines 44-52:

```bibtex
@software{uvplot_tazzari,
  author    = {Marco Tazzari},
  title     = {mtazzari/uvplot},
  month     = oct,
  year      = 2017,
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.1003113},
  url       = {https://doi.org/10.5281/zenodo.1003113}
}
```

License: GNU Lesser General Public License v3 (`simulators/uvplot/LICENSE.txt`).

— *Reference compiled directly from `simulators/uvplot/` HEAD (commit `9a03579`, version
0.2.11). All cited line numbers refer to the working-tree files at that commit.*
