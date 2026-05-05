# pyradiosky — Exhaustive Reference

> *Python objects and interfaces for representing diffuse, extended and compact astrophysical radio sources.*

This document is an in-depth, source-grounded reference for the [pyradiosky](https://github.com/RadioAstronomySoftwareGroup/pyradiosky) package as vendored in `simulators/pyradiosky/`. It is written from a complete reading of the `src/pyradiosky/` source tree (`skymodel.py`, `utils.py`, `spherical_coords_transforms.py`, `cli.py`, `__init__.py`, the `data/` payload and the `docs/tutorial.rst`), the `CHANGELOG.md`, `README.md`, the JOSS paper, and the test layout. It is intended as a primary technical companion when integrating pyradiosky into other code (e.g., RRIVis sky-model loaders, simulators, calibration tooling).

---

## 1. What pyradiosky is

pyradiosky exposes a single primary user class, **`SkyModel`** (defined in `src/pyradiosky/skymodel.py`, ~5080 lines), which holds three families of radio sky representations behind one uniform interface:

* **Compact / point** sources — catalogs with per-component `RA/Dec` (or generic lon/lat in any astropy frame), Stokes I/Q/U/V flux densities (in `Jy` or `K·sr`), spectral information, and optional metadata (errors, beam amplitudes, extended-source group IDs, free-form extra columns).
* **Diffuse / HEALPix** sky maps — pixelised sky temperature/brightness maps (units compatible with `K` or `Jy/sr`) with per-frequency Stokes data on a HEALPix grid.
* **Extended source models** — collections of point components grouped by an `extended_model_group` identifier.

It supports:

* Reading from VOTable, plain text, FHD `.sav` IDL files, and the package's own `.skyh5` HDF5-based format.
* Writing to `.skyh5` and (point-source-only) text files.
* Coordinate transforms across any astropy-supported frame, with HEALPix-aware re-interpolation in `healpix_interp_transform`.
* Polarisation-aware coherency calculation in the source frame and rotated into local Alt/Az for visibility simulators.
* Frequency interpolation and re-sampling for the various spectral types.
* Down-selection, concatenation, unit conversion (Jy ↔ K), conversion between point and HEALPix component types, and rise/set-LST calculations.

The package is part of the **Radio Astronomy Software Group (RASG)** ecosystem alongside `pyuvdata` (`UVBase` is `SkyModel`'s parent class) and `pyuvsim`. Citation: Hazelton *et al.* 2024, JOSS 9(97):6503, DOI [10.21105/joss.06503](https://doi.org/10.21105/joss.06503).

License: **2-clause BSD** (`LICENSE`).

---

## 2. Package layout

```
simulators/pyradiosky/
├── CHANGELOG.md                   # Detailed release history (0.0.1 → 1.1.0+)
├── LICENSE                        # BSD-2-Clause
├── README.md
├── paper.md / paper.bib           # JOSS 2024 paper
├── pyproject.toml                 # PEP 621 metadata, ruff config
├── environment.yaml               # conda env spec
├── setup.py                       # Thin shim
├── MANIFEST.in
├── ci/                            # GitHub Actions configs
├── docs/
│   ├── conf.py / Makefile / make.bat
│   ├── conftest.py
│   ├── developer_docs.rst
│   ├── tutorial.rst               # Doctest-driven user tutorial
│   ├── utility_functions.rst
│   ├── make_index.py / make_skymodel.py
│   ├── Images/                    # Tutorial figures
│   └── references/                # Includes SkyH5 memo PDF
├── src/pyradiosky/
│   ├── __init__.py                # exposes SkyModel, derives __version__
│   ├── skymodel.py                # 5080 LOC, defines SkyModel
│   ├── utils.py                   # stokes/coherency utilities, GLEAM downloader, EoR generator
│   ├── spherical_coords_transforms.py  # vector & basis rotation primitives (310 LOC)
│   ├── cli.py                     # download_gleam, make_flat_spectrum_eor entry points
│   └── data/                      # Test fixtures (see §15)
└── tests/
    ├── conftest.py                # IERS auto-download handling
    ├── test_skymodel.py           # ~4500 LOC of behaviour tests
    ├── test_utils.py
    └── test_spherical_coords_transforms.py
```

---

## 3. Installation, dependencies, version policy

### 3.1 Required

From `pyproject.toml` (matching `README.md`):

| Package           | Min version |
|-------------------|-------------|
| `python`          | ≥ 3.11      |
| `astropy`         | ≥ 6.0       |
| `h5py`            | ≥ 3.7       |
| `numpy`           | ≥ 1.23      |
| `scipy`           | ≥ 1.9       |
| `pyuvdata`        | ≥ 3.2.3     |
| `setuptools_scm`  | ≥ 8.1       |
| `setuptools`      | ≥ 64        |

### 3.2 Optional

| Extra                | Activates                                                                                  | Min version            |
|----------------------|--------------------------------------------------------------------------------------------|------------------------|
| `[healpix]`          | `astropy-healpix` — required for any HEALPix interpolation, indexing, or pixel-area calls. | `astropy-healpix≥1.0.2`|
| `[astroquery]`       | `astroquery` — required for `download_gleam` (Vizier query).                               | `astroquery≥0.4.4`     |
| `[lunarsky]`         | `lunarsky` — enables `MoonLocation` telescope sites and `LunarTopo` Alt/Az transforms.     | `lunarsky≥0.2.5`       |
| `[all]`              | All three above bundled.                                                                   |                        |
| `[doc]`              | `matplotlib`, `pypandoc`, `sphinx`.                                                        |                        |
| `[test]`             | `coverage`, `pre-commit`, `pytest`, `pytest-cov`.                                          |                        |
| `[dev]`              | `[all,test,doc]`.                                                                          |                        |

### 3.3 Installation routes

* `pip install pyradiosky` (PyPI), optionally with extras: `pip install pyradiosky[healpix,lunarsky]`.
* `pip install --no-deps pyradiosky` to skip dependency install.
* Conda: install dependencies from `conda-forge` first, then `pip install .`. The `environment.yaml` provides a working dev environment (`conda env create -f environment.yml`).
* Editable / dev: clone, `pip install -e .[dev]`, `pre-commit install`.

### 3.4 Version derivation

`__version__` is built from `setuptools_scm` with a custom `branch_scheme` (in `__init__.py`) that appends git node, branch name, and a `.dirty` suffix when working trees are dirty. Falls back to `importlib.metadata.version("pyradiosky")` for installed wheels.

### 3.5 Versioning policy

`generation.major.minor`. Breaking changes get ≥ 2 *major* generations of `DeprecationWarning` whenever feasible (`README.md`).

---

## 4. SkyModel class — overview

`SkyModel` subclasses `pyuvdata.uvbase.UVBase`. Every public attribute is internally a `UVParameter` (or its subclass `SkyCoordParameter` / `TelescopeLocationParameter`); `UVBase` dispatches `__init__`, `__eq__`, attribute discovery and the parameter-shape machinery. The class is used in three usage modes:

1. **Empty construction** — `SkyModel()` returns an unpopulated object with all required attributes set to `None`. Populate manually, then call `.check()`.
2. **Direct construction** — `SkyModel(...)` with keyword arguments (see §5).
3. **From file** — `sky = SkyModel.from_file("path.skyh5")` (preferred) or `sky = SkyModel(); sky.read("path")`.

### 4.1 Tolerances

Set in `__init__` and used everywhere:

* `angle_tol = Angle(1e-3, units.arcsec)` (≈ 5 nrad). `_skycoord.tols = self.angle_tol.rad`.
* `freq_tol = 1 * units.Hz` — used by `_freq_array`, `_freq_edge_array`, `_reference_frequency` and as the default `atol` in `at_frequencies`.

### 4.2 Component types (`component_type`)

Acceptable values: `"point"`, `"healpix"`. Set automatically:

* `component_type="healpix"` if explicitly passed, or implicitly when `nside` is provided to `__init__`.
* `component_type="point"` otherwise.

`_set_component_type_params` toggles which `UVParameter`s are required:

| Parameter       | `point` | `healpix` |
|-----------------|---------|-----------|
| `name`          | required| optional  |
| `skycoord`      | required| optional  |
| `hpx_inds`      | optional| required  |
| `nside`         | optional| required  |
| `hpx_order`     | optional| required  |
| `hpx_frame`     | optional| required  |

### 4.3 Spectral types (`spectral_type`)

Acceptable values: `"full"`, `"flat"`, `"subband"`, `"spectral_index"`. Behaviour and constraints:

| Type             | What it means                                                                                          | Required parameters                          | `Nfreqs`           |
|------------------|--------------------------------------------------------------------------------------------------------|----------------------------------------------|--------------------|
| `flat`           | Frequency-independent flux. Optional `reference_frequency`.                                            | `stokes`                                     | `1`                |
| `full`           | Flux specified at every channel.                                                                       | `stokes`, `freq_array`                       | `freq_array.size`  |
| `subband`        | Flux at a set of band centers; band edges (`freq_edge_array`) describe each band.                      | `stokes`, `freq_array` (or `freq_edge_array`)| `freq_array.size`  |
| `spectral_index` | Flux specified at one `reference_frequency` per source, with a power-law `spectral_index`.             | `stokes`, `reference_frequency`, `spectral_index` | `1`            |

`_set_spectral_type_params` mutates the `required` flag on the relevant `UVParameter`s. Note `_Nfreqs.acceptable_vals = [1]` for `flat` / `spectral_index`.

### 4.4 Coordinate frames

A frame is **any subclass of `astropy.coordinates.BaseCoordinateFrame`**. The frame is stored:

* For point components: implicitly via `skycoord.frame`. Read with the `frame` property (`SkyModel.__getattr__` falls back to `skycoord.frame.name`).
* For HEALPix components: explicitly on `hpx_frame` (an `astropy` frame instance, replicated without data). The pixel grid is interpreted in that frame.

Init helpers:

* `_get_frame_obj(frame)` accepts a frame string (resolved via `frame_transform_graph.lookup_name`) or instance.
* `_get_lon_lat_component_names(frame_obj)` returns the lon/lat attribute names for a frame (e.g. `("ra","dec")` for ICRS/FK5/FK4, `("l","b")` for Galactic).
* `_get_frame_desc_str(frame_obj)` produces the file-system-safe descriptor used in text-file column names: `"icrs"`, `"galactic"`, `"j2000"` for `fk5(j2000)`, `"b1950"` for `fk4(b1950)`, `"fk5_J<jyear>"`, `"fk4_B<byear>"`. Other frames raise on text writes.

The `__init__` accepts coords via several mutually exclusive routes:

* `skycoord=...` — pass an `astropy.coordinates.SkyCoord` directly.
* `ra=`, `dec=` (+ implicit `frame="icrs"` if not given, but `frame` is *required*).
* `gl=`, `gb=` (Galactic coordinates).
* `lon=`, `lat=` (generic, requires `frame=...`).

Setting `ra/dec` for a non-RA/Dec frame, or `gl/gb` for a non-Galactic frame, raises `ValueError`.

---

## 5. SkyModel parameters (the data model)

All attributes are `UVParameter`s. Forms are tuples of dimension labels (resolved at check-time using attribute values). Required-ness depends on the active `component_type` / `spectral_type` (see §4).

### 5.1 Topology / counts

* **`Ncomponents`** *(int, required)* — number of components (point sources or HEALPix pixels).
* **`Nfreqs`** *(int, required)* — number of frequency channels (1 for `flat`/`spectral_index`).

### 5.2 Component identification

* **`component_type`** *(str, required, ∈ {"healpix","point"})*.
* **`name`** *(str array, shape `(Ncomponents,)`, required for `point`)* — unique component identifiers.
* **`skycoord`** *(astropy `SkyCoord`, shape `(Ncomponents,)`)* — required for `point`. Its `.frame` is the source frame.
* **`nside`** *(int, required for `healpix`)*.
* **`hpx_order`** *(str, ∈ {"ring","nested"}, required for `healpix`, defaults to `"ring"`)*.
* **`hpx_frame`** *(astropy frame instance, required for `healpix`)*.
* **`hpx_inds`** *(int array, shape `(Ncomponents,)`, required for `healpix`)* — indices into the full `12·nside²` HEALPix grid.

### 5.3 Spectral axis

* **`spectral_type`** *(str, required, ∈ {"full","flat","subband","spectral_index"})*.
* **`freq_array`** *(`Quantity[Hz]`, shape `(Nfreqs,)`)* — band centers; required for `full`/`subband`. Tolerated unit equivalence: `Hz`.
* **`freq_edge_array`** *(`Quantity[Hz]`, shape `(2, Nfreqs)`)* — `[0,:]`=lower edge, `[1,:]`=upper. Required for `subband` (warning + autocompute today, **error in v0.5**). Auto-computed via `_get_freq_edges_from_centers()` if the centers are equally spaced (`uvutils._test_array_constant_spacing` from `pyuvdata`).
* **`reference_frequency`** *(`Quantity[Hz]`, shape `(Ncomponents,)`)* — required for `spectral_index`.
* **`spectral_index`** *(float, shape `(Ncomponents,)`)* — required for `spectral_index`. NaNs trigger a check warning (common for GLEAM sources poorly fit by a power law).

`check()` enforces *only one* of `freq_array` / `reference_frequency` may be set.

### 5.4 Stokes / coherency

* **`stokes`** *(Quantity, shape `(4, Nfreqs, Ncomponents)`, required)*. First axis is `[I, Q, U, V]`. Allowed unit families:

  * `point` → equivalent to `Jy` or `K·sr`.
  * `healpix` → equivalent to `Jy/sr` or `K`.

* **`stokes_error`** *(optional Quantity, same shape, units equivalent to `stokes.unit`)* — per-flux-bin uncertainty.
* **`frame_coherency`** *(Quantity, shape `(2, 2, Nfreqs, Ncomponents)`)* — electric-field correlation matrix in the *source frame* basis (computed and optionally cached by `calc_frame_coherency`). The local Alt/Az coherency is produced on demand by `coherency_calc()`.
* **`beam_amp`** *(float array, shape `(4, Nfreqs, Ncomponents)`, optional)* — beam amplitudes evaluated at the source positions for `[XX, YY, XY, YX]` instrumental polarisations. Populated when reading FHD files that include beam values.

Polarisation cache: `__init__` precomputes:

```python
self._polarized   = np.where(np.any(np.sum(stokes[1:, :, :], axis=0) != 0., axis=0))[0]
self._n_polarized = np.unique(self._polarized).size
```

`coherency_calc()` uses these to skip rotation work for unpolarised components.

### 5.5 Extended-source bookkeeping

* **`extended_model_group`** *(str array, shape `(Ncomponents,)`)* — same string for all components belonging to one extended source (FHD-style models). Empty string for normal point sources.

### 5.6 Catalog metadata

* **`extra_columns`** *(numpy `recarray`, length `Ncomponents`)* — free-form per-component metadata. Populated via `add_extra_columns(names=..., values=..., dtype=None)`. Quantities are *not* supported; values must be 1D arrays of length `Ncomponents`. Internally tracked dtype list is reflected back into `_extra_columns.expected_type` so the parameter type-check stays honest.

### 5.7 Provenance

* **`history`** *(str, required)* — text log; pyradiosky always appends a string of the form `"  Read/written with pyradiosky version: <__version__>."` (from `pyradiosky_version_str`).
* **`filename`** *(str list, optional)* — basenames of input files that contributed.

### 5.8 Time-and-position-derived attributes

These are *transient*: they require `update_positions(time, telescope_location)` to populate, and are cleared by `clear_time_position_specific_params()`. They are also cleared automatically by `assign_to_healpix` and during `concat(clear_time_position=True)`:

* **`time`** *(astropy `Time`)*.
* **`telescope_location`** *(astropy `EarthLocation` or `lunarsky.MoonLocation`)*. The `_telescope_location` parameter is a custom `TelescopeLocationParameter` that uses simple `value == value` equality (`EarthLocation` does not implement standard equality consistently).
* **`alt_az`** *(float array, shape `(2, Ncomponents)`)* — `[alt_rad, az_rad]`.
* **`pos_lmn`** *(float array, shape `(3, Ncomponents)`)* — direction cosines: `l = sin(az)cos(alt)`, `m = cos(az)cos(alt)`, `n = sin(alt)`.
* **`above_horizon`** *(bool array, shape `(Ncomponents,)`)*.

The list of these is exposed as `_time_position_params == ["time", "telescope_location", "alt_az", "pos_lmn", "above_horizon"]`.

### 5.9 Iteration helper

`SkyModel.ncomponent_length_params` is a property that yields all internal `UVParameter` names whose `form == ("Ncomponents",)`, used by methods like `assign_to_healpix` and `concat` to operate generically on per-component arrays.

---

## 6. Construction patterns

### 6.1 Point catalog from arrays

```python
from pyradiosky import SkyModel
from astropy import units as u
from astropy.coordinates import Longitude, Latitude
import numpy as np

ra  = Longitude(np.array([10., 20.]), u.deg)
dec = Latitude(np.array([-30., -25.]), u.deg)
stokes = np.zeros((4, 1, 2)) * u.Jy
stokes[0, ...] = 1.0 * u.Jy

sm = SkyModel(
    name=["src1", "src2"],
    ra=ra, dec=dec, frame="icrs",
    stokes=stokes,
    spectral_type="flat",
)
```

Equivalent forms:

* Pass a pre-built `skycoord=SkyCoord(...)` instead of `ra/dec/frame`.
* Use `gl=`, `gb=` for Galactic.
* Use generic `lon=`, `lat=`, `frame=`.

A single-component object will have `Ncomponents == 1` but `stokes` is auto-reshaped to `(4, Nfreqs, 1)`.

### 6.2 HEALPix map

```python
sm = SkyModel(
    component_type="healpix",
    nside=8, hpx_inds=np.arange(12*8*8),
    hpx_order="ring", frame="icrs",
    stokes=stokes_K,                          # shape (4, Nfreqs, npix), units 'K'
    spectral_type="full",
    freq_array=np.linspace(50e6, 150e6, 10) * u.Hz,
)
```

### 6.3 Empty + `read()`

```python
sm = SkyModel()
sm.read("path/to/file")    # filetype auto-detected from extension
```

Auto-detection rules (from `read`):

| Extension  | Filetype                                                       |
|------------|----------------------------------------------------------------|
| `.txt`     | `text`                                                         |
| `.vot`     | `gleam` (if `"gleam"` in lowercase filename) else `vot`        |
| `.skyh5`   | `skyh5`                                                        |
| `.sav`     | `fhd`                                                          |
| anything else | requires explicit `filetype=` else `ValueError`             |

`from_file(filename, **kwargs)` is the classmethod equivalent.

### 6.4 Required-parameter contract

`__init__` validates that *all* of `[name, skycoord, stokes, spectral_type]` (point) or `[nside, frame, hpx_inds, stokes, spectral_type]` (healpix) are simultaneously set. Spectral-type-specific extras are added: `spectral_index` types additionally require `spectral_index` and `reference_frequency`; `subband` requires `freq_array` *or* `freq_edge_array`; `full` requires `freq_array`.

If any optional spectral-axis parameters (`freq_array`, `freq_edge_array`, `reference_frequency`) are passed without the rest, you get a clear `ValueError` listing what was actually set.

---

## 7. The `check()` method

`SkyModel.check(check_extra=True, run_check_acceptability=True)` runs:

1. Re-runs `_set_spectral_type_params(self.spectral_type)` and `_set_component_type_params(self.component_type)` to make required-flags consistent.
2. Forbids both `freq_array` *and* `reference_frequency` being set.
3. For `subband` without `freq_edge_array`: tries to recompute it from constant-spacing centers, else warns. *This will be hard error in v0.5* (`DeprecationWarning`).
4. Calls `super().check(...)` (`UVBase`) for shape/type checks.
5. Validates `stokes` and `frame_coherency` units against the component type's allowed family (`Jy`/`K·sr` for point, `Jy/sr`/`K` for healpix).
6. `stokes_error.unit` must be equivalent to `stokes.unit`.
7. `freq_array` and `reference_frequency` (if set) must be equivalent to `Hz`.
8. Acceptability checks (warnings, not errors):
   * NaN spectral indices (with hint: switch GLEAM to `subband`).
   * NaN Stokes values (hint: `select(non_nan="any"|"all")`).
   * Negative Stokes I (hint: `select(non_negative=True)`).

Return value: `True` (raises on hard failures).

---

## 8. Frame and coordinate methods

* **`__getattr__(name)`** intercepts a few names:
  * `frame` returns `skycoord.frame.name` or `hpx_frame.name`.
  * Frame-component names (e.g. `ra`, `dec`, `l`, `b`) on point objects pass through to `skycoord.<name>`. On HEALPix objects this works but warns to use `get_lon_lat()` directly (it has to materialise a temporary HEALPix grid).
* **`get_lon_lat()`** — returns the `(lon_attr, lat_attr)` `Quantity` pair using the appropriate component names. For HEALPix, computes `astropy_healpix.HEALPix(nside, order, frame).healpix_to_skycoord(hpx_inds)` and extracts. (Requires `astropy-healpix`.)
* **`transform_to(frame)`** — point-only thin wrapper around `SkyCoord.transform_to`. Updates `self.skycoord`. *Raises `ValueError` for HEALPix* (which would change pixel definitions).
* **`healpix_interp_transform(frame, full_sky=False, inplace=True, run_check=True, ...)`** — HEALPix re-projection: computes new pixel centers in the target frame, then bilinearly interpolates the old map onto them via `astropy_healpix.HEALPix.interpolate_bilinear_skycoord`, masking pixels outside the original `hpx_inds`. Tested to agree with `healpy.Rotator` to 1 part in 10⁻⁵. Notes:
  * Iterates over freq × Stokes individually (slow for big maps).
  * **No Q+iU rotation fix**, so it raises `NotImplementedError` if any of `stokes[1:]` is non-zero (i.e. polarised maps). Stokes Q/U/V are skipped (`if stokes_ind > 0: continue`).
  * `full_sky=True` returns the entire `12·nside²` grid (with zero-flux pixels) instead of dropping pixels where the interpolation mask is set.
  * Number of components can change (different masks → different valid-pixel counts).
  * Recomputes `frame_coherency` if it was previously cached.

---

## 9. HEALPix ↔ point conversions

* **`healpix_to_point(to_jy=True, ...)`**: multiplies `stokes` and (if present) `frame_coherency` by `astropy_healpix.nside_to_pixel_area(nside)`, sets `skycoord` from `get_lon_lat()`, drops `hpx_frame`, switches `component_type` to `"point"`. Component names auto-assigned as `f"nside{nside}_{order}_{ind}"`. Optionally calls `kelvin_to_jansky()`.
* **`_point_to_healpix(to_k=True, ...)`** *(private)*: the inverse — only valid on a point object that *was* a HEALPix object before (i.e. `nside`, `hpx_inds`, `hpx_order` are all still set). Divides by pixel area, restores `hpx_frame`, drops `skycoord`/`name`. Public users should call `assign_to_healpix` instead.
* **`assign_to_healpix(nside, order="ring", to_k=True, full_sky=False, sort=True, inplace=False, ...)`**:
  * Bins point components to their nearest HEALPix pixel via `astropy_healpix.HEALPix.skycoord_to_healpix`.
  * Clears time/position parameters (`clear_time_position_specific_params`).
  * If multiple sources fall in one pixel:
    * **Stokes**: summed.
    * **`stokes_error`**: added in quadrature.
    * Other per-component params: take the value from the first match, but raise if `spectral_index` / `reference_frequency` *vary* across sources in the same pixel (suggests a finer `nside` or pre-`at_frequencies()`). For `beam_amp`, raises if it varies; for `name` (non-numeric), takes the first.
  * Divides `stokes` by `nside_to_pixel_area(nside)`.
  * `full_sky=True` pads with zero-flux pixels (using median spectral index / reference frequency / empty extended-model-group / zero beam_amp / zero stokes_error).
  * `sort=True` sorts by `hpx_inds` and reorders all per-component parameters.
  * Optionally calls `jansky_to_kelvin()`.
  * **Caution**: this physically moves point components to pixel centers (not a coordinate transform).

---

## 10. Unit conversions: K ↔ Jy

* **`kelvin_to_jansky()`** — does nothing if already in Jy-equivalent units. Conversion factor uses `astropy.units.brightness_temperature(freqs, beam_area=1*sr)` via `utils.jy_to_ksr(freqs)`. Branches:
  * For `spectral_index` and `flat`+`reference_frequency`, factor uses `reference_frequency` (per-component).
  * For `freq_array`-based types, factor uses `freq_array` (per-frequency).
  * Else raises (you have a `flat` model with no frequency anchor → no conversion possible).
  After scaling, calls `.to(units.Jy)` to compress the composite unit and recalculates `frame_coherency` if cached.
* **`jansky_to_kelvin()`** is the symmetric inverse, multiplying by `jy_to_ksr(freqs)`.

`utils.jy_to_ksr(freqs)` returns a `Quantity[K·sr/Jy]` of shape equal to `freqs` and is used internally by the SkyModel and by user code that needs the conversion factor outside an object.

---

## 11. Frequency operations

### 11.1 `at_frequencies(freqs, inplace=True, freq_interp_kind="cubic", nan_handling="clip", run_check=True, atol=None)`

Produces a `"full"` spectral-type model, evaluating at the requested `Quantity[Hz]` frequencies. Behaviour by current `spectral_type`:

* **`spectral_index`**: `stokes_new = stokes * (f_new / f_ref)**alpha`. Raises if any `spectral_index` is NaN. `reference_frequency` is dropped.
* **`full`**: subset selection of existing channels — raises if any requested frequency is not already present (within `atol`, default `freq_tol = 1 Hz`). Slices `freq_edge_array` accordingly when present.
* **`subband`**: `scipy.interpolate.interp1d(..., kind=freq_interp_kind)` along the freq axis. Raises if requested freqs are outside the data range. NaN handling is the most subtle bit (also documented in `tutorial.rst`):

  * `"propagate"` — any NaN in a source's stokes triggers all NaN output for that source.
  * `"interp"` — interpolate using only the non-NaN frequencies; out-of-range outputs become NaN; sources with too few non-NaN values to support `freq_interp_kind` fall back to linear (warns).
  * `"clip"` (default) — same as `"interp"` but out-of-range values are clipped to the nearest non-NaN value instead of NaN.
  * Warnings are emitted summarising counts of all-NaN, NaN-high, NaN-low and reduced-order-fallback components.
* **`flat`**: replicates the single-channel stokes across the requested frequencies.

Post-conditions: `spectral_type = "full"`, `Nfreqs = len(freqs)`, `freq_array = freqs`, `reference_frequency = None`, `freq_edge_array = None`. If `frame_coherency` was cached it is recomputed.

### 11.2 Helpers (module level)

* `_get_freq_edges_from_centers(freq_array, tols, raise_error=True)` — derives edges by taking half the mean spacing on either side of each center; raises on non-constant spacing.
* `_get_freq_centers_from_edges(freq_edge_array)` — `mean(axis=0)`.

---

## 12. Time / location, alt-az, coherency

### 12.1 `update_positions(time, telescope_location)`

* Validates `time` is `astropy.time.Time` and `telescope_location` is `EarthLocation` (or `lunarsky.MoonLocation`, only when `lunarsky` is installed; the `_telescope_location` parameter has its `expected_type` extended at runtime to permit it).
* Memoises: skips work if both inputs match the currently stored values.
* Builds `SkyCoord(*get_lon_lat(), frame=hpx_frame)` for HEALPix; uses `self.skycoord` directly for point.
* Transforms to `AltAz(obstime=time, location=telescope_location)` (Earth) or `LunarTopo` (Moon).
* Stores `alt_az = [alt.rad, az.rad]`, derives `pos_lmn`, sets `above_horizon = alt_az[0] > 0`.

### 12.2 `calculate_rise_set_lsts(telescope_latitude, horizon_buffer=0.04364)`

Sets two non-`UVParameter` attributes on `self`: `_rise_lst`, `_set_lst` (1-D arrays per component, in radians, `2π`-wrapped, or NaN for never-rising / never-setting sources). Default buffer ≈ 10 minutes of sky rotation; the catalogue avoids precession/nutation drift relative to J2000.

### 12.3 `cut_nonrising(telescope_latitude, inplace=True, ...)`

Removes sources whose declination/latitude fails the rise condition `tan(lat_telescope)·tan(lat_source) ≥ -1`. Internally calls `select(component_inds=...)`.

### 12.4 `calc_frame_coherency(store=True)`

Computes `frame_coherency = utils.stokes_to_coherency(stokes)` and (default) stores it. The 1/2 prefactor gives `Re[V_xx + V_yy] = I` for unpolarised sources:

```
        ⎡ I + Q       U − iV ⎤
C = ½ · ⎢                    ⎥
        ⎣ U + iV      I − Q  ⎦
```

This coherency is in the **source frame** basis (e.g. RA/Dec or Galactic).

### 12.5 `coherency_calc(store_frame_coherency=True)`

Returns the `(2, 2, Nfreqs, Ncomponents_above_horizon)` coherency in the local **Alt/Az** basis. Pipeline:

1. Build `frame_coherency` if absent.
2. Restrict to `above_horizon` mask (default: all True if `update_positions` not called, with a warning).
3. For unpolarised sources (no Q/U/V anywhere): the rotation is identity, just return the slice.
4. For polarised components only, compute the basis-vector rotation matrix per source and apply
   `C_local = R^T · C_frame · R` via `np.einsum`.

The rotation matrix construction layers two helpers in `_calc_*`:

* **`_calc_average_rotation_matrix()`** — transforms ICRS Cartesian unit basis vectors into Alt/Az (or `lunartopo` if on the Moon), and orthogonalises with `scipy.linalg.orthogonal_procrustes` (the astropy 3D matrix is only orthogonal to ~10⁻⁷).
* **`_calc_rotation_matrix(inds=None)`** — for each source, finds a small "perturbation" rotation that maps the average-transformed source vector to the exact Alt/Az vector. Uses `spherical_coords_transforms.vecs2rot` (axis-angle via `axis_angle_rotation_matrix`).
* **`_calc_coherency_rotation(inds=None)`** — produces the 2×2 basis-vector rotation between θ̂/φ̂ basis at the source position and at the corresponding Alt/Az position via `sct.spherical_basis_vector_rotation_matrix(...)`.

Mathematical primitives live in `spherical_coords_transforms.py`:

| Function                                           | Purpose                                                                                       |
|----------------------------------------------------|-----------------------------------------------------------------------------------------------|
| `r_hat(theta, phi)`                                | Unit radial vector, `(3, Npts)`.                                                              |
| `theta_hat(theta, phi)`                            | Unit θ̂ vector.                                                                                |
| `phi_hat(theta, phi)`                              | Unit φ̂ vector.                                                                                |
| `rotate_points_3d(rot_matrix, theta, phi)`         | Apply 3×3 rotation to one point on the sphere; returns new (β, α). Single-point only.         |
| `spherical_basis_vector_rotation_matrix(theta,phi,rot_matrix,beta=None,alpha=None)` | 2×2 rotation taking θ̂/φ̂ basis to β̂/α̂ basis; computes (β,α) via `rotate_points_3d` if not given. |
| `axis_angle_rotation_matrix(axis, angle)`          | Rodrigues' formula. Validates `axis` is a unit 3-vector.                                      |
| `is_orthogonal(matrix, tol=1e-15)`                 | Test `M·Mᵀ ≈ I`.                                                                              |
| `is_unit_vector(vec, tol=1e-15)`                   | Test ‖v‖ ≈ 1.                                                                                 |
| `vecs2rot(r1=None, r2=None, theta1=...,phi1=..., theta2=..., phi2=...)` | Smallest-angle rotation taking `r1` to `r2`; works from either explicit unit vectors or (θ,φ) pairs. |

Convention: `theta` is **co-latitude** (angle from +z), `phi` is azimuth from +x. Used internally for `(RA, Dec) → (Alt, Az)` rotations after `RA → φ`, `π/2 − Dec → θ`.

---

## 13. Selection (`select`)

```python
sky.select(
    component_inds=None,          # array_like[int]
    lat_range=None,               # Latitude, shape (2,)
    lon_range=None,               # Longitude, shape (2,) — second < first wraps through 0
    min_brightness=None,          # Quantity equivalent to stokes.unit
    max_brightness=None,
    brightness_freq_range=None,   # Quantity[Hz], shape (2,)
    non_nan=None,                 # "any" | "all" | None
    non_negative=False,
    inplace=True,
    run_check=True, check_extra=True, run_check_acceptability=True,
)
```

Notes:

* `non_nan="any"` drops components with any NaN in `stokes`; `non_nan="all"` drops those with NaN at *all* frequencies.
* `non_negative=True` drops components with any negative Stokes I (`stokes[0] < 0`).
* `lon_range` wraps when the second value is less than the first (e.g. `[350°, 10°]` selects RA > 350° ∪ RA < 10°).
* `min_brightness` / `max_brightness` are *per-component* aggregations: `np.min` / `np.max` of `stokes[0]` over the (possibly restricted) frequency window. Currently raises `NotImplementedError` for `spectral_type == "spectral_index"`.
* The implementation delegates the per-axis trimming to `pyuvdata.UVBase._select_along_param_axis({"Ncomponents": component_inds})` (post-1.1.0), which automatically handles every parameter whose `form` mentions `Ncomponents`.
* History string updated: `"  Downselected to specific components using pyradiosky."`.

---

## 14. Concatenation (`concat`)

```python
combined = sky_a.concat(
    sky_b,
    inplace=True,
    clear_time_position=True,        # zero-out time/pos params on both
    verbose_history=False,
    run_check=True, check_extra=True, run_check_acceptability=True,
)
```

Compatibility rules (raise on mismatch):

* `_component_type` and `_spectral_type` must match.
* `subband`/`full`: `_freq_array` must match (and `_freq_edge_array` for `subband`).
* `healpix`: also `_nside`, `_hpx_order`.
* If `clear_time_position=False`, the four time/pos parameters must match.
* Component identifiers must not overlap: same `name`s on point objects, same `hpx_inds` on HEALPix objects → `ValueError`.
* `extra_columns` must either both be present *with the same dtype* or both absent.

Behaviour:

* `name`, `hpx_inds`, `skycoord`, `stokes`, `frame_coherency`, optional Q-shaped extras (`stokes_error`, `extended_model_group`, `beam_amp`, `reference_frequency`, `spectral_index` for `flat`/`full`/`subband` types) are concatenated along the appropriate axis.
* If only one object has an optional parameter, missing values are filled with `NaN` (numeric) or `""` (string) and a warning is emitted. (One exception: cached `frame_coherency` is dropped to `None` — the user must call `calc_frame_coherency()` again.)
* `filename`s are unioned via `pyuvdata.utils.tools._combine_filenames`.
* History is appended: `" Combined skymodels along the component axis using pyradiosky."`. Differences between the two histories are merged using `pyuvdata.utils.history._combine_history_addition` (or fully concatenated when `verbose_history=True`).

---

## 15. File formats: read & write

### 15.1 SkyH5 (the native format)

SkyH5 is an HDF5-based container designed by the pyradiosky team. Memo: `docs/references/skyh5_memo.pdf`. Layout produced by `write_skyh5`:

```
/Header
    Ncomponents        (int)           [always]
    Nfreqs             (int)           [always]
    component_type     (str)           [always]
    spectral_type      (str)           [always]
    history            (str)           [always]
    name               (str array)     [point objects]
    nside              (int)           [healpix]
    hpx_order          (str)           [healpix]
    hpx_inds           (int array)     [healpix]
    freq_array         (Quantity[Hz])  [full / subband]
    freq_edge_array    (Quantity[Hz])  [subband]
    reference_frequency (Quantity[Hz]) [spectral_index]
    spectral_index     (float array)   [spectral_index]
    extended_model_group (str array)   [optional]
    skycoord/          (group)         [point — full SkyCoord.info dict]
       …               (lon, lat, frame, representation_type, …)
    hpx_frame/         (group)         [healpix — frame description]
    extra_columns/     (group)         [optional]
       <name1>         (array, Ncomponents)
       <name2>         …
/Data
    stokes             (4, Nfreqs, Ncomponents)   attrs.unit=str(stokes.unit)
    stokes_error       (4, Nfreqs, Ncomponents)   [optional, attrs.unit=str(stokes_error.unit)]
    beam_amp           (4, Nfreqs, Ncomponents)   [optional]
```

Encoding details (from `_add_value_hdf5_group`/`_get_value_hdf5_group`):

* `Quantity`-typed values are stored as raw arrays with an `attrs["unit"]` string.
* Special `astropy` types are tagged via `attrs["object_type"]`: `"latitude"`, `"longitude"`, `"earthlocation"` (stored as geocentric `Quantity`), `"time"` (stored as `str(Time)`).
* `EarthLocation` round-trips via `EarthLocation.from_geocentric(*value)`.
* `Time` round-trips via `Time(str)`.
* Strings are stored as `np.bytes_` (or `dtype=bytes` arrays) and re-decoded as UTF-8 on read.
* The legacy attribute `attrs["angtype"]` is recognised on read for compatibility with older files (in addition to the current `attrs["object_type"]`).

`read_skyh5(filename, skip_params=False, ...)`:

* Verifies that `/Header` exists.
* Reads `component_type` first to set required-parameter expectations.
* Pulls `skycoord` / `hpx_frame` from a nested group; falls back to legacy `lat/lon/frame` or `ra/dec` keys with a deprecation warning for old skyh5 files written before SkyCoord nesting.
* Honours `skip_params` (`bool` ⇒ all skippable, `str` or list ⇒ specific). Skippable: `extended_model_group`, plus `nside`/`hpx_inds`/`hpx_order` for point and `name` for healpix.
* For old `subband` files lacking `freq_edge_array`: tries to recompute, otherwise downgrades to `"full"` with a warning.
* For files lacking any frame info, defaults to `"icrs"` with a warning.
* Reads `stokes`/`stokes_error`/`beam_amp` from `/Data`, with a fallback to the old location under `/Header`.

`write_skyh5(filename, clobber=False, data_compression=None, ...)`:

* `clobber=False` raises `OSError` if the file exists.
* `data_compression` accepts any HDF5 filter name (e.g. `"gzip"`).
* Always appends the pyradiosky version string to `history` if not already present.
* Always uses chunked datasets for `stokes`, `stokes_error`, `beam_amp`.

### 15.2 GLEAM votable

`read_gleam_catalog(gleam_file, spectral_type="subband", with_error=False, use_paper_freqs=False, ...)`:

* Three modes (`tutorial.rst` describes the catalog in detail):
  * `"flat"`: uses `Fintwide`, anchor freq 200 MHz.
  * `"spectral_index"`: uses `Fintfit200` + `alpha`; some sources have NaN α (not well fit).
  * `"subband"` (**default**): uses 20 sub-band fluxes `Fint076…Fint227` at 8 MHz spacing (76, 84, …, 227 MHz). Some sources have NaN/negative I in some sub-bands.
* Built-in band edge arrays (catalog values vs `use_paper_freqs=True` for the 7.68 MHz × 4 = 30.72 MHz coarse-channel theoretical layout).
* `with_error=True` populates `stokes_error` from the `e_Fint*` columns. The GLEAM paper specifies these are *fitting* errors only, with separate flux-scale errors of 2–3% intra-band or up to 80% to other catalogs.
* Internally calls `read_votable_catalog` with `frame="fk5"`, `id_column="GLEAM"`, etc.
* Required catalog file: `gleam_50srcs.vot` (50-row test fixture under `data/`) for unit tests; full catalog comes from Vizier `VIII/100/gleamegc` (downloadable via `cli.download_gleam`).

### 15.3 Generic VOTable

`read_votable_catalog(votable_file, table_name, id_column, lon_column, lat_column, flux_columns, frame, reference_frequency=None, freq_array=None, freq_edge_array=None, spectral_index_column=None, flux_error_columns=None, history="", ...)`:

* Uses `astropy.io.votable.parse`. Each `*_column` parameter is a *substring* matched against table column names via `_get_matching_fields`, which:
  * Tries casefold substring match.
  * If multiple matches, retries exact casefold match.
  * Else excludes columns starting with `_` (VizieR-computed columns).
  * Raises if still ambiguous (`brittle=True` default).
* Multiple flux columns ⇒ `subband` spectral type; `freq_array` and/or `freq_edge_array` must be supplied. If only `freq_array` is supplied and is regularly spaced, `freq_edge_array` is auto-derived.
* Single flux column with `reference_frequency` ⇒ `spectral_index` (if `spectral_index_column` is also given) or `flat`.
* Unit handling: each flux column's VOTable unit is consulted; the first of `["Jy", "Jy/sr", "K", "K sr"]` that all columns are equivalent to is the working unit. Inconsistent columns ⇒ `ValueError`.
* `flux_error_columns` produce the `stokes_error` array; their units must be consistent with the flux units.

### 15.4 Text catalogs

`read_text_catalog(catalog_csv, ...)`. Tab-separated columns, header parsed with bracketed unit strings (e.g. `[deg]`) ignored. Required columns:

* `source_id` (string ≤ 10 chars).
* Lon column with frame embedded in the name (`ra_icrs`, `ra_j2000`, `ra_b1950`, `l_galactic`, `lon_<frame>`, …).
* Lat column with same frame info.
* `Flux [Jy]` (Stokes I in Jy).

Optional:

* Multi-frequency: `Flux at <N> <unit>` columns for the same set of frequencies (`Hz`, `kHz`, `MHz`, `GHz` — the `pyuvdata.uvbeam.cst_beam.CSTBeam.name2freq` parser is reused).
* `Frequency [Hz]` (single column) makes the model `flat` (or `spectral_index` with an additional `Spectral_Index` column).
* `Flux_error_*` columns (one per flux column) populate `stokes_error`.

`write_text_catalog(filename)`:

* Restrictions: `component_type == "point"`, `stokes.unit` equivalent to `Jy`, `spectral_type != "subband"` (must convert to `full` if you must export). Errors out with a clear message when those conditions are not met. `extended_model_group` must be `None`.
* Writes `source_id`, `<lon>_<frame_desc>` & `<lat>_<frame_desc>`, then per-Stokes-with-non-zero-flux flux columns (and matching error columns if available), `frequency` *or* `reference_frequency`+`spectral_index`, and optionally `rise_lst`/`set_lst` if present on the object.

### 15.5 FHD `.sav` catalogs

`read_fhd_catalog(filename_sav, expand_extended=True, ...)`:

* Uses `scipy.io.readsav`.
* Recognises top-level keys `catalog` *or* `source_array`; raises if neither.
* Frequencies in FHD files are in MHz — pyradiosky multiplies by 1e6 (this used to be a bug, fixed in 0.3.1).
* Polarised Stokes Q/U/V are read directly (FHD provides four per source).
* `BEAM` substructure populates `beam_amp` (`XX, YY, |XY|, |YX|`).
* `expand_extended=True` (default): expands extended-source components from each source's `extend` substructure; assigns each component a derived id `<source_id>_<n>` and propagates `extended_model_group=<source_id>`.
* Duplicate ids are renamed to `<id>-1`, `<id>-2` etc with a warning.
* Resulting model is always `spectral_type="spectral_index"` with `frame="icrs"`.

### 15.6 Universal entry point

`SkyModel.read(filename, filetype=None, **kwargs)` and `SkyModel.from_file(filename, **kwargs)` route to one of the five readers above based on `filetype` (auto-detected from extension if not provided). All format-specific parameters are exposed as keyword arguments on `read` for a single, uniform API surface.

---

## 16. CLI / scripts

Both are exposed as console-script entry points via `[project.scripts]` in `pyproject.toml`:

* **`download_gleam`** — wraps `utils.download_gleam`. Flags: `--path`, `--filename`, `--overwrite`, `--row_limit`, `--for_testing` (CI helper that downloads the 50-source test catalog using a US Vizier mirror).
* **`make_flat_spectrum_eor`** — wraps `utils.flat_spectrum_skymodel`. Flags: `-v/--variance` (K² at the reference channel), `--nside` (power of 2), `--ref_chan` (default 0), `-s/--start_freq`, `-e/--end_freq`, `-N/--nfreqs` (Hz), `--filename` (default `noise_sky.hdf5`), `--frame` (default `icrs`).

Behaviour is described in §18.2 — the generated SkyModel is a noise-like, K-units HEALPix sky with cosmologically-scaled variance per voxel using `astropy.cosmology.Planck15.differential_comoving_volume`.

---

## 17. Top-level `utils.py`

| Function                                                                                                                      | Purpose |
|-------------------------------------------------------------------------------------------------------------------------------|---------|
| `stokes_to_coherency(stokes_arr)`                                                                                             | Builds the 2×2 frame coherency from Stokes (Quantity-only). Validates first axis = 4. |
| `coherency_to_stokes(coherency_matrix)`                                                                                       | Inverse. Returns real array with `stokes_arr.unit`. |
| `jy_to_ksr(freqs)`                                                                                                            | Returns `Quantity[K·sr/Jy]` factor. Wraps `units.brightness_temperature(freqs, beam_area=1*sr)`. |
| `download_gleam(path=".", filename="gleam.vot", overwrite=False, row_limit=None, for_testing=False)`                          | Vizier download helper. `for_testing=True` overrides path/filename to populate the unit-test fixture and switches the Vizier server to `vizier.cfa.harvard.edu`. Requires `astroquery`. |
| `flat_spectrum_skymodel(*, variance, nside, ref_chan=0, ref_zbin=0, redshifts=None, freqs=None, frame="icrs")`                | Generates a Gaussian-noise HEALPix `SkyModel` with the cosmologically-correct voxel volume scaling so the reference channel has variance `variance` (K²) and other channels follow `1/√(voxel_volume)`. Requires `freqs` *or* `redshifts` (one is derived from the other via the 21 cm rest frequency `f21 = 1.420405751e9 Hz`). The expected power-spectrum amplitude `variance · vol(ref_chan)` is recorded in `history`. |

---

## 18. Useful behaviours and gotchas (collected)

### 18.1 The 1/2 coherency factor

`stokes_to_coherency` uses the `1/2` convention so that an unpolarised, 1 Jy source has `frame_coherency = diag(0.5, 0.5)` Jy. Some interferometer codes use a different normalisation — be sure your visibility simulator matches (RRIVis already uses the same convention, see `core/polarization.py`).

### 18.2 NaN and negative Stokes values

`check()` only warns (does not error) on NaN spectral indices, NaN Stokes values, or negative Stokes I. The intended workflow is to call `select(non_nan="any"|"all", non_negative=True)` before any downstream calculation that cannot handle them. `at_frequencies` for `subband` data has its own dedicated NaN-handling matrix (`propagate`/`interp`/`clip`).

### 18.3 GLEAM specifics

* GLEAM `subband` is the default spectral type because some bright sources have NaN spectral indices.
* The 30.72 MHz coarse channels are nominally `4 × 7.68 MHz`. The GLEAM catalog's published frequencies/edges round to 8 MHz spacing. Pyradiosky uses the catalog values by default; `use_paper_freqs=True` switches to the more physically motivated 7.68 MHz layout (centers off by ≤ 0.6 MHz, edges by ≤ 1.08 MHz).

### 18.4 HEALPix → point pixel naming

`healpix_to_point` assigns names `"nside{N}_{ring|nested}_{idx}"`, which is what `_point_to_healpix` and `assign_to_healpix` (when round-tripping) rely on for inversion. Don't rename them if you want the round-trip to succeed.

### 18.5 `astropy-healpix` vs `healpy`

pyradiosky deliberately uses **`astropy-healpix`** rather than `healpy`. All HEALPix calls (`HEALPix(...)`, `nside_to_pixel_area`, `interpolate_bilinear_skycoord`, `cone_search_lonlat`, `neighbours`, `skycoord_to_healpix`, `healpix_to_skycoord`) come from `astropy_healpix`. `healpy` is mentioned only as a comparison target for `healpix_interp_transform` accuracy.

### 18.6 Lunar telescope locations

Set `telescope_location` to a `lunarsky.MoonLocation` and `update_positions` will use `LunarTopo` and `lunarsky.SkyCoord` for the transform. The `_telescope_location.expected_type` is patched at runtime to `(EarthLocation, MoonLocation)` on first use. If `lunarsky` is not installed, anything other than `EarthLocation` raises a clear `ValueError`.

### 18.7 `frame` is required for HEALPix and point on init

Removed in v0.3.0: the implicit ICRS default. You must pass `frame=` when constructing from raw lon/lat or for HEALPix maps; `read_votable_catalog` and `read_text_catalog` carry the same constraint.

### 18.8 `extra_columns`

* Add via `add_extra_columns(names=..., values=..., dtype=None)`.
* Values must be Numpy 1-D arrays (Quantities are *not* supported — store the magnitude and the unit elsewhere).
* They survive `concat` if both objects share the same column names *and* dtypes.
* They survive `select`/`assign_to_healpix`/etc via `_select_along_param_axis`.
* They are written in skyh5 under `/Header/extra_columns/<name>`.

### 18.9 History strings

pyradiosky maintains a verbose, append-only `history` string. All readers append a per-format note (e.g., `" Read/written with pyradiosky version: <ver>."`), and all in-place mutators append a description. Use `history` to debug the exact provenance of any catalog you load.

### 18.10 `__getattr__` quirks

* `sm.frame` returns the frame *name* (string).
* `sm.ra`, `sm.dec`, `sm.l`, `sm.b`, etc. work — they fall through to `skycoord.<attr>` on point objects, *or* compute pixel center positions on HEALPix objects (with a warning encouraging `get_lon_lat()`).
* This means typos in attribute access on a `SkyModel` will produce unfamiliar tracebacks — the `__getattr__` chain ultimately delegates to `super().__getattribute__(name)`.

### 18.11 `select` history vs no-op

`select(...)` returns early (and does not modify `history`) when no constraints are set or all components pass the cuts. Plan around this if you expect a deterministic history-string on every call.

### 18.12 `concat` with `frame_coherency`

If both objects had a cached `frame_coherency`, it is concatenated; if only one has it, both are dropped to `None` with a warning. Always recompute via `calc_frame_coherency()` after a `concat` to be safe.

---

## 19. Worked examples (collected from `tutorial.rst`)

### 19.1 Read GLEAM and visualise

```python
import os
from pyradiosky import SkyModel
from pyradiosky.data import DATA_PATH

sm = SkyModel.from_file(
    os.path.join(DATA_PATH, "gleam_50srcs.vot"),
    with_error=True,
)
# sm.spectral_type == "subband", sm.Nfreqs == 20, sm.Ncomponents == 50
```

### 19.2 GSM-like HEALPix map → Galactic frame

```python
sm = SkyModel.from_file(os.path.join(DATA_PATH, "gsm_icrs.skyh5"))
# nside=8, hpx_order="ring", spectral_type="full", Nfreqs=10
sm_gal = sm.healpix_interp_transform("galactic", inplace=False)
```

### 19.3 Point ↔ HEALPix round trip

```python
sm_pt = sm.copy()
sm_pt.healpix_to_point(to_jy=True)
sm_back = sm_pt.assign_to_healpix(nside=8, order="ring", to_k=True)
np.testing.assert_allclose(sm_back.stokes, sm.stokes)
```

### 19.4 Coherency in local Alt/Az

```python
sm = SkyModel(
    name="offzen", skycoord=icrs_coord,
    stokes=[1.0, 0.2, 0, 0]*u.Jy, spectral_type="flat",
)
sm.calc_frame_coherency()
sm.update_positions(time, telescope_location)
local_C = sm.coherency_calc()  # shape (2, 2, Nfreqs, Nabove_horizon)
```

### 19.5 Generate an EoR-like HEALPix sky

```python
from pyradiosky.utils import flat_spectrum_skymodel
import numpy as np
sm = flat_spectrum_skymodel(
    variance=1e-6, nside=256,
    freqs=np.linspace(150e6, 180e6, 20),
)
# sm.history records the cosmologically-equivalent spectral amplitude in K²·Mpc³
```

### 19.6 Selection patterns

```python
# Restrict by sky region:
sm.select(lat_range=Latitude([-30, -20], u.deg),
          lon_range=Longitude([350, 10], u.deg))   # wraps through 0

# Restrict by flux at a specific frequency band:
sm.select(min_brightness=0.1*u.Jy, max_brightness=1*u.Jy,
          brightness_freq_range=[100, 200]*u.MHz)

# Drop NaN/negative components:
sm.select(non_nan="any", non_negative=True)

# Drop sources that never rise:
sm.cut_nonrising(telescope.lat)
```

---

## 20. Test fixtures shipped under `data/`

| File                                | Notes                                                                  |
|-------------------------------------|------------------------------------------------------------------------|
| `gleam_50srcs.vot`                  | 50-row GLEAM EGC slice with both flux and error columns.               |
| `simple_test.vot`                   | Minimal VOTable for `read_votable_catalog` tests.                      |
| `single_source_old.vot`             | Legacy single-source VOTable.                                          |
| `pointsource_catalog.txt`           | 3-source tab-separated catalog (frame `J2000`).                        |
| `mock_hera_text_2458098.27471.txt`  | Larger text catalog used in HERA-style tests.                          |
| `gsm_icrs.skyh5`                    | Coarse Global Sky Model HEALPix map in ICRS, `nside=8`, `Nfreqs=10`.   |
| `gsm_galactic.skyh5`                | Same as above, in Galactic frame.                                      |
| `healpix_disk.skyh5`                | Partial-sky HEALPix map.                                               |
| `old_skyh5_point_sources.skyh5`     | Legacy skyh5 layout (no `skycoord` group) for backwards-read testing.  |
| `fhd_catalog.sav`                   | Standard FHD point catalog.                                            |
| `fhd_catalog_no_extend.sav`         | Variant without an `extend` field.                                     |
| `fhd_catalog_with_beam_values.sav`  | FHD catalog including `BEAM` substructure for `beam_amp`.              |
| `fhd_catalog_bad.sav`               | Malformed FHD catalog used in error-path tests.                        |
| `fhd_source_array.sav`              | Catalog stored under the alternate `source_array` key.                 |
| `extended_source_test.sav`          | Extended-source FHD test fixture.                                      |

`pyradiosky.data.DATA_PATH` exposes the directory.

---

## 21. Release history (abridged from `CHANGELOG.md`)

* **Unreleased** — bumps `pyuvdata>=3.2.3`; `select` now uses `pyuvdata.UVBase._select_along_param_axis`. Console entry-point fix.
* **1.1.0 (2025-06-26)** — `select(non_nan=...)` and `select(non_negative=...)`, new check warnings for NaNs and negative Stokes I, `skip_params` for skyh5 reads. Bumps `pyuvdata>=3.1.0`, `lunarsky>=0.2.5`, `python>=3.11`. Fixed bug where `run_check=False` on read was ignored.
* **1.0.1 (2024-07-01)** — numpy 2.0 compat (`numpy.bytes_`, `np.isin`); fix `concat` when called serially; `setuptools_scm>=8.1`.
* **1.0.0 (2024-05-09)** — Bumps `astropy>=6.0`, `python>=3.10`, `pyuvdata>=2.4.3`.
* **0.3.1 (2024-02-22)** — SkyH5 memo added. `extra_columns` recarray. Optional arrays moved to `/Data` group. Fixed FHD frequency-units bug (was Hz, should be MHz).
* **0.3.0 (2023-04-10)** — Removed `frame` defaulting; introduced `freq_edge_array`. Removed legacy hdf5 (`read_healpix_hdf5`), `point_to_healpix` public, `to_recarray`/`from_recarray`, `ra_column`/`dec_column`, `source_cuts` method.
* **0.2.0 (2023-02-01)** — `skycoord` and `hpx_frame` attributes, full astropy-frame support, `calc_frame_coherency` method.
* **0.1.3 (2022-02-22)** — Generic `read`/`from_file`. `assign_to_healpix`. `nan_handling` for `at_frequencies`. `lat_range`/`lon_range`/`min_brightness`/`max_brightness` options. `cut_nonrising`.
* **0.1.2 (2021-07-06)** — `concat`. `stokes_error`. `clobber`. Ring/nested HEALPix.
* **0.1.1 (2021-02-17)** — SkyH5 introduced. `from_*` classmethods.
* **0.1.0 (2020-06-29)** — Quantity-only stokes; `jansky_to_kelvin` and inverse; `healpix_to_point`.
* **0.0.x (2020)** — Initial extraction from `pyuvsim`; `at_frequencies`; component_type.

Beyond ~v0.5 the deprecation warning on `freq_edge_array` for `subband` becomes a hard error.

---

## 22. Embedding pyradiosky in another simulator

Practical notes for integrators (and specifically for RRIVis-style consumers):

1. **Carry the unit**: `stokes` is a `Quantity`. For consistent precision (e.g. RRIVis' `PrecisionConfig`-driven dtype), call `.to_value(unit)` after extraction rather than relying on `.value`, and apply the user's chosen dtype via `.astype(precision.real_dtype)`.
2. **Decide the format up-front**: pyradiosky exposes both representations (`point`, `healpix`). Don't mix them in one tile of work — convert with `healpix_to_point()` / `assign_to_healpix()` (with explicit `to_k`/`to_jy` flags so the units stay correct) before passing arrays to the RIME core.
3. **Use `at_frequencies(...)` early**: collapsing to `spectral_type="full"` yields a clean `(4, Nfreqs, N)` Stokes cube that downstream code can treat as a plain array. Otherwise different code paths must handle four spectral types.
4. **For polarised work**: pre-call `calc_frame_coherency()` and rely on `coherency_calc()` only when you have a telescope location and time. The Alt/Az coherency depends on the source frame (RA/Dec vs Galactic), so transform first if you mix sky models in different frames.
5. **For ingest from VizieR-style catalogs without a dedicated reader**: use `read_votable_catalog` with substring-matched column names; pyradiosky tolerates the messy unit metadata in VOTable files via its `is_equivalent` checking.
6. **For long-running pipelines**: `concat`s of many catalogs *can* produce very long history strings — pass `verbose_history=False` (the default) to keep them bounded, or strip `history` after a final consolidation.
7. **HEALPix interpolation is not polarised**: `healpix_interp_transform` will refuse to operate on Q/U/V maps. Either work in the source frame, or convert to point and `transform_to`.
8. **Frame discipline**: the `frame` property is a *string* (just the name). For programmatic comparisons use `sm._get_frame_obj()` to get the actual `BaseCoordinateFrame` and compare with `astropy`'s frame-comparison semantics (e.g. equinox-aware for `fk5`).
9. **Lunar work**: pyradiosky already provides everything you need for moon-based simulators, but `lunarsky` must be installed for `MoonLocation` use.
10. **Empty catalog guardrails**: `select()` raises `ValueError("Select would result in an empty object.")` rather than producing an empty `SkyModel`. Wrap in try/except if your pipeline is allowed to produce empty regions.

---

## 23. Public surface (one-liner index)

### Class

`pyradiosky.SkyModel(...)` — the main object.

### Construction

* `SkyModel.from_file(filename, **kwargs)`
* `SkyModel.from_skyh5(filename, **kwargs)`
* `SkyModel.from_votable_catalog(votable_file, *args, **kwargs)`
* `SkyModel.from_gleam_catalog(gleam_file, **kwargs)`
* `SkyModel.from_text_catalog(catalog_csv, **kwargs)`
* `SkyModel.from_fhd_catalog(filename_sav, **kwargs)`

### Reading (instance methods)

* `read(filename, filetype=None, **kwargs)`
* `read_skyh5(...)`, `read_votable_catalog(...)`, `read_gleam_catalog(...)`, `read_text_catalog(...)`, `read_fhd_catalog(...)`

### Writing

* `write_skyh5(filename, clobber=False, data_compression=None, ...)`
* `write_text_catalog(filename)` (point + Jy + non-subband only)

### Shape / consistency

* `check(check_extra=True, run_check_acceptability=True)`
* `add_extra_columns(*, names, values, dtype=None)`
* `clear_time_position_specific_params()`
* `__eq__(other, check_extra=True, allowed_failures="filename", silent=False)`

### Coordinate / frame

* `transform_to(frame)` (point only)
* `healpix_interp_transform(frame, full_sky=False, inplace=True, ...)` (HEALPix only)
* `get_lon_lat()`

### Component-type interconversion

* `healpix_to_point(to_jy=True, ...)`
* `_point_to_healpix(to_k=True, ...)` (private)
* `assign_to_healpix(nside, order="ring", to_k=True, full_sky=False, sort=True, inplace=False, ...)`

### Units

* `kelvin_to_jansky()`
* `jansky_to_kelvin()`

### Frequency

* `at_frequencies(freqs, inplace=True, freq_interp_kind="cubic", nan_handling="clip", run_check=True, atol=None, ...)`

### Polarisation / coherency

* `calc_frame_coherency(store=True)`
* `coherency_calc(store_frame_coherency=True)`

### Time / location

* `update_positions(time, telescope_location)`
* `calculate_rise_set_lsts(telescope_latitude, horizon_buffer=0.04364)`
* `cut_nonrising(telescope_latitude, inplace=True, ...)`

### Selection / combination

* `select(component_inds=None, lat_range=None, lon_range=None, min_brightness=None, max_brightness=None, brightness_freq_range=None, non_nan=None, non_negative=False, inplace=True, ...)`
* `concat(other, clear_time_position=True, verbose_history=False, inplace=True, ...)`
* `copy()` (inherited from `UVBase`)

### Iteration helpers

* `ncomponent_length_params` *(property)*
* `_time_position_params` *(property)*

### Module-level utilities (`pyradiosky.utils`)

* `stokes_to_coherency(stokes_arr)`
* `coherency_to_stokes(coherency_matrix)`
* `jy_to_ksr(freqs)`
* `download_gleam(path=".", filename="gleam.vot", overwrite=False, row_limit=None, for_testing=False)`
* `flat_spectrum_skymodel(*, variance, nside, ref_chan=0, ref_zbin=0, redshifts=None, freqs=None, frame="icrs")`

### Spherical geometry primitives (`pyradiosky.spherical_coords_transforms`)

* `r_hat`, `theta_hat`, `phi_hat`
* `rotate_points_3d(rot_matrix, theta, phi)`
* `spherical_basis_vector_rotation_matrix(theta, phi, rot_matrix, beta=None, alpha=None)`
* `axis_angle_rotation_matrix(axis, angle)` (Rodrigues)
* `is_orthogonal(matrix, tol=1e-15)`
* `is_unit_vector(vec, tol=1e-15)`
* `vecs2rot(r1=None, r2=None, theta1=None, phi1=None, theta2=None, phi2=None)`

### CLI entry points

* `download_gleam` (in `pyradiosky.cli:download_gleam`)
* `make_flat_spectrum_eor` (in `pyradiosky.cli:make_flat_spectrum_eor`)

---

## 24. Maintainers and links

* Repo: <https://github.com/RadioAstronomySoftwareGroup/pyradiosky>
* Docs: <https://pyradiosky.readthedocs.io/en/latest/>
* JOSS: Hazelton *et al.* (2024), <https://doi.org/10.21105/joss.06503>
* Maintainers: Adam Beardsley (ASU), Bryna Hazelton (UW), Daniel Jacobs (ASU), Paul La Plante (UC Berkeley), Jonathan Pober (Brown).
* Contact: `rasgmanagers@gmail.com`.
* Funding: NSF #1835421 and #1835120.
