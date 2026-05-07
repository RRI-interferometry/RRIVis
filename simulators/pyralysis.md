# Pyralysis — Exhaustive Technical Reference

> **Source:** `simulators/pyralysis/` (git submodule)
> **Upstream:** https://gitlab.com/clirai/pyralysis
> **Docs:** https://pyralysis.readthedocs.io/
> **License:** GNU GPL v3.0-only (see `simulators/pyralysis/LICENSE`)
> **Lead:** Miguel Cárcamo (`miguel.carcamo@usach.cl`)
> **Latest local tag:** `v1.1.0` (2026-04-16). Prior: `v1.0.0` (2026-04-09).
> **Package version anchor:** `simulators/pyralysis/src/pyralysis/__init__.py` reads from `_version.py` written by `setuptools_scm`.

---

## 1. Overview

**Pyralysis** ("PYthon Radio Astronomy anaLYSis and Image Synthesis") is an OOP, large-scale Python framework for radio interferometric **simulation**, **imaging**, and **regularised maximum-likelihood (RML) reconstruction**. Its design choices come straight out of `simulators/pyralysis/CONVENTIONS.md` line 1:

> "Pyralysis … emphasis on Object-Oriented Programming (OOP), High-Performance Computing (HPC), and efficient handling of large-scale data using tools like Numba, CuPy, Dask, and Zarr."

The library covers four main surfaces:

| Surface | Role | Key entry points |
|---------|------|------------------|
| **I/O** | CASA Measurement Set (MS v2/v3) read/write via `dask-ms`; FITS via `dafits`/`astropy`; Zarr storage | `pyralysis.io.DaskMS`, `pyralysis.io.FITS`, `pyralysis.io.zarr` |
| **Data model** | Lazy `xarray.Dataset` wrappers around MS subtables and visibility partitions | `pyralysis.base.Dataset`, `SubMS`, `VisibilitySet`, `Antenna`, `Field`, `Polarization`, `SpectralWindow` |
| **Imaging / RML** | Gridding, weighting, dirty-image formation, measurement operator (Φ, Φᴴ), regularised optimisers | `pyralysis.transformers.*`, `pyralysis.optimization.*`, `pyralysis.reconstruction.*` |
| **Simulation** | Build interferometers + sky sources + noise injectors → synthesised visibilities → MS | `pyralysis.simulation.Simulator`, `pyralysis.models.sky`, `pyralysis.injectors.*` |

Languages: **100% Python**. No C/C++/CUDA in-tree (acceleration is via `numba` JIT, optional `cupy`, optional `pyfftw`, and Dask graph parallelism).

`simulators/pyralysis/pyproject.toml` line 21 sets `description = "PYthon Radio Astronomy anaLYSis and Image Synthesis"`. License declared SPDX `GPL-3.0-only` (line 59). Python support `>=3.9,<3.13` (line 65).

---

## 2. Repository Layout

Top-level (`simulators/pyralysis/`):

```
README.md  CHANGELOG.md  CONTRIBUTING.md  CONVENTIONS.md  DEVELOPMENT_GUIDE.md
LICENSE  MANIFEST.in  pyproject.toml  setup.cfg  pytest.ini  requirements.txt
environment.yml  environment_cudatoolkit.yml
Dockerfile  Dockerfile.prod
.readthedocs.yaml  .pre-commit-config.yaml  .gitlab/  .cursor/
binder/   - mybinder.org config (env + postBuild + README)
datasets/ - tracked test MS bundles (antennae, co65, FREQ78, M87, selfcalband9)
docs/     - Sphinx site (source/*.rst + diagrams/metauml/*.mp)
examples/ - scripts/ (CLI: components & pipeline pairs) + notebooks/ (7 ipynb)
scripts/  - generate_lofar_configs.py
src/pyralysis/   - actual package source (see §3)
tests/    - 193 .py files split into unit/ and integration/
```

Source tree under `simulators/pyralysis/src/pyralysis/`:

```
__init__.py          (re-exports __version__ from setuptools-scm)
base/                (Dataset and table facades — antenna, baseline, field, polarization,
                      spectral_window, observation, subms, visibility_set, basic_polarization,
                      derived_polarization, feedtype, correlation_set)
convolution/         (gridding kernels: bicubic, gaussian, gaussian_sinc, kaiser_bessel,
                      pillbox, pswf1, sinc, spline, c_kernel) + primary_beam/
dft/                 (idft2 — explicit DFT, numpy + numba parallel kernels)
estimators/          (degridding interpolators: nearest, bilinear, plus Degridding facade)
fft/                 (fft2/ifft2 + pluggable backends: numpy/dask, pyfftw)
flaggers/            (Flagger + threshold policies + SumThreshold operator)
grids/               (Grid value object: imsize/cellsize/padding/hermitian_symmetry)
injectors/           (noise injectors: gaussian, phase, thermal, gain, antenna_gain,
                      bandpass, composite)
io/                  (DaskMS, FITS, Zarr, antenna_config_io, coefficient_io, hdf5)
models/              (intensity/, temperature/, faraday/, noise/, sky/)
optimization/        (objective_function + terms/ + optimizer/ + linesearch/ + projection/)
pipelines/           (base + imager/ + simulation/)
reconstruction/      (Image, Parameter, PSF, Mask)
sanitizers/          (NaN/Inf cleaners for MS visibilities/weights)
simulation/          (Simulator, Interferometer, AntennaArray, builders/, core/filters/,
                      antenna_configs/*.cfg — 200+ ALMA/ACA/VLA/ATCA/SKA/MeerKAT/etc files)
systems/             (CoordinateSystem, TimeSystem)
transformers/        (DirtyMapper, Gridder, MeasurementOperator, Shifter, hsymmetry,
                      stokes/, weighting_schemes/)
units/               (Astropy unit helpers, beam_units)
utils/               (cellsize, coordinate_transforms, fits_utils, fourier/, gridding/,
                      interpolations/, polarization/, padding, parsers, noise,
                      decorators/memory, composition/combine_visibilities, …)
```

---

## 3. Installation, Build, Runtime

### 3.1 Build system

`simulators/pyralysis/pyproject.toml` lines 8-12:

```toml
[build-system]
build-backend = "setuptools.build_meta"
requires = ["setuptools>=70.0.0", "setuptools-scm>=8", "wheel"]
```

- Versioning: `setuptools_scm` writes `src/pyralysis/_version.py`; `local_scheme = "no-local-version"` (line 153).
- Package discovery: `[tool.setuptools.packages.find] where = ["src"]` (line 145).
- Bundled antenna `.cfg` files: `package-data` for `pyralysis.simulation.antenna_configs` (line 142).
- Dependencies are dynamic from `requirements.txt` (line 28, line 137).

### 3.2 Hard dependencies (`simulators/pyralysis/requirements.txt`)

| Package | Pin | Role |
|---------|-----|------|
| `numpy` | `>=2.0.0,<3.0.0` | Arrays |
| `scipy` | `==1.15.0` | Scientific routines |
| `dask` | `==2024.10.0` | Lazy graphs / parallel |
| `distributed` | `==2024.10.0` | Cluster / scheduler |
| `dask-ms[complete]` | `>=0.2.29` | xarray-style MAIN-table partitioning of MS |
| `xarray` | `==2024.9` | Labeled arrays |
| `astropy` | `==6.1.0` | Units, coords, time, FITS, WCS |
| `python-casacore` | `==3.7.1` | C++ casacore Python bindings (MS, TAQL) |
| `dafits` | `==2.0.0` | Dask-aware FITS HDU loading |
| `numba` | `>=0.59.1` | JIT (DFT inner kernel, gridding) |
| `numcodecs` | `==0.15.1` | Zarr codecs |
| `multimethod` | `==1.11.2` | Method overloading (used in `objective_function`, conversions) |
| `more-itertools` | `==10.2.0` | Misc iter utilities (e.g. `locate` in `dirty_mapper.py`) |
| `radio-beam` | `==0.3.7` | Restoring-beam fitting |
| `spectral-cube` | `==0.6.5` | FITS cube manipulation |
| `matplotlib` | `==3.10.8` | Plots |
| `ska-ost-array-config` | `==4.5.0` | SKA subarray definitions (private SKA PyPI index) |
| `snakeviz` | `==2.2.0` | Profiling viewer |

### 3.3 Optional extras (`pyproject.toml` lines 67-83)

| Extra | Adds |
|-------|------|
| `pyralysis[pyfftw]` | `pyfftw` FFT backend |
| `pyralysis[cupy]` | `cupy==13.2.0` for GPU arrays |
| `pyralysis[notebook]` | `cmcrameri`, `ipykernel`, `seaborn` |
| `pyralysis[test]` | `six`, `pytest==9.0.2`, `pytest-cov==7.1.0`, `pytest-xdist==3.8.0` |
| `pyralysis[all]` | Aggregate of the above (excluding `cupy`) |

### 3.4 Conda + pip workflow

`simulators/pyralysis/environment.yml` declares conda channels `conda-forge`/`defaults`, sets `PIP_EXTRA_INDEX_URL` to the SKA artefact registry, and pulls system-level binaries that pip cannot ship: `cfitsio`, `libboost-python-devel`, `casacore`. It then runs `pip install --extra-index-url … -r requirements.txt`. A separate `environment_cudatoolkit.yml` adds CUDA tooling for the CuPy path.

### 3.5 Containers

- `simulators/pyralysis/Dockerfile`: NVIDIA CUDA 12.9 + cuDNN base on Ubuntu 24.04, installs casacore-dev, cfitsio-dev, wcslib-dev, boost-python, plus `cupy-cuda12x`. Image is published to GitLab Container Registry.
- `Dockerfile.prod` is a slimmer release variant.

### 3.6 Quick install (per `README.md` line 53)

```bash
pip install --extra-index-url https://artefact.skao.int/repository/pypi-internal/simple pyralysis[all]
# GPU additionally:
pip install --extra-index-url https://artefact.skao.int/repository/pypi-internal/simple pyralysis[all,cupy]
```

The `--extra-index-url` is required to resolve `ska-ost-array-config`, which is hosted only on the SKA Observatory's internal PyPI mirror.

---

## 4. Architecture

### 4.1 Layered view

```
┌──────────────────────────────────────────────────────────────────────────┐
│ USER FACADES                                                             │
│  pyralysis.simulation.Simulator                                          │
│  pyralysis.pipelines.{ImagerPipeline, SimulationPipeline}                │
│  CLI examples (examples/scripts/*_components.py / *_pipeline.py)         │
├──────────────────────────────────────────────────────────────────────────┤
│ ALGORITHM LAYER                                                          │
│  Transformers : Gridder  → DirtyMapper                                   │
│                 MeasurementOperator (Φ, Φᴴ), Shifter, Stokes, hsymmetry  │
│  Optimisers   : LBFGS, ConjugateGradient (FR, PR, HZ, DY, HS, RMIL, LS), │
│                 FISTA, SDMM, ProximalOptimizer, GradientOptimizer        │
│  Terms        : ChiSquared, AmplitudeChiSquared, PhaseChiSquared,        │
│                 L1Norm, L2Norm, Tikhonov, Entropy, TotalFlux, Huber,     │
│                 TVIsotropic, TVAnisotropic, TSV, AugmentedTerm           │
│  LineSearch   : BacktrackingArmijo, GLLArmijo, Brent, Fibonacci,         │
│                 GoldenSection, Goldstein, Fixed, FISTABacktracking       │
│  Seeders      : BarzilaiBorwein (+ AdaptiveMin1/2, Alternating),         │
│                 Cubic / Quadratic Interpolation                          │
│  DFT/FFT      : pyralysis.dft.idft2 (numpy+numba); pyralysis.fft.fft2/ifft2
│                 with pluggable backend (numpy/dask, pyfftw)              │
│  Flaggers     : SumThreshold + ThresholdPolicy (Constant, InvSqrt, PowL) │
│  Sanitizers   : VisibilitySanitizer, WeightSanitizer, CompositeSanitizer │
├──────────────────────────────────────────────────────────────────────────┤
│ DATA MODEL                                                               │
│  pyralysis.base.Dataset (facade) ─aggregates─> Antenna, Baseline, Field, │
│      SpectralWindow, Polarization, Observation                           │
│  Dataset ─composes─> ms_list: List[SubMS]                                │
│  SubMS    ─composes─> VisibilitySet (xarray.Dataset of MAIN columns)     │
│  Reconstruction model: Image extends Parameter; Mask, PSF                │
├──────────────────────────────────────────────────────────────────────────┤
│ HARDWARE / I/O                                                           │
│  Dask graphs (CPU / Dask-Distributed)                                    │
│  Numba @njit(parallel=True) inner kernels (DFT, gridding reductions)     │
│  Optional CuPy arrays (extras)                                           │
│  pyfftw FFT backend (optional)                                           │
│  Zarr persistence + numcodecs                                            │
│  python-casacore + dask-ms for MS v2/v3 read/write                       │
└──────────────────────────────────────────────────────────────────────────┘
```

UML class/composition diagrams are authored in MetaUML/MetaPost under `simulators/pyralysis/docs/source/diagrams/metauml/*.mp` and rendered to PNG in `…/diagrams/png/`. `Dataset`'s docstring (`base/dataset.py` line 36) explicitly cites `fig01_dataset_composition.mp` for the aggregation/composition split between table facades and `SubMS`.

### 4.2 xarray + Dask data flow

```
                   CASA MS on disk (v2 or v3)
                            │
                            │  pyralysis.io.DaskMS.read(chunks=…, sanitize=…)
                            ▼
                ┌──────────────────────────────────┐
                │ daskms.xds_from_ms / xds_from_table │
                │ → list of xarray.Dataset partitions │
                └────────────────┬─────────────────┘
                                 ▼
            CompositeSanitizer (NaN/Inf scrub, optional)
                                 ▼
   Subtables (ANTENNA, FIELD, SPECTRAL_WINDOW, POLARIZATION, OBSERVATION)
   wrapped in pyralysis.base.{Antenna, Field, SpectralWindow,
   Polarization, Observation}
                                 ▼
   MAIN partitions wrapped in pyralysis.base.SubMS (one per FIELD × DDID)
   each holding a pyralysis.base.VisibilitySet(dataset: xarray.Dataset)
                                 ▼
                   pyralysis.base.Dataset (facade)
                                 ▼
        Transformers / Optimisers / Injectors / Sanitizers
        (operate on lazy dask.array views inside xarray.Dataset)
                                 ▼
              dask.compute() on demand → numpy / cupy
                                 ▼
                FITS, MS write, or in-memory reconstruction Image
```

Lazy evaluation runs end-to-end: `Dataset.total_sum_imaging_weights`, `total_sum_squared_imaging_weights`, `total_visibility_number` are all `@cached_property` returning `dask.array.Array` (`base/dataset.py` lines 530-587).

---

## 5. Public API — module by module

> Cite paths anchored at `simulators/pyralysis/src/pyralysis/`.

### 5.1 `pyralysis.base` — data model

`base/__init__.py` exports: `Antenna`, `Baseline`, `BasicPolarization`, `CorrelationSet`, `Dataset`, `DerivedPolarization`, `FeedType`, `Field`, `Observation`, `Polarization`, `SpectralWindow`, `SubMS`, `VisibilitySet`.

#### `Dataset` (`base/dataset.py`)

`@dataclass` facade. Constructor parameters (lines 60-69):

```python
@dataclass
class Dataset:
    antenna: Optional[Antenna] = None
    baseline: Optional[Baseline] = None
    field: Optional[Field] = None
    spws: Optional[SpectralWindow] = None
    polarization: Optional[Polarization] = None
    observation: Optional[Observation] = None
    ms_list: Optional[List[SubMS]] = None
    do_psf_calculation: bool = True
    corrected_column_present: bool = False
```

Auto-runs `calculate_psf()` and `_create_antenna_primary_beam()` in `__post_init__` (line 76). Notable members:

| Member | Kind | Purpose |
|--------|------|---------|
| `STOKES_PARAMS = ['I','Q','U','V']` | class attr | Default Stokes basis |
| `max_baseline`, `min_baseline` | property | Delegated to `Baseline` |
| `max_antenna_diameter` / `min_antenna_diameter` | property | Delegated to `Antenna` |
| `theo_resolution` | property | `λ_min / max_baseline` (radians) |
| `fov` | property | `λ_max / min_antenna_diameter` |
| `psf` / `psf.setter` | property | Lazy `PSF` instance |
| `calculate_weights_sum(stokes=False)` | method | Per-correlation or per-Stokes weight sums (`utils.noise.calculate_weights`) |
| `calculate_psf(stokes=None)` | method | `PSF.from_dataset(self, stokes)` |
| `calculate_theoretical_noise(per_field=False, per_spw=False)` | method | Group by FIELD_ID / SPW_ID; returns dict or list of dict |
| `filter_by_field(field_ids)` / `filter_by_spw(spw_ids)` / `filter_by_polarization(pol_ids)` / `select_subms(indices)` | method | Return new `Dataset` with shared subtables but filtered `ms_list` |
| `total_sum_imaging_weights`, `total_sum_squared_imaging_weights`, `total_visibility_number` | `@cached_property` | Aggregations (lazy `dask.array`) |
| `clear_all_caches()`, `clear_coordinate_caches()`, `clear_visibility_caches()`, `memory_usage()`, `memory_efficient_mode()` | method | Memory ops |
| `__iter__()` | dunder | Yields `("antenna", Antenna), …` for I/O round-trips |

#### `SubMS` and `VisibilitySet`

`SubMS` carries `field_id`, `spw_id`, `polarization_id`, and a `VisibilitySet` (`base/visibility_set.py`). `VisibilitySet` wraps an `xarray.Dataset` of MAIN-table columns (`DATA`, `MODEL_DATA`, `CORRECTED_DATA`, `RESIDUAL_DATA`, `FLAG`, `WEIGHT`, `WEIGHT_SPECTRUM`, `IMAGING_WEIGHT_SPECTRUM`, `UVW`, `ANTENNA1`, `ANTENNA2`, `BASELINE_ID`, …). It exposes:

- `get_weight_column(as_xarray)` — broadcast 2D `WEIGHT` to 3D when needed.
- `create_hermitian_symmetric_data(use_imaging_weights)` — duplicates rows with conjugated UVW for full-plane gridding.
- Cached lazy aggregates `sum_imaging_weights`, `sum_squared_imaging_weights`, `visibility_number`.

#### `Polarization`

Holds `correlation_sets: List[CorrelationSet]`, `feed_kind` ∈ `{"linear", "circular", "mixed"}`, `ncorrs[pol_id]`, `max_ncorrs`. `BasicPolarization` and `DerivedPolarization` provide canonical/derived definitions; `FeedType` enumerates feed bases.

#### Other facades

`Antenna` (positions, diameter, primary-beam binding), `Field` (phase/delay centres, `table_keywords`/`column_keywords`), `SpectralWindow` (per-SPW frequencies, channel widths, `lambda_min`/`lambda_max`), `Baseline` (auto/cross), `Observation` (telescope identity, `ntelescope`).

### 5.2 `pyralysis.io` — I/O

`io/__init__.py`: exports `DaskMS`, `FITS`, `Io`, `ZarrArray`, `ZarrDataset`. `coefficient_io` and `hdf5` import on demand.

#### Abstract `Io` (`io/io.py`)

`@dataclass` ABC with `input_name`/`output_name`, abstract `read()`/`write()`, plus helpers `is_ready`, `can_read`, `can_write`, `exists()`, `ensure_extension()`, `_serialize_dictionary()`.

#### `DaskMS` (`io/daskms.py`, 818 lines)

```python
@dataclass
class DaskMS(Io):
    chunks: Union[dict, List[dict]] = None
    # default _DEFAULT_ROW_CHUNKS = 10000
```

Key methods:

| Method | Description |
|--------|-------------|
| `_detect_ms_version()` | Returns 2 or 3 based on presence of `FIELD_ID`/`SPW_ID` in MAIN |
| `read(read_flagged_data=False, filter_flag_column=False, calculate_psf=True, taql_query=None, chunks=None, sanitize=False, sanitizer=None) → Dataset` | Lazy MS load; partitions MAIN by field/DDID, builds all facades |
| `write(dataset, ...)` | Writes back to MS, optionally only specified subtables |
| `_create_data_description_table(dataset)` | Synthesises DATA_DESCRIPTION from `ms_list` SPW/POL pairs |
| `_write_subtables(dataset, ms_name, tables_to_write, dask_compute)` | Iterates ANTENNA, FIELD, SPECTRAL_WINDOW, POLARIZATION, OBSERVATION |
| `write_xarray_ds(...)` | Wrapper around `daskms.xds_to_table` |

The default sanitiser is `CompositeSanitizer()` (visibility + weight). Read pipeline (lines 320-365 paraphrased):

> "MS v2 uses `DATA_DESC_ID` and lookup tables; MS v3 uses explicit `SPW_ID`/`POL_ID`. … When `sanitize=True`, compact 2D `WEIGHT`/`SIGMA` invalid values are not broadcast into the 3D `FLAG` column; only spectra-shaped columns participate in flag updates."

#### `FITS` (`io/fits.py`)

```python
@dataclass
class FITS(Io):
    hdu: int = 0
    memmap: bool = True
    chunks: Union[int, tuple, str] = "auto"
    lazy_load_hdus: bool = False
    use_dask: bool = True
    preserve_attributes: bool = True
    auto_parse_header: bool = True
```

Reads via `dafits` to a Dask array, infers cellsize (`utils.fits_utils.parse_fits_cellsize`), phase centre (`parse_fits_phase_center`), centre pixel (`parse_fits_center_pixel`), and Stokes axis (`utils.polarization.parse_stokes_from_fits_header`). Returns an `Image` (or `Parameter`) bound to those WCS attributes. Writes back with `astropy.io.fits` honouring `bitpix` derived via `get_bitpix_from_dtype`.

#### `AntennaConfigurationIo` (`io/antenna_config_io.py`)

`@dataclass(Io)` that parses CASA `.cfg` files (header `# observatory=`, `# coordsys=ENU/XYZ/ITRF`, then x y z [diameter] [id]), maps to `EarthLocation`, runs `earth_location_to_local_enu`, and returns an `Interferometer` populated with `AntennaArray`, `CoordinateSystem`, `TimeSystem`. Bundled `.cfg` library lives at `src/pyralysis/simulation/antenna_configs/` and ships ALMA cycle 0–10, ACA, ATCA, CARMA, KAT-7, LOFAR HBA/LBA, MeerKAT, ngVLA rev B/C/D, PdBI, SKA-LOW, SKA-MID, SMA, VLA A/B/BNA/C/CNB/D/DNC, VLBA, WSRT (≈200 files; see `[tool.setuptools.package-data]` glob `"*.cfg"`).

#### Zarr (`io/zarr/`)

`ZarrArray` (single-array persistence with `zarr_compressor` config) and `ZarrDataset` (multi-array dataset). Used by `LBFGS` for spilling correction pairs to disk (`use_disk_storage=True` + `disk_storage_path`).

### 5.3 `pyralysis.transformers` — imaging core

`transformers/__init__.py` re-exports everything from `dirty_mapper`, `gridder`, `hsymmetry`, `measurement_operator`, `shifter`, `transformer`, `wscheme`. Submodule `weighting_schemes/` adds weighting subclasses.

#### `Transformer` (`transformers/transformer.py`)

`@dataclass` ABC: holds `input_data`, `output_data`, defines `transform()` to be overridden.

#### `Gridder` (`transformers/gridder.py`)

```python
@dataclass
class Gridder(Transformer):
    image: Image = None
    ckernel_object: CKernel = None
    use_zarr: Optional[Union[bool, str]] = None
    zarr_store_path: Optional[str] = None
    memory_threshold_gb: float = 10.0
    grid: Optional[Grid] = None
    # InitVars: imsize, cellsize, padding_factor=1.2, hermitian_symmetry
```

If `grid=` is passed it is reused; otherwise a `Grid(imsize, cellsize, padding_factor, hermitian_symmetry)` is built. Delegates `get_imsize/cellsize/padded_imsize/grid_size/uvcellsize/...` to the `Grid`. `transform()` is left abstract for subclasses.

#### `Grid` (`grids/base.py`)

Value object that holds and validates grid geometry. Provides `padded_imsize`, `padded_grid_size` (Hermitian-aware), `uvcellsize` (in wavelengths), and `calculate_uv_pix(uvw_nwavelengths, hermitian_symmetry, use_rounding)` returning per-baseline `(u_pix, v_pix)`.

#### `DirtyMapper` (`transformers/dirty_mapper.py`)

`@dataclass(init=False)` extending `Gridder`. Builds dirty image and dirty beam per Stokes:

```python
DirtyMapper(stokes=["I","Q","U","V"], data_column="CORRECTED_DATA",
            chunks=…, grid=…|imsize=, cellsize=,
            padding_factor=, hermitian_symmetry=, ckernel_object=)
```

State exposed: `uvgridded_visibilities` (complex64 dask array, shape `(n_stokes, *padded_imsize)`), `uvgridded_weights` (float32), private `__uvgridded_weights_original`, `__sum_wt_uv_full`, `sum_wt`, `max_psf`. Pipeline (per docstring lines 19-50): always grid on the full padded UV plane with duplicated Hermitian rows; if `hermitian_symmetry=True`, compress to rfft width via `ifftshift` + `[..., :N//2+1]` before `irfft2`.

#### `MeasurementOperator` (`transformers/measurement_operator.py`, 1568 lines)

```python
@dataclass(init=False)
class MeasurementOperator(Gridder):
    intensity_model: Optional[IntensityModel]
    update_residual_data: bool = False
    # plus all Gridder kwargs
```

Implements the forward operator Φ that maps an `Image` model to model visibilities at the dataset's UV samples:

```
V(u,v) = ∫∫ I(l,m) · A(l,m) · exp(-2πi(u l + v m)) dl dm
```

Steps (per docstring lines 36-52):

1. Image preparation: padding (`utils.padding.apply_padding`), primary-beam multiplication (`Antenna.primary_beam`), intensity scaling via `IntensityModel`.
2. FFT (`pyralysis.fft.fft2` — pluggable backend).
3. Phase shifting via `utils.fourier.phase_shift_grid`.
4. Visibility estimation by `Estimator` (nearest, bilinear, or full degridding kernel). The `transform()` hook is concretised in subclasses — `estimators/Degridding`, `BilinearInterpolation`, `NearestNeighbor` plug in here.

`update_residual_data=True` writes `RESIDUAL_DATA = DATA - MODEL_DATA` after each forward pass; it is off by default to avoid graph blow-up in optimisation loops.

#### `Shifter` (`transformers/shifter.py`)

Phase-rotates visibilities to a new pointing/phase centre via `e^{-2πi(uΔl + vΔm + w(Δn-1))}`.

#### `WeightingScheme` (`transformers/wscheme.py`) and concrete schemes

`WeightingScheme(Transformer)` requires a `Grid` (or builds one) and provides `grid_weights()` which performs rfft-friendly UV binning via `utils.gridding.grid_weights_reduction`. Concrete subclasses in `weighting_schemes/`:

| Class | File | Notes |
|-------|------|-------|
| `Natural` | `natural.py` | Grid optional |
| `Radial` | `radial.py` | Grid optional |
| `Uniform` | `uniform.py` | Grid required |
| `Robust` | `robust.py` | Grid required; takes `robust_parameter` (Briggs) |
| `UVTaper` | `uvtaper.py` | Gaussian taper with `sigma: u.Quantity[m]`; orthogonal to gridded schemes |

#### `hsymmetry`

Helpers for folding a full UV plane into the rfft non-redundant half-plane and back.

#### `transformers/stokes/`

`StokesConverter` performs correlation ↔ Stokes conversion based on `Polarization.feed_kind` and `CorrelationSet`.

### 5.4 `pyralysis.dft` & `pyralysis.fft`

#### DFT (`dft/_idft.py`)

```python
def idft2(x, y, uvw, vis, wt=None,
          sign_convention="negative", numba=False,
          sum_over=None) -> da.Array:
    """Imaging IDFT including the w-term:
       I(l,m) = Σ_k w_k V_k exp(2πi(u_k l + v_k m + w_k(√(1-l²-m²)-1)))
    """
```

- Built on `da.blockwise` with axes `(row, chan, corr, idx)`. Always sums over rows (baselines) first.
- `_idft2` does the dense numpy einsum kernel (`"i,rc->irc"`); `_idft2_numba` is a `@njit(parallel=True, cache=True)` parallel pixel/channel version.
- `sum_over` accepts `"both" | "channels" | "stokes" | "none"` or a tuple of axes; controls whether to collapse channels/correlations after the row reduction.

#### FFT (`fft/_fft2.py` + `fft/backends.py`)

Pluggable backend registry:

```python
class FFTBackend(ABC):
    fft2 / ifft2 / rfft2 / irfft2 / fftshift / ifftshift
```

`NumpyDaskBackend` is always available and dispatches on `isinstance(a, da.Array)` to `da.fft.*` else `np.fft.*`. Optional `PyFFTWBackend` provides a wisdom-cached pyfftw plan (selected via env var or `get_backend("pyfftw")`). `list_available_backends()` enumerates registered backends. Higher-level `fft2`/`ifft2` consult the active backend.

`fft/_chunking.py`, `fft/_normalization.py`, and `fft/_storage.py` handle dask-specific re-chunking, FFT normalisation conventions, and zarr-spilling for large transforms.

### 5.5 `pyralysis.optimization`

Top-level export: `ObjectiveFunction`. Submodules: `terms/`, `optimizer/`, `linesearch/`, `projection/`.

#### `ObjectiveFunction` (`optimization/objective_function.py`)

```python
@dataclass
class ObjectiveFunction:
    terms: List[ObjectiveFunctionTerm] = field(default_factory=list)
    phi: float = 0.0
    dphi: Union[np.ndarray, da.Array] = None
    parameter: InitVar[Union[Parameter, tuple, list]] = None
    chunks: InitVar[Union[tuple, dict]] = None
    persist_gradient: bool = False
```

Methods (signatures and roles):

| Method | Purpose |
|--------|---------|
| `set_terms_parameter(parameter)` | Push the same `Parameter` into every term |
| `configure_parameter_size(shape, chunks=None)` | Allocate `dphi` |
| `_process_visibility_terms()` | Calls `term.model_visibility.transform()` once per unique `MeasurementOperator` |
| `calculate_function(*, mask=None, differentiable_only=False, nondifferentiable_only=False) → float` | Sum of `term.function()` (filterable into smooth-only `f` or non-smooth-only `g` — used by FISTA backtracking) |
| `calculate_gradient(*, mask=None)` | Sum of `term.gradient()` into `dphi` |
| `term_values` | `np.ndarray` of unpenalised per-term values |

Uses `multimethod.multimethod` for overloading on `Parameter` vs raw shape.

#### Terms (`optimization/terms/`)

Base: `ObjectiveFunctionTerm` (data fidelity + regularisation interface), `VisibilityTerm` (subset that owns a `MeasurementOperator`), `AugmentedTerm` (SDMM Lagrangian wrapper).

Visibility terms (`terms/visibility_terms/`):

- `ChiSquared` — `½ Σ w |V_obs - V_model|²` (real-space residual chi-square).
- `AmplitudeChiSquared` — fits |V| amplitudes only.
- `PhaseChiSquared` — fits arg(V) only.

Regularisers (`terms/regularizers/`): `L1Norm`, `L2Norm`, `Tikhonov`, `Entropy`, `TotalFlux`, `Huber`, `TVIsotropic`, `TVAnisotropic` (`total_variation/`), `TSV` (Total Squared Variation).

Normalisation (`terms/normalization/`): `NoNormalization`, `SumWeightsNormalization`, `VisibilityNumberNormalization`, `EffectiveSamplesNormalization`, plus a `NormalizationFactory`, `NormalizationStrategy`, `NormalizationScope` registry.

#### Optimisers (`optimization/optimizer/`)

| Class | File | Algorithm |
|-------|------|-----------|
| `Optimizer` (ABC) | `optimizer.py` | Holds `objective_function`, `linesearch`, `parameter`, `mask`, `projection`, `eps`, `ftol`, `max_iter`, `io_handler` |
| `GradientOptimizer` | `gradient.py` | Steepest descent base |
| `ConjugateGradient` | `conjugate_gradient.py` | + variants `FletcherReeves`, `PolakRibiere`, `HestenesStiefel`, `DaiYuan`, `LiuStorey`, `HagerZhang`, `RMIL`. Powell restart η = 0.2. Convention: `search_direction = -dphi`. |
| `LBFGS` | `lbfgs.py` | Two-loop recursion; `max_corrections=0` ⇒ steepest descent. Disk spill via zarr (`use_disk_storage`, `in_memory_limit`, `disk_storage_path`). `clear_on_curvature_failure`, `memory_threshold_gb`. History managed by `LBFGSHistory` (`history.py`) with `real_scalar_finfo_eps` safety floors. |
| `ProximalOptimizer` | `proximal.py` | Base for proximal gradient methods |
| `FISTA` | `fista.py` | Standard or monotone (MFISTA) via `monotonic` flag; expects exactly one non-smooth term |
| `SDMM` | `sdmm.py` | Simultaneous Direction Method of Multipliers; takes `rho`, an inner `optimizer`, `primal_tol`, `dual_tol`, `persist_z_u` |
| `GradientNormError` | `conjugate_gradient.py` | Raised when ‖∇f‖ → 0 unexpectedly |

Line search (`optimization/linesearch/`): `BacktrackingArmijo`, `GLLArmijo` (Grippo-Lampariello-Lucidi non-monotone), `Brent`, `Fibonacci`, `GoldenSection` (and a `golden_section/` package with `bracketing.py`, `constants.py`, `search.py`), `Goldstein`, `Fixed`, `FISTABacktracking`, plus a `bracketing/` helper.

Step seeders (`linesearch/seeders/`): `BarzilaiBorwein` and adaptive variants `BarzilaiBorweinAdaptiveMin1`, `BarzilaiBorweinAdaptiveMin2`, `BarzilaiBorweinAlternating`; polynomial models `CubicInterpolationSeeder`, `QuadraticInterpolationSeeder`. Base ABC `StepSizeSeeder`.

Projection (`optimization/projection/`): `Projection` (e.g. positivity), `CompositeProjection`.

### 5.6 `pyralysis.reconstruction`

| Class | File | Role |
|-------|------|------|
| `Parameter` | `parameter.py` (986 lines) | Generic optimisation parameter: `data: dask.array`, `cellsize`, `chunks`, gradient API, FITS-aware methods |
| `Image` | `image.py` (1349 lines) | `Parameter` subclass: WCS, `phase_center: SkyCoord`, `center_pixel`, `transform_intensity_units(...)`, beam-area conversions, restoring beam |
| `Mask` | `mask/mask.py` | Boolean mask parameter |
| `PSF` | `psf.py` (602 lines) | Analytic PSF from UV weights. Factory `PSF.from_dataset(dataset, stokes=None)` |

`Image.empty(imsize, cellsize)` creates an empty initial image (used in the README minimal example).

### 5.7 `pyralysis.simulation`

`simulation/__init__.py`: `AntennaArray`, `Interferometer`, `Simulator`. Submodules `antenna_configs/` (CASA `.cfg` library), `builders/dataset_builder.py`, `core/` (Simulator, Interferometer, AntennaArray, filters/, utils.py).

#### `Interferometer` (`simulation/core/interferometer.py`)

Aggregates `AntennaArray`, `CoordinateSystem`, `TimeSystem`, primary beam, `frequencies`, `pointing_position`, `telescope`. Methods include `configure_observation(min_frequency_hz, max_frequency_hz, frequency_step_hz, right_ascension, declination, integration_time, observation_time)` (see README example).

#### `Simulator` (`simulation/core/simulator.py`)

```python
@dataclass
class Simulator:
    interferometer: Interferometer
    chunks: Optional[…] = None
    single_field: bool = True
    field_name: Optional[str] = None
    feed_type: str = "linear"   # or "circular"
    sources: InitVar = None     # → normalised into self.source: Source|CompositeSource
```

Methods:

- `add_source(source: Source)` / `add_composite_source(sources, name)`
- `calculate_uvw_coverage() → (uvw_metres, uvw_wavelengths)`
- `calculate_visibility(...)` and `simulate(create_dataset=True)` orchestrating `DatasetBuilder`
- Internal state `_visibility`, `_uvw_coverage`, `_uvw_metres`, `_dataset`

Antenna filters (`simulation/core/filters/`): `RadiusFilter`, `DiameterFilter`, `IdFilter`, `CompositeFilter` for subarray selection.

SKA helpers (`simulation/antenna_configs/ska_assemblies.py`) wrap `ska-ost-array-config` to produce SKA-LOW / SKA-MID subarray definitions.

### 5.8 `pyralysis.models`

`models/__init__.py` re-exports `IntensityModel`, `PowerLawIntensityModel`, `ConstantTemperatureModel`, `PowerLawSkyTemperatureModel`, `TemperatureModel`.

#### `models.intensity` (`models/intensity/`)

`IntensityModel` ABC with `evaluate(frequency)`. `PowerLawIntensityModel(reference_intensity, reference_frequency, spectral_index, spectral_curvature)` implements `S(ν) = S₀ (ν/ν₀)^(α + β log(ν/ν₀))`.

#### `models.temperature` (`models/temperature/`)

`TemperatureModel` ABC. `ConstantTemperatureModel(temperature)` and `PowerLawSkyTemperatureModel(reference_temperature, reference_frequency, spectral_index)`. Used by thermal noise injectors and `compute_thermal_noise` in `utils.noise`.

#### `models.faraday` (`models/faraday/`)

```
FaradayComponent (ABC)
├─ ThinFaradayComponent(rm)
├─ ThickFaradayComponent(rm_centre, rm_width, kind: ThickKind ∈ {SINC, GAUSSIAN})
└─ FaradayComposite (sum of components, supports `+` operator)
```

Source faraday is set via `Source.add_faraday_component(faraday)`; allows λ²-dependent rotation models (thin screens, thick slabs).

#### `models.noise` (`models/noise/`)

`GaussianNoiseModel`, `PhaseNoiseModel`, `BandpassNoiseModel`, `ComplexGainNoiseModel`, `CompositeNoiseModel`. Many also have a `LogNormalNoiseModel` (`lognormal.py`).

#### `models.sky` (`models/sky/`)

| Class | File |
|-------|------|
| `Source` (ABC) | `source.py` |
| `PointSource` | `point_source.py` |
| `GaussianSource` | `gaussian_source.py` |
| `NonParametricSource` | `nonparametric_source.py` |
| `CompositeSource` | `composite_source.py` |

Common `Source` constructor (per docstring `source.py` lines 27-92):

```python
Source(name=None, sky_position="HH:MM:SS DD:MM:SS"|SkyCoord|tuple,
       direction_cosines=(l,m[,n])|None,
       intensity_model: IntensityModel = PowerLawIntensityModel(...),
       faraday_component: FaradayComponent = None,
       reference_intensity, spectral_index, spectral_curvature, reference_frequency=1.0,
       phase_center=None)
```

If `phase_center` is omitted, it defaults to `sky_position`. Direction cosines are computed via `utils.coordinate_transforms.radec_to_direction_cosines(ra, dec, phase_center)`.

### 5.9 `pyralysis.injectors`

`injectors/__init__.py`: `GaussianNoiseInjector`, `PhaseNoiseInjector`, `ThermalNoiseInjector`, `CompositeNoiseInjector`, `BandpassNoiseInjector`, `AntennaGainInjector`, `GainNoiseInjector`. All extend a base `Injector` interface (see `injectors/base.py`) with an `apply(dataset) → Dataset` operation.

`ThermalNoiseInjector(system_temperature, integration_time, channel_bandwidth, antenna_efficiency=…, …)` adds Gaussian noise calibrated by `σ = SEFD/(η_a · √(2 Δν Δt))`, drawing from `utils.noise.calculate_noise_from_weights`.

### 5.10 `pyralysis.flaggers`

```
ThresholdPolicy  ← base (`models/threshold.py`)
├─ ConstantThreshold
├─ InverseSqrtThreshold
└─ PowerLawThreshold

SumThreshold (`operators/sum_threshold.py`) — Offringa et al. SumThreshold RFI flagger
Flagger — facade combining policy + operator
```

### 5.11 `pyralysis.sanitizers`

`Sanitizer` ABC defines `Action` enum (drop / fill / flag) and `InvalidKind` (NaN / Inf / Negative). `VisibilitySanitizer` and `WeightSanitizer` operate on the corresponding xarray columns; `CompositeSanitizer` chains them. `ColumnSanitizationStats` and `SanitizationResult` are dataclasses returned for diagnostics.

### 5.12 `pyralysis.estimators`

Visibility-from-grid estimators used inside `MeasurementOperator`:

- `NearestNeighbor` (`nearest.py`)
- `BilinearInterpolation` (`bilinear_interpolation.py`)
- `Degridding` (`degridding.py`) — full kernel degridder taking a `CKernel`

### 5.13 `pyralysis.convolution`

Gridding kernels (each subclass of `CKernel`):

| Class | File | Use |
|-------|------|-----|
| `Bicubic` | `bicubic.py` | 4×4 cubic |
| `Gaussian` | `gaussian.py` | Σ truncated Gaussian |
| `GaussianSinc` | `gaussian_sinc.py` | Standard CASA AW kernel |
| `KaiserBessel` | `kaiser_bessel.py` | Tunable β |
| `Pillbox` | `pillbox.py` | Constant cell support |
| `PSWF1` | `pswf1.py` | First-order prolate spheroidal |
| `Sinc` | `sinc.py` | Pure sinc |
| `Spline` | `spline.py` | B-spline |

Primary beam: `convolution/primary_beam/` provides `PrimaryBeam`, `PrimaryBeamStrategy`, coefficient tables (`coefficient_data.py`, `coefficients.py`, `coeffs/`). `Antenna.create_primary_beam(observation, antenna_obs_id)` (`base/dataset.py:154-176`) hooks them up.

### 5.14 `pyralysis.pipelines`

Step pattern (`pipelines/base.py`):

```python
class Step(Generic[T], ABC):
    @abstractmethod
    def execute(self, context: T) -> T: ...

class Pipeline(Generic[T]):
    steps: Sequence[Step[T]]
    def run(self, context: T) -> T: ...
```

Concrete:

- **`ImagerPipeline`** (`pipelines/imager/`) — default order: `LoadData → ApplyVisibilityWeighting → SetupImageGrid → SetupMeasurementOperator → BuildObjective → RunOptimizer → FormDirtyImage → RestoreImage → ExportResults`. Optional `ValidateContext` step. Context: `ImagerContext`.
- **`SimulationPipeline`** (`pipelines/simulation/`) — analogous flow for synthesis. Context: `SimulationContext`.

### 5.15 `pyralysis.systems`

`CoordinateSystem` (handles ENU↔ITRF↔ECEF conversions on top of `astropy.coordinates`) and `TimeSystem` (UTC↔TAI↔LST helpers built on `astropy.time.Time`).

### 5.16 `pyralysis.units`

`beam_units.beam_equivalencies` (Jy/beam ↔ Jy/sr ↔ K), `units_functions.{freq_to_wavelength, uvw_meters_to_nwavelengths, check_units}`.

### 5.17 `pyralysis.utils`

| Module | Notable functions |
|--------|-------------------|
| `array_ops.py` | `sign`, `soft_threshold` (proximal operator helper) |
| `cellsize.py` | `cellsize()` validator: float ⇒ rad, `Quantity` accepted |
| `coordinate_transforms.py` | `calculate_lm`, `calculate_lmn`, `calculate_radec`, `direction_cosines_to_radec`, `radec_to_direction_cosines` |
| `earth_coordinates.py` | `earth_location_to_local_enu` |
| `fits_utils.py` | `parse_fits_cellsize`, `parse_fits_phase_center`, `parse_fits_center_pixel`, `get_bitpix_from_dtype` |
| `image_utils.py` | `get_coordinates(imsize, cellsize, …)` returns l,m grids |
| `padding.py` | `apply_padding`, `remove_padding`, `calculate_padding` |
| `polarization/` | `polarization_conversion.{convert_data_array, convert_weight_array, convert_flag_array}`, `stokes.{CORRELATION_TO_STOKES, VALID_STOKES_LABELS, calculate_stokes_weights_from_correlations, default_stokes_labels, map_stokes_values_to_labels, parse_stokes_from_fits_header, resolve_image_stokes_labels}`, `visibility_polarization.{get_correlation_to_stokes_matrix_for_ms, resolve_requested_stokes, get_adapter}` |
| `gridding/` | `grid_visibilities_reduction`, `grid_weights_reduction`, kernel data structs (`gridding_kernels.py`) |
| `interpolations/` | `bilinear_interpolation`, `linear_interpolation`, `nearest`, `degridding` |
| `fourier/` | `phase_shift_grid` |
| `functional_models/` | `gaussian`, `airy_disk`, `meerkat`, `polynomial` (analytic primary beams) |
| `composition/` | `combine_visibilities` |
| `decorators/` | `memory.temp_local_save` (zarr spill decorator) |
| `parsers.py` | `parse_sky_coord` |
| `sort.py` | Lazy lex-sort helpers |
| `noise.py` | `calculate_weights`, `calculate_noise_from_weights` |
| `constants.py` | k_B, c, etc. |

---

## 6. Core algorithms

### 6.1 Gridding / degridding

Full UV-plane accumulation always uses the padded grid with Hermitian-conjugate row duplication (`base/visibility_set.py:create_hermitian_symmetric_data`). Reduction is performed by `utils.gridding.grid_visibilities_reduction` and `grid_weights_reduction`, both implemented as numba-friendly reductions on Dask blocks. When `hermitian_symmetry=True`, the accumulator is later compressed to the rfft non-redundant width and inverse-transformed with `da.fft.irfft2` (`fft._fft2.ifft2` dispatching through `fft.backends`).

### 6.2 NUFFT-style measurement operator

`MeasurementOperator` does **not** ship a Type-3 NUFFT; instead it composes:

1. Image padding and beam taper.
2. 2D FFT of the padded model (FFT backend pluggable: `numpy/dask` default, `pyfftw` optional).
3. Phase shift to the field centre.
4. Sample the gridded Fourier image at the (u,v) of every visibility by an `Estimator` (nearest, bilinear, or full kernel `Degridding`).

The adjoint Φᴴ is the gridding direction (visibilities → padded UV grid → ifft2 → image), used implicitly by `DirtyMapper`.

### 6.3 Explicit IDFT

When sampling errors of the gridded operator are unacceptable, `pyralysis.dft.idft2` evaluates the exact (non-uniform) inverse 2D DFT including the w-term `w(√(1-l²-m²)-1)`. The numba kernel `_idft2_numba` is `@njit(parallel=True, cache=True)` over pixels and channels (`dft/_idft.py:310-411`).

### 6.4 RML reconstruction

```
min_x  Σ_i λ_i · F_i(x)        with F_i ∈ {ChiSquared, L1Norm, L2Norm, Tikhonov,
                                            Entropy, TotalFlux, Huber, TVIso, TVAniso, TSV, …}
```

Choice of solver:

- **L-BFGS** for smooth + convex problems (default, README example).
- **Conjugate Gradient** family for memory-tight smooth problems.
- **FISTA** when one non-smooth regulariser dominates (e.g. L1, TV).
- **SDMM** when several non-smooth terms must be combined; each gets its own proximal operator and dual variable.

`LBFGS` supports steepest-descent fallback (`max_corrections=0`), curvature-failure clearing, vectorised vs sequential two-loop recursion based on `memory_threshold_gb`, and **Zarr spill** of correction pairs when `use_disk_storage=True`.

### 6.5 RFI flagging

`flaggers.SumThreshold` implements the Offringa SumThreshold algorithm with a `ThresholdPolicy` describing how thresholds shrink with window size (`ConstantThreshold`, `InverseSqrtThreshold`, `PowerLawThreshold`). Operates on the `FLAG` column inside each `VisibilitySet`.

---

## 7. Input / Output formats

| Format | Reader / Writer | Notes |
|--------|-----------------|-------|
| **CASA Measurement Set** (v2 and v3) | `pyralysis.io.DaskMS` via `dask-ms` (`xds_from_ms`, `xds_from_table`, `xds_to_table`) and `python-casacore` | `_detect_ms_version` toggles between `DATA_DESC_ID` and explicit `SPW_ID/POL_ID`. TAQL filtering supported (`taql_query`). Default chunks `{'row': 10000}`. |
| **CASA antenna `.cfg`** | `AntennaConfigurationIo` | Bundled library covers ALMA cycles 0-10, ACA, ATCA, CARMA, KAT-7, LOFAR HBA/LBA, MeerKAT, ngVLA rev B/C/D, PdBI, SKA-LOW/MID, SMA, VLA, VLBA, WSRT |
| **FITS** images | `pyralysis.io.FITS` via `dafits`, `astropy.io.fits`, `astropy.wcs` | Auto-parses cellsize, phase centre, Stokes axis, beam |
| **Zarr** | `ZarrArray`, `ZarrDataset` (`io/zarr/*`) using `numcodecs` compressors | Used for chunked array persistence and L-BFGS history spilling |
| **HDF5** | `io/hdf5.py` | Lightweight pickling of dataset metadata |
| **SKA subarray YAML/JSON** | via `ska-ost-array-config` (private SKA index) | `simulation/antenna_configs/ska_assemblies.py` |
| **Dataset round-trip** | `Dataset.__iter__` yields component dict for serialisation | Used to write back to MS |

---

## 8. Testing & examples

### 8.1 Tests

`simulators/pyralysis/pytest.ini` is minimal; configuration lives in `setup.cfg`. The suite ships **193 Python files** under `tests/`:

```
tests/
├─ conftest.py
├─ unit/         — sanitizers/, dft/, optimization/, io/, simulation/, pipelines/,
│                  utils/, models/, transformers/, injectors/, fft/, reconstruction/,
│                  systems/, convolution/, base/, flaggers/
└─ integration/  — same buckets + fixtures/, test_f1dim.py, test_mask_primary_beam.py
```

CI runs on GitLab (`pipeline.svg` badge) and hooks Codecov (`codecov` badge in `README.md` line 7). `pre-commit` is enforced with hooks defined in `.pre-commit-config.yaml`.

### 8.2 Notebooks (`examples/notebooks/`)

| Notebook | Topic |
|----------|-------|
| `simulation_sandbox.ipynb` | Build interferometer + sources → MS |
| `point_source_sim.ipynb` | Simple point-source simulation |
| `point_source_reconstruction.ipynb` | RML reconstruction of a point source |
| `dirtymapper_sandbox.ipynb` | Dirty image with weighting schemes |
| `optimization_sandbox.ipynb` | Compare optimisers (LBFGS / FISTA / CG) |
| `antenna_filter_sandbox.ipynb` | Subarray selection via filters |
| `sumthreshold_sandbox.ipynb` | RFI flagging with SumThreshold |

Binder integration: `simulators/pyralysis/binder/` (env, postBuild, README) targets the `release` branch.

### 8.3 CLI scripts (`examples/scripts/`)

| Script | Style |
|--------|-------|
| `dirtymapper_components.py` | Explicit class composition |
| `dirtymapper_pipeline.py` | `ImagerPipeline` orchestration |
| `optimization_components.py` / `optimization_pipeline.py` | RML reconstruction |
| `simulation_components.py` / `simulation_pipeline.py` | End-to-end synthesis |

### 8.4 Bundled MS datasets (`datasets/`)

- `antennae/all_fields.ms`
- `co65/co65.ms`
- `FREQ78/FREQ78.ms`
- `M87/SR1_M87_2017_101_{hi,hilo,lo}_hops_netcal_StokesI.selfcal.{LLRR,final}.ms`
- `selfcalband9/hd142_b9cont_self_tav.ms`

These power the integration tests and the notebooks.

---

## 9. Integration & extension points

| Want to … | Hook |
|-----------|------|
| Add a new gridding kernel | Subclass `pyralysis.convolution.c_kernel.CKernel`, implement `evaluate(...)`, register in `convolution/__init__.py` |
| Add a new estimator | Subclass `pyralysis.estimators.…` and slot it into `MeasurementOperator` |
| Add a new regulariser | Subclass `ObjectiveFunctionTerm` (`optimization/terms/`); implement `function()`, `gradient()`, optionally `proximal()` for FISTA/SDMM |
| Add a new optimiser | Subclass `Optimizer` or `GradientOptimizer`/`ProximalOptimizer`; implement `optimize()` |
| Add a new line search | Subclass `LineSearcher` |
| Add a new sky source | Subclass `pyralysis.models.sky.Source`; implement `compute_visibility(...)` |
| Add a noise model | Subclass `pyralysis.models.noise.NoiseModel` then wrap in an `Injector` |
| Add an FFT backend | Subclass `pyralysis.fft.backends.FFTBackend`, register through `get_backend("name")` |
| Add a pipeline step | Subclass `pyralysis.pipelines.base.Step[ContextType]` and inject into `Pipeline.steps` |
| Add a sanitiser | Subclass `pyralysis.sanitizers.Sanitizer`; chain via `CompositeSanitizer` |
| Add a flag policy | Subclass `pyralysis.flaggers.ThresholdPolicy` |
| Use GPU arrays | Install `pyralysis[cupy]`; arrays inside `xarray.Dataset` are still wrapped through `dask.array`, so kernels written with `numpy` API dispatch via `__array_function__` to CuPy when fed CuPy chunks. |
| Use pyfftw | Install `pyralysis[pyfftw]`; backend chosen via env var or `get_backend("pyfftw")` |

---

## 10. Notable internals

- **Lazy-everywhere defaults.** Top-level facades hold `dask.array` payloads inside `xarray.Dataset` objects; `.compute()` is invoked only at sinks (`Dataset.calculate_weights_sum`, FITS write, optimiser convergence test, etc.).
- **Cached-property hot paths.** `Dataset.total_sum_imaging_weights`, `total_sum_squared_imaging_weights`, `total_visibility_number` all `@cached_property` (`base/dataset.py:530-587`); cleared by `clear_visibility_caches`.
- **Composition over inheritance.** `Gridder` and `WeightingScheme` *contain* a `Grid` value object instead of inheriting grid logic. `Dataset` is a facade aggregating table classes; `SubMS` composes `VisibilitySet`.
- **Multi-dispatch.** `multimethod.multimethod` is used throughout (`ObjectiveFunction`, polarisation conversions) to avoid type-switch ladders.
- **Pluggable FFT backend.** `fft.backends.FFTBackend` ABC enables transparent switching between NumPy/Dask and pyfftw; selection persists across `fft2`/`ifft2` calls.
- **L-BFGS disk spill.** `LBFGSHistory` (in `optimizer/history.py`) holds correction pairs in memory until `in_memory_limit`, then spills oldest to a Zarr store at `disk_storage_path`. Combined limit `max_corrections` is enforced across in-memory + on-disk together (`lbfgs.py:46-52`).
- **DFT sign convention.** `pyralysis.dft.idft2` defaults to `sign_convention="negative"` for the **forward** DFT (so the IDFT uses the positive sign). Documented at `simulators/pyralysis/docs/source/signs_and_conventions.rst`.
- **Hermitian symmetry compaction.** `DirtyMapper` always grids on the full padded UV plane (with conjugate-row duplication) for numerical equivalence, then optionally compacts to the rfft half-plane via `ifftshift` + slice before `irfft2` (`dirty_mapper.py:19-50`).
- **Sanitisation contract.** `WeightSanitizer` only updates `FLAG` from spectra-shaped columns; compact 2D `WEIGHT`/`SIGMA` invalids are cleaned but do not propagate into `FLAG` (`daskms.py:362-365`).
- **MS version compatibility.** `DaskMS._detect_ms_version` examines the MAIN table for `FIELD_ID`/`SPW_ID` columns to switch between v2 (DATA_DESC lookup) and v3 (explicit IDs) reads.
- **Dask graph hygiene.** `MeasurementOperator.update_residual_data` is off by default to keep graphs from blowing up inside optimisation loops; the `ObjectiveFunction._process_visibility_terms` method is careful to call `model_visibility.transform()` only **once** per unique `MeasurementOperator` even when shared by multiple terms (`objective_function.py:66-80`).

---

## 11. Known limitations / TODOs

Pulled from `simulators/pyralysis/CHANGELOG.md` and inline docstrings:

- **No tagged, non-trivial deprecation policy.** CHANGELOG `[Unreleased]` (lines 10-34) is currently empty; the project moves directly between minor versions (`v1.0.0` → `v1.1.0`).
- **No CuPy code path is in-tree.** The `cupy` extra installs the runtime but acceleration relies entirely on `dask.array` dispatch; there are no explicit `cupy.*` kernels under `src/pyralysis/`.
- **No w-projection or A-projection wide-field algorithm.** Wide-field accuracy is delivered through the explicit `idft2` w-term; `MeasurementOperator` is a flat-sky FFT operator.
- **`Source.compute_visibility(...)` (in subclasses) is currently flat-sky.** No analytic Bessel-based handling for resolved sources beyond `GaussianSource` is provided.
- **`pyralysis.io.hdf5`** is only 28 lines — minimal helper, not a full alternative to MS persistence.
- **SKA configurations require a private index.** `ska-ost-array-config` is not on public PyPI; without `--extra-index-url …skao.int…`, `pip install` fails.
- **FISTA expects a single non-smooth term** (`fista.py:36-42`). Use SDMM for multiple non-smooth regularisers.
- **Cell sizes default to radians** (`utils/cellsize.py`); raw floats are *not* arcseconds — easy footgun without `astropy.units`.
- **L-BFGS** with `clear_on_curvature_failure=False` (default) keeps stale (s, y) pairs after curvature violations and may take a steepest-descent step on the next iteration anyway (`lbfgs.py:52-55`).
- **MS write path requires complete subtables.** `_validate_write_requirements` (`daskms.py:281-308`) raises if `dataset.antenna`, `field`, `spws`, `polarization`, `observation`, or `ms_list` is `None` for the corresponding requested table.

---

## 12. Minimal usage recipes

### 12.1 Simulate visibilities (from `README.md` lines 67-92)

```python
from pyralysis.io.antenna_config_io import AntennaConfigurationIo
from pyralysis.simulation import Simulator
from pyralysis.models.sky import PointSource
from pyralysis.injectors import ThermalNoiseInjector

interferometer = AntennaConfigurationIo(input_name="path/to/array.cfg").read()
interferometer.configure_observation(
    min_frequency_hz=1e9, max_frequency_hz=1.1e9, frequency_step_hz=1e7,
    right_ascension="12:00:00", declination="45:00:00",
    integration_time=10, observation_time="1h",
)

source = PointSource(
    reference_intensity=1.0,
    sky_position="12:00:00 45:00:00",
    reference_frequency=1e9,
)
sim = Simulator(interferometer=interferometer, sources=source)
dataset = sim.simulate(create_dataset=True)

thermal = ThermalNoiseInjector(system_temperature=50, integration_time=10, channel_bandwidth=1e6)
noisy_dataset = thermal.apply(dataset)
```

### 12.2 Reconstruct (from `README.md` lines 103-125)

```python
from pyralysis.reconstruction import Image
from pyralysis.optimization import ObjectiveFunction
from pyralysis.optimization.terms import ChiSquared, L1Norm
from pyralysis.optimization.optimizer import LBFGS
from pyralysis.measurement import ModelVisibility   # <-- README path; see note below

image = Image.empty(imsize=(512, 512), cellsize=0.001)
model_visibility = ModelVisibility(dataset=noisy_dataset, image=image)
terms = [
    ChiSquared(model_visibility=model_visibility, penalization_factor=1.0),
    L1Norm(penalization_factor=0.01),
]
objective = ObjectiveFunction(term_list=terms, image=image, persist_gradient=True)
optimizer = LBFGS(objective_function=objective, parameter=image)
reconstructed_image = optimizer.optimize()
```

> **Caveat:** the README references `pyralysis.measurement.ModelVisibility`, but no `pyralysis/measurement/` package exists in this checkout (`find src/pyralysis -name 'measurement' -type d` is empty); the equivalent class is `pyralysis.transformers.MeasurementOperator`. Treat the README snippet as illustrative — verify against the live `ImagerPipeline` (`pipelines/imager/steps.py: SetupMeasurementOperator`) for the canonical wiring.

### 12.3 Dirty image via `ImagerPipeline`

```python
from pyralysis.pipelines import ImagerPipeline, ImagerContext

ctx = ImagerContext(ms_path="data.ms", imsize=512, cellsize=1.0,  # arcsec via Quantity
                    weighting="robust", robust_parameter=0.0,
                    stokes=["I"])
pipeline = ImagerPipeline()                       # default 9-step ordering
final_ctx = pipeline.run(ctx)
final_ctx.dirty_image     # Image
final_ctx.dirty_beam      # Image
final_ctx.restored_image  # Image (if RestoreImage step ran)
```

---

## 13. Documentation map (`docs/source/`)

| File | Topic |
|------|-------|
| `index.rst` | Landing page + toctree |
| `installation.rst` | Pip / conda / docker / source / SKA index |
| `quickstart.rst` | First-run notebook script |
| `usage.rst` | Component-level usage |
| `examples.rst` | Notebook & script catalogue |
| `simulation.rst` | Simulator + sources + injectors |
| `faraday.rst` | Faraday components recipe |
| `data_model.rst` | Dataset / SubMS / VisibilitySet (with metauml diagrams) |
| `flaggers.rst` | SumThreshold + policies |
| `gridding.rst` | Grid, kernels, Hermitian symmetry |
| `measurement_operator.rst` | Φ / Φᴴ derivation |
| `array_configuration.rst` | `.cfg` library and SKA assemblies |
| `optimization.rst` | Solvers + line searches |
| `compressed_sensing.rst` | TV / L1 / TSV demos |
| `regularization.rst` | Term catalogue |
| `objective_functions.rst` | `ObjectiveFunction` API |
| `inverse_problem.rst` | RML formulation |
| `image_processing.rst` | `Image`/`Parameter`/`Mask` |
| `parameter_management.rst` | Memory / chunking |
| `signs_and_conventions.rst` | DFT/FFT signs |
| `io_operations.rst` | DaskMS / FITS / Zarr |
| `performance.rst` | Tuning chunks, backends |
| `testing.rst` | Running pytest, coverage |
| `versioning.rst` | SemVer + setuptools_scm policy |
| `glossary.rst`, `faq.rst`, `references.{rst,bib}`, `license.rst` | Reference material |

UML / sequence diagrams: `docs/source/diagrams/metauml/*.mp` rendered to `docs/source/diagrams/png/`.

---

## 14. Citation

```bibtex
@software{carcamo2021pyralysis,
  author = {Miguel Cárcamo},
  title  = {Pyralysis: A Python framework for radio interferometric imaging and simulation},
  year   = {2021},
  url    = {https://gitlab.com/clirai/pyralysis},
  note   = {https://pyralysis.readthedocs.io/}
}
```

License: GPL-3.0-only (`simulators/pyralysis/LICENSE`).
