# healvis — Comprehensive Reference

This document is an exhaustive technical reference for the **healvis** package as
vendored in `simulators/healvis/`. It covers package metadata, every public and
private function, the data-flow during a simulation, the file-format
conventions, the parallel-execution model, the test surface, the helper
scripts, and the design assumptions. The intent is that someone who has never
opened the source tree can rebuild a complete mental model from this document
alone.

Throughout, code paths use the `simulators/healvis/` prefix relative to the
repository root and the line numbers refer to the snapshot vendored in this
repo.

---

## 1. Identity, License, Status

- **Name:** `healvis`
- **Tagline:** "Radio interferometric visibility simulator based on HEALpix
  maps."
- **License:** 3-clause BSD (file `LICENSE`)
- **Owner:** Radio Astronomy Software Group (RASG); originally developed by
  Adam Lanman / Jonathan Pober (Brown University), now hosted at
  `https://github.com/rasg-affiliates/healvis`.
- **Project URLs:** `https://healvis.readthedocs.org` (docs),
  `https://github.com/rasg-affiliates/healvis` (source).
- **Development status:** "Beta" (PyPI classifier *Development Status :: 4*),
  Python 3.6 / 3.7 / 3.8 listed in classifiers — no Python 2 (the unreleased
  CHANGELOG entry explicitly removes Py2). Quoting the README:
  > **Note** This is a tool developed for specific research uses, and is not
  > yet at the development standards of other RASG projects. Use at your own
  > risk.
- **Last released version (per `CHANGELOG.md`):** v1.2.0 (2019-12-23). The
  vendored copy is the master branch beyond that release.
- **Build system:** PEP 517 + `setuptools`+`setuptools_scm` (from
  `pyproject.toml`); the version string is derived from git via the
  `branch_scheme` callable in `setup.py` (returns `+{node}` for `main`, with
  `.dirty` suffix on dirty trees, and includes branch name otherwise).

### 1.1 Top-level layout

```
simulators/healvis/
├── CHANGELOG.md              v1.0.0 → v1.2.0 + Unreleased notes
├── LICENSE                   3-clause BSD
├── MANIFEST.in               Includes README, LICENSE, VERSION, GIT_INFO
├── README.md                 Brief usage notes (rendered above)
├── pyproject.toml            PEP 517 build, setuptools_scm, black config
├── setup.py                  branch_scheme version generator
├── setup.cfg                 metadata, deps, pytest config, console scripts
├── .flake8                   flake8 conventions (ignores E203/E266/E501/...)
├── .pre-commit-config.yaml   trailing-whitespace, flake8, black, rst-backticks
├── .github/                  CI workflow (omitted)
├── .git                      submodule pointer
├── .gitignore
├── ci/
│   ├── mindeps.yaml          Minimum-deps conda env spec
│   └── tests.yaml            Full-deps conda env spec (adds PyGSM via pip)
├── coverage.xml              Last recorded coverage report
├── .coverage                 Coverage SQLite database
├── notebooks/
│   └── running_healvis.ipynb Tutorial notebook (single, ~366KB)
├── scripts/                  CLI / SLURM driver scripts (see §10)
│   ├── make_gsm_shell.py
│   ├── make_imaging_layout.py
│   ├── make_point_sphere.py
│   ├── multibaseline_beamvary_jobs.py
│   ├── skymodel_vis_sim.py   ← preferred CLI entry point
│   ├── view_obs_coverage.py
│   ├── vis_calc.py
│   ├── vis_param_sim.py
│   ├── vis_shell_calc.py
│   ├── configs/obsolete/     Legacy obsparam YAMLs
│   └── telescope_config/     Layout CSV + YAML configs
├── test-reports/             pytest XML output
└── healvis/                  Importable Python package
    ├── __init__.py           Re-exports submodules + version
    ├── version.py            history_string() for output provenance
    ├── beam_model.py         airy_disk, smooth_beam, PowerBeam, AnalyticBeam
    ├── cosmology.py          21-cm cosmological conversions (Planck15)
    ├── observatory.py        Baseline + Observatory; the main RIME engine
    ├── sky_model.py          SkyModel, flat_spectrum_noise_shell, gsm_shell, construct_skymodel
    ├── simulator.py          obsparam parsing + run_simulation pipelines
    ├── utils.py              jy2Tsr, mparray, freq/time array helpers
    ├── data/                 Bundled test data (gsm map, layout, beamfits)
    │   ├── __init__.py       Exposes DATA_PATH = __path__[0]
    │   ├── HERA_NF_dipole_power.beamfits   ~20 MB beamfits used in tests
    │   ├── gsm_nside32.hdf5                 GSM model for tests
    │   ├── perfect_hex37_14.6m.csv          37-element hex used in redundancy test
    │   └── configs/
    │       ├── HERA65_layout.csv
    │       └── obsparam_test.yaml          Canonical obsparam example
    └── tests/                pytest suite (see §9)
        ├── __init__.py       TESTDATA_PATH + assert_raises_message helper
        ├── conftest.py       Session fixture managing IERS download + tempdir
        ├── test_beam_model.py
        ├── test_observatory.py
        ├── test_pspec.py
        ├── test_simulator.py
        ├── test_sky_model.py
        └── test_utils.py
```

### 1.2 Dependencies

Required (`setup.cfg → install_requires`):

| Package          | Purpose                                                 |
|------------------|---------------------------------------------------------|
| `numpy>=1.14`    | Arrays, einsum, FFTs                                    |
| `scipy`          | `scipy.special.j1` (Airy beam), `scipy.interpolate.interp1d` (via UVBeam) |
| `astropy>=2.0`   | Time/coordinate conversions, EarthLocation, units, constants, Planck15 cosmology |
| `astropy-healpix`| HEALPix neighbour & vec functions plus the `healpy` shim used everywhere |
| `numba`          | Listed but not actually imported anywhere in healvis source — kept for future / by `pyuvdata` linkages |
| `h5py`           | Read/write SkyModel HDF5                                |
| `pyyaml`         | obsparam YAML files                                     |
| `pyuvdata`       | `UVData`, `UVBeam`, `pyuvdata.utils` for ECEF/ENU and pol numbering |

Optional extras:

- `gsm`: pulls `pygsm` via git (`git+git://github.com/telegraphic/PyGSM.git`).
  Used by `sky_model.gsm_shell` and `construct_skymodel(sky_type='gsm')`.
  README warns: *"The use of PyGSM within this package is subject to the GNU
  General Public License (GPL), due to its dependency on `healpy`."*
- `all`: `gsm` + `scikit-learn` (Gaussian-Process beam smoothing in
  `beam_model.smooth_beam`).
- `dev`: Sphinx, numpydoc, nbsphinx, coverage, pytest, pytest-cov, pre-commit.

`README.md` install commands:
```
pip install .             # bare
pip install .[gsm]        # + pygsm
pip install .[all]        # + pygsm + sklearn
pip install .[dev]        # documentation/test stack
```

### 1.3 Console scripts (legacy)

Declared in `setup.cfg → scripts`:
`make_gsm_shell.py`, `make_imaging_layout.py`, `make_point_sphere.py`,
`multibaseline_beamvary_jobs.py`, `skymodel_vis_sim.py`,
`view_obs_coverage.py`, `vis_calc.py`, `vis_param_sim.py`,
`vis_shell_calc.py`. These are detailed in §10.

### 1.4 Pytest configuration

From `setup.cfg → [tool:pytest]`:
```
addopts =
    --cov healvis --cov-report term-missing
    --cov-config=.coveragerc
    --cov-report xml:./coverage.xml
    --junitxml=test-reports/xunit.xml
    --verbose
testpaths = healvis/tests
```
Tests live in `healvis/tests/`. Coverage is mandatory in the default
invocation.

### 1.5 Public namespace (`healvis/__init__.py`)

The `__init__` does only two things:

1. Resolve the version via `importlib.metadata.version(__name__)` (falling
   back to `importlib_metadata` on old Pythons; if package is not installed,
   `__version__ = "unknown"`).
2. Re-export every module: `observatory`, `utils`, `sky_model`, `beam_model`,
   `simulator`, `cosmology`. (`version` module is accessible as
   `healvis.version`; not in the import list but importable as a submodule.)

There is **no curated public API**. Users access everything as
`healvis.<submodule>.<name>`. The de-facto entry points are
`healvis.simulator.run_simulation`, `healvis.observatory.Observatory`, and
`healvis.sky_model.SkyModel`.

---

## 2. Mathematical & Physical Model

healvis evaluates the visibility integral by direct summation over a
HEALPix-tessellated sky:

$$
V_{ij}(\nu, t) = \sum_{p \in \text{FoV}}\, B(\hat{s}_p, \nu)\, T(\hat{s}_p, \nu)\,
\exp\!\left(-2\pi i\, \vec{b}_{ij}\cdot\hat{s}_p / \lambda\right) \cdot
\Omega_{\text{pix}} / J_2(\nu)
$$

where:

- $T(\hat{s}_p, \nu)$ is the brightness-temperature map in **Kelvin** at
  HEALPix pixel $p$,
- $B(\hat{s}_p, \nu)$ is the **power** beam (single polarisation) sampled at
  the topocentric direction $(az, za)$ corresponding to pixel $p$ at the
  given pointing,
- $\vec{b}_{ij}$ is the ENU baseline vector (ant2 − ant1; see CHANGELOG
  v1.1.0),
- $\Omega_{\text{pix}} = 4\pi/(12\,N_{\text{side}}^2)$ is the HEALPix solid
  angle,
- $J_2(\nu) = c^2/(2 k_B \nu^2 \cdot \Omega_{\text{ref}})$ is the
  Jy → K·sr factor (`utils.jy2Tsr`, with `bm` the reference solid angle),
  applied as a **division** after summation so the visibilities come out in
  Jy.

The summation is implemented in `Observatory._vis_calc` (see §6.5):

```
sky      = shell[..., pix, :] * horizon_taper * beam_cube     # (Nskies, Npix_FoV, Nfreq)
fringe   = bl.get_fringe(az, za, freqs)                       # (Npix_FoV, Nfreq)
vis_K_sr = np.sum(sky * fringe, axis=-2)                      # (Nskies, Nfreq)
vis_Jy   = vis_K_sr / jy2Tsr(freqs, bm=Ω_pix)
```

Note: this is **not** the full polarised RIME. healvis carries only the
**Stokes-I** sky map and a **single power-beam polarisation** at a time
(picked via `beam_pol`). To get a `pI`-equivalent or per-pol simulation, the
caller iterates `make_visibilities` over polarisations and concatenates.

### 2.1 Conventions

- **Sky map units:** Kelvin brightness temperature. The package converts
  Jy/pix → K via `jy2Tsr` whenever a flux density is needed (see
  `test_observatory.test_vis_calc` for a worked example).
- **Frequency axis:** values denote **channel centres** (CHANGELOG v1.0.0).
- **Time axis:** Julian Date, values denote **bin centres**.
- **Telescope coordinates:** (lat, lon, alt) in (deg, deg, metres).
- **Antenna coordinates in obsparam:** ENU (East-North-Up) in metres.
- **Baseline convention:** `enu = ant2 - ant1` (CHANGELOG v1.1.0, an explicit
  fix).
- **Azimuth:** measured eastward from North (the `y` axis in the rotated
  pixel frame). The CHANGELOG v1.1.0 fix accounts for the ICRS North-pole
  precession when computing azimuth.
- **Pointing centre format:** `[ra_deg, dec_deg]` in ICRS J2000.

### 2.2 Cosmological model (`cosmology.py`)

All redshift conversions use **Planck15** from `astropy.cosmology`. The 21-cm
rest frequency is `f21 = 1.420405751e9` Hz; speed of light is cached as
`c_ms` to avoid re-conversion. The module exposes:

- `comoving_distance(z) -> Mpc` — wrapper around `Planck15.comoving_distance`.
- `dL_df(z)` — comoving differential distance per Hz $[\text{cMpc}/\text{Hz}]$
  using $c\,(1+z)^2/(H_0\,E(z)\,f_{21})$ (Furlanetto+2006 form). Note: uses
  `c_kms = c_ms / 1e3` internally and `cosmo.H0.value` (km/s/Mpc).
- `dL_dth(z)` — comoving transverse distance per radian $[\text{cMpc}/\text{rad}]$
  via `Planck15.comoving_transverse_distance`.
- `dk_deta(z) = 2π / dL_df(z)` — $k_\parallel$ scaling from delay.
- `dk_du(z) = 2π / dL_dth(z)` — $k_\perp$ scaling from baseline length.
- `X2Y(z) = dL_dth(z)**2 * dL_df(z)` — Mpc³ / (sr·Hz).
- `dkpar_dkperp(z)` — wedge slope $H_0\,D_C(z)\,E(z) / (c\,(1+z))$ with
  `H0` converted to m/(s·Mpc).
- `comoving_voxel_volume(z, dnu, omega)` — Mpc³ via
  `cosmo.differential_comoving_volume(z) * dz * omega`, with
  `dz = f21 * (1/nu0 - 1/nu1)` and `nu0/nu1` derived from `dnu`. Supports
  meshgridded broadcasts when any combination of `(z, dnu, omega)` are arrays.

These functions are used both in the flat-spectrum shell normalisation (so
that the resulting EoR-like noise has a well-defined $P(k)$ amplitude) and in
the `test_pspec` validation test.

---

## 3. `healvis.utils` — small but load-bearing

`utils.py` (116 lines) provides four tools used throughout:

### 3.1 `freq_array_to_params(freq_array)` and `time_array_to_params(time_array)`

Inverse of `simulator.parse_*_params`: given a 1-D array, return a dict that
the obsparam parser would have produced. Both raise `ValueError` if the array
has fewer than 2 elements ("must be longer than 1 to give meaningful
results"). Frequency dict carries
`channel_width, Nfreqs, bandwidth, start_freq, end_freq`; time dict carries
`time_cadence (seconds), Ntimes, duration (days), start_time, end_time`. Used
by tests to round-trip parameters.

### 3.2 `jy2Tsr(f, bm=1.0, mK=False)`

Returns the Rayleigh–Jeans conversion factor

$$
\frac{[\text{K} \cdot \text{sr}]}{[\text{Jy}]} = \frac{10^{-23}\,\lambda^2}{2 k_B\,\Omega_{\text{ref}}}
$$

with `lam = c[cm/s] / f[Hz]` and `k_boltz = 1.380658e-16 erg/K`. The `bm`
argument is the **reference solid angle** the visibility was integrated over
(typically the HEALPix pixel area). `mK=True` multiplies the result by
`1e3`. Implementation hard-codes `c.to('cm/s').value` and Boltzmann's
constant in CGS.

### 3.3 `class mparray(np.ndarray)` — shared-memory ndarray

A `numpy.ndarray` subclass whose data buffer is a `multiprocessing.RawArray`,
giving processes spawned via `multiprocessing.Process` direct access to the
same memory without pickling. Implementation:

```python
class mparray(np.ndarray):
    def __init__(self, *args, **kwargs):
        ctype = np.sctype2char(self.dtype)
        arr = mp.RawArray(ctype, self.size)
        self.data = arr
        self.reshape(self.shape)
```

There is a TODO acknowledging that NumPy no longer supports assignment to
`ndarray.data`, and the linked StackOverflow URL (in the source comment)
shows the intended replacement pattern. As-is, this works on the vendored
NumPy version but will eventually break.

It is created via NumPy's standard alloc flow because `__new__` is not
overridden — `mparray((shape,), dtype=...)` triggers `ndarray.__new__` then
`__init__` rebinds `.data` to the RawArray. Round-trip data assignment uses
slicing: `mp_arr[()] = numpy_arr[()]`.

A dedicated test (`test_utils.test_mparray`) launches 5 processes, each
writing into both an `mparray` and a regular `ndarray`, and verifies that
the writes from the other workers are visible in the `mparray` but not in
the per-process copy of the `ndarray`. **Practical use:** `SkyModel` data
buffers and `PowerBeam.data_array` are stored as `mparray` so that
`Observatory.make_visibilities(Nprocs > 1)` does not duplicate the entire
sky shell into every worker.

### 3.4 `enu_array_to_layout(enu_arr, fname)`

Writes a `(Nants, 3)` ENU array out to a healvis-format antenna layout CSV
(`Name Number BeamID E N U`). Used to bootstrap quick layouts from raw
arrays.

### 3.5 `npix2nside(npix)`

Computes `Nside = sqrt(npix/12)` and validates that the result is a power of
two-friendly integer (asserts `log2(npix/12)` is an integer). Raises
`ValueError(f"Invalid number of pixels {npix}")` otherwise.

---

## 4. `healvis.beam_model`

The beam abstractions are deliberately small. There is no concept of
heterogeneous arrays per antenna; every baseline shares the same
`Observatory.beam` object. Only **power beams** (squared-amplitude) are
supported — any electric-field beamfits is converted on read.

### 4.1 `airy_disk(za_array, freqs, diameter=15.0, **kwargs)`

Standalone function returning an Airy power pattern of shape
`(Npix, Nfreqs)`:

```
xvals = (D/2) * sin(za) * 2π * f / c
beam  = (2 * j1(xvals) / xvals)**2     with the limit beam→1 at xvals==0
```

`za_array` is clipped at `π/2` so the beam does not rise to unity at
`za=π` (a numerical artefact of the Bessel form on the lower hemisphere).
`np.true_divide(..., where=~zeros)` avoids division-by-zero warnings; the
zero-mask is then patched to `1.0`. Used both as a quick analytic option
and as the worker behind `AnalyticBeam('airy', diameter=...)`.

### 4.2 `smooth_beam(freqs, beam_array, freq_ls=2.0, noise=1e-10, output_freqs=None)`

Smooths a beam in frequency by fitting a Gaussian Process with
`sklearn.gaussian_process` (Radial Basis Function kernel + WhiteKernel for
noise). Arguments are in MHz internally — `freqs` are divided by `1e6` before
fitting. Handles complex `beam_array` by fitting real and imaginary parts
separately. `freq_ls` is the RBF length-scale in MHz (default 2.0). Raises
`AssertionError` if scikit-learn is missing. The kernel is held fixed
(`optimizer=None`) so this is interpolation-with-smoothing rather than full
hyperparameter selection.

### 4.3 `class PowerBeam(UVBeam)`

Subclass of `pyuvdata.UVBeam` with two roles: (a) load a beamfits and
guarantee the result is a peak-power beam in shared memory, and (b) provide
a fast point-evaluation routine `beam_val`.

`__init__(beamfits=None)`:

1. Calls `UVBeam.__init__()`.
2. If `beamfits` is given, calls `self.read_beamfits(beamfits)`.
3. If `self.beam_type == "efield"`, calls `self.efield_to_power()`. Result is
   always a power beam.
4. Casts the data array to real (any imaginary part is dropped after
   asserting power beams should have zero imaginary).
5. Allocates a shared-memory `mparray` of the same shape and copies the data
   into it; sets `self._data_array.expected_type = float`.

`interp_freq(freqs, inplace=False, kind='linear', run_check=True)`:

Wraps `UVBeam._interp_freq` and assigns the resulting `data_array`,
`Nfreqs`, `freq_array`, `bandpass_array` onto a copy (or self). Removes any
cached `saved_interp_functions`. Prints `"Doing frequency interpolation:
<kind>"`.

`smooth_beam(freqs, inplace=False, freq_ls=2.0, noise=1e-10, run_check=True)`:

Iterates over `polarization_array`, reshapes the `(1,1,Npol,Nfreqs,Naxes2,Naxes1)`
data array, calls `smooth_beam(...)` once per polarisation, then writes the
output back through the same axes. For `pixel_coordinate_system == "az_za"`
data is flattened to `(Nfreqs, Naxes2*Naxes1)` for fitting and reshaped on
write; for `"healpix"` the data is fed in as `(Nfreqs, Npix)`. The
bandpass array is also smoothed. `Nfreqs/freq_array` are updated and any
`saved_interp_functions` cache is cleared. `run_check` calls
`UVBeam.check()` at the end.

`beam_val(az, za, freqs, pol="pI", **kwargs)`:

Evaluates the power beam at arbitrary `(az, za, freq)` triples, returning
shape `(Npix, Nfreqs)` where `Npix == len(za)`. **Important:** does **not**
interpolate in frequency — for each requested frequency it picks the
nearest-neighbour index in `self.freq_array` (`np.argmin(|Δf|)`). To
interpolate beforehand, call `interp_freq` once. Internally:

- Asserts the beam is a power beam.
- Coerces scalars to length-1 arrays.
- Sets `self.interpolation_function` to `"az_za_simple"` or
  `"healpix_simple"` based on `pixel_coordinate_system`.
- Calls either `self._interp_az_za_rect_spline(...)` or
  `self._interp_healpix_bilinear(...)` with `polarizations=[pol]`,
  `reuse_spline=True` (for the az/za branch).
- Returns `interp_beam[0, 0, 0].T` (i.e. `(Nfreqs, Npix).T`).

`pol` is a single string like `"XX"`, `"YY"`, `"pI"` — it must be among the
beam's available polarisations (UVBeam handles that mapping).

### 4.4 `class AnalyticBeam`

Single-polarisation analytic beam supporting four families:

| `beam_type`            | Behaviour                                                                |
|------------------------|--------------------------------------------------------------------------|
| `"uniform"`            | Returns `1.0` everywhere (or a `(Npix, Nfreqs)` ones array).             |
| `"gaussian"`           | Peak-normalised in `za`: `exp(-za² / 2σ²)` with σ = `gauss_width [deg→rad]`. Optionally chromatic: `σ(ν) = σ_ref * (ν/ν_ref)^spectral_index`. |
| `"airy"`               | Calls `airy_disk(za, freqs, diameter)`.                                  |
| any **callable**       | The callable is stored as `self.beam_type` and invoked with `(za, freqs, **kwargs)` returning `(Npix, Nfreqs)`. |

Constructor enforces:

- `gauss_width` required for `"gaussian"`; converted from degrees to radians.
- For Gaussian with non-zero `spectral_index`, `ref_freq` must be set;
  otherwise `ref_freq` defaults to `1.0` (irrelevant for flat spectrum).
- `diameter` required for `"airy"`.
- Raises `NotImplementedError(f"Beam type {x} not available yet.")` for
  unknown strings.

`beam_val(az, za, freqs, **kwargs)` returns `(Npix, Nfreqs)`. The `kwargs`
are forwarded to a callable beam type so users can plumb extra parameters
(e.g. `A.beam_val(az, za, freqs, diameter=15.0)` when `beam_type=airy_disk`).

The Gaussian form uses **only zenith angle**, so it is azimuthally
symmetric; this matches healvis' choice to simulate a single power-beam
polarisation per pass.

---

## 5. `healvis.sky_model`

`sky_model.py` (459 lines) encapsulates HEALPix-shaped multi-frequency sky
maps and IO. Everything is built around a single class.

### 5.1 `class SkyModel`

Dataclass-flavoured container with attribute auto-update.

`valid_params`:
`Npix, Nside, Nskies, Nfreqs, indices, Z_array, ref_chan, ref_freq,
pspec_amp, freqs, data, history`.

`dsets` (HDF5 datasets and dtypes):
- `"data"   → float64`
- `"indices"→ int32`
- `"freqs"  → float64`
- `"history"→ vlen str`

All other attributes are written as HDF5 root attributes.

**Key invariants and behaviour:**

- `__init__(**kwargs)` sets every `valid_params` entry via `_defaults()` to
  `None`, except `history=""`, `Nskies=1`, `ref_chan=0`. Then assigns kwargs
  (raises `KeyError("Invalid SkyModel parameter: ...")` for unknown keys),
  then calls `_update()`.
- `__setattr__` appends every assignment to `_updated`, including
  bookkeeping ones — this is how the dirty-list tracking works. The list is
  cleared after every `_update()`.
- `set_data(data)` is the public entry point; equivalent to
  `self.data = data; self._update()`.
- `_update()` enforces consistency:
  - If `freqs` was assigned: recompute `Z_array = f21/freqs - 1`,
    `r_mpc = comoving_distance(Z_array)`, and `Nfreqs = freqs.size`.
  - If `Nside` was assigned: ensure `Npix = 12*Nside**2`,
    `pix_area_sr = 4π/Npix`, and (if `indices` not also dirty)
    `indices = arange(Npix)`.
  - If `indices` was assigned: `Npix = indices.size`.
  - If `data` was assigned: ensure the array has 3 axes `(Nskies, Npix,
    Nfreqs)`. A 2-D input of shape `(Npix, Nfreqs)` is reshaped to
    `(1, Npix, Nfreqs)`; any other shape mismatch raises
    `ValueError("Invalid data array shape: ...")`.
- `__eq__` compares every `valid_params` entry. Mismatches print
  `"Mismatch:  <key>"` to stdout (used in tests).

**Data array contract:** `data` is `(Nskies, Npix, Nfreqs)` — note the pixel
axis is in the **middle**. There is one branch in `read_hdf5` that handles
file layouts with the pixel axis last (for `pyradiosky` compatibility) and
swaps axes 1/2 after load.

**Key methods:**

`make_flat_spectrum_shell(sigma, shared_memory=False)`: validates that
`freqs, ref_chan, Nside, Npix, Nfreqs` are all non-None, then calls
`flat_spectrum_noise_shell(sigma, freqs, Nside, Nskies, ref_chan,
shared_memory)`, sets `data`, and records `pspec_amp = sigma`,
`ref_freq = freqs[ref_chan]`. The whole-sky noise has zero mean and
amplitude scaled per voxel so the comoving voxel volume cancels.

`read_hdf5(filename, freq_chans=None, shared_memory=False, do_not_overwrite_freqs=False)`:
- Errors with `ValueError("File ... not found.")` if missing.
- If `freq_chans=None`, reads all channels.
- If `do_not_overwrite_freqs=True`, requires `self.freqs` to be set; finds
  the nearest channel in the file for each requested frequency, ensures
  they round-trip exactly (else
  `ValueError("Currently set frequencies do not match any subset of file's frequencies.")`).
- Reads every root attribute back into `self` (so anything in
  `valid_params` saved as an attribute round-trips).
- Reads each dataset in `dsets`. For `"data"`, supports two on-disk axis
  orderings: `(Nskies, Npix, Nfreqs)` (native) or `(Nskies, Nfreqs, Npix)`
  (pyradiosky); the latter is detected by comparing array dimensions to
  `Npix` and `Nfreqs`, and `np.swapaxes(...,1,2)` is applied. If
  `shared_memory=True`, allocates the destination as an `mparray` so
  subprocess workers can read without copying.
- If `Nside` is missing, infers it from `npix2nside(data.shape[1])`.
- Calls `_update()` at the end.
- Prints `"...reading <filename>"`.
- Emits warnings if file lacks `Nfreqs` or `Nside` attributes.

`write_hdf5(filename, clobber=False)`:
- Skips silently with a print if the file exists and `clobber=False`.
- Writes every non-`None` entry from `valid_params`. `dsets` entries are
  stored as datasets with `compression='gzip', compression_opts=9` (scalars
  use a plain `create_dataset`); other entries are stored as root
  attributes.
- Appends `version.history_string()` to the `history` field on write so the
  output records the function/file/version that produced it.
- Prints `"...writing <filename>"`.

### 5.2 `flat_spectrum_noise_shell(sigma, freqs, Nside, Nskies, ref_chan=0, shared_memory=False)`

Constructs an EoR-flavoured Gaussian random shell with shape
`(Nskies, Npix, Nfreqs)` and amplitude scaled to give a flat 3-D power
spectrum:

```
dV0 = comoving_voxel_volume(Z[ref_chan], dnu, Ω_pix)
for fi in range(Nfreqs):
    dV  = comoving_voxel_volume(Z[fi], dnu, Ω_pix)
    amp = sigma * sqrt(dV0 / dV)
    data[:, :, fi] = N(0, amp, (Nskies, Npix))
```

`shared_memory=True` allocates the buffer as an `mparray`. `dnu` is the
finite-difference of `freqs`. The resulting amplitude (`pspec_amp = sigma`)
is what `test_pspec.test_pspec_amp` recovers analytically via
`X2Y(Z_ref) * Bandwidth / beam_sq_int`.

### 5.3 `gsm_shell(Nside, freqs, use_2016=False)`

Builds a shell from the **Global Sky Model** (PyGSM):

- `use_2016=True` → `pygsm.GlobalSkyModel2016(freq_unit='Hz', unit='TCMB')`
- otherwise        → `pygsm.GlobalSkyModel(freq_unit='Hz', basemap='haslam')`

`maps = ...generate(freqs)` returns shape `(Nfreqs, Npix_native)`. healvis
then `hp.Rotator(coord=['G','C']).rotate_map_pixel` each frequency to ICRS
(equatorial) coordinates and `hp.ud_grade(maps[fi], Nside)` to the
requested resolution. Returns `(Npix, Nfreqs)` (transposed). Raises
`AssertionError` if PyGSM is not importable, and `ImportError` if `healpy`
is missing — the function is one of the few places healvis still needs
classic `healpy` (because PyGSM internally returns RING-ordered native
healpy arrays and the rotator does the same).

### 5.4 `construct_skymodel(sky_type, freqs=None, Nside=None, ref_chan=0, Nskies=1, sigma=None, amplitude=None)`

The single dispatch point used by `simulator.run_simulation`:

| `sky_type`         | Effect                                                                            |
|--------------------|-----------------------------------------------------------------------------------|
| `"flat_spec"`      | `sky.make_flat_spectrum_shell(sigma, shared_memory=True)`                          |
| `"gsm"`            | `sky.data = gsm_shell(Nside, freqs)` (PyGSM required)                              |
| `"monopole"`       | `sky.data = mparray((Nskies, Npix, Nfreqs)); sky.data[()] = amplitude`             |
| any other string   | Treated as a path, dispatched to `sky.read_hdf5(sky_type, shared_memory=True, do_not_overwrite_freqs=True)` |

The freq/Nside/ref_chan/Nskies attributes are set on the `SkyModel` before
dispatch. This is the lone entry point that users / scripts touch when
running a simulation.

---

## 6. `healvis.observatory` — the simulation engine

`observatory.py` (407 lines) defines two classes: `Baseline` and
`Observatory`. `Observatory.make_visibilities` is the core RIME loop.

### 6.1 `class Baseline`

Trivial container for an ENU baseline vector and frequency-dependent uvw.

`__init__(ant1_enu=None, ant2_enu=None, enu_vec=None)`:
- Either both antenna ENU positions or a pre-computed `enu_vec` may be given.
- `self.enu = ant2 - ant1` (note convention from CHANGELOG v1.1.0).
- Asserts `enu.size == 3`.

`get_uvw(freq_Hz)`: returns `np.outer(self.enu, 1/(c_ms/freq_Hz))` — i.e.,
shape `(3, Nfreqs)` of uvw in **wavelengths**.

`get_fringe(az, za, freq_Hz, degrees=False)`:
- Optionally converts `az/za` from degrees to radians.
- Casts `freq_Hz` to float (defensive — required because `c_ms / freq_Hz`
  blows up for ints when `freq_Hz` is something like `1`).
- Computes
  `(l, m, n) = (sin(az)sin(za), cos(az)sin(za), cos(za))` — note
  this has **azimuth measured from North** (m is the cosine component, l
  the sine). So this matches the rotated frame defined in
  `Observatory.calc_azza`.
- Sets `self.uvw = self.get_uvw(freq_Hz)`.
- Computes `udotl = einsum('jk,jl->kl', lmn, uvw)` → shape
  `(Npix, Nfreqs)`.
- Returns `cos(2π udotl) + 1j*sin(2π udotl)` (the inline form is reportedly
  faster than `np.exp(2j*pi*udotl)` per source comment).

### 6.2 `class Observatory` — constructor

```python
Observatory(latitude, longitude, fov=None, baseline_array=None,
            freqs=None, nside=None, array=None)
```

- `latitude, longitude` in **decimal degrees**.
- `fov` defaults to `180` degrees if `None` (full hemisphere).
- `baseline_array` (or alias `array`) is a list of `Baseline` instances.
- `freqs` is the channelisation in Hz; if provided, `self.Nfreqs = len(freqs)`.
- `nside`: if provided, allocates a `HEALPix(nside=nside)` instance and
  immediately calls `_set_vectors()` (which precomputes the full
  pixel-vector array as an `mparray`). If `None`, defers until
  `make_visibilities` populates it from the SkyModel's `Nside`.
- `self.beam` starts as `None`; populated by `set_beam` or
  `setup_observatory_from_uvdata`.
- `self.times_jd, self.pointing_centers, self.north_poles` start `None`.
- `self.telescope_location = EarthLocation.from_geodetic(lon, lat)` —
  used as the observation site for any AstroPy AltAz frame.
- `self.do_horizon_taper = False`.

### 6.3 `_set_vectors()`

Computes the unit vector to every pixel centre via `hp.pix2vec(Nside,
arange(Npix))` (using `astropy_healpix.healpy as hp`), shape `(Npix, 3)`.
Stored as an `mparray` so it is shared across worker processes. Re-allocated
on every `make_visibilities` call (with the SkyModel's Nside).

### 6.4 `set_pointings(time_arr)`

Iterates over each JD in `time_arr` and computes:

- The **zenith** position in ICRS: `AltAz(alt=90°, az=0°, obstime=t,
  location=telescope_location).transform_to(ICRS())` → `[ra_deg, dec_deg]`.
  Stored in `self.pointing_centers`.
- The **North-pole horizon position** in ICRS: `AltAz(alt=0°, az=0°,
  obstime=t, location=telescope_location).transform_to(ICRS())`. Stored
  in `self.north_poles` and used by `calc_azza` to define the azimuth
  origin (without it, azimuth defaults to ICRS `[0, 90]` which is wrong
  except very near J2000).
- Records `self.times_jd = time_arr`.

### 6.5 `calc_azza(center, north=None, return_inds=False)`

Given a pointing centre `[lon, lat]` (in degrees) and optionally the ICRS
position of the local **North horizon point**, returns `(za_arr, az_arr)`
in **radians** for every pixel within `fov/2` of the centre:

1. `radius = fov*π/360` (i.e., half-FoV in rad). When
   `do_horizon_taper=True`, `radius` is grown by one
   `pixel_resolution` so partially-up pixels are included.
2. `cvec = hp.ang2vec(*center, lonlat=True)` — pointing centre vector.
3. `nvec = hp.ang2vec(*north, lonlat=True)` (default `[0, 90]`).
4. Builds an orthonormal frame at the pointing centre:
   `xvec = (nvec × cvec) / sin(colat)` (East), `yvec = cvec × xvec` (North).
5. For every pixel vector $\hat{s}$:
   `sdotz = ŝ·c`, `sdotx = ŝ·x`, `sdoty = ŝ·y`.
6. `za = arccos(sdotz)`, `az = arctan2(sdotx, sdoty) % 2π`.
7. Selects pixels with `za <= radius` and returns the slice. With
   `return_inds=True`, also returns the chosen pixel indices.

Validated by `test_az_za` (5° east of zenith → `az=90°, za=5°`) and
`test_az_za_astropy` (independent check against AstroPy's AltAz transform
with sub-arcmin agreement).

### 6.6 `set_fov(fov)`

Trivial setter. Accepts degrees.

### 6.7 `set_beam(beam='uniform', freq_interp_kind='linear', **kwargs)`

Dispatcher:

- If `beam in ('uniform', 'gaussian', 'airy')` or callable →
  `self.beam = AnalyticBeam(beam, **kwargs)`.
- Otherwise → treats `beam` as a beamfits filepath, instantiates
  `PowerBeam(beam)`, immediately interpolates onto `self.freqs` with the
  given `freq_interp_kind`, and records the kind back on the beam.

### 6.8 `beam_sq_int(freqs, Nside, pointing, beam_pol='pI')`

Computes the beam-squared integral $\int B^2(\hat{s}, \nu)\,d\Omega$ using
a HEALPix-pixelised approximation:

```
za, az = calc_azza(pointing)
beam_sq_int = sum(beam.beam_val(az, za, freqs, pol=beam_pol)**2, axis=0) * (4π / Npix)
```

Returns shape `(Nfreqs,)`. Used by `run_simulation` to write the result into
the output UVData's `extra_keywords` (as `bm_sq_<pol>`), and by the pspec
test for normalisation.

### 6.9 `_horizon_taper(za_arr)`

When `do_horizon_taper=True`, re-weights pixels near the FoV edge by the
fraction of the pixel above the horizon:

```
res     = pixel_resolution [rad]
max_za  = fov/2 [rad]
fracs   = 0.5 * (1 - (za - max_za) / res)
fracs[fracs > 1] = 1.0       # don't over-weight fully visible pixels
```

Pixels well below the horizon are clipped to negative weights (which would
zero or invert the contribution); the typical use is to extend the FoV by
one pixel resolution and let this taper smoothly damp the edge contribution.

### 6.10 `_vis_calc(pcents, tinds, shell, vis_array, Nfin, beam_pol='pI')`

The per-process worker. Arguments:

- `pcents`: list of pointing centres assigned to this worker.
- `tinds`: corresponding indices into the global pointing array (so the
  output can be re-ordered later).
- `shell`: the SkyModel data array (`(Nskies, Npix, Nfreqs)`), expected to
  be an `mparray` for `Nprocs > 1`.
- `vis_array`: an `mp.Manager().Queue` for emitting results.
- `Nfin`: shared `mp.Value('i', 0)` for progress reporting.

For each pointing in `pcents`:

1. Look up `north = self.north_poles[tinds[count]]` (warn if missing).
2. `za, az, pix = self.calc_azza(c_, north, return_inds=True)`.
3. `beam_cube = self.beam.beam_val(az, za, self.freqs, pol=beam_pol)` →
   shape `(Npix_FoV, Nfreqs)`.
4. `horizon_taper = self._horizon_taper(za).reshape(1, Npix_FoV, 1)` if
   tapering is on, else `1.0`.
5. `sky = shell[..., pix, :] * horizon_taper * beam_cube` → shape
   `(Nskies, Npix_FoV, Nfreqs)`.
6. For every baseline `bl` in `self.array`:
   - `fringe_cube = bl.get_fringe(az, za, self.freqs)` — shape `(Npix_FoV, Nfreqs)`.
   - `vis = np.sum(sky * fringe_cube, axis=-2)` — shape `(Nskies, Nfreqs)`.
   - `vis_array.put((tind, bl_idx, vis.tolist()))`.
7. Increment `Nfin` (under its lock).
8. Worker named `"0"` periodically prints progress: elapsed minutes,
   estimated remaining hours, and `MaxRSS` from `resource.getrusage`.

### 6.11 `make_visibilities(shell, Nprocs=1, times_jd=None, beam_pol='pI')`

Top-level driver. Steps:

1. `self.healpix = HEALPix(nside=shell.Nside)` and re-precompute pixel
   vectors via `_set_vectors()`.
2. Sanity: `shell.Nfreqs == self.Nfreqs`.
3. Compute `conv_fact = jy2Tsr(self.freqs, bm=pixel_area_sr)` for the
   final K·sr → Jy conversion.
4. Either use existing `pointing_centers` or re-compute via
   `set_pointings(times_jd)`. Errors if neither is available; warns when
   overwriting.
5. Splits `pointing_centers` and the time-index range into `Nprocs` chunks
   with `np.array_split`.
6. Warns loudly if `shell.data` is **not** an `mparray` while `Nprocs > 1`
   ("this will cause duplication.").
7. Spawns `mp.Process(name=str(pi), target=self._vis_calc, ...)` for each
   chunk.
8. Busy-waits (`while Nfin.value < Ntimes and any(p.is_alive() ...): continue`).
9. Drains the result queue, sorts by `(baseline_inds, time_inds)` via
   `np.lexsort`, and constructs:
   - `visibilities`: `(Nblts, Nskies, Nfreqs)` array
   - `time_array`:    `(Nblts,)` JD values (or `None` if `times_jd` was not set)
   - `baseline_array`:`(Nblts,)` baseline indices into `self.array`
10. **Returns** `(visibilities / conv_fact, time_array, baseline_array)`,
    so the visibilities are **in Jy**.

The `Nblts` ordering is `time-major, baseline-minor` (consistent with the
sort on `(baseline_inds, time_inds)`). Auto-correlations come out purely
real (validated by `test_run_simulation`).

### 6.12 Memory model and parallelism

- The most expensive arrays — the sky shell (`SkyModel.data`), the precomputed
  pixel-vector cube (`Observatory._vecs`), and the beamfits data
  (`PowerBeam.data_array`) — live in `mparray` shared memory.
- `multiprocessing.Process` workers fork (Linux) or spawn (macOS) and reach
  back into the same RawArrays without copying.
- The output queue (`mp.Manager().Queue()`) carries small per-pointing
  payloads (`vis.tolist()`) — one entry per `(time, baseline)` pair.
- Progress is printed only by worker named `"0"`. Other workers stay silent.
- The mainloop's `while ... continue` busy-wait pegs one CPU; this is by
  design (simple and avoids Queue.get blocking semantics on edge cases).

---

## 7. `healvis.simulator` — obsparam pipeline

`simulator.py` (964 lines) wires everything together. It contains:

- Three pure-functional **parser** helpers
  (`parse_telescope_params`, `parse_frequency_params`, `parse_time_params`).
- Two **UVData factory** helpers (`setup_uvdata`, `complete_uvdata`).
- An **Observatory factory** (`setup_observatory_from_uvdata`).
- The two **end-to-end pipelines** (`run_simulation` and
  `run_simulation_partial_freq`).

### 7.1 `_parse_layout_csv(layout_csv)`

Reads a whitespace-separated layout CSV with header row
`Name Number BeamID E N U`. Builds a structured-dtype
(`U10, i4, i4, f8, f8, f8`) and returns a `np.genfromtxt` record array. The
header line is read separately to drive `np.format_parser`.

### 7.2 `parse_telescope_params(tele_params)`

Input: dict with `array_layout, telescope_location, telescope_name`.
`telescope_location` may be `(lat_deg, lon_deg, alt_m)` or the string
`"(-30.7..., 21.4..., 1073.0)"` (bare-tuple format from older pyuvsim
configs).

Steps:

1. Validate the layout file exists.
2. Parse the layout via `_parse_layout_csv`.
3. Convert `lat, lon` to radians; compute `tloc_xyz = uvutils.XYZ_from_LatLonAlt(*tloc)`.
4. Compute ECEF antenna positions: `uvutils.ECEF_from_ENU(antpos_enu, *tloc) - tloc_xyz`.
5. Returns dict:
   `Nants_data, Nants_telescope, antenna_names, antenna_numbers,
    antenna_positions, array_layout, telescope_location, telescope_name`.

### 7.3 `parse_frequency_params(freq_params)`

Frequencies are channel **centres** by convention.

If `freq_array` is present, it supersedes everything; otherwise the
following key combinations are searched in order, picking the first match:

1. `start_freq + Nfreqs + channel_width`
2. `start_freq + Nfreqs + bandwidth`
3. `start_freq + Nfreqs + end_freq`
4. `start_freq + end_freq + channel_width`

Validates that `(end - start)/channel_width` is integral when needed,
otherwise raises `ValueError("end_freq - start_freq must be evenly divisible
by channel_width")`. Raises `ValueError("Channel width must be specified if
passed freq_arr has length 1")` for single-channel cases without explicit
width. Raises `KeyError("Couldn't find any proper combination of keys in
freq_params")` if no combination matches.

If `freq_chans` is present (a string parsed by `ast.literal_eval` to
`(start, stop[, step])`), the resulting `freq_array` is sliced to that
range.

Returns `Nfreqs, freq_array (1, Nfreqs), channel_width, bandwidth`. The
`Nspws=1` axis is hardcoded — pyuvdata's spw model is not exercised.

### 7.4 `parse_time_params(time_params)`

Mirror of the frequency parser, with key combinations:

1. `start_time + Ntimes + time_cadence (sec)`
2. `start_time + Ntimes + duration (days)`  (`duration_hours/duration_days` accepted)
3. `start_time + Ntimes + end_time`
4. `start_time + end_time + time_cadence`

`time_array` overrides everything else if present.

Errors mirror the frequency parser:
`"time_cadence must be specified if Ntimes == 1"`,
`"end_time - start_time must be evenly divisible by time_cadence"`,
`"Couldn't find any proper combination of keys in time_params."`.

The `dayspersec = 1/86400` factor is hardcoded.

### 7.5 `complete_uvdata(uv_obj, run_check=True)`

Given a `UVData` object whose arrays are length `Nbls` (baseline-only) or
`Ntimes` (time-only), tile/repeat them up to `Nblts = Nbls * Ntimes`, and
allocate `data_array, flag_array, nsample_array` of the right shape. Also
sets `Nants_data`, LSTs (`set_lsts_from_time_array`), and `uvw_array`
(`set_uvws_from_antenna_positions`). Optionally runs `uv_obj.check()`.

This pattern lets healvis carry around a "lite" UVData (with arrays still in
their compact pre-tile form) until it's actually time to run the simulation,
saving memory.

### 7.6 `setup_uvdata(...)`

Turns the obsparam fragments into a UVData object ready for
`make_visibilities`. Major arguments:

`array_layout, telescope_location, telescope_name, Nfreqs, start_freq,
bandwidth, freq_array, Ntimes, time_cadence, start_time, time_array, bls,
anchor_ant, antenna_nums, no_autos=True, pols=['xx'], make_full=False,
redundancy=None, run_check=True`.

Behaviour:

1. Calls `parse_telescope_params` to read the layout.
2. Computes `lat, lon, alt` from the ECEF telescope position; recomputes
   ENU positions for downstream use.
3. Either uses `freq_array` directly (must be 1-D or `(1, Nfreqs)`) or
   parses one out of `Nfreqs/start_freq/bandwidth`.
4. Same logic for `time_array` vs. `Ntimes/start_time/time_cadence`.
5. Sets `polarization_array` from `pols` (strings → ints via
   `uvutils.polstr2num`).
6. Computes baseline list:
   - Default: every `(a1, a2)` with `a1 <= a2` (i.e., includes autos).
   - `no_autos=True` removes `(a, a)` pairs.
   - `bls` filters to a user-specified list.
   - `anchor_ant` filters to baselines containing the given antenna.
   - `antenna_nums` filters to baselines containing any listed antenna.
   - `redundancy` (a tolerance in metres) calls
     `uvutils.get_antenna_redundancies(...)` and keeps one representative
     per redundant group.
   - Errors with `ValueError("No baselines selected.")` if the result is
     empty.
7. Stores `time_array (length Ntimes)` and `baseline_array (length Nbls)`.
   With `make_full=True`, calls `complete_uvdata` to fully populate.

Validated by `test_setup_uvdata`, `test_setup_light_uvdata`, and
`test_redundant_setup`. Prints `"Nbls: <n>"` to stdout.

Returns the `UVData` object (drift-mode, `instrument="simulator"`,
`object_name="zenith"`, `vis_units="Jy"`).

### 7.7 `setup_observatory_from_uvdata(uv_obj, fov=180, set_pointings=True, beam=None, beam_kwargs={}, beam_freq_interp='cubic', smooth_beam=False, smooth_scale=2.0, freq_chans=None, apply_horizon_taper=False, pointings=None)`

Produces an `Observatory` from a `UVData`:

1. Recovers ENU antenna positions (`uv_obj.get_ENU_antpos()`) and builds a
   `Baseline` for every unique baseline number.
2. Constructs `obs = Observatory(lat, lon, fov, baseline_array, freqs)`.
3. `obs.do_horizon_taper = apply_horizon_taper`.
4. Sets `pointing_centers` from explicit list (if provided) or via
   `obs.set_pointings(unique(time_array))`.
5. Beam dispatch:
   - `UVBeam` instance → deepcopy, *cast its `__class__` to `PowerBeam`*,
     interpolate onto `obs.freqs`.
   - `str` or callable → `obs.set_beam(beam, freq_interp_kind=..., **beam_kwargs)`.
   - `PowerBeam` → `interp_freq` onto `obs.freqs`.
   - `AnalyticBeam` → installed as-is.
6. If the resulting beam is a `PowerBeam` and `smooth_beam=True`, runs
   `obs.beam.smooth_beam(obs.freqs, inplace=True, freq_ls=smooth_scale)`.

Note the **defaults**: `beam_freq_interp='cubic'` here (overrides the
`PowerBeam.__init__` default of `'linear'`), and `smooth_beam=False`.
`run_simulation` re-overrides these via the obsparam `beam` block.

### 7.8 `run_simulation(param_file, Nprocs=1, sjob_id=None, add_to_history='')`

End-to-end pipeline driven by an obsparam YAML (or pre-loaded dict). Phases:

**A. Parse parameters.**

1. Load YAML if `param_file` is a string; else `copy.deepcopy(param_file)`.
2. `freq_dict = parse_frequency_params(param['freq'])` →
   `freq_array = freq_dict['freq_array'][0]`.
3. `time_dict = parse_time_params(param['time'])` →
   `time_array = time_dict['time_array']`.
4. `filing_params = param['filing']`.
5. `skyparam = copy(param['skyparam']); skyparam['freqs'] = freq_array`.
6. `Nskies` from top-level or skyparam (default `1`).
7. `Nprocs` may be overridden by the YAML.

**B. SkyModel construction.**

8. Pop `sky_type` and optional `savepath` from `skyparam`.
9. `sky = sky_model.construct_skymodel(sky_type, **skyparam)`.
10. If `sky_type` is **not** in `('flat_spec', 'gsm')`, assert
    `np.allclose(freq_array, sky.freqs)`. Otherwise (computed sky), if a
    `savepath` was given, write the freshly computed shell to disk.

**C. UVData object.**

11. `uvd_dict` is built from `param['telescope']` plus `freq_array`,
    `time_array`, `pols`, optional `param['select']`, and `make_full`
    (defaults to `False`).
12. `uv_obj = setup_uvdata(**uvd_dict)`.

**D. Observatory.**

13. Pops `beam_type, beam_freq_interp ('cubic'), smooth_beam (False),
    smooth_scale (None), pointings, fov` from the `beam` and top-level
    sections.
14. `apply_horizon_taper = param.pop('do_horizon_taper', False)`.
15. If `pointings` is a list (literal-eval'd from a string), uses it
    directly and disables `set_pointings`.
16. `obs = setup_observatory_from_uvdata(uv_obj, fov=..., set_pointings=...,
    beam=beam_type, beam_kwargs=beam_attr, beam_freq_interp=...,
    smooth_beam=..., smooth_scale=..., apply_horizon_taper=...,
    pointings=...)`.

**E. Simulate.**

17. Iterates over `pols`. For each polarisation:
    - `visibs, time_array, baseline_inds = obs.make_visibilities(sky, Nprocs=Nprocs, beam_pol=pol)`.
    - `beam_sq_int[f'bm_sq_{pol}'] = obs.beam_sq_int(sky.ref_freq, sky.Nside, obs.pointing_centers[0], beam_pol=pol).item()`.
18. Stacks polarisations along a new last axis:
    `visibility = np.moveaxis(visibility, 0, -1)` → shape
    `(Nblts, Nskies, Nfreqs, Npols)`.

**F. Fill out the UVData and write.**

19. Builds a `param_history` string with the parsed sections and prepends
    `version.history_string(notes=add_to_history + param_history)` to
    `uv_obj.history`.
20. Frees the sky array (`del sky.data`).
21. `uv_obj = complete_uvdata(uv_obj)`.
22. `extra_keywords = {'nside': sky.Nside, 'slurm_id': sjob_id, 'fov': obs.fov, **beam_sq_int}`.
    Adds `'bm_fwhm': fwhm` for Gaussian (`fwhm = gauss_width * 2.355`) or
    `'bm_diam': diameter` for Airy. Adds `'skysig'` for `flat_spec`.
23. For each sky `si in range(Nskies)`:
    - Slice `vis = visibility[:, si]`, expand to
      `(Nblts, Nspws, Nfreqs, Npols)` and assign to `uv_obj.data_array`.
    - Run `uv_obj.check()`.
    - Determine `out_format` (`'uvh5' | 'miriad' | 'uvfits'`) from
      `filing_params['format']`.
    - Build the output file name: prefix from
      `filing_params['outfile_name']` or `outfile_prefix`, with appended
      `_fwhm{fwhm:.3f}` for Gaussian or `_diam{x}` for Airy; suffix from
      `filing_params['outfile_suffix']` (auto-set for multi-sky or
      MIRIAD).
    - Create the output directory if missing.
    - Write via `write_uvh5(clobber=...)` / `write_miriad(clobber=...)` /
      `write_uvfits(force_phase=True, spoof_nonessential=True)`.

This is the function `scripts/skymodel_vis_sim.py` invokes.

### 7.9 `run_simulation_partial_freq(freq_chans, uvh5_file, skymod_file, fov=180, beam=None, beam_kwargs={}, beam_freq_interp='linear', smooth_beam=True, smooth_scale=2.0, Nprocs=1, add_to_history=None)`

Lets a long-frequency simulation be parallelised by splitting it across
batch jobs that each write a subset of channels into a pre-existing UVH5
file:

1. Load only metadata of `uvh5_file` (`UVData.read_uvh5(..., read_data=False)`).
2. Convert `polarization_array` back to strings (`uvutils.polnum2str`).
3. Load the relevant `freq_chans` of `skymod_file` as a `SkyModel`
   (`shared_memory=False` because each job uses one process by default).
4. Verify the requested frequencies are a subset of the file's channels.
5. `setup_observatory_from_uvdata(uvd, fov, beam, beam_kwargs,
    freq_chans, beam_freq_interp, smooth_beam, smooth_scale)`.
6. Run `make_visibilities` per polarisation and stack.
7. Build `flags = zeros(...); nsamples = ones(...)`.
8. `uvd.write_uvh5_part(uvh5_file, visibility, flags, nsamples, freq_chans=freq_chans, add_to_history=add_to_history)`.

This is what enables embarrassingly-parallel multi-job simulations: a
single SkyModel HDF5 file plus a single template UVH5 file can be
populated independently across SLURM jobs that each take a `freq_chans`
slice. There is no built-in script for this — the user is expected to
launch their own SLURM array job from a wrapper script.

---

## 8. obsparam YAML schema

healvis's obsparam YAML loosely follows pyuvsim's, but with extra
healvis-specific blocks (`beam`, `skyparam`, top-level `do_horizon_taper`,
`pointings`, `Nskies`, `Nprocs`).

The canonical example is `healvis/data/configs/obsparam_test.yaml`, fully
reproduced and annotated:

```yaml
filing:
  outdir: "./test_out"          # where to drop the output file
  outfile_name: "test_sim"      # base filename (no extension)
  format: 'uvh5'                # uvh5 | miriad | uvfits
  clobber: True                 # overwrite existing
telescope:
  array_layout: ../data/configs/HERA65_layout.csv
  telescope_location: (-30.72152777777791, 21.428305555555557, 1073.0...)
  telescope_name: HERA
freq:                           # See parse_frequency_params for combinations
  start_freq: 100000000.0       # Hz, channel center
  bandwidth: 50000000.0         # Hz, total
  Nfreqs: 10
time:                           # See parse_time_params for combinations
  Ntimes: 5
  time_cadence: 100.0           # seconds
  start_time: 2458098.5521759833
beam:
  beam_type: "airy"             # airy | gaussian | uniform | <beamfits path>
  beam_freq_interp: 'linear'    # for PowerBeam
  smooth_beam: False            # for PowerBeam
  smooth_scale: 2.0             # MHz
  diameter: 15                  # meters, for airy
  gauss_width: 5.0              # degrees, for gaussian
  fov: 110                      # degrees
  pols:
    - 'xx'
    - 'yy'
select:
  no_autos: False
  bls: '[(0,0),(0,11),(0,12),(11,12)]'
skyparam:
  sky_type: 'gsm'               # flat_spec | gsm | monopole | <hdf5 path>
  ref_chan: 0
  sigma: 0.031                  # used by flat_spec (K)
  Nside: 64
  Nskies: 1
  savepath: 'flatspectrum.hdf5' # write generated sky to disk
```

Older YAMLs in `scripts/configs/obsolete/` (e.g.
`obsparam_heragauss_24hours_nside128_threelong.yaml`) follow the older
pyuvsim-shaped layout: `time.integration_time` (now `time.time_cadence`),
flat top-level `Nside, Nskies, sky_sigma, fov`, separate
`telescope.telescope_config_name`, and `select.bls/redundancy`. Those YAMLs
work with `scripts/vis_param_sim.py`, not with `run_simulation`.

The `select` block in `run_simulation` accepts whatever is consumed by
`setup_uvdata` (`bls, anchor_ant, antenna_nums, no_autos, redundancy,
make_full`).

---

## 9. Test suite (`healvis/tests/`)

`pytest` with `--cov healvis --cov-report term-missing` is the entry point;
default `testpaths = healvis/tests`. The vendored `coverage.xml` and
`test-reports/xunit.xml` are recent successful runs.

### 9.1 `tests/__init__.py`

- Defines `TESTDATA_PATH = os.path.join(DATA_PATH, "temporary_test_data/")`
  (created by the conftest fixture).
- Defines `assert_raises_message(exception_type, message, func, *args, **kwargs)`,
  a helper that wraps `pytest.raises` and asserts the error message contains
  the expected substring; supports `nocatch=True` to fail loudly.

### 9.2 `tests/conftest.py`

Single autouse session-scope fixture:

1. Creates the temporary test data directory.
2. Tries to download the IERS table via `astropy.utils.iers`. On URL/HTTP
   errors falls back to the IERS-A mirror; if that fails, disables IERS
   auto-download for the rest of the session (`auto_max_age=None`,
   `auto_download=False`).
3. After tests, restores `auto_max_age=30, auto_download=True` and cleans
   up the temporary directory.

This makes the suite runnable on offline or restricted CI networks.

### 9.3 `test_utils.py`

- `test_mparray`: spawns 5 `mp.Process` workers writing to a shared
  `mparray` and a non-shared `ndarray`. Verifies cross-process visibility
  for the shared array and isolation for the non-shared one. Worker
  contains the assertions; the main thread spins `while any(p.is_alive()): continue`.
- Re-defines `assert_raises_message` (an unused duplicate; the package one
  in `tests/__init__` is imported elsewhere).

### 9.4 `test_beam_model.py`

- `test_PowerBeam`: loads `data/HERA_NF_dipole_power.beamfits`, exercises
  `interp_freq` (inplace vs. returned-copy equivalence), checks
  `beam_val(az, za, freqs, pol='XX')` shape and peak value (~1 at
  zenith), confirms a frequency shift smaller than the nearest-neighbour
  tolerance leaves the result unchanged, and runs `smooth_beam` with
  `freq_ls=2.0`.
- `test_AnalyticBeam`: tests Gaussian, chromatic Gaussian (`spectral_index=-1`),
  Uniform, Airy, and a custom callable (passing `airy_disk` as the
  `beam_type`). Also tests the constructor errors for invalid types and
  missing arguments.

### 9.5 `test_sky_model.py`

- `verify_update`: invariant checker shared across tests — confirms that
  `Npix/indices/Nside` are mutually consistent and that `_updated` is
  cleared after `_update()`.
- `test_update`: rebuild a flat-spectrum shell and check `verify_update`.
- `test_flat_spectrum`: round-trip
  `flat_spectrum_noise_shell(sigma, freqs, 32, 1)` → assert shape
  `(1, Npix, Nfreqs)`.
- `test_write_read`: write a flat-spectrum SkyModel to a tempdir HDF5,
  read it back twice (with `shared_memory=True` and `False`), assert
  the `mparray` survives and that `__eq__` holds (after blanking
  histories).
- `test_fewchannel_read`: `freq_chans=arange(4)` selects 4 channels from
  `gsm_nside32.hdf5`, with and without shared memory.
- `test_freqselect_read`: tests `do_not_overwrite_freqs=True` to load a
  pre-set `freqs` subset.

### 9.6 `test_observatory.py`

The most exercised test file:

- `test_Observatory`: walks all four `set_beam` choices (`uniform,
  gaussian, airy, callable airy_disk`) plus the beamfits path; confirms
  beam types and shapes.
- `test_Baseline`: builds a 15 m E-W baseline; checks `get_fringe`
  shape and verifies fringe at zenith equals `1.0` for all frequencies.
- `test_pointings`: 20 time samples 20 minutes apart from J2000;
  checks that `pointing_centers` RAs increase at the sidereal rate
  (within half a second) and DECs stay at the observatory latitude
  (within 6 arcmin).
- `test_az_za`: places a known pixel 5° east of zenith; confirms
  `calc_azza` returns `(az=90°, za=5°)` exactly.
- `test_vis_calc`: builds an `Nside=32` sky with a single `1 Jy/pix`
  pixel at the pointing centre; runs `make_visibilities` with a uniform
  beam; asserts `Re(V) ≈ 1.0` for all baselines (validates the unit
  point-source visibility).
- `test_offzenith_vis`: places a `1 Jy/pix` pixel 5° off zenith,
  pointing centre at `(0, 0)`; computes the analytic
  `exp(2πi (u·l + v·m + w·n))` and checks the simulator agrees on real
  and imaginary parts.
- `test_gsm_pointing` (skipped if PyGSM missing): asserts a 4-min
  fringe-rate sweep over the galactic-centre transit window
  (`2458000.227...`) peaks at the central time index. Uses `airy`
  beam, `nside=64`.
- `test_az_za_astropy`: independent validation against AstroPy's AltAz
  transform — agreement within 1e-4 rad on `za` and ~1 arcmin on `az`
  (worst at the southern horizon).

### 9.7 `test_pspec.py`

- `test_pspec_amp`: end-to-end sanity check that the **flat-spectrum
  shell amplitude** recovered from a delay-transformed visibility
  matches the analytic `comoving_voxel_volume(Z_ref, dnu, Ω_pix)`
  prediction within 2× the per-bin sample variance. Uses 20 timesteps,
  200 frequency channels (100–150 MHz), Nside=64, FoV=50°, Gaussian
  beam (`gauss_width=7.37`). Runs with `Nprocs=3` to also exercise the
  multi-process path. The maths:
  ```
  vis_Ksr  = jy2Tsr(freqs) * vis_Jy
  _vis     = ifft(vis_Ksr, axis=freq)[:, :Nfreqs//2]
  dspec    = |_vis|**2 * X2Y(Z_ref) * (Bandwidth / beam_sq_int)
  amp_th   = sigma**2 * comoving_voxel_volume(Z_ref, dnu, Ω_pix)
  assert np.isclose(amp_th, mean(dspec), atol=2*amp_th/Ntimes)
  ```

### 9.8 `test_simulator.py`

- `test_setup_uvdata`: three smoke tests of the UVData factory with
  the bundled HERA65 layout — full-array, baseline-list selection,
  baseline-list + `antenna_nums` filter.
- `test_run_simulation`: full pipeline using `obsparam_test.yaml`. Uses
  the bundled GSM HDF5 (`gsm_nside32.hdf5`), the bundled HERA beamfits,
  and writes to `data/sim_testing_out/test_sim.uvh5`. Validates
  `Nfreqs/Ntimes/Nbls/Npols`, presence of the appended history, and
  that auto-correlations are purely real (validates the
  `(Nblts, Nbls, Ntimes)` ordering).
- `test_run_simulation_partial_freq`: writes a stub UVH5 file matching
  the GSM-test frequencies, then fills channels `0:3` via
  `run_simulation_partial_freq` and asserts those channels are non-zero
  while `>=3` are still zero.
- `test_setup_light_uvdata`: confirms the lightweight (`make_full=False`)
  UVData builds with the right `time_array` length (Ntimes, not Nblts)
  while still producing a consistent Observatory.
- `test_parse_freq_params`, `test_parse_time_params`: drive every key
  combination through the parsers and validate via
  `assert_raises_message` for the documented error paths.
- `test_redundant_setup`: builds the 37-element hex with a 0.5 m
  redundancy tolerance and checks `Nbls == 61` (no autos) and `62`
  (with autos).
- `test_freq_time_params`: round-trips a real GSM frequency vector and
  a 239-step time grid through `freq_array_to_params →
  parse_frequency_params` and `time_array_to_params → parse_time_params`.

---

## 10. Helper scripts (`scripts/`)

All scripts are SLURM-friendly (the SBATCH headers are listed in comments)
but most are runnable locally. Detailed inventory:

### 10.1 `skymodel_vis_sim.py` — preferred CLI

Thin SLURM wrapper:

```python
parser.add_argument(dest="param", help="obsparam yaml file")
parser.add_argument("-n", dest="Nproc", help="...overrides SLURM Ncpus", type=int)

if args.Nproc:                                       Nprocs = args.Nproc
elif "SLURM_CPUS_PER_TASK" in os.environ:            Nprocs = int(os.environ['SLURM_CPUS_PER_TASK'])
else:                                                Nprocs = 1
sjob_id = os.environ.get("SLURM_JOB_ID")

healvis.simulator.run_simulation(param_file, Nprocs=Nprocs, sjob_id=sjob_id)
```

### 10.2 `make_gsm_shell.py`

CLI:
```
make_gsm_shell.py <obsparam.yaml> [--nside N] [--clobber]
```
Reads the `freq` block of the obsparam, generates a `gsm_shell` at the
requested resolution (max 512), and writes it to
`skymodels/gsm_<f0>-<f1>MHz_nside<N>.hdf5`.

### 10.3 `make_imaging_layout.py`

Generates a 128-element layout with a Gaussian-distributed core whose
maximum baseline is `30 λ` at 100 MHz. **Currently dead-ends at
`sys.exit()` after computing the resolution-limited baseline length** —
the trailing code is left as a template. Output file is
`dense_imaging_layout.csv`.

### 10.4 `make_point_sphere.py`

Builds a HEALPix shell at `Nside=128` whose only non-zero pixels lie on
an `Nside=12` lattice of declination rings. Pixel values index the
declination ring number — useful for checking imaging fidelity (each
ring should image to a coherent strip). Writes `healvis/data/imaging_test_map.hdf5`.

### 10.5 `multibaseline_beamvary_jobs.py`

Dispatches an `Nensemb=49` × `Nbeams=75` SLURM array (75 beam FWHMs
linearly spanning 15–50°) of `vis_param_sim.py`. Each job pulls
`configs/obsparam_multibl.yaml` and overrides `-b <fw>`.

### 10.6 `view_obs_coverage.py`

Builds an `Observatory` for the HERA site, sets 100 pointings 30 s apart,
and calls `obs.get_observed_region(Nside=128)`. **Note:** this method
does not exist on `Observatory` in the vendored sources — the script
assumes a method that was removed. Treat as legacy / broken.

### 10.7 `vis_calc.py`, `vis_param_sim.py`, `vis_shell_calc.py`

Three closely-related single-purpose drivers from the project's early
history. They predate `run_simulation` and call into the low-level
APIs directly:

- `vis_calc.py`: builds a flat-spectrum noise shell on the fly,
  Gaussian beam, single 14.6 m E-W baseline, writes a MIRIAD file
  named `healvis_gauss<fwhm>d_<hours>hours_<bllen>m_<Nside>nside_<fov>fov_uv`.
- `vis_param_sim.py`: a more elaborate version that **uses pyuvsim** to
  parse the obsparam, then drives healvis directly. Reads
  `gaussian_eor_shell` from the catalog field. Note the references to
  `pyuvsim.simsetup.parse_*` and `pyuvsim.utils.write_uvdata`.
- `vis_shell_calc.py`: like `vis_calc.py` but **reads a sky shell from
  an HDF5 file using the `eorsky` package** (an external project) and
  applies a Gaussian beam.

These are **not** wired into `setup.cfg` test discovery and several use
APIs that have since changed (`obs.set_beam('gaussian', sigma=...)`
instead of `gauss_width=`, `obs.calc_azza(Nside, ...)` instead of the
modern signature, `set_drift()` instead of `_set_drift()`). Treat as
historical reference, not as supported entry points.

### 10.8 `scripts/telescope_config/`

Provided antenna layouts and config YAMLs:

- `HERA65_layout.csv` (65 antennas, `HHnnn` names, ENU in metres)
- `HERA65_config.yaml` (Nants=65, telescope_location, name)
- `MWA65_config.yaml` (sigma=0.23345 → Gaussian beam config)
- `MULTIBL_layout.csv` (synthetic linear baseline ladder — antenna 50 at
  the origin and 64 baselines fanning eastward at ~1.6 m increments)
- `MWA_east_hex_layout.csv` (east-hex MWA tile layout in real ECEF).

### 10.9 `scripts/configs/obsolete/`

Eight legacy obsparam YAMLs (HERA hex, three short, three long, one E-W,
multi-baseline) feeding `vis_param_sim.py`. They use the older
`time.integration_time` / top-level `Nside, fov, sky_sigma`
schema. Useful as templates.

---

## 11. Data files

`healvis/data/` ships:

- `__init__.py` exports `DATA_PATH = __path__[0]` so tests can resolve
  asset paths via `os.path.join(DATA_PATH, ...)`.
- `HERA_NF_dipole_power.beamfits` (~20 MB) — a HERA dipole power
  beamfits used for the PowerBeam tests and `test_run_simulation`.
- `gsm_nside32.hdf5` (~967 KB) — a precomputed Global Sky Model
  shell at `Nside=32` for `test_run_simulation` and the SkyModel IO
  tests.
- `perfect_hex37_14.6m.csv` — 37-element hexagonal layout at 14.6 m
  spacing for redundancy testing.
- `configs/HERA65_layout.csv` — bundled copy of the HERA layout (used
  by tests).
- `configs/obsparam_test.yaml` — the canonical obsparam used by
  `test_run_simulation`.

The `MANIFEST.in` only mentions `healvis/VERSION` and `healvis/GIT_INFO`
(legacy artefacts of the older versioning scheme), but the data files are
shipped because `setup.cfg` enables `include_package_data = True` and the
project uses `setuptools_scm` to vacuum every git-tracked file into the
sdist.

---

## 12. Output products

A `run_simulation` call produces a fully-checked `pyuvdata.UVData` object
written to UVH5 (default), MIRIAD, or UVFITS. Its salient features:

- `instrument = "simulator"`, `object_name = "zenith"`, `vis_units = "Jy"`,
  drift mode (`_set_drift()`).
- `data_array.shape = (Nblts, 1, Nfreqs, Npols)`. `Nblts = Nbls × Ntimes`,
  baseline-major within each time block (per the `lexsort` order).
- `flag_array = zeros(...), nsample_array = ones(...)`.
- `integration_time = zeros(Nblts)` — set to zero per CHANGELOG v1.0.0
  ("not a meaningful parameter in simulation").
- `extra_keywords`:
  - `nside`           — SkyModel resolution.
  - `slurm_id`        — passed through `sjob_id` argument.
  - `fov`             — Observatory field of view in degrees.
  - `bm_sq_<pol>`     — pointing-zero `beam_sq_int` value (one per polarisation).
  - `bm_fwhm`         — Gaussian beam full-width half-max [deg], when applicable.
  - `bm_diam`         — Airy beam diameter [m], when applicable.
  - `skysig`          — `pspec_amp` for `flat_spec` skies.
- `history` — full obsparam echo prefixed by
  `version.history_string(notes=...)`.
- For `Nskies > 1`, one output file per sky realisation, suffixed with
  `_{si}sky_uv`.

A `SkyModel` written by `write_hdf5` is an HDF5 file with:
- Datasets: `data` (gzip-9), `indices`, `freqs`, `history` (vlen str).
- Root attributes: `Npix, Nside, Nskies, Nfreqs, Z_array, ref_chan,
  ref_freq, pspec_amp` (any non-`None` member of `valid_params` not in
  `dsets`).

---

## 13. Design notes & gotchas

- **No polarised RIME.** healvis multiplies a *power* beam by a *Stokes-I*
  sky map. Polarisation enters only by selecting different beam
  polarisations (`pol='XX', 'YY', 'pI', ...`) and emitting them as
  parallel `Npols` columns in the output. There is no Jones/Mueller chain.
- **Beamfits frequency interpolation is only done up-front.**
  `PowerBeam.beam_val` always nearest-neighbour samples in frequency at
  evaluation time. To get smooth frequency behaviour, the obsparam's
  `beam.beam_freq_interp` is used to pre-interpolate the beam onto the
  observation frequencies once during setup; per-baseline calls then have
  zero frequency mismatch.
- **`numba` is in `install_requires` but is not actually imported.** It
  appears to have been retained for compatibility with downstream
  expectations or a planned acceleration that never landed.
- **`mparray` is fragile.** It abuses `ndarray.data` reassignment, which
  newer NumPy versions explicitly forbid. The TODO in `utils.py`
  acknowledges this. Until replaced, run on the conda-forge versions
  pinned in `ci/tests.yaml`.
- **`Observatory.array` is not safely mutable.** Several APIs (e.g.
  `make_visibilities`) reach into `self.array` to iterate baselines, but
  there's no formal setter — replace it via the constructor.
- **`set_pointings` always re-derives north-pole positions** with full
  AstroPy AltAz transforms — for long time arrays this dominates setup
  cost. If you need many pointings, batch them or cache the result.
- **`_vis_calc` busy-waits.** With `Nprocs > 1`, the main process spins
  on `while ... continue`. CPU usage will read as `Nprocs+1`.
- **Output is queue-based.** Workers `put` per-pointing tuples on an
  `mp.Manager().Queue`; the main process drains, sorts, and reshapes.
  This works fine for `Nblts ≤ ~10⁶` but can be slow for very large
  simulations. There's no batching of queue puts.
- **`run_simulation_partial_freq` is the recommended way to scale across
  many cores or nodes.** Each job loads only its frequency slice into
  memory, so the per-job working set scales with `Nfreqs_local × Npix`,
  not with the full bandwidth.
- **Coordinate convention.** Azimuth is measured **eastward from north** in
  the topocentric frame; pointing centres are stored as `[ra_deg, dec_deg]`
  in ICRS J2000. `north_poles` is what makes the azimuth match AstroPy at
  arbitrary epochs.
- **No support for `Nspws > 1`.** Hardcoded to 1 in `setup_uvdata`.
- **No multi-pointing-per-time support.** A single pointing per time bin
  (typically zenith) is built, although `Observatory.pointing_centers` may
  be set manually to anything before `make_visibilities`.
- **`AnalyticBeam` Gaussian width** is supplied in **degrees** (and stored
  as radians internally). Don't confuse with the raw `sigma` keyword that
  some legacy scripts (`vis_calc.py`, `vis_param_sim.py`) pass to
  `set_beam('gaussian', sigma=...)` — that codepath does **not** exist in
  the current `AnalyticBeam`, which expects `gauss_width=`.
- **PyGSM rotation.** `gsm_shell` rotates the GSM map from Galactic to ICRS
  via `hp.Rotator(coord=['G','C']).rotate_map_pixel`, then `ud_grades` to
  the requested resolution. This drops below-`Npix` pixels (the
  `maps[fi, :Npix]` slice) which works because PyGSM's native NSIDE is
  ≥ 512.

---

## 14. Quick reference: typical end-to-end flow

```
                        +-----------------------------------+
obsparam.yaml  ─────────►  simulator.run_simulation         │
                        +------+------------+----------+----+
                               │            │          │
                               ▼            ▼          ▼
              parse_frequency_params   parse_time_params   construct_skymodel
                               │            │          │
                               ▼            ▼          ▼
                        +--------------------------+  SkyModel(data: mparray
                        │  setup_uvdata(...)       │     in shared memory)
                        │  -> UVData (lite)        │
                        +-------------+------------+
                                      │
                                      ▼
                        +--------------------------+
                        │ setup_observatory_       │
                        │   from_uvdata(...)       │
                        │  -> Observatory          │
                        │  (PowerBeam / Analytic-  │
                        │   Beam in shared memory) │
                        +-------------+------------+
                                      │
                          for pol in pols:
                                      │
                                      ▼
                        Observatory.make_visibilities(sky, Nprocs, beam_pol=pol)
                                      │
                          spawns Nprocs workers │
                          each runs _vis_calc on │
                          its share of pointings │
                                      │
                                      ▼
                          (Nblts, Nskies, Nfreqs, Npols) visibility cube [Jy]
                                      │
                                      ▼
                        complete_uvdata + write_uvh5/miriad/uvfits
                                      │
                                      ▼
                          One UVH5/MIRIAD/UVFITS file per sky realisation
```

---

## 15. Where each thing lives — line-precise index

| Symbol                                          | File                  | Notes |
|-------------------------------------------------|-----------------------|-------|
| `__version__`                                   | `healvis/__init__.py:11` | via `importlib.metadata.version` |
| `history_string`                                | `healvis/version.py:10` | annotates HDF5 history with caller info |
| `freq_array_to_params`                          | `healvis/utils.py:10`  | dict-of-{channel_width, Nfreqs, bandwidth, start, end} |
| `time_array_to_params`                          | `healvis/utils.py:38`  | dict-of-{time_cadence(s), Ntimes, duration(d), start, end} |
| `jy2Tsr`                                        | `healvis/utils.py:64`  | Jy ↔ K·sr conversion |
| `mparray`                                       | `healvis/utils.py:80`  | shared-memory ndarray (RawArray-backed) |
| `enu_array_to_layout`                           | `healvis/utils.py:95`  | write antenna ENU CSV |
| `npix2nside`                                    | `healvis/utils.py:111` | invert HEALPix Npix → Nside |
| `f21`, `c_ms`                                   | `healvis/cosmology.py:16` | 21-cm rest freq + speed of light |
| `dkpar_dkperp`, `comoving_distance`, `dL_df`,
  `dL_dth`, `dk_deta`, `dk_du`, `X2Y`,
  `comoving_voxel_volume`                        | `healvis/cosmology.py:20–86` | Planck15-based conversions |
| `airy_disk`                                     | `healvis/beam_model.py:27` | analytic Airy power pattern |
| `smooth_beam`                                   | `healvis/beam_model.py:58` | sklearn GP smoothing |
| `PowerBeam`                                     | `healvis/beam_model.py:105` | UVBeam + shared memory + nearest-freq eval |
| `AnalyticBeam`                                  | `healvis/beam_model.py:322` | uniform / gaussian / airy / callable |
| `SkyModel`                                      | `healvis/sky_model.py:32`  | (Nskies, Npix, Nfreqs) container |
| `flat_spectrum_noise_shell`                     | `healvis/sky_model.py:317` | EoR-flavoured Gaussian shell |
| `gsm_shell`                                     | `healvis/sky_model.py:364` | PyGSM-backed shell |
| `construct_skymodel`                            | `healvis/sky_model.py:408` | dispatch by `sky_type` string |
| `Baseline`                                      | `healvis/observatory.py:28` | enu, get_uvw, get_fringe |
| `Observatory`                                   | `healvis/observatory.py:61` | set_pointings, calc_azza, set_beam, ... |
| `Observatory._set_vectors`                      | `healvis/observatory.py:120` | per-pixel unit vector cube (mparray) |
| `Observatory.set_pointings`                     | `healvis/observatory.py:131` | RA/Dec at zenith + ICRS north pole |
| `Observatory.calc_azza`                         | `healvis/observatory.py:162` | per-pixel az/za in the local rotated frame |
| `Observatory.set_fov`                           | `healvis/observatory.py:222` | trivial setter |
| `Observatory.set_beam`                          | `healvis/observatory.py:228` | analytic vs. beamfits dispatch |
| `Observatory.beam_sq_int`                       | `healvis/observatory.py:252` | ∫B² dΩ via HEALPix sum |
| `Observatory._horizon_taper`                    | `healvis/observatory.py:273` | edge-pixel down-weighting |
| `Observatory._vis_calc`                         | `healvis/observatory.py:287` | per-process worker |
| `Observatory.make_visibilities`                 | `healvis/observatory.py:336` | top-level RIME loop |
| `_parse_layout_csv`                             | `healvis/simulator.py:23`  | structured-dtype layout reader |
| `parse_telescope_params`                        | `healvis/simulator.py:39`  | yaml dict → telescope dict |
| `parse_frequency_params`                        | `healvis/simulator.py:94`  | freq dict → freq_array etc. |
| `parse_time_params`                             | `healvis/simulator.py:204` | time dict → time_array etc. |
| `complete_uvdata`                               | `healvis/simulator.py:327` | tile/repeat (Nbls,Ntimes) → Nblts |
| `setup_uvdata`                                  | `healvis/simulator.py:370` | the UVData factory |
| `setup_observatory_from_uvdata`                 | `healvis/simulator.py:557` | the Observatory factory |
| `run_simulation`                                | `healvis/simulator.py:646` | full obsparam → UVH5 pipeline |
| `run_simulation_partial_freq`                   | `healvis/simulator.py:866` | per-channel-slice driver |

---

## 16. Limitations summarised

- Stokes-I sky × power beam only — no full polarisation propagation.
- No primary-beam direction-dependent effects beyond the global beam (no
  per-station heterogeneity, no beam pointing offsets).
- No w-projection. The `n` term is included exactly in `get_fringe`, but
  imaging integration assumes the FoV has been clipped via `fov`.
- No mutual coupling, ionosphere, or antenna gain effects.
- No DDE/DIE Jones chains.
- No support for `Nspws > 1`.
- No CASA Measurement Set output (only UVH5/MIRIAD/UVFITS via pyuvdata).
- Beam frequency interpolation is nearest-neighbour at evaluation time —
  pre-interpolation onto the observed frequencies is required for smooth
  spectra.
- `mparray` relies on a pattern that newer NumPy versions reject; expect
  brittleness on freshly-installed environments.
- Some legacy scripts call into APIs that have since been renamed
  (`set_drift` vs `_set_drift`, `get_observed_region`, `set_beam(...,
  sigma=)`); those scripts are reference material, not supported tools.

---

*End of healvis.md.*
