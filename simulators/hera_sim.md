# hera_sim — Exhaustive Reference

A deep, top-to-bottom reference for the `hera_sim` package vendored at
`simulators/hera_sim/`. Written from a direct read of every Python
module in `src/hera_sim/` (`__init__.py`, `__yaml_constructors.py`,
`adjustment.py`, `antpos.py`, `beams.py`, `cli_utils.py`,
`components.py`, `defaults.py`, `eor.py`, `foregrounds.py`,
`interpolators.py`, `io.py`, `noise.py`, `rfi.py`, `sigchain.py`,
`simulate.py`, `utils.py`, `vis.py`) and `src/hera_sim/visibilities/`
(`__init__.py`, `cli.py`, `simulators.py`, `matvis.py`, `fftvis.py`,
`pyuvsim_wrapper.py`, `vis_cpu.py` — empty placeholder), the YAML
season configs in `src/hera_sim/config/` (`H1C.yaml`, `H2C.yaml`,
`debug.yaml`), the example simulation configs in `config_examples/`
(`simulator.yaml`, `template_config.yaml`), the entry-point scripts in
`scripts/` (`hera-sim-simulate.py`, `hera-sim-vis.py`), the
`pyproject.toml`, the `README.rst`, the `CHANGELOG.rst`, the
documentation tree under `docs/` (including the `tutorials/`), and the
test layout under `tests/`. Inside RadioSim this is the engine to read
when you want **HERA-flavoured systematic and visibility simulation**:
it is the canonical place where bandpasses, reflections, mutual
coupling, RFI, EoR-like and foreground visibilities, sigchain and
crosstalk effects are assembled into a single `pyuvdata.UVData`-backed
object and where the `MatVis` / `FFTVis` / `UVSim` visibility back-ends
are unified behind one CLI.

`hera_sim` describes itself in its own `README.rst` as a
**"Basic simulation package for HERA-like redundant interferometric
arrays"** with the highlights:

> * **Systematic Models:** Many models of instrumental systematics in
>   various forms, eg. thermal noise, RFI, bandpass gains, cross-talk,
>   cable reflections and foregrounds.
> * **HERA-tuned:** All models have defaults tuned to HERA, with
>   various default "sets" available (eg. H1C, H2C).
> * **Interoperability:** Interoperability with `pyuvdata` datasets
>   and `pyuvsim` configurations.
> * **Ease-of-use:** High-level interface for adding multiple
>   systematics to existing visibilities in a self-consistent way.
> * **Visibility Simulation:** A high-level interface for visibility
>   simulation that is compatible with the configuration definition
>   from `pyuvsim` but is able to call multiple simulator
>   implementations.
> * **Convenience:** Methods for adjusting simulated data to match
>   the times/baselines of a reference dataset.

It is heavily co-designed with `pyuvdata`, `pyuvsim`, `pyradiosky`,
`matvis` and `fftvis`: every observation lives inside a `UVData`
object, every sky catalogue is a `pyradiosky.SkyModel`, every primary
beam is a `pyuvdata.UVBeam` or `pyuvdata.analytic_beam.AnalyticBeam`,
and the visibility simulators are thin wrappers around the underlying
backends. Authors: HERA Team, Steven Murray
(`murray.steveng@gmail.com`). License MIT, Python ≥ 3.11, status
`Development Status :: 4 - Beta`. Documentation lives at
<https://hera-sim.readthedocs.io>; the upstream repo is
<https://github.com/HERA-Team/hera_sim>.

---

## 1. What `hera_sim` is, in one paragraph

`hera_sim` is **two superimposed simulation frameworks under one
namespace**:

1. **A "rough", visibility-space simulator and systematic injector**
   centred on `hera_sim.simulate.Simulator`. Components of the
   instrument and sky (foregrounds, EoR, RFI, thermal noise, bandpass
   gains, reflections, crosstalk, mutual coupling) are subclasses of a
   common `SimulationComponent` ABC declared in
   `hera_sim.components`. Each component is registered into a global
   discoverable registry (`get_all_components`), accepts default
   parameters at instantiation, and overrides or merges the package-
   wide HERA defaults (`hera_sim.defaults`). The `Simulator` walks a
   `pyuvdata.UVData` baseline-time-frequency grid, applies each
   requested component model with deterministic, baseline-aware
   seeding, caches per-baseline filters (delay/fringe), tracks history
   in the `UVData` object, and writes the result to disk in any
   `pyuvdata`-supported format. This is the path used by the
   `hera-sim-simulate.py` CLI and by every `H1C` / `H2C` "rough"
   simulation pipeline. It is *fast*, deliberately schematic, and
   meant for power-spectrum / pipeline-stress testing — not for
   imaging.

2. **A "proper" RIME visibility-simulation framework** centred on
   `hera_sim.visibilities`, providing `ModelData` (a triple of
   `UVData` + `pyradiosky.SkyModel` + `pyuvsim.BeamList`),
   `VisibilitySimulator` (an ABC with `simulate(...)`,
   `validate(...)`, `from_yaml(...)`, `estimate_memory(...)`,
   `compress_data_model(...)`/`restore_data_model(...)` hooks), and
   `VisibilitySimulation` (a dataclass that orchestrates a single
   simulator run). Three concrete back-ends ship inside this layer:
   `UVSim` (wraps `pyuvsim.uvsim.run_uvdata_uvsim`), `MatVis` (wraps
   `matvis.cpu.simulate` / `matvis.gpu.simulate` and reorders into
   `UVData` shape), and `FFTVis` (wraps
   `fftvis.CPUSimulationEngine().simulate`). All three accept the
   same `pyuvsim`-style obsparam YAML, the same `pyradiosky`
   catalog YAMLs, and the same `BeamList`, and are dispatched by the
   `hera-sim-vis.py` CLI through the `load_simulator_from_yaml`
   factory. This is the path used for "true sky" simulations of
   GLEAM-derived catalogues + analytic / FITS HERA beams that feed
   downstream calibration and EoR pipelines.

Both frameworks are tied together through the YAML constructors
declared in `__yaml_constructors.py`: `!Tsky`, `!Beam`, `!Bandpass`,
`!Reflection`, `!antpos`, and `!dimensionful` (astropy-quantity).
These let a single YAML config describe an entire simulation without
custom Python.

---

## 2. Versioning, packaging, and dependencies

From `pyproject.toml`:

| Field | Value |
|---|---|
| `name` | `hera_sim` |
| `description` | A collection of simulation routines describing the HERA instrument. |
| `requires-python` | `>=3.11` |
| Build backend | `setuptools>=64`, `setuptools_scm>=8` (dynamic `__version__`) |
| Authors | HERA Team; Steven Murray (`murray.steveng@gmail.com`) |
| License | MIT |
| Status classifier | `Development Status :: 4 - Beta` |
| Topic classifiers | Astronomy, Physics |
| Script entry-points | `scripts/hera-sim-simulate.py`, `scripts/hera-sim-vis.py` |

**Runtime dependencies** (`[project.dependencies]`):

- `astropy` — units (`u.GHz`, `u.sday`, `u.Jy`, `u.K`),
  `astropy.constants` (`c`, `k_B`), `astropy.coordinates.EarthLocation`,
  `Time`, `Longitude`. Used pervasively for sky-day vs solar-day
  conversions (`u.sday.to("s")` shows up in noise / RFI / fringe-rate
  code) and for the Jy↔K conversion.
- `astropy-healpix` — `astropy_healpix.lonlat_to_healpix`,
  `nside_to_npix`, `nside_to_pixel_area` for the point→HEALPix sky
  conversion in `VisibilitySimulation._convert_point_to_healpix`.
- `cached-property` — used to back `Simulator.integration_time`,
  `Simulator.channel_width`, `ModelData.lsts`, `ModelData.times`,
  `ModelData.freqs`, `Tsky._interpolator`, etc., when memoizing
  derived properties.
- `deprecation` — `@deprecated` decorator used to mark legacy
  `Simulator.add_eor`, `add_foregrounds`, `add_noise`, `add_rfi`,
  `add_gains`, `add_sigchain_reflections`, `add_xtalk` shims.
- `hera-cli-utils>=0.1.0` — `hera_cli_utils.parse_args`,
  `run_with_profiling`, and the `RicherHandler` rich-logging adapter
  used by `scripts/hera-sim-vis.py`.
- `numpy>=2` — ndarrays everywhere; uses
  `np.random.default_rng()` (the new RNG API).
- `pyuvdata>=3.2.0` — `UVData`, `UVBeam`, `BeamInterface`,
  `pyuvdata.utils`, `pyuvdata.analytic_beam.AnalyticBeam`,
  `pyuvdata.UniformBeam`, `pyuvdata.telescopes.Telescope`. The
  `Simulator.data` attribute *is* a `UVData`; the `ModelData.uvdata`
  attribute *is* a `UVData`. This is the lingua franca.
- `pyuvsim>=1.4` — `pyuvsim.simsetup.initialize_uvdata_from_keywords`,
  `initialize_uvdata_from_params`, `initialize_catalog_from_params`,
  `_complete_uvdata`, `uvdata_to_telescope_config`, `BeamList`,
  `pyuvsim.uvsim.run_uvdata_uvsim`, `pyuvsim.simsetup.SkyModelData`.
- `pyyaml>=5.1` — config parsing; yaml constructors for `!Tsky`,
  `!Beam`, `!antpos`, `!dimensionful`, etc.
- `rich` — `rich.console.Console`, `rich.panel.Panel`, `rich.rule.Rule`
  for pretty CLI output in `hera-sim-vis.py`.
- `scipy` — `scipy.interpolate.RectBivariateSpline`, `interp1d`,
  `scipy.optimize.least_squares` (used in `antpos.idealize_antpos` to
  align the redundant-snapped grid back to real positions),
  `scipy.signal.windows.blackmanharris` (in `Bandpass`).
- `typing-extensions;python_version<'3.11'` — strictly defensive given
  the package already requires Python ≥ 3.11.

**Optional extras** (`[project.optional-dependencies]`):

- `vis` — `fftvis @ git+https://github.com/tyler-a-cox/fftvis`,
  `line-profiler`, `matvis>=1.3.3`, `mpi4py`, `pyradiosky>=0.1.2`.
  This is the extra you must install to use the `MatVis`, `FFTVis` and
  `UVSim` simulators (the latter through `pyradiosky` for catalog I/O
  and `mpi4py` for parallel `pyuvsim`).
- `bda` — `bda` (Baseline-Dependent Averaging,
  <https://github.com/HERA-Team/baseline_dependent_averaging>). Only
  used by `scripts/hera-sim-simulate.py` to optionally apply BDA at
  the end of a rough simulation.
- `cal` — `hera-calibration>3.6.1`. Required for `cli_utils.write_calfits`
  (which calls `hera_cal.io.write_cal`) and for `antpos.idealize_antpos`
  (which calls `hera_cal.redcal.get_reds` /
  `hera_cal.redcal.reds_to_antpos`) and for `adjustment.py`
  (`hera_cal.abscal.get_d2m_time_map`, `hera_cal.io.to_HERAData`,
  `hera_cal.utils.lst_rephase`).

**Dev / docs / tests** (`[dependency-groups]`):

- `docs` — `furo`, `ipython`, `nbsphinx`, `numpydoc>=0.8`,
  `sphinx>=1.8,<7.2`, `sphinx-autorun` (note: pinned below 7.2).
- `tests` — `coverage>=4.5.1`, `matplotlib>=3.4.2`, `pytest>=3.5.1`,
  `pytest-cov>=2.5.1`, `uvtools` (used by `Bandpass` for `gen_window`
  taper generation when not "tanh"), `pytest-sugar`, `pytest-xdist`.
- `doctest` — `tests` + `ipykernel>=7.2.0` + `papermill>=2.7.0`
  (used to execute the tutorial notebooks as doctests).
- `dev` — `docs` + `tests` + `prek>=0.3.9` + `ruff>=0.15.10`.

**Soft dependencies** detected at import-time:

- `numba` — used inside `utils.py` for the JIT-accelerated
  `jit_reshape_vis`, `jit_reshape_vis_invert`, `_left_matmul`,
  `_right_matmul`, `_matmul` helpers consumed by
  `MutualCoupling`. Falls back to NumPy when missing.
- `uvtools` — `uvtools.dspec.gen_window` is imported lazily inside
  `sigchain.py`'s `Bandpass` to provide tapers other than `"tanh"`.
- `mpi4py` — only required for distributed `MatVis`, `FFTVis`, or
  `UVSim` runs; the `cli.py` runner detects it at import.
- `bda` / `bda.bda_tools` — optional in `hera-sim-simulate.py`.
- `psutil` — used by `hera-sim-vis.py` to print
  `psutil.virtual_memory().available`.

**Tooling** (`pyproject.toml [tool.*]`):

- `pytest` runs with
  `--cov hera_sim --cov-config=.coveragerc --cov-report xml:./coverage.xml --durations=25 -v`
  and `testpaths = "tests"`.
- `ruff` line-length 88, lint rules `UP, E, W, F, NPY, I`, ignores
  `NPY002` (legacy `np.random` usage flagged as a TODO).
- `ruff` format `docstring-code-format = true`,
  `skip-magic-trailing-comma = true`.

---

## 3. Source-tree layout

```
simulators/hera_sim/
├── AUTHORS.rst, CHANGELOG.rst, CONTRIBUTING.rst, LICENSE.rst, README.rst
├── pyproject.toml, uv.lock
├── config_examples/
│   ├── simulator.yaml          # for hera-sim-vis.py (chooses simulator class)
│   └── template_config.yaml    # for hera-sim-simulate.py (rough sim recipe)
├── docs/
│   ├── index.rst, conf.py, requirements.txt, environment.yaml,
│   ├── tutorials.rst, contributing.rst, notes_for_developers.rst,
│   ├── reference/index.rst        # autosummary tree
│   └── tutorials/
│       ├── end_to_end_example.ipynb
│       ├── hera_sim_cli.rst
│       ├── hera_sim_defaults.ipynb
│       ├── hera_sim_simulator.ipynb
│       ├── hera_sim_tour.ipynb
│       ├── hera_sim_vis_cli.rst
│       ├── mutual_coupling_example.ipynb
│       ├── polybeam_simulation.ipynb
│       └── visibility_simulator.ipynb
├── scripts/
│   ├── hera-sim-simulate.py    # rough sim CLI (entry point)
│   └── hera-sim-vis.py         # visibility sim CLI (entry point)
├── src/hera_sim/
│   ├── __init__.py
│   ├── __yaml_constructors.py  # !Tsky, !Beam, !Bandpass, !Reflection,
│   │                           # !antpos, !dimensionful
│   ├── adjustment.py           # adjust_to_reference / interpolate_to_reference
│   ├── antpos.py               # LinearArray, HexArray, idealize_antpos
│   ├── beams.py                # PolyBeam, PerturbedPolyBeam, ZernikeBeam
│   ├── cli_utils.py            # validate_config, write_calfits
│   ├── components.py           # SimulationComponent ABC + @component decorator
│   ├── config/                 # season YAMLs (H1C, H2C, debug)
│   ├── data/                   # bundled .npy / .npz model files
│   ├── defaults.py             # Defaults singleton + @_defaults decorator
│   ├── eor.py                  # NoiselikeEoR
│   ├── foregrounds.py          # DiffuseForeground, PointSourceForeground
│   ├── interpolators.py        # Tsky, Beam, Bandpass, Reflection
│   ├── io.py                   # empty_uvdata, chunk_sim_and_save
│   ├── noise.py                # ThermalNoise + HERA_Tsky_mdl factory
│   ├── rfi.py                  # RfiStation, Stations, Impulse, Scatter, DTV
│   ├── sigchain.py             # Bandpass, Reflections, ReflectionSpectrum,
│   │                           # CrossCouplingCrosstalk, CrossCouplingSpectrum,
│   │                           # MutualCoupling, OverAirCrossCoupling,
│   │                           # WhiteNoiseCrosstalk, apply_gains, vary_gains_in_time
│   ├── simulate.py             # the Simulator class (high-level interface)
│   ├── utils.py                # delay/fringe filters, white-noise gen, Jy↔K, numba
│   ├── vis.py                  # sim_red_data (white-noise redundant data)
│   └── visibilities/
│       ├── __init__.py         # SIMULATORS registry, load_simulator_from_yaml
│       ├── cli.py              # hera-sim-vis.py runner + argparser
│       ├── simulators.py       # ModelData, VisibilitySimulator, VisibilitySimulation
│       ├── matvis.py           # MatVis simulator
│       ├── fftvis.py           # FFTVis simulator
│       ├── pyuvsim_wrapper.py  # UVSim simulator
│       └── vis_cpu.py          # empty placeholder (legacy alias)
└── tests/
    ├── conftest.py
    ├── test_adjustment.py, test_antpos.py, test_beams.py, test_cli_utils.py,
    ├── test_components.py, test_defaults.py, test_eor.py, test_foregrounds.py,
    ├── test_interpolators.py, test_io.py, test_noise.py, test_rfi.py,
    ├── test_sigchain.py, test_sim_red_data.py, test_simulate_cli.py,
    ├── test_simulator.py, test_utils.py, test_yaml_constructors.py,
    ├── test_visibilities/      # MatVis / FFTVis / UVSim integration tests
    └── testdata/               # healvis_catalog.txt, hera-sim-vis-config/
```

`__init__.py` re-exports the modules and the public names
`SimulationComponent`, `component`, `get_all_components`, `get_model`,
`get_models`, `defaults`, `Bandpass`, `Beam`, `Tsky`, `Simulator`,
`load_simulator_from_yaml`, `simulators`. It also sets two important
package-level paths:

```
DATA_PATH   = Path(__file__).parent / "data"
CONFIG_PATH = Path(__file__).parent / "config"
```

These paths are referenced *by name* throughout the code (e.g. when
`Tsky_mdl` is initialised from `HERA_Tsky_Reformatted.npz`, when
`Bandpass._gen_bandpass` falls back to `HERA_H1C_BANDPASS.npy`, when
`io.empty_uvdata` reads `HERA_LAT_LON_ALT.npy`).

---

## 4. The `data/` payload

Every binary asset shipped inside `src/hera_sim/data/` (this is what
makes `hera_sim` "HERA-tuned"):

| File | Purpose |
|---|---|
| `HERA_LAT_LON_ALT.npy` | HERA telescope `(lat, lon, alt)` — the default `telescope_location` for `io.empty_uvdata`. |
| `HERA_Tsky_Reformatted.npz` | The default sky-temperature model used by `DiffuseForeground` and `ThermalNoise`. Has keys `tsky` (`(npols, nlsts, nfreqs)` K), `freqs` (GHz), `lsts` (rad), `meta` (`{pols: (...)}`). |
| `HERA_Tsky_vs_LST.npz` | Companion sky-temperature dataset. |
| `HERA_H1C_BANDPASS.npy` | Polynomial-coefficient HERA Phase-One bandpass (default for `sigchain.Bandpass._gen_bandpass`). |
| `HERA_H2C_BANDPASS.npy` | Polynomial-coefficient HERA Phase-Two bandpass. |
| `HERA_H4C_BANDPASS.npz` | Phase-Four bandpass arrays. |
| `HERA_H1C_BEAM_INTEGRALS.npz` / `HERA_H2C_BEAM_MODEL.npz` / `HERA_H4C_BEAM_INTEGRALS.npz` | Beam integrals (Ω_p, etc.) used as `omega_p` defaults. |
| `HERA_H1C_BEAM_POLY.npy` / `HERA_H2C_BEAM_POLY.npy` | Polynomial fits to the beam area `Ω_p(ν)` — what `ThermalNoise` falls back to when `omega_p is None`. |
| `HERA_H1C_REFLECTIONS.npz` / `HERA_H4C_REFLECTIONS.npz` | Per-antenna cable reflection tabulations. |
| `HERA_H1C_RFI_STATIONS.npy` / `HERA_H2C_RFI_STATIONS.npy` | 5-tuple definitions (`f0, duty_cycle, strength, std, timescale`) for known RFI stations. Loaded directly by `rfi.Stations` when `stations` is a path. |
| `H37_FR_Filters_small.npz` | Pre-computed 37-antenna fringe-rate filter set. |
| `tutorials_data/` | Catalogs and beam files used by the tutorial notebooks. |

The `interpolators.py` `_check_path` helper resolves any non-absolute
path against `DATA_PATH`, so YAML configs can refer to files like
`HERA_Tsky_Reformatted.npz` directly.

---

## 5. The `SimulationComponent` framework (`components.py`)

### 5.1 The ABC

`SimulationComponent` is the abstract base class that all
"plug-in-able" rough-sim effects subclass. Defined in
`src/hera_sim/components.py`. Its public surface is:

| Class attribute | Default | Meaning |
|---|---|---|
| `is_multiplicative` | `False` | Whether the effect multiplies existing visibilities (gains, reflections) or adds to them (noise, foregrounds, RFI, EoR). Determines how `Simulator._iteratively_apply` combines the component with `data_array`. |
| `is_randomized` | `False` | Whether the effect uses a `np.random.Generator`. When `True`, `Simulator._iteratively_apply` injects an `rng` argument seeded according to the user's seeding mode. |
| `return_type` | `None` | One of `"per_antenna"` (a `dict[int, np.ndarray]` of gains), `"per_baseline"` (a `(nlsts, nfreqs)` complex array per `(ant1, ant2, pol)`), or `"full_array"` (a single `(nblts, nfreqs, npols)` complex array for the whole UVData). |
| `attrs_to_pull` | `{}` | Mapping `param_name → Simulator_attr` that tells `Simulator._update_args` what to fetch from the live `Simulator` for this component's call signature (e.g. `bl_vec`, `autovis`, `autovis_i`, `autovis_j`, `antpair`, `ants`). |
| `_alias` | `()` | Extra string aliases under which this component is registered (e.g. `Bandpass._alias = ("gains", "bandpass_gain")`). |

The lifecycle of a subclass:

1. Define `__init__(self, **kwargs)` that calls
   `super().__init__(**kwargs)` with the *defaults*. The base
   `__init__` stashes them into `self.kwargs`, which then becomes
   the contract for what the model accepts — `self._check_kwargs`
   raises if you pass anything not in there, and
   `self._extract_kwarg_values` returns the (possibly defaults-
   overridden) values in a stable order.
2. Implement `__call__(self, ...)`, which is decorated abstract on
   the base. It receives `lsts, freqs, …` plus whatever was wired
   through `attrs_to_pull` plus the `**kwargs` overrides.
3. The `__init_subclass__` magic does two things:
   - calls `cls._update_call_docstring()`, which copies the
     `Parameters` section of `__init__`'s docstring into the
     `__call__` docstring so users see the merged signature.
   - registers the class into `cls._models` under its own
     lowercased `__name__` and every `_alias`. This is what powers
     `get_models()`/`get_model()` lookups.

`@component` is the class decorator that creates a fresh abstract
subclass with its own private `_models = {}` dict (so the registries
of `Foreground`, `EoR`, `Noise`, `Gain`, `Crosstalk`, `Array`, `RFI`
do not collide). It also registers the abstract class into a global
`_available_components` dict.

### 5.2 Discovery surface

```
get_all_components(with_aliases=False)  -> {component_name: {model_name: ModelClass}}
get_models(cmp_name, with_aliases=False) -> {model_name: ModelClass}
get_model(model_name, cmp=None)          -> ModelClass
list_all_components(with_aliases=True)   -> human-readable string
```

`Simulator._get_component` uses `get_model` to translate a string
alias (or class, or instance) handed to `Simulator.add(...)` into a
concrete class.

### 5.3 The eight component categories

As of this revision, the registry contains **eight `@component`
abstract bases**, each with one or more concrete subclasses:

| Component | Module | Concrete models |
|---|---|---|
| `Array` | `antpos.py` | `LinearArray`, `HexArray` |
| `Foreground` | `foregrounds.py` | `DiffuseForeground` (alias `diffuse_foreground`), `PointSourceForeground` (alias `pntsrc_foreground`) |
| `EoR` | `eor.py` | `NoiselikeEoR` (alias `noiselike_eor`) |
| `Noise` | `noise.py` | `ThermalNoise` (alias `thermal_noise`) |
| `RFI` | `rfi.py` | `Stations` (alias `rfi_stations`), `Impulse` (`rfi_impulse`), `Scatter` (`rfi_scatter`), `DTV` (`rfi_dtv`) |
| `Gain` | `sigchain.py` | `Bandpass` (aliases `gains`, `bandpass_gain`), `Reflections` (`reflection_gains`, `sigchain_reflections`), `ReflectionSpectrum` (`reflection_spectrum`) |
| `Crosstalk` | `sigchain.py` | `CrossCouplingCrosstalk` (`cross_coupling_xtalk`), `CrossCouplingSpectrum` (`cross_coupling_spectrum`, `xtalk_spectrum`), `MutualCoupling` (`mutual_coupling`, `first_order_coupling`), `OverAirCrossCoupling`, `WhiteNoiseCrosstalk` (`whitenoise_xtalk`, `white_noise_xtalk`) |

Note that `RfiStation` (in `rfi.py`) is *not* a `SimulationComponent`
on its own — it is a building block that `Stations` uses internally.
Likewise the `JonesBaselineTerm`-style separation does not exist
here: `MutualCoupling`'s `return_type = "full_array"` is its
distinguishing feature.

---

## 6. The defaults system (`defaults.py`)

`Defaults` is a `_Singleton`-metaclass class, instantiated at
module-import time as `defaults = Defaults()`. Its behaviour:

- Holds `self._raw_config` (the nested dict as loaded from YAML),
  `self._config` (a fully *flattened* `{param: value}` view used for
  lookup), `self._config_name` (`"h1c"`, `"h2c"`, `"debug"`,
  `"custom"`, or `None`), and `self._override_defaults` (the
  global on/off switch).
- `defaults.set(config, refresh=False)` accepts a path-string, a
  season keyword (`"h1c"`, `"h2c"`, `"debug"` — these resolve to
  files under `CONFIG_PATH`), or a dict; it loads the YAML and
  flattens it via `Defaults._unpack_dict` (recursively descending
  into all sub-dicts except for `array_layout`, which is preserved as
  a single mapping value), then `activate()`s the override.
  `_check_config()` warns if any flattened key has multiple values
  in different parts of the nested config.
- `defaults.activate()` / `defaults.deactivate()` toggle whether the
  flattened defaults override the in-code defaults at call time.
- `defaults(name)` returns a single value, `defaults()` returns the
  whole flattened dict.
- `defaults.apply(func_kwargs, **kwargs)` — used inside
  `SimulationComponent._extract_kwarg_values` — keeps every key in
  the model's signature, prefers user `kwargs` if provided, else
  fills in from `self._config`.
- `_handler` is a function decorator (exposed at module-level as
  `_defaults`) that wraps any function so that, when the global
  override is active, kwargs in the function signature get filled
  from the flattened defaults. This is how `io.empty_uvdata` and
  `Bandpass._gen_bandpass` pick up `Ntimes`, `start_time`,
  `bp_poly`, etc., from the active season config without explicit
  argument plumbing.

The shipped season configs:

```
SEASON_CONFIGS = {
    "h1c":   <CONFIG_PATH>/H1C.yaml,
    "h2c":   <CONFIG_PATH>/H2C.yaml,
    "debug": <CONFIG_PATH>/debug.yaml,
}
```

`H1C.yaml` (verbatim summary):

```yaml
setup:
    frequency_array: {Nfreqs: 1024, channel_width: 97656.25, start_freq: 100000000}
    time_array:      {Ntimes: 100, integration_time: 10.7, start_time: 2458119.5}
telescope:
    array_layout: !antpos {array_type: hex, hex_num: 3, sep: 14.6, split_core: False, outriggers: 0}
    bp_poly:    !Bandpass {datafile: HERA_H1C_BANDPASS.npy, interp_kwargs: {interpolator: poly1d}}
    omega_p:    !Beam     {datafile: HERA_H1C_BEAM_POLY.npy, interp_kwargs: {interpolator: poly1d}}
    delay_filter_type: tophat
    fringe_filter_type: tophat
sky:
    Tsky_mdl: !Tsky {datafile: HERA_Tsky_Reformatted.npz, interp_kwargs: {pol: xx}}
```

`H2C.yaml` is identical in structure but with `Nfreqs: 1638`,
`channel_width: 122070.3125`, `start_freq: 46920776.3671875`,
`integration_time: 8.59`, `start_time: 2458119.5`,
`HERA_H2C_BANDPASS.npy`, `HERA_H2C_BEAM_MODEL.npz` (interp1d).

Once `defaults.set("h2c")` is called, every call to
`hera_sim.io.empty_uvdata()` will inherit those frequency/time/array
settings; every call to `Bandpass(...)` falls back to the H2C
bandpass; every `DiffuseForeground` and `ThermalNoise` invocation
picks up the HERA Tsky model and the Phase-Two beam area.

---

## 7. The high-level `Simulator` (`simulate.py`)

### 7.1 Construction

```
Simulator(
    *,
    data: str | UVData | None = None,
    defaults_config: str | dict | None = None,
    redundancy_tol: float = 1.0,
    **kwargs,             # forwarded to io.empty_uvdata when data is None
)
```

- If `data` is `None`, `_initialize_data` calls `io.empty_uvdata(**kwargs)`.
- If `data` is a path, `UVData.read(...)` is used; the raw path is
  also stashed into `self.data.extra_keywords["data_file"]`.
- If `data` is a `UVData`, it is used as-is.
- `_calculate_reds(tol=redundancy_tol)` calls
  `self.data.get_redundancies(tol=tol)` and stores `(red_grps,
  red_vecs, red_lengths)` for later use by the per-redundant-group
  filter cache and by the `seed="redundant"` mode.
- The constructor wires up `self.Ntimes, Nfreqs, Nblts, Npols, Nbls`
  and copies `self.data.get_data`, `get_flags`, `get_antpairs`,
  `get_antpairpols`, `get_pols` onto `self`.
- It also exposes `self.telescope`, `self.ant_1_array`,
  `self.ant_2_array`, `self.polarization_array`, `self.data_array`,
  `self.antpos`, `self.lsts`, `self.freqs`, `self.times`, `self.pols`,
  `self.integration_time`, `self.channel_width` as
  properties / cached properties.

`self.antpos` is the dict from `utils.get_antpos_dict(uvd,
data_ants=True)` — only antennas with data. `self.lsts` is computed
in a wrap-aware way (unique by `time_array`, sliced into
`lst_array`). `self.freqs` is in **GHz** (`np.unique(freq_array)/1e9`).

### 7.2 `add` — the main entry point

```
Simulator.add(
    component: str | type[SimulationComponent] | SimulationComponent,
    *,
    add_vis: bool = True,
    ret_vis: bool = False,
    seed: str | int | None = None,    # "once", "redundant", "initial", int, or None
    vis_filter: Sequence | None = None,
    component_name: str | None = None,
    **kwargs,
) -> np.ndarray | dict[int, np.ndarray] | None
```

Key behaviours, drawn directly from the implementation:

1. **Resolve the model.** `_get_component` accepts a string alias
   (looked up via `components.get_model`), a class, or an instance,
   and returns the class or instance. `model_key` is either the
   user-supplied `component_name` or the lowercased class name —
   this is what is used as the cache key in `self._components`,
   `self._seeds`, `self._antpairpol_cache`.
2. **Sanity check.** `_sanity_check` warns if you try to add a
   multiplicative effect before any visibilities have been
   simulated, or add visibilities after a multiplicative effect has
   already been applied.
3. **Default merge.** When `defaults._override_defaults` is on, any
   parameter in the model's `kwargs` that the user didn't pass is
   pulled from the active defaults dict.
4. **Iterate and apply.** `_iteratively_apply(model, ...)` walks
   every `(ant1, ant2, pol)` antpair-pol via
   `self.data.get_antpairpols()`, asks `_apply_filter` whether the
   current filter passes, calls `_seed_rng` with the requested
   seeding mode, and either:
   - For `is_multiplicative=True` models with `return_type="per_antenna"`,
     pre-simulates a `gains` dict for each feed-pol *once* and then
     multiplies the `data_array` slice by `gi * conj(gj)` baseline
     by baseline.
   - For `return_type="per_baseline"`, calls `model(**args)` and
     `+=`s it into `data_copy[blt_inds, :, pol_ind]`.
   - For `return_type="full_array"`, calls `model(**args)` once and
     `+=`s the whole array (only `MutualCoupling` uses this).
5. **Cache filter pre-computation.** If the model declares
   `is_smooth_in_freq=True` (the default) and accepts
   `delay_filter_kwargs` / `fringe_filter_kwargs` and the simulator
   has previously called `calculate_filters(...)`, the pre-computed
   delay/fringe filters are looked up by redundant-group key (with
   conjugation flipping the fringe filter axis-0).
6. **History.** `_update_history` appends a
   `"hera_sim v{__version__}: Added {component} using parameters: …"`
   block (with `defaults._unpack_dict` flattening) to
   `self.data.history`, and `_update_seeds` writes the chosen
   seeds into `self.data.extra_keywords` (with multi-redundant
   seeds spelled `{component}_seed_{bl_int}`).
7. **Random state book-keeping.** `_seed_rng` supports four modes:
   - `seed=None` — reuse `self._components[key]["rng"]` if present,
     else `np.random.default_rng()`.
   - `seed=int` — use that integer seed exactly once.
   - `seed="once"` — generate one cached seed for the whole array
     (per pol if there is one), so e.g. `pntsrc_foreground` produces
     the same source positions regardless of which antpair-pol is
     being computed.
   - `seed="redundant"` — generate one cached seed *per redundant
     group* (looked up via `self.red_grps`), so all baselines in a
     redundant group share random realizations of `noiselike_eor`
     or `diffuse_foreground`. Conjugation is handled by checking
     the antpairpol cache.
   - `seed="initial"` — generate one seed at the very start of
     `_iteratively_apply` and pass it to every baseline; this is
     intended for `thermal_noise`, where each baseline has its own
     independent realization seeded from a deterministic root.

### 7.3 `get` — recover a previously simulated effect

`Simulator.get(component, key=None)` re-runs the simulation under
the *same* seed and returns the result for a particular antpair / pol /
baseline / antenna. It enforces validation through
`_validate_get_request`, distinguishing multiplicative gains
(retrievable per antenna) from per-baseline visibilities (retrievable
per `(ant1,ant2)`). It honours `seed="initial"` correctly by re-
running the entire array. Conjugation is automatic.

### 7.4 Other public methods

- `apply_defaults(config, refresh=True)` — wrapper around
  `defaults.set`.
- `calculate_filters(delay_filter_kwargs=None, fringe_filter_kwargs=None)` —
  pre-computes one delay filter and one fringe filter per redundant
  group (`utils.gen_delay_filter`, `utils.gen_fringe_filter`).
- `plot_array()` — convenience matplotlib scatter of antenna ENU
  positions.
- `refresh()` — zero `data_array`, clear history, components,
  caches, seeds, extras.
- `write(filename, save_format="uvh5", **kwargs)` — dispatches
  `self.data.write_{save_format}`.
- `chunk_sim_and_save(save_dir, ref_files=None, Nint_per_file=None,
  prefix=None, sky_cmp=None, state=None, filetype="uvh5",
  clobber=True)` — thin wrapper over `io.chunk_sim_and_save` for
  splitting a long simulation into per-file JD chunks that match a
  reference set of HERA files.
- `run_sim(sim_file=None, **sim_params)` — runs a whole list of
  `add(component, **params)` calls from a YAML file or kwargs;
  yields `(component, value)` for any component with `ret_vis=True`.
- Legacy `add_eor`, `add_foregrounds`, `add_noise`, `add_rfi`,
  `add_gains`, `add_sigchain_reflections`, `add_xtalk` shims, all
  marked deprecated since v1.0 and slated for removal in v2.0.

### 7.5 Filtering (`_apply_filter`)

`vis_filter` accepts a flexible vocabulary, all parsed by
`Simulator._apply_filter`:

| Format | Effect |
|---|---|
| `(0,)` | Apply only to baselines including antenna 0. |
| `('xx',)` | Apply only to polarization `'xx'`. |
| `(0, 'yy')` | Apply only to antenna 0 with pol `'yy'`. |
| `(0, 1)` | Apply only to baseline `(0,1)`. |
| `(0, 1, 'yy')` | Apply only to baseline `(0,1)` pol `'yy'`. |
| `[(0,), (1,)]` (multi-key) | All keys must match (logical AND across rules). |
| Mixed lists of antennas + pols | Apply to the intersection of the two. |

---

## 8. Antenna positions (`antpos.py`)

Two registered `Array` models, both invoked by their module-level
singletons (`linear_array`, `hex_array`):

### 8.1 `LinearArray(sep=14.6)`
A purely east-west line:
`antpos[j] = [j*sep, 0, 0]` for `j in range(nants)`. Output is
`{int → np.ndarray of length 3}`. The default 14.6 m separation is
HERA's nominal redundant unit.

### 8.2 `HexArray(sep=14.6, split_core=True, outriggers=2)`
Builds a HERA-shaped hex with optional core split (the "split core"
trident layout that loses `N` antennas and adds a small inset to
each of the three sectors) and optional outriggers (the first ring
sits at the exterior of `hex_num=3` hexagonal radius, and each ring
adds `3R^2 + 9R` antennas). The construction is a faithful port of
the HERA siting requirement to (a) be redundantly calibratable
within the core and (b) have a fully-sampled UV plane when
combined with outriggers, while also (c) avoiding specific HERA-
site obstacles such as the road to MeerKAT. The sector logic
inside the `if split_core:` and `if outriggers:` blocks is hand-
tuned and not reduced to a generic geometric expression.

### 8.3 `idealize_antpos(antpos, bl_error_tol=1.0)`

(Imports `hera_cal.redcal` lazily — requires the `cal` extra.)
Snaps real antenna positions onto a perfectly-redundant integer
lattice via `redcal.get_reds` + `redcal.reds_to_antpos`, then
solves a 12-parameter rigid-body transform (`scipy.optimize.least_squares`)
to put the snapped positions back into the original real-space
frame. Used by `VisibilitySimulation` when
`snap_antpos_to_grid=True` to expose perfect redundancy to the
back-end visibility simulator while still placing the array at the
correct geographic location.

---

## 9. The "rough" sky and systematic models

### 9.1 `eor.NoiselikeEoR` (`eor.py`)

```
NoiselikeEoR(
    eor_amp=1e-5,
    min_delay=None, max_delay=None,
    fringe_filter_type="tophat",
    fringe_filter_kwargs=None,
    rng=None,
)
```

Class flags: `is_smooth_in_freq=False`, `is_randomized=True`,
`return_type="per_baseline"`, `attrs_to_pull={"bl_vec": None}`.
Algorithm:

1. Generate `(nlsts, nfreqs)` complex white noise scaled by
   `eor_amp`.
2. Apply a "rough" delay filter (`utils.rough_delay_filter`) with
   `bl_len_ns=1e10` (effectively infinite — the actual
   bandwidth-limit comes from `min_delay`/`max_delay` override).
3. Apply a fringe-rate filter (`utils.rough_fringe_filter`) using
   the East component of `bl_vec` as the projected EW length.
4. For autocorrelations (zero baseline length), the result is taken
   `|·|` to make it real-valued and positive.

Recommended seeding: `seed="redundant"` so all baselines in a
redundant group share the EoR realization (else the stack is
incoherent).

### 9.2 `foregrounds.DiffuseForeground` (`foregrounds.py`)

```
DiffuseForeground(
    Tsky_mdl=None,                    # required
    omega_p=None,                     # required
    delay_filter_kwargs={"standoff": 0.0, "delay_filter_type": "tophat", "normalize": None},
    fringe_filter_kwargs={"fringe_filter_type": "tophat"},
    rng=None,
)
```

Algorithm:

1. `Tsky = Tsky_mdl(lsts, freqs)` (Kelvin).
2. Convert `Tsky` to Jansky via `utils.jansky_to_kelvin(freqs, omega_p)`
   (which returns the Jy→K factor; division gives K→Jy).
3. For autocorrelations, return that "K-converted-to-Jy" array
   directly (no delay/fringe structure).
4. For cross-correlations, multiply by complex white noise
   (`utils.gen_white_noise`), then delay-filter and fringe-filter
   roughly. The delay filter uses the *full* baseline length; the
   fringe filter uses the EW component.

This is the canonical "fast diffuse foregrounds" model used in HERA
power-spectrum testing. The model is *not* internally consistent
across baselines (each baseline gets its own white-noise realization)
unless `Simulator` is used with `seed="redundant"`. The model is also
not invariant under conjugation (because the rough delay filter is
symmetric); the `Simulator._iteratively_apply` antpairpol cache fixes
this by using `np.conj` for the conjugate baseline.

### 9.3 `foregrounds.PointSourceForeground`

```
PointSourceForeground(
    nsrcs=1000, Smin=0.3, Smax=300, beta=-1.5,
    spectral_index_mean=-1, spectral_index_std=0.5,
    reference_freq=0.15,    # GHz
    rng=None,
)
```

Algorithm:

1. Random sources: `nsrcs` RAs uniform in `[0, 2π)`, spectral
   indices Normal(`spectral_index_mean`, `spectral_index_std`),
   fluxes drawn from a power-law `S^β dS` between `Smin` and
   `Smax` (in Jy) via the closed-form inverse-CDF
   `S = (S_max^α + S_min^α (1-u))^(1/α)` with `α = β + 1`.
2. Beam: a hard-coded HERA Gaussian of FWHM `40' × (f0/freqs)`
   scaled by `2π / sday`, truncated at the horizon.
3. For each source:
   - Find the LST index where the source crosses the meridian.
   - Add the source flux at that LST as `flux * (freqs/f0)^index`.
   - Multiply by a phase `exp(2πi · ν · dτ)` where `dτ` is a
     uniform random delay in `±0.1 * bl_len_ns` (to mimic the NS
     component of the baseline).
4. For each frequency, convolve the source train (in time) with
   `beam[ha] * exp(2πi · 0.9 · bl_len_ns · sin(ha) · ν)` via FFT.

The 0.9 and 10% factors are stated in source-comments to be
heuristic encodings of the assumed NS component of the baseline.
Use at your own risk per the model's docstring.

### 9.4 `noise.ThermalNoise` (`noise.py`)

```
ThermalNoise(
    Tsky_mdl=None, omega_p=None,
    integration_time=None, channel_width=None,
    Trx=0,
    autovis=None, antpair=None,
    rng=None,
)
```

Class flags: `is_randomized=True`, `return_type="per_baseline"`,
`attrs_to_pull={"autovis": None, "antpair": None}`. Algorithm:

1. If `antpair[0] == antpair[1]` (an autocorrelation): return a
   pure receiver-temperature bias `Trx /
   utils.jansky_to_kelvin(freqs, omega_p)`. The model rationale is
   that autocorrelation SNR is so high that adding the noise
   realization is unjustified; we just bias by the receiver
   temperature.
2. Else, compute `Tsky` either from `autovis * jansky_to_kelvin`
   (if `autovis` is provided and non-zero) or from
   `resample_Tsky(lsts, freqs, Tsky_mdl)`. `resample_Tsky` falls
   back to a `Tsky=180 * (freqs/0.18)^-2.5` power-law if no model
   is provided.
3. Add `Trx` to `Tsky`, divide by `√(Δt · Δν)` (radiometer
   equation), convert to Jy, and modulate by `gen_white_noise`.

Defaults: integration time is computed from `mean(np.diff(lsts))`
expressed as a fraction of a sidereal day; channel width is
`mean(np.diff(freqs)) * 1e9`; `omega_p` falls back to the H1C beam
polynomial `np.polyval(np.load("HERA_H1C_BEAM_POLY.npy"), freqs)`.

The legacy module-level functions `sky_noise_jy(lsts, freqs, **)`
and `resample_Tsky` are kept for back-compat. `white_noise(*args)`
is a deprecated thin shim around `utils.gen_white_noise`. The
package also exposes `noise.HERA_Tsky_mdl` — a dict
`{"xx": Tsky(...), "yy": Tsky(...)}` constructed eagerly at import
time from `HERA_Tsky_Reformatted.npz`.

### 9.5 RFI (`rfi.py`)

`rfi.py` provides four `RFI`-component models plus a helper
`RfiStation` building block:

- **`RfiStation(f0, duty_cycle=1, strength=100, std=10, timescale=100, rng=None)`** —
  Models a single broadcast station at frequency `f0`. Random
  magnitude in each LST drawn `Normal(strength, std)`, complex phase
  drawn at random per call. The on/off state follows a sinusoidal
  duty cycle of period `timescale` seconds. Spillover into the
  adjacent channel is handled by linear taper based on the distance
  `|freqs[ch] - f0|/channel_width`. Returns a `(nlsts, nfreqs)`
  complex array.
- **`Stations(stations=None, rng=None)`** (alias `rfi_stations`) —
  Iterates a list of `RfiStation` instances (or 5-tuples that are
  auto-converted, or a `.npy` filepath that is loaded), summing
  their contributions. Used with the bundled
  `HERA_H1C_RFI_STATIONS.npy` / `HERA_H2C_RFI_STATIONS.npy`.
- **`Impulse(impulse_chance=0.001, impulse_strength=20.0, rng=None)`**
  (alias `rfi_impulse`) — Inject broad-band, narrow-time impulses;
  for each LST a Bernoulli(`chance`) draw decides whether to
  inject; impulses get a random delay `U(-300, 300) ns` and the
  signal is `strength * exp(2πi · dly · ν)`.
- **`Scatter(scatter_chance=1e-4, scatter_strength=10, scatter_std=10, rng=None)`**
  (alias `rfi_scatter`) — Random sprinkles of RFI in `(lst, freq)`
  bins; one common amplitude (`Normal(strength, std)`) and per-bin
  random phases.
- **`DTV(dtv_band=(0.174, 0.214), dtv_channel_width=0.008, dtv_chance=1e-4, dtv_strength=10, dtv_std=10, rng=None)`**
  (alias `rfi_dtv`) — Models digital-TV broadcasts: the band is
  divided into 8 MHz subbands, each with its own scalar or per-band
  array of `(chance, strength, std)`. Per LST, Bernoulli sampling
  decides whether the subband is occupied; occupied bins get
  `Normal(strength, std)` amplitudes with random phases.

All four models share `is_randomized=True`,
`return_type="per_baseline"`.

### 9.6 The signal chain (`sigchain.py`)

`sigchain.py` is the largest module (≈ 1700 lines) and packages
nine concrete models split across two abstract bases (`Gain` and
`Crosstalk`) plus three module-level helpers (`apply_gains`,
`vary_gains_in_time`, `gen_*` legacy aliases).

#### `Gain` models

- **`Bandpass(gain_spread=0.1, dly_rng=(-20, 20), bp_poly=None, taper=None, taper_kwds=None, rng=None)`**
  (aliases `gains`, `bandpass_gain`). `is_multiplicative=True`,
  `return_type="per_antenna"`, `attrs_to_pull={"ants": "antpos"}`.
  Internally:
  - `_gen_bandpass(freqs, ants, gain_spread, bp_poly, rng)` — base
    bandpass `np.polyval(bp_poly, freqs)` (with the H1C polynomial
    as default), then per-antenna `δbp = ifft(white_noise * |fft(window
    * bp_base)| * gain_spread)` where the window is
    `scipy.signal.windows.blackmanharris(nfreqs)`. This produces a
    smooth, antenna-specific perturbation around the nominal
    bandpass.
  - `_gen_delay_phase(freqs, ants, dly_rng, rng)` — per-antenna
    `delay = U(*dly_rng)` ns, then `phase = exp(2πi · delay · freqs)`.
  - `taper` may be `None`, a callable, an `np.ndarray`, or a string
    (`"tanh"` triggers `utils.tanh_window`; any other string is
    forwarded to `uvtools.dspec.gen_window` if available).
  - Returns `{ant: bandpass[ant] * phase[ant] * taper}`.
- **`Reflections(amp=None, dly=None, phs=None, conj=False, amp_jitter=0, dly_jitter=0, rng=None)`**
  (aliases `reflection_gains`, `sigchain_reflections`). Multiplicative
  `1 + ε(ν)` per antenna where `ε = amp * exp(2πi · ν · dly + i · phs)`,
  with optional Normal jitter on amp / dly. The static method
  `gen_reflection_coefficient(freqs, amp, dly, phs, conj=False)` is
  the single source of truth for the reflection-coefficient shape
  (and is reused by `CrossCouplingCrosstalk`).
- **`ReflectionSpectrum(n_copies=20, amp_range=(-3, -4), dly_range=(200, 1000), phs_range=(-π, π), amp_jitter=0.05, dly_jitter=30, amp_logbase=10, rng=None)`**
  (alias `reflection_spectrum`). A spectrum of `n_copies`
  reflections; `amps = logspace(*amp_range, n_copies, base=amp_logbase)`
  and `dlys = linspace(*dly_range, n_copies)`, all multiplied
  together via `Reflections`.

#### `Crosstalk` models

- **`CrossCouplingCrosstalk(amp, dly, phs, conj, amp_jitter, dly_jitter, rng)`**
  (alias `cross_coupling_xtalk`). Inherits both `Crosstalk` and
  `Reflections` — uses `gen_reflection_coefficient` to build the
  cross-coupling visibility `vis = autovis * ε`. `attrs_to_pull =
  {"autovis": None}`. `is_multiplicative=False`,
  `return_type="per_baseline"`.
- **`CrossCouplingSpectrum(n_copies=10, amp_range=(-4, -6), dly_range=(1000, 1200), phs_range=(-π, π), amp_jitter=0, dly_jitter=0, amp_logbase=10, symmetrize=True, rng=None)`**
  (aliases `cross_coupling_spectrum`, `xtalk_spectrum`). Sums
  `n_copies` `CrossCouplingCrosstalk` realizations at logarithmically-
  spaced amplitudes and linearly-spaced delays; if `symmetrize`,
  also adds the conjugate-delay copy.
- **`MutualCoupling(uvbeam, reflection, omega_p, ant_1_array, ant_2_array, pol_array, array_layout, coupling_matrix=None, pixel_interp="az_za_simple", freq_interp="cubic", beam_kwargs=None, use_numba=True)`**
  (aliases `mutual_coupling`, `first_order_coupling`).
  `return_type="full_array"`. Implements the Josaitis+ 2022
  first-order mutual-coupling model
  (<https://doi.org/10.1093/mnras/stac916>,
  <https://arxiv.org/abs/2110.10879>). The full math is in the
  source-level docstring; the implementation:
  1. Sanity-checks beams (must be `pyuvdata.UVBeam` in az/za with
     `beam_type="efield"`, or `AnalyticBeam`).
  2. Reshapes the input visibility `(nblts, nfreq, npols)` into a
     `(ntimes, nfreq, 2*nant, 2*nant)` matrix with even/odd indices
     mapping to X/Y feeds (`utils.reshape_vis`,
     numba-accelerated).
  3. Computes the coupling matrix `X_jk = (iΓ/Ω_p) · exp(2πiν τ_jk)
     · J(b_jk) J(b_kj)^† / b_jk` per unique baseline orientation
     (using `utils.find_baseline_orientations`). The Jones matrix
     `J` is computed by interpolating the beam at the horizon
     `ZA = π/2` and the corresponding `AZ`.
  4. Returns `xt_vis = X @ V0 + (X @ V0)^†.transpose(0,1,3,2)` so
     the corrected visibility is symmetric under feed transposition.
  5. Reshapes back to UVData layout.

  This is the most expensive of the systematics models; the
  `utils.matmul`, `_left_matmul`, `_right_matmul`, `_matmul`
  numba-jitted kernels exist primarily to make this affordable.
  See `docs/tutorials/mutual_coupling_example.ipynb` for a worked
  example.
- **`OverAirCrossCoupling(emitter_pos, cable_delays, base_amp, amp_norm, amp_slope, amp_decay_base, n_copies, amp_jitter, dly_jitter, max_delay, amp_decay_fac, rng)`**.
  Implements HERA Memo 104's "receiverator" crosstalk model:
  signal from antenna `i` travels down its cable
  (delay `τ_{i,cable}`) to a re-radiator at position `emitter_pos`
  (typically the receiverator), then over-the-air to antenna `j`
  (delay `τ_{X→j}`). Amplitude `A_i = a · |r_i - r_X|^β`. Builds two
  symmetric `CrossCouplingSpectrum` realizations (one for `ij`, one
  for `ji`) and sums their contributions weighted by `autovis_i`
  and `autovis_j`.
- **`WhiteNoiseCrosstalk(amplitude=3.0, rng=None)`** (aliases
  `whitenoise_xtalk`, `white_noise_xtalk`). Convolves `gen_white_noise`
  with a 50-channel uniform kernel (or `nfreqs/2` if smaller) and
  scales by `amplitude`.

#### Module-level helpers

- `apply_gains(vis, gains, bl)` — apply `g_i · conj(g_j)` to a
  per-baseline visibility, broadcasting if `gain.ndim == 1`.
- `vary_gains_in_time(gains, times, freqs=None, delays=None, parameter="amp"|"phs"|"dly", variation_ref_time=None, variation_timescale=None, variation_amp=0.05, variation_mode="linear"|"sinusoidal"|"noiselike", rng=None)` —
  Adds a time-varying envelope to a static gain dict. The "linear"
  mode produces a triangle wave; "sinusoidal" is `sin(2π · phase)`;
  "noiselike" is `Normal(1, amp)`. For `parameter="dly"`, the
  delays are perturbed in time and re-converted to phase.
- `gen_gains = Bandpass()`, `gen_bandpass = gen_gains._gen_bandpass`,
  `gen_delay_phs = gen_gains._gen_delay_phase`,
  `gen_reflection_coefficient = Reflections.gen_reflection_coefficient`,
  `gen_reflection_gains = Reflections()`,
  `gen_whitenoise_xtalk = WhiteNoiseCrosstalk()` — module-level
  singletons / static-method aliases retained for back-compat with
  the pre-OO `hera_sim` API.

---

## 10. Beams (`beams.py`)

Three analytic-beam models, all subclasses of
`pyuvdata.analytic_beam.AnalyticBeam` (so they slot directly into
`pyuvsim`/`matvis`/`fftvis` `BeamList`s):

### 10.1 `PolyBeam`

```
@dataclass
PolyBeam(beam_coeffs: list[float], spectral_index: float = 0.0,
         ref_freq: float = 1e8, polarized: bool = False)
```

Azimuthally-symmetric Chebyshev beam:
`B(θ, ν) = chebval(2 sin(θ / fscale) - 1, beam_coeffs) / chebval(-1, beam_coeffs)`,
with frequency scaling `fscale = (ν / ref_freq)^spectral_index`. The
beam is normalized to unity at zenith. When `polarized=True`, the
axisymmetric pattern is multiplied by a "Fagnoni-beam" dipole
modulation pattern (`modulate_with_dipole`, with helper functions
`p(za)` and `q(za)` that fit the first/second rings of the 100 MHz
HERA Fagnoni beam by hand). The classmethod
`PolyBeam.like_fagnoni19(**kwargs)` returns a 18-coefficient Cheby
fit to the HERA-19 beam at 100 MHz with `spectral_index=-0.6975`.

### 10.2 `PerturbedPolyBeam(PolyBeam)`

Adds three independent perturbations to the base `PolyBeam`:

- **Mainlobe perturbation:** subtract a Gaussian of FWHM `mainlobe_width`
  and add a Gaussian of FWHM `mainlobe_width * mainlobe_scale`.
- **Sidelobe perturbation:** Fourier (sine) series in `za` (period `π/2`)
  multiplied by a `tanh` step function centred at `mainlobe_width`
  with width `transition_width`, plus a Fourier (sine + cosine)
  series in frequency (period 100 MHz). Both modulations are
  rescaled to `[-0.5, 0.5]` automatically (or the user can override
  via `perturb_zeropoint`).
- **Beam ellipticity / rotation:** a 2-D shear in `(x, y)` with
  `xstretch`, `ystretch`, and an in-plane rotation `rotation` (deg).

`PerturbedPolyBeam.like_fagnoni19(**kwargs)` returns a beam with the
default 8-term sidelobe perturbation found to mimic the 100 MHz
HERA-19 sidelobe ringing.

### 10.3 `ZernikeBeam`

```
@dataclass
ZernikeBeam(beam_coeffs: NDArray[float], spectral_index=0.0,
            ref_freq=1e8, peak_normalized=True)
```

Zernike-polynomial beam up to degree 66 (Z_1 .. Z_66). The internal
`zernike(coeffs, x, y)` static method enumerates every Zernike
polynomial up to `n=10` as a closed-form polynomial in `x, y` and
returns their coefficient-weighted sum. Frequency scaling is the
same `(ν/ref_freq)^spectral_index` as `PolyBeam`. Used in
`docs/tutorials/polybeam_simulation.ipynb` for illustrating how to
plug a Zernike beam into a visibility simulation.

---

## 11. Interpolators (`interpolators.py`)

All interpolators check `_check_path` first: a non-absolute path is
resolved against `DATA_PATH`. They support two file types:

- `.npy` — read with `_read_npy` (pure ndarray).
- `.npz` — read with `_read_npz` (dict-cast for picklability).

### 11.1 `Tsky`

Wraps a 2-D `RectBivariateSpline(lsts, freqs, tsky_data)` with
explicit LST wrapping (the first/last 10 LSTs are duplicated below
0 and above 2π). Required npz keys: `tsky` `(npols, nlsts, nfreqs)`,
`freqs` (GHz), `lsts` (rad), `meta` (dict containing `pols` tuple).
`pol` interp_kwarg selects the polarization slice. Used as the
`!Tsky` YAML constructor target.

### 11.2 `FreqInterpolator` / `Beam` / `Bandpass`

`FreqInterpolator(datafile, obj_type=None, **interp_kwargs)` chooses
between `np.poly1d` (default, requires `.npy` of polynomial
coefficients in decreasing order) and `scipy.interpolate.interp1d`
(requires `.npz` with `freqs` and `obj_type` arrays). `Beam` and
`Bandpass` are `FreqInterpolator(obj_type="beam"/"bandpass")` thin
subclasses.

### 11.3 `Reflection`

Defaults to `interp1d`. Stores both the real and imaginary parts of
the complex reflection coefficient as separate `interp1d` objects
(cubic by default), recombining at call time as `re + 1j · im`.

---

## 12. I/O (`io.py`)

### 12.1 `empty_uvdata(...)`

Decorated with `@_defaults` so the season-specific frequency / time /
array settings are auto-injected. Calls
`pyuvsim.simsetup.initialize_uvdata_from_keywords` to construct a
fully-populated `UVData` object with default
`telescope_location = HERA_LAT_LON_ALT`, `telescope_name="hera_sim"`,
`polarization_array=["xx"]`, `complete=True`. Handles a small
amount of pyuvdata-version detection (`pyuvdata.__version__ < "2.2.0"`
branch for `set_drift()`; otherwise checks `phase_center_catalog`
and calls `fix_phase()` if needed). Also accepts back-compat aliases
`n_freq → Nfreqs`, `n_times → Ntimes`, `antennas → array_layout`,
which raise a `DeprecationWarning`.

### 12.2 `chunk_sim_and_save(sim_uvd, save_dir, ref_files=None, Nint_per_file=None, prefix=None, sky_cmp=None, state=None, filetype="uvh5", clobber=True)`

Splits a long simulation into per-file JD chunks. Either provide
`ref_files` (a list of HERA observational files, from which the JD
pattern `*.{jd_major:7d}.{jd_minor:5d}.*` is extracted via regex)
or `Nint_per_file` directly. Output filenames follow
`save_dir/[{prefix}.]{jd:.5f}[.{sky_cmp}][.{state}].{filetype}`.

---

## 13. Visibility-simulator framework (`visibilities/`)

### 13.1 `ModelData` (`simulators.py`)

```
ModelData(
    *,
    uvdata: UVData | str | Path,
    sky_model: pyradiosky.SkyModel,
    beam_ids: dict[str, int] | Sequence[int] | None = None,
    beams: BeamList | list[AnalyticBeam | UVBeam] | None = None,
    normalize_beams: bool = False,
)
```

Validation lattice:

- `uvdata` may be a path (read with `UVData()`) or an in-memory
  object; `set_rectangularity(force=True)` is called if missing.
- `beams` defaults to `[UniformBeam()]`; if not already a
  `pyuvsim.BeamList`, wraps it (using the first beam's `beam_type`,
  defaulting to `"efield"`); `peak_normalize()` is applied if
  `normalize_beams=True`.
- `beam_ids` may be `None` (one beam → all data ants point to
  index 0; `n_ant` beams → 1:1 mapping by data-ant order), a
  `list/tuple/ndarray` of `len == n_ant` (becomes `{ant_name: idx}`),
  or already a dict.
- `sky_model.at_frequencies(self.freqs * u.Hz)` is called eagerly so
  every back-end can rely on a frequency-resampled catalog.
- `_validate()` raises if any beam is `power` while the sky model
  has any non-zero Q/U/V.
- `cached_property`s: `lsts`, `times` (rectangular-aware),
  `freqs`, `n_beams`.

`ModelData.from_config(config_file, normalize_beams=False)` is the
preferred entry point — it calls
`pyuvsim.simsetup.initialize_uvdata_from_params(config_file,
return_beams=True)` (with `reorder_blt_kw={}`,
`check_kw={'run_check_acceptability': False, 'check_extra': False}`),
then `pyuvsim.simsetup.initialize_catalog_from_params(config_file)`,
then `pyuvsim.simsetup._complete_uvdata(uvdata, inplace=True)`. The
returned object has the full obsparam pipeline applied.

`ModelData.write_config_file(filename, direc=".", beam_filepath=None,
antenna_layout_path=None)` is the inverse — it writes a
pyuvsim-style telescope config, beam list, and antenna layout CSV
for the live UVData object.

### 13.2 `VisibilitySimulator` ABC

| Class attribute | Default | Meaning |
|---|---|---|
| `point_source_ability` | `True` | Whether the simulator handles a point-source `pyradiosky.SkyModel` directly. |
| `diffuse_ability` | `False` | Whether the simulator handles `component_type="healpix"` skies directly. |
| `_functions_to_profile` | `()` | Tuple of underlying functions for `hera-sim-vis.py --profile`. |
| `_blt_order_kws` | `None` | Keywords passed to `UVData.reorder_blts(...)` before simulation; the `MatVis` and `FFTVis` simulators set this to `None`/leave None, but `UVSim` sets `{"order": "time", "minor_order": "baseline"}`. |
| `__version__` | `"unknown"` | The simulator's own version (filled in by subclasses). |

Mandatory subclass override: `simulate(self, data_model: ModelData) -> np.ndarray`.

Optional overrides:

- `validate(self, data_model)` — raise/print sanity checks (e.g.
  `MatVis.validate` insists that every antenna pair is present and
  that polarizations are linear).
- `_from_yaml_dict(cls, cfg)` — transform a YAML dict before
  passing to `cls(**cfg)`.
- `estimate_memory(self, data_model) -> float` — return GB.
- `compress_data_model(self, data_model)` /
  `restore_data_model(self, data_model)` — temporarily reduce
  memory pressure (`MatVis.compress_data_model` zeroes
  `uvw_array` and reduces `integration_time` to a scalar before
  the simulate call, then restores them).

`load_simulator_from_yaml(config)` reads the YAML, pops the
`simulator:` key, looks it up in
`hera_sim.visibilities.SIMULATORS = {"UVSim": ..., "MatVis": ...,
"FFTVis": ...}` (or imports a dotted path for a custom simulator),
and calls `cls.from_yaml(cfg)`.

### 13.3 `VisibilitySimulation` dataclass

```
@dataclass
VisibilitySimulation(
    data_model: ModelData,
    simulator: VisibilitySimulator,
    n_side: int = 32,
    snap_antpos_to_grid: bool = False,
    keep_snapped_antpos: bool = False,
)
```

Workflow:

1. `__post_init__`:
   - If `simulator._blt_order_kws` is set, calls
     `data_model.uvdata.reorder_blts(**kws)`.
   - `simulator.validate(data_model)`.
   - If the sky model is HEALPix but the simulator can't do diffuse,
     `sky_model.healpix_to_point()`.
   - If the sky model is point but the simulator can't do points
     (none of the bundled ones currently), call
     `_convert_point_to_healpix` (deposits each source's flux into
     the corresponding HEALPix pixel of `nside=n_side`, scaled by
     `1/pixel_area` to get the brightness temperature).
2. `simulate()`:
   - If `snap_antpos_to_grid`, save the original antpos, then call
     `antpos.idealize_antpos` and overwrite the telescope's antpos.
   - `simulator.compress_data_model(data_model)`.
   - `vis = simulator.simulate(data_model)`.
   - `self.uvdata.data_array += vis`.
   - `_write_history` appends a `Visibility Simulation performed
     with hera_sim's {Cls} simulator …` block with the simulator's
     repr and version.
   - `simulator.restore_data_model(data_model)`.
   - If `not keep_snapped_antpos`, restore the original antpos.

### 13.4 `UVSim` (`pyuvsim_wrapper.py`)

Thin wrapper around `pyuvsim.uvsim.run_uvdata_uvsim`:

```
UVSim(quiet=True)
_blt_order_kws = {"order": "time", "minor_order": "baseline"}
_functions_to_profile = (pyuvsim.uvsim.run_uvdata_uvsim,)
```

`simulate(data_model)`:
1. Build `beam_dict = {ant_name: beam_id}`.
2. Coerce `sky_model.name` to `np.array` (workaround for an unmerged
   pyuvsim PR).
3. Issue `warnings.warn` and force `data_model.uvdata.reorder_blts("time")`.
4. If pol order isn't AIPS (`[-5,-6,-7,-8]`), warn and call
   `reorder_pols("AIPS")`.
5. Call `pyuvsim.uvsim.run_uvdata_uvsim(input_uv=…,
   beam_list=…, beam_dict=…, catalog=SkyModelData(sky_model),
   quiet=…)`.
6. Return `out_uv.data_array`.

This is the slowest but most-canonical back-end; it is tested in
`tests/test_visibilities/`.

### 13.5 `MatVis` (`matvis.py`)

```
MatVis(
    precision: int = 2,           # 1 = float32/complex64, 2 = float64/complex128
    use_gpu: bool = False,
    mpi_comm = None,
    check_antenna_conjugation: bool = True,
    **kwargs,                     # forwarded to matvis.cpu/gpu.simulate
)
```

`simulate(data_model)`:
1. Determine if polarized: `len(uvdata.polarization_array) != 1` or
   the single pol isn't `'xx' / 'yy'`.
2. Get antenna positions via `data_model.uvdata.get_enu_data_ants()`.
3. Build `beam_ids = [data_model.beam_ids[num2name[i]] for i in
   ant_list]`.
4. For each frequency (skipped if `mpi_comm` round-robin), call
   `matvis.cpu.simulate` (or `matvis.gpu.simulate` if `use_gpu`)
   with `antpos`, `freq`, `times=Time(data_model.times,
   format="jd")`, `skycoords=sky_model.skycoord`,
   `telescope_loc=uvdata.telescope.location`,
   `I_sky=sky_model.stokes[0,i].to("Jy").value`,
   `beam_list`, `beam_idx=beam_ids`,
   `beam_spline_opts=beams.spline_interp_opts`, `precision`,
   `polarized`, `antpairs=[[antlist.index(a), antlist.index(b)] for a,b in antpairs]`.
5. `_reorder_vis` deposits the matvis output back into `visfull`
   in `UVData.data_array` shape; the fast path (rectangular,
   `not time_axis_faster_than_bls`, sorted req_pols) just reshapes
   in place.
6. `_reduce_mpi` does an MPI-SUM reduction on rank 0 if needed.

`compress_data_model` zeroes `uvw_array` and replaces
`integration_time` with a scalar to save memory; `restore_data_model`
re-broadcasts. Validations: every antpair must be present (`Nbls ==
n_ant*(n_ant+1)/2`); blts must be rectangular; if
`check_antenna_conjugation`, no baseline may appear under both
`(i,j)` and `(j,i)` orderings; only linear pols are allowed
(`-5,-6,-7,-8`); for polarized sims, the beam must have `Nfeeds ==
2`.

`estimate_memory` is a closed-form back-of-the-envelope sum of the
sizes of the visibility array, per-antenna vis, raw beam,
interpolated beam, antenna positions, source fluxes, rotation
matrices, source positions, in units of `precision * 4 / 1024^3` GB.

### 13.6 `FFTVis` (`fftvis.py`)

```
FFTVis(
    *,
    precision: int = 2,
    mpi_comm = None,
    check_antenna_conjugation: bool = True,
    **kwargs,                     # forwarded to fftvis.CPUSimulationEngine().simulate
)
```

Same flow as `MatVis`, with two key differences:

- Only one beam is allowed (`len(data_model.beams) != 1` → raise);
  for unpolarized sims the beam is run through
  `matvis.core.beams.prepare_beam_unpolarized`.
- Per-frequency call is to `fftvis.CPUSimulationEngine().simulate(…)`.
- The output transposition logic in `_reorder_vis` differs:
  polarized output is `(0, 3, 1, 2)`-transposed, then reshaped.

`estimate_memory` includes a per-frequency FFT-grid term:
`n_gridx = 8 · ν · max_blx / c`, `n_gridy = 8 · ν · max_bly / c`.

### 13.7 The `hera-sim-vis.py` CLI (`visibilities/cli.py`)

`vis_cli_argparser()` builds an argparser with positional args
`obsparam` (a `pyuvsim`-style obsparam YAML) and `simulator_config`
(the YAML with `simulator: <ClassName>` plus the simulator's
constructor params), and flags:

- `--object_name`
- `--compress {file}` — compress by redundancy, caching the
  selected `blt_inds` array to `{file}` for re-use across runs.
- `--normalize_beams` — peak-normalize beams.
- `--fix_autos` — set autocorrelations to be purely real (strips
  small imaginary numerical noise).
- `--max-auto-imag` — threshold above which an auto-imag failure is
  raised (default `5e-14`).
- `-d, --dry-run` — set up but don't run.
- `--run-auto-check` — run the auto-imag check.
- `--phase-center-name` — override the (single-entry) phase-center
  name.

`run_vis_sim(args)`:
1. `simulator = load_simulator_from_yaml(args.simulator_config)`.
2. `data_model = ModelData.from_config(args.obsparam,
   normalize_beams=args.normalize_beams)`.
3. Print versions of `pyuvdata`, `pyuvsim`, `pyradiosky`,
   `hera_sim`, the simulator, plus simulation cardinals (Nfreqs,
   Ntimes, Npols, Nants, Nsources, Nbeams, vis-array MB, beam-array
   MB, RAM estimate vs available).
4. If `--dry-run`, exit.
5. `simulation = VisibilitySimulation(data_model, simulator)`;
   `simulation.simulate()`.
6. (rank-0 only) Run optional auto-check, then optional redundancy
   compression, then write `data_model.uvdata.write_uvh5(outfile,
   clobber=…, run_check=False, run_check_acceptability=False)`,
   where `outdir` / `outfile_name` / `output_format` come from the
   `filing:` block of the obsparam YAML.

The CLI is wired up by `scripts/hera-sim-vis.py`, which uses
`hera_cli_utils.parse_args(...)` and `run_with_profiling(...)`
(profiles `simulator._functions_to_profile` if `--profile` is
passed).

### 13.8 The empty `vis_cpu.py`

`src/hera_sim/visibilities/vis_cpu.py` is **a 0-byte file** in this
revision — a vestigial pointer to the legacy `VisCPU` simulator
that was removed in `v4.0.0` (`CHANGELOG.rst`: "Removed the HealVis
wrapper. Use pyuvsim instead." and the v4.x reorg). The
docs/reference still mention `hera_sim.visibilities.vis_cpu.VisCPU`
in `docs/reference/index.rst`, but the class no longer exists —
the analogous capability is provided by `MatVis`.

---

## 14. Utilities (`utils.py`)

The most-used library inside `hera_sim`. Exposes:

- **Antenna positions:**
  - `get_antpos_dict(uvd, *, data_ants=False, frame="enu" | "ecef")`.
- **Baseline geometry:**
  - `_get_bl_len_vec(bl_len_ns)` — coerce scalar / 2-vec / 3-vec to
    a length-3 ENU array.
  - `get_bl_len_magnitude(bl_len_ns)`.
- **Filters (delay):**
  - `gen_delay_filter(freqs, bl_len_ns, standoff=0, delay_filter_type="gauss"|"trunc_gauss"|"tophat"|"none", min_delay=None, max_delay=None, normalize=None)`. The horizon is at `4σ`; `gauss` is `exp(-0.5 (τ/σ)²)`, `trunc_gauss` zeroes outside `4σ`, `tophat` zeroes outside `4σ`, `none` is unity.
  - `rough_delay_filter(data, freqs=None, bl_len_ns=None, *, delay_filter=None, **kwargs)` — FFT, multiply by filter, IFFT. Pre-computed filters bypass the gen step.
- **Filters (fringe-rate):**
  - `gen_fringe_filter(lsts, freqs, ew_bl_len_ns, fringe_filter_type="tophat"|"gauss"|"custom"|"none", **filter_kwargs)`. Fringe rates are `np.fft.fftfreq(times.size, dt)`. `tophat` zeroes outside `|fr| > fr_max(ν)`; `gauss` is centred on `fr_max` with width `fr_width`; `custom` accepts a 2-D `(FR_filter, FR_frates, FR_freqs)` packet and `RectBivariateSpline`-interpolates onto the desired grid.
  - `calc_max_fringe_rate(fqs, ew_bl_len_ns)` = `2π / sday * (ν · b_EW)`.
  - `rough_fringe_filter(data, lsts=None, freqs=None, ew_bl_len_ns=None, *, fringe_filter=None, **kwargs)`.
- **Time / coordinates:**
  - `compute_ha(lsts, ra)` — wrapped to `[-π, +π]`.
  - `wrap2pipi(a)` — wrap any array to `[-π, +π]` via `np.fmod`.
- **Random:**
  - `gen_white_noise(size=1, rng=None)` — complex Gaussian with
    unit variance (real and imaginary each variance `1/2`).
- **Jy↔K:**
  - `jansky_to_kelvin(freqs, omega_p)` — returns the Jy → K factor
    `λ² · 1e-26 / (2 k_B Ω_p)`.
  - `Jy2T` deprecated alias.
- **List handling:**
  - `_listify(x)` — string-aware; never iterates over strings.
- **Mutual coupling helpers:**
  - `reshape_vis(vis, ant_1_array, ant_2_array, pol_array, antenna_numbers, n_times, n_freqs, n_ants, n_pols, invert=False, use_numba=True)` — converts between `(Nblts, Nfreqs, Npols)` and `(Ntimes, Nfreqs, 2*Nants, 2*Nants)` matrix layout used by `MutualCoupling`.
  - `matmul(left, right, use_numba=False)` — falls through to `left @ right` unless numba is available, in which case it dispatches to a custom kernel optimized for the shapes that `MutualCoupling` produces (one of `_left_matmul`, `_right_matmul`, `_matmul`).
  - `find_baseline_orientations(antenna_numbers, enu_antpos)` — uses `pyuvdata.utils.redundancy.get_antenna_redundancies` to enumerate baseline orientation angles in `[0, 2π)` and returns `{(ai, aj): θ}` for both directions.
- **Bandpass shaping:**
  - `tanh_window(x, x_min=None, x_max=None, scale_low=1, scale_high=1)`.
- **Numba kernels** (only compiled when `numba` is importable):
  - `jit_reshape_vis`, `jit_reshape_vis_invert`, `_left_matmul`,
    `_right_matmul`, `_matmul`. All `@numba.njit`-decorated.

---

## 15. YAML constructors (`__yaml_constructors.py`)

Imported eagerly by `hera_sim/__init__.py`, so any `yaml.load(...,
Loader=yaml.FullLoader)` call after `import hera_sim` recognizes the
following tags:

| Tag | Constructor | Effect |
|---|---|---|
| `!Tsky` | `interpolators.Tsky(datafile, **interp_kwargs)` | Sky-temperature interpolator. |
| `!Beam` | `interpolators.Beam(datafile, **interp_kwargs)` | Beam-area interpolator. |
| `!Bandpass` | `interpolators.Bandpass(datafile, **interp_kwargs)` | Bandpass interpolator. |
| `!Reflection` | `interpolators.Reflection(datafile, **interp_kwargs)` | Complex reflection interpolator. |
| `!antpos` | `antpos.{array_type}_array(**params)` | Build antpos dict from `LinearArray` / `HexArray` directly in YAML. |
| `!dimensionful` | `value * astropy.units.<units>` | Build a `Quantity` from `{value, units}`. Returns `None` (with a warning) if `units` is missing. |

This is what allows the `template_config.yaml` to reference
`!Tsky`, `!Beam`, etc., without any custom Python; and what allows
`H1C.yaml` / `H2C.yaml` to embed `!antpos` / `!Bandpass` / `!Beam` /
`!Tsky` directly inside the season-specific defaults.

---

## 16. Configuration recipes

### 16.1 The "rough" simulation YAML (`config_examples/template_config.yaml`)

The full structure (verbatim):

```yaml
bda:                                  # optional, applied last via bda.bda_tools.apply_bda
  max_decorr: 0
  pre_fs_int_time: !dimensionful {value: 0.1, units: 's'}
  corr_FoV_angle: !dimensionful {value: 20, units: 'deg'}
  max_time: !dimensionful {value: 16, units: 's'}
  corr_int_time: !dimensionful {value: 2, units: 's'}
filing:
  outdir: '.'
  outfile_name: 'quick_and_dirty_sim.uvh5'
  output_format: 'uvh5'
  clobber: True
freq:
  n_freq: 100
  channel_width: 122070.3125
  start_freq: 46920776.3671875
time:
  n_times: 10
  integration_time: 8.59
  start_time: 2457458.1738949567
telescope:
  array_layout: !antpos {array_type: hex, hex_num: 3, sep: 14.6, split_core: False, outriggers: 0}
  omega_p: !Beam {datafile: HERA_H2C_BEAM_MODEL.npz, interp_kwargs: {interpolator: interp1d, fill_value: extrapolate}}
defaults: 'h2c'
systematics:
  rfi:
    rfi_stations: {seed: once, stations: !!null}
    rfi_impulse: {impulse_chance: 0.001, impulse_strength: 20.0}
    rfi_scatter: {scatter_chance: 0.0001, scatter_strength: 10.0, scatter_std: 10.0}
    rfi_dtv:    {seed: once, dtv_band: [0.174, 0.214], dtv_channel_width: 0.008, …}
  sigchain:
    gains:                {seed: once, gain_spread: 0.1, dly_rng: [-20,20], bp_poly: HERA_H1C_BANDPASS.npy}
    sigchain_reflections: {seed: once, amp: !!null, dly: !!null, phs: !!null}
  crosstalk:
    gen_whitenoise_xtalk: {amplitude: 3.0}
  noise:
    thermal_noise: {seed: initial, Trx: 0}
sky:
  Tsky_mdl: !Tsky {datafile: HERA_Tsky_Reformatted.npz, interp_kwargs: {pol: xx}}
  eor:
    noiselike_eor: {eor_amp: 1e-5, min_delay: !!null, max_delay: !!null, seed: redundant, fringe_filter_type: tophat}
  foregrounds:
    diffuse_foreground:  {seed: redundant, delay_filter_kwargs: {…}, fringe_filter_kwargs: {…}}
    pntsrc_foreground:   {seed: once, nsrcs: 1000, Smin: 0.3, Smax: 300, beta: -1.5, …}
simulation:
  components: [foregrounds, noise, eor, rfi, sigchain]
  exclude:    [sigchain_reflections, gen_whitenoise_xtalk]
```

Consumed by `scripts/hera-sim-simulate.py`. The control flow:

1. `cli_utils.validate_config(config)` validates `freq:`, `time:`,
   `telescope.array_layout` (or accepts `defaults: <season>`).
2. Filing params merged through `cli_utils.get_filing_params`.
3. `hera_sim.defaults.set(config["defaults"], refresh=True)` when
   present.
4. `hera_sim.defaults.set(instrument_parameters, refresh=False)`
   merges the `freq:` + `time:` + `telescope.array_layout` into the
   active defaults.
5. `sim = hera_sim.Simulator()` (no `data` arg → `io.empty_uvdata`
   fed by the active defaults).
6. `omega_p` and `Tsky_mdl` are pushed into the active defaults.
7. The script walks `simulation.components` in order, looking each
   sub-key up in either `sky:` or `systematics:`, and excluding any
   key in `simulation.exclude`. For each survivor it calls
   `sim.add(component, ret_vis=False, **parameters)` (or
   `ret_vis=True` if `--save_all`).
8. Optionally apply BDA (`bda_tools.apply_bda(sim.data,
   **bda_params)`).
9. Append a history note and `sim.write(args.outfile, …)`.

### 16.2 The visibility-simulator YAML (`config_examples/simulator.yaml`)

```yaml
simulator: MatVis
precision: 2
```

Any other key becomes a kwarg to the simulator constructor. For
`FFTVis` the same file is sufficient with `simulator: FFTVis`. For a
custom out-of-package simulator, `simulator: my.module.MyClass`.

The matching obsparam YAML is a standard `pyuvsim` obsparam (the
test fixtures live under `tests/testdata/hera-sim-vis-config/`).

---

## 17. End-to-end usage patterns

### 17.1 Programmatic (rough sim)

```python
import hera_sim
sim = hera_sim.Simulator(
    Nfreqs=100, start_freq=100e6, channel_width=1e5,
    Ntimes=100, start_time=2458119.5, integration_time=10.7,
    array_layout={0: [0,0,0], 1: [14.6,0,0], 2: [29.2,0,0]},
)

sim.add("diffuse_foreground", Tsky_mdl=hera_sim.noise.HERA_Tsky_mdl["xx"],
        omega_p=hera_sim.Beam("HERA_H1C_BEAM_POLY.npy"),
        seed="redundant")
sim.add("noiselike_eor", eor_amp=1e-5, seed="redundant")
sim.add("thermal_noise", Trx=100, seed="initial")
sim.add("rfi_stations", stations="HERA_H1C_RFI_STATIONS.npy", seed="once")
sim.add("gains", gain_spread=0.1, dly_rng=(-20,20), seed="once")
sim.add("reflections", amp=0.01, dly=200, phs=1.0, seed="once")
sim.add("whitenoise_xtalk", amplitude=3.0, seed="initial")

sim.write("simulation.uvh5", save_format="uvh5", clobber=True)
```

### 17.2 Programmatic (proper visibility sim)

```python
from hera_sim.visibilities import ModelData, VisibilitySimulation, FFTVis
data_model = ModelData.from_config("obsparam.yaml", normalize_beams=True)
simulator  = FFTVis(precision=2)
sim = VisibilitySimulation(data_model=data_model, simulator=simulator,
                           snap_antpos_to_grid=True)
sim.simulate()
sim.uvdata.write_uvh5("vis_sim.uvh5", clobber=True)
```

### 17.3 CLI usage

```bash
# Rough sim (the systematics/foregrounds/noise pipeline)
hera-sim-simulate.py config.yaml --save_all --clobber -v

# Proper visibility sim (any back-end)
hera-sim-vis.py obsparam.yaml simulator.yaml \
    --normalize_beams --compress redmask.npy --run-auto-check
```

### 17.4 Reference-data adjustment (`adjustment.py`)

```python
from hera_sim.adjustment import adjust_to_reference, interpolate_to_reference, rephase_to_reference

# Adjust simulated data to match an observational dataset's times/baselines
adjusted = adjust_to_reference(
    target=sim,                # Simulator or UVData or path
    reference=["zen.2458098.45.HH.uvh5", ...],
    interpolate=True, interpolation_axis="time",
    use_reference_positions=False, use_ENU_positions=False,
    position_tolerance=1.0,
    relabel_antennas=True,
    conjugation_convention=None,
    overwrite_telescope_metadata=False,
)
```

The module is a 980-line toolkit for matching simulation metadata to
real HERA observations. It uses `hera_cal.abscal.get_d2m_time_map`,
`hera_cal.io.to_HERAData`, and `hera_cal.utils.lst_rephase` for
rephasing; `match_antennas` walks all candidate translations of the
target antenna lattice to maximize overlap with the reference; and
`interpolate_to_reference` uses cubic `interp1d` /
`RectBivariateSpline` interpolation over time and/or frequency.

---

## 18. Testing layout

`tests/` mirrors the source-module structure 1:1:

| File | Coverage |
|---|---|
| `test_adjustment.py` | `adjustment.py` (requires `hera_cal`). |
| `test_antpos.py` | `LinearArray`, `HexArray`, `idealize_antpos`. |
| `test_beams.py` | `PolyBeam`, `PerturbedPolyBeam`, `ZernikeBeam`. |
| `test_cli_utils.py` | `validate_config`, `write_calfits`, internal validators. |
| `test_components.py` | The discovery framework (`get_models`, `get_model`). |
| `test_defaults.py` | Singleton, season switching, override semantics. |
| `test_eor.py` | `NoiselikeEoR`. |
| `test_foregrounds.py` | `DiffuseForeground`, `PointSourceForeground`. |
| `test_interpolators.py` | `Tsky`, `Beam`, `Bandpass`, `Reflection`. |
| `test_io.py` | `empty_uvdata`, `chunk_sim_and_save`. |
| `test_noise.py` | `ThermalNoise`. |
| `test_rfi.py` | `RfiStation`, `Stations`, `Impulse`, `Scatter`, `DTV`. |
| `test_sigchain.py` | All gain / crosstalk / mutual-coupling models. |
| `test_sim_red_data.py` | `vis.sim_red_data`. |
| `test_simulate_cli.py` | `scripts/hera-sim-simulate.py` end-to-end. |
| `test_simulator.py` | `Simulator.add`, `Simulator.get`, seeding modes, filter cache. |
| `test_utils.py` | Filter / Jy↔K / numba / `_listify` helpers. |
| `test_yaml_constructors.py` | `!Tsky`, `!Beam`, `!Bandpass`, `!antpos`, `!dimensionful`. |
| `test_visibilities/` | `MatVis`, `FFTVis`, `UVSim` integration tests with config-files under `tests/testdata/hera-sim-vis-config/`. |

`conftest.py` provides shared fixtures. `pytest` is configured (in
`pyproject.toml`) with `--cov hera_sim --cov-report xml --durations=25 -v`.

---

## 19. Documentation

`docs/index.rst` includes the README plus a TOC pointing at
`tutorials.rst`, `reference/index.rst`, `contributing.rst`,
`notes_for_developers.rst`, `authors.rst`, `changelog.rst`. Theme:
`furo`. NB extension: `nbsphinx`. Numpy-style docstrings rendered
via `numpydoc`.

`docs/reference/index.rst` autosummarizes the module list (the
order in which `__init__` re-exports them) plus the visibilities
sub-package. Note: the docs still mention
`hera_sim.visibilities.vis_cpu.VisCPU` even though the module is
empty — this is a docs-vs-code drift since the v4.0.0 removal.

`docs/tutorials/`:

- **`hera_sim_tour.ipynb`** — overview of the package and rough-sim
  components.
- **`hera_sim_simulator.ipynb`** — deep dive on the `Simulator`
  class, seeding modes, filter cache, vis_filter examples.
- **`hera_sim_defaults.ipynb`** — the Defaults singleton and how it
  interacts with component defaults.
- **`end_to_end_example.ipynb`** — full pipeline of a rough sim.
- **`mutual_coupling_example.ipynb`** — `MutualCoupling` walkthrough
  (Josaitis+ 2022 model).
- **`polybeam_simulation.ipynb`** — using `PolyBeam` /
  `PerturbedPolyBeam` / `ZernikeBeam` in a visibility simulation.
- **`visibility_simulator.ipynb`** — the `ModelData` /
  `VisibilitySimulation` framework, swapping between `MatVis`,
  `FFTVis`, `UVSim`.
- **`hera_sim_cli.rst`** — companion to `hera-sim-simulate.py`,
  walks through the `template_config.yaml` block by block.
- **`hera_sim_vis_cli.rst`** — companion to `hera-sim-vis.py`,
  shows the `obsparam` + `simulator.yaml` split.

---

## 20. Recent change history (`CHANGELOG.rst` highlights)

The dev branch (post-4.1) has these notable changes:

- **`SimulationComponent.is_randomized` was added** so the
  `Simulator` knows whether to inject a `BitGenerator` argument.
  Components using randomness now have an `rng` attribute treated
  identically to other model parameters.
- **`FFTVis` simulator was added** as a new `VisibilitySimulator`
  using the `fftvis` package (NUFFT-backed), faster than `MatVis`
  for compact arrays with many antennas.
- **All RNG usage was migrated to the new numpy API**
  (`np.random.default_rng()`); the global random state is no
  longer seeded — instead a fresh `BitGenerator` is created from
  whatever seed is desired. The `Simulator` API is unchanged but
  the internal seeding logic was rewritten.
- **Python 3.9 support dropped.**
- **Pyuvdata v2.4.0 API alignment fixes.**

Recent stable releases:

- `v4.1.0` (2023.06.26): performance push for the visibility
  simulators (especially `VisCPU`); `_blt_order_kws`,
  `compress_data_model`, `check_antenna_conjugation`,
  `hera-cli-utils` dependency, taper option in `Bandpass`,
  `interpolators.Reflection` class, Phase-1 / Phase-4 reflection
  npz files.
- `v4.0.0` (2023.05.22): **breaking change** — removed the
  `HealVis` wrapper (use `pyuvsim` instead); always uses future
  `pyuvdata` array shapes.
- `v3.1.1` (2023.02.23): mutual-coupling fix (correct conjugation
  in `J(b_jk) J(b_kj)^†`).

---

## 21. Notable behaviours, gotchas, and design choices

1. **Frequencies are in GHz almost everywhere inside `hera_sim`,
   but in Hz inside `pyuvdata`.** The `Simulator.freqs` property
   converts from `UVData.freq_array` (Hz) → GHz. The visibility-
   simulator framework uses `data_model.freqs` in Hz directly. RFI
   and DTV models specify their bands in *GHz*. This split is a
   common source of bugs when wiring custom models together.
2. **Default polarization is XX-only.** `io.empty_uvdata` defaults
   `polarization_array=["xx"]`; if you want a polarized rough
   simulation, pass `polarization_array=["xx", "yy", "xy", "yx"]`
   yourself.
3. **`hera_sim.defaults` is a singleton.** Calling
   `defaults.set("h1c")` in one corner of the program changes
   defaults for *every* future `hera_sim` call. Notebooks that
   exercise both H1C and H2C should call `defaults.deactivate()`
   between sections (or reload the module).
4. **Seeding modes are model-aware.** `seed="redundant"` is only
   meaningful for per-baseline models (it draws the seed from a
   per-redundant-group cache); it warns if you try to use it with a
   `return_type="full_array"` model like `MutualCoupling`.
5. **Auto-correlations are special-cased everywhere.**
   - `ThermalNoise` returns a pure `Trx` bias for autos.
   - `NoiselikeEoR` takes `|·|` for autos to make them
     real-positive.
   - `OverAirCrossCoupling` returns zeros for autos.
6. **The fringe filter for the conjugate baseline is the original
   filter reversed along axis 0.** This is hand-wired in
   `Simulator._get_filters` to keep visibilities invariant under
   antenna swap.
7. **`MutualCoupling` requires `pyuvdata.UVBeam` E-field beams in
   `az_za` coordinates.** Power beams and HEALPix beams are
   rejected in `_check_beam_is_ok`. If you have an `AnalyticBeam`,
   it's converted to a HEALPix UVBeam at `nside=128` internally.
8. **`UVSim` reorders blts to time-order** (necessary for `pyuvsim`)
   and **reorders pols to AIPS order** if needed. This is destructive
   (in-place) on the `UVData` object.
9. **`MatVis` requires every antpair to be present** (full
   correlation: `Nbls == n_ant * (n_ant + 1) / 2`). If your dataset
   compresses redundancy, expand it before calling `MatVis.simulate`.
10. **`vis_cpu.py` is empty in this revision** but still
    referenced by `docs/reference/index.rst`.
11. **The H1C and H2C default season YAMLs do not enable any
    `systematics:` block** — they only define instrument
    parameters. Systematics live in the user's per-simulation YAML
    (e.g. `template_config.yaml`).
12. **Beam objects are not `SimulationComponent`s.** They're plain
    `dataclass`-decorated subclasses of
    `pyuvdata.analytic_beam.AnalyticBeam` so that `pyuvsim` /
    `matvis` / `fftvis` can consume them without translation.
13. **`PerturbedPolyBeam` rescales its sidelobe perturbations on a
    fixed (1000-point, 100–200 MHz) grid in `__post_init__`** so the
    rescaling is deterministic and independent of the input grid
    passed to `interp()`. Override the auto-zero-point with
    `perturb_zeropoint`.
14. **Random state book-keeping persists.**
    `Simulator._seeds[component_name][key]` retains every drawn
    seed; on `write`, these are flattened into
    `data.extra_keywords` so the simulation is fully reproducible
    from the file alone.
15. **Per-baseline simulators receive the baseline as a 3-vector in
    nanoseconds (not metres).** `Simulator._update_args` converts
    `antpos[ant2] - antpos[ant1]` into ns by dividing by `c [m/ns]`.
16. **The Foreground and EoR models advertise `is_smooth_in_freq`.**
    `DiffuseForeground.is_smooth_in_freq = True` enables the
    pre-computed delay-filter cache; `NoiselikeEoR.is_smooth_in_freq
    = False` disables it.

---

## 22. Tracing relationships to the wider HERA software stack

Inside this RadioSim vendor tree, `hera_sim` sits at the apex of
several other simulators:

- **`matvis`** — invoked by `hera_sim.visibilities.MatVis` for the
  CPU/GPU RIME sum (see `simulators/matvis.md`).
- **`fftvis`** — invoked by `hera_sim.visibilities.FFTVis` for the
  NUFFT RIME (see `simulators/fftvis.md`).
- **`pyuvsim`** — invoked both for the `UVSim` simulator and for
  `ModelData.from_config` (the obsparam pipeline). See
  `simulators/pyuvsim.md`.
- **`pyuvdata`** — `Simulator.data` *is* a `UVData`; everywhere a
  visibility array touches the disk, it does so through `pyuvdata`.
- **`pyradiosky`** — `ModelData.sky_model` is a
  `pyradiosky.SkyModel`. See `simulators/pyradiosky/`.
- **`hera_cal`** — used optionally by `cli_utils.write_calfits`,
  `antpos.idealize_antpos`, and `adjustment.py`. Not used by the
  rough sim itself.
- **`uvtools`** — used optionally by `sigchain.Bandpass` for
  non-`tanh` window tapers (`uvtools.dspec.gen_window`).
- **`bda`** — used optionally by `scripts/hera-sim-simulate.py` for
  baseline-dependent averaging at the end of a rough sim.

When mapping `hera_sim`'s effects into RadioSim's RIME framework
(`src/radiosim/core/`):

- `hera_sim.foregrounds.DiffuseForeground` ↔ a HEALPix sky model
  multiplied by a Gaussian primary beam in RadioSim (no direct
  analogue; HERA sky-temperature models lack the spatial structure
  RadioSim expects).
- `hera_sim.foregrounds.PointSourceForeground` ↔ a synthetic point-
  source catalog in `core/sky/_factories.py`
  (`create_test_sources`), but `hera_sim` bakes in HERA-specific
  geometry (Gaussian beam width, 0.9 NS-component fraction).
- `hera_sim.eor.NoiselikeEoR` has no direct analogue; RadioSim uses
  `core/sky/_loaders_diffuse.py` with PySM3 / PyGDSM diffuse maps.
- `hera_sim.beams.PolyBeam` / `PerturbedPolyBeam` / `ZernikeBeam`
  are *not* in the RadioSim Jones-matrix system; the closest
  equivalents are `core/jones/beam/analytic.AnalyticBeamJones` and
  `core/jones/beam/fits.FITSBeamJones`. Note the architectural
  difference: `hera_sim` beams are `pyuvdata.AnalyticBeam`
  subclasses, while RadioSim beams are full Jones operators.
- `hera_sim.sigchain.MutualCoupling` is roughly comparable to a
  RadioSim baseline-error term; in RadioSim's framework, mutual
  coupling would be a `JonesBaselineTerm` (see
  `core/jones/baseline_errors.py`), not a vis-space full-array
  modifier.
- The `Bandpass` / `Reflections` / `ReflectionSpectrum` /
  `OverAirCrossCoupling` models are *vis-space approximations* of
  what RadioSim would do as Jones terms in
  `core/jones/{bandpass,gain,polarization_leakage,delay,crosshand}.py`.
  In particular, `hera_sim`'s `Bandpass` is a single multiplicative
  per-antenna gain spectrum, equivalent to RadioSim's `BandpassJones`
  ⊗ `GainJones`.

Where RadioSim wants to consume a `hera_sim` simulation, the natural
bridge is:
- read the `UVData` produced by `hera-sim-vis.py` (visibility
  simulation) or `hera-sim-simulate.py` (rough sim) via
  `radiosim.io.readers`;
- treat the bandpass/gain/reflection systematics as already-applied
  (so RadioSim simulators should run with `JonesChain` containing only
  the K (geometric) and E (beam) terms when comparing against
  `hera_sim` outputs).

---

## 23. Quick-reference: every public name

Top-level package re-exports (`hera_sim.__init__`):

```
hera_sim.DATA_PATH
hera_sim.CONFIG_PATH
hera_sim.__version__
hera_sim.adjustment
hera_sim.antpos
hera_sim.beams
hera_sim.cli_utils
hera_sim.eor
hera_sim.foregrounds
hera_sim.interpolators
hera_sim.io
hera_sim.noise
hera_sim.rfi
hera_sim.sigchain
hera_sim.simulate
hera_sim.utils
hera_sim.SimulationComponent
hera_sim.component                    # decorator
hera_sim.get_all_components
hera_sim.get_model
hera_sim.get_models
hera_sim.defaults                     # the Defaults singleton
hera_sim.Bandpass                     # interpolators.Bandpass
hera_sim.Beam                         # interpolators.Beam
hera_sim.Tsky                         # interpolators.Tsky
hera_sim.Simulator                    # simulate.Simulator
hera_sim.load_simulator_from_yaml     # visibilities.load_simulator_from_yaml
hera_sim.simulators                   # the visibilities sub-module
```

Every concrete `SimulationComponent` is registered under its
`__name__.lower()` and any `_alias` strings it declares. The full
mapping is exhaustively listed in §5.3 above.

`hera_sim.visibilities`:

```
SIMULATORS = {"UVSim": UVSim, "MatVis": MatVis, "FFTVis": FFTVis}
ModelData
VisibilitySimulation
VisibilitySimulator
load_simulator_from_yaml
UVSim
MatVis (if matvis installed)
FFTVis (if fftvis installed)
```

---

## 24. Reading order recommendations

If you have a few minutes:

- `README.rst` for the elevator pitch.
- `docs/tutorials/hera_sim_tour.ipynb` for the rough-sim tour.
- `docs/tutorials/visibility_simulator.ipynb` for the visibility-
  simulator framework tour.

If you have an hour:

- `src/hera_sim/components.py` (the plug-in framework).
- `src/hera_sim/simulate.py` (`Simulator.add`, `_iteratively_apply`,
  `_seed_rng`).
- `src/hera_sim/visibilities/simulators.py` (`ModelData`,
  `VisibilitySimulation`, `VisibilitySimulator`).

If you need to extend or debug:

- `src/hera_sim/sigchain.py` (the largest, most-frequently-edited
  file).
- `src/hera_sim/utils.py` (the filter / Jy↔K / numba kernels).
- `tests/test_simulator.py` (the most exhaustive integration test —
  the seeding modes, the filter cache, the vis_filter vocabulary
  are all exercised here).

---

*Document compiled by direct read of the vendored
`simulators/hera_sim/` tree (Python 3.11+, `hera_sim` 4.1.x dev
branch) on RadioSim main as of 2026-05-04. Cross-checked against
`README.rst`, `CHANGELOG.rst`, `pyproject.toml`, and the entire
`docs/tutorials/` tree.*
