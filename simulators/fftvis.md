# fftvis — Exhaustive Reference

A deep, top-to-bottom reference for the `fftvis` simulator vendored at
`simulators/fftvis/`. Written from a direct read of every file in
`src/fftvis/` (`__init__.py`, `wrapper.py`, `utils.py`, `logutils.py`,
`cli.py`, `core/{simulate,beams,utils,antenna_gridding}.py`,
`cpu/{cpu_simulate,beams,nufft,utils}.py`,
`gpu/{gpu_simulate,beams,nufft,utils}.py`), the test suite under `tests/`,
the two tutorial notebooks in `docs/tutorials/`, the CI configuration
(`.github/workflows/ci.yml`, `ci/fftvis_tests.yml`, `.coveragerc`),
`pyproject.toml`, and `README.md`. The vendored copy is at upstream tip
`a6b0b9e` (the only commit visible in this shallow clone) on the
`tyler-a-cox/fftvis` GitHub repo.

`fftvis` describes itself as **"A Non-Uniform Fast Fourier Transform-based
Visibility Simulator"**: a near drop-in replacement for `matvis` that
swaps the dense per-source-per-baseline phase sum for a Flatiron-Institute
NUFFT (`finufft`), giving roughly an order-of-magnitude speedup on large
HERA-like arrays while still producing visibilities that agree with
`matvis` to high precision. Authors: Tyler Cox
(`tyler.a.cox@berkeley.edu`) and Steven Murray (`steven.murray@sns.it`).
License MIT, Python ≥ 3.9, status `Development Status :: 3 - Alpha`. The
package's *raison d'être* is summarized in the README:

> Utilizes the Flatiron Institute NUFFT (finufft) algorithm for fast
> visibility simulations that agree with similar methods (`matvis`) to
> high precision.

It is heavily co-designed with `matvis`: `fftvis` reuses `matvis`'s
coordinate-rotation classes, beam-interpolator base class, and CLI
profiling helpers, and most tests in `tests/test_cpu_simulate.py` cross-
check `fftvis` outputs against `matvis.simulate_vis(...)`. Inside RadioSim
this is the engine to study when you want NUFFT-grade scaling for
zenith-pointing, mostly-coplanar arrays like HERA.

---

## 1. What fftvis is, in one paragraph

For a sky modelled as a discrete set of point sources / pixels with
intensities `I_n(ν)` (or full Stokes `[I, Q, U, V]_n(ν)`) at known
ICRS coordinates, and a single primary-beam pattern shared by every
antenna in the array (per-antenna beams are *not* supported), `fftvis`
evaluates the RIME

```
V^{pq}_{ij}(t,ν) = Σ_n A^p(X_n(t),ν) · C_n(ν) · A^q*(X_n(t),ν) · exp(-2πi · b_{ij}·X_n(t)/λ)
```

by recognizing that the sum is a non-uniform Fourier transform between
the **non-uniform "source" grid** (apparent-flux-weighted source
positions on the local-tangent-plane) and the **non-uniform "baseline"
grid** (`u, v, w = b/λ`). Instead of materializing the
`O(N_src · N_bl)` phase matrix, `fftvis` calls `finufft.nufft2d3` /
`finufft.nufft3d3` (Type-3 NUFFT) at each `(time, freq)` slice. When
the antenna positions are themselves a 2-D rational lattice (HERA-style
hex, redundant grids), `fftvis` infers the lattice basis, scales
baselines to integer modes, and falls back to the much cheaper
`finufft.nufft2d1` (Type-1 NUFFT) producing a regular grid of Fourier
modes which is then index-selected to recover the requested baselines.
Time integrations are parallelized with Ray; per-process linear-algebra
threading is bounded with `threadpoolctl`. The output is a numpy array
of complex visibilities matching the layout used by `matvis.simulate_vis`.

Limitations stated up-front in the README:

1. **No support for per-antenna beams** — every antenna shares a single
   `pyuvdata.UVBeam` / `pyuvdata.AnalyticBeam` (wrapped via
   `pyuvdata.beam_interface.BeamInterface`).
2. **GPU backend is a stub** — every method in
   `src/fftvis/gpu/` raises `NotImplementedError` (the tests assert this).
3. **Diffuse skies must be pixelized** — there is no shapelet, Gaussian-
   component, or HEALPix-resampling layer; the engine consumes a flat
   `(nsources, nfreqs[, 4])` flux array.

---

## 2. Versioning, packaging, and dependencies

`pyproject.toml` (verbatim summary):

| Field | Value |
|---|---|
| `name` | `fftvis` |
| `description` | `An FFT-based visibility simulator` |
| `requires-python` | `>=3.9` |
| Build backend | `setuptools>=64`, `setuptools_scm>8` (dynamic version) |
| Authors | Tyler Cox, Steven Murray |
| License | MIT |

**Runtime dependencies** (`pyproject.toml [project.dependencies]`):

- `numpy` — ndarrays everywhere.
- `matvis>=1.3.2` — `BeamInterpolator`, `CoordinateRotation*`,
  `coordinates.enu_to_az_za`, `prepare_beam_unpolarized`,
  `cli.get_standard_sim_params`, `cli.get_label`. **`fftvis` cannot
  function without `matvis`** — every coordinate transform and beam-
  evaluator base class lives in `matvis.core`.
- `finufft` — Python bindings for the Flatiron-Institute NUFFT (the
  paper cited in the README is *Barnett, Magland & af Klinteberg 2019*,
  arXiv:1808.06736). All three NUFFT primitives (`nufft2d3`, `nufft3d3`,
  `nufft2d1`) are called directly from `cpu/nufft.py`.
- `pyuvdata>=3.1.2` — `UVBeam`, `BeamInterface` for primary beams.
- `numba` — `@nb.jit(nopython=True)` accelerates the in-place rotation
  (`cpu/utils.py:inplace_rot`) and the apparent-flux Mueller-style
  contractions (`cpu/beams.py:get_apparent_flux_polarized*`).
- `ray` — distributed/parallel execution of per-(freq,time)-chunk
  evaluations via `@ray.remote`.
- `threadpoolctl` — bounds BLAS thread counts inside each Ray worker.
- `psutil` — RSS / shared-memory introspection for the progress logger.
- `memray` — optional heap tracing when `trace_mem=True`.
- `typer` + `rich` + `line_profiler` — used by the `fftvis` CLI
  (`cli.py`) for the `run-profile` command.

**Dev/optional dependencies** (`[project.optional-dependencies] dev`):
`mpi4py`, `pytest`, `pytest-cov`, `pytest-xdist`, `pre-commit`,
`pyradiosky`, `pyuvsim[sim]`, `hera_sim`. These are needed to run the
cross-check tests in `tests/test_cpu_simulate.py::test_sim_polarized_sky`
which calls into `pyuvsim.uvsim.run_uvdata_uvsim` and reads sources
from a `pyradiosky.SkyModel`.

**CLI scripts** (`[project.scripts]`):

```
fftvis = fftvis.cli:app
```

i.e. the installed `fftvis` shell command is the `typer` app defined in
`src/fftvis/cli.py`, exposing one subcommand `run-profile` (see §13).

`pyproject.toml` does **not** declare a `tool.setuptools.packages` —
just `package-dir = {"" = "src"}` — so packaging picks up everything
under `src/fftvis/` automatically.

`.coveragerc` excludes `logutils.py`, `cli.py`, and lines guarded by
`# pragma: no cover`, `if trace_mem:`, `if isinstance(beam, UVBeam):`,
`if _nbig != nprocesses:`, and `if eps is None:`. These are the
explicitly-untested code paths.

---

## 3. Repository layout

```
fftvis/
├── pyproject.toml                # build + deps + CLI script + dynamic version
├── README.md                     # short user-facing description
├── LICENSE                       # MIT
├── codecov.yml                   # (empty)
├── .coveragerc                   # what coverage ignores
├── .github/workflows/ci.yml      # GitHub Actions: pytest on Linux+macOS, py3.10–3.12
├── ci/fftvis_tests.yml           # conda env spec used by CI
├── docs/
│   └── tutorials/
│       ├── fftvis_tutorial.ipynb         # main "how to use simulate_vis" notebook
│       └── fftvis_gridded_array.ipynb    # Type-1 vs Type-3 demo on hex/square arrays
├── src/fftvis/
│   ├── __init__.py               # re-exports public API
│   ├── wrapper.py                # `simulate_vis` (the user-facing function) + factories
│   ├── cli.py                    # `fftvis run-profile` typer command
│   ├── utils.py                  # re-exports core utilities + _use_gpu() probe
│   ├── logutils.py               # progress / RSS / tracemalloc helpers (excluded from cov)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── simulate.py           # `SimulationEngine` ABC + default_accuracy_dict
│   │   ├── beams.py              # `BeamEvaluator` ABC (subclass of matvis BeamInterpolator)
│   │   ├── utils.py              # speed_of_light, get_pos_reds, plane→XY rotation, chunking
│   │   └── antenna_gridding.py   # lattice detection → integer baseline indices
│   ├── cpu/
│   │   ├── __init__.py
│   │   ├── cpu_simulate.py       # `CPUSimulationEngine` (the only working backend)
│   │   ├── beams.py              # `CPUBeamEvaluator` (uses pyuvdata + numba kernels)
│   │   ├── nufft.py              # `cpu_nufft2d`, `cpu_nufft3d`, `cpu_nufft2d_type1` (finufft wrappers)
│   │   └── utils.py              # `inplace_rot` (numba) + `prepare_source_catalog`
│   └── gpu/
│       ├── __init__.py
│       ├── gpu_simulate.py       # `GPUSimulationEngine` — NotImplementedError stubs
│       ├── beams.py              # `GPUBeamEvaluator` — NotImplementedError stubs
│       ├── nufft.py              # `gpu_nufft2d`, `gpu_nufft3d` — NotImplementedError stubs
│       └── utils.py              # `inplace_rot` GPU stub
└── tests/
    ├── data/HERA_NicCST_150MHz.txt        # one CST beam pattern, used by every beam test
    ├── test_cpu_simulate.py               # ~1100 LOC: matvis + pyuvsim cross-checks
    ├── test_cpu_beams.py                  # ~700 LOC: spline opts, Mueller kernels, edge cases
    ├── test_beam_evaluator.py             # ~300 LOC: BeamEvaluator base behaviour
    ├── test_wrapper.py                    # ~300 LOC: high-level `simulate_vis` shape/dtype tests
    ├── test_core_utils.py                 # ~220 LOC: get_pos_reds, get_task_chunks, rotations
    ├── test_antenna_gridding.py           # ~80  LOC: griddability detection on hex/square/scattered
    ├── test_gpu_nufft.py                  # asserts every GPU NUFFT raises NotImplementedError
    └── test_gpu_beams.py                  # same for GPU beam evaluator + inplace_rot
```

Total source tree: **~2,000** non-test Python LOC, **~2,800** test LOC,
i.e. tests slightly outweigh production code. The `core ↔ cpu/gpu` split
mirrors `matvis`'s own `matvis/cpu` vs `matvis/gpu` layering.

---

## 4. Architecture

There are three concentric layers, mirrored in the package layout:

```
+-------------------------------------------------------+
|  USER-FACING   wrapper.simulate_vis(...)              |  ← the function 99% of callers use
|                wrapper.create_simulation_engine()     |
|                wrapper.create_beam_evaluator()        |
+-------------------------------------------------------+
|  CORE / ABC    core.simulate.SimulationEngine         |  ← shape, signature, parameter docs
|                core.beams.BeamEvaluator               |  ← inherits matvis.BeamInterpolator
|                core.antenna_gridding (griddability)   |
|                core.utils    (rotation, chunking)     |
+-------------------------------------------------------+
|  CPU BACKEND   cpu.cpu_simulate.CPUSimulationEngine   |  ← the only working backend
|                cpu.beams.CPUBeamEvaluator             |  ← `_cpu_beam_evaluator` global
|                cpu.nufft  (finufft wrappers)          |
|                cpu.utils  (numba kernels, coherency)  |
|  GPU BACKEND   gpu.*       (every method = NotImpl.)  |  ← scaffolding only
+-------------------------------------------------------+
```

The wrapper layer's purpose is to (a) build a `BeamInterface` from
whatever the user passed (raw `UVBeam`, `AnalyticBeam`, or already-
wrapped `BeamInterface`), (b) interpolate that beam onto the requested
frequencies once up-front, (c) call `matvis.core.beams.prepare_beam_
unpolarized` for non-polarized runs, (d) instantiate the right
`SimulationEngine` subclass via the `backend` arg, and (e) delegate.

The ABCs in `core/` exist primarily to hold the rich docstrings and the
exact parameter list — `CPUSimulationEngine.simulate` signature and
shape contract is identical to the abstract `SimulationEngine.simulate`.

---

## 5. Public API surface

`__init__.py` exports exactly:

```python
__all__ = [
    "BeamEvaluator",          # core.beams.BeamEvaluator
    "CPUBeamEvaluator",       # cpu.beams.CPUBeamEvaluator
    "create_beam_evaluator",  # wrapper.create_beam_evaluator
    "SimulationEngine",       # core.simulate.SimulationEngine
    "CPUSimulationEngine",    # cpu.cpu_simulate.CPUSimulationEngine
    "create_simulation_engine",
    "simulate_vis",           # the high-level user function
]
```

Plus modules `fftvis.utils` and `fftvis.logutils` are imported but not
listed in `__all__`.

### 5.1 `simulate_vis`

The single entry point most users touch. Defined in `wrapper.py` as:

```python
def simulate_vis(
    ants: dict,
    fluxes: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray | astropy.time.Time,
    beam,
    telescope_loc: astropy.coordinates.EarthLocation,
    baselines: list[tuple] | None = None,
    precision: int = 2,
    polarized: bool = False,
    eps: float | None = None,
    upsample_factor: Literal[1.25, 2] = 2,
    beam_spline_opts: dict | None = None,
    use_feed: str = "x",
    flat_array_tol: float = 1e-6,
    interpolation_function: str = "az_za_map_coordinates",
    nprocesses: int | None = 1,
    nthreads: int | None = None,
    coord_method: Literal["CoordinateRotationAstropy", "CoordinateRotationERFA"] = "CoordinateRotationERFA",
    coord_method_params: dict | None = None,
    force_use_type3: bool = False,
    force_use_ray: bool = False,
    trace_mem: bool = False,
    backend: Literal["cpu", "gpu"] = "cpu",
) -> np.ndarray
```

**Inputs.**

- `ants` — `dict[int, np.ndarray]` with each value an ENU
  `(x, y, z)` position in metres. The wrapper coerces the values to
  `np.array` immediately; the engine then either rotates them onto the
  XY-plane (Type-3 path) or scales them to integer lattice coords
  (Type-1 path).
- `fluxes` — either:
  - `(nsources, nfreqs)` real-valued Stokes-I (always allowed), or
  - `(nsources, nfreqs, 4)` full-Stokes `[I, Q, U, V]` (allowed *only*
    when `polarized=True`).

  In both cases the "input flux" is split equally between the two
  linear feeds via a global `0.5` factor (`cpu/utils.py`:
  `coherency = 0.5 * sky_model`), so `V_XX + V_YY = I` after summation
  — the same convention as `matvis` and the RadioSim core (see
  `RadioSim/CLAUDE.md` § "The RIME Equation").
- `ra`, `dec` — ICRS, radians; arrays of length `nsources`. The
  docstring says `ra ∈ [0, 2π], dec ∈ [-π, +π]` but the upstream
  `core.simulate.SimulationEngine` docstring (and basic geometry) imply
  `dec ∈ [-π/2, π/2]` — the user-facing docstring is loose here.
- `freqs` — frequencies in Hz, real, length `nfreqs`.
- `times` — either a numpy array of Julian dates (no astropy unit) or
  an `astropy.time.Time` object. If an ndarray is passed, the engine
  promotes it to `Time(times, format="jd")` early in `simulate()`.
- `beam` — a single shared beam: `UVBeam`, `AnalyticBeam`, or an
  already-wrapped `pyuvdata.beam_interface.BeamInterface`. If the beam
  is a `UVBeam` with multiple frequencies, it is interpolated onto the
  user's `freqs` via `beam.interp(freq_array=freqs, new_object=True,
  run_check=False)` *once* before entering the simulation loop — this
  is the key reason the CLI/CI patches `pyuvdata.telescopes.get_telescope`
  in tests, and is excluded from coverage.
- `telescope_loc` — an `astropy.coordinates.EarthLocation` representing
  the array centre. Used by the coordinate-rotation manager to compute
  topocentric vectors at each integration.
- `baselines` — optional list of `(ant_i, ant_j)` tuples (matching keys
  in `ants`). When `None`, the engine derives one representative
  baseline per redundant group by calling
  `core.utils.get_pos_reds(ants, include_autos=True)` and taking the
  first member of each group, so the default output dimension `nbls`
  is the *number of unique redundant groups*, not `N_ant choose 2`.

**Quality / accuracy knobs.**

- `precision` ∈ `{1, 2}` selects the float dtype: `(float32, complex64)`
  or `(float64, complex128)`.
- `eps` — the NUFFT accuracy passed to `finufft`. Defaults via
  `core.simulate.default_accuracy_dict`:
  ```python
  default_accuracy_dict = {1: 6e-8, 2: 1e-13}
  ```
  The `wrapper.simulate_vis` docstring claims `1e-12` for `precision=2`
  but the actual default is `1e-13` — the wrapper docstring is
  slightly out-of-sync with the value the engine uses.
- `upsample_factor` ∈ `{1.25, 2}` — `finufft` `upsampfac`. `2` (default)
  is the standard Cooley-Tukey FFT-friendly factor; `1.25` reduces the
  intermediate grid (faster, less RAM) at some accuracy cost.
- `beam_spline_opts` — passed straight through to
  `pyuvdata.UVBeam.interp` as `spline_opts`. `az_za_simple` accepts
  e.g. `{"kx": 1, "ky": 1}` (RectBivariateSpline degrees);
  `az_za_map_coordinates` accepts `{"order": 1}`
  (`scipy.ndimage.map_coordinates` order).
- `interpolation_function` — `"az_za_simple"` (slower, more accurate
  near the horizon) vs `"az_za_map_coordinates"` (faster, less accurate
  at high spline orders). Default is `"az_za_map_coordinates"`.
- `flat_array_tol` — metres; if every antenna's `z` is within this
  tolerance the array is considered flat and the engine attempts the
  griddability check + Type-1 path. Default `1e-6 m`.
- `force_use_type3` — bypass griddability detection entirely and use
  Type-3 NUFFT.
- `use_feed` — for unpolarized runs only, picks which feed (`"x"` or
  `"y"`) is used by `prepare_beam_unpolarized`.

**Parallelism.**

- `nprocesses` — number of Ray actors (each handling a subset of
  (time, freq) chunks). `None` ⇒ `multiprocessing.cpu_count()`. `1`
  disables Ray *unless* `force_use_ray=True`.
- `nthreads` — total threads to spread across the processes; defaults
  to `cpu_count()`. Threads-per-process ≈ `nthreads // nprocesses` with
  any remainder distributed to the lower-indexed workers.
- `force_use_ray` — even with `nprocesses=1`, run through the Ray path
  (useful for testing / shared-memory introspection).

**Coordinate rotation.**

- `coord_method` — `"CoordinateRotationERFA"` (default; uses ERFA via
  `matvis.core.coords`) or `"CoordinateRotationAstropy"` (uses astropy;
  needed when matching `pyuvsim`).
- `coord_method_params` — keyword args forwarded to the chosen
  rotation class. Notable keys:
  - `update_bcrs_every` (ERFA only) — seconds between BCRS recomputations,
    larger ⇒ faster, less accurate.
  - `source_buffer` — scalar in `(0, 1]` sizing the
    above-horizon source buffer pre-allocated by `matvis`. Tests pass
    `0.5` or `0.75`.

**Diagnostics.**

- `trace_mem` — when `True` and `nprocesses>1`/`force_use_ray=True`,
  asks Ray to record reference-creation sites
  (`RAY_record_ref_creation_sites=1`) and dumps `ray memory --units MB`
  before/after data is `ray.put()`-ed. Independently, each chunk worker
  starts a `memray.Tracker` named `memray-{ts}_{pid}.bin`. Both paths
  are excluded from coverage.

**Output shape.**

| Mode | Shape |
|---|---|
| `polarized=False` | `(nfreqs, ntimes, nbls)` |
| `polarized=True`  | `(nfreqs, ntimes, 2, 2, nbls)` |

`nbls = len(baselines)` if user supplied one, else the number of unique
redundant groups including autos. Output dtype follows the precision
selection: `complex64` (precision=1) or `complex128` (precision=2).

The wrapper itself only does the beam pre-processing and engine
construction — all of the actual work is delegated to
`engine.simulate(...)`, which is where the substantial logic lives.

### 5.2 `create_simulation_engine` and `create_beam_evaluator`

Two thin factories used both internally and by tests:

```python
create_simulation_engine(backend="cpu", **kwargs) -> SimulationEngine
create_beam_evaluator(backend="cpu", **kwargs) -> BeamEvaluator
```

`backend="cpu"` returns the concrete CPU class. `backend="gpu"`:
`create_simulation_engine` does a lazy import of `GPUSimulationEngine`
(which then raises on `simulate(...)` call); `create_beam_evaluator`
short-circuits and immediately raises `NotImplementedError`. Any other
string raises `ValueError`. The CPU beam evaluator factory also
explicitly initializes `evaluator.beam_list = []` and
`evaluator.beam_idx = None` because the `matvis.BeamInterpolator` parent
expects those attributes to be present.

---

## 6. Core abstractions

### 6.1 `core/simulate.py` — `SimulationEngine`

An abstract base class with two abstract methods:

```python
@abstractmethod
def simulate(self, ants, freqs, fluxes, beam, ra, dec, times,
             telescope_loc, baselines=None, precision=2, polarized=False,
             eps=None, upsample_factor=2, beam_spline_opts=None,
             flat_array_tol=0.0, interpolation_function="az_za_map_coordinates",
             nprocesses=1, nthreads=None,
             coord_method="CoordinateRotationERFA",
             coord_method_params=None, force_use_ray=False,
             trace_mem=False, enable_memory_monitor=False) -> np.ndarray
```

```python
@abstractmethod
def _evaluate_vis_chunk(self, time_idx, freq_idx, beam, coord_mgr,
                        rotation_matrix, bls, freqs, complex_dtype, nfeeds,
                        polarized=False, eps=None, upsample_factor=2,
                        beam_spline_opts=None,
                        interpolation_function="az_za_map_coordinates",
                        n_threads=1, is_coplanar=False,
                        basis_matrix=None, type1_n_modes=None,
                        trace_mem=False) -> np.ndarray
```

The ABC is the canonical place for the long parameter docstrings — both
`CPUSimulationEngine` and `GPUSimulationEngine` defer to these for
documentation. It also defines

```python
default_accuracy_dict = {1: 6e-8, 2: 1e-13}
```

which is the source-of-truth for default `eps` (called from both the
wrapper and the engine).

Note an interesting detail: the abstract `simulate` does **not** expose
`force_use_type3`, but `CPUSimulationEngine.simulate` does — so the
griddability-vs-Type-3 toggle is a CPU-specific extension to the
abstract contract.

### 6.2 `core/beams.py` — `BeamEvaluator`

An ABC that **subclasses** `matvis.core.beams.BeamInterpolator` (this
is verified by `test_beam_evaluator_matvis_inheritance`). It serves
two purposes:

1. **Adapter** — exposes `interp(tx, ty, out)` to satisfy the
   `BeamInterpolator` interface that the rest of `matvis`-derived
   coordinate code expects.
2. **fftvis-specific contract** — declares two abstract methods
   `evaluate_beam(...)` and `get_apparent_flux_polarized(...)` whose
   semantics are richer than the bare `interp` (they take `az, za,
   freq, polarized, spline_opts, interpolation_function` and return an
   apparent-flux Mueller-style 2×2-per-source array).

`__init__` initializes the matvis attributes to placeholder values
(`beam_list=[]`, `beam_idx=None`, `polarized=False`, `nant=0`,
`freq=0.0`, `nsrc=0`, `spline_opts={}`, `precision=2`) and calls the
matvis super constructor with those minimal placeholders. The "real"
state is set later from inside `_evaluate_vis_chunk`.

The bridge implementation of `interp` is:

```python
def interp(self, tx, ty, out):
    az, za = matvis.coordinates.enu_to_az_za(
        enu_e=tx, enu_n=ty, orientation="uvbeam"
    )
    self.nsrc = len(az)
    for i, bm in enumerate(self.beam_list):
        beam_values = self.evaluate_beam(
            bm, az, za, self.polarized, self.freq,
            spline_opts=self.spline_opts,
            interpolation_function="az_za_map_coordinates",
        )
        if self.polarized:
            if beam_values.ndim == 3:
                out[i] = beam_values.transpose((1, 0, 2))
            else:
                out[i] = beam_values
        else:
            out[i] = beam_values
    return out
```

i.e. `interp(tx, ty, out)` is just a wrapper over `evaluate_beam`
that does ENU → (az, za) on the way in and a `(nax, nfeed, nsrc) →
(nfeed, nax, nsrc)` transpose on the way out for polarized beams. This
function exists *only* so that the `matvis.core.coords` machinery
(which calls `beam_interp.interp(tx, ty, out)`) can be plugged in
unchanged.

### 6.3 `core/utils.py`

Five exported pieces:

| Symbol | Purpose |
|---|---|
| `IDEALIZED_BL_TOL = 1e-8` | Baseline-comparison tolerance from `redcal.get_reds`. Surfaced for callers but not actually used internally. |
| `speed_of_light = 299792458.0` | m/s — used to convert metres → light-seconds when forming `b/λ`. |
| `get_pos_reds(antpos, decimals=3, include_autos=True)` | Compute redundant baseline groups directly from antenna positions (modified version of the `redcal` function). Used to choose default baselines and as a default `nbls` predictor in tests. |
| `get_plane_to_xy_rotation_matrix(antvecs)` | Least-squares fits a plane `z = α x + β y + γ` to all antennas, then computes the Rodrigues rotation that maps that plane to the XY plane. Returns identity when the plane is already flat (`slope_x ≈ slope_y ≈ 0`). |
| `get_task_chunks(nprocesses, nfreqs, ntimes)` | Combinatorial planner: pick `(freq_chunks, time_chunks, nf, nt)` minimizing chunk size while keeping `nprocesses` buckets. If `ntimes*nfreqs < 2*nprocesses` it returns `1, [slice(None)], [slice(None)], nfreqs, ntimes` to suppress parallelism. Otherwise it grows `nfc` (the number of frequency chunks) until `nfc*nprocesses*chunk_size > total_tasks`, then rebuilds slice lists. |
| `inplace_rot_base(rot, b)` | Reference (un-jit) implementation of the in-place 3×n rotation kernel. The CPU subclass (`cpu/utils.py`) overrides this with a numba-jitted version. |

`get_pos_reds` rounds the integer 2D vector lattice to `decimals=3`
metres and groups baselines by `(u, v)` ignoring `w`. There is one
notable subtlety: it canonicalizes each group so the lowest-index
antenna comes first. This is what makes the auto group `(ai, ai)`
appear and what makes the `(j, i)` half-baselines surface as
`[(bl[1], bl[0]) for bl in red]` when `bly < 0`.

### 6.4 `core/antenna_gridding.py`

The smartest piece of fftvis. It detects when an array is a 2-D
rational lattice and, if so, returns integer baseline indices that can
be used directly as Type-1 NUFFT mode numbers.

Three functions:

```python
find_integer_multiplier(arr, max_denominator=10**6) -> int
can_scale_to_int(arr, tol=1e-9, max_denominator=10**6, max_factor=None) -> (bool, int)
find_lattice_basis(antpos, tol=1e-9) -> np.ndarray | None
check_antpos_griddability(antpos, tol=1e-9, max_denominator=10**6, max_factor=1000) -> (bool, dict, np.ndarray)
```

The high-level recipe (`check_antpos_griddability`):

1. Stack the 3-D antenna positions into `(N_ant, 3)`.
2. Project onto the XY plane (drop `z`) and call `find_lattice_basis`,
   which:
   - Forms all pairwise differences (`N_ant²` 2-vectors);
   - Drops zeros (`norm > tol`);
   - Picks the **shortest** non-zero vector as `basis_vec_1`;
   - Walks the remaining vectors in ascending-norm order, picking the
     first one whose 2-D cross-product with `basis_vec_1` exceeds
     `tol` (i.e. the shortest non-collinear vector) as `basis_vec_2`.
   - Returns `np.column_stack([basis_vec_1, basis_vec_2])` as the
     basis (or `None` if the array has only autos, or a fall-back
     `[[bv1], [0,1]]` if every baseline is collinear).
3. Lift the 2-D basis to a 3×3 with `basis[2,2] = 1.0`.
4. Solve `basis · X = (antvecs - antvecs[0])^T` for `X`, giving the
   antenna positions in *basis coordinates* (where the lattice is
   axis-aligned).
5. `can_scale_to_int(...)` finds the smallest LCM-of-denominators `f`
   such that `f * X` is integer up to `tol`. If `f ≤ max_factor` and
   the rounded values match within `tol`, the array is griddable.
6. If griddable, return:
   - `is_griddable=True`,
   - `modified_antpos = {ant: round(f * X[i]).astype(int)}`,
   - `transform_matrix = basis / f`.
   The `f` factor folds into `transform_matrix` so that
   `transform_matrix @ modified_antpos` reconstructs the original
   antenna positions in metres.
7. Otherwise fall back to identity transform and the unchanged input.

`test_antenna_gridding.py` confirms the heuristic works on linear,
square, square-with-holes, and hex grids, and correctly rejects
randomly scattered arrays and "autos-only" inputs. Notice that an
*array of only autos* is intentionally not griddable (no baselines to
infer a basis from).

This griddability path is purely an optimization — when it triggers,
the slow `nufft2d3` (Type-3) call is replaced by a single `nufft2d1`
(Type-1) call producing a `(n_modes, n_modes)` Fourier grid which is
then index-selected with `model[..., index[0], index[1]]` to obtain
visibilities for only the requested baselines.

---

## 7. The CPU simulation engine

The whole point of `fftvis`. ~560 LOC in `cpu/cpu_simulate.py`.

### 7.1 Top-level data flow inside `simulate(...)`

```
simulate_vis (wrapper)
    ↓ beam.interp(freq_array=freqs)         # interpolate UVBeam onto user freqs
    ↓ BeamInterface(beam)
    ↓ prepare_beam_unpolarized(...)         # only if polarized=False
    ↓ create_simulation_engine("cpu")
    ↓
CPUSimulationEngine.simulate(...)
    ├─ default eps from precision; cast ra/dec/freqs to real_dtype
    ├─ if baselines is None: baselines = [r[0] for r in get_pos_reds(ants)]
    ├─ prepare_source_catalog(fluxes, polarized) -> coherency, polarized_sky_model
    │
    ├─ flat_array_tol check OR force_use_type3:
    │     ├─ TYPE-3 path (general):
    │     │     rotation_matrix = get_plane_to_xy_rotation_matrix(antvecs)
    │     │     rotated_antvecs = rotation_matrix.T @ antvecs.T
    │     │     bls = (rotated_ants[j] - rotated_ants[i]) / c
    │     │     is_coplanar = (|bls.z| < flat_array_tol).all()
    │     └─ TYPE-1 path (griddable + flat):
    │           is_gridded, gridded_antpos, basis_matrix = check_antpos_griddability(ants)
    │           bls = round(gridded_antpos[j] - gridded_antpos[i]).astype(int)
    │           n_modes = 2*max(|bls|) + 1
    │           basis_matrix /= c
    │           is_coplanar = True; rotation_matrix = I
    │
    ├─ wrap times in astropy.Time(..., format="jd")
    ├─ build coord_mgr = CoordinateRotation._methods[coord_method](
    │         flux=coherency, times=times, telescope_loc=...,
    │         skycoords=SkyCoord(ra,dec,frame="icrs"),
    │         precision=..., **coord_method_params)
    ├─ if update_bcrs_every > total span: coord_mgr._set_bcrs(0)   # one-shot
    │
    ├─ get_task_chunks(nprocesses, nfreqs, ntimes) → freq_chunks, time_chunks
    ├─ if nprocesses>1 or force_use_ray: ray.init(num_cpus, object_store_memory=2*required_shm)
    │     and ray.put(bls), ray.put(freqs), ray.put(beam), ray.put(coord_mgr), ...
    ├─ split nthreads across processes; log thread budget
    │
    ├─ for each (freq_chunk, time_chunk, nthreads_per_proc):
    │     fnc(time_idx, freq_idx, beam, coord_mgr, rotation_matrix, bls, freqs, ...)
    │       fnc is either  self._evaluate_vis_chunk          (in-process)
    │                or    _evaluate_vis_chunk_remote.remote (Ray actor)
    │
    ├─ ray.get(futures) (if Ray)
    ├─ assemble vis[tc][..., fc] = future for each chunk into a (ntimes, nbls, nfeeds, nfeeds, nfreqs) array
    └─ return:
          polarized=False: np.moveaxis(vis[..., 0, 0, :], 2, 0)           shape (nfreqs, ntimes, nbls)
          polarized=True:  np.transpose(vis, (4, 0, 2, 3, 1))             shape (nfreqs, ntimes, 2, 2, nbls)
```

The shared-memory budget heuristic that controls `object_store_memory`
sums:

- `bls.nbytes`, `rotation_matrix.nbytes`, `freqs.nbytes`.
- All `np.ndarray` attributes of `coord_mgr` (which holds the
  precomputed BCRS / topocentric arrays).
- If `beam` is a raw `UVBeam`, `beam.data_array.nbytes`. (Excluded from
  coverage.)
- The output visibility cube
  `ntimes * nfreqs * nbls * nax * nfeeds * itemsize`.

The store is sized at `2 × required_shm`. It also sets

```
RAY_object_spilling_threshold=1.0    # do not spill until 100% full
RAY_memory_monitor_refresh_ms=0      # disable memory monitor unless enable_memory_monitor=True
```

with the explicit reasoning in the source (paraphrased): "Generally,
this is a bad idea for the homogenous calculations done here: if a
task goes beyond available memory, the whole simulation should OOM, to
save CPU cycles."

### 7.2 `_evaluate_vis_chunk` — the inner kernel

Per chunk, for every time `t ∈ time_idx`:

1. `coord_mgr.rotate(ti)` updates BCRS / topocentric vectors.
2. `topo, flux, nsim_sources = coord_mgr.select_chunk(0, ti)` returns
   the *above-horizon* sources (`matvis` truncates the dataset to
   `nsim_sources` and pads with sentinels). Truncate `topo[:,
   :nsim_sources]` and `flux[:nsim_sources]`. If `nsim_sources == 0`,
   skip the whole time integration.
3. Compute `(az, za) = matvis.coordinates.enu_to_az_za(topo[0], topo[1],
   orientation="uvbeam")`. **az/za are computed *before* the antenna-
   plane rotation is applied** (because the beam is anchored to the
   array's local-tangent frame, not the rotated frame).
4. If `rotation_matrix != I`, apply it in-place to `topo` via the
   numba-jitted `cpu_utils.inplace_rot`.
5. If `basis_matrix is not None` (Type-1 path), apply
   `basis_matrix.T` in-place to `topo` (this re-expresses topocentric
   source vectors in *the same* basis we used to grid the antennas, so
   the source coordinates and the integer antenna indices live in the
   same coordinate system).
6. `topo *= 2π` (folding the factor of `2π` once instead of in every
   inner loop).
7. For every freq `f ∈ freq_idx`:
   - If Type-3, `uvw = bls * freq` (broadcast multiply).
   - Update the *module-level* singleton `_cpu_beam_evaluator`'s
     `beam_list`, `nsrc`, `polarized`, `freq` attributes (this is the
     mutable state that lets `evaluate_beam` know what to do).
   - `apparent_coherency = _cpu_beam_evaluator.evaluate_beam(beam, az,
     za, polarized, freq, spline_opts=..., interpolation_function=...)`
     and cast to `complex_dtype`.
   - Apply the source flux / coherency:
     - Polarized beam **and** polarized sky:
       `apparent_coherency = np.flip(apparent_coherency, axis=0)`
       (flipping the `nax` axis — `matvis` returns
       `(nax, nfeed, nsrc)`, but the polarized Mueller kernel expects
       `(nfeed, nax, nsrc)`), then in-place
       `get_apparent_flux_polarized(apparent_coherency,
       flux[:, freqidx].T)`. The kernel computes
       `B† C B` per source, leaving the result in-place in
       `apparent_coherency`.
     - Polarized beam, unpolarized sky: in-place
       `get_apparent_flux_polarized_beam(apparent_coherency,
       flux[:, freqidx])` computes `B† I B` per source.
     - Unpolarized beam: just `apparent_coherency *=
       flux[:, freqidx]`.
   - Reshape `apparent_coherency` to `(nfeeds**2, nsim_sources)` (so
     the `(2, 2)` Mueller block becomes a flat `4`-vector along the
     "weights" axis of the NUFFT), cast to `complex_dtype` if needed.
   - Call the NUFFT:
     - **Type-1**: `cpu_nufft2d_type1(topo[0]*freq, topo[1]*freq,
       weights, n_modes=type1_n_modes, index=bls, eps, n_threads,
       upsample_factor)`.
     - **Type-3, coplanar**: `cpu_nufft2d(topo[0], topo[1], weights,
       uvw[0], uvw[1], eps, n_threads, upsample_factor)` — note that
       in this path the source coordinates `topo[0], topo[1]` are
       *not* multiplied by `freq` (the `2π` already absorbed earlier),
       and `uvw = bls*freq` already encodes the wavelength scaling.
     - **Type-3, non-coplanar**: `cpu_nufft3d(topo[0], topo[1],
       topo[2], weights, uvw[0], uvw[1], uvw[2], ...)`.
   - Reshape the NUFFT output (flat-`nfeeds²` along axis 0, `nbls`
     along axis 1) back to `(nfeeds, nfeeds, nbls)`, swap axes so it
     becomes `(nbls, nfeeds, nfeeds)`, and slot into
     `vis[time_index, ..., freqidx]`.

The whole chunk is wrapped in `with threadpool_limits(limits=n_threads,
user_api="blas")` so that BLAS-heavy operations (especially the
`pyuvdata.UVBeam.interp` spline calls) do not over-subscribe threads
when running inside a Ray actor that already has its own CPU pinning.

### 7.3 `_evaluate_vis_chunk_remote`

A free function decorated with `@ray.remote`. It instantiates a fresh
`CPUSimulationEngine()` inside the worker and forwards every kwarg to
`engine._evaluate_vis_chunk(...)`. This is *the* Ray entry point — the
class method itself is not `@ray.remote` because Ray actors-vs-tasks
semantics around class instance state would otherwise force you to make
the engine pickle-friendly. By spinning a fresh instance per task, the
per-task state is bounded and there's nothing to serialize but the
inputs.

The macOS-skipped test
`test_evaluate_vis_chunk_remote_matches_direct` proves direct-call and
remote-call results are bitwise identical (`assert_allclose` with no
explicit tolerance ⇒ default `rtol=1e-7`, `atol=0`).

### 7.4 Beam interpolation flow

`CPUBeamEvaluator.evaluate_beam` is a thin wrapper over
`pyuvdata.beam_interface.BeamInterface.compute_response(...)`:

```python
interp_beam = beam.compute_response(
    az_array=az,
    za_array=za,
    freq_array=np.atleast_1d(freq),
    reuse_spline=True,
    check_azza_domain=False,
    spline_opts=spline_opts,
    interpolation_function=interpolation_function,
)
if polarized:
    interp_beam = interp_beam[:, :, 0, :]      # drop the (single) freq axis
else:
    interp_beam = interp_beam[0, 0, 0, :]      # drop nax, nfeed, nfreq
if check:
    if np.isinf(np.sum(interp_beam)) or np.isnan(np.sum(interp_beam)):
        raise ValueError("Beam interpolation resulted in an invalid value")
return interp_beam
```

Two crucial details:

- `reuse_spline=True` and `check_azza_domain=False` are how `fftvis`
  squeezes performance out of repeated beam evaluation across
  integrations: the spline coefficients are cached by `pyuvdata` and
  the domain validation (which would normally reject sources below the
  horizon) is bypassed because `coord_mgr.select_chunk` has already
  guaranteed every passed source is above the horizon.
- The non-polarized path *requires* `prepare_beam_unpolarized` to have
  been called first (which collapses the `(nax, nfeed)` axes to
  `(1, 1)`), so that `interp_beam[0, 0, 0, :]` is the right thing.

The two Mueller-style apparent-flux kernels are numba-jitted
(`@nb.jit(nopython=True, parallel=False, nogil=False)`) so they run
cleanly inside Ray workers (Numba's GIL lock is fine with the BLAS
threadpool limits we set):

- `get_apparent_flux_polarized_beam(beam, flux)` — for unpolarized
  Stokes-I sky and polarized beams. Computes per source
  `M = B† B; out = M ⊙ I_n`. Specifically `i00 = |B[0,0]|² + |B[1,0]|²`,
  `i01 = B*[0,0]B[0,1] + B*[1,0]B[1,1]`, `i11 = |B[0,1]|² + |B[1,1]|²`,
  multiplied by the scalar `flux[isrc]` and stored in-place. Equivalent
  to `np.einsum("bas,s,bcs->acs", beam.conj(), flux, beam)` (verified
  in `test_get_apparent_flux_polarized_beam`).
- `get_apparent_flux_polarized(beam, coherency)` — for full-Stokes sky.
  Computes per source `B† C B` where `C` is the 2×2 complex coherency
  matrix and writes the result back in-place into `beam`. Equivalent
  to `np.einsum("kin,kmn,mjn->ijn", np.conj(beam), coherency, beam)`
  (verified in `test_get_apparent_flux_polarized_different_shapes`).

The reason both kernels write **back into `beam`** is to keep working-
memory bounded during long simulations — there's no auxiliary scratch
allocation per integration.

### 7.5 The `0.5` Stokes factor

`cpu/utils.py:prepare_source_catalog` is the source of the
half-coherency convention:

```python
if not polarized_sky_model:
    coherency = 0.5 * sky_model
else:
    coherency = 0.5 * np.array(
        [[I+Q,  U+iV ],
         [U-iV, I-Q  ]]
    )
    coherency = np.transpose(coherency, (2, 3, 0, 1))   # (..., 2, 2)
```

This matches the `1/2` factor enforced by RadioSim core
(`RadioSim/CLAUDE.md`: `C = (1/2) × [[I+Q, U-iV], [U+iV, I-Q]]`,
ensuring `V_XX + V_YY = I`). Note the **sign convention**: `fftvis`
puts `U+iV` in the off-diagonal `[0,1]`, which is the standard
linear-feed convention but the conjugate of the convention sometimes
used in single-dish polarimetry texts. This is consistent with `matvis`
and `pyuvsim`.

Validation on shape:

- `polarized=False` ⇒ sky_model must be 2-D `(nsources, nfreqs)`.
- `polarized=True` ⇒ sky_model can be either 2-D or 3-D
  `(nsources, nfreqs, 4)`. With 2-D + polarized=True, the same `0.5*I`
  scaling is used (i.e. it's still treated as Stokes I split equally
  between feeds, but the beam is allowed to mix them).

`ValueError` is raised in any other shape combination, with explicit
error messages distinguishing the two failure modes.

### 7.6 Type-1 vs Type-3 NUFFT in detail

A single line in `_evaluate_vis_chunk` chooses between the two paths:

```python
if use_type1:
    _vis_here = cpu_nufft2d_type1(topo[0]*freq, topo[1]*freq, weights,
                                   n_modes, index=bls, ...)
else:
    if is_coplanar:
        _vis_here = cpu_nufft2d(topo[0], topo[1], weights, uvw[0], uvw[1], ...)
    else:
        _vis_here = cpu_nufft3d(topo[0], topo[1], topo[2], weights,
                                 uvw[0], uvw[1], uvw[2], ...)
```

Three regimes in increasing generality:

1. **Type-1 (`nufft2d1`)** — applies only when antennas form a 2-D
   rational lattice. Outputs a regular grid of Fourier modes; we
   index-select with `index=bls` to recover the visibilities for the
   requested redundant groups. Computation is `O(N_modes² log N_modes
   + N_src)` and reuses the FFT plan across baselines, so this is the
   *fastest* path — the cost barely grows with the number of redundant
   baselines because they're all already in the `(n_modes, n_modes)`
   output. `n_modes = 2 * max(|bls|) + 1` (so the integer indices can
   be both positive and negative).
2. **Type-3 coplanar (`nufft2d3`)** — generic 2-D NUFFT between
   non-uniform source positions and non-uniform `(u, v)` baselines.
   Used when the array is approximately flat (`|bls.z| <
   flat_array_tol`) but not griddable. Cost roughly
   `O(N_src + N_bl + grid_log_grid)` per (time, freq).
3. **Type-3 non-coplanar (`nufft3d3`)** — full 3-D NUFFT with `(u, v,
   w)` and `(x, y, z)`. Used when the array has significant `z`
   spread. Same asymptotic cost class but a larger upsampled grid in
   memory.

`test_simulate_gridded_type1_vs_type3` exhaustively cross-checks
Type-1 against Type-3 with `force_use_type3=False/True` over hex and
square grids, with random shears, rotations, and antenna deletions,
asserting agreement at `atol=1e-5` (precision=2) or `1e-4`
(precision=1).

---

## 8. The NUFFT primitives

`cpu/nufft.py` is **75 lines** — three short wrappers around `finufft`:

```python
def cpu_nufft2d(x, y, weights, u, v, eps, n_threads=1, upsample_factor=2):
    return finufft.nufft2d3(
        x, y, weights,
        np.ascontiguousarray(u), np.ascontiguousarray(v),
        modeord=0, eps=eps, nthreads=n_threads,
        showwarn=0, upsampfac=upsample_factor,
    )

def cpu_nufft3d(x, y, z, weights, u, v, w, eps, upsample_factor=2, n_threads=1):
    return finufft.nufft3d3(
        x, y, z, weights,
        np.ascontiguousarray(u), np.ascontiguousarray(v), np.ascontiguousarray(w),
        modeord=0, eps=eps, nthreads=n_threads,
        showwarn=0, upsampfac=upsample_factor,
    )

def cpu_nufft2d_type1(x, y, weights, n_modes, index, eps,
                      upsample_factor=2, n_threads=1):
    model = finufft.nufft2d1(
        x, y, weights,
        n_modes, modeord=1, eps=eps, nthreads=n_threads,
        showwarn=0, upsampfac=upsample_factor,
    )
    return model[..., index[0], index[1]]
```

Important details on the `finufft` flags:

- `modeord=0` for Type-3 — output uses `+/- frequencies` ordering (not
  FFT order). `modeord=1` for Type-1 — uses FFT (`fftshift`-ed)
  ordering, so `index` can directly contain *signed* integer mode
  numbers.
- `eps` is the **NUFFT accuracy goal** (relative error). `finufft`
  automatically picks an internal grid size to meet it.
- `showwarn=0` suppresses finufft's stderr warnings (e.g. about close
  source pairs).
- `upsampfac=2` is the standard, `1.25` is the only other supported
  value in the `Literal[1.25, 2]` type hint.
- `weights` has shape `(nfeeds**2, nsim_sources)` — i.e. up to four
  parallel Fourier transforms (one per Mueller block) sharing the same
  source positions and baseline targets, exploiting `finufft`'s vector
  signal support to amortize plan setup.

---

## 9. The CPU rotation kernel

`cpu/utils.py:inplace_rot` is a numba-jitted rotation that overwrites
its 3×N input in place:

```python
@nb.jit(nopython=True)
def inplace_rot(rot: np.ndarray, b: np.ndarray):
    nsrc = b.shape[1]
    out = np.zeros(3, dtype=b.dtype)
    for n in range(nsrc):
        out[0] = rot[0,0]*b[0,n] + rot[0,1]*b[1,n] + rot[0,2]*b[2,n]
        out[1] = rot[1,0]*b[0,n] + rot[1,1]*b[1,n] + rot[1,2]*b[2,n]
        out[2] = rot[2,0]*b[0,n] + rot[2,1]*b[1,n] + rot[2,2]*b[2,n]
        b[:,n] = out
```

It is ~ten times faster than `np.dot(rot, b)` for the small `(3, N)`
arrays involved here (because numba avoids both the BLAS call overhead
and the temporary allocation), which matters because it is called
once per integration per chunk. The reference (un-jit) version is
`core/utils.py:inplace_rot_base` and is what the GPU stub
(`gpu/utils.py:inplace_rot`) eventually replaces.

---

## 10. Coordinate rotation flow (matvis-derived)

`fftvis` does **not** implement its own (RA, Dec) → (az, za) machinery
— it reuses `matvis.core.coords.CoordinateRotation`, an ABC with two
concrete subclasses registered in
`CoordinateRotation._methods`:

- `"CoordinateRotationERFA"` — uses `pyerfa` to step through ICRS →
  CIRS → topocentric ENU. Has `update_bcrs_every` (seconds) which
  controls how often the BCRS-frame transform is recomputed. Default
  in `fftvis`.
- `"CoordinateRotationAstropy"` — uses
  `astropy.coordinates.SkyCoord.transform_to(ITRS(...))` directly.
  Slower, but bit-for-bit reproduces the chain `pyuvsim` uses.
  The `test_sim_polarized_sky` test against `pyuvsim` switches to this
  method explicitly.

The rotation manager owns:

- `flux` — the coherency cube (passed in at construction).
- `times` — `astropy.Time`.
- `telescope_loc` — `EarthLocation`.
- `skycoords` — `SkyCoord(ra, dec, frame="icrs")`.
- A `source_buffer` keyword (passed via `coord_method_params`) that
  pre-allocates above-horizon source storage.

Two methods are called per integration:

- `coord_mgr.rotate(ti)` — recompute (az, za) for the `ti`-th time.
- `coord_mgr.select_chunk(0, ti)` — return `(topo, flux, nsim_sources)`
  arrays trimmed to the above-horizon subset (the first `0` argument
  selects beam index 0, which is the only beam fftvis supports).

At the start of each chunk worker, `coord_mgr.setup()` is invoked once
to materialize internal arrays (this is needed because `coord_mgr` was
serialized through `ray.put` and lazy fields aren't carried).

There's a small optimization in `simulate(...)`:

```python
if getattr(coord_mgr, "update_bcrs_every", 0) > (times[-1] - times[0]).to(un.s):
    coord_mgr._set_bcrs(0)
```

If the user is using ERFA and `update_bcrs_every` exceeds the entire
observation length, the BCRS transform is computed once **before**
spawning Ray workers, so each worker doesn't redundantly recompute it.

---

## 11. Parallelism strategy

Three concentric levels of parallelism:

1. **Ray actors over (freq_chunk, time_chunk)** — `nprocesses` workers,
   each receiving its own `time_idx, freq_idx` slice pair from
   `get_task_chunks`. Coalesces frequency chunks before time chunks
   (preferring fewer workers each handling more frequencies).
2. **BLAS threads inside each worker** — bounded by
   `threadpool_limits(limits=n_threads, user_api="blas")`. The thread
   budget is `nthreads // nprocesses` rounded up; remainders go to the
   first `nthreads % nprocesses` workers.
3. **Numba threads inside each kernel** — `parallel=False` on both
   apparent-flux kernels, so this level is not exploited (intentional:
   the kernels are short and parallelism overhead would dominate).

Plus `finufft`'s own `nthreads=...` argument, which by default uses
all available cores. `_evaluate_vis_chunk` passes the same `n_threads`
down so finufft and BLAS share a thread budget.

The Ray initialization parameters are:

```python
ray.init(
    num_cpus=nprocesses,
    object_store_memory=2 * required_shm,
    include_dashboard=False,
)
```

with `RAY_object_spilling_threshold=1.0` (no spill until 100% full),
`RAY_memory_monitor_refresh_ms=0` (disable memory monitor unless
`enable_memory_monitor=True`), and optionally
`RAY_record_ref_creation_sites=1` for trace memory debugging. If a
Ray cluster is already running (e.g. inside a multi-node deployment),
the `ValueError` from a re-init is caught and the existing cluster is
re-used instead.

The macOS test skips (`@pytest.mark.skipif(sys.platform == "darwin")`)
acknowledge that Ray's IPC layer is flaky on macOS — production
fftvis runs assume a Linux host.

---

## 12. Memory tracing

When `trace_mem=True`:

- Inside Ray: `RAY_record_ref_creation_sites=1` enables Ray's
  reference-creation site logging; the engine then shells out to
  `ray memory --units MB > {before,after}-puts.txt` and
  `> after-futures.txt` and `> got-all.txt` at the four key moments
  (before put, after put, after submission, after get).
- Inside each chunk worker: `memray.Tracker(f"memray-{time.time()}_{pid}.bin")`
  is started (and never explicitly stopped — relies on process exit).
  The `.bin` files can be inspected with `python -m memray flamegraph
  *.bin` afterwards.
- The progress logger (`logutils.printmem`, `logutils.memtrace`,
  `logutils.log_progress`) reports per-iteration RSS via `psutil` and
  Python's `tracemalloc` peak. These functions are excluded from
  coverage and currently are *not* called from inside the engine — they
  are leftover scaffolding from the matvis port.

These paths are deliberately heavyweight; they should never run in
production.

---

## 13. The CLI — `fftvis run-profile`

`src/fftvis/cli.py` exposes one Typer subcommand. It is *not* a
"simulate from a YAML config" tool — it is a deliberate
performance-profiling harness. Synopsis:

```
fftvis run-profile --analytic-beam --nfreq 4 --ntimes 60 --nants 350 \
    --hera 5 --nsource 1000 --nprocesses 8 --beam-spline-order 3 --backend cpu
```

It does the following:

1. Calls `matvis.cli.get_standard_sim_params(...)` to construct a
   reference simulation (analytic Gaussian beam by default; supports
   tabulated beams via `--analytic-beam=False`). When `--nside > 0`,
   `nsource = 12 * nside²` (HEALPix pixel count for that resolution).
2. Optionally overrides the antenna positions with
   `hera_sim.antpos.hex_array(hera)` for a HERA-style hex packing
   (where `hera` is the number of rings).
3. Pretty-prints the simulation parameters as a `rich.Rule` /
   `cns.print` table.
4. Wraps the call to `simulate_vis(...)` in `cProfile.runctx(..., str_id)`.
5. Sorts the profile by `cumulative` time and prints the top 50.
6. Calls `flameprof --format=log {str_id} > {str_id}.flame` to produce
   a flame graph — note this is an **unchecked shell-out**: if
   `flameprof` is not on `PATH`, the command will silently produce
   nothing (no `subprocess.check_call`).

`get_label(...)` (also from `matvis.cli`) builds a deterministic
filename from the simulation parameters (`analytic_beam`, `nfreq`,
`ntimes`, `nants`, `nsource`, `gpu`, `coord_method`, `naz`, `nza`),
which is what gets used as the cProfile output path.

The CLI module is excluded from coverage (`.coveragerc: omit =
*/cli.py`) — it's a tool, not a tested feature.

---

## 14. Status of the GPU backend

Every GPU symbol is a stub. From the README:

> GPU backend exists only as a stub implementation (coming soon!)

Concretely:

| File | Class / function | Behaviour |
|---|---|---|
| `gpu/gpu_simulate.py` | `GPUSimulationEngine.simulate` | `raise NotImplementedError("GPU simulation not yet implemented")` |
| `gpu/gpu_simulate.py` | `GPUSimulationEngine._evaluate_vis_chunk` | Same. |
| `gpu/beams.py` | `GPUBeamEvaluator.evaluate_beam` | Same (after setting `self.polarized`, `self.freq`, `self.spline_opts`). |
| `gpu/beams.py` | `GPUBeamEvaluator.get_apparent_flux_polarized` | Same. |
| `gpu/nufft.py` | `gpu_nufft2d` | Same. |
| `gpu/nufft.py` | `gpu_nufft3d` | Same. |
| `gpu/utils.py` | `inplace_rot` | Same. Includes a comment block sketching the future CuPy implementation (`cp.matmul(rot_gpu, b_gpu)`). |

`tests/test_gpu_*.py` are negative tests asserting each of the above
raises `NotImplementedError`. The `wrapper.create_simulation_engine
(backend="gpu")` *does* construct a `GPUSimulationEngine` (so the
import path is exercised), but any actual `simulate` call still
raises. The `wrapper.create_beam_evaluator(backend="gpu")` is more
strict — it raises `NotImplementedError` immediately rather than
returning a stub.

The intended path forward (visible in the comment block of
`gpu/utils.py`) is **CuPy** — the design assumes CUDA + CuPy + a
CuPy-NUFFT backend (likely
[cufinufft](https://github.com/flatironinstitute/cufinufft)).

---

## 15. Test suite — what is verified

`tests/data/HERA_NicCST_150MHz.txt` is a single-frequency CST beam
pattern derived from
`https://github.com/Nicolas-Fagnoni/Simulations`; it's the only beam
pattern in the test data and is consumed by every beam-related test.

### 15.1 `test_cpu_simulate.py` — ~1100 LOC

The integration centerpiece. Headlines:

- `test_simulate(polarized, precision, use_analytic_beam, tilt_array,
  nprocesses, backend, force_use_ray)` — parametrized over 64
  combinations. The test uses
  `matvis._test_utils.get_standard_sim_params` to build a reference
  HERA-like sim, optionally tilts the array (`ant.z = ai * 5`), runs
  `matvis.simulate_vis(...)` and `fftvis.simulate_vis(...)` with the
  same parameters, and asserts agreement at `atol=1e-5` (precision=2)
  or `1e-4` (precision=1). It also checks the output shape for both
  user-supplied baselines and the default redundant-group baselines.
- `test_simulate_gridded_type1_vs_type3(polarized, precision,
  shear_array, rotate_array, remove_antennas, ants ∈ {hex, square})` —
  the Type-1 vs Type-3 cross-check. Important: it tests both
  un-sheared and sheared layouts — sheared layouts require
  `find_lattice_basis` to *not* assume axis-aligned grids.
- `test_sim_polarized_sky(use_analytic_beam)` — the `pyuvsim` cross-
  check. Builds a 4-Stokes random sky via `pyradiosky.SkyModel`, runs
  it through both `pyuvsim.uvsim.run_uvdata_uvsim` and
  `fftvis.simulate_vis(coord_method="CoordinateRotationAstropy",
  interpolation_function="az_za_simple")`, and asserts each of `xx`,
  `xy`, `yx`, `yy` agrees (no atol/rtol — default `assert_allclose`).
- `test_evaluate_vis_chunk_remote_matches_direct` (Linux-only) — proves
  the Ray remote path produces bit-identical results to direct calls.
- `test_simulate_force_use_ray_single_proc` (Linux-only) — exercises
  the `nprocesses=1, force_use_ray=True` Ray-init path and checks the
  `"Initializing with"` log line is emitted.
- `test_chunk_eval_trace_mem` (Linux-only) — exercises the
  `memray.Tracker` path inside a chunk worker.
- A handful of small basic tests (`test_simulate_with_basic_beam`,
  `test_simulate_with_specified_baselines`, `test_beam_interpolation`,
  `test_simulation_with_empty_baselines`, `test_wrapper_simulation`,
  `test_time_array_handling`) verify shape, dtype, and edge cases.

A pre-import monkey-patch at the top of the file:

```python
import pyuvdata.telescopes
if not hasattr(pyuvdata.telescopes, 'get_telescope'):
    def get_telescope(name, **kwargs):
        return Telescope.from_known_telescopes(name, **kwargs)
    pyuvdata.telescopes.get_telescope = get_telescope
```

is needed to support the matvis test helpers under `pyuvdata >= 3.1`,
which dropped the module-level `get_telescope` shim.

### 15.2 `test_cpu_beams.py` — ~700 LOC

Beam-evaluator behaviour:

- `test_beam_interpolators(polarized)` — proves the two interpolation
  functions (`az_za_simple` with `{kx:1, ky:1}` vs
  `az_za_map_coordinates` with `{order:1}`) produce identical beams.
- `test_get_apparent_flux_polarized_beam` and
  `test_get_apparent_flux_polarized_different_shapes` — verify the
  numba kernels match `np.einsum` references.
- `test_beam_evaluator_matvis_inheritance` — confirms `BeamEvaluator`
  is an actual subclass of `matvis.core.beams.BeamInterpolator`.
- `test_polarized_beam_evaluation` — checks output shape `(2, 2,
  nsrc)` and that `check=True` doesn't change values for valid beams.
- `test_evaluate_beam_invalid_values` — patches `compute_response` to
  return NaN/Inf and asserts the `ValueError("Beam interpolation
  resulted in an invalid value")` is raised under `check=True`.
- `test_get_apparent_flux_polarized_edge_cases` — empty source list,
  single source.
- `test_evaluate_beam_with_different_spline_opts`,
  `test_evaluate_beam_additional_paths` — covers the
  `spline_opts=None` branch and various fall-throughs.
- `test_wrapper_beam_creation` — `create_beam_evaluator("gpu")` raises
  `NotImplementedError`; `create_beam_evaluator("invalid")` raises
  `ValueError`.

### 15.3 `test_beam_evaluator.py` — ~300 LOC

Tests of the `BeamEvaluator`/`CPUBeamEvaluator` initialization and the
`interp` bridge method:

- `test_cpu_beam_evaluator_init`, `test_cpu_evaluator_attributes`,
  `test_cpu_evaluator_constructor` — default attribute values.
- `test_evaluate_beam_with_check` — exercises the `check=True` branch
  with a real beam and proves no false positives.
- `test_interp_method`, `test_beam_evaluator_interp_branches`,
  `test_beam_evaluator_interp_polarized_branches` — cover all three
  branches of the `interp` method (non-polarized, polarized with
  `ndim==3`, polarized with `ndim!=3`).
- `test_get_apparent_flux_unpolarized` — sanity check for the
  unpolarized "just multiply" path.

### 15.4 `test_wrapper.py` — ~300 LOC

Top-level shape/dtype tests, calling `simulate_vis` with all knobs at
default:

- `test_simulate_vis_basic` — minimal HERA-like 3-antenna run with
  one CST beam, asserting shape `(nfreqs, ntimes, len(baselines))` and
  non-zero output.
- `test_simulate_vis_all_baselines` — asserts `nbls = N(N-1)/2 + 1`
  (i.e. autos included) when `baselines=None`.
- `test_simulate_vis_precision` — asserts `vis_double.dtype ==
  complex128`, `vis_single.dtype == complex64`, and they agree at
  `rtol=1e-5, atol=1e-5`.

### 15.5 `test_core_utils.py`, `test_antenna_gridding.py`,
`test_gpu_*.py` — cover the remaining surface

- `IDEALIZED_BL_TOL` is a positive float; `speed_of_light == 299792458`.
- `get_task_chunks(3, 30, 1)` ⇒ 3 chunks of 10 frequencies each.
- `get_task_chunks(10, 5, 1)` ⇒ collapses to 1 chunk (fewer tasks than
  procs).
- `get_pos_reds` on a five-antenna cross gives 6 redundant groups
  without autos and 7 with autos; total baselines `5 choose 2 = 10`
  without autos, 15 with.
- `get_plane_to_xy_rotation_matrix` is a proper rotation
  (`R^T R = I`, `det(R) = 1`).
- `inplace_rot_base` correctly applies a 90° z-rotation.
- `_use_gpu()` returns `True` when `cupy` is importable, `False`
  otherwise.
- The griddability test parametrizes over linear, square,
  square-with-holes, hex, scattered, and autos-only inputs and
  verifies the correct True/False label and that the rounded
  positions are integer.
- All GPU stubs raise `NotImplementedError`.

### 15.6 CI

`.github/workflows/ci.yml`:

- Matrix: `{ubuntu-latest, macos-latest} × {3.10, 3.11, 3.12}` (six
  jobs, `fail-fast=false`).
- Sets up MPI via `mpi4py/setup-mpi@v1` (because pyuvsim depends on
  it transitively).
- Installs `pip install .[dev]`.
- Runs `python -m pytest --cov=fftvis --cov-config=./.coveragerc
  --cov-report xml:./coverage.xml --durations=15`.
- Uploads coverage XML to Codecov (`codecov/codecov-action@v6`,
  `flags: unittests`, `fail_ci_if_error: true`).
- macOS jobs set `LOG_LEVEL=WARNING` to suppress logging on macOS.
- The Linux-only Ray tests are gated by
  `@pytest.mark.skipif(sys.platform == "darwin")`.

`ci/fftvis_tests.yml` is a conda env spec (`name: tests`,
channels conda-forge + defaults) for users who prefer to set up the
test env via mamba/conda rather than pip.

---

## 16. Tutorials

`docs/tutorials/`:

- **`fftvis_tutorial.ipynb`** — the canonical "first user run". Shows
  how to build an antenna dict by hand or via `hera_sim.antpos`,
  load a `pyuvdata.UVBeam` from a CST file, build a small ICRS sky,
  and call `simulate_vis(...)`. Concludes with side-by-side plots vs
  `matvis.simulate_vis(...)`.
- **`fftvis_gridded_array.ipynb`** — demonstrates the Type-1 path:
  uses a hex grid + a square grid, sweeps `force_use_type3 ∈ {False,
  True}`, and shows the `~10×` walltime difference for HERA-like
  arrays. Mirrors the matrix in `test_simulate_gridded_type1_vs_type3`.

Both notebooks live in `docs/tutorials/` only — there's no Sphinx
docs build, no `docs/conf.py`, no `readthedocs.yml`. The notebooks
are documentation in their own right.

---

## 17. RIME conventions, in one place

Putting the conventions scattered across the source into one summary,
because they matter when comparing against `matvis`, `pyuvsim`, RadioSim
core, and CASA:

1. **Coordinate frame** — sources are ICRS `(ra, dec)` in radians;
   antenna positions are ENU metres relative to `telescope_loc`
   (an `EarthLocation`). The matvis-derived rotation manager handles
   ICRS → topocentric internally.
2. **Coherency** — `C = (1/2) * [[I+Q, U+iV], [U-iV, I-Q]]`. The
   `0.5` factor lives in `prepare_source_catalog`. This is the
   `IAU/CASA` linear-feed convention used by `pyuvsim`, `matvis`, and
   RadioSim.
3. **Phase factor** — `exp(-2πi · b·X / λ)`. fftvis folds the `2π`
   into the `topo` array (`topo *= 2π`) once per integration, so the
   NUFFT calls themselves see "frequencies" `(u, v) = (b·c/c, b·c/c)
   * f / c = bf/c` in cycles per radian. The sign of the exponent
   matches `matvis` and `pyuvsim`.
4. **Apparent flux** — for source `n`, the apparent coherency seen by
   feeds `(p, q)` of antennas `(i, j)` is
   `B_p(s_n)^* @ C_n @ B_q(s_n)^T`, which fftvis computes as
   `B_p† C_n B_q` in `get_apparent_flux_polarized`.
5. **Output normalization** — `V_XX + V_YY = I` (no factor of 2). Shape
   axes as documented in §5.
6. **Above-horizon culling** — `coord_mgr.select_chunk` returns only
   sources currently above the horizon; below-horizon sources never
   reach the NUFFT.

---

## 18. Limitations & gotchas (verified from the source)

These are not just README claims — they are visible in the code:

1. **One beam for the whole array**. `_cpu_beam_evaluator.beam_list =
   [beam]` is a length-1 list; per-antenna beams would require
   re-evaluating the beam for each antenna and threading antenna
   identity through the NUFFT, which fftvis does not do. (Compare with
   `matvis`, which supports per-antenna beams natively.)
2. **No diffuse-sky support**. The flux array is a flat
   `(nsources, nfreqs[, 4])` cube. Diffuse skies must be pre-pixelized
   (e.g. HEALPix → list of equal-flux point sources). RadioSim has
   `core/visibility_healpix.py` for this, but fftvis itself does not.
3. **`eps` discrepancy**. `default_accuracy_dict[2] = 1e-13`, but
   the `wrapper.simulate_vis` docstring says `1e-12` for precision=2.
   The actual default is `1e-13`.
4. **`flat_array_tol` default mismatch**. The `wrapper.simulate_vis`
   default is `1e-6`, but the abstract `SimulationEngine.simulate`
   default is `0.0`. The wrapper default is what reaches the engine
   in normal usage.
5. **Per-call beam interpolation for multi-frequency UVBeams**.
   The wrapper interpolates the beam onto the user's `freqs` once via
   `beam.interp(freq_array=freqs, new_object=True, run_check=False)`.
   This matters because `pyuvdata.UVBeam.interp` is *itself* expensive
   for high-resolution beams; running fftvis with a multi-frequency
   beam and a small number of frequencies will spend a non-trivial
   share of its time just in this prep step. (The line is excluded
   from coverage.)
6. **Ray on macOS is flaky**. Every Ray test is `@skipif("darwin")`.
   Production fftvis on macOS should run with `nprocesses=1` and
   `force_use_ray=False`.
7. **GPU backend does not function**. Only the scaffolding exists.
8. **`coord_method="CoordinateRotationERFA"` and `pyuvsim`
   disagreement**. The `pyuvsim` cross-check test
   (`test_sim_polarized_sky`) passes only when
   `coord_method="CoordinateRotationAstropy"` and
   `interpolation_function="az_za_simple"`. With the default ERFA +
   `az_za_map_coordinates`, fftvis matches `matvis` but not
   `pyuvsim` exactly.
9. **`logutils.log_progress` references an undefined `rss`**. Line
   `return t, rss` should clearly be `return t, used` (the local
   variable just computed). This module is not currently called from
   the engine, so the bug never surfaces — but anyone resurrecting
   the progress logger will hit it.
10. **The `_cpu_beam_evaluator` singleton is mutated from inside
    `_evaluate_vis_chunk`**. This is fine inside Ray actors (each
    actor has its own module instance) but means the evaluator state
    is *not* re-entrant within a single process — running
    `_evaluate_vis_chunk` from two threads in the same process would
    interleave assignments to `_cpu_beam_evaluator.beam_list`,
    `freq`, `polarized`. This is why the inner loop runs serially
    inside `threadpool_limits(...)` and why parallelism happens at
    the chunk granularity, not the integration granularity.

---

## 19. Comparison vs the rest of the simulators in this repo

For RadioSim purposes, the most useful framing is "how does fftvis
relate to matvis, pyuvsim, and RadioSim core?"

| Property | fftvis | matvis | pyuvsim | RadioSim core |
|---|---|---|---|---|
| Algorithm | Type-3 / Type-1 NUFFT (finufft) | Dense matrix RIME (numpy / CUDA) | Brute-force per-source RIME, MPI | Full RIME with Jones chain |
| Per-antenna beams | **No** | Yes | Yes | Yes |
| Polarized sky | Yes (`(nsrc, nfreq, 4)`) | Yes | Yes | Yes |
| Diffuse sky | Pre-pixelized only | Pre-pixelized only | Yes (component-based) | HEALPix native (`visibility_healpix.py`) |
| GPU | Stub | Working (CuPy) | No | Via JAX/Numba backends |
| Parallelism | Ray + threadpoolctl | Multi-process via Ray | MPI | JAX/Numba/Dask |
| Coord rotation | ERFA / Astropy (matvis) | ERFA / Astropy (own) | Astropy | Custom |
| Backward compat with matvis API | Drop-in (same shape, same beam types) | — | Different | Different |
| Reference accuracy | High (1e-13 default eps) | Reference itself | Reference (slowest, most accurate) | RIME reference |
| Notable strength | ~10× speedup vs matvis on HERA-like arrays | Robust, mature, GPU-ready | Accuracy of record | Multi-component Jones, GPU+CPU backends, full polarization |
| Notable weakness | Same-beam-everywhere, GPU stub, no diffuse | Slower for large N_src | Very slow | Custom RIME, less mature |

The intended fit:

- **For HERA-like arrays with shared beams and many sources** ⇒
  fftvis is the right tool.
- **For per-antenna beams or per-feed gain studies** ⇒ matvis or
  RadioSim.
- **For accuracy-of-record validation** ⇒ pyuvsim.
- **For full Jones-chain, multi-physics simulations** ⇒ RadioSim core.

`matvis.md` and `pyuvsim.md` in this directory are the corresponding
exhaustive references for those simulators.

---

## 20. Concrete usage recipes

### 20.1 Minimal call

```python
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy import units as un
from pyuvdata import UVBeam
from fftvis import simulate_vis

ants = {
    0: np.array([0.0,  0.0, 0.0]),
    1: np.array([14.6, 0.0, 0.0]),
    2: np.array([0.0,  14.6, 0.0]),
}

beam = UVBeam()
beam.read_cst_beam(
    "HERA_NicCST_150MHz.txt",
    frequency=[150e6],
    telescope_name="HERA", feed_name="Dipole", feed_version="1.0",
    feed_pol=["x"], model_name="dipole", model_version="1.0",
)

freqs  = np.array([150e6])
times  = Time(["2020-01-01T00:00:00"], scale="utc")
ra     = np.array([0.0, 0.5])
dec    = np.array([-0.5, 0.0])
fluxes = np.ones((2, 1))   # 2 sources, 1 freq, Stokes I only

vis = simulate_vis(
    ants=ants,
    fluxes=fluxes,
    ra=ra, dec=dec,
    freqs=freqs, times=times,
    beam=beam,
    telescope_loc=EarthLocation.from_geodetic(
        lat=-30.7215*un.deg, lon=21.4283*un.deg, height=1051.69*un.m
    ),
    polarized=False,
)
# vis.shape == (1, 1, 4) -- 4 = 3 cross-baselines + 1 redundant-auto
```

### 20.2 Polarized run with HERA hex layout

```python
from hera_sim.antpos import hex_array
ants = {k: list(v) for k, v in hex_array(7, split_core=False).items()}

vis = simulate_vis(
    ants=ants,
    fluxes=stokes,                # shape (nsrc, nfreq, 4)
    ra=ra, dec=dec,
    freqs=freqs, times=times,
    beam=beam,
    telescope_loc=hera_loc,
    polarized=True,
    precision=2,                  # complex128
    nprocesses=8,
    coord_method_params={"update_bcrs_every": 60.0, "source_buffer": 0.75},
)
# vis.shape == (nfreq, ntime, 2, 2, nbls)
```

### 20.3 Forcing the Type-3 path (for cross-checks)

```python
vis = simulate_vis(
    ants=ants, fluxes=fluxes, ra=ra, dec=dec, freqs=freqs, times=times,
    beam=beam, telescope_loc=hera_loc,
    force_use_type3=True,         # disable griddability detection
    flat_array_tol=1e-6,
)
```

### 20.4 Reproducing the pyuvsim accuracy reference

```python
vis = simulate_vis(
    ants=ants, fluxes=fluxes, ra=ra, dec=dec, freqs=freqs, times=times,
    beam=uvbeam, telescope_loc=hera_loc,
    polarized=True,
    coord_method="CoordinateRotationAstropy",
    interpolation_function="az_za_simple",
    eps=1e-12,
)
```

### 20.5 Profiling

```bash
fftvis run-profile --hera 5 --ntimes 60 --nfreq 4 --nsource 1000 \
                   --nprocesses 8 --backend cpu
```

Produces a `cProfile` dump and a flamegraph file in the current
directory.

---

## 21. Summary

`fftvis` is a focused tool: it accelerates the RIME for a specific —
but very common — observational regime (single-beam HERA-like arrays
with many sources) by recognizing the visibility integral as a NUFFT.
The implementation is small (~2,000 LOC of production Python),
heavily reuses `matvis`'s coordinate and beam infrastructure, and is
parallelized in a deliberate, conservative way (Ray actors over
chunks, BLAS threadpool capping inside each actor, no per-integration
threading). The Type-1 lattice path is the most clever piece — it
reduces redundant-baseline cost for HERA-like grids dramatically by
formulating the whole integration as a single Type-1 NUFFT producing
a regular Fourier-mode grid that's then index-selected. The GPU
backend is on the roadmap but currently exists only as scaffolding.
For RadioSim the relevant integration point is treating fftvis as an
alternative `RIMESimulator`-style strategy: the input shapes
(`(nsrc, nfreq[, 4])` flux, `BeamInterface` beam, ENU antennas, ICRS
sources) are nearly identical to RadioSim's existing point-source RIME,
so plumbing fftvis in as a `simulator/fftvis.py` strategy is a
mechanical exercise rather than a re-design.
