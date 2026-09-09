# Configuration and instrument migration

RadioSim is pre-v1.0. The current API replaces split instrument inputs with
one typed source and does not preserve compatibility aliases.

## Public construction

Use `Simulator.from_yaml(path)`, `from_config(model, base_dir=...)`,
`from_mapping(mapping, base_dir=...)`, or typed parameter construction:

```python
from radiosim import Simulator

simulator = Simulator.from_parameters(
    instrument=instrument,
    baseline_selection=baseline_selection,
    channel_frequencies_hz=(100_000_000.0, 101_500_000.0),
    channel_widths_hz=(1_000_000.0, 1_000_000.0),
    start_time="2025-01-01T00:00:00",
    sky_model=sky_model,
)
```

The direct `Simulator(resolved_runtime)` constructor accepts only
`ResolvedSimulationConfig`. Passing a YAML path to `from_config`, or passing
raw mappings/backend scalars to the direct constructor, is no longer valid.

## Canonical simulation results

`Simulator.run()` now returns an immutable `SimulationResult`, and the singular
`Simulator.result` property exposes the identical last successful result.
There is no plural `Simulator.results` alias or dictionary adapter.

```python
result = simulator.run()
assert result is simulator.result

visibilities = result.visibilities  # (time, baseline, frequency, correlation)
assert result.polarization_basis == "linear_xy"      # data, not a constant
assert result.correlations == ("XX", "XY", "YX", "YY")
stokes_i = result.stokes_i()
```

The correlation labels are no longer fixed. Read `result.correlations` and
`result.polarization_basis` instead of hard-coding `XX, XY, YX, YY`; a circular
result reports `RR, RL, LR, LL`. See the receptor section below.

### Linear output now means X=east, Y=north

The unreleased SCI-006 correction changes polarized linear-output numbers. The
canonical sky brightness matrix is still ordered `(North, East)`, but the
unrotated linear receptor now applies

```text
P = [[0, 1], [1, 0]]
```

before reporting `(X=east, Y=north)`. The corrected zero-rotation products are

```text
XX=(I-Q)/2  XY=(U-iV)/2  YX=(U+iV)/2  YY=(I+Q)/2
XX-YY=-Q
```

Pure-I output and ideal circular output do not change. For ideal or scalar
linear chains, polarized output is related to the earlier value by

```text
V_new = P V_old P^H
[XX, XY, YX, YY]_new = [YY, YX, XY, XX]_old
```

The matrix permutation swaps the cross-hand values; it does not itself take
their complex conjugates. Consumers that applied a manual Q-sign or X/Y
compensation must remove it.

Feed-asymmetric Jones configuration remains keyed to the physical native feed:
for linear antennas, feed `0` is X/east and feed `1` is Y/north; for circular
antennas they are R and L. Do not swap `G`, `B`, `Rc`, `Kd`, `X`, or `D`
parameters. Re-run any persisted polarized-linear reference products and compare
them against the corrected runtime; do not silently retain old fingerprints.
The `P V P^H` shortcut above is not generally valid when feed-dependent Jones
terms do not commute with the basis change, because those terms act in physical
feed coordinates and require a full re-simulation.

Coordinates, masks, provenance, and fingerprints are available through
`result.time_grid`, `result.frequencies_hz`, `result.channel_widths_hz`,
`result.flags`, `result.weights`, `result.scientific_sha256`, and
`result.provenance_sha256`. Stored Stokes I and per-baseline correlation
dictionaries have no replacement; derive Stokes I from the canonical array.

`Simulator.save()` now takes an exact final artifact path:

```python
from radiosim import ResultFormat

simulator.save("output/result", format=ResultFormat.HDF5)
simulator.save("output/result", format=ResultFormat.SUMMARY_JSON)
```

The old `output_dir`, `filename`, and string-format arguments have no
compatibility overload. `json` was removed because it did not contain
visibilities: use `summary_json` for metadata or `hdf5` for a lossless RadioSim
result. Workflow `overwrite`, `skip_overwrite_confirmation`, and
`prompt_for_output_suffix` were replaced directly:

- `workflow.overwrite: removed before v1.0; use workflow.collision_policy`
- `workflow.skip_overwrite_confirmation: removed before v1.0; use collision_policy=replace`
- `workflow.prompt_for_output_suffix: removed before v1.0; use collision_policy=suffix`
- `workflow.result_format=json: removed before v1.0; use summary_json or hdf5`

Use `collision_policy: error|replace|suffix|prompt`. Prompting is CLI-only,
TTY-only, and limited to valid owned manifested runs.

The two ambiguous visualization inputs were removed with the same exact text:

- `workflow.angle_unit: removed before v1.0; use workflow.visibility_phase_unit`
- `workflow.sky_model_frequency_hz: removed before v1.0; no Tier 4 sky renderer consumes it`

`visibility_phase_unit` is exactly `radians` (default) or `degrees` and controls
only the displayed visibility phase axis. There is no workflow sky-image
renderer, so no workflow field selects a sky-model frequency; a future sky
renderer requires its own typed request.

`Simulator.plot` is keyword-only and takes `plot_type`, `output_dir`, `backend`,
`show`, `overwrite`, and `visibility_phase_unit`. The visibility renderers
`plot_visibility`, `plot_heatmaps`, and `plot_modulus_vs_frequency` now accept a
single `SimulationResult` positional argument; the old `moduli_over_time`,
`phases_over_time`, `mjd_time_points`, `total_seconds`, `freqs`, `baselines`,
and `angle_unit` parameters are removed. Rebuild plots from
`simulator.run()` output instead of assembling per-baseline dictionaries.

## Instrument input

Before migration, telescope identity, layout, location, and diameter could be
split across top-level sections. Replace them with one source:

```yaml
instrument:
  source:
    kind: layout_file
    path: antennas.txt
    format: radiosim
    telescope_name: Example Array
  location:
    longitude_deg: 21.4283
    latitude_deg: -30.7215
    height_m: 1050.0
  default_diameter_m: 14.0
  diameter_overrides:
    - antenna:
        kind: name
        name: HH140
      diameter_m: 12.0
```

The retained file-format names are `radiosim`, `casa_loc`,
`measurement_set`, `uvfits`, and `mwa_metafits`. A registry request is instead:

```yaml
instrument:
  source:
    kind: known_telescope
    name: HERA
    registry_policy: offline
```

Removed instrument fields map as follows:

| Removed key | Replacement |
| --- | --- |
| `telescope` / top-level `telescope_name` | `instrument.source.telescope_name` for local layouts, or `instrument.source.name` for a known telescope |
| `antenna_layout` | `instrument.source` with `kind: layout_file` |
| `antenna_positions_file` | `instrument.source.path` |
| `antenna_file_format` | `instrument.source.format` |
| top-level `location` (`lon`, `lat`, `height`) | `instrument.location` (`longitude_deg`, `latitude_deg`, `height_m`) |
| `all_antenna_diameter` | `instrument.default_diameter_m` |
| `use_different_diameters` and `diameters` | typed `instrument.diameter_overrides` entries |
| `use_pyuvdata_telescope`, `use_pyuvdata_location`, `use_pyuvdata_antennas`, `use_pyuvdata_diameters` | choose exactly one discriminated source |
| `antenna_layout.fixed_HPBW` | analytic configuration under `beams` |
| `instrument.location.ra` / `.dec` | no replacement; explicit phase-centre input is not implemented |
| top-level `feeds` | the `receptors` section with `default.basis`, `default.feed_rotation_deg`, and `output_basis` |

Unknown or removed keys fail validation with a focused replacement message.

## Baseline selection

The boolean/parallel-list shape is replaced by typed criteria:

```yaml
baseline_selection:
  correlations: cross
  length_filter:
    mode: ranges
    ranges_m:
      - min_m: 10.0
        max_m: 100.0
  azimuth_ranges_deg:
    - start_deg: 170.0
      end_deg: 10.0
```

`use_autocorrelations` and `use_crosscorrelations` become `correlations`.
`only_selective_baseline_length`, `selective_baseline_lengths`, and
`selective_baseline_tolerance_meters` become a typed `length_filter` using
`targets` or `ranges`. `trim_by_angle_ranges` and
`selective_angle_ranges_deg` become typed `azimuth_ranges_deg`.

## Execution and workflow

The removed top-level sections map to:

- `compute` -> backend/offline values under `execution`;
- top-level `precision` -> `execution.precision`;
- `simulators` -> `execution.simulator: rime`;
- `output` -> CLI-only `workflow`.

`workflow` never enters `ResolvedSimulationConfig`, and Python construction
does not execute post-run actions.

### Worker policy: `run(n_workers=...)` removed

`Simulator.run()` no longer accepts `n_workers`, and `progress` is now
keyword-only. The new signature is:

```python
def run(self, *, progress: bool = True) -> SimulationResult: ...
```

Passing the removed keyword raises Python's own `TypeError`:

```text
run() got an unexpected keyword argument 'n_workers'
```

Solver concurrency is declared in the configuration instead, so it is resolved
once, recorded in `result.resolved_config`, and hashed into
`provenance_sha256` like every other execution policy:

```yaml
execution:
  solver:
    workers: 4        # default 1; clamped to the number of time samples
    executor: thread  # the only supported value
  sky_loading:
    max_workers: 4    # loader-side policy, independent of the solver
    executor: auto
```

`execution.n_workers` — the removed configuration field that never reached the
solver — is likewise rejected, with a message naming both replacements.

Solver workers parallelize the time axis only: each worker computes a
contiguous block of time samples and the blocks are reassembled in time order,
so any `workers` value produces a bit-identical result to `workers: 1`. Choose
it for wall-clock time, never for numerical reasons.

### Sky-loader concurrency and offline policy

`load_models_parallel()` no longer defaults `max_workers` to a hard-coded `8`;
the argument is required, and the value comes from
`execution.sky_loading.max_workers` (`null` means
`min(requests, cpu_count, 8)`). The resolved value and the executor that
actually ran are recorded in the summary JSON, so the knob is observable rather
than assumed.

`execution.offline: true` is now authoritative for loaders under both
executors. A configuration that previously reached the network from inside a
worker despite being offline now fails fast with `ConnectionError` instead. This
is a behavior change for any offline run that was quietly online.

### Solver accumulation restructure

Both solvers assemble one `(B, F, 2, 2)` block per time step and one whole cube
per call, instead of writing each `(time, baseline, frequency)` cell with
`set_at`. Nothing about the result changes — the restructure is asserted
bit-identical against pre-restructure fingerprints for every shipped
configuration — but a functional-array backend no longer pays a whole-cube copy
per cell.

## Backend registry

### `numba` removed; `NumbaBackend` is now `DaskBackend`

The backend named `numba` never compiled a kernel: it called the same NumPy
operations the NumPy backend calls. The name is removed rather than kept as an
alias, because an alias would preserve the false claim.

```yaml
execution:
  backend: dask   # was: numba
```

`execution.backend: numba` is rejected by the schema:

```text
execution.backend=numba: removed before v1.0; the backend never compiled any
kernel. Use execution.backend=dask for the NumPy/Dask backend or
execution.backend=numpy.
```

`get_backend("numba")` raises with the same guidance, and the CLI's
`--backend` choice list is `auto|numpy|jax|dask`. The class is now
`DaskBackend`, reporting `dask-cpu` or `dask-distributed`; `mode="gpu"` and the
CUDA validation path are gone (they validated a device and then ran NumPy), and
`jit_compile()` is gone (it had no caller). The rename adds no compilation and
no acceleration. The `numba` install extra is now `dask`:

```bash
pip install radiosim[dask]   # was: radiosim[numba]
```

### `auto` precedence and `supports_gpu`

`get_backend("auto")` no longer returns the NumPy-delegating backend. It
returns the JAX backend only when JAX reports a **non-CPU** device, and the
NumPy backend otherwise; it never selects Dask, because reporting `dask` for a
run that executes plain NumPy is the misreporting this change exists to remove.
Recorded `actual_backend` provenance values change accordingly, and
`provenance_sha256` changes with them.

`RIMESimulator.supports_gpu` is now `False`. It reported `True` while executing
host-side NumPy.

### PERF-001 deterministic backend selection

The earlier 0.3.0 correction above still allowed `get_backend("auto")` to
probe JAX and select it when the runtime exposed a non-CPU device. `auto` now
checks only whether NumPy can honor the requested precision. It never imports
or probes JAX, never selects Dask, and raises `BackendNotAvailableError` when
NumPy cannot honor the request.

An unqualified `get_backend("jax")` now uses JAX's runtime-default device; it
no longer means “request GPU, then fall back to CPU.” Explicit
`device="cpu"`, `"gpu"`, and `"tpu"` requests and the direct `gpu` / `tpu`
aliases are strict. An unavailable device raises `BackendNotAvailableError`
without a CPU fallback and retains the runtime failure as its cause. Generic
device-resource reporting likewise no longer uses JAX as a fallback; use
`list_backends()` or `get_backend_info()` when explicit JAX discovery is
intended (`PERF-001`).

Third-party `VisibilitySimulator` subclasses now inherit
`supports_gpu = False` rather than `True`. Returning `True` requires an
independently accepted end-to-end accelerator record for that exact
implementation. The shipped `RIMESimulator` remains explicitly false
(`PERF-001`).

### `ArrayBackend` additions

Third-party backend implementations (there are none in tree) gain four members
and one widened signature:

```python
def stack(self, arrays, axis=0): ...          # the solvers' accumulation primitive
def add(self, a, b): ...                      # hybrid component summation
@property
def supports_compilation(self) -> bool: ...   # base default False
def compile(self, func): ...                  # base default: identity
def synchronize(self, arr=None): ...          # now takes the array to block on
```

`synchronize()` without an argument keeps its previous best-effort behavior,
which orders none of the caller's work; pass the array, or a JAX timing
measures dispatch rather than computation.

### Initial 0.3.0 CPU-only JAX dependency

At the Tier 6H migration point, every declared Pixi environment carried a
CPU-only `jax`/`jaxlib`, so the NumPy/JAX parity evidence was measured in the
standard gate rather than skipped. A missing JAX remains a broken standard
environment and fails loudly.

Later PERF-001 readiness work leaves the `default`, `py312`, and `crossval`
environments CPU-only and adds a separate Linux `gpu` environment with a strict
CUDA preflight. That isolated environment is readiness infrastructure, not an
accelerator measurement or capability claim. No accelerator has been measured;
see [the backend guide](user_guide/backends.rst) for the records.

## Hybrid results and serialization

`visibility.sky_representation` accepts a third value, `hybrid`, which solves a
point component and a HEALPix component on one shared instrument, beam system,
receptor set, time grid, and backend, and sums them into one canonical
`SimulationResult`. Every result — hybrid or not — now records which components
it solved:

```python
result.solver.sky_representation      # "point_sources" | "healpix_map" | "hybrid"
result.solver.components              # ("point",) | ("healpix",) | ("point", "healpix")
result.solver.component_element_counts
result.performance.solver_point_seconds
result.performance.solver_healpix_seconds
```

Component names and counts are deterministic and are part of the scientific
identity, so `scientific_sha256` changes for every result, including
single-component ones, and a hybrid result can never collide with a
single-component one over the same instrument and sky numbers. The two timings
are nondeterministic and stay out of both fingerprints. Do not compare
fingerprints across this boundary.

### Two silent conversions became explicit

Before this change, a sky model carrying both kinds of payload was quietly
reduced to whichever one the requested representation named. Both reductions are
now rejected, because both silently changed the science.

`sky_representation: point_sources` against a model that still carries a HEALPix
payload:

```text
visibility.sky_representation=point_sources would discard the HEALPix payload
carried by the resolved sky model. Request hybrid to sum both components, or set
visibility.allow_lossy_point_materialization=true to convert the HEALPix payload
to point sources.
```

`sky_representation: healpix_map` against a model that contributes point
sources:

```text
visibility.sky_representation=healpix_map would rasterize {n} point source(s)
into the HEALPix grid, which quantizes positions to pixel centers. Request
hybrid to sum both components, or set
visibility.allow_lossy_point_rasterization=true to opt in.
```

`visibility.allow_lossy_point_rasterization` is a new boolean, defaulting to
`false`. A configuration that relied on the old silent rasterization must either
move to `hybrid` (which sums both components with no loss) or set the flag and
accept the quantization. None of the shipped configurations relied on it.

### HDF5 schema `4.0.0`

HDF5 results moved to schema `4.0.0`, which adds the optional `jones/` group:
`enabled_terms`, `chain_order`, `term_snapshots_json`, `mount_types_json`, and
`jones_sha256`. The group is written only when a run enables a Jones term, and
a file without it reads as "no optional terms enabled". This preserves the
optional group's structural absence for the current empty optional-term
inventory; it does not preserve pre-SCI-006 visibility values or scientific
fingerprints.

Schema `3.0.0` files are rejected with `UnsupportedSchemaVersionError` and are
not upgraded in place. Re-run the simulation to write a `4.0.0` file.

### HDF5 schema `3.0.0`

HDF5 results moved to schema `3.0.0`. `provenance/solver_json` gains
`components` and `component_element_counts`; `provenance/performance_json`
gains `solver_point_seconds` and `solver_healpix_seconds`. No visibility array
shape, dtype, correlation order, weight, or flag semantics changed.

Schema `1.0.0` and `2.0.0` files are both rejected with
`UnsupportedSchemaVersionError`, and neither is upgraded in place — there is no
upgrade path by design. Re-run the simulation to write a `3.0.0` file.

The reader validates the two new records against the canonical dataclass field
sets, bounds the component list, and cross-checks the declared
`sky_representation` against the embedded resolved configuration, all before
any science payload is allocated. A file whose solver record was edited to
relabel a point-only result as hybrid is rejected as an unsafe input.

### Summary JSON `1.1.0` and standard formats

The summary JSON reports the components and both timings in its existing
`solver` and `performance` blocks, and its `schema.version` moves `1.0.0` →
`1.1.0`. The bump is deliberately minor where the HDF5 schema takes a major
one: nothing was removed or retyped, every `1.0.0` key survives at the same
path with the same meaning, and the document has no reader to break.

Measurement Set and UVFITS have no schema change. Their `HISTORY` gains three
lines, because a summed hybrid visibility is not reconstructible from either
file otherwise:

```text
sky_representation=hybrid
solver_components=point,healpix
solver_component_element_counts=20,3072
```

The same facts also travel inside the `RADIOSIM_PROJECTION_JSON=` record's
`solver` object.

## Beam input

The flat beam object and BeamManager compatibility keys are rejected rather
than translated. Choose one complete tagged mode. The runnable replacement for
the former shared analytic default is:

```yaml
beams:
  mode: analytic
  model:
    kind: circular_aperture
    taper:
      kind: gaussian
      edge_taper_db: 10.0
```

For FITS declarations, use `mode: shared_fits` with `beam.path`,
`mode: per_antenna_fits` with ordered `assignments`, or `mode: mixed` with an
`analytic_model` and ordered analytic/FITS choices. All four modes now resolve,
load, and run through the canonical `BeamSystem`; FITS declarations are limited
to the documented scalar subset and never fall back to analytic evaluation.

| Rejected key | Replacement or reason |
| --- | --- |
| `beam_mode` | `beams.mode` with a complete `analytic`, `shared_fits`, `per_antenna_fits`, or `mixed` shape |
| `per_antenna` | `beams.mode: per_antenna_fits` and `beams.assignments[]` |
| `beam_file` / `beam_file_path` | a tagged FITS source at `beams.beam` or `beams.assignments[].beam` |
| `antenna_beam_map` / `beam_assignment` | ordered tagged `beams.assignments[]` entries |
| `beam_files` / `beams_per_antenna` | ordered FITS assignment entries |
| `beam_peak_normalize` | `normalization: peak` on each FITS source |
| `beam_interp_function` / `beam_freq_interp` | `angular_interpolation: bilinear` and `frequency_interpolation: cubic` or `linear` |
| `beam_za_max_deg` / `beam_za_buffer_deg` | no replacement; Tier 3 requires the full visible hemisphere |
| `beam_freq_buffer_hz` / `beam_freq_buffer_mhz` | no replacement; Tier 3 loads the full declared frequency axis |
| `aperture_shape` | tagged `beams.model.kind` |
| `taper` | `beams.model.taper.kind`, or `taper_profile.kind` for analytical illumination |
| `edge_taper_dB` | `edge_taper_db` on a Gaussian, parabolic, or parabolic-squared direct taper |
| `feed_model` / `feed_computation` | an `analytical_illumination` or `numerical_illumination` model with typed `illumination` |
| `feed_params` | typed `focal_ratio` plus `q`, `b_over_lambda`, or `height_wavelengths` |
| `reflector_type` / `magnification` | a tagged `reflector`; Cassegrain alone accepts `magnification` greater than one |
| `aperture_params` | explicit rectangular lengths or elliptical diameters on the selected model |
| `use_beam_file` / `use_different_beams` | select the corresponding tagged mode directly |
| `default_beam_id` | no replacement; author a complete assignment list |
| `all_beam_response` | no replacement; select a complete tagged model or source |

Unknown fields remain errors. Configuration resolution never reads BeamFITS
content; canonical loading occurs during setup.

### New: `beams.pointing` and `beams.surface_error`

Two optional per-antenna blocks were added. Both are additive: omit them and
the cube, every beam fingerprint, and `scientific_sha256` are exactly what they
were.

```yaml
beams:
  mode: analytic
  model: {kind: circular_aperture}
  pointing:
    default: {azimuth_offset_deg: 0.0, elevation_offset_deg: 0.1}
  surface_error:
    default: {rms_surface_error_m: 0.002}
```

`pointing` is a deterministic mount mispointing, composed as the two encoder
errors of an alt-az mount; `surface_error` is the Ruze (1966) random-surface
RMS in metres, applied to the voltage beam as the square root of the power
efficiency. Both closed forms are public as
`radiosim.core.beam.runtime.ruze_power_efficiency` and `ruze_voltage_factor`.
See the beam guide for the exact geometry and the alt-az keyhole degeneracy.

### New: `beams.squint` and non-scalar `E`

One more optional per-antenna block was added (SCI-005 Stage 2), and it is
additive on the same terms as `pointing` and `surface_error` above: omit it
and the cube, every beam fingerprint, and `scientific_sha256` are exactly what
they were, and the no-squint `BeamSystem.evaluate_jones` call surface,
behaviour, and results are byte-identical to before.

```yaml
beams:
  mode: analytic
  model: {kind: circular_aperture}
  squint:
    default:
      convention: cotton_uson_exact_v1
      reference_frequency_hz: 1.5e8
      per_feed_offset_deg_at_reference: 2.0
      mechanical_feed_position_angle_deg: 35.0
      positive_native_feed: x
```

When a squint block *is* authored, three surfaces widen:

- **`E` is generally full**, not scalar, for a squint-carrying antenna. The
  chain order stays `H G B Rc Kd X D C E P T Z`; only what `E` itself
  contains changes. For any circular receptor the composed `E` still reduces
  to an exact scalar-plus-`sigma_y` form independent of the feed rotation, so
  the non-scalar effect is observable only on a linear receptor. See the beam
  guide's squint section and the Jones-matrix guide's chain-order discussion.
- `BeamSystem.evaluate_jones` gains two keyword-only parameters,
  `boresight_parallactic_rad: float | None = None` and
  `boresight_altitude_rad: float | None = None`. Both are required to be
  exact finite Python floats for a squint-carrying antenna, and required to
  stay `None` for every other antenna.
- `radiosim.core.beam.load_beam_system` gains one keyword-only parameter,
  `receptors: ResolvedReceptorSet | None = None`. It is required whenever any
  resolved antenna carries `beams.squint`, because the composed `E` is built
  from that antenna's own resolved receptor basis and static feed rotation —
  the same authority the solver's `C` term comes from, so the two can never
  disagree.

`beams.squint` is accepted only on the `analytic` beams mode; a `shared_fits`,
`per_antenna_fits`, or `mixed` document that also authors a squint block is
rejected. No migration is needed for a document that does not author
`beams.squint`.

### New: `beams.beam.normalization: uvbeam_peak_common_v1`

A BeamFITS source's `normalization` field accepted exactly one value, `peak`.
It now accepts a second, `uvbeam_peak_common_v1` (SCI-005 Stage 3), and that
one field is the whole activation surface for the full-efield subset.

```yaml
beams:
  mode: shared_fits
  beam:
    kind: fits
    path: beams/shared.beamfits
    normalization: uvbeam_peak_common_v1   # was: peak (still the default)
```

**Nothing changes for a document that keeps `peak`.** The default is unchanged,
and an existing `peak` run has the same resolved configuration, the same beam
fingerprints, the same `scientific_sha256`, the same HDF5 `provenance/beam_json`
bytes, and the same result cube as before.

When the new literal *is* authored, three things widen:

- **`E` is a generally full 2x2 matrix**, not a scalar on the diagonal. The
  file's complete complex `data_array` is converted by the frozen constant
  `M = [[0, 1], [-1, 0]]` into the chain's own mixed-sign tangent pair
  `(-e_theta, +e_az_uv)` — the pair `P` delivers — to give `J_native`, and the
  beam runtime
  composes `E = C^dagger J_native` from the antenna's own resolved receptor
  matrix, so `C E == J_native` exactly. The chain order stays
  `H G B Rc Kd X D C E P T Z`; only what `E` contains changes. Both cross-hand
  correlation products become non-zero, in both output bases. See the beam
  guide's full-efield section and the Jones-matrix guide's chain-order
  discussion.
- **The two literals are two accepted subsets of the same file, not a
  widening.** RadioSim renormalizes nothing under either. A file the scalar
  subset accepts is generally rejected by the full-efield subset, and the
  reverse; in particular the full-efield subset requires the stored
  `basis_vector_array` to be exactly the native identity, requires a
  full-stored-grid unit peak at every intrinsic frequency (a visible-row-only
  peak is rejected), requires the zenith row to satisfy the de-spin predicate
  `J(az_uv) = J(az_ref) R(az_uv - az_ref)`, requires the converted azimuth seam
  to satisfy the second-difference continuity predicate, and requires the
  file's feed pair, feed angles and derived x-orientation to agree with *every*
  antenna it is assigned to under the `receptors` section.
- **`BeamFileProvenance` gained seven optional fields**, appended after
  `normalization_absolute_tolerance` in declaration order:
  `accepted_subset_version`, `radiosim_normalization`, `resolved_feed_array`,
  `derived_x_orientation_verdict`, `basis_vector_convention`,
  `factorization_convention`, and `stored_grid_peak_by_frequency`. Each is
  declared `<type> | None = None` and is left `None` on the `peak` path, where
  it is omitted from both the beam snapshot and the canonical fingerprint
  payload — which is what keeps a `peak` document's bytes unmoved. Existing
  keyword construction of `BeamFileProvenance` stays valid. One existing field
  became nullable: `x_orientation` is now `str | None`, because pyuvdata
  legitimately derives no orientation for a rotated linear receptor or for a
  circular receptor whose static rotation is neither 0 nor pi/2. The scalar
  path still records exactly `"east"`. `basis_vector_convention` records the
  literal `uvbeam_theta_phi_chain_tangent_v1`, whose frozen definition is the
  constant `M` and the mixed-sign chain basis above.

`beams.squint` and `beams.aperture_physics` are accepted only on the `analytic`
beams mode, so neither can be combined with either BeamFITS subset. No
migration is needed for a document that does not author
`uvbeam_peak_common_v1`.

### Moved: `beam/TODO.md` became a scope document

`src/radiosim/core/jones/beam/TODO.md` no longer exists. It was an in-source
wish list shipped inside the installed package with no dispositions and no
owner. It is now `docs/development/beam_physics_scope.md`, a disposition table
in which every item carries its physics, its citation, whether RadioSim
implements it, and — where it does not — the register row that owns it. Nothing
in it is a promise.

## Removed low-level beam APIs

The former `BeamManager`, `BeamFITSHandler`, `BeamJones`,
`AnalyticBeamJones`, and `FITSBeamJones` classes were parallel mutable runtime
surfaces and have been deleted. Their imports fail immediately; there is no
compatibility shim. Resolve strict configuration to canonical assignment state
and use `BeamSystem` or `Simulator.beam_system` instead.

The dictionary-taking `compute_aperture_beam` composition function was also
removed. The mutable registries and feed-to-composition bridges were deleted
with it. The plotting helpers were deleted as a second raw-schema surface. Use
the retained independent numeric primitives under
`radiosim.core.jones.beam.analytic` for formula-level work, or configure a
typed analytic model and evaluate it through `BeamSystem`. Application plots
should consume canonical evaluated data rather than reconstructing a second
raw beam schema.

Old raw beam keys and default IDs have no runtime translation. Old solver
signatures that accepted a manager, handler dictionary, or optional beam
mapping must pass the exact canonical `BeamSystem` required by the current
solver boundary. Missing low-level imports and old call signatures fail
directly instead of selecting identity or analytic fallback behavior.

### Illumination primitives renamed

"Feed" now means the receiving receptor only. The beam subsystem's aperture
**illumination** primitives were renamed so the two vocabularies cannot be
confused, and the module that defines them was rehomed. The renames are direct;
there are no aliases.

| Removed name | Replacement |
| --- | --- |
| `radiosim.core.jones.beam.analytic.feed` (module) | `radiosim.core.jones.beam.analytic.illumination` |
| `corrugated_horn_pattern` | `corrugated_horn_illumination` |
| `open_waveguide_pattern` | `open_waveguide_illumination` |
| `dipole_ground_plane_pattern` | `dipole_ground_plane_illumination` |
| the `theta_feed` keyword on those three functions | `theta_illumination` |

`prime_focus_angle`, `cassegrain_angle`, and `compute_edge_angle` keep their
names and move with the module. Importing them from the package root
`radiosim.core.jones.beam.analytic` is unaffected. `feed_array`, `feed_angle`,
`x_orientation`, and `UnsupportedBeamFeedError` are **not** renamed: those
describe the receiving receptor and are correctly named.

## Receptors and polarization basis

Receptor physics is implemented for ideal orthogonal two-feed receptors. The
removed top-level `feeds` object is replaced by a `receptors` section, and the
correlation labels, HDF5 schema, and Stokes `V` sign all changed with it. There
is no compatibility flag for any of it.

The `feeds` key is rejected with an exact pointer:

- `feeds: top-level 'feeds' was replaced by the Tier 5 receptor model`

The runnable replacement, which is also the default when the section is omitted,
is:

```yaml
receptors:
  default:
    basis: linear
    feed_rotation_deg: 0.0
  overrides: []
  output_basis: auto
```

Old receptor-shaped keys map as follows, each with its own exact message:

| Rejected key | Replacement or reason |
| --- | --- |
| top-level `feeds` | the `receptors` section |
| `receptors.default.feed_type` | `receptors.default.basis`, exactly `linear` or `circular` |
| `receptors.default.n_feeds` | no replacement; every antenna has exactly two feeds, and single-feed or multi-feed antennas are rejected until Tier 7 |
| `receptors.default.feed_angle_deg` | `receptors.default.feed_rotation_deg`, an offset from the nominal orientation of the selected basis |

A `basis` other than `linear` or `circular`, and an `output_basis` other than
`auto`, `linear`, or `circular`, are rejected at schema validation. Naming
`output_basis: auto` for an array with mixed native bases is rejected at
resolution with `AmbiguousOutputBasisError`, which reports both antenna counts;
name the basis explicitly instead. Receptor resolution no longer looks at
`mount_type` at all, and a non-zero `feed_rotation_deg` combined with an enabled
parallactic-angle term is no longer rejected — see **Parallactic angle and
mount types** under *Jones terms and the visibility strategy selector*.

The stub constructors are removed, not deprecated:
`ReceptorConfigJones(feed_type=...)` and
`BasisTransformJones(from_basis=..., to_basis=...)` no longer accept those
keywords, or positional arguments. Both terms are now constructed from a
resolved receptor set and a solver instrument view, and both are always present
in the Jones chain. The solver entry points and the result factory gained a
required `receptors` parameter with no default; pass the
`ResolvedReceptorSet` that `resolve_receptors()` returns.

Three superseded polarization helpers were removed outright, not deprecated,
because Tier 5A's evidence showed each had no production caller:

| Removed symbol | Replacement |
| --- | --- |
| `radiosim.core.polarization.visibility_to_correlations` | no replacement; it hard-keyed the pre-Tier-5 linear labels, which the basis-aware `CORRELATION_LABELS` table (`radiosim.core.polarization_basis`) supersedes |
| `radiosim.core.polarization.mueller_from_jones` | no replacement; it only ever raised `NotImplementedError` and was never reachable from the `radiosim.core` package surface |
| `radiosim.core.receptor.PolarizationBasisName` | `radiosim.core.polarization_basis.PolarizationBasis` |

Each now fails immediately: `radiosim.core.visibility_to_correlations` raises
`AttributeError`, and `from radiosim.core.polarization import
mueller_from_jones` and `from radiosim.core.receptor import
PolarizationBasisName` both raise `ImportError`. There is no migration text
carried by those errors because none of the three had a replacement to name;
this table is that replacement statement.

Two scientific consequences have no opt-out:

- **Stokes `V` sign.** `stokes_to_coherency` now builds
  `[[I + Q, U + iV], [U - iV, I - Q]] / 2`, and `coherency_to_stokes` derives
  `V` from the `[0, 1]` element to match. The previous matrix was the mirror
  image under `V -> -V`. The blast radius is exactly the cross-hand
  correlations of sources with non-zero Stokes `V`; every `V = 0` result, every
  parallel hand, and every `stokes_i()` value is bit-identical to before.
- **Fingerprints.** `scientific_sha256` changes for every result, because the
  resolved receptor set is part of the canonical scientific identity. Do not
  compare fingerprints across this boundary.

HDF5 results moved to schema `2.0.0`, which adds the `receptors` group and makes
the stored correlation labels basis-driven. Schema `1.0.0` files are rejected
with `UnsupportedSchemaVersionError` and are not upgraded in place; re-run the
simulation. (Schema `2.0.0` has since been superseded by `3.0.0` — see
[Hybrid results and serialization](#hybrid-results-and-serialization) — and is
rejected on the same terms.) Measurement Set, UVFITS, the summary JSON, and
every renderer read
the resolved basis rather than assuming the linear labels. See
[`docs/api/io.rst`](api/io.rst) for the complete polarization mapping and
[`docs/user_guide/jones_matrices.rst`](user_guide/jones_matrices.rst) for the
receptor mathematics, the cross-basis output-native interpretation boundary,
and the parallactic-angle boundary.

## Frequency and configuration I/O

Frequency input requires `mode: grid` or `mode: explicit`; explicit centers in
`channel_frequencies_hz` require matching `channel_widths_hz`. Grid input
requires `channel_width` in `frequency_unit`. `load_config()` returns a resolved bundle,
`resolve_config()` accepts a mapping/model plus source context, and
`dump_config()` accepts only a `RadioSimConfig` input model. Removed custom
model methods should be replaced by those functions or standard `model_dump`.

Paths are source-aware: YAML paths use the document parent, mapping/model
relative paths require `base_dir`, and override paths use the captured call
directory. `~` is expanded; environment-variable syntax is rejected.

## Jones terms and the visibility strategy selector

`visibility.calculation_type` was removed before v1.0. It validated
`direct_sum` and `spherical_harmonic`, and no module in `src/radiosim` ever
read either value: `direct_sum` was a no-op and `spherical_harmonic` was
rejected by a validator that named a tier. A document that still sets it is
rejected with:

```text
visibility.calculation_type was removed before v1.0; the solver strategy is
selected by 'execution.simulator' (currently only 'rime').
```

Delete the key. `execution.simulator` is the one strategy selector, its accepted
values are exactly the keys of the simulator registry (`rime`), and that
equality is asserted by a standing test so a second, unread selector cannot
reappear. A spherical-harmonic or m-mode solver is a future simulator
registration, not a value on a removed field.

### Parallactic angle and mount types

`jones.P` is implemented. Two earlier rejections are gone, and two new ones
replace them.

Gone: receptor resolution rejected **every** antenna whose `mount_type` was not
`fixed`, with a message that named a tier rather than a fix, so an ordinary
alt-azimuth array could not be simulated at all. It also rejected a non-zero
`feed_rotation_deg` combined with an enabled parallactic-angle term. Both are
removed. The static feed rotation and the field rotation now compose:
`C_p P_p = M(basis) R(chi_p + alpha_p)`, where
`alpha_p = eta_p psi_p + nu_p el`. Ordinary alt-az has `alpha_p=psi_p`;
Nasmyth right/left retain the signed elevation term.

New, and both raised by `UnsupportedMountTypeError` during Jones resolution,
before any beam or sky is loaded:

```text
antenna 3 has mount_type=phased, which the parallactic-angle term does not
model; supported mounts are alt-az, equatorial, fixed, alt-az+nasmyth-l,
alt-az+nasmyth-r.
```

```text
antenna 3 has mount_type=alt-az, whose feeds rotate with the sky; enable
'jones.P' or the simulation would silently treat it as a fixed mount.
```

The second is the replacement for the removed blanket rejection, and it names
the fix rather than a tier. Conversely, `jones.P` on an array where no antenna's
feeds rotate — every `fixed`, every `equatorial`, or an instrument source that
carried no mount metadata — is rejected as an identity, like every other term
that cannot change the visibilities.

Only a pyuvdata dataset source carries mount metadata; a layout file has no
column for one, and an unspecified mount is the `fixed` case. For a layout-file
configuration the optional `P` contribution is therefore the identity and its
mount handling is unchanged. This does **not** preserve pre-SCI-006 polarized
linear values or `scientific_sha256`: the always-present receptor matrix `C`
changed independently to the east-X permutation.

### The canonical chain order changed: `P` moved sky-side of `C`

The canonical factorization is now

```text
J_p = H_p G_p B_p Rc_p Kd_p X_p D_p C_p E_p P_p T_p Z_p     (K applied separately)
```

Earlier releases placed `P` between `D` and `C`, following Tier 5's
factorization. That is wrong for a circular receptor: a field rotation acts on
the incoming field in the linear topocentric frame, so the physical composite is
`M(basis) R(chi_p + alpha_p) = C R(alpha_p)` and `R(alpha_p)` belongs
sky-side of `C`. Under the old order the `(R, L)` pair would be mixed by a real
2x2 rotation, when `S R(alpha_p) S^H = diag(e^-i alpha_p, e^+i alpha_p)` says
the correct effect is a pair of opposite phases.

This affects only runs with `jones.P` enabled, which could not exist before it
was implemented, so no stored result changes. It supersedes
`Tier5ReceptorFeedPlan.md` §19.1 for `P` and for `P` only; every other term
keeps its position.

### Removed Jones classes

Twenty-six exported Jones classes were removed before v1.0. Every one of them
returned the 2x2 identity for every input, so importing one and adding it to a
chain changed nothing and reported nothing. The table gives the replacement for
each; where the replacement is a field, it lands with the slice that implements
the owning term.

| Removed | Replacement |
| --- | --- |
| `GeometricPhaseJones` | `geometric_phase()`, a module-level function — K is per-baseline and was never a chain term |
| `TimeVariableGainJones` | a `time_model` field on `GainJones` |
| `ElevationGainJones` | an `elevation_curve` field on `GainJones` |
| `PolynomialBandpassJones` | `BandpassJones` with `model.kind: polynomial` |
| `SplineBandpassJones` | `BandpassJones` with `model.kind: spline` |
| `RFIFlaggedBandpassJones` | no replacement; flagging is a data-quality product, not a voltage-domain Jones factor |
| `IXRLeakageJones` | `PolarizationLeakageJones` with `d_terms.kind: ixr` |
| `MuellerLeakageJones` | `PolarizationLeakageJones`; a Mueller matrix is a derived 4x4 view of the same 2x2 Jones |
| `BeamSquintLeakageJones` | the beam subsystem; squint is a beam property, not a D-term |
| `FieldRotationJones` | `ParallacticAngleJones`, which is direction-dependent and subsumes it exactly; enable it with `jones.P` |
| `VLBIFeedRotationJones` | the per-antenna `mount_type` carried by the resolved instrument, which `ParallacticAngleJones` reads (`alt-az+nasmyth-l` and `alt-az+nasmyth-r` included) |
| `TurbulentIonosphereJones` | no replacement; stochastic screens are out of scope |
| `GPSIonosphereJones` | no replacement; RadioSim has no IONEX reader |
| `SaastamoinenTroposphereJones` | `TroposphereJones` with `zenith_delay.kind: saastamoinen` |
| `TurbulentTroposphereJones` | no replacement; stochastic screens are out of scope |
| `TroposphericOpacityJones` | the `opacity` sub-block of `TroposphereJones` |
| `FaradayRotationJones` | `IonosphereJones`, which owns ionospheric Faraday rotation; intrinsic source RM is already applied by the sky model |
| `DifferentialFaradayJones` | a per-antenna RM offset field on `IonosphereJones` |
| `WPhaseJones` | no replacement; the direct-sum RIME already carries `w(n-1)` exactly, so a W term would double-count |
| `WProjectionJones` | no replacement; w-projection is an imaging gridding kernel, not a forward-model factor |
| `WidefieldPolarimetricJones` | `ParallacticAngleJones` to leading order; the exact wide-field projection additionally needs a non-scalar beam |
| `ElementBeamJones` | no replacement; a station element beam belongs inside the beam system |
| `ArrayFactorJones` | no replacement; same |
| `DifferentialBeamJones` | per-antenna beam pointing offsets in the beam system, plus the existing per-antenna diameters and FITS beams |
| `FringeFitJones` | no replacement; fringe fitting is a calibration solution, and its forward-model content is `G` x `Kd` x a phase rate |
| `CrosshandPhaseJones` | renamed `CrosshandJones` |
| `CrosshandDelayJones` | `CrosshandJones`, which carries both the constant phase and the linear delay |
| `FrequencyDependentLeakageJones` | `PolarizationLeakageJones`, which is frequency-capable by construction |

The nineteen names that remain are exported by `radiosim.core.jones`. Each term
declares `term_status`, and every one of them now reads `"implemented"`: the
`"planned"` state existed only while Tier 7 was mid-flight, and no exported
class is in it. `radiosim.core.jones.faraday`, `radiosim.core.jones.wterm` and
`radiosim.core.jones.element_beam` no longer exist as modules.

Twenty-six exported Jones classes were removed in total, and the table above
names the replacement for each.

The solver and simulator `jones_config=` parameter was removed with them. It was
an untyped dictionary, hard-coded to `None` at the only production call site, and
every term it could enable was one of the identity stubs above. A typed `jones:`
configuration section replaces it.

### The Jones evaluation contract is batched

A term used to be asked for one direction at a time, or for all sources through
a second method with a different signature. Both are gone; there is one
keyword-only batched method per contract.

| Removed | Replacement |
| --- | --- |
| `JonesTerm.compute_jones(...)` | `JonesTerm.compute_jones_batch(...)`, keyword-only, returning `(n_dir, 2, 2)` for a direction-dependent term and `(1, 2, 2)` for a direction-independent one |
| `JonesTerm.compute_jones_all_sources(...)` | the same `compute_jones_batch`; the batch *is* all directions |
| `JonesChain.compute_antenna_jones(...)` | `JonesChain.compute_antenna_jones_batch(...)`, or the shared `evaluate_antenna_jones(...)` both solvers call |
| `JonesChain.compute_antenna_jones_all_sources(...)` | as above |
| `JonesChain.compute_baseline_visibility(...)` | no replacement; the solvers own baseline assembly |
| `JonesBaselineTerm.compute_baseline_term(...)` | `JonesBaselineTerm.compute_baseline_factor(...)`, batched |

Directions arrive as a `DirectionBatch` (`radiosim.core.jones.directions`),
which carries the horizontal and equatorial descriptions of one `(time,
frequency)` step. A subclass that implemented the old methods will fail to
instantiate rather than silently do nothing: both batched methods are
`@abstractmethod`.

## Sky-loader network declarations (2026-08-02)

### `LoaderDefinition.network_service` became `network_services`

A loader used to declare at most one network service, as
`network_service: str | None`. That was not expressible for a composite recipe:
`realistic_foreground` dispatches to a diffuse loader **and** a catalog loader,
reaches two services, and could therefore declare neither — so
`get_required_services()` returned nothing for the shipped
`configs/realistic_foreground_example.yaml` and the pre-flight printed
"Network: offline (no network-dependent models)" for a run that then made two
real network calls.

The field is now a tuple, and every loader that can reach the network declares
the union of what it reaches. There is no compatibility shim: the singular
spelling is gone from the package.

| Removed | Replacement |
| --- | --- |
| `LoaderDefinition.network_service: str \| None` | `LoaderDefinition.network_services: tuple[str, ...]` |
| `register_loader(..., network_service="vizier")` | `register_loader(..., network_services=("vizier",))` |
| `definition.metadata()["network_service"]` | `definition.metadata()["network_services"]`, a list |
| a loader declaring nothing | `network_services=()`, which is the default and means "purely local" |

Read the value with `get_required_services(sky_model_config)`, which returns
the set of services a configuration will actually reach. A custom loader that
imports a network client and declares no service is now a test failure, not a
silent one.

## The simulator registry gained a second entry (2026-08-22)

### `execution.simulator` now accepts `mmode` as well as `rime`

`SCI-004` phase M1 registers the m-mode full-sidereal harmonic forward model.
The standing invariant is unchanged and still exact — the accepted values of
`execution.simulator` are exactly the simulator registry keys — but that set is
now `{"rime", "mmode"}`. Nothing about a direct run changes: `rime` remains the
default, its arithmetic, component order, source reduction, result bytes and
fingerprints are unchanged, and an absent `execution.mmode` block never changes
a direct run.

| Was | Now |
| --- | --- |
| `execution.simulator: Literal["rime"]` | `execution.simulator: Literal["rime", "mmode"]` |
| `ExecutionConfig` had no m-mode block | `execution.mmode`, required with `mmode` and rejected with `rime` |
| `obs_time` was one untagged UTC interval | `obs_time` is a two-shape union; the untagged interval is unchanged and remains the only `rime` input |

### `VisibilitySimulator` grew a whole-`SkyModel` boundary

Before this change the registry was not a full-sky strategy boundary:
`calculate_visibilities` accepted `SourceArrays`, so the abstract interface
described only the point component; `Simulator.run()` always called
`core.hybrid.solve_sky` itself; and `solve_sky` dispatched point sources
through the registered object while calling `calculate_visibility_healpix`
directly. Registering a second algorithm against that interface would have been
false architecture, because HEALPix and hybrid runs would still have bypassed
it.

`VisibilitySimulator.solve(request)` is the replacement boundary. It takes one
immutable `SkySolveRequest` carrying the whole resolved `SkyModel` and returns
one `SkySolveOutcome`.

| Was | Now |
| --- | --- |
| `Simulator.run()` called `core.hybrid.solve_sky(...)` directly | `Simulator.run()` calls only `get_simulator(name).solve(request)` |
| a strategy implemented `calculate_visibilities(SourceArrays, ...)` only | a strategy implements `solve(SkySolveRequest)`; `RIMESimulator` still implements both, and its `solve` is a thin wrapper around the same `solve_sky` path |
| `HybridSolveOutcome.component_names` was read by the API | `SkySolveOutcome.components` |

`calculate_visibilities` is not removed: it is still the point-component kernel
`solve_sky` dispatches through, and `RIMESimulator` still implements it
unchanged. `MModeSimulator` deliberately does not: it raises rather than
quietly delegating a point-only call to a direct kernel.

### `RIMESimulator.supports_polarization` is a class attribute, not a property

`SCI-004` Section 9 makes capability truth phase-local and pins two facts in one
assertion: `MModeSimulator.supports_polarization is False` beside the unchanged
`RIMESimulator.supports_polarization is True`. Reading both from the classes
themselves is what makes the pair checkable, so the flag is now a plain class
attribute on both. Instance access is unaffected.
`RIMESimulator.supports_gpu` remains a property and remains `False`; no
end-to-end accelerator run of any RadioSim solver has been measured (register
row `PERF-001`).

### `SimulationResult.solver` is a tagged union

A direct run still carries the unchanged `SolverResultProvenance`, with the same
six fields and byte-identical serialization. An m-mode run carries
`MModeSolverResultProvenance` instead. Readers reconstruct and authenticate
whichever arm was written; a reader that silently relabels an m-mode record as
`rime` is a failure rather than a fallback.

### HEALPix sky coefficients under `mmode` are the pixel measure

The m-mode sky path reads a HEALPix payload as the **pixel measure**
`a_lm = sum_pix(s_pix * Omega_pix * conj(Y_lm(n_pix)))` over canonical-RING
pixel centres — the same measure the private direct oracle sums. A continuous
band-limited reinterpretation of the map, a ring-weighted quadrature, or any
iterated transform is a different sky object and is rejected. This affects only
the `mmode` path; the direct solvers' HEALPix handling is unchanged.

`SCI-004` remains `ROADMAP`. The design gate is accepted and the production
phases are separately gated; registering the strategy does not close the row,
and phase M1 makes no polarized, fingerprint, speed or accelerator claim.

## The m-mode solver became full Stokes (2026-08-24)

### `MModeSimulator.supports_polarization` is now `True`

Capability truth for this solver is phase-local, and phase M2 is the phase whose
acceptance changes it. `MModeSimulator.supports_polarization` is now `True`,
declared on the class itself, beside the unchanged
`RIMESimulator.supports_polarization is True`. Both values are still stated
together in one Tier 7 characterization assertion, which is what makes the pair
checkable.

`MModeSimulator.supports_gpu` is unchanged and remains `False`. A polarized
capability is not a speed claim: no end-to-end accelerator run of this solver
has been measured, and register row `PERF-001` governs every RadioSim
performance statement.

| Was | Now |
| --- | --- |
| `MModeSimulator.supports_polarization is False` | `MModeSimulator.supports_polarization is True` |
| a sky with non-zero `Q`, `U` or `V` was rejected | a full-Stokes sky is integrated |
| every m-mode snapshot carried `execution_path: "scalar"` | `execution_path` is `"scalar"` or `"polarized"`, from the resolved payload |
| `tangent_polarization_frame` was always the literal `not_applicable_scalar_m1` | a linearly polarized run carries the six-key `TangentPolarizationFrame` block |

### A polarized m-mode sky must declare its tangent-polarization frame

`stokes_q` and `stokes_u` are components in a tangent basis, not self-describing
numbers, so a polarized m-mode source now requires the six-key
`tangent_polarization_frame` block described in
[the sky-model guide](user_guide/sky_models.rst). A source that declares linear
polarization without it is rejected during config resolution with the
`mmode_polarization_frame` issue code, before backend allocation, output-path
creation or any harmonic work. An `I`/`V`-only payload may omit the block,
because `V` is a scalar with no tangent-basis dependence.

### The resolved receptor matrix now enters the m-mode kernel

The m-mode kernel is the same reference-phase response the direct RIME builds,
and it now composes each antenna's resolved `M_p = H_p C_p` into the sampled
beam response before forming the coherency, using the same
`radiosim.core.jones.receptor` code objects the direct chain uses.

**Stokes-`I` results are unchanged by this.** For a unitary receptor
`M P^I M^H = (1/2) M M^H = (1/2) I2`, so the factor cancels exactly; every
accepted phase-M1 scalar result, digest and gate value is preserved bit for bit.
It is load-bearing for polarized components only, where omitting it would report
the wrong correlation for any array that is not the default unrotated linear
one.

Two conventions are worth stating explicitly because they are easy to conflate:

- The RadioSim-to-Shaw basis bridge is `D = diag(-1, 1)`; the kernel uses
  `D P^X D` and the sky uses the matching `U_H = -U`. `D` is **not** the SCI-006
  east-X permutation and does not replace it — that permutation is antidiagonal
  and stays inside the antenna Jones matrix. There is no additional fitted or
  configurable `V` flip.
- The constant chain terms act in the **celestial** tangent basis of each
  direction, which is the basis the spin expansions use. Every mount-dependent
  tangent rotation belongs to the `P` term, which is exactly the identity for
  the shipped `fixed` and unspecified mounts. A ground-anchored,
  direction-dependent response would need a measured tangent transport; that is
  not in this scope.

### Nothing about a direct run changes

`rime` remains the default. Its arithmetic, component order, source reduction,
result bytes and fingerprints are unchanged, and `SCI-004` remains `ROADMAP`:
the production phases are separately gated and no phase of this work closes the
row or adds a fingerprint, speed or accelerator claim.


## Typed polarization materialization evidence

`radiosim.core.sky.containers` exposes `PolarizationOperation`,
`PolarizationMaterialization` and `PolarizationMaterializationEvidence`.
The private canonical native identity factory now returns the named evidence
object in place of `_NativeIdentityReceipt`. Its new `brightness_conversion`
field holds the exact enclosing `BrightnessConversion` enum. Consumers verify
that context against the actual owner; a plain string is not an enum substitute.

The twelve scientific record fields, four operation fields and serialized
preimages are unchanged. The context is an additional evidence field, not a
new serialized record field. The former private class names have no aliases.
Constructing a record or evidence object does not prove a conversion occurred;
the actual-value consumer still validates it. This type extraction does not
attach evidence to `HealpixData`, normalize a loader output or enable new
m-mode inputs.


### Native component materialization attachment groundwork

`HealpixData` can carry `tangent_polarization_frame` and
`polarization_materialization` as typed, jointly validated component fields.
Their defaults remain null: existing raw loader models acquire no canonical
convention claim. Frame-only attachment and mapping-to-evidence coercion are
refused. A retained identity is recomputed against actual stored values after
constructor normalization, and `SkyModel` joins its final conversion context.

On attached owners, `replace()` verifies the old evidence before rebuilding and
preserves that exact evidence object. It refuses dropping/reissuing evidence or
changing bound values, frame, grid, channel metadata or storage precision.
Actual unchanged replacements preserve the complete record and sidecars.
Until separately implemented child operations exist, transformations that alter
an attached payload refuse at this boundary. Unbound transformations retain their
existing behavior. The explicit canonical-only completion helper remains private
and binds resolved stored values; earlier constructor conversion history is not
claimed. This groundwork does not normalize loader output, propagate records
through every public operation or admit native public m-mode simulations.

For every attached-owner construction, completion, validation or replacement,
the caller must exclude mutation/rebinding through all payload, frequency-axis
and pixel-ID aliases for the entire interval: old-record validation,
normalization, all scans/hashes and publication. Frozen fields, read-only flags
and `model.replace()` may share storage; they establish neither exclusive
ownership nor a coherent concurrent snapshot. Later sequential stale-alias
checks are separate and do not prove safety against concurrent writers.
