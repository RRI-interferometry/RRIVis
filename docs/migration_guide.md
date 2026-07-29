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
assert result.correlations == ("XX", "XY", "YX", "YY")
stokes_i = result.stokes_i()
```

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
| top-level `feeds` | no replacement; receptor/feed physics is not implemented |

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
