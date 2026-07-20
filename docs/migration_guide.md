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
    start_time="2025-01-01T00:00:00",
    sky_model=sky_model,
)
```

The direct `Simulator(resolved_runtime)` constructor accepts only
`ResolvedSimulationConfig`. Passing a YAML path to `from_config`, or passing
raw mappings/backend scalars to the direct constructor, is no longer valid.

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

## Removed legacy beam keys

The following BeamManager compatibility fields are rejected:
`use_beam_file`, `use_different_beams`, `beam_file_path`, `beam_files`,
`beams_per_antenna`, `default_beam_id`, `beam_freq_interp`,
`beam_freq_buffer_mhz`, `all_beam_response`, and `beam_assignment`. Use only
the strict analytic `beams` fields. FITS/mixed/per-antenna beam execution is
not implemented.

## Frequency and configuration I/O

Frequency input requires `mode: grid` or `mode: explicit`; the explicit field
is `channel_frequencies_hz`. `load_config()` returns a resolved bundle,
`resolve_config()` accepts a mapping/model plus source context, and
`dump_config()` accepts only a `RadioSimConfig` input model. Removed custom
model methods should be replaced by those functions or standard `model_dump`.

Paths are source-aware: YAML paths use the document parent, mapping/model
relative paths require `base_dir`, and override paths use the captured call
directory. `~` is expanded; environment-variable syntax is rejected.
