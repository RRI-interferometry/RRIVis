# Tier 1 configuration and Simulator migration

RadioSim is pre-v1.0. Tier 1 deliberately replaces the divergent legacy
constructors and configuration sections with one strict, source-aware resolver.
The old syntax is not kept through aliases or compatibility shims.

## Public construction API

Use the constructor that matches the source you actually have:

```python
from radiosim import Simulator

yaml_simulator = Simulator.from_yaml("config.yaml")
model_simulator = Simulator.from_config(input_model, base_dir=project_dir)
mapping_simulator = Simulator.from_mapping(mapping, base_dir=project_dir)
parameter_simulator = Simulator.from_parameters(
    antenna_layout=antenna_path,
    antenna_file_format="radiosim",
    antenna_diameter_m=14.0,
    channel_frequencies_hz=(100_000_000.0, 101_500_000.0),
    location={"lat": -30.72, "lon": 21.43, "height": 1073.0},
    start_time="2025-01-01T00:00:00",
    sky_model={"sources": [{"kind": "test_sources"}]},
)

direct_simulator = Simulator(resolved_runtime)
```

The direct constructor accepts only `ResolvedSimulationConfig`.

### Before migration: removed multi-purpose constructor

The following is historical syntax and no longer runs:

```python
# Before migration — removed
simulator = Simulator(
    config=raw_mapping,
    backend="jax",
    precision="fast",
)
```

Replace it with `from_mapping(..., overrides=...)` or one of the other disjoint
constructors.

### Before migration: YAML passed to `from_config`

This old path-taking form was removed:

```python
# Before migration — removed
simulator = Simulator.from_config("config.yaml")
```

Use `Simulator.from_yaml("config.yaml")`. `from_config` now means a typed
`RadioSimConfig` input model.

## Configuration I/O

The public configuration functions have distinct boundaries:

```python
from radiosim.io import dump_config, load_config, resolve_config

bundle = load_config("config.yaml")
runtime = bundle.runtime
workflow = bundle.workflow
provenance = bundle.provenance

dump_config(input_model, "copied-config.yaml")
```

- `load_config()` returns `ResolvedConfiguration`, not the input Pydantic model.
- `resolve_config()` resolves a mapping/model with an explicit
  `ConfigurationSource`.
- `dump_config()` accepts only `RadioSimConfig` and writes the user-facing
  document.

The removed `RadioSimConfig.from_yaml()`, `to_yaml()`, `to_dict()`, and manual
`validate()` methods have no aliases. Use `load_config`, `dump_config`, standard
`model_dump`, and `resolve_config` respectively.

## Required scientific sections

A strict document requires:

- `antenna_layout`;
- `location`;
- `sky_model`;
- `obs_time`; and
- `obs_frequency`.

Hidden HERA location/current-time defaults and partial production documents are
gone. Tests should use complete test builders rather than weakening production
validation.

## Frequency migration

Frequency input is a discriminated union.

### Before migration: implicit grid

```yaml
# Before migration — removed
obs_frequency:
  starting_frequency: 100.0
  frequency_interval: 1.0
  frequency_bandwidth: 10.0
  frequency_unit: MHz
```

Add the discriminator:

```yaml
obs_frequency:
  mode: grid
  starting_frequency: 100.0
  frequency_interval: 1.0
  frequency_bandwidth: 10.0
  frequency_unit: MHz
```

The bandwidth must contain an integral number of intervals. RadioSim preserves
the requested spacing with `start + index * interval`; it does not silently
adjust spacing through `linspace`.

### Before migration: raw frequency escape hatch

```yaml
# Before migration — removed
obs_frequency:
  frequencies_hz: [100000000.0, 101500000.0]
```

Use the typed explicit variant:

```yaml
obs_frequency:
  mode: explicit
  channel_frequencies_hz: [100000000.0, 101500000.0]
```

Explicit values are always Hz, nonempty, finite, positive, and strictly
increasing. One channel and nonuniform spacing are supported. Values are never
sorted or reconstructed.

## `compute`, `precision`, `simulators`, and `output`

The old top-level sections were removed.

### Before migration

```yaml
# Before migration — removed
compute:
  backend: numpy
  offline: true
precision:
  preset: fast
simulators:
  name: rime
output:
  simulation_data_dir: output
  output_file_name: visibilities
  output_file_format: HDF5
  save_simulation_data: true
```

### Current structure

```yaml
execution:
  backend: numpy
  precision:
    preset: fast
  simulator: rime
  offline: true

workflow:
  output_dir: output
  result_filename: visibilities
  result_format: hdf5
  save_results: true
```

`execution` belongs to scientific runtime policy. `workflow` belongs to CLI
post-run orchestration and never enters `ResolvedSimulationConfig`.

`Simulator.from_yaml()` resolves the scientific runtime but does not perform
workflow saving, logging, plotting, prompting, skipping, or browser opening.

## Strict unknown-field rejection

All concrete input models forbid unknown fields. Removed keys such as
`fixed_HPBW`, old BeamManager fields, raw phase-center fields, old top-level
sections, and legacy nested sky sections fail with migration guidance. Loader
extensions use an explicit, registry-validated `options` mapping; arbitrary
Pydantic extras are not an extension mechanism.

## Paths and base directories

Path meaning now depends on the source, not ambient mutation order:

- YAML-relative paths are based at the YAML file's parent.
- Mapping/model inputs with relative paths require `base_dir`.
- Parameter and explicit override paths use the captured invocation directory.
- `~` is expanded; `$VARIABLE` syntax is rejected.
- Input paths are normalized and checked before backend initialization.
- Validation does not create output directories.
- Globs are expanded deterministically and sorted.

If a mapping or typed model contains only absolute scientific paths, its
workflow output path must also be absolute to omit `base_dir`.

## Overrides, backend, and precision

Precedence is:

`explicit override > document value > declared default`

```python
from radiosim import Simulator
from radiosim.io.config import PrecisionInput
from radiosim.io.config_resolution import SimulationOverrides

simulator = Simulator.from_yaml(
    "config.yaml",
    overrides=SimulationOverrides(
        backend="auto",
        precision=PrecisionInput(preset="fast"),
        offline=False,
    ),
)
```

`None` means no override. `auto` is a real backend strategy. Frequency and
location overrides replace complete values. Precision overrides replace the
complete precision tree; they do not deep merge. Explicit unsupported
backend/precision combinations fail during resolution, and unavailable
explicit backends fail instead of silently falling back.

Backend selection is not proof of end-to-end GPU execution. The high-level
path still contains host-side work and incomplete backend coverage, so no
unverified speedup multiplier is part of this migration.

## Input model versus resolved runtime

`RadioSimConfig` represents user-authored values and is the object serialized by
`dump_config`. `ResolvedSimulationConfig` contains final absolute paths, exact
Hz samples, frozen precision, and runtime defaults. `ResolvedConfiguration`
wraps the runtime, CLI workflow, and provenance.

Do not serialize a resolved runtime as if it were a reusable input document.
Use the versioned provenance snapshot for result metadata and `dump_config` for
input documents.

## Deliberately rejected later-tier fields

Tier 1 keeps later work honest by rejecting it before runtime side effects:

- FITS, mixed, and per-antenna beams;
- per-antenna diameter maps;
- baseline-subset selection;
- non-default top-level receptor/feed fields;
- pyuvdata telescope flags;
- UVFITS workflow output;
- explicit visibility-worker control; and
- spherical-harmonic simulator modes.

Do not add local compatibility translations to make these fields appear to
work. Their owning tier must add end-to-end implementation and tests before
removing the corresponding rejection rule.

## CLI migration

Config mode uses the same loader and workflow separation:

```bash
radiosim validate config.yaml
radiosim --config config.yaml
radiosim --config config.yaml --backend auto
radiosim --config config.yaml --offline
radiosim --config config.yaml --online
```

The root backend option defaults to no override. The offline/online pair also
defaults to no override. `radiosim validate` resolves and checks input paths but
does not construct a backend, load scientific inputs, create output, plot, or
open a browser.
