# Instrument source examples

RadioSim resolves one instrument source into canonical antenna identities,
positions, diameters, and location. A configuration cannot combine identities
or positions from different sources.

## Retained layout-file formats

Set `instrument.source.kind` to `layout_file` and choose exactly one format:

- `radiosim`: native ENU text, demonstrated by
  `example_radiosim_format.txt` and `hera_5.txt`.
- `casa_loc`: CASA local-coordinate text, demonstrated by
  `example_casa_loc.cfg`.
- `measurement_set`: a Measurement Set directory.
- `uvfits`: a UVFITS file used as an instrument source, not an output format.
- `mwa_metafits`: an MWA metafits file.

The `radiosim`, `casa_loc`, and `mwa_metafits` formats require an explicit
`telescope_name` and `instrument.location`. Measurement Set and UVFITS sources
can supply identity and location; explicit values take precedence where the
schema permits them.

The native RadioSim rows are:

```text
Name Number BeamID E N U Diameter
HH136 136 0 0.0 0.0 0.0 14.0
```

Matching `diameter_overrides` take precedence, followed by a positive source
diameter, then `default_diameter_m` for antennas that still lack a diameter.
Resolution fails if any antenna remains without a positive diameter.
`BeamID` is retained source metadata; it does not enable per-antenna beam
physics.

```yaml
instrument:
  source:
    kind: layout_file
    path: example_radiosim_format.txt
    format: radiosim
    telescope_name: Example Array
  location:
    longitude_deg: 21.4283
    latitude_deg: -30.7215
    height_m: 1050.0
  default_diameter_m: 14.0
  diameter_overrides: []
```

## Known telescope source

`known_telescope` is a source kind, not a text-file format. Its
`registry_policy` is either `offline` or `allow_network`. Global offline mode
always wins, so an `allow_network` source is rejected when
`execution.offline` is true.

```yaml
instrument:
  source:
    kind: known_telescope
    name: HERA
    registry_policy: offline
```

## Complete configuration

`example_telescope_config.yaml` is a complete strict RadioSim document. Its
relative source path is resolved from this directory by `Simulator.from_yaml`
and `load_config`.
