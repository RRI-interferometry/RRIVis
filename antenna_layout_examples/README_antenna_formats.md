# Antenna layout examples

RadioSim accepts an antenna-layout path and an explicit
`antenna_file_format`. The strict high-level configuration also requires one
positive `all_antenna_diameter`. The current setup applies that diameter to
every antenna; per-antenna diameter and beam assignments are not active
high-level behavior.

## Native RadioSim format

`example_radiosim_format.txt` and `hera_5.txt` use ENU coordinates in meters:

```text
Name Number BeamID E N U Diameter
HH136 136 0 0.0 0.0 0.0 14.0
```

The current high-level path reads name, number, position, and layout metadata,
then applies the configured uniform diameter. `BeamID` and a per-row diameter
must not be interpreted as enabled per-antenna high-level behavior.

```yaml
antenna_layout:
  antenna_positions_file: example_radiosim_format.txt
  antenna_file_format: radiosim
  all_antenna_diameter: 14.0
```

## CASA configuration format

`example_casa_format.cfg` demonstrates a CASA-style layout. Select it with
`antenna_file_format: casa`. Coordinate metadata and diameter columns are read
according to the CASA reader, but the current high-level uniform-diameter rule
still applies during Simulator setup.

## Simple pyuvdata-style text

`example_pyuvdata_format.txt` contains three position values per row. Select it
with `antenna_file_format: pyuvdata`. This layout input is distinct from the
deferred `telescope.use_pyuvdata_*` configuration flags, which the Tier 1
resolver rejects.

## Measurement Set, UVFITS, and MWA readers

The accepted `antenna_file_format` values are `radiosim`, `casa`,
`measurement_set`, `uvfits`, `mwa`, and `pyuvdata`. They describe antenna
layout input readers. An `uvfits` antenna-layout reader does not make UVFITS a
supported result format; `workflow.result_format: uvfits` is rejected.

Measurement Set layout input expects a directory. Other current layout formats
expect a regular file. All input paths are normalized and checked during
configuration resolution.

## Public Python reader

```python
from radiosim.io import read_antenna_positions

antennas = read_antenna_positions(
    "antenna_layout_examples/example_radiosim_format.txt",
    format_type="radiosim",
)
```

## Complete configuration example

`example_telescope_config.yaml` is a complete strict RadioSim document. It is
not a pyuvsim telescope metadata file. Its relative antenna path is resolved
from this directory when loaded with `Simulator.from_yaml` or `load_config`.

## Coordinate notes

- Native RadioSim positions are local East-North-Up values in meters.
- Preserve enough precision for the intended phase accuracy.
- Use one coordinate frame consistently within a file.
- The observatory latitude, longitude, and height belong in the required
  top-level `location` section, not inside the layout document.
