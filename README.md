# RadioSim

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/RRI-interferometry/RadioSim/actions/workflows/ci.yml/badge.svg)](https://github.com/RRI-interferometry/RadioSim/actions)

RadioSim is a Python package for radio-interferometric visibility simulation.
It provides a strict configuration system, a high-level `Simulator` API,
point-source and HEALPix direct-sum paths, analytic primary beams, sky-loader
integration, and an immutable canonical in-memory result.

## Current high-level scope

The current `Simulator` path supports:

- strict YAML, mapping, and typed-model configuration;
- one typed instrument source with canonical identity, location, positions,
  per-antenna diameters, and deterministic provenance;
- exact uniform grids or ordered explicit frequency samples in Hz;
- local synthetic sources, catalog loaders, diffuse models, and supported
  file-backed sky loaders;
- typed correlation, length, and axial-azimuth baseline selection;
- analytic, shared-FITS, per-antenna-FITS, and mixed beam configuration through
  one canonical per-antenna beam system;
- linear or circular two-feed receptors with a static feed rotation and one
  resolved array-wide output polarization basis;
- point-source, HEALPix, or `hybrid` (summed point + HEALPix) direct-sum
  simulation;
- a typed `jones:` section carrying gains (G), bandpass (B), cable reflection
  (Rc), instrumental delay (Kd), cross-hand phase and delay (X), polarization
  leakage (D), parallactic rotation (P), troposphere (T), ionosphere (Z), and
  the two baseline-dependent effects — closure error (M) and time/bandwidth
  smearing (Q);
- separately configurable sky-loader and solver worker policies;
- requested NumPy, JAX, Dask, or `auto` backend selection through one
  resolver; and
- `Simulator.plot_observability()` as a visualization helper.

The strict schema still rejects behavior that is not connected: elliptical or
non-orthogonal feed pairs, single-feed and multi-feed antennas, a
frequency- or time-dependent receptor basis, arbitrary polarized BeamFITS
variants, and spherical-harmonic or m-mode simulator modes. Heterogeneous
beams are active in both visibility paths; observability requires an explicit
canonical reference antenna unless all assigned handlers are scientifically
equivalent.

Every Jones term RadioSim exports implements real physics and declares
`term_status: "implemented"`. The geometric phase (K), the canonical
primary beam (E), the receptor configuration (C), and the output basis
transform (H) are always applied; the nine chain terms above are applied when
`jones:` configures them, and M and Q apply by Hadamard product outside the
chain. No exported term is an unconditional identity stub. Parameter-dependent
identity cases remain where the physical transform is genuinely neutral, while
a configurable `jones:` block whose resolved effect is entirely the identity is
rejected rather than accepted.

## Install

```bash
pip install radiosim
```

For development, use the checked-in Pixi environment:

```bash
git clone https://github.com/RRI-interferometry/RadioSim.git
cd RadioSim
pixi install
pixi run test
```

Optional extras install backend or I/O dependencies. They are named for the
libraries they install, not for devices: the `gpu`, `gpu-cuda`, `gpu-rocm` and
`tpu` extras were removed because RadioSim has measured no accelerator, and an
installable extra called `gpu-cuda` claimed one.

```bash
pip install radiosim[jax]       # optional JAX stack (CPU; see the note below)
pip install radiosim[dask]      # optional NumPy/Dask backend stack
pip install radiosim[ms]        # optional Measurement Set support
```

## Quick start

Validate and run the offline sample:

```bash
pixi run radiosim validate configs/config.yaml
pixi run radiosim --config configs/config.yaml
```

From Python, a YAML path is passed to `from_yaml`:

```python
from radiosim import Simulator

simulator = Simulator.from_yaml("configs/config.yaml")
result = simulator.run(progress=True)
print(result.visibilities.shape)  # time, baseline, frequency, correlation
assert result is simulator.result
```

`Simulator.from_yaml()` resolves scientific runtime state but does not execute
the document's CLI-only `workflow` actions. Config mode owns saving, logging,
plotting, prompting, skipping, and browser behavior.

`Simulator.run()` returns an immutable `SimulationResult`. Its canonical
visibility shape is `(time, baseline, frequency, correlation)`. The correlation
axis is the row-major flattening of the 2x2 visibility matrix, so its labels
follow the resolved polarization basis: `XX, XY, YX, YY` for `linear_xy` (the
default) and `RR, RL, LR, LL` for `circular_rl`. Read `result.correlations` and
`result.polarization_basis` rather than assuming the linear labels. Coordinates
and derived Stokes I are
available through `result.time_grid`, `result.frequencies_hz`,
`result.channel_widths_hz`, `result.correlations`, and `result.stokes_i()`;
`result.flags`, `result.weights`, `result.receptors`,
`result.scientific_sha256`, and
`result.provenance_sha256` complete the current result surface.

Save the last successful result to one exact final path with the typed
`ResultFormat` enum. HDF5 is complete and reconstructable, summary JSON
(`SUMMARY_JSON`) is bounded metadata only, and MS/UVFITS are standard-format
projections. Python
and direct `simulate` calls never prompt or generate suffixes.

`Simulator.plot(plot_type=..., output_dir=..., backend="bokeh", show=...,
overwrite=..., visibility_phase_unit=...)` renders the published
`SimulationResult` into one explicit directory. The renderers read the
canonical coordinate arrays directly — MJD time centers from
`result.time_grid`, channel centers from `result.frequencies_hz`, and the
published baseline order — and derive Stokes I explicitly as the sum of the two
parallel hands, labelled `XX + YY` or `RR + LL` for the result's own basis. Phase
is displayed in `radians` (default) or `degrees`. Browsers open only after
every declared file is written.

For a small programmatic run:

```python
from radiosim import Simulator
from radiosim.io.config import ExecutionConfig, PrecisionInput
from radiosim.io.instrument_config import (
    BaselineSelectionConfig,
    InstrumentConfig,
    InstrumentLocationConfig,
    LayoutFileSourceConfig,
)

instrument = InstrumentConfig(
    source=LayoutFileSourceConfig(
        path="antenna_layout_examples/hera_5.txt",
        format="radiosim",
        telescope_name="HERA",
    ),
    location=InstrumentLocationConfig(
        longitude_deg=21.4283,
        latitude_deg=-30.72152,
        height_m=1073.0,
    ),
    default_diameter_m=14.0,
)

simulator = Simulator.from_parameters(
    instrument=instrument,
    baseline_selection=BaselineSelectionConfig(correlations="all"),
    channel_frequencies_hz=(100_000_000.0, 101_500_000.0, 108_000_000.0),
    channel_widths_hz=(1_000_000.0, 1_000_000.0, 1_000_000.0),
    start_time="2025-01-01T00:00:00",
    duration_seconds=1.0,
    time_step_seconds=1.0,
    sky_model={"sources": [{"kind": "test_sources", "num_sources": 3}]},
    execution=ExecutionConfig(
        backend="numpy",
        precision=PrecisionInput(preset="standard"),
        offline=True,
    ),
)
result = simulator.run(progress=False)
assert result is simulator.result
print(result.visibilities.shape)
print(result.stokes_i().shape)
```

The other public construction paths are:

```python
Simulator(resolved_runtime)
Simulator.from_config(input_model, base_dir=project_dir)
Simulator.from_mapping(mapping, base_dir=project_dir)
```

The direct constructor accepts only `ResolvedSimulationConfig`. Mapping and
typed-model inputs with relative paths require `base_dir`.

## Strict YAML contract

This is a complete small document:

```yaml
instrument:
  source:
    kind: layout_file
    path: ../antenna_layout_examples/hera_5.txt
    format: radiosim
    telescope_name: HERA
  location:
    longitude_deg: 21.4283
    latitude_deg: -30.72152
    height_m: 1073.0
  default_diameter_m: 14.0
  diameter_overrides: []

baseline_selection:
  correlations: all
  length_filter: null
  azimuth_ranges_deg: []

obs_time:
  start_time: "2025-01-01T00:00:00"
  duration_seconds: 1.0
  time_step_seconds: 1.0

obs_frequency:
  mode: explicit
  channel_frequencies_hz: [100000000.0, 101500000.0, 108000000.0]
  channel_widths_hz: [1000000.0, 1000000.0, 1000000.0]

sky_model:
  sources:
    - kind: test_sources
      num_sources: 3

beams:
  mode: analytic
  model:
    kind: circular_aperture
    taper:
      kind: gaussian
      edge_taper_db: 10.0

execution:
  backend: numpy
  precision:
    preset: standard
  simulator: rime
  offline: true

workflow:
  output_dir: output
  result_filename: visibilities
  result_format: hdf5
  collision_policy: error
  save_results: false
  plot_results: false
  open_plots_in_browser: false
  plotting_backend: bokeh
  visibility_phase_unit: radians
  save_log: false
```

`visibility_phase_unit` is exactly `radians` or `degrees` and controls only the
displayed phase axis; canonical stored values stay in radians. Removed workflow
fields are rejected with exact migration text; see the migration guide.

For a uniform frequency grid, use `mode: grid` with positive
`starting_frequency`, `frequency_interval`, `frequency_bandwidth`, a positive
`channel_width`, and one of `Hz`, `kHz`, `MHz`, or `GHz`. The bandwidth must be
an integral number of
intervals. RadioSim constructs `start + index * interval`; it does not alter
the requested spacing with `linspace`.

Unknown fields are rejected. The pre-v1 API intentionally does not translate
removed input shapes; see the migration guide for exact replacements.

The beam schema accepts four complete modes: `analytic`, `shared_fits`,
`per_antenna_fits`, and `mixed`. `Simulator` resolves every assignment to
canonical antenna identity, validates and loads the complete beam system before
device, backend, network, or sky work, and uses that same system in both
visibility solvers and observability planning. FITS failures never fall back to
an analytic beam: there is no analytic fallback for a FITS declaration.
FITS support is the documented scalar E-Jones subset plus the accepted
`uvbeam_peak_common_v1` full-efield subset (`SCI-005` Stage 3, accepted
2026-08-20), not arbitrary full-polarization BeamFITS.

The `receptors` section is a separate concern from `beams`: `beams` describes how
each aperture is *illuminated*, `receptors` describes the *receiving* feeds and
the basis results are reported in. The section is optional, and omitting it is
exactly equivalent to the explicit default:

```yaml
receptors:
  default:
    basis: linear            # or circular
    feed_rotation_deg: 0.0
  overrides: []              # per-antenna basis or rotation
  output_basis: auto         # or linear / circular
```

`basis: circular` makes every antenna natively R/L, and `output_basis: auto`
then resolves to `circular_rl`. Name `linear` or `circular` explicitly to report
a mixed array in one basis; `auto` on a mixed array is rejected with both antenna
counts. `feed_rotation_deg` is a finite static offset from the nominal
orientation of the selected basis. Single-feed and multi-feed antennas,
elliptical or non-orthogonal feed pairs, per-feed angles, and a frequency- or
time-dependent basis are rejected. Mount orientation is handled by `jones.P`:
`fixed` and `equatorial` are neutral, while `alt-az` and the two supported
Nasmyth variants require the term; unknown mount types are rejected. See
[`configs/receptor_circular_example.yaml`](configs/receptor_circular_example.yaml)
for a runnable circular sample and
[the Jones guide](docs/user_guide/jones_matrices.rst) for the receptor
mathematics and the boundaries.

HEALPix advice is derived from the smallest selected-baseline beam-product
feature scale over every exact observation frequency. For endpoint voltage
scales `s_p` and `s_q`, the product scale is
`1 / (1 / s_p + 1 / s_q)` and the pixel limit is that scale divided by the
fixed safety factor five. Advice is logging-only: RadioSim never changes a
requested or loaded NSIDE. A FITS feature scale is a conservative native-grid
representation bound, not a measurement of physical beam bandwidth or FWHM.

## Loading, resolving, and serialization

```python
from radiosim.io import dump_config, load_config, resolve_config

bundle = load_config("configs/config.yaml")
runtime = bundle.runtime
workflow = bundle.workflow
provenance = bundle.provenance

# dump_config accepts the strict user-input model, not a resolved bundle.
dump_config(input_model, "copied-config.yaml")
```

`load_config()` returns a `ResolvedConfiguration` bundle. `dump_config()`
serializes a `RadioSimConfig` input model. `resolve_config()` is the mapping or
typed-model boundary when an explicit `ConfigurationSource` is needed.

Path rules are deterministic:

- YAML-relative paths use the YAML file's parent.
- Mapping/model relative paths require an explicit `base_dir`.
- call-site override paths use the captured invocation directory;
- input paths are normalized and checked before backend initialization;
- validation never creates output directories; and
- glob matches are normalized and sorted.

Override precedence is `explicit override > document value > declared
default`. `None` means no override. A precision override replaces the complete
precision tree. Unsupported backend/precision combinations fail explicitly;
there is no promised silent fallback.

## Backends and performance

NumPy is the deterministic default. `auto` is a real selection strategy, not
a synonym for “keep the document value”: it asks only NumPy to honor the
requested precision and never imports or probes JAX. It never selects Dask.
The selectable names are `numpy`, `jax`, `dask`, and `auto`.
`numba` is removed because the backend behind that name never compiled a
kernel.

Direct `get_backend("jax")` uses JAX's runtime-default device. A named JAX
`cpu`, `gpu`, or `tpu` device is a strict requirement, as are the direct `gpu`
and `tpu` aliases: an unavailable device raises `BackendNotAvailableError`
without a CPU fallback. `list_backends()` and `get_backend_info()` are the
explicit discovery operations and may initialize JAX. These rules implement
the open [`PERF-001`](PostTier8RemediationPlan.md) runtime mitigations; they do
not establish accelerator performance.

Both solvers route their Jones chain, geometric phase, coherency construction,
contraction, and accumulation through the selected backend. Exactly one kernel
is compiled — the baseline-batched per-`(time, frequency)` contraction, under
`backend: jax` only. Astropy coordinate transforms, horizon masking, Planck
brightness conversion, FITS beam interpolation, and the per-time HEALPix
direction cosines run on the host by design; they are listed, with reasons, in
the [backend guide](docs/user_guide/backends.rst).

Measured on an Apple M1 Max (macOS 26.5.2, arm64), `numpy 2.3.2`,
`jax 0.10.2` (CPU-only), `dask 2025.7.0`, commit `ea48d2c`, records committed
at [`output/benchmarks/reference/`](output/benchmarks/reference/):

- Dask is bit-identical to NumPy on all eight benchmarked workloads.
- JAX-CPU agrees with NumPy within `|dV| <= atol + 1e-12*|V|`; worst observed
  absolute deviation `1.7e-11` against an allowed `5.2e-9`.
- JAX-CPU is **slower** than NumPy on every workload measured here — about 3x on
  a 4096-source run and 10-20x on the small parity workloads — plus 0.002-0.80 s
  of first-call compilation. The host-side time and frequency loops give XLA
  nothing to amortize against yet.
- No accelerator was exercised. Every record states `accelerator: "none"` and
  lists `gpu`, `tpu`, and `distributed` as unmeasured.

This repository publishes no unverified speedup multiplier and no GPU
performance number. Reproduce the records with `pixi run bench`; each carries
hardware, accelerator, backend and version, the full precision tree, problem
dimensions, setup versus steady-state timing, compilation time, host-transfer
time, peak memory, and the correctness tolerance against NumPy.

## Output and observability

`Simulator.save(path, format=ResultFormat.HDF5)` uses the exact final artifact
path. Missing canonical extensions are appended; conflicting extensions and
string format arguments are rejected. Config mode stages the config artifact,
optional log, and selected result together, verifies a strict ownership
manifest, and publishes one run directory atomically under the selected
collision policy. The old run survives every pre-publication failure. When
`plot_results` is enabled the renderers write into that same staged directory
with browser presentation disabled, the manifest records every rendered file,
and `open_plots_in_browser` opens the published paths last. A browser failure
is reported separately and never unpublishes the run.

`Simulator.plot_observability()` is a helper associated with the Simulator and
uses the same loaded beam system. It selects the minimum-number antenna only
when all assigned handlers are scientifically equivalent; heterogeneous beam
assignments require an explicit canonical reference antenna. It is not a
separate simulation engine or backend.

## Examples and documentation

- [Configuration guide](docs/user_guide/configuration.rst)
- [Instrument resolution](docs/user_guide/instrument_resolution.rst)
- [Migration guide](docs/migration_guide.md)
- [Python example](examples/scripts/simple_simulation.py)
- [Basic notebook](examples/notebooks/01_basic_usage.ipynb)
- Four shipped YAML samples in [`configs/`](configs/):
  [`configs/config.yaml`](configs/config.yaml) (the offline smoke sample),
  [`configs/hybrid_sky_example.yaml`](configs/hybrid_sky_example.yaml),
  [`configs/realistic_foreground_example.yaml`](configs/realistic_foreground_example.yaml)
  (needs network access at simulation time), and
  [`configs/receptor_circular_example.yaml`](configs/receptor_circular_example.yaml)
- [Antenna layout formats](antenna_layout_examples/README_antenna_formats.md)

## Repository layout

| Path | What it holds |
|---|---|
| `src/radiosim/` | the package: `core/` physics, `backends/`, `io/`, `api/`, `cli/`, `simulator/`, `benchmarks/`, `visualization/`, `utils/` |
| `tests/` | `unit/`, `integration/`, `characterization/` (golden fingerprint pins), `performance/` (benchmark records, never gating), `crossvalidation/` (the optional `crossval` environment), plus the shared `fixtures/` and `support/` helper packages |
| `configs/`, `antenna_layout_examples/` | shipped sample configurations and antenna layout formats |
| `examples/`, `docs/` | the runnable script and notebook, and the Sphinx site |
| `output/benchmarks/reference/` | the committed backend benchmark records every performance sentence on this page cites |
| `output/crossvalidation/` | the committed `pyuvsim 1.4.0` comparison record |
| `simulators/` | 41 third-party simulator checkouts, as git submodules, for reference reading only |

`simulators/` is not part of the package: it is excluded from the wheel and
from Ruff, and a plain `git clone` does not fetch it. The submodules arrive
only with `--recursive` or an explicit `git submodule update --init`, and cost
roughly 3.9 GB checked out.

## Tests

```bash
pixi run test          # the full suite (pytest-xdist; add `-- -n 0` to serialize)
pixi run doctest       # docstring doctests, scoped to src/radiosim
pixi run lint
pixi run check-format
pixi run typecheck
make -C docs clean html
```

`pixi run test` collects `tests/` in full, including the `integration/` suite
(the hybrid, Jones, and CLI-to-artifact end-to-end runs) and the
`characterization/` fingerprint pins, which gate. Two suites are marked `slow`
and are therefore outside the default gate:

```bash
pixi run bench                                       # tests/performance/, writes a benchmark record
pixi run --environment crossval test -- -m crossval  # tests/crossvalidation/, needs pyuvsim
```

Use direct pytest for a focused path; the Pixi `test` task already prepends
`tests/`:

```bash
pixi run python -m pytest tests/unit/test_io
pixi run python -m pytest tests/unit/test_simulator tests/unit/test_cli
```

## License

MIT License. See [LICENSE](LICENSE).
