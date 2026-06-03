# RadioSim - Radio Astronomy Visibility Simulator

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/RRI-interferometry/RadioSim/actions/workflows/ci.yml/badge.svg)](https://github.com/RRI-interferometry/RadioSim/actions)

A Python package for simulating radio interferometry visibilities with GPU acceleration support. RadioSim implements the Radio Interferometer Measurement Equation (RIME) with full polarization support and is designed for 21cm cosmology, EoR research, and general radio astronomy applications.

## Features

- **Full Polarization**: Complete RIME implementation with 2x2 Jones matrices and coherency matrices
- **Jones Matrix Framework**: 46 Jones classes spanning K, E, Z, T, P, D, G, B, F, W, C, H and more. ⚠️ **Status:** only **K** (geometric phase) and **E** (primary beam) compute real effects today; the remaining terms are scaffolded and currently return identity.
- **Sky Catalogs**: 10 VizieR point-source catalogs (GLEAM, MALS, LoTSS, VLSSr, TGSS, WENSS, SUMSS, NVSS, 3C, VLASS) + RACS via CASDA TAP + GSM/GSM2016/LFSM/Haslam diffuse models + PySM3
- **Flexible Beam Models**: Composable analytic beams (aperture shape × illumination taper × feed/reflector model) and FITS-based (pyuvdata UVBeam) patterns, with per-antenna support
- **Measurement Set I/O**: Export to CASA MS format for QuartiCal, WSClean, and CASA calibration
- **High-Level API**: Simple `Simulator` class for notebooks and scripts
- **Type-Safe Configuration**: Pydantic v2-based validation with helpful error messages
- **Precision Control**: Granular per-component precision (float32/float64/float128)
- **Backend Abstraction**: `ArrayBackend` interface with NumPy, JAX, and Numba implementations. ⚠️ **Status:** the RIME solver currently computes on NumPy; JAX/Numba GPU acceleration is on the roadmap (see [Backends & GPU status](#backends--gpu-status)).

## Installation

### Using pip (recommended)

```bash
pip install radiosim
```

### With GPU Support

> ⚠️ **Status:** these extras install the JAX/GPU stack, but the RIME solver does not yet dispatch to it — simulations currently run on NumPy regardless of the selected backend. GPU acceleration is in progress.

**NVIDIA GPU (CUDA 12):**
```bash
pip install radiosim[gpu-cuda]
```

**AMD GPU (ROCm):**
```bash
pip install radiosim[gpu-rocm]
```

**Apple Silicon (Metal):**
```bash
pip install radiosim[gpu]  # Auto-detects Metal on M1/M2/M3/M4
```

**Google TPU:**
```bash
pip install radiosim[tpu]
```

### Using Pixi (development)

```bash
# Clone repository
git clone https://github.com/RRI-interferometry/RadioSim.git
cd RadioSim

# Install with pixi
pixi install

# Run tests
pixi run test
```

### All Optional Dependencies

```bash
pip install radiosim[all]  # GPU, Numba, MS I/O, dev tools, docs
```

## Quick Start

### High-Level API (Recommended)

```python
from radiosim import Simulator

# From configuration file
sim = Simulator.from_config("config.yaml")
results = sim.run(progress=True)
sim.plot(plot_type="all", output_dir="plots/")
sim.save("output/", format="hdf5")

# Or programmatic
sim = Simulator(
    config={
        "antenna_layout": {
            "antenna_positions_file": "hera_5.txt",
            "antenna_file_format": "radiosim",
            "all_antenna_diameter": 14.0,
        },
        "obs_frequency": {
            "frequencies_hz": [150e6, 160e6, 170e6],
            "frequency_unit": "MHz",
        },
        "sky_model": {
            "sources": [{"kind": "test_sources"}],
        },
        "visibility": {"sky_representation": "point_sources"},
        "location": {"lat": -30.72, "lon": 21.43, "height": 1073.0},
        "obs_time": {"start_time": "2025-01-01T00:00:00"},
    },
    backend="auto",
    precision="standard",
)
results = sim.run(progress=True)

# Access results
print(f"Computed {len(results['visibilities'])} baselines")
```

### Low-Level API

```python
from radiosim.core import (
    calculate_visibility,
    generate_baselines,
    read_antenna_positions,
)
from radiosim.backends import get_backend

# Choose backend
backend = get_backend("jax")  # or "numpy", "numba", "auto"

# Load antennas
antennas = read_antenna_positions("antennas.txt", format_type="radiosim")

# Generate baselines
baselines = generate_baselines(antennas)
```

### Backends & GPU status

```python
from radiosim.backends import list_backends, get_backend

# Discover what's installed
print(list_backends())  # {'numpy': True, 'jax': bool, 'numba': bool, ...}

backend = get_backend("auto")  # "numpy" | "jax" | "numba" | "auto"
```

> ⚠️ **Status:** the `ArrayBackend` abstraction, backend selection, and discovery are complete, but the RIME solver in `core/visibility.py` / `core/visibility_healpix.py` currently computes with NumPy directly and does **not** yet dispatch through the selected backend. Choosing `jax`/`numba` will not accelerate a simulation today (and the point-source path may error under JAX). GPU acceleration is actively being wired up.

### Measurement Set Export

```python
from radiosim import Simulator

sim = Simulator.from_config("config.yaml")
results = sim.run()

# Save as Measurement Set
sim.save("output/", format="ms")

# Now you can:
# - View in CASA: casabrowser output/simulation.ms
# - Calibrate with QuartiCal: goquartical output/simulation.ms
# - Image with WSClean: wsclean -name image output/simulation.ms
```

**Note:** MS support requires python-casacore: `pip install radiosim[ms]`

### Jones Matrix Framework

> ⚠️ **Status:** of the 46 Jones classes, only `GeometricPhaseJones` (K) and the beam classes (E) currently compute real effects; the other terms are scaffolded and return identity. The API below is stable, but adding the stub terms does not change the output yet.

```python
from radiosim.backends import get_backend
from radiosim.core.jones import JonesChain, AnalyticBeamJones

backend = get_backend("numpy")

# Terms are added to a chain (canonical order K -> Z -> T -> E -> P -> D -> G -> B;
# the chain applies them in reverse, sky-side first). K is applied separately by the
# RIME solver, so a typical chain configures the primary beam (E):
chain = JonesChain(backend)
chain.add_term(
    AnalyticBeamJones(
        source_altaz=source_altaz,
        frequencies=frequencies,
        diameter=14.0,
        aperture_shape="circular",   # circular | rectangular | elliptical
        taper="gaussian",            # uniform | gaussian | parabolic | parabolic_squared | cosine
    )
)
```

In practice you rarely build chains by hand — `Simulator` / `RIMESimulator` assemble the beam term from your config's `beams:` section.

### Precision Control

```python
from radiosim import Simulator
from radiosim.core.precision import PrecisionConfig

# Use presets
sim = Simulator(backend="numpy", precision="fast")      # float32 where safe
sim = Simulator(backend="numpy", precision="precise")   # float128 for critical paths
sim = Simulator(backend="numpy", precision="standard")  # float64 everywhere (default)

# Granular control
from radiosim.core.precision import JonesPrecision

precision = PrecisionConfig(
    default="float64",
    jones=JonesPrecision(
        geometric_phase="float128",  # Critical for phase accuracy
        beam="float32",              # Less sensitive
    ),
    accumulation="float64",
    output="float32",
)
sim = Simulator(backend="numpy", precision=precision)
```

**Precision presets:**
- `standard`: float64 everywhere (default, ~15 decimal digits)
- `fast`: float32 where safe, float64 for critical paths (2x faster, 50% less memory)
- `precise`: float128 for critical paths, float64 elsewhere (NumPy only, for validation)
- `ultra`: float128 everywhere (NumPy only, very slow, for debugging)

## Command-Line Interface

```bash
# Run simulation from config file (primary mode)
radiosim --config config.yaml

# Override antenna file or output dir
radiosim --config config.yaml --antenna-file antennas.txt --backend jax

# Run simulation with CLI arguments
radiosim simulate \
  --antenna-layout hera_5.txt \
  --frequencies 150,160,170 \
  --sky-model test \
  --output output/ \
  --format hdf5 \
  --backend auto

# Generate a default configuration template
radiosim init --output config.yaml

# Validate a config file
radiosim validate config.yaml

# Check version
radiosim --version
```

## Configuration

RadioSim uses YAML configuration files with Pydantic v2 validation:

```yaml
telescope:
  telescope_name: "HERA"

antenna_layout:
  antenna_positions_file: "/path/to/antennas.txt"
  antenna_file_format: "radiosim"
  all_antenna_diameter: 14.0

beams:
  beam_mode: "analytic"
  aperture_shape: "circular"      # circular, rectangular, elliptical
  taper: "gaussian"               # uniform, gaussian, parabolic, parabolic_squared, cosine

location:
  lat: -30.72
  lon: 21.43
  height: 1073.0

obs_frequency:
  starting_frequency: 100.0
  frequency_interval: 1.0
  frequency_bandwidth: 50.0
  frequency_unit: "MHz"

obs_time:
  start_time: "2025-01-01T00:00:00"
  duration_seconds: 3600.0
  time_step_seconds: 60.0

sky_model:
  flux_unit: "Jy"               # required: Jy, mJy, or uJy
  sources:
    - kind: gleam
      catalog: "gleam_egc"     # gleam_egc, gleam_x_dr1, gleam_x_dr2, ...
      flux_limit: 1.0
      max_rows: 10000

output:
  output_file_format: "HDF5"
  save_simulation_data: true
```

Load and validate configuration:

```python
from radiosim.io.config import load_config, create_default_config

# Load existing config (with validation)
config = load_config("config.yaml")

# Create default config template
create_default_config("default_config.yaml")

# Access with IDE autocomplete
print(config.telescope.telescope_name)
print(config.obs_frequency.n_channels)  # Computed property
```

## Package Structure

```
radiosim/
├── __init__.py              # Public API exports
├── __about__.py             # Version metadata
├── api/
│   └── simulator.py         # Simulator class (recommended entry point)
├── backends/                # Compute backends (ArrayBackend abstraction)
│   ├── base.py              # ArrayBackend ABC
│   ├── numpy_backend.py     # CPU — the active compute path
│   ├── jax_backend.py       # JAX (CUDA/ROCm/Metal/TPU) — not yet wired into the RIME
│   └── numba_backend.py     # Numba + Dask — not yet wired into the RIME
├── core/                    # Core astronomy modules
│   ├── antenna.py           # Multi-format antenna readers (6 formats)
│   ├── baseline.py          # Baseline generation
│   ├── observation.py       # Location/time context
│   ├── polarization.py      # Stokes <-> Coherency algebra
│   ├── precision.py         # PrecisionConfig + presets
│   ├── visibility.py        # Core RIME (point sources)
│   ├── visibility_healpix.py # RIME for HEALPix diffuse maps
│   ├── jones/               # Jones matrix framework (46 classes; only K + E implemented)
│   │   ├── base.py          # JonesTerm ABC
│   │   ├── chain.py         # JonesChain orchestrator
│   │   ├── geometric.py     # K: GeometricPhaseJones (implemented)
│   │   ├── ionosphere.py    # Z: IonosphereJones + variants (stub)
│   │   ├── troposphere.py   # T: TroposphereJones + variants (stub)
│   │   ├── parallactic.py   # P: ParallacticAngleJones + variants (stub)
│   │   ├── gain.py          # G: GainJones, ElevationGainJones (stub)
│   │   ├── bandpass.py      # B: BandpassJones + variants (stub)
│   │   ├── polarization_leakage.py  # D: PolarizationLeakageJones + variants (stub)
│   │   ├── faraday.py       # F: FaradayRotationJones (stub)
│   │   ├── wterm.py         # W: WPhaseJones, WProjectionJones (stub)
│   │   ├── receptor.py      # C/H: ReceptorConfigJones, BasisTransformJones (stub)
│   │   ├── element_beam.py  # Ee/a/dE: ElementBeamJones, ArrayFactorJones (stub)
│   │   ├── delay.py         # Kd/Rc/ff: DelayJones, CableReflectionJones (stub)
│   │   ├── crosshand.py     # X/Kx/DF: CrosshandPhaseJones + variants (stub)
│   │   ├── baseline_errors.py  # M/Q: JonesBaselineTerm ABC (Hadamard, not the chain)
│   │   └── beam/            # Primary beam (E term) — implemented
│   │       ├── analytic/    # composable analytic beam (aperture × taper × feed/reflector)
│   │       └── fits/        # BeamFITSHandler, BeamManager (pyuvdata UVBeam)
│   └── sky/                 # Sky model system (subpackages)
│       ├── containers/      # SkyModel, PointSourceData, HealpixData, SkyProvenance
│       ├── loaders/         # VizieR / diffuse / FITS / BBS / pyradiosky / synthetic
│       ├── registry/        # loader registry + catalog metadata (catalogs.py)
│       ├── operations/      # transforms, factories, point<->healpix conversion, regions
│       ├── combine/         # prepare_sky_model() + physical-disjointness checks
│       ├── recipes/         # realistic_foreground, dN/dS source counts
│       ├── diagnostics/     # power/delay spectra, statistics
│       └── io/              # SkyH5 serialization + module-level plot_* functions
├── simulator/               # RIME simulator (Strategy pattern)
│   ├── base.py              # VisibilitySimulator ABC
│   └── rime.py              # RIMESimulator: O(N_src x N_bl x N_freq)
├── io/                      # I/O and configuration
│   ├── config.py            # Pydantic v2 config models (RadioSimConfig)
│   ├── writers.py           # HDF5/YAML output
│   ├── readers.py           # HDF5 reader
│   ├── measurement_set.py   # CASA MS read/write
│   ├── antenna_readers.py   # Re-exports from core.antenna
│   └── fits_utils.py        # FITS file inspector
├── utils/                   # coordinates, precision plumbing, device/network, validation
│   ├── validation.py        # Pre-flight config validator
│   └── logging.py           # Rich-based logging
├── visualization/           # Plotting (Bokeh/Plotly)
│   ├── bokeh_plots.py       # Interactive Bokeh/Plotly plots
│   └── gsm_plots.py         # GSM sky model plotting
└── cli/
    └── main.py              # Click-based CLI entry point
```

## Sky Models

### Point-Source Catalogs (via VizieR)

| Loader Name | Survey | Frequency |
|-------------|--------|-----------|
| `"gleam"` | GLEAM EGC, X-DR1/DR2, Galactic | 76-200 MHz |
| `"mals"` | MALS DR1/DR2 | 1.2-1.4 GHz |
| `"vlssr"` | VLSSr | 74 MHz |
| `"tgss"` | TGSS ADR1 | 150 MHz |
| `"wenss"` | WENSS | 325 MHz |
| `"sumss"` | SUMSS | 843 MHz |
| `"nvss"` | NVSS | 1.4 GHz |
| `"lotss"` | LoTSS DR1/DR2 | 144 MHz |
| `"3c"` | 3CR | 178 MHz |
| `"vlass"` | VLASS Quick Look Ep.1 | 3 GHz |
| `"racs"` | RACS Low/Mid/High (CASDA TAP) | 887-1655 MHz |

Use typed loader functions for programmatic work. VizieR-backed catalog loaders
require a `region`, `max_rows`, or `allow_full_catalog=True` so large surveys are
not downloaded accidentally.

```python
from radiosim.core.sky import prepare_sky_model
from radiosim.core.sky.loaders import load_gleam, load_lotss, load_racs
from radiosim.core.precision import PrecisionConfig

precision = PrecisionConfig.standard()

# Load GLEAM EGC catalog (sources > 1 Jy)
sky1 = load_gleam(flux_limit=1.0, max_rows=10000, precision=precision)

# Load LoTSS DR2
sky2 = load_lotss(
    release="dr2",
    flux_limit=0.001,
    max_rows=10000,
    precision=precision,
)

# Load RACS-Low via CASDA TAP
sky3 = load_racs(band="low", flux_limit=1.0, max_rows=10000, precision=precision)

# Combine multiple catalogs (runs physical-disjointness checks)
combined = prepare_sky_model([sky1, sky2, sky3], precision=precision)
```

### Diffuse Sky Models

```python
import numpy as np
from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky.loaders import load_diffuse_sky, load_pysm3

frequencies = np.array([150e6, 160e6, 170e6])  # Hz
precision = PrecisionConfig.standard()

# Global Sky Model 2008 (de Oliveira-Costa et al.)
sky = load_diffuse_sky(
    model="gsm2008",
    frequencies=frequencies,
    nside=64,
    precision=precision,
)

# Global Sky Model 2016 (Zheng et al.)
sky = load_diffuse_sky(
    model="gsm2016",
    frequencies=frequencies,
    nside=64,
    precision=precision,
)

# Low-Frequency Sky Model (10-408 MHz)
sky = load_diffuse_sky(
    model="lfsm",
    frequencies=frequencies,
    nside=64,
    precision=precision,
)

# Haslam 408 MHz with spectral scaling
sky = load_diffuse_sky(
    model="haslam",
    frequencies=frequencies,
    nside=64,
    precision=precision,
)

# Planck Sky Model components (PySM3)
sky = load_pysm3(
    components=["s1", "d1"],
    frequencies=frequencies,
    nside=64,
    precision=precision,
)

# PySM3 component-based foreground model
sky = load_pysm3(
    components=["s1", "f1"],
    frequencies=frequencies,
    nside=64,
    precision=precision,
)
```

### Custom Point Sources

```python
from radiosim.core.sky import create_from_arrays
from radiosim.core.precision import PrecisionConfig
import numpy as np

precision = PrecisionConfig.standard()

# Build from arrays directly
sky = create_from_arrays(
    ra_rad=np.deg2rad([0.0, 15.0]),
    dec_rad=np.deg2rad([-30.72, -30.72]),
    flux=np.array([10.0, 5.0]),              # Jy
    spectral_index=np.array([-0.7, -0.8]),
    precision=precision,
)
```

## Beam Models

### Analytic Beams

The analytic beam is composable: an **aperture shape** × an **illumination taper**, optionally driven by a **feed** and **reflector** model.

```yaml
beams:
  beam_mode: "analytic"
  aperture_shape: "circular"    # circular | rectangular | elliptical
  taper: "gaussian"             # uniform | gaussian | parabolic | parabolic_squared | cosine
  edge_taper_dB: 10.0
  feed_model: "none"            # none | corrugated_horn | open_waveguide | dipole_ground_plane
  reflector_type: "prime_focus" # prime_focus | cassegrain
```

| Field | Options |
|-------|---------|
| `aperture_shape` | `circular` (Airy), `rectangular` (sinc), `elliptical` |
| `taper` | `uniform`, `gaussian`, `parabolic`, `parabolic_squared`, `cosine` |
| `feed_model` | `none`, `corrugated_horn`, `open_waveguide`, `dipole_ground_plane` |
| `reflector_type` | `prime_focus`, `cassegrain` |

### FITS Beam Files (shared)

```yaml
beams:
  beam_mode: "fits"
  per_antenna: false
  beam_file: "/path/to/beam.fits"       # pyuvdata UVBeam format
  beam_peak_normalize: true             # normalize to peak (recommended)
  beam_interp_function: "az_za_simple"  # interpolation function
```

### Per-Antenna Beams

```yaml
beams:
  beam_mode: "fits"          # or "mixed" to combine FITS + analytic
  per_antenna: true
  antenna_beam_map:          # keyed by antenna number (string)
    "0": "/path/to/beam0.fits"
    "1": "/path/to/beam1.fits"
    "2": "analytic"          # this antenna uses the analytic beam
```

### Per-Antenna Analytic Beams (heterogeneous arrays)

```python
from radiosim.core.jones.beam import AnalyticBeamJones

# Different dish diameter per antenna (looked up by antenna number)
beam = AnalyticBeamJones(
    source_altaz=source_altaz,
    frequencies=frequencies,
    diameter=14.0,                            # default for all antennas
    aperture_shape="circular",
    taper="gaussian",
    diameter_per_antenna={0: 12.0, 1: 25.0},  # per-antenna override
)
```

## Testing

```bash
# Run all tests (NOTE: `pixi run pytest` does NOT work — use the `test` task)
pixi run test

# Run with coverage
pixi run test -- --cov=radiosim --cov-report=html

# Run specific test categories (pass pytest args after `--`)
pixi run test -- -m "not slow"          # Skip slow/network tests
pixi run test -- -m "not gpu"           # Skip GPU tests
pixi run test -- tests/unit/            # Unit tests only
pixi run test -- tests/integration/     # Integration tests only
pixi run test -- tests/unit/test_jones/ # Jones matrix tests only
```

## Performance

The simulator currently runs on NumPy (see [Backends & GPU status](#backends--gpu-status)). The point-source RIME is `O(N_src × N_bl × N_freq)`, vectorized over sources, with per-antenna Jones caching to avoid recomputing the chain across baselines that share an antenna.

GPU acceleration via JAX/Numba is on the roadmap; benchmarked speedup numbers will be published once backend dispatch is wired into the solver.

## Documentation

- **Project Documentation**: [project.md](project.md) — complete API and architecture reference
- **Migration Guide**: [docs/migration_guide.md](docs/migration_guide.md) — v0.1.x to v0.2.0
- **API Reference**: https://radiosim.readthedocs.io
- **Config Examples**: [configs/](configs/) — 15+ YAML examples
- **Antenna Formats**: [antenna_layout_examples/](antenna_layout_examples/)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes and add tests
4. Ensure tests pass (`pixi run test`)
5. Submit a pull request

## Citation

If you use RadioSim in your research, please cite:

```bibtex
@software{radiosim2025,
  author = {Mandar, Kartik},
  title = {RadioSim: Radio Astronomy Visibility Simulator},
  year = {2025},
  url = {https://github.com/RRI-interferometry/RadioSim}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- The HERA collaboration for inspiration and testing
- The scientific Python ecosystem (NumPy, Astropy, JAX, pyuvdata, pygdsm)
- The radio astronomy community for feedback and suggestions

## Related Projects

- [pyuvsim](https://github.com/RadioAstronomySoftwareGroup/pyuvsim) - Full-featured visibility simulator
- [matvis](https://github.com/HERA-Team/matvis) - Matrix-based visibility simulator
- [fftvis](https://github.com/tyler-a-cox/fftvis) - FFT-based visibility simulator
- [pyuvdata](https://github.com/RadioAstronomySoftwareGroup/pyuvdata) - UV data handling
- [pygdsm](https://github.com/telegraphic/pygdsm) - Global Sky Model
