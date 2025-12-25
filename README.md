# RRIvis - Radio Astronomy Visibility Simulator

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/kartikmandar/RRIvis/actions/workflows/ci.yml/badge.svg)](https://github.com/kartikmandar/RRIvis/actions)

A Python package for simulating radio interferometry visibilities with GPU acceleration support. RRIvis implements the Radio Interferometer Measurement Equation (RIME) with full polarization support and is designed for 21cm cosmology, EoR research, and general radio astronomy applications.

## Features

- **GPU Acceleration**: Universal GPU support via JAX (NVIDIA/AMD/Apple Silicon/TPU) and Numba (CUDA/ROCm)
- **Full Polarization**: Complete RIME implementation with 2x2 Jones matrices and coherency matrices
- **Jones Matrix Framework**: 8 Jones terms (K, E, Z, T, P, G, B, D) for comprehensive instrumental modeling
- **Measurement Set I/O**: Export to CASA MS format for QuartiCal, WSClean, and CASA calibration
- **Multiple Sky Models**: GLEAM, GSM, HEALPix, and custom point sources
- **Flexible Beam Models**: Analytic (Gaussian, Airy) and FITS-based beam patterns
- **High-Level API**: Simple `Simulator` class for notebooks and scripts
- **Backend Abstraction**: Write once, run on CPU or GPU
- **Type-Safe Configuration**: Pydantic-based validation with helpful error messages

## Installation

### Using pip (recommended)

```bash
pip install rrivis
```

### With GPU Support

**NVIDIA GPU (CUDA 12):**
```bash
pip install rrivis[gpu-cuda]
```

**AMD GPU (ROCm):**
```bash
pip install rrivis[gpu-rocm]
```

**Apple Silicon (Metal):**
```bash
pip install rrivis[gpu]  # Auto-detects Metal on M1/M2/M3/M4
```

**Google TPU:**
```bash
pip install rrivis[tpu]
```

### Using Pixi (development)

```bash
# Clone repository
git clone https://github.com/kartikmandar/RRIvis.git
cd RRIvis

# Install with pixi
pixi install

# Run tests
pixi run pytest
```

### All Optional Dependencies

```bash
pip install rrivis[all]  # GPU, Numba, MS I/O, dev tools, docs
```

## Quick Start

### High-Level API (Recommended)

```python
from rrivis import Simulator

# Create simulator with auto-detected backend
sim = Simulator(backend="auto")

# Configure simulation
sim.setup(
    antenna_layout="path/to/antennas.txt",
    frequencies=[100, 150, 200],  # MHz
    sky_model="gleam",
    observation={
        "latitude": -30.72,
        "longitude": 21.43,
        "duration_hours": 4,
    }
)

# Run simulation (with progress bar)
results = sim.run(progress=True)

# Access results
print(f"Computed {len(results.visibilities)} baselines")
print(f"Shape: {results.visibilities[(0,1)].shape}")  # (n_times, n_freqs)

# Save to HDF5
sim.save("simulation_output.h5")
```

### Low-Level API

```python
from rrivis.core import (
    calculate_visibility,
    generate_baselines,
)
from rrivis.io import read_antenna_positions
from rrivis.backends import get_backend

# Choose backend
backend = get_backend("jax")  # or "numpy", "numba", "auto"

# Load antennas
antennas = read_antenna_positions("antennas.txt", format_type="rrivis")

# Generate baselines
baselines = generate_baselines(antennas)

# Calculate visibilities
vis = calculate_visibility(
    baselines=baselines,
    sources=sources,
    frequencies=frequencies,
    backend=backend,
)
```

### GPU Acceleration

```python
from rrivis import Simulator
from rrivis.backends import list_backends

# Check available backends
print(list_backends())  # ['numpy', 'jax', 'numba']

# Use GPU (10-50x faster for large simulations)
sim = Simulator(backend="jax")
results = sim.run()  # Automatically uses GPU if available
```

### Measurement Set Export

Export simulation results to CASA Measurement Set format for use with calibration pipelines:

```python
from rrivis import Simulator

sim = Simulator.from_config("config.yaml")
results = sim.run()

# Save as Measurement Set
sim.save("output/", format="ms")

# Now you can:
# - View in CASA: casabrowser output/simulation.ms
# - Calibrate with QuartiCal: goquartical output/simulation.ms
# - Image with WSClean: wsclean -name image output/simulation.ms
```

**Note:** MS support requires python-casacore: `pip install rrivis[ms]`

### Jones Matrix Framework

```python
from rrivis.core.jones import (
    JonesChain,
    GeometricDelayJones,
    BeamJones,
    GainJones,
    IonosphereJones,
)

# Create Jones chain for full instrumental modeling
jones_chain = JonesChain([
    GeometricDelayJones(),           # K - Geometric phase
    BeamJones(beam_type="gaussian"), # E - Primary beam
    IonosphereJones(tec=10.0),       # Z - Ionosphere
    GainJones(amplitude_std=0.01),   # G - Gain errors
])

# Use in simulation
sim = Simulator()
sim.setup(
    antenna_layout="antennas.txt",
    jones_chain=jones_chain,
)
results = sim.run()
```

### Precision Control

Control numerical precision at different stages of the simulation:

```python
from rrivis import Simulator
from rrivis.core.precision import PrecisionConfig

# Use presets for common scenarios
sim = Simulator(backend="numpy", precision="fast")      # float32 where safe
sim = Simulator(backend="numpy", precision="precise")   # float128 for critical paths
sim = Simulator(backend="numpy", precision="standard")  # float64 everywhere (default)

# Granular control for advanced users
from rrivis.core.precision import JonesPrecision

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
# Run simulation from config file
rrivis --config config.yaml --antenna-file antennas.txt

# With GPU backend
rrivis --config config.yaml --backend jax

# Check version
rrivis --version

# Migration tool (v0.1.x -> v0.2.0)
rrivis-migrate --check src/
rrivis-migrate --apply src/
```

## Configuration

RRIvis uses YAML configuration files with Pydantic validation:

```yaml
telescope:
  telescope_name: "HERA"

antenna_layout:
  antenna_positions_file: "/path/to/antennas.txt"
  antenna_file_format: "rrivis"
  all_antenna_diameter: 14.0

beams:
  beam_mode: "analytic"
  all_beam_response: "gaussian"

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
  time_interval: 1.0
  time_interval_unit: "hours"
  total_duration: 1.0
  total_duration_unit: "days"
  start_time: "2025-01-01T00:00:00"

sky_model:
  gleam:
    use_gleam: true
    flux_limit: 1.0

output:
  output_file_format: "HDF5"
  save_simulation_data: true
```

Load and validate configuration:

```python
from rrivis.io.config import load_config, create_default_config

# Load existing config (with validation)
config = load_config("config.yaml")

# Create default config file
create_default_config("default_config.yaml")

# Access with IDE autocomplete
print(config.telescope.telescope_name)
print(config.obs_frequency.n_channels)  # Computed property
```

## Package Structure

```
rrivis/
├── __init__.py          # Public API exports
├── api/                 # High-level Simulator class
│   └── simulator.py
├── backends/            # Compute backends
│   ├── base.py          # Abstract backend interface
│   ├── numpy_backend.py # CPU baseline
│   ├── jax_backend.py   # GPU via JAX
│   └── numba_backend.py # GPU via Numba
├── core/                # Core astronomy modules
│   ├── antenna.py
│   ├── baseline.py
│   ├── beams.py
│   ├── visibility.py
│   ├── source.py
│   ├── observation.py
│   └── jones/           # Jones matrix framework
│       ├── base.py
│       ├── chain.py
│       ├── geometric.py
│       ├── beam.py
│       ├── gain.py
│       ├── bandpass.py
│       ├── ionosphere.py
│       ├── troposphere.py
│       ├── parallactic.py
│       └── polarization_leakage.py
├── simulator/           # RIME simulator
│   ├── base.py
│   └── rime.py
├── io/                  # I/O and configuration
│   ├── config.py        # Pydantic models
│   ├── writers.py       # HDF5/YAML writers
│   ├── readers.py
│   └── antenna_readers.py
├── visualization/       # Plotting
│   ├── bokeh_plots.py
│   └── gsm_plots.py
└── cli/                 # Command-line interface
    ├── main.py
    └── migrate.py
```

## Sky Models

### GLEAM Catalog
```python
sim.setup(sky_model="gleam", flux_limit=1.0)  # Jy
```

### Global Sky Model (GSM)
```python
sim.setup(sky_model="gsm", nside=64)
```

### Custom Point Sources
```python
from rrivis.core.source import PointSource

sources = [
    PointSource(ra=0.0, dec=-30.0, flux=10.0, spectral_index=-0.7),
    PointSource(ra=15.0, dec=-30.0, flux=5.0, spectral_index=-0.8),
]
sim.setup(sky_model=sources)
```

## Beam Models

### Analytic Beams
```yaml
beams:
  beam_mode: "analytic"
  all_beam_response: "gaussian"  # or "airy", "cosine"
```

### FITS Beam Files
```yaml
beams:
  beam_mode: "shared"
  beam_file: "/path/to/beam.fits"  # pyuvdata UVBeam format
```

### Per-Antenna Beams
```yaml
beams:
  beam_mode: "per_antenna"
  beam_assignment: "from_config"
  antenna_beam_map:
    "ANT001": "/path/to/beam1.fits"
    "ANT002": "/path/to/beam2.fits"
```

## Testing

```bash
# Run all tests
pixi run pytest

# Run with coverage
pixi run pytest --cov=rrivis --cov-report=html

# Run specific test categories
pixi run pytest -m "not slow"          # Skip slow tests
pixi run pytest -m "not gpu"           # Skip GPU tests
pixi run pytest tests/unit/            # Unit tests only
pixi run pytest tests/integration/     # Integration tests only
```

## Performance

Approximate speedups with GPU backends (vs NumPy baseline):

| Simulation Size | JAX (GPU) | Numba (GPU) |
|-----------------|-----------|-------------|
| 100 antennas, 100 sources | 5x | 3x |
| 500 antennas, 1000 sources | 20x | 15x |
| 1000 antennas, 10000 sources | 50x | 40x |

## Documentation

- **Migration Guide**: [docs/migration_guide.md](docs/migration_guide.md)
- **API Reference**: https://rrivis.readthedocs.io
- **Examples**: [examples/](examples/)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes and add tests
4. Ensure tests pass (`pixi run pytest`)
5. Submit a pull request

## Citation

If you use RRIvis in your research, please cite:

```bibtex
@software{rrivis2025,
  author = {Mandar, Kartik},
  title = {RRIvis: Radio Astronomy Visibility Simulator},
  year = {2025},
  url = {https://github.com/kartikmandar/RRIvis}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- The HERA collaboration for inspiration and testing
- The scientific Python ecosystem (NumPy, Astropy, JAX, pyuvdata)
- The radio astronomy community for feedback and suggestions

## Related Projects

- [pyuvsim](https://github.com/RadioAstronomySoftwareGroup/pyuvsim) - Full-featured visibility simulator
- [matvis](https://github.com/HERA-Team/matvis) - Matrix-based visibility simulator
- [fftvis](https://github.com/tyler-a-cox/fftvis) - FFT-based visibility simulator
- [pyuvdata](https://github.com/RadioAstronomySoftwareGroup/pyuvdata) - UV data handling
