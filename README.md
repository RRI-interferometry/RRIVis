# RRIvis - Radio Astronomy Visibility Simulator

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/kartikmandar/RRIvis/actions/workflows/ci.yml/badge.svg)](https://github.com/kartikmandar/RRIvis/actions)

A Python package for simulating radio interferometry visibilities with GPU acceleration support. RRIvis implements the Radio Interferometer Measurement Equation (RIME) with full polarization support and is designed for 21cm cosmology, EoR research, and general radio astronomy applications.

## Features

- **GPU Acceleration**: Universal GPU support via JAX (NVIDIA/AMD/Apple Silicon/TPU) and Numba (CUDA/ROCm)
- **Full Polarization**: Complete RIME implementation with 2x2 Jones matrices and coherency matrices
- **Jones Matrix Framework**: 46 exported classes across 17 files covering K, E, Z, T, P, D, G, B, F, W, C, H and more
- **20+ Sky Catalogs**: 20 VizieR catalogs (GLEAM, MALS, VLSSr, TGSS, WENSS, SUMSS, NVSS, FIRST, LoTSS, AT20G, 3CR, GB6) + RACS via CASDA TAP + GSM/LFSM/Haslam diffuse models + PySM3 + ULSA
- **Flexible Beam Models**: Analytic (Gaussian, Airy, cosine, exponential, short dipole) and FITS-based beam patterns with per-antenna support
- **Measurement Set I/O**: Export to CASA MS format for QuartiCal, WSClean, and CASA calibration
- **High-Level API**: Simple `Simulator` class for notebooks and scripts
- **Backend Abstraction**: Write once, run on CPU or GPU
- **Type-Safe Configuration**: Pydantic v2-based validation with helpful error messages
- **Precision Control**: Granular per-component precision (float32/float64/float128)

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

# From configuration file
sim = Simulator.from_config("config.yaml")
results = sim.run(progress=True)
sim.plot(plot_type="all", output_dir="plots/")
sim.save("output/", format="hdf5")

# Or programmatic
sim = Simulator(
    antenna_layout="hera_5.txt",
    frequencies=[150.0, 160.0, 170.0],   # MHz
    sky_model="test",
    location={"lat": -30.72, "lon": 21.43, "height": 1073.0},
    start_time="2025-01-01T00:00:00",
    backend="auto",
    precision="standard",
)
results = sim.run(progress=True)

# Access results
print(f"Computed {len(results['visibilities'])} baselines")
```

### Low-Level API

```python
from rrivis.core import (
    calculate_visibility,
    generate_baselines,
    read_antenna_positions,
)
from rrivis.backends import get_backend

# Choose backend
backend = get_backend("jax")  # or "numpy", "numba", "auto"

# Load antennas
antennas = read_antenna_positions("antennas.txt", format_type="rrivis")

# Generate baselines
baselines = generate_baselines(antennas)
```

### GPU Acceleration

```python
from rrivis import Simulator
from rrivis.backends import list_backends

# Check available backends
print(list_backends())  # {'numpy': True, 'jax': bool, 'numba': bool, ...}

# Use GPU (10-50x faster for large simulations)
sim = Simulator.from_config("config.yaml", backend="jax")
results = sim.run()
```

### Measurement Set Export

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
    GeometricPhaseJones,
    AnalyticBeamJones,
    IonosphereJones,
    GainJones,
    BandpassJones,
    FaradayRotationJones,
)

# Standard 8-term chain: K -> Z -> T -> E -> P -> D -> G -> B
jones_chain = JonesChain([
    GeometricPhaseJones(),                   # K - Geometric phase
    IonosphereJones(tec=10.0),               # Z - Ionosphere
    AnalyticBeamJones(..., beam_type="gaussian"),  # E - Primary beam
    GainJones(amplitude_std=0.01),           # G - Gain errors
    BandpassJones(),                          # B - Bandpass
])

# Extended terms also available:
# FaradayRotationJones, WPhaseJones, DelayJones,
# CrosshandPhaseJones, ElementBeamJones, ...
```

### Precision Control

```python
from rrivis import Simulator
from rrivis.core.precision import PrecisionConfig

# Use presets
sim = Simulator(backend="numpy", precision="fast")      # float32 where safe
sim = Simulator(backend="numpy", precision="precise")   # float128 for critical paths
sim = Simulator(backend="numpy", precision="standard")  # float64 everywhere (default)

# Granular control
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
# Run simulation from config file (primary mode)
rrivis --config config.yaml

# Override antenna file or output dir
rrivis --config config.yaml --antenna-file antennas.txt --backend jax

# Run simulation with CLI arguments
rrivis simulate \
  --antenna-layout hera_5.txt \
  --frequencies 150,160,170 \
  --sky-model test \
  --output output/ \
  --format hdf5 \
  --backend auto

# Generate a default configuration template
rrivis init --output config.yaml

# Validate a config file
rrivis validate config.yaml

# Check version
rrivis --version
```

## Configuration

RRIvis uses YAML configuration files with Pydantic v2 validation:

```yaml
telescope:
  telescope_name: "HERA"

antenna_layout:
  antenna_positions_file: "/path/to/antennas.txt"
  antenna_file_format: "rrivis"
  all_antenna_diameter: 14.0

beams:
  beam_mode: "analytic"
  all_beam_response: "gaussian"   # gaussian, airy, cosine, exponential, short_dipole
  beam_peak_normalize: true       # normalize FITS beams to peak

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
  gleam:
    use_gleam: true
    gleam_variant: "gleam_egc"  # gleam_egc, gleam_x_dr1, gleam_x_dr2, ...
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

# Create default config template
create_default_config("default_config.yaml")

# Access with IDE autocomplete
print(config.telescope.telescope_name)
print(config.obs_frequency.n_channels)  # Computed property
```

## Package Structure

```
rrivis/
├── __init__.py              # Public API exports
├── __about__.py             # Version metadata
├── api/
│   └── simulator.py         # Simulator class (recommended entry point)
├── backends/                # Compute backends
│   ├── base.py              # ArrayBackend ABC
│   ├── numpy_backend.py     # CPU (always available)
│   ├── jax_backend.py       # GPU via JAX (CUDA/ROCm/Metal/TPU)
│   └── numba_backend.py     # JIT via Numba + Dask distributed
├── core/                    # Core astronomy modules
│   ├── antenna.py           # Multi-format antenna readers (6 formats)
│   ├── baseline.py          # Baseline generation
│   ├── observation.py       # Location/time context
│   ├── polarization.py      # Stokes <-> Coherency algebra
│   ├── precision.py         # PrecisionConfig + presets
│   ├── visibility.py        # Core RIME (point sources)
│   ├── visibility_healpix.py # RIME for HEALPix diffuse maps
│   ├── jones/               # Jones matrix framework (46 classes)
│   │   ├── base.py          # JonesTerm ABC
│   │   ├── chain.py         # JonesChain orchestrator
│   │   ├── geometric.py     # K: GeometricPhaseJones
│   │   ├── ionosphere.py    # Z: IonosphereJones + variants
│   │   ├── troposphere.py   # T: TroposphereJones + variants
│   │   ├── parallactic.py   # P: ParallacticAngleJones + variants
│   │   ├── gain.py          # G: GainJones, ElevationGainJones
│   │   ├── bandpass.py      # B: BandpassJones + variants
│   │   ├── polarization_leakage.py  # D: PolarizationLeakageJones + variants
│   │   ├── faraday.py       # F: FaradayRotationJones
│   │   ├── wterm.py         # W: WPhaseJones, WProjectionJones
│   │   ├── receptor.py      # C/H: ReceptorConfigJones, BasisTransformJones
│   │   ├── element_beam.py  # Ee/a/dE: ElementBeamJones, ArrayFactorJones
│   │   ├── delay.py         # Kd/Rc/ff: DelayJones, CableReflectionJones
│   │   ├── crosshand.py     # X/Kx/DF: CrosshandPhaseJones + variants
│   │   ├── baseline_errors.py  # M/Q: JonesBaselineTerm ABC + baseline terms
│   │   └── beam/            # Primary beam (E term)
│   │       ├── analytic.py  # Gaussian, Airy, cosine, exponential, short_dipole
│   │       └── fits.py      # BeamFITSHandler, BeamManager
│   └── sky/                 # Sky model system
│       ├── model.py         # SkyModel dataclass (main class)
│       ├── catalogs.py      # Catalog metadata (20 VizieR, 3 RACS, 4 diffuse)
│       ├── constants.py     # Physical constants + T_b/Jy conversions
│       ├── _loaders_vizier.py   # VizieR catalog loader mixin
│       ├── _loaders_diffuse.py  # Diffuse sky loader mixin (pygdsm, pysm3, ulsa)
│       └── _loaders_pyradiosky.py  # PyRadioSky file loader mixin
├── simulator/               # RIME simulator (Strategy pattern)
│   ├── base.py              # VisibilitySimulator ABC
│   └── rime.py              # RIMESimulator: O(N_src x N_bl x N_freq)
├── io/                      # I/O and configuration
│   ├── config.py            # Pydantic v2 config models (RRIvisConfig)
│   ├── writers.py           # HDF5/YAML output
│   ├── readers.py           # HDF5 reader
│   ├── measurement_set.py   # CASA MS read/write
│   ├── antenna_readers.py   # Re-exports from core.antenna
│   └── fits_utils.py        # FITS file inspector
├── utils/
│   ├── validation.py        # Pre-flight config validator
│   └── logging.py           # Rich-based logging
├── visualization/           # Plotting
│   ├── bokeh_plots.py       # Interactive Bokeh/Plotly plots
│   └── gsm_plots.py         # GSM sky model plotting
└── cli/
    └── main.py              # Click-based CLI entry point
```

## Sky Models

### Point-Source Catalogs (via VizieR)

| Method | Survey | Frequency |
|--------|--------|-----------|
| `from_gleam()` | GLEAM EGC, X-DR1/DR2, Galactic, SGP, G4Jy | 76-200 MHz |
| `from_mals()` | MALS DR1/DR2/DR3 | 1.2-1.4 GHz |
| `from_vlssr()` | VLSSr | 74 MHz |
| `from_tgss()` | TGSS ADR1 | 150 MHz |
| `from_wenss()` | WENSS | 325 MHz |
| `from_sumss()` | SUMSS | 843 MHz |
| `from_nvss()` | NVSS | 1.4 GHz |
| `from_first()` | FIRST | 1.4 GHz |
| `from_lotss()` | LoTSS DR1/DR2 | 144 MHz |
| `from_at20g()` | AT20G | 20 GHz |
| `from_3c()` | 3CR | 178 MHz |
| `from_gb6()` | GB6 | 4.85 GHz |
| `from_racs()` | RACS Low/Mid/High (CASDA TAP) | 887-1655 MHz |

```python
from rrivis.core.sky.model import SkyModel
from rrivis.core.precision import PrecisionConfig

precision = PrecisionConfig.standard()

# Load GLEAM EGC catalog (sources > 1 Jy)
sky = SkyModel.from_gleam(flux_limit=1.0, precision=precision)

# Load LoTSS DR2
sky = SkyModel.from_lotss(release="dr2", flux_limit=0.001, precision=precision)

# Load RACS-Low via CASDA TAP
sky = SkyModel.from_racs(band="low", flux_limit=1.0, precision=precision)

# Combine multiple catalogs
combined = SkyModel.combine([sky1, sky2], precision=precision)
```

### Diffuse Sky Models

```python
import numpy as np

frequencies = np.array([150e6, 160e6, 170e6])  # Hz

# Global Sky Model 2008 (de Oliveira-Costa et al.)
sky = SkyModel.from_diffuse_sky("gsm2008", frequencies=frequencies, nside=64)

# Global Sky Model 2016 (Zheng et al.)
sky = SkyModel.from_diffuse_sky("gsm2016", frequencies=frequencies, nside=64)

# Low-Frequency Sky Model (10-408 MHz)
sky = SkyModel.from_diffuse_sky("lfsm", frequencies=frequencies, nside=64)

# Haslam 408 MHz with spectral scaling
sky = SkyModel.from_diffuse_sky("haslam", frequencies=frequencies, nside=64)

# Planck Sky Model components (PySM3)
sky = SkyModel.from_pysm3(components=["s1", "d1"], frequencies=frequencies, nside=64)

# Ultra-Long-wavelength Sky Model
sky = SkyModel.from_ulsa(frequencies=frequencies, nside=64)
```

### Custom Point Sources

```python
from rrivis.core.sky.model import SkyModel
import numpy as np

# Build from arrays directly
sky = SkyModel()
sky._ra_rad = np.array([0.0, 0.26])           # radians
sky._dec_rad = np.array([-0.536, -0.536])     # radians
sky._flux_ref = np.array([10.0, 5.0])         # Jy
sky._alpha = np.array([-0.7, -0.8])           # spectral index
sky._stokes_q = np.zeros(2)
sky._stokes_u = np.zeros(2)
sky._stokes_v = np.zeros(2)
sky._native_format = "point_sources"
```

## Beam Models

### Analytic Beams

```yaml
beams:
  beam_mode: "analytic"
  all_beam_response: "gaussian"     # or "airy", "cosine", "exponential", "short_dipole"
  cosine_taper_exponent: 1.0        # for cosine pattern
  exponential_taper_dB: 10.0        # for exponential pattern
```

| Pattern | Description | Best For |
|---------|-------------|----------|
| `gaussian` | Gaussian approximation, fast | Standard simulations |
| `airy` | Exact Airy disk (Bessel function) | High-accuracy sidelobe studies |
| `cosine` | Cosine-tapered (Cassegrain feeds) | Offset-fed reflectors |
| `exponential` | Exponential taper | Feed horns |
| `short_dipole` | Full 2x2 non-diagonal Jones (HERA/MWA) | Low-frequency arrays |

### FITS Beam Files

```yaml
beams:
  beam_mode: "shared"
  beam_file: "/path/to/beam.fits"   # pyuvdata UVBeam format
  beam_peak_normalize: true         # normalize to peak (recommended)
  beam_interp_function: "az_za_simple"  # interpolation function
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

### Per-Antenna Analytic Beams

```python
from rrivis.core.jones.beam import AnalyticBeamJones

# Different HPBW per antenna (e.g., heterogeneous array)
beam = AnalyticBeamJones(
    source_altaz=source_altaz,
    frequencies=frequencies,
    hpbw_radians=0.2,                  # default for all
    beam_type="gaussian",
    hpbw_per_antenna={0: 0.15, 1: 0.25},      # per-antenna override
    beam_type_per_antenna={2: "short_dipole"},  # per-antenna pattern
)
```

## Testing

```bash
# Run all tests
pixi run pytest

# Run with coverage
pixi run pytest --cov=rrivis --cov-report=html

# Run specific test categories
pixi run pytest -m "not slow"          # Skip slow/network tests
pixi run pytest -m "not gpu"           # Skip GPU tests
pixi run pytest tests/unit/            # Unit tests only
pixi run pytest tests/integration/     # Integration tests only
pixi run pytest tests/unit/test_jones/ # Jones matrix tests only
```

## Performance

Approximate speedups with GPU backends (vs NumPy baseline):

| Simulation Size | JAX (GPU) | Numba (GPU) |
|-----------------|-----------|-------------|
| 100 antennas, 100 sources | 5x | 3x |
| 500 antennas, 1000 sources | 20x | 15x |
| 1000 antennas, 10000 sources | 50x | 40x |

## Documentation

- **Project Documentation**: [project.md](project.md) — complete API and architecture reference
- **Migration Guide**: [docs/migration_guide.md](docs/migration_guide.md) — v0.1.x to v0.2.0
- **API Reference**: https://rrivis.readthedocs.io
- **Config Examples**: [configs/](configs/) — 15+ YAML examples
- **Antenna Formats**: [antenna_layout_examples/](antenna_layout_examples/)

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
- The scientific Python ecosystem (NumPy, Astropy, JAX, pyuvdata, pygdsm)
- The radio astronomy community for feedback and suggestions

## Related Projects

- [pyuvsim](https://github.com/RadioAstronomySoftwareGroup/pyuvsim) - Full-featured visibility simulator
- [matvis](https://github.com/HERA-Team/matvis) - Matrix-based visibility simulator
- [fftvis](https://github.com/tyler-a-cox/fftvis) - FFT-based visibility simulator
- [pyuvdata](https://github.com/RadioAstronomySoftwareGroup/pyuvdata) - UV data handling
- [pygdsm](https://github.com/telegraphic/pygdsm) - Global Sky Model
