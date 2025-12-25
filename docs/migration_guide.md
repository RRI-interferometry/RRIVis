# Migration Guide: v0.1.x to v0.2.0

## Overview

Version 0.2.0 introduces major architectural changes that make RRIvis more powerful and easier to use:

- **Package structure**: Now installable via `pip install rrivis`
- **GPU support**: Universal GPU acceleration (NVIDIA/AMD/Apple Silicon/TPU)
- **High-level API**: New `Simulator` class for easy notebook and script usage
- **Backend abstraction**: Auto-detect and use best available hardware
- **Jones matrix framework**: Complete polarization support with 8 Jones terms
- **Type-safe configuration**: Pydantic-based configuration validation

## Breaking Changes

### 1. Import Paths Changed

**Before (v0.1.x):**
```python
# Not importable as a package - CLI only
python src/main.py --config config.yaml
```

**After (v0.2.0):**
```python
# Now importable as a proper Python package
import rrivis
from rrivis import Simulator
from rrivis.core import calculate_visibility
from rrivis.backends import get_backend
```

### 2. CLI Command Changed

**Before:**
```bash
python src/main.py --config config.yaml --antenna-file antennas.txt
```

**After:**
```bash
# Installed CLI command
rrivis --config config.yaml --antenna-file antennas.txt

# Or via pixi
pixi run rrivis --config config.yaml
```

### 3. Module Structure Reorganized

**Before (flat structure):**
```
src/
├── antenna.py
├── baseline.py
├── beams.py
├── visibility.py
├── main.py
└── ...
```

**After (modular package):**
```
src/rrivis/
├── core/           # Core astronomy calculations
│   ├── antenna.py
│   ├── baseline.py
│   ├── beams.py
│   ├── visibility.py
│   └── jones/      # Jones matrix framework
├── backends/       # Compute backends (NumPy, JAX, Numba)
├── simulator/      # RIME simulator implementation
├── api/            # High-level Simulator class
├── io/             # Configuration and I/O
├── cli/            # Command-line interface
└── visualization/  # Plotting modules
```

### 4. Configuration Changes

**Before (raw dictionaries):**
```python
config = yaml.safe_load(open("config.yaml"))
# No validation, runtime errors for typos
```

**After (Pydantic models):**
```python
from rrivis.io.config import RRIvisConfig, load_config

# Type-safe with validation
config = load_config("config.yaml")
print(config.telescope.telescope_name)  # IDE autocomplete works!

# Validation catches errors early
config = RRIvisConfig(
    obs_frequency={"starting_frequency": -100}  # ValidationError!
)
```

### 5. Antenna File Format Parameter

**Before:**
```python
from src.antenna import read_antenna_positions
antennas = read_antenna_positions("file.txt", file_format="rrivis")
```

**After:**
```python
from rrivis.io import read_antenna_positions
antennas = read_antenna_positions("file.txt", format_type="rrivis")
```

## New Features

### GPU Acceleration

RRIvis v0.2.0 supports multiple GPU backends:

```python
from rrivis import Simulator

# Auto-detect best available backend
sim = Simulator(backend="auto")

# Or explicitly choose:
sim = Simulator(backend="jax")    # JAX (NVIDIA/AMD/Apple/TPU)
sim = Simulator(backend="numba")  # Numba (CUDA/ROCm)
sim = Simulator(backend="numpy")  # CPU baseline
```

**Installation for GPU support:**

```bash
# NVIDIA GPU (CUDA 12)
pip install rrivis[gpu-cuda]

# AMD GPU (ROCm)
pip install rrivis[gpu-rocm]

# Apple Silicon (Metal) - auto-detected
pip install rrivis[gpu]

# Google TPU
pip install rrivis[tpu]
```

### High-Level Simulator API

The new `Simulator` class provides a clean, notebook-friendly interface:

```python
from rrivis import Simulator

# Create simulator with sensible defaults
sim = Simulator()

# Configure
sim.setup(
    antenna_layout="path/to/antennas.txt",
    frequencies=[100, 150, 200],  # MHz
    sky_model="gleam",
)

# Run simulation
results = sim.run(progress=True)

# Save results
sim.save("output.h5")
```

### Jones Matrix Framework

Complete polarization support with 8 Jones terms:

```python
from rrivis.core.jones import (
    GeometricDelayJones,    # K - Geometric phase delay
    BeamJones,              # E - Primary beam (direction-dependent)
    IonosphereJones,        # Z - Ionospheric effects
    TroposphereJones,       # T - Tropospheric effects
    ParallacticJones,       # P - Parallactic angle rotation
    GainJones,              # G - Electronic gains
    BandpassJones,          # B - Bandpass response
    PolarizationLeakageJones,  # D - Polarization leakage
)

# Create Jones chain
from rrivis.core.jones import JonesChain
jones = JonesChain([
    GeometricDelayJones(),
    BeamJones(beam_type="gaussian"),
    GainJones(amplitude_std=0.01),
])

# Apply to visibility calculation
vis = calculate_visibility(uvw, sky, jones_chain=jones)
```

### Backend Abstraction

Write backend-agnostic code that runs on CPU or GPU:

```python
from rrivis.backends import get_backend

# Get appropriate backend
backend = get_backend("auto")  # Auto-detect GPU

# Use backend operations (works on CPU or GPU)
x = backend.array([1, 2, 3])
y = backend.sin(x)
result = backend.sum(y)

# Check what's available
from rrivis.backends import list_backends
print(list_backends())  # ['numpy', 'jax', 'numba']
```

## Migration Steps

### Step 1: Update Imports

Use the migration tool to update import statements:

```bash
rrivis-migrate --check src/  # Check what needs updating
rrivis-migrate --apply src/  # Apply changes
```

Or manually update:

```python
# Old imports
from src.antenna import read_antenna_positions
from src.visibility import calculate_visibility

# New imports
from rrivis.io import read_antenna_positions
from rrivis.core import calculate_visibility
```

### Step 2: Update Configuration Files

Configuration files remain largely compatible, but you can now use Pydantic validation:

```python
from rrivis.io.config import load_config

# Validates and provides helpful error messages
config = load_config("old_config.yaml")
```

### Step 3: Use New API (Optional)

For new code, prefer the high-level `Simulator` API:

```python
# Old style (still works)
from rrivis.core import (
    read_antenna_positions,
    generate_baselines,
    get_sources,
    calculate_visibility,
)

antennas = read_antenna_positions("antennas.txt")
baselines = generate_baselines(antennas)
sources = get_sources("gleam", flux_limit=1.0)
vis = calculate_visibility(baselines, sources, frequencies)

# New style (recommended)
from rrivis import Simulator

sim = Simulator()
sim.setup(
    antenna_layout="antennas.txt",
    sky_model="gleam",
    frequencies=[100, 200],
)
results = sim.run()
```

### Step 4: Enable GPU Acceleration (Optional)

If you have a GPU, enable acceleration:

```bash
# Install GPU support
pip install rrivis[gpu-cuda]  # or gpu-rocm, gpu
```

```python
from rrivis import Simulator

# GPU-accelerated simulation
sim = Simulator(backend="jax")
results = sim.run()  # 10-50x faster!
```

## Deprecated Features

The following features are deprecated and will be removed in v0.3.0:

| Deprecated | Replacement |
|------------|-------------|
| `python src/main.py` | `rrivis` CLI command |
| `from src.* import` | `from rrivis.* import` |
| `file_format` parameter | `format_type` parameter |

## Getting Help

- **Documentation**: https://rrivis.readthedocs.io
- **Issues**: https://github.com/kartikmandar/RRIvis/issues
- **Migration tool**: `rrivis-migrate --help`

## Version Compatibility

| RRIvis Version | Python | NumPy | Astropy |
|----------------|--------|-------|---------|
| 0.1.x | 3.11 | >=1.24 | >=5.0 |
| 0.2.x | 3.11-3.12 | >=1.24 | >=5.0 |

## Changelog Summary

### v0.2.0 (2025-12-15)

**Added:**
- Proper Python package structure (`pip install rrivis`)
- GPU support via JAX and Numba backends
- High-level `Simulator` API class
- Jones matrix framework (8 terms)
- Pydantic configuration validation
- Backend abstraction layer
- `rrivis` and `rrivis-migrate` CLI commands
- Comprehensive test suite (376 tests)

**Changed:**
- Module structure reorganized into subpackages
- Import paths changed (`src.*` -> `rrivis.*`)
- `file_format` parameter renamed to `format_type`

**Fixed:**
- Numerous bug fixes and performance improvements
