# RRIvis Examples

This directory contains example scripts and Jupyter notebooks demonstrating how to use RRIvis for radio interferometry visibility simulations.

## Quick Start

### Scripts

Run the simple simulation example:

```bash
# Basic usage (auto-detect best backend)
python scripts/simple_simulation.py

# Use GPU acceleration (requires JAX)
python scripts/simple_simulation.py --backend jax

# Use a configuration file
python scripts/simple_simulation.py --config ../configs/01_parabolic_10db_taper.yaml

# Disable plotting
python scripts/simple_simulation.py --no-plot --output-dir my_results
```

### Notebooks

Open Jupyter notebooks for interactive exploration:

```bash
jupyter notebook notebooks/01_basic_usage.ipynb
```

## Example Descriptions

### Scripts

| Script | Description |
|--------|-------------|
| `simple_simulation.py` | Basic visibility simulation with command-line options |

### Notebooks

| Notebook | Description |
|----------|-------------|
| `01_basic_usage.ipynb` | Introduction to RRIvis: setup, run, plot, and save |

## Prerequisites

Make sure RRIvis is installed:

```bash
# CPU-only installation
pip install rrivis

# With GPU support (NVIDIA)
pip install rrivis[gpu-cuda]

# With GPU support (Apple Silicon)
pip install rrivis[gpu]

# Development installation
pip install -e ".[dev]"
```

## Available Backends

RRIvis supports multiple computation backends:

- **numpy**: CPU baseline (always available)
- **jax**: GPU/TPU acceleration (NVIDIA, AMD, Apple Silicon)
- **numba**: Production backend with Dask support

Check available backends:

```python
from rrivis.backends import list_backends
print(list_backends())
# {'numpy': True, 'jax': True, 'numba': True}
```

## Configuration Files

Example YAML configuration files are in the `../configs/` directory:

- `01_parabolic_10db_taper.yaml` - Standard parabolic dishes
- `02_parabolic_uniform.yaml` - Uniform illumination
- `03_parabolic_airy_pattern.yaml` - Airy disk pattern
- `04_dipole_array.yaml` - Dipole antenna array
- And more...

## Antenna Layouts

Example antenna position files are in `../antenna_layout_examples/`:

- `example_rrivis_format.txt` - Native RRIvis ENU format
- `example_casa_format.cfg` - CASA configuration format
- `example_pyuvdata_format.txt` - pyuvdata format
- `example_antenna_layout.csv` - CSV format
