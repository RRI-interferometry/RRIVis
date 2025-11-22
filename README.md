# RRIvis - Radio Astronomy Visibility Simulator

## Overview

RRIvis is a Python-based application for simulating and visualizing complex visibilities in radio astronomy. It's particularly focused on the Hydrogen Epoch of Reionization Array (HERA) and similar radio telescopes, but can be used with any antenna array configuration.

The simulator calculates visibility data using antenna positions, baselines, sky models, and source catalogs. It supports various plotting libraries and offers options for both polarized and non-polarized visibility calculations.

RRIviz provides a complete pipeline for radio interferometry simulation and visualization, including:
- Antenna array configuration
- Source modeling
- Visibility calculation
- Interactive plotting capabilities

## Features

### Antenna Configuration
- Generate antenna positions or load them from a file
- Support for different antenna diameters (uniform or per-antenna)
- Integration with PYUVData telescope objects
- Support for various antenna types including parabolic dishes, spherical reflectors, phased arrays, and dipoles

### Baseline Calculation
- Dynamically compute baselines between antennas
- Trim baselines based on specified criteria
- Support for auto-correlations and cross-correlations
- Selective baseline filtering based on length

### Sky Model Integration
- Multiple sky model options:
  - Test sources (simple 3-source configuration)
  - GLEAM (GaLactic Extragalactic All-sky MWA) catalog sources
  - GSM (Global Sky Model) 2008 sources
  - HEALPix-based representations of sky models
  - Combined models (GSM+GLEAM)
- Configurable flux limits and HEALPix resolution parameters

### Beam Modeling
- **Beam FITS File Support** (NEW!):
  - Load realistic beam patterns from pyuvdata UVBeam FITS files
  - Full 2×2 Jones matrix support for polarized beams
  - Three beam modes: shared (all antennas), per-antenna (from layout), per-antenna (from config)
  - Automatic interpolation in zenith angle, azimuth, and frequency
  - Correct azimuth convention handling (Astropy ↔ UVBeam)
- Comprehensive analytic beam pattern implementations:
  - Gaussian beam patterns
  - Parabolic dish models with different illumination tapers (uniform, cosine, Gaussian, 10dB/20dB taper)
  - Spherical reflector models
  - Phased array and dipole models
- HPBW (Half Power Beam Width) calculations for different antenna types
- Beam solid angle calculations using HEALPix
- Automatic fallback from beam FITS to analytic beams

### Visibility Calculation
- **Full Polarization Support** (NEW!):
  - Implements full Radio Interferometer Measurement Equation (RIME)
  - 2×2 complex Jones matrices for antenna beams
  - 2×2 coherency matrices for source polarization
  - Outputs XX, XY, YX, YY correlations and Stokes I
  - Africanus/Pauli convention for consistency with modern tools
  - Energy-conserving half-power convention
- Optimized method for visibility computation
- Support for both point source and HEALPix-based sky models
- Accounts for beam patterns and source spectral indices
- Calculates visibility modulus and phase
- Time and frequency domain simulation

### Visualization
- Interactive plots using Bokeh:
  - Antenna layout plots (2D and 3D)
  - Visibility vs. time plots
  - Visibility vs. frequency plots
  - Heatmaps of visibility data
  - Sky model visualizations with time-evolving maps
- Static plots using Matplotlib
- Interactive plots with hover information

### Customizability
- Adjust parameters via a user-provided YAML config or rely on in-code defaults
- Support for fixed HPBW values or frequency-dependent calculations
- Configurable observation parameters (location, timing, frequency range)

### Data Persistence
- Save visibility data and sky model plots for further analysis in HDF5 format
- Option to save simulation data and configuration
- JSON output for visibility results
- HTML files with interactive plots

### Logging
- Logs simulation details to a file and the console
- Comprehensive output of simulation parameters and progress

## Installation

### Using Pixi

1. Install Pixi if you haven't already:
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

2. Clone the repository:
   ```bash
   git clone <repository_url>
   cd RRIviz
   ```

3. Install dependencies and set up the environment:
   ```bash
   pixi install
   ```

4. Run the simulator:
   ```bash
   pixi run start
   ```

## Quick Start

To run a basic simulation with default parameters:

```bash
pixi run start
```

For a customized simulation, create a config file and specify it:

```bash
pixi run start --config /path/to/your/config.yaml --antenna-file /path/to/your/antenna.txt
```

## Usage

### Configuration

You may provide a YAML configuration file to override in-code defaults. If no config is supplied, sensible defaults embedded in the program are used.

Key configuration parameters include:

#### Telescope
- `telescope_type`: Configuration for telescope type
  - `use_PYUVData_telescope`: Boolean to use PYUVData telescope object
  - `use_custom_telescope`: Boolean to use custom telescope configuration
- `telescope_name`: Name of the telescope (e.g., "HERA")

#### Antenna Layout
- `antenna_positions_file`: Absolute path to antenna positions file
- `all_antenna_diameter`: Default diameter for all antennas in meters
- `use_different_diameters`: Boolean to use per-antenna diameters
- `fixed_hpbw`: Optional fixed HPBW value in degrees for all antennas

#### Baseline Selection
- `use_autocorrelations`: Boolean to include auto-correlations
- `use_crosscorrelations`: Boolean to include cross-correlations
- `only_selective_baseline_length`: Boolean to filter baselines by length
- `selective_baseline_lengths`: Array of baseline lengths to include

#### Location
- `lat`: Latitude of observation location in degrees
- `lon`: Longitude of observation location in degrees
- `height`: Height of observation location in meters

#### Sky Model
Multiple sky model options are available:

- `test_sources`: Simple test sources
  - `use_test_sources`: Boolean to enable test sources
  - `num_sources`: Number of test sources
  - `flux_limit`: Flux limit in Janskys

- `test_sources_healpix`: Test sources in HEALPix format
  - `use_test_sources`: Boolean to enable test sources
  - `num_sources`: Number of test sources
  - `flux_limit`: Flux limit in Janskys
  - `nside`: HEALPix resolution parameter

- `gsm_healpix`: Global Sky Model in HEALPix format
  - `use_gsm`: Boolean to enable GSM
  - `flux_limit`: Flux limit in Janskys
  - `nside`: HEALPix resolution parameter

- `gleam`: GLEAM catalog sources
  - `use_gleam`: Boolean to enable GLEAM catalog
  - `gleam_catalogue`: GLEAM catalog identifier
  - `flux_limit`: Flux limit in Janskys

- `gleam_healpix`: GLEAM catalog in HEALPix format
  - `use_gleam`: Boolean to enable GLEAM catalog
  - `flux_limit`: Flux limit in Janskys
  - `nside`: HEALPix resolution parameter

- `gsm+gleam_healpix`: Combined GSM and GLEAM models
  - `use_gsm_gleam`: Boolean to enable combined model
  - `flux_limit`: Flux limit in Janskys
  - `nside`: HEALPix resolution parameter

#### Observation Time
- `time_interval_between_observation`: Time interval between observations
- `time_interval_unit`: Unit for time interval ("seconds", "minutes", "hours", "days")
- `total_duration`: Total duration of observation
- `total_duration_unit`: Unit for total duration ("seconds", "minutes", "hours", "days")
- `start_time`: Start time of observation in ISO format

#### Frequency
- `starting_frequency`: Starting frequency in MHz
- `frequency_interval`: Frequency interval in MHz
- `frequency_bandwidth`: Total frequency bandwidth in MHz
- `frequency_unit`: Unit for frequency parameters

#### Output
- `output_file`: Name of output HDF5 file
- `save_simulation_data`: Boolean to save simulation data
- `plot_results_in_bokeh`: Boolean to generate Bokeh plots
- `simulation_data_dir`: Optional directory for simulation data
- `simulation_subdir`: Optional subdirectory under Downloads/XDG_DOWNLOAD_DIR

### Running the Simulation

Run the simulator using the `src/main.py` script:

```bash
python src/main.py \
  --antenna-file /absolute/path/to/antenna.txt \
  --config /absolute/path/to/config.yaml \
  --sim-data-dir /path/for/simulation/outputs
```

- `--antenna-file` is required unless provided in the config.
- `--config` is optional; if omitted, the built-in defaults are used.
- `--sim-data-dir` is optional; if omitted, outputs are stored under `$(XDG_DOWNLOAD_DIR or ~/Downloads)/RRIVis_simulation_data/<timestamp>`. The program prints the exact location. The `<timestamp>` is formatted as `YYYY-MM-DD_HH-MM-SS_UTC`.

### Example Configuration

A sample `config.yaml` file is provided below:

```yaml
telescope:
  telescope_type:
    use_PYUVData_telescope: False
    use_custom_telescope: True
  telescope_name: "HERA"
antenna_layout:
  antenna_positions_file: "/absolute/path/to/antenna.txt"
  all_antenna_diameter: 14.0
  use_different_diameters: false
  fixed_hpbw: None
baseline_selection:
  use_autocorrelations: True
  use_crosscorrelations: True
  only_selective_baseline_length: True
  selective_baseline_lengths: [14, 28, 42]
location:
  lat: -30.72152777777791
  lon: 21.428305555555557
  height: 1073.0
sky_model:
  test_sources:
    use_test_sources: False
    num_sources: 100
    flux_limit: 50
  test_sources_healpix:
    use_test_sources: False
    num_sources: 100
    flux_limit: 50
    nside: 32
  gsm_healpix:
    use_gsm: False
    flux_limit: 50
    nside: 32
  gleam:
    use_gleam: False
    gleam_catalogue: "VIII/100/gleamegc"
    flux_limit: 50
  gleam_healpix:
    use_gleam: False
    flux_limit: 50
    nside: 32
  gsm+gleam_healpix:
    use_gsm_gleam: False
    flux_limit: 50
    nside: 32
obs_time:
  time_interval_between_observation: 1.0
  time_interval_unit: "hours"
  total_duration: 1
  total_duration_unit: "days"
  start_time: "2025-01-01T00:00:00"
frequency:
  starting_frequency: 50.0
  frequency_interval: 1.0
  frequency_bandwidth: 100.0
  frequency_unit: "MHz"
output:
  output_file: "complex_visibility.h5"
  save_simulation_data: True
  plot_results_in_bokeh: True
  simulation_data_dir: "/absolute/path/for/simulations"
  simulation_subdir: "RRIVis_simulation_data"
```

## Project Architecture

### Core Modules

#### `src/antenna.py`
Handles antenna position reading and management from text files. Supports different antenna diameters and configurations.

#### `src/baseline.py`
Generates baselines between antenna pairs with associated metadata. Supports filtering based on baseline length and correlation type.

#### `src/beams.py`
Implements various beam pattern models and HPBW calculations:
- Gaussian beam patterns
- Parabolic dish models with different illumination tapers
- Spherical reflector models
- Phased array and dipole models
- HPBW calculations using the formula: `HPBW ≈ k * (lambda / D)` where k depends on antenna type

#### `src/observation.py`
Manages observation parameters including geographic location, timing, and coordinate system transformations.

#### `src/visibility.py`
Core visibility calculation engine that computes complex visibilities for each baseline, accounting for beam patterns and source spectral indices.

#### `src/plot.py`
Comprehensive visualization module using Bokeh for interactive plots, including antenna layouts, visibility data, and sky models.

#### `src/gsm_map.py`
Generates visualizations of the Global Sky Model with time-evolving sky maps and observable regions.

#### `src/source.py`
Provides multiple sky model options including test sources, GLEAM catalog sources, and GSM sources.

#### `src/fits.py`
Simple FITS file inspection tool.

#### `src/reading.py`
Utilities for reading HDF5 files.

### Simulator Modules

#### `src/simulators/fftvis.py`
Placeholder for future FFT-based visibility simulator implementation.

#### `src/simulators/matvis.py`
Placeholder for future matrix-based visibility simulator implementation.

### Main Workflow

The main workflow (implemented in `src/main.py`) follows these steps:

1. **Setup Configuration**: Load simulation parameters from a configuration file
2. **Antenna Configuration**: Read antenna positions and generate baselines
3. **Sky Model Selection**: Choose and load appropriate sky model (test sources, GLEAM, GSM)
4. **Observation Setup**: Define location, timing, and frequency parameters
5. **Beam Calculation**: Compute beam patterns and HPBW for each antenna
6. **Visibility Simulation**: Calculate visibilities over time and frequency
7. **Visualization**: Generate plots of results
8. **Data Export**: Save results in various formats (HDF5, JSON, HTML)

## Technical Implementation

### Dependencies
- **Python**: 3.11 (as specified in pixi.toml)
- **NumPy**: For numerical computations
- **Astropy**: For astronomical calculations and coordinate systems
- **HEALPy**: For spherical pixelization
- **Bokeh**: For interactive visualization
- **Matplotlib**: For static plotting
- **Astroquery**: For accessing astronomical catalogs
- **PyYAML**: For configuration file parsing
- **PyGSM**: For Global Sky Model access
- **h5py**: For HDF5 file handling

### HPBW Calculation

The project implements a comprehensive HPBW calculation system based on the relationship:

```
HPBW ≈ k * (lambda / D) in radians
```

Where:
- `lambda` is the wavelength (calculated as speed_of_light / frequency)
- `D` is the dish diameter
- `k` is a coefficient that depends on the antenna type and illumination

Different antenna types have different k values:
- Parabolic dish (uniform): k = 1.02
- Parabolic dish (cosine taper): k = 1.10
- Parabolic dish (Gaussian taper): k = 1.18
- Spherical reflector (uniform): k = 1.05
- Phased array (uniform): k = 1.10
- Dipole: Constant HPBW of π/2 radians (90 degrees)

### Beam Pattern Calculation

The Gaussian primary beam pattern is calculated using:

```
A(theta) = exp(-(theta / (sqrt(2) * theta_HPBW))^2)
```

Where:
- `theta` is the zenith angle
- `theta_HPBW` is the Half Power Beam Width in radians

### Using Beam FITS Files

RRIvis supports loading realistic antenna beam patterns from FITS files in the pyuvdata UVBeam format. This enables simulations with measured or simulated beam patterns including full polarization effects.

#### Beam Modes

Three beam modes are supported:

1. **Analytic Mode** (default): Uses analytic beam patterns (Gaussian, Airy, etc.)
   ```yaml
   beams:
     beam_mode: "analytic"
     all_beam_response: "gaussian"
   ```

2. **Shared Mode**: All antennas use the same beam FITS file
   ```yaml
   beams:
     beam_mode: "shared"
     beam_file: "/path/to/beam.fits"
     beam_za_max_deg: 90.0
     beam_freq_buffer_hz: 1e6
   ```

3. **Per-Antenna Mode**: Different beams for different antennas

   a. **From Layout File**: Uses BeamID column in antenna file
   ```yaml
   beams:
     beam_mode: "per_antenna"
     beam_assignment: "from_layout"
   ```

   Antenna file format with BeamID:
   ```
   Name    Number  BeamID      E    N    U
   ANT001  0       beam_dish   0.0  0.0  0.0
   ANT002  1       beam_dipole 10.0 0.0  0.0
   ```

   b. **From Config**: Explicit antenna-to-beam mapping
   ```yaml
   beams:
     beam_mode: "per_antenna"
     beam_assignment: "from_config"
     antenna_beam_map:
       "ANT001": "/path/to/beam1.fits"
       "ANT002": "/path/to/beam2.fits"
   ```

#### Beam File Requirements

- Format: pyuvdata UVBeam FITS files
- Must contain E-field (not power) beam data
- Should cover the observation's zenith angle range
- Should cover the observation's frequency range
- Coordinate system: AltAz with HEALPix or regular grid

#### Configuration Parameters

- `beam_za_max_deg`: Maximum zenith angle to load from beam file (default: 90.0°)
- `beam_za_buffer_deg`: Buffer around observation ZA range (default: 5.0°)
- `beam_freq_buffer_hz`: Buffer around observation frequencies (default: 1 MHz)

#### Polarization Support

When using beam FITS files, the simulator:
- Loads full 2×2 Jones matrices E(θ,φ,ν) for each antenna
- Converts source Stokes parameters (I, Q, U, V) to coherency matrices
- Computes visibilities using the full RIME: V_ij = E_i @ C @ E_j^H
- Returns correlation products: XX, XY, YX, YY, and total intensity I

Sources without polarization data (GLEAM, GSM) default to Q=U=V=0 (unpolarized).

#### Examples

See the example configurations in `configs/`:
- `09_shared_beam_fits.yaml`: Shared beam example with MWA
- `10_per_antenna_layout.yaml`: Per-antenna beams from layout file
- `11_per_antenna_config.yaml`: Per-antenna beams from config

### Visibility Calculation

The visibility calculation accounts for:
- Source positions and flux densities
- Beam patterns for each antenna (analytic or from FITS files)
- Full polarization through Jones matrices and coherency
- Spectral indices of sources
- Baseline vectors and geometric delays
- Time and frequency variations

## File Structure

```
.
├── .gitignore               # Git ignore rules
├── LICENSE                  # MIT License
├── pixi.lock                # Pixi lock file
├── pixi.toml               # Pixi project configuration
├── README.md               # Project documentation
├── configs/                # Example configuration files
│   ├── 01_parabolic_10db_taper.yaml
│   ├── ...
│   ├── 09_shared_beam_fits.yaml        # Shared beam FITS example
│   ├── 10_per_antenna_layout.yaml      # Per-antenna from layout
│   └── 11_per_antenna_config.yaml      # Per-antenna from config
├── docs/                   # Documentation
│   └── CONVENTIONS.md      # Polarization and coordinate conventions
├── src/                    # Source code
│   ├── __init__.py         # Python package initialization
│   ├── antenna.py         # Antenna position handling + BeamID support
│   ├── baseline.py        # Baseline generation
│   ├── beam_file.py       # Beam FITS file handling (NEW)
│   ├── beams.py           # Beam calculations and HPBW
│   ├── fits.py            # FITS file handling
│   ├── gsm_map.py         # GSM map integration
│   ├── main.html          # HTML interface
│   ├── main.py            # Entry point for the simulator
│   ├── observation.py     # Observation location and time
│   ├── plot.py            # Plotting functions
│   ├── polarization.py    # Polarization utilities (NEW)
│   ├── reading.py         # Reading data utilities
│   ├── source.py          # Source modeling + Stokes parameters
│   ├── visibility.py      # Visibility calculations with full RIME
│   ├── visibility_results.json # Output data
│   └── simulators/        # Simulator modules
│       ├── fftvis.py      # FFT-based visibility simulator
│       └── matvis.py      # Matrix-based visibility simulator
└── tests/                 # Unit tests
    ├── conftest.py        # Test configuration
    ├── test_antenna.py    # Antenna tests
    ├── test_baseline.py   # Baseline tests
    ├── test_beam_file.py  # Beam FITS tests (NEW)
    ├── test_main.py       # Main module tests
    ├── test_observation.py # Observation tests
    ├── test_plot.py       # Plot tests
    ├── test_polarization.py # Polarization tests (NEW)
    └── test_visibility.py # Visibility tests
```

## Development

### Requirements

- Python 3.11
- Key libraries: `numpy`, `astropy`, `healpy`, `matplotlib`, `bokeh`, `astroquery`, `pyyaml`, `pygsm`, `h5py`

### Running Tests

Run unit tests using `pytest`:

```bash
pixi run pytest
```

### Setting up the development environment

It is recommended to use `pixi` to manage the development environment.

1. Install `pixi` if you don't have it already:
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

2. Clone the repository and navigate to it:
   ```bash
   git clone <repository_url>
   cd RRIviz
   ```

3. Install dependencies:
   ```bash
   pixi install
   ```

4. Activate the development environment:
   ```bash
   pixi shell
   ```

## Antenna File Format

### Minimum Format

The antenna file must contain at least these columns (whitespace-separated):

```
Name  Number  BeamID  E  N  U
```

Where:
- `Name`: Antenna name (string)
- `Number`: Antenna number (integer)
- `BeamID`: Beam identifier (integer)
- `E`, `N`, `U`: East, North, and Up coordinates in meters

### Extended Format with Different Diameters

To use different diameters per antenna, add a header column `Diameter` and a numeric value in each row:

```
Name  Number  BeamID  E  N  U  Diameter
HH136 136     0       0.0 0.0 0.0  14.0
HH140 140     0       14.0 0.0 0.0 14.0
HH121 121     0       28.0 0.0 0.0 14.0
```

## Output Files

The simulator generates several output files:

1. **HDF5 File**: Contains complex visibility data and simulation parameters
2. **JSON File**: Contains visibility results in JSON format
3. **HTML Files**: Interactive plots generated with Bokeh
4. **Log File**: Contains simulation details and progress information
5. **Configuration File**: Copy of the configuration used for the simulation

## Future Development

### Planned Features

1. **Advanced Simulator Modules**: Implementation of FFT-based and matrix-based visibility simulators
2. **Additional Sky Models**: Integration with more sky catalogs and polarized source catalogs
3. **Enhanced Visualization**: Polarization-aware plotting (linear polarization vectors, fractional polarization maps)
4. **Performance Optimization**: GPU acceleration for RIME calculations and parallel processing
5. **Web Interface**: Full web-based interface for configuration and visualization
6. **Ionospheric Effects**: Faraday rotation and TEC-based phase delays
7. **Calibration Tools**: Beam calibration and direction-dependent gain calibration

### Contributing

Open-source contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Citation

If you use RRIviz in your research, please cite it as:

```
Mandar, K. (2025). RRIviz: Radio Astronomy Visibility Simulator. 
GitHub repository: https://github.com/<username>/RRIviz
```

## Acknowledgments

- The Hydrogen Epoch of Reionization Array (HERA) team for inspiration and testing
- Contributors to the scientific Python ecosystem (NumPy, Astropy, HEALPy, etc.)
- The radio astronomy community for feedback and suggestions