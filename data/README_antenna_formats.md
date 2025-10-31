# Antenna Layout File Formats

This directory contains example antenna layout files in various formats supported by RRIvis. Each format serves different purposes and has specific use cases.

## Supported Formats

### 1. **RRIvis Native Format** (`example_rrivis_format.txt`)
- **Extension**: `.txt` or `.csv`
- **Coordinate System**: East-North-Up (ENU) in meters
- **Recommended**: Yes, this is the native RRIvis format
- **Configuration Key**: `antenna_file_format: "rrivis"`

**Format**:
```
Name    Number  BeamID  E      N      U      [Diameter]
HH136   136     0       0.0    0.0    0.0    14.0
```

**Columns**:
- `Name`: Antenna identifier (string)
- `Number`: Unique antenna number (integer)
- `BeamID`: Beam model ID (integer)
- `E`: East coordinate (float, meters)
- `N`: North coordinate (float, meters)
- `U`: Up coordinate (float, meters)
- `Diameter`: Optional per-antenna diameter (float, meters)

**Use Cases**:
- Direct HERA array antenna positions
- Simple array configurations
- Custom antenna layouts with known ENU coordinates

---

### 2. **CASA Configuration Format** (`example_casa_format.cfg`)
- **Extension**: `.cfg`
- **Coordinate System**: LOC (Local Tangent Plane) or ITRF/XYZ
- **Primary Tool**: CASA `simobserve` and `simanalyze`
- **Configuration Key**: `antenna_file_format: "casa"`

**Format**:
```
#observatory=HERA
#COFA=-30.72152777777791,21.428305555555557
#coordsys=LOC
# x    y    z    diam  station  ant
0.0   0.0  0.0  14.0  HH136    HH136
```

**Header Lines** (optional but recommended):
- `#observatory=NAME`: Observatory name (string)
- `#COFA=lat,lon`: Center of array coordinates (latitude, longitude in degrees)
- `#coordsys=SYSTEM`: Coordinate system type
  - `LOC`: Local tangent plane (default)
  - `ENU`: East-North-Up
  - `XYZ`: ITRF Cartesian
  - `UTM`: Universal Transverse Mercator

**Data Columns**:
- `x, y, z`: Position coordinates (3 floats)
- `diam`: Antenna diameter in meters (optional)
- `station`: Station/pad name (string)
- `ant`: Antenna name (optional, defaults to station if not provided)

**Use Cases**:
- ALMA, VLA, EVLA observations
- CASA-compatible simulations
- Arrays with LOC or ITRF coordinates

---

### 3. **PyUVData Simple Text Format** (`example_pyuvdata_format.txt`)
- **Extension**: `.txt`
- **Coordinate System**: XYZ or ENU (context-dependent)
- **Simplicity**: Minimal format, no metadata
- **Configuration Key**: `antenna_file_format: "pyuvdata"`

**Format**:
```
# Comments start with #
0.0      0.0      0.0
14.0     0.0      0.0
28.0     0.0      0.0
```

**Columns**:
- `x, y, z`: Position coordinates (3 floats, space or tab separated)
- One antenna per line
- Antenna numbers auto-generated sequentially starting at 0
- Antenna names auto-generated as `ANT000`, `ANT001`, etc.

**Use Cases**:
- Quick prototyping
- Simple antenna arrays
- Data from external simulators
- Testing new configurations

---

### 4. **PyUVSim Antenna Layout CSV** (`example_antenna_layout.csv`)
- **Extension**: `.csv`
- **Coordinate System**: ENU in meters
- **Tool**: PyUVSim (sister project to RRIvis)
- **Configuration Key**: `antenna_file_format: "pyuvdata"` (uses simple format reader)

**Format**:
```
Name,Number,BeamID,E,N,U
ANT1,0,0,0.0,0.0,0.0
ANT2,1,0,50.0,0.0,0.0
```

**Columns**:
- `Name`: Antenna name (string)
- `Number`: Unique antenna index (integer)
- `BeamID`: Beam model ID (integer)
- `E, N, U`: East-North-Up coordinates (floats, meters)

**Use Cases**:
- PyUVSim compatibility
- Standard CSV format for data interchange
- Programmatically generated antenna layouts

---

### 5. **PyUVSim Telescope Configuration YAML** (`example_telescope_config.yaml`)
- **Extension**: `.yaml` or `.yml`
- **Purpose**: Beam definitions and telescope metadata
- **Used With**: Antenna layout CSV files
- **Not directly read as antenna_file_format** but essential for full simulations

**Key Sections**:
```yaml
beam_paths:
  0: !AnalyticBeam
    class: GaussianBeam
    diameter: 14.0
telescope_location: (lat, lon, alt)
telescope_name: NAME
```

**Beam Types**:
- `!UVBeam`: FITS beam files
- `!AnalyticBeam`: Parameterized beams
  - `GaussianBeam`: Gaussian illumination
  - `AiryBeam`: Airy disk pattern
  - `UniformBeam`: Uniform response
  - `ShortDipoleBeam`: Classical dipole

**Use Cases**:
- Full array characterization
- Multiple beam types per array
- Frequency-dependent beam effects

---

### 6. **MWA Metafits Format** (`example_mwa_metafits.txt`)
- **Extension**: `.txt` or `.metafits`
- **Coordinate System**: ENU or geodetic coordinates
- **Primary Array**: Murchison Widefield Array (MWA)
- **Configuration Key**: `antenna_file_format: "mwa"`

**Format**:
```
# coordinates (m) East North Elevation
Tile001MWA  4.0  m  +116.40.09.5  -26.32.48.3  -150.16  264.84  376.90
```

**Columns**:
- Antenna/Tile name (string)
- Diameter with unit (e.g., "4.0 m")
- Latitude (DMS or decimal)
- Longitude (DMS or decimal)
- Elevation/Z coordinate (float)
- X, Y, Z ITRF coordinates (optional)

**Coordinate Formats**:
- DMS: `+116.40.09.5` = 116° 40' 9.5"
- Decimal: `116.668194`

**Use Cases**:
- MWA observations and simulations
- Phased array antenna positions
- Tiles in compact arrays

---

## Format Comparison Table

| Feature | RRIvis | CASA | PyUVData | MWA |
|---------|--------|------|----------|-----|
| **Native Coordinate** | ENU | LOC/XYZ | XYZ/ENU | ENU/DMS |
| **Diameter Support** | Per-antenna | Per-antenna | None | Optional |
| **Metadata** | Name, Number, BeamID | Station info | None | Tile name only |
| **Simplicity** | High | Medium | Very High | Low |
| **CASA Compatible** | No | Yes | No | No |
| **PyUVSim Compatible** | Limited | No | Yes | No |

---

## Coordinate System Reference

### ENU (East-North-Up)
- Local tangent plane coordinates
- Origin at array reference point
- E: East direction (positive east)
- N: North direction (positive north)
- U: Up direction (positive upward)
- Unit: meters

### LOC (Local Tangent Plane)
- Similar to ENU but may have different conventions
- Used in CASA
- Often: X=East, Y=North, Z=Up or Height

### ITRF / XYZ
- Earth-centered Cartesian coordinates
- International Terrestrial Reference Frame
- X: Through 0° latitude, 0° longitude
- Y: Perpendicular to X, in equatorial plane
- Z: Through North Pole
- Unit: meters

### Geodetic (Lat/Lon/Alt)
- Latitude: degrees (-90 to +90, negative = south)
- Longitude: degrees (-180 to +180, negative = west)
- Altitude: meters above ellipsoid

---

## Usage Examples

### Loading in RRIvis

```python
from src.antenna import read_antenna_positions

# RRIvis format
antennas = read_antenna_positions("data/example_rrivis_format.txt", "rrivis")

# CASA format
antennas = read_antenna_positions("data/example_casa_format.cfg", "casa")

# PyUVData simple format
antennas = read_antenna_positions("data/example_pyuvdata_format.txt", "pyuvdata")

# MWA format
antennas = read_antenna_positions("data/example_mwa_metafits.txt", "mwa")
```

### In Configuration File

```yaml
antenna_layout:
  antenna_positions_file: "/path/to/data/example_rrivis_format.txt"
  antenna_file_format: "rrivis"
  use_different_diameters: false
  all_antenna_diameter: 14.0
```

---

## Creating Your Own Files

### From Existing Data

If you have antenna coordinates in another format, use CASA tools:

```python
import analysisUtils as au

# Convert from Measurement Set
au.buildConfigurationFile('observation.ms')
# Creates: observation.ms.cfg
```

### From Known Telescopes

Use PyUVData to extract antenna positions:

```python
from pyuvdata import Telescope

# Load known telescope
hera = Telescope.from_known_telescopes('HERA')

# Extract positions and save
antenna_layout = hera.antenna_positions  # ENU coordinates
```

---

## Notes and Warnings

1. **Coordinate Precision**: Maintain at least meter-level precision for antenna positions
2. **Consistent Units**: All coordinates must be in the same units (typically meters)
3. **Array Reference**: ENU coordinates are relative to array center (specified as COFA in CASA)
4. **Diameter Convention**: Diameter of 0 or negative may cause issues
5. **Beam IDs**: Must match available beam definitions in configuration
6. **Comments**: Use `#` for comment lines in text-based formats

---

## Related Files

- `example_telescope_config.yaml`: Beam and telescope metadata
- `../../src/antenna.py`: Format reader implementations
- `../../README.md`: General RRIvis documentation
