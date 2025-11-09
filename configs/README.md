# RRIvis Configuration Examples

This directory contains example configuration files demonstrating different antenna types, beam patterns, and simulation scenarios.

## Quick Start

### Run a Single Configuration
```bash
python src/main.py --config configs/01_parabolic_10db_taper.yaml
```

### Run All Configurations (Batch Mode)
```bash
python src/main.py --config-dir configs/
```

### Run Specific Configurations
```bash
python src/main.py --configs \
  configs/01_parabolic_10db_taper.yaml \
  configs/03_parabolic_airy_pattern.yaml \
  configs/04_dipole_array.yaml
```

---

## Configuration Files

### 📡 **01_parabolic_10db_taper.yaml** ⭐ RECOMMENDED
**Antenna Type:** Parabolic dishes with 10dB edge taper
**Beam Pattern:** Gaussian approximation
**Use Case:** Standard configuration for most radio telescope observations

**Characteristics:**
- HPBW = 1.15 * λ/D
- ~75% aperture efficiency
- ~-30 dB first sidelobe
- Excellent balance of gain and low sidelobes

**Best For:** Production observations, realistic simulations

---

### 📡 **02_parabolic_uniform.yaml**
**Antenna Type:** Parabolic dishes with uniform illumination
**Beam Pattern:** Gaussian approximation
**Use Case:** Theoretical studies, maximum efficiency scenarios

**Characteristics:**
- HPBW = 1.02 * λ/D (narrowest beam)
- 100% illumination efficiency
- High sidelobes (-17.6 dB)
- Not realistic for actual feeds

**Best For:** Theoretical studies, sensitivity limits

---

### 📡 **03_parabolic_airy_pattern.yaml**
**Antenna Type:** Parabolic dishes with uniform illumination
**Beam Pattern:** Airy disk (accurate Bessel function)
**Use Case:** High-accuracy simulations requiring precise sidelobe modeling

**Characteristics:**
- Accurate Airy pattern: A(θ) = [2*J₁(x)/x]²
- First null at θ = 1.22 * λ/D
- Realistic sidelobe structure
- More computationally expensive

**Best For:** Sidelobe studies, RFI analysis, imaging simulations

---

### 📡 **04_dipole_array.yaml**
**Antenna Type:** Short dipoles (HERA/MWA/LOFAR style)
**Beam Pattern:** Gaussian approximation
**Use Case:** Low-frequency arrays, dipole-based instruments

**Characteristics:**
- HPBW = π/2 radians = 90° (CONSTANT)
- Frequency-independent beamwidth
- Omnidirectional in H-plane
- Figure-8 pattern in E-plane

**Best For:** HERA simulations, low-frequency studies (< 200 MHz)

---

### 📡 **05_spherical_reflector.yaml**
**Antenna Type:** Spherical reflector (FAST/Arecibo style)
**Beam Pattern:** Gaussian approximation
**Use Case:** Spherical reflector telescopes with active surfaces

**Characteristics:**
- HPBW = 1.20 * λ/D (wider due to spherical aberration)
- Gaussian taper for realistic illumination
- Active surface or line feed

**Best For:** FAST simulations, large single-dish studies

---

### 📡 **06_parabolic_gaussian_taper.yaml**
**Antenna Type:** Parabolic dishes with Gaussian feed taper
**Beam Pattern:** Gaussian approximation
**Use Case:** Modern feed designs, low-sidelobe requirements

**Characteristics:**
- HPBW = 1.18 * λ/D
- Very low sidelobes (-40+ dB)
- ~75% aperture efficiency
- Smooth rolloff

**Best For:** RFI mitigation, high-dynamic-range imaging

---

### 📡 **07_cosine_tapered_beam.yaml**
**Antenna Type:** Parabolic dishes with cosine taper
**Beam Pattern:** Cosine-tapered (Cassegrain/offset feeds)
**Use Case:** Cassegrain or offset-fed reflectors

**Characteristics:**
- HPBW = 1.10 * λ/D
- Cosine pattern: A(θ) = cos^n(θ/θ_edge)
- ~81% aperture efficiency
- ~-23 dB first sidelobe

**Best For:** Cassegrain dishes, offset reflectors

---

### 📡 **08_mixed_antenna_types.yaml** 🔬 ADVANCED
**Antenna Type:** Multiple types per antenna
**Beam Pattern:** Multiple patterns per antenna
**Use Case:** Heterogeneous arrays, comparative studies

**Demonstrates:**
- Per-antenna type configuration
- Per-antenna beam pattern selection
- Mixed array simulations

**Best For:** Testing, comparisons, heterogeneous arrays

---

## Antenna Layout Files

The configs use different antenna layout files from `data/`:

| Config | Antenna File | Format | Antennas |
|--------|--------------|--------|----------|
| 01, 03, 05, 06, 07, 08 | `example_rrivis_format.txt` | RRIvis native | 10 antennas in grid |
| 02 | `example_casa_format.cfg` | CASA | 10 antennas in grid |
| 04 | `example_pyuvdata_format.txt` | PyUVData | 10 antennas in grid |

All layouts represent the same 10-antenna grid configuration.

---

## Antenna Types Available

| Type | HPBW Formula | Efficiency | Sidelobes | Use Case |
|------|--------------|------------|-----------|----------|
| `parabolic_uniform` | 1.02 * λ/D | 100% | -17.6 dB | Theoretical |
| `parabolic_cosine` | 1.10 * λ/D | 81% | -23 dB | Cassegrain |
| `parabolic_gaussian` | 1.18 * λ/D | 75% | -40 dB | Low sidelobes |
| `parabolic_10db_taper` | 1.15 * λ/D | 75% | -30 dB | **Standard** ⭐ |
| `parabolic_20db_taper` | 1.25 * λ/D | 62% | -42 dB | Very low sidelobes |
| `spherical_uniform` | 1.05 * λ/D_eff | Varies | Varies | Spherical reflector |
| `spherical_gaussian` | 1.20 * λ/D_eff | Varies | Varies | FAST/Arecibo |
| `phased_array` | 1.10 * λ/D_eff | Varies | Varies | Electronically steered |
| `dipole_short` | π/2 (90°) | N/A | N/A | HERA/MWA/LOFAR |
| `dipole_halfwave` | 78° | N/A | N/A | Half-wave dipole |
| `dipole_folded` | 78° | N/A | N/A | Folded dipole |

---

## Beam Response Patterns

| Pattern | Description | Accuracy | Speed | Use Case |
|---------|-------------|----------|-------|----------|
| `gaussian` | Gaussian approximation | Good | **Fast** | Standard ⭐ |
| `airy` | Airy disk (Bessel) | **Excellent** | Slow | High precision |
| `cosine` | Cosine-tapered | Good | Fast | Cassegrain feeds |
| `exponential` | Exponential taper | Good | Fast | Feed horns |

---

## Parameter Sweeps

Create parameter sweeps by modifying configs:

### Frequency Sweep
Edit `obs_frequency` section:
```yaml
obs_frequency:
  starting_frequency: 50   # Start at 50 MHz
  frequency_interval: 5    # 5 MHz steps
  frequency_bandwidth: 100 # 100 MHz total bandwidth
  frequency_unit: "MHz"
```

### Time Sweep
Edit `obs_time` section:
```yaml
obs_time:
  start_time: "2025-01-15T00:00:00"
  duration_seconds: 7200    # 2 hours
  time_step_seconds: 300    # 5-minute steps
```

### Sky Model Variation
Enable different sky models:
```yaml
sky_model:
  gleam:
    use_gleam: true          # Use GLEAM catalog
    gleam_catalogue: "VIII/105/catalog"
    flux_limit: 10           # Lower limit for more sources
```

---

## Batch Processing Tips

### 1. Quick Test Run (All Configs)
```bash
python src/main.py --config-dir configs/
```

### 2. Compare Antenna Types
```bash
python src/main.py --configs \
  configs/01_parabolic_10db_taper.yaml \
  configs/02_parabolic_uniform.yaml \
  configs/06_parabolic_gaussian_taper.yaml
```

### 3. Compare Beam Patterns
```bash
python src/main.py --configs \
  configs/01_parabolic_10db_taper.yaml \
  configs/03_parabolic_airy_pattern.yaml \
  configs/07_cosine_tapered_beam.yaml
```

### 4. Different Instrument Types
```bash
python src/main.py --configs \
  configs/01_parabolic_10db_taper.yaml \
  configs/04_dipole_array.yaml \
  configs/05_spherical_reflector.yaml
```

---

## Modifying Configs

All configs are minimal - they only specify what differs from defaults.
Missing fields use values from `DEFAULT_CONFIG` in `src/main.py`.

### To customize:
1. Copy an existing config
2. Modify only the fields you want to change
3. Run your custom config

### Example: Quick frequency change
```bash
# Edit config, change:
obs_frequency:
  starting_frequency: 150  # Changed from 100
  frequency_bandwidth: 100 # Changed from 50
```

---

## Output Organization

Each config produces a separate output folder:

**Single config:**
```
RRIVis_simulation_data/
  └── RRIVis_20250102_143022/
```

**Batch mode:**
```
RRIVis_simulation_data/
  ├── RRIVis_batch_01_parabolic_10db_taper_20250102_143022/
  ├── RRIVis_batch_02_parabolic_uniform_20250102_143045/
  ├── RRIVis_batch_03_parabolic_airy_pattern_20250102_143112/
  └── ... (one folder per config)
```

---

## Notes

- All configs use the HERA location by default (-30.72°, 21.43°, 1073m)
- All use 10 test sources for fast simulation
- Modify `sky_model` section to use GLEAM or GSM catalogs
- Set `save_simulation_data: false` for testing without saving
- See `src/main.py` DEFAULT_CONFIG for all available options

---

## Need Help?

```bash
python src/main.py --help
```

For detailed documentation, see: `README.md` in project root
