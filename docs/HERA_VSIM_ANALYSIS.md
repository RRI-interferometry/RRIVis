# Analysis of HERA vsim.py Validation Simulation

## Overview

This document analyzes a HERA (Hydrogen Epoch of Reionization Array) validation simulation run using vsim.py with the matvis simulator. This serves as a reference for creating equivalent configurations in RRIvis.

## Command Overview

The simulation consists of two steps:

```bash
# Step 1: Setup observation parameters
./vsim.py make-obsparams --ants 0~9 --channels 227~257 --sky-model ptsrc256 \
  --n-time-chunks 288 --do-time-chunks 0~3 --simulator matvis \
  --beam-map-csv beam_map_10.csv

# Step 2: Run simulation
./vsim.py runsim --ants 0~9 --channels 227~257 --sky-model ptsrc256 \
  --n-time-chunks 288 --do-time-chunks 0~3 --simulator matvis \
  --beam-map-csv beam_map_10.csv
```

## Detailed Parameter Analysis

### Antennas: `--ants 0~9`

**Configuration:**
- Using 10 antennas numbered 0-9
- Output shows 9 ENU positions (0-8)
- Linear array configuration

**Antenna Positions (ENU coordinates in meters - exact values):**
```python
array([[-105.03529563, -110.72150481,    0.81120331],  # ant 0
       [ -90.42745642, -110.66571205,    0.81868112],  # ant 1
       [ -75.81961721, -110.60991928,    0.82615894],  # ant 2
       [ -61.21177801, -110.55412652,    0.83363676],  # ant 3
       [ -46.6039388 , -110.49833375,    0.84111458],  # ant 4
       [ -31.99609959, -110.44254098,    0.84859239],  # ant 5
       [ -17.38826039, -110.38674822,    0.85607021],  # ant 6
       [  -2.78042118, -110.33095545,    0.86354803],  # ant 7
       [  11.82741802, -110.27516269,    0.87102585]]) # ant 8
```

**Array Characteristics:**
- **Baseline orientation:** Nearly east-west (X varies, Y ~constant)
- **Spacing:** ~14.6 m between consecutive antennas
- **Total extent:** ~117 m (from ant 0 to ant 8)
- **Elevation variation:** Only ~6 cm across array (very flat)
- **Typical HERA configuration:** Subset of hex-array layout

### Frequency: `--channels 227~257`

**Configuration:**
- **30 frequency channels** from full HERA band (channels 227-257)
- **Channel 227:** 74.6307373046875 MHz
- **Channel 256:** 78.0487060546875 MHz
- **Channel 257:** 78.1707763671875 MHz
- **Channel width:** 0.12207031 MHz (122.07031 kHz)
- **Total bandwidth:** ~3.66 MHz

**Frequency Array (MHz - exact values):**
```
[74.6307373046875, 74.7528076171875, 74.8748779296875, 74.9969482421875,
 75.1190185546875, 75.2410888671875, 75.3631591796875, 75.4852294921875,
 75.6072998046875, 75.7293701171875, 75.8514404296875, 75.9735107421875,
 76.0955810546875, 76.2176513671875, 76.3397216796875, 76.4617919921875,
 76.5838623046875, 76.7059326171875, 76.8280029296875, 76.9500732421875,
 77.0721435546875, 77.1942138671875, 77.3162841796875, 77.4383544921875,
 77.5604248046875, 77.6824951171875, 77.8045654296875, 77.9266357421875,
 78.0487060546875, 78.1707763671875]
```

**Note:** Output shows 30 frequencies, suggesting channels 227-256 (inclusive), with channel 257 also shown.

**Scientific Context:**
- **EoR frequency range:** 75-78 MHz corresponds to redshifted 21cm line
- **Redshift z ≈ 17-18:** Cosmic dawn / early EoR
- **Good for:** Foreground characterization, calibration validation
- **Lower than RRI:** RRI typically observes ~10 MHz (ionospheric physics)

### Time Sampling: `--n-time-chunks 288 --do-time-chunks 0~3`

**Configuration:**
- **Total observation:** Divided into 288 time chunks
- **Simulating:** Only chunks 0, 1, 2, 3 (first 4 chunks)
- **Integration time:** 4.986347833333333 seconds per sample (exact)
- **Samples per chunk:** ~15 time samples
- **Total samples shown:** 60 (4 chunks × 15 samples)

**LST Coverage (hours - exact values):**
```
Start: 0.08333419 hours (5 minutes 0.0 seconds)
End:   0.16527863 hours (9 minutes 55.0 seconds)
Duration: 0.08194444 hours (~4.92 minutes = 295.0 seconds)
```

**Complete LST Array (60 values in hours):**
```
[0.08333419, 0.08472308, 0.08611198, 0.08750086, 0.08888975, 0.09027864,
 0.09166753, 0.09305641, 0.09444531, 0.09583419, 0.09722308, 0.09861197,
 0.10000086, 0.10138975, 0.10277864, 0.10416752, 0.10555642, 0.1069453 ,
 0.10833419, 0.10972309, 0.11111197, 0.11250085, 0.11388975, 0.11527864,
 0.11666752, 0.11805642, 0.1194453 , 0.12083419, 0.12222308, 0.12361197,
 0.12500086, 0.12638975, 0.12777863, 0.12916753, 0.13055641, 0.1319453 ,
 0.13333419, 0.13472308, 0.13611196, 0.13750086, 0.13888974, 0.14027863,
 0.14166753, 0.14305641, 0.1444453 , 0.14583419, 0.14722308, 0.14861196,
 0.15000086, 0.15138974, 0.15277863, 0.15416752, 0.15555641, 0.1569453 ,
 0.15833419, 0.15972307, 0.16111197, 0.16250085, 0.16388974, 0.16527863]
```

**Full Observation Extrapolation:**
- **If all 288 chunks were run:**
  - Total time samples: 288 × 15 = 4,320
  - Total duration: 4,320 × 5 s = 21,600 s = **6 hours**
  - LST coverage: ~6 hours of sky rotation

**Why 5-second integrations?**
- HERA correlator typically outputs ~10s integrations
- Shorter integrations useful for:
  - Fringe rate testing
  - RFI flagging validation
  - Fast transient detection tests
  - Higher time resolution for calibration

**Why only 4 chunks?**
- **Test run:** Validating pipeline before full run
- **Parallel processing:** Different chunks might run on different cluster nodes
- **Development:** Testing systematics on subset
- **Computational cost:** Full simulation expensive

### Sky Model: `--sky-model ptsrc256`

**Configuration:**
- **Point source catalog** with 256 sources (ptsrc256)
- **File location (original):** `/lustre/aoc/projects/hera/rchandra/H6C_Validation_Stats/validation-sim/sky_models/ptsrc256/*.skyh5`
- **File location (RRIvis):** `data/fch0227.skyh5` and `data/fch0227_full.skyh5`
- **Format:** `.skyh5` (pyradiosky standard format)

**Sky Model Characteristics:**
- **Type:** Discrete point sources (not diffuse emission)
- **Purpose:**
  - Calibration source catalog
  - Realistic sky model for validation
  - Known input for testing pipeline recovery
- **Includes:**
  - ~785k point sources from GLEAM catalog
  - Spectral type: full (all Stokes at each frequency)
  - Reference frequency: 74.6307 MHz (channel 227)
  - Frame: ICRS

**Available Sky Model Files:**

Two versions available in `data/`:

1. **`fch0227_full.skyh5`** (44 MB) - ⚠️ NOT RECOMMENDED
   - **785,679 sources** (full catalog)
   - **Flux range:** 0.019 to **17,826.78 Jy**
   - **Issue:** Brightest source causes absurdly high |V| values
   - **Problem:** Ultra-bright sources crossing beam/horizon create artifacts
   - **Use case:** Only for testing extreme cases

2. **`fch0227.skyh5`** (40 MB) - ✅ RECOMMENDED
   - **707,111 sources** (78,568 removed = 10% filtered)
   - **Flux range:** 0.023 to **0.45 Jy** (capped)
   - **Filtering:** Brightest and dimmest 5% removed
   - **Benefit:** Eliminates numerical issues and artifacts
   - **Unpolarized:** Stokes Q=U=V=0, only I populated
   - **Created by:** Rajorshi using `validation-sim/core/sky_model.py`

**Critical Finding (from Rajorshi Chandra):**
> "I was getting absurdly high |V| as well, but that got resolved after I cut out
> top 5% brightest pt sources at channel 227 (~75 MHz)"

**Sky Model Analysis:**
```
Filtered vs Full comparison:
- Sources removed: 78,568 (10%)
- Brightest source in full: 17,827 Jy
- Brightest source in filtered: 0.45 Jy
- Brightness ratio: 39,392× difference
- Result: High |V| artifacts eliminated in filtered version
```

**For RRIvis Implementation:**
- ✅ Use `data/fch0227.skyh5` (filtered version)
- Need to convert `.skyh5` to RRIvis source format, OR
- Load directly using pyradiosky and extract:
  - RA, Dec positions (ICRS frame)
  - Stokes I, Q, U, V fluxes
  - Reference frequency: 74.6307 MHz

### Beam Model: `--beam-map-csv beam_map_10.csv`

**Configuration:**
- **Per-antenna beam assignment** from CSV file
- **Beam file pattern:** `NF_HERA_Vivaldi_efield_beam_extrap_i.fits`
- **File location (RRIvis):** `data/NF_HERA_Vivaldi_efield_beam_extrap_0.fits` (821 MB)

**Beam File Components:**
- **NF:** Nearfield HERA Vivaldi antenna
- **Vivaldi:** HERA's dual-polarized Vivaldi antenna design
- **efield:** E-field beam (Jones matrices, not power pattern)
- **extrap:** Extrapolated to horizon (important for wide-field)
- **i:** Index indicating per-antenna beams (different orientations/patterns)

**Available Beam Files in `data/`:**

1. **`NF_HERA_Vivaldi_efield_beam_extrap_0.fits`** (821 MB)
   - HERA Vivaldi antenna E-field pattern
   - Extrapolated to horizon
   - UVBeam format (pyuvdata)
   - Jones matrix (2×2 complex per direction)
   - Frequency coverage: ~50-250 MHz
   - Coordinate system: Azimuth-Zenith Angle

2. **`NF_HERA_Vivaldi_efield_beam_healpix.fits`** (605 MB)
   - Same beam in HEALPix pixelization
   - Alternative format for all-sky coverage
   - More efficient for horizon extrapolation

**Beam Map CSV Format:**
```csv
antenna_number,beam_file
0,NF_HERA_Vivaldi_efield_beam_extrap_0.fits
1,NF_HERA_Vivaldi_efield_beam_extrap_1.fits
...
9,NF_HERA_Vivaldi_efield_beam_extrap_9.fits
```

**Why Per-Antenna Beams?**
- **Mutual coupling:** Antennas affect each other's patterns
- **Orientation differences:** Small pointing/rotation variations
- **Position-dependent effects:** Edge antennas vs. interior antennas
- **Realistic systematics:** Validation requires accurate beam models

**For RRIvis Implementation:**
- ✅ Beam files available in `data/` folder
- Requires `beam_mode: "per_antenna"` or `"shared"`
- For shared mode: All antennas use `beam_extrap_0.fits`
- For per-antenna: Need BeamID column in antenna layout file, OR
- Specify antenna_beam_map in config

### Simulator: `--simulator matvis`

**About matvis:**
- **Matrix-based visibility simulator**
- **Fast computation** via matrix operations
- **GPU-capable** for large-scale simulations
- **Exact RIME:** Implements V_ij = J_i * C * J_j^H

**Advantages:**
- Vectorized operations (NumPy/CuPy)
- Efficient for many sources
- Well-tested in HERA pipeline
- Same algorithm as RRIvis polarization module

## What This Simulation Represents

### HERA H6C Validation Run

This is a **validation simulation** for HERA's H6C data release:

1. **Pipeline Testing**
   - Validate calibration algorithms
   - Test imaging pipelines
   - Verify delay spectrum methods

2. **Systematics Understanding**
   - Beam chromaticity effects
   - Mutual coupling impact
   - Foreground leakage

3. **Known Truth Comparison**
   - Input: Known sky model + beam model
   - Output: Simulated visibilities
   - Recovery: Calibrate and compare

4. **Parallel Processing Strategy**
   - Chunk 0-3 on one node
   - Other chunks on other nodes
   - Combine results later

## Key Differences from Typical RRI Observations

| Parameter | HERA vsim.py | Typical RRI |
|-----------|--------------|-------------|
| **Frequency** | 74-78 MHz (EoR) | ~10 MHz (ionosphere) |
| **Bandwidth** | 3.66 MHz | ~1-2 MHz |
| **Integration** | 5 seconds | Variable (1-10s) |
| **Array config** | Linear/hex subset | Can vary |
| **Beam model** | Per-antenna FITS | Analytic or simple FITS |
| **Sky model** | 256 point sources | Fewer bright sources |
| **Science goal** | Cosmology (EoR) | Ionospheric physics |

## Available Data Files in RRIvis

All necessary files for replicating Rajorshi's HERA validation simulation are available in the `data/` folder.

### Complete File Inventory

```
data/
├── Sky Models (pyradiosky .skyh5 format)
│   ├── fch0227.skyh5                              (40 MB)  ✅ RECOMMENDED
│   │   └── 707,111 sources, flux: 0.023-0.45 Jy (filtered)
│   └── fch0227_full.skyh5                         (44 MB)  ⚠️  NOT RECOMMENDED
│       └── 785,679 sources, flux: 0.019-17,827 Jy (causes high |V|)
│
├── Beam Models (UVBeam FITS format)
│   ├── NF_HERA_Vivaldi_efield_beam_extrap_0.fits (821 MB) ✅ PRIMARY
│   │   └── HERA Vivaldi E-field, extrapolated, Az-ZA coords
│   └── NF_HERA_Vivaldi_efield_beam_healpix.fits  (605 MB) ⚠️  ALTERNATIVE
│       └── Same beam in HEALPix pixelization
│
├── Antenna Layouts
│   ├── antenna.txt                                (226 B)
│   │   └── 3 HERA antennas (HH136, HH140, HH121) with BeamID
│   └── HERA65_layout.csv                          (3.7 KB)
│       └── 65 HERA antennas with BeamID=0 (shared beam)
│
└── Configuration
    └── config.yaml                                (495 B)
        └── Old RRIvis config format (needs updating)
```

### Sky Model Files

**Recommended: `data/fch0227.skyh5`**
```python
Format: HDF5 (.skyh5, pyradiosky)
Sources: 707,111 point sources
Frequency: 74.6307373046875 MHz (channel 227)
Flux range: 0.023 - 0.45 Jy
Polarization: Unpolarized (Q=U=V=0, I only)
Coordinate frame: ICRS
Filtering: Top and bottom 5% sources removed
Purpose: Eliminates high |V| artifacts from ultra-bright sources
```

**Not Recommended: `data/fch0227_full.skyh5`**
```python
Format: HDF5 (.skyh5, pyradiosky)
Sources: 785,679 point sources
Frequency: 74.6307373046875 MHz (channel 227)
Flux range: 0.019 - 17,826.78 Jy (!)
Issue: Brightest source is 39,392× brighter than filtered cap
Problem: Creates numerical overflow and beam artifacts
Use case: Only for testing extreme scenarios
```

### Beam Files

**Primary: `data/NF_HERA_Vivaldi_efield_beam_extrap_0.fits`**
```python
Format: UVBeam FITS (pyuvdata)
Size: 821 MB
Antenna: HERA Vivaldi dual-pol
Beam type: E-field (Jones matrices)
Coordinate system: Azimuth-Zenith Angle
Frequency range: ~50-250 MHz
Polarizations: 2 feeds × 2 basis vectors
Extrapolation: To horizon (critical for wide-field)
Use: Shared beam mode (all antennas) or per-antenna beam 0
```

**Alternative: `data/NF_HERA_Vivaldi_efield_beam_healpix.fits`**
```python
Format: UVBeam FITS (pyuvdata) with HEALPix
Size: 605 MB
Pixelization: HEALPix (better for all-sky)
Content: Same beam as extrap_0.fits
Advantage: More efficient horizon handling
Use: Alternative to Az-ZA format
```

### Antenna Layout Files

**`data/antenna.txt`** (3 antennas)
```
Format: Space-separated columns
Columns: Name Number BeamID E N U
Antennas: 3 (HH136, HH140, HH121)
Coordinates: ENU (meters)
BeamID: All set to 0 (shared beam)
Purpose: Small subset for testing
```

Example content:
```
Name  Number   BeamID   E          N          U
HH136      136        0  -156.5976     2.9439    -0.1819
HH140      140        0   -98.1662     3.1671    -0.3008
HH121      121        0   -90.8139    -9.4618    -0.1707
```

**`data/HERA65_layout.csv`** (65 antennas)
```
Format: Space-separated columns
Columns: Name Number BeamID E N U
Antennas: 65 (full HERA core)
Coordinates: ENU (meters)
BeamID: All set to 0 (shared beam)
Purpose: Full HERA hex-array core
```

### Configuration File

**`data/config.yaml`** (Old format - needs updating)
```yaml
Current format: Legacy RRIvis config
Status: Needs updating to new schema
Contains:
  - Antenna layout file path
  - Telescope: HERA (using pyuvdata)
  - Output directory
  - Sky model: Test sources (not .skyh5 files)

Needs updating to:
  - New beam configuration (beam_mode, beam_file_path)
  - Sky model .skyh5 file paths
  - Observation parameters (LST, duration, frequency)
  - Polarization settings
```

### File Usage Matrix

| Simulation Type | Sky Model | Beam File | Antenna Layout | Config |
|----------------|-----------|-----------|----------------|---------|
| **Quick test (3 ant)** | fch0227.skyh5 | beam_extrap_0.fits | antenna.txt | New config needed |
| **Full HERA core** | fch0227.skyh5 | beam_extrap_0.fits | HERA65_layout.csv | New config needed |
| **Extreme test** | fch0227_full.skyh5 | beam_extrap_0.fits | antenna.txt | New config needed |
| **HEALPix test** | fch0227.skyh5 | beam_healpix.fits | antenna.txt | New config needed |

### Data Sources

**Original locations:**
- Sky models: `/lustre/aoc/projects/hera/rchandra/H6C_Validation_Stats/validation-sim/sky_models/ptsrc256/`
- Beam files: HERA collaboration beam repository
- Antenna layouts: HERA telescope configuration files

**Created by:**
- Sky model filtering: Rajorshi Chandra (validation-sim/core/sky_model.py)
- Beam models: HERA collaboration (CST simulations + nearfield measurements)
- Antenna layouts: HERA survey team

### Known Issues & Solutions

**Issue 1: High |V| values**
- **Cause:** Ultra-bright sources in `fch0227_full.skyh5`
- **Solution:** Use `fch0227.skyh5` (filtered version)
- **Evidence:** Rajorshi confirmed this resolves the issue

**Issue 2: Phase wrapping**
- **Cause:** Not using `np.unwrap()` for visibility phases
- **Solution:** Apply `np.unwrap()` to phase arrays
- **Effect:** Prevents discontinuous jumps from -π to +π

**Issue 3: Beam horizon artifacts**
- **Cause:** Non-extrapolated beams create artifacts at low elevations
- **Solution:** Use `beam_extrap_0.fits` (already extrapolated)
- **Benefit:** Smooth beam pattern to horizon

## Requirements for RRIvis Implementation

To replicate this simulation in RRIvis, you need:

### 1. Antenna Configuration
- [x] Parse antenna ENU positions
- [x] Support custom antenna layouts
- [ ] Handle antenna numbering vs. naming

### 2. Beam Support
- [x] Per-antenna beam assignment
- [x] Jones matrix (E-field) beams
- [x] FITS file loading
- [ ] Beam interpolation at multiple frequencies
- [ ] Azimuth/zenith angle conversion

### 3. Sky Model
- [ ] Load .skyh5 files (or convert externally)
- [ ] Point source catalog with Stokes parameters
- [ ] Spectral index support
- [ ] Multiple sources (256+)

### 4. Frequency/Time
- [x] Multi-channel support
- [x] Integration time specification
- [x] LST calculation
- [ ] Chunk-based processing

### 5. Visibility Calculation
- [x] Full Jones matrix RIME
- [x] Coherency matrix formalism
- [ ] Efficient multi-source summation
- [ ] Output in standard format (HDF5/FITS)

## Computational Considerations

**Simulation Scale:**
- **Baselines:** C(9,2) + 9 = 36 + 9 = 45 (with autocorrs)
- **Frequencies:** 30 channels
- **Times:** 60 samples (or 4320 for full run)
- **Sources:** 256 point sources
- **Total visibilities:** 45 × 30 × 60 = 81,000 (partial run)
- **Full run:** 45 × 30 × 4320 = 5,832,000 visibilities

**Memory Requirements:**
- Complex visibility: 16 bytes (complex128)
- Full pol (4 correlations): 64 bytes per vis
- Partial run: 81k × 64 = 5.2 MB
- Full run: 5.8M × 64 = 373 MB

**Why matvis is fast:**
- Matrix operations avoid loops over baselines
- GPU acceleration possible
- Efficient beam interpolation

## References

- **HERA:** https://reionization.org/
- **matvis:** https://github.com/HERA-Team/matvis
- **pyradiosky:** https://github.com/RadioAstronomySoftwareGroup/pyradiosky
- **H6C Data Release:** HERA collaboration papers

## Notes for Config Creation

When creating equivalent RRIvis config:

1. **Start time:** Use LST 0.0833 hrs, convert to UTC given HERA location
2. **Duration:** 299 seconds (60 samples × 4.986s)
3. **Frequency:** Center at 76.34 MHz, bandwidth 3.66 MHz
4. **Antennas:** Create layout file with 9 positions + BeamID column
5. **Beams:** Point to HERA Vivaldi FITS files (need to obtain)
6. **Sources:** Convert ptsrc256.skyh5 or create equivalent catalog
7. **Output:** Match vsim.py visibility format for validation

## Validation Strategy

To validate RRIvis against vsim.py:

1. **Run both simulators** with identical inputs
2. **Compare visibilities:**
   - Amplitude differences < 0.1%
   - Phase differences < 0.1°
3. **Check systematics:**
   - Beam interpolation accuracy
   - Source position calculations
   - Baseline phase centers
4. **Document differences:**
   - Numerical precision
   - Algorithm variations
   - Convention differences

---

## Appendix: Raw vsim.py Output

### Complete Output Data (Exact Values)

**Commands:**
```bash
./vsim.py make-obsparams --ants 0~9 --channels 227~257 --sky-model ptsrc256 \
  --n-time-chunks 288 --do-time-chunks 0~3 --simulator matvis \
  --beam-map-csv beam_map_10.csv

./vsim.py runsim --ants 0~9 --channels 227~257 --sky-model ptsrc256 \
  --n-time-chunks 288 --do-time-chunks 0~3 --simulator matvis \
  --beam-map-csv beam_map_10.csv
```

**Antenna Info:**
```
get_enu_data_ants:
(array([[-105.03529563, -110.72150481,    0.81120331],
       [ -90.42745642, -110.66571205,    0.81868112],
       [ -75.81961721, -110.60991928,    0.82615894],
       [ -61.21177801, -110.55412652,    0.83363676],
       [ -46.6039388 , -110.49833375,    0.84111458],
       [ -31.99609959, -110.44254098,    0.84859239],
       [ -17.38826039, -110.38674822,    0.85607021],
       [  -2.78042118, -110.33095545,    0.86354803],
       [  11.82741802, -110.27516269,    0.87102585]]),
 array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=int32))
```

**Channel Info:**
```
channels: 227-257
channel_width: [0.12207031] MHz
freq array: [74.6307373046875, 74.7528076171875, 74.8748779296875,
             74.9969482421875, 75.1190185546875, 75.2410888671875,
             75.3631591796875, 75.4852294921875, 75.6072998046875,
             75.7293701171875, 75.8514404296875, 75.9735107421875,
             76.0955810546875, 76.2176513671875, 76.3397216796875,
             76.4617919921875, 76.5838623046875, 76.7059326171875,
             76.8280029296875, 76.9500732421875, 77.0721435546875,
             77.1942138671875, 77.3162841796875, 77.4383544921875,
             77.5604248046875, 77.6824951171875, 77.8045654296875,
             77.9266357421875, 78.0487060546875, 78.1707763671875] MHz
```

**Time Info:**
```
Integration time: 4.986347833333333 s
LST: [0.08333419, 0.08472308, 0.08611198, 0.08750086, 0.08888975, 0.09027864,
      0.09166753, 0.09305641, 0.09444531, 0.09583419, 0.09722308, 0.09861197,
      0.10000086, 0.10138975, 0.10277864, 0.10416752, 0.10555642, 0.1069453 ,
      0.10833419, 0.10972309, 0.11111197, 0.11250085, 0.11388975, 0.11527864,
      0.11666752, 0.11805642, 0.1194453 , 0.12083419, 0.12222308, 0.12361197,
      0.12500086, 0.12638975, 0.12777863, 0.12916753, 0.13055641, 0.1319453 ,
      0.13333419, 0.13472308, 0.13611196, 0.13750086, 0.13888974, 0.14027863,
      0.14166753, 0.14305641, 0.1444453 , 0.14583419, 0.14722308, 0.14861196,
      0.15000086, 0.15138974, 0.15277863, 0.15416752, 0.15555641, 0.1569453 ,
      0.15833419, 0.15972307, 0.16111197, 0.16250085, 0.16388974, 0.16527863] Hrs
```

**Beam Info:**
```
beam file: NF_HERA_Vivaldi_efield_beam_extrap_i.fits
i: ith antenna (per-antenna beam assignment)
```

**Sky Model:**
```
sky model maps for above frequencies:
/lustre/aoc/projects/hera/rchandra/H6C_Validation_Stats/validation-sim/sky_models/ptsrc256/*.skyh5
```

---

**Created:** 2025-01-22
**Purpose:** Understanding HERA validation simulation for RRIvis development
**Status:** Analysis complete, awaiting config implementation
