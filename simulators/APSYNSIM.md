# APSYNSIM — Aperture SYNthesis SIMulator

Exhaustive reference for the bundled `simulators/APSYNSIM/` checkout. This document covers every public-facing feature of the program *and* every internal algorithm in `SCRIPT/APSYNSIM3.py`, so it can be used both as an end-user manual and as a code-reading guide.

---

## 1. Identity, provenance, license

| Field | Value |
|---|---|
| Full name | APSYNSIM — A real-time **AP**erture **SYN**thesis **SIM**ulator |
| Author | Iván Martí-Vidal, Onsala Space Observatory / Nordic ALMA Regional Center node, Chalmers University of Technology, Sweden |
| Acknowledgement | J. Girard (feedback) |
| Contact | `contact@nordic-alma.se` |
| Bundled version | `1.4-b` (see `__version__` at line 44 of `SCRIPT/APSYNSIM3.py`); changelog also lists v1.4 (Sept 2016) as the most recent entry |
| Original distribution | `https://launchpad.net/apsynsim` (per the exercises PDF) |
| License | GNU GPL v3 or later (`LICENSE`); bundled docstrings reproduce the GPL header in every script |
| Third-party | numpy, scipy, matplotlib, Tkinter (standard library); no Astropy, no CASA, no MeasurementSet |
| Submodule status in this repo | `simulators/APSYNSIM` is a git submodule (`.git` is a `gitdir:` pointer to `../../.git/modules/simulators/APSYNSIM`) |

APSYNSIM is a **teaching/demonstration tool**, not a production data-reduction package. It is designed to let students drag antennas around, change observing conditions live, and see the resulting *u*-*v* coverage, dirty beam, dirty image, CLEAN image, and Fourier-space pictures update in real time. It also supports introducing complex-gain corruption to study calibration effects, and adding random Gaussian visibility noise.

---

## 2. Repository layout

```
APSYNSIM/
├── README.md                      # Short user-facing readme
├── LICENSE                        # GPL v3 + third-party-license attributions
├── changelog                      # Versioned change history (v0.1 … v1.4)
├── APSYNSIM_EXERCISES.pdf         # Tutorial booklet (1.4-beta, Sept 2016)
├── .gitignore                     # *~ and *.pyc
├── ARRAYS/                        # 21 bundled array configuration files (*.array)
├── SOURCE_MODELS/                 # 18 bundled sky model files (*.model)
├── PICTURES/                      # PNG/JPG raster sky images referenced from models
├── COMPILE/                       # PyInstaller icon + compile.txt note
└── SCRIPT/
    ├── APSYNSIM2.py               # Python 2 build (3,384 lines)
    ├── APSYNSIM3.py               # Python 3 build (3,384 lines)  ← canonical
    ├── APSYNSIM.config            # Defaults loaded at start
    ├── APSYNSIM.spec              # PyInstaller spec (Windows)
    └── APSYNSIM_MAC.spec          # PyInstaller spec (macOS)
```

`APSYNSIM2.py` and `APSYNSIM3.py` are functionally identical apart from the Python-2 vs Python-3 imports (`tkFileDialog`/`tkMessageBox`/`ScrolledText` vs `tkinter.filedialog`/`tkinter.messagebox`/`tkinter.scrolledtext`) and the matplotlib widget name change `NavigationToolbar2TkAgg` → `NavigationToolbar2Tk`. All references below cite **APSYNSIM3.py**.

---

## 3. Running the program

```shell
cd APSYNSIM/SCRIPT
python APSYNSIM3.py        # Python 3
python APSYNSIM2.py        # legacy Python 2
```

The script auto-locates `APSYNSIM.config`, the `ARRAYS/`, `SOURCE_MODELS/`, and `PICTURES/` directories relative to its own location (`d1 = os.path.dirname(os.path.realpath(__file__))`). If the file is invoked from a different cwd the script falls back to `os.getcwd()`.

The application creates a Tk root window titled `Aperture Synthesis Simulator (I. Marti-Vidal, Onsala Space Observatory) - version <ver>`, instantiates `Interferometer(tkroot=root)`, and enters `Tk.mainloop()`. The recommended way to exit is the **Quit** button or **Quit** menu item — both call `Interferometer.quit()` which destroys the Tk root and calls `sys.exit()`. Closing the window manager X also routes there via `WM_DELETE_WINDOW`.

### 3.1 Runtime dependencies

The startup imports (lines 22–42) are:

- `os`, `sys`, `time`
- `tkinter` (`Tk`, `filedialog`, `messagebox`, `scrolledtext`)
- `matplotlib` with the `TkAgg` backend forced via `mpl.use('TkAgg')`
- `pylab as pl`, `numpy as np`
- `scipy.ndimage.interpolation as spndint` (only for `spndint.zoom` when scaling raster `IMAGE` source models)
- `scipy.optimize as spfit` (only for `spfit.leastsq` to fit a 2-D Gaussian to the central PSF lobe and derive the CLEAN restoring beam)
- `matplotlib.widgets.Slider`, `Button`
- `matplotlib.cm`, `matplotlib.image as plimg`
- `mpl_toolkits.mplot3d.Axes3D` (3-D globe plot in the upper-right corner)
- `matplotlib.backends.backend_tkagg.{FigureCanvasTkAgg, NavigationToolbar2Tk}`
- `matplotlib.backend_bases.NavigationToolbar2`

### 3.2 Configuration file (`SCRIPT/APSYNSIM.config`)

Plain-text key=value, hash-comment-stripped. Recognised keys (the parser at lines 232–245 looks at the leading characters of each non-blank line after stripping spaces):

| Key | Type | Default | Meaning |
|---|---|---|---|
| `Npix` | int | `512` | Image side length in pixels. **Must be a power of 2.** Lower → faster; higher → smoother pixelation. The image is internally `Npix × Npix`; only the central `Npix/2 × Npix/2` quadrant is shown. `self.Nphf = Npix // 2`. |
| `nH` | int | `200` (config-shipped value: `100`) | Number of hour-angle samples between `H_0` and `H_1`. Lower → faster but coarser UV tracks. |
| `DefaultMod` | str | `'Nebula.model'` | File loaded from `SOURCE_MODELS/` at startup |
| `DefaultArray` | str | `'Long_Golay_12.array'` | File loaded from `ARRAYS/` at startup |

> Code note: `DefaultMod` is what the parser searches for, but the variable assigned is `DefaultModel`. The default fallback variable `DefaultMod` (without the `el` suffix) is set at line 219 to `'Nebula.model'`, so this works only because both names happen to be referenced; this is a latent bug pattern but does not break the shipped config.

---

## 4. Main GUI layout

A single matplotlib `Figure` of size `(15, 8)` is embedded into a Tk window via `FigureCanvasTkAgg`. The figure houses five 2-D subplots arranged on a 2×3 grid, plus one 3-D inset and a vertical column of widgets:

```
┌──────────────────────┬──────────────────────┬──────────────────────┐
│  ARRAY CONFIGURATION │       UV PLANE       │     DIRTY BEAM       │
│       (231)          │  (232, gray bg)      │       (233)          │
│                      │                      │                      │
│                      │   3-D Earth globe    │                      │
│                      │   inset (top right)  │                      │
├──────────────────────┼──────────────────────┼──────────────────────┤
│   Sliders / Buttons  │     MODEL IMAGE      │     DIRTY IMAGE      │
│   panel (left)       │       (235)          │       (236)          │
└──────────────────────┴──────────────────────┴──────────────────────┘
```

The control panel uses `pl.axes(...)` with explicit normalised figure coordinates (no `gridspec`):

| Widget | Type | Geometry | Range / behaviour |
|---|---|---|---|
| `lat` | Slider | `[0.07, 0.45, 0.25, 0.04]` | -90° … +90° latitude of the array centre |
| `dec` | Slider | `[0.07, 0.40, 0.25, 0.04]` | -90° … +90° declination of the source |
| `H0` | Slider | `[0.07, 0.35, 0.25, 0.04]` | -12 h … +12 h initial hour angle |
| `H1` | Slider | `[0.07, 0.30, 0.25, 0.04]` | -12 h … +12 h final hour angle |
| `wave` | Slider | `[0.07, 0.25, 0.25, 0.04]` | Min/max wavelengths from `WAVELENGTH` keyword in the loaded `.array` |
| `robust` | Slider | `[0.07, 0.20, 0.25, 0.04]` | Briggs robustness, -2 (≈ uniform) to +2 (≈ natural) |
| `+ Antenna` | Button | `[0.07, 0.14, 0.08, 0.05]` | Insert one antenna at (0, 0). If a slot was vacated by `−` the prior position is restored |
| `− Antenna` | Button | `[0.155, 0.14, 0.08, 0.05]` | Removes the highest-index antenna. Refuses to drop below 2 antennas |
| `Reduce data` | Button | `[0.24, 0.14, 0.08, 0.05]` | Opens the CLEAN GUI (a `CLEANer` instance in a new `Toplevel`) |
| `Save array` | Button | `[0.07, 0.08, 0.08, 0.05]` | Filedialog → write current array to `*.array` |
| `Load array` | Button | `[0.155, 0.08, 0.08, 0.05]` | Filedialog → read `*.array` (default dir = `ARRAYS/`) |
| `Quit` | Button | `[0.155, 0.02, 0.08, 0.05]` | Same as menubar Quit |
| `Load model` | Button | `[0.24, 0.08, 0.08, 0.05]` | Filedialog → read `*.model` (default dir = `SOURCE_MODELS/`) |
| `gamma` | Slider | `[0.46, 0.08, 0.13, 0.02]` (red, white labels) | Gamma correction for the **MODEL IMAGE** display only, 0.1 … 1.0, default 0.5 |
| `Dish size (m)` | Slider | `[0.825, 0.08, 0.10, 0.02]` (red, white labels) | Antenna diameter for primary-beam apodisation, 0 … 100 m. 0 disables PB attenuation. |
| `log(W1/W2)` | Slider | `[0.15, 0.58, 0.12, 0.02]` (red) | Visible only when subarrays are loaded; sets log10 weight ratio between subarrays, -4 … +4 |

Top-of-window menu bar holds two items: **Help** (pops up `__help_text__` in a scrolled-text Toplevel) and **Quit**.

### 4.1 The five subplots

1. **ARRAY CONFIGURATION** (`self.antPlot`, subplot 231, square aspect). Antenna positions in km, plotted as lime markers (red for the second subarray when present). Each antenna is annotated with its 1-based index. Antennas are pickable (`picker=5`); when picked, they can be dragged with `motion_notify_event`. A blue line `antPlotBas` is drawn between the two antennas of a baseline that the user clicks in the UV PLANE. Axis labels: `E-W offset (km)` / `N-S offset (km)`. Title: `ARRAY CONFIGURATION`. Antenna count and (when picked) coordinates are displayed in a top-left text box.

2. **UV PLANE** (`self.UVPlot`, subplot 232, dark-gray facecolor). A grayscale image of `|FT(model image)|` is shown as the background (`UVPlotFFTPlot`), with the projected baseline tracks plotted on top as lime points (red for subarray 2). Both `(u, v)` and `(-u, -v)` are drawn (Hermitian symmetry). Axis labels: `U (Mλ)` or `U (kλ)` (auto-switched), same for V. Three text boxes are anchored to this subplot:
   - `latText` (orange, top-left): `φ = … °  δ = … °  H = …h / …h`
   - `visText` (orange): `Amp: … Jy.   Phase: … deg.` updated when the user picks any UV point
   - `basText` (orange, bottom-left): `Bas i − j  at  H = …h` updated when the user clicks on a UV track
   The U-axis is plotted with reversed sign (per changelog v0.3) so that the convention is self-consistent with the dirty-image E-W axis sense.

3. **DIRTY BEAM** (`self.beamPlot`, subplot 233). Synthesised PSF normalised to peak 1.0 (subplot displays only the central quadrant). Top-left text box (`beamText`) reads `λ = … mm` / `… Jy/beam` / `Δα = …  Δδ = …` and is updated on click (then says `% .2e Jy/beam` at the picked offset). If the central UV pixel is all-zero, `beamText` is overwritten with a warning `WARNING! Too short baselines for such a small image …`.

4. **MODEL IMAGE** (`self.modelPlot`, subplot 235). The pure source brightness (no PSF, no PB) raised to `gamma`, in Jy/pixel, units shown in title `MODEL IMAGE: <total flux> Jy`. Click reports `% .2e Jy/pixel` and Δα/Δδ.

5. **DIRTY IMAGE** (`self.dirtyPlot`, subplot 236). Source convolved with the dirty beam (i.e. the simulated observation after Briggs weighting + corrupting gains + noise). Click reports `% .2e Jy/beam` and Δα/Δδ. Dish-size slider modifies the primary-beam apodisation that is applied **before** taking the FFT for this image.

6. **3-D Earth globe** (`self.spherePlot`, manual axes `[0.53, 0.82, 0.12, 0.12]`, `mpl_toolkits.mplot3d.Axes3D`). A unit sphere with a yellow polyline showing where the antennas (rotated by the latitude) appear *as seen from the source*. The polyline is recomputed in `_plotBeam` from `arrayPath` = `(10·H[1]·cosφ, 10·H[0]·cosφ, 10·sinφ)`. The viewing elevation is locked to the source declination, azimuth 0. Dragging the globe (via `_onPress` / `_onAntennaDrag` / `_onSphere = True`) re-binds the elevation back to the `dec` slider, allowing the user to set declination by tilting the Earth.

### 4.2 Keyboard shortcuts

Bound in `_onKeyPress`:

| Key | Action |
|---|---|
| `Z` (capital) | Zoom in by 2× on the panel under the cursor (forwarded to `_onPress` with `event.button=1, dblclick=True`) |
| `z` (lowercase) | Zoom out by 2× on the panel under the cursor (forwarded with `event.button=3`) |
| `c` / `C` | Toggle colormap between `cm.jet` and `cm.Greys_r`. All four image plots (and the CLEAN GUI panels if open) are re-rendered |
| `u` / `U` | Open `UVPLOTTER2` Toplevel — a 1×3 figure showing **UV-PSF**, **UV-OBSERV.** (FT of dirty image), **UV-SOURCE** (FT of true model image) |

Mouse: left double-click zooms in, right double-click zooms out, on the same logic. Single-click in any image picks a pixel and prints its value.

---

## 5. Array file format (`*.array`)

ASCII, line-oriented, hash comments. Parser is `Interferometer.readAntennas()` (lines 501–624). Whitespace-separated tokens. Recognised keywords:

| Keyword | Args | Meaning | Notes |
|---|---|---|---|
| `LATITUDE` | 1 float, deg | Geographic latitude of array centre | stored as `self.lat` in radians |
| `DECLINATION` | 1 float, deg | Source declination | `self.dec` in radians |
| `HOUR_ANGLE` | 2 floats, hours | Initial / final HA of the observation | `self.Hcov`, in radians, with `Hfac = π/180·15` |
| `WAVELENGTH` | 2 floats, **m** | Min / max wavelengths for the slider. *Internally converted to km*: each value is multiplied by `1.e-3`. The middle value `(min+max)/2` is the active wavelength. Slider always shows mm. | Added in v1.4; previously lived in the model file |
| `DIAMETER` | 1 or 2 floats, m | Antenna dish diameter, used only for primary-beam apodisation. Two values when subarrays are used: first for `ANTENNA`, second for `ANTENNA2` |
| `ANTENNA` | 2 floats, **m** (E-W, N-S offset from origin) | Position of one antenna in subarray 1. Internally divided by 1000 → km | minimum 2 antennas per file |
| `ANTENNA2` | 2 floats, m | Antenna in subarray 2. ≥ 2 of these triggers `self.subarray = True`, enabling subarray weighting and disjoint UV tracks | optional |

Any unrecognised non-comment token raises a Tk error dialog and the read aborts. The parser also enforces:

- `len(antPos) ≥ 2`, else error
- `|lat − dec| ≥ π/2` (target below horizon at all hours) → error
- HA range is intersected with the source-above-horizon arc `±arccos(−tan φ tan δ)` (`Hhor`), so the program never integrates over below-horizon HAs.

After successful parse, `Xmax` (used for axis limits in km) is set to `1.5 × max(|x|, |y|)` over all antennas.

### 5.1 `saveArray()` quirk (Python-2 leftover)

`Interferometer.saveArray()` (lines 1965–1994) still uses `print >> iff, ...` syntax. **This will raise SyntaxError on Python 3**. So the *Save array* button works only in `APSYNSIM2.py`; in the Python-3 build it is a known dead button. Fixing it requires replacing the `print >> iff, fmt % args` lines with `iff.write(fmt % args + '\n')` (or `print(fmt % args, file=iff)`).

### 5.2 Bundled arrays (in `ARRAYS/`)

| File | Latitude (°) | Default δ (°) | HA range (h) | λ range (m) | N(ant) [ + N2 ] | Notes |
|---|---|---|---|---|---|---|
| `Default.array` | 45 | 45 | -12…+12 | 3 mm – 21 cm | 7 | 7-element pseudo array used at first start |
| `ACA.array` | -23 | … | … | mm | … | ALMA Compact Array (12 m main only) |
| `ALMA-ACA-Cycle1-Conf5.array` | -23.028 | -45 | -1…+1 | 0.3 mm – 3 mm | 32 + 9 (subarrays) | 12 m + 7 m mixed; demonstrates the subarray feature with `DIAMETER 12. 7.` |
| `CARMA-A.array`, `CARMA-E.array` | … | … | … | mm | … | CARMA |
| `E-W.array` | … | … | … | … | … | Pure east-west baseline geometry |
| `Golay_12.array`, `Golomb_16.array`, `Long_Golay_12.array` | 45 | 60 | -1.5…1.5 | 3 mm – 21 cm | 12 / 16 / 12 | Aperiodic / minimum-redundancy arrays. `Long_Golay_12` is the shipped startup array. |
| `pseudo-VLA.array`, `VLA-A`/`B`/`C`/`D.array` | 33.9 | 45 | -0.1…+0.1 | 7 mm – 21 cm | 27 | DIAMETER 25 m. The four configurations are stored separately. |
| `PdBI-A`/`B`/`C`/`D.array` | … | … | … | mm | … | Plateau de Bure Interferometer |
| `WSRT.array` | … | … | … | cm | 14 | Westerbork |
| `Two-antennas.array`, `Two-antennas_Long.array` | … | … | … | … | 2 | Pedagogical |

Coordinates inside files are in **metres** even though the GUI displays km — the loader divides by 1000.

---

## 6. Source-model file format (`*.model`)

ASCII, hash comments. Parser is `Interferometer.readModels()` (lines 626–688).

### 6.1 Keywords

| Keyword | Format | Meaning |
|---|---|---|
| `IMSIZE` | `IMSIZE <half-width-arcsec>` | Sets `self.imsize = 2·IMSIZE`. The displayed image therefore spans `±IMSIZE` arcsec from centre. If absent, `imsize` auto-fits to enclose all components (`Xmax × 1.1`). |
| `P` | `P <RA> <Dec> <Flux>` | **Point source.** Δα/Δδ in arcsec, flux in Jy. Pixel-snapped: planted into one pixel of `modelimTrue` if it falls inside the image. |
| `G` | `G <RA> <Dec> <Flux> <σ>` | **Circular Gaussian.** σ in arcsec (note: 1-σ width, not FWHM). Flux is total in Jy; the Gaussian is normalised so `np.sum(gauss) == 1` and then multiplied by `Flux`. |
| `D` | `D <RA> <Dec> <Flux> <radius>` | **Uniform disc.** Radius in arcsec. Flux uniformly distributed across pixels with `(x-x₀)² + (y-y₀)² ≤ R²` (plus the closest pixel as a safety, so even sub-pixel discs render at least one pixel). |
| `IMAGE` | `IMAGE <filename> <peak Jy/pixel>` | **Raster image as model.** PNG/JPG, average of channels 0..2; subtracts min, scales to peak Jy/pixel, then `scipy.ndimage.zoom`-ed to `Npix/2` and pasted into the central quadrant (`Np4` margin on each side). The `filename` is searched first as given, then in `PICTURES/`. |

### 6.2 Internal representation

`self.models` is a list of `[type-char, RA, Dec, flux, size?]`. `self.imfiles` is a list of `[filename, peak]`. `self.modelimTrue` is the rendered `Npix × Npix` float32 image in Jy/pixel; the displayed image is its central `Npix/2 × Npix/2` quadrant — all UI flux/coordinate logic uses that quadrant.

Negative pixel values are clipped to 0 after loading (`self.modelimTrue[self.modelimTrue < 0.0] = 0.0`).

### 6.3 Bundled models (in `SOURCE_MODELS/`)

`Default.model` (3 simple components: G + D + P), `Point-source.model` / `Point-source2.model`, `Double-source.model` / `Double-source-small.model`, `One-Disc.model`, `Five-Gauss.model`, `Discs.model`, `Cloud.model`, `Campanar.model` (uses `Camp_Alfarrasi.jpg`), `Lena.model`, `Faceon-Galaxy.model` (uses `M100.png`/`M100-v2.png`), `Nebula.model` / `Nebula_small.model` (use `Crab.png`), `RadioGalaxy.model` (uses `cyga_21cm.png`), `Gauss-and-bigdisc.model`, `Gauss-and-bigdisc-ALMA.model`, `Point-and-Gauss-ALMA.model`. Source images live in `PICTURES/` with attribution in `PICTURES/Credits.txt` (NRAO/AUI, NOAO/AURA/NSF, M. Bietenholz, Bell tower of Alfarrasi).

---

## 7. Physics and algorithms

This section documents what the code actually computes, line by reference. The numerics are pedagogical, not science-grade.

### 7.1 Coordinate conventions and constants

```python
self.Hfac     = π/180 * 15        # convert hours → radians
self.deg2rad  = π/180
self.deltaAng = 1·deg2rad         # angular increment used for some refresh logic
self.gamma    = 0.5               # display gamma for the model
self.lfac     = 1e6               # u-v unit factor (Mλ); auto-switches to 1e3 (kλ) when array is small
self.W2W1     = 1.0               # subarray weight ratio, controlled by log(W1/W2) slider
self.wavelength = [λmin, λmax, λmid]  # all in km
```

### 7.2 Baselines and *u, v* tracks

`_prepareBaselines` enumerates the unordered antenna pairs in row-major order, populates `self.basnum[Nant, Nant−1]` (per-antenna baseline indices), `self.basidx[n1, n2]` (lookup), `self.antnum[bi] = (n1, n2)` (reverse lookup). Number of baselines `Nbas = Nant·(Nant−1)/2`.

For each baseline, `_setBaselines` computes a 3-vector

```
B[0] = -(y2 - y1) · sin(lat) / λ
B[1] =  (x2 - x1)            / λ
B[2] =  (y2 - y1) · cos(lat) / λ
```

and the projected UV coordinates over the HA grid `H = linspace(H0, H1, nH)`:

```
u =   -B[0]·sin(H) - B[1]·cos(H)
v =   -B[0]·sin(δ)·cos(H) + B[1]·sin(δ)·sin(H) + cos(δ)·B[2]
```

Note that the antenna positions are stored in km and λ is in km, so `u`,`v` are dimensionless (cycles); they are subsequently divided by `lfac` (1e6 → Mλ) for display.

### 7.3 Hour-angle clamping at horizon

`_changeCoordinates` recomputes the rise/set angle `Hhor = arccos(−tan φ · tan δ)` (or 0/π for circumpolar/never-up) and clips `Hcov[0]`/`Hcov[1]` so that no integration time lies below the horizon, sliding the H0/H1 sliders to match.

### 7.4 UV gridding (`_gridUV`)

The continuous `(u, v)` coordinates of every visibility sample are quantised onto the `Npix × Npix` grid with pixel size

```
UVpixsize = 2 / (imsize · π/180/3600)    # cycles per radian, derived from image FoV
```

so the UV-plane Nyquist sampling matches the requested image FoV. For each baseline, two pixels (`+u, +v` and `−u, −v`) per HA sample are added to four arrays:

- `totsampling[Npix, Npix]` — number of samples per pixel (for natural-weighting density)
- `Gsampling[Npix, Npix]` (complex) — same, but each sample is the antenna-pair complex gain `Gains[bi, h]` and its conjugate at the mirrored cell
- `noisemap[Npix, Npix]` (complex) — accumulated gain-amplitude-weighted Gaussian noise samples

When an antenna is dragged, only the affected baseline indices are rewritten and the prior pixel positions (cached in `self.pixpos[bi]`) are first subtracted, so the global maps stay consistent without a full recompute. This is the key that makes the simulation interactive.

### 7.5 Briggs weighting

The robustness slider `R ∈ [-2, 2]` is folded into

```
robfac = (5·10^(-R))^2 · (2·Nbas·nH) / Σ totsampling²
```

and the gridded sampling functions are normalised with

```
robustsamp     = totsampling / (1 + robfac · totsampling)
Grobustsamp    = Gsampling   / (1 + robfac · totsampling)
GrobustNoise   = noisemap    / (1 + robfac · totsampling)
```

which yields uniform weighting at the `R = -2` end (large `robfac` makes the denominator dominate, pushing each cell to ~1) and natural at `R = +2` (small `robfac`, weighting tracks `totsampling` itself).

### 7.6 Dirty beam

```
beam = Re{ ifftshift( ifft2( fftshift( robustsamp ) ) ) } / (1 + W2W1)
```

Then normalised so its central pixel = 1 (`beam /= beamScale`). With subarrays, both `robustsamp` and `robustsamp2` contribute, mixed with `(1, W2W1)/(1+W2W1)` weights, and a single common normalisation `beamScale2`.

### 7.7 Primary-beam apodisation

If the dish-size slider is non-zero, `_setPrimaryBeam` builds a Gaussian apodisation in image space using the 1.22 λ/D rule (FWHM converted to σ via the 2.3548 factor):

```
PB_sigma_arcsec² = 2 · (1220 · 180/π · 3600 · λ / D / 2.3548)²
modelim[0] = modelimTrue · exp( distmat / PB )
```

(`distmat[i,j] = -(x_i² + y_j²)·pixsize²` is the radial squared-arcsec map, so the exponent comes out negative, producing a Gaussian roll-off centred at the image centre.) Subarray 2 uses its own `Diameters[1]`. The FFT used for the dirty image (`modelfft`/`modelfft2`) is computed from the apodised image `modelim`, **not** from `modelimTrue` — but the **MODEL IMAGE** panel always shows the un-apodised `modelimTrue` so the user can see the difference between the true sky and what the array sees.

### 7.8 Dirty image

```
dirtymap = Re{ fftshift( ifft2( ifftshift(GrobustNoise) +
                                modelfft · ifftshift(Grobustsamp) ) ) } / (1 + W2W1)
dirtymap /= beamScale          # so units are Jy/beam relative to the dirty beam peak
```

With a second subarray, the modelfft2 · robustsamp2 term is added with the (W2W1/(1+W2W1)) weight before normalising.

### 7.9 Antenna corrupting gains

`_setGains(An1, An2, H0, H1, G)` (lines 789–805) applies a complex gain `G = Amp · e^{iφ}` to baselines that include antenna `An1`. Three modes:

- `An2 == -1` and `An1 ≥ 0` → all baselines containing `An1` are corrupted (i.e., antenna-based gain).
- `An2 != An1` → only the single baseline `(An1, An2)` is corrupted.
- The conjugate `G*` is applied to the swapped half of the Hermitian pair so the visibilities remain Hermitian.

Affected HA range is `[H0, H1]` integer scan indices (default whole observation, `H0=0`, `H1=nH`). After mutation, the routine triggers `_setBaselines → _setBeam → _plotBeam → _plotDirty`. All gains reset to 1 when an antenna is added/removed (`_prepareBaselines` reallocates `self.Gains`).

### 7.10 Random visibility noise

`_setNoise(noise)` draws `Nbas × nH` complex Gaussian samples with `loc=0, scale=noise`. The CLEAN GUI converts a user-entered single sensitivity (Jy/beam, natural-weighting expected RMS in a source-free observation) to per-sample noise as

```
sensPerSamp = sensit · √Nsamples / √2     where Nsamples = Nbas · nH
```

so the per-sample real and imaginary RMS are each `sensPerSamp`. Each call to `Redo Noise` re-samples a new realisation.

### 7.11 CLEAN deconvolution

Implemented in `CLEANer._CLEAN` (lines 2687–2786) — Hogbom-style scalar CLEAN restricted to user-painted masks:

1. Read user controls: `Gain` (loop gain, default 0.1), `Iterations` (default 100), `Thres` (Jy/beam — pixels below thres are zeroed; negative threshold flips sign so negative components can be cleaned).
2. Build `tempres = residuals · mask` (or `residuals` if no mask was painted).
3. Loop:
   - find `peakpos = argmax(tempres)` and `peakval = residuals[peakpos]`
   - subtract `gain · peakval · roll(psf, peakpos − Npix//2)` from `residuals`
   - update `tempres` only on the masked pixels (`tempres[goods] = residuals[goods]`)
   - increment the **delta-function** model `cleanmodd[peakpos] += gain·peakval`
   - increment the **restored** model `cleanmod += gain·peakval · roll(cleanBeam, peakpos − Npix//2)`
   - update title `CLEAN (n ITER): total Jy`, redraw RMS/peak text boxes.
4. Stop when the iteration budget is exhausted or the threshold is reached.

The **CLEAN beam** is fitted *once per `_reset`* by `scipy.optimize.leastsq` to the central main lobe of the dirty beam where `beam > 0.6`. The fit minimises

```
exp(-(dX² · A + dY² · B + dX·dY · C)) − beam
```

over `(A, B, C)`, then converts to FWHM in arcsec and a position angle:

```
PA          = ½ · atan2(C, A−B)
A_FWHM      = 2.355 · √(2 / (A+B + C/sin(2·PA)))   · imsize/Npix
B_FWHM      = 2.355 · √(2 / (A+B − C/sin(2·PA)))   · imsize/Npix
```

If the fit fails (or the main lobe is < 5 pixels — happens when baselines are way too long for the chosen image size), the CLEAN beam degenerates to a single delta and a Tk error dialog is shown.

### 7.12 Mask drawing

In the CLEAN GUI's RESIDUALS panel, `_doMask` tracks left-or-right-button drags and renders a transient white box. On release, the swept rectangle is added (LMB) or removed (RMB) from the boolean mask `self.bmask`. The mask contour is redrawn via `ResidPlot.contour(... levels=[0.5])`.

### 7.13 Restored / unrestored / +residuals toggles

- `(Un)restore` → flips `self.dorestore`. When unrestored, the CLEAN panel displays the `cleanmodd` delta-function model; when restored, it shows `cleanmod` (delta convolved with the fitted Gaussian beam).
- `+/- Resid` → flips `self.resadd`. When True, the panel shows `cleanmod + residuals`. Cannot be combined with the unrestored mode — it raises an error popup.
- `Rescale` → recomputes `vmin`/`vmax` from the current array (useful to dig into residual structure).

### 7.14 True-source-convolved viewer

`CLEANer._convSource` opens a Toplevel showing `IFT(modelfft · FT(cleanBeam))` — i.e. the *true* source brightness convolved with the CLEAN beam. This is the noise- and sampling-free equivalent of what a perfect CLEAN should asymptote to, and lets the student visually compare deconvolution fidelity. It has its own `Reload` button to refresh after main-window changes.

### 7.15 Auxiliary FFT viewers

Two Toplevel classes live alongside the main `Interferometer`:

- `UVPLOTTER2` (1×3, called by the `u` key on the main window). Panels: `UV - PSF` (= |FT(beam)|), `UV - OBSERV.` (= |FT(dirtymap)|), `UV - SOURCE` (= |FT(modelimTrue)|). Each has its own click-readout of the FFT amplitude.
- `UVPLOTTER` (2×3, called by the `Show FFT` button inside the CLEAN GUI). Panels: `UV-PSF`, `UV-CLEAN (MODEL)` = |FT(cleanmodd)|, `UV-SOURCE` = |FT(modelfft)|, `UV-RESIDUALS (REST.)` = |FT(residuals)|, `UV-CLEAN (REST.)` = |FT(cleanmod)|, `UV-SOURCE (REST.)` = |modelfft · FT(cleanBeam)|. Both have a `Reload` button.

---

## 8. CLEAN GUI window in detail

Opened by the **Reduce data** button. Class `CLEANer`. Layout:

```
┌────────────────────────────────────────────────────────────────────┐
│   RESIDUALS                CLEAN (n ITER): total Jy                │
│   (mask paintable)                                                 │
│ ┌───────────────────┐ ┌───────────────────┐  ┌──── controls ────┐ │
│ │                   │ │                   │  │ Gain:    0.1     │ │
│ │  imshow + contour │ │  imshow CLEAN     │  │ Iterations: 100  │ │
│ │  + transient box  │ │  model            │  │ Thres (Jy/b): 0  │ │
│ │                   │ │                   │  │ [ CLEAN ]        │ │
│ └───────────────────┘ └───────────────────┘  │ [ RELOAD ]       │ │
│                                              │ [ +/- Resid ]    │ │
│                                              │ [(Un)restore]    │ │
│                                              │ [ Rescale ]      │ │
│                                              │ [ Show FFT ]     │ │
│                                              │ [ True source ]  │ │
│                                              │ ─────────────────│ │
│                                              │ Sensit. (Jy/b):0 │ │
│                                              │ [ Redo Noise ]   │ │
│                                              └──────────────────┘ │
│  CALIBRATION ERROR:                                                │
│  Ant.1 [list] Ant.2 [list] | From integ #  0  | To integ # nH     │
│                              Amplitude gain (%) | Phase Gain      │
│  [ APPLY GAIN ] [ RESET GAIN ]                                     │
└────────────────────────────────────────────────────────────────────┘
```

- **RESIDUALS** is mask-paintable: LMB-drag adds a mask region, RMB-drag removes one. White rubber-band box during the drag, contour after release.
- **CLEAN** mirrors x/y limits to RESIDUALS via `sharex`/`sharey`.
- **Sensit.** is in Jy/beam (the natural-weighting RMS the array would achieve in a source-free observation); the program multiplies by √(Nbas·nH)/√2 internally to set per-sample σ. Pressing **Redo Noise** generates a fresh realisation.
- **APPLY GAIN** corrupts the visibilities of the chosen antenna (Ant. 1 only → antenna-based) or baseline (Ant. 1 + Ant. 2 different → baseline-based). The corruption is restricted to integration scans `H0…H1` and is `Amp · e^{iφ}`.
- **RESET GAIN** restores all gains to 1.
- The **RELOAD** button (`_reset`) refits the CLEAN beam, pulls the latest `dirtymap` from the parent, and clears the CLEAN model. **You must press RELOAD** after any change in the main window (wavelength, antennas, declination, etc.) for the CLEAN GUI to reflect those changes — but this also lets users keep a prior CLEAN result on screen for comparison while iterating in the main window.

A subtlety: re-clicking **CLEAN** continues from the existing residuals, allowing chained CLEAN passes with different gains/thresholds.

---

## 9. Class architecture (developer's view)

### 9.1 `Interferometer` (≈ lines 184–2058)

The top-level controller. State, partial list:

- Geometry: `Nant`, `antPos`, `Nant2`, `antPos2`, `Xmax`, `subarray`
- Observation: `lat`, `dec`, `Hcov`, `Hmax`, `wavelength` (3-tuple), `Diameters[2]`
- Image grid: `Npix`, `Nphf = Npix//2`, `imsize`, `Xaxmax`, `nH`
- Visibility tensors: `B[Nbas, 3]`, `u[Nbas, nH]`, `v[Nbas, nH]`, `Gains[Nbas, nH]`, `Noise[Nbas, nH]`, `pixpos[Nbas]`
- Gridded products: `totsampling`, `Gsampling`, `noisemap`, `robustsamp`, `Grobustsamp`, `GrobustNoise`, `beam`, `dirtymap`, `modelimTrue`, `modelim[2]`, `modelfft`, `modelfft2`
- Plot handles: `antPlot`, `UVPlot`, `beamPlot`, `modelPlot`, `dirtyPlot`, `spherePlot`, plus all `*Plot.imshow` returns and text annotations
- Widget dictionaries: `self.wax` (axes) and `self.widget` (Slider/Button)
- Locking flags: `self.lock` (slider re-entrancy), `self.antLock` (add/remove antenna), `self.GUIres` (master mute), `self._onSphere`, `pickAnt`

Public (callable) methods: `quit`, `__init__`, `showError`, `_getHelp`, `GUI`, `readAntennas`, `readModels`, `_changeWavelength`, `_changeCoordinates`, `_setNoise`, `_setGains`, `_prepareBeam`, `_prepareBaselines`, `_setBaselines`, `_gridUV`, `_setBeam`, `_prepareModel`, `_setPrimaryBeam`, `_plotModel`, `_plotModelFFT`, `_plotDirty`, `_plotAntennas`, `_plotBeam`, plus event handlers `_onPick`, `_onAntennaDrag`, `_onRelease`, `_onRobust`, `_onKey{Lat,Dec,H0,H1}`, `_subarrwgt`, `_gammacorr`, `_addAntenna`, `_removeAntenna`, `_onKeyPress`, `_onPress`, `saveArray`, `loadArray`, `loadModel`, `_setDiameter`, `_reduce`.

Refresh chain (the typical update path after a slider/drag event):

```
_setBaselines → _setBeam (calls _gridUV) → _plotBeam → _plotDirty → canvas.draw
```

### 9.2 `CLEANer` (≈ lines 2061–2892)

Each *Reduce data* press creates a new instance attached to `parent.myCLEAN`. Owns its own Tk Toplevel, matplotlib figure (`figCL1`, 12×6), separate frames for figures + controls, and entry widgets dict `self.entries`. Public methods: `quit`, `_ReNoise`, `_doRestore`, `_doRescale`, `_ApplyGain`, `_makeMask`, `_onPick`, `_onPress`, `_onRelease`, `_doMask`, `_AddRes`, `_reCalib`, `_reset`, `_CLEAN`, `_getHelp`, `_showFFT`, `_convSource`, `_CSRead`, `_onCSPick`. The instance can be opened multiple times (each call to `_reduce` simply spawns a new one) so that the user can compare two CLEAN runs side-by-side.

### 9.3 `UVPLOTTER` and `UVPLOTTER2`

Stateless visualizers that read `parent.beam`, `parent.dirtymap`, `parent.modelfft`, `parent.modelimTrue`, and (for `UVPLOTTER`) `parent.myCLEAN.{residuals, cleanmod, cleanmodd, cleanBeam}`. Each has a `Reload` button mapped to its `_FFTRead`.

---

## 10. Interaction and event model

- `pick_event` is dispatched to `_onPick`, which inspects `event.mouseevent.inaxes` and then `event.artist` to figure out whether the user clicked on an antenna marker, a UV track point, or simply on an image pixel (and then prints the corresponding flux/coordinate).
- `motion_notify_event` → `_onAntennaDrag`. If the user is currently dragging an antenna, the antenna position dictionary is updated, baselines for that antenna alone are recomputed (`_setBaselines(antidx)`), the partial-update path of `_gridUV` (which subtracts old pixpos and adds new ones) is taken, and only the affected images are redrawn. This is why dragging is fast even for `Nant ≈ 30`.
- `button_press_event` → `_onPress`. Handles double-click zoom (`_onPress` checks `event.dblclick`) and the 3-D sphere drag.
- `button_release_event` → `_onRelease`. Ends a drag.
- `key_press_event` → `_onKeyPress`. Handles `u/c/Z/z`. The CLEAN GUI rebinds this same handler to its own canvas.

Slider events (`on_changed`) bypass `pick_event` and go directly to `_onKey{Lat,Dec,H0,H1}`, `_onRobust`, `_changeWavelength`, `_gammacorr`, `_setDiameter`, `_subarrwgt`. All these set `self.lock` to avoid recursive triggering when one slider programmatically updates another.

---

## 11. Known issues and pedagogical caveats

1. **`saveArray` is broken on Python 3** — uses `print >> iff, ...` syntax. Bug fix: rewrite as `iff.write(...)` calls. Until then, *Save array* will raise on Python 3.
2. **Default-model name asymmetry** — the config parser writes `DefaultModel`, but the fallback variable is named `DefaultMod`. The shipped config nonetheless works because `Nebula.model` is hard-coded as the ultimate default in `__init__`.
3. **No proper W-projection** — the algorithm projects baselines onto the (u, v) plane assuming a flat sky and ignores the W term entirely. This matches the simulator's pedagogical purpose (small images, single phase centre) but means the dirty image is an FFT of the gridded visibilities rather than a true wide-field image.
4. **Image is a power of two and pixelated** — point sources are pixel-snapped (no fractional rendering); two close P-components can land on the same pixel and be summed without warning.
5. **The model FFT is computed once per image-load**, then re-used. Antenna drags do *not* refit the model FFT (no need; the source is invariant) but wavelength changes do (`_setPrimaryBeam(replotFFT=True)`).
6. **Disc renderer** uses a strict `≤ R²` test, plus a guarantee that the *closest* pixel is also marked, so very small discs render as a single pixel of the correct total flux.
7. **Restored CLEAN beam fit can fail** when (a) the main lobe doesn't have ≥ 5 pixels above 0.6× peak, or (b) `leastsq` diverges. The simulator handles both with a popup and falls back to a delta-function restoring beam, marking that situation visually.
8. **Briggs robustness is approximate** — the formula `(5·10^(-R))² · (2·Nbas·nH) / Σ totsampling²` follows the standard CASA convention but cannot recover the rigorous Briggs result on highly non-uniform UV distributions.
9. **No bandpass, no spectral resolution** — λ is a single scalar (`self.wavelength[2]`); there is no channel axis.
10. **No polarisation** — visibilities are scalar real images.
11. **Subarrays are rigidly two** — `Nant2 > 1` triggers subarray mode, and the rest of the code assumes ≤ 2 subarrays.
12. **GUI race conditions during heavy drags** — comments in `_addAntenna` / `_removeAntenna` use `self.antLock` to debounce; this is fragile and the README warns that Windows refresh can lag with many antennas.

---

## 12. PyInstaller packaging

Two `.spec` files are bundled:

- `APSYNSIM.spec` (Windows) targets `z:\\APSYNSIM\\SCRIPT\\APSYNSIM.py`, with `hiddenimports=['scipy.special._ufuncs_cxx', 'mpl_toolkits', 'mpl_toolkits.mplot3d']`, console-mode EXE, icon `APSYNSIM_icon_small.ico`, UPX compression.
- `APSYNSIM_MAC.spec` (macOS) is the same but with relative paths and the recommendation to run PyInstaller with `-w` (window mode, no console).

Compile recipe (`COMPILE/compile.txt`):

```
e:\PYTHON\python.exe e:\PyInstaller-2.1\pyinstaller.py -y z:\APSYNSIM\SCRIPT\APSYNSIM.spec
```

Note: `__init__.py` must be created in `mpl_toolkits/` of your Python install for PyInstaller to find the 3-D toolkit.

Both spec files reference `APSYNSIM.py`, but the bundled scripts are `APSYNSIM2.py`/`APSYNSIM3.py`; specs must be edited before use.

---

## 13. Tutorial / exercises

`APSYNSIM_EXERCISES.pdf` (Iván Martí-Vidal, Onsala / Chalmers, 18 Sept 2016) ships in the repo. Contents (from the reviewed pages):

- **Section 1.1 Overview** — Python/NumPy/Matplotlib lineage, GPLv3, original Launchpad URL.
- **Section 1.2 Main window. The GUI** — six-pane layout walkthrough for ARRAY CONFIGURATION, UV PLANE, DIRTY BEAM, MODEL IMAGE, DIRTY IMAGE, plus the sliders and program controls.
- Special key listing for `Z`/`z`/`c`/`u` matches the implementation.
- Rest of the booklet (not extracted here) walks through guided exercises on UV coverage as a function of latitude/declination/HA, weighting trade-offs, source brightness recovery, primary-beam correction, subarrays, antenna-gain corruption, and CLEAN deconvolution.

The PDF is the canonical pedagogical companion — refer to it when running this simulator with students.

---

## 14. How APSYNSIM relates to RRIVis

This package is not used by RRIVis at runtime — it is included as one of many reference simulators inside `simulators/`. Useful comparisons to RRIVis (`src/rrivis/`):

| Aspect | APSYNSIM | RRIVis |
|---|---|---|
| Visibility model | Direct grid-FFT of a model image (no per-source RIME) | Per-source RIME `V_pq = Σ J_p · C · J_q^H` (`core/visibility.py`) |
| Polarisation | None | Full Stokes via `core/polarization.py` (with the 1/2 coherency factor) |
| Spectral axis | Single scalar λ | Multi-channel `frequencies` arrays everywhere |
| Sky model | P/G/D + raster image, max ~tens of components | `SkyModel` dataclass with point-source / HEALPix payloads, multi-catalog loaders, BBS, FITS, PRISim, etc. (`core/sky/`) |
| Backend | NumPy + scipy.fft | Pluggable: NumPy, JAX, Numba (`backends/`) |
| Calibration | Manual gain corruption in CLEAN GUI; no calibration solver | Jones-term framework (`core/jones/`) covering the 8 standard terms plus extended F/W/C/H/Ee/etc. |
| Imaging | Briggs-weighted dirty image + Hogbom CLEAN | RRIVis is a *forward simulator* — imaging is delegated to downstream tools (WSClean, RASCIL, etc.) |
| Output | Tk live plots only | HDF5/CASA Measurement Set/YAML (`io/`) |

The two complement each other: APSYNSIM is a single-file teaching GUI; RRIVis is an installable, scriptable, GPU-capable batch simulator with full polarisation and Jones machinery.

---

## 15. Quick reference: file format cheat-sheets

### 15.1 Minimal `.array`

```text
LATITUDE     45.0
DECLINATION  60.0
HOUR_ANGLE   -1.5  1.5
WAVELENGTH   3.e-3   21.e-2          # metres
DIAMETER     25.0                    # optional, metres
ANTENNA      0.000e+00   0.000e+00   # metres, E-W  N-S
ANTENNA      1.000e+03   0.000e+00
# ...
ANTENNA2     -19.4       125.8       # optional second subarray
```

### 15.2 Minimal `.model`

```text
IMSIZE  2.0          # half-width arcsec, optional (auto if omitted)
G   0.0   0.4   1.0   0.1     # type RA Dec Flux σ
D   0.0   0.0   2.0   0.5     # type RA Dec Flux Radius
P  -0.4  -0.5   0.1            # type RA Dec Flux
IMAGE  Crab.png  1.0          # raster, peak Jy/pixel
```

### 15.3 Minimal `APSYNSIM.config`

```text
Npix = 512
nH = 200
DefaultMod = 'Nebula.model'
DefaultArray = 'Long_Golay_12.array'
```

---

## 16. Pointers into the source

A few useful jump-points when reading `SCRIPT/APSYNSIM3.py`:

- Class entry: line 184 `class Interferometer`
- Defaults & config parse: lines 217–246
- Antenna file parser: lines 501–624 (`readAntennas`)
- Model file parser: lines 626–688 (`readModels`)
- UV gridding kernel: lines 929–1035 (`_gridUV`)
- Briggs weighting & beam FFT: lines 1037–1062 (`_setBeam`)
- Image construction (P/G/D + IMAGE): lines 1064–1132 (`_prepareModel`)
- Primary-beam apodisation: lines 1134–1156 (`_setPrimaryBeam`)
- Dirty image: lines 1228–1271 (`_plotDirty`)
- Antenna drag: lines 1609–1633 (`_onAntennaDrag`)
- Zoom-on-double-click logic: lines 1835–1963 (`_onPress`)
- Save/Load: lines 1965–2058
- CLEAN class: lines 2061–2892
- CLEAN beam fit: lines 2606–2672 (`_reset`)
- CLEAN main loop: lines 2687–2786 (`_CLEAN`)
- 1×3 FFT plotter (`u` key): lines 2895–3072 (`UVPLOTTER2`)
- 2×3 FFT plotter (CLEAN-side): lines 3075–3374 (`UVPLOTTER`)
- Application entry: lines 3377–3384.
