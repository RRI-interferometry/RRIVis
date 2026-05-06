# WSClean — Exhaustive Reference

> Source tree: `simulators/wsclean/`. Version examined: **3.7** (`CMakeVersionInfo.txt` → `WSCLEAN_VERSION_STR 3.7`, `WSCLEAN_VERSION_MAJOR 3`, `WSCLEAN_VERSION_MINOR 7`, `WSCLEAN_VERSION_DATE 2026-02-16`).
> Primary author: **André Offringa** (`offringa@gmail.com`). Major contributors: Bas van der Tol, Bram Veenboer.
> License: **GPL v3**. Upstream: <https://gitlab.com/aroffringa/wsclean>. Manual: <https://wsclean.readthedocs.io/>.
> Read by: `WSClean main()` in `main/main.cpp` (34 lines) → `CommandLine::Parse` (`main/commandline.cpp`, 1580 lines) → `WSClean::RunClean()` / `RunPredict()` / `DrawModel()` in `main/wsclean.cpp` (1917 lines).

WSClean is **not a forward simulator** in the same sense as the other tools in `simulators/` (e.g. WODEN, OSKAR, RIMEz, pyradiosky). It is the *reverse* operation: a **wide-field radio interferometric imager / deconvolver / predictor**, distributed by André Offringa. This document is included alongside the simulators because:

1. WSClean's `-predict` mode performs fast image-based visibility prediction (a forward operation that is one of the standard ways to populate `MODEL_DATA` for self-cal or to validate a forward simulator's output).
2. WSClean's `-draw-model` mode renders sky-model component lists into model FITS images, which can then be predicted from. This is a sub-pixel-accurate image-domain replacement for direct prediction.
3. WSClean's gridders (w-stacking, w-gridder/DUCC, IDG, w-towers, direct-FT) are reusable libraries that share the same DFT/FFT mathematics that simulators perform in the forward direction.
4. WSClean is the canonical "ground-truth" tool for converting predicted visibilities back into images, so any RRIVis simulation eventually flows through (or compares against) WSClean output products.

The package is written in **C++20** (was C++17 through 3.x, raised to 20 in `CMakeLists.txt`). It links against Casacore (≥2.0), FFTW3 (≥3.3.5), Boost, CFITSIO, GSL, HDF5, BLAS/LAPACK, Python 3, and optionally **EveryBeam** (for primary-beam evaluation: LOFAR/MWA/VLA/SKA/LWA/ATCA/AARTFAAC), **IDG** (image-domain gridder, GPU-capable), **OpenMPI** (for `wsclean-mp`), and **ska-sdp-func** (for the experimental w-towers gridder, gated by `ENABLE_WTOWERS`).

---

## 1. The W-Stacking Idea (One-Paragraph Algorithm)

WSClean stands for **"w-stacking clean"**. W-stacking is a wide-field gridding alternative to **w-projection**. In w-projection the uv-samples are convolved (in uv space) with a w-dependent kernel before being placed on a single uv grid; in **w-stacking** the samples are placed on one of many w-layers (each a 2-D uv grid) according to their w-value, every layer is FFT'd independently, and the per-layer images are corrected by a per-pixel multiplication of the w-term phase factor and summed:

```
V_pq(u,v,w) → grid onto layer floor((w − w_min)/Δw)
              FFT each layer → I_layer(l,m)
              I(l,m) = Σ_layer  I_layer(l,m) · exp(−2πi · w_layer · (n−1))   with n = √(1−l²−m²)
```

Multiplication per pixel beats convolution per visibility for typical wide-field arrays (MWA, LOFAR), so WSClean is "an order of magnitude faster than CASA's w-projection on MWA data" (`doc/source/introduction.rst:6`). The trade-off: every w-layer must be stored / FFT'd, which costs memory and FFT time. WSClean since version 3.4 defaults to a different gridder, the **w-gridder** (DUCC implementation by Reinecke / Arras / Westermann / Enßlin, theory by Ye/Gull/Tan/Nikolic 2021), which extends the convolutional support into the w-direction and has a much smaller memory footprint while being very accurate; w-stacking remains the fast small-image option.

---

## 2. Top-Level Layout

```
wsclean/
├── chgcentre/        # phase-centre changer; standalone executable
├── distributed/      # MPI scheduler + wsclean-mp executable
├── doc/source/       # Sphinx manual (RST sources)
├── external/         # git submodules: aocommon, radler, schaapcommon
├── gridding/         # all CPU/CPU+IDG gridders, visibility modifiers
├── idg/              # IDG-specific glue (averagebeam, idgmsgridder, facetidg)
├── interface/        # public C++ API: Image() + InMemoryMs
├── io/               # FITS writers, image cache, facet/parset readers
├── main/             # CLI entry point, Settings, WSClean orchestrator
├── math/             # renderers, image ops, sub-pixel rendering
├── model/            # BBS / DP3 sky-model parser, components, SEDs
├── msproviders/      # MS readers, reordered providers, row providers
├── scheduling/       # gridding task graph, threaded + MPI schedulers
├── scripts/          # Docker recipes, Python deconvolution examples
├── structures/       # imaging table, weights, observation info, primary beam
├── system/           # process utilities, application helpers
├── tests/            # gtest unit tests + integration tests + Python tests
├── wgridder/         # DUCC w-gridder (default since 3.4): float + double
└── wtowers/          # optional ska-sdp-func w-towers gridder
```

**Build entry**: `CMakeLists.txt` at top level + `CMakeVersionInfo.txt`. Builds three executables:

| binary | source | purpose |
|---|---|---|
| `wsclean` | `main/main.cpp` | Standard imager / predictor |
| `wsclean-mp` | `distributed/wsclean-mp.cpp` | MPI-distributed imager (multi-node) |
| `chgcentre` | `chgcentre/main.cpp` | Phase-centre rotation tool |

A C++ public API is also provided (`interface/wsclean.h`) for embedding WSClean as a library — it offers a single `void wsclean::Image(const std::string& command_line_parameters, std::vector<InMemoryMs>&& ms_list)` that runs an entire imaging pipeline on in-memory visibility data without touching a Measurement Set on disk (`InMemoryMs` is defined in `interface/inmemoryms.h`).

---

## 3. Architecture (Source-Code-Level)

### 3.1 The Settings class — single source of truth

`main/settings.h` (305 lines) defines `wsclean::Settings`, a plain-data class that holds every tunable in one place. Three enums dispatch behaviour:

```cpp
enum class GridderType { WStacking, WGridder, TunedWGridder, WTowers, DirectFT, IDG, FacetIDG };
enum class DirectFTPrecision { Float, Double, LongDouble };
enum class VisibilityReadMode { kScalar, kDiagonal, kFull };
```

The five `Mode`s of operation (`Settings::Mode`):

* `ImagingMode` (default) — full imaging + optional clean.
* `PredictMode` — populate `MODEL_DATA` from existing model FITS images.
* `RestoreMode` — restore a model image onto a residual image (writes one output FITS).
* `RestoreListMode` — restore a BBS-format source list onto a residual image.
* `DrawModelMode` — sub-pixel-accurate rendering of a BBS sky model into FITS without using any MS.

Selected high-impact fields (full list in `settings.h`):

| Field | Default | Meaning |
|---|---|---|
| `gridderType` | `WGridder` | Gridder selection. Was `WStacking` before 3.4. |
| `gridder_accuracy` | `0.0` (auto) | W-gridder requested accuracy ε; ≤1.01e-5 ⇒ double-precision DUCC. |
| `nWLayers` | 0 (auto) | Number of w-layers for w-stacking. Auto = "1 radian decorrelation". |
| `nWLayersFactor` | 1.0 | Multiplier applied to the auto value. |
| `antialiasingKernelSize` | 7 | Gridding-kernel pixel support (w-stacking). |
| `overSamplingFactor` | 1023 | Oversampling of the prolate-spheroidal/KB kernel. |
| `gridMode` | `KaiserBessel` | Kernel choice: KB, KB-no-sinc, NN, rect, gaussian, Blackman-Nuttall, Blackman-Harris. |
| `imagePadding` | 1.2 | Inversion grid padding factor. |
| `polarizations` | {Stokes I} | Output polarisations; `pol.ParseList` accepts I/Q/U/V/XX/.../RR/.../LL. |
| `weightMode` | Uniform | Briggs/uniform/natural; held in `WeightMode`. |
| `mfWeighting` | false | Auto-on with `-join-channels`. |
| `deconvolutionGain` | 0.1 | Minor-loop "Clean gain". |
| `deconvolutionMGain` | 1.0 | Major-loop ("Cotton-Schwab") gain. 1 = pure Högbom. |
| `majorIterationStrategy` | Dual | `kDual`/`kNormal`/`kFull` for auto-mask second-pass behaviour. |
| `majorAutoMaskIterations` | 2 | Cap on extra major iterations after auto-mask threshold reached. |
| `algorithmType` | `kGenericClean` | Radler algorithm: clean / multiscale / IUWT / MoreSane / Python. |
| `idgMode` | `IDG_DEFAULT` | CPU/GPU/Hybrid. |
| `applyPrimaryBeam` | false | Image-domain Mueller-matrix correction (EveryBeam). |
| `gridWithBeam` | false | A-term gridding with the beam (IDG only). |
| `applyFacetBeam` | false | Per-facet beam during gridding. |
| `facetSolutionFiles` | {} | h5parm solutions per facet. |
| `parallelDeconvolutionMaxSize` | 0 | Subimage size for Dijkstra-split parallel cleaning. |
| `temporaryDirectory` | "" | Where the reordered shadow MS is written. |
| `multiscaleScaleList` | {} | Custom scale list (else: auto powers-of-two from 4×PSF up). |
| `multiscaleDeconvolutionScaleBias` | 0.6 | Scale-selection bias; lower → larger scales preferred. |
| `multiscaleShapeFunction` | tapered-quadratic | or Gaussian (auto-on with `-save-source-list`). |
| `simulateNoise` / `simulatedNoiseStdDev` | false / 0 | Replaces every visibility with N(0, σ²). |
| `baselineDependentAveragingInWavelengths` | 0 | nλ; enables internal BDA. |
| `featherSize` | auto | Feather-zone width when stitching facets; default ≈ 1% of √(W·H). |

`Settings::Validate()` performs cross-field consistency. `Settings::Propagate(verbose)` derives e.g. padded image sizes. `Settings::GetRadlerSettings()` translates the WSClean settings into the deconvolver subsystem's settings.

### 3.2 The execution flow

`main/main.cpp` does only:

```cpp
WSClean wsclean;
if (CommandLine::Parse(wsclean, argc, argv, /*kSlave=*/false, /*kInMemoryData=*/false))
    CommandLine::Run(wsclean);
```

`CommandLine::Parse` (in `main/commandline.cpp`) reads the entire option string, fills `Settings`, and calls `Validate()`. `CommandLine::Run` dispatches:

```
Settings::mode == PredictMode      → WSClean::RunPredict()
                  RestoreMode      → restore loop with FFTConvolver / Renderer
                  RestoreListMode  → restore from BBS source list
                  DrawModelMode    → WSClean::DrawModel()
                  ImagingMode      → WSClean::RunClean()
```

`WSClean::RunClean()` (`main/wsclean.cpp`) drives the imaging loop:

1. Build the **ImagingTable** (`makeImagingTable`) — one entry per (interval × output-channel × polarisation × facet-group × facet) tuple. The table is the iteration backbone.
2. Construct gridding **task factory** + **task manager** (threaded or MPI; see `scheduling/`).
3. Initialise weights cache (`createWeightCache`, `io/imageweightcache.h`).
4. For every independent group:
   * `runFirstInversions` — PSF + dirty.
   * `runMajorIterations` — Cotton–Schwab loop: minor-cycle deconvolution (Radler) → major prediction → re-gridding → re-image.
5. `WriteModelImages` / `stitchFacets` / restore + write FITS, optional component list.

The "compute" is partitioned by the **GriddingTask** abstraction (`scheduling/griddingtask.h`). A task carries:

* operation (PSF-only, predict, dirty + predict combined, ...)
* facet group, polarisation, channel range
* `MsProviderCollection` reference (input MSes)
* an `AverageBeam` accumulator pointer for IDG
* per-task settings overrides

Tasks flow through **`GriddingTaskManager`** which is one of:

| Subclass | Backing | Used when |
|---|---|---|
| `ThreadedScheduler` | `std::thread` pool, in-process | Default |
| `MpiScheduler` | OpenMPI on master | `-np N` with `wsclean-mp` |
| `MpiWorkerScheduler` | OpenMPI worker | Slaves on `wsclean-mp` |

`compound_tasks=true` packs all facets of one channel into one task (reduces I/O when used with `-shared-facet-reads`/`-shared-facet-writes`).

### 3.3 Gridder hierarchy

All MS-aware gridders inherit from **`MsGridder`** (`gridding/msgridder.h`). They fetch visibilities from an **`MsProviderCollection`** and produce one dirty image (inversion) or write back predicted visibilities (degridding).

| Class | File | Key calls |
|---|---|---|
| `WSMSGridder` | `gridding/wsmsgridder.cpp` (560 lines) | the original w-stacking gridder; uses `WStackingGridder<float/double>` (`wstackinggridder.cpp`, 962 lines) under the hood. |
| `WGriddingMSGridder` | `wgridder/wgriddingmsgridder.cpp` (552 lines) | the DUCC w-gridder; calls `ducc0::ms2dirty` / `ducc0::dirty2ms`. Uses `WGridder<float>` for accuracy ≥ 1e-5 and `WGridder<double>` for tighter tolerances (see `wgridder.h`). Has explicit-instantiation TUs `wgridder_double.cpp` + `wgridder_float.cpp` to keep object size and compile-times bounded. Supports a *VisibilityCallbackBuffer* that lets DUCC pull corrected visibilities on-the-fly with facet H5parm/beam solutions applied lazily — this is the engine behind `-shared-facet-reads`. |
| `DirectMsGridder` | `gridding/directmsgridder.cpp` (229 lines) | brute-force DFT (no FFT). Selects via `directFTPrecision` enum: 32-bit / 64-bit / 80-128-bit long double. Diagnostic / accuracy reference. |
| `IdgMsGridder` | `idg/idgmsgridder.cpp` | calls into IDG. Supports a-terms (TEC, dl/dm, diagonal, beam, paf, fourierfit, klfit). |
| `FacetIdgMsGridder` | `idg/facetidgmsgridder.cpp` | Per-facet IDG when `-gridder facet-idg`. Allows non-IQUV polarisations through facets. |
| `WGriddingMSGridder` (TunedWGridder mode) | wgridder/ | uses DUCC's `ms2dirty_tuning` (auto-tuned support / kernel by DUCC). |
| W-towers | `wtowers/` | optional, gated on ska-sdp-func; activated via `-gridder wtowers` and tuned by `-wtowers-subgrid-size`, `-wtowers-kernel-size`, `-wtowers-w-kernel-size`, `-wtowers-padding`, `-wtowers-w-padding`, `-wtowers-accuracy`. |

The class **`MsGridderManager`** (`gridding/msgriddermanager.cpp`, 1437 lines) coordinates gridding across MSes, including *visibility modifiers* that apply on-the-fly corrections:

* **`VisibilityModifier`** (`gridding/visibilitymodifier.cpp`) holds:
  * `H5SolutionData` (h5parm gain solutions, scalar/diagonal/full-Jones, amplitude/phase/both).
  * `BeamResponseCacheChunk` (EveryBeam responses cached at `beamAtermUpdateTime` cadence).
  * `RotateSingleVisibilityToPhaseCenter<NPol>` for direction-dependent PSFs.
  * Per-row time-offset bookkeeping for sequential application.

The matrix formalism is documented inline (`wgridder.h:140-196`): for each visibility, a templated `VisibilityCallback<Mode, NPol, NParms, ApplyBeam, ApplyForward, HasH5Parm, ApplyRotation>` is materialised; this is the core inner loop.

### 3.4 The deconvolver (Radler)

WSClean delegates **all** deconvolution to **Radler**, a separate Anthropic-of-radio-astronomy library shipped as a git submodule under `external/radler/`. `radler::Settings` is computed from `Settings::GetRadlerSettings()`. Radler implements:

| Algorithm | Trigger |
|---|---|
| Generic / Högbom Clean | default (`-mgain == 1`) or no `-multiscale` |
| Cotton–Schwab Clean | any `0 < mgain < 1` |
| Multi-scale Clean | `-multiscale`; tapered-quadratic or Gaussian shapes; auto scale set |
| Adaptive Scale Pixel | `-asp` |
| IUWT (compressed sensing) | `-iuwt` |
| MoreSane | `-moresane-ext`; calls Python PyMORESANE |
| Python | `-python-deconvolution <file.py>`; `deconvolve(residual, model, psf, meta)` |

Multi-scale parameters live on the WSClean side and are forwarded into Radler:

* `multiscaleGain` (0.2 default; "0.1 may be more stable").
* `multiscaleDeconvolutionScaleBias` (0.6).
* `multiscaleConvolutionPadding` (1.1).
* `multiscaleMaxScales` (0 = unlimited).
* `multiscaleShapeFunction` (tapered-quadratic or Gaussian; FWHM_gauss ≈ 0.45·α_quadratic).

Spectral fitting in Radler:

* `kNoFitting`, `kPolynomial`, `kLogPolynomial` (`schaapcommon::fitters::SpectralFittingMode`).
* `spectralFittingTerms` (`-fit-spectral-pol` / `-fit-spectral-log-pol`).
* `forcedSpectrumFilename` (`-force-spectrum file.fits`) — Ceccotti et al. 2023 forced-spectrum method.
* `deconvolutionChannelCount` (`-deconvolution-channels`) — image with N channels but deconvolve at M ≤ N.

Auto-mask & RMS:

* `autoDeconvolutionThresholdSigma` (`-auto-threshold σ`).
* `absoluteDeconvolutionThreshold` (`-abs-threshold`).
* `autoMaskSigma` (`-auto-mask`) and `absoluteAutoMaskThreshold` (`-abs-auto-mask`).
* `localRMSMethod`: `kNone` / `kRmsWindow` / `kRmsAndMinimumWindow`.
* `localRMSWindow` (default 25 PSFs) / `localRMSStrength` (1.0; `local_rms ^ strength`) / `localRMSImage` (file).

Component optimization (after auto-mask threshold):

* `componentOptimizationAlgorithm`: `kClean` / `kGradientDescent` / `kLinearSolver` (`-component-optimization`).

### 3.5 Sky-model parser

`model/` reads BBS/DP3-format text sky models for `-restore-list`, `-draw-model`, and `-save-source-list` outputs.

* `model.{h,cpp}` — Source list container.
* `modelsource.h` / `modelcomponent.h` — single source / component records.
* `bbsmodel.h` — BBS file dialect parsing.
* `measuredsed.h`, `powerlawsed.h`, `spectralenergydistribution.h` — spectral models. Both **logarithmic-polynomial** and **ordinary-polynomial** SI (`LogarithmicSI` column).
* `tokenizer.h`, `modelparser.h` — parser primitives.
* `writemodel.h` — output BBS model (used by `-save-source-list`).

The component-list format produced via `-save-source-list` is:

```
Format = Name, Type, Ra, Dec, I, SpectralIndex, LogarithmicSI,
         ReferenceFrequency='125584411.621094', MajorAxis, MinorAxis, Orientation
s0c0,POINT,08:28:05.152,39.35.08.511,0.000748810650,[-0.00695,-0.0850],false,1.255e8,,,
s1c0,GAUSSIAN,08:31:10.37,41.47.17.131,0.000723,[0.003,-0.116],false,1.255e8,83.6144,83.6144,0
```

Logarithmic SI uses base-10 logs:
```
log S(ν) = log(S₀) + c₀·log(ν/ν₀) + c₁·log(ν/ν₀)² + …
```
Ordinary polynomial subtracts 1 in the base:
```
S(ν) = S₀ + p₀·(ν/ν₀ − 1) + p₁·(ν/ν₀ − 1)² + …
```

### 3.6 MS providers and reordering

`msproviders/` is the I/O layer. The **reordering** subsystem rewrites the MS into a temporary file ordered for sequential channel + polarisation + baseline access — essential for fast multi-pass major iterations.

| Class | Role |
|---|---|
| `ContiguousMS` | Direct read of a Casacore MS without reordering. |
| `ReorderedMsProvider` | Reads from the on-disk reordered shadow file. |
| `InMemoryProvider` | Backs the `interface::Image()` in-memory API. |
| `TimestepBuffer` | Buffers visibilities by timestep. |
| `SynchronizedMs` | Thread-safe wrapper. |
| Row providers | Streaming row-iterator pattern. |
| `Operations` | Reorder, compute weights, etc. |

Settings governing reordering:

* `doReorder` derived from `Settings::determineReorder()`.
* `forceReorder`/`forceNoReorder` (`-reorder`/`-no-reorder`).
* `reuseReorder` / `saveReorder` (`-reuse-reordered`/`-save-reordered`).
* `temporaryDirectory` (`-temp-dir`).
* `parallelReordering` (`-parallel-reordering N`; default 4 — disk-bound).
* `inMemoryData` (`-in-memory`) — read entire MS into RAM.
* `sortMsInTime` (`-sort-ms`) — re-sort in time first.

### 3.7 Image weights and tapering

`structures/imageweights.{h,cpp}` implements:

* Uniform weighting (one-pass-counted on a uv grid).
* Natural weighting (visibility weights only; identity imaging weights).
* Briggs' weighting with robustness `r ∈ [-2, 2]`. WSClean's robust ≠ CASA's robust exactly.
* Super-uniform / sub-uniform via `-super-weight` (gridding cells of `factor × pixel` size).
* Weight rank filter (`-weighting-rank-filter <factor>` and `-weighting-rank-filter-size <pixels>`; default 16). On since 2.2 with a factor of 3.

Tapers (multiplicative on top of all of the above):

* `-taper-gaussian <beamsize>` — UV-domain Gaussian; FWHM in image-domain matches beamsize.
* `-taper-tukey <λ>` — circular outer Tukey, paired with `-maxuv-l`.
* `-taper-inner-tukey <λ>` — circular inner Tukey, paired with `-minuv-l`.
* `-taper-edge <λ>` and `-taper-edge-tukey <λ>` — square-edge taper (decreases visibility coverage near grid edges).
* `-use-weights-as-taper` — separates visibility weights from imaging-weight count: visibility weights become a multiplicative taper rather than entering the per-cell weight. Important when the user wants to up-weight specific baselines via WEIGHT_SPECTRUM.

### 3.8 MF (multi-frequency) weighting

Auto-on with `-join-channels` (force off with `-no-mf-weighting`, force on with `-mf-weighting`). MF weighting grids the weights of all output channels onto a *single* grid before the gridding pass. Without MF weighting, each output channel weighs separately, which biases the channel sum towards naturally-weighted (because high uv-resolution makes each MWA single-channel track sparse). MF weighting guarantees: ∑ channel_image = (image weighted with the MF scheme).

### 3.9 Visibility-weighting modes

`-visibility-weighting-mode normal|squared|unit` selects the multiplicative factor applied to `WEIGHT_SPECTRUM`. `unit` ignores them entirely (useful for EoR power-spectrum error estimation). `squared` is a research mode.

### 3.10 Polarisation handling

Logic lives in `Settings::GetProviderPolarization` and `Settings::checkPolarizations`.

* `-pol I` → reads XX+YY (or LL+RR), takes `min(w_XX, w_YY)` for the visibility weight, then averages.
* `-pol XX,XY,YX,YY` → instrumental linear; XY outputs **two** images (real + imaginary) so the user can rotate to QU.
* `-pol IQUV` → produces all four; recommended with IDG.
* `-pol RR,RL,LR,LL` → circulars.
* `-join-polarizations` → peak-finding in `Σ_pol image_pol²`, components subtracted from each image individually.
* `-link-polarizations <pollist>` → component finding restricted to the link list, but subtraction in *all* imaged pols.
* `-squared-channel-joining` → for `-join-channels`, peak in `Σ_chan image_chan²`. Combined with `-pol QU -join-polarizations` it gives `Σ (Q² + U²)` — the canonical RM-synthesis operator. **`-fit-rm`** in addition fits each cleaned component to a sinusoidal in λ² so RM is solved component-by-component during deconvolution (Offringa & Smirnov 2017).

### 3.11 Coherency / Stokes formalism

Stokes I from XX/YY:
```
I = (XX + YY) / 2   (with weights min(w_XX, w_YY))
```
Mueller-matrix beam correction:
```
corrected = B⁻¹ · vec(V)
```
where B is the 4×4 average Mueller matrix and `vec(V)` flattens the visibility coherency. WSClean stores the Hermitian Mueller matrix as 16 separate FITS images (lower-triangle indexing in `primary_beam_component_images.rst`):
```
[0]
[1]+[ 2]i   [ 3]
[4]+[ 5]i   [ 6]+[ 7]i   [ 8]
[9]+[10]i   [11]+[12]i   [13]+[14]i   [15]
```
Real-only diagonal entries: `wsclean-beam-{0,3,8,15}.fits`. From WSClean 3.2, only entries needed for the requested polarisation are stored.

### 3.12 Faceting

Facets are polygonal subregions defined in a DS9 region file (`-facet-regions facets.reg`). The `ds9_facet_file.rst` documents the (very small) subset of DS9 syntax accepted:

```
polygon(ra_0, dec_0, ..., ra_n, dec_n) # text="ABCD"
point(ra_A, dec_A)
```

Coordinates are in **degrees**. Polygons can be convex or concave. Optional `point()` marks the direction at which DDEs are evaluated for that facet (otherwise the polygon centroid). The text label is used by DP3 (`ddecal.modelnextsteps.ABCD=…`).

Facet imaging supports:

* Per-facet **beam** correction (`-apply-facet-beam`, `-facet-beam-update <s>`, default 120 s).
* Per-facet **gain solutions** from h5parm (`-apply-facet-solutions <file> <name1[,name2]>`; e.g. `amplitude000`, `phase000`).
* Time-frequency smearing (`-apply-time-frequency-smearing`).
* DD-PSFs (direction-dependent PSF grid via `-dd-psf-grid <W> <H>`).
* Shared reads/writes (`-shared-facet-reads`/`-shared-facet-writes`) and compound tasks (`-compound-tasks`); only the wgridder supports shared reads.

`structures/facetutil.cpp` builds the facet list. `main/facetstitching.cpp` (356 lines) performs the feathered stitching of facet images back into the full image. Feather size: `-feather-size <pix>`, default `ceil(0.01 · √(W·H))`.

### 3.13 W-snapshot (Cornwell-2012-like)

Procedure (`w_snapshot_algorithm.rst`):
1. `chgcentre -minw obs.ms` — rotate phase centre to the direction orthogonal to the antenna best-fit plane (close to but not exactly zenith), giving minimal w-terms.
2. `wsclean -shift <orig_ra> <orig_dec> ...` — image in that w-minimal projection but project the image back along the tangent plane onto the original target.

Supported by the IDG, w-stacking, and (since 3.0) the w-gridder. Useful for off-zenith MWA snapshots > 20° from zenith; speed-up factor ~3 at >45°.

### 3.14 IDG (Image-Domain Gridding)

Provided by ASTRON's IDG library. Flag: `-gridder idg` (CPU/GPU/Hybrid via `-idg-mode cpu|gpu|hybrid`). Restrictions:

* Only on MSes with all four polarisations (XX, XY, YX, YY).
* Cannot be used with `-baseline-averaging`.
* Cannot image polarisations independently (must `-join-polarizations` or `-link-polarizations`).

Strengths: a-terms (TEC, dl/dm, diagonal-Jones, time-variable beam, KL/Fourier-fit, PAF) at almost no extra cost. The **a-term parset** (passed via `-aterm-config <file>`) lists corrections in the `aterms = [...]` line and configures each, e.g.:

```
aterms = [tec, dldm, diagonal, fourierfit, klfit, beam, paf]

tec.images = [aterm_tec_t1.fits aterm_tec_t2.fits]   # 5-D: RA, DEC, ANTENNA, FREQ, TIME
tec.window = raised_hann                              # tukey/hann/raised_hann/rect/gaussian

dldm.images = [aterm_dldm.fits]                      # 5-D: RA, DEC, MATRIX(2: dl, dm), FREQ, TIME
dldm.update_interval = 300

diagonal.images = [aterm_diag.fits]                  # 6-D: RA, DEC, MATRIX(4: ReXX, ImXX, ReYY, ImYY), ANTENNA, FREQ, TIME

fourierfit.solutions = solutions.h5
klfit.solutions = solutions.h5
klfit.order = 3

beam.differential = true
beam.update_interval = 120
beam.usechannelfreq = true
beam.frequency_interpolation = true

paf.antenna_map  = [ant_0 ant_1 ...]
paf.beam_map     = [00 01]
paf.beam_pointings = [-09h25m00.0s 55d49m59.0s -09h35m33.0s 54d47m29.0s]
paf.file_template = beammodels/$ANT/CygA_191120_$BEAM_$ANT_I.fits
paf.window = hann
paf.reference_frequency = 1355e6
```

Phase from a TEC value (radians, frequency in Hz):
```
phase = image[pixel] · (-8.44797245e9) / frequency
```

The **a-term kernel size** (`-aterm-kernel-size <pixels>`) controls how smooth the resampled a-term is on the IDG subgrid (default 16 with parset; 5 otherwise). Diagnostics: `-save-aterms` writes `aterm-ev0.fits` / `aterm-realxx0.fits` (eigenvalue / Jones-XX-real, mosaicked per antenna).

### 3.15 Distributed (MPI) imaging

`distributed/wsclean-mp.cpp` builds a separate executable. Topology:

* Master parses, dispatches gridding tasks across nodes, runs deconvolution itself.
* Workers (`worker.cpp`/`worker.h`) execute gridding tasks. Communication uses chunked messages (`mpibig.{h,cpp}`, `taskmessage.h`).

Operational notes (`distributed_imaging.rst`):

* **Only gridding** is parallelised across nodes; reordering and deconvolution stay on master.
* MPI distributes by output channel. `-channels-out 8 -np 8` ⇒ 8× speed-up.
* `-no-work-on-master` — keep master idle (useful when master node has limited resources).
* `-channel-to-node 0,0,1,2` — manual mapping.
* `-max-mpi-message-size <bytes>` — default 2 GB. Suffixes `m`/`g` allowed.
* On a single node, prefer `-parallel-gridding` over MPI.

### 3.16 Sub-pixel sky-model rendering (`-draw-model`)

`math/subpixelrenderer.{h,cpp}` + `math/fourierdomainrenderer.{h,cpp}` + `math/renderer.{h,cpp}` + `math/tophatconvolution.{h,cpp}` implement an image-domain rendering procedure that places sources at non-integer pixel positions accurately. Documented mathematically in `skymodel_resampling.rst`.

* **Point sources**: convolved with a sinc function whose window length is `-sinc-window-size <pix>` (default 127) — Kaiser-windowed sinc.
* **Small Gaussians** (< 50 pix): rendered in the **uv-domain** by evaluating the analytic coherency function on a `W × W` sub-grid, applying a phasor `exp(2πi(l₀·u + m₀·v))` for the off-pixel offset, taper, FFT, paste.
* **Large Gaussians** (≥ 50 pix): rendered directly via DrawGaussian (schaapcommon) sampled out to 20σ.

Rendering parameter selection (Kaiser β, window size W) is described in detail in `skymodel_resampling.rst:1129-1265`. Image extent: `d = max(l_max, m_max) + W·s₀`. Image scale: `s = s₀ / (1 + 2f₀ + 2·d·w_max(λ))` where `f₀ = √(1 + (β/π)²)/W`. Number of pixels: `N = d/s`.

CLI invocation:
```bash
wsclean -draw-model skymodel.txt \
        -draw-frequencies 150e6 1e6 \
        -draw-spectral-terms 2 \
        -sinc-window-size 256 \
        -size 1024 1024 -scale 1arcsec \
        -name skymodel observation.ms
```

`-draw-centre <ra> <dec>` overrides MS phase centre. `-draw-spectral-terms N` produces N coefficient images (term 0 = flux at ν₀, term 1 = SI, …).

### 3.17 Time-frequency smearing model

For the smearing flag `-apply-time-frequency-smearing` the implementation follows `time_frequency_smearing.rst`:

```
Δbl_time(λ) = bl(m) × ncp_uvw / c · ν · θ_rot     ( θ_rot = T_int · 2π / sidereal_day )
Δbl_freq(λ) = bl(m) / c · Δν

smearing(d) = sinc( ⟨Δbl_time, d⟩ ) · sinc( ⟨Δbl_freq, d⟩ )
```

Two simplifying assumptions: (i) over a single integration the apparent rotation in uv space is approximated as a straight line; (ii) the time and frequency smearing are treated as independent.

### 3.18 Baseline-dependent averaging

Two paths:

1. **Read DP3 BDA** since 3.1. Auto-detected via the `BDA_TIME_AXIS` subtable + `BDA_FACTORS` keywords (see `technical_bda_details.rst`). Restrictions when ingesting BDA: `-baseline-averaging`, `-even-timesteps`, `-interval`, `-odd-timesteps`, `-simulate-baseline-noise`, `-simulate-noise` are *not supported*; IDG also doesn't support BDA.
2. **WSClean's own internal BDA** (`-baseline-averaging <nλ>`). Happens during reordering. The averaging factor:
   ```
   nλ = (max_baseline_in_λ) · 2π · (T_avg in s) / 86400
   ```
   Requires `-no-update-model-required`. Cleaning, multiscale, etc. all work. Speed-ups of 4–8× reported on LOFAR / MWA. For LOFAR international stations remove them with `-maxuvw-m` *before* reordering.

---

## 4. Command-Line Reference (Complete)

This section is a flat enumeration of every option exposed by `wsclean::CommandLine::ParseWithoutValidation` (`main/commandline.cpp`). Defaults are taken from `Settings`. For options that take units, `Angle::Parse` accepts: `deg`, `amin`/`asec`, `rad`, default = degrees; `FluxDensity::Parse` accepts `Jy`/`mJy`/`uJy`, default = Jy. Integer options accept `k`/`m`/`g` suffixes.

### 4.1 General

| Option | Description |
|---|---|
| `-version` | Print version + commit hash + library availability flags (EveryBeam, IDG, WGridder, W-Towers). |
| `-help` | Print full help and exit. |
| `-quiet` / `-v` / `-verbose` | Verbosity; default normal. |
| `-log-time` | Prepend timestamps to every log line. |
| `-j <N>` | Number of CPU threads. Default: all. |
| `-parallel-gridding <N>` | Gridders run in parallel; threads-per-gridder = `j / N`. |
| `-parallel-reordering <N>` | Reorder up to N MSes in parallel; default 4 (disk-bound). |
| `-no-work-on-master` | MPI: master idle. |
| `-channel-to-node <list>` | MPI: explicit channel→node map. |
| `-max-mpi-message-size <size>` | MPI: default 2 GB; `g`/`m` accepted. |
| `-mem <%>` | Memory cap as % of total RAM. |
| `-abs-mem <GB>` | Absolute memory cap in GB. |
| `-reorder` / `-no-reorder` | Force on/off reordering. |
| `-reuse-reordered` / `-save-reordered` | Skip the reorder step / keep the reordered file. |
| `-in-memory` | Load entire MS into RAM. |
| `-sort-ms` | Sort by time before processing. |
| `-temp-dir <dir>` | Where to put reordered shadow files. |
| `-update-model-required` (default) / `-no-update-model-required` | Whether MODEL_DATA must be valid. |
| `-no-dirty` | Skip dirty image FITS output. |
| `-save-first-residual` | Output residual after first major iteration. |
| `-save-weights` | Save gridded weight FITS. |
| `-save-uv` | Save FFT(residual) as `<prefix>-uv-real.fits` + `<prefix>-uv-imag.fits`. |
| `-reuse-psf <prefix>` / `-reuse-dirty <prefix>` | Skip PSF/dirty inversion; load FITS from disk. |
| `-dry-run` | Parse and quit. |

### 4.2 Primary beam

| Option | Description |
|---|---|
| `-apply-primary-beam` | Image-domain Mueller-matrix beam correction. EveryBeam-supported instruments only. |
| `-reuse-primary-beam` | Reuse cached `-beam-N.fits` from disk. |
| `-use-differential-lofar-beam` | Apply only the *differential* part (LOFAR scenarios). |
| `-primary-beam-limit <f>` | Clip pb-corrected pixels with response below f. Default 0.005 (0.5%). |
| `-scalar-beam` | Stokes-I-only beam correction using avg of 1/XX, 1/YY (instead of inverting the full 4×4). |
| `-mwa-path <path>` | MWA beam coefficient file location. |
| `-save-psf-pb` | Also output beam-corrected PSF. |
| `-pb-grid-size <npix>` | Resolution of the coarse beam grid (default 32). |
| `-dd-psf-grid <W> <H>` | Multiple PSFs at a W×H grid across the FoV. |
| `-beam-model {Hamaker\|Lobes\|OskarDipole\|OskarSphericalWave}` | EveryBeam model selection (default: Hamaker for LOFAR, OskarSphericalWave for SKA). |
| `-beam-mode {array_factor\|element\|full}` | (Debug) SKA-only beam mode. |
| `-beam-normalisation-mode {none\|preapplied\|full\|amplitude}` | (Debug) SKA-only beam normalisation. |

### 4.3 Image weighting

| Option | Description |
|---|---|
| `-weight {natural\|uniform\|briggs <r>}` | Imaging weighting. Default uniform. Briggs uses `-weight briggs 0.5`. |
| `-super-weight <factor>` | Super-uniform (factor>1) or sub-uniform (factor<1) weighting. |
| `-mf-weighting` / `-no-mf-weighting` | Force MF weighting on/off. Auto-on with `-join-channels`. |
| `-weighting-rank-filter <factor>` | Truncate weight outliers at factor·local_RMS. Default factor=3 since 2.2. |
| `-weighting-rank-filter-size <pix>` | RMS window size; default 16. |
| `-taper-gaussian <θ>` | Gaussian taper (FWHM in image domain). |
| `-taper-tukey <λ>` / `-taper-inner-tukey <λ>` / `-taper-edge-tukey <λ>` | Tukey tapers (outer, inner, edge). |
| `-taper-edge <λ>` | Rectangular edge taper. |
| `-use-weights-as-taper` | Decouple visibility-weight from imaging-weight count. |
| `-store-imaging-weights` | Write `IMAGING_WEIGHT_SPECTRUM` column (only with `-no-reorder`). |

### 4.4 Inversion / image geometry

| Option | Description |
|---|---|
| `-name <prefix>` | Output FITS prefix. Default `wsclean`. |
| `-size <W> <H>` | Trimmed image size in pixels. Both must be even. |
| `-padding <factor>` | Inversion padding. Default 1.2. |
| `-scale <θ>` | Pixel scale; default deg, accepts asec/amin/etc. Default 0.01 deg. |
| `-predict` | PredictMode: just degrid existing model FITS images into MODEL_DATA. |
| `-continue` | Continue a previous run; reads `<prefix>-model.fits` (or `-pb`/`-fpb`) and subtracts to start from a partial model. |
| `-subtract-model` | Subtract MODEL_DATA from DATA at first iteration. |
| `-gridder {wgridder\|tuned-wgridder\|wstacking\|idg\|facet-idg\|wtowers\|direct-ft}` | Default: wgridder since 3.4. |
| `-channels-out <N>` | Split bandwidth into N output channels. |
| `-shift <ra> <dec>` | Tangent-plane shift of phase centre. New in 3.0. |
| `-facet-regions <file.reg>` | Enable faceting using DS9 polygons. |
| `-feather-size <px>` | Feather zone width. Default ≈ 1% of √(W·H). |
| `-gap-channel-division` | Auto-split bandwidth at frequency gaps (helpful for irregular spacing). |
| `-channel-division-frequencies <Hz,Hz,…>` | Manual band-split frequencies. |
| `-no-min-grid-resolution` / `-min-grid-resolution` | (default on) Predict at Nyquist resolution then upscale; aliasing penalty <1%. |
| `-make-psf` / `-make-psf-only` | Always (or only) make PSF. |
| `-skip-final-iteration` | Skip the final inversion after deconvolution. |
| `-visibility-weighting-mode {normal\|squared\|unit}` | How `WEIGHT_SPECTRUM` is applied during gridding. |
| `-baseline-averaging <nλ>` | Internal BDA in λ. |
| `-simulate-noise <σ_Jy>` | Replace each visibility with N(0, σ²). |
| `-simulate-baseline-noise <file>` | Per-baseline σ from a text file (lines `ant1 ant2 σ`). |
| `-idg-mode {cpu\|gpu\|hybrid}` | IDG backend selection. |

### 4.5 Gridder-specific

| Option | Description |
|---|---|
| `-wgridder-accuracy <ε>` | DUCC w-gridder accuracy. Default 1e-4. ≤1.01e-5 ⇒ double-precision. |
| `-wstack-nwlayers <N>` | Manual w-layers. Default = auto. |
| `-wstack-nwlayers-factor <f>` | Multiplier on auto w-layer count. |
| `-wstack-nwlayers-for-size <W> <H>` | Use nwlayers as if image were W×H. |
| `-wstack-grid-mode {kb\|nn\|rect\|kb-no-sinc\|gaus\|bn\|bh}` | W-stacking gridding kernel. |
| `-wstack-kernel-size <pix>` | W-stacking kernel support; default 7. |
| `-wstack-oversampling <N>` | W-stacking oversampling; default 1023. |
| `-wtowers-subgrid-size`, `-wtowers-kernel-size`, `-wtowers-w-kernel-size`, `-wtowers-padding`, `-wtowers-w-padding`, `-wtowers-accuracy` | W-towers tuning. |
| `-direct-ft-precision {float\|double\|ldouble}` | Direct-FT gridder precision. |
| `-compound-tasks` / `-shared-facet-reads` / `-shared-facet-writes` | Facet I/O optimisations. |

### 4.6 A-term gridding (IDG only)

| Option | Description |
|---|---|
| `-aterm-config <parset>` | A-term configuration. |
| `-grid-with-beam` | Convenience: enable beam a-term without parset. |
| `-beam-aterm-update <s>` | Beam recomputation cadence. Default 300 s. |
| `-aterm-kernel-size <px>` | Subgrid kernel; default 16 with parset, 5 without. |
| `-apply-facet-solutions <file> <names>` | h5parm solutions per facet. |
| `-apply-time-frequency-smearing` | DD/facet PSFs include sinc smearing model. |
| `-no-solution-directions-check` | Disable check that #h5 directions == #facets. |
| `-scalar-visibilities` / `-diagonal-visibilities` | Read single-pol or diagonal-only visibilities (bandwidth optimisation). |
| `-apply-facet-beam` | Per-facet beam during gridding. |
| `-facet-beam-update <s>` | Default 120 s. |
| `-save-aterms` | Save eigenvalue + Real(XX) of every a-term update. |

### 4.7 Data selection

| Option | Description |
|---|---|
| `-pol <list>` | XX/XY/YX/YY/I/Q/U/V/RR/RL/LR/LL or `iquv`. |
| `-interval <start> <end>` | Timestep selection (end exclusive). |
| `-intervals-out <N>` | Split selected interval into N output intervals (`tNNNN` suffix). |
| `-even-timesteps` / `-odd-timesteps` | Subset for noise estimation. |
| `-channel-range <start> <end>` | Channel index slice. |
| `-field <list>` | Field id(s); `-field all` images all. |
| `-spws <list>` | Spectral window indices. |
| `-data-column <name>` | Default: CORRECTED_DATA if present, else DATA. |
| `-model-column <name>` | Default MODEL_DATA. |
| `-model-storage-manager {default\|stokes-i\|sisco\|sisco-stokes-i\|sisco-diagonal}` | Storage manager for the MODEL_DATA column. |
| `-maxuvw-m`, `-minuvw-m` | Baseline range (metres). |
| `-maxuv-l`, `-minuv-l` | Baseline range (wavelengths; applied during gridding). |
| `-maxw <%>` | Drop visibilities with `w > pct · w_max`. |

### 4.8 Deconvolution

| Option | Description |
|---|---|
| `-niter <N>` | Max minor iterations. 0 = no clean. |
| `-nmiter <N>` | Max major iterations; default 12. 0 = unlimited. |
| `-auto-threshold <σ>` | Stop when peak < σ·robust_RMS. |
| `-abs-threshold <Jy>` | Absolute stopping threshold (Jy/mJy/uJy). |
| `-auto-mask <σ>` | Build mask in initial passes; continue with mask to threshold. |
| `-abs-auto-mask <Jy>` | Absolute auto-mask threshold. |
| `-auto-mask-nmiter <N>` | Cap on major iterations after auto-mask threshold. Default 2. |
| `-local-rms`, `-local-rms-strength <s>`, `-local-rms-window <#PSFs>`, `-local-rms-method {rms\|rms-with-min}`, `-local-rms-image <file>` | Spatial-RMS thresholding. |
| `-gain <g>` | Minor cleaning gain; default 0.1. |
| `-mgain <g>` | Major cleaning gain; default 1.0 (= Högbom). |
| `-mgain-boosting <b>` | Boost mgain in first iters: `mgain' = 1 − (1−mgain)^b`. Default 1. |
| `-major-iteration-mode {single\|dual\|full}` | Behaviour of auto-mask second pass. Default dual. |
| `-join-polarizations` | Joined-polarization Clean. |
| `-link-polarizations <list>` | Find peaks in subset, subtract from all. |
| `-join-channels` | MF Clean (and turns on MF weighting). |
| `-component-optimization {clean\|gradient-descent\|linear-solver}` | Strategy after auto-mask reached. |
| `-spectral-correction <ν₀> <c0,c1,…>` | Pre-correct expected source spectrum. |
| `-no-fast-subminor` | Disable sub-minor optimisation. |
| `-multiscale` | Enable multi-scale Clean. |
| `-multiscale-scale-bias <b>` | Default 0.6. |
| `-multiscale-max-scales <N>` | Limit auto scales. |
| `-multiscale-scales <list>` | Manual scale list (px). |
| `-multiscale-shape {tapered-quadratic\|gaussian}` | Default tapered-quadratic. |
| `-multiscale-gain <g>` | Sub-minor gain; default 0.2 (try 0.1 if unstable). |
| `-multiscale-convolution-padding <p>` | Default 1.1. |
| `-asp` | Adaptive Scale Pixel. |
| `-no-multiscale-fast-subminor` | Disable fast sub-minor optimisation. |
| `-python-deconvolution <file.py>` | Custom Python deconvolver. |
| `-iuwt` / `-iuwt-snr-test` / `-no-iuwt-snr-test` | IUWT compressed sensing. |
| `-moresane-ext <bin>` / `-moresane-arg <args>` / `-moresane-sl <list>` | MoreSane (PyMORESANE). |
| `-save-source-list` | Output `<prefix>-sources.txt` BBS list (auto-enables Gaussian shape). |
| `-clean-border <%>` | No-clean border. |
| `-fits-mask <file>` / `-casa-mask <file>` | Provide a mask. |
| `-horizon-mask <θ>` | Below-horizon protection mask. |
| `-no-negative` / `-negative` / `-stop-negative` | Negative-component policy. |
| `-fit-spectral-pol <N>` / `-fit-spectral-log-pol <N>` | Polynomial / log-polynomial spectral fit per component. Requires `-join-channels`. |
| `-fit-rm` | Per-component RM fit during cleaning. |
| `-force-spectrum <fits>` | Forced-spectrum deconvolution (Ceccotti+ 2023). |
| `-deconvolution-channels <N>` | Average to N for deconvolution; interpolate after. |
| `-squared-channel-joining` | Σ Q² (or Σ Q²+U²) peak finding. |
| `-parallel-deconvolution <maxsize>` | Sub-image cleaning (Dijkstra split). |
| `-deconvolution-threads <N>` | Limit deconvolution threads (memory pressure). |

### 4.9 Sky-model rendering

| Option | Description |
|---|---|
| `-draw-model <skymodel.txt>` | Render a BBS sky model into FITS. |
| `-draw-centre <ra> <dec>` | Override phase centre. |
| `-draw-frequencies <ν₀> <Δν>` | Central freq / bandwidth (Hz). |
| `-sinc-window-size <px>` | Default 127. |
| `-draw-spectral-terms <N>` | Number of coefficient images. |

### 4.10 Restore mode

| Option | Description |
|---|---|
| `-restore <residual.fits> <model.fits> <output.fits>` | Restore model onto residual; exits afterwards. |
| `-restore-list <residual.fits> <list.txt> <output.fits>` | Same but model is a BBS list. |
| `-beam-size <arcsec>` | Circular beam FWHM. |
| `-beam-shape <maj_asec> <min_asec> <PA_deg>` | Elliptical beam. Units overridable per token. |
| `-fit-beam` / `-no-fit-beam` | Toggle PSF fitting. |
| `-beam-fitting-size <factor>` | Fit box size = factor × theoretical FWHM. Default 10. |
| `-fit-beam-with-negatives` / `-fit-beam-without-negatives` | Default: with. |
| `-theoretic-beam` | Use longest-baseline theoretical beam. |
| `-circular-beam` / `-elliptical-beam` | Force shape. |

---

## 5. FITS Output Conventions

WSClean's FITS files use the standard radio convention: 4 axes — RA, Dec, Frequency, Polarisation. For `-channels-out N`, N+1 image FITS files are produced per polarisation: `wsclean-0000-image.fits`, …, `wsclean-NNNN-image.fits`, plus `wsclean-MFS-image.fits` (the weighted sum, "multi-frequency synthesis").

For each output channel (and the MFS), each polarisation may produce up to seven FITS files:

| Suffix | Meaning |
|---|---|
| `-dirty.fits` | Pre-deconvolution dirty image. |
| `-image.fits` | Restored (clean component model + residual). |
| `-residual.fits` | Image after model subtraction. |
| `-model.fits` | Clean component model. |
| `-psf.fits` | PSF for that channel. |
| `-uv-{real,imag}.fits` | (with `-save-uv`). |
| `-weights.fits` | Gridded weight (with `-save-weights`). |

Beam-corrected variants append `-pb` (image-plane beam correction or facet correction) or `-fpb` (facet beam) before the suffix. Auto-mask outputs add `-auto-mask` to the suffix.

WSClean-specific FITS keywords (from `fits_keywords.rst`):

```
WSCVERSI   - WSClean version
WSCVDATE   - Version date
WSCNWLAY   - nWLayers (w-stacking)
WSCDATAC   - Data column
WSCWEIGH   - Weighting (textual)
WSCGKRNL   - Gridding kernel size
WSCCHANS / WSCCHANE  - Channel range (when set)
WSCTIMES / WSCTIMEE  - Time range (when set)
WSCFIELD   - Field id
WSCNVIS    - Gridded visibility count
WSCENVIS   - Effective visibility count (natural weighting)
WSCVWSUM   - Sum of visibility weights
WSCIMGWG   - Weight when averaging images together
WSCNORMF   - Normalisation factor applied (useful for K conversion)
WSCTHRES   - Manual threshold

# clean
WSCNITER   - max minor iterations
WSCGAIN    - minor gain
WSCMGAIN   - major gain
WSCNEGCM   - negative components allowed?
WSCNEGST   - stop on negative?
WSCMAJOR   - actual major iterations executed
WSCMINOR   - actual minor iterations executed
```

`CDELT3` is the channel **width**, not the inter-channel spacing (these can differ when overlapping bands are imaged together). `CRVAL3` is the central frequency.

---

## 6. Algorithms (Detailed)

### 6.1 Cotton–Schwab (major iterations)

WSClean uses the Cotton–Schwab loop whenever `0 < mgain < 1`:

```
1. Initial inversion: dirty image, PSF (no model so far).
2. Major iteration:
   a. Run Radler minor cycle until peak ≤ (1 − mgain) · peak_initial OR auto-threshold reached.
   b. New components are added to the model image.
   c. Predict residual visibilities from the cumulative model and grid (this is also the MODEL_DATA write).
   d. Subtract from data → re-image to get the next residual.
3. Repeat until threshold or nmiter reached.
4. Final inversion (skipped with -skip-final-iteration).
5. Restore: convolve model with the fitted Gaussian beam, add residual.
```

If `mgain == 1` the loop is purely image-domain Högbom: no MODEL_DATA gets written. Use `-mgain 0.9999` to "force" a Cotton-Schwab-with-mostly-Högbom that still updates MODEL_DATA — useful when the user wants self-cal without paying for major iterations.

The first-iteration boost (`-mgain-boosting b`) replaces mgain with `mgain' = 1 − (1 − mgain)^b` for the first pass and half the boost for the second.

### 6.2 Auto-mask & RMS strategy

`-auto-threshold σ`: stop when peak ≤ σ · MAD-derived RMS.
`-auto-mask σ`: keep cleaning until peak ≤ σ · RMS, but only place new components in pixels that have already received a component. Then continue to the user-specified threshold (e.g. `-auto-threshold 0.3`).

`majorIterationStrategy` (`-major-iteration-mode`) controls behaviour after the mask has been built:

| Mode | Behaviour |
|---|---|
| `single` | Each major iter ends at mgain. |
| `dual` (default) | During mask build, a second major iter is started using the so-far-built mask. |
| `full` | Like dual, but ignores mgain inside the second iter. |

For diverging images (multi-scale large kernels at the image edge, NaN peaks), `-major-iteration-mode single` is the conservative fallback.

### 6.3 Multi-scale (Offringa & Smirnov 2017)

For each scale α (delta + powers of 2 up to image size, unless `-multiscale-scales` is given):

1. Convolve residual with the scale kernel S_α(r) (tapered-quadratic by default; Gaussian if `-save-source-list`).
2. Find peak across all scale-convolved images, weighted by `α^bias` (bias = `-multiscale-scale-bias`).
3. The selected (scale, position) is the new "scale component".
4. Sub-minor loop fits & subtracts within that scale to drain the local peak quickly (`-no-multiscale-fast-subminor` to disable).

Tapered-quadratic to Gaussian width: `σ = (3/16)·α`, FWHM ≈ 0.45·α.

Scale bias semantics: lower → larger scales preferred. With Briggs +0.5 LOFAR imaging, 0.7 was reported to be better than the 0.6 default.

### 6.4 RM-synthesis-friendly Clean

```bash
wsclean -pol QU -fit-rm -join-polarizations \
        -join-channels -squared-channel-joining \
        -channels-out N \
        -niter ... -mgain ... -auto-threshold ... obs.ms
```

With this combination, peak-finding happens in `Σ_chan (Q_chan² + U_chan²)`, and each component is fit to a sinusoidal in λ² (so RM is solved component-by-component). The MFS image is still the linear average (not a sum-of-squares).

### 6.5 Forced-spectrum deconvolution

`-force-spectrum <fits>` (Ceccotti et al. 2023): the FITS file (with as many planes as `-fit-spectral-pol` terms minus one) provides spatially varying SI/curvature/etc. that constrains the polynomial fit during deconvolution. Used when the user wants to inject a known spectral prior (e.g. from a deeper survey at a different band).

### 6.6 Parallel deconvolution

`-parallel-deconvolution <maxsize>`: image is split into ≤ maxsize × maxsize subimages by **Dijkstra's algorithm**, choosing the lowest-absolute-pixel-sum path near the natural splitting boundary. The split is recomputed before every major iteration. Per-subimage Radler instances run in parallel; their results are stitched and feathered together. Not the same as facet-based imaging — parallel deconvolution is purely a deconvolution-time optimisation.

### 6.7 Restoring beam fitting

`-fit-beam` (default if a PSF was made):

1. Estimate FWHM from longest projected baseline + frequency.
2. Fit a 3-parameter Gaussian (major, minor, PA) to a 10× FWHM box around the central PSF pixel.
3. If the fit residual / shape is suspicious, iteratively re-pick the box.

`-circular-beam` constrains minor = major. `-fit-beam-without-negatives` excludes negative pixels from the LSQ fit (default: include).

`-theoretic-beam` skips the fit and writes the predicted Gaussian from longest projected baseline only (faster, slightly less accurate).

### 6.8 Sub-pixel rendering math

See `skymodel_resampling.rst`. Point-source rendering uses a Kaiser-windowed sinc:

```
I(l, m) for a point at (l₀, m₀) is sampled as:
    I[i, j] = I₀ · sinc((i·dl − l₀) / dl) · K(β; (i − l₀/dl) / W)
where K is the Kaiser window of size W and shape parameter β.
```

Frequency response side-lobe level controls Kaiser β. First null:

```
f₀ = √(1 + (β/π)²) / W
```

Image required:

```
N = N_bb · (1 + 2·f₀)
```

Direct rendering of large Gaussians: out to 20σ ⇒ value `e^{-200} ≈ 1.4·10⁻⁸⁷`. UV-domain rendering for medium Gaussians: phase-shift in uv, FFT, paste at the right pixel.

---

## 7. Public C++ API for embedding

`interface/wsclean.h`:

```cpp
namespace wsclean {
    void Image(const std::string& command_line_parameters,
               std::vector<InMemoryMs>&& ms_list);
}
```

`interface/inmemoryms.h` defines:

```cpp
struct InMemoryRow {
    std::vector<std::complex<float>> data;        // [pol × freq], pol fastest
    std::vector<float>               weights;     // same shape; w==0 ⇒ flag
    std::array<double, 3>            uvw;
    double   time;
    uint32_t data_desc_id;
    uint32_t field_id;
    uint32_t antenna1, antenna2;
};

struct InMemoryMs {
    std::vector<InMemoryRow>          rows;
    double                            phase_centre_ra, phase_centre_dec;  // rad
    aocommon::MultiBandData           bands;
    std::vector<std::string>          antenna_names;
    std::vector<aocommon::PolarizationEnum> polarizations;
    bool                              has_frequency_bda;
    bool                              has_time_bda;
    std::string                       telescope_name;
    std::string                       observer;
    std::string                       field_name;
    double                            start_time;   // MJD seconds
    double                            interval;     // seconds (no-BDA case)
};
```

This is the path RRIVis (or any forward simulator) would use to drive WSClean directly with simulated visibilities, avoiding writing a Casacore MS to disk.

## 8. Python deconvolution interface

`-python-deconvolution my.py` invokes a function with the signature:

```python
def deconvolve(residual, model, psf, meta):
    """
    residual : numpy.ndarray, shape (n_chan, n_pol, h, w), in-place modifiable
    model    : numpy.ndarray, shape (n_chan, n_pol, h, w), in-place modifiable
    psf      : numpy.ndarray, shape (n_chan, h, w), shared across pols
    meta     : object exposing major_iter_threshold, final_threshold, mgain,
               iteration_number, max_iterations, square_joined_channels,
               channels[i].frequency, channels[i].weight,
               spectral_fitter.fit(spectrum), spectral_fitter.fit_and_evaluate(spectrum)
    Returns: dict with keys 'residual', 'model', and optionally 'iteration_number'
    """
```

Examples in `scripts/python-examples/`:
* `simple-deconvolution-example.py` — single-pol, single-channel Högbom.
* `mf-deconvolution-example.py` — uses `meta.spectral_fitter` for MF deconvolution.
* `deconvolution-error-test.py` — error-injection test harness.

---

## 9. The `chgcentre` Tool

`chgcentre/main.cpp`. Standalone utility for changing the phase centre of an MS. Recomputes `UVW` from antenna positions, phase centre, and time, and phase-rotates the visibilities.

```
chgcentre [options] <ms> <new ra> <new dec>
```

RA: `00h00m00.0s` or `00:00:00.0`. Dec: `00d00m00.0s` or `00.00.00.0`. With no RA/Dec, it prints current phase centre + zenith + min-w direction.

Options:

| Flag | Meaning |
|---|---|
| `-geozenith` | Per-timestep zenith RA/Dec; produces a non-standard MS. |
| `-flipuvwsign` | Flip UVW signs. Required for LOFAR (historical reasons). |
| `-minw` | Rotate to the direction orthogonal to the antenna best-fit plane (close to but not exactly zenith; gives minimum w-values). |
| `-zenith` | Average zenith. |
| `-only-uvw` | Update UVW without phase-rotating. |
| `-shiftback` | Project visibilities back to original phase centre after rotation (legacy; replaced by WSClean's `-shift` since 3.0). |
| `-f` | Force recalculation. |
| `-datacolumn <name>` | Phase-rotate a specific column. Default: DATA + MODEL_DATA + CORRECTED_DATA. |
| `-from-ms <ms>` | Rotate to match another MS. |

---

## 10. Performance Guide

From `computational_performance.rst` + `parallelization.rst`:

* **Pick the right gridder for your image size**:
  * Small images (≤5k px): w-stacking with `-parallel-gridding`.
  * Medium (5k–10k): w-gridder.
  * Large (>10k): IDG (especially on GPU).
  * Reference: direct-FT (slow but exact).
* `-parallel-gridding N` is preferred over MPI on a single node.
* `-parallel-reordering N` (default 4) is disk-bound; tune per-disk.
* `-parallel-deconvolution maxsize` for big images, especially when IDG shifts the bottleneck back to deconvolution.
* W-stacking: number of w-layers nearly linear in cost. If `chgcentre -minw` reduces w-spread, use it.
* Don't include baselines beyond the imaging resolution. Use `-maxuvw-m` to filter international stations early.
* Don't keep your data in many small MSes; concatenate first with DP3.
* Run a `-dry-run` first to inspect the imaging-table and resource estimates.
* Memory model in `WStackingGridder::PrepareWLayers` (`gridding/wstackinggridder.cpp:65-113`): each FFT thread holds 5× imageSize buffers (2 complex + 1 real); per pass, w-layers are sized to fit the remaining memory. If memory is too tight, the number of FFT threads is reduced.

---

## 11. Self-Calibration Workflow

WSClean is one piece of a self-cal pipeline. Three patterns (`selfcal.rst`):

1. **Calibrate from MODEL_DATA**:
   ```
   wsclean -mgain 0.8 -niter ... obs.ms       # writes MODEL_DATA
   DP3 ddecal.modeldatacolumns=[MODEL_DATA] ...
   ```

2. **Calibrate from a component list**:
   ```
   wsclean -save-source-list -multiscale ... obs.ms
   DP3 predict.sourcedb=wsclean-sources.txt ...
   ```

3. **Predict from an existing model image**:
   ```
   wsclean -predict -name <prefix> obs.ms     # fills MODEL_DATA
   DP3 ...
   ```

For multi-pol self-cal, `-pol XX,YY -join-polarizations` keeps Stokes-I peak-finding while writing per-pol MODEL_DATA.

OTF-mode warning: CASA tasks like `ft` may set keywords that make CASA ignore MODEL_DATA. Run `delmod(vis='myobs.ms', otf=True, scr=False)` before relying on it.

---

## 12. External Submodules

`external/`:

| Submodule | URL | Role |
|---|---|---|
| `aocommon` | git.astron.nl/RD/aocommon | Headers: FITS, polarizations, multibanddata, image, threading. Header-only. |
| `radler` | git.astron.nl/RD/Radler | All deconvolution algorithms (Clean, multiscale, IUWT, Python, MoreSane, ASP, generic). |
| `schaapcommon` | git.astron.nl/RD/schaapcommon | Used modules: `ducc0` (DUCC w-gridder), `facets` (DS9 polygons + faceting), `fitters` (spectral & PSF), `h5parm` (h5 solution files), `math` (resampler, restore, image ops), `reordering` (MS-reordering primitives + `MSSelection`). |

WSClean's CMake checks (lines 50-83 of `CMakeLists.txt`) that the submodules have actually been cloned (otherwise gives a fatal error) and brings them into the include path. The build supports an optional `TARGET_CPU=<cpu>` (e.g. `haswell`, `skylake`) when `PORTABLE=OFF`.

---

## 13. Citation

If you use WSClean for science (`citing_wsclean.rst`):

* **Always cite**: Offringa et al. 2014, *MNRAS* 444, 606 — the WSClean paper.
* **Multi-scale / wideband / auto-mask**: Offringa & Smirnov 2017, *MNRAS* 471, 301.
* **IDG**: Van der Tol, Veenboer & Offringa 2018, *A&A* 616, A27.
* **Forced spectrum**: Ceccotti et al. 2023, *MNRAS* 525, 5063.

BibTeX entries are inlined in `doc/source/citing_wsclean.rst`.

---

## 14. Quick-Reference Recipes

```bash
# 1. Standard MWA imaging with auto-threshold + Cotton-Schwab
wsclean -size 3072 3072 -scale 0.7amin -niter 10000 \
        -mgain 0.8 -auto-threshold 3 obs.ms

# 2. GLEAM-style wide-band MWA
wsclean -name obs-1068210256 -size 4000 4000 -niter 1000000 \
        -mgain 0.95 -weight briggs -1.0 -scale 0.75amin \
        -auto-threshold 1 -auto-mask 5 -multiscale \
        -channels-out 4 -join-channels \
        -pol xx,yy -join-polarizations 1068210256.ms

# 3. Predict from existing model
wsclean -predict -name my-image obs.ms

# 4. Imaging with facets + h5parm + facet beam
wsclean -size 8192 8192 -scale 1asec -mgain 0.7 \
        -niter 1000000 -auto-threshold 3 \
        -gridder wgridder -facet-regions facets.reg \
        -apply-facet-solutions sol.h5 amplitude000,phase000 \
        -apply-facet-beam -shared-facet-reads obs.ms

# 5. RM-synthesis cube
wsclean -pol QU -fit-rm -join-polarizations \
        -join-channels -squared-channel-joining \
        -channels-out 100 -niter 100000 -mgain 0.8 \
        -auto-threshold 1 -auto-mask 5 -size 1024 1024 -scale 1amin obs.ms

# 6. IDG with full beam correction (LOFAR)
wsclean -gridder idg -idg-mode hybrid -grid-with-beam \
        -pol iquv -link-polarizations i \
        -size 6000 6000 -scale 1.5asec \
        -niter 1000000 -mgain 0.7 -auto-threshold 3 obs.ms

# 7. W-snapshot off-zenith MWA
chgcentre -minw obs.ms
wsclean -shift 08h20m00.0s -42d45m00s -size 3072 3072 -scale 0.5amin \
        -mgain 0.8 -auto-threshold 3 obs.ms

# 8. Spectral cube with MF weighting
wsclean -channels-out 768 -mf-weighting -weight briggs -1 obs.ms

# 9. Multi-node imaging (8 channels per node, 4 nodes)
mpirun --hostfile hosts -np 4 wsclean-mp \
       -channels-out 32 -size 10000 10000 -scale 1asec \
       -mgain 0.8 -niter 50000 -auto-threshold 3 obs.ms

# 10. Render BBS sky model into FITS for image-based predict
wsclean -draw-model skymodel.txt -draw-frequencies 150e6 1e6 \
        -draw-spectral-terms 2 -sinc-window-size 256 \
        -size 1024 1024 -scale 1arcsec -name skymodel obs.ms
```

---

## 15. Caveats / Practical Notes

* `-size W H` requires **even** W and H.
* Pixel scale is square: only one `-scale` value applies to both axes.
* Multiple MSes must share the **same phase centre**; otherwise rotate first with `chgcentre` or use `-combine-pointings` with IDG.
* `-channels-out N` divides the *combined* bandwidth across all input MSes; `-interval` and `-intervals-out` operate on the **first** MS's timestep indexing.
* `-baseline-averaging` requires `-no-update-model-required` and is incompatible with IDG.
* `-store-imaging-weights` only updates the column when reordering is **disabled** (`-no-reorder`).
* IDG only works on MSes with all four linear polarisations.
* `-continue` will not work if the previous run used `-no-update-model-required` (no MODEL_DATA present); pre-predict from the model image first.
* Image dimensions, phase centre, and pixel scale must match between the original and the `-continue` run.
* Cleaning may diverge in multi-scale at very large scales near the image edge or when mgain is too high relative to PSF sidelobes (`diverging_clean.rst`). Mitigations: cap scales (`-multiscale-max-scales`), drop `mgain` to 0.6–0.7, use `-major-iteration-mode single`, enlarge the image, or stop earlier (`-nmiter`).
* Frequency-axis FITS: `CDELT3` is *channel width*, not channel-to-channel spacing — they can differ when overlapping bands are imaged together.
* WSClean does **not** look at the frequency axis of input model FITS files when predicting at a different frequency than the model was made — fluxes are *not* extrapolated.

---

## 16. Glossary of WSClean-Specific Terms

| Term | Meaning |
|---|---|
| **W-stacking** | Wide-field gridding by stacking 2-D uv grids per w-layer, FFTing, then per-pixel correcting and summing. WSClean's namesake. |
| **W-gridder** | DUCC implementation: convolutional gridding extended to w with an analytic kernel. Default since 3.4. |
| **W-towers** | (Optional) ska-sdp-func gridder; subgrid-based extension of w-gridding. |
| **W-snapshot** | Imaging in a w-minimised projection then projecting back to the target via tangent-plane shift. |
| **MGAIN** | Major iteration gain — fraction of peak removed before going back to visibilities. |
| **GAIN** | Minor iteration gain — fraction of peak removed per minor cycle. |
| **MFS image** | Multi-Frequency-Synthesis image: weighted average of all output channels. |
| **MF weighting** | Single-grid weighting across all output channels (auto-on with `-join-channels`). |
| **Auto-mask** | Iterative mask construction during clean (Offringa & Smirnov 2017). |
| **Cotton–Schwab** | Major-iteration scheme alternating image-domain Clean with vis-domain prediction. |
| **Högbom** | Pure image-domain Clean (no major iterations). |
| **Facet** | Polygonal subregion with its own DDE correction (h5parm and/or per-facet beam). |
| **Feather** | Smooth blending of facet boundaries during stitching. |
| **A-term** | Direction-dependent gain effect (beam, TEC, dl/dm, ...) applied during gridding. |
| **DD-PSF** | A grid of PSFs at different points in the image (for parallel deconvolution). |
| **BDA** | Baseline-Dependent Averaging. |
| **ASP** | Adaptive Scale Pixel multi-scale variant. |
| **IUWT** | Isotropic Undecimated Wavelet Transform; compressed-sensing deconvolution. |
| **Radler** | Library implementing all WSClean deconvolution algorithms (`external/radler`). |
| **EveryBeam** | Library for primary beam evaluation across instruments (`external` dependency). |
| **IDG** | Image Domain Gridder (`external` dependency); GPU-friendly w/a-term gridder. |
| **PAF** | Phased-Array Feed (Apertif, ASKAP). |
| **DUCC** | Distinctly Useful Code Collection — Reinecke's library underlying the w-gridder. |
| **schaapcommon** | ASTRON shared library used by WSClean and DP3. |
| **MSSS** | Multi-frequency Snapshot Sky Survey (LOFAR). |

---

## 17. Where to Read Further (in this checkout)

| Topic | File |
|---|---|
| Top-level orchestrator | `main/wsclean.cpp`, `main/wsclean.h` |
| Settings | `main/settings.h`, `main/settings.cpp` |
| Command-line parser | `main/commandline.cpp` |
| W-stacking gridder | `gridding/wstackinggridder.cpp` |
| W-gridder (DUCC) | `wgridder/wgridder.h`, `wgridder/wgriddingmsgridder.cpp` |
| Direct-FT gridder | `gridding/directmsgridder.cpp` |
| MS gridder + manager | `gridding/msgridder.cpp`, `gridding/msgriddermanager.cpp` |
| Visibility modifier (h5/beam) | `gridding/visibilitymodifier.cpp` |
| IDG glue | `idg/idgmsgridder.cpp`, `idg/facetidgmsgridder.cpp` |
| Imaging table | `structures/imagingtable.cpp` |
| Image weights | `structures/imageweights.cpp` |
| Primary beam (EveryBeam) | `structures/primarybeam.cpp` |
| Facet utilities | `structures/facetutil.cpp`, `main/facetstitching.cpp` |
| Renderers (sub-pixel) | `math/subpixelrenderer.cpp`, `math/fourierdomainrenderer.cpp`, `math/renderer.cpp` |
| FITS writer | `io/wscfitswriter.cpp`, `io/componentlistwriter.cpp` |
| Image cache | `io/imageweightcache.h`, `io/cachedimageset.h` |
| Sky-model parser | `model/model.cpp` (and headers in `model/`) |
| Reordering / MS providers | `msproviders/*.cpp` |
| Threaded scheduler | `scheduling/threadedscheduler.cpp` |
| MPI scheduler | `scheduling/mpischeduler.cpp`, `scheduling/mpiworkerscheduler.cpp` |
| Distributed entry | `distributed/wsclean-mp.cpp`, `distributed/worker.cpp` |
| Public API | `interface/wsclean.h`, `interface/inmemoryms.h` |
| Python deconvolution examples | `scripts/python-examples/*.py` |
| Sphinx manual | `doc/source/*.rst` |
