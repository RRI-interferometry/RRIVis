# DP3 — Default Pre-Processing Pipeline

> Exhaustive technical reference for the DP3 codebase shipped under
> `simulators/DP3/` (DP3 6.5.1 / `main`, vendored from
> `git.astron.nl/RD/DP3`).
>
> This document is intentionally dense. It indexes every step, every
> ParSet group, the C++ subsystem layout, the calibration solver
> internals, the sky-model format, and the Python plug-in API. Cross
> references use `path/to/file.cc:line` so they stay navigable from a
> reader's IDE.

---

## 1. What DP3 is

DP3 (originally **NDPPP**, the New Default Pre-Processing Pipeline; the
older name is still accepted as a fallback parset filename) is the
ASTRON-developed C++ pipeline for CASA MeasurementSets (MSs). It was
forked from LOFAR Release 3.2 (Sep 2018) into the standalone
[`git.astron.nl/RD/DP3`](https://git.astron.nl/RD/DP3) repository; the
ASTRON LOFAR-tree copy is no longer maintained. Project home page:
<https://dp3.readthedocs.io>.

**Scope.** DP3 reads one (or several) MS, pipes a single time slot at a
time through a chain of *steps*, and writes (or updates) a single MS.
The same binary covers:

- Pre-processing: NaN/INF flagging, RFI flagging (AOFlagger / MAD /
  pre-flagger / UVW flagger / antenna flagger / clipper), baseline
  selection, channel selection.
- Time/frequency averaging, including baseline-dependent averaging
  (BDA), interpolation of flagged samples, upsampling.
- Phase shifting to a new phase centre, station summation
  (super-stations / superterp), data scaling for SEFD correction.
- Demixing of A-team sources (`Demixer`) and bright-source clipping
  (`Clipper`).
- Direction-independent calibration (`GainCal` — StefCal) and
  direction-dependent calibration (`DDECal`) with a rich choice of
  gain-solver algorithms and constraints.
- Sky-model prediction (`Predict`, `OnePredict`, `H5ParmPredict`,
  `WGridderPredict`, `IDGPredict`, `SagecalPredict`, BDA-aware
  `BdaGroupPredict`, optional `FastPredict`).
- Beam application/inversion through EveryBeam (`ApplyBeam`,
  `SetBeam`).
- Calibration solution application (`ApplyCal`) from H5Parm or
  ParmDB.
- Imaging hooks: `IDGImager` (snapshot images per time slot for
  AARTFAAC) and `WSCleanWriter` (write reordered visibilities for
  WSClean).
- Streaming inputs over a Unix socket (`SVPInput`) for cobalt/ALMA.
- Dynamic spectra extraction along a list of source directions
  (`DynSpec`).
- Embedded Python steps (`PythonStep` / `pythondppp`) and a Python
  package (`pythondp3` / `dp3`) that exposes the C++ pipeline.
- Conversion utilities: `Transfer` / `FlagTransfer` between low- and
  high-resolution MSs, `Combine` to add/subtract named buffers, the
  `null` step (sink).

**Heavy dependencies.** DP3 links against `casacore` (≥ 3.7.1 in 6.4),
`aoflagger` (≥ 3.1), `EveryBeam` (≥ 0.7.4 < 0.9), `aocommon`/`xtensor`,
`HDF5`, `boost` (filesystem, program-options, python, test), `Armadillo`
(optional, only for `ScreenConstraint`), `IDG/idgapi` (optional, for
`idgpredict`/`idgimager`), `ducc/wgridder`, `Schaapcommon::h5parm` for
H5Parm I/O, `pybind11`, `GSL`, `CFITSIO`, and optionally
`libdirac`/SAGECal for the LBFGS solver and `sagecalpredict` step.
`BUILD_WITH_CUDA=ON` enables `IterativeDiagonalSolverCuda` and adds
`HAVE_CUDA_SOLVER` (`CMakeLists.txt:42-66`). Other CMake toggles:
`METADATA_COMPRESSION_DEFAULT`, `USE_FAST_PREDICT`,
`ENABLE_SCREENFITTER`, `BUILD_PACKAGES`, `BUILD_TESTING`,
`BUILD_DOCUMENTATION`. Default build flags include
`-Wall -Wnon-virtual-dtor -Wzero-as-null-pointer-constant
-Wduplicated-branches -Wundef -Wvla` and explicitly **do not** use
`--ffast-math` because of NaN handling discrepancies (AST-1502).

**Repository layout.**

| Top-level directory | Role |
|---------------------|------|
| `base/` | Pipeline backbone: `DP3.cc`, `Main.cc`, `DPInfo`, `DPBuffer`, `BdaBuffer`, `Simulator`, `Simulate`, `EstimateMixed`, `PhaseFitter`, `PointSource`/`GaussianSource`, model-component visitor, version. |
| `steps/` | Every step implementation (≈55 step classes). |
| `common/` | `ParameterSet`, `Fields`, `Timer`, `Memory`, `StringTools`, `ProximityClustering`, `ValuePerStationParsing`, plus `KVpair`/`PrettyUnits`/`buffered_lane`/`Median`/`Epsilon`/`baseline_indices` utilities. |
| `ddecal/` | Direction-dependent calibration — `Settings`, `SolverFactory`, `SolutionWriter`, `SolutionResampler`, gain solvers, linear LLS solvers, constraints. |
| `model/` | Sky-model glue layer: `Patch`, `SourceDBUtil`, `SkyModelCache`. |
| `parmdb/` | Legacy ParmDB / SourceDB readers (`SourceDBCasa`, `SourceDBBlob`, `SourceDBSkymodel`, `SkymodelToSourceDB`, `Parm`, `Axis`, `Grid`, `ParmFacade`). |
| `pythondp3/` | pybind11 bindings, the embeddable `PyStep`, `dp3` Python package, queue-based test step. |
| `antennaflagger/` | Standalone outlier flagger used by the `antennaflagger` step (originally for AARTFAAC). |
| `aartfaacreader/` | AARTFAAC antenna-config helper. |
| `blob/` | Binary serialization framework reused by ParmDB. |
| `docs/` | Sphinx config, `commands.rst`, `index.rst`, every step's YAML schema (`docs/schemas/*.yml`), `idg-facetting/idg-facetting.tex`. |
| `external/` | CMake-fetched dependencies (pybind11, etc.). |
| `CMake/` | DP3-specific CMake helpers. |
| `docker/` | `ubuntu_22_04_base`, `ubuntu_24_04_base`, `py_wheel.docker`, install scripts for AOFlagger, EveryBeam, FFTW, HDF5, Lua, SAGECal, Boost, plus `make_wheels.sh`. |
| `scripts/` | Auxiliary user scripts. |
| `ci/` | GitLab CI helpers. |
| `setup.py`, `CMakeLists.txt` | pip-style and CMake builds — both call into the same CMake configuration. |
| `LICENSE`, `README.md`, `CHANGELOG.md` | GPL-3.0-or-later. |

---

## 2. Process model — how a DP3 invocation runs

### 2.1 Entry points

Three are exposed:

1. **CLI binary `DP3`** — `int main` lives in `base/Main.cc` and
   forwards to `dp3::base::ExecuteFromCommandLine(argc, argv)` in
   `base/DP3.cc:479`. The CLI accepts a parset file as the first
   non-`key=value` argument plus any number of `key=value` overrides:

   ```sh
   DP3 my.parset msin=foo.MS msout=bar.MS steps=[average] average.timestep=4
   ```

   With no arguments DP3 falls back to `DP3.parset`, then `DPPP.parset`,
   then `NDPPP.parset` (`base/DP3.cc:461-473`). `-h`/`--help`/`--usage`
   show usage; `-v`/`--version` prints the parsed git-describe version
   from `base/Version.h`.

2. **Pip wheel CLI `DP3.py`** — installed from `pythondp3` so that the
   pip wheel contains a runnable `DP3.py` that simply calls
   `dp3.execute_from_command_line(sys.argv)`
   (`pythondp3/__init__.py:entrypoint`). Replaces the legacy
   `__DP3_from_pip__` executable removed in 6.2.2.

3. **Embedded Python**:

   ```python
   import dp3
   dp3.execute("my.parset", ["msin=foo.MS", "msout=bar.MS"])  # parset+overrides
   first_step = dp3.make_main_steps(parset)                    # build chain
   dp3.make_step("predict", parset, "predict.", dp3.MsType.regular)
   ```

   See §10 for details.

### 2.2 `dp3::base::Execute()` — the canonical run loop

`base/DP3.cc:291-422` is the heart of the program. Pseudocode:

```text
1. Build a ParameterSet (parset) from the file, then layer
   command-line key=value pairs on top.
2. Read global keys: verbosity, time_logging, memory_logging,
   showprogress, showtimings, checkparset, showcounts, numthreads.
3. Initialize aocommon::Logger and aocommon::ThreadPool with
   numthreads (defaults to ProcessorCount()). Mark the threading
   subsystem as initialized so subsequently-constructed Steps know
   not to re-initialize it.
4. Build the step chain (MakeMainSteps).
5. Call firstStep->setInfo(DPInfo()) — propagates a metadata object
   through every step's updateInfo() so each step knows its input
   shape and announces its output shape.
6. Walk the chain printing show() summaries to the logger.
7. If checkparset >= 0, list parset keys that were never queried —
   typo defence. checkparset == 1 turns this into an error.
8. SkyModelCache::GetInstance().Clear() — sky models are loaded once
   in updateInfo() and immediately released to free RAM.
9. Loop: while (firstStep->process(make_unique<DPBuffer>())) {…}
   — empty buffers are pushed in; the input step fills them from
   the MS, downstream steps mutate or consume them. Returns false
   when the input is exhausted.
10. firstStep->finish() — flushes accumulating steps (e.g. AOFlagger,
    averagers).
11. Walk the chain printing showCounts() and showTimings() lines.
```

Threading defaults: `numthreads = environment OMP_NUM_THREADS` if set,
otherwise the system core count. The same ThreadPool is shared by
solvers, the AOFlagger, MADFlagger, demixer, and predict steps.

### 2.3 `MakeMainSteps()` — wiring the pipeline

`base/DP3.cc:487-546`.

1. The implicit input step is created from `msin` / `msin.name` (or
   `stream.socket` for streaming via `SVPInput`):
   `InputStep::CreateReader(parset)` returns either an `MsReader`,
   `MultiMsReader` (multiple MSs concatenated in frequency), or
   `MSBDAReader` if the MS contains BDA data
   (`steps/InputStep.h`).
2. Steps are constructed from `parset.steps` (or the provided substep
   key) by `MakeStepsFromParset()` (`base/DP3.cc:548-603`). Each step
   name's alphabetic prefix is its default `type` (so `average1`,
   `out3`, `count` Just Work).
3. `MakeSingleStep()` is the central registry — a giant `if/else`
   that maps `type` strings (case-insensitive) to step constructors
   (`base/DP3.cc:162-257`). A handful of types accept aliases
   (`aoflagger`/`aoflag`, `averager`/`average`/`squash`,
   `phaseshifter`/`phaseshift`, `demixer`/`demix`,
   `applycal`/`correct`, `gaincal`/`calibrate`, `counter`/`count`,
   `madflagger`/`madflag`, `preflagger`/`preflag`,
   `uvwflagger`/`uvwflag`, `antennaflagger`/`antflag`,
   `bdaaverage`/`bdaaverager`, `stationadder`/`stationadd`,
   `python`/`pythondppp`, `split`/`explode`).
4. After the user steps, the framework appends the implicit
   output step (an `MSWriter`, `MSUpdater`, or `MSBDAWriter`) when
   the chain provides any field that needs writing (see Fields, §3.4).
   `MakeOutputStep()` (`base/DP3.cc:102-160`) decides whether to
   *update* the input MS (`msout` empty, `.`, or equal to `msin`) or
   create a new MS. BDA outputs cannot be in-place updates.
5. A `NullStep` is appended so every step can call
   `getNextStep()->process(...)` unconditionally (`Split` is excluded
   because it owns its own next-step list).
6. `input_step->setFieldsToRead(GetChainRequiredFields(...))` — only
   columns actually consumed downstream are read from the MS
   (Fields-driven I/O minimization, §3.4).

`MakeStepsFromParset()` is reused recursively by container steps
(`Split`, `Predict`/`OnePredict` with `applycal.*` substeps, `Demixer`'s
internal averagers, `BdaDdeCal` substeps, `DDECal` `modelnextsteps`).

### 2.4 Step-chain compatibility

Every step declares an *MS type* it produces (`Step::outputs()`,
`MsType::kRegular` or `MsType::kBda`) and an *MS type* it accepts
(`Step::accepts(MsType)`). When the chain is built each new step is
checked against the previous step's `outputs()` and the build aborts if
incompatible (`base/DP3.cc:583-587`). Most steps are `kRegular`-only;
`BdaAverager`, `BdaExpander`, `MSBDAReader`, `MSBDAWriter`, the
`Predict` BDA path, and `BdaDdeCal` form the BDA chain.

---

## 3. Pipeline framework

### 3.1 `Step` — abstract base class (`steps/Step.h`)

Every step in the pipeline implements:

| Method | Default | Purpose |
|--------|---------|---------|
| `process(unique_ptr<DPBuffer>)` | throws | One time slot of regular data; must call `getNextStep()->process(...)` and return that boolean. |
| `process(unique_ptr<BdaBuffer>)` | throws | BDA variant. |
| `finish()` | pure virtual | Flush; called once after `process` returns false. |
| `updateInfo(const DPInfo&)` | copy through | Receive the upstream `DPInfo`, mutate it (e.g. averaging changes nchan/ntime) and forward to next step. Called by `setInfo()`. |
| `getRequiredFields()` | pure | Which of `kData`/`kFlags`/`kWeights`/`kUvw` the step needs. |
| `getProvidedFields()` | pure | Which of those it modifies. |
| `show(ostream&)` / `showCounts` / `showTimings` | pure / no-op | Diagnostics. |
| `addToMS(string)` | forward | Optional hook to write extra subtables to the output MS (used by `AOFlaggerStep` for the QUALITY tables). |
| `outputs()` / `accepts(MsType)` | regular | BDA negotiation. |

The class also keeps a singly-linked next pointer (`shared_ptr`) and a
back pointer (raw, for cycle-free linkage). A static
`SetThreadingIsInitialized()` flag prevents accidental double-init of
the thread pool when steps are constructed from Python.

`ModelDataStep` is a thin abstract subclass requiring a
`GetFirstDirection()` method and announcing `kDataField` as provided —
implemented by all predict-flavoured steps so DDECal and friends can
introspect direction metadata (`steps/Step.h:172-178`).

### 3.2 `DPBuffer` — the visibility payload (`base/DPBuffer.h`)

Holds a single time slot of regular data:

| Field | Type | Shape | Notes |
|-------|------|-------|-------|
| `time_` | `double` | scalar | Centroid time (MJD seconds). |
| `exposure_` | `double` | scalar | Integration time (s). |
| `row_numbers_` | `casacore::Vector<rownr_t>` | (nbl,) | Reference-counted; empty when slots were inserted. |
| `data_` | `xt::xtensor<complex<float>,3>` | (nbl, nchan, ncorr) | Main visibility buffer. |
| `extra_data_` | `map<string, DataType>` | each (nbl, nchan, ncorr) | Named auxiliary buffers used by predict→ddecal/applycal pipelines and the `combine`/`columnreader`/`reusemodel` features. Names are arbitrary except the empty string (which means the main buffer). |
| `flags_` | `xt::xtensor<bool,3>` | (nbl, nchan, ncorr) | All correlations carry the same value within DP3 (`AST-1373`). |
| `weights_` | `xt::xtensor<float,3>` | (nbl, nchan, ncorr) | Sum-style weights. |
| `uvw_` | `xt::xtensor<double,2>` | (nbl, 3) | Metres. |
| `solution_` | `vector<vector<complex<double>>>` | per-channel × (nant·npol) | Used by the `storebuffer` path (DDECal → ApplyCal in same run). |

Public mutators are `AddData`/`RemoveData`/`HasData`/`GetData(name)`
/`CopyData`/`MoveData`/`TakeData`. Move semantics are pervasive — DP3
moves `unique_ptr<DPBuffer>` between steps to avoid deep copies. The
default `process()` contract is *one buffer in, one buffer out, in-place
mutation allowed*; accumulating steps may swap buffers into private
storage and refill the unique_ptr in `finish()`.

A multi-line comment in `DPBuffer.h:71-91` records the historical
strategy evolution: shallow casacore Arrays → deep copies → unique_ptr
moves. The latter is the current default since 2023.

### 3.3 `BdaBuffer` (`base/BdaBuffer.h`)

Heterogeneous per-row container for baseline-dependently averaged
visibilities. Each `Row` has its own `(time, interval, exposure,
n_channels, n_correlations, offset, uvw)`. Accumulator views are flat
1-D vectors keyed by per-row offsets. Used by `BdaAverager`,
`BdaExpander`, `MSBDAReader`, `MSBDAWriter`, `BdaDdeCal`,
`BdaGroupPredict`, and the BDA-aware `predict` path.

### 3.4 `Fields` — required vs. provided (`common/Fields.h`)

`Fields` is a 4-bit `bitset`-backed value class with bits for `kData`,
`kFlags`, `kWeights`, `kUvw`. Every step exposes
`getRequiredFields()` and `getProvidedFields()`, and `Fields` provides
`UpdateRequirements(required, provided)` semantics:

```
to_reset = provided & ~required;
value |= required;
value &= ~to_reset;
```

`base/DP3.cc:259-289` walks the chain *backwards* to compute the union
of required fields not already provided upstream, so the input step can
selectively read only what is needed (for instance: a chain
`steps=[predict,subtract,msout]` doesn't need the input MS's WEIGHT
column unless something later uses it). `SetChainProvidedFields()` walks
*forward*: when an `OutputStep` is encountered, it adopts the currently
provided fields and resets the running set, so multiple intermediate
output steps each write only what changed since the previous output.

### 3.5 `DPInfo` — pipeline metadata (`base/DPInfo.h`)

Per-pipeline metadata carried through `updateInfo()`. Tracks:

- Antenna table: names, diameters, ITRF positions, `antenna1_`,
  `antenna2_` per baseline; `antennas_used_` and `antenna_map_`
  computed by `setAntUsed()` to track which antennas survive baseline
  selection.
- Channel info: per-baseline `channel_frequencies_`, `channel_widths_`,
  `resolutions_`, `effective_bandwidth_` (BDA → outer vector size = nbl;
  regular → outer size = 1). `total_bandwidth_`, `reference_frequency_`,
  `spectral_window_` index.
- Time info: `first_time_`, `last_time_`, `time_interval_`, `n_times_`,
  per-baseline `time_averaging_factors_` (BDA), and a flag for
  "interval factor is integer" (for BDA writers).
- Pointing: `original_phase_center_`, `phase_center_`, `delay_center_`,
  `tile_beam_direction_`, named extra `extra_directions_` (used by
  `WGridderPredict` regions, etc.).
- Beam state: `beam_correction_mode_` (an integer mirror of
  `everybeam::BeamMode` to keep DP3's public API EveryBeam-free) and
  `beam_correction_direction_`.
- `meta_changed_` is a sentinel set when a step mutates metadata in a
  way that prevents an in-place MS update.
- Polarization set (`aocommon::PolarizationEnum`).
- MS column names: `data_column_name_`, `flag_column_name_`,
  `weight_column_name_`, `antenna_set_`.

`SelectChannels()`, `SelectBaselines()`, `update(chanAvg, timeAvg)`,
`RemoveUnusedAntennas()`, and `setMetaChanged()` are the canonical
mutators. `getInfoIn()`/`getInfoOut()` return read-only references
(except in Python where the wrapper copies them).

---

## 4. ParSet — the configuration language

### 4.1 Format

ParSet files are `key = value` pairs, one per line, comments start with
`#`. Values can be scalars, vectors `[a, b, c]` (any nesting depth), or
records `{k1: v1, k2: v2}`. Every step has a `<stepname>.` prefix so
multiple instances of the same step are allowed (`flag1.type=madflagger
flag2.type=madflagger`). Command-line arguments override (or add to)
file values:

```sh
DP3 my.parset numthreads=8 ddecal.maxiter=200
```

The implementation lives in `common/ParameterSet*` and parses both via
`adoptFile(path)` and `adoptArguments(vector<string>)`
(`base/DP3.cc:296-301`).

### 4.2 Implicit steps

Two pseudo-steps are always implied unless explicitly suppressed:

- `msin` (alias `msin.name`) — created from `InputStep::CreateReader()`.
  Reads a single MS, a glob (`L*_SAP000_SB*`), or a vector of MSs
  (concatenated in frequency, must be sorted unless
  `msin.orderms=true`). When a name does not exist, flagged-zero data
  is inserted iff `msin.missingdata=true` and the chain provides a way
  to derive frequency from the surviving MSs.
- `msout` (`msout.name`) — `MSWriter` (new MS), `MSUpdater` (in-place),
  or `MSBDAWriter` for BDA chains. An empty `msout`, `.`, or a path
  equal to `msin` triggers update mode. A `null` step as the last
  pipeline step suppresses the implicit writer entirely.

Intermediate writes are achieved with explicit `out`/`output`/`msout`
typed steps:

```text
steps=[aoflag, out1, average, out2, applycal]
out1.type=out
out1.name=L123-flagged.MS
out2.type=out
out2.name=L123-averaged.MS
```

### 4.3 Global keys (apply outside any step prefix)

| Key | Default | Effect |
|-----|---------|--------|
| `steps` | required | Vector of step names. |
| `numthreads` | `${OMP_NUM_THREADS}` or core count | Pool size for OpenMP/`aocommon::ThreadPool`. |
| `showprogress` | `true` | Print a `ProgressMeter`. |
| `showtimings` | `true` | Print per-step timing table. |
| `showcounts` | `true` | Print per-step flag counts. |
| `verbosity` | `normal` | `quiet`, `normal`, or `verbose`. |
| `time_logging` | `false` | Prefix every log line with timestamp. |
| `memory_logging` | `false` | Prefix every log line with current Dp3 memory in GB. |
| `checkparset` | `0` | `-1` ignore, `0` warn unused keys, `1` error. (`true`/`false` accepted for back-compat.) |

The "Description of all parameters" YAML schema documents these
explicitly (`docs/schemas/Description of all parameters.yml`).

### 4.4 `msin` keys

(Schema: `docs/schemas/Input.yml`.)

- `msin` / `msin.name` — name, glob, or vector. Concatenation is in
  frequency only.
- `msin.sort` — sort by TIME. Default off.
- `msin.orderms` — sort multiple MSs by frequency. Default on.
- `msin.missingdata` — insert flagged data for missing MSs.
- `msin.baseline` — CASA baseline-selection string (only the
  CASA-syntax variant works in `msin`; full `[[ant,ant], ...]` form is
  available in `Filter` and `PreFlagger`).
- `msin.band` — pick a single SPW (default `-1` = no selection).
- `msin.startchan`, `msin.nchan` — channel range. Both can be
  expressions in `nchan`, e.g. `nchan/32`.
- `msin.starttime` / `msin.endtime` — `casacore::MVTime` strings
  (`19Feb2010/14:01:23.817`); dummy time slots inserted where needed.
  6.4 changed start handling: DP3 uses an existing time slot from the
  MS as the reference instead of the first slot + N×interval.
- `msin.starttimeslot` — integer offset (negative inserts pre-MS slots).
- `msin.ntimes` — limit; `0` = until the end.
- `msin.useflag` — apply existing FLAG/FLAG_ROW (default true).
- `msin.datacolumn` — default `DATA`.
- `msin.extradatacolumns` — extra columns also read into the
  `DPBuffer.extra_data_` map (consumed by `ddecal.modeldatacolumns`,
  `gaincal.modelcolumn`, etc.).
- `msin.weightcolumn` — `WEIGHT_SPECTRUM` if present, else `WEIGHT`.
- `msin.flagcolumn` — `FLAG`.
- `msin.autoweight` / `msin.forceautoweight` — recompute weights from
  auto-correlations: `WEIGHT[a1·p1, a2·p2] = N / (autocorr[a1·p1] *
  autocorr[a2·p2])`, with `N = EXPOSURE * CHAN_WIDTH * WGHT`. Only
  intended for raw LOFAR. `forceautoweight=true` is required if the MS
  has already been DP3-processed (defence against accidental double
  weighting).

### 4.5 `msout` keys

(Schema: `docs/schemas/Output.yml`.)

- `msout` / `msout.name` — empty, `.`, or `==msin` ⇒ in-place update.
- `msout.overwrite` — when creating a new MS.
- `msout.datacolumn`, `msout.weightcolumn`, `msout.flagcolumn` — the
  columns to write. New MSs only accept the canonical names; updates
  can write to any (auto-created if absent).
- `msout.chunkduration` — split output into time-chunked MSs:
  `myobs-000.ms`, `myobs-001.ms`, ….
- `msout.tilesize`, `msout.tilenchan` — casacore tile sizes (default
  in 6.5 is `tilenchan=64`).
- `msout.clusterdesc`, `msout.vdsdir` — VDS file generation (LOFAR
  cluster descriptors).
- Compression / metadata-compression flags (defaults from
  `METADATA_COMPRESSION_DEFAULT`):
  - `msout.scalarflags` — store one flag for four correlations.
  - `msout.antennacompression` — `AntennaStMan` storage manager,
    losslessly compresses ANTENNA1/ANTENNA2.
  - `msout.uvwcompression` — `UvwStMan` storage manager, derives one
    UVW per antenna from the rest. Cannot combine with
    `stationadder.average=true`.
- Storage manager block (`msout.storagemanager.*`):
  - `name`: empty (uncompressed), `stokes_i` (`StokesIStMan`), `dysco`,
    `sisco`. Defaults below describe Dysco/Sisco knobs.
  - Dysco: `databitrate` (default 10), `weightbitrate` (12),
    `distribution` (`TruncatedGaussian` / `Uniform` / `Gaussian` /
    `StudentsT`), `disttruncation` (2.5), `normalization` (`AF` /
    `RF` / `Row`).
  - Sisco (new in 6.5): `predict_level` (`-1`–`2`, with `2` =
    quadratic prediction), `deflate_level` (`1`–`12`),
    `sisco_mode` (`full` / `diagonal` / `stokes_i`).

### 4.6 Streaming input

`stream.socket=/path/to/socket` replaces `msin`; the `SVPInput` step
reads framed packets and produces DPBuffers. Used for
cobalt-streaming LOFAR and ALMA prototypes (added in 6.5). See
`docs/schemas/SVPInput.yml`.

---

## 5. Step library — exhaustive reference

The list below covers every step type registered in
`MakeSingleStep()` (`base/DP3.cc:162-257`), grouped logically. Each
entry quotes the canonical schema name, common aliases, the
header/source pair, and every documented parameter (defaults from
`docs/schemas/`).

### 5.1 Inputs

#### `msin` — `InputStep` / `MsReader` / `MultiMsReader` / `MSBDAReader`

Implicit. See §4.4 for the parameter set. Provides
`getProvidedFields()` = whatever was selected via `setFieldsToRead`.
`HasBda(MeasurementSet)` decides which reader subclass is constructed.

#### `svpinput` — `SVPInput`

Replaces `msin` with a Unix-socket producer. ParSet:

| Key | Default | Doc |
|-----|---------|-----|
| `socket` | `/tmp/svpsock0` | Path of the Unix socket to read packets from. |
| `steps` | `[]` | Sub-pipeline applied to the streamed buffers. |

#### `columnreader` — `MsColumnReader`

Read a column from the MS *as if it were the DATA column*, useful for
testing model-vs-data subtraction:

| Key | Default | Doc |
|-----|---------|-----|
| `column` | `MODEL_DATA` | Name of the column to read. |

### 5.2 Outputs

#### `msout` / `out` / `output` — `MSWriter` / `MSUpdater` / `MSBDAWriter`

Implicit at the end of the chain unless suppressed by a `null` step or
by no provided fields. Parameters are the `msout.*` keys (§4.5). The
intermediate `out` form lets a chain emit several MSs:

```text
steps=[flag, out1, average, out2, applycal]
out1.type=out
out1.name=L123-flagged.MS
out2.type=out
out2.name=L123-averaged.MS
out2.datacolumn=DATA
```

#### `wscleanwriter` — `WSCleanWriter`

Reorder a MS into [WSClean's reordered format][wsclean-reorder] for
downstream imaging without an extra reorder pass. Schema:

| Key | Default | Doc |
|-----|---------|-----|
| `wscleanwriter.name` | from `msout`/`msin` | Output stem. |
| `wscleanwriter.polarization` | `I` | `I/Q/U/V`, `instr`, `diag_instr`, `XX/XY/YX/YY`, `RR/RL/LR/LL`. |
| `wscleanwriter.temporary_directory` | `""` | Working directory for the reorder. |
| `wscleanwriter.chanperfile` | — | Split the reordered files by `nchan`. |

[wsclean-reorder]: https://wsclean.readthedocs.io

#### `null` — `NullStep`

Discards data. Used internally as the final sink, but can also be
explicit (last step) to stop DP3 from writing any output.

### 5.3 Flagging

#### `aoflagger` / `aoflag` — `AOFlaggerStep`

AOFlagger 3.x driver. Memory-bounded sliding-window flagging.
Important parameters (`docs/schemas/AOFlagger.yml`):

- `strategy` — Lua strategy file. Empty ⇒ AOFlagger's HBA default.
  `LBAdefault` recommended for LBA.
- `memoryperc` (int %) / `memorymax` (GB) — memory budget. Limiting
  too aggressively degrades flagging accuracy; aim for ≥ 10 GB.
- `timewindow` — joint time window, default deduced from memory.
- `overlapperc` / `overlapmax` — left/right window padding to soften
  boundary effects.
- `autocorr` — flag autocorrelations (default true).
- `pulsar` — pulsar-friendly strategy.
- `pedantic` — stricter.
- `keepstatistics` — write QUALITY subtables (default true). Inspect
  with `aoqplot`.
- `count.save` / `count.path` — write `.flagfreq`/`.flagstat` tables.

The step writes its own QUALITY and FLAGSTAT subtables via
`addToMS()`.

#### `madflagger` / `madflag` — `MadFlagger`

Median-Absolute-Deviation flagger
(`docs/schemas/MADFlagger.yml`). Parameters:

- `threshold` — TaQL-like expression that may use the `bl` baseline
  length (e.g.
  `iif(bl<100,0.5, iif(bl<500,0.75, iif(bl<1000,0.9,1)))`).
- `timewindow` / `freqwindow` — odd, baseline-length-dependent
  expressions allowed.
- `correlations` — order matters (e.g. `[3,0,1,2]` for YY/XX/XY/YX).
- `applyautocorr` — flag autocorrelations only and propagate to crosses.
- `blmin` / `blmax` — baseline-length filter.
- `count.save` / `count.path` — `.flagfreq`/`.flagstat` output.

#### `preflagger` / `preflag` — `PreFlagger`

Boolean expression over a vast set of selectors
(`docs/schemas/PreFlagger.yml`):

- `mode` — `set`, `setcomplement`/`setother`, `clear`,
  `clearcomplement`/`clearother`.
- `expr` — boolean expression over named keyword sets, e.g.
  `c1 and (c2 or c3)`.
- Time selectors: `timeofday`, `abstime` (date/time), `reltime`
  (since obs start), `timeslot`, `lst`, `azimuth`, `elevation`.
- Geometry: `baseline`, `corrtype` (`auto`/`cross`), `blmin`/`blmax`,
  `uvmmin`/`uvmmax`.
- Frequency / channel: `freqrange` (`[1.2 .. 1.4 MHz, 1.8 MHz +- 50
  KHz]`), `chan` (`[0 .. nchan/32-1, 31*nchan/32 .. nchan-1]`).
- Visibility tests: `amplmin/max`, `phasemin/max`, `realmin/max`,
  `imagmin/max` — vector form selects per-correlation, e.g.
  `amplmin=[100,,,100]`.
- `count.save` / `count.path`.

#### `uvwflagger` / `uvwflag` — `UVWFlagger`

UVW-based flagging (`docs/schemas/UVWFlagger.yml`). Either in metres
(`uvmmin`, `uvmmax`, `umrange`, `vmrange`, `wmrange`, `uvmrange`) or
wavelengths (`uvlambdamin`, `uvlambdamax`, `ulambdarange`,
`vlambdarange`, `wlambdarange`, `uvlambdarange`). UVWs can be
recomputed in another `phasecenter` direction (RA/Dec or moving body
like `SUN`/`JUPITER`). `beammode` allowed for `array_factor`/`element`/
`default` reference UVWs.

#### `madflagger`-like specialty: `clipper` — `Clipper`

Predict-and-clip RFI/source removal. Builds an internal sky-model
prediction and flags everything above `amplmax`. Schema
(`docs/schemas/Clipper.yml`):

- `amplmax` — default `50 Jy` for LBA, `5 Jy` for HBA.
- `timestep`, `freqstep` — re-prediction stride; reuses the previous
  prediction for `step-1` cycles.
- `flagallcorrelations` — flag all four correlations on a hit
  (default true).
- `baseline` — feeds an internal `Filter` substep (new in 6.5).
- All `predict.*` keys are accepted because the clipper instantiates
  a `Predict` step internally.

#### `antennaflagger` / `antflag` — `AntennaFlagger`

Outlier detection at antenna and station level (the
`antennaflagger/Flagger.cc` library). Iterative (`max_iterations`),
sigma-driven. Schema:

| Key | Default | Doc |
|-----|---------|-----|
| `antenna_flagging_sigma` | `3` | Per-antenna threshold. |
| `antenna_flagging_max_iterations` | `5` | Outer loop. |
| `station_flagging_sigma` | `2.5` | Per-station threshold. |
| `station_flagging_max_iterations` | `5` | Outer loop. |

Originally written for AARTFAAC; for manual single-antenna flagging
prefer `PreFlagger` with a baseline selection.

#### `counter` / `count` — `Counter`

Tally flags. Insert anywhere to print cumulative flag percentages.

| Key | Default | Doc |
|-----|---------|-----|
| `showfullyflagged` | `false` | Print all baselines that are 100 % flagged using `ant1&ant2` notation. |
| `save` | `false` | Write `.flagfreq` / `.flagstat` tables. |
| `path` | `""` | Output directory. |
| `warnperc` | `0` | Print a `WARN`-prefixed line for any baseline / channel with > X %. |
| `savetojson` | `false` | Write a JSON file with per-station ratios. |
| `jsonfilename` | `FlagPercentagePerStation.JSON` | Path. |

#### `flagtransfer` — `FlagTransfer`

Copy flags from a low-resolution averaged MS onto a higher-resolution
MS. Deprecated in 6.5 in favour of `transfer` with `flags=true`. Filter
sub-step is allowed if station counts differ.

### 5.4 Selection & shaping

#### `filter` — `Filter`

Channel and/or baseline subset (`docs/schemas/Filter.yml`):

| Key | Default | Doc |
|-----|---------|-----|
| `startchan`, `nchan` | `0`, `0` (=all) | Channel slice; expressions in `nchan` accepted. |
| `baseline` | `""` | CASA baseline-selection string. |
| `blrange` | `""` | Baseline-length ranges (m). |
| `corrtype` | `""` | `auto`/`cross`. |
| `remove` | `false` | Drop unused antennas from ANTENNA / FEED / POINTING / SYSCAL / LOFAR_ANTENNA_FIELD / LOFAR_ELEMENT_FAILURE / QUALITY_BASELINE_STATISTIC subtables, renumber `ANTENNA1/2`. |

#### `phaseshifter` / `phaseshift` — `PhaseShift`

Shift to a new phase centre. Empty `phasecenter` ⇒ original phase
centre (useful as a "shift back" step). Vectors: `[12h31m34.5,
52d14m07.34]` or `[187.5deg, 52.45deg]`. After a chain of shifts the
final shift back to the original phase centre means no new MS needs to
be written.

#### `stationadder` / `stationadd` — `StationAdder`

Build new "super-stations" by summing/averaging baselines. Schema
(`docs/schemas/StationAdder.yml`):

- `stations` — record like `{ST6: 'CS00[2-7]*'}` or
  `{ST6: ['CS00[2-7]*','!CS005*']}` or
  `{ST001:[CS001,CS002,CS003], ST002:[CS004,CS005,CS006]}`.
  Patterns accept `* ? [] {}` and `!`/`^` for negation.
- `minpoints` — flag if fewer unflagged input points than this.
- `useweights` — weight by input weights.
- `average` — weighted average instead of sum (incompatible with
  `msout.uvwcompression`).
- `autocorr` — produce new auto-correlations.
- `sumauto` — autocorr from existing auto-correlations (true) vs from
  cross-correlations (false).

#### `averager` / `average` / `squash` — `Averager`

Time/frequency averaging (`docs/schemas/Averager.yml`):

- `timestep`, `freqstep` — integer factors. The freq factor must
  divide `nchan`.
- `minpoints` / `minperc` — flag the averaged sample if too few
  unflagged points contributed.
- `timeresolution` (s) / `freqresolution` (Hz, accepts `MHz`/`kHz`
  suffix) — alternative spec; overrides `timestep`/`freqstep` if set.
  UVWs are *averaged*, not recomputed. Time slots missing at the end
  are padded with dummy slots so the output remains regular.

#### `bdaaverager` — `BdaAverager`

Baseline-dependent averager (`docs/schemas/BDAAverager.yml`). Outputs a
`BdaBuffer`. Parameters:

- `timebase` (m) — averaging factor = `timebase / blength`, rounded
  down. `0` ⇒ no time averaging.
- `frequencybase` (m) — channels per baseline = `blength /
  frequencybase * orig nchan` rounded up. `0` ⇒ no freq averaging.
- `maxinterval` (s) — clip the time factor for short baselines.
  Rounded down to a multiple of the integration time.
- `minchannels` — minimum channels per averaged baseline.

When BDA feeds into `BdaDdeCal`, the maximum number of averaged
intervals must be ≤ `solint` and the maximum averaged channels ≤
`nchan` of the channel block.

#### `bdaexpander` — `BdaExpander`

Convert a BDA `BdaBuffer` back to a regular `DPBuffer`.

#### `transfer` — `Transfer`

New in 6.5. Transfer **data** *and/or* **flags** from a low-resolution
MS to a high-resolution one (subsumes `flagtransfer`). Filter substep
keys (`baseline`, `blrange`, …) are accepted to align mismatched
baseline counts. Schema:

| Key | Default | Doc |
|-----|---------|-----|
| `source_ms` | required | Lower-resolution MS. |
| `data` | `false` | Transfer data. |
| `flags` | `false` | Transfer flags. |
| `datacolumn` | `DATA` | Column in `source_ms`. |
| `outputbuffername` | "" | Optional named DPBuffer slot for use in a subsequent `combine` step. |

#### `combine` — `Combine`

Add or subtract a named extra DPBuffer from the main data:

| Key | Doc |
|-----|-----|
| `operation` | `add` / `subtract`. |
| `buffername` | Source buffer name. |

Pairs naturally with `transfer.outputbuffername`, `predict.outputmodelname`, or `ddecal.keepmodel/reusemodel`.

#### `upsample` — `Upsample`

Stretch each averaged time slot into N sub-slots (`timestep`),
optionally recomputing UVW (`updateuvw=true`). Useful to repair
non-uniform AARTFAAC time grids before averaging, e.g.

```
[0,2,4,7,9,12]  →  inserted dummies → [0,2,4,6,7,9,11,12]  →  upsample(2) →
[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
```

#### `interpolate` — `Interpolate`

Replace flagged samples with a Gaussian-weighted neighbourhood mean
before averaging. `windowsize` (default 15, must be odd).

> Reference: Offringa, Mertens & Koopmans 2018,
> [arxiv 1901.04752](https://arxiv.org/abs/1901.04752).

#### `nullstokes` — `NullStokes`

Zero out one or more Stokes parameters
(`modify_i`/`modify_q`/`modify_u`/`modify_v`).

### 5.5 Calibration steps

#### `gaincal` / `calibrate` — `GainCal`

Direction-independent gain calibration using StefCal
(`docs/schemas/GainCal.yml`). Settings:

- `caltype` — `scalar`, `scalarphase`, `scalaramplitude`, `diagonal`,
  `diagonalphase`, `diagonalamplitude`, `fulljones`, `tec`,
  `tecandphase`.
- `parmdb` — output ParmDB name (or `*.h5` for H5Parm). Cannot be
  re-used inside the same DP3 run for `applycal`; use
  `applysolution=true` instead.
- `solint` (timeslots), `nchan` (per-block; default 1, 0 = whole band;
  for `tec` modes default is 1 = phase-per-channel TEC fitting).
- `propagatesolutions` — initialize with the previous solution
  (default true).
- `maxiter` (50), `tolerance` (1e-5), `detectstalling` (true),
  `minblperant` (4) — solver settings.
- `applysolution` — apply during the same run.
- Sky-model inputs: `sourcedb`, `sources`, `usebeammodel`,
  `usemodelcolumn`, `modelcolumn`, `applybeamtomodelcolumn`, `reusemodel`.
- Selection: `baseline`, `blrange`, `uvlambdamin`.
- Beam: `onebeamperpatch`, `usechannelfreq`, `beammode`,
  `beamproximitylimit`, `coefficients_path` (e.g. MWA).
- ApplyCal sub-step: `applycal.*`.
- `timeslotsperparmupdate` (500) — buffering for the parmdb writer.
- `debuglevel` ≥ 1 dumps every iterand to `debug.h5`.

#### `applycal` / `correct` — `ApplyCal`

Apply solutions from H5Parm or ParmDB to the data. Multi-correction
chains are first-class (`steps=[applycal] applycal.steps=[amp,phase]
applycal.amp.correction=amplitude000 applycal.phase.correction=phase000`).
Schema:

- `parmdb` — H5Parm (`.h5`) or ParmDB. If empty, applies the buffered
  solution stored by `ddecal.storebuffer`.
- `solset` — H5Parm solset name (mandatory if multiple).
- `correction` — `gain`, `tec`, `clock`, `(common)rotationangle` /
  `rotation`, `(common)scalarphase`, `(common)scalaramplitude`,
  `rotationmeasure`, `fulljones`. With H5Parm the correction is the
  soltab name and the type is auto-detected (except `fulljones`).
- `soltab` — used only for `correction=fulljones`, providing two
  soltab names (amplitude and phase).
- `direction` — pick a direction inside the soltab.
- `updateweights` — propagate beam-weight scaling.
- `interpolation` — `nearest` (default) or `linear`.
- `invert` — apply (true) vs corrupt (false).
- `timeslotsperparmupdate` (200) — buffering.
- `steps`, `<substep>.*` — chained sub-steps inheriting unspecified
  values from the parent.
- `missingantennabehavior` — `error`, `flag`, `unit`.
- `usemodeldata` — apply correction to a named extra-data buffer
  (using the metadata-stored direction) instead of the main data.

`OneApplyCal.cc` is the workhorse for a single correction; `ApplyCal`
is a thin wrapper that builds 1..N `OneApplyCal` substeps.

#### `applybeam` — `ApplyBeam`

EveryBeam application (or its inverse). Schema:

- `direction` — RA/Dec target direction; empty ⇒ phase centre.
- `beamproximitylimit` (60 arcsec) — cluster proximate sources for
  one beam computation.
- `onebeamperpatch` — deprecated; equivalent to a 0-arcsec proximity.
- `usechannelfreq` (true) — per-channel beam (set false for raw LOFAR).
- `updateweights` — rescale weights consistently.
- `invert` (true) — apply (default) vs corrupt.
- `beammode` — `array_factor`, `element`, `full` (= `default`).
- `elementmodel` — `default` (deduce from MS), `hamaker`, `lobes`,
  `oskarsphericalwave`, `oskardipole`, or any model EveryBeam supports.
- `skipstations` — list of stations to skip (advanced/expert).
- `usemodeldata` — apply to a named extra buffer (using its
  per-direction metadata).
- `coefficients_path` — for beam models that need an external file
  (e.g. MWA).

#### `setbeam` — `SetBeam`

Manually overwrite the beam keywords for a column (advanced). Useful
when an external prediction has scaled the visibilities but not the
keywords. `direction`, `beammode` (`none`/`default`/`array_factor`/`element`).

### 5.6 Prediction

`OnePredict` (`steps/OnePredict.h`) is the actual workhorse — it
implements the radio-interferometer measurement equation for point and
Gaussian sources, optionally applying the beam. `Predict` is a wrapper
that adds optional pre-/post-processing sub-steps and BDA awareness;
when `USE_FAST_PREDICT=ON` it can substitute `FastPredict`
(experimental, drop-in).

Common prediction parameters (`docs/schemas/Predict.yml`):

- `usefastpredict` — runtime switch, only available with
  `USE_FAST_PREDICT`.
- `sourcedb` — path to a sky-model file. **Modern DP3 only supports
  the textual `.skymodel`/`.txt` format**; `model/SkyModelCache.h`
  raises an error when given a binary `makesourcedb` output and
  recommends `showsourcedb` to convert. The textual format includes
  flags such as `OrientationIsAbsolute=true` for the corrected
  Gaussian projection (DP3 ≥ 5.3).
- `sources` — patches to use.
- `usebeammodel` — bring in EveryBeam.
- `elementmodel` — same as `applybeam`.
- `operation` — `replace` (default), `subtract`, `add`.
- `outputmodelname` — write to a named extra DPBuffer instead of the
  main one (consumed downstream by `combine` / `ddecal.reusemodel`).
- `applycal.*` — corrupt the prediction with applied solutions
  (`invert` defaults to `false` here so the operation is a corruption
  instead of a correction).
- Beam knobs identical to `applybeam`: `beamproximitylimit`,
  `onebeamperpatch`, `usechannelfreq`, `beammode`,
  `coefficients_path`. `beam_interval` (s) controls how often the
  beam is recomputed (`0` = every time step; `120` ≈ good
  speed/accuracy trade-off).
- `correcttimesmearing` / `correctfreqsmearing` — multiply by sinc
  factors approximating the integration smearing. In 6.5 the
  upsampling time-smearing approximation was replaced by a sinc
  approximation (faster, equally accurate).
- `parallelbaselines` — parallelize over baselines instead of sources.

#### `predict` — `Predict`

Top-level predict step. Sets up sub-steps for beam application,
applycal sub-step, BDA averager bridge, and (optionally) the fast
predict path. Implements both `process(DPBuffer)` and
`process(BdaBuffer)`.

#### `h5parmpredict` — `H5ParmPredict`

Predict (and subtract/add) multi-direction visibilities corrupted by an
H5Parm instrument model — the post-DDECal companion. `directions=[]`
defaults to "all directions in the soltab"; the H5Parm convention is
that directions are tagged with patch names like `[patch1, patch2]`.

#### `wgridderpredict` — `WGridderPredict`

Predict from FITS images via the DUCC wgridder. `images` is a list of
images, one per polynomial frequency term (matching WSClean's
component-list convention). `regions` is a DS9 facets file. With
`sumfacets=false` (default), per-facet visibilities are kept in
separate buffers (consumable by DDECal). `savefacets` writes
`facet<N>.fits`. In 6.5 the predict buffer is sized to 50 % of free
memory (was 10 %).

#### `idgpredict` — `IDGPredict`

Image Domain Gridder facet predict (requires IDG). Same `images` /
`regions` / `savefacets` semantics, plus IDG aterm support
(`aterms`, `saveaterms`, `atermkernelsize`, and per-aterm config blocks
including `<aterm>.beam.*`).

#### `idgimager` — `IDGImager`

Snapshot imager (one image per time step), originally for AARTFAAC.
`image_size`, `image_name` (`%t` is replaced by the time index),
`dl`/`dm` shifts, `scale` (deg/pixel; `-1` = whole sky), `proxy_type`
(`CPU_OPTIMIZED`, `CUDA_GENERIC`, `CPU_REFERENCE`, `HYBRID`).

#### `sagecalpredict` — `SagecalPredict`

Available only with `HAVE_LIBDIRAC` or `HAVE_LIBDIRAC_CUDA`. Drop-in
replacement for `Predict` / `H5ParmPredict` using SAGECal routines.
Frequency smearing is always on. Inside `ddecal`, opt in with
`sagecalpredict=true`.

#### `grouppredict` — `BdaGroupPredict`

For BDA only; groups baselines with similar averaging factors and runs
a regular `Predict` per group. Mostly a benchmark aid; the regular
`predict` BDA path is faster because of the per-group overhead.

### 5.7 DDECal & related

#### `ddecal` — `DDECal` (regular MSs) / `BdaDdeCal` (BDA)

See §6 for the full solver/constraint architecture. Key parset groups:

- **Sky-model directions:**
  - `sourcedb` and `directions` (list of lists of facets), or
  - `modeldatacolumns` (use MS columns directly), or
  - `reusemodel` (model buffers from a previous step in the chain).
  - `idg.images` / `idg.regions` for facet IDG prediction inside
    DDECal.
- **Solution layout:** `mode` (`scalar`, `scalarphase`,
  `scalaramplitude`, `diagonal`, `diagonalphase`, `diagonalamplitude`,
  `fulljones`, `tec`, `tecandphase`, `rotation`, `rotation+diagonal`,
  `faradayrotation`, `leakage`, `leakageamplitude`).
- **Solver:** `solveralgorithm` (`directionsolve`, `directioniterative`,
  `lbfgs`, `hybrid` — runs 1/6 of `maxiter` direction-solving then
  switches to direction-iterative). `llssolver` (`qr`/`svd`/
  `normalequations`) for direction-solving only. `solverlbfgs.*` knobs
  for LBFGS (history, minibatches, dof, min/max solution clamping).
- **Iteration control:** `maxiter`, `tolerance`,
  `detectstalling`, `stepsigma`, `stepsize`, `minvisratio`,
  `propagatesolutions`, `propagateconvergedonly`, `flagunconverged`,
  `flagdivergedonly`. The TEC mode supports an "approximating" first
  stage — `approximatetec`, `maxapproxiter`, `approxchunksize`,
  `approxtolerance`.
- **Cell sizes:** `solint` (time slots; `0` = whole observation),
  `nchan` (channels per block; `0` = whole band; `1` =
  per-channel-and-fit-constraint),
  `solutions_per_direction` (sub-solutions per `solint` per
  direction), `antenna_averaging_factors` (per-antenna sub-interval
  multipliers, also accepts pattern syntax like `[RS*:5, CS*:2,
  [CS001_HBA001, CS002_HBA001]:3]`), `antenna_smoothness_factors`.
- **Constraints:**
  - `coreconstraint` (m) — equal solutions inside a sphere.
  - `antennaconstraint` — explicit groups.
  - `smoothnessconstraint` (Hz, 3-sigma kernel cut-off) plus
    `smoothnessreffrequency`, `smoothnessspectralexponent`,
    `smoothnessrefdistance`, `smoothnessrefantenna`,
    `smoothness_dd_factors`, `smoothness_kernel_truncation`.
- **Initial solutions:** `initialsolutions.h5parm`,
  `initialsolutions.soltab`, `initialsolutions.interpolation`,
  `initialsolutions.missingantennabehavior`. The 6.5 release removed
  `initialsolutions.gaintype` (the gain type is now deduced from the
  initial-solutions file).
- **Output:** `h5parm` (in 6.5 this is mandatory; the legacy
  `instrument.h5` default was deprecated). `statfilename` writes
  step-size diagnostics. `subtract=true` subtracts the corrected model
  from the data. `keepmodel=true` keeps per-direction corrupted
  visibilities as named extra buffers (`<step>.<patch>`) for the next
  step (e.g. another DDECal in a chain). `reusemodel` consumes them.
- **Faraday/rotation/leakage refinements:** `phasereference`,
  `rotationreference`, `rotationdiagonalmode`, `faradaydiagonalmode`,
  `faradaylimit`.
- **Sub-steps on model data:** `modelnextsteps` (steps applied to
  every direction's model), and the per-direction form
  `modelnextsteps.<direction>=[...]` (e.g.
  `modelnextsteps.MODEL_DATA_2=[applyextrabeam]`).
- **Faceted IDG:** `idg.images`, `idg.regions`, `idg.buffersize`,
  `savefacets`.
- **BDA shortcuts:** `grouppredict=true` falls back to the
  regular-baseline grouped predict for benchmarking.
- **Storage:** `storebuffer=true` stores solutions in the DPBuffer for
  use by a downstream `applycal` without H5Parm I/O (single-direction
  only).
- **GPU:** `usegpu=true` (only the iterative diagonal solver,
  `BUILD_WITH_CUDA=1`), `keep_host_buffers`.
- **Misc:** `uvlambdamin` (and other UVW-flagger-style baselines /
  channels filters), `onlypredict=true` (skip solving — useful for
  benchmarking; combined with `keepmodel=true` keeps per-direction
  buffers without overwriting the main data).

`SolutionWriter` (see §6.4) writes the H5Parm; `SolutionResampler`
upsamples solutions to the input MS time grid when needed.

#### `demixer` / `demix` — `Demixer`

Full A-team demixing (`docs/schemas/Demixer.yml`). Two averaging
schedules: `timestep` / `freqstep` for the post-subtract data, and
`demixtimestep` / `demixfreqstep` (or `demixtimeresolution` /
`demixfreqresolution`) for the solver. Sources are categorized as
`subtractsources` (must have a model), `modelsources` (taken into
account when solving but not subtracted; target must NOT be here),
`targetsource` (model used for target; mutually exclusive with
`othersources`), `othersources` (projected away when demixing).
`ignoretarget=true` ignores rather than projecting away.
`baseline`/`blrange`/`corrtype=cross` (default) restrict the demixer
to selected baselines. Solver:

- `propagatesolutions` (true), `defaultgain` (1.0), `maxiter` (50).
- `uselbfgssolver` (false). LBFGS settings:
  `lbfgs.historysize`, `lbfgs.robustdof`, `lbfgs.solution.range`.
- `ntimechunk` — number of demix time slots to process jointly.

After the run the noise ratio before/after demix is appended to the
HISTORY subtable.

### 5.8 Imaging hooks & writers

- `wscleanwriter` — see §5.2.
- `idgimager` — see §5.6 (technically a predict-flavoured imaging
  step).
- `dynspec` — extracts dynamic spectra by phase-shifting to source
  positions and averaging over baselines. Schema:

  | Key | Doc |
  |-----|-----|
  | `sourcelist` | `.skymodel`/`.txt` source list to use as direction targets. |
  | `subtractmodelcolumn` | Pre-subtract foreground from a model column. |
  | `fitsprefix` | Output FITS stem (`<prefix>-<source>-dynspec.fits`). |
  | `h5parmpredict.*` | Internal H5ParmPredict to subtract foregrounds (alternative to `subtractmodelcolumn`). |
  | `applycal.*` | ApplyCal in the source direction (closest direction in the soltab). |
  | `beamcorrection` | Apply the beam in each source direction (default true). |
  | `applybeam.*` | Tuning for the per-source beam. |

### 5.9 Scaling & python

- `scaledata` — Polynomial-in-frequency SEFD correction.
  `stations=[CS*, RS*, *]` selects glob-patterns to apply
  per-pattern `coeffs=[ [1.5, 0.7, 0.04], [1.7, 0.65], [1.2, 0.8] ]`
  (units: MHz). `scalesize` toggles correction for the number of
  used dipoles/tiles (default true for default coefficients, false for
  user-specified). To correct for station size only,
  `stations=*  coeffs=1  scalesize=true`.
- `pythondppp` / `python` — see §10.

### 5.10 Container steps

- `split` / `explode` — copy the data stream into N parallel
  sub-pipelines whose configurations differ on selected keys
  (`replaceparms`). Example:

  ```
  steps=[average,split]
  split.steps=[predict, msout]
  split.replaceparms=[predict.sourcedb, msout.name]
  predict.sourcedb=[skyA.skymodel, skyB.skymodel]
  msout.name=[outA.MS, outB.MS]
  ```

  `split` must be the last step of the parent chain
  (`Split::setNextStep` throws otherwise).

- `out` / `output` / `msout` — intermediate writer (§5.2).

---

## 6. DDECal solver subsystem (`ddecal/`)

### 6.1 Settings (`ddecal/Settings.h`)

`Settings` aggregates every `ddecal.*` parset value into typed members:
mode (`base::CalType`), `solver_algorithm`
(`SolverAlgorithm{kLowRank, kDirectionSolve, kDirectionIterative,
kHybrid, kLBFGS}`), data layout (`SolverDataUse{kSingle, kDual,
kFull}`), per-direction subsolutions, antenna averaging factors,
constraints (core, antenna, smoothness, smoothness DD factors,
antenna-smoothness factors, screen core), solver settings (LLS type,
max iters, tol, step size, stalling), TEC approx settings, rotation
flags, Faraday flags/limits, full LBFGS block (`lbfgs_robust_nu`,
`lbfgs_max_iter`, `lbfgs_history_size`, `lbfgs_minibatches`,
`lbfgs_min_solution`, `lbfgs_max_solution`), GPU flags
(`use_gpu`, `keep_host_buffers`), low-rank counts
(`n_lra_iterations`, `n_lra_power_iterations`), model-data origins
(`model_data_columns`, `reuse_model_data`), facet IDG
(`idg_region_filename`, `idg_image_filenames`), SAGECal switch
(`use_sagecal_predict`), and the directions / sourcedb. Helpers:
`PrepareSubSolutionsPerDirection()`, `GetExpandedSmoothnessDdFactors()`,
`GetReusedDirections()`, `GetSolutionToDirectionVector()`.

### 6.2 Calibration types (`base/CalType.h`)

`CalType{kScalar, kScalarAmplitude, kScalarPhase, kDiagonal,
kDiagonalAmplitude, kDiagonalPhase, kFullJones, kTecAndPhase, kTec,
kTecScreen, kRotationAndDiagonal, kRotation, kFaradayRotation,
kLeakage, kLeakageAmplitude}`. `GetNPolarizations(CalType)` returns
1/2/4 for scalar/diagonal/full modes. `kTecScreen` requires
`ENABLE_SCREENFITTER` (Armadillo).

### 6.3 Gain solvers (`ddecal/gain_solvers/`)

| File | Class | Algorithm |
|------|-------|-----------|
| `SolverBase.{h,cc}` | `SolverBase` | Base class — owns iteration count, accuracy, step size, stalling detector, constraint list, polarization count abstraction, LLS factory, `Solve(FullSolveData/DuoSolveData/UniSolveData, …)` virtual entry points, `Step()` (a damped step toward `next_solutions`), `MakeSolutionsFinite{1,2,4}Pol`, `AssignSolutions`, `ReachedStoppingCriterion`. |
| `SolveData.{h,cc}` | `FullSolveData`, `DuoSolveData`, `UniSolveData` | Solver buffers — weighted data + per-direction model data, packed for the solver. |
| `BdaSolverBuffer.{h,cc}` | `BdaSolverBuffer` | BDA-aware buffer for `BdaDdeCal`. |
| `ScalarSolver.{h,cc}` | `ScalarSolver` | Direction-solving scalar (LLS over directions). |
| `DiagonalSolver.{h,cc}` | `DiagonalSolver` | Direction-solving 2-pol diagonal. |
| `FullJonesSolver.{h,cc}` | `FullJonesSolver` | Direction-solving 4-pol; used for full-Jones, rotation, rotation+diagonal, Faraday rotation, leakage. |
| `IterativeScalarSolver<Visibility>.{h,cc}` | direction-iterative scalar (one direction at a time). Specialized for `complex<float>`, `MC2x2FDiag`, `MC2x2F` matrices. |
| `IterativeDiagonalSolver<Visibility>.{h,cc}` | direction-iterative diagonal (specialized on `MC2x2FDiag`/`MC2x2F`). Default DDECal solver. |
| `IterativeFullJonesSolver.{h,cc}` | direction-iterative 4-pol. |
| `IterativeDiagonalSolverCuda.{h,cc}` | CUDA implementation of the diagonal iterative solver (built when `BUILD_WITH_CUDA=ON`). |
| `DiagonalLowRankSolver.{h,cc}` | LRA solver: `n_lra_iterations`, `n_lra_power_iterations`. |
| `LBFGSSolver.{h,cc}` | LBFGS solver (`HAVE_LIBDIRAC` only). Modes: `kScalar`, `kDiagonal`, `kFull`. Robust noise dof, mini-batching, history size, real/imag clipping. |
| `HybridSolver.{h,cc}` | Composes a list of solvers, calling each with its own `max_iterations`. The first 1/6 of total iterations runs the direction-solving algorithm, then the direction-iterative one — letting calibration converge from a stable but slow start. |
| `kernels/` | CUDA kernels for the GPU diagonal solver. |
| `SolverFactory.{h,cc}` | `CreateSolver(settings, algorithm, station_names)` decides which solver class to instantiate based on `settings.mode` and `algorithm`, attaches constraints, and calls `InitializeSolver()`. |
| `SolverTools.{h,cc}` | Helper math (matrix Hermitian, Frobenius, etc.). |

`SolverFactory.cc:269-320` is the canonical decision tree:

```text
mode = kScalar / kScalarAmplitude        → scalar solver, phase_only=false
mode = kScalarPhase / kTec / kTecAndPhase → scalar solver, phase_only=true
mode = kDiagonal / kDiagonalAmplitude    → diagonal solver, phase_only=false
mode = kDiagonalPhase                    → diagonal solver, phase_only=true
mode = kFullJones / kRotation / kRotationAndDiagonal /
       kFaradayRotation / kLeakage / kLeakageAmplitude
                                         → full-Jones solver, phase_only=false
mode = kTecScreen                        → scalar solver, phase_only=true (requires ScreenFitter)
```

For each mode the algorithm enum picks a concrete class (and rejects
unsupported combos: e.g. low-rank only supports diagonal,
`solver_data_use=kSingle` only on the iterative scalar solver).

### 6.4 Constraints (`ddecal/constraints/`)

Constraints are `Constraint` subclasses with `Initialize`, optional
`PrepareIteration`, `Apply(SolutionSpan, time)`, `Satisfied`, and
optional `GetResult`. The solver invokes `ApplyConstraints` after every
gain step.

| File | Class | Doc |
|------|-------|-----|
| `Constraint.h` | `Constraint` | Base class. `solutions` is a 4-D span `(n_channel_blocks, n_antennas, n_sub_solutions, n_pol)` with the polarization axis fastest-changing. |
| `AmplitudeOnlyConstraint.h` | Forces \|J\|, used by `kScalarAmplitude`/`kDiagonalAmplitude`/`kLeakageAmplitude`. |
| `AntennaConstraint.h` | Equal solutions inside named antenna groups. Used for `coreconstraint` (groups detected by Euclidean distance from station 0 in `DetermineCoreAntennas()`) and `antennaconstraint` (parsed groups, names matched against the antenna table). |
| `AntennaIntervalConstraint.{h,cc}` | Different solution intervals per antenna (averaging factors). Used when `antenna_averaging_factors` is set. |
| `FaradayConstraint.{h,cc}` | Differential Faraday rotation fitted across frequency. `faraday_diagonal_mode` chooses how the residual diagonal is constrained; `faradaylimit` clamps the search range. |
| `RotationConstraint.{h,cc}` | Pure rotation. |
| `RotationAndDiagonalConstraint.{h,cc}` | Joint rotation + diagonal/scalar fit; `rotationdiagonalmode` configures the diagonal sub-mode. |
| `PolarizationLeakageConstraint.{h,cc}` | Leakage solving (`kLeakage`/`kLeakageAmplitude`). |
| `SmoothnessConstraint.{h,cc}` | Convolves solutions with a (truncated by default) Gaussian kernel of given size. Per-antenna scaling (`SetAntennaFactors`), per-direction scaling (`SetDdSmoothingFactors`), DD-weights from model magnitudes. Reference frequency / reference distance allows the kernel size to scale with frequency or with distance to a reference antenna. The kernel is implemented in `KernelSmoother.h`. |
| `KernelSmoother.h` | Gaussian convolution of complex solutions across channel blocks. |
| `TecConstraint.{h,cc}` | Differential TEC fit using `PhaseFitter`. Modes: `kTecOnly`, `kTecAndCommonScalar`. `ApproximateTECConstraint` uses `PieceWisePhaseFitter` first, then falls back to the full constraint. |
| `PieceWisePhaseFitter.h` | Piece-wise local phase fitter for the approximating TEC stage. |
| `PiercePoint.{h,cc}` / `KLFitter.{h,cc}` / `ScreenFitter.{h,cc}` / `ScreenConstraint.{h,cc}` | TEC-screen constraint (Karhunen-Loève fit over pierce points) — only with `ENABLE_SCREENFITTER`. |
| `TecOffsetDelayFitting.{h,cc}` | Helper for combined TEC/offset/delay refits. |

`AddConstraints()` (`ddecal/SolverFactory.cc:170-256`) is the canonical
mapping from `mode` and parset flags to constraint instances. Always:
core/antenna constraint if requested; antenna-interval constraint if
factors given; smoothness if non-zero. Then per mode the appropriate
specialised constraint is appended.

### 6.5 Linear solvers (`ddecal/linear_solvers/`)

`LLSSolver` (`LLSSolver.h`) abstracts the per-iteration linear least
squares step used by the direction-solving algorithms. Three back-ends
selectable via `llssolver`:

- `QRSolver.h` — LAPACK QR.
- `SVDSolver.h` — LAPACK SVD.
- `NormalEquationsSolver.h` — `A^H A x = A^H b`.

`LLSSolver::Make(type, m, n, nrhs)` is the factory; the result is owned
by each gain solver and recreated when matrix dimensions change.

### 6.6 Solutions plumbing

- `Solutions.h` — typedefs for `SolutionTensor` (a 4-D xtensor) and
  `SolutionSpan` (a non-owning view used for hot paths and Python).
- `SolutionResampler.{h,cc}` — interpolates per-channel-block / per-time
  solutions onto the full input grid (used when `solint > 1` or
  `nchan > 1`).
- `SolutionWriter.{h,cc}` — H5Parm output (uses
  `schaapcommon::h5parm`). `Write(...)` takes the time-major
  3-D solution structure plus per-iteration constraint results,
  the source-direction list, the time grid, and the channel-block
  frequencies. The H5Parm contains a soltab per quantity (amplitude,
  phase, tec, rotation, etc.) with axes for time, antenna, frequency,
  direction, and (where relevant) pol. The history is set from the
  parset string.

---

## 7. Sky model

### 7.1 The `.skymodel` text format

`docs/schemas/Predict.yml` and the example
`resources/tNDPPP-generic-skymodel.txt` describe the format. The first
non-comment line is a `FORMAT` declaration; each subsequent line is a
source. Default format:

```
# (Name, Type, Patch, Ra, Dec, I, ReferenceFrequency,
#   SpectralIndex='[]', LogarithmicSI, MajorAxis, MinorAxis,
#   Orientation, OrientationIsAbsolute) = format
, , 0002.2+3139, 00:02:14.141, +031.39.42.012
0002.2+3139, POINT, 0002.2+3139, 00:02:14.141, +031.39.42.012,
   4.845, 6e+07, [-0.6711, -0.1114], true
0010.4+3329, GAUSSIAN, 0010.4+3329, 00:10:25.440, +033.29.39.516,
   8.5526, 6e+07, [-0.8742], true, 93.2, 32.3, 4.6, false
```

Lines whose `Name` and `Type` are empty define a *patch*: the centre
of all subsequent components belonging to that patch. Concrete fields
(`parmdb/SkymodelToSourceDB.cc:30-90`):

- Identification: `Name`, `Type` (`POINT`, `GAUSSIAN`, `SHAPELET`),
  `Patch`, `Cat`.
- Direction: `Ra`, `Dec` (sexagesimal or decimal), `RefType`
  (`Rah`/`Rad`/`Ram`, `Decd`/`Decm`/`Decs`).
- Stokes: `I`, `Q`, `U`, `V`. `RefFreq` (Hz).
- Spectrum: `SpectralIndex` list. With `LogarithmicSI=true`, the model
  is `S(ν) = I × (ν/ν_ref)^(α_0 + α_1 log10(ν/ν_ref) + …)`. With
  `LogarithmicSI=false`, polynomial in `(ν/ν_ref)`.
- Polarization: `RotMeas` (rad·m^-2), `PolFrac`, `PolAng`, `RefWavel`.
- Gaussian: `MajorAxis`, `MinorAxis` (FWHM, arcsec), `Orientation`
  (rad/deg, position angle), `OrientationIsAbsolute` (whether the
  orientation is w.r.t. local declination axis or the phase-centre's
  declination). Recommended `OrientationIsAbsolute=true` (introduced
  in 5.3) for correct projection.
- Shapelets: `IShapelet`, `QShapelet`, `UShapelet`, `VShapelet` —
  point to coefficient files.

### 7.2 In-memory representation (`base/`, `model/`)

- `Direction` (`base/Direction.h`) — `{double ra; double dec;}` in
  radians.
- `Stokes` (`base/Stokes.h`) — `{double I, Q, U, V;}`.
- `ModelComponent` (`base/ModelComponent.h`) — abstract;
  `direction()`, `accept(visitor)`.
- `PointSource` (`base/PointSource.h`) — Stokes, reference frequency,
  spectral terms (`hasLogarithmicSI()`), rotation measure (set via
  `setRotationMeasure(fraction, angle, rm)`). `stokes(freq)` evaluates
  the spectrum.
- `GaussianSource` (`base/GaussianSource.h`) — derives from
  `PointSource`; adds `setMajorAxis(fwhm)`, `setMinorAxis(fwhm)`,
  `setPositionAngle(angle)`, `setPositionAngleIsAbsolute(bool)`. The
  *absolute* flag toggles which projection is used (the historical bug
  was fixed in 5.3.0).
- `Patch` (`model/Patch.h`) — named bag of `ModelComponent`s with a
  centroid `direction` and an aggregated `brightness`. Computes its
  centroid via `ComputeDirection()`. Has a numeric `index` used by
  `OnePredict` to associate beam evaluations.
- `model/SourceDBUtil.{h,cc}` — `SourceDBWrapper(name)` selects either
  the legacy binary `parmdb::SourceDB` or the in-memory
  `parmdb::SourceDBSkymodel` based on the file extension.
  `MakePatchList()` and `MakePatches(...)` expand patches by name (or
  pattern, via `clusterProximateSources(patches, proximity)`),
  flag absolute orientation
  (`CheckAnyOrientationIsAbsolute()`), and check polarization
  (`CheckPolarized()`). `MakeDirectionList()` parses
  `directions=[[a,b],[c]]` form.
- `model/SkyModelCache.h` — global `SkyModelCache::GetInstance()`.
  `GetSkyModel(filename)` lazily loads and caches a `SourceDBWrapper`
  per file. **Refuses non-text inputs**: legacy binary `makesourcedb`
  output is no longer supported and the error message points users to
  `showsourcedb` for back-conversion. The cache is cleared after the
  step chain has finished initializing (`base/DP3.cc:366`).

### 7.3 Sky-model evaluation

`OnePredict` (`steps/OnePredict.h`) is the canonical predictor:

- Reads patches from `SourceDBWrapper.MakePatchList()`.
- Optionally builds a list of clustered patches
  (`beamproximitylimit`).
- Per time slot:
  - Computes per-source `(l, m, n)` from `Direction` and the phase
    centre using `radec2lmn()` (`base/Simulator.h:35-50`).
  - Computes per-station UVW from baseline UVW via
    `SetupUvwSplitting()` (`base/Simulate.h`).
  - Visits each `ModelComponent` (point/Gaussian/shapelet) via the
    `ModelComponentVisitor`. The result is a per-baseline,
    per-channel, per-correlation `(complex<float>) buffer`.
  - Applies the EveryBeam Jones matrices through
    `addBeamToData(...)`. Beams are reused across `beam_interval`
    seconds; per-source clustering reduces beam evaluations further.
  - Optionally applies an `ApplyCal` sub-step (corrupting the model
    with stored gains).
  - Combines into the main DPBuffer or an extra buffer
    (`outputmodelname`). Final operation is `replace`/`add`/`subtract`.
- Smearing: `correcttimesmearing` and `correctfreqsmearing` multiply
  by a sinc factor (`sin(πx)/(πx)`).
- Threading: prediction can be parallelized over sources or over
  baselines (`parallelbaselines=true`). A shared `measures_mutex_` is
  used when multiple `OnePredict` steps run in parallel.

### 7.4 ParmDB / SourceDB legacy

`parmdb/` contains the old casacore-table SourceDB and ParmDB
implementation: `ParmDB`, `ParmFacade`, `ParmCache`, `ParmSet`,
`ParmValue`, `ParmMap`, `ParmDBCasa`, `ParmDBBlob`, `ParmDBLocker`,
`ParmDBMeta`, plus axes and grids (`Axis`, `AxisMapping`, `Box`,
`Grid`). `SourceDB` (`parmdb/SourceDB.h`) wraps one of:

- `SourceDBCasa` (binary CASA Tables format),
- `SourceDBBlob` (custom Blob-stream format),
- `SourceDBSkymodel` (in-memory representation built from the textual
  `.skymodel` file by `SkymodelToSourceDB.cc`).

`showsourcedb.cc` (the executable) lets the user back-convert a binary
SourceDB to text format. `SkymodelToSourceDB::MakeSourceDb(in, out,
outType, format, prefix, suffix, append, average, check, search_info)`
builds either output. New DP3 code should always use the in-memory
`SourceDBSkymodel` route — see §7.2 — since `SkyModelCache` rejects
binary inputs.

The ParmDB is, however, still actively used:

- `ApplyCal` / `OneApplyCal` accept either ParmDB or H5Parm.
- `GainCal` writes a ParmDB unless the parmdb name ends in `.h5` (then
  H5Parm).
- DDECal's `SolutionWriter` writes H5Parm exclusively.

`PhaseFitter` (`base/PhaseFitter.h`) is the workhorse for TEC fitting
inside the TEC constraint: a per-station phase fitter that handles
phase wraps, with `FitDataToTEC1Model()` (TEC only) and
`FitDataToTEC2Model()` (TEC + delay).

---

## 8. Streaming, BDA, and IDG facetting

### 8.1 Streaming

`stream.socket=/path` triggers the `SVPInput` step instead of an
`MsReader`. The reader pumps DPBuffers from a Unix socket framed by
the cobalt/ALMA streaming protocol (`steps/SVPInput.cc`). 6.5 added
support for cobalt (LOFAR) streaming alongside the original ALMA
prototype.

### 8.2 BDA chain

The pipeline can choose between two MS types per step. The BDA chain
typically looks like:

```
msin → bdaaverager → ddecal (BdaDdeCal, BdaSolverBuffer) → msout
or
msin (kBda)  → bdaexpander → predict → msout
```

Steps that explicitly support BDA: `BDAAverager`, `BdaExpander`,
`BdaGroupPredict`, `BdaDdeCal`, `MSBDAReader`, `MSBDAWriter`,
`Predict` (BDA-aware path that constructs an internal `BdaAverager`
when needed), `Clipper`, `OnePredict` (when receiving `BdaBuffer`),
`scaledata`. `MakeOutputStep` rejects in-place updates of BDA MSs.

`docs/idg-facetting/idg-facetting.tex` is a long-form derivation of
how facetted IDG calibration is integrated into DDECal — in-tree
reference document for `idg.images`, `idg.regions`, IDG aterms, and the
gridding kernel sizes.

### 8.3 Beam-direction metadata

Buffers can carry per-direction metadata so that `applybeam` and
`applycal` know which direction a model belongs to without an explicit
parameter. The `extra_directions_` map in `DPInfo` is keyed by buffer
name and stores `Direction{ra, dec}`. `WGridderPredict` and
`IDGPredict` set those when populating the regions list, and
`ApplyBeam`/`ApplyCal` consume them when `usemodeldata=true`. This is
how the new `combine`/`outputmodelname` workflow stays
direction-aware.

---

## 9. Demixing — algorithmic notes

`Demixer` (`steps/Demixer.h`, `base/EstimateMixed.{h,cc}`,
`base/EstimateMixedLBFGS.cc`) implements the joint Jones-matrix
estimation for multiple A-team directions. Highlights:

- Simultaneously estimates Jones matrices in all directions present in
  `subtractsources`, `modelsources`, `targetsource` (if given), and
  `othersources`.
- The "other" directions and (optionally) the target are *projected
  away* in the estimation step to suppress contamination.
- Per-direction averaging windows (`demixtimestep`/`demixfreqstep`)
  can differ from the post-subtract window (`timestep`/`freqstep`).
- LSQfit (Levenberg-Marquardt) is the default solver; `uselbfgssolver`
  switches to LBFGS with a robust noise model
  (`lbfgs.robustdof` ≈ 30 ⇒ Gaussian, smaller ⇒ outlier-robust). The
  LBFGS solver also supports per-component clipping
  (`lbfgs.solution.range`).
- `propagatesolutions=true` warm-starts the next slot with the
  previous slot's solutions; off, the diagonal is initialized to
  `defaultgain` and off-diagonals to zero.
- Sky-model: only point and Gaussian sources are recognized; the
  averaged centroid is used as the per-patch direction.
- Reports a noise-ratio summary appended to the MS HISTORY subtable.

---

## 10. Python API (`pythondp3/`)

### 10.1 Build & layout

`pythondp3/` produces a pybind11 module `dp3.pydp3` plus the high-level
package `dp3` (`pythondp3/__init__.py`). The pip wheel installs both
plus an executable `DP3.py` (`entrypoint`).

```
pythondp3/
├── __init__.py             # high-level dp3 API and StepWrapper
├── pydp3.cc                # PYBIND11_MODULE(pydp3, m)
├── parameterset.cc         # ParameterSet bindings
├── PyDpBuffer.{h,cc}       # DPBuffer bindings (numpy zero-copy)
├── PyDpInfo.cc             # DPInfo bindings
├── PyFields.cc             # Fields bindings
├── pyfitters.cc            # PhaseFitter bindings
├── PyStep.{h,cc}           # Embeddable Python step
├── steps/                  # Python steps shipped with the wheel
│   ├── README.md
│   └── queue.py            # example queue-based step
├── test/                   # pytest unit + integration tests
└── CMakeLists.txt
```

### 10.2 Top-level API (`dp3` package)

Re-exported from `pydp3` (`pythondp3/__init__.py:50`):

```python
from .pydp3 import (
    DPBuffer, DPInfo, Fields, MsType,
    execute, execute_from_command_line,
    get_chain_required_fields, get_n_threads, set_n_threads,
)
```

- `execute(parset_name, argv_list)` — run an existing parset with
  optional CLI overrides. Releases the GIL during execution; raises
  `RuntimeError` on errors.
- `execute_from_command_line(argv_list)` — full CLI parser
  equivalent.
- `make_step(type, parset, prefix, input_type)` — build a single step
  (wraps `MakeSingleStep`); the returned wrapper auto-attaches a
  `null` next step.
- `make_main_steps(parset)` — build a full chain
  (`MakeMainSteps`).
- `get_chain_required_fields(first_step)` — propagate the field
  requirements through a chain.
- `get_n_threads`, `set_n_threads` — thread-pool size.

### 10.3 The `Step` base class

Two layers of Python class wrap the C++ `dp3::steps::Step`:

- `dp3.pydp3.Step` — raw pybind11 binding. Methods: `update_info`,
  `set_info`, `process(buffer)`, `finish`, `get_next_step`,
  `set_next_step`, `get_required_fields`, `get_provided_fields`,
  `show`, `show_timings`, plus `info_in`, `info_out`, `_info_out`
  (writable reference for in-place metadata updates inside
  `update_info`).
- `dp3.Step` (Python) — derives from `dp3.pydp3.Step` and adds shape
  validation; `dp3.StepWrapper` wraps step pointers returned from C++
  to provide the same interface.

### 10.4 Authoring a Python step

(`pythondp3/test/unit/mock/mockpystep.py` is the worked example.) A
custom step inherits from `dp3.pydp3.Step` (the raw class for
maximum control) or `dp3.Step` (with extra checks). It must:

1. Implement `__init__(self, parset, prefix)` and call
   `super().__init__()`.
2. Override `get_required_fields()` and `get_provided_fields()` to
   return `Fields.DATA | Fields.WEIGHTS | …` bitmasks.
3. Implement `process(self, dpbuffer)` to mutate the buffer and call
   `self.get_next_step().process(dpbuffer)`.
4. Optionally override `show()`, `update_info()`, and `finish()`.

Buffers expose numpy-zero-copy views via `np.array(buffer.get_data(),
copy=False)`, ditto for `get_weights`, `get_flags`, `get_uvw`. Setting
`PYTHONPATH` to point at DP3's installed Python directory is required
when the host process is the C++ `DP3` binary; embedded steps run via
the parset:

```
steps=[mystep]
mystep.type=python
mystep.python.module=mockpystep
mystep.python.class=MockPyStep
mystep.datafactor=2.0
mystep.weightsfactor=0.5
```

### 10.5 ParameterSet bindings

`ParameterSet` is bound from `pythondp3/parameterset.cc` and exposes
typed getters (`get_string`, `get_double`, `get_int`, `get_bool`,
`get_string_vector`, `get_int_vector`, …) so Python steps can parse
their own parset prefix consistently.

---

## 11. Diagnostics, logging, statistics

- `aocommon::Logger` is the only output channel; `verbosity`,
  `time_logging`, `memory_logging` are global. All steps print via the
  logger (no direct `std::cout`/`std::cerr` since 6.2).
- Flag percentages: every flagging step (`AOFlagger`, `MADFlagger`,
  `PreFlagger`, `UVWFlagger`, `Counter`, `MsReader` for NaN/Inf) can
  emit `<msbase>_<stepname>.flagfreq` and
  `<msbase>_<stepname>.flagstat` casacore tables. The Python helper
  `lofar.dppp.plotflags` plots them. Multi-subband averaging across a
  list of paths is supported.
- `Counter` additionally produces a JSON file (`savetojson=true`,
  `jsonfilename` defaults to `FlagPercentagePerStation.JSON`).
- `showtimings=true` prints a per-step elapsed-time percentage at the
  end. `showcounts=true` prints per-step flag-count summaries.
- `statfilename` in DDECal writes per-iteration step magnitudes for
  later analysis.
- `debuglevel=1` in `GainCal` writes a full-history `debug.h5` of
  every iterand.
- `addToMS()` is the hook for steps that want to write extra subtables
  (AOFlagger writes QUALITY).

---

## 12. Building DP3

### 12.1 CMake

`CMakeLists.txt:14` pins `DP3_VERSION` (currently 6.5.1; keep in sync
with `setup.py`). Required dependencies at configure time:

- `cmake ≥ 3.15` (3.19+ enables the policies CMP0074 and CMP0110;
  3.24+ flips CMP0135).
- C++20 compiler; GCC ≥ 10.
- `boost ≥ 1.73` (filesystem, program-options, python, test).
- `casacore ≥ 3.7.1`.
- `EveryBeam ≥ 0.7.4 < 0.9.0`.
- `aoflagger ≥ 3.1`, `HDF5`, `CFITSIO`, `GSL`, `Threads`, `Python3`
  (with NumPy).
- Optional: `Armadillo` (TEC-screen), `IDGAPI`, `libdirac` (≥ 0.8.4
  < 0.8.6, for SAGECal/LBFGS), CUDA toolkit.
- pulled by FetchContent: `cudawrappers` (when CUDA on), pybind11.

Typical build:

```sh
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_TESTING=OFF \
         -DUSE_FAST_PREDICT=OFF \
         -DBUILD_WITH_CUDA=OFF
make -j$(nproc)
make install
```

CMake options of interest: `BUILD_PACKAGES` (Debian builds),
`METADATA_COMPRESSION_DEFAULT`, `USE_FAST_PREDICT`,
`ENABLE_SCREENFITTER`, `BUILD_DOCUMENTATION`, `BUILD_TESTING`.

### 12.2 setup.py / pip wheel

`setup.py` invokes CMake under the hood (`CMakeBuild` from
`pybind11/cmake_example`). The pip wheel contains the `dp3` Python
package, the `pydp3` extension, and the `DP3.py` entrypoint. 6.4 added
binary wheels for Python 3.13 and dropped 3.7.

### 12.3 Docker baselines

`docker/ubuntu_22_04_base` and `docker/ubuntu_24_04_base` are the
canonical reference environments. Each pulls fixed git-hashes for
EveryBeam, IDG, AOFlagger, casacore, SAGECal. The `install_*.sh`
scripts under `docker/` document the source-build recipes.

### 12.4 CI

`ci/` plus the gitlab-ci configuration runs the docker image, builds,
runs the gtest/Boost.Test C++ tests, runs the pytest tests under
`pythondp3/test/`, and builds the docs (`jsonschema2rst` converts
`docs/schemas/*.yml` to RST, then Sphinx renders them).

---

## 13. Recent change-log highlights (from `CHANGELOG.md`)

| Version | Notable additions |
|---------|-------------------|
| 6.5.1 (2025-10-09) | Compilation fix for BDA support; `execute`/`execute_from_command_line` argument handling fixed in the Python API. |
| 6.5 (2025-10-01) | `ddecal.antenna_averaging_factors` / `antenna_smoothness_factors`; BDA `ddecal` `keepmodel`/`reusemodel`; `predict.coefficients_path` (e.g. MWA); `predict.outputbuffername`; new `transfer` step (data + flags); new `combine` step (add/subtract named buffers); Sisco compression in casacore; opt-in `FastPredict` via `USE_FAST_PREDICT`; cobalt streaming; numpy 2.x; default `predict` wgridder buffer is now 50 % of memory; `ddecal.h5parm` is mandatory; `ddecal.initialsolutions.gaintype` removed; preflagger faster; `predict` time-smearing approximation switched to sinc; EveryBeam ≥ 0.7.4 required. |
| 6.4.1 (2025-05-22) | New `rotationconstraint` for Faraday rotation; rotation-and-diagonal constraint with multiple directions; missing beam keywords on BDA. |
| 6.4 (2025-04-14) | Metadata compression (`msout.uvwcompression`, `msout.antennacompression`); `clipper.flagallcorrelations`; `ddecal.initialsolutions`; `ddecal.smoothness_dd_factors`; `ddecal.smoothness_kernel_truncation`; `msin.starttime` semantics fixed; casacore 3.7.1; py 3.13 wheel, drop 3.7. |
| 6.3 (2025-01-28) | `flagtransfer` step; new casacore Stokes-I storage manager; `msout.scalarflags`; `ApplyBeam`/`Predict` now use the EveryBeam parser for `elementmodel` (default deduced from MS); EveryBeam 0.7.x; per-direction constraint weights; C++20 / GCC 10. |
| 6.2 (2024-08-29) | Extra MS data columns (`msin.extradatacolumns`); `wgridderpredict`; `applybeam` extra columns; `ddecal.rotationdiagonalmode` / `smoothnessspectralexponent` / `usedualvisibilites`; LBFGS solution-range; `applybeam.skipstations`; DD intervals in BDA `ddecal`; `wscleanwriter`. |
| 6.1 (2024-06-18) | EveryBeam 0.6 support; cached-time-direction beam predict; dish telescopes (SKA-mid); preliminary `clipper`; lower memory predict; AVX/AVX2; xtensor migration. |
| 6.0 (2023-08-11) | Optional SAGECal (libdirac); model-data reuse via `extradata`; thoroughly refactored Python bindings (full pipelines from Python); AARTFAAC-specific steps. |

---

## 14. Quick recipes

### 14.1 Minimal flag/copy

```ini
msin = ~/SB0.MS
msout = SB0-preprocessed.MS
steps = []
```

(Implicitly flags NaN/Inf in the input.)

### 14.2 Flag → average → flag → average

```ini
msin = ~/SB0.MS
msin.startchan = 8
msin.nchan = 240
msout = SB0-averaged.MS

steps = [flag1, count, avg1, flag2, avg2, count]
flag1.type = madflagger
flag1.threshold = 1
flag1.freqwindow = 31
flag1.timewindow = 5
flag1.correlations = [0,3]
flag1.count.save = true
flag1.count.path = $HOME

avg1.type = average
avg1.freqstep = 240

flag2.type = madflagger
flag2.threshold = 2
flag2.timewindow = 51

avg2.type = average
avg2.timestep = 5
```

### 14.3 Predict + DDE calibrate

```ini
msin = obs.MS
msout = obs-calibrated.MS

steps = [predict, ddecal]

predict.type = predict
predict.sourcedb = sky.skymodel
predict.usebeammodel = true
predict.outputmodelname = MODEL_DATA

ddecal.type = ddecal
ddecal.reusemodel = [predict.MODEL_DATA]
ddecal.h5parm = solutions.h5
ddecal.mode = diagonal
ddecal.solveralgorithm = directioniterative
ddecal.solint = 4
ddecal.nchan = 8
ddecal.smoothnessconstraint = 2e6
ddecal.maxiter = 100
```

### 14.4 Demix three A-team sources

```ini
msin = obs.MS
msout = obs-demixed.MS

steps = [demix]
demix.type = demixer
demix.skymodel = ateam.skymodel
demix.subtractsources = [CasA, CygA, TauA]
demix.timestep = 4
demix.freqstep = 4
demix.demixtimestep = 64
demix.demixfreqstep = 64
demix.targetsource = ""
demix.ignoretarget = true
demix.maxiter = 50
```

### 14.5 Apply existing solutions

```ini
msin = obs.MS
msout = obs-corrected.MS
msout.datacolumn = CORRECTED_DATA

steps = [applycal]
applycal.parmdb = solutions.h5
applycal.steps = [amp, phase]
applycal.amp.correction = amplitude000
applycal.phase.correction = phase000
applycal.interpolation = nearest
applycal.invert = true
```

### 14.6 Embed a Python step

```python
# my_invert.py
import numpy as np
from dp3 import Fields
from dp3.pydp3 import Step

class InvertSign(Step):
    def __init__(self, parset, prefix):
        super().__init__()
    def get_required_fields(self):
        return Fields.DATA
    def get_provided_fields(self):
        return Fields.DATA
    def process(self, buf):
        data = np.array(buf.get_data(), copy=False)
        data *= -1.0
        return self.get_next_step().process(buf)
    def finish(self):
        pass
```

```ini
steps = [invert]
invert.type = python
invert.python.module = my_invert
invert.python.class = InvertSign
```

(Run with `PYTHONPATH=$PWD DP3 my.parset`.)

---

## 15. Cross-references

### 15.1 The complete step → header map

(Index of `MakeSingleStep`, `base/DP3.cc:162-257`.)

| Type strings | Header / Class |
|--------------|----------------|
| `aoflagger`, `aoflag` | `steps/AOFlaggerStep.h` → `AOFlaggerStep` |
| `averager`, `average`, `squash` | `steps/Averager.h` → `Averager` |
| `bdaaverage`, `bdaaverager` | `steps/BDAAverager.h` → `BdaAverager` |
| `bdaexpander` | `steps/BdaExpander.h` |
| `madflagger`, `madflag` | `steps/MadFlagger.h` |
| `preflagger`, `preflag` | `steps/PreFlagger.h` |
| `antennaflagger`, `antflag` | `steps/AntennaFlagger.h` (uses `antennaflagger/Flagger.cc`) |
| `uvwflagger`, `uvwflag` | `steps/UVWFlagger.h` |
| `clipper` | `steps/Clipper.h` |
| `combine` | `steps/Combine.h` |
| `columnreader` | `steps/MsColumnReader.h` |
| `counter`, `count` | `steps/Counter.h` |
| `phaseshifter`, `phaseshift` | `steps/PhaseShift.h` |
| `demixer`, `demix` | `steps/Demixer.h` (`base/EstimateMixed.cc`, `base/EstimateMixedLBFGS.cc`) |
| `applybeam` | `steps/ApplyBeam.h` |
| `stationadder`, `stationadd` | `steps/StationAdder.h` |
| `scaledata` | `steps/ScaleData.h` |
| `setbeam` | `steps/SetBeam.h` |
| `filter` | `steps/Filter.h` |
| `applycal`, `correct` | `steps/ApplyCal.h` (`OneApplyCal.h`) |
| `nullstokes` | `steps/NullStokes.h` |
| `predict` | `steps/Predict.h` (`OnePredict.h`, optional `FastPredict.h`) |
| `wgridderpredict` | `steps/WGridderPredict.h` |
| `idgpredict` | `steps/IDGPredict.h` |
| `idgimager` | `steps/IDGImager.h` |
| `upsample` | `steps/Upsample.h` |
| `interpolate` | `steps/Interpolate.h` |
| `grouppredict` | `steps/BdaGroupPredict.h` |
| `sagecalpredict` | `steps/SagecalPredict.h` (only with libdirac) |
| `h5parmpredict` | `steps/H5ParmPredict.h` |
| `gaincal`, `calibrate` | `steps/GainCal.h` |
| `python`, `pythondppp` | `pythondp3/PyStep.h` → `PyStep::create_instance(...)` |
| `split`, `explode` | `steps/Split.h` |
| `ddecal` | `steps/DDECal.h` (regular) / `steps/BdaDdeCal.h` (BDA) |
| `dynspec` | `steps/DynSpec.h` |
| `null` | `steps/NullStep.h` |
| `wscleanwriter` | `steps/WSCleanWriter.h` |
| `flagtransfer` | `steps/FlagTransfer.h` (deprecated) |
| `transfer` | `steps/Transfer.h` |
| `out`, `output`, `msout` (intermediate) | `steps/MSWriter.h` / `steps/MSUpdater.h` / `steps/MSBDAWriter.h` |

### 15.2 Schema directory

`docs/schemas/` contains one YAML per step (and one global). When the
docs are built (`jsonschema2rst docs/schemas/ docs/steps/` then
Sphinx), each YAML is rendered as a parameter table on
[dp3.readthedocs.io](https://dp3.readthedocs.io). Currently shipped:

```
AntennaFlagger.yml      ApplyBeam.yml          ApplyCal.yml
Averager.yml            BDAAverager.yml        Clipper.yml
ColumnReader.yml        Combine.yml            Counter.yml
DDECal.yml              Demixer.yml            DynSpec.yml
Filter.yml              FlagTransfer.yml       GainCal.yml
H5ParmPredict.yml       IDGImager.yml          IDGPredict.yml
Input.yml               Interpolate.yml        MADFlagger.yml
NullStokes.yml          Null.yml               Output.yml
PhaseShift.yml          Predict.yml            PreFlagger.yml
PythonStep.yml          SagecalPredict.yml     ScaleData.yml
SetBeam.yml             Split.yml              StationAdder.yml
SVPInput.yml            Transfer.yml           Upsample.yml
UVWFlagger.yml          WGridderPredict.yml    WSCleanWriter.yml
"Description of all parameters.yml"
"Description of baseline selection parameters.yml"
"AOFlagger.yml"
```

### 15.3 Useful entry-point files when reading the source

| File | Why |
|------|-----|
| `base/Main.cc` | The 33-line CLI shim. |
| `base/DP3.cc` | The Execute / MakeSingleStep / MakeMainSteps logic. |
| `base/DPInfo.h`, `base/DPBuffer.h`, `base/BdaBuffer.h` | The data model. |
| `common/Fields.h` | The four-bit field bitmask. |
| `steps/Step.h` | The abstract base class every step inherits. |
| `ddecal/Settings.h` | All DDECal parset keys in one place. |
| `ddecal/SolverFactory.cc` | The mode/algorithm decision tree. |
| `ddecal/SolutionWriter.h` | H5Parm output schema. |
| `model/SourceDBUtil.h` | Sky-model loading & patch construction. |
| `model/SkyModelCache.h` | Why binary `makesourcedb` outputs are no longer accepted. |
| `pythondp3/__init__.py` | The high-level `dp3` Python API. |
| `pythondp3/pydp3.cc` | The pybind11 module definition. |
| `docs/index.rst` | Human-readable overview. |

---

## 16. Things to know if you are coming from the simulator side

DP3 is *not* a forward simulator like `pyuvsim`, `matvis`, `fftvis`, or
`hera_sim`. Its prediction code (`Predict`/`OnePredict`/`FastPredict`/
`H5ParmPredict`/`SagecalPredict`/`WGridderPredict`/`IDGPredict`)
exists primarily as a *calibration prerequisite*: build the model
visibilities you want to subtract, divide by them, or solve against.
A few practical implications:

- Every prediction lives inside an MS-driven pipeline. There is no
  free-floating "predict to memory" entry point — the C++ API expects
  a `DPInfo` populated from a real (or streamed) MS to know
  baselines, channels, and pointing. Even from Python you need to
  drive it via a parset and an `MsReader` (regular or BDA).
- Sky models are textual `.skymodel` (point + Gaussian + shapelet).
  Diffuse sky cubes (HEALPix maps, GSM, LFSM) are *not supported*; if
  you have a HEALPix model, render it with WSClean / IDG / wgridder
  to a FITS image and feed that into `wgridderpredict` or
  `idgpredict` with a DS9 facets file.
- Polarization conversion is implicit in the LOFAR/EveryBeam Jones
  pipeline: `usebeammodel=true` adds a station-direction Jones; the
  RIME is `V_pq = J_p · C · J_q^H`. The same beam framework drives
  prediction, calibration, and beam correction.
- The "experimental" `usefastpredict=true` is the closest you get to a
  GPU-accelerated forward model in DP3 today; for end-to-end GPU
  pipelines, look at the iterative-diagonal CUDA solver and `usegpu`
  inside DDECal, but note that GPU coverage is currently solver-only.
- For HERA/MWA-style end-to-end simulation it is more idiomatic to use
  one of the dedicated simulators in this directory and feed the
  resulting MS into DP3 only for downstream calibration / averaging /
  flagging.
- DP3 is the canonical pipeline for LOFAR/AARTFAAC and the de-facto
  pipeline for SKA-low/LOFAR2 prototypes; reading
  `docs/idg-facetting/idg-facetting.tex` is the fastest way to
  understand how DDECal interacts with faceted IDG/wgridder predicts
  for full-sky calibration.

---

*Document derived directly from the in-tree source and YAML schemas;
section ordering follows the natural reading order:
binary → pipeline → ParSet → steps → calibration → sky model →
streaming/BDA → Python → diagnostics → build → recipes → indexes.*
