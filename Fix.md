# RadioSim Remediation and Completion Plan

| Plan metadata | Value |
|---|---|
| Status | Tier 1 accepted; Tier 2 accepted after correction; Tier 3 independently accepted on 2026-07-25; all five Tier 3 issues are done; exact-SHA remote CI observed green |
| Prepared | 2026-07-14 |
| Current release | 0.2.0 |
| Baseline commit | `73ae7a3` (`main`, aligned with `origin/main`) |
| Test inventory | 1,178 tests collected from 69 test files |

## 1. Purpose

This document is the source-of-truth plan for resolving the configuration,
high-level API, instrument-model, beam, output, backend, documentation, and
scientific-completeness gaps found during a source-first review of RadioSim.

It records:

- every issue raised during the codebase walkthrough and follow-up questions;
- what the current implementation actually does;
- why the behavior is inconsistent or incomplete;
- the target behavior and architectural direction;
- a dependency-ordered implementation plan;
- tests, acceptance criteria, and stop gates for every tier;
- work that is a normal defect fix versus work that is a larger scientific
  feature.

This is not authorization to implement all tiers in one change. Each tier must
be delivered and reviewed independently.

### 1.1 History-rewrite boundary

The repository history was intentionally rewritten before this plan was
created. Files removed by that rewrite are outside this plan and must not be
restored merely because an older audit or commit mentioned them.

This plan is based on the live checkout at the baseline above. In particular:

- the former `FIX.md`, `CurrentIssues.md`, and `SkyRemediationPlan.md` are not
  present and are not prerequisites for this work;
- deleted GitHub-history content must not be reconstructed;
- `project.md` remains present locally, but is not treated as runtime truth;
- the current source, tests, manifests, README, Sphinx documentation, examples,
  and `AGENTS.md` govern this plan.

## 2. Repository baseline

The live checkout has the following relevant properties:

| Surface | Current state |
|---|---|
| Package version | `0.2.0` |
| Pixi workspace version | `0.2.0` |
| Sphinx version/release | `0.2.0` |
| Pixi lock format | v7 |
| Python package modules | 150 |
| Test files | 69 |
| Tests collected | 1,178 |
| Sample YAML configs | 2 |
| Jones classes | 46 |
| Integration test directory | Only `__init__.py` |
| Performance test directory | Only `__init__.py` |
| Tracked `.github/` directory | Absent |
| `huggingface_space/` directory | Absent |

The version and lockfile inconsistencies are already resolved by commit
`73ae7a3`. They remain documented below because they were part of the original
review and establish the expected release discipline.

## 3. Correct high-level architecture

The observability planner must not be described as a visibility-simulation
product. It is a sibling planning and visualization capability exposed through
the same high-level API.

```mermaid
flowchart TD
    API["Simulator high-level API"]

    API --> CONFIG["Validated and resolved configuration"]
    CONFIG --> SETUP["Instrument, sky, beam, time, frequency, backend setup"]
    SETUP --> SOLVER["Point-source or HEALPix visibility solver"]
    SOLVER --> RESULT["SimulationResult"]
    RESULT --> WRITERS["HDF5, MS, UVFITS, plots"]

    API --> OBS["Observability sidecar"]
    CONFIG --> OBS
    SETUP -. "optional resolved sky and beam" .-> OBS
    OBS --> PLAN["ObservabilityPlan"]
    PLAN --> OBSOUT["Tracks, footprints, source metrics, contours"]
    OBSOUT --> RENDER["Bokeh or Matplotlib renderer"]
```

The visibility branch produces baseline-dependent complex data. The
observability branch produces a renderer-neutral observing plan and does not
calculate baseline visibilities.

### 3.1 Target architecture after remediation

All entry points should pass through the same validation and resolution stages:

```mermaid
flowchart LR
    YAML["YAML path"] --> LOAD["Load and normalize"]
    MAP["Python mapping"] --> LOAD
    MODEL["RadioSimConfig"] --> LOAD

    LOAD --> VALIDATE["Schema and semantic validation"]
    VALIDATE --> RESOLVE["Resolve runtime configuration"]

    RESOLVE --> CTX["SimulationContext"]
    CTX --> POINT["Point RIME"]
    CTX --> HPX["HEALPix direct sum"]
    POINT --> RESULT["SimulationResult"]
    HPX --> RESULT

    CTX --> OBS["ObservabilityPlanner"]
    RESULT --> IO["Writers and result plots"]
    OBS --> OVIS["Observability renderers"]
```

`SimulationContext` is a conceptual target name. The implementation may use
several focused immutable models instead of one large object, but the resolved
state must have a single owner.

## 4. Governing decisions

### 4.1 Pre-v1 API policy

RadioSim is pre-v1.0. The implementation should prefer a coherent replacement
over compatibility shims when the current API is misleading or structurally
wrong.

Therefore:

- do not add adapters merely to preserve legacy beam-config field names;
- remove redundant boolean switches when the presence of typed data expresses
  the same decision more clearly;
- reject unsupported settings instead of accepting silent no-ops;
- do not preserve raw-dictionary behavior if it prevents one reliable
  configuration contract;
- document intentional breaking changes in the changelog and migration guide;
- add this policy explicitly to `docs/contributing.rst` as required by
  `AGENTS.md`.

### 4.2 Truthfulness rule

A field or public class may be in one of four states:

1. implemented and tested;
2. experimental and explicitly gated;
3. unsupported and rejected with an actionable error;
4. absent from the public surface.

RadioSim must not validate a setting, silently ignore it, and then imply that
the setting affected the simulation.

### 4.3 Precedence must be explicit

Where several sources can provide the same value, resolution precedence must
be centralized, documented, and included in result provenance. It must not be
created accidentally through mutation order in `Simulator.setup()`.

### 4.4 Scientific features require scientific tests

Receptor bases, Jones terms, beam interpolation, hybrid-sky summation, and
spherical-harmonic solvers are not complete merely because code executes. They
require analytic invariants, reference cases, and cross-implementation checks.

## 5. Issue register

Status values used below:

- **DONE**: resolved and verified in the current checkout;
- **OPEN**: confirmed live defect or incomplete contract;
- **DECISION**: design choice required before implementation;
- **ROADMAP**: substantial scientific or performance feature;
- **DOCS**: documentation/example/release-truth work.

| ID | Status | Issue | Planned tier |
|---|---|---|---|
| REL-001 | DONE | Pixi workspace version differed from package/Sphinx | 0 |
| REL-002 | DONE | Pixi lockfile used v6 format | 0 |
| CFG-001 | DONE | CLI/API/mapping/model/parameter inputs share one strict resolution pipeline | 1 |
| CFG-002 | DONE | Typed config and every override surface honor centralized backend precedence | 1 |
| CFG-003 | DONE | Unsupported non-default fields are exhaustively classified and rejected before side effects | 1 |
| INS-001 | DONE | Per-antenna diameter map is ignored and loaded values are overwritten | 2 |
| INS-002 | DONE | pyuvdata telescope flags are not wired | 2 |
| INS-003 | DONE | Baseline-selection fields are ignored | 2 |
| BEAM-001 | DONE | Modern FITS/per-antenna beam config is not connected to `Simulator` | 3 |
| BEAM-002 | DONE | `BeamManager` expects a different legacy config contract | 3 |
| BEAM-003 | DONE | HEALPix NSIDE beam advisor reads antenna dictionaries incorrectly | 3 |
| OBS-001 | DONE | Observability is architecturally misclassified as a product | 3 |
| OBS-002 | DONE | Implement the accepted explicit-reference semantics for heterogeneous observability beams | 3 |
| OUT-001 | OPEN | Output controls are only partially honored | 4 |
| OUT-002 | OPEN | Point, HEALPix, and writer time-grid counts disagree | 4 |
| OUT-003 | OPEN | HDF5 drops correlations and forces `complex128` | 4 |
| OUT-004 | OPEN | JSON output contains no visibility data | 4 |
| OUT-005 | OPEN | HDF5 reader uses unsafe `eval()` | 4 |
| OUT-006 | OPEN | UVFITS is accepted by config but unsupported by `save()` | 4 |
| POL-001 | OPEN | Top-level feed/receptor config is ignored | 5 |
| POL-002 | ROADMAP | Receptor and basis-transform Jones terms are identity stubs | 5 |
| RUN-001 | OPEN | `run(n_workers=...)` is unused | 6 |
| RUN-002 | OPEN | Sky loading hard-codes `max_workers=8` | 6 |
| RUN-003 | OPEN | High-level API forces point or HEALPix and cannot preserve hybrid sky | 6 |
| RUN-004 | ROADMAP | Backend abstraction is not yet performance-bearing end to end | 6 |
| SCI-001 | ROADMAP | Most Jones classes are public identity-returning stubs | 7 |
| SCI-002 | ROADMAP | Spherical-harmonic/m-mode mode is advertised but unimplemented | 7 |
| SCI-003 | ROADMAP | Advanced beam-physics TODOs remain | 7 |
| DOC-001 | DOCS | `simple_simulation.py` uses stale private/result APIs | 8 |
| DOC-002 | DOCS | README low-level baseline example is invalid | 8 |
| DOC-003 | DOCS | Sphinx references removed Jones class names | 8 |
| DOC-004 | DOCS | README claims 15+ configs while two exist | 8 |
| DOC-005 | DOCS | README/backend documentation contradicts live backend behavior | 8 |
| DOC-006 | DOCS | `project.md` is stale and still describes RRIVis | 8 |
| DOC-007 | DOCS | `AGENTS.md` describes an absent Hugging Face app | 8 |
| DOC-008 | DOCS | No tracked CI and no real integration/performance suites | 8 |

## 6. Question-by-question findings and target behavior

### 6.1 REL-001 — Version inconsistency

#### Current state

Resolved. The following now agree on `0.2.0`:

- `pyproject.toml`;
- `pixi.toml`;
- `src/radiosim/__about__.py`;
- `docs/conf.py`.

#### Required ongoing control

Add a small release-metadata test or release script that reads all four values
and fails when they diverge. A future version bump should be one command or one
reviewable change set.

#### Acceptance criteria

- one automated check covers every authoritative version surface;
- documentation and CLI report the same release;
- the release checklist includes the version-consistency check.

### 6.2 REL-002 — Pixi v6 lockfile warning

#### Meaning

The warning meant that the dependency solution was current but stored in an
older Pixi lockfile schema. It was not an installation failure and did not mean
that dependencies were unresolved.

#### Current state

Resolved. `pixi.lock` is now v7 and records explicit platform virtual packages.
`pixi install` completes without the v6 warning.

#### Ongoing rule

Lockfile-format upgrades should be isolated or explicitly called out because
they can produce large structural diffs even when package intent is unchanged.

### 6.3 OBS-001 — Why observability is not a simulation product

`ObservabilityPlanner` builds a renderer-neutral `ObservabilityPlan` containing
tracks, footprint masks, contours, snapshots, and source metrics. It does not
calculate baseline-dependent visibilities.

`Simulator.plot_observability()` connects it to the public API, but that makes
it a sibling capability, not a child of the result writer. It can use config and
an optionally prepared sky model without consuming `Simulator.run()` output.

#### Target behavior

- architecture diagrams place observability beside the simulation pipeline;
- `plot_observability()` accepts a resolved context or clearly resolves the
  required subset itself;
- observability never implies that it is plotting calculated visibilities;
- documentation distinguishes visibility plots from observability plots.

#### Heterogeneous-array decision

For different per-antenna beams or diameters, a single array footprint is
ambiguous. Before wiring heterogeneous beams into observability, choose one of:

1. require a `reference_antenna`;
2. display a selected antenna and label it;
3. display an intersection/union envelope with explicit semantics;
4. display several antenna footprints.

Recommended first implementation: require or default a clearly reported
reference antenna, then add envelopes later.

### 6.4 INS-001 — Per-antenna diameters

#### Current behavior

`AntennaLayoutConfig` defines:

- `all_antenna_diameter`;
- `use_different_diameters`;
- `diameters`.

`Simulator.setup()` nevertheless assigns `all_antenna_diameter` to every loaded
antenna. This overwrites values read from supported antenna formats and ignores
the per-antenna map.

The downstream solver is already capable of using heterogeneous diameters:
point visibility builds a per-antenna diameter map, and the HEALPix path reads
each antenna's `diameter`. The missing work is resolution in setup.

#### Target configuration

Prefer a simpler pre-v1 contract:

```yaml
antenna_layout:
  default_diameter_m: 14.0
  diameters_by_antenna:
    "0": 12.0
    "1": 25.0
```

Remove `use_different_diameters`; the override map already expresses the
intent. If retaining current names temporarily is useful for the implementation
branch, remove them before finalizing the public API rather than creating a
long-lived deprecation shim.

#### Resolution precedence

Recommended precedence, highest first:

1. explicit per-antenna config override;
2. diameter supplied by the selected antenna-layout source;
3. explicitly enabled pyuvdata telescope diameter;
4. configured default diameter.

Every resolved diameter must be finite and positive. Unknown mapping keys and
unresolved antenna IDs must fail during setup.

#### Acceptance criteria

- heterogeneous diameters reach both solvers unchanged;
- file-provided diameters are not overwritten unless explicitly overridden;
- result metadata records the resolved diameter and its source per antenna;
- invalid, missing, or unknown mappings fail before sky loading;
- baseline metadata no longer encodes diameters as an opaque string.

#### Resolution (2026-07-20)

**DONE.** Tier 2 replaced the ignored legacy map with typed per-antenna overrides and
one complete canonical instrument. Resolution applies `override > source > configured
default`, records the winning source and the original source fact, rejects unknown,
duplicate, non-finite, non-positive, or unresolved values before runtime side effects,
and passes exact heterogeneous diameters to both solvers, results, plots, and writers.
No consumer invents a 14-metre fallback.

### 6.5 BEAM-001/002 — FITS, per-antenna, and mixed beams

#### Current behavior

The Pydantic schema validates a modern contract:

- `beam_mode: analytic | fits | mixed`;
- `per_antenna`;
- `beam_file`;
- `antenna_beam_map`;
- FITS normalization and interpolation settings.

The high-level simulator initializes `_beam_manager` to `None` and never
assigns it. Setup copies only analytic-beam fields into `_beam_config`.

The existing `BeamManager` uses a different contract:

- `use_beam_file`;
- `use_different_beams`;
- `beam_file_path`;
- `beam_files`;
- `beams_per_antenna`.

Consequently, validated modern beam fields never reach the FITS implementation.

#### Root cause

The configuration surface and the beam manager evolved independently. The
runtime contains the low-level FITS machinery, and the schema describes the
desired high-level behavior, but there is no modern resolution boundary between
them.

#### Target behavior

Refactor `BeamManager` to consume resolved modern beam assignments directly:

| Mode | Resolution |
|---|---|
| `analytic` | No FITS handler; every antenna uses analytic E-Jones |
| `fits`, shared | Load one FITS handler and assign it to every antenna |
| `fits`, per antenna | Resolve one FITS path per antenna |
| `mixed` | Resolve each antenna to either `analytic` or a FITS path |

Do not translate modern fields into the legacy keys as a permanent adapter.

#### Additional requirements

- normalize antenna mapping keys once;
- deduplicate identical FITS paths without changing assignments;
- validate file existence and coverage before simulation;
- propagate `beam_peak_normalize`, interpolation, ZA, and frequency-domain
  settings;
- fail loudly rather than silently falling back to analytic beams;
- use the same manager in point and HEALPix paths;
- use the same resolved beam definition in observability.

#### Acceptance criteria

- shared FITS, per-antenna FITS, and mixed modes have end-to-end tests;
- a known synthetic UVBeam gives predictable Jones values;
- both visibility paths agree for equivalent scalar/diagonal beams;
- missing antenna assignments and out-of-domain requests produce actionable
  errors;
- observability reports which antenna or envelope it displays.

### 6.6 INS-003 — Baseline-selection fields

#### Current behavior

`BaselineSelectionConfig` defines autocorrelation, cross-correlation, length,
tolerance, and angle controls. `generate_baselines()` always creates all
`ant1 <= ant2` combinations, including both autos and crosses. The high-level
simulator does not apply a selection stage.

The current CLI source itself lists advanced baseline filtering as unfinished
modular-API work.

#### Target design

Keep baseline generation and baseline selection separate:

```text
resolved antennas
    -> generate canonical baselines
    -> select baselines
    -> validate non-empty selection
    -> freeze selected baseline inventory
```

Create a focused function such as:

```python
select_baselines(
    baselines,
    *,
    include_autocorrelations,
    include_crosscorrelations,
    lengths_m,
    length_tolerance_m,
    azimuth_ranges_deg,
)
```

The exact public name may differ, but selection logic must not be embedded as a
large conditional block in `Simulator.setup()`.

#### Scientific details

- document ENU baseline azimuth convention;
- support ranges that wrap through 0 degrees;
- filter autocorrelations before computing angle because their direction is
  undefined;
- define whether a baseline matches either orientation or only the stored
  `ant1 -> ant2` direction;
- reject negative lengths and malformed angle ranges at schema validation;
- reject a final empty selection with the resolved criteria in the error.

#### Acceptance criteria

- exact baseline counts for auto-only, cross-only, and combined cases;
- tolerance-boundary tests;
- wrapped-angle and opposite-direction tests;
- selection metadata is saved with results;
- CLI and API produce the same selected baseline set.

#### Resolution (2026-07-20)

**DONE.** Tier 2 now generates one immutable, numeric-order baseline inventory with
`ant2 - ant1` vectors, then applies one typed selection pass shared by every consumer.
Correlation, target/range length, and axial-azimuth filters implement the documented
union/intersection algebra, tolerance boundaries, autocorrelation treatment, stable
empty failure, and complete stage-count provenance. YAML, mapping, model, parameter,
CLI, solver, result, plot, and writer paths use the same selected tuple.

### 6.7 POL-001 — Feed fields

#### Important terminology split

RadioSim currently uses “feed” for two different concepts:

1. `beams.feed_model` describes illumination of an aperture or reflector by a
   horn, waveguide, or dipole. This is wired into analytic beam computation.
2. top-level `feeds` describes receptor polarization/basis and per-antenna feed
   type. This is ignored by `Simulator`.

These concepts must not share ambiguous names in the final API.

#### Why top-level feeds cannot be wired mechanically

The solver assumes a linear X/Y basis and returns `XX`, `XY`, `YX`, and `YY`.
`ReceptorConfigJones` and `BasisTransformJones` are identity stubs. Merely
attaching `feed_type` strings to antennas would falsely imply correct circular
or heterogeneous polarization physics.

#### Target design

- rename beam illumination settings clearly, for example under
  `beams.illumination`;
- replace top-level `feeds` with a typed receptor/basis model;
- support at least `linear` and `circular` bases;
- model feed orientation separately from basis;
- define one output basis for heterogeneous arrays and transform each antenna
  into it;
- generate appropriate correlation labels and file-format polarization codes;
- implement C/H Jones matrices before claiming receptor support.

#### Acceptance criteria

- analytic linear-to-circular transform tests;
- Stokes-to-correlation energy-conservation tests;
- correct `XX/XY/YX/YY` and `RR/RL/LR/LL` labels;
- polarized-source cross-hand tests;
- Measurement Set and UVFITS polarization metadata round trips;
- unsupported heterogeneous combinations fail explicitly until implemented.

### 6.8 INS-002 — pyuvdata telescope flags

#### Current behavior

`TelescopeConfig` defines flags for known telescope, location, antennas, and
diameters. The high-level setup nevertheless requires an antenna file and reads
location directly from config. The flags are not consumed.

Reading antenna information from an MS or UVFITS file is not the same as
resolving `Telescope.from_known_telescopes()`.

#### Target design

Create an instrument resolver responsible for:

- loading a known pyuvdata telescope when requested;
- converting antenna positions into RadioSim's documented coordinate frame;
- merging explicit config, file, and pyuvdata sources according to one
  precedence table;
- resolving location, names, numbers, positions, diameters, and mount metadata;
- retaining provenance for every resolved field;
- detecting inconsistent antenna-number inventories across sources.

Recommended field precedence:

| Field | Highest precedence | Fallbacks |
|---|---|---|
| Location | Explicit config | pyuvdata, project default only if documented |
| Positions | Explicit selected layout source | pyuvdata when enabled |
| Diameter | Per-antenna override | layout, pyuvdata, default |
| Names/numbers | Selected position source | validated mapping |
| Mount/feed metadata | Explicit config | pyuvdata when enabled |

Avoid a master boolean plus several contradictory sub-flags. Prefer an explicit
source selection and per-field override model.

#### Resolution (2026-07-20)

**DONE.** Tier 2 removed the inert pyuvdata booleans and replaced them with an exact
`known_telescope` source contract, distinct from Measurement Set and UVFITS dataset
sources. The resolver loads the selected source through an offline-testable injected
boundary, preserves identity, location, positions, diameters, mount metadata, pyuvdata
version, and registry policy in provenance, disables both registry-update flags where
required, converts relative ECEF scientifically to canonical ENU, and rejects explicit
metadata mismatches. High-level Simulator setup uses this resolved source directly.

### 6.9 CFG-002 — Compute backend ignored by `Simulator.from_config()`

#### Current behavior

`Simulator.from_config()` extracts precision and then constructs `Simulator`
without passing `radiosim_config.compute.backend`. The constructor default
`backend="auto"` therefore wins.

The config-mode CLI separately resolves:

```text
explicit CLI backend unless it is "auto"
    -> otherwise config.compute.backend
```

The Python API and CLI can therefore run the same YAML on different backends.

#### Target behavior

Use `None` to mean “no explicit override” because `auto` is a legitimate
backend value:

```python
Simulator.from_config(path, *, backend: str | None = None)
```

Resolution order:

1. explicit API/CLI override;
2. `config.compute.backend`;
3. documented package default.

The resolved backend name and device must be included in result metadata.

#### Acceptance criteria

- `from_config()` honors `compute.backend`;
- explicit overrides win identically in CLI and Python;
- `auto` remains selectable as an actual strategy;
- unavailable explicit backends fail rather than silently falling back;
- backend-resolution tests do not depend on a physical GPU.

#### Resolved current behavior (2026-07-17)

`Simulator.from_config()` now accepts only typed `RadioSimConfig`, and all
document/API/CLI paths resolve `execution.backend` through the same frozen
override model and resolver. `None` means no override, explicit values win,
`auto` remains a real strategy, and NumPy remains the declared deterministic
default. The resolved strategy is retained in runtime/provenance. Explicitly
unavailable or precision-incompatible backends fail rather than falling back;
the backend factory also rejects any requested precision it cannot represent.

### 6.10 OUT-001 — What “output controls mostly honored” means

The current status is:

| Field | Current status | Required decision/fix |
|---|---|---|
| `simulation_data_dir` | Honored by config-mode CLI | Retain as CLI orchestration |
| `simulation_subdir` | Honored or auto-generated | Retain and test |
| `output_file_name` | Honored by `Simulator.save()` | Retain |
| `output_file_format` | HDF5/JSON/MS dispatched; UVFITS unsupported | Implement or reject UVFITS |
| `save_simulation_data` | Honored by CLI | Retain as CLI orchestration |
| `overwrite_output` | Partially honored | Unify folder and file policy |
| `skip_overwrite_confirmation` | Honored | Clarify noninteractive behavior |
| `prompt_for_output_suffix` | Ignored | Honor it or remove it |
| `plot_results` | Honored | Retain |
| `open_plots_in_browser` | Honored | Retain |
| `plotting_backend` | Passed to plotting | Type and validate choices |
| `save_log_data` | Honored | Retain |
| `angle_unit` | Ignored by high-level path | Thread through or remove |
| `skymodel_frequency` | Ignored | Thread through or remove |

The final design should separate scientific simulation configuration from CLI
workflow/output orchestration. Calling `Simulator.run()` from Python should not
implicitly perform CLI-style saving or prompting.

### 6.11 CFG-001 — Validation differences between entry points

There are currently three validation depths:

| Entry point | Current validation |
|---|---|
| Config-mode CLI | Pydantic construction plus explicit `config.validate()` preflight |
| `Simulator.from_config()` | Pydantic construction, but no explicit semantic preflight |
| `Simulator(config={...})` | Shallow dictionary copy; no full Pydantic validation |

`RadioSimConfig` also allows extra top-level fields, while nested models may
ignore extras under Pydantic defaults. This permits typos and unsupported fields
to survive differently depending on their location.

#### Why raw dictionaries exist

The raw path supports concise programmatic and test configurations. It is
convenient for constructing only the sections needed by a particular test, but
it makes behavior entry-point-dependent and moves failures into setup.

#### Target design

All configuration sources must pass through one boundary:

1. parse/coerce source-specific values;
2. validate types and local field constraints;
3. run semantic and cross-field validation;
4. resolve paths relative to the correct source;
5. resolve runtime defaults and explicit overrides;
6. produce immutable/frozen runtime configuration.

Before changing extras to `forbid`, inventory and type every legitimate live
field, including the explicit `frequencies_hz` array used by the programmatic
constructor.

#### Acceptance criteria

- equivalent YAML, mapping, and typed-model input resolve identically;
- validation errors occur before device detection, network checks, or sky
  loading;
- unsupported settings cannot silently become no-ops;
- partial test fixtures use explicit test builders instead of weakening the
  production constructor;
- path-resolution semantics are identical and documented.

#### Resolved current behavior (2026-07-17)

YAML, raw mapping, typed model, convenience parameters, root config mode,
`validate`, and the exposed `simulate` surface now use the same strict schema,
override, semantic/unsupported, path, and immutable-runtime resolution stages.
Direct `Simulator` construction accepts only `ResolvedSimulationConfig`; the
old raw/partial constructor path is absent. Invalid input fails before backend,
device, network, loader, output, plotting, browser, or workflow side effects.
Path bases and override origins remain explicit in immutable provenance.

### 6.12 CFG-003 — Unsupported fields silently becoming no-ops

#### Resolved current behavior (2026-07-17)

The complete strict input-model inventory covers 281 field occurrences. Every
accepted field is either resolved into supported scientific/workflow behavior
or classified as deferred. The operational deferred-feature matrix covers 34
non-default cases across the capable public entry points; all 34 reject with an
exact configuration error before side effects. Entry points such as
`radiosim simulate` omit deferred fields they cannot express. No unsupported
state is representable in `ResolvedSimulationConfig`, and no generic loader
option, raw mapping, compatibility-key, or partial-fixture escape hatch remains.

## 7. Additional confirmed gaps from the broader review

### 7.1 BEAM-003 — NSIDE beam advisor bug

The original defect was `Simulator.setup()` iterating an antenna dictionary as if
its keys were antenna objects with `.diameter`; a broad exception handler converted
that failure into `beam_fwhm_rad=None` and silently disabled advice. Tier 2 removed
the raw antenna dictionary, so the live code now reads canonical antennas, but the
issue is not closed: the advisor still uses one lowest-frequency/minimum-diameter
FWHM scalar, can choose the widest rather than the limiting beam-product feature, and
still suppresses failures broadly.

Fix this only after diameter resolution has a canonical representation. Do not
patch it with another dictionary/object special case that perpetuates mixed
antenna representations. Closure requires canonical beam assignments, every selected
baseline and exact frequency, the product-feature sampling derivation, defined autos,
typed derivation failure, no arbitrary antenna, no silent NSIDE mutation, and no broad
suppression.

### 7.2 RUN-001/002 — Worker controls

`Simulator.run(n_workers=...)` documents a worker count but never uses it.
Sky-model setup separately hard-codes `max_workers=8`.

These are two different concurrency concerns and should not share one ambiguous
parameter:

- sky-loader concurrency belongs to setup/data loading;
- visibility computation concurrency belongs to a solver execution policy.

Define them separately, for example through typed compute settings, and record
the resolved values in runtime metadata.

### 7.3 RUN-003 — Hybrid sky support

`SkyModel` can preserve point and HEALPix payloads simultaneously, but the
high-level `VisibilityConfig` accepts only `point_sources` or `healpix_map` and
`Simulator.run()` chooses one solver branch.

The correct hybrid implementation is not lossy conversion. It is:

```text
V_total = V_point + V_healpix
```

Both components must use the same resolved baselines, time grid, frequencies,
phase convention, polarization basis, backend policy, and result shape. Sky
provenance/disjointness rules must prevent accidental double counting.

### 7.4 RUN-004 — Backend reality

The backend abstraction is partially wired into point-source array operations,
but the orchestration remains dominated by host-side Python loops, Astropy
coordinate transforms, and host/device transfers. HEALPix code still contains
NumPy-specific operations. JAX/Numba selection therefore does not yet imply
end-to-end acceleration.

The README currently contains mutually inconsistent claims: some sections say
the solver is NumPy-only, while the live point path does dispatch selected array
operations. Documentation should say:

- backend array operations are partially integrated;
- correctness parity is more mature than performance integration;
- GPU acceleration is not yet demonstrated end to end;
- Numba must not be called a production JIT solver unless the solver actually
  uses compiled kernels.

### 7.5 SCI-001 — Jones identity stubs

The package exports 46 Jones classes. Only geometric phase K and primary beam E
currently provide substantive forward-model effects. Most other terms return
identity matrices.

This is not one defect. It is a collection of separate scientific features.
Public identity stubs are risky because a user can add a term to a chain and
observe no change without an error.

Until each term is implemented, choose one truthful policy:

- remove it from the public surface;
- raise `NotImplementedError` when used;
- place it behind an explicit experimental namespace.

Silently multiplying by identity is not an acceptable final behavior.

### 7.6 SCI-002 — Spherical-harmonic mode

The config accepts `calculation_type="spherical_harmonic"`, but setup raises
`NotImplementedError`. This is a roadmap solver, not a small conditional branch.

Until an m-mode solver exists, schema validation should reject the option or the
option should be removed. When implemented, it should be a separate simulator
strategy registered alongside direct RIME, with direct-sum agreement tests on
small tractable skies.

### 7.7 OUT-002 — Time-axis disagreement

The point solver uses floor-like `int(duration / step)`, the HEALPix solver uses
`ceil`, and `Simulator.save()` reconstructs a floor-like time axis. A duration
that is not evenly divisible by the cadence can therefore produce mismatched
result and file coordinates.

Create one authoritative time-grid function. Solvers and writers must consume
the resulting time array rather than independently reconstructing it.

The contract must define whether the observation endpoint is included. That
choice must be tested for divisible and non-divisible durations.

### 7.8 OUT-003/004/005/006 — Output correctness and safety

Current high-level HDF5 output:

- extracts only `I`, with `XX` fallback;
- drops `XY`, `YX`, `YY`, and basis metadata;
- forces `complex128` regardless of configured output precision;
- stringifies much of the metadata;
- stores baseline tuples in group names.

Current JSON output stores metadata, frequencies, and a baseline count, but no
visibility values.

The HDF5 reader uses `eval()` to parse baseline tuples. This is unsafe for
untrusted files and unnecessary even for trusted files.

UVFITS is present in `OutputConfig` but absent from `Simulator.save()`.

These problems should be fixed through a canonical result model and versioned
file schema, not by adding more conversion branches to `Simulator.save()`.

### 7.9 DOC-001 through DOC-008 — Documentation and project truth

Confirmed current drift includes:

- `examples/scripts/simple_simulation.py` references nonexistent
  `sim._sources`;
- it formats the memory-estimate dictionary as a float;
- it treats a per-baseline correlation dictionary as an array with `.shape`;
- README and Sphinx call `generate_baselines(antennas)` without its required
  beam metadata arguments;
- several Sphinx pages refer to nonexistent `GeometricDelayJones` rather than
  the live geometric class;
- README claims 15+ configs while `configs/` has two YAML files;
- `project.md` retains old RRIVis naming and stale claims;
- `AGENTS.md` describes `huggingface_space/`, which is absent;
- no `.github/` workflow is tracked;
- integration and performance directories contain no real tests.

Documentation should be corrected after each API tier, followed by a final
cross-repository truth sweep in Tier 8.

## 8. Tier dependency map

```mermaid
flowchart TD
    T0["Tier 0: Truth, safeguards, and baselines"]
    T1["Tier 1: Unified configuration contract"]
    T2["Tier 2: Instrument and baseline resolution"]
    T3["Tier 3: Beam and observability integration"]
    T4["Tier 4: Result, time, and output model"]
    T5["Tier 5: Receptor and polarization feeds"]
    T6["Tier 6: Hybrid runtime and backend completion"]
    T7["Tier 7: Advanced Jones and m-mode science"]
    T8["Tier 8: Documentation and release reconciliation"]

    T0 --> T1
    T1 --> T2
    T2 --> T3
    T3 --> T4
    T4 --> T5
    T5 --> T6
    T6 --> T7
    T7 --> T8

    T0 -. "CI protects every tier" .-> T8
    T1 -. "output config contract" .-> T4
    T2 -. "resolved antennas" .-> T5
    T3 -. "shared beam semantics" .-> T6
```

The linear order is deliberate. Some implementation work could technically be
parallelized, but merging it out of order would create temporary contracts and
duplicated migration work.

## 9. Tier 0 — Truth, safeguards, and baselines

### Objective

Establish a reliable baseline and prevent later tiers from adding more silent
drift.

### Already complete

- package/Pixi/Sphinx version alignment;
- Pixi lockfile v7 migration;
- creation of this plan.

### Implementation record — 2026-07-14

**Status:** Tier 0 is implemented and passes its local verification gate. The
new GitHub Actions workflow has been structurally validated and its commands
have been mirrored locally where the current macOS arm64 host permits. It has
not run remotely, so this record does not claim that GitHub Actions or the
Linux/macOS Intel jobs are green.

Implemented safeguards:

- added the contributor-facing pre-v1 API/configuration evolution policy;
- added one actionable release-metadata consistency test covering
  `pyproject.toml`, `pixi.toml`, `src/radiosim/__about__.py`, both Sphinx
  version fields, and the installed `radiosim.__version__`;
- added a v7 lock-format test and locked Pixi installs in CI;
- added Python 3.11 and 3.12 Pixi environments with lock solutions for
  `linux-64`, `osx-64`, and `osx-arm64`;
- added a six-combination non-slow CI matrix plus a Linux/Python 3.11 quality
  job for Ruff, release metadata, strict-Pyright debt enforcement, and Sphinx;
- added a checked-in strict-Pyright error ceiling and a typed helper that fails
  on an increase or an unreviewed Pyright-version change and refuses to raise
  the ceiling during an intentional update;
- added the Sphinx configuration-support matrix and linked it from the user
  guide toctree;
- added narrow, tested pre-setup guards for confirmed explicit silent no-ops;
- made the documented `make -C docs html` command invoke Sphinx through the
  checked-in Pixi environment.

#### Exact baseline and post-change results

| Command | Pre-change result | Post-change result |
|---|---|---|
| `pixi install` | exit 0 | exit 0; default Python 3.11 environment installed |
| `pixi run test -- --collect-only -q` | exit 0; 1,178 tests; 3.50 s | exit 0; 1,213 tests; 3.66 s |
| `pixi run test -- -m "not slow"` | exit 0; 1,177 passed, 1 skipped, 26 warnings; 132.60 s | exit 0; 1,212 passed, 1 skipped, 26 warnings; 138.27 s |
| `pixi run test` | exit 0; 1,177 passed, 1 skipped, 26 warnings; 134.31 s | exit 0; 1,212 passed, 1 skipped, 26 warnings; 136.97 s |
| `pixi run lint` | exit 0; all checks passed | exit 0; all checks passed |
| `pixi run check-format` | exit 0; 237 files already formatted | exit 0; 239 files already formatted |
| `pixi run typecheck` | exit 1; 4,600 errors, 0 warnings, 0 information; Pyright 1.1.408; 14.08 s | exit 0; strict ceiling satisfied at 4,600 <= 4,600; Pyright 1.1.408; 13.18 s |
| `make -C docs html` | exit 2; `sphinx-build` not found; Sphinx did not start | exit 0; build succeeded with 242 warnings; 26.67 s |

Additional verification completed:

- `pixi install --locked --environment py312` and `pixi lock --check` passed;
- the focused release/guard suite passed on both Python 3.11 and 3.12
  (39 passed in 6.00 s and 39 passed in 5.86 s, respectively);
- the release-metadata test passed directly (3 passed), including its
  actionable mismatch path without leaving a repository change;
- all new negative guard branches were exercised, the default/working analytic
  beam controls remained accepted, and aggregation of multiple unsupported
  fields was tested;
- the workflow YAML parsed structurally, the new Sphinx page is in the toctree,
  the rendered contributor policy and support matrix were inspected, and
  `git diff --check` passed.

#### Tier 0 disposition of unsupported-option candidates

| Candidate | Disposition |
|---|---|
| `visibility.calculation_type="spherical_harmonic"` | Guarded now; target Tier 7 |
| configured `UVFITS` output | Guarded now; direct `save(format="uvfits")` already fails clearly; target Tier 4 |
| FITS/mixed/per-antenna beam modes, maps, files, and FITS controls | Guarded now; target Tier 3 |
| heterogeneous-diameter switch or non-empty map | Guarded now; target Tier 2 |
| enabled `telescope.use_pyuvdata_*` flags | Guarded now; target Tier 2 |
| non-default top-level receptor/feed fields | Guarded now; target Tier 5 |
| alternative/cross-check simulator settings | Guarded now; target Tier 7 |
| non-default output angle or sky-model plot frequency | Guarded now; target Tier 4 |
| explicit `Simulator.run(n_workers=...)` | Guarded now; target Tier 6 |
| analytic `beams.feed_model`, `feed_computation`, and `feed_params` | Confirmed supported and left enabled |
| baseline-selection settings | Documented and deferred to Tier 2 because current defaults conflict with runtime behavior |
| `compute.backend` entry-point inconsistency | Documented and deferred to Tier 1 |
| `output.prompt_for_output_suffix` policy inconsistency | Documented and deferred to Tier 4 |

Optional GPU/accelerator behavior, optional JAX correctness, Measurement Set
round trips, and other optional scientific backends were not established by
this Tier 0 gate. No remote GitHub Actions result has been observed. Tier 1 and
all later implementation work remain open and unstarted.

### Work items

1. Add the pre-v1 breaking-change note to `docs/contributing.rst`.
2. Add a version-consistency test or release check.
3. Add a tracked CI workflow for supported Python/platform combinations.
4. Run and record the full test baseline.
5. Run type checking and record existing debt rather than requiring an
   unrealistic all-green conversion in the same tier.
6. Establish a “no new type errors” policy until the existing baseline is
   reduced.
7. Add a config-support matrix to developer documentation showing which fields
   are implemented, rejected, or roadmap.
8. Mark currently unsupported high-risk choices as errors where doing so does
   not depend on the Tier 1 redesign.

### Likely files

- `docs/contributing.rst`;
- `pyproject.toml`;
- `.github/workflows/ci.yml`;
- new focused tests under `tests/unit/`;
- `README.md` or a new developer-facing support-matrix document.

### Verification gate

```bash
pixi install
pixi run lint
pixi run check-format
pixi run test -- -m "not slow"
pixi run typecheck
make -C docs html
```

If type checking is not green, record the exact baseline and fail only on an
increase until a dedicated type-debt effort is approved.

### Suggested commits

- `docs: state pre-v1 API evolution policy`
- `test: enforce release metadata consistency`
- `ci: add baseline lint test and docs workflow`

### Exit criteria

- CI runs on every proposed change;
- version drift is automatically detectable;
- unsupported public settings are visibly classified;
- the test and type-check baselines are recorded.

## 10. Tier 1 — Unified configuration contract

### Objective

Ensure YAML, Python mappings, typed config objects, and CLI overrides resolve to
the same validated runtime configuration.

### Design work before coding

1. Inventory every key accessed through `config.get(...)` in source.
2. Compare that inventory with Pydantic fields.
3. Classify each field as scientific configuration, execution policy, output
   workflow, or deprecated/unsupported.
4. Decide the immutable resolved models required by setup.
5. Define override precedence and path-resolution rules.

### Implementation work

1. Add typed fields for legitimate live extras such as `frequencies_hz`.
2. Introduce one loader/normalizer used by:
   - `Simulator.from_config()`;
   - `Simulator(config=...)` or its clean replacement;
   - config-mode CLI;
   - `radiosim validate`.
3. Fold semantic preflight into a normal validation/resolution API instead of a
   separate method that only the CLI remembers to call.
4. Make `Simulator` retain typed or frozen configuration rather than a mutable
   raw dictionary.
5. Use `None` as the explicit-override sentinel for backend and precision.
6. Resolve `compute.backend` consistently.
7. Reject unknown/unsupported fields after the live-field inventory is typed.
8. Replace production partial-config behavior with explicit builders for tests
   and concise programmatic construction.
9. Preserve correct YAML-relative path resolution for every path field, not
   only the antenna file.

### Likely files

- `src/radiosim/io/config.py`;
- `src/radiosim/api/simulator.py`;
- `src/radiosim/cli/main.py`;
- `src/radiosim/utils/frequency.py`;
- `tests/unit/test_io/test_config.py`;
- `tests/unit/test_simulator/test_api.py`;
- new CLI tests under `tests/unit/test_cli/`.

### Required tests

- YAML/mapping/model equivalence;
- API/CLI backend precedence;
- explicit frequency arrays;
- unknown-field rejection;
- unsupported-field rejection;
- cross-field errors before side effects;
- relative paths from nested config locations;
- immutable resolved config behavior.

### Verification gate

```bash
pixi run test -- tests/unit/test_io/test_config.py
pixi run test -- tests/unit/test_simulator/test_api.py
pixi run test -- tests/unit/test_cli/
pixi run lint
pixi run check-format
pixi run test -- -m "not slow"
```

### Suggested commits

- `refactor(config): unify YAML mapping and API validation`
- `fix(api): honor configured backend resolution`
- `test(config): enforce CLI and API parity`

### Exit criteria

- one config has one resolved meaning regardless of entry point;
- `Simulator.from_config()` honors backend and semantic validation;
- raw typos cannot survive silently;
- no runtime stage mutates configuration to create precedence.

## 11. Tier 2 — Instrument and baseline resolution

### Status

**Complete and independently accepted after correction on 2026-07-20.** INS-001,
INS-002, and INS-003 are closed. Tier 3 implementation was not started. The next
authorized task is the separate Tier 3 design gate for beam and observability
integration.

### Objective

Create one resolved instrument inventory before sky and solver setup.

### Implementation work

1. Define typed resolved antenna/telescope models or an equivalent frozen
   internal representation.
2. Normalize antenna IDs and coordinate-frame metadata.
3. Implement known-telescope loading through pyuvdata.
4. Merge explicit config, selected layout source, and pyuvdata metadata using
   documented precedence.
5. Resolve per-antenna diameters without overwriting loaded values.
6. Validate complete, finite, positive diameters.
7. Generate canonical baselines from resolved antennas.
8. Implement a separate baseline-selection function.
9. Replace opaque baseline metadata strings such as `D1D2` with structured
   values or derive them from antenna references.
10. Store instrument and selection provenance in results.

### Likely files

- `src/radiosim/core/antenna.py`;
- `src/radiosim/core/baseline.py`;
- new focused instrument-resolution module under `core/` or `io/`;
- `src/radiosim/api/simulator.py`;
- `src/radiosim/io/config.py`;
- antenna, baseline, API, and config tests.

### Required tests

- uniform and heterogeneous diameter resolution;
- layout-file diameter preservation;
- explicit override precedence;
- pyuvdata location/position/diameter selection;
- source inventory mismatch errors;
- auto/cross baseline counts;
- length tolerance boundaries;
- angle wrap and orientation behavior;
- empty-selection error;
- point/HEALPix receipt of identical resolved antennas.

### Verification gate

```bash
pixi run test -- tests/unit/test_core/ -k "antenna or baseline"
pixi run test -- tests/unit/test_io/ -k "config or antenna"
pixi run test -- tests/unit/test_simulator/
pixi run test -- -m "not slow"
```

### Suggested commits

- `refactor(instrument): resolve telescope and antenna metadata once`
- `fix(instrument): apply heterogeneous antenna diameters`
- `feat(baseline): honor configured baseline selection`

### Exit criteria

- setup has one immutable antenna inventory;
- all enabled metadata sources follow tested precedence;
- every configured baseline-selection field affects the selected set;
- the solver no longer receives placeholder beam metadata from baseline
  generation.

## 12. Tier 3 — Beam and observability integration

### Objective

Connect modern analytic/FITS beam configuration to both solvers and ensure
observability represents the same instrument semantics.

### Implementation work

1. Replace `BeamManager`'s legacy config contract with resolved modern beam
   assignments and one validated `BeamSystem`.
2. Implement shared FITS, per-antenna FITS, and mixed modes.
3. Deduplicate handler instances for repeated paths.
4. Validate antenna coverage and beam domain before simulation.
5. Thread one `BeamSystem` through point and HEALPix solvers.
6. Remove silent analytic fallback on FITS initialization failures.
7. Fix the NSIDE advisor using the canonical antenna representation.
8. Make observability consume the resolved beam model.
9. Implement the selected reference-antenna rule: deterministic minimum-number
   default only after fingerprint equivalence, otherwise an explicit Tier 2 reference.
10. Ensure analytic per-antenna diameter support remains intact.

### Likely files

- `src/radiosim/core/jones/beam/fits/handler.py`;
- `src/radiosim/core/jones/beam/fits/__init__.py`;
- `src/radiosim/core/jones/beam/analytic/`;
- `src/radiosim/core/visibility.py`;
- `src/radiosim/core/visibility_healpix.py`;
- `src/radiosim/api/simulator.py`;
- `src/radiosim/core/observability/`;
- beam, Jones, visibility, observability, config, and API tests.

### Required tests

- shared FITS assignment;
- unique and repeated per-antenna FITS assignment;
- mixed analytic/FITS assignment;
- missing/unknown antenna mapping;
- FITS frequency/ZA-domain errors;
- peak-normalization and interpolation propagation;
- point/HEALPix beam parity;
- observability reference-antenna labeling;
- advisor uses the minimum selected-baseline product-feature scale over every exact
  observation frequency, including defined auto-only selection behavior.

### Verification gate

```bash
pixi run test -- tests/unit/test_jones/
pixi run test -- tests/unit/test_core/ -k "beam or visibility"
pixi run test -- tests/unit/test_observability/
pixi run test -- tests/unit/test_visualization/
pixi run test -- -m "not slow"
```

### Suggested commits

- `refactor(beam): replace legacy BeamManager config contract`
- `feat(beam): wire shared per-antenna and mixed FITS modes`
- `fix(observability): use resolved beam semantics`

### Exit criteria

- every valid beam mode changes the actual forward model as documented;
- FITS errors cannot become silent analytic simulations;
- observability and simulation share the same beam source;
- heterogeneous beam display semantics are explicit.

## 13. Tier 4 — Result, time, and output model

### Objective

Create one canonical result representation and make every writer preserve its
scientific content safely.

### Target result contract

The exact implementation may be a frozen dataclass, Pydantic model, or focused
container, but it must include:

- complex visibility data with named dimensions;
- baseline antenna IDs;
- time coordinates;
- frequency coordinates;
- correlation/feed-basis coordinates;
- antennas and location;
- phase center;
- precision/dtype;
- backend and simulator metadata;
- resolved config and provenance;
- timing/performance metadata.

Prefer a canonical dense visibility shape such as:

```text
(baseline, time, frequency, receptor_p, receptor_q)
```

Named correlation views may be exposed without making a nested dictionary the
storage format.

### Implementation work

1. Define one authoritative observation time-grid function.
2. Pass the resolved time axis into both solvers.
3. Return `SimulationResult` rather than a loosely structured dictionary.
4. Refactor plots and writers to consume the result model.
5. Define a versioned HDF5 schema.
6. Preserve all correlations and configured dtype.
7. Store structured baseline IDs and structured/JSON metadata.
8. Remove `eval()` completely.
9. Decide whether JSON is a full data format or rename it to metadata summary.
10. Implement UVFITS using pyuvdata after the result basis is explicit, or
    remove it from public config until that implementation lands.
11. Unify overwrite, suffix, and noninteractive output policy.
12. Honor or remove `angle_unit` and `skymodel_frequency`.

### Likely files

- new result model under `src/radiosim/core/` or `api/`;
- `src/radiosim/api/simulator.py`;
- `src/radiosim/core/visibility.py`;
- `src/radiosim/core/visibility_healpix.py`;
- `src/radiosim/io/writers.py`;
- `src/radiosim/io/measurement_set.py`;
- visualization modules;
- config and CLI output orchestration;
- new round-trip integration tests.

### Required tests

- divisible and non-divisible time-grid cases;
- point/HEALPix identical coordinates;
- HDF5 round trip for every correlation;
- float/complex precision preservation;
- structured metadata round trip;
- safe rejection of malformed baseline metadata;
- JSON contract test;
- MS round trip;
- UVFITS round trip when implemented;
- overwrite/suffix/noninteractive policy matrix.

### Verification gate

```bash
pixi run test -- tests/unit/test_io/
pixi run test -- tests/unit/test_simulator/
pixi run test -- tests/integration/ -m integration
pixi run test -- -m "not slow"
```

### Suggested commits

- `refactor(result): introduce canonical simulation result model`
- `fix(time): use one observation grid across solvers and writers`
- `refactor(io): add safe versioned visibility serialization`
- `feat(io): add UVFITS round-trip support`

### Exit criteria

- writers do not reconstruct scientific coordinates;
- no correlation or precision is silently discarded;
- no file reader evaluates input as Python code;
- file round trips preserve the result contract;
- every output config field is implemented or absent.

## 14. Tier 5 — Receptor and polarization feeds

### Objective

Implement physically meaningful receptor bases and eliminate ambiguous feed
configuration.

### Design decisions before coding

1. Define the sky polarization basis used internally.
2. Define supported receptor bases.
3. Define feed-angle convention and frame.
4. Define the common output basis for heterogeneous arrays.
5. Define correlation labels and file-format codes.
6. Separate aperture illumination feeds from receiving receptors in schema and
   documentation.

### Implementation work

1. Replace/rename top-level `FeedsConfig` with a typed receptor model.
2. Rename analytic-beam feed settings to illumination terminology.
3. Implement `ReceptorConfigJones` and `BasisTransformJones`.
4. Thread per-antenna receptor definitions into the Jones chain.
5. Generate correlation coordinates from the resolved output basis.
6. Update HDF5, MS, UVFITS, and plots.
7. Reject unsupported mixed-basis cases until their transform is implemented.

### Required tests

- identity for linear-to-linear;
- analytic linear/circular transforms;
- transform inverse/round trip;
- unpolarized energy conservation;
- fully Q/U/V-polarized reference cases;
- correct linear and circular correlation labels;
- heterogeneous antenna transforms into one output basis;
- file-format polarization metadata round trips.

### Verification gate

```bash
pixi run test -- tests/unit/test_jones/ -k "receptor or basis or polarization"
pixi run test -- tests/unit/test_core/ -k polarization
pixi run test -- tests/unit/test_io/
pixi run test -- tests/integration/ -m integration
pixi run test -- -m "not slow"
```

### Suggested commits

- `refactor(config): separate illumination and receptor models`
- `feat(jones): implement receptor basis transforms`
- `feat(result): support linear and circular correlations`

### Exit criteria

- top-level receptor configuration changes calculated correlations;
- basis labels are scientifically correct and serialized;
- no receptor option silently returns identity.

## 15. Tier 6 — Hybrid runtime and backend completion

### Objective

Expose core sky capabilities through the high-level API and make compute-policy
controls meaningful.

### Implementation work

1. Add high-level hybrid representation without lossy conversion.
2. Compute point and HEALPix visibility components on the same coordinates.
3. Sum components through the canonical result model.
4. Preserve component-level timing and provenance.
5. Split sky-loader worker policy from solver worker policy.
6. Remove hard-coded loader worker count.
7. Make solver worker settings effective or remove them.
8. Complete backend parity in the HEALPix path.
9. Reduce host/device transfers and isolate Astropy preprocessing.
10. Decide whether Numba gets real compiled kernels or a less misleading name.
11. Add end-to-end benchmarks before making acceleration claims.

### Required tests

- `V_hybrid == V_point + V_healpix` within precision tolerance;
- no double counting for disjoint and explicitly assumed-disjoint models;
- point/HEALPix/hybrid coordinate identity;
- NumPy/JAX parity for representative point and HEALPix workloads;
- loader-worker policy tests;
- solver-worker behavior tests;
- offline/network loader behavior under worker execution;
- backend error and explicit fallback semantics.

### Performance methodology

Every benchmark record must include:

- hardware and accelerator;
- backend and version;
- precision;
- antenna, baseline, source/pixel, time, and frequency counts;
- setup versus steady-state timing;
- compilation time;
- host/device transfer time;
- peak memory;
- correctness tolerance against NumPy.

### Verification gate

```bash
pixi run test -- tests/unit/test_backends/
pixi run test -- tests/unit/test_core/ -k "backend or hybrid or visibility"
pixi run test -- tests/integration/ -m integration
pixi run test -- tests/performance/ -m performance
pixi run test -- -m "not slow"
```

GPU claims require a real accelerator run; CPU-only collection or skipped GPU
tests are not sufficient evidence.

### Suggested commits

- `feat(simulator): preserve and simulate hybrid sky models`
- `refactor(compute): separate loader and solver concurrency`
- `perf(jax): reduce host device transfers in visibility solvers`
- `perf: add reproducible backend benchmarks`

### Exit criteria

- hybrid sky is a first-class high-level mode;
- every worker setting has observable tested behavior;
- backend documentation matches measured execution;
- acceleration claims have reproducible evidence.

## 16. Tier 7 — Advanced Jones and m-mode science

### Objective

Turn the scientific framework into implemented effects without treating 44+
independent models as one undifferentiated coding task.

### Workstream A — Calibration/receptor terms

- C/H receptor and basis transforms, started in Tier 5;
- G electronic gains;
- B bandpass;
- D polarization leakage;
- cross-hand phase and delay.

### Workstream B — Propagation terms

- ionospheric dispersive phase;
- ionospheric/telescope Faraday rotation;
- tropospheric delay and opacity;
- parallactic and field rotation.

### Workstream C — Wide-field and antenna terms

- W/non-coplanar effects where appropriate to the forward model;
- element beams;
- array factors and mutual coupling;
- differential beam residuals;
- cable reflections and electronic delays.

### Workstream D — Baseline-dependent effects

- multiplicative closure errors;
- time and bandwidth smearing;
- enforce the distinction between Jones matrix-chain terms and
  baseline-dependent Hadamard terms.

### Workstream E — Spherical-harmonic/m-mode simulator

- define the supported observing regime;
- define sky and beam harmonic representations;
- implement a separate registered simulator strategy;
- compare against direct sum on small skies;
- document accuracy, truncation, and performance boundaries.

### Rules for every Jones implementation

1. cite the adopted convention and scientific reference;
2. define units, axes, and sign conventions;
3. add analytic invariants;
4. add backend parity where supported;
5. add a test proving that a nonzero configured effect changes visibility;
6. update public status and remove the stub warning only for that term;
7. do not return identity for unsupported parameter combinations.

### Cross-implementation validation

Use suitable reference implementations where contracts overlap, such as
pyuvsim, matvis, RASCIL, or another scientifically appropriate code. Cross-check
results, not just API shapes.

### Suggested commit pattern

- `feat(jones): implement <specific term>`
- `test(jones): validate <term> against <reference>`
- `feat(simulator): add spherical harmonic strategy`

### Exit criteria

- every exported Jones class is implemented, explicitly experimental, or
  unavailable with an error;
- no public term silently multiplies by identity;
- m-mode is either implemented and tested or absent from accepted config;
- advanced beam TODOs have explicit scientific scope and verification.

## 17. Tier 8 — Documentation and release reconciliation

### Objective

Make every public statement, example, config, and project artifact match the
post-remediation implementation.

### Implementation work

1. Rewrite `examples/scripts/simple_simulation.py` against public APIs only.
2. Execute the example in CI.
3. Fix README and Sphinx low-level baseline examples.
4. Replace removed Jones class names.
5. Update backend status from measured Tier 6 results.
6. Replace the 15+ config claim with the actual curated config set.
7. Decide whether to update, replace, or delete stale `project.md`.
8. Decide whether `huggingface_space/` is restored as an intentionally
   maintained app; otherwise remove it from `AGENTS.md` and public docs.
9. Add real integration tests covering CLI-to-output workflows.
10. Add real performance tests using the Tier 6 methodology.
11. Build Sphinx with warnings treated as errors.
12. Execute or validate every documentation code example.
13. Update changelog and migration guide for breaking config/API changes.
14. Perform a final repository-wide search for old RRIVis naming, nonexistent
    symbols, unsupported claims, and stale version/config counts.

### Verification gate

```bash
pixi run radiosim --version
pixi run radiosim validate configs/config.yaml
pixi run python examples/scripts/simple_simulation.py --help
pixi run test
pixi run lint
pixi run check-format
pixi run typecheck
make -C docs clean html SPHINXOPTS="-W --keep-going"
```

Execute notebook validation using the repository's chosen notebook command once
the example data/network policy is defined.

### Suggested commits

- `fix(examples): update simulation walkthroughs to public API`
- `docs: reconcile config backend beam and Jones behavior`
- `test(integration): cover CLI simulation and output round trips`
- `test(performance): add reproducible solver benchmarks`

### Exit criteria

- every example executes;
- documentation builds with warnings as errors;
- README contains no unsupported claims;
- repository structure descriptions match the live tree;
- CI covers unit, integration, docs, and appropriate optional paths;
- the release notes disclose breaking changes and implemented capabilities.

## 18. Cross-tier sequencing rules

1. Do not wire individual config fields before Tier 1 defines the canonical
   configuration boundary.
2. Do not implement beams before Tier 2 supplies canonical antenna IDs and
   diameters.
3. Do not implement receptor bases before Tier 4 supplies an explicit result
   basis and correlation axis.
4. Do not implement hybrid addition before point and HEALPix results share one
   result/time contract.
5. Do not optimize JAX before parity tests identify the correct reference
   behavior.
6. Do not advertise Jones terms merely because their classes exist.
7. Do not update final documentation early and then let later tiers invalidate
   it; update focused docs per tier and do the full sweep in Tier 8.
8. Do not combine research-heavy Tier 7 work with foundational config/output
   refactors.
9. Do not restore history-rewritten files unless separately and explicitly
   requested.

## 19. Implementation workflow for each tier

Every tier follows the same lifecycle:

### 19.1 Audit and contract

- re-read the live files in scope;
- verify git status and preserve unrelated user changes;
- write the target contract and precedence rules;
- identify breaking changes;
- enumerate exact affected tests and docs.

### 19.2 Characterization

- add tests that capture correct existing behavior;
- add failing regression tests for the confirmed defect;
- avoid pinning accidental implementation details.

### 19.3 Implementation

- make the smallest coherent architectural change for the tier;
- remove replaced paths rather than keeping parallel old/new execution;
- avoid catch-all fallbacks that hide configuration errors;
- record provenance and resolved settings.

### 19.4 Focused verification

- run the exact unit suites for changed modules;
- run lint and format checks;
- run the non-slow suite;
- run optional/GPU/integration tests only when their prerequisites exist and
  report skipped coverage honestly.

### 19.5 Separate review pass

Review after the first green implementation for:

- ignored config fields;
- inconsistent CLI/API behavior;
- silent fallback;
- dtype or correlation loss;
- mutable shared state;
- unsafe file parsing;
- stale examples/docs;
- incomplete provenance;
- missing negative tests.

### 19.6 Handoff

For each tier, report:

- files changed;
- behavior changed;
- breaking changes;
- commands run and results;
- optional paths not tested;
- branch/commit state;
- intentionally deferred next-tier work.

## 20. Test strategy

### 20.1 Fast development loop

```bash
pixi run lint
pixi run check-format
pixi run test -- tests/unit/<affected-area>/
```

### 20.2 Pre-merge functional gate

```bash
pixi run test -- -m "not slow"
pixi run lint
pixi run check-format
```

### 20.3 Full release gate

```bash
pixi run test
pixi run lint
pixi run check-format
pixi run typecheck
make -C docs clean html SPHINXOPTS="-W --keep-going"
```

### 20.4 Required test categories by the end of the plan

- schema and semantic config validation;
- CLI/API resolution parity;
- instrument-source precedence;
- heterogeneous antennas and beams;
- baseline-selection geometry;
- point, HEALPix, and hybrid visibility correctness;
- polarization and basis transforms;
- backend parity;
- safe result serialization and round trips;
- CLI integration workflows;
- documented performance workloads;
- executable examples and docs.

## 21. Global definition of done

The remediation plan is complete only when all of the following are true:

### Configuration and API

- every accepted config field has tested runtime behavior;
- unsupported features are rejected before side effects;
- YAML, CLI, mapping, and typed API inputs resolve identically;
- backend and precision precedence are explicit;
- pre-v1 breaking changes are documented.

### Instrument and beams

- antenna metadata has one canonical resolved representation;
- per-antenna diameters work end to end;
- pyuvdata flags have defined behavior;
- baseline selection is fully honored;
- shared, per-antenna, and mixed beams affect both solvers;
- observability uses explicit compatible beam semantics.

### Results and I/O

- one time grid and one result model are used throughout;
- no correlation, dtype, or coordinate is silently lost;
- HDF5/MS/UVFITS round trips are tested as supported;
- JSON's contract is truthful;
- no reader uses `eval()` or equivalent unsafe parsing;
- every output control is honored or removed.

### Scientific runtime

- hybrid sky is supported without lossy coercion;
- worker settings have tested effects;
- backend documentation reflects measured behavior;
- public Jones terms are implemented or explicitly unavailable;
- spherical-harmonic mode is implemented or absent from accepted config.

### Documentation and engineering

- examples execute using public APIs;
- Sphinx builds with warnings as errors;
- README matches the live config count, paths, APIs, and feature status;
- CI is tracked and green;
- integration and performance suites contain meaningful tests;
- absent products/apps are not described as present;
- removed history files remain removed unless explicitly restored.

## 22. Recommended next action

Tier 0 is locally complete in the current working tree: the focused baseline,
CI workflow, pre-v1 contributor policy, support matrix, and type-debt gate are
present. Hosted CI has not been observed from this uncommitted local state, so
its remote result remains unverified.

The dedicated Tier 1 design and migration gate is documented in
[`Tier1ConfigPlan.md`](Tier1ConfigPlan.md), and its selected architecture is
accepted. Tier 1A through Tier 1D are accepted. The mandatory Tier 1D
acceptance review on 2026-07-15 reconfirmed the three public I/O boundaries,
obsolete-path removal, shared copy-owning normalization, typed YAML parse
failures, narrow path-check skipping, side-effect isolation,
later-slice-only xfail ownership, and clean whitespace. Its live seven-suite
gate reported 168 passed and 5 Tier 1E-owned xfailed tests, with no blocking
defect.

Tier 1E and Tier 1F are accepted after their independent 2026-07-16 reviews.
Config mode uses one `load_config` call with frozen tri-state
scientific/workflow overrides, constructs `Simulator` only from
`bundle.runtime`, and runs workflow separately. The root backend default is
`None`, explicit `auto` is a real override, and paired `--offline/--online`
preserves the document when omitted. Antenna and output overrides use the
captured invocation directory without input or bundle mutation.

Config mode and `radiosim validate` share the complete typed configuration
error renderer. Validate prints resolved source/base, backend, precision,
frequency, and path summaries without constructing Simulator or crossing
backend/device/network/loader/output/browser boundaries. Simulate requires
explicit location and start time, preserves the exact ordered nonuniform-Hz
frequency sequence through typed `from_parameters`, and passes output policy
only to explicit save/workflow calls. The migrated `configs/config.yaml` is the
only Tier 1H-owned sample touched because it is the Tier 1F smoke gate.

All ten Tier 1F-only and both shared Tier 1E/Tier 1F strict xfails are ordinary
passing regressions; repository-wide xfail and XPASS counts are zero. Final
focused Python 3.11 and 3.12 boundaries each report 239 passed. Collection is
1,413 tests; the non-slow and full suites each report 1,412 passed, 1 skipped,
0 xfailed, 0 XPASS, and 26 warnings. Ruff, formatting, and Git whitespace
validation pass. Strict Pyright passes at 4,471 under the unchanged 4,600
ceiling. Real root/simulate help and sample validation pass. Sphinx HTML passes;
the required incremental build reports 64 warnings, while a clean comparison
reports 266, 104 above the recorded 162-warning Tier 1E count. That broad
documentation debt remains assigned to Tier 1H.

The independent Tier 1F review reran the 44-test CLI boundary successfully,
reconfirmed the real sample-validation smoke, found zero repository xfails or
XPASS, and found no blocking Tier 1F defect. `git diff --check` passed; `main`,
HEAD, and `origin/main` remained aligned at
`73ae7a3d2f089d3523463f15f9b6ff569aec068d`; and nothing was staged, committed,
or pushed.

Tier 1G is accepted after independent review and a live 2026-07-17 drift
check. The public dictionary frequency parser, generic
dictionary validation fallback, every active `obs_frequency_config`
alternative, the raw `"frequencies_hz"` escape hatch, and the unused
`RadioSimConfig.generate_output_subdir()` helper are removed. Lower-level sky
combine, regrid, materialization, diffuse/PySM, PyRadioSky, skyh5, and synthetic
paths now consume copied, validated, strictly ascending explicit Hz arrays.
Nonuniform and one-channel arrays remain exact, and no configuration frequency
path sorts, refits, or uses `np.linspace`.

The separate Tier 1G review fixed a writable NumPy buffer retained by frozen
`PrepareSkyOptions` and two import-order issues; no compatibility shim, hidden
fallback, stale active export, CLI/workflow change, Tier 1H work, or Tier 2+
behavior was found. Final Python 3.11 and 3.12 focused boundaries each report
425 passed. The broad sky boundary reports 821 passed. Collection is 1,427;
the non-slow and full suites each report 1,426 passed, 1 skipped, zero xfailed,
zero XPASS, and 26 warnings. Ruff, formatting, Git whitespace, public-removal
smoke, root/simulate help, and real sample validation pass. Strict Pyright
passes at 4,460 under the unchanged 4,600 ceiling. A clean Sphinx build passes
with 265 warnings, one fewer than the Tier 1F clean baseline.

The independent acceptance reconfirmed the removed module and public surfaces,
the rejection-test-only legacy-name occurrences, the passing 122-test focused
gate, the public-removal import smoke, zero xfail/XPASS debt, and clean Git
whitespace. The live pre-Tier 1H drift check then passed 109 focused
cleanup/frequency/sky-support tests plus the public-removal smoke and found no
blocking regression.

Keep `CFG-001`, `CFG-002`, and `CFG-003` open. Hosted CI remains unobserved.
Tier 1 final acceptance has not occurred.

Tier 1H is independently accepted. The current README, Sphinx configuration
and API guides, migration guide, two shipped configs, complete antenna example,
primary Python script, and basic notebook now describe or exercise the strict
Tier 1 configuration architecture. They distinguish input models from resolved
runtime bundles, keep CLI workflow state separate, document discriminated
frequency modes and source-aware paths, state override/backend/precision
precedence, and reject rather than advertise later-tier FITS/per-antenna beams,
heterogeneous diameters, baseline subsets, feeds, pyuvdata flags, UVFITS, and
later simulator modes. Observability is a Simulator helper, and backend
selection is not claimed as proof of complete GPU execution.

The realistic foreground sample preserves its HERA-like Haslam plus GLEAM
intent within the supported analytic-beam/all-baseline contract. The antenna
example is now a complete strict RadioSim document. All three YAML documents
pass real CLI validation. The deterministic built-in script passes help and an
offline/no-plot temporary-directory run without output, and the cleared basic
notebook executes to a temporary artifact with the public API.

Tier 1H added `tests/unit/test_tier1h_documentation.py` and a typed-execution
regression in `tests/unit/test_simulator/test_api.py`. That regression exposed
and fixed one narrow Tier 1 defect: `Simulator.from_parameters` now serializes
typed input sections with `exclude_unset=True`, so a selected precision preset
does not conflict with unset default custom leaves. The default config generator
now emits explicit Tier 1 execution/workflow sections. No compatibility shim or
Tier 2 behavior was added.

After the separate review, the Tier 1H parity module reports 26 passed, the
Tier 1H plus Simulator review boundary 62 passed, and the complete focused
configuration/API/CLI/runtime/frequency boundary 324 passed. Ruff and format
checks pass. Strict Pyright remains at 4,460 diagnostics under the unchanged
4,600 ceiling. Non-slow and full suites each report 1,453 passed, 1 skipped,
zero xfailed, zero XPASS, and 26 warnings. A clean Sphinx build succeeds with
49 warnings, reducing the historical 265-warning baseline by 216; remaining
diagnostics are pre-existing lower-level docstring, historical
highlighting/toctree, and theme-option warnings. `git diff --check` passes.

Tier 1H changed `README.md`, both files under `configs/`, the antenna example
and README, the active configuration/backend/beam/sky/Jones/install/API Sphinx
pages, the migration and historical-status pages, the examples README/script/
notebook, `src/radiosim/io/config.py`, `src/radiosim/api/simulator.py`, the two
focused test modules, and the Tier 1 status records. It did not modify CI,
Pixi/dependency files, `pixi.lock`, the Pyright baseline, generated docs, or
historical design records. No optional GPU, scientific-network, or remote-CI
run was performed.

`CFG-001`, `CFG-002`, and `CFG-003` remain open. The next task is the separate
Tier 1 final-acceptance gate, not Tier 2. Tier 2 has not started, and nothing
was staged, committed, or pushed.

### 2026-07-17 Tier 1H independent acceptance

The independent review accepted Tier 1H after inspecting the implementation,
tests, active and historical documentation, examples, notebook, shipped YAML,
real CLI behavior, rendered backend autodoc, and clean documentation build. It
found and corrected five narrow truth-surface issue groups: blanket package and
CLI GPU/full-polarization claims, an unverified ultra-precision speed
multiplier, a strict-schema error in the documented `gsm2016` alias example,
and active backend autodoc that advertised unsupported Numba CUDA execution,
universal JAX device support, and unverified speedups. Four regressions were
added first and failed against the observed behavior before the corrections.
The production changes are limited to `src/radiosim/__init__.py`,
`src/radiosim/__about__.py`, `src/radiosim/core/precision.py`,
`src/radiosim/backends/{__init__,base,jax_backend,numba_backend}.py`, and
`docs/user_guide/sky_models.rst`; coverage is in
`tests/unit/test_tier1h_documentation.py`.

Final evidence is 30 Tier 1H parity passes, 66 Tier 1H plus Simulator API
passes, and 328 complete focused-boundary passes. Both non-slow and full suites
report 1,457 passed, 1 skipped, 26 warnings, zero xfailed, and zero XPASS. Ruff
lint and formatting pass. Strict Pyright remains at 4,460 diagnostics under the
unchanged 4,600 ceiling. All three shipped YAML documents validate through the
real CLI; root/validate/simulate/example help passes; the temporary offline
example writes no output; and the five-cell notebook executes through an
isolated current-Pixi kernelspec without modifying its source.

The clean Sphinx build succeeds with 49 warnings, independently classified as
39 lower-level docstring/docutils diagnostics, 3 antenna footnotes, 3
historical/not-in-toctree diagnostics, 3 historical HERA highlighting
diagnostics, and 1 pre-existing theme-option warning. No warning indicates a
Tier 1H truth error or duplicate Tier 1 configuration/API registration.

Optional GPU, external scientific-network, and remote-CI verification were not
performed; remote CI remains unobserved. `CFG-001`, `CFG-002`, and `CFG-003`
remain open. Tier 1 final acceptance is still pending, Tier 2 has not started,
and the next task is only the separate final whole-Tier-1 acceptance gate.

### 2026-07-17 final whole-Tier-1 independent acceptance

**Decision:** Tier 1 is locally accepted in full. The final review independently
traced all 16 acceptance groups (A-P) across the live source, strict schema,
resolver, runtime/provenance model, Simulator, CLI/workflow, backend/precision
factory, documentation, samples, and tests. Five regression-first corrections
closed silent backend precision downgrade, a map immutability escape hatch,
standard precision dump/reload conflict, a nonexistent documented backend
method, and a transitional provenance serialization alias. No Tier 2 behavior
was introduced.

Final issue disposition:

- **CFG-001 — DONE.** Every public input form reaches the same ordered schema,
  override, semantic/unsupported, path, and immutable-runtime resolution stages.
  Equivalent YAML, mapping, typed-model, and parameter inputs have equivalent
  scientific meaning; direct construction accepts resolved runtime only.
- **CFG-002 — DONE.** The resolved document backend is honored consistently;
  explicit overrides win, `None` preserves the document, `auto` is real, NumPy
  remains the declared deterministic default, and unavailable or
  precision-incompatible explicit choices fail without fallback.
- **CFG-003 — DONE.** The live strict-model inventory covers 281 field
  occurrences, and the operational deferred-feature inventory covers 34
  non-default cases. All 34 are classified and reject exactly before backend,
  device, network, loader, output, plotting, browser, or workflow side effects.

Final local evidence is 339 focused passes on each of Python 3.11 and 3.12;
1,466 collected tests; 1,465 passed, 1 skipped, and 26 warnings in both the
Python 3.11 non-slow and default full suites; and 1,458 passed, 8 genuine
optional-JAX/data skips, and 26 warnings in the Python 3.12 non-slow suite. Ruff
lint, Ruff format, Git whitespace, the unchanged 4,600-error Pyright ceiling
(4,446 live diagnostics), all three real YAML validations, CLI/example/
config-mode/exact-frequency/notebook smokes, and a clean Sphinx build all pass.
The Sphinx result remains 49 classified pre-existing warnings and contains no
Tier 1 configuration/API truth defect.

Remote CI, optional GPU hardware, and external scientific-network verification
remain unobserved and are not claimed by this local acceptance. Later issue IDs
remain open at their assigned tiers. Tier 2 has not started. The next task may
prepare or review the separate Tier 2 instrument-and-baseline-resolution
boundary. Nothing was staged, committed, pushed, branched, or submitted as a
pull request.

### 2026-07-17 Tier 2 instrument-resolution design gate

The implementation-ready instrument-and-baseline-resolution architecture is
recorded in `Tier2InstrumentPlan.md`. Tier 1 remains locally complete and
accepted. Tier 2 implementation has not started; INS-001, INS-002, and INS-003
remain open. Tier 3 and later work remains untouched, and remote CI remains
unobserved.

### 2026-07-17 independent Tier 2 design acceptance

**Decision:** the Tier 2 instrument-and-baseline-resolution design is accepted
for implementation. The independent review traced the plan against the live
source, all current readers and consumers, the accepted Tier 1 boundary,
relevant tests, and installed pyuvdata 3.2.1 behavior. It found no material
architecture, precedence, coordinate, selection-science, lifecycle, output, or
sequencing defect.

Before acceptance, the review made only minor source-derived corrections: use
pyuvdata's `2_147_483_647` antenna-number ceiling; disable both known-telescope
update defaults and let `UVData.new` derive unprojected UVW; align property
availability with instrument-only resolution/retry behavior; distinguish dense
dependency diameter arrays from row-level missing values; document the exact
registry/execution-offline composition, policy provenance, and Astropy
lock/restoration limitation; add the active CLI migration and legacy I/O
re-export deletion; define the direct/config-file CLI migration with no hidden
14-m default; and add internal 2G review checkpoints and tolerance rationale.
These corrections do not change the selected architecture.

Independent evidence is 144 passed/1 optional-data skip on Python 3.11 and 143
passed/2 optional JAX/data skips on Python 3.12 from the 145-test focused suite,
with four existing warnings in each environment. Ruff lint, Ruff format, and
Git whitespace pass. An offline pyuvdata construction probe confirms ENU/ECEF
round trip and dependency-derived UVW within `2.12e-10 m`, unprojected phase
semantics, Astropy setting restoration, and the identifier ceiling. All seven
Mermaid blocks are structurally paired; Mermaid CLI was unavailable, so no
raster-render result is claimed.

This accepts only the design gate. No production code or Tier 2A test was added.
INS-001, INS-002, and INS-003 remain **OPEN** until implementation completes
through 2H. The next authorized task is Tier 2A characterization only, with the
stop boundary in `Tier2InstrumentPlan.md`; Tier 2B and production work cannot
start until 2A is independently accepted. Remote CI, optional GPU hardware,
external scientific-network behavior, the full suite, Pyright, and Sphinx were
not rerun or claimed for this planning-only gate.

### 2026-07-18 Tier 2A independent acceptance

**Decision:** Tier 2A is independently accepted without correction. The review traced
implementation commit `b7f0dc9f10244d5e6d452407bc9fdb5cd8b8f2f7` against its
parent, the live antenna, baseline, point/HEALPix, Simulator, observability,
visualization, memory, result/save, and Measurement Set writer paths, and every source
and consumer in `Tier2InstrumentPlan.md` sections 4–6. The commit adds exactly the
three approved characterization modules, comprising 1,122 test-only lines, 30 test
functions, and 46 collected cases. No production, configuration, documentation, plan,
CI, dependency, lock, or existing-test file changed.

All six legacy readers, duplicate/malformed behavior, formatter mutability, uniform
diameter overwrite, mutable public/result aliases, complete sorted auto/cross baseline
inventory, exact `position(ant2)-position(ant1)` direction, negative point/HEALPix
phase, current point/HEALPix/observability/writer diameter fallbacks, dead opaque
baseline strings, visualization and count-only memory consumers, and both Measurement
Set missing-vector fallback branches have direct proving evidence. Fakes isolate only
dependency or side-effect boundaries; fixtures use temporary paths; the tests require
no network, physical GPU, browser, persistent output, or user cache. No skip or xfail
was added, no future Tier 2 behavior is asserted, and no minor correction was needed.

Fresh focused runs passed all 46 cases without skips or warnings on Python 3.11.13 and
3.12.13. The 191-case combined boundary reported 190 passed, 1 existing optional-data
skip, and 4 existing warnings on Python 3.11; Python 3.12 reported 189 passed, 2
existing optional JAX/data skips, and the same 4 warnings. Ruff lint passed, all 256
files passed Ruff's format check, and staged/unstaged Git whitespace checks passed. The
full suite, Pyright, Sphinx, remote CI, external-network behavior, and physical GPU
hardware were not run or claimed.

INS-001, INS-002, and INS-003 remain **OPEN** until Tier 2 completes through 2H. Tier
2B has not started; it is now the next authorized implementation slice and belongs to
a fresh task under its own stop and acceptance gates.

### 2026-07-18 Tier 2B independent acceptance

**Decision:** Tier 2B is independently accepted after repairing the Pyright
reproducibility gate. The first review accepted the production schema on substance but
rejected the mandatory gate because the wildcard Pixi dependency resolved Pyright
1.1.408 in Python 3.11 and 1.1.411 in Python 3.12 while the checked-in baseline
recorded 1.1.408. The version-sensitive checker correctly failed that mismatch before
ceiling acceptance. No Tier 2B production defect remained.

Tooling commit `611b3f86a3e638a2bdf20f73ffe20900ca5278cc` pins Pyright
exactly to 1.1.408 and regenerates the v7 Pixi lock. The pin solves for Python 3.11 and
3.12 on Linux x86-64, macOS x86-64, and macOS arm64, establishing manifest = both
environment locks = baseline version. The diagnostic ceiling remains unchanged at
4,600. No checker code, Pyright rule, include/exclude setting, strictness option,
environment, or platform changed.

Fresh raw Pyright reports in both synchronized environments used 1.1.408 and each
reported 4,446 errors, zero warnings, and zero information diagnostics across 153
analyzed files. Neither reported a diagnostic in
`src/radiosim/io/instrument_config.py`. The Tier 2B tests are outside Pyright's
configured `src/radiosim` include and are not claimed as type-checked. Both baseline
commands passed at 4,446 <= 4,600 without changing `pyright-baseline.json`.

The focused Tier 2B module collected 208 cases and passed all 208 on Python 3.11 and
3.12 with zero skips or warnings. The combined configuration boundary passed all 310
cases on both versions with zero skips or warnings. Ruff lint passed, all 258 files
passed Ruff's format check, staged and unstaged Git whitespace checks passed, and no
skip or xfail marker exists in the Tier 2B test. The same two typechecks, two 310-case
boundaries, lint, formatting, and whitespace checks passed again from the committed
tooling state.

Implementation commit `6bcf7854d57555cc23a1192f4628ff2852a7827e` remains the
strict inactive input contract. Test-only correction commit
`db0d849f50995e13c4c2aa01f4268a3fefdc3d88` adds strict invalid-path-type cases,
strengthens normalized and blank layout identity cases, and proves import isolation in
a fresh subprocess without the prior `importlib.reload()` class-identity
contamination. It changes no production behavior. The active schema, resolver, CLI,
Simulator, loaders, runtime, baselines, selection, and writers remain unchanged.

The full suite, Sphinx, remote CI, external-network behavior, live registry downloads,
and physical GPU hardware were not run or claimed. INS-001, INS-002, and INS-003
remain **OPEN**; Tier 2 is not complete. Tier 2C was not started and is now the next
authorized implementation slice under its existing independent stop and acceptance
gate.

### 2026-07-19 Tier 2C independent acceptance

**Decision:** Tier 2C is independently accepted after corrections. Implementation
commit `72f2f63953582fea9f6b4e6c617d98511bb66e17` adds exactly the seven Section 9
canonical public types and changes only the four authorized production, test, and
export files. The review confirmed every field one-to-one, frozen slotted dataclasses,
strict normalization and validation, deterministic ordering and hashing, fresh
mapping-proxy indexes, complete JSON-safe snapshots, identity-equal exports, and the
absence of loaders, coordinate conversion, precedence, baselines, selection,
Simulator, solver, writer, output, observability, dependency, lock, or CI changes.

The review found one caller-ownership defect: `isinstance` accepted mutable subclasses
of the frozen canonical dataclasses, allowing a caller-owned nested model to change
after construction. A regression was added first and failed as expected. Correction
commit `cba011aa052842edef23ae4f050997349781d501` requires exact nested canonical
classes and exact antenna inventory items. It also makes mapping-shaped coordinate
rejection explicit in the test matrix. No public field, fingerprint, snapshot, export,
or later-tier boundary changed.

The fixed independent fingerprint remains
`c57da5979e17852d23c51f15ba6006dac4536ff8b0c44aab3f9caeefbc6cdbf6`. Tests prove
canonical UTF-8 JSON, input-order and Unicode equivalence, negative-zero normalization,
scientific and field-source sensitivity, transport-path independence, hash-mismatch
rejection, fresh snapshot ownership, and public export identity. The complete Tier 2C
module passed 135 cases on Python 3.11.13 and 135 on Python 3.12.13. The combined Tier
2B/Tier 2C boundary passed 343 cases on each version. All four runs had zero failures,
skips, xfails, or warnings. The count increase from 131/339 is exactly the new
mutable-subclass regression and three mapping-coordinate cases.

Ruff lint passed; Ruff format reported 260 files formatted. Pyright 1.1.408 reported
4,446 full-source errors in both environments under the unchanged 4,600 ceiling, and
the Tier 2C production module had zero direct diagnostics on each. Both import/export
smokes, staged and unstaged whitespace checks, and the no-skip/xfail search passed.
The full suite, Sphinx, remote CI, external-network behavior, live registry downloads,
and physical GPU hardware were not run or observed.

INS-001, INS-002, and INS-003 remain **OPEN** until Tier 2 completes and receives final
acceptance through 2H. Tier 2 is not complete. Tier 2D was not started; it is now the
next authorized implementation slice and belongs to a separate fresh task under its
own stop and acceptance gates.

### 2026-07-19 Tier 2D independent acceptance

**Decision:** Tier 2D is independently accepted after corrections. Dependency
prerequisite `832640713c49dc730c2a813927663a0f9067b161` reproducibly pins pyuvdata
3.2.1 in both Python environments and all supported platform locks. Implementation
`37b30f3663a624456ae63b85bcd246c4754466b3` adds exactly the four authorized loader,
coordinate/staging, and test files. Correction
`244ba9f751555274602687adbbd1c81e4b3ccad0` adds selected-source context to normalized
row errors, removes internal records from module `__all__`, and replaces a timing-
sensitive concurrency assertion with event coordination. Regression tests for the
contract defects failed first. Legacy readers and all Tier 2E/later consumers remain
unchanged.

The lock audit found no Python 3.11 reference changes. Python 3.12 necessarily selects
pyradiosky 1.1.0 instead of 1.1.1 because the latter requires pyuvdata 3.2.3 or newer,
and moves numba/llvmlite from Conda 0.65.1/0.47.0 to PyPI 0.66.0/0.48.0 after removal
of pyuvdata 3.2.6's direct Conda numba constraint. Linux and macOS arm64 use wheels;
macOS x86-64 uses source distributions. Removed 3.2.6 transitive constraints are
expected solver consequences. Pyright 1.1.408, its 4,600 ceiling, platforms,
environments, package/project versions, and application dependencies are unchanged;
broader tests found no dependency regression.

Strict RadioSim ENU, replacement `casa_loc`, renamed `mwa_metafits`, metadata-only MS
and UVFITS, and distinct known-telescope loading are accepted. Ambiguous pyuvdata
text and legacy spellings/booleans have no new route. Parsing, identities, dense
metadata, optional source diameters, hashes, error classes/chaining, and source
references are strict. Relative ECEF is converted through public pyuvdata utilities
about an exact `EarthLocation`; independent expected and inverse round-trip results
agree within `1e-6 m`. Explicit/embedded separation tests cover zero, interior,
exactly 1.0 m, and just over 1.0 m. The known loader is injectable, offline by default,
serialized with a re-entrant lock, and restores Astropy configuration after success or
failure. Returned staging is owned, frozen, slotted, deterministic, dependency-free,
and deliberately diameter-incomplete.

Focused Tier 2D tests passed 99/99 on Python 3.11 and 3.12; Tier 2D plus legacy
characterization passed 135/135 on each; and the complete Tier 2 input/model/source
boundary passed 442/442 on each, all without skips, xfails, or warnings. The full
Python 3.11 suite reported 1,953 passed, one optional-data skip, and 26 existing
warnings. Python 3.12 non-slow reported 1,946 passed, eight optional data/JAX skips,
and the same 26 classified existing warnings. Ruff lint/format, both Pyright ceiling
checks at 4,446 diagnostics, zero direct diagnostics in the new modules, dual-Python
imports, lazy-import isolation, and whitespace passed.

Remote CI, physical GPU hardware, external scientific-network/live registry behavior,
and Sphinx remain unobserved. INS-001, INS-002, and INS-003 remain **OPEN**. Tier 2 is
not complete. Tier 2E was not started; it is now the next authorized slice and must be
implemented in a separate fresh task under its own acceptance gate.

### 2026-07-20 Tier 2E independent acceptance

**Decision:** Tier 2E is independently accepted without correction. Implementation
`61d65849461ab3c3ab001f6af5fbf57695dfb3ec` changes exactly the four authorized
canonical-model, final-resolution, and unit-test files. Git history reconstructs the
test-first failure because the parent contains neither the new test module nor the
final resolver surface. No baseline, selection, Simulator, solver, writer, plotting,
observability, root export, dependency, lock, or later-tier behavior changed.

The review accepted every identity, location, identifier, metadata, and diameter
precedence path. Known/local/dataset identities follow their exact stripped-NFC,
case-sensitive rules; explicit and embedded locations retain Tier 2D's 1.0-m contract;
positions remain source-only; generated `casa_loc` identity is recorded; and mount and
BeamID remain inert. Tagged override references use exact number/name namespaces and
reject unknown or repeated targets. Diameter precedence is override over valid source
over configured default; invalid present source values never fall through, incomplete
antennas are aggregated in canonical order, pre-override source values remain in
provenance, and every final diameter is finite and positive with no hidden 14-metre
fallback.

The complete `ResolvedInstrument` is frozen, sorted, copy-owned, hashable, JSON-safe,
and mutation-independent. Finalization reuses the staged location and the canonical
fingerprint seam, records every instrument and antenna provenance field, and does not
repeat source or coordinate work. Determinism, transport independence, field-source
sensitivity, hash revalidation, strict typed input, exact internal export scope, and
success/failure non-mutation are covered.

The model/final-resolver boundary passed 213/213 tests on Python 3.11.13 and 3.12.13;
the complete Tier 2 input/model/source/coordinate/resolution boundary passed 520/520
on each; and 46/46 legacy characterization tests passed on each. These focused runs
had no failures, skips, xfails, or warnings. The full Python 3.11 suite reported 2,031
passed, one optional-data skip, and 26 existing warnings; Python 3.12 non-slow reported
2,024 passed, eight optional data/JAX skips, and the same 26 existing non-instrument
warnings.

Ruff lint/format, both 4,446-diagnostic Pyright ceiling checks, zero direct diagnostics
in the changed production modules, no-skip/xfail and exact-scope guards, and whitespace
passed. Remote CI, physical GPU hardware, external scientific-network/live registry
behavior, and Sphinx remain unobserved. INS-001, INS-002, and INS-003 remain **OPEN**.
Tier 2 is not complete. Tier 2F was not started; it is now the next authorized slice
and must run separately under its own implementation and acceptance gate.

This decision was independently re-verified on 2026-07-20 after benign checkout drift
to acceptance-record commit `ccbe52a7a126f090870c00804de46ab796a158c1`.
The required Python 3.11.13 and 3.12.13 suites each passed exactly 77 focused resolver,
213 model-plus-resolver, 520 complete source/model/resolution, and 268 combined legacy-
characterization boundary cases. Direct dual-Python probes also proved rejection of
different-valued duplicate overrides and malformed source diameters despite an
override, exact handling of numeric-looking names, one staging call per final
resolution, and strict rejection of `InstrumentConfig` subclasses.

The re-verification found no production or test correction to make. Ruff lint and
formatting, Pyright 1.1.408 at 4,446 diagnostics under the unchanged 4,600 ceiling,
zero direct diagnostics in the two production modules, import/export smokes,
no-skip/xfail and no-hidden-fallback searches, later-tier and exact-scope audits, and
staged/unstaged whitespace checks passed. It did not rerun the full repository suite,
Sphinx, remote CI, physical GPU hardware, or external scientific-network/live registry
behavior. INS-001, INS-002, and INS-003 remain **OPEN**, Tier 2 remains incomplete,
and no Tier 2F work was started.

### 2026-07-20 Tier 2F independent acceptance

**Decision:** Tier 2F is independently accepted after corrections. Implementation
`9f42cf084052048d912711f537e696521a3f9654` changes exactly its six authorized
canonical-model, baseline-resolution, export, and test files. Its parent contains none
of the new module, tests, or four models. No root export, legacy baseline implementation,
active configuration, CLI, Simulator, solver, writer, Measurement Set, plotting,
observability, beam, dependency, lock, or Tier 2G behavior changed.

The review independently accepted exact numeric pair ordering and counts, exact
`ant2-ant1` ENU vectors and 3D norms, zero autos, the inclusive `1e-9 m` coincidence
threshold, North-zero/East-90 modulo-180 azimuth, endpoint and antenna-number
invariance, pure vertical geometry, target/range tolerances, normal and wrapped angle
ranges, within-category union, between-category intersection, stable order, truthful
auto exemption, deterministic empty errors, immutable snapshots, ownership, exports,
and the exact error hierarchy.

Regression-first review found that forged public provenance could encode
nontriangular or impossible correlation pipelines, final baseline geometry could
contradict active criteria, and angular tolerance was discontinuous at the axial
zero/180 seam. Correction `667850154740e0830d3535d3eb144c63a13c52eb` changes only
`instrument.py`, `baseline_resolution.py`, and the baseline-selection test. It enforces
triangular and exact correlation counts, selected identity cardinality, final
length/azimuth coherence, and circular axial boundary distance without changing the
accepted schema or later-tier boundary.

The committed Tier 2F suites passed 136/136 on Python 3.11.13 and 136/136 on Python
3.12.13. The affected boundary passed 571/571 and 570 passed/one existing optional-JAX
skip. The full Python 3.11 suite reported 2,167 passed, one existing Vivaldi-FITS skip,
and 26 existing non-instrument warnings. Python 3.12 non-slow reported 2,160 passed,
eight existing skips (seven missing-JAX and one Vivaldi-FITS), and the same 26 warnings.

Ruff lint and formatting passed for all 268 files. Pyright 1.1.408 remained at 4,446
diagnostics in both environments under the unchanged 4,600 ceiling, with zero direct
diagnostics in the two production modules. Fifteen dual-Python adversarial probe
groups, no-skip/xfail, exact-scope, later-tier, export, and whitespace checks passed.
`pyright-baseline.json`, dependencies, and lockfiles are unchanged.

Sphinx, remote CI, physical GPU hardware, and external scientific-network/live
registry behavior remain unobserved. Nothing was pushed or published. INS-001,
INS-002, and INS-003 remain **OPEN** until Tier 2 completes through 2H. Tier 2 remains
incomplete. Tier 2G was not started; it is now the next authorized separate slice under
its existing implementation and independent-acceptance stop gates.

### 2026-07-20 Tier 2G independent acceptance

Tier 2G is independently accepted after corrections. Implementation
`d32a4ff036de7f28afc1c1bfeb536ac103328f53` performs the 53-path atomic cutover to
one strict instrument/selection schema, one resolved state, typed public properties,
one lossless solver adapter, canonical results/provenance, and typed writer boundaries.
The implementation's narrow workflow and characterization-test expansions are
necessary and contain no unrelated cleanup or escape hatches.

Regression-first review found incomplete-state and direct-adapter ownership holes,
mutable runtime subclass paths, non-finite serialization, an active legacy-named path
override, Measurement Set resize/truncation, silently omitted HDF5 metadata,
first-antenna point-beam selection, non-strict solver zips, and misleading direct CLI
help/errors. The primary red probe reported 15 failures and 26 passes, with separate
beam and CLI regressions also observed red. Correction
`dd1b91a3be71aa5de017725c5db2543517e70147` changes 18 focused files and closes each
defect. Its sole path outside the original implementation list is
`src/radiosim/io/__init__.py`, required to replace the actively exported
`AntennaLayoutOverride` with `InstrumentSourcePathOverride`.

The corrected active path rejects all old spellings, resolves paths and offline policy
before runtime side effects, assigns canonical state atomically, preserves it across
later failures, rebuilds later state on retry, and exposes stable typed tuple identity.
Point and HEALPix receive identical IDs, selected pairs, vectors, and heterogeneous
diameters without a 14-metre, first-antenna, missing-ID, or dictionary fallback.
Heterogeneous observability rejects before renderer/browser/file work. JSON and HDF5
preserve one detached instrument snapshot and fail on non-finite metadata. Measurement
Set writes require exact selected keys and exact data shapes, preserve canonical
geometry/diameters, disable both registry-update flags, and use pyuvdata-generated
UVWs. A dual-Python local pyuvdata 3.2.1 probe confirmed row/data/frequency/polarization
alignment without persistent output or network access.

Final focused results are 584 passed, one optional skip, and four existing warnings on
Python 3.11.13; and 583 passed, two optional skips, and four existing warnings on
Python 3.12.13. The canonical boundary is 656/656 on both. Full results are 2,209
passed, one optional skip, and 26 existing warnings on Python 3.11; and 2,202 passed,
eight optional skips, and the same 26 warnings on Python 3.12 non-slow. Ruff and
formatting pass. Pyright 1.1.408 reports 4,265 diagnostics on both under the unchanged
4,600 ceiling, with zero direct adapter diagnostics. Sphinx succeeds with the same 11
classified pre-existing warnings. All shipped YAML validates, the offline script and
temporary-kernel notebook smokes pass, and committed outputs remain clean.

Inactive legacy declarations remain for Tier 2H. Because its old exact list could not
remove all of them, Tier 2H now also owns `src/radiosim/io/config.py` and
`src/radiosim/core/runtime_config.py`. No 2H deletion or issue closure was performed.
Remote CI, physical GPU hardware, and external scientific-network/live registry
behavior remain unobserved. Nothing was pushed or published. INS-001, INS-002, and
INS-003 remain **OPEN**, Tier 2 remains incomplete, and Tier 2H is the next authorized
task only in a separate fresh task.

### 2026-07-20 independent whole-Tier-2 acceptance

**Verdict: Tier 2 is independently accepted after corrections.** The review covered
the complete Tier 2A-through-2H history and the final combined checkout rather than
relying on any slice handoff. Tier 2H implementation
`112f52fb0f903e0361fb6ec38199c081f63a93ed` has the claimed 21-path,
553-insertion/2,455-deletion scope: three production modules and three superseded
characterization suites were deleted, one cleanup suite was added, and the approved
`CLAUDE.md` and sky-support lazy-import changes were necessary. No forwarding module,
compatibility alias, broad root-import guard, later-tier behavior, or unrelated
cleanup remains.

The review found one narrow acceptance defect in active truth surfaces: the
contributor guide still named a deleted reader and the old top-level schema, while two
live module docstrings described integrated Tier 2 code as inactive or deferred. A
new cleanup regression failed first on Python 3.11, then correction
`041ba778f835d1b5d9c11e3e8308e8d217951cdd` updated only those statements and the
regression. The cleanup suite then passed 13/13 on Python 3.11 and Python 3.12. A
separate post-correction source and diff review found no runtime or scientific change.

Source-first review confirmed each Section 31 criterion. Strict discriminated inputs
retain only `radiosim`, `casa_loc`, `mwa_metafits`, `measurement_set`, `uvfits`, and
the distinct `known_telescope` source. Identity, frame, Earth location, relative-ECEF
conversion, metadata-array lengths, source hashes, dependency facts, ordering, and
provenance are validated before backend/device work. Canonical instrument, baseline,
selection, state, and solver-view models enforce exact classes, immutability,
copy-ownership, deterministic JSON-safe snapshots, and fingerprints. Selection is
generated once and reused by both solvers, observability, results, plots, HDF5/JSON,
and Measurement Set writers. Uniform observability uses the exact common diameter;
heterogeneous observability still rejects before renderer/browser/file work, so no
Tier 3 semantics were introduced.

The deleted characterization assertions were mapped to live canonical coverage.
Accepted invariants now have stricter tests for every retained parser, malformed and
duplicate rejection, coordinate conversion, identity/order, diameter precedence and
provenance, baseline count/sign/length/coincidence and filter algebra, point/HEALPix
phase and heterogeneous-diameter parity, Measurement Set signatures and shapes,
observability diameter handling, and immutable independently owned state. Deleted
assertions for mutable dictionaries, opaque strings, permissive parsing, silent
overwrite/truncation, ambiguous formats, and hidden defaults were intentionally not
preserved.

INS-001 is closed by typed override/source/default precedence and complete positive
finite per-antenna diameters; controlled `[12.0, 25.0]` cases reach both solvers and
writers unchanged with field-source provenance. INS-002 is closed by the working
typed known-telescope path, separate dataset sources, explicit offline/registry policy,
disabled registry updates, pyuvdata-version provenance, mismatch rejection, and
high-level Simulator use. INS-003 is closed by canonical pair generation and one
typed correlation/length/axial-azimuth selection pass with exact counts, numeric
boundaries, algebra, empty failure, provenance, and shared consumer identity.

The required canonical boundary passed 704/704 on Python 3.11.13 and 704/704 on
Python 3.12.13. The config/API/CLI/provenance/consumer boundary reported 220 passed,
one optional Vivaldi-data skip, and four known warnings on Python 3.11; Python 3.12
reported 219 passed, two optional JAX/Vivaldi skips, and the same four warnings. The
full Python 3.11 suite collected 2,188 and reported 2,187 passed, one Vivaldi skip, and
26 existing non-instrument warnings. Python 3.12 non-slow reported 2,180 passed, eight
skips (seven unavailable-JAX cases and Vivaldi), and the same 26 warnings. There were
zero xfails and no new Tier 2 skip, skipif, or xfail marker. The warnings are explicit
lossy HEALPix notices, Astropy FITS-unit notices, a NumPy edge-case warning, an
explicit disjointness warning, and Matplotlib figure-reuse notices.

Ruff lint passed, all 267 files passed the Ruff format check, and Pyright 1.1.408
reported exactly 4,135 diagnostics in both environments under the unchanged 4,600
ceiling. Both Git whitespace checks passed. The three RadioSim YAML files validate;
the offline script completes with no save or plot; all five notebook code cells run
through a temporary Pixi kernelspec and temporary output while the committed notebook
remains output-free with SHA-256
`e328741a917535d298a672bdb3cb5b763f22ce3958ae37a6af062017a6079406`. CLI help,
validation output, the instrument guide/toctree, and fresh-process imports are current.
Both Python versions pass forward/reverse import orders, coherent lazy-helper identity,
`import *`, and unknown-attribute checks without a broad exception suppressor.

Fresh Sphinx 8.2.3 `-M html` builds used the same interpreter, corresponding detached
source trees, and new output/doctree directories. The parent
`4eedaf861b60bc020bbca5f1c17aa1a99f52955a` succeeded with 42 warning events and Tier
2H succeeded with 40. The current events classify as 35 pre-existing lower-level
docstring/docutils events, one historical document outside the toctree, three
historical HERA highlighting events, and one unsupported theme option. Tier 2H added
no category and removed exactly two legacy-baseline autodoc events. Both logs render
28 unique `WARNING:` lines because Sphinx suppresses repeated messages while still
counting their events. An incremental parent-to-current replay produced only seven
warnings, demonstrating that the earlier Tier 2G count of 11 was cache-dependent and
not a clean-build baseline. The Tier 2H handoff's 44/42 result was also not reproduced;
the reproducible clean comparison is 42/40, with the same two-warning reduction.

Repository-wide residual searches classify every remaining old name as migration
guidance, explicit rejection/absence coverage, historical plan text, or a substring
of a canonical replacement such as `baseline_resolution`. No active definition,
import, re-export, accepted field, fallback, opaque baseline dictionary, manual UVW
assignment, `np.resize`, first-antenna diameter selection, or generated writer identity
remains. Dependencies, `pixi.lock`, `pyright-baseline.json`, workflows, vendored
simulators, Tier 1 plans, generated outputs, and Tier 3+ implementation were untouched.

Remote CI, live scientific-network and registry behavior, physical GPU hardware, and
the optional Vivaldi data mount remain explicitly unobserved. During the review the
local `origin/main` reference advanced externally from
`cf353dbb1d6727ae308fda71a803576ab353bf5d` to the accepted Tier 2H commit; no fetch,
pull, push, publication, or history rewrite was performed by this task.

The next authorized task is **Tier 3 design gate — beam and observability
integration**. It must first turn the Tier 3 outline into an implementation-ready
design with no unresolved beam or observability decisions. Tier 3 implementation is
not authorized by this acceptance.

### 2026-07-21 Tier 3 beam and observability design gate

**Verdict: design accepted after independent correction and review.** The
implementation-ready architecture is recorded in
[Tier3BeamObservabilityPlan.md](Tier3BeamObservabilityPlan.md). It selects a strict
four-mode beam schema, complete canonical assignments to Tier 2 antenna identity, one
Simulator-local `BeamSystem`, the validated scalar BeamFITS subset, one Jones RIME for
both visibility solvers, a minimum baseline-product NSIDE advisor, and sibling
observability planning with an explicit reference antenna whenever scientific beam
fingerprints differ.

Implementation is split into ten planned, independently accepted units from Tier 3A
through 3I: dependency characterization,
strict schema/path replacement, immutable assignment resolution, BeamFITS validation,
BeamSystem lifecycle/deduplication, shared solver integration, observability
integration, separate 3H.1 NSIDE/provenance and 3H.2 legacy/truth cleanup, and
independent whole-tier acceptance.
Every slice has exact writable files, tests-first evidence, verification commands,
stop conditions, one narrow conventional commit, and a separate acceptance gate.

The pre-design focused baseline collected 205 tests in each environment. Python
3.11.13 reported 204 passed, one optional mounted-data skip, two known warnings, and
zero xfails; Python 3.12.13 reported 200 passed, five optional mounted-data/JAX skips,
two known warnings, and zero xfails. Ruff lint and formatting passed. Pyright 1.1.408
reported 4,135 diagnostics in both environments under the unchanged 4,600 ceiling.
A clean tracked-source Sphinx build succeeded with the accepted 40 classified events;
the live tree had two additional events caused only by local ignored documents. The
full suite, remote CI, physical GPU hardware, external network/registry behavior, and
optional mounted Vivaldi data remain unobserved.

The 2026-07-21 independent review corrected the plan before deciding acceptance. The
corrections include native `complex64` provenance/canonicalization, the exact pyuvdata
basis transform, complete UTC/LST time seams, unique sampling/render/output errors,
the Tier 3F stale-observability runtime guard, all affected visualization wrappers,
truthful 3B documentation, the split 3H.1/3H.2 boundary, explicit new/deleted files,
the actual core beam-projection test path, BEAM-003 closure evidence, and the recovered
205-test command plus a 75-test supplemental boundary. These corrections change no
implementation status and do not start Tier 3A.

No production, test, configuration, dependency, example, workflow, or generated file
was changed, and no implementation slice was started. The independent review accepted
the corrected design on 2026-07-21 after correction commit
`a208b61dce086e4afe3c49e1f2524b4b229a9c16`. `BEAM-001`, `BEAM-002`, and `BEAM-003`
remain **OPEN**; `OBS-001` remains **OPEN**; and design acceptance moves `OBS-002` from
**DECISION** to **OPEN**. No issue is **DONE**. Tier 3A is now the only authorized
implementation slice and has not started; Tier 3B and later work require separate
implementation and acceptance gates.

### 2026-07-21 Tier 3A independent acceptance

**Decision: Tier 3A is accepted after corrections.** Implementation
`bb0678830a38db59e7f2679c6fa8f6a5699a1250` added exactly the two planned test files
and 1,216 lines. The review reproduced the original absent-helper
`ModuleNotFoundError`, read the implementation and governing contracts line by line,
and independently exercised pyuvdata 3.2.1 in Python 3.11.13 and 3.12.13.

Adversarial probes found fail-open science/unsupported enum identity, empty-filename
defaulting, dangling-symlink path escape, mutable or malformed loader schedules, and
a symmetric-only Jones-axis oracle. The primary regression-first slice reported 18
failures/3 passes; the symlink regression separately failed before correction.
Correction `b2de40d88a2f2630e9031fe2eafaee6775ff0499` changes only the same two test
files and makes each boundary fail closed while adding an asymmetric dependency probe.

The corrected focused suites passed 75/75 in both Pythons with no skip, xfail,
failure, or warning. Full Python 3.11 reported 2,262 passed, one existing Vivaldi-data
skip, and 26 existing warnings; Python 3.12 reported 2,255 passed, eight existing
Vivaldi/JAX skips, and the same 26 warnings. Ruff lint/format passed for 269 files;
Pyright 1.1.408 remained at 4,135 diagnostics in both environments under the unchanged
4,600 ceiling; all three YAMLs, the offline example, Git whitespace, scope/isolation
searches, and a fresh tracked-source Sphinx build with the unchanged 40 classified
events passed. No generated BeamFITS remains; the two residual matches are tracked
data inside clean pre-existing simulator submodules.

No production, config, dependency, lock, baseline, README, example, workflow, or
shipped YAML file changed. Nothing was pushed or published. Remote CI, physical GPU,
mounted Vivaldi data, and live scientific network/registry behavior remain unobserved.
`BEAM-001`, `BEAM-002`, `BEAM-003`, `OBS-001`, and `OBS-002` remain **OPEN**; none is
**DONE**. Tier 3B is the next authorized slice and was not started.

### 2026-07-21 Tier 3B independent acceptance

**Decision: Tier 3B is accepted after corrections.** Implementation
`284e29c08567b908fadc6c5739b17ae1a889ed37` introduced the strict four-mode input
schema, immutable source-resolved beam definitions and fingerprints, nested path
resolution, documentation, and fail-closed Simulator guards. The review verified its
30-file scope and accepted the separately authorized eight-line integration-fixture
migration as narrow and necessary.

Adversarial probes found noncanonical or subclassed resolved FITS scalars, blank
provenance keys, hostile Pydantic subclasses at nested and serialization boundaries,
misleading Tier 3C guard wording, and eager schema imports that loaded 208 JAX
modules. The primary regression selection failed 8/10 before correction; the fresh
import regression then failed alone. Correction
`924f25d0378728bc0fe522a89b81355863f9ce8e` closes those gaps with exact/final model
boundaries, canonical concrete path/string requirements, later-slice error wording,
and dependency-light lazy package exports that retain public object identity.

The final focused boundary passed 347/347 on Python 3.11.13 and 347/347 on Python
3.12.13. The authorized integration file passed 16/16 on both. Full Python 3.11
reported 2,415 passed, one existing optional-data skip, and 26 existing warnings;
Python 3.12 reported 2,408 passed, eight existing optional-data/JAX skips, and the
same 26 warnings. There were zero failures and xfails, and no Tier 3B skip, skipif, or
xfail marker.

Ruff lint and the 275-file format check passed. Pyright 1.1.408 reproduced 4,178
diagnostics in both environments under the unchanged 4,600 ceiling; both new beam
modules had zero direct diagnostics. Fresh imports loaded none of the forbidden heavy
or later-tier modules. Fixed analytic/FITS fingerprint probes matched across both
Pythons. All three shipped YAMLs passed using the live `radiosim validate <config>`
subcommand; the handoff's root `--validate` spelling is stale and was recorded as such.
The offline example completed with five antennas, 15 baselines, and two channels.
A corrected clean-source Sphinx 8.2.3 build reproduced the 40-event baseline. Git
whitespace, marker, legacy-field, export, scope, and later-tier leakage searches
passed.

Only direct circular analytic beams are runtime-active. FITS modes and other analytic
variants remain valid input but fail before side effects using
`beam_runtime_fits_pending` or `beam_runtime_analytic_variant_pending`. No Tier 3C
assignment, FITS loading, conversion, cache, solver, or observability implementation
was added. Nothing was pushed or published. Remote CI, physical GPU execution,
mounted Vivaldi data, and live network/registry behavior remain unobserved.
`BEAM-001`, `BEAM-002`, `BEAM-003`, `OBS-001`, and `OBS-002` remain **OPEN**; none is
**DONE**.

### 2026-07-21 Tier 3C independent acceptance

**Decision: Tier 3C is accepted after corrections.** The strict start gate passed on
clean `main` at implementation `8b1b9d8e7c030e864fb3060581b60cfaffa9529b`, parent
`ba044d496da80711078ef3278c53ff5c39c78ece`, with `origin/main` at
`112f52fb0f903e0361fb6ec38199c081f63a93ed` and divergence zero behind/12 ahead.
The implementation changed exactly the seven authorized core/beam and unit-test files
with 1,980 insertions and seven deletions. Its untouched focused suites passed 84/84
on Python 3.11.13 and 84/84 on Python 3.12.13 without failure, skip, xfail, xpass, or
warning.

Independent source review and direct-construction probes found four bounded gaps:
stale nested definition science could retain an old definition digest; stale nested
assignment science could retain an old assignment digest; state construction did not
independently reject duplicate canonical names under different numbers; and mixed
state admitted multiple analytic definitions. Four regression tests each failed
first with `DID NOT RAISE`. Correction
`cccd4805a8cf77b8236d7574339c25806a9ffa58` revalidates every nested immutable
boundary, enforces independent canonical number/name uniqueness, and enforces the
single mixed analytic model. It changes only the beam model and its unit suite.

The corrected focused suites passed 88/88 in both Pythons. A 71-group adversarial
matrix passed exact-type/subclass, malformed scalar/tuple/name/fingerprint,
unknown-before-duplicate, complete coverage, canonical order, all-mode provenance,
dedup/first-identity, BeamID inertness, active/inactive dimension, ownership/frozen
snapshot, error/export, direct-construction, and exact resolver-input boundaries. Both
Pythons reproduced assignment digest
`407b9df278595aebfa1aae895558a73385b9bd7f9f49dbaaf8e00c6bdd480f3c` and state
digest `9e158b191471edf730921f3641d2fe1eb2d8d902775be1665ee1590f01d083eb`.
Fresh-process import and patched file/socket/directory sentinels found no heavy import,
FITS access, warning, network, or filesystem side effect. No Tier 3C skip/xfail path
or Tier 3D API leakage exists.

Full Python 3.11 collected 2,486 tests and reported 2,485 passed, one existing skip,
and 26 existing warnings. Python 3.12 collected 2,486 and reported 2,478 passed, eight
existing skips, and the same 26 warnings. Both had zero failures, xfails, or xpasses.
Ruff passed and all 278 files were formatted. Pyright stayed at 4,178 diagnostics in
both environments under the unchanged 4,600 ceiling; the three changed production
modules had zero direct diagnostics. The stale default Pyright launcher path was
reproduced without changing tooling. All three YAMLs validated at 101/11/1 channels;
the offline example completed with five antennas, 15 baselines, and two channels. A
clean-copy Sphinx build matched the accepted 40-event classification, and its
temporary tree was removed. Git whitespace, marker, scope, ownership, import,
side-effect, and generated-artifact checks passed.

No Tier 3D FITS loading, conversion, caching, runtime, solver, or observability work
was started. Nothing was pushed or published. Remote CI, physical GPU behavior,
mounted Vivaldi data, and live network/registry behavior remain unobserved.
`BEAM-001`, `BEAM-002`, `BEAM-003`, `OBS-001`, and `OBS-002` remain **OPEN**; none is
**DONE**. Tier 3D is the next authorized separate task and has not started.

### 2026-07-23 Tier 3D independent acceptance

**Decision: Tier 3D is accepted after corrections.** The strict gate began on clean
`main` at implementation `6c4d652836cb9393e5dfe692b6f9b547cfeb36bb`, parent
`d1340b393692b3633f0e409057ac3ad1bc624de5`, with `origin/main` at
`112f52fb0f903e0361fb6ec38199c081f63a93ed` and divergence zero behind/15 ahead.
Both Python 3.11.13 and 3.12.13 used pyuvdata 3.2.1. The implementation changed
exactly the nine Tier 3D files and stopped at one standalone FITS evaluator.

Independent probes found that accepted tolerance noise could leak as a non-scalar
Jones matrix, private copied UVBeam arrays remained writeable, hostile dependency
containers and exceptions escaped typed RadioSim failures, nonnumeric directions
leaked NumPy errors, dependency deepcopy could alias source state, and cleanup could
obscure a primary error. The first regression run produced 20 intended failures and
two control passes. Correction
`c9cd3a36ca0a16b1ebd03a4ff4d98a06c5c86c2c`
(`fix(beam): correct Tier 3D BeamFITS contract`) changes only the two Tier 3D
production modules and their two focused unit suites. It adds 26 durable cases,
canonicalizes accepted native/interpolated noise to the X-diagonal scalar `e I2`,
freezes every owned dependency array, rejects deepcopy aliasing, closes hostile
metadata/read/check/interpolation/direction failures under typed errors with causes,
and preserves a primary failure when cleanup also fails.

The final Tier 3D boundary passed 217/217 in both interpreters; the Tier 3B/C and
precision boundary passed 101/101 in both. Full suites collected 2,628 tests:
Python 3.11 reported 2,627 passed, one existing optional-data skip, and 26 existing
warnings; Python 3.12 reported 2,620 passed, eight existing optional-data/JAX skips,
and the same 26 warnings. Ruff lint and the 282-file format check passed. Pyright
1.1.408 stayed at 4,178 diagnostics in both environments under the unchanged 4,600
ceiling, and direct checks of all four changed production files were clean. All
three YAMLs validated at 101/11/1 channels, the forced-offline example completed
with five antennas, 15 baselines, and two channels, and Sphinx reproduced the
accepted 40-event warning baseline from a removed temporary source copy.

The accepted scalar subset, coordinate/Jones ordering, frequency/interpolation
rules, precision, atomic snapshot/hash/race behavior, immutable scientific
provenance, canonical digest
`2123f69cfb6571328e681a8ccd9b3f465a5ab70d19552ba5109712dcc5996e4a`, and native
feature scale `0.5879007626540207` radians all passed independent review. The
FITS-only loaded-handler model is the intentional standalone Tier 3D shape; Tier 3E
owns the final analytic/FITS union and BeamSystem lifecycle.

No BeamSystem, deduplication, per-antenna runtime, Simulator, solver, observability,
renderer, output, browser, or NSIDE recommendation work was started. Nothing was
pushed or published. Remote CI, physical GPU execution, non-macOS platforms,
mounted external datasets, and live network behavior remain unobserved.
`BEAM-001`, `BEAM-002`, `BEAM-003`, `OBS-001`, and `OBS-002` remain **OPEN**; none
is **DONE**. Tier 3E is now the next authorized slice and was not started.

### 2026-07-23 Tier 3E independent acceptance

**Decision: Tier 3E is accepted after corrections.** The strict start gate passed on
clean `main` at implementation `e9497433591ef543743c0a27b26adc61334dde73`,
parent `d5cee392c1418174b939a8d2cd109af2aaf0460d`, with `origin/main` at
`112f52fb0f903e0361fb6ec38199c081f63a93ed` and divergence zero behind/18 ahead.
Both Python 3.11.13 and 3.12.13 used pyuvdata 3.2.1. The implementation changed
exactly the 12 Tier 3E files plus the separately authorized CLI-test migration and
stopped before Tier 3F.

The untouched focused boundary passed 309/309 in both interpreters. Independent
probes then found five bounded defect groups: analytic complex256 could contain only
float64 science; recomputed loaded state admitted mismatched handler ordinals and
frequency axes; copied factory tokens and mutable private dictionaries permitted
runtime forgery/mutation while a missing evaluator leaked `KeyError`; backend
conversion accepted wrong shape/dtype; and required atomic success logging was
absent. The first regression selection collected nine cases and all nine failed.

Correction `49d932b5f823f21a4c800747988d5fbf6551afee`
(`fix(beam): correct Tier 3E BeamSystem contract`) changes only the three Tier 3E
beam production modules and the runtime unit suite. It computes at the paired real
width, lazily uses the already locked extended-math implementation for pi/Bessel
work, rejects unavailable extended science without fallback, validates handler
ordinals/common axes and backend results, removes the reusable token, publishes
immutable map snapshots, converts corrupt runtime lookup to the fixed public error,
and emits deterministic summaries only after full validation. Ten durable cases
include the control that identical FITS science at distinct paths remains two
ordinally distinct handlers.

The scientific oracle matched every direct taper and feed primitive exactly, 18
analytical feed/taper/reflector combinations within
`8.326672684688674e-17`, and six independent `n_radial=256` Hankel combinations
exactly. Fingerprints, feature scales, exact scalar Jones output, FITS and analytic
deduplication, atomic failure/retry, canonical lookup, Simulator lifecycle,
concurrency, CLI guard ordering, exports, and lazy imports all passed. A loaded
snapshot cannot independently infer the original observation tuple without a new
field, so it now enforces the accepted minimum identical ordered handler axes while
the factory and scientific fingerprints retain the complete validated tuple.

The final Tier 3E boundary passed 319/319 in both Pythons and the complete Tier 3D
boundary passed 256/256 in both. Full suites collected 2,670 tests: Python 3.11
reported 2,669 passed, one existing skip, and 26 existing warnings; Python 3.12
reported 2,662 passed, eight existing skips, and the same 26 warnings. The ten-pass
delta is exactly the acceptance suite. Ruff and the 283-file format check passed.
Pyright 1.1.408 remained at 4,178 diagnostics in both environments under the
unchanged 4,600 ceiling; all five changed beam production modules were clean, and
the two historical integration surfaces decreased from 187 parent diagnostics to
183 with zero diagnostic on Tier 3E-added lines. Three YAML validations, the
forced-offline example, Git whitespace/hygiene checks, installed Python 3.11 JAX
conversion, and a removed clean-source Sphinx build with the unchanged 40 events
passed.

Nothing was pushed or published. Remote CI, physical GPU execution, genuine
float128/complex256 execution, non-macOS platforms, mounted external datasets, and
live network behavior remain unobserved. `BEAM-001`, `BEAM-002`, `BEAM-003`,
`OBS-001`, and `OBS-002` remain **OPEN**; none is **DONE**. Tier 3F is the next
authorized separate slice and was not started.

### 2026-07-23 Tier 3F independent acceptance

**Decision: Tier 3F is accepted after corrections.** The strict start gate passed
on clean `main` at implementation `9ab89867cbb5ab8a237f2590fb07c88375f5a9fd`,
parent `37f89c2b0fccc11ba206b3b4c719ea3614ed39e2`, with `origin/main` at
`112f52fb0f903e0361fb6ec38199c081f63a93ed` and divergence zero behind/21 ahead.
Both Python 3.11.13 and 3.12.13 used pyuvdata 3.2.1. The implementation changed
exactly the six authorized production and seven authorized test files; the sparse
HEALPix test is the separately authorized I-only normalization change. No
dependency, lockfile, documentation, generated output, observability, NSIDE,
result-provenance, legacy-deletion, or Tier 3G/H scope leaked in.

Independent source review and external NumPy oracles confirmed one exact
`BeamSystem` through the high-level, RIME, point, and HEALPix paths; explicit
nonoptional solver APIs; exact canonical antenna identity; detached direction
ownership; full `J_p C J_q^H` before extraction; endpoint complex phase and
conjugation; the established single negative geometric phase; point antenna-chain
and HEALPix handler-ID cache boundaries; strict `altitude > 0`; no evaluation for
empty/hidden domains; and I-only `C = (I / 2) I2` followed by trace. Homogeneous and
heterogeneous analytic, shared/per-antenna FITS, and mixed modes entered real
high-level simulation without dictionary re-projection or inner-loop FITS reads.

The external probe matrix covered 31 cases across non-Hermitian Jones matrices,
polarized coherency, scalar phase, autos/crosses, nonzero baselines, point/pixel
parity, every assignment family, cache counts, hostile APIs/state, ownership,
wrong backend outputs, NumPy/Numba/JAX CPU, complex64/128, and atomic no-I/O
lifecycle. Its initial Python 3.11 run passed all cases but exposed a new JAX
complex128-to-complex64 scatter warning; the warning-as-error precision selection
failed 1/6. Two first-added regressions then failed 2/2 in point and polarized
HEALPix paths.

Correction `8fd67302e6af388a2a4ab94bec266bfb33c74ef9`
(`fix(beam): correct Tier 3F visibility contract`) changes only the two visibility
solvers and their backend unit suite. It makes the configured complex output dtype
explicit at container and final reduction boundaries. Portable strict-backend
regressions pass in both Pythons without new skips. Corrected external probes passed
31/31 on Python 3.11 and 29/31 on Python 3.12 with only its two unavailable-JAX
skips; the six Python 3.11 backend/precision cases pass under `-W error`.

The line-level static audit then found 13 introduced Pyright diagnostics on Tier
3F-added lines despite lower aggregate type debt. Correction
`5f434691af73f087de179cc1ea6af167188ca802`
(`fix(beam): type Tier 3F visibility boundaries`) changes only
`api/simulator.py` and `core/visibility.py`, adding explicit override/helper typing
and a typed dynamic-result boundary without runtime changes. The six changed
production modules now report 302 diagnostics versus 475 at the parent and zero on
Tier 3F-added lines. Repository-wide Pyright is 4,005 in both environments under
the unchanged 4,600 ceiling.

The exact Tier 3F boundary passed 114/114 on Python 3.11 and 111 passed/3 established
optional-JAX skips on Python 3.12, with no warnings. Config plus sparse HEALPix
passed 74/74 in each with the same two existing warnings. The accepted Tier 3E
runtime boundary passed 319/319 in each. Full suites collected 2,697 tests:
Python 3.11 reported 2,696 passed, one existing optional-data skip, and 26 existing
warnings; Python 3.12 reported 2,690 passed, seven existing optional-data/JAX skips,
and the same 26 warnings. There were no failures, xfails, xpasses, Tier 3F skips,
or new warning categories.

Ruff lint and the 284-file format check passed. Three YAMLs validated at 101/11/1
channels; the offline example completed with five antennas, 15 baselines, and two
channels. Dual-Python import/signature smokes, Git whitespace and hygiene searches,
and a removed clean-source Sphinx build with the unchanged 40 classified events
passed. The exact Tier 3G observability error occurs before sky, planner, renderer,
directory, file, plot, or browser work for every deferred mode, while the one
homogeneous direct-circular analytic control remains permitted. Legacy public beam
surfaces remain only for the later cleanup slice; canonical solvers do not call
them.

Nothing was pushed or published. Remote CI, physical GPU, non-macOS, mounted
optional data, and live network/registry behavior remain unobserved. Tier 3G was
not implemented. `BEAM-001`, `BEAM-002`, `BEAM-003`, `OBS-001`, and `OBS-002`
remain **OPEN**; none is **DONE**. Tier 3G is now the next authorized separate
slice.

### 2026-07-24 Tier 3G independent acceptance

**Decision: Tier 3G is accepted after corrections.** The strict start gate passed
on clean `main` at implementation
`7d823e42ac32502ede96b9e0689e60711d22859c`, parent
`ca93c08ec908f2c00a2af48ba0a03d3543609dcf`, with `origin/main` at
`112f52fb0f903e0361fb6ec38199c081f63a93ed` and divergence zero behind/25 ahead.
Python 3.11.13 and 3.12.13 both used pyuvdata 3.2.1. The implementation changed
exactly 21 files (13 production and eight tests), with 3,847 insertions and 1,556
deletions. The two Jones initializer files and the migrated beam-solver test were
separately authorized and retain exact parent public API and lazy-import behavior.
No dependency, config, documentation, generated output, NSIDE, or later-tier scope
was present.

Source review confirmed one resolved `BeamSystem` supplies canonical antenna,
frequency, power, projection, lightcurve, footprint, source-metric, and renderer
data. Raw FITS/diameter inputs, independent loads, `_fits_beam_power_func`, first or
nearest fallbacks, permissive compatibility arguments, default browser work,
mutable public plans, and duplicate visualization models are absent. Planning and
publication remain fail-closed and atomic.

Independent regression-first probes found six bounded defect groups: membership
power incorrectly normalized against zenith rather than retaining native loaded
power; exact-horizon power was zeroed; plan and metric cross-fields admitted forged
or contradictory state; projection/contour/lightcurve models admitted ndarray
subclasses and non-finite or inconsistent axes/provenance; time/LST/UTC and source
grid relationships were insufficiently enforced; and renderer cleanup could mask a
primary publication failure. The initial external run was 15 passed/6 failed in
both Pythons, followed by tracked 3/3, 3/3, 2/2, and 1/1 failing selections.

Correction `8a9ab69b70ff8cd13e437df9611c4bc94a82b7bf`
(`fix(observability): correct Tier 3G planning contract`) changes only projection,
geometry, lightcurves, planner, the Bokeh renderer, and four existing Tier 3G unit
suites. It separates native membership power from visible-sky display
normalization, retains horizon evaluation, makes every public model exact, finite,
immutable, and cross-field coherent, and preserves the primary publication error
when cleanup also fails. Nine durable regressions cover the corrected boundaries.

The corrected 49-case external matrix passed in both Pythons with only eight
temporary pyuvdata fixture deprecations per run. It covered shared/distinct FITS
references, identical bytes at different paths, option-distinct loads, equal zenith
values with different fingerprints, heterogeneous analytic and mixed modes,
assignment ordering, exact identity and hostile subclasses, reordered/duplicate/
nearly equal sky channels, circular RA, asymmetric footprints, source metrics,
side-effect sentinels, atomic file races, linked ranges, replacement, and browser
ordering. An independent half-trace Jones oracle matched native power at zenith,
off axis, the horizon, and below it; off-zenith-peak display normalization reached
exactly one while membership retained native resolved-beam power.

The exact Tier 3G boundary collected 109 tests in each environment: 108 passed,
one established optional-Vivaldi-data skip, and two established warning events.
The Tier 3F boundary passed 123/123 on Python 3.11 and 120 passed/three established
optional-JAX skips on Python 3.12, without warnings. Full suites collected 2,744:
Python 3.11 reported 2,743 passed, one skip, and 26 warnings; Python 3.12 reported
2,737 passed, seven skips, and the same 26 warnings. There were no failures,
xfails, or xpasses. The warnings remain exactly one disjointness override, eight
FITS-unit events, 12 lossy HEALPix advisories, one numerical multiply, and four
Matplotlib figure-reuse events.

Ruff lint and the 285-file format check passed. Pyright remained at 3,698
diagnostics in each environment under the unchanged 4,600 ceiling; direct reports
for all 13 production modules decreased from 591 parent diagnostics to 259 and
reported zero diagnostic on Tier 3G-added lines. The YAMLs validated at 101/11/1
channels, the offline example completed at five antennas/15 baselines/two channels,
and a removed clean-source Sphinx build reproduced the classified 40-event
baseline. Import/API parity, whitespace, exact scope, ownership, fallback,
side-effect, collision, marker, later-tier leakage, and generated-artifact checks
passed; all temporary external probe trees were removed.

Nothing was pushed or published. Remote CI, physical GPU, non-macOS, mounted
optional data, and live network/registry behavior remain unobserved. Tier 3H.1,
Tier 3H.2, and Tier 3I were not started. `BEAM-001`, `BEAM-002`, `BEAM-003`,
`OBS-001`, and `OBS-002` remain **OPEN**; none is **DONE**. Tier 3H.1 is the next
authorized separate slice.

### 2026-07-24 Tier 3H.1 independent acceptance

**Decision: Tier 3H.1 is accepted after corrections.** The strict start gate
passed on clean `main` at implementation
`1905fa0d5e2453677d41a9aafec190d2d8766f16`, parent
`4c34f897d8ffdf67ed18b16b52069505f56b0ffd`, with `origin/main` at
`112f52fb0f903e0361fb6ec38199c081f63a93ed` and divergence zero behind/28
ahead. Python 3.11.13 and 3.12.13 both used pyuvdata 3.2.1.

The implementation changed exactly four production, three documentation, and six
test files. The additional `test_sky_prepare_options.py` and
`test_tier1h_documentation.py` edits are separately authorized and narrow: they
prove strict rejection of the two removed advisor fields and migrate only the
obsolete pending-runtime documentation assertion. No dependency, lockfile, config,
visibility solver, writer, observability, generated artifact, Tier 3H.2 deletion,
or Tier 3I work leaked in.

Independent scalar and real-healpy oracles confirmed the selected-baseline harmonic
product over every exact observation frequency, cross/auto/shared/mixed endpoint
semantics, deterministic tie order, analytic aperture support, conservative FITS
native-grid representation bounds, the exact factor-five limit, and the smallest
satisfying power-of-two NSIDE through 65536. Advice never mutates requested or
loaded NSIDE and performs no beam evaluation, FITS read, or diameter/FWHM
approximation.

Regression-first probes found three material defect groups. A fresh pure import
eagerly loaded device and network modules through `utils/__init__.py`; public
requirements admitted forged handler IDs and endpoint-kind/metric contradictions;
and derivation admitted duplicate/noncanonical baselines, invalid handler-frequency
ordering, or inconsistent loaded assignment order without uniformly raising its
owned typed error. The fresh-import test failed in both interpreters, hostile
requirement selections failed before correction, and four durable derivation tests
failed in both interpreters.

Correction `7d202a2f47c9e4870e9da57e0010362f2d9fd9c7`
(`fix(beam): correct Tier 3H.1 sampling contract`) changes only the explicitly
authorized `utils/__init__.py`, `utils/healpix.py`,
`test_beam_sampling.py`, and `test_healpix_utils.py`. It preserves all existing
utility exports lazily, records and validates exact endpoint handler kinds and
canonical IDs, enforces metric-kind consistency, validates detached canonical
baseline/loaded-state copies, rejects duplicates, and converts malformed canonical
state to `BeamSamplingDerivationError`. Five durable regressions cover the
discovered boundaries.

Corrected external probes passed 22 hostile-state and 28 science/ownership cases in
each interpreter; only six temporary-fixture pyuvdata deprecations appeared per
science run. NumPy/Numba/JAX result-provenance checks all passed under Python 3.11;
NumPy/Numba passed under Python 3.12 with its expected unavailable-JAX skip.
Lifecycle sentinels proved pre-sky fail-closed ordering, exact BeamSystem retention
after backend failure, post-sky cleanup, deterministic retry, no partial result,
exact warning text and ULP boundary behavior, and no NSIDE mutation.

Successful point, HEALPix, analytic, shared-FITS, per-antenna-FITS, and mixed runs
publish exactly one new metadata key, `beam_resolution`, equal to a fresh detached
JSON-safe loaded-state snapshot. No runtime evaluator, BeamSystem, UVBeam,
backend/device array, requirement, observability or renderer state, lock, logger,
callable, or mutable alias enters it. The old FWHM utilities/options and approximate
`1.22` logic are absent without aliases or suppression. README/API/beam-guide
claims match active science and do not claim automatic resampling, physical FITS
bandwidth, Tier 3H.2 cleanup, writer redesign, or whole-tier completion.

The exact Tier 3H.1 boundary passed 110/110 in both Pythons with no warnings or
skips. The removed-field/documentation boundary passed 50/50 in each. Tier 3G
regression collected 109 in each, passing 108 with its one established optional
data skip and two established warning events. Full suites collected 2,794:
Python 3.11 reported 2,793 passed/one skip/26 warnings; Python 3.12 reported 2,787
passed/seven skips/26 warnings. The skips are one unmounted optional-Vivaldi-data
case plus six unavailable-JAX cases on 3.12. The warning categories remain one
disjointness override, eight FITS-unit syntax, 12 lossy HEALPix, one numerical
multiply, and four Matplotlib figure-reuse events. There were no failures, xfails,
xpasses, Tier 3H.1 markers, or new warning categories.

Ruff lint and the 286-file format check passed. Pyright remained at 3,692
diagnostics in both environments under the unchanged 4,600 ceiling; direct checks
of the four implementation production modules reported 131 historical diagnostics
and zero on lines changed since the parent. YAML validation reproduced 101/11/1
channels, the offline example completed with five antennas/15 baselines/two
channels, and a removed clean-source Sphinx build reproduced the classified
40-event baseline. Import/API identity, whitespace, exact scope, removal,
ownership, metadata, no-I/O, marker, later-tier leakage, and generated-artifact
audits passed. All temporary trees were removed.

Nothing was pushed or published. Remote CI, physical GPU, non-macOS, mounted
optional data, and live network/registry behavior remain unobserved. Tier 3H.2 and
Tier 3I were not implemented. `BEAM-001`, `BEAM-002`, `BEAM-003`, `OBS-001`, and
`OBS-002` remain **OPEN**; none is **DONE**. Tier 3H.2 is now the next authorized
separate slice; Tier 3I remains unauthorized pending Tier 3H.2 acceptance.

### 2026-07-24 Tier 3H.2 pre-implementation scope correction

**Documentation-only correction; Tier 3H.2 was not implemented.** The live
`tests/unit/test_observability/test_overlay.py` contains
`test_beam_contours_drawn_when_present`, a conditional test of
`/Volumes/CrucialX8/beams/NF_HERA_Vivaldi_power_beam_nside128.fits` that skips with
`Vivaldi FITS not mounted` when the volume is absent. The accepted plan required
removing this mounted Vivaldi test in Section 31 and required no mounted-data test
dependency in Sections 33 and 43, but the former Tier 3H.2 exact file list omitted
the containing test file. Under Section 36, implementation therefore could not
satisfy the final contract without an unauthorized edit.

The plan now narrowly adds
`tests/unit/test_observability/test_overlay.py` to Tier 3H.2. That authorization is
only for removal of the mount-dependent test method and imports made unused solely
by its removal; deterministic offline observability overlay coverage remains, and
the test file is not a fifth module deletion. The four production module deletions
are unchanged. The dual-Python focused verification boundary now includes
`test_tier3_beam_cleanup.py`, `test_tier1h_documentation.py`, and
`test_overlay.py`.

No implementation, source, test, configuration, dependency, lockfile, generated
artifact, or other documentation change occurred, and no implementation test was
run. Historical Tier 3H.1 evidence remains accurate and unchanged. Historical
HERA/Vivaldi filenames and measurements may remain only when clearly labeled
historical, not as active supported behavior. No compatibility shim, fallback,
schema rewrite, solver change, advisor redesign, Tier 3I work, or Tier 4 through 8
work is authorized.

`BEAM-001`, `BEAM-002`, `BEAM-003`, `OBS-001`, and `OBS-002` remain **OPEN**; none
is **DONE**. Tier 3H.2 remains unimplemented and is now the next executable task.
Tier 3I remains unauthorized until Tier 3H.2 is separately implemented and
independently accepted; final issue closure remains exclusively owned by Tier 3I.

### 2026-07-24 Tier 3H.2 independent acceptance

**Decision: Tier 3H.2 is accepted after corrections.** The fail-closed start gate
passed on clean `main` at implementation
`bd1398dd4d06d9d1eb81e46bed56c6aa206c2a42`
(`refactor(beam): remove legacy beam surfaces`), parent
`e905bc39e97f77eaee13d5173d5f3984bd636c5d`. `origin/main` matched the
implementation, divergence was zero behind/zero ahead, and worktree, index, and
untracked state were empty. Python 3.11.13 and 3.12.13 both used pyuvdata 3.2.1.

The implementation changed exactly the 27 authorized paths: 22 modifications,
one added cleanup test, and four deletions, with 649 insertions and 2,498
deletions. The deleted files are exactly `analytic/composed.py`,
`analytic/plotting.py`, `beam/fits/__init__.py`, and `beam/fits/handler.py`.
Canonical `core/beam/fits.py` remains. No dependency, lockfile, configuration,
solver, writer, generated artifact, Tier 3I, or later-tier work leaked in.

Independent full-file review and fresh-process probes confirmed that all deleted
module/import-order combinations fail and that removed managers, handlers, Jones
wrappers, analytic wrappers, registries, package exports, imports, and
compatibility paths are absent. The retained API is exactly
`compute_u_beam`, `airy_voltage_pattern`, `sinc_voltage_pattern`,
`elliptical_airy_voltage_pattern`, `uniform_taper`,
`gaussian_taper_pattern`, `parabolic_taper`,
`parabolic_squared_taper`, `cosine_taper`,
`corrugated_horn_pattern`, `open_waveguide_pattern`,
`dipole_ground_plane_pattern`, `prime_focus_angle`, `cassegrain_angle`,
`compute_edge_angle`, and `compute_hpbw_numerical`, in that order and with exact
defining-object identity.

A separate 30-case matrix passed 30/30 in each interpreter. Independent
NumPy/SciPy oracles covered circular, rectangular, elliptical, taper, horn,
waveguide, dipole, reflector, edge-angle, and bracketing-root HPBW science, plus
scalar/array shape, dtype, ownership, signatures, exact wildcard exports, fresh
imports, all 20 deleted-import orderings, exact scope, mounted-path removal, and
runtime ownership.

The canonical `BeamSystem` remains the sole loaded runtime service through
simulator, point, HEALPix, observability, advice, and result provenance. The
private `_ResolvedBeamJones` adapter stays private; `ElementBeamJones` and
`DifferentialBeamJones` are unchanged; generic analysis/projection remains
usable. Only the mounted Vivaldi conditional test and its sole-use imports were
removed. Nine deterministic offline overlay, contour, track, and projection tests
remain without a replacement network, registry, repository-data, alternate-mount,
or optional-data dependency.

Review found two material in-scope defects. README falsely said observability
accepted uniform diameter arrays and rejected heterogeneous arrays, while the
canonical service supports heterogeneous assignments with an explicit reference.
The analytic initializer also added a `pyright: ignore` to a retained public
export. A strengthened documentation assertion and new no-suppression assertion
failed 2/2 before correction. An intermediate typed re-export also exposed two
Ruff findings before the final form.

Correction `7d880e54ec213fbc8b7b00c6bb42fe12e3719eca`
(`fix(beam): correct Tier 3H.2 cleanup contract`) changes only `README.md`,
`analytic/__init__.py`, `test_tier3_beam_cleanup.py`, and
`test_tier1h_documentation.py`. It states the exact loaded-`BeamSystem` reference
rule and preserves defining-object identity without type suppression. The
regressions pass in both interpreters, direct file Pyright is zero, and Ruff
passes.

Final focused tests passed 74/74 in both environments, with zero skips, xfails,
xpasses, or mounted dependencies and only two established healpy/Matplotlib
events. Full suites collected 2,818: Python 3.11 passed all 2,818 with no skips;
Python 3.12 passed 2,812 with exactly six unavailable-JAX skips across the JAX
backend, sky backend, spectral, visibility backend, and two Jones backend cases.
There were no failures, xfails, or xpasses. Both retained the same 26 warnings:
one disjointness override, eight FITS-unit syntax events, 12 lossy HEALPix
advisories, one numerical-multiply event, and four figure-reuse events.

Ruff passed and all 283 files passed formatting. Pyright reported 3,231 in each
environment under the unchanged 4,600 ceiling, with zero diagnostics across the
67 changed production lines in ten paths. The YAMLs validated at 101/11/1
channels; the forced-offline example completed at five antennas, 15 baselines,
two frequencies, and `(1, 2)` products. Whitespace, exact-scope, import identity,
deletion, residual, mounted-path, documentation, historical-HERA, suppression,
and generated-artifact audits passed. The clean-copy Sphinx 8.2.3 build reproduced
the accepted 40 events: 35 docutils/docstring, one HERA toctree, one theme option,
and three HERA highlighting, with no deleted reference or new category. All
temporary trees were removed.

The active documentation now truthfully describes the four strict modes, five
analytic variants, scalar FITS subset, canonical advice/provenance, current Jones
schema, and simulator lifecycle. It does not overclaim arbitrary FITS/full-Jones,
automatic NSIDE mutation/resampling, GPU FITS interpolation, writer redesign, or
whole-tier acceptance; migration and HERA/Vivaldi names are explicitly
historical.

Remote CI, physical GPU, non-macOS, live network, registry, and external-data
behavior remain unobserved. No PR, tag, release, or deployment was created; the
authorized `main` push follows only after this record is committed and the remote
gate passes. Tier 3I was not started. `BEAM-001`, `BEAM-002`, `BEAM-003`,
`OBS-001`, and `OBS-002` remain **OPEN**; none is **DONE**. Tier 3I is now the
next authorized separate task and final issue closure remains there.

### 2026-07-25 Tier 3I independent whole-tier acceptance

Tier 3 is independently accepted and `BEAM-001`, `BEAM-002`, `BEAM-003`,
`OBS-001`, and `OBS-002` are all **DONE**. This current closure supersedes, but
does not rewrite, the historical slice records above that correctly left all
five issues open.

The fail-closed review started on clean `main` at
`aa01145b534c44c6b33a7681c1d103216ebf4313`
(`fix(ci): restore cross-platform acceptance gates`), parent
`7eb057ec64f8a5f7a92d8cb178a84d7b87b3e9c1`. Local HEAD, `origin/main`, and
remote `refs/heads/main` matched, with zero divergence and no staged,
unstaged, or untracked path. The range from
`8045bb49956ac7f4c04063fb1f9fb9d5928d5d8c` through `aa01145b` is linear:
33 commits, zero merges, 99 paths, 27,261 insertions, and 5,581 deletions.

The earlier Tier 3I attempt was validly rejected because exact-SHA CI had not
passed: run `30102386325` at `7eb057ec` exposed Rich physical-line wrapping in
one Linux CLI assertion and a PyPI llvmlite 0.46.0 source-build failure during
the Intel macOS locked install, so Intel tests never ran. The bounded repair
changes only `pixi.toml`, `pixi.lock`, the CLI simulation test, and the release
metadata tests (373 insertions, 171 deletions). It preserves the CLI failure
contract, adds Conda-resolvable Numba `>=0.64,<0.67`, and enforces the full
environment/platform lock matrix and provenance without weakening workflows,
platforms, Python versions, locked installs, tests, timeouts, or quality gates.
Exact-SHA run `30151413894` then passed quality and all six OS/Python jobs,
including both Intel macOS locked installs and full non-slow test execution.
The successful jobs were quality (`89662306859`), osx-arm64/Python 3.11
(`89662306867`), osx-64/Python 3.12 (`89662306868`), linux-64/Python 3.12
(`89662306870`), linux-64/Python 3.11 (`89662306872`),
osx-arm64/Python 3.12 (`89662306878`), and osx-64/Python 3.11
(`89662306888`).

Independent source, history, and API review proved all four beam modes, all five
analytic variants, and all four public config constructors; exact tagged
assignment and deterministic hostile-assignment failures; one Simulator-local
`BeamSystem`; same-key deduplication with distinct scientific identities;
atomic retry; and immutable, owned, detached state and result provenance. The
pyuvdata 3.2.1 scalar-E-field contract proves snapshot/private-copy/hash/race
handling, identity basis, fixed X/Y feeds, azimuth/zenith-angle horizon coverage,
exact frequencies, normalization, unit bandpass, finite data, typed errors, and
complex64/complex128 preservation.

Both solvers use the canonical evaluator. Independent manual oracles confirmed
`J_p C J_q^H` for autos and crosses, endpoint conjugation, negative geometric
phase, and HEALPix half-trace unpolarized power across the accepted runtime
families and available NumPy/Numba/JAX backends. The canonical advisor uses
every selected baseline and exact frequency, including auto-only selections,
endpoint-product harmonic feature scale, factor five, typed failures, and safe
caps without mutating NSIDE. Observability is a sibling plan-before-render/output
pipeline with exact channel/reference handling, deterministic homogeneous
minimum-number default, mandatory explicit heterogeneous reference, canonical
snapshots/sweeps/drift scans/overlays, JSON-safe detached provenance, and no
premature file, plot, browser, network, registry, or mounted-data side effect.

External primary probes passed 79/79 on Python 3.11 and 78 passed/one
unavailable-JAX skip on Python 3.12, with 29 classified fixture/dependency
future events in each. A supplemental explicit boundary passed 8/8 in both
interpreters; it added every analytic config variant, manual auto/cross RIME,
auto-only factor-five advice, and snapshot/drift/overlay evaluator reuse.
It had no Python 3.11 warning and three upstream Healpy/Matplotlib pending
deprecations on Python 3.12. Both external trees were removed.

The fixed focused boundary collected 954: Python 3.11 passed 954, while Python
3.12 passed 951 with exactly three unavailable-JAX skips. Full collection was
2,830: Python 3.11 passed all 2,830 in 309.98 seconds; Python 3.12 passed 2,824
in 283.44 seconds with exactly six unavailable-JAX skips. Every focused and
full run had zero xfails and zero xpasses. Focused runs retained only two
Healpy/Matplotlib figure-reuse events. Each full run retained exactly 26 known
events: one disjointness advisory, eight FITS unit-syntax events, 12 lossy
HEALPix advisories, one numerical-multiply event, and four Healpy/Matplotlib
figure-reuse events. No new category appeared.

Ruff and all 283 formatting checks passed. Pyright 1.1.408 reported 3,225
diagnostics in both Pixi environments under the unchanged 4,600 ceiling.
Lock parsing proved Conda Numba/llvmlite selections for every environment and
supported platform, with no selected PyPI variant. The shipped YAMLs validated
at 101, 11, and one channel. The forced-offline example completed at five
antennas, 15 baselines, two frequencies, and `(1, 2)` product shapes.
Clean-copy Sphinx 8.2.3 succeeded at the accepted 40 events: 35
docutils/docstring, one HERA toctree, three HERA highlighting, and one theme
option. All temporary trees were removed.

Exact-range, whitespace, scope, import, deletion, legacy-residual,
mounted-Vivaldi, documentation, suppression, compatibility/fallback,
ignored-field, generated-artifact, and Tier 4 leakage audits passed. The
retained analytic surface is exactly the accepted 16 defining functions.
Obsolete managers, mutable registries, FITS handlers, analytic
composed/plotting surfaces, identity fallbacks, and mounted Vivaldi activation
are absent. No new skip, skipif, xfail, warning filter, `pyright: ignore`,
`type: ignore`, `noqa`, or broad exception hides Tier 3 behavior.

The final indivisible closure evidence is:

- **BEAM-001 — DONE:** every accepted config reaches one canonical assignment,
  Simulator evaluation, and both solvers; strict FITS failures are typed and no
  accepted input is ignored.
- **BEAM-002 — DONE:** one local `BeamSystem` replaces the legacy manager/raw-ID
  registry/fallback surfaces and proves deduplication, scientific distinction,
  atomic retry, and detached provenance.
- **BEAM-003 — DONE:** canonical selected-baseline/frequency advice proves
  endpoint-product science, autos, factor five, typed failures, caps, and
  nonmutation; the dictionary advisor is gone.
- **OBS-001 — DONE:** the renderer-neutral sibling planning architecture,
  optional-sky errors, evaluator sharing, and side-effect boundary are proven.
- **OBS-002 — DONE:** equivalence fingerprints, homogeneous defaults, explicit
  heterogeneous references, accepted reference forms, titles/provenance,
  snapshots, sweeps, drift scans, and overlays are proven.

Physical GPU execution, live network/registry/external-data operation, and
production deployment remain genuinely unobserved. Tier 4 was not designed or
implemented; the next possible task is only its separate design/governing gate.
No PR, tag, release, or deployment was created.

### 2026-07-25 Tier 4 design and governing gate

Tier 3 remains independently accepted and `BEAM-001`, `BEAM-002`, `BEAM-003`,
`OBS-001`, and `OBS-002` remain **DONE**. The Tier 4 design gate is complete.
`Tier4ResultOutputPlan.md` is the governing implementation specification for
the canonical observation-time grid, immutable result, solver cutover, safe
versioned HDF5, truthful summary JSON, Measurement Set, UVFITS, visualization,
and transactional output workflow.

The fail-closed gate started on clean `main` at
`bf544540d83fefef77feb157b060c046276a3c25`
(`docs(beam): accept Tier 3 integration`), parent
`aa01145b534c44c6b33a7681c1d103216ebf4313`. After fetch, local HEAD,
`origin/main`, and remote `refs/heads/main` matched with zero divergence and no
staged, unstaged, or untracked path. Exact-head GitHub Actions run
`30165680809` was a successful push run; quality and all six locked
OS/Python jobs succeeded. The two environments retained Python 3.11.13 and
3.12.13, pyuvdata 3.2.1, Pyright 1.1.408, lock format v7, both environments,
and all three locked platforms.

Source-first review confirmed the six live OUT defects and every result/output
call path. Offline temporary pyuvdata and h5py probes ran in both environments
and were removed. They proved explicit standard-format phase projection,
complex64 Measurement Set storage, complex64/complex128 UVFITS preservation,
safe selected auto/cross handling, and h5py complex/string/dimension/atomic
behavior. The required focused boundary collected 279: Python 3.11 passed all
279; Python 3.12 passed 278 with one unavailable-JAX skip. There were no
failures, xfails, or xpasses.

Ruff passed, all 283 files passed formatting, and Pyright reported 3,225
diagnostics in both environments under the unchanged 4,600 ceiling. The three
YAMLs validated at 101, 11, and one channel. The offline example completed at
five antennas, 15 baselines, two frequencies, and `(1, 2)` correlation-product
shapes. Clean-copy Sphinx 8.2.3 succeeded with the accepted 40 events: 35
docutils/docstring, one HERA toctree, one theme option, and three HERA
highlighting events. Whitespace passed and no task-specific temporary output
remained.

This was documentation-only design work. No Tier 4 production code, durable
implementation test, fixture, configuration, dependency, lockfile, CI,
generated artifact, or later-tier behavior was changed. `OUT-001`, `OUT-002`,
`OUT-003`, `OUT-004`, `OUT-005`, and `OUT-006` all remain **OPEN**. Tier 4A
remains unauthorized. The next task is an independent review and acceptance of
`Tier4ResultOutputPlan.md`, not implementation.

### 2026-07-26 Tier 4 design independent acceptance

**The Tier 4 result/output design is independently accepted after bounded
corrections.** `Tier4ResultOutputPlan.md` remains the governing implementation
specification. This current status supersedes, but does not rewrite, the
historical design-gate paragraph above that correctly left Tier 4A
unauthorized pending this review.

The fail-closed review began on clean `main` at design commit
`d468f203989bbcd9f4a00b42f658fa669d5bd0be`
(`docs(output): plan Tier 4 result integration`), parent
`bf544540d83fefef77feb157b060c046276a3c25`. Local, tracking, and remote main
matched with zero divergence. Exact-head GitHub Actions run `30168421011` was a
successful push run; quality and all six locked OS/Python jobs succeeded.

Independent source review confirmed the six live OUT defects and the plan's
current-state trace. Dual-Python external probes confirmed the normalized
half-open time grid, leap-second two-part JD, first-time-zenith standard
projection, canonical/AIPS correlation reorder, c128-to-c64 Measurement Set
conversion, c64/c128 UVFITS preservation, and declared HDF5 dtype/filter/string
combinations. A macOS primitive probe confirmed exclusive rename, directory
swap, and directory fsync behavior.

The review found material planning defects before acceptance. pyuvdata 3.2.1
UVFITS has no `clobber` keyword and requires multi-channel spacing to match
channel width, not merely remain below it. The original slice lists also
omitted first-use CLI, configuration, sample, documentation, export, and
workflow paths; prematurely depended on 4F's high-level save API in 4D; and
left unsafe HDF5/MS surfaces active until 4H. Correction
`a42b96e117d66496cde75cfad09979719fa0d494`
(`docs(output): correct Tier 4 design`) changes only the governing plan and
records the failing evidence. The corrected structured audit reports all 47
sections, nine slices, exact new/existing path state, zero ownership gaps, no
unsafe active-path gap, and no normative ambiguity.

The exact focused boundary passed 279/279 on Python 3.11.13 and 278 with one
unavailable-JAX skip on Python 3.12.13, with zero failures, xfails, or xpasses.
Ruff and all 283 formatting checks passed. Pyright 1.1.408 reported 3,225
diagnostics in both environments under the unchanged 4,600 ceiling. YAML
validation retained 101/11/1 channels; the offline example retained five
antennas, 15 baselines, two frequencies, and `(1,2)` product shapes. Clean-copy
Sphinx 8.2.3 succeeded with the established 40 classified events. No generated
or task-specific temporary artifact remains in the repository.

This acceptance changes planning records only. No Tier 4 production behavior,
durable implementation test, fixture, configuration, dependency, lockfile, CI,
example, or generated artifact changed, and Tier 4 implementation has not
started. `OUT-001`, `OUT-002`, `OUT-003`, `OUT-004`, `OUT-005`, and `OUT-006`
all remain **OPEN**; none is **DONE**.

Tier 4A is now the only next authorized separate slice and remains limited to
its three test-only characterization files. Tier 4B and all later slices remain
unauthorized until Tier 4A is implemented and independently accepted. Physical
GPU, distinct complex256, live network/registry/external-data, non-macOS
dependency probes, and direct Linux atomic-primitive execution remain
unobserved. No PR, tag, release, or deployment was created.

### 2026-07-26 corrected Tier 4A independent acceptance

**Tier 4A is independently accepted after bounded corrections.** This record
preserves the earlier valid rejection: the first warning classifier matched
message substrings without warning classes, accepted the unrelated
`UserWarning("somewhere arbitrary output changed")`, and the plan incorrectly
implied that both projected MS and UVFITS rejected Python-list polarization
arrays. The review correctly stopped the remaining probes and common gates at
that material failure.

The accepted linear candidate is
`ed7695cc69b5ff66921021fea6ff8a33ecdca8f7`
(`test(output): characterize Tier 4 dependencies`),
`d8b8962010f16aecc3157ec58147888bc4b81789`
(`test(output): fail closed on dependency warnings`), and
`4289b41a6380e6c38c68aee92596d1e850b866c1`
(`docs(output): correct polarization characterization`), after
`e65479e06f2405680b014bc29ae9b2252e374d46`. Review began on clean `main` at
`4289b41`, zero behind and two ahead of both `origin/main` and remote `main` at
`ed7695cc`; no staged, unstaged, or untracked path existed. Exact-head push run
`30174270691` for the original commit succeeded in all seven jobs.

The original commit changes only the three characterization modules.
`d8b8962` changes only the pyuvdata characterization: exact warning
class/complete-message tuples, unknown-warning assertion, `finally`
classification on writer failure, exact interpreter-specific sets, and no
baseline-conjugation allowance. `4289b41` changes only
`Tier4ResultOutputPlan.md` and records that projected MS accepts the list while
UVFITS requires integer-array normalization. The target shared adapter still
normalizes before both writers. The full corrected range changes no production
source, config, dependency, lockfile, workflow, example, CI, or generated
artifact.

All twelve external warning cases passed. Exact known warnings classified;
wrong class, substring lookalike, prefixed/suffixed message, unknown, and
known-plus-unknown cases failed closed. Empty and two-known sets were exact.
Unknown warnings failed on writer success and failure. On failure, the
classifier assertion deliberately superseded the writer error while retaining
it as `__context__`; no-warning failures preserved the original error, and
known warnings during collision were classified before the collision
propagated. Python 3.11 sets were calibrated MS `{}`, uncalibrated MS
`{uncalibrated-unit}`, and UVFITS `{}`. Python 3.12 added only
`numpy-where-without-out` to each MS set; UVFITS remained `{}`.

Fresh dual-environment pyuvdata 3.2.1 probes proved list retention in
`UVData.new`, unchanged projected-MS acceptance, four-correlation read-back,
the UVFITS integer-scalar-index failure, successful integer-ndarray
normalization, malformed-polarization rejection, BLT/identity/time/width/flag/
sample contracts, Astropy UVWs, NumPy phase, c128-to-c64 MS conversion,
c64/c128 UVFITS preservation, collision behavior, rejection matrices, exact
warnings, handle closure, and cleanup. Fresh h5py 3.14.0/3.16.0 probes proved
c64/c128 and dimension-label preservation, variable and fixed string
boundaries, object-to-fixed failure, roughly one-MiB values, raw malformed
shape/unknown-version acceptance, replacement-inode behavior, and cleanup.

Different-valued RadioSim probes confirmed the point/HEALPix/save/plot time
count split; aliased mutable result identity and products; current HDF5
I/XX choice, structural correlation omission, dtype promotion, group names,
missing schema, flattened metadata, and direct final write; incomplete JSON;
unsafe arithmetic parsing; directory-before-error; scalar-time MS mismatch
with an independent selection oracle; and both UVFITS rejection surfaces.

Focused characterization passed 32/32 in each interpreter and the pyuvdata
module passed 13/13 in each, with no skips, failures, xfails, xpasses, or
unclassified warnings. The full non-slow suites collected 2,862: Python 3.11
passed 2,862; Python 3.12 passed 2,856 with exactly six unavailable-JAX skips.
Both had zero failures/xfails/xpasses and the established 26 warnings: one
disjointness, eight FITS-unit, 12 lossy HEALPix, one numerical-multiply, and
four figure-reuse events.

Ruff passed, 286 files passed formatting, and both Pyright gates reported 3,225
under the unchanged 4,600 ceiling. YAML validation remained 101/11/1 channels;
the forced-offline example remained five antennas, 15 baselines, two
frequencies, and `(1,2)` products. Clean-copy Sphinx 8.2.3 succeeded with the
established 40 events (35 docutils/docstring, one HERA toctree, three HERA
highlighting, one theme option). Whitespace passed.

Exact scope and suppression audits found no skip/xfail, warning/type/lint
suppression, broad exception, new network or registry access, mounted path,
physical GPU, prompt/browser, persistent artifact, environment/CWD mutation,
test-order dependency, production change, or Tier 4B leakage. The two local
`simplefilter("always")` calls expose all warnings; the only `ignore` match is
inside the exact upstream warning message. Required Sphinx retained its
existing intersphinx inventory access with no new warning category. All
temporary external probe and clean-copy trees were removed.

Physical GPU execution, non-macOS local dependency execution, live
registry/external-data behavior, and behavior outside the two locked
interpreters remain genuinely unobserved. Production remains unchanged.
`OUT-001`, `OUT-002`, `OUT-003`, `OUT-004`, `OUT-005`, and `OUT-006` remain
**OPEN**.

Tier 4B is now the next authorized separate implementation slice. It was not
implemented here. Tier 4C and all later slices remain unauthorized. No PR,
tag, release, or deployment was created.

### 2026-07-27 retrospective Tier 4B independent acceptance

**Tier 4B is independently accepted after one bounded correction.** This
record repairs a sequencing conflict without rewriting history. Tier 4B was
implemented at `a54f0a86692a28e3587730af7bb132cd857c37c4` after accepted Tier
4A, but Tier 4C commit
`ee72153f785619d42b7f1f1405f680b1f647b788` was created before the required
independent Tier 4B checkpoint. The user explicitly authorized this
retrospective review. Future slices remain subject to the original rule: no
slice starts from an unaccepted predecessor.

Original exact-SHA GitHub Actions run `30185991070` succeeded in all seven
jobs. Detached review at `a54f0a8` passed the original focused boundary
283/283 in both Python 3.11.13 and 3.12.13. The actual 41-path implementation
exceeded the prospective 33-path list only through eight necessary explicit
channel-width fixture migrations, now enumerated in
`Tier4ResultOutputPlan.md`.

The review found a loaded-result identity hole: internally consistent
selection metadata could reference antenna ID 99 when the instrument contained
only IDs 0 and 1. Regression-first correction
`25e6a935d7747ee040e2facf795ec3f6120d7f1a`
(`fix(result): validate loaded result identity`) validates instrument,
selection, beam, backend, and solver snapshots and their coherence. The
corrected Tier 4B boundary passed 138/138 in both interpreters, and Tier 4A
still passed 31/31 in both.

This corrected acceptance authorizes review of the already-existing Tier 4C
commit; it does not claim Tier 4C was authorized when created. Production
writers remain deferred, and `OUT-001` through `OUT-006` remain **OPEN**.

### 2026-07-27 corrected Tier 4C independent acceptance

**Tier 4C is independently accepted after bounded corrections.** Review began
from clean `main` at original Tier 4C commit
`ee72153f785619d42b7f1f1405f680b1f647b788`, parent `a54f0a8`. Its exact
30-path implementation range and original seven-job GitHub Actions run
`30189180484` were independently confirmed.

Three material defects were corrected:

- `65ee1b648b4371c697c0329082712395b6ba16e5`
  (`fix(result): require explicit solver backend`) removes silent
  `backend=None` fallback and enforces the canonical frequency boundary.
- `8e51749e3de1f32eba598db5a611c5e7acadd078`
  (`fix(result): publish after successful reporting`) makes result publication
  atomic across late progress/success-rendering failure and adds deterministic
  timing/lifecycle attacks.
- `f86c2727bdb6265102034407198715da7c7549f7`
  (`docs(result): correct output availability`) removes false current-output
  claims from README and two active Sphinx pages and adds a durable regression.
  The two documentation pages were a user-authorized narrow expansion beyond
  the prospective path list.

An external NumPy/Astropy oracle with different antenna IDs, mixed auto/cross
baselines, unequal channel widths, nondivisible time sampling, off-zenith
sources, full Stokes, and heterogeneous complex 2x2 Jones matrices verified
the coherency transform, fringe phase, Jones multiplication, ordering,
baseline direction, conjugation, dtypes, flags, weights, and point/HEALPix
parity. Point maximum errors were `2.95e-16` and `2.38e-16`; scalar HEALPix,
polarized HEALPix, and one-pixel parity errors were at most `3.91e-08`,
`6.36e-14`, and `8.53e-09`. A fresh recording backend proved zero transfers
on solver/factory failure and one transfer per successful high-level result.

Atomic lifecycle, immutable result ownership, snapshot/hash, deterministic
timing, canonical time-grid, singular `result`, explicit backend, typed
save/plot failure, and CLI preflight boundaries all passed adversarial checks.
No active plural result, fifth correlation, compatibility shim, result dict,
duplicate solver transfer, or premature writer dispatch remains.

Tier 4C focused tests collected 228 in each interpreter: 227 passed and one
unavailable-JAX skip. Tier 4B passed 138/138 and Tier 4A passed 31/31 in each.
The complete non-slow suite collected 2,904 in each: 2,898 passed, six
unavailable-JAX skips, and the established 26 warnings on both Python 3.11.13
and 3.12.13, with no failures, xfails, or xpasses.

Ruff passed; all 293 files passed formatting; Pyright 1.1.408 reported 3,121
diagnostics in each environment under the unchanged 4,600 ceiling. The three
YAMLs validated at 101/11/1 channels. The forced-offline example returned
`(1,15,2,4)` with no output. Notebook static integrity, fresh-process import
isolation, clean-copy Sphinx with the established 40 events, whitespace,
scope, artifact, and suppression gates passed. Original exact-SHA GitHub
Actions runs for Tier 4B and Tier 4C were all green. The exact final acceptance
SHA remains subject to the post-push seven-job release gate.

Physical GPU execution, local JAX execution, non-macOS local execution, live
network/registry/external-data behavior, and dynamic notebook execution remain
genuinely unobserved. `OUT-001` through `OUT-006` remain **OPEN**. Tier 4 as a
whole is not accepted.

Tier 4D is now the next authorized separate implementation slice. It was not
started during this review. No PR, tag, release, or deployment was created.
