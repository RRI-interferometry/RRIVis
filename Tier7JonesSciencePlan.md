# Tier 7 Advanced Jones Science Plan

## 1. Identity, status, and governing sources

**Status: design accepted 2026-08-01 by independent review, with one bounded
factual correction applied to Section 20.3 (`docs(jones): correct Tier 7
design`, `5578cc3`) and no decision changed; see `Fix.md`'s 2026-08-01
"Tier 7 design acceptance" note for the full record. Authored 2026-08-01.
Baseline `main` at `ac4fe41` (`docs(sky): document the extragalactic
point-source loader`). Slice **7A is accepted** (2026-08-01, independent
review; three further bounded factual corrections applied to Sections 5.1,
5.6, and 33.2, `docs(jones): correct Tier 7 design`, `79d392d`, no decision
changed; see `Fix.md`'s 2026-08-01 "Tier 7A independent acceptance" note for
the full record). Slice **7B is accepted** (2026-08-01, independent review;
four bounded factual corrections to Sections 13.2, 14.1, 23, and 33.2/34
already applied by the implementer, `docs(jones): correct Tier 7 design`,
`ca02f00`, and ratified by this review with no further correction and no
decision changed -- including the two disclosed HEALPix-only numerical
deltas from closing D4/D9, ruled authorized-and-correct; see `Fix.md`'s
2026-08-01 "Tier 7B independent acceptance" note for the full record).
Slice **7C is accepted** (2026-08-01, independent review; one bounded
factual correction applied to Section 37 criterion 1, plus ratification of
the six bounded corrections the implementer had already applied to Sections
33.2 and 34 (`docs(jones): correct Tier 7 design`, `68458da`), including the
`term_status` base default of `"planned"` rather than Section 23's literal
`"implemented"`, ruled correct on independent analysis; no decision changed;
see `Fix.md`'s 2026-08-01 "Tier 7C independent acceptance" note for the full
record). Slice **7D is accepted** (2026-08-01, independent review; no
factual correction needed beyond the eleven bounded 7D corrections the
implementer had already applied to Sections 33.2 and 34
(`docs(jones): correct Tier 7 design`, `76929e8`), including the
`chain_terms`-excludes-`H`/`C`/`E` correction, ratified by this review with
no further correction and no decision changed; see `Fix.md`'s 2026-08-01
"Tier 7D independent acceptance" note for the full record). Slice **7E is
accepted** (2026-08-01, independent review; two bounded factual corrections
applied to Section 24 (blessing the reduced, feedless R5 message `X` actually
raises) and Section 34's 7G file list (adding `src/radiosim/core/jones/base.py`,
omitted despite that file's own docstring committing 7G to edit it), plus
independent re-derivation of the Section 20.3 IXR correction from first
principles (confirmed correct); no decision changed; see `Fix.md`'s 2026-08-01
"Tier 7E independent acceptance" note for the full record). Slice **7F is
accepted** (2026-08-01, independent review; two bounded corrections applied
directly by the reviewer -- Section 41 Q4 answered from the reviewer's own
numeric probe (bit-identical `E`/`P` swap on both the analytic and FITS beam
paths, since the 7F commits carried no checked-in test for it), and Section 39
gains risk row 11 recording that `jones.P`'s mount types are unreachable from a
layout-file or known-telescope `instrument:` source (only pyuvdata carries
`mount_type`), routed informationally to a future instrument-config tier; no
physics, chain-order, or refinement decision changed; see `Fix.md`'s
2026-08-01 "Tier 7F independent acceptance" note for the full record). Slice
**7G is accepted** (2026-08-01, independent review; no factual correction
needed to this document beyond ratifying the two bounded corrections the
implementer had already applied to Sections 20.8 (the `R -> R^T` Faraday
factor and the wide-field-cancellation correction) and Sections 21.2/21.3/24/26.1/34
(`minimum_elevation_deg` on `Z`, R13's evaluation-not-resolution stage, and
the seven forced writable-file additions); the Faraday sign and the
antenna-common cancellation were both independently re-derived from first
principles and confirmed correct; one bounded documentation-only correction
applied directly by the reviewer outside this plan (two stale "2.28 m"
Saastamoinen docstring values corrected to the tested 2.3070 m,
`docs(jones): correct the Saastamoinen ZHD docstring value`, `36823ae`); no
science, chain-order, or refinement decision changed; see `Fix.md`'s
2026-08-01 "Tier 7G independent acceptance" note for the full record). Slice
**7H is accepted** (2026-08-02, independent review; no bounded correction
required beyond ratifying the two corrections the implementer had already
applied to Sections 15.1, 20.10 and 20.11 (`docs(jones): correct Tier 7
design`, `cb8c87f`) -- the Hadamard-neutral-element ruling (the all-ones
matrix, not `I2`, with `I2` left legal-but-consequential and no additional
guard required beyond R7 and the corrected documentation) and the
residual-delay/fringe-rate geometry (`tau_res` measured from the phase
centre and the ENU fixed-zenith fringe-rate expression), both independently
re-derived from first principles and confirmed correct; no decision changed;
see `Fix.md`'s 2026-08-02 "Tier 7H independent acceptance" note for the full
record). Slice **7I is accepted** (2026-08-02, independent review; the
geometry, Ruze-convention, and accepted-YAML corrections the implementer had
already applied to Section 19.2, plus the six forced Section 34 additions,
all ratified after independent re-derivation; one further bounded correction
applied directly by this review -- a seventh forced Section 34 addition,
`tests/unit/test_core/test_beam_fits.py`, whose own hardcoded `runtime.__all__`
pin the slice's mandated public Ruze functions make wrong -- and one implementer
claim rejected: the assertion that `ea0e98c`'s true Sphinx baseline is 18, not
16, does not survive a clean detached-worktree rebuild (16 warnings,
byte-identical text, at both `ea0e98c` and `e415624`); the established 16
baseline stands, and the 18 the implementer measured is the same
`docs/superpowers/`-contamination artifact 7H already diagnosed for an
in-tree build. No decision changed; see `Fix.md`'s 2026-08-02 "Tier 7I
independent acceptance" note for the full record). Slice **7J is
authorized**.**

This document is the governing implementation specification for Tier 7 of the
RadioSim remediation program, defined by [`Fix.md`](Fix.md) Section 16
("Tier 7 — Advanced Jones and m-mode science"). It closes, or explicitly
redisposes, the three register rows routed to Tier 7:

- `SCI-001` — "Most Jones classes are public identity-returning stubs"
  (`Fix.md` Section 5, detail in Section 7.5);
- `SCI-002` — "Spherical-harmonic/m-mode mode is advertised but unimplemented"
  (`Fix.md` Section 5, detail in Section 7.6);
- `SCI-003` — "Advanced beam-physics TODOs remain" (`Fix.md` Section 5).

Governing sources, in precedence order:

1. `Fix.md` Section 4 (governing decisions: pre-v1 API policy, the truthfulness
   rule, explicit precedence, scientific features require scientific tests);
2. `Fix.md` Section 16 (the Tier 7 workstreams A–E, the seven rules for every
   Jones implementation, cross-implementation validation, exit criteria);
3. `Tier5ReceptorFeedPlan.md` (chain order, receptor conventions, the
   polarization-basis contract, the corrected coherency convention);
4. `Tier6HybridRuntimePlan.md` (backend parity and tolerance discipline, the
   compiled-kernel boundary, solver worker/partition structure, provenance and
   fingerprint discipline, benchmark honesty rules);
5. `CLAUDE.md` (project commands, Implementation Status, terminology split);
6. `AGENTS.md` and `docs/contributing.rst` (pre-v1 policy in the contributor
   surface).

Every characterization statement in Sections 5 through 7 is cited to a file and
a line range **true at `ac4fe41`**. Where a citation names a range, the range is
the enclosing definition, not an approximation.

### 1.1 Baseline note — the six commits after Tier 6 acceptance

Six commits landed on `main` after the Tier 6J re-run acceptance
(`bd38a59..ac4fe41`):

| Commit | Subject | Tier 7 relevance |
|---|---|---|
| `b587301` | `ci: bump actions/checkout to v5` | none |
| (digest harvest) | AVX-512 runner-class digest observation-set additions | none; the observation-set scheme is final per `Tier6HybridRuntimePlan.md` Section 42 |
| `0b89a16` | `feat(sky): add Gervasi 2008, Mandal 2021, and Intema 2017 dN/dS presets` | none |
| `1a900bf` | `feat(sky): add extragalactic_point_sources loader (Mittal et al. 2024, isotropic)` | supplies a large-`N` point sky useful as a Tier 7 performance and smearing fixture |
| `e9fc22a` | `feat(sky): add 2PACF clustering and streamed HEALPix output to extragalactic loader` | same |
| `ac4fe41` | `docs(sky): document the extragalactic point-source loader` | none |

These are **accepted baseline**. Tier 7 does not modify the extragalactic
loader family, the dN/dS presets, the clustering path, or the digest
observation sets. This plan characterizes what exists; it proposes no change to
any of it. No defect was found in them during this gate.

### 1.2 What this plan is not

It is not a Tier 8 documentation rewrite, not a release, and not a performance
program. `PERF-001` (accelerator performance, hardware-gated) and `SKY-002`
(composite-recipe network metadata, routed pre-Tier-8) are **not** absorbed
here; `Fix.md` Section 16 does not ask for them and
`Tier6HybridRuntimePlan.md` Section 42 explicitly forbids folding `PERF-001`
into Tier 7's Jones workstreams. They remain as filed.

## 2. Design-only authority

This gate produced exactly two changes:

1. this file, `Tier7JonesSciencePlan.md`, new at the repository root;
2. a short dated current-status note appended to `Fix.md`.

No production code, test, fixture, configuration, dependency, lockfile, CI
workflow, documentation page, or generated artifact was changed. No `Fix.md`
Section 5 register row was edited and no prior acceptance record was rewritten.
`SCI-001`, `SCI-002`, and `SCI-003` remain `ROADMAP` exactly as recorded.

Read-only probes run during this gate: `git log`, `git status`,
`git check-ignore`, and text searches over `src/`, `tests/`, `docs/`,
`configs/`, `examples/`, `pixi.toml`, and `pyproject.toml`. No test suite, no
lint, no formatter, no type checker, no documentation build, and no network or
remote operation of any kind was run.

Tier 7A remains **unauthorized** until this plan is independently accepted.

## 3. Tier 0–6 dependency and acceptance state

| Tier | State at `ac4fe41` | What Tier 7 inherits and must not break |
|---|---|---|
| 0 | accepted | version truth, CI, pre-v1 policy, type-debt ceiling |
| 1 | accepted | one strict `load_config()` pipeline; unsupported non-default fields are exhaustively classified and rejected before side effects (`CFG-003`) |
| 2 | accepted | frozen `ResolvedInstrument`, `SolverInstrumentView`, canonical baselines, `instrument_sha256` |
| 3 | accepted | one canonical `BeamSystem`; **no second beam runtime may be introduced** |
| 4 | accepted | one authoritative time grid; `SimulationResult`; HDF5/summary/MS/UVFITS contracts; `scientific_sha256` |
| 5 | accepted | canonical chain order, the corrected coherency `C[0,1] = (U+iV)/2`, `PolarizationBasis`, `ResolvedReceptorSet`, the scalar-`E` constraint, `C`/`H` real and unitary |
| 6 | accepted | backend parity (Dask bit-identical, JAX-CPU `rtol=1e-12`), per-time block assembly, exactly one compiled kernel, typed `execution.solver` worker policy, hybrid sky |

Two Tier 5/Tier 6 decisions are **explicitly reopened by Tier 7**, each with its
own design section and its own justification:

- the placement of `P` in the canonical chain order (Section 12) — Tier 5
  Section 19.1 fixed it while `P` was an identity stub and while the order was,
  in its own words, "currently unobservable";
- the `JonesTerm` evaluation contract (Section 13) — the scalar
  `source_idx: int` signature cannot carry HEALPix-scale direction batches.

Nothing else accepted in Tiers 0–6 is reopened.

## 4. What Tier 7 will not claim

Stated up front, so that no slice can drift into claiming it:

- **No accelerator claim.** Tier 7 adds no GPU or TPU number, no speedup claim,
  and no benchmark statement that does not cite a committed record under
  `output/benchmarks/reference/`. The locked JAX is CPU-only
  (`pixi.toml:86-88`). `PERF-001` is untouched.
- **No real-world data ingestion.** No IONEX/GPS total-electron-content maps, no
  radiosonde or weather-model profiles, no measured bandpass or D-term tables
  from an observatory archive, and no ionospheric or tropospheric time series.
  Every propagation term is driven by **explicit configured parameters** with
  documented units. This is why `GPSIonosphereJones` is deleted rather than
  implemented (Section 10).
- **No stochastic screens.** No Kolmogorov phase screen, no turbulent
  troposphere realization, no random gain draw that is not a documented,
  seeded, reproducible function of configuration. Every implemented effect is a
  deterministic function of `(config, antenna, direction, frequency, time)`.
- **No calibration or solving.** Tier 7 implements *forward-model corruption*.
  It does not solve for gains, bandpasses, delays, D-terms, or fringes. This is
  why `FringeFitJones` is deleted (Section 10).
- **No imaging operators.** No gridding, no FFT, no W-projection kernels, no
  A-projection. This is why `WProjectionJones` is deleted (Section 10).
- **No second beam runtime.** `core/beam/`'s `BeamSystem` remains the single
  owner of antenna voltage response (Tier 3). Tier 7 adds no parallel beam
  evaluation path, which is why `ElementBeamJones`, `ArrayFactorJones`, and
  `DifferentialBeamJones` are deleted rather than implemented (Section 10).
- **No non-scalar E.** The accepted E-Jones remains a scalar complex voltage on
  the 2×2 diagonal (`docs/user_guide/jones_matrices.rst:45-48`). General
  polarized BeamFITS, Ludwig-3 cross-polarization, and beam squint stay outside
  the accepted subset; Section 19 records their scope and files them forward.
- **No m-mode solver.** Section 18 descopes Workstream E and closes `SCI-002`
  by **absence from accepted config**, which `Fix.md` Section 16's exit
  criterion explicitly permits. The solver itself is filed as a successor
  roadmap row.
- **No mutual coupling, no near-field, no aberrations.** Section 19.
- **No claim that any implemented term has been validated against an
  observatory pipeline** unless the comparison is recorded as a committed
  evidence artifact naming the reference code, its version, and the compared
  numbers (Section 29).

## 5. Current source inventory

### 5.1 The exported Jones surface

`src/radiosim/core/jones/__init__.py:76-135` declares `__all__` with exactly
**43 names**: three base names (`JonesTerm`, `JonesChain`,
`JonesBaselineTerm`) and **40 concrete term classes**. All 40 are lazily bound
through `_LAZY_EXPORTS` (`:138-202`) and resolved by `__getattr__` (`:205-212`).

`CLAUDE.md` states "46 exported classes (see `core/jones/__init__.py`)". The
true count of `__all__` entries is 43. The three-name discrepancy is a
documentation drift, not a code defect; it is recorded as defect **D0** and
fixed in the same slice that rewrites the Jones documentation.

Of the 40 concrete classes, exactly **three** implement real physics:

| Class | File | Status | Evidence |
|---|---|---|---|
| `GeometricPhaseJones` (`K`) | `core/jones/geometric.py:17-198` | real physics, **never used by any solver** | Section 5.4 |
| `ReceptorConfigJones` (`C`) | `core/jones/receptor.py:343-379+` | real physics (Tier 5) | `Fix.md` Tier 5I acceptance |
| `BasisTransformJones` (`H`) | `core/jones/receptor.py` | real physics (Tier 5) | `Fix.md` Tier 5I acceptance |

The E term used by the solver is **not** an exported class: it is the private
`_ResolvedBeamJones` adapter constructed inside
`src/radiosim/core/visibility.py:855-862`, wrapping `BeamSystem`.

The remaining **37 classes are identity stubs**. Every one of them has a
`compute_jones` whose entire body is

```python
xp = backend.xp
return xp.eye(2, dtype=np.complex128)
```

at these exact locations:

| Term | Class | Identity return |
|---|---|---|
| G | `GainJones` | `core/jones/gain.py:50-61` |
| G | `TimeVariableGainJones` | inherits `gain.py:50-61` (`:64-69`) |
| G | `ElevationGainJones` | inherits `gain.py:50-61` (`:72-90`) |
| B | `BandpassJones` | `core/jones/bandpass.py:44-55` |
| B | `PolynomialBandpassJones` | inherits (`bandpass.py:58-62`) |
| B | `SplineBandpassJones` | inherits (`bandpass.py:65-69`) |
| B | `RFIFlaggedBandpassJones` | inherits (`bandpass.py:72-76`) |
| D | `PolarizationLeakageJones` | `core/jones/polarization_leakage.py:38-49` |
| D | `IXRLeakageJones` | inherits (`:52-65`) |
| D | `MuellerLeakageJones` | inherits (`:68-72`) |
| D | `BeamSquintLeakageJones` | inherits (`:75-79`) |
| P | `ParallacticAngleJones` | `core/jones/parallactic.py:57-68` |
| P | `FieldRotationJones` | inherits (`:71-75`) |
| P | `VLBIFeedRotationJones` | inherits (`:78-88`) |
| Z | `IonosphereJones` | `core/jones/ionosphere.py:53-64` |
| Z | `TurbulentIonosphereJones` | inherits (`:67-75`) |
| Z | `GPSIonosphereJones` | inherits (`:78-82`) |
| T | `TroposphereJones` | `core/jones/troposphere.py:50-61` |
| T | `SaastamoinenTroposphereJones` | inherits (`:64-76`) |
| T | `TurbulentTroposphereJones` | inherits (`:79-83`) |
| T | `TroposphericOpacityJones` | inherits (`:86-106`) |
| F | `FaradayRotationJones` | `core/jones/faraday.py:54-65` |
| F | `DifferentialFaradayJones` | inherits (`:68-81`) |
| W | `WPhaseJones` | `core/jones/wterm.py:62-73` |
| W | `WProjectionJones` | inherits (`:76-88`) |
| Txy | `WidefieldPolarimetricJones` | `core/jones/wterm.py:116-127` |
| Ee | `ElementBeamJones` | `core/jones/element_beam.py:59-70` |
| a | `ArrayFactorJones` | `core/jones/element_beam.py:113-124` |
| dE | `DifferentialBeamJones` | `core/jones/element_beam.py:164-175` |
| Kd | `DelayJones` | `core/jones/delay.py:64-75` |
| Rc | `CableReflectionJones` | `core/jones/delay.py:126-137` |
| ff | `FringeFitJones` | `core/jones/delay.py:200-211` |
| X | `CrosshandPhaseJones` | `core/jones/crosshand.py:44-55` |
| Kx | `CrosshandDelayJones` | `core/jones/crosshand.py:95-106` |
| DF | `FrequencyDependentLeakageJones` | `core/jones/crosshand.py:150-161` |
| M | `BaselineMultiplicativeJones` | `core/jones/baseline_errors.py:101-113` |
| Q | `SmearingFactorJones` | `core/jones/baseline_errors.py:148-160` |

Every stub file carries the module-level docstring line
`"Stub implementation: returns identity matrix. TODO: implement properly."` and
every stub class docstring begins `"Stub: ... TODO: implement properly."`.
**Correction (7A independent acceptance, 2026-08-01):** the claim that a
repository-wide search finds no `TODO` marker anywhere in `src/radiosim`
outside these twelve stub modules is **not true** at `ac4fe41`: `cli/main.py:6`
(`"TODO: Future enhancements for v0.3.0+"`, present since `be231d2`) and
`core/sky/registry/catalogs.py:595` (`"TODO(scientific-coverage): ..."`,
present since `8372dec`) both predate `ac4fe41` and are neither Jones stubs
nor `SCI-001` material. 7C's I20 residual scan (Section 34) must exclude both
paths explicitly rather than assert an empty `TODO` set and then relax it when
it fails. The load-bearing half of the original claim — that the **beam**
subsystem is TODO-free, which is why Section 19's `SCI-003` disposition rests
on `beam/TODO.md` rather than on in-code markers — does hold and is
unaffected. `tests/characterization/test_tier7_current_behavior.py`'s
`test_todo_markers_outside_the_stub_modules` already recorded this correctly;
this is a plan-text fix, not a code or decision change.

**Nothing outside `src/` references any stub class.** A search for the stub
class names across `docs/`, `tests/`, `examples/`, and `configs/` returns no
match for any of `IonosphereJones`, `FaradayRotationJones`, `GainJones`,
`BandpassJones`, `SmearingFactorJones`, `WPhaseJones`, `DelayJones`,
`CrosshandPhaseJones`, or `ElementBeamJones`. The only cross-directory
references to any Jones class are to `GeometricPhaseJones`
(`tests/unit/test_jones/test_backend_jones.py:14,92`) and to the receptor terms.
Deleting a stub class therefore has an empty external blast radius.

### 5.2 Stub constructor anatomy

The stub constructors are not merely inert; several accept parameters that
they discard entirely:

- `IonosphereJones.__init__` (`ionosphere.py:31-43`) accepts `tec` and never
  stores it — the argument is dropped on the floor. `frequencies`,
  `include_faraday`, and `include_delay` are stored but never read.
- `TroposphereJones.__init__` (`troposphere.py:29-40`) stores `elevations` and
  never reads it.
- `GainJones.__init__` (`gain.py:23-33`) stores `gain_sigma` and `_seed` and
  never reads either.
- `BandpassJones.__init__` (`bandpass.py:27-34`) accepts `bandpass_gains` and
  never stores it.
- `PolarizationLeakageJones.__init__` (`polarization_leakage.py:27-28`) accepts
  `d_terms` and never stores it.
- `ParallacticAngleJones.__init__` (`parallactic.py:31-47`) accepts
  `feed_angle_offset` and never stores it.

This is materially worse than "returns identity": a caller can pass a physically
meaningful `tec` or `d_terms` array, receive no error, and observe no effect.

### 5.3 Chain architecture

`src/radiosim/core/jones/chain.py`:

- `JonesChain.__init__` (`:65-72`) holds `self.backend` and a plain
  `self.terms: list[JonesTerm]`.
- `add_term` (`:74-102`) accepts a `position` of `"append"` or `"prepend"`.
  **It does not reject `JonesBaselineTerm`**, despite the class docstring at
  `:42-46` stating "Only `JonesTerm` subclasses may be added here." A
  `JonesBaselineTerm` is not a `JonesTerm` subclass
  (`baseline_errors.py:22-34`), so an isinstance guard is both correct and
  currently absent.
- `compute_antenna_jones` (`:146-199`) seeds `J_total = xp.eye(2,
  dtype=np.complex128)` (`:172`), iterates `reversed(self.terms)` (`:179`), and
  accumulates `J_total = backend.matmul(J_term, J_total)` (`:197`).
- `compute_antenna_jones_all_sources` (`:201-261`) seeds
  `backend.batch_eye((n_sources,), 2, dtype=np.complex128)` (`:228`), iterates
  `reversed(self.terms)` (`:234`), and batch-multiplies (`:259`).
- Both seeds **hard-code `np.complex128`**, ignoring `PrecisionConfig`
  entirely.
- `compute_baseline_visibility` (`:263-312`), `get_enabled_effects`
  (`:314-329`), `get_config` (`:331-341`), and `clear` (`:343-`) complete the
  surface.

The class docstring records two chain orders. The canonical Tier 5 one at
`:25` —

```
J_total = H @ G @ B @ D @ P @ C @ E @ T @ Z   (K applied separately)
```

— and an "extended" one at `:36-37` —

```
J_total = H @ G @ GAINCURVE @ B @ X @ DF @ D @ P @ C @ E @ Ee @ a @ dE
          @ F @ T @ Z @ W   (K, Kd, Rc applied separately)
```

The extended line is undesigned: it places `W` sky-side of `Z`, and it declares
`Kd` and `Rc` "applied separately" even though both are per-antenna diagonal
direction-independent matrices that compose in the chain like `G` and `B`.
Section 12 replaces it.

`src/radiosim/core/jones/base.py`:

- abstract `name` (`:73-83`) and `is_direction_dependent` (`:85-95`);
- non-abstract defaults `is_baseline_dependent` (`:97-106`, `False`),
  `is_time_dependent` (`:108-117`, `False`), `is_frequency_dependent`
  (`:119-130`, **`True`**);
- abstract `compute_jones(antenna_idx, source_idx, freq_idx, time_idx, backend,
  **kwargs) -> (2, 2)` (`:132-161`) — **one direction at a time, by integer
  index**;
- capability hints `is_diagonal` (`:163-173`), `is_scalar` (`:175-185`),
  `is_unitary` (`:187-197`), each defaulting to `False` and each **entirely
  self-reported and unverified**;
- `compute_jones_all_sources` (`:199-231`), whose default implementation is a
  **Python list comprehension over every source** (`:225-228`) calling
  `compute_jones` once per direction;
- `get_config` (`:233-249`) emitting the seven self-reported flags.

The `is_unitary` vacuity flagged at the Tier 5H review (`Fix.md`, Tier 5H
acceptance, adjudication 2) is live: `FaradayRotationJones.is_unitary`
(`faraday.py:51-52`) and `WPhaseJones.is_unitary` (`wterm.py:59-60`) both
return `True` about a matrix that is the identity. `WPhaseJones.is_scalar`
(`:56-57`) and `ArrayFactorJones.is_scalar` (`element_beam.py:110-111`) are
vacuous in the same way. Tier 5H correctly ruled these `SCI-001` material.

### 5.4 Solver integration

**Point path** — `src/radiosim/core/visibility.py`:

- `simulate_visibilities` accepts `jones_config: dict[str, Any] | None = None`
  (`:298`), normalizes `None` to `{}` (`:360-361`), rejects a non-`dict`
  (`:362-363`), rejects a `"beam"` key (`:364-366`), and calls
  `_reject_parallactic_rotation` (`:382`).
- `_reject_parallactic_rotation` (`:76-96`) raises
  `UnsupportedFeedGeometryError` when `jones_config["P"]["enabled"]` is true and
  any resolved receptor has non-zero `feed_rotation_rad` (`:89-96`).
- `_build_jones_chain` (`:742-888`) adds, in order: `H` (`:806`), optional `G`
  (`:809-815`), optional `B` (`:818-825`), optional `D` (`:828-834`), optional
  `P` (`:837-847`), `C` (`:852`), `E` (`:855-863`), optional `T` (`:866-873`),
  optional `Z` (`:876-886`). The optional terms are gated on
  `jones_config.get("<letter>", {}).get("enabled", False)`.
- The solver's inner structure, per time step (`_time_block`, `:497-724`):
  host preprocessing and horizon mask (`:502-517`), per-time Stokes slicing
  (`:520-556`), direction cosines (`:561-564`), a **Python frequency loop**
  (`:610-721`), per-`(t, f)` chain construction (`:642-655`), a **per-antenna
  Jones cache** built by `compute_antenna_jones_all_sources` (`:658-672`),
  baseline batching to `(B, S, 2, 2)` via `xp.stack` (`:681-688`), the
  **geometric phase computed inline** (`:692-697`), the Gaussian envelope
  `(B, S)` (`:700-709`), and the one compiled kernel call (`:713-720`).
- Time blocks are assembled by `execute_time_blocks` (`:731-736`) and stacked
  into one `(T, B, F, 2, 2)` cube (`:739`).

**HEALPix path** — `src/radiosim/core/visibility_healpix.py`:

- **does not use `JonesChain` at all**;
- `_receptor_transforms` (`:93-...`) builds the constant per-antenna `H_p @ C_p`;
- `_evaluate_beam_batch_by_antenna` (`:230-...`) evaluates the beam per antenna
  over the pixel batch and left-multiplies the receptor transform at `:283`;
- `baseline_contraction_for(backend)` at `:466`, `_time_block` at `:479-`, a
  Python frequency loop at `:511-`, the **geometric phase computed inline
  again** at `:625`, and the kernel call at `:640`;
- `calculate_visibility_healpix` has **no `jones_config` parameter at all**.

**Nothing in production ever supplies a non-empty `jones_config`.** The only
production call site is `src/radiosim/core/hybrid.py:292`, which passes
`jones_config=None` as a literal. `RIMESimulator.simulate`
(`src/radiosim/simulator/rime.py:174,249`) and `VisibilitySimulator.simulate`
(`src/radiosim/simulator/base.py:135`) declare the parameter, but
`api/simulator.py` never passes it. Consequently **G, B, D, P, T, and Z cannot
be enabled through any supported entry point at `ac4fe41`**, and F, W, Txy, Ee,
a, dE, Kd, Rc, ff, X, Kx, DF, M, and Q are not reachable from
`_build_jones_chain` at all.

`GeometricPhaseJones` (`geometric.py:17-198`) is **never constructed by any
solver**. The geometric phase is computed inline twice —
`visibility.py:692-697` and `visibility_healpix.py:619-625` — from the same
formula. The exported class is real physics that the forward model does not
use, and the forward model contains two copies of the physics the class was
written to own.

### 5.5 The compiled-kernel boundary

`src/radiosim/core/contraction.py:42-106` defines the one compiled kernel:

```
baseline_contraction(jones_p, jones_q, coherency, phase, envelope, stokes_i, *, backend)
```

with `jones_p`/`jones_q` of shape `(B, S, 2, 2)`, `coherency` `(S, 2, 2)` or
`None`, `phase` `(B, S)`, `envelope` `(B, S)` or the scalar `1.0`, `stokes_i`
`(S,)` or `None`, returning `(B, 2, 2)`. `baseline_contraction_for`
(`:109-144`) applies `backend.compile` when `supports_compilation` is true and
returns the reference function otherwise; the module docstring (`:1-31`)
records that this is the **only** authorized compilation boundary and that
`vmap` is permitted only inside the kernel.

Two properties of that signature matter enormously to Tier 7 and are recorded
here as design inputs, not as later discoveries:

1. `envelope` is already a per-`(baseline, source)` real attenuation multiplied
   into the per-source weight inside the sum (`:104`). A direction-dependent,
   baseline-dependent decorrelation factor — that is, the smearing term `Q` —
   has **exactly** that shape and that position.
2. The kernel's output is a `(B, 2, 2)` block. A direction-independent,
   baseline-dependent 2×2 multiplicative error — that is, the closure term `M` —
   applies to that output elementwise, outside the kernel.

Both baseline-dependent terms therefore attach **without changing the kernel
signature**.

### 5.6 Configuration, precision, and provenance

- `RadioSimConfig`'s sections are `instrument`, `beams`, `receptors`,
  `baseline_selection`, `sky_model`, `obs_time`, `obs_frequency`, `visibility`,
  `execution`, `workflow`. **There is no `jones:` section, and no section of any
  name that configures a Jones term.** A search for `ionosphere`, `faraday`,
  `leakage`, `bandpass`, `gain`, `parallactic`, `smearing`, and `wterm` across
  `src/radiosim/io/` finds only the `JonesPrecisionInput` precision fields.
- `VisibilityConfig` (`io/config.py:1370-1383`) carries
  `calculation_type: Literal["direct_sum", "spherical_harmonic"] =
  "direct_sum"` (`:1373`), `sky_representation` (`:1374-1376`),
  `allow_lossy_point_materialization` (`:1377`), and
  `allow_lossy_point_rasterization` (`:1383`).
- `collect_unsupported_issues` rejects the spherical-harmonic value at
  `io/config.py:2092-2097` with the message
  `"spherical-harmonic calculation is not implemented until Tier 7"` under code
  `"spherical_harmonic_unsupported"`.
- **`calculation_type` is consumed by nothing else.** A repository-wide search
  finds it only at `io/config.py:1373`, `:2092`, `:2094`, in test fixtures and
  assertions, in `docs/user_guide/configuration.rst:66,183,217`, and in the
  four shipped configs (`configs/config.yaml:65`,
  `configs/receptor_circular_example.yaml:75`,
  `configs/hybrid_sky_example.yaml:93`,
  `configs/realistic_foreground_example.yaml:65`). No solver, simulator,
  resolver, or runtime model reads it. The value `direct_sum` is a **silent
  no-op of exactly the class `CFG-003` was raised about**.
- The **honored** strategy selector is `execution.simulator`. It is read at
  `api/simulator.py:163` (`self._simulator_name = resolved.execution.simulator`)
  and used at `:648` (`get_simulator(self._simulator_name)`); it is carried
  through `io/config_resolution.py:738,768-770,1507` and validated at
  `core/runtime_config.py:328-329` (`if self.simulator != "rime": raise
  ValueError("simulator must be 'rime'")`). `io/config.py:2163` already tells
  users "Use `execution.simulator: rime`".
- `JonesPrecision` (`core/precision.py:273-339`) declares per-term precision for
  exactly eight terms: `geometric_phase`, `beam`, `ionosphere`, `troposphere`,
  `parallactic`, `gain`, `bandpass`, `polarization_leakage` (`:302-322`), with
  `get_dtype`/`get_real_dtype` keyed by attribute name (`:331-339`). The config
  mirror is `JonesPrecisionInput` (`io/config.py:1393-1401`). There is **no**
  precision field for `C`, `H`, `Kd`, `Rc`, `X`, `M`, or `Q`.
- `ReceptorConfigJones.compute_jones` (`core/jones/receptor.py:325-340`) returns
  `backend.xp.array(self._matrices[antenna_idx], dtype=np.complex128)` — a
  hard-coded dtype that ignores `PrecisionConfig`, exactly as `JonesChain`'s two
  identity seeds do.
- `core/receptor.py:411-418` rejects any antenna whose `mount_type` is not
  `fixed`, with the message "time-dependent feed orientation requires the
  parallactic-angle term (Tier 7)." **Correction (7A independent acceptance,
  2026-08-01):** this quotes only the second of the two concatenated string
  literals that make up the actual message. The message the code raises is
  the full `f"mount_type={mount_type!r} is unsupported by Tier 5 receptors; "
  "time-dependent feed orientation requires the parallactic-angle term (Tier
  7)."` — a plan-text quoting fix, not a code or decision change.
  `tests/characterization/test_tier7_current_behavior.py`'s
  `test_mount_types_other_than_fixed_are_rejected` already pins both halves.

### 5.7 Documentation and test surfaces

- `docs/api/jones.rst:1-10` states that many terms "remain identity scaffolds";
  `:38-46` warns "A returned identity matrix is not a modeled physical effect";
  `:49-70` `automodule`s the six stub modules `ionosphere`, `troposphere`,
  `parallactic`, `gain`, `bandpass`, `polarization_leakage`.
- `docs/user_guide/jones_matrices.rst:137-146` records the modelling assumption
  that the basis conversion is exact **only** because `D` and `G` are disabled
  stubs, and states explicitly: "When Tier 7 implements `D`, the conversion
  becomes approximate and this statement must be re-examined."
- `:148-158` records the parallactic-angle boundary and that "When Tier 7
  implements `P`, the composition `P_p(t) C_p` becomes the full time-dependent
  receptor orientation and that rejection is removed."
- `:160-175` records the canonical chain order; `:177-185` the scaffolded-term
  warning.
- `docs/user_guide/configuration.rst:217` documents `calculation_type` as
  accepting both values.
- `src/radiosim/core/jones/beam/TODO.md:1-81` is the `SCI-003` artifact: an
  81-line "Beam System — Future Work (v5.0+)" list with seven numbered items —
  cross-polarization models (quadrupolar / IXR / Ludwig-3, `:3-38`), near/far
  field regime (`:40-45`), aperture blockage (`:47-52`), Ruze random surface
  errors (`:54-59`), systematic aberrations (`:61-67`), beam squint (`:69-74`),
  and pointing errors (`:76-81`). Each carries formulae and, in three cases,
  literature citations.
- Jones tests at `ac4fe41`: `tests/unit/test_jones/` holds
  `test_backend_jones.py`, `test_basis_transform.py`, `test_beam_analysis.py`,
  `test_chain_order.py`, `test_receptor.py`. **No test asserts any physical
  behavior of any stub term**, because there is none to assert.
- Acceptance-test precedent: `tests/unit/test_tier4_result_output_acceptance.py`,
  `tests/unit/test_tier5_receptor_acceptance.py`,
  `tests/unit/test_tier6_runtime_acceptance.py`; characterization precedent:
  `tests/characterization/test_tier4_current_behavior.py`,
  `test_tier5_current_behavior.py`, `test_tier6_current_behavior.py`.

### 5.8 Dependency inventory

`pixi.toml:31-72` locks, among others: `numpy >=1.24,<2.5`, `astropy`,
`pyuvdata ==3.2.1`, `healpy`, `scipy`, `python-casacore >=3.7.1,<4`, and via
PyPI `pygdsm`, `pyradiosky`, `pysm3`. `pixi.toml:86-88` adds a CPU-only
`jax`/`jaxlib` feature carried by both environments (`:90-92`).

**`pyuvsim`, `matvis`, and RASCIL are not present** in any environment, and no
extra in `pyproject.toml` declares them. Any cross-implementation comparison
against those codes requires either a new optional environment or an
out-of-environment, recorded manual run. Section 29 decides this.

Independent references that **are** already locked and usable offline:
`astropy` (frames, `EarthLocation`, sidereal time — an independent parallactic
angle can be derived from it), `pyuvdata 3.2.1` (feed-angle and polarization
conventions; `utils` helpers), `healpy` (spherical harmonics), and
`python-casacore` (CASA table and convention constants).

## 6. Current data-flow trace

The path a configured Jones effect would have to travel today, and where it
stops:

```
config.yaml
  └─ load_config()                      io/config.py
       └─ RadioSimConfig                io/config.py
            ├─ instrument / beams / receptors / baseline_selection   → resolved
            ├─ visibility.calculation_type                           → NOWHERE
            ├─ execution.simulator ("rime")   → api/simulator.py:163,648
            └─ (no jones section exists)
  └─ Simulator.setup()                  api/simulator.py
       └─ resolve instrument, beams, receptors, sky, time grid
  └─ Simulator.run()
       └─ get_simulator("rime")         simulator/__init__.py
            └─ RIMESimulator.simulate(..., jones_config=None default)
                 └─ core/hybrid.py:292  jones_config=None   ← HARD-CODED
                      ├─ point_solver.calculate_visibilities(...)
                      │    └─ visibility.py:360-361  jones_config = {}
                      │         └─ _build_jones_chain(:742-888)
                      │              → H, C, E only; every optional
                      │                `enabled` lookup is False
                      └─ calculate_visibility_healpix(...)
                           → no jones_config parameter exists;
                             H@C applied at :283, beam only
```

The forward model at `ac4fe41` is therefore exactly

```
V_pq = sum_s  (H_p C_p E_p) B_s (H_q C_q E_q)^H  *  exp(-2*pi*i*b.s)
```

with `E` a scalar complex voltage and `H_p C_p` exactly `I2` in the default
homogeneous-linear zero-rotation `linear_xy` case. Everything else in the
46-name (truly 43-name) Jones surface is decoration.

## 7. Confirmed defect matrix

Each row is a defect established by reading source at `ac4fe41`, with the
register row it serves and the Tier 7 slice that closes it.

| # | Defect | Evidence | Register | Slice |
|---|---|---|---|---|
| D0 | `CLAUDE.md` claims 46 exported Jones classes; `__all__` has 43 | `core/jones/__init__.py:76-135` | `SCI-001` | 7J |
| D1 | 37 exported classes return the 2×2 identity from `compute_jones` with no error and no warning | Section 5.1 table | `SCI-001` | 7C |
| D2 | Six stub constructors accept physically meaningful parameters and discard them, so a user can pass a real `tec`, `d_terms`, or `gain_sigma` and observe silence | `ionosphere.py:31-43`, `troposphere.py:29-40`, `gain.py:23-33`, `bandpass.py:27-34`, `polarization_leakage.py:27-28`, `parallactic.py:31-47` | `SCI-001` | 7C |
| D3 | No configuration surface reaches any Jones term; `jones_config` is a raw untyped `dict` and the single production call site passes `None` | `core/hybrid.py:292`, `visibility.py:298,360-366` | `SCI-001`, `Fix.md` §4.1 | 7D |
| D4 | `calculate_visibility_healpix` has no `jones_config` parameter, and the HEALPix path does not use `JonesChain` at all, so any term added to the point path would silently not apply to diffuse sky | `visibility_healpix.py:93-,230-,283` | `SCI-001` | 7B |
| D5 | The `JonesTerm` evaluation contract is scalar-per-direction (`compute_jones(source_idx: int)`) with a Python loop default over sources, which cannot carry HEALPix-scale direction batches | `base.py:132-161,199-231` | `SCI-001` | 7B |
| D6 | The geometric phase is implemented three times: once in an exported class no solver uses, and once inline in each solver | `geometric.py:17-198`, `visibility.py:692-697`, `visibility_healpix.py:619-625` | `SCI-001` | 7B |
| D7 | `JonesChain.add_term` does not reject `JonesBaselineTerm`, contradicting its own docstring | `chain.py:42-46,74-102` | `SCI-001` | 7B |
| D8 | `JonesChain` hard-codes `np.complex128` for both identity seeds, ignoring `PrecisionConfig` | `chain.py:172,228` | `SCI-001` | 7B |
| D9 | `ReceptorConfigJones.compute_jones` hard-codes `np.complex128`, ignoring `PrecisionConfig` | `core/jones/receptor.py:340` | `SCI-001` | 7B |
| D10 | `is_unitary`, `is_scalar`, `is_diagonal` are self-reported and unverified; `F` and `W` claim unitarity about an identity matrix | `base.py:163-197`, `faraday.py:51-52`, `wterm.py:56-60`, `element_beam.py:110-111` | `SCI-001` (Tier 5H adjudication 2) | 7B |
| D11 | `JonesChain`'s "extended" chain-order docstring is undesigned: it places `W` sky-side of `Z` and declares the diagonal terms `Kd`/`Rc` "applied separately" | `chain.py:34-37` | `SCI-001` | 7B |
| D12 | The canonical order places `P` correlator-side of `C`. For a circular receptor that composition applies a real 2×2 rotation to the `(R, L)` pair, which is not the physical effect of a field rotation | `chain.py:25`, `Tier5ReceptorFeedPlan.md` §19.1 | `SCI-001` | 7F |
| D13 | `visibility.calculation_type` is read by no solver, resolver, or runtime model; `direct_sum` is a silent no-op and `spherical_harmonic` is a validated-then-rejected value whose message promises Tier 7 | `io/config.py:1373,2092-2097`; no other consumer | `SCI-002`, `CFG-003` class | 7C |
| D14 | Two config fields name the same choice: `visibility.calculation_type` (unhonored) and `execution.simulator` (honored) | `io/config.py:1373`, `api/simulator.py:163,648` | `Fix.md` §4.1 | 7C |
| D15 | `JonesPrecision` declares per-term precision for eight terms only; `C`, `H`, and every extended term have no precision field | `core/precision.py:302-322`, `io/config.py:1393-1401` | `SCI-001` | 7D |
| D16 | `core/receptor.py:411-418` rejects every non-`fixed` mount type, so no alt-az array can be simulated with a rotated receptor — a scientific capability gap held open only by `P` being a stub | `core/receptor.py:411-418` | `SCI-001` | 7F |
| D17 | `_reject_parallactic_rotation` guards a combination that cannot occur, because `jones_config` is always empty; it is dead defensive code whose removal is gated on `P` becoming real | `visibility.py:76-96,382,800` | `SCI-001` | 7F |
| D18 | The sky model already applies rotation-measure Faraday rotation to point-source Stokes inside the frequency loop, so a separately configured `F` term would double-count line-of-sight rotation with no guard | `visibility.py:618-634` (`source_rm_t` into `evaluate_point_flux_at_freq`), `faraday.py` | `SCI-001` | 7G |
| D19 | The non-coplanar `w` contribution is already exact in the inline geometric phase via `bl_w * (n_dir - 1.0)`, so an enabled `W` term would double-count it | `visibility.py:696`, `visibility_healpix.py:619-625`, `wterm.py` | `SCI-001` | 7C |
| D20 | `src/radiosim/core/jones/beam/TODO.md` is an in-source wish list with no dispositions, no register row, and no verification obligation | `beam/TODO.md:1-81` | `SCI-003` | 7I |
| D21 | `docs/api/jones.rst:49-70` and `docs/user_guide/jones_matrices.rst:177-185` document the stub modules as inspectable surface, which becomes false the moment the stubs are deleted | those lines | `SCI-001` | 7C, 7J |

D18 and D19 are the two most consequential findings of this gate: both are
**double-count hazards** that would have turned a newly implemented term into a
silently wrong forward model. Each is resolved by a scope decision (Sections 10
and 11), not by a runtime guard.

## 8. Design decision 1 — the scope policy for a 40-class surface

### 8.1 Decision

`Fix.md` Section 16's objective sentence is the governing constraint: turn the
framework into implemented effects "without treating 44+ independent models as
one undifferentiated coding task." Tier 7 adopts a single scope rule:

> **One class per physical effect. Every parameterization, model variant, or
> data source becomes a field in that effect's configuration. Every class that
> is not one physical effect is deleted.**

Under `Fix.md` Section 4.1 (pre-v1 policy: "prefer a coherent replacement over
compatibility shims") and the empty external blast radius established in
Section 5.1, deletion is the correct disposition for a speculative stub, not a
regrettable one. Nothing outside `src/` imports any of them; no configuration
names any of them; no test asserts anything about any of them.

### 8.2 The four truthfulness states, and which Tier 7 uses

`Fix.md` Section 4.2 allows four states: implemented and tested; experimental
and explicitly gated; unsupported and rejected with an actionable error; absent
from the public surface.

**Tier 7 uses exactly two of them: *implemented and tested*, and *absent*.**

No term ships as "experimental". The rationale is that an experimental gate is
only honest when there is a concrete reason a user would want the effect before
it is trustworthy, and Tier 7 has no such case: every effect it keeps is one it
can implement completely and verify analytically, and every effect it cannot
verify is one it removes. An "experimental" tier here would be a place to hide
work rather than a service to users. This is a stricter reading than the exit
criterion requires, and the plan states it as a deliberate choice, not as an
obligation.

The third state — *rejected with an actionable error* — is used for
**parameter combinations**, not for classes: for example a `mount_type` that
`P` does not model, or an `F`-style sky rotation measure combined with an
ionospheric one (Section 11). Section 24 lists every rejection verbatim.

### 8.3 What "implemented" must mean, per `Fix.md` Section 16

Every kept term must, at its own slice, satisfy all seven rules:

1. cite the adopted convention and scientific reference (Section 20 does this
   for each term, in advance);
2. define units, axes, and sign conventions (Section 20);
3. add analytic invariants (Section 27);
4. add backend parity where supported (Section 28 and the `Tier6` §13.5
   tolerance rule);
5. add a test proving that a nonzero configured effect changes visibility
   (Section 27's invariant **I7**, mandatory per term);
6. update public status and remove the stub warning **only for that term**
   (Section 25's per-term status registry makes this mechanical and testable);
7. never return identity for an unsupported parameter combination — reject it
   (Section 24).

## 9. Design decision 2 — per-class disposition

### 9.1 The table

Forty concrete exported classes. Eleven survive as implemented physics, two
already are (`C`, `H`), one becomes a function, and twenty-six are deleted.

| Class | Term | Disposition | Rationale |
|---|---|---|---|
| `GeometricPhaseJones` | K | **→ function** | K is per-*baseline*; it cannot be a per-antenna chain term, which is why no solver uses it. Becomes `geometric_phase()` in `core/jones/geometric.py`, called by both solvers, removing the two inline copies (D6). |
| `ReceptorConfigJones` | C | **kept, already implemented** | Tier 5. Gains precision correctness (D9) and the batched contract (D5). |
| `BasisTransformJones` | H | **kept, already implemented** | Tier 5. Same. |
| `GainJones` | G | **implement** | Per-antenna, per-feed complex electronic gain. Absorbs the two variants below. |
| `TimeVariableGainJones` | G | **delete** | A time model is a field (`time_model`) on `G`, not a class. |
| `ElevationGainJones` | GAINCURVE | **delete** | A gain curve is a field (`elevation_curve`) on `G`; the polynomial in elevation multiplies the same diagonal matrix. |
| `BandpassJones` | B | **implement** | Per-antenna, per-feed frequency response. Absorbs the three variants below. |
| `PolynomialBandpassJones` | B | **delete** | `model.kind: polynomial`. |
| `SplineBandpassJones` | B | **delete** | `model.kind: spline`. |
| `RFIFlaggedBandpassJones` | B | **delete** | Flagging is a data-quality product, not a voltage-domain Jones factor; RadioSim has no flag array in its result contract. |
| `PolarizationLeakageJones` | D | **implement** | First-order D-terms, frequency-capable. Absorbs the three variants below. |
| `IXRLeakageJones` | D | **delete** | IXR is a *parameterization* of the same leakage: `d_terms.kind: ixr` with `ixr_db`. |
| `MuellerLeakageJones` | D | **delete** | A Mueller matrix is a derived 4×4 view of the same 2×2 Jones; no distinct physics. |
| `BeamSquintLeakageJones` | D | **delete** | Beam squint is a *beam* property (`beam/TODO.md:69-74`); modelling it as a D-term would create the second beam pathway Section 4 forbids. Routed to the successor beam row (Section 19). |
| `ParallacticAngleJones` | P | **implement** | Per-antenna, per-direction field rotation. Absorbs the two variants below. Unlocks `mount_type` beyond `fixed` (D16). |
| `FieldRotationJones` | P | **delete** | Field rotation *is* the per-direction parallactic angle; making `P` direction-dependent subsumes it exactly. |
| `VLBIFeedRotationJones` | P | **delete** | Heterogeneous mounts are a per-antenna `mount_type`, already resolved by `ResolvedInstrument` (`core/instrument.py:316`). |
| `IonosphereJones` | Z | **implement** | Dispersive TEC phase **and** ionospheric Faraday rotation in one term (absorbs `F`, Section 11). |
| `TurbulentIonosphereJones` | Z | **delete** | Stochastic screens are excluded by Section 4. |
| `GPSIonosphereJones` | Z | **delete** | Requires IONEX/GPS data ingestion, excluded by Section 4. |
| `TroposphereJones` | T | **implement** | Zenith delay × mapping function, and opacity attenuation, in one term. |
| `SaastamoinenTroposphereJones` | T | **delete** | `zenith_delay.kind: saastamoinen` is a field on `T`. |
| `TurbulentTroposphereJones` | T | **delete** | Stochastic; excluded by Section 4. |
| `TroposphericOpacityJones` | T | **delete** | Opacity is the `opacity:` sub-block of `T`; delay and opacity are one antenna-side atmospheric factor. |
| `FaradayRotationJones` | F | **delete** | Folded into `Z`. The sky model already owns *intrinsic source* rotation measure (`visibility.py:618-634`); a second free-standing `F` is the D18 double-count hazard. |
| `DifferentialFaradayJones` | F | **delete** | Per-antenna RM offsets are a field on `Z`. |
| `WPhaseJones` | W | **delete** | The `w` phase is already exact in the geometric phase (`bl_w * (n - 1)`, `visibility.py:696`). Enabling `W` would double-count it (D19). |
| `WProjectionJones` | W | **delete** | W-projection is an *imaging* gridding kernel, not a forward-model factor. Section 4. |
| `WidefieldPolarimetricJones` | Txy | **delete** | The direction-dependent sky→topocentric polarization projection *is* the per-direction parallactic rotation to leading order, which implemented `P` provides. The exact wide-field polarimetric projection additionally needs a non-scalar `E`, which Section 4 excludes; routed to the successor beam row. |
| `ElementBeamJones` | Ee | **delete** | Would be a second beam runtime (Section 4, Tier 3). Routed to the successor station-beam row. |
| `ArrayFactorJones` | a | **delete** | Same; a phased-array factor belongs inside a station beam model, not beside it. |
| `DifferentialBeamJones` | dE | **delete** | The leading per-antenna beam difference is a per-antenna **pointing offset**, which Section 19 implements inside `BeamSystem`, and per-antenna diameters/FITS beams already exist (Tier 3). |
| `DelayJones` | Kd | **implement** | Per-antenna, per-feed instrumental delay. Exact, trivial, and testable. |
| `CableReflectionJones` | Rc | **implement** | Cable-reflection ripple; real, well-cited, and a distinct (non-pure-phase) frequency structure from `B`. |
| `FringeFitJones` | ff | **delete** | Fringe fitting is a *calibration solution*, and its forward-model content is exactly `G` × `Kd` × a phase rate. Section 4. |
| `CrosshandPhaseJones` | X | **implement, renamed `CrosshandJones`** | Cross-hand phase and cross-hand delay are the same diagonal matrix, one constant in frequency and one linear; merged into one term with two parameters. |
| `CrosshandDelayJones` | Kx | **delete** | Merged into `CrosshandJones`. |
| `FrequencyDependentLeakageJones` | DF | **delete** | `D` is frequency-capable by construction; a second class for the same matrix with a frequency axis is duplication. |
| `BaselineMultiplicativeJones` | M | **implement** | The canonical baseline-Hadamard closure error; applies to the kernel's `(B, 2, 2)` output. |
| `SmearingFactorJones` | Q | **implement** | Time and bandwidth smearing decorrelation; folds into the kernel's existing `envelope` argument. |

Resulting public Jones surface after Tier 7: **16 names** —
`JonesTerm`, `JonesChain`, `JonesBaselineTerm`, `ReceptorConfigJones` (C),
`BasisTransformJones` (H), `GainJones` (G), `BandpassJones` (B),
`PolarizationLeakageJones` (D), `ParallacticAngleJones` (P),
`IonosphereJones` (Z), `TroposphereJones` (T), `DelayJones` (Kd),
`CableReflectionJones` (Rc), `CrosshandJones` (X),
`BaselineMultiplicativeJones` (M), `SmearingFactorJones` (Q) — plus the
module-level function `geometric_phase` and the private solver-owned
`_ResolvedBeamJones` adapter for E.

**Every one of those 16 names is either a base class, or a term with
implemented physics, a configuration surface, analytic invariants, backend
parity, and an effect-changes-visibility test.** That is the whole of
`SCI-001`'s exit criterion, satisfied by construction rather than by audit.

### 9.2 Why deletion rather than `NotImplementedError`

`Fix.md` Section 7.5 offers three truthful policies for an unimplemented term:
remove it, raise, or gate it experimentally. Raising is the weakest of the
three here, because a `NotImplementedError`-raising `GPSIonosphereJones` still
advertises in the API reference, in `dir()`, and in Sphinx that RadioSim has a
GPS ionosphere model that is nearly ready. It is not nearly ready; RadioSim has
no IONEX reader and no plan to acquire one. Removing the name is the only
statement that is actually true.

The deleted names are recorded in the breaking-change ledger (Section 36) and
in `docs/migration_guide.md`, with the surviving replacement for each, so that
a user who had imported one gets a documented answer rather than a bare
`AttributeError` — the exact gap the Tier 5H review identified for the three
removed receptor keywords.

## 10. Design decision 3 — workstream C is answered by decisions, not by code

`Fix.md` Section 16's Workstream C names five items. Their dispositions:

| Workstream C item | Disposition |
|---|---|
| "W/non-coplanar effects where appropriate to the forward model" | **Already exact and already in the forward model.** The direct-sum RIME evaluates `exp(-2πi (ul + vm + w(n-1)))` with no small-angle or coplanar approximation (`visibility.py:692-697`). There is no residual non-coplanar effect for a `W` term to add; adding one would double-count (D19). Tier 7's deliverable here is a **test that pins the `w(n-1)` term's presence and sign**, plus documentation stating that non-coplanarity is exact in direct summation and that W-projection is an imaging concern. |
| "element beams" | Descoped; routed to a successor row (Section 19.3). A station element beam belongs inside `BeamSystem`. |
| "array factors and mutual coupling" | Descoped; same row. Mutual coupling additionally requires an embedded-element pattern basis RadioSim does not have. |
| "differential beam residuals" | **Delivered differently**: per-antenna deterministic **pointing offsets** in `BeamSystem` (Section 19.2), which is the dominant real differential-beam effect and which composes with the existing per-antenna diameter and per-antenna FITS beam support from Tiers 2 and 3. |
| "cable reflections and electronic delays" | **Implemented** as `Rc` and `Kd` (Sections 20.5, 20.6). |

Stating this explicitly matters: a later reviewer must be able to see that
Workstream C was *decided*, not skipped.

## 11. Design decision 4 — Faraday rotation belongs to Z, and the sky owns intrinsic RM

### 11.1 The hazard

`visibility.py:618-634` passes `source_rm_t` into `evaluate_point_flux_at_freq`
inside the frequency loop, so **intrinsic source rotation measure is already
applied to the point-source Stokes vector** before the coherency matrix is
built. A separately configured `F` Jones term carrying a rotation measure would
rotate `(Q, U)` a second time. Nothing in the current code would detect this;
the result would be a plausible-looking, wrong polarization angle.

### 11.2 The decision

- **Intrinsic (source-frame) Faraday rotation stays owned by the sky model.**
  It is a property of the source, it varies per source, and it is already
  correct. Tier 7 does not touch it.
- **Line-of-sight rotation imposed by the terrestrial ionosphere becomes part
  of `Z`**, whose constructor already carries an `include_faraday` flag
  (`ionosphere.py:35`) and whose docstring in `base.py:26` already describes Z
  as "Ionospheric Faraday rotation + TEC phase". `Z` therefore produces
  `Z = R_F(psi) * exp(i*phi_TEC)` — a rotation composed with a scalar
  dispersive phase.
- **`FaradayRotationJones` and `DifferentialFaradayJones` are deleted**, so
  there is no second place from which a rotation measure can enter the chain.
- The two are distinguishable by frame: source RM rotates the sky-frame `(Q,U)`
  before the coherency matrix; ionospheric Faraday rotates the field in the
  topocentric frame after propagation. They compose, they do not duplicate, and
  because they live in different objects (`SkyModel` vs `jones.Z`) they cannot
  be configured twice by accident.
- Section 27's **I8** invariant pins this: a run with source RM only, a run
  with ionospheric RM only, and a run with both must satisfy the composition
  relation exactly, with the polarization angle rotating by
  `(RM_src + RM_ion) * lambda^2`.

## 12. Design decision 5 — the corrected canonical chain order

### 12.1 The defect being corrected

Tier 5 Section 19.1 fixed

```
J_p = H_p G_p B_p D_p P_p C_p E_p T_p Z_p        (K separate)
```

and said, correctly for its own scope, that the order "is currently
unobservable — all optional terms are disabled by default and identity when
enabled". Implementing `P` makes it observable, and the placement of `P`
correlator-side of `C` is **wrong for a circular receptor** (D12).

A field rotation by angle `psi` acts on the incoming field in the *linear*
topocentric frame. The receptor term is `C = M(basis) R(chi)` with
`M(circular) = S` (`docs/user_guide/jones_matrices.rst:84-101`). The physical
composite for a circular receptor with static rotation `chi` and time-varying
field rotation `psi` is

```
M(circular) R(chi + psi) = S R(chi) R(psi) = C R(psi)
```

so `R(psi)` must sit **sky-side (right) of `C`**. Under Tier 5's order the
composite would be `R(psi) S R(chi)`, which applies a real 2×2 rotation to the
`(R, L)` pair. Since `S R(psi) = diag(e^{-i psi}, e^{+i psi}) S`, the correct
effect on `(R, L)` is a pair of opposite phases, not a mixing rotation. The two
agree only for a linear receptor, where `M = I2` and rotations commute — which
is exactly the case Tier 5 tested.

### 12.2 The corrected order

```
J_p = H_p . G_p . B_p . Rc_p . Kd_p . X_p . D_p . C_p . E_p . P_p . T_p . Z_p
```

leftmost nearest the correlator, applied right-to-left. Read sky-to-correlator,
that is: ionosphere `Z`, troposphere `T`, field rotation `P`, primary beam `E`,
receptor `C`, leakage `D`, cross-hand `X`, electronic delay `Kd`, cable
reflection `Rc`, bandpass `B`, gain `G`, reporting-basis transform `H`.

`K` remains applied separately by the solver (it is per-baseline). `M` and `Q`
are not chain terms at all (Section 14).

Because `JonesChain` composes `terms[0] @ ... @ terms[-1]`
(`chain.py:17-20,179-197`), the add order is literally that left-to-right list.

### 12.3 What is physical in this order, and what is convention

This distinction is load-bearing and must be documented rather than implied:

- **Physical, and testable:** the placement of `P` sky-side of `C` and `E`
  (Section 12.1); the placement of `D` correlator-side of `C` (leakage is
  defined in the receptor's own basis — Tier 5 Section 19.1, unchanged); the
  placement of `Z` and `T` sky-side of `P` (atmospheric propagation happens to
  the sky-frame field before any frame rotation); the placement of `E` between
  `C` and `P`.
- **Convention, because the factors commute:** the relative order of
  `G`, `B`, `Rc`, `Kd`, `X`. All five are diagonal 2×2 matrices in the same
  basis, and diagonal matrices commute. Their mutual order is fixed by this
  document so that the chain has one shape, one provenance string, and one
  test; it is **not** a physical claim, and the plan says so in the
  documentation.
- **Currently unobservable, and flagged:** the relative order of `E` and `P`.
  While `E` is a scalar complex voltage on the diagonal (Section 4), it
  commutes with everything, so `C E P` and `C P E` are numerically identical.
  The order is fixed at `C E P` because that is the physically correct one for
  a future non-scalar `E`, and Section 19.3's successor row records that any
  non-scalar `E` work must re-verify it.

### 12.4 Consequence for Tier 5's accepted record

Tier 5's Section 19.1 order remains correct **for the terms Tier 5
implemented** — `H`, `C`, `E`, and the identity-stub placeholders. Tier 7
supersedes only the placement of `P`, and only because implementing `P` is what
makes the placement physical. The Tier 5 acceptance record is not rewritten;
Section 36's breaking-change ledger records the supersession, and the 7F slice
carries a test that fails under the old order for a circular receptor.

## 13. Design decision 6 — the direction-batched evaluation contract

### 13.1 The problem

`JonesTerm.compute_jones` takes `source_idx: int` and returns one `(2, 2)`
matrix (`base.py:132-161`); the default `compute_jones_all_sources`
(`:199-231`) is a Python list comprehension over sources. For a point sky of a
few thousand sources that is merely slow; for the HEALPix path, where the
direction count is the pixel count, it is unusable — which is precisely why
`visibility_healpix.py` bypasses `JonesChain` entirely (D4) and why the E
adapter overrides the batched method.

Any direction-dependent Tier 7 term — `P`, `Z`, `T`, and `Q` — inherits that
problem. Implementing them against the scalar contract would either be
unusably slow or would force a second per-solver implementation of each.

### 13.2 The decision

Replace the scalar contract with a direction-batched one. This is a breaking
change to a public ABC, which `Fix.md` Section 4.1 explicitly prefers over a
shim, and which affects only four real implementations (`C`, `H`, the private
E adapter, and the K class that is being turned into a function anyway).

```python
# src/radiosim/core/jones/directions.py  (new)

@dataclass(frozen=True)
class DirectionBatch:
    """One immutable batch of sky directions for one (time, frequency) step."""
    alt_rad:   NDArray      # (n_dir,) topocentric altitude
    az_rad:    NDArray      # (n_dir,) topocentric azimuth, N through E
    l:         NDArray      # (n_dir,) direction cosines, phase-centre frame
    m:         NDArray
    n:         NDArray
    ra_rad:    NDArray      # (n_dir,) ICRS right ascension
    dec_rad:   NDArray      # (n_dir,) ICRS declination
    hour_angle_rad: NDArray # (n_dir,) local apparent hour angle
    n_dir:     int
```

`ra_rad`, `dec_rad`, and `hour_angle_rad` are required because `P` is defined in
the equatorial frame and computing it from `(alt, az)` alone loses the
quadrant. They are produced once per time step, host-side, alongside the
existing `_host_preprocess_time_step` work (`visibility.py:502-506`), and are
therefore free.

**Correction (7B implementation, 2026-08-01)** — two field-level changes:

- The direction-cosine fields are named `dir_l`, `dir_m`, `dir_n`, not `l`, `m`,
  `n`.  `l` is an ambiguous identifier that the repository's own lint
  configuration rejects (ruff `E741`, part of the selected `E` rule set), and
  `n` would collide with `n_dir`.  The chosen names are the ones
  `visibility_healpix` already used.
- The equatorial half is the **apparent** description of the same directions,
  derived from `(alt, az)` with the site latitude and the local apparent
  sidereal time, rather than the catalogue ICRS position read off the sky model.
  Three reasons, all discovered by executing the design: pairing an ICRS right
  ascension with an apparent sidereal time is internally inconsistent at the
  equinox-of-date level (of order `1e-2` rad in 2025, measured), so a field
  rotation computed from the mismatched pair would inherit that error; a HEALPix
  map may be stored in galactic coordinates and has no right ascension to read
  at all; and the horizontal-to-equatorial inverse is exact and keeps the
  quadrant, because the hour angle comes from a two-argument arctangent of the
  same two components the forward transform used.  `ra_rad` is therefore an
  apparent right ascension of date; a term needing a catalogue position needs
  the sky model, not the batch.

```python
# src/radiosim/core/jones/base.py  (replacing compute_jones / compute_jones_all_sources)

@abstractmethod
def compute_jones_batch(
    self,
    *,
    antenna_idx: int,
    directions: DirectionBatch,
    frequency_hz: float,
    freq_idx: int,
    time_mjd: float,
    time_idx: int,
    backend: ArrayBackend,
    dtype: DTypeLike,
) -> Any:
    """Return this term's Jones matrices for one antenna over one direction batch.

    Returns
    -------
    array
        Complex, shape ``(n_dir, 2, 2)`` for a direction-dependent term, or
        ``(1, 2, 2)`` for a direction-independent term, in the backend's own
        array domain and in ``dtype``.  A ``(1, 2, 2)`` return broadcasts
        against ``(n_dir, 2, 2)`` and is the required form for a DIE term:
        materialising ``n_dir`` identical copies is forbidden.
    """
```

Notes on the signature, each deliberate:

- **Keyword-only.** Matches the Tier 5 receptor-term constructor discipline
  (`core/jones/receptor.py:210-215`) and makes a mis-ordered call impossible.
- **`dtype` is passed in, not chosen by the term.** This is what closes D8 and
  D9: the solver resolves the per-term dtype from `PrecisionConfig` once and
  hands it down, so no term can hard-code `np.complex128`.
- **`frequency_hz` and `time_mjd` are passed as physical values**, not only as
  indices. Every stub today receives only indices and must have been
  pre-loaded with a `frequencies` array at construction, which is why so many
  of them carry a redundant `frequencies` constructor argument. Passing the
  value removes that whole class of construction coupling.
- **The `(1, 2, 2)` DIE convention is mandatory, not optional.** A DIE term
  that returned `(n_dir, 2, 2)` would multiply the chain's memory by the
  direction count for no reason. Section 27's **I2** invariant tests it.

`JonesChain.compute_antenna_jones_batch` composes the terms in order, seeding
with a `(1, 2, 2)` identity in the resolved dtype, so a chain of purely DIE
terms stays `(1, 2, 2)` all the way through and broadcasts once, at the end,
against the DDE factors.

### 13.3 What is removed

`JonesTerm.compute_jones`, `JonesTerm.compute_jones_all_sources`,
`JonesChain.compute_antenna_jones`,
`JonesChain.compute_antenna_jones_all_sources`, and
`JonesChain.compute_baseline_visibility` (`chain.py:263-312`, which has no
production caller and duplicates the kernel's contraction) are removed. No
deprecation shim, per `Fix.md` Section 4.1.

## 14. Design decision 7 — one shared chain evaluator for both solvers

### 14.1 The decision

Both solvers call one function, defined once:

```python
# src/radiosim/core/jones/evaluate.py  (new)

def evaluate_antenna_jones(
    *,
    chain: JonesChain,
    antenna_rows: Sequence[int],
    directions: DirectionBatch,
    frequency_hz: float,
    freq_idx: int,
    time_mjd: float,
    time_idx: int,
    backend: ArrayBackend,
    dtypes: ResolvedJonesDtypes,
) -> dict[int, Any]:
    """Return {antenna_number: (n_dir, 2, 2)} for one (time, frequency) step."""
```

**Correction (7B implementation, 2026-08-01)** — two signature details:

- The returned mapping is keyed by **antenna row**, not antenna number.  The
  sketch above passes `antenna_rows` and returns `{antenna_number: ...}`, which
  is not derivable from the arguments; and rows are the better key, because
  every chain term indexes the instrument by row, so keying the result the same
  way makes a row/number mix-up structurally impossible instead of something a
  runtime cross-check has to catch.  Each solver maps its selected pairs through
  `row_for_number` once, above the time loop.
- `dtypes: ResolvedJonesDtypes` becomes a single `dtype` until Tier 7D
  introduces `ResolvedJonesDtypes` (`core/jones_terms.py` is in 7D's writable
  list, not 7B's).  7B resolves one chain dtype, from the **accumulation**
  precision as Section 17.1 requires for the seed, and passes it to every term;
  7D replaces it with the per-term resolution without changing any call site's
  shape.

- The **point path** replaces its per-antenna cache loop
  (`visibility.py:658-672`) with one call.
- The **HEALPix path** replaces `_receptor_transforms` (`:93-`) and the
  receptor left-multiplication inside `_evaluate_beam_batch_by_antenna`
  (`:283`) with the same call, where `directions` is the pixel batch. The beam
  evaluation itself stays inside the `E` adapter, so `BeamSystem` remains the
  single beam runtime and the per-handler caching that path already performs is
  preserved inside the adapter rather than in the solver.

This closes D4 permanently: there is exactly one place where a Jones term is
composed, so a term cannot apply to point sources and silently not apply to
diffuse sky.

### 14.2 Why not push the chain into the compiled kernel

Because `Tier6HybridRuntimePlan.md` Section 13.6 authorizes exactly one
compiled kernel and `core/contraction.py` is it. The chain evaluation is
host-orchestrated (it calls astropy-derived quantities and, for `E`, pyuvdata
interpolation) and is explicitly out of scope for compilation. Tier 7 does not
widen the compilation boundary, does not add a second `backend.compile` call
site, and does not change the kernel's signature (Section 5.5). A test asserts
that `src/` contains exactly one `backend.compile` call site, extending the
Tier 6H invariant rather than replacing it.

## 15. Design decision 8 — the baseline-dependent Hadamard path

### 15.1 The decision

`JonesBaselineTerm` (`baseline_errors.py:22-74`) stays a separate ABC and stays
un-addable to `JonesChain` — now **enforced** by an isinstance rejection in
`add_term` (D7). Its evaluation contract is batched in the same way as
`JonesTerm`'s:

```python
@abstractmethod
def compute_baseline_factor(
    self,
    *,
    baseline_pairs: Sequence[tuple[int, int]],
    baseline_uvw_wavelengths: Any,   # (B, 3)
    directions: DirectionBatch | None,
    frequency_hz: float,
    channel_width_hz: float,
    integration_time_s: float,
    backend: ArrayBackend,
    dtype: DTypeLike,
) -> Any:
    """(B, n_dir) real for a DDE factor, or (B, 2, 2) complex for a DIE factor."""
```

The two implemented terms attach at two different, already-existing points:

| Term | Kind | Shape | Where it attaches |
|---|---|---|---|
| `Q` (smearing) | DDE, real, scalar | `(B, n_dir)` | multiplied into the kernel's existing `envelope` argument (`visibility.py:700-709`, `contraction.py:104`) |
| `M` (closure) | DIE, complex, 2×2 | `(B, 2, 2)` | elementwise-multiplied into the kernel's `(B, 2, 2)` output, after the call |

**The compiled kernel's signature does not change.** Its `envelope` argument
already has exactly `Q`'s shape and exactly `Q`'s position inside the sum, and
its return already has exactly `M`'s shape. This is the single most important
structural property of the Tier 7 design and it is why Workstream D is a small
slice rather than a kernel redesign.

**Correction (7H implementation, 2026-08-01) — the final
`compute_baseline_factor` signature, and how a solver knows where a factor
attaches.** Two changes, both forced, neither widening what the terms do.

1. **The signature is the batched one above, reconciled with 7B's accepted
   keyword set.** 7B introduced the method as concrete-and-raising with a
   *per-baseline* keyword set — `baseline_idx`, `antenna_p`, `antenna_q`,
   `freq_idx`, `time_idx`, and no baseline geometry at all — which neither term
   can be written against: `Q`'s two envelopes are functions of the baseline
   vector, and no parameter carried one. The keywords therefore become

   ```python
   compute_baseline_factor(
       *,
       baseline_pairs: tuple[tuple[int, int], ...],   # ordered antenna numbers, selection order
       baseline_uvw_wavelengths: Any,                 # (B, 3), backend array
       directions: DirectionBatch,
       frequency_hz: float,
       freq_idx: int,
       time_mjd: float,
       time_idx: int,
       backend: ArrayBackend,
       dtype: DTypeLike,
   ) -> Any                                           # (B, n_dir) real | (B, 2, 2) complex
   ```

   which is this section's original batched contract plus the two grid indices
   7B added for the same reason it added them to `compute_jones_batch`.
   `channel_width_hz` and `integration_time_s` are **not** parameters; see the
   Section 20.11 correction for where they come from instead.
2. **`hadamard_target` is a new declared property on the ABC**, with exactly two
   values — `"envelope"` and `"correlation"` — naming which of the two
   attachment points in the table above a term's factor belongs to. A solver
   must dispatch on something, and the two candidates already present are both
   wrong: `is_direction_dependent` is a statement about the *physics* that would
   silently become the wiring (a future direction-dependent correlation-side
   factor would attach in the wrong place), and an `isinstance` check against
   the two concrete classes would put the term inventory inside the solver. The
   property is abstract, so a new baseline term cannot omit it.
3. **Both solvers call one shared evaluator**,
   `evaluate_baseline_factors()` in `core/jones/baseline_errors.py`, for the
   same reason Section 14 gives for `evaluate_antenna_jones`: a baseline term
   that reached the point path and silently not the diffuse one would be defect
   D4 again, one axis over. It returns the two products — the envelope factor
   and the correlation factor, each `None` when no term declares that target —
   and shape- and finiteness-checks each one (Section 26).

### 15.2 Ordering rule

`Q` multiplies the Gaussian morphology envelope; when both are present the
product is `envelope_gauss * Q`, and multiplication is commutative, but the
plan fixes the evaluation order so the floating-point result is reproducible.
`M` is applied after the contraction and before the block is cast to
`output_complex_dtype` (`visibility.py:721`), so that closure errors participate
in the accumulation at accumulation precision rather than at output precision.

## 16. Design decision 9 — the `jones:` configuration schema surface

### 16.1 The decision

A new top-level `jones:` section on `RadioSimConfig`, strict and frozen like
every other Tier 1/5/6 section, with one sub-block per implemented term, each
disabled by default. The raw `jones_config: dict | None` parameter is
**removed** from `simulate_visibilities`, `calculate_visibilities`,
`RIMESimulator.simulate`, and `VisibilitySimulator.simulate` (D3, and
`Fix.md` Section 4.1's "do not preserve raw-dictionary behavior"). It is
replaced by a resolved frozen `ResolvedJonesTerms` produced once, before any
solver work, by `resolve_jones_terms(config.jones, instrument, receptors,
obs_frequency, obs_time, precision)`.

Full schema in Section 21; resolved model in Section 22; rejections in Section
24; provenance in Section 25.

### 16.2 The three properties the schema must have

1. **Absence is exactly the current behavior.** An omitted `jones:` section, or
   a `jones:` section with every term omitted, must produce a cube
   bit-identical to `ac4fe41`'s for the same configuration. Section 27's **I1**
   invariant and the 7A characterization pins enforce it.
2. **Presence is never silent.** Every accepted term configuration must change
   the visibilities, or be rejected. There is no accepted configuration whose
   effect is the identity — including, notably, the zero-valued one: a
   `jones.G` block with `amplitude_error: 0.0` and `phase_error_rad: 0.0` is
   **rejected**, not accepted-as-identity, because a user who writes it wants an
   effect (Section 24, rejection R7).
3. **Every accepted term enters the scientific fingerprint.** Two runs that
   differ only in a Jones parameter must produce different `scientific_sha256`
   (Section 25, invariant **I13**).

## 17. Design decision 10 — precision and backend rules

### 17.1 Precision

- `JonesPrecision` (`core/precision.py:273-339`) gains fields for every
  implemented term that lacks one: `receptor` (C/H), `delay` (Kd),
  `cable_reflection` (Rc), `crosshand` (X), `baseline_error` (M), `smearing`
  (Q). `JonesPrecisionInput` (`io/config.py:1393-1401`) mirrors them. The four
  presets in `core/precision.py:522,564,607` gain matching entries. `geometric_phase`
  keeps its `CRITICAL` status and is the dtype used for the shared
  `geometric_phase()` function.
- The solver resolves a frozen `ResolvedJonesDtypes` once per run — one complex
  and one real dtype per term letter — and passes the per-term dtype into
  `compute_jones_batch`. No term chooses its own dtype.
- The chain's identity seed uses the **accumulation** precision
  (`PrecisionInput.accumulation`, `io/config.py:1421`), not a literal, closing
  D8.
- `ReceptorConfigJones` and `BasisTransformJones` stop hard-coding
  `np.complex128` (D9). Because the default precision preset is `float64`
  everywhere, this is bit-identical for every shipped configuration; a
  non-default preset is where it becomes observable, and 7B carries a test for
  exactly that.

### 17.2 Backends

- Every implemented term computes through `backend.xp` and the `ArrayBackend`
  primitives, exactly as `C`/`H` do today, and must therefore run unchanged on
  NumPy, JAX-CPU, and Dask.
- The parity requirement is `Tier6HybridRuntimePlan.md` Section 13.5's tolerance
  rule, unchanged: **Dask bit-identical to NumPy; JAX-CPU within
  `rtol=1e-12`**, on the full `(T, B, F, 2, 2)` cube. Section 28's parity
  matrix requires one parity case **per implemented term**, with the term enabled at a
  physically large value so that parity is testing the term, not the noise
  floor.
- Terms must not use host-only constructs (`float()` on a traced array, Python
  `if` on array values, `.item()`) inside `compute_jones_batch`. Where a term
  needs a host-side quantity — for example the elevation used by `T`'s mapping
  function — it comes in through `DirectionBatch`, which is built host-side by
  design.
- Tier 7 makes **no performance claim**. If a term is slow, that is recorded,
  not optimized, and `pixi run bench`'s reference records are extended only if
  a slice's evidence needs them.

## 18. Design decision 11 — Workstream E (m-mode) is descoped, and `SCI-002` closes by absence

### 18.1 The decision

**RadioSim will not gain a spherical-harmonic / m-mode simulator in Tier 7.**
`visibility.calculation_type` is **removed from the configuration schema
entirely** — both values, not just the unimplemented one — and the m-mode
solver is filed as a new roadmap register row for a successor tier.

### 18.2 Why removing the whole field, and not just the value

`Fix.md` Section 7.6 offers two truthful options: "schema validation should
reject the option **or the option should be removed**". `Fix.md` Section 16's exit
criterion is "m-mode is either implemented and tested **or absent from accepted
config**".

The current state is neither. `calculation_type: spherical_harmonic` is
*accepted by the Pydantic `Literal`* and then rejected by a later
unsupported-issue pass (`io/config.py:2092-2097`) with a message that promises
"until Tier 7" — a promise this tier is deliberately not keeping. And
`calculation_type: direct_sum`, the value four shipped configs actually set, is
read by nothing (D13): it is a silent no-op of exactly the class `CFG-003` was
raised about, sitting in the config file of every user who copies an example.

Keeping the field with one legal value would preserve a second, unhonored name
for a choice that `execution.simulator` already owns and already honors
(`api/simulator.py:163,648`; `core/runtime_config.py:328-329`) — the redundancy
`Fix.md` Section 4.1 tells us to remove (D14).

So: `execution.simulator` becomes the **single** solver-strategy selector. When
an m-mode solver exists, it is registered in `simulator/__init__.py`'s registry
and becomes `execution.simulator: mmode`, with no schema surgery required. A
test asserts that the accepted values of `execution.simulator` equal the keys of
the simulator registry — a standing invariant that makes the `SCI-002` class of
defect structurally impossible to reintroduce (Section 27, **I15**).

### 18.3 Why m-mode is not attempted in Tier 7 — the honest argument

This is a descope, and the plan states it as one rather than dressing it up.

An m-mode solver is not a term; it is a **second complete forward model**. It
requires, at minimum: a defined and enforced observing regime (drift scan, a
fixed pointing, a regular sidereal time grid); a spherical-harmonic
representation of the sky including the polarized components; a
spherical-harmonic representation of every antenna's beam, which for the FITS
beam path means transforming interpolated `az_za` grids to `a_lm` with a
controlled and reported truncation error; the beam-transfer-matrix
construction `B_{lm}` per baseline per frequency; the m-mode transform of the
time axis; per-`m` linear algebra; and a validation program establishing where
it agrees with direct summation and where truncation dominates. It also
interacts with essentially every Tier 4/5/6 contract — the time grid, the
correlation coordinate axis, the hybrid component model, the worker policy, and
the fingerprint.

That is a tier. Attempting it alongside eleven Jones terms would produce
exactly the "one undifferentiated coding task" `Fix.md` Section 16 opens by
warning against, and the predictable outcome is that both halves land
half-verified. The disjunctive exit criterion exists precisely so that this
call can be made honestly.

### 18.4 The register consequence, to be executed at whole-tier acceptance

- `SCI-002` flips to **DONE**, with the closure text stating that it closed by
  **removal from the public configuration surface**, not by implementation, and
  naming the successor row.
- A new row is filed: **`SCI-004` | ROADMAP | Spherical-harmonic / m-mode
  simulator strategy is not implemented; `execution.simulator` accepts only
  `rime`. Requires its own design gate covering observing regime, sky and beam
  harmonic representations, truncation and accuracy boundaries, and direct-sum
  agreement tests | successor tier**.

This gate does not edit the register (Section 2); slice 7K does, and Section 38
specifies the exact evidence required first.

## 19. Design decision 12 — the `SCI-003` beam-physics disposition

### 19.1 What `SCI-003` actually is

The artifact is `src/radiosim/core/jones/beam/TODO.md:1-81` (D20): seven
numbered future-work items with formulae and, in three cases, citations. It is
the only remaining TODO surface in the beam subsystem — the beam runtime itself
is TODO-free at `ac4fe41`.

`Fix.md` Section 16's exit criterion for it is: "advanced beam TODOs have
explicit scientific scope and verification." That is a two-part obligation:
*scope* for all of them, and *verification* for something. It does not require
implementing all seven.

### 19.2 What Tier 7 implements

Two items, chosen because both are real, both are deterministic and analytically
verifiable, and neither breaks the scalar-`E` constraint of Section 4:

- **Per-antenna deterministic pointing offsets** (`TODO.md:76-81`). A
  configured `(delta_az, delta_el)` per antenna shifts the direction at which
  that antenna's beam is evaluated. This is the dominant real
  differential-beam effect, it is what makes the deletion of
  `DifferentialBeamJones` legitimate (Section 9.1), and it is verified by two
  analytic invariants: a pointing offset equal to zero is bit-identical to no
  offset, and an offset of `delta` moves the beam's evaluated peak by exactly
  `delta` for an analytic beam whose peak position is known in closed form.
- **Ruze random-surface efficiency** (`TODO.md:54-59`). The Ruze equation
  `eta_s = exp(-(4 pi sigma / lambda)^2)` is a single real scalar multiplying
  the beam voltage amplitude, deterministic in `sigma` and `lambda`. Verified
  against the closed form at three wavelengths and against the
  `lambda_min ~= 10 sigma` rule of thumb quoted in `TODO.md:59`. The *error
  beam* half of the Ruze decomposition (`TODO.md:57`) is **not** implemented and
  is explicitly scoped out, because it is a stochastic scattered-power model.

Both live inside `core/beam/` and `BeamSystem`, not in a Jones module, so the
Tier 3 single-beam-runtime property is preserved.

**Correction (7I implementation, 2026-08-02) — the pointing geometry, stated
exactly.** The bullet above says only that `(delta_az, delta_el)` "shifts the
direction at which that antenna's beam is evaluated", which is not a convention:
the obvious reading — evaluate at `(alt - delta_el, az - delta_az)` — is
*unimplementable* for RadioSim, whose beams are zenith-pointed. The nominal
boresight is `alt = 90 deg`, so an additive elevation shift sends the peak to
`alt = 90 deg + delta_el`, outside the closed `[-pi/2, pi/2]` domain every beam
evaluator enforces. The slice therefore adopts a genuine rotation, which is what
makes the bullet's own second invariant ("an offset of `delta` moves the beam's
evaluated peak by exactly `delta`") true as a great-circle statement rather than
a small-angle one:

- The offset is a fixed rotation `R(delta_az, delta_el)` of the antenna's beam
  frame relative to the topocentric horizontal frame, composed as the two
  encoder errors of an alt-az mount: first a rotation about the local vertical
  that increases azimuth by `delta_az` (North through East, Section 20.0), then
  a tilt of `delta_el` about the beam frame's horizontal axis, carrying the
  boresight from the zenith toward azimuth zero.
- Composed, the boresight lands at topocentric azimuth `delta_az` and zenith
  angle `delta_el`. The beam is evaluated at the direction expressed in that
  rotated frame, `n_beam = R^T n`.
- Two consequences are exact, and both are asserted: the beam's peak moves by a
  great-circle angle of exactly `|delta_el|`, in the direction of azimuth
  `delta_az`; and `delta_az` alone, with `delta_el = 0`, rotates the pattern
  about the boresight without moving it. The second is the alt-az keyhole
  degeneracy — real physics at a zenith-pointed mount, not an approximation —
  and it is why a pure azimuth offset is inert for a circularly symmetric beam
  and is not inert for the rectangular and elliptical apertures.
- The horizon gate is unchanged and is applied to the **true** topocentric
  altitude, never to the rotated one. A rotation of the beam frame does not move
  the ground: a direction below the true horizon stays zero-response whatever
  the rotation does to it, and a visible direction whose rotated altitude falls
  below zero is zeroed by the evaluator's own forward-hemisphere domain. The
  affected band is `|delta_el|` wide at the horizon, where the response is
  already negligible.

**Correction (7I implementation, 2026-08-02) — the Ruze voltage convention.**
The bullet above says `eta_s = exp(-(4 pi sigma / lambda)^2)` "is a single real
scalar multiplying the beam **voltage** amplitude". Those two clauses cannot
both hold. Ruze (1966) is a **gain** equation: `G = G_0 exp(-(4 pi sigma /
lambda)^2)`, and gain is power. RadioSim's `E` is a voltage beam — every
analytic form in `core/beam/analytic.py` is a voltage pattern, and the RIME
contracts `E_p B E_q^H`, so a factor `f` on the voltage appears as `f^2` in the
visibility of a baseline of two like antennas. Multiplying the voltage by
`eta_s` would therefore reduce the measured power by `eta_s^2`, i.e. would state
Ruze's equation and implement twice its exponent. The slice implements:

- `eta_s(lambda) = exp(-(4 pi sigma / lambda)^2)` — the **power** efficiency,
  the published Ruze quantity, resolved and reported under that name;
- the voltage factor applied to `E` is `sqrt(eta_s) = exp(-(1/2)(4 pi sigma /
  lambda)^2)`, so that the visibility amplitude on a baseline of two antennas
  with the same `sigma` is scaled by exactly `eta_s`.

This is the identical discipline Section 27's **I10** already fixed for the
tropospheric opacity — "the visibility amplitude is scaled by exactly
`exp(-tau_0)`, confirming the `exp(-tau/2)` voltage convention" — applied to the
one other efficiency-like scalar in the tier. **I19** is read accordingly: the
resolved Ruze factor `eta_s` equals `exp(-(4 pi sigma/lambda)^2)` at three
wavelengths, and the baseline amplitude it produces equals `eta_s`.

**Correction (7I implementation, 2026-08-02) — the accepted `beams` YAML.**
Section 36's ledger row B15 says "`beams` config gains pointing-offset and
surface-error fields" and Section 21 writes out only the `jones:` section, so
the accepted shape for these two is recorded here. Both blocks are optional, both
are available in all four `beams.mode` values (a pointing offset is a property of
the mount, not of whether the beam is analytic or tabulated), and both follow
Section 21.3's field-level rules — every angle carries `_deg`, every length `_m`,
`per_antenna` is keyed by the Tier 2 tagged antenna reference and rejects an
unknown or duplicated antenna:

```yaml
beams:
  mode: analytic
  model:
    kind: circular_aperture

  pointing:                              # optional; absent = no offset anywhere
    default:                             # optional array-wide default
      azimuth_offset_deg: 0.0
      elevation_offset_deg: 0.0
    per_antenna:                         # optional; overrides the default
      - antenna: {kind: number, number: 1}
        azimuth_offset_deg: 90.0
        elevation_offset_deg: 0.25

  surface_error:                         # optional; absent = no Ruze factor
    default:
      rms_surface_error_m: 0.001
    per_antenna:
      - antenna: {kind: name, name: ANT0}
        rms_surface_error_m: 0.004
```

Two rules follow the tier's own discipline rather than inventing a new one:

- **The R7 shape, applied to these blocks.** A `pointing` or `surface_error`
  block every one of whose authored numbers is zero is **rejected**, for the
  reason R7 rejects an identity Jones term: a block that is present and has no
  effect is the configuration surface that accepts a value and discards it. The
  check is purely syntactic — it needs no instrument — so it runs at parse time.
  A zero *entry* alongside a non-zero sibling is accepted and is the honest way
  to say "this antenna is perfectly pointed", and it is the path **I19**'s
  zero-offset clause is exercised through from configuration.
- **An inert resolved value is no value.** An offset of exactly `(0, 0)` and a
  surface error of exactly `0.0` resolve to *absent*, not to a stored zero. That
  is what makes **I19**'s "bit-identical to no offset" hold in its strongest
  form: not merely the same cube, but the same `assignment_fingerprint`, the
  same `state_fingerprint`, and the same `scientific_sha256`. Recording "no
  offset" for a zero offset is exact, not lossy — the two configurations are the
  same science.

### 19.3 What Tier 7 scopes but does not implement

`TODO.md` is rewritten from a wish list into an explicit **disposition table**:
for each remaining item, the physics, the citation, why it is out of Tier 7
scope, and the register row that owns it.

| `TODO.md` item | Disposition |
|---|---|
| Cross-polarization models: quadrupolar, IXR conversion, Ludwig-3 (`:3-38`) | Out of scope: requires a non-scalar `E` (Section 4). Routed to `SCI-005`. |
| Near/far field regime (`:40-45`) | Out of scope and **explicitly not needed** for astronomical sources, as `TODO.md:44` itself states. Recorded as a permanent non-goal for simulation, relevant only to holography. |
| Aperture blockage (`:47-52`) | Out of scope: an aperture-integral change to the analytic beam. Routed to `SCI-005`. |
| Ruze surface errors (`:54-59`) | **Efficiency factor implemented**; error-beam decomposition routed to `SCI-005`. |
| Systematic aberrations: defocus, coma, astigmatism, gravitational sag (`:61-67`) | Out of scope: requires a Zernike phase-error basis across the aperture. Routed to `SCI-005`. |
| Beam squint (`:69-74`) | Out of scope: requires polarization-dependent beams (non-scalar `E`). Routed to `SCI-005`. This is also why `BeamSquintLeakageJones` is deleted rather than implemented. |
| Pointing errors (`:76-81`) | **Deterministic offsets implemented**; the statistical gain-reduction formula `<G/G0> = [1 + 4 ln2 (sigma_p/theta_HPBW)^2]^-1` is documented as the expectation of the implemented deterministic model over a pointing distribution, not implemented as a separate stochastic path. |

The new register row to be filed at 7K:

- **`SCI-005` | ROADMAP | Advanced beam physics beyond the accepted scalar-`E`
  subset: polarized/cross-polar beams (Ludwig-3, quadrupolar, IXR), beam squint,
  aperture blockage, Zernike aberrations, and the Ruze error-beam
  decomposition. Each has explicit scientific scope and a citation in
  `src/radiosim/core/beam/TODO.md`; each requires widening the accepted E-Jones
  beyond a scalar diagonal | successor tier**.

`SCI-003` then closes **DONE**: every item has explicit scientific scope, two
have implementations with analytic verification, and the remainder has a named
owner. That is a complete and honest reading of its exit criterion.

### 19.4 Where `TODO.md` lives afterwards

It moves from `src/radiosim/core/jones/beam/TODO.md` to
`docs/development/beam_physics_scope.md` and is referenced from the `SCI-005`
register row. A scope document with a register owner belongs in tracked
documentation, not as a `TODO.md` inside an installed package.

## 20. Scientific conventions, citations, and exact mathematics

### 20.0 Conventions common to every term

- **Frames.** The topocentric horizontal frame is `(alt, az)` with azimuth
  measured **North through East**, matching the accepted beam frame
  (`pixel_coordinate_system == "az_za"`, `core/beam/fits.py:465-471`) and the
  Tier 5 feed-angle convention (`Tier5ReceptorFeedPlan.md` §12.1). The
  equatorial frame is ICRS; position angle is measured **North through East**.
- **Sign of a phase.** RadioSim's geometric phase is `exp(-2 pi i b.s)`
  (`visibility.py:697`). Every added propagation phase adopts the **same**
  `exp(-i phi)` sign convention for a positive excess path length, so that a
  positive delay `tau` gives `exp(-2 pi i nu tau)`. This is stated once here and
  tested once per term (Section 27, **I4**).
- **Units.** Frequency Hz; wavelength m; delay s; TEC in TECU
  (`1 TECU = 1e16 electrons m^-2`); rotation measure rad m^-2; opacity
  dimensionless (nepers, zenith); angles rad internally and deg in
  configuration, with the field name carrying the unit suffix exactly as
  `feed_rotation_deg` does.
- **Basis.** Every 2×2 matrix below is written in the **antenna's own receptor
  basis where the term sits in the chain** (Section 12.2). `Z`, `T`, and `P` are
  sky-side of `C` and are therefore written in the linear topocentric basis;
  `D`, `X`, `Kd`, `Rc`, `B`, `G` are correlator-side of `C` and are written in
  the receptor's basis, whatever that is. This is why `D` and `G` are defined
  per *feed index 0/1* rather than per `x`/`y`.
- **References** are cited per term below. `Fix.md` Section 16 rule 1 requires
  the citation to appear in the implementation's docstring, not only here.

### 20.1 G — complex electronic gain

**Reference.** Hamaker, Bregman & Sault (1996) A&AS 117, 137; Smirnov (2011)
A&A 527, A106, §6.

**Mathematics.** Direction-independent, frequency-independent by default,
diagonal:

```
G_p(t) = g_el(el_ref(t); p) * diag( g_p0(t), g_p1(t) )

g_pf(t) = (1 + a_pf) * exp(i * phi_pf) * s_pf(t)
```

where `a_pf` is a fractional amplitude error, `phi_pf` a phase error in radians,
`f in {0, 1}` the feed index in the antenna's own receptor basis, and `s_pf(t)`
the time model. `g_el` is the optional elevation gain curve, a polynomial in
elevation **of the pointing centre** (a direction-independent quantity, which is
why enabling it does not make `G` a DDE):

```
g_el(el) = sum_k c_k * el^k     (el in degrees, c_0 defaults to 1)
```

Time models (`time_model.kind`): `constant` (`s = 1`); `linear_drift`
(`s(t) = 1 + rate * (t - t0)`, `rate` per hour); `sinusoidal`
(`s(t) = 1 + depth * sin(2 pi (t - t0)/period + phase)`). All three are exactly
reproducible from configuration; none draws a random number.

**Per-antenna values** come from an explicit mapping keyed by antenna number, or
from a single array-wide value, with the same precedence discipline Tier 2 uses
for diameters (explicit per-antenna map beats array-wide default; a key naming
an antenna outside the resolved instrument is rejected).

**Invariants.** `G` is diagonal; `G` is **not** unitary unless every `a_pf` is
zero; `G` commutes with `B`, `Kd`, `Rc`, `X`; a pure phase error leaves all four
correlation amplitudes unchanged and changes only phases; a common amplitude
error `a` on both feeds of both antennas of a baseline scales every correlation
of that baseline by exactly `(1+a)^2`.

### 20.2 B — bandpass

**Reference.** Smirnov (2011) §6; CASA `bandpass` conventions
(van Moorsel et al., CASA Reference Manual, `B` Jones).

**Mathematics.** Direction-independent, frequency-dependent, diagonal:

```
B_p(nu) = diag( b_p0(nu), b_p1(nu) )
```

with `b_pf(nu)` from one of two models:

- `polynomial`: `b(nu) = sum_k c_k * x^k`, where
  `x = (nu - nu_ref) / nu_scale` is the normalized frequency, `nu_ref` defaults
  to the band centre and `nu_scale` to the half-bandwidth, so `x in [-1, 1]`
  across the band. Complex coefficients are accepted.
- `tabulated`: complex gains at explicit node frequencies, interpolated by cubic
  spline in the real and imaginary parts separately. Frequencies outside the
  node range are **rejected**, not extrapolated (rejection R11).

The removed `SplineBandpassJones` is the `tabulated` model; the removed
`PolynomialBandpassJones` is the `polynomial` model.

**Invariants.** `B` is diagonal and frequency-dependent; a constant polynomial
of value 1 is exactly `I2` and is therefore **rejected** as a no-op
configuration (R7); a real, frequency-flat bandpass is equivalent to a `G`
amplitude error, and the plan's test suite pins that equivalence as a
cross-term consistency check.

### 20.3 D — polarization leakage

**Reference.** Hamaker, Bregman & Sault (1996) §4; Sault, Hamaker & Bregman
(1996) A&AS 117, 149; Smirnov (2011) §6.4. IXR: Carozzi & Woan (2011) IEEE
TAP 59, 2058 (already cited in `beam/TODO.md:36`).

**Mathematics.** Direction-independent, optionally frequency-dependent,
non-diagonal, non-unitary. The standard first-order form, written in the
antenna's receptor basis with feed indices `0, 1`:

```
D_p(nu) = [[ 1,           d_p0(nu) ],
           [ -d_p1(nu)*,  1        ]]
```

`d_p0` is the leakage of feed 1's signal into feed 0's chain and `d_p1` the
converse; the conjugate-and-negate on the lower-left is the convention that
makes `D` reduce to a rotation for real, equal `d` values, which is the standard
HBS form. `d` is dimensionless and complex; `|d| ~ 0.01-0.05` is typical.

Parameterizations (`d_terms.kind`):

- `explicit`: complex `d_p0`, `d_p1` per antenna;
- `ixr`: an intrinsic cross-polarization ratio in dB per antenna, converted by
  `|d| = 1 / sqrt(IXR_lin)` with `IXR_lin = 10^(IXR_dB/10)` — equivalently
  `IXR_dB = -20 log10 |d|` — with a configured phase;

  **Correction (Tier 7E implementation, 2026-08-01).** This sentence previously
  read `|d| = (sqrt(IXR_lin) - 1) / (sqrt(IXR_lin) + 1)`, copied from
  `beam/TODO.md:16`, and that formula is **inverted**: it maps a *larger* IXR to
  a *larger* leakage, so a 30 dB antenna — an excellent one — would resolve to
  `|d| = 0.94`, and a 0 dB antenna — a completely depolarizing one — to
  `|d| = 0`. The derivation, from the reference the plan already cites: Carozzi
  & Woan (2011) define `IXR_J = ((kappa + 1)/(kappa - 1))^2` for the condition
  number `kappa` of the Jones matrix; for `D = [[1, d], [-d^*, 1]]` the singular
  values are `1 +- |d|`, so `kappa = (1 + |d|)/(1 - |d|)`. Writing
  `s = sqrt(IXR_lin) = (kappa + 1)/(kappa - 1)` gives `kappa = (s + 1)/(s - 1)`
  and therefore `|d| = (kappa - 1)/(kappa + 1) = 1/s`. The published formula is
  what `(kappa - 1)/(kappa + 1)` becomes if `kappa` and `s` are interchanged.
  The corrected relation has the two limits the physics requires: `|d| -> 0` as
  `IXR_dB -> infinity`, and `|d| = 1` at `IXR_dB = 0`. `beam/TODO.md:16` still
  carries the inverted form; correcting it belongs to Tier 7I, which owns that
  file and rewrites it as `docs/development/beam_physics_scope.md`;
- `frequency_polynomial`: `d(nu)` as a complex polynomial in normalized
  frequency, which is the deleted `FrequencyDependentLeakageJones`.

**Invariants.** `D` is non-unitary for any non-zero `d`; `D(0) = I2`; to first
order in `d`, an unpolarized source acquires cross-hand correlations
`V_01 ~ (I/2)(d_p0 - d_q1)` — an exact, checkable prediction, obtained by
expanding `D_p D_q^H` to first order for the `D_p = [[1, d_p0], [-d_p1*, 1]]`
convention adopted above (`D_q^H` contributes `-d_q1` at `[0,1]`, not
`+d_q1*`; a corrected sign/conjugate independently re-derived and verified
numerically during this design review — the review's Section 3 finding);
`det D = 1 + d_p0 d_p1*`, so `D` is invertible for physical leakages.

**Consequence that must be documented.** `docs/user_guide/jones_matrices.rst:137-146`
states that the receptor basis conversion `H` is exact **only** because `D` and
`G` are disabled. Implementing them makes that statement conditional, and 7E
must rewrite it: with `D` or a feed-asymmetric `G` enabled, converting a
circular-native antenna into a linear output basis (or the reverse) is an
approximation whose error is first order in `d` and in the feed gain ratio. The
plan requires that rewrite in the same slice that implements `D`, not later.

### 20.4 X — cross-hand phase and delay

**Reference.** CASA `crosshand phase` (`Xf`) and `KCROSS` calibration
conventions; Smirnov (2011) §6.

**Mathematics.** Direction-independent, diagonal, unitary. Cross-hand phase and
cross-hand delay are the same matrix with a frequency-constant and a
frequency-linear phase respectively, so they are one term:

```
X_p(nu) = diag( 1, exp( i * ( phi_x + 2 pi nu tau_x ) ) )
```

`phi_x` in radians, `tau_x` in seconds. Only the *relative* phase between the
two feeds is physical, which is why the first entry is exactly 1 rather than a
second free parameter — a second parameter would be degenerate with `G`.

**Invariants.** `X` is unitary and diagonal; `X` commutes with `G`, `B`, `Kd`,
`Rc`; for a linear receptor, a cross-hand phase `phi_x` rotates Stokes `U` into
Stokes `V` by exactly `phi_x` — the classic X-Y phase signature, and the sharpest
available test that the term is both correct and actually applied.

### 20.5 Kd — instrumental delay

**Reference.** Thompson, Moran & Swenson (2017), *Interferometry and Synthesis
in Radio Astronomy*, 3rd ed., §7; CASA `K` Jones.

**Mathematics.** Direction-independent, frequency-dependent, diagonal, unitary:

```
Kd_p(nu) = diag( exp(-2 pi i nu tau_p0), exp(-2 pi i nu tau_p1) )
```

`tau_pf` in seconds, per antenna and per feed. The sign matches Section 20.0.

**Invariants.** Unitary and diagonal; a delay common to both feeds of every
antenna produces a pure baseline-differential phase slope
`exp(-2 pi i nu (tau_p - tau_q))` and therefore **cancels exactly on a
zero-differential baseline** — the cleanest possible effect-is-applied test;
the phase is exactly linear in frequency, checkable by fitting.

### 20.6 Rc — cable reflection

**Reference.** Kern et al. (2020) ApJ 888, 70 (HERA systematics: cable
reflections as a delay-domain ripple); Beardsley et al. (2016) ApJ 833, 102.

**Mathematics.** Direction-independent, frequency-dependent, diagonal,
**non-unitary**:

```
Rc_p(nu) = diag( r_p0(nu), r_p1(nu) )
r_pf(nu) = 1 + A_pf * exp( -2 pi i nu tau_cable,pf + i phi_pf )
```

`A` a dimensionless reflection amplitude (`|A| < 1` enforced), `tau_cable` the
round-trip cable delay in seconds, `phi` a phase offset in radians. This is the
first-order (single-bounce) reflection; multiple bounces are out of scope and
the docstring says so.

**Invariants.** `|r|` oscillates between `1-A` and `1+A` with frequency period
`1/tau_cable`; the delay-domain (frequency-Fourier) transform of a spectrum
corrupted by `Rc` shows a secondary peak at exactly `tau_cable` with relative
amplitude `A` — the physically meaningful and directly testable signature, and
the reason `Rc` is a separate term from `B` rather than a bandpass shape.

### 20.7 P — parallactic angle and field rotation

**Reference.** Thompson, Moran & Swenson (2017) §4.5 and Appendix 4.1;
Hamaker, Bregman & Sault (1996) §5; the pyuvdata/CASA `parangle` convention;
Perley & Butler (2013) ApJS 206, 16 (polarization angle calibration).

**Mathematics.** Direction-**dependent** (this is the crucial correction over
the stub, which declared `is_direction_dependent = True` at
`parallactic.py:53-55` but computed nothing), time-dependent,
frequency-independent, real, unitary. For a direction with hour angle `H` and
declination `dec`, observed from geodetic latitude `lat`, the parallactic angle
is

```
psi(H, dec, lat) = atan2( sin H * cos lat,
                          sin lat * cos dec - cos lat * sin dec * cos H )
```

using the two-argument arctangent so the quadrant is correct over the whole
sky — the reason `DirectionBatch` carries `ra_rad`, `dec_rad`, and
`hour_angle_rad` (Section 13.2). The Jones factor is the real rotation

```
P_p(s, t) = R( eta_p * psi_p(s, t) ),
R(a) = [[ cos a,  sin a ],
        [ -sin a, cos a ]]
```

matching `R(chi)` in the accepted receptor mathematics
(`docs/user_guide/jones_matrices.rst:89-91`) exactly, so that `C_p P_p` composes
into `M(basis) R(chi + psi)` as Section 12.1 requires.

`eta_p` is the mount factor, resolved from each antenna's `mount_type`
(`core/instrument.py:316`), and is what makes heterogeneous arrays correct:

| `mount_type` | `eta` | Meaning |
|---|---|---|
| `alt-az` | `+1` | full parallactic rotation |
| `equatorial` | `0` | feeds track the sky; no rotation |
| `fixed` | `0` | the Tier 5 case; the static `chi` in `C` is the whole rotation |
| `alt-az+nasmyth-r` | `+1`, plus `+el` | Nasmyth right: `psi + el` |
| `alt-az+nasmyth-l` | `+1`, minus `el` | Nasmyth left: `psi - el` |

Any other `mount_type` value present in the resolved instrument is **rejected**
when `P` is enabled (R12) — never silently treated as `alt-az`, per rule 7.

**Wide-field behaviour.** `psi` is evaluated **per direction**, not once per
field. For a narrow field this reduces to a constant rotation; for a wide field
it varies measurably across the primary beam, which is exactly the effect the
deleted `FieldRotationJones` and `WidefieldPolarimetricJones` gestured at.
Section 27's **I9** invariant tests that `psi` varies across a wide direction
batch and that the narrow-field limit recovers the single-angle answer.

**Invariants.** `P` is real and orthogonal, so `P P^T = I2` exactly; `psi` is
antisymmetric in hour angle about transit for a source at the zenith meridian;
`psi = 0` identically for `dec = lat` at `H = 0`; an independently computed
`psi` from astropy's own frame machinery must agree to `1e-10` rad over a dense
`(H, dec, lat)` grid (Section 29's primary cross-implementation check).

**Two Tier 5 consequences, both required in the 7F slice.**

1. `core/receptor.py:411-418`'s blanket rejection of every non-`fixed`
   `mount_type` is **removed** and replaced by the `P`-aware rule: a non-`fixed`
   mount is accepted when `P` is enabled, and rejected with a message naming
   `jones.P` when it is not — because a fixed-feed treatment of an alt-az
   antenna is a silent scientific error, not a default.
2. `_reject_parallactic_rotation` (`visibility.py:76-96,382,800`) is
   **removed**. Its whole purpose was to prevent a static `chi` from being
   composed with a stub `P`; with `P` real and correctly placed sky-side of `C`,
   `C_p P_p = M(basis) R(chi) R(psi) = M(basis) R(chi + psi)` is the full,
   correct, time-dependent receptor orientation, which is precisely what
   `Tier5ReceptorFeedPlan.md` §12.3 and
   `docs/user_guide/jones_matrices.rst:148-158` said would happen "when Tier 7
   implements `P`".

### 20.8 Z — ionosphere (dispersive phase and Faraday rotation)

**Reference.** Thompson, Moran & Swenson (2017) §13.3; Intema et al. (2009)
A&A 501, 1185; Mevius et al. (2016) RaSc 51, 927 (LOFAR ionospheric RM);
Sotomayor-Beltran et al. (2013) A&A 552, A58 (`RMextract` conventions).

**Mathematics.** Direction-dependent, time-dependent, frequency-dependent,
unitary. Two physically distinct effects with the same electron column:

```
Z_p(s, nu, t) = exp( i * phi_TEC ) * R( psi_F )

phi_TEC = -(2 pi / c) * (e^2 / (8 pi^2 eps0 m_e)) * STEC / nu
        = -(2 pi * 1.34e9 / nu) * STEC_TECU        [rad, with STEC in TECU]

psi_F   = RM_ion * lambda^2                        [rad]
```

The dispersive term uses the standard constant `k_TEC = 1.3445e9 Hz TECU^-1`
(equivalently, an excess phase path of `40.308 * TEC / nu^2` metres, TMS
eq. 13.128), and is a **scalar** phase: it multiplies the identity, so it
commutes and does not depend on polarization. The Faraday term is the same real
rotation `R(a)` as `P`.

**Correction (7G implementation, 2026-08-01) — the Faraday factor is `R^T`,
not `R`.** The sentence above is right that the two are the same *kind* of
object and wrong about the orientation, and the difference is observable. `R(a)
= [[cos a, sin a], [-sin a, cos a]]` — the matrix `C` and `P` use — rotates the
**frame**, and therefore *lowers* the observed polarization angle by `a`:
`R B R^H` sends `(Q, U)` through a rotation of `-2a`. Faraday rotation rotates
the **field**, raising the observed angle by `psi_F`, so the matrix that
produces it is `R(psi_F)^T`. The sign is not free: `core/sky/containers/
spectral.py` already rotates a source's own `(Q, U)` by `+RM_src (lambda^2 -
lambda_ref^2)` — an accepted Tier 5C convention — and invariant **I8** requires
the sky's rotation and `Z`'s to **add**. Written with `R`, the composed angle
would be `RM_src - RM_ion` times `lambda^2` and I8 would fail on its own
arithmetic. The implementation therefore writes

```
Z_p(s, nu) = exp( i * phi_TEC ) * F( psi_F )

F(a) = [[ cos a, -sin a ],
        [ sin a,  cos a ]]  =  R(a)^T
```

and says so in its own docstring. Everything else in this section stands.

`STEC` is the **slant** TEC toward the direction, obtained from the configured
vertical TEC by a thin-shell mapping function at shell height `h`:

```
STEC(s) = VTEC / cos( arcsin( R_E * cos(el(s)) / (R_E + h) ) )
```

with `R_E = 6371 km` and `h` configurable (default 350 km). Vertical TEC models
(`tec.kind`): `constant` (one value); `gradient` (a linear gradient in
topocentric East and North, in TECU per km at the pierce point, which is the
minimal model that produces a *differential* between antennas and therefore a
real closure-visible effect).

`RM_ion` is configured directly in rad m^-2, per array or per antenna. It is
**not** derived from `STEC` and a geomagnetic field model, because RadioSim has
no magnetic-field model and Section 4 forbids adding data ingestion; the
docstring says so explicitly and points at `RMextract` for users who want a
physically derived value to supply.

**Invariants.** `Z` is unitary (`Z Z^H = I2` to machine precision) for every
parameter combination; the TEC phase scales exactly as `1/nu` and the Faraday
angle exactly as `1/nu^2`, both checkable by fitting across the band; a spatially
constant `VTEC` with zero gradient produces an **antenna-common, direction-varying**
phase whose baseline-differential part is non-zero only through the direction
dependence — so a single-source-at-zenith observation shows no visibility change
while a wide field does, a discriminating test; `psi_F = 0` and `phi_TEC = 0`
recover `I2` exactly and are rejected as a configuration (R7).

**Correction (7G implementation, 2026-08-01) — the wide-field half of that
sentence is wrong.** An antenna-common scalar phase cancels *exactly*, and the
field width is irrelevant to it: the RIME contracts each source as
`J_p C_s J_q^H`, and with `J_p = J_q = exp(i phi(s)) I2` that is
`exp(i phi) C_s exp(-i phi) = C_s`, source by source and therefore baseline by
baseline. A wide field changes nothing about this, because the cancellation
happens *inside* the sum rather than between its terms. The corrected statement,
which is what 7G tests:

- a `constant` screen changes **no** visibility at all, at any field width, and
  the test asserts that to `1e-14` rather than asserting a small change — an
  implementation that had accidentally made the screen antenna-dependent would
  fail loudly instead of passing a loose bound;
- a `gradient` screen does change them, because its pierce points differ between
  antennas. This is the same reason Section 20.8 already gives for offering the
  gradient model at all, so the correction makes the invariant agree with the
  parameterization rather than changing either;
- the Faraday half is **not** scalar and therefore survives on any array, for a
  polarized sky, even when both antennas share one rotation measure.

The identical statement holds for `T`'s delay (Section 20.9), whose only
antenna-differential parameter is the antenna height, and *not* for `T`'s
opacity, which is a real attenuation of each antenna's voltage and changes the
visibilities of any array. Both are tested.

**The D18 guard.** `Z`'s Faraday rotation and the sky model's per-source
`rotation_measure` are in different objects and different frames (Section 11.2).
The 7G slice adds the composition test **I8** and adds one sentence to the sky
documentation stating that `sky` RM is intrinsic and `jones.Z.faraday` is
ionospheric.

### 20.9 T — troposphere (delay and opacity)

**Reference.** Saastamoinen (1972), in *The Use of Artificial Satellites for
Geodesy*, AGU Geophys. Monogr. 15, 247; Niell (1996) JGR 101, 3227 (mapping
functions); Thompson, Moran & Swenson (2017) §13.1-13.2.

**Mathematics.** Direction-dependent (through elevation), frequency-dependent
(the delay phase), unitary only when opacity is disabled:

```
T_p(s, nu) = a_opacity(s) * exp( -2 pi i nu tau_trop(s) ) * I2

tau_trop(s) = ( ZHD * m_h(el) + ZWD * m_w(el) ) / c
a_opacity(s) = exp( -tau_0(nu) / (2 * sin el) )
```

Both factors are **scalars times the identity**, so `T` is scalar and commutes;
this is stated and tested rather than assumed.

- `ZHD` (zenith hydrostatic delay, metres) is either configured directly or
  computed by the **Saastamoinen** formula — the model the deleted
  `SaastamoinenTroposphereJones` named:
  `ZHD = 0.0022768 * P_0 / (1 - 0.00266 cos(2 lat) - 0.00028 h_km)` with surface
  pressure `P_0` in hPa, antenna geodetic latitude, and height in km. The
  antenna height already exists on the resolved instrument.
- `ZWD` (zenith wet delay, metres) is configured directly; no wet model is
  offered, because every credible one needs humidity and temperature profiles
  RadioSim does not have (Section 4).
- Mapping functions `m_h`, `m_w`: `simple` (`1/sin el`) and `niell` (the
  three-term continued fraction of Niell 1996, with its published coefficients).
- The **factor of 2** in the opacity exponent is deliberate and is the single
  easiest sign-class error in this term: `T` is a *voltage* Jones matrix, and
  the opacity `tau_0` is defined on *power*, so the voltage attenuation is
  `exp(-tau/2)` per antenna, giving `exp(-tau)` in the power-like product
  `J_p C J_q^H` for a baseline of two identical antennas. Section 27's **I10**
  invariant pins exactly this.

**Invariants.** `T` is scalar (proportional to `I2`) for every parameter
combination; with opacity disabled `T` is unitary; the delay phase is exactly
linear in frequency; the opacity attenuation at zenith on a baseline of two
identical antennas is exactly `exp(-tau_0)` in visibility amplitude; the
elevation dependence of `1/sin(el)` diverges at the horizon and is therefore
**rejected below a configurable minimum elevation** rather than allowed to
produce infinities (R13).

### 20.10 M — baseline multiplicative closure error

**Reference.** Smirnov (2011) §1.6 and §7 (baseline-dependent errors as the
residual that closure quantities cannot absorb); TMS (2017) §10.3.

**Mathematics.** A `JonesBaselineTerm`, direction-independent, applied by
Hadamard product to the contracted `(B, 2, 2)` block:

```
V_pq  ->  M_pq (*) V_pq
```

`M_pq` is a complex 2×2 of multiplicative errors, configured per baseline
(keyed by the ordered antenna-number pair) or as one array-wide value. A key
naming a pair that is not in the resolved baseline selection is **rejected**
(R14) rather than ignored.

**Invariants.** `M` is *not* expressible as any product of per-antenna Jones
matrices — that is its defining property. The plan's test asserts it
constructively: a configured `M` breaks the closure phase of a triangle by a
specific, predicted amount, whereas any per-antenna `G` leaves closure phase
exactly invariant. That single test simultaneously proves `M` is applied, proves
it is baseline-dependent, and proves the Hadamard path is distinct from the
matrix chain — which is exactly what `Fix.md` Section 16's Workstream D asks
for ("enforce the distinction").

**Correction (7H implementation, 2026-08-01) — `M`'s two configuration sources
and its duplicate rejection.** This section already says `M` is configured
"per baseline ... or as one array-wide value", while Section 21.2's YAML shows
only `per_baseline`. Both are implemented, with the same precedence rule every
other term uses: an array-wide `matrix` is the default for every selected
baseline, a `per_baseline` entry overrides it for the pair it names, and a
baseline named by neither is exactly `I2`. A block that configures neither, or
whose every resolved matrix is `I2`, is the R7 identity rejection rather than a
new one — an `M` that cannot break closure is an `M` that is not there.
A duplicate `per_baseline` entry is rejected with R5's sentence reduced to the
key that exists, exactly as R5 already provides for a term whose overrides carry
no feed index: `"jones.M.per_baseline contains a duplicate entry for baseline
(<p>, <q>); each baseline may appear once."` `M` is the only term in this tier
whose overrides are keyed by a baseline, so this is the second and last bounded
instance of that named exception, not a general licence.

**Correction (7H implementation, 2026-08-01) — the neutral element of `M` is the
all-ones matrix, not the identity, and Section 21.2's example is wrong.** The
plan's YAML writes `M` as `[[1.02, 0], [0, 0.98]]`, which reads as a diagonal
Jones matrix and is not one: under `(*)` the off-diagonal zeros multiply `V_XY`
and `V_YX` by zero and **null both cross-hands**. The neutral element of a
Hadamard product is `[[1, 1], [1, 1]]`, so that — and not `I2` — is what an
unnamed baseline carries, and what R7 rejects as "exactly the identity". Written
the other way round, a user copying the plan's own example would silently lose
their cross-hand correlations while a rejection told them their configuration
did nothing. The example is corrected in the Section 21.3 block above, the four
entries are described as what they are (a multiplicative factor per correlation,
with `1` meaning "unchanged" and `0` meaning "nulled"), and an `M` of identity
matrices is *accepted*, because nulling both cross-hands is a real configured
effect.

**Correction (7H implementation, 2026-08-01) — new rejection R17.** A complex
parallel-hand factor on an **autocorrelation** is rejected:

> `"jones.M assigns a parallel-hand factor with a non-zero imaginary part to
> autocorrelation baseline (<p>, <p>); an autocorrelation's parallel hands are
> real by construction."`

`<E_x E_x^*>` is real and non-negative and has no phase for a multiplicative
error to corrupt, and the rejection is forced rather than fastidious: this
slice's own end-to-end runs found that an array-wide complex `M` produces a cube
RadioSim's Measurement Set and UVFITS writers refuse as unrepresentable, so
without R17 the failure lands *after* the whole simulation has run instead of
before the first side effect (Section 26.1). The cross-hand entries of an
autocorrelation are unconstrained, because `<E_x E_y^*>` of one antenna is
genuinely complex. R17 belongs to stage 4, the physical-range stage.

### 20.11 Q — time and bandwidth smearing

**Reference.** Bridle & Schwab (1999), in *Synthesis Imaging in Radio
Astronomy II*, ASP Conf. Ser. 180, 371; TMS (2017) §6.4; Smirnov (2011) §7.2.

**Mathematics.** A `JonesBaselineTerm`, direction-**dependent**, real, scalar,
folded into the kernel's `envelope`:

```
Q_pqs = sinc( pi * dnu * tau_g,pqs ) * sinc( pi * dt * dphi/dt|_pqs )
```

where `sinc(x) = sin(x)/x`, `dnu` is the channel width in Hz, `dt` the
integration time in seconds, `tau_g,pqs = (b . s)/c` the geometric delay for
baseline `pq` toward direction `s`, and `dphi/dt` the fringe rate

```
dphi/dt = 2 pi * omega_E * ( u_pq * cos(dec_s) ... )
```

evaluated as `2 pi * omega_E * (-u * sin(H) * cos(dec) + ...)` — the exact
expression, with `omega_E = 7.2921150e-5 rad/s`, derived from the time
derivative of `b.s` at fixed source coordinates. Both factors are `<= 1` and
equal `1` for a source at the phase centre, which is the physically correct
statement that smearing does not decorrelate the phase centre.

`dnu` comes from the resolved observation frequency configuration and `dt` from
the resolved time grid; **neither is a free parameter of the term**, because
inventing a smearing integration time that disagrees with the time grid the
solver actually uses would be a fabrication. The term's own configuration
selects only which of the two factors are active.

**Invariants.** `0 < Q <= 1` always; `Q = 1` exactly at the phase centre;
`Q -> 1` as `dnu -> 0` and `dt -> 0`; smearing reduces amplitude and never
changes phase (it is real); the amplitude reduction on the longest baseline
toward the field edge exceeds that on the shortest, monotonically — a structural
test that catches a sign or a `u`/`v` transposition.

**Correction (7H implementation, 2026-08-01) — four points, each forced by
RadioSim's own conventions rather than by convenience.**

1. **`tau_g` is the delay *relative to the phase centre*, not `b . s`.** Written
   as `b . s / c` the bandwidth factor would be `1` nowhere, least of all at the
   phase centre: at zenith `b . s` is the baseline's own vertical component.
   RadioSim's kernel phase is `exp(-2 pi i (u l + v m + w (n - 1)))`
   (`core/jones/geometric.py`), so the residual delay the correlator has *not*
   removed is

   ```
   tau_res(b, s) = ( u l + v m + w (n - 1) ) / nu
                 = ( b_E l + b_N m + b_U (n - 1) ) / c
   ```

   which vanishes at `l = m = 0, n = 1` and makes the bandwidth factor exactly
   `1` at the phase centre. The `-1` is the same fringe-stopping term Section
   20.0 says `K` carries exactly, and `Q` must be written against the phase the
   kernel actually applies.

2. **The fringe rate, written out.** This section's expression is elided
   (`u * cos(dec_s) ...`) and its fragment is written for a tracking equatorial
   `(u, v, w)` frame, which RadioSim does not have: its baselines are constant
   ENU vectors and its phase centre is the **fixed zenith**, so the entire time
   dependence is the sky rotating through both. Differentiating the kernel phase
   at fixed catalogue coordinates, with `dH/dt = omega_E` and
   `p = (0, cos(lat), sin(lat))` the celestial pole in ENU, gives
   `ds/dt = -omega_E (p x s)`, hence

   ```
   dl/dt = -omega_E ( n cos(lat) - m sin(lat) )      ( = -omega_E cos(dec) cos(H) )
   dm/dt = -omega_E l sin(lat)
   dn/dt = +omega_E l cos(lat)
   ```

   and therefore a fringe rate, **in cycles per second**, of

   ```
   nu_f(b, s) = omega_E [ u ( n cos(lat) - m sin(lat) ) + l ( v sin(lat) - w cos(lat) ) ]
   ```

   with `(u, v, w)` in wavelengths. The time factor is `sinc(pi dt nu_f)`, which
   is the standard `sinc(dt dphi/dt / 2)` with `dphi/dt = 2 pi nu_f`; both
   factors are evaluated as `numpy.sinc(x) = sin(pi x)/(pi x)`, so the value at
   zero argument is exactly `1` and no zero-division guard is needed.

3. **`dnu` and `dt` come from the resolved grids, and Q6's candidate rule is
   rejected.** `dnu` is `ResolvedFrequencyConfig.channel_widths_hz[freq_idx]`
   and `dt` is `ObservationTimeGrid.integration_time_seconds[time_idx]`. Q6
   proposed deriving a width from the spacing to the neighbouring channel on a
   nonuniform grid, with a single-channel observation rejected; that is not
   needed and would be **wrong**, because Tier 1G made a nonuniform explicit
   frequency array first-class *together with* a required per-channel
   `channel_widths_hz`, and that same width is what the result cube and the
   summary already report. A `Q` that smeared over a spacing-derived width would
   decorrelate by a bandwidth the run does not claim to have. A single-channel
   observation is accepted for the same reason: it has a declared width like
   every other. See the Section 41 Q6 resolution.

4. **The two invariant clauses that RadioSim's phase convention changes.**
   `Q <= 1` always. `Q > 0` while the smearing argument is below the first sinc
   zero; beyond it the exact top-hat average genuinely changes sign, and
   clamping it to keep the "`0 < Q`" half of the clause literally true would be
   a fabrication — so the implementation computes the exact sinc and the
   documentation states where the first zero is. `Q = 1` exactly at the phase
   centre holds for the **bandwidth** factor exactly; it holds for the **time**
   factor only on a baseline with no East-West component, because RadioSim's
   phase centre is the fixed zenith rather than a tracked source, so a source at
   the zenith still moves through it during an integration and still
   decorrelates by `sinc(pi dt omega_E u cos(lat))`. That is real physics of a
   drift-scan correlator, not an error: asserting unity there would assert that
   the array tracks. Invariant I12 is corrected to match.

### 20.12 The chain-order summary table

| Position | Term | Class | DDE | Diagonal | Unitary | Scalar |
|---|---|---|---|---|---|---|
| 1 (correlator) | H | `BasisTransformJones` | no | no | yes | no |
| 2 | G | `GainJones` | no | yes | only if lossless | no |
| 3 | B | `BandpassJones` | no | yes | no | no |
| 4 | Rc | `CableReflectionJones` | no | yes | no | no |
| 5 | Kd | `DelayJones` | no | yes | yes | no |
| 6 | X | `CrosshandJones` | no | yes | yes | no |
| 7 | D | `PolarizationLeakageJones` | no | no | no | no |
| 8 | C | `ReceptorConfigJones` | no | no | yes | no |
| 9 | E | `_ResolvedBeamJones` | yes | yes | no | **yes** |
| 10 | P | `ParallacticAngleJones` | yes | no | yes | no |
| 11 | T | `TroposphereJones` | yes | yes | only without opacity | **yes** |
| 12 (sky) | Z | `IonosphereJones` | yes | no | yes | no |
| separate | K | `geometric_phase()` | yes | — | yes | yes |
| Hadamard | M | `BaselineMultiplicativeJones` | no | — | — | — |
| Hadamard | Q | `SmearingFactorJones` | yes | — | — | real scalar |

Every "yes" in the Diagonal, Unitary, and Scalar columns is a **verified**
claim, not a self-report: Section 27's **I2** invariant sweeps each term over
its parameter space and asserts the declared flags numerically. This is the
structural fix for D10.

## 21. Exact configuration schema

### 21.1 Placement and style

A new top-level section, `jones:`, on `RadioSimConfig`, defined in a new module
`src/radiosim/io/jones_config.py` and imported into `io/config.py` — the same
placement pattern Tier 5 used for `receptors:` (`io/receptor_config.py`). Every
model is a `StrictFrozenModel`: extra keys forbidden, instances frozen, unknown
fields rejected at parse time with the standard Tier 1 renderer.

Model-variant blocks use a **discriminated union on `kind`**, matching the
`SkySourceConfig` and beam-model precedent, so an unknown `kind` is rejected by
Pydantic itself rather than by a hand-written branch.

### 21.2 The accepted YAML, in full

```yaml
jones:
  # Every term is absent by default.  An absent term is not in the chain.
  # A present term must have a non-identity effect (rejection R7).

  G:                                  # electronic gain
    amplitude_error: 0.02             # fractional, array-wide default
    phase_error_rad: 0.0
    per_antenna:                      # optional; overrides the defaults
      - antenna: 12
        feed: 0                       # 0 or 1, in the antenna's own basis
        amplitude_error: 0.05
        phase_error_rad: 0.13
    elevation_curve: [1.0, -1.0e-4]   # optional; polynomial in elevation (deg)
    time_model:                       # optional; default {kind: constant}
      kind: linear_drift              # constant | linear_drift | sinusoidal
      rate_per_hour: 0.01

  B:                                  # bandpass
    model:
      kind: polynomial                # polynomial | tabulated
      coefficients: [1.0, 0.0, -0.05] # complex accepted as [re, im] pairs
      reference_frequency_hz: null    # null = band centre
      scale_frequency_hz: null        # null = half-bandwidth
    per_antenna: []                   # same override shape as G

  D:                                  # polarization leakage
    d_terms:
      kind: explicit                  # explicit | ixr | frequency_polynomial
      d0: [0.02, 0.0]                 # [re, im]
      d1: [0.0, 0.02]
    per_antenna: []

  X:                                  # cross-hand phase and delay
    phase_rad: 0.1
    delay_s: 0.0
    per_antenna: []

  Kd:                                 # instrumental delay
    delay_s: 1.0e-9                   # array-wide default, per feed
    per_antenna: []

  Rc:                                 # cable reflection
    amplitude: 0.01                   # |A| < 1
    cable_delay_s: 1.5e-7
    phase_rad: 0.0
    per_antenna: []

  P:                                  # parallactic angle / field rotation
    enabled: true                     # P has no other required parameter;
                                      # `enabled` is the whole configuration

  Z:                                  # ionosphere
    tec:
      kind: constant                  # constant | gradient
      vertical_tec_tecu: 10.0
    shell_height_km: 350.0
    minimum_elevation_deg: 5.0        # required; see the 7G correction below
    faraday:                          # optional
      rotation_measure_rad_m2: 0.5
      per_antenna: []

  T:                                  # troposphere
    zenith_delay:
      kind: saastamoinen              # explicit | saastamoinen
      surface_pressure_hpa: 1013.25   # saastamoinen only
      zenith_hydrostatic_delay_m: 2.3 # explicit only
      zenith_wet_delay_m: 0.05        # both variants
    mapping_function: niell           # simple | niell
    opacity:                          # optional
      zenith_opacity: 0.02
    minimum_elevation_deg: 5.0

  M:                                  # baseline closure error
    per_baseline:
      - antennas: [0, 1]
        matrix: [[[1.02, 0.0], [0.0, 0.0]],
                 [[0.0, 0.0], [0.98, 0.0]]]

  Q:                                  # time and bandwidth smearing
    bandwidth_smearing: true
    time_smearing: true
```

### 21.3 Field-level rules

- Every angle field carries a `_rad` or `_deg` suffix; every time field `_s`;
  every frequency field `_hz`; every length `_m` or `_km`. No unit is implicit.
- Complex numbers are `[re, im]` two-element sequences, never Python complex
  literals in YAML, and never a string. This matches how the config layer
  already handles structured numeric data and keeps the YAML round-trippable.
- `per_antenna` entries are keyed by **antenna number**, validated against the
  resolved instrument, and rejected on an unknown number or a duplicate
  `(antenna, feed)` pair — the same discipline as the Tier 5 receptor overrides
  and the Tier 2 diameter overrides.
- `P` is the only term whose block is a bare `enabled` flag, because the
  parallactic angle has no free parameter: it is fully determined by the
  instrument, the time grid, and the directions. Making it look like the other
  terms by inventing a parameter would be dishonest.
  **Correction (7F implementation, 2026-08-01):** Section 21.2's `P` block
  previously also carried `minimum_elevation_deg`, whose own comment said the
  directions it names "are already masked". It is removed, because this rule
  says the block is a *bare* `enabled` flag and because a field that is
  documented as having no effect is the defect-D2 shape one level up — a
  configuration surface that accepts a value and discards it. The
  `minimum_elevation_deg` field survives on `T` and `Z` (R13), where the mapping
  function genuinely diverges.
  **Correction (7G implementation, 2026-08-01):** Section 21.2's `Z` block did
  not carry the field the sentence above says survives on it, and Section 24's
  R13 names both terms; the field is therefore added to the `Z` block, where the
  thing it guards is the validity of the thin-shell approximation rather than a
  divergence (the slant factor is bounded — about 3.13 at the horizon). On both
  terms it is **required and has no default**: where a model stops being trusted
  is a scientific decision, a default would silently make it RadioSim's, and
  `0` is the explicit way to accept every direction the horizon mask passes.
- `Q` likewise takes only two booleans, because `dnu` and `dt` come from the
  resolved observation configuration (Section 20.11).
  **Correction (7H implementation, 2026-08-01):** both booleans are
  **required** and neither has a default, for the same reason `P.enabled` is
  required — which of the two mechanisms a run models is a scientific decision,
  and a default would silently make it RadioSim's. `M`'s block gains the
  array-wide `matrix` this plan's Section 20.10 already describes, alongside the
  `per_baseline` overrides Section 21.2 shows:

  ```yaml
    M:
      matrix: [[[1.01, 0.0], [0.99, 0.0]],     # optional array-wide default
               [[0.99, 0.0], [1.01, 0.0]]]     # every entry is a factor; 1 = unchanged
      per_baseline:                            # optional overrides
        - antennas: [0, 1]
          matrix: [[[1.02, 0.03], [0.98, -0.01]],
                   [[0.97, 0.01], [1.04, 0.02]]]
  ```

  Section 21.2's own `M` example is **replaced** by this one rather than kept
  alongside it: the original's off-diagonal zeros null both cross-hands under a
  Hadamard product, which is not what a "diagonal-looking" example means to a
  reader. See the Section 20.10 correction.

## 22. Exact resolved runtime model

```python
# src/radiosim/core/jones_terms.py  (new)

@dataclass(frozen=True)
class ResolvedJonesTerms:
    """The one canonical, frozen Jones-term inventory for a run."""
    chain_terms: tuple[JonesTerm, ...]          # in canonical order, H first
    baseline_terms: tuple[JonesBaselineTerm, ...]
    dtypes: ResolvedJonesDtypes
    provenance: JonesProvenance

@dataclass(frozen=True)
class ResolvedJonesDtypes:
    by_term: Mapping[str, tuple[DTypeLike, DTypeLike]]   # letter -> (complex, real)
    accumulation_complex: DTypeLike

@dataclass(frozen=True)
class JonesProvenance:
    enabled_terms: tuple[str, ...]              # ("H","G","C","E","P", ...)
    chain_order: tuple[str, ...]                # the full composed order
    term_snapshots: FrozenDict                  # letter -> frozen config snapshot
    mount_types: FrozenDict                     # antenna number -> mount type
    jones_sha256: str
```

Resolution rules, following `Fix.md` Section 4.3 (precedence must be
centralized, documented, and in provenance):

1. `resolve_jones_terms()` runs **once**, in `Simulator.setup()`, before any
   solver call and before any beam or sky work that could depend on it.
2. It is the only place a `JonesTerm` is constructed. Solvers receive
   `ResolvedJonesTerms` and never see raw configuration.
3. `H`, `C`, and `E` are always present, exactly as today
   (`visibility.py:806,852,855-863`). Every other term is present only if its
   config block is present.
4. `chain_terms` is emitted in the canonical Section 12.2 order regardless of
   the order the user wrote the YAML keys in, so the chain shape is a function
   of *which* terms are enabled, never of file ordering.
5. Precedence within a term: an explicit `per_antenna` entry beats the
   array-wide default; there is no third source, and no environment variable or
   CLI override for Jones parameters.

## 23. Public API changes

```python
# radiosim.core.jones  -- removed (26 names)
GeometricPhaseJones, TimeVariableGainJones, ElevationGainJones,
PolynomialBandpassJones, SplineBandpassJones, RFIFlaggedBandpassJones,
IXRLeakageJones, MuellerLeakageJones, BeamSquintLeakageJones,
FieldRotationJones, VLBIFeedRotationJones, TurbulentIonosphereJones,
GPSIonosphereJones, SaastamoinenTroposphereJones, TurbulentTroposphereJones,
TroposphericOpacityJones, FaradayRotationJones, DifferentialFaradayJones,
WPhaseJones, WProjectionJones, WidefieldPolarimetricJones, ElementBeamJones,
ArrayFactorJones, DifferentialBeamJones, FringeFitJones,
CrosshandDelayJones, FrequencyDependentLeakageJones

# radiosim.core.jones  -- renamed
CrosshandPhaseJones -> CrosshandJones

# radiosim.core.jones  -- new
geometric_phase              # function, replaces GeometricPhaseJones
DirectionBatch               # from .directions
evaluate_antenna_jones       # from .evaluate

# radiosim.core.jones.base -- changed ABC
JonesTerm.compute_jones                 REMOVED
JonesTerm.compute_jones_all_sources     REMOVED
JonesTerm.compute_jones_batch           NEW, keyword-only   (see correction)
JonesTerm.term_status                   NEW, property -> "implemented"

# radiosim.core.jones.chain -- changed
JonesChain.compute_antenna_jones                REMOVED
JonesChain.compute_antenna_jones_all_sources    REMOVED
JonesChain.compute_baseline_visibility          REMOVED
JonesChain.compute_antenna_jones_batch          NEW
JonesChain.add_term                             now rejects JonesBaselineTerm

# radiosim.core.jones.baseline_errors -- changed ABC
JonesBaselineTerm.compute_baseline_term         REMOVED from the ABC
JonesBaselineTerm.compute_baseline_factor       NEW, batched  (see correction)

# radiosim.core -- new
ResolvedJonesTerms, ResolvedJonesDtypes, JonesProvenance, resolve_jones_terms

# radiosim.io -- new
JonesConfig and its per-term models

# radiosim.io.config -- removed
VisibilityConfig.calculation_type

# solver signatures -- removed parameter
simulate_visibilities(..., jones_config=...)          -> jones_terms: ResolvedJonesTerms
calculate_visibilities(..., jones_config=...)         -> jones_terms: ResolvedJonesTerms
calculate_visibility_healpix(...)                     -> gains jones_terms: ResolvedJonesTerms
RIMESimulator.simulate(..., jones_config=...)         -> jones_terms
VisibilitySimulator.simulate(..., jones_config=...)   -> jones_terms
```

**Correction (7B implementation, 2026-08-01)** — three points on the table
above:

- `compute_jones_batch` and `compute_baseline_factor` are introduced as concrete
  methods that raise `NotImplementedError` naming the class, not as
  `@abstractmethod`, and become abstract in the slice that deletes the last
  subclass that does not implement them (7C for `JonesTerm`, 7H for
  `JonesBaselineTerm`).  Section 33.2's 7B correction gives the reasoning.
- `compute_baseline_term` is removed from the `JonesBaselineTerm` ABC at 7B; the
  `M` and `Q` stub bodies that still define it are Tier 7H's to replace, and are
  outside 7B's writable list.
- `GeometricPhaseJones` is removed at **7B**, not 7C, together with the three
  names 7B adds: `geometric_phase`, `DirectionBatch` and
  `evaluate_antenna_jones`.  The "removed (26 names)" list above is therefore
  25 names at 7C plus this one at 7B.  `term_status` is *not* added at 7B: with
  35 identity stubs still exported, a base-class default of `"implemented"`
  would be a lie on every one of them, so it lands with 7C's stub deletion,
  which is also where invariant I20 first asserts it.

## 24. Exact rejection messages

Every rejection is a typed error with a verbatim message, tested by exact
string. `<...>` denotes an interpolated value.

| # | Trigger | Error type | Message |
|---|---|---|---|
| R1 | `visibility.calculation_type` present in a config document | `ConfigIssue` (removed-field guidance) | `"visibility.calculation_type was removed before v1.0; the solver strategy is selected by 'execution.simulator' (currently only 'rime')."` |
| R2 | `jones` present but empty (`jones: {}`) | `InvalidJonesConfigError` | `"jones: is present but configures no term; remove the section or configure at least one term."` |
| R3 | unknown key under `jones:` | Pydantic strict | standard Tier 1 unknown-field rendering, listing the accepted term letters |
| R4 | `per_antenna` names an antenna number absent from the resolved instrument | `JonesAssignmentError` | `"jones.<TERM>.per_antenna references antenna number <n>, which is not in the resolved instrument; known numbers are <...>."` |
| R5 | duplicate `(antenna, feed)` in one term's `per_antenna` | `InvalidJonesConfigError` | `"jones.<TERM>.per_antenna contains a duplicate entry for antenna <n> feed <f>; each (antenna, feed) may appear once."` — for a term whose `per_antenna` carries no feed key (Tier 7E: `X`, whose one parameter is the antenna's own relative phase, not a per-feed value), the message is this same sentence with the pair reduced to the key that exists: `"jones.<TERM>.per_antenna contains a duplicate entry for antenna <n>; each antenna may appear once."` Naming a feed the configuration never wrote would be worse than adapting the wording, and no other term in Tier 7 is feedless, so this is a bounded, named exception rather than a general license to reword R5. |
| R6 | `feed` not in `{0, 1}` | `InvalidJonesConfigError` | `"jones.<TERM>.per_antenna feed=<f> is invalid; feeds are indexed 0 and 1 in the antenna's own receptor basis."` |
| R7 | a term's resolved parameters make it exactly the identity for every antenna, frequency, time, and direction | `IdentityJonesTermError` | `"jones.<TERM> is configured with parameters that make it exactly the identity; a term that cannot change the visibilities must be removed rather than configured."` |
| R8 | `Rc.amplitude` outside `(0, 1)` | `InvalidJonesConfigError` | `"jones.Rc.amplitude=<a> must satisfy 0 < |A| < 1; a reflection cannot return more power than it receives."` |
| R9 | `Z.tec.vertical_tec_tecu` negative | `InvalidJonesConfigError` | `"jones.Z.tec.vertical_tec_tecu=<v> must be non-negative."` |
| R10 | `T.opacity.zenith_opacity` negative | `InvalidJonesConfigError` | `"jones.T.opacity.zenith_opacity=<t> must be non-negative; a negative opacity would amplify."` |
| R11 | tabulated bandpass node range does not cover every observed channel | `InvalidJonesConfigError` | `"jones.B tabulated nodes span <lo>-<hi> Hz but the observation covers <olo>-<ohi> Hz; RadioSim does not extrapolate a bandpass."` |
| R12 | `P` enabled and some antenna has a `mount_type` `P` does not model | `UnsupportedMountTypeError` | `"antenna <n> has mount_type=<m>, which the parallactic-angle term does not model; supported mounts are alt-az, equatorial, fixed, alt-az+nasmyth-l, alt-az+nasmyth-r."` |
| R13 | `T` or `Z` enabled and a direction survives the horizon mask below `minimum_elevation_deg` | `InvalidJonesConfigError` | `"jones.<TERM>.minimum_elevation_deg=<e> excludes no direction, but the mapping function diverges below <e> deg; raise the minimum elevation or the horizon mask."` |
| R14 | `M.per_baseline` names a pair absent from the resolved baseline selection | `JonesAssignmentError` | `"jones.M.per_baseline references baseline (<p>, <q>), which is not in the resolved baseline selection."` |
| R15 | a non-`fixed` `mount_type` is present and `P` is **not** enabled | `UnsupportedMountTypeError` | `"antenna <n> has mount_type=<m>, whose feeds rotate with the sky; enable 'jones.P' or the simulation would silently treat it as a fixed mount."` |
| R16 | `Q` enabled with `bandwidth_smearing: false` and `time_smearing: false` | `InvalidJonesConfigError` | `"jones.Q is enabled with both smearing kinds disabled; remove the section instead."` |
| R17 | `M` assigns a parallel-hand factor with a non-zero imaginary part to an autocorrelation baseline (**added by the 7H implementation, 2026-08-01**; see the Section 20.10 correction for why it is forced) | `InvalidJonesConfigError` | `"jones.M assigns a parallel-hand factor with a non-zero imaginary part to autocorrelation baseline (<p>, <p>); an autocorrelation's parallel hands are real by construction."` |

R15 is the replacement for `core/receptor.py:411-418`'s current blanket
rejection, and it is a **strictly better** contract: it names the fix rather
than the tier.

**Correction (7G implementation, 2026-08-01) — R13's stage, and its wording for
`Z`.** Two changes, both forced, neither touching what is rejected:

1. **R13 is raised at evaluation, not at resolution.** Its trigger is a
   statement about *directions* — "a direction survives the horizon mask below
   `minimum_elevation_deg`" — and no direction exists until a solver has
   resolved one for a `(time, frequency)` step: Section 26.1's stages 3-6 run
   before any sky is loaded, precisely so that they can. The only thing
   resolution could compare `minimum_elevation_deg` against is the solvers'
   horizon mask, which is the constant `alt > 0` and not configurable, so a
   stage-5 R13 would fire for *every* positive minimum elevation and leave `T`
   and `Z` with no accepted configuration. The two terms therefore raise it
   themselves, from `compute_jones_batch`, with the same
   `InvalidJonesConfigError` type and the same message. Section 26.1's stage 5
   keeps everything about those blocks that is decidable without a sky.
2. **R13's final clause is adapted for `Z`.** The message says the mapping
   function "diverges below `<e>` deg", which is true of `T`'s `1/sin(el)` and
   false of `Z`'s thin-shell factor, which is bounded at the horizon. `Z`'s
   message reads "but the thin-shell mapping function is not valid below `<e>`
   deg" and is otherwise R13 verbatim. This is the same bounded, named exception
   R5 already carries for a term with no feeds, and for the same reason: a
   rejection that states something untrue about the term it names is worse than
   one whose wording is adapted to the physics. It is not a licence to reword a
   rejection in general.

**Correction (7F implementation, 2026-08-01) — R7, R12 and R15 for `P`.** Read
literally, the three triggers above are mutually unsatisfiable for two real
instruments, and one of them regresses a Tier 5 protection. The messages are
unchanged; only the triggers are made precise, and each change is forced:

1. **R12 does not depend on `P` being enabled.** Its trigger becomes "some
   antenna has a `mount_type` outside the five `P` models", with or without
   `jones.P`. Gating it on `P` would mean that an antenna whose mount is
   `phased` — rejected outright by Tier 5 today — is silently treated as
   `fixed` in any run that does not configure `P`. That is a *regression*
   against the rejection this slice is replacing, and the message reads
   correctly either way because it names the term rather than the user's
   configuration.
2. **R15 fires only for a mount whose feeds actually rotate** relative to the
   sky: `alt-az`, `alt-az+nasmyth-l`, `alt-az+nasmyth-r`. `equatorial` is
   excluded. Section 20.7's own table gives `equatorial` the mount factor
   `eta = 0`, so `P` is *exactly* the identity for an all-equatorial array;
   demanding `jones.P` for it (R15) and then rejecting the configured `P` as an
   identity (R7) would leave such an array with no accepted configuration at
   all. An unspecified `mount_type` (`null` — every layout-file source produces
   it) is the `fixed` case, which is what preserves invariant I1.
3. **R7 for `P` is mount-aware.** `jones.P` resolves to exactly `I2` for every
   antenna, direction and time when no antenna's mount rotates, and that is the
   R7 condition verbatim ("makes it exactly the identity"). `enabled: false`
   reaches the same rejection by the same route, which is what keeps Section
   21's "there is no `enabled: false`" rule true for the one term that has an
   `enabled` key at all.

Together these make the contract a partition rather than a loop: a rotating
mount requires `P`, a non-rotating array must not configure it, and an
unmodelled mount is rejected in either case.

## 25. Provenance, fingerprint, and serialization

### 25.1 The scientific fingerprint

`jones_sha256` is computed over the canonical `term_snapshots` mapping plus the
resolved chain order plus the resolved mount types, exactly as
`receptor_sha256` is computed for Tier 5, and it enters `scientific_sha256`.
Consequences, all of which are required tests:

- Two runs differing only in a Jones parameter produce different
  `scientific_sha256`.
- Two runs with `jones:` absent produce the **same** `scientific_sha256` as the
  same configuration at `ac4fe41` — the Tier 7A characterization pins this by
  recording the digests before any change and re-asserting them at 7K.
- Nothing filesystem-path-derived enters the hash (the `RUN-005`/`RUN-006`
  lesson): the Jones snapshot is pure configuration.

### 25.2 Serialization

- **HDF5**: a new `jones/` group carrying `enabled_terms`, `chain_order`, the
  per-term snapshot as JSON attributes, and `jones_sha256`; schema version
  bumped, with the reader accepting a file that has no `jones/` group by
  treating it as "no terms enabled" — the same forward/backward posture Tier 6G
  used for the hybrid group.
- **Summary JSON**: a `jones` object with `enabled_terms`, `chain_order`, and
  `jones_sha256`, and each term's resolved parameters.
- **Measurement Set / UVFITS**: **no change**. A corrupted visibility is still a
  visibility; RadioSim does not write calibration tables and Tier 7 does not
  start. The summary and HDF5 carry the corruption record. This is stated
  explicitly so no slice invents a `CALDEVICE` or `BANDPASS` subtable.
- **`SimulationResult`**: gains a `jones` provenance block mirroring the summary.

### 25.3 Observability

The observability product (Tier 3) evaluates beams, not chains. It is
**unchanged** by Tier 7, and a test asserts that enabling any Jones term leaves
every observability output bit-identical — because an observability plot that
silently changed when a bandpass was configured would be a new class of the
same defect this tier exists to remove.

## 26. Error taxonomy

A new module `src/radiosim/core/jones_errors.py`, following the shape of
`core/receptor.py:64-90`:

```
JonesError(RuntimeError)
├── InvalidJonesConfigError(JonesError)          # malformed or physically invalid values
│   ├── IdentityJonesTermError                   # R7
│   └── UnsupportedMountTypeError                # R12, R15
├── JonesAssignmentError(JonesError)             # R4, R14: config names an unknown antenna/baseline
└── JonesEvaluationError(JonesError)             # a term produced a non-finite or wrong-shaped result
```

`JonesEvaluationError` is not decorative. Every term's batch evaluation is
shape- and finiteness-checked once per `(time, frequency)` step in debug-cheap
form (shape and `isfinite` on the first and last elements is not enough — the
check is a full `isfinite().all()` on the block, which is one reduction per
antenna per step and is negligible beside the contraction). A term that
produces `nan` from a legal configuration is a defect that must surface at the
term, not as a silent `nan` in the output cube.

### 26.1 Mandatory failure ordering

The order in which failures are raised is part of the contract, because a user
fixing a configuration must not be sent around a loop:

1. Pydantic strict parse of `jones:` (unknown keys, wrong types, bad `kind`);
2. `collect_config_issues` removed-field guidance (R1);
3. `resolve_jones_terms()` structural validation: `per_antenna` antenna
   existence and duplication (R4, R5, R6), baseline existence (R14);
4. physical-range validation (R8, R9, R10, R16, R17);
5. cross-object consistency: mount types (R12, R15), bandpass coverage (R11);
6. the identity check (R7), **last**, because it needs fully resolved values.

**Correction (7G implementation, 2026-08-01):** stage 5 previously also listed
"minimum elevation (R13)". It cannot: R13's condition is about directions, and
no direction exists at resolution time (Section 24's 7G correction gives the
full argument). R13 is raised by `T` and `Z` from `compute_jones_batch`, which
is after every stage here and after the first solver step — the one rejection in
this tier that a "reject before side effects" ordering cannot cover, because the
thing it inspects is produced by the work it would guard.

All of stages 3-6 run before any beam load, any sky load, any network access,
and any solver work — the Tier 1 "reject before side effects" property, extended
to the new section.

## 27. Scientific invariants — the test oracles

Each invariant is named, is asserted by at least one test, and is listed against
its owning slice in Section 30.

| # | Invariant | Owning slice |
|---|---|---|
| **I1** | **Absence identity.** With `jones:` absent, the `(T, B, F, 2, 2)` cube and the `scientific_sha256` are bit-identical to the 7A characterization pin for every shipped configuration. | 7B, 7C, 7D, re-asserted at 7K |
| **I2** | **Declared flags are true.** For every implemented term, over a parameter sweep: if `is_diagonal()` then the off-diagonals are exactly zero; if `is_scalar()` then the matrix equals its `[0,0]` element times `I2` exactly; if `is_unitary()` then `J J^H = I2` within the term's dtype tolerance. Conversely, a term declaring `False` must have at least one swept parameter where the property fails — so a *vacuous* `True` is impossible to reintroduce. | 7B (framework), each term slice (its own term) |
| **I3** | **DIE broadcast shape.** Every direction-independent term returns exactly `(1, 2, 2)` from `compute_jones_batch`; every DDE term returns `(n_dir, 2, 2)`. | 7B |
| **I4** | **Phase sign.** For every term producing a delay-like phase (`Kd`, `Rc`, `T`, `Z`), a positive delay or excess path produces `exp(-i·positive)`, matching the geometric phase's own sign at `visibility.py:697`. | each term slice |
| **I5** | **Chain order is observed.** Two deliberately non-commuting synthetic terms compose in the documented order — the Tier 5 test, extended to the full 12-term canonical order. | 7B |
| **I6** | **Circular-receptor `P` placement.** For a circular receptor with non-zero static rotation `chi` and non-zero field rotation `psi`, the composed `C P` equals `M(circular) R(chi + psi)` exactly, and the reversed composition `P C` does **not**. This test fails under the Tier 5 order and passes under the corrected one. | 7F |
| **I7** | **Effect changes visibility.** For every implemented term, a run with the term enabled at a physically meaningful value differs from a run with it absent by more than `1e-10` relative in at least one correlation of at least one baseline — `Fix.md` §16 rule 5, made mechanical. | each term slice |
| **I8** | **Faraday composition, no double count.** With source RM only, ionospheric RM only, and both, the polarization angle rotates by `RM_src λ²`, `RM_ion λ²`, and `(RM_src + RM_ion) λ²` respectively, to `1e-12`. | 7G |
| **I9** | **`P` is wide-field.** Over a direction batch spanning 20 degrees, `psi` varies by a measurable, predicted amount, and it converges on the single-direction value in the narrow-field limit. **Correction (7F implementation, 2026-08-01):** this row previously read "over a 0.01-degree batch it is constant to `1e-12`". That is unachievable and is not the physics — `dpsi/dtheta` is of order unity away from the poles, so a 0.01-degree batch spans of order `1e-4` rad of direction and therefore of order `1e-5` rad of `psi`. Asserting `1e-12` there would assert that `P` is *not* wide-field, contradicting the row's own first half. The slice asserts something strictly stronger instead: the spread is first order in the field width (halving the width halves it, to one part in a thousand), and it does reach `1e-12` once the batch is small enough for that scaling to take it there. | 7F |
| **I10** | **Opacity power/voltage factor.** With `T` opacity `tau_0` at zenith on a baseline of two identical antennas, the visibility amplitude is scaled by exactly `exp(-tau_0)`, confirming the `exp(-tau/2)` voltage convention. | 7G |
| **I11** | **`M` breaks closure; `G` does not.** On a three-antenna triangle, an enabled `G` with arbitrary per-antenna phases leaves the closure phase invariant to `1e-12`; an enabled `M` changes it by the predicted amount. | 7H |
| **I12** | **`Q` bounds and phase-centre unity.** `0 < Q <= 1` everywhere; `Q = 1` exactly at the phase centre; `Q` changes amplitude only, leaving every visibility phase unchanged to `1e-12`. **Correction (7H implementation, 2026-08-01):** `Q <= 1` always, and `0 < Q` while the smearing argument is below the first sinc zero — beyond it the exact top-hat average changes sign, and clamping would fabricate. `Q = 1` exactly at the phase centre is asserted of the **bandwidth** factor; the **time** factor is unity there only on a baseline with no East-West component, because RadioSim's phase centre is the fixed zenith and the sky moves through it (Section 20.11's correction). The amplitude-only clause is asserted where `Q > 0` and **per direction**: a visibility summed over several sources that decorrelate by different amounts does move in phase (by about `3e-7` rad on the shipped workload), which is the arithmetic of an average rather than a property of `Q`, so the clause is tested on a single-source cube. | 7H |
| **I13** | **Fingerprint sensitivity.** Changing any single Jones parameter changes `scientific_sha256`; changing no Jones parameter leaves it unchanged; `instrument_sha256` is unchanged by every Jones configuration. | 7D |
| **I14** | **Point/HEALPix agreement.** For a sky expressible both ways, the point and HEALPix paths agree within the Tier 6 tolerance with **every** implemented term enabled — the proof that the shared evaluator (Section 14) really is shared. | 7D, extended each term slice, re-asserted 7K |
| **I15** | **Strategy registry equals config surface.** The accepted values of `execution.simulator` equal the keys of the simulator registry, as a set. | 7C |
| **I16** | **One compiled kernel.** `src/` contains exactly one `backend.compile` call site, and the compiled kernel's signature is unchanged from `ac4fe41`. | 7B, re-asserted 7H and 7K |
| **I17** | **Precision is honored.** Under a non-default precision preset, every term's returned dtype equals the resolved per-term dtype; no term returns `complex128` when `complex64` was resolved. | 7B |
| **I18** | **Observability is inert.** Enabling any Jones term leaves every observability output bit-identical. | 7D |
| **I19** | **Pointing offset and Ruze.** A zero pointing offset is bit-identical to no offset; an offset `delta` moves the analytic beam's peak by exactly `delta`; the Ruze factor equals `exp(-(4 pi sigma/lambda)^2)` at three wavelengths. | 7I |
| **I20** | **No identity survives.** Every exported Jones class has `term_status == "implemented"`; no exported class's `compute_jones_batch` returns an identity for all inputs; a source scan finds no `TODO: implement properly` and no `Stub:` docstring anywhere in `src/radiosim`. | 7C, re-asserted 7K |

## 28. Backend parity matrix and tolerance

Unchanged from `Tier6HybridRuntimePlan.md` §13.4-13.5, applied per term:

| Backend | Tolerance vs NumPy | Scope |
|---|---|---|
| Dask | **bit-identical** | full `(T, B, F, 2, 2)` cube |
| JAX-CPU | `rtol=1e-12`, `atol=0` | full cube |

Per-term requirement: one parity case with **that term alone** enabled at a
large parameter value, and one parity case with **every** term enabled
simultaneously (run once, at 7K). A term whose parity fails is a defect in the
term, not a tolerance to widen.

Terms must not introduce a host-side branch on array values (Section 17.2). The
7B slice adds a parity harness helper so each later slice adds a parity case in
three lines rather than re-deriving the fixture.

## 29. Cross-implementation validation strategy

`Fix.md` §16 requires cross-checking *results*, not API shapes, against a
scientifically appropriate reference. Section 5.8 established that `pyuvsim`,
`matvis`, and RASCIL are **not** in any locked environment, while `astropy`,
`pyuvdata 3.2.1`, `healpy`, and `python-casacore` are.

### 29.1 The two tiers of evidence

**Tier-1 evidence — offline, in the standard gate, mandatory.** Comparisons
against an independent implementation that is already locked, or against a
closed-form published expression evaluated independently in the test.

| Term | Reference | What is compared |
|---|---|---|
| P | **astropy** frames (`AltAz`/`CIRS` + `EarthLocation` + sidereal time), an independent derivation of the same angle | `psi` over a dense `(H, dec, lat)` grid, to `1e-10` rad |
| C, H, P | **pyuvdata 3.2.1** feed-angle and polarization conventions | that the composed feed orientation matches the convention Tier 5 adopted verbatim |
| Z | published `k_TEC = 1.3445e9 Hz TECU^-1` and `40.308 TEC/nu^2` metres (TMS eq. 13.128), evaluated independently | the dispersive phase at three frequencies, to `1e-12` relative |
| T | Saastamoinen (1972) closed form and Niell (1996) coefficients, evaluated independently in the test | `ZHD` and `m(el)` at five elevations |
| Q | Bridle & Schwab (1999) sinc expressions, evaluated independently | decorrelation on the longest and shortest baseline |
| D, G, B, X, Kd, Rc | closed forms from Section 20, evaluated independently | the 2×2 matrix elementwise |
| M | closure-phase algebra | that closure is broken by the predicted amount |

"Evaluated independently in the test" means the test computes the reference
value from the published formula written out in the test body, **not** by
calling the production function — otherwise the test is a tautology. The plan
states this because it is the single most common way a cross-validation suite
becomes worthless.

**Tier-2 evidence — environment-dependent, recorded, never a gate.** A
comparison against `pyuvsim` for a small full-Stokes RIME case with `P` and `D`
enabled. `pyuvsim` is pure Python and pip-installable, but it pins `pyuvdata`,
and RadioSim pins `pyuvdata ==3.2.1` (`pixi.toml:44`), so resolvability is not
established at this gate. **Open question Q1** (Section 41) owns it.

If it resolves, it lands as an optional `crossval` pixi feature and a
`performance`-style marked test that is excluded from the default gate, with
the comparison committed as an evidence artifact under
`output/crossvalidation/`. If it does not resolve, the fallback is a **recorded
manual comparison**: the numbers, the reference version, the machine, and the
date, written into the slice's acceptance record — never a silent skip, and
never a claim of validation that no artifact supports.

### 29.2 What may and may not be claimed

- Permitted: "`P` agrees with an independent astropy-derived parallactic angle
  to `1e-10` rad over a `<...>` grid (test `<name>`)."
- Permitted: "`Z`'s dispersive phase matches TMS eq. 13.128 to `1e-12`."
- **Forbidden** without a committed artifact: "validated against pyuvsim",
  "matches CASA", "cross-checked against RASCIL". Section 4's truthfulness
  boundary applies to validation language exactly as it applies to performance
  language.

## 30. Exact test matrix

New test files, and what each owns. Existing files are extended, not replaced,
except where a removed API makes a test moot.

| File | Owns |
|---|---|
| `tests/characterization/test_tier7_current_behavior.py` | 7A pins: the exact `(T,B,F,2,2)` digests and `scientific_sha256` for the four shipped configs; the 43-name `__all__`; the identity return of all 37 stubs; the absence of a `jones:` section; the `calculation_type` no-op |
| `tests/unit/test_jones/test_direction_batch.py` | `DirectionBatch` construction, immutability, frame consistency (`(alt,az)` vs `(ra,dec,H)` round trip) |
| `tests/unit/test_jones/test_term_contract.py` | I2, I3, I17; the flag-verification sweep applied to every registered term; `add_term` rejecting `JonesBaselineTerm` (D7) |
| `tests/unit/test_jones/test_chain_order.py` (extend) | I5 over the 12-term canonical order; I6 |
| `tests/unit/test_jones/test_geometric_phase.py` | the extracted `geometric_phase()` function; the `w(n-1)` sign pin (Workstream C deliverable); equality with both former inline copies |
| `tests/unit/test_jones/test_gain.py` … `test_smearing.py` | one file per implemented term: closed-form comparison, I2 for that term, I4, I7, the term's own invariants from Section 20 |
| `tests/unit/test_jones/test_backend_parity.py` | Section 28's per-term parity cases |
| `tests/unit/test_core/test_jones_resolution.py` | `resolve_jones_terms` precedence, the failure ordering of Section 26.1, every rejection R2-R16 by exact message |
| `tests/unit/test_io/test_jones_config.py` | schema parse, strictness, discriminated unions, R1, R3 |
| `tests/unit/test_core/test_jones_provenance.py` | I13; HDF5 and summary round trip; MS/UVFITS unchanged |
| `tests/unit/test_core/test_beam_pointing.py` | I19 |
| `tests/unit/test_tier7_jones_acceptance.py` | I1, I14, I16, I18, I20; the whole-tier gate assertions; the residual scan for `TODO: implement properly`, `Stub:`, and every deleted class name |
| `tests/integration/test_jones_end_to_end.py` | one `Simulator.setup().run().save()` per implemented term through HDF5, summary, MS, and UVFITS |

Every rejection in Section 24 has a test asserting the **exact** message string,
following the Tier 5 precedent that made every receptor rejection reproducible
by hand.

## 31. Tests-first implementation strategy

Each term slice follows the same five steps, in order, and the slice's
acceptance record must show them in that order:

1. **Write the invariant tests first, from Section 20's mathematics**, with the
   reference values written out in the test body (Section 29.1). They fail.
2. **Write the rejection tests**, from Section 24's verbatim messages. They fail.
3. Implement the term.
4. **Add the effect-changes-visibility test (I7) and the parity case
   (Section 28)** — these two cannot be written before the term exists because
   they need the config surface, so they are step 4, not step 1, and the slice
   record says so rather than pretending otherwise.
5. Update the term's documentation and its `term_status`, and only its own stub
   warning (`Fix.md` §16 rule 6).

## 32. Common verification gate

Run at the end of every slice; a slice is not proposable until all of it passes:

```
pixi run test -- -m "not slow"          # full non-slow suite, 0 xfail, 0 XPASS
pixi run test                            # full suite
pixi run lint
pixi run check-format
git diff --check
python -m sphinx -b html docs docs/_build/html   # warnings not increased
```

`pixi run typecheck` is run **only** when a slice changes a type-bearing public
signature — which for Tier 7 means 7B, 7C, 7D, and 7K — consistent with the
project convention that it is not part of the standard loop.

`pixi run bench` is **not** run as a gate and produces no Tier 7 claim.

## 33. Tier 7 implementation slices

Eleven slices. Each is independently acceptable, each has an exact writable file
list (Section 34), and no slice may begin before its predecessor is accepted.

### 33.1 Slice ordering rationale

The order is forced by three dependencies and one honesty rule.

- **Dependency 1: the evaluation contract precedes every term.** No term can be
  written against a contract that is about to change, so 7B (the batched
  contract and the shared evaluator) comes before any physics.
- **Dependency 2: truth precedes capability.** 7C deletes the stubs and removes
  `calculation_type` *before* any term is implemented. This is deliberate: it
  means that from 7C onward there is **no moment in the tier's history at which
  a public identity stub exists**. The alternative ordering — implement terms
  first, delete stubs last — would leave the repository in the `SCI-001` state
  through nine slices.
- **Dependency 3: the schema and provenance precede the terms that use them**,
  so 7D carries the schema machinery together with the two simplest terms
  (`G`, `B`), which act as the machinery's first real customers rather than as
  a hypothetical.
- **Honesty rule: `P` before nothing else needs it, but after the schema.**
  `P` is placed at 7F rather than earlier because it is the slice that changes
  an *accepted* Tier 5 contract (the chain order and the mount-type rejection),
  and that change is easier to review when the schema, the provenance, and four
  simpler terms are already in place and passing.

Within the terms, DIE before DDE (`G`, `B`, then `D`, `X`, `Kd`, `Rc`, then
`P`, then `Z`, `T`), because the DIE terms exercise the `(1, 2, 2)` broadcast
path and the DDE terms exercise the direction batch, and a bug in the former
is much easier to localize before the latter exists.

### 33.2 The slices

**7A — characterization, dependency contract, and baseline pins.**
Record what `ac4fe41` actually does, before anything changes: the cube digests
and `scientific_sha256` for all four shipped configs; the 43-name `__all__`; the
identity return of every one of the 37 stubs, asserted individually so that
their later deletion is a visible, deliberate flip; the fact that
`calculation_type` reaches nothing; the fact that `jones_config` is always
`None`. Resolve **Q1** (Section 41) by recording whether `pyuvsim` resolves
against `pyuvdata ==3.2.1` on the three locked platforms. Resolve **Q2** by
recording the direction-batch memory footprint for the largest shipped HEALPix
configuration. No production change.

**Correction (7A independent acceptance, 2026-08-01):** "the cube digests and
`scientific_sha256` for all four shipped configs" is only partly achievable
from this environment, and the implementer's departure from the literal
instruction is ratified rather than corrected by code. `configs/config.yaml`
and `configs/receptor_circular_example.yaml` get the absolute, per-environment
digests this sentence asks for, by delegating to Tier 6's own
`_SHIPPED_CONFIG_FINGERPRINTS`/`_SHIPPED_CONFIG_CUBE_DIGESTS` tables, which
already carry verified values for all six `(platform, python)` environments.
`configs/realistic_foreground_example.yaml` cannot be hermetically pinned at
all (network-dependent; Tier 6A reached the same conclusion) and is recorded
only by source facts plus the Q2 measurement. `configs/hybrid_sky_example.yaml`
has **no** absolute digest in any Tier 6 or Tier 7 table, and this repository
has no `x86_64` host from which to harvest one: inventing a value measured
only on `osx-arm64` and asserting it as ground truth for `linux-64`/`osx-64`
CI is exactly the mistake Tier 6J's whole-tier rejection (`Fix.md`,
2026-07-31) diagnosed as an architecture-level floating-point-non-associativity
trap, not a hypothetical risk. 7A's substitute — an environment-independent
bit-level invariant (the hybrid cube is exactly the backend-domain sum of its
point-only and HEALPix-only components) — is the correct lesson from that
rejection, not a shortfall against this sentence: it holds on every runner
without requiring `x86_64` access, and 7B's "bit-identical to 7A's pins for
every shipped configuration" claim is satisfied for the hybrid config by this
invariant continuing to hold, not by an absolute digest. No CI harvest of
`linux-64`/`osx-64` hybrid digests is required as a 7B (or later) obligation;
one may still be added for redundancy at any later slice's discretion, but it
would add no coverage the additivity invariant does not already provide, and
asserting it without having verified it on those architectures would repeat
Tier 6J's error. No decision changes.

**7B — the evaluation contract and the shared evaluator.**
`DirectionBatch`; `compute_jones_batch` replacing the scalar contract;
`JonesChain.compute_antenna_jones_batch` with a precision-resolved seed (D8);
`add_term` rejecting `JonesBaselineTerm` (D7); `JonesBaselineTerm`'s batched
contract; `geometric_phase()` extracted and both inline copies replaced (D6);
`GeometricPhaseJones` deleted; `evaluate_antenna_jones` used by **both**
solvers, replacing the HEALPix path's private receptor handling (D4); `C`/`H`
dtype correctness (D9); the flag-verification harness (D10, I2); the parity
harness. **Bit-identical to 7A's pins** for every shipped configuration — this
slice adds no physics and its acceptance is exactly that identity plus I2, I3,
I5, I16, I17.

**Correction (7B implementation, 2026-08-01)** — four bounded departures from
the sentence above, each forced by a fact the design gate did not have:

1. **`compute_jones_batch` is concrete-and-raising at 7B, not `@abstractmethod`**
   (and likewise `JonesBaselineTerm.compute_baseline_factor`).  Section 13.2
   states that the new contract "affects only four real implementations", which
   is true of the *implementations* but not of the *declaration*: 7B's writable
   list contains none of the 35 identity-stub modules, so an abstract
   declaration would make every stub impossible to instantiate.  That would
   break the 7A pins Tier 7C and Tier 7D-7H own, and would leave the public
   surface worse than the stub state 7C is about to remove -- a class that
   cannot be constructed at all.  The method therefore raises
   `NotImplementedError` naming the class, and becomes `@abstractmethod` in the
   slice that removes the last non-implementing subclass.
2. **`GeometricPhaseJones` leaves `__all__` at 7B, so three 7C-owned pins move
   with it.**  Section 23 groups the K class with the 26 names 7C removes, while
   Section 33.2 assigns its deletion to 7B.  7B is right -- K is per-baseline and
   cannot be a chain term -- so the name-count, lazy-binding and real-physics
   pins are flipped here for that one name, and the remaining 25 removals and
   the `CrosshandPhaseJones` rename stay with 7C.
3. **Enabling an optional stub term now raises instead of returning identity.**
   This is the one behaviour 7B changes.  `jones_config` is hard-coded to `None`
   at the single production call site (D3), so no shipped configuration, CLI
   invocation or `Simulator` run can reach it; only a direct solver call can.
   On that surface, silence became a typed failure, which is the direction
   `Fix.md` Section 16 asks for.  The absent and empty configurations stay
   bit-identical, which is what governs every real run.
4. **Two HEALPix-only numerical consequences of D4, both at the floating-point
   noise floor and both making the diffuse path agree with the point path.**
   Measured against `e1ae149` on `osx-arm64`/py311: (a) with a circular receptor
   reported in a *linear* basis -- the only case where `C` and `H` are both
   non-identity -- the composition changes from `(H @ C) @ E`, with `H @ C`
   formed once in host `float64`, to the canonical `H @ (C @ E)` the point path
   has always used; maximum relative deviation `3.2e-16`, one ULP of
   `complex128`.  (b) Under a preset whose Jones precision is `float32` but whose
   accumulation precision is `float64` -- `fast` is the shipped example -- the
   diffuse path's per-antenna Jones is now `complex128`, as the point path's
   always was, rather than inheriting the beam's `complex64`; maximum relative
   deviation `8.8e-8` on the shipped hybrid configuration, one ULP of
   `complex64`.  Every shipped configuration at the default precision, every
   point-path workload at every preset, and every 6A/7A environment-keyed pin
   are bit-identical.

**7C — public-surface truth.**
Delete the 26 stub classes and their five now-empty modules; rename
`CrosshandPhaseJones` to `CrosshandJones`; remove the `jones_config` parameter
from every solver and simulator signature; remove
`VisibilityConfig.calculation_type` and its unsupported-issue branch, add the
removed-field guidance (R1), and update the four shipped configs and the
documentation; add I15's registry/config-surface equality test; add I20's
residual scan; update `docs/api/jones.rst`, `docs/user_guide/jones_matrices.rst`,
`docs/user_guide/configuration.rst`, `docs/migration_guide.md`, and the
changelog. **Still bit-identical**: nothing removed here was ever reachable.
After 7C, `Fix.md` §16's "no public term silently multiplies by identity" is
already true.

**Correction (7C implementation, 2026-08-01)** — six bounded departures from the
sentence above and from the sections it refers to, each forced by a fact the
design gate did not have. None changes a decision; each makes a decision
executable.

1. **Three now-empty modules, not five.** The 26 deletions empty exactly
   `faraday.py`, `wterm.py` and `element_beam.py`. The other nine former stub
   modules each keep one or two terms that Tier 7D-7H implement, so they are
   rewritten rather than deleted. §34's 7C writable list is right — it marks
   exactly those three "(delete)" — and this sentence's "five" is an arithmetic
   slip. Relatedly, §23's list under the heading "removed (26 names)" contains
   **27** names, because it includes `GeometricPhaseJones`, which §9.1 disposes
   of as "→ function" rather than as a deletion. The load-bearing count is
   §9.1's: 26 classes deleted, 1 converted to a function (at 7B), 11 kept as
   planned, 2 already implemented — 40 concrete classes accounted for, and 19
   surviving `__all__` entries.

2. **`term_status` is tri-state in effect: `"planned"` is the base default, not
   `"implemented"`.** §23 records the property as `-> "implemented"` and I20 as
   "every exported Jones class has `term_status == "implemented"`". Both are
   true at **7K** — §37 criterion 2 says so explicitly, and §31 step 5 has each
   term slice "update ... its `term_status`", which only makes sense if it
   starts as something else. Asserting `"implemented"` at 7C would require
   claiming it for eleven terms whose `compute_jones_batch` raises: the exact
   vacuous-claim failure mode invariant I2 exists to prevent, one level up. The
   base class therefore returns `"planned"`, `ReceptorConfigJones`,
   `BasisTransformJones` and the private `_ResolvedBeamJones` override it, and
   I20's 7C assertion is the **correspondence**, checked both ways: an
   `"implemented"` term is not an identity for all inputs, and a `"planned"`
   term is not evaluable at all. The rest of I20 — the residual scan for
   `TODO: implement properly`, `Stub:` and the unconditional
   `xp.eye(2, dtype=np.complex128)` — is asserted at 7C in full.

3. **The ABC abstract-flip is not 7C's.** The 7B correction to §33.2 says
   `compute_jones_batch` "becomes `@abstractmethod` in the slice that removes
   the last non-implementing subclass". 7C does not: nine planned `JonesTerm`
   subclasses survive it, and an abstract declaration would make every one of
   them impossible to construct. The flip belongs to **7G**, the slice that
   implements the last of them (`Z` and `T`), and
   `JonesBaselineTerm.compute_baseline_factor`'s flip stays with **7H**, as the
   7B correction already said. 7C removed the classes RadioSim will never
   implement; it did not remove the ones it has not implemented yet.

4. **A planned term declares no capability flag and no constructor.** This goes
   beyond the literal removal ledger and is the direct application of §9.2's own
   argument. A `is_diagonal() -> True` on a term whose matrix cannot be computed
   is a claim about numbers with no numbers, which I2's sweep cannot verify and
   which defect D10 names; and a constructor that accepts `tec=`, `d_terms=` or
   `gain_sigma=` and stores them unread is defect D2, the harm `SCI-001` calls
   "materially worse than returns identity". Both are therefore stripped from
   all eleven planned terms, and each term slice reintroduces its own flags and
   its own constructor together with the physics and the I2 case that verify
   them (§31 steps 3-5). D2 and the vacuous half of D10 are closed at 7C rather
   than at 7D-7H.

5. **`_reject_parallactic_rotation` is kept, callerless.** 7C removes the
   `jones_config` parameter that was the guard's only trigger, so the
   combination it rejects becomes inexpressible through any entry point. Its
   *deletion* is assigned to 7F, together with the R15 replacement, so 7C leaves
   the function and its exact message in place and stops calling it. Tier 5's
   real protection — `resolve_receptors` rejecting every non-`fixed` mount type
   — is untouched. Two tests that reached the guard through `jones_config` are
   re-aimed at the guard directly and at the contract that survives.

6. **`CLAUDE.md` is deliberately left stale.** It still says "46 exported
   classes" and still lists `ElevationGainJones` and `TroposphericOpacityJones`
   in its term table. §34 gives `CLAUDE.md` to **7J** (D0, D21), and 7C does not
   touch it. The 7A pin `test_claude_md_claims_forty_six_exported_jones_classes`
   is updated to record the gap explicitly rather than quietly close it, so the
   staleness is visible to a reviewer rather than latent.

**7D — the `jones:` schema, resolution, provenance, and the first two terms
(`G`, `B`).**
`io/jones_config.py`; `core/jones_terms.py` with `resolve_jones_terms` and the
frozen resolved model; `core/jones_errors.py`; the precision extension (D15);
`jones_sha256` into `scientific_sha256`; HDF5 group, schema bump, and reader;
summary JSON; `SimulationResult` block; `Simulator.setup()` wiring replacing
`hybrid.py:292`'s hard-coded `None` (D3). Then `G` and `B` end to end.
Invariants: I1, I7, I13, I14, I18, plus every rejection that does not need a
later term.

**Correction (7D implementation, 2026-08-01)** — eleven bounded departures
from the sentences above and from the sections they refer to. None changes a
decision; each makes a decision executable, and the first is the only one that
changes what a reviewer should expect to find in a named field.

1. **`chain_terms` carries the configured terms only; `H`, `C` and `E` are
   recorded, not contained.** §22 rule 3 says the three always-on terms "are
   always present", and §22's sketch annotates `chain_terms` "in canonical
   order, H first". Both cannot be satisfied by one tuple built at setup: `E`
   is the solver's `_ResolvedBeamJones` adapter, which closes over the
   directions, frequency and time of the `(time, frequency)` step it is
   evaluated at and therefore cannot exist before the time loop, and `H` and
   `C` come from the resolved *receptor* set rather than from `jones:`.
   `ResolvedJonesTerms.chain_terms` therefore holds exactly what the `jones:`
   section configured, in canonical order, and `provenance.enabled_terms` and
   `provenance.chain_order` hold the **full** composed order including all
   three — which is what a reader of the record needs, and what §22's own
   example (`("H","G","C","E","P", ...)`) shows. `_build_jones_chain` walks
   `CANONICAL_CHAIN_ORDER` once and takes each letter from whichever of the two
   sources owns it, so there is now exactly **one** `chain.add_term(` statement
   in the package and it cannot treat a configured term differently from an
   always-on one.

2. **§34's 7D list names two files that do not exist in that form.** "the HDF5
   group, schema bump, and reader" live in `src/radiosim/io/hdf5.py`, which is
   both writer and reader; `src/radiosim/io/writers.py` does not exist, and
   `src/radiosim/io/readers.py` is a 62-line unrelated debug helper that is no
   part of the result path. "the summary-JSON writer" is
   `src/radiosim/io/summary_json.py`. This is the same class of slip as 7C's
   "five now-empty modules".

3. **Four files outside §34's list are forced by decisions §33.2 already
   took.** `src/radiosim/core/runtime_config.py`, because a new top-level
   configuration section has to reach the runtime through
   `ResolvedSimulationConfig` and `Simulator` holds nothing else;
   `src/radiosim/io/result_errors.py`, because bumping the HDF5 schema means
   the superseded-version guidance now names a different version;
   `docs/api/io.rst` and `docs/migration_guide.md`, because the schema bump is
   documented there and nowhere else.

4. **Ten test files outside §34's list are forced, all of them pins.** Adding a
   top-level section moves four exact-shape pins
   (`tests/unit/test_io/test_config.py`,
   `tests/unit/test_io/test_instrument_config.py`,
   `tests/unit/test_io/test_receptor_config.py`,
   `tests/unit/test_simulator/test_instrument_integration.py`); the schema bump
   moves five (`tests/unit/test_io/test_hdf5_result.py`,
   `tests/unit/test_io/test_result_summary.py`,
   `tests/unit/test_tier4_result_output_acceptance.py`,
   `tests/unit/test_tier1h_documentation.py`,
   `tests/integration/test_hybrid_end_to_end.py`); and the one new integration
   file §30 asks for moves the Tier 6I directory pin in
   `tests/characterization/test_tier6_current_behavior.py`.
   `tests/characterization/test_tier7_current_behavior.py` is also outside the
   list and is where the nine `OWNED BY: Tier 7D` pins are flipped, which is
   the whole point of their being marked.

5. **The HDF5 `jones/` group is optional, and the schema goes 3.0.0 → 4.0.0.**
   §25.2 asks for a bump *and* for a reader that accepts a file with no
   `jones/` group. Those are only consistent if the group is written solely
   when a term was enabled, which is what 7D does: it is the first optional
   group in the format, `_inspect_tree` enforces all-or-nothing so a fragment
   is still an allowlist mismatch, and a run with no `jones:` section produces
   the file it always produced apart from the version string.

6. **An empty Jones snapshot is hashed as nothing at all, not as an empty
   object.** §25.1 requires a `jones:`-absent run to keep the digest it had at
   `ac4fe41`. `_scientific_hash` therefore *skips* the `jones` tag entirely
   when the snapshot is empty. Hashing an empty placeholder would have been
   one line simpler and would have invalidated every environment-keyed cube and
   fingerprint pin in the repository for no scientific reason.

7. **`jones_terms` is a defaulted parameter, not a required one.** §23's
   signature table shows it replacing `jones_config` positionally. It is
   declared `jones_terms: ResolvedJonesTerms = EMPTY_JONES_TERMS` on every
   solver, simulator and `solve_sky` signature, mirroring the
   `solver_execution=SERIAL_SOLVER_EXECUTION` precedent beside it, so a direct
   solver call with no Jones section is exactly the historical forward model
   and the seventy-odd existing direct solver calls in the suite are unchanged.

8. **§22's `FrozenDict` is `MappingProxyType`.** There is no `FrozenDict` type
   in this repository; `ResolvedReceptorSet.receptor_by_antenna` uses
   `MappingProxyType`, and `JonesProvenance` follows it.

9. **`ResolvedJonesDtypes.by_term` is a record, not a dispatch table.** Tier 7B's
   accepted contract composes the whole chain in the accumulation dtype, so
   resolving fifteen per-term precisions and then handing every term the same
   one is what actually happens. Recording the resolution is what makes a later
   decision to dispatch per term a visible change; claiming the dispatch now
   would be the vacuous kind of claim I2 exists to prevent. D15 is closed as
   "no term is without a declared precision", which is the defect as written.

10. **The `G` elevation curve is well defined and degenerate.** §20.1 evaluates
    it at the elevation of the pointing centre. RadioSim's one phase convention
    is zenith drift (`PhaseCenter.altitude_rad == pi/2` exactly), so the curve
    evaluates to a single constant for the whole run: a real, non-identity gain
    that does not vary. It is implemented as specified rather than deferred,
    and the degeneracy is stated in the term's own docstring, in
    `is_time_dependent`, in its test, and in the user guide — because an
    elevation *curve* that never moves reads as working when it is merely well
    defined. §21.2's sinusoidal time model is likewise implemented as specified;
    its unnamed fields are `depth`, `period_hours` and `phase_rad`, keeping the
    per-hour convention §21.2's own `rate_per_hour` establishes.

11. **§31's step order was not followed.** The invariant and rejection tests
    (steps 1-2) were written *after* the implementation (step 3), not before
    it. This is recorded rather than presented otherwise. The tests are
    nonetheless written against §20's published closed forms with the reference
    values in the test bodies (§29.1), and one of them found a real defect —
    `BandpassJones.is_scalar` compared a `(rows, 2, n)` table with a
    `(1, 1, n)` slice using `np.array_equal`, which does not broadcast, so a
    genuinely scalar bandpass reported `False`. A reviewer weighing that
    evidence should weigh the ordering with it.

**7E — `D`, `X`, `Kd`, `Rc` (Workstream A remainder).**
The four remaining direction-independent calibration terms. Includes the
mandated rewrite of `docs/user_guide/jones_matrices.rst:137-146` (the basis
conversion is exact only while `D` and `G` are off), which is the documentation
obligation Tier 5 explicitly left to this tier.

**7F — `P` (Workstream B, and the chain-order correction).**
The parallactic-angle term, direction-batched, with the five mount types; the
corrected canonical order placing `P` sky-side of `C` (D12, I6); removal of
`core/receptor.py:411-418`'s blanket mount rejection in favour of R15;
removal of `_reject_parallactic_rotation` (D17); `docs/user_guide/jones_matrices.rst:148-158`
rewritten. Invariants I6, I9, plus the astropy cross-check (Section 29.1). This
is the largest single physics slice and the only one that supersedes an accepted
Tier 5 decision.

**7G — `Z` and `T` (Workstream B remainder).**
Ionosphere (dispersive phase, thin-shell slant mapping, ionospheric Faraday) and
troposphere (Saastamoinen/explicit zenith delay, simple/Niell mapping,
opacity). Invariants I4, I8 (the D18 double-count guard), I10.

**7H — `M` and `Q` (Workstream D).**
The baseline Hadamard path: `Q` folded into the kernel's existing `envelope`,
`M` applied to the kernel's `(B, 2, 2)` output, **with the kernel signature
unchanged** (I16 re-asserted). Invariants I11, I12. This slice is the one that
"enforces the distinction between Jones matrix-chain terms and
baseline-dependent Hadamard terms" that `Fix.md` §16 Workstream D asks for, and
I11 is the proof.

**7I — beam physics (`SCI-003`).**
Per-antenna deterministic pointing offsets and the Ruze surface-efficiency
factor inside `BeamSystem`; `beam/TODO.md` rewritten as
`docs/development/beam_physics_scope.md` with a disposition and a citation for
every item (D20). Invariant I19.

**7J — cross-validation evidence and documentation truth.**
The Tier-2 cross-validation artifact or its recorded fallback (Section 29.1);
the full documentation pass — `docs/api/jones.rst` rebuilt around 16 names,
`docs/user_guide/jones_matrices.rst` rewritten around implemented physics, a new
`docs/user_guide/jones_terms.rst` documenting every term's mathematics, units,
citation, and configuration; the `CLAUDE.md` Implementation Status rewrite
(D0, D21); the changelog and migration guide completed.

**7K — independent whole-tier acceptance.**
No production change. Re-assert I1, I14, I16, I18, I20 and the whole-tier
criteria of Section 37; produce the Section 38 evidence; flip `SCI-001`,
`SCI-002`, `SCI-003` to `DONE` with their closure text; file `SCI-004` (m-mode)
and `SCI-005` (advanced beam physics); append the acceptance record to `Fix.md`.

## 34. Exact writable file list for every slice

No slice may touch a file outside its list. A slice that discovers it needs one
stops and requests a bounded plan correction, exactly as Tier 5 and Tier 6
required.

### 7A
- `tests/characterization/test_tier7_current_behavior.py` (new)
- `Fix.md` (acceptance record only)

### 7B
- `src/radiosim/core/jones/directions.py` (new)
- `src/radiosim/core/jones/evaluate.py` (new)
- `src/radiosim/core/jones/base.py`
- `src/radiosim/core/jones/chain.py`
- `src/radiosim/core/jones/geometric.py`
- `src/radiosim/core/jones/baseline_errors.py`
- `src/radiosim/core/jones/receptor.py`
- `src/radiosim/core/jones/__init__.py`
- `src/radiosim/core/visibility.py`
- `src/radiosim/core/visibility_healpix.py`
- `tests/unit/test_jones/test_direction_batch.py` (new)
- `tests/unit/test_jones/test_term_contract.py` (new)
- `tests/unit/test_jones/test_geometric_phase.py` (new)
- `tests/unit/test_jones/test_backend_parity.py` (new)
- `tests/unit/test_jones/test_chain_order.py`
- `tests/unit/test_jones/test_backend_jones.py`
- `tests/unit/test_jones/test_receptor.py`
- `tests/unit/test_jones/test_basis_transform.py`
- `tests/characterization/test_tier7_current_behavior.py` (pin flips only)
- `tests/characterization/test_tier5_current_behavior.py` (pin flips only)
- `tests/characterization/test_tier6_current_behavior.py` (pin flips only)
- `tests/unit/test_core/test_visibility_backend.py`
- `Fix.md`

**Correction (7B implementation, 2026-08-01):** the last three entries are added
by this correction, because 7B could not be executed without them and Section 34
requires a bounded plan correction rather than a silent overreach.  Each is
forced by a pin that names the exact mechanism 7B replaces, and in each case the
*property* the pin was written to protect is preserved or strengthened:

- `test_tier5_current_behavior.py` evaluates `C` and `H` through
  `compute_jones` and the chain through `compute_antenna_jones`, both of which
  this slice removes, and pins the HEALPix path as having no chain, which D4
  exists to change.  The Tier 5 properties -- the exact `S` matrix, the exact
  identity for the default linear array, `terms[0] @ ... @ terms[-1]`, and
  exactly one chain implementation -- are all re-asserted through the new
  contract.
- `test_tier6_current_behavior.py` anchors Tier 6D's "the constant `H_p @ C_p`
  is built once, above the time loop" on the literal
  `receptor_transforms = _receptor_transforms(`.  7B replaces that constant
  matrix product with the two run-constant chain terms that produce it, hoisted
  to the same place, so the anchor moves to `_resolved_receptor_terms(` and the
  property is preserved.  The same test's `for freq_idx, freq in enumerate(...)`
  assertion cannot survive -- the batched contract passes a frequency index to
  every term -- and is replaced by the assertion that actually carries Tier 6D's
  meaning: no per-cell output write survives.
- `test_visibility_backend.py` exercises the beam adapter's removed
  `compute_jones_all_sources` and its `antenna_number` cross-check.  That check
  is not weakened but made unnecessary: the shared evaluator keys everything by
  instrument row, so a row/number disagreement is no longer expressible.  The
  test is re-aimed at the invariants that remain checkable.

### 7C
- `src/radiosim/core/jones/gain.py`, `bandpass.py`,
  `polarization_leakage.py`, `parallactic.py`, `ionosphere.py`,
  `troposphere.py`, `delay.py`, `crosshand.py`, `baseline_errors.py`
- `src/radiosim/core/jones/faraday.py` (delete), `wterm.py` (delete),
  `element_beam.py` (delete), `geometric.py`
- `src/radiosim/core/jones/__init__.py`
- `src/radiosim/core/visibility.py`, `visibility_healpix.py`,
  `src/radiosim/core/hybrid.py`
- `src/radiosim/simulator/base.py`, `src/radiosim/simulator/rime.py`,
  `src/radiosim/simulator/__init__.py`
- `src/radiosim/io/config.py`
- `configs/config.yaml`, `configs/receptor_circular_example.yaml`,
  `configs/hybrid_sky_example.yaml`,
  `configs/realistic_foreground_example.yaml`
- `docs/api/jones.rst`, `docs/user_guide/jones_matrices.rst`,
  `docs/user_guide/configuration.rst`, `docs/migration_guide.md`,
  `docs/changelog.rst`
- `tests/unit/test_jones/*`, `tests/unit/test_io/test_config.py`,
  `tests/unit/test_io/test_config_resolution.py`,
  `tests/unit/test_simulator/test_api.py`,
  `tests/unit/test_core/test_hybrid_visibility.py`,
  `tests/unit/test_core/test_runtime_config.py`,
  `tests/fixtures/configs.py`,
  `tests/characterization/test_tier6_current_behavior.py` (config-dict updates only),
  `tests/characterization/test_tier7_current_behavior.py`,
  `tests/unit/test_tier1h_documentation.py`
- `tests/unit/test_tier7_jones_acceptance.py` (new)
- `Fix.md`

**Correction (7C implementation, 2026-08-01):** four entries are added by this
correction, because 7C could not be executed without them and Section 34
requires a bounded plan correction rather than a silent overreach. Each is
forced by a file that names the exact mechanism 7C removes, and in each case the
property the file was written to protect is preserved or strengthened:

- `src/radiosim/core/jones/receptor.py` — the `term_status` property must be
  declared `"implemented"` on `C` and `H`, and there is nowhere else to declare
  it. Invariant I20 is unassertable otherwise: with the base default
  `"planned"` (correction 2 above), a reviewer reading the two implemented terms
  would be told they are not. Nothing else in the file changes except two
  reStructuredText title underlines lengthened by one character each, which
  removes four Sphinx warnings that appear the moment 7C adds the module to
  `docs/api/jones.rst`.
- `tests/characterization/test_tier5_current_behavior.py` — two Tier 5 pins call
  `_build_jones_chain` with the `jones_config` dictionary 7C removes, one of
  them enabling all six optional terms to assert the full nine-term order. The
  Tier 5 property — that the solver adds terms correlator-side first in the
  Section 19.1 order — is re-asserted against the solver's own documented
  factorization plus the ordered positions of the three terms that exist, so
  Tier 7F's reordering of `P` is still a visible flip here.
- `tests/unit/test_core/test_receptor_solver.py` — its Tier 5 parallactic
  rejection test reaches `_reject_parallactic_rotation` through
  `jones_config={"P": {"enabled": True}}`, the trigger 7C removes. It is
  re-aimed at the guard directly (the exact message is still asserted, and is
  also pinned in the 7A characterization) plus the contract that survives: a
  rotated receptor now reaches the solver and is carried.
- `tests/unit/test_core/test_beam_solver_integration.py` — its
  `test_beam_dictionary_in_jones_config_is_rejected` asserts one of the three
  ad-hoc checks that stood in for a schema. The property it protected — that
  `beam_system` is the solver's only beam surface — becomes structural rather
  than guarded, and that is what it now asserts.

### 7D
- `src/radiosim/io/jones_config.py` (new)
- `src/radiosim/core/jones_terms.py` (new)
- `src/radiosim/core/jones_errors.py` (new)
- `src/radiosim/core/jones/gain.py`, `bandpass.py`, `__init__.py`
- `src/radiosim/core/precision.py`, `src/radiosim/io/config.py`,
  `src/radiosim/io/config_resolution.py`
- `src/radiosim/api/simulator.py`, `src/radiosim/core/hybrid.py`,
  `src/radiosim/core/visibility.py`, `src/radiosim/core/visibility_healpix.py`,
  `src/radiosim/simulator/base.py`, `src/radiosim/simulator/rime.py`
- `src/radiosim/core/result.py`, `src/radiosim/io/writers.py`,
  `src/radiosim/io/readers.py`, and the summary-JSON writer
- `src/radiosim/core/__init__.py`, `src/radiosim/io/__init__.py`
- `tests/unit/test_io/test_jones_config.py` (new),
  `tests/unit/test_core/test_jones_resolution.py` (new),
  `tests/unit/test_core/test_jones_provenance.py` (new),
  `tests/unit/test_jones/test_gain.py` (new),
  `tests/unit/test_jones/test_bandpass.py` (new),
  `tests/integration/test_jones_end_to_end.py` (new)
- `tests/unit/test_jones/test_backend_parity.py`,
  `tests/unit/test_tier7_jones_acceptance.py`,
  `tests/unit/test_core/test_precision.py`, `tests/unit/test_core/test_result.py`
- `docs/user_guide/configuration.rst`, `docs/user_guide/jones_terms.rst` (new)
- `Fix.md`

### 7E
- `src/radiosim/core/jones/polarization_leakage.py`, `crosshand.py`, `delay.py`
- `src/radiosim/core/jones/__init__.py`, `src/radiosim/core/jones_terms.py`,
  `src/radiosim/io/jones_config.py`
- `tests/unit/test_jones/test_leakage.py` (new),
  `test_crosshand.py` (new), `test_delay.py` (new),
  `test_cable_reflection.py` (new)
- `tests/unit/test_jones/test_backend_parity.py`,
  `tests/unit/test_core/test_jones_resolution.py`,
  `tests/integration/test_jones_end_to_end.py`
- `docs/user_guide/jones_matrices.rst`, `docs/user_guide/jones_terms.rst`
- `Fix.md`

**Correction (7E implementation, 2026-08-01) — six forced additions.** The
list above omits six files that adding a term to the schema, or carrying out
this slice's own mandated documentation rewrite, *necessarily* touches. The
omission is a defect in the list rather than a boundary the slice should
respect. Each addition is bounded and named:

- `src/radiosim/io/config.py` — the `_KNOWN_FIELDS_BY_PARENT` table and its
  `io/jones_config` import only. The table carries one row per configurable
  section so the unknown-field renderer can list what a section accepts; a new
  term with no row would report an unknown key inside `jones.D` without saying
  what `jones.D` does accept. 7D's list included this file for exactly the same
  reason and 7E's dropped it.
- `tests/unit/test_io/test_jones_config.py` — that table's test, the
  unit-suffix sweep over declared field names, and the
  "an unimplemented term letter is rejected" probe, which used `D` as its
  witness and moves to `P`. All three are assertions *about the schema*, so a
  slice that extends the schema and cannot touch them could not be green.
- `tests/characterization/test_tier7_current_behavior.py` — the 7A pins this
  slice is required to flip (`IMPLEMENTED_TERMS`, `PLANNED_TERMS`, the
  discarded-physics table, the capability-flag sweep). Section 33.2 requires
  each stub's implementation to be "a visible, deliberate flip of a named
  test", which cannot happen from outside the file that names them. 7D's list
  omitted it too and 7D flipped its own pins there.
- `tests/unit/test_tier7_jones_acceptance.py` — `IMPLEMENTED_TERM_NAMES` and
  the two counts it drives, plus invariant **I14**, whose own name is "with
  **every** implemented term enabled". I14's Tier 7D formulation compared the
  set of per-element ratios between the two sky paths, which is well defined
  only while every enabled term is diagonal; `D` mixes the feeds, so 7E
  replaces it with the matrix form `V' = M V M^H` against an independently
  written `M`. This file is in 7D's, 7H's and 7K's lists and should be in every
  term slice's.

- `docs/user_guide/configuration.rst` — the `jones` section only. It states
  "RadioSim implements two configurable terms today, `G` and `B` ... a key for
  any other term letter is rejected at parse time", which 7E makes false. A
  slice that adds four configurable terms and leaves the configuration guide
  saying they are rejected would be shipping exactly the documentation untruth
  this tier exists to remove; Tier 7J's full documentation pass is too late for
  a statement that is wrong the moment this slice lands.

- `tests/unit/test_tier1h_documentation.py` — the two assertions in
  `test_tier5g_jones_guide_states_the_receptor_science_boundaries` that pin the
  exact wording 7E is *required* to change. One asserted the guide contains
  "When Tier 7 implements `D`"; 7E implements it, so the promise is discharged
  and the assertion moves to the sentences that replace it. The other pinned the
  nine-factor chain-order formula, which gains `Rc`, `Kd` and `X`. Both are
  re-aimed rather than deleted: the property being pinned — that the guide
  states the boundary rather than implying there is none — is unchanged.

No other file outside the list was touched.

### 7F
- `src/radiosim/core/jones/parallactic.py`, `chain.py`,
  `src/radiosim/core/jones/__init__.py`
- `src/radiosim/core/jones_terms.py`, `src/radiosim/io/jones_config.py`
- `src/radiosim/core/receptor.py`, `src/radiosim/core/visibility.py`
- `tests/unit/test_jones/test_parallactic.py` (new),
  `tests/unit/test_jones/test_chain_order.py`,
  `tests/unit/test_core/test_receptor_resolution.py`,
  `tests/unit/test_core/test_receptor_solver.py`,
  `tests/unit/test_jones/test_backend_parity.py`,
  `tests/unit/test_core/test_jones_resolution.py`,
  `tests/integration/test_jones_end_to_end.py`,
  `tests/unit/test_tier5_receptor_acceptance.py` (the mount-rejection assertion only)
- `docs/user_guide/jones_matrices.rst`, `docs/user_guide/jones_terms.rst`,
  `docs/migration_guide.md`
- `Fix.md`

**Correction (7F implementation, 2026-08-01) — ten forced additions.** The
list above omits every file that states the canonical chain order, and every
file that pins that statement. 7F's *first* mandate is to move `P` sky-side of
`C` (D12), and the order is written out in six source docstrings and asserted in
four test modules; a slice that changed the constant and left the ten
statements saying something else would be shipping exactly the documentation
untruth this tier exists to remove. The omission is a defect in the list rather
than a boundary the slice should respect. Each addition is bounded and named:

- `src/radiosim/core/jones/base.py` — the `JonesTerm` class docstring's
  "canonical chain, sky → correlator" bullet list, which orders `C` before `P`.
  That list *is* the ABC's statement of the order this slice corrects; two
  bullets swap. Nothing else in the file changes, and the `compute_jones_batch`
  / `term_status` enumerations that Section 34's own 7G correction assigns to
  7G are deliberately left alone.
- `src/radiosim/simulator/base.py`, `src/radiosim/simulator/rime.py` — one
  docstring line each, both reading "The canonical Jones chain is
  `J = H @ G @ B @ D @ P @ C @ E @ T @ Z`". Tier 6H put the line there and
  Tier 6's characterization pins it; 7F is the slice that makes it false.
- `tests/characterization/test_tier5_current_behavior.py`,
  `tests/characterization/test_tier6_current_behavior.py`,
  `tests/characterization/test_tier7_current_behavior.py` — the pins on those
  docstrings and on the two mechanisms this slice deletes. Every one of them
  *names Tier 7F as its owner in its own body* ("OWNED BY: Tier 7F", "which 7F
  deletes", "so 7F's move is visible"), so the flip cannot happen from outside
  the files that name it. 7D's and 7E's lists omitted the Tier 7 file for the
  same reason and both flipped their own pins there.
- `tests/unit/test_tier7_jones_acceptance.py` — `IMPLEMENTED_TERM_NAMES` and
  the two counts it drives, plus invariant **I14**, whose own name is "with
  **every** implemented term enabled". This file is in 7D's, 7H's and 7K's lists
  and should be in every term slice's; 7E's correction said so already.
- `tests/unit/test_io/test_jones_config.py` — the `_KNOWN_FIELDS_BY_PARENT`
  coverage assertion and the "an unimplemented term letter is rejected" probe,
  whose witness is `P` and moves to `Z`. Both are assertions *about the schema*,
  so a slice that extends the schema and cannot touch them could not be green.
- `src/radiosim/io/config.py` — the `_KNOWN_FIELDS_BY_PARENT` table's `jones.P`
  row only. A new term with no row would report an unknown key inside `jones.P`
  without saying what `jones.P` does accept. 7D's list included this file for
  exactly the same reason; 7E's correction re-added it.
- `tests/unit/test_tier1h_documentation.py` — the assertions in
  `test_tier5g_jones_guide_states_the_receptor_science_boundaries` that pin the
  exact wording 7F is *required* to change: the Tier 5 chain-order formula, and
  "``feed_rotation_deg`` is a **static** rotation in the topocentric frame",
  which stops being true the moment `P` is real. Re-aimed rather than deleted,
  because the property being pinned — that the guide states the boundary rather
  than implying there is none — is unchanged.
- `docs/user_guide/configuration.rst` — the `jones` term list and the one
  sentence in the `receptors` section reading "A mount type other than
  ``fixed`` is rejected, and a non-zero ``feed_rotation_deg`` combined with an
  enabled parallactic-angle term is rejected, because the parallactic term is
  not implemented yet." Both statements are false after this slice.

`CLAUDE.md` is **not** added. Its Implementation Status and chain-order line are
Tier 7J's explicit deliverable (D0, D21), they were already stale after 7D and
7E, and 7F does not make them stale in a new way.

Two entries the list *does* carry were not needed and were not touched:
`tests/unit/test_tier5_receptor_acceptance.py` contains no mount-rejection
assertion, and `src/radiosim/core/jones_errors.py` already declared
`UnsupportedMountTypeError` — Tier 7D added the whole taxonomy at once, so R12
and R15 had their type before this slice began.

### 7G
- `src/radiosim/core/jones/ionosphere.py`, `troposphere.py`,
  `src/radiosim/core/jones/__init__.py`
- `src/radiosim/core/jones/base.py` — **correction (Tier 7E independent
  acceptance, 2026-08-01).** The list previously omitted this file. Its own
  `compute_jones_batch` docstring states directly that the method "becomes
  `@abstractmethod` in the slice that implements the last of them -- Tier 7G,
  once `Z` and `T` land," and its `term_status` docstring enumerates "nine
  exported terms ... still `\"planned\"\`" by name. 7G is the slice at which
  that enumeration reaches zero, so 7G is the one slice that can make both
  statements true again rather than merely less wrong; a slice list that never
  named the file would leave no writable-file authority for the edit its own
  governing docstring commits to.
- `src/radiosim/core/jones_terms.py`, `src/radiosim/io/jones_config.py`
- `tests/unit/test_jones/test_ionosphere.py` (new),
  `test_troposphere.py` (new), `test_backend_parity.py`,
  `tests/unit/test_core/test_jones_resolution.py`,
  `tests/integration/test_jones_end_to_end.py`
- `docs/user_guide/jones_terms.rst`, and one sentence in the sky documentation
  distinguishing intrinsic from ionospheric rotation measure
- `Fix.md`

**Correction (7G implementation, 2026-08-01) — seven forced additions.** Each is
a file that *states* something this slice makes false, or a pin that names Tier
7G as its owner in its own body; none widens what the slice does. The same
shape, and the same reasoning, as 7E's and 7F's corrections.

- `tests/characterization/test_tier7_current_behavior.py` — four pins say
  "OWNED BY: Tier 7G" in their own docstrings (the planned-term count, the
  planned-term evaluation refusal, the discarded-physics table, and the
  capability-flag table), and the D18 rotation-measure pin names 7G as the slice
  whose `Z` owns ionospheric rotation. A pin that names its owner cannot be
  flipped from outside the file that names it. 7D, 7E and 7F each flipped their
  own pins here for the same reason.
- `tests/unit/test_tier7_jones_acceptance.py` — `IMPLEMENTED_TERM_NAMES`, the
  counts it drives, and invariant **I14**, whose own wording is "with **every**
  implemented term enabled". This file is in 7D's, 7H's and 7K's lists and
  should be in every term slice's; 7E's correction said so and 7F's repeated it.
- `tests/unit/test_io/test_jones_config.py` — the `_KNOWN_FIELDS_BY_PARENT`
  coverage assertion, the unit-suffix coverage assertion, and the "an
  unimplemented term letter is rejected" probe, whose witness is `Z` and must
  move to `M`. All three are assertions *about the schema*, so a slice that
  extends the schema and cannot touch them could not be green.
- `tests/unit/test_jones/test_term_contract.py` — one test constructs a
  `JonesTerm` subclass that deliberately does not implement
  `compute_jones_batch`, to assert that the base contract raises. The
  `@abstractmethod` flip this slice is *required* to make renders that class
  uninstantiable, so the test must move to the assertion the flip replaces it
  with: the contract is now enforced at construction, and the body still raises
  for a subclass that defers to it.
- `src/radiosim/io/config.py` — the `_KNOWN_FIELDS_BY_PARENT` table's `jones.T`
  and `jones.Z` rows only. A new term with no row would report an unknown key
  inside `jones.T` without saying what `jones.T` accepts. 7D's list included
  this file for exactly this reason, and 7E's and 7F's corrections re-added it.
- `docs/user_guide/jones_matrices.rst` — its "Planned terms" section names `Z`
  and `T` as "exported, documented, and **not implemented**", and its opening
  paragraph counts the implemented terms. Both statements are false after this
  slice, and leaving them would be the documentation untruth this tier exists to
  remove. 7F's list carried this file for the same class of reason.
- `docs/user_guide/configuration.rst` — the `jones` section's sentence
  "RadioSim implements seven configurable terms today" and its list of the
  letters rejected at parse time, which names `Z` and `T`. Also 7F's precedent.

`CLAUDE.md` is **not** added: its Implementation Status and chain-order line are
Tier 7J's explicit deliverable (D0, D21), they have been stale since 7D, and 7G
does not make them stale in a new way.

### 7H
- `src/radiosim/core/jones/baseline_errors.py`,
  `src/radiosim/core/jones/__init__.py`
- `src/radiosim/core/jones_terms.py`, `src/radiosim/io/jones_config.py`
- `src/radiosim/core/visibility.py`, `src/radiosim/core/visibility_healpix.py`
- `tests/unit/test_jones/test_closure_error.py` (new),
  `test_smearing.py` (new), `test_backend_parity.py`,
  `tests/unit/test_core/test_jones_resolution.py`,
  `tests/unit/test_tier7_jones_acceptance.py`,
  `tests/integration/test_jones_end_to_end.py`
- `docs/user_guide/jones_terms.rst`
- `Fix.md`

**Correction (7H implementation, 2026-08-01) — eight forced additions.** Each
is a file that *states* something this slice makes false, a pin that names Tier
7H as its owner in its own body, or a call site of a signature this slice must
extend. None widens what the slice does. The same shape, and the same reasoning,
as 7E's, 7F's and 7G's corrections.

- `tests/characterization/test_tier7_current_behavior.py` — four pins say
  "OWNED BY: Tier 7H" in their own docstrings (the planned-term count, the
  planned-term evaluation refusal, the discarded-physics table, and the
  capability-flag table), and a fifth constructs `BaselineMultiplicativeJones()`
  with no arguments to prove `add_term` rejects it. A pin that names its owner
  cannot be flipped from outside the file that names it; 7D, 7E, 7F and 7G each
  flipped their own pins here.
- `tests/unit/test_io/test_jones_config.py` — the `_KNOWN_FIELDS_BY_PARENT`
  coverage assertion, the unit-suffix coverage assertion, and the
  "an unimplemented term letter is rejected" probe, whose witness is `M` and has
  nowhere left to move: this slice is the one after which no planned term
  exists, so the probe is replaced by the assertion that every accepted letter
  is implemented. All three are assertions *about the schema*, so a slice that
  extends the schema and cannot touch them could not be green.
- `tests/unit/test_jones/test_term_contract.py` and
  `tests/unit/test_jones/test_chain_order.py` — both construct
  `BaselineMultiplicativeJones()` and `SmearingFactorJones()` with no arguments
  (for the D7 `add_term` rejection), and `test_term_contract.py` additionally
  asserts that the *base* `compute_baseline_factor` raises. This slice gives
  both terms constructors that take resolved values and makes the method
  `@abstractmethod`, so both statements must move to the assertions the flip
  replaces them with — exactly the change 7G's correction made to the same file
  for `JonesTerm`.
- `tests/unit/test_jones/test_bandpass.py` — two `resolve_jones_terms(...)` call
  sites. The function gains two required keyword parameters (below), and a call
  site that cannot be updated would fail to collect.
- `src/radiosim/api/simulator.py` — `_ensure_jones_terms` only. It is the one
  production caller of `resolve_jones_terms`, and R14 ("a `per_baseline` pair
  absent from the resolved baseline selection") is a stage-3 rejection, so the
  resolver must be given the selection the run actually has; `Q` likewise needs
  the resolved channel widths. Both are already retained on the simulator at
  exactly that point (`self._instrument_state.selection`,
  `self._resolved.frequency`), so this is three added arguments and nothing
  else. A resolver that could not see the selection could not raise R14 at all,
  and Section 26.1 puts it before the first side effect.
- `src/radiosim/io/config.py` — the `_KNOWN_FIELDS_BY_PARENT` table's `jones.M`
  and `jones.Q` rows only. A new term with no row would report an unknown key
  inside `jones.M` without saying what `jones.M` accepts. 7D's list included
  this file for exactly this reason and 7E, 7F and 7G each re-added it.
- `docs/user_guide/jones_matrices.rst` — its "Planned terms" section names `M`
  and `Q` as "exported, documented, and **not implemented**". After this slice
  there is no planned term at all, so the section is replaced rather than
  edited, and leaving it would be the documentation untruth this tier exists to
  remove.
- `docs/user_guide/configuration.rst` — the `jones` section's sentence
  "RadioSim implements nine configurable terms today" and its statement that a
  key for `M` or `Q` "is rejected at parse time". Both are false after this
  slice. Also 7F's and 7G's precedent.

`CLAUDE.md` is **not** added: its Implementation Status and chain-order line are
Tier 7J's explicit deliverable (D0, D21), they have been stale since 7D, and 7H
does not make them stale in a new way.

### 7I
- `src/radiosim/core/beam/models.py`, `resolution.py`, `runtime.py`,
  `analytic.py`, `errors.py`
- `src/radiosim/io/config.py` (the `beams` section only)
- `src/radiosim/core/jones/beam/TODO.md` (delete)
- `docs/development/beam_physics_scope.md` (new)
- `tests/unit/test_core/test_beam_pointing.py` (new),
  `tests/unit/test_core/test_beam_models.py`,
  `tests/unit/test_core/test_beam_resolution.py`,
  `tests/unit/test_core/test_beam_runtime.py`
- `docs/user_guide/beam_models.rst`, `docs/user_guide/configuration.rst`
- `Fix.md`

**Correction (7I implementation, 2026-08-02; independent review adds a
seventh, same day) — seven forced additions.** Each is a file that *owns*
something this slice must change, a pin that names Tier 7I as its owner in its
own body, a call site that this slice's physics makes incorrect, or a hardcoded
assertion this slice's own mandated public surface makes wrong. None widens
what the slice does. The same shape, and the same reasoning, as 7E's, 7F's,
7G's and 7H's corrections.

- `src/radiosim/io/beam_config.py` — the list says "`src/radiosim/io/config.py`
  (the `beams` section only)", but no `beams` section lives there. `io/config.py`
  imports `BeamsConfig` from `io/beam_config.py`, which has owned every
  user-authored beam input model since Tier 3B; `io/config.py` holds only the
  field, the removed-field guidance table, and the `_KNOWN_FIELDS_BY_PARENT`
  rows. The named file is the one this slice cannot avoid, so it is the one the
  list must name.
- `src/radiosim/io/config_resolution.py` — `_resolve_beam_input` is the single
  function that turns a `BeamsConfig` into a `ResolvedBeamsInput`. A new authored
  block that the resolver cannot see would parse and then vanish, which is
  defect D2's shape exactly.
- `src/radiosim/core/beam/__init__.py` — the beam package's public re-export
  surface. New resolved types that `core/beam/models.py` exports and
  `core/beam/__init__.py` does not are reachable only by submodule path, which
  the package has not done for any other resolved beam value.
- `src/radiosim/core/visibility.py` — `_ResolvedBeamJones`'s per-step cache
  **only**. The adapter caches one evaluated `(n_dir, 2, 2)` block per
  `handler_id`, which is correct exactly as long as two antennas sharing a
  handler have identical responses. Per-antenna pointing offsets and per-antenna
  surface errors are the first thing in the tier's history that breaks that: two
  antennas of the same diameter and model share one analytic handler and must
  now differ. Left alone, the first antenna's response would be silently served
  to every other antenna on its handler — a wrong answer, not a missing feature.
  The slice replaces the cache key with the response key `BeamSystem` publishes,
  which is the `handler_id` itself whenever no offset and no surface error is
  configured, so the absent case is unchanged by construction.
- `tests/characterization/test_tier7_current_behavior.py` — one pin,
  `test_beam_todo_markdown_is_the_sci_003_artifact`, says "OWNED BY: Tier 7I" in
  its own docstring and asserts that `docs/development/beam_physics_scope.md`
  does **not** exist. A pin that names its owner cannot be flipped from outside
  the file that names it; 7D, 7E, 7F, 7G and 7H each flipped their own pins here.
- `tests/unit/test_core/test_beam_solver_integration.py` — it is the file that
  exercises `_ResolvedBeamJones` against a real `BeamSystem`, so it is where the
  shared-handler/differing-response regression above is provable end to end. A
  slice that changes the adapter's cache and cannot touch its integration test
  would be asserting the fix nowhere.
- `tests/unit/test_core/test_beam_fits.py` — Section 19.2 requires the two
  Ruze closed forms to be public
  (`radiosim.core.beam.runtime.ruze_power_efficiency`,
  `ruze_voltage_factor`), and this file's own
  `test_fits_modules_do_not_publish_private_runtime_symbols` hardcodes
  `runtime.__all__ == ["BeamSystem", "load_beam_system"]`. Left alone, that
  assertion would fail the moment the mandated public surface exists; the
  slice cannot both honor Section 19.2's "public and documented" requirement
  and leave this pin unwritten. Flagged as undeclared during independent
  review and ratified here for the same reason as the other six: the file
  owns a pin this slice's own physics makes wrong, and the change is the
  four-line assertion update the pin requires, nothing wider.

Independent review also traced `src/radiosim/core/__init__.py`'s twelve-line
diff (six new resolved names re-exported at the package root) and found it
**not** a further undeclared file: `tests/unit/test_core/test_beam_models.py`
(already on the base 33.2 list) contains
`test_resolved_types_are_exported_only_from_core_boundaries`, which iterates
`core.beam.models.__all__` and asserts every name is also in `core.__all__`.
Adding the two Tier 7I resolved-value families to `models.__all__` (required,
since `core/beam/__init__.py`'s own re-export is one of the six original
forced additions) makes that pre-existing, already-declared test force the
`core/__init__.py` change; it is a consequence of a declared file, not a
seventh gap.

`CLAUDE.md` is **not** added: its Implementation Status and beam paragraph are
Tier 7J's explicit deliverable (D0, D21), and 7I does not make them stale in a
new way — the beam subsystem was already described as the most developed one.

### 7J
- `docs/api/jones.rst`, `docs/user_guide/jones_matrices.rst`,
  `docs/user_guide/jones_terms.rst`, `docs/user_guide/configuration.rst`,
  `docs/migration_guide.md`, `docs/changelog.rst`, `docs/index.rst`
- `CLAUDE.md` (Implementation Status and the Jones sections only)
- `README.md` (the Jones capability statement only)
- `pixi.toml` (the optional `crossval` feature, **only** if Q1 resolved yes)
- `tests/unit/test_tier1h_documentation.py`,
  `tests/crossvalidation/test_pyuvsim_comparison.py` (new, marked, **only** if
  Q1 resolved yes)
- `output/crossvalidation/` (evidence artifact)
- `Fix.md`

### 7K
- `tests/unit/test_tier7_jones_acceptance.py`
- `Fix.md` (register rows `SCI-001`, `SCI-002`, `SCI-003`; new rows `SCI-004`,
  `SCI-005`; the whole-tier acceptance record)
- `Tier7JonesSciencePlan.md` (an acceptance appendix only, never a rewrite of
  the design)

## 35. Independent acceptance gate and stop boundary

After every slice, without exception:

1. the implementer stops and hands off; no slice begins the next slice;
2. an independent reviewer re-derives the slice's claims from source, reruns
   the Section 32 gate, and reproduces at least one rejection message and one
   invariant **by hand**, not by rereading the test;
3. the reviewer writes a dated acceptance record in `Fix.md` naming what was
   verified, what was not observed, and any defect found;
4. a defect found in a slice is either fixed in that slice or explicitly routed
   forward with a named owner — never silently absorbed;
5. no register row flips except at 7K, and then only with the Section 38
   evidence present.

The stop boundary is the same as Tiers 5 and 6: after each accepted slice, work
stops until the next slice is explicitly authorized. Nothing is pushed. Commits
are local and conventional, one per slice, following `Fix.md` §16's suggested
pattern:

- `feat(jones): implement <specific term>`
- `test(jones): validate <term> against <reference>`
- `refactor(jones): batch the Jones evaluation contract` (7B)
- `refactor(jones): remove identity stubs and the unhonored calculation_type` (7C)
- `feat(config): add the typed jones section` (7D)
- `feat(beam): add pointing offsets and Ruze efficiency` (7I)
- `docs(jones): reconcile the Jones documentation with implemented physics` (7J)

## 36. Breaking-change ledger

Every entry needs a `docs/migration_guide.md` line and a changelog line.

| # | Change | Slice | Migration |
|---|---|---|---|
| B1 | `JonesTerm.compute_jones` and `compute_jones_all_sources` removed; replaced by `compute_jones_batch` | 7B | subclass the new keyword-only batched method; see the new `jones_terms` guide |
| B2 | `JonesChain.compute_antenna_jones`, `compute_antenna_jones_all_sources`, `compute_baseline_visibility` removed | 7B | use `compute_antenna_jones_batch`, or `evaluate_antenna_jones` |
| B3 | `JonesBaselineTerm.compute_baseline_term` removed; replaced by `compute_baseline_factor` | 7B | batched signature |
| B4 | `GeometricPhaseJones` removed | 7B | call `radiosim.core.jones.geometric_phase()` |
| B5 | 26 stub classes removed (Section 23) | 7C | per-class replacement table in the migration guide; every entry names the surviving term and the configuration field that replaces it |
| B6 | `CrosshandPhaseJones` renamed `CrosshandJones` | 7C | rename; cross-hand delay is now the `delay_s` field of the same term |
| B7 | `jones_config` parameter removed from four public signatures | 7C | configure `jones:` in YAML; the API takes `ResolvedJonesTerms` |
| B8 | `visibility.calculation_type` removed | 7C | delete the key; use `execution.simulator` |
| B9 | New `jones:` config section | 7D | additive; absence is the previous behavior exactly |
| B10 | `JonesPrecision` gains six fields | 7D | additive with defaults |
| B11 | HDF5 schema bump with a `jones/` group | 7D | readers accept files without the group |
| B12 | `scientific_sha256` changes for any run that configures a Jones term | 7D | fingerprints of runs with no `jones:` section are unchanged, and 7A's pins prove it |
| B13 | Canonical chain order changes: `P` moves sky-side of `C` | 7F | affects only runs with `P` enabled, which cannot exist before 7F; supersedes `Tier5ReceptorFeedPlan.md` §19.1 for `P` only |
| B14 | Non-`fixed` mount types accepted when `P` is enabled; rejected with a new message when it is not | 7F | supersedes `core/receptor.py:411-418` |
| B15 | `beams` config gains pointing-offset and surface-error fields | 7I | additive |
| B16 | `src/radiosim/core/jones/beam/TODO.md` moved to `docs/development/beam_physics_scope.md` | 7I | documentation move |

## 37. Final whole-tier acceptance criteria

All twenty must hold at 7K, each with named evidence.

1. `radiosim.core.jones.__all__` contains exactly 19 names -- Section 9.1's 16
   class names (3 base classes, 13 concrete terms) plus the three non-class
   exports Tier 7B added (`DirectionBatch`, `evaluate_antenna_jones`,
   `geometric_phase`) -- and every one of the 26 removed classes plus the
   renamed `CrosshandPhaseJones` raises on import with a migration-guide
   sentence available. **Correction (7C independent acceptance, 2026-08-01):**
   the "16 names" figure above was Section 9.1's count of *classes* at
   design-gate time, before 7B introduced the three non-class exports; taken
   literally against `__all__`'s actual contents it undercounts by three. This
   is the same species of drift the `68458da` correction already fixed for
   Section 23's "26 removed names" (actually 27, because it includes
   `GeometricPhaseJones`) -- left unreconciled here because Section 34 does not
   give 7C write access to Section 37, and 7C's own writable list is exact
   about the true 19-name total (`tests/unit/test_tier7_jones_acceptance.py::
   test_the_jones_package_exports_exactly_the_surviving_names`, independently
   reproduced by this review: `len(SURVIVING_JONES_NAMES) == 19`, 13 term
   classes, 11 of them planned). No decision changes; 7K's criterion is now
   stated correctly rather than left to a future reviewer to rediscover.
2. Every exported Jones class implements real physics: `term_status ==
   "implemented"` for all of them, and no `compute_jones_batch` in `src/`
   returns an identity for all inputs.
3. A repository-wide scan of `src/radiosim` finds no `TODO: implement
   properly`, no `Stub:` docstring, and no `xp.eye(2, dtype=np.complex128)`
   returned unconditionally.
4. Every implemented term has: a cited convention in its docstring; documented
   units, axes, and signs; at least one analytic invariant test; a backend
   parity case; and an effect-changes-visibility test. (`Fix.md` §16 rules 1-5,
   verified per term.)
5. Every declared capability flag (`is_diagonal`, `is_scalar`, `is_unitary`) is
   numerically verified over a parameter sweep, with a negative case proving
   the flag is not vacuous (I2).
6. Every accepted `jones:` configuration changes the visibilities; every
   configuration that could not is rejected with the exact message of Section 24
   (R7 and the rest), each reproduced by hand at review.
7. `jones:` absent reproduces the 7A cube digests and `scientific_sha256` bit
   for bit, for all four shipped configurations (I1).
8. The point and HEALPix paths agree within the Tier 6 tolerance with every
   term enabled (I14).
9. Dask is bit-identical to NumPy and JAX-CPU agrees within `rtol=1e-12`, per
   term and for all terms together (Section 28).
10. `src/` contains exactly one `backend.compile` call site and the compiled
    kernel's signature is unchanged from `ac4fe41` (I16).
11. The canonical chain order is the Section 12.2 one, proven with
    non-commuting synthetic terms, and the circular-receptor `P` placement test
    (I6) passes.
12. No Jones parameter reaches a solver except through `ResolvedJonesTerms`;
    no raw dict survives; `hybrid.py`'s hard-coded `None` is gone.
13. `jones_sha256` enters `scientific_sha256`; no filesystem path enters either
    (I13).
14. HDF5, summary JSON, and `SimulationResult` carry the Jones provenance; MS
    and UVFITS are unchanged; observability output is unchanged (I18).
15. `execution.simulator`'s accepted values equal the simulator registry keys
    (I15); `visibility.calculation_type` is absent from the schema, from the
    four shipped configs, and from the documentation.
16. Every implemented term respects `PrecisionConfig`; no `complex128` is
    hard-coded anywhere in `src/radiosim/core/jones` (I17).
17. `docs/api/jones.rst`, `docs/user_guide/jones_matrices.rst`, the new
    `docs/user_guide/jones_terms.rst`, `docs/user_guide/configuration.rst`,
    `CLAUDE.md`, and `README.md` describe exactly the implemented surface, with
    no "scaffold", "stub", or "identity" language surviving about a term that
    now exists, and no capability language about one that does not.
18. Beam pointing offsets and Ruze efficiency are implemented and verified
    (I19); `docs/development/beam_physics_scope.md` gives every remaining
    `TODO.md` item a disposition, a citation, and an owning register row.
19. Cross-validation evidence exists at the level Section 29 requires: the
    Tier-1 comparisons are in the gate; the Tier-2 comparison is either a
    committed artifact or an explicitly recorded non-observation. No validation
    claim exists without one of the two.
20. The Section 32 gate passes: full suite, zero xfail, zero XPASS, lint,
    format, whitespace, Sphinx warnings not increased, and typecheck within its
    ceiling.

## 38. Evidence required to close SCI-001, SCI-002, SCI-003

### `SCI-001` — Jones identity stubs

Closes **DONE** only with all of: criteria 1-6, 11, 12, 16 above; the per-term
evidence table showing, for each of the eleven newly implemented terms, its
citation, its invariant test names, its parity result, and its I7 delta; the
26-name removal ledger with a migration line each; and the reviewer's own
by-hand reproduction of at least three rejections and three invariants.

Closure text must state the scope precisely: *every exported Jones class
implements real physics; twenty-six speculative stubs were removed rather than
implemented, each with a documented replacement; no public term multiplies by
identity.*

### `SCI-002` — spherical-harmonic mode

Closes **DONE** by **absence from accepted config**, which `Fix.md` §16's exit
criterion explicitly permits. Evidence: criterion 15; the four shipped configs
carrying no `calculation_type`; the removed-field guidance R1 reproduced by
hand; the registry-equality invariant I15; and the newly filed `SCI-004` row
naming the successor design gate.

Closure text must not imply an m-mode solver exists. Required wording: *closed
by removal of the unimplemented option and of its unhonored sibling value from
the public configuration surface; `execution.simulator` is the single solver
selector and accepts only `rime`; the m-mode solver is filed as `SCI-004`.*

### `SCI-003` — advanced beam-physics TODOs

Closes **DONE** with criterion 18: two items implemented and analytically
verified (pointing offsets, Ruze efficiency), five given explicit scientific
scope with citations in a tracked scope document, and `SCI-005` filed as their
owner. The in-package `TODO.md` no longer exists.

## 39. Risk register

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| 1 | The batched-contract refactor (7B) changes results by a floating-point ulp somewhere, and the "bit-identical" claim fails | medium | 7A pins digests *before* 7B; if a difference appears, it must be **explained and localized** (most likely the chain's identity-seed dtype), never accepted by loosening the pin. If the explanation is legitimate the pin is flipped with the reason recorded — the Tier 6D precedent. |
| 2 | The HEALPix path's per-handler beam caching is lost when it moves behind `evaluate_antenna_jones`, silently costing a large factor on diffuse runs | medium | the cache moves *inside* the `E` adapter rather than being deleted; 7B records a wall-time before/after for the largest shipped HEALPix config as evidence, without making a performance claim |
| 3 | Direction-batched DDE evaluation over HEALPix pixel counts blows host memory: `(n_pix, 2, 2)` complex128 per antenna is ~`16·n_pix` bytes per matrix element block | medium-high | **Q2** measures it at 7A. The mitigation, if needed, is that the HEALPix path already loops over pixel batches; the batch size becomes the direction-batch size, and DIE terms stay `(1,2,2)` by contract (I3) so only genuinely direction-dependent terms scale |
| 4 | The `P` chain-order correction (7F) is judged to contradict an accepted Tier 5 decision and is rejected at review | low-medium | Section 12 states the physics, the algebra, and the exact test (I6) that distinguishes the two orders *before* the slice runs, so the review is over evidence rather than over preference; Section 12.4 records the supersession scope explicitly |
| 5 | `pyuvsim` will not resolve against `pyuvdata ==3.2.1`, leaving Workstream A/B with no external cross-check | medium-high | Q1 answers it at 7A; Section 29's Tier-1 evidence (astropy for `P`, published closed forms for the rest) is designed to be sufficient on its own, and the Tier-2 comparison is explicitly a bonus, not a gate |
| 6 | The identity-rejection rule R7 is over-eager and rejects a legitimate configuration (for example a bandpass whose polynomial is 1 only within rounding) | medium | R7 is evaluated on the *resolved* parameters against an exact-equality test, not a tolerance, so a parameter that is 1 to within `1e-16` but not exactly 1 is accepted; the test suite includes that boundary case deliberately |
| 7 | Eleven terms across seven slices drifts in style, so each term ends up with a slightly different config shape and test shape | medium | Sections 20, 21, 24, and 31 fix the shape in advance: the same `per_antenna` override structure, the same rejection families, the same five-step test order for every term |
| 8 | `scientific_sha256` changes for existing users' runs | certain, and intended | B12; the pins prove that only runs *configuring* a Jones term change, and 7A's digests are the proof for those that do not |
| 9 | The tier is large enough that a slice quietly expands its writable list | medium | Section 34 is exact; Section 35 requires a bounded plan correction rather than an expansion, with the Tier 5/6 precedent |
| 10 | Documentation drifts behind the eleven implementations, ending in one impossible 7J | medium | each term slice owns its own `jones_terms.rst` entry (Section 34), so 7J reconciles rather than writes |
| 11 | **(Added, 7F independent acceptance, 2026-08-01.)** `jones.P`'s five mount types are unreachable from the two config-driven `instrument:` sources: `io/instrument_sources.py` hard-codes `mount_type=None` for both a layout file (line 352) and the known-telescope registry (line 444), so only a pyuvdata dataset (which carries its own `mount_type` array) can produce a non-`fixed` mount today. `P` is real and correctly wired, but a user with an alt-az array described by a plain layout file has no YAML field to declare it, and R12/R15's protection is therefore moot for that source. | medium | informational; no instrument-config field is added by Tier 7 (out of `jones_terms.py`'s writable list). Routed to a future instrument-config tier as an explicit `instrument.source.layout_file.mount_type` (or per-antenna override) field; until then, a heterogeneous non-pyuvdata array cannot exercise `jones.P` at all. |

## 40. Explicit exclusions and the successor boundary

Tier 7 does not implement, and no slice may add:

- a spherical-harmonic / m-mode solver, or any second forward model
  (`SCI-004`);
- any non-scalar E-Jones, polarized BeamFITS, Ludwig-3 decomposition,
  quadrupolar cross-polarization, beam squint, aperture blockage, Zernike
  aberration, or Ruze error-beam decomposition (`SCI-005`);
- station element beams, array factors, or mutual coupling (`SCI-005`);
- any ingestion of external geophysical data: IONEX/GPS TEC, geomagnetic field
  models, radiosonde or numerical-weather-model profiles, or archived
  calibration tables;
- any stochastic screen, random gain draw, or Monte-Carlo realization;
- any calibration or solving capability, including fringe fitting;
- any imaging operator: gridding, FFT, W-projection, A-projection;
- any GPU, TPU, distributed, or performance claim (`PERF-001`);
- any Numba kernel, any second `backend.compile` call site, or any change to the
  compiled kernel's signature;
- `SKY-002`'s composite-recipe network metadata, which remains routed
  pre-Tier-8;
- a repository-wide documentation rewrite or any release work beyond the Jones
  and beam pages this tier's own changes make false (Tier 8);
- physical GPU/TPU validation, live network validation, registry validation,
  deployment, tagging, publishing, or any remote operation.

## 41. Open questions

Each names the slice that must resolve it, the evidence required, and what
happens if the evidence contradicts this plan. No slice may proceed past a
question that blocks it by assuming an answer.

**Q1 — Does a cross-validation reference resolve against the locked
`pyuvdata ==3.2.1`? (blocks the Tier-2 half of Section 29; must be answered in
7A.)** `pyuvsim`, `matvis`, and RASCIL are absent from every locked environment
(`pixi.toml:31-72`), and `pyuvsim` pins `pyuvdata`. 7A must record exact
resolvable versions, or the exact resolution failure, for `linux-64`, `osx-64`,
and `osx-arm64`. If a version resolves, 7J adds an optional `crossval` pixi
feature and a marked, non-gating comparison test. If none resolves, the Tier-2
comparison becomes a **recorded manual run** with the reference version,
machine, date, and compared numbers written into 7J's acceptance record — never
a silent skip, and Section 29.2's forbidden claims stay forbidden. This does not
block any term slice: Section 29.1's Tier-1 evidence is designed to be
sufficient alone.

**Q2 — What is the host-memory cost of direction-batched DDE evaluation on the
largest shipped HEALPix configuration? (blocks 7B's acceptance framing, not its
implementation; must be answered in 7A.)** A DDE term returning `(n_dir, 2, 2)`
complex128 costs `64 · n_dir` bytes per antenna per `(time, frequency)` step,
and the HEALPix direction count is the pixel count. Evidence required:
`tracemalloc` peak for `configs/realistic_foreground_example.yaml` under the
current code, plus the arithmetic for the same configuration with `P`, `Z`, and
`T` enabled. If the projected peak exceeds the current peak by more than a
factor of two, 7B must make the HEALPix direction batch a *chunked* loop whose
chunk size is derived from a memory budget — a documented batching strategy
decided on measured evidence rather than guessed here. The DIE `(1, 2, 2)`
contract (I3) is what keeps this bounded to genuinely direction-dependent terms.

**Q3 — Does any shipped configuration, example, doctest, or fixture set
`visibility.calculation_type` in a way that 7C's removal would break beyond a
mechanical edit? (blocks 7C.)** At this gate the field appears in four shipped
configs (`configs/config.yaml:65`,
`configs/receptor_circular_example.yaml:75`, `configs/hybrid_sky_example.yaml:93`,
`configs/realistic_foreground_example.yaml:65`), in
`docs/user_guide/configuration.rst:66,183,217`, and in eight test locations, all
setting `direct_sum` except two that deliberately exercise the rejection. 7C
must confirm the same for every example script, every doctest, and every fixture
in `tests/fixtures/configs.py` before removing it. If a consumer is found that
uses the value for anything, the removal is re-argued rather than pushed
through.

**Q4 — Is the `E`/`P` relative order genuinely unobservable at the accepted
scalar-E subset? (blocks 7F's acceptance framing.)** Section 12.3 asserts it on
the grounds that a scalar commutes with everything. 7F must confirm it
numerically: swap the two in a test build and assert bit-identity across the
FITS beam path as well as the analytic one, because the FITS path's 2×2
assembly is where a non-scalar element could hide. If it is *not* bit-identical,
the E-Jones is not scalar in that path, which is a Tier 3 contract violation and
must be raised as a new register row rather than absorbed.

**Answered (7F independent acceptance, 2026-08-01).** The 7F implementation
commits did not carry a checked-in test for this specific question, so the
independent reviewer ran the confirming build directly: `Simulator.from_mapping`
with an `alt-az`-restamped two-antenna array, `jones.P` enabled, and a circular
receptor, once through the canonical `CANONICAL_CHAIN_ORDER` and once with `E`
and `P` swapped (monkeypatched on `radiosim.core.visibility.CANONICAL_CHAIN_ORDER`),
for both `beams.mode: analytic` and `beams.mode: shared_fits` (a real
`write_scalar_efield_beamfits` fixture beam). The two visibility cubes were
`np.array_equal` (bit-identical) on both beam paths. Q4 is answered **yes**: the
order is genuinely unobservable at the accepted scalar-`E` subset, and no
register row is needed. See `Fix.md`'s 2026-08-01 "Tier 7F independent
acceptance" note for the exact probe.

**Q5 — Does `M` interact with the Tier 6 per-time block accumulation in a way
that changes the accumulation order? (blocks 7H.)** `M` applies to the kernel's
`(B, 2, 2)` output before the block is cast to `output_complex_dtype`
(`visibility.py:721`). 7H must confirm that this leaves `backend.stack`'s
per-time assembly (`Tier6HybridRuntimePlan.md` §13.3) structurally unchanged and
that the Tier 6 accumulation invariants still hold. If the cast position must
move, that is a Tier 6 contract touch and requires a bounded plan correction
before 7H proceeds.

**Answered (7H implementation, 2026-08-01): no, and the cast position does not
move.** `M` is applied exactly where this paragraph says — to the kernel's
`(B, 2, 2)` return, one expression before the existing
`backend.asarray(block, dtype=output_complex_dtype)` — so the block that reaches
`freq_blocks.append(...)` has the same shape and the same dtype it had at
`d4d1019`. Nothing downstream of that append changes: the per-frequency list, the
per-time `backend.stack(freq_blocks, axis=1)`, the worker partition, and the
final `backend.stack(time_blocks, axis=0)` are untouched in both solvers, and
the two `backend.stack` accumulation call sites per solver are still exactly
two. Tier 6's own characterization and worker-policy pins are untouched by this
slice and pass unchanged, which is the evidence: they assert the assembly shape,
the one-assembly-per-call property, and worker-count-independence, and all three
are properties of code `M` does not enter. The elementwise multiply happens at
accumulation precision, which is what Section 15.2 asks for.

**Q6 — Should `Q`'s smearing use the per-channel width or the total bandwidth
when the frequency grid is nonuniform? (blocks 7H.)** Tier 1G made nonuniform
explicit frequency arrays first-class, so a "channel width" is not uniquely
defined for them. 7H must decide from the resolved frequency model: the
candidate rule is the local spacing to the nearest neighbouring channel, with a
single-channel observation rejected rather than assigned an invented width. The
decision must be recorded in `docs/user_guide/jones_terms.rst` and tested on a
deliberately nonuniform grid.

**Answered (7H implementation, 2026-08-01): neither — the per-channel width is
already resolved, and the candidate rule is rejected.** The question's premise
is wrong: a nonuniform frequency array is not "centres only". Tier 1G's
`ExplicitFrequencyConfig` requires `channel_widths_hz` **alongside**
`channel_frequencies_hz`, of the same length, every value finite and positive
(`io/config.py`), and `_resolve_frequency` carries them into
`ResolvedFrequencyConfig.channel_widths_hz` for both the `grid` and the
`explicit` mode (`io/config_resolution.py`). That same per-channel width is what
`SimulationResult.channel_widths_hz` reports and what the summary's
`minimum_width_hz`/`maximum_width_hz` are computed from. `Q` therefore reads
`channel_widths_hz[freq_idx]` and invents nothing. Deriving a width from the
spacing to the neighbouring channel would make `Q` decorrelate by a bandwidth
that contradicts the one the same run publishes in its own outputs, which is a
worse failure than the one the candidate rule was guarding against; and a
single-channel observation needs no rejection, because it carries a declared
width like every other. Tested on a deliberately nonuniform grid — three
channels with three different declared widths, where the per-channel smearing
tracks each channel's own width and not the spacing — and recorded in
`docs/user_guide/jones_terms.rst`. `dt` is the same story one axis over:
`ObservationTimeGrid.integration_time_seconds` is a resolved per-sample array.
