# Post-Tier-8 Remediation Program Plan

Written 2026-08-05 at `main` `50c2620` (v0.3.0, all of Tiers 0–8 accepted). This
document sequences and sketches the successor work for the eight register rows
left `OPEN`/`ROADMAP` at the end of the remediation program. It is a **program
plan**, not a tier: it grants no writable file lists, defines no acceptance
criteria of its own, and changes no register row. Every substantial work
package below still gets its own design-gate memo before implementation, and
register rows still change only through independent acceptance records with
exact closure text, per the `Fix.md` §4–§5 discipline.

Volatile facts in this document (CI incidence, artifact availability, run IDs)
are cited to the records that hold them **as of the date above** and must be
re-verified at execution time, not trusted from here.

---

## 1. The eight rows and the one that is different

| Row | Status | One line |
|-----|--------|----------|
| `CI-001` | DONE | Closed 2026-08-08 by WP-2/WP-3: the second `linux-64-py311` class was forced on demand, its NumPy/OpenBLAS dispatch axes named, its three cube deltas accepted under Section 13.5, and its digests recorded (`docs/development/ci001_adjudication.md`) |
| `SCI-006` | DONE | Closed 2026-08-11 by WP-5 independent acceptance at exact candidate `f5fa101e`; CI run `31434253575` passed quality, backend parity, and all six compatibility cells, with six authenticated characterization artifacts and exact `P V_old P^H` evidence (`docs/development/sci006_polarization_convention.md`) |
| `SCI-007` | DONE | Closed 2026-08-11 by WP-6 independent acceptance at exact evidence successor `e20f636`: the retained schema-1.2.0 artifact attributes the retained fixture's residual to omitted ICRS-to-operational-apparent tangent-basis transport; exact-SHA CI run `31461141190` passed all eight jobs (`docs/development/sci007_frame_accuracy_bound.md`) |
| `API-001` | DONE | Closed 2026-08-11 by WP-1 independent acceptance at exact candidate `87bdaf2`: NumPy/JAX/Dask broadcasting and six equal-shape backend/dtype byte-identity cells passed |
| `API-002` | DONE | Closed 2026-08-11 by WP-1 independent acceptance at exact candidate `87bdaf2`: all four helpers and `RichHandler` render caller text literally, with no caller markup dependency |
| `PERF-001` | ROADMAP | Accelerator performance undemonstrated; JAX-CPU measured slower than NumPy on every benchmarked workload (`Fix.md` line ~208) |
| `SCI-005` | ROADMAP | Beam physics beyond scalar `E = e·I₂`: five scoped items in `docs/development/beam_physics_scope.md` (`Fix.md` line ~216) |
| `SCI-004` | ROADMAP | The m-mode / spherical-harmonic simulator — a second complete forward model (`Fix.md` line ~215) |

**`CI-001` is not like the others.** Its register row states it "makes `main`
red on one of eight CI jobs" and ends "**Blocks any 'CI is green' claim while
open**". At Tier 8's reconciliation, 11 of the 25 most recent CI runs had
failed, all in the same pin family (`Tier8ReleasePlan.md` §14), at ~38%
incidence on the `linux-64-py311` cell (`Tier8ReleasePlan.md` §5.6, §20 risk
4). The "8/8 green" release snapshot is true of the acceptance SHA (run
`30749117742` at `ce7f525`) and of nothing after it. `CI-001` therefore leads
this program, and — see edge E1 below — its adjudication also gates the one
science change that would regenerate fingerprint pins.

**Execution update, 2026-08-08:** WP-2 and WP-3 are accepted and `CI-001` is
closed. The historical paragraph above is retained because it explains the
priority and experiment design; the authoritative adjudication and successor
gate are in `docs/development/ci001_adjudication.md`. Dependency E1 is
satisfied.

## 2. Evidence base

This plan was assembled from, and cites only:

- `Fix.md` §5 issue register (line ~170) and the tail acceptance records:
  7J (line ~13546), 7K (~13760), Tier 8 design gate and acceptance (~14080,
  ~14230), 8A (~14436), 8B (~14570), 8D (~14953), 8E (~15168), 8F (~15378),
  program completion (~15446).
- `docs/changelog.rst`, the `[0.3.0]` "Known limitations" section (the
  register-wide disclosure surface; names all eight rows by ID).
- `Tier6HybridRuntimePlan.md` §13.6, §21, §27, §37–§39, §41, §42.
- `Tier7JonesSciencePlan.md` §4, §5.5, §12.3, §13.2, §14.2, §18, §19.3, §27,
  §29, §34, §37, §39–§41.
- `Tier8ReleasePlan.md` §4, §5.6, §14, §20, §21, §24.
- `tests/characterization/test_tier6_current_behavior.py` (module docstring:
  the observation-set scheme, axes 1–3, and the Tier 8A instrumentation).
- `output/crossvalidation/2026-08-02-pyuvsim-1.4.0.json` and
  `tests/crossvalidation/test_pyuvsim_comparison.py`.
- `output/benchmarks/reference/` (`20260731T104303Z-darwin-arm64.json`).
- `docs/development/beam_physics_scope.md`.
- Current source for the bounded items: `src/radiosim/utils/logging.py`,
  `src/radiosim/api/simulator.py:773-787`, `src/radiosim/core/polarization.py`,
  `src/radiosim/core/contraction.py`, `.github/workflows/ci.yml`.

## 3. Priority order and dependency edges

| # | Row | Verdict | Effort | Starts |
|---|-----|---------|--------|--------|
| 1 | `CI-001` | Diagnose from live instrumentation; adjudicate under the pre-authorized Tier 8 §14 conditional; then the successor-gate decision | M (elapsed-time-gated on one red run) | Now (WP-2) |
| 2 | `API-002` | Fix: escape at the helper boundary, plus the `RichHandler` companion | XS | DONE 2026-08-11 (WP-1) |
| 3 | `API-001` | Fix: implement broadcasting | XS | DONE 2026-08-11 (WP-1) |
| 4 | `SCI-006` | Selected east-X correction implemented and independently accepted in WP-5 | S + M | DONE 2026-08-11 |
| 5 | `SCI-007` | Reconciled and closed as a documented, test-pinned fixture bound | S–M | DONE 2026-08-11 (WP-6) |
| 6 | `PERF-001` | Four CPU legs landed and exact-SHA CI green; retained CPU acceptance pending; GPU evidence hardware-gated | M + gated | CPU evidence and acceptance now (WP-7) |
| 7 | `SCI-005` | Staged successor tier, own plan document; scalar-preserving items first | XL | Plan doc now; stages gated (WP-8) |
| 8 | `SCI-004` | Feature, not remediation; Q5 driver adopted and full design gate now drafted | XL | Design review now; production phase-gated (WP-9) |

Effort classes: **XS** ≤ half a day · **S** ≈ 1–2 days · **M** ≈ 3–7 days ·
**L** ≈ 2–3 weeks · **XL** = month or more.

**The dependency edges that bind the order:**

- **E1 — `CI-001` adjudication before any pin-regenerating landing** (concretely:
  before the `SCI-006` flip). Regenerating polarized fingerprints means
  re-characterizing every environment cell. On `linux-64-py311` that requires
  capturing **both** machine classes for the new pins, and while the
  discriminator is unnamed, class B cannot be produced on demand — a post-flip
  regeneration would re-enter ~38% red with no recorded observation to
  adjudicate against. The flip waits for the WP-3 adjudication (not for the
  full successor-gate redesign).
- **E2 — `SCI-006` before `SCI-007` and before `SCI-005` stages 2–3.** The
  sign/axis-order ruling defines the frame in which the `-0.0576°` residual is
  measured and the receptor-basis semantics a non-scalar `E` must be expressed
  in. **Satisfied 2026-08-11:** WP-5 is independently accepted; WP-6 and the
  relevant WP-8 stages may now use the ruled east-X frame.
- **E3 — accepted `PERF-001` CPU legs before `SCI-005` Stage 1.** Both touch
  solver call sites around the contraction kernel. A WP-7 implementation or
  green test run is not enough: P-a through P-d must be independently accepted
  before any Stage-1 production edit. Drafting and reviewing the WP-8 design
  gate does not depend on that acceptance.
- **E4 — `SCI-004` last.** `Tier7JonesSciencePlan.md` §18.3 already enumerates
  its contract surface (Tier 4 time grid, Tier 5 correlation axis, Tier 6
  worker/fingerprint contracts, harmonic beam representations). Q5 is now
  resolved, so the design gate may be reviewed. Production M1 waits for
  accepted WP-7 CPU work; the output/characterization phase waits for accepted
  `SCI-005` Stage 2. `CI-001` itself is closed, while its successor-gate
  discipline still governs every new m-mode fingerprint.

**Standing rule:** one fingerprint-regeneration event per landing, each with a
scripted proof of what changed and what stayed bit-identical — the
`Tier7JonesSciencePlan.md` §39 row-1 discipline ("explained and localized …
never accepted by loosening the pin").

## 4. Work packages

| WP | Contents | Effort | Blocked on |
|----|----------|--------|------------|
| WP-1 | Quick wins: `API-002` fix, `API-001` fix | XS + XS | DONE 2026-08-11 |
| WP-2 | `CI-001` evidence: artifact harvest + comparator; fingerprint extension; optional nightly sampler | S–M | DONE 2026-08-08 |
| WP-3 | `CI-001` adjudication (Tier 8 §14 conditional) + successor-gate memo + mechanized verdict | M | DONE 2026-08-08 |
| WP-4 | `SCI-006` convention memo (the ruling) | S | DONE 2026-08-08 |
| WP-5 | `SCI-006` implementation (selected Branch A correction) | M | DONE — independently accepted 2026-08-11 |
| WP-6 | `SCI-007` reconciliation + closure as documented bound | S–M | DONE — independently accepted 2026-08-11 |
| WP-7 | `PERF-001` CPU legs P-a…P-d + docs notes; P-e GPU leg | M; P-e gated | P-e on Q4 (hardware) |
| WP-8 | `SCI-005` design gate, then separately accepted stages 1→3 | XL | Stage 1 after accepted WP-7 CPU scope (E3); stages 2–3 also require their accepted predecessor; WP-5/E2 is satisfied |
| WP-9 | `SCI-004` design gate and phase-separated successor | XL | Design review now; M1 on accepted WP-7 CPU, M3 on accepted WP-8 Stage 2 |

**Original 2026-08-05 sequencing:** WP-1, WP-2, WP-4, WP-7 (P-a…P-d),
WP-8's plan document, and WP-6's prediction computation could start in
parallel (both species pairings could be computed before the ruling; only the
closure waited). WP-1 has since completed as recorded in §§6–7 and §16.

---

## 5. WP-2/WP-3 — `CI-001`: the second digest class

### 5.1 Standing

Second byte-stable digest class on `linux-64-py311`. Red run `30726145633` at
`95a937e` fails five characterization pins; green run `30725507865` at
`47df8fc` has byte-identical `src/` and `tests/`. Class A's
`scientific_sha256` (`89f38f62…`) is byte-identical across three CPU models
and two vendors, so the axis is environmental, not code. Falsified with
evidence: CPU model; NumPy dispatched-feature set (an AMD run lacking
`AVX512FP16`/`AVX512_SPR` matches Intel runs that report them). Ruled out with
evidence: source regression, xdist, numpy/astropy/OpenBLAS version drift, IERS
auto-download, `PYTHONHASHSEED`, thread counts, uninitialized memory. The
remaining hypothesis space named at Tier 8: "hypervisor CPU-feature masking
and `libm`/OpenBLAS runtime dispatch, neither of which current instrumentation
captures" (`Fix.md` `CI-001` row).

Since Tier 8A the instrumentation is live: unconditional machine fingerprints
(CPU model, dispatched features, thread environment, BLAS build) on pass as
well as fail; pass-path reference cubes keyed by matched digest (64 MiB cap);
numeric deltas (`max|dV|`, max relative, differing count, first index) against
every captured reference on failure; and `.github/workflows/ci.yml:85-128`
restores the previous green run's `characterization-<cell>` artifact into each
new run (30-day retention). **The real class-A↔B delta has never been
measured**: the delta reporter has only ever seen its own synthetic 1-ULP
self-test (`max|dV| = 3.55e-15`, 1 of 363600 elements, Tier 8A record),
because no red run has yet coincided with staged reference cubes.

Two governing texts constrain everything here:

- "A set never grows to make a failure go away"
  (`test_tier6_current_behavior.py:271-273`); four prior reflex appends
  (`e3f1987`, `1c90d81`, `e5b20d1`, `0ce72e4`) are explicitly refused as
  precedent (`Tier8ReleasePlan.md` §14).
- The pre-authorized conditional, decided by measurement, verbatim: "If the
  numeric probe shows the divergence is at ULP scale — concretely, within the
  Section 13.5 backend tolerance the project already uses (`rtol=1e-12`) —
  then appending the observed digests **is** justified, the justification is
  'a second reproducible class at ULP scale, discriminator unidentified,
  numeric delta recorded here', and `CI-001` narrows to 'the discriminator is
  unnamed' rather than closing. If the delta is larger than that, nothing is
  appended, `CI-001` stays wide" (`Tier8ReleasePlan.md` §14; armed but never
  triggered in-tier, per §24).

### 5.2 WP-2 — evidence (starts now; no assertion changes)

1. **Harvest.** Pull the `characterization-linux-64-py311` artifacts for every
   CI run since Tier 8A (`gh run list` + `gh run download`; 30-day retention
   applies — harvest before it bites). Build a small committed comparator
   under `tools/` that, given a set of downloaded run artifacts: classifies
   runs by digest class, diffs machine-fingerprint fields between classes, and
   computes cube deltas wherever a red run and a reference cube coexist.
   Working data stays under gitignored `output/`; findings go in the WP report
   and, at adjudication, into `Fix.md`.
2. **Extend the fingerprint** — evidence-path only, the 8A precedent
   ("changes no assertion, no digest, and no test outcome"): add
   `platform.libc_ver()` and the glibc version (targets `libm`); the GitHub
   runner `ImageOS`/`ImageVersion` environment variables (the cheapest
   possible discriminator — if class membership tracks the runner image, the
   axis is the image's `libm` and the search ends); `numpy.show_runtime()`
   including the OpenBLAS runtime core name (the "OpenBLAS runtime dispatch"
   datum); `os.cpu_count()` and `sched_getaffinity`; and cache topology
   (`lscpu` on Linux, `sysctl` on macOS). Every field best-effort,
   cross-platform, exception-swallowed — a diagnostic that can fail a test is
   worse than no diagnostic. Rationale for the cache field: OpenBLAS blocking
   follows detected cache sizes, which vary across VM SKUs with identical CPU
   model strings — one of the few axes that crosses vendors while staying
   byte-stable per class.
3. **Optional sampler.** A separate, non-gating nightly workflow running only
   the characterization suite on `main`, to raise the observation rate above
   the push rate. Plus a `ci.yml` step that, on characterization failure,
   prints the field-level diff between this run's fingerprint and the restored
   previous-green fingerprint — automating the adjudication evidence.

### 5.3 WP-3 — adjudication and the successor gate

**Adjudication** executes the pre-authorized conditional the moment a red run
arrives with references staged: measured delta ≤ `rtol=1e-12` → append class B
with the measured delta and fingerprint diff as the reviewed evidence; `main`
returns green; the row narrows to "discriminator unnamed". Delta larger →
nothing is appended and the investigation changes character (byte-identical
sources across red/green already bound it to the environment, but a
larger-than-tolerance environmental difference is a different scientific
problem and goes back through a design gate).

**The successor-gate decision** (the register's named deferred question) gets
a short design-gate memo. Recommendation to bring to it:

- **Keep observation-set membership as the primary gate.** If the class delta
  is ULP-scale (expected), an `rtol=1e-12` primary gate would have passed both
  classes silently and the fleet heterogeneity would never have been observed
  — the bitwise scheme *worked*. It also retains 1-ULP regression sensitivity
  that a tolerance gate gives up permanently.
- **Mechanize the §14 criterion in the failure path**: on a novel digest,
  print the delta against every reference cube *and* an explicit
  ULP-scale-per-13.5 verdict, so a failing log carries complete adjudication
  evidence.
- **Define the switch trigger now**: a cell that accumulates ≥3 legitimate
  machine classes, or any class that proves non-byte-stable, converts — that
  cell only — to a reference-cube `rtol=1e-12` gate with the digest kept
  advisory (exactly the alternative the register names).
- The memo must explicitly address `Tier6HybridRuntimePlan.md` §42's
  declaration that the observation-set scheme is final: this recommendation
  keeps §42 substantively intact (the scheme is retained; only the failure
  path grows a verdict, and the conversion trigger is defined in advance
  rather than improvised under a red run).

**Touches:** `tests/characterization/test_tier6_current_behavior.py`
(fingerprint fields; failure-path verdict), `.github/workflows/ci.yml`,
`tools/` (comparator), `Fix.md` row at adjudication, one design memo.
**Closure requires:** discriminator named with evidence *or* class B appended
on measured ULP evidence; successor-gate decision recorded; the "blocks CI
green claims" clause lifted by an acceptance record.

### 5.4 WP-2/WP-3 acceptance (2026-08-08)

Accepted. Forced experiment run `31255085487` reproduced both byte-stable
classes on one Intel 8573C runner. NumPy AVX-512 dispatch moves the default
configuration cube; OpenBLAS `SkylakeX` moves the circular and heterogeneous
receptor cubes; forcing both axes to AVX2/Haswell restores the original class
byte-for-byte. The three maximum absolute deltas are `7.11e-15`, `1.57e-21`,
and `3.47e-18`, all within the full Section 13.5 predicate. The second class is
therefore recorded, the primary digest-membership gate is retained, the
failure-path verdict and passing-class manifest are mechanized, and the
predeclared three-class/non-byte-stable conversion trigger is adopted. See
`docs/development/ci001_adjudication.md` for exact commands, artifact
provenance, digests, limitations, and the closure record.

## 6. WP-1a — `API-002`: Rich markup eats bracketed text

**Original standing, 2026-08-05.** All four helpers at
`src/radiosim/utils/logging.py:113-130` interpolated the caller's message into
a markup-parsed f-string
(`console.print(f"[warning]⚠[/warning] {message}", highlight=False)`). Live
symptom: `Simulator.setup()`'s offline pre-flight
(`src/radiosim/api/simulator.py:782-785`) printed "Sky model(s)  require pygdsm
data but network is unavailable" with the bracketed model list eaten
(reachable via `configs/realistic_foreground_example.yaml` offline). A
call-site audit (2026-08-05) found **no caller that intentionally passes
markup** through the helpers; the only intentional markup in `src/` was
`cli/main.py`'s own Panels/Tables, which did not route through these helpers.
`RichHandler(markup=True)` at `logging.py:60` was the same latent bug class for
`logger.*` calls; the audit found no `logger` caller relying on it.

**Fix.** Wrap `message` in `rich.markup.escape()` inside all four helpers,
keeping the styled glyph prefix. Companion one-liner: `markup=False` on the
`RichHandler`. Docstring the helper contract: helpers print literal text.

**Tests (red first).** Capture console output (a `Console` writing to
`StringIO`, or `Console(record=True)`) and assert a bracketed model list
survives `print_warning`; siblings likewise; one message-construction case for
the `simulator.py` call site shape. No config surface, no fingerprint impact.
Changelog entry under the next unreleased section. Effort XS.

**Closure, 2026-08-11.** Implementation commit `62e53e6` escapes caller text
in all four helpers and configures `RichHandler(markup=False)`. Independent
acceptance at exact candidate `87bdaf2` reproduced the old swallowed-list
failure, proved that the real offline-preflight message retains its bracketed
model list, and audited every production helper and logger caller without
finding an intentional Rich-markup dependency. `API-002` is `DONE`.

## 7. WP-1b — `API-001`: `stokes_to_coherency` broadcasting

**Original standing, 2026-08-05.** Rows were assembled with `xp.stack`, which
required one shared shape, so `stokes_to_coherency(np.ones(5))` — the register
row's words: "the single most basic array-input call" — raised because the
scalar **defaults** could not join the stack. The 8B record had already fixed
the docstring truthfulness (`a3ef72d`); this row was the ergonomics gap it
disclosed. Production call
sites (`core/visibility.py:754`, `core/visibility_healpix.py:574`) always
passed four matched-shape arrays and were unaffected either way.

**Decision (resolved by program adoption, 2026-08-05 — gated question Q2):
implement broadcasting**, not documented strictness. One
`xp.broadcast_arrays(...)` after the `asarray` block
(`core/polarization.py:137-141`; present in `numpy`, `jax.numpy`, and
`dask.array`). Genuinely incompatible shapes still raise NumPy's own error —
the strictness that matters survives.

**Bit-identity.** Broadcasting already-equal shapes is an identity view;
every previously valid input takes the identical arithmetic path; pins
untouched. State this in the commit message.

**Tests (red first).** Array-`I` with untouched defaults equals the
explicit-zero-arrays construction; scalar `Q` against array `I` equals a
per-element loop built in the test body (Tier-1 style); a non-broadcastable
pair still raises `ValueError`; dtype preservation at both float widths;
docstring Broadcasting section rewritten and its doctest flipped
(`pixi run doctest` gates it). Effort XS.

**Closure, 2026-08-11.** Implementation commit `3ac6282` inserts the adopted
`xp.broadcast_arrays` step without changing the subsequent arithmetic; test
commit `87bdaf2` adds the missing cross-backend analytic and unchanged-path
proofs. Independent acceptance reproduced the pre-fix failure and then passed
NumPy, JAX-CPU, and Dask mixed-rank cases, incompatible-shape rejection, and
float32/float64 equal-shape byte identity in all six backend/dtype cells.
`API-001` is `DONE`.

## 8. WP-4/WP-5 — `SCI-006`: the Stokes-`Q` sign ruling

### 8.1 Standing

At 7J, mappings 1 (fringe sign) and 2 (coherency Stokes-`V` sign) were
"independently confirmed derived, character for character, against both
codes' own installed source" (`Tier7JonesSciencePlan.md` §1). Mapping 3 — the
**local-basis axis order** — is the open one: for the same sky, the same
BeamFITS `x_orientation="east"` feed, and the same mount, RadioSim's local `Q`
has the opposite sign to `pyuvsim`'s (the swap flips `V` too;
`pyuvsim` binds feed 0 to `data_array[0, 0]`). Recorded agreements
(`output/crossvalidation/2026-08-02-pyuvsim-1.4.0.json`, RadioSim 0.2.0 vs
`pyuvsim` 1.4.0): unpolarized `2.8e-14` relative; polarized total intensity
`2.3e-10`; circular `4.1e-11`; linear after the swap `2.06e-3` (control
without the swap: `0.616`); fitted linear ratio modulus `0.999978`. The 7J
record is explicit: "a characterization, not an endorsement of either sign."
Distinct and *not* in question: the deliberate HBS 1996 / Smirnov 2011
coherency `V`-sign convention documented in `core/polarization.py`'s module
docstring (mapping 2, confirmed).

### 8.2 WP-4 — the ruling memo (S, starts now)

Derive the closed form: for a +`Q` (IAU, north-referenced) source at zero
parallactic angle observed with an `x_orientation="east"` linear feed, what
sign must `XX − YY` carry? Sources to cite: the IAU/IEEE definitions as
interpreted by Hamaker & Bregman 1996 (Understanding radio polarimetry III);
`pyuvdata`'s x-orientation and polarization conventions (the
`simulators/pyuvdata` reference checkout is initialized locally — cite their
docs and source directly); AIPS Memo 117. Then evaluate both codes against the
closed form: RadioSim's side is the local sky-basis axis order behind
`J[feed, sky_basis]` (`core/polarization_basis.py` and the receptor binding in
`core/receptor.py`); `pyuvsim`'s side is the feed-0/`data_array[0,0]` binding.
The memo rules which code is normative. House rule satisfied: a scientific
claim with citations, closing exactly the ruling 7J declined to make.

**Outcome — DONE 2026-08-08.**
`docs/development/sci006_polarization_convention.md` derives the result from
the IAU north-through-east position-angle definition, Hamaker & Bregman's
north/east sky axes, pyuvdata's east-X feed angles, and the RIME. For an IAU
`+Q` source, the north component carries `(I+Q)/2` while the east component
carries `(I-Q)/2`; therefore an east-oriented X feed and north-oriented Y feed
must report `XX - YY = -Q`. RadioSim currently reports `+Q` because
`M(linear)=I2` binds the north-first brightness axis directly to the
east-labelled X row. An independent pyuvsim 1.4.0 source trace and executable
unit-BeamFITS probe produce the required permutation and `-Q`. This selects
Branch A. WP-4 changed no runtime signs, feed order, fingerprints, or retained
cross-validation artifact, so `SCI-006` remains open pending WP-5 acceptance.

### 8.3 WP-5 — implementation (two branches; gated on E1 and Q3)

**Branch A — selected by WP-4; RadioSim is non-normative:** flip. RadioSim
exports `UVData`/UVFITS/MS, so its correlation labels must mean what the
ecosystem that reads those files expects, and `pyuvdata` is that ecosystem's
convention-setter. Touches: the canonical table in
`core/polarization_basis.py` (the single owner — the correct property of the
Tier 5 design); the crossval harness drops its compensating axis swap; the
`crossval` environment re-runs and commits a **new dated artifact** (updating
`test_the_committed_artifact_describes_this_comparison`; Tier-2 remains
non-gating per `Tier7JonesSciencePlan.md` §29); a Tier-1 analytic-invariant
unit test evaluates the memo's closed form in the test body; polarized
characterization pins regenerate across all six cells **with a scripted proof**
that unpolarized workloads are bit-identical and polarized workloads change
exactly in the `Q`/`V`-derived entries; changelog + migration-guide entries
(breaking is fine pre-v1; the ruling covers `V` since the axis swap flips
both).

**Branch B — RadioSim is normative:** document-intentional. The convention
statement (with the citation) lands in the polarization docs; the harness
keeps its swap with the ruling attached; the same analytic test asserts *our*
sign; no pin churn.

Branch B is retained as the rejected alternative for auditability. The memo's
WP-5 contract supersedes this sketch where it is more precise: the one-owner
mapping must preserve ideal circular-output semantics, explicitly audit
feed-asymmetric Jones terms, and compare the deliberate RadioSim/pyradiosky
Stokes-`V` convention difference separately after removing the Q-axis
compensation.

Effort S + M. Landing waits for WP-3 adjudication (edge E1) in Branch A only.

### 8.4 WP-5 — independent acceptance (DONE 2026-08-11)

WP-5 is accepted at exact candidate
`f5fa101e4ac345534636380720ce33ec93a31eae`. A separate read-only reviewer
re-derived the north/east-to-east-X mapping and the linear and circular product
tables, inspected production paths rather than trusting the implementation
notes, reran the optional cross-validation suite, and authenticated the remote
evidence.

GitHub Actions run `31434253575` is a push-triggered `CI` run whose head SHA is
exactly the accepted candidate. Its quality job `93604879477`, backend-parity
job `93604879424`, and all six compatibility jobs completed successfully. The
six characterization artifacts are `9080470037` (linux py311), `9080472471`
(linux py312), `9080524924` (arm64 py311), `9080529903` (arm64 py312),
`9080665431` (Intel macOS py311), and `9081128353` (Intel macOS py312). Every
artifact reports the same workflow-run head SHA; its archive digest and retained
cube contents were independently checked.

Across all cells, the four applicable polarized workloads are byte-exact
`P V_old P^H`; the two unpolarized workloads and both shipped-configuration raw
cubes remain active unchanged references. The feed-asymmetric witness is
validated separately because native-feed terms need not commute with `P`.
Linux py311 and py312 retain both legitimate heterogeneous-receptor dispatch
classes, including the candidates observed under AVX-512/OpenBLAS runtime
dispatch. The original five CI-001 digests remain durable provenance. Four
unchanged shipped-configuration digests also remain active pins; only the
superseded heterogeneous `c7b51d02...` digest is historical-only after its
accepted `9f07661c...` east-X successor was recorded. Observation-set
membership and the Section 13.5
`rtol=1e-12`/scaled-absolute adjudication rule are unchanged.

Fresh exact-candidate checks included 432 focused production/science/output
tests, 88 NumPy/JAX/Dask backend-parity tests, all five optional cross-validation
tests, and the full non-slow, lint, format, typecheck, doctest, and strict Sphinx
gates. Direct `Q+iU` cross-validation records residual
`0.002052050642874229`, fitted rotation `+0.057991427331288835` degrees, ratio
modulus `1.0000830200328927`, and explicit pyradiosky V-mapped residual
`4.0701816228520426e-11`. Those measurements define the still-open SCI-007
input; they are not absorbed into SCI-006.

## 9. WP-6 — `SCI-007`: the refitted `+0.057991°` frame rotation

**Historical design-stage record (2026-08-11).** The
normative design is
`docs/development/sci007_frame_accuracy_bound.md`. It preserves the accepted
WP-5 input: direct `Q+iU` residual `2.052050642874229e-3`, fitted global angle
`+0.057991427331288835°`, ratio modulus `1.0000830200328927`, and explicitly
mapped V residual `4.0701816228520426e-11`.

The design identifies the dominant mechanism as missing transport of the ICRS
catalogue polarization tangent basis into RadioSim's operational apparent
basis. That operational frame is the ideal spherical inverse of topocentric
`AltAz`, using geodetic latitude and local apparent sidereal time. It is
TETE-like/apparent-of-date, not an exact Astropy `TETE` transform. Polar motion,
diurnal aberration, and other topocentric/Earth-orientation details are smaller
remainder terms; refraction is disabled with `pressure=0`.

The sign and granularity are fixed. With
`R(a)=[[cos(a),sin(a)],[-sin(a),cos(a)]]`, RadioSim
`J_RS=S R(psi_RS)`, and pyradiosky `B_local=K.T B_ICRS K`, define
`Delta=wrap_pi(psi_RS+atan2(K[0,1],K[0,0]))`. After the existing
fringe-Hermitian mapping,
`L_RS=exp(+2j*Delta)L_PY` for `L=Q+iU`; RadioSim moves to the pyuvsim
convention with `exp(-2j*Delta)`. The correction is per source and time before
summation. A single global fitted angle is retained only as a failing control.

For the retained three-source, three-time HERA-site fixture, the exact pinned
pyradiosky prediction spans `0.0429704°` to `0.0645015°`; the independent
public Astropy source-to-zenith oracle spans `7.64484e-4` to `1.12004e-3` rad.
The public and exact grids agree within about `1.96%`, and the exact
source-time correction leaves `2.400855498837282e-10` relative linear
residual. The normal gate will enforce the fixture-scoped non-vacuous bounds
`6e-4 < min(abs(Delta))`, `max(abs(Delta)) < 1.2e-3 rad`, and spin-2 effect
`<2.4e-3`; these are not all-sky guarantees.

All transforms and apparent-LST calls must install Astropy's explicit bundled
`IERS_A_FILE` table with downloads disabled. The normal Python 3.11/3.12 test
does not hard-code one digest because the locked environments may bundle
different valid tables; the optional artifact records its exact digest and
per-time EOP values. The `0.041`–`0.063°` CIRS probe is scientifically
consistent and superseded by executable grids. The old `0.200°` scalar is
unreproduced historical evidence and is neither a bound nor a denominator.

Production remains unchanged. The current `PrecisionConfig.ultra()` changes
numeric precision, not frame transport, so it is not described as an existing
solution. A future transport policy needs its own design.

The evidence slice is red-first: retain the raw `<5e-10` failure; add the
bundled-IERS public test in default and `py312`; add the pinned source-time
optional comparison; add a deep validator while the versioned 1.2.0 artifact is
absent; generate that artifact only from an explicitly approved clean source
commit and add it in an evidence-successor commit; run all gates; then obtain
separate read-only authentication of the generating source, successor diff, and
exact acceptance SHA before touching the register. `SCI-007` therefore
remained **OPEN at the design stage**.

**Historical design-slice writable authority.** This design step was
restricted to
`docs/development/sci007_frame_accuracy_bound.md`, `docs/index.rst`, and this
live WP-6 status. It does not authorize or make changes to `Fix.md`, `src/`,
tests, outputs, user documentation, changelog, configuration, workflows, or
locks. The design-stage estimate was S–M for evidence and acceptance.

### 9.1 WP-6 independent acceptance (DONE 2026-08-11)

WP-6 closed `SCI-007` at exact evidence successor
`e20f636788e0b61ae6c854f64cbb7476c3cb9a50`, generated from clean source
`9b50805cf9fe32124800d1e3946a87e3911c376b`. The source contains no artifact;
the direct successor adds only the 33,173-byte schema-1.2.0 record, README
reproduction instructions, and exact digest constants. The retained artifact
is `output/crossvalidation/2026-08-11-pyuvsim-1.4.0-sci007.json`, SHA-256
`3a441ad606f365ac4110e30d9d8c2f3d7f5ea91c481aa70488dea72487e570ba`.
The generator, lockfile, and bundled IERS-A SHA-256 values are
`405a5d9fbee3becb1724d79f173e056e9e5da73cc73e3bed2cc2482d1b346c94`,
`37db432e6ade2dd3e64222d5ccfe532be5671893b24ce29e717a3bbb12f38ade`,
and `ff2d22108e982bd86e326e01d797fa8bd545d51483359dd98e6c08fa5737f667`.

Independent re-derivation confirmed
`Delta=wrap_pi(psi_RS+atan2(K[0,1],K[0,0]))` and the
RadioSim-to-pyuvsim correction `exp(-2j*Delta)` per source/time before
summation. The raw relative linear residual `2.052050642874229e-3` falls to
`2.400855498837282e-10`; the wrong sign gives
`4.103897953509379e-3`. The public angle range is
`7.644842652547723e-4` to `1.1200433324138892e-3` rad, the maximum spin-2
effect is `2.2400861964641163e-3`, public/exact disagreement is
`0.019580918743243865`, and the global-angle control remains
`1.9606576512107846e-4`.

Both validators and the public bound passed in Python 3.11 and 3.12; the
documentation/evidence suite passed 195 tests in both environments; and the
optional cross-validation passed five tests. Exact-SHA CI run `31461141190`
then passed all eight jobs at the accepted successor: backend parity passed 92
tests, every compatibility cell passed 5,475 tests with one skip, and all six
artifact ZIPs were authenticated. All 84 hexadecimal manifest observations
were active pins and no candidate cube was retained.

Closure is deliberately narrow. It documents the retained HERA
three-source/three-time fixture; it is not all-sky or cross-platform frame
validation. Production frame policy, source code, dependencies, the lockfile,
fingerprints, and tolerances are unchanged. `PrecisionConfig.ultra()` remains
a numerical-precision preset and does not perform tangent-basis transport. The
artifact's internal OPEN status is immutable generation-time provenance; this
later independent record owns the DONE transition.

## 10. WP-7 — `PERF-001`: split into five legs

**Design gate recorded 2026-08-11; P-a through P-d implementation and P-e
readiness landed; retained CPU evidence and independent acceptance pending.**
The normative design is
`docs/development/perf001_runtime_mitigations.md`. `PERF-001` remains
**ROADMAP** until real accelerator evidence is accepted or its scope is
formally re-adjudicated.

Backend correctness parity is complete (Dask bit-identical; JAX-CPU within
`rtol=1e-12`). The row's substance, against the committed records
(`output/benchmarks/reference/20260731T104303Z-darwin-arm64.json`: JAX-CPU
3–18× slower steady-state per workload; `max_first_to_repeat_ratio` 493.9;
~208 B per `(baseline, source)` pair):

- **P-a — baseline-axis contraction chunking:** keep the one-parameter public
  factory, six-input compiled leaf, source summation order, and single compile
  site. Target `131072` baseline-source pairs per leaf and split only `B`.
  This bounds source-dependent contraction-leaf working temporaries, not the
  already-materialized `J_p`, `J_q`, phase, or envelope inputs, nor the
  unavoidable baseline-dependent output/assembly storage. When `S > 131072`,
  one baseline is the irreducible exception. Evidence uses frozen v1 records
  plus matched v2 unbounded/production rows. Effort M.
- **P-b — retrace amortization:** for compiling backends only, pad the visible
  source axis on the host to the next power of two after horizon selection but
  before backend conversion, `DirectionBatch`, Jones, phase, and morphology.
  Append a finite copied direction and exact-zero signal. NumPy/Dask stay
  unpadded and byte-identical; JAX retains its existing tolerance. Late JAX
  padding is rejected because its own primitives compile per logical shape.
  Evidence distinguishes logical counts, buckets, and complete leaf shapes.
  Effort S–M.
- **P-c — deterministic automatic selection:** `get_backend("auto")` returns
  precision-compatible NumPy without importing or probing JAX. Explicit
  `list_backends()`/`get_backend_info()` own discovery. Plain `jax` uses its
  runtime-default device; explicit GPU/TPU requests are strict and never fall
  back to CPU. Generic device-resource discovery also stops importing JAX.
  Cold-path timing is recorded but never gated. Effort XS–S.
- **P-d — flip `VisibilitySimulator.supports_gpu`'s ABC default to `False`**
  — the behavior change 8E explicitly parked with `PERF-001`
  (`Fix.md` ~15316-15334). XS.
- **P-e — GPU-ready infrastructure now; evidence hardware-blocked:** add a
  Linux-only, optional, non-gating `jax-gpu` Pixi feature carrying pinned
  `jax[cuda13]`, a strict preflight, and the unchanged full benchmark matrix.
  It uses a separate solve group because the default group pins `jaxlib` to
  `cpu*`; CPU environment package identities must remain unchanged. No workflow
  or public GPU extra is added. Only a real clean-SHA GPU record can support
  acceleration or closure. Until an authorized compatible host is available,
  `supports_gpu` remains false and the row stays ROADMAP.

Fold in the `Tier6HybridRuntimePlan.md` §39 methodology note (tracemalloc
`peak_host_bytes` under-represents JAX's native allocations: `6,115,963` vs
`1,253,968` bytes for the same workload) as a docs line. Freeze the v1 schemas
and retain a separate strict PERF-001 document with memory, solver-memory,
retracing, and backend-resolution records. P-a through P-d are landed with
exact-SHA CI green, but do not satisfy WP-8 edge E3 until the retained CPU
record and whole-slice independent acceptance exist; they do not close P-e.

## 11. WP-8 — `SCI-005`: beam physics beyond scalar `E`

The proposed dedicated gate is
`docs/development/sci005_beam_physics_plan.md`. It was first introduced at
`42a1f27`; its Stage-1 numerical correction is independently approved, while
the acceptance-succession amendment is under fresh independent review and has
not yet become `D1`. Neither record provides implementation or stage-acceptance
evidence. `SCI-005` remains **ROADMAP**.
Near-field simulation remains a permanent non-goal. Station element beams,
array factors, and mutual coupling remain future work outside this closure.

The candidate resolves the three stages as follows:

1. **Scalar aperture physics.** Central/support blockage and deterministic real
   unit-RMS `(n,m)` Zernike surface-height modes compose inside one aperture
   transform normalized to the unmodified ideal aperture. The phase is
   `exp(-i 4*pi*h/lambda)`. The result is not re-peak-normalized. Ruze
   `sigma` and a correlation length do not define a deterministic complex
   voltage: Stage 1 retains coherent Ruze loss in `E` and permits only a
   fully declared covariance-based ensemble-power/autocorrelation diagnostic
   for the scattered error beam. An invented `sqrt(power)` voltage is
   forbidden.
2. **Beam squint.** The two displaced responses form a diagonal `D_b` only in
   native-feed space. Within RadioSim's existing `C E P` factorization the
   correct sky-side matrix is `E=C^dagger D_b C`, which is generally full. The
   exact Cotton/Uson arcsine frequency law is used. Squint is the declared
   `+pi/2` orthogonal to the mechanical off-axis-feed ray; that mechanical
   position angle is distinct from electrical receptor `feed_rotation_deg`,
   and the non-scalar test must prove order matters.
3. **Full cross-polarization.** UVBeam efield ingestion receives one explicit
   matrix-level peak-normalization contract, basis-vector and Ludwig-3
   conversion, `E=C^dagger J_native` receptor factorization, point/HEALPix and
   NumPy/JAX/Dask coverage, output-format behavior for HDF5/UVFITS/MS, and a new
   dated non-gating pyuvsim comparison. Quadrupolar response is an analytic
   oracle rather than an underspecified production model; IXR is a derived
   diagnostic, not another leakage config.

All three stages keep `BeamSystem` as the one beam owner and deliver fully
composed `(B,S,2,2)` Jones matrices to the existing six-input contraction. No
compiled-kernel signature or second `backend.compile` site is allowed. Each
stage has red tests, an exact writable list, a clean-source retained evidence
successor, independent acceptance, and enabled-effect-only fingerprint
regeneration. Stage 1 remains blocked until P-a through P-d are independently
accepted (E3). Stages 2 and 3 then proceed only after the preceding stage is
accepted; WP-5/E2 is already satisfied. The register closes only after a
separate whole-row review accepts all three stages.

## 12. WP-9 — `SCI-004`: the m-mode simulator

A **feature, not remediation** — `Tier7JonesSciencePlan.md` §18.3's words: "a
second complete forward model". Q5 is resolved by the adopted bounded driver:
a HERA-like fixed-zenith drift survey requiring repeated full-sidereal
visibility evaluation, direct-RIME agreement on small polarized skies, and
controlled harmonic truncation error.

The design candidate is
`docs/development/sci004_mmode_design.md`. It corrects an important live-code
assumption: the current simulator registry selects only the point-source
kernel, while `Simulator.run()` always enters `core.hybrid.solve_sky()` and the
HEALPix branch is hard-coded direct. M1 therefore first lifts the registry to a
whole-`SkyModel` request/outcome boundary; `rime` wraps the maintained direct
point/HEALPix/hybrid path and `mmode` becomes a true second strategy.

The candidate freezes a complete uniform `2*pi` ERA grid mapped explicitly to
UTC/UT1, a frozen-CIRS rigid-ERA operational frame with an SCI-007-linked error
budget, IAU North/East tangent metadata, the exact RadioSim-to-Shaw
polarization/Stokes-V bridge, scalar and spin-2 harmonic conventions,
per-antenna and baseline/frequency `B_lm` construction, signed-m DFT
normalization, alias/truncation/quadrature rules, NumPy/JAX/Dask and memory
policy, result/output provenance, fingerprint/CI-001 handling, strict evidence
schemas, and red/source/evidence/acceptance succession.

The public selector remains `execution.simulator`; the removed
`visibility.calculation_type` does not return. A new full-sidereal
`obs_time` variant is necessary because the current uniform-UTC duration and
cadence cannot truthfully stand in for the ERA group coordinate. Design
approval licenses no production claim. M1 waits for accepted WP-7 CPU work;
M3 and whole-row closure require accepted SCI-005 Stage 2. `SCI-004` remains
ROADMAP throughout the phase acceptances and closes only after the final
whole-row review.

## 13. Dispositions argued (close-as-documented and anti-candidates)

1. **`SCI-007` — closed as documented-and-bounded** (WP-6 accepted
   2026-08-11): the retained fixture's milli-radian tangent-basis transport is
   attributed and executable. The close carries a fixture-scoped bound, not a
   production upgrade path; `.ultra()` changes numeric precision only and does
   not implement frame transport.
2. **`API-001` — defensible won't-fix, but the fix is adopted** (Q2): the
   signature's own defaults raise on array input; one `broadcast_arrays` line
   with a bit-identity argument is cheaper than maintaining the justification.
3. **`PERF-001`'s implicit "JAX-CPU should beat NumPy" — close explicitly as a
   non-goal** (one sentence in `docs/user_guide/backends.rst`): JAX's role is
   the compilation-capable backend for future accelerators; NumPy is the CPU
   reference; the committed records already say JAX-CPU is slower and that is
   acceptable. The row itself stays open for the memory/retrace legs and the
   hardware-gated GPU leg.
4. **`SCI-004` — now scheduled by a named need, but not closable by design.**
   Q5 authorizes the reviewed design/phase succession; ROADMAP remains the
   truthful state until the complete simulator and validation programme are
   independently accepted.
5. **Anti-candidates.** `CI-001` cannot be closed as "documented flakiness" —
   its row blocks CI-green claims and the discipline forbids reflexive appends.
   `SCI-006` must be *ruled*, not documented around — it decides what
   RadioSim's exported correlation products mean to every downstream reader.

## 14. Program mechanics

Each work package runs in the established style:

- **Design-gate memo before implementation** for anything substantial: the
  WP-3 successor gate, WP-4's ruling, WP-8's plan document and per-stage
  memos, WP-9. The WP-1 fixes and WP-2 evidence steps sit below the
  substantiality bar and proceed directly (with red tests).
- **Tests first, red evidence recorded** before the fix lands.
- **Bit-identity proofs** for anything claiming refactor-neutrality (WP-7 P-a
  and P-b; WP-1b's unchanged-path argument), scripted, not asserted in prose.
- **One fingerprint-regeneration event per landing**, each with the
  changed/unchanged proof (§3, standing rule). On `linux-64-py311`, any
  regeneration must characterize both machine classes (edge E1).
- **Verbatim typed rejection messages** for any config-surface change, and the
  identity-block rejection convention carries to new `beams:`/`jones:` blocks.
- **Scientific claims carry citations and analytic-invariant tests**;
  cross-validation compares values, never APIs; the Tier-2 comparison never
  gates; the phrasing rule stands — "compared against, with the following
  measured agreements and the following open disagreements", never "validated
  against" (`Tier7JonesSciencePlan.md` §29.2).
- **No speed or GPU claim without a committed benchmark record**; benchmark
  and crossval jobs never gate.
- **Register and changelog**: rows change only through acceptance records with
  exact closure text; every user-visible change lands in `docs/changelog.rst`
  (and `docs/migration_guide.md` for breaking changes). The `[0.3.0]` Known
  limitations section is historical and is not edited; fixes get entries in
  the next release's section.
- **Commits**: small, narrowly scoped, conventional format, no co-author
  lines. Local commits after each verified coherent step; nothing is pushed
  or published without explicit approval.
- **Implementation and acceptance are separate passes**: acceptance re-derives
  claims from current source and fresh probes, never from the implementing
  slice's own claims, and records closure text per row.

## 15. Gated questions

- **Q1 (WP-3 successor gate):** *resolved 2026-08-08 by the WP-3 acceptance* —
  observation-set membership stays primary; the full Section 13.5 verdict is
  mechanized into the failure path; a per-cell conversion to a reference-cube
  gate triggers only at ≥3 legitimate classes or a non-byte-stable class.
- **Q2 (API-001 disposition):** *resolved 2026-08-05 by program adoption* —
  implement broadcasting (see §7).
- **Q3 (WP-5 decision criterion):** *resolved 2026-08-08 by the WP-4 ruling* —
  exported-product semantics follow the IAU definitions as instantiated by
  pyuvdata's declared feed angles. The memo rules RadioSim non-normative, so
  Branch A is selected with its pin regeneration and migration entry.
- **Q4 (blocks WP-7 P-e evidence only):** name the GPU hardware and access path
  (cloud runner or workstation). Until answered, GPU-ready infrastructure may
  land, but no accelerator measurement, evidence, or claim may be accepted.
- **Q5 (WP-9 science driver):** *resolved 2026-08-11 by program adoption* — a
  HERA-like fixed-zenith drift-scan survey requiring repeated full-sidereal
  visibility evaluation, direct-RIME agreement on small polarized skies, and
  controlled spherical-harmonic truncation error. This schedules the design
  gate; it does not accept a production phase or close `SCI-004`.

## 16. Status ledger

| WP | State |
|----|--------------------|
| WP-1 | DONE — independently accepted 2026-08-11; API-001 and API-002 closed |
| WP-2 | DONE — accepted 2026-08-08 |
| WP-3 | DONE — accepted 2026-08-08; CI-001 closed |
| WP-4 | DONE — ruled 2026-08-08; Branch A selected; no runtime change |
| WP-5 | DONE — independently accepted 2026-08-11; SCI-006 closed |
| WP-6 | DONE — independently accepted 2026-08-11; SCI-007 closed as a retained-fixture accuracy bound |
| WP-7 | CPU ACCEPTED; P-e hardware-gated. PERF-001 remains ROADMAP; supports_gpu remains false; no accelerator evidence or claim is accepted. |
| WP-8 | DONE — Stages 1, 2, and 3 ACCEPTED. Stage 1 2026-08-18: D1 `c6a5ce90` -> R1 `e246c5d` -> S1 `881b1a9` -> E1 `bbc2b1b` -> A1 `2281f2f`. Stage 2 (beam squint) 2026-08-19: operative D2 `b6d09b7` (via the two accepted bounded corrections `3d60b6f` and `0c37815`) -> R2 `da18f96` (re-cut superseding `2a5d5aa`) -> S2 `5c94d92` -> E2 `56f7fd5` -> A2 `7523706` (independent reviewer verdict ACCEPT; retained evidence and acceptance certificates in docs/development/; the Stage-2 certificate is the WP-9 M3 export). Stage 3 (full-efield Jones response) 2026-08-20: operative D3 `ef972af` (via the six accepted bounded corrections and the reopened slices the plan's own header records) -> R3 `e22c917` -> S3 `0b5d0da` -> E3 `ec28836` -> A3 `ac269cdd7269da359cf15eeb99930f232b3295e4` (independent reviewer verdict ACCEPT; retained evidence and acceptance certificates in docs/development/). Whole-row closure successor C accepted 2026-08-20 (separate independent review per the plan's §9); `SCI-005` closed **DONE** in `Fix.md` |
| WP-9 | Design gate ACCEPTED 2026-08-21: candidate `978fef6` received its two required fresh independent Phase-0 reviews (physics/governance + computational, staleness mandate; both REJECT with bounded findings), one combined bounded correction was dual-ACCEPTed on pinned bytes/diff (recorded in the design header) and landed 2026-08-21; dated bounded corrections follow the design's §13.7 with supersession citations recorded in its header (R1-authoring reconciliation and S1 feasibility reconciliation, dual-ACCEPTed 2026-08-22; the two-tier acceptance gate, ruled by the program owner, the tier-1 horizon-free shell, the ablation clarification closing every deferred advisory, the evidence-generation reconciliation, the post-source record retention, the guard-interval/independent-membership round, the guard-rows-in-the-retained-projection round, the post-acceptance-repairs round starring the A1->R2 edge, the celestial-tangent-transport round reopening the phase-2 red slice, dual-ACCEPTed 2026-08-23, the resolved-input tangent-frame route round, dual-ACCEPTed 2026-08-23, the direct-RIME-basis round discharging the phase-2 reopening, dual-ACCEPTed 2026-08-23, the singular-capability-pin round, dual-ACCEPTed 2026-08-24, the description-follows-capability round, dual-ACCEPTed 2026-08-24, the un-ignoring-the-granted-reference-records round, dual-ACCEPTed 2026-08-24, the accepted-capability-characterization-envelope round, dual-ACCEPTed 2026-08-24, the performance-product-follows-the-envelope round, dual-ACCEPTed 2026-08-24, the retained-evidence-surfaces round, dual-ACCEPTed 2026-08-24, the honest-backend-axis round, dual-ACCEPTed 2026-08-24, and the scalar-table-kernel-exception round, dual-ACCEPTed 2026-08-24), the latest landing holding the operative `D`; the M1 red slice landed 2026-08-22 with governed re-cuts per the header's reopened-slice records; phase M1 (the scalar `mmode` simulator registry entry) ACCEPTED 2026-08-23 by independent review through the frozen-descendant WP-7 replay gate (succession D 1712575 -> R1 8b9d89e -> S1 8dfc9af -> E1 dc736c6 -> A1; acceptance record docs/development/sci004_mmode_phase1_acceptance.json, all eleven Section 14.3 oracles re-derived including the two replay-deferral discharges); phase M2 (full-Stokes m-mode) ACCEPTED 2026-08-24 by independent review (succession D b9a9d7a8 -> R2 27d2ba45 (live, starred edges per the header) -> S2 39924579 -> E2 50772ec1 -> A2; acceptance record docs/development/sci004_mmode_phase2_acceptance.json, all ten Section 14.3 oracles re-derived); M3 is next, gated on the accepted SCI-005 Stage-2 certificate; SCI-004 remains ROADMAP |
