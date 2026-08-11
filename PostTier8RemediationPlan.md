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
| `SCI-007` | OPEN | Direct-Q/U cross-validation on the accepted east-X frame refits `+0.057991°` (linear residual `2.052e-3`); the frame-species probes remain unreconciled and WP-6 still owes a design-gated executable bound, retained provenance, and independent acceptance |
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
| 5 | `SCI-007` | Reconcile numerically; close as a documented, test-pinned accuracy bound | S–M | Ready; WP-5/E2 satisfied |
| 6 | `PERF-001` | Four CPU legs now with bit-identity proofs; GPU leg hardware-gated | M + gated | CPU legs now (WP-7) |
| 7 | `SCI-005` | Staged successor tier, own plan document; scalar-preserving items first | XL | Plan doc now; stages gated (WP-8) |
| 8 | `SCI-004` | Feature, not remediation; design gate only when a science driver exists | XL | Unscheduled (WP-9) |

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
- **E3 — `PERF-001` CPU legs before `SCI-005` implementation.** Both touch the
  solver call sites around the contraction kernel; the PERF legs are small and
  provably bit-identical, so they land clean under the existing pins first.
- **E4 — `SCI-004` last.** `Tier7JonesSciencePlan.md` §18.3 already enumerates
  its contract surface (Tier 4 time grid, Tier 5 correlation axis, Tier 6
  worker/fingerprint contracts, harmonic beam representations); it wants at
  least `SCI-005` stage 2 and the `CI-001` successor gate settled first.

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
| WP-6 | `SCI-007` reconciliation + closure as documented bound | S–M | Ready; WP-5/E2 satisfied |
| WP-7 | `PERF-001` CPU legs P-a…P-d + docs notes; P-e GPU leg | M; P-e gated | P-e on Q4 (hardware) |
| WP-8 | `SCI-005` plan document, then staged slices 1→3 | XL | Stage 1 after WP-7 (E3); stages 2–3 after WP-5 (E2) |
| WP-9 | `SCI-004` design gate | XL | Q5 (science driver); soft on WP-8, WP-3 |

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

**Post-WP-5 input, not WP-6 acceptance.** The accepted direct comparison fits
`+0.0579914273313°` with linear residual `2.052050642874e-3` and ratio modulus
`1.000083020033`; the explicit pyradiosky V-sign mapping is already at the
`4.070e-11` relative floor. The old `-0.057568764952°` fit belongs to the
pre-SCI-006 compensated comparison. The apparent-equatorial and CIRS
frame-species probes remain unreconciled, so these accepted WP-5 measurements
define WP-6's input rather than its mechanism or conclusion.

WP-6 must establish its own design-gate memo, pin the exact frame species and
Earth-orientation policy, add a non-vacuous normal-environment accuracy bound,
retain a fully provenanced optional cross-validation artifact, and obtain
independent acceptance before SCI-007 closes.

**Plan.** After the WP-5 ruling: recompute the fitted residual in the ruled
frame; then compute a **per-source, per-time prediction** of the frame
difference between RadioSim's idealized apparent-frame inversion and
`pyuvsim`'s chain with astropy, for the exact crossval geometry — resolving
which species pairing each of the two prior probes measured. Success
criterion: the prediction matches the fitted rotation per-source within ~10%,
and removing it drops the linear residual to the `1e-10` order.

**Closure (recommended): documented-and-bounded.** A docs statement naming the
frame species and the neglected terms (polar motion, diurnal aberration) with
an apparent-place citation (SOFA/USNO), plus an **executable probe test**
pinning the bound — so the bound is a test, not prose. `PrecisionConfig`'s
`.ultra()` preset is the named future home for full apparent-place frames if
a use case ever needs them; not now (a host-side per-direction astropy cost
against `PERF-001`, and fingerprint churn, for a milli-degree effect). If the
reconciliation fails to match, the row stays honestly open with the failed
reconciliation recorded. Effort S–M.

## 10. WP-7 — `PERF-001`: split into five legs

Backend correctness parity is complete (Dask bit-identical; JAX-CPU within
`rtol=1e-12`). The row's substance, against the committed records
(`output/benchmarks/reference/20260731T104303Z-darwin-arm64.json`: JAX-CPU
3–18× slower steady-state per workload; `max_first_to_repeat_ratio` 493.9;
~208 B per `(baseline, source)` pair):

- **P-a — chunk the baseline axis inside `baseline_contraction_for`** — the
  mitigation `Tier6HybridRuntimePlan.md` §39 names verbatim, in that exact
  location. The wrapper splits `B`, calls the (compiled) kernel per chunk,
  reassembles. The kernel's own signature stays frozen — the Tier 7 I16
  invariant ("exactly one `backend.compile` site, signature unchanged") keeps
  passing. Fixed chunk size → at most two traced shapes (full + tail, or a
  padded tail). **Bit-identity is provable**: chunking `B` partitions
  baselines only; each visibility's source-sum order is untouched — assert
  byte equality across chunk sizes including chunk=∞ on the reference
  workloads. Evidence: `MemoryScalingRecord` before/after. The one design
  note: chunk-size policy (fixed vs derived from a memory budget — the
  `Tier7JonesSciencePlan.md` §41 Q2 precedent). Effort M.
- **P-b — retrace amortization**: pad the visible-source axis to bucket
  boundaries with zero-flux masking, caller-side; appended zeros preserve
  summation order → bit-identical on the NumPy reference path; JAX stays
  inside its existing parity tolerance. Bounds distinct traced shapes to
  ~log₂ buckets per run. Evidence: `RetracingRecord` before/after. Effort S–M.
- **P-c — lazy auto-probe**: `get_backend("auto")` eagerly imports JAX when
  installed (~450–950 ms first call, `Tier6HybridRuntimePlan.md` §39). Make
  the device probe lazy. XS.
- **P-d — flip `VisibilitySimulator.supports_gpu`'s ABC default to `False`**
  — the behavior change 8E explicitly parked with `PERF-001`
  (`Fix.md` ~15316-15334). XS.
- **P-e — GPU leg, hardware-blocked.** When hardware exists (Q4): an optional
  `gpu` pixi environment mirroring the `crossval` pattern (shared solve group,
  never gates — `Tier7JonesSciencePlan.md` §34), `jax[cuda]` unlocked there
  only, the **existing** benchmark harness run as-is, records committed. Only
  then may any acceleration sentence exist, citing the record. Until then this
  leg cannot start and nothing should pretend otherwise.

Fold in the `Tier6HybridRuntimePlan.md` §39 methodology note (tracemalloc
`peak_host_bytes` under-represents JAX's native allocations: `6,115,963` vs
`1,253,968` bytes for the same workload) as a docs line. All CPU legs are
bit-identical by construction → zero pin churn; land before WP-8 (edge E3).

## 11. WP-8 — `SCI-005`: beam physics beyond scalar `E`

Tier-scale; gets its own plan document (house rule: design gates before
substantial implementation). Five owned items in
`docs/development/beam_physics_scope.md` (near-field is a recorded *permanent
non-goal*, not an SCI-005 item); `Tier7JonesSciencePlan.md` §40 additionally
routes station element beams, array factors, and mutual coupling to this row.
Stage by risk gradient:

1. **Scalar-preserving aperture items** — aperture blockage, Zernike
   aberrations, Ruze error-beam decomposition. Aperture-integral changes; `E`
   stays scalar; receptor contracts and scalar-beam pins untouched except
   where a config block enables an effect. Per the `Fix.md` §16 discipline,
   each item lands with an analytic-invariant test (closed form in the test
   body), an effect-changes-visibility test, and a backend-parity case; the
   config convention carries over — a block resolving to exact identity is
   **rejected**. The scope document leaves blockage/Zernike/error-beam
   uncited; the stage-1 design memo supplies citations (Ruze 1966 is already
   cited for the efficiency factor; standard candidates: Baars 2007, *The
   Paraboloidal Reflector Antenna*; Born & Wolf for Zernike polynomials).
2. **Beam squint** (Cotton & Uson 2008, arXiv:0807.0026) — the first genuine
   widening: `E` becomes a non-scalar *diagonal* (per-hand pointing). This
   triggers the recorded `Tier7JonesSciencePlan.md` §12.3 obligation: the
   `C·E·P` order was fixed "because that is the physically correct one for a
   future non-scalar `E`", and any non-scalar `E` work "must re-verify it" —
   the scalar-`E` order-unobservability bit-identity test flips into an
   order-matters test with an analytic expectation.
3. **Full cross-polarization** (Ludwig-3 / quadrupolar / IXR; Carozzi & Woan
   2011; Ludwig 1973; HBS 1996) — full 2×2 `E`: UVBeam **efield** ingestion in
   `core/beam/fits.py` (today's path is peak-normalized power; the efield
   normalization convention is a named design decision), the solver-owned `E`
   adapter widened, the `PolarizationBasis`/receptor contracts engaged in
   earnest.

**What it does not touch:** the compiled kernel. Jones matrices arrive at
`core/contraction.py` fully composed as `(B, S, 2, 2)`; a non-scalar `E`
changes composition upstream, not the kernel signature or the single
`backend.compile` site — the Tier 7 invariant tests keep passing, and beam
interpolation stays host-side by design (`Tier6HybridRuntimePlan.md` §13.6).
**Validation:** pyuvdata UVBeam efield as the independent reference;
crossed-ideal-dipole closed forms; a crossval extension vs `pyuvsim` with an
efield beam (new dated artifact — `pyuvsim` supports exactly this).
Beam-identity fingerprints regenerate only for workloads that enable the new
effects, with the scalar-case bit-identity proof as the acceptance
centerpiece. **Blockers:** stages 2–3 on the WP-4/WP-5 ruling (E2); stage 1
sequenced after WP-7 (E3). Effort XL, multiple design-gated slices.

## 12. WP-9 — `SCI-004`: the m-mode simulator

A **feature, not remediation** — `Tier7JonesSciencePlan.md` §18.3's words: "a
second complete forward model", with the design-gate checklist already
enumerated there: a defined and enforced observing regime (drift scan); a
spherical-harmonic sky **including polarized components** (spin-2 harmonics
for `Q ± iU`); per-antenna beam harmonics (why it wants `SCI-005` maturity);
the `B_lm` transfer construction per baseline per frequency; the m-mode
transform of the time axis against the Tier 4 time-grid contract, with
verbatim typed rejections for non-uniform-sidereal configurations; per-`m`
solves; and a validation program of direct-sum agreement on small skies with
stated truncation bounds. Mechanically it is what the registry was built for:
a new entry in `simulator/__init__.py`, `execution.simulator: mmode`, "no
schema surgery required" (invariant I15). It extends the characterization
discipline with a second solver's observation families, so it inherits the
`CI-001` successor gate. Starting citations for the eventual design gate:
Shaw et al. 2014 (ApJ 781, 57) and Shaw et al. 2015 (PRD 91, 083514).
Performance claims only via records, as everywhere.

**Recommendation:** write the design gate only when a science driver (a
drift-scan survey use case) is named — gated question Q5. Until then the row
correctly sits as ROADMAP and is not scheduled by register pressure. Effort XL
when it comes.

## 13. Dispositions argued (close-as-documented and anti-candidates)

1. **`SCI-007` — close as documented-and-bounded** (conditional on the WP-6
   reconciliation matching): milli-degree scale, fully attributed, below the
   accuracy floor of everything else in the accepted subset; the close carries
   an *executable* bound plus the named upgrade path (`.ultra()` frames).
2. **`API-001` — defensible won't-fix, but the fix is adopted** (Q2): the
   signature's own defaults raise on array input; one `broadcast_arrays` line
   with a bit-identity argument is cheaper than maintaining the justification.
3. **`PERF-001`'s implicit "JAX-CPU should beat NumPy" — close explicitly as a
   non-goal** (one sentence in `docs/user_guide/backends.rst`): JAX's role is
   the compilation-capable backend for future accelerators; NumPy is the CPU
   reference; the committed records already say JAX-CPU is slower and that is
   acceptable. The row itself stays open for the memory/retrace legs and the
   hardware-gated GPU leg.
4. **`SCI-004` — not closable, and not remediation.** Scheduled by science
   need; its ROADMAP status is the register already saying so.
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
- **Q4 (blocks WP-7 P-e only):** name the GPU hardware and access path (cloud
  runner or workstation). Until answered, no accelerator work and no
  accelerator claims.
- **Q5 (blocks WP-9):** name the science driver (drift-scan survey use case)
  that schedules the m-mode design gate, or leave WP-9 unscheduled.

## 16. Status ledger

| WP | State |
|----|--------------------|
| WP-1 | DONE — independently accepted 2026-08-11; API-001 and API-002 closed |
| WP-2 | DONE — accepted 2026-08-08 |
| WP-3 | DONE — accepted 2026-08-08; CI-001 closed |
| WP-4 | DONE — ruled 2026-08-08; Branch A selected; no runtime change |
| WP-5 | DONE — independently accepted 2026-08-11; SCI-006 closed |
| WP-6 | Design and implementation ready; WP-5/E2 satisfied |
| WP-7 | P-a…P-d ready to start; P-e blocked on Q4 |
| WP-8 | Plan document ready to draft; stage 1 after WP-7; WP-5/E2 satisfied for stages 2–3 |
| WP-9 | Unscheduled pending Q5 |
