---
orphan: true
---

# RadioSim completion ledger

Current programme opened 2026-09-07. This ledger records observed work and
remaining gates; it does not accept a scientific phase or supersede a design.
The user authorizes implementation on main, small commits and ordinary pushes,
prospective independently reviewed corrections, and exact-SHA historical replay
fixtures. No implementation is authored in another worktree.

## Authenticated starting state

- Primary main: `82fb0773890870a6fb90b3ed9b8065df89919a84`.
- Live remote main: `cfc9b10d655a4d9bedbd7d7750c4743f504bbaf9`.
- Main fast-forwarded to that remote tip without parking or modifying the draft.
- External recovery root:
  `/Users/kartikmandar/RadioSim-recovery/20260907-completion/`.
  Each repository bundle contains base SHA, original index, index entries,
  full-index binary patches, changed-file bytes and filesystem identities.
  Applying both patch layers in an isolated index reproduced the exact staged
  entries and final changed-file bytes/deletions for all three bundles.
- Primary unstaged patch SHA-256:
  `bc83d195be6a9d63a3945497595a16c5648eb06ec0c403e4c995966d3e5eccd3`.
- Primary staged patch SHA-256:
  `78aeebb4c4d240aba0899dba69b69db4e55506b6d6eacd826f396fd5a24a00f0`.
  Both hashes were unchanged after fast-forward. The four staged artifact
  deletions remain part of the replacement-source draft, not this ledger slice.
- D28 and D29 stopped candidates are preserved separately at their exact bases.
  They are unaccepted implementation candidates, not source authorities.
- No existing pytest, acceptance, evidence, or Sphinx process was running at
  initial process inspection.
- GitHub run `33701403760` failed at the remote tip; newer runs `33726794068`,
  `33847827991`, `33951179081`, and `34018313735` are cancelled, not successful.
  Per-job diagnosis and final exact-SHA CI remain outstanding.

## Finite work and dependencies

| ID | Authority and current observation | Required result / small slices | Verification and acceptance | State |
|---|---|---|---|---|
| A1 | SCI-004 design Sections 13.7/14; D28/D29 committed review headers remain pending; stopped candidate has unequal review-pin/chain cardinality | Authenticate original records; reviewed prospective historical exceptions and main-only phase ranges; strict validator repair | Two fresh reviews of identical design bytes; full ancestry/path/blob checks and hostile mutations; preserve historical red inventories 29 + 6 + 2 | D30 reviewed and pushed; strict validator implementation pending |
| A2 | D29 five-path candidate and original fingerprint R3 `a65c53a46e84f63c163c5ad15fba8645df33d1d2`; replacement-S3 overlaps two evidence paths | Integrate minimal historical/current binding and oracle deltas by hunk; isolate exact historical imports; freeze prerequisite range tip | Exactly two expected failures and three controls at historical source; explicit source path, PYTHONNOUSERSITE=1 and verified radiosim.__file__; complete governed serial suite | Pending A1 |
| B1 | Failed CI at `cfc9b10`; frame certificate reports outside-slab horizon sign mismatches | Reproduce operational/frozen root/guard/slab issue with environment provenance; reviewed correction if contract changes; focused fix | Independent physics and numerical diagnosis; decisive regression with unchanged scientific budgets; serial and compatibility jobs | Cause reproduced; separate regression/source repair next |
| B2 | Failed CI reports SciPy intersphinx inventory ConnectTimeout | Inspect failed logs and current retrieval; bounded reliability fix if reproduced/justified | Clean warnings-as-errors Sphinx build; retain documentation validation | Fresh retrieval succeeded for all inventories; original CI ConnectTimeout remains an observed transient retrieval failure |
| C1 | D25 production v2 contract; preserved five-file replacement-S3 draft has 2,885 added and 145 removed handwritten lines | Account for every original hunk; separate strict input manifests, result records, evidence/schema, acceptance validation and hostile tests into coherent source commits | Path-independent reconstruction/relocation, semantic mutation and malformed-input rejection; exact nine-key production family and retained manifests | Pending A1/A2 |
| C2 | SCI-004 M3 Sections 10–14; four rejected-artifact deletions and six null sentinels | Complete replacement source range; authenticate disposal; generate new evidence from clean exact terminal source tip; separate E and A commits | HDF5/bounded reads, summary, UVFITS/MS read-back, full correlations/time/solver/provenance; all families; retained non-gating performance; fresh independent manual/numerical/schema acceptance | Pending C1/B1 |
| D1 | SCI-004 Sections 7/9/15; public solver currently rejects HEALPix/hybrid and non-scalar beams; accepted SCI-005 beam contracts remain authoritative | Reviewed successor public support contract; integrate point/HEALPix/hybrid full Stokes; stationary squint and applicable full-efield support through canonical BeamSystem/Jones | Public API and CLI tests; dense/sparse pixel measure, frequencies, tangent/receptor frames, component provenance; common direct oracle and two-tier predicates | Inventory underway; successor contract before implementation |
| D2 | SCI-004 Section 9 defers public backend routing; current dense public path is NumPy | Route dense contraction/synthesis through requested backend; enforce real precision/x64; stream budgeted transfer blocks; implement canonical worker scheduling | Public NumPy/JAX/Dask parity, precision rejection, allocation-before-budget rejection, scheduling instrumentation and worker invariance; independent computational review | Pending D1 contract |
| E | PERF-001 accepted CPU mitigations/readiness; accelerator measurements require an authorized compatible host | Inspect available resources; strict preflight; complete workload matrix and authentic device-memory/timing record if hardware is available | Exact clean source/locked environment, real device and synchronized complex128; independent accelerator acceptance; timing is non-gating | Resource availability unresolved; PERF-001 remains ROADMAP, no accepted accelerator-performance record exists |
| F1 | Live public support must agree with README/CLAUDE, docs, exports/configs/examples and register | Reconcile actual final support, breaking changes and actionable errors; audit first-party TODOs; use authoritative coverage masks only | Public examples/script/notebook, docs scans, packaging/export checks; preserve unknown survey coverage and permanent non-goals | Pending capability work |
| F2 | Pyright is a strict diagnostic debt ceiling; audit reported 2,983 errors under 4,600 | Establish current report; fix relevant changed-code debt; ratchet verified reductions | No new changed-code errors, no raised ceiling or broad silencing; report actual total | Pending baseline |
| G | SCI-004 Section 15 and CI-001 admission discipline | Final source/evidence/acceptance reconciliation and whole-row review; final exact main CI | Unit, integration, non-slow, doctest, lint, format, type ceiling, strict docs, whitespace; both Python versions and six compatibility cells, backend parity; local/live remote equality and no unexplained residue | Pending all required rows, including real external requirements |

## Original draft accounting

The original nine-path draft is now parked in the verified external recovery
bundle so prerequisite red tests and small source commits use committed source.
Reversing only the authenticated five-file unstaged patch and four-file indexed
deletion patch restored all nine paths to their exact current Git blobs and left
the index/checkout clean. Every original hunk remains retained verbatim in the
bundle; both stopped candidate worktrees remain untouched. C1 must replace this provisional accounting with per-hunk dispositions
(retained, corrected, superseded with reason, or obsolete under reviewed successor
contract) before source acceptance. Whole-file copying of overlapping candidates
is not an integration method.

## Commit and verification journal

| Slice | Commit / publication | Evidence |
|---|---|---|
| Recovery and fast-forward | No new commit; main at `cfc9b10` | Three recovery manifest reconstruction checks PASS; original primary patch hashes unchanged |
| Initial ledger | `d432bcb50f60c880aca3d3e599786b9ebe62fa1c`, pushed and live remote verified | Whitespace/source review; fresh Sphinx later found missing inclusion declaration, corrected with explicit orphan metadata in this status slice |
| Reviewed D30 | `d3ddb10ae01ab450f5337d06c9588ce8144cf1e5`, pushed and live remote verified | Two fresh independent ACCEPTs; exact reviewed memo/ledger/diff pins in correction header; no phase acceptance |
| Historical review retention | `860222ac90eaa7b9a2a1c3b282e3ec0f51b7834b`, pushed and live remote verified | Frozen 108,125-byte JSON SHA-256 `eb9b00fcdb7703cb40982bc7e445ba6e042fb45ca26bd0515387dfb644975d54`; exact Git/archive joins independently authenticated |
| Frame diagnosis | External `frame-investigation/REPORT.md` under recovery root | Python 3.12: 279 outside-slab mismatches become zero with installed IERS context, unchanged geometry; controlled one-interval perturbation reproduces on both Python versions |
| Draft type baseline | External `pyright-original-draft.json` under recovery root | 196 files, 2,983 errors, zero warnings; debt ceiling unchanged; this is not a clean type check |

Every later slice records its actual commit and verified push in a subsequent
journal update, avoiding self-referential commit hashes. An unexpected failing
gate stops acceptance of its affected phase; diagnosis and authorized repair
continue within this programme. Cancelled CI jobs and selected passing subsets
cannot establish final closure.
