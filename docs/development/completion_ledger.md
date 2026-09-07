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
- Intermediate source `cf1a976` run `34088524159` completed with lint, metadata,
  types, docs and NumPy/JAX-CPU parity successful; all six compatibility jobs
  were cancelled. This is not complete CI acceptance.

## Finite work and dependencies

| ID | Authority and current observation | Required result / small slices | Verification and acceptance | State |
|---|---|---|---|---|
| A1 | SCI-004 design Sections 13.7/14; D30 review recovery and D31 runtime-input ownership correction | Authenticate original records; reviewed historical exceptions and main-only phase ranges; strict validator repair | Two fresh reviews of identical design bytes; full ancestry/path/blob checks and hostile mutations; preserve historical red inventories 29 + 6 + 2 | Complete terminal R at `567f9ac` independently accepted by both reviewers; D31 remains historical R authority; D30 remains range origin |
| A2 | D29 five-path candidate and original fingerprint R3 `a65c53a46e84f63c163c5ad15fba8645df33d1d2`; replacement-S3 overlaps two evidence paths | Integrate historical/current binding and oracle deltas by hunk; isolate exact historical imports; freeze terminal R | Exactly two expected failures and three controls at historical source; explicit source path, PYTHONNOUSERSITE=1 and verified radiosim.__file__; complete governed serial suite | Exact `567f9ac` replay and disjoint characterization partition pass; complete serial unit suite 7,618 passed, two existing platform skips; original 80 stopped-candidate hunks accounted for; primary source-hunk accounting advances with landed slices |
| B1 | Failed CI at `cfc9b10`; frame certificate reports outside-slab horizon sign mismatches | Reproduce operational/frozen root/guard/slab issue with environment provenance; reviewed correction if contract changes; focused fix | Independent physics and numerical diagnosis; decisive regression with unchanged scientific budgets; serial and compatibility jobs | Repair pushed; 13 frame tests pass on both Python versions and all seven public integration tests pass on Python 3.12; final exact-SHA CI remains pending |
| B2 | Failed CI reports SciPy intersphinx inventory ConnectTimeout | Inspect failed logs and current retrieval; bounded reliability fix if reproduced/justified | Clean warnings-as-errors Sphinx build; retain documentation validation | Fresh retrieval succeeded for all inventories; original CI ConnectTimeout remains an observed transient retrieval failure |
| C1 | D25 production v2 contract, D31 runtime input anchor, D32 identity transitions and D33 lossless certificate storage; preserved draft has strict-schema and same-run-join defects | Account for every original hunk; corrected input manifests, runtime identity bridge, result records, evidence/schema, acceptance and hostile tests in coherent source commits | Path-independent reconstruction/relocation, semantic mutation and malformed-input rejection; exact nine-key production family and retained manifests; eleven modified paths plus four authenticated disposals under D32/D33 | First S, runtime bridge, validated result accessor and current-A topology pushed and independently reviewed; production-v2 factory pushed at a39ceff passes 450 evidence tests and produces the required records for all four actual public families; D32 and D33 designs are pushed; current authority wiring, portable transition/storage validation and observation admission remain pending; generation stays closed |
| C2 | SCI-004 M3 Sections 10–14; four rejected-artifact deletions and six null sentinels | Complete replacement source range; authenticate disposal; generate new evidence from clean exact terminal source tip; separate E and A commits | HDF5/bounded reads, summary, UVFITS/MS read-back, full correlations/time/solver/provenance; all families; retained non-gating performance; fresh independent manual/numerical/schema acceptance | All six approval sentinels are committed literal None; four rejected artifacts remain unchanged; disposal follows completed source logic |
| D1 | SCI-004 Sections 7/9/15; public solver currently rejects HEALPix/hybrid and non-scalar beams; accepted SCI-005 beam contracts remain authoritative | Reviewed successor public support contract; integrate point/HEALPix/hybrid full Stokes; stationary squint and applicable full-efield support through canonical BeamSystem/Jones | Public API and CLI tests; dense/sparse pixel measure, frequencies, tangent/receptor frames, component provenance; common direct oracle and two-tier predicates | Inventory underway; successor contract before implementation |
| D2 | SCI-004 Section 9 defers public backend routing; current dense public path is NumPy | Route dense contraction/synthesis through requested backend; enforce real precision/x64; stream budgeted transfer blocks; implement canonical worker scheduling | Public NumPy/JAX/Dask parity, precision rejection, allocation-before-budget rejection, scheduling instrumentation and worker invariance; independent computational review | Pending D1 contract |
| E | PERF-001 accepted CPU mitigations/readiness; accelerator measurements require an authorized compatible host | Inspect available resources; strict preflight; complete workload matrix and authentic device-memory/timing record if hardware is available | Exact clean source/locked environment, real device and synchronized complex128; independent accelerator acceptance; timing is non-gating | Resource availability unresolved; PERF-001 remains ROADMAP, no accepted accelerator-performance record exists |
| F1 | Live public support must agree with README/CLAUDE, docs, exports/configs/examples and register | Reconcile actual final support, breaking changes and actionable errors; audit first-party TODOs; use authoritative coverage masks only | Public examples/script/notebook, docs scans, packaging/export checks; preserve unknown survey coverage and permanent non-goals | Pending capability work |
| F2 | Pyright is a strict diagnostic debt ceiling; audit reported 2,983 errors under 4,600 | Establish current report; fix relevant changed-code debt; ratchet verified reductions | No new changed-code errors, no raised ceiling or broad silencing; report actual total | Original draft: 2,983 errors; parked baseline: 2,958; focused new history tool has zero errors and evidence-tool diagnostic set is unchanged; final ratchet pending |
| G | SCI-004 Section 15 and CI-001 admission discipline | Final source/evidence/acceptance reconciliation and whole-row review; final exact main CI | Unit, integration, non-slow, doctest, lint, format, type ceiling, strict docs, whitespace; both Python versions and six compatibility cells, backend parity; local/live remote equality and no unexplained residue | Pending all required rows, including real external requirements |

## Original draft accounting

The original nine-path draft is now parked in the verified external recovery
bundle so prerequisite red tests and small source commits use committed source.
Reversing only the authenticated five-file unstaged patch and four-file indexed
deletion patch restored all nine paths to their exact current Git blobs and left
the index/checkout clean. Every original hunk remains retained verbatim in the
bundle; both stopped candidate worktrees remain untouched.
The external `original-hunk-inventory.json` enumerates all 63 primary hunks,
32 D28-candidate hunks and 48 D29-candidate hunks (SHA-256
`f4417e47c3d6038a164cfb448b95aeaad55a9e07eb67003fa68dcae34c19c356`).
C1 must replace this provisional accounting with per-hunk dispositions
(retained, corrected, superseded with reason, or obsolete under reviewed successor
contract) before source acceptance. Whole-file copying of overlapping candidates
is not an integration method.

The external `provenance-audit/original-hunk-disposition-v2.json` now accounts
for every stopped-candidate hunk through exact `87b16ba`: 21 retained, 38
corrected and 21 superseded, each with commit/blob/locator and reason. Its
SHA-256 is `b2d6a7969469a9b42fecba09335434484f8421dd1e8a39f1a1edfaa29fc9310b`.
Primary accounting through `bbd5f3f` is
`provenance-audit/original-hunk-disposition-v8.json`,
SHA-256 `0cbeb0c2fdf6b90880c06526b030bd4ec66d05bcd542e94c7513362bc5820a0e`.
It retains all 63 original identities: twenty hunks landed (ten retained,
ten corrected), and 43 remain pending. The 80 stopped-candidate dispositions
are unchanged from v2. Production-v2 commit `a39ceff` lands donors 001–006,
058 and 059; b17f63a lands corrected donor 015 and ec94dc8 lands corrected donor
034. Donor 057 remains pending as a
whole: its caller wiring landed,
but its fingerprint-row schema did not. New tests receive separate accounting.

## Commit and verification journal

| Slice | Commit / publication | Evidence |
|---|---|---|
| Recovery and fast-forward | No new commit; main at `cfc9b10` | Three recovery manifest reconstruction checks PASS; original primary patch hashes unchanged |
| Initial ledger | `d432bcb50f60c880aca3d3e599786b9ebe62fa1c`, pushed and live remote verified | Whitespace/source review; fresh Sphinx later found missing inclusion declaration, corrected with explicit orphan metadata in this status slice |
| Reviewed D30 | `d3ddb10ae01ab450f5337d06c9588ce8144cf1e5`, pushed and live remote verified | Two fresh independent ACCEPTs; exact reviewed memo/ledger/diff pins in correction header; no phase acceptance |
| Historical review retention | `860222ac90eaa7b9a2a1c3b282e3ec0f51b7834b`, pushed and live remote verified | Frozen 108,125-byte JSON SHA-256 `eb9b00fcdb7703cb40982bc7e445ba6e042fb45ca26bd0515387dfb644975d54`; exact Git/archive joins independently authenticated |
| Portable review authenticator | `8a7d4ea`, publication verified before this journal update | 33 tests pass, including twelve hostile Git settings and GIT_DIFF_OPTS override; both reviewers ACCEPT round-3 complete patch `105ed0a79c3af8f6a116c232b5b12367a83e7c9e1b8aecde6ebf99572a5bea2b`; continuation/range wiring remains pending |
| Structural phase-range authenticator | `e269bd612667abcac4262fc14cf1fe9e27ceaea7`, pushed and live remote verified | Both independent reviewers ACCEPT round-2 patch `1fda53092e195b9e5df85112ccafb4a7f36d52b05b6ceae620b390e4daed1efb`; 63 tests pass in 41.20s; focused tool Pyright zero errors/warnings; semantic source/status checks and phase acceptance remain separate |
| Ordinary SHA-keyed review map | `6809c7f`, pushed and live remote verified before this journal update | Both independent reviewers ACCEPT patch `e43da9dda08966a3c7d524052423f4d491709734d39e47004f525430d0c942b9`; 12 focused tests pass; missing, duplicate, unknown and misattributed reviews reject; existing D26 binding awaits next continuation slice |
| Rejected-review portability | `7525d184e92145c164a5845c606a2a6096b6e0ee`, pushed and live remote verified | Both reviewers ACCEPT; original rejected review reconstructed byte-for-byte from authenticated retained A3 contribution; 4 focused tests pass; no dependency on the original author machine path |
| Complete D30 continuation/range binding | `5ae75297b2fe3e1b88afc977e5cc3917cce152fe`, pushed and live remote verified | Both reviewers ACCEPT round-2 patch `478127c7ec65bf3d856ef3512d0d9d4bb0c0260d26727d0604aebc8f02a9efd4`; 97 tests pass; duplicate/unknown reviews, foreign own-header pins, competing or removed terminal-R metadata reject |
| Historical/current evidence binding | `8ecd3c93c7302337ba7b34c075a39cf591292b0e`, pushed and live remote verified | Both reviewers ACCEPT round-2 patch `b9a661849556799443bf53b61b791de8da41ba3ab84b6a2898af124557999e59`; 301 tests pass; all three original records retain their historical identities; actual generation refuses an authoring tip; focused 112 strict diagnostics unchanged |
| Characterization transition oracle | `834e25b4fda3069bff980385c2c5a5d67cb298bb`, pushed and live remote verified | Both reviewers ACCEPT patch `02f54ca4de1fe651b358643a311df6f8e3b1a34fbb4fc4bad630458f12ece67f`; general observation-set oracle passes; dedicated v2 assertion retains exact ordered manifest requirement; historical replay verified separately |
| Red history and exact-tree imports | `aa683156717b58cece68e0011d0fbb2d70e66271`, pushed and live remote verified | Both reviewers ACCEPT round-2 patch `caa54bcf0970ee3b6e0dbe5b931fe7957b172f2fd66405dd65d3a5be4b2f9c28`; 58 tests pass; exact three-record inheritance and 29 + 6 + 2 inventories verified; package/source symlink escapes reject; fresh replay remains a separate pending repair |
| Original fingerprint replay | Exact immutable source `a65c53a46e84f63c163c5ad15fba8645df33d1d2`; external `historical-fingerprint-probe-default/` under recovery root | Python 3.11 serial replay: exactly two governed assertion failures followed by three passing controls in 943.03s; explicit historical source, user-site exclusion and resolved package import verified; JUnit SHA-256 `370aff752a2f17af428f928860e3dc6a4dd4306d4468f46f4947c127156efdbd`; no frame/collection failure or skip; separate fresh-R replay remains required |
| Mandatory fresh-R replay | `87b16ba16c8a4ab4ff8b9e6bf213c5ce45a41bfe`, pushed and remote verified | Both reviewers ACCEPT; ordered qualified JUnit/terminal-summary classifier, exact-source import and before/after checkout checks; external `terminal-r-87b16ba/validation.json`: Python 3.11 outer test passed in 978.55s and Python 3.12 in 1059.45s, each requiring the two-red/three-control child and separate passing general oracle; observations apply only to exact 87b |
| Reviewed D31 | `f2e5edbcc97450262482672bb322cf926622b208`, pushed and remote verified | Both reviewers ACCEPT identical candidate; exact memo/companion/full-patch pins and finalization in own header; grants runtime input anchor without changing scientific computation or twenty serialized solver keys; no phase acceptance |
| D31 history and bridge scope | `47a4f4e632bcb7f8bb6674570f6fb526c4fa6fa9`, pushed and remote verified | Both reviewers ACCEPT patch `e12eb244f2a05070311306825ede95c7ecd184d0d46a9dc350db2f938d91dae1`; 90 tests pass; exact sole design successor, distinct D30 origin, per-commit and cumulative AST restrictions; focused Pyright zero |
| D31 dependency binding | `e356a3cc743bcf26d8d21af70d22c698b16d4bbc`, pushed and remote verified | Both reviewers ACCEPT patch `de52dc45745e7b1c719abaa1950f69e9fcbb7fe07a2a9902fb7a92a66fb099d6`; root 102 tests pass in 35.54s and independent stable run in 35.20s; another reviewer run failed only its final ambient-status check during concurrent commit-hook handling and is not counted as passing |
| D31 red chain | `32216ca7feef5cf9782a02668e96bd7ba86b1213`, pushed and remote verified | Both reviewers ACCEPT patch `37889f77ac932690df620772aa4a4841f900afcb83a3b49faa88e181500d0d7c`; 74 nonheavy tests pass in 14.03s; original historical identities and heavy replay implementation unchanged |
| D31 evidence roles | `b847fea89610a0788627338c84085f03b4050b68`, publication verified before this journal commit | Both reviewers ACCEPT patch `2fc9712b07a91887d0b1679a3fd11dee53aca21f6a438d6fd1b874f640bd3d21`; 307 tests pass in 38.88s; separate source-read and loaded D31/D30 joins; 112 existing strict tool diagnostics unchanged; generation still refuses an authoring tip |
| Terminal R | `567f9ac68730044fc8e887930d3531d794534412`, pushed and independently accepted | All eleven gates pass; disjoint characterization partition 1 + 5 + 11, exactly two governed failures and three controls within the five-node replay; dependency 102 pass, red 75 pass, Tier 8 21 pass, complete serial units 7,618 pass and two existing complex256 availability skips; strict type total 2,958 under unchanged 4,600 ceiling; external `terminal-r-567f9ac/validation.json` SHA-256 `674f743e535a82d196b2fab98a46b95950886b9fdce932c8b1a72344736430cb` |
| Terminal-R remote CI | Exact `567f9ac`, [run 34063024901](https://github.com/RRI-interferometry/RadioSim/actions/runs/34063024901) | Backend parity and lint/metadata/types/docs succeed; all six compatibility jobs cancelled, with the checked Linux/Python-3.12 annotation recording the 45-minute job limit during non-slow tests; cancelled cells are not accepted; runtime/timeout repair remains pending outside the current source path grant |
| First-S readiness | `72a0a5c5ebd203b63e091342f3655ebf808bac4b`, ordinary push and live remote verified | Both reviewers ACCEPT complete patch `0ec58766f135a6f498d2dad2441d6f78f6f09e6d9be3120fb5f5b55c6f180e10`; 379 candidate tests pass, independent 72 focused tests pass, committed dependency/evidence run 481 pass in 94.75s; lint/format/whitespace pass, type total and focused diagnostic set unchanged; actual committed range resolves terminal R and generation refuses with all four rejected artifacts unchanged |
| Runtime input bridge | `efd9e9289d7b1a98c44710d5242446407d8c7055`, pushed and remote verified | Both reviewers ACCEPT patch `5eee72a37db275baac79a922a3292810eaa0fab0a6b199133062aa2437f22208`; evidence 383 pass, existing m-mode IO seven pass; four adapter cases independently pass for each reviewer; permitted AST subtraction recovers prior solver and fixture exactly; preserved synthetic cube/scientific/snapshot identities unchanged; actual public-run regression remains terminal-S work |
| Result runtime identity accessor | `07662b20e6cf41b1e67f8badce75f62e37727350`, pushed and remote verified | Both reviewers ACCEPT patch `4be7e04926ed08f146de4919f2192e7e6162b1e44babbf8e4533ca94f09a3fd2`; exact live snapshot ownership and lowercase digest required, loaded/foreign/malformed values reject; 13 focused and 39 existing result tests pass; focused 314-diagnostic multiset unchanged, including 16 in result.py; serialization AST unchanged |
| Acceptance sentinel lifecycle | `9f50a4cbf38b72057a097ecc3b85f1126fc9ce67`, pushed and remote verified | Both reviewers ACCEPT patch `56dab7ab677f61c3bfe7c9be087cb217b5523da30ed7e3d0e0c3c02932833759`; mandated serial module 89 pass and six future approved-A skips; two literal-null sentinels, tenth fingerprint oracle and exact historical-REJECT allowance; no disposal, topology repair or independent full numerical/history acceptance; 60 existing diagnostics unchanged; original agent run without plugin-disable flags retained only as development evidence |
| Current-A topology | `babc0fc7cfd541d54250e087371fe5177511c826`, ordinary push and live remote verified | Both reviewers ACCEPT round-2 patch `5528ca8c512b9897b83933d92f1d65a75a9dca54a9bd73ec2990ec50628d6d7a`; committed serial module 120 pass and six future-A skips; 31 real-Git cases include replacement/graft/environment attacks; exact E sole-parent edge and literal-only binding changes; both approval sentinels remain None |
| Reviewed D32 | `bcd79b1d6268859368d77c3f94cef334b001cb37`, pushed and live remote verified | Both independent reviewers ACCEPT frozen contract and finalization; exact two-file design successor authenticated independently; warnings-as-errors Sphinx passes; eleven modified source/test paths plus four disposals, distinct D31 R and D32 S roles, portable per-family identity proofs and independently reviewed observation admission required; no phase or observation acceptance |
| Production characterization-input v2 | `a39ceff157d37e1a60aa19e9dd02afaabd3908a7`, pushed and live remote verified | Both reviewers ACCEPT complete patch `a52459bb45541ade89c92abc3d76d6c15fcc5612ee655acbef9749eb822d0b54`; full evidence module 450 pass, post-D32 focused 58 pass; changed-file strict diagnostics 431 to 426 with none added; all four actual public family records validated, scientific identities remain novel and unadmitted; isolated indivisible factory/schema/test slice |
| Acceptance historical raw bytes | `3b317218fa8239a230e208600f3bcb4bfc2af4b8`, pushed and remote verified | Both reviewers ACCEPT; full module 122 pass and six future-A skips; actual blob replacement/routing regressions independently fail the parent implementation |
| Reviewed D33 storage contract | `343ea0467420d452e9d728f0475167e74721e22f`, pushed and remote verified | Both reviewers ACCEPT final contract and finalization; strict Sphinx passes; fixed bounded inline certificate encoding preserves expanded transition digests; implementation and phase acceptance remain pending |
| History original-object reader | `34fa01c8ab16bbaef3d829eda7afc82e7df94df1`, pushed and remote verified | Both reviewers ACCEPT round 2; 117 tests pass; scoped strict diagnostics 302 to 300 with none added; replacement/graft/routing and effective diff-driver transformations reject; original historical roles remain unchanged |
| Evidence rejected-attempt lifecycle | `b17f63a39ee5489f0bab254191a899ae45c85e9c`, pushed and remote verified | Both reviewers ACCEPT round 2; 498 tests pass and seven future-E skips; four literal None approvals separated from authenticated historical bytes; closed benchmark-directory inventory; scoped strict 298 unchanged |
| Dependency Git and Python isolation | `bafa924563559488469708e8c67a73d4b37454a4`, pushed and remote verified | Both reviewers ACCEPT round 2; 125 tests pass including original Stage2 replay; native startup controls reject inherited Python hooks without changing historical verifier arguments or pins; strict 108 unchanged |
| Acceptance checkout binding | `a7a8fa33f9ff4b7da73058b2cea937b1fbd9ed81`, pushed and remote verified | Both reviewers ACCEPT round 2; 127 tests pass and six future-A skips; five independent Git controls include local core.worktree redirection in primary and registered checkouts; strict 60 unchanged |
| Current-E topology and literal transitions | `ec94dc8337ee3a71671e8fbd61bab4985f452fd9`, pushed and remote verified | Both reviewers ACCEPT; 535 tests pass and seven future-E skips; 37 real-Git cases independently pass, plus 13 independent hostile probes; exact first S-child, sole-parent chain, four core paths and optional factual ledger; complete text preserves everything except four RHS None literals; strict 298 unchanged |
| D33 certificate transport | `50cbf3086b40b0810b9d3c7331be5ef00b5d1b39`, pushed and remote verified | Both reviewers ACCEPT repaired Unicode/number canonicalization; full evidence module 602 pass and seven future-E skips, scoped strict 410 unchanged; eight retained certificates roundtrip and all four expanded transition bytes/digests remain exact; transport only, full scientific validation and publication wiring pending |
| Exact D32/D33 design authentication | `cf1a976640a437cec5181d7697a0ef56e5f49559`, pushed and remote verified | Both reviewers ACCEPT round 2; full history 158 pass; corrected inherited-review fixture rejects and fails under the isolated weakened-check mutation; strict 300 unchanged; D31 behavior preserved; source-range role integration subsequently landed at `732865a` |
| Original-object red reader | `28cd74ec2718535bdb951368a296d5b15f6e0d7f`, pushed and live remote verified | Both independent reviews ACCEPT; eleven native Git controls and two original oracle pins pass; strict 54 unchanged. Filter/cache repair subsequently landed at `7eca44b`; detached checkout repair remains pending |
| D32/D33 source range enforcement | `732865ad1b0da50e96e439b41694e88e9d34bc26`, pushed and live remote verified | Both independent reviews and new-parent confirmations ACCEPT; 183 history tests and 611 combined evidence tests pass, seven future-E skips; scoped strict 598 to 545 with zero added diagnostics. Exact range-only three-path patch excludes the separate evidence size guard subsequently landed at `913d45c` |
| Complete evidence size bound | `913d45cbcf0311fb76ae24a4dc1e3f9cde0adea5`, pushed and live remote verified | Both independent reviews ACCEPT; strict less-than-104857600-byte guard precedes either output, reader bounds bytes before parsing; independent boundary and CLI tests pass. Scoped strict 376 existing diagnostics, zero added; full combined evidence 611 passed and seven future-E skips |
| Shared Section 14 JSON and strict readers | `bbd5f3fbe540628c8f8d5e29b39696fe7d002dcd`, pushed and live remote verified | Both independent reviews ACCEPT corrected round2; full 635 passed and seven future-E skips; 376 to 367 existing scoped type diagnostics, zero added. Six historical canonical artifacts remain byte-identical; production scientific serialization unchanged. Initial malformed-RSS wording and test typing failures retained and repaired |
| Raw oracle patch identity | `7eca44bac74e3033c761d5019e22b8268908e889`, pushed and live remote verified | Both independent reviews ACCEPT; twelve native conversion/cache controls and two original oracle pins pass; strict 54 unchanged. Raw whole-checkout and Python child isolation remain subsequent slices |
| Distinct source design bindings | `d90081bb6382dbf8c233b91fa0f2aac6923d5f11`, pushed and live remote verified | Both independent reviews ACCEPT; 145 serial dependency tests passed, strict 108 unchanged; original historical R nodes remain exact. D32/D33 callable authentication landed; evidence wiring remains pending |
| Scientific segment primitives | `401d834831144e463e786fc11269cf2e723232b5`, pushed and live remote verified | Both independent reviews ACCEPT; 44 focused tests plus independent production-encoding and binary controls pass; strict 367 unchanged. Exact segment bytes, scientific JSON and fixed arrays only; stream, solver, owner and admission joins remain pending |
| Frame regression | `1909829d828078fd36a905aa68cde50fcb4bfa16`, pushed and live remote verified | Two controlled expected-red assertions under each Python version: mismatch count 1, not 0; no collection/import failure |
| Frame repair | `cfad247831629241842ffecd5f7aaa5b2084493c`, pushed and live remote verified | Both independent reviewers ACCEPT frozen full test/source patch `231f48c6d0d4bae419bb0cba0091813c928be40e3100713754c3a7c5904e00cb`; 13 frame tests pass in each environment; 7 public Python-3.12 integration tests pass in 741.40s; solver strict diagnostics remain 78 with no added diagnostic |
| Ledger docs fix | `c245593df808e0a757925d5a02416b4608cd8661`, pushed and live remote verified | Fresh Sphinx build fetched all inventories; explicit ledger orphan declaration restored warnings-as-errors success |
| Frame diagnosis | External `frame-investigation/REPORT.md` under recovery root | Python 3.12: 279 outside-slab mismatches become zero with installed IERS context, unchanged geometry; controlled one-interval perturbation reproduces on both Python versions |
| Draft type baseline | External `pyright-original-draft.json` under recovery root | 196 files, 2,983 errors, zero warnings; debt ceiling unchanged; this is not a clean type check |

Every later slice records its actual commit and verified push in a subsequent
journal update, avoiding self-referential commit hashes. An unexpected failing
gate stops acceptance of its affected phase; diagnosis and authorized repair
continue within this programme. Cancelled CI jobs and selected passing subsets
cannot establish final closure.

## Current source verification stop

The committed production-v2 implementation at `a39ceff` (external
`s3-production-v2/round3/`, patch
`a52459bb45541ade89c92abc3d76d6c15fcc5612ee655acbef9749eb822d0b54`) received
both independent source reviews. Its complete evidence module passes 450 tests;
strict changed-file diagnostics introduce no new errors. The first actual
public scalar run produces the required nine-key family record, fourteen-key
input preimage and twenty-six-key same-run phase preimage. Its historical
snapshot/scientific comparison stopped; that original failure is retained.
The three remaining public-family diagnostics subsequently completed with
`novel_unadmitted` outcomes. Their exact cube, phase and time identities match
history; the restricted certificate substitution reconstructs every historical
snapshot and scientific hash. The complete run retained 279 authenticated
artifacts, including actual HDF5 results and independently rehashable scientific
preimages. All source, helper, driver and working-state checks passed. Aggregate
`unadmitted-family-diagnostic/verification.json` under the round-3 bundle hashes
to `b52742f9e9bf90c7a29d3c48e3a3214752dbea1aff0a352dd4d05ebf1ba77129`.
Both reviewers independently accepted those three causal proofs; the required
D32 append-candidate admission reviews remain separate and pending.

Both reviewers independently accepted the scalar causal diagnosis in
`frame-investigation/certificate-probe/` and `scientific-preimage-probe/`.
The phase, cube and characterization-time identities remain exact. The whole
solver-file hashes changed at `cfad247` and `efd9e92`, propagating through the
certificate's enclosure/error/ledger hashes and the existing solver field.
Substituting only authenticated historical implementation hashes reconstructs
the exact historical certificate. The complete offline scientific preimages
then reconstruct actual `03b7f62a1daaf091ea16b1caa78a0e16a481622bad8e5b78ba5af0dc2dbed45a`
and historical `a8852fd181e8f1ddd08e24e01066bcede2e1e1541fba9a067d47f0febc51c344`;
only the solver certificate field differs. The scientific-preimage artifact
manifest hashes to `f781e42488352b71d0b974b6cd67c3de026af1d01f01a955b603939bf2f0a3e8`.
These are explicitly reconstructed preimages, not another public result.

D32 and D33 now supply the independently reviewed prospective contracts.
All four expanded thirteen-field transition candidates are retained under
`d32-transition-material/`; their independent portable verification SHA-256 is
`ddaccddcfba27dc6c26e7f5979dd6cdcbc2f05779665c61cdb5f665c4bc5c7d5`.
No retained primitive is missing. The exact owner-defined EarthLocation
round trip reconciles result and frame coordinates without a tolerance change.

The expanded four-row array measures 143,135,033 bytes before ordinary evidence
and admission reviews. D33 therefore defines bounded, lossless inline storage
for only the eight certificate values, while preserving the expanded transition
digest domain. The measured canonical encoded array is 38,929,073 bytes before
ordinary evidence and reviews; the complete final artifact must still pass the
specified size guard. Transport primitives landed at `50cbf30`; full certificate,
cascade and portable admission validation remain pending. No observation has
been appended and generation remains closed. The separate committed-blob
transport verification of all eight certificates and four expanded digests is
`d33-certificate-codec/independent-actual-transport/verification.json`, SHA-256
`1d386df11026d5f5f3a06776531c292db839ed15593c9882bf4d7c1b0516b38e`.

The reproduced replacement-ref history bypass is repaired at `34fa01c`.
Its retained original exploit is `d32-history-auth-audit/probe-result.json`,
SHA-256 `8e9e8a247cd1e035b5c7470cd75e7d2aa7ce4f945a76f64ddf9424cebce05819`.
It does not establish that a retained historical record was forged.
The current-E selector landed at `ec94dc8` after 535 passing tests, seven
future-E skips and both independent ACCEPT reviews. Exact D32/D33 authority
wiring, remaining
schema/proof integration, reviewed observation admission and final disposal
precede the complete terminal-source gates and separate E/A acceptance.
