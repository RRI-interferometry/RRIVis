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
| B3 | Section 12.1 requires frozen root bounds in the initial scan partition; the scanner at `786413f` omitted them and only checked the supplied list length | Reviewed finite successor grant, decisive boundary regression, exact-bound inclusion repair, fresh four-family measurements | Independent partition/evaluation checks with unchanged numerical budgets; distinguish measured cube changes from certificate changes; new admission relation requires review | Confirmed at `786413f`; D34 is reviewed and pushed at `90ef12e`, with exact edge authentication at `3563fa7`; guarded activation is pushed at `505ef3d` and the optional workflow entry at `a18aa60`; separate regression is pushed at `b032580` and the bounded numerical repair at `aa933c2`; 35 frame cases pass on both Python versions and all seven public integration tests pass; clean four-family measurements completed at `e7623fb`, with independent exact initial-union and centre-projection checks; all four terminal structures and historical scan reconstructions are verified, with scalar separately postvalidated after its preserved driver failure; B4 source correction was subsequently pushed; the D36 successor in B5 now records completed public gates/capture, while certificate conformity and observation admission remain pending |
| B4 | Independent D34 comparison found that `_pair_roots` returns radians which the certificate labels as turns and converts again; 1,280 root-pair rows per family also violate frozen-bound order | Prospective reviewed solver/history grant; exact direction-owned slab unions, one outward radian conversion, canonical owned indices and independent semantic checks | Nonzero causal unit, overlap, seam and order regressions; unchanged budgets; fresh clean-source captures and numerical admission | Source defect and four-family arithmetic independently confirmed; corrected statistic is 0.004604447903486282 rad versus recorded 0.028930599414858853, under the unchanged 0.1024-rad limit; D35 reviewed and pushed at `dd26e04`; causal regression pushed at `5790241` retains five intended failures and one passing control; historical D34/pending D35 preparation pushed at `368821d` passes 161 affected history cases; strict guard pushed at `083fa84`, atomic D35 activation at `af8d595` and the reviewed numerical repair at `cf67c0b`; 109 frame cases pass on each supported Python version; public gates and fresh capture were pending at this D35 boundary and are now recorded under successor B5; conformity/admission remain pending and historical captures remain unchanged |
| B5 | Sections 4.2/12.1 require independent operational direct kernels/classes, outward error arithmetic and all root/guard partition boundaries; five source defects are confirmed | Reviewed D36 scope; separate causal controls, exact source/guard activation, corrected model, partition and error semantics | Independent model-owner and displaced-root controls; exact-rational radius/accumulation and guarded-partition witnesses; unchanged budgets; both runtimes, fresh captures and complete conformity before admission | D36 design and separate causal regression are pushed at `7752f04` and `216afb1`; activation is pushed at `e33fb26` and numerical repair at `c023036`; all four public integration/characterization gates pass across Python 3.11 and 3.12; fresh four-family capture completed at clean `c023036`, with 901 artifacts independently byte-authenticated; terminal semantic/numerical conformity and observation admission remain pending |
| C1 | D25 production v2 contract, D31 runtime input anchor, D32 identity transitions and D33 lossless certificate storage; preserved draft has strict-schema and same-run-join defects | Account for every original hunk; corrected input manifests, runtime identity bridge, result records, evidence/schema, acceptance and hostile tests in coherent source commits | Path-independent reconstruction/relocation, semantic mutation and malformed-input rejection; exact nine-key production family and retained manifests; D34 authorizes thirteen modified paths plus four authenticated disposals, and one optional exact workflow path; guarded activation and the workflow entry are implemented | First S, runtime bridge, validated result accessor and current-A topology pushed and independently reviewed; production-v2 factory pushed at a39ceff passes 450 evidence tests and produces the required records for all four actual public families; D32 and D33 designs are pushed; D36 at `7752f04` is executable current authority through `e33fb26`; D35, D34 and D33 retain their authenticated historical roles; input14/phase26 outer projections and source/time/scientific-buffer joins pushed at `96e6b40`; receptor native-float, angle, family and ordered antenna joins pushed at `4b93e7a`; remaining nested physical semantics, portable transition validation and observation admission remain pending; generation stays closed |
| C2 | SCI-004 M3 Sections 10–14; four rejected-artifact deletions and six null sentinels | Complete replacement source range; authenticate disposal; generate new evidence from clean exact terminal source tip; separate E and A commits | HDF5/bounded reads, summary, UVFITS/MS read-back, full correlations/time/solver/provenance; all families; retained non-gating performance; fresh independent manual/numerical/schema acceptance | All six approval sentinels are committed literal None; four rejected artifacts remain unchanged; disposal follows completed source logic |
| D1 | SCI-004 Sections 7/9/15; public solver currently rejects HEALPix/hybrid and non-scalar beams; accepted SCI-005 beam contracts remain authoritative | Reviewed successor public support contract; integrate point/HEALPix/hybrid full Stokes; stationary squint and applicable full-efield support through canonical BeamSystem/Jones | Public API and CLI tests; dense/sparse pixel measure, frequencies, tangent/receptor frames, component provenance; common direct oracle and two-tier predicates | Independent prerequisite decisions accepted for common direct/native basis, native conventions, beam domains, finite enclosures, operational enclosure and provider layout; interval translation, remaining native allocation qualification and source grants remain open; no public capability acceptance |
| D2 | SCI-004 Section 9 defers public backend routing; current dense public path is NumPy | Route dense contraction/synthesis through requested backend; enforce real precision/x64; stream budgeted transfer blocks; implement canonical worker scheduling | Public NumPy/JAX/Dask parity, precision rejection, allocation-before-budget rejection, scheduling instrumentation and worker invariance; independent computational review | Execution decisions round3 and provider layout round2 independently accepted for block lifetimes, worker assembly, casts and bounded planning/evidence metadata; remaining provider allocation qualifications and source grant stay pending; current public dense path remains NumPy |
| E | PERF-001 accepted CPU mitigations/readiness; accelerator measurements require an authorized compatible host | Inspect available resources; strict preflight; complete workload matrix and authentic device-memory/timing record if hardware is available | Exact clean source/locked environment, real device and synchronized complex128; independent accelerator acceptance; timing is non-gating | Three accessible NRAO GPU nodes report driver 555.42.06, below the required 580; another node was busy and the Ubuntu host timed out; compatible-host access requested while software work continues; PERF-001 remains ROADMAP with no accepted accelerator-performance record |
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
Primary accounting through `248f6c6` is
`provenance-audit/original-hunk-disposition-v9.json`,
SHA-256 `c86ab5b0634c45d88833e8055b4e2427314dfcc9b48956e77b91f76b62237643`.
The separately frozen `retrospective-duplicate-intent-027-052-055.json`
(SHA-256 `3d2aef754ed3d7936d770ab96ca115e50c6c977887498c68e6329718269a8c92`)
authenticates three earlier duplicate intents: 052 and 053 corrected,
054 retained. Together these records retain all 63 primary identities:
23 intents complete (11 retained, 12 corrected), and 40 remain pending at that
boundary. The separate `original-hunk-addendum-055-bef5750.json` (SHA-256
`0cab6aad058833bf0084002b37cd318d3a439aef4589f23fa36038d44229201c`)
records donor 055 corrected at `bef5750`, after `2461db4`: the explicit consumer
29/6/2 inventories and ordered passing-control joins complete its remaining
intent. Effective primary totals are now 24 complete (11 retained, 13 corrected)
and 39 pending. Donor 027's third nested reference and complete copy/source-relation
tests remain pending. All prior accounting bytes and 143 identities are preserved;
new supporting tests receive no additional donor credit. The 80 stopped-candidate dispositions
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
| Original-object red reader | `28cd74ec2718535bdb951368a296d5b15f6e0d7f`, pushed and live remote verified | Both independent reviews ACCEPT; eleven native Git controls and two original oracle pins pass; strict 54 unchanged. Filter/cache repair subsequently landed at `7eca44b`; detached checkout repair subsequently landed at `ccca81b` |
| D32/D33 source range enforcement | `732865ad1b0da50e96e439b41694e88e9d34bc26`, pushed and live remote verified | Both independent reviews and new-parent confirmations ACCEPT; 183 history tests and 611 combined evidence tests pass, seven future-E skips; scoped strict 598 to 545 with zero added diagnostics. Exact range-only three-path patch excludes the separate evidence size guard subsequently landed at `913d45c` |
| Complete evidence size bound | `913d45cbcf0311fb76ae24a4dc1e3f9cde0adea5`, pushed and live remote verified | Both independent reviews ACCEPT; strict less-than-104857600-byte guard precedes either output, reader bounds bytes before parsing; independent boundary and CLI tests pass. Scoped strict 376 existing diagnostics, zero added; full combined evidence 611 passed and seven future-E skips |
| Shared Section 14 JSON and strict readers | `bbd5f3fbe540628c8f8d5e29b39696fe7d002dcd`, pushed and live remote verified | Both independent reviews ACCEPT corrected round2; full 635 passed and seven future-E skips; 376 to 367 existing scoped type diagnostics, zero added. Six historical canonical artifacts remain byte-identical; production scientific serialization unchanged. Initial malformed-RSS wording and test typing failures retained and repaired |
| Raw oracle patch identity | `7eca44bac74e3033c761d5019e22b8268908e889`, pushed and live remote verified | Both independent reviews ACCEPT; twelve native conversion/cache controls and two original oracle pins pass; strict 54 unchanged. Raw whole-checkout and Python child isolation subsequently landed at `ccca81b` and `1d67b75` |
| Distinct source design bindings | `d90081bb6382dbf8c233b91fa0f2aac6923d5f11`, pushed and live remote verified | Both independent reviews ACCEPT; 145 serial dependency tests passed, strict 108 unchanged; original historical R nodes remain exact. D32/D33 callable authentication landed; evidence wiring subsequently landed at `9db7c1e` |
| Scientific segment primitives | `401d834831144e463e786fc11269cf2e723232b5`, pushed and live remote verified | Both independent reviews ACCEPT; 44 focused tests plus independent production-encoding and binary controls pass; strict 367 unchanged. Exact segment bytes, scientific JSON and fixed arrays only; stream, solver, owner and admission joins remain pending |
| Current D33 evidence binding | `9db7c1ec3d6381929a86c6825766c8ab0c97b64c`, pushed and live remote verified | Both independent reviews ACCEPT; 704 passed and seven future-E skips; strict 367 unchanged. Distinct authenticated D30/D31/D32/D33 roles; exact historical R references remain unchanged |
| Original detached checkout authentication | `ccca81bed7281dbbccad7a22fe1e83650541d242`, pushed and live remote verified | Both independent reviews ACCEPT corrected parent-directory guard; 29 combined controls pass, including two original oracle pins, strict 54 unchanged. Actual exact-R checkout raw-byte probe passes with original config/index/HEAD/registrations unchanged; canonical LFS pointer/materialization and raw modes/types authenticated. Round1 parent-symlink rejection preserved |
| Evidence original Git object/context reader | `248f6c68dff552f245d232280f4a4c1efc54f287`, pushed and live remote verified | Both independent reviews ACCEPT; 719 passed and seven future-E skips, strict 367 unchanged. Raw object/ancestry/tree readers resist replacement, graft and routing substitutions; original six historical artifacts unchanged |
| Evidence all-tracked source authentication | `3fe12f23c0a9f6264534a0c818cf31d59d821aea`, pushed and live remote verified | Both independent reviews ACCEPT; 735 passed and seven future-E skips, strict 367 unchanged. Preflight and postpublication compare every original tracked byte/type/mode, reject redirected parent directories and authenticate canonical LFS materialization; independent assume/skip/stat-cache concealment controls reject. A later same-tree HEAD-substitution gap was repaired at `e997f97` |
| Historical Python child isolation | `1d67b75ba64c6d1112e646fd6ff8697944c7f37c`, pushed and live remote verified | Both independent reviews ACCEPT; 12 author and seven independent native tests pass, strict 54 unchanged. All five children strip inherited Python/pytest routing while preserving exact source, original arguments and normal plugin autoload; composed full replay is recorded separately |
| Composed historical replay | Exact clean `1d67b75ba64c6d1112e646fd6ff8697944c7f37c` | Independent physics ACCEPT and computational PASS confirm the retained full serial red gate: 119 passed in 1106.68s; exact historical child produced two expected failures followed by three passing controls, with separate general control passing. Import assertion, original source bytes and retained JUnit authenticated; HEAD/config/index/registrations and checkout bytes unchanged. `composed-red-replay/after.json` SHA-256 `5d6111c6d3422e75f76ef2f40e9c3845210105d85a0b0a097490e72b48aaaefb` |
| Strict acceptance JSON | `2de1b83060b13fd4ebbaf2aaecd9bd8317efa382`, pushed and live remote verified | Both independent reviewers ACCEPT; full 183 passed and six future-A skips; strict 60 to 54 with zero added. Duplicate keys, invalid Unicode, coerced keys and unfaithful binary64 integers reject; original E/A canonical bytes remain exact |
| Exact scientific identity stream | `614d9683c55aa02b97213935a150b8ca266df641`, pushed and live remote verified | Both independent reviewers ACCEPT; full E 758 passed and seven future-E skips; strict 367 unchanged. Reconstructs ordered 24 common segments plus solver using scientific JSON and little-endian length framing; production linear/circular encoders independently match. Nested semantics and admission remain pending |
| Closed scientific solver snapshot | `bffd81283ca03adcad6a3eaa180ec769f4179022`, pushed and live remote verified | Both independent reviewers ACCEPT exact primary composition; 61 combined stream/solver tests pass; strict 367 unchanged. Exact 20-key snapshot and four family contracts authenticated; retained eight endpoints compatible. Caller must authenticate IERS and certificate joins |
| Frame certificate top-level structure | `0777731363a19fa4995783915f3c4e40007d61f9`, pushed and live remote verified | Both independent reviewers ACCEPT exact primary composition; 34 structural and 61 stream/solver tests pass; strict 367 unchanged. Exact 126-field structure and 125-field hash preimage only; explicit inventory is an indivisible schema slice. Nested semantics, budgets, owner/cascade and admission remain pending |
| Acceptance tracked-file lifecycle | `9bb48823eaa1f4737c0e9a19da7b9ef8030352f1`, pushed and live remote verified | Both independent reviewers ACCEPT corrected round3; full 208 passed and six future-A skips, strict 54 unchanged. Original E HEAD and tracked bytes/types/modes/LFS materialization checked throughout publication; final raw check follows status and exact output inventory. Original filter-side-effect probe now rejects, and restoring old ordering fails the new regression. Round2 REJECT and separate withdrawal of its initial computational ACCEPT remain retained |
| Characterization time reconstruction | `c8c90d22cba9f65ad1951e7444a0fa20e9f9eff5`, pushed and live remote verified | Both independent reviewers ACCEPT corrected round2; full E 857 passed and seven future-E skips, 27 focused tests pass, strict 367 unchanged. Exact time8 and temporal phase projection reconstructed with pinned offline IERS bytes; resource failures are typed and caller cache/config preserved. Original dual REJECT and 852-pass round1 retained. Complete phase/input and scientific-stream joins remain pending |
| Evidence original-source HEAD lifecycle | `e997f97e7a37b83175c6a787dcef5e6a2e3aef96`, pushed and live remote verified | Both independent reviewers ACCEPT; full E 863 passed and seven future-E skips, 24 focused tests pass, strict 367 unchanged. Raw guard checks original HEAD before/after reading, and prepublication/final output checks retain the preflight source SHA. Native same-tree HEAD substitution rejects; removing final HEAD check fails the decisive regression. Scientific owners and schemas unchanged |
| Nested frame manifest identities | `786413f2aa7bdcfc8f695b07aca48921359bf211`, pushed and live remote verified | Both independent reviewers ACCEPT and confirm equivalent full-index patch; 93 focused/related tests pass, strict 367 unchanged after retained five-diagnostic typing stop. Four exact nested manifest schemas, 17 constant rows, 32 retained endpoint manifest digests and 24 local joins verified. Geometry, source/environment authentication, ledgers, budgets, cascade and admission remain pending; additive helper is not wired into generation |
| Retained red case inventories | `2461db48ee2bf9f1586e62c96c4062f231acf158`, pushed and live remote verified | Both independent boundary reviews ACCEPT; 56 focused tests pass, strict 367 unchanged. Fixed ordered 29/6/2 identities, outcomes, disjointness and fixture projections; original consumer unchanged in this additive slice. Literal inventory accounts for most of the 625 added lines |
| Red command/control consumer joins | `bef57509f9031c63e14085fe25c3d713e20d0abf`, pushed and live remote verified | Both independent boundary reviews ACCEPT; 104 focused tests pass, strict 367 unchanged. Executable tool AST equals the independently accepted complete candidate after helper-name/documentation normalization; that candidate passed full E 1,016 tests with seven future-E skips in 329.13s. Original commands, ordered two-red/three-pass controls and six-key reference preserved; six retained direct partition tests pass separately. Incomplete development run and typing/format stops retained |
| Frozen-bound partition repair design | `90ef12e10c869b0928ad0afd51b9f7069729aa26`, pushed and live remote verified | Both independent round2 design reviews ACCEPT, followed by exact finalization and landed parent/blob/full-diff authentication; warnings-as-errors docs pass. Original timing-rationale REJECT retained. Finite partition repair and optional compatibility timeout are authorized; new measured transition relation and admission remain separate |
| Exact D34 edge authentication | `3563fa7dd1d42700a09db4caa7b6a1b78ec71fdd`, pushed and live remote verified | Both independent round2 reviews ACCEPT; full history 212 passed in 436.94s, strict 281 unchanged. Actual ordered reviewer identities and original D32/D33 pins retained. First full run stopped at 210 passes and one live-range boundary failure; original dual REJECT retained. Preparatory pre-D34 positive and inactive-D34 refusal passed; subsequent activation at `505ef3d` restores the live-current positive with all three edges |
| Characterization source inventory | `29e5e7e78ba6dec278e6208e21b792618b2b264f`, pushed and live remote verified | Both independent reviews ACCEPT and confirm unchanged candidate/patch at refreshed parent; 36 focused and 12 existing content controls pass, strict 367 unchanged. Exact source6/row3, sorted four-family inventory and original phase/set digest domains; original module ASTs unchanged. Additive content helper remains unwired; nested science, Git provenance and admission remain separate |
| D34 frame and workflow guards | `0fe8be4b3d62b62c80d1512ee2020f49573157ce`, pushed and live remote verified | Both independent reviews ACCEPT; 58 combined bridge/guard tests and 42 final guard tests pass, strict 281 unchanged. Original tool/test AST preserved; exact whole-frame repair AST and sole raw timeout replacement recognized; subsequent activation supplies ancestry and role checks |
| Coordinated D34 activation | `505ef3d9032f87750bfcf41324e27fdc04265497`, pushed and live remote verified | Both independent round2 reviews ACCEPT; full history 281, dependency 149 and evidence 1,062 pass with seven future-E skips; strict 756 unchanged across five files. Three ordered design edges, five distinct bindings, guarded frame/test paths and optional 17/18-path aggregate active. Original workflow-mode REJECT and cancelled partial run preserved; executable-mode regression fails before repair and 23 workflow controls pass afterward. No numerical repair or admission |
| Compatibility execution budget | `a18aa60d2cfcd1457d11a044c18e8ddecbdbe040`, pushed and live remote verified | Sole workflow change raises compatibility timeout from 45 to 120; exact committed byte guard, unchanged 100644 mode, full patch and landed verification-workflow role authenticated. All matrix cells, commands and other job limits remain exact; final exact-SHA compatibility acceptance remains pending |
| Frozen-bound initial partition regression | `b03258073233a9a8def68cfa71ccd4035210192c`, pushed and live remote verified | Both independent round2 reviews ACCEPT; original scanner produces exactly four intended missing-boundary failures and 14 passing controls. The separately committed S regression verifies the exact 14-boundary union, allocation extent and original owned endpoint identities; original hidden-upper-boundary REJECT retained. This is not another historical M3 red record |
| Frozen-bound initial partition repair | `aa933c24f4547a2303dd6a72f76287c1f60bd8dc`, pushed and live remote verified | Both independent source reviews ACCEPT the exact two-path patch; 35 frame cases pass on each of Python 3.11 and 3.12, and all seven public Python-3.12 integration cases pass in 1,056.18s. Ten added production lines validate exact Fraction endpoints and include their union; all other frame bytes and numerical budgets remain unchanged. Scoped strict diagnostics remain 185. Clean-source four-family measurements and a later reviewed numerical transition remain required; no observation admitted |
| Directed slab geometry guard | `083fa84d8b9050dd5baa423c52308130568429ae`, pushed and live remote verified | Both independent reviews ACCEPT; 27 controls pass on each supported Python version; strict 281 unchanged. Complete native AST states are explicitly qualified for Python 3.11/3.12; original bridge and other owners remain exact. Utility was initially unwired |
| Atomic D35 activation | `af8d5959ef84adbc6ec2d45203ceef2de89486dd`, pushed and live remote verified | Both independent round2 source reviews ACCEPT, with separate final gate authentication: full history 360, dependency 152, evidence 1,172 plus seven existing future-E skips; Python 3.12 activation 26 pass. Strict 756 unchanged. Five owners change atomically; complete source requires the approved repaired geometry and actual separate causal ancestry. Original round1 gate failure and reviews remain preserved |
| Direction-owned slab measure repair | `cf67c0ba27ea4a0b919f600053328d06c3d864de`, pushed and live remote verified | Exact dual-reviewed two-path patch `d5035e4957fe4bbd0bbcaeb1f1693d0d2ae140e93677f41c3473d56a16f82772`; 109 frame cases pass on both Python versions, including the five original causal failures. Scoped strict diagnostics fall from 120 to 118 with no additions. Exact Fraction unions stay in turns, per-direction ownership and canonical index order hold, and unchanged assembly converts outward once. Public integration, fresh measurements and scientific admission remain separate |
| Characterization receptor ownership | `4b93e7a46dc00965548d86836ef26f160ff770c2`, pushed and live remote verified | Both independent round2 source reviews ACCEPT; full serial E 1,248 passed with seven existing future-E skips in 674.04s, 25 affected controls pass on each Python version, strict 367 unchanged. Exact native floats, closed receptor schemas, ordered antenna joins, signed-zero-sensitive angle rules and four-family restrictions; reduced runtime digest remains carried, not reconstructed. Original integer-zero REJECT and interrupted run preserved; Python 3.12 full time/projection correction remains separate |
| Characterization input projections | `96e6b40bc32a451ff4ed804f0008d014a0fc4eb3`, pushed and live remote verified | Both independent round2 reviews ACCEPT; full serial E 1,169 passed with seven future-E skips in 483.84s, 107 focused and 124 dependency controls pass, strict 367 unchanged. Exact input14/phase26 outer schemas, content projections, source/solver/time joins and frequency/correlation/three-time-buffer bit checks; nested physical semantics, generation and admission remain separate. Original test-causality REJECT retained; repaired extra-key controls fail decisively when their guards are disabled |
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

## Historical boundary after direct conformance regressions

The test-only IERS correction is pushed at `272b56d`. Its explicit function-scoped
fixture authenticates the two exact locked test resources and restores the same
cached module owner on success and interruption. Production acceptance remains
unchanged and is tested separately: the default table passes and the newer table
is rejected. Both-runtime focused gates pass 145 cases; full serial default E
passes 1,255 cases with seven unchanged future-E skips. Strict diagnostics remain
367. Original harness failures and the initial Python 3.12 limitation remain
preserved under external `c1-time-runtime-correction-candidate/round1/`.

D36 is reviewed and pushed at `7752f04`, with independent own-header and actual
parent checks. The original exported patch and actual Git patch differ only in
hunk context placement; both reconstruct identical reviewed documents. The
failed equality check and uncredited build are preserved; the subsequent strict
documentation build passed. D35 remained executable authority until the separate
atomic D36 activation, since completed at `e33fb26`. Original D35 geometry and
scientific limits are preserved.

The separate causal commit `216afb1` adds six real-owner assertion controls for
five defects: operational fringe ownership, two model-classification cases,
least-upper radius, directed accumulation and omitted inner operational root
bounds. Each Python version retains six intended failures and three passing
controls. Supplementary isolated runs bind package/solver paths and source hashes
before and after each test in the same process. These establish causal defects,
not repaired production or a measured failure of any actual-family bound.
Evidence is under external `d36-causal-candidate/round1/`, including its original
classifier stop and separately authenticated `origin-supplement/`.

At that boundary, new captures were held until the reviewed source repair and
activation gates passed; the D36 measured boundary below records their completion. The structural full-row terminal traversal has independent dual acceptance
and bounded synthetic tests; independent numerical predicates and admission
remain incomplete. Instrument/selection semantics have a reviewed finite plan,
with implementation pending and beam semantics still separate.

## Historical production-v2 verification stop at a39ceff

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
wiring is committed. Remaining
schema/proof integration, reviewed observation admission and final disposal
precede the complete terminal-source gates and separate E/A acceptance.

The composed replay at `1d67b75` closes the later Git/Python isolation regression
gate. It does not replace terminal-S characterization, full source gates, or
fresh evidence/acceptance. Remaining nested certificate, phase/input, time/stream
joins and source-cascade validation must finish before the coupled evidence
schema cutover and observation admission. The four rejected artifacts remain
unchanged and all six approval constants remain literal `None`.

A subsequent native E-lifecycle probe found that a status clean filter can move
HEAD to another commit with the same tree before the postpublication raw check.
That implementation rediscovered HEAD and accepted the changed commit identity. The variant
that also changed tracked bytes rejected through the status inventory; the
confirmed gap is the original-source HEAD binding. The retained
`e-head-binding-audit/probe-head-only.py` and `.json` preserve the observation.
The bounded repair at `e997f97` retains the preflight source SHA through
publication and checks it before and after raw tracked authentication. Both
independent implementation reviews accept that correction after the full
863-pass/seven-skip evidence suite. No generated evidence has been accepted.

Independent terminal-scan inspection at `786413f` confirmed a separate operative
Section 12.1 violation: `frozen_root_bounds` is accepted and length-checked but
its values never enter the initial partition. All four authenticated retained
certificate/phase pairs contain 10,240 frozen-bound occurrences (5,133 distinct)
absent from the source's 4,193-boundary initial grid, across 2,560 directions.
Two source reviews found no later waiver; a separate exact-rational calculation
reproduced the counts without importing the solver or rerunning a scan.
The retained `d32-terminal-scan-feasibility/retained-boundary-comparison.json`
hash is `c8549b5e84864898459d9ad85c4257c46690d34384df08a713efb8d7c683ac6c`;
the independent root confirmation hashes to
`99ebcfbc4902866da5bc09576d597a17d24b9944bce1a73041b7ab7d8d2e7a17`.
This is an initial-partition comparison, not a new proof of the frozen roots.
Full certificate conformity and observation admission cannot advance on these
records. Reviewed D34 at `90ef12e` permits the finite numerical repair and its
separate S regression, preserving historical D32/D33 evidence and all budgets.
Exact edge authentication at `3563fa7` and guards at `0fe8be4` are composed by
activation at `505ef3d`, including the separate regression predecessor and exact
workflow modes. The optional timeout change is recorded at `a18aa60`; parent
CI `34118006232` exhausted 45 minutes in all six compatibility jobs, while
quality and parity passed. This cancellation is not compatibility acceptance.
The reviewed round6 external driver completed all four public characterization
families once at clean `e7623fb`, with exit zero, unchanged source checkpoints,
successful byte verifiers and passing returned frame certificates. Its final
inventory is `d34-measurement-driver/execution-round6/capture/final-integrity.json`,
SHA-256 `06ceb7e2faa8940ca02c1631aab0bc2f5604436210606e8a4efa29d0027c2d91`.
These are unadmitted measurements, not performance or M3 acceptance evidence.

Independent scalar and three-family comparisons confirm unchanged visibility,
flag, weight, time, frequency and channel-width arrays, unchanged phase values,
and unchanged direct certificate arrays and numerical error-preimage values.
Operational horizon bounds and counts change; the existing source-hash-only
transition therefore remains insufficient. Scalar's old scientific reconstruction
and separate certificate probe retain their original qualifications.

The independent `d34-initial-partition-proof/result.json`, SHA-256
`d1da111fcd5d540bd15209aef27548bc2b2bb729dff495be6da43f855e351805`,
checks all 9,326 actual boundaries per family against the exact required union:
4,193 base points plus 5,133 distinct frozen-root endpoints. All 753,130 retained
centre values match their initial-stream projections bitwise. The accepted bounded
terminal parser then verified the complete scalar and point-I streams. Full-Stokes
and circular reuse the point-I structural result only after independent full-file
and complete-contract byte equality. `d34-terminal-parse-current-round2/completed.json`
is SHA-256 `df521178e3ace078e1d544b049e19503be28c8d909ef1c9da824f72dbea79484`.
Structural proof does not establish celestial-coordinate values or admission.

Historical scalar reconstruction stopped at a string comparison of `0/1` with
`0`; all 4,193 exact rational values and 49 centre vectors match. Its original
execution remains failed. Separate full terminal parsing and independent retained
source/input/root/restoration joins pass; `d34-historical-scalar-postvalidation/retained-scan-proof-summary.json`
has SHA-256 `6a8619886b57a973e65e8d47e3c82e57a5885418ca1aa0ab1938b4f0e3a18934`.
Point-I, full-Stokes and circular historical scans completed under the corrected
driver. No historical job completes a certificate or solve.

The comparison exposed B4's measure-unit and ordering defects. Its preserved
`d34-slab-measure-diagnosis/result.json` is SHA-256
`8855dad281bd1d2b892a670867e582b10aa7c3ff7f9d75ba9a9b7e7fd3a2a4dd`.
The reviewed external proposal defines the exact sum of direction-owned slab
unions and one outward conversion, preserving every budget. D35 now provides the reviewed operative grant at `dd26e04`. Source correction and strict executable activation subsequently landed at `cf67c0b`
and `af8d595`. Public numerical gates, fresh corrected-source measurements,
certificate conformity and numerical transition/admission remain required; no
retained certificate is rewritten.

D35's exact two-document landing follows fresh dual ACCEPT of round 2; the
original round-1 provenance rejection and its bounded sequencing repair remain
preserved. The separate `5790241` regression completes real certificate assembly
with explicitly synthetic unrelated providers. Its five assertion failures expose
turn/radian units, pair order and exact outward conversion; the zero/ownership
control passes. Both independent reviews accept that causal inventory, not a
physical certificate. A reviewer selector also ran all 35 original frame cases,
which passed; the actual expanded inventory and selection mistake are retained.
`d35-causal-regression-candidate/manifest.json` has SHA-256
`015e70232e046d9b3051817771b9c5e575a81e8b5ae3e14d3fcc48949a881c4d`.

Preparation at `368821d` authenticates pending D35 through the unchanged full
raw-object/own-review checks and separates historical D34 consumers. It changes
neither the active three-edge registry nor current completeness or D31 bridge
rules. All 161 affected cases pass; scoped strict diagnostics remain 281, while
the causal test file retains its original 42. Both source reviews ACCEPT.
`d35-history-preparation-candidate/manifest.json` has SHA-256
`6fcdb08a498d3f815071073fd149bc691e7a639f5ca9f04c921ddb0d78b256c3`.

Read-only one-minute Slurm metadata allocations found Quadro RTX 5000 GPUs on
herapost009/herapost010 and an NVIDIA L4 on nmpost037, all with driver 555.42.06.
herapost008 was unavailable because its resources were busy; no driver was
observed there. The separate ubuntu24lts SSH connection timed out. No matching
inventory jobs remained afterward. These observations are hardware metadata,
not selected-JAX-device, locked-environment or performance proof. The transcribed
record `perf-resource-refresh-20260907/observations.json` has SHA-256
`78092dbe5b7717fae872a0217c65a9e788dfd5d1c6aac9284733cf73df73a740`.

The D35 activation completion records under
`d35-atomic-activation-candidate/round2/` authenticate all four successful serial
commands, exact source hashes before/after every gate, and the unchanged protected
numerical patch. `complete-gates-results.json` hashes to
`6a0d0ad3500454a967ee17d1f1eacfa4d7b26ce9677c95ba431d719acdd357e0`;
the independent final gate review hashes to
`bd66f82c579280ee627b32244cc87ca552548e62d1e000694d7d2b77ea8facd1`.
The numerical candidate manifest is
`6086e2008d839671a062670b0058667f0ef54a3271405156054a72925f1111e3`.
Both commits reproduce their complete reviewed patches, and the checkout was
clean after the geometry commit. These outcomes do not complete terminal S or M3.

C1 instrument composition is now connected to the input validator; beam,
sky/direction/transfer and convention/frame checks, then the coupled schema cutover,
remain open. D36 public gates and fresh clean-source capture are complete;
conformity and a further reviewed numerical admission relation remain open.
The bounded profile reader remains in source/construction preparation; it has not
acquired actual capture profiles or supplied numerical proof.
No whole original donor is retired by these scheduling decisions.

Read-only F1 diagnosis confirmed two catalog defects for later scoped repair:
MALS DR1 combines spectral windows with different actual frequencies while the
loader assigns one reference frequency; all four GLEAM entries select peak
`Fpwide` (Jy/beam) as integrated Jy. Independent review selects `Fintwide` in Jy
as the minimal GLEAM repair, explicitly described as the 170–231 MHz integrated
wide-image estimate used at nominal 200 MHz, with the existing spectral-index
policy. No catalog source has changed. Actual VizieR metadata resolves the
suspected MALS/LoTSS coordinate alias discrepancy. The official LoTSS DR2 MOC
omits blanked bad-facet regions, so it does not establish an exact footprint;
unknown coverage remains unchanged. Retained schemas and review records reside
in `f1-release-footprint-research/` and `f1-gleam-flux-research/`.


## D36 measured boundary and current prerequisites

Activation `e33fb26fdff39e2b3f103c8d46f132da11fcbdc7` and numerical repair
`c023036ca70871ab901471dc204f36a4d89b6c4f` are pushed. The activation passed
all six serial history/dependency/evidence gates; the numerical candidate passed
17 focused cases on each supported runtime. The subsequent public gates passed
seven integration cases and one characterization case on each runtime, with no
skips. The complete public-gate authentication is
`d36-public-gates/c023036/root-complete-auth.json`, SHA-256
`269b8a51c7ee9d006a56632da3ec65449889b4c04c428f570a3297936f018a7d`.

The existing capture process completed once, without retry, for all four families
at clean `c023036`. Its final record declares completion with no original failure
or final errors, and explicitly withholds admission. Independent streaming hashes
matched all 901 listed artifacts (7,442,685,965 bytes), exact inventory, source and
helper identities, unchanged environment records and preloaded import identities.
The root record is
`d36-measurement-driver/execution-round1/root-finished-capture-auth.json`, SHA-256
`d51f61effbd8f47b00ac826aebcb13133b5a604ae1b734970649d57efaf258da`.
This checks byte integrity and retained metadata joins; it does not independently
decode numerical payloads or prove terminal conformity. Historical captures and
failed attempts remain unchanged. Source, capture, admission and M3 acceptance
remain separate outcomes.

The M4 prerequisite review records are external under the recovery root:

- Native/direct basis round2: `m4-public-support-decisions/round2/`.
- Native convention/materialization round1: `m4-native-conventions/round1/`;
  root disposition SHA-256
  `da3d22130ab4fb644247ac833d405b15dff666ed6b57fa321ec737ce0d0db041`.
- Beam domains and enclosures: `m4-beam-domain-decisions/round1/` and
  `m4-beam-enclosure-plan/round1/`.
- Execution decisions: `m4-execution-decisions/round3/`.
- Operational enclosure: `m4-operational-enclosure-plan/round1/`; root disposition
  `6118c8408ad855e69fe9b8c792c99facfca8c57e8b5fef6604966e50ecb74d97`.
- Provider layout: `m4-provider-layout-plan/round2/`; root disposition
  `0d0af8158cc113a5037e3793cf8b3f5c935695cec730c815b6805bc3ada82ac5`.

These prospective decisions do not establish implemented public support or
numerical qualification. The bounded codec and finite FITPACK work recorded below
advance individual prerequisites; operational whole-cell bounds, provider internal
allocations, source grants and actual public scientific tests remain required.
The exact-`c023036` CI run
[34180339931](https://github.com/RRI-interferometry/RadioSim/actions/runs/34180339931)
ended with all six compatibility jobs cancelled after their explicit two-hour
limits. Backend-parity and quality jobs passed. Both Linux runtimes reported the
two production-v2 preimage tests FAILED before timeout; their missing terminal
tracebacks leave the cause unresolved. The incomplete macOS logs do not establish
full-suite success. The original diagnosis remains in
`ci-c023-34180339931/round1/report.md`. The recorded pending observation for
`74830bc5f30330a3bbf4124f33ef467447cf1f7c` run
[34192091050](https://github.com/RRI-interferometry/RadioSim/actions/runs/34192091050)
remains historical; no terminal result for that run is inferred here.

The serial diagnostic reporter `c0be678e362570d10d44e29c28885119d947f2d0`
and workflow `ec46f3fa536b35a02f41d3aa7981beb535eae463` initially refused both
Linux jobs before test bodies with `collection configuration drift: ini` (JUnit
zero tests). Lifecycle correction `c238b15e6cc8dc2a64b3cd7d7ee075d5ca788b7c`
passed 29 synthetic cases on each runtime. Actual run
[34196892621](https://github.com/RRI-interferometry/RadioSim/actions/runs/34196892621)
then executed both selected nodes without infrastructure refusal: Python 3.11
failed both with `InvalidResultError: invalid characterization result antennas`;
Python 3.12 passed both. The second 3.11 failure received that antenna error before
its expected identity refusal. Complete raw failures, settled configuration and
source authentication are retained under
`ci-serial-diagnostic-execution/{ec46f3f,c238b15}/round1/`. Per-row versus batched
coordinate arithmetic is an investigation hypothesis, not a confirmed cause or
repair. These diagnostics neither explain the missing original c023 tracebacks
nor establish complete CI success; complete CI remains open.

## Subsequent bounded implementation and evidence checkpoints

The D36 table builder has independent acceptance for the pure, unwired fixed
registration declaration and arithmetic from authenticated derived records.
`d36-profile-registration/table-builder-round1/root-disposition.json` binds the
54-artifact manifest `91aaacb768667d4b7d251870e66148175543b225d5fc88ef0281821de2594af7`.
Both runtimes reproduce 148 historical formula controls and identical declarations;
36 synthetic controls pass on each. The completed four-certificate census observes
85-bit numerator/denominator maxima, which fit the unchanged 256-bit domain and
change no fixed proposed capacity term. Detailed schema owners are bound to the
exact retained inventory. This does not establish a generic hostile parser, reader
source-fit, applied profile/capacity, payload acquisition or scientific admission.

C1 instrument checkpoints are committed in the following sequence. Checkpoint 8
composes the reviewed primitives and connects instrument validation to the input
validator. Other C1 owners and the coupled schema work remain open.

| Checkpoint | Commit | Bounded result |
| --- | --- | --- |
| 1 | `09e2b6f38863f5dbe721b75067854cb437fe1b90` | Test-only independent instrument/selection fixture and public-owner joins; 218 selected cases pass per runtime. |
| 2 | `272a2afa4978644ddc1f72f7d81cd38beebaae1c` | Exact native binary64 scalar/vector primitives; 45 focused cases pass per runtime. |
| 3 | `d1239bffcdc2abed1752e44db60e09dbb59756b5` | Ordered antenna identity, ENU, diameter and provenance primitive; 58 focused cases pass per runtime. |
| 4 | `74830bc5f30330a3bbf4124f33ef467447cf1f7c` | Outer instrument closure and original fingerprint primitive; 33 focused cases pass per runtime, retaining later geodesy obligations. |
| 5 | `59450dc83146fd6ea487afdb95a25e16fe43b385` | Exact original and rebuilt phase-site reconstruction; 34 focused cases pass per runtime. |
| 6 | `cc54a2c192bb6cb642be1429def1155c85da42f2` | Exact selection criteria, counts and ordered native baseline IDs; 32 focused cases pass per runtime. |
| 7 | `31305712742844678985636e4f3c1eb79ba9d91b` | Phase rows and original batched ENU rotations, with independent public-owner joins; 20 focused cases pass per runtime. |
| 8 | `5ac4a23594647be81f3c31347fb74eb3deb486fd` | Complete instrument composition and input-validator connection; 469 selected cases pass per runtime. |

The corresponding `c1-instrument-checkpoint{1,2,3,4,5,6,7,8}/round1/root-commit.json`
records bind reviewed patches, commit outcomes and preserved ambient changes.
Scoped strict diagnostics remain the same 367. Checkpoint 8 retains exact
reference predicate composition; 15 literal test variants are explicitly qualified
as predicate-equivalent coverage, not verbatim case replay. Its 469-case gates do
not constitute a full E gate, completed C1, or observation admission.

The private, unwired native payload codec is committed at
`5ed697ffdfe86efd57c349f0384c6adddfe669a6`. Identity tests at
`ac08dd3262e9b3e87926584cafde4b1d937d83a4`, constructor-aware layout tests at
`52faa0e49340024950ea73113f91d975cfaa086f`, and hostile-input tests at
`47a6b4843c437fe8dc7b6d814f79176295cd87ca` extend its finite coverage. That checkpoint's
focused module passes 57 cases on each runtime with zero scoped strict diagnostics.
These controls distinguish stored values, signed zero, aliases, order and endian
representations, and reject invalid owned metadata. They do not wire loaders,
materialize maps, establish physical equivalence or grant public M4 support.
Test-only scratch instrumentation `be541fb1c62575a7fd8e4fbf867e2c7826cee9fb`
extends that module to 59 passing cases per runtime. Transparent iterator/hash
observers establish finite traversal and conservative parent-capacity accounting
for the declared sparse/dense fixtures; they do not prove hidden native allocation,
all layouts or RSS. Exact records reside under
`m4-native-value-binding-code/round2/` and
`m4-native-binding-{identities,layout,hostile,scratch}/round1/`.

The private canonical stored-native identity helper
`39c1f2f92f8e8d760fff2a88b16363b42750e052` adds a twelve-field record, fixed stored
operation and immutable byte sidecars. Three focused cases pass per runtime,
including independent-oracle coverage, with zero scoped strict diagnostics. It
certifies no earlier conversion history and changes no public owner or loader.
Test-only `e58613b3a3b142f9ab96bfb85261f11285db8e04` extends the module to 18
passing cases per runtime, covering corrupted records/sidecars, stale backing
aliases and null/zero/empty domains. Exact-type and bounded Q/U iteration controls
remain open. Scope and commit records are retained in
`m4-native-identity-materialization-code/round1/` and
`m4-native-identity-hostile-tests/round1/`.

D35 reader construction remains source preparation. The original discovery and
call-transport qualification stops remain in
`d35-profile-reader-construction-preparation/`. The subsequent 96-file/runtime
allowance is solely a finite source-discovery grant (2 MiB/file, 32 MiB/runtime),
not construction execution or a proof of W=1280. The prepared four-slot amendment
and allocation-source work have not produced a qualified aggregate lifetime bound;
raw496 and complete pair caps remain unchanged. The 148-case construction roster
has not run, and no actual capture profile has been acquired by this reader.

FITPACK execution is accepted only for four finite public-constructor observations
and four Python pre-native refusals across the two recorded environments. Maximum
input z size was 70 elements; returned base payload sizes were 96 and 744 bytes in
each runtime. `m4-fitpack-native-execution/round1/root-disposition.json` binds
manifest `5251316c4e7d53f0a91f5fc2ddc5c857c35ed5b258d1bd1e76c11e07a1cc5449`.
The original RECORD nonjoins remain preserved. This supplies neither an all-size
workspace/ABI/allocator/RSS proof nor numerical enclosure, production grant or M4
admission. No historical paragraph, failed attempt or original identity above is
reclassified by these later bounded outcomes.
