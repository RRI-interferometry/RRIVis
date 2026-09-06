# SCI-004 m-mode forward-simulator design gate

**WP-9 design-gate candidate — 2026-08-11**

**Phase-0 bounded correction — 2026-08-21.** The candidate introduced at
`978fef6ddd885355dd06f1deeb04aa2927626d71` received its two required fresh
independent Phase-0 reviews on 2026-08-21 — physics/governance and
computational, both `REJECT` — against pinned candidate bytes
`sha256:01f8c56a32e3f649c576393d53b3ad29967b9c4b69bc2ba82c33ff17312a5591`.
The reviews confirmed the Section 5–6 polarization, rotation-sign, and
harmonic algebra by independent re-derivation and confirmed every Section 2
live-source claim against `257056b`; the recorded blockers were bounded.
This correction resolves every recorded blocker and advisory: it makes the
Section 1/4.2 admissible-scale envelope explicit; replaces the Section 13.2
M1 live-tip WP-7 replay — unsatisfiable for every descendant of the
accepted `v0.4.0` release commit — with the frozen-historical-descendant
replay; adds the Section 13.7 bounded-correction, supersession, and
disposal protocol with the `D0`/operative-`D` authentication in
Section 14.0; and lands the recorded exact-literal repairs (Sections 5.3,
11, 12.1, 13.5, 14.2, 14.3, 15). It is design-only: it implements no
solver, accepts no phase, and does not close the register row. Its exact
pre-landing file bytes
(`sha256:7beb148eda543f21e56c8720f11b51e6e7cfd3f593c013609f0136200437a6aa`)
and parent-relative diff
(`sha256:7d4dc4af03f384ac04deb29411a1c32a1b1916251bd9d903b6bb5e528b3f86c7`)
received separate independent reconfirmation reviews on 2026-08-21 —
physics/governance and computational, both `ACCEPT`, each independently
reproducing the frozen-descendant WP-7 replay byte-for-byte — with one
deferred non-blocking advisory recorded for a future correction: Section
13.1's opening sentences still carry the pre-correction single-fixed-`D`
phrasing and should cross-reference Section 13.7's operative-`D`
definition. This correction's landing commit is the operative `D` of
Section 13.7.

**Bounded correction #30 candidate — 2026-09-07 (authenticated review
recovery and finite main-only phase ranges).** This prospective correction
supersedes only corrections #28/#29's unfulfilled direct-child authoring route,
Section 13.7's universal header-finalization assumption for the two exact
historical exceptions below, and the single-commit R3/S3 requirement. All
historical source, artifact, parent and review observations remain immutable.
M1/M2 acceptance is unchanged; M3 remains rejected/not accepted. Neither this
correction nor a prerequisite repair accepts production or closes SCI-004.

The new programme's status-only bridge is
`d432bcb50f60c880aca3d3e599786b9ebe62fa1c`, sole parent exact D29
`cfc9b10d655a4d9bedbd7d7750c4743f504bbaf9`, touching only
`docs/development/completion_ledger.md`. It accepts no phase. D30's sole parent
is that bridge. The earlier D27/D28/D29 edges are authenticated exactly as they
occurred, not rewritten as edges under this new rule.

**Two finite historical finalization exceptions.** Original independent
reviews exist for D28 and D29. Both commits landed the reviewed bytes, including
the pending-review sentence, without Section 13.7's subsequent final header
sentence. D30 records recovery of those original reviews today; it does not
pretend either historical header contained a finalization it lacked. The
exceptions are keyed by the complete commit identity, never by label, date,
missing metadata, or a general permission to skip review.

| Commit | Exact parent | Memo SHA-256 | Companion ledger SHA-256 | Complete parent-relative binary/full-index diff SHA-256 |
|---|---|---|---|---|
| `67da2b818b89511df8476b7010230c65d6cb6a75` (D28) | `82fb0773890870a6fb90b3ed9b8065df89919a84` | `043ae81935e1b3161a6e0bb60922ebe13048b4f4ca6c49c1f2a31521eb914e2c` | `fbf6eec8c2eee287d36e8fcb117fb2da8e53ac8e2d1e30d337640442488bd014` | `93b042d42892a6063b52ab2090871d8985f3e9ef47a190f2436c729198465795` |
| `cfc9b10d655a4d9bedbd7d7750c4743f504bbaf9` (D29) | `67da2b818b89511df8476b7010230c65d6cb6a75` | `a39e9d662450a9776c88c5d322fecdbb4c739ac82b672e8be2f3019cad95d40b` | `fa860e904fce6dafe92cdc0eabfdae43b1714f131cc6ea9bb01664ee0e72e6ad` | `c589c3feae17a4dd4e4c8a1d1ef15197a4762968a84cd338bb8a1ef78d4162a4` |

Both commits touch exactly `PostTier8RemediationPlan.md` and this memo.
The recovered final review texts have these raw UTF-8 SHA-256 identities,
without added newlines or normalization:

| Design / responsibility | Original reviewer session | Final message SHA-256 |
|---|---|---|
| D28 governance | `01a05d05-9968-7153-a52c-985a43ec5674` | `325dd1e37c1f7ce523f533784da727b7e364ede40467e1fa45a193421d894fc4` |
| D28 oracle/provenance | `01a05d05-d622-7302-8ed1-06979a22628f` | `41e7532617840af20493041da15b4b41dd6832aececa9f38c595b0f298c1e139` |
| D29 governance | `01a06485-8a7a-7412-b85c-61e008764e3a` | `fa7c6ae5cd837042e94d3e6f58c27d38959287038013ed8fd41f6efac75b057c` |
| D29 evidence/provenance | `01a06486-3ddb-78d3-98f3-a40f37f8c3c5` | `572d442592b91f50c38ff94b234ba345b59260df9b7689b4d64ed820367a1ae1` |

All four original verdicts are ACCEPT. The prerequisite must retain the
review-recovery record at `docs/development/sci004_review_recovery.json`,
authenticated against D30's frozen recovery-record SHA-256
`eb9b00fcdb7703cb40982bc7e445ba6e042fb45ca26bd0515387dfb644975d54`.
It contains source session/message identities, exact archive line locators,
readable owner continuation prompts, pin-producing tool responses, and final
response items. Original encrypted dispatch bodies remain encrypted; this
record makes no claim to reconstruct their plaintext.
In particular, the short D29 governance final must be joined to its own
session's candidate pin checks; its verdict alone is insufficient. Validators
must authenticate all record bytes, both Git parents/path sets, both memo and
ledger blobs, both complete diffs, and the four original contribution digests.
Recovered reviewed bytes equal landed bytes only for these two exceptions.
The ordinary pre-landing-versus-landed inequality remains mandatory elsewhere.

Replace positional inference with a total SHA-keyed review map: its key set
must equal the complete enumerated design continuation set, without duplicates,
unknown entries or omissions. Ordinary entries keep their exact header pins and
strict comparisons; these two entries require the stronger recovered-record
joins above. Do not truncate a sequence, remove strict zip semantics from an
ordinary pairing, manufacture a review pin, or use a preservation digest as a
review digest. D30 and future corrections retain Section 13.7 dual independent
review and final-header recording, with no new missing-finalization exception.

**Prospective authoring and phase ranges.** All implementation and commits are
made on main in `/Users/kartikmandar/MacProjects/RadioSim`. Exact-SHA detached
checkouts/exports are permitted solely as immutable verification fixtures.
The verified external recovery bundle may park/reconstruct the identified
original drafts using explicit paths. Before source acceptance every original
hunk must have a retained/corrected/superseded/obsolete disposition with evidence;
no entire overlapping candidate file may silently replace the primary draft.

The prospective sequence is D30 -> P -> R3-range -> S3-range -> E3 -> A3.
P is a finite prerequisite-maintenance range. Each commit has one of these
roles, authenticated from its actual paths and content, never its message:

- `review-recovery`: only the frozen recovery JSON above.
- `frame-regression`: only `tests/unit/test_core/test_sci004_frame.py`.
- `frame-context-repair`: only `src/radiosim/core/mmode/solver.py` and/or
  `src/radiosim/core/mmode/frame.py`, restricted to making the operational
  horizon/sign evaluations use the same authenticated installed IERS context.
  A reproducing ambient-table regression and two independent scientific/code
  reviews are required. No root/slab membership, transform, tolerance, budget,
  accepted fingerprint or historical source byte may be changed under this role.
  If diagnosis requires a different scientific change, review a successor
  contract before implementing it. An unused frame role contributes no commit.
- `status`: only `docs/development/completion_ledger.md`, factual progress and
  exact existing commit/evidence identities, never acceptance or contract text.

R3-range comprises small coherent prerequisite/oracle/validator commits whose
aggregate scientific path inventory is exactly D29's five-path R3 grant, plus
`tools/sci004_phase3_history.py` for strict range/recovery authentication and
`tests/unit/test_sci004_phase3_history.py` for its hostile mutation tests.
Status-only commits may appear explicitly in P or R3-range. R3 contains no
production edits or new historical red observations. The terminal R3 tip is
frozen by exact SHA in the first S3 implementation's evidence-tool constant;
that assignment is source metadata, not an approved-evidence sentinel.
R3's design binding is exact D30. Original fingerprint R3 remains exact
`a65c53a46e84f63c163c5ad15fba8645df33d1d2`; its two-red/three-control replay and
all three disjoint 29 + 6 + 2 red records retain their original source/design
bindings, especially D24 and D25. Replay must set explicit historical source
path and `PYTHONNOUSERSITE=1`, and assert `radiosim.__file__` lies beneath that
exact historical tree before observation. Wrong imports, collection failures,
frame failures, or absent nodes do not substitute for expected failures.

S3-range comprises coherent source commits restricted to D29's five modified
paths and four authenticated rejected-artifact disposals, plus factual
completion-ledger updates. Each disposal is isolated from unrelated work; all
four outputs must be absent and all six approved evidence/acceptance constants
literal None at terminal S3. R3 oracle/history files cannot change in S3.
The terminal S3 tip must be globally clean, contain the complete reviewed v2
production/input/evidence contract, and discharge the full current-phase gates.
Intermediate source slices require relevant focused tests and static checks;
they cannot claim phase acceptance. E3 generation runs only at that exact clean
terminal tip and retains its exact source identity. The underlying production
v2 schema, nine-key family record and scientific numerical predicates are
unchanged by this correction.

The M3 evidence envelope adds exactly one top-level `phase_ranges` object with
keys `prerequisite`, `red`, `source`. Each value has exact keys `base_sha`,
`terminal_sha`, `commits`. `commits` is the complete ordered first-parent list
for the exclusive-base/inclusive-terminal range. Each entry has exact keys
`sha`, `parent_sha`, `role`, `paths`, `parent_diff_sha256`; paths are sorted,
unique repository-relative touched paths, and the digest covers raw
`git diff --no-ext-diff --binary --full-index <parent> <sha> --` bytes. Every SHA must be a
real single-parent non-merge and each parent the preceding tip. P's base is
D30, red's base P's terminal, source's base red's terminal; source terminal
must equal the envelope's `source_sha`, red terminal its `red_commit_sha`.
An empty P has terminal equal base and empty commits; red and source are
nonempty. The evidence validator recomputes every entry and role, full range
coverage, and aggregate path/content authority. Missing, extra, reordered,
wrong-role, wrong-parent, wrong-design or wrong-digest entries reject. All
non-status R3/S3 commits use `red`/`source` roles respectively, with source
artifact-deletion entries using `disposal`; each role preserves its exact
content restrictions. No status commit is silently skipped.

E3 is still one sole-parent direct child of terminal S3, with only its existing
generated artifacts/reproduction/performance and exact evidence-binding
assignments. A3 is still one sole-parent direct child of E3, with only its
existing independent acceptance artifact/bindings and authorized status prose.
A ledger companion hunk in E3/A3 is permitted only to report those same
outcomes, not change authority. An E3/A3 logic/source edit remains invalid.
The active approved-parent first-child rule still selects current attempts;
rejected predecessors are separately authenticated and never eligible. A3
must independently rederive the new range/recovery joins as well as all
existing numerical/schema/manual rejection checks. E3 generator success is
not acceptance. A later whole-row review remains separate.

This finite range rule applies to the reopened M3 only. Public HEALPix/hybrid,
stationary non-scalar beam/Jones, real backend/precision/memory/worker wiring
require a separately reviewed successor contract and public execution evidence.
PERF-001 remains ROADMAP, and no accepted accelerator-performance record exists.
SCI-004 remains ROADMAP; closure C stays locked. Original numerical thresholds,
accepted artifacts and scientific conventions are unchanged.
**Review status: fresh independent physics/governance and computational/
provenance reviews pending against identical complete candidate pins.**
**Review verification — 2026-09-07.** Fresh independent reviewers
`/root/d30_physics_review` and `/root/d30_provenance_review` each returned exact
`ACCEPT` against identical complete round-1 candidate bytes: memo
`sha256:bd1040b817360cba11261b5a0aa68453bba0b6e55cd9d7a86e6cf3764fd141a1`,
companion ledger
`sha256:d0d15132cc9675b755be47dbf27aba3d5f6ed0af63be380a653fa8593fafb438`,
and full parent-relative binary/full-index diff
`sha256:6180f9ecc34cc187d17d61b749de5102d0e49d904aaecb272e19449a655cae2c`.
Both independently authenticated the original four reviewer sessions and all
20 retained raw archive line records against their candidate pins. Only this
final review-verification record and the companion verdict completion postdate
the reviewed bytes under Section 13.7. This landing becomes operative D30;
it accepts no production phase.

**Bounded correction #29 — 2026-09-03 (historical red-record design
binding).** The D28-based future-R3 attempt stopped fail-closed at pre-review
gate 9, the complete serial unit suite. Gates 1--8 passed, but those results
did not authorize candidate pinning, independent review, staging, or commit
after gate 9 returned raw exit `1` with exactly
`3 failed, 7435 passed, 2 skipped, 30 warnings in 597.61s`. The three failures
were
`tests/unit/test_sci004_phase3_evidence.py::test_the_generator_authenticates_and_joins_both_red_records`,
`tests/unit/test_sci004_phase3_evidence.py::test_the_artifact_validator_authenticates_the_fresh_r3_and_both_inputs`,
and
`tests/unit/test_sci004_phase3_evidence.py::test_the_generator_rejects_a_supplement_not_contained_in_fresh_r3`.
The first two raised
`SCI004_M3_EVIDENCE_PREFLIGHT: the post-source red record does not bind the operative D`;
the third was stopped by that same earlier preflight check before reaching its
intended `does not contain` rejection.

All three failures have one deterministic provenance cause. The immutable
correction-#24 post-source record freezes
`design_sha = 4d507bf1333ccaa4c8beec3815370ba0f6043bb2` and
`pre_fix_source_sha = a61526d686ab768f05ecffa80cfd6223d4ee4c62`,
but the committed D28 evidence tool compares that historical `design_sha` to
`_design_sha()`. The helper correctly reads the current phase's
`APPROVED_SCI004_D_SHA`; clean D28 still freezes D26
`93321d331e4f6442d39fe79588be6f05ad4bee42`, while the stopped candidate
correctly advances that current binding to D28
`67da2b818b89511df8476b7010230c65d6cb6a75`. D24, D26, and D28 are distinct,
so neither current binding can equal the immutable record's historical D24
binding. This is a historical-record provenance defect in the committed
evidence tool, not a scientific failure. It does not change or re-observe the
required two-red/three-green characterization partition.

The D28-based three-path candidate at
`/Users/kartikmandar/MacProjects/RadioSim-sci004-r3-characterization-oracle`
remains unreviewed, unstaged, uncommitted, and preserved. Its complete
D28-relative binary/full-index diff remains
`sha256:ff31a389880d48dbdb7e8063bdb4de488f3931a751e492bac121d600f959abc1`;
its working-tree file SHA-256 identities remain
`f579edd63fe19c8f0ae6b9ec776cc9384ad086627d84bc4c6becb86f6019812f`
for `tests/characterization/test_sci004_mmode.py`,
`f4b77629f9f480ea6e66d0afe27a0a0e3a96957db8b46c8b8f129d005a33f8a1`
for `tests/unit/test_sci004_phase3_dependency.py`, and
`d5edf18cdacffc295b180c6a4fca61dfc28447fbd043ce79cb3ff09f7653def3`
for `tests/unit/test_sci004_phase3_red_failures.py`.
Correction #29 does not modify, approve, or repurpose those bytes. D28 remains
immutable Git history; only after D29 lands does D28 become a superseded
design and D29 become the operative design. Correction #29 is design-only and
accepts no phase.

The future evidence implementation must keep five identities distinct and
authenticate each by its own governed role:

- the new evidence envelope's operative design is exact D29;
- the immutable correction-#24 post-source record's design is exact D24
  `4d507bf1333ccaa4c8beec3815370ba0f6043bb2`;
- the immutable correction-#25 fingerprint supplement's design is exact D25
  `ca3c37171aaaeec175b5ad72d324957762303853`;
- the original fingerprint R3 is exact
  `a65c53a46e84f63c163c5ad15fba8645df33d1d2`; and
- the future post-D29 validator/oracle/evidence-binding R3 is unknown until
  that future commit exists.

`_design_sha()` continues to return and authenticate the current phase's
frozen operative design, exact D29 at the future R3 and its successors. The
correction-#24 record is instead authenticated against its own exact D24
design, and the correction-#25 record is authenticated against its own exact
D25 design. Neither historical record may be compared to `_design_sha()`.
There is no newest-memo search, current-tip substitution, or use of a design
SHA as a red-commit SHA. No historical JSON byte is rewritten to name D29.
The split must be enforced by direct strict validator tests that mutate each
historical binding independently and reject substitution of the current D29
binding; a comment or incidental generator pass is insufficient.

The minimum future-R3 authority is proved in two layers. The **mechanical
minimum** adds `tools/sci004_mmode_phase3_evidence.py` to D28's existing
three-path R3 authority: a frozen D24 binding and replacement of the erroneous
correction-#24 comparison are the smallest source change that lets the three
observed tests proceed. The **governed minimum** must also add
`tests/unit/test_sci004_phase3_evidence.py`. The committed tests authenticate
`_design_sha()` as a current frozen binding and exercise the joined red
reference, but they do not name both historical D24/D25 design identities,
mutate them independently, or reject re-equating either with the current
design. Without a direct validator-path change the new invariant would rest
only on implementation structure and incidental execution. The selected
future R3 therefore has exactly five paths, and no sixth:

1. `tests/characterization/test_sci004_mmode.py` owns only the exact
   v1-at-R3/v2-at-S3 transition-aware oracle.
2. `tests/unit/test_sci004_phase3_dependency.py` freezes exact D29 and
   authenticates the finite design/R3 succession.
3. `tests/unit/test_sci004_phase3_red_failures.py` authenticates the preserved
   scientific partition, retained records, and future-R3 path delta/replay.
4. `tools/sci004_mmode_phase3_evidence.py` separates the historical
   correction-#24 and correction-#25 record-design bindings from the current
   envelope design.
5. `tests/unit/test_sci004_phase3_evidence.py` directly proves those bindings
   remain separate and rejects mutation or current-design substitution.

That fifth path may add only strict tests and the minimum fixture/constants
needed to exercise the ruled separation. The evidence-tool change may add
only the minimum historical-binding constants and authentication needed for
both immutable records. Moving the complete dirty S3 evidence implementation
into R3, copying unrelated S3 schema, fingerprint-envelope, PERF-001,
generator, or acceptance work, or using a whole-file copy from the primary
checkout is forbidden. R3 may not edit the red generator, any red JSON record,
production, or a sixth path; deselect or normalize the three failures; add a
skip, xfail, environment, branch-name, commit-message, path-existence, or
mutable-runtime condition; depend on test order or import side effects; or
mutate production or dependency bindings from a test.

D29 directly parents exact D28 as one real single-parent, non-merge,
design-only commit. The future R3 must be a real single-parent, non-merge
direct child of exact D29: in this memo's established succession wording, the
future R3 directly parents D29. Its dependency validator authors
`APPROVED_SCI004_D_SHA = "<exact D29 commit SHA>"`. The old validator-only R3's
D26 binding remains immutable and historically correct, and the stopped
D28-based candidate is not itself the future R3 and cannot be committed over
D28 after D29 exists. Eventual replacement S3 directly parents the future
post-D29 R3; eventual E3 directly parents S3; eventual A3 directly parents E3;
and C remains conditional on a fresh independent A3 `ACCEPT` and the later
whole-row closure review. The exact chain is:

```text
D25
  -> original fingerprint R3
  -> D26
  -> old validator-only R3
  -> D27
  -> D28
  -> D29
  -> future validator/oracle/evidence-binding R3
  -> replacement S3
  -> fresh E3
  -> fresh independent A3
  -> C only after the required acceptance and closure reviews
```

All three M3 red records remain byte-for-byte immutable: the historical record
is 44824 bytes with SHA-256
`486705a8d5e51c08f972c91aeae60f0a0bfeef5480b622515282295a6a3cde05`;
the correction-#24 record is 10857 bytes with SHA-256
`724f75c246ebfcf5956fc40fb2f5e349d91ccca3e6a188b3785a65f4ae4c1e10`;
and the correction-#25 record is 7416 bytes with SHA-256
`6bf1cf94b30961fd7a27519fad1252169155fdeee0e81618ea15115b50fbdb68`.
They retain exactly 29 historical, six correction-#24, and two correction-#25
expected-red cases as a disjoint inventory. The fingerprint command remains
the exact ordered two failures followed by three passes, with no skip, xfail,
xpass, collection failure, missing node, duplicate, or reordering. Its
original oracle-patch SHA-256 remains
`1a9cecdea8d3e597de449c837d1e68bc72a4d29ab7ad9c4232c778c94efa4266`,
authenticated only across D25
`ca3c37171aaaeec175b5ad72d324957762303853` to original fingerprint R3
`a65c53a46e84f63c163c5ad15fba8645df33d1d2`. Correction #29 creates no red
JSON, generator mode, or scientific expected-red node.

The future post-D29 R3 restarts its complete pre-review sequence at gate 1.
The general transition-aware characterization oracle must pass; the exact
five-node partition must return raw pytest exit `1` with two failures followed
by three passes; all remaining characterization tests, the dependency
validator, the red validator, Tier 8, and static checks must pass; and the
complete serial unit suite must return exit code `0`, including all three
evidence-preflight tests that failed in the stopped attempt. Only then may the
future candidate be pinned and independently reviewed. A future R3 with any
of those three failures is not authorized. Correction #29 does not run that
future gate sequence.

The primary D27 checkout is a preserved, hash-authenticated S3 byte donor, not
a venue for authoring D29 or the future R3. It remains untouched throughout
correction #29 and throughout future D29 publication/R3 authoring unless a
later task grants exact authority. Its complete unstaged binary/full-index
diff is
`sha256:bc83d195be6a9d63a3945497595a16c5648eb06ec0c403e4c995966d3e5eccd3`
and its complete staged deletion diff is
`sha256:78aeebb4c4d240aba0899dba69b69db4e55506b6d6eacd826f396fd5a24a00f0`.
The existing stopped R3 candidate likewise remains untouched until D29 has
been accepted, committed locally, and separately published.

The selected future carry-forward route is a new clean linked R3 worktree at
exact published D29; advancing the existing dirty candidate worktree is not
authorized. Before reproduction, a later task must reauthenticate the stopped
candidate's exact three paths, the complete
`ff31a389880d48dbdb7e8063bdb4de488f3931a751e492bac121d600f959abc1`
patch digest, and the three file digests stated above, and prove D29 changes
only the two design documents. It then reproduces that authenticated
three-path D28-relative patch in the clean D29 worktree. Because D29 leaves all
three path bases unchanged, before any new evidence-binding edit the reproduced
D29-relative binary/full-index diff and all three working-tree hashes must be
the same four values. The index must remain empty, there may be no untracked,
added, or deleted path, and the stopped worktree must remain byte-identical.
Only then may the two newly authorized minimum evidence-binding paths be
authored and freshly pinned under a separate future-R3 task.

Every future-R3 hunk in either overlapping evidence path must be classified as
byte-identical to a primary S3 hunk, a strict prerequisite that S3 must retain,
or an R3-only validation hunk that remains valid at S3. The currently preserved
primary evidence-tool bytes already contain the D24/D25 constant separation;
that shared prerequisite may be composed by authenticated hunk identity, not
by whole-file copy. The primary evidence tool also contains still-S3-owned v2
manifest/envelope, fingerprint-join, current-attempt ancestry, and PERF-001
record work, while the primary evidence validator contains still-S3-owned v2
schema, reconstruction, hostile-mutation, and null-binding tests. Those
non-R3 changes prove that both paths remain genuine S3 modifications after the
minimum historical-binding hunks move into R3, but the later S3 task must
reprove their nonempty parent-relative deltas from live bytes.

No ordinary fast-forward of the dirty primary checkout across an R3 that
changes a locally modified path is allowed. After the future R3 is independently
accepted, committed, and separately published, a later explicitly authorized
S3 task creates a clean isolated replacement-S3 worktree at exact R3 and
reconstructs the candidate from the authenticated primary bytes, the four
staged-deletion identities, the committed R3 overlap hunks, and this D29
contract. The three non-overlapping modified paths must reproduce these
preserved primary byte identities unless later design authority changes them:
`src/radiosim/core/result.py` is 111381 bytes with SHA-256
`303513c4168717f8263de1eb7912e8be0f72736dd227d8fd17e649aee865ea9c`;
`tools/sci004_mmode_phase3_acceptance.py` is 36285 bytes with SHA-256
`1993eb5d64c7749eceb5fd0b0eac15544959f50679b5bb220246310060df5632`;
and `tests/unit/test_sci004_phase3_acceptance.py` is 36138 bytes with SHA-256
`7d9f3d19ddf6d011417b163e77f618cd76f970534455f06e7eb5f3b14518dce8`.

Each overlapping evidence path requires an explicit three-way composition:
the exact parent blob at future R3, the preserved intended primary S3 bytes,
and the D29-authorized historical-binding invariant. At D28/D29 the evidence
test base is 121956 bytes with SHA-256
`d55179d553db0b00df6bfee0a7db05f45a4be5ae6825e307fc827eaab43ede3a`
and its preserved primary intended bytes are 157699 bytes with SHA-256
`35ce46fdf011e5b8c66735ac6639e472e9d3977405b8ea374e6752566620fe17`;
the evidence-tool base is 156302 bytes with SHA-256
`26247c5ee7b81a6db012c6029293f93e98e3202a694cd15e969f57c666b06fb7`
and its preserved intended bytes are 196200 bytes with SHA-256
`283647df15cd9bc2974ad7da8e23a90528c7f4334b36e42069e25f1317c86c74`.
Their preserved primary D27-relative binary/full-index diff SHA-256 values are,
respectively,
`f88f9b0f9e6a33a632e666a4809db8010c6435f5e4b9d3a8458cc44c574d326f`
and
`fea892cae846d692f2be9bb103d90e5099db785311f9b039dde234d60a929dfb`.
The later task records raw byte SHA-256 identities before and after
composition; a zero exit from `git apply` or a merge is not preservation. No
R3 hunk or intended S3 hunk may disappear silently.

The eventual S3 retains exactly the same five-modification/four-deletion
parent-relative inventory, subject to the required live proof above that both
overlapping evidence paths retain S3-owned deltas:

- modify `src/radiosim/core/result.py`;
- modify `tools/sci004_mmode_phase3_evidence.py`;
- modify `tests/unit/test_sci004_phase3_evidence.py`;
- modify `tools/sci004_mmode_phase3_acceptance.py`;
- modify `tests/unit/test_sci004_phase3_acceptance.py`;
- delete `docs/development/sci004_mmode_phase3_evidence.json`;
- delete `docs/development/sci004_mmode_phase3_evidence.md`;
- delete
  `output/benchmarks/reference/sci004/20260825T122048Z-macbook-pro-2.json`;
  and
- delete `docs/development/sci004_mmode_phase3_acceptance.json`.

The later isolated S3 task must recreate exactly those four deletions and
authenticate their original blobs and raw SHA-256 values:
`docs/development/sci004_mmode_phase3_evidence.json` maps to blob
`0ea06771313cc454e4a6321425f7d93e73e5c703` and SHA-256
`600b51ac4d70778ee2d3bdf7b8842b83ba77dc34d541784ad1ad7d8e5be5f8ae`;
`docs/development/sci004_mmode_phase3_evidence.md` maps to blob
`5081d9a8207a3949d75fa533ad2f68ef18c2dd51` and SHA-256
`039539a865b5d92e86379f44a324271232e8a947301e380ec7b1b1848e907b4e`;
`output/benchmarks/reference/sci004/20260825T122048Z-macbook-pro-2.json`
maps to blob `85efb0f75f9fcf971ed29f334f8e0cfe59f92b57` and SHA-256
`07e59d3176866a78c17244849d6493365e9d410547e884cf56b254e60babe193`;
and `docs/development/sci004_mmode_phase3_acceptance.json` maps to blob
`747123dc8bc63025355370e1b6d5e3e261d69918` and SHA-256
`283fb5264f5ecd86aed1300ae504b85946cf1f4d36b1c4c09bc92bb4f269421d`.
No tenth S3 path is authorized. The primary checkout remains untouched until
the isolated reconstruction is compared and accepted; any cleanup, branch
movement, or primary replacement requires later explicit authority after the
successor draft is safely committed or otherwise fully authenticated.

Correction #29 fixes no code, creates or accepts no future R3, retries no S3,
generates no E3 or A3, runs no M3 measurements, and changes no source, test,
validator, generator, retained artifact, workflow, configuration, or PERF-001
record. It authorizes no v1 production fallback, dual production domain,
weakened digest, or acceptance claim. M3 remains rejected/not accepted;
SCI-004 and PERF-001 remain ROADMAP; E3 and A3 remain unavailable; and closure
`C` remains locked. No accelerator, GPU, public diffuse/HEALPix,
non-scalar-beam, public end-to-end backend-wiring, speedup,
general-performance, production-readiness, phase-acceptance, or closure claim
is licensed.
**Review status: fresh independent governance/authority and
evidence/provenance correction reviews pending against identical pinned
candidate bytes and the complete parent-relative diff.**

**Bounded correction #28 — 2026-09-01 (characterization-domain oracle
transition authority).** The earlier D27-based replacement-S3 attempt stopped
before any edit when its authority preflight reviewer returned `REJECT`. The
prepared primary-checkout S3 bytes remain preserved and uncommitted. The exact
blocker is the unmarked node
`tests/characterization/test_sci004_mmode.py::test_a_family_pin_is_a_ci001_observation_set_not_a_bare_digest`:
the committed protected module freezes
`CHARACTERIZATION_INPUT_DOMAIN =
"radiosim.sci004.characterization-input.v1"`, while the required replacement
S3 production draft sets `MMODE_CHARACTERIZATION_INPUT_DOMAIN =
"radiosim.sci004.characterization-input.v2"`. Its authoritative serial
preflight exited `1` with the exact assertion difference
`- radiosim.sci004.characterization-input.v1` / `+
radiosim.sci004.characterization-input.v2`. That node necessarily enters the
complete `-m "not slow"` source gate, but its protected test path is outside
S3's five modified paths. This is a succession-authority defect, not evidence
that production v2 is scientifically wrong, and the protected test has not
already been fixed.

Production v2 remains mandatory. Eventual replacement S3 must use exactly
`radiosim.sci004.characterization-input.v2`, the ordered nine-key family
record, the ordered fourteen-key characterization manifest, the complete
same-run `radiosim.mmode-input-identity.v1` phase manifest, every adjacent
digest and cross-manifest join, path-independent relocation identity, and
semantic-instrument-mutation inequality ruled by correction #25. This
correction authorizes no v1 fallback, production dual-domain option,
compatibility shim, evidence-only stand-in, filesystem venue in the scientific
preimage, weakened manifest/digest/projection/relocation/mutation rule, or
other production-semantic change.

D27 remains immutable at
`82fb0773890870a6fb90b3ed9b8065df89919a84`. D28 is its direct single-parent
non-merge child; after this correction lands, D27 is a superseded design and
D28 is the operative design for the future R3 and eventual S3. The old fresh
validator-only R3 at
`929263f3376e472e0b6da53c6c4e093c10ba7465` remains immutable historical
authority. Its dependency validator's D26 binding remains historically
correct because D26 was operative when that R3 was cut; it is never rewritten
in place. Replacement S3 cannot resume until a separately governed post-D28
validator-and-oracle-only R3 exists, and eventual S3 directly parents that new
R3, not D28, D27, or the old R3. The bounded succession is old R3 `->` D27
`->` D28 `->` future validator-and-oracle-only R3 `->` replacement S3 `->`
future E3 `->` future independent A3; this is not an unbounded ancestry or
newest-memo-tip rule. Only D28 is authored now.

The future validator-and-oracle-only R3 may modify exactly these three paths,
and no fourth path:

- `tests/characterization/test_sci004_mmode.py`
- `tests/unit/test_sci004_phase3_dependency.py`
- `tests/unit/test_sci004_phase3_red_failures.py`

It may not modify `src/radiosim/core/result.py`,
`tools/sci004_mmode_phase3_red.py`, any retained red JSON record, any evidence
or acceptance tool or validator, either design/status document, any generated
artifact, or any performance path. PERF-001 remains ROADMAP, and no accepted
accelerator-performance record exists. That future R3 must be a real
single-parent non-merge Git commit, the direct child of exact D28, with exactly
the three-path parent-relative inventory above. It requires its own later
independent review and verification and may be published or fast-forwarded
only under later explicit owner authority. This correction does not create it.

The future characterization edit is an exact transition-aware oracle, not a
singleton literal flip. At exact future R3, production remains exactly
`radiosim.sci004.characterization-input.v1`; the exact seven-key v1 family
record is required, both v2-only manifest keys are absent, and every existing
family, science, solver, dimension, observation-set, and digest assertion
remains active. At replacement S3, production is exactly
`radiosim.sci004.characterization-input.v2`; the exact ordered nine-key family
record and complete ordered fourteen-key characterization manifest are
required, the same-run phase manifest is retained, every adjacent digest and
cross-manifest join is independently rebuilt, relocation preserves input
identity, semantic instrument mutation changes input identity, and no v1
fallback or v1-shaped v2 record is accepted. Any third domain fails. Merely
changing the committed singleton from v1 to v2 at R3 is forbidden because R3
still has v1 production and that change would manufacture an unrecorded red
failure.

The transition-aware oracle may not use unconditional acceptance of both
record shapes; subset or superset key checks; a skip, xfail, marker,
environment condition, branch-name test, commit-message test, path-existence
test, or mutable runtime switch; a test-only production-constant mutation; or
a production v1 fallback. The existing dedicated v2 expected-red node remains
the mechanism that requires v2 at replacement S3. The general
domain/observation-set oracle must accept only the two exact governed states
without contradicting that dedicated requirement.

The future R3 creates no new red observation. The immutable correction-#25
fingerprint supplement
`docs/development/sci004_mmode_phase3_fingerprint_post_source_red_failures.json`
remains exactly 7416 bytes with SHA-256
`6bf1cf94b30961fd7a27519fad1252169155fdeee0e81618ea15115b50fbdb68`,
design SHA `ca3c37171aaaeec175b5ad72d324957762303853`, pre-fix source SHA
`b07925ab14b56b3ca0fa863f806290748a31df6b`, and original oracle-patch
SHA-256
`1a9cecdea8d3e597de449c837d1e68bc72a4d29ab7ad9c4232c778c94efa4266`.
It retains exactly two expected-red fingerprint cases and three passing
controls. The future green transition-contract repair adds no third case and
does not alter that two-red/three-green partition, either prior red record, or
the historical no-overwrite generator.

The future red validator authenticates the fingerprint supplement and its
historical oracle-patch digest at original fingerprint R3
`a65c53a46e84f63c163c5ad15fba8645df33d1d2`, never against the new
characterization bytes. It separately authenticates old fresh R3
`929263f3376e472e0b6da53c6c4e093c10ba7465`, D27, D28, and the new R3 as
exact Git objects; authenticates the new R3's exact three-path delta; and proves
all three red records are inherited byte-for-byte. Its detached new-R3 replay
runs the same ordered two red nodes and three passing controls and requires raw
pytest exit `1`, exactly two failures and three passes, no skip/xfail/xpass or
collection error, detached import provenance, and deterministic cleanup. It
also separately requires the corrected general domain/observation-set oracle
to pass at new R3. No retained JSON record or
`tools/sci004_mmode_phase3_red.py` changes, and no generator mode is added.

The future dependency validator authors
`APPROVED_SCI004_D_SHA = <exact D28 landing commit SHA>` only after D28 exists.
It authenticates this complete direct single-parent non-merge chain:
`ca3c37171aaaeec175b5ad72d324957762303853` (D25) `->`
`a65c53a46e84f63c163c5ad15fba8645df33d1d2` (original fingerprint R3)
`->` `93321d331e4f6442d39fe79588be6f05ad4bee42` (D26) `->`
`929263f3376e472e0b6da53c6c4e093c10ba7465` (old fresh validator-only R3)
`->` `82fb0773890870a6fb90b3ed9b8065df89919a84` (D27) `->` D28 `->`
the future validator-and-oracle-only R3. The old R3 continues to carry its
historically correct D26 binding.

Eventual replacement S3 retains exactly five modified and four deleted paths:

- modify `src/radiosim/core/result.py`
- modify `tools/sci004_mmode_phase3_evidence.py`
- modify `tests/unit/test_sci004_phase3_evidence.py`
- modify `tools/sci004_mmode_phase3_acceptance.py`
- modify `tests/unit/test_sci004_phase3_acceptance.py`
- delete `docs/development/sci004_mmode_phase3_evidence.json`
- delete `docs/development/sci004_mmode_phase3_evidence.md`
- delete
  `output/benchmarks/reference/sci004/20260825T122048Z-macbook-pro-2.json`
- delete `docs/development/sci004_mmode_phase3_acceptance.json`

No tenth S3 path is authorized. Eventual S3 is the direct child of the future
post-D28 R3, uses `design_sha = D28` and `red_commit_sha = future post-D28
R3`, authenticates the fixed historical chain through both, leaves all six
approved bindings literal `None`, inherits all three red records byte-for-byte,
and preserves the full production-v2 contract. It does not directly parent old
R3, D27, or D28. Its complete gate restarts from the beginning under separate
later authority. D28 neither modifies nor rebases the preserved S3 patch.

D28 establishes design authority only. The future R3 authority is defined but
not exercised; the production-v2 draft is preserved but uncommitted; S3 is not
retried; and no generator, E3, A3, acceptance, or closure work is performed.
M3 remains rejected/not accepted, SCI-004 remains ROADMAP, E3 and A3 remain
unavailable, and closure `C` remains locked. PERF-001 remains ROADMAP, and no
accepted accelerator-performance record exists. No accelerator, GPU, public
diffuse/HEALPix, non-scalar-beam, public end-to-end backend-wiring, speedup,
general-performance, PERF-001 acceptance, production-readiness,
phase-acceptance, or closure claim is licensed. The rejected deleted
performance draft is not accepted evidence.
**Review status: fresh independent governance/authority and oracle/provenance
correction reviews pending against identical pinned candidate bytes and the
complete parent-relative diff.**

**Bounded correction #27 — 2026-09-01 (correction-#26 Tier-8 prose citation
gate).** The replacement-S3 continuation stopped fail-closed at the complete
serial unit gate
`pixi run python -m pytest -p no:randomly -p no:xdist tests/unit/`. It returned
exit code `1` after `1500.16` seconds: `1 failed, 7481 passed, 15 skipped, 30
warnings`. The sole failure was
`tests/unit/test_tier8_release_acceptance.py::test_no_accelerator_claim_in_tracked_prose_lacks_a_citation`,
which flagged the then-published correction-#26 paragraph at
`docs/development/sci004_mmode_design.md:234` with the exact text
`backend-wiring, general performance, or speedup claim. No fingerprint`;
`PERF-001` was and remains ROADMAP, not accepted accelerator evidence. All
preceding focused SCI-004 gates were green, including the repaired hermetic
detached-R3 replay; none of those successes overrides the later complete-unit
Tier-8 failure.

The defect exists solely in already-published correction-#26 design prose.
Physical line wrapping split the literal denial phrase `no accelerator`, and
the paragraph contained neither of the scanner's accepted citation tokens,
`output/benchmarks/reference/` or `PERF-001`. This is not a changed scientific
partition, a replacement-S3 implementation failure, accelerator evidence, or
a performance result. It grants no authority to weaken the scanner and no
authority to create E3, run M3 performance, create A3, accept M3, or produce
closure output.

The governed repair changes only that correction-#26 denial paragraph in the
current tree. The same blank-line-separated paragraph now states that
`PERF-001 remains ROADMAP, and no accepted accelerator-performance record
exists.` It continues to state that correction #26 licenses no accelerator,
public HEALPix/diffuse, non-scalar-beam, public end-to-end backend-wiring,
general-performance, or speedup claim. The Tier-8 scanner is unchanged. The
statement neither treats PERF-001 as accepted or closed nor cites the deleted
SCI-004 M3 performance record as accelerator evidence.

Correction #26 remains immutable as Git commit
`93321d331e4f6442d39fe79588be6f05ad4bee42`; correction #27 corrects only its
current-tree prose. This landing becomes the new operative `D`, and correction
#26 becomes a `superseded design` chain entry. Fresh validator-only R3
`929263f3376e472e0b6da53c6c4e093c10ba7465` remains the immutable landed R3
authority. Under Section 13.7's existing correction-between-`R`-and-`S` rule,
correction #27 intervenes between that landed R3 and replacement S3; it does
not reopen, replace, or regenerate R3, and eventual replacement S3 must
directly parent correction #27.

Future S3 validators may authenticate the R3-to-D27 interval only within the
existing five modified S3 source/tool/test paths:

- `src/radiosim/core/result.py`
- `tools/sci004_mmode_phase3_evidence.py`
- `tests/unit/test_sci004_phase3_evidence.py`
- `tools/sci004_mmode_phase3_acceptance.py`
- `tests/unit/test_sci004_phase3_acceptance.py`

The four deletion-only paths remain exactly:

- `docs/development/sci004_mmode_phase3_evidence.json`
- `docs/development/sci004_mmode_phase3_evidence.md`
- `output/benchmarks/reference/sci004/20260825T122048Z-macbook-pro-2.json`
- `docs/development/sci004_mmode_phase3_acceptance.json`

No tenth replacement-S3 path is authorized. M3 remains rejected/not accepted;
SCI-004 remains ROADMAP; E3 and A3 remain unavailable; closure `C` remains
locked. The existing nine-path replacement-S3 draft remains uncommitted and
is neither accepted nor pre-approved by correction #27.
**Review status: fresh independent physics/governance and
computational/provenance correction reviews pending against identical pinned
candidate bytes and the complete parent-relative diff.**
**Final review verification:** the reviewed pre-landing design was 388,551 raw
bytes with SHA-256
`afd9dc6d7e26641b295cc80385067d4f624af09afc814c70fa7d6934c654b831`
and parent-relative binary/full-index diff SHA-256
`599e655ee5490c3c10693df2101d8f67b2ac43bffec12ee117b403325350ad69`;
the reviewed pre-landing ledger was 53,901 raw bytes with SHA-256
`18ef4cf0badfe3540d9e19247103a019c5a5294de813eacac7d8d9a9128b0562`
and parent-relative binary/full-index diff SHA-256
`17b081f4c9787bdc5da499444772f592f3cdccd3ed1d8b8b71512be4cebd5502`;
the complete canonical parent-relative binary/full-index diff SHA-256 was
`0572bbc64983f6bab72f71db0ea9f1f9f016d38a87be3e048c8b640aaa891db1`;
and the fresh round-3 reviewers `/root/c27_physics_governance_review` and
`/root/c27_computational_provenance_review` each returned exact `ACCEPT` on
those identical bytes and complete diff after two bounded fix rounds. Only
this sentence and the companion ledger verdict completion postdate the
reviewed pins.

**Bounded correction #26 — 2026-08-30 (hermetic detached-R3 validator replay
venue).** Correction #25's replacement-S3 governance slice stopped fail-closed
at
`tests/unit/test_sci004_phase3_red_failures.py::test_the_fresh_r3_detached_replay_reproduces_the_five_node_partition`.
The outer serial governance command returned `1 failed, 491 passed, 13
skipped`; the nested five-node replay unexpectedly returned raw pytest exit
`0` and `5 passed`, where the retained record requires exit `1`, two expected
assertion failures, and three passing controls. No replacement-S3 commit,
evidence, performance, or acceptance successor followed.

The failure is a deterministic validator source-provenance defect, not a
changed scientific partition. Correction #25's replay invokes
`sys.executable` from the shared Pixi environment and changes only the child
process working directory to a detached src-layout checkout. A detached
checkout root does not itself put `<detached-worktree>/src` ahead of
site-packages. The shared interpreter processes its editable-install `.pth`
entry, which can therefore select the invoking primary checkout's `src` tree.
The defect was latent while the primary and detached checkouts both presented
the R3 implementation; the dirty replacement-S3 v2 implementation made it
observable by satisfying the two expected-red nodes and turning the entire
five-node replay green. The observed shared interpreter and absolute editable
source locations are diagnostic evidence only. No absolute home, primary
checkout, or temporary-worktree path is a portable scientific contract; every
ruled replay path below is derived at runtime from the temporary detached
worktree or the invoking validator checkout.

This correction follows original fingerprint R3
`a65c53a46e84f63c163c5ad15fba8645df33d1d2`, whose sole parent is accepted
correction #25
`ca3c37171aaaeec175b5ad72d324957762303853`. Correction #25 becomes a
`superseded design` chain entry and this landing becomes operative `D`.
Original fingerprint R3 becomes a `superseded red slice` — specifically the
superseded fingerprint R3 slice — only for the validator mechanism re-cut
here: it remains the immutable Git object that
contains the genuine official five-node observation and oracle delta. Its
parent-relative diff must continue to touch exactly:

- `docs/development/sci004_mmode_phase3_fingerprint_post_source_red_failures.json`
- `tests/characterization/test_sci004_mmode.py`
- `tests/unit/test_sci004_phase3_dependency.py`
- `tests/unit/test_sci004_phase3_red_failures.py`
- `tools/sci004_mmode_phase3_red.py`

The retained supplement
`docs/development/sci004_mmode_phase3_fingerprint_post_source_red_failures.json`
is immutable at `7,416` raw bytes and SHA-256
`6bf1cf94b30961fd7a27519fad1252169155fdeee0e81618ea15115b50fbdb68`.
The complete bytes of that supplement,
`tests/characterization/test_sci004_mmode.py`, and
`tools/sci004_mmode_phase3_red.py` are preserved. The characterization oracle
is still exactly the binary/full-index diff on the real correction-#25-to-R3
edge
`ca3c37171aaaeec175b5ad72d324957762303853 ->
a65c53a46e84f63c163c5ad15fba8645df33d1d2`, with SHA-256
`1a9cecdea8d3e597de449c837d1e68bc72a4d29ab7ad9c4232c778c94efa4266`.
It is not redefined as the empty oracle diff from this correction to the new
validator-only R3. The new R3 neither reintroduces nor regenerates the oracle.
The historical M3 red record remains at SHA-256
`486705a8d5e51c08f972c91aeae60f0a0bfeef5480b622515282295a6a3cde05`;
the correction-#24 post-source red record remains at SHA-256
`724f75c246ebfcf5956fc40fb2f5e349d91ccca3e6a188b3785a65f4ae4c1e10`.

The official fingerprint observation remains canonical and is not replaced.
Its pre-fix source binding remains
`b07925ab14b56b3ca0fa863f806290748a31df6b`. Its two expected-red rows remain
exactly:

- `tests/characterization/test_sci004_mmode.py::test_characterization_input_preimage_is_retained_and_reconstructible`,
  message `characterization input manifest is absent from the family record`,
  `invalid_config_raw_sha256=4c11755ecae7597f8ffb30f7aa5653eda41a58994fae19086bb15109c60558b6`,
  and
  `fixture_identity_sha256=b5c765aaae957ea3d686e3693b9a2469f7e491b0c247ac695e4ad3e0178b8a0b`;
- `tests/characterization/test_sci004_mmode.py::test_characterization_input_identity_is_equal_under_distinct_layout_roots`,
  message `characterization input identity changed under filesystem
  relocation`,
  `invalid_config_raw_sha256=a24c64f4d981fce69c9f6cebaadd1bca0ae52fed0783e94b67ac3d8245df4a4f`,
  and
  `fixture_identity_sha256=98cb2605eacaa5e473fbc573fa135ec91e0b2320e9759bbfa646c706628ef6ac`.

Its three ordered passing controls remain exactly
`tests/characterization/test_sci004_mmode.py::test_every_new_family_records_its_six_section_11_parts[mmode_single_scalar_mode]`,
`tests/characterization/test_sci004_mmode.py::test_distinct_layout_roots_preserve_scientific_and_cube_identities`,
and
`tests/characterization/test_sci004_mmode.py::test_characterization_input_identity_changes_for_semantic_instrument_content`.
The immutable scientific result remains the ordered serial partition `2
failed, 3 passed`, with no skipped, xfailed, xpassed, collection-failed,
missing, duplicated, or reordered node. There is no supplement regeneration,
new fingerprint-generator invocation, new case/control row, or replacement of
the official one-command record.

After this correction is independently accepted, one fresh validator-only R3
directly parents it and may modify exactly:

- `tests/unit/test_sci004_phase3_dependency.py`
- `tests/unit/test_sci004_phase3_red_failures.py`

No other R3 path is writable. The dependency validator freezes this
correction's exact accepted SHA as `APPROVED_SCI004_D_SHA`, extends and
authenticates the complete header-enumerated design chain, and authenticates
correction #25 and original fingerprint R3 from Git objects. It requires the
original R3's exact sole parent, exact five-path diff, the retained
supplement's frozen raw digest, and the original correction-#25-to-R3 oracle
delta above. It classifies the fresh R3 as this correction's sole-parent
direct child, requires its parent-relative diff to touch exactly the two
validator paths, and derives the live replay anchor strictly after the new
operative `D`; original fingerprint R3 can never be silently selected as the
live replacement anchor. The red-record validator continues to authenticate
the immutable supplement at original R3 and implements only the replay-venue,
provenance, classification, and cleanup checks ruled here.

The child replay uses one closed environment construction shared byte-for-byte
by its provenance preflight and five-node pytest execution, equivalent to:

```python
replay_env = os.environ.copy()
replay_env["PYTHONPATH"] = str(worktree / "src")
replay_env["PYTHONNOUSERSITE"] = "1"
```

`PYTHONPATH` is replaced, never appended or prepended to an inherited value;
the dynamically derived detached `src` is its sole ruled entry. User-site
packages are disabled. No primary-checkout literal may enter the child
environment. The detached root itself need not be added to `PYTHONPATH` for
the current src layout; if a later implementation proves another entry
necessary, it requires a further bounded design correction and detached `src`
must remain first.

Before the expensive five-node replay, the validator runs a bounded child
import probe with the same `sys.executable`, detached worktree `cwd`, and exact
closed environment. It resolves `radiosim.__file__`, requires the resolved
target to be a regular file contained by
`<detached-worktree>/src/radiosim`, and separately requires it not to be under
the invoking primary checkout's resolved `src` tree. An absent, non-regular,
symlink-escaped, unparsable, outside-detached, or primary-tree result fails
before pytest. This probe authenticates the validator venue only: it is not a
scientific observation, retained-supplement command, new case/control row, or
licence for evidence or acceptance.

Only after that preflight passes does the validator invoke the same ordered
serial five-node inventory with `sys.executable -m pytest -p no:randomly -p
no:xdist --junit-xml <validator-owned-temporary-path>`. It requires raw pytest
exit `1`; exactly two failed and three passed testcases; zero skipped, xfailed,
xpassed, collection failures, errors, missing, duplicated, or reordered nodes;
and the exact two governed assertion messages above. The three controls remain
green. JUnit and worktree locations are confined to the validator-owned
temporary directory. Success, import mismatch, pytest failure, parse failure,
and assertion failure all run the same deterministic cleanup. Cleanup removes
only the exact worktree registered by this invocation and its exact temporary
directory, treats incomplete removal as a validator failure, and never runs a
repository-wide worktree prune or touches an unrelated worktree.

Regression assertions, confined to the two fresh-R validator paths, prove that
inherited `PYTHONPATH` is replaced; detached `src` is the selected package
root; a primary editable-install path cannot shadow it; the probe rejects any
package file outside detached `src/radiosim`; the exact five-node JUnit
classification and messages remain unchanged; and worktree/JUnit cleanup runs
after both successful and failing child execution. They may not edit, skip,
xfail, weaken, or otherwise change the characterization oracle, supplement,
generator, production implementation, dependencies, lockfiles, workflow, or
performance surface.

The resulting concrete succession is:

```text
ca3c37171aaaeec175b5ad72d324957762303853 D25
  -> a65c53a46e84f63c163c5ad15fba8645df33d1d2 original fingerprint R3
     (scientific observation retained; replay validator venue superseded)
  -> D26
  -> fresh validator-only R3
  -> replacement S3
  -> fresh E3
  -> fresh independent A3
  -> C only after that A3 is ACCEPT
```

Replacement S3 directly parents the new validator-only R3 and retains
correction #25's exact authority: modifications only to
`src/radiosim/core/result.py`, `tools/sci004_mmode_phase3_evidence.py`,
`tests/unit/test_sci004_phase3_evidence.py`,
`tools/sci004_mmode_phase3_acceptance.py`, and
`tests/unit/test_sci004_phase3_acceptance.py`; and deletion only of
`docs/development/sci004_mmode_phase3_evidence.json`,
`docs/development/sci004_mmode_phase3_evidence.md`,
`output/benchmarks/reference/sci004/20260825T122048Z-macbook-pro-2.json`, and
`docs/development/sci004_mmode_phase3_acceptance.json`. Rejected-evidence
disposal remains owned solely by replacement S3. No tenth path or additional
S3 obligation is authorized. The prepared nine-path draft may be reapplied
byte-for-byte only if it remains conformant; the complete phase-3 governance
slice then reruns from its beginning, and only a wholly green governed gate
permits the S3 commit. This correction neither accepts nor pre-approves that
draft.

This correction is design-only and accepts no R3 validator implementation,
source slice, evidence, performance, or acceptance output. M3 remains
rejected/not accepted; SCI-004 remains ROADMAP; closure `C` remains locked;
E3 and A3 remain unavailable. It changes no tolerance, fidelity, convergence,
memory, or performance predicate; no accepted M1/M2 evidence or acceptance;
and no SCI-005 Stage-3 prerequisite or derivative claim. PERF-001 remains
ROADMAP, and no accepted accelerator-performance record exists. It licenses
no accelerator, public HEALPix/diffuse, non-scalar-beam, public end-to-end
backend-wiring, general performance, or speedup claim. No fingerprint
supplement, characterization oracle, official generator, production source,
rejected artifact, dependency, lockfile, workflow, simulator submodule,
evidence, performance, or acceptance output becomes writable here.
**Review status: fresh independent physics/governance and
computational/provenance correction reviews pending against identical pinned
candidate bytes and the complete parent-relative diff.**
**Final review verification:** the reviewed pre-landing design was 383,339 raw
bytes with SHA-256
`193c71c8983d62fdbd29c99d891e42835920ddd71a896be9171e8565a654b3fc`
and parent-relative binary/full-index diff SHA-256
`64b064ef664037cb4c1ecc7a418561c816a8edc5fb1b9b230ea1d53bf1672ff0`;
the reviewed pre-landing ledger was 53,103 raw bytes with SHA-256
`6e7b41fa5a9875220a52bc58dd5244ef97db89661e7a9937d7691dcdfc77d8ba`
and parent-relative binary/full-index diff SHA-256
`ee80811b6638b93f513fca561376417ec28f49d48192d8331a875c6456831bd8`;
the complete fixed-order design-then-ledger parent-relative diff SHA-256 was
`9d711cfa575df6b3dd3152767b0d17f70d3091fde4ab7713b253f68a0e520448`;
and the fresh round-1 reviewers `/root/c26_physics_governance_review` and
`/root/c26_computational_provenance_review` each returned exact `ACCEPT` on
those identical bytes and complete diff. No review round required correction;
only this sentence and the companion ledger verdict completion postdate the
reviewed pins.

**Bounded correction #25 — 2026-08-26 (reconstructible path-independent M3
fingerprints and rejected-attempt retry grammar).** The canonical independent
`A3` record at
`8529da951e2378115ffde8d5da3e2af56f3323d0` returned `REJECT` on required
predicate `SCI-004-14.3-A3-EVERY-FINGERPRINT`, blocker
`m3.fingerprint-input-preimage-not-retained`. Its exact artifact is
`docs/development/sci004_mmode_phase3_acceptance.json`, `8,238` raw bytes,
SHA-256
`283fb5264f5ecd86aed1300ae504b85946cf1f4d36b1c4c09bc92bb4f269421d`.
Its validator change is exactly
`APPROVED_EVIDENCE_SHA="886e62fd9f8328826b388b8960ed7413da26b6d1"`
and
`APPROVED_ACCEPTANCE_ARTIFACT_SHA256="283fb5264f5ecd86aed1300ae504b85946cf1f4d36b1c4c09bc92bb4f269421d"`.
The external independent-review contribution has `10` oracle rows, one blocker,
and raw SHA-256
`43c12807aa9f316af53e6058ebec7f18dd0b6ea66d308cb1c488d77185907d82`.
Both the immutable A3 artifact and that contribution bind
`reviewer_identity="sci004-m3-independent-acceptance-reviewer"` and
`reviewer_independent=true`; fresh R3's dependency/red validator must read the
artifact from the exact A3 Git object and require both literals in addition to
the two raw digests.
The rejected `E3` at
`886e62fd9f8328826b388b8960ed7413da26b6d1` produced the canonical evidence
JSON (9,186,099 bytes, raw SHA-256
`600b51ac4d70778ee2d3bdf7b8842b83ba77dc34d541784ad1ad7d8e5be5f8ae`),
reproduction Markdown (3,503 bytes, raw SHA-256
`039539a865b5d92e86379f44a324271232e8a947301e380ec7b1b1848e907b4e`),
and host-bound performance record
`output/benchmarks/reference/sci004/20260825T122048Z-macbook-pro-2.json`
(58,844 bytes, raw SHA-256
`07e59d3176866a78c17244849d6493365e9d410547e884cf56b254e60babe193`).
Its evidence-validator change is exactly the four assignments
`APPROVED_SOURCE_SHA="b07925ab14b56b3ca0fa863f806290748a31df6b"`,
`APPROVED_ARTIFACT_SHA256="600b51ac4d70778ee2d3bdf7b8842b83ba77dc34d541784ad1ad7d8e5be5f8ae"`,
`APPROVED_PERFORMANCE_PATH="output/benchmarks/reference/sci004/20260825T122048Z-macbook-pro-2.json"`,
and
`APPROVED_PERFORMANCE_SHA256="07e59d3176866a78c17244849d6493365e9d410547e884cf56b254e60babe193"`.
Those immutable Git objects establish that E3 was a candidate, M3 was not
accepted, SCI-004 remained ROADMAP, and closure `C` never became available.

The scientific defect is exact. `mmode_characterization_record` hashes a v1
manifest containing the complete `SimulationResult.resolved_config`; the M3
fixture puts its layout file under an absolute temporary root, and the E3
generator destroys that `TemporaryDirectory`. E3 retains the digest but neither
the manifest nor the result. Its strict validator checks only lower-case
64-hex syntax. Filesystem venue is not a scientific input: two byte-identical
layout documents resolving to the same site, antenna, baseline, receptor, beam,
frequency, sky, Jones, time-grid, and solver values must characterize
identically even when their absolute roots differ. The arbitrary layout path
is therefore removed from characterization identity and replaced by the exact
value-bearing phase input manifest and independently reconstructed instrument,
receptor, and loaded-beam content identities ruled in Sections 11, 14.0, and
14.2. The v1 domain cannot survive that preimage change; the retry uses
`radiosim.sci004.characterization-input.v2` and retains the complete v2
manifest in every fingerprint row. The phase input, grid, solver, cube, and
scientific identities are joins, never children of one another, so the new
relationship is non-circular. All four characterization-input digests are
expected to change solely because the domain and preimage change. The
path-independent `scientific_sha256`, raw-cube, solver-snapshot,
result-derived ERA/UTC characterization-time, harmonic-table, phase-input, and
observation-set identities are expected not to change; any contrary result
pauses for an old/new-cube and equation-level explanation. No tolerance,
fidelity predicate, convergence or memory budget, performance predicate,
capability claim, or elapsed-time threshold changes.

This is a new post-source red delta against committed S3
`b07925ab14b56b3ca0fa863f806290748a31df6b`, not a rewrite of either immutable
red record and not a full historical R3 replacement. Fresh R3 owns exactly the
two expected-red scientific oracle nodes
`tests/characterization/test_sci004_mmode.py::test_characterization_input_preimage_is_retained_and_reconstructible`,
and
`tests/characterization/test_sci004_mmode.py::test_characterization_input_identity_is_equal_under_distinct_layout_roots`.
They respectively govern `SCI-004-14.2-M3-FINGERPRINT-PREIMAGE` and
`SCI-004-11-PATH-INDEPENDENT-CHARACTERIZATION`. A real semantic antenna-layout
mutation already changes v1 identity at the pre-fix source, so
`tests/characterization/test_sci004_mmode.py::test_characterization_input_identity_changes_for_semantic_instrument_content`
is a mandatory passing green control governing
`SCI-004-11-SEMANTIC-INPUT-SEPARATION`, excluded from `cases` and the
expected-red disjoint union. The relocation node is a true scientific oracle:
it compares independently constructed roots and cannot be replaced by a
synthetic validator document whose author supplied both desired digests. R3
retains the two red cases at
`docs/development/sci004_mmode_phase3_fingerprint_post_source_red_failures.json`,
schema
`radiosim.sci004.mmode-phase3-fingerprint-post-source-red-failures.v1`, status
`post-source-expected-red-confirmed`, with `design_sha` equal to this landing,
`pre_fix_source_sha` equal to
`b07925ab14b56b3ca0fa863f806290748a31df6b`, a null `red_commit_sha` and reason
`self-reference: E binds the containing fingerprint-retry R3 commit`, the exact
two node/requirement rows above, and the exact oracle patch digest over only
`tests/characterization/test_sci004_mmode.py`. Its top level has exactly
`schema_version`, `phase`, `status`, `generated_at_utc`, `design_sha`,
`pre_fix_source_sha`, `red_commit_sha`, `red_commit_sha_reason`,
`protected_source_clean`, `authorized_red_paths`, `environment`, `cases`,
`passing_controls`, `commands`, `claims_not_licensed`, `historical_red_record_sha256`,
`correction24_post_source_red_record_sha256`, `oracle_patch_paths`, and
`oracle_patch_sha256`; its environment, case, command, and claim rows use
Section 14.1's exact closed schemas. The two `cases` rows are exactly:

| `case_id` | `requirement_id` | `test_nodeid` | expected kind | expected pattern | command |
|---|---|---|---|---|---:|
| `m3.fingerprint.preimage-retained` | `SCI-004-14.2-M3-FINGERPRINT-PREIMAGE` | `tests/characterization/test_sci004_mmode.py::test_characterization_input_preimage_is_retained_and_reconstructible` | `assertion` | `characterization input manifest is absent from the family record` | `0` |
| `m3.fingerprint.path-independent` | `SCI-004-11-PATH-INDEPENDENT-CHARACTERIZATION` | `tests/characterization/test_sci004_mmode.py::test_characterization_input_identity_is_equal_under_distinct_layout_roots` | `assertion` | `characterization input identity changed under filesystem relocation` | `0` |

For the first row, `invalid_config_raw_sha256` hashes `J` of the exact
five-key object `schema_version`, `family_id`, `layout_document_raw_sha256`,
`root_labels`, `required_record_keys`; the values are respectively
`radiosim.sci004.fingerprint-red-fixture.v1`,
`mmode_single_scalar_mode`,
`a2ce7bace30e2fe962eb6454db1f6c7e2d63a9a28ad559323e824a36fcd2a4e0`,
`["ROOT-A"]`, and the exact nine-key v2 production family-record array in
Section 11 order. It is not the Section 14.2 evidence-row schema.
For the second, it hashes `J` of the exact five-key object
`schema_version`, `family_id`, `layout_document_raw_sha256`, `root_labels`,
`required_equal_identities`, with the same first three values,
`root_labels=["ROOT-A","ROOT-B"]`, and
`required_equal_identities=["scientific_sha256","raw_cube_sha256","era_utc_grid_sha256","input_identity_sha256"]`.
The layout digest is over the exact family layout bytes written by
`family_mapping`, including its final LF. Each `fixture_identity_sha256` then
uses Section 14.0's exact `radiosim.sci004-red-fixture.v1` six-key preimage and
the adjacent `invalid_config_raw_sha256`; root labels are logical tokens, never
retained filesystem locations.

`passing_controls` is an ordered three-row array with the exact closed row
shape `control_id`, `requirement_id`, `purpose`, `test_nodeid`,
`command_index`, `observed_outcome`, `exit_code`, `pass`. Its rows are:

| `control_id` | `requirement_id` | `test_nodeid` | `purpose` |
|---|---|---|---|
| `m3.fingerprint.family-record-schema` | `SCI-004-11-FAMILY-RECORD-SCHEMA` | `tests/characterization/test_sci004_mmode.py::test_every_new_family_records_its_six_section_11_parts[mmode_single_scalar_mode]` | `exact domain-discriminated family-record schema and all pre-existing family joins remain valid` |
| `m3.fingerprint.relocation-science-control` | `SCI-004-11-PATH-INDEPENDENT-CHARACTERIZATION` | `tests/characterization/test_sci004_mmode.py::test_distinct_layout_roots_preserve_scientific_and_cube_identities` | `relocation fixture preserves independently derived scientific and raw-cube identities` |
| `m3.fingerprint.semantic-separation-control` | `SCI-004-11-SEMANTIC-INPUT-SEPARATION` | `tests/characterization/test_sci004_mmode.py::test_characterization_input_identity_changes_for_semantic_instrument_content` | `semantic antenna-layout mutation changes characterization input identity` |

Every row has `command_index=0`, `observed_outcome="pass"`, `exit_code=0`,
and `pass=true`. Unknown, missing, reordered, skipped, xfailed, uncollected, or
false control rows reject.

R3 makes one equal-or-stronger, domain-discriminated edit to the existing
parametrized family-record control. When the imported production input-domain
literal is exactly `radiosim.sci004.characterization-input.v1`, it calls the
current v1 signature, requires only the exact ordered seven-key Section 11
record, requires both new manifest keys absent, and preserves every current
family/scientific/solver/dimension/digest/determinism assertion. When and only
when that literal is exactly
`radiosim.sci004.characterization-input.v2`, it constructs the same-run phase
manifest independently, calls the ruled v2 signature, requires only the exact
ordered nine-key Section 11 record, rederives both complete manifests and their
adjacent digests, and preserves every old assertion. Any other domain, a
subset/superset check, unknown key, removed assertion, unconditional acceptance
of both shapes, or a v2 call falling back to v1 rejects. The dedicated preimage
red oracle additionally requires v2 after replacement S3, so leaving the
domain or production record at v1 cannot satisfy S3. Command `0` exercises the
scalar v1 branch at R3; replacement-S3 validation and E3/A3 replay the v2
branch for all four families.

Command `0` is one serial invocation, with the resolved Pixi default Python as
the executable and arguments exactly
`-m pytest -p no:randomly -p no:xdist --junit-xml
<generator-owned-temporary-directory>/junit.xml`, followed in order by the two
red node IDs above and these three passing controls:
`tests/characterization/test_sci004_mmode.py::test_every_new_family_records_its_six_section_11_parts[mmode_single_scalar_mode]`,
`tests/characterization/test_sci004_mmode.py::test_distinct_layout_roots_preserve_scientific_and_cube_identities`,
and
`tests/characterization/test_sci004_mmode.py::test_characterization_input_identity_changes_for_semantic_instrument_content`.
The generator requires exactly the two expected assertion failures, exactly
the three passing controls, no skip/xfail/collection error, and exit `1`; the
first existing control excludes a malformed family record, the second excludes
a relocated fixture changing science or cube, and the third proves that path
removal has not collapsed semantic instrument content. It derives both the two
`cases` rows and all three `passing_controls` rows from the same parsed JUnit
execution before atomic publication; the temporary JUnit file is never itself
the authority. After R3 commits, the fresh red validator performs a detached
worktree replay at exact fresh R3, parses the same five-node partition, and
requires identical case/control classifications and exact retained row values;
nondeterministic JUnit timing bytes are not compared. Replacement S3, strict E3
validation, and independent A3 replay all three controls again at their own
governed source states.

`oracle_patch_paths` is the one-element array naming the characterization
file. During authoring `oracle_patch_sha256` is raw SHA-256 of the stdout bytes
from
`git -c color.ui=false --no-pager diff --no-ext-diff --binary --full-index
<D25> -- tests/characterization/test_sci004_mmode.py` at the cleanly scoped R3
authoring tree. After R3 commits, validation hashes stdout from the same command
with `<fresh-R3>` inserted before `--` and requires the bytes to be identical.
It pins both prior immutable red
records: historical SHA-256
`486705a8d5e51c08f972c91aeae60f0a0bfeef5480b622515282295a6a3cde05`
and correction-#24 supplement SHA-256
`724f75c246ebfcf5956fc40fb2f5e349d91ccca3e6a188b3785a65f4ae4c1e10`.
Its generator is a dedicated atomic no-overwrite mode in
`tools/sci004_mmode_phase3_red.py`; Section 14.1's expected-red disjoint union
covers the historical nodes, correction-#24's six nodes, and these two nodes
exactly once. The three retained passing-control rows are separately
authenticated and never enter that union.

Fresh R3 may change exactly the new supplemental JSON,
`tests/characterization/test_sci004_mmode.py`,
`tests/unit/test_sci004_phase3_dependency.py`,
`tests/unit/test_sci004_phase3_red_failures.py`, and
`tools/sci004_mmode_phase3_red.py`. The last three paths authenticate this
landing, the exact rejected-attempt Git objects and raw digests, the two prior
red blobs, the new supplement, and R3's own exact authority. Replacement S3
directly parents fresh R3 and may change exactly
`src/radiosim/core/result.py`, `tools/sci004_mmode_phase3_evidence.py`,
`tests/unit/test_sci004_phase3_evidence.py`,
`tools/sci004_mmode_phase3_acceptance.py`, and
`tests/unit/test_sci004_phase3_acceptance.py`; delete the rejected evidence
JSON, reproduction Markdown, named host-bound performance JSON, and rejected
acceptance JSON; and perform no other change. The characterization oracle and
all three red records are immutable across S3. The two tools implement the
ruled schema and current-attempt ancestry selection; their strict tests own the
hostile mutations and return all four evidence plus both acceptance binding
constants to literal null sentinels in the same S3 commit. This one commit
therefore restores the exact Section 13/14 S3 state: both fixed canonical
artifact paths absent, the SCI-004 performance directory empty, and all six
bindings null. Deletion of those rejected working-tree drafts is authorized
disposal, not erasure of history: their exact bytes remain authenticated at
the immutable E3/A3 commits above. No archival copy path is created.

Fixed-path retry selection is ancestry-relative, not a whole-history
`--diff-filter=A` search and never “the newest commit”. Given approved source
`S`, current `E` is the first commit in the ordered first-parent ancestry path
emitted by
`git rev-list --first-parent --ancestry-path --reverse S..HEAD`; it must be the
sole-parent direct child of S and its parent-relative
diff must add the fixed evidence path and one host-bound record while changing
only the four approved evidence constants. Given approved evidence `E`, current
`A` is analogously the first commit emitted by
`git rev-list --first-parent --ancestry-path --reverse E..HEAD`; it must be the
sole-parent direct child of E and its diff must add
the fixed acceptance path while changing only the two acceptance constants and
any separately authorized status prose. Zero, multiple, non-first-parent,
non-direct-child, merge, wrong-path, wrong-blob, or wrong-constant candidates
reject. The expected commit is also bound by the active approved constants and
artifact raw digest before its diff is trusted. This rule selects the retry
introduction even though Git history contains earlier add/delete/add events and
cannot silently select an unrelated branch or rejected attempt.

The interval from correction #24 to this landing is exact. Commit
`944e0ee66ebdaffafab86f4f8f4253a404aa902c` is a `superseded red slice`
touching exactly
`docs/development/sci004_mmode_phase3_post_source_red_failures.json`,
`tests/unit/test_io/test_hdf5_result.py`,
`tests/unit/test_sci004_phase3_dependency.py`,
`tests/unit/test_sci004_phase3_red_failures.py`, and
`tools/sci004_mmode_phase3_red.py`;
`b07925ab14b56b3ca0fa863f806290748a31df6b` is a `superseded
implementation` touching exactly `src/radiosim/io/hdf5.py`,
`tests/unit/test_sci004_phase3_evidence.py`, and
`tools/sci004_mmode_phase3_evidence.py`;
`886e62fd9f8328826b388b8960ed7413da26b6d1` is `superseded evidence`
touching exactly `docs/development/sci004_mmode_phase3_evidence.json`,
`docs/development/sci004_mmode_phase3_evidence.md`,
`tests/unit/test_sci004_phase3_evidence.py`, and
`output/benchmarks/reference/sci004/20260825T122048Z-macbook-pro-2.json`, and
carrying the three rejected-output raw digests above; and
`8529da951e2378115ffde8d5da3e2af56f3323d0` is the new narrowly scoped
`rejected acceptance` kind. That commit's sole parent is the rejected E3 and it
touches exactly `docs/development/sci004_mmode_phase3_acceptance.json` and
`tests/unit/test_sci004_phase3_acceptance.py`, with the latter's diff exactly
the two approved constant assignments. Its
verdict is `REJECT`, and it binds both the canonical artifact and external
review digests above. A rejected-acceptance commit is immutable, unlocks no
successor, cannot change ROADMAP to DONE, and cannot authorize closure C; the
next governed R or correction authenticates its full SHA, parent, exact paths,
verdict, reviewer digest, and raw artifact digest. Its current-tree artifact is
disposed only by the replacement S named here. Correction #24 becomes a
`superseded design` chain commit and this landing becomes operative `D`.

After dual independent review accepts and lands these exact normative bytes,
the retry succession is exactly
`8529da951e2378115ffde8d5da3e2af56f3323d0 -> correction #25 -> fresh
fingerprint-delta R3 -> replacement S3 -> fresh E3 -> fresh independent A3`.
Fresh E3 changes only the newly generated evidence JSON and reproduction
Markdown, one newly named host-bound performance JSON, and the four evidence
constants. Fresh A3 rederives every Section 14.3 oracle again — no passing row
from the rejected attempt carries — and changes only its newly generated
acceptance JSON, the two acceptance constants, and explicitly authorized status
prose. Atomic no-overwrite generation is unchanged. This correction is
design-only: it accepts no source slice or evidence, does not reverse A3's
REJECT, does not accept M3, reopens no accepted M1/M2 artifact, and does not
close SCI-004. SCI-004 remains ROADMAP; closure C remains locked; SCI-005 Stage
3 is neither a prerequisite nor a claim; and no accelerator, diffuse/public
HEALPix, non-scalar-beam, public end-to-end backend-wiring, general performance,
or speedup claim is licensed. No dependency, lock file, workflow, tolerance,
simulator submodule, or previously accepted artifact becomes writable.
**Review status: independent physics/governance and computational correction
reviews pending against pinned candidate bytes and the complete parent-relative
diff.**
**Final review verification:** the reviewed pre-landing design was 370,496 raw bytes with SHA-256 `5d54c4b8c5c0312b29d2391c0de76b51a004b6c0605d2543a51ae2a46bbff2a6` and parent-relative diff SHA-256 `1052133587a3af0489cf079c69e2f7a5b8869f20959bdc3d1cda8ed09d7c1acb`, the reviewed pre-landing ledger was 51,872 raw bytes with SHA-256 `f79329d0e0438ce5ff5c2c65d0b443fa04dae0bffeded5f20b750b64415105af` and parent-relative diff SHA-256 `d9d30802b46941cba5d2c52ce7cd1ef405d0bc966a4a5fcf1dc4d4d12dabff44`, the complete fixed-order design-then-ledger parent-relative diff SHA-256 was `ccea0e4e0477ea43174f64aece99369c31b4ce221cdd0b16d38788e1bdd4dc76`, and after two bounded fix rounds the fresh round-3 reviewers `/root/physics_governance_review` and `/root/computational_provenance_review` each returned exact `ACCEPT`; only this sentence and the companion ledger verdict completion postdate those reviewed pins.

**Bounded correction #24 — 2026-08-25 (sampled current-process RSS and
the polarized-HDF5 post-source oracle).** The committed phase-3 source
slice exposed two bounded defects before `E3`. First, Section 11's
`python_heap_tracemalloc_scoped_v1` instrument changes the execution it
purports to measure: the ordinary 115-second solver call takes about
2 hours 15 minutes under allocation-event tracing, and the required
three fixture-group calls therefore dominate an approximately eight-hour
attempt. The first attempt exposed the cost before a real HDF5 failure, and
the second was interrupted rather than spending the same cost blindly;
together with the allocation-event mechanism, those observations establish
the slowdown as instrumentation overhead rather than an estimate of the
untraced solver's resource behavior. The
ruling replaces that host method with Section 11's exact external
`process_rss_sampled_delta_v1` contract. The sampler is a separate
standard-library-only Python process, observes the generator process at a
10 ms monotonic cadence through `/proc/<pid>/statm` on Linux or
`proc_pidinfo(PROC_PIDTASKINFO)` on Darwin, and cannot add a sampling
thread or allocation hook to the measured process. The three fixture
groups and nine-row product are unchanged; each group still receives one
separate untimed whole-solver measurement copied to its three backend
rows, the memory object keeps exactly its existing ten keys, and the two
budget inequalities plus the recomputed estimate-coverage boolean remain
unchanged.

Second, the four R3 HDF5 nodes construct their nominally polarized
`MModeSolverSnapshot` through `build_mmode_result`'s scalar
`not_applicable_scalar_m1` default. They therefore never exercise the
genuine post-M2 six-key `tangent_polarization_frame` mapping, although `E3`'s
output case uses `mmode_point_full_stokes` and does exercise that mapping.
The committed reader accepts the six field names but authenticates each value
only as a non-empty string, so it does not enforce Section 5.1's schema,
coordinate-frame set, axes, position-angle, linear-complex, or Stokes-V
literals before payload access. Both the oracle and that bounded source check
are incomplete. The fresh R3 re-cut changes the same four existing HDF5 nodes
to pass
`TangentPolarizationFrame.canonical("icrs").as_mapping()` and keeps their
node IDs and governed requirements. The replacement S3 constructs
`TangentPolarizationFrame` from every mapping and translates any invalid
field value to `UnsafeResultInputError` before any science dataset is read.
The fresh R3, not S3, owns a six-case hostile-reader oracle that mutates each
declared field in turn while preserving the valid outer solver-key order and
requires that early rejection.

The retained red-failure record is not regenerated. `S3` already exists and
the four historical nodes are green in the operative tree, so Section 13.7's
post-source retention rule preserves the last genuinely observed record bytes
at their historical
`design_sha=923ae332c02d9b2d4edfddf09d1d61241e9d5a63`; the re-cut validator
authenticates that binding as a header-enumerated chain commit connected to
this landing instead of fabricating a new `expected-red-confirmed`
observation. The required unshadowed ephemeral replay at
`a61526d686ab768f05ecffa80cfd6223d4ee4c62`, with only the six-case hostile
oracle applied, ran those six cases, the four canonical HDF5 nodes, and their
existing green control serially on 2026-08-25. The five controls passed; all
six hostile cases failed because the current reader reached the late generic
`HDF5 result failed canonical model or fingerprint validation` error instead
of the required pre-payload `HDF5 solver_json is invalid` rejection (six
failed, five passed in 3.09 seconds). That replay isolates the exact-value
authentication defect after the outer key-order repair; it is review evidence,
not itself the retained supplemental record.

Because those six cases are a newly discovered red delta rather than the
historical nodes, R3 retains their own separate canonical artifact at
`docs/development/sci004_mmode_phase3_post_source_red_failures.json`, with
schema `radiosim.sci004.mmode-phase3-post-source-red-failures.v1` and
status `post-source-expected-red-confirmed`. It binds `design_sha` to this
correction, `pre_fix_source_sha` to the exact superseded source `a61526d6…`,
and its six cases to the `SCI004_PHASE3_POST_SOURCE_RED_CASES` table; it also
pins the historical record's raw SHA-256 and the exact HDF5-oracle patch
without copying or changing the historical cases.
The historical record continues to enumerate exactly its original phase-red
nodes, while Section 14.1 requires the union of that immutable node set and the
six supplemental nodes to be complete. E3's `red_failure_record` join validates
and binds both artifacts. This is the narrowly scoped post-source red-delta
mechanism of Sections 13.7, 14.1, and 14.2, not a licence to append observations
to an old record.

This one correction governs both findings because they are post-S3 M3
findings and require the same Section 13.7 red/source reopening; splitting
them would duplicate the correction reviews, post-source red-delta re-cut, and
replacement S3 without creating an independently meaningful boundary.
It reopens `7070cc3ddb1c2557d02e4a3f2a89b907575bed0b` as a
`superseded red slice`. The fresh post-source red-delta R3 directly parents this
landing, freezes `APPROVED_SCI004_D_SHA` to this landing, retains the
SCI-005 dependency certificate and phase-3 red-failure artifact
byte-for-byte, pins the latter's raw SHA-256 as
`486705a8d5e51c08f972c91aeae60f0a0bfeef5480b622515282295a6a3cde05`,
and may change exactly
`docs/development/sci004_mmode_phase3_post_source_red_failures.json`,
`tests/unit/test_io/test_hdf5_result.py`,
`tests/unit/test_sci004_phase3_dependency.py`,
`tests/unit/test_sci004_phase3_red_failures.py`, and
`tools/sci004_mmode_phase3_red.py` for the six-key and hostile oracles,
supplemental generation/validation, chain/rebind work, and raw-byte retention
check just described. The tool's historical generation mode remains
fail-closed and cannot overwrite or regenerate the old
`expected-red-confirmed` artifact; its dedicated no-overwrite supplemental
mode emits only the six-case red delta. It also reopens
`a61526d686ab768f05ecffa80cfd6223d4ee4c62` as a `superseded
implementation`. That commit touched exactly `.gitignore`,
`docs/api/io.rst`, `docs/user_guide/backends.rst`,
`docs/user_guide/configuration_support.rst`,
`output/benchmarks/reference/README.md`,
`src/radiosim/benchmarks/__init__.py`,
`src/radiosim/core/mmode/solver.py`, `src/radiosim/core/result.py`,
`src/radiosim/io/hdf5.py`, `src/radiosim/io/standard_visibility.py`,
`src/radiosim/io/summary_json.py`,
`tests/unit/test_sci004_phase3_acceptance.py`,
`tests/unit/test_sci004_phase3_evidence.py`,
`tools/sci004_mmode_phase3_acceptance.py`, and
`tools/sci004_mmode_phase3_evidence.py`. The replacement S3 directly
parents the fresh R3 and may change exactly
`src/radiosim/io/hdf5.py`,
`tools/sci004_mmode_phase3_evidence.py`, and
`tests/unit/test_sci004_phase3_evidence.py`; every other production byte flows
through unchanged from the superseded implementation.

The live succession folds to a plain `R3 -> S3` edge and a longer
`G3 ->* R3` edge. Its interval is exactly
`[62a7d3d90dcbf0488e8b7c875ae5f95acba007b6,
e7902d04ce042bd3a16ab9ae3a336695e971db81,
53ee53c3b829512ef02f81215238090be63937d9,
a07279f4e1220f4e064d747406350df6fd1190fb,
29c702cfc824ad73b2e0aeacd5b4b23bcc6c18cf,
83d98f70fef0bf35977a3b6d4a7101ff67a7a953,
c6cc74bb88bb123b20b4c549bc92da73cc057c1e,
923ae332c02d9b2d4edfddf09d1d61241e9d5a63,
7070cc3ddb1c2557d02e4a3f2a89b907575bed0b,
2422c5765a82e55328c25bb3b8fc08e8377c176f,
6fb8b0a8d54bcf946b32a777f69359c8b83bd527,
a61526d686ab768f05ecffa80cfd6223d4ee4c62, this landing]`.
The four `R3` commits are `superseded red slice` interval commits;
`e7902d04…`, `53ee53c3…`, `29c702cf…`, `83d98f70…`,
`923ae332…`, `2422c576…`, and `6fb8b0a8…` are `superseded design`
interval commits; `a61526d6…` is the one `superseded implementation`;
and this landing is the operative `D`. The fresh R3 then directly
parents this landing and the replacement S3 directly parents that R3.
Section 14.4's equation, star attribution, and S-edge restatement change
accordingly.

The repository-wide pin/oracle audit found the live host-method literal
only in this memo, `tools/sci004_mmode_phase3_evidence.py`, and its strict
unit validator `tests/unit/test_sci004_phase3_evidence.py`. The phase-3
workload/red pins treat `memory` as an opaque outer key and do not pin its
method. Accepted M2 evidence and its tool/tests use their distinct frozen
M2 schema, and the generic PERF-001/Tier-6 benchmark tracemalloc surfaces
govern different records; all remain immutable and unchanged. This
correction supersedes the honest-memory-boolean landing
`6fb8b0a8d54bcf946b32a777f69359c8b83bd527` as the operative design;
the correction itself touches only
`docs/development/sci004_mmode_design.md` and
`PostTier8RemediationPlan.md`, implements no source, accepts no phase,
starts no evidence run, and does not close the register row. **Review
status: dual `ACCEPT`.** Its exact pre-landing memo file bytes
(`sha256:4b595da0c6946ce333c795343ef1a7db7e8c16a7ff1dc05c54af550d8f15b107`),
memo parent-relative diff
(`sha256:6ae375bd2a9d1e880dfa6f9af051f700a4ae40ea433104eaabd3d53506133bc6`),
and ledger parent-relative diff
(`sha256:79a9b91ee2e1156ce5c5866db5d1d28c6fd60d8cfae0ac90b5211a521555aca8`)
received separate independent third-round delta-reconfirmation on 2026-08-25 —
physics/governance and computational, both `ACCEPT` — after the two prior
pinned drafts' blockers were corrected; those three pins name the reviewed
bytes, this verdict wording and the companion ledger verdict necessarily
postdate them under Section 13.7, the fresh dependency validator authenticates
all three from this accepted header, and this correction's landing commit is
the operative `D`.

**Bounded correction — 2026-08-25 (the honest memory boolean).**
Executing the phase-3 evidence generator measured Section 11's memory
predicate unsatisfiable: it required
`measured_host_peak_bytes <= estimated_host_peak_bytes`, but Section
9's `get_memory_estimate()` models the dense pipeline's seven
components by design (`13.8 MiB` on all three acceptance fixtures,
`14,180,352` bytes of it the quadrature directions/weights/Jones term),
while `_mmode_pipeline` builds the every-run Section 4.2 frame
certificate on the same call and its retained row — the complete
Section 14.2 row with its Section 12.1 ledgers — deep-measures
`29,396,321`–`32,629,309` bytes, a strict lower bound on the scoped
tracemalloc peak, so the inequality fails on every one of the nine rows
(`2.0×`–`2.25×`) while both budget inequalities hold with a
thirty-fold margin. Closing it inside the implementation would need an
estimator change no Section 13.5 grant covers, and choosing the
estimate after seeing the measurement would be the condemned
self-comparison form. The ruling makes the boolean honest:
`estimate_covers_measured_host_peak` is the measured relation, retained
as observed and `false` by construction at this phase with its reason
mandatory in `host_measurement_limitations`; the hard predicates are
the two budget inequalities; and Section 9 states the exclusion plainly
— the certificate's retained ledgers are outside the dense estimate and
measured separately. No grant, oracle, or schema-key change is needed:
the memory key list is unchanged and nothing committed pins the
boolean's expected value — verified against the workload-key pins, the
characterization preimage, and the red record. It supersedes the
scalar-table-kernel-exception landing
`2422c5765a82e55328c25bb3b8fc08e8377c176f` as the operative `D`; that
commit becomes a `superseded design` interval commit on the
header-enumerated `D0 -> D` chain, and it touched exactly
`docs/development/sci004_mmode_design.md` and
`PostTier8RemediationPlan.md`. The starred `R3 -> S3` interval grows to
exactly `[2422c576…, this landing]`; `S3` directly parents this
landing per the unchanged rule, and Section 14.4 needs no edit. This
correction is design-only: it implements no solver, accepts no phase,
and does not close the register row. Its exact pre-landing file bytes
(`sha256:1f51e88500392f8c33ff92646ac52e44d755ef1d554a7256bc99a4a43d75ad29`)
and parent-relative diff
(`sha256:b349f03fb48739550b6f8b50881d8bcff9cbf85aaa04a1fa82f3ec615086c40d`)
received separate independent reviews on 2026-08-25 —
physics/governance and computational, both `ACCEPT` with no blocking
findings on the first round. Both reviews reproduced the estimator
figures live at the pinned dimensions (`total_bytes = 14,471,104`,
the quadrature term `14,180,352`, byte-exact), verified the
certificate's unconditional in-call construction and the estimator's
seven-component exclusion by direct code reading, verified the
no-change-needed claim at whole-repository scope (the boolean's
expected value pinned nowhere committed), judged the honest-boolean
disposition the least dishonest of the three options — the estimator
change being literally ungranted and the scope narrowing making the
number misdescribe its own literal — and enumerated the exact
uncommitted `S3` tool lines that still enforce the superseded
predicate as the bounded follow-up work this correction licenses.
Neither review completed an independent end-to-end tracemalloc
reproduction of the measured peak inside its window; the structural
lower-bound argument and the internal ratio arithmetic were verified
instead, and the `E3` run itself retains the measured value. This
correction's landing commit is the operative `D` of Section 13.7.

**Bounded correction — 2026-08-24 (the scalar-table kernel exception).**
Implementing the honest-backend-axis record measured a second gap in
the same area: the routed per-`m` contraction kernel hard-refuses a
one-field block table — `ValueError: the forward product covers exactly
Section 5.3's four science fields`, reproduced live — and two of the
three fixture groups resolve scalar payloads (measured
`max|QUV| = 0.0` for `mmode_single_scalar_mode` and
`mmode_point_stokes_i`; `2.0` for `mmode_point_full_stokes`), so the
`kernel_backend_block` ruled `measured` on every JAX and Dask row is
unconstructible for the two scalar groups, and running the kernel on
synthetic four-field data instead would be a microbenchmark describing
nothing the group's own solve does — the condemned class. Separately,
the shared row series named `per_m_contraction` and `synthesis` have no
in-solve split point: production's `contract_and_synthesize` interleaves
the contraction with the sample accumulation, so a per-stage split of
the shared series would be an invented number. The rulings: the kernel
block gains a third uniform status, `not_applicable_scalar_table`, on
the JAX and Dask rows of scalar groups — the polarized group carries
the real kernel evidence, exactly what its own solve exercises — and
the shared timing series fuses to one honest
`dense_contraction_and_synthesis` series, with the two stage names
reserved to the kernel blocks where they are real. No grant or
red-oracle change is needed, verified completely rather than cited
partially: the committed characterization preimage's
`kernel_stages` line names the kernel-block stages, which are
unchanged; its adjacent
`kernel_backend_block_status: not_applicable|measured` line states the
enumeration true at the retained record's own frozen binding
`923ae332…` — a retained record is immutable truth at its `design_sha`,
exactly as every superseded record before it — and the preimage bytes
are consumed by the tracked generator solely as a `sha256` provenance
preimage, never parsed or asserted against live values, so the third
status falsifies no committed oracle; the workload-key and claims pins
carry `timings` and `kernel_backend_block` only as opaque keys, never
sub-key literals. The next governed re-cut, whenever one occurs for its
own reasons, refreshes the enumeration line as ordinary red-slice
work. It supersedes the honest-backend-axis
landing `923ae332c02d9b2d4edfddf09d1d61241e9d5a63` as the operative
`D`; that commit becomes a `superseded design` interval commit on the
header-enumerated `D0 -> D` chain, and it touched exactly
`docs/development/sci004_mmode_design.md` and
`PostTier8RemediationPlan.md`. Because this landing sits between the
third re-cut `7070cc3ddb1c2557d02e4a3f2a89b907575bed0b` and the future
`S3`, the `R3 -> S3` edge is starred per Section 13.7's
later-phase-commit rule with this landing its exactly one interval
commit, `S3` directly parents this landing, and Section 14.4's order
equation, star attribution, and `S`-edge qualifier are amended
accordingly in this same diff; the third re-cut's frozen
`APPROVED_SCI004_D_SHA` binding (`923ae332…`) is unchanged per the
Section 14.0 rule. This correction is design-only: it implements no
solver, accepts no phase, and does not close the register row. Its
exact pre-landing file bytes
(`sha256:b7f69e7aff9945ea9c35a22062ccfad7f9c63beabf1a53fcabfc6c7997a0b33e`)
and parent-relative diff
(`sha256:49100c3684d088273382316ef2e0dba6079c411065cca6140b28c50588cbba9b`)
received separate independent reviews on 2026-08-24 —
physics/governance and computational, both `ACCEPT` after one applied
fix round that settled a genuine reviewer split: the governance review
found the retained preimage's adjacent two-value status enumeration and
held the draft's partial citation to the standard its predecessor was
rejected under, while the computational review traced the same bytes to
their sole `sha256` consumption and the record's own `923ae332…`
binding where the enumeration was true; the completed verification
sentence carries all four facts, and the governance reconfirmation then
identified the dispositive rule — live, forward-consumed pins trigger
re-cuts; archival, `sha256`-only preimages are immutable truth at their
binding, Section 13.7's superseded-record clause firing only when the
generating commit itself is reopened — verifying it against the rule
text and every prior re-cut precedent rather than accepting the
reconciliation on faith, with one advisory that the "every superseded
record" phrasing rests on the M2 pattern precedent rather than a
literal in-flight-record precedent. This correction's landing commit is
the operative `D` of Section 13.7.

**Bounded correction — 2026-08-24 (the honest backend axis).** Building
the Section 11 performance-record generator measured the nine-row
backend axis vacuous end to end: the public solve path's
`contract_and_synthesize` takes no backend parameter, the two
backend-routed kernels have zero call sites on the solve path,
`transfer.py` hard-codes the NumPy backend, and `request.backend`
touches only the finished cube — so three end-to-end solves under
`numpy`, `jax`, and `dask` produced bit-identical cubes (one distinct
identity across the three, the JAX run "faster" than NumPy as pure
noise), and a conforming record would have named synchronization that
synchronized nothing, device memory never used, and exact-zero
deviations that were self-comparisons — the vacuous-evidence class the
accepted-capability correction already condemned. The accepted M2
backend evidence is kernel-level and genuine (measured `1.9e-15` JAX
deviation at the per-`m` contraction), and Section 9's policy literal
admits exactly those two stages. The ruling keeps the nine-row
inventory intact and makes the row
content honest: every row carries `dense_execution = numpy_host_v1`;
each fixture group measures its end-to-end series once and retains a
`dense_invariance` object with the three identical cube digests as a
measured fact; the JAX and Dask rows carry `kernel_backend_block`
measurements of the two admitted stages where the synchronization,
native memory, and `backend_comparison` evidence is real; the
comparison's reference is re-anchored to the NumPy kernel output; the
workload claims array gains `mmode_end_to_end_backend_execution`
(rendering the tracked five-claim pins stale by design change); and
Section 9 states the wiring deferral plainly — future red-sliced work,
with `A3`'s `claims_not_licensed` carrying the third deferral. The new
row schema renders two tracked pin sets stale by design change, not
defect — `SCI004_WORKLOAD_KEYS` and the claims array in
`src/radiosim/benchmarks/__init__.py`, and their by-value red-oracle
pins in `tests/performance/test_sci004_mmode.py` — so this record
widens both scoped grants to name the workload-row-key-set and
claims-array literals, and reopens, as a `superseded red slice`
interval commit for the next governed re-cut, the red slice
`c6cc74bb88bb123b20b4c549bc92da73cc057c1e`; the re-cut aligns those
pinned literals to this schema as red-slice work, rebinds
`APPROVED_SCI004_D_SHA` to this correction's landing per Section 14.0's
fresh-`R` rule, and regenerates the red record from a globally clean
checkout of the re-cut candidate, Section 14.4's venue vocabulary,
where every phase-3 oracle is genuinely red; uncommitted source work
stays outside that observation tree. It
supersedes the retained-evidence-surfaces landing
`83d98f70fef0bf35977a3b6d4a7101ff67a7a953` as the operative `D`; that
commit becomes a `superseded design` interval commit on the
header-enumerated `D0 -> D` chain, and it touched exactly
`docs/development/sci004_mmode_design.md` and
`PostTier8RemediationPlan.md`. The succession folds again: the
`G3 -> R3` starred interval grows to exactly `[62a7d3d9…, e7902d04…,
53ee53c3…, a07279f4…, 29c702cf…, 83d98f70…, c6cc74bb…, this landing]`,
the re-cut `R3` directly parents this landing, the `R3 -> S3` edge
stays plain — `S3` plainly parents the re-cut `R3` — and Section 14.4
needs no net edit from its current state. Its exact pre-landing file
bytes
(`sha256:4f60cc3b464658a5b8adfcd9dea8417a9651f439c686b34715f662d40652dc5e`)
and parent-relative diff
(`sha256:53ecb8da1ebf1d6e36599863c11cbf70ef5a61fb0322a1f44f5e908581484aad`)
received separate independent reviews on 2026-08-24 —
physics/governance and computational, both `ACCEPT` after one applied
fix round in which both reviews independently verified the end-to-end
vacuity from the solver's own code and rejected the draft on converging
blocking findings: the closed-schema convention demanded exhaustive
`has exactly` enumerations for every new object, the draft's
"no red-oracle change is needed" claim was falsified by the fresh
red slice's own tracked workload-key and claims pins — forcing the two
grant widenings and the reopening this record now carries — and the
row-level comparison's field names were reconciled with its content by
the no-parity label rather than a silent repurposing. Both
reconfirmations verified the eight-commit interval against real git
ancestry and the Section 14.4 net-revert byte-identically. This
correction's landing commit is the operative `D` of Section 13.7. This correction is design-only: it implements no
solver, accepts no phase, and does not close the register row.

**Bounded correction — 2026-08-24 (the retained-evidence surfaces follow
the envelope).** Implementing the phase-3 evidence machinery measured
four ruled surfaces the accepted-capability correction left
contradicting its own amended Section 11, and one accepted red oracle
its performance-product successor falsifies. Section 14.2's M3
`fingerprint_rows` demanded "exactly seven rows" against the four-family
envelope; its `ci_artifacts` demanded the seven-family × six-CI-cell
Cartesian with `run_id`/`job_id`/`artifact_id` fields only a remote
workflow can mint, against the amended harvest sentence binding the
initially run cells and against a local `E3` venue that neither pushes
nor runs workflows — the accepted re-cut `R3` already implements the
narrowed reading (its observation-set oracle quotes the amended harvest
sentence verbatim and no remote-artifact oracle exists); Section 15's
gate sentence and Section 12.2's item 10 repeated the six-cell and
remote-artifact demands; and Section 14.3's `A3` sentence required
"every fingerprint/remote artifact". All five are conformed: four
fingerprint rows in the amended family order; `ci_artifacts` narrowed to
the harvested cells authenticated from the retained observation-set
surface, with remote cells and their workflow artifacts entering
afterwards by the standing admission discipline; the Section 15 and
12.2 sentences and the `A3` clause conformed to match. Separately, the
performance-product alignment the previous correction ordered falsifies
`tests/performance/test_sci004_mmode.py::test_the_official_v1_inventory_is_the_exact_nine_row_product`,
an accepted M2 red oracle that pins the superseded product by value
through an import — invisible to the standard gate because the file is
`performance`+`slow`-marked, and invisible to the previous reviews'
live-byte checks because the pin is import-mediated, a recorded lesson;
Section 13.5's own rule — "No `S` commit
may edit a red oracle or its record. A proved-defective oracle requires
a bounded design successor and a fresh `R`" — decides the repair route:
this correction is the bounded design successor, Section 13.5's `R3`
list now grants that file scoped to the pinned fixture-product literals
only (a red-oracle edit belonging to an `R` commit, with the lapsed
`R2`-grant disclosure), and the phase-3 red slice
`a07279f4e1220f4e064d747406350df6fd1190fb` is reopened as a `superseded
red slice` interval commit for a governed re-cut. The re-cut applies
exactly that literal update as red-slice work, rebinds
`APPROVED_SCI004_D_SHA` to this correction's landing (Section 14.0's
"the operative `D` current at that phase's `R`" — a fresh `R` takes the
`D` current at its own cut, superseding the performance-product
record's frozen-binding note for phase 3), rebinds the derived anchor
to the first successor of this landing, and regenerates the red record
from a globally clean checkout of the re-cut candidate, Section 14.4's
venue vocabulary, where every phase-3 oracle is genuinely red;
uncommitted source work stays outside that observation tree. The
succession therefore folds: the `G3 -> R3` starred interval grows to
exactly `[62a7d3d9…, e7902d04…, 53ee53c3…, a07279f4…, 29c702cf…, this
landing]`, the re-cut `R3` directly parents this landing, the
performance-product correction's `R3 -> S3` star reverts — `S3` plainly
parents the re-cut `R3` — and Section 14.4's equation, attribution, and
`S`-edge qualifier are amended accordingly in this same diff. It supersedes the
performance-product landing
`29c702cfc824ad73b2e0aeacd5b4b23bcc6c18cf` as the operative `D`; that
commit becomes a `superseded design` interval commit on the
header-enumerated `D0 -> D` chain, and it touched exactly
`docs/development/sci004_mmode_design.md` and
`PostTier8RemediationPlan.md`. This correction is design-only: it
implements no solver, accepts no phase, and does not close the register
row. Its exact pre-landing file bytes
(`sha256:d53ed7f0de129ec2dd6f7ca760b110d83b2fbcbc2527aac744a084ba38a40b5a`)
and parent-relative diff
(`sha256:80fe0e816c7b3fba169736f92aa23f0983de36224420913eb3e106b1a5a067d2`)
received separate independent reviews on 2026-08-24 —
physics/governance and computational, both `ACCEPT` after one applied
fix round: the governance review proved the draft's `S3` grant of the
falsified red oracle violated Section 13.5's own twice-stated rule —
which the computational review's first pass had read without
cross-checking, its reconfirmation says so plainly — and the repair
pivoted to the ruled route quoted verbatim in this record: the bounded
design successor plus a fresh `R`. Both reconfirmations verified the
folded six-commit interval against real git ancestry, the star
reversion's document-wide consistency, and — substantively — that the
oracle is genuinely red at a clean re-cut checkout because the aligned
constant exists only in the uncommitted source tree. One advisory
carries to the re-cut author: the dependency validator's
`R3_AUTHORIZED_PATHS` frozenset must gain the granted performance file
in the same re-cut, or the derived anchor will reject its own commit.
This correction's landing commit is the operative `D` of Section 13.7.

**Bounded correction — 2026-08-24 (the performance product follows the
envelope).** Preparing the phase-3 evidence machinery surfaced the one
committed constant the accepted-capability correction left contradicting
it: the accepted `S2` slice's `SCI004_FIXTURE_IDS` in
`src/radiosim/benchmarks/__init__.py` still enumerates the superseded
point/HEALPix/hybrid performance product, the file is granted only by
the closed `S2` list — that grant belonged to the accepted M2 phase and
conveys nothing to `S3` — and the `E3` generator must consume that
surface rather than work around a committed constant that contradicts
the operative design. Section 13.5's `S3` list now grants the file,
scoped to the `SCI004_FIXTURE_IDS` constant and its dependent fixture
literals only, aligning the product to the envelope's three point
groups. The constant is superseded design fallout, not an `S2` defect:
`S2` implemented Section 11 exactly as then ruled, so no phase artifact
is touched and no reopening occurs. Because this landing sits between
the re-cut `R3` `a07279f4e1220f4e064d747406350df6fd1190fb` and the
future `S3`, the `R3 -> S3` edge is starred per Section 13.7's
later-phase-commit rule with this landing its exactly one interval
commit, `S3` directly parents this landing, and Section 14.4's order
equation, star attribution, and `S`-edge qualifier are amended
accordingly in this same diff; the phase's frozen `R3` binding
(`APPROVED_SCI004_D_SHA` = the accepted-capability landing) is
unchanged per the Section 14.0 rule, so every phase-3 `design_sha`
still names that frozen binding while this landing holds the operative
`D`. It supersedes the accepted-capability-characterization-envelope
landing `53ee53c3b829512ef02f81215238090be63937d9` as the operative
`D`; that commit becomes a `superseded design` interval commit on the
header-enumerated `D0 -> D` chain, and it touched exactly
`docs/development/sci004_mmode_design.md` and
`PostTier8RemediationPlan.md`. This correction is design-only: it
implements no solver, accepts no phase, and does not close the register
row. Its exact pre-landing file bytes
(`sha256:f32a4f6793abb983c42f6605b444f9eecaf706be09843dd914b952f4cae43e14`)
and parent-relative diff
(`sha256:f2b5f1abfe7891dc40788713db106c5ef6bbcedb65d6f8fa1b36b92ab142b975`)
received separate independent reviews on 2026-08-24 —
physics/governance and computational, both `ACCEPT` with no blocking
findings on the first round. Both reviews verified the superseded
constant at its source lines, the S2-only lapsed grant, the four-star
Section 14.4 consistency, the frozen `R3` binding's independence from
this landing (the anchor derivation's sole successor is fixed at the
re-cut as a structural git fact), and that nothing accepted pins the
granted file's live bytes; the governance review additionally confirmed
the fallout-not-defect classification against Section 13.7's own
kind definitions. This correction's landing commit is the operative `D`
of Section 13.7.

**Bounded correction — 2026-08-24 (the accepted-capability
characterization envelope).** Implementing the phase-3 source slice
fired Section 13.5's pause rule with measurements on every point. The
public m-mode solve path carries only point components — the HEALPix and
hybrid harmonic machinery accepted at M2 lives in the bounded evidence
fixture — so the two former HEALPix families produced identically zero
cubes passing the Section 7.3 gate vacuously with one shared
`scientific_sha256` (`eb570f32…`), the former hybrid family silently
dropped its diffuse half while its gate passed (`components=['point']`
for a two-representation payload), and a `beams.squint` fixture failed
after `108.8 s` with a `BeamEvaluationError`; separately, three former
family fixtures could not construct their configured skies, the former
`mmode_nonscalar_east_x` reproduced `mmode_point_full_stokes` byte for
byte (`3d88dab0…`) because the shipped default receptor set is east-X,
and two output oracles are defective against accepted literals — the
UVFITS oracle asserts a phase-centre frame the accepted
`ProjectedPhaseCenter` forbids (`icrs` is the ruled literal, with the
original zenith-drift snapshot retained and measured equal), and the
HDF5 fingerprint oracle omits the accepted bytes-decode idiom. The
rulings: Section 11's characterization set narrows to the four-family
accepted-capability envelope (with `mmode_circular_receptor` replacing
the byte-identical duplicate, subject to Section 7.3 qualification at
`S3`), the performance record's fixture product becomes the three
point-family groups, the public path gains two fail-fast Section 8
rejections (`mmode_public_components`, `mmode_public_beam`) under a
scoped `S3` solver grant — closing the silent-drop defect by rejection,
never by unreviewed wiring through an outputs phase — the harvest
sentence binds the initially run cells with later cells entering by the
standing admission discipline, and the family record's derived digests
receive their namespaced characterization domains. The deferrals are
recorded: public diffuse, hybrid, and non-scalar-beam m-mode runs are
future red-sliced work, and `A3`'s `claims_not_licensed` must carry
both. It reopens, as a `superseded red slice` interval commit for a
governed re-cut, the phase-3 red slice
`62a7d3d90dcbf0488e8b7c875ae5f95acba007b6`; the re-cut repairs the five
defective oracles to their measured corrected forms, replaces the
removed family fixtures per the new set, adds the two rejection oracles
and an end-to-end family-run green control — the prior control proved
only that configs resolve, the root cause — rebinds the dependency
validator's derived `R3` anchor and its `R3^` assertion to the starred
edge (the `--diff-filter=A` derivation resolves the superseded
add-commit as an immutable git fact forever and would silently
authenticate it; the re-cut derivation must resolve the live re-cut
commit and assert that it directly parents this correction's landing,
with the docstring and error text updated to match), and regenerates
the red record from a globally clean checkout of the re-cut candidate,
Section 14.4's venue vocabulary, where every phase-3 oracle is
genuinely red; uncommitted source work stays outside that observation
tree. Per Section
13.7's reopened-phase rule the re-cut `R3` directly parents this
correction's landing, the `G3 -> R3` edge is the starred one with
interval `[62a7d3d9…, e7902d04…, this landing]`, and the
un-ignoring correction's `R3 -> S3` star reverts — `S3` plainly parents
the re-cut `R3` — with Section 14.4's equation, attribution, and edge
restatements amended accordingly in this same diff. It supersedes the
un-ignoring landing `e7902d04ce042bd3a16ab9ae3a336695e971db81` as the
operative `D`; that commit becomes a `superseded design` interval commit
on the header-enumerated `D0 -> D` chain, and it touched exactly
`docs/development/sci004_mmode_design.md` and
`PostTier8RemediationPlan.md`. This correction is design-only: it
implements no solver, accepts no phase, and does not close the register
row. Its exact pre-landing file bytes
(`sha256:6ea19f19ee1d368687043477140b7d938d4668ec2aca0c7123a824484f3a0d4d`)
and parent-relative diff
(`sha256:dec9df0fb5b37edcb87067092288f74bc92861c6163a08a0344cc0246819739a`)
received separate independent reviews on 2026-08-24 —
physics/governance and computational, both `ACCEPT` after one applied
fix round. Both reviews independently verified every measurement at
code level — the hard-coded point-only public path, the silent-drop and
vacuous-zero mechanisms, the duplicate family, both defective output
oracles reproduced live — and the governance review judged the
narrow-and-reject ruling correct on the merits, noting the accepted M2
record never licensed public-path HEALPix support, so the deferral is
honest rather than a retreat. The fix round closed the two blocking
findings: Section 13.2's flat `R3^==G3` restatement (the litigated
defect class, caught by both reviews) now carries the unless-starred
qualifier, and the re-cut mandate explicitly orders the dependency
validator's anchor rebind after the computational review proved the
`--diff-filter=A` derivation would silently authenticate the superseded
add-commit forever. This correction's landing commit is the operative
`D` of Section 13.7.

**Bounded correction — 2026-08-24 (un-ignoring the granted reference
records).** Authoring the phase-3 red slice measured the third
unexercisable-grant-class defect: Section 13.5 grants `S3` the
`output/benchmarks/reference/README.md` and `E3` the one
`output/benchmarks/reference/sci004/` performance record, but
`.gitignore`'s `output/benchmarks/reference/*` rule excludes both
(measured by `git check-ignore -v`, rule line 184), and no M3 phase list
grants `.gitignore` — the same class Section 13.2 records for the WP-7
`pixi.toml` freeze and the struck `S2` grant of
`src/radiosim/benchmarks/record.py`. Section 13.5's `S3` list now
grants `.gitignore`, scoped to the negation rules un-ignoring exactly
those two granted paths in the established perf001 block form. Because
the phase-3 red slice `62a7d3d90dcbf0488e8b7c875ae5f95acba007b6` landed
directly on its gate tip before this correction, this landing sits
between `R3` and `S3`: per Section 13.7's later-phase-commit rule the
`R3 -> S3` edge is starred, `S3` directly parents this correction's
landing, and Section 14.4's order equation and its `S`-edge restatement
now show that edge starred; the interval commit is exactly this landing.
Its exact pre-landing file bytes
(`sha256:20e104fe73130431ca1122905d3e99a9236981fe5cba067f6601008a15c121ea`)
and parent-relative diff
(`sha256:ce79e0ec83552f8453968ef08ce70e6a28d439ca7dc202df02beda0d67270752`)
received separate independent reviews on 2026-08-24 —
physics/governance and computational, both `ACCEPT` with no blocking
findings: both reproduced the `check-ignore` defect at its cited rule
and line, the computational review proved by a real-commit worktree
simulation that the phase-3 validator's derived `R3` anchor survives
this landing and that nothing accepted pins `.gitignore`'s bytes, and
the governance review verified the later-phase-commit rule against its
live `R2 ->* S2` precedent in the commit graph. Two advisories stand
for the `S3` implementation: the `pixi.toml` analogy is
verification-infeasibility rather than a blocked write-grant, and the
scope's "negation rules" phrase means the full perf001 four-line block
form including its companion re-ignore lines, so only the one record
file and the README are exposed, never the whole `sci004/` subtree.
It supersedes the description-follows-capability landing
`b9a9d7a8a49974bae4634f24fbc805077cdc4ef8` as the operative `D`; that
commit becomes a `superseded design` interval commit on the
header-enumerated `D0 -> D` chain, and it touched exactly
`docs/development/sci004_mmode_design.md` and
`PostTier8RemediationPlan.md`. This correction is design-only: it
implements no solver, accepts no phase, and does not close the register
row.

**Bounded correction — 2026-08-24 (the description follows the
capability).** Executing the singular-capability-pin ruling left one
live inconsistency the completed source slice cannot lawfully fix:
`MModeSimulator.description` still returns the string naming the
strategy "phase M1, scalar", byte-frozen by a live doctest in
`src/radiosim/simulator/__init__.py` that Section 13.4 grants to no
active list, while the class now truthfully reports
`supports_polarization is True` at accepted M2 — a capability truth
asserted in code and contradicted in its own user-facing description;
`src/radiosim/core/mmode/__init__.py` carries the same scalar-M1 prose.
Both `__init__` modules are granted, unscoped, by Section 13.3's `S1`
list, whose grant belonged to the closed M1 phase and conveys nothing
to `S2`. Section 9's closing sentence read either as phase-local
("while M1 is the accepted phase") or as an unconditional freeze; this
correction resolves that ambiguity in the sentence itself: until
accepted M2 the prose reports scalar-only support, and accepted M2
updates that same prose — including the registry-reported strategy
description — to the polarized truth alongside the two licensed flips,
because capability truth is phase-local and a description contradicting
the flipped property would itself be the defect. Section
13.4's `S2` list accordingly grants the two `__init__` modules, scoped
to the
description doctest and the scalar-M1 prose only. It supersedes the
singular-capability-pin landing
`d806854997cbaf9469c4cf33e36c277e287c37c3` as the operative `D`; that
commit becomes a `superseded design` interval commit on the
header-enumerated `D0 -> D` chain (joining the starred `R2 -> S2`
interval), and it touched exactly
`docs/development/sci004_mmode_design.md` and
`PostTier8RemediationPlan.md`. This correction is design-only: it
implements no solver, accepts no phase, and does not close the register
row. Its exact pre-landing file bytes
(`sha256:be58258c75cdf88e4d838e4fe7753a415d642295e328ba86a3508f540bfc297e`)
and parent-relative diff
(`sha256:d64a4de836331186c3eaa69b45a3d9bdb4b9879ff353e9705b1c47dfb9724d85`)
received separate independent reviews on 2026-08-24 —
physics/governance and computational, both `ACCEPT` after one applied
fix round: both reviews demanded the Section 13.3 `S1`-grant disclosure
in the precedent form, and the governance review proved Section 9's
closing sentence read literally as an unconditional freeze forbidding
this very correction while the computational review had read the same
sentence as phase-local — the ambiguity now resolved in the sentence
itself, which licenses the prose update alongside the two flips while
keeping the description strictly derivative of the one authoritative
pin. Both reviews verified no validator pins Section 9's prose and
re-ran the five phase validator suites green against the fixed bytes.
This correction's landing commit is the operative `D` of Section 13.7.

**Bounded correction — 2026-08-24 (the singular capability pin).**
Implementing the phase-2 capability flip proved the red slice's
capability oracle — retained case
`m2.capability.mmode-supports-polarization`, whose Tier-7
characterization node asserts `MModeSimulator.supports_polarization is
True` at accepted M2 — unreachable inside the writable law:
`tests/unit/test_simulator/test_sci004_strategy.py`'s M1-era test
asserts `is False`, two committed tests asserting opposite values of one
attribute, measured as `1 failed, 1 passed` under either value; and the
strategy file is granted only by Section 13.3's `R1` red-oracle list,
whose grant belonged to the closed M1 phase and conveys nothing to
`S2`. Section 9 already rules this topic in the singular: the
authoritative Tier-7 characterization file pins the capability value,
and only accepted M2 may deliberately flip the m-mode property and the
named Tier-7 characterization assertion to `True` — it licenses no
second pin. The strategy file's duplicate assertion was therefore never
an authoritative capability pin, and the ruling follows Section 9 rather
than creating a parallel pin to keep in lockstep: Section 13.4's `S2`
list now grants the strategy file for exactly the deletion of that
single duplicate test (its capability assertion and its own-attribute
presence check together), while the three M1 non-zero-Stokes rejection
nodes whose IDs the accepted M1 evidence binds in `capability_cases`
stay byte-untouched, the M1 artifacts themselves being immutable
history; the phase-1 validator suites remaining green after the deletion
is part of `S2`'s own gates. The scoped-grant form mirrors the Section
13.4 Tier-6 D15 directory-pin idiom; no prior grant handed an
`R1`-authored oracle file to an implementation phase, so the scope is
deliberately the narrowest possible: a deletion Section 9 already
implies, never an edit to any assertion the phase-2 record binds. Its
exact pre-landing file bytes
(`sha256:bd23d410b2a69e7376ca715e303a591b90d043427f68b40c23ff9efc000be44c`)
and parent-relative diff
(`sha256:b24dd5616e669f74c6e1be93738f1aa91d402b7780b3a3051ffbab7b927dfb6a`)
received separate independent reviews on 2026-08-24 —
physics/governance and computational, both `ACCEPT` after one applied
fix round in which each review rejected a distinct falsifiable claim:
the governance review proved the draft's granting-list citation false
(the strategy file sits in Section 13.3's `R1` block, machine-confirmed
by the phase-1 validator's own `R1` path constant) and proved Section
9's singular-pin language rules for deletion over the draft's
flip-in-lockstep principle, while the computational review proved the
draft's ordinal claim false against the retained record's own array
order and both re-verified the corrected citations byte-exactly. The
governance reconfirmation additionally proved the deletion safe against
every accepted M1 artifact and live validator: the M1 records name the
doomed function only as string data never live-collected, and the
authoritative Tier-7 assertion already carries the own-attribute check,
so no coverage is lost. This correction's landing commit is the
operative `D` of Section 13.7. It
supersedes the direct-RIME-basis landing
`d0ccab7718959dc06a5fb66bc16af9b0524c4546` as the operative `D`; that
commit becomes a `superseded design` interval commit on the
header-enumerated `D0 -> D` chain (joining the starred `R2 -> S2`
interval), and it touched exactly
`docs/development/sci004_mmode_design.md` and
`PostTier8RemediationPlan.md`. This correction is design-only: it
implements no solver, accepts no phase, and does not close the register
row.

**Bounded correction — 2026-08-23 (the direct-RIME basis for constant
receptor cells).** Implementing the celestial-tangent-transport
prescription measured it reproducing the defect it diagnoses: with the
shipped constant receptor matrix as the ground response,
`J^{ground}·R(χ)` is the identity re-expression of the same
zenith-singular field — `e^{2iχ}` winds twice about the local zenith,
measured spread exactly `2.0000` at angular separations `1e-2`, `1e-4`,
and `1e-6` — so the tier-1a shell fails at `7.45e-3` and `7.13e-3`
under the two
transport signs, refining algebraically as `N^{-1.6}`, while the
untransported constant cells in the celestial tangent basis are
machine-exact (`3.84e-15` Q-only) and a smooth projected-receptor
ground field under the same transport is likewise (`1.33e-14`). The
transport law survives as a correct identity for ground-anchored
direction-dependent responses; its defective unstated premise — that
the constant receptor matrix is the `J^{ground}` to transport — is
superseded. Section 6 now anchors the accepted M2 scope to its own
standing direct-RIME mandate: the coherency in the celestial North/East
tangent basis, constant direction-independent chain terms in that same
basis, every mount-dependent rotation owned by `P` — the exact identity
for the shipped `fixed` and unspecified mounts — and the accepted M1
scalar production path has evaluated its harmonics at these same
celestial angles since its acceptance, so the transported-constant
kernel was the sole divergence. Section 7.2's conversion sentence is
re-anchored accordingly; Section 14.3's `A2` V-bridge and
polarized-`B_lm` re-derivations now confirm the identity-`P` scope and
the shared celestial basis instead of a transport sign that does not
exist in the accepted scope (the cancellation premise was re-confirmed:
`7.45e-3` versus `7.13e-3` under the two signs); Section 13.7 gains
the general later-phase-commit form of the starred-edge rule and the
measured-discharge disposal clause; and Section 14.4's order equation
and its `S`-edge restatement now show the `R2 -> S2` edge starred. The
successions: the celestial-tangent-transport landing's recorded
reopening of the phase-2 red slice
`27d2ba45db57eed3d86fae04ece8128131d2d10e` is discharged without a
re-cut — the committed conjugate-placement transfer oracle is
empirically insensitive to the corrected ruling (measured green against
it), its constant-matrix reference is the admissible celestial-basis
model, no sign-sensitive oracle is addable in the accepted scope, and
the retained red record stands unchanged — so `27d2ba45...` remains the
accepted live `R2` rather than becoming a superseded red slice,
reversing that single designation of the transport record before any
re-cut executed. The `R2 -> S2` edge is therefore the starred one:
`S2` directly parents this correction's landing, and the interval
commits are exactly the transport landing
`e02f3975607b821b31c083a197cf7ea23865c062`, the resolved-input landing
`3b28e615ba6e752ce040f0464e3e55c36604b4a3`, and this landing, the first
two `superseded design` chain commits. The unit-level transfer helper's
ENU-labelled quadrature attribute is inherited accepted M1 surface and
keeps its name. It supersedes the resolved-input-route landing
`3b28e615ba6e752ce040f0464e3e55c36604b4a3` as the operative `D`; that
commit becomes a `superseded design` interval commit on the
header-enumerated `D0 -> D` chain, and it touched exactly
`docs/development/sci004_mmode_design.md` and
`PostTier8RemediationPlan.md`. This correction is design-only: it
implements no solver, accepts no phase, and does not close the register
row. Its exact pre-landing file bytes
(`sha256:03e2d4a20e1d362651043b978086582d8cb1ab7e962dfd529e078120df05d112`)
and parent-relative diff
(`sha256:32843f801027139b0f34c98f526e528ceb6089396152565f63ed1e4c360e3265`)
received separate independent reviews on 2026-08-23 — physics and
computational, both `ACCEPT` after one applied fix round. The physics
review independently derived the pole-compensation reconciliation — a
spin-`s` constant field in the expansion's own celestial basis is
exactly the `l=|s|` mode, machine-exact by finite-band representability,
while the local zenith is an uncompensated interior point — re-ran every
probe fresh (winding spread exactly `2.0000` at all three separations;
`3.838e-15` / `7.448e-3` / `7.131e-3` / `1.329e-14`; the asymptotic
`log2` refinement ratio `≈1.6`), verified the direct-RIME anchoring in
the committed Jones code, and established by reading the committed
oracle that its constant-matrix celestial-basis reference never was the
local-basis shortcut the superseded prescription blamed it for — the
load-bearing fact of the discharge. The computational review's blocking
finding forced Section 14.4's two-star equation and `S`-edge qualifier
after proving the draft claimed a 14.4 amendment it had not made, its
advisories yielded the exact two-sign figures and the explicit Section
13.7 measured-discharge clause, and both reviews confirmed the
discharge's empirical predicate against the live oracle. This
correction's landing commit is the operative `D` of Section 13.7.

**Bounded correction — 2026-08-23 (resolved-input route for the tangent
frame).** Implementing the Section 5.1 declaration surface proved the
declared frame unroutable: `io/config.py` parses the six-key
`tangent_polarization_frame` object and Section 10 requires the M2
snapshot to carry it exactly, but the resolved sky-source inputs live in
`src/radiosim/core/runtime_config.py`, which Section 13.4's `S2` list
did not grant — the closed `S1` list's own grant of that path belonged
to the accepted M1 phase and conveys nothing to `S2` —
and the only alternative route — attaching it through the loader — is
closed because `core/sky/loaders/synthetic.py` is not granted either and
the frame is a convention declaration, not a loader parameter (the
implementation confirmed this concretely: the field auto-flowed into
loader options and was rejected as an unexpected keyword until excluded).
Section 13.4's `S2` list now grants
`src/radiosim/core/runtime_config.py`, scoped to one optional
`tangent_polarization_frame` field on the resolved sky-source inputs.
Separately, Section 12.1's terminal-cell magnitude — "of order ten
million rows", written before any measurement existed and left deferred
by the guard-rows correction's review until an authoritative figure
landed — now cites the accepted M1 evidence artifact's measured
`16,835,749` rows. The Section 13.7 reopening of
`27d2ba45db57eed3d86fae04ece8128131d2d10e` recorded by the
celestial-tangent-transport correction stands unchanged; this correction
adds no reopening, and the pending re-cut's validator rebind will
enumerate this landing on the starred `A1 -> R2` interval alongside it.
It supersedes the celestial-tangent-transport landing
`e02f3975607b821b31c083a197cf7ea23865c062` as the operative `D`; that
commit becomes a `superseded design` interval commit on the
header-enumerated `D0 -> D` chain, and it touched exactly
`docs/development/sci004_mmode_design.md` and
`PostTier8RemediationPlan.md`. This correction is design-only: it
implements no solver, accepts no phase, and does not close the register
row. Its exact pre-landing file bytes
(`sha256:585f0d87e030cb726731ff849754add5b406b69d2df13a6803375c210619e8dd`)
and parent-relative diff
(`sha256:f06ea8c8684738086aa080f6cafbe7e5ef21cbbeba88f1d4730c6d34b94cc298`)
received separate independent reviews on 2026-08-23 —
physics/governance and computational, both `ACCEPT` after one applied
fix round. The computational review verified the `16,835,749` figure
twice over — the artifact's own counter field and an independent sum of
all `3,841` summary rows' terminal-cell counts — and confirmed the
unroutability empirically down to the loader's fixed keyword signature;
its blocking finding established the two-stage ledger rule this
correction now writes into Section 13.7, proven by the Phase-0
candidate's own pinned "independent design approval pending" row — a
round that then returned `REJECT`, which a pre-asserted verdict would
have contradicted — and the governance review formally withdrew its
earlier established-practice reading on that evidence. The fix round
also disambiguated the closed `S1` list's own lapsed grant of the same
path. This correction's landing commit is the operative `D` of
Section 13.7.

**Bounded correction — 2026-08-23 (celestial tangent transport in the
transfer kernel).** Implementing the phase-2 source slice against the
accepted `R2` proved one committed red oracle physically wrong, exactly
as the Section 7.3 tier-1a gate is built to prove: with the receptor
response modelled as a direction-independent matrix in the rotating
local `θφ` basis, the horizon-free shell is machine-exact for the spin-0
blocks (`1.5e-13` Stokes `I`, `1.1e-14` `V`) and fails at `4.65e-3` for
the spin-`±2` blocks against a `1.01e-8` limit — figures measured by the
implementation attempt that surfaced the defect, on the bounded
acceptance fixture's diagnostic form, retained here as the surfacing
observation — because the constant-in-ENU
response is singular at the zenith and carries no consistent spin
weight, so the spin-`±2` Gauss-Legendre quadrature loses its spectral
exactness. Sections 6 and 7.2 already rule the correct model — the
kernel's `θφ` components in the celestial tangent basis, the
ground-frame response transported through the same pole-transport
construction Section 5.1 fixes for the sky's `Q/U` tangent frame — and
Section 6 now pins that reading exactly: the measured transport angle,
its sign-defining relation, the kernel form
`J_{p,θφ}(n̂)=J^{ground}_p(n̂)\,R(χ(n̂))`, and the rejection of the
constant-local-basis shortcut as inadmissible rather than an alternative
convention. The committed transfer oracle
`test_the_spin_transfer_matches_the_section_6_conjugate_placement`
builds its reference from exactly that shortcut, so this correction
reopens, as a `superseded red slice` interval commit for a governed
re-cut, the phase-2 red slice
`27d2ba45db57eed3d86fae04ece8128131d2d10e`; the re-cut corrects the
transfer oracle's reference to the transported construction, adds one
oracle sensitive to the transport rotation's sign — the implementation
proved the existing set invariant under a consistent kernel-and-sky sign
flip — and regenerates the red record from a globally clean checkout of
the re-cut candidate, Section 14.4's venue vocabulary, where every
phase-2 oracle is genuinely red; uncommitted source work stays outside
that observation tree. Section
14.3's `A2` sentence now requires the V-bridge and polarized-`B_lm`
re-derivations to establish the tangent-transport sign independently,
because no surviving red oracle can. Separately, the same implementation
round proved Section 13.4's `S2` grant of
`src/radiosim/benchmarks/record.py` unexercisable:
`tools/wp7_perf001_cpu_evidence.py` pins that file's live-tree bytes,
and the WP-7 CPU-evidence validator recomputes the live file's digest
against the accepted certificate, so any edit reddens a validator no
SCI-004 phase may touch — a controlled single-byte experiment confirmed
that one reddening and disconfirmed an initially reported second (the
SCI-005 Stage-1 worktree replay reads only the frozen historical commit
and passed with the byte in place);
the grant is struck (the Section 11 benchmark surface lives in the
still-granted `src/radiosim/benchmarks/__init__.py`), recording the same
defect class Section 13.2 already documents for `pixi.toml`. It
supersedes the post-acceptance-repairs landing
`d8adeaaee1045b930fb7ca7e4bd0905655cd4725` as the operative `D`; that
commit becomes a `superseded design` interval commit on the
header-enumerated `D0 -> D` chain, and it touched exactly
`docs/development/sci004_mmode_design.md` and
`PostTier8RemediationPlan.md`. This correction is design-only: it
implements no solver, accepts no phase, and does not close the register
row. Its exact pre-landing file bytes
(`sha256:cb97177cec7d112fba492952ab6a4857995ed0e283797c9c4468925014e49d7a`)
and parent-relative diff
(`sha256:2e3e65737124d283a45c70093097511464cc48dcf998ef75195e163c851dee1e`)
received separate independent reviews on 2026-08-23 — physics and
computational, both `ACCEPT` after two fix rounds. The physics review
re-derived the spin-transformation algebra from scratch, read the defect
at its exact source lines in both the kernel and the committed oracle's
reference, and reproduced the failure live by Stokes isolation on the
acceptance fixture (`5.3e-14` I-only, `5.4e-15` V-only, `3.17e-3`
Q-only), independently confirming the surfacing observation at the same
orders. The computational review's controlled single-byte experiment
narrowed the `record.py` reddening claim to the one true validator, and
its phantom-citation finding forced the Section 6 paragraph to carry the
complete transport mathematics; the physics review's reconfirmation then
proved that paragraph's first consequence clause sign-inconsistent with
its own verified defining relation and composition — by hand algebra and
by random-Jones-matrix numerics — and the corrected matching-sign form
was re-verified independently by both reviews before landing: the very
defect class this correction eliminates, caught by its own review. This
correction's landing commit is the operative `D` of Section 13.7.

**Bounded correction — 2026-08-23 (post-acceptance repairs on the
`A1 -> R2` edge).** Authoring the phase-2 red slice proved Section
14.4's `R2^ == A1` sole direct-parent edge unsatisfiable: two commits
legitimately repaired the accepted M1 acceptance validator's refusal
probe after `A1` — `fea87708dd8bb4557a11970d4e350e66c58ca4d6` (the
probe's unconditional artifact-absence assertion made state-aware) and
`1d31baac111ec62ec45f73e355d8ad7b83b5fda8` (the preflight's no-overwrite
refusal admitted as a legitimate reason), each touching exactly
`tests/unit/test_sci004_phase1_acceptance.py` — and Section 13.7's five
interval-commit kinds had no category for them, so the edge could be
satisfied only by fabricating history or leaving a defective probe red
at every clean post-`A1` tree. Section 13.7 now records a sixth kind,
`post-acceptance repair` — a repair of an already-accepted phase's
tracked tooling or validator defect, touching only that phase's
Section 13 tool and validator test paths, never production source, a
retained artifact, or this memo, authenticated by the next phase's red
validator by full SHA and exact touched paths, with the enumerating
correction's own dual review obliged to diff each repair against its
pre-repair blob and confirm nothing was relaxed, widened, or removed
without a compensating equal-or-stronger check — and this record
enumerates both commits above under it on the now-starred `A1 -> R2`
edge. Section 14.4's order equation and its `R^` restatement now show
that edge starred, and per its rule the phase-2 `R2` directly
parents this correction's landing commit. Section 13.7 also now states
explicitly what Section 14.0 already implied: an accepted phase
acceptance commit inside the `D0 -> D` range — here `A1`
`445bc83edcf7073511c41b3485ad5d326d4e1552`, whose memo diff is exactly
its authorized append-only acceptance note — is not a chain commit and
needs no interval kind. It supersedes the guard-rows-in-the-retained-
projection landing `1712575e6c634457d9da737e9c144147e3b9bbc4` as the
operative `D`; that commit becomes a `superseded design` interval commit
on the header-enumerated `D0 -> D` chain, and it touched exactly
`docs/development/sci004_mmode_design.md` and
`PostTier8RemediationPlan.md`. It reopens no slice: the phase-2 red
slice is uncommitted working-tree state whose record regenerates at this
landing, and no committed artifact is disposed. This correction is
design-only: it implements no solver, accepts no phase, and does not
close the register row. Its exact pre-landing file bytes
(`sha256:45b3a939ab588ce636bf29613cae3e582f73e5d5cb40935ac8cf40ee5f395646`)
and parent-relative diff
(`sha256:239b1a0fa127be99d41c407d936ded9686baa98e868ead4001e59b0e5ab125a1`)
received separate independent reviews on 2026-08-23 — physics/governance
and computational, both `ACCEPT` after one fix round and one one-word
round: the governance review's two blocking findings (Section 14.4's
order equation and `R^` restatement still asserted the unstarred
`R2^==A1` fact this record contradicts, now starred and qualified; and
the sixth kind's only safeguard was mechanical SHA/path authentication,
now paired with the enumerating correction's mandatory diff-level
non-weakening confirmation) plus the computational review's advisories
(the phase-generic acceptance-note citation, the companion-ledger pin
scope, and this very two-stage record-process sentence made explicit),
with both reviews then discharging the new confirmation duty for the two
enumerated repairs — each diffed in full against its pre-repair blob and
confirmed non-weakening — and the final round fixing one directional
word. This correction's landing commit is the operative `D` of
Section 13.7.

**Bounded correction — 2026-08-23 (guard rows in the retained
projection).** The guard-interval correction ruled strict-validator
checks over `guard_interval` rows — the endpoint-sign rule, the
adjacency-and-position orphan rejection, and the census-reconstruction
partition "together with the retained root enclosures" — while leaving
Section 12.1's projection enumeration at its two-classification form, so
the checks it mandated had no preimage in the retained evidence. The
finished re-implementation surfaced the contradiction, and measured on
the bounded driver that the guards are of order a thousand rows against
the sixteen-million-row full array, so embedding them preserves the
projection's economy. Section 12.1's projection now retains every
`scan_crossing`, `excluded_upper_endpoint`, and `guard_interval` row
verbatim, names the guard rows in the strict validator's embedded-row
checks and Section 14.2's summary joins, and records the asymmetry
this creates: a present guard is authenticated geometrically and
positionally, but a deleted guard row, or a guard outer bound perturbed
within the width cap, is invisible to the strict
validator — the projection cannot distinguish a flank that classified
completely from one whose guard was omitted, and retains nothing beyond
a guard's outer bound to compare that bound against — and both are
discharged,
exactly as any omitted `ceiling_excludes_root` row is, by the mandatory
`A1` re-derivation against `horizon_scan_ledger_sha256`. It supersedes
the guard-interval landing
`52d462668df6583efeceab74a459f83d8b7ea312` as the operative `D`; that
commit becomes a `superseded design` interval commit on the
header-enumerated `D0 -> D` chain, and it touched exactly
`docs/development/sci004_mmode_design.md` and
`PostTier8RemediationPlan.md`. It reopens, as a `superseded red slice`
interval commit for a rebind-only re-cut, the re-cut
`039a057acbaf32ac2b531efc209dbaea2cfbb60a`, which touched exactly
`tests/unit/test_sci004_phase1_dependency.py`; the standing
`46b7703a727fdf3afd258034d274933e81ded289`
superseded-implementation reopening already covers the solver-side
changes, whose re-implementation has not yet landed. This correction is
design-only: it implements no solver, accepts no phase, and does not
close the register row. Its exact pre-landing file bytes
(`sha256:f33b6ca221e65a761cb15d8def92884de6267b93fb6aad79c24d0fdcf1f6e3f9`)
and parent-relative diff
(`sha256:e3ad81f548aedf230a88719ad2a5a484f390978bde9f20e5de49acf6e1706d13`)
received separate independent reviews on 2026-08-23 — computational and
physics/governance, both `ACCEPT` after one applied fix round: the
computational review's mechanical finding (a 7-hex commit citation in
this record expanded to its full 40-hex form) and two advisories — the
row-count figure aligned to the document's established sixteen-million
measurement, and the disposition broadened to name the within-cap
outer-bound perturbation as a second strict-validator-invisible class
discharged by the same `A1` reconciliation — with the physics review
re-deriving the inner-bound-pinned/outer-bound-free mechanism against
the Section 12 cut rules, the computational review tracing the
implemented orphan and width-cap checks to confirm the ruled scope, and
both reviews confirming byte-untouched earlier records. This
correction's landing commit is the operative `D` of Section 13.7.

**Bounded correction — 2026-08-23 (guard intervals and independent
membership).** The S1 re-implementation's dry-run rehearsal surfaced two
defects at the scan's edges. First, the literal "gap-free, overlap-free
half-open partition" demand rejects a range of accepted crossings: near a
root with local slope `beta*L_op`, bisection terminates only after
roughly `log2(1/beta)` extra refinements, so a crossing shallower than
`beta ~ 0.036` drives its innermost splinter to the unresolved floor
while the probe floor still accepts transverse crossings down to
`beta ~ 1.6e-3`. Section 12.1
now gives each retained crossing up to two flanking `guard_interval`
rows — bounded by the probe offset, restoring the partition together with
the root enclosures — and Section 12's exposure machinery cuts at and
error-disks over each crossing's enclosure-plus-guards union, so any
undetected structure inside a guard is certified-bounded physically
rather than assumed absent. Second, the accepted implementation evaluated
the certificate's "operational" membership and interval signs through
the frozen attitude, making two mismatch counters frozen-versus-frozen;
Section 4.2 now states explicitly that both censuses take their
operational values from the same public-API evaluations the scan
consumes, and its membership rule gains the same outside-slabs scope the
sign intervals always had — a slab-interior centre records its
disagreement into the slab accounting instead of being falsely required
to agree — with Section 14.2's counter semantics updated to match. It
supersedes the post-source-record-retention landing
`112570ff2bba42e6ab57be133318e3c0bfe32f7c` as the operative `D`; that
commit becomes a `superseded design` interval commit on the
header-enumerated `D0 -> D` chain, and it touched exactly
`docs/development/sci004_mmode_design.md` and
`PostTier8RemediationPlan.md`. It reopens, as a `superseded red slice`
interval commit for a rebind-only re-cut under the post-source retention
rule, the re-cut `13c34e79967dc28b0d11889b8ab4dcd528de915a`; the standing
`46b7703` superseded-implementation reopening already covers the
solver-side guard and membership changes, whose re-implementation has not
yet landed. This correction is design-only: it implements no solver,
accepts no phase, and does not close the register row. Its exact
pre-landing file bytes
(`sha256:14583080be874be13fdf3178bf14b288e806c4003b6528628e1f6613b15789a4`)
and parent-relative diff
(`sha256:50e62db6b5f76897c3ee0e3b05c97c8869d1ea2aa5f09936c05238c3c61aa3c1`)
received separate independent reviews on 2026-08-23 — computational and
physics/governance, both `ACCEPT` after two applied physics advisories:
the guard justification was narrowed from a universal-impossibility claim
to the derived and empirically confirmed `beta`-scoped mechanism
(termination after roughly `log2(1/beta)` refinements; the `2**-44` floor
binding only below `beta ~ 0.036` against the full enclosure width; the
turn-native probe floor accepting transverse crossings down to
`beta ~ 1.6e-3`), with the physics review measuring the public transform's
noise floor at a real crossing to rule numerical noise out as the binding
constraint and the computational review fixing the unit-consistent form of
both thresholds. This correction's landing commit is the operative `D` of
Section 13.7.

**Bounded correction — 2026-08-23 (post-source record retention).**
Executing the previous correction's pending re-cut proved one of its
obligations self-contradictory: it required the Section 13.7
disposal-and-regeneration of the red-failure record, but the operative
tree now contains the committed `S1` production, every red oracle passes
there, and the tracked red generator correctly refuses ("pytest exited
zero, so nothing was red") — a record regenerated against such a tree
would fabricate `expected-red-confirmed` observations. Section 13.7 now
rules the truthful semantics: when the reopened phase's `S` already
exists, the rebind-only re-cut retains the record's last genuinely
observed bytes and the strict validator authenticates its `design_sha`
as a header-enumerated chain commit connected to the operative `D` —
the same chain-advance rule Section 14.0 already applies to phase
bindings. The pending re-cut's obligation is restated accordingly:
rebind the dependency validator, extend the red-record validator's
`design_sha` authentication to the chain rule, and retain the record
unchanged. It supersedes the evidence-generation-reconciliation landing
`1ae7d5a94434cea35534647d4dbcef692b9e245c` as the operative `D`; that
commit becomes a `superseded design` interval commit on the
header-enumerated `D0 -> D` chain, and it touched exactly
`docs/development/sci004_mmode_design.md` and
`PostTier8RemediationPlan.md`. The standing reopenings are unchanged: the
same pending re-cut of `35db7fb16665e191feb5c6c4ced9aa3e52e5acaa` and the
same `46b7703a727fdf3afd258034d274933e81ded289` superseded
implementation. This correction is design-only: it implements no solver,
accepts no phase, and does not close the register row. Its exact
pre-landing file bytes
(`sha256:d63cf1419678a60bacc7d5cd286a536e61c6fbcfbf5ae3098c7cb280bea9d8ea`)
and parent-relative diff
(`sha256:65daba30fab2db1f064c3bd860ad361d9e01866b5d7d7e74975f3b0226bfb44d`)
received separate independent reviews on 2026-08-23 — computational and
physics/governance, both `ACCEPT`, each reproducing the generator's
refusal empirically and the latter additionally proving the
stash-production alternative a chain-of-custody violation. This
correction's landing commit is the operative `D` of Section 13.7.

**Bounded correction — 2026-08-23 (evidence generation reconciliation).**
Attempting E1 at the globally clean exact `S1` surfaced two defects. First,
both tracked phase generators are stubs: their `generate` sub-commands
unconditionally refuse after preflight, misreading Section 14.4's "runs
only at its globally clean exact `S`" as a prohibition rather than the
venue — Sections 14.2 and 14.3 now state the execution semantics
explicitly. Second, the evidence-embedding letter had silently exploded in
scale: the frame row was required to embed the complete scan terminal-cell
array (measured sixteen million rows, of order gigabytes), the per-sample
membership census (`D*N = 188,209` rows), and one transfer-sample row per
direction and output cell (`552,960` rows), against a house evidence class
of tens to hundreds of kilobytes. Section 12.1 now retains a bounded scan
projection (every crossing row verbatim plus one per-direction summary
row, with the full array's digest computed streamingly and the array
reconstructed by deterministic replay in the `A1` re-derivation), the
membership census in
per-direction visibility-mask rows whose deterministic expansion the
strict validator re-digests, and Section 7.3's transfer-sample ledger as
one direction-concatenation digest row per catalogue grid and output cell,
preserving every omission-detection guarantee at a fraction of the size;
the scan array and the transfer concatenations are the exactly two ruled
replay-deferral exceptions, both discharged by the mandatory `A1`
re-derivation.
It supersedes the ablation-clarification landing
`b8333c52688e9358e4d1747173e70196a60209ab` as the operative `D`; that
commit becomes a `superseded design` interval commit on the
header-enumerated `D0 -> D` chain, and it touched exactly
`docs/development/sci004_mmode_design.md` and
`PostTier8RemediationPlan.md`. It reopens, as a `superseded
implementation` interval commit touching only Section 13.3 `S1` paths, the
M1 source commit `46b7703a727fdf3afd258034d274933e81ded289`, whose
generator completion and economy-schema alignment land in a governed
re-implementation; and it reopens, as a `superseded red slice` interval
commit, the re-cut `35db7fb16665e191feb5c6c4ced9aa3e52e5acaa`, solely for
the operative-`D` rebinding of its dependency validator and the Section
13.7 disposal-and-regeneration of
`docs/development/sci004_mmode_phase1_red_failures.json`; the governed
re-cut directly parents this correction's landing and the
re-implementation directly parents the re-cut, preserving the
`R -> S -> E` grammar. This correction is design-only: it implements no
solver, accepts no phase, and does not close the register row. Its exact
pre-landing file bytes
(`sha256:f052a9654d2770edf2492a5f860617dc76b1326150e7b53354ffd436de3d807b`)
and parent-relative diff
(`sha256:22bde951688b9704ba7feeb61b4994e6b5fab0b91f1e77e8dc4d50cd7dd9d5c6`)
received separate independent reviews on 2026-08-23 — physics/governance
and computational, both `ACCEPT` after one applied advisory: the draft's
claim of a single replay-deferral exception was corrected to the honest
two-exceptions acknowledgment this record carries, the physics review
having established that the earlier Section 7.3 letter performed the
transfer-vector regeneration at `E1` while the concatenation form defers
it to `A1`. This correction's landing commit is the operative `D` of
Section 13.7.

**Bounded correction — 2026-08-23 (ablation clarification and deferred
advisories).** Completing S1 to a fully green suite surfaced one
implementation finding requiring a design clarification: the resolved
`BeamSystem` applies its own below-horizon cut independent of the explicit
`H` factor, so ablating the factor alone leaves a discontinuous integrand
and the tier-1a shell measures at the with-horizon level (`5.80e-2`);
sampling the beam at its exact even continuation `abs(alt)` — the unique
smooth continuation, since an aperture pattern depends on the altitude
through the even `sin(theta) = cos(alt)` — lands the shell at `1.29e-13`.
Section 7.3's tier-1a now rules the ablation as the removal of every
horizon truncation with the beam on its even continuation. This
correction also closes every deferred advisory of the earlier rounds: the
Section 12.1 `5e-12` residual bound is recorded as the deliberate
marginally-stricter rounding of its own `5.005e-12` derivation; the
`2**-44` turn unresolved floor receives its own justification; Section
13.2's commit-identity "gate anchor" is disambiguated from Section 4.1's
coordinate-frame anchor record; and the `lmax` floor paragraph now names
its enforcement through the existing `mmode_truncation_check` semantic
code rather than "schema level". It supersedes the tier-1-horizon-free-
shell landing `a67f3c8401e6d6ca4e6f531757df8cdf1598e941` as the operative
`D`; that commit becomes a `superseded design` interval commit on the
header-enumerated `D0 -> D` chain, and it touched exactly
`docs/development/sci004_mmode_design.md` and
`PostTier8RemediationPlan.md`. It also reopens, for one governed re-cut
that will directly parent this correction's landing, the re-cut M1 red
slice commit `b5af3539324bfc0784dd544d935cb479289692c4` — henceforth a
`superseded red slice` interval commit touching only Section 13.3 `R1`
paths — solely for the operative-`D` rebinding of its dependency
validator and the Section 13.7 disposal-and-regeneration of
`docs/development/sci004_mmode_phase1_red_failures.json`; no red oracle's
content is reopened. This correction is design-only: it implements no
solver, accepts no phase, and does not close the register row. Its exact
pre-landing file bytes
(`sha256:757224b46ae36240020444863043c07ea82b04e6da7e3e5fa32138c1e01f6258`)
and parent-relative diff
(`sha256:f24b8c183709ff3b734a82e5ee558e48720a85f2662ed0f3136b347f95fed9a0`)
received separate independent reviews on 2026-08-23 — physics/governance
and computational, both `ACCEPT`, the former re-deriving the even
continuation for general direction-cosine patterns and the latter
verifying the beam evaluator's independent horizon gate and the working
`abs(alt)` ablation at their exact source lines. This correction's
landing commit is the operative `D` of Section 13.7.

**Bounded correction — 2026-08-23 (tier-1 horizon-free shell).**
Qualifying the retuned acceptance fixture against the two-tier gate proved
tier 1 as first drafted unattainable by the same mechanism as the gate it
replaced: `int(K*Y_lm)` has a discontinuous integrand under the strict
horizon, no finite Gauss-Legendre rule is exact for it, and the measured
with-horizon cross-quadrature residual (`2.07e-3` at the fixture scale,
converging as `nside^-2`) would need of order `1e9` transfer nodes to
reach `1e-8`, while the identical pipeline with the horizon factor removed
passes at `1.15e-10`. This correction splits tier 1: tier 1a gates at
`1e-8` on the horizon-free cross-quadrature shell (`H === 1`, everything
else identical — spectrally exact, so every shared-pipeline defect fails
sharply, while horizon application stays owned by the Section 12 direct
machinery); tier 1b records the with-horizon shell in provenance and
bounds it per acceptance fixture by a reviewed `quadrature_budget_jy`.
It also rewrites the convergent-regime fixture rule to the measured
governing conditions (payload directions well clear of the horizon — the
M1 fixture is circumpolar with zero frozen roots and an exactly-zero
enclosure-error cube; `lmax` pinned by the accepted evidence because the
quarter-to-full factor collapses once `L1` resolves the smooth kernel;
fixtures qualified by measurement with real margin, predicates never
widened), clarifies that the deficit enters the result's provenance
record while the Section 10 snapshot key set is unchanged, and bumps the
Section 11 predicate to `sci004_two_tier_direct.v3` with the split
tier-1 fields. It supersedes the two-tier-acceptance-gate landing
`10ae8628556d7ea95c0b70af086a82cf8bb569ec` as the operative `D`; that
commit becomes a `superseded design` interval commit on the
header-enumerated `D0 -> D` chain, and it touched exactly
`docs/development/sci004_mmode_design.md` and
`PostTier8RemediationPlan.md`. The reopened-slice record of the previous
correction stands unchanged: the same governed re-cut of
`fe3f7865ad4684de8bfa7a305661e4e4bf2fd233`'s integration oracle, not yet
performed, will encode this correction's final two-tier form and the
measured circumpolar fixture. This correction is design-only: it
implements no solver, accepts no phase, and does not close the register
row. Its exact pre-landing file bytes
(`sha256:05e535126a3ae345f50b1772e9e7af7f35ae4e5612e105ba9cd28373b03f12e9`)
and parent-relative diff
(`sha256:df522a6d95d6ef946ec6116f6594d17aa6a401d9c09759bc3ae9485f380dc7c3`)
received separate independent reviews on 2026-08-23 — computational and
physics/governance, both `ACCEPT`, the computational review confirming
the quadrature mechanism from first principles (a step integrand's
Gauss-Legendre error converging at empirical order exactly `2.0` while
the smooth case reaches machine precision), and the physics review
self-correcting its round-4 attainability premise and contributing the
Section 6 shared-horizon-implementation mandate this draft carries. This
correction's landing commit is the operative `D` of Section 13.7.

**Bounded correction — 2026-08-23 (two-tier acceptance gate).** Completing
S1 against the corrected design proved the Section 7.3 every-run `1e-8`
direct-equality gate mathematically unattainable: the transfer kernel
carries the strict horizon step, the band-limited projection of a
discontinuous kernel converges pointwise only algebraically (measured
`L^-1`..`L^-1.5`; `1.52 Jy` deviation against a `2.35e-8 Jy` limit on the
M1 point fixture, a `6.5e7`-fold excess), and reaching `1e-8` would need
`lmax` of order `1e5` against the design's own `4096` ceiling; a
pixel-measure sky inherits the same floor through the quadrature-sampled
above-band remainder, and the fixed `4x`-per-level acceptance rule
likewise assumed spectral convergence. The program owner ruled for the
two-tier gate with point skies retained under honest budgets. This
correction rewrites Section 7.3's gate (tier 1: quadrature-shell fidelity
at `1e-8` plus the standing `5e-12` analytic oracles; tier 2: the
truncation deficit computed against the complete frozen direct oracle on
every run, gated on strict monotone quarter/half/full convergence with a
`>= 2x` quarter-to-full factor, disclosed in provenance, and bounded per
acceptance fixture by a reviewed `truncation_budget_jy`), re-rules
Section 1's reading of Q5's "direct-RIME agreement" accordingly, updates
the Section 11 `direct_comparison` object to
`sci004_two_tier_direct.v2`, the Section 14.2 truncation and
direct-convergence rows, the Section 12.2 family-6 oracle, and the
Section 14.3 reviewer re-derivations, and adds the convergent-regime
fixture rule. It supersedes the S1-feasibility-reconciliation landing
`ef3aa7aac270068ac8ca3d275886ceb25e732d80` as the operative `D`; that
commit becomes a `superseded design` interval commit on the
header-enumerated `D0 -> D` chain, and it touched exactly
`docs/development/sci004_mmode_design.md` and
`PostTier8RemediationPlan.md`. It also reopens, for one governed re-cut
that will directly parent this correction's landing, the re-cut M1 red
slice commit `fe3f7865ad4684de8bfa7a305661e4e4bf2fd233` — henceforth a
`superseded red slice` interval commit touching only Section 13.3 `R1`
paths — whose `tests/integration/test_sci004_mmode.py` pins the
superseded single-predicate gate and whose fixture must be retuned into
the convergent regime; the re-cut regenerates
`docs/development/sci004_mmode_phase1_red_failures.json` under Section
13.7's disposal rule and rebinds the dependency validator's operative-`D`
constants. This correction is design-only: it implements no solver,
accepts no phase, and does not close the register row. Its exact
pre-landing file bytes
(`sha256:437ae3a2e8cd0b733eb21a97435714dd70535038820c136fe39d4ead1dacd069`)
and parent-relative diff
(`sha256:f8ada4348e3ba3598f01510cd5ed10946721ad404ac0a847df37c61e2cc4a760`)
received separate independent reviews on 2026-08-23 — physics/governance
and computational, both `ACCEPT` after one applied blocker round (the
`lmax >= 4` well-posedness floor for the tier-2 convergence levels, whose
minimality both reviewers proved independently) — with one recorded
clarification: the floor is enforced through the existing
`mmode_truncation_check` semantic code, so "reject at schema level" in the
floor paragraph reads as "reject at validation time". This correction's
landing commit is the operative `D` of Section 13.7.

**Bounded correction — 2026-08-22 (S1 feasibility reconciliation).**
Implementing the M1 production slice against the corrected design proved
three of its requirements defective and surfaced three more unlisted-pin
gaps, resolved here as one bounded correction under Section 13.7. It
supersedes the R1-authoring-reconciliation landing
`a3afec87f201d0691430070023ac980c863cb224` as the operative `D`; that
commit becomes a `superseded design` interval commit on the
header-enumerated `D0 -> D` chain, and it touched exactly
`docs/development/sci004_mmode_design.md` and
`PostTier8RemediationPlan.md`. It also reopens, for one governed re-cut
that will directly parent this correction's landing, the M1 red slice
commit `724ef948bb7a251d3269247341e109f8bd2c3893` — henceforth a
`superseded red slice` interval commit touching only Section 13.3 `R1`
paths — whose `tests/integration/test_sci004_mmode.py` pins the superseded
quadrature-policy literal and whose
`tests/unit/test_core/test_sci004_scalar_harmonics.py` constant-map oracle
pins the superseded continuous-field sky interpretation; the re-cut
regenerates `docs/development/sci004_mmode_phase1_red_failures.json` under
Section 13.7's disposal rule and rebinds the dependency validator's
operative-`D` constants. The resolved defects: (1) Section 12.1's
`OperationalHorizonEnclosure` demanded certified interval extensions of the
installed ERFA expression graph, whose coefficient tables pyerfa does not
expose and whose measured evaluation cost lower-bounds at hundreds of hours
per solve; it is replaced by the certified-ceiling scan, complete via the
design-frozen `L_op` derivative ceiling, consuming only the public Astropy
API, with its own new fixed constants. (2) Section 12's ambiguous-piece
error disks required the same interval extension of the complete integrand;
they now use the certified magnitude-ceiling rule. (3) Equal-area HEALPix
pixel-centre transfer quadrature cannot meet the Section 12.2 analytic
transfer oracle under the strict horizon — its visible-area error is
exactly `1/(3*nside)` — and is replaced by the iso-Gauss grid of identical
`12*nside**2` cardinality, renaming the quadrature-policy literal to
`iso-gauss-ring-production-plus-qcheck.v1`. (4) Section 7.1's
map-coefficient construction is ruled to the pixel measure, making the
harmonic sky and the private direct oracle the same object. (5) Section
13.3's `S1` list gains three scoped inventory-pin grants whose pinned
inventories the required registry and config widenings necessarily change.
(6) Section 4.1 records the anchor-location convention and Section 13.2
records the gate-anchor rule for corrections accepted after a gate has
run. This correction is design-only: it implements no solver, accepts no
phase, and does not close the register row. Its exact pre-landing file
bytes
(`sha256:09ea150e8285ea8067b4bc57056390cda2805dcde056c1dbce02364eb1f442bf`)
and parent-relative diff
(`sha256:478dc114c8c1be3a2635389f54d216e1570c7b272c20bab5228e9c03f0e8a82f`)
received separate independent reviews on 2026-08-22 — physics/governance
and computational, both `ACCEPT`, each independently re-deriving the
ceiling lemma, the `1/(3*nside)` defect, and the pixel-measure residual —
with two deferred non-blocking advisories recorded for a future
correction: the `5e-12` operational residual bound is about `0.1%`
stricter than its own cited derivation (the safe direction), and the
`2**-44` turn unresolved floor deserves its own justification sentence
rather than a unit carry-over from the replaced construction. This
correction's landing commit is the operative `D` of Section 13.7.

**Bounded correction — 2026-08-22 (R1-authoring reconciliation).** Authoring
the M1 red slice against the accepted design surfaced one writable-list gap
and confirmed three recorded advisories, resolved here as one bounded
correction under Section 13.7. It supersedes the Phase-0 correction landing
`71d3deb05b0d981653472dff9b17330b3dc9f9cf` as the operative `D`; that commit
becomes a `superseded design` interval commit on the header-enumerated
`D0 -> D` chain, and it touched exactly
`docs/development/sci004_mmode_design.md` and
`PostTier8RemediationPlan.md`. (1) The accepted Tier-6 characterization pin
`tests/characterization/test_tier6_current_behavior.py::test_the_benchmark_harness_task_and_performance_test_now_exist`
asserts the exact file listings of `tests/integration/` and
`tests/performance/`; creating `tests/integration/test_sci004_mmode.py` at
`R1` and `tests/performance/test_sci004_mmode.py` at `R2` therefore requires
widening those pinned listings by exactly the file each phase's own list
names — the pin's documented maintenance convention — yet no SCI-004 phase
list authorized that path before `S3`. Sections 13.3 and 13.4 now authorize
it, scoped to the named listing widenings alone. (2) Section 13.1's closing
sentences still carried the pre-correction single-fixed-`D` phrasing; they
now defer to Section 13.7's `D0`/operative-`D` definition (the deferred
Phase-0 reconfirmation advisory). (3) Section 8's `mmode_static_gain`
message named `jones.gain.time_model.kind`, but the strict config surface
keys the gain block as `jones.G` (`io/jones_config.py`); the frozen message
now names `jones.G.time_model.kind`. (4) Section 10 conflated the
strategy-level backend-native `(T,B,F,2,2)` receptor cube with the public
result container, whose strict shape is `(T,B,F,4)`
(`core/result.py`); the sentence now states both truthfully. This
correction is design-only and reopens no accepted slice: no `R1` commit
exists yet, so the red slice lands against the corrected operative `D`. Its
exact pre-landing file bytes
(`sha256:fe2f8334dda068451c08d86ec43a83bf8654d47d00c4f91f27d377e735d8288a`)
and parent-relative diff
(`sha256:c7e3eda08556a55b22dc33a867116803f24725884f0fb56ab650c4f7201698e9`)
received separate independent reviews on 2026-08-22 — computational and
physics/governance, both `ACCEPT`, the former after one applied mechanical
blocker: the supersession citation this paragraph now carries. This
correction's landing commit is the operative `D` of Section 13.7.

**Source reviewed:** `42a1f27e5f6078ce72960f7d200e8b1e94d399c2`.
Concurrent WP-7 working-tree changes are outside this design and are not
evidence for it.

**Status:** design only. This document implements no solver, accepts no
production phase, changes no public capability claim, and does not close the
register row. `SCI-004` remains **ROADMAP** until the complete production and
validation succession in Sections 13–15 is independently accepted.

## 1. Ruling and adopted science driver

Q5 is resolved by the following bounded driver:

> A HERA-like, fixed-zenith drift-scan survey requiring repeated
> full-sidereal visibility evaluation, with direct-RIME agreement on small
> polarized skies and controlled spherical-harmonic truncation error.

The production name is `execution.simulator: mmode`. It is a second complete
forward model, not a Jones term, a point-source optimization, a map maker, or a
new name for the existing direct sum. It consumes one resolved `SkyModel`,
forms sky and instrument harmonic representations, evaluates independent
forward matrix-vector products for each `m`, and synthesizes the existing
time-domain visibility result.

The observing regime is deliberately narrow:

- the array, beams, receptors, and accepted instrumental terms are fixed in
  the terrestrial frame;
- the phase centre and boresight are the existing fixed zenith;
- the sky is sidereal and fixed over one Earth rotation;
- sample centres are a complete, unflagged, uniformly spaced Earth Rotation
  Angle (ERA) cycle with no duplicated endpoint; and
- the output remains a simulated visibility cube. Noise modelling, inverse
  map making, pseudo-inverses, KL filtering, power-spectrum estimation,
  calibration, tracking, drift-and-shift, missing samples, and partial-day
  windows are not part of `SCI-004`.

The driver's "direct-RIME agreement" is operationally the Section 7.3
two-tier gate: harmonic-pipeline fidelity is gated at `1e-8` through
same-truncation cross-quadrature and analytic oracles, while the
comparison against the complete frozen direct oracle is computed on every
run as a truncation deficit that must converge and is always disclosed —
never asserted as numerical equality, which a band-limited representation
of the strict-horizon kernel cannot attain at any admissible `lmax`.

The driver's "HERA-like" label names the compact drift-scan core regime; it
is not itself an admission rule. The sole admission authority for any
specific site, array, frequency, and sky combination is the Section 4.2
`FrameApplicabilityCertificate` with its fixed budgets. Because the
certificate's wrapped-phase gate amplifies the frozen-vs-operational
direction discrepancy by `2*pi*b*nu/c`, those fixed budgets imply a bounded
admissible scale: at the canonical site the Phase-0 review measured a
full-cycle frozen-vs-operational direction discrepancy of order `3e-6 rad`,
so the fixed `5e-3 rad` phase budget admits roughly
`b_max*nu_max <~ 8e10 m Hz` (about `400 m` at `200 MHz`). A configuration
beyond that envelope is expected to fail certification and is outside the
resolved Q5 scope. The estimate is descriptive and the certificate is
normative; widening the frame budgets or the operational-frame model for a
larger array is a design successor, never a tolerance change.

Shaw et al.'s `v_m=B_m a_m` relation is the scientific starting point, but
RadioSim retains its own Jy normalization, east-X receptor binding, IAU Stokes
convention, result axes, and output formats. A formula copied from the paper is
not allowed to override those existing contracts.

## 2. Live source truth and the required strategy boundary

At the reviewed source the registry is not yet a full-sky strategy boundary:

- `src/radiosim/simulator/__init__.py` contains only `rime`;
- `VisibilitySimulator.calculate_visibilities(...)` accepts `SourceArrays`, so
  its abstract interface describes only the point component;
- `Simulator.run()` always calls `core.hybrid.solve_sky(...)` itself;
- `solve_sky(...)` dispatches point sources through the registered object but
  calls `calculate_visibility_healpix(...)` directly; and
- `SolverResultProvenance`, `ExecutionConfig`,
  `ResolvedExecutionConfig`, and their readers accept only `rime`.

Adding `MModeSimulator.calculate_visibilities(SourceArrays, ...)` to that
registry would therefore be false architecture: HEALPix and hybrid runs would
still bypass it. Phase M1 introduces one immutable `SkySolveRequest` and one
`SkySolveOutcome` at the **whole `SkyModel`** boundary. The request carries the
resolved sky model and point arrays, instrument view, beam system, location,
time/frequency coordinates, receptors, Jones inventory, backend, and worker
policy.

`RIMESimulator.solve(request)` becomes a thin wrapper around the maintained
`core.hybrid.solve_sky` point/HEALPix/hybrid path. Its arithmetic, component
order, source reduction, result bytes, and fingerprints must remain unchanged.
`MModeSimulator.solve(request)` consumes the same whole request and never calls
the direct point or HEALPix production kernels. The high-level API calls only
the selected registered strategy. The standing invariant remains exact:

```text
accepted values of execution.simulator == simulator registry keys
```

After M1 that set is `{"rime", "mmode"}`.
`visibility.calculation_type` was removed and is not restored; no parallel
registry or configuration parser is introduced.

## 3. Canonical full-sidereal coordinate

### 3.1 ERA turns, not rounded radian endpoints, define the grid

The current `ObservationTimeGrid` is uniform in UTC. UTC is an output and
provenance coordinate; it is not the group coordinate that diagonalizes a
transit telescope. The m-mode coordinate is the continuously unwrapped number
of Earth-rotation turns relative to the anchor,

$$
u(t)=\frac{\operatorname{ERA}(t)-\operatorname{ERA}(t_0)}{2\pi},
\qquad u(t_0)=0,
$$

with increasing UT1. Radians are a derived numerical view of this exact turn
coordinate; they do not define endpoint ownership, wrapping, or cycle closure.

Let `N=obs_time.sidereal_samples`. Decode the already validated finite
binary64 `integration_fraction=f` by its exact IEEE-754 integer ratio
`p_f/q_f`, reduced with `q_f>0` and `0<p_f<=q_f`. The source decimal spelling
is not an arithmetic input. Negative zero, NaN, infinity, zero, and values
greater than one reject before this step.

Every rational is normalized to `p/q` with a positive denominator,
`gcd(abs(p),q)=1`, and zero represented only as `0/1`. Its serialized form is
the shortest base-10 ASCII numerator, `/`, and shortest base-10 ASCII
denominator: no leading `+`, whitespace, or leading zero other than the
integer zero itself is allowed. The immutable exact turn grid is constructed
once:

$$
\begin{aligned}
u_k &= \operatorname{reduce}\!\left(\frac{2k}{2N}\right),\\
u^-_k &= \operatorname{reduce}\!\left(
 \frac{2kq_f-p_f}{2Nq_f}\right),\\
u^+_k &= \operatorname{reduce}\!\left(
 \frac{2kq_f+p_f}{2Nq_f}\right),\qquad k=0,\ldots,N-1,\\
\Delta u &= \operatorname{reduce}\!\left(\frac{p_f}{Nq_f}\right),\\
h^-_N &= \operatorname{reduce}\!\left(-\frac{1}{2N}\right),\\
h^+_N &= \operatorname{reduce}\!\left(\frac{2N-1}{2N}\right).
\end{aligned}
$$

Equivalently, the edge formulas are exactly `(2*k-f)/(2*N)` and
`(2*k+f)/(2*N)` with `f` treated as its exact ratio. They are never evaluated
by binary64 subtraction or addition. The canonical unwrapped cycle is

$$
H_N=[h^-_N,h^+_N),
$$

and has exact rational width `h_N^+-h_N^- == 1/1`. Exact rational comparison
proves

```text
h_N^- <= u_k^- < u_k < u_k^+ <= h_N^+
u_k^+ - u_k^- == Delta_u
```

For `f==1` it additionally proves `u_k^+==u_(k+1)^-` for every adjacent
pair, `u_0^-==h_N^-`, and `u_(N-1)^+==h_N^+`. There is no sample at `u=1`;
that value exists only as the virtual closure point.

Set `tau=float.fromhex("0x1.921fb54442d18p+2")`. If `RN` denotes the unique
binary64 round-to-nearest-ties-to-even result of an exact rational, the derived
radian view is

```text
alpha_rad[k] = RN(exact(tau) * u_k)
lower_rad[k] = RN(exact(tau) * u_k^-)
upper_rad[k] = RN(exact(tau) * u_k^+)
delta_alpha_rad = RN(exact(tau) * Delta_u)
horizon_lo_rad = RN(exact(tau) * h_N^-)
horizon_hi_rad = RN(exact(tau) * h_N^+)
```

Each multiplication has one final `RN` and no intermediate binary64
arithmetic. Full-width adjacent derived edges remain bit-identical because the
same normalized rational enters the same conversion. The constructor requires
strictly increasing derived centres and
`lower_rad[k] < alpha_rad[k] < upper_rad[k]`; failure raises the typed
`mmode_exposure_resolution` rejection. Exact-turn equality, not rounded-radian
subtraction, is the closure authority: there is deliberately no assertion
that `horizon_hi_rad-horizon_lo_rad==tau`, and a consumer may not lift a turn
by adding binary64 `tau` or recover topology by dividing radians by `tau`.

The phase-to-time map consumes the exact turn value directly:

$$
\operatorname{JD}_{\rm UT1}(u)
=\operatorname{JD}_{{\rm UT1},0}
 +\frac{u}{1.00273781191135448}.
$$

The ERA-rate literal is treated as its exact decimal rational inside the
two-part-JD implementation before final rounding. Thus `tau` cancels
analytically and is not an input to time ownership or cadence. Centres,
exposure edges, horizon cuts, interval nodes, and the virtual closure point are
mapped from exact turns. The resulting values are converted to two-part UTC
for existing results and writers; each integration time is derived from its
exact `u_k^-`, `u_k^+` pair.
The written integration-width array is the finite binary64 SI-second
`Time(upper_utc[k])-Time(lower_utc[k])` value for every sample. Its exact
identity is
`A("radiosim.mmode-integration-time.v1","integration_time",["sample"],"s",integration_times)`
with dtype `float64-be`; the retained field is
`integration_time_seconds_sha256`.

`CanonicalEraGrid` contains the exact turn object, the derived binary64 view,
and their identities. That same immutable object is passed by identity to the
time mapper, operational isolator, phase ledger, harmonic window, and both
direct oracles. Reconstructing a turn coordinate from `k`, radians, a width,
or an adjacent edge inside any consumer is a validation failure.

The embedded `canonical_era_turn_grid` has exactly:

```text
schema_version, sidereal_samples, integration_fraction_f64be,
integration_fraction_ratio, exposure_width_turn, horizon_lo_turn,
horizon_hi_turn, center_turns, lower_edge_turns, upper_edge_turns
```

Its schema literal is `radiosim.mmode-era-turn-grid.v1`.
`integration_fraction_f64be` is the lowercase 16-hex-character encoding of
`struct.pack(">d",f)` and must decode to `integration_fraction_ratio`. The
three turn arrays contain exactly `N` canonical `p/q` strings in sample order.
Their exact identities are

```text
era_center_turn_sha256 =
  D("radiosim.mmode-era-center-turns.v1",J(center_turns))
era_lower_edge_turn_sha256 =
  D("radiosim.mmode-era-lower-edge-turns.v1",J(lower_edge_turns))
era_upper_edge_turn_sha256 =
  D("radiosim.mmode-era-upper-edge-turns.v1",J(upper_edge_turns))
canonical_era_turn_grid_sha256 =
  D("radiosim.mmode-era-turn-grid.v1",J(canonical_era_turn_grid))
```

The three derived radian arrays use Section 14 `A` with domain
`radiosim.mmode-era-radian-array.v1`, axes `["sample"]`, units `rad`, dtype
`float64-be`, and respective roles `center`, `lower_edge`, and `upper_edge` to
produce `era_center_rad_sha256`, `era_lower_edge_rad_sha256`, and
`era_upper_edge_rad_sha256`.
The embedded `canonical_era_grid` is the Section 14 canonical object having
exactly:

```text
schema_version, canonical_era_turn_grid_sha256,
era_center_turn_sha256, era_lower_edge_turn_sha256,
era_upper_edge_turn_sha256, tau_f64be, delta_alpha_rad_f64be,
horizon_lo_rad_f64be, horizon_hi_rad_f64be,
era_center_rad_sha256, era_lower_edge_rad_sha256,
era_upper_edge_rad_sha256
```

Its schema literal is `radiosim.mmode-era-grid.v2`, and
`canonical_era_grid_sha256` is
`D("radiosim.mmode-era-grid.v2",J(canonical_era_grid))`. All `*_f64be` fields use
the same lowercase big-endian binary64 encoding. The exact turn object, all
eight component digests, and every manifest scalar enter result provenance;
the frame certificate carries both canonical grid digests and joins every
turn-bearing ledger to that exact object.

No network lookup or implicit Astropy table selection is permitted. Resolution
loads exactly the locked resource returned by
`importlib.resources.files("astropy_iers_data") / "data/finals2000A.all"`,
hashes its bytes, opens it with `IERS_A.open`, and installs that object with
`earth_orientation_table.set(...)` around every time and coordinate operation.
`auto_download` is false and degraded accuracy is an error. Sample centres,
exposure boundaries, horizon-split oracle nodes, and the anchor attitude must
all lie inside that installed table's accepted range. The resource path,
package version, SHA-256, final/predictive row status, Astropy/ERFA versions,
per-sample `UT1-UTC`, and UTC/UT1 two-part coordinates enter provenance. A
missing, different, uncovered, extrapolated, or silently substituted table is
a typed rejection.

The fixed construction tolerances are:

- maximum unwrapped ERA centre residual: `2e-11 rad`;
- maximum ERA step residual from retained `tau/N`: `2e-11 rad`;
- UT1 -> UTC -> UT1 round-trip residual: `1e-6 s`.

They are constants, not YAML fields. Evidence records observed maxima and the
fixed limits. A tolerance is never widened because a platform misses it.

### 3.2 Typed time input

The existing untagged UTC interval remains the only `rime` input:

```yaml
obs_time:
  start_time: "2025-01-01T00:00:00"
  duration_seconds: 60.0
  time_step_seconds: 10.0
```

M-mode receives a distinct strict variant:

```yaml
obs_time:
  mode: full_sidereal
  start_time: "2025-01-01T00:00:00"
  sidereal_samples: 257
  integration_fraction: 1.0
```

`sidereal_samples` is a strict positive integer. `integration_fraction` is a
strict finite float in `(0,1]` and scales the top-hat width without removing a
sample. The logical exposure width and its binary64 `delta_alpha`, centres,
and boundaries are exactly the Section 3.1 `CanonicalEraGrid` construction;
the retained boundaries are mapped to UTC for the result integration-width
array. No duration, UTC cadence, explicit time list, flags, or window weights
are accepted in this variant. The old model and every old serialized `rime`
snapshot stay byte-identical when the new variant is absent.

## 4. Operational frame and the SCI-007 error budget

### 4.1 Why the current transform cannot simply be reused

The present direct path transforms ICRS directions to Astropy `AltAz` at every
UTC instant and reconstructs an operational apparent direction/parallactic
angle. That is not a one-parameter rigid rotation about a fixed celestial
`z` axis. Precession-nutation, aberration, polar motion, and tangent-basis
transport prevent the exact relation `B_lm(alpha)=B_lm(0)*exp(i*m*alpha)`.
Calling that path m-mode compatible would recreate the frame ambiguity closed
as a bounded limitation by `SCI-007`.

The accepted m-mode frame literal is
`radiosim.frozen-cirs-rigid-era.v1`. It is site-specific and executable, not a
shorthand for an unspecified Astropy transform. Every operation below runs
inside the installed bundled-IERS context from Section 3.1.

1. `t0` is resolved to two-part UT1 and TT. Polar motion has one normative unit
   conversion. Evaluate

   ```text
   xp_quantity, yp_quantity = installed_iers.pm_xy(t0)
   xp0_arcsec = float(xp_quantity.to_value(u.arcsec))
   yp0_arcsec = float(yp_quantity.to_value(u.arcsec))
   das2r_rad_per_arcsec = float(erfa.DAS2R)
   xp0_rad = xp0_arcsec * das2r_rad_per_arcsec
   yp0_rad = yp0_arcsec * das2r_rad_per_arcsec
   sp0_rad = float(erfa.sp00(tt1, tt2))
   RPOM0 = erfa.pom00(xp0_rad, yp0_rad, sp0_rad)
   ```

   Both `pm_xy` results must be finite scalar quantities convertible to
   arcseconds. `erfa.pom00` receives radians only: passing its unitless
   arcsecond values, applying `DAS2R` twice, or using a different unit path is
   forbidden. The frame certificate serializes finite JSON numbers
   `xp0_arcsec`, `yp0_arcsec`, `das2r_rad_per_arcsec`, `xp0_rad`, `yp0_rad`,
   and `sp0_rad`, together with the exact literals
   `pm_source_unit="arcsec"` and `pom00_argument_unit="rad"`. After parsing
   those numbers as binary64, the validator requires `xp0_rad` and `yp0_rad`
   to be bit-identical to their respective serialized arcsecond values
   multiplied once by serialized `das2r_rad_per_arcsec`. It separately
   requires `das2r_rad_per_arcsec` to have the installed `float(erfa.DAS2R)`
   bits and recomputes `sp0_rad` and `RPOM0`. The CIRS frame is explicitly the
   **geocentric**
   `CIRS(obstime=t0)` frame; passing `location=site` here is forbidden because
   that would add a different topocentric/diurnal-aberration model. The
   frozen anchor record retains the location that was supplied to its
   construction — therefore always absent — and exposes the exact frame
   object every public transform is taken against.
2. Each ICRS direction is transformed once through the public
   `SkyCoord.transform_to(CIRS(obstime=t0))` API. Its North/East tangent
   Jacobian is transported by the public-Astropy oracle below. The resulting
   CIRS direction and tangent columns are frozen for the run.
3. `era0 = erfa.era00(ut11, ut12)`. For a CIRS column-coordinate vector `c`,
   the terrestrial column coordinates at relative phase `alpha` are exactly

   $$
   t(\alpha)=T(\alpha)c,
   \qquad T(\alpha)=RPOM_0 R_3(ERA_0+\alpha),
   $$

   with the SOFA passive matrix

   $$
   R_3(\theta)=
   \begin{bmatrix}
   \cos\theta&\sin\theta&0\\
   -\sin\theta&\cos\theta&0\\
   0&0&1
   \end{bmatrix}.
   $$

   Thus the normative relation is `[ITRS] = RPOM0 R3(ERA) [CIRS]`, matching
   SOFA/ERFA `c2tcio`. `R3(+alpha)` is a passive coordinate rotation; the
   equivalent active sky rotation in fixed ITRS is `A_z(-alpha)=R3(+alpha)`.
   No transpose, longitude sign, or fitted phase offset is allowed. Since
   `R3(a)R3(b)=R3(a+b)`, `T(alpha)=T(0)R3(alpha)` and the Section 6
   `exp(+i*m*alpha)` transfer law follows with the stated harmonic convention.
4. The fixed local ITRS basis is constructed from the resolved geodetic site
   longitude `lambda` and latitude `phi` in this exact `(East, North, Up)`
   order:

   $$
   e_E=(-\sin\lambda,\cos\lambda,0),
   $$
   $$
   e_N=(-\sin\phi\cos\lambda,-\sin\phi\sin\lambda,\cos\phi),
   $$
   $$
   e_U=(\cos\phi\cos\lambda,\cos\phi\sin\lambda,\sin\phi).
   $$

   The operational altitude is `asin(dot(t(alpha), e_U))`; azimuth and beam
   vectors use the same explicit basis. The geodetic site, `era0`, all six
   unit-named polar-motion/s-prime numbers and both unit literals above,
   `RPOM0`, `T(0)`, and their byte-order-normalized SHA-256 enter the frame
   identity.
5. Zenith phase centre, field rotation, beam, fringe, ground-stationary Jones
   terms, and the strict horizon factor are all derived from this one frame
   object. The harmonic solver and the private frozen-frame direct oracle must
   receive that identical object; reconstructing the attitude independently is
   a validation failure.

The tangent transport oracle is also fixed. Let `n,N,E` be the unit ICRS
direction and its canonical North/East tangent columns, with catalogue
longitude retained as the gauge at an exact coordinate pole. For
`q in (N,E)` and `h=2**-12 rad`, construct the public-Astropy ICRS coordinates
with Cartesian unit vectors `n*cos(h) +/- q*sin(h)`, transform both with
`SkyCoord.transform_to(CIRS(obstime=t0))`, and form

$$
d_q(h)=\frac{c_q(+h)-c_q(-h)}{2\sin h},\qquad
d_q=\frac{4d_q(h/2)-d_q(h)}{3}.
$$

Project with `I-c*c.T`, normalize North, Gram-Schmidt East against North, and
require `dot(cross(N_CIRS,E_CIRS),c_CIRS) < 0`, the handedness of the declared
North/East chart. Repeating at `h/2` must change either normalized tangent by
at most `2e-10 rad`. This Richardson construction, using only public Astropy
coordinate objects and transforms, is the acceptance oracle; a private frame
graph helper or a position-angle shortcut is not an equivalent authority.
The same `T(alpha)` transports both tangent columns into ITRS.

The one-time ICRS-to-CIRS tangent transport is load-bearing. It prevents the
`SCI-007` error from being baked into the harmonic sky. The ordinary public
`rime` default remains unchanged; m-mode acceptance does not retroactively
claim it uses the frozen frame.

### 4.2 Explicit approximation budget

Angular samples cannot license this approximation: a sparse phase/pixel probe
does not establish a full-cycle maximum, a North/East angle is gauge-dependent
at a coordinate pole, and a small angular error does not bound a hard horizon
step or a steep beam. Production therefore requires an authenticated
`FrameApplicabilityCertificate` bound to the exact site, IERS bytes, ERA grid,
array/baselines, frequencies, receptors, Jones/beam identities, sky identity,
the ordered transfer-grid catalogue and every grid's direction rows,
precision, and frame-matrix identity. Reuse is allowed only when every bound
identity is byte-identical.

The certificate evaluates the complete run, not eight phases or a pixel
subset:

- its canonical direction ledger has one row for every point-source entry of
  every point component with any non-zero resolved Stokes value and one row
  for every native HEALPix payload pixel with any non-zero resolved Stokes
  value at any run frequency. It also covers the ordered transfer-grid
  catalogue
  `[('production',quadrature_nside)] + [('diagnostic',q) for q in Q_diag]`,
  where `Q_diag` is the numerically sorted unique set of nsides consumed by a
  non-production operand of any Section 7.3 local-shell diagnostic and is
  exactly `[qcheck]` under this design. Every iso-Gauss node of every
  catalogue entry has a distinct, grid-qualified row. Production and
  diagnostic roles remain distinct even if nsides or vectors coincide. Native
  pixels are
  enumerated after coordinate/polarization conversion and canonical RING
  ordering, including every HEALPix component of a hybrid sky. Rows from
  different kinds, components, roles, or grids are never de-duplicated merely
  because their vectors coincide. Its phase set is all `N` centres, every
  exposure boundary, and both endpoints of every certified frozen/operational
  horizon-root enclosure in the exact one-turn cycle;
- for every direction, baseline, frequency, and phase node it evaluates the
  existing geometric-phase function with frozen and pressure-zero operational
  Astropy directions. The maximum wrapped phase difference must be at most
  `5e-3 rad`;
- for every direction it proves complete enumeration of all frozen analytic
  and pressure-zero operational Astropy horizon roots over the cell-centred
  full-cycle exact-turn horizon domain `H_N=[h_N^-,h_N^+)` from the retained
  Section 3.1 grid by the procedure in Section 12.1. Its closed exact-turn
  enclosure contains every retained top-hat boundary and has width exactly
  one; rounded radian endpoints are derived views and never own topology. No
  consumer recomputes either endpoint. The root counts must match. Every
  transverse root
  is labelled rising or setting by its certified derivative sign; only roots
  with the same label may pair. Within a label, the unique order-preserving
  cyclic pairing unwraps each operational root by an integer multiple of
  one exact turn nearest its frozen partner; an ambiguous pairing rejects the
  certificate. Every pair's outward worst-case enclosure displacement must be
  at most `2e-5 rad`. Each pair defines the closed circular **mismatch slab**
  given by the hull of both certified root enclosures under the selected exact
  integer-turn lift. A slab crossing the `H_N` seam is represented as two closed pieces on
  the unwrapped domain; no equality of the operational values at its endpoints
  is assumed. Every exposure is split at every root-enclosure and
  guard-interval endpoint, and a
  positive-width enclosure or guard cell uses Section 12's outward error
  disk rather
  than a smooth rule. Both models evaluate membership and interval signs
  independently — the operational values from the same public-API
  evaluations the scan consumes, never from the frozen attitude. Strict
  `alt > 0`
  membership must agree at every actual sample centre outside the mismatch
  slabs; a centre inside a slab records its models' possibly differing
  memberships into the slab accounting rather than being falsely required
  to agree, exactly as the sign intervals are treated. The above/below
  sign on every certified open root
  interval must agree outside the union of the slabs. Sign differences between
  displaced roots inside a slab are recorded, not falsely required to be zero;
  each direct model applies its own strict altitude numerator there, and the
  complete direct-cube predicate below bounds their physical effect.
  The certificate records the number and total union measure of the slabs,
  whose fixed limit is `2e-5 rad` times the number of paired roots, as well as
  zero unresolved isolation intervals and zero outside-slab sign mismatches;
  and
- it computes independently qualified 64- and 128-node exposure-averaged cubes
  under both frames plus frozen/operational root-enclosure error cubes for
  every `(time, baseline, frequency, correlation)` cell. Both model-specific
  all-cell 64-to-128 maxima pass `1e-11*S_Q`. With
  `U=abs(F128-O128)+EF+EO` and
  `S=max(1 Jy,max(abs(O128)+EO))`, both
  `max(U) <= 5e-5*S + 1e-10 Jy` and
  `norm(U)/max(norm(O128),sqrt(U.size)*1 Jy) <= 5e-5` are mandatory.

The direct cubes use the horizon-split oracle in Section 12, so a shifted hard
step or a native-payload crossing cannot disappear into an angular diagnostic.
Every point and native HEALPix ledger row is a direct-cube contributor whose
every exposure is processed by the common root partition. Production-transfer
rows certify the production harmonic response; diagnostic-transfer rows
certify only the named local-shell operands. Both roles receive the complete
phase/root/slab/sign/membership proof but neither enters the point/native
direct-contributor stream. The coverage ledger records
and requires equality of expected and evaluated per-kind and total direction,
phase-comparison, horizon-trajectory, point/native direct-exposure, and
output-cube cell counts. Direction and tangent-angle
statistics over the same complete set are retained only as diagnostics; no
synthetic celestial poles or sampled angular maximum is an acceptance gate.
The exact run dimensions and `b_max*nu_max`, rather than a generic "HERA-like"
label, are certificate fields. Any incomplete count, unresolved root interval,
outside-slab sign mismatch, phase failure, horizon failure, or cube failure
rejects the run with no YAML waiver. M1 evidence must include a passing
retained bounded-driver certificate, and at least one retained certificate
case must have `b_max*nu_max` within a factor of two of the Section 1
admissible envelope, so the phase budget is exercised rather than assumed by
the smallest fixture; expansion to a different site, array,
frequency range, sky, beam, or IERS identity requires a new certificate and
independent frame review.

For M1--M3 the certificate is computed in memory for every solve, before
harmonic work, with the NumPy direct oracle; it is not a user-supplied path, a
YAML bypass, or a cache lookup. Its canonical bytes and SHA-256 enter result
provenance, and its full cost enters any performance record. This conservative
rule licenses correctness rather than a speed claim. Reusing a certificate
without recomputation, or certifying a broader site/array/frequency domain,
requires a separately reviewed design successor with an authenticated cache
contract.

## 5. Polarization and harmonic conventions

### 5.1 Canonical sky metadata

Today point and HEALPix containers carry numerical `Q/U` arrays but no complete
tangent-basis record. That is insufficient for spin harmonics. M2 introduces a
strict frozen `TangentPolarizationFrame` with exactly:

- `schema_version = "radiosim.sky-tangent-polarization.v1"`;
- `coordinate_frame` (`"icrs"` or `"galactic"`);
- `axes = "north_east"`;
- `position_angle = "north_through_east"`;
- `linear_complex = "q_plus_i_u"`; and
- `stokes_v = "iau_incoming_r_minus_l"`.

Every point or HEALPix payload with non-zero `Q` or `U` must carry it. Point
`Q/U` are defined in the local tangent plane of each catalogue direction;
HEALPix `Q/U` are defined in the local tangent plane of each pixel. Known input
loaders attach and, where necessary, convert their documented source
convention. In particular a HEALPix/CMB `U` convention is converted explicitly
to RadioSim IAU North-through-East before canonical storage; tests pin the sign
with a rotated pure-Q map. A programmatic polarized input without a declared
source convention is rejected. An I/V-only payload may omit the tangent block.

Coordinate conversion transports the tangent basis as well as positions.
Relabelling a Galactic polarized map `icrs`, rotating only pixel indices, or
copying `Q/U` without the spin-2 rotation is forbidden.

### 5.2 RadioSim-to-Shaw basis bridge

RadioSim's sky electric vector is ordered `(North, East)` and its brightness
matrix is

$$
P^I I+P^Q Q+P^U U+P^V_{\rm RS}V
=\frac12\begin{bmatrix}I+Q&U+iV\\U-iV&I-Q\end{bmatrix}.
$$

Shaw et al. use the spherical `(theta, phi)` basis, with `theta` pointing
South and `phi` East. The exact bridge is

$$
D=\operatorname{diag}(-1,1),\qquad
e_{\theta\phi}=D e_{NE},\qquad J_{\theta\phi}=J_{NE}D.
$$

Consequently the fields used in Shaw-form harmonic equations are

$$
I_H=I_{RS},\quad Q_H=Q_{RS},\quad U_H=-U_{RS},\quad V_H=V_{RS}.
$$

In one unchanged ordered basis, RadioSim's `P^V_RS` has the opposite matrix
sign from Shaw's `P^V`; after the required `D` basis bridge the physical IAU
`V` field has the same sign. No additional fitted or configurable V flip is
allowed. The accepted SCI-006 east-X permutation remains inside `J_NE`; it is
not replaced by `D`. Pure `Q`, pure `U`, pure `V`, east-X linear, circular,
and mixed-basis analytic tests must distinguish all of these operations.

### 5.3 Spherical harmonics

The literal `radiosim.shaw-polarized-harmonics.v1` fixes:

- right-handed spherical coordinates `(theta,phi)`, with colatitude
  `theta in [0,pi]` and `phi` increasing eastward in `[0,2*pi)`;
- orthonormal complex Condon-Shortley harmonics,
  `integral(_sY_lm * conj(_sY_l'm')) = delta_ll' delta_mm'`;
- scalar expansions for I and V;
- Shaw's spin-labelled expansions

  $$Q_H+iU_H=\sum a^{(+2)}_{lm}\,{}_{+2}Y_{lm},\qquad
  Q_H-iU_H=\sum a^{(-2)}_{lm}\,{}_{-2}Y_{lm};$$

- scalar reality `a[l,-m]=(-1)^m*conj(a[l,m])` and the explicit paired
  spin-reality relation
  `a^(-2)[l,m]=(-1)^m*conj(a^(+2)[l,-m])`; and
- one packed signed-m-major representation defined below. Invalid `(l,m,s)`
  cells do not exist and are not represented by padding whose value could enter
  a digest.

The science field order is exactly `("I", "+2", "-2", "V")`, with spin
order `(0, +2, -2, 0)`. For each signed `m` in the inclusive ascending range
`-mmax..mmax`, and then for each field in that fixed order, one immutable
`PackedHarmonicBlock` row contains exactly these fields in this order:

```text
m, field_index, field_name, spin, l_start, l_stop,
value_start, value_stop
```

`l_start=max(abs(m),abs(spin))`, `l_stop=lmax+1`,
`value_stop-value_start=l_stop-l_start`, and each row starts at the preceding
row's `value_stop`, beginning at zero. Harmonic values inside a row are in
ascending `l` order. The block table and packed value buffer are inseparable;
the table is serialized under Section 14's exact block-table domain and the
values are normalized C-order big-endian complex128 bytes for identity (the
runtime buffer's native endian is not hashed directly). Sky coefficients have shape
`(frequency, packed_value)`. Baseline-transfer coefficients have shape
`(baseline, frequency, correlation, packed_value)` and use the same table, so
matching a sky and transfer cell never relies on library `alm` order.

The optional per-antenna electric-beam audit cache uses a separate scalar
packed table with rows `(m,l_start,l_stop,value_start,value_stop)`, again
signed-m-major and ascending `l`, and values shaped
`(antenna, frequency, feed, sky_vector, packed_value)`. Feed order is the
resolved receptor-label order and `sky_vector` is exactly `("theta","phi")`.
Those two components are defined in exactly the Section 5.2 bridge basis
`e_thetaphi = D e_NE`, and any receptor-side factor entering the cache uses
the accepted chain-tangent receptor conventions; the cache introduces no
third tangent convention.
It is an audit intermediate only; the normative baseline transfer is built
from the full RIME kernel. The two packed-table SHA-256 values enter scientific
provenance. A rectangular `(...,l,m)` array, padded invalid cell, field reorder,
or implicit healpy packed index is a different convention and is rejected.

The transform adapter has a slow explicit quadrature reference used by analytic
tests, the Section 7.1 pixel-measure projection for native HEALPix payloads,
and the Section 7.3 iso-Gauss production quadrature. Complex transfer fields
are transformed by separate real and imaginary operations and recombined in a
documented order. Scalar and spin transforms are tested against analytic
single modes and against each other under the exact basis bridge. Library
default signs, iteration counts, or packed-`alm` order are never inferred.

## 6. Forward equation and exact axes

For antenna pair `p,q`, frequency `f`, output correlation `c`, and Stokes basis
field `X`, define the reference-phase response from the same Jones and fringe
factors as the direct RIME:

$$
K^X_{pqfc}(\hat n)=
\left[J_{p,\theta\phi}(\hat n)P^X_{\theta\phi}
J^H_{q,\theta\phi}(\hat n)\right]_c
K_{pq}(\hat n)M_{pq}(\hat n)Q_{pq}(\hat n)H(\hat n),
$$

`K` is the existing geometric phase with its accepted sign; `M` is the
accepted baseline closure factor and `Q` may contribute its accepted bandwidth
smearing factor. ERA top-hat integration is owned by the synthesis below, so
the existing `Q.time_smearing` approximation is rejected for `mmode` rather
than applied a second time. The horizon factor is exactly

$$
H(\hat n)=\mathbf 1[\operatorname{alt}(\hat n)>0]
=\begin{cases}1,&\hat n\mathbin{\cdot}e_U>0,\\0,&\hat n\mathbin{\cdot}e_U\le0,
\end{cases}
$$

using Section 4's frozen terrestrial direction and local Up column. Equality
is excluded, matching both maintained direct solvers; no epsilon, beam cutoff,
or half-weight at the horizon is allowed. It is part of the transfer function,
not an after-the-fact time mask, and every displayed `B_lm` integral below is
over the full sphere with this factor present. The private direct oracles
and the harmonic-transfer kernel construction invoke the identical tested
implementation of this horizon predicate — one shared code object, never a
re-derivation of the same formula — so a horizon-application defect cannot
differ between the compared models and thereby escape the Section 7.3
tier-1a horizon-free shell, the tier-2 budgets, and the Section 12 direct
machinery simultaneously. The canonical Jones order, half-power
normalization, east-X/circular receptor semantics, beam normalization, and
four row-major correlation labels are unchanged.
The `θφ` components above are taken in the celestial tangent basis at
`n̂` — the same basis the Section 5.3 spin expansions use — and the
chain enters exactly as the accepted direct RIME applies it: the
coherency is built in the celestial North/East tangent basis of each
direction, the chain's direction-independent terms — the receptor
factors among them — right-multiply it as constant matrices in that
same basis, and the `P` term owns every mount-dependent tangent
rotation, equal to the exact identity for the shipped `fixed` and
unspecified mounts (the accepted parallactic invariant every instrument
source except a pyuvdata dataset produces). For that accepted M2 scope
no transport angle enters the kernel at all: constant cells are
constant coefficients on spin-weighted fields, so they preserve the
integrand's spin weight and the Section 7.3 spin-`±2` Gauss-Legendre
quadrature stays spectrally exact — measured at `3.84e-15` (Q-only)
against the `1.01e-8` limit — and the accepted M1 scalar production
path has evaluated its harmonics at these same celestial angles since
its acceptance. A genuinely ground-anchored, direction-dependent
response — an alt-az mount rotating through a non-identity `P`, or a
future non-scalar ground-frame beam — enters through the measured
tangent transport: both tangent bases at `n̂` from the one
pole-transport construction Section 5.1 fixes for the sky's `Q/U`
tangent frame, each applied to its own frame's pole axis; `χ(n̂)` the
angle of the transported celestial North measured in the local tangent
basis by `atan2` — measured, never assumed — with its sign fixed by the
numerically confirmed defining relation
`(Q+iU)_local = e^{+2iχ}(Q+iU)_celestial`; and the composition
`J_{p,θφ}(n̂) = J^{ground}_p(n̂)\,R(χ(n̂))`, admissible only when
`J^{ground}_p(n̂)` is itself a smooth field on the sphere (measured
machine-exact, `1.33e-14` Q-only, for the projected-receptor form). The
inadmissible object is the local-basis-constant tensor field in any
chart: `e^{2iχ}` winds twice about the local zenith — measured spread
exactly `2.0000` at angular separations `1e-2`, `1e-4`, and `1e-6` — so
transporting a constant `J^{ground}` by `R(χ)` is the identity
re-expression of the same zenith-singular field and reproduces the
defect (`7.45e-3` Q-only transported, refining algebraically as
`N^{-1.6}`, never spectrally), exactly as applying that constant in the
rotating local basis does. An implementation or an oracle reference
built either way is defective rather than an alternative convention.
Section 12.2's "omitted tangent transport" non-vacuity control names
Section 4.1's one-time ICRS-to-CIRS position-and-tangent transport, not
this mount rotation; it stays live and unchanged.
Every SCI-004 cube therefore has `C=4` in the exact resolved matrix order;
`n_correlations!=4`, an omitted cross-hand, or an evidence formula treating
`C` as a free cardinality rejects before work.

At `alpha=0`, define

$$
B^I_{pqfc,lm}=\int K^I_{pqfc}Y_{lm}\,d\Omega,
\quad
B^V_{pqfc,lm}=\int K^V_{pqfc}Y_{lm}\,d\Omega,
$$

$$
B^{(+2)}_{pqfc,lm}=\int(K^Q_{pqfc}-iK^U_{pqfc})
{}_{+2}Y_{lm}\,d\Omega,
$$

$$
B^{(-2)}_{pqfc,lm}=\int(K^Q_{pqfc}+iK^U_{pqfc})
{}_{-2}Y_{lm}\,d\Omega.
$$

The conjugate placement matches the preceding sky expansions and Shaw's
definition that the transfer function itself is expanded in conjugate
harmonics. It is pinned by an explicit numerical integral; changing a library
API call until one test passes is not an alternative convention.

Rigid rotation gives

$$B^X_{pqfc,lm}(\alpha)=B^X_{pqfc,lm}(0)e^{im\alpha}.$$

The forward per-m product is

$$
v_{pqfc,m}=\sum_l\left[
B^I_{pqfc,lm}a^I_{f,lm}
+\frac12B^{(+2)}_{pqfc,lm}a^{(+2)}_{f,lm}
+\frac12B^{(-2)}_{pqfc,lm}a^{(-2)}_{f,lm}
+B^V_{pqfc,lm}a^V_{f,lm}\right].
$$

Here "per-m solve" means this forward matrix-vector contraction. `SCI-004`
does not expose Shaw's map-making pseudo-inverse or solve for a sky.

Let `Delta_u` be the exact exposure width retained by the Section 3.1 grid and
define

$$
w_m=\operatorname{sinc}\!\left(\pi m\,\Delta u\right)
=\begin{cases}
1,&m=0,\\
\sin(\pi m\,\Delta u)/(\pi m\,\Delta u),&m\ne0.
\end{cases}
$$

The normalized discrete transform and exposure-averaged synthesis are

$$
\bar v_m=\frac1N\sum_{k=0}^{N-1}\bar V_k e^{-i2\pi m u_k}
=w_m v_m,\qquad
\bar V_k=\sum_{m=-m_{max}}^{m_{max}}w_m v_m e^{+i2\pi m u_k}.
$$

Every `u_k` and `Delta_u` is the retained exact rational from the same
`CanonicalEraGrid`. Correctly rounded unit-circle/sinpi kernels consume those
turns directly; the DFT, synthesis, and sinc may not regenerate topology from
`k`, `N`, radians, or `tau`. Derived radian arrays remain evaluation/provenance
views only.

`N >= 2*mmax+1` is mandatory, so the retained modes never touch an ambiguous
Nyquist bin. The centre-sample window is the full rectangular periodic window
with weight one; the fixed ERA exposure top hat is a diagonal `w_m` factor, not
a spectral taper. Missing centres, endpoint duplication, non-rectangular
windowing, or a partial cycle would convolve `m` modes and are rejected rather
than silently windowed.

Canonical internal arrays are therefore:

```text
sky scalar/spin coefficients: (frequency, packed_value)
per-antenna electric beam:    (antenna, frequency, feed, sky_vector,
                               scalar_packed_value)
baseline transfer:            (baseline, frequency, correlation, packed_value)
m-mode visibility:            (baseline, frequency, correlation, signed-m)
time-domain result:           (time, baseline, frequency, receptor-row,
                               receptor-column)
```

The packed block tables supply the only `m`, field, spin, and `l` axes for the
first three arrays. The signed-m visibility axis is the explicit ascending
integer vector `[-mmax,...,+mmax]`; correlation order remains the existing
row-major receptor-product order.

Baseline order, frequency order, output correlations, and point/HEALPix hybrid
component order remain the existing canonical orders. No transpose is allowed
to be repaired only at serialization.

## 7. Sky, beam, truncation, and quadrature

### 7.1 Sky coefficients

Point components are not silently rasterized. A delta-function point sky uses
analytic scalar and spin harmonics evaluated at the exact transported source
direction. Spectral indices, tabulated spectra, rotation measure, and Stokes
values are resolved per frequency before the transform. The first production
scope rejects Gaussian morphology because its baseline-dependent envelope is
not one common sky field; adding analytic extended-source harmonics requires a
design successor.

HEALPix maps are converted at each frequency to IAU-canonical I/Q/U/V flux per
solid angle, including the existing brightness-conversion rules, and enter the
harmonic sky as the **pixel measure**: the map's coefficients are exactly

```text
a_lm = sum_pix( s_pix * Omega_pix * conj(Y_lm(n_pix)) )
```

over canonical-RING pixel centres with the equal pixel solid angle
`Omega_pix = 4*pi/npix` — the same measure the private direct oracle sums —
so harmonic-versus-direct agreement tests truncation and nothing else, and a
constant map's `l>0` coefficients carry the pixel-quadrature residue rather
than being zero. A continuous band-limited reinterpretation of the map, a
ring-weighted quadrature, or any iterated transform is a different sky
object and is rejected; an implementation may evaluate the sum through a
library transform only when that call is exactly equivalent to the displayed
sum, with zero iterations and no quadrature weights beyond `Omega_pix`. RING
and NEST inputs must yield identical
coefficients after canonical ordering. Sparse maps remain sparse while loading
but are transformed as the declared full-sky field with absent pixels exactly
zero. A hybrid model adds point and map coefficients in the fixed
`("point", "healpix")` order before any `B_lm a_lm` product; it does not run two
independent m-mode solvers and add rounded outputs.

The private direct oracle does not resample a native HEALPix payload onto the
transfer quadrature. It sums the original native pixel centres in canonical
RING order with their native pixel solid angle and resolved per-frequency
Stokes payload. The transfer grid is a separate harmonic-response domain and
never substitutes for, merges with, or removes a native direct contributor.

### 7.2 Per-antenna and baseline response

For every antenna/frequency, the resolved `BeamSystem` is sampled in the
reference ground frame and converted through the accepted receptor
contracts; a ground-anchored direction-dependent response additionally
carries Section 6's measured tangent transport, while the shipped
fixed-mount scope applies the constant receptor cells directly in the
celestial tangent basis, per Section 6. The electric response harmonics are an auditable
intermediate: their convention and digest are always recorded, while a
materialized cache is retained only when the declared memory budget permits.
The normative baseline `B_lm` is nevertheless formed from the full reference
RIME kernel, including baseline fringe and baseline terms; Phase M1 does not
attempt an unreviewed harmonic convolution of two antenna coefficient sets.

The initial source phase may use current scalar beams. Before whole-row closure
the accepted SCI-005 Stage-2 non-scalar squint fixture must pass the same
construction, proving that diagonal-native/full-sky-side `E`, east-X, and
non-commuting Jones order survive. SCI-005 Stage 3 is not silently claimed by
this row.

### 7.3 Fixed truncation policy

The strict m-mode block declares `lmax`, `mmax`, and `quadrature_nside`.
Validation requires

```text
4 <= lmax <= 4088
0 <= mmax <= lmax
quadrature_nside is a power of two, at least 2
lmax <= 2 * quadrature_nside
sidereal_samples >= 2 * mmax + 1
```

The `4` floor is a well-posedness requirement of the two-tier gate below:
it is the smallest `lmax` for which the tier-2 convergence levels satisfy
`L1 < L2 < lmax` unconditionally, so every legally configurable run can
evaluate the mandatory gate as specified; `lmax` of `2` or `3` would leave
`L2` at or beyond `lmax` and the strict monotone predicate structurally
unsatisfiable, and both reject at validation time through the existing
`mmode_truncation_check` semantic code.

A transfer grid of resolution `nside` is the **iso-Gauss grid**: `3*nside`
Gauss-Legendre colatitude rings — the nodes and weights of the
`3*nside`-point rule in `z = cos(theta)` on `[-1, 1]` — each carrying
`4*nside` uniformly spaced azimuths starting at `phi = 0`, for exactly
`12*nside**2` nodes. A node's quadrature weight is its Gauss-Legendre
weight times `2*pi/(4*nside)`. `3*nside` is even for every accepted
`nside`, so no node lies on the horizon-critical equator, the visible
hemisphere carries exactly half the total weight under any strict horizon
through the equator, and the uniform azimuths annihilate every `m != 0`
mode of an azimuthally constant integrand exactly; equal-area HEALPix
pixel-centre quadrature, whose equatorial ring sits on the horizon and
whose visible-area error is `1/(3*nside)` under the strict `alt > 0`
factor, is rejected as a transfer quadrature. Node enumeration is
ring-major from the north pole, then ascending azimuth index; `nside`
retains its role as the resolution parameter and every `12*nside**2` count
formula is unchanged. The conservative `lmax <= 2*nside` bound keeps the
band comfortably inside the rule's exactness range; it is not by itself an
accuracy proof. Every production
run derives, without configuration knobs,

```text
lcheck = min(lmax + max(8, lmax // 8), 4096)
mcheck = min(lcheck, mmax + max(8, max(1, mmax // 8)))
qcheck = next_power_of_two(max(2 * quadrature_nside, ceil(lcheck / 2)))
```

The transfer-grid inventory additionally derives

```text
Q_diag = sorted(unique(nside of every non-production V(..., nside)
                       operand in the four diagnostics))
```

and under the frozen formulas requires `Q_diag == [qcheck]`. Grid IDs are
exactly `production:<quadrature_nside>` and `diagnostic:<qcheck>`. Creating or
consuming any additional transform grid without adding its distinct role/nside
to `Q_diag`, the transfer-grid catalogue, direction ledger, certificate
identity, and coverage joins is forbidden.

`lcheck > lmax` and `mcheck > mmax` are mandatory, and the resolved full-
sidereal grid must also satisfy `sidereal_samples >= 2*mcheck+1`. Thus the
validation grid can represent the omitted physical `m` tail; the retained-mode
Nyquist rule alone is not misrepresented as a truncation bound.

Four complete synthesized visibility cubes are retained as local attribution
diagnostics around the production cube
`V0=V(lmax,mmax,quadrature_nside)`:

1. quadrature: `V(lmax,mmax,qcheck) - V0`;
2. `l` tail: `V(lcheck,mmax,qcheck) - V(lmax,mmax,qcheck)`;
3. `m` tail: `V(lcheck,mcheck,qcheck) - V(lcheck,mmax,qcheck)`; and
4. combined local shell: `V(lcheck,mcheck,qcheck) - V0`.

The exact diagnostic-grid joins, in that order, have fields
`diagnostic_id`, `lhs_grid_id`, `lhs_lmax`, `lhs_mmax`, `rhs_grid_id`,
`rhs_lmax`, `rhs_mmax`, `lhs_cube_sha256`, `rhs_cube_sha256`, and
`delta_cube_sha256`, and resolve scientifically as:

```text
quadrature:     diagnostic:qcheck,lmax,mmax ; production:qprod,lmax,mmax
l_tail:         diagnostic:qcheck,lcheck,mmax ; diagnostic:qcheck,lmax,mmax
m_tail:         diagnostic:qcheck,lcheck,mcheck ; diagnostic:qcheck,lcheck,mmax
combined_local: diagnostic:qcheck,lcheck,mcheck ; production:qprod,lmax,mmax
```

`qcheck` and `qprod` in an ID are replaced by their canonical base-10 integer
nsides. The first tuple is the left-hand operand. All operands are
zero-extended through signed `m in [-mcheck,+mcheck]`, so every field/block row
has all four diagnostic columns; a null or not-applicable column is forbidden.

Each diagnostic cube covers every time, baseline, frequency, and correlation
output cell. Each contributing field in the exact
`("I","+2","-2","V")` order and every packed signed-m block is also measured
before field summation. There is no short/long-baseline, edge-frequency, pixel,
field, or other deterministic subset. The shell-coverage preimage has exactly
`schema_version`, `input_identity_sha256`, `frame_certificate_sha256`,
`direction_ledger_sha256`, `transfer_grid_catalog_sha256`,
`diagnostic_grid_joins`, `transfer_sample_rows`, `shell_comparison_rows`, and
`field_block_rows`, with
schema literal `radiosim.mmode-shell-coverage.v1`.

`transfer_sample_rows` has exactly `grid_id`, `baseline_index`,
`frequency_index`, `correlation_index`, `field_index`, `field_name`,
`resolved_lmax`, `resolved_mmax`, `block_table_sha256`,
`direction_count`, `packed_sample_value_count`, and
`concatenation_sha256`: one row per catalogue grid, baseline, frequency,
correlation, and fixed-order field, ordered by production then diagnostic
grid, baseline, frequency, correlation, and field.
`concatenation_sha256` is the `A` identity — domain
`radiosim.mmode-transfer-sample-contribution.v1`, axes
`["direction","packed_value"]`, role `transfer_sample_contribution`, units
`visibility_response_sr`, dtype `complex128-be` — over the
direction-ledger-ordered concatenation of every catalogued direction's
packed contribution vector for that cell, appended only as each direction
is evaluated and accumulated. `direction_count` is that grid's catalogue
count and the array's leading dimension. Thus a catalogued node cannot be
omitted, reordered, or substituted while preserving the digest — the same
per-direction guarantee the earlier one-row-per-direction form carried, at
a fraction of the retained size — and the reconstruction of each
concatenation is a per-grid transfer replay the `A1` re-derivation
performs while the strict validator checks the row set, counts, ordering,
and digest form.
For a production row, `resolved_lmax=lmax` and `resolved_mmax=mmax`. For a
diagnostic row they equal `lcheck` and `mcheck`, and the row authenticates the
complete largest diagnostic vector. Every lower-`l` or lower-signed-`m`
qcheck operand in the four joins is the exact block-table projection of that
same retained vector; its operand-cube digest is rebuilt from those projections.
Re-transforming an unrecorded lower operand or claiming coverage from a vector
that cannot project exactly to all three qcheck operand dimensions rejects.

`shell_comparison_rows` has exactly `diagnostic_id`, `time_index`,
`baseline_index`, `frequency_index`, `correlation_index`, and
`absolute_delta_jy_f64be`; rows use the four diagnostic IDs above and then
canonical time/baseline/frequency/correlation order. A row is appended only
after that finite cell comparison executes and its magnitude is retained.
`field_block_rows` has exactly `baseline_index`, `frequency_index`,
`correlation_index`, `field`, `signed_m`, `diagnostic_ids`,
`quadrature_time_value_count`, `quadrature_time_values_sha256`,
`quadrature_max_abs_jy_f64be`, `l_tail_time_value_count`,
`l_tail_time_values_sha256`, `l_tail_max_abs_jy_f64be`,
`m_tail_time_value_count`, `m_tail_time_values_sha256`,
`m_tail_max_abs_jy_f64be`, `combined_local_time_value_count`,
`combined_local_time_values_sha256`, and
`combined_local_max_abs_jy_f64be`. Rows use canonical
baseline/frequency/correlation order, increasing signed `m`, then field order
`("I","+2","-2","V")`, matching the packed signed-m-major convention.
`diagnostic_ids` is always the fixed four-ID array. Every time-value count is
exactly `N`; its digest uses the Section 14 array primitive over the complete
complex128 time vector with domain
`radiosim.mmode-field-block-diagnostic.v1`, role
`<id>_field_block_delta`, axes `["time"]`, units `Jy`, and dtype
`complex128-be`; and the maximum is recomputed from that vector. A row is appended only
after all four zero-extended vectors are finite.

The ledger requires exactly
`(1+len(Q_diag))*B*F*C*4` transfer-sample rows — one per catalogue grid
and output cell, each carrying its complete direction-concatenation
digest —
`4*N*B*F*C` shell-comparison rows and
`B*F*C*4*(2*mcheck+1)` field/block rows. All operand/delta cube digests in the
four join rows use Section 14's visibility-cube primitive.
`shell_coverage_sha256` is
`D("radiosim.mmode-shell-coverage.v1",J(shell_coverage))`. Missing, duplicate, reordered,
non-finite, grid-dangling, or uncovered work rejects the record, while its
magnitude does not license or reject scientific correctness. The
strict validator independently regenerates every field/block diagnostic
time vector, rebuilds their child `A` digests and maxima, reconstructs the
complete embedded row arrays, and requires exact ordered equality before
rebuilding `shell_coverage_sha256`; the transfer-sample concatenation
digests are validated in form, count, and ordering by the strict validator
and reconstructed by per-grid transfer replay in the `A1` re-derivation. The
four maxima, the largest field/block delta, and the reference value
`1e-6*max(1 Jy,max(abs(V(lcheck,mcheck,qcheck))))` are recorded for
attribution. They are not a correctness bound: a finite local shell cannot
exclude power in an arbitrarily more distant omitted `l` or signed-`m` block.

The authoritative acceptance structure is the **two-tier gate**, executed on
**every production run** before any result or output path is created. It
replaces the earlier single `1e-8` direct-equality predicate, which is
mathematically unattainable for this forward model: the transfer kernel
carries the strict horizon step, a band-limited projection of a
discontinuous kernel converges pointwise only algebraically
(`L^-1`..`L^-1.5` measured), and reaching `1e-8` would need `lmax` of order
`1e5`, far beyond the fixed `4096` transform ceiling. For a delta sky the
forward product is exactly `S*K_L(n_s)` — the band-limited kernel sample —
so the deficit against the exact direct sum is a property of the method,
not an implementation defect; a pixel-measure sky inherits the same floor
through the quadrature-sampled above-band remainder. The two tiers
separate what must be numerically exact from what must be honestly
attributed.

**Tier 1 — harmonic-pipeline fidelity.** The strict horizon step poisons
quadrature exactness just as it poisons truncation: `int(K*Y_lm)` has a
discontinuous integrand, no finite Gauss-Legendre rule is exact for it,
and the measured cross-quadrature residual converges only as `nside^-2`
(reaching `1e-8` would need of order `1e9` transfer nodes). Tier 1
therefore has a sharp half and a recorded half.

**Tier 1a — horizon-free shell, gating at `1e-8`.** The run evaluates the
complete harmonic pipeline once with **every horizon truncation** removed
and everything else identical — the same grids, beam object, fringe,
packing, contraction, and synthesis — on both the production and `qcheck`
quadratures. Removing the explicit `H` factor alone is insufficient: the
resolved `BeamSystem` applies its own below-horizon cut, and with only the
factor ablated the shell measures at the with-horizon level. The ablation
therefore samples the beam at the exact even continuation `abs(alt)` — an
aperture pattern depends on the zenith angle through
`sin(theta) = cos(alt)`, an even function of the altitude, so this is its
unique smooth continuation, not a model change — while the fringe, entire
in the direction cosines, stays on the true direction. With `W0` and `W_q` those two horizon-free cubes,
`K = 4*N*B*F`, and `S_num = max(1 Jy, max(abs(W_q)))`, both

```text
max(abs(W0-W_q)) <= 1e-8*S_num + 1e-10 Jy
norm(W0-W_q) / max(norm(W_q), sqrt(K)*1 Jy) <= 1e-8
```

must hold: the integrand is smooth, Gauss-Legendre is spectrally exact
through the band, and the measured horizon-free residual sits at the
`1e-10` level, so any sign, normalization, weight, packing, or dropped-mode
defect in the shared pipeline fails sharply. The horizon-free cubes are
tier-1 internals, never a result. The Section 12.2 analytic single-mode,
DFT, and exposure-sinc oracles at `5e-12` remain the other half of this
tier, and horizon *application* correctness is owned cell by cell by the
Section 12 horizon-split direct machinery, not by this shell.

**Tier 1b — with-horizon shell, recorded and fixture-budgeted.** The
with-horizon quadrature shell `V_q = V(lmax,mmax,qcheck)` is still
computed; `max(abs(V0-V_q))` and its normalized L2 are recorded in the
result's provenance record, and every acceptance fixture declares a
reviewed `quadrature_budget_jy` in the phase evidence that the recorded
maximum must not exceed — the same evidence-not-YAML discipline as the
truncation budget.

**Tier 2 — attributed direct comparison, gating on convergence.** The
complete final 128-node, horizon-split frozen-frame direct cube `F128` and
its root-enclosure error cube `EF` are still computed and retained on
every run; their model/order-qualified digests must equal the
certificate's `frozen_gauss128_cube_sha256` and
`frozen_enclosure_error_cube_sha256`, and an alternate recomputation or
subset is forbidden. In the canonical correlation view all cubes have
shape `[N,B,F,4]`. Define the **truncation deficit**

```text
S_direct = max(1 Jy, max(abs(F128)+EF))
U_direct = abs(V0-F128) + EF
deficit_max = max(U_direct)
deficit_l2 = norm(U_direct) / max(norm(F128), sqrt(K)*1 Jy)
```

The deficit is never called agreement and has no universal numeric limit;
its honesty obligations are convergence and disclosure. With the
convergence levels `L1 = max(2, lmax//4)` and
`L2 = max(L1+1, lmax//2)`, each paired with `min(mmax, level)` and the
production quadrature, the run computes `deficit_max` at `L1`, `L2`, and
`lmax` and requires

```text
deficit_max(L1) > deficit_max(L2) > deficit_max(lmax)
deficit_max(L1) >= 2 * deficit_max(lmax)
```

— strict monotone decrease and at least the factor an algebraic
`L^-1/2` tail would produce over a quartering, so a non-converging or
diverging harmonic representation cannot license a result while the
attainable algebraic rates (`4x`..`8x` for `L^-1`..`L^-1.5`) pass with
margin. A `deficit_max(lmax)` of exact zero satisfies both lines. The
earlier fixed `4x`-per-level acceptance rule assumed spectral convergence
and is withdrawn; this quarter-to-full factor replaces it.

`F128.size`, `EF.size`, `V0.size`, `V_q.size`, and every compared
finite-cell count must each equal `K` in canonical
time/baseline/frequency/correlation order; `EF` is finite and
non-negative. All Section 14 qualified identities, every separately
evaluated count, both tier-1 residuals and their fixed limits, the three
deficit values and the convergence factors, the exact direct output
coverage ledger, and the complete shell/block diagnostic ledger enter
provenance; `deficit_max` and `deficit_l2` also enter the result's
provenance record — not the fixed Section 10 snapshot key set, which is
unchanged — so no consumer can read an m-mode result without its measured
deficit. Acceptance fixtures additionally declare, in the phase evidence,
a per-fixture `truncation_budget_jy` that `deficit_max` must not exceed —
a reviewed evidence field, never a YAML knob, and for a point-sky fixture
an honestly macroscopic one. An acceptance fixture must also sit in the
convergent regime, whose measured governing conditions are geometric, not
merely spectral: every point and native payload direction must stay well
clear of the horizon over the whole cycle — the M1 fixture is circumpolar
with zero frozen horizon roots, which also makes its enclosure-error cube
exactly zero — because near-horizon samples carry a non-decaying Gibbs
error that defeats the monotone predicate at any scale; and the fixture's
`lmax` is pinned by the accepted evidence, because the quarter-to-full
factor is not monotone in `lmax` (once `L1` itself resolves the smooth
kernel, all three levels sit on the horizon-step floor and the factor
collapses). A candidate fixture is qualified by measuring its three-level
deficit sequence and adopting it only with real margin on the `2x` floor;
a predicate is never widened to admit a fixture.

`lmax=4096`, and every input above `4088`, is a typed rejection rather than a
claim of convergence. The `4088` ceiling reserves at least eight additional
multipoles within the fixed `4096` transform ceiling. The same ceiling and
`mcheck` construction reserve a real signed-`m` tail even when `mmax=lmax`.
The remaining local shell
values and their complete block coverage are disclosed as diagnostics and are
not described as either a global truncation proof or direct-RIME equality.

## 8. Strict configuration and rejection contract

The accepted shape is:

```yaml
execution:
  simulator: mmode
  mmode:
    convention: radiosim.mmode-forward.v1
    frame_model: radiosim.frozen-cirs-rigid-era.v1
    harmonic_convention: radiosim.shaw-polarized-harmonics.v1
    lmax: 64
    mmax: 64
    quadrature_nside: 64
    working_memory_bytes: 1073741824
  solver:
    workers: 1
    executor: thread
```

The three convention fields are required exact literals. Integers are strict
and booleans are not integers. `working_memory_bytes` is a strict positive
integer used only for deterministic scheduling; it does not enter
`scientific_sha256`. The resolved chunk schedule and measured peak do enter
provenance. `execution.mmode` is required with `mmode` and forbidden with
`rime`; an absent/default block never changes a direct run.

All accepted Jones terms must be stationary in the ground frame. Current
bandpass, cable reflection, instrumental delay, cross-hand response, leakage,
parallactic rotation, static troposphere/ionosphere, closure error, and `Q`
bandwidth smearing can be represented in the reference transfer. The exact ERA
top hat already owns exposure averaging, so `Q.time_smearing=true` is rejected.
A gain `linear_drift` or `sinusoidal` time model is rejected. Future explicitly
time-varying beams, weather, TEC, pointing, or calibration models are rejected
until a design defines their mode coupling.

The required semantic issue codes and exact messages are:

| Code | Exact message |
|---|---|
| `mmode_block_required` | `execution.simulator='mmode' requires an explicit execution.mmode block.` |
| `mmode_block_forbidden` | `execution.mmode is only valid when execution.simulator='mmode'.` |
| `mmode_time_grid_required` | `execution.simulator='mmode' requires obs_time.mode='full_sidereal'; a UTC-uniform interval is not an m-mode grid.` |
| `rime_time_grid_required` | `obs_time.mode='full_sidereal' is only valid when execution.simulator='mmode'.` |
| `mmode_exposure_resolution` | `obs_time.integration_fraction is too small for distinct canonical binary64 exposure edges at this sidereal_samples.` |
| `mmode_nyquist` | `obs_time.sidereal_samples must be at least 2 * execution.mmode.mmax + 1.` |
| `mmode_tail_nyquist` | `obs_time.sidereal_samples must be at least 2 * resolved mcheck + 1 for the mandatory m-tail diagnostic.` |
| `mmode_quadrature` | `execution.mmode.lmax must be at most 2 * execution.mmode.quadrature_nside.` |
| `mmode_time_smearing` | `execution.simulator='mmode' owns ERA top-hat integration; jones.Q.time_smearing must be false.` |
| `mmode_static_gain` | `execution.simulator='mmode' requires jones.G.time_model.kind='constant'.` |
| `mmode_phase_center` | `execution.simulator='mmode' requires the canonical fixed zenith-drift phase centre.` |
| `mmode_point_morphology` | `execution.simulator='mmode' does not yet support Gaussian point-source morphology; use rime or remove the morphology.` |
| `mmode_polarization_frame` | `polarized m-mode input requires an explicit canonical tangent-polarization frame.` |
| `mmode_iers_range` | `the full-sidereal UTC mapping is outside the available offline IERS table.` |
| `mmode_truncation_check` | `execution.mmode.lmax leaves no room for the required harmonic tail check.` |
| `mmode_horizon_unresolved` | `execution.simulator='mmode' could not certify complete horizon-root isolation; tangent, identically-zero, and unresolved intervals are rejected.` |
| `mmode_m1_scalar_only` | `MModeSimulator phase M1 accepts Stokes I only; non-zero Q, U, or V requires accepted phase M2.` |
| `mmode_public_components` | `execution.simulator='mmode' supports point-source components only in this phase; a HEALPix-bearing sky requires a future accepted phase.` |
| `mmode_public_beam` | `execution.simulator='mmode' supports the scalar beam response only in this phase; a non-scalar resolved beam system requires a future accepted phase.` |

The dynamic frame-certificate rejection is rendered exactly as one line with
the following template and lower-case scientific notation:

```text
execution.simulator='mmode' frame certificate failed: phase_max={phase:.6e} rad (limit=5.000000e-03 rad); horizon_root_count_mismatches={root_count:d}; horizon_root_orientation_mismatches={orientation:d}; horizon_membership_mismatches={membership:d}; horizon_outside_slab_sign_mismatches={outside_sign:d}; horizon_unresolved_intervals={unresolved:d}; horizon_root_max={root_max:.6e} rad (limit=2.000000e-05 rad); horizon_mismatch_measure={mismatch_measure:.6e} rad (limit={mismatch_limit:.6e} rad); cube_max={cube_max:.6e} Jy (limit={cube_limit:.6e} Jy); cube_l2={cube_l2:.6e} (limit=5.000000e-05).
```

If no paired horizon root exists, `horizon_root_max` is rendered as the fixed
numeric `0.000000e+00`; the mismatch measure and limit are also rendered as
`0.000000e+00`. Zero pairs are valid for matching no-root trajectories when
the count, membership, sign, phase, and cube gates pass. If only one model has
roots, the non-zero root-count mismatch is authoritative. Tests assert the
complete deterministic fixture message, all ten gate results, and
pre-allocation failure.

Schema/type failures remain `ConfigSchemaError`; these cross-field/domain
failures remain `ConfigSemanticError`; an accepted schema combined with an
unsupported physical payload remains `UnsupportedConfigError`. Failure occurs
before backend allocation, output path creation, or harmonic work.

## 9. Backend, precision, worker, and memory policy

NumPy is the scientific reference. Astropy frame work, IERS mapping, beam
sampling, HEALPix geometry, and scalar/spin harmonic transforms are host-side
NumPy work for every backend. JAX and Dask may execute only the dense per-m
contractions and time synthesis — kernels whose public-path wiring
through `request.backend` is deferred to a future red-sliced phase, so
`execution.backend` is accepted and end-to-end inert for `mmode` today
and the Section 11 kernel blocks carry the backend evidence. This is
recorded as
`host_harmonics_backend_native_dense_v1`; it is not an end-to-end accelerator
claim, and `MModeSimulator.supports_gpu` is `False` without an independently
accepted exact-solver accelerator record.

Capability truth is phase-local. In M1, `MModeSimulator.supports_polarization`
is explicitly overridden to return `False`, and its request validator rejects
any sky with non-zero Q, U, or V using `mmode_m1_scalar_only`. The authoritative
Tier 7 characterization file
`tests/characterization/test_tier7_current_behavior.py` must pin both that
value and the unchanged `RIMESimulator.supports_polarization is True`; a new
simulator registry entry may not inherit the base class's permissive default.
Only accepted M2, after point, HEALPix, and hybrid full-Stokes direct oracles
pass, may deliberately flip the m-mode property and the named Tier 7
characterization assertion to `True`. Until accepted M2, documentation,
result provenance, and
registry introspection must all continue to report scalar-only support;
accepted M2 updates that same prose — including the m-mode strategy
description the registry reports — to the polarized truth alongside the
two licensed flips, because capability truth is phase-local and a
description contradicting the flipped property would itself be the
defect.

Harmonic geometry is computed in float64 and reference coefficients in
complex128. Before dense work, arrays are cast to the resolved accumulation
dtype; the final cube uses the resolved output dtype. Complex128 JAX requires
x64 and fails explicitly if unavailable. Backend parity uses this fixed
scale-aware predicate for complex128
(`rtol=1e-12`, `atol=1e-12*max(1,max|reference|)`). A separately named
complex64 row uses `rtol=5e-5` and `atol=5e-6*max(1,max|reference|)`; this is a
new low-precision contract and cannot replace the complex128 acceptance row.

`execution.solver.workers` owns independent frequency-block construction and
is clamped to the frequency count. Blocks are assembled in canonical frequency
order, so one worker and many workers meet the same backend predicate.

The complete baseline transfer is never materialized. The deterministic
scheduler orders frequency, signed-m, and baseline blocks, choosing the largest
block that fits `working_memory_bytes` under a component-by-component estimate.
It streams/discards each `B` block after contraction and retains only sky
coefficients, auditable per-antenna coefficients subject to the same budget,
the `v_m` cube, and one output synthesis block. It does not inspect free RAM or
change block order after an allocation failure.

`get_memory_estimate()` reports, separately:

- canonical sky coefficients;
- quadrature directions/weights and sampled Jones fields;
- optional per-antenna harmonic cache;
- the largest baseline-transfer block;
- retained m-mode visibilities;
- time-domain output and synthesis assembly; and
- backend/native allocations not included in the host estimate.

It records logical and scheduled dimensions and a one-block minimum. A budget
smaller than that minimum is rejected before allocation. The seven
components model the dense pipeline by design; the every-run Section 4.2
frame certificate's retained row — the complete Section 14.2 row with
its Section 12.1 ledgers — is excluded from the estimate and measured
separately by the Section 11 record's scoped host peak. Acceptance
measures
host peak and, where available, backend-native peak, and retains the
measured relation between estimate and peak as observed under
Section 11's memory predicate. No speed, scaling, or memory
advantage is claimed without the retained record in Section 11.

## 10. Result, provenance, and existing output formats

The registered strategy returns the backend-native `(T,B,F,2,2)` receptor
cube, and the public result container keeps its existing strict `(T,B,F,4)`
visibility array in the four row-major correlation labels — the exact
row-major flattening of that cube's receptor axes; neither shape changes.
Point, HEALPix, and hybrid remain solver provenance, not
separate output products. M1 widens the solver record to a strict tagged union:
the current `rime` snapshot stays exactly as it is. An m-mode snapshot has the
exact common fields `solver`, `sky_representation`, `convention`,
`execution_path`, `components`, and `component_element_counts`, followed by
exactly:

- `time_grid_convention`, `frame_model`, and `harmonic_convention`;
- `sidereal_samples`, `lmax`, `mmax`, and `quadrature_nside`;
- `quadrature_policy` and `truncation_policy`;
- `tangent_polarization_frame` and `stokes_v_basis_bridge`;
- `iers_table_sha256` and `frame_certificate_sha256`; and
- `transform_execution_policy`.

In M1 `tangent_polarization_frame` is the exact literal
`not_applicable_scalar_m1`; after M2 it is the exact six-key Section 5.1 object.
`stokes_v_basis_bridge` is always `radiosim.stokes-ne-theta-phi.v1`. Neither
field is nullable.

The scientific snapshot excludes worker count and memory budget but includes
every scientific convention and truncation dimension. Backend/device, worker,
chunk, library-version, IERS-path, and timing details remain provenance.
RIME serialization must be byte-identical after the union is introduced.

In-memory, summary JSON, HDF5, UVFITS, and Measurement Set paths all write the
same synthesized UTC sample centres and integration widths. Summary JSON is
still metadata-only. HDF5 preserves the complete tagged solver snapshot.
UVFITS/MS keep the canonical zenith phase centre, east-X/circular feed metadata,
four correlation products, UTC coordinates, and history lines naming the
m-mode/frame/harmonic conventions. Reader round trips must reconstruct and
authenticate the m-mode solver snapshot; a reader that silently labels it
`rime` fails acceptance.

No new public harmonic-output format lands in `SCI-004`. Debug `a_lm`, `B_lm`,
and `v_m` arrays may be retained only in bounded evidence artifacts, not in a
normal result or standard visibility file.

## 11. Retained fingerprints, CI, and performance records

The existing direct families and every shipped configuration must remain
byte-identical. New characterized families are:

```text
mmode_single_scalar_mode
mmode_point_stokes_i
mmode_point_full_stokes
mmode_circular_receptor
```

The set names exactly the capability accepted M2 licenses through the
public solve path. `mmode_circular_receptor` is the full-Stokes point
fixture under the accepted circular receptor basis and must be qualified
by Section 7.3's measured protocol at `S3` before it is pinned; if it
proves unqualifiable, M3 pauses for design correction. The former
HEALPix and hybrid families are deferred: the accepted harmonic
machinery reaches the public path only through a future red-sliced
phase, the public path rejects a HEALPix-bearing payload and a
non-scalar resolved beam system with the Section 8 typed issues before
any work, and `A3`'s `claims_not_licensed` must carry both deferrals.
The former `mmode_nonscalar_east_x` is removed because the shipped
default receptor set is east-X: it reproduced `mmode_point_full_stokes`
byte for byte and characterized nothing.

Each family records the raw cube, `scientific_sha256`, solver snapshot, ERA/UTC
grid, harmonic index table, and input identity. The fingerprint row retains the
complete result-derived `characterization_time_manifest` beside
`era_utc_grid_sha256`; the latter uses the namespaced domain
`radiosim.sci004.characterization-time.v1`, computed from the retained
`SimulationResult` exactly as the strict validator re-derives it, and is
distinct from the phase manifest's `canonical_era_grid_sha256`. Correction
#25 supersedes the path-bearing input domain with
`radiosim.sci004.characterization-input.v2`. Every fingerprint row embeds its
complete Section 14.0 v2 characterization-input manifest beside the digest;
that manifest embeds and joins the exact path-independent phase fixture-input
manifest, rather than the arbitrary filesystem path in `resolved_config`.
Moving byte-identical layout content between absolute roots cannot change the
identity; changing a semantic site, antenna, baseline, receptor, loaded-beam,
frequency, sky, Jones, time-grid, or solver input must change it. Section
14.0's solver-internal domains do not substitute for either result-derived
characterization domain. A changed m-mode pin requires
old/new cubes and an equation-level explanation; no digest is appended merely
because CI printed it.

The production family record and the Section 14.2 evidence row are distinct
closed schemas. At the pre-fix v1 source,
`mmode_characterization_record` has exactly this ordered seven-key record:

```text
family_id, raw_cube_sha256, scientific_sha256, solver_snapshot,
era_utc_grid_sha256, harmonic_index_table_sha256, input_identity_sha256
```

Replacement S3 changes it to exactly this ordered nine-key v2 record:

```text
family_id, raw_cube_sha256, scientific_sha256, solver_snapshot,
characterization_time_manifest, era_utc_grid_sha256,
harmonic_index_table_sha256, characterization_input_manifest,
input_identity_sha256
```

Both complete manifests are direct record members immediately before their
adjacent digests; every old key and assertion remains. In v2 the function's
signature requires keyword-only `family_id` and
`phase_input_identity_manifest`; the latter is the complete same-run
`radiosim.mmode-input-identity.v1` manifest constructed from the solver request,
not `resolved_config` and not a digest-only stand-in. Omission, an unknown key,
a path-bearing value, or a manifest whose independently recomputed phase-input
digest does not join the result's instrument, receptor, beam, frequency, time,
solver, and scientific surfaces raises `InvalidResultError` before a record is
returned. The evidence generator supplies its retained same-family solver
bundle manifest; the two-root and semantic-mutation oracles construct theirs
independently. No architecture that leaves the production family record at v1
while building an evidence-only v2 manifest complies.

The accepted CI-001 successor discipline applies to every new family. The
initial harvest binds exactly the platform/Python cells this phase's
acceptance actually runs on; every other cell and every newly observed
NumPy/OpenBLAS dispatch class enters afterwards by the standing
admission discipline, exactly as the accepted AVX-512 admissions did. A novel class is adjudicated by cubes under Section 9's
fixed complex128 predicate before it can join an observation set. M-mode must not
make a compatibility cell green by weakening a harmonic, direct, frame, or
backend tolerance.

Non-gating performance records live under
`output/benchmarks/reference/sci004/<UTC>-<host>.json`. They use the exact
top-level schema literal `radiosim.benchmark.sci004.v1` and Section 14's
canonical JSON and typed-array digest rules. The record deliberately defines
its own schema rather than extending the accepted
`radiosim.benchmark.perf001.v1` inventory: every SCI-004 row must join a
frame certificate, scientific identity, deterministic block schedule, and
direct/backend comparison that the PERF-001 record has no analogue for, and
each schema remains governed by its own strict validator. The top-level
object has exactly `schema_version`, `provenance`, `workloads`, and
`dense_invariance`. `dense_invariance` is an array with exactly one
entry per comparison group in fixture order; each entry has exactly
`comparison_group_id`, `numpy_cube_sha256`, `jax_cube_sha256`,
`dask_cube_sha256`, and `identical`, where `identical` must be `true` —
the measured backend-invariance of the dense path, retained as fact.

`provenance` has exactly:

```text
schema_version, recorded_at_utc, radiosim_version, source_sha,
git_tree_sha256, working_tree_clean, host_tag, platform, machine, cpu_model,
cpu_count_logical, python_version, pixi_environment, pixi_manifest_sha256,
pixi_lock_sha256, numeric_packages, iers_table_sha256,
transform_execution_policy, workload_count
```

Its schema literal is `radiosim.benchmark.sci004.provenance.v1`.
`recorded_at_utc` is an exact UTC `YYYY-MM-DDTHH:MM:SSZ` string; its
punctuation-stripped value is `<UTC>` in the filename. `host_tag` matches
`[a-z0-9][a-z0-9-]{0,62}` and is the filename's `<host>`. `source_sha` is the
clean exact `S3`; `working_tree_clean` is true; `pixi_environment` is
`default`; `transform_execution_policy` is
`host_harmonics_backend_native_dense_v1`; and `workload_count` is exactly
nine. `numeric_packages` has exactly `astropy`, `dask`, `erfa`, `healpy`,
`iers_package`, `jax`, `jaxlib`, `numpy`, and `scipy`. Each value is a
non-empty normalized distribution-version string or `not-installed`; a
package used by a retained row may not be `not-installed`.

Each workload row has exactly:

```text
workload_id, comparison_group_id, fixture_id, input_identity_sha256,
frame_certificate_sha256, scientific_sha256, result_cube_sha256, source_sha,
working_tree_clean, backend, backend_runtime, device_kind, precision,
accumulation_dtype, result_dtype, workers, n_antennas, n_baselines,
n_frequencies, sidereal_samples, lmax, mmax, quadrature_nside,
n_point_sources, n_healpix_pixels, sky_representation,
working_memory_bytes, resolved_block_dimensions, timings, memory,
direct_comparison, backend_comparison, dense_execution,
kernel_backend_block, claims_not_licensed
```

The official v1 record has exactly the Cartesian product, in this order:

```text
fixture: mmode_single_scalar_mode, mmode_point_stokes_i,
         mmode_point_full_stokes
backend within each fixture: numpy, jax, dask
```

Thus `comparison_group_id == fixture_id`,
`workload_id == fixture_id + ":" + backend + ":standard"`, and the array has
exactly nine rows. The three rows in a fixture group have identical input,
frame-certificate, dimension, precision, worker, and memory-budget fields;
scientific/result cube identities remain backend-qualified and are compared by
the fixed predicate. Input identities are distinct across fixture groups. The sky
representation is `point` for all three fixture groups, with positive
point counts, zero for the absent HEALPix representation, and distinct
input identities across groups.

`backend` is exactly `numpy`, `jax`, or `dask`; `device_kind` is `cpu`;
`precision` is `standard`; and both dtype fields are `complex128`.
`backend_runtime` has exactly `implementation`, `implementation_version`,
`kernel_runtime`, and `kernel_runtime_version`. The implementation/runtime
pairs are respectively NumPy/NumPy, JAX/JAXlib, and Dask/NumPy, with versions
equal to the corresponding provenance values. Counts and byte sizes are exact
JSON integers, booleans are not integers, `workers` and every scientific
dimension are positive where Sections 7--9 require them, and every digest uses
its Section 14 domain.

`resolved_block_dimensions` has exactly:

```text
frequency_block_max, signed_m_block_max, baseline_block_max,
packed_value_block_max, scheduled_block_count, schedule_rows, schedule_sha256
```

Each `schedule_rows` entry has exactly:

```text
block_index, frequency_start, frequency_stop, signed_m_start, signed_m_stop,
baseline_start, baseline_stop, packed_value_count
```

Rows are in actual canonical frequency/signed-m/baseline execution order;
`block_index` is contiguous from zero; ranges are half-open; and
`scheduled_block_count` equals the non-empty array length. Each maximum is
recomputed from the rows. The validator independently rebuilds the complete
deterministic schedule from the scientific configuration and
`working_memory_bytes`, requires exact ordered equality, and hashes the rows
under domain `radiosim.sci004.block-schedule.v1`. Missing, duplicate,
reordered, overlapping, or uncovered work is invalid.

`timings` has exactly:

```text
clock, warmup_iterations, synchronization_method, frame, sky_transform,
beam_transfer, dense_contraction_and_synthesis, host_transfer, total,
direct_reference
```

`clock` is `time.perf_counter_ns`; `warmup_iterations` is positive; and the
synchronization methods are `numpy_eager_v1` for every shared dense
series and respectively `jax_block_until_ready_v1` and `dask_compute_v1`
for the JAX and Dask rows' kernel blocks below. A measured timing series has
exactly `status` and `sample_seconds`, with status `measured` and at least five
finite non-negative samples in execution order. A non-measured series has
exactly `status` and `reason`, where status is `not_applicable` or
`not_measured` and reason is non-empty. No timing-series field is nullable.

`frame`, `sky_transform`, `beam_transfer`,
`dense_contraction_and_synthesis`,
and `total` are measured and have identical sample cardinality and indexed
iterations. The dense series is deliberately fused: production's
`contract_and_synthesize` interleaves the per-`m` contraction with the
sample accumulation and has no sequential split point, so a separate
per-stage split of the shared series would be an invented number; the
stage names `per_m_contraction` and `synthesis` denote exactly the
kernel-block stages below, never shared row series. `host_transfer` is measured or `not_applicable`.
`direct_reference` is measured with at least five samples or `not_measured`;
the correctness comparison remains mandatory either way. Each total sample
begins at solver entry after fixture construction and ends after synchronized
host availability. It includes the complete Section 4.2 frame certificate,
scheduling, all named stages, and orchestration overhead, but excludes imports,
warm-up, memory measurement, and output writing. For each indexed iteration,
total time is not smaller than the sum of applicable named stages. Memory
measurements use separate untimed synchronized calls.

`memory` has exactly:

```text
measurement_scope, estimated_host_peak_bytes, measured_host_peak_bytes,
host_measurement_method, host_measurement_limitations,
measured_native_peak_bytes, measured_native_peak_bytes_reason,
native_measurement_method, native_measurement_limitations,
estimate_covers_measured_host_peak
```

`measurement_scope` is
`single_mmode_solver_call_excluding_fixture_and_output_v1`.
`host_measurement_method` is `process_rss_sampled_delta_v1`. It is the
incremental peak resident set of the **current generator process**, measured by
a separate standard-library-only sampler process so the measured solver carries
neither a sampling thread nor an allocation hook. The generator starts the
sampler with the same Python executable and this phase's tracked evidence tool
under its hidden sampler subcommand, passing exactly the generator PID and the
fixed `10,000,000 ns` interval. The child requires that PID to equal its live
parent PID, then obtains instantaneous RSS as follows:

- on Linux, parse the second decimal field of `/proc/<pid>/statm` as resident
  pages and multiply it by `os.sysconf("SC_PAGE_SIZE")`; and
- on Darwin, call `/usr/lib/libproc.dylib`'s `proc_pidinfo` with
  `PROC_PIDTASKINFO`, require the returned byte count to equal the complete
  `proc_taskinfo` structure size, and read its unsigned
  `pti_resident_size` field.

An unsupported platform, missing/malformed counter, overflow, short Darwin
structure, dead or changed parent, or non-positive page size fails closed. The
child takes the baseline sample, emits and flushes exactly one canonical JSON
`READY` line containing `status`, `target_pid`, `sampling_interval_ns`, and
`baseline_rss_bytes`, and then samples on deadlines advanced from
`time.monotonic_ns()` at exactly 10 ms. After the untimed solver call the parent
writes exactly `STOP\n`; the child takes one synchronous final sample and emits
exactly one canonical JSON `RESULT` line containing `status`, `target_pid`,
`sampling_interval_ns`, `baseline_rss_bytes`, `peak_rss_bytes`,
`final_rss_bytes`, `sample_count`, and `measured_host_peak_bytes`, where

```text
measured_host_peak_bytes =
    max(all periodic samples, baseline, final) - baseline
```

The parent applies bounded READY/result/wait timeouts, requires those two lines
and no additional stdout, empty stderr, exact PID/interval/baseline agreement,
a sample count of at least two including the baseline and final observations,
`peak_rss_bytes >= baseline_rss_bytes`, internally consistent non-negative
integers, and a zero child exit. READY must arrive within 10 seconds; after
`STOP\n`, the result and clean child exit must arrive within 5 seconds. The
legitimate solver call itself has no sampler timeout. Every success, solver
exception, sampler exception, malformed protocol, or timeout closes the pipes
and reaps the sampler; timeout cleanup terminates, then kills and reaps if
necessary. A partial measurement is never retained.
`resource.getrusage().ru_maxrss` is forbidden because it is a
process-lifetime high-water mark (and has platform-dependent units), and shell
`ps` and `psutil` are forbidden because neither supplies this exact
standard-library, current-call contract.

The host limitation array carries, in sorted unique form, that 10 ms sampling
may miss a shorter transient resident-set peak, that a baseline delta does not
count solver allocations satisfied from pages already resident before the
call, that current-process RSS excludes child-process and accelerator-device
memory, and that Section 9's dense estimate excludes the every-run Section 4.2
frame certificate and its retained ledgers. Native method is one of
`jax_device_memory_stats_v1`, `dask_worker_metrics_v1`, or `unavailable`;
`process_rss_sampled_delta_v1` is a host method and is invalid in a native
field. Both limitation arrays are sorted, unique, and non-empty.

`dense_execution` is exactly the literal `numpy_host_v1` on every row:
the public solve path's dense stages are
backend-invariant — `contract_and_synthesize` takes no backend and
`request.backend` reaches no dense array work — a measured fact this
record retains rather than papers over, carried in the top-level
`dense_invariance` array above. Each fixture group measures its
end-to-end timing and memory series once, on the NumPy row, and the JAX
and Dask rows carry those identical shared values; the shared `memory`
object's native method is `unavailable`, never the host RSS method or a
backend-device method — the backend-device methods appear only inside kernel
blocks. `kernel_backend_block` is a
status-discriminated object for row uniformity: on the NumPy row it has
exactly `status` and `reason` with status `not_applicable`; on the JAX
and Dask rows of a fixture group whose resolved payload is scalar — the
production block table carries one field, and the routed contraction
kernel's contract covers exactly Section 5.3's four science fields, so
a per-`m` kernel measurement for such a group would describe nothing
its own solve does — it has exactly `status` and `reason` with status
`not_applicable_scalar_table`, the reason naming that kernel contract;
on the JAX
and Dask rows of a polarized group it has exactly `status`,
`per_m_contraction`, and
`synthesis` with status `measured`, each stage object having exactly
`sample_seconds` (at least five finite non-negative samples in
execution order, separate objects never merged into the row's shared
timing keys), `synchronization_method` (the row's method, applied to
exactly those kernel calls), `native_measurement_method`,
`measured_native_peak_bytes`, `measured_native_peak_bytes_reason`, and
`stage_comparison`; and each `stage_comparison` has exactly
`predicate_id`, `reference_stage_sha256`, `candidate_stage_sha256`,
`expected_cell_count`, `compared_finite_cell_count`,
`reference_scale_jy`, `maximum_absolute_deviation_jy`,
`maximum_relative_deviation`, `rtol`, `atol_jy`, and `pass`, under the
unchanged `sci004_backend_complex128.v1` predicate with the NumPy
kernel output on identical inputs as reference — so every retained
kernel deviation is a real computation's, never a self-comparison.
Wiring `request.backend` through the public dense
stages is future red-sliced work; `execution.backend` is accepted and
end-to-end inert for `mmode` today, and `A3`'s `claims_not_licensed`
must carry that third deferral alongside the two
accepted-capability deferrals.

The only nullable value anywhere in the benchmark document is
`measured_native_peak_bytes`. If it is an integer, its adjacent reason is
exactly `measured` and the native method is not `unavailable`. If it is null,
its reason is non-empty, the method is `unavailable`, and the same reason
occurs in `native_measurement_limitations`. Host/native peaks are scoped
increments over a synchronized pre-call baseline. Validation requires

```text
measured_host_peak_bytes <= working_memory_bytes
estimated_host_peak_bytes <= working_memory_bytes
```

and `estimate_covers_measured_host_peak` to be the measured boolean
`measured_host_peak_bytes <= estimated_host_peak_bytes` — retained as
observed, never chosen. No expected truth value is pinned: sampled RSS delta
and Python-heap allocation-event peak are different measured quantities, and
allocator reuse or sampling cadence can make the former smaller while native
resident pages can make it larger. `host_measurement_limitations` must carry
the four ruled limitations above on every row, and an
`estimated_host_peak_bytes` chosen after seeing the measurement to force the
boolean true remains the condemned self-comparison form.

`direct_comparison` has exactly:

```text
predicate_id, reference_cube_sha256, candidate_cube_sha256,
reference_error_cube_sha256, horizon_free_cube_sha256,
horizon_free_qcheck_cube_sha256, quadrature_shell_cube_sha256,
expected_cell_count, compared_finite_cell_count,
evaluated_error_cell_count, numerical_scale_jy,
horizon_free_shell_max_jy, horizon_free_shell_l2,
horizon_free_shell_max_limit_jy, horizon_free_shell_l2_limit,
quadrature_shell_max_jy, quadrature_shell_l2,
reference_scale_jy, deficit_max_jy, deficit_l2,
deficit_max_quarter_jy, deficit_max_half_jy,
convergence_factor, pass
```

Its predicate ID is `sci004_two_tier_direct.v3`; reference and error
digests are the authenticated frame certificate's final 128-node frozen direct
and frozen enclosure-error cubes; candidate equals `result_cube_sha256`;
the two horizon-free digests are the tier-1a `W0`/`W_q` cubes and
`quadrature_shell_cube_sha256` is the retained with-horizon
`V(lmax,mmax,qcheck)` cube.
With `K=sidereal_samples*n_baselines*n_frequencies*4`,
`S_num=max(1 Jy,max(abs(W_q)))`, and
`S=max(1 Jy,max(abs(reference)+error))`, validation recomputes every value
from Section 7.3's two-tier formulas:

```text
expected_cell_count = compared_finite_cell_count = evaluated_error_cell_count = K
horizon_free_shell_max_jy = max(abs(W0-W_q))
horizon_free_shell_l2 = norm(W0-W_q)/max(norm(W_q),sqrt(K)*1 Jy)
horizon_free_shell_max_limit_jy = 1e-8*S_num + 1e-10 Jy
horizon_free_shell_l2_limit = 1e-8
quadrature_shell_max_jy = max(abs(candidate-V_q))
quadrature_shell_l2 = norm(candidate-V_q)/max(norm(V_q),sqrt(K)*1 Jy)
deficit_max_jy = max(abs(candidate-reference)+error)
deficit_l2 = norm(abs(candidate-reference)+error) /
    max(norm(reference),sqrt(K)*1 Jy)
convergence_factor = deficit_max_quarter_jy / deficit_max_jy
```

`pass` is true exactly when all cells are finite, errors are non-negative,
both tier-1a horizon-free predicates pass, and the Section 7.3
convergence predicates hold
(`deficit_max_quarter_jy > deficit_max_half_jy > deficit_max_jy` and
`convergence_factor >= 2`, with an exact-zero `deficit_max_jy` passing
both). The with-horizon shell values and the deficit values are recorded,
never bounded by a universal limit.

`backend_comparison` has exactly:

```text
predicate_id, reference_workload_id, reference_cube_sha256,
candidate_cube_sha256, expected_cell_count, compared_finite_cell_count,
reference_scale_jy, maximum_absolute_deviation_jy,
maximum_relative_deviation, rtol, atol_jy, pass
```

Its predicate ID is `sci004_backend_complex128.v1`; its reference is the NumPy
row in the same comparison group; `rtol` is exactly `1e-12`; and `atol_jy` is
exactly `1e-12*max(1 Jy,max(abs(reference)))`. With
`S_backend=max(1 Jy,max(abs(reference)))`, the validator recomputes

```text
reference_scale_jy = S_backend
maximum_absolute_deviation_jy = max(abs(candidate-reference))
maximum_relative_deviation = maximum_absolute_deviation_jy/S_backend
```

and every
cell's `abs(candidate-reference) <= atol_jy + rtol*abs(reference)` and both
retained maxima. The NumPy row self-references and has exact-zero deviations; at this
phase every row's deviations are exactly zero because the dense path is
backend-invariant, so this row-level comparison restates the
`dense_invariance` fact in comparison form and licenses no
computed-parity claim — computed parity lives only in the kernel
blocks' `stage_comparison` objects.
All counts equal `K`, all cells are finite, and `pass` is true. The separate M2
complex64 row and its Section 9 predicate remain required but are not
substituted into this standard-complex128 performance inventory.

Every workload row carries this exact lexicographically sorted array:

```text
general_speedup
gpu_or_accelerator_support
mmode_end_to_end_backend_execution
perf001_evidence_or_closure
performance_regression_gate
unmeasured_workloads
```

Unknown, missing, duplicate, non-finite, or wrongly typed values fail
validation. Workload, schedule, and timing-sample order are semantic and may
not be sorted during serialization; limitation and claim arrays must already
be sorted and unique. Validation also rejects dirty or unknown source, a stale
manifest/lock, wrong path/timestamp/host binding, a symlinked or non-regular
output, input-identity mismatch, false comparison/memory predicates, or
overwrite. Generation is atomic and no-overwrite.

A record is evidence only of these nine measured CPU rows. Timing values never
gate CI and license neither a speedup nor a memory/accelerator advantage.
`PERF-001` statements remain governed by separate accepted PERF-001 records.
Any different SCI-004 performance inventory requires a schema-version bump.

## 12. Red-first validation programme

Every implementation phase begins with a red-test commit and a retained record
naming the node ID, expected equation/behavior, observed pre-fix failure, and
why the fixture is not defective.

### 12.1 Horizon-split exposure oracle

An exposure is never integrated across an unsplit horizon discontinuity. For
each frozen CIRS direction, Section 4's attitude makes

$$
\sin(\operatorname{alt}(u))=A\cos(2\pi u)+B\sin(2\pi u)+C.
$$

The oracle derives binary64 `A,B,C` from the recorded matrices and solves this
equation analytically over the complete unwrapped cell-centred cycle `H_N`,
not by sampled sign changes. Their integer-ratio representations determine the exact
sign of `A**2+B**2-C**2`; a floating epsilon does not classify the topology.
Set `R=hypot(A,B)` and `phi=atan2(B,A)`. The exhaustive cases are:

1. `A==B==0` and `C!=0`: the trajectory has the constant sign of `C` and no
   root;
2. `A==B==C==0`: the trajectory is identically on the horizon and is a
   typed `mmode_horizon_unresolved` rejection;
3. `rho2=A**2+B**2>0` and `C**2>rho2`: the direction is circumpolar above or
   below and has no root;
4. `rho2>0` and `C**2==rho2`: the single stationary/tangent equality is a
   typed `mmode_horizon_unresolved` rejection; and
5. `rho2>0` and `C**2<rho2`: the two and only two transverse root angles are
   `phi +/- acos(-C/R)`. Certified interval division by mathematical `2*pi`
   reduces them into turns `[0,1)`, and a unique integer turn lifts each into
   the exact one-turn `H_N`.

The two-root formulas are used only after the exact topology decision. Each
root is placed in a certified exact-turn sign-changing bracket and refined
until the outward radian width is at most `1e-13 rad`; failure to obtain two
distinct transverse brackets rejects
as `mmode_horizon_unresolved`. Each retained root must have an absolute
numerator residual at most `2e-13`, a strictly non-zero certified derivative
interval, and the analytic sign must be verified on every open piece. It is
labelled rising for
`-A*sin(2*pi*u)+B*cos(2*pi*u)>0` and setting for the negative sign. The frozen
root has exactly one integer-turn lift into `H_N`; its periodic copy is not
stored. Correctly rounded `sinpi(2*u)`/`cospi(2*u)` interval kernels, not a
rounded endpoint span or binary64 modulo, certify this topology.

The pressure-zero operational Astropy trajectory is certified independently
of the frozen model, and it consumes only public
`SkyCoord.transform_to(AltAz(obstime=..., location=..., pressure=0))`
values; a private frame surrogate is never an authority. Define
`f_o(u)=sin(alt_operational(u))` using the exact installed bundled-IERS
identity and UTC/UT1 mapping from Section 4. The census domain is the
closed unwrapped interval `[h_N^-,h_N^+]`, with half-open root census
`H_N=[h_N^-,h_N^+)`; it is not assumed periodic because precession,
nutation, polar motion, and IERS interpolation evolve during the cycle,
and this domain contains every exact exposure `[u_k^-,u_k^+]`.

Operational completeness rests on one design-frozen analytic ceiling, not
on an interval extension of the installed ERFA expression graph. The
operational direction is a unit vector transported by the rigid Earth
rotation at exactly one cycle per turn composed with the operational
corrections — polar-motion drift, UT1 interpolation, precession-nutation
evolution, annual and diurnal aberration, and light deflection — whose
combined angular rates over one cycle are bounded by the cited IERS
Conventions (2010) magnitudes at well below `1e-3` of the rigid rate.
Therefore `|d f_o / d u| <= L_op` on the whole domain, with the frozen
ceiling

```text
L_op = 6.2895
```

in `turn^-1` units, exceeding `2*pi*(1+1e-3)`. `L_op` is a constant of
this design, never fitted or configurable; a trajectory violating it would
violate the frozen attitude model itself and is outside the certified
regime.

Isolation is the deterministic **certified-ceiling scan**:

- The initial partition of `[h_N^-,h_N^+]` is the uniform exact-turn grid
  of spacing `h_0 = 2**-12` turn, refined so that every retained centre
  and edge turn from the same Section 3.1 grid object and every frozen
  root bound is also a cell boundary. `f_o` is evaluated exactly once at
  every distinct cell boundary; evaluations may be batched, each boundary
  value is computed once and reused, and the evaluation set is
  deterministic.
- A cell `[a,b]` of exact width `h` is proven root-free when
  `min(|f_o(a)|,|f_o(b)|) > L_op*h`: any zero inside the cell would force
  both endpoint magnitudes to at most `L_op*h` by the ceiling. Such a cell
  is classified `ceiling_excludes_root`.
- A cell with `f_o(a)*f_o(b) < 0` contains a crossing; it is bisected at
  its exactly representable midpoint until the sign-changing bracket's
  outward radian width is at most `1e-11 rad`, giving a certified
  operational root enclosure classified `scan_crossing`. The retained root
  must satisfy `|f_o(midpoint)| <= 5e-12`; the cited derivation
  `L_op * (1e-11/2/(2*pi))` evaluates to `5.005e-12`, and the retained
  `5e-12` is deliberately the marginally stricter rounding — the safe
  direction, reachable only by a trajectory saturating the ceiling
  uniformly across the half-enclosure. A larger residual rejects.
- Any other cell is bisected at its exactly representable midpoint and
  both children re-enter the queue. A cell whose exact width reaches
  `2**-44` turn with neither classification — the deep-tangency
  signature — rejects the entire certificate as
  `mmode_horizon_unresolved`. The `2**-44` turn floor is a constant of
  this scan, spelled in turns because every scan bound is an exact-turn
  quantity: at roughly `3.6e-13 rad` it sits nearly thirty times below
  the `1e-11 rad` refinement target, so a transverse crossing is isolated
  long before any cell reaches it and only a genuine near-tangency can
  drive the queue that deep. No root
  is silently merged or discarded.
- Each retained root's census orientation comes from the signs of `f_o` at
  the two probe turns exactly `1e-8` turn outside its enclosure on either
  side: `rising` for a negative-to-positive transit and `setting` for the
  reverse. Both probe magnitudes must exceed `1e-10`; a smaller, zero, or
  same-side probe sign rejects as `mmode_horizon_unresolved`, so a
  near-tangent transit is rejected rather than misclassified.

Cells own their left endpoint and are half-open on the right for the root
census. A crossing whose enclosure upper bound equals `h_N^+` is an
authenticated endpoint event excluded from `H_N`; there is no assertion
that `f_o(h_N^-)==f_o(h_N^+)`. Only exhaustion of the queue with every
cell classified, zero unresolved cells, and disjoint certified root
enclosures is a complete operational enumeration: the ceiling inequality —
not a coarse sample's luck — is what proves a same-sign cell hides no
crossing, because any crossing inside a width-`h` cell forces both
endpoint magnitudes to at most `L_op*h`, which the root-free rule refuses
to classify without refinement.

Every terminal scan cell emits one canonical row with exactly these
fields, in this order:

```text
direction_id, cell_index, turn_lo, turn_hi, classification,
f_lo_f64be, f_hi_f64be, ceiling_margin_f64be, left_sign, right_sign,
root_turn_lo, root_turn_hi, root_orientation, root_residual_f64be
```

`classification` is exactly `ceiling_excludes_root`, `scan_crossing`,
`guard_interval`, or
`excluded_upper_endpoint`, the last permitted only when the crossing's
enclosure upper bound equals `h_N^+`; it does not enter the root census.
A `guard_interval` row exists because the ceiling rule degenerates beside
a shallow crossing: near a root with local slope `beta*L_op`, a
root-adjacent cell classifies root-free only after roughly `log2(1/beta)`
extra refinements, so bisection terminates for ordinary crossings but the
innermost splinter of a crossing with
`beta < 2**-44 turn / (enclosure width) ~ 0.036` reaches the
unresolved floor before classification — and the probe-magnitude floor
still accepts transverse crossings down to `beta ~ 1.6e-3`, so the
literal partition would reject exactly that accepted shallow range.
Guards close it uniformly and harmlessly. Each retained crossing
therefore owns up to two flanking `guard_interval` rows covering exactly
the gap between its enclosure and the nearest classified neighbours, each
guard's width at most the `1e-8` turn probe offset; the crossing's probe
signs sit at or beyond the guards' outer ends, so an undetected extra
crossing pair inside the guards would leave the probe signs opposite and
is excluded from mattering physically by the Section 12 rule below, which
error-disks the guards. `ceiling_margin_f64be` is
`min(|f_lo|,|f_hi|) - L_op*(turn_hi-turn_lo)`
for a `ceiling_excludes_root` row, which must decode to a positive
binary64, and `F64(0)` otherwise. For a root-free row `left_sign` and
`right_sign` are the endpoint value signs; for a crossing row they are the
probe signs, in `{-1,1}`, and must differ; for a guard row they are the
endpoint value signs with zero permitted at the root-adjacent end. The
three root fields are JSON
null exactly for `ceiling_excludes_root` and `guard_interval`; a crossing
or excluded-endpoint
row has canonical-rational root bounds, `root_orientation` equal to
`rising` or `setting`, and a non-null residual. Root-census reconstruction
counts `scan_crossing` rows only; it rejects any duplicate owned root and
any guard row not adjacent to its crossing's enclosure or another guard.
Every `*_f64be` value is the lowercase 16-hex-character encoding of
`struct.pack(">d", value)`, never a JSON float. Turn and root bounds are
the canonical reduced rationals from Section 3.1. In direction-ledger
order, rows are sorted by exact `turn_lo`, and `cell_index` is contiguous
from zero. Together with the retained root enclosures they form a
gap-free, overlap-free half-open partition of
`H_N`; the closed `h_N^+` endpoint appears only as the last upper bound.
`horizon_isolation_interval_count` is exactly the total number of these
terminal rows across all directions, not a count of visited branch cells.

The scan array is serialized as UTF-8 JSON with the row-field order above,
`ensure_ascii=true`, separators `(',',':')`, and no whitespace or trailing
newline. `horizon_scan_ledger_sha256` is SHA-256 of exactly those bytes,
computed streamingly at generation. The full terminal-cell array is
sixteen million rows for the bounded driver — `16,835,749` terminal rows
including guards, measured by the accepted M1 evidence
artifact — so the retained evidence
embeds a bounded projection of it rather than the array itself: every
`scan_crossing`, `excluded_upper_endpoint`, and `guard_interval` row
verbatim as
`horizon_scan_crossing_rows`, plus one per-direction summary row as
`horizon_scan_summary_rows` with exactly `direction_id`,
`terminal_cell_count`, `boundary_evaluation_count`, `crossing_count`, and
`min_ceiling_margin_f64be`, in direction-ledger order. The scan itself is
deterministic given the frozen constants, the retained grid object, and
the public Astropy API, so the full array is reconstructible by replay:
the `A1` reviewer's certificate re-derivation regenerates it per
direction and re-digests it against `horizon_scan_ledger_sha256`, while
the strict evidence validator checks the embedded crossing and guard
rows, the
summary joins, the summary-count arithmetic
(`horizon_isolation_interval_count` equals the summary rows'
`terminal_cell_count` sum), and the digest's form — one of exactly two
deliberate replay-deferral rulings, alongside Section 7.3's
transfer-sample concatenations, both made because a multi-gigabyte
committed artifact and half-million-row ledgers would defeat review
rather than serve it, and both discharged by the mandatory `A1`
re-derivation.
A present `guard_interval` row is authenticated by the reconstruction
rules above — its adjacency to its crossing's enclosure or another
guard, its position in the neighbouring terminal cell, its endpoint
signs, its null root fields, and its width bound — but the projection
cannot distinguish a flank whose refinement classified completely from
one whose guard row was omitted, and it retains nothing beyond a
guard's outer bound to compare that bound against; deleting a retained
guard row, or perturbing its outer bound within the width cap, is
therefore invisible to the strict validator, and both are discharged,
exactly
as any omitted `ceiling_excludes_root` row is, by that same `A1`
re-derivation against `horizon_scan_ledger_sha256`.
The scan identity is a second canonical JSON object with fields, in order,
`schema_version`, `algorithm_id`, `implementation_files`, `constant_rows`,
`astropy_version`, `erfa_version`, and `iers_table_sha256`. The first two
literals are both `radiosim.mmode-operational-horizon-scan.v1`.
`implementation_files` is the path-sorted array of exact raw-byte SHA-256
rows for `src/radiosim/core/mmode/frame.py` and
`src/radiosim/core/mmode/time.py`, each row having exactly `path` and
`sha256`. `constant_rows` contains every ceiling, spacing, refinement,
root-width, probe-offset, unresolved-width, and residual constant consumed
by the implementation — `L_op`, `h_0`, the `1e-11 rad` enclosure width,
the `1e-8` turn probe offset, the `1e-10` probe-magnitude floor, the
`2**-44` turn unresolved floor, and the `5e-12` residual bound among
them — sorted by `name`; each row has exactly `name`, `type`, and
`value`, where `type` is `binary64`, `integer`, `rational`, or `literal`,
and `value` is respectively f64be hex, a base-10 integer string, `p/q` in
reduced base-10 integers, or the exact string literal. This object uses
the same JSON serialization rule as the scan array. `horizon_scan_sha256`
is SHA-256 of its exact bytes. The strict evidence validator rebuilds the
manifest digest and validates the embedded scan projection as above.
The canonical `FrameApplicabilityCertificate` and every retained frame
evidence row embed this exact object as `horizon_scan_manifest` together
with `horizon_scan_crossing_rows` and `horizon_scan_summary_rows`;
retaining the manifest or the crossing/summary projection only as digests
is forbidden. The frozen analytic census keeps its exact-rational
construction above unchanged: only the operational census is scan-based,
its enclosure width `1e-11 rad` and residual `5e-12` are its own new fixed
constants rather than a widening of any frozen-model constant, and both
models' closed root enclosures enter the unchanged pairing, lift, slab,
sign, and membership machinery.

The remaining horizon proof is also carried as embedded, authenticated
ledgers rather than aggregate counters. The canonical direction ledger has one
row with exactly:

```text
direction_id, source_kind, component_index, source_index,
transfer_role, transfer_nside, cirs_direction_sha256,
active_frequency_mask, active_frequency_count, direction_input_sha256
```

Point/native rows use `transfer_role="none"` and `transfer_nside=0`.
Transfer rows use reserved `component_index=0`, role `production` or
`diagnostic`, and their positive grid nside. Exact IDs are
`point:<component_index>:<source_index>`,
`native_healpix:<component_index>:<source_index>`, and
`transfer_quadrature:<role>:<nside>:<source_index>`. Rows are ordered point,
native, production transfer, then diagnostic transfer grids by increasing
nside; component source indices ascend in canonical-RING order for
point/native groups and in iso-Gauss ring-major order for transfer groups.

`active_frequency_mask` is an `F`-boolean array in run-frequency order. A
point/native element is true exactly when any resolved finite I/Q/U/V value at
that frequency compares unequal to zero; both signed zeros are inactive.
Transfer masks are `[true]*F`; the count equals the mask sum and lies in
`[1,F]`. The mask never reduces phase coverage: every retained direction is
still compared at all `F` frequencies. `cirs_direction_sha256` uses the
Section 14 array primitive with domain `radiosim.mmode-cirs-direction.v1`, role
`cirs_direction`, axes `["cartesian"]`, units `dimensionless`, dtype
`float64-be`, and shape `[3]` over the exact three-vector.

`direction_input_sha256` hashes domain
`radiosim.mmode-direction-input.v1` over an object with exactly
`schema_version`, `direction_id`, `source_kind`, `component_index`,
`source_index`, `transfer_role`, `transfer_nside`,
`cirs_direction_f64be`, `run_frequency_hz_f64be`, `active_frequency_mask`,
`resolved_stokes_iau_f64be`, and `integration_weight_f64be`. Binary64 values
are lowercase f64be strings. Point/native payload is the complete resolved
`[F,4]` IAU-Stokes array; transfer payload is empty. Integration weight is
binary64 one for a point, the retained pixel solid angle for a native row,
and the retained iso-Gauss node quadrature weight for a transfer row. `schema_version` is exactly
`radiosim.mmode-direction-input.v1`. Non-finite payload rejects before hashing. Coincident vectors
are never de-duplicated. The strict validator rebuilds exact ordered rows from
the authenticated input and requires equality.

The embedded `transfer_grid_catalog` contains rows with exactly
`transfer_grid_id`, `transfer_role`, `transfer_nside`,
`expected_direction_count`, `evaluated_direction_count`, and
`direction_id_ledger_sha256`. The production row is first; diagnostic rows
follow by increasing nside. IDs are `production:<nside>` or
`diagnostic:<nside>`; expected count is `12*nside**2`; evaluated count is the
corresponding contiguous direction-ledger slice length; and its ledger digest
is `D("radiosim.mmode-transfer-grid-direction-ids.v1",J(ordered_ids))`.
`transfer_grid_catalog_sha256` is
`D("radiosim.mmode-transfer-grid-catalog.v1",J(transfer_grid_catalog))`.
Source indices are contiguous iso-Gauss ring-major indices.
Repeated uses of one diagnostic nside share its entry, while distinct roles or
entries never collapse. Every horizon ledger joins this complete
grid-qualified direction order.

There is one root-pair row per direction, including a valid zero-root
trajectory, with exactly:

```text
direction_id, frozen_root_count, operational_root_count,
orientation_mismatch_count, pairs
```

A certified root is never collapsed to a midpoint for pairing, splitting, a
slab, or a limit. Each frozen and operational root retains its closed exact-
turn enclosure `[root_turn_lo,root_turn_hi]`, with canonical reduced `p/q`
endpoints and certified `rising` or `setting` orientation. The frozen
enclosure's derivative interval excludes zero throughout; the operational
enclosure's transversality is its probe-sign certificate. Distinct
enclosures for one model are disjoint and ordered. Operational bounds are
copied exactly from their owned scan row; frozen bounds are copied from the
analytic certificate. Replacing either enclosure by a representative point
rejects.

Pairing is separate by orientation. For a candidate
`F=[f_lo,f_hi]`, `O=[o_lo,o_hi]`, and integer lift `j in {-1,0,1}`, define the
exact worst-case displacement
`D_j=max(abs((o_lo+j)-f_hi),abs((o_hi+j)-f_lo))`. The lift is the unique `j`
minimizing `D_j`; a tie rejects. Enumerate the order-preserving cyclic
bijections. A bijection is admissible only when every pair has
`exact(tau)*D_j <= 1/50000 rad`; exactly one admissible bijection must exist
for each non-empty orientation class. Zero or multiple choices reject as
`mmode_horizon_unresolved`.

Every pair entry has exactly:

```text
pair_index, orientation, operational_turn_lift,
frozen_root_turn_lo, frozen_root_turn_hi,
operational_root_turn_lo, operational_root_turn_hi,
lifted_operational_root_turn_lo, lifted_operational_root_turn_hi,
worst_case_delta_turn, worst_case_delta_rad_f64be
```

Lifted bounds equal original bounds plus the integer lift.
`worst_case_delta_rad_f64be` is the least finite binary64 not smaller than
`exact(tau)*D_j`. Pairs sort by exact frozen lower then upper bound and receive
contiguous zero-based indices. Orientation mismatch is the sum of per-label
count differences. A zero-root row has an empty pair array; a passing frame
requires every root to pair with the same orientation.

For each pair, `R_hull=hull(F,O+j)` is the closed union of every possible
shortest root-to-root segment consistent with both enclosures. Projecting it onto the
exact one-turn `H_N` produces one closed piece, or two exactly when it crosses
the exact seam. A zero-measure slab is legal only when both enclosures are the
same singleton. Every slab row has exactly:

```text
direction_id, pair_index, orientation, operational_turn_lift,
worst_case_delta_turn, worst_case_delta_rad_f64be, wraps_seam, pieces
```

Each piece has exactly `piece_index`, `turn_lo`, and `turn_hi`, is sorted by
exact lower bound, and receives a contiguous index. Slab union, complement,
overlap, seam splitting, and measure use exact turns. There is one outside-
slab sign row per non-empty open complement piece, with exactly:

```text
direction_id, interval_index, turn_lo, turn_hi, midpoint_turn,
midpoint_rad_f64be, frozen_sign, operational_sign, match
```

`midpoint_turn` is the exact rational midpoint and must be strictly interior;
the radian field is its one-round view. Both signs are in `{-1,1}`. Finally,
the membership census evaluates one row for every direction and sample
centre, with exactly:

```text
direction_id, sample_index, sample_turn, alpha_rad_f64be,
frozen_visible, operational_visible, match
```

Sample indices are contiguous over the exact retained centre turns. Both
models independently evaluate strict numerator `>0`.
`horizon_membership_ledger_sha256` digests this complete canonical
per-sample array, but the retained evidence embeds it in the compact
per-direction mask form `horizon_membership_mask_rows`: one row per
direction with exactly `direction_id`, `sample_count`,
`frozen_visible_mask_hex`, `operational_visible_mask_hex`, and
`mismatch_count`, where each mask is the lowercase hex encoding of the
sample-ordered visibility bits, most significant bit first, zero-padded to
whole bytes. The expansion from masks to the per-sample rows is
deterministic — `sample_turn` and `alpha_rad_f64be` come from the one
retained grid object — so the strict validator expands the masks,
rebuilds the per-sample array bytes, and re-digests them against the
ledger digest cheaply; nothing is lost, only redundant per-row grid
repetition. Root-pair rows use
direction-ledger order; slab rows then pair index; sign rows exact lower-bound
order with contiguous interval indices; mask rows direction order.
No field in these five ledgers is nullable, and unknown or missing keys reject.

The embedded arrays are named `direction_rows`, `horizon_root_pair_rows`,
`horizon_slab_rows`, `horizon_sign_interval_rows`, and
`horizon_membership_mask_rows`. Each array is serialized alone with Section
14's
lexicographic object-key sorting, `ensure_ascii=true`, separators
`(',',':')`, UTF-8 encoding, and no whitespace or trailing newline. Their
respective SHA-256 fields are `direction_ledger_sha256`,
`horizon_root_pair_ledger_sha256`, `horizon_slab_ledger_sha256`,
`horizon_sign_interval_ledger_sha256`, and
`horizon_membership_ledger_sha256`, the last computed over the expanded
per-sample array as above. They use Section 14 `D` with, in that
order, domains `radiosim.mmode-direction-ledger.v1`,
`radiosim.mmode-horizon-root-pairs.v1`,
`radiosim.mmode-horizon-slabs.v1`,
`radiosim.mmode-horizon-sign-intervals.v1`, and
`radiosim.mmode-horizon-membership.v1`, with canonical array bytes as payload.
Retaining a digest without its array is
forbidden. The strict validator reconstructs every digest, joins every row by
the exact ordered direction IDs, recomputes the pair/slab/sign/membership rows
from the frozen analytic roots, authenticated operational leaves, and retained
sample grid, and requires exact ordered equality. It recomputes every root,
orientation, slab-measure, sign, and membership count or mismatch from these
rows; a self-consistent counter cannot substitute for ledger coverage.

After the Section 4.2 cyclic pairing, the union of both root sets splits the
cycle. Completeness makes each model's sign constant on every resulting open
interval. The strict signs are compared at its deterministic interior
midpoint outside the paired-root mismatch slabs; a wrapped slab and its
complement are checked as separate unwrapped pieces at the `H_N` seam. The
certificate records the number of such intervals, the number evaluated, and
the outside-slab mismatch count. Intervals within a slab may have opposite
signs and contribute to the recorded slab measure and full direct-cube error
instead of causing a logically impossible equality requirement.

For every point/native direction `d` and sample `k`, construct one common exact-
turn partition of `[lower_turn[k],upper_turn[k]]`. Its cuts are the two
retained exposure endpoints and every frozen root-enclosure and every
operational root-enclosure or guard-interval
lower or upper bound strictly inside — an operational crossing's ambiguous
region is the closed union of its enclosure and its flanking guards, and
every piece inside that union is a `root_enclosure`-class piece for the
error-disk rule. Sort and de-duplicate by exact rational
equality. Non-empty pieces are gap-free and overlap-free, receive contiguous
indices, and are shared by all frequencies, baselines, correlations, models,
and quadrature orders.

For each model, a piece interior is exactly `smooth_above`, `smooth_below`, or
`root_enclosure`. A singleton root is only a boundary. A `smooth_above` piece
uses that model's own integrand with independent 64- and 128-node
Gauss-Legendre rules; `smooth_below` is exact zero. No quadrature node is
evaluated in a positive-width `root_enclosure` piece.

Let `g^M_dfbc(u)` be the smooth no-horizon integrand and
`w_k=upper_turn[k]-lower_turn[k]`. On an ambiguous piece `I`, a certified
magnitude ceiling replaces any interval extension of the integrand: define

```text
G_abs = round_up(|payload| * prod(certified factor ceilings))
```

where `|payload|` is the contributor's resolved Stokes magnitude times its
integration weight and each remaining factor of the Section 6 kernel — every
Jones-term operator norm along the chain, the unit-magnitude fringe, and
the accepted `M`/`Q` factor magnitudes — enters through a design-recorded
certified upper bound; the beam factor's ceiling is the recorded peak of
the resolved normalized beam. Each ceiling is a `constant_rows` entry or an
authenticated per-run manifest value, never an ad-hoc estimate. Define
`epsilon^M_{dkfbcp}=round_up(((u_hi-u_lo)/w_k)*G_abs)`. The nominal
ambiguous contribution is exact complex zero and `epsilon` is its
absolute-error radius. Thus the unknown strict-visible subinterval lies
inside a certified disk — its width is at most the `1e-11 rad` operational
enclosure width, so the disk is orders below the Section 4.2 absolute
floor — and no hidden step enters a smooth rule. Inactive payloads have
zero nodes, contribution, and error. Error radii accumulate toward positive
infinity in canonical contributor/piece order into model-qualified
`[N,B,F,C]` cubes.

The embedded `direct_integrand_enclosure_manifest` has exactly
`schema_version`, `algorithm_id`, `implementation_files`, `constant_rows`,
`input_identity_sha256`, and `frame_matrix_sha256`; schema and algorithm are
`radiosim.mmode-direct-integrand-enclosure.v1`. Implementation rows are the
path-sorted raw-byte digests of `src/radiosim/core/mmode/frame.py`,
`src/radiosim/core/mmode/transfer.py`, and
`src/radiosim/core/mmode/solver.py`. Constant rows use Section 12.1's exact
name/type/value schema and include every outward arithmetic, complex-rectangle,
norm, root-cell, and accumulation constant. Its exact identity is
`direct_integrand_enclosure_sha256=D("radiosim.mmode-direct-integrand-enclosure.v1",J(direct_integrand_enclosure_manifest))`;
both object and digest are embedded in the
frame row, and every error preimage joins them.

The embedded `direct_split_rows` array has one row for every exact
direction/sample/frequency/baseline/correlation/piece combination, with exactly:

```text
direction_id, sample_index, frequency_index, baseline_index,
correlation_index, piece_index, turn_lo, turn_hi, payload_active,
frozen_piece_class, operational_piece_class,
frozen_gauss64_node_count, frozen_gauss128_node_count,
operational_gauss64_node_count, operational_gauss128_node_count,
frozen_gauss64_contribution_sha256, frozen_gauss128_contribution_sha256,
operational_gauss64_contribution_sha256,
operational_gauss128_contribution_sha256,
frozen_enclosure_error_f64be, operational_enclosure_error_f64be,
frozen_enclosure_error_preimage_sha256,
operational_enclosure_error_preimage_sha256
```

`piece_class` uses the three literals above. For an active payload, a node
count equals its named order only for `smooth_above`, and zero otherwise; every
count is zero for an inactive payload. Each contribution digest is
`D("radiosim.mmode-direct-piece-cell.v1",J(manifest))`, where the manifest has
exactly `schema_version`, `model`, `gauss_order`, `input_identity_sha256`,
`canonical_era_grid_sha256`, `direction_id`, `sample_index`,
`frequency_index`, `baseline_index`, `correlation_index`, `piece_index`,
`turn_lo`, `turn_hi`, `payload_active`, `piece_class`, `node_turns`,
`node_radians_f64be`, `weights_f64be`, `integrand_reim_f64be`,
`contribution_real_f64be`, and `contribution_imag_f64be`. Schema equals the
domain; model is `frozen` or `operational`; Gauss order is 64 or 128; array
lengths equal the retained node count; and integrand re/im length is twice it.
Empty-node cases authenticate empty arrays and
`contribution_real_f64be=contribution_imag_f64be=F64(0)`.

Each error-preimage digest is
`D("radiosim.mmode-direct-piece-error.v1",J(manifest))`, with exactly the same
keys `schema_version`, `model`, `input_identity_sha256`,
`canonical_era_grid_sha256`, `direction_id`, `sample_index`,
`frequency_index`, `baseline_index`, `correlation_index`, `piece_index`,
`turn_lo`, `turn_hi`, `payload_active`, and `piece_class`, followed by
`direct_integrand_enclosure_sha256`, `integrand_rectangle_f64be`, and
`enclosure_error_f64be`. No Gauss-order field occurs because the enclosure
error is model-qualified and quadrature-order-independent. The rectangle has
exactly four outward `F64` bounds in order
`[real_lo,real_hi,imag_lo,imag_hi]`, with each lower bound no greater than its
upper bound, and the error equals the row's finite non-negative value. The
manifest `schema_version` is exactly
`radiosim.mmode-direct-piece-error.v1`. For an active `root_enclosure` piece,
the rectangle is exactly `[-G_abs,G_abs,-G_abs,G_abs]` from the certified
magnitude ceiling above and the error is the certified Section 12 value. For
`smooth_above`, `smooth_below`, or any inactive piece, the rectangle is exactly
`[F64(0),F64(0),F64(0),F64(0)]` and `enclosure_error_f64be=F64(0)`; no other
zero spelling or unused carried enclosure is permitted. The strict validator
independently rebuilds every preimage.

Rows are ordered by direct-direction order, sample, frequency, baseline,
correlation, then piece, all ascending. For each direction/sample the exact
piece bounds reproduce the independently reconstructed shared partition. The
array is embedded and
`direct_split_ledger_sha256=D("radiosim.mmode-direct-split-ledger.v1",J(rows))`;
retaining only
its digest is forbidden. With `D_direct=D_point+D_native` and `P_dk` the
reconstructed piece count:

```text
expected_direct_exposure_split_count = D_direct*N
expected_direct_split_row_count = B*C*F*sum_d,sum_k(P_dk)
expected_<model>_gauss<q>_node_count =
    sum_rows(<model>_gauss<q>_node_count)
```

Every evaluated count equals its expected count. Exposure count is the exact
distinct `(direction_id,sample_index)` projection, not a counter. Native terms
retain original RING centre, solid angle, and per-frequency payload.
Production and diagnostic transfer rows receive horizon certification but do
not enter direct split rows.

Four model-qualified direct cubes `F64`, `F128`, `O64`, and `O128` are reduced
from this ledger only after piecewise evaluation. All have exact shape
`[N,B,F,4]`, finite complex128 cells, and canonical axis order. The frozen and
operational enclosure-error cubes `EF` and `EO` have the same shape and finite
non-negative float64 cells. Their exact identities are:

```text
frozen_gauss64_cube_sha256 = A(radiosim.mmode-direct-cube.v1,
  frozen_gauss64,[time,baseline,frequency,correlation],Jy,F64)
frozen_gauss128_cube_sha256 = A(radiosim.mmode-direct-cube.v1,
  frozen_gauss128,[time,baseline,frequency,correlation],Jy,F128)
operational_gauss64_cube_sha256 = A(radiosim.mmode-direct-cube.v1,
  operational_gauss64,[time,baseline,frequency,correlation],Jy,O64)
operational_gauss128_cube_sha256 = A(radiosim.mmode-direct-cube.v1,
  operational_gauss128,[time,baseline,frequency,correlation],Jy,O128)
frozen_enclosure_error_cube_sha256 = A(radiosim.mmode-direct-root-error.v1,
  frozen_enclosure_error,[time,baseline,frequency,correlation],Jy,EF)
operational_enclosure_error_cube_sha256 = A(
  radiosim.mmode-direct-root-error.v1,operational_enclosure_error,
  [time,baseline,frequency,correlation],Jy,EO)
```

The literals are passed exactly as strings to Section 14's `A`; the four
direct arrays use `complex128-be` and the error arrays `float64-be`. With
`K=4*N*B*F`:

```text
S_Q = max(1 Jy,max(abs(F128)),max(abs(O128)))
Q_F = max(abs(F128-F64))
Q_O = max(abs(O128-O64))
Q = max(Q_F,Q_O)
Q_limit = 1e-11*S_Q
```

Both compared-cell counts are `K` and `Q<=Q_limit` is mandatory. Frame
applicability uses the certified upper bound
`U=abs(F128-O128)+EF+EO`, scale
`S_frame=max(1 Jy,max(abs(O128)+EO))`, maximum
`max(U) <= 5e-5*S_frame+1e-10 Jy`, and
`norm(U)/max(norm(O128),sqrt(K)*1 Jy) <= 5e-5`.
Unit-beam/single-mode controls additionally use the exact piecewise
exponential antiderivative and meet `5e-12`. A missing root bound, piece,
node, cell, error enclosure, or fixed predicate rejects; adding nodes or
widening a tolerance is forbidden.

### 12.2 Required oracle families

Minimum oracles are:

1. **ERA/DFT:** analytic `exp(i*m*alpha)` for positive, negative, and zero `m`;
   exact sign/normalization and exposure `sinc`; no duplicated endpoint;
   closure and UTC round trip; `N=17,f=1`, exact
   `h_N^+-h_N^-=1/1`, full-width shared-edge identity, and complete retained-
   edge containment in closed `[h_N^-,h_N^+]`; plus one nontrivial binary64
   fraction whose exact IEEE ratio is reconstructed.
2. **Frame:** one-time ICRS-to-CIRS position and tangent transport, rigid group
   composition, pressure-zero operational comparison, a ceiling-excluded
   root-free cell, a scan-crossing refinement to the fixed enclosure width,
   a probe-floor tangency rejection, and the fixed budget.
3. **Scalar harmonics:** individual `Y_lm`, reality, point delta, constant
   HEALPix sky, RING/NEST equality, and point+map coefficient additivity.
4. **Polarization:** individual spin `+2/-2` modes, the spin reality relation,
   HEALPix/CMB-to-IAU U conversion, pure Q/U/V, and the exact `D`/SCI-006
   east-X/circular signs.
5. **Transfer:** analytic unit beam/zero baseline, one non-zero baseline fringe,
   heterogeneous beams, every correlation, frequency scaling, and
   `B_lm(alpha)=B_lm(0)*exp(i*m*alpha)`.
6. **Direct agreement:** small unpolarized and polarized point, HEALPix, and
   hybrid skies through the common frozen-frame direct oracle and the exact
   horizon-split procedure in Section 12.1; the every-run two-tier gate
   (quadrature-shell fidelity at `1e-8`, deficit convergence and
   disclosure) and certificate-digest/count identity; retained local-shell,
   increasing-`lmax`, increasing-`mmax`, and higher-quadrature diagnostics plus
   wrong-sign controls.
7. **Stationarity/rejections:** nonconstant gain, wrong time variant, Nyquist,
   collapsed binary64 exposure edges, quadrature, morphology, missing tangent
   metadata, IERS range, and frame budget, all with exact types/codes/messages.
8. **Execution:** NumPy reference, JAX/Dask behavior, worker invariance, at
   least three memory budgets, deterministic block order, and conservative
   memory estimates.
9. **Results:** in-memory, summary, HDF5, UVFITS, and MS round trips with phase,
   feed, correlation, time, solver, and fingerprint metadata.
10. **Characterization:** every new family, unchanged direct pins, two dispatch
    classes where applicable, the retained harvested-cell observation
    sets (remote artifacts by later admission), and release scans
    that continue to say `SCI-004` is ROADMAP until closure.

The analytic complex128 DFT/single-mode residual is at most `5e-12`. The
physical direct/convergence bound is Section 7.3's scale-aware predicate.
Wrong Fourier sign, wrong V bridge, omitted tangent transport, and omitted
east-X permutation are retained non-vacuity controls and must miss by more than
ten times their corresponding passing residual.

## 13. Phase-separated writable authority

A path not listed for a phase requires a bounded design correction and a new
independent review before edit. The lists do not authorize work before their
dependency gates.

### 13.1 D — this design candidate

- `docs/development/sci004_mmode_design.md` (new)
- `docs/index.rst`
- `PostTier8RemediationPlan.md` (WP-9, Q5, dependency, and ledger wording only)

`D0` introduced this memo with exactly the three paths above; the operative
`D` is Section 13.7's latest independently accepted, header-recorded
design-gate commit. Every later `design_sha` is exactly the operative `D`
frozen for its phase under Section 14.0, never a phase-local memo tip or a
search result.

### 13.2 Dependency-gate tips

Two clean programme tips admit dependencies that necessarily land after `D`
or an earlier SCI-004 phase. They are ancestry points, not SCI-004 source or
evidence commits.

For M1, `D` and the independently accepted WP-7 CPU acceptance commit
`7e5f469c835c1137a3a3a870d27c5d9f5e8f3520` must be ancestors of globally
clean `G1`; ancestry is inclusive, so `G1` may equal the later dependency
commit. The first-parent range `D..G1` contains no merge. The upstream
verifier cannot run against `G1` itself: it requires clean
`HEAD == --descendant` and re-diffs the WP-7-frozen `pixi.toml`/`pixi.lock`
bytes at its accepted source against the descendant, and the independently
accepted `v0.4.0` release commit (`ae2650f`) changed `pixi.toml` after that
freeze, so the protected-source rule rejects every descendant of `ae2650f`
— every legally constructible `G1` — by design. The accepted v0.4.0
release review adjudicated that ending of the WP-7 live-tree freeze
invariant; the WP-7 acceptance itself remains authentic at its own frozen
chain. The M1 gate therefore replays the certificate at the frozen
historical replay descendant
`c6a5ce90ae3160150b1699f97b45bb693d4ed886` — the `descendant_commit`
recorded inside the retained, already-authenticated SCI-005 Stage-1
dependency artifact `docs/development/sci005_stage1_wp7_dependency.json` —
and proves the `G1` ancestry facts from Git objects directly.

At `HEAD == G1`, before any SCI-004 red byte exists, the gate creates a
fresh temporary detached worktree at exactly that replay descendant,
requires `git status --porcelain=v1 --untracked-files=all` to be empty
there, authenticates the invoked tool blob from that tip against the
certificate's `cpu_evidence_tool_sha256`, and runs exactly:

```text
pixi run python tools/wp7_perf001_cpu_evidence.py verify-accepted \
  --acceptance-commit 7e5f469c835c1137a3a3a870d27c5d9f5e8f3520 \
  --descendant c6a5ce90ae3160150b1699f97b45bb693d4ed886
```

It must exit zero with empty stderr and emit the upstream canonical one-line
JSON certificate, including one final LF, byte-identical to the retained
SCI-005 Stage-1 dependency artifact, with schema
`radiosim.perf001.cpu_acceptance_certificate.v1` and exactly:

```text
schema_version, acceptance_commit, evidence_commit, generating_source_sha,
descendant_commit, artifact_path, artifact_sha256,
cpu_evidence_tool_sha256, production_record_validator_sha256,
production_harness_sha256, pixi_manifest_sha256, pixi_lock_sha256,
evidence_diff_paths, acceptance_diff_paths, verdict, passed
```

The values require the named accepted WP-7 `A`, `descendant_commit` equal to
the frozen replay descendant, `verdict=="CPU_ACCEPTED_P_E_HARDWARE_GATED"`,
and `passed==true`. The worktree and its temporary directory are removed on
success or failure. `R1^==G1`. R1 retains those exact stdout bytes at
`docs/development/sci004_mmode_phase1_wp7_dependency.json` and freezes in
`tests/unit/test_sci004_phase1_dependency.py` exactly
`APPROVED_SCI004_D_SHA`, `APPROVED_SCI004_G1_SHA`,
`APPROVED_WP7_CPU_A_SHA`, and `APPROVED_WP7_REPLAY_DESCENDANT_SHA`. No
later phase may change those constants or the retained certificate.

For M3, accepted SCI-004 `A2` and the independently accepted SCI-005 Stage-2
`A2` must both be ancestors of globally clean `G3`; ancestry is inclusive and
the first-parent range from SCI-004 `A2` to `G3` contains no merge. At
`HEAD==G3`, before any M3 red byte exists, run exactly:

```text
pixi run python tools/sci005_stage2_acceptance.py verify \
  --acceptance-commit <40hex-SCI005-A2> --descendant <40hex-G3>
```

It must exit zero with empty stderr and emit one canonical UTF-8 JSON line,
including one final LF, with schema
`radiosim.sci005.stage-acceptance-certificate.v1` and exactly:

```text
schema_version, stage, acceptance_commit_sha, acceptance_artifact_path,
acceptance_artifact_sha256, evidence_commit_sha, evidence_artifact_path,
evidence_artifact_sha256, source_sha, verdict, successor_unlocks
```

The line requires `stage==2`, the named accepted SCI-005 `A2`,
`verdict=="ACCEPT"`, and `successor_unlocks==["SCI004.M3","SCI005.U2"]`.
`R3^==G3` unless a Section 13.7 accepted correction stars the
`G3 -> R3` edge, in which case `R3` directly parents the operative
correction commit and the header enumerates the interval — as the
accepted-capability-characterization-envelope correction does. R3
retains the exact stdout bytes at
`docs/development/sci004_mmode_phase3_sci005_dependency.json` and freezes in
`tests/unit/test_sci004_phase3_dependency.py` exactly
`APPROVED_SCI004_D_SHA`, `APPROVED_SCI004_G3_SHA`, and
`APPROVED_SCI005_STAGE2_A_SHA`.

Every dependency validator creates a fresh temporary detached worktree at
its replay anchor — the frozen WP-7 replay descendant for M1, exact `G3`
for M3 — requires `git status --porcelain=v1
--untracked-files=all` to be empty, authenticates the invoked tool blob from
that tip, runs the exact command there, and compares stdout byte-for-byte,
including the final LF, with the retained file. The M1 validator
additionally proves, from Git objects at the caller's checkout, that `D` and
the WP-7 acceptance commit are ancestors of `G1` and that the first-parent
range `D..G1` contains no merge. The M3 validator additionally
creates a clean detached worktree at exact `R3`, runs the Stage-2 verifier with
`--descendant <R3>`, and requires the stdout bytes to be identical to the
retained G3 line; the verifier output is descendant-independent while both
ancestry checks must pass. Worktrees and temporary directories are removed on
success or failure. A dirty tip, merge, wrong ancestor, changed tool, nonzero
exit, stderr byte, differing stdout, missing LF, or cleanup failure rejects.

A Section 13.7 correction accepted after a gate has run does not re-run
that gate: the commit that was the operative `D` when the gate ran is the
**gate anchor** — a commit identity, unrelated to Section 4.1's
coordinate-frame anchor record — the retained certificate and ancestry
facts stay bound to it, and the dependency validator authenticates the gate anchor as an
ancestor of the operative `D` through the header-enumerated chain instead
of requiring the operative `D` to precede `G1`. A correction that reopens
an already-committed red slice makes that red commit a header-recorded
`superseded red slice` interval commit; the governed re-cut directly
parents the correction landing, regenerates the phase red-failure record
under Section 13.7's disposal rule, and rebinds the frozen constants that
name the operative `D`.

Across `D..G1`, the accepted `D` memo blob, its SCI-004 index entry, the
`Fix.md` SCI-004 row, and the PostTier WP-9 subsection/ledger cells remain
byte-identical. Across SCI-004 `A2..G3`, every SCI-004-owned byte at `A2`,
including prior artifacts and validators, remains byte-identical. Separately
accepted dependency commits may otherwise change shared production paths, and
those exact `G1`/`G3` bytes become the following red baseline. Neither gate
tip contains a new SCI-004 red, source, evidence, or acceptance byte.

### 13.3 Phase 1 — full-sky strategy, time/frame, and scalar core

`R1` may write only the following red oracles and retained failure machinery:

- `tests/unit/test_io/test_sci004_config.py` (new)
- `tests/unit/test_core/test_sci004_era_grid.py` (new)
- `tests/unit/test_core/test_sci004_frame.py` (new)
- `tests/unit/test_core/test_sci004_scalar_harmonics.py` (new)
- `tests/unit/test_core/test_sci004_transfer.py` (new)
- `tests/unit/test_simulator/test_sci004_strategy.py` (new)
- `tests/integration/test_sci004_mmode.py` (new)
- `tests/characterization/test_tier7_current_behavior.py`
- `tests/characterization/test_tier6_current_behavior.py` (the D15
  directory-listing pin only: widen the pinned `tests/integration/` listing
  by exactly `test_sci004_mmode.py` and record the addition in its
  docstring)
- `tests/unit/test_tier7_jones_acceptance.py`
- `tests/unit/test_sci004_phase1_dependency.py` (new immutable dependency and
  design binding validator)
- `tools/sci004_mmode_phase1_red.py` (new)
- `docs/development/sci004_mmode_phase1_wp7_dependency.json` (new exact
  upstream certificate line)
- `docs/development/sci004_mmode_phase1_red_failures.json` (new)
- `tests/unit/test_sci004_phase1_red_failures.py` (new validator)

`S1` may write only production, pre-artifact validators/tools, and active
documentation in these paths:

- `src/radiosim/api/simulator.py`
- `src/radiosim/simulator/base.py`
- `src/radiosim/simulator/rime.py`
- `src/radiosim/simulator/mmode.py` (new)
- `src/radiosim/simulator/__init__.py`
- `src/radiosim/core/hybrid.py`
- `src/radiosim/core/time_grid.py`
- `src/radiosim/core/result.py`
- `src/radiosim/core/runtime_config.py`
- `src/radiosim/core/mmode/__init__.py` (new)
- `src/radiosim/core/mmode/types.py` (new)
- `src/radiosim/core/mmode/time.py` (new)
- `src/radiosim/core/mmode/frame.py` (new)
- `src/radiosim/core/mmode/harmonics.py` (new)
- `src/radiosim/core/mmode/sky.py` (new)
- `src/radiosim/core/mmode/transfer.py` (new)
- `src/radiosim/core/mmode/solver.py` (new)
- `src/radiosim/io/config.py`
- `src/radiosim/io/config_resolution.py`
- `src/radiosim/io/hdf5.py`
- `src/radiosim/io/summary_json.py`
- `tools/sci004_mmode_phase1_evidence.py` (new)
- `tests/unit/test_sci004_phase1_evidence.py` (new strict validator)
- `tools/sci004_mmode_phase1_acceptance.py` (new)
- `tests/unit/test_sci004_phase1_acceptance.py` (new strict validator)
- `tests/unit/test_simulator/test_perf001_capabilities.py` (the
  registered-simulator inventory pin only: widen the expected mapping to
  exactly `{"mmode": False, "rime": False}`)
- `tests/unit/test_io/test_config.py` (the `ExecutionConfig` field-set pin
  only: add exactly `mmode` to the expected field names)
- `tests/characterization/test_tier6_current_behavior.py` (the
  `ExecutionConfig` field-set characterization only: add exactly `mmode` to
  the expected field names)
- `docs/user_guide/configuration.rst`
- `docs/api/algorithms.rst`
- `docs/api/result.rst`
- `docs/migration_guide.md`
- `docs/changelog.rst`

`E1` may write only:

- `docs/development/sci004_mmode_phase1_evidence.json` (new)
- `docs/development/sci004_mmode_phase1_evidence.md` (new reproduction record)
- `tests/unit/test_sci004_phase1_evidence.py` (exact approved-S and artifact
  digest constants only)

`A1` may write only:

- `docs/development/sci004_mmode_phase1_acceptance.json` (new)
- `tests/unit/test_sci004_phase1_acceptance.py` (exact E and acceptance-artifact
  digest constants only)
- `docs/development/sci004_mmode_design.md` (append-only acceptance note)
- `PostTier8RemediationPlan.md` (WP-9 ledger only)

M1 may add no polarized capability claim and no fingerprint pin. It keeps
`supports_polarization=False`, rejects non-zero Q/U/V, and must prove the
wrapper leaves every direct path byte-identical before M2 starts.

### 13.4 Phase 2 — full Stokes, SkyModel components, backend, and memory

`R2` may write only:

- `tests/unit/test_core/test_sci004_polarization.py` (new)
- `tests/unit/test_core/test_sci004_sky_harmonics.py` (new)
- `tests/unit/test_core/test_sci004_transfer.py`
- `tests/unit/test_core/test_sci004_direct_convergence.py` (new)
- `tests/unit/test_backends/test_sci004_backend_parity.py` (new)
- `tests/unit/test_simulator/test_sci004_memory.py` (new)
- `tests/integration/test_sci004_mmode.py`
- `tests/characterization/test_tier7_current_behavior.py`
- `tests/characterization/test_tier6_current_behavior.py` (the D15
  directory-listing pin only: widen the pinned `tests/performance/` listing
  by exactly `test_sci004_mmode.py` and record the addition in its
  docstring)
- `tests/performance/test_sci004_mmode.py` (new, non-gating)
- `tools/sci004_mmode_phase2_red.py` (new)
- `docs/development/sci004_mmode_phase2_red_failures.json` (new)
- `tests/unit/test_sci004_phase2_red_failures.py` (new validator)

`S2` may write only:

- `src/radiosim/simulator/mmode.py`
- `src/radiosim/core/mmode/types.py`
- `src/radiosim/core/mmode/frame.py`
- `src/radiosim/core/mmode/harmonics.py`
- `src/radiosim/core/mmode/sky.py`
- `src/radiosim/core/mmode/transfer.py`
- `src/radiosim/core/mmode/solver.py`
- `src/radiosim/core/polarization.py`
- `src/radiosim/core/sky/containers/point.py`
- `src/radiosim/core/sky/containers/healpix.py`
- `src/radiosim/core/sky/containers/model.py`
- `src/radiosim/core/sky/containers/__init__.py`
- `src/radiosim/core/sky/support/point_builder.py`
- `src/radiosim/core/sky/loaders/_healpix_builder.py`
- `src/radiosim/core/sky/loaders/bbs.py`
- `src/radiosim/core/sky/loaders/pyradiosky.py`
- `src/radiosim/core/sky/loaders/skyh5_multifile.py`
- `src/radiosim/core/sky/combine/engine.py`
- `src/radiosim/core/sky/combine/healpix.py`
- `src/radiosim/core/sky/operations/factories.py`
- `src/radiosim/core/sky/operations/operations.py`
- `src/radiosim/core/beam/runtime.py`
- `src/radiosim/core/jones/directions.py`
- `src/radiosim/core/result.py`
- `src/radiosim/core/runtime_config.py` (one optional
  `tangent_polarization_frame` field on the resolved sky-source inputs
  only)
- `tests/unit/test_simulator/test_sci004_strategy.py` (delete the
  single duplicate scalar-capability test
  `test_mmode_simulator_reports_scalar_only_support_in_m1` only —
  Section 9 names the Tier-7 characterization assertion as the one
  authoritative capability pin licensed to flip at accepted M2; the
  three M1 non-zero-Stokes rejection nodes bound into the accepted M1
  evidence stay byte-untouched)
- `src/radiosim/simulator/__init__.py` (the m-mode strategy description
  doctest and the scalar-M1 module prose only: update the phase-scalar
  wording to the accepted M2 truth)
- `src/radiosim/core/mmode/__init__.py` (the scalar-M1 module prose
  only, likewise)
- `src/radiosim/io/config.py`
- `src/radiosim/io/config_resolution.py`
- `src/radiosim/benchmarks/harness.py`
- `src/radiosim/benchmarks/__init__.py`
- `tools/sci004_mmode_phase2_evidence.py` (new)
- `tests/unit/test_sci004_phase2_evidence.py` (new strict validator)
- `tools/sci004_mmode_phase2_acceptance.py` (new)
- `tests/unit/test_sci004_phase2_acceptance.py` (new strict validator)
- `docs/user_guide/sky_models.rst`
- `docs/user_guide/backends.rst`
- `docs/user_guide/jones_matrices.rst`
- `docs/api/sky.rst`
- `docs/api/algorithms.rst`
- `docs/migration_guide.md`
- `docs/changelog.rst`

`E2` may write only:

- `docs/development/sci004_mmode_phase2_evidence.json` (new)
- `docs/development/sci004_mmode_phase2_evidence.md` (new reproduction record)
- `tests/unit/test_sci004_phase2_evidence.py` (exact approved-S and artifact
  digest constants only)

`A2` may write only:

- `docs/development/sci004_mmode_phase2_acceptance.json` (new)
- `tests/unit/test_sci004_phase2_acceptance.py` (exact E and acceptance-artifact
  digest constants only)
- `docs/development/sci004_mmode_design.md` (append-only acceptance note)
- `PostTier8RemediationPlan.md` (WP-9 ledger only)

### 13.5 Phase 3 — outputs, characterization, and retained records

`R3` may write only:

- `tests/unit/test_io/test_hdf5_result.py`
- `tests/unit/test_io/test_result_summary.py`
- `tests/unit/test_io/test_standard_visibility.py`
- `tests/unit/test_io/test_uvfits.py`
- `tests/unit/test_io/test_measurement_set.py`
- `tests/characterization/test_sci004_mmode.py` (new)
- `tests/unit/test_tier8_release_acceptance.py`
- `tests/unit/test_sci004_phase3_dependency.py` (new immutable dependency and
  design binding validator)
- `tools/sci004_mmode_phase3_red.py` (new)
- `docs/development/sci004_mmode_phase3_sci005_dependency.json` (new exact
  upstream certificate line)
- `docs/development/sci004_mmode_phase3_red_failures.json` (new)
- `tests/unit/test_sci004_phase3_red_failures.py` (new validator)
- `tests/performance/test_sci004_mmode.py` (the pinned fixture-product,
  workload-row-key-set, and claims-array
  literals only: align them to the amended Section
  11 — red-oracle edits, so they belong to an `R`
  commit under this section's own rule; the closed `R2` list's own
  grant of this path belonged to the accepted M2 phase and conveys
  nothing to `R3`)

Correction #24's post-source red-delta R3 is the narrower replacement for that
superseded slice: it directly parents the correction landing and may change
exactly
`docs/development/sci004_mmode_phase3_post_source_red_failures.json`,
`tests/unit/test_io/test_hdf5_result.py`,
`tests/unit/test_sci004_phase3_dependency.py`,
`tests/unit/test_sci004_phase3_red_failures.py`, and
`tools/sci004_mmode_phase3_red.py`. It retains the SCI-005 dependency and
historical red JSON records byte-for-byte, adds only the ruled six-case
supplemental red-delta JSON, freezes `APPROVED_SCI004_D_SHA` to that landing,
and performs only the HDF5-oracle, supplemental-generation, chain/rebind, and
retained-record authentication work recorded in the header.

`S3` may write only:

- `src/radiosim/core/result.py`
- `src/radiosim/io/hdf5.py`
- `src/radiosim/io/summary_json.py`
- `src/radiosim/io/standard_visibility.py`
- `src/radiosim/io/uvfits.py`
- `src/radiosim/io/measurement_set.py`
- `tests/characterization/test_h5py_output_contract.py`
- `tests/characterization/test_pyuvdata_321_output_contract.py`
- `tests/characterization/test_pyuvdata_321_polarization_contract.py`
- `tests/characterization/test_tier6_current_behavior.py`
- `tests/characterization/test_tier7_current_behavior.py`
- `tests/characterization/test_tier8_current_behavior.py`
- `tools/sci004_mmode_phase3_evidence.py` (new)
- `tests/unit/test_sci004_phase3_evidence.py` (new strict validator)
- `tools/sci004_mmode_phase3_acceptance.py` (new)
- `tests/unit/test_sci004_phase3_acceptance.py` (new strict validator)
- `docs/user_guide/configuration_support.rst`
- `docs/user_guide/backends.rst`
- `docs/api/io.rst`
- `output/benchmarks/reference/README.md` (new)
- `.gitignore` (the negation rules un-ignoring the granted
  `output/benchmarks/reference/README.md` and the
  `output/benchmarks/reference/sci004/` record directory only, in the
  established perf001 block form)
- `src/radiosim/core/mmode/solver.py` (the two public-path guards only:
  reject a HEALPix-bearing payload with `mmode_public_components` and a
  non-scalar resolved beam system with `mmode_public_beam`, each with
  its exact Section 8 message before any solver work; no other solver
  change)
- `src/radiosim/benchmarks/__init__.py` (the `SCI004_FIXTURE_IDS`
  constant, its dependent fixture literals, and the
  `SCI004_WORKLOAD_KEYS` and claims-array literals only: align the
  Section 11
  performance-record fixture product and row schema to the
  accepted-capability
  envelope and the honest backend axis; the closed `S2` list's own
  grant of
  this path belonged to the accepted M2 phase and conveys nothing to
  `S3`)

Correction #24's replacement S3 directly parents that fresh R3 and may change
exactly `src/radiosim/io/hdf5.py`,
`tools/sci004_mmode_phase3_evidence.py`, and
`tests/unit/test_sci004_phase3_evidence.py`; every other byte from the
superseded S3 flows through unchanged.

Correction #25's fresh fingerprint-delta R3 is the replacement for correction
#24's now-superseded R3. It directly parents correction #25 and may change
exactly:

- `docs/development/sci004_mmode_phase3_fingerprint_post_source_red_failures.json`
  (new canonical supplement only)
- `tests/characterization/test_sci004_mmode.py` (exactly the four newly
  collected nodes
  `test_characterization_input_preimage_is_retained_and_reconstructible`,
  `test_characterization_input_identity_is_equal_under_distinct_layout_roots`,
  `test_distinct_layout_roots_preserve_scientific_and_cube_identities`, and
  `test_characterization_input_identity_changes_for_semantic_instrument_content`;
  the first two are expected-red and the latter two are mandatory green
  controls; plus only the header-ruled domain-discriminated edit to existing
  parametrized node `test_every_new_family_records_its_six_section_11_parts`)
- `tests/unit/test_sci004_phase3_dependency.py` (correction-#25 chain,
  authority, and rejected-attempt authentication only)
- `tests/unit/test_sci004_phase3_red_failures.py` (three-record expected-red
  disjoint union, the exact passing-control rows/replay, new supplement, and
  raw-byte authentication only)
- `tools/sci004_mmode_phase3_red.py` (dedicated atomic no-overwrite generation
  and validation of the new supplement only)

It retains the dependency certificate, historical phase-3 red record, and
correction-#24 post-source red record byte-for-byte. Its supplemental schema,
status, binding, nodes, oracle-patch digest, and two prior-record digests are
the exact header values; its tool's historical and correction-#24 generation
modes remain fail-closed and cannot overwrite either prior record.
Within the characterization path, fixture support means only constants
`FAMILY_RECORD_V1_KEYS`, `FAMILY_RECORD_V2_KEYS`, and
`FINGERPRINT_RED_LAYOUT_BYTES`, and helpers
`_family_result_and_phase_input_manifest`,
`_characterization_record_for_active_domain`, `_relocated_family_records`, and
`_semantic_layout_mutation`; they may contain only the construction, canonical
preimage, and exact-branch logic those five collected nodes consume. No other
collected node may be added, removed, renamed, skipped, xfailed, weakened, or
have an assertion deleted; no other existing fixture/helper may change. The R3
authority validator parses collection and the parent-relative test diff to
enforce that exact inventory rather than treating an arbitrary test change as
“fixture support.”

Correction #29 requires a later validator/oracle/evidence-binding R3 directly
over D29. That R3 may change exactly
`tests/characterization/test_sci004_mmode.py`,
`tests/unit/test_sci004_phase3_dependency.py`,
`tests/unit/test_sci004_phase3_red_failures.py`,
`tools/sci004_mmode_phase3_evidence.py`, and
`tests/unit/test_sci004_phase3_evidence.py`. Its characterization change owns
only the exact transition-aware v1-at-R3/v2-at-S3 oracle; its dependency
change freezes exact D29 and authenticates the complete fixed chain; its red
validator authenticates the preserved partition, retained records, and
future-R3 delta/replay; and its evidence tool plus strict validator implement
and directly test only the separation between the D29 current-envelope design
and the immutable D24/D25 record-design bindings. No sixth path, new red
record, red-generator mode, production change, or unrelated S3 hunk is
authorized. The future R3 must be a single-parent non-merge direct child of
D29. Eventual replacement S3 directly parents that future R3 and retains the
exact nine-path authority below only after the two overlapping evidence paths
are proved to retain nonempty S3-owned deltas.

Correction #29's eventual replacement S3 directly parents that future R3 and
may change or delete exactly:

- `src/radiosim/core/result.py`
- `tools/sci004_mmode_phase3_evidence.py`
- `tests/unit/test_sci004_phase3_evidence.py`
- `tools/sci004_mmode_phase3_acceptance.py`
- `tests/unit/test_sci004_phase3_acceptance.py`
- `docs/development/sci004_mmode_phase3_evidence.json` (delete rejected E3
  working-tree draft only)
- `docs/development/sci004_mmode_phase3_evidence.md` (delete rejected E3
  reproduction draft only)
- `output/benchmarks/reference/sci004/20260825T122048Z-macbook-pro-2.json`
  (delete rejected host-bound draft only)
- `docs/development/sci004_mmode_phase3_acceptance.json` (delete canonical A3
  REJECT working-tree copy only)

Within the two validator paths it returns exactly the four E3 and two A3
approved bindings to literal `None` while implementing the v2 manifest,
hostile mutation checks, three-red-record joins, and current-attempt ancestry
selection. It may not edit the fresh red oracle or any red record. At its tree
both fixed phase artifact paths are absent, the SCI-004 performance directory
is empty, and all six bindings are null; the tools' synthetic strict fixtures
remain producible in that state. The deleted rejected bytes remain immutable
and authenticated at their exact header-recorded Git commits.

`E3` may write only:

- `docs/development/sci004_mmode_phase3_evidence.json` (new)
- `docs/development/sci004_mmode_phase3_evidence.md` (new reproduction record)
- `tests/unit/test_sci004_phase3_evidence.py` (exact approved-S and artifact
  digest constants only)
- `output/benchmarks/reference/sci004/<UTC>-<host>.json` (one new record)

`A3` may write only:

- `docs/development/sci004_mmode_phase3_acceptance.json` (new)
- `tests/unit/test_sci004_phase3_acceptance.py` (exact E and
  acceptance-artifact digest constants only)
- `docs/development/sci004_mmode_design.md` (append-only acceptance note)
- `PostTier8RemediationPlan.md` (WP-9 ledger only)
- `docs/changelog.rst`
- `docs/migration_guide.md` if review finds the pre-v1 wording incomplete

If an output red test proves a production writer outside this list must change,
M3 pauses for design correction. No `S` commit may edit a red oracle or its
record. A proved-defective oracle requires a bounded design successor and a
fresh `R`; it is not repaired in the production commit. No phase may edit
workflow tolerances, historical records, or a previous phase artifact.

For every phase, the evidence and acceptance generators and validators first
land in `S`, so a clean exact `S` already contains the bytes that will produce
and validate its successors. At `S`, validator constants
`APPROVED_SOURCE_SHA`, `APPROVED_EVIDENCE_SHA`, and the corresponding artifact
SHA-256 are null sentinels; schema/unit fixtures run, and the validator asserts
that the not-yet-authorized retained artifact is absent. `E` may change only
the evidence validator's approved-S/artifact digest constants, never its logic.
`A` may change only the acceptance validator's approved-E/artifact digest
constants, never its logic. The phase reproduction Markdown records the exact
generator argv, Pixi environment, clean `S`, artifact path, and raw artifact
SHA-256. These constrained constant hunks and reproduction bytes are evidence,
not production-source changes.

### 13.6 Whole-row closure successor

The final whole-row closure, and only it, may additionally update:

- `Fix.md` (`SCI-004` ROADMAP -> DONE with dated exact evidence)
- `README.md` and `CLAUDE.md` where the accepted support surface requires it
- `docs/api/algorithms.rst` and `docs/user_guide/configuration_support.rst`

No phase may edit `core/contraction.py`, weaken a tolerance, add a dependency
on `simulators/`, make a benchmark gate CI, or regenerate an unrelated pin.
`pixi.toml`, `pixi.lock`, and `pyproject.toml` are not writable because the
design uses already required Astropy, SciPy, and healpy. A proved dependency
gap requires a design successor before those files change.

### 13.7 Bounded design corrections, supersession, and disposal

`D0 = 978fef6ddd885355dd06f1deeb04aa2927626d71` is the commit that
introduced this memo; its parent-relative diff touches exactly the three
Section 13.1 design-authority paths. The **operative design commit** `D` is
the latest independently accepted, header-recorded design-gate commit: the
Phase-0 acceptance landing, or a later accepted bounded correction. Every
frozen `design_sha` binding names the operative `D` current at its own
phase's `R`.

A bounded design correction is drafted as edits to
`docs/development/sci004_mmode_design.md` alone — plus
`PostTier8RemediationPlan.md` WP-9/Q5/dependency/ledger wording when the
correction changes a fact that ledger states. Its exact pre-landing file
bytes and parent-relative diff are pinned by SHA-256 and receive two fresh
independent reviews (physics/governance and computational). On dual
`ACCEPT` it lands unchanged as one single-parent non-merge commit touching
only those paths, with a dated header record naming what it supersedes, the
pinned SHA-256 values, both verdicts, and any phase slice it reopens. That
landing commit becomes the operative `D`. A correction may reopen a red,
source, or evidence slice; it may not rewrite history, edit a retained
accepted artifact, weaken a tolerance, or accept a production phase.

When an accepted correction intervenes at a Section 14.4 edge, the exact
direct-parent edge it displaces is replaced by a starred edge whose
interval commits this memo's header must enumerate exhaustively by SHA.
Every interval commit is a single-parent non-merge of exactly one
header-recorded kind — status-prose, superseded design, superseded red
slice, superseded implementation, superseded evidence, rejected acceptance, or
post-acceptance repair — touching
exactly the paths its kind allows, mirroring the accepted SCI-005
Section 8.3 machinery; the reopened phase's `R` then directly parents the
operative correction commit, and when accepted corrections intervene
between two already-defined phase commits without reopening the earlier
one — between a landed `R` and its `S`, an `S` and its `E`, or an `E`
and its `A` — the later phase commit likewise directly parents the
operative correction commit, with the header enumerating the interval
identically. A commit the header does not name invalidates
the edge. A `post-acceptance repair` commit repairs an already-accepted
phase's tracked tooling or validator defect discovered at or after that
phase's acceptance; it may touch only that phase's Section 13 tool and
validator test paths — never production source, never a retained
artifact, never this memo — and the next phase's red validator
authenticates each enumerated repair by full SHA and exact touched
paths. The correction that enumerates a repair must, in its own dual
review, diff that repair against its pre-repair blob and confirm that no
assertion, tolerance, or accepted-value set was relaxed, widened, or
removed without a compensating equal-or-stronger check; a repair that
cannot be so confirmed is not enumerable under this kind and requires
the applicable supersession-and-regeneration path instead. A
`rejected acceptance` commit is the canonical failed-closed record of one
independent `A` attempt. It may touch only that phase's fixed acceptance JSON
and the exact approved-E/acceptance-artifact constant assignments in its
Section 13 acceptance validator; its sole parent is the rejected evidence
candidate; its artifact verdict is `REJECT`; and its header entry freezes the
full commit and parent SHAs, exact touched paths, artifact raw SHA-256, external
review-contribution raw SHA-256, reviewer identity, and concrete blocker. The
artifact and commit are immutable. The next governed R or correction
authenticates them from Git objects. This kind unlocks no successor, cannot
change ROADMAP to DONE, and cannot authorize closure C. Its fixed current-tree
artifact may be deleted only by a header-authorized replacement S restoring
the phase's absent-output/null-sentinel state; the immutable Git blob remains
the historical authority.

An accepted
phase acceptance commit inside the `D0 -> D` range is
not a chain commit and needs no interval kind: its memo diff is exactly
its own phase's Section 13 append-only acceptance note, authenticated by
its own
phase machinery under the Section 14.0 rule that tools authenticate the
operative `D` blob plus the separately authorized `A` diffs. A
correction's header record lands with one final sentence appended after
its dual `ACCEPT` — the pinned pre-landing file and diff SHA-256 values
and both verdicts; those pins name the reviewed bytes, which the
appended sentence itself necessarily postdates and which are never
separately committed, so the accepted header text is their only
authority. The companion ledger row follows the same two stages: the
reviewed bytes state the round's review as pending, and the verdict
wording lands only in the same commit as the completed header record.

A recorded reopening whose corrective target is measured not to exist —
the reopened slice already satisfies the corrected design, verified
empirically and recorded with the measurement — is discharged by a later
accepted correction's record without a re-cut: the slice remains the
phase's live commit, its recorded interval kind reverts, and its
retained record stands unchanged.
Disposal follows the accepted SCI-005 Section 7.5 rule exactly: when a
header-recorded correction supersedes an evidence commit, the reopened
slice's `S` restores that phase's Section 13 `S`-state — it deletes the
superseded phase evidence artifact, its reproduction record, and every
header-named host-bound performance record; it returns all approved evidence
constants to the null sentinels. When the rejected evidence has a canonical
`rejected acceptance` child, that same replacement S also deletes the fixed
acceptance JSON and returns both acceptance constants to null. Thus the
regenerating `E` and later `A` run against absent fixed paths, an empty
phase-performance directory where the phase requires it, and six null
sentinels. The regenerating `E` and `A` run under the unchanged rule against
that state. This disposal is authorized only for an artifact the memo header
records as superseded; the `A` that would have accepted it returned `REJECT`,
so removing it is disposal of a rejected draft, not replacement of a record.
The same rule governs a superseded phase red-failure record:
the governed re-cut deletes and regenerates it, since the record was never
accepted — **unless the reopened phase's `S` already exists**, in which
case the record's previously observed nodes can no longer be observed red in
the operative tree and the record is not regenerated: the rebind-only re-cut
retains the record's last genuinely observed bytes, and the strict red-record
validator authenticates the record's `design_sha` as a header-enumerated chain
commit from whose tree the observations were genuinely made, connected to the
operative `D` through the chain — never as a licence to fabricate an
`expected-red-confirmed` observation against a tree where nothing is red.

After such disposal, an active validator identifies a retry introduction only
relative to the active approved parent. For evidence parent `S`, compute the
ordered first-parent ancestry path with
`git rev-list --first-parent --ancestry-path --reverse S..HEAD`; current `E` is
its first commit,
which must be S's sole-parent direct child and must satisfy the complete
Section 13 E diff, blob, and constant bindings. For evidence parent `E`, current
`A` is analogously the first commit from that command with `E..HEAD`, E's
sole-parent direct child,
and must satisfy the complete A diff, blob, verdict, and constant bindings.
The range must be an ancestry path with no merge. Zero or multiple admissible
children, a non-first-parent candidate, or any mismatch rejects. Searching all
history for additions, choosing the newest or first add of the fixed path, or
selecting a commit before the active approved parent is forbidden. Every
header-recorded rejected predecessor remains separately authenticated by exact
commit/path/blob identity and is never eligible for the current-attempt role.

A correction may separately prove a **new post-source red delta** against the
committed `S` it supersedes. That route is valid only when the header records an
unshadowed replay of distinct new oracle nodes and grants a fresh `R` to author
those nodes plus a separate canonical supplemental record. The supplement
binds the new operative `D`, the exact superseded `S` as
`pre_fix_source_sha`, the exact oracle-patch diff, and the immutable historical
record digest; it never appends to, copies cases into, overwrites, or
regenerates the historical record. Section 14.1's disjoint-union rule makes
every historical or supplemental node appear exactly once, and the replacement
`S` may change only production plus its separately granted source validators —
never either red oracle. This exception is phase- and correction-specific; a
prose-only replay or an unrecorded failing test is insufficient. An accepted
artifact is immutable and no commit may touch one.

## 14. Evidence schema and commit succession

All phase records are UTF-8 canonical JSON. Serialization sorts object keys
lexicographically, uses separators `,` and `:`, emits no whitespace or trailing
newline, and uses RFC 8785/ECMAScript shortest-round-trip number serialization
for every finite JSON number; alternate spellings such as `1.0` and `1e0` are
not canonical record bytes. NaN and Infinity are forbidden. Every object
rejects unknown or missing keys. Git identities are
lower-case 40-hex strings and SHA-256 identities are lower-case 64-hex strings.
Counts are non-negative JSON integers; booleans are not counts; residuals and
tolerances are finite non-negative numbers. A nullable field always has an
adjacent non-empty `<field>_reason`; scientific order is represented by JSON
arrays, never sets or maps.

### 14.0 Canonical digest vocabulary and identity joins

The accepted design identity is the operative commit `D` defined in
Section 13.7. R1's dependency validator,
`tests/unit/test_sci004_phase2_red_failures.py` at R2, and R3's dependency
validator each freeze the exact assignment
`APPROVED_SCI004_D_SHA="<40hex-D>"`, naming the operative `D` current at
that phase's `R`. The later bindings byte-match the R1 binding unless a
Section 13.7 accepted correction intervened, in which case the later
binding names the newer operative `D` and its validator authenticates the
header-enumerated correction chain between the two bindings. Every current
SCI-004 red, evidence, acceptance, dependency, and PERF-001 generator and
validator reads the phase-appropriate frozen binding for its own envelope.
An immutable retained record instead keeps the historical `design_sha` and
`pre_fix_source_sha` under which its failures were genuinely observed. This
rule includes both later-superseded post-source supplements: correction #24
keeps exact D24 and correction #25 keeps exact D25 even after D29 becomes the
operative design. The strict validator requires each historical field to equal
its exact header-enumerated ancestor, authenticates that ancestor's kind and
connection to the current `D`, and byte-compares the record to its governed Git
blob. No generator may use this exception to emit a fresh
`expected-red-confirmed` record after `S` exists, and no historical record is
rewritten merely because a later correction changes the current binding.

For correction #29's future R3, the new dependency validator freezes exact D29
as `APPROVED_SCI004_D_SHA`; the old R3 validator remains immutable with its
historically correct D26 binding. The later replacement-S3 evidence envelope
therefore uses `design_sha = D29`; the correction-#24 record independently
uses `design_sha = 4d507bf1333ccaa4c8beec3815370ba0f6043bb2`; the
correction-#25 record independently uses
`design_sha = ca3c37171aaaeec175b5ad72d324957762303853`; and the envelope
uses `red_commit_sha =` the future post-D29
validator/oracle/evidence-binding R3. A design SHA may not stand in for that
red-commit SHA, `_design_sha()` may not stand in for either historical binding,
and no ancestry or newest-memo search may substitute an older or newer design
tip for any exact identity.

Before trusting that value, each tool resolves the bound commit with
`^{commit}` peeling, requires it to be a single-parent non-merge ancestor of
the phase's R/S/E/A commit, and reads the Section 13.1 paths with
`git ls-tree`/`git cat-file` from Git objects rather than the checkout. The
memo must be introduced at `D0` with the exact three-path parent-relative
diff Section 13.7 records; each commit in the header-enumerated chain from
`D0` to the operative `D` must match its recorded kind and touch exactly
the paths its kind allows; and the operative `D` memo blob, the exact
SCI-004 index entry, and the PostTier WP-9/Q5/dependency/ledger hunks must
match the independently accepted correction record. The verifier also
authenticates the operative-`D`-tree `Fix.md` SCI-004 row as ROADMAP without
claiming it was writable at `D`. Later append-only A notes do not change
`design_sha`; tools continue to authenticate the operative `D` blob plus the
separately authorized A diffs. Selecting the first history commit with
matching prose/blob, using the current memo tip, accepting an annotated tag,
or trusting a 40-hex string without these Git-object/diff/ancestry checks is
forbidden.

The M1 G1 and M3 G3 validators additionally require the same accepted D object
and bindings while applying Section 13.2's immutable-byte checks. Thus a
dependency tip cannot silently substitute a revised design even when its
external certificate is valid.

`F64(x)` is the lowercase 16-hex-character encoding of
`struct.pack(">d",x)`. Scientific binary64 values inside identity manifests
are `F64` strings, never JSON decimals. Every topology-bearing `*_turn`,
`*_ratio`, or turn-array value is a Section 3.1 normalized `p/q` string.
`J(x)` is the canonical UTF-8 JSON above with `ensure_ascii=true`.
`U64(n)` is unsigned eight-byte big-endian. For non-empty ASCII domain `d` and
payload bytes `p`, define

```text
D(d,p) = SHA256(d || NUL || U64(len(p)) || p)
```

For a numeric array, `A(d,role,axes,units,array)` gives `D` the payload
`U64(len(header)) || header || U64(len(data)) || data`, where `header=J` of
exactly `axis_order`, `dtype`, `role`, `shape`, and `units`, and `data` is
normalized C-order bytes. Allowed dtypes are `float32-be`, `float64-be`,
`complex64-be`, `complex128-be`, `int64-be`, `uint64-be`, and `bool-u8`;
for each C-order complex element, its big-endian real component is immediately
followed by its big-endian imaginary component. Arrays must be finite. Role, rank, shape,
axes, units, endian, and dtype are therefore authenticated.
`A` rejects unless `d`, `role`, `units`, and every axis name are non-empty
ASCII strings without NUL; axis names are unique;
`len(axis_order)==array.ndim==len(shape)`; every shape entry is a non-negative
exact integer; and `product(shape)*dtype_itemsize==len(data)`. These checks
precede hashing.

The exact-turn ERA grid and Section 12 horizon/direction/direct ledgers retain
their narrower preimages. Raw file hashes, including Pixi files, IERS bytes,
artifacts, stdout, and stderr, are SHA-256 of exact raw bytes and are not
wrapped in `D`. `git_tree_sha256` is
`D("radiosim.sci004.git-tree.v1",stdout)`, where `stdout` is the exact output
of `git ls-tree -r -z --full-tree <source_sha>`. Visibility and scientific
cube hashes not otherwise specialized use `A` with domain
`radiosim.mmode-visibility-cube.v1`, role `visibility_cube`, axes
`["time","baseline","frequency","correlation"]`, units `Jy`, and the
declared result dtype. Packed values and harmonic coefficients have no generic
fallback; they use the exact composite and array domains later in this section.

The following manifests are exact:

- `site_manifest` has exactly `schema_version`, `longitude_deg_f64be`,
  `latitude_deg_f64be`, `height_m_f64be`, and three-element
  `itrs_xyz_m_f64be`; the schema literal equals the domain, the position length
  is three, and WGS84 geodetic-to-ITRS is recomputed.
  `site_sha256=D("radiosim.mmode-site.v1",J(site_manifest))`;
- `utc_manifest` and `ut1_manifest` have exactly `schema_version`,
  `scale`, `axis_order`, `shape`, `center_jd1_f64be`, `center_jd2_f64be`,
  `lower_jd1_f64be`, `lower_jd2_f64be`, `upper_jd1_f64be`, and
  `upper_jd2_f64be`. Schema equals its domain; scale is `utc` or `ut1`;
  `axis_order=["sample"]`; shape is `[N]`; every array length is `N`.
  `utc_sha256=D("radiosim.mmode-utc-grid.v1",J(utc_manifest))` and
  `ut1_sha256=D("radiosim.mmode-ut1-grid.v1",J(ut1_manifest))`;
- `frame_matrix_manifest` has exactly `schema_version`, `era0_rad_f64be`,
  row-major `rpom0_f64be`, row-major `cirs_to_itrs_anchor_f64be`,
  `local_east_itrs_f64be`, `local_north_itrs_f64be`, and
  `local_up_itrs_f64be`; its schema is
  `radiosim.mmode-frame-matrices.v1`, both matrix arrays have exactly nine
  elements, and every local-basis array has exactly three. The three basis
  arrays are recomputed from `site_manifest`, must be orthonormal with the
  Section 4.1 handedness, and `cirs_to_itrs_anchor_f64be` is recomputed as
  `RPOM0 R3(era0)`. `frame_matrix_sha256` is
  `D("radiosim.mmode-frame-matrices.v1",J(frame_matrix_manifest))`; and
- `convention_identity_sha256` uses domain
  `radiosim.mmode-conventions.v1` over an embedded object with exact literals
  for execution, exact-turn/radian-grid, frozen frame, harmonic and tangent
  frames, Stokes bridge, transfer catalogue, quadrature/truncation policy,
  execution policy, field/spin order, DFT sign, exposure window, and strict
  horizon. Its exact keys and values are:

```text
schema_version=radiosim.mmode-conventions.v1
execution=radiosim.mmode-forward.v1
era_turn_grid=radiosim.mmode-era-turn-grid.v1
era_radian_grid=radiosim.mmode-era-grid.v2
frozen_frame=radiosim.frozen-cirs-rigid-era.v1
harmonics=radiosim.shaw-polarized-harmonics.v1
tangent_frame=radiosim.sky-tangent-polarization.v1
stokes_bridge=radiosim.stokes-ne-theta-phi.v1
transfer_catalog=radiosim.mmode-transfer-grid-catalog.v1
quadrature_policy=iso-gauss-ring-production-plus-qcheck.v1
truncation_policy=complete-frozen-direct-plus-local-shells.v1
execution_policy=host_harmonics_backend_native_dense_v1
field_order=[I,+2,-2,V]
spin_order=[0,2,-2,0]
dft_forward=bar_v_m=(1/N)*sum_k(bar_V_k*exp(-i*2*pi*m*u_k))
dft_inverse=bar_V_k=sum_m(bar_v_m*exp(+i*2*pi*m*u_k))
exposure_window=exact-turn-top-hat-sinc.v1
horizon_predicate=sin_altitude_strictly_greater_than_zero
```

No listed convention is implementation-selected.

`certificate_sha256` is exactly

```text
D("radiosim.mmode-frame-certificate.v1",
  J(frame_certificate_row excluding exactly
    fixture_id, certificate_sha256, and pass))
```

No nested input, frame, grid, or cube manifest contains the certificate digest
or a result digest that contains it. Every `frame_certificate_sha256` field is
an alias of this exact value, not another algorithm.

The phase envelope's `source_identities` has exactly `git_tree_sha256`,
`pixi_manifest_sha256`, `pixi_lock_sha256`, `convention_identity_sha256`,
`fixture_input_rows`, and `input_identity_set_sha256`. A fixture-input row has
exactly `fixture_id`, `input_identity_manifest`, and
`input_identity_sha256`; rows are unique and UTF-8 fixture-ID sorted. The
manifest has exactly:

```text
schema_version, site_manifest, site_sha256, iers_table_sha256,
canonical_era_turn_grid, canonical_era_turn_grid_sha256,
canonical_era_grid, canonical_era_grid_sha256,
utc_manifest, utc_sha256, ut1_manifest, ut1_sha256,
mmode_dimensions,
antenna_rows, baseline_rows, frequency_rows,
receptor_rows, correlation_rows, beam_rows, sky_component_rows,
direction_input_rows, jones_term_rows, transfer_grid_catalog,
precision, result_dtype, convention_identity_sha256
```

Its schema is `radiosim.mmode-input-identity.v1`. An antenna row has exactly
`antenna_index`, `name`, and three-element `itrs_xyz_m_f64be`; a baseline row
has `baseline_index`, `antenna1_index`, `antenna2_index`, and three-element
`itrs_vector_m_f64be`; a frequency row has `frequency_index`,
`center_hz_f64be`, and `width_hz_f64be`; `mmode_dimensions` has exactly
`sidereal_samples`, `lmax`, `mmax`, `quadrature_nside`, `lcheck`, `mcheck`, and
`qcheck`, all exact non-negative integers satisfying Sections 3 and 7. A
receptor row has exactly `antenna_index`, `basis`, ordered two-element `labels`,
`feed_rotation_rad_f64be`, and
`feed_angle_rad_f64be`; and a correlation row has
`correlation_index`, `p_label`, and `q_label`, with exactly four rows in
canonical matrix order. A beam row has exactly `beam_index`,
`assigned_antenna_indices`, `class_qualname`, `electric_field_basis`,
`normalization`, `parameter_identity_manifest`, and
`parameter_identity_sha256`; beam rows are class/parameter groups in first-
assigned-antenna order and partition the antenna indices exactly once. A
sky-component row has exactly `component_index`, `representation`,
`coordinate_frame`, `polarization_frame`, `polarization_frame_sha256`,
`morphology_identity_manifest`, `morphology_identity_sha256`, and ordered
`direction_input_sha256s`. A Jones term row has exactly `term_index`,
`term_name`, `class_qualname`, `parameter_identity_manifest`,
`parameter_identity_sha256`, and `time_stationarity`. Parameter, morphology,
and polarization identities are not opaque. A parameter or morphology
identity manifest has exactly `schema_version`, `identity_kind`, `scalar_rows`,
and `array_rows`. A scalar row has exactly `name`, `type`, and `value`; rows are
name-sorted; `type` is `boolean`, `integer`, `binary64`, `rational`, or
`literal`; and `value` is respectively a JSON boolean, canonical base-10
integer string, `F64`, normalized `p/q`, or exact string. An array row has
exactly `name`, `domain`, `role`, `axis_order`, `units`, `dtype`, `shape`, and
`sha256`, is name-sorted, and reconstructs with `A` from the resolved source
payload. Parameter and morphology manifests have schema/domain
`radiosim.mmode-parameter-identity.v1` and
`radiosim.mmode-morphology-identity.v1`; `identity_kind` is the exact resolved
class/kind literal. `polarization_frame` is the exact six-key Section 5.1
object when any resolved Q/U is non-zero, and otherwise the non-null literal
`not_applicable_no_linear_polarization`.
`polarization_frame_sha256` is
`D("radiosim.sky-tangent-polarization.v1",J(polarization_frame))` over that
exact value. Indices are contiguous and arrays retain scientific
order. Each embedded identity manifest must reconstruct its adjacent digest.
Receptor rows join antenna order exactly; `labels` and
`feed_angle_rad_f64be` each have length two, while
`feed_rotation_rad_f64be` is one canonical `F64` value.

A `direction_input_rows` entry has exactly `direction_input_manifest` and
`direction_input_sha256`. It embeds the complete Section 12 direction-input
preimage; the adjacent digest must equal its Section 12 `D` identity. Rows use
the exact direction-ledger order, their digest projection equals every
component's ordered `direction_input_sha256s`, and their complete digest array
equals every direction row. The top-level direction sequence is the exact concatenation of
component rows followed by production and diagnostic transfer rows and equals
the frame direction ledger. `precision` is `standard` or `low`; result dtype
is `complex128` or `complex64` as Section 9 permits.

The manifest excludes fixture labels, paths, backend/device, workers, memory,
timings, outputs, certificates, and result cubes. Its digest domain is
`radiosim.mmode-input-identity.v1`, so
`input_identity_sha256=D(domain,J(input_identity_manifest))`; the complete
ordered row-array digest is
`D("radiosim.sci004-phase-input-set.v1",J(fixture_input_rows))`. The
fixture-input row set equals exactly
the union of every non-rejection result fixture ID in that phase: no orphan,
duplicate, or missing row is allowed.

Correction #25's M3 characterization input is a separate retained projection
with exact domain and schema literal
`radiosim.sci004.characterization-input.v2`. Its
`characterization_input_manifest` has exactly:

```text
schema_version, family_id, fixture_id,
phase_input_identity_manifest, phase_input_identity_sha256,
instrument_manifest, instrument_sha256,
receptor_manifest, receptor_sha256,
loaded_beam_manifest, beam_loaded_fingerprint,
correlations, polarization_basis, frequencies_hz_f64be
```

`family_id == fixture_id` and both values are the exact Section 11 family name.
`phase_input_identity_manifest` is the complete value-bearing
`radiosim.mmode-input-identity.v1` object from the unique same-fixture
`source_identities.fixture_input_rows` entry, byte-for-byte under canonical
JSON; its adjacent digest must independently recompute under that v1 domain and
equal both the nested `phase_input_identity_sha256` and the adjacent phase row.
Filesystem venue is non-semantic. The embedded manifest therefore continues to
exclude fixture paths. Any recursive key whose normalized name is `path`, ends
in `_path`, `_paths`, `_file`, `_files`, `_dir`, `_directory`, `_root`, or
`_uri`, or otherwise denotes a filesystem locator is invalid; a scalar under
such a key is invalid whether absolute, relative, symlink-resolved,
temporary-root, or checkout-root. A string value under a semantic key is not
rejected merely because it contains `/` or resembles a relative token; the
exact closed schemas and projection equality decide its meaning. Layout
content is represented by the embedded site, antenna, baseline, and beam rows,
not by a locator.

`instrument_manifest` has exactly `schema_version`, `site_manifest`,
`antenna_rows`, and `baseline_rows`, copied from the embedded phase input; its
schema/domain is `radiosim.sci004.characterization-instrument.v1` and
`instrument_sha256=D(domain,J(instrument_manifest))`. `receptor_manifest` has
exactly `schema_version` and `receptor_rows`, copied from the embedded phase
input; its schema/domain is
`radiosim.sci004.characterization-receptors.v1` and
`receptor_sha256=D(domain,J(receptor_manifest))`. `loaded_beam_manifest` has
exactly `schema_version` and `beam_rows`, including every parameter manifest,
content-array identity, and exact antenna partition from the embedded phase
input; its schema/domain is
`radiosim.sci004.characterization-loaded-beam.v1` and
`beam_loaded_fingerprint=D(domain,J(loaded_beam_manifest))`. Each complete
submanifest is compared byte-for-byte to its phase-input projection before its
adjacent digest is reconstructed, so no identity is a bare writer-provided
claim. `correlations` is the ordered four-label projection of
`correlation_rows`; `polarization_basis` is the unique basis implied by those
rows and the ordered receptor labels; and `frequencies_hz_f64be` is the ordered
projection of every `frequency_rows.center_hz_f64be`. Each projection is
recomputed and compared exactly.

The characterization digest is

```text
input_identity_sha256 =
    D("radiosim.sci004.characterization-input.v2",
      J(characterization_input_manifest))
```

The complete manifest is retained in each M3 fingerprint row. It contains no
cube, solver-snapshot, scientific-result, or characterization digest, so those
identities cannot feed back into their input. The row separately retains the
complete solver snapshot and its digest, the complete result-derived
characterization-time manifest and its distinct digest, the cube identity, the
result scientific identity, and the observation-set entry under Section 14.2.
The strict evidence validator reimplements `J`, `D`, the v1 phase-input digest,
each content identity, and all projections locally; importing
`mmode_characterization_record` or another production helper is forbidden as
the independent oracle.

`characterization_time_manifest` is not the phase manifest's
`canonical_era_grid` object and its adjacent digest is not the phase
`canonical_era_grid_sha256`. It retains the current result-derived
`radiosim.sci004.characterization-time.v1` preimage with exactly
`schema_version`, `axis_order`, `shape`, `interval_semantics`,
`start_time_iso`, `center_jd1_f64be`, `center_jd2_f64be`, and
`integration_time_seconds_f64be`. Its values require, respectively, that exact
schema literal, `["sample"]`, the one-element sample-count array,
`half_open_sample_centers`, the normalized UTC anchor, and the complete ordered
binary64 arrays. `era_utc_grid_sha256` is exactly
`D("radiosim.sci004.characterization-time.v1",J(characterization_time_manifest))`.
For all four existing families it must remain
`558758efff6d46ea559705bf6b6ab2245bf948a6d6792ed722e048e1ef41d877`;
the separate embedded phase `canonical_era_grid_sha256` remains
`f865447ee34816c865e42d9202f26d388a6072c3f6be068973d9b9510ae357aa`.

The validator joins the two distinct grid surfaces field by field. The row
shape equals `mmode_dimensions.sidereal_samples`; both center-JD arrays equal
the phase `utc_manifest` center arrays byte-for-byte; the normalized
`start_time_iso`, the phase exact center turns, and the locked IERS table are
independently remapped and must reproduce those same UTC centers; and each
integration width is independently reconstructed in seconds from the phase
UTC lower/upper two-part JD rows and must equal the retained ordered width.
It also reconstructs the phase turn/radian grids and their own adjacent
digests. Thus neither digest substitutes for the other, and neither is a bare
writer-provided claim.

The validator recomputes `site_sha256`, `utc_sha256`, and `ut1_sha256` from
the three embedded manifests, reconstructs every exact-turn invariant and
component array identity from `canonical_era_turn_grid`, and then recomputes
both canonical grid digests from the two embedded grid objects before
computing the input identity. This applies independently in M1, M2, and M3; a
later phase never relies on an M1 time-grid row. A time-grid
row for the same fixture must contain byte-identical UTC/UT1 manifests and
digests. A frame row must contain a byte-identical `site_manifest`, and its
embedded `frame_matrix_manifest` must recompute from that site, the unit-named
polar-motion values, retained `era0`, and the fixed Section 4 formulas. A bare
dynamic site, time, or frame digest without its exact value-bearing manifest is
invalid.

Every non-rejection result fixture joins exactly one fixture-input row. Every
row-level input digest equals it. Truncation rows join a certificate by
`(fixture_id,frame_certificate_sha256)` and require identical input, exact grid,
site, frame, IERS, and frozen-direct identities. M2 retains its own
`frame_certificate_cases`; a phase cannot reference an M1 orphan. A red
`fixture_identity_sha256` is domain `radiosim.sci004-red-fixture.v1` over
exactly `phase`, `fixture_id`, `requirement_id`, `test_nodeid`,
`pre_fix_source_sha`, and `invalid_config_raw_sha256`; the last field hashes
the exact invalid fixture bytes rather than metadata alone.

All remaining specialized identities are fixed as follows. A packed-layout
row embeds `block_rows`, `packed_values_reim_f64be`, and
`roundtrip_reim_f64be`. `block_rows` is the exact Section 5.3 signed-m-major,
field-minor table. Each re/im array has length `2*packed_value_count` and
encodes each successive complex value as real then imaginary `F64` strings.
`block_table_sha256` is
`D("radiosim.mmode-packed-block-table.v1",J(block_rows))`. After decoding,
both buffers use `A` with domain `radiosim.mmode-packed-values.v1`, role
`packed_harmonic_values`, axes `["packed_value"]`, units `dimensionless`, and
dtype `complex128-be`. The round trip unpacks and repacks in exact signed-m,
field, ascending-`l` order. `pass` requires byte-identical buffers and equal
`packed_values_sha256` and `roundtrip_sha256` identities; padding is forbidden.
`scientific_sha256` is
`D("radiosim.mmode-scientific-result.v1",J(manifest))`, where the manifest has
exactly `input_identity_sha256`, `canonical_era_grid_sha256`,
`convention_identity_sha256`, `solver_snapshot_sha256`, `time_sha256`,
`feed_sha256`, `correlation_sha256`, and `cube_sha256`. Before/after RIME,
strategy, expected/observed, backend, direct, Gauss, and output visibility-cube
fields use the common array primitive with role `visibility_cube`, except
Section 12's model/order-qualified four direct/error cubes.
`solver_snapshot_sha256` is
`D("radiosim.mmode-solver-snapshot.v1",J(snapshot))` over the exact tagged
snapshot key set:

```text
solver, sky_representation, convention, execution_path,
components, component_element_counts,
time_grid_convention, frame_model, harmonic_convention,
sidereal_samples, lmax, mmax, quadrature_nside,
quadrature_policy, truncation_policy,
tangent_polarization_frame, stokes_v_basis_bridge,
iers_table_sha256, frame_certificate_sha256, transform_execution_policy
```

Unknown or missing keys reject; written and read solver hashes use the same
rule. `time_sha256` is
`D("radiosim.mmode-result-time.v1",J(manifest))`, where the manifest has
exactly `schema_version`, `canonical_era_turn_grid_sha256`,
`canonical_era_grid_sha256`, `utc_sha256`, `ut1_sha256`, and
`integration_time_seconds_sha256`, and schema equals the domain. The width
digest uses Section 3.1's exact `A` rule and is independently reconstructed
from the embedded UTC edge arrays. `feed_sha256` is
`D("radiosim.mmode-result-feeds.v1",J(manifest))` over exactly
`schema_version` and the input identity's ordered `receptor_rows`;
`correlation_sha256` analogously uses domain
`radiosim.mmode-result-correlations.v1` over exactly `schema_version` and the
ordered `correlation_rows`. Schema always equals its domain. The validator
reconstructs these manifests from the same embedded fixture input and result
snapshot; a writer-provided bare digest is not trusted. M1
`rime_before_sha256` and `rime_after_sha256` are visibility-cube identities,
not solver-snapshot identities.
`input_frame_sha256` and `transported_frame_sha256` use `A` with domain
`radiosim.mmode-tangent-frame-array.v1`, axes
`["direction","basis_axis","cartesian"]`, units `dimensionless`, dtype
`float64-be`, and roles `input_tangent_frame` and
`transported_tangent_frame`, respectively. Point/HEALPix/hybrid coefficient
and expected-sum identities are composite. Their value array uses `A` with
domain `radiosim.mmode-harmonic-coefficient-array.v1`, axes
`["frequency","packed_value"]`, units `Jy`, dtype `complex128-be`, and role
`harmonic_coefficient_values`. The final coefficient manifest has exactly
`schema_version`, `block_table_sha256`, `frequency_rows_sha256`, and
`values_array_sha256`; its schema/domain is
`radiosim.mmode-harmonic-coefficients.v1`.
`frequency_rows_sha256=D("radiosim.mmode-frequency-rows.v1",J(frequency_rows))`,
and the named coefficient identity is `D` over that composite manifest. Thus a
buffer with a different packed table or frequency grid cannot share an
identity merely because its shape and bytes match.
`schedule_sha256` is
`D("radiosim.sci004.block-schedule.v1",J(schedule_rows))` over the exact
Section 11 rows. A named
digest not covered by this paragraph, a narrower rule, or a raw-file rule is a
schema error; implementations may not invent a preimage.

The sole discriminated-format exception is the Section 12.1 scan-row form
carried by `horizon_scan_crossing_rows` and reconstructed for the
scan-ledger digest. Its `classification` is the null reason for the
three root fields, so adjacent reason keys are forbidden there. Non-null
root-turn bounds use canonical rationals. Its `*_f64be` values are exact
strings rather than JSON numbers; every value must decode to a finite
binary64, and every decoded residual must be non-negative. For the
scan-ledger digest the validator reconstructs the Section 12.1 listed field
order; the containing phase record itself still uses this section's
lexicographically sorted object keys.

### 14.1 Red-failure records

For phase `i`, `docs/development/sci004_mmode_phase{i}_red_failures.json` has
schema literal `radiosim.sci004.mmode-phase{i}-red-failures.v1` and exactly
this top-level key set:

```text
schema_version, phase, status, generated_at_utc, design_sha,
pre_fix_source_sha, red_commit_sha, red_commit_sha_reason,
protected_source_clean, authorized_red_paths, environment, cases,
commands, claims_not_licensed
```

`phase` is exactly `M1`, `M2`, or `M3`; `status` is
`expected-red-confirmed`; `red_commit_sha` is JSON null with reason
`self-reference: E binds the containing R commit`. The red generator
hashes every protected path outside Section 13's phase-local `R` list before
and after execution, requires those hashes unchanged, and records the sorted
authorized diff in `authorized_red_paths`. Thus an uncommitted red artifact
does not falsely claim a globally clean tree.
`authorized_red_paths` is a sorted unique string array, `protected_source_clean`
must be true, and red `environment` uses the exact environment object in
Section 14.2. Red `claims_not_licensed` is a sorted unique non-empty string
array and must include production, acceptance, fingerprint, and performance
claims.

Every non-empty `cases` row has exactly:

```text
case_id, requirement_id, test_nodeid, invalid_config_raw_sha256,
fixture_identity_sha256,
expected_failure_kind, expected_failure_pattern, command_index, exit_code,
observed_outcome, observed_exception_type, observed_message,
stdout_sha256, stderr_sha256,
fixture_defect_excluded_by, red_failure_confirmed
```

`expected_failure_kind` and `observed_outcome`
are one of `assertion`, `exception`, `import`, `missing-symbol`, or `schema`;
`observed_exception_type` is always a non-empty fully qualified class name;
`exit_code` is non-zero; `observed_message` is the exact retained failure line;
`fixture_defect_excluded_by` is a non-empty test node or analytic oracle; and
`red_failure_confirmed` must be true. A skipped, xfailed, unexpectedly passed,
collection-only, or unrelated failure is invalid. Every new phase red node
appears exactly once.

Each `commands` row has exactly `argv`, `cwd`, `pixi_environment`,
`started_at_utc`, `duration_seconds`, `exit_code`, `stdout_sha256`, and
`stderr_sha256`. `argv` is a non-empty string array executed without a shell,
and `cwd` is the repository-relative `.`. The phase red validator authenticates
the file bytes, schema literal, node set, command hashes, pre-fix SHA, protected
hashes, and expected non-zero outcomes before `S` is allowed to start.

Correction #24's sole post-source supplement is
`docs/development/sci004_mmode_phase3_post_source_red_failures.json`, with
schema `radiosim.sci004.mmode-phase3-post-source-red-failures.v1` and exactly:

```text
schema_version, phase, status, generated_at_utc, design_sha,
pre_fix_source_sha, red_commit_sha, red_commit_sha_reason,
historical_red_record_sha256, oracle_patch_paths, oracle_patch_sha256,
protected_source_clean, authorized_red_paths, environment, cases,
commands, claims_not_licensed
```

Its `phase` is `M3`, status is `post-source-expected-red-confirmed`,
`design_sha` is correction #24's operative `D`, and `pre_fix_source_sha` is
exactly `a61526d686ab768f05ecffa80cfd6223d4ee4c62`. `red_commit_sha` is null
with reason `self-reference: E binds the containing post-source R commit`.
`historical_red_record_sha256` is exactly
`486705a8d5e51c08f972c91aeae60f0a0bfeef5480b622515282295a6a3cde05`;
the validator also byte-compares that working-tree file to its
`7070cc3ddb1c2557d02e4a3f2a89b907575bed0b` Git blob.
`oracle_patch_paths` is exactly
`["tests/unit/test_io/test_hdf5_result.py"]`, and
`oracle_patch_sha256` hashes the raw stdout bytes of exactly this authoring
command:

```text
git diff --no-ext-diff --binary --full-index a61526d686ab768f05ecffa80cfd6223d4ee4c62 -- tests/unit/test_io/test_hdf5_result.py
```

After R3 commits, the validator inserts the containing R3 commit before `--`
and requires byte-identical diff output. Its authorized paths are exactly the
five correction-#24 R3 paths in Section 13.5. The environment, case rows,
command rows, protected-source semantics, and four claim categories are
otherwise the base red-record forms above.

The historical declaration remains `SCI004_PHASE3_RED_CASES`; the six new
rows live only in `SCI004_PHASE3_POST_SOURCE_RED_CASES`. The historical record
node set must equal the former, the supplemental record node set must equal the
latter, the two sets must be disjoint, and their union is the complete phase-M3
red inventory. Each supplemental fixture preimage is the canonical JSON bytes
of its exact hostile six-key frame. Each node requires early
`UnsafeResultInputError` matching `^HDF5 solver_json is invalid$`; against the
superseded source its retained red outcome is `assertion`, with pattern
`HDF5 result failed canonical model or fingerprint validation` and the exact
pytest regex-mismatch message. The tracked red tool's dedicated
`generate-post-source` mode requires `HEAD` at correction #24, its parent at
the exact superseded S3, only the four non-artifact R3 paths dirty, no dirty
production path, and identical production blobs at those two commits. It runs
the six supplemental nodes plus the four canonical HDF5 nodes and existing
green control serially with `python -m pytest -p no:randomly -p no:xdist`,
requires six matching failures and five passing controls, verifies protected
bytes, and atomically creates only the absent supplemental path. The existing
historical `generate` mode remains unable to overwrite or regenerate its
record.

### 14.2 Phase evidence envelopes and result rows

Each phase evidence artifact has its phase-specific schema literal
`radiosim.sci004.mmode-phase{i}-evidence.v1` and exactly:

```text
schema_version, phase, status, generated_at_utc, design_sha, red_commit_sha,
source_sha, evidence_commit_sha, evidence_commit_sha_reason,
working_tree_clean, environment, source_identities, red_failure_record,
results, commands, limitations, claims_not_licensed
```

`status` is `candidate`; `evidence_commit_sha` is JSON null with the exact
self-reference reason
`self-reference: A binds the containing E commit`; `source_sha` is the clean
checked-out `S`; and
`red_failure_record` has exactly `path`, `sha256`, `schema_version`,
`pre_fix_source_sha`, and `validated`. `environment` has exactly `python`,
`platform`, `machine`, `pixi_environment`, `pixi_lock_sha256`,
`astropy_version`, `erfa_version`, `iers_package_version`,
`iers_table_sha256`, and sorted `numeric_packages`. `source_identities` uses
Section 14.0's exact six-key fixture-input schema; an opaque phase-wide input
digest is forbidden.
`numeric_packages` has exactly `numpy`, `scipy`, `healpy`, `jax`, and `dask`,
each a normalized version string. Evidence `commands` uses Section 14.1's exact
command-row shape and requires exit code zero. `limitations` and
`claims_not_licensed` are sorted unique non-empty string arrays.

Correction #24's M3 `red_failure_record` adds exactly one sixth key,
`post_source_delta`, whose value has exactly `path`, `sha256`,
`schema_version`, `pre_fix_source_sha`, and `validated`. The outer five fields
continue to bind the immutable historical M3 record; the nested five bind
`docs/development/sci004_mmode_phase3_post_source_red_failures.json`, its raw
digest, schema literal, exact `a61526d6…` pre-fix source, and `validated=true`.
The M3 evidence generator and validator authenticate both artifacts and require
the evidence envelope's `red_commit_sha` to name their containing fresh R3.

Correction #25 adds exactly one seventh key,
`fingerprint_post_source_delta`, with the same exact five-key reference shape.
It binds the header's fixed
`docs/development/sci004_mmode_phase3_fingerprint_post_source_red_failures.json`
path, schema, exact
`b07925ab14b56b3ca0fa863f806290748a31df6b` pre-fix source, raw digest, and
`validated=true`.
The outer historical reference and correction-#24 `post_source_delta` remain
byte-authenticated at their two header digests. Fresh E3's `red_commit_sha`
names the new R3 containing all three records, and the generator and validator
require the exact three-way disjoint union of their expected-red case node IDs;
the fingerprint supplement's authenticated passing controls are excluded.

The tracked phase generator is part of `S`, and it **executes at the
globally clean exact `S` checkout**: that execution is the `E`-time
generation, and the artifact it writes is precisely what the following
`E` commit adds. A tracked generator whose `generate` refuses to produce
at clean `S` — reading "runs only at its globally clean exact `S`" as a
prohibition instead of the venue — violates this section. Generation
streams its embedded ledgers rather than materializing them wholesale, and
the committed artifact's size stays within the tens of megabytes the
Section 12.1 scan projection, membership masks, and transfer-sample
concatenation rows are designed to permit.
Before opening any output, the generator
requires `git rev-parse HEAD == source_sha`, an empty index/worktree/untracked
set from `git status --porcelain=v1 --untracked-files=all`, the exact Pixi
manifest/lock, and an absent declared output set. For M1 and M2 that set is
only the phase evidence JSON. For M3 it is exactly the phase evidence JSON and
one Section 11 performance record at the retained host-bound path. The M3
generator computes and canonicalizes the performance bytes first, binds their
future raw SHA-256 into `performance_record`, and computes the evidence bytes
before publishing either file. Each file is written by atomic no-overwrite
rename, performance first and evidence last; a partial set is invalid and may
not be reused. `working_tree_clean=true` means the common pre-output check
passed. After publication the generator requires the repository's only new
paths to equal its phase's exact declared set. Expected new artifacts do not
retroactively make the preflight false. Running an untracked copy of the generator, running
from `E`, or generating against a detached/non-`S` SHA is rejected.

M1 `results` has exactly `dependency_certificate`, `time_grid_cases`, `frame_certificate_cases`,
`scalar_harmonic_cases`, `packed_layout_cases`, `transfer_cases`,
`strategy_cases`, `capability_cases`, `direct_identity_cases`,
`truncation_cases`, and `rejection_cases`. Its exact specialized rows are:

- time grid: `fixture_id`, `sidereal_samples`,
  `integration_fraction_f64be`, `canonical_era_turn_grid`,
  `iers_table_sha256`, `era_center_turn_sha256`,
  `era_lower_edge_turn_sha256`, `era_upper_edge_turn_sha256`,
  `canonical_era_turn_grid_sha256`, `tau_f64be`,
  `delta_alpha_rad_f64be`, `horizon_lo_rad_f64be`,
  `horizon_hi_rad_f64be`, `era_center_rad_sha256`,
  `era_lower_edge_rad_sha256`, `era_upper_edge_rad_sha256`,
  `canonical_era_grid`, `canonical_era_grid_sha256`,
  `era_center_max_residual_rad`,
  `era_center_limit_rad`, `era_step_max_residual_rad`, `era_step_limit_rad`,
  `ut1_utc_roundtrip_seconds`, `ut1_utc_roundtrip_limit_seconds`,
  `utc_manifest`, `utc_sha256`, `ut1_manifest`, `ut1_sha256`,
  `integration_time_seconds_sha256`, and `pass`;
- frame certificate: `fixture_id`, `certificate_sha256`, `site_manifest`,
  `site_sha256`, `input_identity_sha256`, `iers_table_sha256`,
  `frame_matrix_manifest`, `frame_matrix_sha256`,
  `canonical_era_turn_grid_sha256`, `canonical_era_grid_sha256`,
  `pm_source_unit`, `pom00_argument_unit`, `xp0_arcsec`, `yp0_arcsec`,
  `das2r_rad_per_arcsec`, `xp0_rad`, `yp0_rad`, `sp0_rad`,
  `diagnostic_qcheck_nsides`, `transfer_grid_catalog`,
  `transfer_grid_catalog_sha256`, `direction_rows`,
  `direction_ledger_sha256`, `horizon_scan_manifest`,
  `horizon_scan_sha256`, `horizon_scan_crossing_rows`,
  `horizon_scan_summary_rows`,
  `horizon_scan_ledger_sha256`, `horizon_root_pair_rows`,
  `horizon_root_pair_ledger_sha256`, `horizon_slab_rows`,
  `horizon_slab_ledger_sha256`, `horizon_sign_interval_rows`,
  `horizon_sign_interval_ledger_sha256`, `horizon_membership_mask_rows`,
  `horizon_membership_ledger_sha256`, `direct_split_rows`,
  `direct_split_ledger_sha256`, `direct_integrand_enclosure_manifest`,
  `direct_integrand_enclosure_sha256`, `sidereal_samples`, `quadrature_nside`,
  `n_baselines`, `n_frequencies`, `n_correlations`,
  `expected_point_direction_count`, `evaluated_point_direction_count`,
  `expected_native_healpix_direction_count`,
  `evaluated_native_healpix_direction_count`,
  `expected_production_transfer_direction_count`,
  `evaluated_production_transfer_direction_count`,
  `expected_diagnostic_transfer_direction_count`,
  `evaluated_diagnostic_transfer_direction_count`,
  `expected_transfer_quadrature_direction_count`,
  `evaluated_transfer_quadrature_direction_count`,
  `expected_direction_count`, `evaluated_direction_count`,
  `expected_phase_comparison_count`, `evaluated_phase_comparison_count`,
  `expected_horizon_trajectory_count`, `evaluated_horizon_trajectory_count`,
  `expected_horizon_root_pair_row_count`,
  `evaluated_horizon_root_pair_row_count`,
  `expected_horizon_membership_count`, `evaluated_horizon_membership_count`,
  `expected_direct_exposure_split_count`,
  `evaluated_direct_exposure_split_count`, `expected_direct_split_row_count`,
  `evaluated_direct_split_row_count`,
  `expected_frozen_gauss64_node_count`,
  `evaluated_frozen_gauss64_node_count`,
  `expected_frozen_gauss128_node_count`,
  `evaluated_frozen_gauss128_node_count`,
  `expected_operational_gauss64_node_count`,
  `evaluated_operational_gauss64_node_count`,
  `expected_operational_gauss128_node_count`,
  `evaluated_operational_gauss128_node_count`,
  `horizon_isolation_interval_count`, `horizon_unresolved_interval_count`,
  `expected_horizon_slab_row_count`, `evaluated_horizon_slab_row_count`,
  `expected_horizon_sign_interval_count`,
  `evaluated_horizon_sign_interval_count`, `horizon_root_count_mismatches`,
  `horizon_root_orientation_mismatches`, `horizon_membership_mismatches`,
  `horizon_outside_slab_sign_mismatches`, `horizon_paired_root_count`,
  `horizon_mismatch_slab_count`, `horizon_mismatch_measure_turn`,
  `horizon_mismatch_measure_rad`, `horizon_mismatch_measure_limit_rad`,
  `horizon_root_max_rad`, `horizon_root_limit_rad`, `phase_max_rad`,
  `phase_limit_rad`, `expected_cube_cell_count`,
  `evaluated_frozen_gauss64_cube_cell_count`,
  `evaluated_frozen_gauss128_cube_cell_count`,
  `evaluated_operational_gauss64_cube_cell_count`,
  `evaluated_operational_gauss128_cube_cell_count`,
  `compared_frozen_gauss_change_cell_count`,
  `compared_operational_gauss_change_cell_count`,
  `evaluated_frozen_enclosure_error_cell_count`,
  `evaluated_operational_enclosure_error_cell_count`,
  `frozen_gauss64_cube_sha256`, `frozen_gauss128_cube_sha256`,
  `operational_gauss64_cube_sha256`, `operational_gauss128_cube_sha256`,
  `frozen_enclosure_error_cube_sha256`,
  `operational_enclosure_error_cube_sha256`, `direct_gauss_scale_jy`,
  `frozen_gauss_change_max_jy`, `operational_gauss_change_max_jy`,
  `direct_gauss_change_max_jy`, `direct_gauss_change_limit_jy`,
  `cube_scale_jy`, `cube_max_jy`, `cube_limit_jy`, `cube_l2`,
  `cube_l2_limit`, `direction_diagnostic_max_rad`,
  `direction_diagnostic_argmax_id`, `direction_diagnostic_argmax_phase`,
  `basis_diagnostic_max_rad`, `basis_diagnostic_argmax_id`,
  `basis_diagnostic_argmax_phase`, and `pass`;
- packed layout: `fixture_id`, `lmax`, `mmax`, `field_order`, `spin_order`,
  `block_count`, `packed_value_count`, `block_rows`,
  `packed_values_reim_f64be`, `roundtrip_reim_f64be`, `block_table_sha256`,
  `packed_values_sha256`, `roundtrip_sha256`, `invalid_cell_count`, and `pass`;
- direct identity: `fixture_id`, `rime_before_sha256`, `rime_after_sha256`,
  `scientific_before_sha256`, `scientific_after_sha256`,
  `byte_identical`, and `pass`.

M1 `dependency_certificate` has exactly `path`, `raw_sha256`, and
`certificate`. `path` is the fixed R1 path in Section 13.2; `raw_sha256` hashes
the complete retained line including its final LF; and `certificate` is the
parsed exact 16-field WP-7 object in Section 13.2. The evidence generator
and validator replay the clean detached-worktree command at the frozen
replay descendant, require exact bytes and bindings, and set no independent
pass flag that could contradict the replay.

M1 `capability_cases` is an exact six-row discriminated array. Property rows
have exactly `case_kind`, `case_id`, `simulator`, `property`,
`expected_boolean`, `observed_boolean`, `tier7_test_nodeid`, and `pass`.
Registry rows have exactly `case_kind`, `case_id`, `expected_names`,
`observed_names`, `tier7_test_nodeid`, and `pass`. Rejection rows have exactly
`case_kind`, `case_id`, `simulator`, `stokes_field`,
`configured_value_f64be`, `exception_type`, `issue_code`, `exact_message`,
`test_nodeid`, and `pass`. In exact order the rows are:

```text
property,mmode-supports-polarization-false,mmode,supports_polarization,false
property,rime-supports-polarization-true,rime,supports_polarization,true
registry,registry-includes-scalar-mmode,[mmode,rime]
rejection,mmode-rejects-nonzero-q,mmode,Q
rejection,mmode-rejects-nonzero-u,mmode,U
rejection,mmode-rejects-nonzero-v,mmode,V
```

Both property rows use
`tests/characterization/test_tier7_current_behavior.py::test_mmode_m1_capability_truth`;
the registry row uses
`tests/unit/test_tier7_jones_acceptance.py::test_the_accepted_simulator_values_equal_the_registry_keys`.
The Q/U/V rejection rows respectively use exact node IDs
`tests/unit/test_simulator/test_sci004_strategy.py::test_mmode_m1_rejects_nonzero_stokes[Q]`,
`tests/unit/test_simulator/test_sci004_strategy.py::test_mmode_m1_rejects_nonzero_stokes[U]`,
and
`tests/unit/test_simulator/test_sci004_strategy.py::test_mmode_m1_rejects_nonzero_stokes[V]`.
Each sets one named Stokes value to binary64 one and requires
`radiosim.io.config_resolution.UnsupportedConfigError`, issue
`mmode_m1_scalar_only`, and the exact Section 8 message. Every expected value
equals observed and all six `pass` values are true; missing, duplicate,
reordered, inherited-base, or M2-flipped evidence fails M1.

For a time-grid row, the validator decodes `integration_fraction_f64be`,
reconstructs and normalizes every exact turn rational, and proves exact
one-turn width, containment, ordering, exposure width, and full-width
adjacency. It rebuilds the three turn-array digests and exact-turn object
digest; derives every radian value with one final RN; rebuilds all three radian
array digests and the combined grid digest; and verifies UTC/UT1 mapping from
exact turns rather than phase bytes. It reconstructs every output integration
width from the embedded UTC edges and requires the exact Section 3.1 array
digest. A predicate requiring rounded
`horizon_hi_rad-horizon_lo_rad==tau` is itself a validation failure. A frame
row with the same fixture joins both canonical grid digests; every phase node,
horizon scan row, membership row, and direct partition uses that object.

For a frame row, the retained site and frame manifests, two unit literals, and
all six unit-named polar-motion/s-prime numbers must satisfy Section 4.1. The
validator checks the site digest and WGS84 position, finite values,
the installed `DAS2R` bits, both one-multiplication identities, the installed
`sp00` result, the recomputed `RPOM0`, `T0`, local ENU triad, and
`frame_matrix_sha256`; a row cannot authenticate a frame matrix while omitting
its preimage or relabelling its angular units.

Let `Q_diag` be the retained sorted diagnostic-nside array,
`D_prod=12*quadrature_nside**2`,
`D_diag=sum(12*q**2 for q in Q_diag)`,
`D_transfer=D_prod+D_diag`, and
`D=D_point+D_native+D_transfer`. The embedded transfer catalogue must have
exactly one production row and one row for every `Q_diag` value, with exact
per-grid expected/evaluated counts and contiguous direction-ID slice digests.
The four production/diagnostic count pairs and total transfer count pair equal
these formulas. Let `P_d` be the exact ordered phase-node ledger for direction
`d`, including centres, exposure boundaries, and both root-enclosure endpoint
sets. With `N=sidereal_samples`, `C=4`, and `P_dk` the reconstructed direct
piece count, exact counts are:

```text
expected_phase_comparison_count = B*F*sum_d(len(P_d))
expected_horizon_trajectory_count = D
expected_horizon_root_pair_row_count = D
expected_horizon_membership_count = D*N
expected_direct_exposure_split_count = (D_point+D_native)*N
expected_direct_split_row_count = B*C*F*sum_d,sum_k(P_dk)
expected_cube_cell_count = N*B*F*C, with C=4
```

Every evaluated count equals its expected count. The direct exposure count is
the exact distinct direction/sample projection of embedded `direct_split_rows`;
the full ledger has one row per direction/sample/frequency/baseline/
correlation/piece and is authenticated by `direct_split_ledger_sha256` under
Section 12.1. Node counts are recomputed by model/order from those rows.
Neither an aggregate counter nor a direction/sample-only stream proves
coverage.

The embedded horizon ledgers and their digests must satisfy every Section 12.1
schema, order, join, and recomputation predicate. In particular,
`evaluated_horizon_root_pair_row_count` and the root-pair array length equal
`D`; `horizon_isolation_interval_count` equals the summary rows'
`terminal_cell_count` sum across the identical ordered direction IDs, with
every embedded scan row joined to its direction's summary and
`crossing_count` totals matching the census; and
`expected_horizon_slab_row_count` equals the sum of all embedded pair-array
lengths and `horizon_paired_root_count`. The evaluated slab-row count,
`horizon_mismatch_slab_count`, and embedded slab-array length must all equal
that expected value. `expected_horizon_sign_interval_count` is the number of
canonical non-empty outside-slab complement pieces recomputed from those slab
rows, and its evaluated count and embedded sign-row length must match.
The membership mask-row length is exactly `D`, each row's `sample_count`
is exactly `N`, and the expanded per-sample array has exactly `D*N` rows.

`horizon_membership_mismatches` counts false `match` values across exactly
the expanded sample-centre rows whose centres lie outside every mismatch
slab; a slab-interior centre's disagreement joins the slab accounting per
Section 4.2 and is excluded from this counter, while the mask rows'
`mismatch_count` values remain the per-direction totals from which the
outside-slab counter is recomputed against the slab geometry;
`horizon_outside_slab_sign_mismatches` counts false `match` values in the sign
rows. `horizon_root_count_mismatches` is the number of root-pair rows whose two
root counts differ, `horizon_root_orientation_mismatches` is the sum of their
`orientation_mismatch_count` values, and `horizon_root_max_rad` is the maximum
outward-decoded embedded `worst_case_delta_rad_f64be`, or exact zero for no
pairs. A midpoint or representative-root delta is forbidden.
All four mismatch counters and `horizon_unresolved_interval_count` must be
zero. `horizon_mismatch_measure_turn` is the exact rational union measure of
the embedded closed slab pieces in the exact one-turn domain. Its radian view
is the outward product with retained tau; overlaps and seam pieces count once.
`horizon_mismatch_measure_limit_rad` is exactly
`2e-5*horizon_paired_root_count`. Root, mismatch-measure, phase, complete
direct-ledger, all four qualified cube/count/digest predicates, both enclosure-
error cubes, both model 64-to-128 reductions, and the certified frame maximum
and L2 bounds are mandatory for `pass`; no diagnostic-angle field enters it.

For the exact frame row, all four qualified direct cubes and both error cubes
use the Section 12 identities, shape `[N,B,F,4]`, and exactly
`K=4*N*B*F` cells. Every evaluated and compared cube count equals `K`; both
error cubes are finite and non-negative. The four node-count pairs equal sums
over embedded direct-split rows. `direct_gauss_scale_jy`, both model changes,
their maximum, and `direct_gauss_change_limit_jy` are exactly the Section 12
`S_Q`, `Q_F`, `Q_O`, `Q`, and `1e-11*S_Q`. `cube_scale_jy`, `cube_max_jy`,
`cube_limit_jy`, `cube_l2`, and `cube_l2_limit` are exactly `S_frame`,
`max(U)`, `5e-5*S_frame+1e-10 Jy`, the stated normalized L2, and `5e-5`.
Frame `pass` is the conjunction of every schema, unit, input, exact-grid,
catalogue, ledger/digest/join/count, root-enclosure/slab, phase, four-cube,
two-error-cube, Gauss-change, maximum, and L2 predicate. An aggregate counter,
unqualified cube alias, omitted cell preimage, or root midpoint cannot
substitute.

M2 `results` has exactly `frame_certificate_cases`, `polarization_cases`,
`sky_component_cases`, `direct_convergence_cases`, `truncation_cases`,
`backend_parity_cases`, `memory_cases`, `capability_cases`, and
`rejection_cases`. Its frame rows reuse the complete M1 frame schema and are
recomputed from M2 inputs; references to M1 evidence are forbidden. A direct-
convergence row has exactly `fixture_id`, `input_identity_sha256`,
`frame_certificate_sha256`, `cube_shape`, `expected_cell_count`,
`compared_finite_cell_count`, `frozen_gauss64_cube_sha256`,
`frozen_gauss128_cube_sha256`, `frozen_enclosure_error_cube_sha256`,
`mmode_cube_sha256`, `gauss_change_max_jy`, `gauss_change_limit_jy`,
`analytic_piecewise_residual`, `analytic_piecewise_limit`,
`direct_scale_jy`, `deficit_max_jy`, `deficit_l2`,
`deficit_max_quarter_jy`, `deficit_max_half_jy`, `convergence_factor`,
`truncation_budget_jy`, `wrong_sign_residuals`, and `pass`. Its three
certificate
digests and counts equal the unique same-fixture M2 frame row, and every
deficit reduction includes its frozen error cube per Section 7.3's tier-2
formulas.

Every row in M1 and M2 `truncation_cases` has exactly `fixture_id`,
`input_identity_sha256`, `frame_certificate_sha256`,
`direction_ledger_sha256`, `transfer_grid_catalog_sha256`,
`production_transfer_grid_id`, `diagnostic_transfer_grid_ids`,
`diagnostic_grid_joins`, `lmax`, `mmax`, `quadrature_nside`, `lcheck`,
`mcheck`, `qcheck`, `sidereal_samples`, `cube_shape`,
`frozen_gauss128_cube_sha256`, `frozen_enclosure_error_cube_sha256`,
`mmode_cube_sha256`, `direct_scale_jy`, `expected_output_cell_count`,
`evaluated_frozen_direct_cell_count`, `evaluated_frozen_error_cell_count`,
`evaluated_mmode_cell_count`, `compared_output_cell_count`, `direct_coverage`,
`direct_coverage_sha256`, `horizon_free_shell_max_jy`,
`horizon_free_shell_l2`, `horizon_free_shell_max_limit_jy`,
`horizon_free_shell_l2_limit`, `quadrature_shell_max_jy`,
`quadrature_shell_l2`, `quadrature_budget_jy`, `deficit_max_jy`,
`deficit_l2`,
`deficit_max_quarter_jy`, `deficit_max_half_jy`, `convergence_factor`,
`truncation_budget_jy`, `expected_shell_comparison_cell_count`,
`evaluated_shell_comparison_cell_count`, `expected_transfer_sample_row_count`,
`evaluated_transfer_sample_row_count`, `expected_field_block_count`,
`evaluated_field_block_count`, `shell_coverage`, `shell_coverage_sha256`,
`quadrature_diagnostic_max_jy`, `l_tail_diagnostic_max_jy`,
`m_tail_diagnostic_max_jy`, `combined_local_diagnostic_max_jy`,
`field_block_diagnostic_max_jy`, `shell_diagnostic_reference_jy`, and `pass`.

`direct_coverage` has exactly `schema_version`, `input_identity_sha256`,
`cube_shape`, `frozen_gauss128_cube_sha256`,
`frozen_enclosure_error_cube_sha256`, `mmode_cube_sha256`, and `rows`, with
schema literal `radiosim.mmode-direct-output-coverage.v1`. Each row has exactly
`sample_index`, `baseline_index`, `frequency_index`, `correlation_index`,
`frozen_real_f64be`, `frozen_imag_f64be`, `frozen_error_f64be`,
`mmode_real_f64be`, `mmode_imag_f64be`, and `upper_delta_f64be`. Rows are
exact `[N,B,F,4]` C-order and `upper_delta` is the outward value
`abs(V0-F128)+EF`. A row is appended only after all named cells were read,
proved finite, and included in maximum and L2 reductions.
`direct_coverage_sha256` is
`D("radiosim.mmode-direct-output-coverage.v1",J(direct_coverage))`; the
validator reconstructs and replays all rows.

`cube_shape` is exactly `[N,B,F,4]`; `K=4*N*B*F`; all four evaluated or
compared direct counts equal `K`; shell comparison count is `4*K`; transfer-
sample count is `(1+len(Q_diag))*B*F*4*4`, with every row's
`direction_count` equal to its grid's `12*nside**2` catalogue count; and
field-
block count is `B*F*4*4*(2*mcheck+1)`. All observed counts equal expected.
The certificate, input, direction, catalogue, production grid ID, exact
diagnostic grid ID array `['diagnostic:<qcheck>']`, four grid joins, final
frozen/error digests, and direct counts equal the unique same-fixture frame
row. Every grid ID resolves exactly once in its catalogue.

`shell_coverage` is the complete embedded Section 7.3 preimage and its domain
is `radiosim.mmode-shell-coverage.v1`; it is not replaceable by counts or a
bare digest. `direct_scale_jy=max(1 Jy,max(abs(F128)+EF))`; the
horizon-free shell values and limits, the with-horizon shell values, the
three deficit values, and
`convergence_factor` are exactly the Section 7.3 two-tier quantities,
recomputed by the validator; `truncation_budget_jy` and
`quadrature_budget_jy` are the fixture's declared budgets, which
`deficit_max_jy` and `quadrature_shell_max_jy` respectively must not
exceed. `pass` requires
every identity, join, digest, count, finite-value, tier-1a horizon-free,
convergence, both-budget, and coverage predicate. The shell
reference and remaining diagnostic magnitudes are exactly recomputed from
embedded rows but remain non-gating attribution values.

`wrong_sign_residuals` has exactly `fourier_sign_jy`, `v_bridge_jy`,
`tangent_transport_jy`, and `east_x_permutation_jy`. A polarization row has
exactly `fixture_id`, `input_frame_sha256`, `transported_frame_sha256`,
`stokes_case`, `expected_cube_sha256`, `observed_cube_sha256`,
`absolute_residual`, `fixed_tolerance`, and `pass`. A sky-component row has
exactly `fixture_id`, `representation`, `point_coefficients_sha256`,
`healpix_coefficients_sha256`, `hybrid_coefficients_sha256`,
`expected_sum_sha256`, `ring_nest_equal`, and `pass`. A capability row has
exactly `simulator`, `property`, `expected`, `observed`, `tier7_test_nodeid`,
and `pass`.

M3 `results` has exactly `dependency_certificate`, `output_cases`,
`fingerprint_rows`, `ci_artifacts`, `performance_record`,
`release_scan_cases`, and `rejection_cases`. M3 begins only after the accepted
SCI-005 Stage-2 record presents unlock token `SCI004.M3`. Its
`dependency_certificate` object has exactly
`sci005_stage2_acceptance_commit_sha`,
`sci005_stage2_acceptance_artifact_sha256`, and
`sci005_stage2_certificate_stdout_sha256`; the validator authenticates all
three against that acceptance record and the fixed R3 dependency path in
Section 13.2. The stdout digest covers the exact retained JSON line including
its final LF. The generator and validator perform both detached-worktree
replays required there (`--descendant G3` and `--descendant R3`), require
byte-identical stdout and empty stderr from both, and require the parsed
acceptance commit/artifact values to equal the first two fields. No green
workflow summary, unlock literal alone, or differently serialized certificate
can satisfy this object.

An output
row has exactly `format`, `fixture_id`, `written_solver_sha256`,
`read_solver_sha256`, `time_sha256`, `feed_sha256`, `correlation_sha256`,
`file_sha256`, `written_cube_sha256`, `read_cube_sha256`,
`scientific_sha256`, and `pass`.

Each `fingerprint_rows` entry has exactly `family_id`, `fixture_id`,
`characterization_input_manifest`, `input_identity_sha256`,
`characterization_time_manifest`, `era_utc_grid_sha256`, `solver_snapshot`,
`solver_snapshot_sha256`,
`cube_sha256`, `scientific_sha256`, `expected_change_reason`, and `pass`.
There are exactly four rows in the amended Section 11 family order, one local
retained pin per family. For every row the validator independently performs
all of the following before `pass` can be true:

- requires the manifest's exact Section 14.0 v2 key set, rejects unknown,
  missing, path-bearing, or non-normalized values, canonicalizes with `J`, and
  recomputes `input_identity_sha256` with `D`;
- requires exact `family_id == fixture_id`, joins the unique same-fixture phase
  input row, and requires the nested v1 manifest and digest to equal that row;
- reconstructs and joins instrument, receptor, loaded-beam, correlations,
  polarization basis, frequencies, both distinct phase ERA and result-derived
  ERA/UTC characterization-time identities, and all semantic solver inputs
  from the retained value-bearing manifests;
- requires the complete retained `solver_snapshot` to have Section 10's exact
  key set, independently recomputes `solver_snapshot_sha256`, and joins its
  dimensions, frame, IERS, tangent-polarization frame, and execution policy to
  the phase input manifest;
- joins `cube_sha256` and `scientific_sha256` to the same family's
  `ci_artifacts` row and retained observation-set entry, and, for the three
  performance fixtures, to every same-fixture performance workload identity;
  the scientific identity is independently rederived at A3 and is never an
  input to the characterization manifest; and
- requires `expected_change_reason` to equal this exact one-line value:

  ```text
  characterization-input v2 removes non-semantic filesystem venue and retains the complete path-independent phase-input preimage; scientific, cube, ERA/UTC characterization-time, solver, phase-input, and observation-set identities are unchanged
  ```

  If any
  named supposedly unchanged identity differs, generation fails before this
  literal can be used and an old/new-cube equation-level adjudication requires
  another correction.

Hostile strict-validator coverage includes at least: missing preimage; unknown
preimage key; forbidden path-bearing or non-normalized value; changed manifest
with stale digest; changed digest with unchanged manifest; instrument,
receptor, loaded-beam, correlation/polarization/frequency, fixture-input,
characterization-time manifest/digest, phase-grid join, `scientific_sha256`, or
retained-observation-set mismatch; two scientifically identical layouts under
distinct temporary roots producing equal v2 identity; and a real semantic
layout-content change producing a different identity. The relocation equality
is a fresh R3 expected-red oracle; the semantic inequality is the fresh R3
mandatory green control. Neither may be replaced by a synthetic document whose
author supplied both expected values.

Each `ci_artifacts` entry has exactly `family_id`, `fixture_id`, `source_sha`,
`environment`, `dispatch_identity`, `cube_sha256`,
`scientific_sha256`, `numeric_delta`, `expected_change_reason`,
`ci001_verdict`, and `pass`. Family order is the exact four-name amended
Section 11
order; for each family, rows cover exactly the platform/Python cells the
amended Section 11 harvest sentence binds — the cells this phase's
acceptance actually runs on — and then the accepted dispatch-identity
observation order, each row authenticated against the retained
observation-set surface at the clean `S3` checkout. The validator
reconstructs that complete inventory from the retained rows,
rejects a missing/extra/duplicate family-cell-dispatch tuple, and joins each
fixture ID to the phase input set. Remote CI cells and their
`run_id`/`job_id`/`artifact_id`/`artifact_sha256` artifacts enter
afterwards by the
standing admission discipline, exactly as the accepted AVX-512
admissions did; the local `E3` venue can retain only what it runs.

The `performance_record` object has exactly `path`, `sha256`,
`schema_version`, `source_sha`, `workload_count`, `workload_identities`,
`authenticated`, and `claims_not_licensed`. `workload_count` is nine.
`workload_identities` has one row in exact Section 11 workload order, each
with exactly `workload_id`, `input_identity_sha256`,
`frame_certificate_sha256`, `scientific_sha256`, and `result_cube_sha256`.
Values equal the referenced benchmark rows. `sha256` authenticates raw
canonical file bytes; source equals benchmark provenance and M3 evidence;
`authenticated` becomes true only after the strict Section 11 parser and every
cross-row predicate pass; claims equal the exact Section 11 array.

A strategy row has exactly `fixture_id`, `requested_registry_key`,
`resolved_class`, `sky_representation`, `direct_strategy_calls`,
`mmode_strategy_calls`, `output_cube_sha256`, and `pass`. A release-scan row
has exactly `scan_id`, `command_index`, `roadmap_occurrences`,
`done_occurrences`, `unsupported_claim_occurrences`, `expected_counts`, and
`pass`; `expected_counts` has exactly those same three integer count names.

Across all phases, an analytic row has exactly `fixture_id`, `equation_id`,
`convention`, `spin`, `l`, `m`, `test_nodeid`, `expected_real`,
`expected_imag`, `observed_real`, `observed_imag`, `absolute_residual`,
`fixed_tolerance`, and `pass`; a rejection row has exactly `fixture_id`,
`config_path`, `exception_type`, `issue_code`, `exact_message`, `test_nodeid`,
`allocation_started`, `output_path_created`, and `pass`; a backend row has
exactly `fixture_id`, `requested_backend`, `actual_backend`, `actual_device`,
`dtype`, `workers`, `working_memory_bytes`, `numpy_sha256`,
`candidate_sha256`, `absolute_max`, `relative_max`, `rtol`, `atol`, and `pass`;
and a memory row has exactly `fixture_id`, `logical_dimensions`,
`block_dimensions`, `included_allocations`, `excluded_allocations`,
`estimated_components`, `estimated_peak_bytes`, `measured_host_peak_bytes`,
`host_measurement_method`, `measured_native_peak_bytes`,
`measured_native_peak_bytes_reason`,
`native_measurement_method`, `working_memory_bytes`, `schedule_rows`,
`schedule_sha256`, and `pass`.

`logical_dimensions` has exactly `n_times`, `n_baselines`, `n_frequencies`,
`n_correlations`, `n_packed_values`, and `n_quadrature_directions`.
`n_quadrature_directions` equals the complete production-plus-diagnostic
transfer catalogue count. `block_dimensions` has exactly
`frequency_block_max`, `signed_m_block_max`, `baseline_block_max`,
`packed_value_block_max`, and `scheduled_block_count`, matching Section 11's
schedule summary. Every allocation
row has exactly `name`, `bytes`, and `measurement_domain`; every estimated
component row has exactly `name` and `bytes`. Allocation and component names
are unique and sorted. `measured_native_peak_bytes` is either a non-negative
integer or null; its reason is exactly `measured` in the former case and a
non-empty measurement limitation in the latter. A scalar/transfer analytic
list uses the analytic-row schema above; there is no second abbreviated row
shape.
`schedule_rows` uses Section 11's exact row schema and
`schedule_sha256=D("radiosim.sci004.block-schedule.v1",J(schedule_rows))`.
The validator rebuilds the complete ordered schedule from the fixture and
`working_memory_bytes`, requires exact row equality, and reconstructs every
`block_dimensions` maximum and count from the embedded rows.

The phase generator fails closed on dirty/unknown source, a non-parent `R`, a
stale manifest/lock, output overwrite, wrong Pixi environment, a missing or
unauthenticated input artifact, incomplete full-block coverage, a non-finite
number, or any false row. The phase validator is also already tracked at `S`.
In its S-state, null approved-digest constants require the official evidence
artifact to be absent while synthetic strict-schema/digest tests pass. `E`
adds the artifact and reproduction Markdown and changes only
`APPROVED_SOURCE_SHA`, `APPROVED_ARTIFACT_SHA256`,
`APPROVED_PERFORMANCE_PATH`, and `APPROVED_PERFORMANCE_SHA256`. In that state
the validator requires the artifact's `source_sha` and the constant to equal
the approved `S`, authenticates the raw artifact bytes, locates current `E`
only by Section 13.7's approved-parent first-parent rule, requires it to be the
sole-parent direct child of `S`, requires its parent-relative diff to introduce
the fixed artifact and current host-bound performance path, and
checks the `S..E` diff against Section 13. It deliberately does **not** require
the current checkout or `E` to equal `source_sha`. It re-runs schema validation
and all cheap digest/oracle checks under default and `py312`; it never selects
the first or newest matching add in whole history and never trusts a workflow
summary. The reproduction record
names the exact tracked generator path/argv, Pixi environment, approved `S`,
artifact path, raw SHA-256, and commands needed to reproduce it. It begins with
the exact MyST front matter `---\norphan: true\n---`, so adding the phase-local
record does not create a strict-Sphinx orphan warning.

### 14.3 Independent acceptance records

Each `docs/development/sci004_mmode_phase{i}_acceptance.json` has schema literal
`radiosim.sci004.mmode-phase{i}-acceptance.v1` and exactly:

```text
schema_version, phase, verdict, generated_at_utc, reviewer_identity,
reviewer_independent, design_sha, red_commit_sha, source_sha,
evidence_commit_sha, evidence_artifact_path, evidence_artifact_sha256,
acceptance_commit_sha, acceptance_commit_sha_reason, reviewed_artifacts,
rederived_oracles, commands, blockers, accepted_limitations,
claims_not_licensed
```

`acceptance_commit_sha` is JSON null with the exact reason
`self-reference: the next R or C binds the containing A commit`.
Every `reviewed_artifacts` row has exactly `path`, `sha256`, `source_sha`, and
`authenticated`; every `rederived_oracles` row has exactly `oracle_id`,
`method`, `observed`, `fixed_limit`, and `pass`. `ACCEPT` requires an independent
reviewer, no false oracle, an empty `blockers` array, exact `S -> E` ancestry,
an authenticated phase evidence artifact, and no production-source path in
the `E..A` diff. `REJECT` requires at least one concrete blocker and does not
unlock the next phase. The acceptance validator authenticates exact bytes and
diff authority; prose appended to this design is not a substitute.

`reviewer_identity` is a non-empty role/task identifier and
`reviewer_independent` must be true. Acceptance `commands` uses Section 14.1's
command-row shape with zero exit codes. Every blocker row has exactly
`blocker_id`, `requirement_id`, `evidence`, and `required_remediation`.
`accepted_limitations` and `claims_not_licensed` are sorted unique non-empty
string arrays. `observed` and `fixed_limit` in a rederived-oracle row are
finite numbers in the oracle's units, which the `method` string names.

The acceptance generator and validator are already tracked at `S`. After an
independent reviewer finishes, the generator **executes at the globally
clean exact `E` checkout** — that execution is the `A`-time generation, and
the artifact it writes is precisely what the following `A` commit adds; a
tracked generator whose `generate` refuses to produce at clean `E` violates
this section. It requires the active evidence validator to authenticate
that `E` and
its approved `S`, and atomically creates the absent acceptance JSON. In its
pre-A state the acceptance validator's null approved-digest constants require
that JSON to be absent while synthetic schema tests pass. `A` adds the JSON,
changes only `APPROVED_EVIDENCE_SHA` and
`APPROVED_ACCEPTANCE_ARTIFACT_SHA256`, and may make the status-document edits
listed in Section 13. The active validator authenticates the approved `E`, raw
acceptance bytes, and current `A` selected only as the first first-parent commit
after the approved `E` under Section 13.7. It requires that sole-parent direct
child to introduce the fixed acceptance path and satisfy exact `E..A`
authority; a prior rejected add of the same path is ineligible. It never
requires the evidence artifact's `source_sha` to equal `E`.

`A1` must independently rederive `CanonicalEraGrid`: all four exact-turn
digests, all three derived-radian array digests, the combined grid digest,
exact `h_N^+-h_N^-=1/1`, full-width adjacency, and an explicit demonstration
that rounded horizon-endpoint subtraction is not a closure predicate. It also
rederives the exact `pm_xy` arcsecond-to-radian conversion,
`RPOM0 R3(ERA0+alpha)` and its active/passive sign, one
public-Astropy tangent Jacobian, one analytic horizon root set, the DFT sign,
and the complete bounded-driver frame certificate. That certificate review
must reconstruct point/native plus production/diagnostic transfer catalogue
slices, active-frequency payload identities, all horizon ledger digests and
joins, enclosure-based pairing and closed-hull slabs, every mismatch/count,
internal-boundary ownership, every direct split cell row/preimage, all four
qualified cube digests, both error cubes, and both certified reductions. It
must also rederive direct-RIME byte identity, the scalar two-tier gate —
the tier-1a horizon-free fidelity predicates, the recorded with-horizon
shell against its declared budget, and the deficit convergence and
budget predicates — with exact direct/shell/transfer-sample
coverage, and all six exact M1 scalar-capability rows.
Its required `m1.wp7-dependency-gate` oracle authenticates D/G1/WP-7 ancestry,
nonmerge and immutable-byte predicates, the raw retained certificate, and the
clean detached-worktree replay at the frozen replay descendant.
`A2` must rederive
the North/East-to-theta/phi V bridge, one polarized `B_lm` equation —
these two confirming, independently of the implementation, that the
accepted M2 scope carries no mount tangent rotation (`P` is exactly the
identity for the shipped `fixed` and unspecified mounts) and that the
kernel's constant receptor cells act in the same celestial tangent
basis as the sky expansion; Section 6's transport-sign obligation binds
only a future ground-anchored response, because a consistent sign flip
through kernel and sky cancels in every paired comparison and no red
oracle can pin it — one
horizon-split exposure, its phase-local frame certificate and four direct/error
cubes, the two-tier gate's fidelity, convergence, and budget predicates,
production/qcheck
catalogue and four grid joins, exact direct/transfer-sample/shell/block
coverage, and local shell diagnostics, plus backend/memory predicates and the
deliberate Tier 7 capability flip. `A3` must authenticate the exact SCI-005
Stage-2 `SCI004.M3` dependency fields, one standard-output round trip, and
every fingerprint and retained observation-set artifact. Its required oracle IDs additionally
include `m3.sci005-dependency-gate`, `m3.performance-schema`,
`m3.performance-provenance`,
`m3.performance-inventory`, `m3.performance-schedule`,
`m3.performance-timing`, `m3.performance-memory`,
`m3.performance-direct-predicate`, and `m3.performance-backend-predicate`.
After correction #25 they also include required identifier
`m3.fingerprint-authentication`. That oracle independently reconstructs every
v2 characterization manifest and digest, every characterization-time preimage
and its distinct ERA/UTC digest, all Section 14.2 joins and hostile mutations,
the equal-across-distinct-roots expected-red scientific oracle, and the
semantic-change inequality green control. It may not import the production
characterization helper as its oracle. Every A3 oracle is rederived afresh;
passing rows in a rejected acceptance record are historical evidence, not
reusable acceptance.
The dependency oracle performs both G3/R3 detached-worktree verifier replays
and authenticates D/A2/G3/SCI-005 ancestry, nonmerge, immutable bytes, and the
raw stdout digest. The A3 validator authenticates the raw performance path/digest, exact S3 and
lock, all nine ordered identity joins, schedules, timing tagged unions and
sample cardinality, host/native memory rules, and both fixed numerical
predicates. It asserts no elapsed-time threshold and requires a release scan
that still reports `SCI-004` as ROADMAP. These are required
`rederived_oracles` identifiers, not optional reviewer prose.

### 14.4 Mandatory commit order

The order is `D ->* G1 -> R1 -> S1 -> E1 -> A1 ->* R2 ->* S2 -> E2 -> A2
->* G3 ->* R3 ->* S3 -> E3 -> A3 -> C`; the `A1 ->* R2`, `R2 ->* S2`,
`G3 ->* R3`, and `R3 ->* S3` stars are the
concrete effects of the header's starred-edge correction records under
the Section 13.7 rule above, which collectively enumerate each such
edge's
interval commits. Each starred edge is inclusive ancestor
reachability through separately authorized, independently accepted programme
commits; every unstarred edge is the sole direct-parent edge. No commit in
any starred first-parent range is a merge. `G1` also has accepted WP-7 CPU
`A` as an authenticated ancestor; `G3` also has accepted SCI-005 Stage-2 `A2`
as an authenticated ancestor. Exact bindings and immutable-byte rules are
Section 13.2's. A Section 13.7 accepted bounded correction that intervenes
at any edge replaces that exact direct-parent edge with a starred edge whose
interval commits the memo header enumerates exhaustively under
Section 13.7's recorded kinds; the reopened phase's `R` then directly
parents the operative correction commit.

Correction #25 makes the concrete M3 history and retry order:

```text
4d507bf1333ccaa4c8beec3815370ba0f6043bb2 D24
  -> 944e0ee66ebdaffafab86f4f8f4253a404aa902c superseded R3
  -> b07925ab14b56b3ca0fa863f806290748a31df6b superseded S3
  -> 886e62fd9f8328826b388b8960ed7413da26b6d1 rejected E3
  -> 8529da951e2378115ffde8d5da3e2af56f3323d0 rejected A3 (REJECT)
  -> D25
  -> fresh fingerprint-delta R3
  -> replacement S3
  -> fresh E3
  -> fresh independent A3
  -> C only after that A3 is ACCEPT
```

The four commits between D24 and D25 have exactly the header-recorded kinds and
paths; D24 becomes a superseded-design chain commit and D25 is operative `D`.
The rejected A3 is not the `A3` in the generic accepted succession and creates
no `A3 -> C` edge. Correction #25's fresh R3 directly parents D25; its
then-ruled replacement S3 direct-parent edge is superseded by the later
header-recorded correction #26, correction #27, correction #28, and correction #29
successions. In the landed current succession, D27 directly parents old fresh
validator-only R3, D28 directly parents D27, and D29 directly parents D28. The
later ruled succession is D29 `->` future
validator/oracle/evidence-binding R3 `->` eventual replacement S3: the new R3
directly parents D29 and S3 directly parents the new R3. The fresh E3/A3 edges
remain future unstarred sole-parent edges.
Disposal in replacement S3 makes those fixed paths genuinely absent before
their no-overwrite generators run; the approved-parent first-parent selection
rule then distinguishes the fresh introductions from the immutable rejected
ones.

Thus `R1^==G1`, while `R2` and `R3` each directly parent the operative
commit of a header-recorded starred-edge correction — `R2` over
`A1 ->* R2`, `R3` over `G3 ->* R3` — the enumerated intervals standing
between each and its gate endpoint; an
unstarred `R` edge's `R^` equals its `A` or `G` endpoint unless a
Section 13.7 accepted correction stars that edge, in which case the
header enumerates the interval and the phase's `R` directly parents the
operative correction commit. Each `S` directly parents its phase `R` —
unless a Section 13.7 accepted correction has starred that edge, in
which case `S` directly parents the operative correction commit per the
Section 13.7 rule, as `S2` does here. Correction #29 instead explicitly
reopens R3: eventual replacement S3 directly parents the post-D29
validator/oracle/evidence-binding R3 — and
contains production plus the already tracked evidence/acceptance tools and
validators, but no phase evidence or acceptance artifact. Each generator runs
only at its globally clean exact `S` or `E`, respectively. Each `E` directly
parents `S` and adds only the generated evidence artifact, its reproduction
record, the exact evidence-validator constants, and E3's authenticated
performance record. Each `A` directly parents and names `E` and changes only
its acceptance artifact/constants and authorized status prose.

`R` artifacts use a null self SHA and are bound to the exact containing commit
by `E`; `E` artifacts use a null self SHA and are bound by `A`; `A` artifacts
use a null self SHA and are bound by the next phase or `C`. An `E` or `A` commit
that changes validator logic or production source, an `S` that changes its red
oracle, or a commit that combines two phase letters is invalid and must be
split.

## 15. Dependency, verification, and closure gates

The design can be accepted now because Q5 is resolved. Production remains
ordered:

- M1 starts only at authenticated clean `G1` after WP-7 P-a through P-d are
  independently accepted and its exact certificate is replayed at the frozen
  historical descendant;
- M2 starts only after accepted M1;
- M3 starts only at authenticated clean `G3` after accepted M2 and accepted
  SCI-005 Stage 2, with both exact Stage-2 verifier replays passing; and
- whole-row closure follows accepted M3 and a separate end-to-end review.

CI-001 is already closed, but its successor-gate discipline still governs new
fingerprints. SCI-005 Stage 3 is not a hidden prerequisite or a hidden claim.

Correction #29 changes only the M3 historical/current design-binding and
future succession authority. It preserves mandatory production v2, defines
but does not author a later five-path validator/oracle/evidence-binding R3,
and does not retry S3. It accepts no source or evidence, does not reverse the
canonical A3 REJECT, does not accept M3, and does not reopen M1 or M2. SCI-004
remains ROADMAP and closure C remains locked until a fresh independent A3
returns ACCEPT and the later whole-row review independently succeeds. PERF-001
remains ROADMAP, and no accepted accelerator-performance record exists. No
accelerator, GPU, diffuse/public HEALPix, non-scalar-beam, public end-to-end
backend-wiring, speedup, general-performance, production-readiness,
phase-acceptance, closure, or unmeasured-workload claim follows from this
correction or from disposal of the rejected drafts.

Each source phase runs at minimum:

```text
pixi run test -- <phase-focused tests>
pixi run test -- tests/unit/
pixi run test -- tests/integration/
pixi run test -- -m "not slow"
pixi run doctest
pixi run lint
pixi run check-format
pixi run typecheck
make -C docs html
git diff --check
```

M2/M3 also run the non-gating performance suite. M3 runs all output
characterizations on the cells its acceptance actually runs, the
relevant dispatch
classes, and a green exact-pin-SHA rerun; remote environment cells enter
afterwards by the standing admission discipline. A retained artifact is
authenticated from its bytes rather than inferred from a green workflow
summary.

For each phase, an independent reviewer inspects the exact `R`, `S`, and `E`
commits, performs that phase's Section 14.3 re-derivations without the
implementation helper, runs one typed rejection by hand, authenticates every
retained artifact, verifies the phase-local diff boundary, and returns a
strict acceptance artifact with `ACCEPT` or `REJECT` and concrete blockers.

`SCI-004` closes only when all of the following are true:

- the registry is a truthful whole-SkyModel boundary and direct RIME is still a
  maintained strategy;
- the adopted HERA-like full-sidereal driver is enforced by typed config;
- ERA/UTC, frozen-frame, tangent-basis, Stokes-V/east-X, scalar/spin harmonic,
  transfer, truncation, and aliasing contracts pass their fixed oracles;
- point, HEALPix, and hybrid full-Stokes cases converge to the common direct
  oracle, including an accepted non-scalar SCI-005 Stage-2 beam;
- NumPy/JAX/Dask behavior, memory scheduling, result/output round trips,
  fingerprints, CI-001 adjudication, and exact-SHA CI are retained;
- no unsupported performance or frame-accuracy claim appears; and
- the register, changelog, migration guide, API, and user documentation agree.

Writing this design, registering a stub, producing one scalar demo, or passing
one workflow does not close the row.

## 16. Primary sources and official conventions

- J. R. Shaw et al., [*All-Sky Interferometry with Spherical Harmonic Transit
  Telescopes*](https://arxiv.org/abs/1302.0327), ApJ 781, 57 (2014),
  [DOI 10.1088/0004-637X/781/2/57](https://doi.org/10.1088/0004-637X/781/2/57):
  transit periodicity, `B_lm`, the `exp(-i*m*phi)` m transform, and independent
  per-m forward systems.
- J. R. Shaw et al., [*Coaxing Cosmic 21 cm Fluctuations from the Polarized Sky
  using m-mode Analysis*](https://arxiv.org/abs/1401.2095), PRD 91, 083514
  (2015), [DOI 10.1103/PhysRevD.91.083514](https://doi.org/10.1103/PhysRevD.91.083514):
  polarized transfer tensors, spin `+2/-2` sky/beam fields, and the full
  `v_m=B_m a_m` block relation.
- IERS, [*IERS Conventions (2010), Technical Note
  36*](https://www.iers.org/SharedDocs/Publikationen/EN/IERS/Publications/tn/TechnNote36/tn36.pdf),
  Chapters 5–6, and the official [IAU SOFA Earth-attitude
  documentation](https://www.iausofa.org/cookbooks): ERA/UT1 and celestial-to-
  terrestrial rotation.
- K. M. Gorski et al., [*HEALPix: A Framework for High-Resolution
  Discretization and Fast Analysis of Data Distributed on the
  Sphere*](https://arxiv.org/abs/astro-ph/0409513), ApJ 622, 759 (2005),
  [DOI 10.1086/427976](https://doi.org/10.1086/427976), plus the official
  [HEALPix polarization conventions](https://healpix.sourceforge.io/html/intro_HEALPix_conventions.htm):
  pixelization, harmonic transforms, and the source U-convention distinction.
- RadioSim's accepted [SCI-006 east-X convention](sci006_polarization_convention.md)
  and [SCI-007 frame bound](sci007_frame_accuracy_bound.md), grounded in the
  IAU polarization resolution and Hamaker/Bregman RIME convention, remain the
  normative project bridges.

The papers establish the formalism, not this implementation. RadioSim
correctness still requires the analytic, direct, retained-artifact, and
independent-acceptance programme above.

## Acceptance notes (append-only)

**Phase M1 accepted — 2026-08-23.** This commit is `A1`, the phase-M1
independent acceptance; `A1^ == E1
dc736c692e4037e15b7e51253067fa262204bde2` binds the succession, and the
next phase's `R` binds this containing commit per the record's
self-reference rule. The retained succession is operative `D`
`1712575e6c634457d9da737e9c144147e3b9bbc4` (through the ten accepted
bounded corrections and the reopened slices this memo's header records)
`-> R1 8b9d89ee7104cfc118825af119e1167c87239bd9 -> S1
8dfc9af889c5d89f1783ac852f7d0cf6d4589740` (the reimplemented production
slice `1251389e5be7bf5c6b19f6b435cadaefbeb9f295` plus the `E1`-state
commit-shape checks) `-> E1 dc736c692e4037e15b7e51253067fa262204bde2`,
independent reviewer verdict `ACCEPT` (identity
`sci004-m1-independent-acceptance-reviewer`; all eleven Section 14.3
oracles re-derived, including the two deliberate replay-deferral
discharges — the full 16,835,749-row operational horizon-scan replay
against `horizon_scan_ledger_sha256` and the 288-row transfer-sample
concatenation regeneration), acceptance artifact
`sha256:19a8ca668e5cc0e29c54206f14c2cafc123b72e468effede1962db563d012002`.
M1 licenses the scalar `mmode` registry entry only: no polarized
capability, no fingerprint pin, and no speed or accelerator claim.
`SCI-004` remains **ROADMAP**; phase M2 is the next slice.


**Phase M2 accepted — 2026-08-24.** This commit is `A2`, the phase-M2
independent acceptance; `A2^ == E2
50772ec1462c3561e350b46be404c5de9e74b8f7` binds the succession, and the
next phase's `R` binds this containing commit per the record's
self-reference rule. The retained succession is operative `D`
`b9a9d7a8a49974bae4634f24fbc805077cdc4ef8` (through the sixteen accepted
bounded corrections and the discharged reopening this memo's header
records) with the live `R2 27d2ba45db57eed3d86fae04ece8128131d2d10e`
reached over the starred `A1 -> R2` edge and `S2
399245793e812ed549fac23c1b69b2c6c61aecd4` directly parenting the
operative correction commit over the starred `R2 -> S2` edge whose
interval the header enumerates, `-> E2
50772ec1462c3561e350b46be404c5de9e74b8f7`, independent reviewer verdict
`ACCEPT` (identity `sci004-m2-independent-acceptance-reviewer`; all ten
Section 14.3 oracles re-derived, including the identity-`P` and
shared-celestial-basis confirmations that replaced the rescoped
transport sign, the phase-local frame certificate, both replay-deferral
discharges — the horizon-scan ledger digest reproduced by fresh-process
replay and all `288` transfer-sample concatenations exact — the
re-qualified fixture's two-tier gate predicates recomputed from
hand-decoded cells, and the deliberate Tier-7 capability flip
authenticated at exactly its nine licensed lines with the full M1
machinery green), acceptance artifact
`sha256:2de45da5ad35caa2105340ce0870f5370e0f840958b51ac424c938a8fbe4b0dd`.
M2 licenses the full-Stokes `mmode` capability: `supports_polarization`
is `True` under the singular Tier-7 pin, with no accelerator claim, no
ground-anchored transport exercised, and the acceptance fixture a
numerical fixture rather than a buildable array. `SCI-004` remains
**ROADMAP**; phase M3 is the next slice.
