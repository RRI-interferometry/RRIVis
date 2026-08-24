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
contractions and time synthesis. This is recorded as
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
smaller than that minimum is rejected before allocation. Acceptance measures
host peak and, where available, backend-native peak, and proves the estimate is
not smaller than the measured scoped peak. No speed, scaling, or memory
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
grid, harmonic index table, and input identity. The family record's grid
and input-identity digests use the namespaced domains
`radiosim.sci004.characterization-time.v1` and
`radiosim.sci004.characterization-input.v1`, computed from the retained
`SimulationResult` exactly as the strict validator re-derives them;
Section 14.0's solver-internal domains do not apply to a result-derived
record. A changed m-mode pin requires
old/new cubes and an equation-level explanation; no digest is appended merely
because CI printed it.

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
object has exactly `schema_version`, `provenance`, and `workloads`.

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
direct_comparison, backend_comparison, claims_not_licensed
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
beam_transfer, per_m_contraction, synthesis, host_transfer, total,
direct_reference
```

`clock` is `time.perf_counter_ns`; `warmup_iterations` is positive; and the
synchronization methods are respectively `numpy_eager_v1`,
`jax_block_until_ready_v1`, and `dask_compute_v1`. A measured timing series has
exactly `status` and `sample_seconds`, with status `measured` and at least five
finite non-negative samples in execution order. A non-measured series has
exactly `status` and `reason`, where status is `not_applicable` or
`not_measured` and reason is non-empty. No timing-series field is nullable.

`frame`, `sky_transform`, `beam_transfer`, `per_m_contraction`, `synthesis`,
and `total` are measured and have identical sample cardinality and indexed
iterations. `host_transfer` is measured or `not_applicable`.
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
`host_measurement_method` is `python_heap_tracemalloc_scoped_v1`. Native method
is one of `jax_device_memory_stats_v1`, `dask_worker_metrics_v1`,
`process_rss_sampled_delta_v1`, or `unavailable`. Both limitation arrays are
sorted, unique, and non-empty.

The only nullable value anywhere in the benchmark document is
`measured_native_peak_bytes`. If it is an integer, its adjacent reason is
exactly `measured` and the native method is not `unavailable`. If it is null,
its reason is non-empty, the method is `unavailable`, and the same reason
occurs in `native_measurement_limitations`. Host/native peaks are scoped
increments over a synchronized pre-call baseline. Validation requires
`estimate_covers_measured_host_peak` to be true and

```text
measured_host_peak_bytes <= estimated_host_peak_bytes
estimated_host_peak_bytes <= working_memory_bytes
```

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
retained maxima. The NumPy row self-references and has exact-zero deviations.
All counts equal `K`, all cells are finite, and `pass` is true. The separate M2
complex64 row and its Section 9 predicate remain required but are not
substituted into this standard-complex128 performance inventory.

Every workload row carries this exact lexicographically sorted array:

```text
general_speedup
gpu_or_accelerator_support
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
    classes where applicable, exact-SHA remote artifacts, and release scans
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
slice, superseded implementation, superseded evidence, or
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
the applicable supersession-and-regeneration path instead. An accepted
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
superseded phase evidence artifact and its reproduction record and returns
the approved evidence constants to the null sentinels — so the
regenerating `E` runs under the unchanged rule against absent paths and
null sentinels. This disposal is authorized only for an artifact the memo
header records as superseded; the `A` that would have accepted it returned
`REJECT`, so removing it is disposal of a rejected draft, not replacement
of a record. The same rule governs a superseded phase red-failure record:
the governed re-cut deletes and regenerates it, since the record was never
accepted — **unless the reopened phase's `S` already exists**, in which
case redness can no longer be observed anywhere in the operative tree and
the record is not regenerated: the rebind-only re-cut retains the record's
last genuinely observed bytes, and the strict red-record validator
authenticates the record's `design_sha` as a header-enumerated chain
commit from whose tree the observations were genuinely made, connected to
the operative `D` through the chain — never as a licence to fabricate an
`expected-red-confirmed` observation against a tree where nothing is red.
An accepted artifact is immutable and no commit may touch one.

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
header-enumerated correction chain between the two bindings. Every SCI-004
red, evidence, acceptance, dependency, and performance generator and
validator reads the phase-appropriate frozen binding and requires every
`design_sha` field to equal it.

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
`input_identity_sha256`, `canonical_era_grid_sha256`,
`solver_snapshot_sha256`, `cube_sha256`, `scientific_sha256`,
`expected_change_reason`, and `pass`. There are exactly seven rows in the
Section 11 family order, one local retained pin per family; each fixture joins
the phase input set and each identity recomputes under Section 14.0.

Each `ci_artifacts` entry has exactly `family_id`, `fixture_id`, `source_sha`,
`environment`, `dispatch_identity`, `run_id`,
`job_id`, `artifact_id`, `artifact_sha256`, `cube_sha256`,
`scientific_sha256`, `numeric_delta`, `expected_change_reason`,
`ci001_verdict`, and `pass`. Family order is the exact seven-name Section 11
order; for each family, CI rows use the accepted six CI-001 platform/Python cells
and then the accepted dispatch-identity observation order. The validator
reconstructs that complete Cartesian inventory from the CI-001 artifact,
rejects a missing/extra/duplicate family-cell-dispatch tuple, and joins each
fixture ID to the phase input set.

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
`APPROVED_SOURCE_SHA` and `APPROVED_ARTIFACT_SHA256` constants. In that state
the validator requires the artifact's `source_sha` and the constant to equal
the approved `S`, authenticates the raw artifact bytes, locates the unique
artifact-introducing `E` commit and requires its direct parent to be `S`, and
checks the `S..E` diff against Section 13. It deliberately does **not** require
the current checkout or `E` to equal `source_sha`. It re-runs schema validation
and all cheap digest/oracle checks under default and `py312`; it never selects
the first matching file or trusts a workflow summary. The reproduction record
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
acceptance bytes, unique introducing `A` commit, and exact `E..A` authority; it
never requires the evidence artifact's `source_sha` to equal `E`.

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
every fingerprint/remote artifact. Its required oracle IDs additionally
include `m3.sci005-dependency-gate`, `m3.performance-schema`,
`m3.performance-provenance`,
`m3.performance-inventory`, `m3.performance-schedule`,
`m3.performance-timing`, `m3.performance-memory`,
`m3.performance-direct-predicate`, and `m3.performance-backend-predicate`.
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
->* G3 ->* R3 -> S3 -> E3 -> A3 -> C`; the `A1 ->* R2`, `R2 ->* S2`, and
`G3 ->* R3`
stars are the
concrete effects of the header's starred-edge correction records under
the Section 13.7 rule above, which collectively enumerate each such
edge's
interval commits. Each starred edge is inclusive ancestor
reachability through separately authorized, independently accepted programme
commits; every unstarred edge is the sole direct-parent edge. No commit in
either starred first-parent range is a merge. `G1` also has accepted WP-7 CPU
`A` as an authenticated ancestor; `G3` also has accepted SCI-005 Stage-2 `A2`
as an authenticated ancestor. Exact bindings and immutable-byte rules are
Section 13.2's. A Section 13.7 accepted bounded correction that intervenes
at any edge replaces that exact direct-parent edge with a starred edge whose
interval commits the memo header enumerates exhaustively under
Section 13.7's recorded kinds; the reopened phase's `R` then directly
parents the operative correction commit.

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
Section 13.7 rule, as `S2` does here — and
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
characterizations, all six remote environment cells, relevant dispatch
classes, and a green exact-pin-SHA rerun. Every artifact is downloaded and
authenticated rather than inferred from a green workflow summary.

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
