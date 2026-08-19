# SCI-005 staged beam-physics design gate

**WP-8 original design gate — 2026-08-11**

**Accepted bounded Stage-1 numerical-contract correction — 2026-08-11.** The
original design accepted at
`42a1f27e5f6078ce72960f7d200e8b1e94d399c2` remains the base governing record.
The bounded correction superseded its Stage-1 numerical terms in Sections 3,
6, 7.2, 8, and 9 and closed ambiguities found during implementation preflight.
Its exact pre-landing file bytes
(`sha256:7b2b384b39a805c4db051be322cbf5e238b35008fb57e31a257fa179b86422f0`)
and parent-relative diff
(`sha256:51cd7555111a32cf58cef3ba132848c51f690134f08e701a1f4aa218a2e5166a`)
received separate independent physics and computational `ACCEPT` verdicts,
then landed unchanged as
`8935052cc4e49e3ff7bb92f645d03cee6b9e8ad2`.

**Acceptance-succession governance correction — 2026-08-11.** Sections 7--9
also freeze the retained acceptance artifacts, strict validators, direct-parent
bindings, and constant-only validator updates that were missing from the
accepted stage design. This amendment is design-only. Its exact bytes require
fresh independent governance and computational review before landing; landing
it does not accept a production stage.

**Source reviewed:** `e63770c3e27e5aee4e09570c53eb1367099b1ae4`, the
accepted WP-7 design commit. Ambient WP-7 implementation work is not evidence
for this memo and does not change its source anchor.

**Bounded Stage-1 ownership correction — 2026-08-14.** The
acceptance-succession amendment at
`58e7fb3d09dbcaec6f8201a778a653b55996c1aa` received its two required fresh
independent reviews on 2026-08-14 — governance and computational, both
`ACCEPT` — binding its exact parent-relative diff
(`sha256:f987ceb7061ea08f43be092d22342a8ef1c752cababd3ddbff500a21e21dcf40`)
and file bytes
(`sha256:1a7843b892c3b1f975aab663618d7e3afe2b6e06511587ce7e695568bb3ed387`).
Red-test authoring against that accepted design then surfaced one genuine
Section 3.2 ambiguity: the leg-wider-than-resolved-diameter rejection named
no owner reachable under Section 2's taxonomy, because per-antenna diameters
resolve only with the instrument. This correction rules the owner (Section
3.2), grants the one append-only error path that ruling needs (Section 7.2),
and records two Stage-1 mechanics that follow from Section 2 unchanged
(Section 3.5). In this header, *landing* means committing an amendment's
bytes; a landed commit becomes the operative design gate only when its exact
bytes are independently accepted, as Section 7.1 requires. The commit
containing this correction supersedes the amendment commit as the operative
`D1` once accepted. Its exact pre-landing file bytes
(`sha256:2fb7c2f328f343fb6054a183e35aa58a7795f6a5ab6efe3c2ea29660a3aae001`)
and parent-relative diff
(`sha256:3d772d7be5d290bd68e64211c9cb5ffd065ec56e1ca7a430810de26172b3e347`)
received separate fresh independent governance and computational `ACCEPT`
verdicts on 2026-08-14, each reconfirmed after a single-sentence Section 3.5
precision fix, and landed with only this record sentence added.

**Bounded Stage-1 Ruze quadrature-domain correction — 2026-08-15.** Stage-1
red-test authoring stopped on a proof that Section 3.4.1's frozen resource caps
and its own frozen Hermite floor cannot both be satisfied. Independent
re-derivation confirms that report and finds the defect deeper than cap
arithmetic. On the red slice's own oracle fixture the frozen rules retain
Poisson orders `[1,5]`, put the Hermite floor at 71 and therefore the first
allowed order at 128, and force levels 128, 256, and 512, whose 5,160,960
transformed wavevectors are 4.92 times the frozen `2**20` bound; the smallest
allowed order already asks for 265,020 aperture nodes against a 65,536
per-transform bound; and the `correlation_length_m: 0.25` example in Section
3.4 resolves a floor of 325, leaves no room for the two higher allowed orders
the rule demands, and cannot run. Two further findings are new. First, the
frozen floor is insufficient as well as unaffordable: measured against an
independent separation-domain reference, the fixture's largest retained order
needs `H=256`, so the two-consecutive-comparison rule needs 1024, which is not
an allowed order — the fixture could not converge at any cap. Second, the
frozen 65,536 per-transform bound rejects the base coherent transform itself,
which needs 158,400 nodes at its Section 3.3 converged level with no Hermite
shift involved. The cause is the choice of quadrature variable, not the size of
the constants: the shifted-wavevector rule costs `O(S*J*(D/L)**4)` and is
tractable only for `L` of order `D` or larger, which is exactly the regime in
which scattered power degenerates to its own closed `L -> infinity` identity,
while reflector surface errors are panel-correlated with `L << D`. This
correction therefore evaluates the identical frozen mixture in the separation
variable, where the same integral costs `O(1)` in `D/L`. It keeps the Poisson
mixture, its tail rule, the covariance convention, the honesty rules, the
public method signature, and every result field that still has a meaning; it
replaces the Hermite abscissa rules, the `H_floor` rule, the allowed-order set,
and every cap with separation-domain rules and caps derived from them; it
bounds v1 to an unobstructed pupil; and in Section 3.4.2 it drops the four
`hermite_*` convergence fields and adds ten separation fields. Its edits are
confined to Sections 3.4.1 and 3.4.2, the four Section 3.5 bullets and the two
Section 8.1 records that name the superseded method literal, order rule, or
field list, one Section 6 test-content phrase, and one Section 10 citation
entry; it adds no writable path and no new error family beyond the
unsupported-obstruction issue code above. Sections 3.1 through 3.3, the coherent
channel, the public method signature, and the diagnostic's physical contract
are unchanged. Under the corrected rules the scattered power agrees with the
red slice's own independent $O(Q^2)$ pair oracle to $6\times10^{-18}$ at the
shipped fixture and to $2\times10^{-16}$ at a $D/L$ of eight, and the
`L -> infinity` identity $B_{\rm sc}=(1-e^{-\mu})|e_{\rm det}|^2$ holds to
$6\times10^{-15}$. Its exact pre-landing file bytes
(`sha256:09b185c4d741e31d65261c8953755adfc83f34e9ff7ea5cf16a75f00b8e75489`)
and parent-relative diff
(`sha256:f5fab63418b4ab13a1d8303a4d2c356640739f88131fd0780d598d2256dc8fa4`)
received separate fresh independent physics and computational `ACCEPT`
verdicts on 2026-08-15 — the physics review re-derived the domain identity,
error budget, and pupil geometry and agreed with the pair oracle through its
own independent prototype; the computational review reproduced the old rules'
infeasibility and every seed-level constant of the new rules exactly — each
verdict reconfirmed after a three-edit delta resolving the reviews' own
findings, and landed with only this record sentence added.

**Stage-2 design gate and bounded succession corrections — 2026-08-19.**
This amendment is the Stage-2 design gate `D2`. It freezes the complete
normative Stage-2 evidence envelope (Section 8.1), completes Section 4's
strict configuration, rejection, geometry, precision, and ownership contract
(Sections 4.1.1, 4.2.1, and 4.3), and grants the three bounded Stage-2
writable-path additions in Section 7.3. It also lands two bounded succession
corrections that Stage-1 closure proved necessary. First, Section 8.3's
frozen edge `D2^ == U1` is infeasible against the actual repository history:
after `U1` landed at `d6eb4b083d4db704cd31abcf70e22cb745291e15`, the strict
Sphinx gate failed on the changelog heading level `U1` itself introduced, and
the repair landed as `3e336095009e72bf4ae6064d5e97d381e063258f`, a
single-parent commit touching only `docs/changelog.rst` — a Section 7.5
status path — with no source, schema, test, tool, artifact, fingerprint,
tolerance, or acceptance-byte change. History is never rewritten, so the
exact edge is replaced by the starred edge `U1 ->* D2` that Section 8.3 now
defines; the observed interval is exactly that one commit. Second, a
memo-only `D2` was impossible: `tests/unit/test_sci005_stage1_dependency.py`
asserted the *checked-out* memo bytes equal the `D1` blob, an interval-freeze
check whose freeze interval (`D1..G1`, operationally through `U1`) is
complete, and whose persistence would turn the standard gate red at any
later design gate — the outcome `U1`'s own commit message already recorded
as requiring phase-awareness before any memo append. This commit therefore
lands, together with the memo amendment, exactly one bounded edit to that
validator: the four lines of
`test_approved_d1_memo_blob_authenticates_against_the_pinned_digest` that
read and compare the checked-out memo (the two-line comment quoting Section
8.3 and the two statements binding `checked_out`) are deleted, and no other
byte of that file changes. The remaining assertions still authenticate the
exact `D1` blob, diff, ancestry, and header pins from Git objects, which stay
true forever. The `APPROVED_SCI005_D1_SHA` binding literal and every other
dependency-validator byte remain immutable. This two-path landing is the one
authorized exception to the memo-only design-commit form, recorded here and
in Sections 7.1 and 8.3; it implements no beam physics, accepts no stage, and
does not close the register row. Its exact pre-landing memo bytes
(`sha256:5c03896c004d0e557af3c53cbf725e597567d530c3aef9f0fdca59324bd50197`),
dependency-validator bytes
(`sha256:e8d0255b2adb56bd11241e5ba12f655a63282caa161f70782926afe2dceb7cdc`),
and parent-relative diff
(`sha256:71effe56ee49336c306a0a5a27c58ffdbf0d0be0d5fa6972201ceb3066008969`)
received separate fresh independent governance/physics and computational
reviews on 2026-08-19 — each initial review returned exactly one major
finding, both findings were resolved by a five-edit delta frozen into these
bytes, and both reviewers then issued their `ACCEPT` verdicts on exactly
these pins — and landed with only this record sentence added.

**Bounded Stage-2 heading and binding correction — 2026-08-19.** Red-test
authoring against the landed gate surfaced three defects, reported rather
than resolved by the authoring pass, and this memo-only correction repairs
them together with a fourth its own reviews then proved necessary. First, the gate's landing edits consumed the two heading lines
`### 4.2 Factorization into the canonical chain` and
`### 4.3 Stage-2 acceptance invariants` while leaving both sections' prose
intact and unrenumbered; the two lines are restored verbatim at their
original positions, so every existing reference to Sections 4.2 and 4.3
resolves again, and no other prose moves. Second, Section 4.1.1 claimed
`beams.squint` would participate in `SimulationOverrides` "exactly as
`beams.pointing` does" — a false premise, because `SimulationOverrides`
carries no `beams` field at all and `beams.pointing` participates only in
the document-schema known-field hint table; the sentence now names that
real surface and adds no override field. Third, two granted surfaces were
left unnamed: the correction freezes `load_beam_system`'s new keyword-only
parameter as `receptors` (typed `ResolvedReceptorSet | None = None`) and
the resolved record as frozen dataclass `ResolvedSquint` stored at
`ResolvedBeamAssignment.squint`. Fourth, this correction's own independent
reviews proved that a bare supersession sentence would leave Sections 7.1
and 8.3 mechanically unverifiable — the superseded gate commit would sit
inside `U1..D2` and fail the status-prose interval rule — so the
correction also amends Section 8.3's second-starred-edge rule to admit the
header-recorded superseded design commits as the interval's terminal
segment, updates the observed-interval and operative-`D2` file-scope
sentences to the real two-commit history, and re-scopes Section 7.1's
recorded four-line-deletion exception to the superseded gate commit. The
commit containing this correction
supersedes the gate commit as the operative `D2` once its exact bytes are
independently accepted; `R2` then directly parents it. Its exact
pre-landing memo bytes
(`sha256:1d8cdb606558756490d6459131e9bca0b87d2b641749376c829e3cbc05e0ee19`)
and parent-relative diff
(`sha256:fdde54572567ae044fbab37526a4943dc26aae063f583d9fb6f0de126846c72e`)
received separate fresh independent governance and computational reviews on
2026-08-19 — both returned the same blocking succession finding against the
first draft, the Fourth repair above resolved it, and both reviewers then
issued their `ACCEPT` verdicts on exactly these pins — and landed with only
this record sentence added.

**Status:** Stage 1 is accepted and closed as a stage: the operative `D1` is
`c6a5ce90ae3160150b1699f97b45bb693d4ed886`, and the accepted succession is
`R1 e246c5d1c092b5321dddfd9506bb1f2d3ae12365 ->
S1 881b1a963b4f3b250b38989335c2ee0ea2a491bd ->
E1 bbc2b1b4d16bce296c2b6f6597c7c180a70f0f7f ->
A1 2281f2f00576abbc98a0a047ce192ad3013aa202 ->
U1 d6eb4b083d4db704cd31abcf70e22cb745291e15`, with the retained evidence and
acceptance artifacts authenticated by their approved validator constants.
The commit containing this amendment is the operative `D2` once its exact
bytes are independently accepted per Section 7.1. Stage 2 may begin `R2`
only after that acceptance, with `R2^ == D2`, and must complete its own
source, evidence, acceptance, and status succession through `U2`. Stage 3
remains a later, sequential, independently accepted slice behind its own
`D3`. This gate implements no beam physics and accepts no stage. `SCI-005`
remains **ROADMAP** until whole-row closure after Stage 3.

## 1. Ruling and bounded scope

WP-8 is three scientific stages, not one implementation commit:

1. **Scalar-preserving aperture physics:** one normalized aperture transform
   composes central blockage, support shadows, and deterministic Zernike
   surface height. Ruze scattered power is an ensemble-power diagnostic; it is
   not fabricated into a deterministic voltage.
2. **Beam squint:** two native feeds sample oppositely displaced scalar beams.
   The response is diagonal only in native-feed space and is transformed into
   the existing sky-side `E` factor without changing Jones-chain order.
3. **Full cross-polarization:** a complete UVBeam efield response is ingested,
   coordinate- and basis-converted, normalized by one common efield operation,
   and factored through the existing receptor and output-basis contracts.

Near-field simulation remains a permanent non-goal. Station element beams,
array factors, mutual coupling, stochastic unseeded surfaces, calibration,
solving, imaging, and a second forward model remain outside `SCI-005`.

The phenomenological quadrupolar expression in the scope document is useful as
a synthetic analytic oracle, but it is not a complete production beam model:
it lacks a globally specified radial envelope and complex phase. Stage 3 will
therefore use it in tests, not expose it as a configurable beam. IXR is retained
as a diagnostic derived from the singular values of an accepted full efield
Jones matrix. It is not a second leakage configuration surface; the existing
`D` term already owns configured instrumental leakage.

## 2. Invariants shared by all three stages

The following rules are normative:

- `BeamSystem` remains the one canonical beam runtime and `_ResolvedBeamJones`
  remains its private solver adapter. No second beam pathway is introduced.
- The canonical sky-to-correlator product remains
  `H G B Rc Kd X D C E P T Z`, with `E` between `C` and `P`.
- Jones matrices reach `core/contraction.py` fully composed with shape
  `(B,S,2,2)`. The six-input contraction leaf, its keyword-only backend, the
  single `backend.compile(...)` site, and the source-reduction order do not
  change for beam physics.
- Beam interpolation, aperture integration, coordinate conversion, and Ruze
  diagnostics remain host-side. Backend conversion occurs only after a finite,
  immutable Jones batch has been formed.
- No absent or disabled beam block changes the resolved configuration,
  assignment/state/scientific fingerprints, result bytes, logs, or output.
- Every explicitly present block must resolve to a real effect. Exact identity
  blocks are rejected, not accepted and discarded.
- Every field is strict and frozen. Booleans are not integers, integers are not
  silently accepted as strict floats, unknown fields fail, and every numeric
  value is finite before instrument or file resolution.
- Input shape/type/unknown-field failures remain `ConfigSchemaError`; resolved
  identity, domain, and cross-field failures remain `ConfigSemanticError`;
  unsupported beam-family combinations remain `UnsupportedConfigError`. File
  metadata then uses the existing typed `UnsupportedBeamTypeError`,
  `UnsupportedBeamFeedError`, `UnsupportedBeamBasisError`,
  `UnsupportedBeamCoordinateError`, or `BeamNormalizationError` family. Tests
  assert the concrete type and stable issue code, not only a message substring.
- Paths and runtime scheduling choices never enter `scientific_sha256`; file
  content, normalized scientific metadata, resolved physical parameters, and
  convention-version literals do.
- New tolerances are fixed by the design and cannot be authored in YAML.
  Existing point/HEALPix, backend, and output tolerances are not widened.
- Fingerprints change only for a workload that explicitly enables the landed
  effect. A regeneration includes old/new cubes and scientific hashes, not
  digest strings alone.
- A scientific citation appears in the implementation docstring and user
  documentation as well as in this design record.

## 3. Stage 1 — one scalar aperture transform

### 3.1 Physical normalization and composition

For unmodified pupil $\mathcal P_0$, complex illumination $A(\mathbf u)$,
obstruction mask $M(\mathbf u)$, and deterministic surface height
$h(\mathbf u)$, the voltage response is

$$
e(\mathbf q,\lambda)=\frac{1}{N_0}
\int_{\mathcal P_0} A(\mathbf u)M(\mathbf u)
\exp\!\left[-i\frac{4\pi}{\lambda}h(\mathbf u)\right]
\exp(-i\mathbf q\!\cdot\!\mathbf u)\,d^2u,
\qquad
N_0=\int_{\mathcal P_0}A(\mathbf u)\,d^2u.
$$

`N0` is always the unmodified ideal-aperture integral. It is not recomputed
after masking, and the modified beam is not re-peak-normalized. Consequently
blockage and aberration loss occur exactly once in `E`. Blockage and Zernike
are masks/phases inside this one integral; their separately evaluated far-field
patterns must never be multiplied as if Fourier transformation distributed
over aperture multiplication.

The strict parent block carries
`normalization: unmodified_ideal_aperture_v1` and at least one effective
`blockage` or `zernike_surface` child. The normalization literal cannot be
omitted or changed, and a parent whose children together resolve to identity is
rejected.

The aperture axes are `(north, east)`. Aperture azimuth $\varphi=0$ points
north and increases through east, matching RadioSim's topocentric azimuth.
Here $h$ is **aperture-equivalent reflector surface-height error**, defined as
one half of the signed reflected optical-path difference. It is not asserted to
be the literal local-normal displacement of a shaped reflector. To first order,
a physical normal displacement $\delta_n$ at incidence angle $i$ maps to
$h=\delta_n\cos i$; Stage 1 never invents $i$ from the beam model. At normal
incidence $h=\delta_n$. Thus the signed excess path is exactly `2*h`, and
RadioSim's positive-delay convention produces `exp(-i * 4*pi*h/lambda)`.
The convention literal is
`radiosim.real_unit_rms_disk_surface_height.v1`.

Stage 1 v1 supports only reflector-like circular analytic models for which the
existing scalar far field determines one exact compact aperture-plane profile.
Let $R=D/2$, $\rho=|\mathbf u|/R$, $p=10^{-T/20}$ for the existing
`edge_taper_db` value $T$, and define, on $0\leq\rho\leq1$,

$$
U(\rho)=1,\qquad
P(\rho)=2(1-\rho^2),\qquad
P_2(\rho)=3(1-\rho^2)^2.
$$

All three have $\int_0^1 A(\rho)\rho\,d\rho=1/2$. Their normalized Hankel
transforms are respectively $2J_1(x)/x$, $8J_2(x)/x^2$, and
$48J_3(x)/x^3$, including their continuous values at $x=0$. The exact Stage-1
profile table is:

| Existing analytic model | Stage-1 aperture profile $A(\rho)$ |
|---|---|
| `circular_aperture`, `taper.kind: uniform` | $U(\rho)$ |
| `circular_aperture`, `taper.kind: parabolic` | $pU(\rho)+(1-p)P(\rho)$ |
| `circular_aperture`, `taper.kind: parabolic_squared` | $pU(\rho)+(1-p)P_2(\rho)$ |
| `analytical_illumination`, `taper_profile.kind: parabolic` | the preceding parabolic expression, with $T$ equal to the existing derived edge taper |
| `analytical_illumination`, `taper_profile.kind: parabolic_squared` | the preceding parabolic-squared expression, with $T$ equal to the existing derived edge taper |

The profile-set convention literal is
`radiosim.circular_stage1_pupil_profiles.v1`; it enters the scientific
fingerprint whenever either Stage-1 feature is explicit.

The table deliberately gives $p$ its existing mixture-weight meaning; it does
not relabel $p$ as the ratio $A(1)/A(0)$. Its normalized analytic Hankel
transform is therefore the current unmodified scalar response; implementation
must recover that response within the existing dtype tolerance before adding a
mask or phase. An analytical model is supported only when its already derived
$T$ is finite and non-negative.

The current direct and derived `gaussian` far-field shortcut does not uniquely
specify a compact disk pupil, and the current direct `cosine` shortcut likewise
has no declared radial-pupil inverse. The accepted `numerical_illumination`
response is a fixed 256-node trapezoidal Hankel rule, not the continuum
transform of a uniquely retained compact-pupil discretization; replacing that
rule when Stage 1 is enabled would change accepted beam physics. Stage 1 v1
therefore rejects
`circular_aperture` with `gaussian` or `cosine`, and
`analytical_illumination` with a `gaussian` taper profile, as well as every
`numerical_illumination` model, using `UnsupportedConfigError` and stable issue
code
`beam.aperture_physics.unsupported_pupil_profile` when `aperture_physics` is
present, or `beam.ruze_power_diagnostic.unsupported_pupil_profile` when only
the diagnostic is present. If both are present, aperture validation runs first
and owns the issue path. These rejections occur only when one of those two
features is explicit; no existing beam with both absent is re-resolved or
changed. Applying a reflector blockage, disk Zernike map, or Ruze diagnostic to
`rectangular_aperture`, `elliptical_aperture`, or a FITS beam is rejected with
the same exception and respective exact issue code
`beam.aperture_physics.unsupported_beam_family` or
`beam.ruze_power_diagnostic.unsupported_beam_family`, under the same
aperture-first ordering. A FITS file already contains its aperture physics;
applying it again would double count an effect that cannot be separated from
the file.

The emitted `ConfigIssue.path` is exactly `beams.aperture_physics` for the
aperture-owned cases. A diagnostic-owned issue uses the exact authored path
`beams.surface_error.default.error_beam_diagnostic` or
`beams.surface_error.per_antenna[i].error_beam_diagnostic`, with the resolved
zero-based `i`. Unsupported-profile messages use exactly
`Stage-1 {feature} requires a canonical circular pupil; resolved model
{model_kind!r} with taper {taper_kind!r} has no supported v1 profile.`, and
unsupported-family messages use exactly
`Stage-1 {feature} does not support resolved beam family {model_kind!r}.`,
where `feature` is the lower-case literal `aperture physics` or
`Ruze power diagnostic`, and an absent taper is rendered as Python `None`.

The deterministic aperture transform retains every resolved real/complex
precision already supported by the analytic beam. Its target-width quadrature
nodes, weights, and accumulation may not pass through float64 when the resolved
dtype is wider. The optional Ruze power diagnostic is narrower in v1: it
supports only `float32`/`complex64` and `float64`/`complex128`. Configuring it
under an extended-precision beam raises `UnsupportedConfigError`, issue code
`beam.ruze_power_diagnostic.unsupported_precision`, and exact message
`Ruze power diagnostics support only float32/complex64 and
float64/complex128 beam precision.` The same beam without the nested diagnostic
retains its existing extended-precision behavior.

### 3.2 Blockage geometry

The strict `beams.aperture_physics.blockage` block contains:

- `central_diameter_ratio`: an exact finite float with `0 < epsilon < 1`;
- `support_legs`: an exact tuple of zero or more legs; and
- for each leg, a unique `position_angle_deg` in `(-180,180]` and a positive
  finite `width_m`.

A leg is the closed radial strip of physical width `width_m` from the edge of
the central shadow to the ideal pupil edge, centred on its mechanical position
angle. Masks combine by set union, so overlapping shadows are removed once.
At instrument resolution a leg wider than the resolved aperture diameter is
rejected. That rejection is owned by beam-assignment resolution:
`core/beam/resolution.py` compares each authored `width_m` against each
assigned antenna's resolved aperture diameter and raises the typed
`InvalidBeamGeometryError` — a new `BeamAssignmentError` subclass added
append-only to `core/beam/errors.py` — whose message names the leg's position
angle, the antenna, the authored width, and the resolved diameter.
Document-stage validation does not own this check because per-antenna
diameters exist only after instrument resolution; Section 2's
`ConfigSemanticError` mapping covers document-resolved failures and is
unchanged by this ownership ruling. No scattering or phase from a support
structure is claimed; Stage 1 models only the geometrical aperture shadow.

The mask is not left to a drawing convention. For aperture coordinates
$\mathbf u=(n,e)$ in metres, $R=D/2$, $r=\|\mathbf u\|$, normalized blockage
diameter $\epsilon$, and leg angle $\beta$ measured North through East, define

$$
\mathbf d_\beta=(\cos\beta,\sin\beta),\qquad
\mathbf p_\beta=(-\sin\beta,\cos\beta),
$$

$$
\begin{aligned}
\mathcal P_0&=\{\mathbf u:r\leq R\},\\
\mathcal C_\epsilon&=\{\mathbf u:r\leq\epsilon R\},\\
\mathcal L(\beta,w)&=\{\mathbf u\in\mathcal P_0:
\epsilon R\leq r\leq R,
\ \mathbf u\!\cdot\!\mathbf d_\beta\geq0,
\ |\mathbf u\!\cdot\!\mathbf p_\beta|\leq w/2\},\\
M(\mathbf u)&=\mathbf 1_{\mathcal P_0\setminus
(\mathcal C_\epsilon\cup\bigcup_j\mathcal L(\beta_j,w_j))}.
\end{aligned}
$$

The corresponding scientific-convention literal is
`radiosim.central_disk_outward_half_strip_ne.v1`.

Thus a leg is one outward half-strip, not an infinite chord and not a pair of
diametrically opposed legs. A physical structure on both sides is authored as
two records separated by 180 degrees. Closed-set boundaries are fixed as
written and have zero continuum measure; all numerical methods use the same
inequalities. `position_angle_deg` is converted once to radians after its
canonical interval resolution. A duplicate means the same resolved angle, not
an antipodal angle. The central disk is unioned with every leg before the
single mask is evaluated, so its shared boundary and leg overlaps cannot be
double-counted.

For uniform illumination, no support legs, and
$x=\pi D\sin\theta/\lambda$, the exact oracle is

$$
e_\epsilon(x)=\frac{2[J_1(x)-\epsilon J_1(\epsilon x)]}{x},
\qquad e_\epsilon(0)=1-\epsilon^2,
\qquad \eta_b=(1-\epsilon^2)^2.
$$

For radial taper $A(\rho)$ the boresight loss is illumination-weighted:

$$
e_\epsilon(x)=
\frac{\int_\epsilon^1 A(\rho)J_0(x\rho)\rho\,d\rho}
     {\int_0^1 A(\rho)\rho\,d\rho}.
$$

It is not generally `(1-epsilon**2)` for tapered illumination. The
implementation must recover the closed uniform formula before it may land.

### 3.3 Real unit-RMS Zernike surface convention

`beams.aperture_physics.zernike_surface` has exactly two fields:
`convention` and `modes`. The latter is a non-empty YAML sequence resolved to
an immutable tuple. Every mode is a strict record with exactly the three keys
`n`, `m`, and `surface_height_coefficient_m`; the first two are exact Python
integers (not booleans), and the coefficient is an exact finite Python float.
For example:

```yaml
zernike_surface:
  convention: radiosim.real_unit_rms_disk_surface_height.v1
  modes:
    - n: 2
      m: 0
      surface_height_coefficient_m: 0.0005
```

No pair-valued `mode`, Noll/OSA index, map/dictionary shorthand, or additional
mode field is accepted. Validation requires

$$
0\leq n\leq32,\qquad |m|\leq n,\qquad n-|m|\;\text{even}.
$$

Duplicate `(n,m)` pairs are rejected. Piston `(0,0)` and tip/tilt `(1,-1)` and
`(1,1)` are rejected because delay and deterministic pointing already own
those effects. An individual zero coefficient is permitted beside a non-zero
sibling, but an all-zero block is an exact identity and is rejected. Resolution
sorts modes by `(n,m)` for a stable fingerprint without changing the exact
mathematical sum.

The radial polynomial is

$$
R_n^{|m|}(\rho)=
\sum_{s=0}^{(n-|m|)/2}
\frac{(-1)^s(n-s)!}
{s!\left(\frac{n+|m|}{2}-s\right)!
\left(\frac{n-|m|}{2}-s\right)!}\rho^{n-2s},
$$

and the real basis is

$$
Z_n^m(\rho,\varphi)=
\begin{cases}
\sqrt{n+1}\,R_n^0(\rho), & m=0,\\
\sqrt{2(n+1)}\,R_n^m(\rho)\cos(m\varphi), & m>0,\\
\sqrt{2(n+1)}\,R_n^{|m|}(\rho)\sin(|m|\varphi), & m<0.
\end{cases}
$$

It obeys

$$
\frac{1}{\pi}\int_0^{2\pi}\int_0^1
Z_n^m Z_{n'}^{m'}\rho\,d\rho\,d\varphi
=\delta_{nn'}\delta_{mm'}.
$$

Thus each coefficient is signed aperture-equivalent reflector surface-height
error in metres--one half of reflected OPD--in a unit-RMS **unobscured disk**
basis. It is a literal physical normal displacement only at normal incidence.
After a blockage mask is applied these ordinary disk functions cease to be
orthogonal over the transmitting annulus; the configuration must not describe
the quadrature sum of coefficients as the RMS over that annulus. No Noll or OSA
single integer is accepted, and neither full OPD nor an unprojected off-axis
normal-displacement coefficient is accepted under the field.

The upper radial order `32` is a v1 computation bound, not a statement that
higher physical modes do not exist. The same aperture transform evaluates mask
and phase together. Its only v1 numerical-method literal is
`boundary_fitted_polar_gauss_legendre_v1`. For every supported profile,
$N_0=\pi R^2$ analytically; it is never re-estimated from numerical nodes. In
normalized coordinates the production integral is

$$
e(\mathbf q)=\frac{1}{\pi}
\sum_p\int_{\rho_{p,0}}^{\rho_{p,1}}A(\rho)\rho
\sum_k\int_{\Phi_k(\rho)}
\exp\{-i[\kappa h(\rho,\varphi)+
\rho(Q_N\cos\varphi+Q_E\sin\varphi)]\}
\,d\varphi\,d\rho,
$$

where $Q_{N,E}=Rq_{N,E}$, $\kappa=4\pi/\lambda$, and
$\Phi_k(\rho)$ are the disjoint transmitting angular intervals. This form is
also the one deterministic transform reused by the Ruze diagnostic below.
For an outer resolved beam-frame direction,
$q_N=(2\pi/\lambda)\cos(\mathrm{alt})\cos(\mathrm{az})$ and
$q_E=(2\pi/\lambda)\cos(\mathrm{alt})\sin(\mathrm{az})$. The existing
pointing transform and true-horizon gate remain exactly where they are in the
canonical beam runtime.

The integration panels fit every hard boundary. With $a_j=w_j/D$, leg $j$
blocks the periodic angular interval
$[\beta_j-\alpha_j(\rho),\beta_j+\alpha_j(\rho)]$, where

$$
\alpha_j(\rho)=
\begin{cases}
\pi/2,&\rho\leq a_j,\\
\operatorname{atan2}\!\left(a_j,\sqrt{\rho^2-a_j^2}\right),&\rho>a_j.
\end{cases}
$$

The intervals are split at zero, mapped into $[0,2\pi)$, sorted by their exact
target-dtype endpoints, and unioned before their complement is integrated.
There is no merge tolerance. The radial panels begin at $\epsilon$ when a
central blockage exists and at zero otherwise. They split at one, every
in-domain saturation radius $a_j$, and every in-domain support-topology radius
where, for circular separation $\delta_{ij}\in[0,\pi]$,

$$
\delta_{ij}=\alpha_i+\alpha_j
\quad\hbox{or}\quad
\delta_{ij}=|\alpha_i-\alpha_j|,
$$

as well as every radius where an endpoint crosses the fixed periodic cut,

$$
\beta_j-\alpha_j(\rho)=0\pmod{2\pi}
\quad\hbox{or}\quad
\beta_j+\alpha_j(\rho)=0\pmod{2\pi}.
$$

Consequently the ordered non-wrapping interval topology and interval count are
constant on the interior of every radial panel.

Topology roots are isolated on the analytic segments formed by the saturation
radii. If a root is exactly representable in the resolved real dtype, that
value is its breakpoint. Otherwise target-dtype bisection continues to adjacent
representable bracketing values $(\rho_-,\rho_+)$ with the pre-root topology at
$\rho_-$ and post-root topology at $\rho_+$; the canonical breakpoint is always
$\rho_+$. Radial panels are left-closed/right-open at every internal breakpoint
and the final panel is right-closed at one, so the rounded topology root belongs
to the post-root panel. Gauss-Legendre never samples a panel endpoint. Two
unequal mathematical root values, or an unequal root and authored/saturation
boundary, that resolve to the same canonical breakpoint raise
`BeamSamplingDerivationError`. Symmetric configurations may make different
leg-pair events share the same mathematical root value; certified-equal values
are deduplicated together with repeated discovery of one event. Immediately
above a saturation radius $a_j$, the panel uses
$\rho=a_j+(\rho_{\rm hi}-a_j)t^2$, $0\leq t\leq1$, including its Jacobian, so
the square-root endpoint behavior is not presented to Gauss-Legendre as a
smooth function. A width ratio that underflows, a distinct boundary or topology
root that collides under the preceding rule, or a non-finite/unsortable endpoint
raises `BeamSamplingDerivationError`; a thin leg can never disappear because a
midpoint missed it.

Initial quadrature order is derived from both polynomial and phase bandwidth.
For $N_{nm}=\sqrt{n+1}$ when $m=0$ and
$N_{nm}=\sqrt{2(n+1)}$ otherwise, define

$$
H_\rho=\sum_{nm}|c_{nm}|N_{nm}n^2,\qquad
H_\varphi=\sum_{nm}|c_{nm}|N_{nm}|m|,
$$

using the bounds $|R_n^{|m|}|\leq1$ and
$|dR_n^{|m|}/d\rho|\leq n^2$. For the complete direction batch,
$n_{\max}=m_{\max}=0$ when there is no Zernike child; otherwise they are the
largest $n$ and $|m|$ in the resolved modes. Then

$$
Q_{\max}=\max_s R\sqrt{q_{N,s}^2+q_{E,s}^2},\qquad
B_\rho=Q_{\max}+\kappa H_\rho,\qquad
B_\varphi=Q_{\max}+\kappa H_\varphi.
$$

For the seed below, $L_p$ is an exact bound on coordinate travel in the
quadrature parameter:

- an ordinary radial panel $\rho\in[\rho_0,\rho_1]$ uses
  $L_p=\rho_1-\rho_0$ and $B=B_\rho$;
- a saturation panel
  $\rho=a+(\rho_1-a)t^2$, $t\in[0,1]$, uses
  $L_p=\max|d\rho/dt|=2(\rho_1-a)$ and $B=B_\rho$; and
- a non-wrapping transmitting angular interval
  $\varphi\in[\varphi_0,\varphi_1]$ uses
  $L_p=\varphi_1-\varphi_0$ in radians and $B=B_\varphi$.

Zero-length panels/intervals are absent, not evaluated. On each positive panel
the seed is exactly

$$
\operatorname{order}(B,L_p,d)=
\max\!\left(16,2(d+1),
8+\left\lceil\frac{8BL_p}{\pi}\right\rceil\right),
$$

with $d=n_{\max}$ for both ordinary and transformed radial panels and
$d=m_{\max}$ angularly. Thus an authored
frequency, diameter, or surface coefficient cannot create an unseeded
far-field or phase oscillation. Gauss-Legendre nodes and weights are generated
at `ceil(-log10(eps))+16` decimal digits, rounded once to the resolved dtype,
and rejected unless they are finite, symmetric, strictly ordered, positive in
weight, and sum to two within `32*eps`. Wider-than-float64 beams never reuse
float64 nodes or accumulation.

At fixed radial order, all angular panels are converged first by doubling their
orders and comparing the complete direction array. Only then is radial order
doubled and the two angularly converged arrays compared. Each dimension needs
two consecutive successful comparisons under
`atol + rtol*max(abs(refined))`; the fixed values remain
`atol=max(1e-12,32*eps)` and `rtol=max(1e-10,32*eps)`. At most four doublings
per dimension are permitted. Before an initial evaluation or doubling, a
per-panel order above 4096, more than `2**24` quadrature nodes per direction,
more than `2**28` direction-node phase evaluations, or a conservative peak
workspace above `2**31` bytes raises `BeamSamplingDerivationError` before
allocation. For one nested aperture evaluation, let radial node $a$ in radial
panel $p$ have $K_{pa}$ disjoint transmitting intervals with current angular
orders $n_{pak}$. The exact two-dimensional node count is

$$
Q=\sum_p\sum_{a=1}^{n_{\rho,p}}
\sum_{k=1}^{K_{pa}}n_{pak}.
$$

The sums run only over topology panels with a positive-measure transmitting
complement. A fully blocked panel is proven zero from its exact interval union
and receives no radial or angular quadrature nodes. Thus `Q=0` only when the
entire transmitting pupil is empty, in which case the aperture transform is
exact `0+0j` with positive-zero components and zero refinement residuals. `Q`
is shared by a
direction batch but is the value governed by the `2**24` per-transform cap. A
batch of $B$ wavevectors consumes exactly $BQ$ direction-node phase evaluations
at that refinement; the `2**28` cap applies to the cumulative sum across every
seed and refinement in the public call. For current node count $Q$, direction
count $S$, and internal
direction batch $B\leq\min(S,256)$, the exact conservative estimate is

$$
E_{\rm aperture}=
r(16Q+12B+8S)+c(4BQ+8B+4S),
$$

where $r$ and $c$ are the resolved real/complex byte widths. The implementation
must reuse buffers within those declared multiplicities; needing another live
buffer requires a design correction. $B$ is the largest power of two that
satisfies the phase-product and byte caps, with a smaller final remainder.
`S=0` returns the existing correctly shaped empty Jones batch without invoking
quadrature. Counts and tolerances are internal and cannot be authored in YAML.
No one-pair or unconverged non-empty result is returned.

Red tests must show disk orthonormality, the exact defocus/conjugation relation
under sign reversal, the uniform blocked Airy invariant, support intervals and
overlaps that cannot be missed by nodes, high-$q$ and high-surface-phase seed
growth, target-width node generation, two-pass convergence, and every typed
pre-allocation or geometry-representability failure under this production rule.

### 3.4 Ruze coherent loss and scattered-power diagnostic

The existing `beams.surface_error` field keeps its accepted coherent-voltage
meaning. In this contract its RMS is likewise aperture-equivalent
surface-height error (half reflected OPD), not an automatically incidence-
corrected physical normal displacement. With RMS $\sigma_h$ and
$s=4\pi\sigma_h/\lambda$,

$$
\langle e\rangle=e^{-s^2/2}e_{\rm deterministic},
\qquad
B_{\rm coherent}=e^{-s^2}|e_{\rm deterministic}|^2.
$$

Stage 1 may add an optional nested diagnostic:

```yaml
surface_error:
  default:
    rms_surface_error_m: 0.001
    error_beam_diagnostic:
      kind: gaussian_covariance_power
      correlation_length_m: 0.25
```

This fragment assumes the assigned analytic beam uses one of Section 3.1's
supported pupil profiles. Adding it to the existing default Gaussian beam is a
typed unsupported-pupil error, not an implicit change of taper.

The literal `gaussian_covariance_power` specifies a real, zero-mean, jointly
Gaussian, second-order stationary aperture-equivalent surface-error field
$\delta h(\mathbf r)$ with

$$
\operatorname{Cov}[\delta h(\mathbf r),\delta h(\mathbf r')]
=\sigma_h^2\rho_h(\mathbf r-\mathbf r'),\qquad
\rho_h(\boldsymbol\Delta)=\exp[-(|\boldsymbol\Delta|/L)^2].
$$

Thus `L` is its one-over-e correlation length and the Gaussian characteristic
function, rather than covariance alone, licenses the mutual-coherence kernel
below. The result literal `gaussian_one_over_e_surface_covariance_v1` names this
complete jointly Gaussian field law plus covariance kernel, not merely the
radial function. Here `rms_surface_error_m` is $\sigma_h$, the pointwise standard
deviation of the random residual after the configured deterministic Zernike
map, not a value inferred from those coefficients, and
$\phi_{\rm det}(\mathbf r)=4\pi h_{\rm det}(\mathbf r)/\lambda$. The
ensemble-average power therefore includes the deterministic phase difference:

$$
\langle |e(\mathbf q)|^2\rangle=\frac{1}{|N_0|^2}
\iint A(\mathbf r)M(\mathbf r)
A^*(\mathbf r')M(\mathbf r')
\exp[-i\mathbf q\cdot(\mathbf r-\mathbf r')]
\exp\{-i[\phi_{\rm det}(\mathbf r)-\phi_{\rm det}(\mathbf r')]\}
\exp\{-s^2[1-\rho_h(\mathbf r-\mathbf r')]\}
\,d^2r\,d^2r'.
$$

The diagnostic reports coherent-main power, total ensemble power, and their
non-negative scattered difference at every direction passed to the public
method. Stage 1 never collapses those directions to a radial profile, even when
the model is rotationally symmetric. It does not enter a cross-correlation
Jones matrix. For independent antenna surfaces,
`<e_p e_q*> = <e_p><e_q*>` when `p != q`; the scattered error power belongs to
the same-surface second moment. Taking `sqrt(B_main+B_error)` would invent a
phase and perfectly correlated structure, so that operation is forbidden.

#### 3.4.1 Required positive covariance-mixture algorithm

Direct evaluation of the displayed double integral is forbidden in production:
for $Q$ aperture nodes it would form $O(SQ^2)$ node pairs. A binary Cartesian
raster and local FFT interpolation are also forbidden: their hard-disk and
off-grid errors cannot satisfy the unchanged beam tolerance at a tractable
grid size. The only Stage-1 production method literal is
`poisson_paired_pupil_separation_v1`, defined here.

Let $\mu=s^2$. The scattered covariance is the positive Poisson mixture

$$
K_{\rm sc}(\boldsymbol\Delta)=
e^{-\mu}\sum_{m=1}^{\infty}
\frac{\mu^m}{m!}
\exp\!\left(-\frac{m|\boldsymbol\Delta|^2}{L^2}\right).
$$

For every integer $m\geq1$,

$$
\frac{1}{\pi}\int_{\mathbb R^2}e^{-|\mathbf t|^2}
\exp\!\left(i\frac{2\sqrt m}{L}
\mathbf t\!\cdot\!\boldsymbol\Delta\right)d^2t
=\exp\!\left(-\frac{m|\boldsymbol\Delta|^2}{L^2}\right).
$$

With the negative-forward deterministic aperture transform from Section 3.3,
the scattered power is therefore

$$
B_{\rm sc}(\mathbf q)=e^{-\mu}
\sum_{m=1}^{\infty}\frac{\mu^m}{m!}
\frac{1}{\pi}\int_{\mathbb R^2}e^{-|\mathbf t|^2}
\left|e_{\rm det}\!\left(
\mathbf q-\frac{2\sqrt m}{L}\mathbf t\right)\right|^2d^2t.
$$

That displayed form fixes the physics and proves the bounds below, but it is
not the production evaluation. Applying the same Gaussian identity to the
$\mathbf t$ integral instead of to $\boldsymbol\Delta$ returns each mixture
term to the separation variable exactly. With $f=AMe^{-i\phi_{\rm det}}$,
$\ell_m=L/\sqrt m$, and the pupil autocorrelation

$$
C(\boldsymbol\Delta)=\int f(\mathbf r)f^*(\mathbf r-\boldsymbol\Delta)\,d^2r,
\qquad
C(-\boldsymbol\Delta)=C^*(\boldsymbol\Delta),
$$

every term obeys the identity

$$
P_m(\mathbf q)\equiv
\frac{1}{\pi}\int_{\mathbb R^2}e^{-|\mathbf t|^2}
\left|e_{\rm det}\!\left(\mathbf q-\tfrac{2\sqrt m}{L}\mathbf t\right)
\right|^2d^2t
=\frac{1}{|N_0|^2}\int_{\mathbb R^2}
C(\boldsymbol\Delta)\,e^{-i\mathbf q\cdot\boldsymbol\Delta}\,
e^{-|\boldsymbol\Delta|^2/\ell_m^2}\,d^2\Delta,
$$

with $B_{\rm sc}=e^{-\mu}\sum_m(\mu^m/m!)P_m$ unchanged. Production evaluates
the right-hand side. The two sides are the same integral, so no physical
content, covariance law, or convention literal changes with the quadrature
variable.

The shifted-wavevector side is forbidden in production because it cannot be
evaluated inside any tractable cap. As a function of $\mathbf t$ its integrand
is entire of exponential type $2\sqrt m D/L$, since $C$ vanishes for
$|\boldsymbol\Delta|>D$; a tensor Gauss-Hermite rule resolves that content only
once $\sqrt{2H}$ reaches that type, so the order grows as $mD^2/L^2$, and each
retained abscissa then needs one aperture transform at
$R\|\mathbf k\|$ of order $\sqrt m D/L$, whose Section 3.3 node count grows as
the square of that argument. The work is $O(SJ(D/L)^4)$, which admits only
$L$ of order $D$ or larger, and there $B_{\rm sc}$ degenerates to the closed
$L\to\infty$ identity. In the separation variable the same integral costs
$O(1)$ in $D/L$: $C$ carries no far-field oscillation, the Gaussian confines
$|\boldsymbol\Delta|$ to a few $\ell_m$, and one $C$ array serves every
retained order and every requested direction.

The internal paired-pupil helper accepts every finite real separation
two-vector and never requires it to correspond to a physical sky direction. It
uses the same boundary-fitted panels, phase-bandwidth seeds, convergence, and
target-width accumulation as Section 3.3, but never calls `evaluate_jones` and
never applies a sky angular-domain or horizon check. The outer requested
directions alone receive the existing pointing transform and true-horizon gate,
and they alone enter the coherent transform $e_{\rm det}(\mathbf q)$.

Every supported $U/P/P_2$ mixture has $A\geq0$ and $N_0=\int A>0$; the mask is
zero/one and the deterministic phase has unit modulus. Consequently

$$
|e_{\rm det}(\mathbf k)|\leq
\frac{\int AM}{\int A}\leq1
$$

for every real two-vector $\mathbf k$, shifted or requested. This both proves
non-negative scattered power and
makes omitted Poisson probability mass a rigorous absolute power-error bound:
the Gaussian in the shifted form has unit mass, so
$0\leq P_m(\mathbf q)\leq1$ for every retained or omitted order. The same
hypotheses bound the separation integrand, because the integral triangle
inequality with Tonelli's theorem gives

$$
\int_{\mathbb R^2}|C(\boldsymbol\Delta)|\,d^2\Delta
\leq\left(\int AM\right)^2\leq|N_0|^2 .
$$

Those two inequalities are configuration-free and are the only inputs the
truncation budget below needs; neither is an asymptotic estimate.

Stage-1 v1 evaluates $C$ only for an unobstructed pupil. When the resolved
antenna also carries an `aperture_physics.blockage` child, the paired region is
the intersection of two shifted copies of Section 3.2's mask, which needs a
second boundary family and a second topology-root family that this version does
not freeze. That combination raises `UnsupportedConfigError` with stable issue
code `beam.ruze_power_diagnostic.unsupported_obstruction`, the diagnostic-owned
`ConfigIssue.path` of Section 3.1, and exact message
`Stage-1 Ruze power diagnostics v1 require an unobstructed pupil; the
resolved aperture physics declares a blockage.` Nothing else is refused: the
coherent `surface_error` loss, the blockage mask in `E`, and every other
Stage-1 feature keep their accepted behaviour, and only the optional nested
diagnostic is unavailable for that antenna. Within one diagnostic path, this
obstruction rejection runs first, before the missing-RMS, unsupported-precision,
and empty-direction rejections, extending Section 3.5's fixed order by name. A
later design may add the paired obstructed region; it is not authorized here.

The Poisson support is a contiguous integer interval
`[poisson_first_order, poisson_last_order]`. With
$p_m=\exp[-\mu+m\log\mu-\lgamma(m+1)]$, resolution chooses the fewest retained
terms, breaking a tie toward the smaller first order, for which

$$
\sum_{m=1}^{m_{\rm first}-1}p_m+
\sum_{m=m_{\rm last}+1}^{\infty}p_m\leq
\tau_P,\qquad \tau_P=\mathrm{atol}/8.
$$

Lower and upper tails are evaluated independently with complemented Poisson
CDFs; retained log-weights are generated outward from the Poisson mode and
summed with `fsum`. Production never evaluates
`exp(-mu) * mu**m / factorial(m)` directly and never renormalizes retained
weights. The total scattered mass is computed as `-expm1(-mu)`. If it is no
larger than $\tau_P$, the exact resolved interval is `[0,0]`, its term count is
zero, and scattered power is positive zero with the whole mass recorded as the
omitted upper bound. More than 256 retained terms raises
`BeamSamplingDerivationError` before any aperture evaluation.

Every retained order shares one separation partition. With
$\ell_{\rm wide}=L/\sqrt{m_{\rm first}}$ and
$\ell_{\rm narrow}=L/\sqrt{m_{\rm last}}$, the separation radius is truncated
at

$$
\delta_{\rm cut}=\min\!\left(D,\;
\ell_{\rm wide}\sqrt{\ln(1/\tau_S)}\right),
\qquad \tau_S=\mathrm{atol}/8 .
$$

Because $C$ vanishes beyond $|\boldsymbol\Delta|=D$, because
$\ell_m\leq\ell_{\rm wide}$ for every retained order, and because
$\int|C|\leq|N_0|^2$, the discarded part of each term is at most
$\exp(-\delta_{\rm cut}^2/\ell_m^2)\leq\tau_S$, while the retained Poisson
weights sum to at most $1-e^{-\mu}\leq1$. Separation truncation therefore costs
at most $\tau_S$ of absolute power, by exact arithmetic on those two displayed
inequalities. With the Poisson tail $\tau_P$ the two truncations sum to at most
`atol/4`, and the convergence comparisons below carry the rest of the same
`atol` budget. `separation_cut_m` and `separation_omitted_bound` retain that
cut and that realized bound; neither is assumed, and the bound is exactly zero
whenever $\delta_{\rm cut}=D$, where $C$ itself vanishes and nothing is
discarded.

The separation partition is polar and boundary fitted. Its radial panels are
$[0,\delta_{\rm cut}]$ when $\delta_{\rm cut}<D$, and otherwise $[0,D/2]$ plus
a transformed panel $\delta=D-(D/2)(1-t)^2$, $0\leq t\leq1$, including its
Jacobian, because $C$ reaches its outer zero like $(D-\delta)^{3/2}$ and that
endpoint must not be presented to Gauss-Legendre as a smooth function. Radial
nodes use Section 3.3's generation, validation, and seed rule with $L_p$ the
panel length in metres, $d=2(n_{\max}+4)$, and

$$
B_\delta=q_{\max}+\frac{2}{\ell_{\rm narrow}}
+\frac{2\kappa(H_\rho+H_\varphi)+2}{R},
\qquad
q_{\max}=\max_s\sqrt{q_{N,s}^2+q_{E,s}^2},
$$

since illumination and surface phase each enter the paired integrand twice and
every supported profile has radial degree at most four.

At each separation radius the $\psi$ integral is periodic and analytic, so it
uses the equispaced trapezoid rule whose order is the smallest power of two not
below

$$
\max\!\left(16,\;
8+\left\lceil4\left(q_{\max}\delta
+2\kappa(H_\rho+H_\varphi)\delta/R+m_{\max}\right)\right\rceil\right).
$$

For the direction factor the trapezoid's aliasing error is exactly
$2\sum_{j\geq1}|J_{jN}(q\delta)|$ by Poisson summation, and
$|J_n(x)|\leq(x/2)^n/n!$ (DLMF 10.14.4; Abramowitz and Stegun 9.1.62) makes
that superexponentially small once $N\geq3q\delta$; for the analytic periodic
factors the same rule converges exponentially (Trefethen and Weideman 2014,
DOI 10.1137/130932132). The rule also nests, so a doubled angular level re-uses
every separation node already evaluated.

$C$ itself is one Section 3.3 aperture integral carrying no wavevector. For
$\boldsymbol\Delta=\delta(\cos\psi,\sin\psi)$ and $b=|1-\delta/R|$, the paired
transmitting set in the unshifted pupil's polar coordinates is

$$
\rho\in[\max(0,\delta/R-1),1],
\qquad
|\varphi-\psi|\leq\Phi(\rho)=
\arccos\!\left(\frac{R^2\rho^2+\delta^2-R^2}{2R\rho\delta}\right),
$$

with $\Phi=\pi$ when that argument is at most $-1$ and no transmitting interval
when it is at least $+1$. The radial panels are $[0,b]$, present only when
$\delta<R$, where the whole circle transmits, and $[b,1]$ under Section 3.3's
transformation $\rho=b+(1-b)t^2$ with its Jacobian, because $\Phi$ meets that
boundary with square-root behaviour. There is exactly one transmitting angular
interval at every interior radius, so this partition has no topology roots and
no merge tolerance; $b$ and the domain ends are its only breakpoints and are
canonicalized by the Section 3.3 rule. At $\delta=D$ the paired region is empty
and $C$ is exact `0+0j` with positive-zero components. The paired integrand is
$A(\rho)A(\rho')\exp\{-i\kappa[h(\rho,\varphi)-h(\rho',\varphi')]\}$, where
$(\rho',\varphi')$ are the polar coordinates of
$\mathbf r-\boldsymbol\Delta$; it carries no far-field oscillation, so its seed
uses Section 3.3's rule with $B=2\kappa(H_\rho+H_\varphi)$ in both dimensions,
$L_p$ the panel length radially and $2\Phi(\rho)$ angularly, and
$d=2(n_{\max}+4)$ radially and $d=2(m_{\max}+1)$ angularly. The real Zernike
basis is a Cartesian polynomial, so its normalized gradient is bounded by
$H_\rho+H_\varphi$ at each of the two points and that seed bound holds even
where the shifted point crosses the pupil centre.

Both quadratures keep Section 3.3's convergence discipline unchanged: doubling,
complete-array comparison, two consecutive successful comparisons per
dimension, at most four doublings per dimension, and the fixed `atol`/`rtol`.
The complete array compared for the two separation dimensions is
$P_m(\mathbf q)$ over every requested direction and every retained Poisson
order jointly; the complete array compared for the two paired-pupil dimensions
is $C$ over every separation node of the current partition. Neither may be
collapsed to a scalar or to the weighted sum before comparison.
Each separation comparison must satisfy one quarter of
`atol + rtol*max(abs(refined))`; otherwise the diagnostic raises
`BeamSamplingDerivationError`. The paired-pupil order is converged once against
the seed separation partition and is then held at its accepted value while the
separation partition is refined, and the two residual sequences are retained
separately. No rule in this method scales with $D/L$; the Hermite abscissae,
the allowed one-axis order set, and the $H_{\rm floor}$ rule they served are
gone.

The diagnostic supports at most `2**18` aperture nodes in any single solve,
`2**20` cumulative separation-node solve presentations, `2**34` cumulative
phase products, and `8*2**30` estimated workspace bytes. The
single-solve bound is sized for the base coherent transform, whose Section 3.3
converged node count at the shipped 14 m fixture is 158,400 and which the
superseded 65,536 bound rejected. Every bound remains a real fail-closed cap
and none is slack: under these rules the shipped fixture seeds a partition of
10,624 separation nodes and resolves 393,088 presentations and
$7.4\times10^{9}$ phase products, each below two thirds of its bound, and a
workload exceeding any bound returns no partial result and
raises `BeamSamplingDerivationError`. The diagnostic uses an internal batch
size equal to the largest power of two no greater than 256 that satisfies all
remaining phase-product and byte caps; a smaller non-power-of-two final batch
is permitted. Counts and the conservative shape-by-shape byte estimate are
checked before every Poisson, separation, paired-pupil, or batch refinement.
For $r=\max(8,\text{beam real bytes})$, $c=\max(16,\text{beam complex bytes})$,
and the current $J,N,Q,B,S$, that estimate is exactly

$$
E_{\rm Ruze}=r(16Q+8BQ+16B+12S+4J+6N)
+c(4BQ+8B+6S+4N).
$$

Buffers must be reused within those multiplicities. The worst-case work is
$O(NQ+SNJ)$ and memory is $O(Q+BQ+N+S)$ for separation-node count $N$,
paired-pupil node count $Q$, batch size $B$, direction count $S$, and
retained-term count $J$; no $Q^2$ pair array exists in production, and no
quantity in either bound depends on $D/L$ or on the Poisson order count beyond
the linear assembly term.

At least float64 weights and accumulation are used for both supported output
widths. Coherent and scattered arrays are cast separately to the beam real
dtype, checked finite and non-negative, and only then is
`total_ensemble_power = coherent_main_power + scattered_power` formed in that
same dtype. There is no clipping: a negative or non-finite weighted sum is an
internal numerical failure and raises `BeamSamplingDerivationError`. The
returned balance is exact in the result dtype; maximum observed
$|e_{\rm det}|$ must not exceed one beyond the unchanged tolerance, and total
power must not exceed one beyond it. Each assembled separation integral is real
because $C(-\boldsymbol\Delta)=C^*(\boldsymbol\Delta)$; the largest absolute
imaginary part actually formed is retained and must not exceed
`atol + rtol*max(abs(scattered_power))`, and a larger one is an internal
numerical failure raising `BeamSamplingDerivationError` rather than a silently
discarded part. Poisson tail, separation truncation bound, two successive
separation residuals, the paired-pupil residuals, and the base coherent
transform's residuals are retained separately; none is hidden in a single
convergence boolean.

This is an explicit scientific narrowing of Phase 5's generic
"effect-changes-visibility" rule: the existing coherent Ruze term must still
change cross-baseline visibility exactly, while the new error-beam diagnostic
must change the retained ensemble-power record and satisfy power balance. A
test requiring that diagnostic power to change a cross-baseline visibility is
itself a design violation.

`sigma` and `L` do not determine a deterministic complex voltage realization.
A later design may authorize per-antenna surface maps, or a complete covariance
plus fixed seed, realization identifier, and explicit inter-antenna correlation
policy. Such a realization must be mutually exclusive with applying Ruze loss
for the same residual error. No such stochastic or seeded surface is authorized
by Stage 1.

#### 3.4.2 Frozen public result

The typed public diagnostic is
`BeamSystem.evaluate_ruze_power_diagnostic(antenna_id, *, altitude_rad,
azimuth_rad, frequency_hz, time_mjd)`. It returns an immutable
`RuzePowerDiagnostic`. Inputs otherwise obey the `evaluate_jones` contract:
`antenna_id` is a canonical `AntennaId`; direction arguments are one-dimensional
NumPy arrays with identical shape `(S,)`; and frequency/time are exact finite
Python floats, with positive frequency. Unlike `evaluate_jones`, the diagnostic
requires `S >= 1` because convergence maxima are part of its result. An empty
pair raises `BeamAngularDomainError` with exact message
`Ruze power diagnostic requires at least one direction.` The result has exactly
these fields:

| Field | Exact resolved type and value |
|---|---|
| `schema_version` | `Literal["radiosim.ruze_power_diagnostic.v1"]` |
| `method` | `Literal["poisson_paired_pupil_separation_v1"]` |
| `antenna_id` | canonical immutable `AntennaId` |
| `covariance_convention` | `Literal["gaussian_one_over_e_surface_covariance_v1"]` |
| `normalization_convention` | `Literal["unmodified_ideal_aperture_v1"]` |
| `frequency_hz`, `time_mjd`, `rms_surface_error_m`, `correlation_length_m` | exact finite Python `float`; the first, third, and fourth are positive |
| `altitude_rad`, `azimuth_rad` | owned, C-contiguous, read-only `float64` arrays of shape `(S,)` |
| `coherent_main_power`, `total_ensemble_power`, `scattered_power` | owned, C-contiguous, read-only arrays of shape `(S,)` in the real component dtype of the beam |
| `convergence` | immutable `RuzePowerConvergence` below |

`RuzePowerConvergence` has exactly these fields, in this order:

```text
real_dtype, complex_dtype,
poisson_mu, poisson_first_order, poisson_last_order, poisson_term_count,
poisson_lower_omitted_mass, poisson_upper_omitted_mass,
poisson_total_omitted_mass, poisson_retained_weight_sum,
separation_cut_m, separation_omitted_bound,
separation_radial_order, separation_angular_order_max,
separation_node_count, separation_evaluation_count,
separation_penultimate_max_abs_delta, separation_final_max_abs_delta,
separation_imaginary_max_abs_residual, separation_topology_sha256,
aperture_method, aperture_partition_count,
aperture_topology_breakpoint_count, aperture_topology_sha256,
aperture_refinement_count, aperture_max_node_count,
aperture_penultimate_max_abs_delta, aperture_final_max_abs_delta,
aperture_q_max, surface_phase_kappa,
surface_radial_derivative_bound, surface_angular_derivative_bound,
fhat_evaluation_count, phase_product_count, batch_size,
atol, rtol, estimated_peak_bytes,
maximum_abs_e_deterministic, minimum_scattered_power, maximum_total_power,
returned_balance_max_abs_residual
```

`real_dtype`/`complex_dtype` are exactly `float32`/`complex64` or
`float64`/`complex128`; `aperture_method` is exactly
`boundary_fitted_polar_gauss_legendre_v1`; and
`aperture_topology_sha256` is lower-case SHA-256 over the canonical
target-dtype radial-breakpoint and periodic-angular-partition manifest of the
final accepted **base coherent** aperture solve. The
manifest byte stream begins with the ASCII domain
`radiosim.aperture_topology.v1\0`, then length-prefixes the real-dtype literal,
resolved central ratio or the literal `none`, ordered `(beta,a)` leg pairs,
ordered radial panels with transformation literal and canonical endpoints, and,
for each radial node of the final
accepted aperture solve in panel/node order, its ordered disjoint transmitting
angular intervals. Earlier failed/refined solves do not enter this digest.
Counts are unsigned little-endian 64-bit;
finite floats are normalized to positive zero, converted to the declared
little-endian real dtype, and emitted as raw bytes; every variable-length byte
string or sequence is preceded by its element count in the same integer
encoding. SHA-256 covers exactly that stream.

`separation_topology_sha256` pins the returned separation partition under the
same conventions. Its stream begins with the ASCII domain
`radiosim.ruze_separation_partition.v1\0`, then length-prefixes the real-dtype
literal, canonical `separation_cut_m`, the ordered separation radial panels with
transformation literal and canonical endpoints, the ordered per-node angular
trapezoid orders, and, for each separation node in panel/node order, its
canonical paired-pupil boundary radius $b$ and ordered paired radial panels
with their transformation literals. Integer, float, and length-prefix encodings
are exactly those above, earlier refined partitions do not enter it, and the
resolved zero-term Poisson case digests the domain prefix followed by the
real-dtype literal and zero counts.

All orders, counts, batch size, and byte fields are exact non-negative Python
integers. Both separation orders are zero exactly for the resolved zero-term
Poisson case; otherwise the radial order is positive and the angular maximum is
a positive power of two not below sixteen. Every other field from
`poisson_mu` through `returned_balance_max_abs_residual` not already classified
as a string, digest, or integer is an exact finite non-negative Python float.
Lower plus upper omitted mass equals total omitted mass in float64 arithmetic;
term count is zero exactly with Poisson interval `[0,0]` and otherwise equals
`last - first + 1`. In the zero-term case retained weight, both separation
orders, separation node count, separation evaluation count, both separation
residuals, the separation imaginary residual, `separation_cut_m`, and
`separation_omitted_bound` are exactly zero; lower omitted
mass is zero and upper and total omitted mass both equal `-expm1(-poisson_mu)`.

Evaluation counts, transform counts, phase products, and refinement counts are
cumulative over the whole public call. Orders describe the returned
refinement. `aperture_partition_count` is exactly
$\sum_p K_p$, where $K_p$ is the constant number of disjoint transmitting
angular intervals at any interior radius of topology-fitted radial panel $p$;
fully blocked panels contribute zero. `aperture_topology_breakpoint_count` is
the cardinality of the sorted unique normalized radial-panel boundary set,
including the active lower boundary (zero or $\epsilon$), every saturation and
canonical topology root, and outer boundary one. Both counts describe the
returned topology and are independent of quadrature node order.

`aperture_q_max` is dimensionless and equals
$\max R\|\mathbf k\|_2$ over every wavevector actually
presented to any aperture solve in the whole call; it is not physical
$\|\mathbf k\|$ in inverse metres. Paired-pupil solves carry no wavevector and
do not enter it, so it now reports the base coherent transform alone.
`aperture_max_node_count` is the maximum
Section 3.3 value of $Q$ over all seed/refinement evaluations of either solve
family; `batch_size` is
the largest separation-node batch actually scheduled; and
`estimated_peak_bytes` is
the maximum declared estimate. `separation_cut_m` is $\delta_{\rm cut}$ in
metres and `separation_omitted_bound` is the realized
$\exp(-\delta_{\rm cut}^2/\ell_{\rm wide}^2)$, exactly zero when
$\delta_{\rm cut}=D$ truncates nothing. `separation_radial_order` is the
returned radial order of the widest separation panel and
`separation_angular_order_max` the largest returned trapezoid order over its
radial nodes. `surface_phase_kappa` is exactly
$4\pi/\lambda$; the two surface derivative fields are the Section 3.3
$H_\rho,H_\varphi$ bounds. The named amplitude and power extrema have their
literal maximum/minimum meanings over all evaluated or returned values as
applicable.

For separation convergence, `penultimate` and `final` are the first and second
of the final two consecutive successful complete-array comparisons, maximized
over both separation dimensions, every requested direction, and every retained
Poisson order. For aperture
convergence, every angular sequence that licenses a radial result and the
radial sequence itself has two final successful comparisons;
`aperture_penultimate_max_abs_delta` is the maximum of the first such licensing
deltas and `aperture_final_max_abs_delta` the maximum of the second, across both
dimensions and over the base coherent solve and every paired-pupil solve
alike. These are not merely the final radial pair. There is no `converged` field
and no false state: failure returns no record.

`separation_node_count` is the returned partition's node total and
`separation_evaluation_count` counts every scheduled tuple
`(separation_radial_node, separation_angular_node, paired_pupil_solve)` across
all attempted separation and paired-pupil orders, including nodes re-presented
after refinement; the nesting of the angular trapezoid means a doubled angular
level increments it only for nodes it actually adds.
`fhat_evaluation_count` counts every wavevector element presented to an
aperture-transform refinement, so reevaluating one wavevector at a new aperture
order increments it again. `phase_product_count` counts every scalar
aperture-node exponential formed in a base or paired-pupil solve plus every
separation-node/direction exponential formed during assembly.
`aperture_refinement_count` counts
every evaluated angular or radial order increase after its seed evaluation;
seed evaluations do not increment that field. These definitions, rather than
wall-clock implementation details or cache hits, own the operation caps.

The dataclasses are frozen, final, and slotted; array fields are detached owned
copies with writes disabled. Unknown fields are impossible through their
constructors and are rejected by the retained-evidence schema. The diagnostic
is available only when the resolved antenna carries the nested diagnostic block
and otherwise raises `BeamEvaluationError` with the stable exact message
`A Ruze power diagnostic is not configured for this antenna.` It never mutates
or substitutes the matrix returned by `evaluate_jones` and never accepts a
backend argument: the complete algorithm is host-side.

### 3.5 Stage-1 rejections and acceptance invariants

Typed errors must preserve these exact semantic families:

- an explicitly present aperture block enables neither blockage nor a non-zero
  allowed Zernike mode;
- aperture physics or a Ruze diagnostic is attached to a non-circular analytic
  family, a FITS source, or a direct/derived/discrete pupil profile excluded by
  Section 3.1;
- a blockage ratio, support width, or resolved support geometry is outside its
  physical domain;
- the unmodified ideal aperture integral `N0` is zero or non-finite;
- `zernike_surface.modes` is absent, empty, not a sequence of exact three-field
  records, or contains an invalid, repeated, piston, or tip/tilt index;
- all Zernike coefficients are zero;
- a diagnostic lacks a positive correlation length, or is authored without a
  positive surface RMS, uses unsupported extended precision, is attached to a
  pupil whose resolved aperture physics declares a blockage, or receives an
  empty direction batch;
- an unknown normalization, covariance, or Zernike convention is supplied; or
- a boundary/topology value is not representable, or the fixed quadrature,
  Poisson-tail, separation-truncation, separation-order, solve-count,
  phase-product, memory,
  convergence, finite-value, amplitude-bound, power-bound, or exact-balance
  predicate cannot be satisfied.

Two Stage-1 mechanics follow from Section 2's taxonomy without changing it.
First, authored-kind failures surface as `ConfigSchemaError` carrying
Pydantic's own issue codes, while value-domain, identity, duplicate, and
document-level cross-field failures surface as `ConfigSemanticError` carrying
the frozen `beam.aperture_physics.*` and `beam.ruze_power_diagnostic.*`
codes; the instrument-stage support-leg geometry rejection uses the typed
error named in Section 3.2 and carries no `ConfigIssue` code. Second,
Section 2's rule that integers are not silently accepted as strict floats is
not delivered by strict Pydantic floats, which silently coerce Python ints
(bools, by contrast, are already rejected by strict float validation); every
Stage-1 float field therefore rejects `bool` and `int` inputs explicitly and
uniformly, reporting Pydantic's own `float_type` issue code through
`ConfigSchemaError`.

When multiple Stage-1 unsupported conditions coexist after schema/semantic
validation, aperture-owned family/profile checks run before any diagnostic
check. Within one feature, family precedes profile and profile precedes
diagnostic precision. Diagnostic paths are visited as `default` followed by
ascending `per_antenna` index. Duplicate diagnostic unsupported issues already
owned by the explicit aperture feature are suppressed. This order, together
with `ConfigIssue` sorting, fixes the first rejection recorded in evidence.

Acceptance requires all of:

- every supported profile in Section 3.1 reproduces its pre-Stage-1 normalized
  scalar Hankel formula within the existing dtype tolerance, while every
  excluded profile fails with the exact typed issue code and absent/default
  Gaussian configurations remain byte-identical;
- the closed blocked-uniform-aperture formula, including boresight loss;
- exact North/East support-leg masks at 0, 90, and 180 degrees, an antipodal-leg
  control, central-to-edge extent, boundary rules, and union-without-double-loss
  overlap cases;
- numerical unit-RMS/orthogonality checks for the declared Zernike basis;
- strict `modes` shape/type/unknown-field rejection and stable `(n,m)` sorting;
- a normal-incidence control and an off-axis control proving that authored $h$
  is half reflected OPD, with a supplied physical normal displacement mapped as
  $h=\delta_n\cos i$ rather than silently treated as $\delta_n$;
- available-platform NumPy `complex256` unmodified-profile and composed
  mask-plus-Zernike projections whose nodes, accumulation, and independent
  oracle never pass through complex128;
- boundary-fitted radial/angular partitions, saturation and topology roots,
  periodic-cut roots, canonical upper-float root ownership/collision failures,
  endpoint transformations, exact per-panel $L_p$ and nested-node $Q$ counts,
  target-width nodes, phase-bandwidth seeds, two consecutive dimensional
  convergence checks, and every fixed resource cap;
- blockage and Zernike each change a visibility when enabled;
- the composed mask-plus-phase result differs from a deliberately wrong
  product of two far-field factors;
- coherent Ruze cross-baseline scaling and ensemble power balance;
- a jointly Gaussian field characteristic-function oracle, plus a non-Gaussian
  covariance-matched counterexample showing that covariance alone does not
  license the declared kernel;
- a small-node independent $O(Q^2)$ pair oracle in tests only that agrees with
  the Poisson/separation result, catches the factor-two, $1/\pi$,
  deterministic-phase, negative-forward aperture-transform sign, and
  normalization controls, and is never imported by production; the separation
  sign itself is not an oracle because the whole-plane integral is invariant
  under $\boldsymbol\Delta\mapsto-\boldsymbol\Delta$ with
  $C(-\boldsymbol\Delta)=C^*(\boldsymbol\Delta)$;
- stable low/high-$\mu$ two-sided Poisson tails, the $\mu\to0$ first-order
  limit, and the $L\to\infty$ identity
  $B_{\rm sc}=(1-e^{-\mu})|e_{\rm det}|^2$;
- a deliberately narrow separation integrand that converges only after
  refinement or fails with the typed cap, plus whole-plane separation
  evaluation without a sky-domain rejection;
- non-negative scattered power without clipping, $|e_{\rm det}|\leq1$, total
  power no greater than one within the unchanged tolerance, exact returned
  balance, and every operation/memory cap firing before prohibited work;
- the exact frozen diagnostic fields, scalar types, array shapes/dtypes,
  ownership/read-only behavior, method/convention literals, and no-backend
  signature;
- proof by spy and data-flow test that requesting the diagnostic neither calls
  nor changes `evaluate_jones`, creates no Jones voltage, and leaves every
  cross-baseline visibility unchanged relative to the same surface RMS without
  the nested diagnostic; configuring the diagnostic does change the scientific
  fingerprint and retained ensemble-power record, while repeated evaluation
  does not mutate either;
- exact diagnostic-only unsupported-pupil, unsupported-family,
  unsupported-precision, and empty-direction rejections while the same beams
  with both Stage-1 features absent retain their prior behavior;
- point and HEALPix coverage;
- NumPy/Dask byte identity and JAX agreement at existing tolerances;
- disabled/default result and fingerprint byte identity; and
- changed fingerprints only for explicitly enabled stage-1 fixtures.

## 4. Stage 2 — native-feed beam squint

### 4.1 Strict physical configuration

`beams.squint` is a strict optional default-plus-per-antenna block. One resolved
record contains:

- `convention: cotton_uson_exact_v1`;
- positive finite `reference_frequency_hz`;
- finite `per_feed_offset_deg_at_reference` in `(0,90)`, the angular
  displacement of **one** hand from the nominal midpoint, not the total
  separation;
- finite `mechanical_feed_position_angle_deg`, resolved into `(-180,180]` and
  measured North through East in the antenna beam frame for the ray from the
  optical axis toward the physical off-axis feed location; and
- `positive_native_feed`, one of `x`, `y`, `r`, or `l`, which must belong to the
  resolved receptor basis. The other feed receives the negative displacement.

The mechanical position angle describes the physical off-axis feed location
and is not `receptors.*.feed_rotation_deg`. The latter is the electrical
receptor-axis convention used to build `C`; changing it cannot rotate the
physical squint displacement. A repeated or unknown per-antenna reference, a
feed label from the wrong basis, or an all-zero squint block is rejected.

The nominal pointing is the midpoint. If the per-feed reference displacement
is $\delta_{\rm ref}$, the exact Cotton/Uson scaling is

$$
\delta(\nu)=\sin^{-1}\!\left[
\frac{\nu_{\rm ref}}{\nu}\sin\delta_{\rm ref}\right].
$$

The setup preflight rejects an observation frequency for which the arcsine
argument leaves `[-1,1]`; it does not clip. The approximation
`delta proportional to 1/nu` may be documented as a small-angle limit but is
not the production law. The squint direction is orthogonal to the
optical-axis/feed plane, not along the feed-location ray. With
`u_feed(beta)=cos(beta)*North + sin(beta)*East`, the v1 handedness is

$$
\mathbf u_{\rm squint,+}(\beta)=
-\sin\beta\,\mathbf N+\cos\beta\,\mathbf E
=\mathbf u_{\rm feed}(\beta+\pi/2).
$$

The `positive_native_feed` beam is evaluated at `+delta*u_squint,+`; the other
feed is evaluated at its negative. This explicit `+pi/2` convention, together
with the label-keyed positive feed, fixes the sign that Cotton/Uson otherwise
state through telescope-specific feed-circle orientation. Swapping the positive
feed reverses the Stokes-V leakage oracle. Both evaluations use exact
great-circle rotations; their total feed-to-feed separation is `2*delta`.

The physical feed-location ray is evaluated in the existing antenna beam frame
before the orthogonal squint direction and sky/receptor basis conversion. It
supports exactly the five mount literals already owned by `jones.P`, with
`None` retaining its accepted `fixed`
interpretation. Reusing the existing mount factors, its direction on the sky is

$$
\beta_{{\rm feed},p}=\operatorname{wrap}(\beta_{\rm mechanical}
+\eta_p\psi+\nu_p\,\mathrm{alt}),
\qquad
\beta_{{\rm squint},p}=\operatorname{wrap}
(\beta_{{\rm feed},p}+\pi/2),
$$

where `(eta_p,nu_p)` is respectively `(1,0)` for alt-az, `(0,0)` for
equatorial/fixed, `(1,+1)` for Nasmyth-right, and `(1,-1)` for Nasmyth-left.
The sign is the same accepted field-rotation sign used by `P`; red tests include
the opposite-sign control. A rotating mount still requires `jones.P`, as it
does today. The apparent beam orientation is never emulated by adding a value
to electrical `feed_rotation_deg`. Unknown mount/beam-frame metadata is
rejected rather than assigned a generic correction.

#### 4.1.1 Strict document contract and typed rejections

`beams.squint` is a strict frozen block with exactly two fields: `default`,
an optional squint record, and `per_antenna`, a tuple of zero or more
per-antenna records. A per-antenna record carries exactly `antenna` — an
exact antenna number or name reference, following the accepted
`beams.pointing` reference forms — plus one complete squint record's five
fields. There is no per-antenna suppression form in v1: the resolved default
applies to every antenna not named by a `per_antenna` record, and an array
in which some antennas must not squint while others do is authored with no
`default` and one record per squinting antenna. A suppression form is a
later design, not authorized here.

Every squint record's five fields are strict in the Section 3.5 sense:
`convention` is exactly the literal `cotton_uson_exact_v1`;
`reference_frequency_hz`, `per_feed_offset_deg_at_reference`, and
`mechanical_feed_position_angle_deg` are exact finite Python floats that
reject `bool` and `int` through Pydantic's own `float_type` issue code; and
`positive_native_feed` is exactly one of the literals `x`, `y`, `r`, or `l`.
Unknown fields, wrong kinds, and missing required fields fail as
`ConfigSchemaError` with Pydantic's own issue codes. `SimulationOverrides`
carries no `beams` field and Stage 2 adds none; the document-schema
known-field hint table that already names `beams.pointing` and
`beams.pointing.default` gains the entries `beams.squint` and
`beams.squint.default`, so unknown-field hints stay exact.

Document-level value and cross-field failures are `ConfigSemanticError`
carrying exactly these frozen codes, paths, and messages, where `{value!r}`
renders the resolved Python value and `{mode!r}` the resolved beams mode:

- `beam.squint.identity_block`, path `beams.squint`, message
  `A beams.squint block must carry a default record or at least one
  per-antenna record.` — an explicitly present block with no `default` and
  an empty `per_antenna` is an exact identity and is rejected;
- `beam.squint.reference_frequency_domain`, path
  `beams.squint.default.reference_frequency_hz` or
  `beams.squint.per_antenna[i].reference_frequency_hz` with the zero-based
  authored index `i`, message `squint reference_frequency_hz must be a
  positive finite frequency in Hz; resolved {value!r}.`;
- `beam.squint.offset_domain`, path
  `beams.squint.default.per_feed_offset_deg_at_reference` or
  `beams.squint.per_antenna[i].per_feed_offset_deg_at_reference`, message
  `squint per_feed_offset_deg_at_reference must lie in the open interval
  (0, 90); resolved {value!r}.`;
- `beam.squint.mechanical_angle_domain`, path
  `beams.squint.default.mechanical_feed_position_angle_deg` or
  `beams.squint.per_antenna[i].mechanical_feed_position_angle_deg`, message
  `squint mechanical_feed_position_angle_deg must lie in (-180, 180];
  resolved {value!r}.` — the authored value is required in the canonical
  interval and is never wrapped.

One unsupported combination is `UnsupportedConfigError` with the frozen code
`beam.squint.unsupported_beam_family`, path `beams.squint`, and exact message
`Stage-2 beam squint supports only the analytic beams mode; resolved beams
mode is {mode!r}.` Stage-2 v1 accepts `beams.squint` only when the resolved
beams mode is `analytic`: a `shared_fits`, `per_antenna_fits`, or `mixed`
document with a squint block is rejected at the document stage with no
antenna-reference matching. A measured file's pattern may already contain
the physical feed displacement, and the scalar accepted subset provides no
metadata by which RadioSim could prove it does not; every analytic model,
including the Stage-1 aperture-physics branch, is supported because squint
only re-evaluates the existing scalar response at displaced directions.

Stage-2 document checks run after every Stage-1 aperture and diagnostic
check in Section 3.5's fixed order. Within `beams.squint`, the identity
check precedes value-domain checks; value-domain checks visit the `default`
record and then ascending `per_antenna` indices, each record's fields in the
declared field order; the unsupported-family check runs last. This order,
together with `ConfigIssue` sorting, fixes the first rejection recorded in
evidence.

Three rejections are owned by beam-system load rather than the document,
because they need resolved instrument, receptor, or frequency state.
`resolve_beam_assignments` resolves per-antenna squint records through the
accepted default-then-override map, so a per-antenna reference that names an
unknown antenna raises the existing typed `UnknownBeamAntennaError` and a
repeated canonical antenna raises the existing typed
`DuplicateBeamAssignmentError`, exactly as `beams.pointing` does today; it
also captures each squint-carrying antenna's resolved mount literal into the
resolved squint record. `load_beam_system` then owns two new typed
rejections, raised before any handler evaluation:

- `SquintFrequencyDomainError` — for every observation channel frequency
  `nu` and every squint-carrying antenna, the exact binary64 argument
  `(reference_frequency_hz / nu) * sin(radians(
  per_feed_offset_deg_at_reference))` must lie in `[-1, 1]`; the preflight
  rejects, never clips, and its message names the antenna, the offending
  observation frequency, the reference frequency, and the reference offset.
  Evaluation later computes the identical binary64 expression, so an
  argument outside the domain at evaluation time is an internal failure.
- `SquintReceptorBasisError` — `positive_native_feed` must belong to the
  assigned antenna's resolved receptor basis: `x`/`y` require the `linear`
  basis and `r`/`l` require the `circular` basis. The message names the
  antenna, the authored label, and the antenna's resolved basis.
  Document-stage validation does not own this check because per-antenna
  receptor bases exist only after receptor resolution, exactly as Section
  3.2 ruled for per-antenna diameters.

Both classes are appended to `core/beam/errors.py` as
`BeamLoadError` subclasses under Section 7.3's bounded grant. The other
native feed of a squint record is fixed by the label pair of its basis —
`x` pairs with `y` and `r` pairs with `l` — and receives the negative
displacement; no second label is authored.

### 4.2 Factorization into the canonical chain

Let the two displaced scalar voltage samples in native-feed order be

$$
D_b=\operatorname{diag}(b_0,b_1).
$$

The physical local response is `D_b C P`. RadioSim's fixed chain is `C E P`, so
for the accepted unitary receptor matrix

$$
E=C^\dagger D_b C,
\qquad C E = D_b C.
$$

`D_b` is diagonal in native-feed space. `E` is generally full in RadioSim's
sky-side space, including for a rotated linear receptor and for a circular
receptor. The old statement that squint merely makes `E` diagonal is retired.
`H` still owns the reporting-basis transform and correlation labels.

Stage 2 must replace the scalar-only order-unobservability oracle with an
analytic non-commuting case: choose unequal finite `b0` and `b1`, a nontrivial
unitary `C`, and a nontrivial `P`; prove `C E P` equals `D_b C P` and differs
from `C P E`. The scalar disabled case remains a separate byte-identity
regression.

The common deterministic `beams.pointing` rotation is applied first to express
the true visible direction in the common beam frame; the `+delta` and `-delta`
great-circle rotations are then taken about that resolved boresight. The horizon
gate remains on true topocentric altitude. A full-efield file accepted by Stage
3 is mutually exclusive with `beams.squint`, because an external full matrix may
already contain squint and provides no metadata by which RadioSim could subtract
it safely.

#### 4.2.1 Displacement geometry, precision, and evaluation ownership

The feed ray and squint direction are beam-frame position angles. With the
resolved mount factors `(eta_p, nu_p)` of Section 4.1 — `None` resolving to
`fixed` — the antenna's feed-ray angle at one time step is

$$
\beta_{{\rm feed},p}=\operatorname{wrap}(\beta_{\rm mechanical}
+\eta_p\psi_p+\nu_p\,\mathrm{alt}_p),
\qquad
\beta_{{\rm squint},p}=\operatorname{wrap}(\beta_{{\rm feed},p}+\pi/2),
$$

where `psi_p` and `alt_p` are the parallactic angle and true altitude of the
antenna's **resolved boresight** — the beam-frame zenith mapped to the sky
through the antenna's accepted pointing rotation, the topocentric zenith when
no pointing offset is configured — not per-direction values. The private
solver-owned adapter owns the boresight computation: once per antenna and
time step it derives the boresight's apparent hour angle and declination
with the same accepted exact inverse horizontal transform `DirectionBatch`
uses, evaluates the accepted `parallactic_angle` formula, and supplies both
values to the beam runtime. For `eta_p == 0` it supplies exactly `0.0` as
the parallactic angle, which the formula multiplies away. For `eta_p != 0`
with a boresight altitude exactly `pi/2` in binary64, the parallactic angle
is undefined; the adapter raises `BeamAngularDomainError` with the exact
message `Beam squint on a rotating mount is undefined at an exactly zenith
boresight.` rather than adopting `arctan2(0, 0)`. An alt-az antenna with no
pointing offset is exactly this case; the evidence records it as a
limitation, and geometry fixtures for rotating mounts carry a non-zero
pointing offset.

`BeamSystem.evaluate_jones` gains exactly two keyword-only parameters,
`boresight_parallactic_rad: float | None = None` and
`boresight_altitude_rad: float | None = None`. When the resolved antenna
carries squint, both must be exact finite Python floats; when it does not,
both must be `None`, and a violation of either rule raises
`BeamEvaluationError`. The no-squint call surface, behavior, and results are
byte-identical to today.

Displacement geometry is exact and binary64 throughout, matching the
accepted `DirectionBatch` and pointing-rotation contract. In the beam-frame
tangent basis at the beam-frame zenith, with
$\hat{\mathbf u}=\cos\beta_{\rm squint}\,\hat{\mathbf N}
+\sin\beta_{\rm squint}\,\hat{\mathbf E}$, the squint rotation axis is the
horizontal unit vector

$$
\hat{\mathbf a}_p=\sin\beta_{{\rm squint},p}\,\hat{\mathbf N}
-\cos\beta_{{\rm squint},p}\,\hat{\mathbf E},
$$

so that rotating the beam-frame zenith by $+\delta$ about
$\hat{\mathbf a}_p$ moves it along $+\hat{\mathbf u}$. For feed sign
$s_f=+1$ on the `positive_native_feed` and $s_f=-1$ on its partner, and
$\delta(\nu)$ from the exact arcsine law evaluated as the Section 4.1.1
binary64 expression, each already pointing-transformed beam-frame direction
unit vector $\hat{\mathbf n}$ is evaluated at the exactly rotated

$$
\hat{\mathbf n}_f=R(-s_f\,\delta(\nu);\hat{\mathbf a}_p)\,\hat{\mathbf n},
$$

by the Rodrigues rotation, then converted back to beam-frame altitude and
azimuth with the same `arctan2` forms the accepted pointing rotation uses.
This samples the feed's pattern rigidly displaced to
$+s_f\,\delta(\nu)\hat{\mathbf u}$; the two evaluations are exact
great-circle rotations, the midpoint of the two displaced centres is the
resolved boresight, and their total separation is `2*delta`. The horizon
gate remains on true topocentric altitude exactly as the accepted pointing
rotation leaves it: only visible directions are rotated, and the
evaluator's own angular-domain behavior applies to a displaced direction
exactly as it applies to a pointing-rotated direction today.

The beam runtime owns `D_b` assembly and the `E` composition. For each of
the two feed evaluations it calls the antenna's existing scalar evaluator —
the analytic path, including any Stage-1 aperture-physics branch — at the
displaced directions, producing per-feed scalar samples `b_+` and `b_-` at
the resolved result dtype. `D_b = diag(b_0, b_1)` is ordered by the
antenna's resolved native feed order `("x","y")` or `("r","l")`, with the
positive displacement on the feed whose label equals
`positive_native_feed`. The runtime constructs the antenna's receptor
matrix at the resolved beam dtype from the antenna's resolved receptor
basis and static rotation using the accepted formulas — `C = M(basis) @
R(chi)` with `R(chi) = [[cos chi, sin chi], [-sin chi, cos chi]]`,
`M(linear) = [[0, 1], [1, 0]]`, and `M(circular) = (1/sqrt(2)) *
[[1, i], [1, -i]]` — and returns the composed

$$
E=C^\dagger D_b\,C
$$

from `evaluate_jones`, with the accepted Ruze voltage factor applied to the
composed `E` exactly where the scalar path applies it today (the factor is
scalar and commutes). Direction geometry is binary64; `b_+`, `b_-`, `C`,
and the composition are evaluated at the resolved beam dtype and never pass
through a narrower width when the resolved dtype is wider than complex128.
FITS-backed definitions never reach this path (Section 4.1.1), so the
extended widths remain analytic-only exactly as accepted. Backend
conversion of the finished `E` batch happens at the existing adapter
boundary and nowhere earlier.

To make this possible, `load_beam_system` gains the resolved receptor set:
the `Simulator` passes its already-resolved `ResolvedReceptorSet` through
the new keyword-only parameter `receptors` (typed
`ResolvedReceptorSet | None = None`) under Section 7.3's bounded
`api/simulator.py` grant, and
`load_beam_system` requires it whenever any resolved antenna carries
squint. Receptor resolution is unchanged and remains the accepted single
authority; the runtime reads only each squint antenna's resolved basis and
static `feed_rotation_rad`, and the solver's own `C` term continues to come
from the same resolved receptor set, so the `C` inside `E` and the chain's
`C` cannot disagree.

The per-antenna response identity widens exactly when squint is present:
the `_response_key` payload gains one `"squint"` sub-object with exactly
the keys `reference_frequency_hz`, `per_feed_offset_deg_at_reference`,
`mechanical_feed_position_angle_deg`, `positive_native_feed`, `mount_type`,
`receptor_basis`, and `feed_rotation_rad`. An antenna without squint
produces a byte-identical key to today. Within one solver step the adapter
cache is keyed on this response identity as today; across steps each
adapter is rebuilt, so the time dependence of a rotating mount's
`beta_feed` never crosses a cache boundary.

The resolved squint record is one frozen dataclass `ResolvedSquint` in
`core/beam/models.py`, stored on the assignment as
`ResolvedBeamAssignment.squint`,
with exactly the fields `convention:
Literal["cotton_uson_exact_v1"]`, `reference_frequency_hz: float`,
`per_feed_offset_deg_at_reference: float`,
`mechanical_feed_position_angle_deg: float`, `positive_native_feed:
Literal["x", "y", "r", "l"]`, and `mount_type` holding one of the five
accepted mount literals. It enters `_assignment_fingerprint` and the
handler-independent scientific fingerprint only when present, through one
payload function whose exact key set is the six field values above plus
the convention literals `"direction_convention":
"feed_ray_plus_half_pi_north_through_east_v1"`, `"frame_convention":
"pointing_then_squint_great_circle_v1"`, and `"factorization_convention":
"receptor_conjugated_native_diagonal_v1"`. Squint is per-antenna state like
pointing and surface error: it never enters the handler preload key, and
two antennas sharing a handler may carry different squint records.

### 4.3 Stage-2 acceptance invariants

Acceptance requires:

- exact midpoint symmetry and total separation `2*delta`;
- the arcsine frequency law at at least three frequencies and a control that
  distinguishes it from the small-angle approximation;
- exact orthogonality to the physical feed-location ray, the declared `+pi/2`
  handedness, label-keyed feed-sign reversal, and mechanical-angle rotation;
- analytic `E=C^dagger D_b C`, physical `C E P` order, and an order-matters
  negative control;
- the first-order Stokes-V leakage sign for a declared R/L assignment, with the
  sign reversed when the assignment reverses;
- scalar disabled/default byte identity;
- point and HEALPix paths and NumPy/JAX/Dask parity;
- fingerprints changed only for squint-enabled fixtures;
- every Section 4.1.1 document rejection with its exact typed exception,
  issue code, path, and message, and every load and evaluation rejection —
  unknown and duplicate per-antenna references, the arcsine frequency
  preflight, the receptor-basis mismatch, and the exactly-zenith rotating
  boresight — with its exact typed exception;
- the mount field-rotation formula on every one of the five accepted mount
  literals, including the opposite-sign control of Section 4.1;
- the widened per-antenna response identity: two antennas sharing one
  handler with different squint or receptor state never share a composed
  `E`, an antenna without squint produces today's byte-identical key, and
  the no-squint `evaluate_jones` call surface and results are
  byte-identical;
- one extended-width `complex256` factorization row per Section 8.1's
  Stage-2 envelope, composed and independently rederived without passing
  through `complex128`; and
- the Ruze voltage factor applied once to the composed `E`, with squint
  composing correctly with the Stage-1 aperture-physics branch.

Stage 2 begins only after Stage 1 is accepted. WP-5's accepted east-X semantics
are a prerequisite, not Stage-2 acceptance evidence.

## 5. Stage 3 — full efield Jones response

### 5.1 Accepted input and normalization

Stage 3 widens the existing private UVBeam evaluator; it does not create a
second FITS loader. The accepted v1 file contract is:

- UVBeam `beam_type == "efield"`, `antenna_type == "simple"`, and exactly the
  `az_za` pixel coordinate system; UVBeam HEALPix files remain rejected in v1,
  while RadioSim's HEALPix **sky solver** must consume the accepted `az_za`
  efield response;
- exactly two vector components and exactly two feeds;
- a canonical ordered feed pair `("x","y")` or `("r","l")` that exactly
  equals every assigned antenna's `ResolvedReceptor.feed_array`;
- finite complex data, a finite **real floating-point**
  `basis_vector_array`, internally consistent `feed_angle` and derived
  `x_orientation`, and complete visible-hemisphere coverage; and
- no phased-array coupling, element delay, or station array factor under this
  stage.

The metadata-to-receptor rule is exact. For every antenna assigned the file,
the two file `feed_angle` values must equal that receptor's
`feed_angle_rad` modulo `2*pi` within the retained `1e-12` radian tolerance:
`(pi/2+chi, chi)` for linear feeds and `(chi, chi)` for circular feeds, where
`chi` is the resolved static `feed_rotation_rad`. The value returned by
pyuvdata 3.2.1's `get_x_orientation_from_feeds()` must be exactly `"east"`,
`"north"`, or `None` as implied by those same feeds and angles; any legacy
`x_orientation` value exposed by the reader must agree. It is consistency
metadata, not a second rotation. A shared file whose row labels or angles match
one assigned antenna but not another is rejected.

The file `mount_type` must be exact `"fixed"`. Here that literal declares that
the stored `az_za` pattern is intrinsic to the antenna beam frame; it does not
override the resolved instrument mount. RadioSim's instrument plus `P` term
remain the sole owners of alt-az/equatorial/Nasmyth field rotation. A non-fixed,
missing, or conflicting file mount is rejected to prevent a second mount
rotation.

The strict config literal is `normalization: uvbeam_peak_common_v1`. The input
must already carry `data_normalization == "peak"`, a unit `bandpass_array`, and,
at every intrinsic frequency, one common full-stored-grid maximum
`max(abs(data_array[:, :, frequency, ...])) == 1` within the existing
dtype-derived fixed normalization tolerance. Below-horizon samples, when
stored, participate in this maximum. This is the exact common factor used by
UVBeam 3.2.1's official peak-normalization operation. RadioSim does **not** call
that mutating operation while loading and never normalizes matrix elements,
feeds, co/cross components, or diagonals independently. A producer may use the
official operation before writing the file, but the simulator accepts or
rejects the committed bytes as authored. This preserves relative feed gain,
leakage, phase, and matrix conditioning. A zero/non-finite reference scale,
non-unit bandpass, unsupported normalization metadata, visible-only
renormalization, or any other silent renormalization is rejected. The
intrinsic-node full-grid maxima and tolerance are recorded before the accepted
complex frequency interpolation. Direction-independent spectral gain remains
the configured `B` term's responsibility.

### 5.2 Coordinates, Ludwig-3, and receptor factorization

Raw UVBeam vector-axis order is not assumed to be RadioSim's field basis.
Interpolation requests `return_basis_vector=True`. For each direction the
returned basis vectors are transformed from UVBeam's documented azimuth/zenith
coordinates into RadioSim's `(north,east)` tangent-field basis. The already
accepted direction-coordinate mapping remains
`az_uv = (pi/2 - az_radiosim) mod 2*pi`.

With UVBeam vector index `a`, native feed row `f`, and RadioSim tangent-field
column `c`, the mapping is explicitly

$$
J_{\rm native}[f,c]=\sum_a
\operatorname{data}[a,f]\,\operatorname{basis}[a,c],
$$

with no conjugation and no implicit transpose. `basis_vector_array` is real by
the pyuvdata 3.2.1 contract; complex phase lives in `data`. Tests therefore use
a legal real, non-identity, non-symmetric basis together with complex efield
samples so that a transpose or conjugation mistake is independently observable.

For a linearly polarized reference, Ludwig's third definition is applied with
$\varphi=0$ north and increasing east:

$$
E_{\rm co}=E_\theta\cos\varphi-E_\phi\sin\varphi,
\qquad
E_{\rm cross}=E_\theta\sin\varphi+E_\phi\cos\varphi.
$$

The basis transform must preserve total field power and be continuous at the
azimuth wrap and zenith. A synthetic crossed-ideal-dipole case fixes signs,
row/column order, and zenith limits. A synthetic quadrupolar response fixes the
principal-plane zeros and the parity of the two feed rows, but does not become
a public production model.

At the zenith, where coordinate azimuth is singular, RadioSim uses the
North/East tangent limit at `az_radiosim=0` and requires all file-grid approaches
to that physical direction to agree after basis-vector conversion within the
fixed basis tolerance. A file whose physical matrix depends on the arbitrary
zenith azimuth is rejected rather than averaged.

After coordinate conversion, let `J_native` be the complete matrix mapping the
RadioSim incident tangent field into the file's native-feed voltages. Because
the canonical chain already applies the resolved ideal receptor `C`, the
sky-side beam factor is

$$
E=C^\dagger J_{\rm native},
\qquad C E=J_{\rm native}.
$$

This factorization prevents double application of feed orientation or circular
conversion. File `feed_angle` and `x_orientation` validate the row identity but
are not applied as another matrix: the complex file data already describe those
physical feed voltages, `C` is built from the exactly matching resolved
receptor, and `E=C^dagger J_native` factors that complete response into the
fixed chain. Any metadata/receptor mismatch is rejected rather than repaired or
guessed. `H` alone converts native receptor voltages into the selected output
basis.

### 5.3 Cross-polar and IXR diagnostics

For each finite nonsingular `J_native`, Stage 3 may report singular values
`sigma_max >= sigma_min > 0`, condition number `kappa`, and

$$
\operatorname{IXR}_J=\left(\frac{\kappa+1}{\kappa-1}\right)^2.
$$

An exactly unitary-scaled matrix has infinite IXR; a singular matrix is marked
with an explicit diagnostic state rather than converted to NaN. IXR is a
derived diagnostic over the full accepted matrix. It does not alter `E`, and no
`ixr_db` beam config is added.

### 5.4 Result and file-format behavior

- **In memory:** `SimulationResult` keeps the selected output correlation basis
  and full cross-hands. Its beam snapshot records file content hash, feed and
  vector basis, normalization convention/factors, and the resolved transform.
- **Summary JSON:** the bounded `beam` block reports the full-efield mode,
  convention versions, content digest, and bounded diagnostic extrema; it does
  not embed direction-sized matrices.
- **HDF5:** `provenance/beam_json` round-trips the complete scientific beam
  snapshot and exact `scientific_sha256`. Visibility data and correlation axes
  already support all four products. A schema bump is required only if the
  reader's fixed field set changes; it must never be performed merely to rename
  scalar wording.
- **UVFITS and Measurement Set:** both serialize the already-corrupted four
  correlation products in the result's declared output basis. Their antenna
  feed metadata comes from `ResolvedReceptorSet`, not from an untrusted file row
  label. Existing HISTORY provenance carries `source_scientific_sha256`.
  Neither format is claimed to preserve a reusable efield beam model, and no
  invented standard table is added.
- **Readers:** HDF5 reconstructs scientific equality. UVFITS/MS reconstruct
  representable visibilities, feed metadata, correlation labels, and source
  scientific identity under their existing projection contracts.

Every writer gets a non-scalar efield round-trip/equivalence case in both
linear and circular output bases. Scalar BeamFITS output remains byte-identical.

### 5.5 Point, HEALPix, backend, and independent comparison

Point and HEALPix solvers consume the same complete `_ResolvedBeamJones` batch.
The HEALPix cache remains keyed on the full response identity; it cannot reuse a
matrix across differing receptors, squint parameters, normalization factors, or
full-efield files. NumPy and Dask must be byte-identical, and JAX must agree at
the existing float64 tolerance after the host matrix is transferred.

The optional `crossval` environment retains `pyuvsim==1.4.0` and
`pyuvdata==3.2.1`. A Stage-3 comparison uses the same full-efield UVBeam file,
full-Stokes point sky, times, frequencies, antennas, and accepted east-X/fringe
mapping in both simulators. It records per-correlation absolute and relative
residuals, the exact source and artifact commits, lock and input hashes, and
every convention mapping. It is non-gating and may license only the sentence
"compared against pyuvsim for the named fixture, with the recorded agreements
and open disagreements". It never licenses an unqualified "validated" claim.

The retained artifact lives under `output/crossvalidation/`, is generated from
a clean source SHA before it exists, and is authenticated by a standard-suite
schema/digest test. No existing cross-validation artifact is overwritten.

### 5.6 Stage-3 acceptance invariants

Acceptance requires:

- rejection of power beams, missing/degenerate basis vectors, unsupported feed
  pairs, inconsistent feed-angle/x-orientation/receptor metadata, non-fixed file
  mounts, unsafe normalization, and complex-valued basis vectors;
- full-stored-grid (not visible-only) unit peak at every intrinsic frequency;
- crossed-ideal-dipole and quadrupolar analytic oracles;
- Ludwig-3 power preservation, zenith limit, wrap continuity, and sign controls;
- exact `C E = J_native` and an output-basis conversion control;
- full Jones order tests with non-commuting `C`, `E`, and `P`;
- point and HEALPix coverage and NumPy/JAX/Dask parity;
- in-memory, summary JSON, HDF5, UVFITS, and MS behavior above;
- a new dated, clean-source, non-gating pyuvsim comparison artifact; and
- scalar/default fingerprints unchanged while full-efield fixtures receive new
  pins through the normal multi-environment characterization process.

## 6. Red-first test programme

Every stage starts with a red-test commit. The red record names the exact test,
expected scientific behavior, actual failure, and why the failure is not a
fixture defect. A stage cannot replace a test with an implementation-shaped
assertion after observing production output.

The minimum new modules are:

- `tests/unit/test_io/test_sci005_beam_config.py` for strict parse and typed
  rejection;
- `tests/unit/test_core/test_sci005_aperture_physics.py` for Stage 1;
- `tests/unit/test_core/test_sci005_ruze_diagnostic.py` for the independent
  small-node pair oracle, Poisson/separation/refinement/resource predicates, and
  frozen result;
- `tests/unit/test_core/test_sci005_beam_squint.py` for Stage 2;
- `tests/unit/test_core/test_sci005_full_efield.py` for Stage 3;
- `tests/integration/test_sci005_beam_physics.py` for point/HEALPix, outputs,
  and effect-through-`Simulator` cases;
- `tests/unit/test_sci005_stage1_dependency.py` for the retained WP-7 CPU
  acceptance certificate and `G1 -> R1` binding;
- `tests/crossvalidation/test_sci005_efield_pyuvsim.py` for the optional Stage-3
  comparison; and
- `tests/unit/test_sci005_evidence.py` for strict retained-evidence and digest
  authentication;
- `tests/unit/test_sci005_stage1_acceptance.py` for Stage-1 acceptance and
  ancestry authentication;
- `tests/unit/test_sci005_stage2_acceptance.py` for Stage-2 acceptance and its
  exported WP-9 M3 dependency certificate; and
- `tests/unit/test_sci005_stage3_acceptance.py` for Stage-3 acceptance and
  closure-parent authentication.

Existing chain, beam-runtime, result-output, backend-parity, release-scan, and
characterization modules are extended only where the stage changes a property
they already own. No `xfail`, tolerance widening, warning suppression, or
benchmark exception is acceptance evidence.

## 7. Exact writable lists

These lists are implementation authority only after independent design review
and the stated dependency. A path not listed here requires a bounded design
correction before it is edited.

### 7.1 Design-only authority

- `docs/development/sci005_beam_physics_plan.md` (this governing memo)
- `docs/index.rst`
- `PostTier8RemediationPlan.md` (WP-8 section and ledger dependency wording)

`D1` is the independently accepted commit containing this amendment. `D2`
follows `U1` through Section 8.3's starred `U1 ->* D2` edge and freezes the
complete normative Stage-2 evidence envelope before `R2`; `D3` directly
parents `U2` and does the same for Stage 3 before `R3`.
Those later design gates may write only the paths above, change no production,
red oracle, retained artifact, or prior acceptance text, and require their own
exact-byte independent design reviews, with one recorded exception: the
superseded original Stage-2 gate commit additionally landed the bounded
four-line deletion in
`tests/unit/test_sci005_stage1_dependency.py` that its header record and
Section 8.3 authorize, changing no other byte of that file; the operative
`D2` and every later design gate touch only the paths above. Omitting an
applicable `D2` or `D3`, or combining it with red tests, invalidates the
following stage.

Stage 1 has one intervening dependency-gate tip `G1`. `D1` must be an ancestor
of globally clean `G1`, and the accepted WP-7 CPU acceptance commit must also
be an ancestor of `G1`; both ancestry tests are inclusive, so `G1` may equal
the accepted WP-7 `A`. At `HEAD == G1`, before any SCI-005 red byte exists, the
exact upstream interface is run as:

```text
pixi run python tools/wp7_perf001_cpu_evidence.py verify-accepted \
  --acceptance-commit <40hex-WP7-A> --descendant <40hex-G1>
```

The command must exit zero and emit the upstream canonical sorted one-line JSON
certificate with schema
`radiosim.perf001.cpu_acceptance_certificate.v1` and exactly these fields:

```text
schema_version, acceptance_commit, evidence_commit, generating_source_sha,
descendant_commit, artifact_path, artifact_sha256,
cpu_evidence_tool_sha256, production_record_validator_sha256,
production_harness_sha256, pixi_manifest_sha256, pixi_lock_sha256,
evidence_diff_paths, acceptance_diff_paths, verdict, passed
```

The certificate requires `acceptance_commit` to be the named WP-7 `A`,
`descendant_commit == G1`, `verdict` exactly
`CPU_ACCEPTED_P_E_HARDWARE_GATED`, and `passed: true`; the upstream verifier
authenticates the exact evidence/source ancestry, raw artifact, tool,
production-validator, harness, manifest, lock, and diff paths named by the
remaining fields. Ledger prose is not a substitute.

`R1` directly parents `G1` and adds the raw certificate line, including its
single final LF, at
`docs/development/sci005_stage1_wp7_dependency.json`. The same red commit
creates `tests/unit/test_sci005_stage1_dependency.py` with exactly one design
binding assignment,
`APPROVED_SCI005_D1_SHA = "<40hex-D1>"`. Its value must name the accepted
ancestor whose memo blob is this amendment; no later stage may change that
literal, and no later stage may change any other dependency-validator byte
beyond the one bounded four-line deletion the accepted `D2` header records —
the completed `D1..G1` interval-freeze assertion on checked-out memo bytes,
removed with no other byte change. Stage-1 evidence and acceptance
derive `design_sha` only from this binding, never by choosing a matching commit
from history.

A later live checkout
cannot replay the command directly because the upstream verifier requires
clean `HEAD == --descendant`. The `R1`, evidence, and acceptance validators
therefore create a fresh temporary directory, attach a detached Git worktree at
exact `G1`, require its status clean, and run the displayed command from that
worktree with the certificate's `acceptance_commit` and `descendant_commit`.
They execute `tools/wp7_perf001_cpu_evidence.py` from the `G1` tree, require its
raw SHA-256 to equal `cpu_evidence_tool_sha256`, and compare stdout byte-for-byte
with the retained path. They remove the worktree and temporary directory on
success or failure; inability to create, authenticate, execute, or clean up the
worktree is a hard failure and never mutates the caller's checkout.

The validators also require both ancestors above and reject a dirty detached
`G1`, a missing/upstream interface mismatch, a nonzero command, stderr output,
another verdict, or a false `passed`. The Stage-1 evidence artifact must retain
this path and raw SHA-256 in its `artifacts` array, so `A1` reviews and
authenticates it through Section 8.2.

### 7.2 Stage 1 red tests, implementation, and evidence

- `src/radiosim/io/beam_config.py`
- `src/radiosim/io/config.py`
- `src/radiosim/io/config_resolution.py`
- `src/radiosim/core/beam/models.py`
- `src/radiosim/core/beam/resolution.py`
- `src/radiosim/core/beam/aperture.py` (new; the only production owner of the
  pupil profiles, support mask, Zernike phase, Ruze power diagnostic, and
  frozen diagnostic records)
- `src/radiosim/core/beam/errors.py` (append-only: exactly one new class
  `InvalidBeamGeometryError(BeamAssignmentError)` with docstring and its
  `__all__` entry; no existing byte changes)
- `src/radiosim/core/beam/analytic.py`
- `src/radiosim/core/beam/runtime.py`
- `src/radiosim/core/beam/__init__.py`
- `tests/unit/test_io/test_sci005_beam_config.py` (new)
- `tests/unit/test_core/test_sci005_aperture_physics.py` (new)
- `tests/unit/test_core/test_sci005_ruze_diagnostic.py` (new)
- `tests/unit/test_core/test_beam_models.py`
- `tests/unit/test_core/test_beam_resolution.py`
- `tests/unit/test_core/test_beam_runtime.py`
- `tests/unit/test_core/test_beam_solver_integration.py`
- `tests/integration/test_sci005_beam_physics.py` (new)
- `tests/characterization/test_tier6_current_behavior.py`
- `tests/characterization/test_tier7_current_behavior.py`
- `tests/characterization/test_tier8_current_behavior.py`
- `tests/unit/test_sci005_stage1_dependency.py` (new strict `R1` validator and
  exact `D1` binding)
- `docs/development/sci005_stage1_wp7_dependency.json` (new; exact upstream
  certificate bytes added only by `R1`)
- `tests/unit/test_sci005_evidence.py` (new)
- `tools/sci005_stage_evidence.py` (new)
- `docs/development/sci005_stage1_evidence.schema.json` (new; normative schema
  transcription authenticated by the evidence tool)
- `tools/sci005_stage1_acceptance.py` (new; Stage-1 acceptance generator and
  ancestry/certificate verifier)
- `tests/unit/test_sci005_stage1_acceptance.py` (new; strict validator with
  approved-E/artifact null sentinels at `S1`)
- `docs/development/sci005_stage1_acceptance.schema.json` (new; literal strict
  schema transcription)
- `docs/user_guide/configuration.rst`
- `docs/user_guide/configuration_support.rst`
- `docs/user_guide/beam_models.rst`
- `docs/api/core.rst`
- `docs/migration_guide.md` only if the accepted config/result contract breaks
  a pre-v1 surface
- `docs/development/sci005_stage1_evidence.json` (new evidence successor only)
- `docs/development/sci005_stage1_acceptance.json` (new acceptance successor
  only)

### 7.3 Stage 2 red tests, implementation, and evidence

- `src/radiosim/io/beam_config.py`
- `src/radiosim/io/config.py`
- `src/radiosim/io/config_resolution.py`
- `src/radiosim/core/beam/models.py`
- `src/radiosim/core/beam/resolution.py`
- `src/radiosim/core/beam/runtime.py`
- `src/radiosim/core/beam/errors.py` (append-only: exactly two new classes
  `SquintFrequencyDomainError(BeamLoadError)` and
  `SquintReceptorBasisError(BeamLoadError)`, each with docstring and its
  `__all__` entry; no existing byte changes)
- `src/radiosim/core/beam/__init__.py` (append-only: the two new error
  exports alone)
- `src/radiosim/core/visibility.py`
- `src/radiosim/api/simulator.py` (exactly the `load_beam_system` call site:
  passing the already-resolved receptor set through the Section 4.2.1
  keyword; no other statement changes)
- `tests/unit/test_io/test_sci005_beam_config.py`
- `tests/unit/test_core/test_sci005_beam_squint.py` (new)
- `tests/unit/test_core/test_beam_runtime.py`
- `tests/unit/test_core/test_beam_solver_integration.py`
- `tests/unit/test_jones/test_chain_order.py`
- `tests/unit/test_jones/test_backend_parity.py`
- `tests/integration/test_sci005_beam_physics.py`
- `tests/characterization/test_tier6_current_behavior.py`
- `tests/characterization/test_tier7_current_behavior.py`
- `tests/characterization/test_tier8_current_behavior.py`
- `tests/unit/test_sci005_evidence.py`
- `tools/sci005_stage_evidence.py`
- `docs/development/sci005_stage2_evidence.schema.json` (new; normative schema
  transcription authenticated by the evidence tool)
- `tools/sci005_stage2_acceptance.py` (new; Stage-2 acceptance generator and
  WP-9 M3 dependency-certificate verifier)
- `tests/unit/test_sci005_stage2_acceptance.py` (new; strict validator with
  approved-E/artifact null sentinels at `S2`)
- `docs/development/sci005_stage2_acceptance.schema.json` (new; literal strict
  schema transcription)
- `docs/user_guide/configuration.rst`
- `docs/user_guide/configuration_support.rst`
- `docs/user_guide/beam_models.rst`
- `docs/user_guide/jones_matrices.rst`
- `docs/migration_guide.md` for the non-scalar-`E` pre-v1 widening
- `docs/development/sci005_stage2_evidence.json` (new evidence successor only)
- `docs/development/sci005_stage2_acceptance.json` (new acceptance successor
  only)

### 7.4 Stage 3 red tests, implementation, and evidence

- `src/radiosim/io/beam_config.py`
- `src/radiosim/io/config.py`
- `src/radiosim/io/config_resolution.py`
- `src/radiosim/core/beam/models.py`
- `src/radiosim/core/beam/fits.py`
- `src/radiosim/core/beam/runtime.py`
- `src/radiosim/core/visibility.py`
- `tests/fixtures/beamfits.py`
- `tests/unit/test_io/test_sci005_beam_config.py`
- `tests/unit/test_core/test_sci005_full_efield.py` (new)
- `tests/unit/test_core/test_beam_fits.py`
- `tests/unit/test_core/test_beam_pyuvdata_contract.py`
- `tests/unit/test_core/test_beam_runtime.py`
- `tests/unit/test_core/test_beam_solver_integration.py`
- `tests/unit/test_core/test_result.py`
- `tests/unit/test_io/test_hdf5_result.py`
- `tests/unit/test_io/test_result_summary.py`
- `tests/unit/test_io/test_uvfits.py`
- `tests/unit/test_io/test_measurement_set.py`
- `tests/unit/test_io/test_standard_visibility.py`
- `tests/unit/test_jones/test_chain_order.py`
- `tests/unit/test_jones/test_backend_parity.py`
- `tests/integration/test_sci005_beam_physics.py`
- `tests/crossvalidation/test_sci005_efield_pyuvsim.py` (new, optional)
- `tests/characterization/test_h5py_output_contract.py`
- `tests/characterization/test_pyuvdata_321_output_contract.py`
- `tests/characterization/test_pyuvdata_321_polarization_contract.py`
- `tests/characterization/test_tier6_current_behavior.py`
- `tests/characterization/test_tier7_current_behavior.py`
- `tests/characterization/test_tier8_current_behavior.py`
- `tests/unit/test_sci005_evidence.py`
- `tools/sci005_stage_evidence.py`
- `docs/development/sci005_stage3_evidence.schema.json` (new; normative schema
  transcription authenticated by the evidence tool)
- `tools/sci005_stage3_acceptance.py` (new; Stage-3 acceptance generator and
  closure-parent certificate verifier)
- `tests/unit/test_sci005_stage3_acceptance.py` (new; strict validator with
  approved-E/artifact null sentinels at `S3`)
- `docs/development/sci005_stage3_acceptance.schema.json` (new; literal strict
  schema transcription)
- `docs/user_guide/configuration.rst`
- `docs/user_guide/configuration_support.rst`
- `docs/user_guide/beam_models.rst`
- `docs/user_guide/jones_matrices.rst`
- `docs/api/core.rst`
- `docs/migration_guide.md`
- `output/crossvalidation/README.md`
- `output/crossvalidation/<date>-sci005-efield-pyuvsim-1.4.0.json` (new)
- `docs/development/sci005_stage3_evidence.json` (new evidence successor only)
- `docs/development/sci005_stage3_acceptance.json` (new acceptance successor
  only)

### 7.5 Evidence, acceptance, and closure successors

The generator, schema, and validator for a stage land in its `S` commit. At
that point its official evidence and acceptance JSON paths are absent, all
synthetic strict-schema tests pass, and that target stage's approved digest
constants are the literal `None`. At `S2` and `S3`, every earlier-stage
artifact and approved constant instead remains pinned and immutable. The
following successor authority is exhaustive:

- `Ei` adds only `docs/development/sci005_stage{i}_evidence.json` and changes
  only `APPROVED_STAGE{i}_SOURCE_SHA` and
  `APPROVED_STAGE{i}_EVIDENCE_ARTIFACT_SHA256` in
  `tests/unit/test_sci005_evidence.py`, from `None` to the exact lower-case
  40- and 64-hexadecimal literals. It may not change validator logic, any other
  test byte or path, schemas, production, documentation, or an earlier
  artifact. `E3` additionally adds exactly one new dated
  `output/crossvalidation/<date>-sci005-efield-pyuvsim-1.4.0.json` generated
  from clean `S3`; that file is authenticated by the Stage-3 evidence artifact
  and is the sole phase-specific addition to this `E` rule.
- `Ai` adds only `docs/development/sci005_stage{i}_acceptance.json`, changes
  only `APPROVED_EVIDENCE_SHA` and
  `APPROVED_ACCEPTANCE_ARTIFACT_SHA256` in
  `tests/unit/test_sci005_stage{i}_acceptance.py`, from `None` to the exact
  lower-case 40- and 64-hexadecimal literals. No import, expression,
  annotation, key, surrounding token, or other literal in either
  approved-constant assignment may change. The validator compares the token
  stream outside those two literal spans to its direct-parent `E` bytes, so a
  same-line logic change fails. No status prose is permitted in `Ai`.
- `Ui` directly parents accepted `Ai` and may change only the status paths
  below. Before it can be committed, the phase-local acceptance tool verifies
  `Ai` and checks the staged `Ai..Ui` diff against that allowlist. It may not
  change acceptance/evidence bytes, approved constants, validators, schemas,
  source, tests, fingerprints, tolerances, or historical status text.

Here and below, `{i}` is the decimal stage number `1`, `2`, or `3`; it is
notation in this memo, not a glob accepted by a tool. The exact files are the
three paths enumerated in Sections 7.2--7.4. `E`, `A`, and `U` are
single-parent commits. A combined phase letter, a merge commit, an artifact
replacement, or a successor with another changed path is invalid.

The status-only paths authorized in a `U` successor are:

- `Fix.md` (dated acceptance text; row stays ROADMAP until whole-row closure);
- `PostTier8RemediationPlan.md` (the accepted stage's WP-8 ledger only);
- `docs/development/sci005_beam_physics_plan.md` (append-only acceptance note);
- `docs/development/beam_physics_scope.md` (only the accepted stage's rows);
- `docs/changelog.rst`;
- `docs/migration_guide.md` if acceptance found its current wording incomplete;
  and
- `README.md` and `CLAUDE.md` only where the accepted stage makes a live
  support statement false.

The status-successor verifier permits a path only when its diff contains no
source, schema, test, tool, artifact, fingerprint, tolerance, or historical
acceptance-byte change. These prose edits do not authenticate acceptance: the
strict acceptance JSON and both approved constants do. The final all-stage
closure successor `C` directly parents `U3`, reconciles the complete scope and
support wording, and must not rewrite historical acceptance text. The status
paths above are not writable by an `R`, `S`, `E`, or `A` commit merely because
it intends eventual acceptance or closure.

No stage may edit `core/contraction.py`, add a compilation site, change the
kernel signature, edit `pixi.lock`, or add a gating workflow. Stage 3 reuses the
existing optional `crossval` environment; any dependency change requires a
design correction and independent review first.

The Stage-3 output contract is deliberately test-only at production level:
the existing generic beam snapshot, four-correlation result, HDF5 provenance,
and standard-visibility projections already carry the required information.
`core/result.py`, `io/hdf5.py`, `io/summary_json.py`,
`io/standard_visibility.py`, `io/uvfits.py`, `io/measurement_set.py`, and
`io/result_errors.py` are therefore not on the Stage-3 production writable
list. If a red output test proves one must change, implementation pauses for a
bounded design correction rather than silently expanding the slice.

## 8. Retained evidence schema and commit succession

### 8.1 Stage evidence records

Each `docs/development/sci005_stageN_evidence.json` is strict and has exactly
the following common fields, in this order:

```text
schema_version, stage, status, generated_at_utc,
design_sha, red_test_sha, source_sha, evidence_sha, working_tree_clean,
radiosim_version, python_version, platform, machine, pixi_environment,
pixi_lock_sha256, scientific_conventions, config_cases, analytic_invariants,
rejection_probes, backend_parity, solver_cases, output_cases,
fingerprint_diff, commands, artifacts, limitations, claims_not_licensed
```

`schema_version` is respectively `radiosim.sci005.stage1.v1`,
`radiosim.sci005.stage2.v1`, or `radiosim.sci005.stage3.v1`. Missing and unknown
fields fail validation. The bounded correction freezes the complete Stage-1
envelope below, and the accepted `D2` freezes the complete Stage-2 envelope
below it. Stage 3 retains its accepted common envelope, but its
stage-specific extensions are deliberately not frozen here; mandatory
`D3` independently freezes its exact normative rows, keys, types, and
cross-field predicates before its red slice. The corresponding `S2` or `S3`
then checks in the literal JSON Schema transcription before its artifact
successor.

For the following contract, JSON `number` means finite, non-boolean binary64;
`integer` means a non-negative, non-boolean JSON integer unless the field says
otherwise; `git_sha` means exactly 40 lower-case hexadecimal characters; and
`sha256` means exactly 64 lower-case hexadecimal characters. A `timestamp` is
canonical UTC
`YYYY-MM-DDTHH:MM:SSZ` with no fractional seconds. Every unspecified string is
non-empty. Every object has `additionalProperties: false`; every listed key is
required, including keys whose value may be null. Every count, residual, and
tolerance is non-negative. Arrays declared sorted are strictly lexical by the
named key and contain no duplicate key.

A `canonical_path` is a UTF-8, POSIX-separator, repository-relative path. It is
non-empty, has no leading slash, backslash, NUL, empty component, `.` component,
or `..` component, is byte-for-byte equal to its normalized form, and resolves
inside the repository without traversing a symlink. Every path field in the
evidence, acceptance, review, and certificate contracts is a
`canonical_path` unless an explicitly temporary CLI argument says absolute.
There are exactly two data-field exceptions: `commands[*].cwd` is the sentinel
`.` defined below, and `artifact_inputs[*].input_path` follows the conditional
repository/absolute Stage-3 rule in the evidence-generation transaction. No
other data-field path may be absolute or escape repository root.

`numeric_projection` is exactly
`{dtype, shape, c_order_sha256, minimum_abs, maximum_abs}`. Its `dtype` is one of
`float32`, `float64`, `float128`, `complex64`, `complex128`, or `complex256`;
`shape` is a non-empty array of integers; its product is positive; the digest
authenticates the C-contiguous raw bytes in that dtype; and extrema are finite
numbers over absolute values. The two extended literals are legal only when
NumPy exposes 16-byte `longdouble`, 32-byte `clongdouble`, and
`finfo(longdouble).nmant > 52`; an alias to float64 is not extended evidence.
`array_projection` is exactly
`{dtype, shape, c_order_sha256, minimum, maximum}`: `dtype` is `float32` or
`float64`, `shape` is the one-element array `[S]` with `S >= 1`, the digest
authenticates the owned C-order raw bytes, and extrema are numbers.
`antenna_projection` is exactly `{number, name}`, with a signed, non-boolean
JSON integer and a non-empty string.

Stage 1 appends, in order, the top-level arrays `pupil_profiles`,
`support_masks`, and `ruze_power_diagnostics` to the common field sequence.
Its exact top-level scalar contract is:

- `schema_version` is `radiosim.sci005.stage1.v1`, `stage` is integer `1`,
  `status` is `candidate`, `generated_at_utc` is a timestamp,
  `working_tree_clean` is true, and `evidence_sha` is JSON null;
- `design_sha`, `red_test_sha`, and `source_sha` are `git_sha`; the last names
  the clean checked-out Stage-1 implementation candidate `S1`;
- `radiosim_version`, `python_version`, `platform`, `machine`, and
  `pixi_environment` are strings, and `pixi_lock_sha256` is a `sha256`; and
- every array field is non-empty except that `limitations` may be empty.

The official `E1` artifact is generated on an available NumPy platform meeting
the extended-width predicate above. This is required because the deterministic
aperture contract retains wider-than-float64 behavior even though the optional
Ruze diagnostic rejects it.

`scientific_conventions` has exactly:

```text
pupil_profile_set: radiosim.circular_stage1_pupil_profiles.v1
aperture_normalization: unmodified_ideal_aperture_v1
aperture_axes: north_east_azimuth_north_through_east_v1
support_mask: radiosim.central_disk_outward_half_strip_ne.v1
zernike_surface: radiosim.real_unit_rms_disk_surface_height.v1
aperture_method: boundary_fitted_polar_gauss_legendre_v1
ruze_covariance: gaussian_one_over_e_surface_covariance_v1
ruze_method: poisson_paired_pupil_separation_v1
```

Every `config_cases` row has exactly
`{case_id, test_node_id, input_sha256, expected_outcome, observed_outcome,
resolved_scientific_sha256, exception_type, issue_code, exact_message, passed}`.
The first two are strings, the input is a `sha256`, outcomes are each
`accepted` or `rejected`, and `passed` is boolean. An accepted observation has a
non-null `sha256` resolution and null error fields; a rejected observation has
a null resolution and three non-null exact error strings. Expected and observed
outcomes must agree. Rows are sorted by unique `case_id`.

Every `analytic_invariants` row has exactly
`{case_id, invariant_id, backend, test_node_id, input_manifest_sha256, expected,
observed, max_abs_residual, max_rel_residual, atol, rtol, passed}`. The first
two and `test_node_id` are strings; `backend` is `numpy`, `jax`, `dask`, or
`independent_oracle`; the input is a `sha256`; expected and observed are
`numeric_projection`; the four metrics are numbers; and `passed` is boolean.
The projections have identical dtype and shape. Rows are sorted by unique
`case_id`. Exactly one `numpy` row each for
`extended_precision_unmodified_profile` and
`extended_precision_mask_plus_zernike` uses `complex256` observed/expected
projections and target-width tolerances; converting either computation or
oracle through complex128 invalidates the row.

Every `rejection_probes` row has exactly
`{case_id, config_path, exception_type, issue_code, exact_message, test_node_id,
input_sha256, passed}`: all except the digest and boolean are strings. It records
the full rendered message and the first owning issue path/code, not a substring.
Rows are sorted by unique `case_id`.

Every `backend_parity` row has exactly
`{case_id, backend, actual_device, real_dtype, complex_dtype, input_sha256,
reference_result_sha256, observed_result_sha256, max_abs_difference,
max_rel_difference, atol, rtol, passed}`. `backend` is `numpy`, `jax`, or `dask`;
the dtype pair is one of the two diagnostic pairs in Section 3.4.2; digests are
`sha256`; difference/tolerance fields are numbers; and `passed` is boolean.
Rows are sorted by `(case_id, backend)` and each retained case contains all
three backends. These parity rows own standard-width Jones/diagnostic parity;
the preceding NumPy analytic-invariant rows exclusively authenticate the
deterministic extended-width contract.

Every `solver_cases` row has exactly
`{case_id, effect, test_node_id, input_sha256, jones_sha256,
visibility_sha256, diagnostic_sha256, jones_call_count,
visibility_changed_element_count, visibility_change_expected, passed}`.
`effect` is one of `blockage`, `zernike`, `ruze_coherent_voltage`, or
`ruze_power_diagnostic_non_visibility`; hashes are `sha256` except that
`diagnostic_sha256` is null for the first three; counts are integers; and the
last two fields are boolean. A diagnostic-only row requires zero Jones calls,
zero changed visibility elements, and false `visibility_change_expected`.
Rows are sorted by unique `case_id`.

Every `output_cases` row has exactly
`{case_id, format, writer_test_node_id, reader_test_node_id, artifact_sha256,
in_memory_sha256, observed_projection_sha256, roundtrip_max_abs_difference,
tolerance, passed}`. `format` is one of `in_memory`, `summary_json`, `hdf5`,
`uvfits`, `measurement_set`, or `reader_projection`; hashes are `sha256` when
non-null; residual/tolerance are numbers when non-null; and `passed` is boolean.
Only `in_memory` may have null reader, artifact, round-trip, and tolerance
fields; every other row has them non-null. Rows are sorted by unique `case_id`.

Every `fingerprint_diff` row has exactly
`{environment, workload, old_scientific_sha256, new_scientific_sha256,
old_raw_cube_sha256, new_raw_cube_sha256, changed_element_count, maximum_delta,
change_expected, test_node_id, passed}`. Digests are `sha256`, count is integer,
maximum delta is a number, the two final state fields are boolean, and remaining
fields are strings. Rows are sorted by `(environment, workload)` and must
include enabled and disabled/default controls.

Every `commands` row has exactly
`{argv, cwd, pixi_environment, started_at_utc, duration_seconds, exit_code,
stdout_sha256, stderr_sha256}`. `argv` is a non-empty string array executed
without a shell; `cwd` is repository-relative `.`; start is a timestamp;
duration is a number; exit code is a signed non-boolean integer and must be zero
for candidate evidence; and the final fields are `sha256`. Command rows retain
execution order. Every `artifacts` row has exactly
`{path, sha256, media_type, role}`; path is a normalized repository-relative
string, digest is `sha256`, and role is one of `schema`, `command_log`,
`output`, `fingerprint`, or `auxiliary`. Artifact rows are sorted by unique
path. `limitations` and `claims_not_licensed` are sorted unique string arrays;
the latter must name Stage-1 acceptance, Stages 2/3, whole-row closure, and a
deterministic Ruze Jones/error voltage as claims not licensed.

The implementation checks in
`docs/development/sci005_stage1_evidence.schema.json` (new) as a literal JSON
Schema transcription of this normative key/type/cross-field contract. That
schema is on the Stage-1 writable list; an `artifacts` row with role `schema`
authenticates its exact bytes. The prose here wins if the red slice exposes a
transcription difference.

Each `pupil_profiles` row has exactly:

```text
case_id: string
model_kind: enum[circular_aperture, analytical_illumination,
                 numerical_illumination, rectangular_aperture,
                 elliptical_aperture, fits]
taper_kind: enum[uniform, parabolic, parabolic_squared, gaussian, cosine] | null
edge_taper_db: number | null
mixture_weight: number | null
profile_convention: enum[U, pU_plus_one_minus_p_P,
                         pU_plus_one_minus_p_P2] | null
hankel_convention: enum[two_J1_over_x, eight_J2_over_x2,
                        forty_eight_J3_over_x3,
                        weighted_linear_combination] | null
outcome: enum[accepted, rejected]
exception_type: string | null
issue_code: string | null
test_node_id: string
max_abs_residual: number | null
tolerance: number | null
```

Accepted rows require non-null profile/Hankel literals, residual, and tolerance,
and null exception/code. Rejected rows require null profile/Hankel/residual/
tolerance and non-null exact exception/code. `edge_taper_db` and
`mixture_weight` are null only where the selected model/taper has neither.

Each `support_masks` row has exactly
`{case_id, diameter_m, central_diameter_ratio, legs, probes,
union_control_id, antipodal_control_id, topology_sha256, test_node_id, passed}`.
`case_id`, `union_control_id`, `antipodal_control_id`, and `test_node_id` are
strings; `topology_sha256` is a `sha256`; diameter and ratio are numbers; and
`passed` is boolean. `legs` is an array of
exact objects `{position_angle_deg: number, width_m: number}` in resolved angle
order. `probes` is an array of exact objects
`{north_m: number, east_m: number, expected_transmitting: boolean,
observed_transmitting: boolean}` in authored probe order. Empty legs are legal;
empty probes are not.

Each `ruze_power_diagnostics` row has exactly
`{case_id, resolved_aperture_scientific_sha256, diagnostic,
direct_pair_oracle, limit_oracles, test_node_ids}`. The first is a string, the
second a sha256, and the last a non-empty array of unique strings in lexical
order. `direct_pair_oracle` is exactly
`{test_node_id: string, aperture_node_count: integer,
direction_count: integer, max_abs_residual: number, tolerance: number}`; both
counts are positive.
`limit_oracles` is an array, ordered by `kind`, of exact objects
`{kind, test_node_id, input_sha256, max_abs_residual, tolerance}`, where the
digest is a `sha256`, `kind` is one of
`mu_first_order`, `infinite_correlation_length`, `asymmetric_phase`,
`entire_plane_shift`, `gaussian_characteristic_function`, or
`covariance_only_counterexample`, and the remaining fields have the preceding
types. Every row contains each of those six kinds exactly once. Rows are sorted
by unique `case_id`.

The `diagnostic` object is the exact JSON projection of the public record. It
has scalar keys `schema_version`, `method`, `covariance_convention`, and
`normalization_convention` with the exact literals in Section 3.4.2;
`antenna_id: antenna_projection`; numeric keys `frequency_hz`, `time_mjd`,
`rms_surface_error_m`, and `correlation_length_m`; projection keys
`altitude_rad`, `azimuth_rad`, `coherent_main_power`,
`total_ensemble_power`, and `scattered_power`; and `convergence` below. It does
not embed array values or a complex voltage.

The `convergence` object has every key in Section 3.4.2's declared order.
`real_dtype`, `complex_dtype`, and `aperture_method` use their exact declared
enums; `aperture_topology_sha256` and `separation_topology_sha256` are each a
sha256. Integer keys are exactly the
following fields:

```text
poisson_first_order, poisson_last_order, poisson_term_count,
separation_radial_order, separation_angular_order_max,
separation_node_count, separation_evaluation_count,
aperture_partition_count, aperture_topology_breakpoint_count,
aperture_refinement_count, aperture_max_node_count,
fhat_evaluation_count, phase_product_count, batch_size, estimated_peak_bytes
```

All remaining convergence keys not already classified as dtype/method strings
or a digest are JSON numbers. Cross-field validation enforces the zero-term
Poisson case, retained interval count, tail sum, separation truncation bound,
power-of-two angular order, caps,
two residuals, dtype pair, non-negative powers, amplitude/total bounds, and
exact result-dtype balance from Section 3.4. Unknown or missing projection keys,
nulls, a backend field, a `converged` boolean, clipping metadata, FFT metadata,
or any direction-sized complex value fail authentication.

#### Stage-2 evidence envelope

Stage 2 appends, in order, the top-level arrays `squint_frequency_laws`,
`squint_geometries`, `native_feed_factorizations`, `stokes_v_leakages`, and
`squint_setup_rejections` to the common field sequence. Its exact top-level
scalar contract is the Stage-1 contract with these substitutions:
`schema_version` is `radiosim.sci005.stage2.v1`, `stage` is integer `2`, and
`source_sha` names the clean checked-out Stage-2 implementation candidate
`S2`. Every array field is non-empty except that `limitations` may be empty.
The official `E2` artifact is generated on an available NumPy platform
meeting the Section 8.1 extended-width predicate, because the envelope
requires one `complex256` factorization row below. `S2` checks in
`docs/development/sci005_stage2_evidence.schema.json` as the literal JSON
Schema transcription of this envelope, authenticated by an `artifacts` row
with role `schema`; the prose here wins if the red slice exposes a
transcription difference. `S2` extends `tests/unit/test_sci005_evidence.py`
with the Stage-2 validation and synthetic-document tests while changing no
Stage-1 validation logic, constant, or synthetic fixture.

Stage 2's `scientific_conventions` has exactly:

```text
squint_frequency_law: cotton_uson_exact_v1
squint_direction: feed_ray_plus_half_pi_north_through_east_v1
squint_beam_frame: pointing_then_squint_great_circle_v1
squint_factorization: receptor_conjugated_native_diagonal_v1
```

The common rows specialize for Stage 2 as follows. `analytic_invariants`
requires no extended-width case identifiers — the Stage-1 sentence naming
`extended_precision_unmodified_profile` and
`extended_precision_mask_plus_zernike` is scoped to Stage 1, and Stage 2's
one required extended row lives in `native_feed_factorizations` below.
`solver_cases.effect` is one of exactly `squint_point` or `squint_healpix`;
each appears at least once, `diagnostic_sha256` is null on every row,
`visibility_change_expected` is true, and
`visibility_changed_element_count` is positive. `backend_parity` retains
the common all-three-backends rule with the `float64`/`complex128` pair for
at least one squint-enabled case. `output_cases` contains at least one
`in_memory` row and one `hdf5` row for a squint-enabled workload.
`rejection_probes` contains each of the five frozen Section 4.1.1 codes —
`beam.squint.identity_block`, `beam.squint.reference_frequency_domain`,
`beam.squint.offset_domain`, `beam.squint.mechanical_angle_domain`, and
`beam.squint.unsupported_beam_family` — at least once, each with its exact
frozen path and message. `fingerprint_diff` keeps the common
enabled-plus-disabled-control rule. `claims_not_licensed` must contain the
exact members `SCI-005 Stage-2 acceptance`, `SCI-005 Stage 3`,
`SCI-005 whole-row closure`, and
`a full cross-polar or measured-efield beam response`; for Stage 2 this
list supersedes the Stage-1-scoped member rule in the common
`commands`/`artifacts` paragraph above, which is scoped to Stage 1.

Stage-1 closure also exposed one generator defect this gate records and
Stage 2 must correct inside its already-writable
`tools/sci005_stage_evidence.py`: the shipped `resolve_design_sha` for
stages other than 1 resolves `HEAD^^{commit}` — git's peel form of
`HEAD^`, not the grandparent — and would therefore record
`design_sha == red_test_sha` for Stage 2. `S2` corrects it to resolve the
exact grandparent of clean `HEAD == S2`, honoring the Section 8.1 rule
that `Di` is the direct parent of `Ri`, and the Stage-2 validation in
`tests/unit/test_sci005_evidence.py` independently authenticates
`red_test_sha^ == design_sha`, `source_sha^ == red_test_sha`, and
`design_sha != red_test_sha` from Git objects, so this defect class
cannot pass validation again.

Each `squint_frequency_laws` row has exactly
`{case_id, reference_frequency_hz, per_feed_offset_deg_at_reference,
samples, small_angle_control_frequency_hz, small_angle_abs_separation,
max_abs_residual, tolerance, test_node_id, passed}`. `case_id` and
`test_node_id` are strings; both frequencies are positive numbers; the
offset is a number in the open interval `(0, 90)`; the two residual fields
are numbers; `tolerance` is a positive number; and `passed` is boolean.
`samples` is an array of at least three exact objects
`{frequency_hz, expected_offset_rad, observed_offset_rad,
small_angle_offset_rad}`, strictly increasing in positive `frequency_hz`,
with every offset a number in the open interval `(0, pi/2)`. Cross-field
validation recomputes `expected_offset_rad` as the binary64
`asin((reference_frequency_hz / frequency_hz) *
sin(radians(per_feed_offset_deg_at_reference)))` and
`small_angle_offset_rad` as the binary64
`radians(per_feed_offset_deg_at_reference) * reference_frequency_hz /
frequency_hz`, requiring each recorded value to agree within an absolute
difference of `1e-15`; recomputes `max_abs_residual` as the largest
`abs(observed_offset_rad - expected_offset_rad)` over the samples,
requiring the recorded value to equal the recomputation in binary64 and to
satisfy `max_abs_residual <= tolerance`;
requires `small_angle_control_frequency_hz` to equal one sample's
`frequency_hz` and differ from `reference_frequency_hz`; requires
`small_angle_abs_separation` to equal that sample's
`abs(small_angle_offset_rad - expected_offset_rad)` in binary64 and to be
at least `8 * tolerance`. Rows are sorted by unique `case_id`.

Each `squint_geometries` row has exactly
`{case_id, mount_type, mechanical_feed_position_angle_deg,
positive_native_feed, receptor_basis, parallactic_angle_rad,
boresight_altitude_rad, frequency_hz, resolved_offset_rad, probes,
test_node_id, passed}`. `mount_type` is one of the five Section 4.1 mount
literals; the mechanical angle is a number in `(-180, 180]`;
`positive_native_feed` is one of `x`, `y`, `r`, `l` and `receptor_basis`
one of `linear`, `circular`, with the label belonging to the basis;
`parallactic_angle_rad` is a signed finite number;
`boresight_altitude_rad` is a number; `frequency_hz` is a positive number;
`resolved_offset_rad` is a number in the open interval `(0, pi/2)`; and
`probes` is a non-empty array of exact objects
`{kind, observed, bound, relation, passed}` whose `kind` values are unique
within the row. `kind` is one of exactly:

```text
orthogonality_dot_abs, handedness_plus_half_pi_residual_rad,
midpoint_center_residual_rad, total_separation_residual_rad,
mount_rotation_residual_rad, opposite_mount_sign_min_abs_delta_rad,
mechanical_rotation_residual_rad, feed_sign_reversal_center_residual_rad
```

`observed` and `bound` are numbers; `relation` is `le` for every kind
except `opposite_mount_sign_min_abs_delta_rad`, whose relation is `ge`; a
probe's `passed` is boolean and equals `observed <= bound` for `le` and
`observed >= bound` for `ge`. Across the whole array every one of the eight
kinds appears at least once, at least one row has `mount_type` `alt-az`,
and at least one row has `mount_type` `fixed`. Rows are sorted by unique
`case_id`.

Each `native_feed_factorizations` row has exactly
`{case_id, receptor_basis, feed_rotation_deg, parallactic_angle_rad,
positive_native_feed, b_plus, b_minus, expected, observed,
factorization_max_abs_residual, chain_order_max_abs_residual,
order_control_max_abs_difference, atol, test_node_id,
passed}`. `receptor_basis` and `positive_native_feed` follow the geometry
row rules; `feed_rotation_deg` and `parallactic_angle_rad` are signed
finite numbers; `b_plus` and `b_minus` are exact objects
`{real, imag}` of signed finite numbers and differ as complex pairs;
`expected` and `observed` are `numeric_projection` values with identical
dtype and shape, dtype `complex128` or `complex256` and shape `[2, 2]` or
`[S, 2, 2]` with `S >= 1`; the three residual fields are numbers; and
`atol` is a positive number. Here
`expected` projects an independently composed `C^dagger diag(b) C` sandwich
and `observed` projects the production composition;
`factorization_max_abs_residual` is the largest entrywise absolute
difference between `observed` and `expected`;
`chain_order_max_abs_residual` is the largest
entrywise absolute difference between the production `C @ E @ P` and the
physical `D_b @ C @ P`; and `order_control_max_abs_difference` is the
largest entrywise absolute difference between `C @ E @ P` and
`C @ P @ E`. Cross-field validation requires
`factorization_max_abs_residual <= atol`,
`chain_order_max_abs_residual <= atol`,
and `order_control_max_abs_difference >= max(1e-3, 1024 * atol)`. The
array contains at least one `circular` row, at least one `linear` row with
non-zero `feed_rotation_deg`, and exactly one row with `case_id`
`extended_precision_native_feed_factorization` whose projections are
`complex256`; that row's production composition and independent oracle
never pass through `complex128`, and narrowing either invalidates the row.
Rows are sorted by unique `case_id`.

Each `stokes_v_leakages` row has exactly
`{case_id, positive_native_feed, reversed_case_id, frequency_hz,
probe_altitude_rad, probe_azimuth_rad, observed_v_over_i, expected_sign,
observed_sign, min_abs_v_over_i, test_node_id, passed}`.
`positive_native_feed` is `r` or `l`; `reversed_case_id` is a string naming
another row; `frequency_hz` is a positive number; the probe coordinates are
signed finite numbers naming a direction on the positive-squint side of the
boresight; `observed_v_over_i` is a signed finite number; `expected_sign`
and `observed_sign` are the exact signed integers `-1` or `1`; and
`min_abs_v_over_i` is a positive number. Cross-field validation requires
`expected_sign` to be `1` exactly when `positive_native_feed` is `r`;
`observed_sign == expected_sign`; the sign of `observed_v_over_i` to equal
`observed_sign`; `abs(observed_v_over_i) >= min_abs_v_over_i`; and the row
named by `reversed_case_id` to exist, name this row reciprocally, and carry
the opposite `positive_native_feed` and the opposite `observed_sign`. Rows
are sorted by unique `case_id`.

Each `squint_setup_rejections` row has exactly
`{case_id, case_kind, exception_type, exact_message, test_node_id,
passed}`, all strings except the boolean, with `case_kind` one of exactly
`unknown_antenna`, `duplicate_antenna`, `frequency_domain`,
`receptor_basis`, or `boresight_degenerate`. Every kind appears at least
once across the array. `exception_type` is frozen per kind:
`UnknownBeamAntennaError`, `DuplicateBeamAssignmentError`,
`SquintFrequencyDomainError`, `SquintReceptorBasisError`, and
`BeamAngularDomainError` respectively, and a `boresight_degenerate` row's
`exact_message` equals the Section 4.2.1 frozen literal. Rows are sorted by
unique `case_id`.

#### Evidence-generation transaction

The exact evidence invocation is:

```text
pixi run python tools/sci005_stage_evidence.py generate \
  --stage <1|2|3> --measurement-record <absolute-temporary-record.json>
```

The read-only measurement record is a strict UTF-8 JSON object. Its common
keys are exactly:

```text
generated_at_utc, scientific_conventions, config_cases,
analytic_invariants, rejection_probes, backend_parity, solver_cases,
output_cases, fingerprint_diff, commands, artifact_inputs, limitations,
claims_not_licensed
```

Stage 1 appends exactly `pupil_profiles`, `support_masks`, and
`ruze_power_diagnostics`; Stage 2 appends exactly the five arrays the
Stage-2 envelope above freezes, in that order; `D3` freezes the Stage-3
input extensions with its evidence rows. Every supplied value uses the
corresponding evidence shape above. Missing/unknown/duplicate keys, a
non-regular or symlink input, a non-finite value, an incomplete row set, a false
row, or a nonzero command fails before a repository write.

Every `artifact_inputs` row has exactly
`{path, input_kind, input_path, media_type, role}` and rows are sorted by unique
target `path`. `input_kind` is `repository` or
`stage3_crossvalidation_temp`. For `repository`, both paths are the same
`canonical_path` already present at exact `Si`. The temporary kind is legal
exactly once, only at Stage 3, only for the dated cross-validation target in
Section 7.4, and its `input_path` is an absolute regular file outside repository
root produced from clean `S3` by the exact D3-frozen cross-validation command.
The tool validates that artifact's strict schema, `source_sha == S3`, package
versions, input hashes, and target basename before importing it. Host temporary
paths never enter evidence or scientific identity. The generator derives the
sorted evidence `artifacts` rows and their raw SHA-256 values; the caller cannot
supply a digest.

The generator derives and forbids caller overrides of `schema_version`,
`stage`, `status`, `design_sha`, `red_test_sha`, `source_sha`, `evidence_sha`,
`working_tree_clean`, the runtime/platform/Pixi fields, `pixi_lock_sha256`, and
`artifacts`. For Stage 1 it resolves `D1` only from the immutable dependency
validator constant above; for Stages 2 and 3 it resolves `Di` as the direct
parent of `Ri`. It resolves `Ri` and exact clean `Si` from the mandatory
ancestry, uses `status: candidate`, `source_sha: Si`, `evidence_sha: null`, and
`working_tree_clean: true`, and authenticates the current lock and every input
byte before constructing output.

Evidence JSON uses UTF-8 without a byte-order mark, LF endings,
`ensure_ascii=false`, `allow_nan=false`, the Section 8.1 key order, two-space
indentation, and one final newline. From globally clean `HEAD == Si`, the tool
prepares in memory the absent evidence JSON and
`tests/unit/test_sci005_evidence.py` with exactly the target stage's two `None`
sentinels replaced by `Si` and `sha256(evidence JSON)`. Stage 3 also prepares
the exact cross-validation target bytes from the authenticated temporary input.
It writes all target files through same-directory temporaries, restores every
original byte and removes every new target on any failure, then requires the
working diff to be exactly the two Section 7.5 `Ei` paths, or those two plus the
single Stage-3 cross-validation path. Success is silent. This transaction owns
the complete admissible pre-`Ei` diff; manual artifact copying or pinning is
forbidden.

Stage 3 additionally embeds the cross-validation artifact path and digest,
reference package versions, input-content hashes, explicit convention mappings,
per-correlation residuals, and unresolved differences. The evidence document
does not copy or summarize a value without authenticating the underlying
artifact digest.

### 8.2 Independent acceptance records

The retained acceptance paths are exactly:

- Stage 1: `docs/development/sci005_stage1_acceptance.json`, validated against
  `docs/development/sci005_stage1_acceptance.schema.json`, generated and
  verified by `tools/sci005_stage1_acceptance.py`, and authenticated by
  `tests/unit/test_sci005_stage1_acceptance.py`;
- Stage 2: `docs/development/sci005_stage2_acceptance.json`, validated against
  `docs/development/sci005_stage2_acceptance.schema.json`, generated and
  verified by `tools/sci005_stage2_acceptance.py`, and authenticated by
  `tests/unit/test_sci005_stage2_acceptance.py`; and
- Stage 3: `docs/development/sci005_stage3_acceptance.json`, validated against
  `docs/development/sci005_stage3_acceptance.schema.json`, generated and
  verified by `tools/sci005_stage3_acceptance.py`, and authenticated by
  `tests/unit/test_sci005_stage3_acceptance.py`.

Each acceptance schema is a checked-in JSON Schema 2020-12 transcription of
this section. It has `additionalProperties: false` at every object, lists every
key as required even when null is permitted, rejects booleans as numbers, and
uses the `git_sha`, `sha256`, `timestamp`, finite-number, sorted-array, and
`canonical_path` meanings already frozen in Section 8.1. Cross-field, Git-object,
raw-byte digest, ordering, and diff-authority predicates that JSON Schema cannot
express are mandatory validator checks, not advisory prose.

Every acceptance JSON has exactly these top-level keys, in this order:

```text
schema_version, stage, verdict, generated_at_utc,
implementation_identity, reviewer_identity, reviewer_independent,
design_sha, red_test_sha, source_sha, evidence_commit_sha,
evidence_artifact_path, evidence_artifact_sha256,
evidence_schema_path, evidence_schema_sha256, toolchain,
acceptance_commit_sha, acceptance_commit_sha_reason, successor_unlocks,
reviewed_artifacts, rederived_oracles, review_checks, commands, blockers,
accepted_limitations, claims_not_licensed
```

The schema literals are respectively
`radiosim.sci005.stage1-acceptance.v1`,
`radiosim.sci005.stage2-acceptance.v1`, and
`radiosim.sci005.stage3-acceptance.v1`; `stage` is respectively integer `1`,
`2`, or `3`. `verdict` is `ACCEPT` or `REJECT`. Both identities are non-empty
role/task identifiers. A retained `ACCEPT` requires
`reviewer_independent: true` and unequal identities; an implementation agent,
its delegated child, or an agent that authored `R`, `S`, or `E` is not
independent merely because it uses another label.

The four commit fields are `git_sha` values. They name the phase's accepted
design `Di`, phase red commit `Ri`, phase source commit `Si`, and evidence
commit `Ei`; Stage 1 therefore records `D1`, Stage 2 records `D2`, and Stage 3
records `D3`.
The evidence artifact and evidence schema paths are exactly
`docs/development/sci005_stage{i}_evidence.json` and
`docs/development/sci005_stage{i}_evidence.schema.json`; both digest fields are
raw-file `sha256` values from `Ei`. `acceptance_commit_sha` is JSON null. Its
reason for `ACCEPT` is exactly one of:

- Stage 1: `self-reference: U1 binds the containing A1 commit`;
- Stage 2: `self-reference: U2 and SCI004.M3 bind the containing A2 commit`;
  or
- Stage 3: `self-reference: U3 binds the containing A3 commit`.

For `REJECT`, the reason is exactly
`not-applicable: REJECT creates no A commit`, regardless of stage.

For an `ACCEPT`, `successor_unlocks` is exactly the sorted array
`["SCI005.U1"]`, `["SCI004.M3", "SCI005.U2"]`, or `["SCI005.U3"]` for
Stages 1, 2, and 3 respectively. For a `REJECT` it is empty. An unlock is a
permission to begin the named red or closure gate after all of that gate's
other dependencies pass; it is not implementation or row acceptance.

`toolchain` has exactly:

```text
evidence_generator_path, evidence_generator_git_blob,
evidence_validator_path, evidence_validator_git_blob,
acceptance_generator_path, acceptance_generator_git_blob,
acceptance_validator_path, acceptance_validator_pre_a_git_blob,
acceptance_schema_path, acceptance_schema_sha256
```

The evidence paths are exactly `tools/sci005_stage_evidence.py` and
`tests/unit/test_sci005_evidence.py`. The acceptance paths and schema path are
the phase-local paths listed at the start of this section. Every `git_blob`
value is the 40-hex Git blob object name read from exact `Ei`; the schema digest
is the raw-file `sha256` from `Ei`. The pre-`A` acceptance-validator blob is
intentional: `Ai` changes only its two approved constants, and the validator
authenticates that token-excluded change against this blob.

Every `reviewed_artifacts` row has exactly
`{path, sha256, source_sha, authenticated}`. Rows are sorted by unique `path`;
the digest is a raw-file `sha256`, `source_sha` equals the top-level `source_sha`,
and `authenticated` is boolean. The path set is exactly the union of the
evidence JSON itself, the evidence schema, the acceptance schema, and every
path named by the evidence JSON's `artifacts` array. An `ACCEPT` requires every
row authenticated and every retained byte reachable by its recorded path or
Git object; a green workflow summary is not an artifact.

Every `rederived_oracles` row has exactly
`{oracle_id, method, observed, fixed_limit, units, passed}`. The identifier,
method, and units are non-empty strings; observed and limit are non-negative
finite numbers; and `passed` is boolean and equals `observed <= fixed_limit`.
Rows are sorted by unique `oracle_id`. An `ACCEPT` contains exactly these IDs:

- Stage 1: `blocked_aperture_transform`, `ruze_limit_oracle`,
  `ruze_pair_oracle`, `unmodified_profile_transform`, and
  `zernike_phase_transform`;
- Stage 2: `mechanical_feed_rotation`, `native_feed_factorization`,
  `noncommuting_chain_order`, and `squint_frequency_law`; and
- Stage 3: `common_efield_normalization`, `ludwig3_basis_conversion`,
  `noncommuting_chain_order`, `receptor_output_basis_factorization`, and
  `standard_output_roundtrip`.

The reviewer derives these values without importing the production helper
under review. A digest-only equality, a re-run of the implementation's own
oracle, or an omitted required identifier fails acceptance.

Every `review_checks` row has exactly
`{check_id, method, expected_outcome, observed_outcome, passed}`. All strings
are non-empty, rows are sorted by unique `check_id`, and every `ACCEPT` has
exactly `artifact_authentication`, `default_disabled_fingerprints`,
`diff_authority`, `gate_replay`, `production_data_flow`, and
`typed_rejection`, all with `passed: true` and equal expected/observed outcomes.
The typed-rejection method names the concrete exception, issue code, exact
message, and config path. The data-flow method names the inspected public entry
point and final `E` composition site. The fingerprint check authenticates old
and new cube bytes as well as both scientific digests.

Acceptance `commands` uses Section 8.1's exact command-row schema and retains
execution order; every exit code is zero for `ACCEPT`. Every `blockers` row has
exactly `{blocker_id, requirement_id, evidence, required_remediation}`, with
four non-empty strings, sorted by unique `blocker_id`. `ACCEPT` requires no
blockers and every oracle/check true. `REJECT` requires at least one concrete
blocker, grants no successor unlock, and is not committed to the canonical
acceptance path. `accepted_limitations` and `claims_not_licensed` are sorted
unique string arrays. The latter is non-empty and names all later stages,
whole-row closure, and every diagnostic or physical claim not established by
the accepted stage.

The independent reviewer supplies one temporary `review_record` object with
exactly these keys:

```text
generated_at_utc, implementation_identity, reviewer_identity,
reviewer_independent, verdict, rederived_oracles, review_checks, commands,
blockers, accepted_limitations, claims_not_licensed
```

Those values use the acceptance shapes above. The generator derives every
commit, path, digest, toolchain, reviewed-artifact, self-reference, and unlock
field; the caller cannot override one. The exact accepting invocation is:

```text
pixi run python tools/sci005_stage{i}_acceptance.py generate \
  --review-record <absolute-temporary-review-record.json>
```

For `REJECT`, the same command additionally requires
`--reject-output <absolute-temporary-rejection.json>`; that output must be
outside repository root and must not already exist. The review record is a
read-only absolute regular-file input and may also live outside the repository.

The generator emits UTF-8 without a byte-order mark, uses LF endings,
`ensure_ascii=false`, `allow_nan=false`, the declared insertion order,
two-space indentation, and a single final newline. It runs only from a globally
clean exact `Ei` and first invokes the active evidence validator. For `ACCEPT`,
it prepares in memory both the previously absent canonical JSON bytes and the
phase validator bytes with exactly `APPROVED_EVIDENCE_SHA: None -> Ei` and
`APPROVED_ACCEPTANCE_ARTIFACT_SHA256: None -> sha256(JSON)` substitutions. It
writes both through same-directory temporary files, restores the original
validator and removes the new artifact on any failure, then requires the
working diff to contain exactly those two paths and those two literal changes.
Success is silent on stdout and stderr. This all-or-rollback generation owns
the complete admissible pre-`Ai` diff; no manual pinning step is permitted.

For `ACCEPT` it refuses an overwrite, a non-independent reviewer, an incomplete
required row set, any false oracle/check, a nonzero command, a blocker, a stale
tool/schema blob, or any retained/member `canonical_path` outside repository
root. For `REJECT` it instead requires a non-empty blocker array, permits false
oracle/check rows and nonzero commands, changes no repository byte or approved
constant, and writes only the explicit temporary rejection path above. Duplicate
JSON keys, symlink inputs/outputs, non-regular inputs, and non-finite values fail
both modes.

The acceptance validator operates on named Git objects rather than whichever
file happens to be checked out. In its `S`/`E` state, both approved constants
are `None`, the official acceptance path must be absent, and synthetic strict
schema/digest/ancestry fixtures must pass. In its `A` state it requires the
constants to equal the exact `Ei` and raw acceptance-file SHA-256, locates the
unique artifact-introducing `Ai`, authenticates every field and tool blob,
checks `Ai^ == Ei`, and requires the `Ei..Ai` diff to be exactly the new
acceptance JSON plus the two literal constant replacements in Section 7.5. It
never infers acceptance from an appended memo sentence or the first similarly
named file.

### 8.3 Commit succession and direct-parent bindings

Commit succession is mandatory:

1. `D1`, `D2`, or `D3`: independently accepted design-only gate for the
   following stage;
2. `G1`: clean Stage-1 WP-7 dependency-gate tip, with `D1` and accepted WP-7
   CPU `A` as authenticated ancestors and no SCI-005 red/source byte;
3. `R1`, `R2`, or `R3`: red tests plus recorded pre-fix failure, no production
   implementation;
4. `S1`, `S2`, or `S3`: one coherent stage implementation, including docs and
   enabled-effect pin changes;
5. `E1`, `E2`, or `E3`: evidence-only successor generated from clean `S`, with
   the artifact recording `source_sha=S`;
6. `A1`, `A2`, or `A3`: independent acceptance successor, with only the exact
   artifact and constant-only validator update; and
7. `U1`, `U2`, or `U3`: status/ledger successor with only the prose authority
   in Section 7.5.

The evidence file cannot truthfully contain its own future Git SHA before it is
committed. `evidence_sha` is JSON null in the file. The exact direct-parent
chain is `D1 ->* G1 -> R1 -> S1 -> E1 -> A1 -> U1 ->* D2 -> R2 -> S2 -> E2
-> A2 -> U2 -> D3 -> R3 -> S3 -> E3 -> A3 -> U3 -> C`. The first starred edge
is ancestor reachability through separately authorized, independently accepted
programme commits, including the WP-7 dependency/interface succession; every
unstarred arrow is the sole direct-parent edge and no named commit is a merge.
Across `D1..G1`, the exact `D1` memo blob, the `Fix.md` SCI-005 row, the WP-8
subsection/ledger cells, and the Stage-1 scope rows remain unchanged. Every
Section 7.2 path marked `new` or `successor only` remains absent; independently
accepted programme commits may otherwise change shared paths and those bytes
become the `G1` red baseline.

The second starred edge, `U1 ->* D2`, is ancestor reachability from `U1` to
`D2` through zero or more separately authorized status-prose commits plus
the superseded commits of the Stage-2 design succession. Every commit in
`U1..D2` other than `D2` itself is a single-parent non-merge of exactly one
of two kinds. A status-prose commit's parent-relative diff touches only
Section 7.5 status paths and contains no source, schema, test, tool,
artifact, fingerprint, tolerance, or historical-acceptance-byte change. A
superseded design commit is an earlier operative `D2` that a later,
independently accepted, header-recorded correction supersedes; the
superseded design commits form the terminal segment of the interval, each
directly parenting the next design commit, and each touches exactly the
paths its own header record names. Across the interval every Section 7.3
path marked `new` or `successor only` remains absent, and the retained
Stage-1 evidence and acceptance artifacts, their schemas, their validators,
their tools, and every approved digest constant remain byte-identical to
`U1`, except through the one bounded four-line dependency-validator
deletion the design succession's header records authorize. The observed
interval at this gate is exactly two commits: the one changelog heading
repair recorded in this memo's header
(`3e336095009e72bf4ae6064d5e97d381e063258f`), followed by the superseded
original gate commit (`3d60b6f428e9ac94407f50dfd9153aad27d5e098`), which
landed the Stage-2 memo amendment plus the bounded four-line
dependency-validator deletion and which the heading-and-binding correction
supersedes. The operative `D2` — the commit containing that correction —
touches exactly `docs/development/sci005_beam_physics_plan.md` and no other
path; the four-line deletion was performed once, by the superseded gate
commit, and is never repeated.

Thus `R1^ == G1`, `Si^ == Ri`, `Ei^ == Si`,
`Ai^ == Ei`, `Ui^ == Ai`,
`R2^ == D2`, `D3^ == U2`, `R3^ == D3`, and `C^ == U3`. The
validator also requires `D1` and the WP-7 certificate's `acceptance_commit` to
be ancestors of `G1` and reauthenticates the exact certificate bytes retained
by `R1`. It requires the `R1` design-binding literal to equal every Stage-1
`design_sha` and remain byte-identical through `S1/E1/A1/U1`; `D2` and `D3`
remain unambiguous direct parents of their red commits. The Stage-2
acceptance validator authenticates the corrected edge from Git objects
alone: it locates `A1` as the unique commit introducing the Stage-1
acceptance artifact authenticated by the Stage-1 approved constants,
requires the unique commit on `D2`'s first-parent ancestry whose direct
parent is `A1` to be `U1` and to satisfy the committed Stage-1
`verify-status` form, requires `A1` to be an ancestor of `D2` through the
committed Stage-1 `verify` form, and checks every interval commit in
`U1..D2` other than `D2` against the status-prose rule above.

At `Ei`, the evidence validator authenticates raw evidence bytes, requires its
`source_sha == Si` and `evidence_sha == null`, requires `Ei^ == Si`, and
requires the `Si..Ei` diff to match Section 7.5. At `Ai`, the acceptance record
binds the exact `Ei`, raw evidence-artifact SHA-256, evidence-schema SHA-256,
and pre-`A` tool blobs. The acceptance validator then authenticates the raw
acceptance bytes through its approved digest constant and requires `Ai^ == Ei`.
`Ui` binds the otherwise self-referential `Ai` through its direct parent; the
next `D` or `C` then binds `Ui`. Any absent edge, wrong raw digest,
previous-artifact mutation, combined phase letter, production change in
`E/A/U`, or validator change beyond the named constant literals invalidates
the succession and grants no unlock.

## 9. Verification and independent acceptance gates

Every phase-local acceptance tool exposes the same read-only verifier:

```text
pixi run python tools/sci005_stage{i}_acceptance.py verify \
  --acceptance-commit <Ai> --descendant <SHA-or-HEAD>
```

It requires both arguments to resolve to commits, requires `Ai` to be an
ancestor of the named descendant, reads all bound files from Git objects, runs
the phase validator, and emits exactly one canonical UTF-8 JSON line with keys
in this order:

```text
schema_version, stage, acceptance_commit_sha, acceptance_artifact_path,
acceptance_artifact_sha256, evidence_commit_sha, evidence_artifact_path,
evidence_artifact_sha256, source_sha, verdict, successor_unlocks
```

The schema literal is `radiosim.sci005.stage-acceptance-certificate.v1`; all
commit and digest values use the strict encodings above. A zero exit requires
the retained verdict `ACCEPT`, exact ancestry/diff/digest validation, and the
stage-specific unlock array. The line uses JSON string escaping with
`ensure_ascii=false`, no optional whitespace (`separators=(",", ":")`), the
declared insertion order, and one final LF. Failure emits no certificate on
stdout and exits non-zero; stderr begins with exactly one of
`SCI005_ACCEPTANCE_ARGUMENT`, `SCI005_ACCEPTANCE_SCHEMA`,
`SCI005_ACCEPTANCE_ANCESTRY`, `SCI005_ACCEPTANCE_DIGEST`,
`SCI005_ACCEPTANCE_DIFF_AUTHORITY`, or `SCI005_ACCEPTANCE_VERDICT`, followed by
one colon, one space, and the detail.

A status successor is checked before and after commit with the same tool:

```text
pixi run python tools/sci005_stage{i}_acceptance.py verify-status \
  --acceptance-commit <Ai> --status-commit <Ui-or-INDEX>
```

The exact sentinel `INDEX` requires `HEAD == Ai` and validates the staged diff;
a Git SHA requires `Ui^ == Ai`. Success is silent. After its `D` gate, the next
stage's red preflight reruns the committed form; `C` does the same after Stage
3 before accepting its direct parent.

This verifier and certificate are the complete SCI-005 export for the WP-9 M3
dependency. This memo does not grant SCI-004 writable authority. Before an M3
red commit may exist, the independently accepted SCI-004 design must name an
exact receiving artifact field set, writable path, generator, and validator
that retain `acceptance_commit_sha`, `acceptance_artifact_sha256`, and the raw
SHA-256 of this certificate line including its final LF. If that receiving
contract is absent, M3 pauses for a bounded SCI-004 design correction and fresh
independent review. Its validator must run the Stage-2 command with accepted
`A2` and the proposed M3 red commit as descendant and require the three retained
values to equal the certificate and accepted artifact. The `SCI004.M3` unlock
literal alone, ledger prose, a green workflow, or a Stage-2 `S/E` commit is
insufficient.

Each source candidate runs, at minimum:

```text
pixi run test -- <stage-focused tests>
pixi run test -- tests/unit/
pixi run test -- tests/integration/
pixi run test -- -m "not slow"
pixi run doctest
pixi run lint
pixi run check-format
pixi run typecheck
make -C docs clean html SPHINXOPTS="-W --keep-going"
pixi run test -- tests/unit/test_tier8_release_acceptance.py
git diff --check
```

Stage 3 additionally runs the optional comparison with
`pixi run --environment crossval -- python -m pytest tests/crossvalidation/ -m crossval`.
It remains non-gating, but a retained Stage-3 artifact is required for
acceptance. Remote characterization evidence covers all six platform/Python
cells and every relevant dispatch class; outputs are compared as cubes, not
only as digests. Any pin change requires a green exact-pin-SHA rerun.

An independent reviewer must inspect the exact `R`, `S`, and `E` commits,
complete every phase-specific oracle and common review check in Section 8.2
without using the implementation helper, authenticate retained artifacts, and
return `ACCEPT` or `REJECT` with concrete blockers. An `ACCEPT` is retained in
the exact phase-local artifact and authenticated by `A`; reviewer prose or an
appended memo sentence is not a substitute. Acceptance cannot be performed by
the implementation pass.

`SCI-005` closes only after all three stage acceptances and a separate whole-row
review establish:

- all scoped central/support blockage and deterministic Zernike behavior is
  implemented under the one normalized aperture transform;
- Ruze coherent loss and ensemble scattered-power claims remain separated and
  no invented error-beam voltage exists;
- squint uses the exact frequency law, mechanical-feed semantics, and
  `E=C^dagger D_b C` factorization;
- full efield ingestion has explicit normalization, coordinate, Ludwig-3,
  receptor, reporting-basis, and output-format semantics;
- point/HEALPix, backend, output, and optional pyuvsim evidence is retained;
- no contraction signature or compilation-boundary change occurred; and
- `Fix.md`, this plan, the scope document, changelog, migration guide, API and
  user docs agree with the accepted implementation.

Until then the register remains **ROADMAP**, even if one or two stages are
accepted.

## 10. Primary sources and official contracts

- Aperture blockage and reflector analysis: [NASA TM X-63186](https://ntrs.nasa.gov/citations/19680013447)
  and [ITU-R SA.2401-0](https://www.itu.int/pub/R-REP-SA.2401-2017).
- Real unit-RMS disk Zernikes: R. J. Noll,
  [*Zernike polynomials and atmospheric turbulence*](https://opg.optica.org/josa/abstract.cfm?uri=josa-66-3-207),
  JOSA 66, 207 (1976), DOI 10.1364/JOSA.66.000207. Annular-basis distinction:
  V. N. Mahajan,
  [*Zernike annular polynomials for imaging systems with annular pupils*](https://doi.org/10.1364/JOSA.71.000075),
  JOSA 71, 75 (1981).
- Random reflector errors: J. Ruze,
  [*The effect of aperture errors on the antenna radiation pattern*](https://doi.org/10.1007/BF02903409)
  (1952), and
  [*Antenna tolerance theory — a review*](https://doi.org/10.1109/PROC.1966.4784)
  (1966).
- Separation-domain quadrature bounds: Bessel-coefficient decay
  $|J_n(x)|\leq(x/2)^n/n!$,
  [DLMF 10.14.4](https://dlmf.nist.gov/10.14.E4) and Abramowitz and Stegun
  9.1.62; exponential convergence of the equispaced trapezoid rule for analytic
  periodic integrands, L. N. Trefethen and J. A. C. Weideman,
  [*The exponentially convergent trapezoidal rule*](https://doi.org/10.1137/130932132),
  SIAM Review 56, 385 (2014).
- Beam squint: J. M. Uson and W. D. Cotton,
  [*Beam squint and Stokes V with off-axis feeds*](https://arxiv.org/abs/0807.0026)
  (2008).
- Cross-polar coordinates: A. Ludwig,
  [*The definition of cross polarization*](https://ntrs.nasa.gov/citations/19730033397)
  (1973), DOI 10.1109/TAP.1973.1140406.
- Radio-interferometric Jones formalism: Hamaker, Bregman, and Sault,
  [*Understanding radio polarimetry. I. Mathematical foundations*](https://ui.adsabs.harvard.edu/abs/1996A%26AS..117..137H/abstract)
  (1996). IXR: Carozzi and Woan,
  [*A fundamental figure of merit for radio polarimeters*](https://eprints.gla.ac.uk/62472/)
  (2011), DOI 10.1109/TAP.2011.2123862.
- File and independent-simulator contracts: official
  [UVBeam documentation](https://pyuvdata.readthedocs.io/en/stable/uvbeam.html),
  [pyuvdata analytic efield documentation](https://pyuvdata.readthedocs.io/en/stable/analytic_beam_tutorial.html),
  and [pyuvsim primary-beam/Jones documentation](https://pyuvsim.readthedocs.io/en/stable/classes.html).

These sources constrain the named equations and formats. They do not by
themselves establish RadioSim implementation correctness; that requires the
stage evidence and independent acceptance above.
