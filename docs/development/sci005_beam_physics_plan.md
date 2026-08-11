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

**Status:** Stage-1 numerical design accepted; succession amendment design-only.
The numerical contract is accepted at the exact commit above. This governance
amendment can become `D1` only after its exact bytes receive the two fresh
reviews required above. It implements no beam physics, accepts no stage, and
does not close the register row. `SCI-005` remains **ROADMAP**. Stage 1 may
begin `R1` only after `D1` and the WP-7 dependency gate below, then must
complete its own source, evidence, acceptance, and status succession. Stages 2
and 3 remain sequential, independently accepted slices even though WP-5 has
satisfied their polarization-convention dependency.

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
rejected. No scattering or phase from a support structure is claimed; Stage 1
models only the geometrical aperture shadow.

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
`poisson_gauss_hermite_aperture_v1`, defined here.

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

The internal aperture-transform helper accepts every finite real two-vector;
Hermite-shifted wavevectors need not correspond to a physical sky direction.
It uses the same boundary-fitted panels, phase-bandwidth seeds, convergence,
and target-width accumulation as Section 3.3, but never calls
`evaluate_jones` and never applies a sky angular-domain or horizon check. The
outer requested directions alone receive the existing pointing transform and
true-horizon gate.

Every supported $U/P/P_2$ mixture has $A\geq0$ and $N_0=\int A>0$; the mask is
zero/one and the deterministic phase has unit modulus. Consequently

$$
|e_{\rm det}(\mathbf k)|\leq
\frac{\int AM}{\int A}\leq1
$$

for every shifted wavevector. This both proves non-negative scattered power and
makes omitted Poisson probability mass a rigorous absolute power-error bound.

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

Each retained Gaussian uses tensor-product physicists' Gauss-Hermite nodes.
For raw positive weights $w_i$, the exact computational weights are
$\bar w_i=w_i/\operatorname{fsum}(w)$, so
$\sum_{ij}\bar w_i\bar w_j=1$. Nodes and weights are generated and validated
under the target-width rule in Section 3.3. Allowed one-axis orders are exactly
`(8,16,32,64,128,256,512)`. To prevent two unresolved narrow results from
falsely agreeing, no convergence comparison may count below the first allowed
order not less than

$$
H_{\rm floor}=8+\left\lceil4\sqrt{m_{\rm last}}D/L\right\rceil.
$$

There must be room for two higher allowed orders or evaluation fails before it
starts. Starting at that allowed floor, two consecutive complete-array
comparisons must satisfy one quarter of
`atol + rtol*max(abs(refined))`; otherwise the diagnostic raises
`BeamSamplingDerivationError`. The aperture helper independently requires at
least two levels and at most four refinements, and its complex result must
converge over every base and Hermite-shifted wavevector, not merely after the
weighted powers are summed.

The diagnostic supports at most 65,536 aperture nodes in any transform,
`2**20` cumulative transformed wavevectors, `2**28` cumulative
aperture-node/wavevector phase products, and `8*2**30` estimated workspace
bytes. It uses an internal batch size equal to the largest power of two no
greater than 256 that satisfies all remaining phase-product and byte caps; a
smaller non-power-of-two final batch is permitted. Counts and the conservative
shape-by-shape byte estimate are checked before every Poisson, Hermite, aperture,
or batch refinement. For $r=\max(8,\text{beam real bytes})$,
$c=\max(16,\text{beam complex bytes})$, and the current $J,H,Q,B,S$, that
estimate is exactly

$$
E_{\rm Ruze}=r(16Q+6H^2+8BQ+16B+12S+4J)
+c(4BQ+8B+6S).
$$

Buffers must be reused within those multiplicities. Exceeding a cap returns no
partial result and raises
`BeamSamplingDerivationError`. The worst-case work is
$O(SJH^2Q)$ and memory is $O(Q+H^2+BQ+S)$ for retained-term count $J$, final
Hermite order $H$, maximum aperture-node count $Q$, and batch size $B$; no
$Q^2$ pair array exists in production.

At least float64 weights and accumulation are used for both supported output
widths. Coherent and scattered arrays are cast separately to the beam real
dtype, checked finite and non-negative, and only then is
`total_ensemble_power = coherent_main_power + scattered_power` formed in that
same dtype. There is no clipping: a negative or non-finite weighted sum is an
internal numerical failure and raises `BeamSamplingDerivationError`. The
returned balance is exact in the result dtype; maximum observed
$|e_{\rm det}|$ must not exceed one beyond the unchanged tolerance, and total
power must not exceed one beyond it. Poisson tail, two successive Hermite
residuals, and the aperture-transform residuals are retained separately; none
is hidden in a single convergence boolean.

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
| `method` | `Literal["poisson_gauss_hermite_aperture_v1"]` |
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
hermite_order, hermite_evaluation_count,
hermite_penultimate_max_abs_delta, hermite_final_max_abs_delta,
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
target-dtype radial-breakpoint and periodic-angular-partition manifest. The
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

All orders, counts, batch size, and byte fields are exact non-negative Python
integers. `hermite_order` is zero only for the resolved zero-term Poisson case;
otherwise it is an allowed positive order. Every other field from
`poisson_mu` through `returned_balance_max_abs_residual` not already classified
as a string, digest, or integer is an exact finite non-negative Python float.
Lower plus upper omitted mass equals total omitted mass in float64 arithmetic;
term count is zero exactly with Poisson interval `[0,0]` and otherwise equals
`last - first + 1`. In the zero-term case retained weight, Hermite evaluation
count, Hermite order, and both Hermite residuals are exactly zero; lower omitted
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
$\max R\|\mathbf k\|_2$ over every base or Hermite-shifted wavevector actually
presented to any aperture solve in the whole call; it is not physical
$\|\mathbf k\|$ in inverse metres. `aperture_max_node_count` is the maximum
Section 3.3 value of $Q$ over all seed/refinement evaluations; `batch_size` is
the largest wavevector batch actually scheduled; and `estimated_peak_bytes` is
the maximum declared estimate. `surface_phase_kappa` is exactly
$4\pi/\lambda$; the two surface derivative fields are the Section 3.3
$H_\rho,H_\varphi$ bounds. The named amplitude and power extrema have their
literal maximum/minimum meanings over all evaluated or returned values as
applicable.

For Hermite convergence, `penultimate` and `final` are the first and second of
the final two consecutive successful complete-array comparisons. For aperture
convergence, every angular sequence that licenses a radial result and the
radial sequence itself has two final successful comparisons;
`aperture_penultimate_max_abs_delta` is the maximum of the first such licensing
deltas and `aperture_final_max_abs_delta` the maximum of the second, across both
dimensions, every relevant direction, shifted wavevector, and retained Poisson
term. These are not merely the final radial pair. There is no `converged` field
and no false state: failure returns no record.

`hermite_evaluation_count` counts every scheduled tuple
`(outer_direction, poisson_order, hermite_i, hermite_j)` across all attempted
Hermite orders, including repeated abscissae after refinement.
`fhat_evaluation_count` counts every wavevector element presented to an
aperture-transform refinement, so reevaluating one wavevector at a new aperture
order increments it again. `phase_product_count` counts every scalar
aperture-node/wavevector exponential formed. `aperture_refinement_count` counts
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
  positive surface RMS, uses unsupported extended precision, or receives an
  empty direction batch;
- an unknown normalization, covariance, or Zernike convention is supplied; or
- a boundary/topology value is not representable, or the fixed quadrature,
  Poisson-tail, Hermite-order, transform-count, phase-product, memory,
  convergence, finite-value, amplitude-bound, power-bound, or exact-balance
  predicate cannot be satisfied.

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
  the Poisson/Gauss-Hermite result, catches the factor-two, $1/\pi$,
  deterministic-phase, negative-forward aperture-transform sign, and
  normalization controls, and is never imported by production; the Hermite
  shift sign itself is not an oracle because the symmetric whole-plane integral
  is invariant under $\mathbf t\mapsto-\mathbf t$;
- stable low/high-$\mu$ two-sided Poisson tails, the $\mu\to0$ first-order
  limit, and the $L\to\infty$ identity
  $B_{\rm sc}=(1-e^{-\mu})|e_{\rm det}|^2$;
- a deliberately narrow Hermite integrand that converges only after refinement
  or fails with the typed cap, plus entire-plane shifted-wavevector evaluation
  without a sky-domain rejection;
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
- point and HEALPix paths and NumPy/JAX/Dask parity; and
- fingerprints changed only for squint-enabled fixtures.

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
  small-node pair oracle, Poisson/Hermite/refinement/resource predicates, and
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
directly parents `U1` and freezes the complete normative Stage-2 evidence
envelope before `R2`; `D3` directly parents `U2` and does the same for Stage 3
before `R3`.
Those later design gates may write only the paths above, change no production,
red oracle, retained artifact, or prior acceptance text, and require their own
exact-byte independent design reviews. Omitting an applicable `D2` or `D3`, or
combining it with red tests, invalidates the following stage.

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
literal or any dependency-validator byte. Stage-1 evidence and acceptance
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
- `src/radiosim/core/visibility.py`
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
envelope below. Stages 2 and 3 retain their accepted common envelope, but their
stage-specific extensions are deliberately not frozen here; mandatory `D2` and
`D3` independently freeze their exact normative rows, keys, types, and
cross-field predicates before their red slices. The corresponding `S2` or `S3`
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
ruze_method: poisson_gauss_hermite_aperture_v1
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
enums; `aperture_topology_sha256` is a sha256. Integer keys are exactly the
following fields:

```text
poisson_first_order, poisson_last_order, poisson_term_count,
hermite_order, hermite_evaluation_count,
aperture_partition_count, aperture_topology_breakpoint_count,
aperture_refinement_count, aperture_max_node_count,
fhat_evaluation_count, phase_product_count, batch_size, estimated_peak_bytes
```

All remaining convergence keys not already classified as dtype/method strings
or a digest are JSON numbers. Cross-field validation enforces the zero-term
Poisson case, retained interval count, tail sum, allowed Hermite order, caps,
two residuals, dtype pair, non-negative powers, amplitude/total bounds, and
exact result-dtype balance from Section 3.4. Unknown or missing projection keys,
nulls, a backend field, a `converged` boolean, clipping metadata, FFT metadata,
or any direction-sized complex value fail authentication.

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
`ruze_power_diagnostics`; `D2` and `D3` freeze their stage-specific input
extensions with their evidence rows. Every supplied value uses the
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
chain is `D1 ->* G1 -> R1 -> S1 -> E1 -> A1 -> U1 -> D2 -> R2 -> S2 -> E2
-> A2 -> U2 -> D3 -> R3 -> S3 -> E3 -> A3 -> U3 -> C`. The starred edge is
ancestor reachability through separately authorized, independently accepted
programme commits, including the WP-7 dependency/interface succession; every
unstarred arrow is the sole direct-parent edge and no named commit is a merge.
Across `D1..G1`, the exact `D1` memo blob, the `Fix.md` SCI-005 row, the WP-8
subsection/ledger cells, and the Stage-1 scope rows remain unchanged. Every
Section 7.2 path marked `new` or `successor only` remains absent; independently
accepted programme commits may otherwise change shared paths and those bytes
become the `G1` red baseline. Thus `R1^ == G1`, `Si^ == Ri`, `Ei^ == Si`,
`Ai^ == Ei`, `Ui^ == Ai`,
`D2^ == U1`, `R2^ == D2`, `D3^ == U2`, `R3^ == D3`, and `C^ == U3`. The
validator also requires `D1` and the WP-7 certificate's `acceptance_commit` to
be ancestors of `G1` and reauthenticates the exact certificate bytes retained
by `R1`. It requires the `R1` design-binding literal to equal every Stage-1
`design_sha` and remain byte-identical through `S1/E1/A1/U1`; `D2` and `D3`
remain unambiguous direct parents of their red commits.

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
