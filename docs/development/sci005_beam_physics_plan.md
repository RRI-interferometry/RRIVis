# SCI-005 staged beam-physics design gate

**WP-8 design-gate candidate — 2026-08-11**

**Source reviewed:** `e63770c3e27e5aee4e09570c53eb1367099b1ae4`, the
accepted WP-7 design commit. Ambient WP-7 implementation work is not evidence
for this memo and does not change its source anchor.

**Status:** design only. This candidate requires an independent design review
before it can authorize a production slice. It implements no beam physics,
accepts no stage, and does not close the register row. `SCI-005` remains
**ROADMAP**. Stage 1 cannot begin until the CPU portion of WP-7 is independently
accepted. Stages 2 and 3 additionally remain sequential, independently accepted
slices even though WP-5 has satisfied their polarization-convention dependency.

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
Positive surface height is defined in the signed local-normal direction that
increases the one-way geometrical path. Reflection gives excess path `2*h`, so
RadioSim's positive-delay convention produces
`exp(-i * 4*pi*h/lambda)`. The convention literal is
`radiosim.real_unit_rms_disk_surface_height.v1`.

Stage 1 supports the reflector-like circular analytic families
`circular_aperture`, `analytical_illumination`, and
`numerical_illumination`. Applying a reflector blockage or disk Zernike map to
`rectangular_aperture`, `elliptical_aperture`, or a FITS beam is rejected. A
FITS file already contains its aperture physics; applying it again would double
count an effect that cannot be separated from the file.

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

`beams.aperture_physics.zernike_surface` contains the exact convention literal
above and a non-empty ordered tuple of mode records. Each record contains a
strict integer pair `(n,m)` and a strict finite `surface_height_coefficient_m`.
Validation requires

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

Thus each coefficient is a signed reflector-surface height in metres in a
unit-RMS **unobscured disk** basis. After a blockage mask is applied these
ordinary disk functions cease to be orthogonal over the transmitting annulus;
the configuration must not describe the quadrature sum of coefficients as the
RMS over that annulus. No Noll or OSA single integer is accepted, and no OPD
coefficient is accepted under a surface-height field.

The upper radial order `32` is a v1 computation bound, not a statement that
higher physical modes do not exist. The same aperture transform evaluates mask
and phase together. Quadrature uses tensor-product Gauss-Legendre radial nodes
and uniform midpoint azimuth nodes. It starts with
`n_rho=max(64,4*(n_max+1))` and `n_phi=max(128,8*(2*n_max+1))`, doubles both
counts together, and accepts the refined result when the maximum change is no
larger than `atol + rtol*max(abs(refined))`. The fixed dtype-derived values are
`atol=max(1e-12,32*eps)` and `rtol=max(1e-10,32*eps)`, matching the existing
beam tolerance rule. Four failed refinements raise
`BeamSamplingDerivationError`; no unconverged result is returned. Counts and
tolerances are internal and cannot be authored in YAML. Red tests must show
disk orthonormality, the exact defocus/conjugation relation under sign reversal,
the uniform blocked Airy invariant, and convergence or typed failure under this
production rule.

### 3.4 Ruze coherent loss and scattered-power diagnostic

The existing `beams.surface_error` field keeps its accepted coherent-voltage
meaning. With reflector-surface RMS $\sigma_h$ and
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

The literal `gaussian_covariance_power` fully specifies
$\rho_h(\Delta)=\exp[-(|\Delta|/L)^2]$; `L` is its one-over-e correlation
length. Here `rms_surface_error_m` is the zero-mean random residual after the
configured deterministic Zernike map, not a value inferred from those
coefficients, and
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
non-negative scattered difference on the exact two-dimensional direction grid
passed to the public method. A one-dimensional radial result is permitted only
when the resolved mask, deterministic phase, and illumination are all
rotationally symmetric; support legs or any `m != 0` Zernike mode force the 2-D
result. It may support an autocorrelation analysis, but it does not enter a
cross-correlation Jones matrix. For independent antenna surfaces,
`<e_p e_q*> = <e_p><e_q*>` when `p != q`; the scattered error power belongs to
the same-surface second moment. Taking `sqrt(B_main+B_error)` would invent a
phase and perfectly correlated structure, so that operation is forbidden.

The implementation evaluates the double integral directly or through its
mathematically equivalent aperture autocorrelation using the same pupil nodes,
dtype-derived tolerances, and four-refinement failure rule as Section 3.3.
Coherent-main, total, and scattered power must all converge on the complete
queried 2-D grid; otherwise `BeamSamplingDerivationError` is raised. The
diagnostic records the method literal, final node counts, residuals, and fixed
tolerances.

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

The typed public diagnostic is
`BeamSystem.evaluate_ruze_power_diagnostic(antenna_id, *, altitude_rad,
azimuth_rad, frequency_hz, time_mjd)`. It returns an immutable
`RuzePowerDiagnostic` carrying the queried directions, coherent-main power,
total ensemble power, scattered power, covariance convention, and convergence
metadata. It is available only when the resolved antenna carries the diagnostic
block and otherwise raises `BeamEvaluationError` with a stable exact message.
It never mutates or substitutes the matrix returned by `evaluate_jones`.

### 3.5 Stage-1 rejections and acceptance invariants

Typed errors must preserve these exact semantic families:

- an explicitly present aperture block enables neither blockage nor a non-zero
  allowed Zernike mode;
- aperture physics is attached to a non-circular analytic family or a FITS
  source;
- a blockage ratio, support width, or resolved support geometry is outside its
  physical domain;
- the unmodified ideal aperture integral `N0` is zero or non-finite;
- a Zernike index is invalid, repeated, piston, or tip/tilt;
- all Zernike coefficients are zero;
- a diagnostic lacks a positive correlation length, or is authored without a
  positive surface RMS; and
- an unknown normalization, covariance, or Zernike convention is supplied.

Acceptance requires all of:

- the closed blocked-uniform-aperture formula, including boresight loss;
- numerical unit-RMS/orthogonality checks for the declared Zernike basis;
- blockage and Zernike each change a visibility when enabled;
- the composed mask-plus-phase result differs from a deliberately wrong
  product of two far-field factors;
- coherent Ruze cross-baseline scaling and ensemble power balance;
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
- `tests/unit/test_core/test_sci005_beam_squint.py` for Stage 2;
- `tests/unit/test_core/test_sci005_full_efield.py` for Stage 3;
- `tests/integration/test_sci005_beam_physics.py` for point/HEALPix, outputs,
  and effect-through-`Simulator` cases;
- `tests/crossvalidation/test_sci005_efield_pyuvsim.py` for the optional Stage-3
  comparison; and
- `tests/unit/test_sci005_evidence.py` for strict retained-evidence and digest
  authentication.

Existing chain, beam-runtime, result-output, backend-parity, release-scan, and
characterization modules are extended only where the stage changes a property
they already own. No `xfail`, tolerance widening, warning suppression, or
benchmark exception is acceptance evidence.

## 7. Exact writable lists

These lists are implementation authority only after independent design review
and the stated dependency. A path not listed here requires a bounded design
correction before it is edited.

### 7.1 Design slice (this candidate)

- `docs/development/sci005_beam_physics_plan.md` (new)
- `docs/index.rst`
- `PostTier8RemediationPlan.md` (WP-8 section and ledger dependency wording)

### 7.2 Stage 1 red tests, implementation, and evidence

- `src/radiosim/io/beam_config.py`
- `src/radiosim/io/config.py`
- `src/radiosim/io/config_resolution.py`
- `src/radiosim/core/beam/models.py`
- `src/radiosim/core/beam/resolution.py`
- `src/radiosim/core/beam/analytic.py`
- `src/radiosim/core/beam/runtime.py`
- `src/radiosim/core/beam/__init__.py`
- `tests/unit/test_io/test_sci005_beam_config.py` (new)
- `tests/unit/test_core/test_sci005_aperture_physics.py` (new)
- `tests/unit/test_core/test_beam_models.py`
- `tests/unit/test_core/test_beam_resolution.py`
- `tests/unit/test_core/test_beam_runtime.py`
- `tests/unit/test_core/test_beam_solver_integration.py`
- `tests/integration/test_sci005_beam_physics.py` (new)
- `tests/characterization/test_tier6_current_behavior.py`
- `tests/characterization/test_tier7_current_behavior.py`
- `tests/characterization/test_tier8_current_behavior.py`
- `tests/unit/test_sci005_evidence.py` (new)
- `tools/sci005_stage_evidence.py` (new)
- `docs/user_guide/configuration.rst`
- `docs/user_guide/configuration_support.rst`
- `docs/user_guide/beam_models.rst`
- `docs/api/core.rst`
- `docs/migration_guide.md` only if the accepted config/result contract breaks
  a pre-v1 surface
- `docs/development/sci005_stage1_evidence.json` (new evidence successor only)

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
- `docs/user_guide/configuration.rst`
- `docs/user_guide/configuration_support.rst`
- `docs/user_guide/beam_models.rst`
- `docs/user_guide/jones_matrices.rst`
- `docs/migration_guide.md` for the non-scalar-`E` pre-v1 widening
- `docs/development/sci005_stage2_evidence.json` (new evidence successor only)

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
- `docs/user_guide/configuration.rst`
- `docs/user_guide/configuration_support.rst`
- `docs/user_guide/beam_models.rst`
- `docs/user_guide/jones_matrices.rst`
- `docs/api/core.rst`
- `docs/migration_guide.md`
- `output/crossvalidation/README.md`
- `output/crossvalidation/<date>-sci005-efield-pyuvsim-1.4.0.json` (new)
- `docs/development/sci005_stage3_evidence.json` (new evidence successor only)

### 7.5 Acceptance and closure successors

After each stage's independent acceptance, and only then:

- `Fix.md` (dated acceptance text; row stays ROADMAP until whole-row closure)
- `PostTier8RemediationPlan.md` (stage ledger only)
- `docs/development/sci005_beam_physics_plan.md` (append-only acceptance note)
- `docs/development/beam_physics_scope.md` (only the accepted stage's rows)
- `docs/changelog.rst`
- `docs/migration_guide.md` if acceptance found its current wording incomplete
- `README.md` and `CLAUDE.md` only where an accepted stage makes a live support
  statement false

The final all-stage closure successor reconciles the complete scope and support
wording but must not rewrite historical acceptance text. The files above are
not writable by an implementation stage merely because it intends eventual
closure.

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

Each `docs/development/sci005_stageN_evidence.json` is strict and has exactly:

- `schema_version`, `stage`, `status`, `generated_at_utc`;
- `design_sha`, `red_test_sha`, `source_sha`, `evidence_sha`, and
  `working_tree_clean`;
- `radiosim_version`, `python_version`, `platform`, `machine`,
  `pixi_environment`, and `pixi_lock_sha256`;
- `scientific_conventions`, `config_cases`, `analytic_invariants`,
  `rejection_probes`, `backend_parity`, and `solver_cases`;
- `output_cases`, `fingerprint_diff`, `commands`, `artifacts`, `limitations`,
  and `claims_not_licensed`.

`schema_version` is respectively `radiosim.sci005.stage1.v1`,
`radiosim.sci005.stage2.v1`, or `radiosim.sci005.stage3.v1`. Missing and unknown
fields fail validation. Every digest is lower-case SHA-256; every count is a
non-negative JSON integer; every residual and tolerance is finite and
non-negative; absent measurements are explicit JSON null with a named reason,
never omitted or encoded as zero.

Every analytic-invariant row names the equation/convention, test node ID,
inputs, expected value, observed value, absolute residual, and fixed tolerance.
Every rejection row names the config path, exception type, exact message, and
test node ID. Backend rows name the backend, actual device, dtype, source and
result digests, maximum absolute/relative difference, and tolerance. Output
rows cover in-memory, summary, HDF5, UVFITS, MS, and reader projections as
applicable. Fingerprint rows name environment, workload, old/new scientific
hashes, old/new raw-cube hashes, changed element count, maximum delta, and
whether change was expected.

Stage 3 additionally embeds the cross-validation artifact path and digest,
reference package versions, input-content hashes, explicit convention mappings,
per-correlation residuals, and unresolved differences. The evidence document
does not copy or summarize a value without authenticating the underlying
artifact digest.

Commit succession is mandatory:

1. `D`: independently accepted design-only commit;
2. `R1`, `R2`, or `R3`: red tests plus recorded pre-fix failure, no production
   implementation;
3. `S1`, `S2`, or `S3`: one coherent stage implementation, including docs and
   enabled-effect pin changes;
4. `E1`, `E2`, or `E3`: evidence-only successor generated from clean `S`, with
   the artifact recording `source_sha=S`; and
5. `A1`, `A2`, or `A3`: independent acceptance/status successor, with no
   production source change.

The evidence file cannot truthfully contain its own future Git SHA before it is
committed. `evidence_sha` is JSON null in the file; the acceptance record binds
the exact `E` commit and raw artifact SHA-256. The final closure commit `C`
parents accepted `A3`, updates the register and scope wording only, and receives
its own exact-SHA local and remote gate. An evidence or acceptance successor
that changes production source is invalid and must be split.

## 9. Verification and independent acceptance gates

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
make -C docs html
git diff --check
```

Stage 3 additionally runs the optional comparison with
`pixi run --environment crossval -- python -m pytest tests/crossvalidation/ -m crossval`.
It remains non-gating, but a retained Stage-3 artifact is required for
acceptance. Remote characterization evidence covers all six platform/Python
cells and every relevant dispatch class; outputs are compared as cubes, not
only as digests. Any pin change requires a green exact-pin-SHA rerun.

An independent reviewer must inspect the exact `S` and `E` commits, re-derive
at least one scientific equation without using the implementation helper,
reproduce at least one typed rejection by hand, inspect production data flow,
authenticate retained artifacts, verify disabled/default fingerprints, and
return `ACCEPT` or `REJECT` with blockers. Acceptance cannot be performed by
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
