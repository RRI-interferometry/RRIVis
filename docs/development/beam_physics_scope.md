---
orphan: true
---

# Beam physics: implemented subset and explicit scope

This document replaces `src/radiosim/core/jones/beam/TODO.md`, the in-source
wish list recorded as defect **D20** of `Tier7JonesSciencePlan.md` and closed by
issue **SCI-003**. It is a *disposition table*, not a wish list: every item that
was on that list appears here with its physics, its citation, whether RadioSim
implements it, and — where it does not — the register row that owns it.

A scope document with a register owner belongs in tracked documentation rather
than as a `TODO.md` inside an installed package (`Tier7JonesSciencePlan.md`
§19.4). Nothing here is a promise; the "Out of scope" rows are statements about
what RadioSim does **not** do.

## The governing constraint

RadioSim's accepted E-Jones is **scalar**: the beam response is one complex
voltage `e` on the diagonal of a 2×2 matrix, `E = e · I2`. Tier 3 enforces it
for FITS beams (a file with cross-polar terms, or with differing X/Y diagonals,
is rejected) and the analytic evaluators construct it directly. Most of the
items below are out of scope for exactly one reason: they require widening `E`
beyond that, which is a tier-scale change and not a beam-model parameter.

## Implemented

### Per-antenna deterministic pointing offsets

`beams.pointing`. A per-antenna `(azimuth_offset_deg, elevation_offset_deg)` is
a fixed rotation of that antenna's beam frame relative to the topocentric
horizontal frame, composed as the two encoder errors of an alt-az mount: a
rotation about the local vertical that increases azimuth (North through East),
then a tilt away from the zenith. Because RadioSim's boresight is the zenith,
the composed boresight lands at topocentric azimuth `azimuth_offset_deg` and
zenith angle `elevation_offset_deg`.

Two consequences are exact and are asserted in
`tests/unit/test_core/test_beam_pointing.py`:

- the beam's peak moves by a great-circle angle of exactly
  `elevation_offset_deg`, in the direction of `azimuth_offset_deg`;
- a pure azimuth offset rotates the pattern about the boresight without moving
  it. This is the alt-az keyhole degeneracy — real physics at a zenith-pointed
  mount — so a pure azimuth offset is inert for a circularly symmetric aperture
  and is not inert for the rectangular and elliptical ones.

The horizon gate stays on the **true** topocentric altitude: a rotation of the
beam frame does not move the ground.

The statistical gain-reduction formula that the old list quoted,

```
<G/G0> = [1 + 4 ln2 (sigma_p / theta_HPBW)^2]^-1
```

is the *expectation* of this deterministic model over a Gaussian pointing
distribution of per-axis RMS `sigma_p`. It is documented here as that
expectation and is **not** implemented as a separate stochastic path: a
simulator that drew random pointings per integration would produce a
non-reproducible cube, which the provenance discipline forbids. The rule of
thumb it implies — `sigma_p < 0.10 theta_HPBW` for 5% accuracy — is a
statement about a real telescope's specification, not about RadioSim.

### Ruze random-surface efficiency

`beams.surface_error`. Ruze's equation gives the **power** efficiency of a
reflector with random surface errors of RMS `sigma`:

```
eta_s(lambda) = exp(-(4 pi sigma / lambda)^2)
```

RadioSim's `E` is a *voltage* beam and the RIME contracts `E_p B E_q^H`, so the
factor applied to `E` is `sqrt(eta_s) = exp(-(1/2)(4 pi sigma / lambda)^2)`.
The visibility amplitude on a baseline of two antennas sharing `sigma` is then
scaled by exactly `eta_s`, which is what the published equation states. This is
the same voltage/power discipline the tropospheric opacity uses for
`exp(-tau/2)`. Both closed forms are public:
`radiosim.core.beam.runtime.ruze_power_efficiency` and
`ruze_voltage_factor`.

The associated rule of thumb, `lambda_min ≈ 10 sigma`, is the wavelength at
which `eta_s` has already fallen to `exp(-(0.4 pi)^2) ≈ 0.206`. It bounds
nothing in the mathematics — the equation is finite and monotone everywhere —
so RadioSim records it here rather than enforcing it as a rejection.

**Reference.** Ruze, J. (1966), *Antenna tolerance theory — a review*,
Proc. IEEE **54**, 633.

## Out of scope, with owners

Every row below is owned by register row **`SCI-005` | ROADMAP | Advanced beam
physics beyond the accepted scalar-`E` subset**.

### Cross-polarization models

Three related models, all requiring a non-scalar `E`.

- **Quadrupolar.** The dominant cross-polar pattern for linearly polarized
  feeds: `epsilon(theta) = epsilon_0 (theta / theta_HPBW)^2` with azimuthal
  dependence `cross_pol = epsilon(theta) sin(2 phi)`, vanishing on the principal
  planes (`phi = 0, pi/2`) and peaking at `phi = pi/4`. The Y feed's cross-polar
  response has the opposite parity to the X feed's, so the assembled Jones
  matrix is `[[co, cross], [-cross, co]]` rather than `diag(co, co)`.
- **IXR conversion.** The intrinsic cross-polarization ratio is a
  basis-independent polarimetric fidelity measure. For a leakage matrix
  `D = [[1, d], [-d*, 1]]` the singular values are `1 ± |d|`, so the condition
  number is `kappa = (1 + |d|)/(1 - |d|)` and Carozzi & Woan's
  `IXR_J = ((kappa + 1)/(kappa - 1))^2` gives

  ```
  |d| = 1 / sqrt(IXR_lin),      IXR_lin = 10^(IXR_dB / 10)
  ```

  equivalently `IXR_dB = -20 log10 |d|`.

  **This corrects the formula the superseded `TODO.md` carried.** That file
  stated `|eps| = (sqrt(IXR_lin) - 1)/(sqrt(IXR_lin) + 1)`, which is inverted:
  it maps a *larger* IXR to a *larger* leakage, so a 30 dB antenna — an
  excellent one — would resolve to `|d| = 0.94` and a 0 dB antenna — a
  completely depolarizing one — to `|d| = 0`. The corrected relation has the two
  limits the physics requires: `|d| → 0` as `IXR_dB → ∞`, and `|d| = 1` at
  `IXR_dB = 0`. It is the form the implemented `D` term uses
  (`Tier7JonesSciencePlan.md` §20.3).
- **Ludwig-3 decomposition.** The standard co/cross definition for linearly
  polarized antennas: `E_co = E_theta cos(phi) - E_phi sin(phi)`,
  `E_cross = E_theta sin(phi) + E_phi cos(phi)`, which preserves total power.

**References.** Carozzi & Woan (2011) IEEE Trans. Antennas Propag. **59**, 2058;
Ludwig (1973) IEEE Trans. Antennas Propag. **AP-21**, 116; Hamaker, Bregman &
Sault (1996) A&AS **117**, 137.

### Near-field and Fresnel regime

The Fresnel diffraction integral for `R < 2 D^2 / lambda`, with Fresnel number
`N_F = D^2 / (4 lambda R)` and on-axis oscillation `I(R) = 4 sin^2(pi N_F / 2)`.

This is a **permanent non-goal for simulation**, not a deferred item.
Astronomical sources are always in the far field, so the regime is irrelevant to
every visibility RadioSim computes. It is relevant only to holography, antenna
test ranges, and drone calibration — measurement activities, not observations.

### Aperture blockage

The subreflector and feed shadow removes the central region from the aperture
integral: `E_blocked = E_aperture - (d/D)^2 E_subreflector_pattern`, with
blockage efficiency `eta_b ≈ (1 - (d_block/D)^2)^2`; support legs add four-fold
diffraction spikes.

Out of scope: it is a change to the aperture integral of every analytic beam
model, not a factor applied around one, and the support-leg term is not
axisymmetric. It would need `blockage_diameter`, `n_support_legs`, and
`support_leg_width` fields.

### Ruze error-beam decomposition

The full Ruze decomposition is `B_total = exp(-sigma_phi^2) B_main +
error_beam`, where the error beam is approximately Gaussian with
`FWHM ≈ 0.53 lambda / L` for correlation length (panel size) `L`.

The efficiency factor is implemented (above); the **error beam is not**. It is a
stochastic scattered-power model: the power the main beam loses reappears in a
broad, randomly structured pedestal whose realization depends on the actual
panel errors. Modelling it deterministically would fabricate structure that the
equation does not determine.

### Systematic aberrations

Defocus (quadratic phase error from axial feed displacement), coma (asymmetric
beam from lateral feed displacement), astigmatism (elliptical beam from dish
warping), and gravitational sag (elevation-dependent deformation).

Out of scope: all four are phase errors across the aperture and want a Zernike
polynomial basis, which is an aperture-integral change rather than a beam-model
parameter.

### Beam squint

The RCP/LCP beam offset `theta_squint ≈ d_offset / (2 f)` produces spurious
Stokes V, `V_meas = V_true + (1/2) I (dA/dn) theta_squint`, and rotates with
parallactic angle on an alt-az mount.

**Implemented — `SCI-005` Stage 2, independently accepted 2026-08-19.**
`beams.squint` applies the exact Cotton/Uson arcsine frequency law with the
squint direction at `+pi/2` from the mechanical feed ray in the beam frame,
mount field rotation evaluated at the resolved boresight, and the two native
feeds sampling oppositely displaced scalar beams; the beam runtime composes
the generally full `E = C^dagger D_b C` from those samples and the antenna's
resolved receptor matrix. For a circular receptor this `E` commutes exactly
with every real rotation, so the order control lives on a rotated linear
receptor. The governing design is
`docs/development/sci005_beam_physics_plan.md` Section 4; the retained
evidence and acceptance records are
`docs/development/sci005_stage2_evidence.json` and
`docs/development/sci005_stage2_acceptance.json`.

**Reference.** Cotton & Uson (2008), arXiv:0807.0026.
