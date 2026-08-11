# SCI-007 polarization-frame accuracy design gate

**WP-6 design-gate record — 2026-08-11**

**Source reviewed:** `8f0adce04d8114b2484ed75ae90c335b8fc1d8fc`, clean before
this documentation slice.

**Status:** independently accepted 2026-08-11 at exact evidence successor
`e20f636788e0b61ae6c854f64cbb7476c3cb9a50`; `SCI-007` is **DONE** as a
retained-fixture accuracy bound. Production code, tolerances, configuration,
the lockfile, fingerprints, and the frame policy remain unchanged.

## 1. Ruling

The `+0.0579914273313` degree angle fitted after the accepted SCI-006 east-X
correction is not another receptor convention to tune. It is the observable
effect of comparing two different polarization tangent bases:

- RadioSim transforms catalogue directions to topocentric `AltAz`, then
  reconstructs an idealized apparent-equatorial direction and parallactic angle
  from `AltAz`, geodetic latitude, and local apparent sidereal time; while
- `pyuvsim` asks `pyradiosky` to transport the catalogue ICRS coherency basis
  into its local apparent basis before evaluating the RIME.

RadioSim currently omits that catalogue-ICRS-to-operational-apparent tangent
basis transport. The dominant WP-6 angle is that omitted transport. Polar
motion, diurnal aberration, and the other topocentric Earth-orientation details
that the ideal spherical inverse does not reproduce are smaller remainder
terms. Refraction is excluded from the comparison by setting `pressure=0`.

WP-6 closes this as a **documented, fixture-scoped accuracy bound**, not by
changing the production Jones chain. A normal-environment unit test will bound
the physical angle with a public Astropy source-to-zenith oracle. The optional
cross-validation will additionally use the exact pinned `pyradiosky 1.1.0`
coherency rotation, per source and time, and must reduce the linear-polarization
residual to below `5e-10`. The historical global fitted angle remains a
non-vacuity control; it is not an allowed correction.

This is not an all-sky accuracy claim. Every numerical limit in this record is
restricted to the retained HERA-site, three-source, three-time cross-validation
fixture in Section 4.

## 2. Question and closure boundary

The question is:

> Does the tangent-basis rotation predicted from the exact fixture explain the
> direct `Q+iU` residual, with the documented sign and at a non-vacuous bound,
> without changing RadioSim's production frame policy?

The design answer is testable:

1. derive the angle independently through public Astropy operations in the
   normal environment;
2. reproduce the exact `pyradiosky` basis matrix in the pinned optional
   environment;
3. apply the exact correction to each source-time contribution before source
   summation; and
4. retain raw, single-global-angle, and exact-correction controls.

The row does **not** close when this memo lands, when a single fitted angle is
reproduced, or when one environment passes. Closure requires the hermetic tests,
the versioned artifact, all stated gates, and a separate read-only acceptance
at an exact candidate SHA. Only that acceptance may change `Fix.md` or call
`SCI-007` done.

## 3. Convention ledger and sign derivation

The following symbols are normative for WP-6.

| Symbol | Meaning |
|---|---|
| `B_NE` | ICRS catalogue coherency in a North/East tangent basis |
| `R(a)` | `[[cos(a), sin(a)], [-sin(a), cos(a)]]` |
| `S` | accepted SCI-006 North/East-to-east-X/north-Y swap `[[0,1],[1,0]]` |
| `psi_RS` | RadioSim parallactic angle from its operational apparent direction |
| `K` | pinned `pyradiosky` two-dimensional coherency-basis rotation |
| `alpha_PY` | angle defined by `K.T = R(alpha_PY)` |
| `L` | complex linear polarization `Q+iU` after the existing fringe-Hermitian mapping |
| `Delta` | wrapped RadioSim-minus-pyradiosky tangent-basis angle |

RadioSim's accepted east-X ideal response is

```text
J_RS = S R(psi_RS).
```

The pinned reference transports its catalogue coherency as

```text
B_local = K.T B_ICRS K,
K.T = R(alpha_PY).
```

Therefore

```text
Delta = wrap_pi(psi_RS - alpha_PY)
      = wrap_pi(psi_RS + atan2(K[0, 1], K[0, 0])).
```

The second equality is the implementation-facing form and fixes the sign that
an angle fitted from an aggregate visibility cannot fix. After the already
derived fringe-Hermitian mapping,

```text
L_RS = exp(+2j * Delta) L_PY.
```

Thus the RadioSim contribution is moved into the `pyuvsim` convention by

```text
L_RS_to_PY = L_RS * exp(-2j * Delta).
```

Moving the reference into RadioSim uses the opposite sign. The factor of two is
the spin-2 transformation of `Q+iU`. The exact correction must be applied to
each source at each time **before** source summation. Multiplying an aggregate
visibility by the fitted `+0.057991...` degree scalar is scientifically
insufficient because the nine fixture angles are not equal.

## 4. Exact fixture and axis order

The evidence fixture is the one already retained by
`tests/crossvalidation/test_pyuvsim_comparison.py` and
`output/crossvalidation/2026-08-08-pyuvsim-1.4.0.json`:

| Quantity | Value |
|---|---|
| Site | longitude `21.42830` deg, geodetic latitude `-30.72152` deg, height `1073.0` m |
| Antennas | synthetic ENU metres `[(0,0,0), (50,0,0), (0,70,0)]` |
| Times | UTC `2025-01-01T00:00:00`, then `+120 s` and `+240 s` |
| Sources | ICRS `(RA, Dec)` degrees `(20,-30.72)`, `(25,-26)`, `(15,-35)` |
| Stokes IQUV Jy | `(3,0.6,-0.4,0.2)`, `(1.5,-0.3,0.5,-0.1)`, `(2.25,0,0,0.9)` |
| Frequencies | `120`, `130`, and `140` MHz |
| Beam | one shared, east-X, unit-response BeamFITS |
| Mount | alt-az, with RadioSim `jones.P` enabled |
| Refraction | disabled explicitly with `pressure=0` |

Every `3 x 3` angle grid below is ordered `[time, source]`: rows are the three
UTC samples and columns are the three sources in the order above. The artifact
must retain the two-part UTC Julian dates (`jd1`, `jd2`) rather than reconstruct
them from rounded strings.

## 5. RadioSim's operational frame

RadioSim first asks Astropy for each catalogue ICRS direction in topocentric
`AltAz`. It then performs the exact ideal spherical inverse

```text
sin(dec)        = sin(lat) sin(alt) + cos(lat) cos(alt) cos(az)
cos(dec) sin(H) = -cos(alt) sin(az)
cos(dec) cos(H) = cos(lat) sin(alt) - sin(lat) cos(alt) cos(az)
RA_app          = LAST_app - H
```

and evaluates `psi_RS` from `(H, dec, geodetic latitude)`. This construction is
an internally consistent, apparent/equinox-of-date, **TETE-like idealization**.
It is not an exact Astropy `TETE` transform and must not be called one. In
particular, inverting ideal spherical horizontal geometry does not undo every
Earth-orientation and observer-velocity term that Astropy used to construct the
topocentric direction.

The idealization is already an intentional production contract. WP-6 does not
replace it. The missing piece exposed by cross-validation is instead the
transport of the input catalogue's ICRS polarization tangent basis into this
operational apparent basis before RadioSim applies its parallactic rotation.

## 6. Public normal-environment oracle

For every fixture source and time, the gating public oracle must:

1. install the explicit bundled IERS table using Section 8's context;
2. construct the source as `SkyCoord(..., frame="icrs")`;
3. construct one `AltAz(obstime=t, location=site, pressure=0)` frame;
4. transform the source to that frame and feed its `alt` and `az` through the
   production `DirectionBatch`/`parallactic_angle` seam to obtain `psi_RS`;
5. construct zenith as `SkyCoord(az=0 deg, alt=90 deg, frame=that_altaz)` and
   transform it to ICRS; and
6. evaluate `source_icrs.position_angle(zenith_icrs)` in exactly that order.

Astropy's `position_angle` is the great-circle bearing of the target zenith at
the source, positive East from North. The azimuth coordinate is singular at
zenith, but the transformed physical direction is well-defined; fixing azimuth
to zero makes its representation deterministic and does not add a convention.

Define

```text
Delta_public = wrap_pi(psi_RS - position_angle(source_ICRS, zenith_ICRS)).
```

This uses public Astropy interfaces only. It measures the physical
catalogue-north-to-local-vertical tangent angle independently; it is not an
algebraic copy of `pyradiosky`, an exact TETE/CIRS transform, or an all-sky
oracle. Its deliberately loose comparison to the pinned exact chain is part of
the optional evidence, not the normal test's dependency.

## 7. Exact optional `pyradiosky` oracle

The exact reconciliation follows the implementation used by `pyuvsim 1.4.0`,
with `pyradiosky==1.1.0` and `astropy==7.1.0` asserted before any measurement.
For each source and time, `SkyModel.update_positions` transforms the ICRS
Cartesian basis into topocentric coordinates; its private rotation helpers
orthogonalize the three-dimensional transform, add the per-source alignment,
and project it to the two-dimensional coherency rotation `K` used by
`coherency_calc`.

That exact pinned private path is appropriate only in the optional
cross-validation. The artifact must name the entry point and private helpers,
and the validator must fail on version or path drift. The normal gate uses the
public oracle instead, so a private API cannot become a production dependency.

Primary sources for this boundary are Astropy's [IERS
configuration](https://docs.astropy.org/en/stable/utils/iers.html),
[`AltAz`](https://docs.astropy.org/en/stable/api/astropy.coordinates.AltAz.html),
[`TETE`](https://docs.astropy.org/en/stable/api/astropy.coordinates.TETE.html),
and [`CIRS`](https://docs.astropy.org/en/stable/api/astropy.coordinates.CIRS.html)
documentation; the pinned [`pyradiosky 1.1.0` source](https://github.com/RadioAstronomySoftwareGroup/pyradiosky/blob/v1.1.0/src/pyradiosky/skymodel.py#L2378-L2677);
and the pinned [`pyuvsim 1.4.0` engine](https://github.com/RadioAstronomySoftwareGroup/pyuvsim/blob/v1.4.0/src/pyuvsim/uvsim.py#L325-L396).

## 8. Hermetic Earth-orientation policy

Every coordinate transform and every apparent-sidereal-time operation in a
WP-6 test or artifact generator must run inside this policy:

```python
table = iers.IERS_A.open(iers.IERS_A_FILE)
with (
    iers.conf.set_temp("auto_download", False),
    iers.earth_orientation_table.set(table),
):
    ...
```

This rejects network-selected and cache-selected table state. Setting only
`auto_download=False` is insufficiently explicit; the process must install the
table object it opened.

The normal test must assert the table class and bundled-file selection and must
record enough context in an assertion failure to diagnose drift. It must **not**
hard-code one table SHA-256 because the locked Python 3.11 and 3.12 environments
may carry different valid bundled Astropy tables. The dated optional artifact,
by contrast, must record its exact Astropy/IERS package version, table class,
basename, file SHA-256, per-time `DUT1`, `xp`, `yp`, and status values. No test
may silently read a user cache.

## 9. Historical probe disposition

Two earlier numbers are retained rather than rewritten:

- The `0.041`--`0.063` degree CIRS north-offset probe measured ICRS tangent
  transport into an apparent-equatorial basis. The executable reconstruction
  gives `0.0412766`--`0.0626888` degrees. It is scientifically consistent with
  the mechanism and is superseded, for closure purposes, by the public and
  exact source-time grids below.
- The old `0.200` degree scalar is **historical, unreproduced, and
  superseded**. No executable method, offset, target frame, wrapping/sign
  definition, per-cell values, command, or IERS state was retained. It is not
  evidence for a bound, must not be used as a denominator, and cannot be
  reconstructed by choosing whichever current frame happens to approach it.

The pre-SCI-006 fitted `-0.057568764952` degree angle is also historical. Its
sign and value included the now-retired Q-axis compensation; it is not a WP-6
target.

## 10. Reproduced prediction grids and residuals

The exact pinned `pyradiosky` prediction, in degrees, is

```text
[[0.054345212925136, 0.064500055921101, 0.042970437265188],
 [0.054345990627089, 0.064500756258174, 0.042971289516061],
 [0.054346765297796, 0.064501454583101, 0.042972137541478]]
```

The independent public source-to-zenith prediction, in degrees, is

```text
[[0.0546099344, 0.0641665154, 0.0438017219],
 [0.0546150544, 0.0641702071, 0.0438077466],
 [0.0546200070, 0.0641737558, 0.0438135715]]
```

The public physical-angle range is `7.64484265255e-4` to
`1.120043332414e-3` radians. Its maximum per-cell relative disagreement with
the exact pinned chain is about `1.96%`; the small difference reflects the
different public physical construction and `pyradiosky`'s exact
orthogonalization/alignment path.

The independently reproduced comparison metrics are:

| Metric | Value |
|---|---:|
| Raw direct linear residual | `2.052050642874229e-3` |
| Best single-global-angle residual | `1.9606576512107846e-4` |
| Exact source-time corrected residual | `2.400855498837282e-10` |
| Total-intensity residual | `2.3139573996814273e-10` |
| Explicitly mapped Stokes-V residual | `4.0701816228520426e-11` |

The global-angle control proves why source-time decomposition is required: it
improves the raw comparison but remains orders of magnitude above the exact
correction.

## 11. Executable bounds and non-vacuity

### Normal environment

The unit test must run in both the default Python 3.11 and locked `py312`
environments, use only RadioSim plus public Astropy, and assert for this fixture:

```text
6e-4 < min(abs(Delta_public))
max(abs(Delta_public)) < 1.2e-3 rad
max(abs(exp(2j * Delta_public) - 1)) < 2.4e-3
```

These are two-sided in substance: the lower bound prevents a vacuous zero-angle
implementation, while the upper bounds pin the documented milli-radian scale
and its spin-2 effect. They are not regression tolerances for arbitrary sky
positions.

### Optional pinned cross-validation

The source-decomposed comparison must additionally assert:

```text
max(abs(Delta_public - Delta_exact) / abs(Delta_exact)) < 0.10
abs(L_reference) > linear_scale * 1e-12       # comparison mask
1e-3 < raw_linear_residual < 5e-3
single_global_angle_residual > 1e-4
exact_source_time_corrected_residual < 5e-10
```

It must retain the unpolarized `<1e-11`, total-intensity, explicit Stokes-V,
and retired-Q-compensation controls already owned by the cross-validation. No
existing tolerance may be weakened to make WP-6 pass.

## 12. Production disposition and future policy

No production frame transport lands in WP-6. The current operational frame is
unchanged, and the documented fixture bound makes its limitation explicit.
Host-side per-direction Astropy transport would add cost precisely where
`PERF-001` remains unresolved and would regenerate scientific fingerprints for
a milli-radian effect.

`PrecisionConfig.ultra()` is **not** the present solution: today it changes
numeric precision, not tangent-basis transport. A separately designed future
frame-transport policy could use an ultra-science preset as its configuration
home, but only after defining sky coverage, performance, provenance, and
fingerprint consequences. This memo does not license such a feature.

## 13. Versioned artifact contract

The successor artifact must use schema `radiosim-crossvalidation-1.2.0` and be
validated deeply rather than checked for a few headline fields. It must contain
all of the following:

| Object | Required fields |
|---|---|
| Identity | schema, recorded UTC, slice, gating=`false`, exact clean generating-source SHA, clean-tree boolean at generation, lockfile SHA-256, exact generation command, and an explicit statement that the committed artifact is an evidence successor of that source commit |
| Reference | `pyuvsim` package/version/entry point, `pyradiosky` package/version, exact private rotation helpers and coherency call used |
| Runtime | Pixi environment and solve group; RadioSim, Python, NumPy, Astropy, pyuvdata, pyuvsim, and pyradiosky versions; platform, machine, and host scope |
| IERS | policy name, `auto_download=false`, table class, bundled basename, exact table SHA-256, package version, and per-time `DUT1`, `xp`, `yp`, and statuses |
| Fixture | EarthLocation; antenna ENU coordinates; source names, ICRS RA/Dec and IQUV; exact `jd1`/`jd2`; frequencies; beam/feed/mount metadata; pressure; cube shape |
| Axes | explicit order and shape for time, source, baseline, frequency, polarization, public grid, and exact grid |
| Equations | definitions of `B_NE`, `R`, `S`, `K`, `alpha_PY`, `Delta`, `L`; wrap interval; fringe-Hermitian and V-sign mappings |
| Correction | exact statement that RadioSim-to-pyuvsim uses `exp(-2j*Delta)` per source/time before summation, and the opposite direction uses the opposite sign |
| Predictions | complete public and exact angle grids in radians and degrees; public-versus-exact per-cell errors; extrema |
| History | structured dispositions for pre-SCI-006 fit, CIRS `0.041`--`0.063` probe, and `0.200` unreproduced scalar |
| Tolerances | every normal and optional bound as named structured values, including the linear mask and fixture scope |
| Metrics | raw, global-angle, exact-correction, I, V, unpolarized, retired-Q, scale, and ratio controls, with numerator/denominator definitions |
| Limits | production unchanged, fixture-only scope, optional-environment scope, private-API pin, platform scope, and explicitly unlicensed claims |

The validator must compare every material scalar, list, shape, version, policy,
equation/sign field, and provenance field. It must fail if the artifact is
absent, stale, generated from a dirty tree, generated from a source SHA other
than the explicitly approved generating-source commit, or silently changes
axis order. The later artifact-containing and acceptance commits are necessarily
evidence successors, so independent acceptance must authenticate their exact
HEADs and diffs separately rather than demand the self-referential condition
that the generated artifact names the commit that first adds that artifact.
Headline residuals without the source-time grids are not a valid 1.2.0
artifact.

## 14. Red-first implementation and verification stages

The evidence slice follows these stages in order:

1. **Design first.** Land this memo and obtain a design review before touching
   tests, output artifacts, or user-facing claims.
2. **Retain the raw failure.** Add a source-decomposed one-off assertion that
   direct, uncorrected linear polarization is `<5e-10`; run it red and retain
   the observed failure near `2.052050642874229e-3` before implementing the
   exact correction in the comparison.
3. **Public bound.** Add the bundled-IERS public-oracle unit test and run it in
   default and `py312`; the test may not import `pyuvsim` or `pyradiosky`.
4. **Exact optional comparison.** Add the pinned source-time `K` extraction,
   sign controls, raw/global/exact residuals, and existing I/V/unpolarized/Q
   controls.
5. **Validator before artifact.** Add the deep 1.2.0 artifact validator and
   retain its expected failure while the dated artifact is absent.
6. **Clean generation.** Generate the artifact from an explicitly approved,
   clean exact source commit; commit the artifact as its evidence successor;
   then rerun the validator, both normal environments, the full optional
   cross-validation, lint, format, doctest, non-slow tests, and strict Sphinx.
7. **Independent acceptance.** A read-only reviewer re-derives the sign from
   `K`, traces the production frame, authenticates the exact source and IERS
   provenance, reruns the normal and optional gates, checks that production and
   tolerances are unchanged, and issues an exact-SHA verdict. Only a successful
   verdict authorizes the `SCI-007` register flip and closure record.

The intended evidence paths are
`tests/unit/test_jones/test_sci007_frame_accuracy.py`,
`tests/crossvalidation/test_pyuvsim_comparison.py`,
`output/crossvalidation/2026-08-11-pyuvsim-1.4.0-sci007.json`,
`output/crossvalidation/README.md`,
`tests/unit/test_tier1h_documentation.py`,
`docs/user_guide/jones_terms.rst`, and `docs/changelog.rst`. They are not
modified by this design-gate slice.

## 15. Design-slice writable boundary

This design-gate slice is restricted to this memo, `docs/index.rst`, and the
live WP-6 status in `PostTier8RemediationPlan.md`. It does not edit `Fix.md`,
production code, tests, outputs, user documentation, changelog, configuration,
workflows, or lockfiles. Recording the design is not implementation and is not
scientific acceptance.

## 16. Evidence successor and independent acceptance

The approved clean generating source is
`9b50805cf9fe32124800d1e3946a87e3911c376b`. The exact accepted evidence
successor is `e20f636788e0b61ae6c854f64cbb7476c3cb9a50`, whose direct
parent is that source. The source contains no SCI-007 artifact; the successor
adds only the artifact, its README reproduction instructions, and the exact
source/artifact digest constants.

The retained record is
`output/crossvalidation/2026-08-11-pyuvsim-1.4.0-sci007.json`:

- size: 33,173 bytes;
- artifact SHA-256:
  `3a441ad606f365ac4110e30d9d8c2f3d7f5ea91c481aa70488dea72487e570ba`;
- generator SHA-256:
  `405a5d9fbee3becb1724d79f173e056e9e5da73cc73e3bed2cc2482d1b346c94`;
- lockfile SHA-256:
  `37db432e6ade2dd3e64222d5ccfe532be5671893b24ce29e717a3bbb12f38ade`;
- bundled IERS-A SHA-256:
  `ff2d22108e982bd86e326e01d797fa8bd545d51483359dd98e6c08fa5737f667`.

Independent review re-derived the sign from `K.T=R(alpha_PY)`. Because
`atan2(K[0,1],K[0,0])=-alpha_PY`, the record's
`Delta=wrap_pi(psi_RS+atan2(K[0,1],K[0,0]))` is the RadioSim-minus-pyradiosky
angle. With the accepted east-X and fringe-Hermitian mappings,
`L_RS=exp(+2j*Delta)L_PY`; the RadioSim-to-pyuvsim correction is therefore
`exp(-2j*Delta)` per source and time before summation.

The retained measurements are:

| Measurement | Accepted value |
|---|---:|
| Raw relative linear residual | `2.052050642874229e-3` |
| Exact source/time corrected residual | `2.400855498837282e-10` |
| Wrong-sign control | `4.103897953509379e-3` |
| Public angle minimum | `7.644842652547723e-4` rad |
| Public angle maximum | `1.1200433324138892e-3` rad |
| Maximum public spin-2 effect | `2.2400861964641163e-3` |
| Maximum public/exact relative disagreement | `0.019580918743243865` |
| Single-global-angle residual | `1.9606576512107846e-4` |

Both standalone validators and the public bound passed in the locked Python
3.11 and 3.12 environments. The documentation/evidence suite passed 195 tests
in each environment, and all five optional cross-validation tests passed.
Exact-SHA CI run `31461141190` then completed successfully at the accepted
successor: backend parity passed 92 tests, all quality and documentation gates
passed, and each of the six compatibility cells passed 5,475 tests with one
skip. The six artifact ZIP digests matched their API records; all 84
hexadecimal manifest observations were members of the active pin tables and no
candidate cube was present.

This acceptance closes the register row without changing production. It is a
bound for the retained HERA-site, three-source, three-time fixture, not an
all-sky or cross-platform frame-validation claim. `PrecisionConfig.ultra()`
changes numerical precision only and does not implement tangent-basis
transport. The artifact's own `SCI-007 OPEN pending independent acceptance`
field remains unchanged because it records the state at generation time; this
evidence-successor review and closure record carry the later DONE decision.
