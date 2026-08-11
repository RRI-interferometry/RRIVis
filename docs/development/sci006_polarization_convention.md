# SCI-006 east-X polarization-convention ruling

**WP-4 design-gate record — 2026-08-08**

**WP-5 acceptance record — 2026-08-11:** the selected production correction is
independently accepted at exact candidate
`f5fa101e4ac345534636380720ce33ec93a31eae`; `SCI-006` is closed. CI run
`31434253575` passed quality, backend parity, and all six compatibility cells,
and all six retained artifacts were authenticated. Section 12 is the durable
acceptance record. The historical WP-4 present-tense trace below describes the
pre-correction implementation that the ruling evaluated.

**Ruling:** RadioSim's current default linear Jones binding is not scientifically
correct for the feed metadata it publishes. RadioSim uses an IAU north/east
sky brightness basis, but its zero-rotation linear receptor matrix is the
identity while feed 0 is labelled `x` and declared east-oriented. The correct
ideal response is the north/east-to-east-X/north-Y permutation. Consequently,
for an IAU `+Q` source at zero parallactic angle,

```text
XX - YY = -Q
```

not `+Q`. `pyuvsim` is an appropriate comparison reference because its result
agrees with the independently derived normative convention, not merely because
it is another simulator. WP-5 must implement the basis correction described
below. This memo does not make that runtime change, and `SCI-006` remains open
until WP-5 is implemented and independently accepted.

## 1. Scope and question

The question is deliberately narrower than "does polarized cross-validation
agree?":

> Given RadioSim's declared IAU sky convention and its declared
> `x_orientation="east"` linear feed pair, which sky component must feed `X`,
> and what sign must `XX - YY` have for positive sky Stokes `Q`?

The trace covers the maintained point-source and HEALPix paths from Stokes
input through brightness construction, receptor and Jones factors, baseline
contraction, correlation-axis labelling, standard-format export, and the
retained `pyuvsim` comparison. It does not adjudicate `SCI-007`'s sub-degree
reference-frame rotation or `SCI-005`'s future non-scalar beam model.

## 2. Convention ledger

These distinctions are essential; using the word "X" for more than one of
them caused the defect.

| Quantity | Convention used for this ruling | Consequence |
|---|---|---|
| Celestial linear reference | position angle starts at North and increases through East | `+Q` is north/south-dominant; `-Q` is east/west-dominant |
| Sky Jones-vector order | `(North, East)` at zero field rotation | the first brightness diagonal is north power |
| RadioSim linear feed order | `(x, y)` | visibility row/column 0 is X; row/column 1 is Y |
| RadioSim zero-rotation feed angles | `(pi/2, 0)` from North toward East | X is east, Y is north |
| Circular handedness | IAU incoming-wave sense | relevant to `V`, not to the linear `Q` ruling |
| Brightness normalization | half-power, `Tr(B) = I` | an unpolarized source gives `XX = YY = I/2` |
| In-memory correlation order | row-major | `(XX, XY, YX, YY)` and `(RR, RL, LR, LL)` |
| AIPS/UVFITS codes | serialization identifiers | identify products; do not define which physical direction X sees |

The IAU/IEEE distinction is limited but important. The IAU resolution adopts
the IEEE definition for the handedness of incoming elliptical polarization and
declares positive circular polarization to be right-handed. That is the
Stokes-`V` issue. The present `Q` ruling follows from the IAU's linear
North-through-East reference and is unchanged by the choice of harmonic
exponential or by an observer-versus-propagation-direction description of
circular handedness.

## 3. Authoritative evidence

1. **IAU.** The XV General Assembly's Commission 40 polarization resolution,
   section 8 on page 21, places the Stokes reference frame in right ascension
   and declination and measures electric-vector position angle from North
   through East. It separately adopts IEEE 211-1969 for incoming elliptical
   polarization. See the [IAU 1973 resolutions
   (PDF)](https://www.iau.org/static/resolutions/IAU1973_French.pdf).

2. **Hamaker & Bregman.** *Understanding radio polarimetry III*, A&AS 117,
   161–165 (1996), section 2, equations (1)–(3), and Figure 1 interpret the
   IAU/IEEE text. Their IAU Cartesian sky frame has `x` toward North and `y`
   toward East. Equation (1) therefore makes positive `Q` stronger along the
   north axis, while positive `U` is at position angle `pi/4`. See the
   [publisher PDF](https://aas.aanda.org/articles/aas/pdf/1996/07/dst6557.pdf).
   These paper-local `x` and `y` names are *sky-coordinate axes*, not
   pyuvdata's physical feed labels.

3. **RIME literature.** Smirnov, *Revisiting the radio interferometer
   measurement equation I*, A&A 527 A106 (2011), equation (7), writes the
   linear-basis brightness matrix and the Jones relation used by RadioSim;
   [DOI 10.1051/0004-6361/201016082](https://doi.org/10.1051/0004-6361/201016082).
   Carozzi & Woan, MNRAS 395, 1558–1568 (2009), section 4 and equation (28),
   make the local spherical brightness basis IAU-conforming when the
   equatorial system is used;
   [DOI 10.1111/j.1365-2966.2009.14642.x](https://doi.org/10.1111/j.1365-2966.2009.14642.x).

4. **IEEE.** [IEEE 145-2025](https://standards.ieee.org/ieee/145/7742/) is the
   current active *Standard for Definitions of Terms for Antennas* and
   supersedes IEEE 145-2013. The detailed historical incoming-wave
   interpretation needed here is available in the IAU resolution and Hamaker
   & Bregman; the public IEEE page establishes the current standards lineage.
   No IEEE handedness choice reverses linear `Q` in a fixed North/East frame.

5. **pyuvdata.** The official [feed-angle and x-orientation
   convention](https://pyuvdata.readthedocs.io/en/stable/conventions.html#feed-angles-and-x-orientation)
   says that zero degrees is North for a fixed zenith-pointing feed and that
   `feed_array=["x", "y"]` with `feed_angle=[pi/2, 0]` is east-oriented X:
   X points East and Y points North. The same page distinguishes the
   half-power (`pol_convention="sum"`) normalization, for which
   `I = XX + YY`, from doubled single-feed conventions.

6. **pyuvsim and pyradiosky source.** `pyuvsim` 1.4.0's
   [`Antenna.get_beam_jones`, lines 137–148](https://github.com/RadioAstronomySoftwareGroup/pyuvsim/blob/v1.4.0/src/pyuvsim/antenna.py#L137-L148)
   explicitly converts `UVBeam` vector-axis order into a Jones matrix whose
   axes are feed by `(theta, phi)`. Its
   [`UVEngine`, lines 325–362](https://github.com/RadioAstronomySoftwareGroup/pyuvsim/blob/v1.4.0/src/pyuvsim/uvsim.py#L325-L362)
   then evaluates `J1 B J2^H`. The installed reference's coherency comes from
   [`pyradiosky` 1.1.0](https://github.com/RadioAstronomySoftwareGroup/pyradiosky/blob/v1.1.0/src/pyradiosky/utils.py#L17-L61).
   Pyradiosky uses the opposite off-diagonal Stokes-`V` sign from RadioSim;
   that known mapping is independent of `Q`.

7. **AIPS products.** AIPS Memo 117, section 3.1.1, page 9, assigns the
   conventional STOKES-axis codes to `RR, LL, RL, LR` and to the four linear
   products (called `VV, HH, VH, HV` in the current revision, with the legacy
   `XX, YY, XY, YX` names noted in its footnote). See [AIPS Memo 117
   (PDF)](https://aips.nrao.edu/TEXT/PUBL/AIPSMEM117.PDF). This establishes
   product-code identity and file ordering; it does not override the feed-angle
   metadata or determine the sign of sky `Q`.

There is no contradiction among these sources. Apparent contradictions arise
only when HBS's sky-axis letters `(x=N, y=E)` are silently equated with an
east-oriented instrument pair `(X=E, Y=N)`.

## 4. RadioSim's executable convention at this commit

The live source trace is:

| Stage | Source | Observed behavior |
|---|---|---|
| Sky Stokes | `src/radiosim/core/sky/` loaders and containers | point and HEALPix paths carry explicit `I,Q,U,V`; a pyradiosky file contributes Stokes columns, not a pyradiosky-built coherency matrix |
| Brightness | `src/radiosim/core/polarization.py` | `B = 1/2 [[I+Q,U+iV],[U-iV,I-Q]]`; first axis is the positive-Q sky axis; `Tr(B)=I` |
| Feed metadata | `src/radiosim/core/receptor.py` | linear feed order is `(x,y)` and zero-rotation feed angles are `(pi/2,0)`, meaning X east and Y north |
| Receptor Jones | `src/radiosim/core/jones/receptor.py` | `C=M R(chi)` with `M(linear)=I2`; at `chi=0`, feed X receives sky axis 0 and feed Y receives sky axis 1 |
| Beam | `src/radiosim/core/beam/` and `src/radiosim/core/visibility.py` | every accepted analytic or BeamFITS response is scalar `E=e I2`; RadioSim canonicalizes the accepted unit BeamFITS into `I2`, so the beam does not repair the axis binding |
| Parallactic angle | `src/radiosim/core/jones/parallactic.py` | `P=R(psi)` rotates the sky field before `C`; it does not change the zero-angle identity mismatch |
| Jones chain | `src/radiosim/core/visibility.py` | both maintained solvers use `J=H G B Rc Kd X D C E P T Z` and one brightness construction |
| Contraction | `src/radiosim/core/contraction.py` | baseline response is `J_p B J_q^H` |
| Result labels | `src/radiosim/core/result.py` and `src/radiosim/core/polarization_basis.py` | the 2-by-2 result is flattened row-major and labelled `(XX,XY,YX,YY)` for linear output |
| Standard export | `src/radiosim/io/standard_visibility.py` | exports feeds `(x,y)`, the resolved angles `(pi/2,0)`, and pyuvdata/AIPS linear product codes; consumers therefore correctly read the first row as east-X |

Thus the current default ideal-linear chain at zero rotation is

```text
J_current = I2,
V_current = B,
XX_current - YY_current = Q.
```

The arithmetic and the exported metadata disagree about what row 0 means.
This is not a mere documentation omission: a standard-format consumer sees
`XX` plus an east-oriented X feed, while the number stored in `XX` is the
north-axis power.

## 5. Equation-level derivation

Let the IAU sky Jones vector at zero parallactic angle be ordered

```text
e_NE = [E_N, E_E]^T.
```

RadioSim's half-power brightness matrix is

```text
       1 [ I+Q   U+iV ]
B_NE = - [             ].
       2 [ U-iV   I-Q ]
```

For an east-oriented X feed, the instrument voltage order is

```text
v_XY = [E_E, E_N]^T = P e_NE,

    [ 0  1 ]
P = [      ].
    [ 1  0 ]
```

For identical ideal antennas and unit geometric phase,

```text
V_XY = P B_NE P^H

       1 [ I-Q   U-iV ]
     = - [             ].
       2 [ U+iV   I+Q ]
```

Therefore

```text
XX = (I-Q)/2        XY = (U-iV)/2
YX = (U+iV)/2       YY = (I+Q)/2

XX + YY = I
XX - YY = -Q
XY + YX = U
-i (XY - YX) = -V.
```

The last line is why this *axis-order* correction changes both `Q` and the
Stokes `V` inferred from linear cross-hands while leaving `I` and `U`
unchanged. It does not revoke RadioSim's deliberate HBS/Smirnov coherency
choice `B[0,1]=(U+iV)/2`; it correctly converts that sky-basis matrix into an
east-X feed basis.

For a static feed rotation `chi` and parallactic angle `psi`, the desired
linear receptor response is

```text
C_linear P_parallactic = P R(chi) R(psi)
                       = P R(chi + psi).
```

The permutation is therefore a fixed basis map, not a fitted rotation and not
a source-dependent coordinate correction.

## 6. Minimal deterministic example

Choose a phase-centre point source and identical ideal antennas so the
geometric fringe is one. Let

```text
I=1, Q=+0.6, U=0, V=0.
```

Positive `Q` means the north component contains more power:

```text
B_NE = [[0.8, 0.0],
        [0.0, 0.2]].
```

The predicted correlation products are:

| Implementation | XX | XY | YX | YY | XX - YY |
|---|---:|---:|---:|---:|---:|
| closed form, east X / north Y | 0.2 | 0 | 0 | 0.8 | -0.6 |
| current RadioSim, `C_linear=I2` | 0.8 | 0 | 0 | 0.2 | +0.6 |
| pyuvsim 1.4.0 unit BeamFITS | 0.2 | 0 | 0 | 0.8 | -0.6 |

This example is independent of fringe sign, circular handedness, Stokes-`V`
coherency sign, interpolation error, and `SCI-007`'s frame rotation.

## 7. Executable evidence and independent pyuvsim trace

The gating analytic oracle is retained in
`tests/unit/test_core/test_sci006_convention_gate.py`. It writes out `P` in the
test body and proves:

- positive IAU `Q` produces negative `XX-YY` for east X;
- the permutation maps locally recovered `(I,Q,U,V)` to `(I,-Q,U,-V)`; and
- a pure-I brightness matrix is bit-identical under the permutation.

Run it with:

```bash
pixi run test -- tests/unit/test_core/test_sci006_convention_gate.py
```

The optional independent reference probe is retained as
`test_pyuvsim_east_x_unit_beam_matches_the_sci006_closed_form` in
`tests/crossvalidation/test_pyuvsim_comparison.py`. It uses pyuvsim's own
`Antenna.get_beam_jones`, pyradiosky's own coherency constructor, and the
existing east-X unit BeamFITS fixture:

```bash
pixi run --environment crossval -- \
  python -m pytest tests/crossvalidation/test_pyuvsim_comparison.py \
  -m crossval -k east_x_unit_beam
```

The evaluated matrices agree to double-precision interpolation round-off with:

```text
J_pyuvsim(feed by theta,phi) = [[0,1],
                                [1,0]]

V_pyuvsim = [[0.2,0],
             [0,0.8]].
```

This identifies pyuvsim's actual convention independently. The existing full
cross-validation remains a characterization: it still carries its explicit
Q-axis compensation until WP-5 changes RadioSim.

## 8. Root cause and rejected explanations

**Root cause:** `src/radiosim/core/jones/receptor.py` uses one matrix identity
for two different concepts. `M(linear)=I2` assumes feed order follows the
north-first HBS sky basis, while receptor resolution and standard export call
the same rows `(X=east,Y=north)`. The missing fixed permutation is at the
sky-to-receptor basis boundary.

The evidence rejects the alternatives:

- **Reference/setup mismatch:** no. The unit beam is scalar and identical in
  both codes, its feed angles are explicit, and pyuvsim's own source constructs
  the permutation.
- **Feed-label ambiguity:** no. RadioSim, pyuvdata, the BeamFITS fixture, and
  the exported `Telescope` metadata all declare X east and Y north.
- **Coordinate-transform or parallactic-angle issue:** no for SCI-006. The
  discrepancy exists at zero rotation before any time-dependent transform.
  The residual transform issue remains separately filed as `SCI-007`.
- **Intentional alternative convention:** no. An internal north-first feed
  order could be valid if products were labelled N/E or X were declared north,
  but RadioSim exports the opposite physical orientation. Internal freedom
  cannot make contradictory exported metadata correct.
- **Stokes-`V` convention mismatch:** no for the Q ruling. RadioSim and
  pyradiosky deliberately differ in the off-diagonal `V` sign, but the
  deterministic example has `V=0`.

## 9. Final ruling, confidence, and limitations

**Final ruling: select WP-5 Branch A.** RadioSim must change its sky-to-linear
feed basis mapping. It must not relabel the present numbers, silently redefine
`x_orientation`, or document the current identity mapping as intentional.

Confidence is **high**. The result is fixed independently by the IAU definition
of `+Q`, the declared pyuvdata feed angles, the RIME, RadioSim's executable
identity mapping, pyuvsim's executable permutation, and a convention-isolating
numeric example. No source needed for the ruling is ambiguous or
contradictory.

Limitations of the ruling:

- it does not select a new Stokes-`V` harmonic-exponential convention;
  RadioSim's existing HBS/Smirnov choice remains in force;
- it does not explain or close the `-0.0576` degree residual (`SCI-007`);
- it does not validate non-scalar or cross-polar BeamFITS responses
  (`SCI-005`); and
- it establishes the design and acceptance contract, not the WP-5 runtime
  implementation.

## 10. Precise WP-5 implementation instructions

### 10.1 Give the basis mapping one owner

Define one immutable sky-to-output-basis table in
`src/radiosim/core/polarization_basis.py`. In mathematical form its two rows
are:

```text
M_linear_xy   = P = [[0,1],[1,0]]
M_circular_rl = S = (1/sqrt(2)) [[1,i],[1,-i]].
```

`P` maps the IAU `(North,East)` sky basis to `(X=east,Y=north)`. `S` maps the
same sky basis directly to RadioSim's IAU `(R,L)` output. Do not scatter copies
of either matrix across the Jones, output, or beam modules.

Derive the receptor and reporting transforms from those canonical matrices:

```text
C(native,chi) = M_native R(chi)
H(native->output) = M_output M_native^H.
```

This produces the four required transforms:

```text
linear   -> linear_xy:   I2
circular -> circular_rl: I2
linear   -> circular_rl: S P
circular -> linear_xy:   P S^H.
```

It also guarantees `H C = M_output R(chi)` when intervening feed-space terms
are identity/scalar. Do **not** apply a universal `P` to the brightness matrix
or scalar E-Jones: that would unnecessarily change ideal circular output and
would hide the physical feed boundary again.

### 10.2 Preserve names and change numerical semantics explicitly

Keep all of these stable:

- configuration tokens `linear`, `circular`, `linear_xy`, and `circular_rl`;
- feed arrays `(x,y)` and `(r,l)`;
- zero-rotation feed angles `(pi/2,0)` for east-X linear feeds;
- correlation-axis orders `(XX,XY,YX,YY)` and `(RR,RL,LR,LL)`;
- AIPS/pyuvdata codes and HDF5/UVFITS/MS field shapes; and
- the half-power brightness normalization and RadioSim's current sky-basis
  Stokes-`V` sign.

Change the meaning of linear products to match those already-declared names.
For ideal homogeneous linear output at zero field rotation, WP-5 must produce

```text
(XX,XY,YX,YY)
= ((I-Q)/2,(U-iV)/2,(U+iV)/2,(I+Q)/2).
```

This is a pre-v1 numerical breaking change. Existing RadioSim linear products
with polarized signal were effectively ordered as north/east while labelled
east-X/north-Y. Migration text must state that the corrected ideal relation is
the matrix permutation

```text
V_new = P V_old P^H,
```

or, in row-major order, `[XX,XY,YX,YY]_new =
[YY,YX,XY,XX]_old` for the ideal/scalar chain. Do not promise that formula for
configurations with non-commuting feed-dependent Jones terms; those require the
audit below.

### 10.3 Audit the full Jones blast radius

Update `src/radiosim/core/jones/receptor.py` to consume the canonical table and
then audit every term between `H` and `C`:

- scalar E and feed-symmetric scalar gains commute and preserve the simple
  permutation relation;
- `D`, `X`, and feed-asymmetric `G`, `B`, `Kd`, or `Rc` operate in native feed
  coordinates and must remain keyed to physical X=east/Y=north after the
  correction;
- nonzero static rotation and `P` must satisfy
  `C_linear P = P R(chi+psi)`;
- ideal circular output must remain
  `RR=(I+V)/2`, `RL=(Q+iU)/2`, `LR=(Q-iU)/2`, `LL=(I-V)/2` under RadioSim's
  existing `V` convention; and
- mixed-native-basis output must agree with the canonical `M_output` when
  intervening terms are ideal, while tests with a feed-asymmetric term must
  prove which physical feed each configured value affects.

An unpolarized brightness matrix is invariant under `P`, but a blanket claim
that *every possible unpolarized configured run* is unchanged would be too
broad when non-scalar feed-space corruptions are present. The required
bit-identity proof applies to the shipped and characterization workloads whose
intervening response is scalar/ideal; other configurations need physics-based
expected values.

### 10.4 Cross-validation and retained artifacts

In `tests/crossvalidation/test_pyuvsim_comparison.py`:

1. remove the compensating `-theirs_Q` comparison and compare Q/U in the same
   east-X feed frame;
2. retain the fringe/Hermitian mapping as its own named operation;
3. compare Stokes `V` with an explicit sign mapping because RadioSim and
   pyradiosky still use opposite coherency `V` signs once the axis-order
   difference is removed;
4. rename comments/tests so mappings 2 and 3 are no longer described as a
   cancelling pair; and
5. rerun the full optional cross-validation and commit a **new dated** JSON
   artifact. Keep `output/crossvalidation/2026-08-02-pyuvsim-1.4.0.json` as the
   historical pre-ruling record; do not overwrite it. Update
   `test_the_committed_artifact_describes_this_comparison` to name the new
   artifact and cases.

The Q-axis correction should remove the order-unity unswapped control. The
remaining linear residual should still be measured, not assumed, because
`SCI-007` is a frame-species question.

### 10.5 Fingerprints and evidence

Use one controlled fingerprint-regeneration event, following the accepted
CI-001 dispatch-class procedure in `docs/development/ci001_adjudication.md`:

- run the before/after workload matrix and retain a machine-readable diff;
- prove shipped and characterization pure-I/scalar workloads are byte-for-byte
  unchanged;
- for applicable ideal linear-output polarized workloads, prove each new
  2-by-2 cube is `P V_old P^H`, not merely "different";
- prove ideal circular-output cubes are unchanged;
- separately explain workloads containing feed-asymmetric Jones terms;
- regenerate affected `scientific_sha256` and raw-cube pins across all six
  `(platform, Python)` cells, including both accepted `linux-64-py311`
  dispatch classes where a changed workload reaches that arithmetic; and
- retain the five accepted CI-001 dispatch-class digests as historical
  adjudication evidence. WP-4 must not and does not alter them.

Never loosen a tolerance or append an unexplained digest.

### 10.6 Documentation and migration

Update at least:

- `src/radiosim/core/polarization.py` and
  `src/radiosim/core/polarization_basis.py` convention prose;
- `src/radiosim/core/jones/receptor.py` equations and docstrings;
- `docs/user_guide/jones_matrices.rst` (replace the current incorrect
  zero-rotation linear product equations);
- `docs/user_guide/configuration.rst` and any API/output convention table that
  equates east X with the first sky axis;
- the `[Unreleased]` changelog; and
- `docs/migration_guide.md`, with the pre-v1 numerical migration above and a
  warning for feed-asymmetric Jones configurations.

No public function signature or configuration key needs to change. Public
*numerical semantics* and standard-format product interpretation do change,
which is why changelog and migration entries are mandatory.

### 10.7 WP-5 acceptance criteria

WP-5 is acceptable only when all of the following are fresh evidence:

1. A production-path Tier-1 test (not a test-local transform) evaluates
   `I=1,Q=0.6,U=V=0`, east X/north Y, and obtains
   `(XX,XY,YX,YY)=(0.2,0,0,0.8)`.
2. Production matrices prove `(I,Q,U,V)->(I,-Q,U,-V)` for linear output and
   pure-I invariance.
3. Nonzero `chi` and `psi` prove `P R(chi+psi)` for linear feeds.
4. Ideal circular-native/output and linear-native/circular-output tests prove
   the stated circular products remain correct.
5. Mixed native bases and at least one feed-asymmetric Jones case prove the
   feed-coordinate semantics rather than relying on commutation.
6. Point-source and HEALPix solvers, NumPy/JAX/Dask parity, result construction,
   HDF5, UVFITS, and Measurement Set round trips all preserve the corrected
   labels, feed metadata, and values.
7. The crossval Q compensation is gone; the new artifact records direct Q/U
   comparison, the explicit pyradiosky V mapping, and a freshly fitted
   SCI-007 residual.
8. The scripted fingerprint proof and all required six-cell pins are reviewed;
   no tolerance is weakened and no digest is unexplained.
9. Changelog, migration, convention docs, doctests, lint, formatting,
   typecheck, the non-slow suite, optional crossval suite, and Sphinx
   warnings-as-errors all pass.
10. An independent acceptance record re-derives the result and only then marks
    `SCI-006` done in `Fix.md`.

Likely implementation/test files include
`src/radiosim/core/polarization_basis.py`,
`src/radiosim/core/jones/receptor.py`, receptor/basis/solver unit tests,
`tests/characterization/test_tier6_current_behavior.py`,
`tests/characterization/test_tier7_current_behavior.py`,
`tests/crossvalidation/test_pyuvsim_comparison.py`, and a new dated file under
`output/crossvalidation/`. The implementation must use the live dependency
graph to confirm the final list rather than treating this forecast as a write
grant.

## 11. Effect on SCI-007

The ruling fixes the frame in which `SCI-007` must be measured. The existing
cross-validation manually applied the same Q-axis reflection that WP-5 will
move into RadioSim, so the remaining residual magnitude near `2.055e-3` is
expected to remain of the same order after the correction. Its fitted angle or
complex-ratio sign must be recomputed because the comparison expression and V
mapping change.

After WP-5:

1. fit `Q+iU` directly in the common east-X frame, with no Q sign
   compensation;
2. evaluate the per-source/per-time apparent-frame prediction described in
   `PostTier8RemediationPlan.md` WP-6; and
3. close `SCI-007` only if the predicted frame-species difference explains the
   new fit and the corrected residual reaches the documented bound.

WP-4 therefore unblocks WP-6's ruled-frame computation but does not supply its
closure evidence.

The accepted WP-5 comparison uses `Q+iU` directly and measures a relative
linear residual of `0.002052050642874229`. Its least-squares complex ratio has
modulus `1.0000830200328927` and a fitted residual rotation of
`+0.057991427331288835` degrees. The explicit pyradiosky V-sign comparison
agrees to `4.0701816228520426e-11` relative. These measurements are recorded in
`output/crossvalidation/2026-08-08-pyuvsim-1.4.0.json`; they refit SCI-007 but do
not close it.

## 12. WP-5 independent acceptance — DONE 2026-08-11

### 12.1 Exact candidate and re-derived result

The accepted candidate is
`f5fa101e4ac345534636380720ce33ec93a31eae` on
`codex/sci006-wp5-candidate`. The branch, its remote, and the clean local tree
all resolved to that SHA during the read-only review. The candidate contains no
dependency, lock-file, simulator-submodule, tolerance, or ignored-output change.

The reviewer re-derived, from the production matrices,

```text
P B P^H = (1/2) [[I-Q, U-iV],
                 [U+iV, I+Q]]
```

and therefore the row-major linear products

```text
(XX, XY, YX, YY) = ((I-Q)/2, (U-iV)/2, (U+iV)/2, (I+Q)/2)
XX - YY = -Q
```

The circular transform remains

```text
(RR, RL, LR, LL) = ((I+V)/2, (Q+iU)/2, (Q-iU)/2, (I-V)/2)
```

Production resolves `C=M_native R(chi)` and
`H=M_output M_native^H`. Mixed native/output bases consequently collapse to
the requested ideal output transform only when no native-feed effect intervenes.
The feed-asymmetric witness correctly keeps gains and other such terms attached
to their physical native feeds instead of applying an invalid commuting
shortcut.

### 12.2 Fresh executable evidence

The final review included:

- 432 focused production, science, output, fingerprint, CI, and documentation
  tests;
- 88 NumPy/JAX/Dask backend-parity tests covering point and HEALPix paths,
  mixed bases, mount rotations, and optional Jones terms;
- five optional `pyuvsim` cross-validation tests at the exact candidate; and
- the full non-slow, lint, formatting, typecheck, doctest, and strict Sphinx
  gates represented by the final exact-SHA CI run.

The cross-validation has no retired Q-axis compensation. It records direct
linear residual `0.002052050642874229`, fitted rotation
`+0.057991427331288835` degrees, ratio modulus `1.0000830200328927`, explicit
pyradiosky V-mapped residual `4.0701816228520426e-11`, and a deliberately
failing retired-compensation control `0.697576347665902`. These values refit
the still-open SCI-007 question; they do not close it.

### 12.3 Remote exact-SHA evidence

GitHub Actions run `31434253575` is a push-triggered `CI` run at the exact
accepted SHA. All eight jobs completed successfully:

| Scope | Job ID |
|---|---:|
| NumPy/JAX-CPU backend parity | `93604879424` |
| lint, metadata, types, doctests, examples, and Sphinx | `93604879477` |
| osx-arm64 / Python 3.12 | `93604879480` |
| osx-64 / Python 3.11 | `93604879503` |
| osx-arm64 / Python 3.11 | `93604879556` |
| linux-64 / Python 3.12 | `93604879588` |
| linux-64 / Python 3.11 | `93604879623` |
| osx-64 / Python 3.12 | `93604879636` |

The six unexpired characterization artifacts all report the same workflow-run
head SHA. Their IDs and API-authenticated archive SHA-256 values are:

| Environment | Artifact ID | Archive SHA-256 |
|---|---:|---|
| linux-64-py311 | `9080470037` | `d2e6c1ed623b16c9370bbf710dd671807a876860fca8ecafe0bd6f1fe0810c5e` |
| linux-64-py312 | `9080472471` | `113b8c427e91813c3a28a08041a726d769cd20a3bd2bd1bd7bec2c70139def90` |
| osx-arm64-py311 | `9080524924` | `41d843304f9da02a6e349093f9f3a5ee303253fca886dff855ede9fdc7a837f9` |
| osx-arm64-py312 | `9080529903` | `52e440c4aab005327626f3d7c64388c066faf79a72a92abd2f0403463c37d0ed` |
| osx-64-py311 | `9080665431` | `a85d7f022ae0978f895fd999c4fc2c3dcdc1713f0999dcd6f84a7237d49a7bd0` |
| osx-64-py312 | `9081128353` | `c3eb2782ff332629422877cbb5adab167568956e9c913b5dc416ad01151eb21b` |

Every artifact has 14 valid observations over the ten required slugs, with
duplicate observations agreeing. Every measured digest belongs to the active
pin table; every retained NPY hashes to its filename under the documented
workload or raw-config recipe; no candidate cube remains in an `observed_cubes`
directory. The one non-hex `an-unrecorded-environment\tmeasured` row is a
pre-existing negative-test fixture. The WP-5 parser reads it, but the comparator
never queries its synthetic slug and it does not enter an accepted pin.

### 12.4 Changed and unchanged cubes

In every compatibility cell, the four applicable polarized workloads are
byte-exact `P V_old P^H`. The two unpolarized workloads and both shipped-config
raw cubes are unchanged active references; both shipped scientific fingerprints
also remain members of their accepted environment tuples. The non-commuting
feed-asymmetric case is covered by its own physical-feed oracle.

The final green artifacts carry the accepted cubes under `reference_cubes` and
therefore do not persist a generated WP-5 comparator JSON report. The
independent review recomputed the filename digests, active membership, and
changed/unchanged relations directly from those retained arrays and manifests.
The controlled local before/after regeneration is retained separately as the
machine-readable
[`sci006_fingerprint_diff.json`](sci006_fingerprint_diff.json), generated by
`tools/wp5_sci006_fingerprint.py compare` from pre-correction `22908c1e` and
post-correction implementation `1efcbc6b`, under Python `3.11.13` and NumPy
`2.3.2`. The original generator was the
`tools/wp5_sci006_fingerprint.py` blob
`3dce64c983327b9792e63314274c7277a2c17fc3` at `1efcbc6b`. Its exact command
was:

```text
pixi run -e default python tools/wp5_sci006_fingerprint.py compare \
  output/wp5-sci006/before-with-feed/osx-arm64-py311 \
  output/wp5-sci006/after-with-feed/osx-arm64-py311 \
  --output output/wp5-sci006/comparison-with-feed-osx-arm64-py311.json
```

The before and after manifest SHA-256 values are respectively
`c1e99cc3c3ece8ae6f7bcc6026c8e0b510a505fedd7e47cd8605d93841fe16db`
and `f3f8b82e890eb495faedd41ee13dca551bb5542a077ff6246eaa15bd5b5bda95`.
The later generator blob `8cf5c956f64f0fedc0cc5be7bdba45b1f55c5e9f` at
closure candidate `de0ed57b` reproduced the report byte-for-byte. The report's
SHA-256 is
`2a053a4fedeb426ebfeb261ac6e33586121fd931a2299fbbad3d166a08f92ef3`;
the regression suite pins its bytes, source heads, workload inventory, exact
relations, and all-pass verdict.

The post-correction heterogeneous-receptor classes are
`eda7da522277fc70576abd882e2b4984a3823e19c5ca588b35b530fc65af107c`
and `9f07661c3348515e5fd1acc478606badd2f4c8a143f67008f8922aabedff04c5`
for linux py311, and
`6b7d21ab4358b8a6597c1018eb4bafa391cc42b4358d656c4f1d06dee5eb2c97`
and `1dd95c932cec803f06476205670a4902f0f53140d78261e08dc7aad5bac9c995`
for linux py312. The latter missing class was observed—not inferred—in run
`31425908525`, job `93577429679`, artifact `9077267004`; its AVX-512/OpenBLAS
`Cooperlake` candidate is byte-exact to the permutation of historical
`11d4c0a5...`, not the AVX2 `73f340f1...` reference.

The original five CI-001 dispatch digests remain durable provenance. Four
unchanged shipped-configuration digests also remain active pins; only the
superseded heterogeneous `c7b51d02...` digest is historical-only after its
accepted `9f07661c...` east-X successor was recorded. Observation-set
membership remains the primary gate, and the Section 13.5 `rtol=1e-12` plus
scaled absolute term remains only the novel-digest adjudication rule. No
tolerance was weakened and no digest was accepted without a retained candidate
and a byte-level explanation.

### 12.5 Verdict and remaining boundary

All ten WP-5 acceptance criteria in Section 10.7 are satisfied. `SCI-006` is
therefore **DONE**. This acceptance does not claim full apparent-place
polarization transport: the post-correction `+0.0579914273` degree residual is
owned by still-open `SCI-007` and must be independently explained and bounded
by WP-6.
