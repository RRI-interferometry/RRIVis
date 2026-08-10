# CI-001 successor-gate and adjudication memo

**Decision date:** 2026-08-08
**Decision:** accept the reproducible `linux-64-py311` dispatch class; retain
observation-set membership as the primary gate; close CI-001.

## Scope and authority

This memo is the Post-Tier-8 WP-2/WP-3 acceptance record. It rechecked the live
checkout and GitHub Actions evidence rather than relying on the plan's
2026-08-05 status ledger. The adjudication applies only to the existing
`linux-64-py311` characterization pins. It changes no scientific algorithm,
configuration, dependency, or tolerance.

The decision applies the float64 predicate from `Tier6HybridRuntimePlan.md`
Section 13.5:

```text
|candidate - reference| <= atol + rtol * |reference|
rtol = 1e-12
atol = 1e-12 * max(1, max|reference|)
```

The absolute term is material. The circular-receptor cube has tiny non-zero
residuals where the accepted reference is exactly zero, so its maximum relative
delta is 1.0 while its maximum absolute delta is only `1.5742e-21`.

## Evidence and provenance

The historical harvest contains 11 runs with usable evidence: four fail-path
logs for the divergent class (`30705549269`, `30719161877`, `30726145633`,
`30734506813`) and seven accepted-class artifacts (`30745291786`,
`30749117742`, `30749776246`, `31093915574`, `31097500244`, `31147771176`,
`31237854462`). The comparator groups the four divergent observations into one
byte-stable class and the seven accepted observations into another. Raw harvest
data remains under gitignored `output/ci001-harvest/`; this memo retains the
findings.

The decisive experiment is [GitHub Actions run 31255085487](https://github.com/RRI-interferometry/RadioSim/actions/runs/31255085487):

- event: `workflow_dispatch`;
- remote source SHA: `34d9a9eed715e2f883b8c84d8158caad15beda82`;
- workflow: `.github/workflows/ci001-forced-experiment.yml`;
- result: all six matrix jobs completed successfully;
- environment: GitHub `ubuntu-24.04`, Pixi `default`, Python 3.11.13,
  NumPy 2.3.2, Astropy 7.1.0, OpenBLAS 0.3.30, glibc 2.39;
- accepted reference source: CI run `31093915574`, SHA
  `9945c4698cb229bf88c59d7bc77cac15488dda7c`, artifact
  `characterization-linux-64-py311` (artifact ID `8964718318`);
- runner image for the discriminating draw: `ubuntu24`, image version
  `20260720.247.2`, Linux `6.17.0-1020-azure`;
- CPU for the discriminating draw: Intel Xeon Platinum 8573C, four visible
  CPUs, four in `sched_getaffinity`, OpenBLAS runtime core `SkylakeX`.

The six experiment artifacts are `ci001-experiment-31255085487-draw1` through
`draw6` (artifact IDs `9021168718`, `9021169284`, `9021169257`, `9021170658`,
`9021169861`, `9021170838`). They retain the baseline fingerprint, `lscpu`,
runtime probes, per-variant pytest logs, exit codes, and summaries for 30 days
(through 2026-09-07). The committed memo is the durable record after those raw
artifacts expire.

Two earlier workflow-dispatch runs (`31124629033`, `31124638337`) ended with
matrix-job cancellations during a wider runner incident and are not treated as
numerical failures. A prior complete run (`31097508999`) sampled only AVX2/Zen
runners; all 24 variants matched the accepted class and therefore supplied a
control, not a divergent observation.

## Exact commands

```bash
gh workflow run ci001-forced-experiment.yml --ref main
gh run watch 31255085487 --exit-status --interval 10

for draw in 1 2 3 4 5 6; do
  gh run download 31255085487 \
    --name "ci001-experiment-31255085487-draw${draw}" \
    --dir "output/ci001-experiment/run-31255085487/draw-${draw}"
done

pixi run python tools/ci001_characterization_comparator.py \
  output/ci001-harvest \
  --experiment output/ci001-experiment/run-31255085487
```

The final repository verification commands are recorded in the acceptance
section below.

## Forced-discrimination result

Five draws exposed AVX2 to NumPy and selected OpenBLAS `Zen`. All four variants
in all five draws matched the accepted class byte-for-byte. Draw 4 exposed
AVX-512 and selected OpenBLAS `SkylakeX`; it reproduced the historical divergent
digests exactly.

| Variant | NumPy dispatch | OpenBLAS core | Result |
|---|---|---|---|
| V1 control | AVX-512 | SkylakeX | divergent class: three characterization tests red |
| V2 NumPy mask | AVX2 effective dispatch | SkylakeX | default config green; heterogeneous and circular cubes divergent |
| V3 BLAS override | AVX-512 | Haswell | heterogeneous and circular cubes green; default config divergent |
| V4 both overrides | AVX2 effective dispatch | Haswell | all 41 tests green; accepted class byte-for-byte |

This names the discriminator. NumPy's AVX-512 dispatch moves the default-config
cube. OpenBLAS's `SkylakeX` runtime core moves the heterogeneous-receptor and
circular-receptor cubes. Both contributions occur on one runner with one locked
environment and disappear independently under their corresponding override.
The earlier same-model observations are explained by hypervisor feature masking:
the CPU model string is not the effective dispatch state.

The V2 probe's private `__cpu_features__` mapping still lists several AVX-512
capability names after the mask. NumPy's `show_runtime()` report, which describes
the effective SIMD dispatch, reports only through AVX2; the independent V2/V3/V4
output changes agree with that effective-dispatch interpretation.

## Numerical adjudication

The three independent raw-cube measurements are:

| Pin | Divergent digest | `max|dV|` | Max relative | Differing elements | First differing index | Section 13.5 |
|---|---|---:|---:|---:|---|---|
| `config.yaml` | `312998e30a7c...` | `7.105427357601002e-15` | `1.815848028416042e-14` | 27,460 / 363,600 | `(0, 0, 22, 0)` | within |
| circular receptor | `708f97d2aadd...` | `1.5742275254395323e-21` | `1.0` (reference zero) | 540 / 1,080 | `(0, 0, 0, 1)` | within |
| heterogeneous receptors | `c7b51d022de6...` | `3.469446951953614e-18` | `6.26764878758987e-17` | 9 / 48 | `(0, 2, 0, 0, 0)` | within |

Every maximum absolute delta is below `1e-12`, the smallest possible Section
13.5 absolute allowance, before the relative term is added. The conclusion is
therefore independent of the reference cube's amplitude. The unchanged 38 of
41 characterization tests on the divergent control, exact reproduction of all
five historical failure digests, and exact return to the accepted class under
V4 distinguish this result from noise, stale artifacts, configuration drift,
or source drift.

The accepted second scientific fingerprints are
`89f38f6277d39c86...` (`config.yaml`) and `1c6e5bfac14b8af5...`
(circular receptor). Their raw-cube partners are the first two raw digests in
the table. Only the heterogeneous-receptor workload among the Section 13.4
micro-workloads changes.

## Gate decision

Observation-set membership remains the primary gate. A tolerance-only primary
gate would have hidden the fleet split and lost byte-level regression
sensitivity. The `rtol=1e-12` policy also remains correct, but it is a
classification/adjudication predicate with its Section 13.5 absolute term—not a
replacement for digest membership and not a bare maximum-relative-delta test.

The failure path now prints an explicit `WITHIN`/`OUTSIDE` Section 13.5 verdict,
including the computed absolute tolerance and maximum tolerance ratio. It also
retains a per-worker observed-digest manifest on pass and failure, so a passing
run remains classifiable after this second class is accepted. The comparator
uses the same full predicate and treats an accepted run without a digest
manifest as unclassified instead of assuming it is the original green class.

The predeclared conversion trigger is adopted unchanged: only a cell that
accumulates at least three legitimate machine classes, or a class that ceases
to be byte-stable, converts to a reference-cube Section 13.5 gate with its
digest advisory. No trigger has fired: this cell has two byte-stable classes.

## Acceptance and closure

WP-2 is accepted: the comparator, expanded fingerprint, nightly sampler,
failure-time fingerprint diff, 30-day artifacts, provenance, regression tests,
and documentation are present and independently exercised.

WP-3 is accepted: the supported experiment reproduced both classes, named the
two dispatch axes, measured the cubes, justified the second class under the full
Section 13.5 predicate, retained the digest gate, mechanized the verdict, and
added only the five observed digests belonging to the three changed numerical
measurements.

CI-001 is closed. Expected class membership is now explicit; a novel digest
still fails. The issue must be reopened (or a successor filed) if a third
legitimate class appears, either accepted class becomes non-byte-stable, or a
class is outside Section 13.5.

## Limitations and residual risks

- The raw Actions artifacts expire after 30 days. This memo retains provenance,
  exact digests, deltas, and commands; future accepted-class CI artifacts will
  recapture full reference cubes for both classes as each class occurs.
- The experiment identifies the effective dispatch axes, not the exact NumPy or
  OpenBLAS source instruction that introduces each last-bit difference.
- The result is specific to the locked `linux-64-py311` environment and the
  three affected cubes. It does not authorize tolerance relaxation or digest
  additions in another cell.
- GitHub-hosted runner images and CPU allocation can change. The expanded
  fingerprints and nightly sampler remain necessary even though CI-001 closes.

## SCI-006 successor-gate observation (2026-08-10)

SCI-006 changed the polarized characterization cubes through the ruled
east-X permutation, so its acceptance had to preserve every historical
dispatch class rather than treating a green workflow as sufficient evidence.
The missing `linux-64-py312` AVX-512 class was observed directly in CI run
[`31425908525`](https://github.com/RRI-interferometry/RadioSim/actions/runs/31425908525)
at exact source SHA `129344a42d2cf47af1c1f088964fcdb209701736`:

- job `93577429679`, `linux-64 / Python 3.12`;
- artifact `9077267004`, `characterization-linux-64-py312`;
- GitHub `ubuntu24` runner image `20260720.247.2`, Linux
  `6.17.0-1020-azure`, AMD EPYC 9V74 with four visible CPUs;
- Python 3.12.13, NumPy 2.4.6, Astropy 8.0.1;
- NumPy effective dispatch includes AVX-512 through `AVX512_SKX` and later
  feature groups;
- OpenBLAS 0.3.33 selected runtime core `Cooperlake`;
- the retained heterogeneous-receptor candidate digest is
  `1dd95c932cec803f06476205670a4902f0f53140d78261e08dc7aad5bac9c995`;
- the retained historical class cube is
  `11d4c0a5afd60d1682d62e5d85dcd3cde7c45d8e6b29411e22ccc35425847c46`.

Loading the retained arrays and applying the ruled row-and-column permutation
proved the candidate is byte-identical to `P V_old P^H` with maximum absolute
difference zero.  It did not match the same transform of the separately
retained AVX2-class historical cube
`73f340f1726163987eef8a387c7634a1e990264c8b23211918eea883749d54b7`.
The artifact also retained the observed-digest manifest and unchanged-workload
references.  This evidence adds one observed class member; it does not relax a
tolerance, replace observation-set membership, or alter CI-001's conversion
trigger.

### Retained pre-SCI-006 py311 dispatch mapping

WP-5 changes the heterogeneous-receptor cube, so the old digest must remain
historical evidence rather than an accepted post-correction pin. Forced run
[`31273416758`](https://github.com/RRI-interferometry/RadioSim/actions/runs/31273416758)
at exact source SHA `424b4b90dcb162f4d54a3cb4f4abf2516269ca44`
captured the missing post-correction `linux-64-py311` dispatch class:

- draw 3, job `93143054023`, artifact `9026308532`
  (`ci001-experiment-31273416758-draw3`);
- GitHub `ubuntu24` image `20260720.247.2`, Linux
  `6.17.0-1020-azure`, Intel Xeon 6973P-C with four visible CPUs;
- Python 3.11.13, NumPy 2.3.2, Astropy 7.1.0;
- NumPy effective dispatch through AVX-512, including `AVX512_SKX`;
- OpenBLAS 0.3.30 runtime core `Cooperlake`;
- retained candidate cube digest
  `9f07661c3348515e5fd1acc478606badd2f4c8a143f67008f8922aabedff04c5`.

The candidate filename was independently authenticated by recomputing the
Section 13.4 content digest. Because `P=P^-1=P^H`, applying `P` on both axes to
that retained cube reconstructs the historical pre-correction cube. Its full
digest is
`c7b51d022de6c917ee8a3359d2f5f20600a8259e52977555b5148dc32a4718c1`,
exactly the previously accepted AVX-512/OpenBLAS dispatch observation. Thus the
durable relation is:

```text
c7b51d022de6c917ee8a3359d2f5f20600a8259e52977555b5148dc32a4718c1
  -- P V_old P^H, byte-identical -->
9f07661c3348515e5fd1acc478606badd2f4c8a143f67008f8922aabedff04c5
```

The active post-SCI-006 pin set correctly contains `9f07661c...` and its
separate Haswell/AVX2-class peer `eda7da52...`; it deliberately does **not**
accept `c7b51d02...`. This preserves the complete CI-001 evidence without
allowing a pre-correction polarized cube to pass the current scientific gate.
