# Cross-implementation validation records

This directory holds the Tier-2 cross-validation evidence that
`Tier7JonesSciencePlan.md` Section 29 requires: a comparison of RadioSim's
visibilities against an **independent simulator**, recorded rather than gated.

It is not the whole of RadioSim's validation. Section 29's Tier-1 evidence —
comparisons against published closed forms written out in the test body, and
against libraries already in the gating environment — lives in the ordinary
unit suite and runs in every gate. This directory exists because a second
*simulator* cannot be a gate: it is not in the locked default environment.

## What is here

| File | Contents |
| --- | --- |
| `2026-08-02-pyuvsim-1.4.0.json` | RadioSim vs `pyuvsim 1.4.0`, measured on `osx-arm64`, 2026-08-02 |
| `2026-08-08-pyuvsim-1.4.0.json` | SCI-006 east-X rerun: direct Q/U, explicit V mapping, and refitted SCI-007 residual |

## Reproducing it

The reference lives in an optional pixi environment that no CI job builds and
no gate runs:

```bash
pixi install --locked -e crossval
pixi run --environment crossval -- \
    python -m pytest tests/crossvalidation/ -m crossval
```

Set ``RADIOSIM_CROSSVAL_METRICS=1`` and add ``-s`` to emit the two measured
case records as machine-readable JSON, matching the dated artifact.

The `crossval` environment shares a solve group with `default`, so it is
`default` plus exactly one package — `pyuvsim-1.4.0-py3-none-any.whl` — on all
three locked platforms. That is deliberate: a comparison run against a
differently resolved stack would be evidence about a different build of
RadioSim.

`pyuvsim 1.4.0` is pinned, not floated. It is the newest release whose metadata
is satisfiable against this repository's `pyuvdata ==3.2.1`; `1.4.2` requires
`pyuvdata >=3.2.3`.

## What may be claimed from it

Only what a named case measured, at the tolerance it measured. The record's own
`claims_not_licensed_by_this_record` field lists the sentences this evidence
does **not** support, and `unresolved` lists the two questions the comparison
opened rather than closed. Section 29.2 of the plan is the governing rule:
"validated against pyuvsim" is a forbidden claim unless it names the quantity
and the tolerance.
