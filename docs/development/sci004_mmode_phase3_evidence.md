---
orphan: true
---

# SCI-004 phase-M3 evidence reproduction record

This record accompanies the phase-M3 candidate evidence envelope and its
retained Section 11 performance record. It names exactly what
`docs/development/sci004_mmode_design.md` Section 14.2 requires: the tracked
generator, its official argument vector, the environment, the approved source
commit, both generated paths, their raw digests, and the commands that reproduce
and re-validate them.

## Generation

- **Generator (tracked at the approved `S3`)**:
  `tools/sci004_mmode_phase3_evidence.py`
- **Argv**: `generate` (no further arguments)
- **Pixi environment**: `default` (the standard-gate environment defined by the
  repository `pixi.toml` and `pixi.lock`, whose SHA-256 digests are recorded in
  the evidence environment and performance provenance)
- **Approved `S3` (`source_sha`)**:
  `b07925ab14b56b3ca0fa863f806290748a31df6b`
- **Evidence artifact path**:
  `docs/development/sci004_mmode_phase3_evidence.json`
- **Evidence artifact byte count**: `9186099`
- **Evidence artifact raw SHA-256**:
  `600b51ac4d70778ee2d3bdf7b8842b83ba77dc34d541784ad1ad7d8e5be5f8ae`
- **Retained performance-record path**:
  `output/benchmarks/reference/sci004/20260825T122048Z-macbook-pro-2.json`
- **Retained performance-record byte count**: `58844`
- **Retained performance-record raw SHA-256**:
  `07e59d3176866a78c17244849d6493365e9d410547e884cf56b254e60babe193`

## Reproduction

Start at a globally clean checkout of exactly the approved `S3` above: the
index and worktree must be clean, there must be no untracked paths,
`git rev-parse HEAD` must equal `source_sha`, the evidence artifact must be
absent, and the retained SCI-004 performance-record directory must contain no
JSON record.

A second checkout must own the Pixi environment that observes it. Run
`pixi install` in that checkout and do not reuse another working tree's
`editable` environment or path; a shared editable installation can import the
wrong source tree and invalidate the evidence venue.

Run the tracked preflight without writing an output:

```bash
pixi run python tools/sci004_mmode_phase3_evidence.py preflight
```

Generate the exact declared output pair with the official plain argument
vector:

```bash
pixi run python tools/sci004_mmode_phase3_evidence.py generate
```

The generator computes and validates both canonical payloads before
publication, then publishes by atomic no-overwrite operations in this order:
the retained performance record first and the evidence envelope last. A partial
set is invalid, cannot be reused, and does not authorize a second generation
run.

Re-validate the existing canonical bytes, schema literals, key order, digests,
and cross-file bindings with both generated paths named explicitly:

```bash
pixi run python tools/sci004_mmode_phase3_evidence.py check \
  --artifact docs/development/sci004_mmode_phase3_evidence.json \
  --performance output/benchmarks/reference/sci004/20260825T122048Z-macbook-pro-2.json
```

The strict successor validator is
`tests/unit/test_sci004_phase3_evidence.py`. At `E3`, its four approved
constants bind the exact `S3` SHA, the evidence artifact raw digest, the
retained performance path, and the performance record raw digest above. The
four bindings flip together; no validator logic changes at `E3`.

This `E3` state remains candidate evidence. Independent `A3` acceptance is a
separate successor phase and remains pending; this record does not accept M3 or
close SCI-004.
