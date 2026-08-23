---
orphan: true
---

# SCI-004 phase-M1 evidence reproduction record

This record accompanies the phase-M1 evidence artifact and names exactly
what `docs/development/sci004_mmode_design.md` Section 14.2 requires: the
tracked generator, the environment, the approved source commit, the
artifact, its raw digest, and the commands that reproduce and re-validate
it.

## Generation

- **Generator (tracked at the approved `S1`)**:
  `tools/sci004_mmode_phase1_evidence.py`
- **Argv**: `generate` (no further arguments)
- **Pixi environment**: `default` (the standard-gate environment defined
  by the repository `pixi.toml`/`pixi.lock` whose SHA-256 digests the
  artifact records in its own environment block)
- **Approved `S1` (`source_sha`)**:
  `8dfc9af889c5d89f1783ac852f7d0cf6d4589740`
- **Artifact path**: `docs/development/sci004_mmode_phase1_evidence.json`
- **Artifact raw SHA-256**:
  `c3a0ee6b72fb6e7c6013d40a30ed1d90ec0771cb0f91de6eb1862bc6ae60b86a`

## Reproduction

At a globally clean checkout of exactly the approved `S1` above (clean
index, clean worktree, no untracked paths, `git rev-parse HEAD` equal to
`source_sha`), with the declared output absent:

```bash
pixi run python tools/sci004_mmode_phase1_evidence.py preflight
```

```bash
pixi run python tools/sci004_mmode_phase1_evidence.py generate
```

`generate` writes the single declared artifact by atomic no-overwrite
rename and prints its path, byte count, and raw SHA-256. To re-validate
an existing artifact's canonical bytes, schema literal, key order, and
cross-field rules without generating:

```bash
pixi run python tools/sci004_mmode_phase1_evidence.py check --artifact docs/development/sci004_mmode_phase1_evidence.json
```

The strict successor validator is
`tests/unit/test_sci004_phase1_evidence.py`; at `E1` its two approved
constants hold exactly the `source_sha` and artifact SHA-256 above, and
its `E1`-state tests authenticate the retained bytes, the unique
artifact-introducing commit, its direct `S1` parent, and the
Section 13.3 `E1` diff authority.
