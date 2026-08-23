---
orphan: true
---

# SCI-004 phase-M2 evidence reproduction record

This record accompanies the phase-M2 evidence artifact and names exactly
what `docs/development/sci004_mmode_design.md` Section 14.2 requires: the
tracked generator, the environment, the approved source commit, the
artifact, its raw digest, and the commands that reproduce and re-validate
it.

## Generation

- **Generator (tracked at the approved `S2`)**:
  `tools/sci004_mmode_phase2_evidence.py`
- **Argv**: `generate` (no further arguments)
- **Pixi environment**: `default` (the standard-gate environment defined
  by the repository `pixi.toml`/`pixi.lock` whose SHA-256 digests the
  artifact records in its own environment block)
- **Approved `S2` (`source_sha`)**:
  `399245793e812ed549fac23c1b69b2c6c61aecd4`
- **Artifact path**: `docs/development/sci004_mmode_phase2_evidence.json`
- **Artifact raw SHA-256**:
  `fc04d9a5115f1fc8609480c531bbf5b49cfc2d4933bd17e2a0d7612ebc91555a`

## Reproduction

At a globally clean checkout of exactly the approved `S2` above (clean
index, clean worktree, no untracked paths, `git rev-parse HEAD` equal to
`source_sha`), with the declared output absent:

```bash
pixi run python tools/sci004_mmode_phase2_evidence.py preflight
```

```bash
pixi run python tools/sci004_mmode_phase2_evidence.py generate
```

`generate` writes the single declared artifact by atomic no-overwrite
rename and prints its path, byte count, and raw SHA-256. To re-validate
an existing artifact's canonical bytes, schema literal, key order, and
cross-field rules without generating:

```bash
pixi run python tools/sci004_mmode_phase2_evidence.py check --artifact docs/development/sci004_mmode_phase2_evidence.json
```

The strict successor validator is
`tests/unit/test_sci004_phase2_evidence.py`; at `E2` its two approved
constants hold exactly the `source_sha` and artifact SHA-256 above, and
its `E2`-state tests authenticate the retained bytes, the unique
artifact-introducing commit, its direct `S2` parent, and the
Section 13.4 `E2` diff authority.
