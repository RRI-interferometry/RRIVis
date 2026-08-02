# Tier 8 Documentation and Release Reconciliation Plan

## 1. Identity, status, and governing sources

**Status: design authored 2026-08-02 on clean `main` at `95a937e`
(`docs(jones): accept Tier 7 integration`). Design accepted 2026-08-02 by
independent review, with eleven bounded citation/count corrections applied
(`docs(release): correct Tier 8 design`, `13b59f3`) and no decision changed;
see `Fix.md`'s 2026-08-02 "Tier 8 design acceptance" note for the full
record, including the CI-001 adjudication and the ruling that the red run at
`95a937e`/HEAD does not block this design-gate acceptance. Three further
corrections were applied at 8A (`docs(release): correct Tier 8 design`,
`41fabbb`): Section 14 items 2 and 3 (the instrumentation) moved from 8D to
8A, 8D item 4 narrowed to the CI re-run and measured decision, and Section
5.4's AGENTS.md defect count corrected to six. **8A is ACCEPTED** (2026-08-02
independent acceptance; see `Fix.md`'s "Tier 8A independent acceptance" note).
**8B is ACCEPTED** (2026-08-02 independent acceptance; see `Fix.md`'s "Tier 8B
independent acceptance" note). **8C is ACCEPTED** (first independent review of
`bd63f1c`/`8c30d37` REJECTED for a non-hermetic gitignore-guard test in
`tests/unit/test_tier8_release_acceptance.py`; repaired by `3c10f31` and
`f78c330`; 2026-08-02 independent re-acceptance verified both repairs from
scratch and accepted with bounded plan corrections to Sections 8 and 17; see
`Fix.md`'s "Tier 8C independent acceptance (re-run)" note) **and 8D is
authorized to begin.** Tier 7 is
accepted as a whole (Tier 7K, 2026-08-02); `SCI-001`, `SCI-002`, `SCI-003` are
`DONE`; `SCI-004`, `SCI-005` are filed `ROADMAP`; `SCI-006`, `SCI-007` stay
`OPEN`; `PERF-001` stays `ROADMAP`; `SKY-002` is `OPEN` and is **absorbed by
this tier** (Section 13); `CI-001` is `OPEN`, filed at 8A; `API-001` is
`OPEN`, filed at 8B independent acceptance (a `stokes_to_coherency` broadcast
ergonomics gap found while verifying 8B's docstring correction — no solver
path reachable, not a Tier 8 blocker). Tier 8 is the final tier of the
remediation program defined by `Fix.md`.**

This document is the governing implementation specification for Tier 8,
defined by [`Fix.md`](Fix.md) Section 17 ("Tier 8 — Documentation and release
reconciliation"). It closes, or explicitly re-dispositions, the eight register
rows routed to Tier 8 (`DOC-001` through `DOC-008`, `Fix.md:219-226`, detail in
`Fix.md` §7.9 at `Fix.md:861-881`), and it absorbs `SKY-002`
(`Fix.md:209`).

Governing sources, in precedence order:

1. `Fix.md` §4 (governing decisions: pre-v1 API policy `Fix.md:128-144`, the
   truthfulness rule `Fix.md:146-156`, explicit precedence, scientific features
   require scientific tests);
2. `Fix.md` §17 (`Fix.md:1586-1664`): the fifteen implementation items, the
   verification gate, the exit criteria;
3. `Fix.md` §18 rule 7 (`Fix.md:1679-1681`): "do not update final documentation
   early and then let later tiers invalidate it; update focused docs per tier
   and do the full sweep in Tier 8" — Tier 8 **is** that sweep;
4. `Fix.md` §21 (global definition of done) and §20 (test strategy);
5. `CLAUDE.md` at `95a937e` (project commands, Implementation Status,
   terminology split) — verified current, with three defects found (§8.4);
6. `AGENTS.md` and `docs/contributing.rst` (contributor surface).

Every characterization statement in Sections 5 through 7 is cited to a file and
a line true at `95a937e` unless explicitly marked as a CI-log or `gh` observation.

### 1.1 Baseline note — the commit this plan was authored against

`95a937e` changed only `Fix.md` (+328) and `Tier7JonesSciencePlan.md` (+48)
relative to `47df8fc`. No source, test, config, doc, or CI file differs between
the two commits. This matters for Section 6.6: `47df8fc`'s CI run was green on
all eight jobs and `95a937e`'s is red on one, with a byte-identical tree.

### 1.2 What this plan is not

- It is **not** a re-litigation of Tier 0–7 physics, config, or output
  decisions. Where a prior tier's accepted decision produces a documentation
  consequence, Tier 8 documents the decision; it does not revisit it.
- It is **not** a publishing act. No `git push`, no tag, no PyPI upload, no
  GitHub release, no readthedocs configuration change. A version *number* may
  change in tracked metadata (Section 16); nothing is published.
- It is **not** the owner of `PERF-001`, `SCI-004`, `SCI-005`, `SCI-006`, or
  `SCI-007`. Tier 8's obligation to those five rows is **disclosure only**
  (Section 9).
- It is **not** the owner of the root cause of the CI fingerprint divergence
  (Section 14). Tier 8 instruments and adjudicates it; it does not claim to
  explain it.

## 2. Design-only authority

This design gate may change exactly two files:

1. this file, `Tier8ReleasePlan.md` (new);
2. `Fix.md`, by **appending one `### 2026-08-02 Tier 8 documentation and
   release reconciliation design gate` status note** at the end of the file,
   in the established `### YYYY-MM-DD ...` form used by every prior gate
   (first such note: `Fix.md:1968`).

The design gate may **not**: add or edit register rows in `Fix.md` §5; write
acceptance records; touch `src/`, `tests/`, `docs/`, `configs/`, `examples/`,
`README.md`, `AGENTS.md`, `CLAUDE.md`, `pyproject.toml`, `pixi.toml`, or
`.github/`. Every such change is a slice's work under Section 17, performed by
an implementer and independently accepted.

Commit for this gate: `docs(release): plan Tier 8 reconciliation`. No
co-author lines. No branch. No push.

## 3. Tier 0–7 dependency and acceptance state

| Tier | State | What Tier 8 inherits |
|---|---|---|
| 0 | accepted | CI workflow, release-metadata test, pre-v1 contributor policy (`docs/contributing.rst:46-50`), strict-Pyright ceiling, Sphinx support matrix |
| 1 | accepted | one strict config contract; `configs/config.yaml` as the smoke sample |
| 2 | accepted | typed instrument sources; `generate_baselines` replaced by `generate_resolved_baselines`/`select_resolved_baselines` (`src/radiosim/core/baseline_resolution.py:113,283`) |
| 3 | accepted | canonical beam system; observability as a helper |
| 4 | accepted | canonical result, `scientific_sha256`/`provenance_sha256`, HDF5/summary/MS/UVFITS |
| 5 | accepted | receptor/basis contract; the `_iter_reference_scan_files` git-scoped scan helper this tier generalizes (`tests/unit/test_tier5_receptor_acceptance.py:132-160`) |
| 6 | accepted | backend parity, benchmark discipline and the committed reference records, hybrid sky; the per-`(platform, python)` fingerprint pin tables |
| 7 | accepted | 19-name Jones surface, `jones:` config section, corrected chain order, beam-physics scope document, pyuvsim crossval artifact |

Tier 8 depends on all seven. It is last precisely because §18 rule 7 forbids
doing this sweep earlier.

## 4. What Tier 8 will not claim

Stated before any work, so no slice can drift into it:

1. **No accelerator claim.** No GPU, TPU, or distributed number appears in any
   document Tier 8 writes. `PERF-001` remains `ROADMAP` and is disclosed.
2. **No m-mode claim.** `SCI-004` remains `ROADMAP`; `execution.simulator`
   accepts only `rime`.
3. **No advanced-beam-physics claim.** `SCI-005` remains `ROADMAP`; the scalar
   `E` boundary stated in `docs/development/beam_physics_scope.md` stands.
4. **No cross-validation-agreement claim beyond the recorded artifact.**
   `SCI-006` (Stokes-`Q` sign) and `SCI-007` (residual `-0.0576°` frame
   rotation) stay `OPEN` and must be named in the release notes, not omitted.
5. **No "validated against pyuvsim" phrasing** anywhere, per `CLAUDE.md`'s
   standing rule; only "compared against, with the following measured
   agreements and the following open disagreements."
6. **No claim that CI is green** unless a named run ID at a named SHA is green
   on all eight jobs at the moment of the claim.
7. **No claim that a documentation example works** unless a test or CI job
   executes it.

## 5. Current inventory — the true state of every Tier 8 surface

All facts below were established at `95a937e` by direct read, execution, or
`gh` inspection.

### 5.1 Examples surface

`examples/` contains `README.md` (51 lines), `scripts/simple_simulation.py`
(137 lines), `notebooks/01_basic_usage.ipynb`.

- `simple_simulation.py` **executes cleanly** and uses only public API. Every
  symbol it calls exists with a matching signature at HEAD:
  `Simulator.from_parameters` (`src/radiosim/api/simulator.py:264`),
  `Simulator.from_yaml` (`:202`), `SimulationOverrides`
  (`src/radiosim/io/config_resolution.py:397`),
  `simulator.get_memory_estimate()` (`src/radiosim/api/simulator.py:1598-1653`,
  returns a mapping; the script reads `estimate.get("total_human", ...)` and
  does **not** format it as a float), `result.visibilities.shape`,
  `result.correlations` (a tuple, `src/radiosim/core/result.py:1026`),
  `result.stokes_i()` (`:940`), `.scientific_sha256`/`.provenance_sha256`
  (`:1040-1041`). There is **no** `sim._sources` reference anywhere in the
  file. Its argparse defines exactly three options — `--config`, `--backend`,
  `--progress` (`examples/scripts/simple_simulation.py:14-37`) — and `--help`
  works.
- `examples/README.md` is **wrong about that script in two ways**:
  - it documents four flags that do not exist — `--no-plot`
    (`examples/README.md:13`, `:30`), `--save --output-dir`
    (`examples/README.md:19-20`), `--plot --output-dir`
    (`examples/README.md:22-23`). Each documented command fails with
    `error: unrecognized arguments`.
  - it names a removed backend: "JAX and Numba can be selected"
    (`examples/README.md:49`). The `numba` name was removed in Tier 6H;
    `get_backend("numba")` raises, and `README.md:351-352` already says so.
  - its "Shipped configurations" list (`examples/README.md:42-47`) names two of
    the four files in `configs/`.
- `01_basic_usage.ipynb` executes cell-by-cell against the current API; no
  mismatch found. It is executed by nothing.

### 5.2 README surface

`README.md` is 432 lines, ten `##` sections (`:12,55,80,191,314,346,382,401,412,430`).

- **Backend and performance section (`README.md:346-380`) is fully truthful.**
  Every numeric claim was re-verified against
  `output/benchmarks/reference/20260731T104303Z-darwin-arm64.json`: the commit
  (`ea48d2c`), the versions, "Dask bit-identical on all eight workloads"
  (all eight records show `max_absolute_deviation: 0.0`), "worst observed
  absolute deviation `1.7e-11` against an allowed `5.2e-9`" (record:
  `1.7280399333685637e-11` vs `5.230094962753472e-09`), "about 3x on a
  4096-source run" (0.1209 s / 0.0401 s = 3.01x), and `accelerator: "none"` in
  every record. `README.md:377` — "This repository publishes no unverified
  speedup multiplier and no GPU performance number" — holds.
- **Jones claims (`README.md:29-33,47-53`) are truthful** against the 19-name
  export list and the Tier 7 acceptance.
- **One stale count**: `README.md:408` says "[Three shipped YAML samples]"
  while `ls configs/*.yaml` returns **four** (`config.yaml`,
  `hybrid_sky_example.yaml`, `realistic_foreground_example.yaml`,
  `receptor_circular_example.yaml`).
- `generate_baselines` appears **zero** times in `README.md`, `docs/`,
  `examples/`, or `src/`.
- The `## Tests` section (`README.md:412-429`) lists five commands and two
  focused pytest invocations. It does not mention `tests/integration/`,
  `tests/characterization/`, `tests/crossvalidation/`, `pixi run bench`, or
  the `crossval` environment.
- `README.md` contains no repository-structure section, so it does not describe
  the 41 git submodules under `simulators/` (Section 5.7).

### 5.3 Sphinx surface

21 tracked pages. `docs/index.rst:10-46` carries the toctree. `docs/conf.py`
enables `autodoc`, `napoleon`, `viewcode`, `intersphinx`, `mathjax`,
`autosummary`, `myst_parser` (`docs/conf.py:20-28`). `nitpicky` is unset,
`suppress_warnings` is unset, `sphinx.ext.doctest` is **not** enabled, and
`myst_heading_anchors` is unset. `release`/`version` are `"0.2.0"`
(`docs/conf.py:15-16`).

**Warning baseline: 16, on a clean tracked worktree.** Verified by rebuilding
with `docs/superpowers/**` excluded (that directory is gitignored,
`.gitignore:211`, and its presence in a working tree inflates the count to 18 —
the same artifact 7H diagnosed and 7I ratified). The 16, grouped:

| # | Category | Origin | Fix cost |
|---|---|---|---|
| 10 | docutils docstring parse errors | `src/radiosim/backends/numpy_backend.py:251-273` (bare `*operands`/`*args` read as emphasis), `src/radiosim/backends/base.py:513` (`|x|` read as a substitution), `src/radiosim/backends/__init__.py` `get_backend` docstring (unexpected indentation / block-quote unindent), `src/radiosim/core/polarization.py:345-372` (five `|...|` substitutions in `jones_matrix_power`) | trivial docstring text |
| 1 | `toc.not_included` | `docs/HERA_VSIM_ANALYSIS.md` is tracked, in no toctree, and carries no `:orphan:` | trivial |
| 1 | unsupported theme option | `docs/conf.py:88` `"display_version": True` | trivial |
| 3 | `misc.highlighting_failure` | `docs/HERA_VSIM_ANALYSIS.md:243` (`csv` lexer unknown), `:364`, `:377` (data dumps annotated `python`) | trivial |
| 1 | `myst.xref_missing` | `docs/migration_guide.md:617` targets `#hybrid-results-and-serialization`; the heading exists at `docs/migration_guide.md:313` but `myst_heading_anchors` is unset so no anchor is generated | one `conf.py` line |

A `-W --keep-going` build today exits 1. Every one of the 16 fails it; none is
pre-suppressed. **None requires a change to Jones, beam, or solver logic** —
the ten source-side fixes are docstring prose only.

**Jones-name residue (`DOC-003`): none.** `src/radiosim/core/jones/__init__.py:95-128`
exports 19 names. Every `*Jones` token in `docs/` that is not in that list
appears in an explicitly historical context: `docs/api/jones.rst:27`
("`CrosshandPhaseJones` was renamed `CrosshandJones`"), `docs/changelog.rst:65-67`
(the `Removed` section), `docs/development/beam_physics_scope.md:186`
("was deleted in Tier 7C rather than implemented"), and the 26-row removal
table at `docs/migration_guide.md:722-750`. `GeometricDelayJones` — the name
`Fix.md:871-872` cites — has **zero** hits in `docs/`.

**API completeness gaps.** `docs/api/` covers `backends`, `benchmarks`, eight
`core/` modules, nine `io/` modules, sixteen `core/jones/` modules, and
`api.simulator.Simulator`. It has **no autodoc page** for:
`radiosim.core.sky` (the entire subpackage — `containers/`, `loaders/`,
`registry/`, `combine/`, `operations/`, `diagnostics/`, `recipes/`,
`support/`, `io/`); `radiosim.simulator` (`base.VisibilitySimulator`,
`rime.RIMESimulator`); `radiosim.utils`; `radiosim.visualization`;
`radiosim.core.observability`; `radiosim.core.result` (referenced only as
`:class:` text at `docs/api/simulator.rst:35` and `docs/api/io.rst:88,90`, so
`SimulationResult`'s own members are rendered nowhere); and, inside `core/`,
`visibility_healpix.py`, `contraction.py`, `hybrid.py`, `phase_center.py`,
`polarization_basis.py`, `precision.py`, `receptor.py`, `solver_partition.py`,
`time_grid.py`.

**Executable-example inventory.** 25 `.. code-block:: python` in `.rst`, 11
fenced `python` blocks in tracked `.md`, 2 `>>>` prompts in
`docs/contributing.rst`, and 299 `>>>` lines across 22 `src/radiosim`
docstrings. **Zero are executed.** `pyproject.toml:144` sets
`--doctest-modules`, but `pyproject.toml:137` sets `testpaths = ["tests"]`, so
collection never reaches `src/`; `pytest --collect-only` reports zero doctest
items. `grep -rn doctest tests/ .github/` finds nothing. The flag is dead
configuration.

### 5.4 Repo-root and agent-facing surface

**`CLAUDE.md` (216 lines, tracked, last rewritten in `53acb60`) is current on
every Tier 7 fact** — the Jones Implementation Status, the 19-name export, the
corrected chain order, `BeamSystem`, the removed `faraday.py`/`wterm.py`/
`element_beam.py`, the removed `visibility.calculation_type`, the backend and
benchmark honesty rules. Three defects:

- `CLAUDE.md:200` — "**Type checker**: MyPy (check_untyped_defs=true,
  ignore_missing_imports=true)". The repository uses **Pyright**:
  `pixi.toml:20` runs `python tools/check_pyright_baseline.py`, which invokes
  `python -m pyright --outputjson` (`tools/check_pyright_baseline.py:18-19`)
  against `[tool.pyright]` (`pyproject.toml:191-206`), with the version pinned
  `pyright = "==1.1.408"`. The string `mypy` appears in no build or tool file.
- `CLAUDE.md:181` — "`writers.py` / `readers.py` — HDF5/YAML simulation I/O".
  `src/radiosim/io/` has **no `writers.py`**; the writers are `hdf5.py`,
  `summary_json.py`, `standard_visibility.py`, `measurement_set.py`,
  `uvfits.py`, and the section omits `atomic_paths.py`, `jones_config.py`,
  `receptor_config.py`, `result_format.py`, `workflow_artifacts.py`.
- `CLAUDE.md:216` — a trailing `TODO:` asking for a contributor note on the
  pre-v1 policy. That note **exists**, at `docs/contributing.rst:46-50`
  ("Pre-v1 API Evolution Policy"), added in Tier 0. The TODO is discharged and
  stale.

**`AGENTS.md` (39 lines, tracked) has six live defects** (corrected from "five"
at 8A: the enumeration below has always listed six, and
`tests/characterization/test_tier8_current_behavior.py` pins all six):

- `AGENTS.md:4` — "`backends/` for NumPy/JAX/**Numba** execution". Removed in
  Tier 6H.
- `AGENTS.md:4` — "Tests are under `tests/` with `unit/`, `integration/`, and
  `performance/` splits". Also present: `characterization/`,
  `crossvalidation/`, `fixtures/`.
- `AGENTS.md:4` — "The Hugging Face app is isolated in `huggingface_space/`."
  The directory does not exist and is not tracked; it was removed in `3266746`
  ("chore: remove huggingface space and sync workflow"). **This is `DOC-007`,
  and it is the only live `huggingface` reference in the repository** — no
  other tracked file mentions `huggingface`, `gradio`, or a Space.
- `AGENTS.md:21` — "doctest collection enabled" and a marker list omitting
  `crossval` (`pyproject.toml:155-161` declares `slow`, `gpu`, `integration`,
  `performance`, `crossval`). Doctest collection is configured but inert (§5.3).
- `AGENTS.md:34` — "Until **RRIVis** reaches a major stable release such as
  `v1.0`" — the only live RRIVis naming in a tracked prose file.
- `AGENTS.md:39` — the same discharged `TODO:` as `CLAUDE.md:216`.

**`project.md` is gitignored** — `git check-ignore -v project.md` returns
`.gitignore:125`. It has never been tracked (`git log --all -- project.md` is
empty; the ignore entry landed in `3ce4523`). It is 1238 lines, titled
"RRIVis — Complete Project Documentation" (`project.md:1`), names the package
`rrivis` (`:31`), the repository `kartikmandar/RRIvis` (`:38`), the docs site
`rrivis.readthedocs.io` (`:39`), claims GPU acceleration (`:41,69,234,451,905,
916,933,1074,1114`), describes the removed `numba_backend.py` (`:70,132-133,
921-933`) and "46 exported classes" (`:41`). **It publishes nothing** — it is
not in the repository, not in the sdist, not on any docs site. Its only
audience is a person or agent reading this working tree.

### 5.5 CI and test surface

**`DOC-008` is materially discharged.** `.github/workflows/ci.yml` (127 lines)
is tracked, badge-linked (`README.md:5`), and demonstrably runs remotely.
Three jobs:

- `compatibility` (`ci.yml:16-68`) — six-cell matrix, linux-64 / osx-64 /
  osx-arm64 × Python 3.11 / 3.12, `pixi run --environment <env> test -- -m "not slow"`.
- `backend-parity` (`ci.yml:69-96`) — ubuntu, CPU-only-JAX assertion plus three
  focused backend paths.
- `quality` (`ci.yml:97-127`) — `pixi run lint`, `pixi run check-format`,
  `pytest tests/unit/test_release_metadata.py`, `pixi run typecheck`,
  `make -C docs html` (`ci.yml:127`).

Eight jobs total. Nothing is `continue-on-error`.

Test inventory (`--collect-only -q -n 0`: **5332 tests collected in 4.76 s**):

| Directory | Files | Character |
|---|---|---|
| `tests/unit/` | 154 | the bulk |
| `tests/characterization/` | 7 | golden output/fingerprint pins; **not** marked slow, so they gate |
| `tests/integration/` | 2 | real: `test_hybrid_end_to_end.py` (269 lines, `pytest.mark.integration`), `test_jones_end_to_end.py` (310 lines, one run+save per implemented Jones term). Unmarked slow, so they gate |
| `tests/performance/` | 1 | `test_backend_benchmarks.py`, `pytestmark = [performance, slow]` (`:73`); asserts record structure and honesty, not speed; never gates (`pixi.toml:24`) |
| `tests/crossvalidation/` | 1 | `test_pyuvsim_comparison.py`, `[crossval, slow]` (`:94`), `importorskip("pyuvsim")`; the `crossval` pixi environment is in no CI job |

Gaps against `Fix.md` §17: **no CI step executes `examples/scripts/simple_simulation.py`**
(item 2); **no notebook validation** exists anywhere; the docs build is not
`-W` (`docs/Makefile:5` has `SPHINXOPTS ?=` and `ci.yml:127` passes no
override) (item 11); **no doctest execution** (item 12); the integration tests
drive the Python `Simulator` API end to end but never the **CLI**
(`radiosim --config ...`) to an on-disk artifact, which is what
`Fix.md:1604` item 9 asks for.

### 5.6 The live CI red at HEAD

`gh run list` shows run **30726145633** at `95a937e` is **failure**, while
**30725507865** at `47df8fc` — a byte-identical source tree (§1.1) — is
success. Five failures, all on `linux-64 / Python 3.11` only:

```
tests/characterization/test_tier6_current_behavior.py::test_shipped_default_config_scientific_fingerprint
tests/characterization/test_tier6_current_behavior.py::test_shipped_circular_receptor_config_scientific_fingerprint
tests/characterization/test_tier6_current_behavior.py::test_section_13_4_workload_fingerprints[heterogeneous_receptor_bases]
tests/characterization/test_tier7_current_behavior.py::test_shipped_default_config_fingerprint_is_unchanged
tests/characterization/test_tier7_current_behavior.py::test_shipped_circular_receptor_config_fingerprint_is_unchanged
```

each with `Failed: ... digest not among those recorded for environment linux-64-py311.`
(assertion text at `tests/characterization/test_tier6_current_behavior.py:462-467`).
`test_tier7_current_behavior.py:213-216` imports the Tier 6 tables, so the five
failures are three distinct measurements.

Established facts, from the CI logs and `git log -S`:

- The environment key is `f"{platform}-{py}"` only
  (`test_tier6_current_behavior.py:385-402`) — it encodes no CPU or thread fact.
- `scientific_sha256` hashes the **raw little-endian bytes** of
  `visibilities`, `flags`, `weights`, the time grid, frequencies and channel
  widths (`src/radiosim/core/result.py:467-475,789-841`), so a 1-ULP change and
  a 100% change are indistinguishable to the gate. The failing logs show the
  **raw cube digest moved too**, so this is a numbers change, not metadata.
- The five measured digests are **byte-stable across three CI runs, two CPU
  vendors and three CPU models** (Intel 8573C, Intel 6973P-C, AMD EPYC 9V74) —
  a reproducible second class, not a race.
- The explanation the module's own prose gives (dispatched vector feature set,
  `test_tier6_current_behavior.py:226-246`) is **falsified**: the originally
  recorded `linux-64-py311` value was measured on an AMD EPYC 9V74, the same
  part that now produces the new class; and within the new class, jobs with
  different `numpy.__cpu_features__` produce identical digests.
- Ruled out with evidence: source regression; xdist presence, worker count, and
  test ordering; numpy/astropy/OpenBLAS drift (`locked: true` in CI, identical
  `astropy-iers-data`, `libopenblas`, `libblas` in fail and pass logs); astropy
  IERS auto-download (three local IERS configurations produce one digest);
  `PYTHONHASHSEED`; thread counts (locally, on `osx-arm64`); uninitialized
  memory in the default chain.
- **The evidence gap is structural**: `_machine_fingerprint()`
  (`test_tier6_current_behavior.py:413-437`) prints **only on failure** and was
  added after the green baseline was harvested, so nothing is recorded about
  any passing `linux-64-py311` runner.
- The established response to a disagreeing runner has been to **append another
  accepted digest**: `e3f1987`, `1c90d81`, `e5b20d1`, `0ce72e4`, four commits in
  four days. Of the last 25 CI runs, 11 failed and **all 11 are this same pin
  family**. On `linux-64-py311` the new class appears in 3 of the 8 runs since
  it first appeared (~38%).

### 5.7 Repository-scan test sites (`Fix.md` §17 item 15)

`git grep -n rglob -- tests` finds **22 call sites across 13 files**. The
hardened reference is `tests/unit/test_tier5_receptor_acceptance.py:132-160`,
which shells out to `git ls-files --cached --others --exclude-standard -z`
scoped to the scan roots, filters by suffix, and drops `__pycache__`.

**`Fix.md:1622-1632`'s list is over-inclusive by two files.** Re-verified:

- `tests/unit/test_io/test_output_atomicity.py:330` rglobs `path`, which is a
  `tmp_path`-derived Measurement Set directory inside the test
  (`:320-333`) — **not a repository scan, not vulnerable**.
- `tests/unit/test_visualization/test_result_plots.py:247` rglobs
  `output = tmp_path / "plots"` (`:242`) — **not a repository scan, not
  vulnerable**.

The true hardening set is **20 call sites in 11 files** (Section 12 gives the
exact table). The demonstrated failure mode is real but latent: no
`.ipynb_checkpoints` exists in the tree today, and the only gitignored files
under `src/` are `src/radiosim.egg-info/*` (no `.py`), so all 20 currently
pass. The vulnerability is that any gitignored `*.py` under `src/` — a
notebook checkpoint, an editor backup, a stale build copy — turns a source
scan into a false failure.

### 5.8 Changelog, migration guide, and release metadata

- `docs/changelog.rst` (181 lines): `[Unreleased]` (`:6-90`, Tier 7),
  `[0.2.0] - 2025-12-15` (`:92-176`), `[0.1.x]` (`:163-176`). The `[0.2.0]`
  entry is preserved verbatim as history with a corrective `.. note::`
  immediately above it (`:97-108`) retracting its "Universal GPU acceleration"
  and "Complete 8-term Jones chain" claims — the treatment 7J ruled sufficient.
  Tier 5, 6 and 7 breaking changes are all present.
- `docs/migration_guide.md` (792 lines, 11 sections) is comprehensive through
  Tier 7, including the 26-class removal table (`:722-750`).
- **Version is `0.2.0` in five places and their agreement is tested**:
  `pyproject.toml:7`, `pixi.toml:3`, `src/radiosim/__about__.py:3`,
  `docs/conf.py:15-16`, and the installed `radiosim.__version__`, enforced by
  `tests/unit/test_release_metadata.py::test_release_metadata_matches_canonical_project_version`
  (`:89-105`), which `ci.yml:118-119` runs explicitly.
- **The package version feeds `provenance_sha256` only, never
  `scientific_sha256`**: `_hash_json(digest, "package_version", _package_version())`
  is at `src/radiosim/core/result.py:857`, inside `_provenance_hash`
  (`:844-859`), not inside `_scientific_hash` (`:789-841`). No test pins a
  literal `provenance_sha256`. A version bump therefore cannot move any
  characterization fingerprint pin.
- **Stale packaging truth**: `pyproject.toml` ships `gpu` (`:61`),
  `gpu-cuda` (`:67`), `gpu-rocm` (`:72`) and `tpu` (`:77`) optional-dependency
  extras and a `"gpu"` keyword (`:20`), advertising accelerator support the
  project explicitly does not claim (`PERF-001`, `README.md:373-374`).
- **Undisclosed repository structure**: `simulators/` holds **41 git
  submodules** (`git submodule status | wc -l` → 41) of third-party simulators,
  ~3.9 GB when checked out, tracked as gitlinks with a `.gitmodules`, excluded
  from Ruff (`pyproject.toml:167`) and from the wheel
  (`[tool.setuptools.packages.find] where = ["src"]`, `pyproject.toml:128-129`).
  They are named in **no** tracked prose file — not `README.md`, not
  `AGENTS.md`, not `CLAUDE.md`, not `docs/contributing.rst`. A contributor
  cloning with `--recursive` gets 3.9 GB with no warning.
- **One binary stale-naming hit**:
  `antenna_layout_examples/1101503312_metafits.fits` contains the FITS card
  `COMMENT Example MWA metafits file for RRIVis testing`. The file is the
  shipped example for the `mwa_metafits` format
  (`antenna_layout_examples/README_antenna_formats.md:17`) and is referenced by
  no test.

## 6. Per-`DOC` row current-state verdicts

This is the table `Fix.md` §7.9 must be re-read against. "Discharged" means the
finding is not reproducible at `95a937e`; the row still closes at Tier 8's
whole-tier gate, but on verified prior-tier evidence rather than new work.

| Row | `Fix.md` text | Verdict at `95a937e` | Evidence | Tier 8 work |
|---|---|---|---|---|
| `DOC-001` | `simple_simulation.py` uses stale private/result APIs | **Discharged** for the script; **live** for its README | script re-verified symbol by symbol and executed (§5.1); no `sim._sources`, no dict-as-float, no dict-as-array. But `examples/README.md:13,19-20,22-23,30` documents four nonexistent flags and `:49` names the removed `numba` backend | 8B: correct `examples/README.md`; add the flag-parity test; execute the example in CI |
| `DOC-002` | README low-level baseline example is invalid | **Fully discharged** (Tier 2) | `generate_baselines` has zero occurrences in the repository; replaced by `generate_resolved_baselines`/`select_resolved_baselines` (`src/radiosim/core/baseline_resolution.py:113,283`). Every README and quickstart code block was executed successfully | 8F: close on evidence; 8B adds the residual scan that keeps it closed |
| `DOC-003` | Sphinx references removed Jones class names | **Fully discharged** (Tier 7J, `53acb60`) | `GeometricDelayJones`: zero hits in `docs/`. All out-of-`__all__` `*Jones` tokens are in captioned historical contexts (§5.3) | 8F: close on evidence; 8C adds the residual scan |
| `DOC-004` | README claims 15+ configs while two exist | **Partially discharged; still stale** | the "15+" claim is gone; `README.md:408` now says "Three shipped YAML samples" and `configs/` holds **four**. `examples/README.md:42-47` names two of four | 8B/8E: make both counts derived, not asserted, and test them |
| `DOC-005` | README/backend documentation contradicts live backend behavior | **Discharged for `README.md`** (Tier 6I, `eea1914`); **live in two other files** | `README.md:346-380` verified claim-by-claim against the committed benchmark record (§5.2). But `examples/README.md:49` still offers "Numba", and `pyproject.toml:20,61,67,72,77` still ships `gpu`/`gpu-cuda`/`gpu-rocm`/`tpu` extras | 8B (`examples/README.md`), 8E (packaging extras) |
| `DOC-006` | `project.md` is stale and still describes RRIVis | **Live as a file, but it is gitignored and has never been tracked** | `git check-ignore -v project.md` → `.gitignore:125`; `git log --all -- project.md` empty; content confirmed stale (§5.4) | 8E: Decision 7 (Section 15) — the tracked surface gains an explicit statement; the untracked file's fate is gated question Q2 |
| `DOC-007` | `AGENTS.md` describes an absent Hugging Face app | **Live**, and it is the sole surviving reference | `AGENTS.md:4`; directory removed in `3266746`; no other tracked file mentions huggingface/gradio/Spaces | 8E: remove the sentence; no restoration (Section 15.2) |
| `DOC-008` | No tracked CI and no real integration/performance suites | **Materially discharged; three named gaps remain** | `.github/workflows/ci.yml` is tracked, badge-linked, and runs remotely on eight jobs; `tests/integration/` (2 real files, 16 tests) and `tests/performance/` (10 record-honesty tests) are real. Gaps: no CLI-to-artifact integration test, no example execution, no notebook validation, no `-W` docs gate, no doctest execution — and CI is **red at HEAD** (§5.6) | 8D: CI shape; 8B/8C: the executed surfaces; `CI-001` filed for the red |

**Net**: of eight rows, two are fully discharged by prior tiers (`DOC-002`,
`DOC-003`), three are partially discharged with a precisely bounded residue
(`DOC-001`, `DOC-004`, `DOC-005`), one is materially discharged with three
named gaps (`DOC-008`), and two are live and small (`DOC-006`, `DOC-007`).
Tier 8 is therefore **narrow in defect count and wide in surface** — its real
weight is in the enforcement mechanisms (Sections 8–12), not in the fixes.

## 7. Design decision 1 — documentation gets the same four-state discipline as code

`Fix.md` §4.2 defines four states for a field or public class. Tier 8 extends
it verbatim to every documented statement. A sentence in a tracked document may
be in exactly one of four states:

1. **Executed** — a test or CI job runs it and fails when it stops working
   (code examples, CLI invocations, flag names);
2. **Scanned** — a residual test asserts a mechanical property of it (a name is
   absent, a count matches the filesystem, a claimed file exists);
3. **Cited** — it carries a file, line, commit, run ID, or record path a reader
   can check by hand (benchmark numbers, cross-validation agreements, register
   dispositions);
4. **Absent** — it is not written.

There is no fifth state. A claim that is none of these four is deleted, not
softened. This is the rule that makes Tier 8 terminal: after it, drift is
detected by the suite instead of by the next review.

**Consequence for §17's exit criterion "README contains no unsupported
claims"**: unsupported is defined as "in none of states 1–3", and the
whole-tier gate proves it by enumeration (Section 18), not by assertion.

## 8. Design decision 2 — Sphinx warnings become errors, after the 16 are fixed

**Decision.** `-W --keep-going` becomes the default and the gate, in this
order within slice 8C: (a) fix all 16 warnings at their source; (b) change
`docs/Makefile`'s `SPHINXOPTS ?=` default to include `-W --keep-going`;
(c) leave `ci.yml:127`'s `make -C docs html` unchanged, so it inherits the
gate. `suppress_warnings` stays **unset** and `nitpicky` stays **off**.

**Why fix rather than suppress.** All ten source-side warnings are docstring
prose defects that make the rendered API documentation wrong — `|x|` renders as
a broken substitution, `*operands` renders as stray emphasis. Suppressing them
would leave the rendered page wrong and the gate green, which is precisely the
failure mode §4.2 forbids.

**Why `--keep-going`.** Without it the build stops at the first error and a
contributor fixes warnings one rebuild at a time. `Fix.md:1644` already
specifies both flags.

**Why not `nitpicky`.** `nitpicky` promotes every unresolved cross-reference,
including references into third-party inventories that `intersphinx` may or may
not resolve depending on network availability. That would make the docs gate
network-sensitive. Out of scope; not a Tier 8 claim.

**Correction, applied at the Tier 8C independent re-acceptance (rejection
repair round).** The paragraph above frames network-sensitivity as a
`nitpicky`-specific risk, but `intersphinx`'s inventory fetch is unconditional
and independent of `nitpicky`: reproduced by blocking the network
(`http_proxy`/`https_proxy` pointed at an unreachable address) and rebuilding
in a fresh detached worktree at `f78c330` — `make -C docs clean html` exits 2
with "build finished with problems, 5 warnings (with warnings treated as
errors)", one `WARNING: failed to reach any of the inventories` per mapped
project (`python`, `numpy`, `astropy`, `scipy`, `jax`). So the `-W` gate *is*
network-sensitive today, with or without `nitpicky`. **Ruling: acceptable,
recorded residue, not a defect requiring a fix.** The gating environment is
GitHub Actions (`ci.yml`'s `quality` job runs `make -C docs html` on a
network-connected runner), so this never bites the actual gate; a contributor
building docs in a network-blocked sandbox already has the documented escape
hatch (`docs/Makefile`'s comment: "Override on the command line (`make -C docs
html SPHINXOPTS=`) only to inspect a broken build; never to land one"), which
is exactly this situation. Vendoring the five inventories was considered and
rejected as disproportionate: it adds a maintenance burden (five files to
refresh against upstream drift) to close a gap that only affects local
inspection builds in an already-rare network-blocked environment, never the
gate that actually ships. No plan or code change follows from this beyond this
paragraph recording the ruling.

**The one non-trivial fix.** `docs/migration_guide.md:617` needs
`myst_heading_anchors` set in `docs/conf.py` (the heading it targets exists at
`:313`; MyST simply generates no anchors today). Setting it generates anchors
for **every** MyST heading, which can introduce duplicate-anchor warnings in
`migration_guide.md` and `HERA_VSIM_ANALYSIS.md`. 8C must rebuild after setting
it and resolve any new warning before (b) lands. If the anchor generation
proves to introduce more warnings than it fixes, the fallback is an explicit
MyST target on that one heading and leaving `myst_heading_anchors` unset —
the implementer chooses on measured evidence and records which.

**Baseline discipline.** The gate is 16 → 0 measured in a **clean detached
worktree**, per the 7I ruling. An in-tree build that reports 18 because
`docs/superpowers/` exists locally is not a regression and must not be treated
as one; 8A records this explicitly so no later slice re-litigates it.

## 9. Design decision 3 — documentation examples are executed, not asserted

**Decision.** Tier 8 makes three example surfaces executable and one
verifiable:

1. **The script.** `examples/scripts/simple_simulation.py` gains a CI step
   (`Fix.md` §17 item 2). It already runs offline in ~1 s with no artifacts,
   so the step is `pixi run python examples/scripts/simple_simulation.py`
   followed by `--help`. It goes in the `quality` job, not the six-cell matrix,
   to keep matrix time flat.
2. **The notebook.** `examples/notebooks/01_basic_usage.ipynb` gains an
   executed check via `jupyter nbconvert --to notebook --execute --stdout`
   (or `nbclient`) in the same job, gated on the dependency already being in
   the default pixi environment. **If it is not, the notebook check is not
   invented**: 8B either adds the dependency to the `default` feature or, if
   that inflates the environment materially, records the notebook as
   state 2 (scanned: every symbol it calls exists) rather than state 1, and
   says so in the plan's acceptance record. `Fix.md:1647-1648` explicitly
   defers this to "the repository's chosen notebook command", so choosing it is
   Tier 8's call, and choosing *not* to execute it is a legitimate, disclosable
   outcome — silently claiming it is executed is not.
3. **Docstring doctests.** `--doctest-modules` at `pyproject.toml:144` is
   currently inert. **Decision: make it real, scoped to `src/`, as a separate
   invocation rather than by widening `testpaths`.** Widening `testpaths` to
   include `src` would change what a bare `pixi run test` collects for every
   contributor and every CI cell, and would put 299 previously unexecuted
   `>>>` lines into the gate all at once. Instead: add a `pixi` task
   `doctest = "python -m pytest --doctest-modules src/radiosim -p no:cacheprovider"`
   and a `quality`-job step that runs it, and remove `--doctest-modules` from
   the shared `addopts` so the dead flag stops implying coverage it does not
   provide. 8B must expect real failures on first run and fix the docstrings —
   that is the point of the item, and the size of that work is gated question
   Q4.
4. **`docs/` prose code blocks.** The 25 `.rst` and 11 `.md` python blocks are
   **not** turned into doctests. They are put in state 2 by the residual scan
   (Section 11): every dotted symbol they reference must be importable and
   every attribute must exist. Converting prose blocks to executable doctests
   would require pinning outputs of simulations that legitimately differ across
   platforms — the exact trap §5.6 documents.

**Rule that keeps `examples/README.md` from drifting again.** 8B adds a test
that extracts every `--flag` token from `examples/README.md` and asserts it
appears in the script's `--help` output, and vice versa for every flag the
parser defines. This converts a prose claim into state 1.

## 10. Design decision 4 — the examples/README correction is to the document, not the script

**Decision.** `examples/README.md` is corrected to describe the three flags
that exist (`--config`, `--backend`, `--progress`); the script does **not**
grow `--no-plot`, `--save`, `--plot`, `--output-dir`.

**Why.** §4.1 prefers the coherent replacement, and §4.2's state 4 ("absent
from the public surface") is a legitimate state. The script's stated contract
is a deterministic offline smoke run that "writes no output artifacts"
(`examples/scripts/simple_simulation.py:3-5`); saving and plotting are already
demonstrated by `docs/quickstart.rst:36-58`, which is executed prose covering
`Simulator.save` and `Simulator.plot` with their real signatures. Adding four
flags to satisfy a stale README would grow the public example surface to match
a document rather than a need, and would add file-writing behavior to the one
example that is safe to run in CI unconditionally.

**Correction (applied at 8B).** §5.1's "`simple_simulation.py` **executes
cleanly**" is true of the default path and **false of `--config`**. `main()`
asserts `result.visibilities.shape == (1, 15, 2, 4)`
(`examples/scripts/simple_simulation.py:125`), which are the *built-in*
example's dimensions; every shipped document is larger
(`configs/config.yaml` yields `(60, 15, 101, 4)`), so
`--config configs/config.yaml` — a command the stale README itself printed —
exits with `AssertionError`. A flag that cannot be exercised is the same
`DOC-001` defect class 8B exists to close, and `examples/README.md` cannot
honestly print a command for it while it holds.

This decision is therefore **narrowed, not reversed**: the script still does
not grow `--no-plot`, `--save`, `--plot` or `--output-dir`, still writes no
artifacts, and still defines exactly three flags — the 8A pin
`test_example_script_defines_exactly_three_flags` stays green. 8B is granted
`examples/scripts/simple_simulation.py` **for the single change of scoping
that dimension assertion to the built-in path** (`if args.config is None:`),
and for nothing else. Section 10's stated reasons — not growing the public
surface, and not adding file-writing behavior — are untouched by that change.

## 11. Design decision 5 — one residual-scan contract for the whole tier

**Decision.** Tier 8 adds a single acceptance module,
`tests/unit/test_tier8_release_acceptance.py`, that owns every state-2 scan,
built on the shared git-scoped file lister from Section 12. It asserts, over
the tracked-plus-unignored file set:

1. **No removed name is documented as live.** For each removed symbol (the 26
   Jones classes at `docs/migration_guide.md:722-750`, `GeometricDelayJones`,
   `generate_baselines`, `numba`/`NumbaBackend`, `visibility.calculation_type`,
   `jones_config`, `combine_models`, `source_format`, `available_formats`),
   any occurrence must be inside an allow-listed historical file
   (`docs/changelog.rst`, `docs/migration_guide.md`,
   `docs/development/beam_physics_scope.md`, `docs/HERA_VSIM_ANALYSIS.md`,
   `Fix.md`, `Tier*Plan.md`, and this acceptance module itself) — the same
   `ALLOWED_REFERENCES` shape Tier 5 already uses
   (`tests/unit/test_tier5_receptor_acceptance.py:104-122`).
2. **No stale project naming.** `RRIVis`/`rrivis`/`RRIvis` (case-insensitive)
   appears in no tracked file outside the historical allow-list. The current
   tracked hits are `AGENTS.md:34` (to be fixed),
   `Tier3BeamObservabilityPlan.md:3223,3486` and `Fix.md` (historical records —
   allow-listed, **never** edited), and the FITS COMMENT card in
   `antenna_layout_examples/1101503312_metafits.fits` (to be fixed,
   Section 15.4).
3. **Counts are derived, not asserted.** The number of `configs/*.yaml` files
   stated in `README.md` and `examples/README.md` must equal
   `len(sorted(Path("configs").glob("*.yaml")))`, and every config named in
   either document must exist. The test parses the count from the prose so the
   prose cannot drift; if a future author prefers not to state a number, the
   test accepts the absence of a numeral but still requires every named file to
   exist.
4. **Every documented path exists.** Every relative path in a Markdown or
   reStructuredText link in `README.md`, `AGENTS.md`, `CLAUDE.md`,
   `examples/README.md`, and `docs/**` resolves to a tracked file or directory.
5. **Every documented symbol exists.** Every `radiosim.`-rooted dotted name in
   a `.. code-block:: python`, a fenced ```python block, or an
   `:class:`/`:func:`/`:meth:` role in `docs/**`, `README.md`, and
   `examples/README.md` is importable and, for attribute paths, present.
6. **Flag parity** for `examples/README.md` ↔ the script's `--help`
   (Section 9).
7. **No GPU/speed claim without a citation.** Any tracked prose line matching
   `gpu|tpu|accelerat|speedup|faster|x speed` outside the historical allow-list
   must, within its enclosing paragraph, cite `output/benchmarks/reference/` or
   name `PERF-001`. This is the enforceable form of `CLAUDE.md`'s standing rule.
8. **Documented commands exist.** Every `pixi run <task>` string in tracked
   prose names a task defined in `pixi.toml`.

Each assertion carries an actionable message naming the offending file, line,
and the exact replacement rule — the discipline every prior tier's acceptance
module uses.

**Why one module and not eight.** These scans share the file lister, the
allow-list vocabulary, and the failure-message style; splitting them across
slices would duplicate all three and make the closure evidence for
`DOC-001..008` harder to read at the whole-tier gate.

## 12. Design decision 6 — the shared git-scoped scan helper (item 15)

**Decision.** Extract `tests/unit/test_tier5_receptor_acceptance.py:132-160`'s
`_iter_reference_scan_files` into a shared helper module,
`tests/support/repo_scan.py` (new package `tests/support/` with an
`__init__.py`), exposing exactly:

```python
def iter_tracked_files(*roots: Path, suffixes: frozenset[str] | None = None) -> list[Path]
def iter_package_sources() -> list[Path]        # src/radiosim/**/*.py, git-scoped
def iter_repository_python() -> list[Path]      # repo-wide *.py, git-scoped
```

backed by one `git ls-files --cached --others --exclude-standard -z --` call,
filtered by suffix, with `__pycache__` dropped, returning sorted absolute
paths. It raises a typed error if `git` is unavailable rather than silently
falling back to `rglob` — a silent fallback would reintroduce exactly the
pollution the helper exists to prevent.

**The exact conversion set — 20 call sites in 11 files:**

| File | Lines | Scan root | Convert |
|---|---|---|---|
| `tests/unit/test_tier5_receptor_acceptance.py` | 129 | `PACKAGE_ROOT` | yes (`_iter_package_sources`); `:132-160` becomes a thin call into the helper |
| `tests/unit/test_tier4_result_output_acceptance.py` | 109 | `PACKAGE_ROOT` | yes |
| `tests/unit/test_tier7_jones_acceptance.py` | 149, 266, 355 | `SOURCE_ROOT`, `JONES_ROOT/beam`, `JONES_ROOT` | yes (all three — 149 is the site the checkpoint file broke) |
| `tests/unit/test_backends/test_compilation_boundary.py` | 169 | `source_root` | yes |
| `tests/unit/test_core/test_cleanup_registry.py` | 106 | `src_root` | yes |
| `tests/unit/test_core/test_sky_no_dataclasses_replace.py` | 27 | `pkg_root` | yes |
| `tests/unit/test_core/test_tier2_instrument_cleanup.py` | 146 | `source_root` | yes |
| `tests/unit/test_core/test_tier3_beam_cleanup.py` | 309 | `core/jones` | yes |
| `tests/characterization/test_tier5_current_behavior.py` | 697, 794, 837, 846 | `SOURCE_ROOT/radiosim` | yes (4) |
| `tests/characterization/test_tier6_current_behavior.py` | 1217, 1441 | `src/radiosim` | yes (2) |
| `tests/characterization/test_tier7_current_behavior.py` | 597, 626, 637, 1567 | `SOURCE_ROOT`, `JONES_ROOT/beam` | yes (4) |

**Explicitly excluded, with reason** (a correction to `Fix.md:1622-1632`):

| File | Line | Why not |
|---|---|---|
| `tests/unit/test_io/test_output_atomicity.py` | 330 | rglobs a `tmp_path`-derived `.ms` directory created by the test (`:320-333`); no repository file can enter it |
| `tests/unit/test_visualization/test_result_plots.py` | 247 | rglobs `tmp_path / "plots"` (`:242`); same reason |

Converting these two would make tests depend on `git` for no benefit and would
break if a future test scanned a directory outside the repository.

**Regression proof.** 8D adds a test that creates a gitignored
`src/radiosim/.ipynb_checkpoints/<name>-checkpoint.py` containing both a
removed Jones class name and a stub marker, runs the two
`tests/unit/test_tier7_jones_acceptance.py` tests the acceptance demonstrated
were vulnerable, asserts they pass, and removes the file — the exact scenario
`Fix.md:1616-1622` records. It must be written so a failure cannot leave the
file behind (`try/finally` or a `tmp`-symlinked fixture), because a leftover
would poison every subsequent scan in the session.

## 13. Design decision 7 — `SKY-002` is absorbed by Tier 8

**Decision. Tier 8 absorbs `SKY-002` and closes it in slice 8D.**

**Why absorb rather than leave standalone.** `SKY-002` (`Fix.md:209`) is not a
sky-model defect; it is a **truthfulness defect of exactly the class §4.2
names**: `Simulator`'s pre-flight prints "Network: offline (no
network-dependent models)" (`src/radiosim/api/simulator.py:773`) for a config
that will then make two real network calls. Its blast radius is a **shipped
example config** — `configs/realistic_foreground_example.yaml` — which Tier 8
must document truthfully in `examples/README.md:45-47` and `README.md`
regardless. Documenting "this config may require network access" while the
program's own pre-flight says the opposite is precisely the drift Tier 8
exists to end. It is also bounded, offline-testable, and touches no solver.

**What is actually wrong.** `loader_registry.register_loader("realistic_foreground", ...)`
(`src/radiosim/core/sky/recipes/realistic_foreground.py:277-297`) passes no
`network_service`. `LoaderDefinition.network_service` is a **singular**
`str | None` (`src/radiosim/core/sky/registry/core.py:202`, mirrored at
`facade.py:49` and `catalogs.py:87,127,195,280`), so the recipe cannot declare
its two services — it internally calls `_load_diffuse` (pygdsm →
`pygdsm_data`, `catalogs.py:473`) and `_load_bright_catalog` (VizieR)
(`recipes/realistic_foreground.py:390,410`).
`LoaderRegistry.network_services` (`facade.py:136-141`) and
`get_required_services` (`src/radiosim/utils/network.py:335-365`) therefore
return `{}`.

**The fix, specified.** Widen the declaration from one service to a tuple:
`network_services: tuple[str, ...] = ()` on `LoaderDefinition`
(`registry/core.py:202`) and every mirror, with the registration keyword
renamed to match; per §4.1 this is a **breaking rename with no compatibility
shim** — `network_service` is removed, not aliased, and every existing
single-service registration becomes a one-element tuple. `network_services`
(the registry property) returns `dict[str, tuple[str, ...]]`,
`get_required_services` unions them, and `realistic_foreground` declares
`("pygdsm_data", "vizier")`. The exact VizieR service token must be read from
`catalogs.py` at implementation, not guessed, so the recipe's declaration is
identical to what `gleam` already declares.

**Tests (offline, no network).** (a) `get_required_services` on the shipped
`configs/realistic_foreground_example.yaml` returns both services; (b) the
pre-flight line for that config is the network-dependent branch, not
`"no network-dependent models"`; (c) a registry-completeness test asserting
that **every** loader whose module imports a network client declares at least
one service — the generalization that stops the next composite recipe from
repeating this; (d) an offline run of that config fails with the actionable
offline error rather than a network attempt.

**Register consequence, executed by the implementer at 8D acceptance**:
`SKY-002` flips `OPEN` → `DONE` with the closure evidence above, its tier
column changing from "pre-Tier-8" to `8`.

## 14. Design decision 8 — the CI fingerprint divergence gets instrumentation and a new row, not a reflex append

**Decision.** Tier 8 does **three** things and refuses a fourth:

1. **Files a new register row `CI-001`** (`OPEN`), owned by the implementer at
   8A acceptance, worded from §5.6's evidence: a second byte-stable
   `scientific_sha256`/cube-digest class on `linux-64-py311` whose discriminator
   is **unidentified**, with the module's own stated discriminator (dispatched
   vector feature set) falsified by the logs, appearing in ~38% of runs on that
   cell, and blocking any "CI is green" claim.
2. **Makes the evidence recoverable.** `_machine_fingerprint()`
   (`tests/characterization/test_tier6_current_behavior.py:413-437`) emits
   **unconditionally** — once per session, on pass as well as fail — so every
   future run records its CPU model, `numpy.__cpu_features__`, thread
   environment, and BLAS build. Today there is literally no record of what any
   green `linux-64-py311` runner was, which is why this divergence is
   undiagnosable. This is the single highest-value bounded act available.
3. **Makes the divergence adjudicable.** The pin failure path gains a
   **numeric** delta: when a recorded reference cube is available the test
   reports `max|ΔV|`, `max relative Δ`, and the index of the first differing
   element alongside the hex digests. The gate today cannot distinguish 1 ULP
   from 100%, and no failing log in the last 25 runs contains a single number.

**What Tier 8 refuses.** It does **not** append the fifth accepted digest class
on the strength of "it reproduces". Four prior commits (`e3f1987`, `1c90d81`,
`e5b20d1`, `0ce72e4`) appended on disagreement, and the justification they
appealed to is now known to be wrong for this observation. Appending a fifth
time under a falsified rationale would violate the module's own written rule
("A set never grows to make a failure go away",
`test_tier6_current_behavior.py:271-273`) and §4.2.

**The conditional that unblocks CI.** Within 8D, after (2) and (3) land, CI is
re-run on the resulting SHA. If the numeric probe shows the divergence is at
ULP scale — concretely, within the Section 13.5 backend tolerance the project
already uses (`rtol=1e-12`) — then appending the observed digests **is**
justified, the justification is "a second reproducible class at ULP scale,
discriminator unidentified, numeric delta recorded here", and `CI-001` narrows
to "the discriminator is unnamed" rather than closing. If the delta is larger
than that, nothing is appended, `CI-001` stays wide, and Tier 8's release notes
disclose a known-red CI leg. **Which of these two the tier lands on is
determined by measurement, not by preference** — see gated question Q3 for the
one thing the user must decide.

**Root cause is explicitly not Tier 8's.** Naming the discriminator needs
runner access or instrumented dumps of intermediate quantities from both
classes; the hypothesis space still includes hypervisor CPU-feature masking and
`libm`/OpenBLAS runtime dispatch, neither of which current instrumentation
captures. `CI-001` carries it forward, in the `PERF-001`/`SCI-006` tradition of
filing rather than absorbing.

**The deeper question is named and deferred.** Whether a bitwise digest is the
right cross-platform gate at all — versus pinning a reference cube and
asserting the tolerance, with the digest kept as advisory — is a real design
question that changes what the gate *means*. It is recorded in `CI-001` as the
successor decision and is **not** made by Tier 8, because making it would
weaken a reproducibility gate at the exact moment the project is reconciling
its truth claims, on evidence that does not yet distinguish "harmless
last-bit dispatch" from "a real numerical difference between platforms".

## 15. Design decision 9 — the four disposal questions

### 15.1 `project.md` (`Fix.md` §17 item 7)

**Decision: Tier 8 does not restore, rewrite, or track `project.md`, and it
does not delete it from the user's working tree either. It makes the tracked
surface state the truth instead.**

**Why.** `project.md` is gitignored (`.gitignore:125`) and has never been
tracked. It is not in the repository, the sdist, the wheel, or any docs site —
so it makes **no public claim**, and `DOC-006`'s premise ("stale documentation
the project ships") is false at `95a937e`. Rewriting 1238 lines of RRIVis-era
prose to produce a second, untracked, unmaintained architecture document
alongside `CLAUDE.md`, `README.md`, and `docs/` would create exactly the
duplicate-truth surface this program spent eight tiers removing. Deleting a
file from the user's working tree is not a repository change and is not a
design gate's or an implementer's call to make unasked.

**What 8E does instead**: adds one line to `.gitignore` near `:125` recording
*why* `project.md` is ignored (superseded by `CLAUDE.md` + `docs/`, retained
locally as a historical artifact only), so that the next reader of the ignore
file — human or agent — is not left guessing. `DOC-006` then closes with the
verdict "not a repository artifact; superseded; the ignore entry says so."
The alternative (the user asks for the local file to be deleted) is **gated
question Q2**.

### 15.2 `huggingface_space/` (`Fix.md` §17 item 8)

**Decision: not restored. `AGENTS.md:4`'s sentence is deleted.**

**Why.** The directory was deliberately removed in `3266746`
("chore: remove huggingface space and sync workflow"). Restoring a Gradio app
would add a public deployment surface, a second dependency set, and a second
place for capability claims to drift — at the tier whose purpose is to
eliminate drift — and nothing in Tiers 0–7 depends on it. `Fix.md:1602-1603`
offers exactly this choice; Tier 8 takes the "otherwise remove it from
`AGENTS.md` and public docs" branch. No public doc other than `AGENTS.md:4`
mentions it, so the removal is one sentence.

### 15.3 Packaging truth — the `gpu`/`tpu` extras

**Decision: the four accelerator extras are removed from
`pyproject.toml`** (`gpu` `:61`, `gpu-cuda` `:67`, `gpu-rocm` `:72`, `tpu`
`:77`) **and the `"gpu"` keyword `:20`**, and the `jax` dependency is offered
under one honest extra name.

**Why.** `pip install radiosim[gpu-cuda]` today installs `jax[cuda12]` and
delivers a package that has never executed on a GPU, whose `auto` backend
selects JAX only when a non-CPU device is present, whose only compiled kernel
is a per-`(time, frequency)` contraction inside host-side Python loops, and
whose own README says every measured JAX run is slower than NumPy. That is a
packaging-level claim in state 4's opposite: advertised and unsupported.
§4.2 permits "unsupported and rejected with an actionable error" or "absent";
an installable extra named `gpu-cuda` is neither. The replacement is a single
`jax` extra with a docstring-level comment pointing at `PERF-001`. When
`PERF-001` closes with measured accelerator evidence, the extras come back
with numbers behind them.

**Blast radius.** `pixi` environments pin their own CPU-only `jaxlib`
(`build = "cpu*"`) and do not read `[project.optional-dependencies]`, so no
environment, lockfile, or CI job changes. This is a metadata-only edit that
`tests/unit/test_release_metadata.py` does not currently cover; 8E extends it
to assert no accelerator-named extra exists while `PERF-001` is open.

### 15.4 The stale-naming sweep (`Fix.md` §17 item 14)

**Inventory at `95a937e`** — `git grep -i rrivis` returns **9 hits in 4 tracked
files**:

| File | Hits | Disposition |
|---|---|---|
| `AGENTS.md:34` | 1 | **Fix** → "RadioSim" |
| `antenna_layout_examples/1101503312_metafits.fits` | 1 | **Fix** — the FITS `COMMENT` card "Example MWA metafits file for RRIVis testing". Cards are fixed 80-byte records, so the replacement text fits in place and the file length is unchanged. The file is the shipped `mwa_metafits` example (`antenna_layout_examples/README_antenna_formats.md:17`) and is referenced by no test, so no fixture digest moves. If in-place rewriting proves to disturb any FITS checksum card, the fallback is to leave the comment and record the exception in the acceptance module's allow-list |
| `Tier3BeamObservabilityPlan.md:3223,3486` | 2 | **Never edited** — historical acceptance records describing a stale interpreter path observed at the time. Allow-listed |
| `Fix.md` (`:224,874,1609,6465,11624`) | 5 | **Never edited** — the register/history document. Allow-listed |

Plus the untracked `project.md` (Section 15.1), outside the tracked surface.

**The sweep is therefore two edits and one allow-list**, and Section 11's
scan 2 makes it permanent. The sweep's second half — "nonexistent symbols,
unsupported claims, and stale version/config counts" — is Section 11's scans
1, 3, 5 and 7, not a manual pass.

## 16. Design decision 10 — Tier 8 owns a version bump to `0.3.0`

**Decision: yes. `0.2.0` → `0.3.0`, in tracked metadata and the changelog only.
No tag, no release, no publish.**

**Why the bump belongs to this tier.** `Fix.md:1664`'s final exit criterion is
"the release notes disclose breaking changes and implemented capabilities". A
`[Unreleased]` heading is not release notes; it is a promise of them. And the
breaking changes are not marginal — across Tiers 1–7 the program removed 26
public Jones classes, removed `visibility.calculation_type` and `jones_config`,
removed the `numba` backend name, removed `BeamJones`/`AnalyticBeamJones`/
`FITSBeamJones`/`BeamManager`/`BeamFITSHandler`, replaced `generate_baselines`,
changed the canonical chain order (moving `P` sky-side of `C`), corrected the
coherency Stokes-`V` sign, changed every FITS-beam fingerprint, and rewrote the
config contract. Under §4.1's pre-v1 policy these are legitimate; under any
versioning convention they are not the same release as `0.2.0`.

**The truth argument, which is decisive.** `docs/changelog.rst:97-108` carries a
corrective note retracting `[0.2.0]`'s "Universal GPU acceleration" and
"Complete 8-term Jones chain" claims. If HEAD keeps the version string
`0.2.0`, then the current, honest, eight-tier-remediated package and the
package whose release notes are formally retracted **share one version
number**. A user reporting a bug against "radiosim 0.2.0" would be
unanswerable. That is itself a documentation-truth defect, and it is Tier 8's
to fix.

**Exact scope of the bump** (five sources, whose agreement is already tested at
`tests/unit/test_release_metadata.py:89-105`):

- `pyproject.toml:7` → `0.3.0`
- `pixi.toml:3` → `0.3.0`
- `src/radiosim/__about__.py:3` → `0.3.0`
- `docs/conf.py:15-16` (`release` and `version`) → `0.3.0`
- the installed `radiosim.__version__` follows from `__about__.py`

plus `docs/changelog.rst`: `[Unreleased]` becomes `[0.3.0] - 2026-08-02`
(the acceptance date, corrected to the actual date at implementation), a fresh
empty `[Unreleased]` is opened above it, and the `[0.3.0]` section is completed
to cover Tiers 1–6 as well as Tier 7 — today's `[Unreleased]` covers Tier 7
only, so 8E must walk the tier acceptance records and add the Tier 1–6
breaking changes that were documented in `docs/migration_guide.md` but never
entered the changelog.

**Also required in `[0.3.0]`, per Section 4**: a "Known limitations" subsection
naming `PERF-001`, `SCI-004`, `SCI-005`, `SCI-006`, `SCI-007`, and (if still
open) `CI-001`, each in one sentence with its register ID. Release notes that
list capabilities without listing these would fail §4.2 at the release level.

**Safety.** Verified: the package version feeds `provenance_sha256` only
(`src/radiosim/core/result.py:857`, inside `_provenance_hash`), never
`scientific_sha256` (`:789-841`), and no test pins a literal
`provenance_sha256`. The bump therefore cannot move any characterization
fingerprint pin. `pyproject.toml`, `pixi.toml`, `docs/conf.py` and
`__about__.py` are the only files containing the literal.

**What is explicitly excluded**: `git tag`, `gh release create`, a PyPI or
TestPyPI upload, a readthedocs configuration change, and any commit push.
Tier 8 prepares a release; it does not perform one. Whether the user wants even
the metadata bump inside Tier 8 is **gated question Q1**.

## 17. Slices

Six slices. Each is a single reviewable commit with an independent acceptance,
in the established program pattern. The writable list for each slice is
exhaustive — a slice touching a file outside its list is a defect.

### 8A — Characterization, residual baseline, and `CI-001`

**Purpose.** Pin the current state before changing it, so every later slice's
effect is measurable, and file the CI row.

**Work.**
1. New `tests/characterization/test_tier8_current_behavior.py` pinning today's
   drift as *current behavior*, each with a comment naming the slice that flips
   it: the four phantom flags in `examples/README.md`; the "Numba" backend
   sentence; the "Three shipped YAML samples" count against four files; the
   Sphinx 16-warning baseline (as a recorded number, not a build); the
   `--doctest-modules`-collects-zero fact; the absent `docs/api` pages for the
   six uncovered subpackages; `AGENTS.md`'s six defects; `CLAUDE.md`'s three;
   `get_required_services({realistic_foreground}) == {}`.
2. Record the clean-worktree Sphinx baseline (16) and the in-tree
   `docs/superpowers/` inflation (18) as an explicit note in the module
   docstring, citing the 7I ruling, so no later slice re-litigates it.
3. File `CI-001` in `Fix.md` §5 per Section 14, with the §5.6 evidence.
4. Land Section 14's items 2 and 3 — the unconditional machine-fingerprint
   emission and the numeric delta on the pin-failure path — in
   `tests/characterization/test_tier6_current_behavior.py`. **Moved here from
   8D** (correction applied at 8A): the emission's whole value is that a green
   run leaves a record of the runner that produced an *accepted* digest, so
   every CI run from this slice onward becomes evidence; deferring it to 8D
   discards 8B's and 8C's runs, which on a ~38% recurrence rate are the most
   likely places the next observation appears. Both changes are evidence-path
   only: no digest table grows, no assertion changes, and a check that supplies
   no cube behaves exactly as before, so `test_tier7_current_behavior.py`'s call
   sites need no edit and stay outside this slice.
5. Record the 5332-test collection baseline and the exact `pixi run test`
   result on this host.

**Writable.** `tests/characterization/test_tier8_current_behavior.py` (new);
`tests/characterization/test_tier6_current_behavior.py` (the Section 14 items 2
and 3 instrumentation and its module-docstring note — **evidence and message
path only**; digest tables untouched); `Fix.md` (register row + slice acceptance
note); `Tier8ReleasePlan.md` (status header).

**Gate.** `pixi run test -- tests/characterization/`, `pixi run lint`,
`pixi run check-format`, plus a bit-identity check that the three hermetic
shipped-config fingerprints are byte-identical to their values at `397c0e1`.

### 8B — Examples, doctests, and executed documentation

**Work.**
1. Correct `examples/README.md`: the three real flags; delete the four phantom
   invocations; replace the "JAX and Numba" sentence with the live backend set
   and a pointer to `README.md:346-380`; list all four `configs/*.yaml` with
   one accurate line each, including that `realistic_foreground_example.yaml`
   needs network (which 8D then makes the pre-flight say too).
2. Add the flag-parity test and the config-count/name test (Section 11 items 3
   and 6) into `tests/unit/test_tier8_release_acceptance.py` (new).
3. Remove `--doctest-modules` from `pyproject.toml:144`; add a `doctest` pixi
   task scoped to `src/radiosim`; fix the docstring doctests it surfaces
   (Section 9 item 3).
4. Decide and record the notebook treatment (Section 9 item 2).
5. Flip the corresponding 8A characterization pins in place.

**Writable.** `examples/README.md`;
`examples/scripts/simple_simulation.py` — **only** the Section 10 correction
scoping the built-in dimension assertion to the built-in path, no flag change
and no new behavior; `pyproject.toml` (pytest `addopts` only); `pixi.toml`
(new task only); `tests/unit/test_tier8_release_acceptance.py` (new);
`tests/characterization/test_tier8_current_behavior.py`; any
`src/radiosim/**/*.py` **docstring** the doctest run proves wrong — no logic
edit; `Fix.md`; `Tier8ReleasePlan.md`.

**Gate.** `pixi run test`, `pixi run doctest`, `pixi run lint`,
`pixi run check-format`,
`pixi run python examples/scripts/simple_simulation.py --help`,
`pixi run python examples/scripts/simple_simulation.py`.

### 8C — Sphinx strictness and API completeness

**Work.**
1. Fix all 16 warnings at source (Section 8): ten docstrings in
   `src/radiosim/backends/{__init__,base,numpy_backend}.py` and
   `src/radiosim/core/polarization.py`; `:orphan:` (or a toctree entry) for
   `docs/HERA_VSIM_ANALYSIS.md`; its three lexer annotations; the
   `display_version` option at `docs/conf.py:88`; the MyST anchor at
   `docs/migration_guide.md:617`.
2. Set `SPHINXOPTS ?= -W --keep-going` in `docs/Makefile`.
3. Add the missing `docs/api/` pages: `sky.rst` (the `core.sky` subpackage),
   `simulator.rst` extension or a new `algorithms.rst` for
   `radiosim.simulator`, `result.rst` for `core.result`, `utils.rst`,
   `visualization.rst`, `observability.rst`, plus automodule entries for the
   nine uncovered `core/` modules. Each new page enters `docs/index.rst`'s
   toctree. Every page must build clean under `-W`, which is the real cost of
   this item — new autodoc coverage surfaces new docstring warnings.
4. Add Section 11 scans 1, 4 and 5 (removed names, documented paths, documented
   symbols) to the acceptance module.

**Writable.** `docs/**` (`conf.py`, `Makefile`, `index.rst`, `api/*`,
`HERA_VSIM_ANALYSIS.md`, `migration_guide.md` anchor only);
`src/radiosim/backends/__init__.py`, `base.py`, `numpy_backend.py`,
`src/radiosim/core/polarization.py` — **docstrings only**; **plus, by the
correction below, `src/radiosim/core/hybrid.py`,
`src/radiosim/core/polarization_basis.py`, `src/radiosim/core/precision.py`,
`src/radiosim/core/sky/combine/regrid.py`,
`src/radiosim/core/sky/io/serialization.py`, `src/radiosim/simulator/base.py`,
`src/radiosim/simulator/rime.py`, `src/radiosim/utils/logging.py` — also
docstrings only**; `tests/unit/test_tier8_release_acceptance.py`;
`tests/characterization/test_tier8_current_behavior.py`; `Fix.md`;
`Tier8ReleasePlan.md`.

**Writable-list correction, applied at 8C.** Section 20 risk 1 predicts that
"new `docs/api/` pages surface new Sphinx warnings" and rules that a page
"lands with the debt fixed or the page is deferred and the deferral recorded".
The four source files named above are the sites of the *sixteen known*
warnings; item 3's six new pages surfaced twenty-six more, in eight further
files. Deferring those pages would fail Section 18 criterion 4, so the debt is
paid and the grant is extended to exactly the eight files, **docstring prose
only** — no signature, default, branch, or constant. Nineteen of the
twenty-six were removed by two `docs/conf.py` settings rather than by any
source edit (`napoleon_use_ivar = True` for duplicate dataclass attribute
descriptions, `napoleon_google_docstring = True` because the package mixes
both styles while only numpydoc was enabled). `src/radiosim/simulator/rime.py`
also carries a "Backend abstraction for CPU/GPU" line that 8E's Section 11
scan 7 owns; 8C leaves it untouched, exactly as it leaves
`src/radiosim/simulator/__init__.py:12`.

**Correction, applied at the Tier 8C independent re-acceptance (rejection
repair round).** Two further scan-7 instances exist beyond the one named
above, both confirmed present at `f78c330` and both left untouched by 8C:
`src/radiosim/simulator/__init__.py:65` (a `See Also` line, "`radiosim.backends
: Backend abstraction for CPU/GPU/TPU`") and
`src/radiosim/simulator/base.py:122-129` (`VisibilitySimulator.supports_gpu`,
the abstract base class's concrete default — docstring "Whether the simulator
supports GPU acceleration. ... Default is True." and the property itself
`return`s `True`). The `base.py` instance is the more material of the two: it
is a "Default is True" capability claim on the shared base class, the same
defect class as the `__init__.py:12` bullet, not a bare mention. The
`__init__.py:65` and `rime.py:132` lines are `See Also` cross-references
naming what the `backends` module abstracts over (device targets a generic
`ArrayBackend` interface is written against), not claims that this simulator
has measured or achieved acceleration; scan 7's matching should target
capability-claim language ("supports GPU", "acceleration via", "Default is
True" beside a device name) rather than a bare "GPU"/"TPU"/"CPU" token, so
these two do not need rewording and do not force a scan-7 false positive.
8E's writable list is extended accordingly (see 8E's entry below).

**One further correction, applied at 8C.** Section 8's "baseline discipline"
paragraph rules that an in-tree build reporting 18 because gitignored
`docs/superpowers/` exists "is not a regression and must not be treated as
one". Making `-W` the default changed what that artifact costs: it would turn
a stray untracked directory into a *failed* build for any contributor who has
one. 8C therefore adds `"superpowers"` to `exclude_patterns` in
`docs/conf.py`, so the clean-worktree and in-tree builds are the same build.
This is not a `suppress_warnings` entry — that setting stays unset — and the
excluded directory is asserted gitignored and empty of tracked files by
`tests/unit/test_tier8_release_acceptance.py`, so the exclusion can never hide
a documented page.

**Gate.** `make -C docs clean html` (now `-W` by default) exits 0 with **zero**
warnings in a clean detached worktree; `pixi run test`; `pixi run lint`;
`pixi run check-format`.

### 8D — Scan hardening, `SKY-002`, CI shape, and the fingerprint instrumentation

**Work.**
1. Create `tests/support/repo_scan.py` and convert all 20 sites (Section 12),
   including the `.ipynb_checkpoints` regression proof.
2. Close `SKY-002` (Section 13), including the `network_service` →
   `network_services` widening and the four tests.
3. CI: add the example-execution step and (per 8B's decision) the notebook
   step and the doctest step to the `quality` job; add a **CLI-to-artifact**
   integration test — `radiosim --config <tmp config>` invoked as a
   subprocess, producing an on-disk run directory whose manifest, HDF5, and
   summary JSON are then read back — which is `Fix.md:1604` item 9's actual
   ask and the one thing `tests/integration/` does not currently do.
4. The CI re-run on 8D's SHA and the measured decision on the digest class.
   (Section 14's items 2 and 3, the instrumentation itself, moved to 8A —
   see that slice's item 4 — so that 8B's and 8C's CI runs are already
   producing the evidence this decision reads.)

**Writable.** `tests/support/__init__.py`, `tests/support/repo_scan.py` (new);
the 10 converted test files; `tests/integration/test_cli_end_to_end.py` (new);
`tests/characterization/test_tier6_current_behavior.py` (instrumentation and
numeric-delta reporting; digest tables only if Section 14's conditional is met);
`src/radiosim/core/sky/registry/{core,facade,catalogs}.py`,
`src/radiosim/core/sky/recipes/realistic_foreground.py`, every other loader
registration touched by the keyword rename, `src/radiosim/utils/network.py`,
`src/radiosim/core/sky/diagnostics/discovery.py`; `.github/workflows/ci.yml`;
new/updated tests under `tests/unit/test_utils/` and `tests/unit/test_core/`;
`Fix.md`; `Tier8ReleasePlan.md`.

**Gate.** `pixi run test`; `pixi run lint`; `pixi run check-format`;
`pixi run typecheck`; the checkpoint regression proof; a named CI run ID at the
slice SHA with its per-job result stated honestly.

### 8E — Final sweep, agent-facing truth, changelog, and release metadata

**Work.**
1. `AGENTS.md`: all six defects (Section 5.4) — Numba, the test-directory
   list, the Hugging Face sentence (`DOC-007`), the doctest/marker sentence,
   the RRIVis naming, the discharged TODO — plus a sentence describing the 41
   `simulators/` submodules and the fact that a plain `git clone` does not
   fetch them.
2. `CLAUDE.md`: the three defects (`:200` MyPy → Pyright with the real command;
   `:181` the `io/` module list; `:216` the discharged TODO), plus the same
   `simulators/` note.
3. `README.md`: the config count at `:408` (made derived and tested); a
   `## Repository layout` addition covering `simulators/`, `tests/`'s five
   directories, `output/benchmarks/reference/`, and `output/crossvalidation/`;
   the `## Tests` section extended to name the integration, characterization,
   performance and crossval suites and `pixi run bench`.
4. The FITS `COMMENT` card (Section 15.4) and the `.gitignore:125` explanatory
   line (Section 15.1).
5. `pyproject.toml`: remove the four accelerator extras and the `"gpu"` keyword
   (Section 15.3); extend `tests/unit/test_release_metadata.py` to assert their
   absence while `PERF-001` is open.
6. The version bump and the `[0.3.0]` changelog section including "Known
   limitations" (Section 16), gated on Q1.
7. Section 11 scans 2, 7 and 8 (naming, GPU-claim citation, `pixi run` task
   existence) added to the acceptance module. **A specific pre-existing
   instance for scan 7 to catch, found and routed here at the 8B independent
   acceptance review**: `src/radiosim/simulator/__init__.py:12`'s module
   docstring states "GPU acceleration via JAX backend" as a bullet under
   `rime`'s current capabilities — an unsupported claim under `CLAUDE.md`'s
   standing rule (no accelerator has ever been measured; the locked JAX build
   is CPU-only and every measured JAX-CPU run is slower than NumPy,
   `output/benchmarks/reference/`). `git log -S` dates the line to the
   original RRIVis→RadioSim rename (`be231d2`), predating every prior tier's
   register rows, so it is not `DOC-005`'s previously enumerated instance
   (README.md/examples/README.md/the packaging extras) but the same defect
   class, missed by Section 5's inventory. It is out of 8B's writable grant
   (not a doctest failure, so 8B's docstring-fix clause does not reach it) and
   is deliberately left unedited here. 8E either rewords the line to name
   `PERF-001` in its enclosing paragraph or deletes the claim outright, and
   confirms scan 7's file set (built on the Section 12 git-scoped lister)
   actually reaches `src/**/*.py` docstrings and not only `docs/**`/`README*`,
   so this instance fails the scan before it is fixed rather than being
   silently out of scope. **Two further instances, found and routed here at
   the Tier 8C independent re-acceptance (rejection repair round)**:
   `src/radiosim/simulator/base.py:122-129`
   (`VisibilitySimulator.supports_gpu`'s docstring, "Whether the simulator
   supports GPU acceleration. ... Default is True.", on a property that
   concretely `return`s `True`) is the same defect class as the `:12` bullet —
   a "Default is True" capability claim inconsistent with the standing rule —
   and 8E rewords or removes it exactly as it does the `:12` bullet.
   `src/radiosim/simulator/__init__.py:65` and `src/radiosim/simulator/rime.py`'s
   own "Backend abstraction for CPU/GPU" `See Also` line are, by contrast,
   scope-naming cross-references to the `backends` module (which does define a
   device-agnostic `ArrayBackend` interface), not claims that this simulator
   has measured or achieved acceleration; scan 7 is scoped to capability-claim
   language ("supports GPU", "acceleration via", "Default is True" beside a
   device name) rather than a bare "GPU"/"TPU"/"CPU" token, so these two `See
   Also` lines are confirmed non-instances and need no edit.

**Writable.** `AGENTS.md`; `CLAUDE.md`; `README.md`; `.gitignore`;
`antenna_layout_examples/1101503312_metafits.fits`; `pyproject.toml`;
`pixi.toml`; `src/radiosim/__about__.py`; `docs/conf.py`;
`docs/changelog.rst`; `docs/contributing.rst` (test-directory description
only); `src/radiosim/simulator/__init__.py` (the `:12` GPU-claim sentence
above, only — no logic edit); `src/radiosim/simulator/base.py` (the
`supports_gpu` docstring at `:122-129` only — no logic edit);
`tests/unit/test_release_metadata.py`;
`tests/unit/test_tier8_release_acceptance.py`;
`tests/characterization/test_tier8_current_behavior.py`; `Fix.md`;
`Tier8ReleasePlan.md`.

**Gate.** the full `Fix.md:1636-1645` verification gate, plus
`pixi run radiosim --version` reporting the bumped version.

### 8F — Whole-tier gate

**Work.** No new capability. Re-prove every criterion in Section 18 from
current source and fresh runs, not from prior slices' claims; close
`DOC-001`..`DOC-008` with the Section 19 evidence map; state `CI-001`'s
disposition; state `SKY-002` as `DONE`; disclose `PERF-001`, `SCI-004`,
`SCI-005`, `SCI-006`, `SCI-007` unchanged; poll CI to completion on the exact
acceptance SHA and report all eight jobs by run ID.

**Writable.** `Fix.md` (register status transitions + the whole-tier acceptance
record); `Tier8ReleasePlan.md` (status header + acceptance appendix).

## 18. Whole-tier criteria

Tier 8 is complete when **all sixteen** hold, each proved by a named command,
test, or artifact:

1. `pixi run python examples/scripts/simple_simulation.py` and `--help` both
   succeed, and CI runs both.
2. Every flag named in `examples/README.md` exists in the script's `--help`,
   and vice versa, proved by test.
3. `make -C docs clean html` builds with `-W --keep-going` by default and
   reports **zero** warnings in a clean detached worktree.
4. `docs/api/` covers `core.sky`, `simulator`, `core.result`, `utils`,
   `visualization`, `core.observability`, and the nine uncovered `core/`
   modules, each reachable from `docs/index.rst`'s toctree.
5. Docstring doctests execute under a named task and a CI step, and pass.
6. Every `radiosim.`-rooted symbol and every relative path in tracked prose
   resolves, proved by test.
7. No removed name appears outside the historical allow-list, proved by test —
   including `GeometricDelayJones`, `generate_baselines`, `numba`,
   `calculation_type`, and the 26 Jones classes.
8. `RRIVis` appears in no tracked file outside the historical allow-list,
   proved by test.
9. Config counts and names in `README.md` and `examples/README.md` are derived
   from `configs/`, proved by test.
10. Every accelerator/speed sentence in tracked prose cites
    `output/benchmarks/reference/` or names `PERF-001`, proved by test.
11. All 20 repository/package scans use the shared git-scoped lister, and a
    gitignored `src/radiosim/.ipynb_checkpoints/*.py` naming a removed class no
    longer fails any test, proved by the regression test.
12. `get_required_services` reports both services for
    `configs/realistic_foreground_example.yaml`, the pre-flight prints the
    network-dependent branch, and every network-importing loader declares a
    service — `SKY-002` `DONE`.
13. A CLI-to-artifact integration test drives `radiosim --config` as a
    subprocess and reads the published run directory back.
14. `CI-001` exists with the §5.6 evidence, the machine fingerprint is emitted
    unconditionally, and pin failures report a numeric delta.
15. Release metadata agrees across all five sources at the chosen version,
    `docs/changelog.rst` has a dated release section covering Tiers 1–7 with a
    "Known limitations" subsection naming every open/roadmap row, and no
    accelerator-named extra is installable while `PERF-001` is open.
16. CI is polled to completion on the acceptance SHA and its result is stated
    per job by run ID — green, or red with the failure named and disclosed in
    the release notes.

## 19. `DOC` closure evidence map

What the whole-tier gate must exhibit for each row. Rows marked *prior-tier*
close without new Tier 8 code, on re-verified evidence.

| Row | Closure evidence |
|---|---|
| `DOC-001` | the script executed in CI (criterion 1); the flag-parity test (2); `examples/README.md` corrected; *prior-tier* for the script body itself, re-verified symbol by symbol at 8F |
| `DOC-002` | *prior-tier* (Tier 2). Evidence: `generate_baselines` has zero repository occurrences, kept true by criterion 7; every README and quickstart block executed at 8F |
| `DOC-003` | *prior-tier* (Tier 7J). Evidence: the 19-name `__all__`, the captioned historical mentions, kept true by criterion 7 |
| `DOC-004` | criterion 9 |
| `DOC-005` | *prior-tier* for `README.md` (Tier 6I), re-verified against the benchmark record at 8F; new work for `examples/README.md` and the packaging extras; kept true by criterion 10 |
| `DOC-006` | Section 15.1's verdict plus the `.gitignore` explanatory line; criterion 8 keeps the tracked surface clean |
| `DOC-007` | `AGENTS.md:4` sentence removed; no `huggingface`/`gradio`/Space reference in any tracked file, proved by an added scan |
| `DOC-008` | criteria 1, 3, 5, 13, 16 — the three named gaps closed (example execution, notebook decision recorded, `-W` docs gate, doctests, CLI-to-artifact test), with the pre-existing CI and integration/performance suites re-verified rather than rebuilt |

## 20. Risk register

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| 1 | New `docs/api/` pages surface new Sphinx warnings, so 8C's "-W with zero warnings" costs far more than the 16 | **high** — `core.sky` alone is a large docstring surface never rendered before | 8C lands the 16 fixes and `-W` **first**, then adds api pages one subpackage per commit-step, rebuilding after each. If a subpackage's docstring debt is large, its page lands with the debt fixed or the page is deferred and the deferral recorded — never with a `suppress_warnings` entry |
| 2 | Enabling doctests on `src/` surfaces many failing `>>>` examples (299 lines, 22 files) | **high** | scoped to its own task, expected to fail on first run; gated question Q4 sets the appetite. Examples that cannot be made deterministic offline are converted to non-doctest code blocks rather than skipped |
| 3 | `network_service` → `network_services` rename touches every loader registration | medium | mechanical, typed, covered by the registry-completeness test; `pixi run typecheck` gates it; no compatibility shim per §4.1 |
| 4 | The CI fingerprint divergence recurs during the tier and blocks acceptance of an unrelated slice | **high** — ~38% on that cell | every slice's gate reports CI per job by run ID and distinguishes "this slice's failure" from "`CI-001` recurrence"; a `CI-001` recurrence never blocks a slice whose own diff cannot reach the fingerprint path |
| 5 | Section 11's prose-parsing scans are brittle (regex over Markdown/RST) | medium | scans parse conservatively and fail *closed* only on unambiguous forms; each carries a documented false-positive escape via the allow-list, and the allow-list is itself asserted non-empty-of-justification |
| 6 | Rewriting the FITS `COMMENT` card corrupts the example file | low | the card is a fixed 80-byte record; verify by reopening with astropy and comparing every other header card and the data array bitwise; fallback is to allow-list the string |
| 7 | The version bump breaks a pin somewhere unanticipated | low | verified: version feeds `provenance_sha256` only (`core/result.py:857`), no literal `provenance_sha256` pin exists; the five-source agreement is already tested |
| 8 | Tier 8 quietly absorbs `PERF-001`/`SCI-004..007` by writing optimistic release notes | medium | Section 4's seven prohibitions plus criterion 15's mandatory "Known limitations" subsection naming every row by ID |
| 9 | Removing the `gpu*` extras breaks a user install | **none pre-v1**, by §4.1 and the project's stated zero-user beta status | recorded in the changelog's breaking-changes list |
| 10 | The plan closes a `DOC` row on prior-tier evidence that has since drifted | low | 8F re-proves every *prior-tier* closure from current source, not from this plan's Section 6 |

## 21. Gated questions

These require the user's decision before the affected work lands. None blocks
8A.

**Q1 — Does Tier 8 perform the `0.2.0` → `0.3.0` metadata bump?**
The plan recommends **yes** (Section 16): a release-reconciliation tier whose
release notes are headed `[Unreleased]`, on a package sharing its version
string with a formally retracted changelog entry, has not reconciled the
release. The bump is metadata + changelog only — no tag, no publish. If the
answer is no, 8E still writes the full `[0.3.0]`-shaped content under
`[Unreleased]` and `DOC` closure is unaffected; only criterion 15's version
clause changes.

**Q2 — What happens to the untracked, gitignored `project.md`?**
The plan's default is **leave it and explain the ignore entry** (Section 15.1).
The alternative — deleting the local file — is a change to the user's working
tree that no agent should make unasked, so it is asked here. A third option
(track a heavily reduced, current version as a project overview) is **not**
recommended: it would create a fourth architecture surface alongside
`README.md`, `CLAUDE.md` and `docs/`.

**Q3 — If 8D's numeric probe shows the `linux-64-py311` divergence is at ULP
scale, may the observed digest class be appended to the pin tables?**
The plan recommends **yes, conditionally** (Section 14): appended with an
honest justification that names the discriminator as unidentified and records
the measured delta, with `CI-001` narrowed rather than closed. Refusing means
`main` stays red on one job for the remainder of the tier and the release notes
must disclose it. This is asked because four prior commits appended on weaker
evidence and the plan is deliberately breaking that reflex.

**Q4 — How much docstring-doctest debt should 8B pay down?**
299 `>>>` lines across 22 files have never executed. Options: (a) fix all, and
let 8B's size follow; (b) fix the public-API modules only
(`api/`, `backends/`, `core/result.py`, `core/sky/` public entry points) and
mark the rest with a tracked follow-up row; (c) scope the doctest task to a
named module list that grows over time. The plan recommends **(a)** if the
first run's failure count is under roughly thirty, and **(b)** otherwise, with
the measured count reported before choosing. This is a scope question, not a
correctness one.

**Q5 — Should the notebook be executed in CI?**
Depends on whether a notebook-execution dependency is acceptable in the
`default` pixi environment (Section 9 item 2). If not, the notebook stays at
state 2 and Tier 8 says so plainly rather than implying execution.

## 22. Verification gate

`Fix.md:1636-1645`, unchanged, plus three additions this plan introduces:

```bash
pixi run radiosim --version
pixi run radiosim validate configs/config.yaml
pixi run python examples/scripts/simple_simulation.py --help
pixi run python examples/scripts/simple_simulation.py          # added
pixi run doctest                                               # added
pixi run test
pixi run lint
pixi run check-format
pixi run typecheck
make -C docs clean html SPHINXOPTS="-W --keep-going"
```

plus, at 8D and 8F, a named CI run ID at the exact SHA with all eight jobs
reported individually. The docs command is retained explicitly even after
`docs/Makefile`'s default changes, so the gate remains correct if a future
contributor changes that default.

Notebook validation runs under whichever command Q5 settles; if Q5 settles on
"not executed", this gate does not gain a notebook line and the whole-tier
record says why.

## 23. Suggested commits

Aligned to the slices; `Fix.md:1650-1655`'s four suggestions are a subset.

- 8A `test(release): pin Tier 8 documentation baseline and file CI-001`
- 8B `fix(examples): reconcile the example surface with the public API`
- 8C `docs: build clean under warnings-as-errors and complete the API reference`
- 8D `test(release): harden repository scans, close SKY-002, and cover the CLI`
- 8E `docs(release): reconcile project truth and prepare the 0.3.0 notes`
- 8F `docs(release): accept Tier 8 and close the remediation program`
