"""Characterize the Tier 8 documentation, packaging, and release baseline.

Every test in this module pins documentation and configuration behavior that
exists on ``main`` **today**, before any Tier 8 remediation.  Each test docstring
names the slice that owns the deliberate flip (``FLIPPED BY: Tier 8x``); that
slice must update the named test in the same commit that changes the behavior.
A test with no ``FLIPPED BY`` line pins behavior Tier 8 preserves.

The distinguishing feature of this tier is that its defects are *sentences*
rather than code paths, so the pins here are assertions about tracked file
content.  That is deliberate and is the whole mechanism of
``Tier8ReleasePlan.md`` Section 7: after Tier 8, a documented statement must be
executed, scanned, cited, or absent, and drift is detected by the suite instead
of by the next review.  These pins are the "scanned" half of that contract
pointed at the *current, wrong* text, so that every later slice's effect is
measurable rather than asserted.

Slice 8A is authorized by ``Tier8ReleasePlan.md`` Section 17 against ``main`` at
``397c0e1`` ("docs(release): accept Tier 8 design").  ``397c0e1``, ``13b59f3``
and ``785d576`` touch only ``Fix.md`` and ``Tier8ReleasePlan.md``, so ``src/``,
``tests/``, ``docs/``, ``configs/`` and ``examples/`` are byte-identical to
``47df8fc``, the last commit with a fully green CI run.  Every source-line
citation in ``Tier8ReleasePlan.md`` Sections 5 to 7 was therefore taken at the
same tree these pins measure.

Tier 8A evidence record
=======================

Section 17 grants 8A a new characterization module, ``Fix.md``, the plan
document, and -- as corrected before this slice landed -- the Tier 6
characterization module that owns the fingerprint instrumentation.  The recorded
evidence lives here, following the Tier 5A, 6A and 7A precedent.

**The Sphinx warning baseline is 16, measured in a clean detached worktree.**
An in-tree build reports **18** because ``docs/superpowers/`` exists in a
working tree and is gitignored (``.gitignore:211``); those two extra warnings are
an artifact of the untracked directory, not a regression, and the ruling that
the clean-worktree number is the governing one was made at Tier 7H (diagnosis)
and ratified at Tier 7I.  ``Tier8ReleasePlan.md`` Section 8 restates it, the
Tier 8 design acceptance independently reproduced the 16 in a fresh detached
worktree at ``785d576``, and this note exists so that no later slice
re-litigates the difference.  The 16 group as ten docutils docstring parse
errors (``backends/numpy_backend.py``'s bare ``*operands``/``*args``,
``backends/base.py``'s ``|x|``, ``backends/__init__.py``'s ``get_backend``
indentation, and five ``|...|`` substitutions in
``core/polarization.py::jones_matrix_power``), one ``toc.not_included``
(``docs/HERA_VSIM_ANALYSIS.md``), one unsupported theme option
(``docs/conf.py:88``), three ``misc.highlighting_failure``
(``docs/HERA_VSIM_ANALYSIS.md:243,364,377``), and one ``myst.xref_missing``
(``docs/migration_guide.md:617``).  8C owns driving that count to zero and
making ``-W --keep-going`` the default; this module pins the *sources* of the
16, which is the part a test can check without running a build.

**Test-collection baseline at this slice.** ``pytest --collect-only -q -n 0``
reported **5332 tests collected in 4.76 s** at the design gate, and the design
acceptance independently reproduced both that number and the fact that adding
``--doctest-modules`` to the invocation changes it by zero, because
``testpaths = ["tests"]`` (``pyproject.toml:137``) means collection never
reaches ``src/``.  The flag at ``pyproject.toml:144`` is therefore dead
configuration, and the 299 ``>>>`` lines in 22 tracked ``src/radiosim``
docstrings have never executed.  8B owns making that real.

**Full-suite result on this host at 8A.** ``pixi run test -- -m "not slow"``
(pytest-xdist, ``-n auto``) on ``osx-arm64``, macOS 26.5.2, Apple M1 Max:
**5338 passed, 1 skipped, 10 deselected** in both gating environments, with 27
warnings under py311 and 41 under py312.  That is the pre-8A baseline of 5322
passed plus this module's 16 tests, with the skip, the deselections and both
warning counts unmoved.  Full collection is **5348** (5332 + 16); ``-m "not
slow"`` collects 5338 of them.  No pin family moved: 8A is instrumentation and
characterization only, no digest table grew or shrank by a character, and the
three hermetic shipped-config fingerprints reproduce byte-identically in both
environments at their recorded ``397c0e1`` values.

**CI state at this slice.** ``main`` is red on exactly one of eight jobs
(``linux-64 / Python 3.11``) for the reason now filed as ``CI-001`` in
``Fix.md`` Section 5.  8A does not fix it and does not append the divergent
digest class; it makes the next occurrence adjudicable, per
``Tier8ReleasePlan.md`` Section 14 items 2 and 3, whose implementation lives in
``tests/characterization/test_tier6_current_behavior.py``.

Tier 8B addendum -- the measured doctest debt, and what was paid
===============================================================

**Q4, measured before choosing.** The first real run of
``pytest --doctest-modules src/radiosim`` collected **54** doctest items and
reported **34 failed, 20 passed**.  Because pytest stops a doctest at its first
failing example, 34 was a *lower* bound on the broken examples: fixing one
routinely exposed the next in the same docstring.  8B took option **(a)**, fix
all, rather than (b)'s public-API-only scope, for two reasons.  First, the
plan's own concrete artifact -- the ``doctest`` task string in Section 9 item
3 -- is scoped to the whole of ``src/radiosim``, so option (b) would have
required inventing a module list the design never wrote and filing a debt row
for a remainder of seventeen failures.  Second, 34 against a stated threshold
of "roughly thirty" is inside the fuzz of that word.  The result is
``pixi run doctest`` green over the whole package: **41 items, all passing**.

The item count fell from 54 to 41 because 13 examples were **not** repaired
into executable doctests but converted to ``.. code-block:: python``.  Each is
one of three kinds that cannot execute hermetically: it needs a configuration
document on disk (``Simulator.from_yaml``), it needs solver state the caller
does not have (``calculate_visibilities``'s ``instrument_view``,
``JonesChain``'s terms), or it makes a live network call
(``get_catalog_columns`` queries VizieR, ``get_racs_columns`` queries CASDA
TAP -- both of which really did hit the network during the 8A-state
measurement).  That is Section 9 item 4's reasoning applied to docstrings: an
example that cannot execute must not be dressed as one that did.

**Q5, decided: the notebook is executed.**  ``jupyter``, ``nbconvert`` and
``ipykernel`` are already in ``pixi.toml``'s ``[dependencies]``, so the
dependency gate in ``Tier8ReleasePlan.md`` Section 9 item 2 is satisfied with
no environment inflation.  ``jupyter nbconvert --to notebook --execute
--stdout examples/notebooks/01_basic_usage.ipynb`` was run at 8B and exited 0,
writing nothing to the working tree.  The notebook is fully offline: five code
cells over the bundled HERA layout and synthetic sources.  8D wires the command
into the ``quality`` job; 8B records the decision and the evidence.

**The one script change.** ``examples/scripts/simple_simulation.py``'s
``--config`` path was broken at 8A and is fixed here under the Section 10
correction recorded in ``Tier8ReleasePlan.md``: ``main()`` asserted the
built-in example's ``(1, 15, 2, 4)`` for every run, so ``--config`` against any
shipped document raised ``AssertionError``.  The assertion is now scoped to the
built-in path.  No flag was added or removed -- the pin below still holds --
and the default run's scientific fingerprint is unchanged.

Tier 8C addendum -- how the sixteen went to zero, and what it cost
=================================================================

**The measurement.** The 8A baseline reproduced exactly: a forced-``-E`` build
of a clean detached worktree at ``ac35159`` reported **16 warnings**.  After
8C the same build reports **0**, and ``make -C docs clean html`` -- now
``-W --keep-going`` by default -- exits 0.

**The in-tree/clean-tree difference is gone rather than tolerated.** 8A
recorded 18-in-tree versus 16-clean as an artifact of the gitignored
``docs/superpowers/`` and asked no later slice to re-litigate it.  Turning
warnings into errors changed the stakes: the artifact would have turned a
stray untracked directory into a *failed* docs build for any contributor who
has one.  8C therefore added ``"superpowers"`` to ``exclude_patterns`` in
``docs/conf.py``.  That is not a suppression -- ``suppress_warnings`` is still
unset, and the two warnings concerned files that are not tracked, not shipped,
and not documentation.  The tracked documentation set now builds identically in
both trees, which is what makes the gate safe to turn on.

**Two conf.py settings did more work than any docstring edit.**
``napoleon_use_ivar = True`` removed nineteen ``duplicate object description``
warnings that appeared the moment the new pages rendered dataclasses carrying
both a numpydoc ``Attributes`` section and annotated fields.
``napoleon_google_docstring = True`` removed seven more: the package mixes
numpydoc and Google style, only numpydoc was enabled, and every ``Args:`` block
in a Google-style docstring was rendering as a block quote -- wrong output, not
merely a warning.  Both changes were measured against the pre-existing page set
first and introduced no warning there.

**The one thing a future reader should know about the gate.**
``sphinx.ext.intersphinx`` emits an unsuppressable ``WARNING`` when it cannot
reach *any* configured inventory, so a fully offline ``make -C docs html`` now
fails on that warning rather than merely reporting it.  ``docs/Makefile``
documents the ``SPHINXOPTS=`` override for inspecting such a build.  This was
observed once during 8C on a transient network failure and did not reproduce;
it is recorded here rather than worked around, because ``nitpicky`` was
rejected in ``Tier8ReleasePlan.md`` Section 8 for exactly this
network-sensitivity reason and the reader deserves to know the residue.

**The writable-list extension 8C took, and why.** Section 17's 8C grant names
four source files for docstring fixes.  Adding the six new API pages surfaced
warnings in seven more -- ``core/hybrid.py``, ``core/polarization_basis.py``,
``core/precision.py``, ``core/sky/combine/regrid.py``,
``core/sky/io/serialization.py``, ``simulator/base.py``,
``simulator/rime.py``, ``utils/logging.py`` -- which is precisely the outcome
``Tier8ReleasePlan.md`` Section 20 risk 1 anticipates ("its page lands with the
debt fixed or the page is deferred").  Deferring the pages would have failed
whole-tier criterion 4, so the debt was paid.  Every edit is prose inside a
docstring: a blank line before a bullet list, a table's column rules widened to
fit its own header, ``*`` escaped, ``|...|`` made literal, and two footnotes
cited so they stop being unreferenced.  No signature, default, branch or
constant moved, and the ``pixi run doctest`` item count is unchanged.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The clean-detached-worktree Sphinx warning count.  Recorded rather than
#: rebuilt: a Sphinx build is far too slow and too environment-sensitive to sit
#: in the standard gate, and ``Tier8ReleasePlan.md`` Section 17 asks 8A for the
#: number, not for the build.  It was 16 at 8A; 8C fixed every one of the
#: sixteen at its source and drove it to zero.
SPHINX_WARNING_BASELINE = 0

#: The same build in a working tree that contains the gitignored
#: ``docs/superpowers/``.  It was 18 at 8A -- the two extra ``toc.not_included``
#: warnings were an artifact of the untracked directory, never a regression.
#: 8C made the two builds identical by excluding that directory in
#: ``docs/conf.py`` rather than by suppressing its warnings, because ``-W`` is
#: now the ``docs/Makefile`` default and a stray untracked directory must not
#: turn a contributor's docs build into a failure.
SPHINX_WARNING_BASELINE_WITH_UNTRACKED_SUPERPOWERS = 0


def _read(relative_path: str) -> str:
    """Return the text of a tracked repository file, for source-truth pins."""
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _script_flags() -> set[str]:
    """Return every ``--flag`` the shipped example script's parser defines."""
    source = _read("examples/scripts/simple_simulation.py")
    return set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', source))


def _documented_flags() -> set[str]:
    """Return every ``--flag`` ``examples/README.md`` gives the example script.

    ``--help`` is excluded: argparse supplies it, so it is real without being an
    ``add_argument`` call, and 8B's flag-parity test makes the same allowance.

    Scoped at 8B to the command blocks that name the script plus the inline code
    spans, rather than to every ``--token`` in the file.  The corrected document
    also shows a ``jupyter nbconvert`` invocation (Q5), whose ``--to`` and
    ``--execute`` are real flags of a different program and are not the example
    parser's business.
    """
    text = _read("examples/README.md")
    blocks = [
        block
        for block in re.findall(r"```bash\n(.*?)```", text, re.DOTALL)
        if "simple_simulation.py" in block
    ]
    spans = re.findall(r"`([^`\n]+)`", text)
    found: set[str] = set()
    for fragment in blocks + spans:
        found |= set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", fragment))
    return found - {"--help"}


def _shipped_config_names() -> list[str]:
    """Return the shipped YAML sample filenames, derived from the filesystem."""
    return sorted(path.name for path in (REPO_ROOT / "configs").glob("*.yaml"))


# ---------------------------------------------------------------------------
# examples/ surface (DOC-001, DOC-005)
# ---------------------------------------------------------------------------


def test_example_script_defines_exactly_three_flags() -> None:
    """Pins the example script's real public surface.

    Preserved, not flipped: ``Tier8ReleasePlan.md`` Section 10 decides that the
    correction is to the document, never to the script, so the script must still
    define exactly these three flags after 8B.  The script's own contract is a
    deterministic offline smoke run that writes no artifacts, which is what makes
    it safe to execute unconditionally in CI (8D); growing it four file-writing
    flags to satisfy a stale README would invert that.
    """
    assert _script_flags() == {"--config", "--backend", "--progress"}


def test_examples_readme_documents_exactly_the_flags_the_script_defines() -> None:
    """FLIPPED AT 8B.  Was ``..._documents_four_flags_the_script_does_not_define``.

    Before 8B the document printed four flags -- ``--no-plot``, ``--save``,
    ``--plot``, ``--output-dir`` -- in copy-pasteable command blocks, each of
    which failed with ``error: unrecognized arguments``.  That was the live half
    of ``DOC-001``.  The document now describes the three flags that exist, and
    the rule that keeps it there is
    ``tests/unit/test_tier8_release_acceptance.py``'s parity scan, which reads
    the *live* ``--help`` rather than the source.  This pin is the cheap
    source-level mirror of it.
    """
    assert _documented_flags() == _script_flags()
    for phantom in ("--no-plot", "--save", "--plot", "--output-dir"):
        assert phantom not in _read("examples/README.md"), phantom


def test_examples_readme_names_the_live_backend_set_and_not_numba() -> None:
    """FLIPPED AT 8B.  Was ``..._offers_the_removed_numba_backend``.

    ``numba`` was removed as a selectable backend name in Tier 6H;
    ``get_backend("numba")`` raises, and ``README.md`` already said so while
    this file still offered it.  Live half of ``DOC-005``.  The replacement
    names the four selectable strategies, says why ``numba`` went, and cites the
    committed benchmark records rather than implying acceleration.
    """
    text = _read("examples/README.md")
    assert "JAX and Numba can be selected" not in text
    assert "Numba" not in _read("README.md")
    for name in ("`numpy`", "`jax`", "`dask`", "`auto`"):
        assert name in text, name
    assert "output/benchmarks/reference/" in text, "backend claim without citation"


def test_examples_readme_lists_all_four_shipped_configurations() -> None:
    """FLIPPED AT 8B.  Was ``..._lists_two_of_the_four_shipped_configurations``.

    The document named ``config.yaml`` and ``realistic_foreground_example.yaml``
    and omitted the hybrid and circular-receptor samples entirely.  All four now
    carry one accurate line, including that
    ``realistic_foreground_example.yaml`` needs network access at simulation
    time -- which 8D then makes the program's own pre-flight say too
    (``SKY-002``).
    """
    text = _read("examples/README.md")
    shipped = _shipped_config_names()
    assert len(shipped) == 4
    assert [name for name in shipped if name in text] == shipped


# ---------------------------------------------------------------------------
# README surface (DOC-004)
# ---------------------------------------------------------------------------


def test_readme_asserts_three_shipped_yaml_samples_against_four_files() -> None:
    """Pins the stale, asserted-rather-than-derived configuration count.

    ``DOC-004``'s original "15+" claim is long gone; what survives is an
    understated literal that no test derives from the filesystem.

    FLIPPED BY: Tier 8E (make the count derived) and the Section 11 scan 3
    acceptance test that keeps it derived.
    """
    assert "[Three shipped YAML samples](configs/)" in _read("README.md")
    assert len(_shipped_config_names()) == 4


# ---------------------------------------------------------------------------
# Sphinx surface (DOC-003, and the -W gate 8C installs)
# ---------------------------------------------------------------------------


def test_sphinx_is_warning_free_and_the_build_is_gated_on_warnings() -> None:
    """FLIPPED AT 8C.  Was ``..._is_not_warning_free_and_is_not_gated_...``.

    Before 8C the build carried sixteen warnings in a clean detached worktree
    and eighteen in a working tree holding the gitignored ``docs/superpowers/``,
    and nothing failed on either.  8C fixed all sixteen at their sources,
    excluded the untracked directory in ``docs/conf.py`` so the two builds are
    now the same build, and made ``-W --keep-going`` the ``docs/Makefile``
    default so ``ci.yml``'s unchanged ``make -C docs html`` inherits the gate.

    The numbers stay recorded rather than rebuilt: a Sphinx build is far too
    slow to sit in the standard suite.  The real proof is the gate itself --
    ``make -C docs clean html`` now exits non-zero on the first warning -- and
    this test pins the wiring that makes that true.

    ``ci.yml`` is deliberately still asserted to pass no ``-W`` of its own.
    That is the whole point of putting the flag in the Makefile default: the
    workflow file is 8D's, and 8C changes no line of it.
    """
    assert SPHINX_WARNING_BASELINE == 0
    assert SPHINX_WARNING_BASELINE_WITH_UNTRACKED_SUPERPOWERS == 0

    makefile = _read("docs/Makefile")
    assert re.search(
        r"^SPHINXOPTS\s*\?=\s*-W\s+--keep-going\s*$", makefile, re.MULTILINE
    ), "docs/Makefile no longer defaults to warnings-as-errors"

    workflow = _read(".github/workflows/ci.yml")
    assert "make -C docs html" in workflow
    assert "-W" not in workflow, "the gate lives in the Makefile default, not here"


def test_the_sixteen_sphinx_warning_sources_are_all_fixed() -> None:
    """FLIPPED AT 8C.  Was ``..._are_all_still_present``.

    Every assertion here is the inverse of the 8A pin, site by site, so the
    sixteen are checkable individually without running a build.  Grouped as
    ``Tier8ReleasePlan.md`` Section 5.3 groups them: ten docutils docstring
    parse errors, one ``toc.not_included``, one unsupported theme option, three
    ``misc.highlighting_failure``, one ``myst.xref_missing``.

    Preserved from here on, not flipped.
    """
    conf = _read("docs/conf.py")
    assert '"display_version": True' not in conf, "unsupported theme option"
    assert "myst_heading_anchors = 3" in conf, "MyST anchors are generated"
    assert "suppress_warnings" not in conf, "nothing is pre-suppressed"
    assert '"superpowers"' in conf, "the gitignored scratch directory is excluded"

    hera = _read("docs/HERA_VSIM_ANALYSIS.md")
    assert ":orphan:" not in hera, "reachable from a toctree, so not an orphan"
    assert "HERA_VSIM_ANALYSIS" in _read("docs/index.rst")
    assert "```csv" not in hera, "no unknown lexer name"
    assert "```python\nFormat: " not in hera, "no data dump annotated python"

    assert "#hybrid-results-and-serialization" in _read("docs/migration_guide.md")

    numpy_backend = _read("src/radiosim/backends/numpy_backend.py")
    assert "\\*operands" in numpy_backend, "the bare star is escaped"
    assert "\\*args" in numpy_backend, "the bare star is escaped"
    assert "``|x|``" in _read("src/radiosim/backends/base.py"), "literal, not a sub"
    polarization = _read("src/radiosim/core/polarization.py")
    power_docstring = polarization.split("def jones_matrix_power(")[1]
    for token in ("``P = |E|²``", "``|J_Xθ|² + |J_Xφ|²``", "``|J_Yθ|² + |J_Yφ|²``"):
        assert token in power_docstring, f"{token} is not a literal"

    backends_init = _read("src/radiosim/backends/__init__.py")
    assert "Backend name:\n\n        - " in backends_init, "bullet list needs a break"


def test_docs_api_covers_the_six_subpackages_and_the_nine_core_modules() -> None:
    """FLIPPED AT 8C.  Was ``..._covers_no_page_for_...``.

    Before 8C, ``SimulationResult``'s own members, the whole ``core.sky``
    subpackage, the solver Strategy classes, ``utils``, ``visualization`` and
    ``core.observability`` rendered nowhere.  Each now has a page, each page is
    reachable from ``docs/index.rst``'s toctree, and the whole set builds clean
    under ``-W``.

    Preserved from here on, not flipped.
    """
    api_pages = sorted((REPO_ROOT / "docs" / "api").glob("*.rst"))
    documented = "\n".join(path.read_text(encoding="utf-8") for path in api_pages)
    covered_subpackages = (
        "radiosim.core.sky",
        "radiosim.simulator",
        "radiosim.core.result",
        "radiosim.utils",
        "radiosim.visualization",
        "radiosim.core.observability",
    )
    for name in covered_subpackages:
        assert f".. automodule:: {name}" in documented, name

    covered_core_modules = (
        "contraction",
        "hybrid",
        "phase_center",
        "polarization_basis",
        "precision",
        "receptor",
        "solver_partition",
        "time_grid",
        "visibility_healpix",
    )
    for module in covered_core_modules:
        assert (REPO_ROOT / "src" / "radiosim" / "core" / f"{module}.py").is_file()
        assert f".. automodule:: radiosim.core.{module}" in documented, module

    index = _read("docs/index.rst")
    for page in api_pages:
        assert f"api/{page.stem}" in index, f"docs/api/{page.name} is in no toctree"


def test_doctests_are_a_real_scoped_invocation_and_not_a_dead_flag() -> None:
    """FLIPPED AT 8B.  Was ``..._is_configured_but_reaches_no_source_file``.

    Before 8B, ``--doctest-modules`` sat in the shared ``addopts`` while
    ``testpaths = ["tests"]`` confined collection to ``tests/``, so it collected
    exactly zero doctest items and the ``>>>`` lines in ``src/radiosim`` had
    never executed.  It is now gone from ``addopts`` -- so a bare
    ``pixi run test`` collects precisely what it collected before -- and the
    ``doctest`` pixi task runs the real thing against the package.  The measured
    debt and how much of it 8B paid are in this module's docstring.

    The CI wiring is deliberately *not* asserted here: adding the step to the
    ``quality`` job is 8D's item 3.

    FLIPPED BY: Tier 8D (the ``ci.yml`` clause below inverts when the step
    lands).
    """
    pyproject = tomllib.loads(_read("pyproject.toml"))
    pytest_config = pyproject["tool"]["pytest"]["ini_options"]
    assert "--doctest-modules" not in pytest_config["addopts"]
    assert pytest_config["testpaths"] == ["tests"]

    pixi_toml = _read("pixi.toml")
    assert 'doctest = "python -m pytest --doctest-modules src/radiosim' in pixi_toml

    listed = subprocess.run(
        ["git", "grep", "-c", ">>>", "--", "src/radiosim"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert listed, "the doctest surface is empty; the task now guards nothing"

    assert "doctest" not in _read(".github/workflows/ci.yml")


# ---------------------------------------------------------------------------
# Agent-facing surface (DOC-006, DOC-007)
# ---------------------------------------------------------------------------


def test_agents_md_carries_its_six_documented_defects() -> None:
    """Pins every live defect in ``AGENTS.md``.

    ``Tier8ReleasePlan.md`` Section 5.4 heads this list "five live defects" and
    then enumerates six bullets; the six are what the file actually carries and
    what 8E must fix, and the plan's summary count was corrected to six when
    this pin was written.  The Hugging Face sentence is ``DOC-007`` and is the
    only surviving reference to a directory removed in ``3266746``; the RRIVis
    naming is the only live one in a tracked prose file outside the historical
    records.

    FLIPPED BY: Tier 8E (all six, plus the ``simulators/`` submodule note).
    """
    text = _read("AGENTS.md")
    assert "`backends/` for NumPy/JAX/Numba execution" in text
    assert "with `unit/`, `integration/`, and `performance/` splits" in text
    assert "The Hugging Face app is isolated in `huggingface_space/`." in text
    assert not (REPO_ROOT / "huggingface_space").exists()
    assert "doctest collection enabled" in text
    assert "`crossval`" not in text
    assert "Until RRIVis reaches a major stable release" in text
    assert text.rstrip().endswith(
        "prefer moving directly to the cleaner replacement unless a deprecation "
        "path is explicitly requested."
    )
    assert "Pre-v1 API Evolution Policy" in _read("docs/contributing.rst")
    assert "simulators/" not in text


def test_claude_md_carries_its_three_documented_defects() -> None:
    """Pins every live defect in ``CLAUDE.md``.

    The type-checker sentence names MyPy while every build and tool file names
    Pyright; the ``io/`` module list names a ``writers.py`` that does not exist
    and omits five modules that do; and the trailing ``TODO`` asks for a
    contributor note that Tier 0 already added.

    FLIPPED BY: Tier 8E.
    """
    text = _read("CLAUDE.md")
    assert "**Type checker**: MyPy" in text
    assert "mypy" not in _read("pixi.toml").lower()
    assert "check_pyright_baseline.py" in _read("pixi.toml")

    assert "`writers.py` / `readers.py` — HDF5/YAML simulation I/O" in text
    io_root = REPO_ROOT / "src" / "radiosim" / "io"
    assert not (io_root / "writers.py").exists()
    for present in ("hdf5.py", "summary_json.py", "standard_visibility.py"):
        assert (io_root / present).is_file()

    assert text.rstrip().endswith(
        "prefer moving directly to the cleaner replacement unless a deprecation "
        "path is explicitly requested."
    )
    assert "simulators/" not in text


# ---------------------------------------------------------------------------
# SKY-002 -- the network pre-flight lies about the shipped recipe config
# ---------------------------------------------------------------------------


def test_realistic_foreground_recipe_declares_no_network_service() -> None:
    """Pins ``SKY-002`` at its registry source and at its shipped-config effect.

    ``LoaderDefinition.network_service`` is singular, so a composite recipe that
    calls two network-backed loaders cannot declare either one; the shipped
    ``configs/realistic_foreground_example.yaml`` therefore reports no required
    service at all, and ``Simulator``'s pre-flight prints "no network-dependent
    models" for a run that makes two real network calls.  The two direct loaders
    the recipe wraps declare their services correctly, which is what makes this
    a metadata gap rather than a missing concept.

    FLIPPED BY: Tier 8D (widen the declaration to ``network_services`` with no
    compatibility shim, declare ``("pygdsm_data", "vizier")`` on the recipe, and
    add the registry-completeness test that stops the next composite recipe from
    repeating it).
    """
    from radiosim.utils.network import get_required_services, get_sky_model_services

    services = get_sky_model_services()
    assert services["gleam"] == "vizier"
    assert services["diffuse_sky"] == "pygdsm_data"
    assert services.get("realistic_foreground") is None

    config = yaml.safe_load(_read("configs/realistic_foreground_example.yaml"))
    sources = config["sky_model"]["sources"]
    assert [entry["kind"] for entry in sources] == ["realistic_foreground"]
    assert get_required_services(config["sky_model"]) == {}

    recipe = _read("src/radiosim/core/sky/recipes/realistic_foreground.py")
    assert "network_service" not in recipe
    assert "_load_diffuse" in recipe and "_load_bright_catalog" in recipe
    assert "network_service: str | None" in _read(
        "src/radiosim/core/sky/registry/core.py"
    )


def test_the_shipped_recipe_config_is_the_one_the_preflight_misreports() -> None:
    """Pins the pre-flight branch the missing metadata selects.

    Kept separate from the registry pin because 8D must flip *both* -- the
    declaration and the user-visible sentence -- and a single test would let one
    of the two land alone.

    FLIPPED BY: Tier 8D.
    """
    simulator_source = _read("src/radiosim/api/simulator.py")
    assert 'print_info(f"Network: {status_label} (no network-dependent models)")' in (
        simulator_source
    )
    assert "required: {', '.join(service_names)}" in simulator_source


# ---------------------------------------------------------------------------
# Packaging surface (DOC-005), and the CI-001 instrumentation this slice landed
# ---------------------------------------------------------------------------


def test_packaging_still_advertises_four_accelerator_extras() -> None:
    """Pins the packaging-level accelerator claim.

    ``pip install radiosim[gpu-cuda]`` installs ``jax[cuda12]`` and delivers a
    package that has never executed on a GPU, whose ``auto`` backend selects JAX
    only when a non-CPU device is present, and whose own README records every
    measured JAX run as slower than NumPy.  ``PERF-001`` stays ``ROADMAP``; the
    extras are the one place the repository still says otherwise.

    FLIPPED BY: Tier 8E (remove the four extras and the ``"gpu"`` keyword,
    offer ``jax`` under one honest extra, and extend
    ``tests/unit/test_release_metadata.py`` to assert their absence while
    ``PERF-001`` is open).
    """
    pyproject = tomllib.loads(_read("pyproject.toml"))
    extras = pyproject["project"]["optional-dependencies"]
    assert {"gpu", "gpu-cuda", "gpu-rocm", "tpu"} <= set(extras)
    assert "gpu" in pyproject["project"]["keywords"]
    assert pyproject["project"]["version"] == "0.2.0"


def test_the_machine_fingerprint_is_now_recorded_on_the_pass_path() -> None:
    """Pins 8A's own ``CI-001`` instrumentation, so a later slice cannot lose it.

    Before 8A, ``_machine_fingerprint()`` was reachable only from
    ``_assert_pinned_digests``'s ``pytest.fail`` branch: the fleet described a
    runner only when that runner *disagreed*, so nothing was ever recorded about
    a machine that produced an accepted digest -- which is the single largest
    reason the ``linux-64-py311`` divergence is undiagnosable.  The record is
    written to gitignored ``output/`` and every failure mode is swallowed, so
    this changes no assertion and no digest.

    Preserved, not flipped.  ``CI-001`` narrows or closes on later evidence; the
    instrumentation that produces that evidence stays.
    """
    from tests.characterization.test_tier6_current_behavior import (
        _machine_fingerprint,
        _record_dir,
        _record_machine_fingerprint,
    )

    _record_machine_fingerprint()
    base = _record_dir()
    assert base is not None
    written = sorted(base.glob("machine-fingerprint-*.txt"))
    assert written, f"no machine fingerprint was recorded in {base}"

    fingerprint = _machine_fingerprint()
    for field in (
        "environment key:",
        "cpu model:",
        "numpy dispatched features:",
        "thread environment:",
        "blas build:",
    ):
        assert field in fingerprint, field


def test_pin_failures_report_a_numeric_delta_when_a_reference_cube_exists() -> None:
    """Pins 8A's ``CI-001`` numeric-delta reporting.

    The gate could not previously tell one ULP from one hundred percent: it
    compared hex digests, and no failing log in the last twenty-five CI runs
    contained a single number.  A failure now reports ``max|dV|``, the maximum
    relative delta, the differing-element count and the first differing index
    against every captured reference cube, and names the nearest recorded
    observation.  A check that supplies no cube -- every call site in
    ``test_tier7_current_behavior.py`` -- behaves exactly as it did before.

    Preserved, not flipped.
    """
    import numpy as np

    from tests.characterization.test_tier6_current_behavior import (
        _assert_pinned_digests,
        _cube_delta,
    )

    reference = np.arange(8, dtype=np.complex128).reshape(2, 2, 2)
    measured = reference.copy()
    measured[1, 1, 1] += 4e-13

    report = _cube_delta(measured, reference)
    assert re.match(r"max\|dV\| = [34]\.\d+e-13,", report), report
    assert "1 of 8 elements differ" in report
    assert "first at index (1, 1, 1)" in report
    assert "identical to the byte" in _cube_delta(reference, reference)
    assert "shape differs" in _cube_delta(reference.ravel(), reference)

    with pytest.raises(pytest.fail.Exception) as failure:
        _assert_pinned_digests(
            ({"nowhere-py0": ("recorded",)}, "an unrecorded environment", "measured"),
        )
    assert "no digest has ever been recorded" in str(failure.value)
    assert "cpu model:" in str(failure.value)
