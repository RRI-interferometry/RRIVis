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

Tier 8D addendum -- what the scans now list, and what CI now executes
====================================================================

**Every repository scan is git-scoped.** ``tests/support/repo_scan.py`` is the
one lister; the twenty call sites of ``Tier8ReleasePlan.md`` Section 12's table
call it, as does this tier's acceptance module's prose lister.  The proof that
this changed the right thing lives in
``tests/unit/test_utils/test_repo_scan.py``: the gitignored
``src/radiosim/.ipynb_checkpoints`` copy that ``Fix.md`` Section 17 item 15
demonstrated could fail two Tier 7 acceptance scans is materialised, both scans
are run against it and pass, and then *the same file* is forced into their
lister and both fail -- so the listing, not a weakened assertion, is why they
pass.  The two ``tmp_path`` rglobs Section 12 excludes are untouched.

**``SKY-002`` is closed.** ``LoaderDefinition.network_service: str | None``
became ``network_services: tuple[str, ...]`` with no compatibility shim, and
``realistic_foreground`` declares both of the services it dispatches to.  The
shipped ``configs/realistic_foreground_example.yaml`` now reports
``Network: offline (forced) (required: pygdsm data, VizieR)``, measured by
running it offline rather than by reading the branch.

**CI executes what the documentation claims.** The ``quality`` job gained the
doctest, example-script and notebook steps, and a working-tree-cleanliness
check after them; the strict docs gate is inherited from ``docs/Makefile``'s
default rather than restated in the workflow, which is why this module still
asserts the workflow passes no Sphinx option of its own.  The
``compatibility`` job gained the ``CI-001`` evidence path: the machine
fingerprint is printed into the job log, ``output/characterization/`` is
uploaded as a per-cell artifact, and the previous successful run's artifact for
the same cell is restored before the tests, so a divergent digest can report a
numeric delta against a cube measured while its digest still matched.

**``CI-001`` at this slice: no recurrence yet, so no measured decision.**  The
instrumentation landed at 8A (``47822a2``).  Every CI run since -- 8A's
``30735775142``, 8B's ``30737560005`` and 8C's ``30741637024`` -- was green on
all eight jobs, including ``linux-64 / Python 3.11``, so the divergence has not
recurred *with* the new evidence attached and Section 14's conditional has
nothing to read.  No digest table grew or shrank by a character at 8D, and
``CI-001`` stays ``OPEN`` exactly as the plan requires when the measurement has
not been taken.  On a ~38% per-run recurrence rate, three green runs is
unremarkable and is not evidence the divergence is gone.

Tier 8E addendum -- the four residual pins, and what the bump did not move
=========================================================================

The pins this module was created to carry are now all flipped.  8E took the
last four: ``README.md``'s "Three shipped YAML samples" against a directory of
four, ``AGENTS.md``'s six defects, ``CLAUDE.md``'s three, and the four
accelerator extras in ``pyproject.toml``.  Each flipped test names what it used
to assert, so the transition stays readable after the drift is gone, and each
rule that keeps the corrected state true lives in
``tests/unit/test_tier8_release_acceptance.py`` (Section 11 scans 2, 3, 7 and
8) or in ``tests/unit/test_release_metadata.py``, not here.

**The version bump moved provenance and nothing else, measured rather than
argued.**  ``Tier8ReleasePlan.md`` Section 16 verified by reading
``core/result.py`` that the package version is hashed into
``provenance_sha256`` (``:857``) and never into ``scientific_sha256``
(``:789-841``).  8E ran all three hermetic shipped configurations immediately
before and immediately after changing the five metadata sources, on one
machine in one environment.  ``scientific_sha256`` is byte-identical across the
bump for all three -- ``config.yaml``
``4bbb7403...b947f2b``, ``receptor_circular_example.yaml``
``be1e86fb...d042203``, ``hybrid_sky_example.yaml`` ``65777dee...ec3808a`` --
and all three ``provenance_sha256`` values changed.  That is the intended
asymmetry: a release-metadata change must be invisible to the scientific
fingerprint and visible to the provenance one, and no characterization pin in
this suite moved.
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


def test_readme_states_the_real_shipped_configuration_count_and_names() -> None:
    """FLIPPED AT 8E.  Was ``..._asserts_three_shipped_yaml_samples_...``.

    ``DOC-004``'s original "15+" claim was long gone by 8A; what survived was an
    understated "Three shipped YAML samples" literal against a directory holding
    four, which no test derived from the filesystem.  8E restates it and names
    every shipped document, and the count is now parsed out of the prose and
    compared with ``configs/`` by
    ``tests/unit/test_tier8_release_acceptance.py`` (Section 11 scan 3), which
    is what stops it drifting again -- this pin only records that the literal
    moved.
    """
    text = _read("README.md")
    assert "[Three shipped YAML samples](configs/)" not in text
    assert "Four shipped YAML samples" in text
    assert len(_shipped_config_names()) == 4
    unnamed = [name for name in _shipped_config_names() if name not in text]
    assert not unnamed, f"README.md no longer names {unnamed}"


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

    FLIPPED AT 8D for its last clause: the ``quality`` job now runs
    ``pixi run doctest``, so the assertion that ``ci.yml`` mentions no doctest
    at all inverts into the assertion that it runs one.  A task nothing
    executes is the state this whole tier exists to end.

    Preserved from here on, not flipped.
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

    assert "run: pixi run doctest" in _read(".github/workflows/ci.yml")


# ---------------------------------------------------------------------------
# CI shape (DOC-008), and the CI-001 evidence path
# ---------------------------------------------------------------------------


def test_ci_executes_every_documented_example_surface() -> None:
    """FLIPPED AT 8D.  Was the ``ci.yml`` clauses of the 8A/8B pins above.

    ``Tier8ReleasePlan.md`` Section 5.5 lists the gaps this closes: no CI step
    executed ``examples/scripts/simple_simulation.py`` (``Fix.md`` Section 17
    item 2), no notebook validation existed anywhere, and no doctest ran.  All
    three are now steps in the ``quality`` job, which is what turns 8B's
    "these documents execute" from a claim into a gate.  Q5 was decided at 8B:
    the notebook is executed, because ``jupyter``/``nbconvert``/``ipykernel``
    are already in the default environment and the notebook is fully offline.

    Preserved from here on, not flipped.
    """
    workflow = _read(".github/workflows/ci.yml")
    for command in (
        "pixi run doctest",
        "python examples/scripts/simple_simulation.py --help",
        "python examples/scripts/simple_simulation.py\n",
        "jupyter nbconvert --to notebook --execute --stdout",
        "examples/notebooks/01_basic_usage.ipynb",
        "make -C docs html",
    ):
        assert command in workflow, command

    assert (REPO_ROOT / "examples" / "scripts" / "simple_simulation.py").is_file()
    assert (REPO_ROOT / "examples" / "notebooks" / "01_basic_usage.ipynb").is_file()


def test_ci_surfaces_the_ci_001_evidence_that_the_runner_would_discard() -> None:
    """Pins the CI-side half of ``Tier8ReleasePlan.md`` Section 14 item 2.

    8A made ``_machine_fingerprint()`` emit on the pass path and made a failing
    pin report a numeric delta against a captured reference cube.  Both land
    under gitignored ``output/characterization/``, which a GitHub Actions
    runner throws away when the job ends -- so on CI the evidence existed and
    was then destroyed.  The ``compatibility`` job now prints the fingerprint
    into the job log, uploads the directory as a per-cell artifact, and
    restores the previous successful run's artifact *before* the tests, so a
    divergent cell has an accepted-digest cube to subtract from.

    Preserved, not flipped.  ``CI-001`` narrows or closes on the evidence this
    wiring produces; the wiring stays.
    """
    workflow = _read(".github/workflows/ci.yml")
    assert "machine-fingerprint-" in workflow, "the fingerprint is never printed"
    assert "actions/upload-artifact@v4" in workflow
    assert "output/characterization/" in workflow
    assert "gh run download" in workflow, "no reference cube is ever restored"
    assert "actions: read" in workflow, "gh run download needs the actions scope"

    from tests.characterization.test_tier6_current_behavior import (
        _MEASURED_ENVIRONMENTS,
        _RECORD_DIR_ENV,
    )

    # The artifact is named for the matrix cell's ``key``, and those keys are
    # spelled exactly like the pins' own environment keys -- so an artifact
    # downloaded from a failing run can be matched to the digest table it
    # disagreed with, and no cell can restore another cell's cubes.
    assert "characterization-${{ matrix.key }}" in workflow
    declared = set(re.findall(r"^\s+key: (\S+)$", workflow, re.MULTILINE))
    assert declared == set(_MEASURED_ENVIRONMENTS), (
        f"the workflow matrix keys {sorted(declared)} are not the six "
        f"characterized environments {sorted(_MEASURED_ENVIRONMENTS)}"
    )

    assert _RECORD_DIR_ENV not in workflow, (
        "the record directory is left at its default, so the uploaded path and "
        "the written path cannot drift apart"
    )


# ---------------------------------------------------------------------------
# Agent-facing surface (DOC-006, DOC-007)
# ---------------------------------------------------------------------------


def test_agents_md_carries_none_of_its_six_documented_defects() -> None:
    """FLIPPED AT 8E.  Was ``..._carries_its_six_documented_defects``.

    ``Tier8ReleasePlan.md`` Section 5.4 heads the list "five live defects" and
    then enumerates six bullets; the six are what the file carried, and the
    plan's summary count was corrected to six when this pin was written at 8A.
    All six are fixed here: the removed ``numba`` backend name, the
    three-of-seven test-directory list, the Hugging Face sentence (``DOC-007``,
    the last surviving reference to a directory removed in ``3266746``), the
    "doctest collection enabled" claim against an inert flag, the ``RRIVis``
    naming, and the ``TODO`` that ``docs/contributing.rst:46-50`` discharged in
    Tier 0.  The ``simulators/`` sentence Section 17 item 1 asks for is here
    too, because 41 unfetched submodules are the single most surprising fact
    about this working tree.
    """
    text = _read("AGENTS.md")
    assert "Numba" not in text
    assert "`backends/` for NumPy/JAX/Dask execution" in text
    for directory in ("unit/", "integration/", "characterization/", "performance/"):
        assert f"`{directory}`" in text
    assert "crossvalidation/" in text
    assert "huggingface" not in text.lower()
    assert not (REPO_ROOT / "huggingface_space").exists()
    assert "doctest collection enabled" not in text
    assert "pixi run doctest" in text
    assert "`crossval`" in text
    assert "RRIVis" not in text
    assert "Until RadioSim reaches a major stable release" in text
    assert "TODO:" not in text
    assert "Pre-v1 API Evolution Policy" in _read("docs/contributing.rst")
    assert "simulators/" in text
    assert "does not fetch them" in text


def test_claude_md_carries_none_of_its_three_documented_defects() -> None:
    """FLIPPED AT 8E.  Was ``..._carries_its_three_documented_defects``.

    The type-checker sentence named MyPy while every build and tool file names
    Pyright; the ``io/`` module list named a ``writers.py`` that does not exist
    and omitted five modules that do; and the trailing ``TODO`` asked for a
    contributor note Tier 0 had already added.  All three are fixed, and the
    ``simulators/`` note Section 17 item 2 asks for is here as well.

    The ``io/`` half is checked against the directory rather than against a
    literal, so a future module that the list omits fails this pin.
    """
    text = _read("CLAUDE.md")
    assert "**Type checker**: MyPy" not in text
    assert "Type check (mypy)" not in text
    #: MyPy is still *named* once, to say it is not used -- the same
    #: retirement-marker escape the acceptance module's removed-name scan
    #: grants.  What must not return is a sentence presenting it as the
    #: repository's checker.
    assert "MyPy is not used anywhere in this repository" in text
    assert "Pyright in strict mode" in text
    assert "tools/check_pyright_baseline.py" in text
    assert "mypy" not in _read("pixi.toml").lower()
    assert "check_pyright_baseline.py" in _read("pixi.toml")

    assert "`writers.py` / `readers.py` — HDF5/YAML simulation I/O" not in text
    assert "There is no `io/writers.py`" in text
    io_root = REPO_ROOT / "src" / "radiosim" / "io"
    assert not (io_root / "writers.py").exists()
    private = {"__init__.py", "model_base.py", "result_errors.py"}
    for module in sorted(io_root.glob("*.py")):
        if module.name in private:
            continue
        assert f"`{module.name}`" in text or f"/ `{module.name}`" in text, (
            f"CLAUDE.md's io/ list omits {module.name}"
        )

    assert "TODO:" not in text
    assert "simulators/" in text
    assert "does not fetch them" in text


# ---------------------------------------------------------------------------
# SKY-002 -- the network pre-flight lies about the shipped recipe config
# ---------------------------------------------------------------------------


def test_realistic_foreground_recipe_declares_both_network_services() -> None:
    """FLIPPED AT 8D.  Was ``..._declares_no_network_service``.

    Before 8D, ``LoaderDefinition.network_service`` was singular, so a composite
    recipe that dispatches to two network-backed loaders could declare neither:
    the shipped ``configs/realistic_foreground_example.yaml`` reported no
    required service at all.  The field is now
    ``network_services: tuple[str, ...]`` with no compatibility shim -- the
    singular spelling is gone from the package -- and the recipe declares
    ``("pygdsm_data", "vizier")``, the exact tokens the two catalog entries it
    reaches declare in ``registry/catalogs.py``.

    Preserved from here on, not flipped.  The rule that keeps it true is
    ``tests/unit/test_core/test_sky_registry.py``'s completeness scan: a loader
    whose module names a network client, or resolves other loaders dynamically,
    must declare at least one service.
    """
    from radiosim.utils.network import get_required_services, get_sky_model_services

    services = get_sky_model_services()
    assert services["gleam"] == ("vizier",)
    assert services["diffuse_sky"] == ("pygdsm_data",)
    assert services["realistic_foreground"] == ("pygdsm_data", "vizier")

    config = yaml.safe_load(_read("configs/realistic_foreground_example.yaml"))
    sources = config["sky_model"]["sources"]
    assert [entry["kind"] for entry in sources] == ["realistic_foreground"]
    assert get_required_services(config["sky_model"]) == {
        "pygdsm_data": ["realistic_foreground"],
        "vizier": ["realistic_foreground"],
    }

    recipe = _read("src/radiosim/core/sky/recipes/realistic_foreground.py")
    assert 'network_services=("pygdsm_data", "vizier")' in recipe
    assert "_load_diffuse" in recipe and "_load_bright_catalog" in recipe
    registry_core = _read("src/radiosim/core/sky/registry/core.py")
    assert "network_services: tuple[str, ...] = ()" in registry_core
    assert "network_service:" not in registry_core, "no singular shim survives"


def test_the_shipped_recipe_config_is_the_one_the_preflight_now_reports() -> None:
    """FLIPPED AT 8D.  Was ``..._is_the_one_the_preflight_misreports``.

    Kept separate from the registry pin because 8D had to flip *both* -- the
    declaration and the user-visible sentence -- and a single test would have
    let one of the two land alone.  The source-level half is here; that the
    branch is really taken for the shipped document, and that an offline run of
    it fails with the actionable offline error rather than a network attempt,
    is asserted by running it in
    ``tests/unit/test_utils/test_network.py::TestShippedRecipeConfigurationServices``.
    """
    from radiosim.utils.network import get_required_services

    simulator_source = _read("src/radiosim/api/simulator.py")
    assert 'print_info(f"Network: {status_label} (no network-dependent models)")' in (
        simulator_source
    )
    assert "required: {', '.join(service_names)}" in simulator_source

    config = yaml.safe_load(_read("configs/realistic_foreground_example.yaml"))
    assert get_required_services(config["sky_model"]), (
        "the recipe config must select the network-dependent pre-flight branch"
    )


# ---------------------------------------------------------------------------
# Packaging surface (DOC-005), and the CI-001 instrumentation this slice landed
# ---------------------------------------------------------------------------


def test_packaging_advertises_no_accelerator_extra() -> None:
    """FLIPPED AT 8E.  Was ``..._still_advertises_four_accelerator_extras``.

    ``pip install radiosim[gpu-cuda]`` installed ``jax[cuda12]`` and delivered a
    package that has never executed on a GPU, whose ``auto`` backend selects JAX
    only when a non-CPU device is present, and whose own README records every
    measured JAX run as slower than NumPy.  ``PERF-001`` stays ``ROADMAP``, and
    the extras were the one place the repository still said otherwise.  8E
    removed the four extras and the ``"gpu"`` keyword and offers the library
    under a single ``jax`` extra.

    The standing rule lives in
    ``tests/unit/test_release_metadata.py::test_no_accelerator_named_extra_is_published``;
    this pin records the transition and, additionally, that the five documents
    and error messages that told a reader to install those extras were
    corrected in the same change rather than left pointing at names that no
    longer resolve.
    """
    pyproject = tomllib.loads(_read("pyproject.toml"))
    extras = pyproject["project"]["optional-dependencies"]
    assert not {"gpu", "gpu-cuda", "gpu-rocm", "tpu"} & set(extras)
    assert "jax" in extras
    assert "gpu" not in pyproject["project"]["keywords"]
    assert extras["all"] == ["radiosim[jax,dask,ms,dev,docs]"]

    for document in (
        "README.md",
        "docs/installation.rst",
        "docs/user_guide/backends.rst",
        "src/radiosim/backends/__init__.py",
        "src/radiosim/backends/jax_backend.py",
    ):
        text = _read(document)
        for removed in ("radiosim[gpu]", "radiosim[gpu-cuda]", "radiosim[tpu]"):
            assert f"pip install {removed}" not in text, (
                f"{document} still tells a reader to install {removed}"
            )


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
