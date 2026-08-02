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
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The clean-detached-worktree Sphinx warning count at 8A.  Recorded rather than
#: rebuilt: a Sphinx build is far too slow and too environment-sensitive to sit
#: in the standard gate, and ``Tier8ReleasePlan.md`` Section 17 asks 8A for the
#: number, not for the build.  8C drives it to zero and flips this constant.
SPHINX_WARNING_BASELINE = 16

#: The same build in a working tree that contains the gitignored
#: ``docs/superpowers/``.  The difference is an artifact, never a regression.
SPHINX_WARNING_BASELINE_WITH_UNTRACKED_SUPERPOWERS = 18


def _read(relative_path: str) -> str:
    """Return the text of a tracked repository file, for source-truth pins."""
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _script_flags() -> set[str]:
    """Return every ``--flag`` the shipped example script's parser defines."""
    source = _read("examples/scripts/simple_simulation.py")
    return set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', source))


def _documented_flags() -> set[str]:
    """Return every ``--flag`` token ``examples/README.md`` puts in a command.

    ``--help`` is excluded: argparse supplies it, so it is real without being an
    ``add_argument`` call, and 8B's flag-parity test must make the same
    allowance.
    """
    text = _read("examples/README.md")
    found = set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", text))
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


def test_examples_readme_documents_four_flags_the_script_does_not_define() -> None:
    """Pins the phantom flags in the example README.

    Each of the four appears in a copy-pasteable command block that fails with
    ``error: unrecognized arguments``.  This is the live half of ``DOC-001``.

    FLIPPED BY: Tier 8B (correct ``examples/README.md``; add the flag-parity test
    that converts this prose claim from state 2 to state 1).
    """
    phantom = _documented_flags() - _script_flags()
    assert phantom == {"--no-plot", "--save", "--plot", "--output-dir"}


def test_examples_readme_offers_the_removed_numba_backend() -> None:
    """Pins the example README's stale backend sentence.

    ``numba`` was removed as a selectable backend name in Tier 6H;
    ``get_backend("numba")`` raises, and ``README.md`` already says so.  Only
    this file still offers it.  Live half of ``DOC-005``.

    FLIPPED BY: Tier 8B (replace with the live backend set and a pointer to
    ``README.md``'s benchmark-cited backend section).
    """
    text = _read("examples/README.md")
    assert "JAX and Numba can be selected" in text
    assert "Numba" not in _read("README.md")


def test_examples_readme_lists_two_of_the_four_shipped_configurations() -> None:
    """Pins the example README's incomplete configuration list.

    FLIPPED BY: Tier 8B (list all four with one accurate line each, including
    that ``realistic_foreground_example.yaml`` needs network access -- which 8D
    then makes the program's own pre-flight say too).
    """
    text = _read("examples/README.md")
    shipped = _shipped_config_names()
    assert len(shipped) == 4
    named = [name for name in shipped if name in text]
    assert named == ["config.yaml", "realistic_foreground_example.yaml"]


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


def test_sphinx_is_not_warning_free_and_is_not_gated_on_warnings() -> None:
    """Pins the recorded 16-warning baseline and the ungated build.

    The number is recorded, not rebuilt -- see this module's docstring for the
    clean-worktree measurement, the 18-warning in-tree artifact, and the 7H/7I
    ruling that the clean number governs.

    FLIPPED BY: Tier 8C (fix all 16 at source, then set
    ``SPHINXOPTS ?= -W --keep-going`` so ``ci.yml``'s ``make -C docs html``
    inherits the gate; both constants become 0).
    """
    assert SPHINX_WARNING_BASELINE == 16
    assert SPHINX_WARNING_BASELINE_WITH_UNTRACKED_SUPERPOWERS == 18
    assert re.search(r"^SPHINXOPTS\s*\?=\s*$", _read("docs/Makefile"), re.MULTILINE)
    assert "make -C docs html" in _read(".github/workflows/ci.yml")
    assert "-W" not in _read(".github/workflows/ci.yml")


def test_the_sixteen_sphinx_warning_sources_are_all_still_present() -> None:
    """Pins each defect site behind the 16, so 8C's fix is checkable per site.

    FLIPPED BY: Tier 8C (every assertion here inverts as its source is fixed).
    """
    conf = _read("docs/conf.py")
    assert '"display_version": True' in conf, "unsupported theme option"
    assert "myst_heading_anchors" not in conf, "no MyST anchors are generated"
    assert "suppress_warnings" not in conf, "nothing is pre-suppressed"

    hera = _read("docs/HERA_VSIM_ANALYSIS.md")
    assert ":orphan:" not in hera, "tracked, in no toctree, and not an orphan"
    assert "HERA_VSIM_ANALYSIS" not in _read("docs/index.rst")
    assert hera.count("```csv") == 1, "unknown csv lexer"
    assert "```python\nFormat: HDF5" in hera, "data dump annotated python"
    assert "```python\nFormat: UVBeam FITS" in hera, "data dump annotated python"

    assert "#hybrid-results-and-serialization" in _read("docs/migration_guide.md")

    numpy_backend = _read("src/radiosim/backends/numpy_backend.py")
    assert "*operands" in numpy_backend, "bare star read as emphasis"
    assert "|x|" in _read("src/radiosim/backends/base.py"), "unknown substitution"
    polarization = _read("src/radiosim/core/polarization.py")
    power_docstring = polarization.split("def jones_matrix_power(")[1]
    for token in ("|E|", "|J_Xθ|", "|J_Xφ|", "|J_Yθ|", "|J_Yφ|"):
        assert token in power_docstring, f"{token} is read as a substitution"


def test_docs_api_covers_no_page_for_six_subpackages_and_nine_core_modules() -> None:
    """Pins the API-reference gap.

    ``docs/api/`` documents backends, benchmarks, part of ``core/``, ``io/``,
    the Jones modules and ``Simulator``, and nothing else -- so
    ``SimulationResult``'s own members, the whole ``core.sky`` subpackage, and
    the solver Strategy classes render nowhere.

    FLIPPED BY: Tier 8C (add the pages, each reachable from ``docs/index.rst``'s
    toctree, each building clean under ``-W``).
    """
    documented = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((REPO_ROOT / "docs" / "api").glob("*.rst"))
    )
    uncovered_subpackages = (
        "radiosim.core.sky",
        "radiosim.simulator",
        "radiosim.core.result",
        "radiosim.utils",
        "radiosim.visualization",
        "radiosim.core.observability",
    )
    for name in uncovered_subpackages:
        assert f".. automodule:: {name}" not in documented, name

    uncovered_core_modules = (
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
    for module in uncovered_core_modules:
        assert (REPO_ROOT / "src" / "radiosim" / "core" / f"{module}.py").is_file()
        assert f".. automodule:: radiosim.core.{module}" not in documented, module


def test_doctest_collection_is_configured_but_reaches_no_source_file() -> None:
    """Pins ``--doctest-modules`` as dead configuration.

    The flag is set, ``testpaths`` confines collection to ``tests/``, and the
    299 ``>>>`` lines in 22 tracked ``src/radiosim`` docstrings therefore never
    execute.  Nothing in ``tests/`` or ``.github/`` mentions doctest either, so
    the coverage the flag implies does not exist anywhere.

    FLIPPED BY: Tier 8B (remove the flag from the shared ``addopts``, add a
    ``doctest`` pixi task scoped to ``src/radiosim``, and fix what it surfaces).
    """
    pyproject = tomllib.loads(_read("pyproject.toml"))
    pytest_config = pyproject["tool"]["pytest"]["ini_options"]
    assert "--doctest-modules" in pytest_config["addopts"]
    assert pytest_config["testpaths"] == ["tests"]

    listed = subprocess.run(
        ["git", "grep", "-c", ">>>", "--", "src/radiosim"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert len(listed) == 22
    assert sum(int(line.rsplit(":", 1)[1]) for line in listed) == 299

    assert "doctest" not in _read(".github/workflows/ci.yml")
    assert "doctest" not in _read("pixi.toml")


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
