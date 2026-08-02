"""Tier 8 release-acceptance scans over the tracked documentation surface.

``Tier8ReleasePlan.md`` Section 11 puts every state-2 scan of the tier in one
module rather than eight, because they share a file lister, an allow-list
vocabulary and a failure-message style.  This module is that home.  It is
created at 8B with Section 11 items **3** (shipped-configuration counts are
derived from the filesystem, and every named configuration exists) and **6**
(``examples/README.md`` documents exactly the flags the example script's parser
defines, and vice versa).  8C added scans **1** (no removed name is documented
as live), **4** (every documented relative path resolves) and **5** (every
documented ``radiosim.``-rooted symbol imports), plus the one guard the ``-W``
docs gate depends on; 8D and 8E add the rest.

The prose file set is listed through ``git ls-files --cached --others
--exclude-standard``, not ``Path.rglob``, for the reason
``Tier8ReleasePlan.md`` Section 12 gives: a gitignored stray file must never be
able to fail a repository scan.  ``_tracked_prose_files`` below is a local,
prose-only instance of that discipline; 8D creates the shared
``tests/support/repo_scan.py`` helper and this module's lister becomes a call
into it.

The distinction from ``tests/characterization/test_tier8_current_behavior.py``
is direction.  The characterization module pins drift *as it is* so that a
slice's effect is measurable; this module asserts the *rule* that stops the
drift returning.  A pin is deleted or inverted when its slice lands; a scan
here is permanent.
"""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_SCRIPT = REPO_ROOT / "examples" / "scripts" / "simple_simulation.py"
EXAMPLES_README = REPO_ROOT / "examples" / "README.md"

#: Roots whose tracked ``.rst``/``.md`` files are the documented prose surface.
PROSE_ROOTS = ("docs", "README.md", "AGENTS.md", "CLAUDE.md", "examples/README.md")

#: Documents that record what the project *used* to be. They name removed
#: symbols on purpose and are never edited to remove that history: the
#: changelog, the migration guide, the beam-physics scope disposition, and the
#: historical HERA analysis. ``Fix.md`` and the ``Tier*Plan.md`` documents are
#: outside ``PROSE_ROOTS`` entirely, so they need no entry here.
HISTORICAL_DOCUMENTS = frozenset(
    {
        "docs/changelog.rst",
        "docs/migration_guide.md",
        "docs/development/beam_physics_scope.md",
        "docs/HERA_VSIM_ANALYSIS.md",
    }
)

#: ``--help`` is supplied by ``argparse`` itself, so it is a real flag without
#: being an ``add_argument`` call.  Both directions of the parity scan make the
#: same allowance, exactly as the 8A characterization helper does.
ARGPARSE_SUPPLIED_FLAGS = frozenset({"--help"})

#: Documents whose stated shipped-configuration count this scan derives from
#: the filesystem.  ``README.md`` joins this tuple at 8E, which owns its stale
#: "Three shipped YAML samples" literal; until then that literal is pinned by
#: ``test_readme_asserts_three_shipped_yaml_samples_against_four_files`` in the
#: characterization module, so it is tracked rather than unwatched.
COUNT_CLAIM_DOCUMENTS = ("examples/README.md",)

#: Documents scanned for named ``configs/<name>.yaml`` references.  Every such
#: reference must resolve to a file that exists.
NAMED_CONFIG_DOCUMENTS = ("README.md", "examples/README.md")

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

#: A sentence that is about the shipped configuration directory at all.  Count
#: claims are only looked for inside one of these, so unrelated numerals --
#: "the strict Tier 1 configuration", "200 synthetic point sources" -- are not
#: mistaken for claims about ``configs/``.
_CONFIG_SENTENCE = re.compile(r"[^.\n]*(?:configs/|shipped)[^.\n]*", re.IGNORECASE)

#: A quantifier immediately in front of a noun naming the shipped documents.
#: Anything that is neither a numeral nor a number word (``the``, ``these``,
#: ``several``) is not a count claim and is ignored, which is the escape
#: Section 11 item 3 grants an author who prefers not to state a number.
_QUANTIFIED_NOUN = re.compile(
    r"(\w+)\s+(?:shipped\s+)?(?:YAML\s+)?"
    r"(?:samples?|configurations?|documents?|files?)\b",
    re.IGNORECASE,
)

_NAMED_CONFIG = re.compile(r"configs/([A-Za-z0-9_.-]+\.yaml)")

_FLAG = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")


#: The public names Tiers 1 to 7 removed. Each must be absent from live prose.
#: The twenty-six Jones classes are the removal table at
#: ``docs/migration_guide.md``; the rest are the named removals in
#: ``Tier8ReleasePlan.md`` Section 11 item 1. Matching is whole-word, so
#: ``generate_resolved_baselines``, ``_combine_models``, ``numba_backend`` and
#: ``jones_config.py`` -- all of which are live and correct -- do not match.
REMOVED_NAMES = (
    "GeometricPhaseJones",
    "TimeVariableGainJones",
    "ElevationGainJones",
    "PolynomialBandpassJones",
    "SplineBandpassJones",
    "RFIFlaggedBandpassJones",
    "IXRLeakageJones",
    "MuellerLeakageJones",
    "BeamSquintLeakageJones",
    "FieldRotationJones",
    "VLBIFeedRotationJones",
    "TurbulentIonosphereJones",
    "GPSIonosphereJones",
    "SaastamoinenTroposphereJones",
    "TurbulentTroposphereJones",
    "TroposphericOpacityJones",
    "FaradayRotationJones",
    "DifferentialFaradayJones",
    "WPhaseJones",
    "WProjectionJones",
    "WidefieldPolarimetricJones",
    "ElementBeamJones",
    "ArrayFactorJones",
    "DifferentialBeamJones",
    "FringeFitJones",
    "CrosshandPhaseJones",
    "GeometricDelayJones",
    "BeamJones",
    "AnalyticBeamJones",
    "FITSBeamJones",
    "BeamManager",
    "BeamFITSHandler",
    "AntennaType",
    "generate_baselines",
    "NumbaBackend",
    "numba",
    "calculation_type",
    "combine_models",
    "source_format",
    "available_formats",
)

#: The retirement markers that make naming a removed symbol legitimate. A line
#: carrying one of these is telling a reader the name is *gone*, which is the
#: whole point of a migration note; a line carrying none is presenting it as
#: live API. This is the documented false-positive escape
#: ``Tier8ReleasePlan.md`` Section 20 risk 5 asks each prose scan to provide:
#: to keep a removed name in a document, say in the same line that it was
#: removed, renamed, replaced, rejected, or that it raises.
RETIREMENT_MARKERS = (
    "remove",
    "renam",
    "replac",
    "reject",
    "raises",
    "delet",
    "deprecat",
    "no longer",
    "there is **no**",
    "there is no",
    "has no",
)

_DOTTED_RADIOSIM = re.compile(r"\bradiosim(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
_PY_ROLE = re.compile(r":(?:class|func|meth|mod|attr|data|exc|obj):`~?([^`<>]+?)`")
_MARKDOWN_LINK = re.compile(r"\[[^\]\n]*\]\(([^)\s]+)\)")
_DOC_ROLE = re.compile(r":doc:`[^`<>]*<([^`<>]+)>`|:doc:`([^`<>]+)`")
_MD_PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
_RST_PYTHON_BLOCK = re.compile(r"\.\. code-block:: python\n(.*?)(?=\n\S|\Z)", re.DOTALL)


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _tracked_prose_files() -> list[Path]:
    """Return every tracked-or-unignored ``.rst``/``.md`` file in the prose set.

    Listed through ``git``, never through ``rglob``: a gitignored notebook
    checkpoint, editor backup, or local scratch document must not be able to
    fail a documentation scan. ``docs/superpowers/`` is exactly such a
    directory, and it is why ``git`` rather than the filesystem is the
    authority here.
    """
    listing = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            *PROSE_ROOTS,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted(
        REPO_ROOT / name
        for name in listing.split("\0")
        if name.endswith((".rst", ".md"))
    )


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _resolves(dotted_name: str) -> bool:
    """Return whether a ``radiosim.``-rooted dotted name names a real object.

    The longest importable prefix is imported and the remaining components are
    walked with ``getattr``, so ``radiosim.core.result.SimulationResult`` and
    ``radiosim.Simulator.from_yaml`` are both checked at full depth.
    """
    parts = dotted_name.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except Exception:
            continue
        for attribute in parts[cut:]:
            if not hasattr(obj, attribute):
                return False
            obj = getattr(obj, attribute)
        return True
    return False


def _documented_radiosim_symbols(text: str) -> set[str]:
    """Return every ``radiosim.``-rooted name a document presents as real API.

    Two surfaces, both of which a reader will take literally: python code
    blocks (fenced in Markdown, ``code-block`` in reStructuredText) and the
    python cross-reference roles. Prose backticks are deliberately excluded --
    ``Tier8ReleasePlan.md`` Section 20 risk 5 asks these scans to fail only on
    unambiguous forms, and a name in running prose is routinely a module path
    rather than an attribute chain.
    """
    names: set[str] = set()
    for block in _MD_PYTHON_BLOCK.findall(text) + _RST_PYTHON_BLOCK.findall(text):
        names |= set(_DOTTED_RADIOSIM.findall(block))
    for match in _PY_ROLE.finditer(text):
        candidate = match.group(1).strip()
        if candidate.startswith("radiosim."):
            names.add(candidate)
    return names


def _shipped_config_names() -> list[str]:
    """Return the shipped YAML sample filenames, derived from the filesystem."""
    return sorted(path.name for path in (REPO_ROOT / "configs").glob("*.yaml"))


def _stated_counts(text: str) -> list[tuple[str, int]]:
    """Return every explicit count claim about the shipped configurations."""
    claims: list[tuple[str, int]] = []
    for sentence in _CONFIG_SENTENCE.findall(text):
        for match in _QUANTIFIED_NOUN.finditer(sentence):
            token = match.group(1).lower()
            if token.isdigit():
                claims.append((match.group(0), int(token)))
            elif token in _NUMBER_WORDS:
                claims.append((match.group(0), _NUMBER_WORDS[token]))
    return claims


def _script_parser_flags() -> set[str]:
    """Return every ``--flag`` the example script's ``--help`` output reports.

    Read from ``--help`` rather than from the source, so the scan measures the
    program a reader actually runs.  Section 9's rule is a parity between the
    document and the *live* parser.
    """
    completed = subprocess.run(
        [sys.executable, str(EXAMPLE_SCRIPT), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return set(_FLAG.findall(completed.stdout))


def _example_script_command_blocks() -> list[str]:
    """Return the fenced ``bash`` blocks that invoke the example script."""
    text = EXAMPLES_README.read_text(encoding="utf-8")
    blocks = re.findall(r"```bash\n(.*?)```", text, re.DOTALL)
    return [block for block in blocks if "simple_simulation.py" in block]


def _documented_flags() -> set[str]:
    """Return every ``--flag`` ``examples/README.md`` attributes to the script.

    Two sources, and deliberately not the whole file: the fenced command blocks
    that invoke the script, and inline code spans in the prose.  A blanket scan
    of every ``--token`` in the document would also collect the flags of other
    programs the document legitimately shows -- ``jupyter nbconvert``'s ``--to``
    and ``--execute``, for instance -- and demand that the example's parser
    define them.
    """
    text = EXAMPLES_README.read_text(encoding="utf-8")
    flags: set[str] = set()
    for block in _example_script_command_blocks():
        flags |= set(_FLAG.findall(block))
    for span in re.findall(r"`([^`\n]+)`", text):
        flags |= set(_FLAG.findall(span))
    return flags


# ---------------------------------------------------------------------------
# Section 11 item 6 -- flag parity (DOC-001)
# ---------------------------------------------------------------------------


def test_examples_readme_documents_no_flag_the_script_does_not_define() -> None:
    """Every ``--flag`` in ``examples/README.md`` exists in the live parser.

    This is the scan that would have caught ``DOC-001``: before 8B the document
    printed four copy-pasteable commands using ``--no-plot``, ``--save``,
    ``--plot`` and ``--output-dir``, each of which exits with ``error:
    unrecognized arguments``.
    """
    parser_flags = _script_parser_flags()
    phantom = _documented_flags() - parser_flags - ARGPARSE_SUPPLIED_FLAGS
    assert not phantom, (
        f"examples/README.md documents {sorted(phantom)}, which "
        f"examples/scripts/simple_simulation.py does not define. Either remove "
        f"the flag from the document or add it to the parser; the live parser "
        f"defines {sorted(parser_flags)}."
    )


def test_examples_readme_documents_every_flag_the_script_defines() -> None:
    """Every flag the live parser defines appears in ``examples/README.md``.

    The reverse direction matters as much: a flag nobody documents is a public
    surface nobody can find, and ``Tier8ReleasePlan.md`` Section 9 asks for the
    parity in both directions.
    """
    documented = _documented_flags() | ARGPARSE_SUPPLIED_FLAGS
    undocumented = _script_parser_flags() - documented
    assert not undocumented, (
        f"examples/scripts/simple_simulation.py defines {sorted(undocumented)}, "
        f"which examples/README.md never mentions. Document the flag or remove "
        f"it from the parser."
    )


def test_every_command_examples_readme_prints_uses_only_real_flags() -> None:
    """The copy-pasteable command blocks specifically must be runnable.

    Kept separate from the whole-document scan because a flag named only in
    prose is a weaker defect than one printed inside a ``bash`` block that a
    reader will paste.  This asserts the stronger property on the stronger
    surface.
    """
    blocks = _example_script_command_blocks()
    assert blocks, "examples/README.md prints no runnable command block"

    allowed = _script_parser_flags() | ARGPARSE_SUPPLIED_FLAGS
    for block in blocks:
        used = set(_FLAG.findall(block))
        unknown = used - allowed
        assert not unknown, (
            f"a command block in examples/README.md passes {sorted(unknown)} to "
            f"the example script, which accepts only {sorted(allowed)}:\n{block}"
        )


# ---------------------------------------------------------------------------
# Section 11 item 3 -- configuration counts are derived, names resolve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("document", COUNT_CLAIM_DOCUMENTS)
def test_stated_shipped_configuration_count_matches_the_filesystem(
    document: str,
) -> None:
    """A stated count of shipped configurations must equal the real count.

    The count is parsed out of the prose rather than asserted as a literal, so
    the document cannot drift away from ``configs/`` the way ``README.md:408``
    did.  A document that states no number at all is accepted -- that is the
    deliberate escape in ``Tier8ReleasePlan.md`` Section 11 item 3 -- but a
    document that states one must state the right one.
    """
    expected = len(_shipped_config_names())
    for phrase, stated in _stated_counts(_read(document)):
        assert stated == expected, (
            f"{document} says {phrase!r} but configs/ holds {expected} YAML "
            f"documents ({', '.join(_shipped_config_names())}). Derive the "
            f"count from the directory or restate it."
        )


@pytest.mark.parametrize("document", NAMED_CONFIG_DOCUMENTS)
def test_every_named_shipped_configuration_exists(document: str) -> None:
    """Every ``configs/<name>.yaml`` a document names must exist."""
    missing = sorted(
        {
            name
            for name in _NAMED_CONFIG.findall(_read(document))
            if not (REPO_ROOT / "configs" / name).is_file()
        }
    )
    assert not missing, (
        f"{document} names {missing}, which do not exist. Shipped documents "
        f"are: {', '.join(_shipped_config_names())}."
    )


def test_examples_readme_describes_every_shipped_configuration() -> None:
    """``examples/README.md`` must account for all four shipped documents.

    Before 8B it listed two of four, so a reader had no way to learn that the
    hybrid and circular-receptor samples exist.  The list is checked against the
    directory rather than against a literal, so adding a fifth document to
    ``configs/`` fails this test until the document describes it.
    """
    text = EXAMPLES_README.read_text(encoding="utf-8")
    unlisted = [name for name in _shipped_config_names() if name not in text]
    assert not unlisted, (
        f"examples/README.md does not mention {unlisted}. Every file in "
        f"configs/ needs one accurate line in the 'Shipped configurations' "
        f"section."
    )


def test_the_network_dependent_shipped_configuration_is_flagged_as_such() -> None:
    """The one document that needs network must say so where it is listed.

    ``configs/realistic_foreground_example.yaml`` validates offline and then
    makes two real network calls at simulation time.  8D makes the program's
    own pre-flight say this (``SKY-002``); until then, and afterwards, the
    document that recommends the file must say it too.
    """
    text = EXAMPLES_README.read_text(encoding="utf-8")
    marker = "realistic_foreground_example.yaml"
    assert marker in text
    start = text.index(marker)
    end = text.find("\n- ", start)
    entry = text[start : end if end != -1 else len(text)]
    assert "network" in entry.lower(), (
        "examples/README.md lists configs/realistic_foreground_example.yaml "
        "without saying that simulating it needs network access."
    )


# ---------------------------------------------------------------------------
# Section 11 item 1 -- no removed name is documented as live (DOC-002, DOC-003)
# ---------------------------------------------------------------------------


def test_no_removed_name_is_documented_as_live() -> None:
    """A removed symbol may be named only where the prose says it is gone.

    This is the scan that keeps ``DOC-002`` and ``DOC-003`` closed. Both were
    discharged by prior tiers -- ``generate_baselines`` has no live occurrence
    and every out-of-``__all__`` ``*Jones`` token sits in a captioned historical
    context -- so the Tier 8 work is not to fix them again but to make their
    return detectable.

    A whole-word match is required, so the live ``generate_resolved_baselines``,
    ``_combine_models``, ``numba_backend.py`` and ``io/jones_config.py`` are not
    hits. An occurrence passes if its document is one of the historical records
    or if its own line carries a retirement marker -- "removed", "renamed",
    "replaced", "rejected", "raises", "there is no". Every current occurrence
    is of the second kind: ``README.md`` says ``numba`` is removed, ``CLAUDE.md``
    says ``get_backend("numba")`` raises, and the configuration guide says
    ``visibility`` has no ``calculation_type`` field.
    """
    patterns = {name: re.compile(rf"\b{re.escape(name)}\b") for name in REMOVED_NAMES}
    offences: list[str] = []
    for path in _tracked_prose_files():
        relative = _relative(path)
        if relative in HISTORICAL_DOCUMENTS:
            continue
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            lowered = line.lower()
            if any(marker in lowered for marker in RETIREMENT_MARKERS):
                continue
            for name, pattern in patterns.items():
                if pattern.search(line):
                    offences.append(f"{relative}:{number} presents {name!r} as live")
    assert not offences, (
        "A symbol removed before v1.0 is named in tracked prose without saying "
        "it was removed:\n  "
        + "\n  ".join(sorted(offences))
        + "\nEither delete the reference, or state on the same line that the "
        "name was removed, renamed, replaced, or is rejected -- the migration "
        "guide and the changelog are the documents allowed to name it freely."
    )


# ---------------------------------------------------------------------------
# Section 11 item 4 -- every documented relative path exists
# ---------------------------------------------------------------------------


def test_every_documented_relative_path_exists() -> None:
    """A link a reader can click must land on a file that is really there.

    Covers Markdown inline links and reStructuredText ``:doc:`` roles across
    the whole prose surface. External schemes and pure ``#anchor`` targets are
    skipped; a ``path#anchor`` target is checked for the path. ``:doc:``
    targets are resolved with both documented source suffixes because the docs
    tree carries ``.rst`` and ``.md`` pages side by side.
    """
    missing: list[str] = []
    for path in _tracked_prose_files():
        relative = _relative(path)
        text = path.read_text(encoding="utf-8")
        for match in _MARKDOWN_LINK.finditer(text):
            target = match.group(1)
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = target.split("#", 1)[0]
            if target and not (path.parent / target).exists():
                missing.append(f"{relative} links to {target!r}")
        for match in _DOC_ROLE.finditer(text):
            target = (match.group(1) or match.group(2)).strip()
            base = (
                REPO_ROOT / "docs" / target.lstrip("/")
                if target.startswith("/")
                else path.parent / target
            )
            if not any(base.with_suffix(suffix).exists() for suffix in (".rst", ".md")):
                missing.append(f"{relative} has :doc:`{target}` with no such page")
    assert not missing, (
        "Documented paths that do not resolve:\n  "
        + "\n  ".join(sorted(missing))
        + "\nFix the link or add the file; a documented path is a claim like "
        "any other."
    )


# ---------------------------------------------------------------------------
# Section 11 item 5 -- every documented symbol exists
# ---------------------------------------------------------------------------


def test_every_documented_radiosim_symbol_is_importable() -> None:
    """Every ``radiosim.``-rooted name in a code block or role must resolve.

    This is the mechanical half of ``DOC-001``'s and ``DOC-002``'s closure: a
    document cannot quietly keep calling a function that a refactor renamed,
    because the name is imported here and its attribute chain walked. The
    check is import-and-``getattr`` only -- nothing is executed, so the scan
    stays offline and fast.
    """
    unresolved: list[str] = []
    for path in _tracked_prose_files():
        relative = _relative(path)
        if relative in HISTORICAL_DOCUMENTS:
            continue
        for name in sorted(_documented_radiosim_symbols(path.read_text("utf-8"))):
            if not _resolves(name):
                unresolved.append(f"{relative} documents {name}")
    assert not unresolved, (
        "Documented symbols that do not exist:\n  "
        + "\n  ".join(sorted(unresolved))
        + "\nUpdate the document to the current public name, or move the "
        "example into the migration guide where historical names belong."
    )


# ---------------------------------------------------------------------------
# The guard the -W documentation gate depends on
# ---------------------------------------------------------------------------


def test_the_directory_excluded_from_the_docs_build_is_gitignored() -> None:
    """``docs/superpowers/`` may be excluded only while it is untracked.

    ``docs/conf.py`` drops that directory from the Sphinx source set so a
    contributor who happens to have the local scratch notes does not get a
    *failed* build from ``-W``. That exclusion is safe exactly as long as the
    directory holds nothing tracked; if it ever did, the exclusion would hide a
    real page from the documentation and from the gate. This test is the
    standing proof of the precondition.

    The proof has to hold in a checkout that does *not* have the scratch
    directory, which is every fresh clone, every detached worktree and every CI
    runner. ``.gitignore``'s entry is a directory-only pattern
    (``docs/superpowers/``), and ``git check-ignore`` can only match a
    directory-only pattern against a path that exists on disk -- so asking it
    about an absent path answers "not ignored" for a repository that is
    configured exactly right. The probe below materialises a throwaway file
    under the directory first, so the answer depends on ``.gitignore`` and
    nothing else, and removes whatever it created afterwards.
    """
    assert '"superpowers"' in _read("docs/conf.py"), (
        "docs/conf.py no longer excludes docs/superpowers/; if that is "
        "deliberate, delete this test with it."
    )
    scratch_directory = REPO_ROOT / "docs" / "superpowers"
    probe = scratch_directory / ".gitignore-probe"
    directory_was_created = not scratch_directory.exists()
    try:
        scratch_directory.mkdir(parents=True, exist_ok=True)
        probe.touch()
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", "docs/superpowers/.gitignore-probe"],
            cwd=REPO_ROOT,
            capture_output=True,
        )
    finally:
        probe.unlink(missing_ok=True)
        if directory_was_created:
            with suppress(OSError):
                scratch_directory.rmdir()
    assert ignored.returncode == 0, (
        "docs/superpowers/ is no longer gitignored, so excluding it from the "
        "Sphinx build could hide tracked documentation. Either restore the "
        "ignore entry or remove the exclude_patterns entry in docs/conf.py."
    )
    assert "docs/superpowers/" in [
        line.strip() for line in _read(".gitignore").splitlines()
    ], (
        "docs/superpowers/ is ignored by some broader pattern rather than by "
        "its own .gitignore entry. Restore the explicit entry so the "
        "exclusion in docs/conf.py stays pinned to a deliberate decision."
    )
    tracked = subprocess.run(
        ["git", "ls-files", "--", "docs/superpowers"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert not tracked, (
        f"docs/superpowers/ now tracks files ({tracked.splitlines()}), which "
        f"docs/conf.py excludes from the build. Move them into the documented "
        f"tree or drop the exclusion."
    )
