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
docs gate depends on.  8E closed the set with scans **2** (the pre-rename
project name appears in no tracked file, binary included), **7** (no
accelerator or speed claim without a citation, over the prose *and* the
package's own docstrings) and **8** (every documented ``pixi run`` task
exists), and extended scan 3 to ``README.md``.

The prose file set is listed through ``git ls-files --cached --others
--exclude-standard``, not ``Path.rglob``, for the reason
``Tier8ReleasePlan.md`` Section 12 gives: a gitignored stray file must never be
able to fail a repository scan.  ``_tracked_prose_files`` below was a local,
prose-only instance of that discipline until 8D created the shared
``tests/support/repo_scan.py`` helper; it is now the prose root and suffix
selection over :func:`tests.support.repo_scan.iter_tracked_files`, which the
suite's other twenty repository scans share.

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
import tomllib
from contextlib import suppress
from pathlib import Path

import pytest

from tests.support.repo_scan import iter_tracked_files

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_SCRIPT = REPO_ROOT / "examples" / "scripts" / "simple_simulation.py"
EXAMPLES_README = REPO_ROOT / "examples" / "README.md"

#: Roots whose tracked ``.rst``/``.md`` files are the documented prose surface.
PROSE_ROOTS = ("docs", "README.md", "AGENTS.md", "CLAUDE.md", "examples/README.md")

#: The prose suffixes those roots are scanned for.
PROSE_SUFFIXES = frozenset({".rst", ".md"})

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
#: the filesystem.  ``README.md`` joined this tuple at 8E, which corrected its
#: stale "Three shipped YAML samples" literal.
COUNT_CLAIM_DOCUMENTS = ("README.md", "examples/README.md")

#: Documents that must account for every file in ``configs/`` by name, so a
#: fifth shipped sample cannot appear undocumented.
CONFIG_INVENTORY_DOCUMENTS = ("README.md", "examples/README.md")

#: The project's pre-rename name, in every casing.  Section 11 scan 2 forbids
#: it in any tracked file, binary included -- the FITS ``COMMENT`` card in
#: ``antenna_layout_examples/1101503312_metafits.fits`` was the last non-prose
#: instance and 8E rewrote it in place.
STALE_PROJECT_NAME = re.compile(rb"rrivis", re.IGNORECASE)

#: Files allowed to name the old project. The register and the tier plans are
#: historical records that are never edited (``Tier8ReleasePlan.md`` Section
#: 15.4 allow-lists ``Fix.md`` and ``Tier3BeamObservabilityPlan.md:3223,3486``
#: by name); the two Tier 8 test modules have to spell the name to assert its
#: absence.
STALE_NAME_ALLOW_LIST = frozenset(
    {
        "Fix.md",
        "tests/unit/test_tier8_release_acceptance.py",
        "tests/characterization/test_tier8_current_behavior.py",
    }
)

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

#: A capability claim about an accelerator: a device name tied to a verb of
#: support, provision or achievement, or a speed multiplier. Deliberately
#: *not* the bare ``gpu``/``tpu`` token, per the ruling recorded at the Tier 8C
#: independent re-acceptance and quoted in ``Tier8ReleasePlan.md`` Section 17
#: item 7: ``radiosim.backends : Backend abstraction for CPU/GPU`` is a
#: cross-reference naming another module's scope, and ``jax.devices("gpu")``,
#: ``device_kind == "gpu"`` and the whole of ``utils/device.py`` are execution
#: facts about the machine. None of them claims this package achieved
#: anything. "GPU acceleration via JAX backend" does.
ACCELERATOR_CLAIM = re.compile(
    # "GPU acceleration", "TPU-accelerated"
    r"\b(?:gpu|tpu)s?\b[^.\n]{0,30}?\baccelerat\w+"
    # "acceleration via JAX", "accelerated on a GPU"
    r"|\baccelerat\w+[^.\n]{0,30}?\b(?:via|on|with)\b[^.\n]{0,25}?"
    r"\b(?:gpu|tpu|jax|cuda|rocm|metal)\b"
    # "supports GPU", "GPU support"
    r"|\bsupports?\b[^.\n]{0,25}?\b(?:gpu|tpu)\b"
    r"|\b(?:gpu|tpu)\b[^.\n]{0,15}?\bsupport(?:ed|s)?\b"
    # "can use GPU backends", "will run on a GPU"
    r"|\b(?:can|will|does)\b[^.\n]{0,25}?\b(?:use|run on|execute on)\b"
    r"[^.\n]{0,20}?\b(?:gpu|tpu)s?\b"
    # "10-100x faster", "3× faster"
    r"|\d\s*(?:[-–]\s*\d+\s*)?(?:x|×)\s*(?:times\s+)?faster",
    re.IGNORECASE,
)

#: The prose surface additionally forbids an uncited bare speed claim. Source
#: docstrings do not: "purely a wall-clock speedup" describing why one code
#: path exists is an implementation note, while the same words on a
#: documentation page are a promise to a reader.
PROSE_SPEED_CLAIM = re.compile(
    ACCELERATOR_CLAIM.pattern + r"|\bspeedups?\b|\bfaster\s+than\b",
    re.IGNORECASE,
)

#: A paragraph carrying an accelerator claim must cite the committed records or
#: name the open register row. This is the enforceable form of ``CLAUDE.md``'s
#: standing rule: never write a speed or GPU claim without citing a record.
CLAIM_CITATIONS = ("output/benchmarks/reference", "PERF-001")

#: Phrases that make a paragraph a *denial* rather than a claim. The rule is
#: "no claim without evidence"; "RadioSim publishes no GPU performance number,
#: because none has been measured" is the evidence position stated in words,
#: and demanding a citation beside it would be demanding a citation for the
#: absence of a claim. Matched per paragraph, not per line, because the denial
#: and the device name it denies are routinely on different lines of the same
#: sentence.
CLAIM_DENIALS = (
    "no accelerator",
    "publishes no",
    "does not establish",
    "does not promise",
    "does not claim",
    "requires a real accelerator run",
    "unmeasured",
    "measured no accelerator",
    "never been measured",
    "has measured none",
)

#: Roots whose Python sources are scanned for accelerator claims.
#: ``tests/`` is excluded: a test that asserts an error message contains the
#: word "GPU" is quoting the package, not claiming anything.
CLAIM_SOURCE_ROOTS = ("src/radiosim",)

#: ``pixi run <token>`` strings whose token is an executable in the environment
#: rather than a declared task. ``pixi run`` accepts both, so the scan must too.
ENVIRONMENT_COMMANDS = frozenset({"python", "radiosim", "jupyter", "pytest"})

#: Options that consume the argument after them, so the scan does not mistake
#: an environment name for a task name.
PIXI_OPTIONS_WITH_VALUE = frozenset({"--environment", "-e", "--manifest-path"})

#: Phrases that mark a documented command as one that deliberately does *not*
#: work -- ``CLAUDE.md`` warns that ``pixi run pytest`` is not on the task list,
#: which is guidance, not a broken instruction.
NEGATED_COMMAND_MARKERS = ("does not work", "does NOT work", "not directly on")

_PIXI_RUN = re.compile(r"pixi run\s+(.*)")
_TASK_TOKEN = re.compile(r"[a-z][\w-]*\Z")

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

    Since 8D the listing itself is :func:`tests.support.repo_scan.iter_tracked_files`,
    the one shared implementation of that discipline
    (``Tier8ReleasePlan.md`` Section 12); this function is now only the prose
    root and suffix selection.
    """
    return iter_tracked_files(
        *(REPO_ROOT / root for root in PROSE_ROOTS),
        suffixes=PROSE_SUFFIXES,
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


@pytest.mark.parametrize("document", CONFIG_INVENTORY_DOCUMENTS)
def test_the_document_describes_every_shipped_configuration(document: str) -> None:
    """The inventory documents must account for every shipped sample by name.

    Before 8B ``examples/README.md`` listed two of four, and before 8E
    ``README.md`` said "Three shipped YAML samples" against a directory holding
    four -- so a reader had no way to learn that the hybrid and
    circular-receptor samples exist.  Both lists are checked against the
    directory rather than against a literal, so adding a fifth document to
    ``configs/`` fails this test until both documents describe it.
    """
    text = _read(document)
    unlisted = [name for name in _shipped_config_names() if name not in text]
    assert not unlisted, (
        f"{document} does not mention {unlisted}. Every file in configs/ needs "
        f"one accurate line in that document's shipped-configuration list."
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
# Section 11 item 2 -- no stale project naming (DOC-006)
# ---------------------------------------------------------------------------


def _allowed_to_name_the_old_project(relative: str) -> bool:
    """Return whether a file may spell the pre-rename project name.

    Two kinds qualify: the historical records (``Fix.md`` and the tier plans,
    which describe what was observed at the time and are never edited), and the
    two Tier 8 test modules, which have to write the name down to assert that
    nothing else does.
    """
    return relative in STALE_NAME_ALLOW_LIST or bool(
        re.fullmatch(r"Tier\w*Plan\.md", relative)
    )


def test_no_tracked_file_carries_the_pre_rename_project_name() -> None:
    """The old project name survives only in the historical records.

    Deliberately a *byte* scan over every tracked-or-unignored file, not a
    prose scan: the last non-prose instance was a FITS ``COMMENT`` card inside
    ``antenna_layout_examples/1101503312_metafits.fits``, which no
    text-file-only scan would have seen. 8E rewrote that card in place (an
    80-byte record replaced by an 80-byte record, so the file length and every
    other card are unchanged) and corrected ``AGENTS.md``'s API-evolution
    sentence, which was the last prose instance.
    """
    offenders: list[str] = []
    for path in iter_tracked_files():
        relative = _relative(path)
        if _allowed_to_name_the_old_project(relative):
            continue
        if STALE_PROJECT_NAME.search(path.read_bytes()):
            offenders.append(relative)
    assert not offenders, (
        "The pre-rename project name appears in:\n  "
        + "\n  ".join(sorted(offenders))
        + "\nThe project is RadioSim. Only Fix.md, the tier plans, and the two "
        "Tier 8 test modules may spell the old name, and they do it to record "
        "history rather than to describe the package."
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
# Section 11 item 7 -- no accelerator claim without a citation (DOC-005)
# ---------------------------------------------------------------------------


def _paragraphs(text: str) -> list[tuple[int, str]]:
    """Split text into blank-line-separated blocks, with 1-based start lines."""
    blocks: list[tuple[int, str]] = []
    line_number = 1
    for block in re.split(r"\n[ \t]*\n", text):
        blocks.append((line_number, block))
        line_number += block.count("\n") + 2
    return blocks


def _uncited_accelerator_claims(path: Path, pattern: re.Pattern[str]) -> list[str]:
    """Return one message per paragraph that claims acceleration without proof."""
    relative = _relative(path)
    offences: list[str] = []
    for start, block in _paragraphs(path.read_text(encoding="utf-8")):
        if not pattern.search(block):
            continue
        if any(citation in block for citation in CLAIM_CITATIONS):
            continue
        lowered = block.lower()
        if any(denial in lowered for denial in CLAIM_DENIALS):
            continue
        for offset, line in enumerate(block.splitlines()):
            if pattern.search(line):
                offences.append(f"{relative}:{start + offset}: {line.strip()}")
    return offences


def test_no_accelerator_claim_in_tracked_prose_lacks_a_citation() -> None:
    """Every GPU or speed claim in the documentation must cite its evidence.

    ``CLAUDE.md`` states the rule -- never write a speed or GPU claim without
    citing a record file -- and this is its enforceable form: the enclosing
    paragraph has to name ``output/benchmarks/reference/`` or the open
    ``PERF-001`` register row. The historical documents are exempt because the
    retracted ``[0.2.0]`` changelog entry is preserved verbatim on purpose,
    under a corrective note that does the citing.
    """
    offences: list[str] = []
    for path in _tracked_prose_files():
        if _relative(path) in HISTORICAL_DOCUMENTS:
            continue
        offences.extend(_uncited_accelerator_claims(path, PROSE_SPEED_CLAIM))
    assert not offences, (
        "Accelerator or speed claims with no evidence in their paragraph:\n  "
        + "\n  ".join(offences)
        + "\nCite output/benchmarks/reference/ or name PERF-001 in the same "
        "paragraph, or delete the claim. No accelerator run of RadioSim has "
        "ever been measured."
    )


def test_no_accelerator_claim_in_the_package_lacks_a_citation() -> None:
    """The same rule, applied to the package's own docstrings and messages.

    Routed here at the 8B and 8C independent acceptances, which found three
    instances no prose scan could reach: ``simulator/__init__.py``'s module
    docstring advertised "GPU acceleration via JAX backend" as a current
    capability of ``rime`` (a line dating to the original project rename, older
    than every register row), and ``simulator/base.py``'s ``supports_gpu``
    docstring said "True if simulator can use GPU backends ... Default is
    True". Both are fixed; this test is why they cannot come back.

    Scoped to capability-claim language rather than to the bare ``gpu`` token,
    for the reason the 8C re-review gave: ``radiosim.backends : Backend
    abstraction for CPU/GPU`` names another module's scope, and
    ``jax.devices("gpu")`` in ``backends/`` or the vendor probes in
    ``utils/device.py`` are facts about the host, not claims about RadioSim.
    """
    offences: list[str] = []
    for root in CLAIM_SOURCE_ROOTS:
        for path in iter_tracked_files(REPO_ROOT / root, suffixes=frozenset({".py"})):
            offences.extend(_uncited_accelerator_claims(path, ACCELERATOR_CLAIM))
    assert not offences, (
        "Accelerator claims in package source with no evidence in their "
        "paragraph:\n  "
        + "\n  ".join(offences)
        + "\nA docstring is documentation. Cite output/benchmarks/reference/ "
        "or name PERF-001 beside the claim, or state the measured truth: "
        "RIMESimulator.supports_gpu is False and no accelerator run exists."
    )


# ---------------------------------------------------------------------------
# Section 11 item 8 -- every documented command exists
# ---------------------------------------------------------------------------


def _documented_pixi_targets(line: str) -> list[str]:
    """Return the first non-option token of every ``pixi run`` in a line.

    ``pixi run`` takes options before the target, and ``--environment <name>``
    takes a value, so a naive "first word after ``pixi run``" reading turns
    ``pixi run --environment crossval test`` into a claim that a ``crossval``
    task exists. Everything after a bare ``--`` is the command pixi executes,
    which is checked the same way: it is either a task or an executable.
    """
    targets: list[str] = []
    for rest in _PIXI_RUN.findall(line):
        tokens = rest.replace("`", " ").split()
        skip_next = False
        for token in tokens:
            if skip_next:
                skip_next = False
                continue
            if token in PIXI_OPTIONS_WITH_VALUE:
                skip_next = True
                continue
            if token.startswith("-"):
                continue
            if _TASK_TOKEN.match(token):
                targets.append(token)
            break
    return targets


def test_every_documented_pixi_task_exists() -> None:
    """A ``pixi run <task>`` a document prints must be a task a reader can run.

    Tasks come and go across a refactor -- ``doctest`` and ``bench`` were both
    added during this programme -- and a document naming one that was renamed
    sends a contributor to a "task not found" error. Direct executables
    (``pixi run python ...``, ``pixi run radiosim ...``) are accepted because
    ``pixi run`` accepts them, and a line that says a command deliberately does
    *not* work is left alone.
    """
    tasks = set(tomllib.loads(_read("pixi.toml"))["tasks"])
    unknown: list[str] = []
    for path in _tracked_prose_files():
        relative = _relative(path)
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if any(marker in line for marker in NEGATED_COMMAND_MARKERS):
                continue
            for token in _documented_pixi_targets(line):
                if token in tasks or token in ENVIRONMENT_COMMANDS:
                    continue
                unknown.append(f"{relative}:{number} runs `pixi run {token}`")
    assert not unknown, (
        "Documented pixi tasks that do not exist:\n  "
        + "\n  ".join(sorted(unknown))
        + f"\npixi.toml defines: {', '.join(sorted(tasks))}."
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
