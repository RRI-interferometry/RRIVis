"""Tier 8 release-acceptance scans over the tracked documentation surface.

``Tier8ReleasePlan.md`` Section 11 puts every state-2 scan of the tier in one
module rather than eight, because they share a file lister, an allow-list
vocabulary and a failure-message style.  This module is that home.  It is
created at 8B with Section 11 items **3** (shipped-configuration counts are
derived from the filesystem, and every named configuration exists) and **6**
(``examples/README.md`` documents exactly the flags the example script's parser
defines, and vice versa).  8C adds scans 1, 4 and 5; 8D and 8E add the rest.

The distinction from ``tests/characterization/test_tier8_current_behavior.py``
is direction.  The characterization module pins drift *as it is* so that a
slice's effect is measurable; this module asserts the *rule* that stops the
drift returning.  A pin is deleted or inverted when its slice lands; a scan
here is permanent.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_SCRIPT = REPO_ROOT / "examples" / "scripts" / "simple_simulation.py"
EXAMPLES_README = REPO_ROOT / "examples" / "README.md"

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


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


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
