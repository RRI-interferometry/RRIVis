"""The shared repository-scan lister, and the pollution it exists to stop.

``Tier8ReleasePlan.md`` Section 12 extracted one git-scoped file lister into
:mod:`tests.support.repo_scan` and converted every repository scan in the suite
onto it.  This module proves the two halves of that contract:

*   **Gitignored files are invisible.**  The exact scenario ``Fix.md``
    Section 17 item 15 records -- a notebook checkpoint under ``src/`` carrying
    a removed Jones class name and a stub marker -- is materialised here, and
    the two acceptance scans that a filesystem walk would have turned red are
    run against it and asserted green.
*   **Everything else is still visible.**  A tracked file and an untracked file
    that ``.gitignore`` does not cover both appear in the listing, so hardening
    the scans did not weaken them.  That half is proved inside a throwaway
    repository built in ``tmp_path``: planting a real violation in *this*
    repository would be seen by every other scan running in parallel under
    ``pytest -n auto``, which is precisely the pollution under test.

The one file this module writes into the working tree is written under a path
first proved gitignored, and is removed in a ``finally``: a leftover would
poison every scan for the rest of the session.
"""

from __future__ import annotations

import importlib
import subprocess
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path

import pytest

from tests.support import repo_scan
from tests.support.repo_scan import (
    PACKAGE_ROOT,
    PYTHON_SUFFIXES,
    REPO_ROOT,
    RepositoryScanError,
    iter_package_sources,
    iter_repository_python,
    iter_tracked_files,
)

#: Where the notebook-checkpoint pollution is materialised.  ``.ipynb_checkpoints``
#: is a directory pattern in ``.gitignore``, which is why the guard below asks
#: ``git check-ignore`` about the directory *after* creating it: a directory-only
#: pattern cannot match a path that does not exist (the hermeticity lesson of the
#: Tier 8C rejection).
CHECKPOINT_DIRECTORY = PACKAGE_ROOT / ".ipynb_checkpoints"

#: A removed Jones class name and a stub marker, both assembled from fragments
#: so that this test file is not itself a carrier of either literal.
_REMOVED_CLASS_NAME = "Geometric" + "PhaseJones"
_STUB_MARKER = "TODO" + ": implement properly"
_IDENTITY_RETURN = "xp." + "eye(2, dtype=np.complex128)"

CHECKPOINT_CONTENT = f'''"""A stale notebook checkpoint of a module that no longer looks like this."""


class {_REMOVED_CLASS_NAME}:
    # {_STUB_MARKER}
    def compute_jones(self, xp, np):
        return {_IDENTITY_RETURN}
'''


def _is_ignored(path: Path) -> bool:
    """Return whether ``git`` ignores ``path`` (which must exist)."""
    return (
        subprocess.run(
            ["git", "check-ignore", "-q", str(path)],
            cwd=REPO_ROOT,
            capture_output=True,
        ).returncode
        == 0
    )


def _make_repository(root: Path) -> None:
    """Build a throwaway git repository with one file of each visibility class."""
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "scan@example.invalid"], cwd=root, check=True
    )
    subprocess.run(["git", "config", "user.name", "Scan"], cwd=root, check=True)

    (root / ".gitignore").write_text("build/\n*.bak\n", encoding="utf-8")
    (root / "tracked.py").write_text("TRACKED = True\n", encoding="utf-8")
    (root / "notes.md").write_text("# tracked prose\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)

    # Untracked and *not* ignored: a scan must still see this one, because the
    # violation it may contain is one a contributor is about to commit.
    (root / "untracked.py").write_text("UNTRACKED = True\n", encoding="utf-8")
    # Ignored two different ways, plus the cache directory the lister drops.
    (root / "build").mkdir()
    (root / "build" / "generated.py").write_text("GENERATED = True\n", encoding="utf-8")
    (root / "editor.py.bak").write_text("BACKUP = True\n", encoding="utf-8")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "cached.py").write_text("CACHED = True\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# The regression proof: gitignored pollution is invisible to the hardened scans
# ---------------------------------------------------------------------------


@pytest.fixture
def notebook_checkpoint_pollution(request: pytest.FixtureRequest) -> Iterator[Path]:
    """Materialise the gitignored checkpoint, and always remove it again.

    The directory is created first and proved gitignored *before* anything is
    written into it, so there is no window in which a violating file is visible
    to the scans running in other ``pytest -n auto`` workers.  The file name is
    per test, and the directory is only removed when it is empty, because the
    two tests using this fixture land on *different* xdist workers and would
    otherwise delete each other's probe mid-test.  Removal is unconditional: a
    leftover would poison every subsequent scan in the session.
    """
    CHECKPOINT_DIRECTORY.mkdir(exist_ok=True)
    checkpoint = CHECKPOINT_DIRECTORY / f"{request.node.name}-checkpoint.py"
    try:
        assert _is_ignored(CHECKPOINT_DIRECTORY), (
            f"{CHECKPOINT_DIRECTORY} is not gitignored, so writing the "
            "pollution probe would be visible to every other scan running in "
            "parallel. Restore the `.ipynb_checkpoints` entry in .gitignore."
        )
        checkpoint.write_text(CHECKPOINT_CONTENT, encoding="utf-8")
        assert _is_ignored(checkpoint)
        yield checkpoint
    finally:
        checkpoint.unlink(missing_ok=True)
        with suppress(OSError):
            CHECKPOINT_DIRECTORY.rmdir()


def test_a_gitignored_checkpoint_under_src_breaks_no_repository_scan(
    notebook_checkpoint_pollution: Path,
) -> None:
    """``Fix.md`` Section 17 item 15's demonstration, now a passing assertion.

    Before Tier 8D the two scans exercised here listed ``src/radiosim`` with
    :meth:`pathlib.Path.rglob`, so a gitignored
    ``src/radiosim/.ipynb_checkpoints/*-checkpoint.py`` holding a removed class
    name or a stub marker failed them -- on one contributor's machine, for a
    file no commit contains.  Both now list through ``git``, so the file is not
    in the scan set at all.
    """
    checkpoint = notebook_checkpoint_pollution

    listed = iter_package_sources()
    assert checkpoint not in listed
    assert checkpoint not in iter_repository_python()
    assert all(".ipynb_checkpoints" not in path.parts for path in listed)

    tier7 = importlib.import_module("tests.unit.test_tier7_jones_acceptance")
    assert _REMOVED_CLASS_NAME in tier7.REMOVED_JONES_NAMES
    tier7.test_a_removed_jones_name_appears_nowhere_in_the_package_source(
        _REMOVED_CLASS_NAME
    )
    tier7.test_no_stub_marker_survives_anywhere_in_the_package()


def test_the_same_file_fails_both_scans_the_moment_it_enters_the_listing(
    notebook_checkpoint_pollution: Path,
) -> None:
    """The other direction: hardening the listing did not soften the assertion.

    Exactly the file the previous test proves harmless is forced into the
    scans' own lister here, and both scans go red.  So the listing is the only
    reason they pass -- a real, tracked or untracked-but-unignored violation
    with the same content is still caught, and the hardening removed no
    assertion strength.
    """
    tier7 = importlib.import_module("tests.unit.test_tier7_jones_acceptance")
    checkpoint = notebook_checkpoint_pollution
    original = tier7._python_sources
    tier7._python_sources = lambda: [*original(), checkpoint]
    try:
        with pytest.raises(AssertionError):
            tier7.test_a_removed_jones_name_appears_nowhere_in_the_package_source(
                _REMOVED_CLASS_NAME
            )
        with pytest.raises(AssertionError):
            tier7.test_no_stub_marker_survives_anywhere_in_the_package()
    finally:
        tier7._python_sources = original


# ---------------------------------------------------------------------------
# The lister's own contract
# ---------------------------------------------------------------------------


def test_the_listing_holds_tracked_and_unignored_files_and_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tracked in, untracked-but-visible in, ignored out, ``__pycache__`` out."""
    root = tmp_path / "repository"
    _make_repository(root)
    monkeypatch.setattr(repo_scan, "REPO_ROOT", root)

    listed = {
        path.relative_to(root).as_posix() for path in repo_scan.iter_tracked_files()
    }
    assert "tracked.py" in listed
    assert "notes.md" in listed
    assert "untracked.py" in listed, (
        "an untracked file .gitignore does not cover must stay visible: the "
        "scan has to catch a violation before it is committed, not after"
    )
    assert "build/generated.py" not in listed
    assert "editor.py.bak" not in listed
    assert "__pycache__/cached.py" not in listed

    python_only = {
        path.relative_to(root).as_posix()
        for path in repo_scan.iter_tracked_files(root, suffixes=PYTHON_SUFFIXES)
    }
    assert python_only == {"tracked.py", "untracked.py"}


def test_a_root_outside_the_repository_is_a_typed_error(tmp_path: Path) -> None:
    """A ``tmp_path`` tree wants ``rglob``; saying so beats listing nothing."""
    with pytest.raises(RepositoryScanError, match="outside the repository"):
        iter_tracked_files(tmp_path)


def test_a_directory_that_is_not_a_work_tree_is_a_typed_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(repo_scan, "REPO_ROOT", tmp_path)
    with pytest.raises(RepositoryScanError, match="failed with exit status"):
        repo_scan.iter_tracked_files()


def test_a_missing_git_is_a_typed_error_and_never_an_rglob_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point of Section 12: no silent fallback to a filesystem walk."""

    def no_git(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(repo_scan.subprocess, "run", no_git)
    with pytest.raises(RepositoryScanError, match="git is not available"):
        iter_package_sources()


def test_every_listed_package_source_is_a_python_file_under_the_package() -> None:
    listed = iter_package_sources()
    assert listed, "the package source listing is empty"
    assert listed == sorted(listed)
    for path in listed:
        assert path.suffix == ".py"
        assert path.is_relative_to(PACKAGE_ROOT)
    assert PACKAGE_ROOT / "__init__.py" in listed
