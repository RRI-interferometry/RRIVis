"""Git-scoped repository file listing for tests that scan the source tree.

Why this module exists
======================

Dozens of tests in this suite assert things about *the repository* -- no
removed symbol survives in ``src/``, no module imports a forbidden name, every
Jones term declares its status.  Each of them needs a list of files, and the
obvious way to build one, :meth:`pathlib.Path.rglob`, is wrong: ``rglob``
walks the filesystem, so it sees files ``git`` does not.  A gitignored
``.ipynb_checkpoints/`` copy of a module, an editor backup, a stale build
tree, a scratch script -- any of them turns a repository scan into a false
failure that reproduces on exactly one contributor's machine and nowhere else.
``Fix.md`` Section 17 item 15 records the demonstration; ``Tier8ReleasePlan.md``
Section 12 rules that every repository scan is listed through ``git``.

The lister is one ``git ls-files --cached --others --exclude-standard`` call.
That set is precisely "files a contributor is accountable for": tracked files,
plus untracked files that are *not* ignored -- so a newly written, not-yet-
added module is still scanned (the scan must catch it before it is committed,
not after), while anything ``.gitignore`` covers is invisible.

No fallback, on purpose
=======================

If ``git`` is unavailable or the repository is not a work tree,
:func:`iter_tracked_files` raises :class:`RepositoryScanError` rather than
falling back to ``rglob``.  A silent fallback would reintroduce exactly the
pollution this module exists to prevent, in the environment least able to
notice it.

No caching, on purpose
======================

Every call shells out afresh.  The negative tests that prove this module works
-- create a gitignored file, assert the scan still passes; create an untracked
*visible* violation, assert the scan fails -- mutate the tree inside a single
pytest session, and a cache would make them assert about a stale listing.  A
listing costs a few milliseconds; correctness of the guard costs more.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

__all__ = [
    "PACKAGE_ROOT",
    "PYTHON_SUFFIXES",
    "REPO_ROOT",
    "RepositoryScanError",
    "iter_package_sources",
    "iter_repository_python",
    "iter_tracked_files",
]

#: The repository root: ``tests/support/repo_scan.py`` -> ``tests/support`` ->
#: ``tests`` -> here.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The installed package's source root, the scan root most tests want.
PACKAGE_ROOT = REPO_ROOT / "src" / "radiosim"

#: The suffix set for a Python-source scan.
PYTHON_SUFFIXES = frozenset({".py"})


class RepositoryScanError(RuntimeError):
    """Raised when the git-scoped file listing cannot be produced.

    Never caught inside this module and never degraded to an ``rglob`` walk:
    a repository scan that silently changes what it scans is worse than one
    that stops.
    """


def _relative_to_repo(root: Path) -> str:
    """Return ``root`` as a repository-relative pathspec.

    Parameters
    ----------
    root : Path
        A path inside :data:`REPO_ROOT`.

    Returns
    -------
    str
        The repository-relative path, in POSIX form, as ``git`` pathspecs
        expect.

    Raises
    ------
    RepositoryScanError
        If ``root`` lies outside the repository.  A test scanning a
        ``tmp_path`` directory wants :meth:`pathlib.Path.rglob`, not this
        module, and saying so loudly is better than listing nothing.
    """
    resolved = Path(root).resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise RepositoryScanError(
            f"{resolved} is outside the repository at {REPO_ROOT}. "
            "iter_tracked_files() lists files git knows about; a directory "
            "created by the test itself (a tmp_path tree) is not one of them "
            "and should be walked with Path.rglob instead."
        ) from exc
    return relative.as_posix()


def iter_tracked_files(
    *roots: Path,
    suffixes: frozenset[str] | None = None,
) -> list[Path]:
    """List the files ``git`` knows about under ``roots``.

    Parameters
    ----------
    *roots : Path
        Directories or files inside the repository.  With no roots, the whole
        repository is listed.
    suffixes : frozenset[str] or None
        If given, keep only files whose :attr:`~pathlib.PurePath.suffix` is in
        this set (``{".py"}``, ``{".rst", ".md"}``, ...).  ``None`` keeps every
        suffix.

    Returns
    -------
    list[Path]
        Sorted absolute paths of existing files: tracked files plus untracked
        files that ``.gitignore`` does not cover, with ``__pycache__`` dropped.

    Raises
    ------
    RepositoryScanError
        If ``git`` is missing, the directory is not a work tree, a root lies
        outside the repository, or ``git`` exits non-zero.
    """
    pathspecs = [_relative_to_repo(root) for root in roots]
    command = [
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        *pathspecs,
    ]
    try:
        listing = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:  # git is not installed
        raise RepositoryScanError(
            "git is not available, so the repository file listing cannot be "
            "produced. These scans deliberately have no Path.rglob fallback: "
            "a filesystem walk sees gitignored files and turns a stray "
            "checkpoint or build artifact into a false test failure."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RepositoryScanError(
            f"`{' '.join(command)}` failed with exit status {exc.returncode} "
            f"in {REPO_ROOT}: {exc.stderr.strip()!r}"
        ) from exc

    found: list[Path] = []
    for relative in listing.stdout.split("\0"):
        if not relative:
            continue
        path = REPO_ROOT / relative
        if suffixes is not None and path.suffix not in suffixes:
            continue
        if "__pycache__" in path.parts:
            continue
        if not path.is_file():
            # `--cached` also lists files staged for deletion.
            continue
        found.append(path)
    return sorted(found)


def iter_package_sources() -> list[Path]:
    """List the tracked Python sources of ``src/radiosim``.

    Returns
    -------
    list[Path]
        Sorted absolute paths, the git-scoped replacement for
        ``(REPO_ROOT / "src" / "radiosim").rglob("*.py")``.
    """
    return iter_tracked_files(PACKAGE_ROOT, suffixes=PYTHON_SUFFIXES)


def iter_repository_python() -> list[Path]:
    """List every Python file in the repository that ``git`` knows about.

    Returns
    -------
    list[Path]
        Sorted absolute paths across ``src/``, ``tests/``, ``examples/``,
        ``scripts/`` and anything else tracked -- the git-scoped replacement
        for a repository-wide ``rglob("*.py")``.
    """
    return iter_tracked_files(suffixes=PYTHON_SUFFIXES)
