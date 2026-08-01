"""Tier 5 whole-tier acceptance: forbidden residuals and closed surfaces.

Every assertion here is a removal or exactness contract for the Tier 5 receptor
and polarization range, so that Tier 5I can re-run one executable statement of
"no superseded polarization helper, no duplicated polarization-basis literal,
and no permissive receptor-stub keyword remains".

The removal ledger this file pins is `Tier5ReceptorFeedPlan.md` §34.8, resolved
against the §43 Q4/Q5 evidence Tier 5A recorded:

``radiosim.core.polarization.visibility_to_correlations``
    Removed.  §24 fixes the removal branch when 5A shows no production caller,
    and 5A showed exactly that: its only references outside its defining module
    were the two re-export lines in ``radiosim.core.__init__``.  It also
    hard-keyed the linear labels ``XX/XY/YX/YY`` plus an ``"I"`` entry, which
    the basis-aware ``CORRELATION_LABELS`` table supersedes.
``radiosim.core.polarization.mueller_from_jones``
    Removed.  It raised ``NotImplementedError`` from an unadvertised public
    module-level name, and §28 forbids pre-v1 deprecation shims, so the §34.8
    "or gate it explicitly as Tier 7" branch is not taken.
``radiosim.core.receptor.PolarizationBasisName``
    Removed.  It was an independent copy of
    ``radiosim.core.polarization_basis.PolarizationBasis``; §34.8's "remove the
    duplicated correlation constants if any survive" names exactly this
    literal, as the §35 Tier 5H correction records.

Three neighbouring helpers -- ``apply_jones_matrices``,
``stokes_I_only_visibility``, and ``jones_matrix_power`` -- share the
no-production-caller state 5A recorded, but §34.8's ledger does not name them,
so Tier 5H deliberately keeps them.
``test_surviving_polarization_surface_is_exact`` pins that decision so a later
slice cannot quietly widen the removal.

The repository-wide reference scan implements the §34.8 stop condition.  It is
an exact-list assertion, not an exemption list: ``src/``, ``configs/``,
``examples/``, and ``docs/`` must be completely clean, and inside ``tests/`` the
only files permitted to name a removed symbol are the removal records
enumerated in :data:`ALLOWED_REFERENCES`.
"""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

import radiosim.core as core_package
import radiosim.core.polarization as polarization_module
import radiosim.core.polarization_basis as polarization_basis_module
import radiosim.core.receptor as receptor_module
from radiosim.core.jones.receptor import BasisTransformJones, ReceptorConfigJones

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "radiosim"

# Section 34.8 removal ledger, keyed by the module that used to define each name.
REMOVED_POLARIZATION_NAMES = ("visibility_to_correlations", "mueller_from_jones")
REMOVED_RECEPTOR_NAMES = ("PolarizationBasisName",)
REMOVED_NAMES = REMOVED_POLARIZATION_NAMES + REMOVED_RECEPTOR_NAMES

# Section 24 permissive stub keywords, removed by 5C and re-pinned here.
REMOVED_RECEPTOR_TERM_KEYWORDS = ("feed_type", "from_basis", "to_basis")

# Polarization helpers Section 34.8 does not name, and that therefore survive.
SURVIVING_POLARIZATION_NAMES = (
    "apply_jones_matrices",
    "coherency_to_stokes",
    "jones_matrix_power",
    "stokes_I_only_visibility",
    "stokes_to_coherency",
)

# The exact `radiosim.core` polarization re-exports after the 5H removals, in
# `radiosim.core.__all__` declaration order.
EXPECTED_CORE_POLARIZATION_EXPORTS = ("stokes_to_coherency", "apply_jones_matrices")

# Directory trees the Section 34.8 stop condition names.
REFERENCE_SCAN_ROOTS = (
    REPOSITORY_ROOT / "src",
    REPOSITORY_ROOT / "tests",
    REPOSITORY_ROOT / "configs",
    REPOSITORY_ROOT / "examples",
    REPOSITORY_ROOT / "docs",
)

REFERENCE_SCAN_SUFFIXES = (".py", ".rst", ".md", ".yaml", ".yml", ".txt", ".cfg")

# The complete set of files allowed to name a removed symbol, and the only reason
# they may: each one exists to assert that the symbol is gone.  The scan below is
# an exact-list assertion rather than an exemption, so a new reference anywhere --
# including a new reference inside one of these files' neighbours -- fails.
REFERENCE_RECORDING_FILES = (
    "docs/migration_guide.md",
    "tests/characterization/test_tier5_current_behavior.py",
    "tests/unit/test_core/test_polarization.py",
    "tests/unit/test_core/test_receptor_resolution.py",
    "tests/unit/test_tier5_receptor_acceptance.py",
)

ALLOWED_REFERENCES = {
    "visibility_to_correlations": (
        "docs/migration_guide.md",
        "tests/characterization/test_tier5_current_behavior.py",
        "tests/unit/test_core/test_polarization.py",
        "tests/unit/test_tier5_receptor_acceptance.py",
    ),
    "mueller_from_jones": (
        "docs/migration_guide.md",
        "tests/characterization/test_tier5_current_behavior.py",
        "tests/unit/test_core/test_polarization.py",
        "tests/unit/test_tier5_receptor_acceptance.py",
    ),
    "PolarizationBasisName": (
        "docs/migration_guide.md",
        "tests/unit/test_core/test_receptor_resolution.py",
        "tests/unit/test_tier5_receptor_acceptance.py",
    ),
}

# The one authoritative spelling of the output-basis literal.
BASIS_LITERAL_TEXT = 'Literal["linear_xy", "circular_rl"]'


def _iter_package_sources() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def _iter_reference_scan_files() -> list[Path]:
    # Scope the walk to files git knows about (tracked, plus untracked files
    # not covered by .gitignore) so gitignored build artifacts -- a stale
    # ``docs/_build/`` in particular -- cannot pollute the residual scan.
    listing = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            *(root.name for root in REFERENCE_SCAN_ROOTS),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    found: list[Path] = []
    for relative in sorted(filter(None, listing.stdout.split("\0"))):
        path = REPOSITORY_ROOT / relative
        if not path.is_file() or path.suffix not in REFERENCE_SCAN_SUFFIXES:
            continue
        if "__pycache__" in path.parts:
            continue
        found.append(path)
    return found


# ---------------------------------------------------------------------------
# Removed names: attribute failure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", REMOVED_POLARIZATION_NAMES)
def test_removed_polarization_names_are_absent_from_their_module(name: str) -> None:
    assert not hasattr(polarization_module, name)
    assert name not in dir(polarization_module)
    with pytest.raises(AttributeError):
        getattr(polarization_module, name)


@pytest.mark.parametrize("name", REMOVED_RECEPTOR_NAMES)
def test_removed_receptor_names_are_absent_from_their_module(name: str) -> None:
    assert not hasattr(receptor_module, name)
    assert name not in receptor_module.__all__
    with pytest.raises(AttributeError):
        getattr(receptor_module, name)


@pytest.mark.parametrize("name", REMOVED_NAMES)
def test_removed_names_are_absent_from_the_core_package(name: str) -> None:
    assert not hasattr(core_package, name)
    assert name not in core_package.__all__
    assert name not in dir(core_package)
    with pytest.raises(AttributeError):
        core_package.__getattr__(name)


# ---------------------------------------------------------------------------
# Removed names: import failure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("module_name", "name"),
    [("radiosim.core.polarization", name) for name in REMOVED_POLARIZATION_NAMES]
    + [("radiosim.core.receptor", name) for name in REMOVED_RECEPTOR_NAMES]
    + [("radiosim.core", name) for name in REMOVED_NAMES],
)
def test_removed_names_cannot_be_imported(module_name: str, name: str) -> None:
    source = f"from {module_name} import {name}\n"
    with pytest.raises(ImportError):
        exec(compile(source, "<tier5h-residual>", "exec"), {})


@pytest.mark.parametrize("name", REMOVED_NAMES)
def test_removed_names_are_defined_nowhere_in_the_package(name: str) -> None:
    for path in _iter_package_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                assert node.name != name, path
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                assert node.id != name, path


@pytest.mark.parametrize("name", REMOVED_NAMES)
def test_removed_names_are_referenced_nowhere_in_the_repository(name: str) -> None:
    """The Section 34.8 stop condition, executable.

    ``src/``, ``configs/``, and ``examples/`` must be completely clean. Inside
    ``tests/`` the only permitted references are the removal records enumerated
    in :data:`ALLOWED_REFERENCES`. ``docs/migration_guide.md`` is the one
    sanctioned ``docs/`` reference -- Tier 5I's routed migration-guide entry
    naming these three removed symbols and stating each has no replacement
    (Tier5ReceptorFeedPlan.md §35 Tier 5I); every other file under ``docs/``
    stays clean.
    """
    references = sorted(
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in _iter_reference_scan_files()
        if name in path.read_text(encoding="utf-8", errors="ignore")
    )
    assert references == sorted(ALLOWED_REFERENCES[name])
    assert all(reference in REFERENCE_RECORDING_FILES for reference in references)
    assert all(
        reference.startswith("tests/") or reference == "docs/migration_guide.md"
        for reference in references
    )


# ---------------------------------------------------------------------------
# One authoritative polarization-basis literal
# ---------------------------------------------------------------------------


def test_the_output_basis_literal_is_defined_exactly_once() -> None:
    """``PolarizationBasis`` is the single spelling of the two output bases."""
    definitions = [
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in _iter_package_sources()
        if BASIS_LITERAL_TEXT in path.read_text(encoding="utf-8")
    ]
    assert definitions == ["core/polarization_basis.py"]


def test_every_shared_correlation_table_is_defined_exactly_once() -> None:
    """Section 34.8: "remove the duplicated correlation constants if any survive".

    B9 removed four duplicated tables in 5E and 5F.  This pins that each one is
    assigned in exactly one module, so no slice can reintroduce a private copy.
    """
    tables = (
        "CORRELATION_LABELS",
        "AIPS_CODES_CANONICAL",
        "AIPS_CODES_FILE_ORDER",
        "PYUVDATA_FEEDS",
        "PYUVDATA_POLARIZATIONS",
    )
    for table in tables:
        assigning: list[str] = []
        for path in _iter_package_sources():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                target = None
                if isinstance(node, ast.AnnAssign):
                    target = node.target
                elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                    target = node.targets[0]
                if isinstance(target, ast.Name) and target.id == table:
                    assigning.append(path.relative_to(PACKAGE_ROOT).as_posix())
        assert assigning == ["core/polarization_basis.py"], table


def test_the_receptor_module_consumes_the_shared_basis_literal() -> None:
    """``core/receptor.py`` must import the literal, not restate it."""
    assert (
        receptor_module.PolarizationBasis is polarization_basis_module.PolarizationBasis
    )
    assert core_package.PolarizationBasis is polarization_basis_module.PolarizationBasis
    source = (PACKAGE_ROOT / "core" / "receptor.py").read_text(encoding="utf-8")
    assert "from radiosim.core.polarization_basis import PolarizationBasis" in source


def test_the_resolved_output_basis_tokens_come_from_the_shared_table() -> None:
    assert set(receptor_module._OUTPUT_BASIS_BY_NATIVE.values()) == set(
        polarization_basis_module.POLARIZATION_BASES
    )


# ---------------------------------------------------------------------------
# Exact surviving surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", SURVIVING_POLARIZATION_NAMES)
def test_surviving_polarization_helpers_are_untouched(name: str) -> None:
    """Section 34.8's ledger names two helpers; the neighbours stay."""
    assert callable(getattr(polarization_module, name))


def test_surviving_polarization_surface_is_exact() -> None:
    exported = tuple(
        name
        for name in core_package.__all__
        if name in set(SURVIVING_POLARIZATION_NAMES) | set(REMOVED_POLARIZATION_NAMES)
    )
    assert exported == EXPECTED_CORE_POLARIZATION_EXPORTS


def test_the_polarization_module_docstring_names_no_removed_helper() -> None:
    docstring = polarization_module.__doc__ or ""
    for name in REMOVED_POLARIZATION_NAMES:
        assert name not in docstring


# ---------------------------------------------------------------------------
# Removed permissive receptor-stub keywords
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("keyword", REMOVED_RECEPTOR_TERM_KEYWORDS)
@pytest.mark.parametrize("term", [ReceptorConfigJones, BasisTransformJones])
def test_removed_stub_keywords_raise_a_migrating_type_error(
    term: type,
    keyword: str,
) -> None:
    with pytest.raises(TypeError) as excinfo:
        term(**{keyword: "linear"})
    message = str(excinfo.value)
    assert keyword in message
    assert "receptors" in message


@pytest.mark.parametrize("term", [ReceptorConfigJones, BasisTransformJones])
def test_removed_stub_keywords_are_not_constructor_parameters(term: type) -> None:
    parameters = inspect.signature(term.__init__).parameters
    for keyword in REMOVED_RECEPTOR_TERM_KEYWORDS:
        assert keyword not in parameters


# ---------------------------------------------------------------------------
# Fresh-process import checks
# ---------------------------------------------------------------------------


def test_a_fresh_process_cannot_reach_any_removed_name() -> None:
    program = "\n".join(
        [
            "import radiosim.core as core",
            "import radiosim.core.polarization as pol",
            "import radiosim.core.receptor as rec",
            "for module, names in (",
            f"    (pol, {REMOVED_POLARIZATION_NAMES!r}),",
            f"    (rec, {REMOVED_RECEPTOR_NAMES!r}),",
            f"    (core, {REMOVED_NAMES!r}),",
            "):",
            "    for name in names:",
            "        assert not hasattr(module, name), (module.__name__, name)",
            "assert 'h5py' not in __import__('sys').modules",
            "assert 'pyuvdata' not in __import__('sys').modules",
            "print('ok')",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPOSITORY_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().endswith("ok")
