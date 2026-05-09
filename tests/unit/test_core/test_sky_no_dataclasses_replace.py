"""Static guard: no module under ``radiosim.core.sky`` may import
``replace`` from :mod:`dataclasses`.

Both ``SkyModel`` and ``SkyProvenance`` define their own ``replace``
methods that re-run pydantic validators (precision-aware dtype casting
on ``SkyModel``; cross-field invariants like
``partial-sky => monopole_k is None`` on ``SkyProvenance``). Calling
``dataclasses.replace`` directly bypasses those validators and lets a
caller produce a state the constructor would reject.

This test walks every ``.py`` file in ``src/radiosim/core/sky`` and
fails if any non-test module imports ``replace`` from
``dataclasses`` — by name, alias, or via ``import dataclasses`` followed
by ``dataclasses.replace(...)``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import radiosim


def _sky_modules() -> list[Path]:
    pkg_root = Path(radiosim.__file__).parent / "core" / "sky"
    return sorted(p for p in pkg_root.rglob("*.py") if p.is_file())


def _imports_replace_from_dataclasses(tree: ast.AST) -> bool:
    """``from dataclasses import replace`` (any alias)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "dataclasses":
            for alias in node.names:
                if alias.name == "replace":
                    return True
    return False


def _calls_dataclasses_replace(tree: ast.AST) -> bool:
    """``import dataclasses`` *and* a ``dataclasses.replace(...)`` call."""
    has_dataclasses_import = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "dataclasses":
                    has_dataclasses_import = True
                    break
    if not has_dataclasses_import:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            value = node.func.value
            if (
                isinstance(value, ast.Name)
                and value.id == "dataclasses"
                and node.func.attr == "replace"
            ):
                return True
    return False


def test_no_dataclasses_replace_in_sky_package() -> None:
    offenders: list[str] = []
    for path in _sky_modules():
        tree = ast.parse(path.read_text())
        if _imports_replace_from_dataclasses(tree) or _calls_dataclasses_replace(tree):
            offenders.append(str(path))
    assert not offenders, (
        "Modules under radiosim.core.sky must not call dataclasses.replace "
        "directly (it bypasses SkyModel/SkyProvenance pydantic validators). "
        "Use the .replace() method on the instance instead. Offenders:\n"
        + "\n".join(f"  - {p}" for p in offenders)
    )
