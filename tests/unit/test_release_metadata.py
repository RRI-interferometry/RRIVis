"""Release metadata and Pixi lock-format safeguards."""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest

import radiosim

ROOT = Path(__file__).resolve().parents[2]


def _read_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _read_string_assignments(path: Path, names: set[str]) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in names:
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, str):
            raise AssertionError(f"{path}:{target.id} must be a string literal")
        values[target.id] = value

    missing = names - values.keys()
    if missing:
        raise AssertionError(f"{path} is missing assignments for: {sorted(missing)}")
    return values


def _assert_release_versions(canonical: str, observed: dict[str, str]) -> None:
    mismatches = {
        source: value for source, value in observed.items() if value != canonical
    }
    if not mismatches:
        return

    details = "\n".join(
        f"- {source}: {value!r}" for source, value in sorted(mismatches.items())
    )
    raise AssertionError(
        "Release metadata drifted from the canonical pyproject.toml "
        f"project.version {canonical!r}:\n{details}\n"
        "Update every listed source in the same release change."
    )


def test_release_metadata_matches_canonical_project_version() -> None:
    pyproject = _read_toml(ROOT / "pyproject.toml")
    pixi = _read_toml(ROOT / "pixi.toml")
    about = _read_string_assignments(
        ROOT / "src/radiosim/__about__.py", {"__version__"}
    )
    sphinx = _read_string_assignments(ROOT / "docs/conf.py", {"version", "release"})

    canonical = str(pyproject["project"]["version"])
    observed = {
        "pixi.toml workspace.version": str(pixi["workspace"]["version"]),
        "src/radiosim/__about__.py __version__": about["__version__"],
        "docs/conf.py version": sphinx["version"],
        "docs/conf.py release": sphinx["release"],
        "radiosim.__version__": radiosim.__version__,
    }

    _assert_release_versions(canonical, observed)


def test_release_metadata_failure_is_actionable() -> None:
    with pytest.raises(AssertionError) as exc_info:
        _assert_release_versions(
            "1.2.3",
            {
                "pixi.toml workspace.version": "1.2.2",
                "docs/conf.py release": "1.0.0",
                "radiosim.__version__": "1.2.3",
            },
        )

    message = str(exc_info.value)
    assert "pixi.toml workspace.version: '1.2.2'" in message
    assert "docs/conf.py release: '1.0.0'" in message
    assert "radiosim.__version__" not in message


def test_pixi_lock_uses_v7_format() -> None:
    lock_path = ROOT / "pixi.lock"
    header = "\n".join(lock_path.read_text(encoding="utf-8").splitlines()[:20])
    match = re.search(r"(?m)^version:\s*(\d+)\s*$", header)
    assert match is not None, "pixi.lock has no top-level version in its header"
    actual = int(match.group(1))
    assert actual == 7, f"pixi.lock format drifted: expected v7, found v{actual}"
