"""Release metadata and Pixi lock-format safeguards."""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

import radiosim

ROOT = Path(__file__).resolve().parents[2]
LOCK_ENVIRONMENTS = {"default": "3.11", "py312": "3.12"}
LOCK_PLATFORMS = ("linux-64", "osx-64", "osx-arm64")


def _read_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _read_pixi_lock() -> dict[str, Any]:
    lock = yaml.safe_load((ROOT / "pixi.lock").read_text(encoding="utf-8"))
    assert isinstance(lock, dict), "pixi.lock must contain a top-level mapping"
    return lock


def _selected_lock_package_refs(
    lock: dict[str, Any],
    *,
    environment: str,
    platform: str,
    package: str,
) -> list[tuple[str, str]]:
    refs = lock["environments"][environment]["packages"][platform]
    selected: list[tuple[str, str]] = []
    for ref in refs:
        assert isinstance(ref, dict) and len(ref) == 1
        provenance, url = next(iter(ref.items()))
        assert isinstance(provenance, str)
        assert isinstance(url, str)
        filename = url.rsplit("/", maxsplit=1)[-1]
        if filename.startswith(f"{package}-"):
            selected.append((provenance, url))
    return selected


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


@pytest.mark.parametrize(("environment", "python_version"), LOCK_ENVIRONMENTS.items())
@pytest.mark.parametrize("platform", LOCK_PLATFORMS)
@pytest.mark.parametrize("package", ("numba", "llvmlite"))
def test_pixi_lock_selects_python_compatible_conda_numba_stack(
    environment: str,
    python_version: str,
    platform: str,
    package: str,
) -> None:
    lock = _read_pixi_lock()
    context = f"{environment}/{platform}/{package}"
    assert lock.get("version") == 7, f"{context}: pixi.lock must remain v7"

    selected = _selected_lock_package_refs(
        lock,
        environment=environment,
        platform=platform,
        package=package,
    )
    assert selected, f"{context}: package is not selected"

    if platform == "osx-64":
        source_fallbacks = [
            url
            for provenance, url in selected
            if provenance == "pypi" and url.endswith(".tar.gz")
        ]
        assert not source_fallbacks, (
            f"{context}: unsupported Intel macOS PyPI source fallback selected: "
            f"{source_fallbacks}"
        )

    assert len(selected) == 1, (
        f"{context}: expected exactly one selected package, found {selected}"
    )
    provenance, url = selected[0]
    assert provenance == "conda", (
        f"{context}: expected Conda provenance, found {provenance}: {url}"
    )
    assert f"/{platform}/" in url, (
        f"{context}: Conda package targets the wrong platform: {url}"
    )

    python_tag = python_version.replace(".", "")
    filename = url.rsplit("/", maxsplit=1)[-1]
    assert f"-py{python_tag}" in filename, (
        f"{context}: Conda build is incompatible with Python {python_version}: "
        f"{filename}"
    )
