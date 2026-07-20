"""Regression tests for the Tier 2H legacy instrument cleanup."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import radiosim
import radiosim.core as core
import radiosim.core.instrument as instrument_models
import radiosim.core.instrument_resolution as instrument_resolution
import radiosim.core.runtime_config as runtime_config
import radiosim.io as io
import radiosim.io.config as input_config
import radiosim.io.instrument_config as instrument_config
from radiosim.api.simulator import Simulator

REMOVED_MODULES = (
    "radiosim.core.antenna",
    "radiosim.core.baseline",
    "radiosim.io.antenna_readers",
)

REMOVED_PUBLIC_NAMES = (
    "read_antenna_positions",
    "read_radiosim_format",
    "read_casa_format",
    "read_pyuvdata_format",
    "read_mwa_format",
    "format_antenna_data",
    "generate_baselines",
)

REMOVED_INPUT_DECLARATIONS = (
    "AntennaFileFormat",
    "AntennaLayoutConfig",
    "TelescopeConfig",
    "FeedsConfig",
    "LocationConfig",
)

REMOVED_RUNTIME_DECLARATIONS = (
    "ResolvedTelescopeConfig",
    "ResolvedAntennaLayoutConfig",
    "ResolvedLocationConfig",
)

CANONICAL_INSTRUMENT_MODELS = (
    "AntennaId",
    "AntennaFieldSource",
    "ResolvedEarthLocation",
    "AntennaProvenance",
    "ResolvedAntenna",
    "InstrumentProvenance",
    "ResolvedInstrument",
)

CANONICAL_BASELINE_MODELS = (
    "ResolvedBaseline",
    "BaselineSelectionCriteriaSnapshot",
    "BaselineSelectionProvenance",
    "ResolvedBaselineSelection",
)


@pytest.mark.parametrize("module_name", REMOVED_MODULES)
def test_deleted_legacy_module_paths_fail_in_fresh_process(module_name: str) -> None:
    probe = f"import importlib\nimportlib.import_module({module_name!r})\n"

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "ModuleNotFoundError" in completed.stderr


@pytest.mark.parametrize("namespace", [radiosim, core, io])
def test_removed_public_names_are_absent(namespace: object) -> None:
    exported = getattr(namespace, "__all__", ())
    for name in REMOVED_PUBLIC_NAMES:
        assert name not in vars(namespace)
        assert name not in exported
        with pytest.raises(AttributeError):
            getattr(namespace, name)


def test_removed_input_declarations_are_absent() -> None:
    for name in REMOVED_INPUT_DECLARATIONS:
        assert name not in vars(input_config)
        assert name not in input_config.__all__
        assert name not in vars(io)
        assert name not in io.__all__
    assert "_LegacyBaselineSelectionConfig" not in vars(input_config)


def test_removed_runtime_declarations_are_absent() -> None:
    for name in REMOVED_RUNTIME_DECLARATIONS:
        assert name not in vars(runtime_config)
        assert name not in runtime_config.__all__
        assert name not in vars(core)
        assert name not in core.__all__
        assert name not in vars(io)
        assert name not in io.__all__


def test_no_legacy_files_or_root_compatibility_guard_remain() -> None:
    package_root = Path(radiosim.__file__).resolve().parent
    for relative_path in (
        "core/antenna.py",
        "core/baseline.py",
        "io/antenna_readers.py",
    ):
        assert not (package_root / relative_path).exists()

    root_source = Path(radiosim.__file__).read_text(encoding="utf-8")
    root_tree = ast.parse(root_source)
    assert "_CORE_AVAILABLE" not in vars(radiosim)
    assert not any(
        isinstance(node, ast.Try)
        and any(
            isinstance(handler.type, ast.Name) and handler.type.id == "ImportError"
            for handler in node.handlers
        )
        for node in ast.walk(root_tree)
    )


def test_owned_source_has_no_legacy_definitions_or_imports() -> None:
    source_root = Path(radiosim.__file__).resolve().parent
    removed_definitions = (
        set(REMOVED_PUBLIC_NAMES)
        | set(REMOVED_INPUT_DECLARATIONS + REMOVED_RUNTIME_DECLARATIONS)
        | {"_LegacyBaselineSelectionConfig"}
    )
    found_definitions: list[str] = []
    found_imports: list[str] = []

    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in removed_definitions:
                    found_definitions.append(
                        f"{path.relative_to(source_root)}:{node.name}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module in REMOVED_MODULES:
                found_imports.append(f"{path.relative_to(source_root)}:{node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in REMOVED_MODULES:
                        found_imports.append(
                            f"{path.relative_to(source_root)}:{alias.name}"
                        )

    assert found_definitions == []
    assert found_imports == []


def test_canonical_public_exports_retain_approved_identity() -> None:
    for name in CANONICAL_INSTRUMENT_MODELS:
        direct = getattr(instrument_models, name)
        assert getattr(core, name) is direct
        assert getattr(radiosim, name) is direct
        assert name in core.__all__
        assert name in radiosim.__all__

    for name in CANONICAL_BASELINE_MODELS:
        direct = getattr(instrument_models, name)
        assert getattr(core, name) is direct
        assert name in core.__all__
        assert name not in radiosim.__all__
        assert not hasattr(radiosim, name)

    assert radiosim.calculate_visibility is core.calculate_visibility
    assert radiosim.Simulator is Simulator


def test_active_input_and_runtime_configuration_exports_remain_intact() -> None:
    for name in (
        "InstrumentConfig",
        "InstrumentLocationConfig",
        "BaselineSelectionConfig",
    ):
        assert getattr(input_config, name) is getattr(instrument_config, name)
        assert name in input_config.__all__

    for name in (
        "ResolvedBeamsConfig",
        "ResolvedSimulationConfig",
        "ResolvedConfiguration",
    ):
        direct = getattr(runtime_config, name)
        assert getattr(core, name) is direct
        assert getattr(io, name) is direct
        assert name in runtime_config.__all__
        assert name in core.__all__
        assert name in io.__all__


def test_active_tier2_truth_surfaces_describe_only_the_live_architecture() -> None:
    repository_root = Path(radiosim.__file__).resolve().parents[2]
    contributor_guide = (repository_root / "CLAUDE.md").read_text(encoding="utf-8")
    rendered_docstrings = "\n".join(
        (
            instrument_config.__doc__ or "",
            instrument_resolution.__doc__ or "",
        )
    )

    for removed_path in (
        "core/antenna.py",
        "core/baseline.py",
        "io/antenna_readers.py",
    ):
        assert removed_path not in contributor_guide

    assert "RadioSimConfig.validate()" not in contributor_guide
    assert "`telescope`, `antenna_layout`" not in contributor_guide
    assert "--telescope-name" in contributor_guide
    assert "--default-diameter-m" in contributor_guide

    for stale_claim in (
        "intentionally inactive",
        "future Tier 2 input contract",
        "remain later Tier 2 slices",
    ):
        assert stale_claim not in rendered_docstrings
