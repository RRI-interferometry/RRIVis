"""Tier 4 whole-tier acceptance: forbidden residuals and closed surfaces.

Every assertion here is a removal or exactness contract for the Tier 4 result
and output range.  The file exists so that Tier 4I can re-run one executable
statement of "no obsolete dictionary, reconstructed-axis, unsafe-reader,
generic-format, compatibility, or stale-dependency surface remains".
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

import radiosim
from radiosim import api as radiosim_api
from radiosim import io as radiosim_io
from radiosim.api.simulator import Simulator
from radiosim.io import hdf5 as hdf5_module
from radiosim.io import measurement_set as measurement_set_module
from radiosim.io.config import RadioSimConfig
from radiosim.io.result_format import ResultFormat

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "radiosim"
PYPROJECT_PATH = REPOSITORY_ROOT / "pyproject.toml"
PIXI_MANIFEST_PATH = REPOSITORY_ROOT / "pixi.toml"
PIXI_LOCK_PATH = REPOSITORY_ROOT / "pixi.lock"

# Modules deleted by the Tier 4 range.  Nothing may import them again.
REMOVED_MODULES = ("radiosim.io.writers",)

# Section 24 removals plus the Section 40 breaking-change ledger names.
REMOVED_PUBLIC_NAMES = (
    "save_visibilities_hdf5",
    "load_visibilities_hdf5",
    "save_config_yaml",
    "write_ms",
    "read_ms",
    "read_ms_dask",
    "ms_info",
    "PYUVDATA_AVAILABLE",
    "CASACORE_AVAILABLE",
    "DASKMS_AVAILABLE",
    "MS_AVAILABLE",
)

PUBLIC_NAMESPACES = (
    ("radiosim", radiosim),
    ("radiosim.api", radiosim_api),
    ("radiosim.io", radiosim_io),
)

# Section 25 removed workflow inputs and their exact rejection boundary.
REMOVED_WORKFLOW_FIELDS = (
    ("overwrite", True, "workflow.overwrite: removed before v1.0"),
    (
        "skip_overwrite_confirmation",
        True,
        "workflow.skip_overwrite_confirmation: removed before v1.0",
    ),
    (
        "prompt_for_output_suffix",
        True,
        "workflow.prompt_for_output_suffix: removed before v1.0",
    ),
    ("angle_unit", "degrees", "workflow.angle_unit: removed before v1.0"),
    (
        "sky_model_frequency_hz",
        150e6,
        "workflow.sky_model_frequency_hz: removed before v1.0",
    ),
)

SHIPPED_CONFIGS = (
    REPOSITORY_ROOT / "configs" / "config.yaml",
    REPOSITORY_ROOT / "configs" / "realistic_foreground_example.yaml",
    REPOSITORY_ROOT / "antenna_layout_examples" / "example_telescope_config.yaml",
)

# Active Tier 4 result and output production modules.
TIER4_ACTIVE_SOURCES = (
    PACKAGE_ROOT / "core" / "time_grid.py",
    PACKAGE_ROOT / "core" / "phase_center.py",
    PACKAGE_ROOT / "core" / "result.py",
    PACKAGE_ROOT / "io" / "atomic_paths.py",
    PACKAGE_ROOT / "io" / "hdf5.py",
    PACKAGE_ROOT / "io" / "measurement_set.py",
    PACKAGE_ROOT / "io" / "result_errors.py",
    PACKAGE_ROOT / "io" / "result_format.py",
    PACKAGE_ROOT / "io" / "standard_visibility.py",
    PACKAGE_ROOT / "io" / "summary_json.py",
    PACKAGE_ROOT / "io" / "uvfits.py",
    PACKAGE_ROOT / "io" / "workflow_artifacts.py",
    PACKAGE_ROOT / "cli" / "workflow.py",
    PACKAGE_ROOT / "api" / "simulator.py",
)


def _iter_package_sources() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


# ---------------------------------------------------------------------------
# Removed modules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_name", REMOVED_MODULES)
def test_removed_modules_have_no_source_file(module_name: str) -> None:
    relative = Path(*module_name.split(".")[1:]).with_suffix(".py")
    assert not (PACKAGE_ROOT / relative).exists()


@pytest.mark.parametrize("module_name", REMOVED_MODULES)
def test_removed_modules_are_not_importable(module_name: str) -> None:
    sys.modules.pop(module_name, None)
    assert importlib.util.find_spec(module_name) is None
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", REMOVED_MODULES)
def test_removed_modules_are_not_referenced_by_active_source(module_name: str) -> None:
    for path in _iter_package_sources():
        assert module_name not in path.read_text(encoding="utf-8"), path


def test_io_package_docstring_lists_no_removed_submodule() -> None:
    docstring = radiosim_io.__doc__ or ""
    assert "writers" not in docstring


# ---------------------------------------------------------------------------
# Removed public names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", REMOVED_PUBLIC_NAMES)
def test_removed_names_are_absent_from_public_namespaces(name: str) -> None:
    for namespace, module in PUBLIC_NAMESPACES:
        assert not hasattr(module, name), namespace
        assert name not in getattr(module, "__all__", ()), namespace
        with pytest.raises(AttributeError):
            module.__getattr__(name)


@pytest.mark.parametrize("name", REMOVED_PUBLIC_NAMES)
def test_removed_names_are_absent_from_owning_modules(name: str) -> None:
    assert not hasattr(measurement_set_module, name)
    assert not hasattr(hdf5_module, name)


@pytest.mark.parametrize("name", REMOVED_PUBLIC_NAMES)
def test_removed_names_are_defined_nowhere_in_the_package(name: str) -> None:
    for path in _iter_package_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                assert node.name != name, path
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                assert node.id != name, path


def test_simulator_exposes_only_the_singular_result() -> None:
    assert not hasattr(Simulator, "results")
    assert isinstance(Simulator.result, property)


# ---------------------------------------------------------------------------
# Exact surviving surface
# ---------------------------------------------------------------------------


def test_public_result_exports_are_exact() -> None:
    assert radiosim_api.__all__ == [
        "Simulator",
        "SimulationResult",
        "LoadedSimulationResult",
        "ObservationTimeGrid",
        "PhaseCenter",
        "ResultFormat",
    ]
    for name in radiosim_api.__all__:
        assert getattr(radiosim, name) is getattr(radiosim_api, name)

    for name in (
        "HDF5ReadLimits",
        "StandardVisibilityData",
        "write_result_hdf5",
        "load_result_hdf5",
        "write_result_summary_json",
        "write_measurement_set",
        "read_measurement_set",
        "write_uvfits",
        "read_uvfits",
    ):
        assert name in radiosim_io.__all__
        assert getattr(radiosim_io, name) is not None


def test_canonical_result_formats_are_exactly_four() -> None:
    assert [member.value for member in ResultFormat] == [
        "hdf5",
        "summary_json",
        "ms",
        "uvfits",
    ]
    assert "json" not in {member.value for member in ResultFormat}


def test_canonical_schema_versions_are_pinned() -> None:
    assert hdf5_module.SCHEMA_NAME == "radiosim.visibility"
    # Tier 6G bumped the HDF5 schema to 3.0.0 for component provenance.
    assert hdf5_module.SCHEMA_VERSION == "3.0.0"
    summary_source = (PACKAGE_ROOT / "io" / "summary_json.py").read_text(
        encoding="utf-8"
    )
    assert '"radiosim.result-summary"' in summary_source


def test_public_result_signatures_are_exact() -> None:
    # Tier 6E removed ``n_workers`` (plan Section 12.1): solver concurrency is
    # declared once in ``execution.solver.workers``, and ``progress`` became
    # keyword-only with it.
    assert str(inspect.signature(Simulator.run)) == (
        "(self, *, progress: 'bool' = True) -> 'SimulationResult'"
    )
    assert str(inspect.signature(Simulator.save)) == (
        "(self, path: 'str | Path', /, *, format: 'ResultFormat' = "
        "<ResultFormat.HDF5: 'hdf5'>, overwrite: 'bool' = False) -> 'Path'"
    )


# ---------------------------------------------------------------------------
# Removed configuration inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field_name", "value", "expected"),
    REMOVED_WORKFLOW_FIELDS,
    ids=[entry[0] for entry in REMOVED_WORKFLOW_FIELDS],
)
def test_removed_workflow_fields_are_rejected(
    field_name: str,
    value: object,
    expected: str,
) -> None:
    from radiosim.io.config import CliWorkflowConfig

    with pytest.raises(ValueError, match=expected.replace(".", r"\.")):
        _ = CliWorkflowConfig(**{field_name: value})


def test_removed_result_format_value_is_rejected() -> None:
    from radiosim.io.config import CliWorkflowConfig

    with pytest.raises(ValueError, match=r"workflow\.result_format=json"):
        _ = CliWorkflowConfig(result_format="json")


@pytest.mark.parametrize("path", SHIPPED_CONFIGS, ids=lambda path: path.name)
def test_shipped_configs_carry_no_removed_workflow_field(path: Path) -> None:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    workflow = document.get("workflow") or {}
    for field_name, _value, _expected in REMOVED_WORKFLOW_FIELDS:
        assert field_name not in workflow, field_name
    assert workflow.get("result_format") != "json"
    parsed = RadioSimConfig.model_validate(document)
    assert parsed.workflow is not None


# ---------------------------------------------------------------------------
# No compatibility or unsafe evaluation path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path", TIER4_ACTIVE_SOURCES, ids=lambda path: f"{path.parent.name}/{path.name}"
)
def test_active_result_sources_use_no_dynamic_evaluation(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden = {"eval", "exec", "literal_eval", "loads", "load"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                assert target.id not in {"eval", "exec"}, path
            elif isinstance(target, ast.Attribute) and target.attr in forbidden:
                owner = target.value
                owner_name = owner.id if isinstance(owner, ast.Name) else ""
                assert owner_name not in {"pickle", "ast", "marshal", "shelve"}, path
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in {"pickle", "marshal", "shelve"}, path
        if isinstance(node, ast.ImportFrom):
            assert node.module not in {"pickle", "marshal", "shelve"}, path


def test_no_compatibility_or_adapter_module_remains() -> None:
    for path in _iter_package_sources():
        stem = path.stem.lower()
        assert "legacy" not in stem, path
        assert "deprecated" not in stem, path
        assert stem not in {"compat", "compatibility", "adapters_compat"}, path


# ---------------------------------------------------------------------------
# Dependency matrix
# ---------------------------------------------------------------------------


def test_dask_ms_is_absent_from_the_python_manifest() -> None:
    manifest = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    optional = manifest["project"]["optional-dependencies"]
    for extra, requirements in optional.items():
        for requirement in requirements:
            normalized = requirement.replace("_", "-").lower()
            assert not normalized.startswith("dask-ms"), f"{extra}: {requirement}"
    assert "python-casacore>=3.5" in optional["ms"]


def test_dask_ms_is_absent_from_the_pixi_manifest_and_lock() -> None:
    assert "dask-ms" not in PIXI_MANIFEST_PATH.read_text(encoding="utf-8")
    lock_text = PIXI_LOCK_PATH.read_text(encoding="utf-8")
    assert "dask-ms" not in lock_text
    assert "dask_ms" not in lock_text
    assert "daskms" not in lock_text


def test_locked_environment_and_platform_matrix_is_unchanged() -> None:
    manifest = tomllib.loads(PIXI_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["workspace"]["platforms"] == ["linux-64", "osx-64", "osx-arm64"]
    assert manifest["environments"] == {"default": ["py311"], "py312": ["py312"]}
    assert manifest["dependencies"]["pyuvdata"] == "==3.2.1"

    lock_text = PIXI_LOCK_PATH.read_text(encoding="utf-8")
    for marker in (
        "\n  default:\n",
        "\n  py312:\n",
        "\n      linux-64:\n",
        "\n      osx-64:\n",
        "\n      osx-arm64:\n",
    ):
        assert marker in lock_text, marker
    assert "pyuvdata-3.2.1-" in lock_text


def test_base_import_graph_excludes_optional_result_dependencies() -> None:
    code = (
        "import sys, radiosim, radiosim.api, radiosim.io; "
        "forbidden={'pyuvdata','casacore','daskms','dask_ms','xarray'}; "
        "print(sorted(name for name in forbidden if name in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "[]"


@pytest.mark.parametrize("module_name", REMOVED_MODULES)
def test_removed_modules_fail_in_a_fresh_process(module_name: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "ModuleNotFoundError" in completed.stderr


@pytest.mark.parametrize("name", REMOVED_PUBLIC_NAMES)
def test_removed_names_fail_in_a_fresh_process(name: str) -> None:
    code = (
        "import radiosim, radiosim.api, radiosim.io\n"
        "for module in (radiosim, radiosim.api, radiosim.io):\n"
        f"    assert not hasattr(module, {name!r}), module.__name__\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
