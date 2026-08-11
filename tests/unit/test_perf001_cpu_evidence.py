"""Clean-source PERF-001 CPU evidence generation and validation contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tarfile
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import cast

import numpy as np
import pytest

from radiosim.backends import get_backend
from radiosim.benchmarks import (
    PERF001_CPU_BACKENDS,
    PERF001_CPU_CANONICAL_INPUT_IDENTITIES,
    PERF001_CPU_WORKLOADS,
    PERF001_REFERENCE_SOURCE_SHA,
    BenchmarkRecordError,
    WorkloadShape,
    authenticate_perf001_references,
    build_perf001_workload_record,
    parse_perf001_evidence_document,
    perf001_control_identity_sha256,
    perf001_input_identity_sha256,
    time_backend_call,
    validate_perf001_cpu_evidence_document,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CPU_EVIDENCE_TOOL = REPOSITORY_ROOT / "tools/wp7_perf001_cpu_evidence.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "wp7_perf001_cpu_evidence_test", CPU_EVIDENCE_TOOL
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _perf001_source_object(relative: str) -> bytes:
    """Read a phase-varying path from the generating source commit ``S``."""
    if PERF001_REFERENCE_SOURCE_SHA:
        source_shas = set(PERF001_REFERENCE_SOURCE_SHA.values())
        assert len(source_shas) == 1
        source_sha = next(iter(source_shas))
    else:
        source_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    return subprocess.run(
        ["git", "show", f"{source_sha}:{relative}"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=True,
    ).stdout


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    return _load_tool()


def test_clean_cpu_inventory_literals_are_frozen() -> None:
    assert PERF001_CPU_BACKENDS == ("numpy", "jax", "dask")
    assert PERF001_CPU_WORKLOADS == (
        "point_unpolarized_1time_2freq",
        "point_polarized_2times",
        "point_gaussian_morphology",
        "healpix_scalar",
        "healpix_polarized",
        "hybrid_point_plus_healpix",
        "heterogeneous_receptor_bases",
        "point_scaled_4096_sources_4times",
    )


def test_tool_owns_the_exact_canonical_backend_parity_point_arrays(
    tool: ModuleType,
) -> None:
    arrays = tool._canonical_point_source_arrays(
        np,
        lst_rad=0.5,
        count=2,
        polarized=True,
        gaussian=True,
    )
    np.testing.assert_array_equal(arrays["ra_rad"], np.array([0.5, 0.51]))
    np.testing.assert_array_equal(arrays["dec_rad"], np.array([-0.536, -0.526]))
    np.testing.assert_array_equal(arrays["flux"], np.array([2.0, 1.0]))
    np.testing.assert_array_equal(arrays["spectral_index"], np.array([-0.7, -0.8]))
    np.testing.assert_array_equal(arrays["stokes_q"], np.array([0.2, 0.0]))
    np.testing.assert_array_equal(arrays["stokes_u"], np.array([0.0, 0.1]))
    np.testing.assert_array_equal(arrays["stokes_v"], np.array([0.05, 0.0]))
    np.testing.assert_array_equal(arrays["major_arcsec"], np.array([120.0, 120.0]))
    scaled = tool._canonical_point_source_arrays(
        np,
        lst_rad=0.5,
        count=4,
        polarized=False,
        gaussian=False,
        seed=20260731,
        spread=True,
    )
    np.testing.assert_array_equal(
        scaled["spectral_index"], np.full(4, -0.7, dtype=np.float64)
    )


def test_canonical_identity_pins_match_between_runtime_and_stdlib_tool(
    tool: ModuleType,
) -> None:
    assert tool.CPU_CANONICAL_INPUT_IDENTITIES == dict(
        PERF001_CPU_CANONICAL_INPUT_IDENTITIES
    )
    assert len(tool.CPU_CANONICAL_INPUT_IDENTITIES) == 20


def test_scientific_identity_binds_all_runtime_fixture_metadata() -> None:
    manifest = {
        "schema_version": "radiosim.perf001.fixture.cpu_workload.v1",
        "configuration": {"execution": {"precision": "standard"}},
        "antenna_layout_sha256": "a" * 64,
        "location_geodetic": {
            "longitude_deg": 21.4283,
            "latitude_deg": -30.72152,
            "height_m": 1073.0,
        },
        "beam_configuration": {"kind": "circular_aperture"},
        "receptor_configuration": {"output_basis": "linear"},
        "healpix_metadata": {
            "nside": 4,
            "ordering": "ring",
            "coordinate_frame": "icrs",
            "i_unit": "K",
            "i_brightness_conversion": None,
        },
    }
    inputs = (("values", np.array([1.0], dtype=np.float64)),)
    expected = perf001_input_identity_sha256(manifest, inputs)
    mutations = (
        ("configuration", {"execution": {"precision": "extended"}}),
        ("antenna_layout_sha256", "b" * 64),
        (
            "location_geodetic",
            {"longitude_deg": 0.0, "latitude_deg": 0.0, "height_m": 0.0},
        ),
        ("beam_configuration", {"kind": "uniform"}),
        ("receptor_configuration", {"output_basis": "circular"}),
        (
            "healpix_metadata",
            {
                "nside": 8,
                "ordering": "nest",
                "coordinate_frame": "galactic",
                "i_unit": "K",
                "i_brightness_conversion": "rayleigh-jeans",
            },
        ),
    )
    for field, replacement in mutations:
        changed = deepcopy(manifest)
        changed[field] = replacement
        assert perf001_input_identity_sha256(changed, inputs) != expected


def test_control_identity_is_manifest_only_canonical_and_domain_separated() -> None:
    manifest = {
        "schema_version": "radiosim.perf001.control.backend_resolution.v1",
        "operation": "get_backend_auto",
        "requested_backend": "auto",
    }
    reordered = {
        "requested_backend": "auto",
        "operation": "get_backend_auto",
        "schema_version": "radiosim.perf001.control.backend_resolution.v1",
    }

    digest = perf001_control_identity_sha256(manifest)

    assert digest == perf001_control_identity_sha256(reordered)
    assert len(digest) == 64
    assert digest != perf001_control_identity_sha256(
        {
            **manifest,
            "operation": "get_device_resources_default",
            "requested_backend": "default",
        }
    )


@pytest.mark.parametrize(
    "manifest",
    [
        {},
        {"schema_version": ""},
        {
            "schema_version": "radiosim.perf001.control.v1",
            "operation": "get_backend_auto",
            "requested_backend": "auto",
        },
        {"schema_version": "radiosim.perf001.control.v1", "sentinel": [0]},
    ],
)
def test_control_identity_rejects_noncanonical_or_sentinel_manifests(
    manifest: dict[str, object],
) -> None:
    with pytest.raises(BenchmarkRecordError):
        perf001_control_identity_sha256(manifest)


def test_authorized_cpu_evidence_tool_exists_without_test_package_imports() -> None:
    source = CPU_EVIDENCE_TOOL.read_text(encoding="utf-8")

    assert "from tests" not in source
    assert "import tests" not in source
    assert "performance-test-only" not in source
    assert "stand-in" not in source
    assert "HealpixData" in source
    assert "AltAz" in source
    assert 'getattr(module, "__spec__", None)' in source


def test_tool_import_isolated_from_runtime_and_science_packages() -> None:
    script = f"""
import importlib.util
import json
import sys
before = set(sys.modules)
spec = importlib.util.spec_from_file_location('isolated_perf001', {str(CPU_EVIDENCE_TOOL)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
introduced = sorted(set(sys.modules) - before)
forbidden = sorted({{'radiosim', 'numpy', 'jax', 'jaxlib', 'astropy', 'dask'}} & {{name.split('.')[0] for name in introduced}})
print(json.dumps({{'forbidden': forbidden}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {"forbidden": []}


def _valid_artifact(tool: ModuleType, root: Path) -> tuple[str, str, str]:
    from tests.unit.test_perf001_runtime_acceptance import _cpu_document, _provenance

    source_sha = "1" * 40
    provenance = _provenance(
        git_sha=source_sha,
        pixi_lock_sha256=tool.PIXI_LOCK_SHA256,
    )
    document = _cpu_document(provenance)
    relative = "output/benchmarks/reference/perf001/20260811T000000Z-darwin-arm64.json"
    artifact = root / relative
    artifact.parent.mkdir(parents=True)
    raw = (
        json.dumps(document.to_json_safe(), allow_nan=False, indent=2, sort_keys=False)
        + "\n"
    ).encode("utf-8")
    artifact.write_bytes(raw)
    return relative, source_sha, hashlib.sha256(raw).hexdigest()


def test_stdlib_loader_validates_the_complete_cpu_document(
    tool: ModuleType, tmp_path: Path
) -> None:
    relative, source_sha, digest = _valid_artifact(tool, tmp_path)

    document, raw, observed_digest = tool.load_and_validate_artifact(
        relative,
        approved_source_sha=source_sha,
        artifact_sha256=digest,
        repository_root=tmp_path,
    )

    assert observed_digest == digest == hashlib.sha256(raw).hexdigest()
    assert tuple(len(document[name]) for name in tool.DOCUMENT_FIELDS[1:]) == (
        24,
        8,
        4,
        6,
        3,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_key",
        "nonfinite",
        "extra_field",
        "missing_field",
        "wrong_source",
        "wrong_lock",
        "wrong_count",
        "pc_precision",
    ],
)
def test_stdlib_loader_rejects_strict_semantic_mutations(
    tool: ModuleType,
    tmp_path: Path,
    mutation: str,
) -> None:
    relative, source_sha, _ = _valid_artifact(tool, tmp_path)
    artifact = tmp_path / relative
    raw = artifact.read_bytes()
    if mutation == "duplicate_key":
        raw = raw.replace(
            b'{\n  "schema_version":',
            b'{\n  "schema_version": "duplicate",\n  "schema_version":',
            1,
        )
    elif mutation == "nonfinite":
        raw = raw.replace(b'"setup_seconds": 0.5', b'"setup_seconds": NaN', 1)
    else:
        decoded = json.loads(raw)
        if mutation == "extra_field":
            decoded["extra"] = True
        elif mutation == "missing_field":
            del decoded["backend_resolution"]
        elif mutation == "wrong_source":
            for collection in tool.DOCUMENT_FIELDS[1:]:
                for row in decoded[collection]:
                    row["provenance"]["git_sha"] = "2" * 40
        elif mutation == "wrong_lock":
            for collection in tool.DOCUMENT_FIELDS[1:]:
                for row in decoded[collection]:
                    row["provenance"]["pixi_lock_sha256"] = "2" * 64
        elif mutation == "wrong_count":
            decoded["workload_benchmarks"].pop()
        elif mutation == "pc_precision":
            decoded["backend_resolution"][0]["context"]["result_dtype"] = "complex128"
        raw = (json.dumps(decoded, allow_nan=False, indent=2) + "\n").encode()
    artifact.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()

    with pytest.raises(tool.CpuEvidenceError):
        tool.load_and_validate_artifact(
            relative,
            approved_source_sha=source_sha,
            artifact_sha256=digest,
            repository_root=tmp_path,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "provenance_unknown",
        "timestamp_negative_zero",
        "workload_backend_version",
        "workload_dimension",
        "workload_timing_order",
        "workload_unmeasured_type",
        "workload_identity_spoof",
        "memory_pair_counts",
        "memory_chunk_string",
        "memory_peak_bool",
        "memory_target_float",
        "memory_pair_identity",
        "memory_pair_context",
        "memory_pair_precision_same",
        "solver_count_string",
        "solver_kernel_policy",
        "solver_target_float",
        "solver_pair_identity",
        "solver_pair_dimension",
        "solver_pair_dimension_same",
        "solver_pair_precision_same",
        "signature_call_count",
        "signature_geometry",
        "retracing_leaf_count",
        "retracing_total",
        "retracing_pair_identity",
        "retracing_pair_scope",
        "retracing_pair_scope_same",
        "retracing_pair_precision_same",
        "backend_context_actual",
        "backend_context_version",
        "backend_sample_count_float",
        "backend_summary_string",
    ),
)
def test_stdlib_and_production_validators_reject_the_same_semantic_mutations(
    tool: ModuleType,
    tmp_path: Path,
    mutation: str,
) -> None:
    relative, source_sha, _ = _valid_artifact(tool, tmp_path)
    artifact = tmp_path / relative
    decoded = json.loads(artifact.read_bytes())
    workloads = decoded["workload_benchmarks"]
    memory = decoded["memory_scaling"]
    solver = decoded["solver_memory"]
    retracing = decoded["retracing"]
    backend = decoded["backend_resolution"]
    if mutation == "provenance_unknown":
        for collection in tool.DOCUMENT_FIELDS[1:]:
            for row in decoded[collection]:
                row["provenance"]["cpu_model"] = "unknown"
    elif mutation == "timestamp_negative_zero":
        for collection in tool.DOCUMENT_FIELDS[1:]:
            for row in decoded[collection]:
                row["provenance"]["recorded_at_utc"] = "2026-08-11T00:00:00-00:00"
    elif mutation == "workload_backend_version":
        workloads[1]["context"]["backend_version"] = "forged"
    elif mutation == "workload_dimension":
        workloads[1]["n_point_sources"] += 1
    elif mutation == "workload_timing_order":
        workloads[0]["steady_state_median_seconds"] = 2.0
    elif mutation == "workload_unmeasured_type":
        workloads[0]["unmeasured"].append(1)
    elif mutation == "workload_identity_spoof":
        for row in workloads[:3]:
            row["context"]["input_identity_sha256"] = "e" * 64
    elif mutation == "memory_pair_counts":
        memory[0]["kernel_pair_counts"] = [memory[0]["logical_pair_count"] + 1]
    elif mutation == "memory_chunk_string":
        memory[0]["kernel_baseline_chunks"] = ["100"]
    elif mutation == "memory_peak_bool":
        memory[0]["peak_host_bytes"] = False
    elif mutation == "memory_target_float":
        memory[1]["target_kernel_pairs"] = 131072.0
    elif mutation == "memory_pair_identity":
        memory[1]["context"]["input_identity_sha256"] = "e" * 64
    elif mutation == "memory_pair_context":
        memory[1]["context"]["precision_output"] = "float32"
    elif mutation == "memory_pair_precision_same":
        for row in memory[:2]:
            row["context"]["result_dtype"] = "complex64"
    elif mutation == "solver_count_string":
        solver[0]["logical_source_counts"] = ["3"]
    elif mutation == "solver_kernel_policy":
        solver[1]["kernel_source_counts"] = [3]
    elif mutation == "solver_target_float":
        solver[0]["target_kernel_pairs"] = 131072.0
    elif mutation == "solver_pair_identity":
        solver[1]["context"]["input_identity_sha256"] = "e" * 64
    elif mutation == "solver_pair_dimension":
        solver[1]["logical_n_baselines"] += 1
    elif mutation == "solver_pair_dimension_same":
        for row in solver[:2]:
            row["logical_n_baselines"] += 1
    elif mutation == "solver_pair_precision_same":
        for row in solver[:2]:
            row["context"]["result_dtype"] = "complex64"
    elif mutation == "signature_call_count":
        retracing[0]["observed_signatures"][0]["call_count"] = "2"
    elif mutation == "signature_geometry":
        retracing[0]["observed_signatures"][0]["jones_p_shape"][-1] = 3
    elif mutation == "retracing_leaf_count":
        retracing[0]["leaf_call_count"] += 1
    elif mutation == "retracing_total":
        retracing[0]["scope_total_seconds"] += 1.0
    elif mutation == "retracing_pair_identity":
        retracing[1]["context"]["input_identity_sha256"] = "e" * 64
    elif mutation == "retracing_pair_scope":
        retracing[1]["measurement_scope"] = "different_scope"
    elif mutation == "retracing_pair_scope_same":
        for row in retracing[:2]:
            row["measurement_scope"] = "different_scope"
    elif mutation == "retracing_pair_precision_same":
        for row in retracing[:2]:
            row["context"]["result_dtype"] = "complex64"
    elif mutation == "backend_context_actual":
        backend[0]["context"]["backend_actual"] = "forged"
    elif mutation == "backend_context_version":
        backend[1]["context"]["backend_version"] = "forged"
    elif mutation == "backend_sample_count_float":
        backend[0]["fresh_process_samples"] = 3.0
    elif mutation == "backend_summary_string":
        backend[0]["minimum_seconds"] = str(backend[0]["minimum_seconds"])
    raw = (json.dumps(decoded, allow_nan=False, indent=2) + "\n").encode()
    artifact.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()

    with pytest.raises(BenchmarkRecordError):
        production_document = parse_perf001_evidence_document(raw)
        validate_perf001_cpu_evidence_document(production_document)
    with pytest.raises(tool.CpuEvidenceError):
        tool.load_and_validate_artifact(
            relative,
            approved_source_sha=source_sha,
            artifact_sha256=digest,
            repository_root=tmp_path,
        )


def test_stdlib_loader_rejects_noncanonical_paths_and_symlinks(
    tool: ModuleType, tmp_path: Path
) -> None:
    relative, source_sha, digest = _valid_artifact(tool, tmp_path)
    artifact = tmp_path / relative
    symlink = artifact.with_name("20260811T000001Z-darwin-arm64.json")
    symlink.symlink_to(artifact)

    for candidate in (
        str(artifact),
        "output/benchmarks/reference/perf001/nested/record.json",
        symlink.relative_to(tmp_path).as_posix(),
    ):
        with pytest.raises(tool.CpuEvidenceError):
            tool.load_and_validate_artifact(
                candidate,
                approved_source_sha=source_sha,
                artifact_sha256=digest,
                repository_root=tmp_path,
            )


def test_stdlib_loader_rejects_an_intermediate_namespace_symlink(
    tool: ModuleType, tmp_path: Path
) -> None:
    real_root = tmp_path / "real"
    relative, source_sha, digest = _valid_artifact(tool, real_root)
    linked_root = tmp_path / "linked"
    linked_root.mkdir()
    (linked_root / "output").symlink_to(real_root / "output", target_is_directory=True)

    with pytest.raises(tool.CpuEvidenceError, match="symlink component"):
        tool.load_and_validate_artifact(
            relative,
            approved_source_sha=source_sha,
            artifact_sha256=digest,
            repository_root=linked_root,
        )


def test_generation_preflight_rejects_an_intermediate_namespace_symlink(
    tool: ModuleType,
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real"
    (real_root / "output/benchmarks/reference/perf001").mkdir(parents=True)
    linked_root = tmp_path / "linked"
    linked_root.mkdir()
    (linked_root / "output").symlink_to(real_root / "output", target_is_directory=True)

    with pytest.raises(tool.CpuEvidenceError, match="symlink component"):
        tool._require_empty_reference_namespace(linked_root)


def test_stdlib_loader_rejects_filename_system_that_disagrees_with_provenance(
    tool: ModuleType,
    tmp_path: Path,
) -> None:
    relative, source_sha, digest = _valid_artifact(tool, tmp_path)
    wrong_relative = relative.replace("-darwin-", "-linux-")
    (tmp_path / wrong_relative).write_bytes((tmp_path / relative).read_bytes())

    with pytest.raises(tool.CpuEvidenceError, match="system differs"):
        tool.load_and_validate_artifact(
            wrong_relative,
            approved_source_sha=source_sha,
            artifact_sha256=digest,
            repository_root=tmp_path,
        )


class _PreflightRunner:
    def __init__(self, root: Path, source_sha: str) -> None:
        self.root = root
        self.source_sha = source_sha
        self.dirty = False
        self.lock_success = True

    def __call__(
        self, command: tuple[str, ...], cwd: Path
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == self.root
        if command[:3] == ("git", "rev-parse", "--show-toplevel"):
            stdout = f"{self.root}\n"
        elif command[:3] == ("git", "rev-parse", "HEAD"):
            stdout = f"{self.source_sha}\n"
        elif command[:3] == ("git", "status", "--porcelain=v1"):
            stdout = " M dirty\n" if self.dirty else ""
        elif command[:2] == ("git", "ls-files"):
            stdout = ""
        else:
            return subprocess.CompletedProcess(
                command, 0 if self.lock_success else 1, "", "stale lock"
            )
        return subprocess.CompletedProcess(command, 0, stdout, "")


def _preflight_dependencies(tool: ModuleType) -> tuple[object, _PreflightRunner]:
    root = REPOSITORY_ROOT
    source_sha = "1" * 40
    runner = _PreflightRunner(root, source_sha)
    default_prefix = root / ".pixi/envs/default"
    default_executable = default_prefix / "bin/python"
    environment = {
        "PIXI_ENVIRONMENT_NAME": "default",
        "PIXI_PROJECT_ROOT": str(root),
        "PIXI_PROJECT_MANIFEST": str(root / "pixi.toml"),
        "CONDA_PREFIX": str(default_prefix),
        "PIXI_EXE": str(default_executable),
    }
    dependencies = tool.PreflightDependencies(
        repository_root=root,
        cwd=root,
        environ=environment,
        prefix=default_prefix,
        executable=default_executable,
        run_command=runner,
        package_identity_check=lambda _dependencies: None,
    )
    return dependencies, runner


def test_preflight_boundaries_are_injectable_and_default_only(
    tool: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies, runner = _preflight_dependencies(tool)
    monkeypatch.setattr(tool, "_require_empty_reference_manifests", lambda _root: None)
    monkeypatch.setattr(tool, "_require_empty_reference_namespace", lambda _root: None)

    assert tool.preflight_generation("1" * 40, dependencies=dependencies) == "1" * 40

    runner.dirty = True
    with pytest.raises(tool.CpuEvidenceError, match="clean worktree"):
        tool.preflight_generation("1" * 40, dependencies=dependencies)
    runner.dirty = False

    wrong_environment = replace(
        dependencies,
        environ={**dependencies.environ, "PIXI_ENVIRONMENT_NAME": "py312"},
    )
    with pytest.raises(tool.CpuEvidenceError, match="Pixi default"):
        tool.preflight_generation("1" * 40, dependencies=wrong_environment)

    runner.lock_success = False
    with pytest.raises(tool.CpuEvidenceError, match="Pixi lock check"):
        tool.preflight_generation("1" * 40, dependencies=dependencies)


def test_default_package_identity_matches_the_lock_and_live_prefix(
    tool: ModuleType,
) -> None:
    pixi_executable = shutil.which("pixi")
    assert pixi_executable is not None
    default_prefix = REPOSITORY_ROOT / ".pixi/envs/default"
    dependencies = tool.PreflightDependencies(
        repository_root=REPOSITORY_ROOT,
        cwd=REPOSITORY_ROOT,
        environ={"PIXI_EXE": pixi_executable},
        prefix=default_prefix,
        executable=default_prefix / "bin/python",
        run_command=tool._run_command,
        package_identity_check=tool._require_cpu_package_identity,
    )

    tool._require_cpu_package_identity(dependencies)


def test_live_package_inventory_detects_installed_prefix_drift(
    tool: ModuleType,
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    prefix = root / ".pixi/envs/default"
    conda_meta = prefix / "conda-meta"
    site_packages = prefix / "lib/python3.11/site-packages"
    dist_info = site_packages / "radiosim-1.2.3.dist-info"
    conda_meta.mkdir(parents=True)
    dist_info.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "radiosim"\nversion = "1.2.3"\n', encoding="utf-8"
    )
    conda_url = "https://example.invalid/osx-arm64/python-3.11.conda"
    conda_digest = "a" * 64
    (conda_meta / "python.json").write_text(
        json.dumps(
            {
                "url": conda_url,
                "sha256": conda_digest,
                "files": [
                    "lib/python3.11/site-packages/python_owned-1.0.dist-info/METADATA"
                ],
            }
        ),
        encoding="utf-8",
    )
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: radiosim\nVersion: 1.2.3\n",
        encoding="utf-8",
    )
    (dist_info / "INSTALLER").write_text("uv-pixi\n", encoding="utf-8")
    (dist_info / "direct_url.json").write_text(
        json.dumps({"url": root.resolve().as_uri(), "dir_info": {"editable": True}}),
        encoding="utf-8",
    )
    locked_rows = [
        {
            "kind": "conda",
            "name": "python",
            "url": conda_url,
            "sha256": conda_digest,
        },
        {
            "kind": "pypi",
            "name": "radiosim",
            "version": None,
            "url": "./",
            "requested_spec": '{ path = ".", editable = true }',
        },
    ]

    tool._require_live_environment_matches(
        root=root, prefix=prefix, locked_rows=locked_rows
    )
    (dist_info / "INSTALLER").write_text("pip\n", encoding="utf-8")
    with pytest.raises(tool.CpuEvidenceError, match="uv-pixi"):
        tool._require_live_environment_matches(
            root=root, prefix=prefix, locked_rows=locked_rows
        )


def test_generation_preflight_failure_never_reaches_measurement_or_writer(
    tool: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    dependencies, runner = _preflight_dependencies(tool)
    runner.dirty = True
    reached: list[str] = []
    monkeypatch.setattr(
        tool,
        "_measure_document",
        lambda *_args, **_kwargs: reached.append("measurement"),
    )

    with pytest.raises(tool.CpuEvidenceError, match="clean worktree"):
        tool.generate("1" * 40, dependencies=dependencies)

    assert reached == []


def test_workload_builder_binds_request_and_standard_precision() -> None:
    from tests.unit.test_perf001_runtime_acceptance import _provenance

    shape = WorkloadShape(
        workload="fixture",
        n_antennas=2,
        n_baselines=3,
        n_point_sources=1,
        n_healpix_pixels=0,
        n_times=1,
        n_frequencies=1,
        sky_representation="point_sources",
        solver_workers=1,
        loader_max_workers=0,
    )
    manifest = {
        "schema_version": "radiosim.perf001.fixture.builder-test.v1",
        "fixture": "builder-test",
    }
    inputs = (("values", np.ones(1, dtype=np.float64)),)

    numpy_backend = get_backend("numpy")
    numpy_timing = time_backend_call(
        lambda: numpy_backend.asarray(np.ones(1, dtype=np.complex128)),
        backend=numpy_backend,
    )
    with pytest.raises(BenchmarkRecordError, match="does not match"):
        build_perf001_workload_record(
            provenance=_provenance(),
            backend=numpy_backend,
            requested="gpu",
            shape=shape,
            timing=numpy_timing,
            numpy_reference=numpy_timing.host_result,
            fixture_manifest=manifest,
            logical_inputs=inputs,
            notes="request mismatch",
        )

    fast_backend = get_backend("numpy", precision="fast")
    fast_timing = time_backend_call(
        lambda: fast_backend.asarray(np.ones(1, dtype=np.complex64)),
        backend=fast_backend,
    )
    with pytest.raises(BenchmarkRecordError, match="standard precision"):
        build_perf001_workload_record(
            provenance=_provenance(),
            backend=fast_backend,
            requested="numpy",
            shape=shape,
            timing=fast_timing,
            numpy_reference=fast_timing.host_result,
            fixture_manifest=manifest,
            logical_inputs=inputs,
            notes="precision mismatch",
        )


def test_simulator_control_identity_binds_executed_config_and_layout(
    tmp_path: Path,
) -> None:
    import radiosim.benchmarks.harness as harness
    from tests.unit.test_perf001_runtime_acceptance import _provenance

    layout = tmp_path / "antennas.txt"
    layout.write_text("canonical layout\n", encoding="utf-8")
    configuration = {"fixture": "canonical"}
    manifest = {
        "schema_version": "radiosim.perf001.control.backend_resolution.v1",
        "operation": "simulator_setup_auto",
        "requested_backend": "auto",
        "fixture": "canonical_minimal_simulator_v1",
        "configuration": configuration,
        "antenna_layout_sha256": hashlib.sha256(layout.read_bytes()).hexdigest(),
    }

    with pytest.raises(BenchmarkRecordError, match="runtime configuration"):
        harness.measure_perf001_backend_resolution(
            provenance=_provenance(),
            operation="simulator_setup_auto",
            control_manifest=manifest,
            repository_root=REPOSITORY_ROOT,
            simulator_configuration={"fixture": "different"},
            simulator_base_dir=tmp_path,
            fresh_process_samples=1,
        )

    layout.write_text("changed layout\n", encoding="utf-8")
    with pytest.raises(BenchmarkRecordError, match="antenna layout"):
        harness.measure_perf001_backend_resolution(
            provenance=_provenance(),
            operation="simulator_setup_auto",
            control_manifest=manifest,
            repository_root=REPOSITORY_ROOT,
            simulator_configuration=configuration,
            simulator_base_dir=tmp_path,
            fresh_process_samples=1,
        )


def test_backend_probe_rejects_malformed_value_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import radiosim.benchmarks.harness as harness

    payload = {
        "duration": True,
        "resolved_backend": "numpy-cpu",
        "backend_version": "2.3.2",
        "device_kind": "cpu",
        "compilation_used": False,
        "jax_distribution_installed": True,
        "jax_in_sys_modules_before": False,
        "jax_in_sys_modules_after": False,
        "jaxlib_in_sys_modules_before": False,
        "jaxlib_in_sys_modules_after": False,
        "radiosim_source_file": str(REPOSITORY_ROOT / "src/radiosim/__init__.py"),
        "detail": {"not": "a string"},
    }
    completed = subprocess.CompletedProcess(
        [sys.executable],
        0,
        "RADIOSIM_PERF001_BACKEND_PROBE=" + json.dumps(payload),
        "",
    )
    monkeypatch.setattr(harness.subprocess, "run", lambda *_args, **_kwargs: completed)

    with pytest.raises(BenchmarkRecordError, match="non-boolean|invalid duration"):
        harness._run_perf001_backend_resolution_probe(
            operation="get_backend_auto",
            repository_root=REPOSITORY_ROOT,
            configuration=None,
            configuration_base_dir=REPOSITORY_ROOT,
        )


def test_backend_probe_imports_from_explicit_isolated_source_root(
    tmp_path: Path,
) -> None:
    from radiosim.benchmarks import harness

    source_root = tmp_path / "snapshot"
    shutil.copytree(REPOSITORY_ROOT / "src/radiosim", source_root / "src/radiosim")

    probe = harness._run_perf001_backend_resolution_probe(
        operation="get_backend_auto",
        repository_root=REPOSITORY_ROOT,
        configuration=None,
        configuration_base_dir=REPOSITORY_ROOT,
        loaded_source_root=source_root,
    )

    assert Path(probe["radiosim_source_file"]) == (
        source_root / "src/radiosim/__init__.py"
    )
    assert probe["resolved_backend"] == "numpy-cpu"


def test_reference_authentication_proves_the_direct_source_to_evidence_edge(
    tmp_path: Path,
) -> None:
    from tests.unit.test_perf001_runtime_acceptance import (
        _accelerator,
        _cpu_document,
        _device_memory,
        _provenance,
    )

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "perf001@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "PERF-001 Test"],
        cwd=tmp_path,
        check=True,
    )
    marker = tmp_path / "source.txt"
    marker.write_text("clean source S\n", encoding="utf-8")
    subprocess.run(["git", "add", "source.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "source"], cwd=tmp_path, check=True)
    source_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    document = _cpu_document(_provenance(git_sha=source_sha))
    relative = "output/benchmarks/reference/perf001/20260811T000000Z-darwin-arm64.json"
    artifact = tmp_path / relative
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps(document.to_json_safe(), allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "output"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "evidence"], cwd=tmp_path, check=True)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()

    authenticated = authenticate_perf001_references(
        repository_root=tmp_path,
        expected_sha256={relative: digest},
        expected_source_sha={relative: source_sha},
    )

    assert authenticated[0].sha256 == digest

    uncommitted_document = replace(
        document,
        workload_benchmarks=(
            replace(document.workload_benchmarks[0], notes="uncommitted bytes"),
            *document.workload_benchmarks[1:],
        ),
    )
    artifact.write_text(
        json.dumps(uncommitted_document.to_json_safe(), allow_nan=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    uncommitted_digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    with pytest.raises(BenchmarkRecordError, match="committed HEAD|clean evidence"):
        authenticate_perf001_references(
            repository_root=tmp_path,
            expected_sha256={relative: uncommitted_digest},
            expected_source_sha={relative: source_sha},
        )

    accelerator_rows = []
    for index in range(0, len(document.workload_benchmarks), 3):
        numpy_row, jax_row, _dask_row = document.workload_benchmarks[index : index + 3]
        accelerator_rows.extend(
            (
                numpy_row,
                replace(
                    jax_row,
                    context=replace(
                        jax_row.context,
                        backend_requested="gpu",
                        backend_actual="jax-gpu-gpu",
                        backend_version=jax_row.provenance.jax_version,
                        device_kind="gpu",
                        compilation_used=True,
                        policy_id="gpu_workload_matrix_v1",
                    ),
                    accelerator=_accelerator(),
                    device_memory=_device_memory(),
                    unmeasured=("tpu", "distributed"),
                ),
            )
        )
    accelerator_document = replace(
        document, workload_benchmarks=tuple(accelerator_rows)
    )
    artifact.write_text(
        json.dumps(accelerator_document.to_json_safe(), allow_nan=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "output"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "--amend", "--no-edit", "-q"], cwd=tmp_path, check=True
    )
    accelerator_digest = hashlib.sha256(artifact.read_bytes()).hexdigest()

    authenticated_accelerator = authenticate_perf001_references(
        repository_root=tmp_path,
        expected_sha256={relative: accelerator_digest},
        expected_source_sha={relative: source_sha},
    )

    assert authenticated_accelerator[0].sha256 == accelerator_digest

    nonexistent = "f" * 40
    spoofed = _cpu_document(_provenance(git_sha=nonexistent))
    artifact.write_text(
        json.dumps(spoofed.to_json_safe(), allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "output"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "--amend", "--no-edit", "-q"], cwd=tmp_path, check=True
    )
    spoofed_digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    with pytest.raises(BenchmarkRecordError, match="valid generating commit"):
        authenticate_perf001_references(
            repository_root=tmp_path,
            expected_sha256={relative: spoofed_digest},
            expected_source_sha={relative: nonexistent},
        )


def _commit_snapshot_fixture(root: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "perf001@example.invalid"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "PERF-001 Test"],
        cwd=root,
        check=True,
    )
    fixture_files = {
        "src/radiosim/__init__.py": "__version__ = 'snapshot'\n",
        "tools/wp7_perf001_cpu_evidence.py": "SNAPSHOT_TOOL = True\n",
        "pixi.toml": (REPOSITORY_ROOT / "pixi.toml").read_text(encoding="utf-8"),
        "pixi.lock": (REPOSITORY_ROOT / "pixi.lock").read_text(encoding="utf-8"),
        "pyproject.toml": "[project]\nname = 'snapshot'\nversion = '1.0'\n",
    }
    for relative, content in fixture_files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "source"], cwd=root, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_source_snapshot_is_exactly_authenticated_to_s_and_detects_tampering(
    tool: ModuleType,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    source_sha = _commit_snapshot_fixture(repository)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    snapshot = tool._export_source_snapshot(
        repository_root=repository,
        approved_source_sha=source_sha,
        workspace=workspace,
    )
    tool._verify_source_snapshot(
        snapshot,
        repository_root=repository,
        approved_source_sha=source_sha,
    )

    # A live-checkout edit cannot change the already authenticated measurement tree.
    (repository / "src/radiosim/__init__.py").write_text(
        "__version__ = 'live-edit'\n", encoding="utf-8"
    )
    tool._verify_source_snapshot(
        snapshot,
        repository_root=repository,
        approved_source_sha=source_sha,
    )

    tool._unseal_source_snapshot(snapshot.root)
    (snapshot.root / "src/radiosim/__init__.py").write_text(
        "__version__ = 'snapshot-tamper'\n", encoding="utf-8"
    )
    with pytest.raises(tool.CpuEvidenceError, match="bytes differ"):
        tool._verify_source_snapshot(
            snapshot,
            repository_root=repository,
            approved_source_sha=source_sha,
        )


def test_source_snapshot_archive_rejects_unsafe_members(
    tool: ModuleType,
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "unsafe.tar"
    with tarfile.open(archive_path, mode="w:") as archive:
        member = tarfile.TarInfo("../escape.py")
        payload = b"unsafe\n"
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(tool.CpuEvidenceError, match="unsafe entry"):
        tool._safe_extract_source_archive(archive_path, tmp_path / "source")
    assert not (tmp_path / "escape.py").exists()


def test_snapshot_worker_uses_isolated_site_free_python(
    tool: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    source_sha = "1" * 40
    artifact = (
        tool.REFERENCE_DIRECTORY
        / (
            "20260811T000000Z-"
            f"{tool.platform.system().lower()}-"
            f"{tool.platform.machine().lower()}.json"
        )
    ).as_posix()

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        observed["kwargs"] = kwargs
        summary = {
            "artifact_path": artifact,
            "artifact_sha256": "2" * 64,
            "generating_source_sha": source_sha,
            "row_count": 45,
            "passed": True,
        }
        return subprocess.CompletedProcess(
            command,
            0,
            tool.WORKER_RESULT_PREFIX + json.dumps(summary) + "\n",
            "",
        )

    monkeypatch.setattr(tool.subprocess, "run", run)
    dependencies = tool.PreflightDependencies(
        repository_root=tmp_path,
        cwd=tmp_path,
        environ={"PYTHONPATH": "/forbidden", "PYTHONHOME": "/forbidden"},
        prefix=tmp_path / ".pixi/envs/default",
        executable=tmp_path / ".pixi/envs/default/bin/python",
        run_command=lambda _command, _cwd: subprocess.CompletedProcess([], 0, "", ""),
        package_identity_check=lambda _dependencies: None,
    )
    snapshot = tool.SourceSnapshot(
        root=tmp_path / "snapshot",
        entries=(),
        manifest_sha256="3" * 64,
    )

    result = tool._run_snapshot_worker(
        dependencies=dependencies,
        snapshot=snapshot,
        approved_source_sha=source_sha,
        captured=tool.datetime(2026, 8, 11, tzinfo=tool.UTC),
    )

    command = observed["command"]
    assert isinstance(command, list)
    assert command[1:4] == ["-I", "-S", "-B"]
    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["cwd"] == tmp_path
    assert "PYTHONPATH" not in kwargs["env"]
    assert "PYTHONHOME" not in kwargs["env"]
    assert result["row_count"] == 45


def test_generate_parent_only_preflights_exports_and_runs_snapshot_worker(
    tool: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_sha = "1" * 40
    dependencies, _runner = _preflight_dependencies(tool)
    calls: list[str] = []

    monkeypatch.setattr(
        tool,
        "preflight_generation",
        lambda *_args, **_kwargs: calls.append("preflight") or source_sha,
    )

    def export(**kwargs: object) -> object:
        calls.append("export")
        workspace = kwargs["workspace"]
        assert isinstance(workspace, Path)
        return tool.SourceSnapshot(
            root=workspace / "source",
            entries=(),
            manifest_sha256="2" * 64,
        )

    monkeypatch.setattr(tool, "_export_source_snapshot", export)
    monkeypatch.setattr(
        tool,
        "_run_snapshot_worker",
        lambda **_kwargs: calls.append("worker")
        or {
            "artifact_path": "output/benchmarks/reference/perf001/record.json",
            "artifact_sha256": "3" * 64,
            "generating_source_sha": source_sha,
            "row_count": 45,
            "passed": True,
        },
    )
    monkeypatch.setattr(
        tool,
        "_measure_document",
        lambda *_args, **_kwargs: pytest.fail("parent measured science workloads"),
    )

    result = tool.generate(
        source_sha,
        dependencies=dependencies,
        moment=tool.datetime(2026, 8, 11, tzinfo=tool.UTC),
    )

    assert calls == ["preflight", "export", "worker"]
    assert result["generating_source_sha"] == source_sha


def test_cli_evidence_edge_requires_clean_exact_direct_successor(
    tool: ModuleType,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    source_sha = _commit_snapshot_fixture(repository)
    relative = "output/benchmarks/reference/perf001/20260811T000000Z-darwin-arm64.json"
    artifact = repository / relative
    artifact.parent.mkdir(parents=True)
    raw = b'{"authenticated":true}\n'
    artifact.write_bytes(raw)
    subprocess.run(["git", "add", relative], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "evidence"], cwd=repository, check=True)

    tool._authenticate_cli_evidence_edge(
        input_path=relative,
        approved_source_sha=source_sha,
        raw=raw,
        repository_root=repository,
    )

    generator = repository / "tools/wp7_perf001_cpu_evidence.py"
    generator.write_text("SNAPSHOT_TOOL = False\n", encoding="utf-8")
    subprocess.run(["git", "add", generator], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "--amend", "--no-edit", "-q"],
        cwd=repository,
        check=True,
    )
    with pytest.raises(tool.CpuEvidenceError, match="generator and Pixi lock"):
        tool._authenticate_cli_evidence_edge(
            input_path=relative,
            approved_source_sha=source_sha,
            raw=raw,
            repository_root=repository,
        )
    committed_generator = subprocess.run(
        ["git", "show", f"{source_sha}:tools/wp7_perf001_cpu_evidence.py"],
        cwd=repository,
        capture_output=True,
        check=True,
    ).stdout
    generator.write_bytes(committed_generator)
    subprocess.run(["git", "add", generator], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "--amend", "--no-edit", "-q"],
        cwd=repository,
        check=True,
    )

    artifact.write_bytes(b'{"authenticated":false}\n')
    with pytest.raises(tool.CpuEvidenceError, match="clean evidence checkout"):
        tool._authenticate_cli_evidence_edge(
            input_path=relative,
            approved_source_sha=source_sha,
            raw=raw,
            repository_root=repository,
        )

    artifact.write_bytes(raw)
    with pytest.raises(tool.CpuEvidenceError, match="valid generating commit"):
        tool._authenticate_cli_evidence_edge(
            input_path=relative,
            approved_source_sha="f" * 40,
            raw=raw,
            repository_root=repository,
        )


def test_acceptance_certificate_allows_only_literal_harness_map_rhs_changes(
    tool: ModuleType,
) -> None:
    source = _perf001_source_object("src/radiosim/benchmarks/harness.py")
    evidence = source.replace(
        b"PERF001_REFERENCE_SHA256: dict[str, str] = {}",
        b"PERF001_REFERENCE_SHA256: dict[str, str] = {'artifact': '"
        + b"1" * 64
        + b"'}",
        1,
    ).replace(
        b"PERF001_REFERENCE_SOURCE_SHA: dict[str, str] = {}",
        b"PERF001_REFERENCE_SOURCE_SHA: dict[str, str] = {'artifact': '"
        + b"2" * 40
        + b"'}",
        1,
    )
    tool._require_reference_manifest_only_change(source, evidence)
    with pytest.raises(tool.CpuEvidenceError, match="only the two literal"):
        tool._require_reference_manifest_only_change(
            source,
            evidence + b"\n# unauthorized evidence-time harness change\n",
        )
    duplicate = evidence.replace(
        b"{'artifact': '" + b"1" * 64 + b"'}",
        b"{'artifact': '" + b"1" * 64 + b"', 'artifact': '" + b"1" * 64 + b"'}",
        1,
    )
    with pytest.raises(tool.CpuEvidenceError, match="duplicate keys"):
        tool._require_reference_manifest_only_change(source, duplicate)


def test_acceptance_certificate_pins_exact_source_status_documents(
    tool: ModuleType,
) -> None:
    memo = _perf001_source_object("docs/development/perf001_runtime_mitigations.md")
    plan = _perf001_source_object("PostTier8RemediationPlan.md")
    assert hashlib.sha256(memo).hexdigest() == tool.ACCEPTED_SOURCE_MEMO_SHA256
    assert hashlib.sha256(plan).hexdigest() == tool.ACCEPTED_SOURCE_PLAN_SHA256
    assert memo.count(tool.EVIDENCE_REPRODUCTION_SENTINEL.encode("utf-8")) == 1
    assert memo.count(tool.ACCEPTANCE_MEMO_STATUS_SENTINEL.encode("utf-8")) == 1
    assert plan.count(tool.ACCEPTANCE_PLAN_STATUS_SENTINEL.encode("utf-8")) == 1
    tool._require_fix_perf001_roadmap_row(
        (REPOSITORY_ROOT / "Fix.md").read_bytes(),
        label="accepted source",
    )


def _accepted_document_transform_fixture(
    tool: ModuleType,
) -> dict[str, bytes | str]:
    source_sha = "1" * 40
    artifact_sha256 = "2" * 64
    artifact_path = (
        "output/benchmarks/reference/perf001/20260811T000000Z-darwin-arm64.json"
    )
    source_memo = (
        b"memo-before\n"
        + tool.EVIDENCE_REPRODUCTION_SENTINEL.encode("utf-8")
        + b"\n"
        + tool.ACCEPTANCE_MEMO_STATUS_SENTINEL.encode("utf-8")
        + b"\nmemo-after\n"
    )
    source_plan = (
        b"plan-before\n"
        + tool.ACCEPTANCE_PLAN_STATUS_SENTINEL.encode("utf-8")
        + b"\nplan-after\n"
    )
    evidence_memo = tool._expected_evidence_memo(
        source_memo,
        source_sha=source_sha,
        artifact_sha256=artifact_sha256,
        artifact_path=artifact_path,
    )
    acceptance_memo = tool._expected_acceptance_memo(evidence_memo)
    acceptance_plan = tool._expected_acceptance_plan(source_plan)
    return {
        "source_sha": source_sha,
        "artifact_sha256": artifact_sha256,
        "artifact_path": artifact_path,
        "source_memo": source_memo,
        "source_plan": source_plan,
        "evidence_memo": evidence_memo,
        "acceptance_memo": acceptance_memo,
        "acceptance_plan": acceptance_plan,
    }


@pytest.mark.parametrize(
    "mutation",
    (
        "evidence_extra_prose",
        "acceptance_extra_prose",
        "acceptance_plan_altered",
        "descendant_memo_altered",
        "descendant_plan_altered",
    ),
)
def test_acceptance_certificate_rejects_any_noncanonical_document_byte(
    tool: ModuleType,
    mutation: str,
) -> None:
    values = _accepted_document_transform_fixture(tool)
    evidence_memo = cast(bytes, values["evidence_memo"])
    acceptance_memo = cast(bytes, values["acceptance_memo"])
    acceptance_plan = cast(bytes, values["acceptance_plan"])
    descendant_memo = acceptance_memo
    descendant_plan = acceptance_plan
    if mutation == "evidence_extra_prose":
        evidence_memo += b"PERF-001 is DONE; supports_gpu=True\n"
    elif mutation == "acceptance_extra_prose":
        acceptance_memo += b"accelerator accepted\n"
    elif mutation == "acceptance_plan_altered":
        acceptance_plan += b"| `PERF-001` | DONE |\n"
    elif mutation == "descendant_memo_altered":
        descendant_memo += b"supports_gpu=True\n"
    else:
        descendant_plan += b"accelerator accepted\n"

    with pytest.raises(tool.CpuEvidenceError, match="exact byte transformation"):
        tool._authenticate_acceptance_document_transforms(
            source_memo=cast(bytes, values["source_memo"]),
            source_plan=cast(bytes, values["source_plan"]),
            evidence_memo=evidence_memo,
            evidence_plan=cast(bytes, values["source_plan"]),
            acceptance_memo=acceptance_memo,
            acceptance_plan=acceptance_plan,
            descendant_memo=descendant_memo,
            descendant_plan=descendant_plan,
            source_sha=cast(str, values["source_sha"]),
            artifact_sha256=cast(str, values["artifact_sha256"]),
            artifact_path=cast(str, values["artifact_path"]),
        )


@pytest.mark.parametrize(
    ("document", "marker"),
    (
        ("evidence", "missing"),
        ("evidence", "duplicate"),
        ("memo", "missing"),
        ("memo", "duplicate"),
        ("plan", "missing"),
        ("plan", "duplicate"),
    ),
)
def test_acceptance_certificate_requires_each_unique_source_marker(
    tool: ModuleType,
    document: str,
    marker: str,
) -> None:
    values = _accepted_document_transform_fixture(tool)
    if document == "evidence":
        raw = cast(bytes, values["source_memo"])
        sentinel = tool.EVIDENCE_REPRODUCTION_SENTINEL.encode("utf-8")

        def call(value: bytes) -> bytes:
            return tool._expected_evidence_memo(
                value,
                source_sha=cast(str, values["source_sha"]),
                artifact_sha256=cast(str, values["artifact_sha256"]),
                artifact_path=cast(str, values["artifact_path"]),
            )

    elif document == "memo":
        raw = cast(bytes, values["evidence_memo"])
        sentinel = tool.ACCEPTANCE_MEMO_STATUS_SENTINEL.encode("utf-8")
        call = tool._expected_acceptance_memo
    else:
        raw = cast(bytes, values["source_plan"])
        sentinel = tool.ACCEPTANCE_PLAN_STATUS_SENTINEL.encode("utf-8")
        call = tool._expected_acceptance_plan
    malformed = raw.replace(sentinel, b"", 1)
    if marker == "duplicate":
        malformed = raw + sentinel + b"\n"
    with pytest.raises(tool.CpuEvidenceError, match="exactly once"):
        call(malformed)


@pytest.mark.parametrize(
    "mutation", ("done", "duplicate", "missing", "missing_line_feed")
)
def test_acceptance_certificate_requires_the_exact_unique_fix_roadmap_row(
    tool: ModuleType,
    mutation: str,
) -> None:
    row = tool.FIX_PERF001_ROADMAP_ROW
    raw = ("header\n" + row + "\nfooter\n").encode("utf-8")
    if mutation == "done":
        raw = raw.replace(b"| PERF-001 | ROADMAP |", b"| PERF-001 | DONE |", 1)
    elif mutation == "duplicate":
        raw += (row + "\n").encode("utf-8")
    elif mutation == "missing":
        raw = b"header\nfooter\n"
    else:
        raw = raw.replace((row + "\n").encode("utf-8"), row.encode("utf-8"), 1)
    with pytest.raises(tool.CpuEvidenceError, match="exact unique ROADMAP row"):
        tool._require_fix_perf001_roadmap_row(raw, label="test commit")


def test_verify_accepted_certificate_binds_exact_s_e_a_and_descendant(
    tool: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.unit.test_perf001_runtime_acceptance import _cpu_document, _provenance

    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "perf001@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "PERF-001 Test"],
        cwd=repository,
        check=True,
    )
    source_paths = (
        "tools/wp7_perf001_cpu_evidence.py",
        "src/radiosim/benchmarks/record.py",
        "src/radiosim/benchmarks/harness.py",
        "pixi.toml",
        "pixi.lock",
        "docs/development/perf001_runtime_mitigations.md",
        "PostTier8RemediationPlan.md",
        "Fix.md",
    )
    for relative in source_paths:
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(_perf001_source_object(relative))
    monkeypatch.setattr(
        tool,
        "ACCEPTED_SOURCE_MEMO_SHA256",
        hashlib.sha256(
            (
                repository / "docs/development/perf001_runtime_mitigations.md"
            ).read_bytes()
        ).hexdigest(),
    )
    monkeypatch.setattr(
        tool,
        "ACCEPTED_SOURCE_PLAN_SHA256",
        hashlib.sha256(
            (repository / "PostTier8RemediationPlan.md").read_bytes()
        ).hexdigest(),
    )
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "source S"], cwd=repository, check=True)
    source_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()

    relative = "output/benchmarks/reference/perf001/20260811T000000Z-darwin-arm64.json"
    document = _cpu_document(
        _provenance(
            git_sha=source_sha,
            pixi_lock_sha256=tool.PIXI_LOCK_SHA256,
        )
    )
    raw = (
        json.dumps(document.to_json_safe(), allow_nan=False, indent=2) + "\n"
    ).encode("utf-8")
    artifact = repository / relative
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    harness = repository / "src/radiosim/benchmarks/harness.py"
    harness_source = harness.read_text(encoding="utf-8")
    harness_source = harness_source.replace(
        "PERF001_REFERENCE_SHA256: dict[str, str] = {}",
        f"PERF001_REFERENCE_SHA256: dict[str, str] = {{{relative!r}: {digest!r}}}",
        1,
    ).replace(
        "PERF001_REFERENCE_SOURCE_SHA: dict[str, str] = {}",
        f"PERF001_REFERENCE_SOURCE_SHA: dict[str, str] = "
        f"{{{relative!r}: {source_sha!r}}}",
        1,
    )
    harness.write_text(harness_source, encoding="utf-8")
    memo = repository / "docs/development/perf001_runtime_mitigations.md"
    memo.write_bytes(
        tool._expected_evidence_memo(
            memo.read_bytes(),
            source_sha=source_sha,
            artifact_sha256=digest,
            artifact_path=relative,
        )
    )
    subprocess.run(
        [
            "git",
            "add",
            relative,
            "src/radiosim/benchmarks/harness.py",
            "docs/development/perf001_runtime_mitigations.md",
        ],
        cwd=repository,
        check=True,
    )
    subprocess.run(["git", "commit", "-qm", "evidence E"], cwd=repository, check=True)
    evidence_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()

    memo.write_bytes(tool._expected_acceptance_memo(memo.read_bytes()))
    plan = repository / "PostTier8RemediationPlan.md"
    plan.write_bytes(tool._expected_acceptance_plan(plan.read_bytes()))
    subprocess.run(
        [
            "git",
            "add",
            "docs/development/perf001_runtime_mitigations.md",
            "PostTier8RemediationPlan.md",
        ],
        cwd=repository,
        check=True,
    )
    subprocess.run(["git", "commit", "-qm", "acceptance A"], cwd=repository, check=True)
    acceptance_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    descendant_marker = repository / "sci005-g1.txt"
    descendant_marker.write_text("dependent gate\n", encoding="utf-8")
    subprocess.run(["git", "add", "sci005-g1.txt"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "dependent descendant"],
        cwd=repository,
        check=True,
    )
    descendant_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()

    certificate = tool.verify_accepted_cpu_certificate(
        acceptance_commit=acceptance_sha,
        descendant=descendant_sha,
        repository_root=repository,
    )

    assert certificate == {
        "schema_version": tool.ACCEPTANCE_CERTIFICATE_SCHEMA,
        "acceptance_commit": acceptance_sha,
        "evidence_commit": evidence_sha,
        "generating_source_sha": source_sha,
        "descendant_commit": descendant_sha,
        "artifact_path": relative,
        "artifact_sha256": digest,
        "cpu_evidence_tool_sha256": hashlib.sha256(
            CPU_EVIDENCE_TOOL.read_bytes()
        ).hexdigest(),
        "production_record_validator_sha256": hashlib.sha256(
            (REPOSITORY_ROOT / "src/radiosim/benchmarks/record.py").read_bytes()
        ).hexdigest(),
        "production_harness_sha256": hashlib.sha256(harness.read_bytes()).hexdigest(),
        "pixi_manifest_sha256": tool.PIXI_MANIFEST_SHA256,
        "pixi_lock_sha256": tool.PIXI_LOCK_SHA256,
        "evidence_diff_paths": [
            "docs/development/perf001_runtime_mitigations.md",
            relative,
            "src/radiosim/benchmarks/harness.py",
        ],
        "acceptance_diff_paths": [
            "PostTier8RemediationPlan.md",
            "docs/development/perf001_runtime_mitigations.md",
        ],
        "verdict": tool.ACCEPTANCE_VERDICT,
        "passed": True,
    }
