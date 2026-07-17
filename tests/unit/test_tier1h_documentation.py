"""Tier 1H contracts for shipped configs, docs, and runnable examples."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

import radiosim
from radiosim.backends import base as backend_base
from radiosim.backends import get_backend, jax_backend, numba_backend
from radiosim.cli.main import cli
from radiosim.core.precision import PrecisionConfig
from radiosim.io.config import (
    RadioSimConfig,
    SkyModelConfig,
    create_default_config,
    dump_config,
    load_config,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SHIPPED_CONFIGS = (
    REPOSITORY_ROOT / "configs" / "config.yaml",
    REPOSITORY_ROOT / "configs" / "realistic_foreground_example.yaml",
    REPOSITORY_ROOT / "antenna_layout_examples" / "example_telescope_config.yaml",
)
CURRENT_API_SURFACES = (
    REPOSITORY_ROOT / "README.md",
    REPOSITORY_ROOT / "docs" / "index.rst",
    REPOSITORY_ROOT / "docs" / "quickstart.rst",
    REPOSITORY_ROOT / "docs" / "user_guide" / "configuration.rst",
    REPOSITORY_ROOT / "docs" / "user_guide" / "configuration_support.rst",
    REPOSITORY_ROOT / "docs" / "user_guide" / "backends.rst",
    REPOSITORY_ROOT / "docs" / "user_guide" / "beam_models.rst",
    REPOSITORY_ROOT / "docs" / "user_guide" / "sky_models.rst",
    REPOSITORY_ROOT / "docs" / "user_guide" / "jones_matrices.rst",
    REPOSITORY_ROOT / "docs" / "installation.rst",
    REPOSITORY_ROOT / "docs" / "api" / "simulator.rst",
    REPOSITORY_ROOT / "docs" / "api" / "io.rst",
    REPOSITORY_ROOT / "docs" / "api" / "jones.rst",
    REPOSITORY_ROOT / "docs" / "api" / "backends.rst",
    REPOSITORY_ROOT / "docs" / "api" / "core.rst",
    REPOSITORY_ROOT / "examples" / "README.md",
    REPOSITORY_ROOT / "examples" / "scripts" / "simple_simulation.py",
    REPOSITORY_ROOT / "antenna_layout_examples" / "README_antenna_formats.md",
)


@pytest.mark.parametrize("config_path", SHIPPED_CONFIGS, ids=lambda path: path.name)
def test_shipped_config_uses_strict_schema_and_resolves(config_path):
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    input_config = RadioSimConfig.model_validate(document)
    bundle = load_config(config_path)

    assert input_config.obs_frequency.mode in {"grid", "explicit"}
    assert bundle.runtime.frequency.channel_frequencies_hz
    assert bundle.provenance.source.config_path == config_path.resolve()


def test_default_config_generator_emits_tier1_input_shape(tmp_path):
    output = tmp_path / "generated.yaml"

    create_default_config(output)
    document = yaml.safe_load(output.read_text(encoding="utf-8"))
    config = RadioSimConfig.model_validate(document)

    assert document["obs_frequency"]["mode"] == "explicit"
    assert document["execution"]["backend"] == "numpy"
    assert document["workflow"]["save_results"] is False
    assert config.obs_frequency.channel_frequencies_hz == (100_000_000.0,)


def test_shipped_input_model_serializes_and_reloads_at_input_boundary(tmp_path):
    source = REPOSITORY_ROOT / "configs" / "config.yaml"
    input_config = RadioSimConfig.model_validate(
        yaml.safe_load(source.read_text(encoding="utf-8"))
    )
    output = tmp_path / "round-trip.yaml"

    dump_config(input_config, output)
    reloaded_input = RadioSimConfig.model_validate(
        yaml.safe_load(output.read_text(encoding="utf-8"))
    )
    resolved = load_config(output, check_input_paths=False)

    assert reloaded_input == input_config
    assert resolved.runtime.frequency.channel_frequencies_hz


def test_simple_example_help_and_offline_smoke(tmp_path):
    script = REPOSITORY_ROOT / "examples" / "scripts" / "simple_simulation.py"

    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--no-plot" in help_result.stdout
    assert "--save" in help_result.stdout

    smoke_result = subprocess.run(
        [sys.executable, str(script), "--no-plot"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert smoke_result.returncode == 0, smoke_result.stderr
    assert "Simulation complete" in smoke_result.stdout
    assert not (tmp_path / "simulation_output").exists()


def test_notebook_source_uses_current_public_api_and_has_no_stale_outputs():
    notebook_path = REPOSITORY_ROOT / "examples" / "notebooks" / "01_basic_usage.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", ()))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "Simulator.from_parameters(" in source
    assert "Simulator(config=" not in source
    assert '"frequencies_hz"' not in source
    assert "Simulator.from_config(" not in source
    assert "simulator._" not in source
    assert ".benchmark(" not in source
    for cell in notebook["cells"]:
        assert cell.get("id")
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []


@pytest.mark.parametrize("path", CURRENT_API_SURFACES, ids=lambda path: path.name)
def test_current_docs_do_not_present_removed_simulator_patterns(path):
    text = path.read_text(encoding="utf-8")

    assert 'Simulator.from_config("' not in text
    assert "Simulator(config=" not in text
    assert "Simulator(backend=" not in text
    assert "sim = Simulator()" not in text
    assert '"frequencies_hz":' not in text


def test_historical_hera_analysis_is_clearly_labelled():
    text = (REPOSITORY_ROOT / "docs" / "HERA_VSIM_ANALYSIS.md").read_text(
        encoding="utf-8"
    )

    assert any("Historical analysis" in line for line in text.splitlines()[:12])


def test_public_docstrings_do_not_overstate_gpu_or_performance_support():
    assert "full polarization support" not in (radiosim.__doc__ or "")
    assert "GPU acceleration" not in (radiosim.__doc__ or "")
    assert "10-20x" not in (PrecisionConfig.ultra.__doc__ or "")


def test_root_cli_help_does_not_claim_end_to_end_gpu_support():
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "GPU support" not in result.output


def test_active_backend_autodoc_avoids_unverified_acceleration_claims():
    rendered_docstrings = "\n".join(
        text
        for text in (
            get_backend.__doc__,
            backend_base.__doc__,
            backend_base.ArrayBackend.__doc__,
            jax_backend.__doc__,
            jax_backend.JAXBackend.__doc__,
            jax_backend.JAXBackend.matmul.__doc__,
            numba_backend.__doc__,
            numba_backend.NumbaBackend.__doc__,
        )
        if text
    )

    for unsupported_claim in (
        "2-10x speedup",
        "10-100x speedup",
        "universal hardware acceleration",
        "Universal hardware support",
        "NumbaBackend: CPU/GPU",
        "GPU support via CUDA",
        "Matrix multiplication (GPU-accelerated)",
    ):
        assert unsupported_claim not in rendered_docstrings


def test_backend_guide_uses_the_real_array_conversion_method():
    text = (REPOSITORY_ROOT / "docs" / "user_guide" / "backends.rst").read_text(
        encoding="utf-8"
    )

    assert "backend.array(" not in text
    assert "backend.asarray(" in text


def test_custom_sky_alias_documentation_uses_strict_options_envelope():
    config = SkyModelConfig.model_validate(
        {
            "sources": [
                {"kind": "gsm2016", "options": {"nside": 128}},
                {
                    "kind": "gsm2016",
                    "options": {"model": "haslam", "nside": 64},
                },
            ]
        }
    )
    text = (REPOSITORY_ROOT / "docs" / "user_guide" / "sky_models.rst").read_text(
        encoding="utf-8"
    )

    assert config.sources[0].options["nside"] == 128
    assert config.sources[1].options["model"] == "haslam"
    assert "kind: gsm2016\n         options:" in text
    assert "kind: gsm2016\n         nside:" not in text
