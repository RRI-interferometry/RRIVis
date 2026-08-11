"""Tier 1H contracts for shipped configs, docs, and runnable examples."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

import radiosim
from radiosim.api import simulator as simulator_api
from radiosim.backends import base as backend_base
from radiosim.backends import dask_backend, get_backend, jax_backend
from radiosim.cli.main import cli
from radiosim.core import visibility, visibility_healpix
from radiosim.core.precision import PrecisionConfig
from radiosim.io import measurement_set, summary_json
from radiosim.io.config import (
    RadioSimConfig,
    SkyModelConfig,
    create_default_config,
    dump_config,
    load_config,
)
from radiosim.simulator import base as simulator_base
from radiosim.simulator import rime as rime_simulator

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SHIPPED_CONFIGS = (
    REPOSITORY_ROOT / "configs" / "config.yaml",
    REPOSITORY_ROOT / "configs" / "realistic_foreground_example.yaml",
    REPOSITORY_ROOT / "configs" / "receptor_circular_example.yaml",
    REPOSITORY_ROOT / "antenna_layout_examples" / "example_telescope_config.yaml",
)
CURRENT_API_SURFACES = (
    REPOSITORY_ROOT / "README.md",
    REPOSITORY_ROOT / "docs" / "index.rst",
    REPOSITORY_ROOT / "docs" / "quickstart.rst",
    REPOSITORY_ROOT / "docs" / "user_guide" / "configuration.rst",
    REPOSITORY_ROOT / "docs" / "user_guide" / "configuration_support.rst",
    REPOSITORY_ROOT / "docs" / "user_guide" / "instrument_resolution.rst",
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
AUTHORIZED_BEAM_TRUTH_SURFACES = (
    REPOSITORY_ROOT / "README.md",
    REPOSITORY_ROOT / "docs" / "user_guide" / "beam_models.rst",
    REPOSITORY_ROOT / "docs" / "user_guide" / "configuration.rst",
    REPOSITORY_ROOT / "docs" / "user_guide" / "configuration_support.rst",
    REPOSITORY_ROOT / "docs" / "api" / "core.rst",
)
FINAL_BEAM_TRUTH_SURFACES = (
    REPOSITORY_ROOT / "README.md",
    REPOSITORY_ROOT / "docs" / "user_guide" / "beam_models.rst",
    REPOSITORY_ROOT / "docs" / "user_guide" / "configuration.rst",
    REPOSITORY_ROOT / "docs" / "user_guide" / "configuration_support.rst",
    REPOSITORY_ROOT / "docs" / "user_guide" / "jones_matrices.rst",
    REPOSITORY_ROOT / "docs" / "api" / "core.rst",
    REPOSITORY_ROOT / "docs" / "api" / "jones.rst",
    REPOSITORY_ROOT / "docs" / "api" / "simulator.rst",
)


@pytest.mark.parametrize("config_path", SHIPPED_CONFIGS, ids=lambda path: path.name)
def test_shipped_config_uses_strict_schema_and_resolves(config_path):
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    input_config = RadioSimConfig.model_validate(document)
    bundle = load_config(config_path)

    assert input_config.obs_frequency.mode in {"grid", "explicit"}
    assert type(document["instrument"]) is dict
    assert document["instrument"]["source"]["kind"] in {
        "layout_file",
        "known_telescope",
    }
    assert "telescope" not in document
    assert "antenna_layout" not in document
    assert "location" not in document
    assert "feeds" not in document
    assert set(document["baseline_selection"]) <= {
        "correlations",
        "length_filter",
        "azimuth_ranges_deg",
    }
    assert document["beams"] == {
        "mode": "analytic",
        "model": {
            "kind": "circular_aperture",
            "taper": {"kind": "gaussian", "edge_taper_db": 10.0},
        },
    }
    assert bundle.runtime.beams.mode == "analytic"
    assert bundle.runtime.frequency.channel_frequencies_hz
    assert bundle.runtime.frequency.channel_widths_hz
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
    assert config.obs_frequency.channel_widths_hz == (1_000_000.0,)


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
    assert resolved.runtime.frequency.channel_widths_hz


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
    assert "--progress" in help_result.stdout
    assert "--save" not in help_result.stdout
    assert "--plot" not in help_result.stdout

    smoke_result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert smoke_result.returncode == 0, smoke_result.stderr
    assert "Simulation complete" in smoke_result.stdout
    assert "Visibility shape (T, B, F, C): (1, 15, 2, 4)" in smoke_result.stdout
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
    assert "channel_widths_hz=" in source
    assert "instrument=instrument" in source
    assert "baseline_selection=baseline_selection" in source
    assert "antenna_layout=" not in source
    assert "antenna_diameter_m=" not in source
    assert "Simulator(config=" not in source
    assert '"frequencies_hz"' not in source
    assert "Simulator.from_config(" not in source
    assert "simulator._" not in source
    assert ".benchmark(" not in source
    assert "result = simulator.run(" in source
    assert "result.visibilities" in source
    assert "result.stokes_i()" in source
    assert "simulator.result" in source
    assert 'results["visibilities"]' not in source
    assert "simulator.save(" not in source
    assert "simulator.plot(" not in source
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
    assert "use_pyuvdata_telescope" not in text
    assert "use_pyuvdata_antennas" not in text
    assert "antenna_file_format: pyuvdata" not in text
    assert "antenna_file_format: casa" not in text


@pytest.mark.parametrize(
    "path",
    (
        REPOSITORY_ROOT / "README.md",
        REPOSITORY_ROOT / "docs" / "user_guide" / "configuration_support.rst",
    ),
    ids=lambda path: path.name,
)
def test_tier4f_output_docs_are_exact(path):
    text = path.read_text(encoding="utf-8")

    assert "ResultFormat" in text
    assert "summary JSON" in text
    assert "HDF5" in text
    assert "exact final" in text
    assert "HDF5 and JSON preserve" not in text
    assert "result saving and plotting are deliberately unavailable" not in text


TIER4G_TRUTH_SURFACES = (
    REPOSITORY_ROOT / "README.md",
    REPOSITORY_ROOT / "docs" / "index.rst",
    REPOSITORY_ROOT / "docs" / "quickstart.rst",
    REPOSITORY_ROOT / "docs" / "api" / "io.rst",
    REPOSITORY_ROOT / "docs" / "api" / "simulator.rst",
    REPOSITORY_ROOT / "docs" / "user_guide" / "configuration.rst",
    REPOSITORY_ROOT / "docs" / "user_guide" / "configuration_support.rst",
)


@pytest.mark.parametrize("path", TIER4G_TRUTH_SURFACES, ids=lambda path: path.name)
def test_tier4g_active_docs_drop_every_removed_plot_surface(path):
    text = path.read_text(encoding="utf-8")

    assert "angle_unit" not in text
    assert "sky_model_frequency_hz" not in text
    assert "Tier 4G" not in text
    assert "fail-closed" not in text
    assert "result plotting remains unavailable" not in text
    assert "remains rejected" not in text


def test_tier4g_configuration_docs_own_the_visibility_phase_unit():
    for path in (
        REPOSITORY_ROOT / "README.md",
        REPOSITORY_ROOT / "docs" / "user_guide" / "configuration.rst",
        REPOSITORY_ROOT / "docs" / "user_guide" / "configuration_support.rst",
    ):
        text = path.read_text(encoding="utf-8")
        assert "visibility_phase_unit" in text
        assert "radians" in text
        assert "degrees" in text


def test_tier4g_simulator_docs_describe_the_canonical_renderer():
    text = (REPOSITORY_ROOT / "docs" / "api" / "simulator.rst").read_text(
        encoding="utf-8"
    )

    assert "Simulator.plot" in text
    assert "visibility_phase_unit" in text
    assert "Stokes I" in text
    assert "browser" in text
    for required in ("plot_type", "output_dir", "overwrite"):
        assert required in text


def test_tier4g_migration_guide_maps_the_removed_visualization_fields():
    migration = (REPOSITORY_ROOT / "docs" / "migration_guide.md").read_text(
        encoding="utf-8"
    )

    assert (
        "workflow.angle_unit: removed before v1.0; "
        "use workflow.visibility_phase_unit" in migration
    )
    assert (
        "workflow.sky_model_frequency_hz: removed before v1.0; "
        "no Tier 4 sky renderer consumes it" in migration
    )
    assert "result plotting remains" not in migration


@pytest.mark.parametrize("path", SHIPPED_CONFIGS, ids=lambda path: path.name)
def test_tier4g_shipped_configs_declare_only_active_workflow_fields(path):
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    workflow = document["workflow"]

    assert "angle_unit" not in workflow
    assert "sky_model_frequency_hz" not in workflow
    assert workflow["visibility_phase_unit"] in {"radians", "degrees"}


def test_tier4d_hdf5_documentation_is_complete_and_bounded():
    text = (REPOSITORY_ROOT / "docs" / "api" / "io.rst").read_text(encoding="utf-8")

    for required in (
        "radiosim.visibility",
        "4.0.0",
        ".h5",
        "write_result_hdf5",
        "load_result_hdf5",
        "HDF5ReadLimits",
        "SimulationResult",
        "LoadedSimulationResult",
        "complex64",
        "complex128",
        "XX, XY, YX, YY",
        "flags",
        "weights",
        "scientific fingerprint",
        "provenance fingerprint",
        "atomic",
        "read-back",
        "fixed-width",
        "VLEN",
        "before any value-read API",
        "no VLEN compatibility reader",
        "legacy unversioned",
        "Simulator.save",
        "ResultFormat.HDF5",
        "result-summary",
        "summary JSON",
        "16 MiB",
        "workflow-manifest.v1",
        "collision_policy",
        "browser",
    ):
        assert required in text
    assert "save_visibilities_hdf5" not in text
    assert "load_visibilities_hdf5" not in text
    summary_schema = (
        f"schema ``{summary_json.SUMMARY_SCHEMA_NAME}`` version "
        f"``{summary_json.SUMMARY_SCHEMA_VERSION}``"
    )
    assert summary_schema in text
    for required in (
        "StandardVisibilityData",
        "first-time zenith",
        "ICRS",
        "complex128",
        "complex64",
        "UVFITS",
        "Measurement Set",
        "atomic",
        "optional dependencies",
        "write_measurement_set",
        "read_measurement_set",
        "write_uvfits",
        "read_uvfits",
        "Simulator.save",
        "Simulator.save",
    ):
        assert required in text
    for removed in (
        "write_ms",
        "read_ms",
        "read_ms_dask",
        "ms_info",
        "PYUVDATA_AVAILABLE",
        "CASACORE_AVAILABLE",
        "DASKMS_AVAILABLE",
        "MS_AVAILABLE",
    ):
        assert removed not in text


@pytest.mark.parametrize(
    "path", AUTHORIZED_BEAM_TRUTH_SURFACES, ids=lambda path: path.name
)
def test_tier3b_beam_truth_surfaces_do_not_present_flat_schema(path):
    text = path.read_text(encoding="utf-8")

    for removed_name in (
        "beam_mode",
        "beam_file",
        "antenna_beam_map",
        "aperture_shape",
        "edge_taper_dB",
        "feed_model",
        "feed_computation",
        "feed_params",
        "reflector_type",
        "aperture_params",
    ):
        assert (
            re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(removed_name)}(?![A-Za-z0-9_])",
                text,
            )
            is None
        )


def test_tier3_beam_docs_distinguish_resolution_from_runtime_activation():
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    guide = (REPOSITORY_ROOT / "docs" / "user_guide" / "beam_models.rst").read_text(
        encoding="utf-8"
    )
    support = (
        REPOSITORY_ROOT / "docs" / "user_guide" / "configuration_support.rst"
    ).read_text(encoding="utf-8")
    api = (REPOSITORY_ROOT / "docs" / "api" / "core.rst").read_text(encoding="utf-8")
    migration = (REPOSITORY_ROOT / "docs" / "migration_guide.md").read_text(
        encoding="utf-8"
    )

    for mode in ("analytic", "shared_fits", "per_antenna_fits", "mixed"):
        assert mode in guide
        assert mode in readme
    for variant in (
        "circular_aperture",
        "rectangular_aperture",
        "elliptical_aperture",
        "analytical_illumination",
        "numerical_illumination",
    ):
        assert variant in guide
    assert "beam_runtime_fits_pending" not in guide
    assert "activates all four modes" in guide
    assert "Visibility-result provenance" in guide
    assert "Schema" in support
    assert "Path resolution" in support
    assert "Simulator runtime" in support
    assert "only direct-circular" not in api
    assert "BeamSystem" in api
    assert "LoadedBeamState" in api
    assert "resolve_beam_assignments" in api
    assert "beam_mode" in migration
    assert "rejected rather than translated" in " ".join(migration.split())


@pytest.mark.parametrize("path", FINAL_BEAM_TRUTH_SURFACES, ids=lambda path: path.name)
def test_tier3h2_active_docs_do_not_publish_removed_beam_surfaces(path):
    text = path.read_text(encoding="utf-8")

    for removed_name in (
        "BeamManager",
        "BeamFITSHandler",
        "BeamJones",
        "AnalyticBeamJones",
        "FITSBeamJones",
        "compute_aperture_beam",
        "APERTURE_SHAPES",
        "TAPER_FUNCTIONS",
        "FEED_MODELS",
        "REFLECTOR_TYPES",
        "plot_beam_pattern",
        "plot_beam_comparison",
        "plot_beam_2d",
        "plot_feed_illumination",
    ):
        assert removed_name not in text, (path, removed_name)


def test_tier3h2_final_runtime_and_science_truth_is_documented():
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    beam_guide = (
        REPOSITORY_ROOT / "docs" / "user_guide" / "beam_models.rst"
    ).read_text(encoding="utf-8")
    support = (
        REPOSITORY_ROOT / "docs" / "user_guide" / "configuration_support.rst"
    ).read_text(encoding="utf-8")
    jones = (REPOSITORY_ROOT / "docs" / "user_guide" / "jones_matrices.rst").read_text(
        encoding="utf-8"
    )
    simulator = (REPOSITORY_ROOT / "docs" / "api" / "simulator.rst").read_text(
        encoding="utf-8"
    )

    for mode in ("analytic", "shared_fits", "per_antenna_fits", "mixed"):
        assert mode in readme
        assert mode in beam_guide
        assert mode in support
    for variant in (
        "circular_aperture",
        "rectangular_aperture",
        "elliptical_aperture",
        "analytical_illumination",
        "numerical_illumination",
    ):
        assert variant in support
    assert "one canonical per-antenna ``BeamSystem``" in beam_guide
    assert "does not read FITS content" in beam_guide
    assert "no analytic fallback" in readme
    assert "scalar E-Jones" in " ".join(jones.split())
    assert "Tier 7" in jones
    assert "beam_system" in simulator
    assert "beam_state" in simulator
    assert "automatic NSIDE mutation" not in beam_guide
    assert (
        "accepts uniform resolved diameter arrays and rejects heterogeneous arrays"
        not in " ".join(readme.split())
    )


def test_tier3h2_migration_guide_maps_every_removed_low_level_surface():
    migration = (REPOSITORY_ROOT / "docs" / "migration_guide.md").read_text(
        encoding="utf-8"
    )
    for removed_name in (
        "BeamManager",
        "BeamFITSHandler",
        "BeamJones",
        "AnalyticBeamJones",
        "FITSBeamJones",
        "compute_aperture_beam",
        "mutable registries",
        "plotting helpers",
    ):
        assert removed_name in migration
    assert "fail immediately" in migration
    assert "compatibility shim" in migration
    assert "BeamSystem" in migration


def test_tier3h2_hera_analysis_separates_history_from_current_support():
    text = (REPOSITORY_ROOT / "docs" / "HERA_VSIM_ANALYSIS.md").read_text(
        encoding="utf-8"
    )

    assert "Current RadioSim support boundary" in text
    assert "historical evidence, not shipped dependencies" in text
    assert "accepted scalar BeamFITS subset" in text
    assert "does not establish compatibility" in text
    assert "currently rejects FITS/per-antenna beams" not in text
    assert "All necessary files" not in text


def test_tier2g_truth_surfaces_and_example_inventory_are_current():
    guide = REPOSITORY_ROOT / "docs" / "user_guide" / "instrument_resolution.rst"
    guide_text = guide.read_text(encoding="utf-8")
    antenna_formats = (
        REPOSITORY_ROOT / "antenna_layout_examples" / "README_antenna_formats.md"
    ).read_text(encoding="utf-8")
    index = (REPOSITORY_ROOT / "docs" / "index.rst").read_text(encoding="utf-8")
    script = (
        REPOSITORY_ROOT / "examples" / "scripts" / "simple_simulation.py"
    ).read_text(encoding="utf-8")

    assert guide.is_file()
    for result_format in ("HDF5", "summary JSON", "Measurement Set", "UVFITS"):
        assert result_format in guide_text
    assert "remain later separately gated work" not in guide_text
    assert "Matching `diameter_overrides` take precedence" in antenna_formats
    assert "Source diameters are used first" not in antenna_formats
    assert "user_guide/instrument_resolution" in index
    assert "InstrumentConfig(" in script
    assert "BaselineSelectionConfig(" in script
    assert "antenna_layout=" not in script
    assert "antenna_diameter_m=" not in script
    assert (
        REPOSITORY_ROOT / "antenna_layout_examples" / "example_casa_loc.cfg"
    ).is_file()
    assert not (
        REPOSITORY_ROOT / "antenna_layout_examples" / "example_casa_format.cfg"
    ).exists()
    assert not (
        REPOSITORY_ROOT / "antenna_layout_examples" / "example_pyuvdata_format.txt"
    ).exists()


def test_precision_documentation_describes_the_component_tree() -> None:
    precision_source = (
        REPOSITORY_ROOT / "src" / "radiosim" / "core" / "precision.py"
    ).read_text(encoding="utf-8")
    backend_guide = (
        REPOSITORY_ROOT / "docs" / "user_guide" / "backends.rst"
    ).read_text(encoding="utf-8")
    standard = PrecisionConfig.standard()

    assert standard.default == "float64"
    assert standard.sky_model.healpix_maps == "float32"
    assert re.search(r"float32 HEALPix map\s+storage", backend_guide)
    for overbroad_claim in (
        "All float64",
        "float64 everywhere",
        "all components set to float64",
        "float128 everywhere",
    ):
        assert overbroad_claim not in precision_source
    assert "float64 throughout" not in backend_guide


def test_historical_hera_analysis_is_clearly_labelled():
    text = (REPOSITORY_ROOT / "docs" / "HERA_VSIM_ANALYSIS.md").read_text(
        encoding="utf-8"
    )

    assert any("Historical analysis" in line for line in text.splitlines()[:12])


def test_public_docstrings_do_not_overstate_gpu_or_performance_support():
    assert "full polarization support" not in (radiosim.__doc__ or "")
    assert "GPU acceleration" not in (radiosim.__doc__ or "")
    assert "10-20x" not in (PrecisionConfig.ultra.__doc__ or "")


def test_active_instrument_integration_docstrings_match_public_contract():
    rendered = "\n".join(
        (
            simulator_api.Simulator.__doc__ or "",
            visibility.__doc__ or "",
            visibility.calculate_visibility.__doc__ or "",
            visibility_healpix.calculate_visibility_healpix.__doc__ or "",
            measurement_set.__doc__ or "",
            simulator_base.VisibilitySimulator.__doc__ or "",
            rime_simulator.RIMESimulator.__doc__ or "",
        )
    )

    assert "all-baseline inventory" not in rendered
    assert "Supports both beam FITS files" not in rendered
    assert "CPU/GPU/TPU acceleration" not in rendered
    assert "antennas=results" not in rendered
    assert "baselines=results" not in rendered
    assert "location=results" not in rendered
    assert "_instrument_state" not in rendered
    assert "10-50× speedup" not in rendered
    assert "Calculate visibilities with GPU acceleration" not in rendered


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
            dask_backend.__doc__,
            dask_backend.DaskBackend.__doc__,
        )
        if text
    )

    for unsupported_claim in (
        "2-10x speedup",
        "10-100x speedup",
        "universal hardware acceleration",
        "Universal hardware support",
        "NumbaBackend: CPU/GPU",
        "DaskBackend: CPU/GPU",
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


TIER5G_TRUTH_SURFACES = (
    REPOSITORY_ROOT / "README.md",
    REPOSITORY_ROOT / "CLAUDE.md",
    REPOSITORY_ROOT / "docs" / "index.rst",
    REPOSITORY_ROOT / "docs" / "api" / "io.rst",
    REPOSITORY_ROOT / "docs" / "api" / "jones.rst",
    REPOSITORY_ROOT / "docs" / "api" / "simulator.rst",
    REPOSITORY_ROOT / "docs" / "migration_guide.md",
    REPOSITORY_ROOT / "docs" / "quickstart.rst",
    REPOSITORY_ROOT / "docs" / "user_guide" / "beam_models.rst",
    REPOSITORY_ROOT / "docs" / "user_guide" / "configuration.rst",
    REPOSITORY_ROOT / "docs" / "user_guide" / "configuration_support.rst",
    REPOSITORY_ROOT / "docs" / "user_guide" / "jones_matrices.rst",
)


@pytest.mark.parametrize("path", TIER5G_TRUTH_SURFACES, ids=lambda path: path.name)
def test_tier5g_docs_no_longer_deny_receptor_physics(path):
    text = " ".join(path.read_text(encoding="utf-8").split())

    for stale in (
        "receptor/feed physics is not implemented",
        "including receptor/feed physics",
        "Full receptor/basis/polarization physics",
        "Receptor physics and later simulator modes remain separate work",
        "full receptor/polarization physics are rejected",
        "Full receptor, basis, and polarization physics remains a later "
        "scientific boundary",
    ):
        assert stale not in text, (path, stale)


@pytest.mark.parametrize("path", TIER5G_TRUTH_SURFACES, ids=lambda path: path.name)
def test_tier5g_docs_never_present_the_linear_labels_as_the_only_labels(path):
    text = " ".join(path.read_text(encoding="utf-8").split())

    for stale in (
        "the correlation order is ``XX, XY, YX, YY``",
        "correlations ordered as `XX, XY, YX, YY`",
        "with the exact correlations ``XX, XY, YX, YY``",
        "Stokes I is derived explicitly as ``XX + YY`` through",
        "derive Stokes I as ``XX + YY``",
        "derive Stokes I explicitly as `XX + YY`",
    ):
        assert stale not in text, (path, stale)
    if "XX + YY" in text or "XX, XY, YX, YY" in text:
        # Wherever a page names the linear labels it must also name the
        # circular ones, or it reads as the only possible axis.
        assert "RR" in text, path


@pytest.mark.parametrize("path", TIER5G_TRUTH_SURFACES, ids=lambda path: path.name)
def test_tier5g_docs_publish_no_retired_illumination_identifier(path):
    text = path.read_text(encoding="utf-8")

    for retired in ("_feed_response", "_feed_angles"):
        assert retired not in text, (path, retired)
    if path.name == "migration_guide.md":
        # The migration guide is the one surface that must still name the old
        # identifiers, because that is what a reader is searching for.
        return
    for retired in (
        "theta_feed",
        "corrugated_horn_pattern",
        "open_waveguide_pattern",
        "dipole_ground_plane_pattern",
        "analytic.feed",
    ):
        assert retired not in text, (path, retired)


def test_tier5g_configuration_guide_documents_every_receptor_mode():
    text = (REPOSITORY_ROOT / "docs" / "user_guide" / "configuration.rst").read_text(
        encoding="utf-8"
    )
    collapsed = " ".join(text.split())

    for required in (
        "Receptor declarations",
        "feed_rotation_deg",
        "output_basis",
        "basis: circular",
        "basis: linear",
        "output_basis: circular",
        "output_basis: linear",
        "output_basis: auto",
        "{kind: number, number: 3}",
        "{kind: name, name: HERA-11}",
        "XX, XY, YX, YY",
        "RR, RL, LR, LL",
        "result.polarization_basis",
        "parallel hands",
        # FLIPPED BY: Tier 7F.  The guide said "A mount type other than ``fixed``
        # is rejected ... because the parallactic term is not implemented yet",
        # which stopped being true when ``P`` landed.  What replaces it is the
        # statement the reader now needs: the static rotation and the field
        # rotation compose, and the mount pairing belongs to ``jones.P``.
        "``feed_rotation_deg`` is the **static** part of the orientation",
        "Receptor resolution does not look at ``mount_type`` at all",
    ):
        assert required in collapsed, required
    assert "illumination" in collapsed


def test_tier5g_jones_guide_states_the_receptor_science_boundaries():
    text = (REPOSITORY_ROOT / "docs" / "user_guide" / "jones_matrices.rst").read_text(
        encoding="utf-8"
    )
    collapsed = " ".join(text.split())

    for required in (
        "ReceptorConfigJones",
        "BasisTransformJones",
        "Cross-basis reporting and interpretation",
        "Parallactic-angle boundary",
        "Chain order",
    ):
        assert required in text, required
    assert "is always an exact unitary coordinate change" in collapsed
    assert "only when those matrices commute with ``H``" in collapsed
    # FLIPPED BY: Tier 7E.  Until this slice the guide said the conversion was
    # exact "because ``D`` and ``G`` are planned rather than implemented", and
    # promised to re-examine the claim when Tier 7 implemented ``D``.  Tier 7E
    # implemented it, so the promise is discharged: the guide now names the
    # configurations that break output-native equivalence and says what to do
    # instead.  The assertion moves with it rather than being deleted, because
    # the property
    # being pinned -- that the guide states the boundary rather than implying
    # there is none -- is the same one.
    assert "any non-zero leakage" in collapsed
    assert "set ``receptors.output_basis`` to the antennas' own basis" in collapsed
    assert "When Tier 7 implements ``D``" not in collapsed
    # FLIPPED BY: Tier 7F.  Until this slice the guide said ``P`` was "planned
    # rather than implemented", that only ``mount_type: fixed`` was accepted,
    # and that ``feed_rotation_deg`` was therefore *the* rotation.  ``P`` is
    # real, so the promise is discharged and the assertion moves to the sentence
    # that replaces it: the static rotation is one half of a composition whose
    # other half is a term, and the two add.  The property being pinned -- that
    # the guide states the boundary rather than implying there is none -- is
    # unchanged.
    assert (
        "``feed_rotation_deg`` is the **static** part of the receptor "
        "orientation" in collapsed
    )
    assert "is planned rather than implemented" not in collapsed
    assert "the static feed rotation and the field rotation **add**" in collapsed
    # The canonical order gained ``Rc``, ``Kd`` and ``X`` when Tier 7E made them
    # real, and Tier 7F moved ``P`` sky-side of ``C`` (defect D12).
    assert (
        "J_p = H_p\\, G_p\\, B_p\\, Rc_p\\, Kd_p\\, X_p\\, D_p\\, C_p\\, E_p\\, "
        "P_p\\, T_p\\, Z_p" in text
    )
    assert "D_p\\, P_p\\, C_p" not in text
    assert "V_{RR} &= (I + V)/2" in text
    assert "U + iV" in text


def test_tier5g_io_reference_is_truthful_about_schema_and_basis():
    text = (REPOSITORY_ROOT / "docs" / "api" / "io.rst").read_text(encoding="utf-8")
    collapsed = " ".join(text.split())

    assert "schema version ``4.0.0``" in collapsed
    assert "schema version ``1.0.0`` is the complete" not in collapsed
    assert "explicitly maps ``XX, XY, YX, YY`` into each file" not in collapsed
    for required in (
        "receptors",
        "output_basis",
        "receptor_sha256",
        "feed_rotation_rad",
        "feed_angle_rad",
        "RR, RL, LR, LL",
        "circular_rl",
        "linear_xy",
        "Polarization mapping",
        "UnsupportedSchemaVersionError",
        "CORR_TYPE",
        "feed_array",
    ):
        assert required in text, required


def test_tier5g_migration_guide_maps_the_receptor_and_illumination_changes():
    text = (REPOSITORY_ROOT / "docs" / "migration_guide.md").read_text(encoding="utf-8")

    assert "no replacement; receptor/feed physics is not implemented" not in text
    assert (
        "| top-level `feeds` | the `receptors` section with `default.basis`, "
        "`default.feed_rotation_deg`, and `output_basis` |" in text
    )
    for required in (
        "## Receptors and polarization basis",
        "receptors.default.feed_type",
        "receptors.default.n_feeds",
        "receptors.default.feed_angle_deg",
        "AmbiguousOutputBasisError",
        # FLIPPED BY: Tier 7F.  ``UnsupportedFeedGeometryError`` was the type of
        # the blanket mount rejection this slice removed; the guide now names
        # its replacement, and both new messages verbatim.
        "UnsupportedMountTypeError",
        "whose feeds rotate with the sky; enable",
        "Parallactic angle and mount types",
        "UnsupportedSchemaVersionError",
        "ReceptorConfigJones(feed_type=...)",
        "BasisTransformJones(from_basis=..., to_basis=...)",
        "resolve_receptors()",
        "scientific_sha256",
        "Illumination primitives renamed",
        "corrugated_horn_illumination",
        "open_waveguide_illumination",
        "dipole_ground_plane_illumination",
        "radiosim.core.jones.beam.analytic.illumination",
    ):
        assert required in text, required
    assert "[[I + Q, U + iV], [U - iV, I - Q]] / 2" in text


def test_tier5g_claude_status_lists_the_implemented_jones_terms():
    text = (REPOSITORY_ROOT / "CLAUDE.md").read_text(encoding="utf-8")

    assert "Only **K** (`GeometricPhaseJones`, geometric phase) and **E**" not in text
    assert "Only K and E implement real physics" not in text
    # FLIPPED BY: Tier 7J.  Tier 5G pinned "Only K, E, C, and H implement real
    # physics" and the Tier 5 chain order, which were true when written and
    # stopped being true one term slice at a time through 7D-7H.  Section 34
    # gives `CLAUDE.md` to this slice precisely so the flip happens once, here.
    # The property being pinned is unchanged: the status section names exactly
    # which terms carry physics, and the RIME section names the exact chain
    # order the code composes.
    assert "Only K, E, C, and H implement real physics" not in text
    assert "every exported term implements real physics" in text
    assert "`B = (1/2) × [[I+Q, U+iV], [U-iV, I-Q]]`" in text
    assert "[[I+Q, U-iV], [U+iV, I-Q]]" not in text
    assert "`receptors`" in text
    assert "J_p = H_p G_p B_p D_p P_p C_p E_p T_p Z_p" not in text
    assert "J_p = H_p G_p B_p Rc_p Kd_p X_p D_p C_p E_p P_p T_p Z_p" in text


def test_tier5g_shipped_receptor_sample_declares_a_circular_array():
    path = REPOSITORY_ROOT / "configs" / "receptor_circular_example.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))

    config = RadioSimConfig.model_validate(document)
    bundle = load_config(path)

    assert document["receptors"]["default"]["basis"] == "circular"
    assert document["receptors"]["output_basis"] == "auto"
    assert config.receptors.default.basis == "circular"
    assert config.receptors.default.feed_rotation_deg == 0.0
    assert config.receptors.overrides == ()
    assert config.receptors.output_basis == "auto"
    assert bundle.runtime.receptors.output_basis == "auto"
    assert document["execution"]["offline"] is True


def test_tier5g_default_shipped_sample_spells_out_the_linear_default():
    document = yaml.safe_load(
        (REPOSITORY_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")
    )

    config = RadioSimConfig.model_validate(document)

    assert document["receptors"]["default"]["basis"] == "linear"
    assert (
        config.receptors
        == RadioSimConfig.model_validate(
            {key: value for key, value in document.items() if key != "receptors"}
        ).receptors
    )


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


# =========================================================================
# Tier 6I -- Section 26 documentation truth
#
# Every assertion below is the residual of a statement Tier 6 made false, or
# newly provable. They are written as "the false thing is gone AND the true
# thing is present", because only the second half stops the sweep from being
# undone by a later deletion.
# =========================================================================

TIER6I_ACTIVE_BACKEND_SURFACES = (
    REPOSITORY_ROOT / "README.md",
    REPOSITORY_ROOT / "CLAUDE.md",
    REPOSITORY_ROOT / "docs" / "installation.rst",
    REPOSITORY_ROOT / "docs" / "quickstart.rst",
    REPOSITORY_ROOT / "docs" / "user_guide" / "backends.rst",
    REPOSITORY_ROOT / "docs" / "user_guide" / "configuration.rst",
    REPOSITORY_ROOT / "docs" / "user_guide" / "configuration_support.rst",
    REPOSITORY_ROOT / "docs" / "api" / "backends.rst",
)


@pytest.mark.parametrize(
    "path", TIER6I_ACTIVE_BACKEND_SURFACES, ids=lambda path: path.name
)
def test_tier6i_active_docs_never_offer_the_removed_numba_backend(path):
    """Tier 6H removed the name; no active document may still offer it.

    ``docs/migration_guide.md`` is deliberately absent from the list: its whole
    job is to name the removed identifier. Every other surface here instructs a
    reader to *do* something, and instructing them to select ``numba`` or to
    install ``radiosim[numba]`` is instructing them to hit an error.
    """
    text = path.read_text(encoding="utf-8")

    assert "radiosim[numba]" not in text, path
    for stale in (
        "JAX/Numba",
        "JAX or Numba",
        "JAX, Numba",
        "NumPy, JAX, Numba",
        "numpy | jax | numba",
        "numba | auto",
        "Numba backend",
    ):
        assert stale not in text, f"{path}: {stale}"

    # The name may still appear, but only while being removed. A document that
    # mentions it without saying so is offering it.
    if "numba" in text.lower():
        assert "removed" in text.lower(), path


def test_tier6i_backend_guide_states_the_measured_position():
    """Section 26.2: the guide reports measurement, not disclaimer."""
    text = (REPOSITORY_ROOT / "docs" / "user_guide" / "backends.rst").read_text(
        encoding="utf-8"
    )

    # The stale prose Section 26.1/26.2 names, gone.
    assert "incomplete backend coverage" not in text
    assert "Numba" not in text

    # The backend table, the auto precedence, and the rename.
    assert "``dask``" in text
    assert "non-CPU" in text
    assert "Renamed from ``numba`` before v1.0." in text

    # The compilation boundary.
    assert "baseline_contraction" in text
    assert "Exactly **one** kernel is compiled" in text

    # Every host-side stage, named in one place with a reason.
    for stage in (
        "Astropy coordinate transforms",
        "Horizon masking",
        "Planck brightness conversion",
        "FITS beam interpolation",
        "HEALPix direction cosines",
    ):
        assert stage in text, stage

    # The measured position, with its record citation.
    assert "output/benchmarks/reference/" in text
    assert "bit-identical" in text
    assert 'accelerator: "none"' in text
    assert "pixi run bench" in text


def test_tier6i_backend_guide_makes_no_uncited_speed_or_gpu_claim():
    """Section 26: a speed or GPU sentence without a record does not ship."""
    text = (REPOSITORY_ROOT / "docs" / "user_guide" / "backends.rst").read_text(
        encoding="utf-8"
    )

    # No speedup multiplier in either direction may appear without the record
    # set being cited in the same document.
    speedup_claims = re.findall(r"\b\d+(?:\.\d+)?x\b", text)
    if speedup_claims:
        assert "output/benchmarks/reference/" in text

    # And nothing may claim the accelerator that was never run.
    for forbidden in (
        "GPU acceleration",
        "runs on a GPU",
        "GPU-accelerated",
        "universal hardware acceleration",
    ):
        assert forbidden not in text, forbidden


def test_tier6i_readme_reports_measured_backend_reality():
    """Section 26.1."""
    text = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")

    assert "incomplete backend coverage" not in text
    assert "Do not infer complete GPU execution" not in text
    assert "NumPy, JAX, Dask, or `auto`" in text
    assert "output/benchmarks/reference/" in text
    assert "bit-identical to NumPy" in text
    assert '`accelerator: "none"`' in text
    assert "pixi run bench" in text


def test_tier6i_configuration_guide_documents_the_new_blocks_and_hybrid():
    """Section 26.3: the three new blocks and the hybrid mode, with rejections."""
    text = (REPOSITORY_ROOT / "docs" / "user_guide" / "configuration.rst").read_text(
        encoding="utf-8"
    )

    assert "sky_loading:" in text
    assert "solver:" in text
    assert "allow_lossy_point_rasterization" in text
    assert "sky_representation: hybrid" in text
    assert "V_total = V_point + V_healpix" in text

    # The Section 18.3 rejections, verbatim.
    for rejection in (
        "execution.n_workers: not a field; use execution.sky_loading.max_workers for",
        "execution.solver.workers must be a positive integer.",
        "execution.solver.executor=process: unsupported; the solver closure holds beam",
        "visibility.sky_representation=hybrid requires a sky model with both a",
        "visibility.sky_representation=point_sources would discard the HEALPix payload",
        "visibility.sky_representation=healpix_map would rasterize {n} point source(s)",
    ):
        assert rejection in text, rejection


def test_tier6i_claude_status_matches_post_tier6_reality():
    """Section 26.4, as amended: the stale sentences are gone."""
    text = (REPOSITORY_ROOT / "CLAUDE.md").read_text(encoding="utf-8")

    # The three sentences Section 26 names.
    assert "`jit`/`vmap`/`jit_compile` are defined but never applied" not in text
    assert (
        "passes config validation but raises `NotImplementedError` at runtime"
        not in text
    )
    assert "rejected during config validation" in text

    # The sentences Tier 6H's rename falsified.
    assert "`numba_backend.py` — Numba/Dask" not in text
    assert "`ArrayBackend` (NumPy/JAX/Numba)" not in text
    assert "GPU backends (JAX/Numba) are scaffolded" not in text
    assert '`get_backend("auto" | "numpy" | "jax" | "numba")`' not in text
    assert '`get_backend("auto" | "numpy" | "jax" | "dask")`' in text
    assert "dask_backend.py" in text
    assert "(NumPy, JAX, Dask)" in text

    # And the newly provable ones.
    assert "core/contraction.py" in text
    assert "output/benchmarks/reference/" in text
    assert "pixi run bench" in text


def test_tier6i_migration_guide_maps_every_tier6_breaking_change():
    """Section 26.5: one entry per Section 36 row."""
    text = (REPOSITORY_ROOT / "docs" / "migration_guide.md").read_text(encoding="utf-8")

    for entry in (
        "execution.backend: numba",  # C2
        "load_models_parallel()",  # C3
        "`execution.offline: true` is now authoritative",  # C4
        "Solver accumulation restructure",  # C5
        "run() got an unexpected keyword argument 'n_workers'",  # C6
        "sky_representation` accepts a third value, `hybrid`",  # C7
        "allow_lossy_point_rasterization",  # C8, C9
        "component_element_counts",  # C10
        "`scientific_sha256` changes for every result",  # C11
        "`provenance_sha256` changes with them",  # C12
        "HDF5 schema `3.0.0`",  # C13
        "HDF5 schema `4.0.0`",  # Tier 7D
        "`NumbaBackend` is now `DaskBackend`",  # C14
        "no longer returns the NumPy-delegating backend",  # C15
        "`RIMESimulator.supports_gpu` is now `False`",  # C16
        "def supports_compilation(self) -> bool",  # C17
        "CPU-only JAX is a declared dependency",  # C18
    ):
        assert entry in text, entry


# ---------------------------------------------------------------------------
# Tier 7J: documentation truth for the post-Tier-7 Jones surface.
#
# `Tier7JonesSciencePlan.md` Section 26 defects D0 and D21 are documentation
# drift, not code defects: `CLAUDE.md` claimed a class count and a term
# disposition that the implementation slices falsified, and `docs/api/jones.rst`
# grouped now-implemented terms under "Planned terms".  Section 37 criterion 17
# is the acceptance form of the same thing.  These tests are the residual scan
# that keeps the fix from silently regressing; each fails on the text that was
# there before this slice.
# ---------------------------------------------------------------------------

TIER7J_JONES_SURFACES = (
    REPOSITORY_ROOT / "README.md",
    REPOSITORY_ROOT / "CLAUDE.md",
    REPOSITORY_ROOT / "docs" / "index.rst",
    REPOSITORY_ROOT / "docs" / "changelog.rst",
    REPOSITORY_ROOT / "docs" / "migration_guide.md",
    REPOSITORY_ROOT / "docs" / "api" / "jones.rst",
    REPOSITORY_ROOT / "docs" / "user_guide" / "beam_models.rst",
    REPOSITORY_ROOT / "docs" / "user_guide" / "configuration.rst",
    REPOSITORY_ROOT / "docs" / "user_guide" / "configuration_support.rst",
    REPOSITORY_ROOT / "docs" / "user_guide" / "jones_matrices.rst",
    REPOSITORY_ROOT / "docs" / "user_guide" / "jones_terms.rst",
)

# Each entry is a sentence a reader could take as "RadioSim does not model
# this", about a term RadioSim now models.  They are matched against
# whitespace-collapsed text so a reflow cannot hide one.
TIER7J_FALSIFIED_CLAIMS = (
    "Other exported Jones classes are scaffolding",
    "Polarization leakage, parallactic rotation, gains, bandpass",
    "including polarization leakage, parallactic rotation, gains, bandpass",
    "Polarization leakage (``D``), parallactic rotation (``P``), gains "
    "(``G``), bandpass (``B``), elliptical",
    "because the parallactic-angle term is not implemented",
    "Polarization leakage and a beam that genuinely differs between the two "
    "feeds remain Tier 7 work",
    "Only K, E, C, and H implement real physics",
    '``"planned"`` for the rest',
    'which is `"implemented"` or `"planned"`',
    "Planned terms",
)


def test_parallactic_guide_uses_effective_mount_angles_for_observables() -> None:
    """Nasmyth and heterogeneous baselines must not be described by plain psi."""
    text = " ".join(
        (REPOSITORY_ROOT / "docs" / "user_guide" / "jones_terms.rst")
        .read_text(encoding="utf-8")
        .split()
    )

    assert r":math:`2\alpha`" in text
    assert r"\alpha_p-\alpha_q" in text
    assert r"\alpha_p+\alpha_q" in text
    assert r"2\psi" not in text
    assert r"e^{-2i\psi}" not in text


@pytest.mark.parametrize("path", TIER7J_JONES_SURFACES, ids=lambda path: path.name)
def test_tier7j_no_truth_surface_still_denies_an_implemented_term(path):
    text = " ".join(path.read_text(encoding="utf-8").split())

    for stale in TIER7J_FALSIFIED_CLAIMS:
        assert " ".join(stale.split()) not in text, (path.name, stale)


@pytest.mark.parametrize("path", TIER7J_JONES_SURFACES, ids=lambda path: path.name)
def test_tier7j_no_truth_surface_claims_an_unmeasured_validation(path):
    """Section 29.2: a validation claim names its quantity and tolerance."""
    text = " ".join(path.read_text(encoding="utf-8").split())

    for forbidden in (
        "validated against CASA",
        "cross-checked against RASCIL",
        "matches CASA",
        "validated against matvis",
    ):
        assert forbidden not in text, (path.name, forbidden)
    if "pyuvsim" in text:
        # Wherever a page names pyuvsim it must also name the version, because
        # Section 41's Q1 answer is specific to 1.4.0 and "latest" is a moving
        # claim.
        assert "1.4.0" in text, path.name


def test_tier7j_api_reference_documents_every_implemented_term_module():
    """Section 37 criterion 17, for ``docs/api/jones.rst``."""
    text = (REPOSITORY_ROOT / "docs" / "api" / "jones.rst").read_text(encoding="utf-8")

    for module in (
        "radiosim.core.jones.base",
        "radiosim.core.jones.chain",
        "radiosim.core.jones.directions",
        "radiosim.core.jones.evaluate",
        "radiosim.core.jones.geometric",
        "radiosim.core.jones.receptor",
        "radiosim.core.jones.parallactic",
        "radiosim.core.jones.ionosphere",
        "radiosim.core.jones.troposphere",
        "radiosim.core.jones.polarization_leakage",
        "radiosim.core.jones.crosshand",
        "radiosim.core.jones.delay",
        "radiosim.core.jones.bandpass",
        "radiosim.core.jones.gain",
        "radiosim.core.jones.baseline_errors",
    ):
        assert f".. automodule:: {module}\n" in text, module

    # Every module the package actually ships is documented, so a future term
    # cannot be added without a reference entry.
    package = REPOSITORY_ROOT / "src" / "radiosim" / "core" / "jones"
    shipped = {path.stem for path in package.glob("*.py") if path.stem != "__init__"}
    documented = {
        name.rsplit(".", 1)[-1]
        for name in re.findall(
            r"^\.\. automodule:: (radiosim\.core\.jones\.\w+)$", text, re.MULTILINE
        )
    }
    assert shipped == documented, shipped ^ documented

    assert "nineteen names" in text
    assert 'There is no ``"planned"`` term left.' in text


def test_tier7j_claude_md_describes_the_post_tier7_jones_surface():
    """Section 26 defect D0: the class count and the term disposition."""
    text = (REPOSITORY_ROOT / "CLAUDE.md").read_text(encoding="utf-8")

    assert "46 exported classes" not in text
    assert "exports exactly **19 names**" in text
    assert '`term_status` is `"implemented"` for all of them' in text
    assert "TODO: implement properly" not in text
    # The removed modules must be named as removed, not listed as extended
    # terms, because a reader following the old list would import three modules
    # that no longer exist.
    assert "**Extended terms**: `faraday.py` (F), `wterm.py` (W)" not in text
    assert "**Removed modules**" in text
    # And the two baseline terms must be described as outside the chain.
    assert "Hadamard product" in text
    assert "output/crossvalidation/" in text


def test_tier7j_readme_states_the_implemented_jones_capability():
    text = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")

    assert "a typed `jones:` section" in text
    assert 'term_status: "implemented"' in text
    assert "No exported term is an unconditional identity stub" in text


def test_tier7j_configuration_guide_shows_every_configurable_term():
    """Section 37 criterion 17, for ``docs/user_guide/configuration.rst``.

    The guide's ``jones:`` example is the first thing a reader copies, so a
    term missing from it reads as a term that cannot be configured.
    """
    text = (REPOSITORY_ROOT / "docs" / "user_guide" / "configuration.rst").read_text(
        encoding="utf-8"
    )
    block = text.split("   jones:\n", 1)
    assert len(block) == 2, "the configuration guide has no jones: example"
    example = block[1].split("\nTerms are applied", 1)[0]

    for term in ("G", "B", "Rc", "Kd", "X", "D", "P", "T", "Z", "M", "Q"):
        assert f"\n     {term}:\n" in example, term
    assert "There is no ``enabled: false``" in text


def test_tier7j_changelog_and_migration_guide_carry_every_tier7_ledger_row():
    """Section 36 rows B1-B16 each need a migration line and a changelog line."""
    changelog = (REPOSITORY_ROOT / "docs" / "changelog.rst").read_text(encoding="utf-8")
    migration = (REPOSITORY_ROOT / "docs" / "migration_guide.md").read_text(
        encoding="utf-8"
    )

    for entry in (
        "compute_jones_batch",  # B1, B3
        "evaluate_antenna_jones",  # B2
        "geometric_phase()",  # B4
        "Twenty-six exported Jones classes were removed",  # B5
        "renamed `CrosshandJones`",  # B6
        "`jones_config=` parameter was removed",  # B7
        "visibility.calculation_type",  # B8
        "jones",  # B9
        "HDF5 schema `4.0.0`",  # B11
        "`P` moved sky-side of `C`",  # B13
        "UnsupportedMountTypeError",  # B14
        "`beams.pointing` and `beams.surface_error`",  # B15
        "beam_physics_scope.md",  # B16
    ):
        assert entry in migration, entry

    for entry in (
        "Every Jones term now implements real physics",
        "typed** ``jones:`` **configuration section",
        "Direction-batched Jones evaluation",
        "Beam pointing offsets and Ruze surface efficiency",
        "The canonical Jones chain order",
        "``pyuvsim 1.4.0``",
        "output/crossvalidation/",
    ):
        assert entry in changelog, entry


def test_tier7j_crossvalidation_evidence_is_committed_and_bounded():
    """Section 29 and Section 37 criterion 19."""
    directory = REPOSITORY_ROOT / "output" / "crossvalidation"
    records = sorted(directory.glob("*.json"))
    assert records, directory
    assert (directory / "README.md").is_file()

    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["gating"] is False
        if record.get("schema") == "radiosim-crossvalidation-1.2.0":
            # The SCI-007 record has a deliberately stricter schema and is
            # authenticated by the dedicated deep validator tests below.
            assert path.name == "2026-08-11-pyuvsim-1.4.0-sci007.json"
            continue
        assert record["reference"]["version"] == "1.4.0"
        # A recorded comparison without a measured number is not evidence.
        for case in record["cases"]:
            assert case["test"]
            assert case["mapping_applied"]
            assert any(key.startswith("measured") for key in case)
        # And what it does not license has to be written down, so a later
        # reader cannot promote it into an unqualified claim.
        assert record["claims_not_licensed_by_this_record"]
        assert all(item["routed_to"] for item in record["unresolved"])


def test_tier7j_the_crossvalidation_module_is_excluded_from_the_default_gate():
    """The Tier-2 comparison must never gate (Section 29.1)."""
    source = (
        REPOSITORY_ROOT / "tests" / "crossvalidation" / "test_pyuvsim_comparison.py"
    ).read_text(encoding="utf-8")

    assert "pytestmark = [pytest.mark.crossval, pytest.mark.slow]" in source
    assert "pytest.importorskip(" in source
    assert '"pyuvsim",' in source

    # The marker is registered, so it is a selector rather than an unknown-mark
    # warning, and no CI job builds the environment that carries pyuvsim.
    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "crossval: marks cross-implementation comparisons" in pyproject
    workflow = (REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    assert "crossval" not in workflow


SCI007_EVIDENCE_TOOL = REPOSITORY_ROOT / "tools" / "wp6_sci007_evidence.py"
SCI007_EVIDENCE_ARTIFACT = (
    REPOSITORY_ROOT
    / "output"
    / "crossvalidation"
    / "2026-08-11-pyuvsim-1.4.0-sci007.json"
)
SCI007_APPROVED_SOURCE_SHA = "9b50805cf9fe32124800d1e3946a87e3911c376b"
SCI007_ARTIFACT_SHA256 = (
    "3a441ad606f365ac4110e30d9d8c2f3d7f5ea91c481aa70488dea72487e570ba"
)
_SCI007_VALIDATOR_TEST_SOURCE_SHA = "0" * 40


def _load_sci007_evidence_tool():
    spec = importlib.util.spec_from_file_location(
        "_radiosim_sci007_evidence_test", SCI007_EVIDENCE_TOOL
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sci007_prediction_record(module):
    public_rad = copy.deepcopy(module._EXPECTED_PUBLIC_RADIANS)
    exact_deg = copy.deepcopy(module._EXPECTED_EXACT_DEGREES)
    exact_rad = [[math.radians(value) for value in row] for row in exact_deg]
    public_deg = [[math.degrees(value) for value in row] for row in public_rad]
    difference_rad = [
        [public_rad[row][column] - exact_rad[row][column] for column in range(3)]
        for row in range(3)
    ]
    difference_deg = [
        [public_deg[row][column] - exact_deg[row][column] for column in range(3)]
        for row in range(3)
    ]
    absolute_rad = [[abs(value) for value in row] for row in difference_rad]
    relative = [
        [absolute_rad[row][column] / abs(exact_rad[row][column]) for column in range(3)]
        for row in range(3)
    ]
    public_flat = [value for row in public_rad for value in row]
    exact_flat = [value for row in exact_rad for value in row]
    relative_flat = [value for row in relative for value in row]
    return {
        "public": {"radians": public_rad, "degrees": public_deg},
        "exact": {"radians": exact_rad, "degrees": exact_deg},
        "public_minus_exact": {
            "radians": difference_rad,
            "degrees": difference_deg,
            "absolute_rad": absolute_rad,
            "relative": relative,
        },
        "extrema": {
            "public_min_abs_rad": min(abs(value) for value in public_flat),
            "public_max_abs_rad": max(abs(value) for value in public_flat),
            "exact_min_abs_rad": min(abs(value) for value in exact_flat),
            "exact_max_abs_rad": max(abs(value) for value in exact_flat),
            "public_exact_max_relative": max(relative_flat),
            "public_spin2_effect_max": max(
                abs(complex(math.cos(2.0 * value), math.sin(2.0 * value)) - 1.0)
                for value in public_flat
            ),
        },
    }


def _sci007_pinned_metric(scalars, contract):
    numerator_units, denominator_name, denominator_units, definition = contract
    return {
        "value": scalars["value"],
        "numerator_value": scalars["numerator_value"],
        "numerator_units": numerator_units,
        "denominator_name": denominator_name,
        "denominator_value": scalars["denominator_value"],
        "denominator_units": denominator_units,
        "definition": definition,
    }


def _sci007_metric_record(module):
    metrics = {
        name: _sci007_pinned_metric(
            scalars,
            module._RELATIVE_METRIC_CONTRACT[name],
        )
        for name, scalars in module._EXPECTED_RELATIVE_METRIC_SCALARS.items()
    }
    metrics.update(
        {
            "intensity_scale": {
                "value": module._EXPECTED_SCALE_VALUES["intensity_scale"],
                "units": "Jy",
                "definition": "max(abs(I_PY))",
            },
            "linear_scale": {
                "value": module._EXPECTED_SCALE_VALUES["linear_scale"],
                "units": "Jy",
                "definition": "max(abs(L_PY))",
            },
        }
    )
    metrics["fitted_complex_ratio"] = {
        **copy.deepcopy(module._EXPECTED_FITTED_COMPLEX_RATIO),
        "definition": "vdot(L_PY,L_RS)/vdot(L_PY,L_PY) over valid cells",
    }
    metrics["linear_cells"] = {
        "valid": 27,
        "total": 27,
        "mask_definition": "abs(L_reference) > linear_scale * 1e-12",
    }
    metrics["source_additivity"] = {
        name: _sci007_pinned_metric(
            scalars,
            module._SOURCE_ADDITIVITY_CONTRACT[name],
        )
        for name, scalars in module._EXPECTED_SOURCE_ADDITIVITY_SCALARS.items()
    }
    return metrics


def _valid_sci007_record(module):
    predictions = _sci007_prediction_record(module)
    samples = [
        {
            "time_index": index,
            "jd1": module._EXPECTED_JD1[index],
            "jd2": module._EXPECTED_JD2[index],
            "dut1_seconds": module._EXPECTED_DUT1_SECONDS[index],
            "xp_arcsec": module._EXPECTED_XP_ARCSEC[index],
            "yp_arcsec": module._EXPECTED_YP_ARCSEC[index],
            "dut1_status": {"code": 0, "name": "FROM_IERS_B"},
            "polar_motion_status": {"code": 0, "name": "FROM_IERS_B"},
        }
        for index in range(3)
    ]
    return {
        "schema": module.SCHEMA,
        "recorded_utc": "2026-08-11T00:00:00Z",
        "slice": module.SLICE,
        "gating": False,
        "identity": module._identity(_SCI007_VALIDATOR_TEST_SOURCE_SHA),
        "reference": copy.deepcopy(module._REFERENCE),
        "runtime": copy.deepcopy(module._RUNTIME),
        "iers": module._iers_record(samples),
        "fixture": copy.deepcopy(module._fixture()),
        "axes": copy.deepcopy(module._AXES),
        "equations": copy.deepcopy(module._EQUATIONS),
        "correction": copy.deepcopy(module._CORRECTION),
        "predictions": predictions,
        "history": copy.deepcopy(module._HISTORY),
        "tolerances": copy.deepcopy(module._TOLERANCES),
        "metrics": _sci007_metric_record(module),
        "limits": copy.deepcopy(module._LIMITS),
    }


def _replace_sci007_value(record, path, value):
    target = record
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def test_sci007_validator_is_standard_library_only_at_import_time():
    script = f"""
import builtins
import importlib.util

blocked = {{"numpy", "astropy", "pyuvsim", "pyradiosky", "pyuvdata"}}
original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked:
        raise AssertionError(f"optional dependency imported: {{name}}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
spec = importlib.util.spec_from_file_location("sci007_evidence", {str(SCI007_EVIDENCE_TOOL)!r})
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.SCHEMA == "radiosim-crossvalidation-1.2.0"
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("spoof", "error"),
    (
        ("environment-name", "PIXI_ENVIRONMENT_NAME=crossval"),
        ("prefix", "not running from the repository's locked crossval environment"),
        ("project-root", "PIXI_PROJECT_ROOT"),
        ("manifest", "PIXI_PROJECT_MANIFEST"),
        ("working-directory", "generation must run from repository root"),
    ),
)
def test_sci007_generation_rejects_spoofed_pixi_provenance(
    spoof, error, tmp_path, monkeypatch
):
    module = _load_sci007_evidence_tool()
    expected_prefix = REPOSITORY_ROOT / ".pixi" / "envs" / "crossval"
    expected_manifest = REPOSITORY_ROOT / "pixi.toml"
    monkeypatch.setenv("PIXI_ENVIRONMENT_NAME", "crossval")
    monkeypatch.setenv("PIXI_PROJECT_ROOT", str(REPOSITORY_ROOT))
    monkeypatch.setenv("PIXI_PROJECT_MANIFEST", str(expected_manifest))
    monkeypatch.setattr(module.sys, "prefix", str(expected_prefix))
    monkeypatch.chdir(REPOSITORY_ROOT)
    module._assert_generation_environment()

    if spoof == "environment-name":
        monkeypatch.setenv("PIXI_ENVIRONMENT_NAME", "default")
    elif spoof == "prefix":
        monkeypatch.setattr(module.sys, "prefix", str(tmp_path / "crossval"))
    elif spoof == "project-root":
        monkeypatch.setenv("PIXI_PROJECT_ROOT", str(tmp_path))
    elif spoof == "manifest":
        monkeypatch.setenv("PIXI_PROJECT_MANIFEST", str(tmp_path / "pixi.toml"))
    else:
        monkeypatch.chdir(tmp_path)

    with pytest.raises(module.EvidenceError, match=re.escape(error)):
        module._assert_generation_environment()


def test_sci007_in_memory_record_has_exact_complete_contract():
    module = _load_sci007_evidence_tool()
    record = _valid_sci007_record(module)

    assert module.SCHEMA == "radiosim-crossvalidation-1.2.0"
    assert set(record) == {
        "schema",
        "recorded_utc",
        "slice",
        "gating",
        "identity",
        "reference",
        "runtime",
        "iers",
        "fixture",
        "axes",
        "equations",
        "correction",
        "predictions",
        "history",
        "tolerances",
        "metrics",
        "limits",
    }
    assert record["fixture"]["primary"]["aggregate_shape"] == [3, 3, 3, 4]
    assert record["fixture"]["primary"]["source_decomposed_shape"] == [
        3,
        3,
        3,
        3,
        4,
    ]
    assert record["axes"]["polarization"] == {
        "order": ["XX", "XY", "YX", "YY"],
        "shape": [4],
    }
    assert record["equations"]["Delta"] == (
        "Delta=wrap_pi(psi_RS+atan2(K[0,1],K[0,0]))"
    )
    assert record["correction"]["radiosim_to_pyuvsim"]["factor"] == ("exp(-2j*Delta)")
    module.validate_record(
        record, approved_source_sha=_SCI007_VALIDATOR_TEST_SOURCE_SHA
    )


@pytest.mark.parametrize(
    ("path", "value", "error"),
    (
        (("schema",), "radiosim-crossvalidation-1.1.0", "schema changed"),
        (
            ("reference", "pyuvsim", "version"),
            "1.4.1",
            "reference.pyuvsim.version changed",
        ),
        (
            ("runtime", "packages", "pyuvsim"),
            "1.4.1",
            "runtime.packages.pyuvsim changed",
        ),
        (
            ("identity", "generating_source_sha"),
            "f" * 40,
            "identity.generating_source_sha changed",
        ),
        (
            ("identity", "clean_tree_at_generation"),
            1,
            "identity.clean_tree_at_generation type changed",
        ),
        (
            ("identity", "lockfile", "sha256"),
            "f" * 64,
            "identity.lockfile.sha256 changed",
        ),
        (
            ("iers", "table_sha256"),
            "0" * 64,
            "iers metadata.table_sha256 changed",
        ),
        (
            ("iers", "samples", 0, "dut1_status", "code"),
            False,
            "iers.samples[0].dut1_status.code type changed",
        ),
        (
            ("fixture", "primary", "sources", 0, "ra_deg"),
            20.0001,
            "fixture.primary.sources[0].ra_deg changed",
        ),
        (
            ("fixture", "primary", "site", "authored", "height_m"),
            1073,
            "fixture.primary.site.authored.height_m type changed",
        ),
        (
            ("axes", "source_cube", "shape"),
            [3, 3, 3, 3, 5],
            "axes.source_cube.shape",
        ),
        (
            ("equations", "Delta"),
            "Delta=wrap_pi(psi_RS-atan2(K[0,1],K[0,0]))",
            "equations.Delta changed",
        ),
        (
            ("correction", "radiosim_to_pyuvsim", "factor"),
            "exp(+2j*Delta)",
            "correction.radiosim_to_pyuvsim.factor changed",
        ),
        (
            ("tolerances", "normal", "public_max_abs_rad", "value"),
            1.3e-3,
            "tolerances.normal.public_max_abs_rad.value changed",
        ),
        (
            ("predictions", "public", "radians", 0, 0),
            0.0010531231588649,
            "predictions.public.radians",
        ),
        (
            ("predictions", "public", "radians"),
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "predictions.public.radians must have shape [3,3]",
        ),
        (
            ("metrics", "exact_source_time_relative", "numerator_value"),
            1.0e-6,
            "metrics.exact_source_time_relative.value changed",
        ),
        (
            ("limits", "production_frame_policy_changed"),
            True,
            "limits.production_frame_policy_changed changed",
        ),
        (
            ("limits", "production_code_changed"),
            0,
            "limits.production_code_changed type changed",
        ),
    ),
    ids=(
        "schema",
        "reference-version",
        "runtime-version",
        "source-sha",
        "identity-bool-as-int",
        "lock-sha",
        "iers-sha",
        "iers-int-as-bool",
        "fixture-scalar",
        "fixture-float-as-int",
        "axis-shape",
        "delta-formula",
        "correction-sign",
        "tolerance",
        "prediction-scalar",
        "prediction-shape",
        "metric",
        "production-limit",
        "limit-bool-as-int",
    ),
)
def test_sci007_validator_rejects_representative_semantic_mutations(path, value, error):
    module = _load_sci007_evidence_tool()
    record = _valid_sci007_record(module)
    _replace_sci007_value(record, path, value)

    with pytest.raises(module.EvidenceError, match=re.escape(error)):
        module.validate_record(
            record, approved_source_sha=_SCI007_VALIDATOR_TEST_SOURCE_SHA
        )


def test_sci007_validator_rejects_negative_absolute_error_metrics():
    module = _load_sci007_evidence_tool()
    record = _valid_sci007_record(module)
    metric = record["metrics"]["intensity_relative"]
    metric["value"] = -metric["value"]
    metric["numerator_value"] = -metric["numerator_value"]

    with pytest.raises(module.EvidenceError, match="value must be nonnegative"):
        module.validate_record(
            record, approved_source_sha=_SCI007_VALIDATOR_TEST_SOURCE_SHA
        )


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_sci007_validator_rejects_in_memory_nonfinite_numbers(value):
    module = _load_sci007_evidence_tool()
    record = _valid_sci007_record(module)
    record["metrics"]["raw_linear_relative"]["value"] = value

    with pytest.raises(module.EvidenceError, match="must be finite"):
        module.validate_record(
            record, approved_source_sha=_SCI007_VALIDATOR_TEST_SOURCE_SHA
        )


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
def test_sci007_json_loader_rejects_nonfinite_numbers(tmp_path, constant):
    module = _load_sci007_evidence_tool()
    path = tmp_path / "nonfinite.json"
    path.write_text(f'{{"value": {constant}}}\n', encoding="utf-8")

    with pytest.raises(module.EvidenceError, match="non-finite JSON number"):
        module._load_json(path)


def test_sci007_json_loader_rejects_duplicate_keys(tmp_path):
    module = _load_sci007_evidence_tool()
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema": "first", "schema": "second"}\n', encoding="utf-8")

    with pytest.raises(module.EvidenceError, match="duplicate JSON key: 'schema'"):
        module._load_json(path)


def test_sci007_artifact_validator_authenticates_raw_bytes(tmp_path):
    module = _load_sci007_evidence_tool()
    record = _valid_sci007_record(module)
    path = tmp_path / "record.json"
    raw = (
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    path.write_bytes(raw)
    measured_sha256 = hashlib.sha256(raw).hexdigest()

    assert (
        module.validate_artifact(
            path,
            approved_source_sha=_SCI007_VALIDATOR_TEST_SOURCE_SHA,
            artifact_sha256=measured_sha256,
        )
        == record
    )
    with pytest.raises(module.EvidenceError, match="artifact SHA-256 mismatch"):
        module.validate_artifact(
            path,
            approved_source_sha=_SCI007_VALIDATOR_TEST_SOURCE_SHA,
            artifact_sha256="f" * 64,
        )


def test_sci007_artifact_validator_hashes_and_decodes_one_byte_snapshot(
    tmp_path, monkeypatch
):
    module = _load_sci007_evidence_tool()
    first_record = _valid_sci007_record(module)
    second_record = copy.deepcopy(first_record)
    second_record["recorded_utc"] = "2026-08-11T00:00:01Z"
    first_raw = (
        json.dumps(first_record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    second_raw = (
        json.dumps(second_record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    path = tmp_path / "record.json"
    path.write_bytes(first_raw)
    expected_sha256 = hashlib.sha256(first_raw).hexdigest()
    original_open = Path.open
    opened_snapshots = []

    def swapping_open(target, mode="r", *args, **kwargs):
        if target == path and mode == "rb":
            raw = first_raw if not opened_snapshots else second_raw
            opened_snapshots.append(raw)
            return io.BytesIO(raw)
        return original_open(target, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swapping_open)
    validated = module.validate_artifact(
        path,
        approved_source_sha=_SCI007_VALIDATOR_TEST_SOURCE_SHA,
        artifact_sha256=expected_sha256,
    )

    assert validated["recorded_utc"] == first_record["recorded_utc"]
    assert opened_snapshots == [first_raw]


def test_sci007_artifact_validator_fails_closed_when_artifact_is_absent(tmp_path):
    module = _load_sci007_evidence_tool()

    with pytest.raises(module.EvidenceError, match="artifact is absent"):
        module.validate_artifact(
            tmp_path / "absent.json",
            approved_source_sha=_SCI007_VALIDATOR_TEST_SOURCE_SHA,
            artifact_sha256="0" * 64,
        )


def test_sci007_canonical_artifact_is_committed_and_deeply_validated():
    # Expected source-stage failure: the approved clean source commit must not
    # contain its own successor artifact. The evidence successor replaces both
    # digest placeholders and adds the canonical file, making this test green.
    assert SCI007_EVIDENCE_ARTIFACT.is_file(), (
        "expected source-stage failure: generate the canonical SCI-007 artifact "
        "from the approved clean source commit, then add it only in its "
        "evidence successor"
    )
    assert re.fullmatch(r"[0-9a-f]{40}", SCI007_APPROVED_SOURCE_SHA), (
        "replace SCI007_APPROVED_SOURCE_SHA in the evidence successor"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", SCI007_ARTIFACT_SHA256), (
        "replace SCI007_ARTIFACT_SHA256 in the evidence successor"
    )

    module = _load_sci007_evidence_tool()
    record = module.validate_artifact(
        SCI007_EVIDENCE_ARTIFACT,
        approved_source_sha=SCI007_APPROVED_SOURCE_SHA,
        artifact_sha256=SCI007_ARTIFACT_SHA256,
    )
    assert record["identity"]["generating_source_sha"] == (SCI007_APPROVED_SOURCE_SHA)
    assert record["identity"]["artifact_absent_at_generation"] is True
    assert record["identity"]["clean_tree_at_generation"] is True
    assert record["gating"] is False

    readme = (SCI007_EVIDENCE_ARTIFACT.parent / "README.md").read_text(encoding="utf-8")
    assert SCI007_EVIDENCE_ARTIFACT.name in readme
    assert "SCI-007" in readme
