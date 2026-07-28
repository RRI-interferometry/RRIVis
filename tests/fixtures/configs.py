"""Reusable configuration builders for configuration contract tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from radiosim.io.config import RadioSimConfig

if TYPE_CHECKING:
    from radiosim.core.runtime_config import ResolvedConfiguration


def write_minimal_antenna_file(tmp_path: Path) -> Path:
    """Write a deterministic two-antenna file in native RadioSim format."""
    antenna_path = tmp_path / "antennas.txt"
    antenna_path.write_text(
        "Name Number BeamID E N U Diameter\n"
        "ANT0 0 0 0.0 0.0 0.0 14.0\n"
        "ANT1 1 0 14.0 0.0 0.0 14.0\n"
    )
    return antenna_path


def _deep_merge(
    base: dict[str, Any],
    updates: Mapping[str, object],
) -> dict[str, Any]:
    """Return a test-owned recursive merge without mutating either input."""
    merged = deepcopy(base)
    for key, value in updates.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def valid_config_mapping(
    tmp_path: Path,
    *,
    frequency: Mapping[str, object] | None = None,
    sky_sources: Sequence[Mapping[str, object]] | None = None,
    **section_overrides: Mapping[str, object],
) -> dict[str, Any]:
    """Build a complete local mapping for the strict target input schema."""
    antenna_path = write_minimal_antenna_file(tmp_path)
    config: dict[str, Any] = {
        "instrument": {
            "source": {
                "kind": "layout_file",
                "path": str(antenna_path),
                "format": "radiosim",
                "telescope_name": "Tier1ATestArray",
            },
            "location": {
                "longitude_deg": 21.4283,
                "latitude_deg": -30.72152,
                "height_m": 1073.0,
            },
            "default_diameter_m": 14.0,
        },
        "baseline_selection": {"correlations": "all"},
        "beams": {
            "mode": "analytic",
            "model": {
                "kind": "circular_aperture",
                "taper": {"kind": "gaussian", "edge_taper_db": 10.0},
            },
        },
        "obs_time": {
            "start_time": "2025-01-01T00:00:00",
            "duration_seconds": 2.0,
            "time_step_seconds": 1.0,
        },
        "obs_frequency": {
            "mode": "grid",
            "starting_frequency": 100.0,
            "frequency_interval": 1.0,
            "frequency_bandwidth": 2.0,
            "channel_width": 1.0,
            "frequency_unit": "MHz",
        },
        "sky_model": {
            "flux_unit": "Jy",
            "sources": [
                {
                    "kind": "test_sources",
                    "representation": "point_sources",
                    "num_sources": 2,
                    "distribution": "uniform",
                    "seed": 1,
                }
            ],
        },
        "visibility": {
            "calculation_type": "direct_sum",
            "sky_representation": "point_sources",
        },
        "execution": {
            "backend": "numpy",
            "offline": True,
            "precision": {"preset": "standard"},
            "simulator": "rime",
        },
        "workflow": {
            "output_dir": str(tmp_path / "output"),
            "run_subdir": "run",
            "result_filename": "visibilities",
            "result_format": "hdf5",
            "collision_policy": "error",
            "save_results": False,
            "plot_results": False,
            "open_plots_in_browser": False,
            "save_log": False,
        },
    }
    if frequency is not None:
        config["obs_frequency"] = deepcopy(dict(frequency))
    if sky_sources is not None:
        config["sky_model"]["sources"] = deepcopy(list(sky_sources))
    beam_override = section_overrides.get("beams")
    merged = _deep_merge(config, section_overrides)
    if beam_override is not None:
        # Tagged beam modes are complete alternatives, not mergeable fragments.
        merged["beams"] = deepcopy(dict(beam_override))
    return merged


def valid_input_config(tmp_path: Path, **overrides: object) -> RadioSimConfig:
    """Build the strict target Pydantic model through normal validation."""
    return RadioSimConfig.model_validate(valid_config_mapping(tmp_path, **overrides))


def resolved_config(tmp_path: Path, **overrides: object) -> ResolvedConfiguration:
    """Build a real offline Tier 1D resolved bundle through production code."""
    from radiosim.io.config_resolution import (
        ConfigurationSource,
        resolve_config,
    )

    config = valid_input_config(tmp_path, **overrides)
    source = ConfigurationSource.for_model(
        base_dir=tmp_path,
        invocation_dir=tmp_path,
        label="test resolved config",
    )
    return resolve_config(config, source=source)


def legacy_runtime_config_mapping(
    tmp_path: Path,
    **section_overrides: Mapping[str, object],
) -> dict[str, Any]:
    """Build the old raw shape solely for pre-Tier-1E runtime characterization."""
    antenna_path = write_minimal_antenna_file(tmp_path)
    legacy: dict[str, Any] = {
        "telescope": {"telescope_name": "Tier1BLegacyRuntimeArray"},
        "antenna_layout": {
            "antenna_positions_file": str(antenna_path),
            "antenna_file_format": "radiosim",
            "all_antenna_diameter": 14.0,
        },
        "beams": {
            "mode": "analytic",
            "model": {
                "kind": "circular_aperture",
                "taper": {"kind": "gaussian", "edge_taper_db": 10.0},
            },
        },
        "location": {"lat": -30.72152, "lon": 21.4283, "height": 1073.0},
        "obs_time": {
            "start_time": "2025-01-01T00:00:00",
            "duration_seconds": 2.0,
            "time_step_seconds": 1.0,
        },
        "obs_frequency": {
            "starting_frequency": 100.0,
            "frequency_interval": 1.0,
            "frequency_bandwidth": 2.0,
            "channel_width": 1.0,
            "frequency_unit": "MHz",
        },
        "sky_model": {"sources": [{"kind": "test_sources", "num_sources": 2}]},
        "visibility": {
            "calculation_type": "direct_sum",
            "sky_representation": "point_sources",
        },
        "compute": {"backend": "numpy", "offline": True},
        "precision": {"preset": "standard"},
        "output": {
            "simulation_data_dir": str(tmp_path / "output"),
            "simulation_subdir": "run",
            "output_file_name": "visibilities",
            "output_file_format": "HDF5",
            "save_simulation_data": False,
            "plot_results": False,
            "open_plots_in_browser": False,
            "save_log_data": False,
        },
    }
    return _deep_merge(legacy, section_overrides)


def write_config_yaml(
    tmp_path: Path,
    config: Mapping[str, object] | None = None,
    *,
    name: str = "config.yaml",
) -> Path:
    """Write a deterministic YAML config for public loader and CLI tests."""
    config_path = tmp_path / name
    data = valid_config_mapping(tmp_path) if config is None else deepcopy(dict(config))
    config_path.write_text(yaml.safe_dump(data, sort_keys=False))
    return config_path
