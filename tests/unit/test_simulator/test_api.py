"""Tests for the public Simulator constructor contract."""

from pathlib import Path

import pytest

from radiosim.api import Simulator

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HERA5_ANTENNAS = _REPO_ROOT / "antenna_layout_examples" / "hera_5.txt"


def test_simulator_rejects_removed_sky_model_shortcut():
    with pytest.raises(TypeError, match="sky_model"):
        Simulator(sky_model="test")


def test_simulator_accepts_tagged_sky_model_config():
    sim = Simulator(
        config={
            "sky_model": {
                "sources": [{"kind": "test_sources", "num_sources": 4}],
            },
            "visibility": {"sky_representation": "point_sources"},
        }
    )

    assert sim.config["sky_model"]["sources"][0]["kind"] == "test_sources"


def test_simulator_defaults_visibility_sky_representation():
    sim = Simulator(config={"sky_model": {"sources": [{"kind": "test_sources"}]}})

    assert sim.config["visibility"]["sky_representation"] == "point_sources"


def test_simulator_setup_resolves_sky_with_obs_frequency_config():
    """Regression: setup must not pass both frequencies and obs_frequency_config."""
    sim = Simulator(
        config={
            "antenna_layout": {
                "antenna_positions_file": str(_HERA5_ANTENNAS),
                "antenna_file_format": "radiosim",
            },
            "obs_frequency": {
                "starting_frequency": 100,
                "frequency_interval": 50,
                "frequency_bandwidth": 50,
                "frequency_unit": "MHz",
            },
            "obs_time": {"start_time": "2025-01-01T00:00:00"},
            "sky_model": {
                "flux_unit": "Jy",
                "sources": [{"kind": "test_sources", "num_sources": 2, "seed": 1}],
            },
            "visibility": {"sky_representation": "point_sources"},
        }
    )

    sim.setup()

    assert sim._sky_model is not None
    assert sim._sky_model.point is not None
