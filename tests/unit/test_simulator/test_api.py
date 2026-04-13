"""Tests for the public Simulator constructor contract."""

import pytest

from rrivis.api import Simulator


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
