"""Characterization guards for the canonical MS and observability boundaries."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from astropy.constants import c

from radiosim.api import Simulator
from radiosim.core.observability import ObservabilityPlanner
from radiosim.io.measurement_set import write_ms
from tests.fixtures.configs import resolved_config


def test_write_ms_exposes_only_canonical_instrument_inputs():
    parameters = inspect.signature(write_ms).parameters

    assert "instrument" in parameters
    assert "selection" in parameters
    for removed in (
        "antennas",
        "baselines",
        "location",
        "telescope_name",
        "phase_center_ra",
        "phase_center_dec",
    ):
        assert removed not in parameters


def test_observability_field_radius_requires_explicit_positive_diameter():
    with pytest.raises(ValueError, match="beam_diameter_m is required"):
        ObservabilityPlanner(frequency_mhz=150.0)._resolve_field_radius_deg()

    planner = ObservabilityPlanner(frequency_mhz=150.0, beam_diameter_m=28.0)
    radius = planner._resolve_field_radius_deg()
    expected = np.degrees(1.22 * (c.value / 150e6) / 28.0) / 2.0
    assert radius == pytest.approx(expected)


def test_simulator_observability_uses_exact_uniform_state_before_setup(
    tmp_path,
    monkeypatch,
):
    captured: dict[str, object] = {}

    class FakePlanner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def build(self):
            return object()

    class FakeRenderer:
        def __init__(self, _plan, **_kwargs):
            pass

        def create_plot(self):
            return "layout"

        def save(self, *_args, **_kwargs):
            raise AssertionError("save should not be called")

    monkeypatch.setattr(
        "radiosim.core.observability.ObservabilityPlanner",
        FakePlanner,
    )
    monkeypatch.setattr(
        "radiosim.visualization.observability.ObservabilityBokehRenderer",
        FakeRenderer,
    )
    simulator = Simulator(
        resolved_config(
            tmp_path,
            instrument={
                "diameter_overrides": [
                    {
                        "antenna": {"kind": "number", "number": 0},
                        "diameter_m": 31.0,
                    },
                    {
                        "antenna": {"kind": "number", "number": 1},
                        "diameter_m": 31.0,
                    },
                ]
            },
        ).runtime
    )

    layout = simulator.plot_observability(
        lst_start_hours=1.0,
        lst_end_hours=2.0,
        open_in_browser=False,
    )

    assert layout == "layout"
    assert len(simulator.antennas) == 2
    assert simulator._backend is None
    assert captured["beam_diameter_m"] == 31.0
    assert isinstance(captured["beam_diameter_m"], float)
