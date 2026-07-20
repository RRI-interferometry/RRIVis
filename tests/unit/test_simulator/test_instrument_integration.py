"""End-to-end contracts for the Tier 2G authoritative instrument state."""

from __future__ import annotations

import inspect

import pytest

from radiosim.api.simulator import Simulator
from radiosim.core.instrument_resolution import DiameterResolutionError
from radiosim.core.observability.planner import (
    HeterogeneousObservabilityUnsupportedError,
)
from radiosim.io.config import RadioSimConfig
from radiosim.io.instrument_config import (
    BaselineSelectionConfig,
    InstrumentConfig,
)


def _instrument_mapping(tmp_path, *, include_diameters: bool = True):
    antenna_path = tmp_path / "antennas.txt"
    diameter_column = " Diameter" if include_diameters else ""
    first_diameter = " 12.0" if include_diameters else ""
    second_diameter = " 25.0" if include_diameters else ""
    antenna_path.write_text(
        f"Name Number BeamID E N U{diameter_column}\n"
        f"ANT0 0 0 0.0 0.0 0.0{first_diameter}\n"
        f"ANT1 1 0 14.0 0.0 0.0{second_diameter}\n",
        encoding="utf-8",
    )
    return {
        "instrument": {
            "source": {
                "kind": "layout_file",
                "path": str(antenna_path),
                "format": "radiosim",
                "telescope_name": "Tier2G Array",
            },
            "location": {
                "longitude_deg": 21.4283,
                "latitude_deg": -30.72152,
                "height_m": 1073.0,
            },
        },
        "baseline_selection": {"correlations": "cross"},
        "beams": {"beam_mode": "analytic"},
        "obs_time": {
            "start_time": "2025-01-01T00:00:00",
            "duration_seconds": 1.0,
            "time_step_seconds": 1.0,
        },
        "obs_frequency": {
            "mode": "explicit",
            "channel_frequencies_hz": [100_000_000.0],
        },
        "sky_model": {
            "sources": [{"kind": "test_sources", "num_sources": 1, "seed": 7}]
        },
        "execution": {"backend": "numpy", "offline": True},
    }


def test_active_schema_owns_typed_instrument_and_selection(tmp_path):
    config = RadioSimConfig.model_validate(_instrument_mapping(tmp_path))

    assert tuple(config.model_fields) == (
        "instrument",
        "beams",
        "baseline_selection",
        "sky_model",
        "obs_time",
        "obs_frequency",
        "visibility",
        "execution",
        "workflow",
    )
    assert type(config.instrument) is InstrumentConfig
    assert type(config.baseline_selection) is BaselineSelectionConfig


def test_from_parameters_accepts_only_typed_instrument_inputs():
    parameters = inspect.signature(Simulator.from_parameters).parameters

    assert "instrument" in parameters
    assert "baseline_selection" in parameters
    for removed in (
        "antenna_layout",
        "antenna_file_format",
        "antenna_diameter_m",
        "location",
    ):
        assert removed not in parameters


def test_public_instrument_properties_fail_consistently_before_resolution(tmp_path):
    simulator = Simulator.from_mapping(_instrument_mapping(tmp_path), base_dir=tmp_path)

    for property_name in ("instrument", "antennas", "baselines"):
        with pytest.raises(
            RuntimeError,
            match="^Instrument resolution has not completed$",
        ):
            getattr(simulator, property_name)


def test_instrument_only_resolution_is_atomic_immutable_and_selected(tmp_path):
    simulator = Simulator.from_mapping(_instrument_mapping(tmp_path), base_dir=tmp_path)

    simulator._ensure_instrument_state()

    assert simulator.instrument.name == "Tier2G Array"
    assert simulator.antennas is simulator.instrument.antennas
    assert simulator.antennas is simulator.antennas
    assert tuple(antenna.diameter_m for antenna in simulator.antennas) == (12.0, 25.0)
    assert simulator.baselines is simulator.baselines
    assert tuple(
        (baseline.ant1.number, baseline.ant2.number) for baseline in simulator.baselines
    ) == ((0, 1),)
    assert simulator.baselines[0].vector_enu_m == (14.0, 0.0, 0.0)


def test_instrument_failure_precedes_backend_and_assigns_no_partial_state(
    tmp_path,
    monkeypatch,
):
    calls: list[str] = []

    def forbidden_device():
        calls.append("device")
        pytest.fail("device detection ran before instrument resolution")

    monkeypatch.setattr(
        "radiosim.utils.device.get_device_resources",
        forbidden_device,
    )
    simulator = Simulator.from_mapping(
        _instrument_mapping(tmp_path, include_diameters=False),
        base_dir=tmp_path,
    )

    with pytest.raises(DiameterResolutionError, match="incomplete antenna diameters"):
        simulator.setup()

    assert calls == []
    assert simulator._instrument_state is None
    assert simulator._is_setup is False


def test_later_setup_failure_retains_exact_instrument_state_for_retry(
    tmp_path,
    monkeypatch,
):
    simulator = Simulator.from_mapping(_instrument_mapping(tmp_path), base_dir=tmp_path)
    calls = 0

    def fail_device():
        nonlocal calls
        calls += 1
        raise RuntimeError("device unavailable")

    monkeypatch.setattr("radiosim.utils.device.get_device_resources", fail_device)

    with pytest.raises(RuntimeError, match="device unavailable"):
        simulator.setup()
    retained = simulator.instrument

    with pytest.raises(RuntimeError, match="device unavailable"):
        simulator.setup()

    assert simulator.instrument is retained
    assert calls == 2
    assert simulator._is_setup is False


def test_observability_rejects_heterogeneous_diameters_before_side_effects(
    tmp_path,
    monkeypatch,
):
    simulator = Simulator.from_mapping(_instrument_mapping(tmp_path), base_dir=tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("observability side effect occurred before diameter rejection")

    monkeypatch.setattr("radiosim.utils.device.get_device_resources", forbidden)
    monkeypatch.setattr("webbrowser.open", forbidden)

    with pytest.raises(
        HeterogeneousObservabilityUnsupportedError,
        match=r"12\.0.*25\.0.*Tier 3",
    ):
        simulator.plot_observability(open_in_browser=True)

    assert simulator._instrument_state is not None
    assert simulator._backend is None
