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


def test_public_instrument_properties_have_exact_return_annotations():
    assert inspect.signature(Simulator.instrument.fget).return_annotation == (
        "ResolvedInstrument"
    )
    assert inspect.signature(Simulator.antennas.fget).return_annotation == (
        "tuple[ResolvedAntenna, ...]"
    )
    assert inspect.signature(Simulator.baselines.fget).return_annotation == (
        "tuple[ResolvedBaseline, ...]"
    )


def test_simulator_exact_type_boundaries_reject_mutable_subclasses(tmp_path):
    from pydantic import ConfigDict

    from radiosim.core.runtime_config import ResolvedSimulationConfig

    class MutableRadioSimConfig(RadioSimConfig):
        model_config = ConfigDict(extra="forbid", frozen=False)

    class MutableResolvedSimulationConfig(ResolvedSimulationConfig):
        pass

    mapping = _instrument_mapping(tmp_path)
    input_subclass = MutableRadioSimConfig.model_validate(mapping)
    with pytest.raises(TypeError, match="only RadioSimConfig"):
        Simulator.from_config(input_subclass, base_dir=tmp_path)

    resolved = Simulator.from_mapping(mapping, base_dir=tmp_path).config
    runtime_subclass = MutableResolvedSimulationConfig(
        **{
            name: getattr(resolved, name)
            for name in ResolvedSimulationConfig.__dataclass_fields__
        }
    )
    with pytest.raises(TypeError, match="only ResolvedSimulationConfig"):
        Simulator(runtime_subclass)


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


def test_instrument_failure_reloads_source_on_successful_retry(tmp_path, monkeypatch):
    import radiosim.core.instrument_resolution as resolution_module

    real_resolve = resolution_module.resolve_instrument
    calls = 0

    def fail_once(config):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise DiameterResolutionError("transient instrument failure")
        return real_resolve(config)

    monkeypatch.setattr(resolution_module, "resolve_instrument", fail_once)
    simulator = Simulator.from_mapping(_instrument_mapping(tmp_path), base_dir=tmp_path)

    with pytest.raises(DiameterResolutionError, match="transient"):
        simulator._ensure_instrument_state()
    assert simulator._instrument_state is None

    simulator._ensure_instrument_state()
    assert calls == 2
    assert simulator.instrument.name == "Tier2G Array"


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


def test_sky_failure_retry_reuses_instrument_and_recreates_backend(
    tmp_path,
    monkeypatch,
):
    import radiosim.backends as backends_module
    import radiosim.core.sky.combine.pipeline as pipeline_module

    real_get_backend = backends_module.get_backend
    real_prepare = pipeline_module.prepare_sky_model
    backend_instances = []
    prepare_calls = 0

    def record_backend(*args, **kwargs):
        backend = real_get_backend(*args, **kwargs)
        backend_instances.append(backend)
        return backend

    def fail_prepare_once(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        if prepare_calls == 1:
            raise RuntimeError("sky preparation failed")
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(backends_module, "get_backend", record_backend)
    monkeypatch.setattr(pipeline_module, "prepare_sky_model", fail_prepare_once)
    simulator = Simulator.from_mapping(_instrument_mapping(tmp_path), base_dir=tmp_path)

    with pytest.raises(RuntimeError, match="sky preparation failed"):
        simulator.setup()
    retained = simulator.instrument
    assert simulator._backend is None
    assert simulator._sky_model is None
    assert simulator._is_setup is False

    simulator.setup()
    assert simulator.instrument is retained
    assert len(backend_instances) == 2
    assert backend_instances[0] is not backend_instances[1]
    assert simulator._is_setup is True


def test_run_does_not_print_banner_before_instrument_success(tmp_path, monkeypatch):
    banners: list[tuple[object, ...]] = []
    simulator = Simulator.from_mapping(
        _instrument_mapping(tmp_path, include_diameters=False),
        base_dir=tmp_path,
    )
    monkeypatch.setattr(
        "radiosim.api.simulator.print_header",
        lambda *args, **kwargs: banners.append(args),
    )

    with pytest.raises(DiameterResolutionError):
        simulator.run(progress=True)

    assert banners == []


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
