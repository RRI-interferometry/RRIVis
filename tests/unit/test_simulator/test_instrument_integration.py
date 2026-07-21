"""End-to-end contracts for the Tier 2G authoritative instrument state."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
from astropy.constants import c
from astropy.coordinates import EarthLocation
from astropy.time import Time

import radiosim.core.visibility as visibility_module
import radiosim.core.visibility_healpix as healpix_visibility_module
from radiosim.api.simulator import Simulator
from radiosim.backends import get_backend
from radiosim.core.instrument_adapters import SolverInstrumentView
from radiosim.core.instrument_resolution import DiameterResolutionError
from radiosim.core.observability.planner import (
    HeterogeneousObservabilityUnsupportedError,
)
from radiosim.core.visibility import calculate_visibility
from radiosim.core.visibility_healpix import calculate_visibility_healpix
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
        "beams": {
            "mode": "analytic",
            "model": {
                "kind": "circular_aperture",
                "taper": {"kind": "gaussian", "edge_taper_db": 10.0},
            },
        },
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


def _one_metre_solver_view(tmp_path) -> SolverInstrumentView:
    mapping = _instrument_mapping(tmp_path)
    mapping["instrument"]["source"]["path"] = str(tmp_path / "one-metre.txt")
    (tmp_path / "one-metre.txt").write_text(
        "Name Number BeamID E N U Diameter\n"
        "ANT1 1 0 0.0 0.0 0.0 12.0\n"
        "ANT2 2 0 1.0 0.0 0.0 25.0\n",
        encoding="utf-8",
    )
    simulator = Simulator.from_mapping(mapping, base_dir=tmp_path)
    simulator._ensure_instrument_state()
    return SolverInstrumentView.from_state(simulator._instrument_state)


def _point_source_arrays(frequency_hz: float) -> dict[str, object]:
    zeros = np.zeros(1, dtype=np.float64)
    return {
        "ra_rad": zeros.copy(),
        "dec_rad": zeros.copy(),
        "flux": np.array([2.0]),
        "spectral_index": zeros.copy(),
        "stokes_q": zeros.copy(),
        "stokes_u": zeros.copy(),
        "stokes_v": zeros.copy(),
        "ref_freq": np.array([frequency_hz]),
        "rotation_measure": zeros.copy(),
        "spectral_coeffs": None,
        "per_channel_flux": None,
        "per_channel_stokes_q": None,
        "per_channel_stokes_u": None,
        "per_channel_stokes_v": None,
        "channel_frequencies": None,
        "major_arcsec": zeros.copy(),
        "minor_arcsec": zeros.copy(),
        "pa_deg": zeros.copy(),
    }


class _FixedAltAzSkyCoord:
    def __init__(self, **_kwargs):
        pass

    def transform_to(self, _frame):
        return SimpleNamespace(
            az=SimpleNamespace(rad=np.array([np.pi / 2.0])),
            alt=SimpleNamespace(rad=np.array([np.pi / 3.0])),
        )


class _IdentityJonesChain:
    def compute_antenna_jones_all_sources(self, *, n_sources, **_kwargs):
        return np.broadcast_to(np.eye(2, dtype=np.complex128), (n_sources, 2, 2))


class _FixedPixelCoordinates:
    def __len__(self):
        return 1

    def transform_to(self, _frame):
        return SimpleNamespace(
            az=SimpleNamespace(rad=np.array([np.pi / 2.0])),
            alt=SimpleNamespace(rad=np.array([np.pi / 3.0])),
        )


class _OnePixelHealpix:
    nside = 1
    pixel_solid_angle = 1.0
    pixel_coords = _FixedPixelCoordinates()

    @staticmethod
    def get_map_at_frequency(_frequency):
        return np.array([1.0])


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


def test_layout_visualizers_consume_canonical_identity_positions_and_diameters(
    tmp_path,
    monkeypatch,
):
    from radiosim.visualization.bokeh_plots import (
        plot_antenna_layout,
        plot_antenna_layout_3d_plotly,
    )

    simulator = Simulator.from_mapping(
        _instrument_mapping(tmp_path),
        base_dir=tmp_path,
    )
    simulator._ensure_instrument_state()

    figure_2d = plot_antenna_layout(simulator.antennas, open_in_browser=False)
    assert figure_2d.renderers[0].data_source.data == {
        "E": [0.0, 14.0],
        "N": [0.0, 0.0],
        "Number": ["0", "1"],
        "Name": ["ANT0", "ANT1"],
    }

    import plotly.io as pio

    captured: dict[str, object] = {}

    def capture_html(figure, **_kwargs):
        captured["figure"] = figure
        return "<div>captured</div>"

    monkeypatch.setattr(pio, "to_html", capture_html)
    output = plot_antenna_layout_3d_plotly(
        simulator.antennas,
        save_simulation_data=True,
        folder_path=str(tmp_path),
        open_in_browser=False,
    )

    assert Path(output) == tmp_path / "antenna_layout_3d.html"
    ring_traces = [
        trace
        for trace in captured["figure"].data
        if getattr(trace, "mode", None) == "lines"
        and getattr(trace.line, "color", None) == "#1f2a44"
    ]
    assert len(ring_traces) == 1
    assert tuple(ring_traces[0].x).count(None) == 2
    ring_x = [value for value in ring_traces[0].x if value is not None]
    assert min(ring_x) == pytest.approx(-6.0)
    assert max(ring_x) == pytest.approx(26.5)


def test_point_and_healpix_keep_canonical_negative_phase_sign(tmp_path, monkeypatch):
    monkeypatch.setattr(visibility_module, "SkyCoord", _FixedAltAzSkyCoord)
    monkeypatch.setattr(
        visibility_module,
        "_build_jones_chain",
        lambda *_args, **_kwargs: _IdentityJonesChain(),
    )
    monkeypatch.setattr(
        healpix_visibility_module,
        "_compute_beam_power_pattern",
        lambda *, zenith_angles, **_kwargs: np.ones_like(zenith_angles),
    )
    wavelength_m = 2.0
    frequency_hz = c.value / wavelength_m
    instrument = _one_metre_solver_view(tmp_path)
    location = EarthLocation.from_geodetic(0.0, 0.0, 0.0)
    obstime = Time("2024-01-01T00:00:00")
    wavelengths = np.array([wavelength_m]) * u.m
    frequencies = np.array([frequency_hz])

    point_result = calculate_visibility(
        instrument=instrument,
        source_arrays=_point_source_arrays(frequency_hz),
        location=location,
        obstime=obstime,
        wavelengths=wavelengths,
        freqs=frequencies,
        duration_seconds=1.0,
        time_step_seconds=1.0,
        return_correlations=False,
        backend=get_backend("numpy"),
    )
    point_matrix = point_result[(1, 2)][0, 0]

    healpix_result = calculate_visibility_healpix(
        sky_model=SimpleNamespace(
            healpix=_OnePixelHealpix(),
            has_polarized_healpix_maps=False,
            brightness_conversion="rayleigh-jeans",
            model_name="one-pixel-phase-sign",
        ),
        instrument=instrument,
        location=location,
        obstime=obstime,
        wavelengths=wavelengths,
        freqs=frequencies,
        duration_seconds=1.0,
        time_step_seconds=1.0,
        output_units="K.sr",
        backend=get_backend("numpy"),
    )

    assert point_matrix[0, 0] == pytest.approx(-1j)
    assert point_matrix[1, 1] == pytest.approx(-1j)
    assert point_matrix[0, 1] == 0.0
    assert healpix_result["visibilities"][0, 0, 0] == pytest.approx(-1j)


def test_memory_estimation_uses_canonical_selected_inventory_counts(tmp_path):
    mapping = _instrument_mapping(tmp_path)
    mapping["baseline_selection"] = {"correlations": "all"}
    simulator = Simulator.from_mapping(mapping, base_dir=tmp_path)
    simulator._ensure_instrument_state()
    captured: dict[str, int] = {}

    class FakeSimulatorStrategy:
        def get_memory_estimate(self, **kwargs):
            captured.update(kwargs)
            return {"total_bytes": 1000, "total_human": "1.0 KB"}

    simulator._is_setup = True
    simulator._source_arrays = {"ra_rad": np.zeros(4)}
    simulator._frequencies_hz = np.zeros(5)
    simulator._simulator = FakeSimulatorStrategy()
    simulator._backend = SimpleNamespace(precision=None)

    estimate = simulator.get_memory_estimate()

    assert captured == {
        "n_antennas": 2,
        "n_baselines": 3,
        "n_sources": 4,
        "n_frequencies": 5,
    }
    assert estimate["precision_factor"] == 1.0


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


def test_observability_uses_exact_uniform_canonical_diameter_before_setup(
    tmp_path,
    monkeypatch,
):
    mapping = _instrument_mapping(tmp_path)
    antenna_path = Path(mapping["instrument"]["source"]["path"])
    antenna_path.write_text(
        "Name Number BeamID E N U Diameter\n"
        "ANT0 0 0 0.0 0.0 0.0 31.0\n"
        "ANT1 1 0 14.0 0.0 0.0 31.0\n",
        encoding="utf-8",
    )
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
    simulator = Simulator.from_mapping(mapping, base_dir=tmp_path)

    layout = simulator.plot_observability(
        lst_start_hours=1.0,
        lst_end_hours=2.0,
        open_in_browser=False,
    )

    assert layout == "layout"
    assert simulator._backend is None
    assert captured["beam_diameter_m"] == 31.0
    assert type(captured["beam_diameter_m"]) is float
