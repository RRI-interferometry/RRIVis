"""Tier 4G contracts for canonical-result visibility renderers."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from radiosim.api import Simulator
from radiosim.core.result import ResultUnavailableError, SimulationResult
from radiosim.io.result_errors import OutputCollisionError, OutputPathError
from radiosim.visualization.errors import (
    ResultBrowserError,
    ResultPlotContractError,
)
from tests.fixtures.configs import valid_config_mapping


def _mapping(tmp_path: Path) -> dict[str, object]:
    return valid_config_mapping(
        tmp_path,
        obs_time={
            "start_time": "2025-01-01T00:00:00",
            "duration_seconds": 3.0,
            "time_step_seconds": 1.0,
        },
        frequency={
            "mode": "explicit",
            "channel_frequencies_hz": [100e6, 101.5e6, 103e6],
            "channel_widths_hz": [1e6, 0.5e6, 1e6],
        },
        sky_model={"sources": [{"kind": "test_sources", "num_sources": 2, "seed": 5}]},
    )


@pytest.fixture(scope="module")
def canonical_result(tmp_path_factory) -> SimulationResult:
    tmp_path = tmp_path_factory.mktemp("result_plots")
    simulator = Simulator.from_mapping(_mapping(tmp_path), base_dir=tmp_path)
    return simulator.run(progress=False)


@pytest.fixture
def plotted_simulator(tmp_path) -> Simulator:
    simulator = Simulator.from_mapping(_mapping(tmp_path), base_dir=tmp_path)
    simulator.run(progress=False)
    return simulator


def _line_sources(document) -> list[dict[str, object]]:
    from bokeh.models import GlyphRenderer

    sources: list[dict[str, object]] = []
    for model in document.select({"type": GlyphRenderer}):
        data = model.data_source.data
        if "x" in data and "y" in data:
            sources.append(dict(data))
    return sources


def _forbid_browser(monkeypatch) -> list[str]:
    import webbrowser

    opened: list[str] = []

    def record(url, *_args, **_kwargs):
        opened.append(url)
        return True

    monkeypatch.setattr(webbrowser, "open", record)
    return opened


def test_visibility_renderer_uses_exact_canonical_time_coordinates(
    canonical_result,
    monkeypatch,
):
    from radiosim.visualization.bokeh_plots import plot_visibility

    opened = _forbid_browser(monkeypatch)
    document = plot_visibility(canonical_result)

    expected_time = canonical_result.time_grid.to_mjd()
    stokes = canonical_result.stokes_i()
    sources = _line_sources(document)

    assert sources
    for data in sources:
        np.testing.assert_array_equal(np.asarray(data["x"]), expected_time)
    modulus = np.abs(stokes[:, 0, 0])
    assert any(np.array_equal(np.asarray(data["y"]), modulus) for data in sources), (
        "first baseline Stokes I modulus is not plotted from the canonical result"
    )
    assert opened == []


def test_visibility_renderer_derives_stokes_i_and_honours_the_phase_unit(
    canonical_result,
):
    from radiosim.visualization.bokeh_plots import plot_visibility

    stokes = canonical_result.stokes_i()
    np.testing.assert_array_equal(
        stokes,
        canonical_result.visibilities[..., 0] + canonical_result.visibilities[..., 3],
    )
    radian_phase = np.unwrap(np.angle(stokes[:, 0, 0]))

    radians = _line_sources(plot_visibility(canonical_result))
    degrees = _line_sources(
        plot_visibility(canonical_result, visibility_phase_unit="degrees")
    )

    assert any(np.array_equal(np.asarray(data["y"]), radian_phase) for data in radians)
    assert any(
        np.array_equal(np.asarray(data["y"]), np.degrees(radian_phase))
        for data in degrees
    )


def test_frequency_renderer_uses_exact_canonical_frequency_coordinates(
    canonical_result,
):
    from radiosim.visualization.bokeh_plots import plot_modulus_vs_frequency

    document = plot_modulus_vs_frequency(canonical_result)
    sources = _line_sources(document)

    assert sources
    for data in sources:
        np.testing.assert_array_equal(
            np.asarray(data["x"]),
            canonical_result.frequencies_hz,
        )
    assert not any(
        np.array_equal(
            np.asarray(data["x"]),
            canonical_result.frequencies_hz / 1e6,
        )
        for data in sources
    )


def test_heatmap_renderer_uses_exact_canonical_coordinate_extents(canonical_result):
    from bokeh.models import GlyphRenderer

    from radiosim.visualization.bokeh_plots import plot_heatmaps

    document = plot_heatmaps(canonical_result)
    times = canonical_result.time_grid.to_mjd()
    frequencies = canonical_result.frequencies_hz
    images = [
        model.data_source.data
        for model in document.select({"type": GlyphRenderer})
        if "image" in model.data_source.data
    ]

    assert images
    for data in images:
        assert data["x"] == [float(times[0])]
        assert data["dw"] == [float(times[-1] - times[0])]
        assert data["y"] == [float(frequencies[0])]
        assert data["dh"] == [float(frequencies[-1] - frequencies[0])]


@pytest.mark.parametrize(
    "renderer_name",
    ["plot_visibility", "plot_heatmaps", "plot_modulus_vs_frequency"],
)
def test_visibility_renderers_reject_non_canonical_input(
    canonical_result,
    renderer_name,
):
    import radiosim.visualization.bokeh_plots as renderers

    renderer = getattr(renderers, renderer_name)
    parameters = inspect.signature(renderer).parameters

    assert list(parameters)[0] == "result"
    assert "moduli_over_time" not in parameters
    assert "phases_over_time" not in parameters
    assert "total_seconds" not in parameters
    assert "mjd_time_points" not in parameters
    assert "angle_unit" not in parameters
    for name, parameter in parameters.items():
        if name != "result":
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY

    with pytest.raises(ResultPlotContractError):
        renderer({"visibilities": canonical_result.visibilities})
    with pytest.raises(ResultPlotContractError):
        renderer(None)


def test_visibility_renderer_rejects_an_unknown_phase_unit(canonical_result):
    from radiosim.visualization.bokeh_plots import plot_visibility

    with pytest.raises(ResultPlotContractError):
        plot_visibility(canonical_result, visibility_phase_unit="gradians")


def test_plot_signature_matches_the_canonical_contract():
    parameters = inspect.signature(Simulator.plot).parameters

    assert list(parameters) == [
        "self",
        "plot_type",
        "output_dir",
        "backend",
        "show",
        "overwrite",
        "visibility_phase_unit",
    ]
    for name, parameter in parameters.items():
        if name != "self":
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["plot_type"].default == "all"
    assert parameters["output_dir"].default is None
    assert parameters["backend"].default == "bokeh"
    assert parameters["show"].default is True
    assert parameters["overwrite"].default is False
    assert parameters["visibility_phase_unit"].default == "radians"


def test_plot_requires_a_published_result_without_side_effects(tmp_path):
    simulator = Simulator.from_mapping(_mapping(tmp_path), base_dir=tmp_path)
    output = tmp_path / "plots"

    with pytest.raises(ResultUnavailableError):
        simulator.plot(output_dir=output, show=False)

    assert not output.exists()


@pytest.mark.parametrize(
    "arguments",
    [
        {"plot_type": "hostile"},
        {"backend": "hostile"},
        {"backend": "matplotlib"},
        {"visibility_phase_unit": "hostile"},
        {"show": "yes"},
        {"overwrite": "yes"},
    ],
)
def test_plot_contract_failures_precede_any_output(
    plotted_simulator,
    tmp_path,
    arguments,
):
    output = tmp_path / "contract-plots"
    request: dict[str, object] = {"output_dir": output, "show": False}
    request.update(arguments)

    with pytest.raises(ResultPlotContractError):
        plotted_simulator.plot(**request)

    assert not output.exists()


def test_plot_requires_an_explicit_output_directory(plotted_simulator):
    with pytest.raises(OutputPathError):
        plotted_simulator.plot(show=False)


def test_plot_writes_declared_files_and_opens_browsers_last(
    plotted_simulator,
    tmp_path,
    monkeypatch,
):
    import webbrowser

    output = tmp_path / "plots"
    observed: list[tuple[str, bool]] = []

    def record(url, *_args, **_kwargs):
        observed.append((url, output.is_dir()))
        return True

    monkeypatch.setattr(webbrowser, "open", record)

    written = plotted_simulator.plot(output_dir=output, show=True)

    assert type(written) is tuple
    assert [path.name for path in written] == [
        "antenna_layout.html",
        "visibility-phase-lsts.html",
        "heatmaps-freq-time.html",
        "modulus-phase-freq.html",
    ]
    for path in written:
        assert path.parent == output
        assert path.is_file()
        assert path.stat().st_size > 0
    assert [url for url, _ in observed] == [path.as_uri() for path in written]
    assert all(existed for _url, existed in observed)


def test_plot_selects_one_family_and_refuses_silent_replacement(
    plotted_simulator,
    tmp_path,
    monkeypatch,
):
    _forbid_browser(monkeypatch)
    output = tmp_path / "plots"

    written = plotted_simulator.plot(
        plot_type="visibility",
        output_dir=output,
        show=False,
    )

    assert [path.name for path in written] == ["visibility-phase-lsts.html"]
    original = written[0].read_bytes()

    with pytest.raises(OutputCollisionError):
        plotted_simulator.plot(
            plot_type="visibility",
            output_dir=output,
            show=False,
        )

    assert written[0].read_bytes() == original
    replaced = plotted_simulator.plot(
        plot_type="visibility",
        output_dir=output,
        show=False,
        overwrite=True,
    )
    assert replaced == written


def test_plot_reports_browser_failure_after_publication(
    plotted_simulator,
    tmp_path,
    monkeypatch,
):
    import webbrowser

    output = tmp_path / "plots"

    def fail(*_args, **_kwargs):
        raise OSError("no browser")

    monkeypatch.setattr(webbrowser, "open", fail)

    with pytest.raises(ResultBrowserError):
        plotted_simulator.plot(
            plot_type="visibility",
            output_dir=output,
            show=True,
        )

    assert (output / "visibility-phase-lsts.html").is_file()
