"""Tests for Bokeh observability rendering."""

import inspect
from pathlib import Path

import numpy as np
import pytest
from bokeh.layouts import Column
from bokeh.models import DataTable, Div, GridPlot

from radiosim.api.simulator import Simulator
from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky import create_from_arrays
from radiosim.visualization.observability import ObservabilityBokehRenderer


def _point_sky():
    zeros = np.zeros(4)
    return create_from_arrays(
        ra_rad=np.deg2rad([30.0, 45.0, 60.0, 120.0]),
        dec_rad=np.deg2rad([-30.0, -28.0, -32.0, -30.0]),
        flux=np.array([10.0, 5.0, 8.0, 1.0]),
        spectral_index=zeros,
        stokes_q=zeros,
        stokes_u=zeros,
        stokes_v=zeros,
        model_name="test_points",
        brightness_conversion="planck",
        precision=PrecisionConfig.standard(),
    )


class TestObservabilityBokehRenderer:
    def test_summary_layout_contains_tables(self, tmp_path):
        plan = _tier3g_plan(tmp_path, include_sources=True)

        layout = ObservabilityBokehRenderer(
            plan,
            show_source_colorbar=True,
        ).create_plot()

        assert isinstance(layout, Column)
        assert len(list(layout.select({"type": DataTable}))) >= 1
        assert len(list(layout.select({"type": Div}))) >= 1

    def test_source_table_headings_show_canonical_reference_identity(self, tmp_path):
        plan = _tier3g_plan(tmp_path, include_sources=True)

        layout = ObservabilityBokehRenderer(plan).create_plot()

        headings = [
            div.text for div in layout.select({"type": Div}) if "<h3" in div.text
        ]
        expected = (
            f"ref {plan.reference_antenna.number}:{plan.reference_antenna.name} "
            f"{plan.reference_scientific_fingerprint[:12]} "
            f"({plan.reference_selection_reason})"
        )
        assert headings
        assert all(expected in heading for heading in headings)

    def test_lst_axis_uses_wrapped_hour_range(self, tmp_path):
        plan = _tier3g_plan(tmp_path, x_axis="lst")

        figure = ObservabilityBokehRenderer(plan).create_plot()

        assert figure.x_range.start == -12
        assert figure.x_range.end == 12

    def test_snapshot_mode_returns_gridplot(self, tmp_path):
        plan = _tier3g_plan(tmp_path, mode="snapshots")

        layout = ObservabilityBokehRenderer(plan).create_plot()

        assert isinstance(layout, GridPlot)
        assert len(layout.children) >= 1


def _tier3g_plan(
    tmp_path: Path,
    *,
    include_sources: bool = False,
    x_axis: str = "ra",
    mode: str = "summary",
):
    antenna_path = tmp_path / "renderer-antennas.txt"
    antenna_path.write_text(
        "Name Number BeamID E N U Diameter\n"
        "ANT0 0 0 0.0 0.0 0.0 14.0\n"
        "ANT1 1 0 14.0 0.0 0.0 14.0\n",
        encoding="utf-8",
    )
    simulator = Simulator.from_mapping(
        {
            "instrument": {
                "source": {
                    "kind": "layout_file",
                    "path": str(antenna_path),
                    "format": "radiosim",
                    "telescope_name": "Renderer Array",
                },
                "location": {
                    "longitude_deg": 21.0,
                    "latitude_deg": -30.0,
                    "height_m": 1000.0,
                },
            },
            "baseline_selection": {"correlations": "cross"},
            "beams": {
                "mode": "analytic",
                "model": {
                    "kind": "circular_aperture",
                    "taper": {"kind": "uniform"},
                },
            },
            "obs_time": {
                "start_time": "2025-01-01T00:00:00",
                "duration_seconds": 1.0,
                "time_step_seconds": 1.0,
            },
            "obs_frequency": {
                "mode": "explicit",
                "channel_frequencies_hz": [150_000_000.0],
            },
            "sky_model": {
                "sources": [{"kind": "test_sources", "num_sources": 1, "seed": 1}]
            },
            "execution": {"backend": "numpy", "offline": True},
        },
        base_dir=tmp_path,
    )
    if include_sources:
        simulator._sky_model = _point_sky()
    return simulator.plan_observability(
        grid_resolution_deg=10.0,
        include_source_metrics=include_sources,
        x_axis=x_axis,
        mode=mode,
    )


class TestTier3GRendererPersistence:
    def test_exact_constructor_and_save_signatures(self):
        constructor = inspect.signature(ObservabilityBokehRenderer).parameters
        save_parameters = inspect.signature(ObservabilityBokehRenderer.save).parameters

        assert tuple(constructor) == ("plan", "show_source_colorbar", "color_scale")
        assert tuple(save_parameters) == (
            "self",
            "layout",
            "output_dir",
            "filename",
            "overwrite",
            "open_in_browser",
        )

    def test_invalid_render_option_precedes_output_side_effect(
        self,
        tmp_path,
        monkeypatch,
    ):
        import radiosim.core.observability as observability

        plan = _tier3g_plan(tmp_path)
        missing = tmp_path / "must-not-exist"

        def forbidden(*_args, **_kwargs):
            pytest.fail("invalid render input reached output work")

        monkeypatch.setattr(Path, "mkdir", forbidden)
        with pytest.raises(observability.ObservabilityRenderError):
            ObservabilityBokehRenderer(plan, color_scale="invalid")
        assert not missing.exists()

    @pytest.mark.parametrize(
        "filename",
        ["", " ", "../plot.html", "nested/plot.html", "plot.txt", ".html"],
    )
    def test_invalid_filename_creates_no_temporary_file(
        self,
        tmp_path,
        filename,
    ):
        import radiosim.core.observability as observability

        renderer = ObservabilityBokehRenderer(_tier3g_plan(tmp_path))
        layout = renderer.create_plot()
        before = tuple(tmp_path.iterdir())

        with pytest.raises(observability.ObservabilityOutputError):
            renderer.save(
                layout,
                output_dir=tmp_path,
                filename=filename,
            )

        assert tuple(tmp_path.iterdir()) == before

    def test_collision_does_not_overwrite_and_has_no_cause(self, tmp_path):
        import radiosim.core.observability as observability

        target = tmp_path / "observability.html"
        target.write_text("existing", encoding="utf-8")
        renderer = ObservabilityBokehRenderer(_tier3g_plan(tmp_path))
        layout = renderer.create_plot()

        with pytest.raises(observability.ObservabilityOutputCollisionError) as caught:
            renderer.save(
                layout,
                output_dir=tmp_path,
                filename=target.name,
            )

        assert caught.value.__cause__ is None
        assert target.read_text(encoding="utf-8") == "existing"
        assert not tuple(tmp_path.glob(".observability.html.*"))

    def test_render_publish_browser_side_effect_order(
        self,
        tmp_path,
        monkeypatch,
    ):
        import radiosim.visualization.observability.bokeh_renderer as renderer_module

        events: list[str] = []
        renderer = ObservabilityBokehRenderer(_tier3g_plan(tmp_path))
        layout = renderer.create_plot()

        def fake_save(_layout, *, filename, resources, title):
            del resources, title
            events.append("render")
            Path(filename).write_text("rendered", encoding="utf-8")

        real_link = renderer_module.os.link

        def recorded_link(source, target):
            events.append("publish")
            return real_link(source, target)

        def recorded_browser(uri):
            events.append("browser")
            assert (tmp_path / "ordered.html").exists()
            assert uri.startswith("file:")
            return True

        monkeypatch.setattr(renderer_module, "save", fake_save)
        monkeypatch.setattr(renderer_module.os, "link", recorded_link)
        monkeypatch.setattr("webbrowser.open", recorded_browser)

        output = renderer.save(
            layout,
            output_dir=tmp_path,
            filename="ordered.html",
            open_in_browser=True,
        )

        assert output == tmp_path / "ordered.html"
        assert events == ["render", "publish", "browser"]
        assert output.read_text(encoding="utf-8") == "rendered"

    def test_browser_failure_retains_published_file_and_chains_cause(
        self,
        tmp_path,
        monkeypatch,
    ):
        import radiosim.core.observability as observability
        import radiosim.visualization.observability.bokeh_renderer as renderer_module

        renderer = ObservabilityBokehRenderer(_tier3g_plan(tmp_path))
        layout = renderer.create_plot()

        def fake_save(_layout, *, filename, resources, title):
            del resources, title
            Path(filename).write_text("rendered", encoding="utf-8")

        def fail_browser(_uri):
            raise RuntimeError("browser unavailable")

        monkeypatch.setattr(renderer_module, "save", fake_save)
        monkeypatch.setattr("webbrowser.open", fail_browser)

        with pytest.raises(observability.ObservabilityBrowserError) as caught:
            renderer.save(
                layout,
                output_dir=tmp_path,
                filename="browser.html",
                open_in_browser=True,
            )

        assert str(caught.value.__cause__) == "browser unavailable"
        assert (tmp_path / "browser.html").read_text(encoding="utf-8") == "rendered"

    def test_render_failure_removes_private_temporary_file(
        self,
        tmp_path,
        monkeypatch,
    ):
        import radiosim.core.observability as observability
        import radiosim.visualization.observability.bokeh_renderer as renderer_module

        renderer = ObservabilityBokehRenderer(_tier3g_plan(tmp_path))
        layout = renderer.create_plot()

        def fail_save(*_args, **_kwargs):
            raise OSError("controlled render write failure")

        monkeypatch.setattr(renderer_module, "save", fail_save)

        with pytest.raises(observability.ObservabilityOutputError) as caught:
            renderer.save(
                layout,
                output_dir=tmp_path,
                filename="failed.html",
            )

        assert str(caught.value.__cause__) == "controlled render write failure"
        assert not (tmp_path / "failed.html").exists()
        assert not tuple(tmp_path.glob(".failed.html.*"))
