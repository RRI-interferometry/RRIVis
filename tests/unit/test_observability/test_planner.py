"""Tests for observability planning."""

import inspect
import json
import subprocess
import sys
from pathlib import Path

import healpy as hp
import numpy as np
import pytest

from radiosim.api.simulator import Simulator
from radiosim.core.observability import ObservabilityPlanner
from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky import HealpixData, create_from_arrays
from radiosim.core.sky.containers.model import SkyModel
from radiosim.io.instrument_config import (
    AntennaNameReference,
    AntennaNumberReference,
)


@pytest.fixture
def precision():
    return PrecisionConfig.standard()


def _point_sky(precision):
    ras = np.deg2rad([30.0, 45.0, 60.0, 120.0])
    decs = np.deg2rad([-30.0, -28.0, -32.0, -30.0])
    fluxes = np.array([10.0, 5.0, 8.0, 1.0])
    zeros = np.zeros(4)
    return create_from_arrays(
        ra_rad=ras,
        dec_rad=decs,
        flux=fluxes,
        spectral_index=zeros,
        stokes_q=zeros,
        stokes_u=zeros,
        stokes_v=zeros,
        model_name="test_points",
        brightness_conversion="planck",
        precision=precision,
    )


def _combined_sky(precision, *, coordinate_frame: str = "icrs"):
    point_sky = _point_sky(precision)
    nside = 4
    npix = hp.nside2npix(nside)
    maps = np.linspace(10.0, 100.0, npix, dtype=np.float32)[None, :]
    return SkyModel(
        point=point_sky.point,
        healpix=HealpixData(
            maps=maps,
            nside=nside,
            frequencies=np.array([150e6], dtype=np.float64),
            coordinate_frame=coordinate_frame,
        ),
        reference_frequency=150e6,
        model_name="combined_sky",
        precision=precision,
    )


def _tier3g_mapping(
    tmp_path: Path,
    *,
    diameters: tuple[float, float] = (14.0, 14.0),
    frequencies: tuple[float, ...] = (100_000_000.0,),
) -> dict[str, object]:
    antenna_path = tmp_path / "tier3g-antennas.txt"
    antenna_path.write_text(
        "Name Number BeamID E N U Diameter\n"
        f"ANT9 9 0 0.0 0.0 0.0 {diameters[0]}\n"
        f"ANT2 2 0 14.0 0.0 0.0 {diameters[1]}\n",
        encoding="utf-8",
    )
    return {
        "instrument": {
            "source": {
                "kind": "layout_file",
                "path": str(antenna_path),
                "format": "radiosim",
                "telescope_name": "Tier3G Array",
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
            "duration_seconds": 120.0,
            "time_step_seconds": 1.0,
        },
        "obs_frequency": {
            "mode": "explicit",
            "channel_frequencies_hz": list(frequencies),
        },
        "sky_model": {
            "sources": [{"kind": "test_sources", "num_sources": 1, "seed": 7}]
        },
        "execution": {"backend": "numpy", "offline": True},
    }


def _tier3g_simulator(
    tmp_path: Path,
    *,
    diameters: tuple[float, float] = (14.0, 14.0),
    frequencies: tuple[float, ...] = (100_000_000.0,),
) -> Simulator:
    return Simulator.from_mapping(
        _tier3g_mapping(
            tmp_path,
            diameters=diameters,
            frequencies=frequencies,
        ),
        base_dir=tmp_path,
    )


class TestTier3GPlannerContracts:
    def test_fresh_core_import_does_not_load_render_or_beam_dependencies(self):
        script = """
import json
import sys

import radiosim.core.observability

forbidden = ("bokeh", "healpy", "matplotlib", "pyuvdata", "webbrowser")
print(json.dumps([name for name in forbidden if name in sys.modules]))
"""

        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )

        assert json.loads(completed.stdout) == []

    def test_public_error_hierarchy_is_complete_and_only_core_exported(self):
        import radiosim.core.observability as observability

        error_names = (
            "ObservabilityError",
            "InvalidObservabilityReferenceError",
            "InvalidObservabilityContextError",
            "ObservabilitySkyUnavailableError",
            "UnsupportedObservabilitySemanticsError",
            "ObservabilityRenderError",
            "ObservabilityOutputError",
            "ObservabilityOutputCollisionError",
            "ObservabilityBrowserError",
        )
        root = observability.ObservabilityError
        assert issubclass(root, RuntimeError)
        for name in error_names:
            error = getattr(observability, name)
            assert name in observability.__all__
            assert issubclass(error, root)

        import radiosim

        assert all(not hasattr(radiosim, name) for name in error_names)

    def test_window_and_option_models_are_frozen_slotted_and_strict(self):
        import radiosim.core.observability as observability

        utc = observability.UTCObservabilityWindow(
            kind="utc",
            start_time_iso="2025-01-01T00:00:00",
            duration_seconds=10.0,
            source="resolved_utc",
        )
        lst = observability.LSTObservabilityWindow(
            kind="lst",
            start_hours=23.0,
            end_hours=1.0,
            wraps_midnight=True,
            source="explicit_lst",
            beam_evaluation_time_mjd=60_676.0,
        )
        options = observability.ObservabilityOptions()

        assert utc.kind == "utc"
        assert lst.wraps_midnight is True
        assert tuple(options.__dataclass_fields__) == (
            "x_axis",
            "background_layer",
            "footprint_model",
            "field_radius_deg",
            "mode",
            "snapshot_step_seconds",
            "footprint_step_seconds",
            "beam_time_reference",
            "beam_contour_min_db",
            "beam_contour_max_db",
            "grid_resolution_deg",
            "max_point_sources",
            "top_n_sources",
            "nearby_source_count",
            "nearby_buffer_deg",
            "include_source_metrics",
        )
        with pytest.raises((AttributeError, TypeError)):
            options.mode = "snapshots"

        invalid_options = (
            {"snapshot_step_seconds": 0.0},
            {"grid_resolution_deg": 10.1},
            {"beam_contour_min_db": 0.0},
            {"beam_contour_max_db": 1.0},
            {"max_point_sources": True},
            {"top_n_sources": 2, "max_point_sources": 1},
            {"nearby_source_count": 2, "max_point_sources": 1},
            {"nearby_buffer_deg": 181.0},
            {"field_radius_deg": 5.0},
            {"footprint_model": "manual_circular", "field_radius_deg": None},
            {"footprint_model": "rectangular_approx"},
        )
        for values in invalid_options:
            error = (
                observability.UnsupportedObservabilitySemanticsError
                if values.get("footprint_model") == "rectangular_approx"
                else observability.InvalidObservabilityContextError
            )
            with pytest.raises(error):
                observability.ObservabilityOptions(**values)

    @pytest.mark.parametrize("beam_time_reference", ["start", "midpoint", "end"])
    def test_utc_and_wrapped_lst_time_reference_provenance(
        self,
        tmp_path,
        beam_time_reference,
    ):
        simulator = _tier3g_simulator(tmp_path)

        utc = simulator.plan_observability(
            beam_time_reference=beam_time_reference,
            grid_resolution_deg=10.0,
        )
        lst = simulator.plan_observability(
            lst_start_hours=23.0,
            lst_end_hours=1.0,
            beam_time_reference=beam_time_reference,
            footprint_step_seconds=3600.0,
            grid_resolution_deg=10.0,
        )

        assert utc.window_source == "resolved_utc"
        assert utc.observation_start_iso is not None
        assert utc.observation_end_iso is not None
        assert np.isfinite(utc.beam_time_reference_mjd)
        assert lst.window_source == "explicit_lst"
        assert lst.observation_start_iso is None
        assert lst.observation_end_iso is None
        assert lst.beam_time_reference_mjd == pytest.approx(60_676.0, abs=1.0)
        expected_lst = {
            "start": 23.0,
            "midpoint": 0.0,
            "end": 1.0,
        }[beam_time_reference]
        assert lst.beam_time_reference_lst_hours == pytest.approx(expected_lst)

    def test_partial_lst_override_is_invalid(self, tmp_path):
        import radiosim.core.observability as observability

        simulator = _tier3g_simulator(tmp_path)
        with pytest.raises(observability.InvalidObservabilityContextError):
            simulator.plan_observability(
                lst_start_hours=1.0,
                grid_resolution_deg=10.0,
            )

    def test_heterogeneous_assignments_require_explicit_reference(self, tmp_path):
        import radiosim.core.observability as observability

        simulator = _tier3g_simulator(tmp_path, diameters=(12.0, 25.0))

        with pytest.raises(observability.InvalidObservabilityReferenceError):
            simulator.plan_observability(grid_resolution_deg=10.0)

    @pytest.mark.parametrize(
        ("reference", "expected_number", "expected_name"),
        [
            (AntennaNumberReference(number=9), 9, "ANT9"),
            (AntennaNameReference(name="ANT2"), 2, "ANT2"),
        ],
    )
    def test_explicit_name_and_number_reference_resolution(
        self,
        tmp_path,
        reference,
        expected_number,
        expected_name,
    ):
        simulator = _tier3g_simulator(tmp_path, diameters=(12.0, 25.0))

        plan = simulator.plan_observability(
            reference_antenna=reference,
            grid_resolution_deg=10.0,
        )

        assert plan.reference_antenna.number == expected_number
        assert plan.reference_antenna.name == expected_name
        assert plan.reference_selection_reason == "explicit"

    def test_homogeneous_default_uses_minimum_canonical_number(self, tmp_path):
        simulator = _tier3g_simulator(tmp_path)

        plan = simulator.plan_observability(grid_resolution_deg=10.0)

        assert plan.reference_antenna.number == 2
        assert plan.reference_antenna.name == "ANT2"
        assert plan.reference_selection_reason == "homogeneous_default_minimum_number"
        assert plan.reference_scientific_fingerprint[:12] in plan.title
        assert "number=2" in plan.title
        assert "name='ANT2'" in plan.title

    def test_multichannel_requires_explicit_exact_channel(self, tmp_path):
        import radiosim.core.observability as observability

        simulator = _tier3g_simulator(
            tmp_path,
            frequencies=(100_000_000.0, 120_000_000.0),
        )

        with pytest.raises(observability.InvalidObservabilityContextError):
            simulator.plan_observability(grid_resolution_deg=10.0)

        plan = simulator.plan_observability(
            channel_index=1,
            grid_resolution_deg=10.0,
        )
        assert plan.channel_index == 1
        assert plan.frequency_hz == 120_000_000.0

        for invalid in (True, -1, 2):
            with pytest.raises(observability.InvalidObservabilityContextError):
                simulator.plan_observability(
                    channel_index=invalid,
                    grid_resolution_deg=10.0,
                )

    @pytest.mark.parametrize(
        "sky_request",
        [
            {"background_layer": "diffuse"},
            {"include_source_metrics": True},
        ],
    )
    def test_requested_sky_payload_must_already_be_prepared(
        self,
        tmp_path,
        sky_request,
    ):
        import radiosim.core.observability as observability

        simulator = _tier3g_simulator(tmp_path)

        with pytest.raises(observability.ObservabilitySkyUnavailableError):
            simulator.plan_observability(
                grid_resolution_deg=10.0,
                **sky_request,
            )

    def test_prepared_exact_channel_background_and_source_metrics(
        self,
        tmp_path,
        precision,
    ):
        simulator = _tier3g_simulator(
            tmp_path,
            frequencies=(150_000_000.0,),
        )
        simulator._sky_model = _combined_sky(
            precision,
            coordinate_frame="galactic",
        )

        plan = simulator.plan_observability(
            background_layer="diffuse",
            include_source_metrics=True,
            footprint_step_seconds=120.0,
            grid_resolution_deg=10.0,
            top_n_sources=2,
        )

        assert plan.projected_background is not None
        assert plan.projected_background.shape == plan.footprint_mask.shape
        assert plan.source_metrics is not None
        assert len(plan.source_metrics.ra_deg) == 4
        assert len(plan.source_metrics.top_visible_indices) <= 2

    def test_beam_only_plan_before_setup_has_no_later_side_effects(
        self,
        tmp_path,
        monkeypatch,
    ):
        simulator = _tier3g_simulator(tmp_path)

        def forbidden(*_args, **_kwargs):
            pytest.fail("beam-only observability planning crossed its side-effect seam")

        monkeypatch.setattr("radiosim.utils.device.get_device_resources", forbidden)
        monkeypatch.setattr("radiosim.backends.get_backend", forbidden)
        monkeypatch.setattr("radiosim.simulator.get_simulator", forbidden)
        monkeypatch.setattr("radiosim.utils.network.get_network_status", forbidden)
        monkeypatch.setattr(
            "radiosim.core.sky.operations.parallel.load_models_parallel",
            forbidden,
        )
        monkeypatch.setattr("webbrowser.open", forbidden)
        monkeypatch.setattr(Path, "mkdir", forbidden)

        plan = simulator.plan_observability(grid_resolution_deg=10.0)

        assert plan.reference_antenna.number == 2
        assert simulator._backend is None
        assert simulator._simulator is None
        assert simulator._sky_model is None
        assert simulator._is_setup is False

    def test_actual_reference_half_power_footprint_uses_full_jones(
        self,
        tmp_path,
        monkeypatch,
    ):
        from radiosim.core.beam import BeamSystem

        simulator = _tier3g_simulator(tmp_path)

        def off_diagonal_gaussian(
            self,
            antenna_id,
            *,
            altitude_rad,
            azimuth_rad,
            frequency_hz,
            time_mjd,
            backend=None,
        ):
            del self, antenna_id, azimuth_rad, frequency_hz, time_mjd
            za = np.pi / 2.0 - np.asarray(altitude_rad)
            power = np.exp(-((za / np.deg2rad(12.0)) ** 2))
            result = np.zeros(za.shape + (2, 2), dtype=np.complex128)
            result[..., 0, 1] = np.sqrt(2.0 * power) * np.exp(0.7j)
            result[np.asarray(altitude_rad) <= 0.0] = 0.0
            if backend is not None:
                return backend.asarray(result, dtype=backend.default_complex_dtype)
            result.setflags(write=False)
            return result

        monkeypatch.setattr(BeamSystem, "evaluate_jones", off_diagonal_gaussian)
        plan = simulator.plan_observability(
            footprint_step_seconds=120.0,
            grid_resolution_deg=2.0,
        )

        zenith_ra = plan.track_ra_deg[0]
        dec_idx = int(np.argmin(np.abs(plan.dec_grid_deg - plan.latitude_deg)))
        center_idx = int(np.argmin(np.abs(plan.ra_grid_deg - zenith_ra)))
        outside_idx = int(np.argmin(np.abs(plan.ra_grid_deg - (zenith_ra + 20.0))))
        assert plan.footprint_mask[dec_idx, center_idx]
        assert not plan.footprint_mask[dec_idx, outside_idx]
        assert plan.footprint_provenance == "reference_beam_half_power"
        assert plan.power_convention == "half_trace_unpolarized"

    def test_plan_provenance_is_json_safe_and_arrays_are_owned_read_only(
        self,
        tmp_path,
    ):
        simulator = _tier3g_simulator(tmp_path)

        first = simulator.plan_observability(grid_resolution_deg=10.0)
        second = simulator.plan_observability(grid_resolution_deg=10.0)
        snapshot = first.provenance_snapshot()

        assert "track_ra_deg" not in snapshot
        assert snapshot["reference_antenna"] == {"number": 2, "name": "ANT2"}
        assert first.track_ra_deg.flags.owndata
        assert not first.track_ra_deg.flags.writeable
        assert first.track_ra_deg is not second.track_ra_deg
        assert first.__hash__ is None

    def test_removed_observability_arguments_and_permissive_kwargs_are_absent(self):
        plan_parameters = inspect.signature(Simulator.plan_observability).parameters
        plot_parameters = inspect.signature(Simulator.plot_observability).parameters
        planner_parameters = inspect.signature(ObservabilityPlanner).parameters

        for removed in (
            "beam_reference",
            "rectangular_approx",
            "save_path",
            "beam_fits_path",
            "beam_diameter_m",
            "latitude_deg",
            "longitude_deg",
            "kwargs",
        ):
            assert removed not in plan_parameters
            assert removed not in plot_parameters
            assert removed not in planner_parameters
