"""Tier 4C integration contracts for canonical simulation results."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from radiosim.api import Simulator
from radiosim.core.result import ResultUnavailableError, SimulationResult
from radiosim.io.result_format import ResultFormat
from tests.fixtures.configs import valid_config_mapping


def _mapping(
    tmp_path: Path,
    *,
    duration_seconds: float = 2.5,
    time_step_seconds: float = 1.0,
    sky_representation: str = "point_sources",
) -> dict[str, object]:
    sky_source: dict[str, object] = {
        "kind": "test_sources",
        "num_sources": 1,
        "seed": 17,
    }
    if sky_representation == "healpix_map":
        sky_source.update({"representation": "healpix_map", "nside": 1})
    return valid_config_mapping(
        tmp_path,
        obs_time={
            "start_time": "2025-01-01T00:00:00",
            "duration_seconds": duration_seconds,
            "time_step_seconds": time_step_seconds,
        },
        frequency={
            "mode": "explicit",
            "channel_frequencies_hz": [100e6, 101.5e6],
            "channel_widths_hz": [1e6, 0.5e6],
        },
        sky_model={"sources": [sky_source]},
        visibility={"sky_representation": sky_representation},
    )


def test_run_publishes_one_canonical_result_with_exact_axes(tmp_path):
    simulator = Simulator.from_mapping(_mapping(tmp_path), base_dir=tmp_path)

    assert simulator.result is None
    returned = simulator.run(progress=False)

    assert type(returned) is SimulationResult
    assert returned is simulator.result
    assert not hasattr(simulator, "results")
    assert returned.visibilities.shape == (3, 3, 2, 4)
    assert returned.flags.shape == returned.visibilities.shape
    assert returned.weights.shape == returned.visibilities.shape
    assert returned.correlations == ("XX", "XY", "YX", "YY")
    assert returned.time_grid is simulator.config.observation.time_grid
    np.testing.assert_array_equal(returned.frequencies_hz, [100e6, 101.5e6])
    np.testing.assert_array_equal(returned.channel_widths_hz, [1e6, 0.5e6])
    np.testing.assert_allclose(
        returned.stokes_i(),
        returned.visibilities[..., 0] + returned.visibilities[..., 3],
    )


@pytest.mark.parametrize(
    "sky_representation",
    ["point_sources", "healpix_scalar", "healpix_polarized"],
)
def test_result_factory_owns_the_single_host_transfer(
    tmp_path,
    monkeypatch,
    sky_representation,
):
    simulator = Simulator.from_mapping(
        _mapping(
            tmp_path,
            sky_representation=(
                "point_sources"
                if sky_representation == "point_sources"
                else "healpix_map"
            ),
        ),
        base_dir=tmp_path,
    )
    simulator.setup()
    if sky_representation == "healpix_polarized":
        healpix = simulator._sky_model.healpix
        assert healpix is not None
        polarized = healpix.replace(
            q_maps=np.full_like(healpix.maps, 0.1),
            u_maps=np.full_like(healpix.maps, 0.05),
            v_maps=np.zeros_like(healpix.maps),
        )
        simulator._sky_model = simulator._sky_model.replace(healpix=polarized)
    transfers = 0
    original = simulator._backend.to_numpy

    def counted(value):
        nonlocal transfers
        transfers += 1
        return original(value)

    monkeypatch.setattr(simulator._backend, "to_numpy", counted)

    result = simulator.run(progress=False)

    assert type(result) is SimulationResult
    assert transfers == 1


def test_factory_failure_after_transfer_publishes_nothing(tmp_path, monkeypatch):
    import radiosim.core.result as result_module

    simulator = Simulator.from_mapping(_mapping(tmp_path), base_dir=tmp_path)
    simulator.setup()
    transfers = 0
    native_to_numpy = simulator._backend.to_numpy
    native_factory = result_module.build_simulation_result

    def counted(value):
        nonlocal transfers
        transfers += 1
        return native_to_numpy(value)

    def fail_after_construction(**kwargs):
        native_factory(**kwargs)
        raise RuntimeError("controlled post-transfer factory failure")

    monkeypatch.setattr(simulator._backend, "to_numpy", counted)
    monkeypatch.setattr(
        result_module,
        "build_simulation_result",
        fail_after_construction,
    )

    with pytest.raises(RuntimeError, match="post-transfer factory failure"):
        simulator.run(progress=False)

    assert transfers == 1
    assert simulator.result is None


def test_last_success_publication_is_atomic_across_failures_and_retries(
    tmp_path,
    monkeypatch,
):
    simulator = Simulator.from_mapping(_mapping(tmp_path), base_dir=tmp_path)
    simulator.setup()
    solver = simulator._simulator
    original = solver.calculate_visibilities
    transfers = 0
    native_to_numpy = simulator._backend.to_numpy

    def counted(value):
        nonlocal transfers
        transfers += 1
        return native_to_numpy(value)

    def fail(*args, **kwargs):
        raise RuntimeError("controlled solver failure")

    monkeypatch.setattr(simulator._backend, "to_numpy", counted)
    monkeypatch.setattr(solver, "calculate_visibilities", fail)
    with pytest.raises(RuntimeError, match="controlled solver failure"):
        simulator.run(progress=False)
    assert simulator.result is None
    assert transfers == 0

    monkeypatch.setattr(solver, "calculate_visibilities", original)
    first = simulator.run(progress=False)
    assert simulator.result is first
    assert transfers == 1

    monkeypatch.setattr(solver, "calculate_visibilities", fail)
    with pytest.raises(RuntimeError, match="controlled solver failure"):
        simulator.run(progress=False)
    assert simulator.result is first
    assert transfers == 1

    monkeypatch.setattr(solver, "calculate_visibilities", original)
    second = simulator.run(progress=False)
    assert second is simulator.result
    assert second is not first
    assert transfers == 2


def test_late_success_rendering_failure_never_publishes_a_result(
    tmp_path,
    monkeypatch,
):
    import radiosim.api.simulator as simulator_module

    simulator = Simulator.from_mapping(_mapping(tmp_path), base_dir=tmp_path)
    simulator.setup()
    native_print_success = simulator_module.print_success

    def fail_success_rendering(message):
        if message.startswith("Simulation complete"):
            raise RuntimeError("controlled late success-rendering failure")
        native_print_success(message)

    monkeypatch.setattr(simulator_module, "print_success", fail_success_rendering)
    with pytest.raises(RuntimeError, match="late success-rendering failure"):
        simulator.run(progress=True)
    assert simulator.result is None

    monkeypatch.setattr(simulator_module, "print_success", native_print_success)
    first = simulator.run(progress=False)
    monkeypatch.setattr(simulator_module, "print_success", fail_success_rendering)
    with pytest.raises(RuntimeError, match="late success-rendering failure"):
        simulator.run(progress=True)
    assert simulator.result is first


def test_memory_estimate_uses_exact_canonical_time_count(tmp_path, monkeypatch):
    simulator = Simulator.from_mapping(_mapping(tmp_path), base_dir=tmp_path)
    simulator.setup()
    captured: dict[str, object] = {}

    def estimate(**kwargs):
        captured.update(kwargs)
        return {"total_bytes": 1, "total_human": "1 B"}

    monkeypatch.setattr(simulator._simulator, "get_memory_estimate", estimate)

    simulator.get_memory_estimate()

    assert captured["n_times"] == len(simulator.config.observation.time_grid) == 3


def test_save_requires_result_and_plot_remains_unavailable_without_side_effects(
    tmp_path,
    monkeypatch,
):
    import builtins
    import logging
    import webbrowser

    from radiosim.visualization.errors import ResultPlotContractError

    simulator = Simulator.from_mapping(_mapping(tmp_path), base_dir=tmp_path)
    output = tmp_path / "must-not-exist.h5"

    def forbidden(*_args, **_kwargs):
        pytest.fail("unavailable result workflow crossed a side-effect boundary")

    native_import = builtins.__import__
    forbidden_imports = (
        "h5py",
        "pyuvdata",
        "casacore",
        "radiosim.io.writers",
        "radiosim.visualization.bokeh_plots",
        "radiosim.visualization.gsm_plots",
        "radiosim.visualization.observability",
        "radiosim.visualization.sky",
        "bokeh",
        "matplotlib",
    )

    def guarded_import(name, *args, **kwargs):
        if name.startswith(forbidden_imports):
            pytest.fail(f"unavailable result workflow imported {name}")
        return native_import(name, *args, **kwargs)

    def assert_fail_closed(*, include_save: bool):
        operations = [
            (
                lambda: simulator.plot(
                    plot_type="hostile",
                    output_dir=output,
                    backend="hostile",
                    show=True,
                    overwrite=True,
                ),
                ResultPlotContractError,
                "plot_type must be one of",
            ),
        ]
        if include_save:
            operations.insert(
                0,
                (
                    lambda: simulator.save(
                        output,
                        format=ResultFormat.HDF5,
                        overwrite=True,
                    ),
                    ResultUnavailableError,
                    "no successfully published SimulationResult",
                ),
            )
            operations.append(
                (
                    lambda: simulator.plot(
                        output_dir=output,
                        show=True,
                        overwrite=True,
                    ),
                    ResultUnavailableError,
                    "requires a successfully published SimulationResult",
                ),
            )
        for operation, error_type, message in operations:
            with pytest.raises(error_type, match=message):
                operation()

    with monkeypatch.context() as guarded:
        guarded.setattr(builtins, "__import__", guarded_import)
        guarded.setattr(builtins, "open", forbidden)
        guarded.setattr(Path, "exists", forbidden)
        guarded.setattr(Path, "mkdir", forbidden)
        guarded.setattr(logging, "getLogger", forbidden)
        guarded.setattr(webbrowser, "open", forbidden)
        assert_fail_closed(include_save=True)

    assert not output.exists()
    simulator.run(progress=False)

    with monkeypatch.context() as guarded:
        guarded.setattr(builtins, "__import__", guarded_import)
        guarded.setattr(builtins, "open", forbidden)
        guarded.setattr(Path, "exists", forbidden)
        guarded.setattr(Path, "mkdir", forbidden)
        guarded.setattr(logging, "getLogger", forbidden)
        guarded.setattr(webbrowser, "open", forbidden)
        assert_fail_closed(include_save=False)

    assert not output.exists()
    assert simulator.save(output, format=ResultFormat.HDF5) == output
    assert output.is_file()


def test_solver_api_has_only_the_canonical_time_grid_contract():
    from radiosim.core.visibility import calculate_visibility
    from radiosim.core.visibility_healpix import calculate_visibility_healpix

    for function in (calculate_visibility, calculate_visibility_healpix):
        parameters = inspect.signature(function).parameters
        assert "time_grid" in parameters
        for removed in (
            "obstime",
            "wavelengths",
            "duration_seconds",
            "time_step_seconds",
            "return_correlations",
        ):
            assert removed not in parameters
