"""Target-contract tests for the typed parameter-driven simulate command."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from radiosim.cli.main import cli
from tests.fixtures.configs import write_minimal_antenna_file


def _simulate_args(antenna_path, *extra: str) -> list[str]:
    return [
        "simulate",
        "--antenna-layout",
        str(antenna_path),
        "--telescope-name",
        "CLI Array",
        "--default-diameter-m",
        "14",
        "--frequencies",
        "100,101.5,108",
        "--latitude",
        "-30.72152",
        "--longitude",
        "21.4283",
        "--height",
        "1073",
        "--start-time",
        "2025-01-01T00:00:00",
        *extra,
    ]


def test_simulate_uses_typed_parameters_and_preserves_nonuniform_hz(
    tmp_path, recording_simulator
):
    antenna_path = write_minimal_antenna_file(tmp_path)
    output_dir = tmp_path / "output"

    result = CliRunner().invoke(
        cli,
        _simulate_args(
            antenna_path,
            "--sky-model",
            "test",
            "--output",
            str(output_dir),
            "--format",
            "json",
            "--backend",
            "numpy",
        ),
    )

    assert result.exit_code == 0, result.output
    simulator = recording_simulator.instances[0]
    assert simulator.ran is True
    assert simulator.config.frequency.channel_frequencies_hz == (
        100e6,
        101.5e6,
        108e6,
    )
    assert simulator.config.frequency.source_mode == "explicit"
    assert simulator.config.execution.backend_strategy == "numpy"
    assert simulator.config.instrument.location.latitude_deg == pytest.approx(-30.72152)
    assert simulator.config.instrument.default_diameter_m == 14.0
    assert simulator.config.baseline_selection.correlations == "all"
    assert simulator.config.observation.start_time_iso.startswith("2025-01-01T00:00:00")
    assert simulator.save_calls == [
        ((str(output_dir),), {"format": "json", "overwrite": False})
    ]


@pytest.mark.parametrize(
    ("sky_alias", "kind", "representation"),
    [
        ("test", "test_sources", "point_sources"),
        ("gleam", "gleam", "point_sources"),
        ("gsm", "diffuse_sky", "healpix_map"),
    ],
)
def test_simulate_resolves_registered_sky_alias_to_typed_inputs(
    tmp_path, recording_simulator, sky_alias, kind, representation
):
    antenna_path = write_minimal_antenna_file(tmp_path)

    result = CliRunner().invoke(
        cli, _simulate_args(antenna_path, "--sky-model", sky_alias)
    )

    assert result.exit_code == 0, result.output
    runtime = recording_simulator.instances[0].config
    assert runtime.sky_model.sources[0].kind == kind
    assert runtime.visibility["sky_representation"] == representation


@pytest.mark.parametrize(
    "frequencies",
    ["100,100", "101,100", "0,100", "-1,100", "nan,100", "100,inf"],
)
def test_simulate_rejects_invalid_frequency_sequences_through_resolver(
    tmp_path, recording_simulator, frequencies
):
    antenna_path = write_minimal_antenna_file(tmp_path)
    args = _simulate_args(antenna_path)
    args[args.index("--frequencies") + 1] = frequencies

    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 1
    assert "channel_frequencies_hz" in result.output
    assert recording_simulator.instances == []


def test_simulate_rejects_missing_antenna_before_simulator_construction(
    tmp_path, recording_simulator
):
    missing_antenna = tmp_path / "missing.txt"

    result = CliRunner().invoke(cli, _simulate_args(missing_antenna))

    assert result.exit_code == 1
    assert recording_simulator.instances == []
    assert "instrument.source.path" in result.output


def test_simulate_requires_explicit_location_and_start_time(
    tmp_path, recording_simulator
):
    antenna_path = write_minimal_antenna_file(tmp_path)

    result = CliRunner().invoke(
        cli,
        [
            "simulate",
            "--antenna-layout",
            str(antenna_path),
            "--telescope-name",
            "CLI Array",
            "--frequencies",
            "100,101",
        ],
    )

    assert result.exit_code == 2
    assert "Missing option '--latitude'" in result.output
    assert recording_simulator.instances == []


def test_simulate_help_documents_required_observation_inputs():
    result = CliRunner().invoke(cli, ["simulate", "--help"])

    assert result.exit_code == 0
    for option in (
        "--telescope-name",
        "--latitude",
        "--longitude",
        "--height",
        "--start-time",
    ):
        line = next(line for line in result.output.splitlines() if option in line)
        assert "required" in line.lower()
    assert "--duration-seconds" in result.output
    assert "--time-step-seconds" in result.output
    assert "--default-diameter-m" in result.output
    assert "--correlations" in result.output
