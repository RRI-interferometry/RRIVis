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
        "--channel-widths-mhz",
        "1,0.5,2",
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

    assert result.exit_code == 1
    assert "result saving is temporarily unavailable" in result.output
    assert recording_simulator.instances == []
    assert not output_dir.exists()


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

    assert result.exit_code == 1
    assert "temporarily unavailable" in result.output
    assert recording_simulator.instances == []


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
    count = len(frequencies.split(","))
    args[args.index("--channel-widths-mhz") + 1] = ",".join(["1"] * count)

    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 1
    assert "temporarily unavailable" in result.output
    assert recording_simulator.instances == []


def test_simulate_rejects_width_length_mismatch_before_simulator_construction(
    tmp_path,
    recording_simulator,
):
    antenna_path = write_minimal_antenna_file(tmp_path)
    args = _simulate_args(antenna_path)
    args[args.index("--channel-widths-mhz") + 1] = "1,1"

    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 1
    assert "temporarily unavailable" in result.output
    assert recording_simulator.instances == []


def test_simulate_rejects_missing_antenna_before_simulator_construction(
    tmp_path, recording_simulator
):
    missing_antenna = tmp_path / "missing.txt"

    result = CliRunner().invoke(cli, _simulate_args(missing_antenna))

    assert result.exit_code == 1
    assert recording_simulator.instances == []
    assert "temporarily unavailable" in result.output


def test_simulate_reports_missing_diameter_without_an_implicit_default(
    tmp_path,
):
    antenna_path = tmp_path / "missing-diameter.txt"
    antenna_path.write_text(
        "Name Number BeamID E N U\nANT0 0 0 0.0 0.0 0.0\nANT1 1 0 14.0 0.0 0.0\n",
        encoding="utf-8",
    )
    args = _simulate_args(antenna_path)
    default_index = args.index("--default-diameter-m")
    del args[default_index : default_index + 2]
    assert "--default-diameter-m" not in args

    result = CliRunner().invoke(cli, args)

    normalized_output = " ".join(result.output.split())
    assert result.exit_code == 1
    assert "temporarily unavailable" in normalized_output


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
            "--channel-widths-mhz",
            "1,1",
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
        "--channel-widths-mhz",
    ):
        line = next(line for line in result.output.splitlines() if option in line)
        assert "required" in line.lower()
    assert "--duration-seconds" in result.output
    assert "--time-step-seconds" in result.output
    assert "--default-diameter-m" in result.output
    assert "--correlations" in result.output


def test_root_help_uses_real_native_layout_and_scopes_config_path_override():
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    normalized = " ".join(result.output.split())
    assert "--antenna-layout antenna_layout_examples/hera_5.txt" in normalized
    assert "Path override for a config layout_file instrument source" in normalized
    assert "HERA65.csv" not in result.output


def test_direct_simulate_fails_at_tier4c_output_preflight_before_runtime(
    tmp_path,
    recording_simulator,
):
    antenna_path = write_minimal_antenna_file(tmp_path)
    output = tmp_path / "must-not-exist"

    result = CliRunner().invoke(
        cli,
        _simulate_args(
            antenna_path,
            "--sky-model",
            "test",
            "--output",
            str(output),
        ),
    )

    assert result.exit_code == 1
    assert "temporarily unavailable" in result.output
    assert recording_simulator.instances == []
    assert not output.exists()
