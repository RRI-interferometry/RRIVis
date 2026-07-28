"""Target-contract tests for the typed parameter-driven simulate command."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from radiosim.cli.main import cli
from radiosim.io.result_format import ResultFormat
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
        "--output",
        str(antenna_path.parent / "direct-result"),
        *extra,
    ]


def test_simulate_uses_typed_parameters_and_preserves_nonuniform_hz(
    tmp_path, recording_simulator
):
    antenna_path = write_minimal_antenna_file(tmp_path)
    output_path = tmp_path / "output-summary"

    result = CliRunner().invoke(
        cli,
        _simulate_args(
            antenna_path,
            "--sky-model",
            "test",
            "--output",
            str(output_path),
            "--format",
            "summary_json",
            "--backend",
            "numpy",
        ),
    )

    assert result.exit_code == 0, result.output
    simulator = recording_simulator.instances[0]
    assert simulator.config.frequency.channel_frequencies_hz == (
        100e6,
        101.5e6,
        108e6,
    )
    assert simulator.config.frequency.channel_widths_hz == (1e6, 0.5e6, 2e6)
    assert simulator.save_calls == [
        (
            (str(output_path),),
            {"format": ResultFormat.SUMMARY_JSON, "overwrite": False},
        )
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
    simulator = recording_simulator.instances[0]
    assert simulator.config.sky_model.sources[0].kind == kind
    assert simulator.config.visibility["sky_representation"] == representation


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
    assert "obs_frequency.explicit.channel_frequencies_hz" in result.output
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
    assert "--channel-widths-mhz must contain the same number" in result.output
    assert recording_simulator.instances == []


def test_simulate_rejects_missing_antenna_before_simulator_construction(
    tmp_path, recording_simulator
):
    missing_antenna = tmp_path / "missing.txt"

    result = CliRunner().invoke(cli, _simulate_args(missing_antenna))

    assert result.exit_code == 1
    assert recording_simulator.instances == []
    assert "instrument.source.path" in result.output


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
    assert "incomplete antenna diameters" in normalized_output


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


def test_direct_simulate_saves_exact_final_target_without_workflow_policy(
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

    assert result.exit_code == 0, result.output
    simulator = recording_simulator.instances[0]
    assert simulator.ran is True
    assert simulator.save_calls == [
        (
            (str(output),),
            {"format": ResultFormat.HDF5, "overwrite": False},
        )
    ]


def test_direct_simulate_rejects_legacy_json_with_exact_guidance(
    tmp_path,
    recording_simulator,
):
    antenna_path = write_minimal_antenna_file(tmp_path)

    result = CliRunner().invoke(
        cli,
        _simulate_args(antenna_path, "--format", "json"),
    )

    normalized = " ".join(result.output.split())
    assert result.exit_code == 2
    assert (
        "format 'json' was removed before v1.0 because it did not contain "
        "visibility data; use 'summary_json' for metadata or 'hdf5' for a "
        "lossless RadioSim result"
    ) in normalized
    assert recording_simulator.instances == []
