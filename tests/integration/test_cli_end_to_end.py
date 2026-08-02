"""The command line, driven as a user drives it, to artifacts on disk.

``Fix.md`` Section 17 item 9 asks for the one thing ``tests/integration/`` did
not do: run ``radiosim --config <document>`` **as a subprocess** and read the
run directory it publishes back off the filesystem.  The two existing
integration modules drive the Python :class:`~radiosim.api.Simulator` API end to
end, which exercises the solver but not the CLI, not the workflow transaction,
and not the published artifacts.  ``Tier8ReleasePlan.md`` Section 17 assigns
this module to slice 8D.

What "end to end" means here, precisely:

*   a YAML document is written to ``tmp_path`` -- nothing in the repository is
    read or written, so the test is hermetic and runs in a fresh clone;
*   the installed ``radiosim`` console script is invoked in a child process, so
    argument parsing, configuration resolution, the run, the atomic staging and
    publication, and the process exit status are all real;
*   the published directory is then read with no help from the writer: the
    manifest's SHA-256 digests are recomputed from the bytes on disk, the HDF5
    file is opened with ``h5py``, and the summary JSON is parsed with
    :mod:`json`;
*   the two runs are cross-checked -- the same document written in two result
    formats must report the same ``scientific_sha256``, which is what makes the
    published artifacts evidence of *one* simulation rather than two.

Every run is ``execution.offline: true`` over synthetic sources and a
two-antenna layout written into ``tmp_path``, so no network, no catalog and no
repository fixture is involved.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.fixtures.configs import valid_config_mapping

pytestmark = pytest.mark.integration

#: Files a run with ``save_results: true`` and no log or plots publishes.
EXPECTED_ARTIFACTS = ("manifest.json", "resolved-config.yaml")

MANIFEST_SCHEMA = "radiosim.workflow-manifest.v1"


def _cli_command() -> list[str]:
    """Return the argv prefix that invokes the CLI the documentation names.

    The console script installed beside this interpreter is preferred, because
    ``radiosim --config ...`` is the documented surface and the entry point is
    part of what this test covers.  ``python -m radiosim.cli.main`` is the
    fallback for an environment that has the package importable without the
    script on disk; it runs the same ``main()``.
    """
    script = Path(sys.executable).parent / (
        "radiosim.exe" if os.name == "nt" else "radiosim"
    )
    if script.is_file():
        return [str(script)]
    return [sys.executable, "-m", "radiosim.cli.main"]


def _write_config(tmp_path: Path, *, subdir: str, result_format: str) -> Path:
    """Write a minimal, offline, synthetic-sky configuration document."""
    mapping = valid_config_mapping(
        tmp_path,
        workflow={
            "output_dir": str(tmp_path / "output"),
            "run_subdir": subdir,
            "result_filename": "visibilities",
            "result_format": result_format,
            "collision_policy": "error",
            "save_results": True,
            "plot_results": False,
            "open_plots_in_browser": False,
            "save_log": False,
        },
    )
    document = tmp_path / f"{subdir}.yaml"
    document.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    return document


def _run_cli(document: Path, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [*_cli_command(), "--config", str(document)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert completed.returncode == 0, (
        f"radiosim --config {document} exited {completed.returncode}\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )
    return completed


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_manifest(run: Path) -> dict[str, Any]:
    document = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert document["schema"] == MANIFEST_SCHEMA
    return document


@pytest.fixture(scope="module")
def _published_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Publish one HDF5 run and one summary-JSON run, once for this module.

    Module-scoped because each entry is a real subprocess simulation; the tests
    below only read the results, and re-running the CLI per test would buy
    nothing but seconds.
    """
    tmp_path = tmp_path_factory.mktemp("cli-end-to-end")
    runs: dict[str, Path] = {}
    for subdir, result_format in (("hdf5-run", "hdf5"), ("json-run", "summary_json")):
        document = _write_config(tmp_path, subdir=subdir, result_format=result_format)
        _run_cli(document, tmp_path)
        runs[result_format] = tmp_path / "output" / subdir
    return runs


def test_the_cli_publishes_a_complete_run_directory(
    _published_runs: dict[str, Path],
) -> None:
    """The transaction leaves exactly the artifacts its manifest claims."""
    run = _published_runs["hdf5"]
    assert run.is_dir()
    published = sorted(path.name for path in run.iterdir())
    assert published == sorted([*EXPECTED_ARTIFACTS, "visibilities.h5"])
    assert not any(name.startswith(".") for name in published), "staging residue"

    manifest = _read_manifest(run)
    listed = {entry["path"]: entry for entry in manifest["artifacts"]}
    assert set(listed) == {"resolved-config.yaml", "visibilities.h5"}
    for name, entry in listed.items():
        assert entry["kind"] == "file"
        assert entry["sha256"] == _sha256(run / name), (
            f"{name} does not hash to the digest its own manifest records"
        )


def test_the_published_hdf5_reads_back_as_the_simulation_that_was_run(
    _published_runs: dict[str, Path],
) -> None:
    """Opened with ``h5py`` alone: the cube, its axes, and its identity."""
    h5py = pytest.importorskip("h5py")
    run = _published_runs["hdf5"]

    with h5py.File(run / "visibilities.h5", "r") as handle:
        visibilities = handle["data/visibilities"][...]
        flags = handle["data/flags"][...]
        weights = handle["data/weights"][...]
        labels = [
            item.decode() for item in handle["coordinates/correlation/labels"][...]
        ]
        frequencies = handle["coordinates/frequency/center_hz"][...]
        times = handle["coordinates/time/utc_jd1"][...]
        antenna1 = handle["coordinates/baseline/antenna1_number"][...]
        scientific = handle.attrs["scientific_sha256"]
        dimension_order = handle.attrs["dimension_order"]

    # The document declares two antennas, one baseline pair set, two time steps
    # and three channels; the cube's shape is the arithmetic of that document.
    n_time, n_baseline, n_freq, n_corr = visibilities.shape
    assert (n_time, n_freq, n_corr) == (2, 3, 4)
    assert n_baseline == len(antenna1) == 3
    assert flags.shape == weights.shape == visibilities.shape
    assert not flags.any(), "an offline synthetic run flags nothing"
    assert labels == ["XX", "XY", "YX", "YY"]
    assert len(frequencies) == n_freq and len(times) == n_time
    assert visibilities.any(), "the published cube is entirely zero"

    order = (
        dimension_order.decode()
        if isinstance(dimension_order, bytes)
        else str(dimension_order)
    )
    assert order.replace(" ", "") == "time,baseline,frequency,correlation"

    digest = scientific.decode() if isinstance(scientific, bytes) else str(scientific)
    assert len(digest) == 64 and int(digest, 16) >= 0


def test_the_published_summary_json_describes_the_same_simulation(
    _published_runs: dict[str, Path],
) -> None:
    """The second format, parsed with ``json``, and cross-checked against the first."""
    h5py = pytest.importorskip("h5py")
    json_run = _published_runs["summary_json"]

    published = sorted(path.name for path in json_run.iterdir())
    assert published == sorted([*EXPECTED_ARTIFACTS, "visibilities.summary.json"])
    manifest = _read_manifest(json_run)
    for entry in manifest["artifacts"]:
        assert entry["sha256"] == _sha256(json_run / entry["path"])

    summary = json.loads(
        (json_run / "visibilities.summary.json").read_text(encoding="utf-8")
    )
    assert summary["schema"] == {"name": "radiosim.result-summary", "version": "1.2.0"}
    assert summary["result"]["shape"] == [2, 3, 3, 4]
    assert summary["result"]["axis_counts"] == {
        "time": 2,
        "baseline": 3,
        "frequency": 3,
        "correlation": 4,
    }
    assert summary["result"]["flag_count"] == 0
    assert summary["correlation"]["labels"] == ["XX", "XY", "YX", "YY"]
    assert summary["execution"]["offline"] is True
    assert summary["backend"]["requested_backend"] == "numpy"
    assert summary["backend"]["actual_backend"] == "numpy-cpu"
    assert summary["backend"]["device_kind"] == "cpu"

    with h5py.File(_published_runs["hdf5"] / "visibilities.h5", "r") as handle:
        hdf5_digest = handle.attrs["scientific_sha256"]
    hdf5_digest = (
        hdf5_digest.decode() if isinstance(hdf5_digest, bytes) else str(hdf5_digest)
    )
    assert summary["result"]["scientific_sha256"] == hdf5_digest, (
        "two CLI runs of one document in two result formats disagree about the "
        "science they contain"
    )


def test_the_resolved_configuration_artifact_round_trips(
    _published_runs: dict[str, Path],
) -> None:
    """The run records the document it actually resolved, not the one supplied."""
    run = _published_runs["hdf5"]
    resolved = yaml.safe_load(
        (run / "resolved-config.yaml").read_text(encoding="utf-8")
    )
    runtime = resolved["scientific_runtime"]
    assert runtime["execution"]["offline"] is True
    assert runtime["execution"]["backend_strategy"] == "numpy"
    assert runtime["execution"]["simulator"] == "rime"
    assert runtime["sky_model"]["sources"][0]["kind"] == "test_sources"
    assert resolved["workflow"]["result_format"] == "hdf5"
    assert resolved["workflow"]["run_subdir"] == "hdf5-run"
    assert resolved["provenance"]["source"]


def test_an_unreadable_configuration_fails_the_process_with_a_message(
    tmp_path: Path,
) -> None:
    """The CLI's failure path is a nonzero exit and a message, not a traceback."""
    document = tmp_path / "broken.yaml"
    document.write_text(
        "instrument: {source: {kind: no_such_source}}\n", encoding="utf-8"
    )

    completed = subprocess.run(
        [*_cli_command(), "--config", str(document)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode != 0
    assert (completed.stdout + completed.stderr).strip()
    assert not (tmp_path / "output").exists(), "a failed run published a directory"
