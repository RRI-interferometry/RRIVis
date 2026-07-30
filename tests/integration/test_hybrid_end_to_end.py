"""Tier 6F: hybrid solve mode end to end, from YAML to one canonical result.

Covers ``Tier6HybridRuntimePlan.md`` Section 27 rows ``H1`` (config-level
additivity), ``H2``, ``H5``, ``H6``, and ``H7``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

from radiosim.api import Simulator
from radiosim.core.result import SimulationResult
from tests.fixtures.configs import hybrid_config_mapping

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(tmp_path: Path, component: str) -> SimulationResult:
    directory = tmp_path / component
    directory.mkdir(exist_ok=True)
    data = hybrid_config_mapping(directory, component=component)
    return Simulator.from_mapping(data, base_dir=directory).run(progress=False)


def test_hybrid_run_equals_the_sum_of_two_single_component_runs(tmp_path) -> None:
    """H1/S1 at configuration level: bit-identical additivity on NumPy."""
    hybrid = _run(tmp_path, "hybrid")
    point = _run(tmp_path, "point")
    healpix = _run(tmp_path, "healpix")

    expected = point.visibilities + healpix.visibilities
    assert hybrid.visibilities.dtype == expected.dtype
    assert hybrid.visibilities.shape == expected.shape
    assert hybrid.visibilities.tobytes() == expected.tobytes()

    # Neither component is a no-op, so the equality is not vacuous.
    assert np.any(point.visibilities != 0)
    assert np.any(healpix.visibilities != 0)
    assert hybrid.visibilities.tobytes() != point.visibilities.tobytes()
    assert hybrid.visibilities.tobytes() != healpix.visibilities.tobytes()


def test_hybrid_coordinates_match_both_single_component_runs(tmp_path) -> None:
    """H2/S2: one shared time grid, frequency axis, and baseline selection."""
    hybrid = _run(tmp_path, "hybrid")
    point = _run(tmp_path, "point")
    healpix = _run(tmp_path, "healpix")

    for other in (point, healpix):
        assert hybrid.time_grid.start_time_iso == other.time_grid.start_time_iso
        assert hybrid.time_grid.cadence_seconds == other.time_grid.cadence_seconds
        assert len(hybrid.time_grid) == len(other.time_grid)
        assert np.array_equal(hybrid.frequencies_hz, other.frequencies_hz)
        assert np.array_equal(hybrid.channel_widths_hz, other.channel_widths_hz)
        assert hybrid.correlations == other.correlations
        assert hybrid.polarization_basis == other.polarization_basis
        assert [(b.ant1, b.ant2) for b in hybrid.selection.baselines] == [
            (b.ant1, b.ant2) for b in other.selection.baselines
        ]


def test_hybrid_publishes_exactly_one_canonical_result(tmp_path) -> None:
    """Section 9.1: one result object, one fingerprint, no second cube."""
    directory = tmp_path / "hybrid"
    directory.mkdir(exist_ok=True)
    simulator = Simulator.from_mapping(
        hybrid_config_mapping(directory), base_dir=directory
    )
    result = simulator.run(progress=False)

    assert type(result) is SimulationResult
    assert result is simulator.result
    assert not hasattr(simulator, "results")
    assert result.visibilities.ndim == 4
    assert result.visibilities.shape[-1] == 4


def test_hybrid_provenance_records_components_and_true_counts(tmp_path) -> None:
    """H6: representation, component list, and per-component element counts."""
    directory = tmp_path / "hybrid"
    directory.mkdir(exist_ok=True)
    simulator = Simulator.from_mapping(
        hybrid_config_mapping(directory), base_dir=directory
    )
    result = simulator.run(progress=False)

    assert result.solver.sky_representation == "hybrid"
    assert result.solver.components == ("point", "healpix")
    assert result.solver.component_element_counts == (
        simulator._sky_model.n_point_sources,
        simulator._sky_model.n_healpix_pixels,
    )
    assert result.solver.execution_path == "polarized"

    snapshot = result.solver.to_snapshot()
    assert tuple(snapshot["components"]) == ("point", "healpix")
    assert tuple(snapshot["component_element_counts"]) == (
        simulator._sky_model.n_point_sources,
        simulator._sky_model.n_healpix_pixels,
    )


def test_single_component_runs_declare_one_component(tmp_path) -> None:
    point = _run(tmp_path, "point")
    healpix = _run(tmp_path, "healpix")

    assert point.solver.components == ("point",)
    assert point.solver.component_element_counts[0] > 0
    assert point.performance.solver_healpix_seconds == 0.0
    assert healpix.solver.components == ("healpix",)
    assert healpix.solver.component_element_counts[0] > 0
    assert healpix.performance.solver_point_seconds == 0.0


def test_hybrid_component_timings_are_positive_and_bounded(tmp_path) -> None:
    """H7: both components are timed, and neither invents time."""
    hybrid = _run(tmp_path, "hybrid")
    performance = hybrid.performance

    assert performance.solver_point_seconds > 0.0
    assert performance.solver_healpix_seconds > 0.0
    assert (
        performance.solver_point_seconds + performance.solver_healpix_seconds
        <= performance.solver_seconds
    )
    # Timings are nondeterministic and must stay out of both fingerprints.
    assert "solver_point_seconds" not in repr(hybrid.to_summary_snapshot())


def test_hybrid_stokes_i_is_the_sum_of_the_component_stokes_i(tmp_path) -> None:
    """H5/S5: a disjoint hybrid model does not double count."""
    hybrid = _run(tmp_path, "hybrid")
    point = _run(tmp_path, "point")
    healpix = _run(tmp_path, "healpix")

    def stokes_i(result: SimulationResult) -> np.ndarray:
        xx = result.visibilities[..., result.correlations.index("XX")]
        yy = result.visibilities[..., result.correlations.index("YY")]
        return xx + yy

    assert np.array_equal(stokes_i(hybrid), stokes_i(point) + stokes_i(healpix))


def test_assume_disjoint_warns_and_still_enforces_monopole_consistency(
    tmp_path,
) -> None:
    """H5's second half: the escape is narrow and says so."""
    directory = tmp_path / "assume"
    directory.mkdir(exist_ok=True)
    data = hybrid_config_mapping(directory)
    data["sky_model"]["mixed_model_policy"] = "error"
    data["sky_model"]["assume_disjoint"] = True

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = Simulator.from_mapping(data, base_dir=directory).run(progress=False)

    messages = [str(entry.message) for entry in caught]
    assert any("assume_disjoint" in message for message in messages)
    assert any("monopole" in message.lower() for message in messages)
    assert result.solver.components == ("point", "healpix")


def test_shipped_hybrid_example_runs_and_reports_both_components(tmp_path) -> None:
    """The shipped configuration is a real hybrid run, not an illustration."""
    document = yaml.safe_load(
        (REPO_ROOT / "configs" / "hybrid_sky_example.yaml").read_text(encoding="utf-8")
    )
    workflow = dict(document.get("workflow") or {})
    workflow["output_dir"] = str(tmp_path)
    workflow["save_results"] = False
    workflow["save_log"] = False
    workflow["plot_results"] = False
    document["workflow"] = workflow

    assert document["visibility"]["sky_representation"] == "hybrid"
    result = Simulator.from_mapping(document, base_dir=REPO_ROOT / "configs").run(
        progress=False
    )

    assert result.solver.sky_representation == "hybrid"
    assert result.solver.components == ("point", "healpix")
    assert result.solver.component_element_counts[0] == 20
    assert result.solver.component_element_counts[1] == 12 * 16 * 16
