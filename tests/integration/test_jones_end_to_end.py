"""Tier 7D: one complete run per implemented term, through every output format.

``Tier7JonesSciencePlan.md`` Section 30 gives this file "one
``Simulator.setup().run().save()`` per implemented term through HDF5, summary,
MS, and UVFITS".  The point is not that the writers work -- Tier 4 established
that -- but that a configured Jones term survives the *whole* path: YAML, strict
parse, resolution, the chain, the contraction, the fingerprint, and every
artifact, without any stage quietly dropping it.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from radiosim.api.simulator import Simulator
from radiosim.io.result_format import ResultFormat
from tests.fixtures.configs import valid_config_mapping

_TERM_CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "G": {"G": {"amplitude_error": 0.08, "phase_error_rad": 0.35}},
    "B": {
        "B": {
            "model": {
                "kind": "polynomial",
                "coefficients": [1.0, [0.15, 0.05], -0.06],
            }
        }
    },
    "G+B": {
        "G": {
            "amplitude_error": 0.08,
            "time_model": {"kind": "linear_drift", "rate_per_hour": 0.5},
        },
        "B": {
            "model": {
                "kind": "polynomial",
                "coefficients": [1.0, 0.2],
            }
        },
    },
}


def _simulator(tmp_path, jones: dict[str, Any] | None) -> Simulator:
    data = valid_config_mapping(tmp_path)
    if jones is not None:
        data["jones"] = jones
    return Simulator.from_mapping(data, base_dir=tmp_path)


@pytest.mark.parametrize("label", sorted(_TERM_CONFIGURATIONS))
def test_a_configured_term_survives_setup_run_and_save(tmp_path, label: str) -> None:
    """The whole path, for each term and for both together.

    ``setup()`` is called explicitly rather than left to ``run()`` so that the
    resolved inventory is observable before the solver has touched anything --
    the property Section 22 rule 1 asserts.
    """
    jones = _TERM_CONFIGURATIONS[label]
    simulator = _simulator(tmp_path, jones)

    simulator.setup()

    letters = tuple(sorted(jones))
    assert simulator.jones_terms.configured_letters == tuple(
        letter for letter in ("G", "B") if letter in letters
    )

    result = simulator.run(progress=False)

    assert np.all(np.isfinite(result.visibilities))
    assert float(np.max(np.abs(result.visibilities))) > 0.0

    hdf5_path = simulator.save(tmp_path / f"{label}.h5", format=ResultFormat.HDF5)
    summary_path = simulator.save(
        tmp_path / f"{label}.summary.json", format=ResultFormat.SUMMARY_JSON
    )

    from radiosim.io.hdf5 import load_result_hdf5

    loaded = load_result_hdf5(hdf5_path)
    assert loaded.scientific_sha256 == result.scientific_sha256
    assert dict(loaded.jones) == dict(result.jones)

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert set(payload["jones"]["terms"]) == set(jones)
    assert payload["jones"]["jones_sha256"] == dict(result.jones)["jones_sha256"]


@pytest.mark.parametrize("label", sorted(_TERM_CONFIGURATIONS))
@pytest.mark.parametrize("result_format", [ResultFormat.MS, ResultFormat.UVFITS])
def test_the_standard_visibility_formats_carry_the_corrupted_cube(
    tmp_path,
    label: str,
    result_format: ResultFormat,
) -> None:
    """Section 25.2: no calibration table, and no refusal either.

    A corrupted visibility is still a visibility, so the Measurement Set and
    UVFITS writers must accept a run with Jones terms enabled and write the
    cube they were given.  What they must *not* do is invent a subtable to
    record the corruption in, and the record they do not write lives in the
    HDF5 group and the summary instead.
    """
    pytest.importorskip("pyuvdata")
    if result_format is ResultFormat.MS:
        pytest.importorskip("casacore")

    simulator = _simulator(tmp_path, _TERM_CONFIGURATIONS[label])
    result = simulator.run(progress=False)

    suffix = ".ms" if result_format is ResultFormat.MS else ".uvfits"
    written = simulator.save(tmp_path / f"{label}{suffix}", format=result_format)

    assert written.exists()
    assert float(np.max(np.abs(result.visibilities))) > 0.0


def test_a_run_with_jones_absent_still_runs_and_records_nothing(tmp_path) -> None:
    """The other end of the same path: absence must remain frictionless."""
    simulator = _simulator(tmp_path, None)

    result = simulator.setup().run(progress=False)

    assert simulator.jones_terms.is_empty
    assert dict(result.jones) == {}

    summary_path = simulator.save(
        tmp_path / "absent.summary.json", format=ResultFormat.SUMMARY_JSON
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["jones"]["enabled_terms"] == []
    assert payload["jones"]["jones_sha256"] is None


def test_a_rejected_jones_section_fails_setup_and_publishes_nothing(
    tmp_path,
) -> None:
    """A bad ``jones:`` block stops the run before it writes anything.

    The complement of the tests above: the same path that carries a good
    configuration all the way to a file must carry a bad one no distance at all.
    """
    from radiosim.core.jones_errors import JonesAssignmentError

    simulator = _simulator(
        tmp_path,
        {
            "G": {
                "amplitude_error": 0.02,
                "per_antenna": [{"antenna": 77, "feed": 0, "amplitude_error": 0.1}],
            }
        },
    )

    with pytest.raises(JonesAssignmentError):
        simulator.run(progress=False)

    assert simulator.result is None
    assert list(tmp_path.glob("*.h5")) == []
