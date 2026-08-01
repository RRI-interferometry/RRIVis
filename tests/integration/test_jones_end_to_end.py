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
    "D": {
        "D": {
            "d_terms": {"kind": "explicit", "d0": [0.03, 0.01], "d1": [-0.02, 0.04]},
            "per_antenna": [
                {
                    "antenna": 1,
                    "feed": 0,
                    "d_term": {"kind": "ixr", "ixr_db": 25.0, "phase_rad": 0.4},
                }
            ],
        }
    },
    "X": {"X": {"phase_rad": 0.42, "delay_s": 5.0e-9}},
    "Kd": {
        "Kd": {
            "delay_s": 3.0e-9,
            "per_antenna": [{"antenna": 0, "feed": 1, "delay_s": -8.0e-9}],
        }
    },
    "Rc": {"Rc": {"amplitude": 0.06, "cable_delay_s": 1.5e-7, "phase_rad": 0.2}},
    "P": {"P": {"enabled": True}},
    # Tier 7G.  ``T`` carries both of its halves and ``Z`` both of its own, so a
    # run that silently dropped one would not survive the digest comparison
    # below.  ``minimum_elevation_deg`` is 1 degree rather than the plan's
    # example 5: the shipped workload is a point sky near zenith, so nothing is
    # excluded either way, and a value a user would plausibly write is more
    # useful in a shipped example than a value chosen to be safe.
    "T": {
        "T": {
            "zenith_delay": {
                "kind": "saastamoinen",
                "surface_pressure_hpa": 1013.25,
                "zenith_wet_delay_m": 0.15,
            },
            "mapping_function": "niell",
            "minimum_elevation_deg": 1.0,
            "opacity": {"zenith_opacity": 0.05},
        }
    },
    "Z": {
        "Z": {
            "tec": {
                "kind": "gradient",
                "vertical_tec_tecu": 25.0,
                "gradient_east_tecu_per_km": 0.3,
            },
            "shell_height_km": 300.0,
            "minimum_elevation_deg": 1.0,
            "faraday": {
                "rotation_measure_rad_m2": 1.2,
                "per_antenna": [{"antenna": 1, "rotation_measure_rad_m2": -0.6}],
            },
        }
    },
    "all": {
        "G": {"amplitude_error": 0.05},
        "B": {"model": {"kind": "polynomial", "coefficients": [1.0, 0.1]}},
        "Rc": {"amplitude": 0.04, "cable_delay_s": 2.0e-7},
        "Kd": {"delay_s": 2.0e-9},
        "X": {"phase_rad": 0.3},
        "D": {"d_terms": {"kind": "ixr", "ixr_db": 30.0}},
        "P": {"enabled": True},
        "T": {
            "zenith_delay": {"kind": "explicit", "zenith_hydrostatic_delay_m": 2.3},
            "minimum_elevation_deg": 1.0,
            "opacity": {"zenith_opacity": 0.02},
        },
        "Z": {
            "tec": {"kind": "constant", "vertical_tec_tecu": 15.0},
            "minimum_elevation_deg": 1.0,
            "faraday": {"rotation_measure_rad_m2": 0.8},
        },
    },
}

#: The mount types each configuration needs on the resolved instrument.  Only
#: ``P`` needs anything: every other term's physics is in the document, while
#: ``P``'s is in the instrument, and on the shipped fixture -- whose layout file
#: carries no mount column -- every mount is unspecified and therefore
#: non-rotating, which makes ``jones.P`` exactly the identity and rejection R7
#: refuse it.
_MOUNT_TYPES: dict[str, str | None] = {"P": "alt-az", "all": "alt-az"}

#: The configurable term letters, in canonical chain order -- the order the
#: resolved inventory must report regardless of how the document was written.
#: ``P`` sits after ``D`` because Tier 7F moved it sky-side of ``C``
#: (``Tier7JonesSciencePlan.md`` Section 12.2, defect D12).
#: After Tier 7G this is *every* configurable term: the only two letters left
#: out are ``M`` and ``Q``, which are baseline-dependent and not chain terms.
_CANONICAL_LETTERS: tuple[str, ...] = (
    "G",
    "B",
    "Rc",
    "Kd",
    "X",
    "D",
    "P",
    "T",
    "Z",
)


def _simulator(
    tmp_path,
    jones: dict[str, Any] | None,
    *,
    mount_types: str | None = None,
) -> Simulator:
    from tests.unit.test_core.test_jones_resolution import restamp_mount_types

    data = valid_config_mapping(tmp_path)
    if jones is not None:
        data["jones"] = jones
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    if mount_types is not None:
        simulator._ensure_instrument_state()
        restamp_mount_types(simulator, mount_types)
    return simulator


@pytest.mark.parametrize("label", sorted(_TERM_CONFIGURATIONS))
def test_a_configured_term_survives_setup_run_and_save(tmp_path, label: str) -> None:
    """The whole path, for each term and for both together.

    ``setup()`` is called explicitly rather than left to ``run()`` so that the
    resolved inventory is observable before the solver has touched anything --
    the property Section 22 rule 1 asserts.
    """
    jones = _TERM_CONFIGURATIONS[label]
    simulator = _simulator(tmp_path, jones, mount_types=_MOUNT_TYPES.get(label))

    simulator.setup()

    assert simulator.jones_terms.configured_letters == tuple(
        letter for letter in _CANONICAL_LETTERS if letter in jones
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

    simulator = _simulator(
        tmp_path, _TERM_CONFIGURATIONS[label], mount_types=_MOUNT_TYPES.get(label)
    )
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
