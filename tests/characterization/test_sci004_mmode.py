"""SCI-004 phase-M3 characterization pins for the new m-mode families.

``docs/development/sci004_mmode_design.md`` Section 11 names the seven families
this phase adds:

.. code-block:: text

    mmode_single_scalar_mode
    mmode_point_stokes_i
    mmode_healpix_stokes_i
    mmode_point_full_stokes
    mmode_healpix_full_stokes
    mmode_hybrid_full_stokes
    mmode_nonscalar_east_x

and fixes what each one has to record: "Each family records the raw cube,
``scientific_sha256``, solver snapshot, ERA/UTC grid, harmonic index table, and
input identity. A changed m-mode pin requires old/new cubes and an
equation-level explanation; no digest is appended merely because CI printed
it."  Section 12.2's tenth oracle family asks for "every new family, unchanged
direct pins, two dispatch classes where applicable, exact-SHA remote artifacts,
and release scans that continue to say ``SCI-004`` is ROADMAP until closure",
and the accepted phase-M2 acceptance record reserves exactly this ground: its
``claims_not_licensed`` array opens with "a retained characterization/fingerprint
pin for any M2 family (that is Phase M3's scope)".

**What is red here, and why.** No production surface assembles a family record
today, so every family oracle below fails on the absent
``radiosim.core.result.mmode_characterization_record`` import before it runs
anything. That import is deliberately the first statement of each oracle: the
capability is what is missing, and paying a full m-mode solve to observe an
absence that is already decidable would make the red slice slower without
making it truer. Once the surface exists, each oracle runs its family's
configuration through the public API and pins what Section 11 lists.

**Why the pins are observation sets, not bare digests.** Section 11 continues:
"The accepted CI-001 successor discipline applies to every new family. All six
platform/Python cells and every already recognized NumPy/OpenBLAS dispatch class
are harvested. A novel class is adjudicated by cubes under Section 9's fixed
complex128 predicate before it can join an observation set."  A digest harvested
on one developer machine is therefore not a pin, and this module never writes
one: it requires the pin surface to *be* an observation set and leaves the
harvest to the phase's evidence and acceptance stages.

**Why the family dimensions are the accepted ones.** Section 7.3's every-run
two-tier gate is unforgiving, and its qualified truncation is the accepted
fixture's: the phase-M2 integration fixture records that ``lmax = 16`` "is
pinned by the accepted evidence, not chosen for convenience".  A measurement
taken while authoring this slice confirms that reading -- the same geometry at
``lmax = mmax = 8``, ``quadrature_nside = 4`` and ``33`` sidereal samples fails
the tier-1a horizon-free shell at ``3.480803e-06`` Jy against its
``2.261455e-08`` Jy limit -- so the family fixtures below reuse the accepted
``49`` samples, ``lmax = mmax = 16`` and ``quadrature_nside = 8`` rather than a
cheaper set nobody has qualified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

#: Section 11's seven new characterized families, in the order the memo prints
#: them. Production must declare exactly this tuple.
SECTION_11_FAMILIES: tuple[str, ...] = (
    "mmode_single_scalar_mode",
    "mmode_point_stokes_i",
    "mmode_healpix_stokes_i",
    "mmode_point_full_stokes",
    "mmode_healpix_full_stokes",
    "mmode_hybrid_full_stokes",
    "mmode_nonscalar_east_x",
)

#: Section 11's six recorded parts, plus the family identifier that joins them.
FAMILY_RECORD_KEYS: frozenset[str] = frozenset(
    {
        "family_id",
        "raw_cube_sha256",
        "scientific_sha256",
        "solver_snapshot",
        "era_utc_grid_sha256",
        "harmonic_index_table_sha256",
        "input_identity_sha256",
    }
)

#: The accepted phase-M2 fixture's qualified truncation and geometry.
FAMILY_SIDEREAL_SAMPLES = 49
FAMILY_LMAX = 16
FAMILY_MMAX = 16
FAMILY_QUADRATURE_NSIDE = 8
FAMILY_BASELINE_EAST_M = 4.0
FAMILY_DISH_DIAMETER_M = 2.5
FAMILY_SOURCE_DEC_DEG = -75.0
FAMILY_FREQUENCY_HZ = 50e6
FAMILY_CHANNEL_WIDTH_HZ = 1e6
FAMILY_WORKING_MEMORY_BYTES = 1 << 30

MMODE_CONVENTION = "radiosim.mmode-forward.v1"
TANGENT_FRAME_SCHEMA = "radiosim.sky-tangent-polarization.v1"

#: Section 5.1's exact six-key canonical tangent-polarization block. A polarized
#: m-mode input is rejected without it, so every full-Stokes family declares it.
CANONICAL_TANGENT_FRAME: dict[str, str] = {
    "schema_version": TANGENT_FRAME_SCHEMA,
    "coordinate_frame": "icrs",
    "axes": "north_east",
    "position_angle": "north_through_east",
    "linear_complex": "q_plus_i_u",
    "stokes_v": "iau_incoming_r_minus_l",
}
MMODE_FRAME_MODEL = "radiosim.frozen-cirs-rigid-era.v1"
MMODE_HARMONIC_CONVENTION = "radiosim.shaw-polarized-harmonics.v1"

#: Each family's sky payload and receptor declaration. The dimensions, geometry
#: and time grid are shared, so a family is exactly its input identity.
_FAMILY_SKY: dict[str, dict[str, Any]] = {
    "mmode_single_scalar_mode": {
        "sources": [
            {
                "kind": "test_sources",
                "representation": "point_sources",
                "num_sources": 1,
                "distribution": "uniform",
                "seed": 1,
                "dec_deg": FAMILY_SOURCE_DEC_DEG,
                "dec_range_deg": 0.0,
                "spectral_index": 0.0,
                "polarization_fraction": 0.0,
                "stokes_v_fraction": 0.0,
            }
        ]
    },
    "mmode_point_stokes_i": {
        "sources": [
            {
                "kind": "test_sources",
                "representation": "point_sources",
                "num_sources": 3,
                "distribution": "uniform",
                "seed": 2,
                "dec_deg": FAMILY_SOURCE_DEC_DEG,
                "dec_range_deg": 0.0,
                "spectral_index": 0.0,
                "polarization_fraction": 0.0,
                "stokes_v_fraction": 0.0,
            }
        ]
    },
    "mmode_healpix_stokes_i": {
        "sources": [
            {
                "kind": "test_sources",
                "representation": "healpix_map",
                "num_sources": 3,
                "distribution": "uniform",
                "seed": 3,
                "nside": 16,
                "dec_deg": FAMILY_SOURCE_DEC_DEG,
                "dec_range_deg": 0.0,
                "spectral_index": 0.0,
                "polarization_fraction": 0.0,
                "stokes_v_fraction": 0.0,
            }
        ]
    },
    "mmode_point_full_stokes": {
        "sources": [
            {
                "kind": "test_sources",
                "representation": "point_sources",
                "num_sources": 3,
                "distribution": "uniform",
                "seed": 4,
                "dec_deg": FAMILY_SOURCE_DEC_DEG,
                "dec_range_deg": 0.0,
                "spectral_index": 0.0,
                "polarization_fraction": 0.2,
                "stokes_v_fraction": 0.1,
                "tangent_polarization_frame": CANONICAL_TANGENT_FRAME,
            }
        ]
    },
    "mmode_healpix_full_stokes": {
        "sources": [
            {
                "kind": "test_sources",
                "representation": "healpix_map",
                "num_sources": 3,
                "distribution": "uniform",
                "seed": 5,
                "nside": 16,
                "dec_deg": FAMILY_SOURCE_DEC_DEG,
                "dec_range_deg": 0.0,
                "spectral_index": 0.0,
                "polarization_fraction": 0.2,
                "stokes_v_fraction": 0.1,
                "tangent_polarization_frame": CANONICAL_TANGENT_FRAME,
            }
        ]
    },
    "mmode_hybrid_full_stokes": {
        "sources": [
            {
                "kind": "test_sources",
                "representation": "point_sources",
                "num_sources": 2,
                "distribution": "uniform",
                "seed": 6,
                "dec_deg": FAMILY_SOURCE_DEC_DEG,
                "dec_range_deg": 0.0,
                "spectral_index": 0.0,
                "polarization_fraction": 0.2,
                "stokes_v_fraction": 0.1,
                "tangent_polarization_frame": CANONICAL_TANGENT_FRAME,
            },
            {
                "kind": "test_sources",
                "representation": "healpix_map",
                "num_sources": 2,
                "distribution": "uniform",
                "seed": 7,
                "nside": 16,
                "dec_deg": FAMILY_SOURCE_DEC_DEG,
                "dec_range_deg": 0.0,
                "spectral_index": 0.0,
                "polarization_fraction": 0.2,
                "stokes_v_fraction": 0.1,
                "tangent_polarization_frame": CANONICAL_TANGENT_FRAME,
            },
        ]
    },
    "mmode_nonscalar_east_x": {
        "sources": [
            {
                "kind": "test_sources",
                "representation": "point_sources",
                "num_sources": 3,
                "distribution": "uniform",
                "seed": 8,
                "dec_deg": FAMILY_SOURCE_DEC_DEG,
                "dec_range_deg": 0.0,
                "spectral_index": 0.0,
                "polarization_fraction": 0.2,
                "stokes_v_fraction": 0.1,
                "tangent_polarization_frame": CANONICAL_TANGENT_FRAME,
            }
        ]
    },
}

#: The east-X family declares the SCI-006 receptor convention explicitly; every
#: other family takes the shipped default.
_FAMILY_RECEPTORS: dict[str, dict[str, Any]] = {
    "mmode_nonscalar_east_x": {"default": {"basis": "linear", "feed_rotation_deg": 0.0}}
}


def family_mapping(tmp_path: Path, family_id: str) -> dict[str, Any]:
    """Return one family's complete configuration mapping.

    The geometry, truncation and time grid are the accepted phase-M2 fixture's;
    only the sky payload and the receptor declaration vary, so two families
    differ exactly in their input identity.
    """
    if family_id not in SECTION_11_FAMILIES:
        raise KeyError(family_id)
    tmp_path.mkdir(parents=True, exist_ok=True)
    layout = tmp_path / f"{family_id}-antennas.txt"
    layout.write_text(
        "Name Number BeamID E N U Diameter\n"
        f"ANT0 0 0 0.0 0.0 0.0 {FAMILY_DISH_DIAMETER_M}\n"
        f"ANT1 1 0 {FAMILY_BASELINE_EAST_M} 0.0 0.0 {FAMILY_DISH_DIAMETER_M}\n",
        encoding="utf-8",
    )
    mapping: dict[str, Any] = {
        "instrument": {
            "source": {
                "kind": "layout_file",
                "path": str(layout),
                "format": "radiosim",
                "telescope_name": "SCI-004 M3 family array",
            },
            "location": {
                "longitude_deg": 21.4,
                "latitude_deg": -30.7,
                "height_m": 1000.0,
            },
            "default_diameter_m": FAMILY_DISH_DIAMETER_M,
        },
        "baseline_selection": {"correlations": "all"},
        "beams": {
            "mode": "analytic",
            "model": {
                "kind": "circular_aperture",
                "taper": {"kind": "gaussian", "edge_taper_db": 10.0},
            },
        },
        "obs_time": {
            "mode": "full_sidereal",
            "start_time": "2025-01-01T00:00:00",
            "sidereal_samples": FAMILY_SIDEREAL_SAMPLES,
            "integration_fraction": 1.0,
        },
        "obs_frequency": {
            "mode": "explicit",
            "channel_frequencies_hz": [FAMILY_FREQUENCY_HZ],
            "channel_widths_hz": [FAMILY_CHANNEL_WIDTH_HZ],
        },
        "sky_model": {"flux_unit": "Jy", **_FAMILY_SKY[family_id]},
        "execution": {
            "backend": "numpy",
            "offline": True,
            "simulator": "mmode",
            "mmode": {
                "convention": MMODE_CONVENTION,
                "frame_model": MMODE_FRAME_MODEL,
                "harmonic_convention": MMODE_HARMONIC_CONVENTION,
                "lmax": FAMILY_LMAX,
                "mmax": FAMILY_MMAX,
                "quadrature_nside": FAMILY_QUADRATURE_NSIDE,
                "working_memory_bytes": FAMILY_WORKING_MEMORY_BYTES,
            },
        },
    }
    receptors = _FAMILY_RECEPTORS.get(family_id)
    if receptors is not None:
        mapping["receptors"] = receptors
    return mapping


#: The exact retained bytes of the family inventory this slice pins. The red
#: record hashes them as each case's ``invalid_config_raw_sha256``, so the
#: record names the inventory the observation was made against.
FAMILY_INVENTORY_BYTES = ("\n".join(SECTION_11_FAMILIES) + "\n").encode("utf-8")

_PHASE3_FAMILY_GREEN_CONTROL = (
    "tests/characterization/test_sci004_mmode.py::"
    "test_every_family_configuration_resolves_to_the_mmode_strategy"
    "[mmode_hybrid_full_stokes]"
)


def _phase3_family_case(
    case_id: str,
    requirement_id: str,
    function: str,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "requirement_id": requirement_id,
        "test_nodeid": f"tests/characterization/test_sci004_mmode.py::{function}",
        "expected_failure_kind": "missing-symbol",
        "expected_failure_pattern": (
            r"cannot import name "
            r"'(MMODE_CHARACTERIZATION_FAMILIES|mmode_characterization_record)' "
            r"from 'radiosim\.core\.result'"
        ),
        "fixture_defect_excluded_by": _PHASE3_FAMILY_GREEN_CONTROL,
        "fixture_bytes": FAMILY_INVENTORY_BYTES,
    }


SCI004_PHASE3_RED_CASES: tuple[dict[str, object], ...] = (
    _phase3_family_case(
        "m3.characterization.family-inventory",
        "sci004.section-11.seven-new-characterized-families",
        "test_production_declares_exactly_the_seven_section_11_families",
    ),
    *(
        _phase3_family_case(
            f"m3.characterization.family-record.{family_id}",
            "sci004.section-11.family-records-its-six-parts",
            (f"test_every_new_family_records_its_six_section_11_parts[{family_id}]"),
        )
        for family_id in SECTION_11_FAMILIES
    ),
    _phase3_family_case(
        "m3.characterization.dispatch-class-observation-set",
        "sci004.section-11.ci001-observation-set-not-a-bare-digest",
        "test_a_family_pin_is_a_ci001_observation_set_not_a_bare_digest",
    ),
)

SCI004_PHASE3_RED_GREEN_CONTROLS: tuple[str, ...] = (_PHASE3_FAMILY_GREEN_CONTROL,)


# --- green control ------------------------------------------------------------


@pytest.mark.parametrize("family_id", SECTION_11_FAMILIES)
def test_every_family_configuration_resolves_to_the_mmode_strategy(
    tmp_path: Path,
    family_id: str,
) -> None:
    """The seven family fixtures are valid m-mode inputs at this tip.

    This is the fixture-defect exclusion the red record cites: the oracles below
    fail because no production surface assembles a family record, not because
    the configurations they would run are malformed. It deliberately stops at
    resolution -- Section 7.3's every-run gate makes a full solve of all seven
    families minutes of work, and resolution is what excludes a defective
    fixture for a missing-symbol failure.
    """
    from radiosim.api.simulator import Simulator

    simulator = Simulator.from_mapping(
        family_mapping(tmp_path, family_id), base_dir=tmp_path
    )

    assert simulator.config.execution.simulator == "mmode"
    assert simulator.config.execution.mmode is not None
    assert simulator.config.execution.mmode.lmax == FAMILY_LMAX
    assert simulator.config.execution.mmode.mmax == FAMILY_MMAX
    assert len(simulator.config.observation.time_grid) == FAMILY_SIDEREAL_SAMPLES


# --- Section 11 family oracles ------------------------------------------------


def test_production_declares_exactly_the_seven_section_11_families() -> None:
    """Section 11 names seven new characterized families, and only seven."""
    from radiosim.core.result import (  # noqa: F401
        MMODE_CHARACTERIZATION_FAMILIES,
        mmode_characterization_record,
    )

    assert tuple(MMODE_CHARACTERIZATION_FAMILIES) == SECTION_11_FAMILIES
    assert len(set(MMODE_CHARACTERIZATION_FAMILIES)) == 7


@pytest.mark.parametrize("family_id", SECTION_11_FAMILIES)
def test_every_new_family_records_its_six_section_11_parts(
    tmp_path: Path,
    family_id: str,
) -> None:
    """Section 11: "Each family records the raw cube, ``scientific_sha256``,
    solver snapshot, ERA/UTC grid, harmonic index table, and input identity."

    The record's digests are required to join the result they came from, so a
    family pin cannot quietly describe a different run: the scientific
    fingerprint is the result's own, the snapshot is the exact Section 10
    twenty-key m-mode arm, and the harmonic index table follows the family's
    ``(lmax, mmax)``.
    """
    from radiosim.api.simulator import Simulator
    from radiosim.core.result import (
        MMODE_SOLVER_SNAPSHOT_KEYS,
        mmode_characterization_record,
    )

    result = Simulator.from_mapping(
        family_mapping(tmp_path, family_id), base_dir=tmp_path
    ).run(progress=False)
    record = mmode_characterization_record(result, family_id=family_id)

    assert set(record) == FAMILY_RECORD_KEYS
    assert record["family_id"] == family_id
    assert record["scientific_sha256"] == result.scientific_sha256
    snapshot = record["solver_snapshot"]
    assert tuple(snapshot) == MMODE_SOLVER_SNAPSHOT_KEYS
    assert snapshot["solver"] == "mmode"
    assert snapshot["lmax"] == FAMILY_LMAX
    assert snapshot["mmax"] == FAMILY_MMAX
    assert snapshot["sidereal_samples"] == FAMILY_SIDEREAL_SAMPLES
    for field in (
        "raw_cube_sha256",
        "era_utc_grid_sha256",
        "harmonic_index_table_sha256",
        "input_identity_sha256",
    ):
        value = record[field]
        assert isinstance(value, str) and len(value) == 64, field
        assert all(character in "0123456789abcdef" for character in value), field
    # A family record is deterministic: the same result yields the same pin.
    assert mmode_characterization_record(result, family_id=family_id) == record


def test_a_family_pin_is_a_ci001_observation_set_not_a_bare_digest() -> None:
    """Section 11: "The accepted CI-001 successor discipline applies to every new
    family. All six platform/Python cells and every already recognized
    NumPy/OpenBLAS dispatch class are harvested. A novel class is adjudicated by
    cubes under Section 9's fixed complex128 predicate before it can join an
    observation set."

    A single-machine digest is therefore not a pin, and the surface that holds
    the pins must say so in its own shape: an observation set keyed by dispatch
    class, never one string.
    """
    from radiosim.core.result import (
        MMODE_CHARACTERIZATION_FAMILIES,
        mmode_characterization_observation_set,
        mmode_characterization_record,  # noqa: F401
    )

    for family_id in MMODE_CHARACTERIZATION_FAMILIES:
        observations = mmode_characterization_observation_set(family_id)
        assert isinstance(observations, dict)
        assert observations, family_id
        for dispatch_class, digests in observations.items():
            assert isinstance(dispatch_class, str) and dispatch_class
            assert isinstance(digests, tuple)
            assert digests, dispatch_class
            for digest in digests:
                assert isinstance(digest, str) and len(digest) == 64
