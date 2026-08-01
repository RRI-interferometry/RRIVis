"""Tier 5B resolved-receptor precedence, rejection, and fingerprint contract."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from radiosim.core.instrument import (
    AntennaFieldSource,
    AntennaId,
    _compute_instrument_sha256,
)
from radiosim.core.instrument_resolution import resolve_instrument
from radiosim.core.receptor import (
    AmbiguousOutputBasisError,
    InvalidReceptorConfigError,
    ReceptorAssignmentError,
    ReceptorError,
    ReceptorProvenance,
    ResolvedReceptor,
    ResolvedReceptorSet,
    UnsupportedFeedGeometryError,
    UnsupportedReceptorBasisError,
    resolve_receptors,
)
from radiosim.io.instrument_config import InstrumentConfig
from radiosim.io.receptor_config import ReceptorsConfig

_NAMES = ("HERA-10", "HERA-11", "HERA-12", "HERA-13", "HERA-14")


def _instrument(tmp_path: Path, count: int = 5):
    tmp_path.mkdir(parents=True, exist_ok=True)
    layout = tmp_path / "receptor_layout.txt"
    rows = ["Name Number BeamID E N U Diameter"]
    for index in range(count):
        rows.append(f"{_NAMES[index]} {index} 0 {14.0 * index} 0.0 0.0 14.0")
    layout.write_text("\n".join(rows) + "\n")
    config = InstrumentConfig.model_validate(
        {
            "source": {
                "kind": "layout_file",
                "path": str(layout),
                "format": "radiosim",
                "telescope_name": "Tier5BReceptorArray",
            },
            "location": {
                "longitude_deg": 21.4283,
                "latitude_deg": -30.72152,
                "height_m": 1073.0,
            },
            "default_diameter_m": 14.0,
        }
    )
    return resolve_instrument(config)


def _with_mount_type(instrument, mount_type: str | None):
    antennas = tuple(
        replace(antenna, mount_type=mount_type) for antenna in instrument.antennas
    )
    sha256 = _compute_instrument_sha256(
        instrument.name,
        instrument.location,
        antennas,
        telescope_name_source=instrument.provenance.telescope_name_source,
        location_source=instrument.provenance.location_source,
    )
    return replace(
        instrument,
        antennas=antennas,
        provenance=replace(instrument.provenance, instrument_sha256=sha256),
    )


def _config(**payload: object) -> ReceptorsConfig:
    return ReceptorsConfig.model_validate(payload)


def test_default_resolution_reproduces_the_nominal_linear_array(tmp_path):
    instrument = _instrument(tmp_path)

    resolved = resolve_receptors(ReceptorsConfig(), instrument)

    assert type(resolved) is ResolvedReceptorSet
    assert resolved.output_basis == "linear_xy"
    assert [antenna_id.number for antenna_id in resolved.receptor_by_antenna] == [
        0,
        1,
        2,
        3,
        4,
    ]
    for receptor in resolved.receptor_by_antenna.values():
        assert type(receptor) is ResolvedReceptor
        assert receptor.basis == "linear"
        assert receptor.feed_rotation_rad == 0.0
        assert receptor.feed_array == ("x", "y")
        assert receptor.feed_angle_rad == (math.pi / 2.0, 0.0)
        assert receptor.source is AntennaFieldSource.CONFIG_DEFAULT


def test_homogeneous_circular_resolution_uses_the_nominal_circular_pair(tmp_path):
    instrument = _instrument(tmp_path)

    resolved = resolve_receptors(_config(default={"basis": "circular"}), instrument)

    assert resolved.output_basis == "circular_rl"
    for receptor in resolved.receptor_by_antenna.values():
        assert receptor.basis == "circular"
        assert receptor.feed_array == ("r", "l")
        assert receptor.feed_angle_rad == (0.0, 0.0)


@pytest.mark.parametrize(
    ("configured_deg", "expected_deg"),
    [
        (0.0, 0.0),
        (45.0, 45.0),
        (-15.0, -15.0),
        (180.0, 180.0),
        (-180.0, 180.0),
        (190.0, -170.0),
        (540.0, 180.0),
        (360.0, 0.0),
        (-450.0, -90.0),
    ],
)
def test_feed_rotation_is_normalized_into_the_half_open_turn(
    tmp_path, configured_deg, expected_deg
):
    instrument = _instrument(tmp_path)

    resolved = resolve_receptors(
        _config(default={"feed_rotation_deg": configured_deg}),
        instrument,
    )

    expected_rad = math.radians(expected_deg)
    for receptor in resolved.receptor_by_antenna.values():
        assert receptor.feed_rotation_rad == pytest.approx(expected_rad, abs=1e-15)
        assert -math.pi < receptor.feed_rotation_rad <= math.pi


@pytest.mark.parametrize("basis", ["linear", "circular"])
def test_feed_angles_follow_the_documented_offset_formula(tmp_path, basis):
    instrument = _instrument(tmp_path)
    chi = math.radians(30.0)

    resolved = resolve_receptors(
        _config(default={"basis": basis, "feed_rotation_deg": 30.0}),
        instrument,
    )

    expected = (math.pi / 2.0 + chi, chi) if basis == "linear" else (chi, chi)
    for receptor in resolved.receptor_by_antenna.values():
        assert receptor.feed_angle_rad == pytest.approx(expected, abs=1e-15)


def test_partial_override_replaces_only_the_declared_fields(tmp_path):
    instrument = _instrument(tmp_path)

    resolved = resolve_receptors(
        _config(
            default={"basis": "circular", "feed_rotation_deg": 10.0},
            overrides=[
                {
                    "antenna": {"kind": "number", "number": 3},
                    "feed_rotation_deg": 30.0,
                }
            ],
            output_basis="circular",
        ),
        instrument,
    )

    overridden = resolved.receptor_by_antenna[AntennaId(3, "HERA-13")]
    untouched = resolved.receptor_by_antenna[AntennaId(2, "HERA-12")]

    assert overridden.basis == "circular"
    assert overridden.feed_rotation_rad == pytest.approx(math.radians(30.0))
    assert overridden.source is AntennaFieldSource.EXPLICIT_OVERRIDE
    assert untouched.feed_rotation_rad == pytest.approx(math.radians(10.0))
    assert untouched.source is AntennaFieldSource.CONFIG_DEFAULT


def test_name_and_number_references_select_the_same_canonical_antenna(tmp_path):
    instrument = _instrument(tmp_path)

    by_number = resolve_receptors(
        _config(
            overrides=[
                {"antenna": {"kind": "number", "number": 1}, "feed_rotation_deg": 20.0}
            ]
        ),
        instrument,
    )
    by_name = resolve_receptors(
        _config(
            overrides=[
                {
                    "antenna": {"kind": "name", "name": "HERA-11"},
                    "feed_rotation_deg": 20.0,
                }
            ]
        ),
        instrument,
    )

    assert by_number.receptor_by_antenna == by_name.receptor_by_antenna
    assert by_number.provenance.receptor_sha256 == by_name.provenance.receptor_sha256


def test_mixed_array_under_auto_is_rejected_with_the_exact_message(tmp_path):
    instrument = _instrument(tmp_path)

    with pytest.raises(AmbiguousOutputBasisError) as error:
        resolve_receptors(
            _config(
                overrides=[
                    {"antenna": {"kind": "number", "number": 2}, "basis": "circular"}
                ]
            ),
            instrument,
        )

    assert str(error.value) == (
        "receptors.output_basis='auto' cannot resolve a mixed array "
        "(linear antennas: 4, circular antennas: 1); set receptors.output_basis "
        "to 'linear' or 'circular'."
    )


@pytest.mark.parametrize(
    ("requested", "expected"),
    [("linear", "linear_xy"), ("circular", "circular_rl")],
)
def test_explicit_output_basis_accepts_a_mixed_array(tmp_path, requested, expected):
    instrument = _instrument(tmp_path)

    resolved = resolve_receptors(
        _config(
            overrides=[
                {"antenna": {"kind": "number", "number": 2}, "basis": "circular"}
            ],
            output_basis=requested,
        ),
        instrument,
    )

    assert resolved.output_basis == expected
    assert resolved.provenance.requested_output_basis == requested
    assert resolved.provenance.output_basis_rule == f"explicit_{requested}"


@pytest.mark.parametrize(
    ("basis", "expected"),
    [("linear", "linear_xy"), ("circular", "circular_rl")],
)
def test_auto_resolves_a_homogeneous_array_from_the_native_basis(
    tmp_path, basis, expected
):
    instrument = _instrument(tmp_path)

    resolved = resolve_receptors(_config(default={"basis": basis}), instrument)

    assert resolved.output_basis == expected
    assert resolved.provenance.requested_output_basis == "auto"
    assert resolved.provenance.output_basis_rule == f"auto_homogeneous_{basis}"


def test_unknown_override_antenna_is_rejected_with_the_exact_message(tmp_path):
    instrument = _instrument(tmp_path)

    with pytest.raises(ReceptorAssignmentError) as error:
        resolve_receptors(
            _config(
                overrides=[
                    {
                        "antenna": {"kind": "number", "number": 0},
                        "feed_rotation_deg": 1.0,
                    },
                    {
                        "antenna": {"kind": "number", "number": 1},
                        "feed_rotation_deg": 2.0,
                    },
                    {
                        "antenna": {"kind": "number", "number": 91},
                        "feed_rotation_deg": 3.0,
                    },
                ]
            ),
            instrument,
        )

    assert str(error.value) == (
        "receptors.overrides[2] references antenna number 91, which is absent "
        "from the resolved instrument."
    )


def test_unknown_override_antenna_name_is_rejected(tmp_path):
    instrument = _instrument(tmp_path)

    with pytest.raises(ReceptorAssignmentError) as error:
        resolve_receptors(
            _config(
                overrides=[
                    {
                        "antenna": {"kind": "name", "name": "HERA-99"},
                        "feed_rotation_deg": 3.0,
                    }
                ]
            ),
            instrument,
        )

    assert str(error.value) == (
        "receptors.overrides[0] references antenna name 'HERA-99', which is "
        "absent from the resolved instrument."
    )


def test_cross_kind_duplicate_override_is_rejected_with_the_exact_message(tmp_path):
    instrument = _instrument(tmp_path)

    with pytest.raises(ReceptorAssignmentError) as error:
        resolve_receptors(
            _config(
                overrides=[
                    {
                        "antenna": {"kind": "number", "number": 0},
                        "feed_rotation_deg": 1.0,
                    },
                    {
                        "antenna": {"kind": "number", "number": 1},
                        "feed_rotation_deg": 2.0,
                    },
                    {
                        "antenna": {"kind": "number", "number": 2},
                        "feed_rotation_deg": 3.0,
                    },
                    {
                        "antenna": {"kind": "name", "name": "HERA-11"},
                        "feed_rotation_deg": 4.0,
                    },
                ]
            ),
            instrument,
        )

    assert str(error.value) == (
        "receptors.overrides[3] duplicates antenna 'HERA-11', already set by "
        "receptors.overrides[1]."
    )


@pytest.mark.parametrize(
    "mount_type", ["alt-az", "equatorial", "phased", "alt-az+nasmyth-l", "fixed", None]
)
def test_receptor_resolution_does_not_look_at_the_mount_type(tmp_path, mount_type):
    """FLIPPED BY: Tier 7F, which moved the mount rule to the term that owns it.

    Tier 5 rejected every non-``fixed`` mount here, with a message that named a
    tier rather than a fix, because a time-dependent feed orientation had no
    term to carry it.  Tier 7F implements ``P``, so a rotating mount is a
    configuration question rather than a deferral, and it is answered by
    rejections R12 and R15 in ``resolve_jones_terms``
    (``Tier7JonesSciencePlan.md`` Section 24; breaking-change ledger row B14).

    What this file still owns is the property Tier 5 actually cared about: a
    resolved receptor is the *static* part of the orientation, and receptor
    resolution stays pure and instrument-independent apart from the antenna
    inventory.  ``phased`` is in the sweep deliberately: even a mount ``P`` does
    not model is not this function's business, and R12 is what rejects it.
    """
    instrument = _with_mount_type(_instrument(tmp_path), mount_type)

    resolved = resolve_receptors(ReceptorsConfig(), instrument)

    assert resolved.output_basis == "linear_xy"
    assert len(resolved.receptor_by_antenna) == len(instrument.antennas)
    for receptor in resolved.receptor_by_antenna.values():
        assert receptor.feed_rotation_rad == 0.0


def test_the_mount_rejection_message_is_gone_from_the_receptor_module():
    """The Tier 5 message named a tier; its replacement names the fix."""
    import inspect

    from radiosim.core import receptor as receptor_module

    source = inspect.getsource(receptor_module)
    assert "is unsupported by Tier 5 receptors" not in source
    assert "_SUPPORTED_MOUNT_TYPE" not in source


def test_unsupported_basis_from_a_non_schema_caller_is_rejected(tmp_path):
    instrument = _instrument(tmp_path)
    config = ReceptorsConfig()
    config.default.__dict__["basis"] = "elliptical"

    with pytest.raises(UnsupportedReceptorBasisError) as error:
        resolve_receptors(config, instrument)

    assert "elliptical" in str(error.value)


def test_unsupported_output_basis_from_a_non_schema_caller_is_rejected(tmp_path):
    instrument = _instrument(tmp_path)
    config = ReceptorsConfig()
    config.__dict__["output_basis"] = "stokes"

    with pytest.raises(InvalidReceptorConfigError) as error:
        resolve_receptors(config, instrument)

    assert "output_basis" in str(error.value)


def test_receptor_error_taxonomy_is_hierarchical():
    assert issubclass(ReceptorError, RuntimeError)
    assert issubclass(InvalidReceptorConfigError, ReceptorError)
    for subclass in (
        UnsupportedReceptorBasisError,
        UnsupportedFeedGeometryError,
        AmbiguousOutputBasisError,
        ReceptorAssignmentError,
    ):
        assert issubclass(subclass, InvalidReceptorConfigError)


def test_receptor_sha256_is_stable_under_override_reordering(tmp_path):
    instrument = _instrument(tmp_path)
    first = {"antenna": {"kind": "number", "number": 1}, "feed_rotation_deg": 30.0}
    second = {"antenna": {"kind": "name", "name": "HERA-13"}, "basis": "circular"}

    forward = resolve_receptors(
        _config(overrides=[first, second], output_basis="linear"),
        instrument,
    )
    reversed_ = resolve_receptors(
        _config(overrides=[second, first], output_basis="linear"),
        instrument,
    )

    assert forward.receptor_by_antenna == reversed_.receptor_by_antenna
    assert forward.provenance.receptor_sha256 == reversed_.provenance.receptor_sha256
    assert forward.provenance.override_applications != (
        reversed_.provenance.override_applications
    )


@pytest.mark.parametrize(
    "changed",
    [
        {"default": {"basis": "circular"}},
        {"default": {"feed_rotation_deg": 1.0}},
        {"output_basis": "circular"},
        {
            "overrides": [
                {"antenna": {"kind": "number", "number": 0}, "feed_rotation_deg": 5.0}
            ]
        },
    ],
)
def test_receptor_sha256_changes_when_any_resolved_value_changes(tmp_path, changed):
    instrument = _instrument(tmp_path)

    baseline = resolve_receptors(ReceptorsConfig(), instrument)
    modified = resolve_receptors(_config(**changed), instrument)

    assert modified.provenance.receptor_sha256 != baseline.provenance.receptor_sha256


def test_receptor_sha256_covers_the_resolved_antenna_inventory(tmp_path):
    three = resolve_receptors(ReceptorsConfig(), _instrument(tmp_path / "a", count=3))
    five = resolve_receptors(ReceptorsConfig(), _instrument(tmp_path / "b", count=5))

    assert three.provenance.receptor_sha256 != five.provenance.receptor_sha256
    assert len(three.receptor_by_antenna) == 3


def test_resolution_does_not_change_the_instrument_fingerprint(tmp_path):
    instrument = _instrument(tmp_path)
    before = instrument.provenance.instrument_sha256

    resolve_receptors(_config(default={"basis": "circular"}), instrument)

    assert instrument.provenance.instrument_sha256 == before


def test_tampered_receptor_sha256_is_rejected(tmp_path):
    instrument = _instrument(tmp_path)
    resolved = resolve_receptors(ReceptorsConfig(), instrument)

    with pytest.raises(ValueError, match="receptor_sha256"):
        ResolvedReceptorSet(
            output_basis=resolved.output_basis,
            receptor_by_antenna=resolved.receptor_by_antenna,
            provenance=replace(resolved.provenance, receptor_sha256="0" * 64),
        )


def test_resolved_models_are_frozen_and_the_mapping_is_read_only(tmp_path):
    instrument = _instrument(tmp_path)
    resolved = resolve_receptors(ReceptorsConfig(), instrument)
    receptor = next(iter(resolved.receptor_by_antenna.values()))

    with pytest.raises(FrozenInstanceError):
        resolved.output_basis = "circular_rl"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        receptor.basis = "circular"  # type: ignore[misc]
    with pytest.raises(TypeError):
        resolved.receptor_by_antenna[AntennaId(0, "HERA-10")] = receptor  # type: ignore[index]


def test_snapshot_is_json_safe_deterministic_and_complete(tmp_path):
    instrument = _instrument(tmp_path)
    resolved = resolve_receptors(
        _config(
            default={"basis": "linear", "feed_rotation_deg": 30.0},
            overrides=[
                {"antenna": {"kind": "number", "number": 4}, "basis": "circular"}
            ],
            output_basis="circular",
        ),
        instrument,
    )

    snapshot = resolved.to_snapshot()

    assert snapshot is not resolved.to_snapshot()
    assert snapshot == resolved.to_snapshot()
    assert json.loads(json.dumps(snapshot, allow_nan=False)) == snapshot
    assert snapshot["output_basis"] == "circular_rl"
    assert snapshot["receptor_sha256"] == resolved.provenance.receptor_sha256
    assert len(snapshot["receptors"]) == 5
    assert snapshot["receptors"][4]["basis"] == "circular"
    assert snapshot["receptors"][4]["source"] == "explicit_override"
    assert snapshot["receptors"][0]["antenna_name"] == "HERA-10"


def test_provenance_records_the_ordered_override_applications(tmp_path):
    instrument = _instrument(tmp_path)

    resolved = resolve_receptors(
        _config(
            overrides=[
                {"antenna": {"kind": "number", "number": 4}, "feed_rotation_deg": 5.0},
                {"antenna": {"kind": "name", "name": "HERA-10"}, "basis": "linear"},
            ]
        ),
        instrument,
    )
    provenance = resolved.provenance

    assert type(provenance) is ReceptorProvenance
    assert [item.index for item in provenance.override_applications] == [0, 1]
    assert [item.antenna.number for item in provenance.override_applications] == [4, 0]
    assert provenance.override_applications[0].feed_rotation_applied is True
    assert provenance.override_applications[0].basis_applied is False
    assert provenance.override_applications[1].basis_applied is True
    assert provenance.override_applications[1].feed_rotation_applied is False


def test_resolution_is_exported_from_the_core_package():
    import radiosim.core as core_package

    assert core_package.resolve_receptors is resolve_receptors
    for name in (
        "ResolvedReceptor",
        "ResolvedReceptorSet",
        "ReceptorProvenance",
        "resolve_receptors",
        "ReceptorError",
        "InvalidReceptorConfigError",
        "UnsupportedReceptorBasisError",
        "UnsupportedFeedGeometryError",
        "AmbiguousOutputBasisError",
        "UnsupportedBasisTransformError",
        "ReceptorAssignmentError",
    ):
        assert name in core_package.__all__
        assert getattr(core_package, name) is not None


def test_resolution_rejects_foreign_argument_types(tmp_path):
    instrument = _instrument(tmp_path)

    with pytest.raises(TypeError):
        resolve_receptors(object(), instrument)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        resolve_receptors(ReceptorsConfig(), object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tier 5H: one authoritative polarization-basis literal (Section 34.8)
# ---------------------------------------------------------------------------


def test_polarization_basis_name_is_removed_from_the_receptor_module():
    """The duplicate Literal Section 34.8 removes must be unreachable."""
    import radiosim.core.receptor as receptor_module

    removed = "PolarizationBasisName"
    assert not hasattr(receptor_module, removed)
    assert removed not in receptor_module.__all__
    with pytest.raises(AttributeError):
        getattr(receptor_module, removed)
    with pytest.raises(ImportError):
        exec(
            compile(
                "from radiosim.core.receptor import PolarizationBasisName\n",
                "<tier5h>",
                "exec",
            ),
            {},
        )


def test_the_receptor_module_consumes_the_canonical_basis_literal():
    """``core/receptor.py`` must import ``PolarizationBasis``, not restate it."""
    import radiosim.core.polarization_basis as basis_module
    import radiosim.core.receptor as receptor_module

    assert receptor_module.PolarizationBasis is basis_module.PolarizationBasis
    assert set(receptor_module._OUTPUT_BASIS_BY_NATIVE.values()) == set(
        basis_module.POLARIZATION_BASES
    )


def test_the_resolved_output_basis_is_a_canonical_basis_token(tmp_path):
    """A resolved set must report a token the shared table recognizes."""
    import radiosim.core.polarization_basis as basis_module

    instrument = _instrument(tmp_path)
    for basis, expected in (("linear", "linear_xy"), ("circular", "circular_rl")):
        resolved = resolve_receptors(
            ReceptorsConfig.model_validate({"default": {"basis": basis}}),
            instrument,
        )
        assert resolved.output_basis == expected
        assert resolved.output_basis in basis_module.POLARIZATION_BASES
        assert resolved.output_basis in basis_module.CORRELATION_LABELS
