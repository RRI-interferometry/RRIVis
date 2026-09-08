"""Independent canonical identity record oracle; no loader or scientific run."""

import hashlib
import json
import struct
from dataclasses import replace
from typing import Literal, cast

import numpy as np
import pytest

from radiosim.core.sky.containers._polarization_materialization import (
    complete_native_identity,
    require_native_identity,
)
from radiosim.core.sky.containers._polarization_payload import bind_healpix_payload
from radiosim.core.sky.containers.constants import BrightnessConversion as BC
from radiosim.core.sky.containers.healpix import HealpixData
from radiosim.core.sky.containers.point import TangentPolarizationFrame


@pytest.mark.parametrize("coordinate", ["icrs", "galactic"])
def test_complete_identity_has_independent_twelve_field_oracle(coordinate: str) -> None:
    owner = HealpixData(
        nside=1,
        coordinate_frame=coordinate,
        frequencies=np.array([1.0]),
        hpx_inds=np.array([4]),
        maps=np.array([[2.0]], dtype=np.float32),
        q_maps=np.array([[1.0]], dtype=np.float32),
    )
    frame = TangentPolarizationFrame.canonical(coordinate)
    payload = bind_healpix_payload(owner, brightness_conversion=BC.PLANCK)
    frame_fields = {
        "schema_version": "radiosim.sky-tangent-polarization.v1",
        "coordinate_frame": coordinate,
        "axes": "north_east",
        "position_angle": "north_through_east",
        "linear_complex": "q_plus_i_u",
        "stokes_v": "iau_incoming_r_minus_l",
    }
    declaration = json.dumps(
        {
            "schema_version": "radiosim.native-identity-declaration.v1",
            "source_profile": "radiosim_ne_iau_v1",
            "source_frame": coordinate,
            "tangent_polarization_frame": frame_fields,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    parameters = json.dumps(
        {
            "schema_version": "radiosim.polarization-identity-operation.v1",
            "algorithm": "stored_native_identity_v1",
            "input_frame": coordinate,
            "output_frame": coordinate,
            "payload_metadata_sha256": hashlib.sha256(
                payload.metadata_json
            ).hexdigest(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    expected = {
        "schema_version": "radiosim.polarization-materialization.v1",
        "component_kind": "healpix",
        "source_profile": "radiosim_ne_iau_v1",
        "declaration_origin": "programmatic",
        "declaration_digest": hashlib.sha256(declaration).hexdigest(),
        "source_frame": coordinate,
        "output_frame": coordinate,
        "input_payload_sha256": payload.payload_sha256,
        "output_payload_sha256": payload.payload_sha256,
        "operations": [
            {
                "kind": "identity",
                "input_sha256": payload.payload_sha256,
                "output_sha256": payload.payload_sha256,
                "parameters_sha256": hashlib.sha256(parameters).hexdigest(),
            }
        ],
        "parent_materialization_ids": [],
    }
    encoded = json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    expected["materialization_id"] = hashlib.sha256(
        b"RADIOSIM_POLARIZATION_MATERIALIZATION_V1\n"
        + struct.pack("<Q", len(encoded))
        + encoded
    ).hexdigest()
    actual = complete_native_identity(
        owner,
        brightness_conversion=BC.PLANCK,
        source_profile="radiosim_ne_iau_v1",
        tangent_frame=frame,
    )
    assert actual.record.as_mapping() == expected and len(expected) == 12
    assert (
        actual.declaration_json == declaration
        and actual.identity_parameters_json == parameters
    )
    assert (
        actual.payload_metadata_json == payload.metadata_json
        and actual.tangent_frame == frame
    )
    assert owner.maps.dtype == np.dtype(
        np.float32
    ) and owner.maps.tobytes() == struct.pack("<f", 2.0)
    require_native_identity(
        owner,
        brightness_conversion=BC.PLANCK,
        source_profile="radiosim_ne_iau_v1",
        tangent_frame=frame,
        expected=actual,
    )
    assert (
        complete_native_identity(
            owner,
            brightness_conversion=BC.PLANCK,
            source_profile="radiosim_ne_iau_v1",
            tangent_frame=frame,
        )
        == actual
    )
    with pytest.raises(ValueError, match="materialization mismatch"):
        require_native_identity(
            owner,
            brightness_conversion=BC.RAYLEIGH_JEANS,
            source_profile="radiosim_ne_iau_v1",
            tangent_frame=frame,
            expected=actual,
        )


def test_nonzero_linear_requires_frame_and_all_values_are_checked_first() -> None:
    owner = HealpixData(
        nside=1,
        frequencies=np.array([1.0]),
        hpx_inds=np.array([0]),
        maps=np.zeros((1, 1)),
        q_maps=np.ones((1, 1)),
    )
    with pytest.raises(ValueError, match="requires an explicit"):
        _ = complete_native_identity(
            owner,
            brightness_conversion=BC.PLANCK,
            source_profile="radiosim_ne_iau_v1",
            tangent_frame=None,
        )
    bad = np.array([[np.nan]])
    bad.flags.writeable = False
    object.__setattr__(owner, "v_maps", bad)
    with pytest.raises(ValueError, match="nonfinite"):
        _ = complete_native_identity(
            owner,
            brightness_conversion=BC.PLANCK,
            source_profile="radiosim_ne_iau_v1",
            tangent_frame=None,
        )


def _identity_owner() -> HealpixData:
    return HealpixData(
        nside=1,
        frequencies=np.array([1.0]),
        hpx_inds=np.array([4]),
        maps=np.array([[2.0]]),
        q_maps=np.array([[1.0]]),
    )


def test_native_identity_rejects_changed_record_fields() -> None:
    owner = _identity_owner()
    frame = TangentPolarizationFrame.canonical("icrs")
    receipt = complete_native_identity(
        owner,
        brightness_conversion=BC.PLANCK,
        source_profile="radiosim_ne_iau_v1",
        tangent_frame=frame,
    )
    fields = (
        "schema_version",
        "component_kind",
        "source_profile",
        "declaration_origin",
        "declaration_digest",
        "source_frame",
        "output_frame",
        "input_payload_sha256",
        "output_payload_sha256",
        "materialization_id",
    )
    for field in fields:
        changed = replace(receipt.record, **{field: "changed"})
        assert changed != receipt.record
        with pytest.raises(ValueError, match="materialization mismatch"):
            require_native_identity(
                owner,
                brightness_conversion=BC.PLANCK,
                source_profile="radiosim_ne_iau_v1",
                tangent_frame=frame,
                expected=replace(receipt, record=changed),
            )
    require_native_identity(
        owner,
        brightness_conversion=BC.PLANCK,
        source_profile="radiosim_ne_iau_v1",
        tangent_frame=frame,
        expected=receipt,
    )


def test_native_identity_rejects_changed_operation_and_parents() -> None:
    owner = _identity_owner()
    frame = TangentPolarizationFrame.canonical("icrs")
    receipt = complete_native_identity(
        owner,
        brightness_conversion=BC.PLANCK,
        source_profile="radiosim_ne_iau_v1",
        tangent_frame=frame,
    )
    operation = receipt.record.operations[0]
    records = [
        replace(receipt.record, operations=(replace(operation, **{field: "changed"}),))
        for field in ("kind", "input_sha256", "output_sha256", "parameters_sha256")
    ]
    records.extend(
        [
            replace(receipt.record, operations=()),
            replace(receipt.record, operations=(operation, operation)),
            replace(receipt.record, parent_materialization_ids=("0" * 64,)),
        ]
    )
    for record in records:
        assert record != receipt.record
        with pytest.raises(ValueError, match="materialization mismatch"):
            require_native_identity(
                owner,
                brightness_conversion=BC.PLANCK,
                source_profile="radiosim_ne_iau_v1",
                tangent_frame=frame,
                expected=replace(receipt, record=record),
            )


def test_native_identity_rejects_changed_sidecars() -> None:
    owner = _identity_owner()
    frame = TangentPolarizationFrame.canonical("icrs")
    receipt = complete_native_identity(
        owner,
        brightness_conversion=BC.PLANCK,
        source_profile="radiosim_ne_iau_v1",
        tangent_frame=frame,
    )
    changed = [
        replace(receipt, declaration_json=receipt.declaration_json + b" "),
        replace(
            receipt, identity_parameters_json=receipt.identity_parameters_json + b" "
        ),
        replace(receipt, payload_metadata_json=receipt.payload_metadata_json + b" "),
        replace(receipt, tangent_frame=None),
        replace(receipt, tangent_frame=TangentPolarizationFrame.canonical("galactic")),
    ]
    for altered in changed:
        with pytest.raises(ValueError, match="materialization mismatch"):
            require_native_identity(
                owner,
                brightness_conversion=BC.PLANCK,
                source_profile="radiosim_ne_iau_v1",
                tangent_frame=frame,
                expected=altered,
            )
    detached = receipt.record.as_mapping()
    detached["operations"] = []
    detached["materialization_id"] = "changed"
    assert receipt.record.as_mapping()["operations"] != []
    require_native_identity(
        owner,
        brightness_conversion=BC.PLANCK,
        source_profile="radiosim_ne_iau_v1",
        tangent_frame=frame,
        expected=receipt,
    )


def test_native_identity_rejects_stale_backing_alias() -> None:
    backing = np.array([[2.0]])
    alias = backing.view()
    owner = HealpixData(
        nside=1,
        frequencies=np.array([1.0]),
        hpx_inds=np.array([4]),
        maps=backing,
    )
    assert np.shares_memory(alias, owner.maps)
    assert alias.flags.writeable and not owner.maps.flags.writeable
    receipt = complete_native_identity(
        owner,
        brightness_conversion=BC.PLANCK,
        source_profile="radiosim_ne_iau_v1",
        tangent_frame=None,
    )
    require_native_identity(
        owner,
        brightness_conversion=BC.PLANCK,
        source_profile="radiosim_ne_iau_v1",
        tangent_frame=None,
        expected=receipt,
    )
    alias[0, 0] = 3.0
    assert owner.maps.tobytes() == struct.pack("<d", 3.0)
    with pytest.raises(ValueError, match="materialization mismatch"):
        require_native_identity(
            owner,
            brightness_conversion=BC.PLANCK,
            source_profile="radiosim_ne_iau_v1",
            tangent_frame=None,
            expected=receipt,
        )


@pytest.mark.parametrize("profile", ["other", None, 7])
def test_native_identity_rejects_invalid_profile(profile: object) -> None:
    with pytest.raises(ValueError, match="explicit canonical source profile"):
        _ = complete_native_identity(
            _identity_owner(),
            brightness_conversion=BC.PLANCK,
            source_profile=cast(Literal["radiosim_ne_iau_v1"], profile),
            tangent_frame=TangentPolarizationFrame.canonical("icrs"),
        )


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "coordinate_frame",
        "axes",
        "position_angle",
        "linear_complex",
        "stokes_v",
    ],
)
def test_native_identity_rejects_malformed_frame(field: str) -> None:
    frame = TangentPolarizationFrame.canonical("icrs")
    object.__setattr__(frame, field, "invalid")
    with pytest.raises(ValueError):
        _ = complete_native_identity(
            _identity_owner(),
            brightness_conversion=BC.PLANCK,
            source_profile="radiosim_ne_iau_v1",
            tangent_frame=frame,
        )


def test_native_identity_rejects_conflicting_or_nonframe() -> None:
    for frame in (TangentPolarizationFrame.canonical("galactic"), object()):
        with pytest.raises(
            ValueError, match="coordinate does not match|canonical tangent frame"
        ):
            _ = complete_native_identity(
                _identity_owner(),
                brightness_conversion=BC.PLANCK,
                source_profile="radiosim_ne_iau_v1",
                tangent_frame=cast(TangentPolarizationFrame, frame),
            )


def test_native_identity_accepts_null_zero_and_empty() -> None:
    identities: list[str] = []
    for kind in ("absent", "zero", "empty"):
        n = 0 if kind == "empty" else 1
        owner = HealpixData(
            nside=1,
            frequencies=np.array([1.0]),
            hpx_inds=np.arange(n),
            maps=np.ones((1, n)),
            v_maps=np.ones((1, n)),
            q_maps=np.full((1, n), -0.0) if kind == "zero" else None,
            u_maps=np.zeros((1, n)) if kind == "zero" else None,
        )
        before = tuple(a.tobytes() for a in (owner.maps, owner.v_maps) if a is not None)
        receipt = complete_native_identity(
            owner,
            brightness_conversion=BC.PLANCK,
            source_profile="radiosim_ne_iau_v1",
            tangent_frame=None,
        )
        assert (
            json.loads(receipt.declaration_json)["tangent_polarization_frame"] is None
        )
        assert receipt.record.parent_materialization_ids == ()
        assert len(receipt.record.operations) == 1
        operation = receipt.record.operations[0]
        assert operation.kind == "identity"
        assert operation.input_sha256 == operation.output_sha256
        identities.append(receipt.record.input_payload_sha256)
        require_native_identity(
            owner,
            brightness_conversion=BC.PLANCK,
            source_profile="radiosim_ne_iau_v1",
            tangent_frame=None,
            expected=receipt,
        )
        assert before == tuple(
            a.tobytes() for a in (owner.maps, owner.v_maps) if a is not None
        )
        if kind == "zero":
            assert owner.q_maps is not None and owner.u_maps is not None
            assert owner.q_maps.tobytes() == struct.pack("<d", -0.0)
            assert owner.u_maps.tobytes() == struct.pack("<d", 0.0)
    assert len(set(identities)) == 3
    owner = HealpixData(
        nside=1,
        frequencies=np.array([1.0]),
        hpx_inds=np.array([0]),
        maps=np.zeros((1, 1)),
        u_maps=np.ones((1, 1)),
    )
    with pytest.raises(ValueError, match="requires an explicit"):
        _ = complete_native_identity(
            owner,
            brightness_conversion=BC.PLANCK,
            source_profile="radiosim_ne_iau_v1",
            tangent_frame=None,
        )
