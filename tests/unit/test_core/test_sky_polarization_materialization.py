"""Independent canonical identity record oracle; no loader or scientific run."""

import hashlib
import json
import struct

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
