"""Private canonical stored-native identity; no loader or admission authority."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from ._polarization_payload import bind_healpix_payload
from .constants import BrightnessConversion
from .healpix import HealpixData
from .point import TangentPolarizationFrame


@dataclass(frozen=True, slots=True)
class _IdentityOperation:
    kind: str
    input_sha256: str
    output_sha256: str
    parameters_sha256: str

    def as_mapping(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "parameters_sha256": self.parameters_sha256,
        }


@dataclass(frozen=True, slots=True)
class _Materialization:
    schema_version: str
    component_kind: str
    source_profile: str
    declaration_origin: str
    declaration_digest: str
    source_frame: str
    output_frame: str
    input_payload_sha256: str
    output_payload_sha256: str
    operations: tuple[_IdentityOperation, ...]
    parent_materialization_ids: tuple[str, ...]
    materialization_id: str

    def as_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "component_kind": self.component_kind,
            "source_profile": self.source_profile,
            "declaration_origin": self.declaration_origin,
            "declaration_digest": self.declaration_digest,
            "source_frame": self.source_frame,
            "output_frame": self.output_frame,
            "input_payload_sha256": self.input_payload_sha256,
            "output_payload_sha256": self.output_payload_sha256,
            "operations": [operation.as_mapping() for operation in self.operations],
            "parent_materialization_ids": list(self.parent_materialization_ids),
            "materialization_id": self.materialization_id,
        }


@dataclass(frozen=True, slots=True)
class _NativeIdentityReceipt:
    record: _Materialization
    tangent_frame: TangentPolarizationFrame | None
    declaration_json: bytes
    identity_parameters_json: bytes
    payload_metadata_json: bytes


def _json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _has_linear(owner: HealpixData) -> bool:
    for array in (owner.q_maps, owner.u_maps):
        if array is not None:
            with np.nditer(
                array,
                flags=["external_loop", "buffered", "zerosize_ok"],
                op_flags=[["readonly"]],
                order="C",
                buffersize=65_536,
            ) as chunks:
                for block in chunks:
                    values = cast(NDArray[np.float32 | np.float64], block)
                    if np.any(values != 0):
                        return True
    return False


def complete_native_identity(
    owner: HealpixData,
    *,
    brightness_conversion: BrightnessConversion,
    source_profile: Literal["radiosim_ne_iau_v1"],
    tangent_frame: TangentPolarizationFrame | None,
) -> _NativeIdentityReceipt:
    """Complete one declared identity on exclusively owned, already stored words.

    All payload finiteness checks precede the bounded Q/U pass. No original
    pre-constructor words, conversion history or concurrent snapshot is claimed.
    """
    if (
        type(cast(object, source_profile)) is not str
        or cast(object, source_profile) != "radiosim_ne_iau_v1"
    ):
        raise ValueError("identity requires the explicit canonical source profile")
    frame_mapping: dict[str, str] | None = None
    if tangent_frame is not None:
        if not isinstance(cast(object, tangent_frame), TangentPolarizationFrame):
            raise ValueError("expected a canonical tangent frame")
        frame_mapping = tangent_frame.as_mapping()
        if any(type(value) is not str for value in frame_mapping.values()):
            raise ValueError("tangent frame literals must be exact strings")
        canonical = TangentPolarizationFrame.canonical(tangent_frame.coordinate_frame)
        if frame_mapping != canonical.as_mapping():
            raise ValueError("tangent frame literals are not canonical")
    payload = bind_healpix_payload(owner, brightness_conversion=brightness_conversion)
    if _has_linear(owner) and tangent_frame is None:
        raise ValueError("nonzero Q/U requires an explicit canonical tangent frame")
    if (
        tangent_frame is not None
        and tangent_frame.coordinate_frame != owner.coordinate_frame
    ):
        raise ValueError("tangent frame coordinate does not match native owner")
    declaration = _json(
        {
            "schema_version": "radiosim.native-identity-declaration.v1",
            "source_profile": source_profile,
            "source_frame": owner.coordinate_frame,
            "tangent_polarization_frame": frame_mapping,
        }
    )
    parameters = _json(
        {
            "schema_version": "radiosim.polarization-identity-operation.v1",
            "algorithm": "stored_native_identity_v1",
            "input_frame": owner.coordinate_frame,
            "output_frame": owner.coordinate_frame,
            "payload_metadata_sha256": hashlib.sha256(
                payload.metadata_json
            ).hexdigest(),
        }
    )
    operation = _IdentityOperation(
        "identity",
        payload.payload_sha256,
        payload.payload_sha256,
        hashlib.sha256(parameters).hexdigest(),
    )
    declaration_digest = hashlib.sha256(declaration).hexdigest()
    body = {
        "schema_version": "radiosim.polarization-materialization.v1",
        "component_kind": "healpix",
        "source_profile": source_profile,
        "declaration_origin": "programmatic",
        "declaration_digest": declaration_digest,
        "source_frame": owner.coordinate_frame,
        "output_frame": owner.coordinate_frame,
        "input_payload_sha256": payload.payload_sha256,
        "output_payload_sha256": payload.payload_sha256,
        "operations": [operation.as_mapping()],
        "parent_materialization_ids": [],
    }
    encoded = _json(body)
    if len(encoded) >= 2**64:
        raise ValueError("materialization preimage length exceeds uint64")
    identity = hashlib.sha256(
        b"RADIOSIM_POLARIZATION_MATERIALIZATION_V1\n"
        + struct.pack("<Q", len(encoded))
        + encoded
    ).hexdigest()
    record = _Materialization(
        "radiosim.polarization-materialization.v1",
        "healpix",
        source_profile,
        "programmatic",
        declaration_digest,
        owner.coordinate_frame,
        owner.coordinate_frame,
        payload.payload_sha256,
        payload.payload_sha256,
        (operation,),
        (),
        identity,
    )
    return _NativeIdentityReceipt(
        record, tangent_frame, declaration, parameters, payload.metadata_json
    )


def require_native_identity(
    owner: HealpixData,
    *,
    brightness_conversion: BrightnessConversion,
    source_profile: Literal["radiosim_ne_iau_v1"],
    tangent_frame: TangentPolarizationFrame | None,
    expected: _NativeIdentityReceipt,
) -> None:
    """Rebuild from actual values/declaration, refusing stale or altered receipts."""
    if not isinstance(
        cast(object, expected), _NativeIdentityReceipt
    ) or expected != complete_native_identity(
        owner,
        brightness_conversion=brightness_conversion,
        source_profile=source_profile,
        tangent_frame=tangent_frame,
    ):
        raise ValueError("native identity materialization mismatch")
