"""Sequential owner-boundary controls; no concurrent snapshot qualification."""

from dataclasses import replace as replace_record
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky.containers._polarization_materialization import (
    complete_model_native_identity,
)
from radiosim.core.sky.containers.constants import BrightnessConversion as BC
from radiosim.core.sky.containers.healpix import HealpixData
from radiosim.core.sky.containers.model import SkyModel
from radiosim.core.sky.containers.point import TangentPolarizationFrame
from radiosim.core.sky.containers.polarization_materialization import (
    PolarizationMaterializationEvidence,
)


def _frame() -> TangentPolarizationFrame:
    return TangentPolarizationFrame(
        schema_version="radiosim.sky-tangent-polarization.v1",
        coordinate_frame="icrs",
        axes="north_east",
        position_angle="north_through_east",
        linear_complex="q_plus_i_u",
        stokes_v="iau_incoming_r_minus_l",
    )


def _bound() -> tuple[SkyModel, HealpixData, NDArray[np.float64]]:
    # Keep a distinct writable base; freezing its view is not ownership.
    backing = np.full((1, 12), 2.0, dtype=np.float64)
    raw = HealpixData(
        maps=backing.view(),
        q_maps=np.full((1, 12), 1.0, dtype=np.float64),
        frequencies=np.array([100_000_000.0], dtype=np.float64),
        nside=1,
    )
    model = SkyModel(healpix=raw, precision=PrecisionConfig.precise())
    bound = complete_model_native_identity(
        model, source_profile="radiosim_ne_iau_v1", tangent_frame=_frame()
    )
    owner = bound.healpix
    assert owner is not None
    assert owner.polarization_materialization is not None
    assert np.shares_memory(backing, owner.maps)
    assert backing.flags.writeable
    assert not owner.maps.flags.writeable
    assert owner.maps[0, 0] == 2.0
    return bound, owner, backing


@pytest.mark.parametrize(
    "operation",
    ["require_dense", "to_dense", "reordered", "crop", "zero", "model", "complete"],
)
def test_stale_backing_refuses_noop_boundaries(operation: str) -> None:
    model, owner, backing = _bound()
    receipt = owner.polarization_materialization
    words = backing.tobytes()
    backing[0, 0] = 3.0  # Sequential mutation, finished before validation starts.
    assert owner.maps[0, 0] == 3.0
    assert not owner.maps.flags.writeable
    try:
        with pytest.raises(ValueError, match="materialization"):
            if operation == "require_dense":
                _ = owner.require_dense("stale-control")
            elif operation == "to_dense":
                _ = owner.to_dense()
            elif operation == "reordered":
                _ = owner.reordered("ring")
            elif operation == "crop":
                _ = owner.cropped_to_mask(np.ones(12, dtype=bool))
            elif operation == "zero":
                _ = owner.zero_outside_mask(np.ones(12, dtype=bool))
            elif operation == "model":
                _ = model.replace()
            else:
                _ = complete_model_native_identity(
                    model,
                    source_profile="radiosim_ne_iau_v1",
                    tangent_frame=_frame(),
                )
        assert owner.polarization_materialization is receipt
        assert backing[0, 0] == 3.0  # Refusal did not heal the input.
    finally:
        backing[0, 0] = 2.0
    assert backing.tobytes() == words
    owner.validate_polarization_materialization()


def test_old_staleness_cannot_be_hidden_by_valid_replacement() -> None:
    model, owner, backing = _bound()
    receipt = owner.polarization_materialization
    clean = owner.replace(maps=owner.maps.copy())
    assert not np.shares_memory(clean.maps, backing)
    assert clean.polarization_materialization is receipt
    clean.validate_polarization_materialization()
    backing[0, 0] = 3.0
    try:
        # Both proposed outputs match the original receipt; only the old
        # owner's validation can reject this attempted repair/replacement.
        with pytest.raises(ValueError, match="materialization"):
            _ = owner.replace(maps=clean.maps)
        with pytest.raises(ValueError, match="materialization"):
            _ = model.replace(healpix=clean)
        assert backing[0, 0] == 3.0
        assert clean.maps[0, 0] == 2.0
        assert owner.polarization_materialization is receipt
    finally:
        backing[0, 0] = 2.0
    assert owner.replace(maps=clean.maps).polarization_materialization is receipt
    replaced = model.replace(healpix=clean)
    assert replaced.healpix is not None
    assert replaced.healpix.polarization_materialization is receipt


def test_reissued_equal_receipt_cannot_replace_parent_receipt() -> None:
    model, owner, _ = _bound()
    receipt = owner.polarization_materialization
    assert receipt is not None
    reissued = replace_record(receipt)
    assert reissued == receipt and reissued is not receipt
    with pytest.raises(ValueError, match="drop or reissue"):
        _ = owner.replace(polarization_materialization=reissued)
    # New independent construction is allowed; deriving an existing model
    # through its replace interface must preserve that model's parent object.
    other = HealpixData(
        maps=owner.maps,
        q_maps=owner.q_maps,
        frequencies=owner.frequencies,
        nside=1,
        i_brightness_conversion="planck",
        tangent_polarization_frame=_frame(),
        polarization_materialization=reissued,
    )
    assert other.polarization_materialization is reissued
    with pytest.raises(ValueError, match="drop or reissue"):
        _ = model.replace(healpix=other)
    assert owner.polarization_materialization is receipt


@pytest.mark.parametrize("field", ["frame", "evidence", "frame_only"])
def test_attachment_rejects_mapping_coercion_and_frame_only(field: str) -> None:
    _, owner, _ = _bound()
    receipt = owner.polarization_materialization
    assert receipt is not None
    frame = _frame()
    # Deliberately valid contents: rejection must be about missing typed
    # ownership, not a malformed declaration or changed scientific preimage.
    frame_dict = {
        "schema_version": "radiosim.sky-tangent-polarization.v1",
        "coordinate_frame": "icrs",
        "axes": "north_east",
        "position_angle": "north_through_east",
        "linear_complex": "q_plus_i_u",
        "stokes_v": "iau_incoming_r_minus_l",
    }
    evidence_dict = {
        "record": receipt.record,
        "tangent_frame": receipt.tangent_frame,
        "declaration_json": receipt.declaration_json,
        "identity_parameters_json": receipt.identity_parameters_json,
        "payload_metadata_json": receipt.payload_metadata_json,
        "brightness_conversion": BC.PLANCK,
    }
    expected = "typed frame" if field == "frame" else "materialization"
    with pytest.raises(ValueError, match=expected):
        _ = HealpixData(
            maps=owner.maps,
            q_maps=owner.q_maps,
            frequencies=owner.frequencies,
            nside=1,
            i_brightness_conversion="planck",
            tangent_polarization_frame=(
                cast(TangentPolarizationFrame, frame_dict)
                if field == "frame"
                else frame
            ),
            polarization_materialization=(
                None
                if field == "frame_only"
                else cast(PolarizationMaterializationEvidence, evidence_dict)
                if field == "evidence"
                else receipt
            ),
        )
    owner.validate_polarization_materialization()


@pytest.mark.parametrize("context", [BC.RAYLEIGH_JEANS, "planck"])
def test_corrupt_context_reaches_actual_consumer_without_coercion(
    context: object,
) -> None:
    _, owner, _ = _bound()
    receipt = owner.polarization_materialization
    assert receipt is not None
    changed = replace_record(receipt, brightness_conversion=context)
    assert changed.record is receipt.record
    assert changed.payload_metadata_json is receipt.payload_metadata_json
    with pytest.raises(ValueError, match="materialization"):
        _ = HealpixData(
            maps=owner.maps,
            q_maps=owner.q_maps,
            frequencies=owner.frequencies,
            nside=1,
            i_brightness_conversion="planck",
            tangent_polarization_frame=_frame(),
            polarization_materialization=changed,
        )
    owner.validate_polarization_materialization(brightness_conversion=BC.PLANCK)
    with pytest.raises(ValueError, match="materialization"):
        owner.validate_polarization_materialization(
            brightness_conversion=BC.RAYLEIGH_JEANS
        )
