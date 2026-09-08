"""Independent byte oracle for the private stored-native binding codec."""

import hashlib
import json
import struct

import numpy as np
import pytest

from radiosim.core.sky.containers._polarization_payload import (
    bind_healpix_payload,
    require_healpix_binding,
)
from radiosim.core.sky.containers.constants import BrightnessConversion as BC
from radiosim.core.sky.containers.healpix import HealpixData


def test_literal_native_preimage_and_stale_consumer() -> None:
    owner = HealpixData(
        nside=1,
        frequencies=np.array([2.0, 1.0]),
        hpx_inds=np.array([7, 3]),
        maps=np.array([[1.0, -0.0], [3.0, 4.0]], dtype="<f4"),
    )
    names = ["frequencies", "pixel_ids", "I", "Q", "U", "V"]
    descriptors = [
        {
            "name": "frequencies",
            "present": True,
            "dtype": "<f8",
            "shape": [2],
            "byte_count": 16,
        },
        {
            "name": "pixel_ids",
            "present": True,
            "dtype": "<i8",
            "shape": [2],
            "byte_count": 16,
        },
        {
            "name": "I",
            "present": True,
            "dtype": "<f4",
            "shape": [2, 2],
            "byte_count": 16,
        },
        *[
            {
                "name": name,
                "present": False,
                "dtype": None,
                "shape": None,
                "byte_count": 0,
            }
            for name in ("Q", "U", "V")
        ],
    ]
    metadata = {
        "schema_version": "radiosim.polarization-payload-metadata.v1",
        "component_kind": "healpix",
        "coordinate_frame": "icrs",
        "ordering": "ring",
        "nside": 1,
        "units": dict(zip(names, ["Hz", "1", "K", "K", "K", "K"], strict=True)),
        "conversion": {
            "sky_I": "planck",
            "map_I": None,
            "map_Q": "rayleigh-jeans",
            "map_U": "rayleigh-jeans",
            "map_V": "rayleigh-jeans",
            "effective_Q": "rayleigh-jeans",
            "effective_U": "rayleigh-jeans",
            "effective_V": "rayleigh-jeans",
        },
        "arrays": descriptors,
    }
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    words = [
        struct.pack("<dd", 2.0, 1.0),
        struct.pack("<qq", 7, 3),
        struct.pack("<ffff", 1.0, -0.0, 3.0, 4.0),
        b"",
        b"",
        b"",
    ]
    preimage = b"RADIOSIM_POLARIZATION_PAYLOAD_V1\n" + struct.pack("<Q", len(encoded))
    preimage += encoded + b"".join(struct.pack("<Q", len(b)) + b for b in words)
    actual = bind_healpix_payload(owner, brightness_conversion=BC.PLANCK)
    assert actual.metadata_json == encoded
    assert actual.preimage_byte_count == len(preimage)
    assert actual.payload_sha256 == hashlib.sha256(preimage).hexdigest()
    require_healpix_binding(owner, brightness_conversion=BC.PLANCK, expected=actual)
    with pytest.raises(ValueError, match="binding mismatch"):
        require_healpix_binding(
            owner, brightness_conversion=BC.RAYLEIGH_JEANS, expected=actual
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_in_inactive_component_refuses(value: float) -> None:
    owner = HealpixData(
        nside=1,
        frequencies=np.array([1.0]),
        hpx_inds=np.array([0]),
        maps=np.zeros((1, 1)),
        v_maps=np.array([[value]]),
    )
    with pytest.raises(ValueError, match="nonfinite"):
        _ = bind_healpix_payload(owner, brightness_conversion=BC.PLANCK)


@pytest.mark.parametrize("nside", [0, 3, 2**30, 2**64])
def test_corrupt_nside_refuses(nside: int) -> None:
    owner = HealpixData(
        nside=1,
        frequencies=np.array([1.0]),
        hpx_inds=np.array([0]),
        maps=np.ones((1, 1)),
    )
    object.__setattr__(owner, "nside", nside)
    with pytest.raises(ValueError, match="nside"):
        _ = bind_healpix_payload(owner, brightness_conversion=BC.PLANCK)


def test_empty_frequency_refuses_but_empty_sparse_pixels_bind() -> None:
    owner = HealpixData(
        nside=1,
        frequencies=np.array([1.0]),
        hpx_inds=np.array([], dtype=np.int64),
        maps=np.empty((1, 0)),
    )
    assert bind_healpix_payload(owner, brightness_conversion=BC.PLANCK)
    frequencies, maps = np.empty(0), np.empty((0, 0))
    frequencies.flags.writeable = maps.flags.writeable = False
    object.__setattr__(owner, "frequencies", frequencies)
    object.__setattr__(owner, "maps", maps)
    with pytest.raises(ValueError, match="nonempty"):
        _ = bind_healpix_payload(owner, brightness_conversion=BC.PLANCK)


def test_aggregate_preflight_precedes_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    from radiosim.core.sky.containers import _polarization_payload as codec

    owner = HealpixData(
        nside=1,
        frequencies=np.array([1.0]),
        hpx_inds=np.array([0]),
        maps=np.ones((1, 1)),
    )
    binding = bind_healpix_payload(owner, brightness_conversion=BC.PLANCK)
    limit = len(binding.metadata_json) + 8

    def bounded_length(value: int) -> bytes:
        if value > limit:
            raise ValueError("bounded aggregate overflow")
        return struct.pack("<Q", value)

    def forbidden_hash(*args: object, **kwargs: object) -> None:
        raise AssertionError("hash constructed before aggregate refusal")

    monkeypatch.setattr(codec, "_length", bounded_length)
    monkeypatch.setattr(codec.hashlib, "sha256", forbidden_hash)
    with pytest.raises(ValueError, match="bounded aggregate overflow"):
        _ = bind_healpix_payload(owner, brightness_conversion=BC.PLANCK)


def test_uint64_aggregate_boundary_without_payload() -> None:
    from radiosim.core.sky.containers import _polarization_payload as codec

    overhead = len(b"RADIOSIM_POLARIZATION_PAYLOAD_V1\n") + 8 + 6 * 8
    sizes = [2**64 - 1 - overhead, 0, 0, 0, 0, 0]
    assert codec.preimage_length(0, sizes) == 2**64 - 1
    with pytest.raises(ValueError, match="uint64"):
        _ = codec.preimage_length(1, sizes)


def test_alias_mutation_invalidates_readonly_owner_binding() -> None:
    backing = np.array([[1.0, 2.0]])
    owner = HealpixData(
        nside=1,
        frequencies=np.array([1.0]),
        hpx_inds=np.array([4, 8]),
        maps=backing.view(),
    )
    assert np.shares_memory(backing, owner.maps)
    assert not owner.maps.flags.writeable and backing.flags.writeable
    before = bind_healpix_payload(owner, brightness_conversion=BC.PLANCK)
    require_healpix_binding(owner, brightness_conversion=BC.PLANCK, expected=before)
    backing[0, 1] = 3.0
    assert owner.maps[0, 1] == 3.0 and not owner.maps.flags.writeable
    with pytest.raises(ValueError, match="binding mismatch"):
        require_healpix_binding(owner, brightness_conversion=BC.PLANCK, expected=before)
    after = bind_healpix_payload(owner, brightness_conversion=BC.PLANCK)
    assert after.metadata_json == before.metadata_json
    assert after.payload_sha256 != before.payload_sha256
    require_healpix_binding(owner, brightness_conversion=BC.PLANCK, expected=after)


@pytest.mark.parametrize("component", ["Q", "U", "V"])
def test_component_absence_zero_and_signed_zero_have_distinct_identity(
    component: str,
) -> None:
    hashes: list[str] = []
    metadata: list[bytes] = []
    counts: list[int] = []
    for value in (None, 0.0, -0.0):
        values = None if value is None else np.array([[value]], dtype=np.float64)
        owner = HealpixData(
            nside=1,
            frequencies=np.array([1.0]),
            hpx_inds=np.array([0]),
            maps=np.ones((1, 1)),
            q_maps=values if component == "Q" else None,
            u_maps=values if component == "U" else None,
            v_maps=values if component == "V" else None,
        )
        record = bind_healpix_payload(owner, brightness_conversion=BC.PLANCK)
        hashes.append(record.payload_sha256)
        metadata.append(record.metadata_json)
        counts.append(record.preimage_byte_count)
        descriptor = json.loads(record.metadata_json)["arrays"][
            ["Q", "U", "V"].index(component) + 3
        ]
        assert descriptor["present"] is (value is not None)
        assert descriptor["byte_count"] == (0 if value is None else 8)
        if values is not None:
            assert values.tobytes() == struct.pack("<d", value)
    assert len(set(hashes)) == 3
    assert metadata[1] == metadata[2]
    assert counts[1] == counts[2]


@pytest.mark.parametrize(
    "change",
    ["frame", "ordering", "ids", "rows", "paired", "map_conversion", "sky_conversion"],
)
def test_constructor_owned_metadata_and_pixel_order_bind(change: str) -> None:
    original = HealpixData(
        nside=1,
        frequencies=np.array([2.0, 1.0]),
        hpx_inds=np.array([7, 3]),
        maps=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
    )
    permutation = change in ("rows", "paired")
    changed = HealpixData(
        nside=1,
        frequencies=np.array([2.0, 1.0]),
        hpx_inds=np.array([3, 7] if change in ("ids", "paired") else [7, 3]),
        maps=np.array(
            [[2.0, 1.0], [4.0, 3.0]] if permutation else [[1.0, 2.0], [3.0, 4.0]]
        ),
        coordinate_frame="galactic" if change == "frame" else "icrs",
        ordering="nest" if change == "ordering" else "ring",
        i_brightness_conversion="planck" if change == "map_conversion" else None,
    )
    context = BC.RAYLEIGH_JEANS if change == "sky_conversion" else BC.PLANCK
    before = bind_healpix_payload(original, brightness_conversion=BC.PLANCK)
    after = bind_healpix_payload(changed, brightness_conversion=context)
    assert after.payload_sha256 != before.payload_sha256
    with pytest.raises(ValueError, match="binding mismatch"):
        require_healpix_binding(changed, brightness_conversion=context, expected=before)
    if change in ("ids", "rows", "paired"):
        assert after.metadata_json == before.metadata_json
    if change == "paired":
        # Same physical per-ID tensor, different retained row order: identity differs.
        assert original.hpx_inds is not None and changed.hpx_inds is not None
        assert dict(zip(original.hpx_inds, original.maps[0], strict=True)) == dict(
            zip(changed.hpx_inds, changed.maps[0], strict=True)
        )
