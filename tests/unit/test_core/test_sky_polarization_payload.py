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
