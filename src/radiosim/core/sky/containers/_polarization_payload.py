"""Private stored-native identity codec; no normalization or admission authority."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .constants import BrightnessConversion
from .healpix import HealpixData

_CHUNK = 65_536
_DOMAIN = b"RADIOSIM_POLARIZATION_PAYLOAD_V1\n"
_NAMES = ("frequencies", "pixel_ids", "I", "Q", "U", "V")


@dataclass(frozen=True, slots=True)
class _PayloadBinding:
    metadata_json: bytes
    payload_sha256: str
    preimage_byte_count: int


def _length(value: int) -> bytes:
    if not 0 <= value < 2**64:
        raise ValueError("payload length exceeds uint64")
    return struct.pack("<Q", value)


def preimage_length(metadata_size: int, sizes: list[int]) -> int:
    for size in (metadata_size, *sizes):
        _ = _length(size)
    total = len(_DOMAIN) + 8 + metadata_size + sum(8 + size for size in sizes)
    _ = _length(total)
    return total


def bind_healpix_payload(
    owner: HealpixData, *, brightness_conversion: BrightnessConversion
) -> _PayloadBinding:
    """Bind owned stored words; callers must exclude concurrent alias mutation.

    No payload-sized copy is made. At most one iterator buffer, one serialized
    block, three comparison masks and one dense-ID block coexist (27 bytes per
    element at worst, each at most _CHUNK elements). Fixed metadata is separate.
    Read-only flags are discipline checks, not immutable-storage capabilities.
    """
    if not isinstance(cast(object, owner), HealpixData) or not isinstance(
        cast(object, brightness_conversion), BrightnessConversion
    ):
        raise ValueError("expected HealpixData and BrightnessConversion owners")
    nside = owner.nside
    if type(nside) is not int or not 1 <= nside <= 2**29 or nside & (nside - 1):
        raise ValueError("invalid nside")
    if owner.coordinate_frame not in ("icrs", "galactic") or owner.ordering not in (
        "ring",
        "nest",
    ):
        raise ValueError("invalid coordinate frame or ordering")
    if owner.maps.ndim != 2:
        raise ValueError("I must have shape (frequency, pixel)")
    f, n = owner.maps.shape
    if f == 0:
        raise ValueError("frequency axis must be nonempty")
    for scalar in (nside, f, n):
        _ = _length(scalar)
    arrays = (
        owner.frequencies,
        owner.hpx_inds,
        owner.maps,
        owner.q_maps,
        owner.u_maps,
        owner.v_maps,
    )
    units = dict(
        zip(
            _NAMES,
            ("Hz", "1", owner.i_unit, owner.q_unit, owner.u_unit, owner.v_unit),
            strict=True,
        )
    )
    if any(units[name] != "K" for name in _NAMES[2:]):
        raise ValueError("stored units must be K, including absent components")
    conversions = (
        owner.i_brightness_conversion,
        owner.q_brightness_conversion,
        owner.u_brightness_conversion,
        owner.v_brightness_conversion,
    )
    allowed = ("planck", "rayleigh-jeans")
    if conversions[0] not in (*allowed, None) or any(
        value not in allowed for value in conversions[1:]
    ):
        raise ValueError("invalid conversion metadata")
    descriptors: list[dict[str, object]] = []
    sizes: list[int] = []
    for index, (name, array) in enumerate(zip(_NAMES, arrays, strict=True)):
        shape = (f,) if index == 0 else (n,) if index == 1 else (f, n)
        if array is None and index == 1:
            if n != 12 * nside**2:
                raise ValueError("dense pixel count mismatch")
            dtype, size = "<i8", n * 8
        elif array is None and index >= 3:
            descriptors.append(
                {
                    "name": name,
                    "present": False,
                    "dtype": None,
                    "shape": None,
                    "byte_count": 0,
                }
            )
            sizes.append(0)
            continue
        else:
            if not isinstance(array, np.ndarray) or array.shape != shape:
                raise ValueError(f"{name}: shape mismatch")
            valid = array.dtype.kind == ("i" if index == 1 else "f")
            valid &= array.dtype.itemsize in ((8,) if index < 2 else (4, 8))
            if not valid or array.flags.writeable:
                raise ValueError(f"{name}: unsupported dtype or writeable owner")
            dtype, size = array.dtype.str, array.size * array.dtype.itemsize
        _ = _length(size)
        sizes.append(size)
        descriptors.append(
            {
                "name": name,
                "present": True,
                "dtype": dtype,
                "shape": list(shape),
                "byte_count": size,
            }
        )
    metadata = {
        "schema_version": "radiosim.polarization-payload-metadata.v1",
        "component_kind": "healpix",
        "coordinate_frame": owner.coordinate_frame,
        "ordering": owner.ordering,
        "nside": nside,
        "units": units,
        "conversion": dict(
            sky_I=brightness_conversion.value,
            **dict(zip(("map_I", "map_Q", "map_U", "map_V"), conversions, strict=True)),
            effective_Q="rayleigh-jeans",
            effective_U="rayleigh-jeans",
            effective_V="rayleigh-jeans",
        ),
        "arrays": descriptors,
    }
    encoded = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    count = preimage_length(len(encoded), sizes)
    digest = hashlib.sha256(_DOMAIN + _length(len(encoded)) + encoded)

    def consume(block: NDArray[np.float32 | np.float64 | np.int64], index: int) -> None:
        if index == 1:
            if np.any(block < 0) or np.any(block >= 12 * nside**2):
                raise ValueError("pixel ID outside grid")
        elif not np.all(np.isfinite(block)) or (index == 0 and np.any(block <= 0)):
            raise ValueError("nonfinite science values or nonpositive frequency")
        digest.update(block.tobytes(order="C"))

    for index, (array, descriptor) in enumerate(zip(arrays, descriptors, strict=True)):
        size = descriptor["byte_count"]
        assert isinstance(size, int)
        digest.update(_length(size))
        if array is None:
            if index == 1:
                for start in range(0, n, _CHUNK):
                    consume(
                        np.arange(start, min(start + _CHUNK, n), dtype="<i8"), index
                    )
            continue
        with np.nditer(
            array,
            flags=["external_loop", "buffered", "zerosize_ok"],
            op_flags=[["readonly"]],
            order="C",
            buffersize=_CHUNK,
        ) as chunks:
            for block in chunks:
                consume(cast(NDArray[np.float32 | np.float64 | np.int64], block), index)
    if any(array is not None and array.flags.writeable for array in arrays):
        raise ValueError("owner became writeable")
    return _PayloadBinding(encoded, digest.hexdigest(), count)


def require_healpix_binding(
    owner: HealpixData,
    *,
    brightness_conversion: BrightnessConversion,
    expected: _PayloadBinding,
) -> None:
    """Reject a stale binding; no declaration or conversion truth is implied."""
    if (
        not isinstance(cast(object, expected), _PayloadBinding)
        or bind_healpix_payload(owner, brightness_conversion=brightness_conversion)
        != expected
    ):
        raise ValueError("stored native payload binding mismatch")
