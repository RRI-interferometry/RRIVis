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


def test_constructor_preserves_layouts_with_equal_logical_words() -> None:
    logical = np.array([[1.0, -0.0, 3.0], [4.0, 5.0, 6.0]], dtype="<f8")
    backing = np.empty((2, 6), dtype="<f8")
    backing[:, ::2] = logical
    inputs = (logical.copy(), np.asfortranarray(logical), backing[:, ::2])
    digests: list[str] = []
    for index, values in enumerate(inputs):
        owner = HealpixData(
            nside=1,
            frequencies=np.array([2.0, 1.0]),
            hpx_inds=np.array([8, 1, 4]),
            maps=values,
        )
        assert np.shares_memory(owner.maps, values)
        assert owner.maps.dtype.str == "<f8"
        assert owner.maps.strides == values.strides
        assert not owner.maps.flags.writeable
        if index == 0:
            assert owner.maps.flags.c_contiguous
        elif index == 1:
            assert owner.maps.flags.f_contiguous and not owner.maps.flags.c_contiguous
        else:
            assert (
                not owner.maps.flags.c_contiguous and not owner.maps.flags.f_contiguous
            )
        assert owner.maps.tobytes(order="C") == struct.pack(
            "<6d", 1.0, -0.0, 3.0, 4.0, 5.0, 6.0
        )
        digests.append(
            bind_healpix_payload(owner, brightness_conversion=BC.PLANCK).payload_sha256
        )
    assert len(set(digests)) == 1


def test_actual_endian_and_width_words_have_distinct_identity() -> None:
    hashes: list[str] = []
    for dtype, format_code in (
        ("<f4", "<ff"),
        (">f4", ">ff"),
        ("<f8", "<dd"),
        (">f8", ">dd"),
    ):
        values = np.array([[1.0, -0.0]], dtype=dtype)
        owner = HealpixData(
            nside=1,
            frequencies=np.array([1.0], dtype=">f8"),
            hpx_inds=np.array([0, 1], dtype=">i8"),
            maps=values,
        )
        # Map words survive construction; channels/IDs normalize to native widths.
        assert owner.maps.dtype.str == dtype and np.shares_memory(values, owner.maps)
        assert owner.frequencies.dtype == np.dtype(np.float64)
        assert owner.hpx_inds is not None and owner.hpx_inds.dtype == np.dtype(np.int64)
        assert owner.maps.tobytes() == struct.pack(format_code, 1.0, -0.0)
        assert owner.frequencies.tobytes() == struct.pack("<d", 1.0)
        assert owner.hpx_inds.tobytes() == struct.pack("<qq", 0, 1)
        binding = bind_healpix_payload(owner, brightness_conversion=BC.PLANCK)
        descriptor = json.loads(binding.metadata_json)["arrays"][2]
        assert descriptor["dtype"] == dtype
        hashes.append(binding.payload_sha256)
    assert len(set(hashes)) == 4


def test_dense_and_explicit_canonical_ids_have_equal_binding() -> None:
    maps = np.arange(12, dtype=np.float64).reshape(1, 12)
    dense = HealpixData(nside=1, frequencies=np.array([1.0]), maps=maps.copy())
    explicit = HealpixData(
        nside=1,
        frequencies=np.array([1.0]),
        maps=maps.copy(),
        hpx_inds=np.arange(12, dtype=np.int64),
    )
    assert dense.hpx_inds is None
    assert explicit.hpx_inds is not None
    assert explicit.hpx_inds.tobytes() == struct.pack("<12q", *range(12))
    assert (
        dense.maps.tobytes()
        == explicit.maps.tobytes()
        == struct.pack("<12d", *range(12))
    )
    assert bind_healpix_payload(
        dense, brightness_conversion=BC.PLANCK
    ) == bind_healpix_payload(explicit, brightness_conversion=BC.PLANCK)


def test_paired_channel_permutation_preserves_values_not_identity() -> None:
    first = HealpixData(
        nside=1,
        frequencies=np.array([2.0, 1.0]),
        hpx_inds=np.array([4]),
        maps=np.array([[3.0], [5.0]]),
        q_maps=np.array([[7.0], [11.0]]),
    )
    second = HealpixData(
        nside=1,
        frequencies=np.array([1.0, 2.0]),
        hpx_inds=np.array([4]),
        maps=np.array([[5.0], [3.0]]),
        q_maps=np.array([[11.0], [7.0]]),
    )
    assert first.frequencies.tobytes() == struct.pack("<dd", 2.0, 1.0)
    assert second.frequencies.tobytes() == struct.pack("<dd", 1.0, 2.0)
    assert first.q_maps is not None and second.q_maps is not None
    assert dict(zip(first.frequencies, first.maps[:, 0], strict=True)) == dict(
        zip(second.frequencies, second.maps[:, 0], strict=True)
    )
    assert dict(zip(first.frequencies, first.q_maps[:, 0], strict=True)) == dict(
        zip(second.frequencies, second.q_maps[:, 0], strict=True)
    )
    a = bind_healpix_payload(first, brightness_conversion=BC.PLANCK)
    b = bind_healpix_payload(second, brightness_conversion=BC.PLANCK)
    assert a.metadata_json == b.metadata_json
    assert a.payload_sha256 != b.payload_sha256
    with pytest.raises(ValueError, match="binding mismatch"):
        require_healpix_binding(second, brightness_conversion=BC.PLANCK, expected=a)


@pytest.mark.parametrize(
    "case",
    [
        "frame",
        "ordering",
        "I_unit",
        "absent_Q_unit",
        "I_conversion",
        "Q_conversion",
        "I_rank",
        "Q_shape",
        "frequency_shape",
        "frequency_dtype",
        "ID_shape",
        "ID_dtype",
        "ID_negative",
        "ID_upper",
        "I_integer",
        "U_complex",
        "V_object",
    ],
)
def test_corrupt_real_owner_fields_refuse(case: str) -> None:
    owner = HealpixData(
        nside=1,
        frequencies=np.array([1.0]),
        hpx_inds=np.array([0]),
        maps=np.zeros((1, 1)),
    )
    field: str
    value: object
    if case in (
        "frame",
        "ordering",
        "I_unit",
        "absent_Q_unit",
        "I_conversion",
        "Q_conversion",
    ):
        field = {
            "frame": "coordinate_frame",
            "ordering": "ordering",
            "I_unit": "i_unit",
            "absent_Q_unit": "q_unit",
            "I_conversion": "i_brightness_conversion",
            "Q_conversion": "q_brightness_conversion",
        }[case]
        value = "unsupported"
    else:
        field = {
            "I_rank": "maps",
            "Q_shape": "q_maps",
            "frequency_shape": "frequencies",
            "frequency_dtype": "frequencies",
            "ID_shape": "hpx_inds",
            "ID_dtype": "hpx_inds",
            "ID_negative": "hpx_inds",
            "ID_upper": "hpx_inds",
            "I_integer": "maps",
            "U_complex": "u_maps",
            "V_object": "v_maps",
        }[case]
        if case == "I_rank":
            array = np.zeros(1)
        elif case == "Q_shape":
            array = np.zeros((1, 2))
        elif case == "frequency_shape":
            array = np.ones(2)
        elif case == "frequency_dtype":
            array = np.ones(1, dtype=np.float32)
        elif case == "ID_shape":
            array = np.zeros((1, 1), dtype=np.int64)
        elif case == "ID_dtype":
            array = np.zeros(1, dtype=np.uint64)
        elif case in ("ID_negative", "ID_upper"):
            array = np.array([-1 if case == "ID_negative" else 12], dtype=np.int64)
        elif case == "I_integer":
            array = np.zeros((1, 1), dtype=np.int64)
        elif case == "U_complex":
            array = np.zeros((1, 1), dtype=np.complex128)
        else:
            array = np.zeros((1, 1), dtype=object)
        array.flags.writeable = False
        value = array
    object.__setattr__(owner, field, value)
    with pytest.raises(ValueError):
        _ = bind_healpix_payload(owner, brightness_conversion=BC.PLANCK)


@pytest.mark.parametrize("value", [np.nan, np.inf, -1.0, 0.0])
def test_corrupt_frequency_domain_refuses(value: float) -> None:
    owner = HealpixData(
        nside=1,
        frequencies=np.array([1.0]),
        hpx_inds=np.array([0]),
        maps=np.zeros((1, 1)),
    )
    frequencies = np.array([value])
    frequencies.flags.writeable = False
    object.__setattr__(owner, "frequencies", frequencies)
    with pytest.raises(ValueError, match="nonfinite|nonpositive"):
        _ = bind_healpix_payload(owner, brightness_conversion=BC.PLANCK)


@pytest.mark.parametrize("component", ["maps", "q_maps", "u_maps", "v_maps"])
def test_each_nonfinite_component_refuses_on_zero_signal(component: str) -> None:
    owner = HealpixData(
        nside=1,
        frequencies=np.array([1.0]),
        hpx_inds=np.array([0]),
        maps=np.zeros((1, 1)),
    )
    values = np.array([[np.nan]])
    values.flags.writeable = False
    object.__setattr__(owner, component, values)
    with pytest.raises(ValueError, match="nonfinite"):
        _ = bind_healpix_payload(owner, brightness_conversion=BC.PLANCK)


def test_constructor_float16_and_absent_nonkelvin_are_private_refusals() -> None:
    half = HealpixData(
        nside=1,
        frequencies=np.array([1.0]),
        hpx_inds=np.array([0]),
        maps=np.zeros((1, 1), dtype=np.float16),
    )
    assert half.maps.dtype == np.dtype(np.float16)
    with pytest.raises(ValueError, match="unsupported dtype"):
        _ = bind_healpix_payload(half, brightness_conversion=BC.PLANCK)
    absent = HealpixData(
        nside=1,
        frequencies=np.array([1.0]),
        hpx_inds=np.array([0]),
        maps=np.zeros((1, 1)),
        q_unit="Jy/sr",
    )
    assert absent.q_maps is None and absent.q_unit == "Jy/sr"
    with pytest.raises(ValueError, match="including absent"):
        _ = bind_healpix_payload(absent, brightness_conversion=BC.PLANCK)


@pytest.mark.parametrize("field", ["maps", "frequencies", "hpx_inds", "q_maps"])
def test_writable_array_owner_refuses(field: str) -> None:
    owner = HealpixData(
        nside=1,
        frequencies=np.array([1.0]),
        hpx_inds=np.array([0]),
        maps=np.zeros((1, 1)),
        q_maps=np.zeros((1, 1)),
    )
    array = {
        "maps": owner.maps,
        "frequencies": owner.frequencies,
        "hpx_inds": owner.hpx_inds,
        "q_maps": owner.q_maps,
    }[field]
    assert array is not None
    array.flags.writeable = True
    try:
        with pytest.raises(ValueError, match="writeable"):
            _ = bind_healpix_payload(owner, brightness_conversion=BC.PLANCK)
    finally:
        array.flags.writeable = False


def test_invalid_runtime_owner_and_conversion_context_refuse() -> None:
    from typing import cast

    owner = HealpixData(
        nside=1,
        frequencies=np.array([1.0]),
        hpx_inds=np.array([0]),
        maps=np.zeros((1, 1)),
    )
    with pytest.raises(ValueError, match="owners"):
        _ = bind_healpix_payload(
            cast(HealpixData, object()), brightness_conversion=BC.PLANCK
        )
    with pytest.raises(ValueError, match="owners"):
        _ = bind_healpix_payload(owner, brightness_conversion=cast(BC, "planck"))


@pytest.mark.parametrize("dense", [False, True])
def test_actual_chunk_and_observable_scratch_envelope(
    dense: bool, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys
    import weakref
    from types import CodeType, FrameType
    from typing import Literal, cast

    from numpy.typing import NDArray

    from radiosim.core.sky.containers import _polarization_payload as codec

    limit = 65_536
    n = 196_608 if dense else limit + 3
    channels = 1 if dense else 2
    backing = np.arange(channels * n, dtype=np.float64).reshape(channels, n)
    owner = HealpixData(
        nside=128,
        frequencies=np.arange(1, channels + 1, dtype=np.float64),
        hpx_inds=None if dense else np.arange(n, dtype=np.int64),
        maps=backing if dense else np.asfortranarray(backing),
        q_maps=None if dense else np.ones((channels, n), dtype=np.float32),
    )
    expected = bind_healpix_payload(owner, brightness_conversion=BC.PLANCK)
    Array = NDArray[np.generic]

    def span(array: Array) -> tuple[int, int]:
        offset = sum(
            (size - 1) * stride
            for size, stride in zip(array.shape, array.strides, strict=True)
        )
        return array.ctypes.data, array.ctypes.data + offset + array.dtype.itemsize

    owners: list[tuple[int, int]] = []
    for array in (owner.maps, owner.q_maps, owner.frequencies, owner.hpx_inds):
        if array is not None:
            while isinstance(array.base, np.ndarray):
                array = array.base
            owners.append(span(array))
    live: dict[
        int, tuple[weakref.ReferenceType[Array], int | None, tuple[int, int] | None]
    ] = {}
    parents: dict[int, int] = {}
    large_parents: set[int] = set()
    active_parent: int | None = None
    peak = 0
    peak_views = 0
    short_parent_retained = False
    closed_parent_retained = False
    max_required_closed = 0
    max_charged_closed = 0
    in_consume = False
    consume_index = -1
    descriptor_index = -1
    element_counts = [0] * 6
    stream_hashes = [hashlib.sha256() for _ in range(6)]
    chunks_seen = 0
    masks_seen = 0
    payload_updates = 0
    dense_intervals: list[tuple[int, int]] = []
    original_nditer, original_arange = np.nditer, np.arange
    original_any, original_all, original_sha = np.any, np.all, hashlib.sha256
    consume_code = next(
        c
        for c in codec.bind_healpix_payload.__code__.co_consts
        if isinstance(c, CodeType) and c.co_name == "consume"
    )

    def measure(extra_bytes: int = 0) -> None:
        nonlocal peak, peak_views, short_parent_retained, closed_parent_retained
        nonlocal max_required_closed, max_charged_closed
        intervals: list[tuple[int, int]] = []
        required_parents: set[int] = set()
        parent_view_bytes: dict[int, int] = {}
        retained_parents: set[int] = set()
        for identity, (reference, parent, extent) in list(live.items()):
            array = reference()
            if array is None:
                del live[identity]
                continue
            if parent is not None:
                retained_parents.add(parent)
                parent_view_bytes[parent] = max(
                    parent_view_bytes.get(parent, 0), array.nbytes
                )
                if parent in large_parents and array.nbytes < parents[parent]:
                    short_parent_retained = True
                    closed_parent_retained |= active_parent is None
                    if active_parent is None:
                        required_parents.add(parent)
            elif extent is not None:
                intervals.append(extent)
        intervals.sort()
        total, right = 0, -1
        for start, end in intervals:
            total += max(0, end - max(start, right))
            right = max(right, end)
        peak_views = max(peak_views, total)
        if active_parent is not None:
            retained_parents.add(active_parent)
        # Each unexposed C parent gets its full configured reserve, never a tail span.
        reserve = sum(parents[parent] for parent in retained_parents)
        required = sum(parents[parent] for parent in required_parents)
        assert all(
            parent_view_bytes[parent] <= parents[parent] for parent in parent_view_bytes
        )
        assert reserve >= required, "retained parent capacity undercharged"
        if required:
            max_required_closed = max(max_required_closed, required)
            max_charged_closed = max(max_charged_closed, reserve)
        peak = max(peak, total + reserve + extra_bytes)

    def observe(array: Array) -> None:
        root = array
        while isinstance(root.base, np.ndarray):
            root = root.base
        parent: int | None = None
        extent: tuple[int, int] | None = None
        if isinstance(cast(object, root.base), original_nditer):
            parent = id(root.base)
            assert parent in parents, "unregistered iterator parent"
        else:
            assert root.base is None, "unresolved storage parent"
            lo, hi = span(root)
            if not any(start <= lo and hi <= end for start, end in owners):
                extent = (lo, hi)
        live[id(array)] = (weakref.ref(array), parent, extent)
        measure()

    def profile(frame: FrameType, event: str, arg: object) -> None:
        nonlocal chunks_seen, in_consume, consume_index
        if frame.f_code is consume_code and event == "return":
            in_consume = False
        if event == "call" and frame.f_code is consume_code:
            in_consume = True
            consume_index = int(frame.f_locals["index"])
            assert consume_index == descriptor_index
            block = cast(Array, frame.f_locals["block"])
            assert block.size <= limit
            chunks_seen += 1
            element_counts[consume_index] += block.size
            observe(block)

    def iterator(
        array: Array,
        *,
        flags: list[Literal["external_loop", "buffered", "zerosize_ok"]],
        op_flags: list[list[Literal["readonly"]]],
        order: Literal["C"],
        buffersize: int,
    ) -> np.nditer:
        nonlocal active_parent
        assert buffersize == limit
        result = original_nditer(
            array, flags=flags, op_flags=op_flags, order=order, buffersize=buffersize
        )
        active_parent = id(result)
        parents[active_parent] = buffersize * array.dtype.itemsize
        if array.size > buffersize:
            large_parents.add(active_parent)
        measure()
        return result

    def arange(start: int, stop: int, *, dtype: str) -> Array:
        assert dtype == "<i8" and 0 < stop - start <= limit
        result = original_arange(start, stop, dtype=dtype)
        dense_intervals.append((start, stop))
        observe(result)
        return result

    def mask_any(array: Array) -> np.bool_:
        nonlocal masks_seen
        assert array.dtype == np.dtype(bool) and array.size <= limit
        masks_seen += 1
        observe(array)
        return original_any(array)

    def mask_all(array: Array) -> np.bool_:
        nonlocal masks_seen
        assert array.dtype == np.dtype(bool) and array.size <= limit
        masks_seen += 1
        observe(array)
        return original_all(array)

    class Hash:
        def __init__(self, data: bytes) -> None:
            self.original = original_sha(data)

        def update(self, data: bytes) -> None:
            nonlocal payload_updates, active_parent, descriptor_index
            if in_consume:
                assert len(data) <= limit * 8
                payload_updates += 1
                stream_hashes[consume_index].update(data)
                measure(len(data))
            else:
                descriptor_index += 1
                expected_size = [
                    channels * 8,
                    n * 8,
                    channels * n * 8,
                    0 if dense else channels * n * 4,
                    0,
                    0,
                ][descriptor_index]
                assert data == struct.pack("<Q", expected_size)
                active_parent = (
                    None  # Previous context exited before next frame header.
                )
                measure()
            self.original.update(data)

        def hexdigest(self) -> str:
            return self.original.hexdigest()

    def forbidden_pixels(instance: HealpixData) -> None:
        raise AssertionError("dense pixel_indices property must not allocate")

    previous = sys.getprofile()
    with monkeypatch.context() as patch:
        patch.setattr(np, "nditer", iterator)
        patch.setattr(np, "arange", arange)
        patch.setattr(np, "any", mask_any)
        patch.setattr(np, "all", mask_all)
        patch.setattr(hashlib, "sha256", Hash)
        patch.setattr(HealpixData, "pixel_indices", property(forbidden_pixels))
        try:
            sys.setprofile(profile)
            actual = bind_healpix_payload(owner, brightness_conversion=BC.PLANCK)
        finally:
            sys.setprofile(previous)
    assert actual == expected
    assert chunks_seen == payload_updates
    expected_counts = [channels, n, channels * n, 0 if dense else channels * n, 0, 0]
    assert element_counts == expected_counts and descriptor_index == 5
    for index, count in enumerate(expected_counts):
        literal = original_sha()
        for position in range(count):
            value = position + 1 if index == 0 else 1 if index == 3 else position
            literal.update(
                struct.pack("<q" if index == 1 else "<f" if index == 3 else "<d", value)
            )
        assert stream_hashes[index].hexdigest() == literal.hexdigest()
    assert masks_seen > chunks_seen
    if not dense:
        assert short_parent_retained and closed_parent_retained
        assert max_required_closed >= limit * 8
        assert max_charged_closed >= max_required_closed
    assert dense_intervals == (
        [(0, limit), (limit, 2 * limit), (2 * limit, 3 * limit)] if dense else []
    )
    # Finite visible storage + conservative iterator capacity, not total RSS.
    assert peak <= 27 * limit
    with capsys.disabled():
        print(
            json.dumps(
                {
                    "dense": dense,
                    "tracked_conservative_peak": peak,
                    "required_closed_parent_capacity": max_required_closed,
                    "charged_closed_parent_capacity": max_charged_closed,
                    "exposed_scratch_peak": peak_views,
                    "chunks": chunks_seen,
                    "masks": masks_seen,
                    "short_parent_retained": short_parent_retained,
                    "closed_parent_retained": closed_parent_retained,
                }
            )
        )
