"""Independent canonical identity record oracle; no loader or scientific run."""

import hashlib
import json
import struct
import sys
from collections.abc import Callable, Iterator
from dataclasses import replace
from types import FrameType, FunctionType, TracebackType
from typing import Literal, Protocol, TypeVar, cast

import numpy as np
import pytest
from numpy.typing import NDArray

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


def test_identity_requires_exact_profile_and_frame_string_types() -> None:
    class DeclaredString(str):
        pass

    owner = _identity_owner()
    frame = TangentPolarizationFrame.canonical("icrs")
    with pytest.raises(ValueError, match="explicit canonical source profile"):
        _ = complete_native_identity(
            owner,
            brightness_conversion=BC.PLANCK,
            source_profile=cast(
                Literal["radiosim_ne_iau_v1"], DeclaredString("radiosim_ne_iau_v1")
            ),
            tangent_frame=frame,
        )
    for invalid in (DeclaredString("north_east"), 7):
        altered = TangentPolarizationFrame.canonical("icrs")
        object.__setattr__(altered, "axes", invalid)
        with pytest.raises(ValueError, match="literals must be exact strings"):
            _ = complete_native_identity(
                owner,
                brightness_conversion=BC.PLANCK,
                source_profile="radiosim_ne_iau_v1",
                tangent_frame=altered,
            )


_BoundaryValue = TypeVar("_BoundaryValue")


def _boundary_value(reference: _BoundaryValue, value: object) -> _BoundaryValue:
    del reference
    return cast(_BoundaryValue, value)


def test_identity_consumer_refuses_nonreceipt() -> None:
    owner = _identity_owner()
    frame = TangentPolarizationFrame.canonical("icrs")
    receipt = complete_native_identity(
        owner,
        brightness_conversion=BC.PLANCK,
        source_profile="radiosim_ne_iau_v1",
        tangent_frame=frame,
    )
    # Explicit runtime-boundary corruption, without importing a private receipt type.
    with pytest.raises(ValueError, match="materialization mismatch"):
        require_native_identity(
            owner,
            brightness_conversion=BC.PLANCK,
            source_profile="radiosim_ne_iau_v1",
            tangent_frame=frame,
            expected=_boundary_value(receipt, object()),
        )


def test_identity_preserves_all_actual_arrays_on_success_and_refusal() -> None:
    arrays = [np.array([[value]], dtype=np.float64) for value in (2.0, 1.0, 0.0, 0.5)]
    aliases = [array.view() for array in arrays]
    owner = HealpixData(
        nside=1,
        frequencies=np.array([1.0]),
        hpx_inds=np.array([4]),
        maps=arrays[0],
        q_maps=arrays[1],
        u_maps=arrays[2],
        v_maps=arrays[3],
    )
    actual = (
        owner.frequencies,
        owner.hpx_inds,
        owner.maps,
        owner.q_maps,
        owner.u_maps,
        owner.v_maps,
    )
    assert all(array is not None for array in actual)
    for alias, array in zip(aliases, actual[2:], strict=True):
        assert array is not None and np.shares_memory(alias, array)
        assert alias.flags.writeable and not array.flags.writeable
    before = [
        (
            id(array),
            array.dtype.str,
            array.shape,
            array.strides,
            str(array.flags),
            array.tobytes(),
        )
        for array in actual
        if array is not None
    ]
    frame = TangentPolarizationFrame.canonical("icrs")
    receipt = complete_native_identity(
        owner,
        brightness_conversion=BC.PLANCK,
        source_profile="radiosim_ne_iau_v1",
        tangent_frame=frame,
    )
    require_native_identity(
        owner,
        brightness_conversion=BC.PLANCK,
        source_profile="radiosim_ne_iau_v1",
        tangent_frame=frame,
        expected=receipt,
    )
    with pytest.raises(ValueError, match="requires an explicit"):
        _ = complete_native_identity(
            owner,
            brightness_conversion=BC.PLANCK,
            source_profile="radiosim_ne_iau_v1",
            tangent_frame=None,
        )
    after = (
        owner.frequencies,
        owner.hpx_inds,
        owner.maps,
        owner.q_maps,
        owner.u_maps,
        owner.v_maps,
    )
    assert before == [
        (
            id(array),
            array.dtype.str,
            array.shape,
            array.strides,
            str(array.flags),
            array.tobytes(),
        )
        for array in after
        if array is not None
    ]
    assert all(alias.flags.writeable for alias in aliases)


def test_public_materialization_evidence_joins_explicit_context() -> None:
    from radiosim.core.sky.containers import (
        PolarizationMaterialization,
        PolarizationMaterializationEvidence,
        PolarizationOperation,
    )

    owner = _identity_owner()
    frame = TangentPolarizationFrame.canonical("icrs")
    receipt = complete_native_identity(
        owner,
        brightness_conversion=BC.PLANCK,
        source_profile="radiosim_ne_iau_v1",
        tangent_frame=frame,
    )
    assert type(cast(object, receipt)) is PolarizationMaterializationEvidence
    assert type(cast(object, receipt.record)) is PolarizationMaterialization
    assert type(cast(object, receipt.record.operations[0])) is PolarizationOperation
    assert receipt.brightness_conversion is BC.PLANCK
    for context in (BC.RAYLEIGH_JEANS, cast(BC, "planck")):
        altered = replace(receipt, brightness_conversion=context)
        assert altered.record == receipt.record
        assert altered.payload_metadata_json == receipt.payload_metadata_json
        with pytest.raises(ValueError, match="materialization mismatch"):
            require_native_identity(
                owner,
                brightness_conversion=BC.PLANCK,
                source_profile="radiosim_ne_iau_v1",
                tangent_frame=frame,
                expected=altered,
            )
    require_native_identity(
        owner,
        brightness_conversion=BC.PLANCK,
        source_profile="radiosim_ne_iau_v1",
        tangent_frame=frame,
        expected=receipt,
    )


class _IteratorContext(Protocol):
    def __next__(self) -> object: ...

    def __enter__(self) -> object: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> object: ...


class _IteratorDelegate(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> _IteratorContext: ...


class _ReductionDelegate(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...


class _LinearObservation:
    """Observe exposed chunks and masks; no C allocation or backing-size claim."""

    def __init__(self, expectations: dict[int, tuple[str, int, int]]) -> None:
        scanner = complete_native_identity.__globals__["_has_linear"]
        assert isinstance(scanner, FunctionType)
        assert (
            scanner.__code__.co_filename
            == complete_native_identity.__code__.co_filename
        )
        self.code = scanner.__code__
        self.expectations = expectations  # operand id -> label, size, nonzero index
        self.original_iterator = cast(_IteratorDelegate, np.nditer)
        self.original_any = cast(_ReductionDelegate, np.any)
        self.previous_profile = sys.getprofile()
        self.active = False
        self.entries = 0
        self.exits = 0
        self.open_contexts = 0
        self.closed_contexts = 0
        self.exit_attempts = 0
        self.exit_exception_ids: list[int | None] = []
        self.pending_at_exit: list[bool] = []
        self.cleanup_errors: list[str] = []
        self.mask_failure: AssertionError | None = None
        self.events: list[tuple[str, int, int]] = []
        self.current: tuple[str, int, int, int] | None = None
        self.mask_pending = False
        self.astype_receivers: list[int] = []
        self.receiver_ids: set[int] = set()

    def profile(self, frame: FrameType, event: str, arg: object) -> None:
        if frame.f_code is self.code:
            if event == "call":
                assert not self.active
                self.active = True
                self.entries += 1
            elif event == "return":
                self.active = False
                self.exits += 1
            elif event == "c_call" and getattr(arg, "__name__", None) == "astype":
                receiver = getattr(arg, "__self__", None)
                if isinstance(receiver, np.ndarray):
                    assert id(cast(object, receiver)) in self.receiver_ids
                    self.astype_receivers.append(id(cast(object, receiver)))
        if self.previous_profile is not None:
            _ = self.previous_profile(frame, event, arg)

    def iterator(self, operand: NDArray[np.float64], **kwargs: object) -> object:
        if not self.active:
            return self.original_iterator(operand, **kwargs)
        assert id(operand) in self.expectations
        assert operand.dtype == np.dtype(np.float64)
        assert kwargs == {
            "flags": ["external_loop", "buffered", "zerosize_ok"],
            "op_flags": [["readonly"]],
            "order": "C",
            "buffersize": 65_536,
        }
        self.receiver_ids.add(id(operand))
        inner = self.original_iterator(operand, **kwargs)
        label, count, nonzero = self.expectations[id(operand)]
        observer = self

        class ObservedIterator:
            def __enter__(self) -> "ObservedIterator":
                _ = inner.__enter__()
                observer.open_contexts += 1
                self.offset = 0
                return self

            def __iter__(self) -> Iterator[NDArray[np.float64]]:
                return self

            def __next__(self) -> NDArray[np.float64]:
                assert not observer.mask_pending
                block = cast(NDArray[np.float64], next(inner))
                assert block.dtype == np.dtype(np.float64)
                size = block.size
                assert 0 < size <= 65_536 and self.offset + size <= count
                observer.receiver_ids.add(id(block))
                for index, value in enumerate(block):
                    assert struct.pack("<d", float(value)) == struct.pack(
                        "<d", 1.0 if self.offset + index == nonzero else 0.0
                    )
                observer.current = (label, self.offset, size, nonzero)
                observer.mask_pending = True
                observer.events.append((label, self.offset, size))
                self.offset += size
                return block

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                traceback: TracebackType | None,
            ) -> object:
                observer.pending_at_exit.append(observer.mask_pending)
                observer.exit_exception_ids.append(None if exc is None else id(exc))
                observer.exit_attempts += 1
                try:
                    result = inner.__exit__(exc_type, exc, traceback)
                    observer.closed_contexts += 1
                    if exc is None and observer.pending_at_exit[-1]:
                        observer.cleanup_errors.append("normal_exit_with_pending_mask")
                    return result
                except BaseException as cleanup_error:
                    observer.cleanup_errors.append(type(cleanup_error).__name__)
                    if exc is None:
                        raise
                    return False  # Keep the existing causal exception primary.
                finally:
                    observer.open_contexts -= 1
                    observer.mask_pending = False
                    observer.current = None

        return ObservedIterator()

    def any(self, values: NDArray[np.bool_], *args: object, **kwargs: object) -> object:
        if self.active:
            if self.mask_failure is not None:
                raise self.mask_failure
            assert self.current is not None and self.mask_pending
            _, offset, size, nonzero = self.current
            assert values.dtype == np.dtype(np.bool_)
            assert values.shape == (size,) and values.nbytes == size <= 65_536
            assert not args and not kwargs
            for index, value in enumerate(values):
                assert bool(value) == (offset + index == nonzero)
            self.mask_pending = False
            self.current = None
        return self.original_any(values, *args, **kwargs)

    def run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        action: Callable[[], None],
        owner: HealpixData,
    ) -> None:
        before = _linear_snapshot(owner)
        primary: BaseException | None = None
        try:
            with monkeypatch.context() as patch:
                patch.setattr(np, "nditer", self.iterator)
                patch.setattr(np, "any", self.any)
                sys.setprofile(self.profile)
                action()
        except BaseException as error:
            primary = error
            raise
        finally:
            sys.setprofile(self.previous_profile)
            checks = (
                sys.getprofile() is self.previous_profile,
                cast(object, np.nditer) is self.original_iterator
                and np.any is self.original_any,
                not self.active and self.open_contexts == 0,
                self.entries == self.exits and not self.mask_pending,
                self.astype_receivers == [],
                before == _linear_snapshot(owner),
            )
            self.cleanup_errors.extend(
                f"cleanup_check_{index}"
                for index, valid in enumerate(checks)
                if not valid
            )
            if primary is None:
                assert not self.cleanup_errors


def _linear_snapshot(owner: HealpixData) -> list[tuple[object, ...]]:
    # Outside scanner profiling: these hashes are preservation checks, not scratch measurements.
    arrays = {
        "frequencies": owner.frequencies,
        "hpx_inds": owner.hpx_inds,
        "I": owner.maps,
        "Q": owner.q_maps,
        "U": owner.u_maps,
        "V": owner.v_maps,
    }
    return [
        (name, None)
        if array is None
        else (
            name,
            id(array),
            array.dtype.str,
            array.shape,
            array.strides,
            str(array.flags),
            hashlib.sha256(array.tobytes()).hexdigest(),
        )
        for name, array in arrays.items()
    ]


def _linear_owner(
    q: NDArray[np.float64] | None, u: NDArray[np.float64] | None, count: int
) -> HealpixData:
    return HealpixData(
        nside=128,
        frequencies=np.array([1.0]),
        hpx_inds=np.arange(count),
        maps=np.zeros((1, count)),
        q_maps=q,
        u_maps=u,
    )


def _complete_linear(owner: HealpixData) -> None:
    _ = complete_native_identity(
        owner,
        brightness_conversion=BC.PLANCK,
        source_profile="radiosim_ne_iau_v1",
        tangent_frame=TangentPolarizationFrame.canonical("icrs"),
    )


@pytest.mark.parametrize("strided", [False, True])
def test_linear_late_u_has_ordered_bounded_exposure(
    monkeypatch: pytest.MonkeyPatch,
    strided: bool,
) -> None:
    count = 131_075
    q_backing = np.zeros((1, count * (2 if strided else 1)))
    u_backing = np.zeros_like(q_backing)
    q = q_backing[:, ::2] if strided else q_backing
    u = u_backing[:, ::2] if strided else u_backing
    u[0, -1] = 1.0
    owner = _linear_owner(q, u, count)
    assert owner.q_maps is not None and owner.u_maps is not None
    assert np.shares_memory(q, owner.q_maps) and np.shares_memory(u, owner.u_maps)
    assert owner.q_maps.strides == q.strides and owner.u_maps.strides == u.strides
    assert owner.q_maps.flags.c_contiguous is (not strided)
    observer = _LinearObservation(
        {
            id(owner.q_maps): ("Q", count, -1),
            id(owner.u_maps): ("U", count, count - 1),
        }
    )
    observer.run(monkeypatch, lambda: _complete_linear(owner), owner)
    assert observer.entries == 1 and observer.closed_contexts == 2
    assert observer.pending_at_exit == [False, False]
    labels = [label for label, _, _ in observer.events]
    assert labels == sorted(labels)
    for label in ("Q", "U"):
        assert sum(size for name, _, size in observer.events if name == label) == count


def test_linear_observer_mask_failure_preserves_cause_and_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _linear_owner(np.zeros((1, 1)), np.zeros((1, 1)), 1)
    assert owner.q_maps is not None
    marker = AssertionError("controlled observer mask boundary")
    original = cast(_IteratorDelegate, np.nditer)
    delegated: list[int] = []

    def witnessed_iterator(*args: object, **kwargs: object) -> object:
        inner = original(*args, **kwargs)

        class WitnessedContext:
            def __iter__(self) -> "WitnessedContext":
                return self

            def __next__(self) -> object:
                return next(inner)

            def __enter__(self) -> object:
                return inner.__enter__()

            def __exit__(self, *exc_args: object) -> object:
                result = inner.__exit__(
                    *cast(
                        tuple[
                            type[BaseException] | None,
                            BaseException | None,
                            TracebackType | None,
                        ],
                        exc_args,
                    )
                )
                if exc_args[1] is marker:
                    delegated.append(id(exc_args[1]))
                return result

        return WitnessedContext()

    with monkeypatch.context() as patch:
        patch.setattr(np, "nditer", witnessed_iterator)
        observer = _LinearObservation({id(owner.q_maps): ("Q", 1, -1)})
        observer.mask_failure = marker
        with pytest.raises(AssertionError) as caught:
            observer.run(monkeypatch, lambda: _complete_linear(owner), owner)
        assert np.nditer is witnessed_iterator
    assert delegated == [id(marker)] and np.nditer is original
    assert caught.value is marker
    assert observer.exit_exception_ids == [id(marker)]
    assert observer.pending_at_exit == [True]
    assert observer.exit_attempts == observer.closed_contexts == 1
    assert observer.open_contexts == 0 and observer.entries == observer.exits == 1
    assert not observer.mask_pending and observer.current is None
    assert observer.cleanup_errors == []
    assert sys.getprofile() is observer.previous_profile
    assert np.any is observer.original_any


def test_linear_early_q_return_closes_without_u_exposure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    count = 131_075
    q = np.zeros((1, count))
    q[0, 0] = 1.0
    owner = _linear_owner(q, np.zeros_like(q), count)
    assert owner.q_maps is not None and owner.u_maps is not None
    observer = _LinearObservation(
        {
            id(owner.q_maps): ("Q", count, 0),
            id(owner.u_maps): ("U", count, -1),
        }
    )
    observer.run(monkeypatch, lambda: _complete_linear(owner), owner)
    assert observer.entries == observer.exits == 1
    assert observer.exit_attempts == observer.closed_contexts == 1
    assert len(observer.events) == 1
    label, offset, size = observer.events[0]
    assert label == "Q" and offset == 0 and 0 < size <= 65_536 < count
    assert observer.pending_at_exit == [False]
    assert observer.exit_exception_ids == [None] and observer.cleanup_errors == []


@pytest.mark.parametrize("empty", [False, True])
def test_linear_absent_or_empty_planes_enter_without_exposed_blocks(
    monkeypatch: pytest.MonkeyPatch,
    empty: bool,
) -> None:
    q = np.zeros((1, 0)) if empty else None
    u = np.zeros((1, 0)) if empty else None
    owner = _linear_owner(q, u, 0 if empty else 1)
    expectations: dict[int, tuple[str, int, int]] = {}
    if empty:
        assert owner.q_maps is not None and owner.u_maps is not None
        assert owner.q_maps is not owner.u_maps
        expectations = {
            id(owner.q_maps): ("Q", 0, -1),
            id(owner.u_maps): ("U", 0, -1),
        }
    else:
        assert owner.q_maps is None and owner.u_maps is None
    observer = _LinearObservation(expectations)
    observer.run(monkeypatch, lambda: _complete_linear(owner), owner)
    assert observer.entries == observer.exits == 1 and observer.events == []
    contexts = 2 if empty else 0
    assert observer.exit_attempts == observer.closed_contexts == contexts
    assert observer.pending_at_exit == [False] * contexts
    assert observer.exit_exception_ids == [None] * contexts
    assert observer.cleanup_errors == []


@pytest.mark.parametrize("component", ["u_maps", "v_maps"])
def test_linear_nonfinite_is_rejected_before_actual_scanner_entry(
    monkeypatch: pytest.MonkeyPatch,
    component: str,
) -> None:
    owner = _linear_owner(np.zeros((1, 1)), np.zeros((1, 1)), 1)
    bad = np.array([[np.nan]])
    bad.flags.writeable = False
    object.__setattr__(owner, component, bad)
    observer = _LinearObservation({})
    with pytest.raises(ValueError, match="nonfinite"):
        observer.run(monkeypatch, lambda: _complete_linear(owner), owner)
    assert observer.entries == observer.exits == 0
    assert observer.events == [] and observer.current is None
    assert observer.exit_attempts == observer.closed_contexts == 0
    assert observer.pending_at_exit == observer.exit_exception_ids == []
    assert observer.cleanup_errors == []
