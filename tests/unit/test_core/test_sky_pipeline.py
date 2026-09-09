"""Tests for sky-model orchestration helpers."""

import concurrent.futures
import inspect
import struct

import healpy as hp
import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky import HealpixData, PointSourceData
from radiosim.core.sky.combine import pipeline as pipeline_module
from radiosim.core.sky.combine.options import PrepareSkyOptions
from radiosim.core.sky.combine.pipeline import prepare_sky_model
from radiosim.core.sky.containers._polarization_materialization import (
    complete_model_native_identity,
)
from radiosim.core.sky.containers.constants import BrightnessConversion as BC
from radiosim.core.sky.containers.model import SkyFormat, SkyModel
from radiosim.core.sky.containers.point import TangentPolarizationFrame
from radiosim.core.sky.operations import parallel as parallel_module
from radiosim.core.sky.operations.parallel import (
    LoaderExecutionRecord,
    WorkerPolicyError,
    load_models_parallel,
    recommend_executor_for_loaders,
)


@pytest.fixture
def precision():
    return PrecisionConfig.standard()


def make_healpix_model(
    *,
    nside: int = 8,
    freqs: np.ndarray | None = None,
    precision: PrecisionConfig,
) -> SkyModel:
    if freqs is None:
        freqs = np.array([100e6, 101e6], dtype=np.float64)
    npix = hp.nside2npix(nside)
    return SkyModel(
        healpix=HealpixData(
            maps=np.ones((len(freqs), npix), dtype=np.float32),
            nside=nside,
            frequencies=freqs,
        ),
        reference_frequency=float(freqs[0]),
        model_name="diffuse",
        precision=precision,
    )


class TestPrepareSkyModel:
    def test_existing_healpix_explicit_frequency_grid_is_respected(self, precision):
        sky = make_healpix_model(
            freqs=np.array([100e6, 101e6], dtype=np.float64),
            precision=precision,
        )
        with pytest.raises(ValueError, match="frequency grid does not match"):
            prepare_sky_model(
                [sky],
                representation=SkyFormat.HEALPIX,
                nside=None,
                frequencies=np.array([100e6, 101e6, 102e6]),
            )

    def test_representation_none_returns_single_model_unchanged(self, precision):
        sky = make_healpix_model(precision=precision)
        out = prepare_sky_model([sky])
        assert out is sky

    def test_representation_none_preserves_hybrid_inputs(self, precision):
        # One healpix model + one point model → hybrid output preserved.
        healpix_sky = make_healpix_model(precision=precision)
        point_sky = SkyModel(
            point=PointSourceData(
                ra_rad=np.array([0.5]),
                dec_rad=np.array([0.1]),
                flux=np.array([1.0]),
                spectral_index=np.array([-0.7]),
                stokes_q=np.array([0.0]),
                stokes_u=np.array([0.0]),
                stokes_v=np.array([0.0]),
                ref_freq=np.array([100e6]),
            ),
            reference_frequency=100e6,
            model_name="src",
            precision=precision,
        )
        out = prepare_sky_model(
            [healpix_sky, point_sky],
            representation=None,
            mixed_model_policy="allow",
            precision=precision,
        )
        # Hybrid: both formats populated.
        assert SkyFormat.POINT_SOURCES in out.formats
        assert SkyFormat.HEALPIX in out.formats

    @pytest.mark.parametrize("field", ("beam_fwhm_rad", "nside_safety_factor"))
    def test_removed_beam_advisor_fields_are_strictly_rejected(
        self,
        field,
        precision,
    ):
        sky = make_healpix_model(precision=precision)

        with pytest.raises(ValidationError, match=field):
            PrepareSkyOptions(**{field: 1.0})
        with pytest.raises(TypeError, match=field):
            prepare_sky_model(
                [sky],
                representation=SkyFormat.HEALPIX,
                nside=8,
                **{field: 1.0},
            )


# ---------------------------------------------------------------------------
# Tier 6C -- the loader driver consumes an explicit, recorded worker policy.
#
# ``Tier6HybridRuntimePlan.md`` Section 11.2: no hard-coded pool size survives,
# ``max_workers`` has no default, ``executor: auto`` keeps the registry-driven
# choice, an *explicit* process request is rejected rather than silently
# degraded, and an auto degradation is recorded rather than only logged.
# ---------------------------------------------------------------------------


def _synthetic_requests(count: int) -> list[tuple[str, dict[str, object]]]:
    return [
        (
            "test_sources",
            {"num_sources": 2 + index, "distribution": "uniform", "seed": 10 + index},
        )
        for index in range(count)
    ]


def _record_pool_sizes(monkeypatch, sizes: list[int]) -> None:
    """Capture the pool size the driver actually asks a thread pool for."""
    real = concurrent.futures.ThreadPoolExecutor

    class RecordingThreadPool(real):  # type: ignore[misc, valid-type]
        def __init__(self, max_workers=None, *args, **kwargs):
            sizes.append(max_workers)
            super().__init__(max_workers, *args, **kwargs)

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", RecordingThreadPool)


class TestLoaderWorkerPolicy:
    def test_load_models_parallel_has_no_worker_count_default(self):
        """Section 27 W7 -- no caller can silently inherit a pool size."""
        signature = inspect.signature(load_models_parallel)
        assert signature.parameters["max_workers"].default is inspect.Parameter.empty
        assert signature.parameters["max_workers"].annotation == "int"
        assert "8" not in inspect.getsource(load_models_parallel)

    @pytest.mark.parametrize(("max_workers", "expected"), [(1, 1), (2, 2), (8, 4)])
    def test_the_requested_worker_count_sizes_the_pool(
        self, monkeypatch, max_workers, expected, precision
    ):
        """The knob is not a no-op: it reaches ``ThreadPoolExecutor``."""
        sizes: list[int] = []
        _record_pool_sizes(monkeypatch, sizes)

        models, record = load_models_parallel(
            _synthetic_requests(4),
            max_workers,
            precision=precision,
            strict=True,
            executor="thread",
        )

        assert sizes == [expected]
        assert record.max_workers == max_workers
        assert len(models) == 4

    @pytest.mark.parametrize("max_workers", [1, 2, 8])
    def test_the_worker_count_never_changes_the_loaded_models(
        self, max_workers, precision
    ):
        """Section 11.5 -- index-preserving order under any pool size."""
        reference, _ = load_models_parallel(
            _synthetic_requests(4),
            1,
            precision=precision,
            strict=True,
            executor="thread",
        )
        models, _ = load_models_parallel(
            _synthetic_requests(4),
            max_workers,
            precision=precision,
            strict=True,
            executor="thread",
        )

        assert [model.model_name for model in models] == [
            model.model_name for model in reference
        ]
        for produced, expected in zip(models, reference, strict=True):
            assert produced.point is not None
            assert expected.point is not None
            assert np.array_equal(produced.point.ra_rad, expected.point.ra_rad)
            assert np.array_equal(produced.point.dec_rad, expected.point.dec_rad)
            assert np.array_equal(produced.point.flux, expected.point.flux)

    def test_a_non_positive_worker_count_is_rejected_before_any_loader_runs(
        self, monkeypatch, precision
    ):
        def forbidden(*args, **kwargs):
            pytest.fail("a rejected worker policy reached a loader")

        monkeypatch.setattr(parallel_module, "_run_one_loader", forbidden)

        for value in (0, -1):
            with pytest.raises(WorkerPolicyError, match="positive integer"):
                load_models_parallel(
                    _synthetic_requests(1),
                    value,
                    precision=precision,
                    strict=True,
                    executor="thread",
                )

    def test_auto_keeps_the_registry_driven_executor_choice(self, precision):
        assert recommend_executor_for_loaders(_synthetic_requests(1)) == "process"

        _models, record = load_models_parallel(
            _synthetic_requests(1),
            1,
            precision=precision,
            strict=True,
            executor="auto",
        )

        assert record.requested_executor == "auto"
        assert record.actual_executor == "process"
        assert record.degraded_reason is None

    def test_an_explicit_thread_request_overrides_the_recommendation(self, precision):
        _models, record = load_models_parallel(
            _synthetic_requests(1),
            1,
            precision=precision,
            strict=True,
            executor="thread",
        )

        assert record.requested_executor == "thread"
        assert record.actual_executor == "thread"
        assert record.degraded_reason is None

    def test_an_explicit_process_request_is_rejected_when_kwargs_cannot_pickle(
        self, monkeypatch, precision
    ):
        """Section 27 E8 -- the exact Section 18.3 message, verbatim."""
        monkeypatch.setattr(
            parallel_module,
            "_pickle_probe",
            lambda *args, **kwargs: ("test_sources", "cannot pickle 'function' object"),
        )

        def forbidden(*args, **kwargs):
            pytest.fail("a rejected executor policy reached a loader")

        monkeypatch.setattr(parallel_module, "_run_one_loader", forbidden)

        with pytest.raises(WorkerPolicyError) as excinfo:
            load_models_parallel(
                _synthetic_requests(1),
                1,
                precision=precision,
                strict=True,
                executor="process",
            )

        assert str(excinfo.value) == (
            "execution.sky_loading.executor=process was requested explicitly, but "
            "loader arguments for test_sources cannot be pickled: cannot pickle "
            "'function' object. Use execution.sky_loading.executor=auto to allow a "
            "thread fallback, or thread to force it."
        )

    def test_a_real_unpicklable_kwarg_reaches_the_same_rejection(self, precision):
        requests = [("test_sources", {"num_sources": 1, "region": lambda: None})]

        with pytest.raises(WorkerPolicyError) as excinfo:
            load_models_parallel(
                requests,
                1,
                precision=precision,
                strict=True,
                executor="process",
            )

        message = str(excinfo.value)
        assert message.startswith(
            "execution.sky_loading.executor=process was requested explicitly, but "
            "loader arguments for test_sources cannot be pickled: "
        )
        assert message.endswith(
            "Use execution.sky_loading.executor=auto to allow a thread fallback, or "
            "thread to force it."
        )

    def test_an_auto_degradation_is_recorded_not_merely_logged(
        self, monkeypatch, precision
    ):
        monkeypatch.setattr(
            parallel_module,
            "_pickle_probe",
            lambda *args, **kwargs: ("test_sources", "cannot pickle 'function' object"),
        )

        models, record = load_models_parallel(
            _synthetic_requests(1),
            1,
            precision=precision,
            strict=True,
            executor="auto",
        )

        assert len(models) == 1
        assert record.requested_executor == "auto"
        assert record.actual_executor == "thread"
        assert record.degraded_reason == (
            "loader arguments for test_sources cannot be pickled: "
            "cannot pickle 'function' object"
        )


class TestLoaderExecutionRecord:
    def test_the_record_is_frozen_and_json_safe(self):
        record = LoaderExecutionRecord(
            requested_executor="auto",
            actual_executor="thread",
            max_workers=3,
            degraded_reason="because",
        )

        with pytest.raises(AttributeError):
            record.max_workers = 4  # type: ignore[misc]
        assert record.to_snapshot() == {
            "requested_executor": "auto",
            "actual_executor": "thread",
            "max_workers": 3,
            "degraded_reason": "because",
        }

    def test_the_record_round_trips_through_a_history_line(self):
        record = LoaderExecutionRecord(
            requested_executor="process",
            actual_executor="process",
            max_workers=8,
        )
        line = record.to_history_line()

        assert line.startswith("RADIOSIM_SKY_LOADER_JSON=")
        assert LoaderExecutionRecord.from_history(("unrelated", line)) == record
        assert LoaderExecutionRecord.from_history(("unrelated",)) is None

    def test_a_malformed_history_line_decodes_to_nothing(self):
        assert (
            LoaderExecutionRecord.from_history(("RADIOSIM_SKY_LOADER_JSON={",)) is None
        )
        assert (
            LoaderExecutionRecord.from_history(
                ('RADIOSIM_SKY_LOADER_JSON={"max_workers": 1}',)
            )
            is None
        )

    def test_the_record_rejects_an_impossible_policy(self):
        with pytest.raises(ValueError, match="positive integer"):
            LoaderExecutionRecord(
                requested_executor="auto",
                actual_executor="thread",
                max_workers=0,
            )
        with pytest.raises(ValueError, match="actual_executor"):
            LoaderExecutionRecord(
                requested_executor="auto",
                actual_executor="auto",  # type: ignore[arg-type]
                max_workers=1,
            )


def _native_prepare_input(hybrid: bool = False) -> tuple[SkyModel, NDArray[np.float64]]:
    backing = np.full((1, 3), 2.0)
    owner = HealpixData(
        maps=backing.view(),
        q_maps=np.ones((1, 3)),
        u_maps=np.full((1, 3), -0.5),
        v_maps=np.full((1, 3), 0.25),
        hpx_inds=np.array([4, 0, 9]),
        frequencies=np.array([100e6]),
        nside=1,
    )
    point = (
        PointSourceData(
            ra_rad=np.array([0.0]),
            dec_rad=np.array([0.0]),
            flux=np.array([1.0]),
            spectral_index=np.array([-0.7]),
            stokes_q=np.zeros(1),
            stokes_u=np.zeros(1),
            stokes_v=np.zeros(1),
            ref_freq=np.array([100e6]),
        )
        if hybrid
        else None
    )
    model = SkyModel(healpix=owner, point=point, precision=PrecisionConfig.precise())
    bound = complete_model_native_identity(
        model,
        source_profile="radiosim_ne_iau_v1",
        tangent_frame=TangentPolarizationFrame.canonical("icrs"),
    )
    assert bound.healpix is not None and np.shares_memory(backing, bound.healpix.maps)
    assert backing.flags.writeable and not bound.healpix.maps.flags.writeable
    return bound, backing


def _native_prepare_state(model: SkyModel) -> tuple[object, ...]:
    owner = model.healpix
    assert owner is not None
    evidence = owner.polarization_materialization
    frame = owner.tangent_polarization_frame
    arrays = (
        owner.maps,
        owner.q_maps,
        owner.u_maps,
        owner.v_maps,
        owner.hpx_inds,
        owner.frequencies,
    )
    return (
        tuple(
            None
            if array is None
            else (
                id(array),
                array.dtype.str,
                array.shape,
                array.strides,
                str(array.flags),
                array.tobytes(),
            )
            for array in arrays
        ),
        model.brightness_conversion,
        owner.nside,
        owner.ordering,
        owner.coordinate_frame,
        owner.i_brightness_conversion,
        owner.q_brightness_conversion,
        owner.u_brightness_conversion,
        owner.v_brightness_conversion,
        owner.i_unit,
        owner.q_unit,
        owner.u_unit,
        owner.v_unit,
        id(frame),
        None if frame is None else frame.as_mapping(),
        id(evidence),
        None
        if evidence is None
        else (
            evidence.record.as_mapping(),
            evidence.declaration_json,
            evidence.identity_parameters_json,
            evidence.payload_metadata_json,
            evidence.brightness_conversion,
        ),
    )


@pytest.mark.parametrize(
    "hybrid,target",
    [
        (False, None),
        (False, SkyFormat.HEALPIX),
        (True, None),
        (True, SkyFormat.POINT_SOURCES),
    ],
)
def test_prepare_preserves_exact_native_identity(
    hybrid: bool,
    target: SkyFormat | None,
) -> None:
    model, backing = _native_prepare_input(hybrid)
    assert model.healpix is not None
    evidence = model.healpix.polarization_materialization
    assert evidence is not None
    owner = model.healpix
    assert owner.hpx_inds is not None
    assert owner.hpx_inds.tobytes() == struct.pack("<3q", 4, 0, 9)
    assert owner.frequencies.tobytes() == struct.pack("<d", 100e6)
    for array, value in zip(
        (owner.maps, owner.q_maps, owner.u_maps, owner.v_maps),
        (2.0, 1.0, -0.5, 0.25),
        strict=True,
    ):
        assert array is not None and array.dtype == np.dtype("float64")
        assert array.tobytes() == struct.pack("<3d", value, value, value)
    before = _native_prepare_state(model)
    result = prepare_sky_model([model], representation=target)
    assert result is model and result.healpix is owner
    assert result.healpix is not None
    assert result.healpix.polarization_materialization is evidence
    assert before == _native_prepare_state(result)
    assert backing.tobytes() == struct.pack("<3d", 2.0, 2.0, 2.0)


@pytest.mark.parametrize("hybrid", [False, True])
def test_prepare_rejects_stale_native_before_fast_return(hybrid: bool) -> None:
    model, backing = _native_prepare_input(hybrid)
    assert prepare_sky_model([model]) is model
    backing[0, 0] = 3.0
    assert model.healpix is not None and model.healpix.maps[0, 0] == 3.0
    with pytest.raises(ValueError, match="materialization"):
        _ = prepare_sky_model([model])


@pytest.mark.parametrize("second_attached", [False, True])
def test_prepare_refuses_attached_combine_before_engine(
    monkeypatch: pytest.MonkeyPatch,
    second_attached: bool,
) -> None:
    bound, _ = _native_prepare_input()
    second = (
        _native_prepare_input()[0]
        if second_attached
        else make_healpix_model(
            nside=1,
            freqs=np.array([100e6]),
            precision=PrecisionConfig.precise(),
        )
    )

    def forbidden(*args: object, **kwargs: object) -> SkyModel:
        raise AssertionError("combine engine must not consume attached input")

    monkeypatch.setattr(pipeline_module, "_combine_models", forbidden)
    with pytest.raises(ValueError, match="requires child materialization"):
        _ = prepare_sky_model(
            [bound, second], assume_disjoint=True, mixed_model_policy="allow"
        )


def test_prepare_refuses_attached_conversion_before_converter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound, _ = _native_prepare_input()

    def forbidden(*args: object, **kwargs: object) -> SkyModel:
        raise AssertionError("converter must not consume attached input")

    monkeypatch.setattr(pipeline_module, "materialize_point_sources_model", forbidden)
    with pytest.raises(ValueError, match="requires child materialization"):
        _ = prepare_sky_model([bound], representation="point_sources", allow_lossy=True)
    with pytest.raises(ValueError, match="allow_lossy=True"):
        _ = prepare_sky_model([bound], representation="point_sources")


def test_prepare_bound_grid_errors_and_raw_passthrough_remain() -> None:
    bound, _ = _native_prepare_input()
    with pytest.raises(ValueError, match="nside does not match"):
        _ = prepare_sky_model([bound], nside=2)
    with pytest.raises(ValueError, match="frequency grid does not match"):
        _ = prepare_sky_model([bound], frequencies=np.array([101e6]))
    raw = SkyModel(
        healpix=HealpixData(
            maps=np.full((1, 3), 2.0),
            q_maps=np.ones((1, 3)),
            u_maps=np.full((1, 3), -0.5),
            v_maps=np.full((1, 3), 0.25),
            hpx_inds=np.array([4, 0, 9]),
            frequencies=np.array([100e6]),
            nside=1,
        ),
        precision=PrecisionConfig.precise(),
    )
    assert raw.healpix is not None and raw.healpix.polarization_materialization is None
    assert raw.healpix.tangent_polarization_frame is None
    assert raw.healpix.q_maps is not None and raw.healpix.u_maps is not None
    assert raw.healpix.v_maps is not None and raw.healpix.hpx_inds is not None
    before = _native_prepare_state(raw)
    assert prepare_sky_model([raw]) is raw
    assert before == _native_prepare_state(raw)
    assert raw.healpix.tangent_polarization_frame is None
    assert raw.healpix.polarization_materialization is None


def test_prepare_checks_actual_bound_context_not_passthrough_override() -> None:
    bound, _ = _native_prepare_input()
    assert bound.brightness_conversion is BC.PLANCK
    assert prepare_sky_model([bound], brightness_conversion="rayleigh-jeans") is bound
    object.__setattr__(bound, "brightness_conversion", BC.RAYLEIGH_JEANS)
    with pytest.raises(ValueError, match="materialization"):
        _ = prepare_sky_model([bound])
