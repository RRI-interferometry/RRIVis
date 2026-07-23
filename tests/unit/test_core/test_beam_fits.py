"""Standalone BeamFITS load and provenance tests for Tier 3D."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from radiosim.core.beam import (
    BeamAngularDomainError,
    BeamDependencyError,
    BeamFileChangedError,
    BeamFileProvenance,
    BeamFileReadError,
    BeamFrequencyDomainError,
    BeamNormalizationError,
    LoadedBeamHandlerState,
    NonFiniteBeamResponseError,
    ResolvedFITSBeamDefinition,
    UnsupportedBeamBasisError,
    UnsupportedBeamCoordinateError,
    UnsupportedBeamFeedError,
    UnsupportedBeamPrecisionError,
    UnsupportedBeamTypeError,
)
from radiosim.core.beam import models as beam_models
from radiosim.core.precision import JonesPrecision, PrecisionConfig
from tests.fixtures.beamfits import (
    BeamScienceVariant,
    UnsupportedBeamVariant,
    build_beam_variant,
    build_scalar_efield_uvbeam,
    write_scalar_efield_beamfits,
)


def _definition(path: Path, *, frequency_interpolation: str = "linear"):
    normalized = path.resolve(strict=False)
    payload = {
        "path": normalized,
        "normalization": "peak",
        "angular_interpolation": "bilinear",
        "frequency_interpolation": frequency_interpolation,
    }
    return ResolvedFITSBeamDefinition(
        "fits",
        normalized,
        "peak",
        "bilinear",
        frequency_interpolation,
        "beams.beam.path",
        beam_models._definition_fingerprint("fits", payload),
    )


class _MemoryLoader:
    def __init__(self, beam: Any, *, source_to_change: Path | None = None) -> None:
        self.beam = beam
        self.source_to_change = source_to_change
        self.paths: list[Path] = []

    def read(self, path: Path):
        self.paths.append(path)
        if self.source_to_change is not None:
            with self.source_to_change.open("ab") as stream:
                stream.write(b"changed during dependency read")
        return self.beam


def _load_memory_variant(
    tmp_path: Path,
    variant: UnsupportedBeamVariant,
    *,
    observations: tuple[float, ...] = (100e6,),
    interpolation: str = "linear",
):
    source = tmp_path / f"{variant.value}.beamfits"
    source.write_bytes(f"fixed transport for {variant.value}".encode())
    loader = _MemoryLoader(build_beam_variant(variant).beam)
    from radiosim.core.beam.fits import _load_fits_handler

    return _load_fits_handler(
        _definition(source, frequency_interpolation=interpolation),
        observation_frequencies_hz=observations,
        precision=PrecisionConfig.standard(),
        loader=loader,
        handler_ordinal=0,
    )


def test_valid_generated_beamfits_loads_through_private_standalone_seam(
    tmp_path: Path,
) -> None:
    written = write_scalar_efield_beamfits(tmp_path)
    from radiosim.core.beam.fits import _load_fits_handler

    loaded = _load_fits_handler(
        _definition(written.path),
        observation_frequencies_hz=(100e6, 105e6, 130e6),
        precision=PrecisionConfig.standard(),
        handler_ordinal=0,
    )

    assert loaded.state.kind == "fits"
    assert loaded.state.file.sha256 == written.sha256
    assert loaded.state.file.native_dtype == "complex128"


def test_loaded_public_provenance_is_immutable_and_dependency_free(
    tmp_path: Path,
) -> None:
    written = write_scalar_efield_beamfits(tmp_path, dtype=np.complex64)
    from radiosim.core.beam.fits import _load_fits_handler

    loaded = _load_fits_handler(
        _definition(written.path),
        observation_frequencies_hz=(100e6,),
        precision=PrecisionConfig.fast(),
        handler_ordinal=3,
    )

    assert loaded.state.handler_id.startswith("beam-0003-")
    assert loaded.state.file.native_dtype == "complex64"
    assert not hasattr(loaded.state, "evaluator")
    with pytest.raises((AttributeError, TypeError)):
        loaded.state.handler_id = "changed"  # type: ignore[misc]


def test_public_provenance_field_order_and_detached_snapshot(tmp_path: Path) -> None:
    written = write_scalar_efield_beamfits(tmp_path)
    from radiosim.core.beam.fits import _load_fits_handler

    loaded = _load_fits_handler(
        _definition(written.path),
        observation_frequencies_hz=(100e6,),
        precision=PrecisionConfig.standard(),
        handler_ordinal=0,
    )

    assert tuple(field.name for field in dataclasses.fields(BeamFileProvenance)) == (
        "resolved_path",
        "size_bytes",
        "sha256",
        "pyuvdata_version",
        "beam_type",
        "antenna_type",
        "pixel_coordinate_system",
        "mount_type",
        "data_normalization",
        "feed_array",
        "x_orientation",
        "data_shape",
        "native_dtype",
        "frequency_min_hz",
        "frequency_max_hz",
        "frequency_count",
        "azimuth_step_rad",
        "zenith_angle_step_rad",
        "zenith_angle_max_rad",
        "basis_tolerance",
        "scalar_absolute_tolerance",
        "scalar_relative_tolerance",
        "normalization_absolute_tolerance",
    )
    assert tuple(
        field.name for field in dataclasses.fields(LoadedBeamHandlerState)
    ) == (
        "handler_id",
        "kind",
        "definition_fingerprint",
        "scientific_fingerprint",
        "file",
        "voltage_feature_scale_by_frequency",
    )
    snapshot = loaded.state.to_snapshot()
    assert isinstance(snapshot["file"], dict)
    assert snapshot["file"]["resolved_path"] == str(written.path)
    snapshot["file"]["feed_array"].append("caller mutation")
    assert loaded.state.file.feed_array == ("x", "y")
    assert not any(
        isinstance(value, np.ndarray)
        for value in dataclasses.astuple(loaded.state.file)
    )


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
@pytest.mark.parametrize("interpolation", ("linear", "cubic"))
def test_valid_sources_record_native_dtype_and_requested_interpolation(
    tmp_path: Path,
    dtype: Any,
    interpolation: str,
) -> None:
    written = write_scalar_efield_beamfits(tmp_path, dtype=dtype)
    from radiosim.core.beam.fits import _load_fits_handler

    loaded = _load_fits_handler(
        _definition(written.path, frequency_interpolation=interpolation),
        observation_frequencies_hz=(100e6, 105e6, 130e6),
        precision=PrecisionConfig.standard(),
        handler_ordinal=1,
    )

    assert loaded.state.file.native_dtype == np.dtype(dtype).name
    assert loaded.state.file.data_shape == (2, 2, 4, 5, 8)
    assert loaded.state.file.frequency_min_hz == 100e6
    assert loaded.state.file.frequency_max_hz == 130e6
    assert loaded.state.file.frequency_count == 4


def test_snapshot_path_mode_hash_and_cleanup_are_exact(tmp_path: Path) -> None:
    written = write_scalar_efield_beamfits(tmp_path)
    beam = build_scalar_efield_uvbeam()

    class InspectingLoader:
        requested: Path | None = None
        digest: str | None = None

        def read(self, path: Path):
            self.requested = path
            self.digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert path.name == "beam.beamfits"
            assert path.parent.name.startswith("radiosim-beam-")
            assert stat_mode(path) == 0o600
            assert path != written.path
            return beam

    def stat_mode(path: Path) -> int:
        return os.stat(path).st_mode & 0o777

    loader = InspectingLoader()
    from radiosim.core.beam.fits import _load_fits_handler

    loaded = _load_fits_handler(
        _definition(written.path),
        observation_frequencies_hz=(100e6,),
        precision=PrecisionConfig.standard(),
        loader=loader,
        handler_ordinal=0,
    )

    assert loader.digest == written.sha256 == loaded.state.file.sha256
    assert loader.requested is not None
    assert not loader.requested.exists()


def test_snapshot_is_removed_when_dependency_read_fails(tmp_path: Path) -> None:
    written = write_scalar_efield_beamfits(tmp_path)

    class FailingLoader:
        requested: Path | None = None

        def read(self, path: Path):
            self.requested = path
            raise OSError("injected unreadable snapshot")

    loader = FailingLoader()
    from radiosim.core.beam.fits import _load_fits_handler

    with pytest.raises(BeamFileReadError) as caught:
        _load_fits_handler(
            _definition(written.path),
            observation_frequencies_hz=(100e6,),
            precision=PrecisionConfig.standard(),
            loader=loader,
            handler_ordinal=0,
        )
    assert isinstance(caught.value.__cause__, OSError)
    assert loader.requested is not None
    assert not loader.requested.exists()


def test_source_change_during_dependency_read_is_a_typed_race(tmp_path: Path) -> None:
    written = write_scalar_efield_beamfits(tmp_path)
    loader = _MemoryLoader(
        build_scalar_efield_uvbeam(),
        source_to_change=written.path,
    )
    from radiosim.core.beam.fits import _load_fits_handler

    with pytest.raises(BeamFileChangedError, match="source changed"):
        _load_fits_handler(
            _definition(written.path),
            observation_frequencies_hz=(100e6,),
            precision=PrecisionConfig.standard(),
            loader=loader,
            handler_ordinal=0,
        )
    assert loader.paths and not loader.paths[0].exists()


def test_source_change_during_scientific_validation_is_a_typed_race(
    tmp_path: Path,
) -> None:
    written = write_scalar_efield_beamfits(tmp_path)
    beam = build_scalar_efield_uvbeam()
    original_check = beam.check

    def changing_check(**kwargs: Any) -> bool:
        result = original_check(**kwargs)
        with written.path.open("ab") as stream:
            stream.write(b"changed during validation")
        return result

    beam.check = changing_check
    from radiosim.core.beam.fits import _load_fits_handler

    with pytest.raises(BeamFileChangedError):
        _load_fits_handler(
            _definition(written.path),
            observation_frequencies_hz=(100e6,),
            precision=PrecisionConfig.standard(),
            loader=_MemoryLoader(beam),
            handler_ordinal=0,
        )


def test_source_change_while_snapshot_is_streamed_is_a_typed_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    written = write_scalar_efield_beamfits(tmp_path)
    from radiosim.core.beam import fits

    original_read = fits.os.read
    changed = False

    def changing_read(fd: int, size: int) -> bytes:
        nonlocal changed
        chunk = original_read(fd, size)
        if chunk and not changed:
            changed = True
            with written.path.open("ab") as stream:
                stream.write(b"changed while snapshotting")
        return chunk

    monkeypatch.setattr(fits.os, "read", changing_read)
    with pytest.raises(BeamFileChangedError):
        fits._load_fits_handler(
            _definition(written.path),
            observation_frequencies_hz=(100e6,),
            precision=PrecisionConfig.standard(),
            handler_ordinal=0,
        )


def test_cleanup_failure_is_reported_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    written = write_scalar_efield_beamfits(tmp_path)
    from radiosim.core.beam import fits

    created: list[Path] = []

    class FailingCleanupTemporaryDirectory:
        def __init__(self, *, prefix: str) -> None:
            self._real = tempfile_directory(prefix=prefix)
            self.name = self._real.name
            created.append(Path(self.name))

        def cleanup(self) -> None:
            raise OSError("injected cleanup failure")

    tempfile_directory = fits.tempfile.TemporaryDirectory
    monkeypatch.setattr(
        fits.tempfile,
        "TemporaryDirectory",
        FailingCleanupTemporaryDirectory,
    )
    with pytest.raises(BeamFileReadError, match="cleanup failed") as caught:
        fits._load_fits_handler(
            _definition(written.path),
            observation_frequencies_hz=(100e6,),
            precision=PrecisionConfig.standard(),
            handler_ordinal=0,
        )
    assert isinstance(caught.value.__cause__, OSError)
    for path in created:
        shutil.rmtree(path)


def test_post_load_source_mutation_cannot_change_private_evaluation(
    tmp_path: Path,
) -> None:
    written = write_scalar_efield_beamfits(tmp_path)
    from radiosim.core.beam.fits import _load_fits_handler

    loaded = _load_fits_handler(
        _definition(written.path),
        observation_frequencies_hz=(110e6,),
        precision=PrecisionConfig.standard(),
        handler_ordinal=0,
    )
    coordinates = (
        np.array([np.pi / 4.0]),
        np.array([0.0]),
        110e6,
        60_000.0,
    )
    before = loaded.evaluator.evaluate_numpy(*coordinates)
    written.path.write_bytes(b"unrelated replacement bytes")
    after = loaded.evaluator.evaluate_numpy(*coordinates)

    np.testing.assert_array_equal(after, before)


def test_loader_returned_object_mutation_does_not_alias_evaluator(
    tmp_path: Path,
) -> None:
    source = tmp_path / "memory.beamfits"
    source.write_bytes(b"stable source bytes")
    beam = build_scalar_efield_uvbeam()
    loader = _MemoryLoader(beam)
    from radiosim.core.beam.fits import _load_fits_handler

    loaded = _load_fits_handler(
        _definition(source),
        observation_frequencies_hz=(110e6,),
        precision=PrecisionConfig.standard(),
        loader=loader,
        handler_ordinal=0,
    )
    altitude = np.array([np.pi / 4.0])
    azimuth = np.array([0.0])
    before = loaded.evaluator.evaluate_numpy(altitude, azimuth, 110e6, 60_000.0)
    beam.data_array[...] = 0.0
    beam.basis_vector_array[...] = 0.0
    beam.freq_array[...] = -1.0
    after = loaded.evaluator.evaluate_numpy(altitude, azimuth, 110e6, 60_000.0)

    np.testing.assert_array_equal(after, before)


def test_complex64_source_is_privately_canonicalized_to_owned_complex128(
    tmp_path: Path,
) -> None:
    source = tmp_path / "complex64-memory.beamfits"
    source.write_bytes(b"complex64 memory transport")
    beam = build_scalar_efield_uvbeam(dtype=np.complex64)
    original_data = beam.data_array
    from radiosim.core.beam.fits import _load_fits_handler

    loaded = _load_fits_handler(
        _definition(source),
        observation_frequencies_hz=(100e6,),
        precision=PrecisionConfig.standard(),
        loader=_MemoryLoader(beam),
        handler_ordinal=0,
    )
    private_data = loaded.evaluator._beam.data_array

    assert loaded.state.file.native_dtype == "complex64"
    assert private_data.dtype == np.dtype(np.complex128)
    assert private_data.flags.owndata
    assert not np.shares_memory(private_data, original_data)


def test_tolerated_native_scalar_noise_is_canonicalized_to_x_diagonal(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tolerated-native-noise.beamfits"
    source.write_bytes(b"tolerated native scalar noise")
    beam = build_scalar_efield_uvbeam()
    tolerance = 0.5 * (1e-12 + 1e-10)
    residual = tolerance * beam.data_array[0, 0]
    beam.data_array[1, 0] = residual
    beam.data_array[0, 1] = -residual
    beam.data_array[1, 1] += residual
    beam.check = lambda **kwargs: True
    original_x = np.array(beam.data_array[0, 0], copy=True)
    from radiosim.core.beam.fits import _load_fits_handler

    loaded = _load_fits_handler(
        _definition(source),
        observation_frequencies_hz=(100e6,),
        precision=PrecisionConfig.standard(),
        loader=_MemoryLoader(beam),
        handler_ordinal=0,
    )
    private_data = loaded.evaluator._beam.data_array

    np.testing.assert_array_equal(private_data[0, 0], original_x)
    np.testing.assert_array_equal(private_data[1, 1], original_x)
    np.testing.assert_array_equal(private_data[0, 1], 0.0)
    np.testing.assert_array_equal(private_data[1, 0], 0.0)


def test_private_dependency_arrays_are_owned_non_memmap_and_read_only(
    tmp_path: Path,
) -> None:
    source = tmp_path / "private-read-only.beamfits"
    source.write_bytes(b"private read-only state")
    beam = build_scalar_efield_uvbeam()
    memmap_path = tmp_path / "native-data.memmap"
    mapped = np.memmap(
        memmap_path,
        mode="w+",
        dtype=np.complex128,
        shape=beam.data_array.shape,
    )
    mapped[...] = beam.data_array
    beam.data_array = mapped
    beam.check = lambda **kwargs: True
    from radiosim.core.beam.fits import _load_fits_handler

    loaded = _load_fits_handler(
        _definition(source),
        observation_frequencies_hz=(100e6,),
        precision=PrecisionConfig.standard(),
        loader=_MemoryLoader(beam),
        handler_ordinal=0,
    )
    private_beam = loaded.evaluator._beam

    for name in (
        "data_array",
        "basis_vector_array",
        "freq_array",
        "axis1_array",
        "axis2_array",
        "feed_array",
        "feed_angle",
        "bandpass_array",
    ):
        private = getattr(private_beam, name)
        source_array = getattr(beam, name)
        assert type(private) is np.ndarray
        assert private.flags.owndata
        assert private.flags.c_contiguous
        assert not private.flags.writeable
        assert not np.shares_memory(private, source_array)
        with pytest.raises(ValueError):
            private.flat[0] = private.flat[0]


def test_dependency_deepcopy_cannot_alias_mutable_source_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "hostile-deepcopy.beamfits"
    source.write_bytes(b"hostile deepcopy")
    beam = build_scalar_efield_uvbeam()
    beam.check = lambda **kwargs: True
    monkeypatch.setattr(
        type(beam),
        "__deepcopy__",
        lambda self, memo: self,
        raising=False,
    )
    from radiosim.core.beam.fits import _load_fits_handler

    with pytest.raises(BeamFileReadError, match="detach") as caught:
        _load_fits_handler(
            _definition(source),
            observation_frequencies_hz=(100e6,),
            precision=PrecisionConfig.standard(),
            loader=_MemoryLoader(beam),
            handler_ordinal=0,
        )
    assert isinstance(caught.value.__cause__, TypeError)


def test_missing_pyuvdata_at_production_load_is_typed_and_chained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    written = write_scalar_efield_beamfits(tmp_path)
    from radiosim.core.beam import runtime
    from radiosim.core.beam.fits import _load_fits_handler

    original = runtime.importlib.import_module

    def missing(name: str):
        if name == "pyuvdata":
            raise ModuleNotFoundError("injected missing dependency")
        return original(name)

    monkeypatch.setattr(runtime.importlib, "import_module", missing)
    with pytest.raises(BeamDependencyError) as caught:
        _load_fits_handler(
            _definition(written.path),
            observation_frequencies_hz=(100e6,),
            precision=PrecisionConfig.standard(),
            handler_ordinal=0,
        )
    assert isinstance(caught.value.__cause__, ModuleNotFoundError)


@pytest.mark.parametrize("version", ("3.2.0", "3.2.2", "4.0.0"))
def test_non_pinned_dependency_version_fails_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: str,
) -> None:
    source = tmp_path / f"pyuvdata-{version}.beamfits"
    source.write_bytes(version.encode())
    beam = build_scalar_efield_uvbeam()
    from radiosim.core.beam import fits

    monkeypatch.setattr(fits.importlib.metadata, "version", lambda name: version)
    with pytest.raises(BeamDependencyError, match="exactly '3.2.1'"):
        fits._load_fits_handler(
            _definition(source),
            observation_frequencies_hz=(100e6,),
            precision=PrecisionConfig.standard(),
            loader=_MemoryLoader(beam),
            handler_ordinal=0,
        )


def test_requested_complex256_fails_before_file_or_loader_work(tmp_path: Path) -> None:
    absent = tmp_path / "never-opened.beamfits"

    class ForbiddenLoader:
        def read(self, path: Path):
            raise AssertionError(path)

    precision = PrecisionConfig(
        jones=JonesPrecision(beam="float128"),
    )
    from radiosim.core.beam.fits import _load_fits_handler

    with pytest.raises(UnsupportedBeamPrecisionError, match="complex256"):
        _load_fits_handler(
            _definition(absent),
            observation_frequencies_hz=(100e6,),
            precision=precision,
            loader=ForbiddenLoader(),
            handler_ordinal=0,
        )


def test_unsupported_antenna_type_and_vector_dimensions_are_typed(
    tmp_path: Path,
) -> None:
    from radiosim.core.beam.fits import _load_fits_handler

    cases = (
        ("antenna_type", "phased_array", UnsupportedBeamTypeError),
        ("Naxes_vec", 1, UnsupportedBeamBasisError),
        ("Ncomponents_vec", 1, UnsupportedBeamBasisError),
    )
    for index, (attribute, value, error_type) in enumerate(cases):
        beam = build_scalar_efield_uvbeam()
        setattr(beam, attribute, value)
        beam.check = lambda **kwargs: True
        source = tmp_path / f"forged-{index}.beamfits"
        source.write_bytes(f"forged-{index}".encode())
        with pytest.raises(error_type):
            _load_fits_handler(
                _definition(source),
                observation_frequencies_hz=(100e6,),
                precision=PrecisionConfig.standard(),
                loader=_MemoryLoader(beam),
                handler_ordinal=0,
            )


def test_all_zero_peak_and_nonfinite_bandpass_fail_closed(tmp_path: Path) -> None:
    from radiosim.core.beam.fits import _load_fits_handler

    zero = build_scalar_efield_uvbeam()
    zero.data_array[...] = 0.0
    nonfinite_bandpass = build_scalar_efield_uvbeam()
    nonfinite_bandpass.bandpass_array[1] = np.nan
    for index, (beam, error_type) in enumerate(
        (
            (zero, BeamNormalizationError),
            (nonfinite_bandpass, NonFiniteBeamResponseError),
        )
    ):
        source = tmp_path / f"normalization-{index}.beamfits"
        source.write_bytes(f"normalization-{index}".encode())
        with pytest.raises(error_type):
            _load_fits_handler(
                _definition(source),
                observation_frequencies_hz=(100e6,),
                precision=PrecisionConfig.standard(),
                loader=_MemoryLoader(beam),
                handler_ordinal=0,
            )


@pytest.mark.parametrize(
    "variant,error_type,observations,interpolation",
    (
        (UnsupportedBeamVariant.POWER, UnsupportedBeamTypeError, (100e6,), "linear"),
        (
            UnsupportedBeamVariant.CIRCULAR_FEEDS,
            UnsupportedBeamFeedError,
            (100e6,),
            "linear",
        ),
        (
            UnsupportedBeamVariant.WRONG_FEED_ORDER,
            UnsupportedBeamFeedError,
            (100e6,),
            "linear",
        ),
        (
            UnsupportedBeamVariant.WRONG_X_ORIENTATION,
            UnsupportedBeamFeedError,
            (100e6,),
            "linear",
        ),
        (
            UnsupportedBeamVariant.WRONG_FEED_ANGLES,
            UnsupportedBeamFeedError,
            (100e6,),
            "linear",
        ),
        (
            UnsupportedBeamVariant.NON_FIXED_MOUNT,
            UnsupportedBeamFeedError,
            (100e6,),
            "linear",
        ),
        (
            UnsupportedBeamVariant.NONIDENTITY_BASIS,
            UnsupportedBeamBasisError,
            (100e6,),
            "linear",
        ),
        (
            UnsupportedBeamVariant.NONFINITE_BASIS,
            NonFiniteBeamResponseError,
            (100e6,),
            "linear",
        ),
        (
            UnsupportedBeamVariant.CROSS_POLAR,
            UnsupportedBeamBasisError,
            (100e6,),
            "linear",
        ),
        (
            UnsupportedBeamVariant.UNEQUAL_DIAGONALS,
            UnsupportedBeamBasisError,
            (100e6,),
            "linear",
        ),
        (
            UnsupportedBeamVariant.NON_PEAK_NORMALIZATION,
            BeamNormalizationError,
            (100e6,),
            "linear",
        ),
        (
            UnsupportedBeamVariant.NON_UNIT_BANDPASS,
            BeamNormalizationError,
            (100e6,),
            "linear",
        ),
        (
            UnsupportedBeamVariant.PEAK_LABEL_NONPEAK_DATA,
            BeamNormalizationError,
            (100e6,),
            "linear",
        ),
        (
            UnsupportedBeamVariant.HEALPIX,
            UnsupportedBeamCoordinateError,
            (100e6,),
            "linear",
        ),
        (
            UnsupportedBeamVariant.WRONG_COORDINATE_METADATA,
            UnsupportedBeamCoordinateError,
            (100e6,),
            "linear",
        ),
        (
            UnsupportedBeamVariant.IRREGULAR_AZIMUTH,
            UnsupportedBeamCoordinateError,
            (100e6,),
            "linear",
        ),
        (
            UnsupportedBeamVariant.IRREGULAR_ZENITH_ANGLE,
            UnsupportedBeamCoordinateError,
            (100e6,),
            "linear",
        ),
        (
            UnsupportedBeamVariant.INCOMPLETE_AZIMUTH_CLOSURE,
            UnsupportedBeamCoordinateError,
            (100e6,),
            "linear",
        ),
        (
            UnsupportedBeamVariant.SHORT_ZA,
            BeamAngularDomainError,
            (100e6,),
            "linear",
        ),
        (
            UnsupportedBeamVariant.OUT_OF_FREQUENCY_COVERAGE,
            BeamFrequencyDomainError,
            (130e6,),
            "linear",
        ),
        (
            UnsupportedBeamVariant.DUPLICATE_FREQUENCY,
            BeamFrequencyDomainError,
            (100e6,),
            "linear",
        ),
        (
            UnsupportedBeamVariant.DECREASING_FREQUENCY,
            BeamFrequencyDomainError,
            (100e6,),
            "linear",
        ),
        (
            UnsupportedBeamVariant.NONPOSITIVE_FREQUENCY,
            BeamFrequencyDomainError,
            (100e6,),
            "linear",
        ),
        (
            UnsupportedBeamVariant.NONFINITE_FREQUENCY,
            NonFiniteBeamResponseError,
            (100e6,),
            "linear",
        ),
        (
            UnsupportedBeamVariant.NONFINITE_DATA,
            NonFiniteBeamResponseError,
            (100e6,),
            "linear",
        ),
        (
            UnsupportedBeamVariant.WRONG_NATIVE_DTYPE,
            UnsupportedBeamPrecisionError,
            (100e6,),
            "linear",
        ),
        (
            UnsupportedBeamVariant.INVALID_DATA_SHAPE,
            BeamFileReadError,
            (100e6,),
            "linear",
        ),
    ),
)
def test_every_rejected_fixture_maps_to_exact_typed_error(
    tmp_path: Path,
    variant: UnsupportedBeamVariant,
    error_type: type[Exception],
    observations: tuple[float, ...],
    interpolation: str,
) -> None:
    with pytest.raises(error_type):
        _load_memory_variant(
            tmp_path,
            variant,
            observations=observations,
            interpolation=interpolation,
        )


def test_cubic_nonexact_requires_four_channels_but_exact_channel_does_not(
    tmp_path: Path,
) -> None:
    fixture = build_beam_variant(UnsupportedBeamVariant.OUT_OF_FREQUENCY_COVERAGE)
    source = tmp_path / "three-channels.beamfits"
    source.write_bytes(b"three channels")
    from radiosim.core.beam.fits import _load_fits_handler

    with pytest.raises(BeamFrequencyDomainError, match="at least 4"):
        _load_fits_handler(
            _definition(source, frequency_interpolation="cubic"),
            observation_frequencies_hz=(105e6,),
            precision=PrecisionConfig.standard(),
            loader=_MemoryLoader(fixture.beam),
            handler_ordinal=0,
        )
    loaded = _load_fits_handler(
        _definition(source, frequency_interpolation="cubic"),
        observation_frequencies_hz=(110e6,),
        precision=PrecisionConfig.standard(),
        loader=_MemoryLoader(fixture.beam),
        handler_ordinal=0,
    )
    assert loaded.state.file.frequency_count == 3


def test_strict_frequency_tolerance_snaps_below_but_interpolates_at_boundary(
    tmp_path: Path,
) -> None:
    beam = build_scalar_efield_uvbeam().select(
        freq_chans=np.array([0, 1, 2]),
        inplace=False,
    )
    assert beam is not None
    source = tmp_path / "one-channel.beamfits"
    source.write_bytes(b"one channel")
    from radiosim.core.beam.fits import _load_fits_handler

    base = 100e6
    nominal_boundary = base + 1e-6
    below = float(np.nextafter(nominal_boundary, base))
    at_or_above = float(np.nextafter(nominal_boundary, np.inf))
    snapped = _load_fits_handler(
        _definition(source, frequency_interpolation="cubic"),
        observation_frequencies_hz=(below,),
        precision=PrecisionConfig.standard(),
        loader=_MemoryLoader(beam),
        handler_ordinal=0,
    )
    assert snapped.state.file.frequency_count == 3
    with pytest.raises(BeamFrequencyDomainError, match="at least 4"):
        _load_fits_handler(
            _definition(source, frequency_interpolation="cubic"),
            observation_frequencies_hz=(at_or_above,),
            precision=PrecisionConfig.standard(),
            loader=_MemoryLoader(beam),
            handler_ordinal=0,
        )


@pytest.mark.parametrize(
    "observations,error_type",
    (
        ([], BeamFrequencyDomainError),
        ((), BeamFrequencyDomainError),
        ((100e6, 100e6), BeamFrequencyDomainError),
        ((110e6, 100e6), BeamFrequencyDomainError),
        ((99e6,), BeamFrequencyDomainError),
        ((131e6,), BeamFrequencyDomainError),
        ((np.nan,), NonFiniteBeamResponseError),
        ((np.float64(100e6),), NonFiniteBeamResponseError),
    ),
)
def test_observation_frequency_preflight_is_exact(
    tmp_path: Path,
    observations: Any,
    error_type: type[Exception],
) -> None:
    written = write_scalar_efield_beamfits(tmp_path)
    from radiosim.core.beam.fits import _load_fits_handler

    with pytest.raises(error_type):
        _load_fits_handler(
            _definition(written.path),
            observation_frequencies_hz=observations,
            precision=PrecisionConfig.standard(),
            handler_ordinal=0,
        )


def test_scientific_fingerprint_identity_and_transport_exclusions(
    tmp_path: Path,
) -> None:
    first = write_scalar_efield_beamfits(tmp_path, filename="first.beamfits")
    second_path = tmp_path / "second.beamfits"
    shutil.copyfile(first.path, second_path)
    from radiosim.core.beam.fits import _load_fits_handler

    first_loaded = _load_fits_handler(
        _definition(first.path),
        observation_frequencies_hz=(100e6, 120e6),
        precision=PrecisionConfig.standard(),
        handler_ordinal=0,
    )
    second_loaded = _load_fits_handler(
        _definition(second_path),
        observation_frequencies_hz=(100e6, 120e6),
        precision=PrecisionConfig.standard(),
        handler_ordinal=9,
    )

    assert first_loaded.state.definition_fingerprint != (
        second_loaded.state.definition_fingerprint
    )
    assert first_loaded.state.scientific_fingerprint == (
        second_loaded.state.scientific_fingerprint
    )
    assert first_loaded.state.handler_id != second_loaded.state.handler_id
    assert str(first.path) not in first_loaded.state.scientific_fingerprint


def test_scientific_fingerprint_changes_with_bytes_dtype_options_and_channels(
    tmp_path: Path,
) -> None:
    canonical = write_scalar_efield_beamfits(
        tmp_path,
        dtype=np.complex128,
        filename="canonical128.beamfits",
    )
    distinct = write_scalar_efield_beamfits(
        tmp_path,
        dtype=np.complex128,
        variant=BeamScienceVariant.DISTINCT,
        filename="distinct128.beamfits",
    )
    single = write_scalar_efield_beamfits(
        tmp_path,
        dtype=np.complex64,
        filename="canonical64.beamfits",
    )
    from radiosim.core.beam.fits import _load_fits_handler

    def load(path: Path, interpolation: str, frequencies: tuple[float, ...]):
        return _load_fits_handler(
            _definition(path, frequency_interpolation=interpolation),
            observation_frequencies_hz=frequencies,
            precision=PrecisionConfig.standard(),
            handler_ordinal=0,
        ).state.scientific_fingerprint

    baseline = load(canonical.path, "linear", (100e6, 120e6))
    assert load(distinct.path, "linear", (100e6, 120e6)) != baseline
    assert load(single.path, "linear", (100e6, 120e6)) != baseline
    assert load(canonical.path, "cubic", (100e6, 120e6)) != baseline
    assert load(canonical.path, "linear", (100e6, 110e6)) != baseline


def test_mtime_only_change_does_not_change_scientific_identity(tmp_path: Path) -> None:
    written = write_scalar_efield_beamfits(tmp_path)
    from radiosim.core.beam.fits import _load_fits_handler

    first = _load_fits_handler(
        _definition(written.path),
        observation_frequencies_hz=(100e6,),
        precision=PrecisionConfig.standard(),
        handler_ordinal=0,
    ).state.scientific_fingerprint
    current = written.path.stat()
    changed_ns = current.st_mtime_ns + 5_000_000
    os.utime(written.path, ns=(current.st_atime_ns, changed_ns))
    time.sleep(0.001)
    second = _load_fits_handler(
        _definition(written.path),
        observation_frequencies_hz=(100e6,),
        precision=PrecisionConfig.standard(),
        handler_ordinal=0,
    ).state.scientific_fingerprint

    assert second == first


def test_scientific_fingerprint_payload_contains_science_and_excludes_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    written = write_scalar_efield_beamfits(tmp_path)
    from radiosim.core.beam import fits

    captured: list[dict[str, Any]] = []
    original_digest = fits._canonical_digest

    def recording_digest(payload: dict[str, Any]) -> str:
        captured.append(payload)
        return original_digest(payload)

    monkeypatch.setattr(fits, "_canonical_digest", recording_digest)
    fits._load_fits_handler(
        _definition(written.path),
        observation_frequencies_hz=(100e6, 120e6),
        precision=PrecisionConfig.standard(),
        handler_ordinal=0,
    )

    assert len(captured) == 1
    payload = captured[0]
    rendered = repr(payload)
    assert payload["schema_version"] == "tier3-beam-v1"
    assert payload["accepted_subset_version"] == "tier3-scalar-v1"
    assert payload["fits_content_sha256"] == written.sha256
    assert payload["pyuvdata_version"] == "3.2.1"
    assert payload["contracts"]["scalar_jones"] == "e_i2_no_conjugation"
    assert payload["observation_frequencies_hz"] == (100e6, 120e6)
    assert str(written.path) not in rendered
    for excluded in (
        "resolved_path",
        "path_provenance_key",
        "definition_fingerprint",
        "st_dev",
        "st_ino",
        "st_mtime_ns",
        "st_ctime_ns",
        "handler_ordinal",
    ):
        assert excluded not in rendered


def test_fixed_scientific_digest_matches_independent_canonical_json(
    tmp_path: Path,
) -> None:
    source = tmp_path / "fixed.beamfits"
    source_bytes = b"fixed scientific fingerprint transport"
    source.write_bytes(source_bytes)
    from radiosim.core.beam.fits import _load_fits_handler

    loaded = _load_fits_handler(
        _definition(source),
        observation_frequencies_hz=(100e6, 120e6),
        precision=PrecisionConfig.standard(),
        loader=_MemoryLoader(build_scalar_efield_uvbeam()),
        handler_ordinal=0,
    )
    delta_za = np.pi / 8.0
    delta_az = np.pi / 4.0
    positive_za = np.arange(1, 5, dtype=np.float64) * delta_za
    horizontal = np.arccos(
        np.cos(positive_za) ** 2 + np.sin(positive_za) ** 2 * np.cos(delta_az)
    )
    scale = 2.0 * min(delta_za, float(np.min(horizontal)))
    expected_payload = {
        "schema_version": "tier3-beam-v1",
        "kind": "fits_handler",
        "accepted_subset_version": "tier3-scalar-v1",
        "pyuvdata_version": "3.2.1",
        "fits_content_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "validated_metadata": {
            "beam_type": "efield",
            "antenna_type": "simple",
            "pixel_coordinate_system": "az_za",
            "mount_type": "fixed",
            "data_normalization": "peak",
            "feed_array": ("x", "y"),
            "x_orientation": "east",
            "feed_angle_rad": (np.pi / 2.0, 0.0),
            "data_shape": (2, 2, 4, 5, 8),
            "native_dtype": "complex128",
            "native_frequencies_hz": (100e6, 110e6, 120e6, 130e6),
            "azimuth_start_rad": 0.0,
            "azimuth_step_rad": delta_az,
            "azimuth_count": 8,
            "zenith_angle_start_rad": 0.0,
            "zenith_angle_step_rad": delta_za,
            "zenith_angle_max_rad": np.pi / 2.0,
            "zenith_angle_count": 5,
        },
        "contracts": {
            "basis": "finite_identity_2x2",
            "scalar_jones": "e_i2_no_conjugation",
            "normalization": "positive_unit_peak_and_unit_bandpass",
            "basis_tolerance": 1e-12,
            "feed_angle_tolerance_rad": 1e-12,
            "scalar_absolute_tolerance": 1e-12,
            "scalar_relative_tolerance": 1e-10,
            "normalization_absolute_tolerance": 1e-12,
            "frequency_match_tolerance_hz": 1e-6,
            "azimuth_closure_tolerance_rad": 1e-10,
            "horizon_coverage_tolerance_rad": 1e-10,
        },
        "load_options": {
            "normalization": "peak",
            "angular_interpolation": "bilinear",
            "frequency_interpolation": "linear",
            "interpolation_function": "az_za_simple",
            "spline_opts": {"kx": 1, "ky": 1, "s": 0},
        },
        "observation_frequencies_hz": (100e6, 120e6),
        "native_grid_representation_scales": ((100e6, scale), (120e6, scale)),
    }

    def canonical(value: Any) -> Any:
        if isinstance(value, float):
            return value.hex().lower()
        if isinstance(value, dict):
            return {key: canonical(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [canonical(item) for item in value]
        return value

    expected = hashlib.sha256(
        json.dumps(
            canonical(expected_payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert (
        expected == "2123f69cfb6571328e681a8ccd9b3f465a5ab70d19552ba5109712dcc5996e4a"
    )
    assert loaded.state.scientific_fingerprint == expected


def test_public_model_direct_construction_cannot_bypass_scientific_coherence(
    tmp_path: Path,
) -> None:
    written = write_scalar_efield_beamfits(tmp_path)
    from radiosim.core.beam.fits import _load_fits_handler

    loaded = _load_fits_handler(
        _definition(written.path),
        observation_frequencies_hz=(100e6,),
        precision=PrecisionConfig.standard(),
        handler_ordinal=0,
    )
    with pytest.raises(ValueError, match="beam_type"):
        dataclasses.replace(loaded.state.file, beam_type="power")
    with pytest.raises(ValueError, match="feed_array"):
        dataclasses.replace(loaded.state.file, feed_array=("y", "x"))
    with pytest.raises(ValueError, match="data_shape"):
        dataclasses.replace(loaded.state.file, data_shape=(2, 2, 4, 5))
    with pytest.raises(ValueError, match="handler_id"):
        dataclasses.replace(loaded.state, handler_id="beam-0000-deadbeefdead")
    with pytest.raises(ValueError, match="strictly increasing"):
        dataclasses.replace(
            loaded.state,
            voltage_feature_scale_by_frequency=((110e6, 0.1), (100e6, 0.1)),
        )


def test_fresh_imports_are_lazy_and_do_not_initialize_backends() -> None:
    script = """
import sys
import radiosim.core.beam
import radiosim.core.beam.models
import radiosim.core.beam.fits
import radiosim.core.beam.runtime
for forbidden in ('pyuvdata', 'jax', 'numba'):
    assert forbidden not in sys.modules, (forbidden, sorted(sys.modules))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_fits_modules_do_not_publish_private_runtime_symbols() -> None:
    import radiosim.core as core
    import radiosim.core.beam as beam_package
    from radiosim.core.beam import fits, runtime

    assert fits.__all__ == []
    assert runtime.__all__ == []
    assert "_load_fits_handler" not in sys.modules["radiosim.core.beam"].__dict__
    assert beam_package.BeamFileProvenance is BeamFileProvenance
    assert beam_package.LoadedBeamHandlerState is LoadedBeamHandlerState
    assert "BeamFileProvenance" in beam_package.__all__
    assert "LoadedBeamHandlerState" in beam_package.__all__
    assert not hasattr(core, "BeamFileProvenance")
    assert not hasattr(core, "LoadedBeamHandlerState")


@pytest.mark.parametrize(
    "variant,error_name",
    (
        (UnsupportedBeamVariant.POWER, "UnsupportedBeamTypeError"),
        (UnsupportedBeamVariant.CIRCULAR_FEEDS, "UnsupportedBeamFeedError"),
        (UnsupportedBeamVariant.NONIDENTITY_BASIS, "UnsupportedBeamBasisError"),
    ),
)
def test_unsupported_metadata_uses_required_typed_errors(
    tmp_path: Path,
    variant: UnsupportedBeamVariant,
    error_name: str,
) -> None:
    from radiosim.core.beam import errors
    from radiosim.core.beam.fits import _load_fits_handler

    fixture = build_beam_variant(variant)

    class Loader:
        def read(self, path: Path):
            return fixture.beam

    error_type = getattr(errors, error_name)
    source = tmp_path / f"{variant.value}.beamfits"
    source.write_bytes(b"fixture transport")
    with pytest.raises(error_type):
        _load_fits_handler(
            _definition(source),
            observation_frequencies_hz=(100e6,),
            precision=PrecisionConfig.standard(),
            loader=Loader(),
            handler_ordinal=0,
        )


@pytest.mark.parametrize(
    "name,attribute,value,error_type",
    (
        ("scalar-feed", "feed_array", 1, UnsupportedBeamFeedError),
        (
            "string-feed-angle",
            "feed_angle",
            np.array(["bad", "bad"]),
            UnsupportedBeamFeedError,
        ),
        (
            "string-basis",
            "basis_vector_array",
            np.full((2, 2, 5, 8), "bad"),
            UnsupportedBeamBasisError,
        ),
        (
            "string-frequency",
            "freq_array",
            np.array(["bad"] * 4),
            BeamFrequencyDomainError,
        ),
        (
            "string-azimuth",
            "axis1_array",
            np.array(["bad"] * 8),
            UnsupportedBeamCoordinateError,
        ),
        (
            "string-bandpass",
            "bandpass_array",
            np.array(["bad"] * 4),
            BeamNormalizationError,
        ),
        (
            "array-beam-type",
            "beam_type",
            np.array(["efield", "power"]),
            UnsupportedBeamTypeError,
        ),
    ),
)
def test_hostile_dependency_metadata_is_typed_and_chained(
    tmp_path: Path,
    name: str,
    attribute: str,
    value: object,
    error_type: type[Exception],
) -> None:
    source = tmp_path / f"{name}.beamfits"
    source.write_bytes(name.encode())
    beam = build_scalar_efield_uvbeam()
    setattr(beam, attribute, value)
    beam.check = lambda **kwargs: True
    from radiosim.core.beam.fits import _load_fits_handler

    with pytest.raises(error_type) as caught:
        _load_fits_handler(
            _definition(source),
            observation_frequencies_hz=(100e6,),
            precision=PrecisionConfig.standard(),
            loader=_MemoryLoader(beam),
            handler_ordinal=0,
        )
    assert caught.value.__cause__ is not None


@pytest.mark.parametrize(
    "stage,error",
    (
        ("read", OverflowError("hostile dependency read")),
        ("check", IndexError("hostile dependency check")),
    ),
)
def test_unexpected_dependency_failures_are_typed_and_chained(
    tmp_path: Path,
    stage: str,
    error: Exception,
) -> None:
    source = tmp_path / f"{stage}-failure.beamfits"
    source.write_bytes(stage.encode())
    beam = build_scalar_efield_uvbeam()

    class Loader:
        def read(self, path: Path):
            if stage == "read":
                raise error
            beam.check = lambda **kwargs: (_ for _ in ()).throw(error)
            return beam

    from radiosim.core.beam.fits import _load_fits_handler

    with pytest.raises(BeamFileReadError) as caught:
        _load_fits_handler(
            _definition(source),
            observation_frequencies_hz=(100e6,),
            precision=PrecisionConfig.standard(),
            loader=Loader(),
            handler_ordinal=0,
        )
    assert caught.value.__cause__ is error


def test_cleanup_failure_preserves_primary_typed_error_and_dependency_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "primary-and-cleanup-failure.beamfits"
    source.write_bytes(b"primary and cleanup failure")
    from radiosim.core.beam import fits

    dependency_error = OverflowError("primary dependency failure")

    class Loader:
        def read(self, path: Path):
            raise dependency_error

    class FailingCleanupTemporaryDirectory:
        def __init__(self, *, prefix: str) -> None:
            self._real = temporary_directory(prefix=prefix)
            self.name = self._real.name

        def cleanup(self) -> None:
            self._real.cleanup()
            raise OSError("secondary cleanup failure")

    temporary_directory = fits.tempfile.TemporaryDirectory
    monkeypatch.setattr(
        fits.tempfile,
        "TemporaryDirectory",
        FailingCleanupTemporaryDirectory,
    )
    with pytest.raises(BeamFileReadError, match="could not read") as caught:
        fits._load_fits_handler(
            _definition(source),
            observation_frequencies_hz=(100e6,),
            precision=PrecisionConfig.standard(),
            loader=Loader(),
            handler_ordinal=0,
        )
    assert caught.value.__cause__ is dependency_error
    assert any(
        "cleanup failed" in note for note in getattr(caught.value, "__notes__", ())
    )
