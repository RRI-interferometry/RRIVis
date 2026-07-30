"""Canonical standard-visibility model and projection contracts."""

from __future__ import annotations

import copy
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import radiosim.io.standard_visibility as standard_visibility_module
from radiosim.api.simulator import Simulator
from radiosim.backends import get_backend
from radiosim.core.phase_center import PhaseCenter
from radiosim.core.result import (
    BackendResultProvenance,
    ResultPerformance,
    SolverResultProvenance,
    build_simulation_result,
)
from radiosim.io.result_errors import (
    FormatRepresentationError,
    UnsafeResultInputError,
    UnsupportedPolarizationBasisError,
)
from radiosim.io.standard_visibility import (
    PROJECTED_PHASE_SCHEMA,
    PROJECTION_HISTORY_PREFIX,
    PROJECTION_TRANSFORMATION,
    ProjectedPhaseCenter,
    StandardReadLimits,
    StandardVisibilityData,
    basis_for_file_codes,
    build_standard_visibility_data,
    normalize_autocorrelations,
    project_simulation_result,
    projection_record_from_history,
    require_feed_polarization_coupling,
    require_polarization_basis,
)


def _result_mapping(
    tmp_path: Path,
    *,
    frequencies_hz: tuple[float, ...],
    channel_widths_hz: tuple[float, ...],
    receptors: dict[str, object] | None = None,
) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    layout = tmp_path / "standard-antennas.txt"
    layout.write_text(
        "Name Number BeamID E N U Diameter\n"
        "A3 3 0 0 0 0 12\n"
        "A7 7 0 17 -4 2 15\n"
        "A12 12 0 -8 29 5 19\n",
        encoding="utf-8",
    )
    mapping: dict[str, object] = {
        "instrument": {
            "source": {
                "kind": "layout_file",
                "path": str(layout),
                "format": "radiosim",
                "telescope_name": "Tier4E Array",
            },
            "location": {
                "longitude_deg": 21.4,
                "latitude_deg": -30.7,
                "height_m": 1000.0,
            },
        },
        "baseline_selection": {"correlations": "all"},
        "beams": {
            "mode": "analytic",
            "model": {
                "kind": "circular_aperture",
                "taper": {"kind": "gaussian", "edge_taper_db": 10.0},
            },
        },
        "obs_time": {
            "start_time": "2025-01-01T00:00:00",
            "duration_seconds": 2.0,
            "time_step_seconds": 1.0,
        },
        "obs_frequency": {
            "mode": "explicit",
            "channel_frequencies_hz": list(frequencies_hz),
            "channel_widths_hz": list(channel_widths_hz),
        },
        "sky_model": {
            "sources": [{"kind": "test_sources", "num_sources": 1, "seed": 11}]
        },
        "execution": {"backend": "numpy", "offline": True},
    }
    if receptors is not None:
        mapping["receptors"] = receptors
    return mapping


def build_standard_result(
    tmp_path: Path,
    *,
    dtype: str = "complex128",
    frequencies_hz: tuple[float, ...] = (100e6, 101.5e6),
    channel_widths_hz: tuple[float, ...] = (1.5e6, 1.5e6),
    receptors: dict[str, object] | None = None,
):
    """Build a nontrivial canonical result with autos and crosses."""
    simulator = Simulator.from_mapping(
        _result_mapping(
            tmp_path,
            frequencies_hz=frequencies_hz,
            channel_widths_hz=channel_widths_hz,
            receptors=receptors,
        ),
        base_dir=tmp_path,
    )
    simulator._ensure_instrument_state()
    simulator._ensure_receptor_set()
    simulator._ensure_beam_system()
    backend = get_backend("numpy")
    shape = (
        len(simulator.config.observation.time_grid),
        len(simulator._instrument_state.selection.baselines),
        len(frequencies_hz),
        2,
        2,
    )
    real = np.arange(1, np.prod(shape) + 1, dtype=np.float64).reshape(shape)
    receptor = (real + 1j * (real * 0.125 + 0.03125)).astype(dtype)
    for baseline_index, baseline in enumerate(
        simulator._instrument_state.selection.baselines
    ):
        if baseline.ant1 == baseline.ant2:
            receptor[:, baseline_index, :, 0, 0] = receptor[
                :, baseline_index, :, 0, 0
            ].real
            receptor[:, baseline_index, :, 1, 1] = receptor[
                :, baseline_index, :, 1, 1
            ].real
            receptor[:, baseline_index, :, 1, 0] = np.conj(
                receptor[:, baseline_index, :, 0, 1]
            )
    return build_simulation_result(
        receptor_visibilities=receptor,
        backend=backend,
        time_grid=simulator.config.observation.time_grid,
        frequencies_hz=frequencies_hz,
        channel_widths_hz=channel_widths_hz,
        instrument=simulator.instrument,
        selection=simulator._instrument_state.selection,
        beam_state=simulator.beam_state,
        receptors=simulator.receptors,
        phase_center=PhaseCenter(),
        backend_provenance=BackendResultProvenance(
            requested_backend="numpy",
            actual_backend="numpy",
            requested_precision={"output": dtype},
            actual_precision={"output": dtype},
            result_dtype=dtype,
        ),
        solver_provenance=SolverResultProvenance(
            solver="rime",
            sky_representation="point_sources",
            convention="radiosim.rime-zenith-drift.v1",
            execution_path="polarized",
            components=("point",),
            component_element_counts=(3,),
        ),
        resolved_config=simulator.config.to_json_safe(),
        configuration_provenance=None,
        performance=ResultPerformance(
            setup_seconds=0.0,
            solver_seconds=0.0,
            solver_point_seconds=0.0,
            solver_healpix_seconds=0.0,
            result_construction_seconds=0.0,
            host_transfer_seconds=0.0,
            total_seconds=0.0,
        ),
        history=("Tier 4E standard fixture",),
    )


def _phase() -> ProjectedPhaseCenter:
    return ProjectedPhaseCenter(
        longitude_rad=1.25,
        latitude_rad=-0.4,
        reference_utc_jd1=2460676.0,
        reference_utc_jd2=0.5,
        original_phase_snapshot=dict(PhaseCenter().to_snapshot()),
        transformation="astropy-zenith-icrs+pyuvdata-phase_to_time.v1",
    )


def _standard_data(**overrides: object) -> StandardVisibilityData:
    shape = (2, 2, 2, 4)
    visibilities = (
        np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
        + 1j * np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 7.0
    ).astype(np.complex128)
    values: dict[str, object] = {
        "format": "ms",
        "visibilities": visibilities,
        "flags": np.zeros(shape, dtype=np.bool_),
        "weights": np.full(shape, 2.5, dtype=np.float32),
        "utc_jd1": np.array([2460676.0, 2460676.0]),
        "utc_jd2": np.array([0.5, 0.50001]),
        "exposure_seconds": np.array([2.0, 2.0]),
        "frequencies_hz": np.array([100e6, 101.5e6]),
        "channel_widths_hz": np.array([1.25e6, 2.0e6]),
        "correlations": ("XX", "XY", "YX", "YY"),
        "antenna1_numbers": np.array([3, 3], dtype=np.int64),
        "antenna2_numbers": np.array([3, 7], dtype=np.int64),
        "uvw_m": np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
                [[0.0, 0.0, 0.0], [4.0, 5.0, 6.0]],
            ],
            dtype=np.float64,
        ),
        "telescope_snapshot": {
            "name": "Tier4E Array",
            "instrument": "Tier4E Array",
            "location_itrs_xyz_m": [1.0, 2.0, 3.0],
            "antennas": [
                {
                    "number": 3,
                    "name": "A3",
                    "position_enu_m": [0.0, 0.0, 0.0],
                    "diameter_m": 12.0,
                },
                {
                    "number": 7,
                    "name": "A7",
                    "position_enu_m": [17.0, -4.0, 2.0],
                    "diameter_m": 15.0,
                },
            ],
        },
        "phase_center": _phase(),
        "history": ("projected",),
        "source_scientific_sha256": "a" * 64,
        "source_provenance_sha256": "b" * 64,
    }
    values.update(overrides)
    return build_standard_visibility_data(**values)


def test_standard_read_limit_defaults_and_exact_integer_validation() -> None:
    limits = StandardReadLimits()
    assert limits == StandardReadLimits(
        max_times=10_000_000,
        max_baselines=10_000_000,
        max_frequencies=1_000_000,
        max_antennas=1_000_000,
        max_visibility_elements=100_000_000,
        max_data_bytes=2_147_483_648,
    )
    for value in (True, np.int64(1), 0, -1):
        with pytest.raises((TypeError, ValueError)):
            StandardReadLimits(max_times=value)


def test_projected_phase_center_is_exact_immutable_and_json_safe() -> None:
    phase = _phase()
    assert phase.schema_version == "radiosim.projected-phase-center.v1"
    assert phase.kind == "sidereal"
    assert phase.frame == "icrs"
    assert dict(phase.original_phase_snapshot) == dict(PhaseCenter().to_snapshot())
    assert phase == _phase()
    with pytest.raises((AttributeError, TypeError)):
        phase.longitude_rad = 0.0
    for field, value in (
        ("longitude_rad", True),
        ("longitude_rad", 2 * math.pi),
        ("latitude_rad", math.pi),
        ("reference_utc_jd1", np.float64(1.0)),
    ):
        kwargs = {
            "longitude_rad": 1.0,
            "latitude_rad": 0.5,
            "reference_utc_jd1": 2460676.0,
            "reference_utc_jd2": 0.5,
            "original_phase_snapshot": dict(PhaseCenter().to_snapshot()),
            "transformation": "projection",
        }
        kwargs[field] = value
        with pytest.raises((TypeError, ValueError)):
            ProjectedPhaseCenter(**kwargs)


def test_standard_visibility_owns_bytes_backed_arrays_and_nested_state() -> None:
    visibilities = np.ones((2, 2, 2, 4), dtype=np.complex128)
    snapshot = {
        "name": "Array",
        "instrument": "Array",
        "location_itrs_xyz_m": [1.0, 2.0, 3.0],
        "antennas": [
            {
                "number": 3,
                "name": "A3",
                "position_enu_m": [0.0, 0.0, 0.0],
                "diameter_m": 12.0,
            },
            {
                "number": 7,
                "name": "A7",
                "position_enu_m": [17.0, -4.0, 2.0],
                "diameter_m": 15.0,
            },
        ],
    }
    standard = _standard_data(
        visibilities=visibilities,
        telescope_snapshot=snapshot,
    )
    expected_snapshot = copy.deepcopy(snapshot)
    visibilities[...] = 99
    snapshot["name"] = "mutated"
    assert not np.any(standard.visibilities == 99)
    assert standard.telescope_snapshot["name"] == "Array"
    for array in (
        standard.visibilities,
        standard.flags,
        standard.weights,
        standard.utc_jd1,
        standard.utc_jd2,
        standard.exposure_seconds,
        standard.frequencies_hz,
        standard.channel_widths_hz,
        standard.antenna1_numbers,
        standard.antenna2_numbers,
        standard.uvw_m,
    ):
        assert type(array) is np.ndarray
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flags.writeable = True
    assert standard == _standard_data(
        visibilities=np.ones_like(visibilities),
        telescope_snapshot=expected_snapshot,
    )
    with pytest.raises(TypeError):
        hash(standard)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("format", "hdf5", "format"),
        # Correlation-label rejection is typed as
        # UnsupportedPolarizationBasisError from Tier 5F and is asserted by
        # test_standard_visibility_rejects_reordered_and_mixed_correlations.
        ("visibilities", np.zeros((2, 2, 2, 3), dtype=np.complex64), "shape"),
        ("weights", np.ones((2, 2, 2, 4), dtype=np.float64), "float32"),
        ("exposure_seconds", np.array([2.0, 0.0]), "positive"),
        ("frequencies_hz", np.array([100e6, np.inf]), "finite"),
        ("antenna2_numbers", np.array([3, 3], dtype=np.int64), "baseline"),
    ],
)
def test_standard_visibility_rejects_invalid_axes(
    field: str,
    value: object,
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        _standard_data(**{field: value})


def test_standard_visibility_rejects_subclass_and_memmap_aliases(
    tmp_path: Path,
) -> None:
    class ArraySubclass(np.ndarray):
        pass

    subclass = np.ones((2, 2, 2, 4), dtype=np.complex128).view(ArraySubclass)
    path = tmp_path / "mapped.dat"
    mapped = np.memmap(path, mode="w+", shape=(2, 2, 2, 4), dtype=np.complex128)
    for value in (subclass, mapped):
        standard = _standard_data(visibilities=value)
        assert type(standard.visibilities) is np.ndarray
        assert not isinstance(standard.visibilities, np.memmap)
        assert not np.shares_memory(standard.visibilities, value)


def test_standard_public_factory_and_projection_signatures_are_stable() -> None:
    assert tuple(inspect.signature(build_standard_visibility_data).parameters) == (
        "format",
        "visibilities",
        "flags",
        "weights",
        "utc_jd1",
        "utc_jd2",
        "exposure_seconds",
        "frequencies_hz",
        "channel_widths_hz",
        "correlations",
        "antenna1_numbers",
        "antenna2_numbers",
        "uvw_m",
        "telescope_snapshot",
        "phase_center",
        "history",
        "source_scientific_sha256",
        "source_provenance_sha256",
    )
    assert tuple(inspect.signature(project_simulation_result).parameters) == (
        "result",
        "format",
    )


@pytest.mark.parametrize("format_name", ["ms", "uvfits"])
def test_shared_projection_has_canonical_blt_polarization_and_phase(
    tmp_path: Path,
    format_name: str,
) -> None:
    result = build_standard_result(tmp_path, dtype="complex64")
    before = result.visibilities.tobytes()

    projected = project_simulation_result(result, format=format_name)
    uvdata = projected.uvdata

    baseline_count = len(result.selection.baselines)
    np.testing.assert_array_equal(
        uvdata.ant_1_array,
        np.tile(
            [baseline.ant1.number for baseline in result.selection.baselines],
            len(result.time_grid),
        ),
    )
    np.testing.assert_array_equal(
        uvdata.ant_2_array,
        np.tile(
            [baseline.ant2.number for baseline in result.selection.baselines],
            len(result.time_grid),
        ),
    )
    np.testing.assert_allclose(
        uvdata.time_array,
        np.repeat(result.time_grid.to_jd(), baseline_count),
        rtol=0.0,
        atol=5e-10,
    )
    assert np.asarray(uvdata.polarization_array).tolist() == [-5, -7, -8, -6]
    assert len(uvdata.phase_center_catalog) == 1
    catalog = next(iter(uvdata.phase_center_catalog.values()))
    assert catalog["cat_type"] == "sidereal"
    assert catalog["cat_frame"] == "icrs"
    assert projected.data.phase_center.longitude_rad == pytest.approx(
        catalog["cat_lon"],
        abs=1e-12,
    )
    assert projected.data.phase_center.latitude_rad == pytest.approx(
        catalog["cat_lat"],
        abs=1e-12,
    )
    assert result.visibilities.tobytes() == before


def test_shared_projection_rejects_large_parallel_auto_imaginary_part(
    tmp_path: Path,
) -> None:
    result = build_standard_result(tmp_path)
    forged = np.array(result.visibilities, copy=True)
    auto_index = next(
        index
        for index, baseline in enumerate(result.selection.baselines)
        if baseline.ant1 == baseline.ant2
    )
    forged[:, auto_index, :, 0] += 1j
    object.__setattr__(
        result,
        "visibilities",
        np.ndarray(forged.shape, dtype=forged.dtype, buffer=forged.tobytes()),
    )

    with pytest.raises(FormatRepresentationError, match="autocorrelation"):
        project_simulation_result(result, format="ms")


# ---------------------------------------------------------------------------
# Tier 5F: the standard-format writer is basis aware
# ---------------------------------------------------------------------------

CIRCULAR_RECEPTORS: dict[str, object] = {"default": {"basis": "circular"}}


def test_standard_visibility_consumes_the_shared_correlation_table() -> None:
    """Section 20.1: the fourth duplicated correlation site is gone (defect D4).

    OWNED BY: Tier 5F.
    """
    source = inspect.getsource(standard_visibility_module)
    assert "radiosim.core.polarization_basis" in source
    assert '("XX", "XY", "YX", "YY")' not in source
    assert '("RR", "RL", "LR", "LL")' not in source
    for name in ("CANONICAL_CORRELATIONS", "CANONICAL_CODES", "FILE_CODES"):
        assert not hasattr(standard_visibility_module, name)


@pytest.mark.parametrize(
    ("labels", "basis"),
    [
        (("XX", "XY", "YX", "YY"), "linear_xy"),
        (("RR", "RL", "LR", "LL"), "circular_rl"),
    ],
    ids=["linear", "circular"],
)
def test_standard_visibility_accepts_both_accepted_correlation_sets(
    labels: tuple[str, ...],
    basis: str,
) -> None:
    standard = _standard_data(correlations=labels)
    assert standard.correlations == labels
    assert require_polarization_basis(labels) == basis


@pytest.mark.parametrize(
    "labels",
    [
        ("XX", "YY", "XY", "YX"),
        ("RR", "LL", "RL", "LR"),
        ("XX", "XY", "LR", "LL"),
        ("XX", "XY", "YX"),
        "XXXYYXYY",
        None,
    ],
)
def test_standard_visibility_rejects_reordered_and_mixed_correlations(
    labels: object,
) -> None:
    with pytest.raises(
        UnsupportedPolarizationBasisError,
        match="XX,XY,YX,YY or RR,RL,LR,LL",
    ):
        _ = require_polarization_basis(labels)
    with pytest.raises(UnsupportedPolarizationBasisError):
        _standard_data(correlations=labels)


def test_nominal_feed_angles_match_the_resolved_receptor_convention(
    tmp_path: Path,
) -> None:
    """Section 14.4: the written feed angles are the zero-rotation nominal pair."""
    for receptors, basis in (
        (None, "linear_xy"),
        (CIRCULAR_RECEPTORS, "circular_rl"),
    ):
        simulator = Simulator.from_mapping(
            _result_mapping(
                tmp_path / basis,
                frequencies_hz=(100e6, 101.5e6),
                channel_widths_hz=(1.5e6, 1.5e6),
                receptors=receptors,
            ),
            base_dir=tmp_path / basis,
        )
        simulator._ensure_instrument_state()
        simulator._ensure_receptor_set()
        resolved = simulator.receptors
        assert resolved.output_basis == basis
        for receptor in resolved.receptor_by_antenna.values():
            assert receptor.feed_rotation_rad == 0.0
            assert (
                receptor.feed_angle_rad
                == standard_visibility_module._NOMINAL_FEED_ANGLES_RAD[basis]
            )


@pytest.mark.parametrize("format_name", ["ms", "uvfits"])
@pytest.mark.parametrize(
    ("receptors", "basis", "labels", "feeds", "codes", "angles"),
    [
        (
            None,
            "linear_xy",
            ("XX", "XY", "YX", "YY"),
            ["x", "y"],
            [-5, -7, -8, -6],
            [math.pi / 2.0, 0.0],
        ),
        (
            CIRCULAR_RECEPTORS,
            "circular_rl",
            ("RR", "RL", "LR", "LL"),
            ["r", "l"],
            [-1, -3, -4, -2],
            [0.0, 0.0],
        ),
    ],
    ids=["linear", "circular"],
)
def test_projection_writes_the_resolved_basis_into_pyuvdata(
    tmp_path: Path,
    format_name: str,
    receptors: dict[str, object] | None,
    basis: str,
    labels: tuple[str, ...],
    feeds: list[str],
    codes: list[int],
    angles: list[float],
) -> None:
    """Section 22.1: feed_array, feed_angle, and the code order follow the basis."""
    result = build_standard_result(tmp_path, receptors=receptors)
    assert result.polarization_basis == basis
    assert result.correlations == labels

    projected = project_simulation_result(result, format=format_name)
    telescope = projected.uvdata.telescope
    antenna_count = len(result.instrument.antennas)

    assert telescope.feed_array.tolist() == [feeds] * antenna_count
    np.testing.assert_allclose(
        np.asarray(telescope.feed_angle),
        np.tile(angles, (antenna_count, 1)),
        rtol=0.0,
        atol=0.0,
    )
    assert telescope.Nfeeds == 2
    assert list(telescope.mount_type) == ["fixed"] * antenna_count
    assert np.asarray(projected.uvdata.polarization_array).tolist() == codes
    assert projected.data.correlations == labels
    assert projected.uvdata.check() is True


def test_projection_history_records_the_basis_and_receptor_fingerprint(
    tmp_path: Path,
) -> None:
    """Section 22.3: the projection record gains two short scalars."""
    result = build_standard_result(tmp_path, receptors=CIRCULAR_RECEPTORS)
    projected = project_simulation_result(result, format="uvfits")
    record, _lines = projection_record_from_history(projected.uvdata.history)
    assert record["polarization_basis"] == "circular_rl"
    assert record["receptor_sha256"] == result.receptors.provenance.receptor_sha256
    assert (
        len(projected.uvdata.history.encode("utf-8"))
        <= standard_visibility_module._PROJECTION_HISTORY_LIMIT
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("polarization_basis", "elliptical"),
        ("polarization_basis", 5),
        ("receptor_sha256", "C" * 64),
        ("receptor_sha256", "abc"),
    ],
)
def test_projection_history_rejects_invalid_basis_or_receptor_fingerprint(
    field: str,
    value: object,
) -> None:
    record = _projection_json_record()
    record[field] = value
    with pytest.raises(UnsafeResultInputError):
        _ = projection_record_from_history(_projection_history(record))


@pytest.mark.parametrize(
    "field",
    ["polarization_basis", "receptor_sha256"],
)
def test_projection_history_requires_the_new_record_fields(field: str) -> None:
    record = _projection_json_record()
    del record[field]
    with pytest.raises(UnsafeResultInputError, match="unexpected record fields"):
        _ = projection_record_from_history(_projection_history(record))


@pytest.mark.parametrize(
    ("codes", "basis"),
    [
        ([-5, -7, -8, -6], "linear_xy"),
        ([-5, -6, -7, -8], "linear_xy"),
        ([-1, -3, -4, -2], "circular_rl"),
        ([-1, -2, -3, -4], "circular_rl"),
    ],
)
def test_file_code_sets_identify_both_bases_in_either_order(
    codes: list[int],
    basis: str,
) -> None:
    """Section 14.2 as corrected: MS keeps memory order, UVFITS descends."""
    assert basis_for_file_codes(np.asarray(codes, dtype=np.int64)) == basis


@pytest.mark.parametrize(
    "codes",
    [
        [-5, -6, -7, -1],
        [-5, -6, -7],
        [1, 2, 3, 4],
        [-5, -5, -6, -7],
    ],
)
def test_file_code_sets_reject_mixed_and_short_axes(codes: list[int]) -> None:
    with pytest.raises(
        FormatRepresentationError,
        match="unsupported polarization layout",
    ):
        _ = basis_for_file_codes(np.asarray(codes, dtype=np.int64))


def test_read_path_rejects_feeds_that_disagree_with_the_polarization_axis() -> None:
    """Tier 5A Q3: pyuvdata does not couple feeds to polarizations; RadioSim does."""
    mismatched = SimpleNamespace(
        telescope=SimpleNamespace(feed_array=np.array([["r", "l"], ["r", "l"]])),
    )
    with pytest.raises(FormatRepresentationError, match="disagree with its"):
        require_feed_polarization_coupling(mismatched, "linear_xy")

    matching = SimpleNamespace(
        telescope=SimpleNamespace(feed_array=np.array([["r", "l"], ["r", "l"]])),
    )
    require_feed_polarization_coupling(matching, "circular_rl")

    # An input carrying no feed metadata is left to the format-specific reader.
    require_feed_polarization_coupling(SimpleNamespace(telescope=None), "linear_xy")


@pytest.mark.parametrize(
    ("receptors", "labels"),
    [
        (None, ("XX", "XY", "YX", "YY")),
        (CIRCULAR_RECEPTORS, ("RR", "RL", "LR", "LL")),
    ],
    ids=["linear", "circular"],
)
def test_autocorrelation_normalization_acts_on_the_parallel_hands_of_each_basis(
    tmp_path: Path,
    receptors: dict[str, object] | None,
    labels: tuple[str, ...],
) -> None:
    """Section 22.1: the (0, 3) literal is derived, and the cross hands are kept."""
    result = build_standard_result(tmp_path, receptors=receptors)
    assert result.correlations == labels
    forged = np.array(result.visibilities, copy=True)
    auto_index = next(
        index
        for index, baseline in enumerate(result.selection.baselines)
        if baseline.ant1 == baseline.ant2
    )
    forged[:, auto_index, :, 0] = 3.0 + 0.0j
    forged[:, auto_index, :, 3] = 5.0 + 0.0j
    forged[:, auto_index, :, 1] = 1.0 + 2.0j
    forged[:, auto_index, :, 2] = 1.0 - 2.0j
    object.__setattr__(
        result,
        "visibilities",
        np.ndarray(forged.shape, dtype=forged.dtype, buffer=forged.tobytes()),
    )

    data, normalized = normalize_autocorrelations(result)
    assert normalized == 0
    np.testing.assert_array_equal(data[:, auto_index, :, 1], 1.0 + 2.0j)
    np.testing.assert_array_equal(data[:, auto_index, :, 2], 1.0 - 2.0j)

    forged = np.array(data, copy=True)
    forged[:, auto_index, :, 3] += 1j
    object.__setattr__(
        result,
        "visibilities",
        np.ndarray(forged.shape, dtype=forged.dtype, buffer=forged.tobytes()),
    )
    with pytest.raises(FormatRepresentationError, match="parallel-hand"):
        _ = normalize_autocorrelations(result)


def _projection_json_record() -> dict[str, object]:
    return {
        "schema": PROJECTED_PHASE_SCHEMA,
        "projected_phase": {
            "schema_version": PROJECTED_PHASE_SCHEMA,
            "kind": "sidereal",
            "frame": "icrs",
            "longitude_rad": 1.25,
            "latitude_rad": -0.4,
            "reference_utc_jd1": 2460676.0,
            "reference_utc_jd2": 0.5,
            "original_phase_snapshot": dict(PhaseCenter().to_snapshot()),
            "transformation": PROJECTION_TRANSFORMATION,
        },
        "source_scientific_sha256": "a" * 64,
        "source_provenance_sha256": "b" * 64,
        "input_visibility_dtype": "complex128",
        "stored_visibility_dtype": "complex64",
        "input_weight_dtype": "float64",
        "stored_weight_dtype": "float32",
        "polarization_basis": "linear_xy",
        "receptor_sha256": "c" * 64,
        "instrument": {"name": "array"},
        "beam": {"kind": "analytic"},
        "solver": {"solver": "rime"},
    }


def _projection_history(record: dict[str, object]) -> str:
    return PROJECTION_HISTORY_PREFIX + json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
    )


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_projection_history_rejects_duplicate_keys_and_nonfinite_constants(
    constant: str,
) -> None:
    valid = _projection_history(_projection_json_record())
    decoded, _lines = projection_record_from_history(valid)
    assert decoded["schema"] == PROJECTED_PHASE_SCHEMA

    duplicate = valid.replace(
        "{",
        '{"schema":"radiosim.projected-phase-center.v1",',
        1,
    )
    with pytest.raises(UnsafeResultInputError, match="duplicate"):
        projection_record_from_history(duplicate)

    nonfinite = valid.replace(
        '"longitude_rad":1.25',
        f'"longitude_rad":{constant}',
    )
    with pytest.raises(UnsafeResultInputError, match="non-finite|constant"):
        projection_record_from_history(nonfinite)


@pytest.mark.parametrize(
    ("history", "message"),
    [
        ("\ud800", "UTF-8"),
        (PROJECTION_HISTORY_PREFIX + "{}\x00", "NUL"),
        ("ordinary history", "exactly one"),
        (
            PROJECTION_HISTORY_PREFIX + "{}\n" + PROJECTION_HISTORY_PREFIX + "{}",
            "exactly one",
        ),
        (PROJECTION_HISTORY_PREFIX + "{}trailing", "trailing"),
        (PROJECTION_HISTORY_PREFIX + "[]", "JSON object"),
    ],
)
def test_projection_history_rejects_unsafe_text_and_structure(
    history: str,
    message: str,
) -> None:
    with pytest.raises(
        (UnsafeResultInputError, FormatRepresentationError),
        match=message,
    ):
        projection_record_from_history(history)


def test_projection_history_rejects_oversized_utf8_before_json_decode() -> None:
    history = PROJECTION_HISTORY_PREFIX + '{"value":"' + ("é" * 8_000) + '"}'
    with pytest.raises(UnsafeResultInputError, match="16000 UTF-8 bytes"):
        projection_record_from_history(history)


def test_projection_history_rejects_spoofed_pyuvdata_trailing_marker() -> None:
    history = (
        _projection_history(_projection_json_record())
        + " Read/written with pyuvdata version: 3.2.1.attacker"
    )
    with pytest.raises(UnsafeResultInputError, match="trailing"):
        projection_record_from_history(history)


def test_projection_history_enforces_exact_depth_boundary() -> None:
    at_limit = _projection_json_record()
    nested: object = "leaf"
    for _ in range(62):
        nested = {"child": nested}
    at_limit["instrument"] = nested
    projection_record_from_history(_projection_history(at_limit))

    over_limit = _projection_json_record()
    nested = "leaf"
    for _ in range(63):
        nested = {"child": nested}
    over_limit["instrument"] = nested
    with pytest.raises(UnsafeResultInputError, match="nesting"):
        projection_record_from_history(_projection_history(over_limit))
