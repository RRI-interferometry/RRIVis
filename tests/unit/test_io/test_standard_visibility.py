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
    sky_representation: str = "point_sources",
    components: tuple[str, ...] = ("point",),
    component_element_counts: tuple[int, ...] = (3,),
    receptor_matrix: np.ndarray | None = None,
):
    """Build a nontrivial canonical result with autos and crosses.

    ``sky_representation`` flows to both the resolved configuration and the
    solver provenance, so a hybrid fixture is a coherent result rather than a
    point-only result wearing a hybrid label (Tier 6G, plan Section 19).
    """
    mapping = _result_mapping(
        tmp_path,
        frequencies_hz=frequencies_hz,
        channel_widths_hz=channel_widths_hz,
        receptors=receptors,
    )
    mapping["visibility"] = {"sky_representation": sky_representation}
    simulator = Simulator.from_mapping(mapping, base_dir=tmp_path)
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
    if receptor_matrix is None:
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
    else:
        matrix = np.asarray(receptor_matrix, dtype=dtype)
        if matrix.shape != (2, 2):
            raise ValueError("receptor_matrix must have shape (2, 2)")
        receptor = np.broadcast_to(matrix, shape).copy()
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
            sky_representation=sky_representation,
            convention="radiosim.rime-zenith-drift.v1",
            execution_path="polarized",
            components=components,
            component_element_counts=component_element_counts,
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


# ---------------------------------------------------------------------------
# Tier 6G: MS/UVFITS HISTORY names every solved component (Section 19; H10)
# ---------------------------------------------------------------------------


_HISTORY_COMPONENT_CASES = [
    ("point_sources", ("point",), (3,), "point", "3"),
    ("healpix_map", ("healpix",), (3072,), "healpix", "3072"),
    ("hybrid", ("point", "healpix"), (3, 3072), "point,healpix", "3,3072"),
]


@pytest.mark.parametrize("format_name", ["ms", "uvfits"])
@pytest.mark.parametrize(
    ("representation", "components", "counts", "component_text", "count_text"),
    _HISTORY_COMPONENT_CASES,
    ids=[case[0] for case in _HISTORY_COMPONENT_CASES],
)
def test_projection_history_names_every_solved_component(
    tmp_path: Path,
    format_name: str,
    representation: str,
    components: tuple[str, ...],
    counts: tuple[int, ...],
    component_text: str,
    count_text: str,
) -> None:
    """Section 19: a summed hybrid is not reconstructible without this line."""
    result = build_standard_result(
        tmp_path,
        sky_representation=representation,
        components=components,
        component_element_counts=counts,
    )
    projected = project_simulation_result(result, format=format_name)

    lines = projected.data.history
    assert f"sky_representation={representation}" in lines
    assert f"solver_components={component_text}" in lines
    assert f"solver_component_element_counts={count_text}" in lines

    record, _lines = projection_record_from_history(projected.uvdata.history)
    solver = record["solver"]
    assert solver["sky_representation"] == representation
    assert list(solver["components"]) == list(components)
    assert list(solver["component_element_counts"]) == list(counts)
    assert (
        len(projected.uvdata.history.encode("utf-8"))
        <= standard_visibility_module._PROJECTION_HISTORY_LIMIT
    )


# ==============================================================================
# SCI-005 Stage 3: the shared reader-projection contract for a full efield run
# ==============================================================================
#
# ``docs/development/sci005_beam_physics_plan.md`` Section 5.4's reader bullet:
# "UVFITS/MS reconstruct representable visibilities, feed metadata, correlation
# labels, and source scientific identity under their existing projection
# contracts."  Section 5.4's ten-row minimum stands "beside whatever
# ``reader_projection`` rows the readers' own contracts require", and this is
# the one seam both formats pass through.
#
# Section 7.5 makes ``io/standard_visibility.py`` unwritable at Stage 3: the
# projection already carries all four correlation products in the declared
# output basis, so this measures what it emits for a non-scalar ``E``.

_STAGE3_PROJECTION_BASES = {
    "linear": ("linear_xy", ("XX", "XY", "YX", "YY"), ("x", "y")),
    "circular": ("circular_rl", ("RR", "RL", "LR", "LL"), ("r", "l")),
}


@pytest.mark.parametrize("format_name", ["uvfits", "ms"])
@pytest.mark.parametrize("authored_basis", sorted(_STAGE3_PROJECTION_BASES))
def test_the_projection_carries_a_full_efield_run_in_the_declared_basis(
    tmp_path: Path,
    authored_basis: str,
    format_name: str,
) -> None:
    """One shared ``reader_projection`` witness for both export formats."""
    from tests.fixtures.beamfits import run_full_efield_workload

    basis, labels, feeds = _STAGE3_PROJECTION_BASES[authored_basis]
    workload = run_full_efield_workload(tmp_path, output_basis=authored_basis)
    result = workload.result
    projected = project_simulation_result(result, format=format_name)

    assert projected.data.format == format_name
    assert projected.data.correlations == labels
    assert projected.data.visibilities.shape[-1] == 4
    for index in (1, 2):
        assert float(np.max(np.abs(projected.data.visibilities[..., index]))) > 0.0

    observed = {
        str(feed).lower()
        for feed in np.asarray(projected.uvdata.telescope.feed_array).reshape(-1)
    }
    assert observed == set(feeds)
    require_feed_polarization_coupling(projected.uvdata, basis)

    record, _lines = projection_record_from_history(projected.uvdata.history)
    assert record["polarization_basis"] == basis
    assert record["source_scientific_sha256"] == result.scientific_sha256
    assert record["receptor_sha256"] == result.receptors.provenance.receptor_sha256


# ==============================================================================
# SCI-004 phase M3: Section 10's standard-format outputs for an m-mode result
# ==============================================================================
#
# ``docs/development/sci004_mmode_design.md`` Section 13.5 opens the phase-M3 red
# slice on the five output writers, and Section 12.2's ninth oracle family is
# "Results: in-memory, summary, HDF5, UVFITS, and MS round trips with phase,
# feed, correlation, time, solver, and fingerprint metadata".  Section 10 fixes
# what those paths must carry for the m-mode arm of the strict tagged solver
# union:
#
#   "In-memory, summary JSON, HDF5, UVFITS, and Measurement Set paths all write
#   the same synthesized UTC sample centres and integration widths. [...]
#   UVFITS/MS keep the canonical zenith phase centre, east-X/circular feed
#   metadata, four correlation products, UTC coordinates, and history lines
#   naming the m-mode/frame/harmonic conventions. Reader round trips must
#   reconstruct and authenticate the m-mode solver snapshot; a reader that
#   silently labels it ``rime`` fails acceptance."
#
# This module owns the shared m-mode fixture the phase-M3 output oracles use,
# because ``io/standard_visibility.py`` is the one seam UVFITS and MS both pass
# through.  ``build_mmode_result`` is deliberately *not* a solver run: it
# resolves a real full-sidereal m-mode configuration -- so the published UTC
# sample centres and integration widths are the genuine Section 3.1 ERA-derived
# ones, six distinct widths rather than one repeated cadence -- and pairs it
# with a synthetic receptor cube and a production ``MModeSolverSnapshot``.  What
# the output contract transports is metadata identity, not physics; the physics
# of the cube and of the frame certificate is accepted M1/M2 scope and is
# re-measured from real runs by the phase-M3 characterization families in
# ``tests/characterization/test_sci004_mmode.py``.

from radiosim.core.mmode.solver import (  # noqa: E402
    DirectGateRecord,
    MModeSolverSnapshot,
)
from radiosim.core.mmode.types import (  # noqa: E402
    MMODE_CONVENTION,
    MMODE_EXECUTION_POLICY,
    MMODE_FRAME_MODEL,
    MMODE_HARMONIC_CONVENTION,
    MMODE_QUADRATURE_POLICY,
    MMODE_STOKES_BRIDGE,
    MMODE_TANGENT_FRAME_M1,
    MMODE_TIME_GRID_CONVENTION,
    MMODE_TRUNCATION_POLICY,
)
from radiosim.core.result import (  # noqa: E402
    MMODE_SOLVER_SNAPSHOT_KEYS,
    MModeSolverResultProvenance,
)

#: The shared phase-M3 fixture's exact m-mode dimensions.  ``sidereal_samples``
#: is the smallest value Section 7.3's mandatory m-tail diagnostic admits for
#: ``mmax = 4``: ``mcheck = min(lcheck, mmax + max(8, max(1, mmax // 8))) = 12``
#: and the Nyquist rule needs ``2 * mcheck + 1``.
MMODE_FIXTURE_LMAX = 4
MMODE_FIXTURE_MMAX = 4
MMODE_FIXTURE_QUADRATURE_NSIDE = 2
MMODE_FIXTURE_SIDEREAL_SAMPLES = 25
MMODE_FIXTURE_WORKING_MEMORY_BYTES = 1 << 26

#: The exact retained bytes of the fixture's m-mode configuration override.  The
#: phase-M3 red record hashes these as each case's
#: ``invalid_config_raw_sha256``, so the record names the configuration the
#: observation was made from rather than a description of it.
MMODE_FIXTURE_BYTES = f"""\
obs_time:
  mode: full_sidereal
  start_time: "2025-01-01T00:00:00"
  sidereal_samples: {MMODE_FIXTURE_SIDEREAL_SAMPLES}
  integration_fraction: 1.0
execution:
  simulator: mmode
  mmode:
    convention: {MMODE_CONVENTION}
    frame_model: {MMODE_FRAME_MODEL}
    harmonic_convention: {MMODE_HARMONIC_CONVENTION}
    lmax: {MMODE_FIXTURE_LMAX}
    mmax: {MMODE_FIXTURE_MMAX}
    quadrature_nside: {MMODE_FIXTURE_QUADRATURE_NSIDE}
    working_memory_bytes: {MMODE_FIXTURE_WORKING_MEMORY_BYTES}
""".encode()


def _mmode_fixture_digest(role: str) -> str:
    """Return one deterministic 64-hex stand-in digest for a snapshot field.

    Section 10 requires the m-mode snapshot to carry ``iers_table_sha256`` and
    ``frame_certificate_sha256``; the *output* contract transports them and
    never interprets them, so this fixture derives them from its own identity
    instead of running the solver that would compute them.  The characterization
    families measure the real ones from real runs.
    """
    import hashlib

    return hashlib.sha256(
        b"radiosim.sci004.phase3-output-fixture.v1\x00"
        + role.encode("ascii")
        + b"\x00"
        + MMODE_FIXTURE_BYTES
    ).hexdigest()


def _mmode_direct_gate(cell_count: int) -> DirectGateRecord:
    """Return a passing Section 7.3 gate record for the output fixture.

    The values are the exact-agreement corner Section 7.3 admits explicitly --
    "with an exact-zero ``deficit_max_jy`` passing both" -- so the fixture never
    asserts a numerical claim it did not measure.
    """
    zero = "0" * 64
    return DirectGateRecord(
        predicate_id="sci004_two_tier_direct.v3",
        reference_cube_sha256=zero,
        candidate_cube_sha256=zero,
        reference_error_cube_sha256=zero,
        horizon_free_cube_sha256=zero,
        horizon_free_qcheck_cube_sha256=zero,
        quadrature_shell_cube_sha256=zero,
        expected_cell_count=cell_count,
        compared_finite_cell_count=cell_count,
        evaluated_error_cell_count=cell_count,
        numerical_scale_jy=1.0,
        horizon_free_shell_max_jy=0.0,
        horizon_free_shell_l2=0.0,
        horizon_free_shell_max_limit_jy=1e-8,
        horizon_free_shell_l2_limit=1e-8,
        quadrature_shell_max_jy=0.0,
        quadrature_shell_l2=0.0,
        reference_scale_jy=1.0,
        deficit_max_jy=0.0,
        deficit_l2=0.0,
        deficit_max_quarter_jy=0.0,
        deficit_max_half_jy=0.0,
        convergence_factor=2.0,
        pass_=True,
    )


def build_mmode_result(
    tmp_path: Path,
    *,
    dtype: str = "complex128",
    frequencies_hz: tuple[float, ...] = (100e6, 101.5e6),
    channel_widths_hz: tuple[float, ...] = (1.5e6, 1.5e6),
    receptors: dict[str, object] | None = None,
    sky_representation: str = "point_sources",
    components: tuple[str, ...] = ("point",),
    component_element_counts: tuple[int, ...] = (3,),
    tangent_polarization_frame: object = MMODE_TANGENT_FRAME_M1,
):
    """Build one canonical m-mode result on a genuine full-sidereal UTC grid.

    The configuration is resolved through the ordinary public path, so
    ``result.time_grid`` is the Section 3.1 grid mapped from exact ERA turns and
    its integration widths are the retained exposure-edge widths -- not a
    repeated cadence.  That distinction is what several Section 10 oracles below
    measure.
    """
    mapping = _result_mapping(
        tmp_path,
        frequencies_hz=frequencies_hz,
        channel_widths_hz=channel_widths_hz,
        receptors=receptors,
    )
    mapping["visibility"] = {"sky_representation": sky_representation}
    mapping["obs_time"] = {
        "mode": "full_sidereal",
        "start_time": "2025-01-01T00:00:00",
        "sidereal_samples": MMODE_FIXTURE_SIDEREAL_SAMPLES,
        "integration_fraction": 1.0,
    }
    mapping["execution"] = {
        "backend": "numpy",
        "offline": True,
        "simulator": "mmode",
        "mmode": {
            "convention": MMODE_CONVENTION,
            "frame_model": MMODE_FRAME_MODEL,
            "harmonic_convention": MMODE_HARMONIC_CONVENTION,
            "lmax": MMODE_FIXTURE_LMAX,
            "mmax": MMODE_FIXTURE_MMAX,
            "quadrature_nside": MMODE_FIXTURE_QUADRATURE_NSIDE,
            "working_memory_bytes": MMODE_FIXTURE_WORKING_MEMORY_BYTES,
        },
    }
    simulator = Simulator.from_mapping(mapping, base_dir=tmp_path)
    simulator._ensure_instrument_state()
    simulator._ensure_receptor_set()
    simulator._ensure_beam_system()
    backend = get_backend("numpy")
    baselines = simulator._instrument_state.selection.baselines
    time_grid = simulator.config.observation.time_grid
    shape = (len(time_grid), len(baselines), len(frequencies_hz), 2, 2)
    real = np.arange(1, np.prod(shape) + 1, dtype=np.float64).reshape(shape)
    receptor = (real + 1j * (real * 0.0625 + 0.015625)).astype(dtype)
    for baseline_index, baseline in enumerate(baselines):
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
    snapshot = MModeSolverSnapshot(
        input_identity_sha256=_mmode_fixture_digest("input_identity"),
        sky_representation=sky_representation,
        execution_path="polarized",
        components=components,
        component_element_counts=component_element_counts,
        sidereal_samples=len(time_grid),
        lmax=MMODE_FIXTURE_LMAX,
        mmax=MMODE_FIXTURE_MMAX,
        quadrature_nside=MMODE_FIXTURE_QUADRATURE_NSIDE,
        iers_table_sha256=_mmode_fixture_digest("iers_table"),
        frame_certificate_sha256=_mmode_fixture_digest("frame_certificate"),
        direct_gate=_mmode_direct_gate(
            len(time_grid) * len(baselines) * len(frequencies_hz) * 4
        ),
        frozen_gauss128_cube_sha256=_mmode_fixture_digest("frozen_gauss128_cube"),
        frozen_enclosure_error_cube_sha256=_mmode_fixture_digest(
            "frozen_enclosure_error_cube"
        ),
        tangent_polarization_frame=tangent_polarization_frame,
    )
    return build_simulation_result(
        receptor_visibilities=receptor,
        backend=backend,
        time_grid=time_grid,
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
        solver_provenance=MModeSolverResultProvenance(snapshot=snapshot),
        resolved_config=simulator.config.to_json_safe(),
        configuration_provenance=None,
        performance=ResultPerformance(
            setup_seconds=1.0,
            solver_seconds=2.0,
            solver_point_seconds=2.0,
            solver_healpix_seconds=0.0,
            result_construction_seconds=0.5,
            host_transfer_seconds=0.25,
            total_seconds=3.75,
        ),
        history=("simulated",),
    )


#: The three convention literals Section 10 requires the projected HISTORY to
#: name.  They are imported from production rather than restated, so the oracle
#: cannot drift from the snapshot the solver publishes.
MMODE_HISTORY_CONVENTIONS: tuple[str, ...] = (
    MMODE_CONVENTION,
    MMODE_FRAME_MODEL,
    MMODE_HARMONIC_CONVENTION,
)

_PHASE3_GREEN_CONTROL = (
    "tests/unit/test_io/test_standard_visibility.py::"
    "test_shared_projection_has_canonical_blt_polarization_and_phase[uvfits]"
)


def _phase3_case(
    case_id: str,
    requirement_id: str,
    function: str,
    *,
    module: str = "tests/unit/test_io/test_standard_visibility.py",
    expected_failure_kind: str = "missing-symbol",
    expected_failure_pattern: str = (r"has no attribute 'components'"),
    green_control: str = _PHASE3_GREEN_CONTROL,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "requirement_id": requirement_id,
        "test_nodeid": f"{module}::{function}",
        "expected_failure_kind": expected_failure_kind,
        "expected_failure_pattern": expected_failure_pattern,
        "fixture_defect_excluded_by": green_control,
        "fixture_bytes": MMODE_FIXTURE_BYTES,
    }


SCI004_PHASE3_RED_CASES: tuple[dict[str, object], ...] = (
    _phase3_case(
        "m3.standard.mmode-projects-to-four-correlations",
        "sci004.section-10.mmode-projects-with-four-correlation-products",
        "test_an_mmode_result_projects_with_four_correlation_products",
    ),
    _phase3_case(
        "m3.standard.history-names-the-mmode-conventions",
        "sci004.section-10.history-names-the-mmode-frame-and-harmonic-conventions",
        "test_the_projected_history_names_the_mmode_frame_and_harmonic_conventions",
    ),
    _phase3_case(
        "m3.standard.record-carries-the-tagged-mmode-snapshot",
        "sci004.section-10.projection-record-carries-the-tagged-mmode-snapshot",
        "test_the_projection_record_carries_the_complete_tagged_mmode_snapshot",
    ),
    _phase3_case(
        "m3.standard.reader-never-relabels-mmode-as-rime",
        "sci004.section-10.reader-never-relabels-mmode-as-rime",
        "test_the_projection_reader_never_relabels_an_mmode_run_as_rime",
    ),
    _phase3_case(
        "m3.standard.synthesized-utc-centres-and-widths",
        "sci004.section-10.projection-writes-the-synthesized-utc-grid",
        "test_the_projection_writes_the_synthesized_utc_centres_and_widths",
    ),
)

SCI004_PHASE3_RED_GREEN_CONTROLS: tuple[str, ...] = (_PHASE3_GREEN_CONTROL,)


def test_an_mmode_result_projects_with_four_correlation_products(
    tmp_path: Path,
) -> None:
    """Section 10: the m-mode arm reaches the shared standard projection.

    "Point, HEALPix, and hybrid remain solver provenance, not separate output
    products", and the public result "keeps its existing strict ``(T, B, F, 4)``
    visibility array in the four row-major correlation labels".  The projection
    is the one seam UVFITS and MS share, so this is where the tagged union first
    has to be representable.
    """
    result = build_mmode_result(tmp_path)

    projected = project_simulation_result(result, format="uvfits")

    assert projected.data.format == "uvfits"
    assert len(projected.data.correlations) == 4
    assert projected.data.visibilities.shape[-1] == 4
    assert projected.data.visibilities.shape[0] == len(result.time_grid)
    assert result.solver.solver == "mmode"


def test_the_projected_history_names_the_mmode_frame_and_harmonic_conventions(
    tmp_path: Path,
) -> None:
    """Section 10: "history lines naming the m-mode/frame/harmonic conventions".

    The three literals are production's own, imported above, so a writer that
    invented a fourth spelling would fail here rather than pass a paraphrase.
    """
    result = build_mmode_result(tmp_path)

    projected = project_simulation_result(result, format="uvfits")
    history = str(projected.uvdata.history)
    # "History *lines*", so the embedded projection record does not count: that
    # JSON already carried the snapshot before this phase, and reading the
    # requirement as satisfied by it would make the sentence say nothing.
    plain = [
        line
        for line in history.splitlines()
        if not line.startswith(PROJECTION_HISTORY_PREFIX)
    ]

    for literal in MMODE_HISTORY_CONVENTIONS:
        assert any(literal in line for line in plain), literal
    assert (
        len(history.encode("utf-8"))
        <= standard_visibility_module._PROJECTION_HISTORY_LIMIT
    )


def test_the_projection_record_carries_the_complete_tagged_mmode_snapshot(
    tmp_path: Path,
) -> None:
    """Section 10: HDF5 and the standard projection carry the *complete* snapshot.

    The record's ``solver`` entry must be the exact twenty-key m-mode arm in
    Section 10's order, never a truncated or re-sorted subset.
    """
    result = build_mmode_result(tmp_path)

    projected = project_simulation_result(result, format="uvfits")
    record, _lines = projection_record_from_history(projected.uvdata.history)
    solver = record["solver"]

    assert isinstance(solver, dict)
    assert tuple(solver) == MMODE_SOLVER_SNAPSHOT_KEYS
    assert solver["solver"] == "mmode"
    assert solver["convention"] == MMODE_CONVENTION
    assert solver["time_grid_convention"] == MMODE_TIME_GRID_CONVENTION
    assert solver["frame_model"] == MMODE_FRAME_MODEL
    assert solver["harmonic_convention"] == MMODE_HARMONIC_CONVENTION
    assert solver["quadrature_policy"] == MMODE_QUADRATURE_POLICY
    assert solver["truncation_policy"] == MMODE_TRUNCATION_POLICY
    assert solver["stokes_v_basis_bridge"] == MMODE_STOKES_BRIDGE
    assert solver["transform_execution_policy"] == MMODE_EXECUTION_POLICY
    assert record["source_scientific_sha256"] == result.scientific_sha256


def test_the_projection_reader_never_relabels_an_mmode_run_as_rime(
    tmp_path: Path,
) -> None:
    """Section 10: "a reader that silently labels it ``rime`` fails acceptance".

    The negative half is asserted as well as the positive one, because the
    failure mode this sentence names is a *successful* read that quietly
    produces the wrong arm.
    """
    result = build_mmode_result(tmp_path)

    projected = project_simulation_result(result, format="uvfits")
    record, _lines = projection_record_from_history(projected.uvdata.history)
    solver = record["solver"]

    assert solver["solver"] != "rime"
    assert solver["convention"] != "radiosim.rime-zenith-drift.v1"
    assert "sky_representation=" in str(projected.uvdata.history)
    assert solver["sky_representation"] == result.solver.sky_representation


def test_the_projection_writes_the_synthesized_utc_centres_and_widths(
    tmp_path: Path,
) -> None:
    """Section 10: every path writes "the same synthesized UTC sample centres
    and integration widths".

    The m-mode grid's widths come from Section 3.1's retained exposure edges, so
    they are *not* one repeated cadence; the fixture is chosen so that the
    distinction is measurable rather than notional.
    """
    result = build_mmode_result(tmp_path)
    widths = np.asarray(result.time_grid.integration_time_seconds, dtype=np.float64)

    assert len(set(widths.tolist())) > 1, "the ERA-derived widths are not uniform"

    projected = project_simulation_result(result, format="uvfits")
    stored = np.asarray(projected.uvdata.integration_time, dtype=np.float64)
    stored_by_time = stored.reshape(len(result.time_grid), -1)

    for index, width in enumerate(widths):
        assert np.all(stored_by_time[index] == width), index
