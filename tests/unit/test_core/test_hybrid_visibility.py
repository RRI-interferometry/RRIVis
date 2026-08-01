"""Tier 6F: the hybrid solve mode, its rejections, and canonical summation.

Covers ``Tier6HybridRuntimePlan.md`` Section 27 rows ``H1``, ``H3``, ``H4``,
``H8``, ``H11``, and ``E1``-``E3``, plus the ``ArrayBackend.add`` primitive the
summation is built on.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from radiosim.api import Simulator
from radiosim.backends import get_backend
from radiosim.core.hybrid import (
    HYBRID_COMPONENT_NAMES,
    HybridSkyError,
    check_representation_compatibility,
    component_names_for_representation,
    solve_sky,
)
from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky.containers.healpix import HealpixData
from radiosim.core.sky.containers.model import SkyFormat, SkyModel
from radiosim.core.sky.operations.factories import create_from_arrays
from tests.fixtures.configs import hybrid_config_mapping, valid_config_mapping

# The Section 18.3 runtime rejections, byte for byte.
HYBRID_MISSING_PAYLOAD_MESSAGE = (
    "visibility.sky_representation=hybrid requires a sky model with both a "
    "point-source payload and a HEALPix payload; the resolved model carries "
    "only {point_sources}. Request point_sources or healpix_map, or add a "
    "source of the missing kind."
)
POINT_WOULD_DROP_MAPS_MESSAGE = (
    "visibility.sky_representation=point_sources would discard the HEALPix "
    "payload carried by the resolved sky model. Request hybrid to sum both "
    "components, or set visibility.allow_lossy_point_materialization=true to "
    "convert the HEALPix payload to point sources."
)
HEALPIX_WOULD_RASTERIZE_MESSAGE = (
    "visibility.sky_representation=healpix_map would rasterize 2 point "
    "source(s) into the HEALPix grid, which quantizes positions to pixel "
    "centers. Request hybrid to sum both components, or set "
    "visibility.allow_lossy_point_rasterization=true to opt in."
)


def _point_model(n_sources: int = 2) -> SkyModel:
    precision = PrecisionConfig.standard()
    return create_from_arrays(
        ra_rad=np.linspace(0.1, 0.2, n_sources),
        dec_rad=np.zeros(n_sources),
        flux=np.linspace(1.0, 2.0, n_sources),
        reference_frequency=150e6,
        precision=precision,
    )


def _healpix_model() -> SkyModel:
    precision = PrecisionConfig.standard()
    frequencies = np.asarray([100e6, 150e6], dtype=np.float64)
    return SkyModel(
        healpix=HealpixData(
            maps=np.full((frequencies.size, 12), 3.0, dtype=np.float64),
            nside=1,
            frequencies=frequencies,
            coordinate_frame="icrs",
        ),
        precision=precision,
    )


def _hybrid_model() -> SkyModel:
    return _point_model().replace(healpix=_healpix_model().healpix)


# =========================================================================
# The backend summation primitive
# =========================================================================


def test_backend_add_is_the_backend_domain_summation_primitive() -> None:
    """The hybrid sum routes through the backend, not Python ``+``."""
    backend = get_backend("numpy")
    left = np.arange(8, dtype=np.complex128).reshape(2, 2, 2)
    right = np.full((2, 2, 2), 0.5 + 0.25j, dtype=np.complex128)

    summed = backend.add(left, right)

    assert np.array_equal(summed, left + right)
    assert summed.dtype == np.dtype(np.complex128)


# =========================================================================
# H4 / component identity
# =========================================================================


def test_component_names_follow_the_fixed_section_8_3_order() -> None:
    assert HYBRID_COMPONENT_NAMES == ("point", "healpix")
    assert component_names_for_representation("point_sources") == ("point",)
    assert component_names_for_representation("healpix_map") == ("healpix",)
    assert component_names_for_representation("hybrid") == ("point", "healpix")
    with pytest.raises(ValueError, match="unsupported sky representation"):
        component_names_for_representation("m_mode")


def test_hybrid_components_receive_the_identical_shared_objects(tmp_path) -> None:
    """H4: object identity, not equality, across the two component calls."""
    simulator = Simulator.from_mapping(
        hybrid_config_mapping(tmp_path), base_dir=tmp_path
    )
    simulator.setup()

    point_calls: list[dict[str, object]] = []
    healpix_calls: list[dict[str, object]] = []

    import radiosim.core.visibility_healpix as healpix_module

    original_point = simulator._simulator.calculate_visibilities
    original_healpix = healpix_module.calculate_visibility_healpix

    def record_point(**kwargs):
        point_calls.append(kwargs)
        return original_point(**kwargs)

    def record_healpix(**kwargs):
        healpix_calls.append(kwargs)
        return original_healpix(**kwargs)

    solve_sky(
        sky_representation="hybrid",
        sky_model=simulator._sky_model,
        source_arrays=simulator._source_arrays,
        point_solver=type(
            "Recorder", (), {"calculate_visibilities": staticmethod(record_point)}
        )(),
        backend=simulator._backend,
        instrument=simulator._solver_instrument_view,
        beam_system=simulator.beam_system,
        location=simulator._location,
        time_grid=simulator._resolved.observation.time_grid,
        frequencies=simulator._frequencies_hz,
        receptors=simulator.receptors,
        solver_execution=simulator._resolved.execution.solver,
    )
    healpix_module.calculate_visibility_healpix = record_healpix
    try:
        solve_sky(
            sky_representation="healpix_map",
            sky_model=simulator._sky_model,
            source_arrays=simulator._source_arrays,
            point_solver=simulator._simulator,
            backend=simulator._backend,
            instrument=simulator._solver_instrument_view,
            beam_system=simulator.beam_system,
            location=simulator._location,
            time_grid=simulator._resolved.observation.time_grid,
            frequencies=simulator._frequencies_hz,
            receptors=simulator.receptors,
            solver_execution=simulator._resolved.execution.solver,
        )
    finally:
        healpix_module.calculate_visibility_healpix = original_healpix

    assert len(point_calls) == 1
    assert len(healpix_calls) == 1
    point, healpix = point_calls[0], healpix_calls[0]
    for key in ("instrument", "beam_system", "location", "time_grid", "receptors"):
        assert point[key] is healpix[key], key
        assert point[key] is not None
    assert point["instrument"] is simulator._solver_instrument_view
    assert point["beam_system"] is simulator.beam_system
    assert point["receptors"] is simulator.receptors
    assert point["time_grid"] is simulator._resolved.observation.time_grid
    assert point["backend"] is healpix["backend"] is simulator._backend
    assert point["frequencies"] is healpix["frequencies"]


# =========================================================================
# H1 -- additivity, bit-identical on NumPy
# =========================================================================


def test_hybrid_is_bit_identical_to_the_sum_of_its_components(tmp_path) -> None:
    """H1/S1: the same shared inputs, three solves, exact additivity."""
    simulator = Simulator.from_mapping(
        hybrid_config_mapping(tmp_path), base_dir=tmp_path
    )
    simulator.setup()
    shared = {
        "sky_model": simulator._sky_model,
        "source_arrays": simulator._source_arrays,
        "point_solver": simulator._simulator,
        "backend": simulator._backend,
        "instrument": simulator._solver_instrument_view,
        "beam_system": simulator.beam_system,
        "location": simulator._location,
        "time_grid": simulator._resolved.observation.time_grid,
        "frequencies": simulator._frequencies_hz,
        "receptors": simulator.receptors,
        "solver_execution": simulator._resolved.execution.solver,
    }

    hybrid = solve_sky(sky_representation="hybrid", **shared)
    point = solve_sky(sky_representation="point_sources", **shared)
    healpix = solve_sky(sky_representation="healpix_map", **shared)

    expected = np.asarray(point.receptor_visibilities) + np.asarray(
        healpix.receptor_visibilities
    )
    actual = np.asarray(hybrid.receptor_visibilities)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()

    assert hybrid.component_names == ("point", "healpix")
    assert point.component_names == ("point",)
    assert healpix.component_names == ("healpix",)
    assert hybrid.component_element_counts == (
        simulator._sky_model.n_point_sources,
        simulator._sky_model.n_healpix_pixels,
    )
    assert hybrid.execution_path == "polarized"


def test_hybrid_summation_is_order_independent_for_two_components(tmp_path) -> None:
    """S3: the fixed order is a reproducibility choice, not an arithmetic one."""
    simulator = Simulator.from_mapping(
        hybrid_config_mapping(tmp_path), base_dir=tmp_path
    )
    simulator.setup()
    shared = {
        "sky_model": simulator._sky_model,
        "source_arrays": simulator._source_arrays,
        "point_solver": simulator._simulator,
        "backend": simulator._backend,
        "instrument": simulator._solver_instrument_view,
        "beam_system": simulator.beam_system,
        "location": simulator._location,
        "time_grid": simulator._resolved.observation.time_grid,
        "frequencies": simulator._frequencies_hz,
        "receptors": simulator.receptors,
        "solver_execution": simulator._resolved.execution.solver,
    }
    point = np.asarray(
        solve_sky(sky_representation="point_sources", **shared).receptor_visibilities
    )
    healpix = np.asarray(
        solve_sky(sky_representation="healpix_map", **shared).receptor_visibilities
    )

    assert (point + healpix).tobytes() == (healpix + point).tobytes()


# =========================================================================
# H11 -- an empty component is still a component
# =========================================================================


def test_hybrid_with_a_zero_flux_point_component_still_sums(tmp_path) -> None:
    """H11: a present-but-contributionless payload needs no special case."""
    simulator = Simulator.from_mapping(
        hybrid_config_mapping(tmp_path), base_dir=tmp_path
    )
    simulator.setup()
    zeroed = dict(simulator._source_arrays)
    zeroed["flux"] = np.zeros_like(np.asarray(zeroed["flux"]))
    shared = {
        "sky_model": simulator._sky_model,
        "point_solver": simulator._simulator,
        "backend": simulator._backend,
        "instrument": simulator._solver_instrument_view,
        "beam_system": simulator.beam_system,
        "location": simulator._location,
        "time_grid": simulator._resolved.observation.time_grid,
        "frequencies": simulator._frequencies_hz,
        "receptors": simulator.receptors,
        "solver_execution": simulator._resolved.execution.solver,
    }

    hybrid = solve_sky(sky_representation="hybrid", source_arrays=zeroed, **shared)
    healpix = solve_sky(
        sky_representation="healpix_map", source_arrays=zeroed, **shared
    )

    assert (
        np.asarray(hybrid.receptor_visibilities).tobytes()
        == np.asarray(healpix.receptor_visibilities).tobytes()
    )
    # The component is still declared, with its true (nonzero) element count.
    assert hybrid.component_names == ("point", "healpix")
    assert hybrid.component_element_counts[0] == len(zeroed["ra_rad"])


# =========================================================================
# E1-E3 -- the exact Section 18.3 runtime rejections
# =========================================================================


def test_hybrid_request_with_one_payload_is_rejected() -> None:
    """E1, byte for byte."""
    point_only = _point_model()
    with pytest.raises(HybridSkyError) as excinfo:
        check_representation_compatibility(
            sky_representation="hybrid",
            contributed_models=[point_only],
            resolved_model=point_only,
            allow_lossy_point_rasterization=False,
        )
    assert str(excinfo.value) == HYBRID_MISSING_PAYLOAD_MESSAGE


def test_point_request_that_would_drop_maps_is_rejected() -> None:
    """E2, byte for byte, for both the surviving and the dropped payload."""
    hybrid = _hybrid_model()
    with pytest.raises(HybridSkyError) as excinfo:
        check_representation_compatibility(
            sky_representation="point_sources",
            contributed_models=[hybrid],
            resolved_model=hybrid,
            allow_lossy_point_rasterization=False,
        )
    assert str(excinfo.value) == POINT_WOULD_DROP_MAPS_MESSAGE

    # The combination already dropped the contributor's maps, so the resolved
    # model no longer carries them.  The request is rejected all the same.
    with pytest.raises(HybridSkyError) as dropped:
        check_representation_compatibility(
            sky_representation="point_sources",
            contributed_models=[hybrid, _point_model()],
            resolved_model=_point_model(4),
            allow_lossy_point_rasterization=False,
        )
    assert str(dropped.value) == POINT_WOULD_DROP_MAPS_MESSAGE


def test_point_request_over_a_point_only_model_is_accepted() -> None:
    """A HEALPix-only contributor converted under the existing opt-in is fine."""
    check_representation_compatibility(
        sky_representation="point_sources",
        contributed_models=[_point_model(), _healpix_model()],
        resolved_model=_point_model(4),
        allow_lossy_point_rasterization=False,
    )


def test_healpix_request_that_would_rasterize_is_rejected() -> None:
    """E3, byte for byte, and the opt-in that restores the old behavior."""
    healpix_only = _healpix_model()
    with pytest.raises(HybridSkyError) as excinfo:
        check_representation_compatibility(
            sky_representation="healpix_map",
            contributed_models=[_point_model(2), healpix_only],
            resolved_model=healpix_only,
            allow_lossy_point_rasterization=False,
        )
    assert str(excinfo.value) == HEALPIX_WOULD_RASTERIZE_MESSAGE

    check_representation_compatibility(
        sky_representation="healpix_map",
        contributed_models=[_point_model(2), healpix_only],
        resolved_model=healpix_only,
        allow_lossy_point_rasterization=True,
    )


def test_healpix_request_over_a_diffuse_only_model_needs_no_opt_in() -> None:
    healpix_only = _healpix_model()
    check_representation_compatibility(
        sky_representation="healpix_map",
        contributed_models=[healpix_only],
        resolved_model=healpix_only,
        allow_lossy_point_rasterization=False,
    )


# =========================================================================
# The same three rejections through the high-level API
# =========================================================================


def test_setup_rejects_a_hybrid_request_over_a_point_only_config(tmp_path) -> None:
    data = valid_config_mapping(
        tmp_path,
        visibility={"sky_representation": "hybrid"},
    )
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    with pytest.raises(HybridSkyError) as excinfo:
        simulator.setup()
    assert str(excinfo.value) == HYBRID_MISSING_PAYLOAD_MESSAGE


def test_setup_rejects_a_point_request_that_would_drop_maps(
    tmp_path, monkeypatch
) -> None:
    """The D3 shape: one loader whose model already carries both payloads.

    Driven here by patching the combine entry point, because no in-tree
    *offline* loader returns a hybrid model.  Two separate contributors take a
    different route: the point-source concatenation refuses the HEALPix-only
    contributor first (Section 20.1 step 8 precedes step 9), which is asserted
    in ``tests/unit/test_core/test_sky_combine.py``.
    """
    from radiosim.core.sky.combine import pipeline

    data = valid_config_mapping(tmp_path)
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    original = pipeline.prepare_sky_model

    def hybridize(*args, **kwargs):
        resolved = original(*args, **kwargs)
        return resolved.replace(healpix=_healpix_model().healpix)

    monkeypatch.setattr(pipeline, "prepare_sky_model", hybridize)

    with pytest.raises(HybridSkyError) as excinfo:
        simulator.setup()
    assert str(excinfo.value) == POINT_WOULD_DROP_MAPS_MESSAGE


def test_setup_rejects_a_point_request_over_two_mixed_contributors(tmp_path) -> None:
    """Section 20.1: combination refuses first, and its message names hybrid."""
    data = hybrid_config_mapping(tmp_path)
    data["visibility"] = {
        "sky_representation": "point_sources",
    }
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    with pytest.raises(ValueError, match="sky_representation=hybrid"):
        simulator.setup()


def test_setup_rejects_a_healpix_request_that_would_rasterize(tmp_path) -> None:
    data = valid_config_mapping(
        tmp_path,
        visibility={
            "sky_representation": "healpix_map",
        },
    )
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    with pytest.raises(HybridSkyError) as excinfo:
        simulator.setup()
    assert str(excinfo.value) == HEALPIX_WOULD_RASTERIZE_MESSAGE


def test_setup_accepts_the_rasterization_opt_in(tmp_path) -> None:
    """The capability survives; only its silence is removed."""
    data = valid_config_mapping(
        tmp_path,
        visibility={
            "sky_representation": "healpix_map",
            "allow_lossy_point_rasterization": True,
        },
    )
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulator.setup()
    assert simulator._sky_model.healpix is not None
    assert simulator._sky_model.point is None


# =========================================================================
# H3, H8 -- reproducibility and fingerprint separation
# =========================================================================


def test_two_hybrid_runs_produce_one_scientific_fingerprint(tmp_path) -> None:
    """H3/S3."""
    data = hybrid_config_mapping(tmp_path)
    first = Simulator.from_mapping(data, base_dir=tmp_path).run(progress=False)
    second = Simulator.from_mapping(data, base_dir=tmp_path).run(progress=False)
    assert first.scientific_sha256 == second.scientific_sha256
    assert first.visibilities.tobytes() == second.visibilities.tobytes()


def test_hybrid_and_point_only_fingerprints_differ(tmp_path) -> None:
    """H8: component names and counts are inside ``scientific_sha256``."""
    hybrid = Simulator.from_mapping(
        hybrid_config_mapping(tmp_path / "hybrid"), base_dir=tmp_path / "hybrid"
    ).run(progress=False)
    point = Simulator.from_mapping(
        hybrid_config_mapping(tmp_path / "point", component="point"),
        base_dir=tmp_path / "point",
    ).run(progress=False)

    assert hybrid.solver.sky_representation == "hybrid"
    assert point.solver.sky_representation == "point_sources"
    assert hybrid.scientific_sha256 != point.scientific_sha256


@pytest.fixture(autouse=True)
def _make_component_dirs(tmp_path: Path):
    for name in ("hybrid", "point", "healpix"):
        (tmp_path / name).mkdir(exist_ok=True)
    yield


def test_sky_format_gains_no_hybrid_member() -> None:
    """Section 8.1: hybrid is a solve mode, never a payload representation."""
    assert {member.value for member in SkyFormat} == {"point_sources", "healpix_map"}
