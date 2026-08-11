"""PERF-001 P-b source-axis bucketing contract."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import radiosim.core.visibility as point_visibility
from radiosim.backends import get_backend
from radiosim.core.runtime_config import ResolvedSolverExecutionConfig
from radiosim.core.source_bucketing import (
    IDENTITY_SOURCE_BUCKET_POLICY,
    PRODUCTION_SOURCE_BUCKET_POLICY,
    _pad_host_repeated,
    _pad_host_zeros,
    _pad_reference_frequencies,
    _resolve_source_bucket,
)
from radiosim.core.visibility import _calculate_visibility, calculate_visibility
from radiosim.core.visibility_healpix import (
    _calculate_visibility_healpix,
    calculate_visibility_healpix,
)
from tests.fixtures.beamfits import write_scalar_efield_beamfits


@pytest.mark.parametrize(
    ("logical_count", "expected_kernel_count"),
    [
        (1, 1),
        (2, 2),
        (3, 4),
        (4, 4),
        (7, 8),
        (8, 8),
        (9, 16),
        (16, 16),
        (17, 32),
        (31, 32),
        (32, 32),
        (33, 64),
    ],
)
def test_compiled_production_policy_uses_next_power_of_two(
    logical_count: int,
    expected_kernel_count: int,
) -> None:
    plan = _resolve_source_bucket(
        logical_count,
        supports_compilation=True,
        policy=PRODUCTION_SOURCE_BUCKET_POLICY,
    )

    assert plan.logical_count == logical_count
    assert plan.kernel_count == expected_kernel_count
    assert plan.padding_count == expected_kernel_count - logical_count
    assert logical_count <= plan.kernel_count < 2 * logical_count


@pytest.mark.parametrize("supports_compilation", [False, True])
def test_identity_policy_is_an_observable_unpadded_reference(
    supports_compilation: bool,
) -> None:
    plan = _resolve_source_bucket(
        17,
        supports_compilation=supports_compilation,
        policy=IDENTITY_SOURCE_BUCKET_POLICY,
    )

    assert (plan.logical_count, plan.kernel_count, plan.padding_count) == (17, 17, 0)


def test_noncompiling_production_backend_is_unpadded() -> None:
    plan = _resolve_source_bucket(
        17,
        supports_compilation=False,
        policy=PRODUCTION_SOURCE_BUCKET_POLICY,
    )

    assert (plan.logical_count, plan.kernel_count, plan.padding_count) == (17, 17, 0)


@pytest.mark.parametrize("logical_count", [-1, 0])
def test_bucket_plan_requires_a_nonzero_logical_count(logical_count: int) -> None:
    with pytest.raises(ValueError, match="logical_count must be positive"):
        _resolve_source_bucket(
            logical_count,
            supports_compilation=True,
            policy=PRODUCTION_SOURCE_BUCKET_POLICY,
        )


def test_bucket_plan_rejects_unknown_private_policy() -> None:
    with pytest.raises(ValueError, match="source bucket policy"):
        _resolve_source_bucket(
            3,
            supports_compilation=True,
            policy="nearest_multiple_of_eight",
        )


def test_dummy_rows_repeat_finite_direction_metadata_and_zero_signal() -> None:
    plan = _resolve_source_bucket(
        3,
        supports_compilation=True,
        policy=PRODUCTION_SOURCE_BUCKET_POLICY,
    )
    directions = np.array([0.25, 0.5, 0.75], dtype=np.float64)
    signal = np.array([3.0, 2.0, 1.0], dtype=np.float32)

    padded_directions = _pad_host_repeated(directions, plan)
    padded_signal = _pad_host_zeros(signal, plan)

    np.testing.assert_array_equal(padded_directions[:3], directions)
    np.testing.assert_array_equal(padded_signal[:3], signal)
    assert padded_directions.dtype == directions.dtype
    assert padded_signal.dtype == signal.dtype
    assert np.all(np.isfinite(padded_directions))
    assert padded_directions[3] == directions[0]
    assert padded_signal[3] == 0.0


def test_zero_padding_supports_source_axes_zero_and_one() -> None:
    plan = _resolve_source_bucket(
        3,
        supports_compilation=True,
        policy=PRODUCTION_SOURCE_BUCKET_POLICY,
    )
    source_major = np.arange(6, dtype=np.float64).reshape(3, 2)
    source_minor = source_major.T.copy()

    padded_major = _pad_host_zeros(source_major, plan, axis=0)
    padded_minor = _pad_host_zeros(source_minor, plan, axis=1)

    np.testing.assert_array_equal(padded_major[:3], source_major)
    np.testing.assert_array_equal(padded_minor[:, :3], source_minor)
    np.testing.assert_array_equal(padded_major[3], np.zeros(2))
    np.testing.assert_array_equal(padded_minor[:, 3], np.zeros(2))


def test_dummy_reference_frequency_copies_a_positive_finite_logical_value() -> None:
    plan = _resolve_source_bucket(
        3,
        supports_compilation=True,
        policy=PRODUCTION_SOURCE_BUCKET_POLICY,
    )
    reference_frequencies = np.array([0.0, 151e6, 0.0], dtype=np.float64)

    padded = _pad_reference_frequencies(
        reference_frequencies,
        plan,
        fallback_hz=150e6,
    )

    np.testing.assert_array_equal(padded[:3], reference_frequencies)
    assert padded[3] == 151e6
    assert np.isfinite(padded[3])
    assert padded[3] > 0.0


def test_dummy_reference_frequency_uses_valid_fallback_when_catalogue_has_none() -> (
    None
):
    plan = _resolve_source_bucket(
        3,
        supports_compilation=True,
        policy=PRODUCTION_SOURCE_BUCKET_POLICY,
    )

    padded = _pad_reference_frequencies(
        np.zeros(3, dtype=np.float64),
        plan,
        fallback_hz=150e6,
    )

    assert padded[3] == 150e6
    assert np.isfinite(padded[3])
    assert padded[3] > 0.0


def test_private_policy_is_absent_from_both_public_solver_signatures() -> None:
    assert (
        "_source_bucket_policy"
        not in inspect.signature(calculate_visibility).parameters
    )
    assert (
        "_source_bucket_policy"
        not in inspect.signature(calculate_visibility_healpix).parameters
    )
    assert (
        inspect.signature(_calculate_visibility)
        .parameters["_source_bucket_policy"]
        .kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert (
        inspect.signature(_calculate_visibility_healpix)
        .parameters["_source_bucket_policy"]
        .kind
        is inspect.Parameter.KEYWORD_ONLY
    )


def _assert_jax_bucket_tolerance(reference: object, candidate: object) -> None:
    backend = get_backend("jax", device="cpu")
    reference_array = np.asarray(backend.to_numpy(reference))
    candidate_array = np.asarray(backend.to_numpy(candidate))
    scale = max(1.0, float(np.max(np.abs(reference_array))))

    assert reference_array.shape == candidate_array.shape
    assert reference_array.dtype == candidate_array.dtype
    assert np.all(
        np.abs(candidate_array - reference_array)
        <= 1e-12 * scale + 1e-12 * np.abs(reference_array)
    )


class _ThreeVisibleSkyCoord:
    """Astropy stand-in whose three logical sources are safely above horizon."""

    def __init__(self, **_kwargs: object) -> None:
        pass

    def transform_to(self, _frame: object) -> SimpleNamespace:
        return SimpleNamespace(
            az=SimpleNamespace(rad=np.array([0.1, 0.2, 0.3], dtype=np.float64)),
            alt=SimpleNamespace(rad=np.array([0.8, 0.9, 1.0], dtype=np.float64)),
        )


class _ThreeVisiblePixels:
    def __len__(self) -> int:
        return 3

    def transform_to(self, _frame: object) -> SimpleNamespace:
        return _ThreeVisibleSkyCoord().transform_to(_frame)


class _ThreePixelHealpix:
    nside = 1
    pixel_solid_angle = 1.0
    pixel_coords = _ThreeVisiblePixels()

    def __init__(self) -> None:
        self._stokes = (
            np.array([2.0, 1.0, 0.5], dtype=np.float64),
            np.array([0.2, 0.1, 0.0], dtype=np.float64),
            np.array([0.0, 0.1, 0.05], dtype=np.float64),
            np.array([0.05, 0.0, 0.0], dtype=np.float64),
        )

    def get_map_at_frequency(self, _frequency: float) -> np.ndarray:
        return self._stokes[0]

    def get_stokes_maps_at_frequency(
        self,
        _frequency: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._stokes


def _three_source_arrays() -> dict[str, object]:
    zeros = np.zeros(3, dtype=np.float64)
    return {
        "ra_rad": zeros.copy(),
        "dec_rad": zeros.copy(),
        "flux": np.array([2.0, 1.0, 0.5], dtype=np.float64),
        "spectral_index": np.array([-0.7, -0.8, -0.9], dtype=np.float64),
        "stokes_q": np.array([0.2, 0.1, 0.0], dtype=np.float64),
        "stokes_u": np.array([0.0, 0.1, 0.05], dtype=np.float64),
        "stokes_v": np.array([0.05, 0.0, 0.0], dtype=np.float64),
        "ref_freq": np.full(3, 100e6, dtype=np.float64),
        "rotation_measure": zeros.copy(),
        "spectral_coeffs": None,
        "per_channel_flux": None,
        "per_channel_stokes_q": None,
        "per_channel_stokes_u": None,
        "per_channel_stokes_v": None,
        "channel_frequencies": None,
        "major_arcsec": np.full(3, 120.0, dtype=np.float64),
        "minor_arcsec": np.full(3, 60.0, dtype=np.float64),
        "pa_deg": np.full(3, 30.0, dtype=np.float64),
    }


@pytest.mark.parametrize("beam_family", ["analytic", "shared_fits"])
def test_padded_complete_solvers_keep_dummy_signal_neutral_for_beams(
    tmp_path,
    monkeypatch,
    beam_family: str,
) -> None:
    """Repeated dummy directions are safe through analytic and FITS beams."""
    from tests.unit.test_core.test_beam_solver_integration import (
        FREQUENCIES,
        LOCATION,
        TIME_GRID,
        _solver_components,
    )

    if beam_family == "analytic":
        beams: dict[str, object] = {"mode": "analytic"}
    else:
        beam_path = write_scalar_efield_beamfits(
            tmp_path,
            filename="perf001-p-b.beamfits",
        ).path
        beams = {
            "mode": "shared_fits",
            "beam": {"kind": "fits", "path": beam_path.name},
        }

    simulator, instrument, beam_system = _solver_components(tmp_path, beams)
    monkeypatch.setattr(point_visibility, "SkyCoord", _ThreeVisibleSkyCoord)
    backend = get_backend("jax", device="cpu")
    private_common = {
        "instrument": instrument,
        "beam_system": beam_system,
        "location": LOCATION,
        "time_grid": TIME_GRID,
        "frequencies": FREQUENCIES,
        "backend": backend,
        "receptors": simulator.receptors,
    }

    point_identity = _calculate_visibility(
        source_arrays=_three_source_arrays(),
        **private_common,
        _source_bucket_policy=IDENTITY_SOURCE_BUCKET_POLICY,
    )
    point_bucketed = _calculate_visibility(
        source_arrays=_three_source_arrays(),
        **private_common,
        _source_bucket_policy=PRODUCTION_SOURCE_BUCKET_POLICY,
    )

    healpix_sky = SimpleNamespace(
        healpix=_ThreePixelHealpix(),
        has_polarized_healpix_maps=True,
        brightness_conversion="rayleigh-jeans",
        model_name="perf001-three-pixel",
    )
    healpix_identity = _calculate_visibility_healpix(
        sky_model=healpix_sky,
        include_polarization=True,
        **private_common,
        _source_bucket_policy=IDENTITY_SOURCE_BUCKET_POLICY,
    )
    healpix_bucketed = _calculate_visibility_healpix(
        sky_model=healpix_sky,
        include_polarization=True,
        **private_common,
        _source_bucket_policy=PRODUCTION_SOURCE_BUCKET_POLICY,
    )

    _assert_jax_bucket_tolerance(point_identity, point_bucketed)
    _assert_jax_bucket_tolerance(healpix_identity, healpix_bucketed)


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("solver_kind", ["point", "healpix"])
def test_padded_solver_with_direction_and_baseline_terms_matches_identity_control(
    tmp_path,
    workers: int,
    solver_kind: str,
) -> None:
    """P-b stays neutral through P, Q, and serial/parallel scheduling."""
    from tests.characterization.test_tier6_current_behavior import (
        WORKLOAD_LOCATION,
        WORKLOAD_TIME_GRID,
        _workload_healpix_model,
        _workload_point_sources,
    )
    from tests.unit.test_core.test_jones_resolution import (
        solver_components_with_jones,
    )

    instrument, beam_system, receptors, jones_terms, frequencies = (
        solver_components_with_jones(
            tmp_path,
            {
                "P": {"enabled": True},
                "Q": {"bandwidth_smearing": True, "time_smearing": True},
            },
            mount_types="alt-az",
        )
    )
    backend = get_backend("jax", device="cpu")
    execution = ResolvedSolverExecutionConfig(workers=workers, executor="thread")
    common = {
        "instrument": instrument,
        "beam_system": beam_system,
        "location": WORKLOAD_LOCATION,
        "time_grid": WORKLOAD_TIME_GRID,
        "frequencies": frequencies,
        "backend": backend,
        "receptors": receptors,
        "jones_terms": jones_terms,
        "solver_execution": execution,
    }

    if solver_kind == "point":
        sources = _workload_point_sources(polarized=True, gaussian=True)
        for name, values in tuple(sources.items()):
            if isinstance(values, np.ndarray) and values.ndim == 1:
                sources[name] = np.concatenate((values, values[-1:]))
        sources["ra_rad"][-1] += 0.01
        sources["dec_rad"][-1] += 0.01
        identity = _calculate_visibility(
            source_arrays=sources,
            **common,
            _source_bucket_policy=IDENTITY_SOURCE_BUCKET_POLICY,
        )
        bucketed = _calculate_visibility(
            source_arrays=sources,
            **common,
            _source_bucket_policy=PRODUCTION_SOURCE_BUCKET_POLICY,
        )
    else:
        sky_model = _workload_healpix_model(polarized=True)
        identity = _calculate_visibility_healpix(
            sky_model=sky_model,
            include_polarization=True,
            **common,
            _source_bucket_policy=IDENTITY_SOURCE_BUCKET_POLICY,
        )
        bucketed = _calculate_visibility_healpix(
            sky_model=sky_model,
            include_polarization=True,
            **common,
            _source_bucket_policy=PRODUCTION_SOURCE_BUCKET_POLICY,
        )

    _assert_jax_bucket_tolerance(identity, bucketed)


def test_zero_visible_short_circuit_precedes_bucket_resolution(
    tmp_path,
    monkeypatch,
) -> None:
    """The accepted exact-zero route never invents or schedules dummy sources."""
    import radiosim.core.visibility_healpix as healpix_visibility
    from tests.unit.test_core.test_beam_solver_integration import (
        FREQUENCIES,
        LOCATION,
        TIME_GRID,
        _NonVisibleHealpix,
        _NonVisibleSkyCoord,
        _solver_components,
    )

    simulator, instrument, beam_system = _solver_components(
        tmp_path,
        {"mode": "analytic"},
    )
    monkeypatch.setattr(point_visibility, "SkyCoord", _NonVisibleSkyCoord)

    def forbidden_bucket(*_args: object, **_kwargs: object) -> None:
        pytest.fail("zero-visible time step reached source bucket resolution")

    monkeypatch.setattr(point_visibility, "_resolve_source_bucket", forbidden_bucket)
    monkeypatch.setattr(
        healpix_visibility,
        "_resolve_source_bucket",
        forbidden_bucket,
    )
    backend = get_backend("jax", device="cpu")
    common = {
        "instrument": instrument,
        "beam_system": beam_system,
        "location": LOCATION,
        "time_grid": TIME_GRID,
        "frequencies": FREQUENCIES,
        "backend": backend,
        "receptors": simulator.receptors,
        "_source_bucket_policy": PRODUCTION_SOURCE_BUCKET_POLICY,
    }

    point = _calculate_visibility(
        source_arrays=_three_source_arrays(),
        **common,
    )
    healpix = _calculate_visibility_healpix(
        sky_model=SimpleNamespace(
            healpix=_NonVisibleHealpix(),
            has_polarized_healpix_maps=False,
            brightness_conversion="rayleigh-jeans",
            model_name="perf001-non-visible",
        ),
        **common,
    )

    np.testing.assert_array_equal(backend.to_numpy(point), 0.0)
    np.testing.assert_array_equal(backend.to_numpy(healpix), 0.0)


@pytest.mark.parametrize("backend_name", ["numpy", "dask"])
@pytest.mark.parametrize(
    ("include_polarization", "output_units"),
    [(False, "Jy"), (False, "K.sr"), (True, "Jy")],
)
def test_fast_healpix_preserves_pre_p_b_cast_before_scale_bytes(
    tmp_path,
    monkeypatch,
    backend_name: str,
    include_polarization: bool,
    output_units: str,
) -> None:
    """Linear HEALPix scaling keeps the accepted fast-precision byte order."""
    import radiosim.core.visibility_healpix as healpix_visibility
    from tests.unit.test_core.test_beam_solver_integration import (
        FREQUENCIES,
        LOCATION,
        TIME_GRID,
        _solver_components,
    )

    simulator, instrument, beam_system = _solver_components(
        tmp_path,
        {"mode": "analytic"},
    )
    backend_kwargs = {"mode": "cpu"} if backend_name == "dask" else {}
    backend = get_backend(backend_name, precision="fast", **backend_kwargs)
    scale = 1.0000000596046448
    healpix = _ThreePixelHealpix()
    healpix.pixel_solid_angle = scale
    healpix._stokes = (
        np.array([1.00000006, 1.23456789, 2.34567891], dtype=np.float64),
        np.array([0.20000003, 0.10000002, 0.05000001], dtype=np.float64),
        np.array([0.05000003, 0.10000004, 0.20000005], dtype=np.float64),
        np.array([0.02500003, 0.07500004, 0.12500005], dtype=np.float64),
    )
    monkeypatch.setattr(
        healpix_visibility,
        "rayleigh_jeans_factor",
        lambda _frequency, _omega: scale,
    )

    captured_coherency: list[np.ndarray] = []
    original_factory = healpix_visibility.baseline_contraction_for

    def recording_factory(resolved_backend):
        contraction = original_factory(resolved_backend)

        def recording_contraction(
            j_p,
            j_q,
            coherency,
            phase,
            envelope,
            stokes_i,
        ):
            captured_coherency.append(
                np.array(resolved_backend.to_numpy(coherency), copy=True)
            )
            return contraction(
                j_p,
                j_q,
                coherency,
                phase,
                envelope,
                stokes_i,
            )

        return recording_contraction

    monkeypatch.setattr(
        healpix_visibility,
        "baseline_contraction_for",
        recording_factory,
    )

    result = _calculate_visibility_healpix(
        sky_model=SimpleNamespace(
            healpix=healpix,
            has_polarized_healpix_maps=include_polarization,
            brightness_conversion="rayleigh-jeans",
            model_name="perf001-fast-cast-order",
        ),
        instrument=instrument,
        beam_system=beam_system,
        location=LOCATION,
        time_grid=TIME_GRID,
        frequencies=FREQUENCIES,
        backend=backend,
        receptors=simulator.receptors,
        output_units=output_units,
        include_polarization=include_polarization,
        _source_bucket_policy=PRODUCTION_SOURCE_BUCKET_POLICY,
    )

    assert np.all(np.isfinite(backend.to_numpy(result)))
    assert len(captured_coherency) == 1
    legacy_stokes = [
        backend.asarray(values, dtype=backend.default_real_dtype) * scale
        for values in healpix._stokes
    ]
    if include_polarization:
        expected = point_visibility.stokes_to_coherency(
            *legacy_stokes,
            xp=backend.xp,
        )
    else:
        expected = backend.batch_eye(
            (len(legacy_stokes[0]),),
            2,
            dtype=backend.default_complex_dtype,
        )
        expected = expected * (legacy_stokes[0] / 2.0)[:, None, None]
    expected_array = np.asarray(backend.to_numpy(expected))

    assert captured_coherency[0].dtype == expected_array.dtype
    assert captured_coherency[0].tobytes() == expected_array.tobytes()


def test_healpix_planck_converts_logical_temperatures_before_zero_flux_padding(
    tmp_path,
    monkeypatch,
    caplog,
) -> None:
    """No dummy thermodynamic temperature enters nonlinear Planck conversion."""
    import radiosim.core.visibility_healpix as healpix_visibility
    from tests.unit.test_core.test_beam_solver_integration import (
        FREQUENCIES,
        LOCATION,
        TIME_GRID,
        _solver_components,
    )

    simulator, instrument, beam_system = _solver_components(
        tmp_path,
        {"mode": "analytic"},
    )
    backend = get_backend("jax", device="cpu")
    logical_temperature_counts: list[int] = []
    coherency_fluxes: list[np.ndarray] = []
    original_coherency = healpix_visibility.stokes_to_coherency

    def observed_planck(
        temperature_k: np.ndarray,
        _frequency_hz: float,
        _omega_pixel: float,
    ) -> np.ndarray:
        logical_temperature_counts.append(len(temperature_k))
        return np.arange(1, len(temperature_k) + 1, dtype=np.float64)

    def observed_coherency(
        stokes_i: object,
        stokes_q: object,
        stokes_u: object,
        stokes_v: object,
        *,
        xp: object,
    ) -> object:
        coherency_fluxes.append(np.asarray(backend.to_numpy(stokes_i)))
        return original_coherency(
            stokes_i,
            stokes_q,
            stokes_u,
            stokes_v,
            xp=xp,
        )

    monkeypatch.setattr(
        healpix_visibility,
        "_host_planck_flux_density",
        observed_planck,
    )
    monkeypatch.setattr(
        healpix_visibility,
        "stokes_to_coherency",
        observed_coherency,
    )
    caplog.set_level("DEBUG", logger=healpix_visibility.__name__)

    result = _calculate_visibility_healpix(
        sky_model=SimpleNamespace(
            healpix=_ThreePixelHealpix(),
            has_polarized_healpix_maps=True,
            brightness_conversion="planck",
            model_name="perf001-planck-order",
        ),
        instrument=instrument,
        beam_system=beam_system,
        location=LOCATION,
        time_grid=TIME_GRID,
        frequencies=FREQUENCIES,
        backend=backend,
        receptors=simulator.receptors,
        include_polarization=True,
        _source_bucket_policy=PRODUCTION_SOURCE_BUCKET_POLICY,
    )

    assert np.all(np.isfinite(backend.to_numpy(result)))
    assert logical_temperature_counts == [3]
    assert len(coherency_fluxes) == 1
    np.testing.assert_array_equal(coherency_fluxes[0], [1.0, 2.0, 3.0, 0.0])
    assert "logical=3, kernel=4" in caplog.text
