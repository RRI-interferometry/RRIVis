"""Section 27 rows B1 and B2: NumPy/JAX-CPU and NumPy/Dask solver parity.

Every workload in ``Tier6HybridRuntimePlan.md`` Section 13.4 is solved three
times -- once on NumPy (the reference), once on CPU JAX, once on the Dask
backend -- and compared under the Section 13.5 rule:

- JAX-CPU: ``|V_jax - V_numpy| <= atol + rtol * |V_numpy|`` with
  ``rtol = 1e-12`` and ``atol = 1e-12 * max(1, max|V_numpy|)``. Bit-identity is
  **not** required and must not be asserted: XLA may fuse and reorder the source
  reduction, so a float64 sum over ``N`` terms can legitimately differ in its
  last bits. Tier 6A measured these workloads as bit-identical *before* the
  restructure, on tiny inputs where XLA has little to fuse; that measurement is
  explicitly not licence to tighten this to equality.
- Dask: bit-identical, because the backend delegates to the same NumPy
  operations. Asserting anything weaker there would hide a real defect.

None of this is skippable. A CPU-only ``jax``/``jaxlib`` is a declared
dependency of every pixi environment exactly so these rows are measured
(Sections 28, 31, 32.8).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from radiosim.api import Simulator
from radiosim.backends import get_backend
from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky import HealpixData, SkyModel
from radiosim.core.source_bucketing import IDENTITY_SOURCE_BUCKET_POLICY
from radiosim.core.time_grid import build_observation_time_grid
from radiosim.core.visibility import _calculate_visibility, calculate_visibility
from radiosim.core.visibility_healpix import (
    _calculate_visibility_healpix,
    calculate_visibility_healpix,
)
from tests.fixtures.configs import valid_config_mapping

# Section 13.5 tolerance for float64 accumulation.
RTOL = 1e-12
ATOL_SCALE = 1e-12

FREQUENCIES = np.array([100e6, 101e6], dtype=np.float64)
LOCATION = EarthLocation.from_geodetic(21.4283 * u.deg, -30.72152 * u.deg, 1073.0 * u.m)
OBSTIME = Time("2025-01-01T00:00:00")
TIME_GRID = build_observation_time_grid(
    start_time=OBSTIME.isot, duration_seconds=2.0, cadence_seconds=1.0
)
SINGLE_TIME_GRID = build_observation_time_grid(
    start_time=OBSTIME.isot, duration_seconds=1.0, cadence_seconds=1.0
)
LST_RAD = OBSTIME.sidereal_time("apparent", longitude=LOCATION.lon).rad


def _solver_components(tmp_path: Path, **overrides: Any):
    data = valid_config_mapping(tmp_path, **overrides)
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    simulator._ensure_instrument_state()
    simulator._ensure_receptor_set()
    simulator._ensure_beam_system()
    from radiosim.core.instrument_adapters import SolverInstrumentView

    return (
        SolverInstrumentView.from_state(simulator._instrument_state),
        simulator.beam_system,
        simulator.receptors,
    )


def _point_sources(
    *,
    polarized: bool,
    gaussian: bool,
    n_sources: int = 2,
    per_channel: bool = False,
) -> dict[str, Any]:
    n = n_sources
    zeros = np.zeros(n, dtype=np.float64)
    q = zeros.copy()
    u_stokes = zeros.copy()
    v = zeros.copy()
    if polarized:
        q[0] = 0.2
        u_stokes[min(1, n - 1)] = 0.1
        v[0] = 0.05
    flux = np.linspace(2.0, 1.0, n, dtype=np.float64)
    per_channel_flux = (
        np.vstack([flux, flux * 1.1]).astype(np.float64) if per_channel else None
    )
    return {
        "ra_rad": LST_RAD + np.arange(n, dtype=np.float64) * 0.01,
        "dec_rad": -0.536 + np.arange(n, dtype=np.float64) * 0.01,
        "flux": flux,
        "spectral_index": np.linspace(-0.7, -0.8, n, dtype=np.float64),
        "stokes_q": q,
        "stokes_u": u_stokes,
        "stokes_v": v,
        "ref_freq": np.full(n, 100e6, dtype=np.float64),
        "rotation_measure": zeros.copy(),
        "spectral_coeffs": None,
        "per_channel_flux": per_channel_flux,
        "per_channel_stokes_q": (
            np.vstack([q, q * 1.1]).astype(np.float64)
            if polarized and per_channel
            else None
        ),
        "per_channel_stokes_u": (
            np.vstack([u_stokes, u_stokes * 1.1]).astype(np.float64)
            if polarized and per_channel
            else None
        ),
        "per_channel_stokes_v": (
            np.vstack([v, v * 1.1]).astype(np.float64)
            if polarized and per_channel
            else None
        ),
        "channel_frequencies": FREQUENCIES.copy() if per_channel else None,
        "major_arcsec": np.full(n, 120.0) if gaussian else zeros.copy(),
        "minor_arcsec": np.full(n, 60.0) if gaussian else zeros.copy(),
        "pa_deg": np.full(n, 30.0) if gaussian else zeros.copy(),
    }


def _healpix_model(*, polarized: bool) -> SkyModel:
    npix = 12
    maps = np.linspace(1.0, 2.0, npix, dtype=np.float64)
    maps = np.vstack([maps, maps * 1.1])
    return SkyModel(
        healpix=HealpixData(
            maps=maps,
            nside=1,
            frequencies=FREQUENCIES,
            coordinate_frame="icrs",
            q_maps=np.full_like(maps, 0.1) if polarized else None,
            u_maps=np.full_like(maps, 0.05) if polarized else None,
            v_maps=np.full_like(maps, 0.02) if polarized else None,
        ),
        model_name="tier6h-parity",
        brightness_conversion="rayleigh-jeans",
        precision=PrecisionConfig.standard(),
    )


_HETEROGENEOUS_RECEPTORS = {
    "default": {"basis": "linear", "feed_rotation_deg": 0.0},
    "overrides": [{"antenna": {"kind": "number", "number": 1}, "basis": "circular"}],
    "output_basis": "linear",
}


def _run_point(
    backend_name: str,
    tmp_path: Path,
    *,
    polarized: bool = True,
    gaussian: bool = False,
    heterogeneous: bool = False,
    single_time: bool = False,
    n_sources: int = 2,
    per_channel: bool = False,
    source_bucket_policy: str | None = None,
) -> np.ndarray:
    overrides = {"receptors": _HETEROGENEOUS_RECEPTORS} if heterogeneous else {}
    instrument, beam_system, receptors = _solver_components(tmp_path, **overrides)
    backend = _backend(backend_name)
    solver = (
        calculate_visibility if source_bucket_policy is None else _calculate_visibility
    )
    kwargs: dict[str, Any] = {}
    if source_bucket_policy is not None:
        kwargs["_source_bucket_policy"] = source_bucket_policy
    cube = solver(
        instrument=instrument,
        beam_system=beam_system,
        source_arrays=_point_sources(
            polarized=polarized,
            gaussian=gaussian,
            n_sources=n_sources,
            per_channel=per_channel,
        ),
        location=LOCATION,
        time_grid=SINGLE_TIME_GRID if single_time else TIME_GRID,
        frequencies=FREQUENCIES,
        backend=backend,
        receptors=receptors,
        **kwargs,
    )
    return np.asarray(backend.to_numpy(cube))


def _run_healpix(
    backend_name: str,
    tmp_path: Path,
    *,
    polarized: bool,
    source_bucket_policy: str | None = None,
) -> np.ndarray:
    instrument, beam_system, receptors = _solver_components(tmp_path)
    backend = _backend(backend_name)
    solver = (
        calculate_visibility_healpix
        if source_bucket_policy is None
        else _calculate_visibility_healpix
    )
    kwargs: dict[str, Any] = {}
    if source_bucket_policy is not None:
        kwargs["_source_bucket_policy"] = source_bucket_policy
    cube = solver(
        _healpix_model(polarized=polarized),
        instrument=instrument,
        beam_system=beam_system,
        location=LOCATION,
        time_grid=TIME_GRID,
        frequencies=FREQUENCIES,
        backend=backend,
        receptors=receptors,
        include_polarization=polarized,
        **kwargs,
    )
    return np.asarray(backend.to_numpy(cube))


def _run_hybrid(backend_name: str, tmp_path: Path) -> np.ndarray:
    """Section 13.4 row 6: the hybrid workload is the sum of both components."""
    point = _run_point(backend_name, tmp_path, polarized=True)
    healpix = _run_healpix(backend_name, tmp_path, polarized=True)
    return point + healpix


def _backend(name: str):
    if name == "jax":
        return get_backend("jax", device="cpu")
    if name == "dask":
        return get_backend("dask", mode="cpu")
    return get_backend("numpy")


#: Every row of the Section 13.4 parity matrix.
WORKLOADS = {
    "point_unpolarized_1time_2freq": lambda name, tmp: _run_point(
        name, tmp, polarized=False, single_time=True
    ),
    "point_polarized_2times": lambda name, tmp: _run_point(name, tmp, polarized=True),
    "point_gaussian_morphology": lambda name, tmp: _run_point(
        name, tmp, polarized=True, gaussian=True
    ),
    "healpix_scalar": lambda name, tmp: _run_healpix(name, tmp, polarized=False),
    "healpix_polarized": lambda name, tmp: _run_healpix(name, tmp, polarized=True),
    "hybrid_point_plus_healpix": _run_hybrid,
    "heterogeneous_receptor_bases": lambda name, tmp: _run_point(
        name, tmp, polarized=True, heterogeneous=True
    ),
}


def assert_within_section_13_5(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Assert the Section 13.5 predicate and return the measured deviation."""
    assert candidate.shape == reference.shape
    assert candidate.dtype == reference.dtype
    scale = max(1.0, float(np.max(np.abs(reference))))
    atol = ATOL_SCALE * scale
    deviation = float(np.max(np.abs(candidate - reference)))
    allowed = atol + RTOL * np.abs(reference)
    assert np.all(np.abs(candidate - reference) <= allowed), (
        f"max deviation {deviation:.3e} exceeds atol={atol:.3e} + rtol={RTOL:.0e}*|V|"
    )
    return deviation


@pytest.mark.parametrize("workload", sorted(WORKLOADS))
def test_b1_numpy_and_jax_cpu_agree_within_the_stated_tolerance(
    tmp_path, workload: str
) -> None:
    """B1: all seven Section 13.4 workloads, NumPy vs JAX-CPU."""
    reference = WORKLOADS[workload]("numpy", tmp_path)
    candidate = WORKLOADS[workload]("jax", tmp_path)

    # A digest of an all-zero cube would prove nothing.
    assert float(np.max(np.abs(reference))) > 0.0
    assert_within_section_13_5(reference, candidate)


@pytest.mark.parametrize("workload", sorted(WORKLOADS))
def test_b2_dask_is_bit_identical_to_numpy(tmp_path, workload: str) -> None:
    """B2: the Dask backend delegates to NumPy, so nothing weaker is acceptable."""
    reference = WORKLOADS[workload]("numpy", tmp_path)
    candidate = WORKLOADS[workload]("dask", tmp_path)

    assert float(np.max(np.abs(reference))) > 0.0
    assert candidate.dtype == reference.dtype
    assert np.array_equal(candidate, reference)


def test_hybrid_parity_row_also_satisfies_the_additivity_invariant(tmp_path) -> None:
    """Section 13.4's hybrid row carries the Section 9.2 additivity requirement."""
    point = _run_point("jax", tmp_path, polarized=True)
    healpix = _run_healpix("jax", tmp_path, polarized=True)
    summed = _run_hybrid("jax", tmp_path)

    assert np.array_equal(summed, point + healpix)


@pytest.mark.parametrize(
    "workload",
    [
        "point_unpolarized",
        "point_polarized_gaussian",
        "point_polarized_per_channel",
        "healpix_scalar",
        "healpix_polarized",
    ],
)
def test_p_b_jax_bucketed_solver_matches_same_solver_identity_control(
    tmp_path,
    workload: str,
) -> None:
    """P-b dummy slots remain neutral through complete point/HEALPix routes."""

    def run(policy: str | None) -> np.ndarray:
        if workload == "point_unpolarized":
            return _run_point(
                "jax",
                tmp_path,
                polarized=False,
                n_sources=3,
                source_bucket_policy=policy,
            )
        if workload == "point_polarized_gaussian":
            return _run_point(
                "jax",
                tmp_path,
                polarized=True,
                gaussian=True,
                n_sources=3,
                source_bucket_policy=policy,
            )
        if workload == "point_polarized_per_channel":
            return _run_point(
                "jax",
                tmp_path,
                polarized=True,
                n_sources=3,
                per_channel=True,
                source_bucket_policy=policy,
            )
        return _run_healpix(
            "jax",
            tmp_path,
            polarized=workload == "healpix_polarized",
            source_bucket_policy=policy,
        )

    reference = run(IDENTITY_SOURCE_BUCKET_POLICY)
    candidate = run(None)

    assert float(np.max(np.abs(reference))) > 0.0
    assert_within_section_13_5(reference, candidate)


@pytest.mark.parametrize("backend_name", ["numpy", "dask"])
@pytest.mark.parametrize("solver_kind", ["point", "healpix"])
def test_p_b_noncompiling_backends_are_byte_identical_to_identity_control(
    tmp_path,
    backend_name: str,
    solver_kind: str,
) -> None:
    """P-b does not pad or reorder NumPy/Dask source arrays."""
    if solver_kind == "point":
        reference = _run_point(
            backend_name,
            tmp_path,
            polarized=True,
            gaussian=True,
            n_sources=3,
            per_channel=True,
            source_bucket_policy=IDENTITY_SOURCE_BUCKET_POLICY,
        )
        candidate = _run_point(
            backend_name,
            tmp_path,
            polarized=True,
            gaussian=True,
            n_sources=3,
            per_channel=True,
        )
    else:
        reference = _run_healpix(
            backend_name,
            tmp_path,
            polarized=True,
            source_bucket_policy=IDENTITY_SOURCE_BUCKET_POLICY,
        )
        candidate = _run_healpix(
            backend_name,
            tmp_path,
            polarized=True,
        )

    assert candidate.dtype == reference.dtype
    assert np.array_equal(candidate, reference)
    assert candidate.tobytes() == reference.tobytes()


def test_parity_is_measured_rather_than_skipped() -> None:
    """Section 31: after 6H there is no JAX skip left, in any module.

    The six ``pytest.importorskip("jax")`` guards Tier 6A counted are gone --
    not replaced by a differently-worded skip. A missing JAX is now a broken
    environment, and every one of these modules must fail loudly rather than
    quietly report a green run that measured nothing.
    """
    repository_root = Path(__file__).resolve().parents[3]
    formerly_skipping = (
        "tests/unit/test_backends/test_jax_backend.py",
        "tests/unit/test_core/test_sky_backend.py",
        "tests/unit/test_core/test_sky_spectral.py",
        "tests/unit/test_core/test_visibility_backend.py",
        "tests/unit/test_jones/test_backend_jones.py",
    )
    for relative in formerly_skipping:
        source = (repository_root / relative).read_text(encoding="utf-8")
        assert 'importorskip("jax")' not in source, relative
