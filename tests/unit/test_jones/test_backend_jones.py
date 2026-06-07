"""Backend portability tests for batched Jones matrix construction."""

import numpy as np
import pytest

from radiosim.backends import get_backend
from radiosim.backends.base import BackendNotAvailableError
from radiosim.core.jones import JonesChain, JonesTerm
from radiosim.core.jones.beam.analytic import AnalyticBeamJones
from radiosim.core.jones.beam.analytic.composed import compute_aperture_beam
from radiosim.core.jones.beam.fits import FITSBeamJones
from radiosim.core.jones.geometric import GeometricPhaseJones


def _get_optional_backend(name: str):
    if name == "jax":
        pytest.importorskip("jax")
        kwargs = {"device": "cpu"}
    elif name == "numba":
        pytest.importorskip("numba")
        kwargs = {"mode": "cpu"}
    else:
        kwargs = {}

    try:
        return get_backend(name, **kwargs)
    except BackendNotAvailableError as exc:
        pytest.skip(str(exc))


class _DirectionDependentJones(JonesTerm):
    @property
    def name(self) -> str:
        return "test"

    @property
    def is_direction_dependent(self) -> bool:
        return True

    def compute_jones(
        self,
        antenna_idx: int,
        source_idx: int | None,
        freq_idx: int,
        time_idx: int,
        backend,
        **kwargs,
    ):
        scale = 1 if source_idx is None else source_idx + 1
        return backend.eye_complex(2, dtype=np.complex128) * scale


class _MissingBeamManager:
    def get_jones_matrix(self, **kwargs):
        return None


@pytest.mark.parametrize("backend_name", ["numpy", "numba"])
def test_default_all_source_jones_stacks_backend_matrices(backend_name: str):
    backend = _get_optional_backend(backend_name)
    term = _DirectionDependentJones()

    result = term.compute_jones_all_sources(
        antenna_idx=0,
        n_sources=3,
        freq_idx=0,
        time_idx=0,
        backend=backend,
    )

    result_np = backend.to_numpy(result)
    assert result_np.shape == (3, 2, 2)
    np.testing.assert_allclose(result_np[:, 0, 0], [1, 2, 3])
    np.testing.assert_allclose(result_np[:, 1, 1], [1, 2, 3])
    np.testing.assert_allclose(result_np[:, 0, 1], 0)
    np.testing.assert_allclose(result_np[:, 1, 0], 0)


@pytest.mark.parametrize("backend_name", ["numpy", "numba"])
def test_chain_all_source_identity_and_geometric_phase(backend_name: str):
    backend = _get_optional_backend(backend_name)
    chain = JonesChain(backend)

    identity = backend.to_numpy(
        chain.compute_antenna_jones_all_sources(
            antenna_idx=0,
            n_sources=2,
            freq_idx=0,
            time_idx=0,
        )
    )
    np.testing.assert_allclose(identity, np.broadcast_to(np.eye(2), (2, 2, 2)))

    chain.add_term(
        GeometricPhaseJones(
            source_lmn=np.array([[0.0, 0.0, 1.0], [0.25, 0.0, np.sqrt(0.9375)]]),
            wavelengths=np.array([1.0]),
        )
    )
    result = backend.to_numpy(
        chain.compute_antenna_jones_all_sources(
            antenna_idx=0,
            n_sources=2,
            freq_idx=0,
            time_idx=0,
            baseline_uvw=np.array([1.0, 0.0, 0.0]),
        )
    )

    expected_phase = np.exp(-2j * np.pi * np.array([0.0, 0.25]))
    np.testing.assert_allclose(result[:, 0, 0], expected_phase)
    np.testing.assert_allclose(result[:, 1, 1], expected_phase)
    np.testing.assert_allclose(result[:, 0, 1], 0)
    np.testing.assert_allclose(result[:, 1, 0], 0)


def test_fits_beam_fallback_identity_uses_backend_batch_eye():
    backend = _get_optional_backend("numpy")
    term = FITSBeamJones(
        beam_manager=_MissingBeamManager(),
        source_altaz=np.array([[1.0, 0.0], [0.5, 0.25]]),
        frequencies=np.array([100e6]),
    )

    result = backend.to_numpy(
        term.compute_jones_all_sources(
            antenna_idx=0,
            n_sources=2,
            freq_idx=0,
            time_idx=0,
            backend=backend,
        )
    )

    np.testing.assert_allclose(result, np.broadcast_to(np.eye(2), (2, 2, 2)))


def test_jax_chain_all_source_construction_is_functional():
    backend = _get_optional_backend("jax")
    chain = JonesChain(backend)
    chain.add_term(_DirectionDependentJones())

    result = backend.to_numpy(
        chain.compute_antenna_jones_all_sources(
            antenna_idx=0,
            n_sources=2,
            freq_idx=0,
            time_idx=0,
        )
    )

    np.testing.assert_allclose(result[:, 0, 0], [1, 2])


def test_compute_aperture_beam_builds_on_backend():
    """The analytic E-Jones builder must construct on the given backend device
    (functional, immutable-safe) instead of via host-NumPy in-place assignment."""
    backend = _get_optional_backend("jax")
    theta = np.linspace(0.0, 0.05, 6)
    phi = np.zeros_like(theta)

    jones = compute_aperture_beam(
        theta=theta,
        phi=phi,
        frequency=100e6,
        diameter=14.0,
        backend=backend,
    )

    # Built on the backend device, not as a host NumPy array.
    assert not isinstance(jones, np.ndarray)
    assert tuple(jones.shape) == (6, 2, 2)

    jones_np = backend.to_numpy(jones)
    # Diagonal beam: zero off-diagonals, equal on-diagonals.
    np.testing.assert_allclose(jones_np[:, 0, 1], 0)
    np.testing.assert_allclose(jones_np[:, 1, 0], 0)
    np.testing.assert_allclose(jones_np[:, 0, 0], jones_np[:, 1, 1])
    # Matches the host-NumPy reference (default backend=None path).
    ref = compute_aperture_beam(theta=theta, phi=phi, frequency=100e6, diameter=14.0)
    np.testing.assert_allclose(jones_np, ref, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("backend_name", ["numpy", "numba", "jax"])
def test_analytic_beam_jones_all_sources_matches_numpy(backend_name: str):
    """AnalyticBeamJones (E-term) yields matching Jones across backends and,
    for device backends, returns an on-device array (beam-inclusive parity)."""
    reference = _get_optional_backend("numpy")
    backend = _get_optional_backend(backend_name)

    n = 5
    alts = np.linspace(np.pi / 2, np.pi / 2 - 0.05, n)
    azs = np.linspace(0.0, 1.0, n)
    source_altaz = np.column_stack([alts, azs])
    freqs = np.array([100e6], dtype=np.float64)

    def _beam() -> AnalyticBeamJones:
        return AnalyticBeamJones(
            source_altaz=source_altaz, frequencies=freqs, diameter=14.0
        )

    expected = reference.to_numpy(
        _beam().compute_jones_all_sources(0, n, 0, 0, reference)
    )
    actual = backend.to_numpy(_beam().compute_jones_all_sources(0, n, 0, 0, backend))

    assert actual.shape == (n, 2, 2)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-9)
