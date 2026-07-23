"""Backend portability tests for batched Jones matrix construction."""

import numpy as np
import pytest

import radiosim.core.jones as jones_module
import radiosim.core.visibility as visibility_module
from radiosim.api import Simulator
from radiosim.backends import get_backend
from radiosim.backends.base import BackendNotAvailableError
from radiosim.core.instrument import AntennaId
from radiosim.core.instrument_adapters import SolverInstrumentView
from radiosim.core.jones import JonesChain, JonesTerm
from radiosim.core.jones.geometric import GeometricPhaseJones
from tests.fixtures.configs import valid_config_mapping


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


@pytest.mark.parametrize("backend_name", ["numpy", "numba", "jax"])
def test_resolved_beam_jones_chain_matches_canonical_system(
    tmp_path,
    backend_name: str,
):
    """The private solver E-term preserves canonical values on every backend."""
    backend = _get_optional_backend(backend_name)
    simulator = Simulator.from_mapping(
        valid_config_mapping(tmp_path),
        base_dir=tmp_path,
    )
    simulator._ensure_instrument_state()
    simulator._ensure_beam_system()
    instrument = SolverInstrumentView.from_state(simulator._instrument_state)
    altitude = np.linspace(np.pi / 2, np.pi / 2 - 0.05, 5)
    azimuth = np.linspace(0.0, 1.0, 5)
    term = visibility_module._ResolvedBeamJones(
        beam_system=simulator.beam_system,
        instrument=instrument,
        altitude_rad=altitude,
        azimuth_rad=azimuth,
        frequency_hz=100e6,
        time_mjd=60_000.0,
    )
    chain = JonesChain(backend)
    chain.add_term(term)

    actual = backend.to_numpy(
        chain.compute_antenna_jones_all_sources(
            antenna_idx=0,
            n_sources=5,
            freq_idx=0,
            time_idx=0,
            antenna_number=0,
        )
    )
    expected = simulator.beam_system.evaluate_jones(
        AntennaId(0, "ANT0"),
        altitude_rad=altitude,
        azimuth_rad=azimuth,
        frequency_hz=100e6,
        time_mjd=60_000.0,
    )

    assert actual.shape == (5, 2, 2)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-9)


def test_resolved_beam_jones_remains_private_to_visibility_solver():
    assert "_ResolvedBeamJones" not in jones_module.__all__
    assert not hasattr(jones_module, "_ResolvedBeamJones")
