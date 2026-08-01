"""Backend portability tests for direction-batched Jones matrix construction."""

import numpy as np
import pytest

import radiosim.core.jones as jones_module
import radiosim.core.visibility as visibility_module
from radiosim.api import Simulator
from radiosim.backends import get_backend
from radiosim.backends.base import BackendNotAvailableError
from radiosim.core.instrument import AntennaId
from radiosim.core.instrument_adapters import SolverInstrumentView
from radiosim.core.jones import DirectionBatch, JonesChain, JonesTerm
from tests.fixtures.configs import valid_config_mapping


def _get_optional_backend(name: str):
    if name == "jax":
        kwargs = {"device": "cpu"}
    elif name == "dask":
        pytest.importorskip("dask")
        kwargs = {"mode": "cpu"}
    else:
        kwargs = {}

    try:
        return get_backend(name, **kwargs)
    except BackendNotAvailableError as exc:
        pytest.skip(str(exc))


def _directions(n_dir: int) -> DirectionBatch:
    """A direction batch whose entries are distinguishable per direction."""
    values = np.linspace(0.1, 1.0, n_dir)
    return DirectionBatch(
        alt_rad=values,
        az_rad=values,
        dir_l=values,
        dir_m=values,
        dir_n=values,
        ra_rad=values,
        dec_rad=values,
        hour_angle_rad=values,
        n_dir=n_dir,
    )


def _evaluate(chain: JonesChain, *, n_dir: int, antenna_idx: int = 0):
    return chain.compute_antenna_jones_batch(
        antenna_idx=antenna_idx,
        directions=_directions(n_dir),
        frequency_hz=1.0e8,
        freq_idx=0,
        time_mjd=60_000.0,
        time_idx=0,
        dtype=np.complex128,
    )


class _DirectionDependentJones(JonesTerm):
    """A DDE probe whose value varies along the direction axis."""

    @property
    def name(self) -> str:
        return "test"

    @property
    def is_direction_dependent(self) -> bool:
        return True

    def compute_jones_batch(
        self,
        *,
        antenna_idx: int,
        directions,
        frequency_hz: float,
        freq_idx: int,
        time_mjd: float,
        time_idx: int,
        backend,
        dtype,
    ):
        scale = backend.asarray(
            np.arange(1, directions.n_dir + 1, dtype=np.float64),
            dtype=dtype,
        )
        return (
            backend.batch_eye((directions.n_dir,), 2, dtype=dtype)
            * scale[:, None, None]
        )


@pytest.mark.parametrize("backend_name", ["numpy", "dask"])
def test_direction_dependent_term_returns_one_matrix_per_direction(backend_name: str):
    backend = _get_optional_backend(backend_name)
    term = _DirectionDependentJones()

    result = term.compute_jones_batch(
        antenna_idx=0,
        directions=_directions(3),
        frequency_hz=1.0e8,
        freq_idx=0,
        time_mjd=60_000.0,
        time_idx=0,
        backend=backend,
        dtype=np.complex128,
    )

    result_np = backend.to_numpy(result)
    assert result_np.shape == (3, 2, 2)
    np.testing.assert_allclose(result_np[:, 0, 0], [1, 2, 3])
    np.testing.assert_allclose(result_np[:, 1, 1], [1, 2, 3])
    np.testing.assert_allclose(result_np[:, 0, 1], 0)
    np.testing.assert_allclose(result_np[:, 1, 0], 0)


@pytest.mark.parametrize("backend_name", ["numpy", "dask"])
def test_empty_chain_is_a_single_broadcastable_identity(backend_name: str):
    """An empty chain is ``(1, 2, 2)``, not ``n_dir`` copies of the identity."""
    backend = _get_optional_backend(backend_name)
    chain = JonesChain(backend)

    identity = backend.to_numpy(_evaluate(chain, n_dir=5))

    assert identity.shape == (1, 2, 2)
    np.testing.assert_allclose(identity[0], np.eye(2))


@pytest.mark.parametrize("backend_name", ["numpy", "dask", "jax"])
def test_chain_broadcasts_the_direction_axis_from_the_scalar_seed(backend_name: str):
    """The ``(1, 2, 2)`` seed broadcasts against a ``(n_dir, 2, 2)`` factor."""
    backend = _get_optional_backend(backend_name)
    chain = JonesChain(backend)
    chain.add_term(_DirectionDependentJones())

    result = backend.to_numpy(_evaluate(chain, n_dir=2))

    assert result.shape == (2, 2, 2)
    np.testing.assert_allclose(result[:, 0, 0], [1, 2])


@pytest.mark.parametrize("backend_name", ["numpy", "dask", "jax"])
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

    directions = DirectionBatch(
        alt_rad=altitude,
        az_rad=azimuth,
        dir_l=np.cos(altitude) * np.sin(azimuth),
        dir_m=np.cos(altitude) * np.cos(azimuth),
        dir_n=np.sin(altitude),
        ra_rad=np.zeros(5),
        dec_rad=np.zeros(5),
        hour_angle_rad=np.zeros(5),
        n_dir=5,
    )
    actual = backend.to_numpy(
        chain.compute_antenna_jones_batch(
            antenna_idx=0,
            directions=directions,
            frequency_hz=100e6,
            freq_idx=0,
            time_mjd=60_000.0,
            time_idx=0,
            dtype=np.complex128,
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


def test_resolved_beam_jones_evaluates_each_handler_once_per_batch(tmp_path):
    """The per-handler cache moved from the HEALPix solver into the adapter."""
    simulator = Simulator.from_mapping(
        valid_config_mapping(tmp_path),
        base_dir=tmp_path,
    )
    simulator._ensure_instrument_state()
    simulator._ensure_beam_system()
    instrument = SolverInstrumentView.from_state(simulator._instrument_state)
    altitude = np.full(4, 1.2)
    azimuth = np.linspace(0.0, 1.0, 4)
    term = visibility_module._ResolvedBeamJones(
        beam_system=simulator.beam_system,
        instrument=instrument,
        altitude_rad=altitude,
        azimuth_rad=azimuth,
        frequency_hz=100e6,
        time_mjd=60_000.0,
    )
    backend = get_backend("numpy")
    directions = DirectionBatch(
        alt_rad=altitude,
        az_rad=azimuth,
        dir_l=np.cos(altitude) * np.sin(azimuth),
        dir_m=np.cos(altitude) * np.cos(azimuth),
        dir_n=np.sin(altitude),
        ra_rad=np.zeros(4),
        dec_rad=np.zeros(4),
        hour_angle_rad=np.zeros(4),
        n_dir=4,
    )

    calls = 0
    original = type(simulator.beam_system).evaluate_jones

    def counted(self, antenna_id, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, antenna_id, **kwargs)

    type(simulator.beam_system).evaluate_jones = counted
    try:
        for antenna_idx in range(len(instrument.antenna_numbers)):
            term.compute_jones_batch(
                antenna_idx=antenna_idx,
                directions=directions,
                frequency_hz=100e6,
                freq_idx=0,
                time_mjd=60_000.0,
                time_idx=0,
                backend=backend,
                dtype=np.complex128,
            )
    finally:
        type(simulator.beam_system).evaluate_jones = original

    # One shared analytic handler covers the whole array in this fixture.
    assert len(simulator.beam_system.state.handlers) == 1
    assert calls == 1


def test_resolved_beam_jones_remains_private_to_visibility_solver():
    assert "_ResolvedBeamJones" not in jones_module.__all__
    assert not hasattr(jones_module, "_ResolvedBeamJones")
