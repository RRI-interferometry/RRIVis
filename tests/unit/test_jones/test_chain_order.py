"""Tier 5D: the canonical Jones chain composition order.

``Tier5ReceptorFeedPlan.md`` Section 19.1 fixes the factorization, leftmost
nearest the correlator::

    J_p = H_p G_p B_p D_p P_p C_p E_p T_p Z_p        (K applied separately)

:class:`~radiosim.core.jones.chain.JonesChain` composes
``terms[0] @ terms[1] @ ... @ terms[-1]``, so that factorization is exactly the
order in which the solver must *add* the terms.  ``C`` and ``H`` are the first
non-commuting factors RadioSim composes, which is why the order is proven here
with deliberately non-commuting synthetic terms (invariant S13) rather than
inferred from a run whose factors all commute.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation

from radiosim.api import Simulator
from radiosim.backends import get_backend
from radiosim.core.beam import BeamSystem
from radiosim.core.instrument import AntennaId
from radiosim.core.instrument_adapters import SolverInstrumentView
from radiosim.core.jones import JonesChain, JonesTerm
from radiosim.core.receptor import ResolvedReceptorSet, UnsupportedFeedGeometryError
from radiosim.core.visibility import _build_jones_chain
from tests.fixtures.configs import valid_config_mapping

FREQUENCIES_HZ = np.array([100_000_000.0], dtype=np.float64)
TIME_MJD = 60_676.0
LOCATION = EarthLocation.from_geodetic(
    21.4283 * u.deg,
    -30.72152 * u.deg,
    1073.0 * u.m,
)

#: Section 18.1 ``S``, written from the plan rather than imported.
PLAN_S = (1.0 / np.sqrt(2.0)) * np.array(
    [[1.0, 1.0j], [1.0, -1.0j]],
    dtype=np.complex128,
)

# Two deliberately non-commuting, non-unitary matrices.  ``FIRST @ SECOND`` and
# ``SECOND @ FIRST`` differ in every entry.
FIRST = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=np.complex128)
SECOND = np.array([[1.0, 0.0], [3.0j, 1.0]], dtype=np.complex128)

CANONICAL_ORDER = ("H", "G", "B", "D", "P", "C", "E", "T", "Z")


class _ConstantJones(JonesTerm):
    """A direction-independent term returning one fixed non-unitary matrix."""

    def __init__(self, label: str, matrix: np.ndarray) -> None:
        self._label = label
        self._matrix = np.array(matrix, dtype=np.complex128, copy=True)

    @property
    def name(self) -> str:
        return self._label

    @property
    def is_direction_dependent(self) -> bool:
        return False

    def compute_jones(
        self,
        antenna_idx: int,
        source_idx: int | None,
        freq_idx: int,
        time_idx: int,
        backend: Any,
        **kwargs: Any,
    ) -> Any:
        return backend.xp.array(self._matrix, dtype=np.complex128)


def _simulator(tmp_path: Path, receptors: dict[str, object] | None) -> Simulator:
    mapping = valid_config_mapping(
        tmp_path,
        frequency={
            "mode": "explicit",
            "channel_frequencies_hz": FREQUENCIES_HZ.tolist(),
            "channel_widths_hz": [1e6],
        },
    )
    if receptors is not None:
        mapping["receptors"] = receptors
    simulator = Simulator.from_mapping(mapping, base_dir=tmp_path)
    simulator._ensure_instrument_state()
    simulator._ensure_receptor_set()
    simulator._ensure_beam_system()
    return simulator


def _solver_components(
    tmp_path: Path,
    receptors: dict[str, object] | None = None,
) -> tuple[SolverInstrumentView, BeamSystem, ResolvedReceptorSet]:
    simulator = _simulator(tmp_path, receptors)
    return (
        SolverInstrumentView.from_state(simulator._instrument_state),
        simulator.beam_system,
        simulator.receptors,
    )


def _chain(
    tmp_path: Path,
    jones_config: dict[str, object],
    receptors: dict[str, object] | None = None,
    *,
    n_sources: int = 2,
) -> JonesChain:
    instrument, beam_system, receptor_set = _solver_components(tmp_path, receptors)
    return _build_jones_chain(
        get_backend("numpy"),
        jones_config,
        instrument,
        np.full(n_sources, 1.0, dtype=np.float64),
        np.full(n_sources, 0.5, dtype=np.float64),
        FREQUENCIES_HZ[0],
        0,
        n_sources,
        LOCATION,
        TIME_MJD,
        beam_system,
        receptor_set,
    )


def plan_rotation(chi_rad: float) -> np.ndarray:
    """Section 18.1 ``R(chi)``, written from the plan."""
    return np.array(
        [
            [np.cos(chi_rad), np.sin(chi_rad)],
            [-np.sin(chi_rad), np.cos(chi_rad)],
        ],
        dtype=np.complex128,
    )


def _antenna_id(instrument: SolverInstrumentView, row: int) -> AntennaId:
    return AntennaId(instrument.antenna_numbers[row], instrument.antenna_names[row])


# ---------------------------------------------------------------------------
# S13: the composition rule itself
# ---------------------------------------------------------------------------


def test_chain_composes_the_first_added_term_leftmost() -> None:
    """``JonesChain`` composes ``terms[0] @ ... @ terms[-1]`` (S13)."""
    backend = get_backend("numpy")
    chain = JonesChain(backend)
    chain.add_term(_ConstantJones("first", FIRST))
    chain.add_term(_ConstantJones("second", SECOND))

    forward = FIRST @ SECOND
    reversed_product = SECOND @ FIRST
    assert not np.allclose(forward, reversed_product)

    np.testing.assert_allclose(
        np.asarray(chain.compute_antenna_jones(0, None, 0, 0)),
        forward,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(chain.compute_antenna_jones_all_sources(0, 3, 0, 0))[2],
        forward,
        rtol=0.0,
        atol=0.0,
    )


def test_chain_docstring_states_the_canonical_section_19_1_order() -> None:
    """The documented order is the order the solver composes."""
    docstring = JonesChain.__doc__ or ""
    assert "J_total = H @ G @ B @ D @ P @ C @ E @ T @ Z" in docstring
    assert "terms[0] @ terms[1] @ ... @ terms[-1]" in docstring
    # The stale baseline factorization must not survive anywhere.
    assert "B @ G @ D @ P @ E @ T @ Z @ K" not in docstring


# ---------------------------------------------------------------------------
# S13: the point solver's add order
# ---------------------------------------------------------------------------


def test_point_solver_adds_every_term_in_the_canonical_order(tmp_path: Path) -> None:
    """Every optional term enabled reproduces the Section 19.1 order exactly."""
    chain = _chain(
        tmp_path,
        {
            "Z": {"enabled": True},
            "T": {"enabled": True},
            "P": {"enabled": True},
            "D": {"enabled": True},
            "G": {"enabled": True},
            "B": {"enabled": True},
        },
    )
    assert tuple(term.name for term in chain.terms) == CANONICAL_ORDER


def test_point_solver_always_carries_the_receptor_terms(tmp_path: Path) -> None:
    """``C`` and ``H`` are always present, exactly as ``E`` always is."""
    chain = _chain(tmp_path, {})
    assert tuple(term.name for term in chain.terms) == ("H", "C", "E")


def test_receptor_terms_keep_their_canonical_neighbours(tmp_path: Path) -> None:
    """``C`` sits between the electronics-side DIEs and the sky-side DDEs."""
    chain = _chain(
        tmp_path,
        {"P": {"enabled": True}, "G": {"enabled": True}, "T": {"enabled": True}},
    )
    names = [term.name for term in chain.terms]
    assert names.index("H") == 0
    assert names.index("G") < names.index("P") < names.index("C")
    assert names.index("C") < names.index("E") < names.index("T")


# ---------------------------------------------------------------------------
# S13 at the solver: a chain whose factors genuinely do not commute
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rotation_deg", [0.0, 37.0])
def test_composed_chain_equals_h_times_c_times_e(
    tmp_path: Path,
    rotation_deg: float,
) -> None:
    """The composed antenna Jones is exactly ``H @ C @ E``, not any permutation.

    A circular receptor in a linear output basis makes ``H`` and ``C`` two
    different non-commuting matrices, so this pins the product order and not
    merely the term inventory.
    """
    instrument, beam_system, receptor_set = _solver_components(
        tmp_path,
        {
            "default": {"basis": "circular", "feed_rotation_deg": rotation_deg},
            "output_basis": "linear",
        },
    )
    backend = get_backend("numpy")
    n_sources = 2
    altitude = np.full(n_sources, 1.0, dtype=np.float64)
    azimuth = np.full(n_sources, 0.5, dtype=np.float64)
    chain = _build_jones_chain(
        backend,
        {},
        instrument,
        altitude,
        azimuth,
        FREQUENCIES_HZ[0],
        0,
        n_sources,
        LOCATION,
        TIME_MJD,
        beam_system,
        receptor_set,
    )

    receptor = PLAN_S @ plan_rotation(np.deg2rad(rotation_deg))
    transform = PLAN_S.conj().T
    beam = np.asarray(
        beam_system.evaluate_jones(
            _antenna_id(instrument, 0),
            altitude_rad=altitude,
            azimuth_rad=azimuth,
            frequency_hz=float(FREQUENCIES_HZ[0]),
            time_mjd=TIME_MJD,
        )
    )

    composed = np.asarray(
        chain.compute_antenna_jones_all_sources(
            antenna_idx=0,
            n_sources=n_sources,
            freq_idx=0,
            time_idx=0,
            antenna_number=instrument.antenna_numbers[0],
        )
    )
    expected = transform @ receptor @ beam
    np.testing.assert_allclose(composed, expected, rtol=0.0, atol=1e-15)

    if rotation_deg != 0.0:
        # The reversed product is a genuinely different matrix, so the test
        # above is a statement about order and not about commuting factors.
        assert not np.allclose(expected, receptor @ transform @ beam)


# ---------------------------------------------------------------------------
# Rejections that only become reachable once the chain carries receptors
# ---------------------------------------------------------------------------


def test_parallactic_term_with_a_rotated_receptor_is_rejected(tmp_path: Path) -> None:
    """Section 12.3: ``P`` plus a non-zero rotation is a Tier 7 configuration."""
    with pytest.raises(UnsupportedFeedGeometryError) as excinfo:
        _chain(
            tmp_path,
            {"P": {"enabled": True}},
            {"default": {"feed_rotation_deg": 15.0}},
        )

    assert str(excinfo.value) == (
        "a non-zero feed_rotation_deg cannot be combined with an enabled "
        "parallactic-angle term until Tier 7 implements it."
    )


def test_parallactic_term_is_accepted_without_a_rotation(tmp_path: Path) -> None:
    """The rejection is about the rotation, not about ``P`` itself."""
    chain = _chain(tmp_path, {"P": {"enabled": True}})
    assert "P" in [term.name for term in chain.terms]


def test_resolved_receptors_are_required_by_the_chain_builder() -> None:
    """``receptors`` has no default: no caller can silently skip the terms."""
    import inspect

    parameters = inspect.signature(_build_jones_chain).parameters
    assert "receptors" in parameters
    assert parameters["receptors"].default is inspect.Parameter.empty
