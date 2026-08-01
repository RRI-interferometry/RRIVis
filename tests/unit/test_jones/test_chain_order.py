"""Tier 5D and Tier 7F: the canonical Jones chain composition order.

``Tier7JonesSciencePlan.md`` Section 12.2 fixes the factorization, leftmost
nearest the correlator::

    J_p = H_p G_p B_p Rc_p Kd_p X_p D_p C_p E_p P_p T_p Z_p   (K separate)

:class:`~radiosim.core.jones.chain.JonesChain` composes
``terms[0] @ terms[1] @ ... @ terms[-1]``, so that factorization is exactly the
order in which the solver must *add* the terms.  ``C`` and ``H`` are the first
non-commuting factors RadioSim composes, which is why the order is proven here
with deliberately non-commuting synthetic terms (invariant S13) rather than
inferred from a run whose factors all commute.

CORRECTED BY: Tier 7F.  ``Tier5ReceptorFeedPlan.md`` Section 19.1 placed ``P``
*correlator-side* of ``C``, and said -- correctly for its own scope -- that the
placement was unobservable while every optional term was an identity.  Section
12.1 shows it is wrong for a circular receptor: the physical composite is
``M(circular) R(chi + psi) = C R(psi)``, so ``R(psi)`` must sit sky-side of
``C``.  Under the Tier 5 order the composite applies a real 2x2 rotation to the
``(R, L)`` pair, when the correct effect is a pair of opposite phases.  The two
agree only for a linear receptor, where ``M = I2`` and rotations commute --
which is exactly the case Tier 5 tested.  Invariant **I6** below is the test
that separates them.
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
from radiosim.core.jones import DirectionBatch, JonesChain, JonesTerm
from radiosim.core.receptor import ResolvedReceptorSet
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

CANONICAL_ORDER = (
    "H",
    "G",
    "B",
    "Rc",
    "Kd",
    "X",
    "D",
    "C",
    "E",
    "P",
    "T",
    "Z",
)


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

    def compute_jones_batch(
        self,
        *,
        antenna_idx: int,
        directions: DirectionBatch,
        frequency_hz: float,
        freq_idx: int,
        time_mjd: float,
        time_idx: int,
        backend: Any,
        dtype: Any,
    ) -> Any:
        return backend.xp.array(self._matrix[None, :, :], dtype=dtype)


def _directions(n_dir: int) -> DirectionBatch:
    """A direction batch for a chain whose terms may or may not use it."""
    values = np.linspace(0.1, 1.0, n_dir)
    return DirectionBatch(
        alt_rad=np.full(n_dir, 1.0),
        az_rad=np.full(n_dir, 0.5),
        dir_l=values,
        dir_m=values,
        dir_n=values,
        ra_rad=values,
        dec_rad=values,
        hour_angle_rad=values,
        n_dir=n_dir,
    )


def _composed(chain: JonesChain, *, n_dir: int = 2, antenna_idx: int = 0) -> np.ndarray:
    """Evaluate a chain over a direction batch and return it as a host array."""
    return np.asarray(
        chain.compute_antenna_jones_batch(
            antenna_idx=antenna_idx,
            directions=_directions(n_dir),
            frequency_hz=float(FREQUENCIES_HZ[0]),
            freq_idx=0,
            time_mjd=TIME_MJD,
            time_idx=0,
            dtype=np.complex128,
        )
    )


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
    receptors: dict[str, object] | None = None,
    *,
    n_sources: int = 2,
) -> JonesChain:
    instrument, beam_system, receptor_set = _solver_components(tmp_path, receptors)
    return _build_jones_chain(
        get_backend("numpy"),
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

    # A chain of purely direction-independent terms stays (1, 2, 2) and
    # broadcasts once, at the end, instead of carrying n_dir identical copies.
    composed = _composed(chain, n_dir=3)
    assert composed.shape == (1, 2, 2)
    np.testing.assert_allclose(composed[0], forward, rtol=0.0, atol=0.0)


def test_chain_composes_the_full_twelve_term_canonical_order() -> None:
    """Invariant I5, over every term the designed chain can carry.

    Twelve deliberately non-commuting synthetic terms, added in the documented
    order, must compose to exactly ``terms[0] @ ... @ terms[-1]``.  Each factor
    is a distinct shear or rotation, so any transposition of any adjacent pair
    changes the product -- the test is about the order and not about the term
    inventory.
    """
    designed_order = ("H", "G", "B", "Rc", "Kd", "X", "D", "C", "E", "P", "T", "Z")
    matrices = [
        np.array(
            [
                [1.0 + 0.05 * index, 0.3 + 0.1 * index],
                [0.2j * (index + 1), 1.0 + 0.07j * index],
            ],
            dtype=np.complex128,
        )
        for index in range(len(designed_order))
    ]

    backend = get_backend("numpy")
    chain = JonesChain(backend)
    for label, matrix in zip(designed_order, matrices, strict=True):
        chain.add_term(_ConstantJones(label, matrix))

    # Right-to-left, because that is the order the chain applies its factors in
    # (sky-side first).  Matrix multiplication is associative in exact
    # arithmetic but not in floating point, so an oracle that bracketed the
    # product the other way would differ in the last bits and could only be
    # asserted to a tolerance.
    expected = matrices[-1]
    for matrix in reversed(matrices[:-1]):
        expected = matrix @ expected

    composed = _composed(chain)
    assert tuple(term.name for term in chain.terms) == designed_order
    np.testing.assert_allclose(composed[0], expected, rtol=0.0, atol=0.0)

    # Every adjacent transposition really does change the product, so the
    # assertion above is a statement about order and not about commuting
    # factors.  The bracketing here is irrelevant at this tolerance.
    for index in range(len(matrices) - 1):
        swapped = list(matrices)
        swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
        product = swapped[0]
        for matrix in swapped[1:]:
            product = product @ matrix
        assert not np.allclose(product, expected)


def test_chain_rejects_a_baseline_dependent_term() -> None:
    """Defect D7: ``add_term`` enforces what its docstring always claimed."""
    import radiosim.core.jones as jones_package

    chain = JonesChain(get_backend("numpy"))
    with pytest.raises(TypeError) as excinfo:
        chain.add_term(jones_package.BaselineMultiplicativeJones())
    assert "JonesBaselineTerm" in str(excinfo.value)
    assert chain.terms == []

    with pytest.raises(TypeError):
        chain.add_term(object())  # type: ignore[arg-type]


def test_chain_docstring_states_the_canonical_section_19_1_order() -> None:
    """The documented order is the order the solver composes."""
    docstring = JonesChain.__doc__ or ""
    assert "J_total = H @ G @ B @ Rc @ Kd @ X @ D @ C @ E @ P @ T @ Z" in docstring
    assert "terms[0] @ terms[1] @ ... @ terms[-1]" in docstring
    # Neither the pre-Tier-5 factorization nor the superseded Tier 5 one may
    # survive as a statement of what the chain *is*.
    assert "B @ G @ D @ P @ E @ T @ Z @ K" not in docstring
    assert "J_total = H @ G @ B @ D @ P @ C @ E @ T @ Z" not in docstring


# ---------------------------------------------------------------------------
# S13: the point solver's add order
# ---------------------------------------------------------------------------


def test_point_solver_carries_exactly_the_three_terms_that_exist(
    tmp_path: Path,
) -> None:
    """``H``, ``C`` and ``E``, and nothing else.

    FLIPPED BY: Tier 7C.  The gate version enabled six optional terms through a
    ``jones_config`` dictionary and asserted the full ``CANONICAL_ORDER``.  All
    six of those terms multiplied by the identity, the dictionary was hard-coded
    to ``None`` at the only production call site, and 7C removed both.  The
    order they will occupy is still the plan's, and it is still asserted -- from
    the solver's own source, below, and by the synthetic non-commuting terms
    above -- but the chain now contains only terms that exist.
    """
    chain = _chain(tmp_path)
    assert tuple(term.name for term in chain.terms) == ("H", "C", "E")
    assert {term.name for term in chain.terms} <= set(CANONICAL_ORDER)


def test_the_chain_builder_reserves_the_canonical_slots_in_order(
    tmp_path: Path,
) -> None:
    """The three surviving terms sit in their canonical relative positions.

    ``C`` between the electronics-side DIEs and the sky-side DDEs, ``H``
    leftmost, ``E`` sky-side of ``C``: the neighbours a later slice's ``G``,
    ``P``, ``T`` and ``Z`` must respect.  Asserted against ``CANONICAL_ORDER``
    itself so that adding a term out of order fails here.
    """
    chain = _chain(tmp_path)
    names = [term.name for term in chain.terms]
    positions = [CANONICAL_ORDER.index(name) for name in names]
    assert positions == sorted(positions)
    assert names.index("H") == 0
    assert names.index("C") < names.index("E")


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
        chain.compute_antenna_jones_batch(
            antenna_idx=0,
            directions=DirectionBatch(
                alt_rad=altitude,
                az_rad=azimuth,
                dir_l=np.cos(altitude) * np.sin(azimuth),
                dir_m=np.cos(altitude) * np.cos(azimuth),
                dir_n=np.sin(altitude),
                ra_rad=np.zeros(n_sources),
                dec_rad=np.zeros(n_sources),
                hour_angle_rad=np.zeros(n_sources),
                n_dir=n_sources,
            ),
            frequency_hz=float(FREQUENCIES_HZ[0]),
            freq_idx=0,
            time_mjd=TIME_MJD,
            time_idx=0,
            dtype=np.complex128,
        )
    )
    expected = transform @ receptor @ beam
    np.testing.assert_allclose(composed, expected, rtol=0.0, atol=1e-15)

    if rotation_deg != 0.0:
        # The reversed product is a genuinely different matrix, so the test
        # above is a statement about order and not about commuting factors.
        assert not np.allclose(expected, receptor @ transform @ beam)


# ---------------------------------------------------------------------------
# The chain builder's remaining contract
# ---------------------------------------------------------------------------


def test_a_rotated_receptor_is_carried_by_the_chain_without_a_rejection(
    tmp_path: Path,
) -> None:
    """Section 12.3's rejection had exactly one trigger, and 7C removed it.

    FLIPPED BY: Tier 7C.  ``_reject_parallactic_rotation`` fired only when a
    ``jones_config`` enabled ``P``; with the dictionary gone the combination it
    guarded could not be expressed through any entry point.

    FLIPPED BY: Tier 7F, which deleted the guard outright.  ``P`` is real and
    sits sky-side of ``C``, so ``C_p P_p = M(basis) R(chi + psi)`` is the full,
    correct, time-dependent receptor orientation rather than a static rotation
    silently composed with a stub -- which is exactly what
    ``Tier5ReceptorFeedPlan.md`` Section 12.3 said would happen "when Tier 7
    implements ``P``".  The blanket mount-type rejection beside it is now
    rejection R15, which names the fix rather than the tier.
    """
    chain = _chain(tmp_path, {"default": {"feed_rotation_deg": 15.0}})
    assert tuple(term.name for term in chain.terms) == ("H", "C", "E")


def test_resolved_receptors_are_required_by_the_chain_builder() -> None:
    """``receptors`` has no default: no caller can silently skip the terms."""
    import inspect

    parameters = inspect.signature(_build_jones_chain).parameters
    assert "receptors" in parameters
    assert parameters["receptors"].default is inspect.Parameter.empty


# ---------------------------------------------------------------------------
# I6: the corrected placement of P, at the solver's own chain builder
# ---------------------------------------------------------------------------


def _chain_with_parallactic(
    tmp_path: Path,
    receptors: dict[str, object] | None,
    *,
    n_sources: int = 2,
    mount_types: Any = "alt-az",
):
    """Return the solver's chain with a resolved ``P`` spliced into it."""
    from tests.unit.test_core.test_jones_resolution import solver_components_with_jones

    overrides: dict[str, Any] = {
        "frequency": {
            "mode": "explicit",
            "channel_frequencies_hz": FREQUENCIES_HZ.tolist(),
            "channel_widths_hz": [1e6],
        }
    }
    if receptors is not None:
        overrides["receptors"] = receptors
    instrument, beam_system, receptor_set, jones_terms, _frequencies = (
        solver_components_with_jones(
            tmp_path,
            {"P": {"enabled": True}},
            mount_types=mount_types,
            **overrides,
        )
    )
    chain = _build_jones_chain(
        get_backend("numpy"),
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
        jones_terms=jones_terms,
    )
    return chain, instrument, beam_system, receptor_set


def test_the_solver_places_p_sky_side_of_c_and_e(tmp_path: Path) -> None:
    """The chain the solver actually builds carries the corrected order."""
    chain, _instrument, _beams, _receptors = _chain_with_parallactic(tmp_path, None)

    names = [term.name for term in chain.terms]
    assert names == ["H", "C", "E", "P"]
    positions = [CANONICAL_ORDER.index(name) for name in names]
    assert positions == sorted(positions)


def test_the_composed_chain_is_the_receptor_at_the_combined_angle(
    tmp_path: Path,
) -> None:
    """Invariant I6, through the solver, for a circular receptor.

    ``H C E P`` with a static ``chi`` and a field rotation ``psi`` must equal
    ``T(circular -> linear) M(circular) R(chi + psi) E``.  Under the Tier 5
    order the ``P`` factor would sit correlator-side of ``C`` and the product
    would be ``T M(circular) R(psi) R(chi)``... which is the *same* matrix only
    because ``M`` is a left factor there; the distinguishing statement is the
    one below, where the reversed placement is built explicitly and differs.
    """
    from radiosim.core.jones.parallactic import parallactic_angle

    chi_deg = 24.0
    receptors = {
        "default": {"basis": "circular", "feed_rotation_deg": chi_deg},
        "output_basis": "linear",
    }
    n_sources = 2
    altitude = np.full(n_sources, 1.0, dtype=np.float64)
    azimuth = np.full(n_sources, 0.5, dtype=np.float64)

    chain, instrument, beam_system, _receptor_set = _chain_with_parallactic(
        tmp_path, receptors, n_sources=n_sources
    )

    directions = DirectionBatch.from_horizontal(
        alt_rad=altitude,
        az_rad=azimuth,
        dir_l=np.cos(altitude) * np.sin(azimuth),
        dir_m=np.cos(altitude) * np.cos(azimuth),
        dir_n=np.sin(altitude),
        latitude_rad=float(LOCATION.lat.rad),
        local_sidereal_time_rad=0.0,
    )
    composed = np.asarray(
        chain.compute_antenna_jones_batch(
            antenna_idx=0,
            directions=directions,
            frequency_hz=float(FREQUENCIES_HZ[0]),
            freq_idx=0,
            time_mjd=TIME_MJD,
            time_idx=0,
            dtype=np.complex128,
        )
    )

    psi = parallactic_angle(
        hour_angle_rad=directions.hour_angle_rad,
        dec_rad=directions.dec_rad,
        latitude_rad=float(LOCATION.lat.rad),
    )
    beam = np.asarray(
        beam_system.evaluate_jones(
            _antenna_id(instrument, 0),
            altitude_rad=altitude,
            azimuth_rad=azimuth,
            frequency_hz=float(FREQUENCIES_HZ[0]),
            time_mjd=TIME_MJD,
        )
    )
    transform = PLAN_S.conj().T
    chi = np.deg2rad(chi_deg)

    combined = np.stack(
        [
            transform @ PLAN_S @ plan_rotation(chi + float(angle)) @ beam[index]
            for index, angle in enumerate(psi)
        ]
    )
    np.testing.assert_allclose(composed, combined, rtol=0.0, atol=1e-14)

    # And the superseded order really is a different matrix: ``P`` between
    # ``D`` and ``C`` would multiply the (R, L) pair by a real rotation.
    reversed_placement = np.stack(
        [
            transform
            @ plan_rotation(float(angle))
            @ PLAN_S
            @ plan_rotation(chi)
            @ beam[index]
            for index, angle in enumerate(psi)
        ]
    )
    assert not np.allclose(combined, reversed_placement, atol=1e-6)


def test_a_field_rotation_is_two_phases_on_a_circular_receptor(
    tmp_path: Path,
) -> None:
    """The physical statement the correction exists for.

    Reported in the receptor's own ``circular_rl`` basis, the composite
    ``H C P (C)^-1`` is ``diag(e^{-i psi}, e^{+i psi})``: a field rotation
    phases ``R`` and ``L`` oppositely and does not mix them.  A real 2x2
    rotation of the ``(R, L)`` pair -- which is what the Tier 5 order produces
    -- would have a non-zero off-diagonal, and the assertion below is exactly
    that it does not.
    """
    from radiosim.core.jones.parallactic import parallactic_angle

    receptors = {
        "default": {"basis": "circular", "feed_rotation_deg": 0.0},
        "output_basis": "circular",
    }
    n_sources = 2
    altitude = np.full(n_sources, 1.0, dtype=np.float64)
    azimuth = np.full(n_sources, 0.5, dtype=np.float64)

    chain, _instrument, _beams, _receptor_set = _chain_with_parallactic(
        tmp_path, receptors, n_sources=n_sources
    )
    directions = DirectionBatch.from_horizontal(
        alt_rad=altitude,
        az_rad=azimuth,
        dir_l=np.cos(altitude) * np.sin(azimuth),
        dir_m=np.cos(altitude) * np.cos(azimuth),
        dir_n=np.sin(altitude),
        latitude_rad=float(LOCATION.lat.rad),
        local_sidereal_time_rad=0.0,
    )
    psi = parallactic_angle(
        hour_angle_rad=directions.hour_angle_rad,
        dec_rad=directions.dec_rad,
        latitude_rad=float(LOCATION.lat.rad),
    )

    receptor_term = chain.get_term("C")
    parallactic_term = chain.get_term("P")
    assert receptor_term is not None and parallactic_term is not None

    receptor_matrix = np.asarray(
        receptor_term.compute_jones_batch(
            antenna_idx=0,
            directions=directions,
            frequency_hz=float(FREQUENCIES_HZ[0]),
            freq_idx=0,
            time_mjd=TIME_MJD,
            time_idx=0,
            backend=get_backend("numpy"),
            dtype=np.complex128,
        )
    )[0]
    rotation = np.asarray(
        parallactic_term.compute_jones_batch(
            antenna_idx=0,
            directions=directions,
            frequency_hz=float(FREQUENCIES_HZ[0]),
            freq_idx=0,
            time_mjd=TIME_MJD,
            time_idx=0,
            backend=get_backend("numpy"),
            dtype=np.complex128,
        )
    )

    for index, angle in enumerate(psi):
        composite = receptor_matrix @ rotation[index] @ np.linalg.inv(receptor_matrix)
        expected = np.diag(
            np.array(
                [np.exp(-1j * float(angle)), np.exp(1j * float(angle))],
                dtype=np.complex128,
            )
        )
        np.testing.assert_allclose(composite, expected, rtol=0.0, atol=1e-14)
