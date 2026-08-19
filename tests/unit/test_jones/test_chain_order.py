"""Tier 5D and Tier 7F: the canonical Jones chain composition order.

``Tier7JonesSciencePlan.md`` Section 12.2 fixes the factorization, leftmost
nearest the correlator::

    J_p = H_p G_p B_p Rc_p Kd_p X_p D_p C_p E_p P_p T_p Z_p   (K separate)

:class:`~radiosim.core.jones.chain.JonesChain` composes
``terms[0] @ terms[1] @ ... @ terms[-1]``, so that factorization is exactly the
order in which the solver must *add* the terms.  The order is proven here with
deliberately non-commuting synthetic terms (invariant S13) rather than inferred
from a run whose selected factors happen to commute.

CORRECTED BY: Tier 7F.  ``Tier5ReceptorFeedPlan.md`` Section 19.1 placed ``P``
*correlator-side* of ``C``, and said -- correctly for its own scope -- that the
placement was unobservable while every optional term was an identity.  Section
12.1 shows it is wrong for a circular receptor: the physical composite is
``M(circular) R(chi_p + alpha_p) = C R(alpha_p)``, so ``R(alpha_p)`` must sit
sky-side of ``C``, where ``alpha_p=eta_p psi_p+nasmyth_p el``. Under the Tier 5
order the composite applies a real 2x2 rotation to the
``(R, L)`` pair, when the correct effect is a pair of opposite phases.  SCI-006
also makes the native linear matrix ``P`` rather than ``I2``; ``P`` does not
commute with a general rotation, so both native bases now distinguish the two
placements.  Invariant **I6** below is the test that separates them.
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
PLAN_P = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

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
    """Defect D7: ``add_term`` enforces what its docstring always claimed.

    ANCHOR UPDATED BY: Tier 7H.  ``BaselineMultiplicativeJones`` no longer takes
    zero arguments -- it takes the baselines it was resolved against and one
    matrix each -- so the probe is a real term rather than an empty one.  What
    is asserted is unchanged.
    """
    import numpy as np

    import radiosim.core.jones as jones_package

    chain = JonesChain(get_backend("numpy"))
    closure_error = jones_package.BaselineMultiplicativeJones(
        baseline_pairs=((0, 1),),
        matrices=np.full((1, 2, 2), 1.5, dtype=np.complex128),
    )
    with pytest.raises(TypeError) as excinfo:
        chain.add_term(closure_error)
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


def test_point_solver_with_empty_optional_inventory_carries_owned_h_c_e(
    tmp_path: Path,
) -> None:
    """The current empty optional-term inventory composes ``H``, ``C``, ``E``.

    FLIPPED BY: Tier 7C.  The gate version enabled six optional terms through a
    ``jones_config`` dictionary and asserted the full ``CANONICAL_ORDER``.  All
    six of those terms multiplied by the identity, the dictionary was hard-coded
    to ``None`` at the only production call site, and 7C removed both.  All
    optional terms are now implemented and are spliced into their canonical
    slots when configured.  This test deliberately supplies the empty optional
    inventory and pins only the three factors the solver always owns.
    """
    chain = _chain(tmp_path)
    assert tuple(term.name for term in chain.terms) == ("H", "C", "E")
    assert {term.name for term in chain.terms} <= set(CANONICAL_ORDER)


def test_empty_optional_chain_respects_the_owned_canonical_slots(
    tmp_path: Path,
) -> None:
    """The three solver-owned factors sit in canonical relative positions.

    ``C`` between the electronics-side DIEs and the sky-side DDEs, ``H``
    leftmost, ``E`` sky-side of ``C``: configured ``G``, ``P``, ``T``, ``Z``
    and the other optional terms are spliced around these slots.  Asserted
    against ``CANONICAL_ORDER`` itself so that adding a term out of order fails
    here.
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
    transform = PLAN_P @ PLAN_S.conj().T
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

    FLIPPED BY: Tier 7F, which deleted the guard outright and placed the now-real
    ``P`` sky-side of ``C``; SCI-006 then fixed the native linear matrix.  The
    current composition is ``C_p P_p=M(basis)R(chi_p+alpha_p)``, with
    ``alpha_p=eta_p psi_p+nasmyth_p el``.  The blanket mount-type rejection
    beside the old guard is now rejection R15, which names the fix rather than
    the tier.
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

    For this ordinary alt-az case ``alpha=psi``. ``H C E P`` with static
    ``chi`` must equal
    ``T(circular -> linear) M(circular) R(chi + psi) E``. Under the Tier 5
    correlator-side placement the product would instead be
    ``T(circular -> linear) R(psi) M(circular) R(chi) E``. The reversed
    placement is built explicitly below and differs.
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
    transform = PLAN_P @ PLAN_S.conj().T
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


# ---------------------------------------------------------------------------
# SCI-005 Stage 2: the replacement order oracle, with a non-scalar E
# ---------------------------------------------------------------------------
#
# ``docs/development/sci005_beam_physics_plan.md`` Section 4.2 retires the
# scalar-only order oracle outright:
#
#     "Stage 2 must replace the scalar-only order-unobservability oracle with
#     an analytic non-commuting case: choose unequal finite ``b0`` and ``b1``, a
#     nontrivial unitary ``C`` built on a **rotated linear** receptor, and a
#     nontrivial ``P``; prove ``C E P`` equals ``D_b C P`` and differs from
#     ``C P E``. The scalar disabled case remains a separate byte-identity
#     regression."
#
# The two halves live in different files by design. The ``C E P == D_b C P``
# statement is an ``E``-composition property and is asserted against an
# independently built ``D_b`` in
# ``tests/unit/test_core/test_sci005_beam_squint.py``. What belongs *here* is
# the chain-order half: with squint enabled on a rotated linear receptor the
# solver's own ``E`` no longer commutes with ``C``, so ``H @ C @ E`` becomes an
# observable claim about the order the solver composes rather than a claim that
# happens to hold because every factor is a scalar multiple of the identity.
#
# The receptor really has to be linear. Section 4.2 rules that for any circular
# receptor ``C = S R(chi)``,
#
#     E = C^dagger diag(b0, b1) C
#       = ((b0 + b1)/2) I2 - ((b0 - b1)/2) sigma_y,
#
# independent of ``chi``, with exactly equal diagonals and exact commutation
# with every real rotation -- so a circular fixture could not express an
# order-matters control at all. Its exact vanishing is a retained witness in
# its own right, and the companion assertion at the end of the test below keeps
# that witness next to the oracle it explains.
#
# Every existing test above is untouched. ``test_composed_chain_equals_h_
# times_c_times_e`` remains exactly the disabled-case regression Section 4.2
# asks to keep: same fixture, no squint block, scalar ``E``.


def _squint_block(positive_native_feed: str) -> dict[str, Any]:
    """The Stage-2 squint block, in the frozen Section 4.1 field spelling.

    The shipped layout carries no mount type, so both antennas resolve to
    ``fixed`` and the boresight the adapter derives is well defined at zenith
    (Section 4.2.1 rules only the *rotating* mount undefined there).  The feed
    label is a parameter because Section 4.1.1 requires it to belong to the
    resolved receptor basis: ``x``/``y`` for ``linear``, ``r``/``l`` for
    ``circular``.
    """
    return {
        "default": {
            "convention": "cotton_uson_exact_v1",
            "reference_frequency_hz": 1.5e8,
            "per_feed_offset_deg_at_reference": 2.0,
            "mechanical_feed_position_angle_deg": 35.0,
            "positive_native_feed": positive_native_feed,
        }
    }


def _squint_solver_components(
    tmp_path: Path,
    receptors: dict[str, object],
    *,
    positive_native_feed: str = "x",
) -> tuple[SolverInstrumentView, BeamSystem, ResolvedReceptorSet]:
    """The same solver pieces as :func:`_solver_components`, with squint on.

    Written separately rather than by widening ``_simulator`` so that no
    existing test's fixture changes.  ``tmp_path`` is created if absent:
    ``valid_config_mapping`` writes the antenna layout straight into it without
    creating parents, and a caller that wants a second system in its own
    subdirectory would otherwise have to remember.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    mapping = valid_config_mapping(
        tmp_path,
        frequency={
            "mode": "explicit",
            "channel_frequencies_hz": FREQUENCIES_HZ.tolist(),
            "channel_widths_hz": [1e6],
        },
        beams={
            "mode": "analytic",
            "model": {"kind": "circular_aperture", "taper": {"kind": "uniform"}},
            "squint": _squint_block(positive_native_feed),
        },
    )
    mapping["receptors"] = receptors
    simulator = Simulator.from_mapping(mapping, base_dir=tmp_path)
    simulator._ensure_instrument_state()
    simulator._ensure_receptor_set()
    simulator._ensure_beam_system()
    return (
        SolverInstrumentView.from_state(simulator._instrument_state),
        simulator.beam_system,
        simulator.receptors,
    )


def test_a_squint_enabled_chain_makes_the_h_c_e_order_observable(
    tmp_path: Path,
) -> None:
    """Section 4.2's replacement oracle: ``E`` no longer commutes with ``C``.

    A **rotated linear** receptor reported in a circular output basis gives
    three genuinely different matrices -- a non-identity ``H``, ``C = P
    R(chi)`` and a full ``E = C^dagger D_b C`` -- and the composed antenna
    Jones must be exactly ``H @ C @ E``. Two negative controls make that a
    statement about order: ``C @ E`` differs from ``E @ C``, so the middle pair
    does not commute, and the permuted products are different matrices.

    Section 4.2 requires the receptor to be linear here and says why in exact
    algebra: with ``C = P R(chi)`` the composed ``E`` is
    ``R(chi)^dagger diag(b1, b0) R(chi)``, whose diagonals differ by
    ``cos(2 chi) (b1 - b0)`` and whose off-diagonal is
    ``sin(2 chi) (b1 - b0) / 2`` -- both non-zero at ``chi = 31 deg`` with
    unequal samples. The circular receptor cannot express this control at all,
    which the companion assertion at the end of this test demonstrates rather
    than asserts on faith.

    ``E`` is read back from the chain's own ``E`` slot rather than recomputed,
    because what is under test here is the *composition* order; the physics of
    ``E`` itself is pinned in ``test_sci005_beam_squint.py``.
    """
    instrument, beam_system, receptor_set = _squint_solver_components(
        tmp_path,
        {
            "default": {"basis": "linear", "feed_rotation_deg": 31.0},
            "output_basis": "circular",
        },
    )
    backend = get_backend("numpy")
    n_sources = 3
    # Directions off the boresight, where the two displaced native samples are
    # unequal and ``E`` is therefore not a multiple of the identity.
    altitude = np.array([np.pi / 2.0 - 0.05, np.pi / 2.0 - 0.08, np.pi / 2.0 - 0.03])
    azimuth = np.array([0.4, 2.1, 4.0])

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
    assert [term.name for term in chain.terms] == ["H", "C", "E"]

    directions = DirectionBatch(
        alt_rad=altitude,
        az_rad=azimuth,
        dir_l=np.cos(altitude) * np.sin(azimuth),
        dir_m=np.cos(altitude) * np.cos(azimuth),
        dir_n=np.sin(altitude),
        ra_rad=np.zeros(n_sources),
        dec_rad=np.zeros(n_sources),
        hour_angle_rad=np.zeros(n_sources),
        n_dir=n_sources,
    )

    def _term(name: str) -> np.ndarray:
        term = chain.get_term(name)
        assert term is not None
        return np.asarray(
            term.compute_jones_batch(
                antenna_idx=0,
                directions=directions,
                frequency_hz=float(FREQUENCIES_HZ[0]),
                freq_idx=0,
                time_mjd=TIME_MJD,
                time_idx=0,
                backend=backend,
                dtype=np.complex128,
            )
        )

    transform = _term("H")
    receptor = _term("C")
    beam = _term("E")

    # The pre-Stage-2 situation this test replaces: a scalar ``E`` commutes
    # with everything, so the order could not be seen.  It is not scalar here.
    assert beam.shape == (n_sources, 2, 2)
    assert float(np.max(np.abs(beam[:, 0, 1]))) > 1e-3
    assert float(np.max(np.abs(beam[:, 0, 0] - beam[:, 1, 1]))) > 1e-3

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
    expected = transform @ receptor @ beam
    np.testing.assert_allclose(composed, expected, rtol=0.0, atol=1e-14)

    # Order controls.  ``C`` and ``E`` genuinely do not commute now, and the
    # permuted product is a different matrix.
    assert float(np.max(np.abs(receptor @ beam - beam @ receptor))) > 1e-3
    assert not np.allclose(expected, transform @ beam @ receptor, atol=1e-6)
    assert not np.allclose(expected, receptor @ transform @ beam, atol=1e-6)

    # Companion witness: the same squint on a *circular* receptor gives the
    # Section 4.2 identity instead -- exactly equal diagonals and exact
    # commutation with every real rotation -- which is why the oracle above
    # cannot be built on one.  ``E`` is still generally full there.
    _instrument, circular_beams, circular_receptors = _squint_solver_components(
        tmp_path / "circular",
        {
            "default": {"basis": "circular", "feed_rotation_deg": 17.0},
            "output_basis": "linear",
        },
        positive_native_feed="r",
    )
    circular_chain = _build_jones_chain(
        backend,
        instrument,
        altitude,
        azimuth,
        FREQUENCIES_HZ[0],
        0,
        n_sources,
        LOCATION,
        TIME_MJD,
        circular_beams,
        circular_receptors,
    )
    circular_e_term = circular_chain.get_term("E")
    assert circular_e_term is not None
    circular_beam = np.asarray(
        circular_e_term.compute_jones_batch(
            antenna_idx=0,
            directions=directions,
            frequency_hz=float(FREQUENCIES_HZ[0]),
            freq_idx=0,
            time_mjd=TIME_MJD,
            time_idx=0,
            backend=backend,
            dtype=np.complex128,
        )
    )
    rotation = plan_rotation(0.6)
    assert (
        float(np.max(np.abs(circular_beam[:, 0, 0] - circular_beam[:, 1, 1]))) <= 1e-14
    )
    assert (
        float(np.max(np.abs(circular_beam @ rotation - rotation @ circular_beam)))
        <= 1e-14
    )
    assert float(np.max(np.abs(circular_beam[:, 0, 1]))) > 1e-3


# ---------------------------------------------------------------------------
# SCI-005 Stage 3: the order oracle with a generally full efield ``E``
# ---------------------------------------------------------------------------
#
# ``docs/development/sci005_beam_physics_plan.md`` Section 5.6 requires "full
# Jones order tests with non-commuting ``C``, ``E``, and ``P``", and Section
# 8.1's ``receptor_factorizations`` row fixes the three measurements exactly:
#
#     ``factorization_max_abs_residual`` is the largest entrywise absolute
#     difference between the production ``C @ E`` and ``J_native``;
#     ``chain_order_max_abs_residual`` is the largest entrywise absolute
#     difference between ``C @ E @ P`` and ``J_native @ P``; and
#     ``order_control_max_abs_difference`` is the largest entrywise absolute
#     difference between ``C @ E @ P`` and ``C @ P @ E``.
#
# and requires ``noncommuting_component >= max(1e-3, 1024 * atol)`` on every
# row, recomputed from the composed ``E`` as
# ``abs(E[0][0] - E[1][1]) + abs(E[0][1] + E[1][0])``. Section 8.1 states why
# Stage 3 needs no per-basis split, unlike Stage 2: "Stage 2's
# circular-receptor vanishing does not recur here because it followed from
# ``D_b`` being diagonal with the circular ``C``, which a general
# ``J_native`` is not."
#
# ``P`` is written here as an explicit real rotation rather than taken from a
# ``jones:`` block, because the two products above are statements about
# matrices and the solver's own optional-term inventory is not what Section 8.1
# measures. Every existing test in this module is untouched.

STAGE3_NORMALIZATION = "uvbeam_peak_common_v1"

#: Section 5.2.1's dtype-derived converted-matrix ``atol`` at ``complex128``,
#: and Section 8.1's frozen separation bound built on it.
STAGE3_ATOL = max(1e-12, 32.0 * float(np.finfo(np.float64).eps))
STAGE3_SEPARATION_BOUND = max(1e-3, 1024.0 * STAGE3_ATOL)


def _efield_solver_components(
    tmp_path: Path,
    receptors: dict[str, object],
    *,
    feed_array: tuple[str, str] = ("x", "y"),
    feed_rotation_deg: float = 0.0,
) -> tuple[SolverInstrumentView, BeamSystem, ResolvedReceptorSet]:
    """The same solver pieces as :func:`_solver_components`, on a full-efield
    ``shared_fits`` document carrying Section 5.1.1's activation literal."""
    from tests.fixtures.beamfits import EfieldScienceVariant, write_efield_beamfits

    tmp_path.mkdir(parents=True, exist_ok=True)
    written = write_efield_beamfits(
        tmp_path,
        science=EfieldScienceVariant.QUADRUPOLAR,
        feed_array=feed_array,
        feed_rotation_rad=np.radians(feed_rotation_deg),
    )
    mapping = valid_config_mapping(
        tmp_path,
        frequency={
            "mode": "explicit",
            "channel_frequencies_hz": FREQUENCIES_HZ.tolist(),
            "channel_widths_hz": [1e6],
        },
        beams={
            "mode": "shared_fits",
            "beam": {
                "kind": "fits",
                "path": str(written.path),
                "normalization": STAGE3_NORMALIZATION,
            },
        },
    )
    mapping["receptors"] = receptors
    simulator = Simulator.from_mapping(mapping, base_dir=tmp_path)
    simulator._ensure_instrument_state()
    simulator._ensure_receptor_set()
    simulator._ensure_beam_system()
    return (
        SolverInstrumentView.from_state(simulator._instrument_state),
        simulator.beam_system,
        simulator.receptors,
    )


def _stage3_directions(n_sources: int) -> tuple[np.ndarray, np.ndarray, DirectionBatch]:
    altitude = np.pi / 2.0 - np.array([0.30, 0.55, 0.80][:n_sources])
    azimuth = np.array([0.4, 2.1, 4.0][:n_sources])
    return (
        altitude,
        azimuth,
        DirectionBatch(
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
    )


@pytest.mark.parametrize(
    ("receptor_basis", "feed_rotation_deg", "output_basis"),
    [
        ("linear", 0.0, "linear"),
        ("linear", 31.0, "circular"),
        ("circular", 0.0, "circular"),
        ("circular", 17.0, "linear"),
    ],
)
def test_a_full_efield_chain_makes_the_h_c_e_order_observable(
    tmp_path: Path,
    receptor_basis: str,
    feed_rotation_deg: float,
    output_basis: str,
) -> None:
    """Section 5.6's full order test, on all four
    ``(receptor_basis, output_basis)`` combinations Section 8.1 requires.

    A general ``J_native`` makes the composed ``E = C^dagger J_native``
    genuinely non-commuting for *every* receptor basis, which is exactly the
    property Stage 2's diagonal ``D_b`` did not have on a circular receptor.
    """
    feeds = ("x", "y") if receptor_basis == "linear" else ("r", "l")
    instrument, beam_system, receptor_set = _efield_solver_components(
        tmp_path,
        {
            "default": {
                "basis": receptor_basis,
                "feed_rotation_deg": feed_rotation_deg,
            },
            "output_basis": output_basis,
        },
        feed_array=feeds,
        feed_rotation_deg=feed_rotation_deg,
    )
    backend = get_backend("numpy")
    n_sources = 3
    altitude, azimuth, directions = _stage3_directions(n_sources)

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
    assert [term.name for term in chain.terms] == ["H", "C", "E"]

    def _term(name: str) -> np.ndarray:
        term = chain.get_term(name)
        assert term is not None
        return np.asarray(
            term.compute_jones_batch(
                antenna_idx=0,
                directions=directions,
                frequency_hz=float(FREQUENCIES_HZ[0]),
                freq_idx=0,
                time_mjd=TIME_MJD,
                time_idx=0,
                backend=backend,
                dtype=np.complex128,
            )
        )

    transform = _term("H")
    receptor = _term("C")
    beam = _term("E")

    assert beam.shape == (n_sources, 2, 2)
    # Section 8.1's exact algebraic non-commutation condition, recomputed from
    # the composed ``E`` itself.
    noncommuting = np.abs(beam[:, 0, 0] - beam[:, 1, 1]) + np.abs(
        beam[:, 0, 1] + beam[:, 1, 0]
    )
    assert float(np.max(noncommuting)) >= STAGE3_SEPARATION_BOUND

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
    np.testing.assert_allclose(
        composed,
        transform @ receptor @ beam,
        rtol=0.0,
        atol=1e-14,
    )

    # Section 8.1's three retained residuals, with an explicit real rotation
    # standing in for ``P``.
    native = receptor @ beam
    field_rotation = plan_rotation(0.61)
    chain_order_residual = float(
        np.max(np.abs(receptor @ beam @ field_rotation - native @ field_rotation))
    )
    order_control = float(
        np.max(
            np.abs(receptor @ beam @ field_rotation - receptor @ field_rotation @ beam)
        )
    )
    assert chain_order_residual <= STAGE3_ATOL
    assert order_control >= STAGE3_SEPARATION_BOUND
    # ``C`` and ``E`` themselves do not commute, and the permuted products are
    # different matrices.
    assert float(np.max(np.abs(receptor @ beam - beam @ receptor))) >= (
        STAGE3_SEPARATION_BOUND
    )
    assert not np.allclose(composed, transform @ beam @ receptor, atol=1e-6)
    assert not np.allclose(composed, receptor @ transform @ beam, atol=1e-6)
