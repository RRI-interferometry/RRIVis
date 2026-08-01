"""Tier 7B: the direction-batched term contract and the flag-verification harness.

Owns three invariants from ``Tier7JonesSciencePlan.md`` Section 27:

* **I3** -- shape.  Every direction-independent term returns exactly
  ``(1, 2, 2)``; every direction-dependent term returns ``(n_dir, 2, 2)``.
* **I17** -- precision.  Every term returns the dtype it was handed, never a
  dtype of its own choosing.
* **I2** -- declared flags are true.  Every ``is_diagonal`` / ``is_scalar`` /
  ``is_unitary`` that a term declares ``True`` is verified numerically over a
  parameter sweep, and every ``False`` needs a witness -- a swept parameter where
  the property genuinely fails.  That converse is the part that matters: defect
  D10 was terms declaring unitarity and scalarity about a matrix that was the
  2x2 identity, which is trivially both, so a vacuous ``True`` must be
  impossible to reintroduce.

It also owns D7: ``JonesChain.add_term`` rejecting a ``JonesBaselineTerm``.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pytest

from radiosim.backends import get_backend
from radiosim.core.instrument_adapters import SolverInstrumentView
from radiosim.core.jones import DirectionBatch, JonesChain, JonesTerm
from radiosim.core.jones.baseline_errors import JonesBaselineTerm
from radiosim.core.jones.receptor import BasisTransformJones, ReceptorConfigJones
from radiosim.core.receptor import ResolvedReceptorSet
from tests.unit.test_jones.test_receptor import make_instrument_view, make_receptor_set

BACKEND = get_backend("numpy")
N_DIR = 4

BASES = ("linear", "circular")
ROTATIONS_DEG = (0.0, 17.0, -63.0, 90.0)
OUTPUT_BASES = ("linear_xy", "circular_rl")


# ---------------------------------------------------------------------------
# Fixtures: one instrument view and one receptor set per swept parameter
# ---------------------------------------------------------------------------


def _directions(n_dir: int = N_DIR) -> DirectionBatch:
    values = np.linspace(0.2, 1.2, n_dir)
    return DirectionBatch(
        alt_rad=values,
        az_rad=values / 2.0,
        dir_l=np.cos(values) * np.sin(values / 2.0),
        dir_m=np.cos(values) * np.cos(values / 2.0),
        dir_n=np.sin(values),
        ra_rad=values,
        dec_rad=-values,
        hour_angle_rad=values / 3.0,
        n_dir=n_dir,
    )


def _instrument_view(count: int = 2) -> SolverInstrumentView:
    """Reuses the Tier 5 receptor-test fixtures rather than re-deriving them."""
    return make_instrument_view(count)


def _receptor_set(
    per_antenna: tuple[tuple[str, float], ...],
    output_basis: str,
) -> ResolvedReceptorSet:
    return make_receptor_set(per_antenna, output_basis)


def _swept_terms() -> list[tuple[str, JonesTerm]]:
    """Every implemented ``JonesTerm``, over its whole parameter space.

    ``C`` and ``H`` are the two terms the chain always carries at 7B, and their
    parameter space -- receptor basis, feed rotation, reporting basis -- is
    small enough to enumerate exhaustively rather than sample.
    """
    view = _instrument_view(2)
    terms: list[tuple[str, JonesTerm]] = []
    for basis, rotation, output_basis in itertools.product(
        BASES, ROTATIONS_DEG, OUTPUT_BASES
    ):
        receptors = _receptor_set(((basis, rotation), (basis, rotation)), output_basis)
        label = f"{basis}/chi={rotation:+.1f}deg/{output_basis}"
        terms.append(
            (
                f"C[{label}]",
                ReceptorConfigJones(receptors=receptors, instrument=view),
            )
        )
        terms.append(
            (
                f"H[{label}]",
                BasisTransformJones(receptors=receptors, instrument=view),
            )
        )
    return terms


def _evaluate(term: JonesTerm, *, dtype: Any = np.complex128) -> np.ndarray:
    return np.asarray(
        term.compute_jones_batch(
            antenna_idx=0,
            directions=_directions(),
            frequency_hz=1.5e8,
            freq_idx=0,
            time_mjd=60_000.0,
            time_idx=0,
            backend=BACKEND,
            dtype=dtype,
        )
    )


# ---------------------------------------------------------------------------
# I3 -- shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,term", _swept_terms(), ids=lambda value: str(value)[:40]
)
def test_a_direction_independent_term_returns_exactly_one_matrix(
    label: str,
    term: JonesTerm,
) -> None:
    """I3: a DIE term broadcasts; it never materialises ``n_dir`` copies."""
    assert term.is_direction_dependent is False
    assert _evaluate(term).shape == (1, 2, 2)


def test_a_direction_dependent_term_returns_one_matrix_per_direction(tmp_path) -> None:
    """I3, the other half, on the one direction-dependent term 7B has: ``E``."""
    import radiosim.core.visibility as visibility_module
    from radiosim.api import Simulator
    from tests.fixtures.configs import valid_config_mapping

    simulator = Simulator.from_mapping(
        valid_config_mapping(tmp_path),
        base_dir=tmp_path,
    )
    simulator._ensure_instrument_state()
    simulator._ensure_beam_system()
    instrument = SolverInstrumentView.from_state(simulator._instrument_state)
    directions = _directions()
    term = visibility_module._ResolvedBeamJones(
        beam_system=simulator.beam_system,
        instrument=instrument,
        altitude_rad=directions.alt_rad,
        azimuth_rad=directions.az_rad,
        frequency_hz=1.0e8,
        time_mjd=60_000.0,
    )

    assert term.is_direction_dependent is True
    result = np.asarray(
        term.compute_jones_batch(
            antenna_idx=0,
            directions=directions,
            frequency_hz=1.0e8,
            freq_idx=0,
            time_mjd=60_000.0,
            time_idx=0,
            backend=BACKEND,
            dtype=np.complex128,
        )
    )
    assert result.shape == (N_DIR, 2, 2)


# ---------------------------------------------------------------------------
# I17 -- precision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize(
    "label,term", _swept_terms(), ids=lambda value: str(value)[:40]
)
def test_a_term_returns_the_dtype_it_was_handed(
    label: str,
    term: JonesTerm,
    dtype: Any,
) -> None:
    """I17: no term reaches for ``complex128`` when something else was resolved."""
    assert _evaluate(term, dtype=dtype).dtype == dtype


def test_the_chain_seed_honours_the_resolved_dtype() -> None:
    """Defect D8: the identity seed came from a literal, now from the caller."""
    chain = JonesChain(BACKEND)
    empty = np.asarray(
        chain.compute_antenna_jones_batch(
            antenna_idx=0,
            directions=_directions(),
            frequency_hz=1.5e8,
            freq_idx=0,
            time_mjd=60_000.0,
            time_idx=0,
            dtype=np.complex64,
        )
    )
    assert empty.shape == (1, 2, 2)
    assert empty.dtype == np.complex64

    view = _instrument_view(2)
    receptors = _receptor_set((("circular", 0.4), ("circular", 0.4)), "linear_xy")
    chain.add_term(BasisTransformJones(receptors=receptors, instrument=view))
    chain.add_term(ReceptorConfigJones(receptors=receptors, instrument=view))
    loaded = np.asarray(
        chain.compute_antenna_jones_batch(
            antenna_idx=0,
            directions=_directions(),
            frequency_hz=1.5e8,
            freq_idx=0,
            time_mjd=60_000.0,
            time_idx=0,
            dtype=np.complex64,
        )
    )
    assert loaded.dtype == np.complex64


def test_no_evaluation_path_hard_codes_a_complex_dtype() -> None:
    """The ``np.complex128`` literals of defects D8 and D9 are gone.

    A source scan of the evaluation methods themselves, not a behavioural probe,
    because the defect was precisely that the literal was invisible at every
    reachable configuration: the default preset resolves ``complex128`` too, so
    only reading the source proves the dtype is no longer chosen there.  The
    scan is scoped to the methods rather than to whole files, because the host
    constant matrices ``S`` and ``R(chi)`` are legitimately ``complex128``
    literals -- they are the exact canonical values, not a runtime dtype choice.
    """
    import inspect

    from radiosim.core.jones.receptor import _ReceptorTermBase

    for method in (
        JonesChain.compute_antenna_jones_batch,
        _ReceptorTermBase.compute_jones_batch,
    ):
        body = inspect.getsource(method)
        body = "\n".join(
            line for line in body.splitlines() if "``" not in line and '"""' not in line
        )
        assert "complex128" not in body, method.__qualname__


# ---------------------------------------------------------------------------
# I2 -- declared flags are true, and non-vacuous
# ---------------------------------------------------------------------------


def _is_diagonal(matrix: np.ndarray) -> bool:
    return bool(matrix[0, 1] == 0.0 and matrix[1, 0] == 0.0)


def _is_scalar(matrix: np.ndarray) -> bool:
    return bool(np.array_equal(matrix, matrix[0, 0] * np.eye(2, dtype=matrix.dtype)))


def _is_unitary(matrix: np.ndarray) -> bool:
    return bool(np.allclose(matrix @ matrix.conj().T, np.eye(2), rtol=0.0, atol=1e-15))


PROPERTY_CHECKS = {
    "is_diagonal": _is_diagonal,
    "is_scalar": _is_scalar,
    "is_unitary": _is_unitary,
}


#: Flags that no swept term ever declares ``False``, with the reason.  Listing
#: them explicitly is what stops the converse check below from degenerating into
#: a skip when every term happens to declare a property.
UNIVERSALLY_DECLARED = {
    "is_unitary": (
        "C and H are products of unitary factors at every basis, rotation and "
        "output basis, so unitarity is declared unconditionally and truthfully"
    ),
}


@pytest.mark.parametrize("property_name", sorted(PROPERTY_CHECKS))
def test_every_declared_true_flag_is_numerically_true(property_name: str) -> None:
    """I2, forward direction: a declared property must hold on the numbers."""
    check = PROPERTY_CHECKS[property_name]
    verified = 0
    for label, term in _swept_terms():
        if not getattr(term, property_name)():
            continue
        matrix = _evaluate(term)[0]
        assert check(matrix), (
            f"{label} declares {property_name}() but its matrix does not have "
            f"that property: {matrix!r}"
        )
        verified += 1

    assert verified > 0, (
        f"no swept term declares {property_name}(), so the forward direction of "
        "I2 would pass vacuously for this property"
    )


@pytest.mark.parametrize("property_name", sorted(PROPERTY_CHECKS))
def test_every_declared_false_flag_has_a_witness(property_name: str) -> None:
    """I2, converse: a declared ``False`` must fail somewhere in the sweep.

    Without this half, a term could declare every flag ``False`` and pass the
    forward test vacuously -- the mirror image of defect D10, and just as
    uninformative.
    """
    swept = _swept_terms()
    declared_false = {
        label.split("[")[0]
        for label, term in swept
        if not getattr(term, property_name)()
    }
    if not declared_false:
        assert property_name in UNIVERSALLY_DECLARED, (
            f"no term declares {property_name}() False and no reason is recorded "
            "for that; add the reason or add a term that declares it False"
        )
        return

    witnessed: set[str] = set()
    for label, term in swept:
        if getattr(term, property_name)():
            continue
        if not PROPERTY_CHECKS[property_name](_evaluate(term)[0]):
            witnessed.add(label.split("[")[0])

    assert witnessed == declared_false, (
        f"{declared_false - witnessed} declare {property_name}() False but the "
        "property never actually fails over the swept parameter space, so the "
        "declaration is vacuous"
    )


def test_unitarity_is_declared_true_and_verified_everywhere() -> None:
    """Both receptor terms are unitary at every swept parameter, not by accident."""
    for label, term in _swept_terms():
        assert term.is_unitary() is True, label
        assert _is_unitary(_evaluate(term)[0]), label


def test_the_identity_case_is_not_the_only_case_swept() -> None:
    """The sweep must contain a genuinely non-identity matrix.

    Defect D10 was a claim verified only against the identity.  This asserts the
    harness itself cannot degenerate the same way.
    """
    identity = np.eye(2, dtype=np.complex128)
    matrices = [_evaluate(term)[0] for _, term in _swept_terms()]
    assert any(np.array_equal(matrix, identity) for matrix in matrices)
    assert any(not np.allclose(matrix, identity) for matrix in matrices)


# ---------------------------------------------------------------------------
# D7 -- the chain rejects a baseline-dependent term
# ---------------------------------------------------------------------------


def test_add_term_rejects_a_baseline_dependent_term() -> None:
    import radiosim.core.jones as jones_package

    chain = JonesChain(BACKEND)
    for factory in (
        jones_package.BaselineMultiplicativeJones,
        jones_package.SmearingFactorJones,
    ):
        term = factory()
        assert isinstance(term, JonesBaselineTerm)
        assert not isinstance(term, JonesTerm)
        with pytest.raises(TypeError) as excinfo:
            chain.add_term(term)
        assert "JonesBaselineTerm" in str(excinfo.value)
    assert chain.terms == []


def test_add_term_rejects_something_that_is_not_a_jones_term() -> None:
    chain = JonesChain(BACKEND)
    with pytest.raises(TypeError) as excinfo:
        chain.add_term("not a term")  # type: ignore[arg-type]
    assert "not a JonesTerm" in str(excinfo.value)


def test_the_base_contract_cannot_be_left_unimplemented() -> None:
    """A term that does not implement the contract cannot be constructed at all.

    FLIPPED BY: Tier 7G.  ``compute_jones_batch`` was concrete-and-raising while
    terms were still ``term_status: planned``, because an abstract declaration
    would have made every one of them impossible to instantiate.  ``Z`` and
    ``T`` were the last two, so the method is now ``@abstractmethod`` and the
    contract is enforced at construction rather than at first use -- which is
    strictly earlier and strictly harder to get wrong.

    The body is kept and still raises, for the subclass that declares the method
    and then defers to it; both halves are asserted here.
    """

    class _Unimplemented(JonesTerm):
        @property
        def name(self) -> str:
            return "unimplemented"

        @property
        def is_direction_dependent(self) -> bool:
            return False

    assert "compute_jones_batch" in JonesTerm.__abstractmethods__
    with pytest.raises(TypeError) as excinfo:
        _Unimplemented()  # type: ignore[abstract]
    assert "compute_jones_batch" in str(excinfo.value)

    class _Deferring(_Unimplemented):
        def compute_jones_batch(self, **kwargs: Any) -> Any:
            return super().compute_jones_batch(**kwargs)

    with pytest.raises(NotImplementedError) as raised:
        _evaluate(_Deferring())
    assert "compute_jones_batch" in str(raised.value)


def test_the_baseline_contract_raises_rather_than_returning_an_identity() -> None:
    import radiosim.core.jones as jones_package

    term = jones_package.BaselineMultiplicativeJones()
    with pytest.raises(NotImplementedError) as excinfo:
        term.compute_baseline_factor(
            baseline_idx=0,
            antenna_p=0,
            antenna_q=1,
            directions=_directions(),
            frequency_hz=1.5e8,
            freq_idx=0,
            time_mjd=60_000.0,
            time_idx=0,
            backend=BACKEND,
            dtype=np.complex128,
        )
    assert "compute_baseline_factor" in str(excinfo.value)
