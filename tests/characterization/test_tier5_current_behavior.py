"""Characterize the Tier 5 polarization, receptor, and Jones-chain baseline.

Every test in this module pins behavior that exists on `main` **today**, before
any Tier 5 production change.  Each test docstring names the slice that owns the
deliberate flip (``OWNED BY: Tier 5x``); a later slice must update the named test
in the same commit that changes the behavior.  A test with no ``OWNED BY`` line
pins behavior Tier 5 preserves.

Tier 5A evidence record
=======================

Slice 5A is the evidence gate for `Tier5ReceptorFeedPlan.md` §43.  The plan's
Section 35 grants 5A exactly two writable files, both of them test modules, so
the recorded evidence lives here rather than in the plan document.

Q1 — the Stokes ``V`` convention (blocks 5C).  **The plan's Section 10.2
correction stands.**  Retrieved 2026-07-29:

* Hamaker, Bregman & Sault 1996, A&AS 117, 137 (plan reference R1;
  https://aas.aanda.org/articles/aas/pdf/1996/07/dst6484.pdf).  Eq. (3) defines
  the coherency vector in the geometric ``xy`` representation as the ordered
  outer product ``(e_x e_x*, e_x e_y*, e_y e_x*, e_y e_y*)``.  Eq. (8) defines
  the Stokes vector as ``e^S = T e^+`` with

      T = [[1, 0, 0,  1],
           [1, 0, 0, -1],
           [0, 1, 1,  0],
           [0, -i, i, 0]]

  and Eq. (9) gives the inverse ``S = T^-1``

      S = (1/2) * [[1,  1, 0,  0],
                   [0,  0, 1,  i],
                   [0,  0, 1, -i],
                   [1, -1, 0,  0]]

  Read row by row against the Eq. (3) ordering, Eq. (9) *is* the Stokes-to-
  coherency map in the linear basis and gives ``<e_x e_y*> = (U + iV)/2``.
  Equivalently from Eq. (8), ``V = i(<e_y e_x*> - <e_x e_y*>)``, so
  ``Im<e_x e_y*> = +V/2``.
* Smirnov 2011, A&A 527, A106 (plan reference R3;
  https://www.aanda.org/articles/aa/full_html/2011/03/aa16082-10/aa16082-10.html).
  Eq. (7) gives the linear brightness matrix ``B = [[I+Q, U+iV], [U-iV, I-Q]]``
  and Section 6.3 gives the circular form ``B_c = H B H^H =
  [[I+V, Q+iU], [Q-iU, I-V]]``, i.e. ``RR = I + V``.
* ``codex-africanus`` (ska-sa/ratt-ru, ``africanus/model/coherency/
  conversion.py``, lines 13-22) implements

      "XY": lambda u, v: u + v * 1j        "YX": lambda u, v: u - v * 1j
      "RR": lambda i, v: i + v             "LL": lambda i, v: i - v

  **This directly contradicts the attribution in RadioSim's own module
  docstring** (`src/radiosim/core/polarization.py:22-27`), which labels the
  current ``C[0,1] = (U - iV)/2`` the "Africanus/Pauli" convention and claims it
  "Matches: Codex-Africanus".  Africanus implements the opposite sign.
  ``test_polarization_docstring_carries_the_iau_hbs_attribution`` pinned that
  false claim at 5A and, after the Tier 5C flip, pins its removal.
* Contrary evidence, recorded rather than suppressed: ``pyradiosky``
  (RadioAstronomySoftwareGroup, ``src/pyradiosky/utils.py``
  ``stokes_to_coherency``) builds ``0.5 * [[I+Q, U-1j*V], [U+1j*V, I-Q]]``, and
  Hamaker 2006, A&A 456, 395 Eq. (3) prints ``[[I+Q, U-iV], [U+iV, I-Q]]``.  The
  baseline therefore agrees with pyradiosky, not with the primary references the
  plan cites.  RadioSim's ``pyradiosky_file`` loader reads Stokes ``I/Q/U/V``
  columns, never a pyradiosky coherency matrix, so 5C's correction does not
  create a data-path inconsistency — but after 5C the two packages will disagree
  on the sign of ``V`` in the coherency matrix.
* Thompson, Moran & Swenson 3rd ed. §4.7 (plan reference R4) could not be
  retrieved: every open-access route tried (OAPEN bitstream, SpringerLink
  chapter and content PDF) either reset the connection or redirected to an
  authentication host.  R4 is therefore **not** independently confirmed here.
  R1 and R3 — one of which the plan already cites as the origin of the Jones
  formalism — are consistent with each other and with codex-africanus, and are
  taken as sufficient.

Q2 — where ``resolve_receptors()`` is invoked (blocks 5B).  ``resolve_instrument()``
has exactly one caller in the tree: ``Simulator._ensure_instrument_state``
(`src/radiosim/api/simulator.py:414-416`).  ``resolve_config()`` in
`src/radiosim/io/config_resolution.py` never resolves an instrument; it only
carries the typed ``instrument`` configuration section.  The resolved instrument
therefore first exists inside ``Simulator``, so the Section 25.2 ordering — after
instrument resolution, before beam load — can only be honoured in
``Simulator.setup()``, between ``self._ensure_instrument_state()``
(`src/radiosim/api/simulator.py:526`) and ``self._ensure_beam_system()``
(`:530`).  ``Simulator.observability()`` calls the same two helpers directly at
`:1193-1194`, so 5B must add an idempotent ``_ensure_receptor_set()`` helper and
call it at both sites rather than inlining the resolution once.
``test_resolve_instrument_has_exactly_one_caller_inside_the_simulator`` pins
this.

Q3 is answered in ``test_pyuvdata_321_polarization_contract.py``.

Q4 — ``visibility_to_correlations`` (resolved in 5H).  It has no production
caller; the only references outside its own module are the two re-export lines
in ``radiosim.core.__init__``.  Pinned by
``test_superseded_polarization_helpers_have_no_production_caller``.

Q5 — ``mueller_from_jones`` (resolved in 5H).  It raises ``NotImplementedError``
as the plan says, but it is **not** publicly exported: it is absent from
``radiosim.core.__all__`` and from the ``radiosim.core`` namespace, reachable
only as ``radiosim.core.polarization.mueller_from_jones``.  Pinned by
``test_mueller_from_jones_is_module_public_but_unimplemented``.

Contradictions for the 5A acceptance reviewer
=============================================

1. Section 10.2 and Section 43 Q1 state that the docstring's "Africanus/Pauli"
   attribution reflects what codex-africanus implements.  It does not;
   codex-africanus implements the sign the plan proposes to move *to*.  The
   correction stands, but the plan's characterization of the defect's origin is
   wrong.
2. Section 43 Q3 assumes ``Telescope.new(feeds=["r","l"], feed_angle=...)``
   configures circular feeds.  In pyuvdata 3.2.1 the ``feeds`` argument is
   silently ignored unless ``x_orientation`` is also supplied, and
   ``feed_array`` must be given with shape ``(Nants, Nfeeds)``.  See
   ``test_pyuvdata_321_polarization_contract.py``.
3. Section 43 Q5 asserts ``mueller_from_jones`` is publicly exported.  It is not.
4. Section 43 Q4 asks whether ``visibility_to_correlations`` has a production
   caller.  It does not — and neither do ``stokes_I_only_visibility``,
   ``apply_jones_matrices``, ``jones_matrix_power``, or ``mueller_from_jones``.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation

import radiosim.core.polarization as polarization_module
import radiosim.core.result as result_module
import radiosim.core.visibility as visibility_module
import radiosim.core.visibility_healpix as visibility_healpix_module
import radiosim.io.hdf5 as hdf5_module
import radiosim.io.standard_visibility as standard_visibility_module
from radiosim.api import Simulator
from radiosim.backends import get_backend
from radiosim.core.beam import BeamSystem
from radiosim.core.instrument import AntennaId
from radiosim.core.instrument_adapters import SolverInstrumentView
from radiosim.core.jones.base import JonesTerm
from radiosim.core.jones.chain import JonesChain
from radiosim.core.jones.receptor import BasisTransformJones, ReceptorConfigJones
from radiosim.core.polarization import (
    coherency_to_stokes,
    mueller_from_jones,
    stokes_to_coherency,
)
from radiosim.core.receptor import resolve_receptors
from radiosim.core.result import SimulationResult
from radiosim.core.visibility import _build_jones_chain
from radiosim.io.receptor_config import ReceptorsConfig
from tests.fixtures.configs import valid_config_mapping

FREQUENCIES_HZ = np.array([100_000_000.0], dtype=np.float64)
LOCATION = EarthLocation.from_geodetic(
    21.4283 * u.deg,
    -30.72152 * u.deg,
    1073.0 * u.m,
)

# The linear-to-circular basis matrix of Tier5ReceptorFeedPlan.md Section 18.1,
# rows ordered (R, L) and columns (x, y).  Written out here so the circular
# consequence below is derived from the plan, not from RadioSim source.
PLAN_S_MATRIX = (1.0 / np.sqrt(2.0)) * np.array(
    [[1.0, 1.0j], [1.0, -1.0j]],
    dtype=np.complex128,
)

SOURCE_ROOT = Path(inspect.getfile(result_module)).parents[2]


def _solver_components(tmp_path: Path) -> tuple[SolverInstrumentView, BeamSystem]:
    simulator = Simulator.from_mapping(
        valid_config_mapping(
            tmp_path,
            frequency={
                "mode": "explicit",
                "channel_frequencies_hz": FREQUENCIES_HZ.tolist(),
                "channel_widths_hz": [1e6],
            },
        ),
        base_dir=tmp_path,
    )
    simulator._ensure_instrument_state()
    simulator._ensure_beam_system()
    return (
        SolverInstrumentView.from_state(simulator._instrument_state),
        simulator.beam_system,
    )


def _resolve_instrument(tmp_path: Path):
    """Return the canonical ResolvedInstrument behind ``_solver_components``."""
    simulator = Simulator.from_mapping(
        valid_config_mapping(
            tmp_path,
            frequency={
                "mode": "explicit",
                "channel_frequencies_hz": FREQUENCIES_HZ.tolist(),
                "channel_widths_hz": [1e6],
            },
        ),
        base_dir=tmp_path,
    )
    simulator._ensure_instrument_state()
    return simulator._instrument_state.instrument


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


# ---------------------------------------------------------------------------
# Stokes V convention (defect D3)
# ---------------------------------------------------------------------------


def test_stokes_to_coherency_places_plus_iv_in_the_upper_right() -> None:
    """Pins the corrected C[0,1] = (U + iV)/2.

    FLIPPED BY: Tier 5C.  The baseline built ``C[0,1] = (U - iV)/2`` and this
    test pinned it; Tier 5C applied the Section 10.2 correction, ratified by the
    Q1 evidence in this module's docstring, so the pin now records the corrected
    construction.  Tier 5 preserves this from here on.
    """
    stokes_i, stokes_q, stokes_u, stokes_v = 10.0, 2.0, -1.0, 0.5
    coherency = stokes_to_coherency(stokes_i, stokes_q, stokes_u, stokes_v)

    expected = 0.5 * np.array(
        [
            [stokes_i + stokes_q, stokes_u + 1j * stokes_v],
            [stokes_u - 1j * stokes_v, stokes_i - stokes_q],
        ],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(coherency, expected, rtol=0.0, atol=0.0)

    # The half-power convention is independent of the V sign and is preserved.
    np.testing.assert_allclose(
        coherency[0, 0] + coherency[1, 1],
        stokes_i,
        rtol=0.0,
        atol=1e-15,
    )
    # The sign is only observable in the cross hands, and only for V != 0.
    assert coherency[0, 1].imag == pytest.approx(+stokes_v / 2.0)
    assert coherency[1, 0].imag == pytest.approx(-stokes_v / 2.0)


def test_coherency_to_stokes_derives_v_from_the_upper_right_element() -> None:
    """Pins the corrected inverse V = 2 * Im(C[0,1]) and its round trip.

    FLIPPED BY: Tier 5C.  The baseline derived V from ``C[1,0]``; the derivation
    moved to ``C[0,1]`` in the same commit as the forward correction, because
    changing only one direction would break the round trip pinned below.
    """
    stokes = (7.5, -1.25, 3.0, -2.0)
    coherency = stokes_to_coherency(*stokes)

    assert coherency_to_stokes(coherency)[3] == pytest.approx(
        2.0 * coherency[0, 1].imag
    )
    np.testing.assert_allclose(
        np.asarray(coherency_to_stokes(coherency), dtype=np.float64),
        np.asarray(stokes, dtype=np.float64),
        rtol=0.0,
        atol=1e-14,
    )


def test_corrected_convention_maps_a_positive_v_source_to_pure_rr() -> None:
    """Pins the circular consequence of the corrected V sign.

    Under the Section 18.1 basis matrix S a source with V = +I emerges as pure
    RR.  The baseline produced pure LL, which is the observable defect
    Section 10.2 names.

    FLIPPED BY: Tier 5C, in the same commit as the construction change.
    """
    total_flux = 4.0
    coherency = stokes_to_coherency(total_flux, 0.0, 0.0, total_flux)
    circular = PLAN_S_MATRIX @ coherency @ PLAN_S_MATRIX.conj().T

    # RR is index [0, 0] and LL is index [1, 1] in the (R, L) row ordering.
    assert circular[0, 0].real == pytest.approx(total_flux, abs=1e-15)
    assert circular[1, 1].real == pytest.approx(0.0, abs=1e-15)

    # Stated as the Section 18.4 relations RR = (I + V)/2, LL = (I - V)/2.
    stokes_i, stokes_v = 9.0, 3.0
    circular = (
        PLAN_S_MATRIX
        @ stokes_to_coherency(stokes_i, 0.0, 0.0, stokes_v)
        @ PLAN_S_MATRIX.conj().T
    )
    assert circular[0, 0].real == pytest.approx((stokes_i + stokes_v) / 2.0)
    assert circular[1, 1].real == pytest.approx((stokes_i - stokes_v) / 2.0)


def test_polarization_docstring_carries_the_iau_hbs_attribution() -> None:
    """Pins the removal of the module docstring claim the Q1 evidence refuted.

    The baseline text labelled ``C[0,1] = (U - iV)/2`` the "Africanus/Pauli"
    convention and claimed it matched codex-africanus, which in fact implements
    ``XY = U + iV``.  Tier 5C replaced the text with the IAU/HBS attribution and
    an explicit statement of the pyradiosky divergence (risk register).

    FLIPPED BY: Tier 5C, in the same commit as the construction change.
    """
    docstring = polarization_module.__doc__ or ""
    assert "Africanus/Pauli" not in docstring
    assert "Matches: Codex-Africanus" not in docstring
    assert "NOT: (U + iV) / 2      (Smirnov 2011 alternative)" not in docstring

    assert "C[0,1] = (U + iV) / 2" in docstring
    assert "Hamaker, Bregman & Sault 1996" in docstring
    assert "Smirnov 2011" in docstring
    assert "codex-africanus" in docstring
    assert "pyradiosky" in docstring


# ---------------------------------------------------------------------------
# Receptor terms (defect D1/D2)
# ---------------------------------------------------------------------------


def test_receptor_and_basis_transform_terms_carry_real_physics(tmp_path) -> None:
    """Pins both Tier 5 Jones terms as the Section 18.2 / 18.3 mathematics.

    At the baseline both classes accepted a permissive ``feed_type`` /
    ``from_basis`` / ``to_basis`` construction and returned the identity
    regardless, so a configured circular array silently produced linear
    visibilities.  Tier 5C removed both stub constructors and implemented the
    real matrices; the terms are still not wired into any chain, which is
    Tier 5D.

    FLIPPED BY: Tier 5C, in the same commit as the implementation.
    """
    backend = get_backend("numpy")
    identity = np.eye(2, dtype=np.complex128)

    # The permissive stub constructors are gone outright (Section 24).
    for call in (
        lambda: ReceptorConfigJones(feed_type="linear"),
        lambda: ReceptorConfigJones(feed_type="circular"),
        lambda: BasisTransformJones(from_basis="linear", to_basis="circular"),
        lambda: BasisTransformJones(from_basis="circular", to_basis="linear"),
        ReceptorConfigJones,
        BasisTransformJones,
    ):
        with pytest.raises(TypeError):
            call()

    instrument, _ = _solver_components(tmp_path)
    resolved = _resolve_instrument(tmp_path)

    # Default linear array: both terms are exactly I2, so the default
    # configuration cannot perturb any existing result.
    linear = resolve_receptors(ReceptorsConfig(), resolved)
    for term in (
        ReceptorConfigJones(receptors=linear, instrument=instrument),
        BasisTransformJones(receptors=linear, instrument=instrument),
    ):
        for antenna_idx in range(len(instrument.antenna_numbers)):
            jones = term.compute_jones(antenna_idx, None, 0, 0, backend)
            np.testing.assert_array_equal(np.asarray(jones), identity)

    # Circular array: C is the Section 18.1 basis matrix, H stays I2 because
    # the native basis already is the output basis.
    circular = resolve_receptors(
        ReceptorsConfig.model_validate({"default": {"basis": "circular"}}),
        resolved,
    )
    receptor_term = ReceptorConfigJones(receptors=circular, instrument=instrument)
    transform_term = BasisTransformJones(receptors=circular, instrument=instrument)
    np.testing.assert_allclose(
        np.asarray(receptor_term.compute_jones(0, None, 0, 0, backend)),
        PLAN_S_MATRIX,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_array_equal(
        np.asarray(transform_term.compute_jones(0, None, 0, 0, backend)),
        identity,
    )

    assert receptor_term.name == "C"
    assert transform_term.name == "H"
    assert receptor_term.is_direction_dependent is False
    assert transform_term.is_direction_dependent is False
    # Unitarity is now a truthful claim rather than an artefact of being I2.
    assert receptor_term.is_unitary() is True
    assert transform_term.is_unitary() is True
    assert receptor_term.is_diagonal() is False


# ---------------------------------------------------------------------------
# Jones chain composition and solver term order (defect D7)
# ---------------------------------------------------------------------------


def test_jones_chain_composes_the_first_added_term_leftmost() -> None:
    """Pins JonesChain semantics with two deliberately non-commuting terms.

    Tier 5D preserves this composition rule and changes only the order in which
    the solver adds terms, so this test is a stable anchor for S13.
    """
    first = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    second = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=np.complex128)
    assert not np.allclose(first @ second, second @ first)

    backend = get_backend("numpy")
    chain = JonesChain(backend)
    chain.add_term(_ConstantJones("first", first))
    chain.add_term(_ConstantJones("second", second))

    np.testing.assert_allclose(
        np.asarray(chain.compute_antenna_jones(0, None, 0, 0)),
        first @ second,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(chain.compute_antenna_jones_all_sources(0, 3, 0, 0))[1],
        first @ second,
        rtol=0.0,
        atol=0.0,
    )


def test_point_solver_adds_chain_terms_in_the_canonical_order(
    tmp_path: Path,
) -> None:
    """Pins the Section 19.1 solver term order H G B D P C E T Z.

    At the baseline the additions ran in the inverted order Z T E P D G B, so
    the bandpass reached the sky field first.  Combined with the composition
    rule above, the corrected order yields J = H G B D P C E T Z with K applied
    separately.

    FLIPPED BY: Tier 5D, in the same commit as the reordering.
    """
    instrument, beam_system = _solver_components(tmp_path)
    receptors = resolve_receptors(ReceptorsConfig(), _resolve_instrument(tmp_path))
    n_sources = 2
    chain = _build_jones_chain(
        get_backend("numpy"),
        {
            "Z": {"enabled": True},
            "T": {"enabled": True},
            "P": {"enabled": True},
            "D": {"enabled": True},
            "G": {"enabled": True},
            "B": {"enabled": True},
        },
        instrument,
        np.full(n_sources, 1.0, dtype=np.float64),
        np.full(n_sources, 0.5, dtype=np.float64),
        FREQUENCIES_HZ[0],
        0,
        n_sources,
        LOCATION,
        60_676.0,
        beam_system,
        receptors,
    )

    assert [term.name for term in chain.terms] == [
        "H",
        "G",
        "B",
        "D",
        "P",
        "C",
        "E",
        "T",
        "Z",
    ]


def test_point_solver_chain_always_carries_the_receptor_terms(
    tmp_path: Path,
) -> None:
    """Pins H, C, and E as the always-enabled chain terms.

    At the baseline E was the only always-enabled term, so no receptor state
    could reach the visibilities.

    FLIPPED BY: Tier 5D, in the same commit as the solver integration.
    """
    instrument, beam_system = _solver_components(tmp_path)
    receptors = resolve_receptors(ReceptorsConfig(), _resolve_instrument(tmp_path))
    chain = _build_jones_chain(
        get_backend("numpy"),
        {},
        instrument,
        np.array([1.0], dtype=np.float64),
        np.array([0.5], dtype=np.float64),
        FREQUENCIES_HZ[0],
        0,
        1,
        LOCATION,
        60_676.0,
        beam_system,
        receptors,
    )
    assert [term.name for term in chain.terms] == ["H", "C", "E"]


def test_healpix_solver_never_constructs_a_jones_chain() -> None:
    """Pins the HEALPix path as a direct beam evaluation with no chain.

    Tier 5D routed the receptor terms into this path by left-multiplying the
    per-antenna beam Jones by the constant ``H_p @ C_p`` (Section 19.3), which
    is exact because both terms are direction, time, and frequency independent.
    No second chain implementation was introduced.

    FLIPPED BY: Tier 5D, in the same commit as the solver integration.
    """
    source = inspect.getsource(visibility_healpix_module)
    assert "JonesChain" not in source
    assert "_build_jones_chain" not in source
    assert "beam_system.evaluate_jones" in source
    # The receptor factor reaches this path as a matrix, not as a chain.
    assert "_receptor_transforms" in source
    assert "basis_transform_matrix" in source
    assert "receptor_matrix" in source

    # The point-source path is the only chain builder today.
    assert "JonesChain" in inspect.getsource(visibility_module)


def test_beam_jones_response_is_a_scalar_multiple_of_the_identity(
    tmp_path: Path,
) -> None:
    """Pins the Tier 3 E = e I2 boundary that Tier 5 relies on.

    E commuting with every receptor matrix is what makes the Section 10.1
    sky-linear decomposition exact rather than approximate.  Tier 5 preserves
    this constraint (Section 42).
    """
    instrument, beam_system = _solver_components(tmp_path)
    antenna_id = AntennaId(instrument.antenna_numbers[0], instrument.antenna_names[0])
    jones = np.asarray(
        beam_system.evaluate_jones(
            antenna_id,
            altitude_rad=np.array([1.2], dtype=np.float64),
            azimuth_rad=np.array([0.4], dtype=np.float64),
            frequency_hz=float(FREQUENCIES_HZ[0]),
            time_mjd=60_676.0,
        )
    )
    assert jones.shape == (1, 2, 2)
    np.testing.assert_array_equal(jones[:, 0, 1], 0.0)
    np.testing.assert_array_equal(jones[:, 1, 0], 0.0)
    np.testing.assert_array_equal(jones[:, 0, 0], jones[:, 1, 1])


# ---------------------------------------------------------------------------
# Correlation constants and result indexing (defects D4 and D6)
# ---------------------------------------------------------------------------


def test_all_four_correlation_constant_sites_now_share_the_table() -> None:
    """Tracks the four duplicated correlation contracts of Section 6.3.

    OWNED BY: Tier 5E and Tier 5F, which replace all four with the single
    ``radiosim.core.polarization_basis`` constant.

    NARROWED BY: Tier 5C.  Two clauses of the 5A pin were collateral rather
    than part of the D4 contract, and Tier 5C's own mandated deliverables
    (Sections 34.3 and 35: add ``core/polarization_basis.py``; rewrite the
    ``core/polarization.py`` attribution to cite codex-africanus, whose
    mapping names ``"RR"`` and ``"LL"``) falsified both.

    RENAMED AND FLIPPED BY: Tier 5E, for the two sites Section 35 grants it --
    ``core/result.py`` and ``io/hdf5.py``.

    RENAMED AND FLIPPED AGAIN BY: Tier 5F, for its own two sites.
    ``io/standard_visibility.py`` no longer defines
    ``CANONICAL_CORRELATIONS``, ``CANONICAL_CODES``, or ``FILE_CODES``; it
    imports the shared table like the other three, so defect D4 is now closed
    outright.  The circular-label scan that guarded the writer and the
    Measurement Set path is inverted below: circular *labels* still appear
    nowhere in either module, because both derive every label and code from the
    shared table rather than spelling any basis out.
    """
    assert not hasattr(result_module, "_CORRELATIONS")
    assert not hasattr(hdf5_module, "CORRELATIONS")
    assert not hasattr(hdf5_module, "AIPS_CODES")
    for name in ("CANONICAL_CORRELATIONS", "CANONICAL_CODES", "FILE_CODES"):
        assert not hasattr(standard_visibility_module, name)

    # All four sites now consume the shared table and nothing else.
    assert (SOURCE_ROOT / "radiosim" / "core" / "polarization_basis.py").exists()
    for module in (result_module, hdf5_module, standard_visibility_module):
        source = inspect.getsource(module)
        assert "core.polarization_basis" in source
        assert '("XX", "XY", "YX", "YY")' not in source
        assert '("RR", "RL", "LR", "LL")' not in source

    # No production module spells a correlation label out any more: the writer,
    # the Measurement Set path, and every other module read the shared table.
    label_hits = {
        path.name
        for path in (SOURCE_ROOT / "radiosim").rglob("*.py")
        if path.name != "polarization_basis.py"
        and any(
            token in path.read_text()
            for token in ('"RR"', '"LL"', '"rr"', '"ll"', '"XX"', '"YY"')
        )
    }
    assert label_hits.isdisjoint({"standard_visibility.py", "measurement_set.py"})


def test_pyuvdata_construction_is_basis_driven() -> None:
    """Pins the fourth correlation site, inside the standard-format writer.

    OWNED BY: Tier 5F.  FLIPPED BY: Tier 5F -- the writer now passes an
    explicit per-basis ``feed_array``/``feed_angle`` pair (the Tier 5A Q3
    construction-form correction) and a ``polarization_array`` read from
    ``PYUVDATA_POLARIZATIONS``.  The deprecated ``x_orientation`` shorthand is
    gone: pyuvdata 3.2.1 ignores it whenever ``feed_array`` and ``feed_angle``
    are both supplied, so retaining it for the linear path would have been dead
    code.
    """
    source = inspect.getsource(standard_visibility_module)
    assert 'feeds=["x", "y"]' not in source
    assert "x_orientation" not in source
    assert 'polarization_array=["xx", "xy", "yx", "yy"]' not in source
    assert "feed_array=np.tile(" in source
    assert "feed_angle=np.tile(" in source
    assert "polarization_array=list(PYUVDATA_POLARIZATIONS[basis])" in source


def test_stokes_i_derives_its_indices_from_the_correlation_labels() -> None:
    """Records that ``stokes_i()`` consults ``self.correlations`` (defect D6).

    OWNED BY: Tier 5E.  FLIPPED BY: Tier 5E -- the fixed ``0``/``3`` literals
    are gone and the indices come from
    ``radiosim.core.polarization_basis.parallel_hand_indices``.
    """
    source = inspect.getsource(SimulationResult.stokes_i)
    assert "self.visibilities[..., 0] + self.visibilities[..., 3]" not in source
    assert "parallel_hand_indices(self.correlations)" in source

    class _Linear:
        correlations = ("XX", "XY", "YX", "YY")
        visibilities = np.arange(8, dtype=np.complex128).reshape(1, 1, 2, 4)

    class _Circular:
        correlations = ("RR", "RL", "LR", "LL")
        visibilities = np.arange(8, dtype=np.complex128).reshape(1, 1, 2, 4)

    class _Hostile:
        correlations = ("XX", "YY", "XY", "YX")
        visibilities = np.arange(8, dtype=np.complex128).reshape(1, 1, 2, 4)

    for holder in (_Linear, _Circular):
        np.testing.assert_array_equal(
            SimulationResult.stokes_i(holder()),
            holder.visibilities[..., 0] + holder.visibilities[..., 3],
        )
    with pytest.raises(ValueError, match="accepted correlation coordinate set"):
        SimulationResult.stokes_i(_Hostile())


def test_polarization_basis_is_data_driven_at_every_result_construction_site() -> None:
    """Records the removal of the Section 6.3 literals.

    OWNED BY: Tier 5E.  FLIPPED BY: Tier 5E -- every construction site and the
    scientific fingerprint now read the resolved receptor output basis.
    """
    source = inspect.getsource(result_module)
    assert 'polarization_basis="linear_xy"' not in source
    assert '_hash_json(digest, "polarization_basis", "linear_xy")' not in source
    assert '_hash_json(digest, "polarization_basis", polarization_basis)' in source
    assert '_hash_json(digest, "receptor", receptor_snapshot)' in source
    # Both construction sites, and both fingerprint calls, are data driven.
    assert source.count("correlations=CORRELATION_LABELS[polarization_basis]") == 2
    assert source.count("correlations=correlation_labels") == 2
    assert source.count("polarization_basis = receptors.output_basis") == 1
    assert source.count("basis_for_correlations(correlation_labels)") == 1
    assert (
        'raise InvalidResultError("correlations must be exactly XX, XY, YX, YY")'
        not in source
    )


# ---------------------------------------------------------------------------
# Q2, Q4, and Q5 source evidence
# ---------------------------------------------------------------------------


def test_resolve_instrument_has_exactly_one_caller_inside_the_simulator() -> None:
    """Records the Q2 answer: the receptor resolution host is ``Simulator``.

    ``resolve_config()`` never produces a resolved instrument, so the Section
    25.2 ordering can only be honoured inside ``Simulator``.
    """
    callers = sorted(
        path.relative_to(SOURCE_ROOT).as_posix()
        for path in (SOURCE_ROOT / "radiosim").rglob("*.py")
        if "resolve_instrument(" in path.read_text()
        and path.name != "instrument_resolution.py"
    )
    assert callers == ["radiosim/api/simulator.py"]

    simulator_source = inspect.getsource(Simulator.setup)
    instrument_position = simulator_source.index("self._ensure_instrument_state()")
    receptor_position = simulator_source.index("self._ensure_receptor_set()")
    beam_position = simulator_source.index("self._ensure_beam_system()")
    assert instrument_position < receptor_position < beam_position

    import radiosim.io.config_resolution as config_resolution_module

    assert "resolve_instrument" not in inspect.getsource(config_resolution_module)


def test_superseded_polarization_helpers_have_no_production_caller() -> None:
    """Records the Q4 answer and the state of the other legacy helpers.

    OWNED BY: Tier 5H, which decides the fate of each helper on this evidence.
    """
    helpers = (
        "visibility_to_correlations",
        "stokes_I_only_visibility",
        "apply_jones_matrices",
        "jones_matrix_power",
        "mueller_from_jones",
    )
    for helper in helpers:
        callers = sorted(
            path.relative_to(SOURCE_ROOT).as_posix()
            for path in (SOURCE_ROOT / "radiosim").rglob("*.py")
            if f"{helper}(" in path.read_text()
            and path.name not in {"polarization.py", "__init__.py"}
        )
        assert callers == [], f"{helper} unexpectedly has a production caller"

    # stokes_to_coherency, by contrast, is the one live entry point.
    live = sorted(
        path.relative_to(SOURCE_ROOT).as_posix()
        for path in (SOURCE_ROOT / "radiosim").rglob("*.py")
        if "stokes_to_coherency(" in path.read_text()
        and path.name not in {"polarization.py", "__init__.py"}
    )
    assert live == [
        "radiosim/core/visibility.py",
        "radiosim/core/visibility_healpix.py",
    ]


def test_mueller_from_jones_is_module_public_but_unimplemented() -> None:
    """Records the Q5 answer, and one correction to its premise.

    Section 43 Q5 states that ``mueller_from_jones`` "raises
    ``NotImplementedError`` while being publicly exported".  The first half is
    true; the second is not.  It is *not* re-exported from ``radiosim.core`` and
    is not in ``radiosim.core.__all__`` — it is reachable only as
    ``radiosim.core.polarization.mueller_from_jones``, an undecorated public
    module-level name.  ``jones_matrix_power`` and ``stokes_I_only_visibility``
    are in the same state; ``apply_jones_matrices``,
    ``visibility_to_correlations``, and ``stokes_to_coherency`` are the three
    names the package does re-export.

    OWNED BY: Tier 5H, which either removes it or gates it explicitly as Tier 7.
    """
    import radiosim.core as core_package

    assert "mueller_from_jones" not in core_package.__all__
    assert not hasattr(core_package, "mueller_from_jones")
    assert sorted(
        name
        for name in core_package.__all__
        if name
        in {
            "stokes_to_coherency",
            "apply_jones_matrices",
            "visibility_to_correlations",
            "coherency_to_stokes",
            "stokes_I_only_visibility",
            "jones_matrix_power",
            "mueller_from_jones",
        }
    ) == [
        "apply_jones_matrices",
        "stokes_to_coherency",
        "visibility_to_correlations",
    ]

    with pytest.raises(NotImplementedError):
        mueller_from_jones(np.eye(2, dtype=np.complex128))


def test_receptor_configuration_surface_exists() -> None:
    """Pins the Tier 5B production surfaces that replaced the 5A absence pin.

    Flipped by Tier 5B, which introduced every path this test once asserted
    absent.  The Section 25.2 ordering is pinned by the Q2 test above.
    """
    from radiosim.io.config import RadioSimConfig

    assert "receptors" in RadioSimConfig.model_fields
    assert (SOURCE_ROOT / "radiosim" / "core" / "receptor.py").exists()
    assert (SOURCE_ROOT / "radiosim" / "io" / "receptor_config.py").exists()
