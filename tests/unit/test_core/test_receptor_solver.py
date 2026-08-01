"""Tier 5D: resolved receptors change the computed visibilities.

Both solver paths consume one :class:`ResolvedReceptorSet` and apply
``H_p @ C_p`` to every antenna's Jones matrix
(``Tier5ReceptorFeedPlan.md`` Sections 19.2 and 19.3).  The oracles here are
written from the plan — the ``S`` matrix of Section 18.1, the correlation table
of Section 18.4, and the rotation invariants of Section 18.5 — and never from
the production constants, so a matching implementation defect cannot hide.

The invariants asserted are S1, S4, S5, S7, S8, S10 and S12.

Tier 5D does not touch the result model: a circular run is still stamped
``linear_xy`` with ``XX``/``XY``/``YX``/``YY`` labels, which is why every
assertion below reads the raw ``(2, 2)`` receptor cube (Section 34.4).
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

import radiosim.core.visibility as point_visibility
import radiosim.core.visibility_healpix as healpix_visibility
from radiosim.api import Simulator
from radiosim.backends import get_backend
from radiosim.core.beam import BeamSystem
from radiosim.core.instrument import AntennaId
from radiosim.core.instrument_adapters import SolverInstrumentView
from radiosim.core.receptor import (
    ResolvedReceptorSet,
    UnsupportedFeedGeometryError,
)
from radiosim.core.time_grid import build_observation_time_grid
from radiosim.core.visibility import calculate_visibility
from radiosim.core.visibility_healpix import calculate_visibility_healpix
from radiosim.simulator import RIMESimulator, VisibilitySimulator
from tests.fixtures.configs import valid_config_mapping

FREQUENCY_HZ = 100_000_000.0
FREQUENCIES = np.array([FREQUENCY_HZ], dtype=np.float64)
LOCATION = EarthLocation.from_geodetic(0.0 * u.deg, 0.0 * u.deg, 0.0 * u.m)
OBSTIME = Time("2024-01-01T00:00:00")
TIME_GRID = build_observation_time_grid(
    start_time=OBSTIME.isot,
    duration_seconds=1.0,
    cadence_seconds=1.0,
)
ALTITUDE_RAD = np.pi / 3.0
AZIMUTH_RAD = 0.0

#: Section 18.1 ``S``, rows ordered ``(R, L)`` and columns ``(x, y)``.  Written
#: from the plan so the oracle is independent of the implementation.
PLAN_S = (1.0 / np.sqrt(2.0)) * np.array(
    [[1.0, 1.0j], [1.0, -1.0j]],
    dtype=np.complex128,
)
IDENTITY = np.eye(2, dtype=np.complex128)


def plan_rotation(chi_rad: float) -> np.ndarray:
    """Section 18.1 ``R(chi)``, written from the plan."""
    return np.array(
        [
            [np.cos(chi_rad), np.sin(chi_rad)],
            [-np.sin(chi_rad), np.cos(chi_rad)],
        ],
        dtype=np.complex128,
    )


def plan_receptor(basis: str, chi_deg: float, output_basis: str) -> np.ndarray:
    """Section 18.2 and 18.3: the combined ``H @ C`` for one antenna."""
    leading = PLAN_S if basis == "circular" else IDENTITY
    receptor = leading @ plan_rotation(np.deg2rad(chi_deg))
    native_output = "circular_rl" if basis == "circular" else "linear_xy"
    if native_output == output_basis:
        transform = IDENTITY
    elif output_basis == "circular_rl":
        transform = PLAN_S
    else:
        transform = PLAN_S.conj().T
    return transform @ receptor


def plan_coherency(
    stokes_i: float,
    stokes_q: float,
    stokes_u: float,
    stokes_v: float,
) -> np.ndarray:
    """Section 9.1 ``B = (1/2) [[I+Q, U+iV], [U-iV, I-Q]]``, from the plan."""
    return 0.5 * np.array(
        [
            [stokes_i + stokes_q, stokes_u + 1.0j * stokes_v],
            [stokes_u - 1.0j * stokes_v, stokes_i - stokes_q],
        ],
        dtype=np.complex128,
    )


# ---------------------------------------------------------------------------
# One-source, one-pixel, zero-baseline fixtures
# ---------------------------------------------------------------------------


class _FixedAltAzSkyCoord:
    def __init__(self, **_kwargs):
        pass

    def transform_to(self, _frame):
        return SimpleNamespace(
            az=SimpleNamespace(rad=np.array([AZIMUTH_RAD])),
            alt=SimpleNamespace(rad=np.array([ALTITUDE_RAD])),
        )


class _FixedPixelCoordinates:
    def __len__(self) -> int:
        return 1

    def transform_to(self, _frame):
        return SimpleNamespace(
            az=SimpleNamespace(rad=np.array([AZIMUTH_RAD])),
            alt=SimpleNamespace(rad=np.array([ALTITUDE_RAD])),
        )


class _OnePixelHealpix:
    nside = 1
    pixel_solid_angle = 1.0
    pixel_coords = _FixedPixelCoordinates()

    def __init__(self, stokes: tuple[float, float, float, float]) -> None:
        self._stokes = tuple(np.array([value], dtype=np.float64) for value in stokes)

    def get_map_at_frequency(self, _frequency):
        return self._stokes[0]

    def get_stokes_maps_at_frequency(self, _frequency):
        return self._stokes


def _healpix_sky(stokes: tuple[float, float, float, float], *, polarized: bool):
    return SimpleNamespace(
        healpix=_OnePixelHealpix(stokes),
        has_polarized_healpix_maps=polarized,
        brightness_conversion="rayleigh-jeans",
        model_name="tier5d-one-pixel",
    )


def _source_arrays(stokes: tuple[float, float, float, float]) -> dict[str, object]:
    zeros = np.zeros(1, dtype=np.float64)
    stokes_i, stokes_q, stokes_u, stokes_v = stokes
    return {
        "ra_rad": zeros.copy(),
        "dec_rad": zeros.copy(),
        "flux": np.array([stokes_i], dtype=np.float64),
        "spectral_index": zeros.copy(),
        "stokes_q": np.array([stokes_q], dtype=np.float64),
        "stokes_u": np.array([stokes_u], dtype=np.float64),
        "stokes_v": np.array([stokes_v], dtype=np.float64),
        "ref_freq": np.array([FREQUENCY_HZ], dtype=np.float64),
        "rotation_measure": zeros.copy(),
        "spectral_coeffs": None,
        "per_channel_flux": None,
        "per_channel_stokes_q": None,
        "per_channel_stokes_u": None,
        "per_channel_stokes_v": None,
        "channel_frequencies": None,
        "major_arcsec": zeros.copy(),
        "minor_arcsec": zeros.copy(),
        "pa_deg": zeros.copy(),
    }


def _mapping(tmp_path: Path, receptors: dict[str, object] | None) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    mapping = valid_config_mapping(
        tmp_path,
        baseline_selection={"correlations": "cross"},
        frequency={
            "mode": "explicit",
            "channel_frequencies_hz": [FREQUENCY_HZ],
            "channel_widths_hz": [1e6],
        },
    )
    if receptors is not None:
        mapping["receptors"] = receptors
    return mapping


def _solver_components(
    tmp_path: Path,
    receptors: dict[str, object] | None = None,
) -> tuple[SolverInstrumentView, BeamSystem, ResolvedReceptorSet]:
    simulator = Simulator.from_mapping(
        _mapping(tmp_path, receptors),
        base_dir=tmp_path,
    )
    simulator._ensure_instrument_state()
    simulator._ensure_receptor_set()
    simulator._ensure_beam_system()
    view = SolverInstrumentView.from_state(simulator._instrument_state)
    return _zero_baseline(view), simulator.beam_system, simulator.receptors


def _zero_baseline(view: SolverInstrumentView) -> SolverInstrumentView:
    return SolverInstrumentView(
        antenna_numbers=view.antenna_numbers,
        antenna_names=view.antenna_names,
        positions_enu_m=np.zeros_like(view.positions_enu_m),
        diameters_m=view.diameters_m,
        row_index_by_number=view.row_index_by_number,
        selected_pairs=view.selected_pairs,
        baseline_vectors_enu_m=np.zeros_like(view.baseline_vectors_enu_m),
    )


def _beam_matrices(
    beam_system: BeamSystem,
    view: SolverInstrumentView,
) -> dict[int, np.ndarray]:
    altitude = np.array([ALTITUDE_RAD], dtype=np.float64)
    azimuth = np.array([AZIMUTH_RAD], dtype=np.float64)
    return {
        number: np.asarray(
            beam_system.evaluate_jones(
                AntennaId(number, name),
                altitude_rad=altitude,
                azimuth_rad=azimuth,
                frequency_hz=FREQUENCY_HZ,
                time_mjd=float(OBSTIME.mjd),
            )
        )[0]
        for number, name in zip(view.antenna_numbers, view.antenna_names, strict=True)
    }


def _expected_matrix(
    beam_system: BeamSystem,
    view: SolverInstrumentView,
    stokes: tuple[float, float, float, float],
    receptor_by_number: dict[int, np.ndarray],
) -> np.ndarray:
    """Return ``(H C E)_p B (H C E)_q^H`` for the single selected baseline."""
    beams = _beam_matrices(beam_system, view)
    ant1, ant2 = view.selected_pairs[0]
    jones_p = receptor_by_number[ant1] @ beams[ant1]
    jones_q = receptor_by_number[ant2] @ beams[ant2]
    return jones_p @ plan_coherency(*stokes) @ jones_q.conj().T


def _point(
    view: SolverInstrumentView,
    beam_system: BeamSystem,
    receptors: ResolvedReceptorSet,
    stokes: tuple[float, float, float, float],
    monkeypatch: pytest.MonkeyPatch,
) -> np.ndarray:
    monkeypatch.setattr(point_visibility, "SkyCoord", _FixedAltAzSkyCoord)
    return np.asarray(
        calculate_visibility(
            instrument=view,
            beam_system=beam_system,
            source_arrays=_source_arrays(stokes),
            location=LOCATION,
            time_grid=TIME_GRID,
            frequencies=FREQUENCIES,
            backend=get_backend("numpy"),
            receptors=receptors,
        )
    )[0, 0, 0]


def _healpix(
    view: SolverInstrumentView,
    beam_system: BeamSystem,
    receptors: ResolvedReceptorSet,
    stokes: tuple[float, float, float, float],
    monkeypatch: pytest.MonkeyPatch,
    *,
    polarized: bool = True,
) -> np.ndarray:
    monkeypatch.setattr(healpix_visibility, "rayleigh_jeans_factor", lambda *_: 1.0)
    return np.asarray(
        calculate_visibility_healpix(
            sky_model=_healpix_sky(stokes, polarized=polarized),
            instrument=view,
            beam_system=beam_system,
            location=LOCATION,
            time_grid=TIME_GRID,
            frequencies=FREQUENCIES,
            backend=get_backend("numpy"),
            output_units="K.sr" if not polarized else "Jy",
            include_polarization=polarized,
            receptors=receptors,
        )
    )[0, 0, 0]


# ---------------------------------------------------------------------------
# S1 — the default configuration cannot perturb any existing result
# ---------------------------------------------------------------------------


def test_default_receptors_reproduce_the_receptor_free_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S1: linear, ``chi = 0``, ``auto`` gives ``C = H = I2`` in both paths."""
    view, beam_system, receptors = _solver_components(tmp_path)
    assert receptors.output_basis == "linear_xy"
    stokes = (2.0, 0.3, 0.2, -0.1)
    expected = _expected_matrix(
        beam_system,
        view,
        stokes,
        dict.fromkeys(view.antenna_numbers, IDENTITY),
    )

    point = _point(view, beam_system, receptors, stokes, monkeypatch)
    healpix = _healpix(view, beam_system, receptors, stokes, monkeypatch)

    np.testing.assert_allclose(point, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(healpix, expected, rtol=0.0, atol=0.0)


def test_default_receptors_reproduce_the_section_18_4_linear_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S1 restated as physics: the linear correlation table of Section 18.4."""
    view, beam_system, receptors = _solver_components(tmp_path)
    stokes_i, stokes_q, stokes_u, stokes_v = 2.0, 0.3, 0.2, -0.1
    point = _point(
        view,
        beam_system,
        receptors,
        (stokes_i, stokes_q, stokes_u, stokes_v),
        monkeypatch,
    )
    gain = _scalar_beam_product(beam_system, view)

    np.testing.assert_allclose(point[0, 0], gain * (stokes_i + stokes_q) / 2.0)
    np.testing.assert_allclose(point[1, 1], gain * (stokes_i - stokes_q) / 2.0)
    np.testing.assert_allclose(point[0, 1], gain * (stokes_u + 1.0j * stokes_v) / 2.0)
    np.testing.assert_allclose(point[1, 0], gain * (stokes_u - 1.0j * stokes_v) / 2.0)


def _scalar_beam_product(
    beam_system: BeamSystem,
    view: SolverInstrumentView,
) -> complex:
    """Return ``e_p e_q*`` for the single selected baseline (Tier 3 ``E = e I2``)."""
    beams = _beam_matrices(beam_system, view)
    ant1, ant2 = view.selected_pairs[0]
    return complex(beams[ant1][0, 0] * np.conj(beams[ant2][0, 0]))


# ---------------------------------------------------------------------------
# S4 — the circular output basis
# ---------------------------------------------------------------------------


def test_circular_receptors_reproduce_the_section_18_4_circular_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S4: ``RR=(I+V)/2``, ``RL=(Q+iU)/2``, ``LR=(Q-iU)/2``, ``LL=(I-V)/2``."""
    view, beam_system, receptors = _solver_components(
        tmp_path,
        {"default": {"basis": "circular"}},
    )
    assert receptors.output_basis == "circular_rl"
    stokes_i, stokes_q, stokes_u, stokes_v = 2.0, 0.3, 0.2, -0.1
    point = _point(
        view,
        beam_system,
        receptors,
        (stokes_i, stokes_q, stokes_u, stokes_v),
        monkeypatch,
    )
    gain = _scalar_beam_product(beam_system, view)

    np.testing.assert_allclose(point[0, 0], gain * (stokes_i + stokes_v) / 2.0)
    np.testing.assert_allclose(point[1, 1], gain * (stokes_i - stokes_v) / 2.0)
    np.testing.assert_allclose(point[0, 1], gain * (stokes_q + 1.0j * stokes_u) / 2.0)
    np.testing.assert_allclose(point[1, 0], gain * (stokes_q - 1.0j * stokes_u) / 2.0)


def test_a_positive_stokes_v_source_is_pure_rr_in_a_circular_array(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``V = +I`` source lands entirely in ``RR`` and nowhere else."""
    view, beam_system, receptors = _solver_components(
        tmp_path,
        {"default": {"basis": "circular"}},
    )
    point = _point(view, beam_system, receptors, (1.0, 0.0, 0.0, 1.0), monkeypatch)
    healpix = _healpix(view, beam_system, receptors, (1.0, 0.0, 0.0, 1.0), monkeypatch)
    gain = _scalar_beam_product(beam_system, view)

    for cube in (point, healpix):
        np.testing.assert_allclose(cube[0, 0], gain, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(cube[0, 1], 0.0, atol=1e-14)
        np.testing.assert_allclose(cube[1, 0], 0.0, atol=1e-14)
        np.testing.assert_allclose(cube[1, 1], 0.0, atol=1e-14)


def test_a_circular_run_differs_from_a_linear_run_on_a_polarized_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 5D stop condition: the configuration reaches the visibilities."""
    linear_view, linear_beams, linear_receptors = _solver_components(tmp_path / "lin")
    circular_view, circular_beams, circular_receptors = _solver_components(
        tmp_path / "circ",
        {"default": {"basis": "circular"}},
    )
    stokes = (1.0, 0.0, 0.0, 1.0)

    linear = _point(linear_view, linear_beams, linear_receptors, stokes, monkeypatch)
    circular = _point(
        circular_view,
        circular_beams,
        circular_receptors,
        stokes,
        monkeypatch,
    )
    assert not np.allclose(linear, circular)

    unpolarized = (1.0, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(
        _point(linear_view, linear_beams, linear_receptors, unpolarized, monkeypatch),
        _point(
            circular_view,
            circular_beams,
            circular_receptors,
            unpolarized,
            monkeypatch,
        ),
        rtol=1e-12,
        atol=1e-14,
    )


# ---------------------------------------------------------------------------
# S5 — unpolarized energy conservation in every basis and rotation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("basis", ["linear", "circular"])
@pytest.mark.parametrize("rotation_deg", [0.0, 30.0, 45.0, 90.0, -15.0])
def test_unpolarized_energy_is_conserved_in_every_basis_and_rotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    basis: str,
    rotation_deg: float,
) -> None:
    """S5: parallel hands sum to ``I`` and cross hands vanish (Section 18.6)."""
    view, beam_system, receptors = _solver_components(
        tmp_path,
        {"default": {"basis": basis, "feed_rotation_deg": rotation_deg}},
    )
    stokes = (3.0, 0.0, 0.0, 0.0)
    gain = _scalar_beam_product(beam_system, view)

    for cube in (
        _point(view, beam_system, receptors, stokes, monkeypatch),
        _healpix(view, beam_system, receptors, stokes, monkeypatch),
        _healpix(view, beam_system, receptors, stokes, monkeypatch, polarized=False),
    ):
        np.testing.assert_allclose(
            cube[0, 0] + cube[1, 1],
            gain * stokes[0],
            rtol=1e-12,
            atol=1e-14,
        )
        np.testing.assert_allclose(cube[0, 1], 0.0, atol=1e-14)
        np.testing.assert_allclose(cube[1, 0], 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# S7 and S8 — the rotation invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rotation_deg", [17.0, 45.0, -32.0])
def test_linear_feed_rotation_rotates_q_and_u_by_twice_chi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rotation_deg: float,
) -> None:
    """S7: ``Q' = Q cos 2chi + U sin 2chi``, ``I`` and ``V`` unchanged."""
    view, beam_system, receptors = _solver_components(
        tmp_path,
        {"default": {"feed_rotation_deg": rotation_deg}},
    )
    stokes_i, stokes_q, stokes_u, stokes_v = 2.0, 0.4, -0.3, 0.15
    cube = _point(
        view,
        beam_system,
        receptors,
        (stokes_i, stokes_q, stokes_u, stokes_v),
        monkeypatch,
    )
    gain = _scalar_beam_product(beam_system, view)
    two_chi = 2.0 * np.deg2rad(rotation_deg)
    rotated_q = stokes_q * np.cos(two_chi) + stokes_u * np.sin(two_chi)
    rotated_u = -stokes_q * np.sin(two_chi) + stokes_u * np.cos(two_chi)

    np.testing.assert_allclose(
        cube[0, 0] + cube[1, 1], gain * stokes_i, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        cube[0, 0] - cube[1, 1], gain * rotated_q, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        cube[0, 1] + cube[1, 0], gain * rotated_u, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        (cube[0, 1] - cube[1, 0]) / 1.0j,
        gain * stokes_v,
        rtol=1e-12,
        atol=1e-14,
    )


@pytest.mark.parametrize("rotation_deg", [17.0, 45.0, -32.0])
def test_circular_feed_rotation_only_phases_the_cross_hands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rotation_deg: float,
) -> None:
    """S8: ``RR``/``LL`` invariant, ``RL -> e^{-2i chi} RL``."""
    stokes = (2.0, 0.4, -0.3, 0.15)
    view, beam_system, receptors = _solver_components(
        tmp_path / "rotated",
        {"default": {"basis": "circular", "feed_rotation_deg": rotation_deg}},
    )
    reference_view, reference_beams, reference_receptors = _solver_components(
        tmp_path / "reference",
        {"default": {"basis": "circular"}},
    )
    rotated = _point(view, beam_system, receptors, stokes, monkeypatch)
    reference = _point(
        reference_view,
        reference_beams,
        reference_receptors,
        stokes,
        monkeypatch,
    )
    phase = np.exp(-2.0j * np.deg2rad(rotation_deg))

    np.testing.assert_allclose(rotated[0, 0], reference[0, 0], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(rotated[1, 1], reference[1, 1], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        rotated[0, 1], phase * reference[0, 1], rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        rotated[1, 0], np.conj(phase) * reference[1, 0], rtol=1e-12, atol=1e-14
    )


# ---------------------------------------------------------------------------
# S10 — heterogeneous arrays reported in one common output basis
# ---------------------------------------------------------------------------


def test_circular_native_in_a_linear_output_basis_matches_a_linear_array(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S10: the change of representation is exact for ideal orthogonal feeds."""
    stokes = (2.0, 0.4, -0.3, 0.15)
    linear_view, linear_beams, linear_receptors = _solver_components(tmp_path / "lin")
    converted_view, converted_beams, converted_receptors = _solver_components(
        tmp_path / "conv",
        {"default": {"basis": "circular"}, "output_basis": "linear"},
    )
    assert converted_receptors.output_basis == "linear_xy"

    np.testing.assert_allclose(
        _point(
            converted_view, converted_beams, converted_receptors, stokes, monkeypatch
        ),
        _point(linear_view, linear_beams, linear_receptors, stokes, monkeypatch),
        rtol=1e-12,
        atol=1e-14,
    )


def test_a_mixed_array_reports_one_common_output_basis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S10 for the heterogeneous case: one circular antenna, one linear."""
    stokes = (2.0, 0.4, -0.3, 0.15)
    mixed_view, mixed_beams, mixed_receptors = _solver_components(
        tmp_path / "mixed",
        {
            "overrides": [
                {"antenna": {"kind": "number", "number": 1}, "basis": "circular"}
            ],
            "output_basis": "linear",
        },
    )
    assert mixed_receptors.native_basis_counts == {"linear": 1, "circular": 1}
    linear_view, linear_beams, linear_receptors = _solver_components(tmp_path / "lin")

    np.testing.assert_allclose(
        _point(mixed_view, mixed_beams, mixed_receptors, stokes, monkeypatch),
        _point(linear_view, linear_beams, linear_receptors, stokes, monkeypatch),
        rtol=1e-12,
        atol=1e-14,
    )


def test_the_solver_applies_each_antennas_own_receptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-antenna override reaches exactly that antenna's Jones matrix."""
    stokes = (2.0, 0.4, -0.3, 0.15)
    view, beam_system, receptors = _solver_components(
        tmp_path,
        {
            "overrides": [
                {
                    "antenna": {"kind": "number", "number": 1},
                    "feed_rotation_deg": 40.0,
                }
            ]
        },
    )
    expected = _expected_matrix(
        beam_system,
        view,
        stokes,
        {
            view.antenna_numbers[0]: plan_receptor("linear", 0.0, "linear_xy"),
            view.antenna_numbers[1]: plan_receptor("linear", 40.0, "linear_xy"),
        },
    )

    np.testing.assert_allclose(
        _point(view, beam_system, receptors, stokes, monkeypatch),
        expected,
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        _healpix(view, beam_system, receptors, stokes, monkeypatch),
        expected,
        rtol=1e-12,
        atol=1e-14,
    )


# ---------------------------------------------------------------------------
# S12 — the two solver paths agree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "receptor_config",
    [
        {"default": {"basis": "circular"}},
        {"default": {"basis": "circular", "feed_rotation_deg": 22.5}},
        {"default": {"basis": "circular"}, "output_basis": "linear"},
    ],
)
def test_point_and_healpix_agree_on_a_circular_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receptor_config: dict[str, object],
) -> None:
    """S12: both paths carry the same ``H_p @ C_p``."""
    view, beam_system, receptors = _solver_components(tmp_path, receptor_config)
    stokes = (2.0, 0.4, -0.3, 0.15)

    np.testing.assert_allclose(
        _point(view, beam_system, receptors, stokes, monkeypatch),
        _healpix(view, beam_system, receptors, stokes, monkeypatch),
        rtol=1e-12,
        atol=1e-14,
    )


def test_the_scalar_healpix_path_reports_zero_cross_hands_by_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Section 19.3: the I-only path applies ``H_p @ C_p`` rather than assuming it."""
    view, beam_system, receptors = _solver_components(
        tmp_path,
        {"default": {"basis": "circular", "feed_rotation_deg": 30.0}},
    )
    stokes = (4.0, 0.0, 0.0, 0.0)
    scalar = _healpix(
        view, beam_system, receptors, stokes, monkeypatch, polarized=False
    )
    expected = _expected_matrix(
        beam_system,
        view,
        stokes,
        dict.fromkeys(
            view.antenna_numbers,
            plan_receptor("circular", 30.0, "circular_rl"),
        ),
    )

    np.testing.assert_allclose(scalar, expected, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# Now-reachable rejections and the solver contract
# ---------------------------------------------------------------------------


def test_the_parallactic_rotation_guard_is_no_longer_reachable_from_a_solver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Section 12.3's rejection lost its only trigger at Tier 7C.

    FLIPPED BY: Tier 7C, which removed the ``jones_config`` parameter
    (``Tier7JonesSciencePlan.md`` Section 33.2).  The guard fired only when that
    dictionary enabled ``P``, and no supported entry point could arrange it, so
    what this test can still assert is the *contract*: a rotated receptor now
    reaches the solver and is carried, and the guard's exact message is pinned
    directly by ``tests/characterization/test_tier7_current_behavior.py`` until
    Tier 7F replaces it with rejection R15.

    Tier 5's real protection is untouched: ``resolve_receptors`` still rejects
    every non-``fixed`` mount type, which is what actually keeps a
    time-dependent feed orientation out of the solver.
    """
    view, beam_system, receptors = _solver_components(
        tmp_path,
        {"default": {"feed_rotation_deg": 15.0}},
    )
    monkeypatch.setattr(point_visibility, "SkyCoord", _FixedAltAzSkyCoord)

    assert "jones_config" not in inspect.signature(calculate_visibility).parameters

    cube = np.asarray(
        calculate_visibility(
            instrument=view,
            beam_system=beam_system,
            source_arrays=_source_arrays((1.0, 0.0, 0.0, 0.0)),
            location=LOCATION,
            time_grid=TIME_GRID,
            frequencies=FREQUENCIES,
            backend=get_backend("numpy"),
            receptors=receptors,
        )
    )
    assert cube.shape[-2:] == (2, 2)
    assert float(np.max(np.abs(cube))) > 0.0

    # The guard itself is unchanged, and still raises when called directly.
    with pytest.raises(UnsupportedFeedGeometryError):
        point_visibility._reject_parallactic_rotation(
            {"P": {"enabled": True}}, receptors
        )


def test_every_solver_entry_point_requires_the_resolved_receptors() -> None:
    """Section 28: a required parameter with no default, on every entry point."""
    for function in (
        calculate_visibility,
        calculate_visibility_healpix,
        VisibilitySimulator.calculate_visibilities,
        RIMESimulator.calculate_visibilities,
    ):
        parameters = inspect.signature(function).parameters
        assert "receptors" in parameters, function
        assert parameters["receptors"].default is inspect.Parameter.empty, function


def test_the_solvers_reject_anything_other_than_a_resolved_receptor_set(
    tmp_path: Path,
) -> None:
    """A receptor set is canonical state, never a loose mapping."""
    view, beam_system, _receptors = _solver_components(tmp_path)

    with pytest.raises(TypeError, match="receptors"):
        calculate_visibility(
            instrument=view,
            beam_system=beam_system,
            source_arrays=_source_arrays((1.0, 0.0, 0.0, 0.0)),
            location=LOCATION,
            time_grid=TIME_GRID,
            frequencies=FREQUENCIES,
            backend=get_backend("numpy"),
            receptors={"default": {"basis": "linear"}},
        )

    with pytest.raises(TypeError, match="receptors"):
        calculate_visibility_healpix(
            sky_model=_healpix_sky((1.0, 0.0, 0.0, 0.0), polarized=False),
            instrument=view,
            beam_system=beam_system,
            location=LOCATION,
            time_grid=TIME_GRID,
            frequencies=FREQUENCIES,
            backend=get_backend("numpy"),
            receptors=None,
        )
