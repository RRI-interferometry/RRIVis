"""Tier 3F integration tests for one canonical visibility BeamSystem."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
from astropy.constants import c
from astropy.coordinates import EarthLocation
from astropy.time import Time

import radiosim.core.visibility as point_visibility
import radiosim.core.visibility_healpix as healpix_visibility
from radiosim.api import Simulator
from radiosim.backends import get_backend
from radiosim.core.beam import BeamSystem
from radiosim.core.instrument import AntennaId
from radiosim.core.instrument_adapters import SolverInstrumentView
from radiosim.core.polarization import stokes_to_coherency
from radiosim.core.time_grid import build_observation_time_grid
from radiosim.core.visibility import calculate_visibility
from radiosim.core.visibility_healpix import calculate_visibility_healpix
from radiosim.io.instrument_config import AntennaNumberReference
from radiosim.simulator import RIMESimulator, VisibilitySimulator
from tests.fixtures.beamfits import (
    BeamScienceVariant,
    write_scalar_efield_beamfits,
)
from tests.fixtures.configs import valid_config_mapping

FREQUENCY_HZ = 100_000_000.0
FREQUENCIES = np.array([FREQUENCY_HZ], dtype=np.float64)
WAVELENGTHS = np.array([c.value / FREQUENCY_HZ], dtype=np.float64) * u.m
LOCATION = EarthLocation.from_geodetic(0.0 * u.deg, 0.0 * u.deg, 0.0 * u.m)
OBSTIME = Time("2024-01-01T00:00:00")
TIME_GRID = build_observation_time_grid(
    start_time=OBSTIME.isot,
    duration_seconds=1.0,
    cadence_seconds=1.0,
)
ALTITUDE_RAD = np.pi / 3.0
AZIMUTH_RAD = 0.0


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


class _NonVisibleCoordinates:
    def __len__(self) -> int:
        return 2

    def transform_to(self, _frame):
        return SimpleNamespace(
            az=SimpleNamespace(rad=np.array([0.0, np.pi / 2.0])),
            alt=SimpleNamespace(rad=np.array([0.0, -0.1])),
        )


class _NonVisibleSkyCoord:
    def __init__(self, **_kwargs):
        pass

    def transform_to(self, _frame):
        return _NonVisibleCoordinates().transform_to(_frame)


class _OnePixelHealpix:
    nside = 1
    pixel_solid_angle = 1.0
    pixel_coords = _FixedPixelCoordinates()

    def __init__(
        self,
        *,
        stokes_i: float,
        stokes_q: float = 0.0,
        stokes_u: float = 0.0,
        stokes_v: float = 0.0,
    ) -> None:
        self._stokes = tuple(
            np.array([value], dtype=np.float64)
            for value in (stokes_i, stokes_q, stokes_u, stokes_v)
        )

    def get_map_at_frequency(self, _frequency):
        return self._stokes[0]

    def get_stokes_maps_at_frequency(self, _frequency):
        return self._stokes


class _NonVisibleHealpix:
    nside = 1
    pixel_solid_angle = 1.0
    pixel_coords = _NonVisibleCoordinates()

    @staticmethod
    def get_map_at_frequency(_frequency):
        return np.ones(2, dtype=np.float64)

    @staticmethod
    def get_stokes_maps_at_frequency(_frequency):
        zeros = np.zeros(2, dtype=np.float64)
        return np.ones(2, dtype=np.float64), zeros, zeros, zeros


def _beam_mapping(
    tmp_path: Path,
    beams: dict[str, object],
    *,
    healpix: bool = False,
    heterogeneous_diameters: bool = False,
) -> dict[str, object]:
    sky_source: dict[str, object] = {
        "kind": "test_sources",
        "num_sources": 1,
        "seed": 7,
    }
    if healpix:
        sky_source.update({"representation": "healpix_map", "nside": 1})
    data = valid_config_mapping(
        tmp_path,
        baseline_selection={"correlations": "cross"},
        beams=beams,
        frequency={
            "mode": "explicit",
            "channel_frequencies_hz": [FREQUENCY_HZ],
            "channel_widths_hz": [1e6],
        },
        sky_model={"sources": [sky_source]},
        visibility={
            "sky_representation": "healpix_map" if healpix else "point_sources"
        },
    )
    if heterogeneous_diameters:
        Path(data["instrument"]["source"]["path"]).write_text(
            "Name Number BeamID E N U Diameter\n"
            "ANT0 0 0 0.0 0.0 0.0 12.0\n"
            "ANT1 1 0 14.0 0.0 0.0 25.0\n",
            encoding="utf-8",
        )
    return data


def _solver_components(
    tmp_path: Path,
    beams: dict[str, object],
    *,
    heterogeneous_diameters: bool = False,
) -> tuple[Simulator, SolverInstrumentView, BeamSystem]:
    simulator = Simulator.from_mapping(
        _beam_mapping(
            tmp_path,
            beams,
            heterogeneous_diameters=heterogeneous_diameters,
        ),
        base_dir=tmp_path,
    )
    simulator._ensure_instrument_state()
    simulator._ensure_receptor_set()
    simulator._ensure_beam_system()
    return (
        simulator,
        SolverInstrumentView.from_state(simulator._instrument_state),
        simulator.beam_system,
    )


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


def _source_arrays(
    *,
    stokes_i: float,
    stokes_q: float = 0.0,
    stokes_u: float = 0.0,
    stokes_v: float = 0.0,
) -> dict[str, object]:
    zeros = np.zeros(1, dtype=np.float64)
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


def _healpix_sky(
    *,
    stokes_i: float,
    stokes_q: float = 0.0,
    stokes_u: float = 0.0,
    stokes_v: float = 0.0,
    polarized: bool,
):
    return SimpleNamespace(
        healpix=_OnePixelHealpix(
            stokes_i=stokes_i,
            stokes_q=stokes_q,
            stokes_u=stokes_u,
            stokes_v=stokes_v,
        ),
        has_polarized_healpix_maps=polarized,
        brightness_conversion="rayleigh-jeans",
        model_name="tier3f-one-pixel",
    )


def _expected_matrix(
    beam_system: BeamSystem,
    view: SolverInstrumentView,
    *,
    stokes_i: float,
    stokes_q: float,
    stokes_u: float,
    stokes_v: float,
) -> np.ndarray:
    altitude = np.array([ALTITUDE_RAD], dtype=np.float64)
    azimuth = np.array([AZIMUTH_RAD], dtype=np.float64)
    matrices = {}
    for number, name in zip(
        view.antenna_numbers,
        view.antenna_names,
        strict=True,
    ):
        matrices[number] = beam_system.evaluate_jones(
            AntennaId(number, name),
            altitude_rad=altitude,
            azimuth_rad=azimuth,
            frequency_hz=FREQUENCY_HZ,
            time_mjd=float(OBSTIME.mjd),
        )[0]
    ant1, ant2 = view.selected_pairs[0]
    coherency = stokes_to_coherency(
        np.array([stokes_i]),
        np.array([stokes_q]),
        np.array([stokes_u]),
        np.array([stokes_v]),
        xp=np,
    )[0]
    return matrices[ant1] @ coherency @ matrices[ant2].conj().T


def test_solver_signatures_require_beam_system_and_remove_legacy_inputs():
    point_parameters = inspect.signature(calculate_visibility).parameters
    healpix_parameters = inspect.signature(calculate_visibility_healpix).parameters
    abstract_parameters = inspect.signature(
        VisibilitySimulator.calculate_visibilities
    ).parameters
    rime_parameters = inspect.signature(RIMESimulator.calculate_visibilities).parameters

    for parameters in (
        point_parameters,
        healpix_parameters,
        abstract_parameters,
        rime_parameters,
    ):
        assert "beam_system" in parameters
        assert parameters["beam_system"].default is inspect.Parameter.empty
        assert "beam_manager" not in parameters
    assert "beam_config" not in healpix_parameters
    assert "kwargs" not in rime_parameters


def test_point_and_healpix_preserve_differential_complex_fits_phase(
    tmp_path,
    monkeypatch,
):
    canonical = write_scalar_efield_beamfits(
        tmp_path,
        variant=BeamScienceVariant.CANONICAL,
        filename="canonical.beamfits",
    )
    distinct = write_scalar_efield_beamfits(
        tmp_path,
        variant=BeamScienceVariant.DISTINCT,
        filename="distinct.beamfits",
    )
    beams = {
        "mode": "per_antenna_fits",
        "assignments": [
            {
                "antenna": {"kind": "number", "number": 0},
                "beam": {"kind": "fits", "path": canonical.path.name},
            },
            {
                "antenna": {"kind": "number", "number": 1},
                "beam": {"kind": "fits", "path": distinct.path.name},
            },
        ],
    }
    simulator, original_view, beam_system = _solver_components(tmp_path, beams)
    view = _zero_baseline(original_view)
    backend = get_backend("numpy")
    monkeypatch.setattr(point_visibility, "SkyCoord", _FixedAltAzSkyCoord)
    monkeypatch.setattr(healpix_visibility, "rayleigh_jeans_factor", lambda *_: 1.0)

    point = calculate_visibility(
        instrument=view,
        beam_system=beam_system,
        source_arrays=_source_arrays(
            stokes_i=2.0,
            stokes_q=0.3,
            stokes_u=0.2,
            stokes_v=-0.1,
        ),
        location=LOCATION,
        time_grid=TIME_GRID,
        frequencies=FREQUENCIES,
        backend=backend,
        receptors=simulator.receptors,
    )[0, 0, 0]
    healpix = calculate_visibility_healpix(
        sky_model=_healpix_sky(
            stokes_i=2.0,
            stokes_q=0.3,
            stokes_u=0.2,
            stokes_v=-0.1,
            polarized=True,
        ),
        instrument=view,
        beam_system=beam_system,
        location=LOCATION,
        time_grid=TIME_GRID,
        frequencies=FREQUENCIES,
        include_polarization=True,
        backend=backend,
        receptors=simulator.receptors,
    )[0, 0, 0]
    expected = _expected_matrix(
        beam_system,
        view,
        stokes_i=2.0,
        stokes_q=0.3,
        stokes_u=0.2,
        stokes_v=-0.1,
    )

    assert abs(expected[0, 0].imag) > 1e-3
    np.testing.assert_allclose(point, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(healpix, expected, rtol=1e-12, atol=1e-12)


def test_i_only_healpix_returns_full_receptor_matrix(
    tmp_path,
    monkeypatch,
):
    simulator, original_view, beam_system = _solver_components(
        tmp_path,
        {"mode": "analytic"},
        heterogeneous_diameters=True,
    )
    view = _zero_baseline(original_view)
    monkeypatch.setattr(healpix_visibility, "rayleigh_jeans_factor", lambda *_: 1.0)

    result = calculate_visibility_healpix(
        sky_model=_healpix_sky(stokes_i=2.0, polarized=False),
        instrument=view,
        beam_system=beam_system,
        location=LOCATION,
        time_grid=TIME_GRID,
        frequencies=FREQUENCIES,
        output_units="K.sr",
        backend=get_backend("numpy"),
        receptors=simulator.receptors,
    )
    expected = _expected_matrix(
        beam_system,
        view,
        stokes_i=2.0,
        stokes_q=0.0,
        stokes_u=0.0,
        stokes_v=0.0,
    )

    np.testing.assert_allclose(
        result[0, 0, 0],
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(expected[0, 0], expected[1, 1])
    assert expected[0, 1] == 0.0
    assert expected[1, 0] == 0.0


@pytest.mark.parametrize("family", ["analytic", "shared_fits", "mixed"])
@pytest.mark.parametrize(
    ("stokes_q", "stokes_u", "stokes_v"),
    [(0.0, 0.0, 0.0), (0.3, 0.2, -0.1)],
)
def test_point_healpix_full_matrix_parity_for_canonical_families(
    tmp_path,
    monkeypatch,
    family,
    stokes_q,
    stokes_u,
    stokes_v,
):
    if family == "analytic":
        beams: dict[str, object] = {"mode": "analytic"}
    else:
        fits = write_scalar_efield_beamfits(
            tmp_path,
            filename=f"parity-{family}.beamfits",
        )
        if family == "shared_fits":
            beams = {
                "mode": "shared_fits",
                "beam": {"kind": "fits", "path": fits.path.name},
            }
        else:
            beams = {
                "mode": "mixed",
                "assignments": [
                    {
                        "antenna": {"kind": "number", "number": 0},
                        "beam": {"kind": "analytic"},
                    },
                    {
                        "antenna": {"kind": "number", "number": 1},
                        "beam": {"kind": "fits", "path": fits.path.name},
                    },
                ],
            }
    simulator, original_view, beam_system = _solver_components(tmp_path, beams)
    view = _zero_baseline(original_view)
    monkeypatch.setattr(point_visibility, "SkyCoord", _FixedAltAzSkyCoord)
    monkeypatch.setattr(healpix_visibility, "rayleigh_jeans_factor", lambda *_: 1.0)

    point = calculate_visibility(
        instrument=view,
        beam_system=beam_system,
        source_arrays=_source_arrays(
            stokes_i=2.0,
            stokes_q=stokes_q,
            stokes_u=stokes_u,
            stokes_v=stokes_v,
        ),
        location=LOCATION,
        time_grid=TIME_GRID,
        frequencies=FREQUENCIES,
        backend=get_backend("numpy"),
        receptors=simulator.receptors,
    )[0, 0, 0]
    healpix = calculate_visibility_healpix(
        sky_model=_healpix_sky(
            stokes_i=2.0,
            stokes_q=stokes_q,
            stokes_u=stokes_u,
            stokes_v=stokes_v,
            polarized=True,
        ),
        instrument=view,
        beam_system=beam_system,
        location=LOCATION,
        time_grid=TIME_GRID,
        frequencies=FREQUENCIES,
        include_polarization=True,
        backend=get_backend("numpy"),
        receptors=simulator.receptors,
    )[0, 0, 0]

    np.testing.assert_allclose(point, healpix, rtol=1e-12, atol=1e-12)


def test_point_healpix_auto_and_cross_matrix_parity(
    tmp_path,
    monkeypatch,
):
    simulator, original_view, beam_system = _solver_components(
        tmp_path,
        {"mode": "analytic"},
        heterogeneous_diameters=True,
    )
    pairs = ((0, 0), (0, 1), (1, 1))
    view = SolverInstrumentView(
        antenna_numbers=original_view.antenna_numbers,
        antenna_names=original_view.antenna_names,
        positions_enu_m=np.zeros_like(original_view.positions_enu_m),
        diameters_m=original_view.diameters_m,
        row_index_by_number=original_view.row_index_by_number,
        selected_pairs=pairs,
        baseline_vectors_enu_m=np.zeros((len(pairs), 3), dtype=np.float64),
    )
    monkeypatch.setattr(point_visibility, "SkyCoord", _FixedAltAzSkyCoord)
    monkeypatch.setattr(healpix_visibility, "rayleigh_jeans_factor", lambda *_: 1.0)

    point = calculate_visibility(
        instrument=view,
        beam_system=beam_system,
        source_arrays=_source_arrays(
            stokes_i=2.0,
            stokes_q=0.3,
            stokes_u=0.2,
            stokes_v=-0.1,
        ),
        location=LOCATION,
        time_grid=TIME_GRID,
        frequencies=FREQUENCIES,
        backend=get_backend("numpy"),
        receptors=simulator.receptors,
    )
    healpix = calculate_visibility_healpix(
        sky_model=_healpix_sky(
            stokes_i=2.0,
            stokes_q=0.3,
            stokes_u=0.2,
            stokes_v=-0.1,
            polarized=True,
        ),
        instrument=view,
        beam_system=beam_system,
        location=LOCATION,
        time_grid=TIME_GRID,
        frequencies=FREQUENCIES,
        include_polarization=True,
        backend=get_backend("numpy"),
        receptors=simulator.receptors,
    )

    for index, _pair in enumerate(pairs):
        np.testing.assert_allclose(
            point[0, index, 0],
            healpix[0, index, 0],
            rtol=1e-12,
            atol=1e-12,
        )


def test_horizon_and_below_horizon_batches_skip_beam_evaluation(
    tmp_path,
    monkeypatch,
):
    simulator, view, beam_system = _solver_components(
        tmp_path,
        {"mode": "analytic"},
    )
    point_sources = _source_arrays(stokes_i=1.0)
    for key in (
        "ra_rad",
        "dec_rad",
        "flux",
        "spectral_index",
        "stokes_q",
        "stokes_u",
        "stokes_v",
        "ref_freq",
        "rotation_measure",
        "major_arcsec",
        "minor_arcsec",
        "pa_deg",
    ):
        point_sources[key] = np.repeat(np.asarray(point_sources[key]), 2)

    def forbidden(*_args, **_kwargs):
        pytest.fail("non-visible batch evaluated its BeamSystem")

    monkeypatch.setattr(point_visibility, "SkyCoord", _NonVisibleSkyCoord)
    monkeypatch.setattr(BeamSystem, "evaluate_jones", forbidden)
    point = calculate_visibility(
        instrument=view,
        beam_system=beam_system,
        source_arrays=point_sources,
        location=LOCATION,
        time_grid=TIME_GRID,
        frequencies=FREQUENCIES,
        backend=get_backend("numpy"),
        receptors=simulator.receptors,
    )
    healpix = calculate_visibility_healpix(
        sky_model=SimpleNamespace(
            healpix=_NonVisibleHealpix(),
            has_polarized_healpix_maps=True,
            brightness_conversion="rayleigh-jeans",
            model_name="tier3f-non-visible",
        ),
        instrument=view,
        beam_system=beam_system,
        location=LOCATION,
        time_grid=TIME_GRID,
        frequencies=FREQUENCIES,
        include_polarization=True,
        backend=get_backend("numpy"),
        receptors=simulator.receptors,
    )

    np.testing.assert_array_equal(point, 0.0)
    np.testing.assert_array_equal(healpix, 0.0)


def test_empty_point_source_batch_skips_fits_evaluation(
    tmp_path,
    monkeypatch,
):
    fits = write_scalar_efield_beamfits(tmp_path)
    simulator, view, beam_system = _solver_components(
        tmp_path,
        {
            "mode": "shared_fits",
            "beam": {"kind": "fits", "path": fits.path.name},
        },
    )
    source_arrays = _source_arrays(stokes_i=1.0)
    for key, value in tuple(source_arrays.items()):
        if isinstance(value, np.ndarray):
            source_arrays[key] = value[:0]

    def forbidden(*_args, **_kwargs):
        pytest.fail("empty point batch evaluated its BeamSystem")

    monkeypatch.setattr(BeamSystem, "evaluate_jones", forbidden)
    result = calculate_visibility(
        instrument=view,
        beam_system=beam_system,
        source_arrays=source_arrays,
        location=LOCATION,
        time_grid=TIME_GRID,
        frequencies=FREQUENCIES,
        backend=get_backend("numpy"),
        receptors=simulator.receptors,
    )

    np.testing.assert_array_equal(result, 0.0)


def test_healpix_evaluates_once_per_handler_id_not_numeric_equality(
    tmp_path,
    monkeypatch,
):
    first = write_scalar_efield_beamfits(tmp_path, filename="first.beamfits")
    second = write_scalar_efield_beamfits(tmp_path, filename="second.beamfits")
    beams = {
        "mode": "per_antenna_fits",
        "assignments": [
            {
                "antenna": {"kind": "number", "number": 0},
                "beam": {"kind": "fits", "path": first.path.name},
            },
            {
                "antenna": {"kind": "number", "number": 1},
                "beam": {"kind": "fits", "path": second.path.name},
            },
        ],
    }
    simulator, view, beam_system = _solver_components(tmp_path, beams)
    calls: list[AntennaId] = []
    original_evaluate = BeamSystem.evaluate_jones

    def counted_evaluate(self, antenna_id, **kwargs):
        calls.append(antenna_id)
        return original_evaluate(self, antenna_id, **kwargs)

    monkeypatch.setattr(BeamSystem, "evaluate_jones", counted_evaluate)

    calculate_visibility_healpix(
        sky_model=_healpix_sky(stokes_i=1.0, polarized=False),
        instrument=view,
        beam_system=beam_system,
        location=LOCATION,
        time_grid=TIME_GRID,
        frequencies=FREQUENCIES,
        output_units="K.sr",
        backend=get_backend("numpy"),
        receptors=simulator.receptors,
    )

    assert len(beam_system.state.handlers) == 2
    assert len(calls) == 2
    assert {antenna.number for antenna in calls} == {0, 1}


def test_healpix_shared_handler_is_evaluated_once_per_batch(
    tmp_path,
    monkeypatch,
):
    shared = write_scalar_efield_beamfits(tmp_path)
    simulator, view, beam_system = _solver_components(
        tmp_path,
        {
            "mode": "shared_fits",
            "beam": {"kind": "fits", "path": shared.path.name},
        },
    )
    calls = 0
    original_evaluate = BeamSystem.evaluate_jones

    def counted_evaluate(self, antenna_id, **kwargs):
        nonlocal calls
        calls += 1
        return original_evaluate(self, antenna_id, **kwargs)

    monkeypatch.setattr(BeamSystem, "evaluate_jones", counted_evaluate)

    calculate_visibility_healpix(
        sky_model=_healpix_sky(stokes_i=1.0, polarized=True),
        instrument=view,
        beam_system=beam_system,
        location=LOCATION,
        time_grid=TIME_GRID,
        frequencies=FREQUENCIES,
        include_polarization=True,
        backend=get_backend("numpy"),
        receptors=simulator.receptors,
    )

    assert len(beam_system.state.handlers) == 1
    assert calls == 1


@pytest.mark.parametrize(
    "beams",
    [
        {
            "mode": "analytic",
            "model": {
                "kind": "rectangular_aperture",
                "north_length_m": 14.0,
                "east_length_m": 12.0,
            },
        },
        {"mode": "shared_fits"},
        {"mode": "mixed"},
    ],
)
def test_high_level_point_run_activates_every_accepted_beam_family(
    tmp_path,
    beams,
):
    beam_path = write_scalar_efield_beamfits(
        tmp_path,
        filename=f"{beams['mode']}.beamfits",
    ).path
    if beams["mode"] == "shared_fits":
        beams = {
            "mode": "shared_fits",
            "beam": {"kind": "fits", "path": beam_path.name},
        }
    elif beams["mode"] == "mixed":
        beams = {
            "mode": "mixed",
            "assignments": [
                {
                    "antenna": {"kind": "number", "number": 0},
                    "beam": {"kind": "analytic"},
                },
                {
                    "antenna": {"kind": "number", "number": 1},
                    "beam": {"kind": "fits", "path": beam_path.name},
                },
            ],
        }
    simulator = Simulator.from_mapping(
        _beam_mapping(tmp_path, beams),
        base_dir=tmp_path,
    )

    results = simulator.run(progress=False)

    assert simulator.beam_system.state is simulator.beam_state
    assert not hasattr(simulator, "_beam_config")
    assert not hasattr(simulator, "_beam_manager")
    assert np.all(np.isfinite(results.visibilities))


def test_high_level_i_only_healpix_parallel_hands_use_half_power(
    tmp_path,
):
    simulator = Simulator.from_mapping(
        _beam_mapping(tmp_path, {"mode": "analytic"}, healpix=True),
        base_dir=tmp_path,
    )

    visibility = simulator.run(progress=False).visibilities[0, 0, 0]

    np.testing.assert_allclose(visibility[0], (visibility[0] + visibility[3]) / 2.0)
    np.testing.assert_allclose(visibility[3], (visibility[0] + visibility[3]) / 2.0)
    np.testing.assert_array_equal(visibility[1], 0.0)
    np.testing.assert_array_equal(visibility[2], 0.0)


@pytest.mark.parametrize("family", ["analytic", "shared_fits", "mixed"])
def test_high_level_healpix_run_activates_every_accepted_beam_family(
    tmp_path,
    family,
):
    if family == "analytic":
        beams: dict[str, object] = {"mode": "analytic"}
    else:
        fits = write_scalar_efield_beamfits(
            tmp_path,
            filename=f"healpix-{family}.beamfits",
        )
        if family == "shared_fits":
            beams = {
                "mode": "shared_fits",
                "beam": {"kind": "fits", "path": fits.path.name},
            }
        else:
            beams = {
                "mode": "mixed",
                "assignments": [
                    {
                        "antenna": {"kind": "number", "number": 0},
                        "beam": {"kind": "analytic"},
                    },
                    {
                        "antenna": {"kind": "number", "number": 1},
                        "beam": {"kind": "fits", "path": fits.path.name},
                    },
                ],
            }
    simulator = Simulator.from_mapping(
        _beam_mapping(tmp_path, beams, healpix=True),
        base_dir=tmp_path,
    )

    results = simulator.run(progress=False)

    assert np.all(np.isfinite(results.stokes_i()))
    np.testing.assert_allclose(
        results.stokes_i(),
        results.visibilities[..., 0] + results.visibilities[..., 3],
    )


@pytest.mark.parametrize(
    ("beams", "requires_reference"),
    [
        (
            {
                "mode": "analytic",
                "model": {
                    "kind": "rectangular_aperture",
                    "north_length_m": 14.0,
                    "east_length_m": 12.0,
                },
            },
            False,
        ),
        ({"mode": "shared_fits"}, False),
        ({"mode": "mixed"}, True),
    ],
)
def test_observability_modes_use_canonical_beam_without_renderer_work(
    tmp_path,
    monkeypatch,
    beams,
    requires_reference,
):
    beam_path = write_scalar_efield_beamfits(
        tmp_path,
        filename=f"observability-{beams['mode']}.beamfits",
    ).path
    if beams["mode"] == "shared_fits":
        beams = {
            "mode": "shared_fits",
            "beam": {"kind": "fits", "path": beam_path.name},
        }
    elif beams["mode"] == "mixed":
        beams = {
            "mode": "mixed",
            "assignments": [
                {
                    "antenna": {"kind": "number", "number": 0},
                    "beam": {"kind": "analytic"},
                },
                {
                    "antenna": {"kind": "number", "number": 1},
                    "beam": {"kind": "fits", "path": beam_path.name},
                },
            ],
        }
    simulator = Simulator.from_mapping(
        _beam_mapping(tmp_path, beams),
        base_dir=tmp_path,
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("beam-only planning constructed a renderer")

    monkeypatch.setattr(
        "radiosim.visualization.observability.ObservabilityBokehRenderer",
        forbidden,
    )

    reference = AntennaNumberReference(number=0) if requires_reference else None
    plan = simulator.plan_observability(
        reference_antenna=reference,
        grid_resolution_deg=10.0,
    )

    assert plan.reference_antenna.number == 0
    assert plan.reference_selection_reason == (
        "explicit" if requires_reference else "homogeneous_default_minimum_number"
    )
    assert plan.reference_handler_id
    assert plan.reference_scientific_fingerprint
    assert simulator._backend is None


def test_the_solver_has_no_second_beam_keyword_to_reject(
    tmp_path,
    monkeypatch,
):
    """FLIPPED BY: Tier 7C, which removed the ``jones_config`` parameter.

    The gate version asserted that ``jones_config={"beam": {}}`` was rejected
    with a bespoke ``TypeError``: an ad-hoc guard standing in for a schema, one
    of the three that ``Tier7JonesSciencePlan.md`` Section 33.2 removes with the
    dictionary itself.  The property it protected -- that ``beam_system`` is the
    solver's only beam surface -- is now structural rather than guarded, and
    that is what this asserts.  ``jones_config`` is an ordinary unexpected
    keyword now, exactly like the removed beam keywords below.
    """
    simulator, view, beam_system = _solver_components(
        tmp_path,
        {"mode": "analytic"},
    )
    monkeypatch.setattr(point_visibility, "SkyCoord", _FixedAltAzSkyCoord)

    parameters = inspect.signature(calculate_visibility).parameters
    assert "jones_config" not in parameters
    assert [name for name in parameters if "beam" in name] == ["beam_system"]

    with pytest.raises(TypeError, match="jones_config"):
        calculate_visibility(
            instrument=view,
            beam_system=beam_system,
            source_arrays=_source_arrays(stokes_i=1.0),
            location=LOCATION,
            time_grid=TIME_GRID,
            frequencies=FREQUENCIES,
            backend=get_backend("numpy"),
            jones_config={"beam": {}},
            receptors=simulator.receptors,
        )


def test_removed_solver_beam_keywords_raise_ordinary_type_errors(
    tmp_path,
    monkeypatch,
):
    simulator, view, beam_system = _solver_components(
        tmp_path,
        {"mode": "analytic"},
    )
    monkeypatch.setattr(point_visibility, "SkyCoord", _FixedAltAzSkyCoord)

    common = {
        "instrument": view,
        "beam_system": beam_system,
        "location": LOCATION,
        "obstime": OBSTIME,
        "wavelengths": WAVELENGTHS,
        "freqs": FREQUENCIES,
        "duration_seconds": 1.0,
        "time_step_seconds": 1.0,
        "backend": get_backend("numpy"),
    }
    with pytest.raises(TypeError, match="beam_manager"):
        calculate_visibility(
            source_arrays=_source_arrays(stokes_i=1.0),
            beam_manager=object(),
            **common,
            receptors=simulator.receptors,
        )
    with pytest.raises(TypeError, match="beam_config"):
        calculate_visibility_healpix(
            sky_model=_healpix_sky(stokes_i=1.0, polarized=False),
            beam_config={},
            **common,
            receptors=simulator.receptors,
        )
    with pytest.raises(TypeError, match="beam_manager"):
        RIMESimulator().calculate_visibilities(
            instrument=view,
            beam_system=beam_system,
            source_arrays=_source_arrays(stokes_i=1.0),
            frequencies=FREQUENCIES,
            backend=get_backend("numpy"),
            location=LOCATION,
            time_grid=TIME_GRID,
            beam_manager=object(),
        )
