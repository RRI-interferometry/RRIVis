"""Legacy generation characterization and canonical baseline consumers."""

from __future__ import annotations

from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
from astropy.constants import c
from astropy.coordinates import EarthLocation
from astropy.time import Time

import radiosim.core.visibility as visibility_module
import radiosim.core.visibility_healpix as healpix_visibility_module
from radiosim.api import Simulator
from radiosim.backends import get_backend
from radiosim.core.baseline import generate_baselines
from radiosim.core.instrument_adapters import SolverInstrumentView
from radiosim.core.visibility import calculate_visibility
from radiosim.core.visibility_healpix import calculate_visibility_healpix
from tests.fixtures.configs import resolved_config, valid_config_mapping


def _antenna(number, position, diameter):
    return {
        "Name": f"ANT{number}",
        "Number": number,
        "BeamID": None,
        "Position": position,
        "diameter": diameter,
    }


def test_generate_baselines_characterizes_inventory_count_and_pair_order():
    antennas = {
        "outer-nine": _antenna(9, (9.0, 0.0, 0.0), 19.0),
        "outer-two": _antenna(2, (2.0, 0.0, 0.0), 12.0),
        "outer-five": _antenna(5, (5.0, 0.0, 0.0), 15.0),
    }
    beams = {2: "beam-2", 5: "beam-5", 9: "beam-9"}
    responses = {2: "response-2", 5: "response-5", 9: "response-9"}

    baselines = generate_baselines(antennas, beams, responses)

    assert list(baselines) == [
        (2, 2),
        (2, 5),
        (2, 9),
        (5, 5),
        (5, 9),
        (9, 9),
    ]
    assert len(baselines) == 3 * (3 + 1) // 2


def test_generate_baselines_characterizes_sign_length_autocorrelation_and_mutability():
    antennas = {
        7: _antenna(7, (4.0, 2.0, -1.0), 17.0),
        2: _antenna(2, (1.0, -2.0, 3.0), 12.0),
    }
    beams = {2: "gaussian", 7: "gaussian"}

    baselines = generate_baselines(antennas, beams, beams)

    np.testing.assert_array_equal(baselines[(2, 2)]["BaselineVector"], [0, 0, 0])
    assert baselines[(2, 2)]["Length"] == 0.0
    np.testing.assert_array_equal(
        baselines[(2, 7)]["BaselineVector"],
        np.array([3.0, 4.0, -4.0]),
    )
    assert baselines[(2, 7)]["Length"] == pytest.approx(np.sqrt(41.0))
    assert isinstance(baselines[(2, 7)]["BaselineVector"], np.ndarray)
    assert isinstance(baselines[(2, 7)]["Length"], np.floating)
    assert baselines[(2, 7)]["BaselineVector"].flags.writeable

    original_length = baselines[(2, 7)]["Length"]
    baselines[(2, 7)]["BaselineVector"][0] = 99.0
    # Length is a separate NumPy scalar snapshot, not a live vector view.
    assert baselines[(2, 7)]["Length"] == original_length


def test_generate_baselines_nested_number_collision_silently_collapses_inventory():
    """Undesirable legacy behavior: the later nested Number silently wins."""
    antennas = {
        "first": _antenna(4, (1.0, 0.0, 0.0), 10.0),
        "second": _antenna(4, (9.0, 0.0, 0.0), 20.0),
    }

    baselines = generate_baselines(
        antennas,
        beams_per_antenna={4: "second-beam"},
        beam_response_per_antenna={4: "second-response"},
    )

    assert list(baselines) == [(4, 4)]
    assert baselines[(4, 4)]["D1D2"] == "20.0_20.0"


def test_generate_baselines_characterizes_exact_opaque_legacy_strings():
    antennas = {
        1: _antenna(1, (0.0, 0.0, 0.0), 12.5),
        3: _antenna(3, (1.0, 0.0, 0.0), 25.0),
    }

    baseline = generate_baselines(
        antennas,
        beams_per_antenna={1: "beam-a", 3: "beam-b"},
        beam_response_per_antenna={1: "response-a", 3: "response-b"},
    )[(1, 3)]

    assert baseline["D1D2"] == "12.5_25.0"
    assert baseline["BT1BT2"] == "beam-a_beam-b"
    assert baseline["A1A2"] == "response-a_response-b"


def test_generate_baselines_characterizes_required_metadata_error_wrapping():
    missing_number = {
        1: {
            "Name": "BROKEN",
            "Position": (0.0, 0.0, 0.0),
        }
    }
    with pytest.raises(KeyError) as missing_number_error:
        generate_baselines(missing_number, {}, {})
    assert isinstance(missing_number_error.value.__cause__, KeyError)

    antennas = {
        1: _antenna(1, (0.0, 0.0, 0.0), 12.0),
        2: _antenna(2, (1.0, 0.0, 0.0), 12.0),
    }
    with pytest.raises(ValueError) as missing_beam_error:
        generate_baselines(
            antennas,
            beams_per_antenna={1: "only-one"},
            beam_response_per_antenna={1: "only-one"},
        )
    assert isinstance(missing_beam_error.value.__cause__, KeyError)


def _point_source_arrays(frequency_hz: float) -> dict[str, object]:
    zeros = np.zeros(1, dtype=np.float64)
    return {
        "ra_rad": zeros.copy(),
        "dec_rad": zeros.copy(),
        "flux": np.array([2.0]),
        "spectral_index": zeros.copy(),
        "stokes_q": zeros.copy(),
        "stokes_u": zeros.copy(),
        "stokes_v": zeros.copy(),
        "ref_freq": np.array([frequency_hz]),
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


def _solver_instrument_view(
    tmp_path,
    *,
    diameters: tuple[float, float] = (12.0, 25.0),
) -> SolverInstrumentView:
    data = valid_config_mapping(
        tmp_path,
        baseline_selection={"correlations": "cross"},
    )
    (tmp_path / "antennas.txt").write_text(
        "Name Number BeamID E N U Diameter\n"
        f"ANT1 1 0 0.0 0.0 0.0 {diameters[0]}\n"
        f"ANT2 2 0 1.0 0.0 0.0 {diameters[1]}\n",
        encoding="utf-8",
    )
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    simulator._ensure_instrument_state()
    return SolverInstrumentView.from_state(simulator._instrument_state)


class _FixedAltAzSkyCoord:
    def __init__(self, **_kwargs):
        pass

    def transform_to(self, _frame):
        return SimpleNamespace(
            az=SimpleNamespace(rad=np.array([np.pi / 2.0])),
            alt=SimpleNamespace(rad=np.array([np.pi / 3.0])),
        )


class _IdentityJonesChain:
    def compute_antenna_jones_all_sources(self, *, n_sources, **_kwargs):
        return np.broadcast_to(np.eye(2, dtype=np.complex128), (n_sources, 2, 2))


def test_point_solver_uses_only_baseline_vector_with_current_negative_phase(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(visibility_module, "SkyCoord", _FixedAltAzSkyCoord)
    monkeypatch.setattr(
        visibility_module,
        "_build_jones_chain",
        lambda *_args, **_kwargs: _IdentityJonesChain(),
    )
    wavelength_m = 2.0
    frequency_hz = c.value / wavelength_m
    # With alt=60 deg and az=90 deg, l=0.5. The one-metre East vector is
    # 0.5 wavelengths, so b.l=0.25 and exp(-2*pi*i*b.l) is exactly -i.
    instrument = _solver_instrument_view(tmp_path)

    result = calculate_visibility(
        instrument=instrument,
        source_arrays=_point_source_arrays(frequency_hz),
        location=EarthLocation.from_geodetic(0.0, 0.0, 0.0),
        obstime=Time("2024-01-01T00:00:00"),
        wavelengths=np.array([wavelength_m]) * u.m,
        freqs=np.array([frequency_hz]),
        duration_seconds=1.0,
        time_step_seconds=1.0,
        return_correlations=False,
        backend=get_backend("numpy"),
    )

    matrix = result[(1, 2)][0, 0]
    assert matrix[0, 0] == pytest.approx(-1j)
    assert matrix[1, 1] == pytest.approx(-1j)
    assert matrix[0, 1] == 0.0


class _FixedPixelCoordinates:
    def __len__(self):
        return 1

    def transform_to(self, _frame):
        return SimpleNamespace(
            az=SimpleNamespace(rad=np.array([np.pi / 2.0])),
            alt=SimpleNamespace(rad=np.array([np.pi / 3.0])),
        )


class _OnePixelHealpix:
    nside = 1
    pixel_solid_angle = 1.0
    pixel_coords = _FixedPixelCoordinates()

    @staticmethod
    def get_map_at_frequency(_frequency):
        return np.array([1.0])


def test_healpix_solver_uses_canonical_vector_phase_and_exact_diameters(
    tmp_path,
    monkeypatch,
):
    captured_diameters: list[float] = []

    def unit_beam(*, zenith_angles, diameter, **_kwargs):
        captured_diameters.append(diameter)
        return np.ones_like(zenith_angles)

    monkeypatch.setattr(
        healpix_visibility_module,
        "_compute_beam_power_pattern",
        unit_beam,
    )
    sky_model = SimpleNamespace(
        healpix=_OnePixelHealpix(),
        has_polarized_healpix_maps=False,
        brightness_conversion="rayleigh-jeans",
        model_name="one-pixel-characterization",
    )
    wavelength_m = 2.0
    frequency_hz = c.value / wavelength_m
    instrument = _solver_instrument_view(tmp_path)

    result = calculate_visibility_healpix(
        sky_model=sky_model,
        instrument=instrument,
        location=EarthLocation.from_geodetic(0.0, 0.0, 0.0),
        obstime=Time("2024-01-01T00:00:00"),
        wavelengths=np.array([wavelength_m]) * u.m,
        freqs=np.array([frequency_hz]),
        duration_seconds=1.0,
        time_step_seconds=1.0,
        output_units="K.sr",
        backend=get_backend("numpy"),
    )

    assert result["visibilities"][0, 0, 0] == pytest.approx(-1j)
    assert sorted(captured_diameters) == [12.0, 25.0]


def test_point_beam_receives_complete_canonical_diameter_map(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def capture_analytic_beam(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        visibility_module,
        "AnalyticBeamJones",
        capture_analytic_beam,
    )

    visibility_module._build_jones_chain(
        backend=get_backend("numpy"),
        jones_config={},
        instrument=_solver_instrument_view(tmp_path),
        alt_rad=np.array([1.0]),
        az_rad=np.array([0.0]),
        freq=100e6,
        freq_idx=0,
        n_sources=1,
        location=None,
        time_idx=0,
    )

    assert captured["diameter"] == 12.0
    assert captured["diameter_per_antenna"] == {1: 12.0, 2: 25.0}


def test_memory_estimation_consumes_only_current_inventory_counts(tmp_path):
    captured: dict[str, int] = {}

    class FakeSimulatorStrategy:
        def get_memory_estimate(self, **kwargs):
            captured.update(kwargs)
            return {"total_bytes": 1000, "total_human": "1.0 KB"}

    simulator = Simulator(resolved_config(tmp_path).runtime)
    simulator._ensure_instrument_state()
    simulator._is_setup = True
    simulator._source_arrays = {"ra_rad": np.zeros(4)}
    simulator._frequencies_hz = np.zeros(5)
    simulator._simulator = FakeSimulatorStrategy()
    simulator._backend = SimpleNamespace(precision=None)

    estimate = simulator.get_memory_estimate()

    assert captured == {
        "n_antennas": 2,
        "n_baselines": 3,
        "n_sources": 4,
        "n_frequencies": 5,
    }
    assert estimate["precision_factor"] == 1.0
