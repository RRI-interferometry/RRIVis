"""Backend parity tests for visibility/RIME calculations."""

import numpy as np
import pytest
from astropy import units as u
from astropy.constants import c
from astropy.coordinates import EarthLocation
from astropy.time import Time

import radiosim.core.visibility as visibility_module
import radiosim.core.visibility_healpix as healpix_visibility_module
from radiosim.api import Simulator
from radiosim.backends import get_backend
from radiosim.backends.base import BackendNotAvailableError
from radiosim.core.instrument_adapters import SolverInstrumentView
from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky.containers.healpix import HealpixData
from radiosim.core.sky.containers.model import SkyModel
from radiosim.core.visibility import calculate_visibility
from radiosim.core.visibility_healpix import calculate_visibility_healpix
from tests.fixtures.configs import valid_config_mapping

FREQS = np.array([100e6], dtype=np.float64)
WAVELENGTHS = np.array([c.value / FREQS[0]], dtype=np.float64) * u.m
LOCATION = EarthLocation.from_geodetic(0.0 * u.deg, 0.0 * u.deg, 0.0 * u.m)
OBSTIME = Time("2024-01-01T00:00:00")


def _instrument_view(tmp_path) -> SolverInstrumentView:
    data = valid_config_mapping(
        tmp_path,
        baseline_selection={"correlations": "cross"},
    )
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    simulator._ensure_instrument_state()
    return SolverInstrumentView.from_state(simulator._instrument_state)


def _heterogeneous_instrument_view(tmp_path) -> SolverInstrumentView:
    data = valid_config_mapping(
        tmp_path,
        baseline_selection={"correlations": "cross"},
        instrument={
            "diameter_overrides": [
                {
                    "antenna": {"kind": "name", "name": "ANT1"},
                    "diameter_m": 25.0,
                }
            ]
        },
    )
    lines = (tmp_path / "antennas.txt").read_text().splitlines()
    lines[1] = lines[1].removesuffix("14.0") + "12.0"
    (tmp_path / "antennas.txt").write_text("\n".join(lines) + "\n")
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    simulator._ensure_instrument_state()
    return SolverInstrumentView.from_state(simulator._instrument_state)


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


def _source_arrays() -> dict[str, np.ndarray | None]:
    lst_rad = OBSTIME.sidereal_time("apparent", longitude=LOCATION.lon).rad
    return {
        "ra_rad": np.array([lst_rad, lst_rad + 0.01], dtype=np.float64),
        "dec_rad": np.array([0.0, 0.01], dtype=np.float64),
        "flux": np.array([1.0, 0.5], dtype=np.float64),
        "spectral_index": np.array([-0.7, -0.8], dtype=np.float64),
        "stokes_q": np.array([0.1, 0.0], dtype=np.float64),
        "stokes_u": np.array([0.0, 0.05], dtype=np.float64),
        "stokes_v": np.array([0.0, 0.0], dtype=np.float64),
        "ref_freq": np.array([100e6, 100e6], dtype=np.float64),
        "rotation_measure": np.zeros(2, dtype=np.float64),
        "spectral_coeffs": None,
        "per_channel_flux": None,
        "per_channel_stokes_q": None,
        "per_channel_stokes_u": None,
        "per_channel_stokes_v": None,
        "channel_frequencies": None,
        "major_arcsec": np.zeros(2, dtype=np.float64),
        "minor_arcsec": np.zeros(2, dtype=np.float64),
        "pa_deg": np.zeros(2, dtype=np.float64),
    }


def _healpix_model(*, polarized: bool = False) -> SkyModel:
    nside = 1
    npix = 12
    maps = np.ones((1, npix), dtype=np.float64)
    q_maps = np.full((1, npix), 0.1, dtype=np.float64) if polarized else None
    u_maps = np.full((1, npix), 0.05, dtype=np.float64) if polarized else None
    v_maps = np.zeros((1, npix), dtype=np.float64) if polarized else None
    return SkyModel(
        healpix=HealpixData(
            maps=maps,
            nside=nside,
            frequencies=FREQS,
            coordinate_frame="icrs",
            q_maps=q_maps,
            u_maps=u_maps,
            v_maps=v_maps,
        ),
        model_name="backend-test",
        brightness_conversion="rayleigh-jeans",
        precision=PrecisionConfig.standard(),
    )


def test_point_source_visibility_numba_matches_numpy(tmp_path):
    numpy_backend = _get_optional_backend("numpy")
    numba_backend = _get_optional_backend("numba")

    expected = calculate_visibility(
        instrument=_instrument_view(tmp_path),
        source_arrays=_source_arrays(),
        location=LOCATION,
        obstime=OBSTIME,
        wavelengths=WAVELENGTHS,
        freqs=FREQS,
        duration_seconds=1.0,
        time_step_seconds=1.0,
        backend=numpy_backend,
    )
    actual = calculate_visibility(
        instrument=_instrument_view(tmp_path),
        source_arrays=_source_arrays(),
        location=LOCATION,
        obstime=OBSTIME,
        wavelengths=WAVELENGTHS,
        freqs=FREQS,
        duration_seconds=1.0,
        time_step_seconds=1.0,
        backend=numba_backend,
    )

    for corr in ("XX", "XY", "YX", "YY", "I"):
        np.testing.assert_allclose(
            actual[(0, 1)][corr],
            expected[(0, 1)][corr],
            rtol=1e-10,
            atol=1e-10,
        )


def test_point_source_visibility_jax_matches_numpy(tmp_path):
    numpy_backend = _get_optional_backend("numpy")
    jax_backend = _get_optional_backend("jax")

    expected = calculate_visibility(
        instrument=_instrument_view(tmp_path),
        source_arrays=_source_arrays(),
        location=LOCATION,
        obstime=OBSTIME,
        wavelengths=WAVELENGTHS,
        freqs=FREQS,
        duration_seconds=1.0,
        time_step_seconds=1.0,
        backend=numpy_backend,
    )
    actual = calculate_visibility(
        instrument=_instrument_view(tmp_path),
        source_arrays=_source_arrays(),
        location=LOCATION,
        obstime=OBSTIME,
        wavelengths=WAVELENGTHS,
        freqs=FREQS,
        duration_seconds=1.0,
        time_step_seconds=1.0,
        backend=jax_backend,
    )

    np.testing.assert_allclose(
        actual[(0, 1)]["I"],
        expected[(0, 1)]["I"],
        rtol=1e-5,
        atol=1e-7,
    )


@pytest.mark.parametrize("polarized", [False, True])
def test_healpix_visibility_numba_matches_numpy(tmp_path, polarized: bool):
    sky_model = _healpix_model(polarized=polarized)
    numpy_backend = _get_optional_backend("numpy")
    numba_backend = _get_optional_backend("numba")

    expected = calculate_visibility_healpix(
        sky_model,
        instrument=_instrument_view(tmp_path),
        location=LOCATION,
        obstime=OBSTIME,
        wavelengths=WAVELENGTHS,
        freqs=FREQS,
        duration_seconds=1.0,
        time_step_seconds=1.0,
        include_polarization=polarized,
        backend=numpy_backend,
    )
    actual = calculate_visibility_healpix(
        sky_model,
        instrument=_instrument_view(tmp_path),
        location=LOCATION,
        obstime=OBSTIME,
        wavelengths=WAVELENGTHS,
        freqs=FREQS,
        duration_seconds=1.0,
        time_step_seconds=1.0,
        include_polarization=polarized,
        backend=numba_backend,
    )

    np.testing.assert_allclose(
        actual["visibilities"],
        expected["visibilities"],
        rtol=1e-10,
        atol=1e-10,
    )


def test_point_and_healpix_paths_preserve_heterogeneous_instrument_values(
    tmp_path,
    monkeypatch,
):
    view = _heterogeneous_instrument_view(tmp_path)
    backend = _get_optional_backend("numpy")

    assert view.antenna_numbers == (0, 1)
    assert view.selected_pairs == ((0, 1),)
    np.testing.assert_array_equal(view.baseline_vectors_enu_m, [[14.0, 0.0, 0.0]])
    np.testing.assert_array_equal(view.diameters_m, [12.0, 25.0])

    point_result = calculate_visibility(
        instrument=view,
        source_arrays=_source_arrays(),
        location=LOCATION,
        obstime=OBSTIME,
        wavelengths=WAVELENGTHS,
        freqs=FREQS,
        duration_seconds=1.0,
        time_step_seconds=1.0,
        backend=backend,
    )
    assert point_result[(0, 1)]["I"].shape == (1, 1)

    point_beam: dict[str, object] = {}

    def capture_analytic_beam(**kwargs):
        point_beam.update(kwargs)
        return object()

    monkeypatch.setattr(
        visibility_module,
        "AnalyticBeamJones",
        capture_analytic_beam,
    )
    visibility_module._build_jones_chain(
        backend=backend,
        jones_config={},
        instrument=view,
        alt_rad=np.array([1.0]),
        az_rad=np.array([0.0]),
        freq=FREQS[0],
        freq_idx=0,
        n_sources=1,
        location=LOCATION,
        time_idx=0,
    )
    assert point_beam["diameter_per_antenna"] == {0: 12.0, 1: 25.0}

    healpix_diameters: list[float] = []

    def capture_healpix_beam(*, zenith_angles, diameter, **kwargs):
        healpix_diameters.append(diameter)
        return np.ones_like(zenith_angles)

    monkeypatch.setattr(
        healpix_visibility_module,
        "_compute_beam_power_pattern",
        capture_healpix_beam,
    )
    healpix_result = calculate_visibility_healpix(
        _healpix_model(),
        instrument=view,
        location=LOCATION,
        obstime=OBSTIME,
        wavelengths=WAVELENGTHS,
        freqs=FREQS,
        duration_seconds=1.0,
        time_step_seconds=1.0,
        backend=backend,
    )

    assert healpix_result["baseline_keys"] == ((0, 1),)
    assert healpix_result["visibilities"].shape == (1, 1, 1)
    assert sorted(healpix_diameters) == [12.0, 25.0]
