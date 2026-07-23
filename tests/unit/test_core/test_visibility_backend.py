"""Backend parity tests for visibility/RIME calculations."""

import numpy as np
import pytest
from astropy import units as u
from astropy.constants import c
from astropy.coordinates import EarthLocation
from astropy.time import Time

import radiosim.core.visibility as visibility_module
from radiosim.api import Simulator
from radiosim.backends import get_backend
from radiosim.backends.base import BackendNotAvailableError
from radiosim.core.instrument import AntennaId
from radiosim.core.instrument_adapters import (
    InstrumentAdapterInvariantError,
    SolverInstrumentView,
)
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


def _solver_components(tmp_path) -> tuple[SolverInstrumentView, object]:
    data = valid_config_mapping(
        tmp_path,
        baseline_selection={"correlations": "cross"},
    )
    simulator = Simulator.from_mapping(data, base_dir=tmp_path)
    simulator._ensure_instrument_state()
    simulator._ensure_beam_system()
    return (
        SolverInstrumentView.from_state(simulator._instrument_state),
        simulator.beam_system,
    )


def _heterogeneous_solver_components(
    tmp_path,
) -> tuple[SolverInstrumentView, object]:
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
    simulator._ensure_beam_system()
    return (
        SolverInstrumentView.from_state(simulator._instrument_state),
        simulator.beam_system,
    )


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
    instrument, beam_system = _solver_components(tmp_path)

    expected = calculate_visibility(
        instrument=instrument,
        beam_system=beam_system,
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
        instrument=instrument,
        beam_system=beam_system,
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
    instrument, beam_system = _solver_components(tmp_path)

    expected = calculate_visibility(
        instrument=instrument,
        beam_system=beam_system,
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
        instrument=instrument,
        beam_system=beam_system,
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
    instrument, beam_system = _solver_components(tmp_path)

    expected = calculate_visibility_healpix(
        sky_model,
        instrument=instrument,
        beam_system=beam_system,
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
        instrument=instrument,
        beam_system=beam_system,
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
):
    view, beam_system = _heterogeneous_solver_components(tmp_path)
    backend = _get_optional_backend("numpy")

    assert view.antenna_numbers == (0, 1)
    assert view.selected_pairs == ((0, 1),)
    np.testing.assert_array_equal(view.baseline_vectors_enu_m, [[14.0, 0.0, 0.0]])
    np.testing.assert_array_equal(view.diameters_m, [12.0, 25.0])

    point_result = calculate_visibility(
        instrument=view,
        beam_system=beam_system,
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

    first = beam_system.evaluate_jones(
        AntennaId(0, "ANT0"),
        altitude_rad=np.array([1.0]),
        azimuth_rad=np.array([0.0]),
        frequency_hz=float(FREQS[0]),
        time_mjd=float(OBSTIME.mjd),
    )
    second = beam_system.evaluate_jones(
        AntennaId(1, "ANT1"),
        altitude_rad=np.array([1.0]),
        azimuth_rad=np.array([0.0]),
        frequency_hz=float(FREQS[0]),
        time_mjd=float(OBSTIME.mjd),
    )
    assert not np.array_equal(first, second)

    healpix_result = calculate_visibility_healpix(
        _healpix_model(),
        instrument=view,
        beam_system=beam_system,
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


def test_point_beam_rejects_inconsistent_solver_antenna_number(tmp_path):
    view, beam_system = _solver_components(tmp_path)
    beam = visibility_module._ResolvedBeamJones(
        beam_system=beam_system,
        instrument=view,
        altitude_rad=np.array([1.0]),
        azimuth_rad=np.array([0.0]),
        frequency_hz=float(FREQS[0]),
        time_mjd=float(OBSTIME.mjd),
    )

    with pytest.raises(InstrumentAdapterInvariantError, match="disagree"):
        beam.compute_jones_all_sources(
            antenna_idx=0,
            n_sources=1,
            freq_idx=0,
            time_idx=0,
            backend=get_backend("numpy"),
            antenna_number=7,
        )
