"""Backend parity tests for visibility/RIME calculations."""

import numpy as np
import pytest
from astropy import units as u
from astropy.constants import c
from astropy.coordinates import EarthLocation
from astropy.time import Time

from radiosim.backends import get_backend
from radiosim.backends.base import BackendNotAvailableError
from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky.containers.healpix import HealpixData
from radiosim.core.sky.containers.model import SkyModel
from radiosim.core.visibility import calculate_visibility
from radiosim.core.visibility_healpix import calculate_visibility_healpix

FREQS = np.array([100e6], dtype=np.float64)
WAVELENGTHS = np.array([c.value / FREQS[0]], dtype=np.float64) * u.m
LOCATION = EarthLocation.from_geodetic(0.0 * u.deg, 0.0 * u.deg, 0.0 * u.m)
OBSTIME = Time("2024-01-01T00:00:00")
ANTENNAS = {
    1: {"diameter": 14.0},
    2: {"diameter": 14.0},
}
BASELINES = {(1, 2): {"BaselineVector": np.array([10.0, 0.0, 0.0], dtype=np.float64)}}


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


def test_point_source_visibility_numba_matches_numpy():
    numpy_backend = _get_optional_backend("numpy")
    numba_backend = _get_optional_backend("numba")

    expected = calculate_visibility(
        ANTENNAS,
        BASELINES,
        _source_arrays(),
        LOCATION,
        OBSTIME,
        WAVELENGTHS,
        FREQS,
        duration_seconds=1.0,
        time_step_seconds=1.0,
        backend=numpy_backend,
    )
    actual = calculate_visibility(
        ANTENNAS,
        BASELINES,
        _source_arrays(),
        LOCATION,
        OBSTIME,
        WAVELENGTHS,
        FREQS,
        duration_seconds=1.0,
        time_step_seconds=1.0,
        backend=numba_backend,
    )

    for corr in ("XX", "XY", "YX", "YY", "I"):
        np.testing.assert_allclose(
            actual[(1, 2)][corr],
            expected[(1, 2)][corr],
            rtol=1e-10,
            atol=1e-10,
        )


def test_point_source_visibility_jax_matches_numpy():
    numpy_backend = _get_optional_backend("numpy")
    jax_backend = _get_optional_backend("jax")

    expected = calculate_visibility(
        ANTENNAS,
        BASELINES,
        _source_arrays(),
        LOCATION,
        OBSTIME,
        WAVELENGTHS,
        FREQS,
        duration_seconds=1.0,
        time_step_seconds=1.0,
        backend=numpy_backend,
    )
    actual = calculate_visibility(
        ANTENNAS,
        BASELINES,
        _source_arrays(),
        LOCATION,
        OBSTIME,
        WAVELENGTHS,
        FREQS,
        duration_seconds=1.0,
        time_step_seconds=1.0,
        backend=jax_backend,
    )

    np.testing.assert_allclose(
        actual[(1, 2)]["I"],
        expected[(1, 2)]["I"],
        rtol=1e-5,
        atol=1e-7,
    )


@pytest.mark.parametrize("polarized", [False, True])
def test_healpix_visibility_numba_matches_numpy(polarized: bool):
    sky_model = _healpix_model(polarized=polarized)
    numpy_backend = _get_optional_backend("numpy")
    numba_backend = _get_optional_backend("numba")

    expected = calculate_visibility_healpix(
        sky_model,
        antennas=ANTENNAS,
        baselines=BASELINES,
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
        antennas=ANTENNAS,
        baselines=BASELINES,
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
