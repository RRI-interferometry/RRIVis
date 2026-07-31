"""Backend dispatch tests for sky conversion and materialization."""

import numpy as np
import pytest

from radiosim.backends import get_backend
from radiosim.backends.base import BackendNotAvailableError
from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky import create_from_arrays
from radiosim.core.sky.combine.pipeline import prepare_sky_model
from radiosim.core.sky.operations.convert import (
    HealpixConversionConfig,
    PointSourceHealpixInputs,
    bin_scaled_flux,
    point_sources_to_healpix_maps,
)
from radiosim.core.sky.operations.operations import materialize_healpix_model

FREQS = np.array([100e6, 120e6], dtype=np.float64)


def _get_optional_backend(name: str):
    if name == "jax":
        kwargs = {"device": "cpu"}
    elif name == "dask":
        pytest.importorskip("dask")
        kwargs = {"mode": "cpu"}
    else:
        kwargs = {}

    try:
        return get_backend(name, **kwargs)
    except BackendNotAvailableError as exc:
        pytest.skip(str(exc))


def _source_kwargs() -> dict:
    return {
        "ra_rad": np.array([0.0, 0.01, 1.0], dtype=np.float64),
        "dec_rad": np.array([0.0, 0.01, -0.2], dtype=np.float64),
        "flux": np.array([1.0, 2.0, 0.5], dtype=np.float64),
        "spectral_index": np.array([-0.7, -0.8, -0.5], dtype=np.float64),
        "spectral_coeffs": None,
        "stokes_q": np.array([0.1, 0.0, 0.02], dtype=np.float64),
        "stokes_u": np.array([0.0, 0.05, 0.01], dtype=np.float64),
        "stokes_v": np.zeros(3, dtype=np.float64),
        "rotation_measure": np.zeros(3, dtype=np.float64),
        "nside": 8,
        "frequencies": FREQS,
        "ref_frequency": np.full(3, 100e6, dtype=np.float64),
        "brightness_conversion": "rayleigh-jeans",
        "polarization_brightness_conversion": "rayleigh-jeans",
    }


def _grouped_point_kwargs(kwargs: dict):
    return PointSourceHealpixInputs(
        ra_rad=kwargs["ra_rad"],
        dec_rad=kwargs["dec_rad"],
        flux=kwargs["flux"],
        spectral_index=kwargs["spectral_index"],
        spectral_coeffs=kwargs["spectral_coeffs"],
        stokes_q=kwargs["stokes_q"],
        stokes_u=kwargs["stokes_u"],
        stokes_v=kwargs["stokes_v"],
        rotation_measure=kwargs["rotation_measure"],
        ref_frequency=kwargs["ref_frequency"],
    ), HealpixConversionConfig(
        nside=kwargs["nside"],
        frequencies=kwargs["frequencies"],
        brightness_conversion=kwargs["brightness_conversion"],
        polarization_brightness_conversion=kwargs["polarization_brightness_conversion"],
    )


def _point_model():
    kwargs = _source_kwargs()
    return create_from_arrays(
        ra_rad=kwargs["ra_rad"],
        dec_rad=kwargs["dec_rad"],
        flux=kwargs["flux"],
        spectral_index=kwargs["spectral_index"],
        stokes_q=kwargs["stokes_q"],
        stokes_u=kwargs["stokes_u"],
        stokes_v=kwargs["stokes_v"],
        reference_frequency=100e6,
        model_name="backend-point",
        precision=PrecisionConfig.standard(),
    )


def test_bin_scaled_flux_numba_matches_numpy():
    backend = _get_optional_backend("dask")
    ipix = np.array([0, 1, 0], dtype=np.int64)
    flux = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    alpha = np.array([-0.7, -0.8, -0.9], dtype=np.float64)

    expected = bin_scaled_flux(
        ipix,
        flux,
        alpha,
        None,
        120e6,
        100e6,
        4,
    )
    actual = bin_scaled_flux(
        ipix,
        flux,
        alpha,
        None,
        120e6,
        100e6,
        4,
        backend=backend,
    )

    np.testing.assert_allclose(actual, expected)


def test_bin_scaled_flux_jax_returns_backend_array():
    import jax

    backend = _get_optional_backend("jax")

    result = bin_scaled_flux(
        np.array([0, 1], dtype=np.int64),
        np.array([1.0, 2.0], dtype=np.float64),
        np.array([-0.7, -0.8], dtype=np.float64),
        None,
        120e6,
        100e6,
        4,
        backend=backend,
    )

    assert isinstance(result, jax.Array)


def test_point_sources_to_healpix_maps_numba_matches_numpy():
    backend = _get_optional_backend("dask")
    kwargs = _source_kwargs()

    sources, config = _grouped_point_kwargs(kwargs)
    expected = point_sources_to_healpix_maps(sources, config)
    actual = point_sources_to_healpix_maps(sources, config, backend=backend)

    for actual_arr, expected_arr in zip(actual[:4], expected[:4], strict=True):
        if expected_arr is None:
            assert actual_arr is None
        else:
            np.testing.assert_allclose(actual_arr, expected_arr)
    assert actual[4] == expected[4]


def test_materialize_healpix_model_numba_matches_numpy():
    backend = _get_optional_backend("dask")
    sky = _point_model()

    expected = materialize_healpix_model(sky, nside=8, frequencies=FREQS)
    actual = materialize_healpix_model(
        sky,
        nside=8,
        frequencies=FREQS,
        backend=backend,
    )

    np.testing.assert_allclose(actual.healpix.maps, expected.healpix.maps)
    np.testing.assert_allclose(actual.healpix.q_maps, expected.healpix.q_maps)
    np.testing.assert_allclose(actual.healpix.u_maps, expected.healpix.u_maps)


def test_prepare_sky_model_accepts_backend_override():
    backend = _get_optional_backend("dask")
    sky = _point_model()

    prepared = prepare_sky_model(
        [sky],
        representation="healpix_map",
        nside=8,
        frequencies=FREQS,
        backend=backend,
    )

    assert prepared.healpix is not None
    assert prepared.healpix.maps.shape[0] == len(FREQS)
