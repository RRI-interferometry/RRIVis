"""Characterization + behavior tests for the operations-heavy cleanup.

Covers ``operations/convert.py`` (point<->HEALPix conversion) and
``operations/subtraction.py`` (multi-frequency Gaussian fit). The tests are
written as *invariants* (flux conservation, peak-pixel location, dtype/shape,
parameter recovery within tolerance) rather than golden snapshots so that a
behaviour-preserving refactor keeps them green while an incorrect refactor
violates an invariant.

Owner: operations-heavy (Phase 2, Task 2.4).
"""

from __future__ import annotations

import healpy as hp
import numpy as np
import pytest

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky.containers.constants import rayleigh_jeans_factor
from radiosim.core.sky.operations.convert import (
    bin_sources_to_flux,
    healpix_map_to_point_arrays,
    point_sources_to_healpix_maps,
)
from radiosim.core.sky.operations.subtraction import _fit_multifreq_gaussian
from radiosim.core.sky.support.healpix_geometry import (
    gnomonic_rotate,
    pixel_solid_angle,
)

# ---------------------------------------------------------------------------
# point_sources_to_healpix_maps
# ---------------------------------------------------------------------------


def _single_source_inputs(nside=8, ra=0.5, dec=0.3, flux=10.0):
    return {
        "ra_rad": np.array([ra]),
        "dec_rad": np.array([dec]),
        "flux": np.array([flux]),
        "spectral_index": np.array([0.0]),
        "spectral_coeffs": None,
        "stokes_q": None,
        "stokes_u": None,
        "stokes_v": None,
        "rotation_measure": None,
        "nside": nside,
        "frequencies": np.array([150e6]),
        "ref_frequency": 150e6,
        "brightness_conversion": "rayleigh-jeans",
    }


def test_i_only_single_source_single_freq_flux_conserved():
    """A single source's full flux density lands in exactly one pixel and the
    map integrates (in K -> Jy) back to the injected flux."""
    nside = 8
    kw = _single_source_inputs(nside=nside, flux=10.0)
    i_maps, q_maps, u_maps, v_maps, stats = point_sources_to_healpix_maps(**kw)

    npix = hp.nside2npix(nside)
    assert i_maps.shape == (1, npix)
    assert q_maps is None and u_maps is None and v_maps is None
    # Exactly one occupied pixel.
    occupied = np.flatnonzero(i_maps[0] > 0)
    assert occupied.size == 1

    # Correct peak pixel.
    expected_pix = hp.ang2pix(nside, np.pi / 2 - kw["dec_rad"][0], kw["ra_rad"][0])
    assert int(occupied[0]) == int(expected_pix)

    # Flux conservation: convert the single occupied K value back to Jy via
    # the RJ inverse of the K-forward conversion.
    omega = pixel_solid_angle(nside)
    k_val = float(i_maps[0, occupied[0]])
    jy = k_val * rayleigh_jeans_factor(150e6, omega)
    assert jy == pytest.approx(10.0, rel=1e-6)

    assert stats["n_sources"] == 1
    assert stats["n_collisions"] == 0


def test_i_only_dtype_respects_output_dtype():
    kw = _single_source_inputs()
    i_maps, *_ = point_sources_to_healpix_maps(output_dtype=np.float32, **kw)
    assert i_maps.dtype == np.float32
    i_maps64, *_ = point_sources_to_healpix_maps(output_dtype=np.float64, **kw)
    assert i_maps64.dtype == np.float64


def test_iquv_allocates_when_polarized():
    nside = 8
    kw = _single_source_inputs(nside=nside)
    kw["stokes_q"] = np.array([2.0])
    kw["stokes_u"] = np.array([1.0])
    kw["stokes_v"] = np.array([0.5])
    i_maps, q_maps, u_maps, v_maps, _ = point_sources_to_healpix_maps(**kw)
    npix = hp.nside2npix(nside)
    for m in (q_maps, u_maps, v_maps):
        assert m is not None
        assert m.shape == (1, npix)
    # Q flux conserved through K round-trip.
    omega = pixel_solid_angle(nside)
    occ = np.flatnonzero(i_maps[0] > 0)[0]
    q_jy = float(q_maps[0, occ]) * rayleigh_jeans_factor(150e6, omega)
    assert q_jy == pytest.approx(2.0, rel=1e-6)


def test_multi_freq_flat_spectrum_constant_in_jy():
    nside = 8
    kw = _single_source_inputs(nside=nside)
    kw["frequencies"] = np.array([100e6, 150e6, 200e6])
    i_maps, *_ = point_sources_to_healpix_maps(**kw)
    npix = hp.nside2npix(nside)
    assert i_maps.shape == (3, npix)
    occ = np.flatnonzero(i_maps[0] > 0)[0]
    omega = pixel_solid_angle(nside)
    for fi, f in enumerate([100e6, 150e6, 200e6]):
        jy = float(i_maps[fi, occ]) * rayleigh_jeans_factor(f, omega)
        assert jy == pytest.approx(10.0, rel=1e-6)


def test_empty_input_returns_zero_maps():
    kw = _single_source_inputs()
    kw["ra_rad"] = np.zeros(0)
    kw["dec_rad"] = np.zeros(0)
    kw["flux"] = np.zeros(0)
    kw["spectral_index"] = np.zeros(0)
    i_maps, q, u, v, stats = point_sources_to_healpix_maps(**kw)
    assert not np.any(i_maps)
    assert q is None and u is None and v is None
    assert stats["n_sources"] == 0


# --- F5: tolerance-based polarization presence ---


def test_tiny_q_does_not_trigger_iquv_allocation():
    """A 1e-30 Stokes Q must be treated as unpolarized (no Q/U/V maps)."""
    kw = _single_source_inputs()
    kw["stokes_q"] = np.array([1e-30])
    kw["stokes_u"] = np.array([0.0])
    kw["stokes_v"] = np.array([0.0])
    _, q_maps, u_maps, v_maps, _ = point_sources_to_healpix_maps(**kw)
    assert q_maps is None and u_maps is None and v_maps is None


def test_meaningful_q_does_trigger_iquv_allocation():
    kw = _single_source_inputs()
    kw["stokes_q"] = np.array([1.0])
    kw["stokes_u"] = np.array([0.0])
    kw["stokes_v"] = np.array([0.0])
    _, q_maps, _, _, _ = point_sources_to_healpix_maps(**kw)
    assert q_maps is not None


# ---------------------------------------------------------------------------
# bin_sources_to_flux (B3 routing target)
# ---------------------------------------------------------------------------


def test_bin_sources_to_flux_conserves_total():
    nside = 8
    npix = hp.nside2npix(nside)
    ipix = np.array([3, 3, 100])
    flux = np.array([1.0, 2.0, 4.0])
    out = bin_sources_to_flux(ipix, flux, np.zeros(3), None, 150e6, 150e6, npix)
    assert out.shape == (npix,)
    assert out[3] == pytest.approx(3.0)
    assert out[100] == pytest.approx(4.0)
    assert out.sum() == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# healpix_map_to_point_arrays (C4 precision threading)
# ---------------------------------------------------------------------------


def _one_pixel_map(nside=8, pix=10, k=5.0):
    npix = hp.nside2npix(nside)
    m = np.zeros(npix, dtype=np.float64)
    m[pix] = k
    return m


def test_healpix_to_point_default_float64():
    temp_map = _one_pixel_map()
    arrays = healpix_map_to_point_arrays(temp_map, 150e6, "rayleigh-jeans", warn=False)
    assert arrays["flux"].dtype == np.float64
    assert arrays["ra_rad"].dtype == np.float64
    assert arrays["flux"].size == 1


def test_healpix_to_point_precision_fast_float32():
    """A fast() precision round-trip yields float32 point arrays (C4)."""
    temp_map = _one_pixel_map()
    arrays = healpix_map_to_point_arrays(
        temp_map,
        150e6,
        "rayleigh-jeans",
        warn=False,
        precision=PrecisionConfig.fast(),
    )
    flux_dt = PrecisionConfig.fast().sky_model.get_dtype("flux")
    pos_dt = PrecisionConfig.fast().sky_model.get_dtype("source_positions")
    assert arrays["flux"].dtype == flux_dt
    assert arrays["stokes_q"].dtype == flux_dt
    assert arrays["ra_rad"].dtype == pos_dt
    assert arrays["ref_freq"].dtype == flux_dt


def test_healpix_to_point_peak_pixel_location():
    nside = 8
    pix = 42
    temp_map = _one_pixel_map(nside=nside, pix=pix, k=3.0)
    arrays = healpix_map_to_point_arrays(temp_map, 150e6, "rayleigh-jeans", warn=False)
    theta, phi = hp.pix2ang(nside, pix)
    assert arrays["ra_rad"][0] == pytest.approx(phi)
    assert arrays["dec_rad"][0] == pytest.approx(np.pi / 2 - theta)


def test_point_healpix_roundtrip_flux():
    """point -> HEALPix -> point recovers the injected flux (RJ, bit-exact I)."""
    nside = 16
    ra, dec, flux = 0.7, 0.2, 12.0
    i_maps, *_ = point_sources_to_healpix_maps(
        ra_rad=np.array([ra]),
        dec_rad=np.array([dec]),
        flux=np.array([flux]),
        spectral_index=np.array([0.0]),
        spectral_coeffs=None,
        stokes_q=None,
        stokes_u=None,
        stokes_v=None,
        rotation_measure=None,
        nside=nside,
        frequencies=np.array([150e6]),
        ref_frequency=150e6,
        brightness_conversion="rayleigh-jeans",
        output_dtype=np.float64,
    )
    arrays = healpix_map_to_point_arrays(i_maps[0], 150e6, "rayleigh-jeans", warn=False)
    assert arrays["flux"].size == 1
    assert float(arrays["flux"][0]) == pytest.approx(flux, rel=1e-6)


# ---------------------------------------------------------------------------
# gnomonic helper invariance (B4)
# ---------------------------------------------------------------------------


def test_gnomonic_rotate_zero_at_tangent_point():
    x, y = gnomonic_rotate(np.array([0.4]), np.array([0.3]), 0.4, 0.3)
    assert x[0] == pytest.approx(0.0, abs=1e-12)
    assert y[0] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# _fit_multifreq_gaussian parameter recovery (A6)
# ---------------------------------------------------------------------------


def _injected_gaussian_patch(n_freq=3, n_side_grid=15):
    """Build a tangent-plane grid + a known multi-freq elliptical Gaussian."""
    rng = np.linspace(-0.02, 0.02, n_side_grid)
    xx, yy = np.meshgrid(rng, rng)
    x = xx.ravel()
    y = yy.ravel()

    x0_true, y0_true = 0.002, -0.001
    sigma_M_true, sigma_m_true = 0.008, 0.005
    pa_true = 0.0
    amps_true = np.array([10.0, 6.0, 4.0])[:n_freq]

    z = np.empty((n_freq, x.size), dtype=np.float64)
    cos_pa = np.cos(pa_true)
    sin_pa = np.sin(pa_true)
    xr = (x - x0_true) * cos_pa + (y - y0_true) * sin_pa
    yr = -(x - x0_true) * sin_pa + (y - y0_true) * cos_pa
    shape = np.exp(-0.5 * (xr**2 / sigma_M_true**2 + yr**2 / sigma_m_true**2))
    for fi in range(n_freq):
        z[fi] = amps_true[fi] * shape
    truth = {
        "x0": x0_true,
        "y0": y0_true,
        "sigma_major": sigma_M_true,
        "sigma_minor": sigma_m_true,
        "amps": amps_true,
    }
    sigma_init = 0.006
    return x, y, z, sigma_init, truth


def test_fit_multifreq_gaussian_recovers_params():
    x, y, z, sigma_init, truth = _injected_gaussian_patch()
    fit, ok = _fit_multifreq_gaussian(x, y, z, sigma_init_rad=sigma_init)
    assert ok
    assert fit["x0"] == pytest.approx(truth["x0"], abs=1e-3)
    assert fit["y0"] == pytest.approx(truth["y0"], abs=1e-3)
    assert fit["sigma_major"] == pytest.approx(truth["sigma_major"], rel=0.15)
    assert fit["sigma_minor"] == pytest.approx(truth["sigma_minor"], rel=0.15)
    amps = np.asarray(fit["amplitudes"])
    assert amps.shape == (3,)
    np.testing.assert_allclose(amps, truth["amps"], rtol=0.1)
    # Major >= minor canonicalisation.
    assert fit["sigma_major"] >= fit["sigma_minor"]


def test_fit_multifreq_gaussian_baselines_shape():
    x, y, z, sigma_init, _ = _injected_gaussian_patch()
    fit, ok = _fit_multifreq_gaussian(x, y, z, sigma_init_rad=sigma_init)
    baselines = np.asarray(fit["baselines"])
    assert baselines.shape == (3, 3)


def test_fit_multifreq_gaussian_shape_validation():
    x = np.linspace(-0.01, 0.01, 20)
    y = np.linspace(-0.01, 0.01, 20)
    with pytest.raises(ValueError):
        _fit_multifreq_gaussian(x, y, np.zeros(20), sigma_init_rad=0.005)
    with pytest.raises(ValueError):
        _fit_multifreq_gaussian(x, y, np.zeros((3, 5)), sigma_init_rad=0.005)
