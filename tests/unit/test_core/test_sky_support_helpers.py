"""Tests for the shared core/sky support helpers (Phase 1 dedup helpers).

These helpers consolidate logic duplicated across the sky package
(backend casting, HEALPix geometry, frequency-config resolution, astropy
``Quantity`` unwrapping, coverage provenance, and point-source building).
Where a helper replaces an existing inline implementation, the test
asserts equivalence against that original implementation so the dedup is
provably behavior-preserving.
"""

from __future__ import annotations

import astropy.units as u
import healpy as hp
import numpy as np
import pytest

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky.containers import (
    PointSourceData,
    SkyCoverage,
    SkyFootprint,
)
from radiosim.core.sky.operations.region import BoxRegion
from radiosim.core.sky.support.backend_helpers import maybe_asarray
from radiosim.core.sky.support.frequencies import resolve_frequency_config
from radiosim.core.sky.support.healpix_geometry import (
    gnomonic_rotate,
    pixel_solid_angle,
    ring_ordered_row,
)
from radiosim.core.sky.support.point_builder import point_source_data_from_mapping
from radiosim.core.sky.support.provenance_coverage import coverage_provenance
from radiosim.core.sky.support.quantities import to_value

# ---------------------------------------------------------------------------
# backend_helpers.maybe_asarray
# ---------------------------------------------------------------------------


def test_maybe_asarray_none_passthrough():
    assert maybe_asarray(None, None) is None


def test_maybe_asarray_none_passthrough_with_backend():
    from radiosim.backends import get_backend

    backend = get_backend("numpy")
    assert maybe_asarray(backend, None) is None


def test_maybe_asarray_casts_with_numpy_when_backend_none():
    out = maybe_asarray(None, [1, 2, 3], dtype=np.float64)
    assert out.dtype == np.float64
    assert out.tolist() == [1.0, 2.0, 3.0]


def test_maybe_asarray_no_dtype_passthrough():
    out = maybe_asarray(None, [1, 2, 3])
    np.testing.assert_array_equal(out, np.asarray([1, 2, 3]))


def test_maybe_asarray_uses_backend_when_present():
    from radiosim.backends import get_backend

    backend = get_backend("numpy")
    out = maybe_asarray(backend, [1.0, 2.0], dtype=np.float32)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(np.asarray(out), [1.0, 2.0])


# ---------------------------------------------------------------------------
# healpix_geometry.pixel_solid_angle
# ---------------------------------------------------------------------------


def test_pixel_solid_angle_matches_formula():
    assert pixel_solid_angle(64) == pytest.approx(4 * np.pi / (12 * 64**2))


def test_pixel_solid_angle_matches_constants_helper():
    from radiosim.core.sky.containers.constants import (
        pixel_solid_angle as constants_psa,
    )

    for nside in (8, 16, 32, 64, 128):
        assert pixel_solid_angle(nside) == pytest.approx(constants_psa(nside))


# ---------------------------------------------------------------------------
# healpix_geometry.gnomonic_rotate
# ---------------------------------------------------------------------------


def _legacy_gnomonic_patch_coords(center_pix, patch_pix, nside):
    """Verbatim copy of subtraction._gnomonic_patch_coords for equivalence."""
    theta0, phi0 = hp.pix2ang(nside, center_pix)
    theta, phi = hp.pix2ang(nside, patch_pix)

    lat0 = np.pi / 2.0 - theta0
    lat = np.pi / 2.0 - theta
    dlon = phi - phi0

    cos_c = np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(dlon)
    cos_c = np.where(cos_c <= 1e-12, 1e-12, cos_c)
    x = np.cos(lat) * np.sin(dlon) / cos_c
    y = (np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(dlon)) / cos_c
    return x, y


def test_gnomonic_rotate_matches_legacy_subtraction_convention():
    nside = 64
    center = 12345
    patch = hp.query_disc(
        nside, hp.pix2vec(nside, center), 3.0 * hp.nside2resol(nside), inclusive=True
    )
    theta0, phi0 = hp.pix2ang(nside, center)
    theta, phi = hp.pix2ang(nside, patch)
    ra0 = phi0
    dec0 = np.pi / 2.0 - theta0
    ra = phi
    dec = np.pi / 2.0 - theta

    x, y = gnomonic_rotate(ra, dec, ra0, dec0)
    x_legacy, y_legacy = _legacy_gnomonic_patch_coords(center, patch, nside)

    np.testing.assert_allclose(x, x_legacy, rtol=0, atol=0)
    np.testing.assert_allclose(y, y_legacy, rtol=0, atol=0)


def test_gnomonic_rotate_tangent_point_is_origin():
    ra0 = 1.2
    dec0 = -0.3
    x, y = gnomonic_rotate(np.array([ra0]), np.array([dec0]), ra0, dec0)
    assert x[0] == pytest.approx(0.0, abs=1e-12)
    assert y[0] == pytest.approx(0.0, abs=1e-12)


def test_gnomonic_rotate_small_offset_approx_angular_separation():
    # For a small offset purely in declination the y-coordinate is ~ ddec.
    ra0, dec0 = 0.5, 0.1
    ddec = 1e-4
    x, y = gnomonic_rotate(np.array([ra0]), np.array([dec0 + ddec]), ra0, dec0)
    assert x[0] == pytest.approx(0.0, abs=1e-10)
    assert y[0] == pytest.approx(ddec, rel=1e-4)


# ---------------------------------------------------------------------------
# healpix_geometry.ring_ordered_row
# ---------------------------------------------------------------------------


def test_ring_ordered_row_scatters_sparse():
    row = ring_ordered_row(np.array([5.0, 7.0]), np.array([1, 3]), npix=4, fill=0.0)
    assert row.tolist() == [0.0, 5.0, 0.0, 7.0]


def test_ring_ordered_row_custom_fill():
    row = ring_ordered_row(np.array([5.0, 7.0]), np.array([1, 3]), npix=4, fill=np.nan)
    assert np.isnan(row[0]) and np.isnan(row[2])
    assert row[1] == 5.0 and row[3] == 7.0


def test_ring_ordered_row_matches_legacy_pyradiosky_scatter():
    # Mirror the pyradiosky._ring_ordered_row dense-scatter branch:
    #   full = np.zeros(npix); full[pix] = row
    npix = 12 * 4**2
    pix = np.array([0, 5, 17, 100, npix - 1])
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    legacy = np.zeros(npix, dtype=np.float64)
    legacy[pix] = values
    out = ring_ordered_row(values, pix, npix=npix, fill=0.0)
    np.testing.assert_array_equal(out, legacy)
    assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# frequencies.resolve_frequency_config
# ---------------------------------------------------------------------------


def test_resolve_frequency_config_requires_exactly_one():
    with pytest.raises(ValueError):
        resolve_frequency_config(frequencies=None, obs_frequency_config=None)
    with pytest.raises(ValueError):
        resolve_frequency_config(
            frequencies=np.array([100e6]),
            obs_frequency_config={"frequencies_hz": [100e6]},
        )


def test_resolve_frequency_config_explicit_array_sorted_float64():
    out = resolve_frequency_config(frequencies=np.array([150e6, 100e6]))
    assert out.tolist() == [100e6, 150e6]
    assert out.dtype == np.float64


def test_resolve_frequency_config_from_obs_config():
    config = {
        "starting_frequency": 100.0,
        "frequency_interval": 1.0,
        "frequency_bandwidth": 4.0,
        "frequency_unit": "MHz",
    }
    out = resolve_frequency_config(obs_frequency_config=config)
    assert out.dtype == np.float64
    assert out[0] == pytest.approx(100e6)
    # Ascending.
    assert np.all(np.diff(out) > 0)


def test_resolve_frequency_config_from_raw_frequencies_hz():
    config = {"frequencies_hz": [150e6, 100e6, 120e6]}
    out = resolve_frequency_config(obs_frequency_config=config)
    assert out.tolist() == [100e6, 120e6, 150e6]


# ---------------------------------------------------------------------------
# quantities.to_value
# ---------------------------------------------------------------------------


def test_to_value_unwraps_quantity_and_passes_arrays():
    assert to_value(150 * u.MHz, u.Hz) == pytest.approx(150e6)
    np.testing.assert_array_equal(to_value(np.array([1.0, 2.0]), u.Hz), [1.0, 2.0])


def test_to_value_quantity_array():
    q = np.array([100.0, 150.0]) * u.MHz
    np.testing.assert_allclose(to_value(q, u.Hz), [100e6, 150e6])


def test_to_value_plain_scalar_passthrough():
    out = to_value(5.0, u.Hz)
    assert np.asarray(out).tolist() == 5.0


# ---------------------------------------------------------------------------
# provenance_coverage.coverage_provenance
# ---------------------------------------------------------------------------


def test_coverage_provenance_full_sky():
    cov = coverage_provenance(is_full_sky=True, nside=64)
    assert cov.sky_coverage == SkyCoverage.FULL_SKY
    assert cov.coverage_fraction == 1.0
    assert cov.coverage_footprint is None


def test_coverage_provenance_partial_from_region():
    region = BoxRegion(ra_deg=180.0, dec_deg=0.0, width_deg=20.0, height_deg=20.0)
    cov = coverage_provenance(is_full_sky=False, nside=64, region=region)
    assert cov.sky_coverage == SkyCoverage.PARTIAL_SKY
    assert isinstance(cov.coverage_footprint, SkyFootprint)
    assert 0.0 < cov.coverage_fraction < 1.0
    assert cov.coverage_fraction == pytest.approx(
        cov.coverage_footprint.coverage_fraction
    )


def test_coverage_provenance_matches_diffuse_call_site():
    # Reproduce the diffuse.py / synthetic.py inline decision.
    region = BoxRegion(ra_deg=10.0, dec_deg=-30.0, width_deg=30.0, height_deg=15.0)
    # The diffuse/synthetic call sites use region.footprint() with the
    # default footprint nside; the helper defaults to the same.
    coverage_footprint = region.footprint()
    expected_coverage = SkyCoverage.PARTIAL_SKY
    expected_fraction = coverage_footprint.coverage_fraction

    cov = coverage_provenance(is_full_sky=False, region=region)
    assert cov.sky_coverage == expected_coverage
    assert cov.coverage_fraction == pytest.approx(expected_fraction)
    assert cov.coverage_footprint == coverage_footprint


def test_coverage_provenance_full_sky_no_region():
    cov = coverage_provenance(is_full_sky=True, nside=64, region=None)
    assert cov.sky_coverage == SkyCoverage.FULL_SKY
    assert cov.coverage_fraction == 1.0
    assert cov.coverage_footprint is None


# ---------------------------------------------------------------------------
# point_builder.point_source_data_from_mapping
# ---------------------------------------------------------------------------


def _basic_columns(n=3):
    return {
        "ra_rad": np.linspace(0.0, 1.0, n),
        "dec_rad": np.linspace(-0.5, 0.5, n),
        "flux": np.linspace(1.0, 3.0, n),
        "spectral_index": np.full(n, -0.7),
        "stokes_q": np.zeros(n),
        "stokes_u": np.zeros(n),
        "stokes_v": np.zeros(n),
        "ref_freq": np.full(n, 150e6),
    }


def test_point_source_data_from_mapping_builds_valid_object():
    precision = PrecisionConfig.standard()
    psd = point_source_data_from_mapping(_basic_columns(), precision)
    assert isinstance(psd, PointSourceData)
    assert psd.n_sources == 3


def test_point_source_data_from_mapping_applies_precision_dtypes():
    precision = PrecisionConfig.fast()  # source_positions/flux/si all float32
    psd = point_source_data_from_mapping(_basic_columns(), precision)
    assert psd.ra_rad.dtype == np.float32
    assert psd.dec_rad.dtype == np.float32
    assert psd.flux.dtype == np.float32
    assert psd.spectral_index.dtype == np.float32
    assert psd.stokes_q.dtype == np.float32
    assert psd.ref_freq.dtype == np.float32


def test_point_source_data_from_mapping_standard_precision_float64():
    precision = PrecisionConfig.standard()
    psd = point_source_data_from_mapping(_basic_columns(), precision)
    assert psd.ra_rad.dtype == np.float64
    assert psd.flux.dtype == np.float64


def test_point_source_data_from_mapping_optional_sub_blocks():
    precision = PrecisionConfig.standard()
    cols = _basic_columns()
    cols["rotation_measure"] = np.full(3, 2.0)
    cols["major_arcsec"] = np.full(3, 10.0)
    cols["minor_arcsec"] = np.full(3, 5.0)
    cols["pa_deg"] = np.full(3, 30.0)
    cols["source_name"] = np.array(["a", "b", "c"])
    psd = point_source_data_from_mapping(cols, precision)
    assert psd.polarization is not None
    assert psd.morphology is not None
    assert psd.metadata is not None
    np.testing.assert_array_equal(psd.morphology.major_arcsec, [10.0, 10.0, 10.0])


def test_point_source_data_from_mapping_matches_create_from_arrays():
    from radiosim.core.sky.operations.factories import create_from_arrays

    precision = PrecisionConfig.fast()
    cols = _basic_columns()
    expected = create_from_arrays(
        ra_rad=cols["ra_rad"],
        dec_rad=cols["dec_rad"],
        flux=cols["flux"],
        spectral_index=cols["spectral_index"],
        stokes_q=cols["stokes_q"],
        stokes_u=cols["stokes_u"],
        stokes_v=cols["stokes_v"],
        ref_freq=cols["ref_freq"],
        precision=precision,
    ).point
    psd = point_source_data_from_mapping(cols, precision)
    assert psd == expected
