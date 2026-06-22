"""Tests for the 2026-05 sky-model correctness fix batch (issues 10-19)."""

from __future__ import annotations

import healpy as hp
import numpy as np
import pytest

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky import (
    HealpixData,
    MonopoleConvention,
    SkyCoverage,
    SkyModel,
    SkyProvenance,
    SourceSubtractionStatus,
    create_from_arrays,
    materialize_healpix_model,
    materialize_point_sources_model,
    with_monopole,
)
from radiosim.core.sky.containers.spectral import compute_spectral_scale
from radiosim.core.sky.io.serialization import load_skyh5, save_skyh5
from radiosim.core.sky.operations.convert import (
    HealpixConversionConfig,
    PointSourceHealpixInputs,
    healpix_map_to_point_arrays,
    point_sources_to_healpix_maps,
)


@pytest.fixture
def precision() -> PrecisionConfig:
    return PrecisionConfig.standard()


# -----------------------------------------------------------------------------
# Fix 11 — with_monopole rejects non-scalar value_k.
# -----------------------------------------------------------------------------


def _full_sky_diffuse(precision: PrecisionConfig, *, nside: int = 8) -> SkyModel:
    npix = hp.nside2npix(nside)
    return SkyModel(
        healpix=HealpixData(
            maps=np.full((2, npix), 100.0, dtype=np.float32),
            nside=nside,
            frequencies=np.array([150e6, 160e6]),
        ),
        model_name="diffuse",
        provenance=SkyProvenance(
            angular_resolution_rad=(hp.nside2resol(nside), float(np.pi)),
            sky_coverage=SkyCoverage.FULL_SKY,
            coverage_fraction=1.0,
            monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
            monopole_k=100.0,
            source_subtraction=SourceSubtractionStatus.NONE,
        ),
        precision=precision,
    )


def test_with_monopole_rejects_array_value(precision: PrecisionConfig) -> None:
    sky = _full_sky_diffuse(precision)
    with pytest.raises(TypeError, match="must be a scalar"):
        with_monopole(sky, value_k=np.array([1.0, 2.0]))


def test_with_monopole_accepts_scalar(precision: PrecisionConfig) -> None:
    sky = _full_sky_diffuse(precision)
    out = with_monopole(sky, value_k=2.5)
    assert out.provenance.monopole_k == pytest.approx(102.5)


# -----------------------------------------------------------------------------
# Fix 14 — materialize_point_sources_model updates angular resolution.
# -----------------------------------------------------------------------------


def test_materialize_point_sources_updates_angular_resolution(
    precision: PrecisionConfig,
) -> None:
    nside = 16
    npix = hp.nside2npix(nside)
    sky = SkyModel(
        healpix=HealpixData(
            maps=np.ones((2, npix), dtype=np.float32),
            nside=nside,
            frequencies=np.array([150e6, 160e6]),
        ),
        model_name="diffuse",
        provenance=SkyProvenance(
            angular_resolution_rad=(0.001, 0.002),  # something arbitrary
            sky_coverage=SkyCoverage.FULL_SKY,
            coverage_fraction=1.0,
            monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
            monopole_k=1.0,
            source_subtraction=SourceSubtractionStatus.NONE,
        ),
        precision=precision,
    )
    point = materialize_point_sources_model(sky, frequency=150e6, lossy=True)
    assert point.provenance.angular_resolution_rad is not None
    lo, hi = point.provenance.angular_resolution_rad
    assert lo == pytest.approx(float(hp.nside2resol(nside)))
    assert hi == pytest.approx(float(np.pi))


# -----------------------------------------------------------------------------
# Fix 15 — pixel-collision provenance note.
# -----------------------------------------------------------------------------


def test_materialize_healpix_records_pixel_collisions_in_notes(
    precision: PrecisionConfig,
) -> None:
    # Two sources at the exact same position to force a collision at any nside.
    sky = create_from_arrays(
        ra_rad=np.array([1.0, 1.0, 2.0]),
        dec_rad=np.array([0.5, 0.5, -0.3]),
        flux=np.array([1.0, 2.0, 3.0]),
        spectral_index=np.array([-0.7, -0.7, -0.7]),
        reference_frequency=150e6,
        precision=precision,
    )
    out = materialize_healpix_model(sky, nside=8, frequencies=np.array([150e6]))
    assert out.provenance.notes is not None
    assert "pixel_collisions=" in out.provenance.notes


def test_materialize_healpix_no_note_when_no_collisions(
    precision: PrecisionConfig,
) -> None:
    sky = create_from_arrays(
        ra_rad=np.array([0.1, 3.0]),
        dec_rad=np.array([0.5, -0.5]),
        flux=np.array([1.0, 2.0]),
        spectral_index=np.array([-0.7, -0.7]),
        reference_frequency=150e6,
        precision=precision,
    )
    out = materialize_healpix_model(sky, nside=64, frequencies=np.array([150e6]))
    notes = out.provenance.notes or ""
    assert "pixel_collisions=" not in notes


# -----------------------------------------------------------------------------
# Fix 17 — SkyProvenance round-trips through skyh5.
# -----------------------------------------------------------------------------


def test_save_and_load_skyh5_preserves_provenance(
    precision: PrecisionConfig, tmp_path
) -> None:
    nside = 8
    npix = hp.nside2npix(nside)
    prov = SkyProvenance(
        flux_completeness_jy=(0.5, float("inf")),
        flux_completeness_freq_hz=150e6,
        angular_resolution_rad=(hp.nside2resol(nside), float(np.pi)),
        sky_coverage=SkyCoverage.FULL_SKY,
        coverage_fraction=1.0,
        monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
        monopole_k=42.0,
        source_subtraction=SourceSubtractionStatus.ABOVE_THRESHOLD,
        source_subtraction_threshold_jy=1.0,
        source_subtraction_freq_hz=150e6,
        source_subtraction_method="gaussian_fit_inpaint",
        notes="round-trip-test",
    )
    sky = SkyModel(
        healpix=HealpixData(
            maps=np.full((1, npix), 5.0, dtype=np.float32),
            nside=nside,
            frequencies=np.array([150e6]),
            i_brightness_conversion="planck",
        ),
        model_name="rt",
        provenance=prov,
        precision=precision,
    )

    path = tmp_path / "rt.skyh5"
    save_skyh5(sky, str(path))
    reloaded = load_skyh5(str(path), precision=precision)

    assert (
        reloaded.provenance.source_subtraction
        == SourceSubtractionStatus.ABOVE_THRESHOLD
    )
    assert reloaded.provenance.monopole_k == pytest.approx(42.0)
    assert reloaded.provenance.notes == "round-trip-test"
    assert reloaded.provenance.flux_completeness_jy == (0.5, float("inf"))
    assert reloaded.provenance.source_subtraction_method == "gaussian_fit_inpaint"


# -----------------------------------------------------------------------------
# Fix 19 — log-polynomial spectral scale matches the standard form.
# -----------------------------------------------------------------------------


def test_compute_spectral_scale_matches_standard_log_polynomial() -> None:
    rng = np.random.default_rng(0)
    n = 50
    alpha = rng.uniform(-2.0, 0.5, size=n)
    coeffs = rng.uniform(-0.5, 0.5, size=(n, 3))
    coeffs[:, 0] = alpha  # column 0 = simple spectral index
    ref = 150e6
    for ratio_freq in (75e6, 150e6, 300e6):
        scale = compute_spectral_scale(alpha, coeffs, ratio_freq, ref)
        log_r = np.log10(ratio_freq / ref)
        expected = 10.0 ** (
            coeffs[:, 0] * log_r + coeffs[:, 1] * log_r**2 + coeffs[:, 2] * log_r**3
        )
        np.testing.assert_allclose(scale, expected, rtol=1e-12, atol=0)


# -----------------------------------------------------------------------------
# Fix 10 — polarization_brightness_conversion knob.
# -----------------------------------------------------------------------------


def test_healpix_map_to_point_planck_pol_rejects_negative_values() -> None:
    nside = 8
    npix = hp.nside2npix(nside)
    temp = np.full(npix, 5.0)
    q = np.full((1, npix), -0.5)  # negative Q
    with pytest.raises(ValueError, match="strictly positive Stokes Q"):
        healpix_map_to_point_arrays(
            temp,
            150e6,
            "planck",
            healpix_q_maps=q,
            observation_frequencies=np.array([150e6]),
            freq_index=0,
            polarization_brightness_conversion="planck",
            warn=False,
        )


def test_healpix_map_to_point_rj_pol_accepts_negative_values() -> None:
    nside = 8
    npix = hp.nside2npix(nside)
    temp = np.full(npix, 5.0)
    q = np.full((1, npix), -0.5)
    out = healpix_map_to_point_arrays(
        temp,
        150e6,
        "planck",
        healpix_q_maps=q,
        observation_frequencies=np.array([150e6]),
        freq_index=0,
        polarization_brightness_conversion="rayleigh-jeans",
        warn=False,
    )
    # Sign is preserved by RJ.
    assert np.all(out["stokes_q"] < 0)


def test_point_to_healpix_planck_pol_rejects_negative_binned_pixels() -> None:
    ra = np.array([0.5, 1.5])
    dec = np.array([0.1, -0.1])
    flux = np.array([1.0, 1.0])
    alpha = np.array([-0.7, -0.7])
    stokes_q = np.array([-0.2, -0.2])  # negative Q
    stokes_u = np.zeros(2)
    stokes_v = np.zeros(2)
    sources = PointSourceHealpixInputs(
        ra_rad=ra,
        dec_rad=dec,
        flux=flux,
        spectral_index=alpha,
        spectral_coeffs=None,
        stokes_q=stokes_q,
        stokes_u=stokes_u,
        stokes_v=stokes_v,
        rotation_measure=None,
        ref_frequency=150e6,
    )
    config = HealpixConversionConfig(
        nside=16,
        frequencies=np.array([150e6]),
        brightness_conversion="planck",
        polarization_brightness_conversion="planck",
    )
    with pytest.raises(ValueError, match="strictly positive Stokes Q"):
        point_sources_to_healpix_maps(sources, config)
