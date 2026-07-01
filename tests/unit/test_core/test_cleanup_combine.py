"""Characterization + behavior tests for the combine owner cleanup.

These pin the invariants of ``combine_healpix`` (both brightness
conversions), ``concat_point_sources`` (precision + ragged columns),
``merge_provenance`` (now in ``combine.merge``), and the disjointness
alpha tunable. They are written as invariant / known-analytic assertions
(shapes, dtypes, additivity, energy/peak, round-trip consistency) so an
incorrect refactor violates an invariant rather than a golden snapshot.
"""

from __future__ import annotations

import numpy as np
import pytest

from radiosim.core.precision import PrecisionConfig
from radiosim.core.sky import (
    MonopoleConvention,
    SkyModel,
    SkyProvenance,
    create_from_arrays,
)
from radiosim.core.sky.combine.concat import concat_point_sources
from radiosim.core.sky.combine.disjointness import check_physical_disjointness
from radiosim.core.sky.combine.healpix import combine_healpix
from radiosim.core.sky.containers import (
    HealpixData,
    SkyCoverage,
    SourceSubtractionStatus,
)
from radiosim.core.sky.containers.constants import (
    BrightnessConversion,
    brightness_temp_to_flux_density,
)

# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _precision() -> PrecisionConfig:
    return PrecisionConfig.standard()


def _healpix_sky(
    *,
    nside: int,
    frequencies: np.ndarray,
    fill: float,
    name: str,
    conversion: BrightnessConversion = BrightnessConversion.PLANCK,
    precision: PrecisionConfig | None = None,
) -> SkyModel:
    """A full-sky HEALPix model whose Stokes-I maps are a constant ``fill`` K."""
    import healpy as hp

    npix = hp.nside2npix(nside)
    maps = np.full((len(frequencies), npix), fill, dtype=np.float64)
    return SkyModel(
        healpix=HealpixData(
            maps=maps,
            nside=nside,
            frequencies=np.asarray(frequencies, dtype=np.float64),
            coordinate_frame="icrs",
            i_brightness_conversion=conversion.value,
        ),
        model_name=name,
        brightness_conversion=conversion,
        provenance=SkyProvenance(
            monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
        ),
        precision=precision if precision is not None else _precision(),
    )


def _point_sky(
    *,
    ra: float,
    dec: float,
    flux: float,
    ref_freq: float,
    name: str,
    precision: PrecisionConfig | None = None,
    conversion: BrightnessConversion = BrightnessConversion.PLANCK,
) -> SkyModel:
    p = precision if precision is not None else _precision()
    return create_from_arrays(
        ra_rad=np.array([ra]),
        dec_rad=np.array([dec]),
        flux=np.array([flux]),
        spectral_index=np.array([0.0]),
        stokes_q=np.array([0.0]),
        stokes_u=np.array([0.0]),
        stokes_v=np.array([0.0]),
        ref_freq=np.array([ref_freq]),
        reference_frequency=ref_freq,
        precision=p,
        model_name=name,
        brightness_conversion=conversion,
        provenance=SkyProvenance(
            monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
        ),
    )


# --------------------------------------------------------------------------- #
# combine_healpix — shared invariants for both conversions
# --------------------------------------------------------------------------- #


CONVERSIONS = [BrightnessConversion.RAYLEIGH_JEANS, BrightnessConversion.PLANCK]


@pytest.mark.parametrize("conversion", CONVERSIONS)
@pytest.mark.parametrize("n_freq", [1, 3])
def test_combine_healpix_shapes_and_keys(conversion, n_freq):
    nside = 8
    freqs = np.linspace(100e6, 200e6, n_freq)
    a = _healpix_sky(
        nside=nside, frequencies=freqs, fill=2.0, name="a", conversion=conversion
    )
    b = _healpix_sky(
        nside=nside, frequencies=freqs, fill=3.0, name="b", conversion=conversion
    )

    out = combine_healpix(
        [a, b],
        ref_nside=nside,
        ref_freqs=freqs,
        ref_frequency=None,
        brightness_conversion=conversion,
        precision=_precision(),
    )
    import healpy as hp

    npix = hp.nside2npix(nside)
    # Emitted-key contract (G1): these exact keys are present.
    for key in (
        "healpix_maps",
        "healpix_q_maps",
        "healpix_u_maps",
        "healpix_v_maps",
        "healpix_nside",
        "observation_frequencies",
        "coordinate_frame",
        "ordering",
        "reference_frequency",
    ):
        assert key in out
    assert out["healpix_maps"].shape == (n_freq, npix)
    assert out["healpix_nside"] == nside
    np.testing.assert_array_equal(out["observation_frequencies"], freqs)
    # No polarization present -> pol cubes are None.
    assert out["healpix_q_maps"] is None


@pytest.mark.parametrize("conversion", CONVERSIONS)
def test_combine_healpix_two_diffuse_additive(conversion):
    """Two constant diffuse maps add: combined I == per-pixel sum (both paths).

    For RJ this is exactly linear in T_b; for Planck the Jy round-trip of two
    equal-frequency constant maps still recovers fill_a + fill_b in T_b because
    T->Jy->(sum)->T is consistent at fixed frequency/solid-angle.
    """
    nside = 8
    freqs = np.array([150e6])
    fill_a, fill_b = 2.0, 5.0
    a = _healpix_sky(
        nside=nside, frequencies=freqs, fill=fill_a, name="a", conversion=conversion
    )
    b = _healpix_sky(
        nside=nside, frequencies=freqs, fill=fill_b, name="b", conversion=conversion
    )

    out = combine_healpix(
        [a, b],
        ref_nside=nside,
        ref_freqs=freqs,
        ref_frequency=None,
        brightness_conversion=conversion,
        precision=_precision(),
    )
    result = np.asarray(out["healpix_maps"][0], dtype=np.float64)
    # RJ is exactly linear; Planck is additive only in the deep Rayleigh-Jeans
    # limit (150 MHz, few-K maps), so allow a small nonlinearity tolerance.
    rtol = 1e-5 if conversion == BrightnessConversion.RAYLEIGH_JEANS else 1e-2
    np.testing.assert_allclose(result, fill_a + fill_b, rtol=rtol)


def test_combine_healpix_rj_linear_scaling():
    """RJ path: scaling every input map by k scales the combined map by k."""
    nside = 8
    freqs = np.array([120e6, 180e6])
    conv = BrightnessConversion.RAYLEIGH_JEANS
    a = _healpix_sky(
        nside=nside, frequencies=freqs, fill=1.0, name="a", conversion=conv
    )
    b = _healpix_sky(
        nside=nside, frequencies=freqs, fill=4.0, name="b", conversion=conv
    )
    a2 = _healpix_sky(
        nside=nside, frequencies=freqs, fill=10.0, name="a2", conversion=conv
    )
    b2 = _healpix_sky(
        nside=nside, frequencies=freqs, fill=40.0, name="b2", conversion=conv
    )

    base = combine_healpix(
        [a, b],
        ref_nside=nside,
        ref_freqs=freqs,
        ref_frequency=None,
        brightness_conversion=conv,
        precision=_precision(),
    )
    scaled = combine_healpix(
        [a2, b2],
        ref_nside=nside,
        ref_freqs=freqs,
        ref_frequency=None,
        brightness_conversion=conv,
        precision=_precision(),
    )
    np.testing.assert_allclose(
        np.asarray(scaled["healpix_maps"], dtype=np.float64),
        10.0 * np.asarray(base["healpix_maps"], dtype=np.float64),
        rtol=1e-5,
    )


@pytest.mark.parametrize("conversion", CONVERSIONS)
def test_combine_healpix_with_point_contribution_raises_peak(conversion):
    """Adding a point source raises the peak above the smooth-diffuse level."""
    nside = 16
    freqs = np.array([150e6])
    diffuse = _healpix_sky(
        nside=nside, frequencies=freqs, fill=2.0, name="d", conversion=conversion
    )
    point = _point_sky(
        ra=0.3, dec=0.2, flux=5.0, ref_freq=150e6, name="p", conversion=conversion
    )

    with_pt = combine_healpix(
        [diffuse, point],
        ref_nside=nside,
        ref_freqs=freqs,
        ref_frequency=150e6,
        brightness_conversion=conversion,
        precision=_precision(),
    )
    without_pt = combine_healpix(
        [diffuse],
        ref_nside=nside,
        ref_freqs=freqs,
        ref_frequency=150e6,
        brightness_conversion=conversion,
        precision=_precision(),
    )
    peak_with = float(np.max(with_pt["healpix_maps"][0]))
    peak_without = float(np.max(without_pt["healpix_maps"][0]))
    assert peak_with > peak_without
    # The point source deposits positive flux into exactly one pixel; the
    # rest of the sky equals the diffuse-only result.
    n_above = int(np.sum(np.asarray(with_pt["healpix_maps"][0]) > peak_without + 1e-6))
    assert n_above == 1


@pytest.mark.parametrize("conversion", CONVERSIONS)
def test_combine_healpix_point_flux_conserved_in_jy(conversion):
    """The injected point-source flux (Jy) is conserved across the round-trip.

    Convert the single hot pixel's combined T_b back to Jy and subtract the
    diffuse background's Jy contribution; the remainder equals the source flux.
    """
    nside = 16
    freqs = np.array([150e6])
    fill = 2.0
    src_flux = 5.0
    diffuse = _healpix_sky(
        nside=nside, frequencies=freqs, fill=fill, name="d", conversion=conversion
    )
    point = _point_sky(
        ra=0.3, dec=0.2, flux=src_flux, ref_freq=150e6, name="p", conversion=conversion
    )

    out = combine_healpix(
        [diffuse, point],
        ref_nside=nside,
        ref_freqs=freqs,
        ref_frequency=150e6,
        brightness_conversion=conversion,
        precision=_precision(),
    )
    import healpy as hp

    npix = hp.nside2npix(nside)
    omega = 4 * np.pi / npix
    method = conversion.value
    tb = np.asarray(out["healpix_maps"][0], dtype=np.float64)
    hot = int(np.argmax(tb))
    jy_hot = brightness_temp_to_flux_density(
        np.array([tb[hot]]), 150e6, omega, method=method
    )[0]
    jy_bg = brightness_temp_to_flux_density(
        np.array([fill]), 150e6, omega, method=method
    )[0]
    np.testing.assert_allclose(jy_hot - jy_bg, src_flux, rtol=1e-4)


def test_combine_healpix_planck_low_freq_matches_rj():
    """Planck consistency at low freq: Planck ~ RJ in the hν << kT regime.

    At 50 MHz and modest T_b the Planck and RJ combine paths must agree to
    high relative precision (the classic Rayleigh-Jeans limit).
    """
    nside = 8
    freqs = np.array([50e6])
    a_rj = _healpix_sky(
        nside=nside,
        frequencies=freqs,
        fill=20.0,
        name="a",
        conversion=BrightnessConversion.RAYLEIGH_JEANS,
    )
    b_rj = _healpix_sky(
        nside=nside,
        frequencies=freqs,
        fill=30.0,
        name="b",
        conversion=BrightnessConversion.RAYLEIGH_JEANS,
    )
    a_pl = _healpix_sky(
        nside=nside,
        frequencies=freqs,
        fill=20.0,
        name="a",
        conversion=BrightnessConversion.PLANCK,
    )
    b_pl = _healpix_sky(
        nside=nside,
        frequencies=freqs,
        fill=30.0,
        name="b",
        conversion=BrightnessConversion.PLANCK,
    )

    rj = combine_healpix(
        [a_rj, b_rj],
        ref_nside=nside,
        ref_freqs=freqs,
        ref_frequency=None,
        brightness_conversion=BrightnessConversion.RAYLEIGH_JEANS,
        precision=_precision(),
    )
    pl = combine_healpix(
        [a_pl, b_pl],
        ref_nside=nside,
        ref_freqs=freqs,
        ref_frequency=None,
        brightness_conversion=BrightnessConversion.PLANCK,
        precision=_precision(),
    )
    np.testing.assert_allclose(
        np.asarray(pl["healpix_maps"], dtype=np.float64),
        np.asarray(rj["healpix_maps"], dtype=np.float64),
        rtol=1e-3,
    )


@pytest.mark.parametrize("conversion", CONVERSIONS)
def test_combine_healpix_output_dtype_follows_precision(conversion):
    nside = 8
    freqs = np.array([150e6])
    fast = PrecisionConfig.fast()
    a = _healpix_sky(
        nside=nside,
        frequencies=freqs,
        fill=2.0,
        name="a",
        conversion=conversion,
        precision=fast,
    )
    b = _healpix_sky(
        nside=nside,
        frequencies=freqs,
        fill=3.0,
        name="b",
        conversion=conversion,
        precision=fast,
    )
    out = combine_healpix(
        [a, b],
        ref_nside=nside,
        ref_freqs=freqs,
        ref_frequency=None,
        brightness_conversion=conversion,
        precision=fast,
    )
    expected = fast.sky_model.get_dtype("healpix_maps")
    assert np.asarray(out["healpix_maps"]).dtype == expected


def test_combine_healpix_propagates_ref_frequency():
    """G2: the accepted ref_frequency surfaces in the output dict."""
    nside = 8
    freqs = np.array([150e6])
    a = _healpix_sky(nside=nside, frequencies=freqs, fill=2.0, name="a")
    out = combine_healpix(
        [a],
        ref_nside=nside,
        ref_freqs=freqs,
        ref_frequency=150e6,
        brightness_conversion=BrightnessConversion.PLANCK,
        precision=_precision(),
    )
    assert out["reference_frequency"] == 150e6


# --------------------------------------------------------------------------- #
# concat_point_sources — precision + ragged columns
# --------------------------------------------------------------------------- #


def test_concat_honors_precision_fast_yields_float32():
    """C2: concat under fast() returns float32 core arrays."""
    fast = PrecisionConfig.fast()
    a = _point_sky(ra=0.1, dec=0.2, flux=1.0, ref_freq=150e6, name="a", precision=fast)
    b = _point_sky(ra=0.3, dec=0.4, flux=2.0, ref_freq=150e6, name="b", precision=fast)
    data = concat_point_sources([a, b], precision=fast)
    assert data["ra_rad"].dtype == fast.sky_model.get_dtype("source_positions")
    assert data["flux"].dtype == fast.sky_model.get_dtype("flux")
    assert data["spectral_index"].dtype == fast.sky_model.get_dtype("spectral_index")


def test_concat_fully_missing_optional_column_no_length_error():
    """C2: a fully-missing optional extra-column no longer raises.

    Build two models where neither carries a given extra column; the result
    simply omits it (or yields a correctly-sized object array) instead of a
    misattributed length error.
    """
    a = _point_sky(ra=0.1, dec=0.2, flux=1.0, ref_freq=150e6, name="a")
    b = _point_sky(ra=0.3, dec=0.4, flux=2.0, ref_freq=150e6, name="b")
    data = concat_point_sources([a, b], precision=_precision())
    # No extra columns were declared, so the dict is empty rather than carrying
    # a zero-length sentinel of the wrong length.
    assert data["extra_columns"] == {}
    assert data["ra_rad"].shape[0] == 2


def test_concat_partial_missing_object_column_is_full_length():
    """C2: a column present on one model and absent on the other yields a
    full-length object array filled with None for the missing rows."""
    from radiosim.core.sky import create_from_arrays

    # Model 'a' carries an extra int column; model 'b' does not.
    a = create_from_arrays(
        ra_rad=np.array([0.1]),
        dec_rad=np.array([0.2]),
        flux=np.array([1.0]),
        spectral_index=np.array([0.0]),
        ref_freq=np.array([150e6]),
        reference_frequency=150e6,
        extra_columns={"foo": np.array([42], dtype=np.int64)},
        precision=_precision(),
        model_name="a",
        provenance=SkyProvenance(
            monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB
        ),
    )
    b = _point_sky(ra=0.3, dec=0.4, flux=2.0, ref_freq=150e6, name="b")

    data = concat_point_sources([a, b], precision=_precision())
    foo = data["extra_columns"]["foo"]
    assert foo.shape[0] == 2  # full length, not just the present rows
    assert foo[0] == 42
    assert foo[1] is None


# --------------------------------------------------------------------------- #
# merge_provenance (now combine.merge)
# --------------------------------------------------------------------------- #


def test_merge_provenance_full_sky_coverage_and_notes():
    from radiosim.core.sky.combine.merge import merge_provenance

    freqs = np.array([150e6])
    a = _healpix_sky(nside=8, frequencies=freqs, fill=2.0, name="a")
    b = _healpix_sky(nside=8, frequencies=freqs, fill=3.0, name="b")
    a = a.replace(
        provenance=a.provenance.replace(
            sky_coverage=SkyCoverage.FULL_SKY, notes="layer-a"
        )
    )
    b = b.replace(
        provenance=b.provenance.replace(
            sky_coverage=SkyCoverage.FULL_SKY, notes="layer-b"
        )
    )
    merged = merge_provenance([a, b])
    assert merged.sky_coverage == SkyCoverage.FULL_SKY
    assert merged.coverage_fraction == 1.0
    assert "layer-a" in merged.notes and "layer-b" in merged.notes


def test_merge_provenance_double_cmb_drops_monopole():
    from radiosim.core.sky.combine.merge import merge_provenance

    freqs = np.array([150e6])
    a = _healpix_sky(nside=8, frequencies=freqs, fill=2.0, name="a")
    b = _healpix_sky(nside=8, frequencies=freqs, fill=3.0, name="b")
    prov = SkyProvenance(
        monopole_convention=MonopoleConvention.ABSOLUTE_WITH_CMB,
        sky_coverage=SkyCoverage.FULL_SKY,
        monopole_k=2.725,
    )
    a = a.replace(provenance=prov)
    b = b.replace(provenance=prov)
    merged = merge_provenance([a, b])
    # Two ABSOLUTE_WITH_CMB inputs must not double-count the CMB monopole.
    assert merged.monopole_k is None
    assert "double-count" in (merged.notes or "")


def test_merge_provenance_source_subtraction_promotion():
    from radiosim.core.sky.combine.merge import merge_provenance

    freqs = np.array([150e6])
    a = _healpix_sky(nside=8, frequencies=freqs, fill=2.0, name="a")
    b = _healpix_sky(nside=8, frequencies=freqs, fill=3.0, name="b")
    a = a.replace(
        provenance=a.provenance.replace(source_subtraction=SourceSubtractionStatus.ALL)
    )
    b = b.replace(
        provenance=b.provenance.replace(source_subtraction=SourceSubtractionStatus.ALL)
    )
    merged = merge_provenance([a, b])
    assert merged.source_subtraction == SourceSubtractionStatus.ALL


# --------------------------------------------------------------------------- #
# disjointness alpha tunable (G6)
# --------------------------------------------------------------------------- #


def _diffuse_subtracted(nside, freqs, threshold_jy, nu_hz, name):
    sky = _healpix_sky(nside=nside, frequencies=freqs, fill=2.0, name=name)
    return sky.replace(
        provenance=sky.provenance.replace(
            source_subtraction=SourceSubtractionStatus.ABOVE_THRESHOLD,
            source_subtraction_threshold_jy=threshold_jy,
            source_subtraction_freq_hz=nu_hz,
            angular_resolution_rad=(0.05, 0.1),
        )
    )


def _catalog_with_completeness(completeness_jy, nu_hz, name):
    sky = _point_sky(ra=0.3, dec=0.2, flux=1.0, ref_freq=nu_hz, name=name)
    return sky.replace(
        provenance=sky.provenance.replace(
            flux_completeness_jy=(completeness_jy, completeness_jy * 5.0),
            flux_completeness_freq_hz=nu_hz,
            angular_resolution_rad=(1e-5, 1e-4),
            monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
        )
    )


def test_disjointness_alpha_changes_threshold():
    """G6: a custom alpha changes the scaled subtraction threshold and hence
    the pass/fail boundary of the completeness rule."""
    nside = 8
    freqs = np.array([150e6])
    # Diffuse subtracted at 1 Jy @ 150 MHz; catalog completeness 0.7 Jy @ 60 MHz.
    diffuse = _diffuse_subtracted(nside, freqs, threshold_jy=1.0, nu_hz=150e6, name="d")
    catalog = _catalog_with_completeness(0.7, nu_hz=60e6, name="c")
    models = [diffuse, catalog]

    # Scaling 1 Jy from 150->60 MHz: factor (60/150)**alpha.
    # alpha = -2.0 -> factor = (0.4)**-2 = 6.25 -> 6.25 Jy > 0.7 -> NOT disjoint (raises).
    with pytest.raises(ValueError):
        check_physical_disjointness(models, "error", alpha=-2.0)

    # alpha = +2.0 -> factor = (0.4)**2 = 0.16 -> 0.16 Jy <= 0.7 -> disjoint (passes).
    check_physical_disjointness(models, "error", alpha=2.0)


def test_prepare_sky_model_exposes_subtraction_alpha():
    """G6: the subtraction-scaling alpha is a prepare_sky_model tunable."""
    from radiosim.core.sky import PrepareSkyOptions

    # The option must exist on PrepareSkyOptions (threaded through pipeline).
    opts = PrepareSkyOptions(subtraction_scaling_alpha=-1.5)
    assert opts.subtraction_scaling_alpha == -1.5


def test_assume_disjoint_skips_double_count_but_keeps_monopole():
    """assume_disjoint bypasses double-count rules but not monopole checks."""
    nside = 8
    freqs = np.array([150e6])
    diffuse = _healpix_sky(nside=nside, frequencies=freqs, fill=2.0, name="d")
    diffuse = diffuse.replace(
        provenance=diffuse.provenance.replace(
            source_subtraction=SourceSubtractionStatus.NONE,
        )
    )
    catalog = _catalog_with_completeness(0.7, nu_hz=60e6, name="c")
    models = [diffuse, catalog]

    with pytest.raises(ValueError):
        check_physical_disjointness(models, "error")

    with pytest.warns(UserWarning, match="assume_disjoint"):
        check_physical_disjointness(models, "error", assume_disjoint=True)

    catalog_bad = catalog.replace(
        provenance=catalog.provenance.replace(
            monopole_convention=MonopoleConvention.MEAN_SUBTRACTED,
        )
    )
    with pytest.raises(ValueError, match="monopole conventions"):
        check_physical_disjointness(
            [diffuse, catalog_bad],
            "error",
            assume_disjoint=True,
        )


def test_prepare_sky_model_exposes_assume_disjoint():
    """assume_disjoint is a prepare_sky_model tunable on PrepareSkyOptions."""
    from radiosim.core.sky import PrepareSkyOptions

    opts = PrepareSkyOptions(assume_disjoint=True)
    assert opts.assume_disjoint is True


def test_combine_engine_has_no_all_block():
    import ast
    from pathlib import Path

    source = Path("src/radiosim/core/sky/combine/engine.py").read_text()
    tree = ast.parse(source)
    assert not any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
        for node in tree.body
    )


def test_prepare_sky_model_resolves_target_once(monkeypatch):
    """Pipeline resolves target representation once and threads it through."""
    import radiosim.core.sky.combine.engine as engine
    import radiosim.core.sky.combine.pipeline as pipeline

    calls: list[tuple[object, object]] = []
    original = engine.resolve_target_representation

    def spy_resolve(models, requested):
        calls.append((models, requested))
        return original(models, requested)

    monkeypatch.setattr(engine, "resolve_target_representation", spy_resolve)
    monkeypatch.setattr(pipeline, "resolve_target_representation", spy_resolve)

    freqs = np.array([150e6])
    a = _healpix_sky(nside=8, frequencies=freqs, fill=2.0, name="a")
    b = _healpix_sky(nside=8, frequencies=freqs, fill=3.0, name="b")

    from radiosim.core.sky import prepare_sky_model

    prepare_sky_model(
        [a, b],
        representation="healpix_map",
        nside=8,
        precision=_precision(),
    )
    assert len(calls) == 1
