"""Physical-disjointness checker + monopole-consistency tests.

Covers the pass/warn/error matrix for ``_combine_models``:

* Rule 2.1 — diffuse ``source_subtraction=ALL`` ⇒ pass
* Rule 2.2 — diffuse ``ABOVE_THRESHOLD`` with threshold ≤ catalog S_min ⇒ pass
* Rule 2.3 — angular-resolution scale separation ⇒ pass
* Unknown provenance on a cross-type pair ⇒ fail-closed under ``error``
* Incompatible monopole conventions ⇒ hard error under every policy
* Combined provenance: monopole_k sums; conventions merge; source_sub & angular
  range propagate sensibly.
"""

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
)
from radiosim.core.sky.combine.engine import _combine_models


@pytest.fixture
def precision() -> PrecisionConfig:
    return PrecisionConfig.standard()


def _point(
    *,
    precision: PrecisionConfig,
    flux_min_jy: float = 5.0,
    flux_max_jy: float = 20.0,
    ref_freq_hz: float = 150e6,
    beam_fwhm_arcsec: float = 120.0,
    model_name: str = "catalog",
    monopole_convention: MonopoleConvention = MonopoleConvention.ABSOLUTE_NO_CMB,
    monopole_k: float | None = None,
) -> SkyModel:
    """Construct a synthetic point-source model with explicit provenance."""
    rng = np.random.default_rng(42)
    n = 8
    sky = create_from_arrays(
        ra_rad=rng.uniform(0, 2 * np.pi, n),
        dec_rad=rng.uniform(-np.pi / 2, np.pi / 2, n),
        flux=np.linspace(flux_min_jy, flux_max_jy, n),
        reference_frequency=ref_freq_hz,
        model_name=model_name,
        precision=precision,
    )
    beam_rad = beam_fwhm_arcsec * np.pi / 180.0 / 3600.0
    return sky.replace(
        provenance=SkyProvenance(
            flux_completeness_jy=(flux_min_jy, flux_max_jy),
            flux_completeness_freq_hz=ref_freq_hz,
            angular_resolution_rad=(beam_rad, float(np.pi)),
            sky_coverage=SkyCoverage.FULL_SKY,
            coverage_fraction=1.0,
            monopole_convention=monopole_convention,
            monopole_k=monopole_k,
            source_subtraction=SourceSubtractionStatus.NONE,
            notes=f"test/{model_name}",
        )
    )


def _diffuse(
    *,
    precision: PrecisionConfig,
    nside: int = 8,
    freqs_hz: np.ndarray | None = None,
    pixel_value_k: float = 100.0,
    model_name: str = "diffuse",
    source_subtraction: SourceSubtractionStatus = SourceSubtractionStatus.NONE,
    threshold_jy: float | None = None,
    threshold_freq_hz: float | None = None,
    angular_lo_arcmin: float = 56.0,
    monopole_convention: MonopoleConvention = MonopoleConvention.ABSOLUTE_NO_CMB,
    monopole_k: float | None = None,
) -> SkyModel:
    freqs_hz = (
        np.asarray([150e6, 160e6], dtype=np.float64) if freqs_hz is None else freqs_hz
    )
    npix = hp.nside2npix(nside)
    lo_rad = angular_lo_arcmin * np.pi / 10800.0
    return SkyModel(
        healpix=HealpixData(
            maps=np.full((len(freqs_hz), npix), pixel_value_k, dtype=np.float32),
            nside=nside,
            frequencies=freqs_hz,
            coordinate_frame="icrs",
        ),
        model_name=model_name,
        provenance=SkyProvenance(
            angular_resolution_rad=(lo_rad, float(np.pi)),
            sky_coverage=SkyCoverage.FULL_SKY,
            coverage_fraction=1.0,
            monopole_convention=monopole_convention,
            monopole_k=monopole_k,
            source_subtraction=source_subtraction,
            source_subtraction_threshold_jy=threshold_jy,
            source_subtraction_freq_hz=threshold_freq_hz,
            source_subtraction_method=(
                "gaussian_fit_inpaint" if threshold_jy is not None else None
            ),
        ),
        precision=precision,
    )


class TestDisjointnessPassRules:
    def test_all_subtracted_passes(self, precision):
        d = _diffuse(
            precision=precision,
            source_subtraction=SourceSubtractionStatus.ALL,
            monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
        )
        p = _point(precision=precision)
        # No exception under default policy="error".
        combined = _combine_models(
            [d, p],
            precision=precision,
            nside=8,
            frequencies=np.asarray([150e6, 160e6]),
        )
        assert combined.healpix is not None

    def test_threshold_below_catalog_min_passes(self, precision):
        # Haslam-style subtraction at 2 Jy @ 408 MHz scales (α=-0.7) to
        # ~4.29 Jy @ 150 MHz. Catalog S_min = 5 Jy > 4.29 ⇒ disjoint.
        d = _diffuse(
            precision=precision,
            source_subtraction=SourceSubtractionStatus.ABOVE_THRESHOLD,
            threshold_jy=2.0,
            threshold_freq_hz=408e6,
        )
        p = _point(precision=precision, flux_min_jy=5.0, flux_max_jy=20.0)
        combined = _combine_models(
            [d, p],
            precision=precision,
            nside=8,
            frequencies=np.asarray([150e6, 160e6]),
        )
        assert combined.healpix is not None

    def test_angular_scale_separation_passes(self, precision):
        # Diffuse θ_max < point θ_min ⇒ scale-separated.
        # Set diffuse to a tiny band-limit (θ_max = 0.01 rad) and point to a
        # beam much larger so θ_min > 0.01 rad.
        nside = 8
        freqs = np.asarray([150e6, 160e6])
        # Manually construct diffuse with angular_max < point angular_min.
        d_prov = SkyProvenance(
            angular_resolution_rad=(1e-3, 0.01),
            sky_coverage=SkyCoverage.FULL_SKY,
            coverage_fraction=1.0,
            monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
            source_subtraction=SourceSubtractionStatus.NONE,
        )
        d = _diffuse(precision=precision, nside=nside, freqs_hz=freqs)
        d = d.replace(provenance=d_prov)
        p = _point(precision=precision, beam_fwhm_arcsec=60 * 60)  # 1 deg
        # point θ_min = 1 deg ≈ 0.0175 rad > 0.01 rad ⇒ disjoint by scale.
        combined = _combine_models(
            [d, p], precision=precision, nside=nside, frequencies=freqs
        )
        assert combined.healpix is not None


class TestDisjointnessFailures:
    def test_none_subtraction_fails_under_error(self, precision):
        d = _diffuse(
            precision=precision,
            source_subtraction=SourceSubtractionStatus.NONE,
        )
        p = _point(precision=precision)
        with pytest.raises(ValueError, match="double-counting"):
            _combine_models(
                [d, p],
                precision=precision,
                mixed_model_policy="error",
                nside=8,
                frequencies=np.asarray([150e6, 160e6]),
            )

    def test_none_subtraction_warns_under_warn(self, precision):
        d = _diffuse(
            precision=precision,
            source_subtraction=SourceSubtractionStatus.NONE,
        )
        p = _point(precision=precision)
        with pytest.warns(UserWarning, match="double-counting"):
            combined = _combine_models(
                [d, p],
                precision=precision,
                mixed_model_policy="warn",
                nside=8,
                frequencies=np.asarray([150e6, 160e6]),
            )
        assert combined.healpix is not None

    def test_none_subtraction_silent_under_allow(self, precision):
        d = _diffuse(
            precision=precision,
            source_subtraction=SourceSubtractionStatus.NONE,
        )
        p = _point(precision=precision)
        # Should not warn and not raise.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # escalate any warning to error
            combined = _combine_models(
                [d, p],
                precision=precision,
                mixed_model_policy="allow",
                nside=8,
                frequencies=np.asarray([150e6, 160e6]),
            )
        assert combined.healpix is not None

    def test_unknown_provenance_fails_closed(self, precision):
        """UNKNOWN on a diffuse+point pair must fail-closed under error."""
        # Make a diffuse model with no provenance declared (UNKNOWN).
        d = _diffuse(precision=precision)
        d = d.replace(provenance=SkyProvenance())  # UNKNOWN
        p = _point(precision=precision)
        with pytest.raises(ValueError, match="UNKNOWN"):
            _combine_models(
                [d, p],
                precision=precision,
                nside=8,
                frequencies=np.asarray([150e6, 160e6]),
            )

    def test_error_message_names_actionable_provenance_fields(self, precision):
        d = _diffuse(precision=precision)
        d = d.replace(
            provenance=SkyProvenance(
                monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
            )
        )  # source-subtraction / scale provenance UNKNOWN, but monopole is declared
        p = _point(precision=precision)

        with pytest.raises(ValueError) as exc_info:
            _combine_models(
                [d, p],
                precision=precision,
                nside=8,
                frequencies=np.asarray([150e6, 160e6]),
            )

        message = str(exc_info.value)
        assert "source_subtraction" in message
        assert "source_subtraction_threshold_jy" in message
        assert "source_subtraction_freq_hz" in message
        assert "flux_completeness_jy" in message
        assert "flux_completeness_freq_hz" in message
        assert "angular_resolution_rad" in message
        assert "SkyProvenance(" in message

    def test_threshold_above_catalog_min_fails(self, precision):
        """Diffuse subtracted at 10 Jy @ 150 MHz, catalog S_min = 5 Jy ⇒ overlap."""
        d = _diffuse(
            precision=precision,
            source_subtraction=SourceSubtractionStatus.ABOVE_THRESHOLD,
            threshold_jy=10.0,
            threshold_freq_hz=150e6,
        )
        p = _point(precision=precision, flux_min_jy=5.0, flux_max_jy=20.0)
        with pytest.raises(ValueError, match="double-counted"):
            _combine_models(
                [d, p],
                precision=precision,
                nside=8,
                frequencies=np.asarray([150e6, 160e6]),
            )


class TestMonopoleConsistency:
    @pytest.mark.parametrize("policy", ["error", "warn", "allow"])
    def test_mixed_convention_always_raises(self, precision, policy):
        a = _diffuse(
            precision=precision,
            monopole_convention=MonopoleConvention.ABSOLUTE_WITH_CMB,
        )
        b = _diffuse(
            precision=precision,
            monopole_convention=MonopoleConvention.MEAN_SUBTRACTED,
            model_name="other",
        )
        with pytest.raises(ValueError, match="monopole conventions"):
            _combine_models(
                [a, b],
                precision=precision,
                mixed_model_policy=policy,
                nside=8,
                frequencies=np.asarray([150e6, 160e6]),
            )

    def test_same_absolute_convention_passes(self, precision):
        a = _diffuse(
            precision=precision,
            source_subtraction=SourceSubtractionStatus.ALL,
            monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
        )
        b = _diffuse(
            precision=precision,
            source_subtraction=SourceSubtractionStatus.ALL,
            monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
            model_name="other",
        )
        combined = _combine_models(
            [a, b],
            precision=precision,
            nside=8,
            frequencies=np.asarray([150e6, 160e6]),
        )
        assert (
            combined.provenance.monopole_convention
            is MonopoleConvention.ABSOLUTE_NO_CMB
        )


class TestCombinedProvenancePropagation:
    def test_monopole_k_sums_when_all_declared(self, precision):
        a = _diffuse(
            precision=precision,
            source_subtraction=SourceSubtractionStatus.ALL,
            monopole_k=100.0,
            model_name="a",
        )
        b = _diffuse(
            precision=precision,
            source_subtraction=SourceSubtractionStatus.ALL,
            monopole_k=50.0,
            model_name="b",
        )
        combined = _combine_models(
            [a, b],
            precision=precision,
            nside=8,
            frequencies=np.asarray([150e6, 160e6]),
        )
        assert combined.provenance.monopole_k == pytest.approx(150.0)

    def test_monopole_k_measured_from_healpix_when_any_missing(self, precision):
        """For HEALPix outputs, _combine_models falls back to the measured
        pixel-area-weighted mean of the combined cube when the per-layer
        merge produced None (e.g. because a contributor lacked ``monopole_k``).
        """
        a = _diffuse(
            precision=precision,
            source_subtraction=SourceSubtractionStatus.ALL,
            pixel_value_k=100.0,
            monopole_k=100.0,
        )
        b = _diffuse(
            precision=precision,
            source_subtraction=SourceSubtractionStatus.ALL,
            pixel_value_k=50.0,
            monopole_k=None,
            model_name="other",
        )
        combined = _combine_models(
            [a, b],
            precision=precision,
            nside=8,
            frequencies=np.asarray([150e6, 160e6]),
        )
        # Measured mean should reflect the sum 100 + 50 = 150 K of both layers.
        assert combined.provenance.monopole_k == pytest.approx(150.0, rel=1e-4)

    def test_combined_source_subtraction_status_merges_conservatively(self, precision):
        a = _diffuse(
            precision=precision,
            source_subtraction=SourceSubtractionStatus.ALL,
        )
        b = _diffuse(
            precision=precision,
            source_subtraction=SourceSubtractionStatus.ALL,
            model_name="other",
        )
        combined = _combine_models(
            [a, b],
            precision=precision,
            nside=8,
            frequencies=np.asarray([150e6, 160e6]),
        )
        assert combined.provenance.source_subtraction is SourceSubtractionStatus.ALL

        # Mixing ALL + NONE across two DIFFUSE models should not fire the
        # disjointness warning (no cross-type pair), but the combined
        # provenance still downgrades to UNKNOWN since the inputs disagree.
        c = _diffuse(
            precision=precision,
            source_subtraction=SourceSubtractionStatus.NONE,
            model_name="none",
        )
        mixed = _combine_models(
            [a, c],
            precision=precision,
            nside=8,
            frequencies=np.asarray([150e6, 160e6]),
        )
        assert mixed.provenance.source_subtraction is SourceSubtractionStatus.UNKNOWN

    def test_above_threshold_provenance_preserved_when_homogeneous(self, precision):
        a = _diffuse(
            precision=precision,
            source_subtraction=SourceSubtractionStatus.ABOVE_THRESHOLD,
            threshold_jy=2.0,
            threshold_freq_hz=408e6,
        )
        b = _diffuse(
            precision=precision,
            source_subtraction=SourceSubtractionStatus.ABOVE_THRESHOLD,
            threshold_jy=2.0,
            threshold_freq_hz=408e6,
            model_name="other",
        )
        combined = _combine_models(
            [a, b],
            precision=precision,
            nside=8,
            frequencies=np.asarray([150e6, 160e6]),
        )
        assert (
            combined.provenance.source_subtraction
            is SourceSubtractionStatus.ABOVE_THRESHOLD
        )
        assert combined.provenance.source_subtraction_threshold_jy == pytest.approx(2.0)
        assert combined.provenance.source_subtraction_freq_hz == pytest.approx(408e6)
        assert combined.provenance.source_subtraction_method == "gaussian_fit_inpaint"


class TestMergeProvenanceMonopoleDoubleCount:
    """``merge_provenance`` must refuse to sum monopoles that would
    double-count the CMB or alias an UNKNOWN convention onto absolutes.

    These tests target ``merge_provenance`` directly (not ``_combine_models``)
    so the post-merge measured-monopole fallback in ``_combine_as_healpix_merge``
    cannot mask a wrongly-summed value.
    """

    def test_two_with_cmb_drops_monopole_k(self, precision):
        from radiosim.core.sky.combine.merge import merge_provenance

        a = _diffuse(
            precision=precision,
            source_subtraction=SourceSubtractionStatus.ALL,
            monopole_convention=MonopoleConvention.ABSOLUTE_WITH_CMB,
            monopole_k=2.725,
        )
        b = _diffuse(
            precision=precision,
            source_subtraction=SourceSubtractionStatus.ALL,
            monopole_convention=MonopoleConvention.ABSOLUTE_WITH_CMB,
            monopole_k=2.725,
            model_name="other",
        )
        merged = merge_provenance([a, b])
        assert merged.monopole_k is None
        assert merged.notes is not None
        assert "double-count the CMB" in merged.notes

    def test_with_cmb_plus_no_cmb_still_sums(self, precision):
        from radiosim.core.sky.combine.merge import merge_provenance

        a = _diffuse(
            precision=precision,
            source_subtraction=SourceSubtractionStatus.ALL,
            monopole_convention=MonopoleConvention.ABSOLUTE_WITH_CMB,
            monopole_k=2.725,
        )
        b = _diffuse(
            precision=precision,
            source_subtraction=SourceSubtractionStatus.ALL,
            monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
            monopole_k=12.0,
            model_name="residual",
        )
        merged = merge_provenance([a, b])
        # CMB + sky residual is a legitimate addition.
        assert merged.monopole_k == pytest.approx(2.725 + 12.0)

    def test_unknown_alongside_absolute_drops_monopole_k(self, precision):
        from radiosim.core.sky.combine.merge import merge_provenance

        a = _diffuse(
            precision=precision,
            source_subtraction=SourceSubtractionStatus.ALL,
            monopole_convention=MonopoleConvention.UNKNOWN,
            monopole_k=10.0,
            model_name="unknown",
        )
        b = _diffuse(
            precision=precision,
            source_subtraction=SourceSubtractionStatus.ALL,
            monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
            monopole_k=5.0,
            model_name="absolute",
        )
        merged = merge_provenance([a, b])
        assert merged.monopole_k is None
        assert merged.notes is not None
        assert "UNKNOWN" in merged.notes


# =============================================================================
# UNKNOWN monopole convention policy (PR 11)
# =============================================================================


class TestUnknownMonopolePolicy:
    """``mixed_model_policy`` controls whether UNKNOWN monopole_convention
    is a hard error, a warning, or silently allowed."""

    def _two_unknown_diffuse(self, precision):
        a = _diffuse(
            precision=precision,
            source_subtraction=SourceSubtractionStatus.ALL,
            monopole_convention=MonopoleConvention.UNKNOWN,
            monopole_k=None,
            model_name="a",
        )
        b = _diffuse(
            precision=precision,
            source_subtraction=SourceSubtractionStatus.ALL,
            monopole_convention=MonopoleConvention.UNKNOWN,
            monopole_k=None,
            model_name="b",
        )
        return a, b

    def test_default_error_policy_rejects_unknown(self, precision):
        from radiosim.core.sky.combine.disjointness import (
            check_physical_disjointness,
        )

        a, b = self._two_unknown_diffuse(precision)
        with pytest.raises(ValueError, match="monopole_convention=UNKNOWN"):
            check_physical_disjointness([a, b], "error")

    def test_warn_policy_emits_warning(self, precision):
        import warnings

        from radiosim.core.sky.combine.disjointness import (
            check_physical_disjointness,
        )

        a, b = self._two_unknown_diffuse(precision)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_physical_disjointness([a, b], "warn")
        assert any("monopole_convention=UNKNOWN" in str(w.message) for w in caught)

    def test_allow_policy_passes_silently(self, precision):
        import warnings

        from radiosim.core.sky.combine.disjointness import (
            check_physical_disjointness,
        )

        a, b = self._two_unknown_diffuse(precision)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_physical_disjointness([a, b], "allow")
        # No UNKNOWN-monopole warning under allow.
        assert not any("monopole_convention=UNKNOWN" in str(w.message) for w in caught)
