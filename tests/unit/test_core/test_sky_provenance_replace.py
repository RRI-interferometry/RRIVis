"""Tests for ``SkyProvenance.replace`` — Pydantic-validated field updates.

Verifies that ``replace`` re-runs every cross-field validator that the
constructor runs, so callers cannot use it to escape into invalid states
(partial-sky + monopole_k, footprint inconsistent with sky_coverage,
source_subtraction status mismatched with threshold, etc.).
"""

from __future__ import annotations

import numpy as np
import pytest

from rrivis.core.sky import (
    MonopoleConvention,
    SkyCoverage,
    SkyFootprint,
    SkyProvenance,
    SourceSubtractionStatus,
)


def _full_sky_footprint(nside: int = 8) -> SkyFootprint:
    npix = 12 * nside * nside
    return SkyFootprint(nside=nside, hpx_inds=np.arange(npix, dtype=np.int64))


def _partial_footprint(nside: int = 8, n_keep: int = 100) -> SkyFootprint:
    return SkyFootprint(nside=nside, hpx_inds=np.arange(n_keep, dtype=np.int64))


class TestReplaceIdentity:
    def test_no_changes_round_trips(self):
        prov = SkyProvenance(
            sky_coverage=SkyCoverage.FULL_SKY,
            monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
            monopole_k=12.0,
        )
        assert prov.replace() == prov

    def test_unknown_field_raises_typeerror(self):
        prov = SkyProvenance()
        with pytest.raises(TypeError, match="not_a_field"):
            prov.replace(not_a_field=42)


class TestReplaceReRunsValidators:
    def test_partial_sky_with_monopole_raises(self):
        """The validator that forbids partial-sky + monopole_k must fire on replace."""
        prov = SkyProvenance(
            sky_coverage=SkyCoverage.FULL_SKY,
            monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
            monopole_k=2.5,
        )
        # Today (without re-validation) this would silently produce an invalid object.
        with pytest.raises(ValueError, match="monopole_k"):
            prov.replace(
                sky_coverage=SkyCoverage.PARTIAL_SKY,
                coverage_fraction=0.5,
            )

    def test_above_threshold_requires_threshold(self):
        prov = SkyProvenance(
            source_subtraction=SourceSubtractionStatus.ABOVE_THRESHOLD,
            source_subtraction_threshold_jy=1.0,
        )
        with pytest.raises(ValueError, match="threshold_jy"):
            prov.replace(source_subtraction_threshold_jy=None)

    def test_threshold_without_status_rejected(self):
        prov = SkyProvenance()
        with pytest.raises(ValueError, match="source_subtraction"):
            prov.replace(
                source_subtraction_threshold_jy=2.0,
                source_subtraction=SourceSubtractionStatus.NONE,
            )


class TestReplaceFootprintReDerivation:
    def test_swapping_to_partial_footprint_rederives_coverage(self):
        full = _full_sky_footprint(nside=8)
        prov = SkyProvenance(coverage_footprint=full)
        assert prov.sky_coverage is SkyCoverage.FULL_SKY
        assert prov.coverage_fraction == pytest.approx(1.0)

        partial = _partial_footprint(nside=8, n_keep=100)
        new_prov = prov.replace(coverage_footprint=partial)
        assert new_prov.sky_coverage is SkyCoverage.PARTIAL_SKY
        assert new_prov.coverage_fraction == pytest.approx(partial.coverage_fraction)

    def test_swapping_to_full_footprint_rederives_coverage(self):
        partial = _partial_footprint(nside=8, n_keep=10)
        prov = SkyProvenance(coverage_footprint=partial)
        assert prov.sky_coverage is SkyCoverage.PARTIAL_SKY

        full = _full_sky_footprint(nside=8)
        new_prov = prov.replace(coverage_footprint=full)
        assert new_prov.sky_coverage is SkyCoverage.FULL_SKY
        assert new_prov.coverage_fraction == pytest.approx(1.0)

    def test_explicit_coverage_with_new_footprint_must_match(self):
        full = _full_sky_footprint(nside=8)
        prov = SkyProvenance(coverage_footprint=full)
        partial = _partial_footprint(nside=8, n_keep=10)

        with pytest.raises(ValueError, match="sky_coverage"):
            prov.replace(
                coverage_footprint=partial,
                sky_coverage=SkyCoverage.FULL_SKY,
            )

    def test_dropping_footprint_preserves_explicit_full_sky(self):
        full = _full_sky_footprint(nside=8)
        prov = SkyProvenance(coverage_footprint=full)
        assert prov.sky_coverage is SkyCoverage.FULL_SKY
        # Removing the footprint without changing sky_coverage should
        # preserve FULL_SKY (validator at line 442 only requires
        # coverage_fraction == 1.0).
        new_prov = prov.replace(coverage_footprint=None)
        assert new_prov.coverage_footprint is None
        assert new_prov.sky_coverage is SkyCoverage.FULL_SKY
        assert new_prov.coverage_fraction == pytest.approx(1.0)


class TestReplacePreservesUnchangedFields:
    def test_only_targeted_fields_change(self):
        prov = SkyProvenance(
            sky_coverage=SkyCoverage.FULL_SKY,
            monopole_convention=MonopoleConvention.ABSOLUTE_NO_CMB,
            monopole_k=2.0,
            angular_resolution_rad=(0.001, np.pi),
            flux_completeness_jy=(0.05, 1000.0),
            flux_completeness_freq_hz=150e6,
            notes="initial",
        )
        new_prov = prov.replace(notes="updated", monopole_k=3.5)
        assert new_prov.notes == "updated"
        assert new_prov.monopole_k == 3.5
        # Unchanged fields carry through.
        assert new_prov.sky_coverage is SkyCoverage.FULL_SKY
        assert new_prov.angular_resolution_rad == (0.001, np.pi)
        assert new_prov.flux_completeness_jy == (0.05, 1000.0)
        assert new_prov.flux_completeness_freq_hz == 150e6
