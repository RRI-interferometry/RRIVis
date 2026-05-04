"""Tests for HEALPix sky-footprint metadata."""

import healpy as hp
import numpy as np
import pytest

from rrivis.core.sky import SkyCoverage, SkyFootprint, SkyProvenance, SkyRegion


class TestSkyFootprint:
    def test_from_mask_roundtrip(self):
        nside = 8
        mask = np.zeros(hp.nside2npix(nside), dtype=bool)
        mask[[1, 5, 9]] = True

        footprint = SkyFootprint.from_mask(mask, nside=nside)

        assert footprint.nside == nside
        assert footprint.coordinate_frame == "icrs"
        np.testing.assert_array_equal(footprint.hpx_inds, np.array([1, 5, 9]))
        np.testing.assert_array_equal(footprint.to_mask(), mask)

    def test_union_and_intersection(self):
        footprint_a = SkyFootprint(nside=8, hpx_inds=np.array([1, 2, 3]))
        footprint_b = SkyFootprint(nside=8, hpx_inds=np.array([3, 4, 5]))

        union = footprint_a.union(footprint_b)
        intersection = footprint_a.intersect(footprint_b)

        np.testing.assert_array_equal(union.hpx_inds, np.array([1, 2, 3, 4, 5]))
        np.testing.assert_array_equal(intersection.hpx_inds, np.array([3]))

    def test_region_helper_uses_canonical_grid(self):
        footprint = SkyRegion.cone(180.0, -30.0, 10.0).footprint()

        assert footprint.nside == 256
        assert footprint.coordinate_frame == "icrs"
        assert 0.0 < footprint.coverage_fraction < 1.0


class TestSkyProvenanceFootprint:
    def test_provenance_derives_fraction_and_coverage(self):
        footprint = SkyFootprint(nside=8, hpx_inds=np.array([1, 2, 3]))

        provenance = SkyProvenance(coverage_footprint=footprint)

        assert provenance.sky_coverage is SkyCoverage.PARTIAL_SKY
        assert provenance.coverage_fraction == pytest.approx(3 / hp.nside2npix(8))

    def test_inconsistent_fraction_is_rejected(self):
        footprint = SkyFootprint(nside=8, hpx_inds=np.array([1, 2, 3]))

        with pytest.raises(ValueError, match="inconsistent with coverage_footprint"):
            SkyProvenance(
                sky_coverage=SkyCoverage.PARTIAL_SKY,
                coverage_fraction=0.5,
                coverage_footprint=footprint,
            )
