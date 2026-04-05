"""Tests for rrivis.core.sky.region — SkyRegion spatial filtering."""

import healpy as hp
import numpy as np
import pytest

from rrivis.core.sky import SkyRegion

# ---------------------------------------------------------------------------
# Cone
# ---------------------------------------------------------------------------


class TestCone:
    def test_cone_basic(self):
        """Cone at (180, 0) with radius 10 deg: center included, antipode excluded."""
        region = SkyRegion.cone(ra_deg=180.0, dec_deg=0.0, radius_deg=10.0)

        ra_rad = np.deg2rad(np.array([180.0, 0.0, 185.0]))
        dec_rad = np.deg2rad(np.array([0.0, 0.0, 3.0]))
        mask = region.contains(ra_rad, dec_rad)

        assert mask[0] is np.True_  # center point
        assert mask[1] is np.False_  # antipode
        assert mask[2] is np.True_  # near center

    def test_cone_validation(self):
        """Cone with non-positive radius raises ValueError."""
        with pytest.raises(ValueError, match="radius_deg must be positive"):
            SkyRegion.cone(ra_deg=0.0, dec_deg=0.0, radius_deg=0.0)
        with pytest.raises(ValueError, match="radius_deg must be positive"):
            SkyRegion.cone(ra_deg=0.0, dec_deg=0.0, radius_deg=-5.0)


# ---------------------------------------------------------------------------
# Box
# ---------------------------------------------------------------------------


class TestBox:
    def test_box_basic(self):
        """Box at (180, 0) with 20x10 deg: interior included, exterior excluded."""
        region = SkyRegion.box(
            ra_deg=180.0, dec_deg=0.0, width_deg=20.0, height_deg=10.0
        )

        ra_rad = np.deg2rad(np.array([180.0, 180.0, 200.0]))
        dec_rad = np.deg2rad(np.array([3.0, 20.0, 0.0]))
        mask = region.contains(ra_rad, dec_rad)

        assert mask[0] is np.True_  # inside
        assert mask[1] is np.False_  # above box (dec=20 > half_h=5)
        assert mask[2] is np.False_  # outside RA range

    def test_box_ra_wrap(self):
        """Box near RA=0 wraps around: point at RA=1 deg is inside."""
        region = SkyRegion.box(
            ra_deg=359.0, dec_deg=0.0, width_deg=10.0, height_deg=10.0
        )
        # RA=1 should be inside (359-5=354 to 359+5=364 -> wraps to 4 deg)
        ra_rad = np.deg2rad(np.array([1.0, 180.0]))
        dec_rad = np.deg2rad(np.array([0.0, 0.0]))
        mask = region.contains(ra_rad, dec_rad)

        assert mask[0] is np.True_  # crosses RA=0
        assert mask[1] is np.False_  # far away

    def test_box_validation(self):
        """Box with non-positive dimensions raises ValueError."""
        with pytest.raises(
            ValueError, match="width_deg and height_deg must be positive"
        ):
            SkyRegion.box(ra_deg=0.0, dec_deg=0.0, width_deg=0.0, height_deg=10.0)
        with pytest.raises(
            ValueError, match="width_deg and height_deg must be positive"
        ):
            SkyRegion.box(ra_deg=0.0, dec_deg=0.0, width_deg=10.0, height_deg=-1.0)


# ---------------------------------------------------------------------------
# Union
# ---------------------------------------------------------------------------


class TestUnion:
    def test_union_basic(self):
        """Union of two non-overlapping cones contains points in either cone."""
        c1 = SkyRegion.cone(ra_deg=0.0, dec_deg=0.0, radius_deg=5.0)
        c2 = SkyRegion.cone(ra_deg=180.0, dec_deg=0.0, radius_deg=5.0)
        union = SkyRegion.union([c1, c2])

        ra_rad = np.deg2rad(np.array([0.0, 180.0, 90.0]))
        dec_rad = np.deg2rad(np.array([0.0, 0.0, 0.0]))
        mask = union.contains(ra_rad, dec_rad)

        assert mask[0] is np.True_  # in first cone
        assert mask[1] is np.True_  # in second cone
        assert mask[2] is np.False_  # in neither

    def test_union_flattening(self):
        """Union of unions is flattened — no nested union objects."""
        c1 = SkyRegion.cone(ra_deg=0.0, dec_deg=0.0, radius_deg=5.0)
        c2 = SkyRegion.cone(ra_deg=90.0, dec_deg=0.0, radius_deg=5.0)
        c3 = SkyRegion.cone(ra_deg=180.0, dec_deg=0.0, radius_deg=5.0)

        inner_union = SkyRegion.union([c1, c2])
        outer_union = SkyRegion.union([inner_union, c3])

        # All sub-regions should be atomic (cone), not union
        from rrivis.core.sky.region import UnionRegion

        assert isinstance(outer_union, UnionRegion)
        for sub in outer_union._sub_regions:
            assert not isinstance(sub, UnionRegion)
        assert len(outer_union._sub_regions) == 3

    def test_union_single(self):
        """Union of a single region returns that region directly."""
        from rrivis.core.sky.region import ConeRegion

        c1 = SkyRegion.cone(ra_deg=0.0, dec_deg=0.0, radius_deg=5.0)
        result = SkyRegion.union([c1])
        assert isinstance(result, ConeRegion)
        assert result is c1


# ---------------------------------------------------------------------------
# HEALPix mask
# ---------------------------------------------------------------------------


class TestHealpixMask:
    def test_healpix_mask_cone(self):
        """Cone healpix_mask returns a valid boolean mask with partial coverage."""
        nside = 32
        npix = hp.nside2npix(nside)
        region = SkyRegion.cone(ra_deg=180.0, dec_deg=0.0, radius_deg=20.0)
        mask = region.healpix_mask(nside)

        assert mask.dtype == bool
        assert mask.shape == (npix,)
        assert 0 < np.sum(mask) < npix

    def test_healpix_mask_box(self):
        """Box healpix_mask returns a valid boolean mask with partial coverage."""
        nside = 32
        npix = hp.nside2npix(nside)
        region = SkyRegion.box(
            ra_deg=180.0, dec_deg=0.0, width_deg=40.0, height_deg=20.0
        )
        mask = region.healpix_mask(nside)

        assert mask.dtype == bool
        assert mask.shape == (npix,)
        assert 0 < np.sum(mask) < npix


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_cone(self):
        """Cone repr contains expected substrings."""
        r = SkyRegion.cone(ra_deg=83.6, dec_deg=22.0, radius_deg=5.0)
        s = repr(r)
        assert "SkyRegion.cone" in s
        assert "ra=" in s
        assert "dec=" in s
        assert "radius=" in s

    def test_repr_box(self):
        """Box repr contains expected substrings."""
        r = SkyRegion.box(ra_deg=180.0, dec_deg=-30.0, width_deg=20.0, height_deg=10.0)
        s = repr(r)
        assert "SkyRegion.box" in s
        assert "width=" in s
        assert "height=" in s

    def test_repr_union(self):
        """Union repr contains expected substrings."""
        c1 = SkyRegion.cone(ra_deg=0.0, dec_deg=0.0, radius_deg=5.0)
        c2 = SkyRegion.cone(ra_deg=90.0, dec_deg=0.0, radius_deg=5.0)
        u = SkyRegion.union([c1, c2])
        s = repr(u)
        assert "SkyRegion.union" in s
        assert "2 sub-regions" in s
