"""Tests for the public angular-scale HEALPix recommendation utility."""

from __future__ import annotations

import importlib
import math

import healpy as hp
import numpy as np
import pytest


def _healpix_module():
    return importlib.import_module("radiosim.utils.healpix")


def _recommend(target_angular_scale_rad: object) -> int:
    return _healpix_module().recommend_nside_for_angular_scale(target_angular_scale_rad)


def test_exact_pixel_scale_boundary_is_accepted() -> None:
    target = float(hp.nside2resol(32))

    assert _recommend(target) == 32


@pytest.mark.parametrize(
    "target",
    (
        float(hp.nside2resol(1)),
        np.deg2rad(5.0),
        np.deg2rad(1.0),
        np.deg2rad(0.01),
        float(hp.nside2resol(65536)),
    ),
)
def test_recommendation_is_the_smallest_satisfying_power_of_two(
    target: float,
) -> None:
    nside = _recommend(target)

    assert type(nside) is int
    assert nside > 0
    assert nside & (nside - 1) == 0
    assert hp.nside2resol(nside) <= target
    if nside > 1:
        assert hp.nside2resol(nside // 2) > target


@pytest.mark.parametrize(
    "invalid",
    (
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        -float("inf"),
        True,
        False,
        None,
        "0.1",
        object(),
    ),
)
def test_invalid_angular_scale_is_rejected(invalid: object) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        _recommend(invalid)


def test_target_finer_than_retained_maximum_raises() -> None:
    target = math.nextafter(float(hp.nside2resol(65536)), 0.0)

    with pytest.raises(ValueError, match="65536"):
        _recommend(target)


def test_old_fwhm_advisor_names_are_removed() -> None:
    module = _healpix_module()

    assert not hasattr(module, "recommend_nside_for_beam")
    assert not hasattr(module, "pixel_too_coarse")
    assert "recommend_nside_for_beam" not in module.__all__
    assert "pixel_too_coarse" not in module.__all__
