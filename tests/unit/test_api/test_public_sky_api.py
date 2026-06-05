"""Smoke tests for the public sky package export contract."""

from __future__ import annotations

import radiosim.core.sky as sky


def test_sky_all_names_resolve():
    for name in sky.__all__:
        assert hasattr(sky, name), f"missing public sky export: {name}"


def test_prepare_sky_model_is_public_combine_entrypoint():
    assert hasattr(sky, "prepare_sky_model")
    assert hasattr(sky, "PrepareSkyOptions")
    assert hasattr(sky, "regrid_healpix_model")
    assert not hasattr(sky, "combine_models")
    assert "combine_models" not in sky.__all__
