"""Tests for rrivis.io.config sky-source parsing."""

from __future__ import annotations

import pytest

from rrivis.io.config import (
    GleamSourceConfig,
    RRIvisConfig,
    TestSourcesConfig,
    parse_sky_source_config,
)


def test_from_yaml_requires_tagged_source_shape(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
sky_model:
  sources:
    - kind: gleam
      flux_limit: 1.5
""".lstrip()
    )

    config = RRIvisConfig.from_yaml(config_path)
    assert len(config.sky_model.sources) == 1

    source = config.sky_model.sources[0]
    assert isinstance(source, GleamSourceConfig)
    assert source.kind == "gleam"
    assert source.catalog == "gleam_egc"

    kind, kwargs = source.to_loader_request()
    assert kind == "gleam"
    assert kwargs["catalog"] == "gleam_egc"


def test_parse_sky_source_config_accepts_tagged_kind_shape():
    source = parse_sky_source_config(
        {"kind": "test_sources", "representation": "healpix_map"}
    )

    assert isinstance(source, TestSourcesConfig)
    assert source.kind == "test_sources"
    assert source.representation == "healpix_map"
    assert source.nside == 64


def test_nested_loader_key_shape_is_rejected():
    with pytest.raises(Exception, match="kind"):
        parse_sky_source_config({"gleam": {"flux_limit": 1.0}})


def test_source_specific_region_and_brightness_override_global_context():
    source = parse_sky_source_config(
        {
            "kind": "gleam",
            "brightness_conversion": "rayleigh-jeans",
            "region": {
                "shape": "cone",
                "center_ra_deg": 180.0,
                "center_dec_deg": 0.0,
                "radius_deg": 5.0,
            },
        }
    )

    kind, kwargs = source.to_loader_request(
        region="global_region",
        brightness_conversion="planck",
    )

    assert kind == "gleam"
    assert kwargs["brightness_conversion"] == "rayleigh-jeans"
    assert kwargs["region"] != "global_region"


def test_pyradiosky_file_source_exposes_spectral_loss_policy():
    source = parse_sky_source_config(
        {
            "kind": "pyradiosky_file",
            "filename": "example.skyh5",
            "spectral_loss_policy": "error",
        }
    )

    kind, kwargs = source.to_loader_request()

    assert kind == "pyradiosky_file"
    assert kwargs["spectral_loss_policy"] == "error"
