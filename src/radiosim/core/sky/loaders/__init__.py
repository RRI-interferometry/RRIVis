"""Typed public sky-loader functions."""

from __future__ import annotations

from .bbs import load_bbs
from .diffuse import load_diffuse_sky, load_pysm3
from .fits import load_fits_image
from .pyradiosky import load_pyradiosky_file
from .synthetic import load_test_sources
from .vizier import (
    load_3c,
    load_gleam,
    load_lotss,
    load_mals,
    load_nvss,
    load_racs,
    load_sumss,
    load_tgss,
    load_vlass,
    load_vlssr,
    load_wenss,
)

__all__ = [
    "load_test_sources",
    "load_diffuse_sky",
    "load_pysm3",
    "load_fits_image",
    "load_pyradiosky_file",
    "load_bbs",
    "load_gleam",
    "load_mals",
    "load_lotss",
    "load_vlssr",
    "load_tgss",
    "load_wenss",
    "load_sumss",
    "load_nvss",
    "load_3c",
    "load_vlass",
    "load_racs",
]
