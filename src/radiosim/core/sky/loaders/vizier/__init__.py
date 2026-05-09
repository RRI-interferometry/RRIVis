"""VizieR + CASDA TAP loaders for radio point-source catalogs.

Public entry points: ``load_<catalog>`` wrappers for GLEAM, MALS,
LoTSS, NVSS, RACS, SUMSS, TGSS, VLASS, VLSSR, WENSS, and 3C.

Inspection helpers (``list_point_catalogs``, ``get_point_catalog_metadata``,
``get_catalog_columns``) and their RACS counterparts are also re-exported.

Internal modules:

- ``core``           — ``_load_from_vizier_catalog`` + column extractors.
- ``provenance``     — ``_build_point_catalog_provenance``.
- ``point_catalogs`` — the 10 simple/explicit VizieR loaders.
- ``racs``           — ``load_racs`` + CASDA TAP plumbing.
- ``inspect``        — ``list_*`` and ``get_*`` metadata helpers.
"""

from .inspect import (
    get_catalog_columns,
    get_point_catalog_metadata,
    get_racs_columns,
    get_racs_metadata,
    list_point_catalogs,
    list_racs_catalogs,
)
from .point_catalogs import (
    load_3c,
    load_gleam,
    load_lotss,
    load_mals,
    load_nvss,
    load_sumss,
    load_tgss,
    load_vlass,
    load_vlssr,
    load_wenss,
)
from .provenance import _build_point_catalog_provenance
from .racs import load_racs

__all__ = [
    "_build_point_catalog_provenance",
    "get_catalog_columns",
    "get_point_catalog_metadata",
    "get_racs_columns",
    "get_racs_metadata",
    "list_point_catalogs",
    "list_racs_catalogs",
    "load_3c",
    "load_gleam",
    "load_lotss",
    "load_mals",
    "load_nvss",
    "load_racs",
    "load_sumss",
    "load_tgss",
    "load_vlass",
    "load_vlssr",
    "load_wenss",
]
