"""Sky-model combination subpackage.

Public entry points:

- :func:`prepare_sky_model` — the canonical user-facing combine/materialize
  orchestrator (see :mod:`.pipeline`).
- :class:`PrepareSkyOptions` — the validated options bundle for it
  (including ``assume_disjoint`` and ``mixed_model_policy`` disjointness
  controls).

The lower-level engine (``_combine_models``) and the per-strategy building
blocks (``combine_healpix``, ``concat_point_sources``, ``merge_provenance``,
``regrid_healpix_model``) are exposed here for tests and advanced internal
callers; they are not part of the package's top-level public surface.
"""

from __future__ import annotations

from .concat import concat_point_sources
from .disjointness import (
    MixedModelPolicy,
    check_physical_disjointness,
    classify_model,
    resolve_brightness_conversion,
    resolve_combination_params,
)
from .engine import (
    CombineHealpixData,
    _combine_models,
    resolve_target_representation,
)
from .healpix import combine_healpix
from .merge import merge_provenance
from .options import PrepareSkyOptions
from .pipeline import prepare_sky_model
from .regrid import regrid_healpix_model

__all__ = [
    "CombineHealpixData",
    "MixedModelPolicy",
    "PrepareSkyOptions",
    "_combine_models",
    "check_physical_disjointness",
    "classify_model",
    "combine_healpix",
    "concat_point_sources",
    "merge_provenance",
    "prepare_sky_model",
    "regrid_healpix_model",
    "resolve_brightness_conversion",
    "resolve_combination_params",
    "resolve_target_representation",
]
