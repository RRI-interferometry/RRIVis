# radiosim/core/sky/_protocols.py
"""Structural Protocol types for the sky module's data flow.

These Protocols describe the shape of objects that get passed between
``model``, ``operations``, ``combine``, and the ``_combine_*`` modules.
They are import-cycle-free: type-only consumers (``operations``, the
``_combine_*`` files) can annotate parameters and return types without
importing the concrete :class:`radiosim.core.sky.model.SkyModel` class.

Runtime construction of a ``SkyModel`` still requires the real class —
the dispatcher in :mod:`combine` and the factory functions in
:mod:`_factories` import it directly.

Currently provided:

- :class:`SkyModelLike` — minimum surface used by the combine pipeline
  (``point``, ``healpix``, ``provenance``, ``brightness_conversion``,
  ``model_name``, ``reference_frequency``, plus the ``replace`` method).

Adding new Protocols is preferred over adding new lazy imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..containers.constants import BrightnessConversion
    from ..containers.data import HealpixData, PointSourceData, SkyProvenance


@runtime_checkable
class SkyModelLike(Protocol):
    """Structural type for ``SkyModel`` instances passed across module
    boundaries.

    The combine and operations pipelines only need read access to the
    payload attributes plus the ``replace`` method (returns a
    ``SkyModelLike``). Adopting this protocol in type annotations lets a
    consumer module avoid importing the concrete ``SkyModel`` class —
    which in turn lets ``model.py`` evolve without forcing every
    downstream module to re-resolve its import graph.
    """

    point: PointSourceData | None
    healpix: HealpixData | None
    provenance: SkyProvenance
    brightness_conversion: BrightnessConversion
    model_name: str | None
    reference_frequency: float | None

    def replace(self, **changes: object) -> SkyModelLike: ...
