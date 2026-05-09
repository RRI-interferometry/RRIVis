"""Pydantic options object for :func:`prepare_sky_model`.

Bundles every kwarg with cross-field validation in one place so callers
can build a config once, validate it at construction, and reuse it for
multiple sky-prep runs (or serialise it for run-reproducibility).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from ..containers.constants import BrightnessConversion
from ..containers.model import SkyFormat
from .disjointness import MixedModelPolicy

_OPTIONS_CONFIG = ConfigDict(arbitrary_types_allowed=True)


@dataclass(frozen=True, config=_OPTIONS_CONFIG)
class PrepareSkyOptions:
    """All knobs accepted by :func:`prepare_sky_model`.

    Construct once, validate immediately, pass to :func:`prepare_sky_model`
    via the ``options=`` kwarg. Individual fields can still be overridden
    per call by passing them as keyword arguments — those override the
    options-object values.

    Cross-field rules enforced at construction:

    - ``frequencies`` and ``obs_frequency_config`` are mutually exclusive.
    - ``frequency`` (single-channel mode), when set, must be strictly
      positive.
    - ``nside_safety_factor`` must be strictly positive.
    - ``mixed_model_policy`` is one of ``"error"``, ``"warn"``, ``"allow"``.
    """

    representation: SkyFormat | str | None = None
    nside: int | None = None
    frequencies: np.ndarray | None = None
    frequency: float | None = None
    obs_frequency_config: dict[str, Any] | None = None
    allow_lossy: bool = False
    mixed_model_policy: MixedModelPolicy = "error"
    brightness_conversion: BrightnessConversion | str | None = None
    precision: Any = None
    memmap_path: str | None = None
    beam_fwhm_rad: float | None = None
    nside_safety_factor: float = 5.0

    @model_validator(mode="after")
    def _validate_state(self) -> PrepareSkyOptions:
        if self.frequencies is not None and self.obs_frequency_config is not None:
            raise ValueError(
                "PrepareSkyOptions: frequencies and obs_frequency_config are "
                "mutually exclusive — pass exactly one."
            )
        if self.nside_safety_factor <= 0:
            raise ValueError(
                "PrepareSkyOptions: nside_safety_factor must be strictly positive, "
                f"got {self.nside_safety_factor!r}."
            )
        if self.frequency is not None and self.frequency <= 0:
            raise ValueError(
                "PrepareSkyOptions: frequency must be strictly positive when set, "
                f"got {self.frequency!r}."
            )
        if self.mixed_model_policy not in ("error", "warn", "allow"):
            raise ValueError(
                "PrepareSkyOptions: mixed_model_policy must be 'error', 'warn', "
                f"or 'allow', got {self.mixed_model_policy!r}."
            )
        return self

    def merged(self, **overrides: Any) -> PrepareSkyOptions:
        """Return a new options object with ``overrides`` applied.

        Validators re-run on the resulting instance.
        """
        if not overrides:
            return self
        import dataclasses as _dc

        data: dict[str, Any] = {
            field.name: getattr(self, field.name) for field in _dc.fields(self)
        }
        unknown = set(overrides) - set(data)
        if unknown:
            raise TypeError(
                "PrepareSkyOptions.merged() received unsupported fields: "
                f"{sorted(unknown)}"
            )
        data.update(overrides)
        return PrepareSkyOptions(**data)
