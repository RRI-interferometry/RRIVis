"""Pydantic options object for :func:`prepare_sky_model`.

Bundles every kwarg with cross-field validation in one place so callers
can build a config once, validate it at construction, and reuse it for
multiple sky-prep runs (or serialise it for run-reproducibility).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import ConfigDict, field_validator, model_validator
from pydantic.dataclasses import dataclass

from radiosim.backends import ArrayBackend
from radiosim.core.precision import PrecisionConfig

from ..containers.constants import SYNCHROTRON_SPECTRAL_INDEX, BrightnessConversion
from ..containers.model import SkyFormat
from ..support.frequencies import validate_observation_frequencies
from .disjointness import MixedModelPolicy

_OPTIONS_CONFIG = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


@dataclass(frozen=True, config=_OPTIONS_CONFIG)
class PrepareSkyOptions:
    """All knobs accepted by :func:`prepare_sky_model`.

    Construct once, validate immediately, pass to :func:`prepare_sky_model`
    via the ``options=`` kwarg. Individual fields can still be overridden
    per call by passing them as keyword arguments — those override the
    options-object values.

    Cross-field rules enforced at construction:

    - ``frequencies`` is copied and validated as an explicit ordered Hz axis.
    - ``frequency`` (single-channel mode), when set, must be strictly
      positive.
    - ``mixed_model_policy`` is one of ``"error"``, ``"warn"``, ``"allow"``.
    - ``assume_disjoint`` defaults to ``False``; when ``True``, only
      double-count rules are skipped (monopole checks still run).
    """

    representation: SkyFormat | str | None = None
    nside: int | None = None
    frequencies: Sequence[float] | np.ndarray | None = None
    frequency: float | None = None
    allow_lossy: bool = False
    mixed_model_policy: MixedModelPolicy = "error"
    # When True, skip point-vs-diffuse double-counting rules in the
    # physical-disjointness check while still enforcing monopole consistency.
    # Emits a UserWarning; narrower than mixed_model_policy="allow".
    assume_disjoint: bool = False
    brightness_conversion: BrightnessConversion | str | None = None
    # ``PrecisionConfig`` / ``ArrayBackend`` are concrete here: neither
    # ``radiosim.core.precision`` nor ``radiosim.backends`` imports the sky
    # package at module load, so importing them eagerly does not create a
    # cycle, and pydantic (with ``arbitrary_types_allowed=True``) resolves the
    # annotations to build the validation schema.
    # Resolved early in :func:`prepare_sky_model` via
    # :func:`~radiosim.core.sky.support.precision.resolve_combine_precision`
    # (explicit value, else first input model, else error).
    precision: PrecisionConfig | None = None
    backend: ArrayBackend | None = None
    memmap_path: str | None = None
    # Power-law spectral index used to scale a diffuse model's
    # source-subtraction threshold to a point catalog's completeness frequency
    # in the physical-disjointness check. Default −0.7 (synchrotron).
    subtraction_scaling_alpha: float = SYNCHROTRON_SPECTRAL_INDEX

    @field_validator("frequencies", mode="before")
    @classmethod
    def validate_frequencies(cls, value: Any) -> np.ndarray | None:
        if value is None:
            return None
        frequencies = validate_observation_frequencies(
            value,
            label="PrepareSkyOptions frequencies",
        )
        frequencies.setflags(write=False)
        return frequencies

    @model_validator(mode="after")
    def _validate_state(self) -> PrepareSkyOptions:
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
