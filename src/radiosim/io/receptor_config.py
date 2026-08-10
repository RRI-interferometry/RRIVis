"""Strict, frozen receptor and polarization-basis input models.

These models own the *receiving receptor* vocabulary — ``basis``, ``feed``, and
``feed_rotation``.  Aperture *illumination* is a disjoint concept owned by the
beam subsystem under ``beams.model.illumination`` and must never be configured
here (``Tier5ReceptorFeedPlan.md`` Section 15).

Antenna tagging reuses the Tier 2 :data:`AntennaReference` discriminated union
rather than redefining antenna identity.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from radiosim.io.instrument_config import (
    AntennaNameReference,
    AntennaNumberReference,
    AntennaReference,
)
from radiosim.io.model_base import StrictFrozenModel

ReceptorBasis = Literal["linear", "circular"]
OutputBasisRequest = Literal["auto", "linear", "circular"]

_StrictFiniteFloat = Annotated[
    float,
    Field(strict=True, allow_inf_nan=False),
]


def _reference_identity(reference: AntennaReference) -> tuple[str, int | str]:
    """Return the normalized comparison key for one tagged antenna reference."""
    if type(reference) is AntennaNumberReference:
        return ("number", reference.number)
    if type(reference) is AntennaNameReference:
        return ("name", reference.name)
    raise ValueError("antenna must be an exact Tier 2 AntennaReference")


class ReceptorDefinitionConfig(StrictFrozenModel):
    """The receptor pair shared by every antenna without an override.

    Parameters
    ----------
    basis
        ``linear`` for an ``x``/``y`` pair or ``circular`` for an ``r``/``l``
        pair.  Both feeds of an antenna are ideal, orthogonal, and share the
        basis.
    feed_rotation_deg
        One rotation offset from the nominal orientation of the selected basis,
        in degrees, in the topocentric horizontal frame.  It is normalized into
        ``(-180, 180]`` at resolution.
    """

    basis: ReceptorBasis = "linear"
    feed_rotation_deg: _StrictFiniteFloat = 0.0


class ReceptorOverrideConfig(StrictFrozenModel):
    """One partial per-antenna receptor override.

    Parameters
    ----------
    antenna
        Exactly one tagged Tier 2 antenna reference.
    basis
        Optional replacement basis; ``None`` keeps the default.
    feed_rotation_deg
        Optional replacement rotation offset; ``None`` keeps the default.
    """

    antenna: AntennaReference
    basis: ReceptorBasis | None = None
    feed_rotation_deg: _StrictFiniteFloat | None = None


class ReceptorsConfig(StrictFrozenModel):
    """Frozen Tier 5 receptor input before any instrument resolution.

    Parameters
    ----------
    default
        The receptor definition applied to every antenna first.
    overrides
        Ordered, immutable partial per-antenna overrides.
    output_basis
        The requested common output basis for the whole array.  ``auto``
        follows a homogeneous array and rejects a mixed one.

    Notes
    -----
    The default instance resolves every antenna as linear with zero feed
    rotation and a ``linear_xy`` output basis.  Under the declared east-X
    convention its receptor matrix is the North/East-to-X/Y permutation, not
    the historical identity matrix.
    """

    default: ReceptorDefinitionConfig = Field(default_factory=ReceptorDefinitionConfig)
    overrides: tuple[ReceptorOverrideConfig, ...] = ()
    output_basis: OutputBasisRequest = "auto"

    @model_validator(mode="after")
    def require_override_content(self) -> Self:
        for index, override in enumerate(self.overrides):
            if override.basis is None and override.feed_rotation_deg is None:
                raise ValueError(
                    f"receptors.overrides[{index}] must set at least one of "
                    "'basis' or 'feed_rotation_deg'"
                )
        return self

    @model_validator(mode="after")
    def reject_repeated_override_references(self) -> Self:
        seen: dict[tuple[str, int | str], int] = {}
        for index, override in enumerate(self.overrides):
            identity = _reference_identity(override.antenna)
            previous = seen.get(identity)
            if previous is not None:
                raise ValueError(
                    f"receptors.overrides[{index}] repeats the antenna reference "
                    f"already used by receptors.overrides[{previous}]"
                )
            seen[identity] = index
        return self


__all__ = [
    "OutputBasisRequest",
    "ReceptorBasis",
    "ReceptorDefinitionConfig",
    "ReceptorOverrideConfig",
    "ReceptorsConfig",
]
