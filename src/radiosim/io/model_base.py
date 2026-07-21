"""Lightweight shared base for strict user-input configuration models."""

from pydantic import BaseModel, ConfigDict


class StrictFrozenModel(BaseModel):
    """Shared base for every concrete user-input model."""

    model_config = ConfigDict(extra="forbid", frozen=True)


__all__ = ["StrictFrozenModel"]
