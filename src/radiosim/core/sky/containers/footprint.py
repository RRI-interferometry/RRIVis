"""Footprint, coverage, and provenance enums + SkyFootprint dataclass.

Leaf module within ``containers/`` — imports only numpy, healpy, pydantic,
and the package-internal :mod:`._shared` helpers.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import field_validator, model_validator
from pydantic.dataclasses import dataclass

from ..support.healpy import lazy_healpy as hp
from ._shared import _FROZEN_NDARRAY_CONFIG


class MonopoleConvention(str, Enum):
    """How a sky model represents the sky-average (DC) brightness temperature.

    The enum values are the canonical strings used for serialization and for
    cross-model compatibility checks during combination.
    """

    ABSOLUTE_WITH_CMB = "absolute_with_cmb"
    ABSOLUTE_NO_CMB = "absolute_no_cmb"
    MEAN_SUBTRACTED = "mean_subtracted"
    UNKNOWN = "unknown"


class SkyCoverage(str, Enum):
    """Whether a sky model represents the full sky or a subset of it."""

    FULL_SKY = "full_sky"
    PARTIAL_SKY = "partial_sky"
    UNKNOWN = "unknown"


class SourceSubtractionStatus(str, Enum):
    """Whether discrete sources have been removed from a sky model's payload."""

    NONE = "none"
    ABOVE_THRESHOLD = "above_threshold"
    ALL = "all"
    UNKNOWN = "unknown"


DEFAULT_COVERAGE_FOOTPRINT_NSIDE = 256
DEFAULT_COVERAGE_FOOTPRINT_COORDINATE_FRAME = "icrs"


def _normalize_coordinate_frame(coordinate_frame: str) -> str:
    """Lowercase a frame name and validate it is 'icrs' or 'galactic'."""
    frame = str(coordinate_frame).lower()
    if frame not in {"icrs", "galactic"}:
        raise ValueError(
            f"coordinate_frame must be 'icrs' or 'galactic', got {coordinate_frame!r}."
        )
    return frame


def _normalize_ordering(ordering: str) -> str:
    """Lowercase an ordering name and validate it is 'ring' or 'nest'."""
    scheme = str(ordering).lower()
    if scheme not in {"ring", "nest"}:
        raise ValueError(f"ordering must be 'ring' or 'nest', got {ordering!r}.")
    return scheme


@dataclass(frozen=True, eq=False, config=_FROZEN_NDARRAY_CONFIG)
class SkyFootprint:
    """Sparse HEALPix support mask for a sky product's angular footprint."""

    nside: int
    hpx_inds: np.ndarray
    coordinate_frame: str = DEFAULT_COVERAGE_FOOTPRINT_COORDINATE_FRAME

    @field_validator("nside", mode="before")
    @classmethod
    def _validate_nside(cls, value: object) -> int:
        n = int(value)  # type: ignore[arg-type]
        if not hp.isnsideok(n):
            raise ValueError(f"SkyFootprint.nside must be a valid NSIDE, got {n}.")
        return n

    @field_validator("coordinate_frame", mode="before")
    @classmethod
    def _validate_coordinate_frame(cls, value: object) -> str:
        return _normalize_coordinate_frame(str(value))

    @field_validator("hpx_inds", mode="before")
    @classmethod
    def _validate_hpx_inds_shape(cls, value: object) -> np.ndarray:
        arr = np.asarray(value, dtype=np.int64)
        if arr.ndim != 1:
            raise ValueError(
                "SkyFootprint.hpx_inds must be a 1-D integer array of pixel indices."
            )
        if arr.size:
            arr = np.unique(arr)
        return arr

    @model_validator(mode="after")
    def _validate_indices_in_range(self) -> SkyFootprint:
        if self.hpx_inds.size:
            full_n_pixels = hp.nside2npix(self.nside)
            if np.any(self.hpx_inds < 0) or np.any(self.hpx_inds >= full_n_pixels):
                raise ValueError(
                    "SkyFootprint.hpx_inds contains indices outside the valid "
                    f"range [0, {full_n_pixels})."
                )
        return self

    @classmethod
    def from_mask(
        cls,
        mask: np.ndarray,
        *,
        nside: int,
        coordinate_frame: str = DEFAULT_COVERAGE_FOOTPRINT_COORDINATE_FRAME,
    ) -> SkyFootprint:
        """Build a sparse footprint from a dense boolean HEALPix mask."""
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 1:
            raise ValueError("SkyFootprint.from_mask requires a 1-D boolean mask.")
        expected = hp.nside2npix(int(nside))
        if mask.size != expected:
            raise ValueError(
                "SkyFootprint.from_mask got a mask of length "
                f"{mask.size}, expected {expected} for nside={int(nside)}."
            )
        return cls(
            nside=int(nside),
            coordinate_frame=coordinate_frame,
            hpx_inds=np.flatnonzero(mask),
        )

    @property
    def full_n_pixels(self) -> int:
        """Total number of HEALPix pixels on the footprint grid."""
        return int(hp.nside2npix(self.nside))

    @property
    def coverage_fraction(self) -> float:
        """Fraction of the full sky covered by this footprint."""
        return float(self.hpx_inds.size / self.full_n_pixels)

    @property
    def is_full_sky(self) -> bool:
        """True when the footprint covers every pixel on its grid."""
        return self.hpx_inds.size == self.full_n_pixels

    def to_mask(self) -> np.ndarray:
        """Materialize the sparse footprint to a dense boolean HEALPix mask."""
        mask = np.zeros(self.full_n_pixels, dtype=bool)
        mask[self.hpx_inds] = True
        return mask

    def _require_compatible(self, other: SkyFootprint) -> None:
        if not isinstance(other, SkyFootprint):
            raise TypeError(
                "SkyFootprint operations require another SkyFootprint, got "
                f"{type(other).__name__}."
            )
        if self.nside != other.nside:
            raise ValueError(
                "SkyFootprint operations require matching nside values, got "
                f"{self.nside} and {other.nside}."
            )
        if self.coordinate_frame != other.coordinate_frame:
            raise ValueError(
                "SkyFootprint operations require matching coordinate frames, got "
                f"{self.coordinate_frame!r} and {other.coordinate_frame!r}."
            )

    def union(self, *others: SkyFootprint) -> SkyFootprint:
        """Return the geometric union with one or more compatible footprints."""
        if not others:
            return self
        parts = [self.hpx_inds]
        for other in others:
            self._require_compatible(other)
            parts.append(other.hpx_inds)
        return SkyFootprint(
            nside=self.nside,
            coordinate_frame=self.coordinate_frame,
            hpx_inds=np.unique(np.concatenate(parts)),
        )

    def intersect(self, other: SkyFootprint) -> SkyFootprint:
        """Return the geometric intersection with another compatible footprint."""
        self._require_compatible(other)
        return SkyFootprint(
            nside=self.nside,
            coordinate_frame=self.coordinate_frame,
            hpx_inds=np.intersect1d(
                self.hpx_inds,
                other.hpx_inds,
                assume_unique=True,
            ),
        )

    def intersect_mask(self, mask: np.ndarray) -> SkyFootprint:
        """Intersect the footprint with a dense boolean mask on the same grid."""
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 1 or mask.size != self.full_n_pixels:
            raise ValueError(
                "SkyFootprint.intersect_mask requires a 1-D boolean mask of length "
                f"{self.full_n_pixels}."
            )
        return SkyFootprint(
            nside=self.nside,
            coordinate_frame=self.coordinate_frame,
            hpx_inds=self.hpx_inds[mask[self.hpx_inds]],
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SkyFootprint):
            return NotImplemented
        return (
            self.nside == other.nside
            and self.coordinate_frame == other.coordinate_frame
            and np.array_equal(self.hpx_inds, other.hpx_inds)
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.nside,
                self.coordinate_frame,
                self.hpx_inds.tobytes(),
            )
        )
