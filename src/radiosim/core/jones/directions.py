"""The batch of sky directions every Jones term is evaluated over.

``Tier7JonesSciencePlan.md`` Section 13.2 replaces the scalar
``compute_jones(source_idx: int)`` contract with a direction-batched one, and
:class:`DirectionBatch` is the batch: one immutable, host-side, ``float64``
description of every direction a solver has resolved for one
``(time, frequency)`` step.

Why the batch carries two frames
--------------------------------
``(alt, az)`` alone is not sufficient.  The parallactic-angle term ``P`` is
defined in the equatorial frame, so ``ra_rad``, ``dec_rad`` and
``hour_angle_rad`` are first-class fields rather than something each term
re-derives, differently, from whatever it happens to have.

They are the **apparent** equatorial description of the very same directions,
derived here from the horizontal one with the site latitude and the local
apparent sidereal time, via the exact inverse of the transform that produced
``(alt, az)``.  Three consequences, all deliberate:

* The two halves cannot disagree.  Pairing a catalogue ICRS position with an
  apparent sidereal time would be inconsistent at the equinox-of-date level --
  of order ``1e-2`` radians in 2025 -- and a field rotation computed from the
  mismatched pair would inherit that error.
* The quadrant survives, because the hour angle comes from a two-argument
  arctangent of the same two components the forward transform used, not from an
  arcsine that folds two solutions together.
* The batch works for a sky in any frame.  A HEALPix map may be stored in
  galactic coordinates and has no right ascension to read; its directions still
  have an apparent equatorial description, and this is it.

``ra_rad`` is therefore an *apparent* right ascension of date, not a catalogue
ICRS one.  A term that needs the catalogue position of a source needs the sky
model, not the direction batch.

Why the arrays are host arrays
------------------------------
Section 13.3 fixes the host boundary: astropy coordinate work is host-side by
design and its outputs cross to the backend exactly once per time step.
``DirectionBatch`` sits on the host side of that boundary.  A term that needs a
device array asks the backend for one; a term that needs a host-side scalar
quantity -- an elevation for a mapping function, say -- reads it here instead of
copying back from the device, which is what makes
``Tier7JonesSciencePlan.md`` Section 17.2's "no host-side branch on a traced
array" rule satisfiable.

Field naming
------------
The plan's sketch names the direction cosines ``l``, ``m``, ``n``.  ``l`` is an
ambiguous identifier that the repository's own lint configuration rejects
(ruff ``E741``), and ``n`` would collide with ``n_dir``, so the fields carry the
solvers' existing spelling -- ``dir_l``, ``dir_m``, ``dir_n`` -- which is also
what ``visibility_healpix`` already calls them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "DirectionBatch",
    "DirectionBatchError",
    "equatorial_from_horizontal",
    "hour_angle_from_lst",
]


class DirectionBatchError(ValueError):
    """A direction batch was constructed from inconsistent inputs."""


#: The eight per-direction arrays, in declaration order.
_ARRAY_FIELDS = (
    "alt_rad",
    "az_rad",
    "dir_l",
    "dir_m",
    "dir_n",
    "ra_rad",
    "dec_rad",
    "hour_angle_rad",
)


def _read_only_float64(name: str, values: Any) -> npt.NDArray[np.float64]:
    """Return one owned, C-contiguous, finite, read-only ``float64`` array."""
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.ndim != 1:
        raise DirectionBatchError(
            f"DirectionBatch.{name} must be a one-dimensional array, got "
            f"{array.ndim} dimensions"
        )
    if not np.all(np.isfinite(array)):
        raise DirectionBatchError(f"DirectionBatch.{name} must be finite")
    array.setflags(write=False)
    return array


def _wrap_to_pi(values: Any) -> npt.NDArray[np.float64]:
    """Wrap angles to ``[-pi, pi)``."""
    array = np.asarray(values, dtype=np.float64)
    return np.mod(array + np.pi, 2.0 * np.pi) - np.pi


def hour_angle_from_lst(
    local_sidereal_time_rad: float,
    ra_rad: Any,
) -> npt.NDArray[np.float64]:
    """Return the local apparent hour angle, wrapped to ``[-pi, pi)``.

    ``H = LST - RA``, with both angles in the same (apparent) frame.  The wrap is
    explicit so that every consumer sees the same branch: a term that needs
    ``cos H`` or ``sin H`` is indifferent, but one that compares hour angles is
    not.
    """
    return _wrap_to_pi(float(local_sidereal_time_rad) - np.asarray(ra_rad))


def equatorial_from_horizontal(
    *,
    alt_rad: Any,
    az_rad: Any,
    latitude_rad: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return ``(hour_angle_rad, dec_rad)`` for topocentric ``(alt, az)``.

    The exact inverse of the standard horizontal transform, with azimuth measured
    North through East::

        sin(dec)          =  sin(lat) sin(alt) + cos(lat) cos(alt) cos(az)
        cos(dec) sin(H)   = -cos(alt) sin(az)
        cos(dec) cos(H)   =  cos(lat) sin(alt) - sin(lat) cos(alt) cos(az)

    The hour angle comes from the two-argument arctangent of the last two, so it
    keeps the quadrant that an arcsine of either one alone would lose.  At a
    celestial pole both components vanish and the hour angle is undefined; the
    arctangent returns ``0`` there, which is the conventional choice and is
    harmless because every direction-dependent use of ``H`` is multiplied by
    ``cos(dec)``.
    """
    altitude = np.asarray(alt_rad, dtype=np.float64)
    azimuth = np.asarray(az_rad, dtype=np.float64)
    sin_latitude = math.sin(float(latitude_rad))
    cos_latitude = math.cos(float(latitude_rad))

    sin_altitude = np.sin(altitude)
    cos_altitude = np.cos(altitude)
    cos_azimuth = np.cos(azimuth)

    sin_declination = sin_latitude * sin_altitude + cos_latitude * (
        cos_altitude * cos_azimuth
    )
    declination = np.arcsin(np.clip(sin_declination, -1.0, 1.0))
    hour_angle = np.arctan2(
        -cos_altitude * np.sin(azimuth),
        cos_latitude * sin_altitude - sin_latitude * cos_altitude * cos_azimuth,
    )
    return hour_angle, declination


@dataclass(frozen=True, eq=False)
class DirectionBatch:
    """One immutable batch of sky directions for one ``(time, frequency)`` step.

    Parameters
    ----------
    alt_rad, az_rad : ndarray
        Topocentric altitude and azimuth in radians, azimuth measured North
        through East -- the astropy ``AltAz`` convention the solvers already use.
    dir_l, dir_m, dir_n : ndarray
        Direction cosines in the local ENU frame: ``l`` East, ``m`` North,
        ``n`` Up.
    ra_rad, dec_rad : ndarray
        **Apparent** right ascension and declination of date, in radians -- the
        equatorial description of the same directions, not a catalogue position.
    hour_angle_rad : ndarray
        Local apparent hour angle in radians, wrapped to ``[-pi, pi)``.
    n_dir : int
        The number of directions.  Passed explicitly and checked against every
        array, so a batch assembled from mismatched masks fails at construction
        rather than inside a term.

    Notes
    -----
    Every array is copied, promoted to ``float64``, checked for finiteness, and
    made read-only, so a term cannot mutate the batch its neighbours will see.
    Equality is identity (``eq=False``): element-wise array comparison has no
    single truth value, and a generated ``__eq__`` would raise rather than
    answer.
    """

    alt_rad: npt.NDArray[np.float64]
    az_rad: npt.NDArray[np.float64]
    dir_l: npt.NDArray[np.float64]
    dir_m: npt.NDArray[np.float64]
    dir_n: npt.NDArray[np.float64]
    ra_rad: npt.NDArray[np.float64]
    dec_rad: npt.NDArray[np.float64]
    hour_angle_rad: npt.NDArray[np.float64]
    n_dir: int

    def __post_init__(self) -> None:
        if type(self.n_dir) is not int:
            raise DirectionBatchError("DirectionBatch.n_dir must be an exact int")
        if self.n_dir < 0:
            raise DirectionBatchError("DirectionBatch.n_dir must be non-negative")
        for name in _ARRAY_FIELDS:
            array = _read_only_float64(name, getattr(self, name))
            if array.size != self.n_dir:
                raise DirectionBatchError(
                    f"DirectionBatch.{name} has {array.size} entries but n_dir is "
                    f"{self.n_dir}; every direction array must have exactly n_dir "
                    "entries"
                )
            object.__setattr__(self, name, array)

    @classmethod
    def from_horizontal(
        cls,
        *,
        alt_rad: Any,
        az_rad: Any,
        dir_l: Any,
        dir_m: Any,
        dir_n: Any,
        latitude_rad: float,
        local_sidereal_time_rad: float,
    ) -> DirectionBatch:
        """Build a batch from a solver's host-preprocessing outputs.

        The direction cosines are passed in rather than recomputed here: each
        solver owns a named ``_host_direction_cosines`` stage
        (``Tier6HybridRuntimePlan.md`` Section 13.3) and this constructor must
        not become a second, silently divergent copy of that trigonometry.

        The equatorial half *is* computed here, from ``(alt, az)`` with the site
        latitude and the local apparent sidereal time, because there must be
        exactly one definition of it and neither solver had one before.
        """
        altitude = np.asarray(alt_rad, dtype=np.float64)
        hour_angle, declination = equatorial_from_horizontal(
            alt_rad=altitude,
            az_rad=az_rad,
            latitude_rad=latitude_rad,
        )
        return cls(
            alt_rad=altitude,
            az_rad=az_rad,
            dir_l=dir_l,
            dir_m=dir_m,
            dir_n=dir_n,
            ra_rad=np.mod(float(local_sidereal_time_rad) - hour_angle, 2.0 * np.pi),
            dec_rad=declination,
            hour_angle_rad=hour_angle,
            n_dir=int(altitude.size),
        )

    def __len__(self) -> int:
        return self.n_dir

    def __repr__(self) -> str:
        return f"DirectionBatch(n_dir={self.n_dir})"

    @property
    def field_names(self) -> tuple[str, ...]:
        """The declared field names, in order -- used by the contract tests."""
        return tuple(field.name for field in fields(self))
