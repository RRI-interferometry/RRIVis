"""The parallactic-angle term (P).

``P_p(s, t)`` is the rotation of an antenna's feeds relative to the sky frame.
For a direction with hour angle ``H`` and declination ``dec`` observed from
geodetic latitude ``lat``, the parallactic angle is

.. code-block:: text

    psi(H, dec, lat) = atan2( sin H cos lat,
                              sin lat cos dec - cos lat sin dec cos H )

and the Jones factor is the real rotation

.. code-block:: text

    P_p(s, t) = R( eta_p psi_p(s, t) + nasmyth_p el(s) )

    R(a) = [[ cos a,  sin a ],
            [-sin a,  cos a ]]

Define ``alpha_p = eta_p psi_p + nasmyth_p el``.  This is the same ``R`` the
accepted receptor mathematics uses (``docs/user_guide/jones_matrices.rst``), so
``C_p P_p`` composes into ``M(basis) R(chi_p + alpha_p)`` exactly.  Ordinary
alt-az has ``alpha_p=psi_p``; Nasmyth mounts retain the signed elevation term.

Why the two-argument arctangent
-------------------------------
The quadrant must survive over the whole sky.  An ``arcsin`` form folds the
second and third quadrants onto the first and fourth and is wrong for a source
below the pole, while passing every narrow-field test.

Why ``P`` is direction-dependent
--------------------------------
``psi`` is evaluated **per direction**, not once per field.  For a narrow field
it reduces to a constant rotation; for a wide one it varies measurably across
the primary beam.  That is the effect the two deleted wide-field rotation
classes gestured at, and a per-direction ``P`` subsumes it exactly rather than
approximately -- see ``docs/migration_guide.md`` for both names.

Mounts
------
``eta_p`` is the mount factor, resolved from each antenna's ``mount_type``
(``core/instrument.py``), and it is what makes a heterogeneous array correct:

====================  ======  =============================================
``mount_type``        eta     Meaning
====================  ======  =============================================
``alt-az``            ``+1``  full parallactic rotation
``equatorial``        ``0``   the feeds track the sky; no relative rotation
``fixed``             ``0``   the feeds are fixed to the ground
``alt-az+nasmyth-r``  ``+1``  Nasmyth right: ``psi + el``
``alt-az+nasmyth-l``  ``+1``  Nasmyth left: ``psi - el``
====================  ======  =============================================

An **unspecified** mount (``None``) is the ``fixed`` case.  Every instrument
source RadioSim reads except a pyuvdata dataset produces ``None`` -- a layout
file has no mount column and the known-telescope registry supplies none -- so
this is the choice invariant I1 rests on: the optional ``P`` contribution is
exactly the identity for those current sources.  Any value outside the five
listed mount types is rejected at resolution (R12) rather than silently treated
as ``alt-az``.

Chain position
--------------
Tier 7F moved ``P`` **sky-side** of ``C`` and ``E``
(``Tier7JonesSciencePlan.md`` Section 12, defect D12).  For a circular receptor
Tier 5 placed the rotation correlator-side, producing ``P C`` and applying a
real 2x2 rotation to the ``(R, L)`` output pair.  The corrected sky-side
placement produces ``C P``; since
``S R(alpha_p) = diag(e^{-i alpha_p}, e^{+i alpha_p}) S``, its circular effect
is a pair of opposite phases.  SCI-006 makes the native linear matrix ``P`` rather than
``I2``; the old ``P C`` placement is therefore wrong for both accepted native
bases.

One latitude, not one per antenna
---------------------------------
``psi`` is evaluated at the array's own geodetic latitude, which is the same
latitude ``DirectionBatch`` inverted the horizontal transform with.  Using a
per-antenna latitude derived from an ENU offset would pair a per-antenna
latitude with a batch-wide ``(H, dec)``, and ``directions.py`` states why the
two halves of the batch cannot be allowed to disagree.  What *is* per antenna is
the mount, which is where a heterogeneous array's physics actually lives.

Statelessness
-------------
``psi`` is recomputed on every call rather than memoized against the last
direction batch.  ``execute_time_blocks`` may evaluate two time steps
concurrently, so a cache keyed on the batch would be a data race with a wrong
answer rather than a slow one.

References
----------
Thompson, Moran & Swenson (2017), *Interferometry and Synthesis in Radio
Astronomy*, 3rd ed., Section 4.5 and Appendix 4.1.
Hamaker, Bregman & Sault (1996), A&AS 117, 137, Section 5.
Perley & Butler (2013), ApJS 206, 16 (polarization angle calibration).
Smirnov (2011), A&A 527, A106 (Paper I), Section 6.4.
The pyuvdata/CASA ``parangle`` convention.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import numpy.typing as npt

from radiosim.core.jones.base import JonesTerm
from radiosim.core.jones_errors import require_finite_jones_block

if TYPE_CHECKING:  # pragma: no cover - typing only
    from radiosim.core.jones.directions import DirectionBatch

__all__ = [
    "MOUNT_FACTORS",
    "ROTATING_MOUNT_TYPES",
    "SUPPORTED_MOUNT_TYPES",
    "ParallacticAngleJones",
    "mount_factors",
    "parallactic_angle",
]

#: ``mount_type`` -> ``(eta, nasmyth)``: the parallactic factor and the signed
#: elevation term.  Section 20.7's table, as data, so the five rows exist in
#: exactly one place.
MOUNT_FACTORS: Final[Mapping[str, tuple[float, float]]] = MappingProxyType(
    {
        "alt-az": (1.0, 0.0),
        "equatorial": (0.0, 0.0),
        "fixed": (0.0, 0.0),
        "alt-az+nasmyth-r": (1.0, 1.0),
        "alt-az+nasmyth-l": (1.0, -1.0),
    }
)

#: The five mount types ``P`` models, in the order rejection R12 names them.
SUPPORTED_MOUNT_TYPES: Final[tuple[str, ...]] = (
    "alt-az",
    "equatorial",
    "fixed",
    "alt-az+nasmyth-l",
    "alt-az+nasmyth-r",
)

#: The mounts whose feeds actually rotate relative to the sky.  R15 requires
#: ``jones.P`` for exactly these, and for no others: ``equatorial`` feeds track
#: the sky, so ``P`` is exactly ``I2`` for them and demanding the term would
#: collide with the R7 identity rejection.
ROTATING_MOUNT_TYPES: Final[frozenset[str]] = frozenset(
    {"alt-az", "alt-az+nasmyth-l", "alt-az+nasmyth-r"}
)

#: An unspecified mount is the ``fixed`` case; see the module docstring.
_UNSPECIFIED_MOUNT: Final[str] = "fixed"


def mount_factors(mount_type: str | None) -> tuple[float, float]:
    """Return ``(eta, nasmyth)`` for one resolved mount type.

    Parameters
    ----------
    mount_type : str or None
        A value from :data:`SUPPORTED_MOUNT_TYPES`, or ``None`` for an
        instrument source that carried no mount metadata.

    Returns
    -------
    tuple
        The parallactic factor and the signed elevation coefficient.

    Raises
    ------
    ValueError
        The mount type is outside the five ``P`` models.  Reaching this from a
        configuration is rejection R12, which is raised with the antenna number
        by :func:`~radiosim.core.jones_terms.resolve_jones_terms`; this is the
        constructor's own last line of defence.
    """
    if mount_type is None:
        return MOUNT_FACTORS[_UNSPECIFIED_MOUNT]
    try:
        return MOUNT_FACTORS[mount_type]
    except (KeyError, TypeError):
        raise ValueError(
            f"mount_type={mount_type!r} is not one of the five mount types the "
            f"parallactic-angle term models: {', '.join(SUPPORTED_MOUNT_TYPES)}."
        ) from None


def parallactic_angle(
    *,
    hour_angle_rad: Any,
    dec_rad: Any,
    latitude_rad: float,
) -> npt.NDArray[np.float64]:
    """Return the parallactic angle for each direction, in radians.

    The closed form of Section 20.7, using the two-argument arctangent so the
    quadrant is correct over the whole sky.  Public because it is the quantity
    the invariant tests compare against an independent oracle, and because a
    caller that wants the angle without the matrix should not have to invert one.

    Parameters
    ----------
    hour_angle_rad, dec_rad : array_like
        The apparent local hour angle and declination of each direction --
        ``DirectionBatch.hour_angle_rad`` and ``DirectionBatch.dec_rad``, which
        are derived together from the same horizontal pair and therefore cannot
        disagree.
    latitude_rad : float
        The site's geodetic latitude, the same one the batch was built with.

    Returns
    -------
    ndarray
        ``float64`` angles in ``[-pi, pi]``, measured North through East: the
        position angle of the zenith as seen from the direction.
    """
    hour_angle = np.asarray(hour_angle_rad, dtype=np.float64)
    dec = np.asarray(dec_rad, dtype=np.float64)
    sin_latitude = math.sin(float(latitude_rad))
    cos_latitude = math.cos(float(latitude_rad))
    return np.arctan2(
        np.sin(hour_angle) * cos_latitude,
        sin_latitude * np.cos(dec) - cos_latitude * np.sin(dec) * np.cos(hour_angle),
    )


class ParallacticAngleJones(JonesTerm):
    """Parallactic-angle rotation ``P`` (Section 20.7).

    Constructed only by
    :func:`~radiosim.core.jones_terms.resolve_jones_terms`, from a validated
    ``jones.P`` block plus the resolved instrument's latitude and mount types.
    The block carries no parameter beyond ``enabled``, because the parallactic
    angle is fully determined by the instrument, the time grid and the
    directions; inventing one to make ``P`` look like the other terms would be
    dishonest (Section 21.3).

    Parameters
    ----------
    latitude_rad : float
        The array's geodetic latitude in radians.
    mount_types : sequence of str or None
        One resolved ``mount_type`` per solver antenna row, in row order.
        ``None`` is the unspecified case and behaves as ``fixed``.

    Raises
    ------
    ValueError
        ``mount_types`` is empty, the latitude is not finite, or some entry is
        outside the five modelled mounts (rejection R12 one level up).
    """

    def __init__(
        self,
        *,
        latitude_rad: float,
        mount_types: Sequence[str | None],
    ) -> None:
        latitude = float(latitude_rad)
        if not math.isfinite(latitude):
            raise ValueError("ParallacticAngleJones latitude_rad must be finite")
        resolved = tuple(mount_types)
        if not resolved:
            raise ValueError(
                "ParallacticAngleJones mount_types must have one entry per antenna row"
            )
        factors = [mount_factors(mount) for mount in resolved]
        self._latitude_rad = latitude
        self._mount_types = resolved
        self._eta = np.array([eta for eta, _ in factors], dtype=np.float64)
        self._nasmyth = np.array([nasmyth for _, nasmyth in factors], dtype=np.float64)
        self._eta.setflags(write=False)
        self._nasmyth.setflags(write=False)

    @property
    def name(self) -> str:
        return "P"

    @property
    def term_status(self) -> str:
        """``"implemented"``: ``P`` carries the exact Section 20.7 mathematics."""
        return "implemented"

    @property
    def latitude_rad(self) -> float:
        """The geodetic latitude ``psi`` is evaluated at."""
        return self._latitude_rad

    @property
    def mount_types(self) -> tuple[str | None, ...]:
        """One resolved mount type per antenna row, in row order."""
        return self._mount_types

    @property
    def is_direction_dependent(self) -> bool:
        """``True``: ``psi`` varies across the field (invariant I9)."""
        return True

    @property
    def is_time_dependent(self) -> bool:
        """``True``: the hour angle is what makes the field rotate at all."""
        return True

    @property
    def is_frequency_dependent(self) -> bool:
        """``False``: a geometric rotation of the feeds is achromatic."""
        return False

    def is_diagonal(self) -> bool:
        """``True`` only when the resolved array makes ``P`` exactly ``I2``.

        A real rotation is diagonal only at angle zero.  Computed from the
        resolved mounts rather than hard-coded, because a hard-coded flag is a
        claim nothing checks -- the vacuous-flag failure mode invariant I2
        exists to prevent.  R7 makes the ``True`` case unreachable from a
        document.
        """
        return self.is_identity()

    def is_scalar(self) -> bool:
        """``True`` under the same condition that makes every ``P`` exactly ``I2``."""
        return self.is_identity()

    def is_unitary(self) -> bool:
        """``True`` always: a real rotation is orthogonal, so ``P P^H = I2``.

        This is a per-antenna norm-preservation statement.  When the same ``P``
        acts on both sides, an unpolarized coherency is unchanged; heterogeneous
        ``P_p`` and ``P_q`` need not leave an individual baseline matrix or its
        trace unchanged.
        """
        return True

    def is_identity(self) -> bool:
        """``True`` when no antenna's feeds rotate relative to the sky.

        That is the whole of an all-``fixed``, all-``equatorial``, or
        mount-unspecified array, and it is exactly R7's condition: the term
        cannot change the visibilities for any direction, time or frequency, so
        configuring it is a mistake rather than a no-op.
        """
        return bool(np.all(self._eta == 0.0) and np.all(self._nasmyth == 0.0))

    def field_angles(
        self,
        antenna_idx: int,
        directions: DirectionBatch,
    ) -> npt.NDArray[np.float64]:
        """Return one antenna's rotation angle for every direction, in radians.

        ``eta_p psi + nasmyth_p el``.  Public because it is the quantity the
        invariant tests compare against the transcribed closed form, and because
        it is what a diagnostic plot of a field rotation needs.
        """
        row = int(antenna_idx)
        if row < 0 or row >= self._eta.size:
            raise IndexError(
                f"ParallacticAngleJones has mounts for {self._eta.size} antenna "
                f"rows; row {row} is out of range."
            )
        eta = float(self._eta[row])
        nasmyth = float(self._nasmyth[row])
        angles = np.zeros(directions.n_dir, dtype=np.float64)
        if eta != 0.0:
            angles = angles + eta * parallactic_angle(
                hour_angle_rad=directions.hour_angle_rad,
                dec_rad=directions.dec_rad,
                latitude_rad=self._latitude_rad,
            )
        if nasmyth != 0.0:
            angles = angles + nasmyth * np.asarray(directions.alt_rad, dtype=np.float64)
        return angles

    def compute_jones_batch(
        self,
        *,
        antenna_idx: int,
        directions: DirectionBatch,
        frequency_hz: float,
        freq_idx: int,
        time_mjd: float,
        time_idx: int,
        backend: Any,
        dtype: Any,
    ) -> Any:
        """Return this antenna's ``(n_dir, 2, 2)`` rotations on the device.

        Direction-dependent, so the return is one matrix per direction and never
        a single broadcast one (invariant I3).  Frequency-independent, so
        ``frequency_hz`` and ``freq_idx`` are accepted and ignored; the time
        enters through the direction batch's hour angles rather than through
        ``time_mjd``, because the batch is what the solver already resolved at
        this step and re-deriving the geometry from the clock would be a second,
        silently divergent copy of it.

        The trigonometry runs on the host over the batch's own ``float64``
        arrays -- no array value is branched on and no device value is read back
        (Section 17.2).
        """
        angles = self.field_angles(antenna_idx, directions)
        cosine = np.cos(angles)
        sine = np.sin(angles)
        block = np.empty((angles.size, 2, 2), dtype=np.complex128)
        block[:, 0, 0] = cosine
        block[:, 0, 1] = sine
        block[:, 1, 0] = -sine
        block[:, 1, 1] = cosine
        require_finite_jones_block(self.name, block)
        return backend.xp.array(block, dtype=dtype)

    def get_config(self) -> dict[str, Any]:
        """Include the resolved latitude and mounts in the term's record."""
        config = super().get_config()
        config["latitude_rad"] = self._latitude_rad
        config["mount_types"] = list(self._mount_types)
        return config
