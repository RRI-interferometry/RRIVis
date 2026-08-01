"""The ionospheric term (Z).

``Z_p(s, nu, t)`` carries both ionospheric effects in one factor: the
dispersive phase from the total electron content along the line of sight, and
the ionospheric Faraday rotation of the same column::

    Z_p(s, nu) = exp( i phi_TEC ) * F( psi_F )

    phi_TEC = -2 pi k_TEC * sTEC(s) / nu,   k_TEC = 40.308e16 / c Hz TECU^-1
    psi_F   = RM_ion * lambda^2

    F(a) = [[ cos a, -sin a ],
            [ sin a,  cos a ]]

The dispersive half is a **scalar** phase -- it multiplies the identity, so it
commutes and does not touch polarization -- and it scales exactly as
``1 / nu``.  The Faraday half is a real rotation and scales exactly as
``1 / nu^2`` through ``lambda^2``.  Both are direction-dependent, the first
through the slant mapping and the second not at all (see below).

The slant mapping
-----------------
``sTEC`` is the *slant* electron column toward the direction, obtained from the
configured vertical column by the standard thin-shell mapping at shell height
``h``::

    sTEC(s) = VTEC * 1 / cos( arcsin( R_E cos(el) / (R_E + h) ) )

with ``R_E = 6371 km``.  The factor is bounded -- about ``3.13`` at the horizon
for ``h = 350 km`` -- so ``Z`` does not diverge the way ``T``'s ``1/sin(el)``
does; what fails near the horizon is the thin-shell *approximation* itself,
which is why ``minimum_elevation_deg`` guards this term too (rejection R13,
adapted for the physics: see :meth:`IonosphereJones.reject_low_elevation`).

Two vertical-TEC models are offered.  ``constant`` is one number for the whole
array, which produces an antenna-common but direction-varying phase: a single
source at zenith then changes no visibility at all, while a wide field does.
``gradient`` adds a linear gradient in topocentric East and North evaluated **at
each antenna's own ionospheric pierce point**, which is the minimal model that
makes the phase differ between antennas and therefore the minimal model with a
closure-visible effect.

Why ``RM_ion`` is configured and not derived
--------------------------------------------
``RM_ion`` is given directly in rad m^-2, per array or per antenna.  It is
**not** derived from ``sTEC`` and a geomagnetic field model, because RadioSim
has no magnetic-field model and ``Tier7JonesSciencePlan.md`` Section 4 forbids
adding data ingestion for one.  A user who wants a physically derived value
should compute it with ``RMextract`` (Mevius 2018) and configure the number.
For the same reason the configured value is used as written rather than scaled
by the slant factor: it is *the* line-of-sight rotation measure, not a vertical
one waiting to be mapped, and invariant I8 pins the composition it takes part
in.

Why ``F`` is the transpose of the receptor rotation ``R``
---------------------------------------------------------
``R(a) = [[cos a, sin a], [-sin a, cos a]]`` -- the rotation ``C``, ``P`` and
the receptor mathematics use -- rotates the *frame*, so it *decreases* the
observed polarization angle by ``a``.  Faraday rotation rotates the *field*, so
it *increases* the observed angle by ``psi_F``, and the matrix that does that is
``R(a)^T = F(a)``.  The sign is not free: the sky model already rotates a
source's own ``(Q, U)`` by ``+RM_src (lambda^2 - lambda_ref^2)``
(``core/sky/containers/spectral.py``), and invariant I8 requires the two
rotations to *add* rather than to fight, which fixes ``F`` and not ``R`` here.

``Z`` owns *ionospheric* Faraday rotation only.  A source's intrinsic rotation
measure belongs to the sky model and is applied there, inside the solver's
frequency loop; the two live in different objects and different frames, they
compose, and they cannot be configured twice by accident
(``Tier7JonesSciencePlan.md`` Section 11, defect D18).  Stochastic turbulent
screens and IONEX/GPS ingestion are out of scope (Section 4).

References
----------
Thompson, Moran & Swenson (2017), *Interferometry and Synthesis in Radio
Astronomy*, 3rd ed., Section 13.3 and eq. 13.128 -- the dispersive phase and the
``40.308 TEC / nu^2`` excess path.
Intema et al. (2009), A&A 501, 1185 -- the thin-shell pierce-point model.
Mevius et al. (2016), Radio Sci. 51, 927 -- LOFAR ionospheric rotation measures.
Sotomayor-Beltran et al. (2013), A&A 552, A58 -- ``RMextract`` conventions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import numpy.typing as npt

from radiosim.core.jones.base import JonesTerm
from radiosim.core.jones_errors import (
    InvalidJonesConfigError,
    require_finite_jones_block,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from radiosim.core.jones.directions import DirectionBatch

__all__ = [
    "EARTH_RADIUS_M",
    "SPEED_OF_LIGHT_M_PER_S",
    "TEC_PHASE_CONSTANT_HZ_PER_TECU",
    "IonosphereJones",
    "ResolvedTecModel",
    "faraday_angle_rad",
    "pierce_point_offset_m",
    "slant_factor",
]

#: The speed of light in vacuum, in metres per second.
#:
#: Written out rather than imported from
#: ``radiosim.core.sky.containers.constants``, which holds the same exact value:
#: importing anything from ``radiosim.core.sky`` pulls that package's whole
#: ``__init__`` chain -- loaders, and through them a network client library --
#: into every import of ``radiosim.core.jones``, and
#: ``tests/unit/test_core/test_tier3_beam_cleanup.py`` pins the Jones package's
#: freedom from exactly that.  The duplication is safe because ``c`` is *defined*
#: to be this integer by the SI, not measured, and the two constants are asserted
#: equal in this module's tests so a divergence could not go unnoticed.
SPEED_OF_LIGHT_M_PER_S: Final[float] = 299_792_458.0

#: The mean Earth radius the thin-shell mapping is built on, in metres
#: (``Tier7JonesSciencePlan.md`` Section 20.8).
EARTH_RADIUS_M: Final[float] = 6_371_000.0

#: ``k_TEC``: the dispersive phase constant in Hz per TECU, so that
#: ``phi = -2 pi k_TEC sTEC / nu`` with ``sTEC`` in TECU.
#:
#: Derived rather than transcribed.  The ionospheric excess *path* is
#: ``40.308 TEC / nu^2`` metres with ``TEC`` in electrons m^-2 (TMS eq. 13.128),
#: one TECU is ``1e16`` electrons m^-2, and a path ``L`` is a phase
#: ``2 pi nu L / c`` -- so the constant is ``40.308e16 / c`` and equals the
#: ``1.3445e9`` the plan quotes to five significant figures.  Writing the
#: derivation instead of the number is what makes the ``1/nu`` scaling and the
#: sign checkable rather than asserted.
TEC_PHASE_CONSTANT_HZ_PER_TECU: Final[float] = 40.308e16 / SPEED_OF_LIGHT_M_PER_S


def slant_factor(alt_rad: Any, *, shell_height_m: float) -> npt.NDArray[np.float64]:
    """Return the thin-shell slant factor for each direction.

    ``1 / cos(arcsin(R_E cos(el) / (R_E + h)))``, evaluated as
    ``1 / sqrt(1 - x^2)`` because ``cos(arcsin x) = sqrt(1 - x^2)`` exactly and
    the round trip through two transcendental functions is both slower and less
    accurate.

    Parameters
    ----------
    alt_rad : array_like
        Direction altitudes in radians.  Only ``|el| <= pi/2`` is meaningful; the
        factor is symmetric in the sign of the altitude.
    shell_height_m : float
        The thin shell's height above the surface, in metres.

    Returns
    -------
    ndarray
        ``float64`` factors, ``1`` at zenith and about ``3.13`` at the horizon
        for the default 350 km shell.  Always finite: the shell is above the
        surface, so the argument of the square root is bounded away from zero.
    """
    altitude = np.asarray(alt_rad, dtype=np.float64)
    ratio = EARTH_RADIUS_M / (EARTH_RADIUS_M + float(shell_height_m))
    sine_shell_zenith = ratio * np.cos(altitude)
    return 1.0 / np.sqrt(1.0 - sine_shell_zenith**2)


def pierce_point_offset_m(
    alt_rad: Any,
    az_rad: Any,
    *,
    shell_height_m: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the ground-projected ``(east, north)`` offset of the pierce point.

    The line of sight punches the thin shell at an Earth-centred angle
    ``alpha = z - z_shell`` from the antenna, where ``z`` is the local zenith
    angle and ``sin(z_shell) = R_E cos(el) / (R_E + h)``; the ground projection
    of that arc is ``R_E alpha`` along the azimuth.  Azimuth is measured North
    through East, so the East component carries ``sin(az)`` and the North
    component ``cos(az)`` (Section 20.0).

    The offset is what makes the ``gradient`` TEC model produce a *differential*
    between antennas: two antennas looking at the same direction pierce the shell
    at points separated by their own baseline, and at low elevation by rather
    more than that.
    """
    altitude = np.asarray(alt_rad, dtype=np.float64)
    azimuth = np.asarray(az_rad, dtype=np.float64)
    ratio = EARTH_RADIUS_M / (EARTH_RADIUS_M + float(shell_height_m))
    shell_zenith = np.arcsin(np.clip(ratio * np.cos(altitude), -1.0, 1.0))
    earth_angle = (0.5 * np.pi - altitude) - shell_zenith
    distance = EARTH_RADIUS_M * earth_angle
    return distance * np.sin(azimuth), distance * np.cos(azimuth)


def faraday_angle_rad(rotation_measure_rad_m2: float, frequency_hz: float) -> float:
    """Return ``RM lambda^2`` in radians, the ionospheric rotation angle.

    Public because it is the closed form the invariant tests compare against and
    because it is the quantity a reader of a resolved ``Z`` actually wants: the
    ``lambda^2`` law is the whole observational signature of Faraday rotation.
    """
    wavelength_m = SPEED_OF_LIGHT_M_PER_S / float(frequency_hz)
    return float(rotation_measure_rad_m2) * wavelength_m * wavelength_m


@dataclass(frozen=True, slots=True)
class ResolvedTecModel:
    """The resolved vertical-TEC model of one ``jones.Z`` block.

    Parameters
    ----------
    vertical_tec_tecu
        The vertical column at the array reference position, in TECU.
        Non-negative: a negative electron column is rejected at resolution (R9).
    gradient_east_tecu_per_km, gradient_north_tecu_per_km
        The linear gradient of the vertical column with the pierce point's
        topocentric East and North offset, in TECU per kilometre.  Both zero is
        the ``constant`` model.

    Notes
    -----
    The gradient is a *local linear expansion*, and RadioSim does not clamp it:
    far enough from the reference position a steep gradient extrapolates the
    vertical column through zero, which is a column no ionosphere has.  The
    honest reading of such a configuration is that the model has been used
    outside its validity, and :meth:`vertical_tec_tecu` reports what was
    configured rather than silently flooring it -- a floor would make one
    document mean different things at different field widths.
    """

    vertical_tec_tecu: float
    gradient_east_tecu_per_km: float = 0.0
    gradient_north_tecu_per_km: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "vertical_tec_tecu",
            "gradient_east_tecu_per_km",
            "gradient_north_tecu_per_km",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"ResolvedTecModel {name} must be finite")
            object.__setattr__(self, name, value)

    @property
    def is_uniform(self) -> bool:
        """``True`` when no gradient is configured: one column for the array."""
        return (
            self.gradient_east_tecu_per_km == 0.0
            and self.gradient_north_tecu_per_km == 0.0
        )

    @property
    def is_zero(self) -> bool:
        """``True`` when the model is identically zero everywhere."""
        return self.vertical_tec_tecu == 0.0 and self.is_uniform

    def vertical_tec_tecu_at(
        self,
        east_m: Any,
        north_m: Any,
    ) -> npt.NDArray[np.float64]:
        """Return the vertical column at topocentric offsets, in TECU."""
        east_km = np.asarray(east_m, dtype=np.float64) / 1000.0
        north_km = np.asarray(north_m, dtype=np.float64) / 1000.0
        return (
            self.vertical_tec_tecu
            + self.gradient_east_tecu_per_km * east_km
            + self.gradient_north_tecu_per_km * north_km
        )


class IonosphereJones(JonesTerm):
    """Ionospheric dispersive phase and Faraday rotation ``Z`` (Section 20.8).

    Constructed only by
    :func:`~radiosim.core.jones_terms.resolve_jones_terms`, from a validated
    ``jones.Z`` block plus the resolved instrument's antenna positions.

    Parameters
    ----------
    tec_model : ResolvedTecModel
        The vertical-TEC model, already resolved and range-checked.
    antenna_positions_enu_m : ndarray
        ``(n_antenna_rows, 3)`` topocentric East, North, Up positions in metres,
        in solver antenna-row order.  Only the horizontal pair is used, and only
        by the ``gradient`` model -- but it is required either way, because a
        term that accepted the positions only sometimes would have two shapes.
    shell_height_m : float
        The thin shell's height above the surface, in metres.  Positive.
    rotation_measures_rad_m2 : ndarray
        ``(n_antenna_rows,)`` ionospheric rotation measure per antenna, in
        rad m^-2.  All zero disables the Faraday half.
    minimum_elevation_deg : float
        The elevation below which this run declines to trust the thin-shell
        mapping (R13).  ``0`` accepts every direction the horizon mask passes.

    Raises
    ------
    ValueError
        A shape mismatch, a non-positive shell height, a non-finite value, or a
        minimum elevation outside ``[0, 90)``.
    """

    def __init__(
        self,
        *,
        tec_model: ResolvedTecModel,
        antenna_positions_enu_m: npt.NDArray[np.float64],
        shell_height_m: float,
        rotation_measures_rad_m2: npt.NDArray[np.float64],
        minimum_elevation_deg: float,
    ) -> None:
        if type(tec_model) is not ResolvedTecModel:
            raise TypeError("IonosphereJones tec_model must be a ResolvedTecModel")
        positions = np.array(
            antenna_positions_enu_m, dtype=np.float64, copy=True, order="C"
        )
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 1:
            raise ValueError(
                "IonosphereJones antenna_positions_enu_m must have shape "
                f"(n_antenna_rows, 3), got {positions.shape}"
            )
        if not bool(np.isfinite(positions).all()):
            raise ValueError("IonosphereJones antenna_positions_enu_m must be finite")
        rotation_measures = np.array(
            rotation_measures_rad_m2, dtype=np.float64, copy=True, order="C"
        )
        if rotation_measures.shape != (positions.shape[0],):
            raise ValueError(
                "IonosphereJones rotation_measures_rad_m2 must have one entry per "
                f"antenna row, got {rotation_measures.shape} for "
                f"{positions.shape[0]} rows"
            )
        if not bool(np.isfinite(rotation_measures).all()):
            raise ValueError("IonosphereJones rotation_measures_rad_m2 must be finite")
        height = float(shell_height_m)
        if not math.isfinite(height) or height <= 0.0:
            raise ValueError("IonosphereJones shell_height_m must be positive")
        minimum_elevation = float(minimum_elevation_deg)
        if not math.isfinite(minimum_elevation) or not 0.0 <= minimum_elevation < 90.0:
            raise ValueError("IonosphereJones minimum_elevation_deg must be in [0, 90)")
        positions.setflags(write=False)
        rotation_measures.setflags(write=False)
        self._tec_model = tec_model
        self._positions_enu_m = positions
        self._shell_height_m = height
        self._rotation_measures_rad_m2 = rotation_measures
        self._minimum_elevation_deg = minimum_elevation

    # ------------------------------------------------------------------ shape

    @property
    def name(self) -> str:
        return "Z"

    @property
    def term_status(self) -> str:
        """``"implemented"``: ``Z`` carries the exact Section 20.8 mathematics."""
        return "implemented"

    @property
    def tec_model(self) -> ResolvedTecModel:
        """The resolved vertical-TEC model."""
        return self._tec_model

    @property
    def shell_height_m(self) -> float:
        """The thin shell's height above the surface, in metres."""
        return self._shell_height_m

    @property
    def rotation_measures_rad_m2(self) -> npt.NDArray[np.float64]:
        """One ionospheric rotation measure per antenna row, read-only."""
        return self._rotation_measures_rad_m2

    @property
    def minimum_elevation_deg(self) -> float:
        """The elevation floor below which R13 refuses to evaluate."""
        return self._minimum_elevation_deg

    @property
    def is_direction_dependent(self) -> bool:
        """``True``: the slant column varies with elevation across the field."""
        return True

    @property
    def is_time_dependent(self) -> bool:
        """``True``: a direction's elevation, and so its column, changes with time."""
        return True

    @property
    def is_frequency_dependent(self) -> bool:
        """``True``: ``1/nu`` in the phase and ``1/nu^2`` in the rotation.

        Computed from the resolved numbers rather than hard-coded, so the claim
        is false exactly when the term carries neither effect -- which R7 makes
        unreachable from a document.
        """
        return not self._tec_model.is_zero or self._has_faraday

    @property
    def _has_faraday(self) -> bool:
        return bool(np.any(self._rotation_measures_rad_m2 != 0.0))

    def is_diagonal(self) -> bool:
        """``True`` only without Faraday rotation, where ``Z`` is a scalar phase.

        A real rotation is diagonal only at angle zero, and the angle is zero at
        every frequency exactly when every rotation measure is.
        """
        return not self._has_faraday

    def is_scalar(self) -> bool:
        """``True`` under the same condition: ``exp(i phi) I2`` is scalar."""
        return not self._has_faraday

    def is_unitary(self) -> bool:
        """``True`` always: a scalar phase times a real rotation is unitary.

        This is the flag that is a genuine constant for ``Z``.  The ionosphere
        delays and rotates the field; it does not absorb it, which is the
        physical difference from ``T``'s opacity.
        """
        return True

    def is_identity(self) -> bool:
        """``True`` when the column and every rotation measure are exactly zero.

        R7's condition verbatim: with no electrons there is no phase and no
        rotation at any frequency, direction or time, so the term cannot change
        the visibilities and configuring it is a mistake rather than a no-op.
        """
        return self._tec_model.is_zero and not self._has_faraday

    # --------------------------------------------------------------- physics

    def reject_low_elevation(self, directions: DirectionBatch) -> None:
        """Raise R13 when a direction below the configured floor reaches ``Z``.

        The rejection is raised **here**, at evaluation, and not at resolution:
        the condition R13 names -- "a direction survives the horizon mask below
        ``minimum_elevation_deg``" -- is a statement about directions, and no
        direction exists until the solver has resolved one for a
        ``(time, frequency)`` step.  Section 26.1's stage 5 can and does check
        every part of the configuration that is decidable without a sky; this
        part is not one of them.

        The message is R13's, with its final clause adapted to what is true of
        ``Z``: the thin-shell factor is bounded at the horizon, so it does not
        *diverge* the way ``T``'s ``1/sin(el)`` does -- what fails below a few
        degrees is the thin-shell approximation itself.  This is the same bounded,
        named adaptation Section 24 already grants R5 for a term without feeds,
        and not a licence to reword a rejection in general.
        """
        if self._minimum_elevation_deg <= 0.0 or directions.n_dir == 0:
            return
        lowest_deg = math.degrees(float(np.min(directions.alt_rad)))
        if lowest_deg >= self._minimum_elevation_deg:
            return
        raise InvalidJonesConfigError(
            f"jones.Z.minimum_elevation_deg={self._minimum_elevation_deg} excludes "
            f"no direction, but the thin-shell mapping function is not valid below "
            f"{self._minimum_elevation_deg} deg; raise the minimum elevation or the "
            "horizon mask."
        )

    def slant_tec_tecu(
        self,
        antenna_idx: int,
        directions: DirectionBatch,
    ) -> npt.NDArray[np.float64]:
        """Return this antenna's slant electron column per direction, in TECU.

        Public because it is the quantity the invariant tests compare against the
        transcribed closed form, and because a diagnostic of an ionospheric
        screen wants the column rather than the matrix.
        """
        row = self._require_row(antenna_idx)
        altitude = np.asarray(directions.alt_rad, dtype=np.float64)
        if self._tec_model.is_uniform:
            vertical = np.full(
                altitude.shape, self._tec_model.vertical_tec_tecu, dtype=np.float64
            )
        else:
            east_offset, north_offset = pierce_point_offset_m(
                altitude,
                directions.az_rad,
                shell_height_m=self._shell_height_m,
            )
            vertical = self._tec_model.vertical_tec_tecu_at(
                east_offset + float(self._positions_enu_m[row, 0]),
                north_offset + float(self._positions_enu_m[row, 1]),
            )
        return vertical * slant_factor(altitude, shell_height_m=self._shell_height_m)

    def dispersive_phase_rad(
        self,
        antenna_idx: int,
        directions: DirectionBatch,
        frequency_hz: float,
    ) -> npt.NDArray[np.float64]:
        """Return ``-2 pi k_TEC sTEC / nu`` per direction, in radians.

        Negative for a positive column, matching the geometric phase's own
        ``exp(-2 pi i b.s)`` sign: a positive excess path is a negative phase on
        every RadioSim propagation term (invariant I4).
        """
        return (
            -2.0
            * math.pi
            * TEC_PHASE_CONSTANT_HZ_PER_TECU
            * self.slant_tec_tecu(antenna_idx, directions)
            / float(frequency_hz)
        )

    def faraday_angle_rad(self, antenna_idx: int, frequency_hz: float) -> float:
        """Return this antenna's ``RM lambda^2`` rotation angle, in radians."""
        row = self._require_row(antenna_idx)
        return faraday_angle_rad(
            float(self._rotation_measures_rad_m2[row]), frequency_hz
        )

    def _require_row(self, antenna_idx: int) -> int:
        row = int(antenna_idx)
        if row < 0 or row >= self._positions_enu_m.shape[0]:
            raise IndexError(
                f"IonosphereJones has {self._positions_enu_m.shape[0]} antenna rows; "
                f"row {row} is out of range."
            )
        return row

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
        """Return this antenna's ``(n_dir, 2, 2)`` ionospheric matrices.

        Direction-dependent, so the return is one matrix per direction
        (invariant I3).  The time enters through the direction batch's own
        altitudes rather than through ``time_mjd``: the batch is what the solver
        resolved at this step, and re-deriving the geometry from the clock would
        be a second, silently divergent copy of it.

        All trigonometry runs on the host over the batch's ``float64`` arrays; no
        array value is branched on and no device value is read back
        (Section 17.2).
        """
        self.reject_low_elevation(directions)
        phase = self.dispersive_phase_rad(antenna_idx, directions, frequency_hz)
        phasor = np.exp(1j * phase)
        angle = self.faraday_angle_rad(antenna_idx, frequency_hz)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        block = np.empty((phase.size, 2, 2), dtype=np.complex128)
        block[:, 0, 0] = phasor * cosine
        block[:, 0, 1] = -phasor * sine
        block[:, 1, 0] = phasor * sine
        block[:, 1, 1] = phasor * cosine
        require_finite_jones_block(self.name, block)
        return backend.xp.array(block, dtype=dtype)

    def get_config(self) -> dict[str, Any]:
        """Include the resolved screen parameters in the term's record."""
        config = super().get_config()
        config["vertical_tec_tecu"] = self._tec_model.vertical_tec_tecu
        config["gradient_east_tecu_per_km"] = self._tec_model.gradient_east_tecu_per_km
        config["gradient_north_tecu_per_km"] = (
            self._tec_model.gradient_north_tecu_per_km
        )
        config["shell_height_m"] = self._shell_height_m
        config["rotation_measures_rad_m2"] = [
            float(value) for value in self._rotation_measures_rad_m2
        ]
        config["minimum_elevation_deg"] = self._minimum_elevation_deg
        return config
